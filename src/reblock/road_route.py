"""The planarized road graph a walker actually routes over.

`unary_union` planarization nodes every crossing/touch into a shared vertex -- the whole point,
since a walker can turn where two roads cross -- but it also DESTROYS the association between a
segment and the road row it came from, and with it that road's `width_m`. `build_roadnet` recovers
it by asking which original road corridors cover each segment's MIDPOINT and taking the WIDEST:
two overlapping roads occupy one corridor and the wider one governs, the same convention
`displacement` uses when it unions corridors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import shapely
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely import STRtree
from shapely.ops import unary_union

if TYPE_CHECKING:
    # Deferred: `PermeabilityParams` is used only as a type (safe under
    # `from __future__ import annotations`). A real top-level import would round-trip
    # permeability -> road_route -> permeability once Task 4 wires `edge_conductances` to call into
    # this module -- the same budget<->permeability cycle `mesh.py.parcel_radii` was written to
    # avoid, one hop further down the chain.
    from reblock.permeability import PermeabilityParams


@dataclass(frozen=True)
class RoadNet:
    """The planarized road graph: undirected segments between `nodes` (`nodes[k]` an (x, y) pair),
    each segment's Euclidean length, its width recovered by midpoint lookup, and whether it is
    one-way."""
    nodes: NDArray[np.float64]        # (K, 2)
    seg_a: NDArray[np.int64]
    seg_b: NDArray[np.int64]
    seg_len: NDArray[np.float64]
    seg_width: NDArray[np.float64]
    seg_oneway: NDArray[np.bool_]


def build_roadnet(roads: GeoDataFrame, params: PermeabilityParams) -> RoadNet:
    """The planarized road graph, with each segment's width (and one-way flag) recovered by
    midpoint lookup.

    `unary_union` nodes every crossing into a shared vertex -- which is the whole point, since a
    walker can turn where two roads cross -- but it also DESTROYS the association between a segment
    and the road row it came from, and with it that road's `width_m`. Recover it by asking which
    original corridors cover each segment's midpoint and taking the WIDEST: two overlapping roads
    occupy one corridor and the wider governs, the same convention `displacement` uses when it
    unions corridors. Width and one-way are recovered TOGETHER from the same governing (widest)
    road -- see the comment at the accumulation loop below for why that matters.
    """
    from reblock.budget import _explode_segments
    from reblock.permeability import buildable_widths

    if roads is None or len(roads) == 0:
        e = np.zeros(0, dtype=np.int64)
        return RoadNet(np.zeros((0, 2), dtype=np.float64), e, e, np.zeros(0, dtype=np.float64),
                       np.zeros(0, dtype=np.float64), np.zeros(0, dtype=bool))

    widths, oneway = buildable_widths(roads, params)
    pairs = _explode_segments([unary_union(list(roads.geometry))])

    coords: dict[tuple[float, float], int] = {}
    a_idx: list[int] = []
    b_idx: list[int] = []
    for pa, pb in pairs:
        for p in (pa, pb):
            coords.setdefault(p, len(coords))
        a_idx.append(coords[pa])
        b_idx.append(coords[pb])
    # `.reshape(-1, 2)` keeps the empty case (0, 2), not (0,) -- with no reshape, an all-degenerate
    # `roads` (every road explodes to zero segments) makes `nodes` 1-D and the `np.hypot` unpack
    # below raises `TypeError: hypot() takes from 2 to 3 positional arguments but 0 were given`.
    nodes = np.array(sorted(coords, key=lambda k: coords[k]), dtype=np.float64).reshape(-1, 2)
    sa = np.asarray(a_idx, dtype=np.int64)
    sb = np.asarray(b_idx, dtype=np.int64)
    seg_len = np.hypot(*(nodes[sa] - nodes[sb]).T)

    mids = shapely.points((nodes[sa] + nodes[sb]) / 2.0)
    corridors = [g.buffer(float(w) / 2.0) for g, w in zip(roads.geometry, widths, strict=True)]
    tree = STRtree(corridors)

    # shapely 2's `tree.query(query_geoms, predicate=...)` returns
    # `(index_into_query_geoms, index_into_tree_geoms)` and tests `predicate(query_geoms[i],
    # tree_geoms[j])`. The tree is built over `corridors` and queried with `mids`, so `hit_seg`
    # indexes `mids`/segments and `hit_road` indexes `corridors`/roads -- that part of the sketch
    # was already right. The predicate itself was wrong: "covers" tests
    # `mids[i].covers(corridors[j])`, a POINT covering a POLYGON, which is false for every
    # non-degenerate corridor (verified empirically -- a road's own buffered midpoint returns zero
    # hits under "covers"). We want the corridor covering the point, i.e. `mids[i].covered_by(
    # corridors[j])`, which is predicate="covered_by".
    hit_seg, hit_road = tree.query(mids, predicate="covered_by")

    seg_w = np.zeros(len(sa), dtype=np.float64)
    seg_o = np.zeros(len(sa), dtype=bool)   # default two-way: matches `buildable_widths`' own
                                             # column-absent default (`oneway=False`)
    for s, r in zip(hit_seg.tolist(), hit_road.tolist(), strict=True):
        if widths[r] > seg_w[s]:
            # Width and one-way are set TOGETHER, from the SAME governing (widest) road -- not an
            # independent AND of one-way flags across every corridor covering this midpoint. AND
            # breaks the "widest governs" convention: e.g. a one-way 12 m road and a two-way 7 m
            # road both cover a segment's midpoint. Widest wins on width (12, the one-way road's),
            # so one-way must also be that road's (True) -- but ANDing in the 7 m road's False
            # flips the result to False regardless of order. Tying the assignment to the same `>`
            # comparison that tracks the max width fixes this by construction.
            seg_w[s] = widths[r]
            seg_o[s] = bool(oneway[r])

    uncovered = seg_w == 0.0
    if uncovered.any():
        # A midpoint no corridor covers (buffer-approximation edge case at a corridor's rounded
        # cap): fall back to the narrowest road, for width AND one-way TOGETHER -- the same
        # single-governing-road invariant as above, rather than leaving one-way at an unrelated
        # default.
        narrowest = int(np.argmin(widths))
        seg_w[uncovered] = widths[narrowest]
        seg_o[uncovered] = bool(oneway[narrowest])

    return RoadNet(nodes, sa, sb, seg_len, seg_w, seg_o)


# Cap on the (points x segments) temporaries `_project` allocates in one pass. The nearest-segment
# search is a brute-force sweep -- an STRtree would win asymptotically but loses on the sizes that
# actually occur (a block's road net is hundreds of segments) and would still need this arithmetic
# to recover `t`. Chunking keeps peak memory flat instead of quadratic in block size: at 1e6
# elements the pass holds ~50 MB of float64 whatever the block.
_PROJECT_ELEMS = 1_000_000


def _project(
    net: RoadNet, pts: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """`(segment, t, offset)` per point: the index of the NEAREST segment, the fraction along it at
    which the point's perpendicular projection lands (clamped into `[0, 1]`, so a point past an end
    projects onto that end), and the distance from the point to that projection.

    Ties -- a point exactly equidistant from two segments -- go to the LOWEST segment index, which
    is `np.argmin`'s documented behaviour over a segment order `build_roadnet` fixes
    deterministically. Determinism is the property that matters (the same block must score the same
    number twice); WHICH segment wins is almost always immaterial, because the equidistant case is
    reached in practice at a shared NODE, where both segments offer the same entry point at the
    same zero offset.
    """
    ax = net.nodes[net.seg_a, 0]
    ay = net.nodes[net.seg_a, 1]
    abx = net.nodes[net.seg_b, 0] - ax
    aby = net.nodes[net.seg_b, 1] - ay
    # `_explode_segments` drops zero-length segments, so this is strictly positive and needs no
    # guard -- a degenerate segment would otherwise divide by zero here.
    len2 = net.seg_len**2

    n = pts.shape[0]
    seg = np.zeros(n, dtype=np.int64)
    frac = np.zeros(n, dtype=np.float64)
    offset = np.zeros(n, dtype=np.float64)
    step = max(1, _PROJECT_ELEMS // net.seg_a.size)
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        apx = pts[lo:hi, 0, None] - ax[None, :]
        apy = pts[lo:hi, 1, None] - ay[None, :]
        t = np.clip((apx * abx + apy * aby) / len2, 0.0, 1.0)
        dx = apx - t * abx
        dy = apy - t * aby
        d2 = dx * dx + dy * dy
        near = np.argmin(d2, axis=1)
        rows = np.arange(hi - lo)
        seg[lo:hi] = near
        frac[lo:hi] = t[rows, near]
        offset[lo:hi] = np.sqrt(d2[rows, near])
    return seg, frac, offset


def _travel_graph(net: RoadNet, seg_res: NDArray[np.float64], *, directed: bool) -> csr_matrix:
    """The CSR the route search runs over: one arc per segment, weighted by its RESISTANCE.

    With nothing one-way the graph carries HALF the arcs and `dijkstra(directed=False)` supplies
    the reverse of each internally -- there is no reason to materialize both.

    The COO -> CSR conversion SUMS duplicate `(row, col)` entries, which would silently make a
    doubled segment twice as resistive rather than leaving it alone. It cannot happen here:
    `_explode_segments` deduplicates by `frozenset` of the endpoint pair, so no two segments share
    an unordered endpoint pair.
    """
    k = net.nodes.shape[0]
    if directed:
        two_way = ~net.seg_oneway
        rows = np.concatenate([net.seg_a, net.seg_b[two_way]])
        cols = np.concatenate([net.seg_b, net.seg_a[two_way]])
        data = np.concatenate([seg_res, seg_res[two_way]])
    else:
        rows, cols, data = net.seg_a, net.seg_b, seg_res
    return csr_matrix((data, (rows, cols)), shape=(k, k))


def route_resistance(
    net: RoadNet,
    pts_a: NDArray[np.float64],
    pts_b: NDArray[np.float64],
    params: PermeabilityParams,
    cutoff: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Minimum series resistance from `pts_a[i]` to `pts_b[i]` along `net`, access legs INCLUDED,
    or `inf` where the best route does not come in at or under `cutoff[i]`.

    ## Resistance, not length

    A segment costs `seg_len / (g_road_per_m * seg_width)` and a route costs the SUM over its
    segments, so the search minimizes RESISTANCE rather than distance. That is what makes a
    mixed-width route well defined: a short narrow alley and a long wide street trade off
    correctly, and no arbitrary "which width does this route have" rule is needed.

    ## Projections attach where they land

    A point's projection lands at fraction `t` along its nearest segment, and the route pays

        r_leg  = |point - projection| / (g * w)          the approach to the network
        offset = t * seg_len / (g * w)                   the partial segment to one end of it

    with the node-to-node minimum `D` in between, minimized over both ends at each side. Snapping
    to the nearest NODE instead would charge up to half a segment -- `topology`'s median segment is
    4.83 m, against access legs that are themselves sub-metre, so that is the dominant term rather
    than a rounding error.

    The two pieces are SUMMED (approach, then travel along the segment) rather than combined by
    Pythagoras into a single `|point - node|`. They agree whenever the point lies on the network,
    which is the case the spec's `r_leg` justification cares about, but only the sum survives
    planarization: splitting a segment between a projection and the node it was reaching leaves the
    summed cost identical (`|p-m| + |m-a| = |p-a|` measured ALONG the segment) while the
    hypotenuse form strictly increases it. The monotonicity proof needs road resistance never to
    RISE when a road is added, and a new road splits existing segments at every crossing it makes.

    When both projections land on ONE segment the answer is `|t_a - t_b| * seg_len / (g * w)` plus
    the two `r_leg`s, and no graph search is involved -- going out to an end and back could only
    cost more (undirected), and where a one-way segment forbids the direct traversal the route
    round the network is still considered.

    ## The entry point is ASSIGNED here, and the spec wants it MINIMIZED

    Each point enters the network at its NEAREST segment. The design spec instead makes the entry
    a joint minimization over every projection point in `N(R)`, and says so for a reason: as roads
    are added `N(R)` grows and every route resistance falls, so the joint minimum is non-increasing
    and the road term is monotone -- which is the whole Loewner argument. An assignment has no such
    property, and MEASURED on the pinned block it does fail: over prefixes of three methods' road
    sets, ~7% of (edge, prefix-step) pairs see this resistance RISE, worst case 5.8-8.8x, because a
    newly added road lands fractionally nearer a centroid and captures its entry. In the sharpest
    observed case a road 0.19 m nearer took an entry onto a DISCONNECTED component and sent a
    finite route to `inf`. This is the same shape as `3a8dd25` ("network_efficiency monotone via
    fixed entry mapping"), which the spec cites as having killed three earlier attempts.

    Restricting to the nearest segment is what the implementation plan specifies and it is what
    keeps this to one bounded multi-source solve; the honest minimization needs a per-point search
    from a virtual source wired to every node, which is the all-pairs cost the early exit exists to
    avoid. It is recorded here rather than papered over: the metric's monotonicity is tested on
    real blocks (the spec's A4), and that test is where this has to be settled.

    ## The early exit is EXACT

    `dijkstra` runs with `limit=cutoff.max()`, which is legitimate rather than approximate:
    `D <= offset + D + offset + legs = R`, so any route with `R <= cutoff[i]` has `D` under the
    limit and is found exactly. Routes that lose to `cutoff[i]` come back `inf` -- the caller takes
    `max(footpath, 1/R)` with `cutoff = 1/footpath_g`, so it discards those whatever their value.
    A generous cutoff therefore returns a BIT-IDENTICAL answer to an infinite one, which the
    Loewner monotonicity argument depends on: an early exit that changed the computed function by
    even a rounding step would void it.

    ## `inf`, not a sentinel

    No route means `inf`, so the caller's `1.0 / R` is `0.0` exactly, with no branch, no epsilon
    and no warning, and `max(footpath, 0.0)` leaves the edge at footpath. A disconnected road
    component needing no special-case rule is a stated design property, and a sentinel would put
    the rule back at every call site. `R` is never `0` for a mesh edge (its two parcel centroids
    are distinct), so the reciprocal is finite.

    ## One multi-source solve

    One `dijkstra` call over the DISTINCT `pts_a`-side entry nodes, not one call per pair: covered
    edges join ADJACENT parcels, so neighbouring pairs enter the network at the same handful of
    nodes and a per-pair loop would recompute the same search many times over, at Python call
    overhead per pair. The cost is `O(U * (S + K log K))` for `U` distinct entry nodes over `K`
    nodes and `S` segments, with `limit` confining each search to a ball rather than the whole
    component, and `O(U * K)` for the dense result.
    """
    n = int(pts_a.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if net.seg_a.size == 0:
        return np.full(n, np.inf, dtype=np.float64)

    g = params.g_road_per_m
    seg_res = net.seg_len / (g * net.seg_width)

    sidx_a, t_a, off_a = _project(net, pts_a)
    sidx_b, t_b, off_b = _project(net, pts_b)
    legs = off_a / (g * net.seg_width[sidx_a]) + off_b / (g * net.seg_width[sidx_b])

    # Row 0 is the `seg_a` end of each point's segment, row 1 the `seg_b` end.
    ends_a = np.stack([net.seg_a[sidx_a], net.seg_b[sidx_a]])
    ends_b = np.stack([net.seg_a[sidx_b], net.seg_b[sidx_b]])
    part_a = np.stack([t_a * seg_res[sidx_a], (1.0 - t_a) * seg_res[sidx_a]])
    part_b = np.stack([t_b * seg_res[sidx_b], (1.0 - t_b) * seg_res[sidx_b]])

    directed = bool(net.seg_oneway.any())
    if directed:
        # A one-way segment permits travel from `seg_a` to `seg_b` only, and that governs the
        # partial segments at both ends as much as it governs the graph: leaving a projection
        # BACKWARDS towards the `seg_a` end is not a route, nor is arriving at the `seg_b` end and
        # running back to the projection. A zero-length partial is exempt -- a projection sitting
        # exactly on a node travels nowhere to reach it.
        part_a[0] = np.where(net.seg_oneway[sidx_a] & (part_a[0] > 0.0), np.inf, part_a[0])
        part_b[1] = np.where(net.seg_oneway[sidx_b] & (part_b[1] > 0.0), np.inf, part_b[1])

    sources = np.unique(ends_a)
    node_dist = dijkstra(_travel_graph(net, seg_res, directed=directed), directed=directed,
                         indices=sources, limit=float(np.max(cutoff)))
    row_of = np.zeros(net.nodes.shape[0], dtype=np.int64)
    row_of[sources] = np.arange(sources.size, dtype=np.int64)

    path = np.full(n, np.inf, dtype=np.float64)
    for u in (0, 1):
        rows = row_of[ends_a[u]]
        for v in (0, 1):
            path = np.minimum(path, part_a[u] + node_dist[rows, ends_b[v]] + part_b[v])

    shared = sidx_a == sidx_b
    if shared.any():
        direct = np.abs(t_a - t_b) * seg_res[sidx_a]
        if directed:
            direct = np.where(net.seg_oneway[sidx_a] & (t_b < t_a), np.inf, direct)
        path = np.where(shared, np.minimum(path, direct), path)

    total = legs + path
    return np.where(total > cutoff, np.inf, total)
