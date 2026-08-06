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


# Cap on the (points x segments) temporaries one chunk of the projection sweep allocates. Every
# point is projected onto EVERY segment -- that is what makes the entry a joint minimization rather
# than an assignment -- so the sweep is quadratic in block size unless it is chunked. At 1e6
# elements a chunk holds ~50 MB of float64 whatever the block.
_CHUNK_ELEMS = 1_000_000


def _usable_widths(net: RoadNet, params: PermeabilityParams) -> NDArray[np.float64]:
    """Corridor width each segment offers ONE direction of travel, once the margin is set aside.

    This is `road_conductance`'s own `usable` evaluated at `lane_width` for the segment's own
    direction, vectorized per SEGMENT rather than per road -- a two-way road's directions split
    what is left after the once-paid margin, a one-way road gives its permitted direction all of it.

    Pricing a segment at its FULL `seg_width` instead would inflate a 7 m two-way road from 3.0 m
    of usable corridor to 7.0 m, and that is a capacity re-base wearing a geometry fix's clothes.
    What this module changes is the LENGTH -- a route instead of a crow-flies line -- and holding
    the capacity convention fixed is exactly what makes `L >= d` mean "road conductance strictly
    falls". The repo keeps those two apart deliberately; the 2026-07-31 width re-base moved
    footprint without moving conductance for the same reason.

    Never zero in practice: `buildable_widths` floors a road at 4.0 m one-way / 7.0 m two-way, both
    of which leave usable = 3.0 m against a 1.0 m margin. `tests/test_road_route.py` pins this
    against `lane_width` and `road_conductance` themselves so the two cannot drift.
    """
    lane = np.where(net.seg_oneway, net.seg_width,
                    params.road_margin_m + (net.seg_width - params.road_margin_m) / 2.0)
    return np.maximum(0.0, lane - params.road_margin_m)


def _projections(net: RoadNet, pts: NDArray[np.float64],
                 ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """`(t, offset)` of every point against EVERY segment: where the perpendicular projection lands
    along the segment (clamped into `[0, 1]`) and how far the point is from that landing.

    Both are `(P, S)`. The joint minimization needs every segment, not the nearest one -- the whole
    defect being fixed is that picking one segment is an ASSIGNMENT, and an assignment can move.
    """
    ax = net.nodes[net.seg_a, 0]
    ay = net.nodes[net.seg_a, 1]
    abx = net.nodes[net.seg_b, 0] - ax
    aby = net.nodes[net.seg_b, 1] - ay
    # `_explode_segments` drops zero-length segments, so this is strictly positive and needs no
    # guard -- a degenerate segment would otherwise divide by zero here.
    len2 = net.seg_len**2
    apx = pts[:, 0, None] - ax[None, :]
    apy = pts[:, 1, None] - ay[None, :]
    t = np.clip((apx * abx + apy * aby) / len2, 0.0, 1.0)
    dx = apx - t * abx
    dy = apy - t * aby
    return t, np.sqrt(dx * dx + dy * dy)


def _incidence(net: RoadNet) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """`(order, starts, nodes)` for a segmented minimum over the segments incident to each node.

    `np.concatenate([seg_a, seg_b])[order]` is grouped by node, `starts` are the group boundaries
    and `nodes` names the group. `np.minimum.reduceat` over those groups is the vectorized "best
    segment incident to this node" -- `np.minimum.at` would express it directly but is a `ufunc.at`
    scatter, orders of magnitude slower, and this runs once per point chunk.
    """
    inc = np.concatenate([net.seg_a, net.seg_b])
    order = np.argsort(inc, kind="stable")
    nodes, starts = np.unique(inc[order], return_index=True)
    return order, starts.astype(np.int64), nodes.astype(np.int64)


def _reach_costs(net: RoadNet, pts: NDArray[np.float64], walk: NDArray[np.float64],
                 seg_res: NDArray[np.float64], t: NDArray[np.float64], off: NDArray[np.float64],
                 inc: tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]],
                 *, exit_side: bool, directed: bool) -> NDArray[np.float64]:
    """`(P, K)` cheapest cost linking each point to each NODE, over every entry point on the net.

    Two ways to link a point to node `k`, and the answer is the min of them:

    * walk straight to `k` -- `|c - k| * walk`, the candidate for `k` itself as the entry point;
    * walk to the projection on some segment incident to `k`, then travel the partial segment --
      `off * walk + (t or 1-t) * seg_res`.

    That pair IS the exact candidate set. Along one segment the cost is `|c - p|/kappa_walk` (convex
    in `p`) plus a network distance that is piecewise linear in `p`, so the minimum sits at the
    projection or at an endpoint, and nowhere else. The projection is the one that usually binds now
    that walking is the dearer rate -- when both rates were equal the endpoint always won, which is
    exactly why the old road-rate leg let a route walk clean across a block.

    `exit_side` mirrors it for the far end of the route (node -> point instead of point -> node),
    which matters only under one-way segments: leaving a projection BACKWARDS is not a route, and
    neither is arriving at the far end and running back to it. A zero-length partial is exempt --
    a projection sitting on a node travels nowhere to reach it.
    """
    order, starts, nodes = inc
    base = off * walk[:, None]
    to_a = base + t * seg_res[None, :]
    to_b = base + (1.0 - t) * seg_res[None, :]
    if directed:
        ow = net.seg_oneway[None, :]
        if exit_side:
            to_b = np.where(ow & (to_b > base), np.inf, to_b)
        else:
            to_a = np.where(ow & (to_a > base), np.inf, to_a)
    via = np.minimum.reduceat(np.concatenate([to_a, to_b], axis=1)[:, order], starts, axis=1)

    direct = np.hypot(pts[:, 0, None] - net.nodes[None, :, 0],
                      pts[:, 1, None] - net.nodes[None, :, 1]) * walk[:, None]
    direct[:, nodes] = np.minimum(direct[:, nodes], via)
    return direct


def _network_arcs(net: RoadNet, seg_res: NDArray[np.float64], *, directed: bool,
                  ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """The road network's own arcs, both directions explicitly.

    Explicitly, rather than leaning on `dijkstra(directed=False)`, because the search runs over a
    graph AUGMENTED with one virtual source per point: those arcs must stay one-way, so the whole
    solve is directed and the network has to carry its own reverse arcs. One-way segments carry
    only `seg_a -> seg_b`.
    """
    two_way = ~net.seg_oneway if directed else np.ones(net.seg_a.size, dtype=bool)
    rows = np.concatenate([net.seg_a, net.seg_b[two_way]])
    cols = np.concatenate([net.seg_b, net.seg_a[two_way]])
    data = np.concatenate([seg_res, seg_res[two_way]])
    return rows, cols, data


def route_resistance(
    net: RoadNet,
    pts_a: NDArray[np.float64],
    pts_b: NDArray[np.float64],
    params: PermeabilityParams,
    cutoff: NDArray[np.float64],
    walk_res_per_m: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Minimum series resistance from `pts_a[i]` to `pts_b[i]` via `net`, access legs INCLUDED, or
    `inf` where the best route does not come in at or under `cutoff[i]`.

    ## Two rates, and why that is the whole design

    Travel ALONG a road segment costs `seg_len / (g_road_per_m * usable_width)` and a route costs
    the SUM over its segments, so the search minimizes RESISTANCE rather than distance -- which is
    what makes a mixed-width route well defined, since a short narrow alley and a long wide street
    then trade off correctly.

    The ACCESS LEG -- centroid to wherever the route joins the network -- is walking, and is priced
    at `walk_res_per_m[i]` per metre, the caller's own footpath rate for that edge
    (`1 / (footpath_g * dist)`). It is not road travel and must not be priced as though it were.

    That single distinction is load-bearing. With the leg at ROAD rate a straight line is the
    cheapest travel per metre anywhere in the model, so minimizing the entry point over the whole
    network buys shortcuts: on the spec's own zigzag fixture the direct 40 m leg to the far node
    equals the straight road's length and the two score bit-identically -- defect D1 restored in
    full -- and on real blocks the median detour collapses from 1.78 to 1.11, giving back 82-86% of
    D2. Measured legs are 2.48-2.83 m median, p90 5.0-7.7 m, so the spec's premise that they are
    "sub-metre and immaterial" does not hold and the rate is not a free choice.

    With walking as the dearer rate the model says something simple and geometric:

        take the road iff  L_i + L_j + (road resistance in walk-metres)  <  d
        benefit over the footpath ~ d / (L_i + L_j)

    -- worth a lot to a parcel fronting a road, nothing to one far from any.

    ## The entry point is MINIMIZED, not assigned

    Every point is projected onto EVERY segment, and the candidate set per point is the projection
    onto each segment plus every node (`_reach_costs` proves that is exhaustive). This is the
    spec's joint minimization over `N(R)`, and it is what makes the metric monotone: as roads are
    added `N(R)` only grows and every network distance only falls, so the minimum is
    non-increasing. Picking the NEAREST segment instead is an ASSIGNMENT, and an assignment moves
    -- measured, it made 7% of prefix steps RISE, worst 5.8-8.8x, once a road landing 0.19 m nearer
    captured an entry onto a disconnected component. That is the failure `3a8dd25` records as
    having killed three earlier attempts.

    Planarization refines rather than reroutes, so refinement preserves it exactly: splitting a
    segment between a projection and the node it was reaching leaves `off * walk + partial` and the
    two halves summing to the identical cost, and nodes are only ever added.

    ## The early exit is EXACT

    One `dijkstra` over the network augmented with a virtual source per point, seeded at
    `_reach_costs`, with `limit=cutoff.max()`. Any route with `R <= cutoff[i]` has every prefix of
    itself under the limit, so it is found exactly; only routes the caller would discard can be
    truncated, since it takes `max(footpath, 1/R)` with `cutoff = 1/footpath_g`. Seed arcs above
    `cutoff[i]` are dropped for the same reason -- a route through that node already costs more
    than the cutoff. A generous cutoff therefore returns a BIT-IDENTICAL answer to an infinite one,
    which the Loewner monotonicity argument depends on.

    ## `inf`, not a sentinel

    No route means `inf`, so the caller's `1.0 / R` is `0.0` exactly -- no branch, no epsilon, no
    warning -- and `max(footpath, 0.0)` leaves the edge at footpath. A disconnected road component
    needing no special-case rule is a stated design property, and a sentinel would put the rule
    back at every call site.
    """
    n = int(pts_a.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if net.seg_a.size == 0:
        return np.full(n, np.inf, dtype=np.float64)

    g = params.g_road_per_m
    usable = _usable_widths(net, params)
    seg_res = net.seg_len / (g * usable)
    directed = bool(net.seg_oneway.any())
    inc = _incidence(net)
    k = int(net.nodes.shape[0])
    arc_rows, arc_cols, arc_data = _network_arcs(net, seg_res, directed=directed)
    limit = float(np.max(cutoff))

    out = np.empty(n, dtype=np.float64)
    step = max(1, _CHUNK_ELEMS // max(net.seg_a.size, k))
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        m = hi - lo
        walk = walk_res_per_m[lo:hi]
        cut = cutoff[lo:hi]
        t_a, off_a = _projections(net, pts_a[lo:hi])
        t_b, off_b = _projections(net, pts_b[lo:hi])

        d0 = _reach_costs(net, pts_a[lo:hi], walk, seg_res, t_a, off_a, inc,
                          exit_side=False, directed=directed)
        d1 = _reach_costs(net, pts_b[lo:hi], walk, seg_res, t_b, off_b, inc,
                          exit_side=True, directed=directed)

        # A route that never reaches a node: both points enter the SAME segment and travel within
        # it. The node-to-node form above cannot express it, and it is the right answer whenever
        # both projections sit mid-segment -- going out to an end and back would cost more.
        along = np.abs(t_a - t_b) * seg_res[None, :]
        if directed:
            along = np.where(net.seg_oneway[None, :] & (t_b < t_a), np.inf, along)
        in_seg = ((off_a + off_b) * walk[:, None] + along).min(axis=1)

        # Virtual source per point, wired to every node it can afford to reach. Dropping seeds
        # above the pair's own cutoff is exact: a route through that node already loses.
        src, dst = np.nonzero(d0 <= cut[:, None])
        rows = np.concatenate([arc_rows, (k + src).astype(np.int64)])
        cols = np.concatenate([arc_cols, dst.astype(np.int64)])
        data = np.concatenate([arc_data, d0[src, dst]])
        aug = csr_matrix((data, (rows, cols)), shape=(k + m, k + m))
        dist = dijkstra(aug, directed=True, indices=np.arange(k, k + m), limit=limit)

        out[lo:hi] = np.minimum(in_seg, (dist[:, :k] + d1).min(axis=1))

    return np.where(out > cutoff, np.inf, out)
