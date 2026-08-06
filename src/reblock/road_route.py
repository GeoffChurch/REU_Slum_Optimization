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
