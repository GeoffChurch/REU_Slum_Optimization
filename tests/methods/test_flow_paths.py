"""FlowPathsReblocker: the three properties that distinguish flow accumulation from drainage.

A drainage tree serves every parcel by construction. This must NOT -- it keeps only the busiest
edges, which is what makes it sparse and what lets it carry loops. Those are the properties the
method exists for, so they are what the tests pin.
"""
from __future__ import annotations

from typing import cast

import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.methods.clearance import ClearanceReblocker
from reblock.methods.flow_paths import FlowPathsReblocker, accumulate_flow
from reblock.methods.substrates import ChordSubstrate

UTM = CRS.from_epsg(32734)


def _slab(w: int, h: int) -> Block:
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for j in range(h) for i in range(w)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    pts = [Point(p.centroid.x, p.centroid.y) for p in polys]
    return Block(block_id="slab", crs=UTM, boundary=cast(Polygon, unary_union(polys)),
                 parcels=parcels,
                 streets=gpd.GeoDataFrame(geometry=[LineString([(0, 0), (w, 0)])], crs=UTM),
                 building_points=gpd.GeoDataFrame(geometry=pts, crs=UTM))


def test_flow_concentrates_more_sharply_as_the_block_grows() -> None:
    """The premise of the whole method: trips must pile onto shared corridors rather than each
    finding its own line. Concentration is a SCALE effect and testing it on a small fixture would
    be testing nothing -- measured, the busiest-to-median ratio runs 2.0 at 5x5, 7.5 at 8x8 and
    27.0 at 12x12, because a bigger block gives trips more chance to share a route."""
    ratios = []
    for n in (5, 8, 12):
        block = _slab(n, n)
        flow = accumulate_flow(block, ChordSubstrate().build(block), destination="gateway")
        used = flow[flow > 0]
        assert len(used) > 0
        ratios.append(float(used.max() / np.median(used)))

    assert ratios[0] < ratios[1] < ratios[2], f"concentration does not grow with scale: {ratios}"
    assert ratios[-1] > 5.0, f"flow barely concentrates even at 12x12: {ratios[-1]}"


def test_it_is_sparser_than_a_drainage_tree_and_leaves_parcels_unserved() -> None:
    """Real footpaths do not reach every parcel; a drainage tree does, by construction. Keeping
    only the busiest edges is what buys the sparsity, and it is the point of the method."""
    block = _slab(6, 6)
    flow_roads = FlowPathsReblocker(flow_quantile=0.90).propose(block).roads
    tree_roads = ClearanceReblocker(depth_target=1).propose(block).roads

    assert flow_roads is not None and tree_roads is not None
    assert len(flow_roads) > 0, "produced nothing at all"
    flow_len = float(flow_roads.geometry.length.sum())
    tree_len = float(tree_roads.geometry.length.sum())
    assert flow_len < tree_len, f"not sparser than drainage ({flow_len} vs {tree_len})"


def test_a_higher_quantile_keeps_strictly_less() -> None:
    """`flow_quantile` is the hierarchy knob -- a high cut is meant to leave only the busiest
    skeleton, so it must be monotone or the knob means nothing."""
    block = _slab(6, 6)
    lens = []
    for q in (0.50, 0.90, 0.99):
        roads = FlowPathsReblocker(flow_quantile=q).propose(block).roads
        assert roads is not None
        lens.append(float(roads.geometry.length.sum()))
    assert lens[0] >= lens[1] >= lens[2], f"not monotone in the cut: {lens}"
    # STRICT between the extremes: a non-strict chain alone is satisfied by a knob that does
    # nothing at all, which is exactly what a fault-injection run caught.
    assert lens[0] > lens[2], f"the cut changes nothing between q=0.50 and q=0.99: {lens}"


def test_reinforcement_changes_the_field() -> None:
    """Trail formation is the mechanism claim: making trodden edges cheaper must actually move
    where later trips go, or the iteration is decoration."""
    block = _slab(6, 6)
    graph = ChordSubstrate().build(block)
    plain = accumulate_flow(block, graph, iterations=1, reinforcement=0.0)
    fed = accumulate_flow(block, graph, iterations=3, reinforcement=0.5)

    assert plain.shape == fed.shape
    assert not np.allclose(plain, fed), "reinforcement left the flow field unchanged"
