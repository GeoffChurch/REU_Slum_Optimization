"""DemandGreedyReblocker: the prior must steer routing, and must not steer it off the substrate.

Those two properties are exactly what separate it from `clearance` (which has no prior) and from
`osm_footpaths` (which uses the lines as the roads and inherits their coverage gaps).
"""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import replace
from typing import cast

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from reblock.buildings import SpacingDiscs
from reblock.contracts import Block
from reblock.methods.demand_greedy import (
    DemandGreedyReblocker,
    demand_edge_weights,
    field_demand,
)
from reblock.methods.desire_lines import (
    DesireField,
    NoDesire,
    WeightedLines,
    mapped_field,
)
from reblock.methods.osm_footpaths import block_footpaths
from reblock.methods.substrates import ChordSubstrate
from reblock.permeability import DEFAULT_ROAD_WIDTH_M

UTM = CRS.from_epsg(32734)

# Every setting spelled once, at `conf/method/demand_greedy.yaml`'s values but with no prior; each
# test supplies its own prior and varies what it is about with `replace`.
DEMAND = DemandGreedyReblocker(desire_source=NoDesire(), substrate=ChordSubstrate(), buffer_m=3.0,
                               eps=0.1, gamma=1.0, depth_target=2, max_roads=400,
                               road_width_m=DEFAULT_ROAD_WIDTH_M)


def _slab(w: int, h: int) -> Block:
    """`w` x `h` unit parcels, one building each, street frontage along the bottom edge only."""
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for j in range(h) for i in range(w)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    pts = [Point(p.centroid.x, p.centroid.y) for p in polys]
    return Block(block_id="slab", crs=UTM, boundary=cast(Polygon, unary_union(polys)),
                 parcels=parcels,
                 streets=gpd.GeoDataFrame(geometry=[LineString([(0, 0), (w, 0)])], crs=UTM),
                 building_geometries=gpd.GeoDataFrame(geometry=pts, crs=UTM),
                 source_content_hash=None, building_tier=SpacingDiscs)


def _line_at(block: Block, y: float) -> gpd.GeoDataFrame:
    minx, _, maxx, _ = block.boundary.bounds
    return gpd.GeoDataFrame(geometry=[LineString([(minx, y), (maxx, y)])], crs=block.crs)


class _Source:
    """A mapped source returning one fixed line, both as the network and as its one-group field;
    `identity` set makes it cacheable."""

    def __init__(self, lines: gpd.GeoDataFrame, identity: Hashable = ("test", "fixed")) -> None:
        self._lines, self.identity = lines, identity

    def footpaths(self, bbox: tuple[float, float, float, float], crs: CRS) -> gpd.GeoDataFrame:
        del bbox
        return self._lines.to_crs(crs)

    def desire_field(self, block: Block) -> DesireField:
        return mapped_field(block_footpaths(self, block))


def _weights(field: DesireField) -> np.ndarray:
    return demand_edge_weights(ChordSubstrate().build(_slab(4, 4)), field, buffer_m=3.0, eps=0.1,
                               gamma=1.0)


def test_demand_makes_corridor_edges_cheaper_and_nothing_more_expensive() -> None:
    """The mechanism in one assertion. Without this the method is just a slower clearance."""
    block = _slab(4, 4)
    uniform = _weights(NoDesire().desire_field(block))
    primed = _weights(mapped_field(_line_at(block, 2.0)))

    assert (uniform > 0).all()
    assert (primed <= uniform + 1e-9).all(), "the prior made some edge MORE expensive"
    assert (primed < uniform - 1e-9).any(), "the demand corridor changed no edge cost at all"


def test_demand_is_the_total_weight_of_the_corridors_holding_a_point() -> None:
    """Groups add, each by its own weight; an empty group and a zero weight add nothing.

    FAULT INJECTION: setting a hit to the group's weight instead of adding it (`out[hit] =`, which
    is all a binary corridor needed) lets the last group overwrite the rest -- here the zero-weight
    one, reading [0.25, 0, 0, 0].
    """
    low, high = _line_at(_slab(4, 4), 1.0), _line_at(_slab(4, 4), 3.0)
    empty = gpd.GeoDataFrame(geometry=[], crs=UTM)
    field = DesireField(groups=(WeightedLines(low, 0.25), WeightedLines(high, 0.5),
                                WeightedLines(empty, 7.0), WeightedLines(high, 0.0)))
    xy = np.array([[2.0, 1.0], [2.0, 2.0], [2.0, 3.0], [2.0, 6.0]])
    np.testing.assert_array_equal(field_demand(field, xy, buffer_m=1.0), [0.25, 0.75, 0.5, 0.0])


def test_a_mapped_field_is_the_binary_corridor() -> None:
    """One group at weight 1 is inside-or-not -- the demand `osm_footpaths`' lines have always had,
    the corridor's boundary counting as inside."""
    field = mapped_field(_line_at(_slab(4, 4), 2.0))
    xy = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 3.5], [1.0, 0.0]])
    np.testing.assert_array_equal(field_demand(field, xy, buffer_m=1.0), [1.0, 1.0, 0.0, 0.0])


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf")])
def test_a_weight_must_be_a_finite_non_negative_number(weight: float) -> None:
    with pytest.raises(ValueError, match="finite and >= 0"):
        WeightedLines(gpd.GeoDataFrame(geometry=[], crs=UTM), weight)


def test_a_sparse_prior_still_reaches_every_parcel() -> None:
    """The reason to use desire lines as a PRIOR rather than as the roads: one line must still
    yield a network that serves the whole block, which `osm_footpaths` cannot promise."""
    block = _slab(4, 4)
    proposal = replace(
        DEMAND, desire_source=_Source(_line_at(block, 2.0)), depth_target=1).propose(block)

    assert cast(int, proposal.params["demand_segments"]) == 1
    assert cast(int, proposal.params["demand_groups"]) == 1
    assert cast(int, proposal.params["max_depth_after"]) <= 1
    roads = proposal.roads
    assert roads is not None and len(roads) > 1, "a one-line prior produced a one-road network"


def test_the_prior_can_only_choose_among_substrate_edges() -> None:
    """A corridor laid diagonally across the block must not pull a road off the substrate.

    Every road's first vertex is by construction the parcel it serves (its representative point,
    which in this fixture is also its building point) -- clearance does the same, so that is not
    the invariant. The invariant is that everything AFTER that anchor is a substrate node, and
    substrate edges run between parcels: no cost field, however tempting, can route through a
    building.
    """
    block = _slab(4, 4)
    diagonal = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (4, 4)])], crs=block.crs)
    roads = replace(
        DEMAND, desire_source=_Source(diagonal), depth_target=1).propose(block).roads
    assert roads is not None

    nodes = {(round(x, 6), round(y, 6)) for x, y in ChordSubstrate().build(block).pts}
    street = unary_union(list(block.streets.geometry))
    for road in roads.geometry:
        for x, y in list(road.coords)[1:]:
            on_node = (round(x, 6), round(y, 6)) in nodes
            on_street = street.distance(Point(x, y)) <= 1e-6   # the final snap to the street
            assert on_node or on_street, f"road left the substrate at ({x}, {y})"


def test_a_live_source_makes_the_method_uncacheable() -> None:
    """A live fetch must propagate None upward, or a memoized proposal could be served for OSM
    that has since changed -- the same propagation ClearanceReblocker does for its substrate."""
    live = _Source(gpd.GeoDataFrame(geometry=[], crs=UTM), identity=None)
    assert replace(DEMAND, desire_source=live).identity is None
    assert replace(DEMAND, desire_source=NoDesire()).identity is not None


def test_two_sources_on_one_block_do_not_share_a_proposal_id() -> None:
    """The eval cache keys on (block, proposal_id). Two desire sources route two different trees on
    one block, so the id must tell them apart -- the consensus and the OSM prior both drive this
    method.

    FAULT INJECTION: dropping the source hash from the id makes the two ids equal.
    """
    block = _slab(4, 4)
    a = replace(DEMAND, desire_source=_Source(_line_at(block, 2.0), ("a",))).propose(block)
    b = replace(DEMAND, desire_source=_Source(_line_at(block, 2.0), ("b",))).propose(block)
    assert a.proposal_id != b.proposal_id
