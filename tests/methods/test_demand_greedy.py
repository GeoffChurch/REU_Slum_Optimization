"""DemandGreedyReblocker: the prior must steer routing, and must not steer it off the substrate.

Those two properties are exactly what separate it from `clearance` (which has no prior) and from
`osm_footpaths` (which uses the lines as the roads and inherits their coverage gaps).
"""
from __future__ import annotations

from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.methods.demand_greedy import DemandGreedyReblocker, demand_edge_weights
from reblock.methods.substrates import ChordSubstrate

UTM = CRS.from_epsg(32734)


def _slab(w: int, h: int) -> Block:
    """`w` x `h` unit parcels, one building each, street frontage along the bottom edge only."""
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for j in range(h) for i in range(w)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    pts = [Point(p.centroid.x, p.centroid.y) for p in polys]
    return Block(block_id="slab", crs=UTM, boundary=cast(Polygon, unary_union(polys)),
                 parcels=parcels,
                 streets=gpd.GeoDataFrame(geometry=[LineString([(0, 0), (w, 0)])], crs=UTM),
                 building_points=gpd.GeoDataFrame(geometry=pts, crs=UTM))


def _line_at(block: Block, y: float) -> gpd.GeoDataFrame:
    minx, _, maxx, _ = block.boundary.bounds
    return gpd.GeoDataFrame(geometry=[LineString([(minx, y), (maxx, y)])], crs=block.crs)


class _Source:
    """A DesireLineSource returning one fixed line; `identity` set makes it cacheable."""

    def __init__(self, lines: gpd.GeoDataFrame, identity: object = ("test", "fixed")) -> None:
        self._lines, self.identity = lines, identity

    def desire_lines(self, bbox: tuple[float, float, float, float], crs: CRS
                     ) -> gpd.GeoDataFrame:
        del bbox
        return self._lines.to_crs(crs)


def test_demand_makes_corridor_edges_cheaper_and_nothing_more_expensive() -> None:
    """The mechanism in one assertion. Without this the method is just a slower clearance."""
    block = _slab(4, 4)
    graph = ChordSubstrate().build(block)
    uniform = demand_edge_weights(graph, gpd.GeoDataFrame(geometry=[], crs=block.crs))
    primed = demand_edge_weights(graph, _line_at(block, 2.0))

    assert (uniform > 0).all()
    assert (primed <= uniform + 1e-9).all(), "the prior made some edge MORE expensive"
    assert (primed < uniform - 1e-9).any(), "the demand corridor changed no edge cost at all"


def test_a_sparse_prior_still_reaches_every_parcel() -> None:
    """The reason to use desire lines as a PRIOR rather than as the roads: one line must still
    yield a network that serves the whole block, which `osm_footpaths` cannot promise."""
    block = _slab(4, 4)
    proposal = DemandGreedyReblocker(
        desire_source=_Source(_line_at(block, 2.0)), depth_target=1).propose(block)

    assert cast(int, proposal.params["demand_segments"]) == 1
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
    roads = DemandGreedyReblocker(
        desire_source=_Source(diagonal), depth_target=1).propose(block).roads
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
    assert DemandGreedyReblocker(desire_source=live).identity is None
    assert DemandGreedyReblocker(desire_source=None).identity is not None
