from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.contracts import Block
from reblock.methods.topology import TopologyMethod

UTM = CRS.from_epsg(32643)


def _grid(n: int) -> Block:
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(n) for j in range(n)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _split_square(size: float = 10.0) -> Block:
    """A square block split into two parcels by a single mid vertical party line.

    The shared interior edge `(size/2, 0)-(size/2, size)` has BOTH endpoints on
    the boundary (they sit on the bottom and top edges) yet its midpoint is
    `size/2` units from any street -- exactly the "chord across a notch" that an
    endpoint-only street match wrongly reads as a road, while a whole-edge
    (within-corridor) match correctly rejects it.
    """
    h = size / 2
    left = Polygon([(0, 0), (h, 0), (h, size), (0, size)])
    right = Polygon([(h, 0), (size, 0), (size, size), (h, size)])
    parcels = gpd.GeoDataFrame({"parcel_id": [0, 1]}, geometry=[left, right], crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="split", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _grid_with_south_street_only(n: int) -> Block:
    """Same NxN grid as `_grid`, but `Block.streets` is only the south side of
    the boundary (a proper subset), not the whole boundary. This should leave
    every non-bottom-row parcel initially interior -- a much larger initial
    interior set than `_grid`'s full-boundary streets, which only strands the
    single center parcel -- proving `Block.streets` (not `define_roads()`'s
    boundary shortcut) drives the initial road set.
    """
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(n) for j in range(n)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    south = LineString([(i, 0) for i in range(n + 1)])
    streets = gpd.GeoDataFrame(geometry=[south], crs=UTM)
    return Block(block_id="g_south", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_proposes_roads_for_interior_parcel() -> None:
    proposal = TopologyMethod().propose(_grid(3))
    assert proposal.method == "topology" and proposal.crs == UTM
    assert proposal.roads is not None and len(proposal.roads) >= 1
    assert proposal.roads.geometry.length.sum() > 0
    assert proposal.edges is not None
    assert set(proposal.edges.columns) >= {"road", "interior", "barrier"}
    assert proposal.edges.crs == UTM
    assert len(proposal.edges) > 0


def test_propose_is_deterministic_across_runs() -> None:
    block = _grid(3)
    a = TopologyMethod(seed=0).propose(block)
    b = TopologyMethod(seed=0).propose(block)
    assert a.roads is not None and b.roads is not None
    assert sorted(g.wkt for g in a.roads.geometry) == sorted(g.wkt for g in b.roads.geometry)


def test_all_interior_parcels_connected() -> None:
    import random

    from topology import build_all_roads

    from reblock.derive.parcel_graph import to_parcel_graph
    ppg = to_parcel_graph(_grid(3))
    ppg.graph.define_roads()  # type: ignore[no-untyped-call]
    ppg.graph.define_interior_parcels()  # type: ignore[no-untyped-call]
    random.seed(0)
    build_all_roads(ppg.graph, alpha=2.0, vquiet=True)  # type: ignore[no-untyped-call]
    ppg.graph.define_interior_parcels()  # type: ignore[no-untyped-call]
    assert len(ppg.graph.interior_parcels) == 0


def test_propose_accepts_explicit_prior_none() -> None:
    # `prior` is unused today (topology is block-independent) but must be
    # accepted so TopologyMethod structurally satisfies the Method protocol.
    proposal = TopologyMethod(seed=0).propose(_grid(3), prior=None)
    assert proposal.roads is not None and len(proposal.roads) >= 1


def test_proposal_id_encodes_alpha_and_seed() -> None:
    proposal = TopologyMethod(alpha=2.0, seed=0).propose(_grid(3))
    assert proposal.proposal_id == "topology_a2.0_s0"


def test_interior_chord_with_boundary_endpoints_is_not_marked_road() -> None:
    # Mutation check for the street->edge marking. The mid party line of a
    # split square has both endpoints ON the boundary but a midpoint 5 units
    # from any street; an endpoint-only match marks it road (wrong), a
    # whole-edge/within-corridor match rejects it. (Fails under the old
    # endpoint-only predicate; passes once the edge itself must lie on the
    # street.)
    proposal = TopologyMethod(seed=0).propose(_split_square(10.0))
    assert proposal.edges is not None
    party_line = LineString([(5.0, 0.0), (5.0, 10.0)])
    marked = [bool(is_road)
              for geom, is_road in zip(proposal.edges.geometry, proposal.edges["road"],
                                       strict=True)
              if geom.equals(party_line)]
    assert marked, "party-line edge missing from proposal.edges"
    assert not any(marked), "interior party line wrongly marked road=True"


def test_partial_streets_yield_a_different_larger_proposal() -> None:
    # `Block.streets` here is only the south side of the boundary, a proper
    # subset of it -- so if `propose` truly consumes `Block.streets` (rather
    # than always deriving the initial road set from the full boundary via
    # `define_roads()`), far more parcels start out interior (6 of 9, vs. just
    # the 1 center parcel for full-boundary streets) and the greedy builder
    # must add substantially more new road to resolve them all.
    full = TopologyMethod(seed=0).propose(_grid(3))
    partial = TopologyMethod(seed=0).propose(_grid_with_south_street_only(3))
    assert full.roads is not None and partial.roads is not None
    assert partial.roads.geometry.length.sum() > full.roads.geometry.length.sum()
    assert len(partial.roads) > len(full.roads)
    assert (sorted(g.wkt for g in partial.roads.geometry)
            != sorted(g.wkt for g in full.roads.geometry))
