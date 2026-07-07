from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon, box

from reblock.contracts import Block, Proposal
from reblock.eval.kcomplexity import KComplexityEval, WeakDualKEval
from reblock.methods.topology import TopologyMethod

UTM = CRS.from_epsg(32643)


def _grid_block(n: int, ox: float = 0.0, oy: float = 0.0) -> Block:
    polys = [Polygon([(ox + i, oy + j), (ox + i + 1, oy + j),
                      (ox + i + 1, oy + j + 1), (ox + i, oy + j + 1)])
             for i in range(n) for j in range(n)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _grid5() -> Block:
    polys = [box(i, j, i + 1, j + 1) for i in range(5) for j in range(5)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(25))}, geometry=polys, crs=UTM)
    b = cast(Polygon, parcels.geometry.union_all())
    return Block(block_id="g5", crs=UTM, boundary=b, parcels=parcels,
                 streets=gpd.GeoDataFrame(geometry=[b.exterior], crs=UTM))


def test_delta_k_from_interior_connector() -> None:
    # 3x3 grid: centre parcel enclosed -> k_before = 2. An interior road from
    # boundary node (1,0) up to the centre's corner (1,1) reaches the centre
    # -> k_after = 1. The BFS peel (parcel_access_layers) agrees with the old
    # weak-dual on this grid.
    block = _grid_block(3)
    connector = gpd.GeoDataFrame(geometry=[LineString([(1, 0), (1, 1)])], crs=UTM)
    proposal = Proposal(block_id="g", crs=UTM, roads=connector, method="topology")

    metrics = KComplexityEval().score(block, proposal)
    v = metrics.values
    assert v["k_before"] == 2
    assert v["k_after"] == 1
    assert v["delta_k"] == 1
    assert v["added_road_length_m"] == 1.0

    access_before = metrics.fields["access_before"]
    access_after = metrics.fields["access_after"]
    assert access_before.index.name == "parcel_id"
    assert access_after.index.name == "parcel_id"
    assert access_after.max() == 1


def test_delta_k_with_nonzero_origin() -> None:
    # Same 3x3-grid-plus-connector scenario, but the block is based at
    # (100, 200). The peel is purely geometric (distances/adjacency), so it
    # needs no origin-relative bookkeeping to get this right.
    block = _grid_block(3, ox=100.0, oy=200.0)
    connector = gpd.GeoDataFrame(geometry=[LineString([(101, 200), (101, 201)])], crs=UTM)
    proposal = Proposal(block_id="g", crs=UTM, roads=connector, method="topology")

    v = KComplexityEval().score(block, proposal).values
    assert v["k_before"] == 2
    assert v["k_after"] == 1
    assert v["delta_k"] == 1
    assert v["added_road_length_m"] == 1.0


def test_no_roads_leaves_k_unchanged() -> None:
    # Contract: with no proposed roads, k_after == k_before, delta_k == 0,
    # and added_road_length_m == 0. Uses roads=None (the empty-roads path).
    block = _grid_block(3)
    proposal = Proposal(block_id="g", crs=UTM, roads=None, method="topology")

    v = KComplexityEval().score(block, proposal).values
    assert v["k_after"] == v["k_before"]
    assert v["delta_k"] == 0.0
    assert v["added_road_length_m"] == 0.0


def test_topology_roads_are_street_connected_and_unchanged() -> None:
    # Gate: topology's interior roads reach block.streets, so the connectivity-
    # aware metric neither drops its access (k stays 1) nor flags disconnection.
    block = _grid5()
    proposal = TopologyMethod(alpha=2.0, seed=0).propose(block)
    m = KComplexityEval().score(block, proposal)
    assert m.values["k_after"] == 1.0
    assert m.values["connected_road_frac"] == 1.0
    assert m.values["n_road_components"] >= 1


def test_diagnostics_present_and_zero_for_no_roads() -> None:
    block = _grid5()
    empty = Proposal(block_id="g5", crs=UTM, roads=gpd.GeoDataFrame(geometry=[], crs=UTM),
                     method="none")
    m = KComplexityEval().score(block, empty)
    assert m.values["n_road_components"] == 0.0
    assert m.values["connected_road_frac"] == 0.0


def test_geometric_access_emitted() -> None:
    # The geometric (Dijkstra-metres) access measure rides alongside the
    # topological peel-k: a scalar summary in .values and a per-parcel
    # field in .fields, indexed like the peel fields.
    block = _grid5()
    proposal = Proposal(block_id="g5", crs=UTM, roads=None, method="none")

    m = KComplexityEval().score(block, proposal)
    assert m.values["geometric_access_max_m"] >= 0.0
    assert len(m.fields["geometric_access_m"]) == len(block.parcels)


def test_weakdual_k_pins_old_behavior() -> None:
    # WeakDualKEval retains the old topology-weak-dual logic verbatim, for
    # Brelsford/literature comparability, and emits no per-parcel fields.
    block = _grid_block(3)
    proposal = Proposal(block_id="g", crs=UTM, roads=None, method="topology")

    metrics = WeakDualKEval().score(block, proposal)
    assert metrics.eval == "weakdual_k"
    assert metrics.values["k_before"] == 2
    assert metrics.fields == {}
