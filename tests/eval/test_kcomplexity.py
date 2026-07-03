from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.contracts import Block, Proposal
from reblock.eval.kcomplexity import KComplexityEval

UTM = CRS.from_epsg(32643)


def _grid_block(n: int, ox: float = 0.0, oy: float = 0.0) -> Block:
    polys = [Polygon([(ox + i, oy + j), (ox + i + 1, oy + j),
                      (ox + i + 1, oy + j + 1), (ox + i, oy + j + 1)])
             for i in range(n) for j in range(n)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_delta_k_from_interior_connector() -> None:
    # 3x3 grid: centre parcel enclosed -> k_before = 2. An interior road from
    # boundary node (1,0) up to the centre's corner (1,1) reaches the centre
    # -> k_after = 1. Values verified in Task 1's k_complexity road-sensitivity.
    block = _grid_block(3)
    connector = gpd.GeoDataFrame(geometry=[LineString([(1, 0), (1, 1)])], crs=UTM)
    proposal = Proposal(block_id="g", crs=UTM, roads=connector, method="topology")

    v = KComplexityEval().score(block, proposal).values
    assert v["k_before"] == 2
    assert v["k_after"] == 1
    assert v["delta_k"] == 1
    assert v["added_road_length_m"] == 1.0


def test_delta_k_with_nonzero_origin() -> None:
    # Same 3x3-grid-plus-connector scenario, but the block is based at
    # (100, 200) so ppg.origin is nonzero. This exercises _endpoint_keys'
    # origin subtraction: the connector's absolute coords (101,200)-(101,201)
    # only match the graph edge once shifted back by the origin. If the
    # `- origin` shift were dropped, the connector would fail to match any
    # graph edge, no edge's .road would flip, and k_after would stay 2
    # (delta_k=0) -- so this test fails without the shift and passes with it.
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
