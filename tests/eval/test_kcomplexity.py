from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.contracts import Block, Proposal
from reblock.eval.kcomplexity import KComplexityEval

UTM = CRS.from_epsg(32643)


def _grid_block(n: int) -> Block:
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
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
