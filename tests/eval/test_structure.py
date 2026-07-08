import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon, box

from reblock.contracts import Block, Proposal
from reblock.eval.structure import StructureEval

UTM = CRS.from_epsg(32734)


def _grid_block(n: int) -> Block:
    polys, ids = [], []
    for i in range(n):
        for j in range(n):
            polys.append(box(i, j, i + 1, j + 1))
            ids.append(i * n + j)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    boundary = Polygon([(0, 0), (n, 0), (n, n), (0, n)])
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_structure_eval_emits_the_basis() -> None:
    block = _grid_block(3)
    roads = gpd.GeoDataFrame(geometry=[LineString([(1.5, 0), (1.5, 3)])], crs=UTM)
    m = StructureEval().score(block, Proposal(block_id="g", crs=UTM, roads=roads, method="x"))
    for key in ("meshedness", "four_way_fraction", "dead_end_fraction", "n_crossings",
                "n_dead_ends", "circuity", "throughput_ratio", "geometric_access_p95_m",
                "added_road_length_per_parcel", "n_cross_block_streets"):
        assert key in m.values
    assert m.eval == "structure"
    assert m.values["circuity"] >= 1.0
    assert m.values["n_cross_block_streets"] == 0.0   # no interior_boundaries on a lone block
