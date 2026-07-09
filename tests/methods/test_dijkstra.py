from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, street_connectivity
from reblock.methods.dijkstra import _reblock_dijkstra

UTM = CRS.from_epsg(32643)


def _grid_block(n: int) -> Block:
    polys, ids = [], []
    for i in range(n):
        for j in range(n):
            polys.append(Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]))
            ids.append(i * n + j)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="grid", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_reblock_dijkstra_is_deterministic() -> None:
    block = _grid_block(5)
    r1, r2 = _reblock_dijkstra(block), _reblock_dijkstra(block)
    assert [g.wkt for g in r1.geometry] == [g.wkt for g in r2.geometry]


def test_reblock_dijkstra_roads_all_reach_the_street() -> None:
    # forest rooted at street + attached spurs -> every segment street-connected
    block = _grid_block(5)
    roads = _reblock_dijkstra(block)
    assert len(roads) > 0
    conn = street_connectivity(block.streets, roads, STREET_TOL)
    assert conn.connected_frac == 1.0


def test_reblock_dijkstra_has_ordered_drainage() -> None:
    roads = _reblock_dijkstra(_grid_block(5))
    drains = list(roads["drain"])
    assert all(d >= 1 for d in drains)               # every road serves >=1 parcel
    assert drains == sorted(drains, reverse=True)     # arterials first
    assert max(drains) > 1                             # a real arterial exists (shared prefix)
