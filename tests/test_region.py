from typing import cast

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.methods.dijkstra import DijkstraReblocker
from reblock.region import region_block

UTM = CRS.from_epsg(32643)


def _grid_block(x0: int, y0: int, n: int, block_id: str = "grid") -> Block:
    polys, ids = [], []
    for i in range(n):
        for j in range(n):
            polys.append(Polygon([
                (x0 + i, y0 + j), (x0 + i + 1, y0 + j),
                (x0 + i + 1, y0 + j + 1), (x0 + i, y0 + j + 1)]))
            ids.append(i * n + j)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, unary_union(polys))
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id=block_id, crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_region_block_unions_two_adjacent_blocks_dropping_the_shared_interior_edge() -> None:
    a = _grid_block(0, 0, 3, "a")
    b = _grid_block(3, 0, 3, "b")
    rb = region_block([a, b])

    assert len(rb.parcels) == 18
    assert sorted(rb.parcels["parcel_id"]) == list(range(18))
    assert rb.crs == UTM

    street_union = unary_union(rb.streets.geometry).buffer(1e-6)
    shared_edge = LineString([(3, 0), (3, 3)])
    outer_edge = LineString([(0, 0), (0, 3)])
    assert not shared_edge.within(street_union)
    assert outer_edge.within(street_union)


def test_region_block_methods_produce_cross_block_roads() -> None:
    # A region built from two adjacent 3x3 blocks sharing the edge x=3: since region_block
    # drops that interior edge from streets, dijkstra must route across it to reach the
    # street on either side -- proving joint (not per-block) reblocking.
    a = _grid_block(0, 0, 3, "a")
    b = _grid_block(3, 0, 3, "b")
    rb = region_block([a, b])

    roads = DijkstraReblocker().propose(rb).roads
    assert roads is not None and len(roads) > 0

    boundary_line = LineString([(3, 0), (3, 3)])
    assert any(geom.intersects(boundary_line) for geom in roads.geometry)


def test_region_block_rejects_empty_list() -> None:
    with pytest.raises(ValueError):
        region_block([])


def test_region_block_rejects_crs_mismatch() -> None:
    a = _grid_block(0, 0, 3, "a")
    other_crs = CRS.from_epsg(32644)
    b = Block(block_id="b", crs=other_crs, boundary=a.boundary,
              parcels=a.parcels.set_crs(other_crs, allow_override=True),
              streets=a.streets.set_crs(other_crs, allow_override=True))
    with pytest.raises(ValueError):
        region_block([a, b])
