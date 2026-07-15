import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon
from reblock.emit import pct_paved, pct_displaced


def _roads(*lines):
    return gpd.GeoDataFrame(geometry=[LineString(l) for l in lines], crs="EPSG:32734")


def test_pct_paved_is_buffer_area_over_block_area():
    roads = _roads([(0, 0), (100, 0)])
    block_area = 10_000.0
    expected = roads.geometry.buffer(3.0).union_all().area / block_area
    assert abs(pct_paved(roads, 3.0, block_area) - expected) < 1e-9
    assert 0.0 < pct_paved(roads, 3.0, block_area) < 1.0


def test_pct_paved_empty_or_zero_area_is_zero():
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:32734")
    assert pct_paved(empty, 3.0, 10_000.0) == 0.0
    assert pct_paved(None, 3.0, 10_000.0) == 0.0
    assert pct_paved(_roads([(0, 0), (100, 0)]), 3.0, 0.0) == 0.0


def test_pct_displaced_is_fraction_of_points_in_corridor():
    roads = _roads([(0, 0), (100, 0)])          # corridor is |y| <= 3 along the x-axis
    pts = gpd.GeoDataFrame(geometry=[Point(50, 0), Point(50, 1), Point(50, 50), Point(50, 80)],
                           crs="EPSG:32734")     # first two inside, last two outside
    assert abs(pct_displaced(roads, 3.0, pts) - 0.5) < 1e-9


def test_pct_displaced_empty_roads_or_no_points_is_zero():
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:32734")
    assert pct_displaced(gpd.GeoDataFrame(geometry=[], crs="EPSG:32734"), 3.0, pts) == 0.0
    assert pct_displaced(None, 3.0, pts) == 0.0
    assert pct_displaced(_roads([(0, 0), (100, 0)]), 3.0,
                         gpd.GeoDataFrame(geometry=[], crs="EPSG:32734")) == 0.0
