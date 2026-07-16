import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point

from reblock.emit import pct_displaced, pct_paved


def _roads(*lines: list[tuple[float, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[LineString(line) for line in lines], crs="EPSG:32734")


def test_pct_paved_is_buffer_area_over_block_area() -> None:
    roads = _roads([(0, 0), (100, 0)])
    block_area = 10_000.0
    expected = roads.geometry.buffer(3.0).union_all().area / block_area
    assert abs(pct_paved(roads, 3.0, block_area) - expected) < 1e-9
    assert 0.0 < pct_paved(roads, 3.0, block_area) < 1.0


def test_pct_paved_empty_or_zero_area_is_zero() -> None:
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:32734")
    assert pct_paved(empty, 3.0, 10_000.0) == 0.0
    assert pct_paved(None, 3.0, 10_000.0) == 0.0
    assert pct_paved(_roads([(0, 0), (100, 0)]), 3.0, 0.0) == 0.0


def test_pct_displaced_is_mean_disk_graze_probability() -> None:
    # pct_displaced now wires to reblock.budget.displacement (Sum c_i / n), whose disk-graze
    # arithmetic is unit-tested in test_budget -- this just checks the wiring. Two points sit ON
    # the corridor (d=0 -> c=1 regardless of radius), two sit well clear of it (d >> r -> c=0), so
    # Sum c_i == 2 and pct_displaced == 2/4 == 0.5.
    roads = _roads([(0, 0), (100, 0)])          # corridor is |y| <= 3 along the x-axis
    pts = gpd.GeoDataFrame(geometry=[Point(50, 0), Point(50, 1), Point(50, 50), Point(50, 80)],
                           crs="EPSG:32734")     # first two inside, last two outside
    radii = np.full(len(pts), 5.0)
    assert abs(pct_displaced(roads, 3.0, pts, radii) - 0.5) < 1e-9


def test_pct_displaced_empty_roads_or_no_points_is_zero() -> None:
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:32734")
    radii = np.full(len(pts), 5.0)
    assert pct_displaced(gpd.GeoDataFrame(geometry=[], crs="EPSG:32734"), 3.0, pts, radii) == 0.0
    assert pct_displaced(None, 3.0, pts, radii) == 0.0
    empty_pts = gpd.GeoDataFrame(geometry=[], crs="EPSG:32734")
    assert pct_displaced(_roads([(0, 0), (100, 0)]), 3.0, empty_pts, np.zeros(0)) == 0.0
