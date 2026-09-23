import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from reblock.budget import displacement, road_corridor
from reblock.buildings import Discs
from reblock.emit import pct_displaced, pct_paved
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, with_width


def _roads(*lines: list[tuple[float, float]]) -> gpd.GeoDataFrame:
    return with_width(
        gpd.GeoDataFrame(geometry=[LineString(line) for line in lines], crs="EPSG:32734"),
        DEFAULT_ROAD_WIDTH_M)


def test_pct_paved_is_buffer_area_over_block_area() -> None:
    roads = _roads([(0, 0), (100, 0)])
    block_area = 10_000.0
    # half-width read off the roads, not hardcoded -- this survives a width re-base
    half = float(roads["width_m"].iloc[0]) / 2.0
    expected = roads.geometry.buffer(half).union_all().area / block_area
    assert abs(pct_paved(roads, block_area) - expected) < 1e-9
    assert 0.0 < pct_paved(roads, block_area) < 1.0


def test_pct_paved_empty_or_zero_area_is_zero() -> None:
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:32734")
    assert pct_paved(empty, 10_000.0) == 0.0
    assert pct_paved(None, 10_000.0) == 0.0
    assert pct_paved(_roads([(0, 0), (100, 0)]), 0.0) == 0.0


def test_pct_displaced_is_mean_overlap_fraction() -> None:
    # pct_displaced wires to reblock.budget.displacement (Sum c_i / n), whose overlap arithmetic is
    # unit-tested in test_budget -- this checks the wiring. Two discs sit on the corridor (c > 0),
    # two well clear of it (c == 0), so the mean is the two near shares over FOUR buildings.
    roads = _roads([(0, 0), (100, 0)])
    pts = gpd.GeoDataFrame(geometry=[Point(50, 0), Point(50, 1), Point(50, 50), Point(50, 80)],
                           crs="EPSG:32734")
    b = Discs(pts, np.full(len(pts), 5.0))
    c = b.displacement(road_corridor(roads))
    assert c[0] > 0.0 and c[1] > 0.0 and c[2] == 0.0 and c[3] == 0.0
    assert pct_displaced(roads, b) == pytest.approx(displacement(b, roads) / 4, abs=1e-15)
    assert pct_displaced(roads, b) == pytest.approx((c[0] + c[1]) / 4, abs=1e-15)


def test_pct_displaced_empty_roads_or_no_points_is_zero() -> None:
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:32734")
    b = Discs(pts, np.full(len(pts), 5.0))
    assert pct_displaced(gpd.GeoDataFrame(geometry=[], crs="EPSG:32734"), b) == 0.0
    assert pct_displaced(None, b) == 0.0
    empty = Discs(gpd.GeoDataFrame(geometry=[], crs="EPSG:32734"), np.zeros(0))
    assert pct_displaced(_roads([(0, 0), (100, 0)]), empty) == 0.0
