"""`building_count` is a CHOICE, resolved upstream, and the choice reaches the cache key.

kblock ships an Ecopia-derived count; this project's parcels come from Open Buildings points.
They agree to a median 1.10x city-wide and diverge 5-20x on settlements newer than Ecopia's
imagery -- which is the population the screens exist to find. See `reblock.data.counts`.
"""
from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import Point, Polygon

from reblock.data.counts import BuildingCount, KblockCount, OpenBuildingsCount, resolved

UTM = CRS.from_epsg(32734)
Fixture: TypeAlias = tuple[gpd.GeoDataFrame, Path]


@pytest.fixture
def block_and_points(tmp_path: Path) -> Fixture:
    """One block whose shipped count (2) disagrees with the points inside it (5)."""
    poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    blocks = gpd.GeoDataFrame({"block_id": ["b1"], "building_count": [2]},
                              geometry=[poly], crs=UTM)
    pts = gpd.GeoDataFrame(
        geometry=[Point(x, 50) for x in (10, 20, 30, 40, 50)]
        + [Point(500, 500)],                     # outside: must not be counted
        crs=UTM)
    path = tmp_path / "pts.parquet"
    pts.to_parquet(path)
    return blocks, path


def test_both_strategies_satisfy_the_protocol() -> None:
    assert isinstance(KblockCount(), BuildingCount)
    assert isinstance(OpenBuildingsCount(), BuildingCount)


def test_kblock_returns_the_shipped_column(block_and_points: Fixture) -> None:
    blocks, path = block_and_points
    assert KblockCount().counts(blocks, path).tolist() == [2.0]


def test_open_buildings_counts_points_inside_the_block(block_and_points: Fixture) -> None:
    """FAULT INJECTION: `predicate="within"` instead of `"contains"` asks whether the BLOCK is
    inside a point -- it returns nothing and this reads 0.0, silently, with no error."""
    blocks, path = block_and_points
    assert OpenBuildingsCount().counts(blocks, path).tolist() == [5.0]


def test_resolved_overwrites_the_column_so_metrics_never_choose(block_and_points: Fixture) -> None:
    """The point of resolving upstream: downstream reads `building_count` and cannot tell.

    FAULT INJECTION: adding a new column instead of overwriting leaves every metric on the old
    number and this fails -- which is what the screen did before 2026-09-16.
    """
    blocks, path = block_and_points
    assert resolved(blocks, path, OpenBuildingsCount())["building_count"].tolist() == [5.0]
    assert resolved(blocks, path, KblockCount())["building_count"].tolist() == [2.0]
    assert blocks["building_count"].tolist() == [2], "must not mutate the caller's frame"


def test_the_two_strategies_have_different_identities() -> None:
    """They produce different counts, so they must never share a cache entry.

    FAULT INJECTION: returning a constant identity from both makes a run that switched sources
    silently reuse the other's screen selection.
    """
    assert KblockCount().identity != OpenBuildingsCount().identity
