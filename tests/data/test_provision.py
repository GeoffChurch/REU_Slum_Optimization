import shutil
from pathlib import Path

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import Point, box

from reblock.data.provision import (
    cached_kblock_source,
    ensure_city_data,
    filter_to_shortlist,
    tiles_for,
)

ROOT = Path(__file__).resolve().parents[1]
CT_BLOCKS = ROOT / "data" / "kblock" / "blocks_capetown_sample.parquet"
CT_BLD = ROOT / "data" / "kblock" / "buildings_capetown_sample.parquet"


def _seed(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy(CT_BLOCKS, cache / "blocks_capetown_full.parquet")
    shutil.copy(CT_BLD, cache / "buildings_capetown_full.parquet")


def test_ensure_city_data_uses_cache_no_download(tmp_path: Path) -> None:
    _seed(tmp_path)
    bp, dp = ensure_city_data("capetown", cache_dir=tmp_path)   # must NOT hit the network
    assert bp.exists() and dp.exists()
    assert bp == tmp_path / "blocks_capetown_full.parquet"


def test_cached_kblock_source_builds_from_cache(tmp_path: Path) -> None:
    _seed(tmp_path)
    src = cached_kblock_source("capetown", block_ids=["ZAF.9.3.1_1_44882"], cache_dir=tmp_path)
    blocks = list(src.region().blocks)
    assert [b.block_id for b in blocks] == ["ZAF.9.3.1_1_44882"]


WGS = CRS.from_epsg(4326)


def _tiles() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"tile_url": ["a.csv.gz", "b.csv.gz", "c.csv.gz"]},
        geometry=[box(18, -34, 19, -33), box(36, -2, 37, -1), box(25, -20, 26, -19)],
        crs=WGS)


def test_tiles_for_returns_only_intersecting_tiles() -> None:
    shortlist = gpd.GeoDataFrame(
        geometry=[box(18.5, -33.9, 18.6, -33.8), box(36.8, -1.3, 36.9, -1.2)], crs=WGS)
    assert sorted(tiles_for(shortlist, _tiles())) == ["a.csv.gz", "b.csv.gz"]


def test_tiles_for_is_not_fooled_by_the_bounding_rectangle() -> None:
    """The bug this replaces: a bbox around a ZAF+KEN shortlist spans everything between them."""
    shortlist = gpd.GeoDataFrame(
        geometry=[box(18.5, -33.9, 18.6, -33.8), box(36.8, -1.3, 36.9, -1.2)], crs=WGS)
    assert "c.csv.gz" not in tiles_for(shortlist, _tiles())


def test_filter_to_shortlist_keeps_only_points_inside_a_block() -> None:
    shortlist = gpd.GeoDataFrame(geometry=[box(18.5, -33.9, 18.6, -33.8)], crs=WGS)
    points = gpd.GeoDataFrame(
        {"confidence": [0.9, 0.9]},
        geometry=[Point(18.55, -33.85), Point(25.0, -30.0)],   # inside, then far away
        crs=WGS)
    kept = filter_to_shortlist(points, shortlist)
    assert len(kept) == 1
    point_geom = kept.geometry.iloc[0]
    assert point_geom.x == pytest.approx(18.55)  # type: ignore[attr-defined]


def test_filter_to_shortlist_deduplicates_overlapping_polygons() -> None:
    """A point inside multiple overlapping shortlist blocks is returned exactly once."""
    # Two overlapping boxes
    shortlist = gpd.GeoDataFrame(
        geometry=[box(18.5, -33.9, 18.7, -33.7), box(18.6, -33.8, 18.8, -33.6)],
        crs=WGS)
    # One point inside both overlapping boxes
    points = gpd.GeoDataFrame(
        {"confidence": [0.9]},
        geometry=[Point(18.65, -33.75)],
        crs=WGS)
    kept = filter_to_shortlist(points, shortlist)
    assert len(kept) == 1
    assert kept.iloc[0]["confidence"] == 0.9
