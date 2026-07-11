"""KblockSource.block_geometries: building_count is exposed alongside block_id + geometry
(a per-block building count, present for kblock sources) -- additive to the existing
block_id/geometry accessor, used by RegionBuilders to budget growth on buildings (a parcel
proxy). See docs/superpowers/specs/2026-07-11-dense-cluster-region-builder-design.md Design.1.
"""
from pathlib import Path

from reblock.data.kblock import KblockSource

DJI_BLOCKS = Path(__file__).resolve().parent / "data" / "kblock" / "blocks_dji_sample.parquet"
DJI_BLD = Path(__file__).resolve().parent / "data" / "kblock" / "buildings_dji_sample.parquet"


def test_block_geometries_includes_building_count() -> None:
    src = KblockSource(DJI_BLOCKS, DJI_BLD, region_id="dji")
    bg = src.block_geometries()
    assert "building_count" in bg.columns
    assert "block_id" in bg.columns and "geometry" in bg.columns
    row = bg[bg.block_id == "DJI.3_1_3238"].iloc[0]
    assert int(row.building_count) == 53           # matches the parquet


def test_block_geometries_building_count_survives_block_ids_filter() -> None:
    src = KblockSource(DJI_BLOCKS, DJI_BLD, region_id="dji", block_ids=["DJI.3_1_3238"])
    bg = src.block_geometries()
    assert list(bg["block_id"]) == ["DJI.3_1_3238"]
    assert int(bg.iloc[0]["building_count"]) == 53


def test_block_geometries_building_count_survives_bbox_window() -> None:
    src = KblockSource(DJI_BLOCKS, DJI_BLD, region_id="dji")
    allg = src.block_geometries()
    minx, miny, maxx, maxy = allg.total_bounds
    sub = (minx, miny, (minx + maxx) / 2, (miny + maxy) / 2)
    windowed = src.block_geometries(sub)
    assert 0 < len(windowed) < len(allg)
    assert "building_count" in windowed.columns
