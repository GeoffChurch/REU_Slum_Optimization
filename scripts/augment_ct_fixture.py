#!/usr/bin/env python
"""One-time: add building_count + block_area_m2 to the committed Cape Town sample fixture
by joining them from the raw ZAF geodata (cached under outputs/kblock_raw, or downloaded)
onto the SAME 301 committed block_ids. Geometry + block set unchanged; two columns added.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from scripts.fetch_kblock_fixtures import download_dataverse_blocks

FIXTURE = Path("tests/data/kblock/blocks_capetown_sample.parquet")
RAW = Path("outputs/kblock_raw/ZAF_geodata.parquet")


def main() -> None:
    if not RAW.exists():
        download_dataverse_blocks("ZAF", RAW)
    fx = gpd.read_parquet(FIXTURE)
    fx["block_id"] = fx["block_id"].astype(str)
    if {"building_count", "block_area_m2"} <= set(fx.columns):
        print("already augmented")
        return
    raw = pd.read_parquet(RAW, columns=["block_id", "building_count", "block_area_m2"])
    raw["block_id"] = raw["block_id"].astype(str)
    merged = fx.merge(raw, on="block_id", how="left", validate="one_to_one")
    assert merged["building_count"].notna().all(), "some fixture blocks missing from raw ZAF"
    assert len(merged) == len(fx)
    gpd.GeoDataFrame(merged, geometry="geometry", crs=fx.crs).to_parquet(FIXTURE)
    print(f"augmented {FIXTURE}: +building_count +block_area_m2 ({len(merged)} blocks)")


if __name__ == "__main__":
    main()
