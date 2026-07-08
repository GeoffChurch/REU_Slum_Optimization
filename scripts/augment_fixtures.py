#!/usr/bin/env python
"""One-time: add building_count + block_area_m2 to the committed blocks sample fixtures
by joining them from the matching raw geodata (cached under outputs/kblock_raw, or downloaded)
onto the SAME committed block_ids. Geometry + block set unchanged; two columns added.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from scripts.fetch_kblock_fixtures import download_dataverse_blocks

CITIES = [
    ("ZAF", Path("tests/data/kblock/blocks_capetown_sample.parquet"),
     Path("outputs/kblock_raw/ZAF_geodata.parquet")),
    ("DJI", Path("tests/data/kblock/blocks_dji_sample.parquet"),
     Path("outputs/kblock_raw/DJI_geodata.parquet")),
]


def augment(iso3: str, fixture: Path, raw: Path) -> None:
    if not raw.exists():
        download_dataverse_blocks(iso3, raw)
    fx = gpd.read_parquet(fixture)
    fx["block_id"] = fx["block_id"].astype(str)
    if {"building_count", "block_area_m2"} <= set(fx.columns):
        print(f"{fixture}: already augmented")
        return
    raw_df = pd.read_parquet(raw, columns=["block_id", "building_count", "block_area_m2"])
    raw_df["block_id"] = raw_df["block_id"].astype(str)
    merged = fx.merge(raw_df, on="block_id", how="left", validate="one_to_one")
    assert merged["building_count"].notna().all(), f"some fixture blocks missing from raw {iso3}"
    assert len(merged) == len(fx)
    gpd.GeoDataFrame(merged, geometry="geometry", crs=fx.crs).to_parquet(fixture)
    print(f"augmented {fixture}: +building_count +block_area_m2 ({len(merged)} blocks)")


def main() -> None:
    for iso3, fixture, raw in CITIES:
        augment(iso3, fixture, raw)


if __name__ == "__main__":
    main()
