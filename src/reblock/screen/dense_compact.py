"""DenseCompactScreen: flag dense/compact informal blocks. Cheap pass = vectorized
density (+ optional k) gate over free kblock columns; fine pass = build only survivors
(reusing KblockSource) and keep those whose mean parcel access-depth clears mean_depth_min.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from reblock.data.kblock import KblockSource
from reblock.derive.access import parcel_access_layers

log = logging.getLogger(__name__)


class DenseCompactScreen:
    def __init__(self, blocks_path: str | Path, buildings_path: str | Path, *,
                 density_min: float = 30.0, mean_depth_min: float = 1.3,
                 k_min: float | None = None, min_buildings: int = 10) -> None:
        self.blocks_path = Path(blocks_path)
        self.buildings_path = Path(buildings_path)
        self.density_min = density_min
        self.mean_depth_min = mean_depth_min
        self.k_min = k_min
        self.min_buildings = min_buildings

    def _cheap_survivors(self, blocks: gpd.GeoDataFrame) -> list[str]:
        bid = blocks["block_id"].astype(str)
        density: pd.Series = blocks["building_count"] / (blocks["block_area_m2"] / 1e4)
        mask: pd.Series = density >= self.density_min
        if self.k_min is not None:
            mask = mask & (blocks["k_complexity"] >= self.k_min)
        return sorted(bid[mask.to_numpy()])

    def select(self) -> list[str]:
        blocks = gpd.read_parquet(
            self.blocks_path,
            columns=["block_id", "k_complexity", "building_count", "block_area_m2", "geometry"])
        survivors = self._cheap_survivors(blocks)
        log.info("cheap pass: %d/%d blocks pass density_min=%.1f%s",
                 len(survivors), len(blocks), self.density_min,
                 f", k_min={self.k_min}" if self.k_min is not None else "")
        if not survivors:
            return []
        log.info("fine pass: building %d survivor blocks (Voronoi + peel) -- the slow step",
                 len(survivors))
        src = KblockSource(self.blocks_path, self.buildings_path, region_id="screen",
                            min_buildings=self.min_buildings, block_ids=survivors)
        kept = [blk.block_id for blk in src.region().blocks
                if float(parcel_access_layers(blk, None).mean()) >= self.mean_depth_min]
        log.info("fine pass: kept %d blocks with mean access-depth >= %.2f",
                 len(kept), self.mean_depth_min)
        return sorted(kept)
