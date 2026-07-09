"""DenseCompactScreen: flag dense/compact informal blocks. Cheap pass = vectorized
density (+ optional k) gate over free kblock columns; fine pass = build only survivors
(reusing the source's KblockSource paths) and keep those whose mean parcel access-depth
clears mean_depth_min. The fine-pass depth goes through reblock.derivations.access_before
(a derive() call), so building a survivor here is an L1 hit when run() later scores it.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd

from reblock.contracts import Source
from reblock.data.kblock import KblockSource
from reblock.derivations import access_before

log = logging.getLogger(__name__)


class DenseCompactScreen:
    def __init__(self, *, density_min: float = 30.0, mean_depth_min: float = 1.3,
                 max_depth_min: float | None = None, k_min: float | None = None,
                 min_buildings: int = 10) -> None:
        self.density_min = density_min
        self.mean_depth_min = mean_depth_min
        self.max_depth_min = max_depth_min
        self.k_min = k_min
        self.min_buildings = min_buildings

    def _cheap_survivors(self, blocks: gpd.GeoDataFrame) -> list[str]:
        bid = blocks["block_id"].astype(str)
        density: pd.Series = blocks["building_count"] / (blocks["block_area_m2"] / 1e4)
        mask: pd.Series = density >= self.density_min
        if self.k_min is not None:
            mask = mask & (blocks["k_complexity"] >= self.k_min)
        return sorted(bid[mask.to_numpy()])

    def select(self, source: Source) -> list[str]:
        if not isinstance(source, KblockSource):
            raise TypeError(
                f"DenseCompactScreen needs a KblockSource (kblock columns); "
                f"got {type(source).__name__}")
        blocks = gpd.read_parquet(
            source.blocks_path,
            columns=["block_id", "k_complexity", "building_count", "block_area_m2", "geometry"])
        survivors = self._cheap_survivors(blocks)
        log.info("cheap pass: %d/%d blocks pass density_min=%.1f%s",
                 len(survivors), len(blocks), self.density_min,
                 f", k_min={self.k_min}" if self.k_min is not None else "")
        if not survivors:
            return []
        log.info("fine pass: building %d survivor blocks (Voronoi + peel) -- the slow step",
                 len(survivors))
        src = KblockSource(source.blocks_path, source.buildings_path, region_id="screen",
                           min_buildings=self.min_buildings, block_ids=survivors)
        # One access-depth series per block; keep those clearing the mean gate (and the
        # optional max gate), ranked deepest-parcel-first so a downstream max_blocks picks
        # the worst-access blocks rather than an alphabetical slice.
        ranked: list[tuple[float, str]] = []
        for blk in src.region().blocks:
            depths = access_before(blk)
            mean_d, max_d = float(depths.mean()), float(depths.max())
            if mean_d < self.mean_depth_min:
                continue
            if self.max_depth_min is not None and max_d < self.max_depth_min:
                continue
            ranked.append((max_d, blk.block_id))
        ranked.sort(key=lambda r: (-r[0], r[1]))   # max-depth desc; ties by block_id asc
        log.info("fine pass: kept %d blocks (mean-depth >= %.2f%s), ranked by max access-depth",
                 len(ranked), self.mean_depth_min,
                 "" if self.max_depth_min is None else f", max-depth >= {self.max_depth_min:.1f}")
        return [bid for _, bid in ranked]
