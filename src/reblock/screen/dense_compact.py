"""DenseCompactScreen: flag dense/compact informal blocks. Cheap pass = vectorized
density (+ optional k) gate over free kblock columns; fine pass = build only survivors
(reusing the source's KblockSource paths) and keep those whose mean parcel access-depth
clears mean_depth_min. The fine-pass depth goes through reblock.derivations.access_before
(a derive() call), so building a survivor here is an L1 hit when run() later scores it.
"""
from __future__ import annotations

import logging
import time

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
        t0 = time.perf_counter()
        blocks = gpd.read_parquet(
            source.blocks_path,
            columns=["block_id", "k_complexity", "building_count", "block_area_m2", "geometry"])
        survivors = self._cheap_survivors(blocks)
        log.info("cheap pass: %d/%d blocks pass density_min=%.1f%s (%.1f%%, %.1fs)",
                 len(survivors), len(blocks), self.density_min,
                 f", k_min={self.k_min}" if self.k_min is not None else "",
                 100.0 * len(survivors) / len(blocks), time.perf_counter() - t0)
        if not survivors:
            return []
        n_surv = len(survivors)
        log.info("fine pass: building %d survivor blocks (Voronoi + peel) -- the slow step", n_surv)
        src = KblockSource(source.blocks_path, source.buildings_path, region_id="screen",
                           min_buildings=self.min_buildings, block_ids=survivors)
        # One access-depth series per block; keep those clearing the mean gate (and the
        # optional max gate), ranked deepest-parcel-first so a downstream max_blocks picks
        # the worst-access blocks rather than an alphabetical slice. Count per-gate drops
        # and log coarse progress (~10 lines) so the slow fine pass isn't silent.
        t1 = time.perf_counter()
        ranked: list[tuple[float, str]] = []
        dropped_mean = dropped_max = 0
        step = max(1, n_surv // 10)
        for i, blk in enumerate(src.region().blocks, 1):
            depths = access_before(blk)
            mean_d, max_d = float(depths.mean()), float(depths.max())
            if mean_d < self.mean_depth_min:
                dropped_mean += 1
            elif self.max_depth_min is not None and max_d < self.max_depth_min:
                dropped_max += 1
            else:
                ranked.append((max_d, blk.block_id))
            if i % step == 0 or i == n_surv:
                log.info("fine pass: built %d/%d (%d kept so far)", i, n_surv, len(ranked))
        ranked.sort(key=lambda r: (-r[0], r[1]))   # max-depth desc; ties by block_id asc
        drops = f"{dropped_mean} on mean<{self.mean_depth_min:.2f}"
        if self.max_depth_min is not None:
            drops += f", {dropped_max} on max<{self.max_depth_min:.1f}"
        log.info("fine pass: kept %d/%d in %.1fs (dropped %s), ranked by max access-depth",
                 len(ranked), n_surv, time.perf_counter() - t1, drops)
        if ranked:
            log.info("fine pass: kept blocks span max access-depth %.0f (deepest) .. %.0f",
                     ranked[0][0], ranked[-1][0])
        return [bid for _, bid in ranked]
