"""DenseCompactScreen: flag dense/compact informal blocks. Cheap pass = vectorized
density (+ optional k) gate over free kblock columns; fine pass = build only survivors
(reusing the source's KblockSource paths) and keep those whose mean parcel access-depth
clears mean_depth_min. The fine-pass depth goes through reblock.derivations.access_before
(a derive() call), so building a survivor here is an L1 hit when run() later scores it.

The whole selection is itself memoized: select() routes through derivations.screen_selection
(a derive() keyed on the source content hash + gate params), so a rerun with the same source
and gates returns the ranked block_ids from one L2 lookup -- seconds (a content hash + a
lookup), not the minutes the fine pass takes to walk thousands of survivor blocks.
"""
from __future__ import annotations

import logging
import time

import geopandas as gpd
import numpy as np
import pandas as pd

from reblock.contracts import Source
from reblock.data.kblock import KblockSource
from reblock.derivations import ScreenSelectionInput, access_before, screen_selection
from reblock.derive_graph import source_hash

log = logging.getLogger(__name__)


def _depth_proxy(blocks: gpd.GeoDataFrame) -> pd.Series:
    """Cheap per-block estimate of parcel access depth (rings from a street), from the FREE kblock
    columns + block outline: ``sqrt(n * A) / P`` where n=building_count, A=block_area_m2, P=block
    perimeter (metres). Derivation: max ring depth ~ inradius / parcel-width; inradius ~ 2A/P
    (hydraulic radius), parcel-width ~ sqrt(A/n), so the ratio ~ 2*sqrt(nA)/P (the constant drops
    out of a threshold). On real Cape Town blocks this ranks true access depth ~5x better than
    building density (Spearman 0.76 vs 0.15 on max depth): deep nesting is frontage-starvation,
    which the perimeter P captures and a density n/A cannot. It equals the free closed form of
    "how many parcel-widths is the deepest parcel from egress" (an explicit per-parcel Euclidean
    distance-to-egress pass gives the same ranking at much higher cost)."""
    n = blocks["building_count"].to_numpy(dtype=float)
    A = blocks["block_area_m2"].to_numpy(dtype=float)
    perim = blocks.to_crs(blocks.estimate_utm_crs()).geometry.length.to_numpy()
    return pd.Series(np.sqrt(n * A) / np.where(perim > 0, perim, np.nan), index=blocks.index)


def _cheap_survivors(blocks: gpd.GeoDataFrame, *, depth_proxy_min: float,
                     k_min: float | None) -> list[str]:
    bid = blocks["block_id"].astype(str)
    mask: pd.Series = _depth_proxy(blocks) >= depth_proxy_min
    if k_min is not None:
        mask = mask & (blocks["k_complexity"] >= k_min)
    return sorted(bid[mask.to_numpy()])


def _compute_selection(inp: ScreenSelectionInput) -> list[str]:
    """The full screen (cheap density prune + fine access-depth pass), ranked deepest-first.
    Run via derivations.screen_selection's derive() so its (source-hash + gates)-keyed result
    is memoized; the per-survivor depth here also goes through the cached access_before."""
    t0 = time.perf_counter()
    blocks = gpd.read_parquet(
        inp.blocks_path,
        columns=["block_id", "k_complexity", "building_count", "block_area_m2", "geometry"])
    survivors = _cheap_survivors(blocks, depth_proxy_min=inp.depth_proxy_min, k_min=inp.k_min)
    log.info("cheap pass: %d/%d blocks pass depth_proxy_min=%.2f%s (%.1f%%, %.1fs)",
             len(survivors), len(blocks), inp.depth_proxy_min,
             f", k_min={inp.k_min}" if inp.k_min is not None else "",
             100.0 * len(survivors) / len(blocks), time.perf_counter() - t0)
    if not survivors:
        return []
    n_surv = len(survivors)
    log.info("fine pass: building %d survivor blocks (Voronoi + peel) -- the slow step", n_surv)
    src = KblockSource(inp.blocks_path, inp.buildings_path, region_id="screen",
                       min_buildings=inp.min_buildings, block_ids=survivors)
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
        if mean_d < inp.mean_depth_min:
            dropped_mean += 1
        elif inp.max_depth_min is not None and max_d < inp.max_depth_min:
            dropped_max += 1
        else:
            ranked.append((max_d, blk.block_id))
        if i % step == 0 or i == n_surv:
            log.info("fine pass: built %d/%d (%d kept so far)", i, n_surv, len(ranked))
    ranked.sort(key=lambda r: (-r[0], r[1]))   # max-depth desc; ties by block_id asc
    drops = f"{dropped_mean} on mean<{inp.mean_depth_min:.2f}"
    if inp.max_depth_min is not None:
        drops += f", {dropped_max} on max<{inp.max_depth_min:.1f}"
    log.info("fine pass: kept %d/%d in %.1fs (dropped %s), ranked by max access-depth",
             len(ranked), n_surv, time.perf_counter() - t1, drops)
    if ranked:
        log.info("fine pass: kept blocks span max access-depth %.0f (deepest) .. %.0f",
                 ranked[0][0], ranked[-1][0])
    return [bid for _, bid in ranked]


class DenseCompactScreen:
    def __init__(self, *, depth_proxy_min: float = 1.5, mean_depth_min: float = 1.3,
                 max_depth_min: float | None = None, k_min: float | None = None,
                 min_buildings: int = 10) -> None:
        self.depth_proxy_min = depth_proxy_min
        self.mean_depth_min = mean_depth_min
        self.max_depth_min = max_depth_min
        self.k_min = k_min
        self.min_buildings = min_buildings

    def select(self, source: Source) -> list[str]:
        if not isinstance(source, KblockSource):
            raise TypeError(
                f"DenseCompactScreen needs a KblockSource (kblock columns); "
                f"got {type(source).__name__}")
        inp = ScreenSelectionInput(
            source_hash=source_hash(source.blocks_path, source.buildings_path),
            blocks_path=str(source.blocks_path), buildings_path=str(source.buildings_path),
            depth_proxy_min=self.depth_proxy_min, mean_depth_min=self.mean_depth_min,
            max_depth_min=self.max_depth_min, k_min=self.k_min,
            min_buildings=self.min_buildings)
        return screen_selection(inp)
