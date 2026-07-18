"""DenseCompactScreen: flag blocks by a configured BlockMetric. Cheap pass = the metric's
vectorized `proxy` over free kblock columns, pre-filtering to the top `proxy_keep_pct`% by proxy
(peel metrics only); fine pass = build only survivors (reusing the source's KblockSource paths),
score them with the metric's `fine` (using the real peel depth when `needs_peel`), and keep those
the `gate` selects. The fine-pass depth goes through reblock.derivations.access_before (a
derive() call), so building a survivor here is an L1 hit when run() later scores it. A metric
with `needs_peel=False` (pure geometry/density) skips the peel entirely -- fine is scored
straight from the cheap columns.

The whole selection is itself memoized: select() routes through derivations.screen_selection
(a derive() keyed on the source content hash + metric + gate + pre-filter), so a rerun with the
same source, metric, and gate returns the ranked block_ids from one L2 lookup -- seconds (a
content hash + a lookup), not the minutes the fine pass takes to walk thousands of survivor
blocks.
"""
from __future__ import annotations

import logging
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd

from reblock.contracts import Source
from reblock.data.kblock import KblockSource
from reblock.derivations import ScreenSelectionInput, access_before, screen_selection
from reblock.derive_graph import source_hash
from reblock.metric import BlockMetric, Gate

log = logging.getLogger(__name__)

_FINE_PASS_THRESHOLD = 32   # below this many survivors, fork/IPC overhead isn't worth it -> serial


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


def _chunk_depths(
    args: tuple[str, str, int, list[str]],
) -> list[tuple[str, float, float]]:
    """Build a chunk of survivor blocks and return `(block_id, max_depth, mean_depth)` for each.
    Module-level (not a closure) so a fork `ProcessPoolExecutor` can dispatch it. Reads the parquet
    once for the whole chunk (`block_ids` windows the read), amortizing per-block I/O; each block's
    Voronoi tessellation is local, so a chunked build is identical to building all at once."""
    blocks_path, buildings_path, min_buildings, block_ids = args
    src = KblockSource(blocks_path, buildings_path, region_id="screen",
                       min_buildings=min_buildings, block_ids=block_ids)
    out: list[tuple[str, float, float]] = []
    for blk in src.region().blocks:
        d = access_before(blk)
        out.append((str(blk.block_id), float(d.max()), float(d.mean())))
    return out


def _survivor_depths(
    survivors: list[str], blocks_path: str, buildings_path: str, min_buildings: int,
) -> list[tuple[str, float, float]]:
    """`(block_id, max_depth, mean_depth)` for every survivor. Each block's Voronoi+peel+access is
    independent, so fork a process pool across survivor chunks (mirrors `arterial`'s fork-pool
    pattern) with a serial fallback -- `< _FINE_PASS_THRESHOLD` survivors, `workers <= 1`, or no
    `fork` start method. Result order is irrelevant (the caller sorts). access_before still memoizes
    per block inside each worker, so a later rerun is a cache hit."""
    workers = min(16, max(1, (os.cpu_count() or 2) - 1))
    use_pool = (workers > 1 and len(survivors) >= _FINE_PASS_THRESHOLD
                and "fork" in multiprocessing.get_all_start_methods())
    if not use_pool:
        return _chunk_depths((blocks_path, buildings_path, min_buildings, survivors))
    chunks = [survivors[i::workers] for i in range(workers)]   # round-robin -> even load
    args = [(blocks_path, buildings_path, min_buildings, c) for c in chunks if c]
    log.info("fine pass: %d survivors across %d fork workers", len(survivors), len(args))
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("fork")) as ex:
        return [row for chunk in ex.map(_chunk_depths, args) for row in chunk]


def _compute_selection(inp: ScreenSelectionInput) -> list[tuple[str, float]]:
    """Full screen under the configured metric: proxy over all blocks -> recall pre-filter (peel
    metrics only) -> (peel survivors iff metric.needs_peel) -> metric.fine -> gate -> ranked
    (block_id, fine_score) highest-first. Memoized via screen_selection.identity (metric + gate)."""
    metric, gate = inp.metric, inp.gate
    blocks = gpd.read_parquet(
        inp.blocks_path,
        columns=["block_id", "building_count", "block_area_m2", "geometry"])
    bid = blocks["block_id"].astype(str).to_numpy()
    count = blocks["building_count"].to_numpy(dtype=float)
    utm = blocks.to_crs(blocks.estimate_utm_crs())
    area = (blocks["block_area_m2"].to_numpy(dtype=float) if "block_area_m2" in blocks.columns
            else utm.geometry.area.to_numpy())
    perim = utm.geometry.length.to_numpy()
    eligible = count >= inp.min_buildings
    proxy = metric.proxy(blocks).to_numpy()

    if not metric.needs_peel:
        scores = {str(bid[i]): metric.fine(0.0, count[i], area[i], perim[i])
                  for i in range(len(bid)) if eligible[i] and np.isfinite(proxy[i])}
    else:
        # recall pre-filter: keep the top proxy_keep_pct% by proxy among eligible blocks, then peel.
        order = [i for i in np.argsort(proxy)[::-1] if eligible[i] and np.isfinite(proxy[i])]
        k = max(1, math.ceil(len(order) * inp.proxy_keep_pct / 100.0))
        survivors = [str(bid[i]) for i in order[:k]]
        idx = {b: i for i, b in enumerate(bid)}
        depth_by = {b: mx for b, mx, _ in                                   # {bid: max_depth}
                    _survivor_depths(survivors, inp.blocks_path, inp.buildings_path,
                                     inp.min_buildings)}
        scores = {b: metric.fine(depth_by.get(b, 0.0), count[idx[b]], area[idx[b]], perim[idx[b]])
                  for b in survivors}

    kept = gate.keep(scores)
    ranked = sorted(((scores[b], b) for b in kept), key=lambda r: (-r[0], r[1]))
    log.info("screen: %d/%d blocks selected by metric=%s (needs_peel=%s)",
             len(ranked), len(bid), metric.name, metric.needs_peel)
    return [(b, s) for s, b in ranked]


class DenseCompactScreen:
    def __init__(self, metric: BlockMetric, gate: Gate, *, proxy_keep_pct: float = 30.0,
                 min_buildings: int = 10) -> None:
        self.metric = metric
        self.gate = gate
        self.proxy_keep_pct = proxy_keep_pct
        self.min_buildings = min_buildings

    def _selection_input(self, source: Source) -> ScreenSelectionInput:
        if not isinstance(source, KblockSource):
            raise TypeError(
                f"DenseCompactScreen needs a KblockSource (kblock columns); "
                f"got {type(source).__name__}")
        return ScreenSelectionInput(
            source_hash=source_hash(source.blocks_path, source.buildings_path),
            blocks_path=str(source.blocks_path), buildings_path=str(source.buildings_path),
            metric=self.metric, gate=self.gate, proxy_keep_pct=self.proxy_keep_pct,
            min_buildings=self.min_buildings)

    def select(self, source: Source) -> list[str]:
        return [bid for bid, _ in screen_selection(self._selection_input(source))]

    def selection_scores(self, source: Source) -> dict[str, float]:
        """block_id -> the metric's fine score for the flagged blocks (memoized screen_selection
        lookup) -- what region_map's coloring keys on."""
        return dict(screen_selection(self._selection_input(source)))
