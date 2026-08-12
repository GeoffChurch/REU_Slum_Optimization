"""Does tier 2 make the access method affordable at REGION scale? That was the entire point.

Baseline for this exact block: the `depth` example variant runs in 1,115 s WITHOUT
`greedy_arterial_access_displacement` and had not finished after 41,700 s (11.6 h) with it. The
projection from measured per-candidate cost was 29.6 h on 16 workers.

Two things also get measured here because they set the floor once the peel is gone:

  * `_candidate_chords` -- 469k chords built and WKT-sorted per step, and the anchor set GROWS as
    committed roads add network vertices, so this is quadratic across steps and tier 2 does not
    touch it.
  * the bulk ranking itself -- 255 s serial per step at 469k chords (`rank_throughput.py`). The
    chunk loop is chunk-independent, so a thread pool should scale it IF shapely releases the GIL
    inside `STRtree.query`. That is an empirical question about shapely, not about this code, so it
    is measured rather than assumed.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from shapely import STRtree

from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.arterial import SnapToBoundary
from reblock.methods.arterial.engines import _greedy_shortlist
from reblock.methods.arterial.primitives import _anchor_points, _candidate_chords, _deep_targets
from reblock.methods.arterial.shortlist import CHUNK, RANK_RADIUS, FirstOrder
from reblock.permeability import DEFAULT_ROAD_WIDTH_M
from scripts.perf.snap_vs_peel import region_block_cached

SHORTLIST = 512
MAX_ROADS = 15
THREADS = 8            # measured optimum -- see shortlist.first_order_score
SCALING = False        # the thread-scaling table is already measured and recorded; skip the re-run


def _threaded_rank(chords: list, weights: np.ndarray, ptree: STRtree, btree: STRtree,
                   half_w: float, threads: int) -> np.ndarray:
    """The same chunk loop as `rank_candidates`, spread over a thread pool."""
    los = list(range(0, len(chords), CHUNK))
    out = np.zeros(len(chords))

    def work(lo: int) -> tuple[int, np.ndarray]:
        arr = np.asarray(chords[lo:lo + CHUNK], dtype=object)
        src, tgt = ptree.query(arr, predicate="dwithin", distance=RANK_RADIUS)
        gain = np.bincount(src, weights=weights[tgt], minlength=len(arr))
        bsrc, _ = btree.query(arr, predicate="dwithin", distance=half_w)
        nb = np.bincount(bsrc, minlength=len(arr)).astype(float)
        return lo, gain / np.maximum(nb, 1.0)

    with ThreadPoolExecutor(max_workers=threads) as ex:
        for lo, vals in ex.map(work, los):
            out[lo:lo + len(vals)] = vals
    return out


def main() -> int:
    block = region_block_cached()
    n = len(block.parcels)
    half_w = DEFAULT_ROAD_WIDTH_M / 2.0
    print(f"\nregion block: {n:,} parcels, {len(block.building_points):,} buildings\n")

    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    ptree = STRtree(list(block.parcels.geometry))
    btree = STRtree(list(block.building_points.geometry))
    depths = parcel_access_layers(block, None, tol=STREET_TOL, adj=adj, unreached_depth=n + 1)
    weights = depths.loc[block.parcels["parcel_id"]].to_numpy(dtype=float) ** 2 - 1.0
    anchors = _anchor_points(list(block.streets.geometry), 32, 0)
    targets = _deep_targets(block, None, 8, adj)
    chords = _candidate_chords(anchors, targets)
    print(f"  step-0 candidates: {len(chords):,}\n")

    if SCALING:
        print("  RANK SCALING (does shapely release the GIL inside STRtree.query?)")
        base = None
        for threads in (1, 4, 8, 16):
            t0 = time.perf_counter()
            _threaded_rank(chords, weights, ptree, btree, half_w, threads)
            dt = time.perf_counter() - t0
            base = base or dt
            print(f"    {threads:>2} threads: {dt:7.1f} s   {base / dt:5.2f}x", flush=True)

    print(f"\n  FULL RUN, shortlist={SHORTLIST}, max_roads={MAX_ROADS}, threads={THREADS}")
    # PER-STEP progress, because the first attempt at this run was killed at 73 minutes having
    # printed nothing at all -- the totals only come at the end, so a kill destroyed every minute of
    # it. Now each step lands on disk as it completes and a kill costs one step, not the run.
    t0 = time.perf_counter()
    last = [t0]

    def tick(step: int, n_cand: int, n_roads: int) -> None:
        now = time.perf_counter()
        print(f"    step {step:>2}: {n_cand:>7,} cand  {now - last[0]:6.1f} s  "
              f"(total {(now - t0) / 60:5.1f} min, {n_roads} roads)", flush=True)
        last[0] = now

    roads = _greedy_shortlist(block, realizer=SnapToBoundary(), objective="access",
                              cost="displacement",
                              half_width_m=half_w, workers=16, max_roads=MAX_ROADS,
                              selector=FirstOrder(SHORTLIST, threads=THREADS), on_step=tick)
    dt = time.perf_counter() - t0
    print(f"    {len(roads)} road rows, {float(roads.geometry.length.sum()):,.0f} m "
          f"in {dt / 60:.1f} min")
    print(f"\n    vs 11.6 h without finishing, and a 29.6 h projection -> "
          f"{29.6 * 3600 / dt:,.0f}x the projection")
    print(f"    vs the 1,115 s the whole `depth` variant costs WITHOUT this method: "
          f"{dt / 1115:.1f}x that budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
