"""Where does the access-objective greedy actually spend its time at REGION scale?

The backlog entry "Making the access objective affordable at region scale" records two suspects for
the >=37x penalty, and says to settle which one dominates BEFORE building tier 2:

  * the PEEL -- `_score("access", ...)` runs a full BFS over all 11,006 parcels to score one local
    road. Tier 2 (first-order local gain) attacks exactly this.
  * `_snap`  -- a Dijkstra over the region's parcel-boundary graph, once per candidate. Tier 2 does
    NOT touch this. If snapping dominates, the shortlist has to be formed BEFORE snapping (rank the
    unsnapped chord, snap only the survivors), which is a different change.

The evidence pointing at `_snap` is indirect: a `max_anchors=24` run (~276 candidates/step) should
have taken ~6 minutes at 85 ms/peel and had not finished after 66. That says per-candidate cost is
larger than the peel alone, not which term is larger.

So: rebuild step 0 of `_greedy_arterials` on the real region block and time the two calls separately
over the same candidates. No estimates, no extrapolation from block scale -- the peel and the
Dijkstra scale differently in the parcel count, which is why block-scale timings mislead here.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import cast

import numpy as np

from reblock.budget import access_burden
from reblock.contracts import Block, Screen, Source
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.arterial.primitives import (
    _anchor_points,
    _candidate_chords,
    _deep_targets,
    _explode,
    _snap_graph,
)
from reblock.methods.arterial.realize import _snap
from reblock.methods.arterial.scoring import _score
from reblock.methods.boundary_graph import _boundary_graph

CACHE = Path("scratchpad/perf/region_block.pkl")
N_SAMPLE = 60          # candidates to time; each is one Dijkstra + one 11k-parcel BFS
LAM = 2.0
HALF_W = 3.0


def region_block_cached() -> Block:
    """The `multiblock_depth` region as ONE Block -- the same object `region_reblock` hands the
    method. Pickled, because growing it costs minutes and this script gets re-run."""
    if CACHE.exists():
        with CACHE.open("rb") as fh:
            return cast(Block, pickle.load(fh))
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from reblock.pipeline import build_regions
    from reblock.region import RegionBuilder, region_block

    overrides = ["metric=depth", "data=capetown_full", "screen=dense_compact",
                 "region_builder=dense_cluster", "region_builder.max_buildings=3000",
                 "max_blocks=1"]
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config", overrides=overrides)
    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    rb = cast(RegionBuilder, instantiate(cfg.region_builder))
    t0 = time.perf_counter()
    region = build_regions(source, screen, rb, None, 1)[0]
    blk = region_block(region)
    print(f"  built region: {len(region)} blocks, {len(blk.parcels):,} parcels "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("wb") as fh:
        pickle.dump(blk, fh)
    return blk


def main() -> None:
    block = region_block_cached()
    n = len(block.parcels)
    print(f"\nregion block: {n:,} parcels, {len(block.streets)} street rows\n")

    t0 = time.perf_counter()
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    t_adj = time.perf_counter() - t0

    t0 = time.perf_counter()
    g = _boundary_graph(block.parcels)
    sg = _snap_graph(g)
    t_graph = time.perf_counter() - t0

    t0 = time.perf_counter()
    base_burden = access_burden(parcel_access_layers(
        block, None, tol=STREET_TOL, adj=adj, unreached_depth=n + 1))
    t_peel0 = time.perf_counter() - t0

    print(f"  ONCE PER BLOCK      parcel_adjacency {t_adj:8.2f} s")
    print(f"                      boundary graph   {t_graph:8.2f} s  "
          f"({g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges)")
    print(f"                      base peel        {t_peel0:8.2f} s  (burden {base_burden:.3f})\n")

    # --- step 0 exactly as `_greedy_arterials` sets it up (committed empty) ---
    t0 = time.perf_counter()
    anchors = _anchor_points(list(block.streets.geometry), 32, 0)
    targets = _deep_targets(block, None, 8, adj)
    candidates = _candidate_chords(anchors, targets)
    t_cand = time.perf_counter() - t0
    print(f"  STEP 0              {len(anchors):,} anchors -> {len(candidates):,} candidates "
          f"({t_cand:.1f} s to enumerate)\n")

    idx = np.linspace(0, len(candidates) - 1, N_SAMPLE).astype(int)
    sample = [candidates[i] for i in idx]

    snap_t, peel_t, reals = [], [], []
    for chord in sample:
        t0 = time.perf_counter()
        real = _snap(chord, sg, LAM)
        snap_t.append(time.perf_counter() - t0)
        reals.append(real)
    for real in reals:
        if real is None or real.length == 0:
            continue
        from reblock.methods.arterial.primitives import _union_with
        trial = _explode(_union_with(None, real), block.crs, 2.0 * HALF_W)
        t0 = time.perf_counter()
        _score("access", block, trial, adj, base_burden, None)
        peel_t.append(time.perf_counter() - t0)

    s, p = np.array(snap_t), np.array(peel_t)
    tot = s.mean() + p.mean()
    print(f"  PER CANDIDATE ({len(s)} sampled, {len(p)} snapped to a real road)\n")
    print(f"    {'':16}{'mean':>10}{'median':>10}{'max':>10}{'share':>9}")
    print(f"    {'_snap':16}{s.mean() * 1e3:>9.1f}m{np.median(s) * 1e3:>9.1f}m"
          f"{s.max() * 1e3:>9.1f}m{s.mean() / tot:>8.0%}")
    print(f"    {'peel (_score)':16}{p.mean() * 1e3:>9.1f}m{np.median(p) * 1e3:>9.1f}m"
          f"{p.max() * 1e3:>9.1f}m{p.mean() / tot:>8.0%}")
    print(f"    {'total':16}{tot * 1e3:>9.1f}m\n")

    step_s = tot * len(candidates)
    print(f"  ONE STEP at {len(candidates):,} candidates: {step_s:,.0f} s serial "
          f"= {step_s / 16 / 60:,.0f} min on 16 workers")
    print(f"  15 STEPS:                       {step_s * 15 / 16 / 3600:,.1f} h on 16 workers\n")
    print(f"  If tier 2 made the peel FREE:   {s.mean() * len(candidates) * 15 / 16 / 3600:,.1f} h")
    print(f"  If snapping were made free:     {p.mean() * len(candidates) * 15 / 16 / 3600:,.1f} h")


if __name__ == "__main__":
    main()
