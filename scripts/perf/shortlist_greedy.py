"""Tier 2 as an actual greedy: reduce each step's candidates with an injected selector.

Mirrors `_greedy_arterials` step for step and changes exactly one thing -- which candidates reach
`eval_candidate` -- in the same spirit as the lazy engine, `_greedy_arterials_lazy` ("reuses
arterial's exact scoring machinery unchanged; only changes which candidates get scored each step").
The per-step state, the scorer, the fork pool and the `_best_candidate` reduce are all the shipped
ones, so any difference in the output comes from the shortlist and nothing else.

## Why the ranking is judged end-to-end and not per-step

The obvious acceptance test -- "does the shortlist contain the candidate the exact greedy picked?"
-- is the wrong test here, and measurably so. Three results together:

  * the first-order estimate tracks the exact benefit at Spearman **+0.95** (`rank_decompose.py`),
    so the estimate is not the problem;
  * yet the exact winner lands around the median of the estimate's order
    (`first_order_rank.py`), i.e. the argmax is not recoverable from a +0.95 ranking;
  * and the exact greedy's own argmax flips under a **1e-10** perturbation of the gains, moving
    burden reduction by up to 13 points (notes/2026-08-09-greedy-arterial-is-near-tie-sensitive.md).

The third explains the second. The candidate gains are densely near-tied, so "the exact winner" is
not a stable target -- it is one arbitrary draw, and no approximation can reproduce an arbitrary
draw. Demanding that the shortlist recover it would hold this method to a standard the method does
not meet against itself.

What CAN be asked is whether the shortlist run lands in the same place. The reference band is
already measured: 1e-10 gain perturbations move burden reduction by a median of 0.0000 and a max of
0.1356 across seeds. A shortlist whose spread sits inside that band is doing no more damage than the
tie-breaking already does.

## The ranking

    score = (sum of (d^2 - 1) over parcels the CHORD fronts) / (buildings within its corridor)

Numerator: one peel per step, then a bulk `dwithin` over the parcel tree.
Denominator: a second bulk `dwithin` over the building tree. Chosen by measurement, not taste --
against the exact `displacement` it correlates **+0.88** where the chord's own length manages only
+0.65, and it costs the same single query.
"""
from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from geopandas import GeoDataFrame
from shapely import STRtree
from shapely.geometry import LineString

import reblock.methods.arterial.scoring as art
from reblock.budget import (
    _BlockScoringContext,
    access_burden,
    building_radii,
    corridor_distance,
    displacement,
    road_drainage,
)
from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.arterial.primitives import (
    _anchor_points,
    _candidate_chords,
    _deep_targets,
    _explode,
    _merge,
    _planarize,
    _snap_graph,
)
from reblock.methods.arterial.scoring import (
    _PARALLEL_THRESHOLD,
    _score,
    _StepState,
    eval_candidate,
)
from reblock.methods.boundary_graph import _boundary_graph
from scripts.perf.selectors import CandidateSelector, RankContext, ScoreAll


def greedy_shortlist(block: Block, *, mode: str, objective: str, n_anchors: int = 32,
                     top_k: int = 8, lam: float = 2.0, max_roads: int = 15,
                     cost: str = "length", half_width_m: float, workers: int = 16,
                     max_anchors: int = 0, selector: CandidateSelector | None = None,
                     on_step: Callable[[int, int, int], None] | None = None) -> GeoDataFrame:
    """`_greedy_arterials` with the step's candidate list reduced by an injected `selector`.

    `selector=None` (or `ScoreAll()`) scores everything, which IS the exact greedy -- verified
    bit-identical to `_greedy_arterials` on real blocks by `control_check.py`, and the control arm
    of every comparison here. Every other selector changes which candidates get scored and nothing
    else: the per-step state, `eval_candidate`, the fork pool and the `_best_candidate` reduce are
    all the shipped ones.

    `on_step(step, n_candidates, n_committed)` fires after each commit. A region run takes over an
    hour and reports nothing until it returns, so without this a kill destroys the whole run's
    evidence -- which is exactly what happened to the first attempt, at 73 minutes.
    """
    selector = selector if selector is not None else ScoreAll()
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    base_burden = access_burden(parcel_access_layers(
        block, None, tol=STREET_TOL, adj=adj, unreached_depth=len(block.parcels) + 1))
    sg = _snap_graph(_boundary_graph(block.parcels))
    streets = list(block.streets.geometry)
    ctx = (_BlockScoringContext(block) if objective in ("efficiency", "directness") else None)
    radii = building_radii(block.building_points)
    parcel_tree = STRtree(list(block.parcels.geometry))
    building_tree = STRtree(list(block.building_points.geometry))
    ids = block.parcels["parcel_id"]

    committed: list[LineString] = []
    needs_depths = not isinstance(selector, ScoreAll)
    while len(committed) < max_roads:
        network = [*streets, *committed]
        anchors = _anchor_points(network, n_anchors, max_anchors)
        base_merged = _merge(committed)
        base = _explode(base_merged, block.crs, 2.0 * half_width_m)
        base_val = _score(objective, block, base, adj, base_burden, ctx)
        curr_roads = base if len(committed) else None
        targets = _deep_targets(block, curr_roads, top_k, adj)
        committed_dist = building_xy = None
        if cost == "displacement_fast":
            building_xy = np.asarray(list(block.building_points.geometry), dtype=object)
            committed_dist = corridor_distance(block.building_points, base) if len(base) else None
        committed_disp = (displacement(block.building_points, radii, base)
                          if cost == "displacement" else 0.0)
        step = ctx.step(base) if (ctx is not None and mode == "buildable") else None

        candidates = _candidate_chords(anchors, targets)
        n_cand = len(candidates)

        # --- the one difference from `_greedy_arterials` ---
        if needs_depths:
            # ONE peel per step (not per candidate) gives every parcel's depth under what is
            # already committed. `.loc[ids]` puts it in the positional order `parcel_tree` indexes.
            # Computed even for selectors that ignore it (RandomSample), so the null model pays the
            # same per-step overhead as the ranking it is being compared against -- otherwise the
            # timing comparison would flatter it for a reason unrelated to selection quality.
            depths = parcel_access_layers(block, curr_roads, tol=STREET_TOL, adj=adj,
                                          unreached_depth=len(block.parcels) + 1)
            candidates = selector.select(candidates, RankContext(
                depths=depths.loc[ids].to_numpy(dtype=float), parcel_tree=parcel_tree,
                building_tree=building_tree, half_width_m=half_width_m, step=len(committed)))
        # --- everything below is the shipped path, unmodified ---

        use_pool = (workers > 1 and len(candidates) >= _PARALLEL_THRESHOLD
                    and "fork" in multiprocessing.get_all_start_methods())
        assert art._STEP_STATE is None, "per-step state holder is not reentrant"
        art._STEP_STATE = _StepState(
            step=step, sg=sg, base_val=base_val, base_merged=base_merged, committed=committed,
            mode=mode, objective=objective, cost=cost, lam=lam, half_width_m=half_width_m,
            committed_disp=committed_disp, committed_dist=committed_dist,
            building_xy=building_xy, block=block, radii=radii,
            crs=block.crs, adj=adj, base_burden=base_burden, ctx=ctx)
        try:
            if use_pool:
                with ProcessPoolExecutor(max_workers=workers,
                                         mp_context=multiprocessing.get_context("fork")) as ex:
                    results = list(ex.map(eval_candidate, candidates,
                                          chunksize=max(1, len(candidates) // (workers * 4))))
            else:
                results = [eval_candidate(chord) for chord in candidates]
        finally:
            art._STEP_STATE = None
        _, best_real = art._best_candidate(results)
        if best_real is None:
            break
        assert isinstance(best_real, LineString)
        committed.append(best_real)
        if on_step is not None:
            on_step(len(committed), n_cand, len(committed))

    roads = _planarize(committed, block.crs, 2.0 * half_width_m)
    roads["drain"] = road_drainage(block, roads) if len(roads) else []
    return roads


__all__ = ["greedy_shortlist"]
