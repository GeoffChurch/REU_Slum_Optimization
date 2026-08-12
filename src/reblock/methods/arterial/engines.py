"""Search engines for GreedyArterialReblocker: `_greedy_arterials` is the exact greedy (every
candidate re-scored every step, optionally fork-pooled), and `_greedy_arterials_lazy` is the CELF /
lazy-greedy driver (pop-and-rescore only the heap top, per a pluggable `policies` candidate set).
Both funnel every candidate through `scoring.eval_candidate` against the SAME frozen per-step
state -- the lazy engine reuses the exact engine's scoring machinery unchanged; only which
candidates get scored each step differs.
"""
from __future__ import annotations

import heapq
import multiprocessing
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
from geopandas import GeoDataFrame
from shapely import STRtree
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

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
from reblock.methods.arterial import scoring
from reblock.methods.arterial.policies import CandidatePolicySpec, Grow
from reblock.methods.arterial.primitives import (
    _anchor_points,
    _candidate_chords,
    _deep_targets,
    _explode,
    _merge,
    _planarize,
    _snap_graph,
)
from reblock.methods.arterial.realize import ChordRealizer
from reblock.methods.arterial.scoring import (
    _PARALLEL_THRESHOLD,
    _best_candidate,
    _score,
    _StepState,
)

# Explicit self-alias (mypy's --no-implicit-reexport convention, see reblock.compare's
# `compare_report as compare_report`): tests reach `engines.eval_candidate` directly (as an
# instrumentable seam, same spirit as `_PARALLEL_THRESHOLD`'s test-monkeypatch design), which
# --strict would otherwise flag as accessing a name this module merely imports, not exports.
from reblock.methods.arterial.scoring import eval_candidate as eval_candidate
from reblock.methods.arterial.shortlist import CandidateSelector, FirstOrder, RankContext
from reblock.methods.boundary_graph import _boundary_graph


def _greedy_arterials(block: Block, *, realizer: ChordRealizer, objective: str, n_anchors: int = 32,
                      top_k: int = 8, max_roads: int = 15,
                      cost: str = "length", half_width_m: float,
                      workers: int = 16, max_anchors: int = 0) -> GeoDataFrame:
    """Greedily commit the straight arterial with the best objective gain per unit cost until
    `max_roads` are placed or no candidate improves. `realizer` turns each candidate chord into the
    road that is actually scored and committed (see `ChordRealizer`).
    `cost` in {"length" (Delta-benefit/metre), "displacement" (Delta-benefit/expected buildings
    newly displaced within `half_width_m` of any committed road, via the extent-aware disk
    `budget.displacement` over `block.building_points` -- see `budget.building_radii`), "repulsion"
    (Delta-benefit per the road's OWN quadratic-tail proximity to the building field via
    `budget.repulsion` -- a constant-per-candidate, never-zero soft cost, so CELF-safe and never
    degenerate)}. A
    beneficial candidate whose denominator is zero (a zero-length road can't occur -- filtered
    below -- but a zero-marginal-displacement road can) ranks ABOVE every positive-denominator
    candidate (infinite gain) rather than being divided by zero or skipped: take the free
    navigability first.

    `workers` parallelizes each step's candidate scoring across a fork process pool (byte-identical
    to serial). `workers <= 1`, a step with `< _PARALLEL_THRESHOLD` candidates, or a platform
    without `fork` all take the literal serial path (a true no-op vs the pool, not a 1-worker
    pool).

    `max_anchors` is a CAP, not a mode switch: 0 means uncapped (byte-identical to every network
    vertex + arc-length samples); a positive value only ever REDUCES the anchor count below that
    uncapped set, never inflates it, falling back to ~`max_anchors` arc-length samples when the
    uncapped family does not already fit -- see `_anchor_points`."""
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    base_burden = access_burden(parcel_access_layers(
        block, None, tol=STREET_TOL, adj=adj, unreached_depth=len(block.parcels) + 1))
    g = _boundary_graph(block.parcels)
    sg = _snap_graph(g)                    # precomputed once per block -- see `_snap`
    # Raw street geometries (may be a MultiLineString for a holed/courtyard block) -- do NOT
    # filter to LineString, or Multi* streets get dropped and the proposal comes back empty;
    # _anchor_points explodes Multi* internally.
    streets: list[BaseGeometry] = list(block.streets.geometry)
    # ONE scoring context per block (frozen reps/sources/src_euclid/street geometry), shared by
    # every candidate score below -- only built for the metric objectives that use it.
    ctx = (_BlockScoringContext(block) if objective in ("efficiency", "directness") else None)
    # Constant across every step (depends only on block.building_points), so computed ONCE here
    # rather than per-step.
    radii = building_radii(block.building_points)

    committed: list[LineString] = []                        # realized geometry, in commit order
    while len(committed) < max_roads:
        network: list[BaseGeometry] = [*streets, *committed]
        anchors = _anchor_points(network, n_anchors, max_anchors)
        base_merged = _merge(committed)              # unary_union(committed), once per step
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
        # Route per-candidate scoring by realizer. BUILDABLE trials are boundary-snapped (they join
        # the committed/street network at shared graph vertices), so the incremental
        # `step.score_candidate` is bit-exact to `_score(objective, _planarize(committed+[real]))`
        # while skipping the per-candidate full entry re-derivation -- the perf win. Likewise, a
        # buildable trial's noding is bit-exact under `_union_with`'s incremental `unary_union`
        # (§4 -- meets the network only at shared vertices), so the `access`-objective buildable
        # path (no `ctx`, so no `step`) uses that incremental union too. ASPIRATIONAL trials are
        # free chords crossing committed edges at float interior points, where the incremental
        # planarize noding is NOT bit-exact (design "Bug 2"), so those always use the full
        # `_planarize(committed + [real])` re-union below (still frozen-constants fast via `ctx`
        # when scored, for efficiency/directness).
        step = ctx.step(base) if (ctx is not None and realizer.snaps) else None

        # Evaluate every candidate against the frozen per-step state, via the module-level holder
        # (COW-inheritable by the fork pool). Both the serial and pool paths funnel through the SAME
        # holder set / `finally`-clear and the SAME `_best_candidate` reduce; the pool only replaces
        # the comprehension with a fork `map`. Fall to serial when a pool isn't worth it or fork is
        # unavailable: `workers <= 1` (a true no-op vs today, NOT a 1-worker pool), a small step
        # (`< _PARALLEL_THRESHOLD` candidates -- spawn/IPC overhead dominates), or no `fork` start
        # method (never pickle the CSR/graph context per task).
        candidates = _candidate_chords(anchors, targets)
        use_pool = (workers > 1 and len(candidates) >= _PARALLEL_THRESHOLD
                    and "fork" in multiprocessing.get_all_start_methods())
        # `_STEP_STATE` lives in `scoring` (the fork pool's children run `scoring.eval_candidate`,
        # which reads it as ITS OWN module global) -- every reference here MUST go through the
        # `scoring` module object, never a `from ... import _STEP_STATE` binding, or workers would
        # see `None` regardless of what this loop sets.
        assert scoring._STEP_STATE is None, (
            "eval_candidate's per-step state holder is not reentrant")
        scoring._STEP_STATE = _StepState(
            step=step, sg=sg, base_val=base_val, base_merged=base_merged, committed=committed,
            realizer=realizer, objective=objective, cost=cost, half_width_m=half_width_m,
            committed_disp=committed_disp, committed_dist=committed_dist,
            building_xy=building_xy, block=block, radii=radii,
            crs=block.crs, adj=adj, base_burden=base_burden, ctx=ctx)
        try:
            if use_pool:
                # Explicit fork context so children inherit `_STEP_STATE` via COW; `map` preserves
                # input order (chunksize/worker-count-independent) so the reduce sees the serial
                # sequence and the result is byte-identical to serial.
                with ProcessPoolExecutor(max_workers=workers,
                                         mp_context=multiprocessing.get_context("fork")) as ex:
                    results = list(ex.map(eval_candidate, candidates,
                                          chunksize=max(1, len(candidates) // (workers * 4))))
            else:
                results = [eval_candidate(chord) for chord in candidates]
        finally:
            scoring._STEP_STATE = None
        _, best_real = _best_candidate(results)
        if best_real is None:                               # no candidate improves -> stop
            break
        # `eval_candidate`'s `real` is always a LineString (the chord itself for aspirational, or
        # `_snap`'s boundary-graph path for buildable) -- `_best_candidate` is typed over the wider
        # `BaseGeometry` so it stays reusable for any future non-LineString candidate shape.
        assert isinstance(best_real, LineString)
        committed.append(best_real)

    roads = _planarize(committed, block.crs, 2.0 * half_width_m)
    roads["drain"] = road_drainage(block, roads) if len(roads) else []
    return roads


def _score_all(chords: list[LineString], use_pool: bool, workers: int
               ) -> list[tuple[float, BaseGeometry | None]]:
    """Return list[(gain, real)] via the same `eval_candidate` the exact path uses -- pooled over a
    fork process pool when it is worth it (workers>1, at least `_PARALLEL_THRESHOLD` chords, fork
    available; byte-identical to serial because `map` preserves order), else serial."""
    if use_pool and len(chords) >= _PARALLEL_THRESHOLD and \
            "fork" in multiprocessing.get_all_start_methods():
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("fork")) as ex:
            return list(ex.map(eval_candidate, chords,
                               chunksize=max(1, len(chords) // (workers * 4))))
    return [eval_candidate(c) for c in chords]


def _iter_live(heap: list[tuple[float, str, str, LineString, int]], live: set[str]
               ) -> Iterator[LineString]:
    """Distinct live chords currently in the heap (for a full re-score rebuild)."""
    seen: set[str] = set()
    for _neg, _real_wkt, key, chord, _at in heap:
        if key in live and key not in seen:
            seen.add(key)
            yield chord


def _greedy_arterials_lazy(block: Block, *, realizer: ChordRealizer, objective: str,
                           n_anchors: int = 32,
                           top_k: int = 8, max_roads: int = 15,
                           cost: str = "length", half_width_m: float,
                           workers: int = 16, policy_spec: CandidatePolicySpec,
                           rescore_every: int = 0, max_anchors: int = 0) -> GeoDataFrame:
    """CELF lazy-greedy driver: commit the best gain-per-cost arterial one at a time, but instead of
    re-scoring every candidate every step (the exact `_greedy_arterials`), drive selection with a
    max-heap and pop-re-score only the heap top until it is fresh under the current committed set.
    Reuses arterial's EXACT scoring machinery unchanged (via `eval_candidate` +
    `scoring._STEP_STATE`), so with `rescore_every=1` + the `Faithful` policy spec it is
    byte-identical to the exact greedy."""
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    base_burden = access_burden(parcel_access_layers(
        block, None, tol=STREET_TOL, adj=adj, unreached_depth=len(block.parcels) + 1))
    sg = _snap_graph(_boundary_graph(block.parcels))
    streets = list(block.streets.geometry)
    ctx = _BlockScoringContext(block) if objective in ("efficiency", "directness") else None
    policy = policy_spec.build(block, streets, n_anchors, top_k, adj, max_anchors)
    # Constant across every step (depends only on block.building_points), so computed ONCE here
    # rather than per-step.
    radii = building_radii(block.building_points)

    committed: list[LineString] = []
    real_of: dict[str, BaseGeometry] = {}          # wkt(chord) -> realized geometry (snap-stable)
    # (-gain, real.wkt, chord.wkt, chord, scored_at_step). Ordered by (-gain, real_wkt, key) to
    # match `_best_candidate`'s tie-break on the REALIZED geometry's wkt exactly (a snapping
    # realizer's `real` is the boundary-graph path, whose wkt differs from the chord's
    # -- ordering by chord.wkt instead would resolve equal-gain ties differently than exact and
    # break the `rescore_every=1` + `faithful` byte-identity oracle). `key` (== chord.wkt) is kept
    # as the THIRD element -- after real_wkt so ties order correctly, before the raw `chord` so
    # comparison never falls through to unorderable `LineString`s -- and remains the identity used
    # by `live`, `real_of`, and the policies' `removed_keys` (policies emit chord.wkt keys).
    heap: list[tuple[float, str, str, LineString, int]] = []
    live: set[str] = set()
    pending = policy.initial()
    use_pool = workers > 1

    while len(committed) < max_roads:
        step = len(committed)
        base_merged = _merge(committed)
        base = _explode(base_merged, block.crs, 2.0 * half_width_m)
        base_val = _score(objective, block, base, adj, base_burden, ctx)
        committed_disp = 0.0
        committed_dist = building_xy = None
        if cost in ("displacement", "displacement_fast"):
            committed_disp = displacement(block.building_points, radii, base)
        if cost == "displacement_fast":
            building_xy = np.asarray(list(block.building_points.geometry), dtype=object)
            committed_dist = corridor_distance(block.building_points, base) if len(base) else None
        stepctx = ctx.step(base) if (ctx is not None and realizer.snaps) else None
        assert scoring._STEP_STATE is None, (
            "eval_candidate's per-step state holder is not reentrant")
        scoring._STEP_STATE = _StepState(
            step=stepctx, sg=sg, base_val=base_val, base_merged=base_merged, committed=committed,
            realizer=realizer, objective=objective, cost=cost, half_width_m=half_width_m,
            committed_disp=committed_disp, committed_dist=committed_dist,
            building_xy=building_xy, block=block, radii=radii,
            crs=block.crs, adj=adj, base_burden=base_burden, ctx=ctx)
        try:
            # eager-score candidates entering this step
            if rescore_every and step > 0 and step % rescore_every == 0:
                # Full re-score of everything live: drop the stale heap and re-score every live
                # chord AND the just-added candidates from this step's `after_commit` (still in
                # `pending`) so the pool exactly matches the exact greedy's regenerated set. Keeping
                # `pending` here is load-bearing -- overwriting it drops the newly-added candidates.
                pending = [*_iter_live(heap, live), *pending]
                heap = []
            for chord, (gain, real) in zip(pending, _score_all(pending, use_pool, workers),
                                           strict=True):
                if real is None:
                    continue
                key = chord.wkt
                real_of[key] = real
                live.add(key)
                heapq.heappush(heap, (-gain, real.wkt, key, chord, step))
            pending = []
            # pop-and-re-score the top until it is fresh under this committed set
            while heap:
                neg, real_wkt, key, chord, at = heap[0]
                if key not in live:                # committed/removed since it was pushed -- drop
                    heapq.heappop(heap)
                    continue
                if at == step:                     # top is fresh under this committed set -> winner
                    break
                heapq.heappop(heap)
                gain, real = eval_candidate(chord)
                if real is None:
                    live.discard(key)
                    continue
                real_of[key] = real
                heapq.heappush(heap, (-gain, real.wkt, key, chord, step))
            if not heap or -heap[0][0] <= 0.0:
                break
            neg, real_wkt, key, chord, at = heapq.heappop(heap)
        finally:
            scoring._STEP_STATE = None
        winner = real_of[key]                   # commit the realized geometry
        assert isinstance(winner, LineString)   # always a LineString (chord / snapped path)
        committed.append(winner)
        live.discard(key)
        added, removed_keys = policy.after_commit(committed, len(committed))
        for k in removed_keys:
            live.discard(k)
        pending = added

    roads = _planarize(committed, block.crs, 2.0 * half_width_m)
    roads["drain"] = road_drainage(block, roads) if len(roads) else []
    return roads


def _greedy_shortlist(block: Block, *, realizer: ChordRealizer, objective: str,
                      selector: CandidateSelector, n_anchors: int = 32,
                      top_k: int = 8, max_roads: int = 15,
                      cost: str = "length", half_width_m: float,
                      workers: int = 16, max_anchors: int = 0,
                      on_step: Callable[[int, int, int], None] | None = None) -> GeoDataFrame:
    """`_greedy_arterials` with the step's candidate list reduced by an injected `selector` --
    tier 2. Mirrors `_greedy_arterials` step for step and changes exactly one thing: which
    candidates reach `eval_candidate`. The per-step state, the scorer, the fork pool and the
    `_best_candidate` reduce are all the SAME ones `_greedy_arterials` uses, so any difference in
    the output comes from the shortlist and nothing else -- verified by
    `test_shortlist_with_non_binding_k_is_the_exact_engine`, which sets `selector` to a `FirstOrder`
    wide enough to never bind and checks the two engines produce byte-identical roads.

    `on_step(step, n_candidates, n_committed)` fires after each commit. A region run can take over
    an hour and reports nothing until it returns, so without this a kill destroys the whole run's
    evidence -- see docs/superpowers/notes/2026-08-11-max-anchors-is-a-region-scale-win.md.
    """
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    base_burden = access_burden(parcel_access_layers(
        block, None, tol=STREET_TOL, adj=adj, unreached_depth=len(block.parcels) + 1))
    g = _boundary_graph(block.parcels)
    sg = _snap_graph(g)
    # Raw street geometries (may be a MultiLineString for a holed/courtyard block) -- do NOT
    # filter to LineString, or Multi* streets get dropped and the proposal comes back empty;
    # _anchor_points explodes Multi* internally.
    streets: list[BaseGeometry] = list(block.streets.geometry)
    ctx = (_BlockScoringContext(block) if objective in ("efficiency", "directness") else None)
    radii = building_radii(block.building_points)
    # The two trees the ranking queries against -- built once per block, like `_snap_graph` above.
    parcel_tree = STRtree(list(block.parcels.geometry))
    building_tree = STRtree(list(block.building_points.geometry))
    ids = block.parcels["parcel_id"]

    committed: list[LineString] = []
    while len(committed) < max_roads:
        network: list[BaseGeometry] = [*streets, *committed]
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
        step = ctx.step(base) if (ctx is not None and realizer.snaps) else None

        # --- the one difference from `_greedy_arterials`: reduce the candidate list ---
        candidates = _candidate_chords(anchors, targets)
        n_cand = len(candidates)                    # pre-shortlist -- what `on_step` reports
        # ONE peel per step (not per candidate) gives every parcel's depth under what is already
        # committed. `.loc[ids]` puts it in the positional order `parcel_tree` indexes. Computed
        # for every selector, including one that ignores it, so a timing comparison between
        # selectors is never flattered by one of them skipping work the others pay for.
        depths = parcel_access_layers(block, curr_roads, tol=STREET_TOL, adj=adj,
                                      unreached_depth=len(block.parcels) + 1)
        candidates = selector.select(candidates, RankContext(
            depths=depths.loc[ids].to_numpy(dtype=float), parcel_tree=parcel_tree,
            building_tree=building_tree, half_width_m=half_width_m, step=len(committed)))
        # --- everything below is the shipped path, unmodified ---

        use_pool = (workers > 1 and len(candidates) >= _PARALLEL_THRESHOLD
                    and "fork" in multiprocessing.get_all_start_methods())
        # See `_greedy_arterials`' matching comment: `_STEP_STATE` lives in `scoring` and must be
        # written/read through the qualified module object, never a rebound local import.
        assert scoring._STEP_STATE is None, (
            "eval_candidate's per-step state holder is not reentrant")
        scoring._STEP_STATE = _StepState(
            step=step, sg=sg, base_val=base_val, base_merged=base_merged, committed=committed,
            realizer=realizer, objective=objective, cost=cost, half_width_m=half_width_m,
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
            scoring._STEP_STATE = None
        _, best_real = _best_candidate(results)
        if best_real is None:                               # no candidate improves -> stop
            break
        assert isinstance(best_real, LineString)
        committed.append(best_real)
        if on_step is not None:
            on_step(len(committed), n_cand, len(committed))

    roads = _planarize(committed, block.crs, 2.0 * half_width_m)
    roads["drain"] = road_drainage(block, roads) if len(roads) else []
    return roads


@runtime_checkable
class ArterialEngine(Protocol):
    """How the greedy searches: which candidates get scored exactly, each step.

    Injected rather than picked by three co-dependent flags. Every engine reuses the same scoring
    machinery (`eval_candidate`, `_STEP_STATE`, the fork pool) -- only candidate selection differs.
    """

    @property
    def identity(self) -> EngineIdentity: ...

    def run(self, block: Block, *, objective: str, cost: str, realizer: ChordRealizer,
            n_anchors: int, top_k: int, max_roads: int, half_width_m: float,
            workers: int, max_anchors: int) -> GeoDataFrame: ...


@dataclass(frozen=True)
class ExactEngine:
    """Score every candidate, every step. The reference path every other engine is checked
    against."""

    @property
    def identity(self) -> EngineIdentity:
        return self

    def run(self, block: Block, *, objective: str, cost: str, realizer: ChordRealizer,
            n_anchors: int, top_k: int, max_roads: int, half_width_m: float,
            workers: int, max_anchors: int) -> GeoDataFrame:
        return _greedy_arterials(
            block, objective=objective, cost=cost, realizer=realizer, n_anchors=n_anchors,
            top_k=top_k, max_roads=max_roads, half_width_m=half_width_m, workers=workers,
            max_anchors=max_anchors)


@dataclass(frozen=True)
class LazyEngine:
    """CELF lazy-greedy: drive selection with a max-heap of stale upper bounds.

    VALID ONLY FOR SUBMODULAR OBJECTIVES. That holds for directness; it does NOT hold for
    access-burden reduction, where it was measured diverging from the exact greedy on 6 of 6 blocks
    and SLOWER in 4 of 6 -- an approximation for no speed. Use ShortlistEngine for access.
    """

    policy: CandidatePolicySpec = Grow()
    rescore_every: int = 0          # 0 = pure lazy; N = full re-score every N commits

    @property
    def identity(self) -> EngineIdentity:
        # `policy` and `rescore_every` both affect the proposal -- no field to strip, so `self`
        # (embedding the raw `policy` spec) is the identity outright, same as ExactEngine. `Grow`/
        # `Fixed`/`Faithful` are frozen, zero-field dataclasses, so a policy IS its own identity
        # too; there is deliberately no `CandidatePolicySpec.identity` seam to route through here.
        return self

    def run(self, block: Block, *, objective: str, cost: str, realizer: ChordRealizer,
            n_anchors: int, top_k: int, max_roads: int, half_width_m: float,
            workers: int, max_anchors: int) -> GeoDataFrame:
        return _greedy_arterials_lazy(
            block, objective=objective, cost=cost, realizer=realizer, n_anchors=n_anchors,
            top_k=top_k, max_roads=max_roads, half_width_m=half_width_m, workers=workers,
            policy_spec=self.policy, rescore_every=self.rescore_every, max_anchors=max_anchors)


@dataclass(frozen=True)
class ShortlistEngine:
    """Tier 2: rank every candidate by a cheap first-order estimate, score only the top `k` exactly.

    Needed because CELF is invalid for the access objective (not submodular). Ranks by
    (sum of d^2-1 over parcels the chord fronts) / (buildings in its corridor) -- chosen by
    measurement: against exact displacement it correlates +0.92 where chord length manages +0.65,
    for the same single bulk `dwithin`.

    `k=512` is the value every region result was measured at. Saturation bounds it only from ABOVE:
    512/1024/2048/4096 produce a bit-identical network, so overshooting is free and the unmeasured
    direction is downward. `threads=8` is the measured optimum (354.9 s at 1, 104.3 s at 8, and
    134.0 s at 16 -- the STRtree query is memory-bandwidth bound). At block scale threads is a
    no-op by construction: a few thousand candidates is one chunk.
    """

    k: int = 512
    threads: int = 8

    @property
    def identity(self) -> EngineIdentity:
        return ShortlistIdentity(k=self.k)      # threads cannot change the roads

    def run(self, block: Block, *, objective: str, cost: str, realizer: ChordRealizer,
            n_anchors: int, top_k: int, max_roads: int, half_width_m: float,
            workers: int, max_anchors: int) -> GeoDataFrame:
        return _greedy_shortlist(
            block, objective=objective, cost=cost, realizer=realizer, n_anchors=n_anchors,
            top_k=top_k, max_roads=max_roads, half_width_m=half_width_m, workers=workers,
            max_anchors=max_anchors, selector=FirstOrder(self.k, threads=self.threads))


@dataclass(frozen=True)
class ShortlistIdentity:
    """ShortlistEngine's proposal-affecting part. `threads` is excluded deliberately."""

    k: int


EngineIdentity: TypeAlias = ExactEngine | LazyEngine | ShortlistIdentity
