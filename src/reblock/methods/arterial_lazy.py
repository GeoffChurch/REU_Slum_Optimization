"""CELF / lazy-greedy engine + candidate policies for GreedyArterialReblocker. Reuses arterial's
exact scoring machinery unchanged; only changes which candidates get scored each step."""
from __future__ import annotations

import heapq
import multiprocessing
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from geopandas import GeoDataFrame
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

import reblock.methods.arterial as _art
from reblock.budget import _BlockScoringContext, access_burden
from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.arterial import (
    _PARALLEL_THRESHOLD,
    _anchor_points,
    _candidate_chords,
    _deep_targets,
    _explode,
    _merge,
    _snap_graph,
    _StepState,
    _xy,
    eval_candidate,
)
from reblock.methods.dijkstra import _boundary_graph, _rnd


def _road_vertices(road: LineString) -> list[tuple[float, float]]:
    return [_rnd(_xy(c)) for c in road.coords]


def _committed_gdf(committed: list[LineString], block: Block) -> GeoDataFrame | None:
    import geopandas as gpd
    return gpd.GeoDataFrame(geometry=list(committed), crs=block.crs) if committed else None


@dataclass
class _FixedPolicy:
    _initial: list[LineString]

    def initial(self) -> list[LineString]:
        return self._initial

    def after_commit(self, committed: list[LineString], step: int
                     ) -> tuple[list[LineString], list[str]]:
        return [], []


@dataclass
class _GrowPolicy:
    block: Block
    adj: list[set[int]]
    anchors: list[tuple[float, float]]         # accumulates committed-road vertices
    top_k: int
    seen: set[str]                              # wkt of every candidate ever emitted
    _initial: list[LineString]

    def initial(self) -> list[LineString]:
        return self._initial

    def after_commit(self, committed: list[LineString], step: int
                     ) -> tuple[list[LineString], list[str]]:
        for v in _road_vertices(committed[-1]):
            if v not in self.anchors:
                self.anchors.append(v)
        self.anchors.sort()
        targets = _deep_targets(self.block, _committed_gdf(committed, self.block),
                                self.top_k, self.adj)
        cands = _candidate_chords(self.anchors, targets)
        added = [ls for ls in cands if ls.wkt not in self.seen]
        for ls in added:
            self.seen.add(ls.wkt)
        return added, []


@dataclass
class _FaithfulPolicy:
    block: Block
    streets: list[BaseGeometry]
    n_anchors: int
    adj: list[set[int]]
    top_k: int
    live: set[str]
    _initial: list[LineString]

    def initial(self) -> list[LineString]:
        return self._initial

    def after_commit(self, committed: list[LineString], step: int
                     ) -> tuple[list[LineString], list[str]]:
        network = [*self.streets, *committed]
        cands = _candidate_chords(
            _anchor_points(network, self.n_anchors),
            _deep_targets(self.block, _committed_gdf(committed, self.block), self.top_k, self.adj))
        now = {ls.wkt: ls for ls in cands}
        added = [ls for k, ls in now.items() if k not in self.live]
        removed = [k for k in self.live if k not in now]
        self.live = set(now.keys())
        return added, removed


def _make_policy(name: str, block: Block, streets: Sequence[BaseGeometry],
                 n_anchors: int, top_k: int, adj: list[set[int]]
                 ) -> _FixedPolicy | _GrowPolicy | _FaithfulPolicy:
    anchors0 = _anchor_points(list(streets), n_anchors)
    targets0 = _deep_targets(block, None, top_k, adj)
    initial = _candidate_chords(anchors0, targets0)
    if name == "fixed":
        return _FixedPolicy(initial)
    if name == "grow":
        return _GrowPolicy(block, adj, list(anchors0), top_k, {ls.wkt for ls in initial}, initial)
    if name == "faithful":
        return _FaithfulPolicy(block, list(streets), n_anchors, adj, top_k,
                               {ls.wkt for ls in initial}, initial)
    raise ValueError(f"unknown candidate_policy {name!r}")


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


def _greedy_arterials_lazy(block: Block, *, mode: str, objective: str, n_anchors: int = 32,
                           top_k: int = 8, lam: float = 2.0, max_roads: int = 15,
                           cost: str = "length", corridor_m: float = 3.0, workers: int = 16,
                           candidate_policy: str = "grow", rescore_every: int = 0) -> GeoDataFrame:
    """CELF lazy-greedy driver: commit the best gain-per-cost arterial one at a time, but instead of
    re-scoring every candidate every step (the exact `_greedy_arterials`), drive selection with a
    max-heap and pop-re-score only the heap top until it is fresh under the current committed set.
    Reuses arterial's EXACT scoring machinery unchanged (via `eval_candidate` + `_art._STEP_STATE`),
    so with `rescore_every=1` + the `faithful` policy it is byte-identical to the exact greedy."""
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    base_burden = access_burden(parcel_access_layers(
        block, None, tol=STREET_TOL, adj=adj, unreached_depth=len(block.parcels) + 1))
    sg = _snap_graph(_boundary_graph(block.parcels))
    streets = list(block.streets.geometry)
    ctx = _BlockScoringContext(block) if objective in ("efficiency", "directness") else None
    policy = _make_policy(candidate_policy, block, streets, n_anchors, top_k, adj)

    committed: list[LineString] = []
    real_of: dict[str, BaseGeometry] = {}          # wkt(chord) -> realized geometry (snap-stable)
    # (-gain, real.wkt, chord.wkt, chord, scored_at_step). Ordered by (-gain, real_wkt, key) to
    # match `_best_candidate`'s tie-break on the REALIZED geometry's wkt exactly (in buildable mode
    # `real = _snap(chord, sg, lam)` is the boundary-graph path, whose wkt differs from the chord's
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
        base = _explode(base_merged, block.crs)
        base_val = _art._score(objective, block, base, adj, base_burden, ctx)
        committed_disp = 0
        if cost == "displacement":
            from reblock.budget import displacement_count
            committed_disp = displacement_count(block.building_points, base, corridor_m)
        stepctx = ctx.step(base) if (ctx is not None and mode == "buildable") else None
        assert _art._STEP_STATE is None, "eval_candidate's per-step state holder is not reentrant"
        _art._STEP_STATE = _StepState(
            step=stepctx, sg=sg, base_val=base_val, base_merged=base_merged, committed=committed,
            mode=mode, objective=objective, cost=cost, lam=lam, corridor_m=corridor_m,
            committed_disp=committed_disp, block=block, crs=block.crs, adj=adj,
            base_burden=base_burden, ctx=ctx)
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
            _art._STEP_STATE = None
        winner = real_of[key]                   # commit the realized geometry
        assert isinstance(winner, LineString)   # always a LineString (chord / snapped path)
        committed.append(winner)
        live.discard(key)
        added, removed_keys = policy.after_commit(committed, len(committed))
        for k in removed_keys:
            live.discard(k)
        pending = added

    return _explode(_merge(committed), block.crs)
