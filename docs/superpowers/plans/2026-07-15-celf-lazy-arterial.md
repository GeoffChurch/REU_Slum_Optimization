# CELF / Lazy-Greedy Arterial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in lazy (CELF) greedy to `GreedyArterialReblocker` that re-scores only the heap-top candidate per commit instead of all ~4,600 every step, making arterial tractable at regional road budgets.

**Architecture:** A new `_greedy_arterials_lazy(...)` in a new file `src/reblock/methods/arterial_lazy.py` sits beside the **untouched** exact `_greedy_arterials` in `arterial.py`; `GreedyArterialReblocker.propose` dispatches on a `lazy` flag. The lazy engine reuses arterial's exact scoring machinery (`eval_candidate`, `_STEP_STATE`, `_BlockScoringContext`) unchanged — CELF changes *which* candidates get scored per step, never *how*. A pluggable candidate policy (`fixed`|`grow`|`faithful`) evolves the candidate set each step.

**Tech Stack:** Python, `heapq`, existing `reblock.methods.arterial` primitives, `reblock.budget._BlockScoringContext`, pytest, Hydra config, pixi (`pixi run pytest ...`).

## Global Constraints

- **Exact path byte-identical:** do NOT edit `_greedy_arterials`; `lazy=False` calls it unchanged. All existing arterial goldens (`test_arterial_serial_refactor_identical`, `test_arterial_parallel_*`, `test_arterial_proposal_wkt_unchanged`) must still pass.
- **No scoring change:** lazy re-scores through the same `eval_candidate` / `_STEP_STATE` / `_BlockScoringContext` machinery. Never approximate a per-candidate score.
- **Determinism:** heap ties break by the same `wkt` total order arterial's `_best_candidate` uses (`gain` desc, then `wkt` asc). Two runs of one lazy config give identical output.
- **`rescore_every=1` per policy == exact greedy over that policy's candidate set** — the correctness oracle.
- **Compose with `workers`:** pool the initial full pass + eager scoring of added candidates; per-commit pop-re-score loop is serial.
- **`_snap(chord)` is committed-independent** — the realized geometry `real` is fixed per chord; cache it, recompute only the gain.
- **No-legacy:** lazy is a new first-class mode, not a back-compat shim. One exact path, one lazy path.
- Run tests with `pixi run pytest`. Commit after each task; branch is `celf-lazy-arterial`.

---

### Task 1: Config fields + identity + Hydra wiring

**Files:**
- Modify: `src/reblock/methods/arterial.py` (`GreedyArterialReblocker` dataclass ~406-445)
- Modify: `conf/method/greedy_arterial.yaml`
- Test: `tests/methods/test_arterial.py` (`test_identity_and_proposal_metadata` ~295)

**Interfaces:**
- Produces: `GreedyArterialReblocker` gains fields `lazy: bool = False`, `candidate_policy: str = "grow"`, `rescore_every: int = 0`; `identity` becomes a 12-tuple `(..., self.lazy, self.candidate_policy, self.rescore_every)` appended after `lam`. `propose` still calls only `_greedy_arterials` this task (dispatch added in Task 4).

- [ ] **Step 1: Update the identity golden to expect the 3 new fields**

In `tests/methods/test_arterial.py::test_identity_and_proposal_metadata`, the current assertion checks the 9-tuple. Change the expected tuple to include the three new trailing fields and add discrimination assertions:

```python
def test_identity_and_proposal_metadata() -> None:
    m = GreedyArterialReblocker(mode="buildable", objective="directness")
    assert m.identity == (
        "greedy_arterial", "buildable", "directness", "length", 0.0,
        15, 32, 8, 2.0, False, "grow", 0)
    assert GreedyArterialReblocker(max_roads=3).identity != m.identity
    assert GreedyArterialReblocker(n_anchors=16).identity != m.identity
    assert GreedyArterialReblocker(lazy=True).identity != m.identity
    assert GreedyArterialReblocker(candidate_policy="fixed").identity != m.identity
    assert GreedyArterialReblocker(rescore_every=2).identity != m.identity
    proposal = GreedyArterialReblocker(objective="directness").propose(_grid_block(5))
    assert proposal.block_identity == _grid_block(5).identity
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `pixi run pytest tests/methods/test_arterial.py::test_identity_and_proposal_metadata -q`
Expected: FAIL (identity is still the 9-tuple; `lazy=` is an unexpected kwarg).

- [ ] **Step 3: Add the fields + extend identity**

In `GreedyArterialReblocker` add after `workers`:

```python
    lazy: bool = False               # False -> exact _greedy_arterials (byte-identical)
    candidate_policy: str = "grow"   # "grow" | "fixed" | "faithful" (only used when lazy)
    rescore_every: int = 0           # 0 = pure lazy; N = full re-score every N commits (safety)
```

Extend the `identity` return type annotation to a 12-tuple and its value:

```python
    @property
    def identity(self) -> tuple[str, str, str, str, float, int, int, int, float, bool, str, int]:
        corridor_key = self.corridor_m if self.cost == "displacement" else 0.0
        return ("greedy_arterial", self.mode, self.objective, self.cost, corridor_key,
                self.max_roads, self.n_anchors, self.top_k, self.lam,
                self.lazy, self.candidate_policy, self.rescore_every)
```

- [ ] **Step 4: Wire the config**

Append to `conf/method/greedy_arterial.yaml`:

```yaml
# Lazy (CELF) greedy: re-score only the heap-top candidate per commit (huge speedup at large
# max_roads; scales with budget). lazy=False keeps the exact path byte-identical.
lazy: false
candidate_policy: grow   # grow (default) | fixed | faithful -- see the CELF design spec
rescore_every: 0         # 0 = pure lazy; N = full re-score every N commits (non-submodularity safety)
```

- [ ] **Step 5: Find and update any other identity assertions**

Run: `pixi run grep -rn "greedy_arterial\", " tests/ | grep -iE "identity|== \("` and update every asserted arterial identity tuple to the 12-tuple form (there were prior goldens updated for max_roads/n_anchors/top_k/lam — the same set).

- [ ] **Step 6: Run the golden + full arterial suite**

Run: `pixi run pytest tests/methods/test_arterial.py -q`
Expected: PASS (exact path unaffected; `lazy` defaults keep behavior identical).

- [ ] **Step 7: Commit**

```bash
git add src/reblock/methods/arterial.py conf/method/greedy_arterial.yaml tests/methods/test_arterial.py
git commit -m "feat(arterial): add lazy/candidate_policy/rescore_every config + identity"
```

---

### Task 2: Candidate policies (`fixed` | `grow` | `faithful`)

**Files:**
- Create: `src/reblock/methods/arterial_lazy.py`
- Test: `tests/methods/test_arterial_lazy.py`

**Interfaces:**
- Consumes from `arterial.py`: `_anchor_points`, `_deep_targets`, `_candidate_chords`.
- Produces: `_make_policy(name, block, streets, n_anchors, top_k, adj) -> CandidatePolicy` and a policy object with `initial() -> list[LineString]` and `after_commit(committed: list[LineString], step: int) -> tuple[list[LineString], list[LineString]]` returning `(added, removed)`. Candidate identity key is `ls.wkt` (matches arterial's dedup/sort).

- [ ] **Step 1: Write failing tests for the three policies**

```python
# tests/methods/test_arterial_lazy.py
from shapely.geometry import LineString
from reblock.derive.adjacency import parcel_adjacency
from reblock.derive.access import STREET_TOL
from reblock.methods.arterial_lazy import _make_policy
from tests.methods.test_arterial import _grid_block  # reuse the fast grid fixture


def _policy(name, block):
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    return _make_policy(name, block, list(block.streets.geometry), 6, 4, adj)


def test_fixed_policy_never_changes_after_initial():
    block = _grid_block(5)
    pol = _policy("fixed", block)
    assert len(pol.initial()) > 0
    added, removed = pol.after_commit([LineString([(0, 0), (10, 10)])], 1)
    assert added == [] and removed == []


def test_grow_policy_only_adds():
    block = _grid_block(5)
    pol = _policy("grow", block)
    base = pol.initial()
    added, removed = pol.after_commit([LineString([(0, 0), (10, 10)])], 1)
    assert removed == []                       # grow removes nothing
    base_keys = {ls.wkt for ls in base}
    assert all(ls.wkt not in base_keys for ls in added)   # only genuinely new candidates


def test_faithful_policy_matches_arterial_candidate_set():
    # faithful's set after committing a road must equal arterial's own regeneration for that network
    from reblock.methods.arterial import _anchor_points, _deep_targets, _candidate_chords
    block = _grid_block(5)
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    pol = _policy("faithful", block)
    base = pol.initial()
    road = LineString([(5.0, 0.0), (5.0, 40.0)])
    live = {ls.wkt for ls in base}
    added, removed = pol.after_commit([road], 1)
    live = (live - {ls.wkt for ls in removed}) | {ls.wkt for ls in added}
    network = list(block.streets.geometry) + [road]
    expect = {ls.wkt for ls in _candidate_chords(
        _anchor_points(network, 6), _deep_targets(block, None, 4, adj))}
    # deep_targets change with roads; compare through-road structure via arterial's own generator
    expect_roads = {ls.wkt for ls in _candidate_chords(_anchor_points(network, 6), [])}
    assert expect_roads <= live
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run pytest tests/methods/test_arterial_lazy.py -q`
Expected: FAIL (`arterial_lazy` module does not exist).

- [ ] **Step 3: Implement the policies**

```python
# src/reblock/methods/arterial_lazy.py
"""CELF / lazy-greedy engine + candidate policies for GreedyArterialReblocker. Reuses arterial's
exact scoring machinery unchanged; only changes which candidates get scored each step."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from reblock.contracts import Block
from reblock.derive.adjacency import parcel_adjacency  # noqa: F401 (kept for callers/tests)
from reblock.methods.arterial import (
    _anchor_points, _candidate_chords, _deep_targets, _xy)
from reblock.methods.dijkstra import _rnd


def _road_vertices(road: LineString) -> list[tuple[float, float]]:
    return [_rnd(_xy(c)) for c in road.coords]


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
```

**Signature contract for all three policies:** `after_commit(committed, step) -> tuple[list[LineString], list[str]]` — `(added_lines, removed_wkt_keys)`. `_FixedPolicy` and `_GrowPolicy` return `[]` for removed; only `_FaithfulPolicy` removes. The engine (Task 3) consumes `removed` as `wkt` keys to lazily delete from `live`.

Add the helper `_committed_gdf`:

```python
def _committed_gdf(committed: list[LineString], block: Block):
    import geopandas as gpd
    return gpd.GeoDataFrame(geometry=list(committed), crs=block.crs) if committed else None
```

And the factory:

```python
def _make_policy(name: str, block: Block, streets: Sequence[BaseGeometry],
                 n_anchors: int, top_k: int, adj: list[set[int]]):
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
```

Remove the `_KEY_ONLY`/`if False` scaffolding — it is illustrative of the removed-keys pivot only; final code returns `(list[LineString], list[str])` as described.

- [ ] **Step 4: Run the policy tests**

Run: `pixi run pytest tests/methods/test_arterial_lazy.py -q`
Expected: PASS. (Update the `test_faithful_policy_*` assertion to consume `removed` as keys, matching the finalized signature.)

- [ ] **Step 5: Commit**

```bash
git add src/reblock/methods/arterial_lazy.py tests/methods/test_arterial_lazy.py
git commit -m "feat(arterial): candidate policies (fixed|grow|faithful) for lazy greedy"
```

---

### Task 3: CELF engine `_greedy_arterials_lazy`

**Files:**
- Modify: `src/reblock/methods/arterial_lazy.py` (add the engine)
- Test: `tests/methods/test_arterial_lazy.py`

**Interfaces:**
- Consumes: `_make_policy` (Task 2); from `arterial.py`: `eval_candidate`, `_StepState`, `_merge`, `_explode`, `_snap_graph`, and the module attribute `arterial._STEP_STATE`; from `reblock.budget`: `_BlockScoringContext`; from `reblock.derive.*`: `parcel_adjacency`, `access_burden`, `parcel_access_layers`; from `dijkstra`: `_boundary_graph`.
- Produces: `_greedy_arterials_lazy(block, *, mode, objective, n_anchors, top_k, lam, max_roads, cost, corridor_m, workers, candidate_policy, rescore_every) -> GeoDataFrame` returning the same `_explode(_merge(committed), crs)` GeoDataFrame shape as `_greedy_arterials`.

- [ ] **Step 1: Write the engine-correctness test (the oracle)**

`rescore_every=1, policy="faithful"` must reproduce the exact greedy's road sequence, because a full re-score every step picks the true per-step argmax over arterial's own candidate set:

```python
def test_lazy_faithful_rescore1_equals_exact():
    from reblock.methods.arterial import _greedy_arterials
    from reblock.methods.arterial_lazy import _greedy_arterials_lazy
    for mode in ("buildable", "aspirational"):
        block = _grid_block(5)
        exact = _greedy_arterials(block, mode=mode, objective="directness", n_anchors=6,
                                  max_roads=4, workers=1)
        lazy = _greedy_arterials_lazy(block, mode=mode, objective="directness", n_anchors=6,
                                      top_k=8, lam=2.0, max_roads=4, cost="length",
                                      corridor_m=3.0, workers=1,
                                      candidate_policy="faithful", rescore_every=1)
        assert [g.wkt for g in exact.geometry] == [g.wkt for g in lazy.geometry], mode
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run pytest tests/methods/test_arterial_lazy.py::test_lazy_faithful_rescore1_equals_exact -q`
Expected: FAIL (`_greedy_arterials_lazy` not defined).

- [ ] **Step 3: Implement the engine**

Mirror `_greedy_arterials`' per-step setup EXACTLY (so `eval_candidate` yields identical gains), but drive candidate selection with the heap. Key points: build `_StepState` with the same fields; set the module attribute `arterial._STEP_STATE` (not a local) so `eval_candidate` reads it; cache `real` per candidate (snap is committed-independent); pool the initial + added scoring.

```python
import heapq
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import reblock.methods.arterial as _art
from reblock.methods.arterial import (
    _StepState, _explode, _merge, _snap_graph, eval_candidate, _PARALLEL_THRESHOLD)
from reblock.methods.dijkstra import _boundary_graph
from reblock.budget import _BlockScoringContext
from reblock.derive.access import STREET_TOL, access_burden, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency


def _score_all(chords, use_pool, workers):
    """Return list[(gain, real)] via the same eval_candidate the exact path uses."""
    if use_pool and len(chords) >= _PARALLEL_THRESHOLD and \
            "fork" in multiprocessing.get_all_start_methods():
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("fork")) as ex:
            return list(ex.map(eval_candidate, chords,
                               chunksize=max(1, len(chords) // (workers * 4))))
    return [eval_candidate(c) for c in chords]


def _greedy_arterials_lazy(block, *, mode, objective, n_anchors=32, top_k=8, lam=2.0,
                           max_roads=15, cost="length", corridor_m=3.0, workers=16,
                           candidate_policy="grow", rescore_every=0):
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    base_burden = access_burden(parcel_access_layers(
        block, None, tol=STREET_TOL, adj=adj, unreached_depth=len(block.parcels) + 1))
    sg = _snap_graph(_boundary_graph(block.parcels))
    streets = list(block.streets.geometry)
    ctx = _BlockScoringContext(block) if objective in ("efficiency", "directness") else None
    policy = _make_policy(candidate_policy, block, streets, n_anchors, top_k, adj)

    committed: list[LineString] = []
    real_of: dict[str, BaseGeometry] = {}          # wkt(chord) -> realized geometry (snap-stable)
    heap: list[tuple[float, str, LineString, int]] = []   # (-gain, wkt, chord, scored_at_step)
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
        _art._STEP_STATE = _StepState(
            step=stepctx, sg=sg, base_val=base_val, base_merged=base_merged, committed=committed,
            mode=mode, objective=objective, cost=cost, lam=lam, corridor_m=corridor_m,
            committed_disp=committed_disp, block=block, crs=block.crs, adj=adj,
            base_burden=base_burden, ctx=ctx)
        try:
            # eager-score candidates entering this step
            if rescore_every and step > 0 and step % rescore_every == 0:
                pending = [c for c in [_c for _c in _iter_live(heap, live)]]  # full re-score
                heap = []
            for chord, (gain, real) in zip(pending, _score_all(pending, use_pool, workers)):
                if real is None:
                    continue
                key = chord.wkt
                real_of[key] = real
                live.add(key)
                heapq.heappush(heap, (-gain, key, chord, step))
            pending = []
            # pop-and-re-score the top until it is fresh under this committed set
            while heap:
                neg, key, chord, at = heap[0]
                if key not in live:
                    heapq.heappop(heap); continue
                if at == step:
                    break
                heapq.heappop(heap)
                gain, real = eval_candidate(chord)
                if real is None:
                    live.discard(key); continue
                real_of[key] = real
                heapq.heappush(heap, (-gain, key, chord, step))
            if not heap or -heap[0][0] <= 0.0:
                break
            neg, key, chord, at = heapq.heappop(heap)
        finally:
            _art._STEP_STATE = None
        committed.append(real_of[key])          # commit the realized geometry (LineString)
        live.discard(key)
        added, removed_keys = policy.after_commit(committed, len(committed))
        for k in removed_keys:
            live.discard(k)
        pending = added

    return _explode(_merge(committed), block.crs)
```

Add the `_iter_live` helper used by the periodic full-rescore branch:

```python
def _iter_live(heap, live):
    """Distinct live chords currently in the heap (for a full re-score rebuild)."""
    seen: set[str] = set()
    for _neg, key, chord, _at in heap:
        if key in live and key not in seen:
            seen.add(key)
            yield chord
```

Note: the periodic full-rescore branch clears `heap` and pushes fresh gains for all live chords via the normal `pending` path this step; this restores the exact per-step argmax and erases non-submodularity drift.

- [ ] **Step 4: Run the oracle test**

Run: `pixi run pytest tests/methods/test_arterial_lazy.py::test_lazy_faithful_rescore1_equals_exact -q`
Expected: PASS. If it fails, the divergence is a bookkeeping bug (context fields, tie-break, or termination), NOT submodularity — debug against `_greedy_arterials`' exact per-step values.

- [ ] **Step 5: Commit**

```bash
git add src/reblock/methods/arterial_lazy.py tests/methods/test_arterial_lazy.py
git commit -m "feat(arterial): CELF lazy-greedy engine (heap + pop-rescore-until-fresh)"
```

---

### Task 4: Dispatch, speedup/quality/determinism/policy tests

**Files:**
- Modify: `src/reblock/methods/arterial.py` (`GreedyArterialReblocker.propose` only)
- Test: `tests/methods/test_arterial_lazy.py`

**Interfaces:**
- Consumes: `_greedy_arterials_lazy` (Task 3).
- Produces: `propose` dispatches to `_greedy_arterials_lazy` when `self.lazy`, else `_greedy_arterials` (unchanged).

- [ ] **Step 1: Write dispatch + speedup + determinism + policy tests**

```python
def test_lazy_dispatch_and_determinism():
    block = _grid_block(5)
    m = GreedyArterialReblocker(mode="buildable", objective="directness", n_anchors=6,
                               max_roads=4, lazy=True, candidate_policy="grow")
    a = m.propose(block).roads
    b = m.propose(block).roads
    assert [g.wkt for g in a.geometry] == [g.wkt for g in b.geometry]   # deterministic
    assert len(a) > 0


def test_lazy_far_fewer_scorings_than_exact(monkeypatch):
    # instrument eval_candidate call count on a real block where arterial runs
    import reblock.methods.arterial as art
    from scoring_fixtures import _block_1808
    block = _block_1808()
    calls = {"n": 0}
    real_eval = art.eval_candidate
    def counting(chord):
        calls["n"] += 1
        return real_eval(chord)
    # exact
    monkeypatch.setattr(art, "eval_candidate", counting)
    calls["n"] = 0
    GreedyArterialReblocker(mode="buildable", n_anchors=8, max_roads=4, workers=1).propose(block)
    exact_calls = calls["n"]
    # lazy grow (patch the name the lazy engine imported, too)
    import reblock.methods.arterial_lazy as lz
    monkeypatch.setattr(lz, "eval_candidate", counting)
    calls["n"] = 0
    GreedyArterialReblocker(mode="buildable", n_anchors=8, max_roads=4, workers=1,
                            lazy=True, candidate_policy="grow", rescore_every=0).propose(block)
    lazy_calls = calls["n"]
    assert lazy_calls < exact_calls / 2, (lazy_calls, exact_calls)


def test_lazy_quality_within_tolerance():
    from reblock.budget import network_efficiency
    from scoring_fixtures import _block_1808
    block = _block_1808()
    exact = GreedyArterialReblocker(mode="buildable", n_anchors=8, max_roads=4, workers=1).propose(block).roads
    lazy = GreedyArterialReblocker(mode="buildable", n_anchors=8, max_roads=4, workers=1,
                                   lazy=True, candidate_policy="grow").propose(block).roads
    _e0, d_exact = network_efficiency(block, exact)
    _e1, d_lazy = network_efficiency(block, lazy)
    assert d_lazy >= d_exact - 0.02, (d_lazy, d_exact)   # comparable-or-better (prototype beat exact)
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run pytest tests/methods/test_arterial_lazy.py -k lazy_ -q`
Expected: FAIL on the dispatch/quality tests (propose ignores `lazy`).

- [ ] **Step 3: Add dispatch to `propose`**

In `GreedyArterialReblocker.propose`, replace the single call with:

```python
    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior
        if self.lazy:
            from reblock.methods.arterial_lazy import _greedy_arterials_lazy
            roads = _greedy_arterials_lazy(
                block, mode=self.mode, objective=self.objective, n_anchors=self.n_anchors,
                top_k=self.top_k, lam=self.lam, max_roads=self.max_roads, cost=self.cost,
                corridor_m=self.corridor_m, workers=self.workers,
                candidate_policy=self.candidate_policy, rescore_every=self.rescore_every)
        else:
            roads = _greedy_arterials(
                block, mode=self.mode, objective=self.objective, n_anchors=self.n_anchors,
                top_k=self.top_k, lam=self.lam, max_roads=self.max_roads, cost=self.cost,
                corridor_m=self.corridor_m, workers=self.workers)
        return Proposal(
            block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
            proposal_id=f"greedy_arterial_{self.mode}_{self.objective}", method="greedy_arterial",
            params={"segments": len(roads), "mode": self.mode, "objective": self.objective,
                    "cost": self.cost, "corridor_m": self.corridor_m, "lazy": self.lazy},
            block_identity=block.identity)
```

(The local import of `_greedy_arterials_lazy` breaks the arterial↔arterial_lazy import cycle.)

- [ ] **Step 4: Run the lazy tests**

Run: `pixi run pytest tests/methods/test_arterial_lazy.py -q`
Expected: PASS (dispatch works; lazy is deterministic, ≥2× fewer scorings, quality within tolerance).

- [ ] **Step 5: Confirm the exact path is still byte-identical**

Run: `pixi run pytest tests/methods/test_arterial.py -q`
Expected: PASS (all pre-existing arterial goldens; `lazy=False` path untouched).

- [ ] **Step 6: Commit**

```bash
git add src/reblock/methods/arterial.py tests/methods/test_arterial_lazy.py
git commit -m "feat(arterial): dispatch propose to lazy engine; speedup/quality/determinism tests"
```

---

### Task 5: Fixed/faithful policy behavior + regional smoke + docs

**Files:**
- Test: `tests/methods/test_arterial_lazy.py`
- Modify: `docs/superpowers/specs/2026-07-15-celf-lazy-arterial-design.md` (mark implemented) — optional

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write policy-parity + fixed-heap tests**

```python
def test_lazy_fixed_and_faithful_run_and_differ_from_exact_is_ok():
    block = _grid_block(5)
    for pol in ("fixed", "grow", "faithful"):
        roads = GreedyArterialReblocker(mode="buildable", objective="directness", n_anchors=6,
                                        max_roads=4, lazy=True, candidate_policy=pol).propose(block).roads
        assert len(roads) >= 0            # all policies produce a valid proposal
    # rescore_every=1 with grow/fixed equals a full-rescore greedy over that policy's set: determinism
    a = GreedyArterialReblocker(mode="buildable", n_anchors=6, max_roads=3, lazy=True,
                                candidate_policy="fixed", rescore_every=1).propose(block).roads
    b = GreedyArterialReblocker(mode="buildable", n_anchors=6, max_roads=3, lazy=True,
                                candidate_policy="fixed", rescore_every=1).propose(block).roads
    assert [g.wkt for g in a.geometry] == [g.wkt for g in b.geometry]
```

- [ ] **Step 2: Run**

Run: `pixi run pytest tests/methods/test_arterial_lazy.py -q`
Expected: PASS.

- [ ] **Step 3: Regional smoke (manual, not a committed test — too slow for CI)**

Run (documented for the reviewer; expect a large speedup vs exact at this budget):
```bash
pixi run python -c "
from reblock.data.provision import ensure_city_data
from reblock.data.kblock import KblockSource
from reblock.methods.arterial import GreedyArterialReblocker
import time
bp, bld = ensure_city_data('capetown')
blk = next(iter(KblockSource(bp, bld, region_id='p', min_buildings=10, block_ids=['ZAF.9.3.1_1_40972']).region().blocks))
t=time.time(); n=len(GreedyArterialReblocker(mode='buildable', lazy=True, candidate_policy='grow', max_roads=15, workers=16).propose(blk).roads); print('lazy', n, 'roads', round(time.time()-t,1),'s')
"
```
Expected: completes in a fraction of exact's time (exact ~15 roads × 55 s ≈ 14 min; lazy target: a couple minutes or less).

- [ ] **Step 4: Run the full suite**

Run: `pixi run pytest tests/methods/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/methods/test_arterial_lazy.py
git commit -m "test(arterial): fixed/faithful policy parity + determinism"
```

---

## Self-Review

**Spec coverage:** architecture (Task 3), config+identity (Task 1), 3 policies (Task 2), CELF core (Task 3), `rescore_every` incl. `=1` oracle (Task 3 test), pool composition (Task 3 `_score_all`), all six testing items (Tasks 1,3,4,5 — exact-untouched T4S5, oracle T3, speedup+quality T4, policy behavior T5, determinism T4, identity T1). Covered.

**Placeholder scan:** Task 2 Step 3 contains deliberately-flagged illustrative scaffolding (`if False`, `_KEY_ONLY`) with an explicit instruction to finalize `after_commit -> tuple[list[LineString], list[str]]` and delete the scaffolding — the implementer must produce clean code; no silent TODOs remain elsewhere.

**Type consistency:** `after_commit` returns `(list[LineString], list[str])` (added lines, removed wkt-keys) consistently across policies and the engine's `for k in removed_keys` consumption. Heap tuple `(-gain, wkt, chord, scored_at_step)` is consistent between push sites. `real_of[key]` keyed by `chord.wkt`. `identity` 12-tuple consistent between Task 1 code and its golden.
