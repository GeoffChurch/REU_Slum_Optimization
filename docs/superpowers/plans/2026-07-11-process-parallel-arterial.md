# Process-Parallel Arterial — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize `GreedyArterialReblocker`'s per-step candidate loop across cores with a fork process pool (workers inherit the frozen state COW), ~7–9× on real blocks, with byte-identical proposed roads.

**Architecture:** Refactor `_greedy_arterials`' inner loop into a module-level `eval_candidate(chord) -> (gain, real_geometry)` + a shared `_best_candidate` reduce; run the candidates either serially or through a per-step explicit-fork `ProcessPoolExecutor`. See spec `docs/superpowers/specs/2026-07-11-process-parallel-arterial-design.md` — read it. Spikes: `scratchpad/spike.py` (threads, GIL-bound), `scratchpad/spike2.py` (fork, 7.5×@16 / 8.6×@24).

## Global Constraints

- **Byte-identical proposals:** the parallel path produces the EXACT same `roads` (geometry + commit order) as serial, on every block + both modes. `workers=1` = serial no-op. This is the acceptance gate.
- **The reduce keeps the serial sentinel:** `best_gain, best_real = 0.0, None`; tie-break gated on `best_real is not None` — a `gain<=0` candidate never wins (the termination condition). ONE shared `_best_candidate` used by serial + parallel.
- **Return geometry, not WKT** (WKB pickle is lossless; `shapely.to_wkt` rounds).
- **Fork pinned:** `mp_context=multiprocessing.get_context("fork")`; non-fork → serial fallback.
- Verify each task with `pixi run check`; run to completion.

---

### Task 1: Extract `eval_candidate` + shared `_best_candidate` reduce (serial only, no parallelism)

**Files:** Modify `src/reblock/methods/arterial.py`; Test: `tests/methods/test_arterial.py`.

**Interfaces — Produces:**
- module-level `eval_candidate(chord: LineString) -> tuple[float, BaseGeometry | None]` reading a module-level `_STEP_STATE` holder.
- `_best_candidate(results: Iterable[tuple[float, BaseGeometry | None]]) -> tuple[float, BaseGeometry | None]` (the shared reduce, `(0.0, None)` init + `best_real is not None`-gated tie-break by `real.wkt`).

- [ ] **Step 1: Write the direct reduce unit test** (`tests/methods/test_arterial.py`):
```python
from shapely.geometry import LineString
from reblock.methods.arterial import _best_candidate
def test_best_candidate_reduce():
    L = lambda a,b: LineString([a,b])
    # (a) all non-positive gains -> (0.0, None)  [termination]
    assert _best_candidate([(-0.1, L((0,0),(1,1))), (0.0, L((0,0),(2,2))), (-0.05, L((0,0),(3,3)))]) == (0.0, None)
    # (b) positive-gain tie -> deterministic min-wkt winner, order-independent
    r1, r2 = L((0,0),(1,0)), L((0,0),(0,1))
    lo = r1 if r1.wkt < r2.wkt else r2
    fwd = _best_candidate([(0.4, r1), (0.4, r2)]); rev = _best_candidate([(0.4, r2), (0.4, r1)])
    assert fwd[1].wkt == lo.wkt and rev[1].wkt == lo.wkt and fwd[0] == 0.4
    # (c) inf gain wins; inf ties tie-break by wkt
    inf = float("inf")
    assert _best_candidate([(0.9, r1), (inf, r2)])[0] == inf
    # (d) None reals skipped
    assert _best_candidate([(0.0, None), (0.5, r1)]) == (0.5, r1) or _best_candidate([(0.0,None),(0.5,r1)])[1].wkt==r1.wkt
```
- [ ] **Step 2: Run — FAIL** (`_best_candidate`/`eval_candidate` undefined).
- [ ] **Step 3: Refactor.** Extract the loop body (`arterial.py:262-286`) into module-level `eval_candidate(chord)` reading a `_STEP_STATE` module holder (a dataclass/dict with `step, sg, base_val, base_merged, committed, mode, objective, cost, lam, corridor_m, committed_disp, block, crs`). Extract `_best_candidate(results)` with the EXACT serial logic. In `_greedy_arterials`, per step: set `_STEP_STATE`, run `results = [eval_candidate(c) for c in _candidate_chords(...)]` (SERIAL still), `best_gain, best_real = _best_candidate(results)`, clear `_STEP_STATE` in `finally`. No processes yet.
- [ ] **Step 4: Run — PASS** + proposal-identity: add `test_arterial_serial_refactor_identical` asserting `propose(block).roads` WKT-identical to the pre-refactor (use the pinned `ref_values_1808` arterial WKT + `DJI.3_1_3238`) on a block that terminates before max_roads, both modes.
- [ ] **Step 5: `pixi run check` green; commit** `refactor: extract eval_candidate + shared _best_candidate reduce (serial)`.

---

### Task 2: Fork process pool + `workers` config + threshold

**Files:** Modify `src/reblock/methods/arterial.py`, `conf/method/greedy_arterial.yaml`; Test: `tests/methods/test_arterial.py`.

**Interfaces:** `GreedyArterialReblocker(..., workers: int = 16)`; threaded to `_greedy_arterials(..., workers=...)`.

- [ ] **Step 1: Write the proposal-identity + determinism tests.**
```python
def test_arterial_parallel_identical_to_serial():
    for mode in ("buildable", "aspirational"):
        block = _block()  # DJI.3_1_3238 or a deep region; terminates before max_roads
        serial = GreedyArterialReblocker(mode=mode, workers=1).propose(block).roads
        par = GreedyArterialReblocker(mode=mode, workers=16).propose(block).roads
        assert sorted(g.wkt for g in serial.geometry) == sorted(g.wkt for g in par.geometry)

def test_arterial_parallel_deterministic():
    block = _block()
    a = GreedyArterialReblocker(workers=16).propose(block).roads
    b = GreedyArterialReblocker(workers=16).propose(block).roads
    assert sorted(g.wkt for g in a.geometry) == sorted(g.wkt for g in b.geometry)
```
- [ ] **Step 2: Run — FAIL** (`workers` param absent).
- [ ] **Step 3: Implement.** Add `workers` to the dataclass + `_greedy_arterials`. Per step: if `workers <= 1 or len(candidates) < THRESHOLD` (128) → serial `[eval_candidate(c) for c in candidates]`; else set `_STEP_STATE`, create `ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork"))` (assert `_STEP_STATE` was unset before — reentrancy guard), `results = list(ex.map(eval_candidate, candidates, chunksize=max(1, len(candidates)//(workers*4))))`, `finally` clear `_STEP_STATE`; `_best_candidate(results)`. If `get_context("fork")` unavailable → serial. Add `workers: 16` to `conf/method/greedy_arterial.yaml`.
- [ ] **Step 4: Run — PASS** (identity, determinism; run identity a few times). `pixi run pytest tests/methods/test_arterial.py -q`.
- [ ] **Step 5: Re-measure** `propose` on DJI.3_1_3238 at workers ∈ {1,16,24}; record. `pixi run check` green; **commit** `perf: process-parallel arterial candidate scoring (workers config)`.

---

### Task 3: Soak test + docs + final measurement

**Files:** Test: `tests/methods/test_arterial.py`; Modify `README.md`.

- [ ] **Step 1: Soak test** `test_arterial_parallel_soak`: ~30 back-to-back `GreedyArterialReblocker(workers=16).propose(small_block)` calls complete without error/hang (guards semaphore/resource-tracker accumulation across many short-lived pools). Keep the block small so the test is fast.
- [ ] **Step 2: Run — PASS.**
- [ ] **Step 3: README** — arterial paragraph: note `method.workers` (default 16) process-parallelizes candidate scoring (~7–9× on multicore; `workers=1` = serial); mention it's why big regions/flagship are now tractable.
- [ ] **Step 4: Final measurement** — `propose` on DJI.3_1_3238 (91 par) and a ~443-parcel dense_cluster region at workers ∈ {1,16,24}; record the speedups in the commit.
- [ ] **Step 5: `pixi run check` green; commit** `test+docs: arterial parallel soak test + workers docs + measurements`.

## Self-Review

- **Spec coverage:** Task 1 = extraction + the shared reduce + the Critical direct unit test + serial-identity; Task 2 = the fork pool + workers config + threshold + parallel-identity/determinism/geometry-roundtrip; Task 3 = soak + docs + measurement. The frozen-state tuple (`base_merged`/`committed` included) is in Task 1 Step 3.
- **Placeholders:** `_block()` helper (Task 2) is named for the implementer to point at DJI.3_1_3238 / a deep region that terminates early; THRESHOLD=128 and workers=16 are concrete.
- **Type consistency:** `eval_candidate -> (float, BaseGeometry|None)` and `_best_candidate` consume the same tuple in Tasks 1-2; `workers: int` is the sole new field.
