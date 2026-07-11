# Process-Parallel Arterial Candidate Scoring — Design

**Status:** draft for review · **Date:** 2026-07-11

**Goal:** Parallelize `GreedyArterialReblocker`'s per-step candidate loop across CPU cores with a **fork-based process pool** (workers inherit the frozen per-block/per-step state copy-on-write), for a measured ~7–9× on real blocks, with **byte-identical proposed roads** (behavior-preserving). This makes big regions (the deep flagship, dense_cluster neighborhoods) reblock in minutes instead of tens of minutes.

**Architecture:** Refactor `_greedy_arterials`' inner `for chord in _candidate_chords(...)` loop into (a) a pure, deterministic per-candidate evaluation and (b) a parallel-map + reduce. The map uses a **fork** `multiprocessing`/`ProcessPoolExecutor` created *after* the step's frozen state (`step`, `sg`, `base_val`, mode/objective/cost) exists, so worker processes inherit it via COW — no pickling of the CSR/graph context; only the small chord in and `(gain, wkt)` out. Serial fallback below a work threshold and for `workers=1`.

**Tech Stack:** Python `concurrent.futures.ProcessPoolExecutor` (or `multiprocessing.get_context("fork")`), the existing arterial internals, Hydra config, pixi, pytest, mypy --strict.

**Why processes not threads (measured):** threading is GIL-bound and *slower* (0.66–0.84×) — the per-candidate work is dominated by GIL-held Python (`_snap`'s `nx.shortest_path`, the CSR build, entry re-projection, shapely). Fork/process pool: 4.5× (8w), **7.5× (16w)**, 8.6× (24w) on a 91-parcel block; memory is a non-issue (COW). See `scratchpad/spike*.py`.

## Global Constraints

- **Byte-identical proposals.** The parallel path must produce the EXACT same `roads` GeoDataFrame (same geometry, same commit order) as the serial path, on every block and both modes. This is the acceptance gate. `workers=1` must take the serial path (a literal no-op vs today).
- **Deterministic.** The best-candidate reduction uses the existing total-order tie-break (`gain`, then `real.wkt`), applied to the COLLECTED results — order-independent, so parallel == serial regardless of completion order.
- **Additive / opt-in default that's safe.** A new `workers` param on `GreedyArterialReblocker` (config `conf/method/greedy_arterial.yaml`), default `16`. `workers=1` = today's behavior. No other method changes.
- **Fork-only.** Uses the `fork` start method (Linux). On non-fork platforms or if fork is unavailable, fall back to serial (documented) — do NOT pickle the context per task.
- `mypy --strict`, ruff clean, `pixi run check` green.

## Design

### 1. Extract the per-candidate evaluation
Factor the loop body into a pure function of `(chord)` given the step's frozen state:
```
eval_candidate(chord) -> (gain: float, real_wkt: str | None)
```
It performs exactly today's work per mode: `_snap(chord, sg, lam)` (buildable) or the chord itself (aspirational); the `step.score_candidate` / `_union_with` / `_planarize` + `_score` branch; the `cost` denominator (length or displacement); and `gain`. Returns `real` as WKT (picklable across the process boundary) + the gain. `None` real → `(0.0, None)` (skipped). This function must be **module-level** (picklable for the pool) and read the step state from module-level globals set per step (§3).

### 2. Reduce
Collect `[(gain, real_wkt), ...]`, reconstruct `real` from WKT for the winner, and apply the existing selection: `gain > best_gain or (tie and real.wkt < best_real.wkt)`. Because the tie-break is a total order over distinct WKT, the argmax is unique and independent of result order → identical to serial.

### 3. Per-step fork pool + COW state
Per greedy step: stash the current `(step, sg, base_val, mode, objective, cost, lam, corridor_m, committed_disp, block)` in a module-level holder; create a fork `ProcessPoolExecutor(max_workers=workers)` (children inherit the holder via COW — the CSR/numpy/graph are NOT pickled); `map(eval_candidate, candidates, chunksize=...)`; tear down. A fresh pool per step is required because the frozen state (`step`, committed) changes each commit; fork is cheap (COW) so ~15 pools over a proposal is fine (measured overhead acceptable).
- **Chunking:** `chunksize = max(1, len(candidates) // (workers * 4))` to amortize IPC.
- **Threshold:** if `len(candidates) < THRESHOLD` (e.g. 128) or `workers <= 1`, run the serial loop (fork/pool overhead not worth it on small steps).

### 4. Config
`conf/method/greedy_arterial.yaml` gains `workers: 16`. Threaded through `GreedyArterialReblocker.propose` → `_greedy_arterials(..., workers=...)`. `greedy_arterial_buildable`/`aspirational`/`displacement` compare entries inherit it (override per entry if desired). The flagship may set `region_builder`-scoped `method.workers=24`.

## Correctness strategy

- **Proposal-identity gate (the acceptance test):** for several blocks (a small one, the 91-parcel `DJI.3_1_3238`, a deep region) and BOTH modes, assert `GreedyArterialReblocker(workers=16).propose(block).roads` is WKT-identical (sorted) to `GreedyArterialReblocker(workers=1).propose(block).roads`. This is the non-negotiable behavior-preservation check.
- **Determinism:** repeated `workers=16` runs → identical roads (no order dependence).
- **`workers=1` is exactly serial:** confirm it takes the non-pool code path (not a 1-worker pool) so it's a true no-op vs today; the existing arterial tests (unchanged) must stay green under the default.
- **Re-measure:** time `propose` on `DJI.3_1_3238` (91 parcels) and the flagship's 443-parcel region at workers ∈ {1, 16, 24}; report the speedup.

## Risks

- **Fork after imports:** forking a process with scipy/numpy/geopandas loaded is fine for pure compute (no threads held, no open handles written). The workers only READ the inherited frozen state + allocate locally. The joblib derive-cache is read-only on this path (scoring reads the frozen `step`, doesn't hit the L2 cache), so no concurrent cache writes.
- **Module-global state for workers** is the standard fork-pool pattern but is mutable global state; it is set/cleared around each step's pool and never read on the serial path. Guard against leakage (clear after the step).
- **Small-block regression:** the threshold + `workers` default guard against fork overhead dominating on tiny blocks; the proposal-identity test covers small blocks too.
- **Non-fork platforms:** fall back to serial (documented); no pickled-context path (would be slow + is out of scope).

## Task decomposition

1. **Extract `eval_candidate` + serial refactor (no parallelism yet).** Refactor `_greedy_arterials`' loop body into the pure module-level `eval_candidate` reading a module-level step-state holder, called serially. Gate: arterial tests + a new proposal-identity test (serial-refactor == pre-refactor) green — proves the extraction is behavior-preserving before adding processes.
2. **Fork process pool + reduce + `workers` config + threshold.** Add the per-step fork pool, the reduce, the `workers` param + config, the threshold + `workers=1` serial path. Gate: proposal-identity (`workers=16` == `workers=1` == serial) on small/91-parcel/deep-region blocks + both modes; determinism; re-measure the speedup.
3. **Docs + measurement.** README arterial note (`method.workers`, the perf characteristic); final speedup numbers on 91-parcel + the flagship region at {1,16,24}.

## Out of scope (follow-ups)
- Vectorized/GPU candidate scoring — only available for the resistance objective (rank-1 GEMM); a separate resistance-greedy project.
- Parallelizing across greedy STEPS (steps are sequential — the winner changes the base).
- A persistent cross-step pool (the per-step frozen state changes; fork-per-step is simpler + measured-acceptable).
