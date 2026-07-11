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
Factor the loop body into a pure, **module-level** function of `(chord)` given the step's frozen state:
```
eval_candidate(chord) -> (gain: float, real: BaseGeometry | None)
```
It performs exactly today's work per mode: `_snap(chord, sg, lam)` (buildable) or the chord itself (aspirational); the `step.score_candidate` / `_union_with` / `_planarize` + `_score` branch; the `cost` denominator (length or displacement); and `gain`. A `None`/zero-length `real` → `(0.0, None)`. **Return the shapely GEOMETRY, not WKT** — `ProcessPoolExecutor` pickles the return value via WKB (full float64, lossless), whereas `shapely.to_wkt()` defaults to `rounding_precision=6` (lossy); returning the geometry sidesteps the precision question entirely so the winner's `real` is bit-identical to the serial path's, and the next step's `unary_union(committed + [real])` is unchanged. The function reads the step state from a module-level holder set per step (§3).

### 2. Reduce — ONE shared implementation with the serial sentinel (Critical)
The serial loop is **not a plain argmax**: it initializes `best_gain, best_real = 0.0, None` and its tie-break is additionally gated on `best_real is not None`, so a candidate with `gain <= 0` can NEVER win — this is exactly the "no candidate improves → stop" termination (`arterial.py:287`), which fires on every proposal's final step. A naive `best = None` argmax would wrongly commit a zero-gain road there and change the geometry.

Therefore: factor a SINGLE `_best_candidate(results) -> (gain, real | None)` used by BOTH the serial path and the parallel-collect path, with the exact `(0.0, None)` init and `best_real is not None`-gated tie-break (compare `real.wkt` in the parent — exact, since same geometry → same wkt). There is one reduce to get right. Determinism has two independent guarantees: (a) `Executor.map` returns results in **input order** regardless of completion/chunksize (CPython-verified), so the reduce sees the same sequence as serial; (b) the tie-break is a total order over distinct wkt, so the argmax is order-independent anyway (defense in depth). The `inf` gain (`denom<=0 and raw>0`) path reduces correctly (`inf == inf` tie-breaks by wkt).

### 3. Per-step fork pool + COW state
Per greedy step: stash the FULL current state — `(step, sg, base_val, base_merged, committed, mode, objective, cost, lam, corridor_m, committed_disp, block)` — in a module-level holder. (`base_merged` is needed by the `access`-objective buildable branch and the displacement fallback; `committed` by the aspirational full-`_planarize` branch — omitting either breaks 2 of the 3 scoring branches.) Then create the pool with an **explicit fork context** — `ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork"))` — so children inherit the holder via COW; the CSR/numpy/graph are NOT pickled. `map(eval_candidate, candidates, chunksize=...)`; tear down; **clear the holder**. Do NOT rely on `ProcessPoolExecutor()`'s implicit default start method (version/platform-dependent); if `fork` is unavailable, fall back to the serial path (never pickle the context per task). A fresh pool per step is required because `step`/`committed` change each commit; fork is cheap (COW).
- **Non-reentrancy guard:** assert the holder is empty before setting it (detects any future across-block parallel dispatch racing on the shared holder), and clear it in a `finally`. Today's pipeline is single-threaded per `propose`, so this is a guard, not a lock.
- **Chunking:** `chunksize = max(1, len(candidates) // (workers * 4))` to amortize IPC (does not affect the result order — `map` preserves input order).
- **Threshold:** if `len(candidates) < THRESHOLD` (e.g. 128) or `workers <= 1`, run the serial loop via the SAME `eval_candidate` + `_best_candidate` (fork/pool overhead not worth it on small steps; `workers=1` is a true serial no-op vs today, not a 1-worker pool).

### 4. Config
`conf/method/greedy_arterial.yaml` gains `workers: 16`. Threaded through `GreedyArterialReblocker.propose` → `_greedy_arterials(..., workers=...)`. `greedy_arterial_buildable`/`aspirational`/`displacement` compare entries inherit it (override per entry if desired). The flagship may set `region_builder`-scoped `method.workers=24`.

## Correctness strategy

- **Direct reduce unit test (guards the Critical):** a fast, geometry-free test of `_best_candidate` asserting (a) all-non-positive gains → `(0.0, None)` (the termination case a naive `best=None` argmax gets wrong); (b) a positive-gain tie → deterministic min-`wkt` winner, identical for forward and reversed input order; (c) a mix including `inf` gains reduces correctly. This is independent of which blocks the proposal-identity test happens to hit.
- **Proposal-identity gate (the acceptance test):** for several blocks (a small one, the 91-parcel `DJI.3_1_3238`, a deep region) and BOTH modes, assert `GreedyArterialReblocker(workers=16).propose(block).roads` is WKT-identical (sorted) to `GreedyArterialReblocker(workers=1).propose(block).roads` AND to the pre-refactor output. **Include at least one block that terminates BEFORE `max_roads`** (so the all-non-positive-gain reduce path is exercised end-to-end, not just positive-gain steps). Run it a few times (fork races are low-probability per run).
- **Geometry round-trip:** assert the winner's `real` returned across the process boundary is coordinate-identical to the serial path's (not just `.equals()`), so a future upstream precision change fails fast here.
- **`workers=1` is exactly serial:** confirm it takes the non-pool path (not a 1-worker pool) — a true no-op vs today; existing arterial tests (unchanged) stay green under the default `workers=16`.
- **Soak test:** several dozen back-to-back `propose()` calls (or a small multi-region pipeline run) at `workers ∈ {16, 24}` complete without leaking/accumulating semaphores or hanging — validates the ~15-pools-per-propose × many-blocks churn at production scale, not just one pool in isolation.
- **Re-measure:** time `propose` on `DJI.3_1_3238` (91 parcels) and the flagship's 443-parcel region at workers ∈ {1, 16, 24}; report the speedup.

## Risks

- **Fork after imports:** forking a process with scipy/numpy/geopandas loaded is fine for pure compute (no threads held, no open handles written). The workers only READ the inherited frozen state + allocate locally. The joblib derive-cache is read-only on this path (scoring reads the frozen `step`, doesn't hit the L2 cache), so no concurrent cache writes.
- **Module-global state for workers** is the standard fork-pool pattern but is mutable global state; set/cleared around each step's pool (cleared in `finally`) and never read on the serial path. Safe today only because dispatch is single-threaded per `propose` — a **non-reentrancy assertion** (holder empty before set) makes any future across-block parallelism fail loudly instead of racing (a fork by one block right after another overwrites the holder → silently wrong scoring). See §3.
- **Fork races (scipy/GEOS):** no BLAS in the in-scope hot path (`_snap`=networkx+shapely ufuncs, `score_candidate`=csgraph+numpy elementwise; `factorized`/SuperLU is only the resistance objective, out of scope), and no parent thread pool before the fork — so residual fork-race risk is theoretical. Mitigate by running the identity/determinism tests a few times, at 24-32 workers.
- **Small-block regression:** the threshold + `workers` default guard against fork overhead dominating on tiny blocks; the proposal-identity test covers small blocks too.
- **Non-fork platforms:** pin `mp_context=get_context("fork")`; if unavailable, fall back to serial (documented); no pickled-context path (would be slow + is out of scope).
- **Determinism:** primary guarantee is `Executor.map`'s input-order-preserving results (chunksize/worker-count-independent); the total-order tie-break is defense in depth (§2).

## Task decomposition

1. **Extract `eval_candidate` + shared `_best_candidate` reduce + serial refactor (no parallelism yet).** Refactor `_greedy_arterials`' loop body into the pure module-level `eval_candidate` (returns `(gain, real_geometry)`) reading a module-level step-state holder, and the shared `_best_candidate` reduce with the exact `(0.0, None)` init + `best_real is not None` guard, called serially. Gate: the **direct `_best_candidate` unit test** (all-non-positive → `(0.0,None)`; positive tie deterministic + order-independent; inf ties) + arterial tests + a proposal-identity test (serial-refactor == pre-refactor, on a block that terminates before `max_roads`) — proves the extraction AND the reduce are behavior-preserving before adding processes.
2. **Fork process pool + `workers` config + threshold.** Add the per-step explicit-fork-context pool (holder + non-reentrancy guard + `finally` clear), the `workers` param + config, the threshold + `workers=1` serial path. Gate: proposal-identity (`workers=16` == `workers=1` == serial, incl. an early-terminating block + both modes, run a few times); determinism (repeat runs identical); geometry round-trip coordinate-identity; re-measure.
3. **Soak test + docs + measurement.** A soak test (several dozen back-to-back `propose()` at `workers ∈ {16,24}` — no semaphore leak/hang). README arterial note (`method.workers`, perf characteristic). Final speedup numbers on 91-parcel + the flagship 443-parcel region at {1,16,24}.

## Out of scope (follow-ups)
- Vectorized/GPU candidate scoring — only available for the resistance objective (rank-1 GEMM); a separate resistance-greedy project.
- Parallelizing across greedy STEPS (steps are sequential — the winner changes the base).
- A persistent cross-step pool (the per-step frozen state changes; fork-per-step is simpler + measured-acceptable).
