# The arterial, fast and with a displacement budget: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `greedy_arterial_access_displacement` scores a candidate's access gain incrementally (the same commits, several times faster), proposes from an anchor sampler whose cap holds, and stops at a required displacement budget; a study on six lineup blocks decides the presets.

**Architecture:** The access gain of a snapped road becomes a per-step object, `_AccessStep`, that lowers the step's base peel from the parcels the road fronts instead of re-peeling the block, and replaces street connectivity per candidate with one liveness test against the step's seed geometry.
Anchor sampling becomes a Strategy, `AnchorSampler` (`Uncapped`, `PerLineFallback` = today's capped sampler, `HardCap` = the new one), injected into `GreedyArterialReblocker` in place of `n_anchors` and `max_anchors`.
The engines stop when the committed roads' displacement share reaches a required keyword-only `max_displacement` and report why they stopped (`GreedyRun.stop`).
`research/roadless/arterial_study.py` runs the arms through the lineup's own scoring path (road_rescore's `prefixes_one` and `cx_network`), and `research/roadless/arterial_check.py` holds the real-block guards, each failing under a named fault.

**Tech Stack:** Python 3.12 (uv), shapely 2 (STRtree, `dwithin`, `prepare`), numpy, pandas, geopandas, pytest (with `-n 0 --no-cov`), mypy --strict, ruff; GeoffChurch/cluster_submit via research/roadless/cluster.py for the study.

**Spec:** the brief of 2026-10-18 (the caller's task: three changes and the study that judges them) and its measurements, copied into the worktree by Task 1 (`research/roadless/arterial_rows/before/`) because their scratchpad dies on reboot.
The numbers the plan argues from:
- `max_roads` counts greedy commits (engines.py, each engine's `while len(committed) < max_roads`); `n_roads` counts planarized segments.
- The loop ends at `max_roads` or when no shortlisted candidate has gain > 0; 55 of 62 lineup networks end with every parcel at depth 1, at a median 54 road km/km² (cycle_native 29).
- The lineup lifted the preset's cap of 60 to ARTERIAL_MAX_ROADS = 300 (road_rescore.py, `lineup_arms`), its whole-network area D a median 0.206; scoring reads prefixes at area D 0.02, 0.05, 0.1, 0.15, 0.2.
- On 20269 one step grows from 10 s (nothing committed) to 51 s (607 segments): ranking 1.5 s to 22 s (candidates 7,112 to 141,750 at ~0.15 ms each; anchors 112 to 525 under a cap of 128), exact evaluation of k = 512 at 17 to 54 ms each (gain 9 to 51 ms, of which the peel 8 to 44 ms and street connectivity 0.4 to 16 ms; `_explode` 1 to 4 ms; the union 0 to 2 ms; snap ~2 to 3 ms; the cost 4.6 to 0.8 ms).
- `canonical_order` (research/roadless/common.py) orders the whole network street-first by drainage over the whole network, so a scored prefix of a shorter network need not be a scored prefix of the longer one.

## Global Constraints

Every task's requirements include this section.

**Where the work happens**
- Execution happens in a git worktree, never in the main checkout: another agent is finishing the consolidated pass there and long local jobs run from it.
- The worktree: `git -C /home/gchurchill/src/reblock worktree add -b research/arterial-speed .worktrees/arterial-speed research/roadless`, after `grep -qx '/.worktrees/' .git/info/exclude || echo '/.worktrees/' >> .git/info/exclude` (a local ignore, so the main checkout's `git status` never offers the tree to the other agent).
- In each worktree, once: `uv sync --frozen` (1 process, ~2 min; the venv links into uv's cache). Never `uv cache clean`.
- In the main checkout, never touch research/roadless/stopping_review/, research/group_steiner/, research/roadless/{ids58,ids59,ov16,ov19,ov_a,ov_b,ss_miss}.txt, or research/roadless/road_rows/.
  The baseline is read from the main checkout, read-only, by absolute path: `BASE=/home/gchurchill/src/reblock/research/roadless/road_rows/lineup-caf14fa2`.
- Edit only `src/reblock/methods/arterial/`, `conf/`, `tests/`, the `scripts/` callers the type checker names, and `research/roadless/`.
  Never edit a Block producer (`src/reblock/kblock/`, `region*`, `derivations*`, `derive/`, which holds `access.py`), nor `src/reblock/buildings.py` or `src/reblock/budget.py`: a Block producer's edit colds the whole derive cache (a multi-hour regeneration). An arterial edit misses only the arterial's own entries.

**Code rules (global CLAUDE.md)**
- No legacy shims: when an interface changes, every caller moves and the old path is deleted (`_Unioned`, `_union_with`, `_anchor_points`, the `max_anchors: 0` sentinel, `ARTERIAL_MAX_ROADS`).
- Strategy resolved upstream: `AnchorSampler` is a Protocol with injected instances, built where config is read (Hydra `_target_`), never a mode string.
- No getattr/.get defaults on closed sets; frozen dataclasses and NamedTuples; the study's arm and the guard's fault are CLI names, looked up once with `[]` (no default).
- Result-changing parameters are required and keyword-only below the edge, and in the run identity: `max_displacement`, the sampler's `n` and `max_anchors`, the study's arm (its network name carries a digest of the method's identity).
- Strict mypy on everything touched; ruff.

**Gates (each from the worktree's root)**
- Types (src, tests, scripts): `uv run mypy` (its file list is pyproject's; 1 process, ~3 min): no error.
- Types (research): `MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent research/roadless/arterial_check.py research/roadless/arterial_steps.py research/roadless/arterial_study.py`: no error in these files; for road_rescore.py, the count of errors reported for it does not rise (record it before editing).
- Lint: `uv run ruff check src tests scripts` and `uv run ruff check research/roadless/arterial_check.py research/roadless/arterial_steps.py research/roadless/arterial_study.py`; touched existing research modules pass `uv run ruff check --select F,E9 <path>` (check they pass before editing).
- Tests: `uv run pytest -n 0 --no-cov <files>`. pyproject's `addopts` carry `-n auto`, one process per core: always pass `-n 0`.
  The `parallel` tests of tests/methods/test_arterial.py and tests/methods/test_engines.py fork up to 16 workers for a few seconds: deselect them (`-k "not parallel"`) until the caller confirms the slots, then run them once per task that touches scoring.
- Guards: research guards run with `PYTHONPATH=.` and `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`, as `uv run python research/roadless/arterial_check.py ...`.
  Each guard is shown both ways: it passes on the code, and fails under each of its faults (a patch inside the guard's process, free_roads_check.py's `_patched`; no file is edited to inject one); a fault that goes unseen makes the run exit non-zero.
  A pytest guard does the same with `monkeypatch`: a test that the check passes, and one test per fault that the same check reports a failure.

**Processes**
- Local compute is capped at 8 processes in total and the other work holds 8 now. Every run step below states its process count and expected duration.
  Before a local run, list the other work (`ps -eo pid,etime,pcpu,args --sort=-pcpu | head -20`); if the run would take the total past 8, wait for the caller to say slots are free.
- Single-threaded numerics everywhere (the variables above); kill by PID, never `pkill -f`.
- A run longer than ~2 min goes through the Bash tool's `run_in_background`, which notifies on exit; never a polling loop or `sleep N; cmd`. Cluster runs are waited on with `uv run python research/roadless/cluster.py wait <run> --timeout <T>` in `run_in_background` (exit 0 passed, 3 failed, 4 timed out, 5 blind).

**Commits and records**
- `git -c core.hooksPath=.githooks commit -m "research: roadless -- <what, with its numbers>" -m "<body>" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RoNpoEMZ1kz2hogCsAsmXJ" -- <paths>` (a new file `git add`ed first, a deletion `git rm`ed).
- Push after each commit: `git push -u origin research/arterial-speed` (Ruling 8). Never push main. research/roadless is pushed only in Task 7.
- research/roadless/NOTES.md and BACKLOG.md are column-wrapped at 99; this plan and other new Markdown keep one sentence per line.
- Owner rules from memory: FP differences within tolerance are accepted (never chase bit-identity beyond what a test pins); dominated ideas go to research NOTES and the wiki's dead-end record, never a published page; changed figures just change, without before/after ceremony.

## Rulings (owner to check)

1. **The budget is count-weighted displacement**, `Σ c_i / max(n_buildings, 1)` over the committed corridor, the quantity cycle_native's and resistance_lp's `max_displacement: 0.20` cap, so the lineup's three capped methods mean one thing by 0.2.
   The lineup scores area-weighted prefixes, and the two differ: 39240's stored network has count D 0.512 and area D 0.338, so at count 0.2 its area D is near 0.13 and the 0.15 and 0.2 area budgets go unreached there, as they can for cycle_native and resistance_lp. Lens A (0.05) and the 220 study's 0.10 sit clear. Say "area" and Task 3's meter weights by footprint area instead.
2. **The crossing road is kept:** the loop checks before each step and stops once the share is at least `max_displacement`, so a network overshoots by at most one road (~0.003 of D per commit on 20269) and every area budget it covers is a true prefix. cycle_native instead skips candidates that would cross; that would cost a displacement evaluation per shortlisted candidate here and leave the 0.2 budget always unreached.
3. **`max_roads` stays, as a safety cap that should never bind**, with every run's stop reason recorded (`params["stop"]`: `max_displacement`, `no_gain` or `max_roads`) so a bind is seen, which cycle_native's `range(60)` was not. The directness presets keep it as their real budget (15). The access pair goes from 60 to 1000 commits, and road_rescore's `ARTERIAL_MAX_ROADS` override is deleted: the preset carries its own budget now. Task 5 flags any `max_roads` stop; the study's blocks show the commits a stop at 0.2 takes.
4. **Three samplers, no sentinel.** `max_anchors: 0` meant "uncapped" (the directness presets). It becomes its own Strategy, `Uncapped(n)`, and both capped samplers require `max_anchors >= 1`. If the study deletes `PerLineFallback`, the uncapped presets are untouched by that deletion.
5. **The lineup's rows root hashes the arterial's whole identity** (`repr((free_roads.lineup_free().identity, lineup_arms()[ARTERIAL].identity))`), not only `max_displacement`, since the sampler changes the networks too. Every lineup row then sits under an old root: Task 7 migrates them (the six other methods' rows moved, the arterial's re-proposed on the lineup's 62 blocks), after the consolidated pass returns and only on the owner's yes, with the task-hours Task 5 extrapolates.
6. **The study runs on orion** (research/roadless/cluster.py `submit cpu`, every arm of a block on one node): 12 tasks, ~10 task-h, ~3 h wall. The local machine has no free slot. Say "local" and Task 5 runs it here, 2 processes once slots free, ~6 h wall.
7. **The dominance rule for the samplers:** `HardCap` dominates `PerLineFallback` only if it is faster on every block and its median paired difference (`HardCap - PerLineFallback`, over the blocks that reach the budget in both) is >= 0 for each of L_P, L_Pq and L_J2 at both area D 0.05 and 0.10 (road_rescore's REPORT_BUDGETS). `PerLineFallback` is dominated only if slower on every block and behind on all six medians. Otherwise both stay, as two named presets. With six blocks this is a coarse rule; say if you want another.
8. **research/arterial-speed is pushed to origin as it goes** (GeoffChurch/REU_Slum_Optimization is public; research branches are already pushed there, research/group-steiner among them), so no commit lives only on this disk. It merges into research/roadless in Task 7.
9. **The repulsion twin follows the displacement default:** `greedy_arterial_access_repulsion` gets the same sampler, `max_displacement` and `max_roads` as the displacement default (compare_config: "BOTH variants move together"). A speed-first twin of it is not made (unmeasured).

## Review Focus

1. A block with no buildings (`n_buildings = 0`): the share is 0 and never divides by zero; the run ends on `no_gain` or `max_roads`. Pinned in Task 3 (`test_no_buildings_never_stops_on_displacement`).
2. A parcel with no adjacency path to the street (stranded at `n_parcels + 1`) that a road reaches: the incremental gain lowers it to 1 and stays exact. Pinned in Task 1 (`test_access_step_lowers_a_stranded_parcel`).
3. Streets given as a MultiLineString (a courtyard block): the incremental gain matches the re-peel on every candidate. Pinned in Task 1 (the `_holed_block` case of `test_access_step_matches_the_full_repeel`).
4. A run that commits nothing (no candidate gains): `GreedyRun(stop="no_gain", commits=0)` and an empty road frame that still carries `drain`. Pinned in Task 3 (`test_a_run_with_no_gain_commits_nothing`).
5. `HardCap` where two samples snap (`_rnd`) to one point, or the network has hundreds of short lines (the 525-anchor case): never more than `max_anchors`, sorted, unique, deterministic. Pinned in Task 2 (`test_hard_cap_never_exceeds_max_anchors`) and its real-block guard (`anchors`).

## Files

- Modify `src/reblock/methods/arterial/objectives.py`: `_AccessStep` added, `_Unioned` deleted, `_AccessBlock` holds the parcel tree (Task 1).
- Modify `src/reblock/methods/arterial/primitives.py`: `_union_with` deleted (Task 1); `_anchor_points`, `_merged_lines`, `_arclength_samples`, `_vertices_and_samples` moved out (Task 2).
- Create `src/reblock/methods/arterial/anchors.py`: `AnchorSampler`, `Uncapped`, `PerLineFallback`, `HardCap`, `AnchorIdentity` (Task 2).
- Create `src/reblock/methods/arterial/stop.py`: `DisplacementMeter`, `Stop`, `GreedyRun` (Task 3).
- Modify `src/reblock/methods/arterial/{engines,policies,reblocker,__init__}.py` (Tasks 2, 3).
- Modify `conf/compare_config.yaml`, `conf/method/greedy_arterial*.yaml` and any other conf entry targeting `GreedyArterialReblocker` (Tasks 2, 3, 6).
- Create `tests/methods/test_arterial_access_step.py`, `tests/methods/test_arterial_anchors.py`, `tests/methods/test_arterial_stop.py`, `tests/data/scoring/arterial_access_ref.json`; modify the arterial tests and `scripts/perf/*` the type checker names (Tasks 1 to 3).
- Create `research/roadless/arterial_steps.py` (the per-step log and the before/after comparison), `research/roadless/arterial_check.py` (guards `gain`, `anchors`), `research/roadless/arterial_study.py` (the study's runs and report) (Tasks 1, 2, 4).
- Create `research/roadless/arterial_rows/before/` (the brief's measurements) and `research/roadless/arterial_rows/summary_{runs,lenses,prefixes}.csv` (Tasks 1, 5); raw study rows under `research/roadless/arterial_rows/study/` stay untracked.
- Modify `research/roadless/road_rescore.py` (`lineup_arms`, `lineup_rows`, the docstring's cap), `research/roadless/NOTES.md`, `research/roadless/BACKLOG.md` (Tasks 3, 6).

---

### Task 1: An incremental exact access gain

**Files:**
- Modify: `src/reblock/methods/arterial/objectives.py` (`_AccessBlock`, `Access.for_block`; `_Unioned` deleted)
- Modify: `src/reblock/methods/arterial/primitives.py` (`_union_with` deleted), `scripts/perf/snap_vs_peel.py` (its `_union_with(None, real)` becomes `unary_union([real])`)
- Create: `tests/methods/test_arterial_access_step.py`, `tests/data/scoring/arterial_access_ref.json`
- Create: `research/roadless/arterial_steps.py`, `research/roadless/arterial_check.py`
- Create: `research/roadless/arterial_rows/before/` (copies of the brief's `microbench.py`, `micro_20269.csv`, `depth_end.py`, `depth_end.csv`, `depth_end_2.csv`, `anchors.py`, `steps.py` from `/tmp/claude-1641171234/-home-gchurchill-src-reblock/fe8870c0-fbd9-4712-ac98-aebcb951b199/scratchpad/arterial/`)

**Interfaces:**
- Produces: `_AccessStep` (a `StepGain`): frozen dataclass with fields `neighbours: list[set[int]]`, `parcel_tree: STRtree`, `seed: BaseGeometry | None`, `tol: float`, `depths: NDArray[np.int64]`, `burden: int`, `base_burden: float`, `base: float`; methods `live(real: LineString) -> bool`, `seeded(real: LineString) -> NDArray[np.intp]`, `relaxed_burden(seeds: NDArray[np.intp]) -> int`, `of(real: LineString) -> float`. The three helpers are the faults' seams.
- Produces: `_AccessBlock(adjacency: ParcelAdjacency, base_burden: float, parcel_tree: STRtree)`; `at_step` returns `_AccessStep` when `realizer.snaps`, `_Replanarized` otherwise (unchanged for free chords).
- Produces: `arterial_steps.StepLog`: a context manager that patches `engines._best_candidate` and `engines._candidate_chords` (the module's documented instrument seams) and records one row per step: `step, t_end, step_s, anchors, candidates, committed, commit_wkb`; `arterial_steps.py steps <out> <id> <max_roads>` and `compare <dir_a> <dir_b> <ids,>`.
- Produces: `arterial_check.py gain <ids,> <baseline-root> [fault]` and `both-ways <guards,> <ids,> <baseline-root>`, with `GUARDS` and `FAULTS` as free_roads_check.py has them.

- [ ] **Step 1: Make the worktrees and keep the measurements**

Create the worktree (Global Constraints), then a detached base tree at the same commit for the comparison: `git worktree add --detach .worktrees/arterial-base research/arterial-speed`; `uv sync --frozen` in each (1 process each, ~2 min).
Copy the files listed above into `research/roadless/arterial_rows/before/` and commit them (`research: roadless -- the arterial's measured step costs before the speed work (20269: 10 s to 51 s per step)`).

- [ ] **Step 2: Pin today's access proposal**

In the worktree, before any source edit, write `tests/data/scoring/arterial_access_ref.json` = `{"access_displacement_snap": {"wkt": sorted(g.wkt for g in roads.geometry)}}` from a one-off snippet (1 process, < 1 min) proposing on `scoring_fixtures._block_1808()` with `GreedyArterialReblocker(realizer=SnapToBoundary(lam=2.0), objective=Access(), cost=Displacement(), n_anchors=32, max_anchors=128, top_k=8, max_roads=15, road_width_m=7.0, workers=1, engine=ShortlistEngine(k=512, threads=1))`.
Add `test_access_snap_proposal_wkt_unchanged` to `tests/methods/test_arterial_access_step.py`: the same proposal's sorted WKT equals the file's, exactly.
Run: `uv run pytest -n 0 --no-cov tests/methods/test_arterial_access_step.py -v` (1 process, < 1 min). Expected: PASS on today's code.

- [ ] **Step 3: Write the equivalence tests**

In `tests/methods/test_arterial_access_step.py` (fixtures from `tests.methods.test_arterial`: `_grid_block`, `_deep_block`, `_holed_block`, `_two_arm_block`; `scoring_fixtures._block_1808`):
- `_mismatches(block: Block, steps: int) -> tuple[list[str], int, int]`: runs the Access + Displacement + SnapToBoundary(2.0) greedy for `steps` commits (`engines._greedy_arterials`, workers 1), and at every step, for every candidate of `_candidate_chords(anchors, targets)` realized by the realizer, compares `scorer.at_step(net, realizer).of(real)` with the reference `scorer.value(_explode(unary_union([*([net.merged] if net.merged is not None else []), real]), block.crs, 2 * hw)) - scorer.value(net.roads)`; returns the mismatches (as strings naming block, step, chord), the count of non-live candidates and the count with gain > 0.
- `test_access_step_matches_the_full_repeel`: for 1808, `_deep_block()`, `_holed_block()` and `_two_arm_block(...)` at 3 steps each: no mismatch (exact `==`), at least one candidate with gain > 0 overall, and at least one non-live candidate overall (the fixture set's purpose; if no listed fixture yields one, add a fixture block whose snapped spur from an anchor starts at an interior boundary node, and assert the count on it).
- `test_access_step_lowers_a_stranded_parcel`: a grid block with one parcel separated from its neighbours and the street by more than `STREET_TOL`; a road reaching it: the incremental gain equals the reference and the parcel's depth goes from `n_parcels + 1` to 1.
- One fault test each, `test_access_step_guard_sees_always_live`, `..._no_min`, `..._streets_only`: with the fault patched in (below), `_mismatches` on the same blocks returns at least one mismatch.
  Faults: `always-live` patches `_AccessStep.live` to return True; `no-min` patches `_AccessStep.relaxed_burden` with a BFS that assigns `layer + 1` to every unvisited neighbour without comparing to the base depth; `streets-only` patches `_AccessStep.live` to test against the streets' union only, ignoring committed roads.

Run: `uv run pytest -n 0 --no-cov tests/methods/test_arterial_access_step.py -v` (1 process, ~1 min). Expected: FAIL (`_AccessStep` not defined).

- [ ] **Step 4: Implement `_AccessStep`**

In objectives.py: `Access.for_block` builds `parcel_tree = STRtree(list(adjacency.block.parcels.geometry))` (positional, the peel's order).
`_AccessBlock.at_step(committed, realizer)`, for a snapping realizer: `depths = parcel_access_layers(adjacency, committed.roads, unreached=past_every_parcel).to_numpy(dtype=np.int64)` once; `burden = int((depths ** 2).sum())`; `base = 1.0 - burden / base_burden` (`0.0` when `base_burden == 0.0`, as `value` does); `seed = street_connectivity(block.streets, committed.roads, adjacency.tol).seed_geom`, `shapely.prepare`d.
Assert, naming the block, that every committed segment is within `tol` of `seed` (every committed road had a positive gain, so it was street-connected, and liveness is never lost; the gain's exactness rests on it).
`of(real)`: `0.0` if `base_burden == 0.0` or not `live(real)`; else `(1.0 - relaxed_burden(seeded(real)) / base_burden) - base`, the same float arithmetic as `value(trial) - base`, because burdens are exact integers.
`live(real)`: `seed is not None and shapely.dwithin(real, seed, tol)` (a road's touch-component reaches the street iff the road is within `tol` of the street or of a committed road, all of which are live).
`seeded(real)`: `parcel_tree.query(real, predicate="dwithin", distance=tol)`, kept where `depths > 1`, sorted.
`relaxed_burden(seeds)`: copy the base depths, set the seeds to 1, breadth-first from the seeds over `neighbours`, lowering a neighbour only when `layer + 1` is below its current depth; return `burden` minus the sum of `old² - new²` over the lowered parcels.
Delete `_Unioned` and `primitives._union_with`; move `scripts/perf/snap_vs_peel.py` to `unary_union([real])`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -n 0 --no-cov tests/methods/test_arterial_access_step.py tests/methods/test_arterial.py tests/methods/test_arterial_strategies.py tests/methods/test_engines.py -k "not parallel" -v` (1 process, ~5 min). Expected: PASS, the pinned WKT among them.
With the caller's go-ahead for a few seconds of 16 forks: the same files with `-k parallel` (the pool path inherits `_AccessStep` by copy-on-write). Expected: PASS.

- [ ] **Step 6: The real-block guard `gain`**

`arterial_check.py gain <ids,> <baseline-root> [fault]`: per block, the stored `<baseline-root>/scratch/networks/<id>/greedy_arterial_access_displacement.ordered.parquet`; at S in (0, 50, 150, 300) segments (capped at the network's length) its first S roads are the committed network (`engines._step_network`); the step's shortlist exactly as `_greedy_shortlist` builds it (the preset from `road_rescore.lineup_arms()[ARTERIAL]`, `FirstOrder(512, threads=1)`); every shortlisted chord realized and scored both ways as in Step 3.
A mismatch fails unless every parcel whose depth differs lies within 1e-9 m of `tol` (a knife-edge FP difference: counted and printed, accepted).
It prints per S: candidates, non-live, gain > 0, accepted FP differences, and ms per candidate for the reference and the incremental gain. Under a fault it stops at the first failure.
Faults as in Step 3. Blocks: ZAF.9.3.1_1_20269 and ZAF.9.3.1_1_38616.
Run: `both-ways gain ZAF.9.3.1_1_20269,ZAF.9.3.1_1_38616 $BASE` (1 process, ~15 min, `run_in_background`). Expected: the guard passes; each fault fails.
If `always-live` goes unseen on both blocks, add the study's other blocks (Task 4) until it is seen; if none shows it, stop and report to the caller: the branch is then defended by the pytest fixture alone, the owner's call.
Expected in the print: the incremental gain under 1 ms per candidate at every S, against 9 to 51 ms for the reference.

- [ ] **Step 7: The same commits as today, timed per step**

`arterial_steps.py steps <out> <id> <max_roads>`: proposes with `dataclasses.replace(road_rescore.lineup_arms()[ARTERIAL], max_roads=<max_roads>)`, under `StepLog`, writing `<out>/steps_<id>.parquet` after every step (a killed run keeps its rows). It imports `common` and `road_rescore` from `research/roadless` relative to the working directory, so the same file runs against either tree.
`compare <dir_a> <dir_b> <ids,>`: per block, the steps whose `commit_wkb` agree, the first that does not (with both trees' top-two gains there), and the median and total per-step time ratio.
Run, both at once (2 processes, ~30 min, `run_in_background` each): from `.worktrees/arterial-base`, `PYTHONPATH=. uv run python ../arterial-speed/research/roadless/arterial_steps.py steps <scratch>/before ZAF.9.3.1_1_20269,ZAF.9.3.1_1_6310,ZAF.9.3.1_1_39240 40`; from `.worktrees/arterial-speed`, the same into `<scratch>/after` (`<scratch>`: this session's scratchpad).
Then `compare <scratch>/before <scratch>/after ZAF.9.3.1_1_20269,ZAF.9.3.1_1_6310,ZAF.9.3.1_1_39240` (1 process, seconds).
Expected: 40 of 40 commits identical on each block; per-step time down by at least 2x (the exact evaluation, 8.8 of 10.4 s at S = 0, falls ~10x; ranking does not move).
A differing commit is a failure unless the two trees' top gains differ by FP only and the guard's accepted FP list explains it; report it either way.

- [ ] **Step 8: Gates and commit**

Run the gates (Global Constraints) on everything touched. Commit `objectives.py`, `primitives.py`, `snap_vs_peel.py`, the tests, the JSON, `arterial_steps.py`, `arterial_check.py` (`research: roadless -- the arterial's access gain scored incrementally against the step's peel: <n> commits identical on 3 blocks, a step <x>x faster at S <= 80, the gain <y> ms -> <z> ms per candidate at 300 segments`) and push.
Remove the base tree: `git worktree remove .worktrees/arterial-base`.

### Task 2: The anchor sampler as a Strategy, and a cap that holds

**Files:**
- Create: `src/reblock/methods/arterial/anchors.py`
- Modify: `src/reblock/methods/arterial/{primitives,engines,policies,reblocker,__init__}.py`
- Modify: every conf entry targeting `reblock.methods.arterial.GreedyArterialReblocker` (`grep -rln "GreedyArterialReblocker" conf`)
- Create: `tests/methods/test_arterial_anchors.py`; modify the tests and `scripts/perf/*` that `uv run mypy` names
- Modify: `research/roadless/arterial_check.py` (guard `anchors`)

**Interfaces:**
- Consumes: Task 1's `_AccessStep` (unchanged) and `tests/data/scoring/arterial_access_ref.json`.
- Produces: `AnchorSampler` (runtime_checkable Protocol): `identity -> AnchorIdentity` and `points(network: Sequence[BaseGeometry]) -> list[tuple[float, float]]`.
- Produces: `Uncapped(n: int)` (every network vertex plus `n` arc-length samples: today's `max_anchors: 0`); `PerLineFallback(n: int, max_anchors: int)` (today's capped `_anchor_points`, moved unchanged: the uncapped family if it fits, else per-line arc-length samples, ~`max_anchors` plus one per line, whichever is smaller); `HardCap(n: int, max_anchors: int)` (the uncapped family if it fits, else at most `max_anchors` points at one global arc-length stride over the merged network). All frozen; `__post_init__` raises ValueError for `n < 0`, and for `max_anchors < 1` on the two capped ones. `AnchorIdentity: TypeAlias = Uncapped | PerLineFallback | HardCap`.
- Produces: `GreedyArterialReblocker.anchors: AnchorSampler` in place of `n_anchors` and `max_anchors`; the engine functions, `ArterialEngine.run` and the three engine classes take `anchors: AnchorSampler` in their place; `CandidatePolicySpec.build(adjacency, streets, anchors: AnchorSampler, top_k: int)`.

- [ ] **Step 1: Write the sampler tests**

In `tests/methods/test_arterial_anchors.py`:
- `_cap_violations(sampler: AnchorSampler, networks: list[list[BaseGeometry]]) -> list[str]`: every network where `len(points) > max_anchors`, or the points are unsorted or repeated.
- `test_hard_cap_never_exceeds_max_anchors`: networks of 60 disjoint 3 m segments plus one 200 m line, and a 40-vertex zigzag; `HardCap(n=32, max_anchors=16)` has no violation, while `PerLineFallback(n=32, max_anchors=16)` returns more than 16 on the 61-line network (the measured bug, pinned as a fact of the kept sampler).
- `test_hard_cap_guard_sees_per_line_restart` (the fault): with `anchors._global_samples` patched to the per-line `_arclength_samples`, `_cap_violations` for HardCap is non-empty.
- `test_hard_cap_is_the_uncapped_family_when_it_fits`: on a network whose family has fewer than `max_anchors` points, `HardCap`, `PerLineFallback` and `Uncapped` with the same `n` return the same list.
- `test_hard_cap_spreads_by_arc_length`: on one straight 100 m line with `Uncapped(32)` over the cap, `HardCap(n=32, max_anchors=10)` returns 10 points 10 m apart (offset half a stride).
- `test_capped_samplers_refuse_a_zero_cap`: `HardCap(32, 0)` and `PerLineFallback(32, 0)` raise ValueError.
- `test_samplers_split_the_identity`: the three samplers with equal numbers give three identities and three proposal ids.
Move the anchor tests of tests/methods/test_arterial.py here, rewritten as `PerLineFallback(...).points(...)` and `Uncapped(...).points(...)`.
Run: `uv run pytest -n 0 --no-cov tests/methods/test_arterial_anchors.py -v` (1 process, < 1 min). Expected: FAIL (no module `anchors`).

- [ ] **Step 2: Implement the samplers**

Move `_merged_lines`, `_arclength_samples`, `_vertices_and_samples` and `_anchor_points`' body from primitives.py into anchors.py (`PerLineFallback.points` is `_anchor_points` with its docstring and measured note; `_anchor_points` is deleted).
`HardCap.points`: the uncapped family if `len <= max_anchors`, else `_global_samples(lines, total, max_anchors)`: positions `(j + 0.5) * total / max_anchors` for `j` in `range(max_anchors)`, walked along the lines in `_merged_lines`' order, each `_rnd`-snapped, the set sorted.

- [ ] **Step 3: Inject the sampler**

Replace `n_anchors`/`max_anchors` with `anchors` through engines.py, policies.py (`_seed`, `_FaithfulPolicy` keeps `anchors`) and reblocker.py (the field comment says which sampler is which); export the four names from `__init__.py`.
Conf: `n_anchors: 32` + `max_anchors: 128` becomes `anchors: {_target_: reblock.methods.arterial.PerLineFallback, n: 32, max_anchors: 128}`; `n_anchors: 32` + `max_anchors: 0` becomes `anchors: {_target_: reblock.methods.arterial.Uncapped, n: 32}`. The compare_config comment on `max_anchors: 128` says it is now `PerLineFallback`'s cap, which the measured anchor counts exceed (112 to 525 on 20269).
Move every caller `uv run mypy` names (the tests, scripts/perf's engine calls and reblocker constructions) to a sampler with the same numbers.
Run: `uv run mypy` (1 process, ~3 min). Expected: no error.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -n 0 --no-cov tests/methods/test_arterial_anchors.py tests/methods/test_arterial_access_step.py tests/methods/test_arterial.py tests/methods/test_arterial_strategies.py tests/methods/test_engines.py tests/methods/test_policies.py tests/test_code_closure.py -k "not parallel" -v` (1 process, ~5 min). Expected: PASS; the pinned access WKT (now built with `PerLineFallback(32, 128)`) unchanged, which is the proof the move changed no proposal.

- [ ] **Step 5: The real-block guard `anchors`**

`arterial_check.py anchors <ids,> <baseline-root> [fault]`: on 20269's stored ordered network at S in (0, 150, 300, 450, 607): `PerLineFallback(32, 128)` gives 112, 229, 323, 409, 525 anchors (`before/micro_20269.csv`; the move preserved them) and `HardCap(32, 128)` at most 128 at every S. Fault `per-line`: `anchors._global_samples` patched to the per-line samples, so HardCap exceeds 128 at S 607.
Run: `both-ways anchors ZAF.9.3.1_1_20269 $BASE` (1 process, < 1 min). Expected: the guard passes; the fault fails.

- [ ] **Step 6: Gates and commit**

Gates, then commit (`research: roadless -- the arterial's anchor sampler a Strategy (Uncapped, PerLineFallback, HardCap); HardCap holds its cap where today's reaches 525 anchors at a cap of 128 on 20269`) and push.

### Task 3: A displacement budget that stops the greedy

**Files:**
- Create: `src/reblock/methods/arterial/stop.py`
- Modify: `src/reblock/methods/arterial/{engines,reblocker,__init__}.py`
- Modify: every conf entry targeting `GreedyArterialReblocker`
- Create: `tests/methods/test_arterial_stop.py`; modify the arterial tests and scripts/perf callers mypy names
- Modify: `research/roadless/road_rescore.py` (`ARTERIAL_MAX_ROADS` and its override deleted; `lineup_rows`; the docstrings that say 300)

**Interfaces:**
- Consumes: Task 2's `AnchorSampler`.
- Produces: `DisplacementMeter` with `of(block: Block, half_width_m: float) -> DisplacementMeter` (classmethod), `add(real: LineString) -> None` (the road buffered by `half_width_m` into a `reblock.buildings.IncrementalOverlap`), `share() -> float` (`total() / max(len(block.building_geometries), 1)`, cycle_native's denominator).
- Produces: `Stop = Literal["max_displacement", "no_gain", "max_roads"]`; `GreedyRun(NamedTuple)`: `roads: GeoDataFrame`, `commits: int`, `stop: Stop`.
- Produces: every engine function and `ArterialEngine.run` take `max_displacement: float` and return `GreedyRun`; before each step a loop stops with `max_displacement` if `meter.share() >= max_displacement`, else with `max_roads` if `len(committed) >= max_roads`; a step with no improving candidate stops with `no_gain`; after each commit, `meter.add(real)`.
- Produces: `GreedyArterialReblocker`: after its fields, `_: KW_ONLY` and `max_displacement: float`; `__post_init__` raises ValueError unless `0 < max_displacement <= 1`; `params` gain `"commits"`, `"stop"`, `"max_displacement"`.
- Produces: `road_rescore.lineup_rows()` = `ROADLESS / "road_rows" / f"lineup-{sha256(repr((free_roads.lineup_free().identity, lineup_arms()[ARTERIAL].identity))).hexdigest()[:8]}"`.

- [ ] **Step 1: Write the stop tests**

In `tests/methods/test_arterial_stop.py` (Access + Displacement + SnapToBoundary on `_two_arm_block(...)` or another fixture block with buildings, `Uncapped(8)`, workers 1):
- `test_meter_share_matches_budget_displacement`: after adding k committed roads, `meter.share()` equals `reblock.budget.displacement(block.buildings, _planarize(roads, crs, 2 * hw)) / max(len(block.building_geometries), 1)` within 1e-9.
- `test_displacement_stop_is_a_commit_prefix`: an unbounded run (`max_displacement=1.0`, `max_roads=6`) with `engines._best_candidate` wrapped to record each commit; with `d` set to the share after its third commit, a run at `max_displacement=d` returns `commits == 3`, `stop == "max_displacement"` and roads equal to `_planarize(commits[:3], ...)` (exact WKT), so the boundary `share == d` stops.
- `test_displacement_stop_guard_sees_a_deaf_meter` (the fault): with `DisplacementMeter.add` patched to a no-op, the previous test's run does not stop at 3.
- `test_stop_reasons`: `max_roads=2` on a block that would commit more gives `stop == "max_roads"`, commits 2; an unbounded run on a small block that saturates gives `"no_gain"`.
- `test_a_run_with_no_gain_commits_nothing`: on a block where no candidate gains (every parcel already fronts a street), `GreedyRun(stop="no_gain", commits=0)` and empty roads carrying `drain`.
- `test_no_buildings_never_stops_on_displacement`: `tests.block_fixtures.no_buildings`: the run ends on `no_gain` or `max_roads`.
- `test_max_displacement_splits_the_identity` and `test_max_displacement_out_of_range_is_refused` (0.0 and 1.5 raise ValueError).
Run: `uv run pytest -n 0 --no-cov tests/methods/test_arterial_stop.py -v` (1 process, ~1 min). Expected: FAIL (no module `stop`).

- [ ] **Step 2: Implement the meter and the stop**

`stop.py` as in Interfaces; the three engine loops and the reblocker as in Interfaces (the lazy engine's loop too). `propose` reads `GreedyRun`'s fields by name.
Conf: every arterial entry gains `max_displacement: 0.20`. The access pair's `max_roads: 60` becomes `max_roads: 1000`, its comment rewritten: `max_displacement` is the budget (count-weighted, as cycle_native's and resistance_lp's), `max_roads` a safety cap expected never to bind, the stop reason in every proposal's params. Directness presets keep `max_roads: 15` as their budget (0.20 cannot bind at 15 roads: they end at 0.5 to 2.6% displacement).
Every test or script constructing the reblocker passes `max_displacement` (`1.0` where a test pins a road-count-limited output: displacement cannot exceed 1, so it never binds; say so in a comment). The pinned access WKT test uses `max_displacement=1.0`.
road_rescore.py: delete `ARTERIAL_MAX_ROADS` and its `replace` in `lineup_arms` (the preset now carries its budget); `lineup_rows` as in Interfaces, its docstring saying the root changes whenever the arterial's identity does.

- [ ] **Step 3: Run the tests**

Run: `uv run mypy` (1 process, ~3 min), then `uv run pytest -n 0 --no-cov tests/methods/test_arterial_stop.py tests/methods/test_arterial_anchors.py tests/methods/test_arterial_access_step.py tests/methods/test_arterial.py tests/methods/test_arterial_strategies.py tests/methods/test_engines.py tests/methods/test_policies.py tests/test_code_closure.py -k "not parallel" -v` (1 process, ~5 min). Expected: no type error; PASS, the pinned WKT unchanged.
With the caller's go-ahead: the same files with `-k parallel` (seconds, up to 16 forks). Expected: PASS.

- [ ] **Step 4: Gates and commit**

Gates (road_rescore.py: `--select F,E9`, and its mypy error count not risen), then commit (`research: roadless -- the arterial stops at a required max_displacement (0.20, count-weighted, the crossing road kept) and records why it stopped; max_roads a safety cap at 1000 on the access pair; the lineup's rows root hashes the arterial's identity`) and push.
The lineup's rows now sit under an old root: nothing in this branch reads them until Task 7.

### Task 4: The study's runner and report

**Files:**
- Create: `research/roadless/arterial_study.py`

**Interfaces:**
- Consumes: `arterial_steps.StepLog`; `road_rescore.lineup_arms`, `ARTERIAL`, `Sampler`, `net_path`, `proposal_row`, `PrefixJob`, `prefixes_one`, `_BLOCKS`, `cx_network`, `cx_dir`, `prefix_dir`, `write_parquet`; `gap_clear.complex_of(metrics.RUNTIME, factorization=gap_factor.factorization_of("cpu"))` (as `lineup_one` builds it).
- Produces: `Arm(NamedTuple)`: `anchors: AnchorSampler`, `max_displacement: float`, `max_roads: int`; `ARMS: dict[str, Arm]` = `cur02` (`PerLineFallback(32, 128)`, 0.20, 1000), `cap02` (`HardCap(32, 128)`, 0.20, 1000), `smoke` (`HardCap(32, 128)`, 0.20, 5).
- Produces: `method_of(arm: Arm) -> GreedyArterialReblocker` (`dataclasses.replace(lineup_arms()[ARTERIAL], anchors=..., max_displacement=..., max_roads=...)`) and `network_name(arm_name: str, method) -> str` (`f"arterial-{arm_name}-{sha256(repr(method.identity)).hexdigest()[:8]}"`).
- Produces: `arterial_study.py run <rows> <id> <arm> [--replace]` and `report <rows> <baseline-root> <ids,>`.

- [ ] **Step 1: `run`**

Single-threaded check (road_rescore's `_check_single_threaded`); build the block; propose under `Sampler` and `StepLog` directly (`method.propose`, never the derive cache: a cache hit would carry no steps); write the network to `net_path(<rows>/scratch, id, name, False)`, the steps to `<rows>/steps/<id>__<name>.parquet`, and a proposal row to `proposal_row(<rows>, id, name, False)` with `wall_s, cpu_s, peak_rss_gb, commits, stop, n_roads, road_m, host`; then `prefixes_one(PrefixJob(<rows>, <rows>/scratch, id, name))` and `cx_network(<rows>, <rows>/scratch, metric, block, name, replace=...)`.
An existing proposal row refuses the run unless `--replace` (`clearings.check_new`).
`<rows>` is `research/roadless/arterial_rows/study`.

- [ ] **Step 2: `report`**

Reads the study's rows and the baseline's (`<baseline-root>/proposals`, `prefix_dir`, the cx rows under `cx_dir(<baseline-root>, metric)`, the network `greedy_arterial_access_displacement`) and writes to `research/roadless/arterial_rows/`:
- `summary_runs.csv`: block, area_km2, net (baseline, each arm), wall_s, commits, n_roads, stop, step_s median/p90/max, anchors max, candidates max, the count share at the stop, the whole network's D_count and D_area.
- `summary_lenses.csv`: block, net, budget (each of BUDGETS and full), m, D_area, reach (`D_area >= budget - 1e-9`), L_P, L_Pq, L_J2, enclosed, and each lens minus the baseline's at that budget.
- `summary_prefixes.csv`: for `cur02` against the baseline (the same greedy, stopped earlier), per block and budget: both prefixes' m, the symmetric difference of their road unions in metres and as a share of the baseline prefix's length, and whether they are identical; this is how far canonical_order's re-ordering moves the scored prefixes.
- Two fidelity checks, printed and in `summary_runs.csv`: `cur02`'s network lies inside the baseline's (its union's length outside the baseline union buffered by 1e-6 m is 0: a commit-order prefix of the same greedy), and where `cur02` stopped on `no_gain` its network equals the baseline's (`geom_equals_exact`, tolerance 0, as road_rescore's `same_as_cached`).
- The verdict of Ruling 7 for the two samplers, printed with its six medians, the blocks ahead and behind for each, and the wall ratios.

- [ ] **Step 3: A smoke run, then gates and commit**

Run: `PYTHONPATH=. uv run python research/roadless/arterial_study.py run research/roadless/arterial_rows/study ZAF.9.3.1_1_6310 smoke` (1 process, ~6 min with scoring, `run_in_background`), then `report research/roadless/arterial_rows/study $BASE ZAF.9.3.1_1_6310` (1 process, < 1 min). Expected: 5 commits, `stop == "max_roads"`, a steps file of 5 rows, and lens rows at every budget the 5 roads reach (the fidelity checks apply to `cur02` only; for `smoke` the report prints its run and lenses).
Delete the smoke rows (`rm -r research/roadless/arterial_rows/study`, untracked), run the gates, commit `arterial_study.py` and push.

### Task 5: Run the study and decide the presets

**Files:**
- Create: `research/roadless/arterial_rows/summary_{runs,lenses,prefixes}.csv`

**Interfaces:**
- Consumes: Task 4's `run` and `report`; the arms `cur02` and `cap02`.
- Produces: the verdict (Ruling 7), each arm's walls and stops, the prefix movement, and an extrapolation of the lineup migration's task-hours (Task 7).

Blocks: six lineup blocks of 0.08 to 0.46 km² with stored current-code networks and lens rows under `$BASE`.

| block | km² | buildings | stored wall | stored D count / area | cur02 expected | cap02 expected |
|---|---|---|---|---|---|---|
| ZAF.9.3.1_1_6310 | 0.084 | 1090 | 4.0 h | 0.46 / 0.45 | ~40 min | ~15 min |
| ZAF.9.3.1_1_20269 | 0.103 | 1046 | 2.8 h | 0.31 / 0.34 | ~30 min | ~10 min |
| ZAF.9.3.1_1_40144 | 0.153 | 1083 | 4.3 h | 0.43 / 0.46 | ~40 min | ~15 min |
| ZAF.9.3.1_1_39240 | 0.208 | 1081 | 3.0 h | 0.51 / 0.34 | ~25 min | ~10 min |
| ZAF.9.3.1_1_8480 | 0.387 | 1021 | 5.5 h | 0.08 / 0.12 | ~2.5 h | ~40 min |
| ZAF.9.3.1_1_38616 | 0.461 | 1347 | 12.7 h | 0.34 / 0.25 | ~2 h | ~30 min |

The expected walls rest on the per-step model after Task 1 (~4.6 s plus ~0.036 s per committed segment with today's sampler, ~5 s flat with `HardCap`) times the commits to the stop, plus ~5 min of scoring; Task 1's measured speedup replaces the model before submitting.
8480 saturates (every parcel at depth 1) at count D 0.08, so its `cur02` runs to `no_gain` and must reproduce the stored network: the end-to-end fidelity check, and the whole-run speedup of Task 1 (the stored walls' host is unrecorded: say so beside the ratio).

- [ ] **Step 1: Submit**

Commit and push everything (cluster_submit refuses a dirty tracked tree). Check orion's load (`ssh ai sinfo -N -p defq -o '%N %c %T %C'`, `scontrol show node orion`).
Run: `uv run python research/roadless/cluster.py submit cpu arterial-study --node orion --cpus 1 --mem 6 --items cur02,cap02 --time 8:00:00 ZAF.9.3.1_1_6310,ZAF.9.3.1_1_20269,ZAF.9.3.1_1_40144,ZAF.9.3.1_1_39240,ZAF.9.3.1_1_8480,ZAF.9.3.1_1_38616 -- research/roadless/arterial_study.py run research/roadless/arterial_rows/study {id} {item}` (locally 1 process, ~2 min to write the bank; on orion 12 tasks of 1 CPU and 6 GB, the scoring's measured peak 2.5 to 4.2 GB; ~10 task-h, ~3 h wall, the longest 8480's `cur02`).
Then `cluster.py wait arterial-study --timeout 10h` in `run_in_background`, and `cluster.py sync arterial-study` when it exits 0 (on 3 or 4, `resubmit`, and report which tasks failed and why).
If the owner chose local (Ruling 6): the same `run` commands, 2 at a time once the caller confirms 2 free slots, `cap02` first (~1.5 h for the six), then `cur02` (~4.5 h).

- [ ] **Step 2: Report and decide**

Run: `PYTHONPATH=. uv run python research/roadless/arterial_study.py report research/roadless/arterial_rows/study $BASE <the six ids>` (1 process, ~2 min).
Expected: both fidelity checks pass on every block (8480's `cur02` identical to the stored network); no `max_roads` stop (a `max_roads` stop is reported to the caller before anything else, as it means the safety cap binds).
Record, for the NOTES: each arm's wall against the stored wall; the per-step curve; the verdict with its numbers; how far the `cur02` prefixes moved from the full networks' at each budget, and how many area budgets went unreached under the count-weighted stop (Ruling 1).
Extrapolate the migration (Task 7): fit wall against area (and buildings) over the six blocks for the arm the presets will use, and give the 62 lineup blocks' total task-hours, the oversized ones (up to 52 km²) named as the uncertain part.
Commit the three summary CSVs (`research: roadless -- the arterial study on six blocks: cur02 <a> h and cap02 <b> h against the stored <c> h; <verdict>`) and push.

### Task 6: The presets, and the records

**Files:**
- Modify: `conf/compare_config.yaml` (the access arterials), `src/reblock/methods/arterial/anchors.py` and the conf/tests that name a deleted sampler (only if one is dominated)
- Modify: `research/roadless/NOTES.md`, `research/roadless/BACKLOG.md`

**Interfaces:**
- Consumes: Task 5's verdict.
- Produces: the named presets.

- [ ] **Step 1: Set the presets the verdict gives**

If neither sampler dominates (Ruling 7): `greedy_arterial_access_displacement` keeps `PerLineFallback(32, 128)` (quality-first) and a new `greedy_arterial_access_displacement_fast` takes `HardCap(32, 128)` (speed-first), each with a comment of what it optimizes and gives up, with the study's numbers; the repulsion twin keeps the default's sampler (Ruling 9).
If `HardCap` dominates: delete `PerLineFallback`, its tests and its guard's case; every conf entry and test using it moves to `HardCap` with the same numbers (migrate, no shim); one preset.
If `PerLineFallback` dominates: delete `HardCap` the same way.
The lineup keeps `greedy_arterial_access_displacement` (the quality-first operating point) unless the owner says otherwise.
Run: `uv run mypy` and `uv run pytest -n 0 --no-cov tests/methods/test_arterial*.py tests/methods/test_engines.py tests/methods/test_policies.py -k "not parallel"` (1 process, ~8 min). Expected: no error; PASS.

- [ ] **Step 2: Write the records**

NOTES.md, a new section after the last (`### The arterial: an incremental gain, a cap that holds, a displacement budget (2026-10-18)`, column-wrapped at 99): the measured step costs before; Task 1's identical commits and speedup; the anchor counts under each sampler; the stop rule and its rulings; the study's table, verdict and prefix movement; the migration's estimate.
BACKLOG.md: the item that names the arterial's `max_roads: 60` keeping it short of Lens A (BACKLOG 4(b)'s "Open:" sentence) and the lineup's "arterial at 300 roads" lines updated to the new budget, and a line for any idea the study left open (for example a `HardCap` that keeps committed-road junctions, if `HardCap` lost quality), each with the change that would put it back on the frontier.
A deleted sampler goes to the wiki's dead-end record with its numbers: load the `wiki` skill, read `~/wiki/AGENTS.md`, and add the entry under `~/wiki/pages/reblock/` (reblock is GeoffChurch/REU_Slum_Optimization).

- [ ] **Step 3: Gates and commit**

Commit conf, any deleted sampler's files, NOTES.md and BACKLOG.md (`research: roadless -- the arterial's presets from the study: <presets>`) and push; commit the wiki page in `~/wiki` per its AGENTS.md.

### Task 7 (owner's yes only, after the consolidated pass returns): merge and migrate the lineup's rows

**Files:**
- Merge: research/arterial-speed into research/roadless (in the main checkout, once the other agent's pass has returned and its tree is clean)
- Move: `research/roadless/road_rows/lineup-caf14fa2/` to `research/roadless/road_rows/lineup-<new digest>/` (untracked data)

**Interfaces:**
- Consumes: Task 3's `lineup_rows()`; Task 5's task-hour estimate; Task 6's presets.

- [ ] **Step 1: Ask**

Report to the caller Task 5's estimate for re-proposing the arterial on the lineup's 62 blocks and wait for the owner's yes; do nothing below without it.

- [ ] **Step 2: Merge and move**

In the main checkout: `git merge --no-ff research/arterial-speed`, the gates, `git push origin research/roadless`.
Move the old root to `lineup_rows()`'s new name; delete from it the arterial's rows only (`proposals/*__greedy_arterial_access_displacement*`, `prefixes_*/*__greedy_arterial_access_displacement.parquet`, the cx rows of that network, `scratch/networks/*/greedy_arterial_access_displacement*`), so no row of the old arterial reads as the new one's.

- [ ] **Step 3: Re-propose and report**

Re-propose the arterial on the lineup's blocks with road_rescore's `lineup-one` through cluster.py `submit cpu` on orion (the task-hours and `--time` from Task 5's fit; `--mem` from the study's measured peaks), wait with `cluster.py wait` in `run_in_background`, sync, then `PYTHONPATH=. uv run python research/roadless/consolidated.py lineup-report` (1 process, ~5 min).
Commit the regenerated `consolidated_rows/summary_lineup_*.csv` and the NOTES lines on what moved, and push research/roadless.
