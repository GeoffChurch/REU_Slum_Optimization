# Pruning through epistemetry, seqscreen deleted: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** reblock's screens (consolidated `screen-report`) decide through GeoffChurch/epistemetry, pinned at 165f98b, by a preset `Screen` built once at the edge from a decision-rule Strategy; seqscreen.py and epireplay.py are deleted, every caller moved and every stored decision re-derived.

**Architecture:** A new `research/roadless/screening.py` holds what is still reblock's: the rule Strategy (`Rule`: `RealizedLoss`, the default, and `ZeroOne`, whose per-arm target is `OwnStake`), the adapter from a live screen's rows to epistemetry's `Arrived`, and the reading of a `Stopped` record as summary_screen.csv's stop.
A screen's configuration and identity is epistemetry's `Screen(policy=ValueOfMoreUnits(...), drop=..., seed=...)`, and `Screen.record()` is the run's identity.
The live loop is epistemetry's `advance`: a wave passes the screen's stored decisions back as its ledger, and `Running.want` names the next blocks.
consolidated.py's edge builds the named presets (`screen_presets()`) and injects one into `screen-report`, which never asks which rule it holds.
common.py holds the block sets (TUNING, and HELD in the order the screens run it).
`research/roadless/screening_check.py` holds the guards, each failing under a named fault, the acceptance runs and the statistical comparison of two sets of them.
A temporary `research/roadless/screening_migration.py` puts the fourth replay's stored runs in the acceptance runs' shape for that comparison before anything is deleted.

**Tech Stack:** Python 3.12 (uv), numpy, pandas, GeoffChurch/epistemetry 165f98b (`design.screen`, `design.replay`, `design.decide`, `stats.records`).
research/roadless has no test framework: the guards follow gap_clear_check.py's convention (a guard returns its failures; under a fault it must fail, or the run exits non-zero saying the fault went unseen).

**Spec:** research/roadless/NOTES.md, "Pruning by epistemetry: the realized loss as the target" (the fourth replay: its rule, numbers, guards and feedback); research/roadless/BACKLOG.md, "Pruning by epistemetry's decision layer"; epistemetry's docs/superpowers/specs/2026-10-18-replay-driver-and-records.md (approved at 29ff53d, built by 165f98b), §10.3 reblock's part, except its line that runs Task 4 at 452052c, which the owner superseded on 2026-10-18 (Task 4 compares statistically after the move); the owner's decision of 2026-10-17: the consolidated pass prunes with seqscreen (its screens are done), then reblock moves to epistemetry and seqscreen is deleted, so that epistemetry is dogfooded by several projects rather than shadowed by a second decision layer inside reblock.

## Global Constraints

Copied from the brief; every task's requirements include this section.

**Sequencing**
- Execution starts only after the consolidated pass returns.
  Until then another agent edits NOTES.md, BACKLOG.md, consolidated.py and the consolidated_rows CSVs; locate every edit by its function name or heading, never by a line number from this plan.
- Execution is subagent-driven (superpowers:subagent-driven-development), implementers and reviewers on opus.

**No legacy shims**
- When a format changes, migrate the data and delete the old path.
- Never keep a reader for a format no longer written.

**Strategy pattern**
- Resolve a choice once, upstream (a Protocol plus an injected instance).
- No string registries or enum switches for a fixed set.
  The one name lookup is the command line's preset, resolved once in `consolidated.main` (a CLI argument: dynamic access with no default).

**No closed-set dynamic access**
- No getattr/.get defaults on closed sets; frozen dataclasses and NamedTuples.
- Result-changing parameters are required and keyword-only below the edge, and part of the run's identity (here `Screen.record()`; an acceptance run's order seed is stored in its row).
- Every reblock dataclass is `frozen=True, kw_only=True`; every reblock `Recordable` declares `namespace="reblock"`.
- JSON (the fixture, the decision records) is read by key, with no default, at one boundary function each.

**Keep the frontier, delete the dominated**
- seqscreen is not dominated on every axis by the numbers: the replays tie it within noise (the fourth replay: loss below it at every rate, one wrong call and four asks more in 2,600 pairwise runs).
- It is deleted by the owner's decision to dogfood epistemetry, not by dominance.
  The NOTES say so plainly.

**Guards**
- Every guard is shown to fail under fault injection: run it as it is (passes), then under each of its faults (fails), with `both-ways`.
- Guards run from the repo root with `PYTHONPATH=.`, single-threaded: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1`.
- A fault is a patch inside the guard's process (gap_clear_check.py's `_patched`); no file is edited to inject one.

**Repo and process rules**
- Commit with `git -c core.hooksPath=.githooks commit -m "<subject>" -m "<body>" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RoNpoEMZ1kz2hogCsAsmXJ" -- <paths>` (a new file is `git add`ed first; a deletion is `git rm`).
- Subjects follow the branch's style: `research: roadless -- <what, with its numbers>`.
- Push after each commit: `git push origin research/roadless`.
  Never push main.
- Never touch research/roadless/stopping_review/, research/group_steiner/, research/roadless/road_rows/, the untracked CSVs in research/roadless/consolidated_rows/, or research/roadless/{ids58,ids59,ov16,ov19,ov_a,ov_b,ss_miss}.txt.
- Don't edit src/reblock. ~/src/epistemetry and ~/src/cluster_submit are read-only (read epistemetry with `git -C ~/src/epistemetry show 165f98b:<path>`).
- At most 8 local processes in total; single-threaded numeric pools; kill by PID; no polling loops.
  A run longer than a few minutes goes through the Bash tool's `run_in_background`, which notifies on exit.
- Never `uv cache clean`: every checkout's .venv links into uv's cache (2c7ccef2).
- NOTES.md and BACKLOG.md are column-wrapped at 99; this plan and other new Markdown is one sentence per line.
- Ruff skips research/ (pyproject `extend-exclude`), so the gate is explicit paths: the two new modules pass `uv run ruff check research/roadless/screening.py research/roadless/screening_check.py` (the repo's rules, lines of 100); every touched existing module passes `uv run ruff check --select F,E9 <path>` (check it passes before editing).
- Types: `MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent <paths>` reports no error in screening.py or screening_check.py; for a touched existing module the count of errors it reports does not rise (record it before editing).

## Rulings

1. **Target commit 165f98b31c7114ab9fc3b902c9dfc79a67623d99** (2026-10-18), which lands GeoffChurch/epistemetry #1-#4 (and #5-#8 before them): the live driver `advance` with `replay` built on it, `ValueOfMoreUnits` recordable configuration, `Step`/`Judgement`/`Decision`/`Running`/`Stopped`/`Screen` recordable, per-unit covariates on the `Design` read by `ProportionalTo`, every dataclass keyword-only, records namespaced, and every draw keyed by a seed and its place (`design.screen.stream`).
   A newer commit is a later pin move, checked by the `acceptance` guard (Task 6).
2. **The pin lives in pyproject.toml**, as cluster_submit's does: epistemetry joins the `research` dependency group with `[tool.uv.sources]` at the full SHA.
   Today's mechanism (`uv run --with "epistemetry @ git+...@674dba4"`, an ephemeral overlay named only in epireplay.py's docstring) goes with epireplay.py. epistemetry requires Python >=3.12 (still at 165f98b; GeoffChurch/epistemetry #9 asks why), so research/roadless's `requires-python` rises to >=3.12 (the venv is 3.12.13; main is untouched). cluster_submit mirrors every git source of uv.lock to the cluster before a run, so the nodes get epistemetry without GitHub access.
   Accepted by the owner (2026-10-17).
3. **Two presets, set at consolidated's edge.** `realized-loss` (the default: AdoptKeepAskByStake over the bracket's five geometric rates, equally weighted, over FUTURE uses; the lowest realized loss at every rate in the fourth replay).
   `fewest-wrong` (AdoptKeepAsk over the bracket's ends, each arm priced by its own stake; 37 wrong calls against 67 in 1,400 distinct pairwise runs (each design without the veto, C1m4/p64 with it too), at 318 asks against 284 and more loss at the bracket's bottom and middle).
   The second stays because the fourth replay puts it on the frontier (owner 2026-10-17).
   The nine-point sensitivity is no preset: asks are free (owner 2026-10-16), so no operating point wants fewest asks, and its numbers stay in NOTES.
   `SharedStake` is not ported: on one arm it is the own price, and reblock screens pairwise.
4. **Block sets in common.py as literals.** `common.TUNING` (13) and `common.HELD` (46) beside `common.large_blocks()`.
   HELD is written in the order the screens run its blocks: seqscreen's permutation of the sorted held-out set (`np.random.default_rng(0).permutation(46)`), frozen, because phase 5's rows were run in that order and are a prefix of it.
   At 165f98b `random_orders` draws its permutation from `stream(seed, PERMUTE, o)`, which no longer gives that permutation, so the order is data, not a seed.
   HELD stops being derived from the git-ignored raster-era large82_simp_vs_greedy.parquet.
5. **Deleted with seqscreen, no caller outside it:** its loaders (`simp`, `polished`, `searched`, `LOADERS`, `specs_of`), the `next` command, `question`, `ORDERS`, `STOPPED`, `POWER`.
   A live screen of a non-SIMP variant would need a runner like consolidated `screen`.
   The ask's number survives as summary_screen.csv's `break_even` column.
6. **The replays' data is deleted after the move** (owner 2026-10-17, overruling "keep"), but only once Task 4's comparison has passed: epireplay*.parquet, epireplay*_jobs/, epireplay*_screens.parquet and epireplay4_*.log (git-ignored, about 35 MB; Task 8 Step 5). epireplay_designs.json becomes screen_designs.json, reblock's real-data acceptance fixture: epistemetry's spec (owner, 2026-10-16) puts each project's real-data acceptance test in that project, run through its pin.
   Its five many-arm designs stay in it, read by nothing until a many-arm screen exists (BACKLOG, Task 9).
   The only stored seqscreen decisions anything reads are summary_screen.csv's, which Task 7 re-derives; screen_rows/ and the store's clearings/screen_rows CSVs hold the screens' runs, not decisions, and stay as they are.
7. **Numbers that move.** At 165f98b every draw is keyed by a seed and its place, so no replay reproduces the fourth replay's runs bit for bit (epistemetry's spec §5.2).
   Task 4 therefore compares summary rates statistically, run against stored run on the same order.
   Where a run draws nothing (a design no longer than the floor: its truth, and every run of C1m4/p64) it must be reproduced exactly.
   In phase 5, `.r2l2`'s stop may move (and the columns computed over the blocks a stop used); no call should; the five 15-block screens should not move (Task 7 Step 3).
8. **Unchanged modelling choices:** pairwise screens only (consolidated screens each variant against the default, as today); the base's GPU time charged 0, as seqscreen and every replay charged it; step 1 past a 15-block floor.
9. **One `Screen` per preset, whatever the population.** `horizons = tuple(range(1, len(common.HELD) + 1))`: `ValueOfMoreUnits` caps each horizon at the units left, so a screen of u <= 46 blocks values 1..left exactly as the fourth replay's `range(1, u + 1)` did.
   A fixture or a screen with more units than the horizons is refused (Task 3, `designs()`; Task 7, `screen_data`).
10. **A live screen continues from its stored ledger** (epistemetry's spec §5.3, §6.4): `screen-report` passes each variant's stored decisions back to `advance`, so a wave never re-values the decisions that chose the earlier waves.
    The ledger is stored with the blocks it was made on and the `Screen` record it was made under; a ledger whose screen differs from the injected preset, or whose blocks are not a prefix of the variant's rows, is refused before anything is decided.
    Within one pin, continuing equals deriving afresh (epistemetry's resume property), and the `screens` guard checks both.
    At a pin move whose streams differ, the phase-5 screens are re-derived afresh and the moved stops reported in the pin move's commit (question 1 below).
11. **GeoffChurch/epistemetry #10** (a replay charges a step unit by unit; a live parallel wave pays the whole step).
    It does not bite the adoption test of Task 4: both sides are replays charged unit by unit, and both presets share the floor's step, so neither the comparison nor Ruling 3's ranking can move by it.
    It bites the forecast of a live screen's cost where a veto falls inside a multi-unit step, which with step 1 is only the floor's first step (C1m4/p64 under the veto: about 6 blocks charged where a 12-block floor wave pays 12).
    Task 4 measures the gap on its runs (`wave_cost` beside `cost`) and Task 9 reports it, and comments it on #10 (the owner's yes, 2026-10-18).

**The owner's answers (2026-10-18):**
1. At a pin move whose random streams differ, the finished phase-5 screens are re-derived afresh (Ruling 10), with no guard branch for ledgers made at an older pin.
2. Task 9 Step 4 posts Task 4's `wave_cost` measurement as a comment on GeoffChurch/epistemetry #10.

## Review Focus

1. A held-out block with no default time in the pass's rows (the cost model's covariate would be NaN, and `Arrived` would refuse it without naming the block): `screen_data` raises naming the block before any decision.
   Pinned by Task 7's `screens` guard (its negative case).
2. A screen whose rule wants a block not run yet, including one with fewer blocks run than the floor: `Running`, its summary row with empty stop and call and its next blocks from `Running.want`, never a call.
   Pinned by Task 5's `adapter` guard (its cut screens).
3. A collapse inside the floor's first step (C1m4/p64 under the veto): `Stopped` with `vetoed == {0: KEEP}`, read as "collapse", at the collapsing block.
   Pinned by Task 5's `adapter` guard and Task 4's exact check.
4. A screen with no more blocks than the floor (C1m4/p64 without the veto, 12 blocks; `.m2`, 15): `Stopped` on all of them, read as "ran out".
   Pinned by Task 5's `adapter` guard and Task 7's phase-5 check.
5. A variant's rows that are not a prefix of its order (a block run out of order, or HELD's order changed): SystemExit naming the variant before any decision.
   Pinned by Task 7's `screens` guard (fault `shuffled`).
6. A stored ledger made under another preset, or on blocks the rows no longer start with: SystemExit naming the variant before `advance` (which can tell a ledger's positions from the data's but not its contents, epistemetry's spec §12).
   Pinned by Task 7's `screens` guard (its negative cases).

## Files

- Create `research/roadless/screening.py`: the rule Strategy (`Rule`, `RealizedLoss`, `ZeroOne`), `OwnStake`, the live adapter `arrived_of`, `why_of`, `CALLS`.
- Create `research/roadless/screening_check.py`: the fixture's loader, the acceptance runs (`runs`), the statistical comparison (`equivalent`), and the guards `target`, `records`, `adapter`, `acceptance`, `screens`.
- Create `research/roadless/screen_acceptance.csv`: the acceptance runs at the pin, both presets (Task 4 writes it).
- Create `research/roadless/consolidated_rows/screen_decisions.json`: each screen's latest `Running` or `Stopped` record with its blocks, under the preset's `Screen` record (Task 7).
- Create, then delete: `research/roadless/screening_migration.py` (Tasks 4 and 8).
- Rename: `research/roadless/epireplay_designs.json` to `research/roadless/screen_designs.json` (Task 3).
- Modify: `pyproject.toml`, `uv.lock`, `research/roadless/common.py`, `research/roadless/consolidated.py`, `research/roadless/region_leaks.py`, `research/roadless/consolidated_rows/summary_screen.csv`, `research/roadless/NOTES.md`, `research/roadless/BACKLOG.md`.
- Delete: `research/roadless/seqscreen.py`, `research/roadless/epireplay.py`.
- Git-ignored, written by the runs: `research/roadless/screen_acceptance_jobs/<rev7>/` (`*.parquet`; deleted in Task 8 Step 5).

---

### Task 1: Pin epistemetry at 165f98b

**Files:**
- Modify: `pyproject.toml` (`[project] requires-python`, `[dependency-groups] research`, `[tool.uv.sources]`)
- Modify: `uv.lock` (regenerated)

**Interfaces:**
- Produces: `import epistemetry` in the project's environment at 165f98b31c7114ab9fc3b902c9dfc79a67623d99, for every later task and on every cluster node.

- [ ] **Step 1: Record the environment before**

Run: `uv pip freeze --exclude-editable > $SCRATCH/freeze_before.txt` ($SCRATCH the session's scratchpad).

- [ ] **Step 2: Edit pyproject.toml**

`requires-python = ">=3.12"`; append `"epistemetry",` to the `research` group (its comment: the decision layer reblock's screens run through, Rulings 1-2); under `[tool.uv.sources]`, beside cluster_submit:

```toml
epistemetry = { git = "https://github.com/GeoffChurch/epistemetry.git", rev = "165f98b31c7114ab9fc3b902c9dfc79a67623d99" }
```

Leave `[tool.ruff] target-version` and `[tool.mypy] python_version` as they are (they serve src/reblock, which does not import epistemetry).

- [ ] **Step 3: Lock, and check nothing else moved**

Run: `uv lock` (never `--upgrade`), then `git diff uv.lock | grep -E '^[-+](name|version) = '`.
Expected: `+name = "epistemetry"`, `+version = "0.1.0"`, and removals only of the forks that served Python < 3.12 alone (numpy 2.4.6 and its like: each removed `[[package]]` block's `resolution-markers` all read `python_full_version < '3.12'`; check each).
Any other version change: stop and report.

- [ ] **Step 4: Sync and check the environment and the import**

Run: `uv sync && uv pip freeze --exclude-editable | diff $SCRATCH/freeze_before.txt -`
Expected: one line, `> epistemetry @ git+https://github.com/GeoffChurch/epistemetry.git@165f98b31c7114ab9fc3b902c9dfc79a67623d99`.
Then: `uv run python -I -c "from epistemetry.design.decide import AdoptKeepAskByStake as T; from epistemetry.design.screen import advance, Screen, ProportionalTo; print(T(rates=(1e-6,), weights=(1.0,), uses=820.0).price)"` prints `1e-06`, and `grep -c 'epistemetry.git?rev=165f98b31c7114ab9fc3b902c9dfc79a67623d99#165f98b31c7114ab9fc3b902c9dfc79a67623d99' uv.lock` prints `1` or more.

- [ ] **Step 5: Check cluster_submit can mirror it**

Run: `uv run python -I -c "import pathlib; from cluster_submit.envs import locked_git_sources; print(locked_git_sources(pathlib.Path('uv.lock').read_text()))"`
Expected: two sources, cluster_submit's and epistemetry's, and no SystemExit (the shape `<url>?rev=<sha>#<sha>` is one it knows).

- [ ] **Step 6: Commit and push**

Subject: `research: roadless -- pin GeoffChurch/epistemetry 165f98b in the research group (pyproject [tool.uv.sources], as cluster_submit): requires-python >=3.12 (epistemetry requires it; uv refuses the lock otherwise), the lock's Python 3.11 forks dropped; the 3.12 environment gains epistemetry alone; cluster_submit mirrors the new git source`
Paths: `pyproject.toml uv.lock`

---

### Task 2: The block sets in common.py

**Files:**
- Modify: `research/roadless/common.py` (after `large_blocks`)
- Modify: `research/roadless/region_leaks.py` (`import seqscreen`; `TUNING = seqscreen.TUNING` and its uses)
- Modify: `research/roadless/consolidated.py` (its copy of `TUNING` in the road-lineup section, and `report`'s use)

**Interfaces:**
- Produces: `common.TUNING: tuple[str, ...]` (the 13 large blocks the presets were tuned on, seqscreen.TUNING's order) and `common.HELD: tuple[str, ...]` (the 46 other large blocks that fit the retired raster's greedy at h 0.5, in the order the screens run them: Ruling 4).
  Later tasks read both.

- [ ] **Step 1: Add the two constants**

`TUNING` is seqscreen.TUNING's literal, moved.
`HELD` is the literal printed once by `PYTHONPATH=. uv run python -c "import sys; sys.path.insert(0, 'research/roadless'); import numpy as np, seqscreen; h = seqscreen.blocks_of('held'); print(tuple(h[i] for i in np.random.default_rng(seqscreen.ORDER_SEED).permutation(len(h))))"`.
Each gets a comment with its provenance (TUNING: BACKLOG "Tune on the 13, report on the 44"; HELD: "the large blocks large82_simp_vs_greedy.parquet holds, less TUNING, in seqscreen's order (default_rng(0)'s permutation of them sorted), the order phase 5 ran them in; frozen 2026-10-17.
The order is data: no seed regenerates it").

- [ ] **Step 2: Check them against seqscreen, once**

Run: `PYTHONPATH=. uv run python -c "import sys; sys.path.insert(0, 'research/roadless'); import numpy as np, common, seqscreen; h = seqscreen.blocks_of('held'); assert common.TUNING == seqscreen.TUNING; assert common.HELD == tuple(h[i] for i in np.random.default_rng(0).permutation(46)); assert len(set(common.HELD)) == 46 and not set(common.HELD) & set(common.TUNING); assert set(common.HELD) | set(common.TUNING) <= set(common.large_blocks()); print('ok')"`
Expected: `ok`.
This is a migration check, not a guard: it reads the parquet this task stops reading.

- [ ] **Step 3: Move the callers**

region_leaks.py: drop `import seqscreen` and the alias; its uses read `common.TUNING`. consolidated.py: delete its copy (with the comment "seqscreen.TUNING (not imported: its loaders read the rows)"); `report` reads `common.TUNING`.

- [ ] **Step 4: Check imports and lint**

Run: `PYTHONPATH=. uv run python -c "import sys; sys.path.insert(0, 'research/roadless'); import region_leaks, consolidated; print(len(region_leaks.common.TUNING))"`
Expected: `13`.
Then ruff `--select F,E9` on the three files passes.

- [ ] **Step 5: Commit and push**

Subject: `research: roadless -- the block sets in common.py (TUNING, 13; HELD, 46, the held-out screens' blocks in the order phase 5 ran them, frozen from seqscreen's permutation and checked equal): region_leaks and consolidated read common.TUNING (consolidated's copy gone); HELD no longer read from the raster-era large82_simp_vs_greedy.parquet`
Paths: `research/roadless/common.py research/roadless/region_leaks.py research/roadless/consolidated.py`

---

### Task 3: The rule Strategy and the presets

**Files:**
- Create: `research/roadless/screening.py`
- Create: `research/roadless/screening_check.py`
- Modify: `research/roadless/consolidated.py` (the screen's edge constants and `screen_presets()`, beside `SCREENS`)
- Rename: `research/roadless/epireplay_designs.json` to `research/roadless/screen_designs.json` (`git mv`, content unchanged); epireplay.py's `DESIGNS` follows it (one line), so every commit until Task 8 imports cleanly.

**Interfaces:**
- Consumes: Task 1's environment; `common.HELD` (Task 2) for the horizons.
- Imports from epistemetry 165f98b (every constructor keyword-only):
  - `epistemetry.design.screen`: `ValueOfMoreUnits`, `Bootstrap`, `ProportionalTo`, `SameTarget`, `TargetOf`, `Seen`, `Screen`, `VetoBelow`, `NoVeto`;
  - `epistemetry.design.decide`: `AdoptKeepAskByStake`, `AdoptKeepAsk`, `Priced`, `PricedTarget`, `ValueOfInformation`, `StandardErrors`, `StopRule`, `KEEP`, `ASK`, `ADOPT`;
  - `epistemetry.design.replay`: `Design`, `Order`;
  - `epistemetry.stats.records`: `Recordable`, `revive`.
- Produces, in screening.py:
  - `class Rule(Protocol)`: `target_of(self) -> TargetOf`, `stop(self) -> StopRule`.
    A Rule is a builder: what it builds (the policy's `target` and `stop`) is recorded inside `Screen.record()`, so a Rule records nothing itself.
  - `RealizedLoss` (frozen, kw_only): `rates: tuple[float, ...]` (Lens A per GPU-second, increasing), `weights: tuple[float, ...]`, `uses: float`, `margin: float` (Monte Carlo standard errors a reading must clear).
    `target_of()` is `SameTarget(target=AdoptKeepAskByStake(rates=self.rates, weights=self.weights, uses=self.uses))`; `stop()` is `ValueOfInformation(margin=StandardErrors(k=self.margin))` (the price is the target's, carried in the evidence; no rule holds one).
  - `OwnStake(Recordable, namespace="reblock")` (frozen, kw_only), a `TargetOf`: `low: float, high: float, price_rate: float, uses: float`; `of(self, seen: Seen, *, arm: int) -> Priced` returns `Priced(target=AdoptKeepAsk(low=self.low, high=self.high), price=self.price_rate / self.uses / stake)`, `stake = float(abs(o[:, 0].mean() - self.price_rate * o[:, 1].mean()))` over `o = seen.arms[arm].outcomes` (the arm's own prefix: epireplay.stake's expression).
    A zero stake makes `Priced` refuse an infinite price, naming it: an arm exactly at break-even at the price rate has no stake to price.
  - `ZeroOne` (frozen, kw_only): `rates, weights, uses, margin` as `RealizedLoss`; `target_of()` is `OwnStake(low=self.rates[0], high=self.rates[-1], price_rate=float(np.asarray(self.rates) @ np.asarray(self.weights)), uses=self.uses)`; `stop()` as `RealizedLoss`'s.
  - `CALLS: Mapping[int, str] = {KEEP: "keep", ASK: "ask", ADOPT: "adopt"}` (calls are int rows at 165f98b, the truth's included).
- Produces, in consolidated.py (the edge):
  - `RATES_PER_GPU_SECOND = (2e-6, 1.9e-5)`, `RATE_POINTS = 5`, `FUTURE = 820.0`, `SCREEN_FLOOR = 15`, `SCREEN_STEP = 1`, `COLLAPSE = 0.1`, `SCREEN_OUTER = 4000`, `SCREEN_INNER = 200`, `SCREEN_BLOCKS = 20`, `SCREEN_DRAWS = 4000`, `SCREEN_SEED = (0,)`, each with seqscreen's comment on where it came from (RATES' two calls on record, MIN_BLOCKS' replay, COLLAPSE's BACKLOG entry, FUTURE's horizon, DRAWS) or, for the seed, "the live screens' draws, keyed by it and each draw's place (epistemetry `stream`)".
  - `screen_presets() -> dict[str, Screen]`: `"realized-loss"` (`RealizedLoss`) and `"fewest-wrong"` (`ZeroOne`), both over `rates = tuple(float(r) for r in np.geomspace(*RATES_PER_GPU_SECOND, RATE_POINTS))`, `weights = (1.0 / RATE_POINTS,) * RATE_POINTS`, `uses=FUTURE`, `margin=0.0`, each built by one local function:

```python
Screen(policy=ValueOfMoreUnits(belief=Bootstrap(), functionals=((1.0, 0.0), (0.0, 1.0)),  # gain, extra GPU-s
                               target=rule.target_of(), horizons=tuple(range(1, len(common.HELD) + 1)),
                               step=SCREEN_STEP, cost=ProportionalTo(covariate="base_seconds"), stop=rule.stop(),
                               floor=SCREEN_FLOOR, outer=SCREEN_OUTER, inner=SCREEN_INNER, blocks=SCREEN_BLOCKS,
                               draws=SCREEN_DRAWS),
       drop=VetoBelow(threshold=-COLLAPSE, column=0, call=KEEP), seed=SCREEN_SEED)
```

  Each preset's comment says what it optimizes and what it gives up (Ruling 3).
  The horizons' comment is Ruling 9.
- Produces, in screening_check.py:
  - `Fixture(NamedTuple)`: `name: str, many: bool, blocks: tuple[str, ...], gain: np.ndarray (u, k), seconds: np.ndarray (u, k), base_seconds: np.ndarray (u,), orders: tuple[np.ndarray, ...]`, with `design(self) -> Design`: `Design(outcomes=np.stack([gain, seconds - base_seconds[:, None]], axis=2), costs=seconds, base_costs=np.zeros(u), covariates={"base_seconds": base_seconds})`.
  - `designs() -> dict[str, Fixture]`: screen_designs.json, every key read by name, shapes checked (epireplay._plan's checks, 100 orders each), each design's units at most `len(common.HELD)` (Ruling 9), in the file's order.
  - `realized_risk(draws: np.ndarray, *, rates: tuple[float, ...], weights: tuple[float, ...]) -> tuple[float, int]`: outside epistemetry, one arm's (gain, extra) draws to the weighted mean over rates of E|g - r c| where the posterior-mean call at r is wrong, and the call over the rates (epireplay.realized_risk, weighted).
  - `own_stake(outcomes: np.ndarray, *, price_rate: float) -> float`: outside epistemetry, |mean gain - price_rate x mean extra| over one arm's prefix (outcomes (n, 2)).
  - The guard registry and CLI: `GUARDS: dict[str, Callable[[], list[str]]]`, `FAULTS: dict[str, dict[str, Callable[[], AbstractContextManager[None]]]]`, `main(guard: str, fault: str | None)`, `both_ways(guards: list[str])`, as gap_clear_check.py's.

- [ ] **Step 1: Write screening.py**

The module docstring says what a screen is now (NOTES, the move's section, Task 9): epistemetry's `Screen` run by `advance`, the rule a Strategy, the presets at consolidated's edge, `Screen.record()` the identity.
It says that the pin lives in pyproject.toml and how a pin moves:
- edit the `rev`, then `uv lock --upgrade-package epistemetry`;
- run `screening_check.py both-ways target,records,adapter,acceptance,screens`;
- if `acceptance` fails only because the random numbers changed, regenerate the runs (`screening_check.py runs`) and pass `screening_check.py equivalent` against the committed CSV before committing the new one (Task 4);
- re-derive the phase-5 screens if `screens` fails the same way (Ruling 10), and commit.

- [ ] **Step 2: Add the edge constants and `screen_presets()` to consolidated.py**

- [ ] **Step 3: Rename the fixture**

Run: `git mv research/roadless/epireplay_designs.json research/roadless/screen_designs.json`, and point epireplay.py's `DESIGNS` at the new name.

- [ ] **Step 4: Write the guard `target`**

For each preset of `consolidated.screen_presets()`, on every pairwise fixture, order 0's first 15, 20 and 25 blocks (as many as the design has), with `seen = f.design().arrived(f.orders[0]).seen(reach=[n], n=n, live=frozenset({0}))`, its prefix `prefix = f.orders[0][:n]`, and draws `Bootstrap().of(outcomes[prefix, 0], population=u).draw(np.eye(2), np.random.default_rng([7, i]), draws=SCREEN_DRAWS)`:

```python
# policy = preset.policy; target = policy.target.of(seen, arm=0); rates, weights,
# price_rate = rates @ weights and FUTURE from the edge's constants
assert policy.stop == ValueOfInformation(margin=StandardErrors(k=0.0))
# realized-loss:
loss, call = realized_risk(draws, rates=rates, weights=weights)
assert math.isclose(float(target.risk(draws)), FUTURE * loss, rel_tol=1e-9, abs_tol=1e-18)
assert int(np.asarray(target.decide(draws)).reshape(-1)[0]) == call
assert math.isclose(target.price, price_rate, rel_tol=1e-15)
# fewest-wrong: the arm's own stake, recomputed here from the prefix
assert target.target == AdoptKeepAsk(low=rates[0], high=rates[-1])
assert math.isclose(target.price, price_rate / FUTURE / own_stake(outcomes[prefix, 0], price_rate=price_rate),
                    rel_tol=1e-12)
```

The price a decision used is read off its reading: on x2/p64, r2/base and pPg/p256 at the 20-block prefix, `j = policy.next(seen, rng=np.random.default_rng([8, i]))` gives `j.readings[0]["price"] == target.price` (rel 1e-15) and `j.rules[0] == policy.stop`.
Faults: `dropped-uses` (`RealizedLoss.target_of` patched to build its target with `uses=1.0`: the risk check fails); `unstaked` (`OwnStake.of` patched to price at `price_rate / uses`, the stake left out: the own-price check fails).
(452052c's fault `unpriced` is gone: at 165f98b a stop holds no price, so a stop priced apart from its target cannot be built.)

- [ ] **Step 5: Write the guard `records`**

For each preset: `revive(json.loads(json.dumps(screen.record(), allow_nan=False))) == screen`, and `revive(screen.record()).record() == screen.record()`; the record's `policy.target` names `epistemetry.SameTarget` for realized-loss and `reblock.OwnStake` for fewest-wrong.
Fault: `array-rates` (`np.geomspace` patched so the edge builds its rates as a numpy array instead of a tuple of floats, the slip seqscreen's arrays invite: the round trip fails or raises, which the guard reports as a failure).
(The ledger's round trip is the `screens` guard's, on reblock's own file: Task 7.)

- [ ] **Step 6: Run both ways**

Run: `PYTHONPATH=. uv run python research/roadless/screening_check.py both-ways target,records` (single-threaded env)
Expected: `target: passes`, `records: passes`, then each fault's line `FAILS on ...`, and `both ways, target, records: as expected`.

- [ ] **Step 7: Lint and types**

Run the ruff and mypy gates (Global Constraints) on screening.py, screening_check.py and consolidated.py.

- [ ] **Step 8: Commit and push**

Subject: `research: roadless -- screening.py on epistemetry 165f98b: the rule a Strategy (Rule: RealizedLoss, the stake target's own price; ZeroOne, each arm's target OwnStake priced at its own stake), the presets realized-loss and fewest-wrong as epistemetry Screens built at consolidated's edge (Screen.record() the run's identity; MonteCarlo, RatioToBase and the hand-written identity not written: ValueOfMoreUnits records them, ProportionalTo reads the base's seconds as a Design covariate); the frozen designs renamed screen_designs.json (reblock's real-data acceptance fixture); guards target and records, each failing under its faults (dropped-uses, unstaked; array-rates)`
Paths: `research/roadless/screening.py research/roadless/screening_check.py research/roadless/consolidated.py research/roadless/screen_designs.json research/roadless/epireplay_designs.json research/roadless/epireplay.py`

---

### Task 4: The acceptance runs at 165f98b, compared statistically with the fourth replay

The fourth replay's runs (674dba4) threaded one generator through each run; at 165f98b each draw is keyed by the order's seed and its place, so its runs cannot be reproduced bit for bit (Ruling 7).
Instead both presets are replayed at 165f98b on the same designs and the same block orders, and their summary rates are compared, run against stored run on the same order, with a pre-registered test.
The new runs become screen_acceptance.csv, the acceptance guard's expected runs (Task 6) and the reference for every later pin move.
Replay 4's rows stay on disk until this comparison passes; Task 8 deletes them only after it.

**The comparison set.** For each preset, the 13 pairwise designs, orders 0-99, in the no-veto setting, and C1m4/p64 in the veto setting too (the one design with a block below -COLLAPSE; elsewhere the veto never fires, so a design's veto runs equal its no-veto runs exactly and are one sample, counted once): 1,400 runs a preset, 2,800 in all.
Order o of design f is `Order(units=tuple(int(x) for x in f.orders[o]), seed=(ACCEPTANCE_SEED, o))`, `ACCEPTANCE_SEED = 0`; the truth is `Truth.of(design, screen.policy, seed=TRUTH_SEED, drop=<the setting's>)` (`TRUTH_SEED = (0,)`: the whole design's posterior is a point, every unit observed, so the truth draws nothing that matters, the reason at the definition); scoring `AsksApart(ask=ASK)`; the veto setting's drop is `screen.drop`, the other's `NoVeto()`.

**The test (pre-registered here, 2026-10-18).** Per run: `units` (the blocks the stop used), the cost fraction (`cost` over the design's whole cost, `seconds.sum()`), `wrong` (0 or 1) and `asks` (0 or 1).
For each preset and each of the four, `d_i = new_i - old_i` over the 1,400 pairs (same design, setting, order).
1. **Exact, where nothing is drawn:** every (preset, design, setting)'s truth equals the stored `full`; every run of a design no longer than the floor (C1m4/p64, 12 blocks: decided in its floor step, in both settings) equals its stored run in `units`, `call`, `cost`, `wrong`, `asks`.
2. **Equivalence:** the 95% percentile bootstrap interval of mean(d) (10,000 resamples of the pairs, `np.random.default_rng(0)`) lies inside +/- delta, delta half the distance between the two presets' means on the stored runs, so the move cannot shift either preset by half of what separates them and Ruling 3 stands.
   On the comparison set (computed 2026-10-18 from epireplay4.parquet): realized-loss 15.03 blocks, cost fraction 0.443, wrong 0.0479, asks 0.203; fewest-wrong 16.25, 0.473, 0.0264, 0.227; so delta = 0.609 blocks, 0.0150, 0.0107 and 0.0121 a run.
   `equivalent` recomputes the deltas from the old runs and refuses to judge unless they match these to three significant figures.
3. **No detectable shift:** a two-sided sign-flip permutation test of mean(d) (100,000 random flips, `np.random.default_rng(1)`; p = (1 + #{|flipped mean| >= |mean|}) / 100,001) for the 8 (preset, quantity) pairs, and for `units` per (preset, design) (26 more), Holm-corrected together at family-wise 0.05.
   The test is exact under the null: on one order, the old and the new run are two draws of one policy under independent streams, so swapping a pair's labels leaves their joint law unchanged.
   For `wrong` and `asks` it is McNemar's exact test.
4. **Power, checked before running:** the two presets' paired spread on one order, a pessimistic stand-in for one preset's seed-to-seed spread (two policies differ on an order at least as often as one policy under two streams; sd 4.5 blocks; 30 and 44 discordant pairs in 1,400 for wrong and asks), which puts the intervals' half-widths near 0.24 blocks, 0.008 and 0.009: inside every delta.
   An equivalent move passes with high probability.

Pass: 1 and 2 hold for both presets and 3 rejects nothing.
If 1 or 2 fails: the move does not proceed; report the intervals and the differing cells, delete nothing.
If 2 holds and 3 rejects: something besides the streams changed between 674dba4 and 165f98b, small enough to keep the ruling; stop and report to the owner with the per-design table, delete nothing.

**#10 (Ruling 11), measured on the same runs.** Each new run also records `wave_cost`: its cost were every step paid whole by every arm the step named, as a live parallel wave pays it (from `Run.decisions`: decision d's step charges each arm in `d.judgement.step.arms` its costs on the order's units `d.at .. d.at + units - 1`, past a veto included).
`wave_cost - cost` is the replay's under-charge; with step 1 it can be non-zero only where a veto falls inside the floor's step.

**Files:**
- Modify: `research/roadless/screening_check.py` (`runs`, `equivalent`, their commands)
- Create: `research/roadless/screening_migration.py` (temporary; Task 8 deletes it)
- Create: `research/roadless/screen_acceptance.csv`
- Its jobs: `research/roadless/screen_acceptance_jobs/<rev7>/` (git-ignored by `*.parquet`; `<rev7>` the pinned rev's first 7 characters, read from pyproject.toml's `[tool.uv.sources]` by `pinned_rev()`, one boundary, no default)

**Interfaces:**
- Consumes: `consolidated.screen_presets()`, `screening_check.designs()`, `Fixture.design()`, epistemetry's `replay`, `Truth`, `Order`, `AsksApart`, `NoVeto`, `ASK`.
- Produces, in screening_check.py:
  - `runs(*, workers: int) -> pd.DataFrame` and the command `runs <workers>`: one job per (preset, design, setting, 25 orders), a parquet each, resumable (a job whose file exists is skipped), then gathered and written to screen_acceptance.csv with columns `preset, plan, veto, order, seed, units, call, cost, wrong, asks, truth, wave_cost` (`seed` the order's seed as JSON, `call` and `truth` arm 0's int).
    `workers` changes no result (exempt, the reason at the definition).
  - `equivalent(old: pd.DataFrame, new: pd.DataFrame, *, fixtures: Mapping[str, Fixture], resamples: int, flips: int) -> list[str]`: the test above, reading only `preset, plan, veto, order, units, call, cost, wrong, asks, truth`; it returns its failures and prints, per preset, each quantity's means, mean(d), interval, delta and p, the per-design table, and the Holm verdicts.
    The command `equivalent <old.csv> <new.csv>` passes `resamples=10_000, flips=100_000`, the values pre-registered above.
- Produces, in screening_migration.py: `old_runs() -> pd.DataFrame`: epireplay4.parquet's rows for the comparison set (policies `"by stake step 1, 5 rates"` as realized-loss and `"adopt/keep/ask step 1, own price"` as fewest-wrong, through a dict with no default after filtering to those two; pairwise arms only), in the CSV's columns (`call = calls[0]`, `truth = full[0]`); `main()`: `equivalent(old_runs(), pd.read_csv(<the CSV>), ...)`, exit non-zero on a failure; and the command `self-check` (Step 2).
  It reads only those summary columns, never 674dba4's decision records.

- [ ] **Step 1: Write `runs`, `equivalent` and `old_runs`**

- [ ] **Step 2: Show the comparison sees a change, before the real run**

On the stored runs alone: `equivalent(old, old with units + 1 on every run of r2/base, ...)` (a shift inside the deltas, which the sign-flip test must find) and `equivalent(old, old with realized-loss's wrong flipped on 20 runs, ...)` (outside its delta) each return failures, and `equivalent(old, old, ...)` returns none.
Run: `PYTHONPATH=. uv run python research/roadless/screening_migration.py self-check` (the three cases, built in memory).
Expected: `as expected` (seconds).

- [ ] **Step 3: Run it in the background**

Run (with `run_in_background`, single-threaded env, 6 workers): `PYTHONPATH=. uv run python research/roadless/screening_check.py runs 6`
Expected: 112 jobs (2 presets x (13 + 1) design-settings x 4 chunks), 2,800 rows written to screen_acceptance.csv.
The fourth replay spent about 5.4 worker-hours on realized-loss's 1,300 no-veto pairwise runs and 7.3 on fewest-wrong's (their logs), C1m4/p64 seconds: about 13 worker-hours, so about 2.5 h on 6 workers, the longest chunk (both/p64 under fewest-wrong) about 30 min.
The harness prints each job's minutes; if the first ten average more than twice the fourth replay's per-run time, report before waiting.
Task 5's code may be written meanwhile.

- [ ] **Step 4: Compare**

Run: `PYTHONPATH=. uv run python research/roadless/screening_migration.py`
Expected: the exact checks hold (14 truths a preset, C1m4/p64's 200 runs a preset); every interval inside its delta; no Holm rejection; `pass`.
Then print the #10 measurement from the CSV: per preset, the share of runs with `wave_cost > cost`, and the mean of `(wave_cost - cost)` over the design's whole cost, over all runs and over C1m4/p64's veto runs.
On any failure, stop as the test says: do not commit the CSV, delete nothing.

- [ ] **Step 5: Commit and push**

Subject: `research: roadless -- the move's comparison: both presets replayed at epistemetry 165f98b on the fourth replay's designs and orders (2,800 runs; seeded streams, so statistical, not bit for bit): truths and C1m4/p64's draw-free runs equal; mean differences in blocks, cost fraction, wrong calls and asks inside half the presets' gap (intervals ...), no shift found (sign-flip, Holm); #10's wave under-charge ...; screen_acceptance.csv the acceptance runs at the pin`
Paths: `research/roadless/screening_check.py research/roadless/screening_migration.py research/roadless/screen_acceptance.csv`

---

### Task 5: The live adapter

epistemetry's `advance` is the live loop (#1); what stays reblock's is feeding it a screen's rows in the order they were run.

**Files:**
- Modify: `research/roadless/screening.py`
- Modify: `research/roadless/screening_check.py` (guard `adapter`)

**Interfaces:**
- Consumes: Task 3's presets, `designs()`, `Fixture.design()`; epistemetry's `advance`, `Arrived`, `Prefix`, `Running`, `Stopped`, `replay`, `Truth`, `Order`.
- Produces, in screening.py:
  - `arrived_of(*, gain: np.ndarray, seconds: np.ndarray, base_seconds: np.ndarray) -> Arrived`: `gain` (n,), b - a on the blocks run, in the screen's order; `seconds` (n,), b's GPU-seconds on them; `base_seconds` (u,), a's on every block of the screen, in its order, those run then the rest.
    The units are positions in the screen's order: `Arrived(order=np.arange(u), arms=(Prefix(units=np.arange(n), outcomes=np.column_stack([gain, seconds - base_seconds[:n]]), costs=seconds),), base_costs=np.zeros(n), covariates={"base_seconds": base_seconds})`.
  - `Why = Literal["settled", "collapse", "ran out"]`.
  - `why_of(s: Stopped, *, arm: int, units: int) -> Why`: "collapse" if `arm in s.vetoed`, "ran out" if `s.arm_units[arm] == units`, else "settled" (its rule met at its last judgement): read off the record, as epistemetry's `Stopped` says, never stored as a label.

- [ ] **Step 1: Write the guard `adapter` first**

Under `screen_presets()["realized-loss"]`, on r2/base (a near-tie ask), x2/p64 (an ask) and pPg/p256 (a keep) without the veto, C1m4/p64 (a collapse, 12 blocks) in both settings, and a degenerate design (p64/base with every gain 0) without the veto, orders 0-2 (18 cases), with `i = f.orders[o]`, `order = Order(units=tuple(int(x) for x in i), seed=(ACCEPTANCE_SEED, o))` and `drop` the setting's:

```python
run = replay(design, preset.policy, orders=[order], truth=Truth.of(design, preset.policy, seed=TRUTH_SEED, drop=drop),
             scoring=AsksApart(ask=ASK)).runs[0]
live = replace(preset, drop=drop, seed=order.seed)
s = advance(live, ledger=(), arrived=arrived_of(gain=f.gain[i, 0], seconds=f.seconds[i, 0],
                                                base_seconds=f.base_seconds[i]))
assert isinstance(s, Stopped) and s.decisions == run.decisions and s.arm_units == run.arm_units
assert np.array_equal(np.array(s.calls), run.calls) and frozenset(s.vetoed) == run.vetoed
# the same screen cut short of its stop waits for exactly the blocks the replay took next
m = run.units - 1
cut = advance(live, ledger=(), arrived=arrived_of(gain=f.gain[i[:m], 0], seconds=f.seconds[i[:m], 0],
                                                  base_seconds=f.base_seconds[i]))
assert isinstance(cut, Running) and cut.decisions == run.decisions[:len(cut.decisions)]
assert cut.want[0][0] == m
```

The cut case applies where `run.units > 1` and the run did not stop on a collapse.
The full data run past the stop on most orders, so the first case also shows surplus blocks change nothing.
Absolute expectations, beside the replay:
- the degenerate design stops at `SCREEN_FLOOR` with KEEP, `why_of` "settled", in every order;
- C1m4/p64 under the veto stops with `vetoed == {0: KEEP}`, "collapse", at the position of its first block below `-COLLAPSE`;
- without the veto it stops "ran out" at 12 (Review Focus 3, 4).
Faults:
- `misaligned`: `arrived_of` patched to pass the covariate in the fixture's own block order, not the screen's: `ProportionalTo` prices each upcoming block by another block's base time, and the decisions differ.
- `gross-extra`: `arrived_of` patched to put b's gross seconds in the extra column, not `seconds - base_seconds`: the targets read the wrong cost and the calls differ.

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONPATH=. uv run python research/roadless/screening_check.py adapter`
Expected: AttributeError on `arrived_of` (not written yet).

- [ ] **Step 3: Write `arrived_of` and `why_of`**

- [ ] **Step 4: Run both ways**

Run: `PYTHONPATH=. uv run python research/roadless/screening_check.py both-ways adapter` (about 20 min single-threaded)
Expected: `adapter: passes`, each of `misaligned`, `gross-extra` `FAILS on ...`, `as expected`.

- [ ] **Step 5: Lint, types, commit and push**

Subject: `research: roadless -- screening.arrived_of, a live screen's rows as epistemetry's Arrived (positions in the screen's order, the base's seconds the cost model's covariate), and why_of, the stop read off a Stopped record; guard adapter: on r2/base, x2/p64, pPg/p256, C1m4/p64 (both settings) and a degenerate design, advance on the adapter's data is epistemetry's replay decision for decision, cut short it waits for the replay's next block; faults misaligned, gross-extra`
Paths: `research/roadless/screening.py research/roadless/screening_check.py`

---

### Task 6: The acceptance guard

reblock's real-data acceptance test for epistemetry: at any pin, the production preset reproduces the acceptance runs Task 4 compared and committed.

**Files:**
- Modify: `research/roadless/screening_check.py` (guard `acceptance`)

**Interfaces:**
- Consumes: `screen_acceptance.csv` (Task 4), `designs()`, `screen_presets()["realized-loss"]`, Task 4's orders (`seed` from each row), truths and scoring.
- Produces: guard `acceptance`, run on a pool of `--workers W` (single-threaded; a count that changes no result, so it is exempt from the required-parameter rule).

- [ ] **Step 1: Write the guard**

For realized-loss's rows of the CSV with orders 0-9 (13 designs without the veto, C1m4/p64 with it: 140 runs), in the file's order: the run's `units`, call, `wrong`, `asks` and truth equal the row's exactly, its `cost` within rel 1e-12.
Under a fault it returns at the first design that differs.
Fault: `dropped-uses` (as in `target`).

- [ ] **Step 2: Run both ways**

Run: `PYTHONPATH=. uv run python research/roadless/screening_check.py both-ways acceptance --workers 6`
Expected: `acceptance: passes` (140 of 140; about 35 worker-minutes), then `FAILS on <design> ...` under `dropped-uses`, `as expected`.

- [ ] **Step 3: Lint, types, commit and push**

Subject: `research: roadless -- guard acceptance, reblock's real-data acceptance test for epistemetry's pin (its spec, owner 2026-10-16): the realized-loss preset reproduces screen_acceptance.csv's runs on orders 0-9 exactly (140: 13 pairwise designs, C1m4/p64 under the veto too); fault dropped-uses; a pin move that changes the random numbers regenerates the runs and passes equivalent first`
Paths: `research/roadless/screening_check.py`

---

### Task 7: consolidated `screen-report` decides through epistemetry

**Files:**
- Modify: `research/roadless/consolidated.py` (`screen_report`, `main`, the module docstring's screen lines and its `screen-report` paragraph, `_bb_interval`'s docstring)
- Modify: `research/roadless/consolidated_rows/summary_screen.csv` (re-derived)
- Create: `research/roadless/consolidated_rows/screen_decisions.json`
- Modify: `research/roadless/screening_check.py` (guard `screens`)

**Interfaces:**
- Consumes: `screen_presets()`, `screening.arrived_of`, `why_of`, `CALLS`, `common.HELD`, epistemetry's `advance`, `Running`, `Stopped`, `Screen`, `revive`.
- Produces, in consolidated.py:
  - `ScreenData(NamedTuple)`: `eligible: tuple[str, ...]` (the variant's order: HELD's blocks it is built for), `run: tuple[str, ...]` (its blocks run, a prefix of it), `arrived: Arrived` (through `arrived_of`).
  - `screen_data() -> dict[str, ScreenData]`: every variant but the default, from the rows, independent of the preset; it raises (SystemExit, naming it) for rows off the order or without the default's beside them, for an eligible block with no default time in the pass's rows, and for more eligible blocks than the horizons (Ruling 9).
  - `Stored(NamedTuple)`: `blocks: tuple[str, ...], state: Running | Stopped`.
  - `read_decisions(screen: Screen) -> dict[str, Stored]`: `ROWS/screen_decisions.json`, the one JSON boundary: no file is no ledger yet (`{}`); otherwise `revive(json["screen"])` must equal `screen` (else SystemExit naming both records and saying that removing the file starts the screens afresh under the injected preset), and every key is read by name.
  - `ScreenTable(NamedTuple)`: `table: pd.DataFrame` (summary_screen.csv's columns, plus `break_even` and `screen`), `states: dict[str, Stored]`, `upcoming: dict[str, list[str]]` (each running screen's next blocks).
  - `screen_table(screen: Screen, *, data: Mapping[str, ScreenData], ledgers: Mapping[str, Stored], wave: int) -> ScreenTable`: computes, writes nothing.
  - `screen_report(screen: Screen, *, wave: int) -> None`: `screen_table(screen, data=screen_data(), ledgers=read_decisions(screen), wave=wave)`; writes `ROWS/summary_screen.csv` and `ROWS/screen_decisions.json` (`{"screen": screen.record(), "screens": {variant: {"blocks": [...], "state": state.record()}}}`, `allow_nan=False`) and prints as today.
  - CLI: `screen-report <preset> <wave>`; `main` resolves the preset once with `screen_presets()[name]`, an unknown name a SystemExit listing the presets.

- [ ] **Step 1: Write the guard `screens` first**

The preset is the one whose Screen equals `revive(json["screen"])`; none is a failure.
With `data = screen_data()` and `stored = read_decisions(preset)`:
- continued: `screen_table(preset, data=data, ledgers=stored, wave=0)`; fresh: the same with `ledgers={}`;
- each, against the committed files: `stop`, `call`, `used`, `eligible`, `run` exact; every float column within rel 1e-9; every state's `record()` equal to the JSON's, its blocks equal;
- every stored state round-trips: `revive(json.loads(json.dumps(rec, allow_nan=False))).record() == rec`;
- every judged arm's verdict re-derived from the record is its take: `d.judgement.rules[j].met_reading(d.judgement.readings[j]) == (j not in d.judgement.step.arms)`;
- every rule equals the preset's `policy.stop`, and every reading's `price` is the target's: `AdoptKeepAskByStake.price` for realized-loss, `price_rate / FUTURE / own_stake(<the arm's prefix at d.at>)` for fewest-wrong (rel 1e-12).

Negative cases (Review Focus 1, 6), each patched in memory:
- one rest block's default time dropped from the pass's rows: `screen_data` raises SystemExit naming that block;
- the JSON's `screen` record swapped for the other preset's: `read_decisions` raises SystemExit;
- one variant's stored `blocks` with its first block swapped for a block off its order: `screen_table` raises SystemExit naming the variant.

Faults:
- `shuffled`: `common.HELD` patched to itself reshuffled by `np.random.default_rng(1)`; `screen_data` raises "rows off its order", which the guard reports as a failure (Review Focus 5).
- `altered-record`: the loaded JSON patched so one judged reading's `price` is times 1 + 1e-6; the fresh states differ from the stored, and the price check fails.

- [ ] **Step 2: Rewrite `screen_report` as `screen_data`, `read_decisions`, `screen_table` and `screen_report`**

- A variant's order is `[k for k in common.HELD if relax.built_for_holds(screen_plan(v)) or not holds[k]]`; its checks are as today (rows a prefix; the default beside each).
- The base seconds are the default's re-run on the blocks run, then the pass's default times on the rest; a rest block without one raises naming it.
- The ledger is `ledgers[v].state.decisions` when `v` is stored, its `blocks` a prefix of the variant's run (else SystemExit), or `()`.
- `advance(screen, ledger=..., arrived=data[v].arrived)` gives the state: `Stopped` fills `used` (`arm_units[0]`), `stop` (`why_of(s, arm=0, units=len(eligible))`) and `call` (`CALLS[s.calls[0][0]]`); `Running` leaves `stop` and `call` empty, `used` the blocks run, and lists `eligible[w : w + max(wave, len(want))]` from its first wanted position `w = state.want[0][0]`.
- The stored blocks are the blocks run.
- `break_even` is the mean gain over the extra GPU-seconds a block over the blocks used: the rate an ask turns on (seqscreen's `question` printed it).
- `screen` is `json.dumps(screen.record(), sort_keys=True)`.
- The lens columns, time columns and B's order are computed as today.
- `import seqscreen` goes.
  `_bb_interval`'s docstring calls it the Bayesian bootstrap (Rubin 1981) rather than seqscreen's posterior: a reporting interval, not a decision.

- [ ] **Step 3: The phase-5 check, before anything is committed**

Run: `PYTHONPATH=. uv run python research/roadless/consolidated.py screen-report realized-loss 0` (no JSON yet: every screen derived afresh), then

```bash
PYTHONPATH=. uv run python -c "import io, subprocess, pandas as pd; f = 'research/roadless/consolidated_rows/summary_screen.csv'; old = pd.read_csv(io.StringIO(subprocess.check_output(['git', 'show', 'HEAD:' + f], text=True))).set_index('variant'); new = pd.read_csv(f).set_index('variant'); print(pd.concat({'seqscreen': old[['used', 'stop', 'call']], 'epistemetry': new[['used', 'stop', 'call']]}, axis=1))"
```

Expected, per the fourth replay (its realized loss made seqscreen's call in 1,286 of 1,300 runs a setting and stopped where it did in 1,152, earlier mostly on near-tie asks and clear keeps):
- every call equal: `r2l2` ask; `pP`, `seed`, `m2`, `mma`, `doc` keep;
- `pP`, `seed`, `mma`, `doc` 15 blocks, settled; `m2` 15, ran out (drawn from nothing: its 15 blocks are the floor's step);
- `r2l2` settled within its 24 blocks run, at 20 or another count.
  A different count is the methods' difference the fourth replay measured (r2/base: its stop earlier in 47 of 100 orders) and 165f98b's streams, not a fault.

If a call differs, if one of the five stops differs, or if `r2l2` is `Running` past its 24 blocks: restore the CSV (`git checkout -- research/roadless/consolidated_rows/summary_screen.csv`), delete the JSON, and stop to report.
Running more blocks is GPU time and needs the owner's yes.

- [ ] **Step 4: Run both ways**

Run: `PYTHONPATH=. uv run python research/roadless/screening_check.py both-ways screens`
Expected: `screens: passes`, `shuffled` and `altered-record` each `FAILS on ...`, `as expected`.

- [ ] **Step 5: Lint, types, commit and push**

Subject: `research: roadless -- consolidated screen-report decides through epistemetry's advance (screen-report <preset> <wave>, the preset a Screen built once at the edge and injected; realized-loss by default; each wave continues from the screen's stored ledger): phase 5's six screens re-derived, the calls seqscreen's (.r2l2 ask after N blocks; .pP, seed, MMA, damped OC keep at 15; .m2 keep, ran out at 15); summary_screen.csv gains break_even and screen (the Screen's record), screen_decisions.json each screen's Stopped record with its blocks; guard screens re-derives both afresh and from the ledger, each verdict and price from its record (faults shuffled, altered-record)`
Paths: `research/roadless/consolidated.py research/roadless/consolidated_rows/summary_screen.csv research/roadless/consolidated_rows/screen_decisions.json research/roadless/screening_check.py`

---

### Task 8: Delete seqscreen, epireplay and the migration

**Files:**
- Delete: `research/roadless/seqscreen.py`, `research/roadless/epireplay.py`, `research/roadless/screening_migration.py`

**Interfaces:**
- Consumes: nothing from them remains (Tasks 2 and 7 moved every caller; Task 4's comparison passed and is committed).

- [ ] **Step 1: Delete**

Run: `git rm research/roadless/seqscreen.py research/roadless/epireplay.py research/roadless/screening_migration.py`

- [ ] **Step 2: No reader left**

Run: `git grep -n -E 'seqscreen|epireplay|screening_migration' -- research/roadless ':!research/roadless/NOTES.md' ':!research/roadless/BACKLOG.md' ':!research/roadless/stopping_review'`
Expected: no import, path or call; prose that names them as history (screen_designs.json's `meta.source`, HELD's provenance comment, a docstring citing NOTES) is all that may remain, each line read and listed in the commit message.

- [ ] **Step 3: Everything still runs**

Run: `PYTHONPATH=. uv run python -c "import sys; sys.path.insert(0, 'research/roadless'); import region_leaks, consolidated, screening, screening_check"`, then `PYTHONPATH=. uv run python research/roadless/screening_check.py both-ways target,records,adapter,screens` and `... both-ways acceptance --workers 6`.
Expected: the imports succeed; every guard passes and fails under each of its faults.

- [ ] **Step 4: Commit and push**

Subject: `research: roadless -- seqscreen.py and epireplay.py deleted (the owner's decision of 2026-10-17, to dogfood epistemetry; the replays tie seqscreen within noise, it is not dominated), with screening_migration.py, the move's one reader of the fourth replay's rows; no reader of either remains; every guard passes both ways after the deletion`
Paths: `research/roadless/seqscreen.py research/roadless/epireplay.py research/roadless/screening_migration.py`

- [ ] **Step 5: Delete the replays' data (the owner said "delete" to Ruling 6, 2026-10-17), only after Task 4 passed**

Check first that Task 4's commit is in `git log` with its `pass`; if the comparison did not pass, skip this step and say so.
Run: `ls -d research/roadless/epireplay*.parquet research/roadless/epireplay*_jobs research/roadless/epireplay4_*.log research/roadless/screen_acceptance_jobs`, check the list holds nothing else, then `rm -r` exactly those paths.
They are git-ignored (screen_acceptance.csv holds every acceptance run): nothing to commit.

---

### Task 9: The records

**Files:**
- Modify: `research/roadless/NOTES.md` (a new section at the end)
- Modify: `research/roadless/BACKLOG.md` (located by the headings and text named below)
- Maybe: the wiki, `~/wiki` (GeoffChurch/wiki-private), through the `wiki` skill

**Interfaces:**
- Consumes: the numbers Tasks 4, 6 and 7 printed and committed.

- [ ] **Step 1: NOTES, "### Pruning by epistemetry: the move (owner 2026-10-17)"**

Run `git status` first; the pass's agent may have left NOTES mid-edit.
Append the section after the last one, wrapped at 99, current numbers only, no before/after:
- why: the owner's decision, to dogfood epistemetry; seqscreen deleted by decision, not dominance; the fourth replay's tie in one sentence with its counts;
- what a screen is now: epistemetry 165f98b's `Screen` run by `advance` (the rule Strategy in screening.py, the presets and what each gives up, `Screen.record()` as the identity, each wave continuing from its stored ledger), the pin in pyproject and how it moves, Python >=3.12;
- the comparison (Task 4): why statistical (seeded streams), the test and its deltas, each preset's intervals and p-values, the exact checks;
- #10: the replay's under-charge against a live wave, as measured on Task 4's runs;
- phase 5's verdicts as summary_screen.csv now holds them;
- the guards, each with its fault's result;
- what was deleted, and where the replays' data and the fixture stand (Ruling 6).

The consolidated pass's own section keeps what it says of the pass (it pruned with seqscreen); any number in it that summary_screen.csv no longer holds is edited to the CSV's value.
Check: `awk 'length($0) > 99' research/roadless/NOTES.md` prints nothing from the new section (`python3 .superpowers/sdd/2026-10-16-consolidated-pass/handoff_files/reflow.py` re-wraps a draft at 99, if it is still there).

- [ ] **Step 2: BACKLOG**

Wrapped at 99; current state only:
- "Pruning by epistemetry's decision layer" (Experiment design): done, the move; it replaces "seqscreen stays", the 452052c lead and "the pin moves only with a next replay" with the pin in pyproject, `screening_check.py acceptance` at every pin move, and `runs` plus `equivalent` when a pin changes the random numbers.
- "The rate bracket" (Open): `consolidated.RATES_PER_GPU_SECOND` for `seqscreen.RATES`.
- "Sequential stopping" (How we run experiments) and the status lines naming seqscreen.py: what screens now (epistemetry's `advance` on a `Screen`: ValueOfMoreUnits on the finite-population bootstrap, AdoptKeepAskByStake over the bracket's five rates, FUTURE uses, a 15-block floor, the collapse veto).
- "*Generalize kernelcore's outcome model*": the stop was built in seqscreen.py, now epistemetry's.
- Under "Not pursued, with what would change that":
  - *seqscreen* (deleted 2026-10-17 by decision, not dominance): it would come back if a pin move's comparison shows epistemetry's screen leaving its decisions; the code is at the commit before Task 8's.
  - *Many-arm screens* (the per-arm screen, SharedStake; their designs kept in screen_designs.json): they would come back with a screen of several variants run together.
- The epistemetry issues: #1-#9 filed by reblock (Task 10), #1-#8 closed by 165f98b, #9 open; #10 (within-step charging) with Task 4's measurement.

- [ ] **Step 3: The wiki, only if something methodological was learned**

Candidates:
- a migration across a change of random streams proved by a paired, pre-registered comparison (sign flips exact under the null because the two runs on one order are exchangeable; deltas set at half the gap between the alternatives the decision chose between), if it worked as designed;
- the measured gap between a replay's unit-by-unit charge and a live wave's, if it is not negligible.
Use the `wiki` skill, `wiki search --lex "<question>"` from `~/wiki` first, and add to the page found rather than open a new one.
Nothing learned: nothing written.

- [ ] **Step 4: Comment on GeoffChurch/epistemetry #10 (the owner's yes, 2026-10-18)**

Run: `gh issue comment 10 --repo GeoffChurch/epistemetry --body "<Task 4's wave_cost measurement: the share of runs and the cost fraction under-charged, per preset, all runs and C1m4/p64 under the veto; step 1 past a 15-block floor; reblock's commit>"`.

- [ ] **Step 5: Commit and push**

Subject: `research: roadless -- NOTES "Pruning by epistemetry: the move" and BACKLOG: screens decide through epistemetry 165f98b (advance on a Screen; presets realized-loss and fewest-wrong); seqscreen deleted by the owner's decision, not dominance; the statistical comparison, #10's under-charge, phase 5's verdicts and the guards`
Paths: `research/roadless/NOTES.md research/roadless/BACKLOG.md`

---

### Task 10: Feedback for epistemetry (done)

**Filed 2026-10-17 at the owner's word ("Add all the items on GitHub as issues, including the three pieces as feature requests"):** GeoffChurch/epistemetry #1 (live driver), #2 (ValueOfMoreUnits Recordable), #3 (Decision records), #4 (cost-model covariates), #5 (NamedTuple revive, bug), #6 (stop priced from its target), #7 (kw_only), #8 (flat record namespace), #9 (Python floor, question).
**Landed:** #5-#8 by 2026-10-18, #1-#4 at 165f98b (closed 2026-10-18); this plan uses them in place of the reblock code each would have needed (`Screening.decide`, the hand-assembled identity, `RatioToBase`, `decision_record`/`decision_of`, `MonteCarlo`).
#9 is open.
**Filed by epistemetry's session:** #10 (within-step charging, Ruling 11) and #11 (design.adaptive's `Decision` codec, which reblock does not use).
The rounding-tie issue once held for Task 4 is moot: Task 4 no longer compares a rescaled replay run for run.

**Files:** none in the repo (BACKLOG gets the issue numbers in Task 9 Step 2).

- [x] **Step 1: Ask the owner which to file** (all, 2026-10-17)
- [x] **Step 2: File the approved ones** (#1-#9, 2026-10-17)

---

## Execution notes for the controller

- **GeoffChurch/epistemetry #1-#4 have landed** (165f98b, 2026-10-18); this plan was re-cut onto it the same day: Tasks 3, 5 and 7 use `Screen`, `advance`, `Running`/`Stopped` records and `ProportionalTo`, and Task 4 is the statistical comparison the owner chose on 2026-10-18 in place of a bit-for-bit proof at 452052c.
- Ruling 3 accepted by the owner (2026-10-17): `fewest-wrong` stays as the second preset (37 wrong calls against 67).
- Ruling 6 overruled by the owner (2026-10-17): **delete** the replays' data after the move; Task 8 Step 5 runs only after Task 4's comparison has passed.
- Ruling 2 accepted by the owner (2026-10-17): epistemetry joins the research group and requires-python rises to >=3.12 on research/roadless.
- Start only once the consolidated pass's agent has returned and `git status` shows no edits of theirs to consolidated.py, NOTES.md, BACKLOG.md or consolidated_rows/.
- Task 4's run takes about 2.5 h in the background.
  Task 5's code can be written meanwhile; its guard does not depend on Task 4's output.
  Task 6 needs Task 4's CSV, and Task 8 needs every task before it.
- Stop points that need the owner: Task 4's exact or equivalence checks failing, or its shift test rejecting; a phase-5 call or stop outside Task 7 Step 3's expectations.
