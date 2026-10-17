# Pruning through epistemetry, seqscreen deleted: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** reblock's screens (consolidated `screen-report`) decide through GeoffChurch/epistemetry, pinned at 452052c, by a decision-rule Strategy built once at the edge; seqscreen.py and epireplay.py are deleted, every caller moved and every stored decision re-derived.

**Architecture:** A new `research/roadless/screening.py` holds a screen's configuration and its live loop.
The configuration is the rule Strategy (`Rule`: `RealizedLoss`, the default, and `ZeroOne`) and `Screening` (the rule plus every other result-changing number of a screen), all `Recordable`, so `Screening.record()` is the run's identity.
The live loop runs epistemetry's `ValueOfMoreUnits` on a screen whose later blocks are not run yet.
consolidated.py's edge builds the named presets and injects one into `screen-report`, which never asks which rule it holds.
common.py holds the block sets (TUNING, HELD).
`research/roadless/screening_check.py` holds the guards, each failing under a named fault.
A temporary `research/roadless/screening_migration.py` proves the move against the fourth replay's stored runs before anything is deleted.

**Tech Stack:** Python 3.12 (uv), numpy, pandas, GeoffChurch/epistemetry 452052c (`design.decide`, `design.replay`, `stats.records`).
research/roadless has no test framework: the guards follow gap_clear_check.py's convention (a guard returns its failures; under a fault it must fail, or the run exits non-zero saying the fault went unseen).

**Spec:** research/roadless/NOTES.md, "Pruning by epistemetry: the realized loss as the target" (the fourth replay: its rule, numbers, guards and feedback); research/roadless/BACKLOG.md, "Pruning by epistemetry's decision layer" (with the lead added at 7d8e87f0 on what 452052c changes); the owner's decision of 2026-10-17: the consolidated pass prunes with seqscreen (its screens are done), then reblock moves to epistemetry and seqscreen is deleted, so that epistemetry is dogfooded by several projects rather than shadowed by a second decision layer inside reblock.

## Global Constraints

Copied from the brief; every task's requirements include this section.

**Sequencing**
- Execution starts only after the consolidated pass returns. Until then another agent edits NOTES.md, BACKLOG.md, consolidated.py and the consolidated_rows CSVs; locate every edit by its function name or heading, never by a line number from this plan.
- Execution is subagent-driven (superpowers:subagent-driven-development), implementers and reviewers on opus.

**No legacy shims**
- When a format changes, migrate the data and delete the old path.
- Never keep a reader for a format no longer written.

**Strategy pattern**
- Resolve a choice once, upstream (a Protocol plus an injected instance).
- No string registries or enum switches for a fixed set. The one name lookup is the command line's preset, resolved once in `consolidated.main` (a CLI argument: dynamic access with no default).

**No closed-set dynamic access**
- No getattr/.get defaults on closed sets; frozen dataclasses and NamedTuples.
- Result-changing parameters are required and keyword-only below the edge, and part of the run's identity (here `Screening.record()`).
- JSON (the fixture, the decision records) is read by key, with no default, at one boundary function each.

**Keep the frontier, delete the dominated**
- seqscreen is not dominated on every axis by the numbers: the replays tie it within noise (the fourth replay: loss below it at every rate, one wrong call and four asks more in 2,600 pairwise runs).
- It is deleted by the owner's decision to dogfood epistemetry, not by dominance. The NOTES say so plainly.

**Guards**
- Every guard is shown to fail under fault injection: run it as it is (passes), then under each of its faults (fails), with `both-ways`.
- Guards run from the repo root with `PYTHONPATH=.`, single-threaded: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1`.
- A fault is a patch inside the guard's process (gap_clear_check.py's `_patched`); no file is edited to inject one.

**Repo and process rules**
- Commit with `git -c core.hooksPath=.githooks commit -m "<subject>" -m "<body>" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RoNpoEMZ1kz2hogCsAsmXJ" -- <paths>` (a new file is `git add`ed first; a deletion is `git rm`).
- Subjects follow the branch's style: `research: roadless -- <what, with its numbers>`.
- Push after each commit: `git push origin research/roadless`. Never push main.
- Never touch research/roadless/stopping_review/, research/group_steiner/, or research/roadless/{ids58,ids59,ov16,ov19,ov_a,ov_b,ss_miss}.txt.
- Don't edit src/reblock. ~/src/epistemetry and ~/src/cluster_submit are read-only (read epistemetry with `git -C ~/src/epistemetry show 452052c:<path>`).
- At most 8 local processes in total; single-threaded numeric pools; kill by PID; no polling loops. A run longer than a few minutes goes through the Bash tool's `run_in_background`, which notifies on exit.
- Never `uv cache clean`: every checkout's .venv links into uv's cache (2c7ccef2).
- NOTES.md and BACKLOG.md are column-wrapped at 99; this plan and other new Markdown is one sentence per line.
- Ruff skips research/ (pyproject `extend-exclude`), so the gate is explicit paths: the two new modules pass `uv run ruff check research/roadless/screening.py research/roadless/screening_check.py` (the repo's rules, lines of 100); every touched existing module passes `uv run ruff check --select F,E9 <path>` (check it passes before editing).
- Types: `MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent <paths>` reports no error in screening.py or screening_check.py; for a touched existing module the count of errors it reports does not rise (record it before editing).

## Rulings (owner to check)

1. **Target commit 452052c4694188cf3c6523370cdc1a28c95fa684**, the newest on GeoffChurch/epistemetry's master (2026-10-17). It has everything the move needs: `AdoptKeepAskByStake(rates, weights, uses)` with `target.price`, every rule `Recordable` with `stats.records.revive`, value equality for `Design`, `Order`, `Truth`, `Run`. A newer commit is a later pin move, checked by the `acceptance` guard (Task 6).
2. **The pin lives in pyproject.toml**, as cluster_submit's does: epistemetry joins the `research` dependency group with `[tool.uv.sources]` at the full SHA. Today's mechanism (`uv run --with "epistemetry @ git+...@674dba4"`, an ephemeral overlay named only in epireplay.py's docstring) goes with epireplay.py. epistemetry requires Python >=3.12 and uv refuses the lock while reblock declares >=3.11 (checked in a scratch project), so research/roadless's `requires-python` rises to >=3.12 (the venv is 3.12.13; main is untouched). cluster_submit mirrors every git source of uv.lock to the cluster before a run, so the nodes get epistemetry without GitHub access.
3. **Two presets, set at consolidated's edge.** `realized-loss` (the default: AdoptKeepAskByStake over the bracket's five geometric rates, equally weighted, over FUTURE uses; the lowest realized loss at every rate in the fourth replay). `fewest-wrong` (AdoptKeepAsk over the bracket's ends, each arm priced by its own stake; 37 wrong calls against 67 in 2,600 pairwise runs, at 318 asks against 284 and more loss at the bracket's bottom and middle). The second stays because the fourth replay puts it on the frontier; say so if it should go to the dead-end record instead, and Task 3 drops `ZeroOne` and `OwnStake`. The nine-point sensitivity is no preset: asks are free (owner 2026-10-16), so no operating point wants fewest asks, and its numbers stay in NOTES. `SharedStake` is not ported: on one arm it is the own price, and reblock screens pairwise.
4. **Block sets in common.py as literals.** `common.TUNING` (13) and `common.HELD` (46, sorted) beside `common.large_blocks()`; HELD stops being derived from the git-ignored raster-era large82_simp_vs_greedy.parquet. The held-out order is epistemetry's `random_orders(46, count=1, rng=np.random.default_rng(0))`: the same permutation seqscreen used (checked), its stream that generator's first spawned child. The rates, FUTURE, the floor, the collapse, the Monte Carlo sizes and the order's seed are constants at consolidated's edge, recorded in `Screening.record()`.
5. **Deleted with seqscreen, no caller outside it:** its loaders (`simp`, `polished`, `searched`, `LOADERS`, `specs_of`), the `next` command, `question`, `ORDERS`, `STOPPED`, `POWER`. A live screen of a non-SIMP variant would need a runner like consolidated `screen`. The ask's number survives as summary_screen.csv's `break_even` column.
6. **The replays' data stays as records no code reads**, as phase 3 kept the raster's rows: epireplay*.parquet, epireplay*_jobs/, epireplay*_screens.parquet and epireplay4_*.log (git-ignored, about 35 MB; their decision records are 674dba4's format, which nothing will read). Say "delete" and Task 8's last step removes them. epireplay_designs.json becomes screen_designs.json, reblock's real-data acceptance fixture: epistemetry's spec (owner, 2026-10-16) puts each project's real-data acceptance test in that project, run through its pin. The only stored seqscreen decisions anything reads are summary_screen.csv's, which Task 7 re-derives; screen_rows/ and the store's clearings/screen_rows CSVs hold the screens' runs, not decisions, and stay as they are.
7. **Numbers that may move.** In the fourth replay's runs, none of the stops, calls, GPU-seconds, wrong calls or asks: at 452052c both sides of the stop scale by `uses` (820). Its recorded prices and readings are 820 times larger (the price PRICE_RATE / FUTURE = 1.01e-8 becomes the mean rate 8.30e-6). In phase 5, `.r2l2`'s stop may move (and the columns computed over the blocks a stop used); its call should not; the other five screens should not move. Task 4 and Task 7 say how each is checked.
8. **Unchanged modelling choices:** pairwise screens only (consolidated screens each variant against the default, as today; the many-arm screen is not used); the base's GPU time charged 0, as seqscreen and every replay charged it.

## Review Focus

1. A held-out block with no default time in the pass's rows (the cost model's base seconds would be NaN): `screen_table` raises naming the block before any decision. Pinned by Task 7's `screens` guard (its negative case).
2. A screen whose rule wants a block not run yet, including one with fewer blocks run than the floor: `Running`, its summary row with empty stop and call and its next wave listed, never a call. Pinned by Task 5's `live` guard (its truncated screens).
3. A collapse inside the floor's first step (C1m4/p64 under the veto): `Stopped(why="collapse", call=KEEP)` at the collapsing block. Pinned by Task 5's `live` guard.
4. A screen with no more blocks than the floor (C1m4/p64 without the veto, 12 blocks; `.m2`, 15): `Stopped(why="ran out")` on all of them. Pinned by Task 5's `live` guard and Task 7's phase-5 check.
5. A variant's rows that are not a prefix of its order (a block run out of order, or the order's seed changed): SystemExit naming the variant before any decision. Pinned by Task 7's `screens` guard (fault `shuffled`).

## Files

- Create `research/roadless/screening.py`: the rule Strategy, `Screening`, the cost model, the live loop, the decision records.
- Create `research/roadless/screening_check.py`: the fixture's loader and the guards `target`, `records`, `live`, `acceptance`, `screens`.
- Create `research/roadless/screen_acceptance.csv`: the acceptance guard's expected runs (Task 4 writes it).
- Create `research/roadless/consolidated_rows/screen_decisions.json`: the screens' decisions, rules as records (Task 7).
- Create, then delete: `research/roadless/screening_migration.py` (Tasks 4 and 8).
- Rename: `research/roadless/epireplay_designs.json` to `research/roadless/screen_designs.json` (Task 3).
- Modify: `pyproject.toml`, `uv.lock`, `research/roadless/common.py`, `research/roadless/consolidated.py`, `research/roadless/region_leaks.py`, `research/roadless/consolidated_rows/summary_screen.csv`, `research/roadless/NOTES.md`, `research/roadless/BACKLOG.md`.
- Delete: `research/roadless/seqscreen.py`, `research/roadless/epireplay.py`.

---

### Task 1: Pin epistemetry at 452052c

**Files:**
- Modify: `pyproject.toml` (`[project] requires-python`, `[dependency-groups] research`, `[tool.uv.sources]`)
- Modify: `uv.lock` (regenerated)

**Interfaces:**
- Produces: `import epistemetry` in the project's environment at 452052c4694188cf3c6523370cdc1a28c95fa684, for every later task and on every cluster node.

- [ ] **Step 1: Edit pyproject.toml**

`requires-python = ">=3.12"`; append `"epistemetry",` to the `research` group; under `[tool.uv.sources]`, beside cluster_submit:

```toml
epistemetry = { git = "https://github.com/GeoffChurch/epistemetry.git", rev = "452052c4694188cf3c6523370cdc1a28c95fa684" }
```

Leave `[tool.ruff] target-version` and `[tool.mypy] python_version` as they are (they serve src/reblock, which does not import epistemetry).

- [ ] **Step 2: Lock, and check nothing else moved**

Run: `uv lock`, then `git diff uv.lock | grep -E '^[-+](name|version) = '`
Expected: only `+name = "epistemetry"` and `+version = "0.1.0"`. If any other package's version changes, stop and report; do not pass `--upgrade`.

- [ ] **Step 3: Sync and check the import**

Run: `uv sync && uv run python -I -c "from epistemetry.design.decide import AdoptKeepAskByStake as T; print(T(rates=(1e-6,), weights=(1.0,), uses=820.0).price)"`
Expected: `1e-06`. Then `grep -c 'epistemetry.git?rev=452052c4694188cf3c6523370cdc1a28c95fa684#452052c4694188cf3c6523370cdc1a28c95fa684' uv.lock` prints `1` or more.

- [ ] **Step 4: Check cluster_submit can mirror it**

Run: `uv run python -I -c "import pathlib; from cluster_submit.envs import locked_git_sources; print(locked_git_sources(pathlib.Path('uv.lock').read_text()))"`
Expected: two sources, cluster_submit's and epistemetry's, and no SystemExit (the shape `<url>?rev=<sha>#<sha>` is one it knows).

- [ ] **Step 5: Commit and push**

Subject: `research: roadless -- pin GeoffChurch/epistemetry 452052c in the research group (pyproject [tool.uv.sources], as cluster_submit): requires-python >=3.12 (epistemetry requires it; uv refuses the lock otherwise); no other package moved; cluster_submit mirrors the new git source`
Paths: `pyproject.toml uv.lock`

---

### Task 2: The block sets in common.py

**Files:**
- Modify: `research/roadless/common.py` (after `large_blocks`)
- Modify: `research/roadless/region_leaks.py` (`import seqscreen`; `TUNING = seqscreen.TUNING` and its three uses)
- Modify: `research/roadless/consolidated.py` (its copy of `TUNING` in the report section, and `report`'s use)

**Interfaces:**
- Produces: `common.TUNING: tuple[str, ...]` (the 13 large blocks the presets were tuned on, seqscreen.TUNING's order) and `common.HELD: tuple[str, ...]` (the 46 other large blocks that fit the retired raster's greedy at h 0.5, sorted). Later tasks read both.

- [ ] **Step 1: Add the two constants**

`TUNING` is seqscreen.TUNING's literal, moved. `HELD` is the literal of `seqscreen.blocks_of("held")`, printed once with `PYTHONPATH=. uv run python -c "import sys; sys.path.insert(0, 'research/roadless'); import seqscreen; print(seqscreen.blocks_of('held'))"`. Each gets a comment with its provenance (BACKLOG "Tune on the 13, report on the 44"; HELD: "the large blocks large82_simp_vs_greedy.parquet holds, less TUNING, frozen 2026-10-17").

- [ ] **Step 2: Check them against seqscreen, once**

Run: `PYTHONPATH=. uv run python -c "import sys; sys.path.insert(0, 'research/roadless'); import common, seqscreen; assert common.TUNING == seqscreen.TUNING; assert common.HELD == tuple(seqscreen.blocks_of('held')); assert len(common.HELD) == 46 and not set(common.HELD) & set(common.TUNING); assert set(common.HELD) | set(common.TUNING) <= set(common.large_blocks()); print('ok')"`
Expected: `ok`. This is a migration check, not a guard: it reads the parquet this task stops reading.

- [ ] **Step 3: Move the callers**

region_leaks.py: drop `import seqscreen` and the alias; its uses read `common.TUNING`. consolidated.py: delete its copy (with the comment "seqscreen.TUNING (not imported: its loaders read the rows)"); `report` reads `common.TUNING`.

- [ ] **Step 4: Check imports and lint**

Run: `PYTHONPATH=. uv run python -c "import sys; sys.path.insert(0, 'research/roadless'); import region_leaks, consolidated; print(len(region_leaks.common.TUNING))"`
Expected: `13`. Then ruff `--select F,E9` on the three files passes.

- [ ] **Step 5: Commit and push**

Subject: `research: roadless -- the block sets in common.py (TUNING, 13; HELD, 46, the held-out screens' blocks, frozen from seqscreen.blocks_of("held") and checked equal): region_leaks and consolidated read common.TUNING (consolidated's copy gone); HELD no longer read from the raster-era large82_simp_vs_greedy.parquet`
Paths: `research/roadless/common.py research/roadless/region_leaks.py research/roadless/consolidated.py`

---

### Task 3: The rule Strategy and the screen's configuration

**Files:**
- Create: `research/roadless/screening.py`
- Create: `research/roadless/screening_check.py`
- Modify: `research/roadless/consolidated.py` (the screen's edge constants and `screenings()`, beside `SCREENS`)
- Rename: `research/roadless/epireplay_designs.json` to `research/roadless/screen_designs.json` (`git mv`, content unchanged); epireplay.py's `DESIGNS` follows it (one line), so it runs until Task 8 deletes it.

**Interfaces:**
- Consumes: Task 1's environment; `common.HELD` (Task 2) only in later tasks.
- Produces, in screening.py (every dataclass `frozen=True, kw_only=True`; every field required):
  - `MonteCarlo(Recordable)`: `outer: int, inner: int, blocks: int, draws: int`.
  - `class Rule(Protocol)`: `target(self) -> DrawTarget`, `stop_of(self) -> StopOf`, `record(self) -> dict[str, object]`.
  - `RealizedLoss(Recordable)`: `rates: tuple[float, ...]` (Lens A per GPU-second, increasing), `weights: tuple[float, ...]`, `uses: float`, `margin: float` (Monte Carlo standard errors a reading must clear). `target()` is `AdoptKeepAskByStake(rates=, weights=, uses=)`; `stop_of()` is `EveryArm(ValueOfInformation(price=self.target().price, margin=StandardErrors(k=self.margin)))`: the price is the target's, never stated beside it.
  - `OwnStake(Recordable)`, a `StopOf`: `price_rate: float, uses: float, margin: float`; `of(self, seen: Seen, *, arm: int) -> ValueOfInformation` prices arm `arm` at `price_rate / uses / stake`, `stake = float(abs(o[:, :1].mean(0) - price_rate * o[:, 1:].mean(0))[0])` over `o = seen.arms[arm].outcomes` (epireplay.stake's expression, for the migration's parity).
  - `ZeroOne(Recordable)`: `rates, weights, uses, margin` as `RealizedLoss`; `target()` is `AdoptKeepAsk(low=rates[0], high=rates[-1])`; `stop_of()` is `OwnStake(price_rate=float(np.asarray(rates) @ np.asarray(weights)), uses=, margin=)`.
  - `RatioToBase` (`frozen=True, eq=False, kw_only=True`; data, not configuration: an epistemetry `CostModel`): `base_seconds: np.ndarray`; `predict(arm, seen, seen_costs, upcoming)` and `predict_base(seen, seen_costs, upcoming)` both return `float(seen_costs.sum() / self.base_seconds[seen].sum() * self.base_seconds[upcoming].sum())` (epireplay.RatioToBase's expression, exactly).
  - `Screening(Recordable)`: `rule: Rule, floor: int, collapse: float, step: int, mc: MonteCarlo, order_seed: int`, with
    - `policy(self, base_seconds: np.ndarray) -> ValueOfMoreUnits`: `belief=Bootstrap()`, `functionals=np.eye(2)` (the outcome columns: gain, extra GPU-seconds), `target=self.rule.target()`, `horizons=tuple(range(1, len(base_seconds) + 1))`, `step=self.step`, `cost=RatioToBase(base_seconds=base_seconds)`, `stop=self.rule.stop_of()`, `floor=self.floor`, and `outer`, `inner`, `blocks`, `draws` from `self.mc`;
    - `drop(self) -> VetoBelow`: `VetoBelow(threshold=-self.collapse, column=0, call=KEEP)`;
    - `order(self, units: int) -> Order`: `random_orders(units, count=1, rng=np.random.default_rng(self.order_seed))[0]`.
  - `decision_record(d: Decision) -> dict[str, object]`: `{"at", "arm_units", "live", "step": {"units", "arms"}, "rules": {str(j): rule.record()}, "readings": {str(j): reading}}`, JSON as it stands.
  - `decision_of(record: Mapping[str, object]) -> Decision`: the inverse, each rule through `revive`, refused (ValueError) unless a `ValueOfInformation` (both presets' stops record one).
  - `CALLS: dict[int, str] = {KEEP: "keep", ASK: "ask", ADOPT: "adopt"}`.
- Produces, in consolidated.py (the edge):
  - `RATES_PER_GPU_SECOND = (2e-6, 1.9e-5)`, `RATE_POINTS = 5`, `FUTURE = 820.0`, `SCREEN_FLOOR = 15`, `COLLAPSE = 0.1`, `SCREEN_MC = screening.MonteCarlo(outer=4000, inner=200, blocks=20, draws=4000)`, `SCREEN_ORDER_SEED = 0`, each with seqscreen's comment on where it came from (RATES' two calls on record, MIN_BLOCKS' replay, COLLAPSE's BACKLOG entry, FUTURE's horizon).
  - `screenings() -> dict[str, screening.Screening]`: `"realized-loss"` (`RealizedLoss`) and `"fewest-wrong"` (`ZeroOne`), both over `rates = tuple(float(r) for r in np.geomspace(*RATES_PER_GPU_SECOND, RATE_POINTS))`, `weights = (1.0 / RATE_POINTS,) * RATE_POINTS`, `uses=FUTURE`, `margin=0.0`, and the same floor, collapse, `step=1`, MC and seed. Each preset's comment says what it optimizes and what it gives up (Ruling 3).
- Produces, in screening_check.py:
  - `Fixture(NamedTuple)`: `name: str, many: bool, blocks: tuple[str, ...], gain: np.ndarray (u, k), seconds: np.ndarray (u, k), base_seconds: np.ndarray (u,), orders: tuple[np.ndarray, ...]`, with `design(self) -> Design` (`outcomes = np.stack([gain, seconds - base_seconds[:, None]], axis=2)`, `costs = seconds`, `base_costs = np.zeros(u)`).
  - `designs() -> dict[str, Fixture]`: screen_designs.json, every key read by name, shapes checked (epireplay._plan's checks), in the file's order.
  - `realized_risk(draws: np.ndarray, *, rates: tuple[float, ...], weights: tuple[float, ...]) -> tuple[float, int]`: outside epistemetry, one arm's (gain, extra) draws to the weighted mean over rates of E|g - r c| where the posterior-mean call at r is wrong, and the call over the rates (epireplay.realized_risk, weighted).
  - `own_stake(outcomes: np.ndarray, *, price_rate: float) -> float`: outside epistemetry, |mean gain - price_rate x mean extra| over one arm's prefix (outcomes (n, 2)).
  - The guard registry and CLI: `GUARDS: dict[str, Callable[[], list[str]]]`, `FAULTS: dict[str, dict[str, Callable[[], AbstractContextManager[None]]]]`, `main(guard: str, fault: str | None)`, `both_ways(guards: list[str])`, as gap_clear_check.py's.

- [ ] **Step 1: Write the configuration in screening.py**

The module docstring says what a screen is now (NOTES, the move's section, Task 9), that the pin lives in pyproject.toml, and how a pin moves: edit the `rev`, `uv lock --upgrade-package epistemetry`, run `screening_check.py both-ways target,records,live,acceptance,screens`, commit.

- [ ] **Step 2: Add the edge constants and `screenings()` to consolidated.py**

- [ ] **Step 3: Rename the fixture**

Run: `git mv research/roadless/epireplay_designs.json research/roadless/screen_designs.json`, and point epireplay.py's `DESIGNS` at the new name.

- [ ] **Step 4: Write the guard `target`**

For each preset of `consolidated.screenings()`, on every pairwise fixture, order 0's first 15, 20 and 25 blocks (as many as the design has), with draws `Bootstrap().of(outcomes[prefix, 0], population=u).draw(np.eye(2), np.random.default_rng([7, i]), draws=SCREEN_MC.draws)` and `seen` that prefix's `Seen`:

```python
# target = config.rule.target(); stop_of = config.rule.stop_of(); rates, weights and
# price_rate = rates @ weights from the edge's constants, FUTURE the edge's
# realized-loss:
loss, call = realized_risk(draws, rates=rates, weights=weights)
assert math.isclose(float(target.risk(draws)), FUTURE * loss, rel_tol=1e-9, abs_tol=1e-18)
assert int(np.asarray(target.decide(draws)).reshape(-1)[0]) == call
rule = revive(stop_of.of(seen, arm=0).record())
assert isinstance(rule, ValueOfInformation)
assert math.isclose(rule.price, float(np.asarray(rates) @ np.asarray(weights)), rel_tol=1e-15)
assert rule.margin.k == 0.0
# fewest-wrong: the arm's own stake, recomputed here from the prefix
assert math.isclose(rule.price, price_rate / FUTURE / own_stake(outcomes[prefix, 0], price_rate=price_rate),
                    rel_tol=1e-12)
```

Faults: `unpriced` (`RealizedLoss.stop_of` patched to price at `target.price / target.uses`, 674dba4's pairing beside a target with uses: the price check fails); `dropped-uses` (`RealizedLoss.target` patched to build its target with `uses=1.0`: the risk check fails); `unstaked` (`OwnStake.of` patched to leave the stake out: the own-price check fails).

- [ ] **Step 5: Write the guard `records`**

For each preset: `revive(config.record()) == config` and `revive(json.loads(json.dumps(config.record()))) == config`. Then `replay` (epistemetry's) under the preset on x2/p64's orders 0-1 and 2/p64's order 0 (two arms), with `config.policy(f.base_seconds)`, `Truth.of(design, policy, rng=np.random.default_rng(config.order_seed), drop=config.drop())` and `AsksApart(ask=ASK)`; for every decision `d` of every run:

```python
assert decision_of(json.loads(json.dumps(decision_record(d)))) == d
```

Fault: `altered-record` (`screening.decision_record` patched to write each rule's price times 1 + 1e-6: the round trip differs).

- [ ] **Step 6: Run both ways**

Run: `PYTHONPATH=. uv run python research/roadless/screening_check.py both-ways target,records` (single-threaded env)
Expected: `target: passes`, `records: passes`, then each fault's line `FAILS on ...`, and `both ways, target, records: as expected`.

- [ ] **Step 7: Lint and types**

Run the ruff and mypy gates (Global Constraints) on screening.py, screening_check.py and consolidated.py.

- [ ] **Step 8: Commit and push**

Subject: `research: roadless -- screening.py, the screen's configuration on epistemetry 452052c: the rule a Strategy (Rule: RealizedLoss, the price its target's; ZeroOne, each arm at its own stake), Screening (floor, collapse, step, Monte Carlo, order seed) Recordable as the run's identity; consolidated's edge holds the constants and the presets realized-loss and fewest-wrong; the frozen designs renamed screen_designs.json (reblock's real-data acceptance fixture); guards target and records, each failing under its faults (unpriced, dropped-uses, unstaked; altered-record)`
Paths: `research/roadless/screening.py research/roadless/screening_check.py research/roadless/consolidated.py research/roadless/screen_designs.json research/roadless/epireplay_designs.json research/roadless/epireplay.py`

---

### Task 4: Equivalence with the fourth replay, and the acceptance fixture

The fourth replay's realized-loss runs (674dba4) are re-run through `screenings()["realized-loss"]` at 452052c and compared run by run; a subset of its 0-1 rows through `screenings()["fewest-wrong"]`.
What 452052c changes is the scale: the target's risk is `uses` (820) times larger and its price the mean rate, so every value and price is 820 times the stored one, and no stop or call may move unless a reading sat within rounding of its price.

**Files:**
- Create: `research/roadless/screening_migration.py` (temporary; Task 8 deletes it)
- Create: `research/roadless/screen_acceptance.csv`
- Its jobs: `research/roadless/screening_migration_jobs/` (git-ignored by `*.parquet`)

**Interfaces:**
- Consumes: `consolidated.screenings()`, `Screening.policy`, `screening_check.designs()`, `Fixture.design()`, epistemetry's `replay`, `Truth.of`, `Order`, `AsksApart`, `NoVeto`.
- Reads: `research/roadless/epireplay4.parquet` (its rows with `policy == "by stake step 1, 5 rates"`, 3,100, and `"adopt/keep/ask step 1, own price"` for x2/p64 orders 0-19 and 2/p64 orders 0-9, no veto). This is the one reader of 674dba4's rule records, and it goes in Task 8.
- Produces: `screen_acceptance.csv` with columns `plan, veto, order, units, call, cost, wrong, asks, full`: the 13 pairwise designs, orders 0-9, both settings (260 rows), from this task's realized-loss runs.

- [ ] **Step 1: Write the harness**

`run <workers>`: one job per (design, setting, preset), a parquet each, resumable (a job whose file exists is skipped).
- Realized loss: 13 pairwise designs in both settings and 5 many-arm designs without the veto (31 jobs).
- Fewest wrong: x2/p64 orders 0-19 and 2/p64 orders 0-9, no veto (2 jobs).
- Each order is `Order(units=f.orders[o], rng=np.random.default_rng([0, o]).spawn(2)[1])`, the stream every replay of these designs drew from.
- The truth is `Truth.of(design, policy, rng=np.random.default_rng(0), drop=<the setting's>)`: `config.drop()` with the veto, `NoVeto()` without.
- Scoring is `AsksApart(ask=ASK)`.
- Rows as epireplay's `run_orders` wrote them, decisions through `decision_record`.

`compare`: for every stored row, the new row on the same (plan, arms, veto, order):

```python
assert (new.units, new.arm_units, new.calls, new.cost, new.wrong, new.asks, new.full, new.vetoed, new.given) \
    == (old.units, old.arm_units, old.calls, old.cost, old.wrong, old.asks, old.full, old.vetoed, old.given)
# per decision: at, arm_units, live, step's units and arms equal; per judged arm:
assert math.isclose(new_rule.price, scale * old_record["price"], rel_tol=1e-12)
assert new_rule.margin.k == old_record["margin_k"]
# every reading key equal; each value None in both, or
assert abs(new_value - scale * old_value) <= tol
```

For the realized loss, `scale = 820` and `tol = 1e-9 * target.price`. For fewest wrong, `scale = 1`, `tol = 0` (its target and its price are unchanged; the price's rate is `weights @ rates` where 674dba4's was `BRACKET.mean()`, hence the rel 1e-12 on prices).
A run that differs is printed with its first differing decision and, for the horizon that decided it, `(value_per_cost - price) / price`.
It prints the counts in all, and writes screen_acceptance.csv only when every run agrees.

`fault unpriced`: x2/p64 orders 0-4 run now with the stop priced at `target.price / target.uses`, compared as above. It must report differing runs, and exits 0 only if it does.

- [ ] **Step 2: Run it in the background**

Run (with `run_in_background`, single-threaded env, at most 6 workers): `PYTHONPATH=. uv run python research/roadless/screening_migration.py run 6`
Expected: 33 jobs written. The fourth replay spent 15.1 worker-hours on the realized loss, so about 3 h; its longest job (both/p64) about 84 min. Task 5's code may be written meanwhile.

- [ ] **Step 3: Compare**

Run: `PYTHONPATH=. uv run python research/roadless/screening_migration.py compare`
Expected:
- realized loss: 3,100 of 3,100 runs equal (2,600 pairwise, 500 many-arm), every price 820 times the stored one, every reading within 1e-9 of the price, with the worst difference printed;
- fewest wrong: 30 of 30 runs equal, readings bit for bit;
- screen_acceptance.csv written (260 rows).
A difference whose decisive reading sits within 1e-12 of its price relative is a rounding tie: record it in the commit message and in Task 9's NOTES. Any other difference stops the plan: report it, do not delete anything.

- [ ] **Step 4: Show the comparison sees a change**

Run: `PYTHONPATH=. uv run python research/roadless/screening_migration.py fault unpriced`
Expected: differing runs reported (the screens run past their stops), exit 0.

- [ ] **Step 5: Commit and push**

Subject: `research: roadless -- the move's equivalence: the fourth replay's realized-loss runs at 452052c through screening (realized-loss), N of 3,100 equal (blocks, calls, GPU-seconds, wrong calls, asks, truth), prices and readings 820 (uses) times the stored, within 1e-9 of the price; the 0-1 rows (fewest-wrong) 30 of 30, bit for bit; the comparison sees the unpriced fault; screen_acceptance.csv (260 pairwise runs, orders 0-9, both settings) the acceptance guard's expected runs`
Paths: `research/roadless/screening_migration.py research/roadless/screen_acceptance.csv`

---

### Task 5: The live loop

`Screening.decide` runs the policy on a screen whose later blocks are not run, exactly as epistemetry's `replay` would run it on the finished screen: the same decisions, the same random numbers, the same stop and call.

**Files:**
- Modify: `research/roadless/screening.py`
- Modify: `research/roadless/screening_check.py` (guard `live`)

**Interfaces:**
- Consumes: Task 3's `Screening`, `RatioToBase`, `designs()`.
- Produces:
  - `Screen(NamedTuple)`: `gain: np.ndarray` (n,), b - a on the blocks run, in the order; `seconds: np.ndarray` (n,), b's GPU-seconds on them; `base_seconds: np.ndarray` (u,), a's on every block of the order, those run then the rest.
  - `Why = Literal["settled", "collapse", "ran out"]`.
  - `Stopped(NamedTuple)`: `n: int, why: Why, call: int, decisions: tuple[Decision, ...]`.
  - `Running(NamedTuple)`: `n: int, decisions: tuple[Decision, ...]`.
  - `Screening.decide(self, s: Screen, *, stream: np.random.Generator) -> Stopped | Running`.

- [ ] **Step 1: Write the guard `live` first**

On the fixtures r2/base (a near-tie ask), x2/p64 (an ask), pPg/p256 (a keep) and C1m4/p64 (a collapse, 12 blocks), each in both settings, plus a degenerate design (p64/base with every gain 0), orders 0-2, under `screenings()["realized-loss"]`:

```python
run = replay(design, config.policy(f.base_seconds), orders=[order], truth=truth, scoring=AsksApart(ask=ASK)).runs[0]
i = order.units
v = config.decide(Screen(gain=f.gain[i, 0], seconds=f.seconds[i, 0], base_seconds=f.base_seconds[i]), stream=order.rng)
assert isinstance(v, Stopped) and v.n == run.units and v.call == int(run.calls[0, 0])
assert v.decisions == run.decisions
# the same screen cut short of its stop runs on, its decisions the replay's first ones
cut = config.decide(Screen(gain=f.gain[i[:run.units - 1], 0], seconds=f.seconds[i[:run.units - 1], 0],
                           base_seconds=f.base_seconds[i]), stream=order.rng)
assert isinstance(cut, Running) and cut.decisions == run.decisions[:len(cut.decisions)]
```

The cut case applies where `run.units > 1` and the run did not stop on a collapse. With no veto, an arm whose rule is met on a near tie may stop anywhere from the floor; the comparison is order by order, so this is exact.
Absolute expectations, beside the replay:
- the degenerate design stops at `config.floor` with KEEP, "settled", in every order;
- C1m4/p64 under the veto stops "collapse" with KEEP at the position of its first block below `-config.collapse`;
- without the veto it stops "ran out" at 12 (Review Focus 3, 4).
`run.decisions` holds epistemetry's own `Decision`s, so `==` compares rules (Recordable dataclasses) and readings exactly.

Faults:
- `no-veto`: `Screening.drop` patched to `NoVeto()`; C1m4's collapse goes unseen.
- `floor-ignored`: `Screening.policy` patched to build its policy with `floor=1`; the degenerate design stops at 1, not at `config.floor`.
- `uncopied`: `decide` draws from the caller's stream itself rather than a copy; the cut screen, decided on the same `order.rng` after the full one, differs.

- [ ] **Step 2: Run it to see it fail**

Run: `PYTHONPATH=. uv run python research/roadless/screening_check.py live`
Expected: AttributeError on `Screen` (not written yet).

- [ ] **Step 3: Write `decide`**

It mirrors `epistemetry.design.replay.replay`'s loop for one arm, on positions 0..u-1 of the order:
- positions index `s.base_seconds`, so the cost model sees what the replay's sees;
- `r = _fresh(stream)`, one generator for every decision and the final call; `_fresh` is a module-level function returning `copy.deepcopy(stream)`, the seam the `uncopied` fault patches to return the stream itself;
- each decision's `Seen` is the arm's prefix (`Prefix(units=np.arange(n), outcomes=<gain, extra>[:n], costs=s.seconds[:n])`), `base=np.arange(n)`, `base_costs=np.zeros(n)`, `live`, `upcoming=np.arange(n, u)`;
- the `Decision` is recorded before acting, as `replay` records it;
- a step's units are taken one at a time, each through `self.drop().verdict(np.array([gain, extra]))`, so a collapse inside a step stops it there;
- the screen is `Running` the moment it needs a position at or past `len(s.gain)`;
- `STOP` before the last position is "settled"; reaching position u is "ran out"; a veto is "collapse" with the drop rule's call; otherwise the call is `policy.calls(seen, rng=r)[0, 0]`.

- [ ] **Step 4: Run both ways**

Run: `PYTHONPATH=. uv run python research/roadless/screening_check.py both-ways live`
Expected: `live: passes`, each of `no-veto`, `floor-ignored`, `uncopied` `FAILS on ...`, `as expected`.

- [ ] **Step 5: Lint, types, commit and push**

Subject: `research: roadless -- screening.Screening.decide, the live screen: epistemetry's ValueOfMoreUnits on the blocks run so far, a Stopped (settled, collapse, ran out) or Running verdict with its decisions; guard live: on r2/base, x2/p64, pPg/p256, C1m4/p64 (both settings) and a degenerate design, the live screen is epistemetry's replay run for run, decision for decision, and cut short it runs on; faults no-veto, floor-ignored, uncopied`
Paths: `research/roadless/screening.py research/roadless/screening_check.py`

---

### Task 6: The acceptance guard

reblock's real-data acceptance test for epistemetry: at any pin, the production preset reproduces the 260 runs Task 4 verified.

**Files:**
- Modify: `research/roadless/screening_check.py` (guard `acceptance`)

**Interfaces:**
- Consumes: `screen_acceptance.csv` (Task 4), `designs()`, `screenings()["realized-loss"]`, Task 4's order streams (`np.random.default_rng([0, o]).spawn(2)[1]`) and truths.
- Produces: guard `acceptance`, run on a pool of `--workers W` (single-threaded; a count that changes no result, so it is exempt from the required-parameter rule).

- [ ] **Step 1: Write the guard**

For each pairwise design, in the file's order, orders 0-9, both settings: the run's `units`, call, `cost`, `wrong`, `asks` and truth equal the CSV's row exactly. Under a fault it returns at the first design that differs.
Fault: `unpriced` (as in `target`).

- [ ] **Step 2: Run both ways**

Run: `PYTHONPATH=. uv run python research/roadless/screening_check.py both-ways acceptance --workers 6`
Expected: `acceptance: passes` (260 of 260; about 70 worker-minutes), then `FAILS on p64/base ...` under `unpriced`, `as expected`.

- [ ] **Step 3: Lint, types, commit and push**

Subject: `research: roadless -- guard acceptance, reblock's real-data acceptance test for epistemetry's pin (its spec, owner 2026-10-16): the realized-loss preset reproduces screen_acceptance.csv's 260 runs exactly (13 pairwise designs, orders 0-9, both settings); fault unpriced`
Paths: `research/roadless/screening_check.py`

---

### Task 7: consolidated `screen-report` decides through epistemetry

**Files:**
- Modify: `research/roadless/consolidated.py` (`screen_report`, `main`, the module docstring's screen lines and its `screen-report` paragraph, `_bb_interval`'s docstring)
- Modify: `research/roadless/consolidated_rows/summary_screen.csv` (re-derived)
- Create: `research/roadless/consolidated_rows/screen_decisions.json`
- Modify: `research/roadless/screening_check.py` (guard `screens`)

**Interfaces:**
- Consumes: `screenings()`, `Screening.order`, `Screening.decide`, `Screen`, `Stopped`, `Running`, `decision_record`, `decision_of`, `CALLS`, `common.HELD`.
- Produces:
  - `ScreenTable(NamedTuple)`: `table: pd.DataFrame` (summary_screen.csv's columns, plus `break_even` and `rule`), `decisions: dict[str, list[dict[str, object]]]` (variant to its decision records), `upcoming: dict[str, list[str]]` (each running screen's next `wave` blocks).
  - `screen_table(config: screening.Screening, *, wave: int) -> ScreenTable`: computes, writes nothing.
  - `screen_report(config: screening.Screening, *, wave: int) -> None`: writes `ROWS/summary_screen.csv` and `ROWS/screen_decisions.json` (`{"screening": config.record(), "screens": {variant: decisions}}`) and prints as today.
  - CLI: `screen-report <preset> <wave>`; `main` resolves the preset once with `screenings()[name]`, an unknown name a SystemExit listing the presets.

- [ ] **Step 1: Write the guard `screens` first**

The preset is the one whose Screening equals `revive(json["screening"])`; none is a failure. Then `t = screen_table(preset, wave=0)` against the committed files:
- `stop`, `call`, `used`, `eligible`, `run` exact; every float column within rel 1e-9;
- `t.decisions` equal to the JSON's, decision by decision, through `decision_of`, readings within rel 1e-9;
- every judged arm's verdict re-derived from the record is its take: `rule.met_reading(readings[j]) == (j not in step.arms)`;
- every rule equals the preset's (`EveryArm`'s one rule for realized-loss).

Negative case (Review Focus 1): with one rest block's default time dropped from the pass's rows (patched in memory), `screen_table` raises SystemExit naming that block.
Faults:
- `shuffled`: `Screening.order` patched to return its order reshuffled by `np.random.default_rng(1)`; `screen_table` raises "rows off its order", which the guard reports as a failure (Review Focus 5).
- `altered-record`: as in `records`; the re-derived decisions differ from the JSON's.

- [ ] **Step 2: Rewrite `screen_report` as `screen_table` and `screen_report`**

- The order is `config.order(len(common.HELD))`, its units indexing `common.HELD`; the variant's order and its checks are as today (relax.built_for_holds; rows a prefix; the default beside each).
- The base seconds are the default's re-run on the blocks run, then the pass's default times on the rest; a rest block without one raises naming it.
- `config.decide(screening.Screen(...), stream=order.rng)` gives the verdict: `Stopped` fills `used`, `stop` (its `why`) and `call` (`CALLS[call]`); `Running` leaves `stop` and `call` empty, `used` the blocks run, and lists its next wave.
- `break_even` is the mean gain over the extra GPU-seconds a block over the blocks used: the rate an ask turns on (seqscreen's `question` printed it).
- `rule` is `json.dumps(config.record(), sort_keys=True)`.
- The lens columns, time columns and B's order are computed as today.
- `import seqscreen` goes. `_bb_interval`'s docstring calls it the Bayesian bootstrap (Rubin 1981) rather than seqscreen's posterior: a reporting interval, not a decision.

- [ ] **Step 3: The phase-5 check, before anything is committed**

Run: `PYTHONPATH=. uv run python research/roadless/consolidated.py screen-report realized-loss 0`, then

```bash
PYTHONPATH=. uv run python -c "import io, subprocess, pandas as pd; f = 'research/roadless/consolidated_rows/summary_screen.csv'; old = pd.read_csv(io.StringIO(subprocess.check_output(['git', 'show', 'HEAD:' + f], text=True))).set_index('variant'); new = pd.read_csv(f).set_index('variant'); print(pd.concat({'seqscreen': old[['used', 'stop', 'call']], 'epistemetry': new[['used', 'stop', 'call']]}, axis=1))"
```

Expected, per the fourth replay (its realized loss made seqscreen's call in 1,286 of 1,300 runs a setting and stopped where it did in 1,152, earlier mostly on near-tie asks and clear keeps):
- every call equal: `r2l2` ask; `pP`, `seed`, `m2`, `mma`, `doc` keep;
- `pP`, `seed`, `mma`, `doc` 15 blocks, settled; `m2` 15, ran out;
- `r2l2` settled within its 24 blocks run, at 20 or another count. A different count is the methods' difference the fourth replay measured (r2/base: its stop earlier in 47 of 100 orders), not 452052c's.

If a call differs, if one of the five stops differs, or if `r2l2` runs on past its 24 blocks: restore the CSV (`git checkout -- research/roadless/consolidated_rows/summary_screen.csv`), delete the JSON, and stop to report. Running more blocks is GPU time and needs the owner's yes.

- [ ] **Step 4: Run both ways**

Run: `PYTHONPATH=. uv run python research/roadless/screening_check.py both-ways screens`
Expected: `screens: passes`, `shuffled` and `altered-record` each `FAILS on ...`, `as expected`.

- [ ] **Step 5: Lint, types, commit and push**

Subject: `research: roadless -- consolidated screen-report decides through epistemetry (screen-report <preset> <wave>, the preset a Screening built once at the edge and injected; realized-loss by default): phase 5's six screens re-derived, the calls seqscreen's (.r2l2 ask after N blocks; .pP, seed, MMA, damped OC keep at 15; .m2 keep, ran out at 15); summary_screen.csv gains break_even and rule (the Screening's record), screen_decisions.json each screen's decisions with its rules as records; guard screens re-derives both and each verdict from its record (faults shuffled, altered-record)`
Paths: `research/roadless/consolidated.py research/roadless/consolidated_rows/summary_screen.csv research/roadless/consolidated_rows/screen_decisions.json research/roadless/screening_check.py`

---

### Task 8: Delete seqscreen, epireplay and the migration

**Files:**
- Delete: `research/roadless/seqscreen.py`, `research/roadless/epireplay.py`, `research/roadless/screening_migration.py`

**Interfaces:**
- Consumes: nothing from them remains (Tasks 2 and 7 moved every caller).

- [ ] **Step 1: Delete**

Run: `git rm research/roadless/seqscreen.py research/roadless/epireplay.py research/roadless/screening_migration.py`

- [ ] **Step 2: No reader left**

Run: `git grep -n -E 'seqscreen|epireplay|screening_migration' -- research/roadless ':!research/roadless/NOTES.md' ':!research/roadless/BACKLOG.md' ':!research/roadless/stopping_review'`
Expected: no import, path or call; prose that names them as history (screen_designs.json's `meta.source`, a docstring citing NOTES) is all that may remain, each line read and listed in the commit message.

- [ ] **Step 3: Everything still runs**

Run: `PYTHONPATH=. uv run python -c "import sys; sys.path.insert(0, 'research/roadless'); import region_leaks, consolidated, screening, screening_check"`, then `PYTHONPATH=. uv run python research/roadless/screening_check.py both-ways target,records,live,screens` and `... both-ways acceptance --workers 6`.
Expected: the imports succeed; every guard passes and fails under each of its faults.

- [ ] **Step 4: Commit and push**

Subject: `research: roadless -- seqscreen.py and epireplay.py deleted (the owner's decision of 2026-10-17, to dogfood epistemetry; the replays tie seqscreen within noise, it is not dominated), with screening_migration.py, the move's one reader of 674dba4's records; no reader of either remains; every guard passes both ways after the deletion`
Paths: `research/roadless/seqscreen.py research/roadless/epireplay.py research/roadless/screening_migration.py`

- [ ] **Step 5: Only if the owner says "delete" to Ruling 6**

Run: `ls -d research/roadless/epireplay*.parquet research/roadless/epireplay*_jobs research/roadless/epireplay4_*.log research/roadless/screening_migration_jobs`, check the list holds nothing else, then `rm -r` exactly those paths. They are git-ignored: nothing to commit.

---

### Task 9: The records

**Files:**
- Modify: `research/roadless/NOTES.md` (a new section at the end)
- Modify: `research/roadless/BACKLOG.md` (located by the headings and text named below)
- Maybe: the wiki, `~/wiki` (GeoffChurch/wiki-private), through the `wiki` skill

**Interfaces:**
- Consumes: the numbers Tasks 4, 6 and 7 printed and committed.

- [ ] **Step 1: NOTES, "### Pruning by epistemetry: the move (owner 2026-10-17)"**

Run `git status` first; the pass's agent may have left NOTES mid-edit. Append the section after the last one, wrapped at 99, current numbers only, no before/after:
- why: the owner's decision, to dogfood epistemetry; seqscreen deleted by decision, not dominance; the fourth replay's tie in one sentence with its counts;
- what a screen is now: screening.py on 452052c (the rule Strategy, the presets and what each gives up, Screening's record as the identity, the live loop), the pin in pyproject and how it moves, Python >=3.12;
- the equivalence (Task 4's counts and worst reading difference);
- phase 5's verdicts as summary_screen.csv now holds them;
- the guards, each with its fault's result;
- what was deleted, and where the replays' data and the fixture stand (Ruling 6).

The consolidated pass's own section keeps what it says of the pass (it pruned with seqscreen); any number in it that summary_screen.csv no longer holds is edited to the CSV's value.
Check: `awk 'length($0) > 99' research/roadless/NOTES.md` prints nothing from the new section (`python3 .superpowers/sdd/2026-10-16-consolidated-pass/handoff_files/reflow.py` re-wraps a draft at 99, if it is still there).

- [ ] **Step 2: BACKLOG**

Wrapped at 99; current state only:
- "Pruning by epistemetry's decision layer" (Experiment design): done, the move; it replaces "seqscreen stays", the 452052c lead and "the pin moves only with a next replay" with the pin in pyproject and `screening_check.py acceptance` at every pin move.
- "The rate bracket" (Open): `consolidated.RATES_PER_GPU_SECOND` for `seqscreen.RATES`.
- "Sequential stopping" (How we run experiments) and the status lines naming seqscreen.py: what screens now (screening.py through epistemetry: ValueOfMoreUnits on the finite-population bootstrap, AdoptKeepAskByStake over the bracket's five rates, FUTURE uses, a 15-block floor, the collapse veto).
- "*Generalize kernelcore's outcome model*": the stop was built in seqscreen.py, now epistemetry's.
- Under "Not pursued, with what would change that":
  - *seqscreen* (deleted 2026-10-17 by decision, not dominance): it would come back if a pin move's acceptance guard shows epistemetry's screen leaving its decisions; the code is at the commit before Task 8's.
  - *Many-arm screens* (the per-arm screen, SharedStake): they would come back with a screen of several variants run together.
- The epistemetry issues filed in Task 10, by number, once filed.

- [ ] **Step 3: The wiki, only if something methodological was learned**

Candidates:
- a rounding tie found in Task 4 (a stop that moved under a pure rescaling of the loss);
- the live loop found to drift from `replay` in Task 5 (a policy's live loop and its replay must be one loop, as a rule's reading already is in epistemetry's stopping rules).
Use the `wiki` skill, `wiki search --lex "<question>"` from `~/wiki` first, and add to the page found rather than open a new one. Nothing learned: nothing written.

- [ ] **Step 4: Commit and push**

Subject: `research: roadless -- NOTES "Pruning by epistemetry: the move" and BACKLOG: screens decide through epistemetry 452052c (screening.py; presets realized-loss and fewest-wrong); seqscreen deleted by the owner's decision, not dominance; the equivalence, phase 5's verdicts and the guards`
Paths: `research/roadless/NOTES.md research/roadless/BACKLOG.md`

---

### Task 10: Feedback for epistemetry (needs the owner's yes)

**Filed 2026-10-17 at the owner's word ("Add all the items on GitHub as issues, including the three pieces as feature requests"):** items 1-9 below are GeoffChurch/epistemetry #1 (live driver), #2 (ValueOfMoreUnits Recordable), #3 (Decision records), #4 (cost-model covariates), #5 (NamedTuple revive, bug), #6 (stop priced from its target), #7 (kw_only), #8 (flat record namespace), #9 (Python floor, question). Item 10 is filed only if Task 4 finds a rounding tie. If epistemetry lands #1, #2, #3 or #4 before execution, the matching parts of Tasks 3, 5 and 7 use them instead of reblock's own code (Screening's hand-assembled identity, `decide`, `decision_record`/`decision_of`, RatioToBase's held data); the guards stay.

**Files:** none in the repo (BACKLOG gets the issue numbers in Task 9 Step 2, or a follow-up commit).

- [x] **Step 1: Ask the owner which to file** (all, 2026-10-17)

The issues, each with the reblock case behind it:
1. **A live driver for a replayed policy.** reblock's consolidated screens decide on blocks run so far, so screening.Screening.decide re-implements `replay`'s loop (the floor step, the per-unit veto, the step bookkeeping, the final call). Only reblock's `live` guard ties the two. A driver that advances as units arrive, with `replay` built on it, would make a live screen and its replay one loop, as `StopRule.met_reading` already makes a rule one.
2. **`ValueOfMoreUnits` is not Recordable.** Its `functionals` is an array and its cost model may hold data, so reblock's run identity is assembled by hand (Screening's record carries floor, step and the Monte Carlo sizes beside the rule).
3. **A cost model with per-unit covariates cannot record.** reblock's RatioToBase holds the base's GPU-seconds on every block. The base's runs exist, so `Design.base_costs` is 0 and cannot carry them. Covariates on the `Design`, shown through `Seen`, would leave every `CostModel` pure configuration.
4. **`record()` accepts what `revive` cannot rebuild.** A NamedTuple field records as a list, and `revive` raises ("annotated <class MC>, which a list cannot become"; reproduced at 452052c). epireplay's MonteCarlo was a NamedTuple. Refuse at record time, or rebuild NamedTuples.
5. **The stop is still paired with its target by hand.** `EveryArm(ValueOfInformation(price=target.price, ...))` builds with any price, so reblock's `unpriced` fault (674dba4's PRICE_RATE / FUTURE beside a target with uses) is constructible and caught only by reblock's guard. A stop derived from the target would make it unrepresentable.
6. **`Decision`, `Step` and `Judgement` have no record form.** reblock's screen_decisions.json needs hand-written `decision_record` and `decision_of` around the Recordable rules. A record and `revive` for decisions would close the ledger.
7. **Configuration dataclasses take positional arguments.** `AdoptKeepAskByStake(rates, weights, uses)`, `AdoptKeepAsk(low, high)` and `ValueOfInformation(price, margin)` do; `kw_only`, as `ValueOfMoreUnits` already is, would turn a swapped bracket or a bare number into a TypeError.
8. **One flat namespace for every record.** reblock's MonteCarlo, RealizedLoss, ZeroOne, OwnStake and Screening register beside epistemetry's own names, so a later class of the same name in epistemetry, or in another consumer imported in the same process, breaks the import.
9. **Python floor** (minor). `requires-python >=3.12` made reblock's research branch raise its floor from 3.11. epistemetry uses no 3.12-only syntax at 452052c (checked); lower it, or say why 3.12.
10. Only if Task 4 found one: the rounding tie (a stop that moved under the rescaling by `uses`), with its design, order and margin.

- [x] **Step 2: File the approved ones** (#1-#9)

Run, per approved item: `gh issue create --repo GeoffChurch/epistemetry --title "<title>" --body "<body: the case, reblock's commit that shows it, what would resolve it>"`
Expected: an issue URL each, recorded in BACKLOG (Task 9 Step 2, or a commit `research: roadless -- BACKLOG: the epistemetry issues filed (#a, #b, ...)`).

---

## Execution notes for the controller

- Start only once the consolidated pass's agent has returned and `git status` shows no edits of theirs to consolidated.py, NOTES.md, BACKLOG.md or consolidated_rows/.
- Task 4's run takes about 3 h in the background. Task 5's code can be written meanwhile; its guard does not depend on Task 4's output. Task 6 needs Task 4's CSV, and Task 8 needs every task before it.
- Stop points that need the owner: a difference in Task 4 that is not a rounding tie; a phase-5 call or stop outside Task 7 Step 3's expectations; Ruling 3 (the second preset) and Ruling 6 (the replays' data) if the owner overrules them; Task 10 entirely.
