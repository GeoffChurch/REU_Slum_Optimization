# The add/remove search as one Strategy: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One engine (Step + SearchState + Seq/Until combinators + Records) that the greedy, exchange, grow-then-prune and SIMP's polish run on as presets, their old loops deleted, plus backward floating search as the first new schedule.

**Architecture:** search.py holds the engine and the add/remove steps; a Step is one member's whole round, moved over nearly unchanged; a Record says which states the run must score and writes its rows; presets parse once at the edge into config objects bound per block.
Agreement with the current code is checked on three small blocks on the CPU, and any deviation is flagged and explained before the old path goes.

**Tech Stack:** Python 3.12, numpy, pandas, the research code's own solvers (pyamg on the CPU, cupy on the GPU); no test framework in research/roadless, so verification is the agreement harness of Task 1.

**Spec:** docs/superpowers/specs/2026-10-08-add-remove-search-design.md

## Global Constraints

- Branch `research/roadless`; commit with `git -c core.hooksPath=.githooks commit` (never `--no-verify`); push after each task (`git push origin research/roadless`).
- Commit messages end with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01RoNpoEMZ1kz2hogCsAsmXJ`.
- Lint: `uv run ruff check .` from the repo root must pass (the hook runs it); keep research lines at most 100 characters, matching the files.
- Numeric process pools single-threaded: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1` on every CPU run.
- One GPU process at a time on this machine; check `nvidia-smi --query-compute-apps=pid --format=csv,noheader` is empty before a GPU run.
- No `sleep N; cmd` waiters and no `pkill -f`; long runs go in the background.
- No legacy paths: once a member's preset agrees, its old loop is deleted and its callers moved; no shims.
- Result-changing parameters are required and keyword-only below the edge (no defaults on reach, width, pool, eps, rtol, tries, cap).
- Closed sets as names: NamedTuples and dataclasses, never string keys or positional tuples; no `getattr`/`hasattr` type tests.
- Do not modify tracked files while a `cluster.py submit` is running.

## Rulings on the spec (made writing this plan)

1. **No `wants_score` on Step.** The engine scores a new state exactly when the Record wants it and the step did not; that is the same solve the picker's `score` flag made (same clearing, `_exact` is a fresh solve), so `Picker.pick` loses its `score` parameter. Cost if wrong: the agreement check flags a greedy row.
2. **GreedyAdd, Reached and GreedyRows live in search.py, not clear.py.** search.py imports clear; clear.py importing search at module level would be a cycle, so `clear.greedy_block` imports search inside the function (commented). Cost if wrong: a later move of greedy_block.
3. **Exchange re-scores the state in its world inside its first round**, not the engine: the step knows its world, the engine does not need to.
4. **ConditionalAdd does not use the Chooser**: it has one candidate per round.
5. **`With(record, part)`** runs a part under its own Record: grow-then-prune's grow phase is Silent, its prune PruneRows.
6. **cut_to_budget stays in relax.py**: callers cut `s.order` after the greedy phase (search.py must not import relax).

## Review Focus

1. **A SearchState shared across phases with a stale score.** After an unscored greedy phase `s.score` is NaN; an Exchange must re-score before comparing (Task 4 tests it through polish_greedy's agreement row).
2. **c.removed left set after a polish or a seed.** relax.one assumes `c.removed` is all False outside the greedy; Incumbent.polish and the seed must reset it (Task 4 checks `c.removed.any()` is False after `one`).
3. **Memory held across a round.** GreedyAdd must not keep the Tension past the picker's `del t, H` (Task 7's memprobe, old against new).
4. **A tie broken differently** (argsort kind, `<` against `<=`) silently changing picks: the harness must report the first diverging step (Task 1's fault injection proves it does).
5. **Floating search looping**: a conditional add that reproduces an archived state must be refused (Task 8 asserts every kept add strictly improves its level).

---

### Task 1: The agreement harness and the baseline

**Files:**
- Create: `.superpowers/sdd/2026-10-08-add-remove-search/agree.py` (git-ignored workspace; not committed)
- Create: `.superpowers/sdd/2026-10-08-add-remove-search/bank.pkl`

**Interfaces:**
- Produces: `agree.py run <label> <commit>`, `agree.py compare <label_a> <label_b>`, `agree.py coverage <label>`; trees at `.superpowers/sdd/2026-10-08-add-remove-search/trees/<label>`.

The check blocks: `ZAF.9.3.1_1_19421` (112 buildings), `ZAF.9.3.1_1_19510` (88), `ZAF.9.3.1_1_38138` (63). The presets and their commands (run from a tree's root, `PYTHONPATH=.`, the main checkout's interpreter `~/src/reblock/.venv/bin/python`, `REBLOCK_BLOCK_BANK=<bank.pkl>`, the single-thread variables):

| preset | command | rows |
|---|---|---|
| greedy M4 | `research/roadless/clear.py some 1 M4 0.5 0.15 area 2 uni cpu <ids>` | `research/roadless/clear_rows_M4_h0.5_area_p2/<id>.parquet` |
| greedy B0.01g3 | same with `B0.01g3` | `clear_rows_B0.01g3_h0.5_area_p2/` |
| greedy S0.01cat | same with `S0.01cat` | `clear_rows_S0.01cat_h0.5_area_p2/` |
| greedy S0.01catw4 | same with `S0.01catw4` | `clear_rows_S0.01catw4_h0.5_area_p2/` |
| polish_greedy | `research/roadless/polish_greedy.py <id> 2 cpu S0.01cat 64 8 uni 0.05 0.5 x2` (one per block) | `polish_rows/uni/S0.01cat.P64w8x2/D0.05/<id>_p2.parquet` |
| grow-then-prune | `research/roadless/search.py <ids> GP3xS0.01catr0.005m8 2 cpu uni 0.15` | `clear_rows_GP3xS0.01catr0.005m8_h0.5_area_p2/` |
| SIMP hard | `research/roadless/relax.py one <ids> 2 cpu fw0.q3.i10.t0.001.e0.0001.k1s1e-06.p256w8x2.GS0.01cate1 uni 1 0.05` | `relax_rows/uni/<plan>/D0.05/<id>_p2.parquet` |

- [ ] **Step 1: Write the bank.**

```bash
cd ~/src/reblock
WS=.superpowers/sdd/2026-10-08-add-remove-search; mkdir -p $WS
PYTHONPATH=. uv run python -c "
import sys; sys.path.insert(0, 'research/roadless')
import common
common.write_bank(['ZAF.9.3.1_1_19421', 'ZAF.9.3.1_1_19510', 'ZAF.9.3.1_1_38138'], '$WS/bank.pkl')"
```

- [ ] **Step 2: Write agree.py.**

```python
"""The add/remove refactor's agreement check: the presets run in a git worktree per label (fresh
rows, the CPU), compared row by row. run <label> <commit>; compare <a> <b>; coverage <label>."""
from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parent
REPO = WS.parents[2]
PY = REPO / ".venv" / "bin" / "python"
IDS = ["ZAF.9.3.1_1_19421", "ZAF.9.3.1_1_19510", "ZAF.9.3.1_1_38138"]
SIMP = "fw0.q3.i10.t0.001.e0.0001.k1s1e-06.p256w8x2.GS0.01cate1"
R = "research/roadless"
GREEDY = ["M4", "B0.01g3", "S0.01cat", "S0.01catw4"]
JOBS = 6                                        # concurrent CPU processes (a shared machine)
SKIP = {"t", "t_greedy"}                        # wall times: never compared


def commands() -> list[tuple[str, list[str]]]:
    ids = ",".join(IDS)
    out = [(f"greedy {p}", [f"{R}/clear.py", "some", "1", p, "0.5", "0.15", "area", "2", "uni",
                            "cpu", ids]) for p in GREEDY]
    out += [(f"polish {b}", [f"{R}/polish_greedy.py", b, "2", "cpu", "S0.01cat", "64", "8",
                             "uni", "0.05", "0.5", "x2"]) for b in IDS]
    out.append(("GP", [f"{R}/search.py", ids, "GP3xS0.01catr0.005m8", "2", "cpu", "uni", "0.15"]))
    out += [(f"SIMP {b}", [f"{R}/relax.py", "one", b, "2", "cpu", SIMP, "uni", "1", "0.05"])
            for b in IDS]
    return out


def rows(tree: Path) -> dict[tuple[str, str], pd.DataFrame]:
    """(preset, block) -> its rows, from a tree's row directories."""
    rr = tree / R
    found = {}
    for p in GREEDY + ["GP3xS0.01catr0.005m8"]:
        for b in IDS:
            f = rr / f"clear_rows_{p}_h0.5_area_p2" / f"{b}.parquet"
            if f.exists():
                found[(p, b)] = pd.read_parquet(f)
    for b in IDS:
        for key, f in ((("polish", b), rr / "polish_rows/uni/S0.01cat.P64w8x2/D0.05" / f"{b}_p2.parquet"),
                       (("SIMP", b), rr / f"relax_rows/uni/{SIMP}/D0.05" / f"{b}_p2.parquet")):
            if f.exists():
                found[key] = pd.read_parquet(f)
    return found


def run(label: str, commit: str) -> None:
    tree = WS / "trees" / label
    if not tree.exists():
        subprocess.run(["git", "-C", str(REPO), "worktree", "add", "--detach", str(tree), commit],
                       check=True)
    env = dict(os.environ, PYTHONPATH=".", REBLOCK_BLOCK_BANK=str(WS / "bank.pkl"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMBA_NUM_THREADS="1")
    (tree / "logs").mkdir(exist_ok=True)

    def one(item):
        name, cmd = item
        log = tree / "logs" / (name.replace(" ", "_") + ".log")
        with open(log, "w") as fh:
            rc = subprocess.run([str(PY), "-u", *cmd], cwd=tree, env=env, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        print(f"{label}: {name} rc={rc}", flush=True)
        return rc

    with ThreadPoolExecutor(JOBS) as ex:
        bad = [rc for rc in ex.map(one, commands()) if rc]
    if bad:
        raise SystemExit(f"{len(bad)} commands failed: see {tree}/logs")


def _same(a, b) -> bool:
    if isinstance(a, (list, np.ndarray)) or isinstance(b, (list, np.ndarray)):
        return list(np.asarray(a).ravel()) == list(np.asarray(b).ravel())
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def compare(a: str, b: str) -> int:
    """Every (preset, block): identical, or the first row and columns that differ. Returns the
    number that differ."""
    ra, rb = rows(WS / "trees" / a), rows(WS / "trees" / b)
    differ = 0
    for key in sorted(set(ra) | set(rb)):
        if key not in ra or key not in rb:
            print(f"{key}: only in {a if key in ra else b}")
            differ += 1
            continue
        x, y = ra[key], rb[key]
        cols = [c for c in x.columns if c not in SKIP]
        if list(x.columns) != list(y.columns) or len(x) != len(y):
            print(f"{key}: shape {x.shape} {list(x.columns)} vs {y.shape} {list(y.columns)}")
            differ += 1
        first = None
        for i in range(min(len(x), len(y))):
            bad = [c for c in cols if c in y.columns and not _same(x[c].iloc[i], y[c].iloc[i])]
            if bad:
                first = (i, bad)
                break
        if first is None and len(x) == len(y):
            print(f"{key}: identical ({len(x)} rows)")
            continue
        if first is not None:
            i, bad = first
            differ += 1
            print(f"{key}: FIRST DIFFERENCE at row {i}: "
                  + "; ".join(f"{c} {x[c].iloc[i]!r} vs {y[c].iloc[i]!r}" for c in bad))
    return differ


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "compare":
        raise SystemExit(1 if compare(sys.argv[2], sys.argv[3]) else 0)
    else:
        raise SystemExit("run <label> <commit> | compare <a> <b>")
```

- [ ] **Step 3: Run the current code twice (the CPU noise floor).**

```bash
cd ~/src/reblock; WS=.superpowers/sdd/2026-10-08-add-remove-search
BASE=$(git rev-parse HEAD); echo "BASE $BASE" >> $WS/progress.md
uv run python $WS/agree.py run base1 $BASE > $WS/base1.log 2>&1   # in the background: ~1 h
uv run python $WS/agree.py run base2 $BASE > $WS/base2.log 2>&1
uv run python $WS/agree.py compare base1 base2
```
Expected: every (preset, block) `identical`.
If one differs, the CPU path is not deterministic there: record which, and from here on compare that preset to the base's spread instead of exactly (note it in the ledger).

- [ ] **Step 4: Prove the harness catches a difference.**
  - In `trees/base2/research/roadless/relax.py`'s `exchange`, change the tie rule `if Jn < J and` to `if Jn <= J and`.
  - Delete `trees/base2/research/roadless/polish_rows` and `trees/base2/research/roadless/relax_rows`, then `agree.py run base2 <BASE>` again (the greedy and GP scripts skip rows that exist; polish and SIMP are recomputed).
  - `agree.py compare base1 base2`.
    Expected: a FIRST DIFFERENCE on a polish or SIMP row.
    If none, the tie never occurs on these blocks: say so in the ledger, and inject a fault that must show instead (in `exchange`, `top = np.argpartition(e, k - 1)[:k]` to `[:max(k - 1, 1)]`), and compare again.
  - Revert (`git -C <tree> checkout -- research/roadless/relax.py`), delete the polish and SIMP rows again, and rerun so base2 is clean.

- [ ] **Step 5: Ledger.** Append to `$WS/progress.md`: the base commit, the noise floor result, the fault-injection result.

(No commit: the workspace is git-ignored.)

---

### Task 2: The engine

**Files:**
- Modify: `research/roadless/search.py` (add the engine above GrowPrune; GrowPrune and grow_prune untouched in this task)

**Interfaces:**
- Produces (in search.py): `Score(J, P)`, `UNSCORED`, `EXACT`, `Outcome(cleared, restored, score, world)`, `SearchState(c, power, score, world, order, scorings, movable)` with `SearchState.start(c, power, score, world)` and property `D`; Protocols `Step.step(s) -> Outcome | None`, `Record.wants(s) -> bool` / `on_step(s, o)`, `Stop.done(s) -> bool`, `Part.__call__(s, rec) -> int`, `Valuer.value(x) -> float`; `apply(s, o, rec)`; parts `Do(step)`, `Until(part, stop)`, `Seq(parts)`, `With(rec, part)`; stops `Reached(d)`, `Emptied()`, `Spent(n)`; record `Silent()`.

- [ ] **Step 1: Add the engine to search.py**, after the imports (add `from dataclasses import dataclass` and `from typing import Protocol` to the imports; `import clear` as a module, since `apply` must look `clear._exact` up at call time for memprobe_greedy's patch):

```python
class Score(NamedTuple):
    J: float                    # J_power
    P: float                    # P (J_1); NaN in an eps world


UNSCORED = Score(np.nan, np.nan)


class _Exact:
    """The exact world: real geometry, the metric conductance, RTOL_SCORE (clear._exact)."""


EXACT = _Exact()


class Valuer(Protocol):
    """An eps world bound to a block (relax.Relaxation): J of a clearing x there."""

    def value(self, x: np.ndarray) -> float: ...


class Outcome(NamedTuple):
    cleared: list[int]          # buildings the round clears, in pick order
    restored: list[int]         # buildings it puts back
    score: Score | None         # the new state's score in `world`, if the round has one
    world: object | None        # EXACT or a Valuer


@dataclass
class SearchState:
    """One block's search: c.removed is the clearing (only `apply` changes it); `score` is the
    current state's in `world` (UNSCORED, None: not scored); `order` every building cleared, in
    pick order (relax.cut_to_budget's); `scorings` the candidate scorings spent (the caps);
    `movable` the buildings a move may touch."""
    c: Clearing
    power: float
    score: Score
    world: object | None
    order: list[int]
    scorings: int
    movable: np.ndarray

    @classmethod
    def start(cls, c: Clearing, power: float, score: Score, world: object | None) -> SearchState:
        """From c's current clearing, nothing spent, every building movable."""
        return cls(c, power, score, world, [], 0, np.ones(c.n, dtype=bool))

    @property
    def D(self) -> float:
        return float(self.c.cost[self.c.removed].sum())


class Step(Protocol):
    """One round of one member: the move it makes from `s`, or None (none: the round's member
    found nothing to do or nothing better)."""

    def step(self, s: SearchState) -> Outcome | None: ...


class Record(Protocol):
    """What a run owes: which states must be scored exactly, and what it writes of each step."""

    def wants(self, s: SearchState) -> bool: ...

    def on_step(self, s: SearchState, o: Outcome) -> None: ...


class Stop(Protocol):
    def done(self, s: SearchState) -> bool: ...


class Part(Protocol):
    """A schedule: runs on `s` under `rec`, returns the moves it made."""

    def __call__(self, s: SearchState, rec: Record) -> int: ...


def apply(s: SearchState, o: Outcome, rec: Record) -> None:
    """Make the move; score the new state exactly if `rec` wants it and the round did not (the
    same solve a picker's own scoring made); tell `rec`."""
    c = s.c
    if o.cleared:
        c.removed[o.cleared] = True
        s.order.extend(o.cleared)
    if o.restored:
        c.removed[o.restored] = False
    s.score, s.world = (UNSCORED, None) if o.score is None else (o.score, o.world)
    if rec.wants(s) and s.world is not EXACT:
        s.score, s.world = Score(*clear._exact(c, [], s.power)), EXACT   # patchable (memprobe)
    rec.on_step(s, o)


@dataclass(frozen=True)
class Do:
    """One round of a step."""
    step: Step

    def __call__(self, s: SearchState, rec: Record) -> int:
        o = self.step.step(s)
        if o is None:
            return 0
        apply(s, o, rec)
        return 1


@dataclass(frozen=True)
class Until:
    """`part` again and again until `stop`, or until it makes no move."""
    part: Part
    stop: Stop

    def __call__(self, s: SearchState, rec: Record) -> int:
        n = 0
        while not self.stop.done(s):
            k = self.part(s, rec)
            if k == 0:
                break
            n += k
        return n


@dataclass(frozen=True)
class Seq:
    parts: tuple[Part, ...]

    def __call__(self, s: SearchState, rec: Record) -> int:
        n = 0
        for p in self.parts:
            n += p(s, rec)
        return n


@dataclass(frozen=True)
class With:
    """`part` under its own record (a phase whose states the run does not owe)."""
    rec: Record
    part: Part

    def __call__(self, s: SearchState, rec: Record) -> int:
        return self.part(s, self.rec)


@dataclass(frozen=True)
class Reached:
    """The greedy's stop: D at or past d, or nothing left to clear."""
    d: float

    def done(self, s: SearchState) -> bool:
        return s.D >= self.d - 1e-12 or bool(s.c.removed.all())


@dataclass(frozen=True)
class Emptied:
    def done(self, s: SearchState) -> bool:
        return not s.c.removed.any()


@dataclass(frozen=True)
class Spent:
    n: int

    def done(self, s: SearchState) -> bool:
        return s.scorings >= self.n


@dataclass(frozen=True)
class Silent:
    """Owes nothing: no state scored for it, nothing written."""

    def wants(self, s: SearchState) -> bool:
        return False

    def on_step(self, s: SearchState, o: Outcome) -> None:
        pass
```

Note `Reached.done` must equal `not (c.cost[c.removed].sum() < d - 1e-12 and not c.removed.all())`, clear.grow's loop condition.

- [ ] **Step 2: Import check and lint.**

```bash
cd ~/src/reblock && PYTHONPATH=. uv run python -c "import sys; sys.path.insert(0,'research/roadless'); import search; print(search.Until, search.EXACT)" && uv run ruff check .
```
Expected: prints the class and the sentinel; `All checks passed!`.

- [ ] **Step 3: Commit.**

```bash
git add research/roadless/search.py
git -c core.hooksPath=.githooks commit -m "research: roadless -- the add/remove engine: SearchState, Step, Record, Seq/Until/With, the stops (search.py)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RoNpoEMZ1kz2hogCsAsmXJ"
git push origin research/roadless
```

---

### Task 3: The greedy on the engine

**Files:**
- Modify: `research/roadless/search.py` (add GreedyAdd, GreedyRows)
- Modify: `research/roadless/clear.py` (Picker.pick loses `score`; Spread's reach and width required; greedy_block on the engine; delete `grow`)
- Modify: `research/roadless/gramprobe.py`, `memprobe_greedy.py`, `profile_round.py` (their `grow` / `pick(..., True)` calls)

**Interfaces:**
- Consumes: Task 2's engine.
- Produces: `search.GreedyAdd(picker)`, `search.GreedyRows(block_id, c, power, J0, P0, t0)` with `.rows: list[dict]`; `Picker.pick(c, t, J, power) -> Picked` (Picked.J NaN when the picker did not score); `Spread(delta, source, *, reach, width)`; `clear.SPREAD_REACH = 3.0` (the edge's constant, used by picker_of).

- [ ] **Step 1: search.py: GreedyAdd and GreedyRows.**

```python
@dataclass(frozen=True)
class GreedyAdd:
    """The greedy's round: the tension at the current clearing, then `picker` clears (its own
    score, if choosing needed one: Screened, Spread over several batches)."""
    picker: clear.Picker

    def step(self, s: SearchState) -> Outcome | None:
        pk = self.picker.pick(s.c, s.c.tension(s.power), s.score.J, s.power)
        return Outcome(pk.cleared, [], None if np.isnan(pk.J) else Score(pk.J, pk.P1), EXACT)


class GreedyRows:
    """The greedy's rows (clear.greedy_block's format): every state scored exactly; per step D,
    perm (J_power), perm1 (P) and the step's buildings in pick order, after a step-0 row."""

    def __init__(self, block_id: str, c: Clearing, power: float, J0: float, P0: float,
                 t0: float):
        self.block_id, self.n, self.power, self.J0, self.P0, self.t0 = (
            block_id, c.n, power, J0, P0, t0)
        self.rows = [dict(block=block_id, n=c.n, step=0, D=0.0, perm=0.0, perm1=0.0, cleared=[],
                          P0=P0, t=0.0)]

    def wants(self, s: SearchState) -> bool:
        return True

    def on_step(self, s: SearchState, o: Outcome) -> None:
        self.rows.append(dict(block=self.block_id, n=self.n, step=len(self.rows), D=s.D,
                              perm=1 - (s.score.J / self.J0) ** (1 / self.power),
                              perm1=1 - s.score.P / self.P0, cleared=o.cleared, P0=self.P0,
                              t=time.time() - self.t0))
        if self.n > 1000:
            r = self.rows[-1]
            print(f"  {self.block_id} step {r['step']} D {r['D']:.3f} perm' {r['perm']:.3f}"
                  f" {r['t']:.0f}s", flush=True)
```

- [ ] **Step 2: clear.py: the pickers.**
  - `Picker` docstring and signature: `def pick(self, c: Clearing, t: Tension, J: float, power: float) -> Picked: ...`, the docstring saying a picker that needs no exact solve to choose returns J and P1 as NaN (the engine scores the state if its record wants it).
  - `Screened.pick(self, c, t, J, power)`: body unchanged.
  - `Batched.pick(self, c, t, J, power)`: last line `return Picked(taken, np.nan, np.nan)`.
  - `Spread`: fields `delta: float`, `source: GramSource`, then `_: KW_ONLY`, `reach: float`, `width: int` (import `KW_ONLY` from dataclasses); `pick(self, c, t, J, power)` with `if len(batches) == 1:` (no `and not score`).
  - Add after `picker_of`'s docstring constants: `SPREAD_REACH = 3.0   # the shortlist's reach, in population steps (the measured default)` at module level near RTOL constants, and in picker_of: `Spread(float(m.group(1)), Catchment(sweep), reach=SPREAD_REACH, width=1 if m.group(2) is None else int(m.group(2)))`.
  - Delete `grow`.

- [ ] **Step 3: clear.py: greedy_block on the engine.** Replace its body from `t0 = time.time()` through the loop with:

```python
    import search   # search imports clear: here, not at the top
    t0 = time.time()
    rec = search.GreedyRows(b.block_id, c, power, J0, P0, t0)
    s = search.SearchState.start(c, power, search.Score(J0, P0), search.EXACT)
    search.Until(search.Do(search.GreedyAdd(picker)), search.Reached(d_max))(s, rec)
    rows = rec.rows
```
and delete the old `rows = [...]` initialisation above it (GreedyRows makes the step-0 row). The tmp/replace write and the final print are unchanged.

- [ ] **Step 4: the probes.**
  - gramprobe.py, the loop `for _pk, _D in clear.grow(c, picker, 2.0, d_max, J0, True): pass` becomes
    ```python
    s = search.SearchState.start(c, 2.0, search.Score(J0, c.sc.P0), search.EXACT)
    search.Until(search.Do(search.GreedyAdd(picker)), search.Reached(d_max))(
        s, search.GreedyRows(bid, c, 2.0, J0, c.sc.P0, time.time()))
    ```
    (add `import search`; `bid` is main's block id argument; check the name in the file).
  - memprobe_greedy.py, the final loop becomes a Record that prints what the loop printed:
    ```python
    class _Print:
        def wants(self, s):
            return True

        def on_step(self, s, o):
            print(f"  step {step[0]} D {s.D:.3f} perm' {1 - (s.score.J / J0) ** 0.5:.3f}",
                  flush=True)

    s = search.SearchState.start(c, 2.0, search.Score(J0, c.sc.P0), search.EXACT)
    search.Until(search.Do(search.GreedyAdd(picker)), search.Reached(d_max))(s, _Print())
    ```
    Its `clear._exact = exact_` patch keeps working: `search.apply` and the pickers look `_exact` up on the clear module at call time.
  - profile_round.py: `out = pk.pick(c, t, J, 2.0)`.

- [ ] **Step 5: Agreement.** Commit locally first (Step 6's commit, not yet pushed), then `cd ~/src/reblock; WS=.superpowers/sdd/2026-10-08-add-remove-search; uv run python $WS/agree.py run t3 $(git rev-parse HEAD)` and `compare base1 t3`.
Expected: the four `greedy` presets identical on the three blocks.
polish, GP and SIMP also run on this tree (they still use their old loops but call pickers): identical too.
Any difference: report the first row and columns, investigate, and record the explanation in the ledger before going on.

- [ ] **Step 6: Lint, commit (before Step 5's run), push (after it agrees).**

```bash
uv run ruff check . && git add research/roadless/{search,clear,gramprobe,memprobe_greedy,profile_round}.py
git -c core.hooksPath=.githooks commit -m "research: roadless -- the greedy on the add/remove engine: GreedyAdd and GreedyRows; the engine scores a state its record wants (Picker.pick loses score); Spread's reach and width required; clear.grow deleted

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RoNpoEMZ1kz2hogCsAsmXJ"
git push origin research/roadless
```

---

### Task 4: Exchange, polish_greedy and SIMP on the engine

**Files:**
- Modify: `research/roadless/search.py` (add `choose`)
- Modify: `research/roadless/relax.py` (Exchange; `_pairs` takes `pool`; Incumbent.polish and relax.one's seed/track on the engine; delete `exchange`)
- Modify: `research/roadless/polish_greedy.py`

**Interfaces:**
- Consumes: Tasks 2 and 3.
- Produces: `search.choose(tiers, score, r, s, width, tries) -> tuple[float, np.ndarray] | None`; `relax.Exchange(world, *, budget, width, pairs, pool, tries)`; `relax._pairs(g, ins, outs, cost, left, pool)`; `relax.PAIR_POOL` stays as the edge constant.

- [ ] **Step 1: search.py: choose** (relax.exchange's scoring loop, unchanged in what it scores and in what order):

```python
def choose(tiers, score, r: np.ndarray, s: SearchState, width: int,
           tries: int) -> tuple[float, np.ndarray] | None:
    """Score the best `width` moves of each tier (fewer as `tries` runs out), in order of their
    linearized estimate, and return (J, clearing) of the best that beats s.score.J, trying a
    tier only if no earlier one did; None if none does. A tier is a callable giving (moves,
    estimates): moves a (4, M) array, rows 0-1 buildings added and 2-3 closed (-1: none),
    estimates inf where infeasible. Each scoring counts in s.scorings."""
    best = None
    for tier in tiers:
        mv, e = tier()
        k = min(width, tries - s.scorings, int(np.isfinite(e).sum()))
        if k == 0:
            continue
        top = np.argpartition(e, k - 1)[:k]
        for m in top[np.argsort(e[top])]:
            rn = r.copy()
            for x in mv[:2, m]:
                if x >= 0:
                    rn[x] = 1.0
            for x in mv[2:, m]:
                if x >= 0:
                    rn[x] = 0.0
            s.scorings += 1
            Jn = score(rn)
            if Jn < s.score.J and (best is None or Jn < best[0]):
                best = (Jn, rn)
        if best is not None:
            break
    return best
```

- [ ] **Step 2: relax.py: Exchange** (replaces `exchange`; `_singles` unchanged; `_pairs` gains `pool` in place of the global `PAIR_POOL`):

```python
@dataclass(frozen=True)
class Exchange:
    """One round of exchange refinement within s.movable: the single moves (add one the budget
    left allows, or swap one in for a cleared one) ranked by their linearized change from the
    world's gradient at the 0/1 clearing (adding a: g_a; closing b: -g_b); the best `width`
    scored in `world` (search.choose), the best kept if it improves; with `pairs`, a round whose
    singles improve nothing then tries the two-in-one-out and one-in-two-out moves among the
    `pool` most promising (floating search's escalation). `tries` caps the scorings (with the
    schedule's Spent(tries)). The first round scores the state in `world` if it is not."""
    world: Relaxation           # bound to the block: value and value_sgrad share its warm start
    _: KW_ONLY
    budget: float
    width: int
    pairs: bool
    pool: int
    tries: int

    def step(self, s: search.SearchState) -> search.Outcome | None:
        c = s.c
        r = c.removed.astype(float)
        if s.world is not self.world:
            s.score, s.world = search.Score(self.world.value(r), np.nan), self.world
        g = self.world.value_sgrad(r)[1]
        left = self.budget + 1e-12 - float(c.cost @ r)
        movable = np.flatnonzero(s.movable)
        ins, outs = movable[r[movable] == 0], movable[r[movable] > 0]
        tiers = [lambda: _singles(g, ins, outs, c.cost, left)]
        if self.pairs:
            tiers.append(lambda: _pairs(g, ins, outs, c.cost, left, self.pool))
        best = search.choose(tiers, self.world.value, r, s, self.width, self.tries)
        if best is None:
            return None
        Jn, rn = best
        return search.Outcome(np.flatnonzero((rn > 0) & (r == 0)).tolist(),
                              np.flatnonzero((rn == 0) & (r > 0)).tolist(),
                              search.Score(Jn, np.nan), self.world)
```
(`import search` in relax.py's imports; `KW_ONLY` from dataclasses.) Delete `exchange`.

- [ ] **Step 3: relax.py: Incumbent.polish on the engine.** Replace the `J0 = self.J` ... `exchange(...)` lines with:

```python
        J0 = self.J
        c = self.rel.c
        c.removed[:] = self.r > 0
        s = search.SearchState(c, self.rel.power, search.Score(self.J, np.nan), self.scorer, [], 0,
                               np.isin(np.arange(c.n), movable))
        moves = search.Until(search.Do(Exchange(self.scorer, budget=self.budget, width=width,
                                                pairs=pairs, pool=PAIR_POOL, tries=tries)),
                             search.Spent(tries))(s, search.Silent())
        self.J, self.r, used = s.score.J, c.removed.astype(float), s.scorings
        c.removed[:] = False                    # relax.one keeps the clearing in x, not here
```
the `log(...)` line unchanged.

- [ ] **Step 4: relax.py: relax.one's seed and track.** Replace the two lines `order = [j for pk, _ in clear.grow(...)]` and `c.removed[:] = False` with:

```python
        s = search.SearchState.start(c, power, search.Score(rel.J0, np.nan), search.EXACT)
        search.Until(search.Do(search.GreedyAdd(picker)), search.Reached(budget))(
            s, search.Silent())
        c.removed[:] = False
        order = s.order
```

- [ ] **Step 5: polish_greedy.py on the engine.** Replace `t0 = time.time()` through the `exchange(...)` call with:

```python
    t0 = time.time()
    s = search.SearchState.start(c, power, search.Score(c.sc.J(c.sc.u0, power), c.sc.P0),
                                 search.EXACT)
    search.Until(search.Do(search.GreedyAdd(picker)), search.Reached(budget))(s, search.Silent())
    t_greedy = time.time() - t0
    r = relax.cut_to_budget(s.order, c.cost, budget)
    c.removed[:] = r > 0
    rel = relax.Relaxation(c, power, rtol=RTOL)
    scorer = relax.Relaxation(c, power, q=1.0, rtol=RTOL, eps=SCORE_EPS, params=c.p)
    greedy_perm = rel.perm(rel.exact(r))
    moves = search.Until(search.Do(relax.Exchange(scorer, budget=budget, width=width, pairs=pairs,
                                                  pool=relax.PAIR_POOL, tries=tries)),
                         search.Spent(tries))(s, search.Silent())
    rp, J, used = c.removed.astype(float), s.score.J, s.scorings
```
(`import search`; the rest of main unchanged: it uses `rp`, `moves`, `used`, `r`.)
Check the first Exchange round scores `scorer.value(r)` before `scorer.value_sgrad(r)`, as the old call did (its argument `scorer.value(r)` was evaluated before exchange ran).

- [ ] **Step 6: Agreement.** Commit locally (no push yet), `agree.py run t4 <commit>`, `compare base1 t4`: everything identical (polish rows: `perm, greedy_perm, D, greedy_D, moves, scorings, cleared`; SIMP rows: every column but `t`).
Any difference: first row and columns, investigated, explained in the ledger.

- [ ] **Step 7: The clearing is left empty (review focus 2).** On one check block, CPU:

```bash
cd ~/src/reblock && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
REBLOCK_BLOCK_BANK=.superpowers/sdd/2026-10-08-add-remove-search/bank.pkl PYTHONPATH=. uv run python -c "
import sys; sys.path.insert(0, 'research/roadless')
import relax
seen = []
make = relax._clearing
relax._clearing = lambda *a: seen.append(make(*a)) or seen[-1]
relax.one('ZAF.9.3.1_1_38138', 2.0, 'cpu',
          relax.plan_of('fw0.q3.i10.t0.001.e0.0001.k1s1e-06.p256w8x2.GS0.01cate1'), 'uni', 0.05)
assert not seen[0].removed.any(), 'relax.one left buildings removed'
print('clearing empty after one')"
```
Expected: `clearing empty after one`.

- [ ] **Step 8: Lint, amend if needed, push** (message: "research: roadless -- exchange on the add/remove engine: relax.Exchange and search.choose; polish_greedy, Incumbent.polish and SIMP's seed and track run on it; relax.exchange deleted").

---

### Task 5: Grow-then-prune on the engine

**Files:**
- Modify: `research/roadless/search.py`

**Interfaces:**
- Consumes: Tasks 2-3.
- Produces: `RestoreBatch(restore, shortlist, screen)`; `PruneRows(block_id, c, power, J0, P0, d_max, t0)` with `.frame() -> pd.DataFrame`; a `Rows` Protocol (a Record with `frame()`); `GrowPrune(grow, picker, restore, shortlist, *, screen_eps)` with `build(block_id, c, power, J0, P0, d_max, t0, picker, screen) -> tuple[Part, Rows]`; `prune_block(b, plan, picker, screen_of, power, along, device, d_max) -> pd.DataFrame` replacing `grow_prune`; `EPS_SCREEN` stays as the edge constant given to plan_of.

- [ ] **Step 1: RestoreBatch** (restore_batch unchanged, its `screen` annotated `Valuer`):

```python
@dataclass(frozen=True)
class RestoreBatch:
    """Grow-then-prune's round: restore_batch's buildings put back, unscored (the engine scores
    the state if the record owes it)."""
    restore: float
    shortlist: int
    screen: Valuer

    def step(self, s: SearchState) -> Outcome | None:
        batch = restore_batch(s.c, s.power, self.restore, self.shortlist, self.screen)
        return Outcome([], [int(j) for j in batch], None, None)
```

- [ ] **Step 2: PruneRows.**

```python
class PruneRows:
    """Grow-then-prune's rows: every state at or below d_max with something cleared, scored
    exactly; written in increasing D in the greedy's format, `cleared` in index order (the
    states are nested), `t` the time when written."""

    def __init__(self, block_id: str, c: Clearing, power: float, J0: float, P0: float,
                 d_max: float, t0: float):
        self.block_id, self.n, self.power, self.J0, self.P0, self.d_max, self.t0 = (
            block_id, c.n, power, J0, P0, d_max, t0)
        self.states: list[tuple[float, float, float, np.ndarray]] = []

    def wants(self, s: SearchState) -> bool:
        return s.D <= self.d_max + 1e-12 and bool(s.c.removed.any())

    def on_step(self, s: SearchState, o: Outcome) -> None:
        if self.wants(s):
            self.states.append((s.D, s.score.J, s.score.P, s.c.removed.copy()))

    def frame(self) -> pd.DataFrame:
        rows = [dict(block=self.block_id, n=self.n, step=0, D=0.0, perm=0.0, perm1=0.0,
                     cleared=[], P0=self.P0, t=0.0)]
        prev = np.zeros(self.n, dtype=bool)
        for D, J, P, r in reversed(self.states):
            rows.append(dict(block=self.block_id, n=self.n, step=len(rows), D=D,
                             perm=1 - (J / self.J0) ** (1 / self.power), perm1=1 - P / self.P0,
                             cleared=np.flatnonzero(r & ~prev).tolist(), P0=self.P0,
                             t=time.time() - self.t0))
            prev = r
        return pd.DataFrame(rows)
```

- [ ] **Step 3: GrowPrune, plan_of, prune_block, main.**
  - GrowPrune becomes a frozen dataclass: fields `grow: float`, `picker: str`, `restore: float`, `shortlist: int`, then `_: KW_ONLY`, `screen_eps: float`; `name` unchanged; plan_of passes `screen_eps=EPS_SCREEN`.
  - search.py stops importing relax (ruling 6): delete `from relax import Relaxation`; the screen is bound at the edge (`main`) and passed in as `screen_of`.

```python
class Rows(Record, Protocol):
    def frame(self) -> pd.DataFrame: ...


# in GrowPrune:
    def build(self, block_id: str, c: Clearing, power: float, J0: float, P0: float,
              d_max: float, t0: float, picker: clear.Picker,
              screen: Valuer) -> tuple[Part, Rows]:
        rec = PruneRows(block_id, c, power, J0, P0, d_max, t0)
        return Seq((With(Silent(), Until(Do(GreedyAdd(picker)), Reached(self.grow * d_max))),
                    Until(Do(RestoreBatch(self.restore, self.shortlist, screen)),
                          Emptied()))), rec


def prune_block(b, plan: GrowPrune, picker: clear.Picker,
                screen_of: Callable[[Clearing, float], Valuer], power: float, along: str,
                device: str, d_max: float) -> pd.DataFrame:
    p = lifted.Params(3.0, 8, along=lifted.along_of(along, lifted.scans_of(device)),
                      solver=lifted.solver_of(device))
    mesh = lifted.UniformMesh(0.5, offset=lifted.OFFSET)
    c = Clearing(b, mesh, p, population=common.POPULATIONS["area"])
    J0, P0 = c.sc.J(c.sc.u0, power), c.sc.P0
    t0 = time.time()
    part, rec = plan.build(b.block_id, c, power, J0, P0, d_max, t0, picker, screen_of(c, power))
    part(SearchState.start(c, power, Score(J0, P0), EXACT), rec)
    rows = rec.frame()
    print(f"{time.strftime('%H:%M:%S')} {b.block_id} n={c.n} {plan.name}: "
          f"{len(rows) - 1} states; perm' {rows.perm.iloc[-1]:.3f} at D {rows.D.iloc[-1]:.3f}  "
          f"{time.time() - t0:.0f}s", flush=True)
    return rows
```
  - `main` (the edge) resolves the picker once, `picker = picker_of(plan.picker, sweep_of(device))`, and binds the screen with a local import (commented: the edge binds the eps world; search.py's library code does not import relax):
    `import relax` then `screen_of = lambda c, power: relax.Relaxation(c, power, q=1.0, rtol=RTOL_SCORE, eps=plan.screen_eps)`, and calls `prune_block(b, plan, picker, screen_of, power, along, device, d_max)`.
    The old screen was built after the grow phase; building it before changes nothing it computes (its warm start is empty until its first value), which the agreement check confirms.
  - The old "grown to D" print goes (the grow phase is a part now): say so in the commit message.
  - Delete grow_prune.

- [ ] **Step 4: Agreement.** Commit locally; `agree.py run t5 <commit>`; `compare base1 t5`: GP identical on the three blocks (the grow phase is no longer scored: only `t` changes).
Any difference: first row, investigated, explained.

- [ ] **Step 5: Lint, commit, push** ("grow-then-prune on the add/remove engine: RestoreBatch, PruneRows, GrowPrune.parts; the grow phase no longer scores states it never records; grow_prune deleted").

---

### Task 6: Coverage of the agreement check

**Files:**
- Modify: `.superpowers/sdd/2026-10-08-add-remove-search/agree.py` (add `coverage`)

- [ ] **Step 1:** Add `coverage <label>`: runs the polish and SIMP commands for the three blocks in a fresh worktree of the label's commit under a runner that wraps, before `runpy.run_path(script, run_name="__main__")`:
  - `relax._pairs` (count calls: the pairs tier fired),
  - `clear._spread` (count calls per Spread.pick; with width 4, a round whose distinct batches are fewer than 4 is a dedup: wrap `Spread.pick` to compare `len(set(map(frozenset, ...)))`, or simply count rounds where two `_spread` calls in one pick returned the same set),
  - `relax.cut_to_budget` (count calls where the result's D is below the budget by more than the next building in the order would add: a partial step skipped one),
  - `search.restore_batch` with shortlist (count calls where the shortlist reordered the first k).
  Print the counts per block.
- [ ] **Step 2:** Run it on the current HEAD. For any branch at zero, find a block where it fires (the 59's small blocks; `ZAF.9.3.1_1_19537`, 247 buildings, for the pairs tier) and add that block to IDS, rerun `run base3 <BASE>` and `run t6 <HEAD>` for the new block only (temporarily IDS = [that block]) and `compare base3 t6`.
- [ ] **Step 3:** Ledger: the counts, any added block and its comparison.

---

### Task 7: GPU memory, old against new

**Files:** none changed (a measurement).

- [ ] **Step 1:** With the GPU idle, run memprobe_greedy at the base commit (in `trees/base1`) and at HEAD, one after the other, on `ZAF.9.3.1_1_20543` under the translucent search, recording the peak:

```bash
cd <tree> && (while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done) > mem.txt & MON=$!
CUDA_PATH=/usr PYTHONPATH=. ~/src/reblock/.venv/bin/python -u research/roadless/memprobe_greedy.py ZAF.9.3.1_1_20543 uni@ss100k0.5n2r30 0.02 default > memprobe.log 2>&1
kill $MON; sort -n mem.txt | tail -1
```
(run in the background; the two runs sequentially, never together).
Expected: the new peak within 2% of the old.
More: find what the engine holds (a Tension kept past the picker) and fix it before going on.
- [ ] **Step 2:** Ledger: both peaks.

---

### Task 8: Floating search

**Files:**
- Modify: `research/roadless/search.py`

**Interfaces:**
- Consumes: Tasks 2, 3, 5.
- Produces: `Archive(step)` with `level(D)`, `offer(D, score, removed)`, `beats(D, J)`; `ConditionalAdd(picker, archive, d_max)`; `FloatRows(block_id, c, power, J0, P0, d_max, archive, t0)` with `.frame()`; `FloatPrune(grow, picker, restore, shortlist, cap, *, screen_eps)`, its `name` `FL<grow>x<picker>r<restore>[m<k>]c<cap>`, and `build(...)` with GrowPrune's signature; a `SearchPlan` Protocol (`name`, `picker`, `screen_eps`, `build(...) -> tuple[Part, Rows]`) that both satisfy; `plan_of` returning either.

- [ ] **Step 1: Archive, ConditionalAdd, FloatRows.**

```python
class Archive:
    """The best exact state per budget level (level = ceil(D / step)): floating search's
    comparison (Pudil) and its rows."""

    def __init__(self, step: float):
        self.step = step
        self.best: dict[int, tuple[float, Score, np.ndarray]] = {}   # level -> (D, score, clearing)

    def level(self, D: float) -> int:
        return int(np.ceil(D / self.step - 1e-9))

    def offer(self, D: float, score: Score, removed: np.ndarray) -> None:
        L = self.level(D)
        if L not in self.best or score.J < self.best[L][1].J:
            self.best[L] = (D, score, removed.copy())

    def beats(self, D: float, J: float) -> bool:
        L = self.level(D)
        return L not in self.best or J < self.best[L][1].J


@dataclass(frozen=True)
class ConditionalAdd:
    """Floating search's conditional inclusion: a greedy add from the current state, scored
    exactly, kept only if it beats the archive's best at its level (and the level is one the
    run records); None otherwise, which ends this run of conditional adds."""
    picker: clear.Picker
    archive: Archive
    d_max: float

    def step(self, s: SearchState) -> Outcome | None:
        c = s.c
        pk = self.picker.pick(c, c.tension(s.power), s.score.J, s.power)
        D = s.D + float(c.cost[pk.cleared].sum())
        if self.archive.level(D) > self.archive.level(self.d_max):
            return None
        J, P = (pk.J, pk.P1) if not np.isnan(pk.J) else clear._exact(c, pk.cleared, s.power)
        s.scorings += 1
        if not self.archive.beats(D, J):
            return None
        return Outcome(pk.cleared, [], Score(J, P), EXACT)


class FloatRows:
    """Floating search's rows: the archive's best state per level, in increasing D, each with
    its full clearing (`clearing`, index order; the states are not nested, so no `cleared`
    deltas: cleared_through does not read these), after a D 0 row."""

    def __init__(self, block_id: str, c: Clearing, power: float, J0: float, P0: float,
                 d_max: float, archive: Archive, t0: float):
        self.block_id, self.n, self.power, self.J0, self.P0, self.d_max, self.t0 = (
            block_id, c.n, power, J0, P0, d_max, t0)
        self.archive = archive

    def wants(self, s: SearchState) -> bool:
        return s.D <= self.d_max + 1e-12 and bool(s.c.removed.any())

    def on_step(self, s: SearchState, o: Outcome) -> None:
        if s.world is EXACT and self.wants(s):
            self.archive.offer(s.D, s.score, s.c.removed)

    def frame(self) -> pd.DataFrame:
        rows = [dict(block=self.block_id, n=self.n, D=0.0, perm=0.0, perm1=0.0, clearing=[],
                     P0=self.P0, t=0.0)]
        for L in sorted(self.archive.best):
            D, sc, r = self.archive.best[L]
            rows.append(dict(block=self.block_id, n=self.n, D=D,
                             perm=1 - (sc.J / self.J0) ** (1 / self.power),
                             perm1=1 - sc.P / self.P0, clearing=np.flatnonzero(r).tolist(),
                             P0=self.P0, t=time.time() - self.t0))
        return pd.DataFrame(rows)
```

- [ ] **Step 2: FloatPrune and the SearchPlan Protocol.** FloatPrune has GrowPrune's fields plus `cap: int` (before the keyword-only `screen_eps`), name `FL{grow:g}x{picker}r{restore:g}[m{shortlist}]c{cap}`, and

```python
    def build(self, block_id: str, c: Clearing, power: float, J0: float, P0: float,
              d_max: float, t0: float, picker: clear.Picker,
              screen: Valuer) -> tuple[Part, Rows]:
        archive = Archive(self.restore)
        rec = FloatRows(block_id, c, power, J0, P0, d_max, archive, t0)
        return Seq((With(Silent(), Until(Do(GreedyAdd(picker)), Reached(self.grow * d_max))),
                    Until(Seq((Do(RestoreBatch(self.restore, self.shortlist, screen)),
                               Until(Do(ConditionalAdd(picker, archive, d_max)),
                                     Spent(self.cap)))),
                          Emptied()))), rec
```
`SearchPlan(Protocol)`: `name` (property), `picker: str`, `screen_eps: float`, `build(...)`; prune_block and main take a `SearchPlan` (no branch on which plan it is).
plan_of: `GP...` as now; `FL([0-9.]+)x(\S+?)r([0-9.]+)(?:m(\d+))?c(\d+)` to FloatPrune; anything else raises.

- [ ] **Step 3: Termination check on one small block, CPU.**

```bash
cd ~/src/reblock && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
PYTHONPATH=. uv run python research/roadless/search.py ZAF.9.3.1_1_19421 FL3xS0.01catr0.005m8c100 2 cpu uni 0.15
```
Expected: finishes; rows in `clear_rows_FL3xS0.01catr0.005m8c100_h0.5_area_p2/`; D increasing; at every level perm at least GP's at the same level on this block (the archive keeps the prune's own states, so floating never ends worse at a level it visited).
Add a temporary assertion while running it (not committed) in ConditionalAdd: a kept add's J is strictly below the archive's previous best at its level.

- [ ] **Step 4: Lint, commit, push** ("floating search: backward floating on grow-then-prune (FL...c<cap>), conditional adds kept only if they beat the archive's best at their level; FloatRows with full clearings").

- [ ] **Step 5: First run** on 19421, 19510, 19537 (CPU, locally, in the background) and 9712, 22422 (GPU, the cluster: `cluster.py submit blocks fl-5 --mesh 0.5 ...`), `FL3xS0.01catr0.005m8c200` at d_max 0.15.
Compare at D 0.05 and 0.15 with GP3xS0.01catr0.005m8 (existing rows for these five) and the greedy S0.01cat.

---

### Task 9: Record and clean up

**Files:**
- Modify: `research/roadless/NOTES.md`, `research/roadless/BACKLOG.md`

- [ ] **Step 1:** NOTES: a section "The add/remove search as one Strategy (owner 2026-10-08: prioritized)": the engine's shape in three sentences; the agreement result (identical / each explained deviation); the GPU memory comparison; floating search's first numbers against GP and the greedy.
- [ ] **Step 2:** BACKLOG: the "Add/remove subset search" entry marked built, floating search's result, and what is next (the substitute-aware restore score).
- [ ] **Step 3:** Remove the worktrees: `git worktree remove .superpowers/sdd/2026-10-08-add-remove-search/trees/<label>` for each, then `git worktree prune`.
- [ ] **Step 4:** Commit and push.
