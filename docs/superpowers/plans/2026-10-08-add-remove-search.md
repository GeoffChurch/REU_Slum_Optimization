# The add/remove search as one Strategy: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every add/remove search in research/roadless (the greedy's three pickers, exchange, grow-then-prune, SIMP's polish) runs as a configuration of one `Round` (rank, refine, build, evaluate, accept) inside one engine (SearchState, schedules, records); their old loops are deleted; backward floating search is the first new schedule.

**Architecture:** search.py holds the engine and every part; clear.py keeps the physics (Clearing, Tension, `exact`, Catchment) and the greedy's CLI; relax.py keeps Relaxation (now also an eps-world `World`) and builds its eps worlds per block.
Each task moves one member and every caller of its old loop, then checks agreement with the current code on three small blocks on the CPU; deviations are reported and explained, not forbidden.

**Tech Stack:** Python 3.12, numpy, pandas, the research code's solvers (pyamg on the CPU); no test framework in research/roadless: verification is the agreement harness (Task 1) and smoke imports.

**Spec:** docs/superpowers/specs/2026-10-08-add-remove-search-design.md (third version).

## Global Constraints

- Branch `research/roadless`; commit with `git -c core.hooksPath=.githooks commit` (never `--no-verify`); push after each task: `git push origin research/roadless`.
- Commit messages end with the two lines `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01RoNpoEMZ1kz2hogCsAsmXJ`.
- The repo's `ruff check .` skips research/ (pyproject `extend-exclude`), so the gate for this plan is explicit paths: `uv run ruff check --select F,E9 research/roadless/{search,clear,relax,polish_greedy,gramprobe,memprobe_greedy,profile_round,releaseprobe}.py` must pass (it passes today); keep lines at most 100 characters, as the files are.
- Every CPU run single-threaded: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1`.
- One GPU process at a time on this machine: `nvidia-smi --query-compute-apps=pid --format=csv,noheader` empty before a GPU run.
- Long runs in the background (`run_in_background`), never `sleep N; cmd` waiters, never `pkill -f`.
- No legacy paths: the task that deletes an old loop moves every caller of it in the same task; no shims, no re-exports.
- Result-changing parameters (width, tries, reach, pool, eps, rtol, cap, step, delta, gap, m) are required, keyword-only where a class has more than one, never defaulted below the edge.
- Closed sets as names: NamedTuples and dataclasses; no positional tuples whose positions mean something, no `getattr`/`hasattr` type tests.
- No tracked file edited while a `cluster.py submit` command is running (a queued or running job is fine; the submit command itself takes a minute).

## Rulings on the spec

1. **One evaluator rule replaces NoScore.** `All(world)` scores a tier only if the acceptor needs a score or the tier holds more than one move; a one-move tier under `Only` or `Best` is passed through unscored, and `apply` scores the state if the record wants it (the same solve: same clearing, `exact` a fresh solve). So no `NoScore` class.
2. **Builders return tier thunks** (`list[Callable[[], Tier]]`), built from arrays the ranking held, never capturing the ranking itself; `Round.step` drops the ranking before calling them, so the tension's system and AMG hierarchy are freed before any scoring solve (Spread's `del t, H`).
3. **`Round` carries a `name`** (the preset it realizes: the picker spec for the greedy) because `clear.run` names row directories by it.
4. **`_singles` and `_pairs` move to search.py** with the Swaps builder (search.py must not import relax); `relax.PAIR_POOL` stays relax's edge constant and is passed in.
5. **Eps worlds are built by their callers per block** (polish_greedy, Incumbent, search.py's CLI through a local `import relax`), not bound from config objects: the spec's "built once where the run is configured" is the caller for these.
6. **`clear.exact(c, removed, power)` is the one exact entry point**; `_exact` survives until Task 5 (its last user, grow-then-prune's prune), then goes; memprobe_greedy patches `clear.exact` from Task 3.

## Review Focus

1. **A state compared in the wrong world.** `Top` and `IfBetter` compare against `s.score`; after an unscored greedy it is NaN. `Top.evaluate` raises if `s.world is not self.world` (a schedule missing its `Rescore`); Task 4's polish rows exercise it.
2. **c.removed left set.** relax.one assumes `c.removed` all False outside its greedy; Task 4 Step 7 asserts it after `one`.
3. **The tension held through scoring.** A tier thunk that captures the ranking keeps the AMG hierarchy alive; Task 7's memprobe compares peaks.
4. **Tie order.** `np.argsort` kinds and strict `<` / `>` are copied exactly from the old code; the harness reports the first diverging row (Task 1 Step 4 proves it does).
5. **Floating search looping or overrunning d_max.** Task 8 asserts every kept add strictly improves its level, and conditional adds stop above d_max (`Above`) and refuse a move past it.

---

### Task 1: The agreement harness and the baseline

**Files:**
- Create: `.superpowers/sdd/2026-10-08-add-remove-search/agree.py` and `bank.pkl` (git-ignored workspace; not committed).

**Interfaces:**
- Produces: `agree.py run <label> <commit> [preset ...]`, `agree.py compare <a> <b>`; trees at `.superpowers/sdd/2026-10-08-add-remove-search/trees/<label>`.

Check blocks `ZAF.9.3.1_1_19421` (112 buildings), `ZAF.9.3.1_1_19510` (88), `ZAF.9.3.1_1_38138` (63).
Measured CPU times (one thread, 19421): tension 8 s, restore tension 7.7 s, exact 3.2 s, an eps 1e-6 screen value ~7 s, value_sgrad at rtol 1e-5 20 s; a GP command is ~75 min per block, SIMP 30 -- 50 min, the greedy up to 20, polish_greedy ~10.
So every command runs per block, in parallel (21 commands a run), and a run is ~1.3 h wall.

- [ ] **Step 1: The bank.**

```bash
cd ~/src/reblock; WS=.superpowers/sdd/2026-10-08-add-remove-search; mkdir -p $WS
PYTHONPATH=. uv run python -c "
import sys; sys.path.insert(0, 'research/roadless')
import common
common.write_bank(['ZAF.9.3.1_1_19421', 'ZAF.9.3.1_1_19510', 'ZAF.9.3.1_1_38138'], '$WS/bank.pkl')"
```

- [ ] **Step 2: agree.py.**

```python
"""The add/remove refactor's agreement check: the presets run per block in a git worktree per
label (fresh rows, the CPU, one thread each), then compared row by row.

    run <label> <commit> [preset ...]     # presets: greedy polish GP SIMP (default: all)
    compare <a> <b>
"""
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
GP = "GP3xS0.01catr0.005m8"
R = "research/roadless"
GREEDY = ["M4", "B0.01g3", "S0.01cat", "S0.01catw4"]
JOBS = 24                                       # 48 cores, shared: half
SKIP = {"t", "t_greedy"}                        # wall times: never compared


def commands(presets: set[str]) -> list[tuple[str, list[str]]]:
    out = []
    for b in IDS:
        if "greedy" in presets:
            out += [(f"greedy {p} {b}", [f"{R}/clear.py", "some", "1", p, "0.5", "0.15", "area",
                                         "2", "uni", "cpu", b]) for p in GREEDY]
        if "polish" in presets:
            out.append((f"polish {b}", [f"{R}/polish_greedy.py", b, "2", "cpu", "S0.01cat", "64",
                                        "8", "uni", "0.05", "0.5", "x2"]))
        if "GP" in presets:
            out.append((f"GP {b}", [f"{R}/search.py", b, GP, "2", "cpu", "uni", "0.15"]))
        if "SIMP" in presets:
            out.append((f"SIMP {b}", [f"{R}/relax.py", "one", b, "2", "cpu", SIMP, "uni", "1",
                                      "0.05"]))
    return out


def rows(tree: Path) -> dict[tuple[str, str], pd.DataFrame]:
    """(preset, block) -> its rows."""
    rr = tree / R
    paths = {}
    for b in IDS:
        for p in GREEDY + [GP]:
            paths[(p, b)] = rr / f"clear_rows_{p}_h0.5_area_p2" / f"{b}.parquet"
        paths[("polish", b)] = rr / "polish_rows/uni/S0.01cat.P64w8x2/D0.05" / f"{b}_p2.parquet"
        paths[("SIMP", b)] = rr / f"relax_rows/uni/{SIMP}/D0.05" / f"{b}_p2.parquet"
    return {k: pd.read_parquet(f) for k, f in paths.items() if f.exists()}


def run(label: str, commit: str, presets: set[str]) -> None:
    tree = WS / "trees" / label
    if not tree.exists():
        subprocess.run(["git", "-C", str(REPO), "worktree", "add", "--detach", str(tree), commit],
                       check=True)
    env = dict(os.environ, PYTHONPATH=".", REBLOCK_BLOCK_BANK=str(WS / "bank.pkl"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMBA_NUM_THREADS="1")
    (tree / "logs").mkdir(exist_ok=True)

    def one(item: tuple[str, list[str]]) -> int:
        name, cmd = item
        with open(tree / "logs" / (name.replace(" ", "_") + ".log"), "w") as fh:
            rc = subprocess.run([str(PY), "-u", *cmd], cwd=tree, env=env, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        print(f"{label}: {name} rc={rc}", flush=True)
        return rc

    with ThreadPoolExecutor(JOBS) as ex:
        bad = [rc for rc in ex.map(one, commands(presets)) if rc]
    if bad:
        raise SystemExit(f"{len(bad)} commands failed: see {tree}/logs")


def _same(a, b) -> bool:
    if isinstance(a, (list, np.ndarray)) or isinstance(b, (list, np.ndarray)):
        return list(np.asarray(a).ravel()) == list(np.asarray(b).ravel())
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def compare(a: str, b: str) -> int:
    """Each (preset, block): identical, or its first differing row and columns. Returns how many
    differ."""
    ra, rb = rows(WS / "trees" / a), rows(WS / "trees" / b)
    differ = 0
    for key in sorted(set(ra) | set(rb)):
        if key not in ra or key not in rb:
            print(f"{key}: only in {a if key in ra else b}")
            differ += 1
            continue
        x, y = ra[key], rb[key]
        if list(x.columns) != list(y.columns) or len(x) != len(y):
            print(f"{key}: shape {x.shape} {list(x.columns)} vs {y.shape} {list(y.columns)}")
            differ += 1
        cols = [c for c in x.columns if c not in SKIP and c in y.columns]
        for i in range(min(len(x), len(y))):
            bad = [c for c in cols if not _same(x[c].iloc[i], y[c].iloc[i])]
            if bad:
                differ += 1
                print(f"{key}: FIRST DIFFERENCE at row {i}: "
                      + "; ".join(f"{c} {x[c].iloc[i]!r} vs {y[c].iloc[i]!r}" for c in bad))
                break
        else:
            if len(x) == len(y):
                print(f"{key}: identical ({len(x)} rows)")
    return differ


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], set(sys.argv[4:]) or {"greedy", "polish", "GP", "SIMP"})
    elif sys.argv[1] == "compare":
        raise SystemExit(1 if compare(sys.argv[2], sys.argv[3]) else 0)
    else:
        raise SystemExit(__doc__)
```

- [ ] **Step 3: The current code, twice, concurrently (the CPU noise floor).**

```bash
cd ~/src/reblock; WS=.superpowers/sdd/2026-10-08-add-remove-search
BASE=$(git rev-parse HEAD); echo "BASE $BASE" >> $WS/progress.md
# both in the background (run_in_background), at the same time: ~1.3 h
uv run python $WS/agree.py run base1 $BASE > $WS/base1.log 2>&1
uv run python $WS/agree.py run base2 $BASE > $WS/base2.log 2>&1
# when both are done:
uv run python $WS/agree.py compare base1 base2
```
Expected: every (preset, block) `identical`.
A preset that differs is not deterministic on the CPU there: record which in the ledger; later comparisons treat its difference against base1 as no worse than base2's.

- [ ] **Step 4: Prove the harness reports a difference.**
  - In `trees/base2/research/roadless/relax.py`'s `exchange`, change `top = np.argpartition(e, k - 1)[:k]` to `top = np.argpartition(e, k - 1)[:max(k - 1, 1)]`.
  - Delete `trees/base2/research/roadless/polish_rows` and `trees/base2/research/roadless/relax_rows`; `agree.py run base2 <BASE> polish SIMP`; `agree.py compare base1 base2`.
    Expected: a FIRST DIFFERENCE on a polish row (fewer scorings at least).
  - Revert (`git -C $WS/trees/base2 checkout -- research/roadless/relax.py`), delete those rows again, rerun `polish SIMP` so base2 is clean, and compare once more: identical.

- [ ] **Step 5: Ledger** (`$WS/progress.md`): BASE, the noise floor, the fault-injection result. Nothing to commit.

---

### Task 2: The engine and the parts' protocols

**Files:**
- Modify: `research/roadless/search.py` (the engine above GrowPrune; GrowPrune, restore_batch, grow_prune untouched except the relax import)
- Modify: `research/roadless/clear.py` (add `exact`)

**Interfaces:**
- Produces in search.py: `Score`, `UNSCORED`, `World` (Protocol), `EXACT`, `Move`, `Candidate`, `Tier` (Protocol), `Batches`, `IndexMoves`, `Outcome`, `SearchState` (+ `start`, `D`), `Step`, `Record`, `Stop`, `Part`, `Ranker`, `Refiner`, `Builder`, `Evaluator`, `Acceptor` (Protocols), `Round`, `apply`, `after`, parts `Do`, `Until`, `Seq`, `With`, `Rescore`, stops `Reached`, `Emptied`, `Spent`, `Above`, `AnyOf`, record `Silent`, evaluators `All`, `Top`, acceptors `Only`, `Best`, `IfBetter`, keys `LowestJ`, `GainPerCost`.
- Produces in clear.py: `exact(c, removed, power) -> tuple[float, float]`.

- [ ] **Step 1: clear.exact**, next to `_exact` (which stays until Task 5):

```python
def exact(c: Clearing, removed: np.ndarray, power: float) -> tuple[float, float]:
    """(J_power, P) of the clearing `removed`: real geometry, the metric conductance,
    RTOL_SCORE. The one exact entry point (memprobe_greedy instruments it)."""
    op = c.open_(removed)
    sol = lifted.solve(c.sc.grid, op, c.sc.f, c.p, rtol=RTOL_SCORE)
    return c.sc.J(c.sc.home_u_of(sol, op), power), sol.P
```

- [ ] **Step 2: search.py stops importing relax at module level.** Delete `from relax import Relaxation`; in `grow_prune`, where `screen = Relaxation(...)` is built, write `import relax  # the eps world is the caller's to build; search.py does not import relax at the top` and `screen = relax.Relaxation(c, power, q=1.0, rtol=RTOL_SCORE, eps=EPS_SCREEN)`. Annotate `restore_batch`'s `screen` as `Valuer` (defined below). Change the `from clear import ...` line to `import clear` plus the names still used (`from clear import RTOL_SCORE, Clearing, _exact, grow, picker_of, rows_dir, sweep_of`) until Task 3 and 5 remove them.

- [ ] **Step 3: The engine.** Add `from collections.abc import Callable`, `from dataclasses import dataclass`, `from typing import Generic, Protocol, TypeVar`, and after the imports:

```python
R = TypeVar("R")


class Score(NamedTuple):
    J: float                    # J_power
    P: float                    # P (J_1); NaN in an eps world


UNSCORED = Score(np.nan, np.nan)


class World(Protocol):
    """Where a clearing is scored."""

    def score(self, s: SearchState, removed: np.ndarray) -> Score: ...


class _Exact:
    """The exact world: real geometry, the metric conductance, RTOL_SCORE (clear.exact, looked
    up at call time so memprobe_greedy's instrumentation applies)."""

    def score(self, s: SearchState, removed: np.ndarray) -> Score:
        return Score(*clear.exact(s.c, removed, s.power))


EXACT = _Exact()


class Valuer(Protocol):
    """An eps world's J of a design x (relax.Relaxation)."""

    def value(self, x: np.ndarray) -> float: ...


class Move(NamedTuple):
    add: list[int]              # in pick order
    restore: list[int]


class Candidate(NamedTuple):
    move: Move
    est: float                  # its linearized change in J (lower better)
    score: Score | None         # in the evaluator's world, if scored


class Tier(Protocol):
    """One tier of a round's moves: their estimates, each move materialised only when looked at."""
    est: np.ndarray

    def move(self, i: int) -> Move: ...


@dataclass(frozen=True)
class Batches:
    """A tier of explicit moves (a batch's buildings in pick order)."""
    moves: list[Move]
    est: np.ndarray

    def move(self, i: int) -> Move:
        return self.moves[i]


@dataclass(frozen=True)
class IndexMoves:
    """A tier of small moves as a (4, M) array: rows 0-1 buildings added, 2-3 restored, -1
    none (exchange's ~10^6 swaps, materialised only for the few scored)."""
    idx: np.ndarray
    est: np.ndarray

    def move(self, i: int) -> Move:
        return Move([int(x) for x in self.idx[:2, i] if x >= 0],
                    [int(x) for x in self.idx[2:, i] if x >= 0])


class Outcome(NamedTuple):
    cleared: list[int]          # in pick order
    restored: list[int]
    score: Score | None         # the new state's, in `world`
    world: World | None


@dataclass
class SearchState:
    """One block's search: c.removed is the clearing (only `apply` changes it); `score` the
    current state's in `world` (UNSCORED, None: not scored); `order` every building cleared, in
    pick order (relax.cut_to_budget's); `scorings` the candidate scorings spent (the caps);
    `movable` the buildings a move may touch."""
    c: Clearing
    power: float
    score: Score
    world: World | None
    order: list[int]
    scorings: int
    movable: np.ndarray

    @classmethod
    def start(cls, c: Clearing, power: float, score: Score, world: World | None) -> SearchState:
        """From c's current clearing, nothing spent, every building movable."""
        return cls(c, power, score, world, [], 0, np.ones(c.n, dtype=bool))

    @property
    def D(self) -> float:
        return float(self.c.cost[self.c.removed].sum())


def after(removed: np.ndarray, m: Move) -> np.ndarray:
    """The clearing after move m."""
    r = removed.copy()
    r[m.add] = True
    r[m.restore] = False
    return r


class Step(Protocol):
    def step(self, s: SearchState) -> Outcome | None: ...


class Record(Protocol):
    """What a run owes: which states must be scored exactly, and what it keeps of each step."""

    def wants(self, s: SearchState) -> bool: ...

    def on_step(self, s: SearchState, o: Outcome) -> None: ...


class Stop(Protocol):
    def done(self, s: SearchState) -> bool: ...


class Part(Protocol):
    """A schedule: runs on `s` under `rec`, returns the moves it made."""

    def __call__(self, s: SearchState, rec: Record) -> int: ...


class Ranker(Protocol[R]):
    def rank(self, s: SearchState) -> R: ...


class Refiner(Protocol[R]):
    def refine(self, s: SearchState, r: R, first_k: int) -> R: ...


class Builder(Protocol[R]):
    """Candidate moves as tiers (a later tier only if no move of an earlier one is accepted),
    each a thunk that must not hold the ranking (Round drops it before any is called)."""

    def tiers(self, s: SearchState, r: R) -> list[Callable[[], Tier]]: ...

    def size(self, s: SearchState, r: R) -> int: ...


class Evaluator(Protocol):
    world: World

    def evaluate(self, s: SearchState, t: Tier, needs_score: bool) -> list[Candidate]: ...


class Acceptor(Protocol):
    @property
    def needs_score(self) -> bool: ...     # whether choosing needs the candidates scored

    def accept(self, s: SearchState, cands: list[Candidate]) -> Candidate | None: ...


@dataclass(frozen=True, kw_only=True)
class Round(Generic[R]):
    """One round of an add/remove search: rank the buildings (first order), optionally refine
    a shortlist's ranking in a costlier world, build tiers of candidate moves, then per tier
    evaluate and accept; the first tier with an accepted move gives the outcome."""
    name: str
    rank: Ranker[R]
    refine: Refiner[R] | None
    build: Builder[R]
    evaluate: Evaluator
    accept: Acceptor

    def step(self, s: SearchState) -> Outcome | None:
        r = self.rank.rank(s)
        if self.refine is not None:
            r = self.refine.refine(s, r, self.build.size(s, r))
        tiers = self.build.tiers(s, r)
        del r                   # the tension's system and hierarchy, before any scoring solve
        for tier in tiers:
            cand = self.accept.accept(s, self.evaluate.evaluate(s, tier(),
                                                                self.accept.needs_score))
            if cand is not None:
                return Outcome(cand.move.add, cand.move.restore, cand.score,
                               None if cand.score is None else self.evaluate.world)
        return None


def apply(s: SearchState, o: Outcome, rec: Record) -> None:
    """Make the move; score the new state exactly if `rec` wants it and the round did not;
    tell `rec`."""
    c = s.c
    if o.cleared:
        c.removed[o.cleared] = True
        s.order.extend(o.cleared)
    if o.restored:
        c.removed[o.restored] = False
    s.score, s.world = (UNSCORED, None) if o.score is None else (o.score, o.world)
    if rec.wants(s) and s.world is not EXACT:
        s.score, s.world = EXACT.score(s, c.removed), EXACT
    rec.on_step(s, o)


@dataclass(frozen=True)
class Do:
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
class Rescore:
    """The state scored in `world` (not a candidate scoring): before a phase that compares in
    it."""
    world: World

    def __call__(self, s: SearchState, rec: Record) -> int:
        s.score, s.world = self.world.score(s, s.c.removed), self.world
        return 0


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
class Above:
    d: float

    def done(self, s: SearchState) -> bool:
        return s.D > self.d + 1e-12


@dataclass(frozen=True)
class AnyOf:
    stops: tuple[Stop, ...]

    def done(self, s: SearchState) -> bool:
        return any(x.done(s) for x in self.stops)


@dataclass(frozen=True)
class Silent:
    """Owes nothing: no state scored for it, nothing kept."""

    def wants(self, s: SearchState) -> bool:
        return False

    def on_step(self, s: SearchState, o: Outcome) -> None:
        pass


@dataclass(frozen=True)
class All:
    """Score every move of a tier in `world`, unless it has one and no score is needed (Spread's
    'score only if there is a choice', Screened's always, as one rule)."""
    world: World

    def evaluate(self, s: SearchState, t: Tier, needs_score: bool) -> list[Candidate]:
        n = len(t.est)
        if n == 1 and not needs_score:
            return [Candidate(t.move(0), float(t.est[0]), None)]
        out = []
        for i in range(n):
            m = t.move(i)
            s.scorings += 1
            out.append(Candidate(m, float(t.est[i]), self.world.score(s, after(s.c.removed, m))))
        return out


@dataclass(frozen=True, kw_only=True)
class Top:
    """Score the best `width` moves of a tier by estimate (fewer as `tries` runs out), in
    estimate order (exchange's)."""
    world: World
    width: int
    tries: int

    def evaluate(self, s: SearchState, t: Tier, needs_score: bool) -> list[Candidate]:
        if s.world is not self.world:
            raise ValueError("the state is not scored in this world: Rescore before the phase")
        e = t.est
        k = min(self.width, self.tries - s.scorings, int(np.isfinite(e).sum()))
        if k == 0:
            return []
        top = np.argpartition(e, k - 1)[:k]
        out = []
        for i in top[np.argsort(e[top])]:
            m = t.move(int(i))
            s.scorings += 1
            out.append(Candidate(m, float(e[i]), self.world.score(s, after(s.c.removed, m))))
        return out


@dataclass(frozen=True)
class Only:
    """The tier's one move."""

    @property
    def needs_score(self) -> bool:
        return False

    def accept(self, s: SearchState, cands: list[Candidate]) -> Candidate | None:
        return cands[0] if cands else None


class Key(Protocol):
    """Lower is better."""

    def of(self, s: SearchState, cand: Candidate) -> float: ...


@dataclass(frozen=True)
class LowestJ:
    def of(self, s: SearchState, cand: Candidate) -> float:
        return cand.score.J


@dataclass(frozen=True)
class GainPerCost:
    """Screened's: the exact gain over the state's J per unit of population, negated."""

    def of(self, s: SearchState, cand: Candidate) -> float:
        return -(s.score.J - cand.score.J) / s.c.cost[cand.move.add[0]]


@dataclass(frozen=True)
class Best:
    """Always a move: the first with the lowest key (one candidate: it, unscored if it was)."""
    key: Key

    @property
    def needs_score(self) -> bool:
        return False

    def accept(self, s: SearchState, cands: list[Candidate]) -> Candidate | None:
        if len(cands) <= 1:
            return cands[0] if cands else None
        best, bestv = None, np.inf
        for cand in cands:
            v = self.key.of(s, cand)
            if v < bestv:
                best, bestv = cand, v
        return best


@dataclass(frozen=True)
class IfBetter:
    """The first scored move with the lowest J, if strictly below the state's (exchange's)."""

    @property
    def needs_score(self) -> bool:
        return True

    def accept(self, s: SearchState, cands: list[Candidate]) -> Candidate | None:
        best = None
        for cand in cands:
            if cand.score.J < s.score.J and (best is None or cand.score.J < best.score.J):
                best = cand
        return best
```
Checks for exactness against the old code (write them as comments only where a reader would wonder):
- `Best` with `GainPerCost`: the old Screened kept the first `v = (J - Jj) / cost[j]` strictly greater than the best so far, from -inf; `-(J - Jj) / cost` strictly less from +inf picks the same one (negation is exact). Spread kept the first `Jb` strictly below the best: `LowestJ` the same.
- `needs_score` is a read-only property of each acceptor class (Only and Best never need a score to choose, IfBetter and BeatsArchive always do), not a setting.

- [ ] **Step 4: Smoke and lint.**

```bash
cd ~/src/reblock && PYTHONPATH=. uv run python -c "import sys; sys.path.insert(0,'research/roadless'); import search, relax, polish_greedy, clear; print(search.Round, search.EXACT)"
uv run ruff check --select F,E9 research/roadless/{search,clear,relax,polish_greedy,gramprobe,memprobe_greedy,profile_round,releaseprobe}.py
```
Expected: prints, `All checks passed!`. (Importing relax first and search second, and the reverse, must both work: `python -c "import relax, search"` and `"import search, relax"`.)

- [ ] **Step 5: Commit and push** ("research: roadless -- the add/remove engine and its parts' protocols (search.py): Round = rank, refine, build, evaluate, accept; SearchState, schedules, stops, records; clear.exact; search.py no longer imports relax at the top").

---

### Task 3: The greedy as Rounds

**Files:**
- Modify: `research/roadless/search.py` (AddTension, Gains, TopSingles, Spaced, Diverse, greedy_round, GreedyRows; grow_prune's grow phase on the engine)
- Modify: `research/roadless/clear.py` (delete grow, Picked, Picker, Screened, Batched, Spread, picker_of; greedy_block, run and the CLI use search; GramSource, Catchment, `_spread`, `_next_target` stay)
- Modify: `research/roadless/relax.py` (relax.one's seed and track), `polish_greedy.py` (its greedy phase), `gramprobe.py`, `memprobe_greedy.py`, `profile_round.py`, `releaseprobe.py`

**Interfaces:**
- Consumes: Task 2.
- Produces: `search.greedy_round(spec: str, sweep: clear.Sweep) -> Round` (`M<m>`, `B<d>g<gap>`, `S<d>cat[w<w>]`; anything else raises), `search.SPREAD_REACH = 3.0`, `search.GreedyRows(block_id, c, power, J0, P0, t0)` with `.rows`, `search.greedy(rnd, d_max) -> Part` (Until(Do(rnd), Reached(d_max))).

- [ ] **Step 1: The add ranking and the three builders** (search.py). The bodies are the old pickers', split at the point where they scored:

```python
class Gains(NamedTuple):
    t: clear.Tension            # the round's tension (Catchment sweeps its system)
    T: np.ndarray               # t.g / cost: gain per unit of population (removed: -inf)


@dataclass(frozen=True)
class AddTension:
    def rank(self, s: SearchState) -> Gains:
        t = s.c.tension(s.power)
        return Gains(t, t.g / s.c.cost)


@dataclass(frozen=True)
class TopSingles:
    """The top m by tension per unit of population, one building each (Screened's)."""
    m: int

    def size(self, s: SearchState, r: Gains) -> int:
        return self.m

    def tiers(self, s: SearchState, r: Gains) -> list[Callable[[], Tier]]:
        top = [int(j) for j in np.argsort(-r.T)[:self.m] if not s.c.removed[j]]
        est = -r.T[top]
        return [lambda: Batches([Move([j], []) for j in top], est)]


@dataclass(frozen=True, kw_only=True)
class Spaced:
    """By tension per unit of population, every building not within `gap` metres of one taken,
    until D reaches the next multiple of `delta` (Batched's)."""
    delta: float
    gap: float

    def size(self, s: SearchState, r: Gains) -> int:
        return len(self._taken(s, r))

    def _taken(self, s: SearchState, r: Gains) -> list[int]:
        c = s.c
        D, target = clear._next_target(c, self.delta)
        taken: list[int] = []
        near: set[int] = set()
        for j in (int(j) for j in np.argsort(-r.T) if not c.removed[j]):
            if D >= target - 1e-12:
                break
            if j in near:
                continue
            taken.append(j)
            D += float(c.cost[j])
            near.update(int(k) for k in c.sc.tree.query(c.sc.polys[j], predicate="dwithin",
                                                         distance=self.gap))
        return taken

    def tiers(self, s: SearchState, r: Gains) -> list[Callable[[], Tier]]:
        taken = self._taken(s, r)
        return [lambda: Batches([Move(taken, [])], np.zeros(1))]


@dataclass(frozen=True, kw_only=True)
class Diverse:
    """Shortlist by tension per unit of population up to `reach` times the round's population
    step, measure how alike the candidates' effects are with `source`, and build `width`
    batches with clear._spread (the k-th started from the shortlist's k-th best), distinct
    (Spread's)."""
    delta: float
    source: clear.GramSource
    reach: float
    width: int

    def size(self, s: SearchState, r: Gains) -> int:
        return len(self._batches(s, r)[0].add)

    def _batches(self, s: SearchState, r: Gains) -> list[Move]:
        c = s.c
        D, target = clear._next_target(c, self.delta)
        order = [int(j) for j in np.argsort(-r.T) if not c.removed[j]]
        cum = np.cumsum(c.cost[order])
        cand = order[:int(np.searchsorted(cum, self.reach * (target - D))) + 1]
        H = self.source.gram(c, r.t, cand, s.power)
        gain, cost = r.t.g[cand], c.cost[cand]
        firsts = [None] + [int(i) for i in np.argsort(-gain / cost)[1:self.width]]
        batches: dict[frozenset[int], list[int]] = {}   # distinct batches, each in pick order
        for f in firsts:
            chosen = clear._spread(gain, cost, H, target - D, first=f)
            batches.setdefault(frozenset(chosen), chosen)
        return [Move([cand[i] for i in b], []) for b in batches.values()]

    def tiers(self, s: SearchState, r: Gains) -> list[Callable[[], Tier]]:
        moves = self._batches(s, r)
        return [lambda: Batches(moves, np.zeros(len(moves)))]


SPREAD_REACH = 3.0      # Diverse's shortlist, in population steps (the measured default)


def greedy_round(spec: str, sweep: clear.Sweep) -> Round[Gains]:
    """`M4`: the top 4 scored exactly, the best gain per unit of population; `B0.01g3`: a batch
    spaced 3 m apart to the next 0.01 of population; `S0.01cat`: a catchment-diverse batch to
    the next 0.01, `S0.01catw4` the best of 4 such batches. (Impact, Sketch and the no-spacing
    null were measured and dominated: NOTES, "Batching"; Impact again on the GPU, "GPU-resident
    rounds".)"""
    # names rebuilt from the parsed values, as the pickers' were (row directories)
    if m := re.fullmatch(r"M(\d+)", spec):
        k = int(m[1])
        return Round(name=f"M{k}", rank=AddTension(), refine=None, build=TopSingles(k),
                     evaluate=All(EXACT), accept=Best(GainPerCost()))
    if m := re.fullmatch(r"B([0-9.]+)g([0-9.]+)", spec):
        delta, gap = float(m[1]), float(m[2])
        return Round(name=f"B{delta:g}g{gap:g}", rank=AddTension(), refine=None,
                     build=Spaced(delta=delta, gap=gap), evaluate=All(EXACT), accept=Only())
    if m := re.fullmatch(r"S([0-9.]+)cat(?:w(\d+))?", spec):
        delta, width = float(m[1]), 1 if m[2] is None else int(m[2])
        source = clear.Catchment(sweep)
        return Round(name=f"S{delta:g}{source.name}" + (f"w{width}" if width > 1 else ""),
                     rank=AddTension(), refine=None,
                     build=Diverse(delta=delta, source=source, reach=SPREAD_REACH, width=width),
                     evaluate=All(EXACT), accept=Best(LowestJ()))
    raise ValueError(f"unknown picker {spec!r}")


def greedy(rnd: Round, d_max: float) -> Part:
    """The greedy: `rnd` until D reaches d_max."""
    return Until(Do(rnd), Reached(d_max))
```
Check against the old code, and say so in the report: Diverse calls `_batches` twice when a refiner asks `size` (none of the greedy rounds has one, so `size` is never called for them); `TopSingles` estimates are not used to choose (All scores every move); Batched's old `D` loop and Spread's shortlist are copied line for line; the Tension lives only in `Gains`, which Round drops before the thunks run (the thunks capture `top`, `taken`, `moves`: lists, not the ranking).
`re` is already imported in search.py. Every part with more than one field is `@dataclass(frozen=True, kw_only=True)` (the Global Constraint), so they are built by keyword.

- [ ] **Step 2: GreedyRows** (search.py):

```python
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

- [ ] **Step 3: clear.py.** Delete `Picked`, `Picker`, `Screened`, `Batched`, `Spread`, `grow`, `picker_of`.
  - Type-only import for the annotations: `from typing import TYPE_CHECKING` and `if TYPE_CHECKING: import search as engine` (search imports clear at run time), annotations written `engine.Round` (`from __future__ import annotations` is present, so they are never evaluated). After the deletions, drop `import re` and `NamedTuple` from clear.py's imports if ruff F401 reports them unused, and rewrite the module docstring's lines on the pickers (clear.py:8-11) to name `search.greedy_round`.
  - `greedy_block(b, rnd, h, d_max, out, population, power, along, solver, search=None)`: its parameter `picker: Picker` becomes `rnd: engine.Round`; its body from `t0 = time.time()` through the loop becomes

```python
    import search as engine     # search imports clear, so here; `search` is the conductance
    t0 = time.time()
    rec = engine.GreedyRows(b.block_id, c, power, J0, P0, t0)
    engine.greedy(rnd, d_max)(engine.SearchState.start(c, power, engine.Score(J0, P0),
                                                       engine.EXACT), rec)
    rows = rec.rows
```
  and the old `rows = [...]` initialisation above it goes (GreedyRows makes the step-0 row).
  - `_one` passes `_CFG["picker"]` unchanged (it now holds a Round); `run(workers, picker, ...)` keeps its parameter name `picker`, annotated `engine.Round`, and names the directory by `picker.name` as now.
  - The CLI's `run|some` branch: `picker_of(sys.argv[3], sweep_of(sys.argv[9]))` becomes `engine.greedy_round(sys.argv[3], sweep_of(sys.argv[9]))` with `import search as engine` inside that branch (commented as above).
  - Everywhere in clear.py the module is `engine`, never `search`: `greedy_block`, `run` and `Clearing` have a parameter `search` (the search conductance).
  - `Picker`, `Tension` and `Clearing` annotations elsewhere in clear.py: none use Picker after the deletions (grep to confirm).

- [ ] **Step 4: The other callers.**
  - relax.py `one`: `picker = clear.picker_of(spec, clear.sweep_of(device))` and the `order = [...]` / `c.removed[:] = False` lines become

```python
        rnd = search.greedy_round(spec, clear.sweep_of(device))
        s = search.SearchState.start(c, power, search.Score(rel.J0, np.nan), search.EXACT)
        search.greedy(rnd, budget)(s, search.Silent())
        c.removed[:] = False
        order = s.order
```
  (add `import search` at the top of relax.py: search does not import relax, so no cycle.)
  - polish_greedy.py: `picker = clear.picker_of(picker_spec, clear.sweep_of(device))` becomes `rnd = search.greedy_round(picker_spec, clear.sweep_of(device))`; the greedy lines `order = [j for pk, _ in clear.grow(...) ...]`, `t_greedy = ...`, `c.removed[:] = False`, `r = relax.cut_to_budget(order, ...)` become

```python
    s = search.SearchState.start(c, power, search.Score(c.sc.J(c.sc.u0, power), c.sc.P0),
                                 search.EXACT)
    search.greedy(rnd, budget)(s, search.Silent())
    t_greedy = time.time() - t0
    c.removed[:] = False
    r = relax.cut_to_budget(s.order, c.cost, budget)
```
  (exchange stays as it is until Task 4; `import search`.)
  - search.py grow_prune: `picker_of(plan.picker, sweep_of(device))` becomes `greedy_round(plan.picker, sweep_of(device))`; its grow loop `for _pk, _D in grow(c, ..., plan.grow * d_max, J0, True): pass` becomes `greedy(rnd, plan.grow * d_max)(SearchState.start(c, power, Score(J0, P0), EXACT), Silent())` (the grow phase's states were scored and thrown away: now they are not scored; only `t` can change). Drop `grow` and `picker_of` from its `from clear import` line.
  - gramprobe.py: `picker = clear.picker_of("S0.01cat", ...)` becomes `rnd = search.greedy_round("S0.01cat", clear.sweep_of("gpu"))`; its patch `object.__setattr__(picker.source, "gram", gram_)` becomes `object.__setattr__(rnd.build.source, "gram", gram_)` (and `gram = rnd.build.source.gram` wherever it saved the original); its loop becomes `search.greedy(rnd, d_max)(search.SearchState.start(c, 2.0, search.Score(J0, c.sc.P0), search.EXACT), search.GreedyRows(bid, c, 2.0, J0, c.sc.P0, time.time()))` (a record that wants every state, so each step's exact solve stays in what it times, as the old `grow(..., True)` did; `bid` is main's block id, check the name). `import search`.
  - memprobe_greedy.py imports the module as `import search as engine` (its `main` has a local `search`, the search conductance) and uses `engine.` below; the same renames for its picker (`rnd = engine.greedy_round(...)`) and gram patch; `exact = clear.exact` and `clear.exact = exact_` with `exact_(c_, removed, power)` calling `exact(c_, removed, power)` (the scoring report around it unchanged); its loop becomes

```python
    class _Print:
        def wants(self, s):
            return True

        def on_step(self, s, o):
            print(f"  step {step[0]} D {s.D:.3f} perm' {1 - (s.score.J / J0) ** 0.5:.3f}",
                  flush=True)

    engine.greedy(rnd, d_max)(engine.SearchState.start(c, 2.0, engine.Score(J0, c.sc.P0),
                                                       engine.EXACT), _Print())
```
  - profile_round.py: `pk = picker_of(picker, sweep_of(solver))` becomes `rnd = search.greedy_round(picker, sweep_of(solver))`; the timed `t = c.tension(2.0)` / `pk.pick(c, t, J, 2.0, True)` pair becomes one timed `search.Do(rnd)(s, rec)` with `s = search.SearchState.start(c, 2.0, search.Score(J, c.sc.P0), search.EXACT)` and `rec = search.GreedyRows(bid, c, 2.0, J, c.sc.P0, time.time())` (so the step's exact solve is timed, as `pick(..., True)` scored inside), printed as `step {secs}s ({len(s.order)} cleared)`; its docstring says it profiles one round (rank, build, evaluate, the state's exact score) instead of tension and pick separately. `import search`; drop `picker_of` from its clear import.
  - releaseprobe.py: `clear.picker_of("S0.01cat", ...)` becomes `search.greedy_round("S0.01cat", clear.sweep_of("gpu"))` (`import search` inside `variant`, next to its other imports).
  - `grep -n "picker_of\|\.pick(\|clear\.grow\|Screened\|Batched\|Spread(" research/roadless/*.py` finds nothing else (fix anything it does).

- [ ] **Step 5: Smoke, lint, commit locally** (no push yet).

```bash
cd ~/src/reblock && for m in "relax, search" "search, relax" "clear" "polish_greedy" "gramprobe" "memprobe_greedy" "profile_round" "releaseprobe"; do PYTHONPATH=. uv run python -c "import sys; sys.path.insert(0,'research/roadless'); import $m" || echo "FAILED $m"; done
uv run ruff check --select F,E9 research/roadless/{search,clear,relax,polish_greedy,gramprobe,memprobe_greedy,profile_round,releaseprobe}.py
git add research/roadless/{search,clear,relax,polish_greedy,gramprobe,memprobe_greedy,profile_round,releaseprobe}.py
git -c core.hooksPath=.githooks commit -m "research: roadless -- the greedy as Rounds: AddTension, TopSingles / Spaced / Diverse, Best / Only (search.greedy_round replaces picker_of); the pickers, Picker and clear.grow deleted, every caller on the engine

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RoNpoEMZ1kz2hogCsAsmXJ"
```

- [ ] **Step 6: Agreement, every preset** (the greedy runs inside polish, GP and SIMP too):
`uv run python $WS/agree.py run t3 $(git rev-parse HEAD)` (background, ~1.3 h), then `compare base1 t3`.
Expected: all identical. Any difference: the first row and its columns; find the cause (a tie, an order, a solve) and either fix it or explain it in the ledger. Then push.

---

### Task 4: Exchange as a Round

**Files:**
- Modify: `research/roadless/search.py` (Graded, Grad, Gradient, Swaps, `_singles`, `_pairs`, `exchange_round`)
- Modify: `research/roadless/relax.py` (Relaxation.score; Incumbent.polish on the engine; delete exchange, `_singles`, `_pairs`)
- Modify: `research/roadless/polish_greedy.py`

**Interfaces:**
- Consumes: Tasks 2-3.
- Produces: `search.exchange_round(world, *, budget, width, pairs, pool, tries) -> Round[Grad]`, `search.polish(world, *, budget, width, pairs, pool, tries) -> Part` (Seq(Rescore(world), Until(Do(round), Spent(tries)))), `relax.Relaxation.score(s, removed) -> search.Score`.

- [ ] **Step 1: search.py.** Move `_singles` and `_pairs` from relax.py, `_pairs` taking `pool` in place of the global `PAIR_POOL`, and both returning `IndexMoves(mv, est)` instead of the pair `(mv, est)` (their bodies otherwise verbatim):

```python
def _pairs(g, ins, outs, cost, left, pool: int):
    """Two in for one out and one in for two out, among the `pool` buildings most promising
    to add (most negative gradient) and cheapest to close."""
    pin = ins[np.argsort(g[ins], kind="stable")[:pool]]
    pout = outs[np.argsort(-g[outs], kind="stable")[:pool]]
    ...                                                     # the rest unchanged
```
and add:

```python
class Graded(World, Protocol):
    """An eps world with its gradient (relax.Relaxation)."""

    def value_sgrad(self, x: np.ndarray) -> tuple[float, np.ndarray]: ...


class Grad(NamedTuple):
    g: np.ndarray               # dJ/ds at the 0/1 clearing r: adding a is g_a, closing b -g_b
    r: np.ndarray               # the clearing, as 0/1 floats


@dataclass(frozen=True)
class Gradient:
    world: Graded

    def rank(self, s: SearchState) -> Grad:
        r = s.c.removed.astype(float)
        return Grad(self.world.value_sgrad(r)[1], r)


@dataclass(frozen=True, kw_only=True)
class Swaps:
    """Add one the budget left allows or swap one in for a cleared one, within s.movable; with
    `pairs`, a second tier of two-in-one-out and one-in-two-out among the `pool` most
    promising (floating search's escalation: their summed estimates would crowd the singles
    out of one ranking)."""
    budget: float
    pairs: bool
    pool: int

    def size(self, s: SearchState, r: Grad) -> int:
        return 1

    def tiers(self, s: SearchState, r: Grad) -> list[Callable[[], Tier]]:
        cost, g, x = s.c.cost, r.g, r.r
        left = self.budget + 1e-12 - float(cost @ x)
        movable = np.flatnonzero(s.movable)
        ins, outs = movable[x[movable] == 0], movable[x[movable] > 0]
        out: list[Callable[[], Tier]] = [lambda: _singles(g, ins, outs, cost, left)]
        if self.pairs:
            out.append(lambda: _pairs(g, ins, outs, cost, left, self.pool))
        return out


def exchange_round(world: Graded, *, budget: float, width: int, pairs: bool, pool: int,
                   tries: int) -> Round[Grad]:
    """Exchange refinement's round: moves ranked by their linearized change from the world's
    gradient at the 0/1 clearing, the best `width` scored in it, the best kept if it improves."""
    return Round(name=f"P{tries}w{width}" + ("x2" if pairs else ""), rank=Gradient(world),
                 refine=None, build=Swaps(budget=budget, pairs=pairs, pool=pool),
                 evaluate=Top(world=world, width=width, tries=tries), accept=IfBetter())


def polish(world: Graded, *, budget: float, width: int, pairs: bool, pool: int,
           tries: int) -> Part:
    """Exchange refinement until a round improves nothing or `tries` scorings are spent."""
    return Seq((Rescore(world), Until(Do(exchange_round(world, budget=budget, width=width,
                                                        pairs=pairs, pool=pool, tries=tries)),
                                      Spent(tries))))
```
Check against the old `exchange` (say so in the report): `left`, `ins`/`outs` (np.flatnonzero of the mask equals the old sorted `movable` index array), the per-tier `k`, `argpartition` then `argsort`, the scoring count before each score, the strict `<` against J and against the best, pairs only when singles accept nothing. One thing differs: the old loop's first `grad(r)` came after the caller's `score(r)`; `Rescore` scores first too, so the warm-start order is the same.

- [ ] **Step 2: relax.py.** `Relaxation.score`:

```python
    def score(self, s: search.SearchState, removed: np.ndarray) -> search.Score:
        """The eps world as a search.World: J of the clearing (no P here)."""
        return search.Score(self.value(removed.astype(float)), np.nan)
```
Incumbent.polish: the lines from `J0 = self.J` through the `exchange(...)` call become

```python
        J0 = self.J
        c = self.rel.c
        c.removed[:] = self.r > 0
        s = search.SearchState(c, self.rel.power, search.Score(self.J, np.nan), self.scorer, [],
                               0, np.isin(np.arange(c.n), movable))
        moves = search.Until(
            search.Do(search.exchange_round(self.scorer, budget=self.budget, width=width,
                                            pairs=pairs, pool=PAIR_POOL, tries=tries)),
            search.Spent(tries))(s, search.Silent())
        self.J, self.r, used = s.score.J, c.removed.astype(float), s.scorings
        c.removed[:] = False                    # relax.one keeps its clearings in x, not here
```
(the incumbent's J is already in its scorer's world, so no Rescore: the state starts there.)
Delete `exchange`, `_singles`, `_pairs`; keep `PAIR_POOL` (the edge's constant, passed in).

- [ ] **Step 3: polish_greedy.py.** After the Task 3 lines, the `rel`/`scorer`/`greedy_perm`/`exchange(...)` lines become

```python
    c.removed[:] = r > 0
    rel = relax.Relaxation(c, power, rtol=RTOL)
    scorer = relax.Relaxation(c, power, q=1.0, rtol=RTOL, eps=SCORE_EPS, params=c.p)
    greedy_perm = rel.perm(rel.exact(r))
    s = search.SearchState.start(c, power, search.UNSCORED, None)  # the polish's own count
    moves = search.polish(scorer, budget=budget, width=width, pairs=pairs, pool=relax.PAIR_POOL,
                          tries=tries)(s, search.Silent())
    rp, J, used = c.removed.astype(float), s.score.J, s.scorings
```
`polish` returns the moves `Seq` counted (Rescore counts 0). The polish starts a fresh SearchState: the greedy's scorings (M4, S0.01catw4 score candidates) must not count against its tries, as the old `exchange` began its count at 0 (relax.py:329). Update polish_greedy's docstring (lines 3 and 11) where it names the picker classes or `exchange`.

- [ ] **Step 4: Smoke, lint, commit locally** (as Task 3 Step 5, message "research: roadless -- exchange as a Round: Gradient, Swaps, Top, IfBetter, Rescore (search.polish); Relaxation is a World; polish_greedy and Incumbent.polish on the engine; relax.exchange deleted").

- [ ] **Step 5: Agreement, polish and SIMP:** `agree.py run t4 <commit> polish SIMP`; for t4's greedy and GP rows, copy them from t3's tree is NOT needed: compare only polish and SIMP keys (`compare base1 t4` reports the others as "only in base1": ignore those lines). Expected: polish and SIMP identical.

- [ ] **Step 6: The clearing left empty (review focus 2).** A full SIMP run (~40 min): run it in the background.

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
Expected: `clearing empty after one`. Then push.

---

### Task 5: The restore round and grow-then-prune

**Files:**
- Modify: `research/roadless/search.py`
- Modify: `research/roadless/clear.py` (delete `_exact`)

**Interfaces:**
- Consumes: Tasks 2-3.
- Produces: `Order`, `RestoreTension`, `Screen(world=, m=)`, `ToLevel(step)`, `restore_round(*, step, shortlist, screen) -> Round[Order]`, `PruneRows(block_id, c, power, J0, P0, d_max, t0)` with `.frame()`, `Rows` (Protocol: a Record with `frame()`), `Built(part, rows)` (NamedTuple), `GrowPrune(*, grow, picker, restore, shortlist, screen_eps)` (frozen, kw_only) with `build(block_id, c, power, J0, P0, d_max, t0, rnd, screen) -> Built`, `prune_block(b, plan, rnd, screen_of, power, along, device, d_max) -> pd.DataFrame`.

- [ ] **Step 1: The restore round** (restore_batch's body, split):

```python
class Order(NamedTuple):
    order: np.ndarray           # the cleared buildings, least loss of J per unit of
                                # population first


@dataclass(frozen=True)
class RestoreTension:
    """The restore tension (Clearing.tension(restore=True): the finite difference toward
    everything put back), per unit of population."""

    def rank(self, s: SearchState) -> Order:
        c = s.c
        on = np.flatnonzero(c.removed)
        loss = -c.tension(s.power, restore=True).g[on] / c.cost[on]
        return Order(on[np.argsort(loss, kind="stable")])


@dataclass(frozen=True, kw_only=True)
class Screen:
    """The first-order ranking's top max(m, 2k) re-ranked by each one's own loss in `world`
    (the first-order ranking alone, rank correlation 0.6 -- 0.7 with the exact loss, lost 0.22
    Lens A on a small block where exact backward elimination beat the greedy). Its solves
    count in s.scorings."""
    world: Valuer
    m: int

    def refine(self, s: SearchState, r: Order, first_k: int) -> Order:
        c = s.c
        cand = r.order[:max(self.m, 2 * first_k)]
        x = c.removed.astype(float)
        now = self.world.value(x)
        own = np.empty(len(cand))
        for i, j in enumerate(cand):
            x[j] = 0.0
            own[i] = (self.world.value(x) - now) / c.cost[j]
            x[j] = 1.0
        s.scorings += len(cand) + 1
        return Order(np.concatenate([cand[np.argsort(own, kind="stable")], r.order[len(cand):]]))


@dataclass(frozen=True)
class ToLevel:
    """Put back, in ranking order, until the cleared share falls to the next multiple of
    `step` below (at most one building past it)."""
    step: float

    def size(self, s: SearchState, r: Order) -> int:
        c = s.c
        D = float(c.cost[c.removed].sum())
        target = (np.ceil(D / self.step - 1e-9) - 1) * self.step
        return int(np.searchsorted(np.cumsum(c.cost[r.order]), D - target - 1e-12)) + 1

    def tiers(self, s: SearchState, r: Order) -> list[Callable[[], Tier]]:
        batch = [int(j) for j in r.order[:self.size(s, r)]]
        return [lambda: Batches([Move([], batch)], np.zeros(1))]


def restore_round(*, step: float, shortlist: int, screen: Valuer) -> Round[Order]:
    return Round(name=f"r{step:g}" + (f"m{shortlist}" if shortlist else ""),
                 rank=RestoreTension(),
                 refine=Screen(world=screen, m=shortlist) if shortlist else None,
                 build=ToLevel(step), evaluate=All(EXACT), accept=Only())
```
Check against restore_batch (say so in the report): `D` from `c.cost[c.removed].sum()` equals the old `c.cost[on].sum()` (same elements, same order); `k` from the first-order order feeds the shortlist, then `k` is recomputed on the refined order, as before; the screen's `now` and each `value` in the same order.

- [ ] **Step 2: PruneRows and the Rows protocol.**

```python
class Rows(Record, Protocol):
    def frame(self) -> pd.DataFrame: ...


class Built(NamedTuple):
    part: Part
    rows: Rows


class PruneState(NamedTuple):
    D: float
    score: Score
    removed: np.ndarray


class PruneRows:
    """Grow-then-prune's rows: every state at or below d_max with something cleared, scored
    exactly; written in increasing D in the greedy's format, `cleared` in index order (the
    states are nested), `t` the time when written."""

    def __init__(self, block_id: str, c: Clearing, power: float, J0: float, P0: float,
                 d_max: float, t0: float):
        self.block_id, self.n, self.power, self.J0, self.P0, self.d_max, self.t0 = (
            block_id, c.n, power, J0, P0, d_max, t0)
        self.states: list[PruneState] = []

    def wants(self, s: SearchState) -> bool:
        return s.D <= self.d_max + 1e-12 and bool(s.c.removed.any())

    def on_step(self, s: SearchState, o: Outcome) -> None:
        if self.wants(s):
            self.states.append(PruneState(s.D, s.score, s.c.removed.copy()))

    def frame(self) -> pd.DataFrame:
        rows = [dict(block=self.block_id, n=self.n, step=0, D=0.0, perm=0.0, perm1=0.0,
                     cleared=[], P0=self.P0, t=0.0)]
        prev = np.zeros(self.n, dtype=bool)
        for st in reversed(self.states):
            rows.append(dict(block=self.block_id, n=self.n, step=len(rows), D=st.D,
                             perm=1 - (st.score.J / self.J0) ** (1 / self.power),
                             perm1=1 - st.score.P / self.P0,
                             cleared=np.flatnonzero(st.removed & ~prev).tolist(), P0=self.P0,
                             t=time.time() - self.t0))
            prev = st.removed
        return pd.DataFrame(rows)
```

- [ ] **Step 3: GrowPrune, plan_of, prune_block, main.**
  - GrowPrune becomes `@dataclass(frozen=True, kw_only=True)` with fields `grow: float`, `picker: str`, `restore: float`, `shortlist: int`, `screen_eps: float`; `name` unchanged; `plan_of` builds it by keyword with `screen_eps=EPS_SCREEN`; and

```python
    def build(self, block_id: str, c: Clearing, power: float, J0: float, P0: float,
              d_max: float, t0: float, rnd: Round, screen: Valuer) -> Built:
        rec = PruneRows(block_id, c, power, J0, P0, d_max, t0)
        return Built(Seq((With(Silent(), greedy(rnd, self.grow * d_max)),
                          Until(Do(restore_round(step=self.restore, shortlist=self.shortlist,
                                                 screen=screen)), Emptied()))), rec)
```
  - grow_prune and restore_batch are replaced by

```python
def prune_block(b, plan: GrowPrune, rnd: Round, screen_of: Callable[[Clearing, float], Valuer],
                power: float, along: str, device: str, d_max: float) -> pd.DataFrame:
    p = lifted.Params(3.0, 8, along=lifted.along_of(along, lifted.scans_of(device)),
                      solver=lifted.solver_of(device))
    mesh = lifted.UniformMesh(0.5, offset=lifted.OFFSET)
    c = Clearing(b, mesh, p, population=common.POPULATIONS["area"])
    J0, P0 = c.sc.J(c.sc.u0, power), c.sc.P0
    t0 = time.time()
    built = plan.build(b.block_id, c, power, J0, P0, d_max, t0, rnd, screen_of(c, power))
    built.part(SearchState.start(c, power, Score(J0, P0), EXACT), built.rows)
    rows = built.rows.frame()
    print(f"{time.strftime('%H:%M:%S')} {b.block_id} n={c.n} {plan.name}: "
          f"{len(rows) - 1} states; perm' {rows.perm.iloc[-1]:.3f} at D {rows.D.iloc[-1]:.3f}  "
          f"{time.time() - t0:.0f}s", flush=True)
    return rows
```
  - `main` (the edge) resolves the round once, `rnd = greedy_round(plan.picker, sweep_of(device))`, and binds the screen: `import relax  # the edge binds the eps world; search.py's library code does not import relax` then `screen_of = lambda c, power: relax.Relaxation(c, power, q=1.0, rtol=RTOL_SCORE, eps=plan.screen_eps)` (a `def` if ruff's E731 is selected; it is not in this plan's gate), and calls `prune_block(b, plan, rnd, screen_of, power, along, device, d_max)`.
    The old screen was built after the grow phase; building it before changes nothing it computes (no solve until its first value), which the agreement check confirms.
    The old "grown to D" print goes (say so in the commit message).
  - clear.py: delete `_exact` (no caller left: grep).

- [ ] **Step 4: Smoke, lint, commit locally** (message "research: roadless -- grow-then-prune on the engine: RestoreTension, Screen, ToLevel (restore_round), PruneRows, GrowPrune.build; grow_prune, restore_batch and clear._exact deleted").

- [ ] **Step 5: Agreement, GP:** `agree.py run t5 <commit> GP`, `compare base1 t5` (GP keys). Expected: identical but `t`. Then push.

---

### Task 6: What the check covered

**Files:**
- Modify: `.superpowers/sdd/2026-10-08-add-remove-search/agree.py` (add `coverage`)

- [ ] **Step 1:** Add `coverage <label> <commit>`: a fresh worktree of HEAD, and for each block a Python process that imports the script's module, wraps the functions below on that module object, and calls the script's `main(...)` with its argv (for relax: `relax.main("one", [b], 2.0, "cpu", relax.plan_of(SIMP), "uni", 1, 0.05)`; for polish_greedy: `polish_greedy.main(b, 2.0, "cpu", "S0.01cat", 64, 8, "uni", 0.05, common.mesh_of("0.5"), True)`; for search: `search.main([b], search.plan_of(GP), 2.0, "cpu", "uni", 0.15)`; for the greedy, in-process (clear.run forks a pool even with one worker, so counters there would rise in the child): `rnd = search.greedy_round("S0.01catw4", clear.sweep_of("cpu")); d = clear.rows_dir(rnd.name, 0.5, "area", 2.0, "uni"); d.mkdir(exist_ok=True); clear._CFG = dict(picker=rnd, h=0.5, d_max=0.15, dir=d, pop="area", power=2.0, along=lifted.along_of("uni"), solver=lifted.solver_of("cpu"), search=None); clear._BLOCKS = common.build_blocks([b]); assert clear._one(0) is None` (`_one` returns the block id on failure)), writing rows into the coverage tree only. Counters, printed per block:
  - `search._pairs` calls (the pair tier evaluated);
  - in `search.Diverse._batches`, rounds where the distinct batches number fewer than `width` (wrap the method: compare `len(result)` with `self.width`);
  - `relax.cut_to_budget` calls whose result skips a building of the order that a later taken one follows (compare the result with the order's prefix);
  - in `search.Screen.refine`, rounds where the refined order's first `first_k` differ as a set from the first-order's.
  (relax.main with workers 1 maps `_run` in-process, so module patches apply.)
- [ ] **Step 2:** Run it. A counter at zero on all three blocks: find a block where it fires (19537, 247 buildings, for the pair tier; check NOTES' small blocks), add it to IDS temporarily, `run base3 <BASE> <preset>` and `run t6 <HEAD> <preset>` for that preset only, `compare base3 t6`.
- [ ] **Step 3:** Ledger: the counts per block, any block added, its comparison.

---

### Task 7: GPU memory, old against new

- [ ] **Step 1:** GPU idle; run memprobe_greedy at BASE (in `trees/base1`) twice, then at HEAD once, one after another (never together), on `ZAF.9.3.1_1_20543` under the translucent search, each in the background:

```bash
cd <tree> && CUDA_PATH=/usr PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 ~/src/reblock/.venv/bin/python -u research/roadless/memprobe_greedy.py ZAF.9.3.1_1_20543 uni@ss100k0.5n2r30 0.02 default > memprobe_<label>.log 2>&1
```
Compare the largest "peak" / pool figure memprobe itself prints per step across the three logs (its own report, not nvidia-smi samples).
Expected: HEAD within the two BASE runs' spread, or within 2% of their larger.
More: find what the engine holds across a round (a thunk capturing the ranking, `s` holding a Tension) and fix it before Task 8.
- [ ] **Step 2:** Ledger: the three peaks.

---

### Task 8: Floating search

**Files:**
- Modify: `research/roadless/search.py`

**Interfaces:**
- Consumes: Tasks 2, 3, 5.
- Produces: `Level` (NamedTuple: D, score, removed), `Archive(step)` with `level(D)`, `offer(D, score, removed)`, `beats(D, J, removed)`, `best`, `scored` (the conditional adds' scorings); `ArchiveSpent(archive, cap)` (a Stop); `BeatsArchive(archive=, d_max=)`; `FloatRows(block_id, c, power, J0, P0, d_max, archive, t0)` with `.frame()`; `FloatPrune(*, grow, picker, restore, shortlist, cap, screen_eps)` with GrowPrune's `build` signature; `SearchPlan` (Protocol: `name`, `picker`, `screen_eps`, `build`), taken by prune_block and main; `plan_of` returning either.

- [ ] **Step 1:**

```python
class Level(NamedTuple):
    D: float
    score: Score
    removed: np.ndarray


class Archive:
    """The best exact state per budget level (level = ceil(D / step)): floating search's
    comparison (Pudil) and its rows; `scored` counts the conditional adds' scorings (the cap's:
    the restore screen's solves are not its)."""

    def __init__(self, step: float):
        self.step = step
        self.best: dict[int, Level] = {}
        self.scored = 0

    def level(self, D: float) -> int:
        return int(np.ceil(D / self.step - 1e-9))

    def offer(self, D: float, score: Score, removed: np.ndarray) -> None:
        L = self.level(D)
        if L not in self.best or score.J < self.best[L].score.J:
            self.best[L] = Level(D, score, removed.copy())

    def beats(self, D: float, J: float, removed: np.ndarray) -> bool:
        """Strictly better than the level's best, and not the same clearing (an archived state
        re-added scores the same J on the CPU; on the GPU only to rounding)."""
        lv = self.best.get(self.level(D))
        return lv is None or (J < lv.score.J and not np.array_equal(removed, lv.removed))


@dataclass(frozen=True)
class ArchiveSpent:
    archive: Archive
    cap: int

    def done(self, s: SearchState) -> bool:
        return self.archive.scored >= self.cap


@dataclass(frozen=True, kw_only=True)
class BeatsArchive:
    """Floating search's conditional inclusion: the lowest-J candidate, kept only if its D is
    within d_max and its J beats the archive's best at its level; every other scored candidate
    within d_max is offered to the archive (the kept one reaches it through FloatRows)."""
    archive: Archive
    d_max: float

    @property
    def needs_score(self) -> bool:
        return True

    def accept(self, s: SearchState, cands: list[Candidate]) -> Candidate | None:
        self.archive.scored += len(cands)
        if not cands:
            return None
        best = min(cands, key=lambda cand: cand.score.J)    # the first lowest, as Spread's
        after_ = {id(cand): after(s.c.removed, cand.move) for cand in cands}
        D_of = {k: float(s.c.cost[r].sum()) for k, r in after_.items()}   # as s.D will be
        r, D = after_[id(best)], D_of[id(best)]
        kept = D <= self.d_max + 1e-12 and self.archive.beats(D, best.score.J, r)
        for cand in cands:
            if (cand is not best or not kept) and D_of[id(cand)] <= self.d_max + 1e-12:
                self.archive.offer(D_of[id(cand)], cand.score, after_[id(cand)])
        return best if kept else None


class FloatRows:
    """Floating search's rows: the archive's best per level, in increasing D, each with its full
    clearing (`clearing`, index order: the states are not nested, so no `cleared` deltas and
    cleared_through does not read them), after a D 0 row. Every exact state at or below d_max
    is offered to the archive."""

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
            lv = self.archive.best[L]
            rows.append(dict(block=self.block_id, n=self.n, D=lv.D,
                             perm=1 - (lv.score.J / self.J0) ** (1 / self.power),
                             perm1=1 - lv.score.P / self.P0,
                             clearing=np.flatnonzero(lv.removed).tolist(), P0=self.P0,
                             t=time.time() - self.t0))
        return pd.DataFrame(rows)
```

- [ ] **Step 2: FloatPrune and SearchPlan.** FloatPrune: `@dataclass(frozen=True, kw_only=True)`, GrowPrune's fields plus `cap: int`; name `FL{grow:g}x{picker}r{restore:g}[m{shortlist}]c{cap}`;

```python
    def build(self, block_id: str, c: Clearing, power: float, J0: float, P0: float,
              d_max: float, t0: float, rnd: Round, screen: Valuer) -> Built:
        archive = Archive(self.restore)
        rec = FloatRows(block_id, c, power, J0, P0, d_max, archive, t0)
        add = dataclasses.replace(rnd, name=f"{rnd.name}|archive", evaluate=All(EXACT),
                                  accept=BeatsArchive(archive=archive, d_max=d_max))
        restore = restore_round(step=self.restore, shortlist=self.shortlist, screen=screen)
        stop = AnyOf((ArchiveSpent(archive, self.cap), Above(d_max)))
        return Built(Seq((With(Silent(), greedy(rnd, self.grow * d_max)),
                          Until(Seq((Do(restore), Until(Do(add), stop))), Emptied()))), rec)
```
(the conditional add is the greedy's round with its acceptor swapped: the table's "new row", in code.)
`class SearchPlan(Protocol)`: `name` (property), `picker: str`, `screen_eps: float`, `build(...)` as above; prune_block and main take a `SearchPlan`.
plan_of: `GP...` as now; `re.fullmatch(r"FL([0-9.]+)x(\S+?)r([0-9.]+)(?:m(\d+))?c(\d+)", spec)` to FloatPrune; else raise.
`Above(d_max)` stops conditional adds while D is past d_max (no tension spent there); `BeatsArchive` refuses a move past d_max (a level holding d_max but above it). `import dataclasses` in search.py for `replace`.

- [ ] **Step 3: Termination check, CPU, 19421 (in the background, ~1.5 h).** Temporary instrumentation, not committed: in `FloatRows.on_step`, before `offer`,

```python
        if o.cleared:                       # a kept conditional add (restores clear nothing)
            L = self.archive.level(s.D)
            prev = self.archive.best.get(L)
            assert s.world is EXACT and s.D <= self.d_max + 1e-12, (s.D, s.world)
            assert prev is None or s.score.J < prev.score.J, (L, s.score.J, prev.score.J)
            KEPT.append((L, float(s.score.J), np.packbits(s.c.removed).tobytes()))
```
(`KEPT: list = []` at module level; the grow phase runs under `Silent`, so `o.cleared` here is only ever a conditional add), and at the top of `BeatsArchive.accept`: `assert s.D <= self.d_max + 1e-12` (Above stopped add rounds above d_max).
In the run below, after `prune_block` returns, print `len(KEPT)`, assert `KEPT` is not empty and its clearings are distinct, and print `archive.scored`.

```bash
cd ~/src/reblock && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
REBLOCK_BLOCK_BANK=.superpowers/sdd/2026-10-08-add-remove-search/bank.pkl PYTHONPATH=. \
uv run python research/roadless/search.py ZAF.9.3.1_1_19421 FL3xS0.01catr0.005m8c100 2 cpu uni 0.15
```
Expected: finishes; at least one kept add; rows in `research/roadless/clear_rows_FL3xS0.01catr0.005m8c100_h0.5_area_p2/` with D increasing.
Compare with the CPU GP rows of `trees/t5` for 19421 (not the rows on disk, which may be the GPU's), mapping GP's D through the same `Archive.level`: FL at least GP's at every level at or above the level the first kept add started from (the archive held the prune's own states there); below it, report only.
Remove the instrumentation before Step 4.

- [ ] **Step 4: Smoke, lint, commit, push** ("floating search: backward floating on grow-then-prune (FL...c<cap>): the greedy's round with BeatsArchive as its acceptor after each restore; FloatRows with full clearings").

- [ ] **Step 5: First run.** 19421, 19510, 19537 on the CPU locally (in the background, one process each, the four thread variables at 1, a bank written for them), and 9712, 22422 on the cluster:

```bash
cd ~/src/reblock && uv run python research/roadless/cluster.py submit blocks fl-5 --mesh 0.5 ZAF.9.5.4_1_9712,ZAF.9.3.1_1_22422 -- research/roadless/search.py {id} FL3xS0.01catr0.005m8c200 2 gpu uni 0.15
uv run python research/roadless/cluster.py wait fl-5 --timeout 20h      # in the background
uv run python research/roadless/cluster.py sync fl-5
```
Compare at D 0.05, 0.10, 0.15 with GP3xS0.01catr0.005m8 (rows on disk for all five) and the greedy S0.01cat.

---

### Task 9: Record and clean up (after fl-5 is synced)

- [ ] **Step 1:** NOTES.md: a section "The add/remove search as one Strategy (owner 2026-10-08: prioritized)": the Round table in three sentences, the agreement result (identical, or each deviation and its explanation), the GPU memory figures, floating search's first numbers against GP and the greedy.
- [ ] **Step 2:** BACKLOG.md: "Add/remove subset search" marked built; floating search's result; next: the substitute-aware restore (ToLevel built with Diverse's discount) and screened adds (an add round with a Screen refiner), each one Round.
- [ ] **Step 3:** `git worktree remove` each tree under the workspace, `git worktree prune`.
- [ ] **Step 4:** Commit and push.
