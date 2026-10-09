"""Add/remove subset search (plus-l take-away-r; wiki pages/methods/plus-l-take-away-r.md): the
greedy's add moves and first-order restore moves (a cleared building put back) under a
schedule. First schedule, grow then prune (the mycooc vocabulary recipe): the greedy clears up
to `grow` x d_max, then each round restores the cleared buildings whose clearing is worth least
per unit of population (Clearing.tension(restore=True): the greedy's finite-difference tension
toward everything put back; the derivative at the 0/1 clearing understates it, measured: Lens A
0.38 / 0.37 vs the greedy's 0.60 / 0.53 on two small blocks), re-solving between rounds, down
to nothing. Every state at or below d_max is scored exactly; restoring only ever shrinks the set,
so the states are nested and the rows come out in the greedy's format (every budget in one
run). A gate is judged with its companions cleared, the greedy's blind spot; the risk is
substitutes (each cheap to restore alone) restored in one batch, hence small batches.

Second schedule, floating search (backward floating, Pudil): grow-then-prune with conditional
adds after each restore round, the greedy's own round kept only if its exact J beats the best
archived at its budget level (the restore step's lattice) and its D is within d_max, until a
cap on their scorings; the fix for the prune's collapse on gated blocks through substitutes.
Its rows are the archive's best per level, each with its full clearing (not nested).

    CUDA_PATH=/usr PYTHONPATH=. uv run python research/roadless/search.py <ids,> <spec> <p> <cpu|gpu> [along] [d_max]

spec: GP<grow>x<greedy picker>r<restore share per round>[m<shortlist>], e.g.
GP3xS0.01catr0.005m8; FL<same>c<cap on the conditional adds' scorings>, e.g.
FL3xS0.01catr0.005m8c100.
"""
from __future__ import annotations

import dataclasses
import os
import re
import sys
import time
from pathlib import Path
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, NamedTuple, Protocol, TypeVar

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402
import clear  # noqa: E402
from clear import RTOL_SCORE, Clearing, rows_dir, sweep_of  # noqa: E402

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


@dataclass(frozen=True, kw_only=True)
class IndexMoves:
    """A tier of small moves as a (4, M) array: rows 0-1 buildings added, 2-3 restored, -1
    none (exchange's ~10^6 swaps, materialised only for the few scored)."""
    idx: np.ndarray
    est: np.ndarray

    def move(self, i: int) -> Move:
        return Move(add=[int(x) for x in self.idx[:2, i] if x >= 0],
                    restore=[int(x) for x in self.idx[2:, i] if x >= 0])


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
    def start(cls, c: Clearing, *, power: float, score: Score, world: World | None
              ) -> SearchState:
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
        if s.world is not EXACT:
            raise ValueError("gain per cost needs the state scored exactly")
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
        return [lambda: Batches([Move(add=[j], restore=[]) for j in top], est)]


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
        return [lambda: Batches([Move(add=taken, restore=[])], np.zeros(1))]


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
        return [Move(add=[cand[i] for i in b], restore=[]) for b in batches.values()]

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


class GreedyRows:
    """The greedy's rows (clear.greedy_block's format): every state scored exactly; per step D,
    perm (J_power), perm1 (P) and the step's buildings in pick order, after a step-0 row."""

    def __init__(self, *, block_id: str, c: Clearing, power: float, J0: float, P0: float,
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


def _singles(g, ins, outs, cost, left):
    """Add one (budget left permitting) or swap one in for one out: moves as rows (added,
    added, closed, closed; -1 for none) and their linearized changes."""
    ia, ib = np.meshgrid(ins, outs, indexing="ij")
    none_s, none_p = np.full(len(ins), -1), np.full(ia.size, -1)
    mv = np.concatenate([np.stack([ins, none_s, none_s, none_s]),
                         np.stack([ia.ravel(), none_p, ib.ravel(), none_p])], axis=1)
    est = np.concatenate([np.where(cost[ins] <= left, g[ins], np.inf),
                          np.where(cost[ia] <= left + cost[ib], g[ia] - g[ib], np.inf).ravel()])
    return IndexMoves(idx=mv, est=est)


def _pairs(g, ins, outs, cost, left, pool: int):
    """Two in for one out and one in for two out, among the `pool` buildings most promising
    to add (most negative gradient) and cheapest to close."""
    pin = ins[np.argsort(g[ins], kind="stable")[:pool]]
    pout = outs[np.argsort(-g[outs], kind="stable")[:pool]]
    i, j = np.triu_indices(len(pin), 1)
    a1, a2, b = np.repeat(pin[i], len(pout)), np.repeat(pin[j], len(pout)), np.tile(pout, len(i))
    k, m = np.triu_indices(len(pout), 1)
    a, b1, b2 = np.repeat(pin, len(k)), np.tile(pout[k], len(pin)), np.tile(pout[m], len(pin))
    mv = np.concatenate([np.stack([a1, a2, b, np.full(len(b), -1)]),
                         np.stack([a, np.full(len(a), -1), b1, b2])], axis=1)
    est = np.concatenate([
        np.where(cost[a1] + cost[a2] <= left + cost[b], g[a1] + g[a2] - g[b], np.inf),
        np.where(cost[a] <= left + cost[b1] + cost[b2], g[a] - g[b1] - g[b2], np.inf)])
    return IndexMoves(idx=mv, est=est)


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
        return Grad(g=self.world.value_sgrad(r)[1], r=r)


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


EPS_SCREEN = 1e-6               # the eps world the shortlist is scored in (the real one to ~1e-3)


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
        return [lambda: Batches([Move(add=[], restore=batch)], np.zeros(1))]


def restore_round(*, step: float, shortlist: int, screen: Valuer) -> Round[Order]:
    return Round(name=f"r{step:g}" + (f"m{shortlist}" if shortlist else ""),
                 rank=RestoreTension(),
                 refine=Screen(world=screen, m=shortlist) if shortlist else None,
                 build=ToLevel(step), evaluate=All(EXACT), accept=Only())


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

    def __init__(self, *, block_id: str, c: Clearing, power: float, J0: float, P0: float,
                 d_max: float, t0: float):
        self.block_id, self.n, self.power, self.J0, self.P0, self.d_max, self.t0 = (
            block_id, c.n, power, J0, P0, d_max, t0)
        self.states: list[PruneState] = []

    def wants(self, s: SearchState) -> bool:
        return s.D <= self.d_max + 1e-12 and bool(s.c.removed.any())

    def on_step(self, s: SearchState, o: Outcome) -> None:
        if self.wants(s):
            self.states.append(PruneState(D=s.D, score=s.score, removed=s.c.removed.copy()))

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


@dataclass(frozen=True, kw_only=True)
class GrowPrune:
    grow: float                 # the greedy clears up to grow x d_max
    picker: str                 # the greedy's picker
    restore: float              # population share restored per round
    shortlist: int              # first-order shortlist rescored per round (0: first order only)
    screen_eps: float           # the eps world the shortlist is scored in

    @property
    def name(self) -> str:
        return (f"GP{self.grow:g}x{self.picker}r{self.restore:g}"
                + (f"m{self.shortlist}" if self.shortlist else ""))

    def build(self, *, block_id: str, c: Clearing, power: float, J0: float, P0: float,
              d_max: float, t0: float, rnd: Round, screen: Valuer) -> Built:
        rec = PruneRows(block_id=block_id, c=c, power=power, J0=J0, P0=P0, d_max=d_max, t0=t0)
        return Built(Seq((With(Silent(), greedy(rnd, self.grow * d_max)),
                          Until(Do(restore_round(step=self.restore, shortlist=self.shortlist,
                                                 screen=screen)), Emptied()))), rec)


class Level(NamedTuple):
    D: float
    score: Score
    removed: np.ndarray


class Archive:
    """The best exact state per budget level (level = ceil(D / step)): floating search's
    comparison (Pudil) and its rows; `scored` counts the conditional adds' scorings (the cap's:
    the restore screen's solves are not its)."""

    def __init__(self, *, step: float):
        self.step = step
        self.best: dict[int, Level] = {}
        self.scored = 0

    def level(self, D: float) -> int:
        return int(np.ceil(D / self.step - 1e-9))

    def offer(self, D: float, score: Score, removed: np.ndarray) -> None:
        L = self.level(D)
        if L not in self.best or score.J < self.best[L].score.J:
            self.best[L] = Level(D=D, score=score, removed=removed.copy())

    def beats(self, D: float, J: float, removed: np.ndarray) -> bool:
        """Strictly better than the level's best, and not the same clearing (an archived state
        re-added scores the same J on the CPU; on the GPU only to rounding)."""
        lv = self.best.get(self.level(D))
        return lv is None or (J < lv.score.J and not np.array_equal(removed, lv.removed))


@dataclass(frozen=True, kw_only=True)
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

    def __init__(self, *, block_id: str, c: Clearing, power: float, J0: float, P0: float,
                 d_max: float, archive: Archive, t0: float):
        self.block_id, self.n, self.power, self.J0, self.P0, self.d_max, self.t0 = (
            block_id, c.n, power, J0, P0, d_max, t0)
        self.archive = archive

    def wants(self, s: SearchState) -> bool:
        return s.D <= self.d_max + 1e-12 and bool(s.c.removed.any())

    def on_step(self, s: SearchState, o: Outcome) -> None:
        if self.wants(s):               # apply scored it exactly
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


@dataclass(frozen=True, kw_only=True)
class FloatPrune:
    grow: float                 # the greedy clears up to grow x d_max
    picker: str                 # the greedy's picker
    restore: float              # population share restored per round (and the archive's step)
    shortlist: int              # first-order shortlist rescored per round (0: first order only)
    cap: int                    # the conditional adds' scorings
    screen_eps: float           # the eps world the shortlist is scored in

    @property
    def name(self) -> str:
        return (f"FL{self.grow:g}x{self.picker}r{self.restore:g}"
                + (f"m{self.shortlist}" if self.shortlist else "") + f"c{self.cap}")

    def build(self, *, block_id: str, c: Clearing, power: float, J0: float, P0: float,
              d_max: float, t0: float, rnd: Round, screen: Valuer) -> Built:
        archive = Archive(step=self.restore)
        rec = FloatRows(block_id=block_id, c=c, power=power, J0=J0, P0=P0, d_max=d_max,
                        archive=archive, t0=t0)
        # the conditional add is the greedy's round with its acceptor swapped
        add = dataclasses.replace(rnd, name=f"{rnd.name}|archive", evaluate=All(EXACT),
                                  accept=BeatsArchive(archive=archive, d_max=d_max))
        restore = restore_round(step=self.restore, shortlist=self.shortlist, screen=screen)
        stop = AnyOf((ArchiveSpent(archive=archive, cap=self.cap), Above(d_max)))
        return Built(Seq((With(Silent(), greedy(rnd, self.grow * d_max)),
                          Until(Seq((Do(restore), Until(Do(add), stop))), Emptied()))), rec)


class SearchPlan(Protocol):
    """A schedule's preset (GrowPrune, FloatPrune): its name (the rows' directory), the greedy
    picker its add round is built from, the eps world of its restore screen, and its parts.
    Read-only members: a frozen dataclass's fields are not settable variables."""

    @property
    def name(self) -> str: ...

    @property
    def picker(self) -> str: ...

    @property
    def screen_eps(self) -> float: ...

    def build(self, *, block_id: str, c: Clearing, power: float, J0: float, P0: float,
              d_max: float, t0: float, rnd: Round, screen: Valuer) -> Built: ...


def plan_of(spec: str) -> SearchPlan:
    if m := re.fullmatch(r"GP([0-9.]+)x(\S+?)r([0-9.]+)(?:m(\d+))?", spec):
        return GrowPrune(grow=float(m.group(1)), picker=m.group(2), restore=float(m.group(3)),
                         shortlist=0 if m.group(4) is None else int(m.group(4)),
                         screen_eps=EPS_SCREEN)
    if m := re.fullmatch(r"FL([0-9.]+)x(\S+?)r([0-9.]+)(?:m(\d+))?c(\d+)", spec):
        return FloatPrune(grow=float(m.group(1)), picker=m.group(2), restore=float(m.group(3)),
                          shortlist=0 if m.group(4) is None else int(m.group(4)),
                          cap=int(m.group(5)), screen_eps=EPS_SCREEN)
    raise ValueError(f"unknown search {spec!r}")


def prune_block(b, plan: SearchPlan, *, rnd: Round,
                screen_of: Callable[[Clearing, float], Valuer], power: float,
                along: str, device: str, d_max: float) -> pd.DataFrame:
    p = lifted.Params(3.0, 8, along=lifted.along_of(along, lifted.scans_of(device)),
                      solver=lifted.solver_of(device))
    mesh = lifted.UniformMesh(0.5, offset=lifted.OFFSET)
    c = Clearing(b, mesh, p, population=common.POPULATIONS["area"])
    J0, P0 = c.sc.J(c.sc.u0, power), c.sc.P0
    t0 = time.time()
    built = plan.build(block_id=b.block_id, c=c, power=power, J0=J0, P0=P0, d_max=d_max,
                       t0=t0, rnd=rnd, screen=screen_of(c, power))
    built.part(SearchState.start(c, power=power, score=Score(J0, P0), world=EXACT), built.rows)
    rows = built.rows.frame()
    print(f"{time.strftime('%H:%M:%S')} {b.block_id} n={c.n} {plan.name}: "
          f"{len(rows) - 1} states; perm' {rows.perm.iloc[-1]:.3f} at D {rows.D.iloc[-1]:.3f}  "
          f"{time.time() - t0:.0f}s", flush=True)
    return rows


def main(ids: list[str], plan: SearchPlan, power: float, device: str, along: str,
         d_max: float) -> None:
    import relax  # the edge binds the eps world; search.py's library code does not import relax

    def screen_of(c: Clearing, pw: float) -> Valuer:
        return relax.Relaxation(c, pw, q=1.0, rtol=RTOL_SCORE, eps=plan.screen_eps)

    rnd = greedy_round(plan.picker, sweep_of(device))
    out = rows_dir(plan.name, 0.5, "area", power, along)
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for b in common.build_blocks(ids):
        f = out / f"{b.block_id}.parquet"
        if f.exists():
            continue
        try:
            rows = prune_block(b, plan, rnd=rnd, screen_of=screen_of, power=power, along=along,
                              device=device, d_max=d_max)
        except Exception as e:
            print(f"{b.block_id} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
            failed.append(b.block_id)
            continue
        tmp = f.with_suffix(f".{os.getpid()}.tmp")
        rows.to_parquet(tmp)
        os.replace(tmp, f)
    if failed:
        raise SystemExit(f"{len(failed)} blocks failed: {', '.join(failed)}")


if __name__ == "__main__":
    import search       # run as a script this file would be `__main__`, and relax's `search`
                        # a second copy of the engine (its own EXACT and classes)
    search.main(sys.argv[1].split(","), search.plan_of(sys.argv[2]), float(sys.argv[3]),
                sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else "uni",
                float(sys.argv[6]) if len(sys.argv) > 6 else 0.15)
