"""Pruning by epistemetry's decision layer: a replay against seqscreen.py (BACKLOG, "Pruning by
epistemetry's decision layer"; NOTES, "Pruning by epistemetry: a replay against seqscreen").

Does GeoffChurch/epistemetry's ValueOfMoreUnits on the finite-population bootstrap (pinned at
fc921f1, run through an ephemeral overlay: reblock does not depend on it) decide which variants to
keep better than seqscreen.py, on comparisons already run on every block? A design (`Plan`) is a
base and its variants on the blocks every one of them has, read through seqscreen's own reader
(simp_compare's specs): a unit is a block, an arm a variant's paired gain over the base in Lens A,
a unit's cost its variants' GPU-seconds (the base has run on every block). A block whose row of an
arm is a free-first run under metric B (the overwrites of 2026-10-11, NOTES "The stored clearings,
committed") is left out, so every pair is within one metric. Every policy runs on the same
orders, one at a time through epistemetry's `replay`: `random_orders` from ORDER_SEED, the
permutations seqscreen.main draws (its 20 first, then 80 more from the same generator):

- `SeqScreen`: seqscreen unchanged, as a Policy -- its stop (the value of more blocks averaged
  over the rate bracket), floor and veto, and its call over the bracket. An ask resolves at the
  owner's rate (seqscreen.call at that rate: the answer to its question).
- `Epi` at one rate: ValueOfMoreUnits with FinitePopulation; every horizon to the last block;
  seqscreen's cost model (`RatioToBase`); the floor MIN_BLOCKS; ValueOfInformation at no Monte
  Carlo margin (seqscreen has none); tau = rate x the extra GPU-seconds a block so far, rebuilt
  each step as seqscreen re-estimates it. Its target is a Strategy (`Aim`):
  - `ThreeWayAt`, the brief's: three-way at tau per arm (functional d_j / tau_j, margin 1). Its
    risk is a probability of a wrong class, so its price is seqscreen's (rate / FUTURE) over the
    Lens A a wrong call would cost at the means so far, mean_j |mu_j - tau_j|.
  - `LinearAt`: seqscreen's own loss as an epistemetry DrawTarget (`Regret`), priced as seqscreen
    prices it. Reported beside the brief's, not part of the decision.
  `Vetoed` adds the collapse veto (a block more than COLLAPSE below the base stops the screen with
  "keep"), running the inner policy's batches a block at a time so the veto sees every block.
- many arms: one `Epi` over every variant of a design (no veto: a collapse would drop one arm,
  which the layer cannot), against seqscreen once per variant on the same orders and blocks.

Judged at RATES' ends and log-midpoint (JUDGED) against seqscreen.call on all of the design's
blocks at the same tau; the realized loss per governed block is a wrong call's regret |mu - tau|
plus rate / FUTURE x the GPU-seconds spent (NOTES, "Stopping by the value of more blocks").

The decision rule, fixed before any run (the brief): adopt epistemetry for the consolidated pass's
pruning only if, for its candidate, (1) the median realized loss is below seqscreen's at every
rate tested and (2) it makes no more wrong calls in all. Candidates: pairwise, `ThreeWayAt` with
the veto, against `SeqScreen` (per rate, the sum over the pairwise designs of each one's median
over orders; wrong calls summed over designs, orders and rates); many-arm, `ThreeWayAt` over all
arms, against the sum of seqscreen's pairwise screens per order (the same statistics over the
many-arm designs).

The amendment (the coordinator, 2026-10-15, committed here before any adopt/keep/ask run; the
fixed-rate runs above, at fc921f1, are reported under the original rule, as a secondary table):
the pin moves to c1784af, and the primary comparison is epistemetry's AdoptKeepAsk over seqscreen's
own bracket (RATES' ends; a unit's gain and its extra GPU-seconds two columns of one ball), scored
with AsksApart (an ask is never wrong; a definite call is wrong where it differs from the whole
design's call over the bracket, an ask included). It runs twice: with no veto anywhere
(seqscreen's patched off inside the process, no file edited; the truth without it), and with the
veto everywhere (seqscreen's own, the wrapper around epistemetry's policy, the truth "keep" on a
collapse). Asks go to the owner, so they are a cost. The rule, amended: adopt epistemetry only
if, against seqscreen in the same veto setting, (1) its median realized loss is below seqscreen's
at every judged rate, (2) it makes no more wrong calls in all and (3) no more asks in all.
Realized loss at rate r: |mu - r x extra| where the call resolved at r (an ask answered at r from
the screen's own means, as seqscreen's question is) differs from the whole design's call at r,
plus r / FUTURE x the GPU-seconds spent. The statistics as before: per rate the sum over the
designs of each one's median over orders; wrong calls and asks summed over designs and orders.
Candidates: pairwise, AdoptKeepAsk in each veto setting (it must pass in both); many-arm,
AdoptKeepAsk over every arm (no veto: no many-arm design has a collapse) against seqscreen's
screens of each variant summed per order. Its price: seqscreen prices a GPU-second at the
bracket's mean rate over FUTURE, and AdoptKeepAsk's risk is a probability of another call, so
the price is that over the Lens A a wrong call costs at the means so far, at that rate. Reported
beside them: how often the likeliest call at a stop differs from the posterior mean's (seqscreen's
call on the same blocks).

    EPI='epistemetry @ git+https://github.com/GeoffChurch/epistemetry@c1784af'
    uv run --with "$EPI" python research/roadless/epireplay.py run <fixed|bracket>
    uv run --with "$EPI" python research/roadless/epireplay.py guards <none|floor|ignored|shuffled>
    uv run --with "$EPI" python research/roadless/epireplay.py report <fixed|bracket>

with OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS=1 (WORKERS processes).
"""
from __future__ import annotations

import contextlib
import functools
import io
import math
import re
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from epistemetry.design.decide import (ADOPT, ASK, KEEP, AdoptKeepAsk, DrawTarget, StandardErrors,
                                      ThreeWay, ValueOfInformation)
from epistemetry.design.replay import (AsksApart, Bootstrap, Design, Exact, Policy,
                                       ValueOfMoreUnits, random_orders, replay)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clear  # noqa: E402
import relax  # noqa: E402
import seqscreen as ss  # noqa: E402
import simp_compare  # noqa: E402

BUDGET = 0.05                                  # Lens A's D, as the screens ran
BRACKET = np.geomspace(*ss.RATES, ss.RATE_POINTS)   # seqscreen's stop and call
JUDGED = np.geomspace(*ss.RATES, 3)            # the bracket's ends and its log-midpoint
ROWS = HERE / "epireplay.parquet"              # git-ignored; `run fixed` regenerates it
ROWS_BRACKET = HERE / "epireplay_bracket.parquet"   # `run bracket`'s, gathered from its jobs'
JOBS_BRACKET = HERE / "epireplay_bracket_jobs"     # a parquet a job, kept: a rerun resumes
VETO = ss.COLLAPSE                             # the veto's; patched to inf where it is off
PRICE_RATE = float(BRACKET.mean())             # seqscreen prices its stop's GPU-seconds at it
CODE = {"keep": KEEP, "ask": ASK, "adopt": ADOPT}   # seqscreen's calls as AdoptKeepAsk's
WORKERS = 8
ORDERS = 100        # orders a design: seqscreen.main's ORDERS first, then more from its generator


class MonteCarlo(NamedTuple):
    """epistemetry's draws: outer draws of the next blocks, inner draws of the rest given them,
    batch-means blocks, and the draws a call is made from (seqscreen's value uses DRAWS paths)."""
    outer: int
    inner: int
    blocks: int
    draws: int


MC = MonteCarlo(outer=ss.DRAWS, inner=200, blocks=20, draws=ss.DRAWS)


# ---- designs --------------------------------------------------------------------------------

class Arm(NamedTuple):
    """One set of rows, as simp_compare's specs name it."""
    name: str
    kind: str      # simp_compare.LOADERS': simp, greedy, polished
    along: str
    what: str

    @property
    def spec(self) -> str:
        return f"{self.name}={self.kind},{self.along},{self.what}"


P = "fw0.q3.i10.t0.001.e0.0001.k1s1e-06"       # SIMP + the eps-scored incumbent
BASE = Arm("base", "simp", "uni", P)
R2 = Arm("r2", "simp", "uni", f"{P}.r2")
P64 = Arm("p64", "simp", "uni", f"{P}.p64w8")
X2 = Arm("x2", "simp", "uni", f"{P}.p256w8x2")
BOTH = Arm("both", "simp", "uni", f"{P}.r2l2.p64w8")
C1M4 = Arm("C1m4", "simp", "uni", f"{P}.p64w8.C1.m4")
P256 = Arm("p256", "simp", "uni", f"{P}.p256w8")
PPG = Arm("pPg", "simp", "uni", f"{P}.pP256w8.gS0.01cat")
GS = Arm("GS", "simp", "uni", f"{P}.p256w8x2.GS0.01cat")
G01 = Arm("g01", "greedy", "uni", "S0.01cat")
G005 = Arm("g005", "greedy", "uni", "S0.005cat")
CHEAP = Arm("cheap", "polished", "uni", "S0.01cat.P64w8")
CHEAPX2 = Arm("cheapx2", "polished", "uni", "S0.01cat.P256w8x2")


class Plan(NamedTuple):
    """A design: `base` and `variants` on the blocks of `which` that all of them have. `card`:
    where every arm's rows ran (NOTES), one kind of card a design."""
    name: str
    card: str
    which: str
    base: Arm
    variants: tuple[Arm, ...]


CLUSTER, LOCAL = "cluster V100 / RTX 8000", "local RTX 6000 Ada"
PAIRS = (    # the six held-out screens seqscreen was built on, then every other pair on one card
    Plan("p64/base", CLUSTER, "held", BASE, (P64,)),
    Plan("x2/base", CLUSTER, "held", BASE, (X2,)),
    Plan("both/base", CLUSTER, "held", BASE, (BOTH,)),
    Plan("both/p64", CLUSTER, "held", P64, (BOTH,)),
    Plan("r2/base", CLUSTER, "held", BASE, (R2,)),
    Plan("x2/p64", CLUSTER, "held", P64, (X2,)),
    Plan("pPg/p256", LOCAL, "held", P256, (PPG,)),
    Plan("GS/p256", LOCAL, "held", P256, (GS,)),
    Plan("cheapx2/cheap", LOCAL, "held", CHEAP, (CHEAPX2,)),
    Plan("p256/cheap", LOCAL, "held", CHEAP, (P256,)),
    Plan("g005/g01", LOCAL, "held", G01, (G005,)),
    Plan("cheap/g01", LOCAL, "held", G01, (CHEAP,)),
    Plan("C1m4/p64", CLUSTER, "tuning", P64, (C1M4,)),   # a collapse (30848), for the veto
)
MANY = (
    Plan("4/base", CLUSTER, "held", BASE, (R2, P64, X2, BOTH)),
    Plan("2/p64", CLUSTER, "held", P64, (X2, BOTH)),
    Plan("2/p256", LOCAL, "held", P256, (PPG, GS)),
    Plan("2/cheap", LOCAL, "held", CHEAP, (CHEAPX2, P256)),
    Plan("2/g01", LOCAL, "held", G01, (G005, CHEAP)),
)
PLANS = {p.name: p for p in PAIRS + MANY}      # the command line's names


def row_dir(arm: Arm) -> Path:
    """Where simp_compare's reader for `arm`'s kind finds its rows."""
    if arm.kind == "simp":
        return relax.rows_of(relax.plan_of(arm.what), arm.along, BUDGET)
    if arm.kind == "polished":
        return HERE / "polish_rows" / arm.along / arm.what / f"D{BUDGET:g}"
    if arm.kind == "greedy":
        return clear.rows_dir(arm.what, 0.5, "area", simp_compare.POWER, arm.along)
    raise ValueError(f"no rows for kind {arm.kind!r}")


def free_first(arm: Arm) -> set[str]:
    """The blocks whose row of `arm` is a free-first run's under metric B (it carries `enclosed`):
    the default's and the cheap preset's rows of 12 blocks, overwritten 2026-10-11 20:17."""
    return {pd.read_parquet(f, columns=["block"]).block.iloc[0]
            for f in sorted(row_dir(arm).glob("*.parquet"))
            if "enclosed" in pq.read_schema(f).names}


class Data(NamedTuple):
    plan: Plan
    blocks: list[str]
    d: np.ndarray       # (u, k) each variant's Lens A minus the base's
    t: np.ndarray       # (u, k) each variant's GPU-seconds
    ta: np.ndarray      # (u,) the base's
    dropped: list[str]  # blocks every arm has, left out for a free-first row

    @property
    def mu(self) -> np.ndarray:
        return self.d.mean(0)

    @property
    def extra(self) -> np.ndarray:
        return (self.t - self.ta[:, None]).mean(0)

    def column(self, j: int) -> Data:
        return self._replace(d=self.d[:, [j]], t=self.t[:, [j]])


@functools.cache
def data(plan: Plan) -> Data:
    arms = (plan.base, *plan.variants)
    per = ss.specs_of(plan.which, BUDGET, [a.spec for a in arms])
    have = per[plan.base.name].index            # seqscreen.main's order: the base's, sorted
    for a in plan.variants:
        have = have.intersection(per[a.name].index)
    free = set().union(*(free_first(a) for a in arms))
    blocks = [b for b in have if b not in free]
    a = per[plan.base.name]
    return Data(plan, blocks,
                np.column_stack([(per[v.name].perm[blocks] - a.perm[blocks]).to_numpy()
                                 for v in plan.variants]),
                np.column_stack([per[v.name].t[blocks].to_numpy() for v in plan.variants]),
                a.t[blocks].to_numpy(), [b for b in have if b in free])


def truth(x: Data, rate: float) -> np.ndarray:
    """Each variant's call on all of the design's blocks at `rate`: adopt or not."""
    return np.array([ss.call(float(m), float(e), np.array([rate])) == "adopt"
                     for m, e in zip(x.mu, x.extra, strict=True)])


def orders(u: int) -> list[np.ndarray]:
    """ORDERS permutations from one generator seeded ORDER_SEED: seqscreen.main's (its ORDERS
    first) and more of them."""
    return random_orders(u, count=ORDERS, rng=np.random.default_rng(ss.ORDER_SEED))


# ---- the policies ---------------------------------------------------------------------------

@dataclass(eq=False)
class Trace:
    """The order a policy was handed: the units to come when it had seen none. The Policy
    protocol says which units are upcoming but not which were seen, so a policy that needs the
    seen blocks' base times keeps it; the same-orders guard reads it."""
    order: np.ndarray | None = None

    def see(self, n: int, upcoming: np.ndarray) -> np.ndarray:
        if n == 0:
            self.order = upcoming.copy()
        if self.order is None or not np.array_equal(self.order[n:], upcoming):
            raise ValueError("the replay changed the order under a policy")
        return self.order

    def units(self, n: int, population: int) -> np.ndarray:
        """The units a `calls` was handed: the order's first n, or (before any order) the whole
        design in index order, which replay asks first."""
        if self.order is None:
            if n != population:
                raise ValueError(f"calls on {n} of {population} units before any order")
            return np.arange(population)
        return self.order[:n]


class SeqCalls(Protocol):
    """What seqscreen's call on `s` (its every block seen) is reported as."""

    def of(self, s: ss.Screen) -> np.ndarray: ...


def seq_call(s: ss.Screen, rates: np.ndarray) -> str:
    """seqscreen's call on all of `s`: stop_at with the floor at its blocks returns at the last
    (the collapse or the call), drawing no value, so the generator is never read."""
    stop = ss.stop_at(s, rates=rates, future=ss.FUTURE, floor=len(s.d),
                      rng=np.random.default_rng(0))
    assert stop is not None
    return stop.call


@dataclass(frozen=True)
class AtRates:
    """The fixed-rate comparison's: the call over the bracket, then at each judged rate."""

    def of(self, s: ss.Screen) -> np.ndarray:
        return np.array([seq_call(s, r) for r in (BRACKET, *JUDGED[:, None])])


@dataclass(frozen=True)
class Coded:
    """The bracket comparison's: the call over the bracket, as AdoptKeepAsk codes it."""

    def of(self, s: ss.Screen) -> np.ndarray:
        return np.array([CODE[seq_call(s, BRACKET)]])


@dataclass(eq=False)
class SeqScreen:
    """seqscreen.py, unchanged, as an epistemetry Policy for one variant against its base: `next`
    is stop_at on the blocks seen (a's times on every block in the order), replayed from the
    first block with the order's generator, as seqscreen.main replays it; `calls` is stop_at's
    call on the blocks seen (`calling`)."""
    t: np.ndarray       # b's GPU-seconds, (u,)
    ta: np.ndarray      # a's
    o: int              # the order's number: its generator, seeded [ORDER_SEED, o]
    floor: int
    calling: SeqCalls
    trace: Trace

    def next(self, seen: np.ndarray, seen_costs: np.ndarray, upcoming: np.ndarray, *,
             rng: np.random.Generator) -> int:
        order = self.trace.see(len(seen), upcoming)
        s = ss.Screen(seen[:, 0], self.t[order[:len(seen)]], self.ta[order])
        stop = ss.stop_at(s, rates=BRACKET, future=ss.FUTURE, floor=self.floor,
                          rng=np.random.default_rng([ss.ORDER_SEED, self.o]))
        return 0 if stop is not None else 1

    def calls(self, seen: np.ndarray, *, population: int, rng: np.random.Generator) -> np.ndarray:
        i = self.trace.units(len(seen), population)
        return self.calling.of(ss.Screen(seen[:, 0], self.t[i], self.ta[i]))


@dataclass(frozen=True, eq=False)
class RatioToBase:
    """seqscreen's cost model: an upcoming block costs the base's GPU-seconds on it times the
    variants' to the base's ratio over the blocks seen. `predict` is handed the upcoming units but
    not the seen ones, so the policy builds it each step with them (`seen`)."""
    seen: np.ndarray
    base: np.ndarray

    def predict(self, seen_costs: np.ndarray, upcoming: np.ndarray) -> float:
        return float(seen_costs.sum() / self.base[self.seen].sum() * self.base[upcoming].sum())


@dataclass(frozen=True, eq=False)
class Regret:
    """seqscreen's loss as an epistemetry DrawTarget: adopt each arm whose mean exceeds its tau;
    the risk sum_j min(E(tau_j - x_j)+, E(x_j - tau_j)+), in Lens A per governed block."""
    tau: np.ndarray

    def risk(self, draws: np.ndarray) -> np.ndarray:
        return np.asarray(np.minimum(np.maximum(self.tau - draws, 0).mean(-2),
                                     np.maximum(draws - self.tau, 0).mean(-2)).sum(-1))

    def regret(self, draws: np.ndarray) -> np.ndarray:
        adopt = draws.mean(-2) > self.tau
        return np.asarray(np.where(adopt[..., None, :], np.maximum(self.tau - draws, 0),
                                   np.maximum(draws - self.tau, 0)).sum(-1))

    def decide(self, draws: np.ndarray) -> np.ndarray:
        return np.asarray(draws.mean(0) > self.tau)


class Aimed(NamedTuple):
    functionals: np.ndarray     # (p, k) rows over the arms
    target: DrawTarget
    price: float                # ValueOfInformation's: the target's risk per GPU-second


class Aim(Protocol):
    """A target for the decision at tau, from the means and taus so far."""
    name: str

    def at(self, mu: np.ndarray, tau: np.ndarray, *, rate: float) -> Aimed: ...

    def adopted(self, decision: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ThreeWayAt:
    """The brief's: epistemetry's ThreeWay at tau_j per arm (functional d_j / tau_j, margin 1:
    better is adopt, within or worse keep). Its risk is the probability of a wrong class, so a
    GPU-second is worth rate / FUTURE over the Lens A a wrong call costs at the means so far."""

    @property
    def name(self) -> str:
        return "three-way"

    def at(self, mu: np.ndarray, tau: np.ndarray, *, rate: float) -> Aimed:
        if np.any(tau <= 0):
            raise ValueError(f"three-way needs tau > 0 (a variant dearer than its base): {tau}")
        return Aimed(np.diag(1 / tau), ThreeWay(margin=1.0),
                     rate / ss.FUTURE / float(np.mean(np.abs(mu - tau))))

    def adopted(self, decision: np.ndarray) -> np.ndarray:
        return np.asarray(decision == 1)


@dataclass(frozen=True)
class LinearAt:
    """seqscreen's loss (`Regret`), priced as seqscreen prices it, rate / FUTURE."""

    @property
    def name(self) -> str:
        return "linear"

    def at(self, mu: np.ndarray, tau: np.ndarray, *, rate: float) -> Aimed:
        return Aimed(np.eye(len(tau)), Regret(tau=tau), rate / ss.FUTURE)

    def adopted(self, decision: np.ndarray) -> np.ndarray:
        return np.asarray(decision, dtype=bool)


@dataclass(eq=False)
class Epi:
    """epistemetry's ValueOfMoreUnits on FinitePopulation at one rate, matched to seqscreen,
    rebuilt each step: tau and the cost model's ratio come from the blocks seen. The floor is held
    here too: no target can be built before a block is seen."""
    aim: Aim
    rate: float
    t: np.ndarray       # (u, k) the variants' GPU-seconds
    ta: np.ndarray      # (u,) the base's
    floor: int
    mc: MonteCarlo
    trace: Trace

    def _policy(self, seen: np.ndarray, i: np.ndarray) -> tuple[ValueOfMoreUnits, Aim]:
        tau = self.rate * (self.t[i] - self.ta[i, None]).mean(0)
        aimed = self.aim.at(seen.mean(0), tau, rate=self.rate)
        return ValueOfMoreUnits(
            belief=Bootstrap(), functionals=aimed.functionals, target=aimed.target,
            horizons=tuple(range(1, len(self.ta) + 1)), cost=RatioToBase(seen=i, base=self.ta),
            stop=ValueOfInformation(price=aimed.price, margin=StandardErrors(k=0.0)),
            floor=self.floor, outer=self.mc.outer, inner=self.mc.inner, blocks=self.mc.blocks,
            draws=self.mc.draws), self.aim

    def next(self, seen: np.ndarray, seen_costs: np.ndarray, upcoming: np.ndarray, *,
             rng: np.random.Generator) -> int:
        n = len(seen)
        order = self.trace.see(n, upcoming)
        if n < self.floor:
            return min(self.floor - n, len(upcoming))
        policy, _ = self._policy(seen, order[:n])
        return policy.next(seen, seen_costs, upcoming, rng=rng)

    def calls(self, seen: np.ndarray, *, population: int, rng: np.random.Generator) -> np.ndarray:
        policy, aim = self._policy(seen, self.trace.units(len(seen), population))
        return aim.adopted(policy.calls(seen, population=population, rng=rng))


@dataclass(eq=False)
class EpiBracket:
    """epistemetry's ValueOfMoreUnits on FinitePopulation with AdoptKeepAsk over seqscreen's
    bracket: the units' columns are the variants' gains, then their extra GPU-seconds, so the
    urn keeps each block's pair. Rebuilt each step for the price (the regret a wrong call costs
    at the means so far, at PRICE_RATE) and the cost model's ratio; the floor held here, as in
    `Epi`."""
    ta: np.ndarray      # (u,) the base's GPU-seconds
    floor: int
    mc: MonteCarlo
    trace: Trace

    def _policy(self, seen: np.ndarray, i: np.ndarray) -> ValueOfMoreUnits:
        m = seen.shape[1] // 2
        miss = float(np.mean(np.abs(seen[:, :m].mean(0) - PRICE_RATE * seen[:, m:].mean(0))))
        return ValueOfMoreUnits(
            belief=Bootstrap(), functionals=np.eye(2 * m),
            target=AdoptKeepAsk(low=ss.RATES[0], high=ss.RATES[1]),
            horizons=tuple(range(1, len(self.ta) + 1)), cost=RatioToBase(seen=i, base=self.ta),
            stop=ValueOfInformation(price=PRICE_RATE / ss.FUTURE / miss,
                                    margin=StandardErrors(k=0.0)),
            floor=self.floor, outer=self.mc.outer, inner=self.mc.inner, blocks=self.mc.blocks,
            draws=self.mc.draws)

    def next(self, seen: np.ndarray, seen_costs: np.ndarray, upcoming: np.ndarray, *,
             rng: np.random.Generator) -> int:
        n = len(seen)
        order = self.trace.see(n, upcoming)
        if n < self.floor:
            return min(self.floor - n, len(upcoming))
        return self._policy(seen, order[:n]).next(seen, seen_costs, upcoming, rng=rng)

    def calls(self, seen: np.ndarray, *, population: int, rng: np.random.Generator) -> np.ndarray:
        return self._policy(seen, self.trace.units(len(seen), population)).calls(
            seen, population=population, rng=rng)


@dataclass(eq=False)
class Vetoed:
    """reblock's collapse veto around a policy of one variant: a block more than COLLAPSE below
    the base (its gain, the first column) stops the screen with `keep`. The inner policy's batches
    run a block at a time (`pending`), so the veto sees every block as seqscreen's does; the inner
    policy is asked at the same blocks with the same generator as without the veto."""
    inner: Epi | EpiBracket
    keep: bool | int    # the inner policy's call of keep
    pending: int = 0    # blocks of the inner policy's last batch still to run

    def next(self, seen: np.ndarray, seen_costs: np.ndarray, upcoming: np.ndarray, *,
             rng: np.random.Generator) -> int:
        if len(seen) and seen[-1, 0] < -ss.COLLAPSE:
            return 0
        if self.pending:
            self.pending -= 1
            return 1
        k = self.inner.next(seen, seen_costs, upcoming, rng=rng)
        self.pending = max(k - 1, 0)
        return min(k, 1)

    def calls(self, seen: np.ndarray, *, population: int, rng: np.random.Generator) -> np.ndarray:
        if np.any(seen[:, 0] < -ss.COLLAPSE):
            return np.array([self.keep])
        return self.inner.calls(seen, population=population, rng=rng)


# ---- jobs -----------------------------------------------------------------------------------

class Verdict(NamedTuple):
    rate: float
    adopt: tuple[bool, ...]     # per arm
    bracket: str                # seqscreen's call over the bracket ("" for epistemetry's)


class Maker(Protocol):
    name: str

    def make(self, x: Data, o: int, trace: Trace) -> Policy: ...

    def verdicts(self, calls: np.ndarray) -> list[Verdict]: ...


@dataclass(frozen=True)
class SeqScreenMaker:
    floor: int
    calling: SeqCalls

    @property
    def name(self) -> str:
        return "seqscreen"

    def make(self, x: Data, o: int, trace: Trace) -> Policy:
        (j,) = range(x.d.shape[1])
        return SeqScreen(t=x.t[:, j], ta=x.ta, o=o, floor=self.floor, calling=self.calling,
                         trace=trace)

    def verdicts(self, calls: np.ndarray) -> list[Verdict]:
        return [Verdict(float(r), (c == "adopt",), str(calls[0]))
                for r, c in zip(JUDGED, calls[1:], strict=True)]


@dataclass(frozen=True)
class EpiMaker:
    aim: Aim
    rate: float
    veto: bool
    floor: int
    mc: MonteCarlo

    @property
    def name(self) -> str:
        return f"{self.aim.name}{'+veto' if self.veto else ''}"

    def make(self, x: Data, o: int, trace: Trace) -> Policy:
        p = Epi(self.aim, self.rate, x.t, x.ta, self.floor, self.mc, trace)
        return Vetoed(p, keep=False) if self.veto else p

    def verdicts(self, calls: np.ndarray) -> list[Verdict]:
        return [Verdict(self.rate, tuple(bool(c) for c in calls), "")]


class Job(NamedTuple):
    plan: Plan
    arm: int | None     # the variant a pairwise screen of a many-arm design runs; None: all
    maker: Maker


def jobs(*, floor: int, mc: MonteCarlo) -> list[Job]:
    """The fixed-rate comparison's (the original rule's)."""
    epi = [EpiMaker(aim, float(r), veto, floor, mc) for aim in (ThreeWayAt(), LinearAt())
           for r in JUDGED for veto in (True, False)]
    seq = SeqScreenMaker(floor, AtRates())
    out = [Job(p, None, m) for p in PAIRS for m in (seq, *epi)]
    for p in MANY:
        out += [Job(p, j, seq) for j in range(len(p.variants))]
        out += [Job(p, None, m) for m in epi if not m.veto]
    return out


def run_job(job: Job, *, given: dict[int, np.ndarray] | None = None) -> list[dict]:
    """Every order of `job`'s design through epistemetry's replay; `given` swaps an order (the
    same-orders guard's fault)."""
    x = data(job.plan) if job.arm is None else data(job.plan).column(job.arm)
    design = Design(outcomes=x.d, costs=x.t.sum(1))
    rows = []
    for o, order in enumerate(orders(len(x.blocks))):
        trace = Trace()
        run = replay(design, job.maker.make(x, o, trace),
                     orders=[order if given is None or o not in given else given[o]],
                     rng=np.random.default_rng([ss.ORDER_SEED, o]), scoring=Exact()).runs[0]
        assert trace.order is not None
        for v in job.maker.verdicts(run.calls):
            rows.append(dict(plan=job.plan.name, arms=",".join(
                a.name for a in (job.plan.variants if job.arm is None
                                 else (job.plan.variants[job.arm],))),
                policy=job.maker.name, order=o, rate=v.rate, units=run.units, cost=run.cost,
                adopt=list(v.adopt), bracket=v.bracket, given=trace.order.tolist()))
    return rows


@dataclass(frozen=True)
class SeqBracketMaker:
    """seqscreen in the bracket comparison: its call over the bracket, coded."""
    floor: int

    @property
    def name(self) -> str:
        return "seqscreen"

    def make(self, x: Data, o: int, trace: Trace) -> Policy:
        return SeqScreenMaker(self.floor, Coded()).make(x, o, trace)


@dataclass(frozen=True)
class EpiBracketMaker:
    veto: bool
    floor: int
    mc: MonteCarlo

    @property
    def name(self) -> str:
        return f"adopt/keep/ask{'+veto' if self.veto else ''}"

    def make(self, x: Data, o: int, trace: Trace) -> Policy:
        p = EpiBracket(x.ta, self.floor, self.mc, trace)
        return Vetoed(p, keep=KEEP) if self.veto else p


class BracketJob(NamedTuple):
    plan: Plan
    arm: int | None     # the variant a pairwise screen of a many-arm design runs; None: all
    maker: SeqBracketMaker | EpiBracketMaker
    veto: bool          # the setting: the veto everywhere (seqscreen's, the wrapper, the truth),
                        # or nowhere (seqscreen's patched off in this process)


def bracket_jobs(*, floor: int, mc: MonteCarlo) -> list[BracketJob]:
    out = [BracketJob(p, None, m, veto) for veto in (False, True) for p in PAIRS
           for m in (SeqBracketMaker(floor), EpiBracketMaker(veto, floor, mc))]
    for p in MANY:      # no veto: a collapse would drop one arm, and none of them has one
        out += [BracketJob(p, j, SeqBracketMaker(floor), False) for j in range(len(p.variants))]
        out.append(BracketJob(p, None, EpiBracketMaker(False, floor, mc), False))
    return out


def bracket_design(x: Data) -> Design:
    """The units' gains, then their extra GPU-seconds: AdoptKeepAsk's functionals, paired."""
    return Design(outcomes=np.hstack([x.d, x.t - x.ta[:, None]]), costs=x.t.sum(1))


def run_bracket_job(job: BracketJob, *, given: dict[int, np.ndarray] | None = None) -> list[dict]:
    """Every order of `job`'s design through epistemetry's replay, scored with AsksApart, in its
    veto setting; `given` swaps an order (the same-orders guard's fault)."""
    x = data(job.plan) if job.arm is None else data(job.plan).column(job.arm)
    design = bracket_design(x)
    ss.COLLAPSE = VETO if job.veto else math.inf
    rows = []
    try:
        for o, order in enumerate(orders(len(x.blocks))):
            trace = Trace()
            rep = replay(design, job.maker.make(x, o, trace),
                         orders=[order if given is None or o not in given else given[o]],
                         rng=np.random.default_rng([ss.ORDER_SEED, o]),
                         scoring=AsksApart(ask=ASK))
            run = rep.runs[0]
            assert trace.order is not None
            rows.append(dict(
                plan=job.plan.name, arms=",".join(
                    a.name for a in (job.plan.variants if job.arm is None
                                     else (job.plan.variants[job.arm],))),
                policy=job.maker.name, veto=job.veto, order=o, units=run.units, cost=run.cost,
                calls=run.calls.tolist(), full=rep.full_calls.tolist(), wrong=run.wrong,
                asks=run.asks, given=trace.order.tolist()))
    finally:
        ss.COLLAPSE = VETO
    return rows


def _run_one(job: Job) -> list[dict]:
    return run_job(job)


def job_file(job: BracketJob) -> Path:
    arm = "all" if job.arm is None else job.plan.variants[job.arm].name
    name = f"{job.plan.name}_{arm}_{job.maker.name}_{'veto' if job.veto else 'none'}"
    return JOBS_BRACKET / (re.sub(r"[^A-Za-z0-9+.-]", "_", name) + ".parquet")


def _bracket_one(job: BracketJob) -> str:
    rows = run_bracket_job(job)
    pd.DataFrame(rows).to_parquet(job_file(job))
    return f"{job.plan.name} {rows[0]['policy']} {rows[0]['arms']} veto {job.veto}"


def run_bracket() -> None:
    JOBS_BRACKET.mkdir(exist_ok=True)
    every = bracket_jobs(floor=ss.MIN_BLOCKS, mc=MC)
    todo = sorted((j for j in every if not job_file(j).exists()),     # the longest first
                  key=lambda j: j.plan not in MANY or j.arm is not None)
    print(f"{len(every) - len(todo)} of {len(every)} jobs done before", flush=True)
    with Pool(WORKERS) as pool:
        for k, done in enumerate(pool.imap_unordered(_bracket_one, todo)):
            print(f"{k + 1}/{len(todo)} {done}", flush=True)
    rows = pd.concat([pd.read_parquet(job_file(j)) for j in every], ignore_index=True)
    rows.to_parquet(ROWS_BRACKET)
    print(f"{len(rows)} rows -> {ROWS_BRACKET}")


def run() -> None:
    todo = sorted(jobs(floor=ss.MIN_BLOCKS, mc=MC),     # the many-arm screens, the longest, first
                  key=lambda j: j.plan not in MANY or j.arm is not None)
    for p in PAIRS + MANY:
        x = data(p)
        print(f"{p.name:14s} {len(x.blocks):2d} blocks ({p.card}), dropped {len(x.dropped)}: "
              f"mu {np.array2string(x.mu, precision=5)}, "
              f"extra s {np.array2string(x.extra, precision=1)}", flush=True)
    rows = []
    with Pool(WORKERS) as pool:
        for k, r in enumerate(pool.imap_unordered(_run_one, todo)):
            rows += r
            print(f"{k + 1}/{len(todo)} {r[0]['plan']} {r[0]['policy']} {r[0]['arms']} "
                  f"rate {r[0]['rate']:.2g}", flush=True)
    pd.DataFrame(rows).to_parquet(ROWS)
    print(f"{len(rows)} rows -> {ROWS}")


# ---- the guards -----------------------------------------------------------------------------

def guard_wrapper(*, floor: int) -> bool:
    """The wrapper reproduces seqscreen's own replay, order by order: stop_at on each ordered
    screen as seqscreen.main runs it (its stop and call), and main's printed summary where the
    design has every one of its blocks; in both comparisons (the bracket's with the veto, as
    seqscreen runs). Fault `floor`: the wrapper's floor one below."""
    ok = True
    for p in PAIRS:
        x = data(p)
        rows = pd.DataFrame(run_job(Job(p, None, SeqScreenMaker(floor, AtRates()))))
        got = rows[rows.rate == JUDGED[0]].sort_values("order")
        coded = pd.DataFrame(run_bracket_job(BracketJob(p, None, SeqBracketMaker(floor), True)))
        s = ss.Screen(x.d[:, 0], x.t[:, 0], x.ta)
        own = [ss.stop_at(ss.Screen(s.d[i], s.t[i], s.ta[i]), rates=BRACKET, future=ss.FUTURE,
                          floor=ss.MIN_BLOCKS, rng=np.random.default_rng([ss.ORDER_SEED, o]))
               for o, i in enumerate(orders(len(x.blocks)))]
        same = [(int(g.units), g.bracket) == (st.n, st.call) for g, st in
                zip(got.itertuples(), own, strict=True)]
        same += [(int(g.units), g.calls[0]) == (st.n, CODE[st.call]) for g, st in
                 zip(coded.sort_values("order").itertuples(), own, strict=True)]
        line = ""
        if not x.dropped:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                ss.main(p.which, BUDGET, [p.base.spec, p.variants[0].spec],
                        [f"{p.variants[0].name}/{p.base.name}"])
            line = out.getvalue().splitlines()[1]
            m = re.search(r"median (\d+) blocks \((\d+)-(\d+)\).*calls (\{.*\}), (\d+) of", line)
            assert m is not None, line
            first = got[got.order < ss.ORDERS]      # main replays its ORDERS
            n = first.units.to_numpy()
            calls = {c: int((first.bracket == c).sum()) for c in sorted(set(first.bracket))}
            same.append(m.groups()[:4] == (str(int(np.median(n))), str(n.min()), str(n.max()),
                                           str(calls)))
        print(f"  {p.name:14s} {sum(same)} of {len(same)} match"
              f"{' (main: ' + line.split(': ', 1)[1][:60] + '...)' if line else ''}")
        ok &= all(same)
    return ok


def degenerate() -> Data:
    """p64 against base with p64's Lens A replaced by the base's: a variant that changes nothing,
    at p64's GPU time."""
    x = data(PLANS["p64/base"])
    return x._replace(d=np.zeros_like(x.d))


def guard_degenerate(*, floor: int) -> bool:
    """epistemetry's policy (adopt/keep/ask, and each aim and rate of the fixed comparison; with
    and without the veto) on a variant equal to its base stops at the floor with "keep", every
    order. Fault `ignored`: the floor set to 1."""
    x = degenerate()
    ok = True
    for veto in (True, False):
        m_ = EpiBracketMaker(veto, floor, MC)
        runs = [replay(bracket_design(x), m_.make(x, o, Trace()), orders=[order],
                       rng=np.random.default_rng([ss.ORDER_SEED, o]),
                       scoring=AsksApart(ask=ASK)).runs[0]
                for o, order in enumerate(orders(len(x.blocks)))]
        good = [r_.units == ss.MIN_BLOCKS and np.all(r_.calls == KEEP) for r_ in runs]
        print(f"  {m_.name:16s}: {sum(good)} of {len(good)} stop at {ss.MIN_BLOCKS} with keep "
              f"(units {sorted({r_.units for r_ in runs})})")
        ok &= all(good)
    design = Design(outcomes=x.d, costs=x.t.sum(1))
    for aim in (ThreeWayAt(), LinearAt()):
        for r in JUDGED:
            for veto in (True, False):
                m = EpiMaker(aim, float(r), veto, floor, MC)
                runs = [replay(design, m.make(x, o, Trace()), orders=[order],
                               rng=np.random.default_rng([ss.ORDER_SEED, o]),
                               scoring=Exact()).runs[0]
                        for o, order in enumerate(orders(len(x.blocks)))]
                good = [r_.units == ss.MIN_BLOCKS and not r_.calls.any() for r_ in runs]
                print(f"  {m.name:16s} rate {r:.1e}: {sum(good)} of {len(good)} stop at "
                      f"{ss.MIN_BLOCKS} with keep (units {sorted({r_.units for r_ in runs})})")
                ok &= all(good)
    return ok


def same_orders(rows: pd.DataFrame) -> bool:
    """Every policy compared on an order was handed that order: the given orders equal
    seqscreen.main's, per design."""
    ok = True
    for p in PAIRS + MANY:
        canon = orders(len(data(p).blocks))
        r = rows[rows.plan == p.name]
        bad = [(g.policy, g.arms, g.order) for g in r.itertuples()
               if not np.array_equal(np.asarray(g.given), canon[g.order])]
        if bad:
            print(f"  {p.name}: {len(bad)} runs on another order, e.g. {bad[0]}")
        ok &= not bad
    return ok


def guard_orders(*, shuffled: bool) -> bool:
    """The same-orders check on one design's every policy, in both comparisons; fault
    `shuffled`: one policy (adopt/keep/ask without the veto) handed a reshuffled order 3."""
    p = PLANS["both/p64"]
    given = {3: np.random.default_rng(1).permutation(orders(len(data(p).blocks))[3])}
    rows = [r for job in jobs(floor=ss.MIN_BLOCKS, mc=MC) if job.plan == p
            for r in run_job(job)]
    for bj in bracket_jobs(floor=ss.MIN_BLOCKS, mc=MC):
        if bj.plan == p:
            fault = shuffled and isinstance(bj.maker, EpiBracketMaker) and not bj.veto
            rows += run_bracket_job(bj, given=given if fault else None)
    return same_orders(pd.DataFrame(rows))


FAULTS = ("floor", "ignored", "shuffled")


def guards(fault: str | None) -> None:
    if fault is not None and fault not in FAULTS:
        raise SystemExit(f"unknown fault {fault!r}: one of {', '.join(FAULTS)}")
    results = {
        "wrapper": guard_wrapper(floor=ss.MIN_BLOCKS - (fault == "floor")),
        "degenerate": guard_degenerate(floor=1 if fault == "ignored" else ss.MIN_BLOCKS),
        "same orders": guard_orders(shuffled=fault == "shuffled"),
    }
    for name, path in (("fixed", ROWS), ("bracket", ROWS_BRACKET)):
        if path.exists():
            results[f"same orders, the {name} run"] = same_orders(pd.read_parquet(path))
    for k, v in results.items():
        print(f"{k}: {'pass' if v else 'FAIL'}")
    raise SystemExit(0 if all(results.values()) else 1)


# ---- the report -----------------------------------------------------------------------------

def judged(rows: pd.DataFrame) -> pd.DataFrame:
    """Each run's wrong calls, regret, cost fraction and realized loss at its rate."""
    out = []
    for g in rows.itertuples():
        p = PLANS[g.plan]
        x = data(p)
        cols = [i for i, v in enumerate(p.variants) if v.name in g.arms.split(",")]
        mu, extra, total = x.mu[cols], x.extra[cols], x.t[:, cols].sum()
        right = truth(x, g.rate)[cols]
        wrong = np.asarray(g.adopt) != right
        regret = float(np.sum(wrong * np.abs(mu - g.rate * extra)))
        out.append(dict(plan=g.plan, arms=g.arms, policy=g.policy, order=g.order, rate=g.rate,
                        units=g.units, cost=g.cost, frac=g.cost / total, wrong=int(wrong.sum()),
                        regret=regret, loss=regret + g.rate / ss.FUTURE * g.cost,
                        bracket=g.bracket))
    return pd.DataFrame(out)


def _mr(s: pd.Series, f: str) -> str:
    return f"{f.format(s.median())} ({f.format(s.min())} -- {f.format(s.max())})"


SUMMED = "seqscreen per variant"    # seqscreen's pairwise screens of a many-arm design, summed


def screens(j: pd.DataFrame) -> pd.DataFrame:
    """One row a screen (design, policy, order, rate); a many-arm design's seqscreen screens,
    one a variant, summed per order into one (SUMMED), its cost fraction of every arm's blocks."""
    many = j.plan.isin([p.name for p in MANY]) & (j.policy == "seqscreen")
    summed = (j[many].groupby(["plan", "order", "rate"], as_index=False)
              .agg(units=("units", "sum"), cost=("cost", "sum"), wrong=("wrong", "sum"),
                   regret=("regret", "sum"), loss=("loss", "sum"))
              .assign(policy=SUMMED, bracket=""))
    summed["frac"] = summed.cost / summed.plan.map(lambda n: data(PLANS[n]).t.sum())
    return pd.concat([j[~many], summed], ignore_index=True)


def rule(t: pd.DataFrame, candidate: str, against: str) -> tuple[bool, str]:
    """The decision rule on screens `t`: the candidate's median realized loss (summed over the
    designs) below `against`'s at every judged rate, and no more wrong calls in all."""
    med = t.groupby(["policy", "rate", "plan"]).loss.median().groupby(["policy", "rate"]).sum()
    wrong = t.groupby("policy").wrong.sum()
    below = [med[(candidate, r)] < med[(against, r)] for r in JUDGED]
    ok = all(below) and wrong[candidate] <= wrong[against]
    return ok, (f"{candidate} against {against}: loss (1e-6, summed medians) "
                + ", ".join(f"{r:.1e} {1e6 * med[(candidate, r)]:.1f} vs "
                            f"{1e6 * med[(against, r)]:.1f}" for r in JUDGED)
                + f"; wrong calls {wrong[candidate]} vs {wrong[against]}: "
                + ("ADOPT" if ok else "keep seqscreen"))


def report() -> None:
    t = screens(judged(pd.read_parquet(ROWS)))
    t.to_parquet(HERE / "epireplay_screens.parquet")
    print(f"designs ({ORDERS} orders each; truth at the judged rates, A adopt k keep):")
    for p in PAIRS + MANY:
        x = data(p)
        print(f"  {p.name:14s} {len(x.blocks):2d} blocks ({len(x.dropped)} dropped), {p.card}; "
              f"mu {' '.join(f'{m:+.5f}' for m in x.mu)}, extra s "
              f"{' '.join(f'{e:+.1f}' for e in x.extra)}; "
              + " ".join("".join("A" if a else "k" for a in truth(x, r)) for r in JUDGED))
    print("\nper design, policy and rate: blocks, cost fraction, realized loss (1e-6), each the "
          "median (range) over orders; wrong calls in all orders")
    for p in PAIRS + MANY:
        for r in JUDGED:
            sel = t[(t.plan == p.name) & (t.rate == r)]
            for pol, g in sel.groupby("policy", sort=False):
                asks = (f", asks {int((g.bracket == 'ask').sum())}" if pol == "seqscreen"
                        and p in PAIRS else "")
                print(f"  {p.name:14s} {r:.1e} {pol:22s} {_mr(g.units, '{:.0f}'):16s} "
                      f"{_mr(g.frac, '{:.2f}'):20s} {_mr(1e6 * g.loss, '{:.1f}'):26s} "
                      f"wrong {g.wrong.sum()}{asks}")
    pairs = t[t.plan.isin([p.name for p in PAIRS])]
    many = t[t.plan.isin([p.name for p in MANY])]
    print("\ntotals per rate (sum over designs of the median loss, 1e-6; mean of the designs' "
          "median cost fractions; wrong calls in all):")
    for name, sel in (("pairwise", pairs), ("many-arm", many)):
        tot = (sel.groupby(["policy", "rate", "plan"])
               .agg(loss=("loss", "median"), frac=("frac", "median"), wrong=("wrong", "sum"))
               .groupby(["policy", "rate"])
               .agg(loss=("loss", "sum"), frac=("frac", "mean"), wrong=("wrong", "sum")))
        print(f"  {name}:")
        print(tot.assign(loss=lambda d: (1e6 * d.loss).round(1), frac=lambda d: d.frac.round(3))
              .to_string())
    print("\nthe rule:")
    for sel, cand, against in ((pairs, "three-way+veto", "seqscreen"),
                               (many, "three-way", SUMMED)):
        print("  candidate " + rule(sel, cand, against)[1])
    for sel, cand, against in ((pairs, "three-way", "seqscreen"),
                               (pairs, "linear+veto", "seqscreen"),
                               (pairs, "linear", "seqscreen"), (many, "linear", SUMMED)):
        print("  reported  " + rule(sel, cand, against)[1])


def judged_bracket(rows: pd.DataFrame) -> pd.DataFrame:
    """Each bracket run: AsksApart's wrong calls and asks (replay's), the posterior mean's call at
    the stop (seqscreen's call on the blocks seen) against the run's, and at each judged rate the
    call resolved (an ask answered from the screen's own means) against the whole design's, its
    regret and the realized loss."""
    out = []
    for g in rows.itertuples():
        p = PLANS[g.plan]
        x = data(p)
        seen = np.asarray(g.given)[:g.units]
        vetoed = VETO if g.veto else math.inf
        cols = [i for i, v in enumerate(p.variants) if v.name in g.arms.split(",")]
        row = dict(plan=g.plan, arms=g.arms, policy=g.policy, veto=g.veto, order=g.order,
                   units=g.units, cost=g.cost, frac=g.cost / x.t[:, cols].sum(), wrong=g.wrong,
                   asks=g.asks, apart=0)
        for r in JUDGED:
            row[f"wrong@{r:.1e}"], row[f"loss@{r:.1e}"] = 0, r / ss.FUTURE * g.cost
        for k, j in enumerate(cols):
            mu, extra = float(x.mu[j]), float(x.extra[j])
            ms, es = float(x.d[seen, j].mean()), float(np.mean(x.t[seen, j] - x.ta[seen]))
            whole = KEEP if np.any(x.d[:, j] < -vetoed) else CODE[ss.call(mu, extra, BRACKET)]
            assert g.full[k] == whole, (g.plan, g.policy, g.full, whole)
            stop = KEEP if np.any(x.d[seen, j] < -vetoed) else CODE[ss.call(ms, es, BRACKET)]
            row["apart"] += int(g.calls[k] != stop)
            for r in JUDGED:
                rr = np.array([r])
                right = whole != KEEP and (whole == ADOPT or ss.call(mu, extra, rr) == "adopt")
                said = g.calls[k] == ADOPT or (g.calls[k] == ASK and stop != KEEP
                                               and ss.call(ms, es, rr) == "adopt")
                if said != right:
                    row[f"wrong@{r:.1e}"] += 1
                    row[f"loss@{r:.1e}"] += abs(mu - r * extra)
        out.append(row)
    return pd.DataFrame(out)


def report_bracket() -> None:
    j = judged_bracket(pd.read_parquet(ROWS_BRACKET))
    many = j.plan.isin([p.name for p in MANY]) & (j.policy == "seqscreen")
    agg = {c: (c, "sum") for c in ["units", "cost", "wrong", "asks", "apart",
                                   *(f"{k}@{r:.1e}" for k in ("wrong", "loss") for r in JUDGED)]}
    summed = (j[many].groupby(["plan", "veto", "order"], as_index=False).agg(**agg)
              .assign(policy=SUMMED))
    summed["frac"] = summed.cost / summed.plan.map(lambda n: data(PLANS[n]).t.sum())
    t = pd.concat([j[~many], summed], ignore_index=True)
    t.to_parquet(HERE / "epireplay_bracket_screens.parquet")
    loss = [f"loss@{r:.1e}" for r in JUDGED]
    print(f"designs ({ORDERS} orders each): the whole design's call over the bracket, no veto / "
          "veto")
    for p in PAIRS + MANY:
        x = data(p)
        calls = [ss.call(float(m), float(e), BRACKET) for m, e in zip(x.mu, x.extra, strict=True)]
        low = [f"{x.d[:, k].min():+.3f}" for k in range(len(p.variants))]
        print(f"  {p.name:14s} {len(x.blocks):2d} blocks; {' '.join(calls)}; worst block {low}")
    print("\nper design, veto setting and policy: blocks, cost fraction, each the median (range)"
          " over orders; wrong calls (AsksApart), asks, stops where the likeliest call differs"
          " from the posterior mean's, all in all orders; the median realized loss (1e-6) at "
          + ", ".join(f"{r:.1e}" for r in JUDGED))
    for p in PAIRS + MANY:
        for veto in (False, True):
            sel = t[(t.plan == p.name) & (t.veto == veto)]
            for pol, g in sel.groupby("policy", sort=False):
                print(f"  {p.name:14s} {'veto' if veto else '    '} {pol:22s} "
                      f"{_mr(g.units, '{:.0f}'):16s} {_mr(g.frac, '{:.2f}'):20s} "
                      f"wrong {g.wrong.sum():3d} asks {g.asks.sum():3d} apart {g.apart.sum():3d} "
                      f"loss " + " ".join(f"{1e6 * g[c].median():.1f}" for c in loss))
    pairs = t[t.plan.isin([p.name for p in PAIRS])]
    mt = t[t.plan.isin([p.name for p in MANY])]
    print("\ntotals (sum over designs of the median loss, 1e-6; mean of the designs' median cost"
          " fractions; wrong calls, asks and likeliest-vs-mean differences in all):")
    for name, sel in (("pairwise", pairs), ("many-arm", mt)):
        per = sel.groupby(["veto", "policy", "plan"]).agg(
            **{c: (c, "median") for c in [*loss, "frac"]},
            wrong=("wrong", "sum"), asks=("asks", "sum"), apart=("apart", "sum"))
        tot = per.groupby(["veto", "policy"]).agg(
            **{c: (c, "sum") for c in [*loss, "wrong", "asks", "apart"]}, frac=("frac", "mean"))
        print(f"  {name}:")
        print(tot.assign(**{c: (1e6 * tot[c]).round(1) for c in loss},
                         frac=tot.frac.round(3)).to_string())
    print("\nthe amended rule:")
    for sel, veto, cand, against in ((pairs, False, "adopt/keep/ask", "seqscreen"),
                                     (pairs, True, "adopt/keep/ask+veto", "seqscreen"),
                                     (mt, False, "adopt/keep/ask", SUMMED)):
        u = sel[sel.veto == veto]
        med = u.groupby(["policy", "plan"])[loss].median().groupby("policy").sum()
        wrong, asks = u.groupby("policy").wrong.sum(), u.groupby("policy").asks.sum()
        below = [med.loc[cand, c] < med.loc[against, c] for c in loss]
        ok = all(below) and wrong[cand] <= wrong[against] and asks[cand] <= asks[against]
        print(f"  {'veto' if veto else 'no veto'}: {cand} against {against}: loss (1e-6) "
              + ", ".join(f"{c[5:]} {1e6 * med.loc[cand, c]:.1f} vs "
                          f"{1e6 * med.loc[against, c]:.1f}" for c in loss)
              + f"; wrong {wrong[cand]} vs {wrong[against]}; asks {asks[cand]} vs "
              f"{asks[against]}: {'ADOPT' if ok else 'keep seqscreen'}")


if __name__ == "__main__":
    {("run", "fixed"): run, ("run", "bracket"): run_bracket, ("report", "fixed"): report,
     ("report", "bracket"): report_bracket,
     ("guards", "none"): lambda: guards(None), ("guards", "floor"): lambda: guards("floor"),
     ("guards", "ignored"): lambda: guards("ignored"),
     ("guards", "shuffled"): lambda: guards("shuffled")}[tuple(sys.argv[1:3])]()
