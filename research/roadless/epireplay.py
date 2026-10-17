"""Pruning by epistemetry's decision layer, a fourth time: the realized loss as the screen's own
target, replayed against seqscreen.py (BACKLOG, "Pruning by epistemetry's decision layer"; NOTES,
"Pruning by epistemetry: the realized loss as the target").

Three replays kept seqscreen (NOTES, "Pruning by epistemetry: a replay against seqscreen" and
"Pruning by epistemetry, again: step-wise re-decision and per-arm screens", with its "A price per
arm"): at c1784af; at 64aae7f, re-deciding after every block, with per-arm screens; at 3a3b9a3,
each arm priced by its own stake. The last named what would still have to change: a linear-loss
target over the rate bracket, priced in the owner's units, and a price on asks. GeoffChurch/
epistemetry 674dba4 builds the first, `AdoptKeepAskByStake(rates, weights)`, on 046675b's API
(`Truth.of`, `Judgement`, `Decision`'s readings, `Run.order`). The owner decided the second
(2026-10-16): asks are queued and cost 0; an ask records the question and leaves the default as it
is, and is answered later, at the owner's true rate, from the screen's means. This replays both
targets side by side under the same rule, through an ephemeral overlay (reblock does not depend on
epistemetry):

    EPI='epistemetry @ git+https://github.com/GeoffChurch/epistemetry@674dba4'
    uv run --with "$EPI" python research/roadless/epireplay.py check
    uv run --with "$EPI" python research/roadless/epireplay.py run <group>[,<group>...] <workers>
    uv run --with "$EPI" python research/roadless/epireplay.py guards <fault>
    uv run --with "$EPI" python research/roadless/epireplay.py guard <guard>
    uv run --with "$EPI" python research/roadless/epireplay.py report

with OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS=1; the groups are GROUPS' (seqscreen, zero-one, stake,
sensitivity), each job a file, so a rerun resumes; `guards none` runs every guard, `guards <fault>`
(FAULTS) the one it breaks, `guard <guard>` one guard without its fault.

THE DESIGNS are the first replay's, frozen as data in DESIGNS (committed), the only design data
read here: its 13 pairwise designs (the collapse, C1m4/p64, among them) and 5 many-arm ones,
extracted once by the first replay's own code (`git archive 0c2e588e^ research/roadless`, its
`data` and `orders`, seqscreen handed simp_compare's loaders as when it ran) from raster-era rows
that nothing here reads. Per design: the names, the block ids, each arm's Lens A gain over the
base and its GPU-seconds, the base's GPU-seconds, and the 100 orders (seqscreen.main's 20, then 80
more from its generator). `check` holds them to the first replay's stored runs: every whole-design
call and seqscreen's replay of every order.

THE TRUTH is one per design and veto setting: seqscreen's call over the bracket on all of the
design's blocks (the drop rule's KEEP on a collapse), `Truth.of` the wrapped seqscreen, handed to
every replay on the design. Every policy's and target's own call on the whole design is checked to
be it (the one-truth guard): a target that changed it would change the test, not the screen.

THE POLICIES run on the same orders through epistemetry's `replay`, each order carrying its own
stream (`Maker.stream`), in two veto settings (`Setting`): everywhere (the harness's VetoBelow at
seqscreen's COLLAPSE, strictly below, for every policy and the truth alike) or nowhere (NoVeto,
and seqscreen's own COLLAPSE patched off inside the process; no file edited):
- `SeqScreen`: seqscreen unchanged, as a Policy of one arm: its stop (stop_at on the blocks seen,
  replayed from the first block with the order's generator, as seqscreen.main replays it), floor
  and veto, and its call over the bracket.
- ValueOfMoreUnits on FinitePopulation, matched to seqscreen as in the earlier replays: an arm's
  outcome its gain and its extra GPU-seconds (two columns of one ball); floor MIN_BLOCKS; every
  horizon to the last block; seqscreen's cost model (`RatioToBase`); `step` 1 (re-decide after
  every block, as seqscreen does: a card runs one block at a time here); no Monte Carlo margin.
  Many-arm: one such policy over every variant of a design, each arm stopping on its own value
  (the per-arm screen), against seqscreen once per variant on the same orders, summed. Its target,
  a Strategy (`ZeroOne` or `ByStake`):
  - `ZeroOne`, the third replay's, reproduced: AdoptKeepAsk over the bracket's ends, a 0-1 risk
    (the probability of another call), each arm priced by its own stake (`OwnStake`: seqscreen's
    price, the bracket's mean rate over FUTURE, over the Lens A a wrong call on that arm costs at
    its means so far) or, many-arm, by the live arms' stakes averaged (`SharedStake`);
  - `ByStake`, the realized loss as the target: AdoptKeepAskByStake over weighted rate points, a
    wrong call at rate r losing |gain - r x extra|; at each rate the call is the one the posterior
    means favour, ASK where the points disagree, and the risk that call's expected loss, summed
    over arms and averaged over the points. Its value is in Lens A, so one price serves every arm
    with no stake: sum_r w_r r / FUTURE per GPU-second (exact, the loss and the cost both linear
    in r). Over BRACKET, seqscreen's five geometric rates equally weighted (the points its stop
    averages over: price PRICE_RATE / FUTURE), pairwise and many-arm; and, for its sensitivity,
    over SENSITIVITY, nine rates evenly spaced in log rate across RATES, pairwise.
Each run records its decisions (`Run.decisions`: each arm's prefix, the step, and each judged
arm's rule and the reading that decided it), which the guards read.

THE RULE, fixed (the first replay's, amended): adopt epistemetry for the consolidated pass's
pruning only if, against seqscreen in the same veto setting, (1) its median realized loss is below
seqscreen's at every judged rate (per rate, the sum over designs of each one's median over
orders), (2) it makes no more wrong calls in all and (3) no more asks in all: pairwise in both
settings; many-arm against the pairwise screens it replaces. The realized loss at rate r: for each
arm whose call resolved at r (an ask answered at r from the arm's own means) differs from the
whole design's call at r, |mu - r x extra|; plus r / FUTURE x the GPU-seconds spent. Wrong calls
and asks by AsksApart (an ask is never wrong; a definite call differing from the whole design's,
an ask included, is). Asks costing 0 changes how the new target weighs asking, not (3).
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cache
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple, Protocol
from unittest import mock

import numpy as np
import pandas as pd
from epistemetry.design.decide import (ADOPT, ASK, KEEP, AdoptKeepAsk, AdoptKeepAskByStake,
                                      DrawTarget, Recorded, StandardErrors, StopRule,
                                      ValueOfInformation)
from epistemetry.design.replay import (STOP, AsksApart, Bootstrap, Decision, Design, DropRule,
                                       EveryArm, Judgement, NoVeto, Order, Policy, Seen, Step,
                                       StopOf, Truth, ValueOfMoreUnits, VetoBelow, replay)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import seqscreen as ss  # noqa: E402

DESIGNS = HERE / "epireplay_designs.json"      # committed: the designs, frozen
JOBS = HERE / "epireplay4_jobs"                # git-ignored, a parquet a job: a rerun resumes
ROWS = HERE / "epireplay4.parquet"             # git-ignored: every job's rows, gathered by report
SCREENS = HERE / "epireplay4_screens.parquet"  # git-ignored: `report`'s judged screens
FIRST = HERE / "epireplay_bracket.parquet"     # the first replay's runs (records, `check`)
FIRST_JOBS = HERE / "epireplay_bracket_jobs"   # ... as its jobs wrote them
THIRD = HERE / "epireplay3.parquet"            # the third replay's runs (860e8960, records)
THIRD_JOBS = HERE / "epireplay3_jobs"          # ... as its jobs wrote them
BRACKET = np.geomspace(*ss.RATES, ss.RATE_POINTS)   # seqscreen's stop and call
SENSITIVITY = np.geomspace(*ss.RATES, 9)       # nine rates, evenly in log rate across RATES
JUDGED = np.geomspace(*ss.RATES, 3)            # the bracket's ends and its log-midpoint
PRICE_RATE = float(BRACKET.mean())             # seqscreen prices its stop's GPU-seconds at it
CODE = {"keep": KEEP, "ask": ASK, "adopt": ADOPT}   # seqscreen's calls as AdoptKeepAsk's
ORDERS = 100        # orders a design, in DESIGNS
SUMMED = "seqscreen per variant"   # a many-arm design's pairwise seqscreen screens, summed


class MonteCarlo(NamedTuple):
    """epistemetry's draws: outer draws of the next blocks, inner draws of the rest given them,
    batch-means blocks, and the draws a call is made from (seqscreen's value uses DRAWS paths)."""
    outer: int
    inner: int
    blocks: int
    draws: int


MC = MonteCarlo(outer=ss.DRAWS, inner=200, blocks=20, draws=ss.DRAWS)


# ---- the designs ----------------------------------------------------------------------------

class Arm(NamedTuple):
    name: str
    spec: str       # its rows as seqscreen named them: a record, read by nothing here


@dataclass(frozen=True, eq=False, kw_only=True)
class Plan:
    """A design: `arms` against `base` on `blocks` (the blocks every arm has, less `dropped`, the
    free-first rows), each arm's Lens A gain over the base and GPU-seconds, the base's, and the
    orders every policy runs. Not comparable with `==`: it holds arrays."""
    name: str
    card: str
    which: str
    many: bool
    base: Arm
    arms: tuple[Arm, ...]
    blocks: tuple[str, ...]
    dropped: tuple[str, ...]
    gain: np.ndarray            # (u, k)
    seconds: np.ndarray         # (u, k)
    base_seconds: np.ndarray    # (u,)
    orders: tuple[np.ndarray, ...]

    @property
    def units(self) -> int:
        return len(self.blocks)

    @property
    def mu(self) -> np.ndarray:
        return self.gain.mean(0)

    @property
    def extra(self) -> np.ndarray:
        return (self.seconds - self.base_seconds[:, None]).mean(0)

    def arm(self, name: str) -> int:
        return [a.name for a in self.arms].index(name)

    def column(self, j: int) -> Plan:
        """Arm j's pairwise screen against the base."""
        return replace(self, arms=(self.arms[j],), gain=self.gain[:, [j]],
                       seconds=self.seconds[:, [j]])

    def design(self) -> Design:
        """epistemetry's: each arm's gain and extra GPU-seconds (the targets' pair), what each arm
        cost on each block, and the base's runs, which exist: 0."""
        return Design(outcomes=np.stack([self.gain, self.seconds - self.base_seconds[:, None]],
                                        axis=2),
                      costs=self.seconds, base_costs=np.zeros(self.units))


def _plan(d: dict) -> Plan:
    """One design of DESIGNS, checked: the JSON boundary, every key read by name."""
    gain = np.ascontiguousarray(np.array(d["gain"], dtype=float).T)
    seconds = np.ascontiguousarray(np.array(d["seconds"], dtype=float).T)
    base_seconds = np.array(d["base_seconds"], dtype=float)
    arms = tuple(Arm(a["name"], a["spec"]) for a in d["arms"])
    blocks = tuple(d["blocks"])
    u, k = len(blocks), len(arms)
    orders = tuple(np.array(o, dtype=int) for o in d["orders"])
    if gain.shape != (u, k) or seconds.shape != (u, k) or base_seconds.shape != (u,):
        raise ValueError(f"{d['name']}: shapes {gain.shape} {seconds.shape} {base_seconds.shape}"
                         f" for {u} blocks and {k} arms")
    if len(orders) != ORDERS or any(sorted(o.tolist()) != list(range(u)) for o in orders):
        raise ValueError(f"{d['name']}: {ORDERS} orders of its {u} blocks expected")
    return Plan(name=d["name"], card=d["card"], which=d["which"], many=d["many"],
                base=Arm(d["base"]["name"], d["base"]["spec"]), arms=arms, blocks=blocks,
                dropped=tuple(d["dropped"]), gain=gain, seconds=seconds,
                base_seconds=base_seconds, orders=orders)


@cache
def plans() -> dict[str, Plan]:
    """DESIGNS, by name, in its order: the pairwise designs, then the many-arm ones."""
    return {p.name: p for p in map(_plan, json.loads(DESIGNS.read_text())["designs"])}


def pairwise() -> list[Plan]:
    return [p for p in plans().values() if not p.many]


def many_arm() -> list[Plan]:
    return [p for p in plans().values() if p.many]


# ---- the veto settings ----------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class Setting:
    """Where the collapse veto applies: the harness's drop rule, for every policy and the truth,
    and seqscreen's own COLLAPSE inside the process (inf: off)."""
    veto: bool
    drop: DropRule
    collapse: float


EVERYWHERE = Setting(veto=True, drop=VetoBelow(threshold=-ss.COLLAPSE, column=0, call=KEEP),
                     collapse=ss.COLLAPSE)
NOWHERE = Setting(veto=False, drop=NoVeto(), collapse=math.inf)


@contextlib.contextmanager
def collapse_at(level: float) -> Iterator[None]:
    """seqscreen's veto at `level` (inf: off) inside the block, restored after."""
    kept = ss.COLLAPSE
    ss.COLLAPSE = level
    try:
        yield
    finally:
        ss.COLLAPSE = kept


# ---- seqscreen as a policy, and the truth ---------------------------------------------------

def seq_call(s: ss.Screen) -> str:
    """seqscreen's call on all of `s` over the bracket: stop_at with the floor at its blocks
    returns at the last (the collapse or the call), drawing no value, so its generator is never
    read."""
    stop = ss.stop_at(s, rates=BRACKET, future=ss.FUTURE, floor=len(s.d),
                      rng=np.random.default_rng(0))
    assert stop is not None
    return stop.call


@dataclass(frozen=True, eq=False, kw_only=True)
class SeqScreen:
    """seqscreen.py, unchanged, as an epistemetry Policy: `next` (one arm) is stop_at on the
    blocks seen (the base's times on every block of the order), replayed from the first block with
    the order's generator (a copy of the run's stream, which it never draws from itself), as
    seqscreen.main replays it, and a block more while it runs on, no rule judged; `calls` is
    stop_at's call on each arm's blocks seen."""
    base_seconds: np.ndarray    # (u,)
    floor: int

    def next(self, seen: Seen, *, rng: np.random.Generator) -> Judgement:
        (arm,) = seen.arms
        order = np.concatenate([seen.base, seen.upcoming])
        stop = ss.stop_at(ss.Screen(arm.outcomes[:, 0], arm.costs, self.base_seconds[order]),
                          rates=BRACKET, future=ss.FUTURE, floor=self.floor,
                          rng=copy.deepcopy(rng))
        return Judgement.unjudged(STOP if stop is not None else Step(units=1, arms=seen.live))

    def calls(self, seen: Seen, *, rng: np.random.Generator) -> np.ndarray:
        return np.array([[CODE[seq_call(ss.Screen(arm.outcomes[:, 0], arm.costs,
                                                  self.base_seconds[arm.units]))]]
                         for arm in seen.arms])


def truth_of(plan: Plan, setting: Setting) -> Truth:
    """The one truth of `plan` in `setting`: seqscreen's call over the bracket on all its blocks,
    the drop rule's on a collapse (`Truth.of`, the drop rule travelling with it)."""
    with collapse_at(setting.collapse):
        return Truth.of(plan.design(), SeqScreen(base_seconds=plan.base_seconds,
                                                 floor=ss.MIN_BLOCKS),
                        rng=np.random.default_rng(ss.ORDER_SEED), drop=setting.drop)


# ---- epistemetry's policy: the cost model, the prices, the targets --------------------------

@dataclass(frozen=True, eq=False, kw_only=True)
class RatioToBase:
    """seqscreen's cost model as an epistemetry CostModel: the units to come cost the base's
    GPU-seconds on them times the ratio of what the units seen were charged to the base's seconds
    on them: an arm's, b's time to a's (seqscreen's); the base's, its own charge per second of its
    time (0 here, its runs existing)."""
    base_seconds: np.ndarray    # (u,)

    def predict(self, arm: int, seen: np.ndarray, seen_costs: np.ndarray,
                upcoming: np.ndarray) -> float:
        return self._ratio(seen, seen_costs, upcoming)

    def predict_base(self, seen: np.ndarray, seen_costs: np.ndarray,
                     upcoming: np.ndarray) -> float:
        return self._ratio(seen, seen_costs, upcoming)

    def _ratio(self, seen: np.ndarray, seen_costs: np.ndarray, upcoming: np.ndarray) -> float:
        return float(seen_costs.sum() / self.base_seconds[seen].sum()
                     * self.base_seconds[upcoming].sum())


def stake(outcomes: np.ndarray) -> float:
    """The Lens A a wrong call on one arm costs at its means so far (its gain and extra
    GPU-seconds over the units it observed), at PRICE_RATE: |gain - PRICE_RATE x extra|."""
    return float(np.abs(outcomes[:, :1].mean(0) - PRICE_RATE * outcomes[:, 1:].mean(0))[0])


def priced(at_stake: float) -> StopRule:
    """seqscreen's price of a GPU-second (PRICE_RATE / FUTURE) over the Lens A at stake: the
    exchange rate between a GPU-second and AdoptKeepAsk's risk (a probability of another call); no
    Monte Carlo margin, as seqscreen has none."""
    return ValueOfInformation(price=PRICE_RATE / ss.FUTURE / at_stake,
                              margin=StandardErrors(k=0.0))


@dataclass(frozen=True)
class OwnStake:
    """Each arm priced by its own stake, as seqscreen prices each variant on its own."""

    @property
    def name(self) -> str:
        return "own price"

    def of(self, seen: Seen, *, arm: int) -> StopRule:
        return priced(stake(seen.arms[arm].outcomes))


@dataclass(frozen=True)
class SharedStake:
    """The second replay's price (22394aa2, when the layer took one rule for every arm): the live
    arms, which share one prefix, priced together by the mean of their stakes. On one arm it is the
    own price. In the many-arm screens it made one wrong call and two asks fewer than the own price
    in 500 runs, and more loss at every rate (NOTES), so it stays beside it."""

    @property
    def name(self) -> str:
        return "shared price"

    def of(self, seen: Seen, *, arm: int) -> StopRule:
        return priced(float(np.mean([stake(seen.arms[j].outcomes) for j in sorted(seen.live)])))


@dataclass(frozen=True)
class Unstaked:
    """The migration guard's fault: the 0-1 risk priced as the realized loss is, PRICE_RATE /
    FUTURE per GPU-second, the stake left out."""

    @property
    def name(self) -> str:
        return "unstaked price"

    def of(self, seen: Seen, *, arm: int) -> StopRule:
        return ValueOfInformation(price=PRICE_RATE / ss.FUTURE, margin=StandardErrors(k=0.0))


@dataclass(frozen=True, kw_only=True)
class ZeroOne:
    """The third replay's target: AdoptKeepAsk over [low, high] (Lens A per GPU-second), a 0-1
    risk, each arm priced through the Lens A at stake by `price`."""
    low: float
    high: float
    price: OwnStake | SharedStake | Unstaked

    @property
    def kind(self) -> str:
        return "adopt/keep/ask"

    @property
    def priced(self) -> str:
        return self.price.name

    def draw_target(self) -> DrawTarget:
        return AdoptKeepAsk(low=self.low, high=self.high)

    def stop_of(self) -> StopOf:
        return self.price

    def record(self) -> dict[str, object]:
        return {"target": "AdoptKeepAsk", "low": self.low, "high": self.high,
                "price": self.price.name, "price_rate": PRICE_RATE, "future": ss.FUTURE}


@dataclass(frozen=True, kw_only=True)
class ByStake:
    """The realized loss as the target: AdoptKeepAskByStake over `rates` (Lens A per GPU-second,
    increasing) weighted by `weights`, its value in Lens A, so every arm is priced at
    sum_r w_r r / `future` per GPU-second, `future` the block runs a decision governs."""
    rates: tuple[float, ...]
    weights: tuple[float, ...]
    future: float

    @property
    def kind(self) -> str:
        return "by stake"

    @property
    def priced(self) -> str:
        return f"{len(self.rates)} rates"

    @property
    def price(self) -> float:
        return float(np.dot(self.weights, self.rates)) / self.future

    def draw_target(self) -> DrawTarget:
        return AdoptKeepAskByStake(rates=self.rates, weights=self.weights)

    def stop_of(self) -> StopOf:
        return EveryArm(ValueOfInformation(price=self.price, margin=StandardErrors(k=0.0)))

    def record(self) -> dict[str, object]:
        return {"target": "AdoptKeepAskByStake", "rates": list(self.rates),
                "weights": list(self.weights), "future": self.future, "price": self.price}


def equal(n: int) -> tuple[float, ...]:
    """n equal weights."""
    return (1.0 / n,) * n


@dataclass(frozen=True, eq=False)
class KeepAll:
    """The per-arm drop guard's fault: every live arm observes while any goes on, so an arm the
    inner policy drops keeps observing (its judgement's rules and readings kept)."""
    inner: Policy

    def next(self, seen: Seen, *, rng: np.random.Generator) -> Judgement:
        j = self.inner.next(seen, rng=rng)
        if not j.step.arms:
            return j
        return Judgement(step=Step(units=j.step.units, arms=seen.live), rules=j.rules,
                         readings=j.readings)

    def calls(self, seen: Seen, *, rng: np.random.Generator) -> np.ndarray:
        return self.inner.calls(seen, rng=rng)


def bare(policy: Policy) -> Policy:
    """No fault: the policy as made."""
    return policy


# ---- the makers -----------------------------------------------------------------------------

class Maker(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def floor(self) -> int: ...

    def policy(self, plan: Plan) -> Policy: ...

    def stream(self, o: int) -> np.random.Generator: ...

    def record(self) -> dict[str, object]: ...


@dataclass(frozen=True, kw_only=True)
class SeqMaker:
    floor: int

    @property
    def name(self) -> str:
        return "seqscreen"

    def policy(self, plan: Plan) -> Policy:
        if len(plan.arms) != 1:
            raise ValueError(f"seqscreen screens one arm; {plan.name} has {len(plan.arms)}")
        return SeqScreen(base_seconds=plan.base_seconds, floor=self.floor)

    def stream(self, o: int) -> np.random.Generator:
        """seqscreen.main's generator for order o."""
        return np.random.default_rng([ss.ORDER_SEED, o])

    def record(self) -> dict[str, object]:
        return {"policy": "seqscreen", "floor": self.floor, "rates": BRACKET.tolist(),
                "future": ss.FUTURE, "collapse": ss.COLLAPSE}


@dataclass(frozen=True, kw_only=True)
class EpiMaker:
    step: int
    floor: int
    target: ZeroOne | ByStake
    mc: MonteCarlo

    @property
    def name(self) -> str:
        return f"{self.target.kind} step {self.step}, {self.target.priced}"

    def policy(self, plan: Plan) -> ValueOfMoreUnits:
        return ValueOfMoreUnits(
            belief=Bootstrap(), functionals=np.eye(2), target=self.target.draw_target(),
            horizons=tuple(range(1, plan.units + 1)), step=self.step,
            cost=RatioToBase(base_seconds=plan.base_seconds), stop=self.target.stop_of(),
            floor=self.floor, outer=self.mc.outer, inner=self.mc.inner, blocks=self.mc.blocks,
            draws=self.mc.draws)

    def stream(self, o: int) -> np.random.Generator:
        """Order o's stream as every replay of these designs has drawn epistemetry's runs from it:
        the second spawned from [ORDER_SEED, o] (the first was its truth's, when each order had
        its own)."""
        return np.random.default_rng([ss.ORDER_SEED, o]).spawn(2)[1]

    def record(self) -> dict[str, object]:
        return {"policy": "ValueOfMoreUnits", "step": self.step, "floor": self.floor,
                **self.target.record(), "mc": self.mc._asdict()}


def zero_one(*, price: OwnStake | SharedStake | Unstaked, floor: int,
             mc: MonteCarlo) -> EpiMaker:
    """The third replay's policy: step 1, AdoptKeepAsk over the bracket's ends at `price`."""
    return EpiMaker(step=1, floor=floor,
                    target=ZeroOne(low=ss.RATES[0], high=ss.RATES[1], price=price), mc=mc)


def by_stake(*, rates: np.ndarray, weights: tuple[float, ...], future: float, floor: int,
             mc: MonteCarlo) -> EpiMaker:
    """The realized loss as the target: step 1, AdoptKeepAskByStake over `rates` and
    `weights`, priced at their weighted mean over `future`."""
    return EpiMaker(step=1, floor=floor, target=ByStake(
        rates=tuple(float(r) for r in rates), weights=weights, future=future), mc=mc)


def stake_maker(*, floor: int, mc: MonteCarlo) -> EpiMaker:
    """(c): over BRACKET, equally weighted, priced at PRICE_RATE / FUTURE."""
    return by_stake(rates=BRACKET, weights=equal(len(BRACKET)), future=ss.FUTURE, floor=floor,
                    mc=mc)


def nine_maker(*, floor: int, mc: MonteCarlo) -> EpiMaker:
    """(d): over SENSITIVITY, equally weighted, priced at their mean over FUTURE."""
    return by_stake(rates=SENSITIVITY, weights=equal(len(SENSITIVITY)), future=ss.FUTURE,
                    floor=floor, mc=mc)


# ---- runs -----------------------------------------------------------------------------------

def one_each(calls: np.ndarray, plan: Plan) -> list[int]:
    """A run's calls, (arms, 1), as one call per arm."""
    return [int(c) for c in np.asarray(calls).reshape(len(plan.arms))]


def decision_record(d: Decision) -> dict[str, object]:
    """A decision as the rows keep it (JSON): each judged arm's rule as its own record."""
    return dict(at=d.at, arm_units=list(d.arm_units), live=sorted(d.live), units=d.step.units,
                arms=sorted(d.step.arms), rules={str(j): r.record() for j, r in d.rules.items()},
                readings={str(j): dict(v) for j, v in d.readings.items()})


def run_orders(plan: Plan, *, maker: SeqMaker | EpiMaker, setting: Setting,
               orders: Sequence[int], given: Mapping[int, np.ndarray],
               outer: Callable[[Policy], Policy]) -> list[dict]:
    """`plan`'s `orders` through one epistemetry replay in `setting`, each order with its maker's
    stream, scored with AsksApart against the one truth (`truth_of`); the policy inside `outer` (a
    guard's fault, or `bare`). `given` hands an order another permutation (the same-orders guard's
    fault). Each row's order is the one its run carries (`Run.order`)."""
    design = plan.design()
    policy = outer(maker.policy(plan))
    handed = [Order(units=given[o] if o in given else plan.orders[o], rng=maker.stream(o))
              for o in orders]
    the = truth_of(plan, setting)
    with collapse_at(setting.collapse):
        rep = replay(design, policy, orders=handed, truth=the, scoring=AsksApart(ask=ASK))
    identity = json.dumps(maker.record())
    return [dict(plan=plan.name, arms=",".join(a.name for a in plan.arms), policy=maker.name,
                 identity=identity, veto=setting.veto, order=o, units=run.units,
                 arm_units=list(run.arm_units), cost=run.cost, calls=one_each(run.calls, plan),
                 full=one_each(rep.truth.calls, plan), wrong=run.wrong, asks=run.asks,
                 vetoed=sorted(run.vetoed), given=run.order.units.tolist(),
                 decisions=json.dumps([decision_record(d) for d in run.decisions]))
            for o, run in zip(orders, rep.runs, strict=True)]


class Decided(NamedTuple):
    """A row's decision, from `Run.decisions` (JSON in the rows): each judged arm's rule (every
    epistemetry policy here stops on ValueOfInformation) and the reading that decided it."""
    at: int
    arm_units: tuple[int, ...]
    live: frozenset[int]
    units: int
    arms: frozenset[int]
    rules: Mapping[int, ValueOfInformation]
    readings: Mapping[int, Recorded]


def rule_of(record: Mapping[str, object]) -> ValueOfInformation:
    """A judged arm's rule from its record (`StopRule.record`), read by name."""
    if record["stop"] != "value_of_information":
        raise ValueError(f"a rule of another kind: {record}")
    price, k = record["price"], record["margin_k"]
    assert isinstance(price, float) and isinstance(k, float)
    return ValueOfInformation(price=price, margin=StandardErrors(k=k))


def decisions_of(row: pd.Series | tuple) -> list[Decided]:
    return [Decided(at=d["at"], arm_units=tuple(d["arm_units"]), live=frozenset(d["live"]),
                    units=d["units"], arms=frozenset(d["arms"]),
                    rules={int(j): rule_of(r) for j, r in d["rules"].items()},
                    readings={int(j): r for j, r in d["readings"].items()})
            for d in json.loads(row.decisions)]


class Job(NamedTuple):
    plan: str
    arm: int | None     # a many-arm design's variant, screened alone; None: the design whole
    maker: SeqMaker | EpiMaker
    setting: Setting


def job_plan(job: Job) -> Plan:
    p = plans()[job.plan]
    return p if job.arm is None else p.column(job.arm)


def job_file(job: Job) -> Path:
    arm = "all" if job.arm is None else plans()[job.plan].arms[job.arm].name
    name = (f"{job.plan}_{arm}_{job.maker.name}_f{job.maker.floor}_"
            f"{'veto' if job.setting.veto else 'none'}")
    return JOBS / (re.sub(r"[^A-Za-z0-9+.-]", "_", name) + ".parquet")


def rows_of(job: Job) -> pd.DataFrame:
    """`job`'s rows: from its file, or run now and kept."""
    f = job_file(job)
    if not f.exists():
        JOBS.mkdir(exist_ok=True)
        pd.DataFrame(run_orders(job_plan(job), maker=job.maker, setting=job.setting,
                                orders=range(ORDERS), given={}, outer=bare)).to_parquet(f)
    return pd.read_parquet(f)


def groups(*, floor: int, mc: MonteCarlo) -> dict[str, list[Job]]:
    """The replay's jobs by group. Pairwise, in both settings; many-arm (no unit of any is below
    -COLLAPSE, so the settings agree: NOWHERE, as the earlier replays ran them):
    - seqscreen: every pairwise design, and every many-arm design once per variant;
    - zero-one (b): the third replay's, at its own price, pairwise and many-arm, and many-arm at
      the shared price;
    - stake (c): the realized loss over BRACKET, pairwise and many-arm;
    - sensitivity (d): the realized loss over SENSITIVITY, pairwise."""
    own = zero_one(price=OwnStake(), floor=floor, mc=mc)
    both = zero_one(price=SharedStake(), floor=floor, mc=mc)
    stake_, nine = stake_maker(floor=floor, mc=mc), nine_maker(floor=floor, mc=mc)
    settings = (NOWHERE, EVERYWHERE)

    def whole(maker: SeqMaker | EpiMaker, *, many: bool) -> list[Job]:
        out = [Job(p.name, None, maker, s) for s in settings for p in pairwise()]
        return out + ([Job(p.name, None, maker, NOWHERE) for p in many_arm()] if many else [])

    seq = SeqMaker(floor=floor)
    return {"seqscreen": [Job(p.name, None, seq, s) for s in settings for p in pairwise()]
            + [Job(p.name, j, seq, NOWHERE) for p in many_arm() for j in range(len(p.arms))],
            "zero-one": whole(own, many=True) + [Job(p.name, None, both, NOWHERE)
                                                  for p in many_arm()],
            "stake": whole(stake_, many=True),
            "sensitivity": whole(nine, many=False)}


GROUPS = ("seqscreen", "zero-one", "stake", "sensitivity")


def every_job() -> list[Job]:
    g = groups(floor=ss.MIN_BLOCKS, mc=MC)
    return [j for name in GROUPS for j in g[name]]


def _priority(job: Job) -> tuple[int, int]:
    """The longest first: the per-arm screens, then epistemetry's pairwise, then seqscreen;
    within, the most block-arms."""
    p = job_plan(job)
    kind = (0 if isinstance(job.maker, EpiMaker) and p.many and job.arm is None else
            1 if isinstance(job.maker, EpiMaker) else 2)
    return kind, -p.units * len(p.arms)


def _one(job: Job) -> str:
    """`job` run and written, unless another run has claimed it: a claim file beside its rows,
    made atomically and removed when they are written, so runs on overlapping groups share the
    jobs (one killed leaves its claims, to remove by hand)."""
    arm = "all" if job.arm is None else job.arm
    what = f"{job.plan} {job.maker.name} {arm} veto {job.setting.veto}"
    claim = job_file(job).with_suffix(".claim")
    JOBS.mkdir(exist_ok=True)
    try:
        os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return f"{what}: claimed by another run"
    start = time.monotonic()
    try:
        rows_of(job)
    finally:
        claim.unlink()
    return f"{what} ({(time.monotonic() - start) / 60:.1f} min)"


def run(names: Sequence[str], workers: int) -> None:
    """The jobs of the groups `names` not yet written, on `workers` processes (a count that
    changes no result)."""
    g = groups(floor=ss.MIN_BLOCKS, mc=MC)
    chosen = [j for name in names for j in g[name]]
    todo = sorted((j for j in chosen if not job_file(j).exists()), key=_priority)
    print(f"{len(chosen) - len(todo)} of {len(chosen)} jobs done before", flush=True)
    with Pool(workers) as pool:
        for k, done in enumerate(pool.imap_unordered(_one, todo)):
            print(f"{k + 1}/{len(todo)} {done}", flush=True)


# ---- the data file against the first replay -------------------------------------------------

def whole_call(p: Plan, j: int, *, veto: bool) -> int:
    """Arm j's call on all of `p`'s blocks over the bracket; keep on a collapse under the veto."""
    if veto and np.any(p.gain[:, j] < -ss.COLLAPSE):
        return KEEP
    return CODE[ss.call(float(p.mu[j]), float(p.extra[j]), BRACKET)]


def seq_replay(p: Plan, j: int, o: int, *, setting: Setting, floor: int) -> tuple[int, int, float]:
    """seqscreen's own replay of arm j on order o (stop_at on the ordered screen, the order's
    generator, as seqscreen.main runs it): blocks, call and GPU-seconds."""
    i = p.orders[o]
    with collapse_at(setting.collapse):
        stop = ss.stop_at(ss.Screen(p.gain[i, j], p.seconds[i, j], p.base_seconds[i]),
                          rates=BRACKET, future=ss.FUTURE, floor=floor,
                          rng=np.random.default_rng([ss.ORDER_SEED, o]))
    assert stop is not None     # every block run: it stops by the last
    return stop.n, CODE[stop.call], float(p.seconds[i[:stop.n], j].sum())


def stored(rows_path: Path, jobs_dir: Path) -> pd.DataFrame:
    """An earlier replay's runs, checked to be rows of its jobs (whose directory may also hold a
    guard's faulted jobs, which its run did not gather)."""
    rows = pd.read_parquet(rows_path)
    jobs_ = pd.concat([pd.read_parquet(f) for f in sorted(jobs_dir.glob("*.parquet"))],
                      ignore_index=True)

    def as_text(d: pd.DataFrame) -> list[tuple[str, ...]]:
        """Each row's values as text, a list column's by its values (an all-empty column comes
        back from parquet as object arrays)."""
        def text(v: object) -> str:
            return repr(v.tolist()) if isinstance(v, np.ndarray) else repr(v)
        return list(zip(*(d[c].map(text) for c in rows.columns), strict=True))

    written = set(as_text(jobs_))
    if not all(r in written for r in as_text(rows)):
        raise ValueError(f"{rows_path.name} is not rows of its jobs")
    return rows


def check() -> bool:
    """DESIGNS against the first replay's stored runs: every row's order is the design's, every
    whole-design call is the data's, and seqscreen's replay of every order (blocks, call,
    GPU-seconds) is the stored one."""
    first = stored(FIRST, FIRST_JOBS)
    print(f"{len(first)} stored runs ({FIRST.name} = {FIRST_JOBS.name}/'s "
          f"{len(list(FIRST_JOBS.glob('*.parquet')))} jobs)")
    orders = full = seq = seq_n = 0
    for g in first.itertuples():
        p = plans()[g.plan]
        cols = [p.arm(a) for a in g.arms.split(",")]
        orders += np.array_equal(np.asarray(g.given), p.orders[g.order])
        full += list(g.full) == [whole_call(p, j, veto=g.veto) for j in cols]
        if g.policy == "seqscreen":
            (j,) = cols
            seq_n += 1
            got = seq_replay(p, j, g.order, setting=EVERYWHERE if g.veto else NOWHERE,
                             floor=ss.MIN_BLOCKS)
            seq += got == (g.units, int(g.calls[0]), g.cost)
    n = len(first)
    print(f"  orders: {orders} of {n} runs were handed the design's order")
    print(f"  whole-design calls: {full} of {n} runs' equal the data's")
    print(f"  seqscreen: {seq} of {seq_n} runs (blocks, call, GPU-seconds) replayed exactly")
    ok = orders == n and full == n and seq == seq_n
    for p in plans().values():
        calls = " ".join(f"{a.name} {ss.call(float(m), float(e), BRACKET)}"
                         for a, m, e in zip(p.arms, p.mu, p.extra, strict=True))
        print(f"  {p.name:14s} {p.units:2d} blocks ({len(p.dropped)} dropped), {p.card}: {calls};"
              f" worst block {p.gain.min():+.3f}")
    print(f"check: {'pass' if ok else 'FAIL'}")
    return ok


# ---- the guards -----------------------------------------------------------------------------

class Agreement(NamedTuple):
    """Two sets of runs on the same orders, order by order."""
    runs: int       # blocks, calls and GPU-seconds equal
    scored: int     # wrong calls, asks and the truth equal
    calls: int      # calls differing
    blocks: int     # blocks differing


def same_runs(a: pd.DataFrame, b: pd.DataFrame) -> Agreement:
    a, b = a.sort_values("order"), b.sort_values("order")
    if a.order.tolist() != b.order.tolist():
        raise ValueError("the runs are not on the same orders")
    pairs = list(zip(a.itertuples(), b.itertuples(), strict=True))
    return Agreement(
        runs=sum((x.units, list(x.calls), x.cost) == (y.units, list(y.calls), y.cost)
                 for x, y in pairs),
        scored=sum((x.wrong, x.asks, list(x.full)) == (y.wrong, y.asks, list(y.full))
                   for x, y in pairs),
        calls=sum(list(x.calls) != list(y.calls) for x, y in pairs),
        blocks=sum(x.units != y.units for x, y in pairs))


def own_prices(rows: pd.DataFrame) -> tuple[int, int]:
    """At every judged decision of every run, each judged arm's price as its run recorded it
    (`Decision.rules`) against its own stake, computed apart from the policy from the design and
    the arm's prefix then (`Decision.arm_units`): (prices that are the arm's own, prices where the
    arm's own and the next arm's differ, so a wrong one shows)."""
    own = differ = 0
    for r in rows.itertuples():
        p, order = plans()[r.plan], np.asarray(r.given)
        extra = p.seconds - p.base_seconds[:, None]
        for d in decisions_of(r):
            for j, rule in d.rules.items():
                want = [PRICE_RATE / ss.FUTURE
                        / abs(p.gain[order[:d.arm_units[i]], i].mean()
                              - PRICE_RATE * extra[order[:d.arm_units[i]], i].mean())
                        for i in (j, (j + 1) % len(p.arms))]
                if not math.isclose(want[0], want[1], rel_tol=1e-9):
                    differ += 1
                    own += math.isclose(rule.price, want[0], rel_tol=1e-9)
    return own, differ


def guard_migration(*, price: OwnStake | Unstaked) -> bool:
    """The migration to 674dba4. With AdoptKeepAsk, the third replay's runs reproduce exactly
    (blocks, calls, GPU-seconds; and the scoring: wrong calls, asks, the truth): its 2,600 pairwise
    step-1 runs at the own price (both settings) and its 500 many-arm runs at each price, from this
    replay's jobs; and so do x2/p64's first 20 orders and 2/p64's first 10, run now at the price
    `price`. Every price the own price gave a live arm, read from the runs' decisions, is that
    arm's own stake's (2/p64's now and every many-arm run of the own price). Fault `unstaked`: the
    runs now priced as the realized loss is, the stake left out."""
    third = stored(THIRD, THIRD_JOBS)
    own = zero_one(price=OwnStake(), floor=ss.MIN_BLOCKS, mc=MC)
    both = zero_one(price=SharedStake(), floor=ss.MIN_BLOCKS, mc=MC)
    ok = True
    for name, maker, plans_, settings in (
            ("pairwise, step 1 at its own price", own, pairwise(), (NOWHERE, EVERYWHERE)),
            ("many-arm, step 1 at its own price", own, many_arm(), (NOWHERE,)),
            ("many-arm, step 1 at the shared price", both, many_arm(), (NOWHERE,))):
        runs = scored = total = 0
        for setting in settings:
            for p in plans_:
                got = rows_of(Job(p.name, None, maker, setting))
                ref = third[(third.plan == p.name) & (third.veto == setting.veto)
                            & (third.policy == maker.name)]
                a = same_runs(got, ref)
                runs, scored, total = runs + a.runs, scored + a.scored, total + len(ref)
                ok &= a.runs == a.scored == len(ref) == ORDERS
        print(f"  {name}: {runs} of {total} runs the third replay's (blocks, calls, GPU-seconds),"
              f" {scored} scored alike")
    m = zero_one(price=price, floor=ss.MIN_BLOCKS, mc=MC)
    for p, orders in ((plans()["x2/p64"], range(20)), (plans()["2/p64"], range(10))):
        got = pd.DataFrame(run_orders(p, maker=m, setting=NOWHERE, orders=orders, given={},
                                      outer=bare))
        ref = third[(third.plan == p.name) & ~third.veto & (third.policy == own.name)
                    & third.order.isin(orders)]
        a = same_runs(got, ref)
        print(f"  {p.name}, orders {orders[0]}-{orders[-1]}, now ({m.name}): {a.runs} of {len(ref)}"
              f" runs the third replay's ({a.calls} with other calls, {a.blocks} other blocks)")
        ok &= a.runs == len(ref) == len(orders)
        if p.many:
            o, d = own_prices(got)
            print(f"    prices: {o} of {d} (where the arm's own and the next arm's differ) the "
                  "arm's own")
            ok &= o == d > 0
    o, d = own_prices(pd.concat([rows_of(Job(p.name, None, own, NOWHERE)) for p in many_arm()],
                                ignore_index=True))
    print(f"  prices, every many-arm run at the own price: {o} of {d} the arm's own")
    return ok & (o == d > 0)


def main_line(p: Plan) -> str:
    """seqscreen.main's summary line of `p`, its reader handed DESIGNS' rows (the base's Lens A 0,
    so a variant's is its gain; an ask's question, which divides by the base's mean, silenced)."""
    (arm,) = p.arms
    per = {p.base.name: pd.DataFrame({"perm": np.zeros(p.units), "t": p.base_seconds},
                                     index=list(p.blocks)),
           arm.name: pd.DataFrame({"perm": p.gain[:, 0], "t": p.seconds[:, 0]},
                                  index=list(p.blocks))}
    out = io.StringIO()
    with (mock.patch.object(ss, "specs_of", lambda which, budget, specs: per),
          mock.patch.object(ss, "question", lambda *args: ""), contextlib.redirect_stdout(out)):
        ss.main(p.which, 0.05, [p.base.spec, arm.spec], [f"{arm.name}/{p.base.name}"])
    return out.getvalue().splitlines()[1]


def guard_wrapper(*, floor: int) -> bool:
    """seqscreen as a Policy (its floor `floor`) reproduces seqscreen's own replay, order by order
    (blocks, call, GPU-seconds), on every pairwise design in both settings, and seqscreen.main's
    printed summary on the designs without dropped blocks. Fault `floor`: the wrapper's floor one
    below."""
    ok, total, match = True, 0, 0
    for p in pairwise():
        same = []
        for setting in (NOWHERE, EVERYWHERE):
            got = rows_of(Job(p.name, None, SeqMaker(floor=floor), setting)).sort_values("order")
            same += [(g.units, int(g.calls[0]), g.cost)
                     == seq_replay(p, 0, g.order, setting=setting, floor=ss.MIN_BLOCKS)
                     for g in got.itertuples()]
            if setting.veto and not p.dropped:
                line = main_line(p)
                m = re.search(r"median (\d+) blocks \((\d+)-(\d+)\).*calls (\{.*\}), (\d+) of",
                              line)
                assert m is not None, line
                twenty = got[got.order < ss.ORDERS]     # main replays its ORDERS
                n = twenty.units.to_numpy()
                said = [{KEEP: "keep", ASK: "ask", ADOPT: "adopt"}[int(c[0])]
                        for c in twenty.calls]
                calls = {c: said.count(c) for c in sorted(set(said))}
                same.append(m.groups()[:4] == (str(int(np.median(n))), str(n.min()),
                                               str(n.max()), str(calls)))
        print(f"  {p.name:14s} {sum(same)} of {len(same)} match")
        total, match = total + len(same), match + sum(same)
        ok &= all(same)
    print(f"  in all {match} of {total}")
    return ok


def realized_risk(draws: np.ndarray, *, rates: np.ndarray) -> tuple[float, int]:
    """Outside epistemetry: one arm's posterior draws of (gain, extra GPU-seconds) -> the expected
    loss of the call the draws' means favour at each of `rates` (Lens A per GPU-second; adopt where
    the mean of g - r c is above 0), E|g - r c| over the draws where that call is wrong, averaged
    over the rates with equal weight; and that call over the rates (ADOPT at every one, KEEP at
    none, ASK where they disagree)."""
    g, c = draws[:, 0], draws[:, 1]
    losses, adopted = [], []
    for r in rates:
        v = g - r * c
        adopt = bool(v.mean() > 0.0)
        losses.append(float(np.mean(np.abs(v) * ((v > 0.0) != adopt))))
        adopted.append(adopt)
    call = ADOPT if all(adopted) else KEEP if not any(adopted) else ASK
    return float(np.mean(losses)), call


def guard_target(*, fault: str | None) -> bool:
    """(c)'s target as used. At every judged decision of its runs (x2/p64's and 2/p64's first 5
    orders run now by the maker under test and, without a fault, every run of the `stake` group),
    for each judged arm: its rule, as the run recorded it, is ValueOfInformation at
    PRICE_RATE / FUTURE with no margin; and on the posterior's draws there (the arm's prefix, the
    policy's belief, MC.draws draws) the screen's own risk (the policy's target) equals the risk
    recomputed here (`realized_risk`, seqscreen's five rates in Lens A per GPU-second, equally
    weighted), per arm and summed over the arms, and the target's call equals the one recomputed.
    Faults: `per-hour`, the target's rates (and so its price) in Lens A per GPU-hour;
    `unnormalized`, its weights 1 each."""
    m = stake_maker(floor=ss.MIN_BLOCKS, mc=MC)
    if fault == "per-hour":
        m = by_stake(rates=BRACKET * 3600, weights=equal(len(BRACKET)), future=ss.FUTURE,
                     floor=ss.MIN_BLOCKS, mc=MC)
    elif fault == "unnormalized":
        m = by_stake(rates=BRACKET, weights=(1.0,) * len(BRACKET), future=ss.FUTURE,
                     floor=ss.MIN_BLOCKS, mc=MC)
    try:
        m.policy(plans()["x2/p64"])
    except ValueError as e:
        print(f"  {m.name}: epistemetry refuses the target: {e}")
        return False
    sets = [("x2/p64 and 2/p64, orders 0-4, now", pd.DataFrame(
        [r for p in (plans()["x2/p64"], plans()["2/p64"])
         for r in run_orders(p, maker=m, setting=NOWHERE, orders=range(5), given={},
                             outer=bare)]))]
    if fault is None:
        sets.append(("every run of `stake`", pd.concat(
            [rows_of(j) for j in groups(floor=ss.MIN_BLOCKS, mc=MC)["stake"]],
            ignore_index=True)))
    want_price = PRICE_RATE / ss.FUTURE
    ok = True
    for name, rows in sets:
        n = pairwise_n = many_n = arms = risks = sums = calls = prices = made = 0
        worst = 0.0
        for r in rows.itertuples():
            p, order = plans()[r.plan], np.asarray(r.given)
            made += json.loads(r.identity) == m.record()
            target = m.policy(p).target
            design = p.design()
            for t, d in enumerate(decisions_of(r)):
                if not d.rules:
                    continue
                n += 1
                many_n += p.many
                pairwise_n += not p.many
                theirs_sum = mine_sum = 0.0
                for j, rule in sorted(d.rules.items()):
                    arms += 1
                    prices += (math.isclose(rule.price, want_price, rel_tol=1e-12)
                               and rule.margin.k == 0.0)
                    prefix = order[:d.arm_units[j]]
                    draws = Bootstrap().of(design.outcomes[prefix, j], population=p.units).draw(
                        np.eye(2), np.random.default_rng([ss.ORDER_SEED, r.order, t, j]),
                        draws=MC.draws)
                    theirs = float(target.risk(draws))
                    mine, call = realized_risk(draws, rates=BRACKET)
                    risks += math.isclose(theirs, mine, rel_tol=1e-9, abs_tol=1e-18)
                    calls += int(np.asarray(target.decide(draws)).reshape(-1)[0]) == call
                    if mine > 0:
                        worst = max(worst, abs(theirs - mine) / mine)
                    theirs_sum, mine_sum = theirs_sum + theirs, mine_sum + mine
                sums += math.isclose(theirs_sum, mine_sum, rel_tol=1e-9, abs_tol=1e-18)
        print(f"  {name}: {made} of {len(rows)} runs made by it; {n} judged decisions "
              f"({pairwise_n} pairwise, {many_n} many-arm), {arms} arms: the price {prices} of "
              f"{arms}; the risk {risks} of {arms} (worst relative difference {worst:.1e}), "
              f"summed over arms {sums} of {n}; the call {calls} of {arms}")
        ok &= made == len(rows) and n >= (5 if name.endswith("now") else 100)
        ok &= prices == risks == calls == arms and sums == n
    return ok


def degenerate() -> Plan:
    """p64 against base with p64's gains replaced by the base's: a variant that changes nothing,
    at p64's GPU time."""
    p = plans()["p64/base"]
    return replace(p, gain=np.zeros_like(p.gain))


def guard_degenerate(*, floor: int) -> bool:
    """Every epistemetry policy (the 0-1 target at the own price, which on one arm is the shared
    one, and the realized loss over five and nine rates), in both settings, on a variant equal to
    its base stops at the floor with "keep", every order. Fault `ignored`: the floor set to 1."""
    p = degenerate()
    ok = True
    for m in (zero_one(price=OwnStake(), floor=floor, mc=MC), stake_maker(floor=floor, mc=MC),
              nine_maker(floor=floor, mc=MC)):
        for setting in (NOWHERE, EVERYWHERE):
            rows = run_orders(p, maker=m, setting=setting, orders=range(ORDERS), given={},
                              outer=bare)
            good = [r["units"] == ss.MIN_BLOCKS and r["calls"] == [KEEP] for r in rows]
            blocks = sorted({r["units"] for r in rows})
            print(f"  {m.name} {'veto' if setting.veto else 'none'}: {sum(good)} of {len(good)} "
                  f"stop at {ss.MIN_BLOCKS} with keep (blocks {blocks})")
            ok &= all(good)
    return ok


def collapse_pair() -> Plan:
    """The per-arm drop guard's design: C1m4 against p64 on its 12 tuning blocks (30848 collapses)
    beside C1m4 with that block's gain the base's (no collapse)."""
    p = plans()["C1m4/p64"]
    calm = np.where(p.gain[:, 0] < -ss.COLLAPSE, 0.0, p.gain[:, 0])
    return replace(p, name="C1m4+calm/p64", arms=(p.arms[0], Arm("calm", "C1m4, 30848 as p64")),
                   gain=np.column_stack([p.gain[:, 0], calm]),
                   seconds=np.column_stack([p.seconds[:, 0], p.seconds[:, 0]]))


def charged(p: Plan, order: np.ndarray, reach: Sequence[int]) -> float:
    """What each arm's prefix of `order` costs, summed (the base's runs exist: 0)."""
    return sum(float(p.seconds[order[:m], j].sum()) for j, m in enumerate(reach))


class Drops(NamedTuple):
    """A run's arms dropped on value (left out of a step while another goes on), whether each
    stayed where it was dropped (live at no later decision, its prefix then its prefix at every
    later decision and at the end), and whether every judged arm's verdict, re-derived from the
    record (`rule.met_reading(reading)`), is its take (met: left out of the step)."""
    drops: int
    kept: bool
    verdicts: bool


def value_drops(row: pd.Series | tuple) -> Drops:
    ds = decisions_of(row)
    drops, kept, verdicts = 0, True, True
    for t, d in enumerate(ds):
        for j, rule in d.rules.items():
            verdicts &= rule.met_reading(d.readings[j]) == (j not in d.arms)
        if not d.arms:
            continue
        for j in d.live - d.arms:
            drops += 1
            kept &= all(j not in e.live and e.arm_units[j] == d.at for e in ds[t + 1:])
            kept &= row.arm_units[j] == d.at
    return Drops(drops=drops, kept=kept, verdicts=verdicts)


def guard_per_arm(*, observing: bool) -> bool:
    """The per-arm drop, read from each run's record, for both targets. In a many-arm screen an
    arm the veto drops observes none of the later units and is charged for none (`collapse_pair`,
    every order, the veto everywhere); an arm that stops on value stops observing and paying while
    the others go on, its verdict in the record its take (4/base's first 10 orders run now and,
    without the fault, every many-arm run of `run`). Fault `observing`: the dropped arm kept
    observing (`KeepAll`, and no veto)."""
    ok = True
    outer, setting = (KeepAll, NOWHERE) if observing else (bare, EVERYWHERE)
    makers = (zero_one(price=OwnStake(), floor=ss.MIN_BLOCKS, mc=MC),
              stake_maker(floor=ss.MIN_BLOCKS, mc=MC))
    p = collapse_pair()
    for m in makers:
        good = []
        for row in run_orders(p, maker=m, setting=setting, orders=range(ORDERS), given={},
                              outer=outer):
            order = np.asarray(row["given"])
            at = int(np.flatnonzero(p.gain[order, 0] < -ss.COLLAPSE)[0]) + 1   # its collapse
            good.append(row["arm_units"] == [at, p.units] and row["vetoed"] == [0]
                        and row["calls"][0] == KEEP
                        and row["cost"] == charged(p, order, [at, p.units]))
        print(f"  veto, {m.name}: {sum(good)} of {len(good)} runs of {p.name}: the collapsing arm "
              f"observed none past its collapse and was charged for none, the other ran on")
        ok &= all(good)
    q = plans()["4/base"]
    sets = [(f"4/base, orders 0-9, now, {m.name}", pd.DataFrame(run_orders(
        q, maker=m, setting=NOWHERE, orders=range(10), given={}, outer=outer))) for m in makers]
    if not observing:
        many = [j for j in every_job()
                if j.arm is None and plans()[j.plan].many and isinstance(j.maker, EpiMaker)]
        sets += [(f"every many-arm run of `run`, {name}", pd.concat(
            [rows_of(j) for j in many if j.maker.name == name], ignore_index=True))
            for name in sorted({j.maker.name for j in many})]
    for name, rs in sets:
        n_drops = n_held = paid_less = n_verdicts = 0
        for r in rs.itertuples():
            d = value_drops(r)
            fair = r.cost == charged(plans()[r.plan], np.asarray(r.given), r.arm_units)
            n_drops += d.drops
            n_held += bool(d.drops) and d.kept and fair
            n_verdicts += d.verdicts
            paid_less += min(r.arm_units) < r.units
            ok &= d.kept and fair and d.verdicts
        print(f"  value: {name}: in {paid_less} of {len(rs)} runs an arm paid for fewer blocks "
              f"than the screen ran; {n_drops} arms dropped on value in {n_held} runs, each kept "
              f"where it stopped and charged so; verdicts the takes in {n_verdicts} of {len(rs)}")
        ok &= paid_less > 0 and n_held == paid_less
    return ok


def same_orders(rows: pd.DataFrame) -> bool:
    """Every policy compared on an order was handed that order (the order its run carries)."""
    ok = True
    for name, r in rows.groupby("plan", sort=False):
        canon = plans()[name].orders
        bad = [(g.policy, g.arms, g.veto, g.order) for g in r.itertuples()
               if not np.array_equal(np.asarray(g.given), canon[g.order])]
        if bad:
            print(f"  {name}: {len(bad)} runs on another order, e.g. {bad[0]}")
        ok &= not bad
    print(f"  orders: {len(rows)} runs checked")
    return ok


def one_truth(rows: pd.DataFrame) -> bool:
    """One truth for every policy and target. Per design (a many-arm design's variants each
    alone for seqscreen) and setting: the truth (`truth_of`) is every arm's whole-design call from
    the data, every policy's and target's own call on the whole design (`Truth.of` with its own
    policy) is it, and every run was scored against it (each row's truth, and its wrong calls and
    asks by AsksApart from its calls)."""
    ok, n_truths, n_own, n_rows = True, 0, 0, 0
    scoring = AsksApart(ask=ASK)
    for job in every_job():
        p, s = job_plan(job), job.setting
        the = truth_of(p, s)
        n_truths += 1
        data = [whole_call(p, j, veto=s.veto) for j in range(len(p.arms))]
        ok &= one_each(the.calls, p) == data
        with collapse_at(s.collapse):
            own = Truth.of(p.design(), job.maker.policy(p),
                           rng=np.random.default_rng(ss.ORDER_SEED), drop=s.drop)
        same = np.array_equal(own.calls, the.calls) and own.drop == the.drop
        n_own += same
        if not same:
            print(f"  {p.name} {job.maker.name} veto {s.veto}: its own truth "
                  f"{one_each(own.calls, p)}, not {one_each(the.calls, p)}")
        ok &= same
        mine = rows[(rows.plan == job.plan) & (rows.arms == ",".join(a.name for a in p.arms))
                    & (rows.policy == job.maker.name) & (rows.veto == s.veto)]
        for g in mine.itertuples():
            n_rows += 1
            w, a = scoring.score(np.asarray(g.calls)[:, None], the.calls)
            ok &= list(g.full) == data and (g.wrong, g.asks) == (w, a)
    print(f"  truth: {n_truths} jobs' truths the data's whole-design calls, {n_own} of "
          f"{n_truths} policies' own truths it, {n_rows} runs scored against it")
    return ok


def guard_orders(*, shuffled: bool) -> bool:
    """Same orders and one truth, on every run of `run`; fault `shuffled`: one policy (the
    realized loss over BRACKET, no veto, on both/p64) handed a reshuffled order 3."""
    rows = pd.concat([rows_of(j) for j in every_job()], ignore_index=True)
    if shuffled:
        p = plans()["both/p64"]
        m = stake_maker(floor=ss.MIN_BLOCKS, mc=MC)
        swap = run_orders(p, maker=m, setting=NOWHERE, orders=[3],
                          given={3: np.random.default_rng(1).permutation(p.orders[3])},
                          outer=bare)
        hit = ((rows.plan == p.name) & (rows.policy == m.name) & ~rows.veto & (rows.order == 3))
        assert hit.sum() == 1
        rows = pd.concat([rows[~hit], pd.DataFrame(swap)], ignore_index=True)
    return same_orders(rows) & one_truth(rows)


FAULTS = {"migration": ("unstaked",), "target": ("per-hour", "unnormalized"),
          "wrapper": ("floor",), "degenerate": ("ignored",), "per-arm drop": ("observing",),
          "same orders, one truth": ("shuffled",)}     # each guard's faults


def guards(*, only: str | None, fault: str | None) -> None:
    """Every guard (or `only`) without a fault, or the guard `fault` belongs to, with it."""
    if fault is not None and not any(fault in f for f in FAULTS.values()):
        raise SystemExit(f"unknown fault {fault!r}: one of {FAULTS}")
    if only is not None and only not in FAULTS:
        raise SystemExit(f"unknown guard {only!r}: one of {', '.join(FAULTS)}")
    checks: dict[str, Callable[[], bool]] = {
        "migration": lambda: guard_migration(
            price=Unstaked() if fault == "unstaked" else OwnStake()),
        "target": lambda: guard_target(fault=fault),
        "wrapper": lambda: guard_wrapper(floor=ss.MIN_BLOCKS - (fault == "floor")),
        "degenerate": lambda: guard_degenerate(floor=1 if fault == "ignored" else ss.MIN_BLOCKS),
        "per-arm drop": lambda: guard_per_arm(observing=fault == "observing"),
        "same orders, one truth": lambda: guard_orders(shuffled=fault == "shuffled")}
    results = {}
    for name, guard in checks.items():
        if (fault in FAULTS[name]) if fault is not None else only in (None, name):
            print(f"{name}:", flush=True)
            results[name] = guard()
    for k, v in results.items():
        print(f"{k}: {'pass' if v else 'FAIL'}")
    raise SystemExit(0 if all(results.values()) else 1)


# ---- the report -----------------------------------------------------------------------------

LOSS = [f"loss@{r:.1e}" for r in JUDGED]


def judged(rows: pd.DataFrame) -> pd.DataFrame:
    """Each run: AsksApart's wrong calls and asks; the block-arm runs; how many arms' call at the
    stop differs from the posterior mean's on what that arm saw (seqscreen's call); and at each
    judged rate each arm's call resolved (an ask answered from that arm's own means) against the
    whole design's, the regret and the realized loss."""
    out = []
    for g in rows.itertuples():
        p = plans()[g.plan]
        cols = [p.arm(a) for a in g.arms.split(",")]
        given = np.asarray(g.given)
        vetoed = ss.COLLAPSE if g.veto else math.inf
        row = dict(plan=g.plan, arms=g.arms, policy=g.policy, veto=g.veto, order=g.order,
                   units=g.units, runs=int(sum(g.arm_units)), cost=g.cost,
                   frac=g.cost / p.seconds[:, cols].sum(), wrong=g.wrong, asks=g.asks, apart=0)
        for r in JUDGED:
            row[f"wrong@{r:.1e}"], row[f"loss@{r:.1e}"] = 0, r / ss.FUTURE * g.cost
        for k, j in enumerate(cols):
            seen = given[:g.arm_units[k]]
            mu, extra = float(p.mu[j]), float(p.extra[j])
            ms = float(p.gain[seen, j].mean())
            es = float(np.mean(p.seconds[seen, j] - p.base_seconds[seen]))
            whole = KEEP if np.any(p.gain[:, j] < -vetoed) else CODE[ss.call(mu, extra, BRACKET)]
            if g.full[k] != whole:
                raise ValueError(f"{g.plan} {g.policy}: the truth {list(g.full)}, not {whole}")
            stop = KEEP if np.any(p.gain[seen, j] < -vetoed) else CODE[ss.call(ms, es, BRACKET)]
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


def screens(j: pd.DataFrame) -> pd.DataFrame:
    """One row a screen; a many-arm design's seqscreen screens, one a variant, summed per order
    into one (SUMMED), its cost fraction of every arm's blocks."""
    many = j.plan.isin([p.name for p in many_arm()]) & (j.policy == "seqscreen")
    agg = {c: (c, "sum") for c in ["runs", "cost", "wrong", "asks", "apart",
                                   *(f"{k}@{r:.1e}" for k in ("wrong", "loss") for r in JUDGED)]}
    summed = (j[many].groupby(["plan", "veto", "order"], as_index=False).agg(
        units=("units", "max"), **agg).assign(policy=SUMMED))
    summed["frac"] = summed.cost / summed.plan.map(lambda n: plans()[n].seconds.sum())
    return pd.concat([j[~many], summed], ignore_index=True)


def totals(sel: pd.DataFrame) -> pd.DataFrame:
    """Per setting and policy: the sum over designs of the median loss, the mean of the designs'
    median cost fractions, and wrong calls, asks and likeliest-vs-mean differences in all."""
    per = sel.groupby(["veto", "policy", "plan"]).agg(
        **{c: (c, "median") for c in [*LOSS, "frac"]},
        wrong=("wrong", "sum"), asks=("asks", "sum"), apart=("apart", "sum"))
    return per.groupby(["veto", "policy"]).agg(**{c: (c, "sum") for c in [*LOSS, "wrong", "asks",
                                                                         "apart"]},
                                              frac=("frac", "mean"))


def rule(sel: pd.DataFrame, *, veto: bool, candidate: str, against: str) -> str:
    """The rule's three criteria, each on its own, and the decision."""
    tot = totals(sel).loc[veto]
    c, a = tot.loc[candidate], tot.loc[against]
    below = [c[x] < a[x] for x in LOSS]
    one, two, three = all(below), c.wrong <= a.wrong, c.asks <= a.asks
    return (f"{'veto' if veto else 'no veto'}: {candidate} against {against}:\n"
            f"      (1) loss (1e-6) " + ", ".join(
                f"{x[5:]} {1e6 * c[x]:.1f} vs {1e6 * a[x]:.1f}{'' if b else ' (not below)'}"
                for x, b in zip(LOSS, below, strict=True)) + f": {'holds' if one else 'fails'}\n"
            f"      (2) wrong {c.wrong:.0f} vs {a.wrong:.0f}: {'holds' if two else 'fails'}\n"
            f"      (3) asks {c.asks:.0f} vs {a.asks:.0f}: {'holds' if three else 'fails'}\n"
            f"      -> {'ADOPT' if one and two and three else 'keep seqscreen'}")


def _cell(g: pd.DataFrame, col: str, *, peak: bool, width: int) -> str:
    b = f"{g[col].median():g}" + (f" ({g[col].max():g})" if peak else "")
    return f"{b:{width}s} {g.frac.median():.2f} {g.wrong.sum():3d} {g.asks.sum():4d}"


def report() -> None:
    rows = pd.concat([rows_of(j) for j in every_job()], ignore_index=True)
    rows.to_parquet(ROWS)
    t = screens(judged(rows.drop(columns=["decisions", "vetoed", "identity"])))
    t.to_parquet(SCREENS)
    own = zero_one(price=OwnStake(), floor=ss.MIN_BLOCKS, mc=MC).name
    both = zero_one(price=SharedStake(), floor=ss.MIN_BLOCKS, mc=MC).name
    stake_ = stake_maker(floor=ss.MIN_BLOCKS, mc=MC).name
    nine = nine_maker(floor=ss.MIN_BLOCKS, mc=MC).name
    pairs = t[t.plan.isin([p.name for p in pairwise()])]
    mt = t[t.plan.isin([p.name for p in many_arm()])]
    short = {"adopt": "A", "keep": "k", "ask": "?"}
    print("many-arm, per design (block-arm runs and cost medians over the 100 orders, the max in "
          "parentheses; wrong calls and asks in all; the median realized loss, 1e-6, at "
          + ", ".join(f"{r:.2g}" for r in JUDGED) + "):")
    for p in many_arm():
        calls = "".join(short[ss.call(float(m), float(e), BRACKET)]
                        for m, e in zip(p.mu, p.extra, strict=True))
        for pol in (SUMMED, own, both, stake_):
            g = mt[(mt.plan == p.name) & (mt.policy == pol)]
            print(f"    {p.name:8s} {calls:5s} {pol:36s} {_cell(g, 'runs', peak=True, width=10)}"
                  "  " + " ".join(f"{1e6 * g[c].median():6.1f}" for c in LOSS))
    print("\npairwise, per design (blocks, cost, wrong, asks; the median realized loss, 1e-6):")
    for p in pairwise():
        for veto in (False, True):
            g = pairs[(pairs.plan == p.name) & (pairs.veto == veto)]
            print(f"    {p.name:14s} {'veto' if veto else 'none'} "
                  + "  ".join(_cell(g[g.policy == q], "units", peak=True, width=9)
                              for q in ("seqscreen", own, stake_, nine)))
            print(f"    {'':19s} " + "  ".join(
                " ".join(f"{1e6 * g[g.policy == q][c].median():5.1f}" for c in LOSS)
                for q in ("seqscreen", own, stake_, nine)))
    print("\nthe rule (realized loss, 1e-6, the sum over designs of each one's median over orders;"
          " wrong calls and asks in all; cost the mean of the designs' median fractions):")
    for label, sel, veto, pols in (
            ("no veto", pairs, False, ("seqscreen", own, stake_, nine)),
            ("veto", pairs, True, ("seqscreen", own, stake_, nine)),
            ("many-arm", mt, False, (SUMMED, own, both, stake_))):
        tot = totals(sel).loc[veto]
        for k, q in enumerate(pols):
            c = tot.loc[q]
            print(f"    {label if k == 0 else '':8s} {q:36s} "
                  + " ".join(f"{1e6 * c[x]:7.1f}" for x in LOSS)
                  + f"   {c.wrong:4.0f} {c.asks:4.0f}  {c.frac:.3f}  apart {c.apart:.0f}")
    print()
    for sel, veto, against in ((pairs, False, "seqscreen"), (pairs, True, "seqscreen"),
                               (mt, False, SUMMED)):
        print("  " + rule(sel, veto=veto, candidate=stake_, against=against))
    print("  beside it, the sensitivity:")
    for veto in (False, True):
        print("  " + rule(pairs, veto=veto, candidate=nine, against="seqscreen"))
    print("  beside it, the 0-1 rows:")
    for sel, veto, cand, against in ((pairs, False, own, "seqscreen"),
                                     (pairs, True, own, "seqscreen"), (mt, False, own, SUMMED),
                                     (mt, False, both, SUMMED)):
        print("  " + rule(sel, veto=veto, candidate=cand, against=against))
    print("\nthe sensitivity, run by run: nine rates against five (orders whose calls, blocks "
          "differ):")
    raw = rows.set_index(["plan", "veto", "order", "policy"])
    moved_calls = moved_blocks = 0
    for p in pairwise():
        for veto in (False, True):
            a = raw.loc[(p.name, veto)].xs(stake_, level="policy").sort_index()
            b = raw.loc[(p.name, veto)].xs(nine, level="policy").sort_index()
            c = sum(list(x) != list(y) for x, y in zip(a.calls, b.calls, strict=True))
            u = int((a.units != b.units).sum())
            moved_calls, moved_blocks = moved_calls + c, moved_blocks + u
            if c or u:
                print(f"    {p.name:14s} {'veto' if veto else 'none'}: calls {c}, blocks {u}")
    print(f"    in all: calls in {moved_calls} runs, blocks in {moved_blocks} of "
          f"{2 * ORDERS * len(pairwise())}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["check"]:
        sys.exit(0 if check() else 1)
    elif args == ["report"]:
        report()
    elif len(args) == 3 and args[0] == "run":
        names = args[1].split(",")
        if not set(names) <= set(GROUPS):
            raise SystemExit(f"unknown group in {args[1]!r}: groups {', '.join(GROUPS)}")
        run(names, int(args[2]))
    elif len(args) == 2 and args[0] == "guards":
        guards(only=None, fault=None if args[1] == "none" else args[1])
    elif len(args) == 2 and args[0] == "guard":
        guards(only=args[1], fault=None)
    else:
        raise SystemExit(__doc__)
