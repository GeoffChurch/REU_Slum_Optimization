"""Pruning by epistemetry's decision layer: a replay against seqscreen.py (BACKLOG, "Pruning by
epistemetry's decision layer"; NOTES, "Pruning by epistemetry, again: step-wise re-decision and
per-arm screens" and its "A price per arm").

The first replay (NOTES, "Pruning by epistemetry: a replay against seqscreen") kept seqscreen and
named what would have to change: re-deciding after every block, and a per-arm drop in many-arm
screens. GeoffChurch/epistemetry built both (bbe2163, ValueOfMoreUnits' `step`; 64aae7f, per-arm
screens in its replay harness) and the second replay (22394aa2) kept seqscreen again, tracing the
many-arm screens' remaining gap partly to one price shared by the live arms. epistemetry 3a3b9a3
asks each arm's stop rule of a `StopOf`, so each arm can be priced by its own stake; this replays
that under the same rule, through an ephemeral overlay (reblock does not depend on epistemetry):

    EPI='epistemetry @ git+https://github.com/GeoffChurch/epistemetry@3a3b9a3'
    uv run --with "$EPI" python research/roadless/epireplay.py check
    uv run --with "$EPI" python research/roadless/epireplay.py run
    uv run --with "$EPI" python research/roadless/epireplay.py guards <fault>
    uv run --with "$EPI" python research/roadless/epireplay.py report

with OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS=1 (WORKERS processes); `guards none` runs every guard,
`guards <fault>` (other, floor, ignored, observing or shuffled) the one it breaks.

THE DESIGNS are the first replay's, frozen as data in DESIGNS (committed), the only design data
read here: its 13 pairwise designs (the collapse, C1m4/p64, among them) and 5 many-arm ones,
extracted once by the first replay's own code (`git archive 0c2e588e^ research/roadless`, its
`data` and `orders`, seqscreen handed simp_compare's loaders as when it ran) from raster-era rows
that nothing here reads. Per design: the names, the block ids, each arm's Lens A gain over the
base and its GPU-seconds, the base's GPU-seconds, and the 100 orders (seqscreen.main's 20, then 80
more from its generator). `check` holds them to the first replay's stored runs: every whole-design
call and seqscreen's replay of every order.

THE POLICIES run on the same orders through epistemetry's `replay`, one call a design, each order
carrying its own stream (`Maker.stream`), scored against one truth computed once (each arm's call
on the whole design over the rate bracket), in two veto settings (`Setting`): everywhere (the
harness's VetoBelow at seqscreen's COLLAPSE, strictly below, for every policy and the truth alike)
or nowhere (NoVeto, and seqscreen's own COLLAPSE patched off inside the process; no file edited):
- `SeqScreen`: seqscreen unchanged, as a Policy of one arm: its stop (stop_at on the blocks seen,
  replayed from the first block with the order's generator, as seqscreen.main replays it), floor
  and veto, and its call over the bracket.
- ValueOfMoreUnits on FinitePopulation with AdoptKeepAsk over seqscreen's bracket, matched to
  seqscreen as in the first replay: an arm's outcome its gain and its extra GPU-seconds (two
  columns of one ball); floor MIN_BLOCKS; every horizon to the last block; seqscreen's cost model
  (`RatioToBase`); at `step` 1 (re-decide after every block, as seqscreen does: a card runs one
  block at a time here, the local card one GPU process and the cluster one card a block-task, so a
  card's batch is one block); and ValueOfInformation at no Monte Carlo margin, each arm priced by
  its own stake (`OwnStake`): seqscreen's price (the bracket's mean rate over FUTURE) over the Lens
  A a wrong call on that arm costs at its means so far, as seqscreen prices each variant on its
  own. On one arm it is the second replay's price exactly.
- many-arm: one such policy over every variant of a design, each arm stopping on its own value at
  its own price (the per-arm screen), against seqscreen once per variant on the same orders,
  summed; beside it the second replay's price, the live arms' stakes averaged (`SharedStake`).
Each run records its decisions (`Run.decisions`), which the guards read.

THE RULE, fixed (the first replay's, amended): adopt epistemetry for the consolidated pass's
pruning only if, against seqscreen in the same veto setting, (1) its median realized loss is below
seqscreen's at every judged rate (per rate, the sum over designs of each one's median over
orders), (2) it makes no more wrong calls in all and (3) no more asks in all: pairwise in both
settings; many-arm against the pairwise screens it replaces. The realized loss at rate r: for each
arm whose call resolved at r (an ask answered at r from the arm's own means) differs from the
whole design's call at r, |mu - r x extra|; plus r / FUTURE x the GPU-seconds spent. Wrong calls
and asks by AsksApart (an ask is never wrong; a definite call differing from the whole design's,
an ask included, is).
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import re
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cache
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple, Protocol
from unittest import mock

import numpy as np
import pandas as pd
from epistemetry.design.decide import (ADOPT, ASK, KEEP, AdoptKeepAsk, StandardErrors, StopRule,
                                      ValueOfInformation)
from epistemetry.design.replay import (STOP, AsksApart, Bootstrap, Design, DropRule, NoVeto, Order,
                                       Policy, Prefix, Seen, Step, ValueOfMoreUnits, VetoBelow,
                                       replay, truth)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import seqscreen as ss  # noqa: E402

DESIGNS = HERE / "epireplay_designs.json"      # committed: the designs, frozen
JOBS = HERE / "epireplay3_jobs"                # git-ignored, a parquet a job: a rerun resumes
ROWS = HERE / "epireplay3.parquet"             # git-ignored: `run`'s jobs gathered
SCREENS = HERE / "epireplay3_screens.parquet"  # git-ignored: `report`'s judged screens
FIRST = HERE / "epireplay_bracket.parquet"     # the first replay's runs (records)
FIRST_JOBS = HERE / "epireplay_bracket_jobs"   # ... as its jobs wrote them
SECOND = HERE / "epireplay2.parquet"           # the second replay's runs (22394aa2, records)
SECOND_JOBS = HERE / "epireplay2_jobs"         # ... as its jobs wrote them
BRACKET = np.geomspace(*ss.RATES, ss.RATE_POINTS)   # seqscreen's stop and call
JUDGED = np.geomspace(*ss.RATES, 3)            # the bracket's ends and its log-midpoint
PRICE_RATE = float(BRACKET.mean())             # seqscreen prices its stop's GPU-seconds at it
CODE = {"keep": KEEP, "ask": ASK, "adopt": ADOPT}   # seqscreen's calls as AdoptKeepAsk's
ORDERS = 100        # orders a design, in DESIGNS
WORKERS = 4         # processes; this machine is shared
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
        """epistemetry's: each arm's gain and extra GPU-seconds (AdoptKeepAsk's pair), what each
        arm cost on each block, and the base's runs, which exist: 0."""
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


# ---- the policies ---------------------------------------------------------------------------

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
    """seqscreen.py, unchanged, as an epistemetry Policy of one arm: `next` is stop_at on the
    blocks seen (the base's times on every block of the order), replayed from the first block with
    the order's generator (a copy of the run's stream, which it never draws from itself), as
    seqscreen.main replays it, and a block more while it runs on; `calls` is stop_at's call on the
    blocks seen."""
    base_seconds: np.ndarray    # (u,)
    floor: int

    def next(self, seen: Seen, *, rng: np.random.Generator) -> Step:
        (arm,) = seen.arms
        order = np.concatenate([seen.base, seen.upcoming])
        stop = ss.stop_at(ss.Screen(arm.outcomes[:, 0], arm.costs, self.base_seconds[order]),
                          rates=BRACKET, future=ss.FUTURE, floor=self.floor,
                          rng=copy.deepcopy(rng))
        return STOP if stop is not None else Step(units=1, arms=seen.live)

    def calls(self, seen: Seen, *, rng: np.random.Generator) -> np.ndarray:
        (arm,) = seen.arms
        return np.array([[CODE[seq_call(ss.Screen(arm.outcomes[:, 0], arm.costs,
                                                  self.base_seconds[arm.units]))]]])


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


class Price(Protocol):
    """epistemetry's StopOf, named: the rule each arm is judged by at a decision."""
    @property
    def name(self) -> str: ...

    def of(self, seen: Seen, *, arm: int) -> StopRule: ...


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
    own price. In the many-arm screens it makes one wrong call and two asks fewer than the own
    price in 500 runs, and more loss at every rate (NOTES), so it stays beside it."""

    @property
    def name(self) -> str:
        return "shared price"

    def of(self, seen: Seen, *, arm: int) -> StopRule:
        return priced(float(np.mean([stake(seen.arms[j].outcomes) for j in sorted(seen.live)])))


@dataclass(frozen=True)
class OtherStake:
    """The migration guard's fault: each arm priced by the next arm's stake (on one arm, its
    own)."""

    @property
    def name(self) -> str:
        return "another arm's price"

    def of(self, seen: Seen, *, arm: int) -> StopRule:
        return priced(stake(seen.arms[(arm + 1) % len(seen.arms)].outcomes))


@dataclass(frozen=True, eq=False)
class KeepAll:
    """The per-arm drop guard's fault: every live arm observes while any goes on, so an arm the
    inner policy drops keeps observing."""
    inner: Policy

    def next(self, seen: Seen, *, rng: np.random.Generator) -> Step:
        step = self.inner.next(seen, rng=rng)
        return step if not step.arms else Step(units=step.units, arms=seen.live)

    def calls(self, seen: Seen, *, rng: np.random.Generator) -> np.ndarray:
        return self.inner.calls(seen, rng=rng)


def bare(policy: Policy) -> Policy:
    """No fault: the policy as made."""
    return policy


class Maker(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def floor(self) -> int: ...

    def policy(self, plan: Plan) -> Policy: ...

    def stream(self, o: int) -> np.random.Generator: ...


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


@dataclass(frozen=True, kw_only=True)
class EpiMaker:
    step: int
    floor: int
    price: OwnStake | SharedStake | OtherStake
    mc: MonteCarlo

    @property
    def name(self) -> str:
        return f"adopt/keep/ask step {self.step}, {self.price.name}"

    def policy(self, plan: Plan) -> Policy:
        return ValueOfMoreUnits(
            belief=Bootstrap(), functionals=np.eye(2),
            target=AdoptKeepAsk(low=ss.RATES[0], high=ss.RATES[1]),
            horizons=tuple(range(1, plan.units + 1)), step=self.step,
            cost=RatioToBase(base_seconds=plan.base_seconds), stop=self.price, floor=self.floor,
            outer=self.mc.outer, inner=self.mc.inner, blocks=self.mc.blocks, draws=self.mc.draws)

    def stream(self, o: int) -> np.random.Generator:
        """Order o's stream as every replay of these designs has drawn epistemetry's runs from it:
        the second spawned from [ORDER_SEED, o] (the first was its truth's, when each order had
        its own)."""
        return np.random.default_rng([ss.ORDER_SEED, o]).spawn(2)[1]


# ---- runs -----------------------------------------------------------------------------------

def one_each(calls: np.ndarray, plan: Plan) -> list[int]:
    """A run's calls, (arms, 1), as one call per arm."""
    return [int(c) for c in np.asarray(calls).reshape(len(plan.arms))]


def run_orders(plan: Plan, *, maker: SeqMaker | EpiMaker, setting: Setting,
               orders: Sequence[int], given: Mapping[int, np.ndarray],
               outer: Callable[[Policy], Policy]) -> list[dict]:
    """`plan`'s `orders` through one epistemetry replay in `setting`, each order with its maker's
    stream, scored with AsksApart against the truth computed once; the policy inside `outer` (a
    guard's fault, or `bare`). `given` hands an order another permutation (the same-orders guard's
    fault)."""
    design = plan.design()
    policy = outer(maker.policy(plan))
    handed = [Order(units=given[o] if o in given else plan.orders[o], rng=maker.stream(o))
              for o in orders]
    with collapse_at(setting.collapse):
        calls = truth(design, policy, rng=np.random.default_rng(ss.ORDER_SEED), drop=setting.drop)
        rep = replay(design, policy, orders=handed, truth=calls, scoring=AsksApart(ask=ASK),
                     drop=setting.drop)
    return [dict(plan=plan.name, arms=",".join(a.name for a in plan.arms), policy=maker.name,
                 veto=setting.veto, order=o, units=run.units, arm_units=list(run.arm_units),
                 cost=run.cost, calls=one_each(run.calls, plan), full=one_each(rep.truth, plan),
                 wrong=run.wrong, asks=run.asks, vetoed=sorted(run.vetoed),
                 given=h.units.tolist(),
                 decisions=json.dumps([dict(at=d.at, live=sorted(d.live), units=d.step.units,
                                            arms=sorted(d.step.arms)) for d in run.decisions]))
            for o, h, run in zip(orders, handed, rep.runs, strict=True)]


class Decided(NamedTuple):
    """A row's decision, from `Run.decisions` (JSON in the rows)."""
    at: int
    live: frozenset[int]
    units: int
    arms: frozenset[int]


def decisions_of(row: pd.Series | tuple) -> list[Decided]:
    return [Decided(at=d["at"], live=frozenset(d["live"]), units=d["units"],
                    arms=frozenset(d["arms"])) for d in json.loads(row.decisions)]


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


def candidate(*, floor: int, mc: MonteCarlo) -> EpiMaker:
    """epistemetry's policy here: step 1, each arm at its own price."""
    return EpiMaker(step=1, floor=floor, price=OwnStake(), mc=mc)


def shared(*, floor: int, mc: MonteCarlo) -> EpiMaker:
    """The candidate with the second replay's price."""
    return replace(candidate(floor=floor, mc=mc), price=SharedStake())


def jobs(*, floor: int, mc: MonteCarlo) -> list[Job]:
    """The replay's: on every pairwise design in both settings, seqscreen and the candidate; on
    every many-arm design (no unit of any is below -COLLAPSE, so the settings agree: NOWHERE, as
    the earlier replays ran them), seqscreen once per variant and the per-arm screen at its own
    price and at the shared one."""
    epi, out = candidate(floor=floor, mc=mc), []
    for setting in (NOWHERE, EVERYWHERE):
        for p in pairwise():
            out += [Job(p.name, None, SeqMaker(floor=floor), setting),
                    Job(p.name, None, epi, setting)]
    for p in many_arm():
        out += [Job(p.name, j, SeqMaker(floor=floor), NOWHERE) for j in range(len(p.arms))]
        out += [Job(p.name, None, epi, NOWHERE),
                Job(p.name, None, shared(floor=floor, mc=mc), NOWHERE)]
    return out


def _priority(job: Job) -> tuple[int, int]:
    """The longest first: the per-arm screens, then epistemetry's pairwise, then seqscreen;
    within, the most block-arms."""
    p = job_plan(job)
    kind = (0 if isinstance(job.maker, EpiMaker) and p.many and job.arm is None else
            1 if isinstance(job.maker, EpiMaker) else 2)
    return kind, -p.units * len(p.arms)


def _one(job: Job) -> str:
    rows_of(job)
    arm = "all" if job.arm is None else job.arm
    return f"{job.plan} {job.maker.name} {arm} veto {job.setting.veto}"


def run() -> None:
    every = jobs(floor=ss.MIN_BLOCKS, mc=MC)
    todo = sorted((j for j in every if not job_file(j).exists()), key=_priority)
    print(f"{len(every) - len(todo)} of {len(every)} jobs done before", flush=True)
    with Pool(WORKERS) as pool:
        for k, done in enumerate(pool.imap_unordered(_one, todo)):
            print(f"{k + 1}/{len(todo)} {done}", flush=True)
    rows = pd.concat([rows_of(j) for j in every], ignore_index=True)
    rows.to_parquet(ROWS)
    print(f"{len(rows)} rows -> {ROWS}")


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

def same_runs(a: pd.DataFrame, b: pd.DataFrame) -> int:
    """Orders whose runs agree exactly: blocks, calls and GPU-seconds."""
    a, b = a.sort_values("order"), b.sort_values("order")
    if a.order.tolist() != b.order.tolist():
        raise ValueError("the runs are not on the same orders")
    return sum((x.units, list(x.calls), x.cost) == (y.units, list(y.calls), y.cost)
               for x, y in zip(a.itertuples(), b.itertuples(), strict=True))


def seen_at(p: Plan, order: np.ndarray, reach: Sequence[int], at: int,
            live: frozenset[int]) -> Seen:
    """What a policy was shown at a decision: arm j's first reach[j] units of `order`, the base's
    first `at`."""
    design = p.design()
    return Seen(arms=tuple(Prefix(units=order[:m], outcomes=design.outcomes[order[:m], j],
                                  costs=design.costs[order[:m], j]) for j, m in enumerate(reach)),
                base=order[:at], base_costs=design.base_costs[order[:at]], live=live,
                upcoming=order[at:])


def own_prices(rows: pd.DataFrame, price: Price) -> tuple[int, int]:
    """At every decision past the floor of every run, each live arm's price under `price` against
    its own stake, computed apart from the policy from the design: (prices that are the arm's
    own, prices where the arm's own and the next arm's differ, so a wrong one shows)."""
    own = differ = 0
    for r in rows.itertuples():
        p, order = plans()[r.plan], np.asarray(r.given)
        extra = p.seconds - p.base_seconds[:, None]
        reach = [0] * len(p.arms)
        for d in decisions_of(r):
            for j in d.live:
                reach[j] = d.at
            if d.at < ss.MIN_BLOCKS:
                continue
            seen = seen_at(p, order, reach, d.at, d.live)
            for j in d.live:
                want = [PRICE_RATE / ss.FUTURE
                        / abs(p.gain[order[:reach[i]], i].mean()
                              - PRICE_RATE * extra[order[:reach[i]], i].mean())
                        for i in (j, (j + 1) % len(p.arms))]
                got = price.of(seen, arm=j)
                assert isinstance(got, ValueOfInformation)
                if not math.isclose(want[0], want[1], rel_tol=1e-9):
                    differ += 1
                    own += math.isclose(got.price, want[0], rel_tol=1e-9)
    return own, differ


def guard_migration(*, price: OwnStake | OtherStake) -> bool:
    """The migration to 3a3b9a3. The candidate (each arm at its own price) reproduces the second
    replay's pairwise step-1 runs exactly (blocks, calls, GPU-seconds, both settings): on one arm
    its own price is the second replay's shared one; and the shared price reproduces its many-arm
    runs. On many-arm designs, where the arms' stakes differ, the price `price` gives each arm at
    every decision is its own stake's, read from each run's decisions (2/p64's first 10 orders run
    now and, without the fault, every many-arm run of the candidate). Fault `other`: each arm
    priced by another arm's stake."""
    second = stored(SECOND, SECOND_JOBS)
    epi, both = candidate(floor=ss.MIN_BLOCKS, mc=MC), shared(floor=ss.MIN_BLOCKS, mc=MC)
    ok = True
    for name, maker, plans_, settings in (
            ("pairwise, step 1 at its own price", epi, pairwise(), (NOWHERE, EVERYWHERE)),
            ("many-arm, step 1 at the shared price", both, many_arm(), (NOWHERE,))):
        match = total = 0
        for setting in settings:
            for p in plans_:
                got = rows_of(Job(p.name, None, maker, setting))
                ref = second[(second.plan == p.name) & (second.veto == setting.veto)
                             & (second.policy == "adopt/keep/ask step 1")]
                k = same_runs(got, ref)
                match, total = match + k, total + len(ref)
                ok &= k == len(ref) == ORDERS
        print(f"  {name}: {match} of {total} runs the second replay's")
    m = replace(epi, price=price)
    q = plans()["2/p64"]
    sets = [("2/p64, orders 0-9, now", pd.DataFrame(run_orders(
        q, maker=m, setting=NOWHERE, orders=range(10), given={}, outer=bare)))]
    if m == epi:
        sets.append(("every many-arm run of the candidate", pd.concat(
            [rows_of(Job(p.name, None, epi, NOWHERE)) for p in many_arm()], ignore_index=True)))
    for name, rows in sets:
        own, differ = own_prices(rows, price)
        print(f"  many-arm, {m.price.name}: {name}: {own} of {differ} prices (where the arm's own "
              "and the next arm's differ) the arm's own")
        ok &= own == differ > 0
    return ok


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


def degenerate() -> Plan:
    """p64 against base with p64's gains replaced by the base's: a variant that changes nothing,
    at p64's GPU time."""
    p = plans()["p64/base"]
    return replace(p, gain=np.zeros_like(p.gain))


def guard_degenerate(*, floor: int) -> bool:
    """The candidate, in both settings, on a variant equal to its base stops at the floor with
    "keep", every order. Fault `ignored`: the floor set to 1."""
    p = degenerate()
    m = EpiMaker(step=1, floor=floor, price=OwnStake(), mc=MC)
    ok = True
    for setting in (NOWHERE, EVERYWHERE):
        rows = run_orders(p, maker=m, setting=setting, orders=range(ORDERS), given={},
                          outer=bare)
        good = [r["units"] == ss.MIN_BLOCKS and r["calls"] == [KEEP] for r in rows]
        print(f"  {m.name} {'veto' if setting.veto else 'none'}: {sum(good)} of {len(good)} stop"
              f" at {ss.MIN_BLOCKS} with keep (blocks {sorted({r['units'] for r in rows})})")
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


def value_drops(row: pd.Series | tuple) -> tuple[int, bool]:
    """A run's arms dropped on value (left out of a step while another goes on, in its
    `Run.decisions`), and whether each stayed where it was dropped: live at no later decision, its
    blocks the ones seen when it was dropped."""
    ds = decisions_of(row)
    drops, kept = 0, True
    for t, d in enumerate(ds):
        if not d.arms:
            continue
        for j in d.live - d.arms:
            drops += 1
            kept &= all(j not in e.live for e in ds[t + 1:])
            kept &= row.arm_units[j] == d.at
    return drops, kept


def guard_per_arm(*, observing: bool) -> bool:
    """The per-arm drop, read from each run's record. In a many-arm screen an arm the veto drops
    observes none of the later units and is charged for none (`collapse_pair`, every order, the
    veto everywhere); an arm that stops on value stops observing and paying while the others go
    on (4/base's first 10 orders run now and, without the fault, every many-arm run of `run`).
    Fault `observing`: the dropped arm kept observing (`KeepAll`, and no veto)."""
    ok = True
    outer, setting = (KeepAll, NOWHERE) if observing else (bare, EVERYWHERE)
    m = candidate(floor=ss.MIN_BLOCKS, mc=MC)
    p = collapse_pair()
    good = []
    for r in run_orders(p, maker=m, setting=setting, orders=range(ORDERS), given={},
                        outer=outer):
        order = np.asarray(r["given"])
        at = int(np.flatnonzero(p.gain[order, 0] < -ss.COLLAPSE)[0]) + 1   # its collapse
        good.append(r["arm_units"] == [at, p.units] and r["vetoed"] == [0]
                    and r["calls"][0] == KEEP and r["cost"] == charged(p, order, [at, p.units]))
    print(f"  veto: {sum(good)} of {len(good)} runs of {p.name}: the collapsing arm observed "
          f"none past its collapse and was charged for none, the other ran on")
    ok &= all(good)
    q = plans()["4/base"]
    sets = [("4/base, orders 0-9, now", pd.DataFrame(run_orders(
        q, maker=m, setting=NOWHERE, orders=range(10), given={}, outer=outer)))]
    if not observing and all(job_file(j).exists() for j in jobs(floor=ss.MIN_BLOCKS, mc=MC)):
        sets.append(("every many-arm run of `run`", pd.concat(
            [rows_of(j) for j in jobs(floor=ss.MIN_BLOCKS, mc=MC)
             if j.arm is None and plans()[j.plan].many], ignore_index=True)))
    for name, rs in sets:
        n_drops = n_held = paid_less = 0
        for r in rs.itertuples():
            drops, kept = value_drops(r)
            fair = r.cost == charged(plans()[r.plan], np.asarray(r.given), r.arm_units)
            n_drops += drops
            n_held += bool(drops) and kept and fair
            paid_less += min(r.arm_units) < r.units
            ok &= kept and fair
        print(f"  value: {name}: in {paid_less} of {len(rs)} runs an arm paid for fewer blocks "
              f"than the screen ran; {n_drops} arms dropped on value in {n_held} runs, each kept "
              "where it stopped and charged so")
        ok &= paid_less > 0 and n_held == paid_less
    return ok


def same_orders(rows: pd.DataFrame) -> bool:
    """Every policy compared on an order was handed that order."""
    ok = True
    for name, r in rows.groupby("plan", sort=False):
        canon = plans()[name].orders
        bad = [(g.policy, g.arms, g.veto, g.order) for g in r.itertuples()
               if not np.array_equal(np.asarray(g.given), canon[g.order])]
        if bad:
            print(f"  {name}: {len(bad)} runs on another order, e.g. {bad[0]}")
        ok &= not bad
    print(f"  {len(rows)} runs checked")
    return ok


def guard_orders(*, shuffled: bool) -> bool:
    """The same-orders check on every run of `run`; fault `shuffled`: one policy (the candidate, no
    veto, on both/p64) handed a reshuffled order 3."""
    every = jobs(floor=ss.MIN_BLOCKS, mc=MC)
    rows = pd.concat([rows_of(j) for j in every], ignore_index=True)
    if shuffled:
        p = plans()["both/p64"]
        m = candidate(floor=ss.MIN_BLOCKS, mc=MC)
        swap = run_orders(p, maker=m, setting=NOWHERE, orders=[3],
                          given={3: np.random.default_rng(1).permutation(p.orders[3])},
                          outer=bare)
        hit = ((rows.plan == p.name) & (rows.policy == m.name) & ~rows.veto & (rows.order == 3))
        assert hit.sum() == 1
        rows = pd.concat([rows[~hit], pd.DataFrame(swap)], ignore_index=True)
    return same_orders(rows)


FAULTS = ("other", "floor", "ignored", "observing", "shuffled")


GUARDS = ("migration", "wrapper", "degenerate", "per-arm drop", "same orders")


def guards(fault: str | None) -> None:
    """Every guard without a fault, or the guard a fault belongs to, with it."""
    if fault is not None and fault not in FAULTS:
        raise SystemExit(f"unknown fault {fault!r}: one of {', '.join(FAULTS)}")
    checks = {
        "migration": lambda: guard_migration(price=OtherStake() if fault == "other"
                                             else OwnStake()),
        "wrapper": lambda: guard_wrapper(floor=ss.MIN_BLOCKS - (fault == "floor")),
        "degenerate": lambda: guard_degenerate(floor=1 if fault == "ignored"
                                               else ss.MIN_BLOCKS),
        "per-arm drop": lambda: guard_per_arm(observing=fault == "observing"),
        "same orders": lambda: guard_orders(shuffled=fault == "shuffled")}
    run_ = GUARDS if fault is None else (GUARDS[FAULTS.index(fault)],)
    results = {}
    for name in run_:
        print(f"{name}:", flush=True)
        results[name] = checks[name]()
    for k, v in results.items():
        print(f"{k}: {'pass' if v else 'FAIL'}")
    raise SystemExit(0 if all(results.values()) else 1)


# ---- the report -----------------------------------------------------------------------------

LOSS = [f"loss@{r:.1e}" for r in JUDGED]
# the first replay's policy beside this one's (the veto is a column)
EARLIER = {(False, "adopt/keep/ask"): "first: adopt/keep/ask",
           (True, "adopt/keep/ask+veto"): "first: adopt/keep/ask"}


def earlier() -> pd.DataFrame:
    """The first replay's whole-horizon adopt/keep/ask runs, named as here, each arm's blocks the
    screen's (one prefix then)."""
    f = stored(FIRST, FIRST_JOBS)
    f = f[[(v, q) in EARLIER for v, q in zip(f.veto, f.policy, strict=True)]]
    return f.assign(policy=[EARLIER[(v, q)] for v, q in zip(f.veto, f.policy, strict=True)],
                    arm_units=[[u] * len(a.split(",")) for u, a in zip(f.units, f.arms,
                                                                       strict=True)])


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
    tot = totals(sel).loc[veto]
    c, a = tot.loc[candidate], tot.loc[against]
    below = [c[x] < a[x] for x in LOSS]
    ok = all(below) and c.wrong <= a.wrong and c.asks <= a.asks
    return (f"{'veto' if veto else 'no veto'}: {candidate} against {against}: loss (1e-6) "
            + ", ".join(f"{x[5:]} {1e6 * c[x]:.1f} vs {1e6 * a[x]:.1f}" for x in LOSS)
            + f"; wrong {c.wrong:.0f} vs {a.wrong:.0f}; asks {c.asks:.0f} vs {a.asks:.0f}: "
            + ("ADOPT" if ok else "keep seqscreen"))


def _cell(g: pd.DataFrame, col: str, *, peak: bool, width: int) -> str:
    b = f"{g[col].median():g}" + (f" ({g[col].max():g})" if peak else "")
    return f"{b:{width}s} {g.frac.median():.2f} {g.wrong.sum():3d} {g.asks.sum():4d}"


def report() -> None:
    mine = pd.concat([rows_of(j) for j in jobs(floor=ss.MIN_BLOCKS, mc=MC)], ignore_index=True)
    t = screens(judged(pd.concat([mine.drop(columns=["decisions", "vetoed"]), earlier()],
                                 ignore_index=True)))
    t.to_parquet(SCREENS)
    k1 = candidate(floor=ss.MIN_BLOCKS, mc=MC).name
    k2 = shared(floor=ss.MIN_BLOCKS, mc=MC).name
    pairs = t[t.plan.isin([p.name for p in pairwise()])]
    mt = t[t.plan.isin([p.name for p in many_arm()])]
    short = {"adopt": "A", "keep": "k", "ask": "?"}
    print("many-arm, per design (block-arm runs and cost medians over the 100 orders, the max in "
          "parentheses; wrong calls and asks in all; the median realized loss, 1e-6, at "
          + ", ".join(f"{r:.2g}" for r in JUDGED) + "):")
    for p in many_arm():
        calls = "".join(short[ss.call(float(m), float(e), BRACKET)]
                        for m, e in zip(p.mu, p.extra, strict=True))
        for pol in (SUMMED, k1, k2, "first: adopt/keep/ask"):
            g = mt[(mt.plan == p.name) & (mt.policy == pol)]
            print(f"    {p.name:8s} {calls:5s} {pol:36s} {_cell(g, 'runs', peak=True, width=10)}"
                  "  " + " ".join(f"{1e6 * g[c].median():6.1f}" for c in LOSS))
    print("\npairwise, per design: the candidate against seqscreen (blocks, cost, wrong, asks)")
    for p in pairwise():
        for veto in (False, True):
            g = pairs[(pairs.plan == p.name) & (pairs.veto == veto)]
            print(f"    {p.name:14s} {'veto' if veto else 'none'} "
                  + "  ".join(_cell(g[g.policy == q], "units", peak=True, width=9)
                              for q in ("seqscreen", k1)))
    print("\nthe rule (realized loss, 1e-6, the sum over designs of each one's median over orders;"
          " wrong calls and asks in all; cost the mean of the designs' median fractions):")
    for label, sel, veto, pols in (
            ("no veto", pairs, False, ("seqscreen", k1)),
            ("veto", pairs, True, ("seqscreen", k1)),
            ("many-arm", mt, False, (SUMMED, k1, k2, "first: adopt/keep/ask"))):
        tot = totals(sel).loc[veto]
        for k, q in enumerate(pols):
            c = tot.loc[q]
            print(f"    {label if k == 0 else '':8s} {q:36s} "
                  + " ".join(f"{1e6 * c[x]:7.1f}" for x in LOSS)
                  + f"   {c.wrong:4.0f} {c.asks:4.0f}  {c.frac:.3f}  apart {c.apart:.0f}")
    print()
    for sel, veto, against in ((pairs, False, "seqscreen"), (pairs, True, "seqscreen"),
                               (mt, False, SUMMED)):
        print("  " + rule(sel, veto=veto, candidate=k1, against=against))
    print("  beside it: " + rule(mt, veto=False, candidate=k2, against=SUMMED))


if __name__ == "__main__":
    {("check",): lambda: sys.exit(0 if check() else 1), ("run",): run, ("report",): report,
     ("guards", "none"): lambda: guards(None),
     **{("guards", f): (lambda f=f: guards(f)) for f in FAULTS}}[tuple(sys.argv[1:3])]()
