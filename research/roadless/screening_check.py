"""Guards of the screens on GeoffChurch/epistemetry (screening.py's rule, consolidated's presets),
each with its faults (a patch in one process, no file edited, as gap_clear_check.py's): a guard
passes on the code as it is and must fail under each of its faults.

    S=research/roadless/screening_check.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S target [fault]            # faults: dropped-uses, unstaked
    uv run python $S records [fault]           # fault: array-rates
    uv run python $S adapter [fault]           # faults: misaligned, gross-extra
    uv run python $S both-ways <guards,>       # each guard, then under each of its faults
    uv run python $S runs <workers>            # the acceptance runs -> screen_acceptance.csv
    uv run python $S equivalent <old.csv> <new.csv>   # two sets of acceptance runs compared

THE FIXTURE is DESIGNS (screen_designs.json, committed: the replays' frozen designs, reblock's
real-data acceptance fixture), read at one boundary (`designs`): per design its block ids, each
arm's Lens A gain over the base and its GPU-seconds, the base's GPU-seconds, and 100 orders of
its blocks (13 pairwise designs, then 5 many-arm ones).

`target`: each preset's target, as the preset's policy builds it at a decision, against the loss
recomputed here. For each preset, on every pairwise design, at order 0's first PREFIXES blocks
(as many as the design has), on the posterior's draws there (the arm's prefix, the bootstrap,
SCREEN_DRAWS draws): the stop is ValueOfInformation with no margin; realized-loss's risk is
FUTURE x the realized loss recomputed here (`realized_risk`, the edge's rates in Lens A per
GPU-second, equally weighted), its call the one recomputed, its price the mean rate;
fewest-wrong's target is AdoptKeepAsk over the bracket's ends, priced at the mean rate over
FUTURE over the arm's own stake recomputed here (`own_stake`). On READ_OFF at READ_AT blocks the
price a decision used is read off its reading: the target's, under the policy's own stop. Faults:
`dropped-uses` (RealizedLoss builds its target over 1 use, not FUTURE: the risk is off),
`unstaked` (OwnStake prices at price_rate / uses, the stake left out: the own price is off).
`records`: each preset's `Screen.record()` round-trips, through JSON and again, and names its
target's type (realized-loss epistemetry.SameTarget, fewest-wrong reblock.OwnStake). Fault
`array-rates` (consolidated.SCREEN_RATES the array np.geomspace returns, not a tuple of floats,
the slip seqscreen's arrays invite: the record refuses it, which the guard reports).
`adapter`: a live screen fed through screening.arrived_of is epistemetry's replay. Under
realized-loss, on r2/base (a near-tie ask), x2/p64 (an ask) and pPg/p256 (a keep) without the
veto, C1m4/p64 (a collapse, 12 blocks) in both settings, and p64/base with every gain 0 without
the veto, each on its orders 0-2 (18 cases): `advance` on the order's rows (the order's seed, the
setting's drop rule) is the replay's run of that order decision for decision, with its prefixes,
calls and drops, on all the blocks (most orders run past the stop: surplus blocks change
nothing); cut one block short of its stop (where it used more than one and no collapse stopped
it) it is `Running`, its decisions the replay's so far, waiting for the block the replay took
next. Beside the replay: the zero-gain design stops at the floor, keeping the default, "settled",
in every order; C1m4/p64 under the veto at its first block below -COLLAPSE, vetoed to KEEP,
"collapse"; without it, "ran out" at 12 (screening.why_of). Faults: `misaligned` (arrived_of
passes the covariate in the fixture's own block order, not the screen's: ProportionalTo prices
each block to come by another block's base time), `gross-extra` (b's gross GPU-seconds in the
extra column, not seconds - base_seconds: the targets read the wrong cost).

THE ACCEPTANCE RUNS (`runs`, screen_acceptance.csv, committed): each preset replayed with
epistemetry's `replay` on every pairwise design's 100 orders without the collapse veto, and with
it where it fires (C1m4/p64; elsewhere a design's veto runs equal its no-veto runs exactly, one
sample counted once), order o's draws keyed by (ACCEPTANCE_SEED, o): 1,400 runs a preset. One job
a (preset, design, setting, CHUNK orders), a parquet each under screen_acceptance_jobs/<rev7>
(git-ignored), resumable. `equivalent` compares two sets of them run against run on the same
order, by the test pre-registered 2026-10-18 (its docstring): where every draw is keyed by a seed
and its place, a pin whose streams differ cannot reproduce a run bit for bit, only its law.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import tomllib
import traceback
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from functools import cache
from multiprocessing import Pool
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from epistemetry.design.decide import (
    ADOPT,
    ASK,
    KEEP,
    AdoptKeepAsk,
    AdoptKeepAskByStake,
    Priced,
    PricedTarget,
    StandardErrors,
    ValueOfInformation,
)
from epistemetry.design.replay import AsksApart, Design, Order, Run, Truth, replay
from epistemetry.design.screen import (
    Arrived,
    Bootstrap,
    DropRule,
    NoVeto,
    Running,
    SameTarget,
    Screen,
    Seen,
    Stopped,
    TargetOf,
    ValueOfMoreUnits,
    advance,
)
from epistemetry.stats.records import revive

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import consolidated  # noqa: E402
import screening  # noqa: E402

DESIGNS = HERE / "screen_designs.json"   # committed: the designs, frozen
ORDERS = 100                 # orders a design, in DESIGNS
PREFIXES = (15, 20, 25)      # target: order 0's first blocks a target is checked at
READ_OFF = ("x2/p64", "r2/base", "pPg/p256")   # target: a decision's price read off its reading,
READ_AT = 20                                    # at this many blocks


# ---- the fixture ----------------------------------------------------------------------------

class Fixture(NamedTuple):
    """A design of DESIGNS: its blocks, each arm's Lens A gain over the base and GPU-seconds, the
    base's GPU-seconds, and the orders of its blocks every replay runs."""
    name: str
    many: bool
    blocks: tuple[str, ...]
    gain: np.ndarray            # (u, k)
    seconds: np.ndarray         # (u, k)
    base_seconds: np.ndarray    # (u,)
    orders: tuple[np.ndarray, ...]

    def design(self) -> Design:
        """epistemetry's: each arm's gain and extra GPU-seconds (the targets' pair), what each arm
        cost on each block, the base's runs, which exist (0), and the base's GPU-seconds on every
        block, the covariate the cost model scales by (ProportionalTo)."""
        u = len(self.blocks)
        return Design(outcomes=np.stack([self.gain, self.seconds - self.base_seconds[:, None]],
                                        axis=2),
                      costs=self.seconds, base_costs=np.zeros(u),
                      covariates={screening.BASE_SECONDS: self.base_seconds})


def _fixture(d: Mapping[str, Any]) -> Fixture:
    """One design of DESIGNS, checked: the JSON boundary, every key read by name."""
    name, many = d["name"], d["many"]
    if not isinstance(name, str) or not isinstance(many, bool):
        raise ValueError(f"a design's name is a string and `many` a bool: {name!r}, {many!r}")
    gain = np.ascontiguousarray(np.array(d["gain"], dtype=float).T)
    seconds = np.ascontiguousarray(np.array(d["seconds"], dtype=float).T)
    base_seconds = np.array(d["base_seconds"], dtype=float)
    blocks = tuple(d["blocks"])
    odd = [b for b in blocks if not isinstance(b, str)]
    if odd:
        raise ValueError(f"{name}: a block id is a string, got {odd}")
    u, k = len(blocks), len(d["arms"])
    orders = tuple(np.array(o, dtype=int) for o in d["orders"])
    if gain.shape != (u, k) or seconds.shape != (u, k) or base_seconds.shape != (u,):
        raise ValueError(f"{name}: shapes {gain.shape} {seconds.shape} {base_seconds.shape} for "
                         f"{u} blocks and {k} arms")
    if len(orders) != ORDERS or any(sorted(o.tolist()) != list(range(u)) for o in orders):
        raise ValueError(f"{name}: {ORDERS} orders of its {u} blocks expected")
    if u > len(common.HELD):
        raise ValueError(f"{name}: {u} blocks, more than the screens' horizons value "
                         f"({len(common.HELD)}, common.HELD's)")
    return Fixture(name=name, many=many, blocks=blocks, gain=gain, seconds=seconds,
                   base_seconds=base_seconds, orders=orders)


@cache
def designs() -> dict[str, Fixture]:
    """DESIGNS, by name, in its order: the pairwise designs, then the many-arm ones."""
    return {f.name: f for f in map(_fixture, json.loads(DESIGNS.read_text())["designs"])}


# ---- the loss and the stake, recomputed outside epistemetry ---------------------------------

def realized_risk(draws: np.ndarray, *, rates: tuple[float, ...],
                  weights: tuple[float, ...]) -> tuple[float, int]:
    """One arm's posterior draws of (gain, extra GPU-seconds) -> the expected loss of the call the
    draws' means favour at each of `rates` (Lens A per GPU-second; adopt where the mean of g - r c
    is above 0), E|g - r c| over the draws where that call is wrong, its mean over the rates
    weighted by `weights`; and that call over the rates (ADOPT at every one, KEEP at none, ASK
    where they disagree)."""
    g, c = draws[:, 0], draws[:, 1]
    loss, adopted = 0.0, []
    for r, w in zip(rates, weights, strict=True):
        v = g - r * c
        adopt = bool(v.mean() > 0.0)
        loss += w * float(np.mean(np.abs(v) * ((v > 0.0) != adopt)))
        adopted.append(adopt)
    return loss, ADOPT if all(adopted) else KEEP if not any(adopted) else ASK


def own_stake(outcomes: np.ndarray, *, price_rate: float) -> float:
    """The Lens A a wrong call on one arm costs at its means over its prefix (`outcomes`, (n, 2):
    gain, extra GPU-seconds): |mean gain - price_rate x mean extra|."""
    return float(abs(outcomes[:, 0].mean() - price_rate * outcomes[:, 1].mean()))


# ---- guards ---------------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class Edge:
    """The edge's numbers, recomputed here from consolidated's constants: the rates (Lens A per
    GPU-second), their weights, the mean rate and the uses a call governs."""
    rates: tuple[float, ...]
    weights: tuple[float, ...]
    price_rate: float
    future: float


def _edge() -> Edge:
    rates = tuple(float(r) for r in np.geomspace(*consolidated.RATES_PER_GPU_SECOND,
                                                 consolidated.RATE_POINTS))
    weights = (1.0 / consolidated.RATE_POINTS,) * consolidated.RATE_POINTS
    return Edge(rates=rates, weights=weights,
                price_rate=float(np.asarray(rates) @ np.asarray(weights)),
                future=consolidated.FUTURE)


#: One preset's checks of its target at a decision: (the target, the posterior's draws there, the
#: arm's prefix's outcomes, the edge) -> each check's name and whether it holds.
Checks = Callable[[PricedTarget, np.ndarray, np.ndarray, Edge], dict[str, bool]]


def _by_stake(target: PricedTarget, draws: np.ndarray, mine: np.ndarray,
              e: Edge) -> dict[str, bool]:
    loss, call = realized_risk(draws, rates=e.rates, weights=e.weights)
    return {"risk": math.isclose(float(target.risk(draws)), e.future * loss, rel_tol=1e-9,
                                 abs_tol=1e-18),
            "call": int(np.asarray(target.decide(draws)).reshape(-1)[0]) == call,
            "price": math.isclose(target.price, e.price_rate, rel_tol=1e-15)}


def _by_own_stake(target: PricedTarget, draws: np.ndarray, mine: np.ndarray,
                  e: Edge) -> dict[str, bool]:
    return {"target": (isinstance(target, Priced)
                       and target.target == AdoptKeepAsk(low=e.rates[0], high=e.rates[-1])),
            "price": math.isclose(target.price, e.price_rate / e.future
                                  / own_stake(mine, price_rate=e.price_rate), rel_tol=1e-12)}


#: Each preset by name, with its checks: every preset of screen_presets() is here.
TARGET_CHECKS: tuple[tuple[str, Checks], ...] = (("realized-loss", _by_stake),
                                                  ("fewest-wrong", _by_own_stake))


def target() -> list[str]:
    presets, e = consolidated.screen_presets(), _edge()
    if set(presets) != {name for name, _ in TARGET_CHECKS}:
        return [f"presets {sorted(presets)}, checked {[name for name, _ in TARGET_CHECKS]}"]
    pairwise = [f for f in designs().values() if not f.many]
    unknown = sorted(set(READ_OFF) - {f.name for f in pairwise})
    if unknown or READ_AT not in PREFIXES:
        return [f"READ_OFF names {unknown}, no pairwise design" if unknown
                else f"READ_AT {READ_AT} is not among PREFIXES {PREFIXES}"]
    bad: list[str] = []
    i = 0
    for name, checks in TARGET_CHECKS:
        policy = presets[name].policy
        if not isinstance(policy, ValueOfMoreUnits):
            bad.append(f"{name}: a {type(policy).__name__}, not ValueOfMoreUnits")
            continue
        stop = policy.stop == ValueOfInformation(margin=StandardErrors(k=0.0))
        held: Counter[str] = Counter()
        at = read = 0
        first = ""
        for f in pairwise:
            design, u = f.design(), len(f.blocks)
            arrived = design.arrived(f.orders[0])
            for n in sorted({min(m, u) for m in PREFIXES}):
                seen = arrived.seen(reach=[n], n=n, live=frozenset({0}))
                prefix = f.orders[0][:n]
                draws = Bootstrap().of(design.outcomes[prefix, 0], population=u).draw(
                    np.eye(2), np.random.default_rng([7, i]), draws=consolidated.SCREEN_DRAWS)
                t = policy.target.of(seen, arm=0)
                result = checks(t, draws, design.outcomes[prefix, 0], e)
                if f.name in READ_OFF and n == READ_AT:
                    j = policy.next(seen, rng=np.random.default_rng([8, i]))
                    price = j.readings[0]["price"] if 0 in j.readings else None
                    result["reading"] = (price is not None
                                         and math.isclose(price, t.price, rel_tol=1e-15)
                                         and j.rules[0] == policy.stop)
                    read += 1
                for check, ok in result.items():
                    held[check] += ok
                    if not ok and not first:
                        first = f"{f.name} at {n} blocks, {check}"
                at += 1
                i += 1
        counts = {check: (ok, read if check == "reading" else at) for check, ok in held.items()}
        wrong = ([f"{check} ({total - ok} of {total})" for check, (ok, total) in counts.items()
                  if ok != total] + ([] if stop else ["its stop"])
                 + ([] if read == len(READ_OFF) else [f"{read} decisions read of {len(READ_OFF)}"]))
        print(f"{name}: {at} prefixes of {len(pairwise)} designs, {read} decisions read; "
              + ", ".join(f"{check} {ok} of {total}" for check, (ok, total) in counts.items())
              + f"; stop {'ok' if stop else policy.stop}"
              + (" -> ok" if not wrong else f" -> FAILS (first: {first})" if first
                 else " -> FAILS"), flush=True)
        bad += [f"{name} {w}" for w in wrong]
    return bad


#: Each preset's target's record type, `Screen.record()`'s policy.target.
RECORD_TYPES = (("realized-loss", "epistemetry.SameTarget"), ("fewest-wrong", "reblock.OwnStake"))


def _target_type(record: Mapping[str, object]) -> object:
    """A screen's record's policy's target's type (a TypeError where the record nests no record
    there)."""
    policy = record["policy"]
    t = policy["target"] if isinstance(policy, Mapping) else None
    if not isinstance(t, Mapping):
        raise TypeError(f"the record's policy holds no target record: {policy!r}")
    return t["type"]


def records() -> list[str]:
    presets = consolidated.screen_presets()
    if set(presets) != {name for name, _ in RECORD_TYPES}:
        return [f"presets {sorted(presets)}, checked {[name for name, _ in RECORD_TYPES]}"]
    bad: list[str] = []
    for name, kind in RECORD_TYPES:
        screen = presets[name]
        try:
            rec = screen.record()
            through_json = revive(json.loads(json.dumps(rec, allow_nan=False))) == screen
            again = revive(rec).record() == rec
            named = _target_type(rec)
        except (TypeError, ValueError) as err:
            print(f"{name}: the round trip raises {type(err).__name__}: {err} -> FAILS",
                  flush=True)
            bad.append(f"{name} (raises)")
            continue
        ok = through_json and again and named == kind
        print(f"{name}: through JSON {'==' if through_json else '!='} the screen, revived and "
              f"recorded again {'==' if again else '!='} its record, target {named} "
              f"(want {kind}) -> {'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(name)
    return bad


ADAPTER_PRESET = "realized-loss"   # adapter: the preset its live screens run
ADAPTER_ORDERS = 3                 # adapter: each case on its design's orders 0 .. 2


class Expected(NamedTuple):
    """What a case's live screen must show on its own, beside the replay: each arm's prefix at the
    stop, why the arm stopped (`screening.why_of`), the arms the drop rule dropped with its call,
    and the calls (None: whatever the replay's)."""
    arm_units: tuple[int, ...]
    why: screening.Why
    vetoed: Mapping[int, int]
    calls: tuple[tuple[int, ...], ...] | None


class AdapterCase(NamedTuple):
    """A design the adapter guard screens live, with the collapse veto or without it, and what a
    screen of it must show on its own given an order's blocks (None: nothing beyond the replay)."""
    fixture: Fixture
    veto: bool
    expected: Callable[[np.ndarray], Expected | None]


def adapter_cases() -> list[AdapterCase]:
    """r2/base (a near-tie ask), x2/p64 (an ask) and pPg/p256 (a keep) without the veto; C1m4/p64
    (a collapse, 12 blocks, fewer than the floor) in both settings; and p64/base with every gain
    0 (a screen with nothing to learn) without the veto."""
    d = designs()
    c1m4 = d["C1m4/p64"]
    zero = d["p64/base"]._replace(name="p64/base, every gain 0",
                                  gain=np.zeros_like(d["p64/base"].gain))

    def collapse(i: np.ndarray) -> Expected:
        # Stopped at its first block below -COLLAPSE, its prefix ending there.
        first = int(np.flatnonzero(c1m4.gain[i, 0] < -consolidated.COLLAPSE)[0])
        return Expected(arm_units=(first + 1,), why="collapse", vetoed={0: KEEP},
                        calls=((KEEP,),))

    def ran_out(i: np.ndarray) -> Expected:
        return Expected(arm_units=(len(c1m4.blocks),), why="ran out", vetoed={}, calls=None)

    def settled(i: np.ndarray) -> Expected:
        return Expected(arm_units=(consolidated.SCREEN_FLOOR,), why="settled", vetoed={},
                        calls=((KEEP,),))

    def beside(i: np.ndarray) -> None:
        return None

    return [AdapterCase(fixture=d["r2/base"], veto=False, expected=beside),
            AdapterCase(fixture=d["x2/p64"], veto=False, expected=beside),
            AdapterCase(fixture=d["pPg/p256"], veto=False, expected=beside),
            AdapterCase(fixture=c1m4, veto=False, expected=ran_out),
            AdapterCase(fixture=c1m4, veto=True, expected=collapse),
            AdapterCase(fixture=zero, veto=False, expected=settled)]


class Live(NamedTuple):
    """A case's live screen at its stop, the replay's run of the order, and the blocks the cut
    screen was given (None: not cut)."""
    stopped: Stopped
    run: Run
    cut: int | None


def _live(preset: Screen, f: Fixture, *, veto: bool, o: int) -> Live:
    """`preset` live on `f`'s order `o`, its rows fed through screening.arrived_of, against
    epistemetry's replay of the order, by the plan's checks as written (an AssertionError is the
    first that fails): on all the blocks it is the replay decision for decision, with its
    prefixes, calls and drops; cut one block short of its stop (where it used more than one and no
    collapse stopped it), it is the replay's decisions so far, waiting for the replay's next
    block."""
    design = f.design()
    drop: DropRule = preset.drop if veto else NoVeto()
    i = f.orders[o]
    order = Order(units=tuple(int(x) for x in i), seed=(ACCEPTANCE_SEED, o))
    run = replay(design, preset.policy, orders=[order],
                 truth=Truth.of(design, preset.policy, seed=TRUTH_SEED, drop=drop),
                 scoring=AsksApart(ask=ASK)).runs[0]
    live = replace(preset, drop=drop, seed=order.seed)
    s = advance(live, ledger=(), arrived=screening.arrived_of(gain=f.gain[i, 0],
                                                              seconds=f.seconds[i, 0],
                                                              base_seconds=f.base_seconds[i]))
    assert isinstance(s, Stopped) and s.decisions == run.decisions and s.arm_units == run.arm_units
    assert np.array_equal(np.array(s.calls), run.calls) and frozenset(s.vetoed) == run.vetoed
    if run.units <= 1 or run.vetoed:
        return Live(stopped=s, run=run, cut=None)
    # the same screen cut short of its stop waits for exactly the blocks the replay took next
    m = run.units - 1
    cut = advance(live, ledger=(), arrived=screening.arrived_of(gain=f.gain[i[:m], 0],
                                                                seconds=f.seconds[i[:m], 0],
                                                                base_seconds=f.base_seconds[i]))
    assert isinstance(cut, Running) and cut.decisions == run.decisions[:len(cut.decisions)]
    assert cut.want[0][0] == m
    return Live(stopped=s, run=run, cut=m)


def adapter() -> list[str]:
    if not __debug__:
        return ["the adapter's checks are asserts, and python -O strips them"]
    preset = consolidated.screen_presets()[ADAPTER_PRESET]
    bad: list[str] = []
    cases = cut = 0
    for case in adapter_cases():
        f, u = case.fixture, len(case.fixture.blocks)
        for o in range(ADAPTER_ORDERS):
            where = f"{f.name} {setting(case.veto)} order {o}"
            cases += 1
            try:
                live = _live(preset, f, veto=case.veto, o=o)
            except AssertionError as err:
                failed = traceback.extract_tb(err.__traceback__)[-1].line
                print(f"{where}: FAILS on `{failed}`", flush=True)
                bad.append(f"{where} (`{failed}`)")
                continue
            s = live.stopped
            why = screening.why_of(s, arm=0, units=u)
            e = case.expected(f.orders[o])
            ok = e is None or (s.arm_units == e.arm_units and why == e.why
                               and dict(s.vetoed) == dict(e.vetoed)
                               and (e.calls is None or s.calls == e.calls))
            cut += live.cut is not None
            print(f"{where}: stops at {s.arm_units[0]} of {u} blocks, "
                  f"{screening.CALLS[s.calls[0][0]]} ({why}), {len(s.decisions)} decisions as the "
                  f"replay's; " + ("not cut" if live.cut is None
                                   else f"cut to {live.cut} blocks, it waits for the next "
                                        f"(position {live.cut})")
                  + ("" if e is None else f"; want {e.arm_units[0]} blocks, {e.why}")
                  + (" -> ok" if ok else " -> FAILS"), flush=True)
            if not ok:
                bad.append(f"{where} (stops at {s.arm_units}, {why}, vetoed {dict(s.vetoed)}, "
                           f"calls {s.calls}; want {e})")
    print(f"adapter: {cases} live screens against their replays, {cut} of them cut short",
          flush=True)
    return bad


GUARDS: dict[str, Callable[[], list[str]]] = {"target": target, "records": records,
                                              "adapter": adapter}


# ---- faults ---------------------------------------------------------------------------------

@contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    old = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


def _over_one_use(self: screening.RealizedLoss) -> TargetOf:
    """RealizedLoss's target over 1 use, not `uses`: its risk and its value FUTURE times small."""
    return SameTarget(target=AdoptKeepAskByStake(rates=self.rates, weights=self.weights, uses=1.0))


def _unstaked(self: screening.OwnStake, seen: Seen, *, arm: int) -> Priced:
    """OwnStake priced as the realized loss is, price_rate / uses, the arm's stake left out."""
    return Priced(target=AdoptKeepAsk(low=self.low, high=self.high),
                  price=self.price_rate / self.uses)


def _array_rates() -> AbstractContextManager[None]:
    """The edge's rates the array np.geomspace returns, as seqscreen builds its rates."""
    return _patched(consolidated, "SCREEN_RATES",
                    np.geomspace(*consolidated.RATES_PER_GPU_SECOND, consolidated.RATE_POINTS))


def _own_order(base_seconds: np.ndarray) -> np.ndarray:
    """The base's GPU-seconds as their design lists its blocks (DESIGNS's order, not a screen's),
    the design found by the values: every design on one base lists its blocks in one order,
    refused otherwise."""
    key = sorted(base_seconds.tolist())
    own = [f.base_seconds for f in designs().values() if sorted(f.base_seconds.tolist()) == key]
    if not own or any(not np.array_equal(b, own[0]) for b in own):
        raise ValueError(f"no one design's block order holds these {len(key)} base times")
    return own[0]


def _misaligned() -> AbstractContextManager[None]:
    """screening.arrived_of passing the covariate in the fixture's own block order, not the
    screen's: ProportionalTo prices each block to come by another block's base time."""
    original = screening.arrived_of

    def arrived_of(*, gain: np.ndarray, seconds: np.ndarray, base_seconds: np.ndarray) -> Arrived:
        a = original(gain=gain, seconds=seconds, base_seconds=base_seconds)
        return replace(a, covariates={screening.BASE_SECONDS: _own_order(base_seconds)})

    return _patched(screening, "arrived_of", arrived_of)


def _gross_extra() -> AbstractContextManager[None]:
    """screening.arrived_of putting b's gross GPU-seconds in the extra column, not its seconds over
    the base's: the targets read the wrong cost."""
    original = screening.arrived_of

    def arrived_of(*, gain: np.ndarray, seconds: np.ndarray, base_seconds: np.ndarray) -> Arrived:
        a = original(gain=gain, seconds=seconds, base_seconds=base_seconds)
        (arm,) = a.arms
        return replace(a, arms=(replace(arm, outcomes=np.column_stack([gain, seconds])),))

    return _patched(screening, "arrived_of", arrived_of)


FAULTS: dict[str, dict[str, Callable[[], AbstractContextManager[None]]]] = {
    "target": {"dropped-uses": lambda: _patched(screening.RealizedLoss, "target_of",
                                                _over_one_use),
               "unstaked": lambda: _patched(screening.OwnStake, "of", _unstaked)},
    "records": {"array-rates": _array_rates},
    "adapter": {"misaligned": _misaligned, "gross-extra": _gross_extra},
}


# ---- the acceptance runs --------------------------------------------------------------------

ACCEPTANCE = HERE / "screen_acceptance.csv"   # committed: the acceptance runs at the pin
JOBS = HERE / "screen_acceptance_jobs"        # git-ignored: a parquet a job, under the pin's rev7
PYPROJECT = HERE.parents[1] / "pyproject.toml"
ACCEPTANCE_SEED = 0     # order o's runs draw from the seed (ACCEPTANCE_SEED, o) (epistemetry's
                        # stream)
TRUTH_SEED = (0,)       # the truth's draws: every unit observed, the whole design's posterior is a
                        # point, so they decide nothing
CHUNK = 25              # orders a job. Exempt: it changes no result (a run draws from its order's
                        # seed alone, and the truth from TRUTH_SEED, whatever job it is in)
COLUMNS = ("preset", "plan", "veto", "order", "seed", "units", "call", "cost", "wrong", "asks",
           "truth", "wave_cost")


@cache
def pinned_rev() -> str:
    """epistemetry's pinned revision, the full SHA in pyproject.toml's [tool.uv.sources]: the one
    place it is written, read by name with no default."""
    rev = tomllib.loads(PYPROJECT.read_text())["tool"]["uv"]["sources"]["epistemetry"]["rev"]
    if not (isinstance(rev, str) and len(rev) == 40):
        raise ValueError(f"epistemetry's pin in {PYPROJECT} is a full SHA, got {rev!r}")
    return rev


class Cell(NamedTuple):
    """A design and a setting a preset's acceptance runs cover: with the screen's collapse veto
    (`Screen.drop`) or without it (epistemetry `NoVeto`)."""
    plan: str
    veto: bool


def comparison(screen: Screen) -> list[Cell]:
    """The cells `screen`'s acceptance runs cover: every pairwise design without the veto, and
    with it where it drops the arm on some block (C1m4/p64, the one design with a block below
    -COLLAPSE: elsewhere the veto never fires, so a design's veto runs equal its no-veto runs
    exactly and are one sample, counted once)."""
    cells = []
    for f in designs().values():
        if f.many:
            continue
        cells.append(Cell(plan=f.name, veto=False))
        if any(screen.drop.verdict(row) is not None for row in f.design().outcomes[:, 0]):
            cells.append(Cell(plan=f.name, veto=True))
    return cells


def setting(veto: bool) -> str:
    return "veto" if veto else "no veto"


class Job(NamedTuple):
    """A preset's runs on one cell, on orders first .. first + CHUNK - 1."""
    preset: str
    plan: str
    veto: bool
    first: int


def jobs() -> list[Job]:
    """Every job, a chunk of orders of every cell of both presets before the next chunk, so the
    first jobs time every kind of run."""
    presets = consolidated.screen_presets()
    names, plans = list(presets), list(designs())
    every = [Job(preset=name, plan=c.plan, veto=c.veto, first=first)
             for name, screen in presets.items() for c in comparison(screen)
             for first in range(0, ORDERS, CHUNK)]
    return sorted(every, key=lambda j: (j.first, plans.index(j.plan), j.veto,
                                        names.index(j.preset)))


def job_file(job: Job) -> Path:
    plan = job.plan.replace("/", "~")
    return (JOBS / pinned_rev()[:7]
            / f"{job.preset}_{plan}_{'veto' if job.veto else 'none'}_{job.first:02d}.parquet")


def _record(screen: Screen) -> str:
    """A screen's identity as a job's rows keep it: its record, as JSON."""
    return json.dumps(screen.record(), sort_keys=True, allow_nan=False)


def _only(calls: np.ndarray) -> int:
    """A pairwise screen's one call (one arm, one call), refused otherwise."""
    ((call,),) = calls.tolist()
    return int(call)


def wave_cost(design: Design, run: Run) -> float:
    """What `run` would cost were every step paid whole by every arm it named, as a live parallel
    wave pays it (GeoffChurch/epistemetry#10): decision d's step charges each arm it names its
    costs on the order's units d.at .. d.at + units - 1, past a veto included, and the base on
    every unit a wave covers. An arm's steps tile a prefix of the order (a live arm a step leaves
    out is dropped for good, checked here), so this is `replay`'s own charge, written as it
    writes it, on the waves' prefixes: `run.cost` exactly, where no veto cuts a step short."""
    order = np.asarray(run.order.units, dtype=int)
    reach = []
    for j in range(design.costs.shape[1]):
        end = 0
        for d in run.decisions:
            step = d.judgement.step
            if j in step.arms:
                if d.at != end:
                    raise ValueError(f"arm {j}'s steps do not tile a prefix: one at {d.at} after "
                                     f"one ending at {end}")
                end = d.at + step.units
        reach.append(end)
    return (sum(float(design.costs[order[:m], j].sum()) for j, m in enumerate(reach))
            + float(design.base_costs[order[:max(reach)]].sum()))


def acceptance_rows(job: Job) -> pd.DataFrame:
    """`job`'s runs: its preset's policy replayed on its orders under the cell's drop rule, each
    order's draws keyed by (ACCEPTANCE_SEED, o), scored by AsksApart against the policy's own
    calls on the whole design (`Truth.of`); each row with the screen's record (`_record`), so a
    job made under another screen is not gathered."""
    screen = consolidated.screen_presets()[job.preset]
    f = designs()[job.plan]
    design = f.design()
    drop: DropRule = screen.drop if job.veto else NoVeto()
    truth = Truth.of(design, screen.policy, seed=TRUTH_SEED, drop=drop)
    numbers = range(job.first, job.first + CHUNK)
    orders = [Order(units=tuple(int(x) for x in f.orders[o]), seed=(ACCEPTANCE_SEED, o))
              for o in numbers]
    rep = replay(design, screen.policy, orders=orders, truth=truth, scoring=AsksApart(ask=ASK))
    record = _record(screen)
    return pd.DataFrame([{"preset": job.preset, "plan": job.plan, "veto": job.veto, "order": o,
                          "seed": json.dumps(list(run.order.seed)), "units": run.units,
                          "call": _only(run.calls), "cost": run.cost, "wrong": run.wrong,
                          "asks": run.asks, "truth": _only(truth.calls),
                          "wave_cost": wave_cost(design, run), "screen": record}
                         for o, run in zip(numbers, rep.runs, strict=True)])


def _one(job: Job) -> tuple[Job, float]:
    """`job` run and written (whole or not at all: a killed job leaves no file), and its
    minutes."""
    start = time.monotonic()
    rows = acceptance_rows(job)
    f = job_file(job)
    part = f.with_name(f.stem + ".part.parquet")
    rows.to_parquet(part, index=False)
    os.replace(part, f)
    return job, (time.monotonic() - start) / 60


def in_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Runs in the presets' order, then the fixture's designs', without the veto first, then by
    order (an unknown preset or design raises)."""
    names, plans = list(consolidated.screen_presets()), list(designs())
    rank = frame.assign(_preset=[names.index(p) for p in frame["preset"]],
                        _plan=[plans.index(p) for p in frame["plan"]])
    return (rank.sort_values(["_preset", "_plan", "veto", "order"])
            .drop(columns=["_preset", "_plan"]).reset_index(drop=True))


def gather(every: Sequence[Job]) -> pd.DataFrame:
    """Every job's rows, each job checked whole and made under its preset's screen as it is now."""
    presets = consolidated.screen_presets()
    frames = []
    for job in every:
        rows = pd.read_parquet(job_file(job))
        if len(rows) != CHUNK or set(rows["screen"]) != {_record(presets[job.preset])}:
            raise SystemExit(f"{job_file(job)}: not {CHUNK} runs under {job.preset}'s screen as "
                             f"it is now; delete it and run again")
        frames.append(rows.drop(columns="screen"))
    every_run = in_order(pd.concat(frames, ignore_index=True))
    return pd.DataFrame({c: every_run[c] for c in COLUMNS})


def runs(*, workers: int) -> pd.DataFrame:
    """The acceptance runs: every job not yet written, on `workers` processes (exempt: a count
    that changes no result, each job's runs drawing from their own orders' seeds), then all of
    them gathered into ACCEPTANCE."""
    every = jobs()
    todo = [j for j in every if not job_file(j).exists()]
    print(f"epistemetry {pinned_rev()[:7]}: {len(every) - len(todo)} of {len(every)} jobs done "
          f"before", flush=True)
    job_file(every[0]).parent.mkdir(parents=True, exist_ok=True)
    with Pool(workers) as pool:
        for k, (job, minutes) in enumerate(pool.imap_unordered(_one, todo), start=1):
            print(f"{k}/{len(todo)} {job.preset} {job.plan} {setting(job.veto)} orders "
                  f"{job.first}-{job.first + CHUNK - 1} ({minutes:.1f} min, "
                  f"{minutes / CHUNK:.2f} a run)", flush=True)
    frame = gather(every)
    frame.to_csv(ACCEPTANCE, index=False)
    print(f"wrote {ACCEPTANCE.name}: {len(frame)} runs", flush=True)
    return frame


# ---- the comparison of two sets of acceptance runs ------------------------------------------

RESAMPLES = 10_000      # the bootstrap's resamples of the pairs (pre-registered 2026-10-18)
FLIPS = 100_000         # the sign-flip test's random flips (pre-registered)
ALPHA = 0.05            # Holm's family-wise level (pre-registered)
FLIP_CHUNK = 10_000     # flips drawn at once. Exempt: Generator.random fills its draws in order,
                        # so the flips are the same numbers however they are chunked
READ = ["preset", "plan", "veto", "order", "units", "call", "cost", "wrong", "asks", "truth"]
KEY = ["preset", "plan", "veto", "order"]


class Quantities(NamedTuple):
    """The per-run quantities the comparison reads: the blocks the stop used, the share of the
    design's whole cost (its `seconds.sum()`) spent, a wrong call and an ask (0 or 1 each)."""
    units: float
    cost_fraction: float
    wrong: float
    asks: float


#: The equivalence margins, pre-registered 2026-10-18: half the distance between the two presets'
#: means on the fourth replay's runs of the comparison set (realized-loss 15.03 blocks, 0.443,
#: 0.0479, 0.203; fewest-wrong 16.25, 0.473, 0.0264, 0.227).
DELTAS = Quantities(units=0.609, cost_fraction=0.0150, wrong=0.0107, asks=0.0121)


def quantities(runs: pd.DataFrame, *, fixtures: Mapping[str, Fixture]) -> pd.DataFrame:
    """Each run's Quantities, one column each, in their order."""
    whole = np.array([float(fixtures[p].seconds.sum()) for p in runs["plan"]])
    return pd.DataFrame({"units": runs["units"].to_numpy(dtype=float),
                         "cost_fraction": runs["cost"].to_numpy(dtype=float) / whole,
                         "wrong": runs["wrong"].to_numpy(dtype=float),
                         "asks": runs["asks"].to_numpy(dtype=float)})


def sign_flip(d: np.ndarray, *, flips: int, rng: np.random.Generator) -> float:
    """The two-sided sign-flip p of mean(d) = 0: (1 + #{|flipped mean| >= |mean|}) / (flips + 1),
    each flip negating each pair's difference with probability 1/2. A flipped sum within rounding
    of the observed one counts as at least as far out (the conservative side). For a 0-1
    quantity it is McNemar's exact test, by Monte Carlo."""
    total = float(d.sum())
    tol = 1e-9 * float(np.abs(d).sum())
    hits = 0
    for start in range(0, flips, FLIP_CHUNK):
        negated = rng.random((min(FLIP_CHUNK, flips - start), len(d))) < 0.5
        flipped = total - 2.0 * (negated @ d)
        hits += int(np.count_nonzero(np.abs(flipped) >= abs(total) - tol))
    return (1 + hits) / (flips + 1)


def mcnemar(d: np.ndarray) -> float:
    """McNemar's exact two-sided p for a 0-1 quantity's differences d (-1, 0 or 1 a pair), the
    value `sign_flip` estimates."""
    up, down = int(np.count_nonzero(d > 0)), int(np.count_nonzero(d < 0))
    n = up + down
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(up, down) + 1)) / (1 << n))


def holm(ps: Sequence[float]) -> list[float]:
    """Holm's step-down adjusted p-values: a test is rejected at family-wise ALPHA iff its
    adjusted p is at most ALPHA."""
    m, adjusted, running = len(ps), [0.0] * len(ps), 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda i: ps[i])):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


def _three(x: float) -> float:
    """x to three significant figures."""
    return float(f"{x:.3g}")


class _Test(NamedTuple):
    """One sign-flip test of the Holm family: whose, of what, and its p."""
    preset: str
    what: str
    p: float


def equivalent(old: pd.DataFrame, new: pd.DataFrame, *, fixtures: Mapping[str, Fixture],
               resamples: int, flips: int) -> list[str]:
    """Two sets of acceptance runs compared run against run on the same order (same preset,
    design, setting and order), by the test pre-registered 2026-10-18 (the epistemetry move's
    plan, Task 4), reading only READ; its failures, none when they are one policy's runs under
    two streams. Per preset and quantity (`Quantities`), d = new - old over the pairs:
    1. exact, where nothing is drawn: every (preset, design, setting)'s truth equal, and every run
       of a design no longer than the screens' floor (C1m4/p64) equal in units, call, cost (to
       1e-12 relative, a sum's rounding), wrong and asks;
    2. equivalence: the 95% percentile bootstrap interval of mean(d) (`resamples` resamples of a
       preset's pairs, np.random.default_rng(0)) inside +/- DELTAS, refusing to judge unless the
       old runs give DELTAS (half the two presets' gap) to three significant figures;
    3. no detectable shift: the sign-flip p of mean(d) (`flips` flips, np.random.default_rng(1)
       afresh for each test) for each preset's four quantities, and for the units of each
       (preset, design), Holm-corrected together at family-wise ALPHA, rejecting nothing.
    Prints, per preset, each quantity's means, mean(d), interval, delta and p (McNemar's exact p
    beside a 0-1 quantity's), the per-design table and the Holm verdicts."""
    a, b = (pd.DataFrame({c: frame[c] for c in READ}) for frame in (old, new))
    if a.duplicated(KEY).any() or b.duplicated(KEY).any():
        return ["pairs: a run appears twice in the old or the new runs"]
    m = a.merge(b, on=KEY, how="outer", suffixes=("_old", "_new"), indicator=True)
    alone = m[m["_merge"] != "both"]
    if len(alone):
        first = alone.iloc[0]
        return [f"pairs: {len(alone)} runs in one set only, first {first['preset']} "
                f"{first['plan']} {setting(bool(first['veto']))} order {first['order']} "
                f"({first['_merge']})"]
    m = in_order(m.drop(columns="_merge"))
    presets = list(pd.unique(m["preset"]))
    q_old = quantities(m.rename(columns=lambda c: c.removesuffix("_old")), fixtures=fixtures)
    q_new = quantities(m.rename(columns=lambda c: c.removesuffix("_new")), fixtures=fixtures)
    if len(presets) != 2:
        return [f"deltas: two presets give the margins, the runs hold {presets}"]
    means = [q_old[(m["preset"] == p).to_numpy()].mean() for p in presets]
    deltas = Quantities(*(abs(float(means[0][q] - means[1][q])) / 2 for q in Quantities._fields))
    if Quantities(*map(_three, deltas)) != DELTAS:
        return [f"deltas: the old runs give {Quantities(*map(_three, deltas))}, pre-registered "
                f"{DELTAS}: not the runs the margins were registered on"]

    bad: list[str] = []
    floor = {name for name, f in fixtures.items() if len(f.blocks) <= consolidated.SCREEN_FLOOR}
    exact: dict[str, tuple[int, int, int, int]] = {}
    for p in presets:
        g = m[m["preset"] == p]
        truths = equal_truths = 0
        for (plan, veto), cell in g.groupby(["plan", "veto"], sort=False):
            told, tnew = set(cell["truth_old"]), set(cell["truth_new"])
            truths += 1
            if len(told) == 1 and told == tnew:
                equal_truths += 1
            else:
                bad.append(f"exact: {p} {plan} {setting(bool(veto))} truth {sorted(tnew)}, "
                           f"stored {sorted(told)}")
        fixed = g[g["plan"].isin(floor)]
        equal_runs = 0
        for row in fixed.to_dict("records"):
            differ = [c for c in ("units", "call", "wrong", "asks")
                      if row[f"{c}_new"] != row[f"{c}_old"]]
            if not math.isclose(row["cost_new"], row["cost_old"], rel_tol=1e-12):
                differ.append("cost")
            if differ:
                bad.append(f"exact: {p} {row['plan']} {setting(bool(row['veto']))} order "
                           f"{row['order']}: " + ", ".join(
                               f"{c} {row[f'{c}_new']} against {row[f'{c}_old']}"
                               for c in differ))
            else:
                equal_runs += 1
        exact[p] = (equal_truths, truths, equal_runs, len(fixed))

    tests: list[_Test] = []
    rows: dict[str, list[tuple[str, float, float, float, float, float, tuple[float, float]]]] = {}
    for p in presets:
        at = (m["preset"] == p).to_numpy()
        n = int(at.sum())
        idx = np.random.default_rng(0).integers(0, n, size=(resamples, n))
        rows[p] = []
        for q in Quantities._fields:
            o, w = q_old[q].to_numpy()[at], q_new[q].to_numpy()[at]
            d = w - o
            lo, hi = (float(x) for x in np.percentile(d[idx].mean(axis=1), [2.5, 97.5]))
            delta = float(getattr(DELTAS, q))
            if not (-delta < lo and hi < delta):
                bad.append(f"interval: {p} {q} [{lo:+.4f}, {hi:+.4f}] not inside "
                           f"+/-{delta}")
            pv = sign_flip(d, flips=flips, rng=np.random.default_rng(1))
            binary = bool(np.isin(o, (0.0, 1.0)).all() and np.isin(w, (0.0, 1.0)).all())
            mc = mcnemar(d) if binary else math.nan
            tests.append(_Test(preset=p, what=q, p=pv))
            rows[p].append((q, float(o.mean()), float(w.mean()), float(d.mean()), pv, mc,
                            (lo, hi)))
    per_design: dict[str, list[tuple[str, int, float, float, float, int, int, int, int]]] = {}
    for p in presets:
        per_design[p] = []
        for plan in pd.unique(m.loc[m["preset"] == p, "plan"]):
            at = ((m["preset"] == p) & (m["plan"] == plan)).to_numpy()
            d = q_new["units"].to_numpy()[at] - q_old["units"].to_numpy()[at]
            pv = sign_flip(d, flips=flips, rng=np.random.default_rng(1))
            tests.append(_Test(preset=p, what=f"units on {plan}", p=pv))
            cell = m[at]
            per_design[p].append((str(plan), int(at.sum()),
                                  float(q_old["units"].to_numpy()[at].mean()),
                                  float(q_new["units"].to_numpy()[at].mean()), pv,
                                  int(cell["wrong_old"].sum()), int(cell["wrong_new"].sum()),
                                  int(cell["asks_old"].sum()), int(cell["asks_new"].sum())))
    adjusted = dict(zip(tests, holm([t.p for t in tests]), strict=True))
    for t, ap in adjusted.items():
        if ap <= ALPHA:
            bad.append(f"holm: {t.preset} {t.what} p {t.p:.2g}, Holm {ap:.2g} <= {ALPHA}")

    def verdict(t: _Test) -> str:
        return f"{adjusted[t]:.3g} {'REJECTED' if adjusted[t] <= ALPHA else 'kept'}"

    by = {(t.preset, t.what): t for t in tests}
    for p in presets:
        et, nt, er, nr = exact[p]
        print(f"{p}: {int((m['preset'] == p).sum())} pairs; exact: {et} of {nt} truths equal, "
              f"{er} of {nr} runs of {', '.join(sorted(floor))} equal", flush=True)
        print(f"  {'quantity':<14}{'old':>9}{'new':>9}{'mean(d)':>10}  {'95% interval':<21}"
              f"{'delta':>7}{'p (flips)':>11}{'McNemar':>9}  Holm", flush=True)
        for q, mo, mw, md, pv, mc, (lo, hi) in rows[p]:
            print(f"  {q:<14}{mo:>9.4f}{mw:>9.4f}{md:>+10.4f}  [{lo:+.4f}, {hi:+.4f}]  "
                  f"{float(getattr(DELTAS, q)):>7.4g}{pv:>11.2g}"
                  f"{'' if math.isnan(mc) else f'{mc:.2g}':>9}  {verdict(by[p, q])}", flush=True)
        print(f"  {'design':<15}{'pairs':>6}{'units old':>10}{'new':>8}{'mean(d)':>9}"
              f"{'p (flips)':>11}  {'Holm':<15}{'wrong old/new':>14}{'asks old/new':>13}",
              flush=True)
        for plan, n, uo, un, pv, wo, wn, ao, an in per_design[p]:
            print(f"  {plan:<15}{n:>6}{uo:>10.2f}{un:>8.2f}{un - uo:>+9.2f}{pv:>11.2g}  "
                  f"{verdict(by[p, f'units on {plan}']):<15}{f'{wo}/{wn}':>14}"
                  f"{f'{ao}/{an}':>13}", flush=True)
    rejected = [t for t, ap in adjusted.items() if ap <= ALPHA]
    print(f"Holm over {len(tests)} tests at family-wise {ALPHA}: "
          + ("no rejection" if not rejected else "rejects " + ", ".join(
              f"{t.preset} {t.what}" for t in rejected)), flush=True)
    return bad


# ---- the command line -----------------------------------------------------------------------

def main(guard: str, fault: str | None) -> None:
    if fault is None:
        bad = GUARDS[guard]()
    else:
        with FAULTS[guard][fault]():
            bad = GUARDS[guard]()
    verdict = "FAILS on " + ", ".join(bad) if bad else "passes"
    print(f"{guard}{'' if fault is None else f' under the fault {fault}'}: {verdict}", flush=True)
    if bad and fault is None:
        raise SystemExit(1)
    if not bad and fault is not None:
        raise SystemExit(f"the fault {fault} went unseen: the guard guards nothing")


def both_ways(guards: list[str]) -> None:
    """Each guard as it is, then under each of its faults (FAULTS), one after another in this
    process. Raises naming every run not as `main` expects."""
    wrong = []
    for guard in guards:
        for fault in (None, *FAULTS[guard]):
            try:
                main(guard, fault)
            except SystemExit as e:
                wrong.append(f"{guard}{'' if fault is None else f' under {fault}'}: {e.code}")
    print(f"both ways, {', '.join(guards)}: "
          + ("as expected" if not wrong else "NOT as expected: " + "; ".join(wrong)), flush=True)
    if wrong:
        raise SystemExit(1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 2 and a[0] == "both-ways" and set(a[1].split(",")) <= set(GUARDS):
        both_ways(a[1].split(","))
        raise SystemExit(0)
    if len(a) == 2 and a[0] == "runs" and a[1].isdigit() and int(a[1]) >= 1:
        runs(workers=int(a[1]))
        raise SystemExit(0)
    if len(a) == 3 and a[0] == "equivalent":
        failures = equivalent(pd.read_csv(a[1]), pd.read_csv(a[2]), fixtures=designs(),
                              resamples=RESAMPLES, flips=FLIPS)
        print("pass" if not failures else "FAILS on " + "; ".join(failures), flush=True)
        raise SystemExit(1 if failures else 0)
    if len(a) not in (1, 2) or a[0] not in GUARDS or (len(a) == 2 and a[1] not in FAULTS[a[0]]):
        raise SystemExit(__doc__)
    main(a[0], a[1] if len(a) == 2 else None)
