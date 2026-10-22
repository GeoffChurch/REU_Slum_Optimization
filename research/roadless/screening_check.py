"""Guards of the screens on GeoffChurch/epistemetry (screening.py's rule, consolidated's presets),
each with its faults (a patch in one process, no file edited, as gap_clear_check.py's): a guard
passes on the code as it is and must fail under each of its faults.

    S=research/roadless/screening_check.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S target [fault]            # faults: dropped-uses, unstaked
    uv run python $S records [fault]           # fault: array-rates
    uv run python $S adapter [fault]           # faults: misaligned, gross-extra
    uv run python $S why [fault]               # fault: ran-out-first
    uv run python $S screens [fault]           # faults: shuffled, altered-record, wave-seeded
    uv run python $S acceptance [fault] [--workers W]   # fault: dropped-uses
    uv run python $S both-ways <guards,> [--workers W]  # each guard, then under each of its faults
    uv run python $S runs [--workers W]        # the acceptance runs -> screen_acceptance.csv
    uv run python $S equivalent <old.csv> <new.csv>   # two sets of acceptance runs compared

`--workers W` is the processes `runs` and the pooled guards (POOLED: `acceptance`) run on, 1 when
not given (exempt from the required-parameter rule: a count that changes no result, each run
drawing from its own seed). It is refused on a command that runs on no pool (a both-ways run
takes it when one of its guards is pooled).

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
`why`: why_of's precedence, on a screen `advance` stops so: C1m4/p64 under the veto, its gains
clipped at 0 but on order 0's last block, set below -COLLAPSE, is vetoed on the screen's last
block, its prefix the whole order: "collapse", not "ran out". Fault `ran-out-first` (why_of asks
whether the prefix is the whole order before whether the arm was vetoed: the last-block collapse
reads "ran out").
`screens`: the screens' committed decisions (consolidated's summary_screen.csv and
screen_decisions.json, under consolidated.SCREENS_METRIC) are what the code derives from the
screens' rows. The preset is the one whose Screen is the JSON's. On the screens' data
(`consolidated.screen_data`), every screen derived afresh, and every screen continued from its
stored ledger (`screen_table`, wave 0), is the committed table (every column; a float one to 1e-9
relative) and the stored states, record for record, with their blocks; every stored state
round-trips through JSON; every judged arm's verdict, re-derived from its rule and reading alone,
is its take; every rule is the preset's stop, and every reading's price the target's (PRICES,
from the edge recomputed here: realized-loss's the mean rate, fewest-wrong's the mean rate over
FUTURE over the arm's own stake on its prefix then; to 1e-12 relative). A live wave's ledger
(epistemetry's resume property: continuing from a prefix of a ledger gives what starting afresh
gives): the longest stored ledger cut to the first half of its decisions (rounded up), as a wave
stores it, Running for the rest of its last decision's step, on the blocks run when that decision
was made (`_cut`), read back through the JSON and continued on the whole rows, is the screens
derived afresh, the table and every state. And refused before any decision: a held-out block's
default time dropped from the pass's rows (`screen_data`, naming the block), the JSON's metric or
screen another's, a stored state or screen record epistemetry cannot read back (a field dropped,
a field added: `read_decisions`, saying that removing the JSON starts afresh), a variant's stored
blocks starting off its order (`screen_table`, naming the variant).
Faults: `shuffled` (common.HELD reshuffled by np.random.default_rng(1): the rows are off every
screen's order, which screen_data refuses), `altered-record` (one judged reading's stored price
times 1 + 1e-6: the screens derived afresh are not the stored ones, and the price is not the
target's), `wave-seeded` (consolidated's advance continuing a ledger under a seed keyed by the
ledger's length, each wave its own stream: the cut ledger's continuation draws other numbers than
the screens derived afresh).
`acceptance`: reblock's real-data acceptance test for epistemetry's pin (its spec, owner
2026-10-16). realized-loss, as the presets build it now, replays ACCEPTANCE's runs of it on orders
0-9 (13 pairwise designs without the veto, C1m4/p64 with it too: 140 runs, each order's draws
from the row's seed), a design and setting at a time in the file's order, on a pool of W
processes: each run's units, call, wrong, asks and truth equal the row's exactly, its cost to
1e-12 relative (a sum's rounding). It returns at the first design that differs. Fault
`dropped-uses` (as `target`'s). A pin move whose streams differ fails it: the runs are then
regenerated (`runs`) and pass `equivalent` against the committed ones first (screening.py).

THE ACCEPTANCE RUNS (`runs`, screen_acceptance.csv, committed): each preset replayed with
epistemetry's `replay` on every pairwise design's 100 orders without the collapse veto, and with
it where it fires (C1m4/p64; elsewhere a design's veto runs equal its no-veto runs exactly, one
sample counted once), order o's draws keyed by (ACCEPTANCE_SEED, o): 1,400 runs a preset. One job
a (preset, design, setting, CHUNK orders), a parquet each under screen_acceptance_jobs/<rev7>
(git-ignored), resumable. `equivalent` compares two sets of them run against run on the same
order, by the test pre-registered 2026-10-18 (its docstring), its margins half the two presets'
gap on the old runs (`deltas_of`): where every draw is keyed by a seed and its place, a pin whose
streams differ cannot reproduce a run bit for bit, only its law.
"""
from __future__ import annotations

import copy
import io
import json
import math
import os
import sys
import time
import tomllib
import traceback
from collections import Counter
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from functools import cache, partial
from multiprocessing import Pool, get_context
from pathlib import Path
from typing import Any, NamedTuple, Protocol

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
from epistemetry.design.replay import AsksApart, Design, Order, Replay, Run, Truth, replay
from epistemetry.design.screen import (
    Arrived,
    Bootstrap,
    Decision,
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
import metrics  # noqa: E402
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


ADAPTER_PRESET = "realized-loss"   # adapter and why: the preset their live screens run
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
    """A case's live screen at its stop, and the blocks the cut screen was given (None: not
    cut)."""
    stopped: Stopped
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
        return Live(stopped=s, cut=None)
    # the same screen cut short of its stop waits for exactly the blocks the replay took next
    m = run.units - 1
    cut = advance(live, ledger=(), arrived=screening.arrived_of(gain=f.gain[i[:m], 0],
                                                                seconds=f.seconds[i[:m], 0],
                                                                base_seconds=f.base_seconds[i]))
    assert isinstance(cut, Running) and cut.decisions == run.decisions[:len(cut.decisions)]
    assert cut.want[0][0] == m
    return Live(stopped=s, cut=m)


LAST_COLLAPSE = "C1m4/p64"   # why: the design whose screen it collapses on its last block


def last_block_collapse(preset: Screen) -> list[str]:
    """why_of's precedence, on a screen `advance` stops so: LAST_COLLAPSE (fewer blocks than the
    floor, so the floor's first step runs them all) under `preset`'s veto, its gains clipped at 0
    but on order 0's last block, set below -COLLAPSE. The arm is vetoed on the screen's last block,
    so its prefix is the whole order, and why_of must read "collapse", not "ran out"."""
    f = designs()[LAST_COLLAPSE]
    i, u = f.orders[0], len(f.blocks)
    gain = np.maximum(f.gain[:, 0], 0.0)
    gain[i[-1]] = -2 * consolidated.COLLAPSE
    arrived = screening.arrived_of(gain=gain[i], seconds=f.seconds[i, 0],
                                   base_seconds=f.base_seconds[i])
    s = advance(preset, ledger=(), arrived=arrived)
    where = f"{LAST_COLLAPSE} under the veto, only order 0's last block below -COLLAPSE"
    if not isinstance(s, Stopped):
        print(f"{where}: {type(s).__name__}, not Stopped -> FAILS", flush=True)
        return [f"{where} ({type(s).__name__})"]
    why = screening.why_of(s, arm=0, units=u)
    ok = s.arm_units == (u,) and dict(s.vetoed) == {0: KEEP} and why == "collapse"
    print(f"{where}: stops at {s.arm_units[0]} of {u} blocks, vetoed {dict(s.vetoed)}, {why}; "
          f"want {u} blocks, vetoed {{0: {KEEP}}}, collapse" + (" -> ok" if ok else " -> FAILS"),
          flush=True)
    return [] if ok else [f"{where} (stops at {s.arm_units}, vetoed {dict(s.vetoed)}, {why})"]


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


def why() -> list[str]:
    return last_block_collapse(consolidated.screen_presets()[ADAPTER_PRESET])


#: A preset's price a judged reading must hold: (the arm's prefix's outcomes at the decision, the
#: edge recomputed here) -> the price.
Price = Callable[[np.ndarray, Edge], float]


def _stake_price(prefix: np.ndarray, e: Edge) -> float:
    """realized-loss's: its target's own price (the mean rate), whatever the arm's prefix."""
    return AdoptKeepAskByStake(rates=e.rates, weights=e.weights, uses=e.future).price


def _own_price(prefix: np.ndarray, e: Edge) -> float:
    """fewest-wrong's: the mean rate over FUTURE over the arm's own stake on its prefix."""
    return e.price_rate / e.future / own_stake(prefix, price_rate=e.price_rate)


#: Each preset by name, with its price: every preset of screen_presets() is here.
PRICES: tuple[tuple[str, Price], ...] = (("realized-loss", _stake_price),
                                         ("fewest-wrong", _own_price))
SUMMARY = consolidated.ROWS / "summary_screen.csv"   # screens: the committed table


def _table_differences(table: pd.DataFrame, committed: pd.DataFrame) -> list[str]:
    """The columns in which `table`, read back from its CSV as the committed one is read, is not
    `committed`: exactly, but a float column to 1e-9 relative (NaN where NaN)."""
    new = pd.read_csv(io.StringIO(table.to_csv(index=False)))
    if list(new.columns) != list(committed.columns) or new["variant"].tolist() != committed[
            "variant"].tolist():
        return [f"columns {list(new.columns)} and variants {new['variant'].tolist()}"]
    differ = []
    for c in new.columns:
        a, b = new[c], committed[c]
        if pd.api.types.is_float_dtype(a) and pd.api.types.is_float_dtype(b):
            same = bool(np.allclose(a.to_numpy(), b.to_numpy(), rtol=1e-9, atol=0.0,
                                    equal_nan=True))
        else:
            same = a.equals(b)
        if not same:
            differ.append(c)
    return differ


def _refused(what: str, call: Callable[[], object], *, naming: str) -> list[str]:
    """`call` must raise SystemExit, its message naming `naming`: a negative case of `screens`.
    Any other error is a failure of the case, reported as one, not a crash of the guard."""
    try:
        call()
    except SystemExit as e:
        ok = naming in str(e.code)
        print(f"  {what}: refused{'' if ok else f', not naming {naming}: {e.code}'}"
              + (" -> ok" if ok else " -> FAILS"), flush=True)
        return [] if ok else [f"{what} (refused, not naming {naming})"]
    except Exception as e:
        print(f"  {what}: raised {type(e).__name__} ({e}), not refused -> FAILS", flush=True)
        return [f"{what} (raised {type(e).__name__}, not refused)"]
    print(f"  {what}: not refused -> FAILS", flush=True)
    return [f"{what} (not refused)"]


def _cut(stored: consolidated.Stored, *, run: Sequence[str]) -> consolidated.Stored:
    """`stored` as a live wave would have stored it part way: the first half of its decisions
    (rounded up), Running for the rest of the last one's step (`advance`'s `want`: each arm it
    names, from its prefix then to the step's end, the units positions in the screen's order), on
    the blocks run when that decision was made (the first `at` of `run`). Refused (ValueError)
    unless the cut keeps a judged decision and leaves one to make."""
    ds = stored.state.decisions
    k = (len(ds) + 1) // 2
    last = ds[k - 1]
    step = last.judgement.step
    if not (any(d.judgement.rules for d in ds[:k]) and any(d.judgement.rules for d in ds[k:])
            and step.arms):
        raise ValueError(f"a ledger of {len(ds)} decisions cut to {k} keeps no judged decision, "
                         f"or leaves none to make")
    want = {a: tuple(range(last.arm_units[a], last.at + step.units)) for a in sorted(step.arms)}
    return consolidated.Stored(blocks=tuple(run[:last.at]),
                               state=Running(decisions=ds[:k], want=want))


def _continued_from_cut(name: str, preset: Screen, j: Any, *,
                        data: Mapping[str, consolidated.ScreenData],
                        stored: Mapping[str, consolidated.Stored],
                        afresh: consolidated.ScreenTable, metric: str) -> list[str]:
    """The live-wave path: the longest stored ledger cut part way (`_cut`), written into the
    stored JSON `j` in its place, read back (`read_decisions`) and continued on the whole rows
    (`screen_table`, wave 0), is `afresh`, the screens derived afresh: the table, and every state
    with its blocks."""
    if not stored:
        return ["cut: no stored ledger to cut"]
    v = max(stored, key=lambda w: len(stored[w].state.decisions))
    try:
        cut = _cut(stored[v], run=data[v].run)
    except ValueError as e:
        print(f"{name} cut: {v}'s ledger: {e} -> FAILS", flush=True)
        return [f"cut: {e}"]
    entry = json.loads(json.dumps({"blocks": list(cut.blocks), "state": cut.state.record()},
                                  allow_nan=False))
    partway = {**j, "screens": {**j["screens"], v: entry}}
    try:
        with _patched(consolidated, "_decisions_json", lambda: partway):
            ledgers = consolidated.read_decisions(preset, metric=metric)
        st = consolidated.screen_table(preset, data=data, ledgers=ledgers, wave=0, metric=metric)
    except SystemExit as refusal:
        print(f"{name} cut: refused: {refusal.code} -> FAILS", flush=True)
        return [f"cut: refused ({refusal.code})"]
    differ = _table_differences(st.table,
                                pd.read_csv(io.StringIO(afresh.table.to_csv(index=False))))
    unlike = sorted(w for w in set(st.states) | set(afresh.states)
                    if w not in st.states or w not in afresh.states
                    or st.states[w].state.record() != afresh.states[w].state.record()
                    or st.states[w].blocks != afresh.states[w].blocks)
    running = isinstance(ledgers[v].state, Running)
    print(f"{name} continued from {v}'s ledger cut to {len(cut.state.decisions)} of "
          f"{len(stored[v].state.decisions)} decisions ({'Running' if running else 'NOT Running'}"
          f" on {len(cut.blocks)} blocks): the table "
          + ("as afresh" if not differ else f"differs from afresh in {differ}") + "; the states "
          + ("as afresh" if not unlike else f"differ from afresh on {unlike}")
          + (" -> ok" if running and not differ and not unlike else " -> FAILS"), flush=True)
    return ([] if running else [f"cut: {v}'s cut ledger read back not Running"]) + (
        [f"cut: the table differs from afresh in {differ}"] if differ else []) + (
        [f"cut: the states differ from afresh on {unlike}"] if unlike else [])


def screens() -> list[str]:
    j = consolidated._decisions_json()
    if j is None:
        print(f"screens: no {consolidated.DECISIONS.name} -> FAILS", flush=True)
        return [f"no {consolidated.DECISIONS.name}"]
    presets = consolidated.screen_presets()
    if set(presets) != {name for name, _ in PRICES}:
        return [f"presets {sorted(presets)}, priced {[name for name, _ in PRICES]}"]
    recorded = revive(j["screen"])
    named = [name for name, screen in presets.items() if screen == recorded]
    if len(named) != 1:
        print(f"screens: {consolidated.DECISIONS.name}'s screen is {len(named)} presets' -> FAILS",
              flush=True)
        return [f"the stored screen is {len(named)} presets'"]
    (name,), metric = named, consolidated.SCREENS_METRIC
    preset = presets[name]
    policy = preset.policy
    if not isinstance(policy, ValueOfMoreUnits):
        return [f"{name}: a {type(policy).__name__}, not ValueOfMoreUnits"]
    try:
        data = consolidated.screen_data(metric=metric)
        stored = consolidated.read_decisions(preset, metric=metric)
        tables = {"continued": consolidated.screen_table(preset, data=data, ledgers=stored,
                                                         wave=0, metric=metric),
                  "afresh": consolidated.screen_table(preset, data=data, ledgers={}, wave=0,
                                                      metric=metric)}
    except SystemExit as refusal:
        print(f"screens: refused: {refusal.code} -> FAILS", flush=True)
        return [f"refused ({refusal.code})"]
    committed = pd.read_csv(SUMMARY)
    bad: list[str] = []
    for how, st in tables.items():
        differ = _table_differences(st.table, committed)
        if set(st.states) != set(j["screens"]):
            unlike = sorted(set(st.states) ^ set(j["screens"]))
        else:
            unlike = [v for v, x in st.states.items()
                      if x.state.record() != j["screens"][v]["state"]
                      or list(x.blocks) != j["screens"][v]["blocks"]]
        print(f"{name} {how}: {len(st.states)} screens; the table "
              + ("as committed" if not differ else f"differs in {differ}") + "; the states "
              + ("as stored" if not unlike else f"differ on {unlike}")
              + (" -> ok" if not differ and not unlike else " -> FAILS"), flush=True)
        bad += ([f"{how}: the table differs in {differ}"] if differ else []) + (
            [f"{how}: the states differ on {unlike}"] if unlike else [])
    bad += _continued_from_cut(name, preset, j, data=data, stored=stored, afresh=tables["afresh"],
                               metric=metric)

    e, price_of = _edge(), dict(PRICES)[name]
    judged = 0
    wrong: Counter[str] = Counter()
    first: dict[str, str] = {}
    for v, x in stored.items():
        rec = j["screens"][v]["state"]
        if revive(json.loads(json.dumps(rec, allow_nan=False))).record() != rec:
            wrong["round trip"] += 1
            first.setdefault("round trip", v)
        for d in x.state.decisions:
            for arm, rule in d.judgement.rules.items():
                judged += 1
                reading = d.judgement.readings[arm]
                price = reading["price"]
                want = price_of(data[v].arrived.arms[arm].outcomes[:d.arm_units[arm]], e)
                for check, ok in (("verdict", rule.met_reading(reading)
                                   == (arm not in d.judgement.step.arms)),
                                  ("rule", rule == policy.stop),
                                  ("price", price is not None
                                   and math.isclose(price, want, rel_tol=1e-12))):
                    if not ok:
                        wrong[check] += 1
                        first.setdefault(check, f"{v} at {d.at} blocks")
    totals = {"round trip": len(stored), "verdict": judged, "rule": judged, "price": judged}
    print(f"{name} records: {len(stored)} stored states, {judged} judged readings; "
          + ", ".join(f"{check} {n - wrong[check]} of {n}" for check, n in totals.items())
          + (" -> ok" if not wrong else f" -> FAILS (first: {first})"), flush=True)
    bad += [f"{check} ({n}, first {first[check]})" for check, n in wrong.items()]
    if not judged:
        bad.append("no judged reading stored: the record checks checked nothing")

    rest = [(v, d.eligible[len(d.run)]) for v, d in data.items() if len(d.eligible) > len(d.run)]
    other = next((s for n, s in presets.items() if n != name), None)
    ahead = next((v for v in data if v in stored), None)
    if not rest or other is None or ahead is None:
        return bad + ["the negative cases have no rest block, other preset or stored ledger"]
    v, block = rest[0]
    original = consolidated._pass_rows

    def untimed(*, metric: str) -> pd.DataFrame:
        rows = original(metric=metric).copy()
        rows.loc[block, "t"] = np.nan
        return rows

    print("negative cases, each refused before any decision:", flush=True)
    with _patched(consolidated, "_pass_rows", untimed):
        bad += _refused(f"{v}'s first rest block {block} untimed in the pass's rows",
                        lambda: consolidated.screen_data(metric=metric), naming=block)
    with _patched(consolidated, "_decisions_json", lambda: {**j, "metric": metrics.PER_BLOCK}):
        bad += _refused("the stored metric another's (metrics.PER_BLOCK)",
                        lambda: consolidated.read_decisions(preset, metric=metric),
                        naming=metrics.PER_BLOCK)
    with _patched(consolidated, "_decisions_json", lambda: {**j, "screen": other.record()}):
        bad += _refused("the stored screen the other preset's",
                        lambda: consolidated.read_decisions(preset, metric=metric),
                        naming="injected preset")
    starts = f"removing {consolidated.DECISIONS.name} starts the screens afresh"
    state = j["screens"][ahead]["state"]
    dropped = {**j, "screens": {**j["screens"], ahead: {
        **j["screens"][ahead], "state": {k: x for k, x in state.items() if k != "decisions"}}}}
    with _patched(consolidated, "_decisions_json", lambda: dropped):
        bad += _refused(f"{ahead}'s stored state without its decisions, which revive cannot read",
                        lambda: consolidated.read_decisions(preset, metric=metric), naming=starts)
    with _patched(consolidated, "_decisions_json",
                  lambda: {**j, "screen": {**j["screen"], "pin": pinned_rev()}}):
        bad += _refused("the stored screen with a field Screen lacks, which revive cannot read",
                        lambda: consolidated.read_decisions(preset, metric=metric), naming=starts)
    off = {**stored, ahead: stored[ahead]._replace(blocks=(common.TUNING[0],)
                                                   + stored[ahead].blocks[1:])}
    bad += _refused(f"{ahead}'s stored blocks starting {common.TUNING[0]}, off its order",
                    lambda: consolidated.screen_table(preset, data=data, ledgers=off, wave=0,
                                                      metric=metric), naming=ahead)
    return bad


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


def _dropped_uses() -> AbstractContextManager[None]:
    """RealizedLoss building its target over 1 use (`_over_one_use`)."""
    return _patched(screening.RealizedLoss, "target_of", _over_one_use)


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


def _ran_out_first(s: Stopped, *, arm: int, units: int) -> screening.Why:
    """why_of with its first two branches swapped: a prefix the whole order reads "ran out" before
    the veto is looked at."""
    if s.arm_units[arm] == units:
        return "ran out"
    if arm in s.vetoed:
        return "collapse"
    return "settled"


def _shuffled() -> AbstractContextManager[None]:
    """common.HELD reshuffled by np.random.default_rng(1): the screens' rows, run in its order, are
    off every screen's order."""
    held = common.HELD
    return _patched(common, "HELD",
                    tuple(held[i] for i in np.random.default_rng(1).permutation(len(held))))


def _altered_record() -> AbstractContextManager[None]:
    """The stored ledgers as loaded with one judged reading's price times 1 + 1e-6 (the first
    stored screen's first judged decision's, its first arm's)."""
    original = consolidated._decisions_json

    def altered() -> Any:
        j = copy.deepcopy(original())
        if j is None:
            return None
        for e in j["screens"].values():
            for d in e["state"]["decisions"]:
                readings = d["judgement"]["readings"]
                if readings:
                    readings[min(readings, key=int)]["price"] *= 1 + 1e-6
                    return j
        raise SystemExit("altered-record: no judged reading stored to alter")

    return _patched(consolidated, "_decisions_json", altered)


def _wave_seeded() -> AbstractContextManager[None]:
    """consolidated's `advance` continuing a ledger under a seed keyed by the ledger's length (each
    wave its own stream): the decisions a continued screen adds draw other numbers than a screen
    derived afresh draws in their places."""
    def continued(screen: Screen, *, ledger: tuple[Decision, ...],
                  arrived: Arrived) -> Running | Stopped:
        seeded = replace(screen, seed=(*screen.seed, len(ledger))) if ledger else screen
        return advance(seeded, ledger=ledger, arrived=arrived)     # epistemetry's own

    return _patched(consolidated, "advance", continued)


#: Each guard's faults by name (and so every guard's name: each has at least one).
FAULTS: dict[str, dict[str, Callable[[], AbstractContextManager[None]]]] = {
    "target": {"dropped-uses": _dropped_uses,
               "unstaked": lambda: _patched(screening.OwnStake, "of", _unstaked)},
    "records": {"array-rates": _array_rates},
    "adapter": {"misaligned": _misaligned, "gross-extra": _gross_extra},
    "why": {"ran-out-first": lambda: _patched(screening, "why_of", _ran_out_first)},
    "screens": {"shuffled": _shuffled, "altered-record": _altered_record,
                "wave-seeded": _wave_seeded},
    "acceptance": {"dropped-uses": _dropped_uses},
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


def replayed(screen: Screen, f: Fixture, *, veto: bool, orders: Sequence[Order]) -> Replay:
    """`screen`'s policy replayed on `f`'s `orders` with the screen's collapse veto or without it,
    scored by AsksApart against the policy's own calls on the whole design (`Truth.of`,
    TRUTH_SEED) under the same drop rule."""
    design = f.design()
    drop: DropRule = screen.drop if veto else NoVeto()
    truth = Truth.of(design, screen.policy, seed=TRUTH_SEED, drop=drop)
    return replay(design, screen.policy, orders=orders, truth=truth, scoring=AsksApart(ask=ASK))


def acceptance_rows(job: Job) -> pd.DataFrame:
    """`job`'s runs (`replayed`), each order's draws keyed by (ACCEPTANCE_SEED, o); each row with
    the screen's record (`_record`), so a job made under another screen is not gathered."""
    screen = consolidated.screen_presets()[job.preset]
    f = designs()[job.plan]
    numbers = range(job.first, job.first + CHUNK)
    orders = [Order(units=tuple(int(x) for x in f.orders[o]), seed=(ACCEPTANCE_SEED, o))
              for o in numbers]
    rep = replayed(screen, f, veto=job.veto, orders=orders)
    design, record = f.design(), _record(screen)
    return pd.DataFrame([{"preset": job.preset, "plan": job.plan, "veto": job.veto, "order": o,
                          "seed": json.dumps(list(run.order.seed)), "units": run.units,
                          "call": _only(run.calls), "cost": run.cost, "wrong": run.wrong,
                          "asks": run.asks, "truth": _only(rep.truth.calls),
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
    """Every job's rows, each job checked whole, made under its preset's screen as it is now, and
    its own: its preset's, design's and setting's runs on its orders in turn, each order o's seed
    (ACCEPTANCE_SEED, o)."""
    presets = consolidated.screen_presets()
    frames = []
    for job in every:
        f = job_file(job)
        rows = pd.read_parquet(f)
        if len(rows) != CHUNK or set(rows["screen"]) != {_record(presets[job.preset])}:
            raise SystemExit(f"{f}: not {CHUNK} runs under {job.preset}'s screen as it is now; "
                             f"delete it and run again")
        numbers = list(range(job.first, job.first + CHUNK))
        if not (set(rows["preset"]) == {job.preset} and set(rows["plan"]) == {job.plan}
                and set(rows["veto"]) == {job.veto} and rows["order"].tolist() == numbers
                and [json.loads(s) for s in rows["seed"]]
                == [[ACCEPTANCE_SEED, o] for o in numbers]):
            raise SystemExit(f"{f}: its rows are not those of {job.preset} on {job.plan} "
                             f"{setting(job.veto)}, orders {numbers[0]}-{numbers[-1]} in turn, "
                             f"each order o's seed [{ACCEPTANCE_SEED}, o]; delete it and run again")
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


# ---- the acceptance guard -------------------------------------------------------------------

ACCEPT_PRESET = "realized-loss"   # acceptance: the preset whose runs it replays (the default)
ACCEPT_ORDERS = 10                # acceptance: each design and setting on orders 0 .. 9


class Outcome(NamedTuple):
    """What the acceptance guard checks of a run: the blocks the stop used, the call, the cost,
    whether the call is wrong and whether it asks (0 or 1 each), and the truth it was scored
    against."""
    units: int
    call: int
    cost: float
    wrong: int
    asks: int
    truth: int


def differences(new: Outcome, stored: Outcome) -> list[str]:
    """Each field in which `new` is not `stored`: exactly, but the cost to 1e-12 relative (a
    sum's rounding)."""
    differ = [f"{q} {getattr(new, q)} against {getattr(stored, q)}" for q in Outcome._fields
              if q != "cost" and getattr(new, q) != getattr(stored, q)]
    if not math.isclose(new.cost, stored.cost, rel_tol=1e-12):
        differ.append(f"cost {new.cost!r} against {stored.cost!r}")
    return differ


class Stored(NamedTuple):
    """A design and setting's acceptance runs as ACCEPTANCE holds them, in its order: each run's
    order, the seed its draws are keyed by, and its Outcome."""
    cell: Cell
    orders: tuple[int, ...]
    seeds: tuple[tuple[int, ...], ...]
    outcomes: tuple[Outcome, ...]


def _seed(text: str) -> tuple[int, ...]:
    """A row's seed, a JSON list of one or more non-negative integers."""
    seed = json.loads(text)
    if not (isinstance(seed, list) and seed
            and all(isinstance(w, int) and not isinstance(w, bool) and w >= 0 for w in seed)):
        raise ValueError(f"a seed is a JSON list of non-negative integers, got {text!r}")
    return tuple(seed)


#: ACCEPTANCE's columns the guard reads, each read as this type (the CSV's one boundary).
STORED_TYPES = {"preset": str, "plan": str, "veto": bool, "order": int, "seed": str, "units": int,
                "call": int, "cost": float, "wrong": int, "asks": int, "truth": int}


def stored_acceptance(screen: Screen) -> list[Stored]:
    """ACCEPT_PRESET's runs in ACCEPTANCE on orders 0 .. ACCEPT_ORDERS - 1, one Stored a design
    and setting, in the order the file first meets them; refused unless they are each design and
    setting of `screen`'s `comparison`, each on those orders once, ascending (a design and
    setting's rows need not be contiguous in the file)."""
    frame = pd.read_csv(ACCEPTANCE, usecols=list(STORED_TYPES), dtype=STORED_TYPES)
    mine: pd.DataFrame = frame[((frame["preset"] == ACCEPT_PRESET)
                                & (frame["order"] < ACCEPT_ORDERS)).to_numpy()]
    by_cell: dict[Cell, list[Mapping[Hashable, Any]]] = {}
    for r in mine.to_dict("records"):
        by_cell.setdefault(Cell(plan=str(r["plan"]), veto=bool(r["veto"])), []).append(r)
    stored = [Stored(cell=cell, orders=tuple(int(r["order"]) for r in rs),
                     seeds=tuple(_seed(str(r["seed"])) for r in rs),
                     outcomes=tuple(Outcome(units=int(r["units"]), call=int(r["call"]),
                                            cost=float(r["cost"]), wrong=int(r["wrong"]),
                                            asks=int(r["asks"]), truth=int(r["truth"]))
                                    for r in rs))
              for cell, rs in by_cell.items()]
    want, got = comparison(screen), [s.cell for s in stored]
    if sorted(got) != sorted(want) or any(s.orders != tuple(range(ACCEPT_ORDERS)) for s in stored):
        raise SystemExit(f"{ACCEPTANCE.name}: {ACCEPT_PRESET}'s runs on orders 0-"
                         f"{ACCEPT_ORDERS - 1} are not those of {len(want)} designs and settings "
                         f"({', '.join(f'{c.plan} {setting(c.veto)}' for c in want)}), each on "
                         f"each order once, ascending")
    return stored


class Reproduced(NamedTuple):
    """A design and setting's acceptance runs replayed now, in the stored order, and the minutes
    they took."""
    outcomes: tuple[Outcome, ...]
    minutes: float


def _reproduced(s: Stored) -> Reproduced:
    """`s`'s runs replayed under ACCEPT_PRESET as the presets build it now (`replayed`), each
    order's draws from its stored seed. Built in the worker: a fault is a patch in the guard's
    process, which a forked worker inherits."""
    start = time.monotonic()
    f = designs()[s.cell.plan]
    rep = replayed(consolidated.screen_presets()[ACCEPT_PRESET], f, veto=s.cell.veto,
                   orders=[Order(units=tuple(int(x) for x in f.orders[o]), seed=seed)
                           for o, seed in zip(s.orders, s.seeds, strict=True)])
    truth = _only(rep.truth.calls)
    return Reproduced(outcomes=tuple(Outcome(units=r.units, call=_only(r.calls), cost=r.cost,
                                             wrong=r.wrong, asks=r.asks, truth=truth)
                                     for r in rep.runs),
                      minutes=(time.monotonic() - start) / 60)


def acceptance(*, workers: int) -> list[str]:
    """ACCEPT_PRESET as the presets build it now, replaying ACCEPTANCE's runs of it on orders
    0 .. ACCEPT_ORDERS - 1 (`stored_acceptance`), a design and setting at a time in the file's
    order, on a forked pool of `workers` processes (exempt: a count that changes no result, each
    run drawing from its row's seed); its failure, the first design and setting whose runs are not
    the stored ones (`differences`), at which it returns."""
    stored = stored_acceptance(consolidated.screen_presets()[ACCEPT_PRESET])
    total = sum(len(s.orders) for s in stored)
    print(f"acceptance: {ACCEPT_PRESET} at epistemetry {pinned_rev()[:7]} against "
          f"{ACCEPTANCE.name}'s {total} runs of it on orders 0-{ACCEPT_ORDERS - 1} "
          f"({len(stored)} designs and settings), on {workers} "
          f"process{'' if workers == 1 else 'es'}", flush=True)
    equal = 0
    # Forked, so the workers run under any fault patched in this process.
    with get_context("fork").Pool(workers) as pool:
        for s, r in zip(stored, pool.imap(_reproduced, stored), strict=True):
            where = f"{s.cell.plan} {setting(s.cell.veto)}"
            differ = {o: d for o, new, old in zip(s.orders, r.outcomes, s.outcomes, strict=True)
                      if (d := differences(new, old))}
            equal += len(s.orders) - len(differ)
            first = next((f"first order {o}: {', '.join(d)}" for o, d in differ.items()), "")
            print(f"{where}: {len(s.orders) - len(differ)} of {len(s.orders)} runs equal "
                  f"({r.minutes:.1f} min)" + (" -> ok" if not differ else f" -> FAILS, {first}"),
                  flush=True)
            if differ:
                # the first design that differs ends the guard (leaving the pool terminates it)
                return [f"{where} ({len(differ)} of {len(s.orders)} runs differ; {first})"]
    print(f"acceptance: {equal} of {total} runs equal", flush=True)
    return []


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


def quantities(runs: pd.DataFrame, *, fixtures: Mapping[str, Fixture]) -> pd.DataFrame:
    """Each run's Quantities, one column each, in their order."""
    whole = np.array([float(fixtures[p].seconds.sum()) for p in runs["plan"]])
    return pd.DataFrame({"units": runs["units"].to_numpy(dtype=float),
                         "cost_fraction": runs["cost"].to_numpy(dtype=float) / whole,
                         "wrong": runs["wrong"].to_numpy(dtype=float),
                         "asks": runs["asks"].to_numpy(dtype=float)})


def deltas_of(old: pd.DataFrame, *, fixtures: Mapping[str, Fixture]) -> Quantities:
    """The equivalence margins by the rule pre-registered 2026-10-18: in each quantity, half the
    gap between the two presets' means on the old runs (`old`, exactly two presets' runs: refused
    otherwise), so a change smaller than what tells the presets apart."""
    presets = list(pd.unique(old["preset"]))
    if len(presets) != 2:
        raise ValueError(f"two presets give the margins, the old runs hold {presets}")
    q = quantities(old, fixtures=fixtures)
    one, other = (q[(old["preset"] == p).to_numpy()].mean() for p in presets)
    return Quantities(**{f: abs(float(one[f] - other[f])) / 2 for f in Quantities._fields})


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


class _Test(NamedTuple):
    """One sign-flip test of the Holm family: whose, of what, and its p."""
    preset: str
    what: str
    p: float


class _Exact(NamedTuple):
    """A preset's exact check: its (design, setting)s' truths equal, of those compared, and its
    runs of the floor's designs equal, of those compared."""
    truths_equal: int
    truths: int
    runs_equal: int
    runs: int


class _QuantityRow(NamedTuple):
    """A quantity's line of a preset's table: its old and new means, mean(d), the bootstrap
    interval of mean(d), the sign-flip p, and McNemar's exact p (nan for a quantity not 0-1)."""
    quantity: str
    old: float
    new: float
    mean_d: float
    low: float
    high: float
    p: float
    mcnemar: float


class _DesignRow(NamedTuple):
    """A design's line of a preset's table: its pairs, its old and new mean units, the sign-flip p
    of their difference, and its wrong calls and asks, old and new."""
    plan: str
    pairs: int
    units_old: float
    units_new: float
    p: float
    wrong_old: int
    wrong_new: int
    asks_old: int
    asks_new: int


def equivalent(old: pd.DataFrame, new: pd.DataFrame, *, fixtures: Mapping[str, Fixture],
               resamples: int, flips: int) -> list[str]:
    """Two sets of acceptance runs compared run against run on the same order (same preset,
    design, setting and order), by the test pre-registered 2026-10-18 (the epistemetry move's
    plan, Task 4), reading only READ; its failures, none when they are one policy's runs under
    two streams. Per preset and quantity (`Quantities`), d = new - old over the pairs:
    1. exact, where nothing is drawn: every (preset, design, setting)'s truth equal, and every run
       of a pairwise design no longer than the screens' floor (C1m4/p64) equal in units, call,
       cost (to 1e-12 relative, a sum's rounding), wrong and asks, failing a preset with no such
       run to compare;
    2. equivalence: the 95% percentile bootstrap interval of mean(d) (`resamples` resamples of a
       preset's pairs, np.random.default_rng(0)) inside +/- the margin, half the two presets'
       gap in the quantity on the old runs (`deltas_of`);
    3. no detectable shift: the sign-flip p of mean(d) (`flips` flips, np.random.default_rng(1)
       afresh for each test) for each preset's four quantities, and for the units of each
       (preset, design), Holm-corrected together at family-wise ALPHA, rejecting nothing.
    Prints the margins, then per preset each quantity's means, mean(d), interval, margin and p
    (McNemar's exact p beside a 0-1 quantity's), the per-design table and the Holm verdicts."""
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
    try:
        deltas = deltas_of(a, fixtures=fixtures)
    except ValueError as err:
        return [f"deltas: {err}"]
    print("margins, half the two presets' gap on the old runs: "
          + ", ".join(f"{q} {getattr(deltas, q):.4g}" for q in Quantities._fields), flush=True)

    bad: list[str] = []
    floor = {name for name, f in fixtures.items()
             if not f.many and len(f.blocks) <= consolidated.SCREEN_FLOOR}
    exact: dict[str, _Exact] = {}
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
        if not len(fixed):
            bad.append(f"exact: {p} compares no run of a pairwise design no longer than the "
                       f"floor ({consolidated.SCREEN_FLOOR} blocks): the check of the draw-free "
                       f"runs would check nothing")
        exact[p] = _Exact(truths_equal=equal_truths, truths=truths, runs_equal=equal_runs,
                          runs=len(fixed))

    tests: list[_Test] = []
    rows: dict[str, list[_QuantityRow]] = {}
    for p in presets:
        at = (m["preset"] == p).to_numpy()
        n = int(at.sum())
        idx = np.random.default_rng(0).integers(0, n, size=(resamples, n))
        rows[p] = []
        for q in Quantities._fields:
            o, w = q_old[q].to_numpy()[at], q_new[q].to_numpy()[at]
            d = w - o
            lo, hi = (float(x) for x in np.percentile(d[idx].mean(axis=1), [2.5, 97.5]))
            delta = float(getattr(deltas, q))
            if not (-delta < lo and hi < delta):
                bad.append(f"interval: {p} {q} [{lo:+.4f}, {hi:+.4f}] not inside "
                           f"+/-{delta:.4g}")
            pv = sign_flip(d, flips=flips, rng=np.random.default_rng(1))
            binary = bool(np.isin(o, (0.0, 1.0)).all() and np.isin(w, (0.0, 1.0)).all())
            mc = mcnemar(d) if binary else math.nan
            tests.append(_Test(preset=p, what=q, p=pv))
            rows[p].append(_QuantityRow(quantity=q, old=float(o.mean()), new=float(w.mean()),
                                        mean_d=float(d.mean()), low=lo, high=hi, p=pv,
                                        mcnemar=mc))
    per_design: dict[str, list[_DesignRow]] = {}
    for p in presets:
        per_design[p] = []
        for plan in pd.unique(m.loc[m["preset"] == p, "plan"]):
            at = ((m["preset"] == p) & (m["plan"] == plan)).to_numpy()
            d = q_new["units"].to_numpy()[at] - q_old["units"].to_numpy()[at]
            pv = sign_flip(d, flips=flips, rng=np.random.default_rng(1))
            tests.append(_Test(preset=p, what=f"units on {plan}", p=pv))
            cell = m[at]
            per_design[p].append(_DesignRow(
                plan=str(plan), pairs=int(at.sum()),
                units_old=float(q_old["units"].to_numpy()[at].mean()),
                units_new=float(q_new["units"].to_numpy()[at].mean()), p=pv,
                wrong_old=int(cell["wrong_old"].sum()), wrong_new=int(cell["wrong_new"].sum()),
                asks_old=int(cell["asks_old"].sum()), asks_new=int(cell["asks_new"].sum())))
    adjusted = dict(zip(tests, holm([t.p for t in tests]), strict=True))
    for t, ap in adjusted.items():
        if ap <= ALPHA:
            bad.append(f"holm: {t.preset} {t.what} p {t.p:.2g}, Holm {ap:.2g} <= {ALPHA}")

    def verdict(t: _Test) -> str:
        return f"{adjusted[t]:.3g} {'REJECTED' if adjusted[t] <= ALPHA else 'kept'}"

    by = {(t.preset, t.what): t for t in tests}
    for p in presets:
        e = exact[p]
        print(f"{p}: {int((m['preset'] == p).sum())} pairs; exact: {e.truths_equal} of "
              f"{e.truths} truths equal, {e.runs_equal} of {e.runs} runs of "
              f"{', '.join(sorted(floor))} equal", flush=True)
        print(f"  {'quantity':<14}{'old':>9}{'new':>9}{'mean(d)':>10}  {'95% interval':<21}"
              f"{'margin':>7}{'p (flips)':>11}{'McNemar':>9}  Holm", flush=True)
        for r in rows[p]:
            print(f"  {r.quantity:<14}{r.old:>9.4f}{r.new:>9.4f}{r.mean_d:>+10.4f}  "
                  f"[{r.low:+.4f}, {r.high:+.4f}]  {float(getattr(deltas, r.quantity)):>7.4g}"
                  f"{r.p:>11.2g}{'' if math.isnan(r.mcnemar) else f'{r.mcnemar:.2g}':>9}  "
                  f"{verdict(by[p, r.quantity])}", flush=True)
        print(f"  {'design':<15}{'pairs':>6}{'units old':>10}{'new':>8}{'mean(d)':>9}"
              f"{'p (flips)':>11}  {'Holm':<15}{'wrong old/new':>14}{'asks old/new':>13}",
              flush=True)
        for dr in per_design[p]:
            print(f"  {dr.plan:<15}{dr.pairs:>6}{dr.units_old:>10.2f}{dr.units_new:>8.2f}"
                  f"{dr.units_new - dr.units_old:>+9.2f}{dr.p:>11.2g}  "
                  f"{verdict(by[p, f'units on {dr.plan}']):<15}"
                  f"{f'{dr.wrong_old}/{dr.wrong_new}':>14}{f'{dr.asks_old}/{dr.asks_new}':>13}",
                  flush=True)
    rejected = [t for t, ap in adjusted.items() if ap <= ALPHA]
    print(f"Holm over {len(tests)} tests at family-wise {ALPHA}: "
          + ("no rejection" if not rejected else "rejects " + ", ".join(
              f"{t.preset} {t.what}" for t in rejected)), flush=True)
    return bad


# ---- the command line -----------------------------------------------------------------------

class Pooled(Protocol):
    """A guard that runs on a pool of `workers` processes."""
    def __call__(self, *, workers: int) -> list[str]: ...


#: The guards by name that run in this process alone.
SERIAL: dict[str, Callable[[], list[str]]] = {"target": target, "records": records,
                                              "adapter": adapter, "why": why, "screens": screens}
#: The guards by name that run on a pool, its size `--workers W`.
POOLED: dict[str, Pooled] = {"acceptance": acceptance}


def guards(*, workers: int) -> dict[str, Callable[[], list[str]]]:
    """Every guard by name (FAULTS' names), POOLED's on a pool of `workers` processes."""
    return {**SERIAL, **{name: partial(g, workers=workers) for name, g in POOLED.items()}}


def main(guard: str, fault: str | None, *, workers: int) -> None:
    run = guards(workers=workers)[guard]
    if fault is None:
        bad = run()
    else:
        with FAULTS[guard][fault]():
            bad = run()
    verdict = "FAILS on " + ", ".join(bad) if bad else "passes"
    print(f"{guard}{'' if fault is None else f' under the fault {fault}'}: {verdict}", flush=True)
    if bad and fault is None:
        raise SystemExit(1)
    if not bad and fault is not None:
        raise SystemExit(f"the fault {fault} went unseen: the guard guards nothing")


def both_ways(names: list[str], *, workers: int) -> None:
    """Each guard as it is, then under each of its faults (FAULTS), one after another in this
    process. Raises naming every run not as `main` expects."""
    wrong = []
    for guard in names:
        for fault in (None, *FAULTS[guard]):
            try:
                main(guard, fault, workers=workers)
            except SystemExit as e:
                wrong.append(f"{guard}{'' if fault is None else f' under {fault}'}: {e.code}")
    print(f"both ways, {', '.join(names)}: "
          + ("as expected" if not wrong else "NOT as expected: " + "; ".join(wrong)), flush=True)
    if wrong:
        raise SystemExit(1)


if __name__ == "__main__":
    a = sys.argv[1:]
    count: int | None = None
    if a[-2:-1] == ["--workers"]:
        if not (a[-1].isdigit() and int(a[-1]) >= 1):
            raise SystemExit(__doc__)
        count, a = int(a[-1]), a[:-2]
    named = a[1].split(",") if len(a) == 2 and a[0] == "both-ways" else a[:1]
    if count is not None and not (a == ["runs"] or set(named) & set(POOLED)):
        raise SystemExit(f"--workers is the processes of `runs` and of the pooled guards "
                         f"({', '.join(POOLED)}); `{' '.join(a)}` runs on no pool")
    workers = 1 if count is None else count   # exempt: a count that changes no result
    if len(a) == 2 and a[0] == "both-ways" and set(named) <= set(FAULTS):
        both_ways(named, workers=workers)
        raise SystemExit(0)
    if a == ["runs"]:
        runs(workers=workers)
        raise SystemExit(0)
    if len(a) == 3 and a[0] == "equivalent":
        failures = equivalent(pd.read_csv(a[1]), pd.read_csv(a[2]), fixtures=designs(),
                              resamples=RESAMPLES, flips=FLIPS)
        print("pass" if not failures else "FAILS on " + "; ".join(failures), flush=True)
        raise SystemExit(1 if failures else 0)
    if len(a) not in (1, 2) or a[0] not in FAULTS or (len(a) == 2 and a[1] not in FAULTS[a[0]]):
        raise SystemExit(__doc__)
    main(a[0], a[1] if len(a) == 2 else None, workers=workers)
