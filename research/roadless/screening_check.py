"""Guards of the screens on GeoffChurch/epistemetry (screening.py's rule, consolidated's presets),
each with its faults (a patch in one process, no file edited, as gap_clear_check.py's): a guard
passes on the code as it is and must fail under each of its faults.

    S=research/roadless/screening_check.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S target [fault]            # faults: dropped-uses, unstaked
    uv run python $S records [fault]           # fault: array-rates
    uv run python $S both-ways <guards,>       # each guard, then under each of its faults

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
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
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
from epistemetry.design.replay import Design
from epistemetry.design.screen import Bootstrap, SameTarget, Seen, TargetOf, ValueOfMoreUnits
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
                      covariates={"base_seconds": self.base_seconds})


def _fixture(d: Mapping[str, Any]) -> Fixture:
    """One design of DESIGNS, checked: the JSON boundary, every key read by name."""
    name, many = d["name"], d["many"]
    if not isinstance(name, str) or not isinstance(many, bool):
        raise ValueError(f"a design's name is a string and `many` a bool: {name!r}, {many!r}")
    gain = np.ascontiguousarray(np.array(d["gain"], dtype=float).T)
    seconds = np.ascontiguousarray(np.array(d["seconds"], dtype=float).T)
    base_seconds = np.array(d["base_seconds"], dtype=float)
    blocks = tuple(str(b) for b in d["blocks"])
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
    bad: list[str] = []
    pairwise = [f for f in designs().values() if not f.many]
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
        wrong = [f"{check} ({total - ok} of {total})" for check, (ok, total) in counts.items()
                 if ok != total] + ([] if stop else ["its stop"])
        print(f"{name}: {at} prefixes of {len(pairwise)} designs, {read} decisions read; "
              + ", ".join(f"{check} {ok} of {total}" for check, (ok, total) in counts.items())
              + f"; stop {'ok' if stop else policy.stop}"
              + (f" -> FAILS (first: {first})" if wrong else " -> ok"), flush=True)
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


GUARDS: dict[str, Callable[[], list[str]]] = {"target": target, "records": records}


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


FAULTS: dict[str, dict[str, Callable[[], AbstractContextManager[None]]]] = {
    "target": {"dropped-uses": lambda: _patched(screening.RealizedLoss, "target_of",
                                                _over_one_use),
               "unstaked": lambda: _patched(screening.OwnStake, "of", _unstaked)},
    "records": {"array-rates": _array_rates},
}


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
    if len(a) not in (1, 2) or a[0] not in GUARDS or (len(a) == 2 and a[1] not in FAULTS[a[0]]):
        raise SystemExit(__doc__)
    main(a[0], a[1] if len(a) == 2 else None)
