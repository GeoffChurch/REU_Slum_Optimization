"""The screens' rule on GeoffChurch/epistemetry: a Strategy injected into epistemetry's `Screen`
(NOTES, the section on the move onto epistemetry 165f98b).

A screen asks of a variant against SIMP's default, one held-out block at a time in
common.HELD's order: adopt it, keep the default, or ask the owner. It is now epistemetry's
`Screen` (a policy, a drop rule and a seed: epistemetry.design.screen), carried on by `advance`
as each block's rows arrive, its decisions a ledger that `advance` continues from. Its policy is
`ValueOfMoreUnits`: past a floor, it values the next blocks for every horizon on the
finite-population Bayesian bootstrap of the variant's own blocks, and goes on while some horizon
is worth its GPU time. What it values, prices and calls each arm by, and the stop that reads it,
is the RULE, a Strategy (`Rule`): `RealizedLoss` (a wrong call judged by what it loses) or
`ZeroOne` (by whether it is wrong, each arm priced by its own stake). A rule is chosen once, where
the presets are built, at consolidated's edge (`consolidated.screen_presets`: every number that
can change a call is a constant there), and nothing below asks which rule it holds.
`Screen.record()` is a screen's identity: the rule's target and stop, the horizons, the floor,
the step, the cost model, the Monte Carlo sizes, the collapse veto and the seed are all in it,
and the record revives as the screen itself.

THE PIN. epistemetry is pinned in pyproject.toml (the `research` group, `[tool.uv.sources]` at a
full SHA). A pin moves so:
- edit the `rev`, then `uv lock --upgrade-package epistemetry`;
- run `screening_check.py both-ways target,records,adapter,acceptance,screens`;
- if `acceptance` fails only because the random numbers changed, regenerate the runs
  (`screening_check.py runs`) and pass `screening_check.py equivalent` against the committed CSV
  before committing the new one;
- re-derive the phase-5 screens if `screens` fails the same way, and commit.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from epistemetry.design.decide import (
    ADOPT,
    ASK,
    KEEP,
    AdoptKeepAsk,
    AdoptKeepAskByStake,
    Priced,
    StandardErrors,
    StopRule,
    ValueOfInformation,
)
from epistemetry.design.screen import SameTarget, Seen, TargetOf
from epistemetry.stats.records import Recordable

#: A call as the summaries spell it (epistemetry's calls are int rows at 165f98b, the truth's too).
CALLS: Mapping[int, str] = {KEEP: "keep", ASK: "ask", ADOPT: "adopt"}


class Rule(Protocol):
    """What a screen values, prices and calls each arm by (`target_of`), and the stop that reads
    it (`stop`). A rule is a builder: what it builds is recorded inside `Screen.record()`, so a
    rule records nothing itself."""

    def target_of(self) -> TargetOf: ...

    def stop(self) -> StopRule: ...


@dataclass(frozen=True, kw_only=True)
class RealizedLoss:
    """The realized loss as the target (the fourth replay's): AdoptKeepAskByStake over `rates`
    (Lens A per GPU-second, increasing) weighted by `weights`, a wrong call at rate r losing
    |gain - r x extra GPU-seconds| on each of `uses` future block runs. Its price, the weighted
    mean rate, is the target's own, carried in the evidence: no rule holds one. The stop is the
    value of more blocks against that price, cleared by `margin` Monte Carlo standard errors."""
    rates: tuple[float, ...]
    weights: tuple[float, ...]
    uses: float
    margin: float

    def target_of(self) -> TargetOf:
        return SameTarget(target=AdoptKeepAskByStake(rates=self.rates, weights=self.weights,
                                                     uses=self.uses))

    def stop(self) -> StopRule:
        return ValueOfInformation(margin=StandardErrors(k=self.margin))


@dataclass(frozen=True, kw_only=True)
class OwnStake(Recordable, namespace="reblock"):
    """Each arm's target at a decision: AdoptKeepAsk over [low, high] (a 0-1 risk: the probability
    of another call), priced by the arm's own stake, the Lens A a wrong call on it costs at its
    means so far, |mean gain - price_rate x mean extra GPU-seconds| over its own prefix: a
    GPU-second is worth `price_rate` / `uses` of Lens A, so price_rate / uses / stake of the risk.
    An arm exactly at break-even at the price rate has no stake to price: the division by its
    zero stake raises."""
    low: float
    high: float
    price_rate: float
    uses: float

    def of(self, seen: Seen, *, arm: int) -> Priced:
        o = seen.arms[arm].outcomes
        stake = float(abs(o[:, 0].mean() - self.price_rate * o[:, 1].mean()))
        return Priced(target=AdoptKeepAsk(low=self.low, high=self.high),
                      price=self.price_rate / self.uses / stake)


@dataclass(frozen=True, kw_only=True)
class ZeroOne:
    """The third replay's target: AdoptKeepAsk over the bracket's ends (`rates`' first and last),
    each arm priced by its own stake (`OwnStake`) at the weighted mean rate over `uses`. The stop
    is `RealizedLoss`'s."""
    rates: tuple[float, ...]
    weights: tuple[float, ...]
    uses: float
    margin: float

    def target_of(self) -> TargetOf:
        return OwnStake(low=self.rates[0], high=self.rates[-1],
                        price_rate=float(np.asarray(self.rates) @ np.asarray(self.weights)),
                        uses=self.uses)

    def stop(self) -> StopRule:
        return ValueOfInformation(margin=StandardErrors(k=self.margin))
