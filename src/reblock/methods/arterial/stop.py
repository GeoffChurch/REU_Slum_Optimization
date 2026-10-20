"""When the arterial greedy stops: at an injected displacement budget (`DisplacementBudget`), at
its `max_roads` safety cap, or when no candidate gains -- and which of the three it was, so a cap
that binds is seen.

The budget Strategy is `NoBudget` (never reached: the directness presets, whose budget is
`max_roads`) or `Destroyed(share=..., tau=...)` (the access presets): the ramped destroyed-area
displacement of the committed roads (reblock.destroyed: each building's destroyed portion X_i,
ramped to min(1, X_i / tau), weighted by footprint area as research/roadless common.py's
AreaPopulation weights it) at or past `share`. That is the research lineup's own area D, so the
stop and the scored prefixes agree. `Destroyed`'s meter is incremental (IncrementalDestroyed: only
the buildings a committed road touches are recomputed). Every run records its budget's share at
the stop (None under NoBudget, which measures nothing).

The engines check the budget BEFORE each step and keep the road that crosses it, so a stopped run
is a commit prefix of a longer one and overshoots by at most one road.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol, TypeAlias, runtime_checkable

import numpy as np
import shapely
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from shapely.geometry import LineString

from reblock.contracts import Block
from reblock.destroyed import IncrementalDestroyed, RampedDestroyed

Stop = Literal["budget", "no_gain", "max_roads"]


class GreedyRun(NamedTuple):
    """An engine's result: the planarized roads, the greedy commits behind them, why it stopped,
    and the budget's share at the stop (None under NoBudget)."""
    roads: GeoDataFrame
    commits: int
    stop: Stop
    displaced: float | None


@runtime_checkable
class BudgetMeter(Protocol):
    """One run's measure of its committed roads against its budget."""

    def add(self, real: LineString) -> None: ...

    def reached(self) -> bool: ...

    def share(self) -> float | None: ...


@runtime_checkable
class DisplacementBudget(Protocol):
    """The run's displacement budget: a meter for one block's run."""

    @property
    def identity(self) -> BudgetIdentity: ...

    def start(self, block: Block, half_width_m: float) -> BudgetMeter: ...


class _Unmeasured:
    """NoBudget's meter: measures nothing, never reached."""

    def add(self, real: LineString) -> None:
        pass

    def reached(self) -> bool:
        return False

    def share(self) -> float | None:
        return None


@dataclass(frozen=True)
class NoBudget:
    """Never reached: the run ends on `max_roads` or when nothing gains."""

    @property
    def identity(self) -> BudgetIdentity:
        return self

    def start(self, block: Block, half_width_m: float) -> BudgetMeter:
        return _Unmeasured()


class DestroyedMeter:
    """The committed corridor's ramped destroyed-area D (the module's docstring), each committed
    road buffered by the half-width, as the cost prices it. A block with no buildings has D 0.0."""

    def __init__(self, budget: Destroyed, block: Block, half_width_m: float) -> None:
        self._budget = budget
        self._half_width_m = half_width_m
        self._destroyed = IncrementalDestroyed(budget.measure, block.buildings)
        area = np.asarray(shapely.area(np.asarray(block.buildings.outlines.to_numpy(),
                                                  dtype=object)), dtype=np.float64)
        # AreaPopulation's weights exactly (research/roadless common.py): footprint area over its
        # mean, every building 1 when no footprint has area.
        mean = float(area.mean()) if len(area) else 0.0
        self._w: NDArray[np.float64] = area / mean if mean > 0.0 else np.ones(len(area))
        self._w_total = float(self._w.sum())

    def add(self, real: LineString) -> None:
        self._destroyed.add(real.buffer(self._half_width_m))

    def share(self) -> float:
        if self._w_total == 0.0:
            return 0.0
        return float((self._destroyed.c() * self._w).sum() / self._w_total)

    def reached(self) -> bool:
        return self.share() >= self._budget.share


def _check_share(name: str, value: float) -> None:
    if not (math.isfinite(value) and 0.0 < value <= 1.0):
        raise ValueError(f"{name} is a share in (0, 1], got {value}")


@dataclass(frozen=True, kw_only=True)
class Destroyed:
    """Reached once the committed roads' ramped destroyed-area D (`tau`) is at or past `share`."""
    share: float
    tau: float

    def __post_init__(self) -> None:
        _check_share("share", self.share)
        _check_share("tau", self.tau)

    @property
    def identity(self) -> BudgetIdentity:
        return self

    @property
    def measure(self) -> RampedDestroyed:
        return RampedDestroyed(tau=self.tau)

    def start(self, block: Block, half_width_m: float) -> DestroyedMeter:
        return DestroyedMeter(self, block, half_width_m)


BudgetIdentity: TypeAlias = NoBudget | Destroyed
