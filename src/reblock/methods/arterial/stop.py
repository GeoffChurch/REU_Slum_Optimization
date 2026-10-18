"""When the arterial greedy stops: at an injected displacement budget (`DisplacementBudget`), at
its `max_roads` safety cap, or when no candidate gains -- and which of the three it was, so a cap
that binds is seen.

`FootprintMeter` keeps the committed corridor's overlap with every building incrementally
(`reblock.buildings.IncrementalOverlap`, one road at a time) and reads two shares off it:

- `scored_area`, the research lineup's area D: `sum_i c_i w_i / sum_i w_i`, with `c_i` the share
  of building i's outline the corridor covers and `w_i` its footprint area (research/roadless
  common.py's `displacement` under `AreaPopulation`, weights and all);
- `destroyed_area`: `sum_{c_i > 0} a_i / sum_i a_i`, a touched building counted WHOLE (it is
  destroyed). Never below `scored_area`, and measured at 3 -- 11x it on the lineup's blocks.

The budget Strategy reads those shares and says whether the run has reached it: `NoBudget` (never:
the directness presets, whose budget is `max_roads`), `ScoredArea(share=...)` (the access presets)
or `DestroyedArea(share=...)` (kept selectable for the owner's later decision on the project's
displacement definition). Every run records both shares at its stop, whichever budget it ran.

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

from reblock.buildings import Extents, IncrementalOverlap
from reblock.contracts import Block

Stop = Literal["budget", "no_gain", "max_roads"]


class Shares(NamedTuple):
    """The committed corridor's displacement, both ways (the module's docstring)."""
    scored_area: float
    destroyed_area: float


class GreedyRun(NamedTuple):
    """An engine's result: the planarized roads, the greedy commits behind them, why it stopped,
    and the displacement shares at the stop."""
    roads: GeoDataFrame
    commits: int
    stop: Stop
    shares: Shares


class FootprintMeter:
    """The committed roads' displacement shares (the module's docstring), kept incrementally: each
    committed road's corridor (the road buffered by the half-width, as the cost prices it) touches
    only the buildings it reaches. A block with no buildings has both shares 0.0 forever."""

    def __init__(self, buildings: Extents, half_width_m: float) -> None:
        self._half_width_m = half_width_m
        self._overlap = IncrementalOverlap(buildings)
        area = np.asarray(shapely.area(np.asarray(buildings.outlines.to_numpy(), dtype=object)),
                          dtype=np.float64)
        self._area: NDArray[np.float64] = area
        self._area_total = float(area.sum())
        # AreaPopulation's weights exactly (common.py): footprint area over its mean, every
        # building 1 when no footprint has area (a block of coincident points).
        mean = float(area.mean()) if len(area) else 0.0
        self._w: NDArray[np.float64] = area / mean if mean > 0.0 else np.ones(len(area))
        self._w_total = float(self._w.sum())

    @classmethod
    def of(cls, block: Block, half_width_m: float) -> FootprintMeter:
        return cls(block.buildings, half_width_m)

    def add(self, real: LineString) -> None:
        """Commit `real`'s corridor."""
        self._overlap.add(real.buffer(self._half_width_m))

    def shares(self) -> Shares:
        if self._w_total == 0.0:
            return Shares(scored_area=0.0, destroyed_area=0.0)
        c = self._overlap.c()
        scored = float((c * self._w).sum() / self._w_total)
        destroyed = (float((self._area * (c > 0.0)).sum() / self._area_total)
                     if self._area_total > 0.0 else 0.0)
        return Shares(scored_area=scored, destroyed_area=destroyed)


@runtime_checkable
class DisplacementBudget(Protocol):
    """Whether the committed roads' displacement has reached the run's budget."""

    @property
    def identity(self) -> BudgetIdentity: ...

    def reached(self, shares: Shares) -> bool: ...


def _check_share(share: float) -> None:
    if not (math.isfinite(share) and 0.0 < share <= 1.0):
        raise ValueError(f"a displacement budget is a share in (0, 1], got {share}")


@dataclass(frozen=True)
class NoBudget:
    """Never reached: the run ends on `max_roads` or when nothing gains."""

    @property
    def identity(self) -> BudgetIdentity:
        return self

    def reached(self, shares: Shares) -> bool:
        return False


@dataclass(frozen=True, kw_only=True)
class ScoredArea:
    """Reached once the lineup's scored area D is at or past `share`."""
    share: float

    def __post_init__(self) -> None:
        _check_share(self.share)

    @property
    def identity(self) -> BudgetIdentity:
        return self

    def reached(self, shares: Shares) -> bool:
        return shares.scored_area >= self.share


@dataclass(frozen=True, kw_only=True)
class DestroyedArea:
    """Reached once the destroyed footprint share (a touched building counted whole) is at or
    past `share`. On a block as small as 1808 (ten buildings) 0.20 binds after one road."""
    share: float

    def __post_init__(self) -> None:
        _check_share(self.share)

    @property
    def identity(self) -> BudgetIdentity:
        return self

    def reached(self, shares: Shares) -> bool:
        return shares.destroyed_area >= self.share


BudgetIdentity: TypeAlias = NoBudget | ScoredArea | DestroyedArea
