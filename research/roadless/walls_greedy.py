"""A greedy over W's exact gains: an optimizer fit for walls that leak (the controller 2026-10-21,
the owner's yes: SIMP's default is weak under W, its last iterate grey; NOTES, "Walls that leak",
stage 3). Clear buildings one at a time, each time the one with the largest exact gain in J_power
per unit of its cost (the area population's share: the destroyed area's D) among those that still
fit the budget, each gain scored by W's discrete scorer (gap_walls.WallModel: the clearing's state,
its solve, J of the homes' escape times), until none fits.

Two Strategies, each in the run's identity (its rows' directory beside a SIMP plan's,
relax_rows/<metric>/<name>/D<budget>/):
- `Full` (`Gx`): every candidate re-scored at every step.
- `Lazy` (`GxL`, CELF): each candidate keeps its gain from the clearing it was last scored at; the
  top one by gain per unit cost is re-scored at the current clearing and taken once it is on top
  with a current gain. Where gains only fall as the clearing grows it takes Full's choices; where
  they can rise (two neighbours opening a way together) it may not, so it is checked against Full
  on the pilot blocks (NOTES).
Every score solves from the current clearing's sides (a diode's loop warm from the last state taken;
the symmetric walls one factorization a score). Ties in gain per unit cost go to the lowest index.

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/walls_greedy.py
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import gap_walls as gw
import numpy as np
from numpy.typing import NDArray


class Step(NamedTuple):
    """A building taken: its gain in J (the scorer's, at the clearing before it), its gain per
    unit cost, J after it, D after it, and the scores spent so far."""
    building: int
    gain: float
    ratio: float
    J: float
    D: float
    scores: int


class Result(NamedTuple):
    """A greedy's run: J at the start, its steps in order, the scores it spent."""
    J0: float
    steps: list[Step]
    scores: int


class Scorer:
    """W's exact J of a clearing (WallModel's state and solve), each solve from the current
    clearing's sides; the current clearing, its J and its sides."""

    def __init__(self, wm: gw.WallModel, *, power: float):
        self.wm, self.power = wm, power
        self.removed = np.zeros(wm.n, dtype=bool)
        self.sides = wm.sides0.copy()
        self.J, _sides = self.score(self.removed)
        self.scores = 0

    def score(self, removed: NDArray[np.bool_]) -> tuple[float, NDArray[np.int8]]:
        """(J_power of the clearing `removed`, its solve's sides)."""
        wm = self.wm
        st = wm.state(removed)
        s = wm.walls.solver(wm, st, self.sides).solve(wm.rhs(st, wm.inj))
        return wm.J(wm.home_u(s.u), self.power), wm.directions(s.u)

    def gain(self, j: int) -> tuple[float, NDArray[np.int8]]:
        """The current clearing's J less J with building j cleared too, and that state's sides."""
        r = self.removed.copy()
        r[j] = True
        J, sides = self.score(r)
        self.scores += 1
        return self.J - J, sides

    def take(self, j: int, gain: float, sides: NDArray[np.int8] | None) -> None:
        """Clear j, its gain `gain` taken; its state's sides `sides`, or None: solved again for
        them (a gain kept from the first pass, whose sides are not stored: a building's sides
        cost a static passage's byte each)."""
        self.removed[j] = True
        self.J -= gain
        if sides is None:
            _J, sides = self.score(self.removed)
            self.scores += 1
        self.sides = sides

    @property
    def D(self) -> float:
        return float(self.wm.cost[self.removed].sum())


class Greedy(Protocol):
    """A greedy's rule (module docstring): its name, its run within `budget` (a share of the
    cost)."""

    @property
    def name(self) -> str: ...

    def run(self, sc: Scorer, *, budget: float) -> Result: ...


def _fits(sc: Scorer, j: int, left: float) -> bool:
    return not sc.removed[j] and float(sc.wm.cost[j]) <= left


@dataclass(frozen=True)
class Full:
    """Every candidate re-scored at every step (`Gx`)."""

    @property
    def name(self) -> str:
        return "Gx"

    def run(self, sc: Scorer, *, budget: float) -> Result:
        cost = sc.wm.cost
        J0, left, steps = sc.J, budget + 1e-12, []
        while True:
            best: tuple[float, int, float, NDArray[np.int8]] | None = None
            for j in range(sc.wm.n):
                if not _fits(sc, j, left):
                    continue
                g, sides = sc.gain(j)
                if best is None or g / cost[j] > best[0]:
                    best = (g / float(cost[j]), j, g, sides)
            if best is None:
                break
            ratio, j, g, sides = best
            sc.take(j, g, sides)
            left -= float(cost[j])
            steps.append(Step(building=j, gain=g, ratio=ratio, J=sc.J, D=sc.D, scores=sc.scores))
        return Result(J0=J0, steps=steps, scores=sc.scores)


@dataclass(frozen=True)
class Lazy:
    """CELF (`GxL`): stale gains as upper bounds, the top re-scored at the current clearing and
    taken once it is on top with a current gain (module docstring)."""

    @property
    def name(self) -> str:
        return "GxL"

    def stale(self, scored_at: int, step: int) -> bool:
        """Whether a gain scored at step `scored_at` must be re-scored at step `step`."""
        return scored_at != step

    def run(self, sc: Scorer, *, budget: float) -> Result:
        cost = sc.wm.cost
        J0, left, steps = sc.J, budget + 1e-12, []
        # (-gain per unit cost, building, the step it was scored at); per building scored at the
        # current step its gain and its state's sides (none kept from the first pass)
        heap: list[tuple[float, int, int]] = []
        fresh: dict[int, tuple[float, NDArray[np.int8] | None]] = {}
        for j in range(sc.wm.n):
            if _fits(sc, j, left):
                g, _sides = sc.gain(j)
                heap.append((-g / float(cost[j]), j, 0))
                fresh[j] = (g, None)
        heapq.heapify(heap)
        step = 0
        while heap:
            key, j, at = heapq.heappop(heap)
            if not _fits(sc, j, left):
                continue
            if self.stale(at, step):
                g, sides = sc.gain(j)
                fresh[j] = (g, sides)
                heapq.heappush(heap, (-g / float(cost[j]), j, step))
                continue
            g, kept = fresh[j]
            sc.take(j, g, kept)
            left -= float(cost[j])
            step += 1
            fresh = {}
            steps.append(Step(building=j, gain=g, ratio=-key, J=sc.J, D=sc.D, scores=sc.scores))
        return Result(J0=J0, steps=steps, scores=sc.scores)


def greedy_of(name: str) -> Full | Lazy:
    """The greedy a name spells (`Gx`, `GxL`), resolved where a run is configured."""
    for g in (Full(), Lazy()):
        if g.name == name:
            return g
    raise ValueError(f"unknown greedy {name!r}: Gx or GxL")


def run(wm: gw.WallModel, rule: Full | Lazy, *, budget: float, power: float) -> Result:
    """The greedy `rule` on W's model `wm` within `budget`, J_power its objective."""
    return rule.run(Scorer(wm, power=power), budget=budget)
