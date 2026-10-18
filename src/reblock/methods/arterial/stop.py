"""When the arterial greedy stops: at a required displacement budget, measured as the footprint
area destroyed, or at its `max_roads` safety cap, or when no candidate gains -- and which of the
three it was, so a cap that binds is seen.

`DisplacementMeter.share` is the footprint area of every building the committed corridor touches
at all, a touched building counted WHOLE (it is destroyed), over the block's total footprint
area: `sum_{c_i > 0} area_i / sum_i area_i`, with `c_i` the overlap `displacement` measures. It is
never below the fractional share `sum_i c_i area_i / sum_i area_i` that the research lineup scores
as area D, so a run stopped at a destroyed-area budget can sit below the same scored budget.

The engines check the budget BEFORE each step and keep the road that crosses it, so a stopped run
is a commit prefix of a longer one and overshoots by at most one road.
"""
from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
import shapely
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from shapely.geometry import LineString

from reblock.buildings import Extents
from reblock.contracts import Block

Stop = Literal["max_displacement", "no_gain", "max_roads"]


class GreedyRun(NamedTuple):
    """An engine's result: the planarized roads, the greedy commits behind them, and why it
    stopped."""
    roads: GeoDataFrame
    commits: int
    stop: Stop


class DisplacementMeter:
    """The destroyed footprint share of the roads committed so far (the module's docstring), kept
    incrementally: each committed road's corridor queries only the buildings it reaches, and a
    building joins the touched set when its overlap with that corridor has positive area. A
    building with no area destroys no area, so it never moves the share; a block with no buildings
    (or none with area) has share 0.0 forever."""

    def __init__(self, buildings: Extents, half_width_m: float) -> None:
        self._buildings = buildings
        self._half_width_m = half_width_m
        self._outlines = np.asarray(buildings.outlines.to_numpy(), dtype=object)
        self._area: NDArray[np.float64] = np.asarray(shapely.area(self._outlines),
                                                     dtype=np.float64)
        self._total = float(self._area.sum())
        self._touched = np.zeros(len(self._area), dtype=bool)
        self._destroyed = 0.0

    @classmethod
    def of(cls, block: Block, half_width_m: float) -> DisplacementMeter:
        return cls(block.buildings, half_width_m)

    def add(self, real: LineString) -> None:
        """Commit `real`'s corridor (the road buffered by the half-width, as the cost prices it)."""
        piece = real.buffer(self._half_width_m)
        hit = self._buildings.touching(piece)
        new = hit[~self._touched[hit] & (self._area[hit] > 0.0)]
        if not len(new):
            return
        overlap = np.asarray(shapely.area(shapely.intersection(self._outlines[new], piece)),
                             dtype=np.float64)
        now = new[overlap > 0.0]
        self._touched[now] = True
        self._destroyed += float(self._area[now].sum())

    def share(self) -> float:
        return self._destroyed / self._total if self._total > 0.0 else 0.0
