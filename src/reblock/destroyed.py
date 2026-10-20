"""Ramped destroyed-area displacement (owner 2026-10-18): one rule for every building.

The destroyed portion of building i under a corridor C is

    X_i = 1 - area(largest piece of O_i - C) / area(O_i),

so a corner clip destroys the clipped share, and a road through a building destroys the corridor
plus every piece but the largest (what is left standing is the largest piece). Building i then
counts as

    c_i = min(1, X_i / tau)

destroyed: 0 -> tau of it destroyed ramps 0 -> 1 (tau 0.10 in the research lineup), and the
weighted displacement is D = sum_i c_i w_i / sum_i w_i under a population's weights w. A building
with no area (coincident points) is destroyed iff the corridor covers its point, as the project's
overlap fraction has it (reblock.buildings).

The research lineup scores every road network by it (research/roadless common.displacement, the
prefixes' cuts) and the arterial stops on it (methods/arterial/stop.py's `Destroyed`); the public
package's `reblock.buildings` / `reblock.budget` keep the partial overlap until the site revamp.
The two evaluations here agree to floating point: `RampedDestroyed.c` over a whole corridor, and
`IncrementalDestroyed` one corridor piece at a time (each building keeps the part of its outline
the pieces so far cover, and only the buildings a new piece touches are recomputed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

from reblock.buildings import Extents


def _largest_piece_area(rest: NDArray[np.object_]) -> NDArray[np.float64]:
    """Per geometry of `rest`, the area of its largest part (0 for an empty one)."""
    big = np.zeros(len(rest), dtype=np.float64)
    if len(rest):
        parts, idx = shapely.get_parts(rest, return_index=True)
        if len(parts):
            np.maximum.at(big, idx, shapely.area(parts))
    return big


def _destroyed_portion(outlines: NDArray[np.object_], area: NDArray[np.float64],
                       cut: NDArray[np.object_]) -> NDArray[np.float64]:
    """X for buildings of positive area: 1 - largest piece of (outline - cut) over the outline's
    area, clipped to [0, 1] (a cut that only touches an outline leaves it whole, to FP)."""
    rest = shapely.difference(outlines, cut)
    return np.asarray(np.clip(1.0 - _largest_piece_area(rest) / area, 0.0, 1.0),
                      dtype=np.float64)


@dataclass(frozen=True, kw_only=True)
class RampedDestroyed:
    """The measure (the module's docstring). `tau` is required: it changes every result."""
    tau: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.tau) and 0.0 < self.tau <= 1.0):
            raise ValueError(f"tau is a destroyed share in (0, 1], got {self.tau}")

    @property
    def identity(self) -> RampedDestroyed:
        return self

    def ramp(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.minimum(1.0, x / self.tau)

    def destroyed(self, buildings: Extents, corridor: BaseGeometry | None) -> NDArray[np.float64]:
        """X_i for every building under the whole `corridor` (0 where it does not reach)."""
        outlines = np.asarray(buildings.outlines.to_numpy(), dtype=object)
        x = np.zeros(len(outlines), dtype=np.float64)
        if corridor is None or corridor.is_empty or not len(outlines):
            return x
        area = np.asarray(shapely.area(outlines), dtype=np.float64)
        hit = buildings.touching(corridor)
        pos = hit[area[hit] > 0.0]
        if len(pos):
            x[pos] = _destroyed_portion(outlines[pos], area[pos],
                                        np.full(len(pos), corridor, dtype=object))
        zero = np.flatnonzero(area <= 0.0)
        if len(zero):
            x[zero] = (shapely.distance(shapely.points(buildings.xy[zero]), corridor)
                       <= 0.0).astype(np.float64)
        return x

    def c(self, buildings: Extents, corridor: BaseGeometry | None) -> NDArray[np.float64]:
        """c_i = min(1, X_i / tau) for every building."""
        return self.ramp(self.destroyed(buildings, corridor))

    def share(self, buildings: Extents, corridor: BaseGeometry | None,
              w: NDArray[np.float64]) -> float:
        """D = sum c_i w_i / sum w_i (0.0 with no buildings or no weight)."""
        total = float(w.sum()) if len(w) else 0.0
        if total == 0.0:
            return 0.0
        return float((self.c(buildings, corridor) * w).sum() / total)


class IncrementalDestroyed:
    """`RampedDestroyed.destroyed` under a corridor that grows one piece at a time: each building
    keeps its covered region P_i = O_i & C, a new piece K updates only the buildings it touches
    (P_i <- P_i | (O_i & K), X_i = 1 - largest piece of (O_i - P_i) / area(O_i)), and O_i - P_i
    is O_i - C exactly."""

    def __init__(self, measure: RampedDestroyed, buildings: Extents) -> None:
        self.measure = measure
        self._buildings = buildings
        self._outlines = np.asarray(buildings.outlines.to_numpy(), dtype=object)
        self._area = np.asarray(shapely.area(self._outlines), dtype=np.float64)
        self._covered = np.full(len(self._outlines), None, dtype=object)
        self._x = np.zeros(len(self._outlines), dtype=np.float64)
        zero = self._area <= 0.0
        self._zero_idx = np.flatnonzero(zero)
        self._zero_pts = np.asarray(shapely.points(buildings.xy[zero]), dtype=object)

    def add(self, piece: BaseGeometry) -> None:
        hit = self._buildings.touching(piece)
        pos = hit[self._area[hit] > 0.0]
        if len(pos):
            new = shapely.intersection(self._outlines[pos], piece)
            old = self._covered[pos]
            has = np.array([g is not None for g in old], dtype=bool)
            merged = np.array(new, dtype=object)
            if has.any():
                merged[has] = shapely.union(np.array(list(old[has]), dtype=object), new[has])
            self._covered[pos] = merged
            self._x[pos] = _destroyed_portion(self._outlines[pos], self._area[pos], merged)
        if len(self._zero_idx):
            on = shapely.distance(self._zero_pts, piece) <= 0.0
            self._x[self._zero_idx[on]] = 1.0

    def destroyed(self) -> NDArray[np.float64]:
        return self._x.copy()

    def c(self) -> NDArray[np.float64]:
        return self.measure.ramp(self._x)
