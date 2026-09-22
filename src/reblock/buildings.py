"""Building geometry at three tiers, as a capability ladder the type checker can enforce.

Open Buildings ships three levels of detail and we can now read all three:

    Positions  (xy)  <-  Extents  (radii, clearance, displacement)  <-  Shapes  (polygons)

A function that needs a SIZE takes `Extents` and cannot be handed bare positions; one that needs a
true OUTLINE takes `Shapes` and rejects discs. Both are type errors at the call site rather than
runtime crashes -- the same reason `x.field` beats `getattr(x, "field", default)`.

The tier is resolved ONCE where config and data are read, and the instance is passed down;
downstream never asks which tier it has. Availability is data-dependent, so a configured tier the
data cannot supply must raise at LOAD -- loudly, naming the block -- and never degrade silently to
a coarser tier, because a silent degrade is exactly the failure that cannot be seen in the output.

WHY THE TIER MATTERS, measured on `ZAF.9.3.1_1_40972`: discs report 0.734 of buildings served by a
3.5 m channel where real footprints report 0.548 -- a 34% relative OVER-read. The error is
width-dependent (+0.17 at 3.0 m, +0.04 at 6.0 m), which pins the mechanism: a narrow channel
squeezes through the corner gap between two circles inscribed in abutting rectangles, and that gap
does not physically exist. It is NOT an area effect -- true median footprint area (16.9 m^2)
slightly exceeds the disc area (15.0 m^2). See
`docs/superpowers/notes/2026-09-21-vehicle-access-and-the-gate-width-knob.md`, Result 8.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Protocol, runtime_checkable

import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from shapely.geometry.base import BaseGeometry

AREA_COL = "area_in_meters"
# The one place a radius is INVENTED: a lone building has no nearest neighbour to
# measure against. Lives beside the tiers that measure one, not in budget.py.
DEFAULT_BUILDING_RADIUS_M = 3.0   # unchanged from budget.py -- this is a migration
# Nearest CENTRES considered per query point. Radii vary, so nearest-centre is not nearest-SURFACE
# and a few neighbours must be checked; 12 is `vehicle_access.py`'s vetted value.
_K = 12


@runtime_checkable
class Positions(Protocol):
    """Tier 1: where the buildings are, and nothing else."""

    @property
    def xy(self) -> NDArray[np.float64]:
        """(n, 2) centre coordinates in the block's CRS."""

    def __len__(self) -> int: ...


@runtime_checkable
class Extents(Positions, Protocol):
    """Tier 2: buildings that have a SIZE, so clearance and displacement are defined."""

    @property
    def radii(self) -> NDArray[np.float64]:
        """Per-building equivalent radius, in metres."""

    def clearance(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        """Distance from each of `pts` (m, 2) to the nearest building SURFACE, floored at 0."""

    def displacement(self, corridor: BaseGeometry | None) -> NDArray[np.float64]:
        """Per-building displaced fraction `c_i` in [0, 1] for a road corridor polygon."""


@runtime_checkable
class Shapes(Extents, Protocol):
    """Tier 3: buildings with a true outline."""

    @property
    def polygons(self) -> GeoSeries:
        """Per-building footprint geometry."""


def _disc_clearance(xy: NDArray[np.float64], radii: NDArray[np.float64],
                    pts: NDArray[np.float64]) -> NDArray[np.float64]:
    """`min_i(|p - c_i| - r_i)`, floored at 0 -- distance to the nearest disc's surface.

    Analytic, NOT a raster EDT over a rasterized disc mask: rasterizing by centre-inclusion drops
    every cell whose centre falls outside r but which the disc still overlaps, so the obstacle
    reads systematically SMALLER and the clearance systematically WIDER.
    """
    if len(xy) == 0:
        return np.full(len(pts), np.inf)
    d, idx = cKDTree(xy).query(pts, k=min(_K, len(xy)))
    d = np.atleast_2d(d)
    idx = np.atleast_2d(idx)
    return np.maximum((d - radii[idx]).min(axis=1), 0.0)


def _disc_displacement(radii: NDArray[np.float64], d: NDArray[np.float64]
                       ) -> NDArray[np.float64]:
    """`c_i = clip(1 - d_i/r_i, 0, 1)`. r_i = 0 counts as fully displaced iff it is inside."""
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(radii > 0.0, 1.0 - d / radii, np.where(d <= 0.0, 1.0, 0.0))
    return np.clip(c, 0.0, 1.0)


@dataclass(frozen=True)
class SpacingDiscs:
    """Tier 2 from positions ALONE: radius = half the nearest-neighbour distance.

    This is the historical default, and naming it is the point. The radius is a fact about
    SPACING, not about the building -- so this tier structurally cannot distinguish dense-small
    from sparse-large fabric. Measured: median true footprint radius is 2.30 m on BOTH example
    blocks while NN/2 gives 2.19 m and 2.61 m. Prefer `AreaDiscs` whenever `area_in_meters` is
    present, which in this corpus is always.
    """

    points: GeoDataFrame

    def __len__(self) -> int:
        return len(self.points)

    @cached_property
    def xy(self) -> NDArray[np.float64]:
        return np.c_[self.points.geometry.x.to_numpy(), self.points.geometry.y.to_numpy()]

    @cached_property
    def radii(self) -> NDArray[np.float64]:
        n = len(self.points)
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        if n < 2:
            return np.full(n, DEFAULT_BUILDING_RADIUS_M, dtype=np.float64)
        dist, _ = cKDTree(self.xy).query(self.xy, k=2)     # k=2: self (0) + nearest other
        return (dist[:, 1] * 0.5).astype(np.float64)

    def clearance(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        return _disc_clearance(self.xy, self.radii, pts)

    def displacement(self, corridor: BaseGeometry | None) -> NDArray[np.float64]:
        if corridor is None:
            return np.zeros(len(self.points))
        return _disc_displacement(
            self.radii, self.points.geometry.distance(corridor).to_numpy())


@dataclass(frozen=True)
class AreaDiscs:
    """Tier 2 from positions + measured area: radius = sqrt(area / pi), the equal-area disc.

    Free -- `area_in_meters` already rides along in the buildings parquet and was being dropped at
    the reader. Unlike `SpacingDiscs` the radius is a property of the BUILDING, so density and
    size are finally independent.
    """

    points: GeoDataFrame

    def __post_init__(self) -> None:
        if AREA_COL not in self.points:
            raise ValueError(
                f"AreaDiscs needs the {AREA_COL!r} column; got {list(self.points.columns)}. "
                "KBlockSource.building_points must read it (it long read columns=['geometry'] "
                "and dropped it).")

    def __len__(self) -> int:
        return len(self.points)

    @cached_property
    def xy(self) -> NDArray[np.float64]:
        return np.c_[self.points.geometry.x.to_numpy(), self.points.geometry.y.to_numpy()]

    @cached_property
    def radii(self) -> NDArray[np.float64]:
        a = self.points[AREA_COL].to_numpy(dtype=np.float64)
        return np.sqrt(np.maximum(a, 0.0) / np.pi)

    def clearance(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        return _disc_clearance(self.xy, self.radii, pts)

    def displacement(self, corridor: BaseGeometry | None) -> NDArray[np.float64]:
        if corridor is None:
            return np.zeros(len(self.points))
        return _disc_displacement(
            self.radii, self.points.geometry.distance(corridor).to_numpy())


@dataclass(frozen=True)
class Footprints:
    """Tier 3: the real Open Buildings outlines.

    `clearance` is the exact distance to the polygon, and `displacement` is the true OVERLAP
    FRACTION `area(footprint & corridor) / area(footprint)` rather than a disc approximation --
    so a corridor clipping a building's corner costs a corner, not a whole home.
    """

    footprints: GeoDataFrame

    def __len__(self) -> int:
        return len(self.footprints)

    @cached_property
    def polygons(self) -> GeoSeries:
        return self.footprints.geometry

    @cached_property
    def xy(self) -> NDArray[np.float64]:
        c = self.footprints.geometry.centroid
        return np.c_[c.x.to_numpy(), c.y.to_numpy()]

    @cached_property
    def radii(self) -> NDArray[np.float64]:
        return np.sqrt(self.footprints.geometry.area.to_numpy(dtype=np.float64) / np.pi)

    @cached_property
    def _tree(self) -> object:
        from shapely import STRtree
        return STRtree(list(self.footprints.geometry))

    def clearance(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        import shapely
        if len(self.footprints) == 0:
            return np.full(len(pts), np.inf)
        _, dist = self._tree.query_nearest(                     # type: ignore[attr-defined]
            shapely.points(pts[:, 0], pts[:, 1]), return_distance=True, all_matches=False)
        return np.asarray(dist, dtype=np.float64)

    def displacement(self, corridor: BaseGeometry | None) -> NDArray[np.float64]:
        if corridor is None:
            return np.zeros(len(self.footprints))
        area = self.footprints.geometry.area.to_numpy(dtype=np.float64)
        taken = self.footprints.geometry.intersection(corridor).area.to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(area > 0.0, taken / area, 0.0)
        return np.clip(c, 0.0, 1.0)

