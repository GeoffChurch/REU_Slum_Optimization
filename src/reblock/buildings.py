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
from functools import cached_property, partial
from typing import Protocol, runtime_checkable

import numpy as np
import shapely
from geopandas import GeoDataFrame, GeoSeries
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry.base import BaseGeometry

AREA_COL = "area_in_meters"
# The published POINT each footprint row was sited from: the one the block's parcels were
# tessellated on. A second geometry column beside the outline, set by the loader for every tier.
ANCHOR_COL = "anchor"
# The one place a radius is INVENTED: a lone building has no nearest neighbour to
# measure against. Lives beside the tiers that measure one, not in budget.py.
DEFAULT_BUILDING_RADIUS_M = 3.0   # unchanged from budget.py -- this is a migration
# Nearest CENTRES considered per query point. Radii vary, so nearest-centre is not nearest-SURFACE
# and a few neighbours must be checked; 12 is `vehicle_access.py`'s vetted value.
_K = 12


def tier_identity(tier: object) -> str:
    """Stable name of a building tier FACTORY, for the derivation cache key.

    Methods read `block.buildings`, so their output depends on the tier -- and a cache keyed only on
    `(source_content_hash, block_id)` would hand back results computed under a DIFFERENT tier, with
    no error and numbers that look right. Unwraps `functools.partial`, which is what a Hydra
    `_partial_: true` config produces.
    """
    # A CLOSED set -- a tier class, or the partial Hydra's `_partial_: true` wraps it in -- so
    # discriminate with isinstance, not `getattr(tier, "func", tier)`: a defaulted getattr cannot
    # fail, so it would silently name the wrong thing if a third kind ever turned up.
    fn = tier.func if isinstance(tier, partial) else tier
    if not isinstance(fn, type):
        raise TypeError(f"a building tier must be a class, or a partial of one; got {fn!r}")
    return f"{fn.__module__}.{fn.__qualname__}"


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

    @property
    def outlines(self) -> GeoSeries:
        """Each building's SHAPE at this tier -- a disc for the disc tiers, the real footprint for
        `Footprints`. What displacement integrates over, and what a figure should draw."""

    def touching(self, geom: BaseGeometry) -> NDArray[np.intp]:
        """Indices of the buildings whose outline intersects `geom` -- the only ones a road there
        can displace. Answered from the tier's own spatial index."""

    def displacement(self, corridor: BaseGeometry | None) -> NDArray[np.float64]:
        """Per-building displaced fraction `c_i` in [0, 1]: the share of the building's OUTLINE
        the road corridor covers. See `_overlap_fraction`."""


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


def _overlap_fraction(outlines: GeoSeries, tree: STRtree, area: NDArray[np.float64],
                      xy: NDArray[np.float64], corridor: BaseGeometry | None
                      ) -> NDArray[np.float64]:
    """`c_i = area(outline_i & corridor) / area(outline_i)`: the share of each building the road
    takes. ONE formula for every tier -- a disc tier integrates over its disc, `Footprints` over
    the real outline -- so displacement means the same thing at every tier and tiers compare.

    Exact, not sampled. It is continuous in road position (it was BINARY encroachment that would
    have been a step function), and only buildings the corridor touches can be non-zero, so the
    `STRtree` prefilter makes it one vectorised intersection over the few it hits.

    Replaces `clip(1 - d_centre / r, 0, 1)`, which is not the overlap of anything: it saturates
    at 1 when the corridor reaches the CENTRE, where true overlap is ~0.5. Measured on
    ZAF.9.3.1_1_40972 against real footprints it over-counted displacement by 59%
    (clearance_looped) and 38% (euclidean_grid) -- unevenly by method, which matters because
    Lens A compares methods at EQUAL displacement.

    A building with no extent (r = 0: coincident points) is displaced iff the corridor covers its
    point -- the convention the disc formula used, kept because it has no area to take a share of.
    """
    n = len(outlines)
    c = np.zeros(n, dtype=np.float64)
    if corridor is None or n == 0 or corridor.is_empty:
        return c
    hit = tree.query(corridor, predicate="intersects")
    if len(hit):
        a = area[hit]
        taken = shapely.area(shapely.intersection(outlines.to_numpy()[hit], corridor))
        with np.errstate(divide="ignore", invalid="ignore"):
            c[hit] = np.where(a > 0.0, taken / a, 0.0)
    zero = area <= 0.0
    if zero.any():
        c[zero] = (shapely.distance(shapely.points(xy[zero]), corridor) <= 0.0).astype(np.float64)
    return np.clip(c, 0.0, 1.0)


@dataclass(frozen=True)
class Discs:
    """Discs whose sizes are already KNOWN: `radii[i]` for the building at `points[i]`.

    `SpacingDiscs` and `AreaDiscs` DERIVE a radius (from spacing, from measured area); this takes
    one. Needed wherever the field is fixed by something other than a Source -- the authoring
    bundle's per-building anchor and radius, which the browser rebuilds its block from whatever
    tier baked them, and fixtures that pin displacement against the analytic disc-segment area.
    """

    points: GeoDataFrame
    radii: NDArray[np.float64]

    def __post_init__(self) -> None:
        if len(self.radii) != len(self.points):
            raise ValueError(f"{len(self.radii)} radii for {len(self.points)} buildings")

    def __len__(self) -> int:
        return len(self.points)

    @cached_property
    def xy(self) -> NDArray[np.float64]:
        return np.c_[self.points.geometry.x.to_numpy(), self.points.geometry.y.to_numpy()]

    @cached_property
    def outlines(self) -> GeoSeries:
        return self.points.geometry.buffer(self.radii)

    @cached_property
    def _outline_tree(self) -> STRtree:
        return STRtree(self.outlines.to_numpy())

    @cached_property
    def _outline_area(self) -> NDArray[np.float64]:
        return self.outlines.area.to_numpy(dtype=np.float64)

    def clearance(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        return _disc_clearance(self.xy, self.radii, pts)

    def touching(self, geom: BaseGeometry) -> NDArray[np.intp]:
        return np.asarray(self._outline_tree.query(geom, predicate="intersects"), dtype=np.intp)

    def displacement(self, corridor: BaseGeometry | None) -> NDArray[np.float64]:
        return _overlap_fraction(self.outlines, self._outline_tree, self._outline_area,
                                 self.xy, corridor)


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

    @cached_property
    def outlines(self) -> GeoSeries:
        return self.points.geometry.buffer(self.radii)

    @cached_property
    def _outline_tree(self) -> STRtree:
        return STRtree(self.outlines.to_numpy())

    @cached_property
    def _outline_area(self) -> NDArray[np.float64]:
        return self.outlines.area.to_numpy(dtype=np.float64)

    def touching(self, geom: BaseGeometry) -> NDArray[np.intp]:
        return np.asarray(self._outline_tree.query(geom, predicate="intersects"), dtype=np.intp)

    def displacement(self, corridor: BaseGeometry | None) -> NDArray[np.float64]:
        return _overlap_fraction(self.outlines, self._outline_tree, self._outline_area,
                                 self.xy, corridor)


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
                "KBlockSource.building_geometries must read it (it long read columns=['geometry'] "
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

    @cached_property
    def outlines(self) -> GeoSeries:
        return self.points.geometry.buffer(self.radii)

    @cached_property
    def _outline_tree(self) -> STRtree:
        return STRtree(self.outlines.to_numpy())

    @cached_property
    def _outline_area(self) -> NDArray[np.float64]:
        return self.outlines.area.to_numpy(dtype=np.float64)

    def touching(self, geom: BaseGeometry) -> NDArray[np.intp]:
        return np.asarray(self._outline_tree.query(geom, predicate="intersects"), dtype=np.intp)

    def displacement(self, corridor: BaseGeometry | None) -> NDArray[np.float64]:
        return _overlap_fraction(self.outlines, self._outline_tree, self._outline_area,
                                 self.xy, corridor)


@dataclass(frozen=True)
class Footprints:
    """Tier 3: the real Open Buildings outlines.

    `clearance` is the exact distance to the polygon, and `displacement` is the true OVERLAP
    FRACTION `area(footprint & corridor) / area(footprint)` rather than a disc approximation --
    so a corridor clipping a building's corner costs a corner, not a whole home.

    `xy` is each row's `ANCHOR_COL` point, NOT the polygon's centroid. The parcels were
    tessellated on the anchors, and a centroid sits millimetres off its anchor -- enough to leave
    a building on a block edge contained by no parcel at all, and to be given radius 0 there.
    """

    footprints: GeoDataFrame

    def __post_init__(self) -> None:
        kinds = set(self.footprints.geometry.geom_type.dropna().unique())
        bad = kinds - {"Polygon", "MultiPolygon"}
        if bad:
            raise ValueError(
                f"Footprints needs polygon outlines; got {sorted(bad)}. A POINT has zero area, so "
                "every radius would be 0 and every displacement nonsense -- silently. Point data "
                "is the SpacingDiscs/AreaDiscs tier; fetch the polygon tiles for this one.")
        if ANCHOR_COL not in self.footprints:
            raise ValueError(
                f"Footprints needs the {ANCHOR_COL!r} column (each outline's published point); got "
                f"{list(self.footprints.columns)}. The polygon centroid is not a substitute: the "
                "parcels were tessellated on the anchors, and a centroid can fall outside them.")

    def __len__(self) -> int:
        return len(self.footprints)

    @cached_property
    def polygons(self) -> GeoSeries:
        return self.footprints.geometry

    @cached_property
    def xy(self) -> NDArray[np.float64]:
        a = GeoSeries(self.footprints[ANCHOR_COL])
        return np.c_[a.x.to_numpy(), a.y.to_numpy()]

    @cached_property
    def radii(self) -> NDArray[np.float64]:
        return np.sqrt(self.footprints.geometry.area.to_numpy(dtype=np.float64) / np.pi)

    @cached_property
    def outlines(self) -> GeoSeries:
        return self.footprints.geometry

    @cached_property
    def _outline_tree(self) -> STRtree:
        return STRtree(self.outlines.to_numpy())

    @cached_property
    def _outline_area(self) -> NDArray[np.float64]:
        return self.outlines.area.to_numpy(dtype=np.float64)

    def clearance(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        if len(self.footprints) == 0:
            return np.full(len(pts), np.inf)
        _, dist = self._outline_tree.query_nearest(
            shapely.points(pts[:, 0], pts[:, 1]), return_distance=True, all_matches=False)
        return np.asarray(dist, dtype=np.float64)

    def touching(self, geom: BaseGeometry) -> NDArray[np.intp]:
        return np.asarray(self._outline_tree.query(geom, predicate="intersects"), dtype=np.intp)

    def displacement(self, corridor: BaseGeometry | None) -> NDArray[np.float64]:
        return _overlap_fraction(self.outlines, self._outline_tree, self._outline_area,
                                 self.xy, corridor)



class IncrementalOverlap:
    """EXACT displacement under a corridor that grows one piece at a time.

    Displacement is the share of each building's outline the corridor covers, and that composes
    under union by AREA -- not by `min` (the old centre distance) and not by `max`: two pieces
    covering DIFFERENT slices of one building both count, two covering the SAME slice count once.
    So keep each building's covered region `P_i = O_i & C`, and let a new piece `K` touch only the
    buildings it reaches:

        P_i <- P_i | (O_i & K),        c_i = area(P_i) / area(O_i)

    Exact, local, and never a union of `K` into the whole corridor -- which is what made scoring a
    candidate against a grown corridor expensive. `delta` scores a piece without committing it
    (safe to call from a fork pool); `add` commits it.

    Used where displacement must be enforced or scored incrementally: `greedy_arterial`'s
    `displacement_fast` and `resistance_lp`'s rounding, which is the step that enforces its cap.
    """

    def __init__(self, buildings: Extents) -> None:
        self._buildings = buildings
        n = len(buildings)
        self._outlines = np.asarray(buildings.outlines.to_numpy(), dtype=object)
        self._area = np.asarray(shapely.area(self._outlines), dtype=np.float64)
        self._covered = np.full(n, None, dtype=object)     # P_i; None until a piece touches i
        self._taken = np.zeros(n, dtype=np.float64)        # area(P_i)
        # A building with no extent is displaced iff the corridor covers its point.
        zero = self._area <= 0.0
        self._zero_idx = np.flatnonzero(zero)
        self._zero_pts = np.asarray(shapely.points(buildings.xy[zero]), dtype=object)
        self._zero_hit = np.zeros(len(self._zero_idx), dtype=bool)

    def c(self) -> NDArray[np.float64]:
        """Per-building displaced fraction under everything added so far."""
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.clip(np.where(self._area > 0.0, self._taken / self._area, 0.0), 0.0, 1.0)
        c[self._zero_idx] = self._zero_hit.astype(np.float64)
        return c

    def total(self) -> float:
        return float(self.c().sum())

    def _pieces(self, piece: BaseGeometry
                ) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.object_]]:
        """For the buildings `piece` touches: their indices, their covered area AFTER adding it,
        and the new covered regions `P_i | (O_i & piece)`."""
        hit = self._buildings.touching(piece)
        if not len(hit):
            return hit, np.zeros(0), np.zeros(0, dtype=object)
        new_part = shapely.intersection(self._outlines[hit], piece)
        old = self._covered[hit]
        has = np.array([g is not None for g in old], dtype=bool)
        merged = np.array(new_part, dtype=object)
        if has.any():
            merged[has] = shapely.union(np.array(list(old[has]), dtype=object), new_part[has])
        return hit, np.asarray(shapely.area(merged), dtype=np.float64), merged

    def _zero_newly(self, piece: BaseGeometry) -> NDArray[np.bool_]:
        if not len(self._zero_idx):
            return np.zeros(0, dtype=bool)
        return (shapely.distance(self._zero_pts, piece) <= 0.0) & ~self._zero_hit

    def delta(self, piece: BaseGeometry) -> float:
        """How much `piece` would add to total displacement, without committing it."""
        hit, taken_after, _ = self._pieces(piece)
        d = 0.0
        if len(hit):
            a = self._area[hit]
            pos = a > 0.0
            with np.errstate(divide="ignore", invalid="ignore"):
                before = np.clip(np.where(pos, self._taken[hit] / a, 0.0), 0.0, 1.0)
                after = np.clip(np.where(pos, taken_after / a, 0.0), 0.0, 1.0)
            d += float((after - before)[pos].sum())
        return d + float(self._zero_newly(piece).sum())

    def add(self, piece: BaseGeometry) -> None:
        """Commit `piece` to the corridor."""
        hit, taken_after, merged = self._pieces(piece)
        if len(hit):
            self._covered[hit] = merged
            self._taken[hit] = taken_after
        newly = self._zero_newly(piece)
        if newly.any():
            self._zero_hit |= newly
