"""A composable BlockMetric algebra: primitives (Depth/Density/Compactness/Count) and combinators
(Power/Product), each a node exposing `proxy` (fast, from cheap columns), `fine` (true, uses the
peel depth), `needs_peel`, and `identity` (a hashable cache key). A per-metric `Gate` selects.
Only `Depth`'s proxy and fine differ (proxy estimates depth as sqrt(nA)/P; fine uses the real peel);
the geometry primitives are closed forms identical in both. `needs_peel` is an OR over the tree, so
the screen peels iff the expression contains a Depth. See the design spec."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast, runtime_checkable

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from pyproj import CRS

_Identity = tuple[object, ...]

# Absolute floor for the density_compactness metric (n/P², units m^-2). Mirrored by
# conf/metric/density_compactness.yaml; tests/test_metric.py fails if they drift.
#
# CALIBRATED, not chosen. The gate used to be `percentile 30`, which was only ever meant as the
# instrument for finding an absolute threshold -- and a percentile cannot BE the threshold,
# because it re-defines the population every time the corpus changes. Measured: Cape Town's
# percentile-30 cut (n/P² = 1.98e-4) selects 7.6% of the ZAF+KEN censused corpus, so "top 30%"
# means two different things on the two corpora, off by a factor of four.
#
# 3.55e-4 is Cape Town's percentile-10 cut, chosen because that is where the pool's TRUE peel
# depth steps up -- median depth 3 at the 30% and 20% cuts, 4 from 10% on, with the share at
# depth>=4 going 37% -> 43% -> 50% -> 55% (5%) -> 60% (2%). Tightening past 10% keeps buying
# depth, but slowly, while the pool shrinks fast (1,646 -> 823 -> 330 Cape Town blocks).
#
# Sanity check on the magnitude: for a compact block P² is 16A (square) to 4πA (circle), so this
# floor is roughly a density of 4,500-5,700 buildings/km² -- an informal-settlement density, and
# well above the 95th percentile of the rural district polygons the census found (see
# notes/2026-07-28-osm-census-results.md).
DENSITY_COMPACTNESS_FLOOR = 3.55e-4


def _cols(blocks: GeoDataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(count, area, perim) Series from the free kblock columns -- perimeter in a metric CRS so it's
    comparable across blocks; area from `block_area_m2` when present else the reprojected geometry
    area. Blocks already in a projected (metric) CRS are used as-is: re-estimating and reprojecting
    an already-metric CRS can land in the wrong UTM zone near zone/false-easting boundaries and
    distort lengths and areas."""
    crs = blocks.crs
    already_projected = crs is not None and CRS.from_user_input(crs).is_projected
    utm = blocks if already_projected else blocks.to_crs(blocks.estimate_utm_crs())
    count = blocks["building_count"].astype(float)
    area = (blocks["block_area_m2"].astype(float) if "block_area_m2" in blocks.columns
            else utm.geometry.area)
    perim = utm.geometry.length
    return count.reset_index(drop=True), area.reset_index(drop=True), perim.reset_index(drop=True)


@runtime_checkable
class BlockMetric(Protocol):
    # `@property`, not plain attributes: every implementation is a frozen dataclass field (or,
    # for Power/Product's needs_peel, an actual @property) -- both read-only under mypy's
    # structural check, so the protocol must declare them read-only too (mirrors `identity` below).
    @property
    def name(self) -> str: ...

    @property
    def needs_peel(self) -> bool: ...

    def proxy(self, blocks: GeoDataFrame) -> pd.Series: ...

    def fine(self, depth: float, count: float, area: float, perim: float) -> float: ...

    @property
    def identity(self) -> _Identity: ...


@dataclass(frozen=True)
class Depth:
    name: str = "depth"
    needs_peel: bool = True

    def proxy(self, blocks: GeoDataFrame) -> pd.Series:
        count, area, perim = _cols(blocks)
        return cast(pd.Series, np.sqrt(count * area) / perim.where(perim > 0))

    def fine(self, depth: float, count: float, area: float, perim: float) -> float:
        return depth

    @property
    def identity(self) -> _Identity:
        return ("depth",)


@dataclass(frozen=True)
class Density:
    name: str = "density"
    needs_peel: bool = False

    def proxy(self, blocks: GeoDataFrame) -> pd.Series:
        count, area, _ = _cols(blocks)
        return cast(pd.Series, count / area.where(area > 0))

    def fine(self, depth: float, count: float, area: float, perim: float) -> float:
        return count / area if area > 0 else 0.0

    @property
    def identity(self) -> _Identity:
        return ("density",)


DEPTH_DENSITY_PROXY_FLOOR = 0.0128
"""Absolute floor for `depth_density_proxy` = sqrt(nA)/P * n/A. Mirrored by
conf/metric/depth_density_proxy.yaml; tests/test_metric.py fails if they drift.

CALIBRATED against real ground truth, 2026-08-08 -- the City of Cape Town's own informal-settlement
structure survey (117,336 dwellings digitised from Feb 2018 aerial photography at 1:200, Edinburgh
DataShare doi:10.7488/ds/2758), clustered into 189 settlement extents. 683 of 16,451 Cape Town
blocks are informal by that measure.

Chosen to match the SHIPPED pool size, because at equal size this metric strictly dominates the
`density_compactness` floor it replaces -- better precision AND better recall, so adopting it costs
nothing and adjudicates no trade-off:

    floor      blocks   precision   recall
    0.0128      1,646       27.6%    66.6%     <- this
    n/P^2       1,644       24.5%    58.9%     <- DENSITY_COMPACTNESS_FLOOR, for comparison
    0.0181        593       52.1%    45.2%     max-F1
    0.0284        165       81.2%    19.6%     Cape Town p99

Tightening is a deliberate project decision, not a default: it buys a lot of precision and costs a
lot of recall, and it changes which blocks every downstream comparison runs on. The two tighter
values above are measured and one edit away.
"""


@dataclass(frozen=True)
class DepthProxy:
    """`Depth`'s cheap estimator, sqrt(nA)/P, promoted to a metric in its own right.

    The point is `needs_peel = False`. `Depth` needs a Voronoi tessellation and a BFS peel per
    block; this is the same quantity read straight off the free kblock columns, so a screen built
    from it never runs the fine pass at all.

    MEASURED WORTH (notes/2026-08-08-c14-the-two-stage-screen-and-whether-you-need-stage-two.md):
    against real ground truth, `Product(DepthProxy, Density)` selects informal blocks at 81.7%
    precision in the top 1% with NO peel, where the full two-stage pipeline -- peel every survivor,
    rank by true `Product(Depth, Density)` -- reaches 84.1%. The entire expensive stage buys 2.4
    points. Hence this exists, and hence it is the default.

    `fine` and `proxy` are the same closed form here, unlike `Depth` where they differ by
    definition.
    """

    name: str = "depth_proxy"
    needs_peel: bool = False

    def proxy(self, blocks: GeoDataFrame) -> pd.Series:
        count, area, perim = _cols(blocks)
        return cast(pd.Series, np.sqrt(count * area) / perim.where(perim > 0))

    def fine(self, depth: float, count: float, area: float, perim: float) -> float:
        return math.sqrt(count * area) / perim if perim > 0 else 0.0

    @property
    def identity(self) -> _Identity:
        return ("depth_proxy",)


@dataclass(frozen=True)
class Compactness:
    name: str = "compactness"
    needs_peel: bool = False

    def proxy(self, blocks: GeoDataFrame) -> pd.Series:
        _, area, perim = _cols(blocks)
        return cast(pd.Series, area / (perim.where(perim > 0) ** 2))

    def fine(self, depth: float, count: float, area: float, perim: float) -> float:
        return area / perim ** 2 if perim > 0 else 0.0

    @property
    def identity(self) -> _Identity:
        return ("compactness",)


@dataclass(frozen=True)
class Count:
    name: str = "count"
    needs_peel: bool = False

    def proxy(self, blocks: GeoDataFrame) -> pd.Series:
        count, _, _ = _cols(blocks)
        return count

    def fine(self, depth: float, count: float, area: float, perim: float) -> float:
        return count

    @property
    def identity(self) -> _Identity:
        return ("count",)


@dataclass(frozen=True)
class Power:
    base: BlockMetric
    exp: float
    name: str = "power"

    def proxy(self, blocks: GeoDataFrame) -> pd.Series:
        return self.base.proxy(blocks) ** self.exp

    def fine(self, depth: float, count: float, area: float, perim: float) -> float:
        return cast(float, self.base.fine(depth, count, area, perim) ** self.exp)

    @property
    def needs_peel(self) -> bool:
        return self.base.needs_peel

    @property
    def identity(self) -> _Identity:
        return ("power", self.exp, self.base.identity)


@dataclass(frozen=True)
class Product:
    terms: Sequence[BlockMetric]
    name: str = "product"

    def proxy(self, blocks: GeoDataFrame) -> pd.Series:
        out = self.terms[0].proxy(blocks)
        for t in self.terms[1:]:
            out = out * t.proxy(blocks)
        return out

    def fine(self, depth: float, count: float, area: float, perim: float) -> float:
        out = 1.0
        for t in self.terms:
            out *= t.fine(depth, count, area, perim)
        return out

    @property
    def needs_peel(self) -> bool:
        return any(t.needs_peel for t in self.terms)

    @property
    def identity(self) -> _Identity:
        return ("product", tuple(t.identity for t in self.terms))


@dataclass(frozen=True)
class Gate:
    kind: Literal["absolute", "percentile"]
    value: float

    def keep(self, scores: Mapping[str, float]) -> set[str]:
        """The selected block_ids. `absolute` keeps score >= value; `percentile` keeps the top
        `value`% by score (ties included at the cutoff score)."""
        if not scores:
            return set()
        if self.kind == "absolute":
            return {b for b, s in scores.items() if s >= self.value}
        k = max(1, math.ceil(len(scores) * self.value / 100.0))
        cutoff = sorted(scores.values(), reverse=True)[k - 1]
        return {b for b, s in scores.items() if s >= cutoff}

    @property
    def identity(self) -> _Identity:
        return ("gate", self.kind, self.value)
