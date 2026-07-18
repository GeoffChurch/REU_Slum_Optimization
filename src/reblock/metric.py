"""A composable BlockMetric algebra: primitives (Depth/Density/Compactness) and combinators
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
