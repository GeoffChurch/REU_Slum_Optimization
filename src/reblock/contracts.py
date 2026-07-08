"""Canonical typed contracts — the waist every layer adapts to."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from geopandas import GeoDataFrame
from pyproj import CRS
from shapely.geometry import Polygon

if TYPE_CHECKING:
    import pandas as pd


def _require_columns(gdf: GeoDataFrame, cols: set[str], name: str) -> None:
    missing = cols - set(gdf.columns)
    if missing:
        raise ValueError(f"{name} is missing required column(s): {sorted(missing)}")


def _require_projected(crs: CRS, name: str) -> None:
    if crs is None or not CRS.from_user_input(crs).is_projected:
        raise ValueError(f"{name} must have a projected (metric) CRS, got: {crs}")


@dataclass(frozen=True)
class Region:
    region_id: str
    crs: CRS
    blocks: Iterable[Block]
    roads: GeoDataFrame | None = None
    attrs: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Block:
    block_id: str
    crs: CRS
    boundary: Polygon
    parcels: GeoDataFrame
    streets: GeoDataFrame
    source_content_hash: str = ""   # content hash of the Source's file(s); "" => uncacheable
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_projected(self.crs, "Block.crs")
        _require_columns(self.parcels, {"parcel_id", "geometry"}, "Block.parcels")
        if self.parcels.empty:
            raise ValueError("Block.parcels must be non-empty")
        _require_columns(self.streets, {"geometry"}, "Block.streets")


@dataclass(frozen=True)
class Proposal:
    block_id: str
    crs: CRS
    roads: GeoDataFrame | None = None
    edges: GeoDataFrame | None = None
    proposal_id: str = ""
    method: str = ""
    params: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Metrics:
    block_id: str
    method: str
    eval: str
    values: Mapping[str, float]
    fields: Mapping[str, pd.Series] = field(default_factory=dict)


@dataclass(frozen=True)
class Result:
    block: Block
    proposal: Proposal
    metrics: tuple[Metrics, ...]

    def metric(self, eval: str, key: str) -> float:
        for m in self.metrics:
            if m.eval == eval:
                return m.values[key]
        raise KeyError(f"no metric {key!r} for eval {eval!r}")


class Source(Protocol):
    def region(self) -> Region: ...


class Method(Protocol):
    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal: ...


class RegionMethod(Protocol):
    def propose(self, region: Region) -> Iterable[Proposal]: ...


class Eval(Protocol):
    def score(self, block: Block, proposal: Proposal) -> Metrics: ...


class Screen(Protocol):
    def select(self) -> list[str]: ...   # informal block_ids, sorted
