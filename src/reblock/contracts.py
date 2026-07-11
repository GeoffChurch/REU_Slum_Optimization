"""Canonical typed contracts — the waist every layer adapts to."""
from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import geopandas as gpd
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely.geometry import Polygon

if TYPE_CHECKING:
    import pandas as pd


def _empty_points() -> GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry")


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
    building_points: GeoDataFrame = field(default_factory=_empty_points)  # real sites; may be empty

    def __post_init__(self) -> None:
        _require_projected(self.crs, "Block.crs")
        _require_columns(self.parcels, {"parcel_id", "geometry"}, "Block.parcels")
        if self.parcels.empty:
            raise ValueError("Block.parcels must be non-empty")
        _require_columns(self.streets, {"geometry"}, "Block.streets")

    @property
    def identity(self) -> tuple[str, str] | None:
        """Content-address for the derivation cache: (source hash, block_id), or
        None when the source hash is unknown (synthetic/test blocks -> uncacheable,
        so they never key-collide). See reblock.derive_graph.derive."""
        return (self.source_content_hash, self.block_id) if self.source_content_hash else None


@dataclass(frozen=True)
class Proposal:
    block_id: str
    crs: CRS
    roads: GeoDataFrame | None = None
    edges: GeoDataFrame | None = None
    proposal_id: str = ""
    method: str = ""
    params: Mapping[str, object] = field(default_factory=dict)
    block_identity: Hashable | None = None

    @property
    def identity(self) -> tuple[Hashable, str] | None:
        """(block_identity, proposal_id) -- proposal_id encodes method+params, so
        it distinguishes proposals for a block. None when block_identity is unknown."""
        return (self.block_identity, self.proposal_id) if self.block_identity is not None else None


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


BBox = tuple[float, float, float, float]   # (minx, miny, maxx, maxy), source CRS-agnostic input


class Source(Protocol):
    def region(self) -> Region: ...
    # block_id + geometry (+ building_count when the source has it -- optional column):
    def block_geometries(self, bbox: BBox | None = None) -> GeoDataFrame: ...
    # points; may be empty:
    def building_points(self, bbox: BBox | None = None) -> GeoDataFrame: ...


class Method(Protocol):
    @property
    def identity(self) -> object: ...
    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal: ...


class Eval(Protocol):
    def score(self, block: Block, proposal: Proposal) -> Metrics: ...


class Screen(Protocol):
    def select(self, source: Source) -> list[str] | None: ...   # selected block_ids, or None => all
