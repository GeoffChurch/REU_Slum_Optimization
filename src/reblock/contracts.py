"""Canonical typed contracts — the waist every layer adapts to."""
from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from geopandas import GeoDataFrame
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon

from reblock.buildings import Extents, tier_identity

if TYPE_CHECKING:
    import pandas as pd

    from reblock.data.counts import BuildingCount


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
    roads: GeoDataFrame | None
    attrs: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Block:
    block_id: str
    crs: CRS
    boundary: Polygon | MultiPolygon   # a block is a Polygon; a gappy region is a MultiPolygon
    parcels: GeoDataFrame
    streets: GeoDataFrame
    source_content_hash: str | None   # content hash of the Source's file(s); None => uncacheable
    # real sites; may be empty. Named for GEOMETRY, not points: at tier 3 it holds polygons.
    building_geometries: GeoDataFrame
    # The injected building-geometry TIER (reblock.buildings), as the STRATEGY rather than a bound
    # instance: SpacingDiscs | AreaDiscs | Footprints. Resolved once where config and data are
    # read, passed down, and applied to this block's own points by `buildings` below -- so no call
    # site re-derives a radius or asks which tier it has.
    #
    # It is a factory and NOT a prebuilt `Extents` for a reason worth keeping: `dataclasses.replace`
    # copies every field it is not given, so a stored instance survives
    # `replace(block, building_geometries=...)` and silently describes the OLD points. That
    # desync is
    # unrepresentable here, because the tier is a function OF the data instead of a copy of it.
    building_tier: Callable[[GeoDataFrame], Extents]
    attrs: Mapping[str, object] = field(default_factory=dict)

    @cached_property
    def buildings(self) -> Extents:
        """This block's buildings at the configured tier; never desyncs from the frame."""
        return self.building_tier(self.building_geometries)

    def __post_init__(self) -> None:
        _require_projected(self.crs, "Block.crs")
        _require_columns(self.parcels, {"parcel_id", "geometry"}, "Block.parcels")
        if self.parcels.empty:
            raise ValueError("Block.parcels must be non-empty")
        _require_columns(self.streets, {"geometry"}, "Block.streets")
        if self.source_content_hash == "":
            raise ValueError("Block.source_content_hash must be a hash, or None when uncacheable")

    @property
    def identity(self) -> tuple[str, str, str] | None:
        """Content-address for the derivation cache: (source hash, block_id), or
        None when the source hash is unknown (synthetic/test blocks -> uncacheable,
        so they never key-collide). See reblock.derive_graph.derive."""
        # The TIER is part of what a block IS to a derivation: methods read `self.buildings`, so
        # the same block at two tiers gives two answers. Without it here, switching tiers would
        # hit the cache and return the OTHER tier's result -- no error, plausible numbers.
        return ((self.source_content_hash, self.block_id, tier_identity(self.building_tier))
                if self.source_content_hash is not None else None)


@dataclass(frozen=True)
class Proposal:
    block_id: str
    crs: CRS
    roads: GeoDataFrame | None
    edges: GeoDataFrame | None
    proposal_id: str
    method: str
    params: Mapping[str, object]
    block_identity: Hashable | None

    def __post_init__(self) -> None:
        # It keys the derivation cache beside `block_identity` and names the rendered file, so an
        # empty one would collide with every other empty one in both places.
        if not self.proposal_id:
            raise ValueError("Proposal.proposal_id must be non-empty")

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
    fields: Mapping[str, pd.Series]


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
    def building_geometries(self, bbox: BBox | None = None) -> GeoDataFrame: ...


class Method(Protocol):
    @property
    def identity(self) -> object: ...
    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal: ...


class Eval(Protocol):
    def score(self, block: Block, proposal: Proposal) -> Metrics: ...


class Screen(Protocol):
    def select(self, source: Source) -> list[str] | None: ...   # selected block_ids, or None => all


@runtime_checkable
class CountingScreen(Protocol):
    """A `Screen` that resolved a `BuildingCount` upstream and can hand it downstream.

    Region growth and the region map both score on building counts, and both must use the SAME
    count the screen ranked on -- until 2026-09-19 neither did, and nothing failed, because each
    read the source's vendor column and got a plausible number.

    A Protocol tested with `isinstance`, not `getattr(screen, "counts", None)`: whether a screen
    HAS this capability is a type question, and making it one means mypy flags a call site that
    cannot supply a counter rather than a default quietly substituting the wrong column.
    """

    def select(self, source: Source) -> list[str] | None: ...

    @property
    def counts(self) -> BuildingCount: ...
