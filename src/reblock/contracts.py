"""Canonical typed contracts — the waist every layer adapts to."""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd
import shapely
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon

from reblock.buildings import Extents, tier_identity

if TYPE_CHECKING:
    from reblock.data.counts import BuildingCount
    from reblock.metric import BlockMetric


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
    # Content hash of the Source's files AND the code that read them (`derive_graph.reader_hash`);
    # None => uncacheable.
    source_content_hash: str | None
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
    proposal_id: str       # a display label: it names rendered files and log lines
    method: str
    params: Mapping[str, object]

    def __post_init__(self) -> None:
        # It names the rendered file, so an empty one would collide with every other empty one.
        if not self.proposal_id:
            raise ValueError("Proposal.proposal_id must be non-empty")

    @cached_property
    def identity(self) -> str:
        """A content hash of the roads: the part of a Proposal that a derivation over
        `(block, proposal)` reads. The block rides the key as its own input.

        Not a hand-written id. One per method had to tell apart every configuration the method's
        own identity does, and `cycle_native`'s omitted its substrate; a truncated road set
        (`replace(p, roads=prefix)`) kept its id and had to switch the cache off by hand. Content
        makes every different road set a different key and lets equal ones share one.
        """
        h = hashlib.sha256(self.crs.to_wkt().encode())
        if self.roads is not None:
            h.update(repr([(str(n), str(t)) for n, t in self.roads.dtypes.items()]).encode())
            geometry = self.roads.geometry.name
            for name in self.roads.columns:
                if name == geometry:
                    for wkb in shapely.to_wkb(self.roads.geometry.to_numpy()):    # exact bytes
                        h.update(len(wkb).to_bytes(8, "little") + wkb)
                else:
                    h.update(pd.util.hash_pandas_object(self.roads[name], index=False)
                             .to_numpy().tobytes())
        return h.hexdigest()


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


@runtime_checkable
class Source(Protocol):
    def region(self) -> Region: ...
    # block_id + geometry (+ building_count when the source has it -- optional column):
    def block_geometries(self, bbox: BBox | None = None) -> GeoDataFrame: ...
    # points; may be empty:
    def building_geometries(self, bbox: BBox | None = None) -> GeoDataFrame: ...

    def restricted(self, block_ids: Sequence[str]) -> Source:
        """This source yielding only `block_ids`: how a caller builds a subset's Blocks.

        A new source, never a narrowing of this one. The configured source is shared by every
        stage and emitter of a run, and narrowing it in place was a side channel: region building
        set the filter to the members, and each entry point had to remember to clear it again
        before an emitter read the whole metro. An id the source does not have raises when the
        restricted source is read."""
        ...


@runtime_checkable
class CountableSource(Source, Protocol):
    """A `Source` backed by one building-point file over its whole corpus: the file a
    `BuildingCount` counts. Asked for with `isinstance` where a count is resolved, so a source
    with no such file (a parcel shapefile) fails there by name instead of on a missing
    attribute."""

    @property
    def buildings_path(self) -> Path: ...


@runtime_checkable
class Method(Protocol):
    @property
    def identity(self) -> Hashable | None: ...   # None: uncacheable (see `derive_graph.derive`)
    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal: ...


@runtime_checkable
class Eval(Protocol):
    def score(self, block: Block, proposal: Proposal) -> Metrics: ...


@runtime_checkable
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


@runtime_checkable
class ScoringScreen(CountingScreen, Protocol):
    """A `CountingScreen` that ranks blocks by a `BlockMetric` and hands back each flagged block's
    score.

    Region growth ranks its candidates by the SAME metric the screen flagged on, and the region map
    colours by it. Both ask with `isinstance`, not `getattr(screen, "metric", None)`: a screen that
    does not score is a type question, and a missing metric was a default that quietly substituted
    "no metric" for whichever screen was passed.
    """

    @property
    def metric(self) -> BlockMetric: ...

    def selection_scores(self, source: Source) -> dict[str, float]: ...
