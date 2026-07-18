"""Per-block derivations, each memoized through the single derive() primitive
(reblock.derive_graph) on the L2 identities. Before/after are SEPARATE functions
so their distinct fn.identity gives distinct cache keys -- no roads_key. The
algorithm bodies live in reblock.derive.* / reblock.data.kblock and are reused
verbatim; this module only adds the derive() memoization layer (which replaced
F2's four per-function cache wrappers).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
from shapely.geometry import Point, Polygon

from reblock.contracts import Block, Method, Proposal
from reblock.derive.access import parcel_access_layers
from reblock.derive.geometric_access import geometric_access_distances
from reblock.derive_graph import derive

if TYPE_CHECKING:
    from reblock.metric import BlockMetric, Gate


def _access_before_impl(block: Block) -> pd.Series:
    return parcel_access_layers(block, None)


def access_before(block: Block) -> pd.Series:
    return derive(_access_before_impl, block)


def _access_after_impl(block: Block, proposal: Proposal) -> pd.Series:
    return parcel_access_layers(block, proposal.roads)


def access_after(block: Block, proposal: Proposal) -> pd.Series:
    return derive(_access_after_impl, block, proposal)


def _geometric_after_impl(block: Block, proposal: Proposal) -> pd.Series:
    return geometric_access_distances(block, proposal.roads)


def geometric_after(block: Block, proposal: Proposal) -> pd.Series:
    return derive(_geometric_after_impl, block, proposal)


def _propose_impl(method: Method, block: Block) -> Proposal:
    return method.propose(block)


def propose(method: Method, block: Block) -> Proposal:
    return derive(_propose_impl, method, block)


@dataclass(frozen=True)
class VoronoiInput:
    """Identified carrier for the Voronoi build: derive() keys on .identity
    (never the geometry); a missing source_id makes it uncacheable (bypass)."""
    source_id: str
    block_id: str
    poly: Polygon
    points: list[Point]
    crs: Any

    @property
    def identity(self) -> tuple[str, str, str] | None:
        return ("voronoi", self.source_id, self.block_id) if self.source_id else None


def _voronoi_impl(vin: VoronoiInput) -> Any:
    from reblock.data.kblock import _voronoi_parcels  # local import avoids a cycle
    return _voronoi_parcels(vin.poly, vin.points, vin.crs)


def voronoi(vin: VoronoiInput) -> Any:
    return derive(_voronoi_impl, vin)


@dataclass(frozen=True)
class ScreenSelectionInput:
    """Identified carrier for a DenseCompactScreen selection: derive() keys on .identity
    (the source content hash + the metric + gate + pre-filter), never the paths -- so a rerun
    with the same source + metric + gate returns the ranked block_ids from one L2 lookup instead
    of rebuilding the thousands of survivor blocks (Voronoi + access-depth) the fine pass walks.
    A missing source hash makes it uncacheable (bypass). The selection logic is hashed into the
    derive() key (screen/dense_compact.py and metric.py are in derive_graph._DERIVATION_MODULES),
    so a gating/ranking change busts the cache automatically -- no hand-bumped version."""
    source_hash: str
    blocks_path: str
    buildings_path: str
    metric: BlockMetric             # the scorer (carried for _compute_selection; identity below)
    gate: Gate                      # the selection gate
    proxy_keep_pct: float          # cheap recall pre-filter: keep top-% by proxy (peel metrics)
    min_buildings: int

    @property
    def identity(self) -> tuple[object, ...] | None:
        return (("dense-compact-screen", self.source_hash, self.metric.identity,
                 self.gate.identity, self.proxy_keep_pct, self.min_buildings)
                if self.source_hash else None)


def _screen_selection_impl(inp: ScreenSelectionInput) -> list[tuple[str, float]]:
    from reblock.screen.dense_compact import _compute_selection  # local import avoids a cycle
    return _compute_selection(inp)


def screen_selection(inp: ScreenSelectionInput) -> list[tuple[str, float]]:
    return derive(_screen_selection_impl, inp)
