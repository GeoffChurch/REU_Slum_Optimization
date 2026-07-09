"""Per-block derivations, each memoized through the single derive() primitive
(reblock.derive_graph) on the L2 identities. Before/after are SEPARATE functions
so their distinct fn.identity gives distinct cache keys -- no roads_key. The
algorithm bodies live in reblock.derive.* / reblock.data.kblock and are reused
verbatim; this module only adds the derive() memoization layer (superseding the
four reblock.cache wrappers, deleted in a later layer).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from shapely.geometry import Point, Polygon

from reblock.contracts import Block, Proposal
from reblock.derive.access import parcel_access_layers
from reblock.derive.geometric_access import geometric_access_distances
from reblock.derive_graph import derive


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
