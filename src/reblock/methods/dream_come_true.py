"""DreamComeTrueReblocker: the reblocker whose 'proposed roads' are the REAL informal circulation
network for the region -- the worn footpaths people already walk -- rather than a synthesized one.
The desire-lines come from a pluggable DesireLineSource (OSM in Phase 1; a satellite-imagery
detector later). The method is source-agnostic: fetch desire-lines for the region bbox, clip them
to the block, drop the parts that merely retrace existing streets, and return the interior remainder
as the intervention. See docs/superpowers/specs/2026-07-15-dream-come-true-design.md."""
from __future__ import annotations

import hashlib
from collections.abc import Hashable
from dataclasses import dataclass
from typing import cast

import geopandas as gpd
from shapely.ops import unary_union

from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL
from reblock.methods.desire_lines import DesireLineSource


def _interior_desire_lines(lines: gpd.GeoDataFrame, block: Block) -> gpd.GeoDataFrame:
    """Clip `lines` to the block, subtract the existing-street corridor (STREET_TOL buffer), and
    keep the interior LineString remainder above the tolerance length -- the added intervention,
    excluding the perimeter/inter-block streets that are already egress."""
    empty = gpd.GeoDataFrame(geometry=[], crs=block.crs)
    if lines.empty:
        return empty
    clipped = lines.clip(block.boundary)
    if clipped.empty:
        return empty
    street_corridor = unary_union(list(block.streets.geometry)).buffer(STREET_TOL)
    remainder = clipped.geometry.difference(street_corridor).explode(index_parts=False)
    mask = ((~remainder.is_empty)
            & (remainder.geom_type == "LineString")
            & (remainder.length > STREET_TOL))
    # geopandas-stubs' GeoSeries.__getitem__ resolves boolean-mask indexing to the
    # scalar-return overload (-> BaseGeometry) instead of the array-return one; cast to
    # correct it, mirroring the same fixup in reblock.data.shapefile._prepared.
    kept = cast(gpd.GeoSeries, remainder[mask])
    return gpd.GeoDataFrame(geometry=list(kept), crs=block.crs)


@dataclass
class DreamComeTrueReblocker:
    source: DesireLineSource
    corridor_m: float = 3.0

    @property
    def identity(self) -> Hashable:
        # Propagate an uncacheable (live) source up so the derivation cache bypasses -- else two
        # different live OSM pulls would key-collide (mirrors clearance + PrebuiltSubstrate).
        if self.source.identity is None:
            return None
        return ("dream_come_true", self.source.identity, float(self.corridor_m))

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior  # accepted for Method conformance; routing is block-only
        bbox = gpd.GeoSeries([block.boundary], crs=block.crs).to_crs(4326).total_bounds
        lines = self.source.desire_lines(
            (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])), block.crs)
        roads = _interior_desire_lines(lines, block)
        # proposal_id encodes the config so Proposal.identity distinguishes configs on a block
        # (mirrors clearance) -- else two DreamComeTrue configs collide in the eval cache. The
        # source identity is hashed (distinct-per-config yet filesystem-clean -- it feeds render
        # filenames); corridor_m stays literal for legibility. A live (uncacheable) source has
        # drift-prone roads, so its eval must bypass too: block_identity -> None -> uncacheable.
        if self.source.identity is not None:
            src_hash = hashlib.sha256(str(self.source.identity).encode()).hexdigest()[:8]
            pid, block_identity = f"dream_come_true:c{self.corridor_m:g}:{src_hash}", block.identity
        else:
            pid, block_identity = "dream_come_true", None
        return Proposal(
            block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
            proposal_id=pid, method="dream_come_true",
            params={"segments": len(roads), "corridor_m": self.corridor_m},
            block_identity=block_identity)
