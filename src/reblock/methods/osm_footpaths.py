"""OsmFootpathsReblocker: the reblocker whose 'proposed roads' are the REAL informal circulation
network for the region -- the worn footpaths people already walk, as mapped in OpenStreetMap --
rather than a synthesized one. It fetches those footpaths through a pluggable DesireLineSource
(`OSMDesireLines`), clips them to the block, drops the parts that merely retrace existing streets,
and returns the interior remainder as the intervention. Deriving desire-lines instead from satellite
imagery or from the building-point geometry was explored and dropped -- neither cheap signal matches
OSM's human-mapped network (see docs/superpowers/notes/2026-07-15-desire-line-detection.md)."""
from __future__ import annotations

import hashlib
from collections.abc import Hashable
from dataclasses import dataclass
from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL
from reblock.methods.desire_lines import DesireLineSource
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, with_width


def interior_desire_lines(
    lines: gpd.GeoDataFrame,
    boundary: BaseGeometry,
    streets: BaseGeometry,
    crs: CRS,
    *,
    tol: float = STREET_TOL,
) -> gpd.GeoDataFrame:
    """Clip `lines` to `boundary`, subtract the `streets` corridor (a `tol` buffer), and keep the
    interior LineString remainder longer than `tol` -- the added intervention, excluding the
    perimeter/inter-block streets that are already egress.

    Pure geometry: takes boundary/streets/crs rather than a Block, so the country-wide OSM census
    can call it for blocks that have no building points (and therefore no Voronoi parcels, and
    therefore cannot be constructed as a Block at all). `tol` is exposed because the census sweeps
    it -- OSM ways are digitized against different imagery than the kblock outlines, so a
    boundary-running path more than `tol` off the outline reads as interior.
    """
    empty = gpd.GeoDataFrame(geometry=[], crs=crs)
    if lines.empty:
        return empty
    clipped = lines.clip(cast(Polygon | MultiPolygon, boundary))
    if clipped.empty:
        return empty
    remainder = clipped.geometry.difference(streets.buffer(tol)).explode(index_parts=False)
    mask = ((~remainder.is_empty)
            & (remainder.geom_type == "LineString")
            & (remainder.length > tol))
    # geopandas-stubs' GeoSeries.__getitem__ resolves boolean-mask indexing to the
    # scalar-return overload (-> BaseGeometry) instead of the array-return one; cast to
    # correct it, mirroring the same fixup in reblock.data.shapefile._prepared.
    kept = cast(gpd.GeoSeries, remainder[mask])
    return gpd.GeoDataFrame(geometry=list(kept), crs=crs)


@dataclass
class OsmFootpathsReblocker:
    source: DesireLineSource
    # Total width of the corridor each imported footpath is treated as.
    road_width_m: float = DEFAULT_ROAD_WIDTH_M

    @property
    def identity(self) -> Hashable:
        # Propagate an uncacheable (live) source up so the derivation cache bypasses -- else two
        # different live OSM pulls would key-collide (mirrors clearance + PrebuiltSubstrate).
        if self.source.identity is None:
            return None
        return ("osm_footpaths", self.source.identity, float(self.road_width_m))

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior  # accepted for Method conformance; routing is block-only
        bbox = gpd.GeoSeries([block.boundary], crs=block.crs).to_crs(4326).total_bounds
        lines = self.source.desire_lines(
            (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])), block.crs)
        roads = interior_desire_lines(
            lines, block.boundary, unary_union(list(block.streets.geometry)), block.crs)
        # proposal_id encodes the config so Proposal.identity distinguishes configs on a block
        # (mirrors clearance) -- else two OsmFootpaths configs collide in the eval cache. The
        # source identity is hashed (distinct-per-config yet filesystem-clean -- it feeds render
        # filenames); road_width_m stays literal for legibility. A live (uncacheable) source has
        # drift-prone roads, so its eval must bypass too: block_identity -> None -> uncacheable.
        if self.source.identity is not None:
            src_hash = hashlib.sha256(str(self.source.identity).encode()).hexdigest()[:8]
            pid, block_identity = f"osm_footpaths:w{self.road_width_m:g}:{src_hash}", block.identity
        else:
            pid, block_identity = "osm_footpaths", None
        return Proposal(
            block_id=block.block_id, crs=block.crs, edges=None,
            roads=with_width(roads, self.road_width_m),
            proposal_id=pid, method="osm_footpaths",
            params={"segments": len(roads), "road_width_m": self.road_width_m},
            block_identity=block_identity)
