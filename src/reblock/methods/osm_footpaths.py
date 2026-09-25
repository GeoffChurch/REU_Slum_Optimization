"""OsmFootpathsReblocker: the reblocker whose 'proposed roads' are the REAL informal circulation
network for the region -- the worn footpaths people already walk, as mapped in OpenStreetMap --
rather than a synthesized one. It fetches those footpaths through a pluggable `FootpathSource`
(`OSMDesireLines`, `PbfDesireLines`), clips them to the block, drops the parts that merely retrace
existing streets, and returns the interior remainder as the intervention (`block_footpaths`).
Deriving desire-lines instead from satellite imagery or from the building-point geometry was
explored and dropped -- neither cheap signal matches OSM's human-mapped network (see
docs/superpowers/notes/2026-07-15-desire-line-detection.md).

## What the width means

An imported footpath is an ALIGNMENT -- evidence of where people already walk -- not a width claim.
A real footpath is 1.5-3 m; what this method proposes is to WIDEN it into a street along that proven
desire line, so `road_width_m` is the width of the road built there, and the displacement it scores
is the cost of the buildings that must go to make room. That is why the default is a full two-way
street rather than anything footpath-sized.

Making them one-way instead (cheaper: 4 m against 7 m) is MARGINAL rather than useful: interior
footpath geometry is mostly bridges, and Robbins forbids orienting a bridge. On 30 Nairobi blocks
the OSM census says carry >=250 m of interior footpath, 21/30 are bridges end to end and only
**18.8% of all footpath metres are orientable at all**. Combined with the finding that one-way loses
to two-way at equal buildable footprint anyway
(`notes/2026-07-31-one-way-is-dominated.md`), there is no case for it here.
"""
from __future__ import annotations

import hashlib
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union

from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL
from reblock.derive_graph import config_identity
from reblock.permeability import with_width


@runtime_checkable
class FootpathSource(Protocol):
    """The MAPPED footpath network: every way in a bbox, as it was drawn. `OSMDesireLines` reads it
    from Overpass, `reblock.data.osm_extract.PbfDesireLines` from a local extract; a later imagery
    detector would be a third."""

    def footpaths(
        self, bbox_wgs84: tuple[float, float, float, float], crs: CRS
    ) -> gpd.GeoDataFrame: ...
    @property
    def identity(self) -> Hashable: ...


def _reach_street(line: LineString, streets: BaseGeometry, reach: float) -> LineString:
    """Extend back to the street either end that the street-corridor subtraction cut off.

    `difference(streets.buffer(tol))` leaves a cut end lying ON the corridor boundary -- at distance
    `tol` from the street, to within floating point. `street_connectivity` then asks whether a road
    comes WITHIN `tol` of a street, so for a footpath drawn meeting the street the answer is decided
    by whether `0.5000000000000001 <= 0.5`. Both sides read the same `STREET_TOL`, so the collision
    is structural rather than a bad calibration: MEASURED over 10 Cape Town blocks, moving the
    connectivity test by ONE CENTIMETRE (0.50 -> 0.51 m) takes street-connected footpath length from
    70.0% to 87.5% and fully-connected blocks from 4/10 to 8/10 -- and nothing moves again out to
    5 m, the signature of a threshold artifact rather than a tolerance curve. That artifact is what
    made `osm_footpaths` look like 93% floating road
    (`specs/2026-07-30-road-first-mesh-design.md`, D3).

    Reconnecting at the source fixes it for every consumer and makes the result independent of `tol`
    -- which matters because `tol` cannot simply be shrunk below the access seam: it also sets how
    far off the kblock outline an OSM way may run and still count as the same feature, and the
    census sweeps it upward for exactly that reason.

    A footpath drawn meeting the street therefore keeps meeting it. One drawn RUNNING ALONG the
    street still loses its whole overlapping length, which is what the subtraction is for; the
    connector that replaces a cut end is at most `tol` long and lies inside the street corridor,
    where it displaces nothing.
    """
    if streets.is_empty:
        return line
    coords = list(line.coords)
    ends = []
    for pos in (0, -1):
        end = Point(coords[pos])
        d = end.distance(streets)
        # `0 <` excludes an end already touching the street, whose connector would be zero-length.
        ends.append(nearest_points(end, streets)[1] if 0.0 < d <= reach else None)
    head, tail = ends
    if head is not None:
        coords.insert(0, (head.x, head.y))
    if tail is not None:
        coords.append((tail.x, tail.y))
    return LineString(coords)


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
    perimeter/inter-block streets that are already egress. Ends that the subtraction cut are
    reconnected to the street (`_reach_street`), so what comes back reaches its egress the way the
    mapped path does.

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
    # A cut end sits at distance `tol` from the street by construction; the slack only absorbs the
    # double round-off that decides `0.5000000000000001 <= 0.5`, and is far too small to invent a
    # connection that the subtraction did not sever.
    reach = tol * (1.0 + 1e-6)
    return gpd.GeoDataFrame(
        # `mask` already dropped everything but LineStrings; a GeoSeries element type cannot say so.
        geometry=[_reach_street(cast(LineString, g), streets, reach) for g in kept], crs=crs)


def block_bbox_wgs84(block: Block) -> tuple[float, float, float, float]:
    """The block's bounds in EPSG:4326, (min_lon, min_lat, max_lon, max_lat) -- what a
    `FootpathSource` is queried with."""
    b = gpd.GeoSeries([block.boundary], crs=block.crs).to_crs(4326).total_bounds
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


def interior_footpaths(lines: gpd.GeoDataFrame, block: Block) -> gpd.GeoDataFrame:
    """`interior_desire_lines` against the block's own boundary and streets."""
    return interior_desire_lines(
        lines, block.boundary, unary_union(list(block.streets.geometry)), block.crs)


def block_footpaths(source: FootpathSource, block: Block) -> gpd.GeoDataFrame:
    """The block's share of the mapped network: every footpath in its bbox, clipped to it with the
    street corridor subtracted. Widthless -- an alignment, not a road."""
    return interior_footpaths(source.footpaths(block_bbox_wgs84(block), block.crs), block)


@dataclass
class OsmFootpathsReblocker:
    source: FootpathSource
    # Total width of the corridor each imported footpath is treated as.
    road_width_m: float

    @property
    def identity(self) -> Hashable | None:
        return config_identity(self)

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior  # accepted for Method conformance; routing is block-only
        roads = block_footpaths(self.source, block)
        # The label names render files, so it tells configurations apart: the source identity is
        # hashed (filesystem-clean), road_width_m stays literal for legibility.
        source = self.source.identity
        src = "" if source is None else f":{hashlib.sha256(str(source).encode()).hexdigest()[:8]}"
        return Proposal(
            block_id=block.block_id, crs=block.crs, edges=None,
            roads=with_width(roads, self.road_width_m),
            proposal_id=f"osm_footpaths:w{self.road_width_m:g}{src}", method="osm_footpaths",
            params={"segments": len(roads), "road_width_m": self.road_width_m})
