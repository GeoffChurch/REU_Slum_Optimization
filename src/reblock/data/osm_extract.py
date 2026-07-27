"""Country-wide OSM footpath census (tier T1): per-block interior-footpath coverage over the
whole ZAF+KEN block corpus, computed from the blocks parquet + OSM linework alone -- no
building points, no Voronoi parcels, no Block. See
docs/superpowers/specs/2026-07-27-ot-retrieval-substrate-phase1-design.md.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import geopandas as gpd
import pyogrio
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from reblock.methods.osm_footpaths import interior_desire_lines

# The shipped osm_footpaths tag set, imported by conf/desire_source/_footpath_tags.yaml so the
# census and the method can never disagree about what a footpath is.
FOOTPATH_TAGS: tuple[str, ...] = (
    "path", "footway", "track", "steps", "pedestrian", "living_street")
# Tags informal paths are SOMETIMES mapped under. Counted separately and never mixed into the
# primary columns, so the cost of widening the filter is visible before anyone re-extracts.
NEAR_MISS_TAGS: tuple[str, ...] = ("service", "residential", "unclassified")
# OSM ways are digitized against different imagery than the kblock outlines, so a boundary-running
# path more than STREET_TOL off the outline reads as interior. Measured: the count gate moves only
# 2.6 points across this range while total length drops ~18%, so tolerance matters for
# donor-quality ranking, not for the coverage census -- but report both and let the data say so.
TOLERANCES: tuple[float, ...] = (0.5, 2.0, 5.0)


def interiority_row(
    block_id: str,
    boundary: BaseGeometry,
    footpaths: gpd.GeoDataFrame,
    near_miss: gpd.GeoDataFrame,
    crs: CRS,
) -> dict[str, object]:
    """One census row: interior segment count and length at every tolerance, for the primary
    footpath tags and (separately) the near-miss tags.

    For a kblock block `streets` IS the outline, so the street corridor is `boundary.boundary`.
    """
    streets = boundary.boundary
    row: dict[str, object] = {"block_id": block_id, "boundary_length_m": float(streets.length)}
    for label, lines in (("interior", footpaths), ("near_miss", near_miss)):
        for tol in TOLERANCES:
            kept = interior_desire_lines(lines, boundary, streets, crs, tol=tol)
            row[f"n_{label}_segments_{tol}"] = int(len(kept))
            row[f"{label}_length_m_{tol}"] = (
                float(kept.geometry.length.sum()) if len(kept) else 0.0)
    return row


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pbf_lines(pbf_path: Path, tags: Sequence[str] = FOOTPATH_TAGS) -> gpd.GeoDataFrame:
    """Every `highway` way of the given tag classes in a .osm.pbf, as EPSG:4326 LineStrings.

    Filters OGR-side via `where` so a country extract does not materialize every South African
    `highway=track` in Python. NOTE: this reduces what crosses into pandas; it does NOT avoid the
    GDAL OSM driver building its multi-GB temp SQLite database, which happens regardless.

    The census must call this ONCE per UTM batch and query an STRtree per block -- not once per
    block. `DesireLineSource.desire_lines` is a per-bbox API and there are 1.81M blocks.
    """
    quoted = ", ".join(f"'{t}'" for t in tags)
    return cast(gpd.GeoDataFrame, pyogrio.read_dataframe(
        pbf_path, layer="lines", where=f"highway IN ({quoted})", use_arrow=True))


@dataclass
class PbfDesireLines:
    """A DesireLineSource backed by a local Geofabrik .osm.pbf extract.

    A second implementation alongside OSMDesireLines, not a replacement: the operating ranges are
    disjoint (a PBF covers its extract; Overpass covers any bbox). At 1.81M blocks a bulk extract
    is the only workable option -- one 0.25-degree Overpass tile is ~29 MB / 40k ways / 7 s, and
    ZAF+KEN is ~4,598 such tiles against Overpass's ~1 GB/day fair-use policy, versus 766 MB of
    PBF once.

    `identity` is stable (unlike OSMDesireLines' None-when-live), so osm_footpaths becomes
    cacheable when driven by this source.
    """

    pbf_path: Path
    tags: Sequence[str] = FOOTPATH_TAGS
    _cache: gpd.GeoDataFrame | None = field(default=None, init=False, repr=False)

    def desire_lines(
        self, bbox_wgs84: tuple[float, float, float, float], crs: CRS
    ) -> gpd.GeoDataFrame:
        if self._cache is None:
            self._cache = read_pbf_lines(self.pbf_path, self.tags)
        minx, miny, maxx, maxy = bbox_wgs84
        window = self._cache.cx[minx:maxx, miny:maxy]
        return window.to_crs(crs)

    @property
    def identity(self) -> tuple[str, str, tuple[str, ...]]:
        return ("pbf", _file_sha256(self.pbf_path), tuple(self.tags))
