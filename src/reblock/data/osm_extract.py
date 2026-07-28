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
from shapely import STRtree
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
    *,
    tolerances: Sequence[float] = TOLERANCES,
) -> dict[str, object]:
    """One census row: interior segment count and length at each requested tolerance, for the
    primary footpath tags and (separately) the near-miss tags.

    `tolerances` defaults to the full diagnostic sweep. A corpus-wide run passes a single value:
    the 400-block spike found 0.5 -> 5 m moves the coverage gate by only 2.6 points while total
    interior length drops 17.8%, so the sweep informs donor-quality ranking, not the census, and
    paying for it 1.8M times over buys nothing.

    For a kblock block `streets` IS the outline, so the street corridor is `boundary.boundary`.
    """
    streets = boundary.boundary
    row: dict[str, object] = {"block_id": block_id, "boundary_length_m": float(streets.length)}
    for label, lines in (("interior", footpaths), ("near_miss", near_miss)):
        for tol in tolerances:
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
    if isinstance(tags, str):
        # A bare str IS a Sequence[str], so `tags="path"` typechecks but iterates
        # character-by-character -- silently building `highway IN ('p','a','t','h')`, which
        # matches nothing rather than raising. Reject it explicitly.
        raise TypeError(
            f"tags must be a sequence of tag strings, not a bare str ({tags!r}); pass a list "
            f"or tuple, e.g. tags=[{tags!r}]"
        )
    tags = tuple(tags)
    if not tags:
        # `highway IN ()` is invalid OGR SQL syntax -- raise our own clear message instead of
        # letting an obscure OGR/SQL error surface from inside pyogrio.
        raise ValueError("tags must not be empty")
    quoted = ", ".join("'{}'".format(t.replace("'", "''")) for t in tags)
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
    # compare=False on both caches: a DataFrame's truth value is ambiguous, so the
    # dataclass-generated __eq__ would raise (`ValueError: The truth value of a DataFrame is
    # ambiguous`) comparing two instances that have each populated their own cache -- these are
    # memoization state, not part of what identifies a PbfDesireLines.
    _cache: gpd.GeoDataFrame | None = field(default=None, init=False, repr=False, compare=False)
    # (stat_signature, digest) memo for `identity`, keyed on (st_size, st_mtime_ns) so a changed
    # file re-hashes but a repeat access on an unchanged file does not pay another full-file
    # SHA-256 (measured 0.77s/873MB -- osm_footpaths reads .identity ~3x/block, which would
    # otherwise cost ~1s/block of pure hashing against a 3.31ms/block budget).
    _digest_cache: tuple[tuple[int, int], str] | None = field(
        default=None, init=False, repr=False, compare=False)

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
        st = self.pbf_path.stat()
        signature = (st.st_size, st.st_mtime_ns)
        if self._digest_cache is None or self._digest_cache[0] != signature:
            self._digest_cache = (signature, _file_sha256(self.pbf_path))
        return ("pbf", self._digest_cache[1], tuple(self.tags))


def utm_zone_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat). 326xx north, 327xx south."""
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def assert_zone_fit(lon: float, epsg: int) -> None:
    """Raise if `lon` is more than 3.5 degrees from the zone's central meridian.

    Load-bearing: `estimate_utm_crs()` on a whole-country extent returns a single zone with NO
    error, and transverse Mercator stays conformal, so nothing crashes -- you just get a silent
    scale bias (measured +0.72% at Cape Town, +1.23% at lon 16.5, +3.46% at lon 41.9 under one
    country-wide UTM). That biases interior_length_m by 1-3%. This makes a forgotten batch loud.
    """
    zone = epsg - (32600 if epsg < 32700 else 32700)
    central = 6 * zone - 183
    if abs(lon - central) > 3.5:
        raise ValueError(
            f"longitude {lon} is outside UTM zone {zone} (central meridian {central}); "
            f"batch blocks by zone via utm_zone_epsg before projecting")


def census_rows(
    blocks: gpd.GeoDataFrame,
    footpaths: gpd.GeoDataFrame,
    near_miss: gpd.GeoDataFrame,
    epsg: int,
    *,
    tolerances: Sequence[float] = TOLERANCES,
) -> list[dict[str, object]]:
    """Census rows for one UTM batch. `blocks` is in EPSG:4326; everything is reprojected to
    `epsg` once, then each block queries an STRtree rather than re-reading the layer.

    `tolerances` is forwarded to `interiority_row` -- see there for why a corpus run passes one.
    """
    crs_m = CRS.from_epsg(epsg)
    blocks_m = blocks.to_crs(crs_m)
    fp_m = (footpaths.to_crs(crs_m) if len(footpaths)
             else footpaths.set_crs(crs_m, allow_override=True))
    nm_m = (near_miss.to_crs(crs_m) if len(near_miss)
             else near_miss.set_crs(crs_m, allow_override=True))
    fp_tree = STRtree(list(fp_m.geometry)) if len(fp_m) else None
    nm_tree = STRtree(list(nm_m.geometry)) if len(nm_m) else None

    rows: list[dict[str, object]] = []
    for block_id, geom in zip(blocks_m["block_id"], blocks_m.geometry, strict=True):
        near_fp = (fp_m.iloc[fp_tree.query(geom)] if fp_tree is not None
                   else gpd.GeoDataFrame(geometry=[], crs=crs_m))
        near_nm = (nm_m.iloc[nm_tree.query(geom)] if nm_tree is not None
                   else gpd.GeoDataFrame(geometry=[], crs=crs_m))
        row = interiority_row(str(block_id), geom, near_fp, near_nm, crs_m,
                              tolerances=tolerances)
        # The qualified filter is a building-count band, which does NOT bound block AREA: the
        # spike found 5 of 251 covered blocks carrying >5 km of "interior" footpath on 90-293
        # buildings (max 26.5 km on 258 buildings, vs a 356 m median) -- huge polygons where the
        # clip captures a whole neighbourhood. Emit area so the guard is applied downstream on
        # data rather than guessed here.
        row["area_m2"] = float(geom.area)
        rows.append(row)
    return rows
