"""Building geometry for the blocks a run PROPOSES on, as an injected strategy.

A run reads buildings in two phases that want different data. Screening, building counts and
region-growth depth run over the WHOLE corpus and need only points: depth comes from parcels, which
are the Voronoi of centroids and so identical at every tier. Proposing runs over the handful of
member blocks the screen chose, and wants the best geometry there is. `KblockSource.buildings_path`
keeps feeding the first phase; a `BuildingSource` feeds the second.

`FootprintTiles` is what makes real outlines affordable. Open Buildings ships each S2 level-4 cell
as ONE gzipped CSV, and gzip cannot be range-read, so the unit of cost is the TILE, not the block:
a single block's footprints cost the whole ~320 MB tile. So it fetches exactly the tiles covering
the blocks asked for, once, and caches each as spatially-sorted parquet. A region costs the tiles
it touches -- usually one, since growth is spatially local -- and every later block in a cached
tile is free. For ZAF+KEN that is ~640 MB for the example regions against 14.09 GB for the corpus.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from scripts.fetch_kblock_fixtures import (
    OB_FLOAT_PRECISION,
    OB_MIN_CONFIDENCE,
    OPEN_BUILDINGS_TILES_URL,
    _download_to,
    _request,
)

DEFAULT_FOOTPRINT_CACHE = Path.home() / ".cache" / "reblock" / "footprints"
# Rows per row group in a cached tile. Rows are sorted spatially before writing, so a row group
# covers a small patch and `read_parquet(bbox=...)` skips the rest of the tile.
_ROW_GROUP = 20_000
_CHUNK = 400_000          # CSV rows parsed at a time: bounds peak memory on a multi-GB tile
# The PARSED tile schema. It is part of the cache path, so changing what a cached tile holds can
# never serve a stale one: v1 lacked the published lat/lon and the CSV row, and would have been
# reused by a check that only looked for SOURCE_SHA256. Bump it whenever `_tile` writes new columns.
_FORMAT = 2
_SOURCE_SHA = "SOURCE_SHA256"
_OB_ROW = "ob_row"          # original CSV row: storage is sorted spatially, output is not


def tiles_for(shortlist: gpd.GeoDataFrame, tiles: gpd.GeoDataFrame) -> list[str]:
    """Open Buildings tile URLs whose S2 cell intersects any shortlist block.

    Returns the manifest's `tile_url` as-is, and those are the POLYGON tiles; the points variant is
    the same path under a different prefix. Measured: tiles.geojson has 333 features, 20 of which
    cover ZAF+KEN (3.78 GB gzipped as points; the polygon variants are 14.09 GB). A single-centroid
    tile lookup is correct only for a bbox smaller than one cell.
    """
    joined = gpd.sjoin(tiles.to_crs(shortlist.crs or "EPSG:4326"), shortlist,
                       how="inner", predicate="intersects")
    return sorted(set(joined["tile_url"]))


class BuildingFrame(NamedTuple):
    """What a `BuildingSource` hands back. Named, not a bare tuple: positions that carry meaning
    are exactly where inserting a field leaves `t[1]` valid, the right type, and wrong."""

    buildings: gpd.GeoDataFrame     # every column the data carries, in the DATA's own CRS
    # one PUBLISHED point per row, in the same order; the parcels grow from these
    anchors: gpd.GeoSeries
    read_from: list[Path]           # the files it came from, folded into the block content hash


@runtime_checkable
class BuildingSource(Protocol):
    def for_blocks(self, blocks: gpd.GeoDataFrame) -> BuildingFrame:
        """Buildings covering these block polygons, their anchor points, and the files read.

        The ANCHOR is Open Buildings' published point, never a centroid computed here: measured on
        ZAF.9.3.1_1_40972, the polygon CSV's published lat/lon equal the point tier's in 263/263
        buildings to < 1e-6 m, while polygon centroids taken in UTM are off by 0.7 mm median and
        2.7 mm max. Tiny -- yet all 263 parcels changed and 2.1% of parcel area disagreed, up to
        87 m^2 in one parcel, because near-cocircular sites flip the Voronoi's TOPOLOGY under a
        sub-millimetre nudge. A tier comparison would then confound the building model with a
        different tessellation. Anchored on the published point, parcels are bit-identical.

        The files matter too: a cache keyed on data it did not read cannot notice it changing.
        """


@dataclass(frozen=True)
class ParquetBuildings:
    """One buildings parquet, read whole: today's behaviour, and the default.

    Returns the frame in its native CRS. Reprojecting here as well as in the caller would move
    coordinates at the floating-point level, which moves the Voronoi -- enough to break the exact
    reproduction every published figure currently has.
    """

    path: Path

    def for_blocks(self, blocks: gpd.GeoDataFrame) -> BuildingFrame:
        bld = gpd.read_parquet(self.path)
        return BuildingFrame(bld, bld.geometry, [self.path])    # a point IS its published anchor


def _fetch(url: str, dest: Path) -> None:
    _download_to(url, dest, timeout=1800)


@dataclass(frozen=True)
class FootprintTiles:
    """Real Open Buildings OUTLINES, provisioned per S2 tile for exactly the blocks asked for.

    `fetch` is injected so tests never touch the network; `manifest` likewise may be given, else
    it is downloaded once and cached beside the tiles.
    """

    cache_dir: Path = DEFAULT_FOOTPRINT_CACHE
    min_confidence: float = OB_MIN_CONFIDENCE
    fetch: Callable[[str, Path], None] = _fetch

    def for_blocks(self, blocks: gpd.GeoDataFrame) -> BuildingFrame:
        wgs = blocks.to_crs(4326)
        urls = tiles_for(wgs, self._manifest())
        if not urls:
            raise ValueError(
                f"no Open Buildings tile covers these {len(blocks)} block(s) "
                f"(bounds {tuple(np.round(wgs.total_bounds, 4))})")
        bounds = tuple(float(v) for v in wgs.total_bounds)
        parts: list[gpd.GeoDataFrame] = []
        shas: list[Path] = []
        for url in urls:
            tile = self._tile(url)
            shas.append(tile / _SOURCE_SHA)
            for part in sorted(tile.glob("part-*.parquet")):
                parts.append(gpd.read_parquet(part, bbox=bounds))
        got = [p for p in parts if len(p)]
        if not got:
            empty = gpd.GeoDataFrame(geometry=[], crs=4326)
            return BuildingFrame(empty, empty.geometry, shas)
        # Back to CSV order. NOT for the parcels: `_voronoi_parcels` is order-independent (verified
        # -- permuting its input gives identical parcels in identical order; GEOS orders cells by
        # geometry). It is for the BUILDING ROWS: stored sorted by lat/lon for bbox reads, they
        # would otherwise come back in a different order from the point tier's, so any per-building
        # comparison across tiers would pair the wrong buildings -- which is exactly how a bogus
        # 62 m "anchor shift" was once measured here, row by row, between two orders.
        bld = pd.concat(got, ignore_index=True).sort_values(_OB_ROW, kind="stable")
        bld = gpd.GeoDataFrame(bld.reset_index(drop=True), geometry="geometry", crs=4326)
        anchors = gpd.GeoSeries(gpd.points_from_xy(bld["longitude"], bld["latitude"]), crs=4326)
        return BuildingFrame(bld, anchors, shas)

    def _manifest(self) -> gpd.GeoDataFrame:
        path = self.cache_dir / "tiles.geojson"
        if not path.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(_request(OPEN_BUILDINGS_TILES_URL), timeout=120) as r:
                path.write_text(json.dumps(json.load(r)))
        return gpd.read_file(path)

    def _tile(self, url: str) -> Path:
        """The cached, parsed tile directory for `url`, building it on first use.

        Written to a sibling temp dir and renamed into place only when complete, so a crash
        mid-parse can never leave a half-tile that looks cached.
        """
        token = url.rsplit("/", 1)[-1].split("_", 1)[0]          # "1dd_buildings.csv.gz" -> "1dd"
        root = self.cache_dir / f"v{_FORMAT}"
        out = root / token
        if (out / _SOURCE_SHA).exists():
            return out
        tmp = root / f".{token}.partial"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        gz = tmp / "tile.csv.gz"
        self.fetch(url, gz)
        with gzip.open(gz, "rt") as fh:
            for k, chunk in enumerate(pd.read_csv(
                    fh, usecols=["latitude", "longitude", "area_in_meters", "confidence",
                                 "geometry"], chunksize=_CHUNK,
                    float_precision=OB_FLOAT_PRECISION)):   # MUST match the point tier's read
                chunk = chunk[chunk["confidence"] >= self.min_confidence]
                if chunk.empty:
                    continue
                # Sort spatially so each row group covers a small patch and bbox reads skip the
                # rest; unsorted, every row group would span the whole tile and skip nothing.
                chunk = chunk.assign(**{_OB_ROW: chunk.index.to_numpy()})   # the CSV row
                chunk = chunk.sort_values(["latitude", "longitude"])
                gdf = gpd.GeoDataFrame(
                    chunk[["latitude", "longitude", "area_in_meters", "confidence",
                           _OB_ROW]].reset_index(drop=True),
                    geometry=shapely.from_wkt(chunk["geometry"].to_numpy()), crs=4326)
                gdf.to_parquet(tmp / f"part-{k:04d}.parquet", row_group_size=_ROW_GROUP,
                               write_covering_bbox=True)
        # Content identity of the SOURCE download, recorded once. Hashing the parquet parts on
        # every region() call instead would re-read hundreds of MB just to build a cache key.
        (tmp / _SOURCE_SHA).write_text(hashlib.sha256(gz.read_bytes()).hexdigest())
        gz.unlink()
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)
        return out
