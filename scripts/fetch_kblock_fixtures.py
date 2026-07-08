#!/usr/bin/env python
"""Fetch + prep the committed kblock test fixtures (DJI + Cape Town).

Reproducible, parameterized prep script (CI never runs it; `KblockSource`'s tests only
consume its committed output under ``tests/data/kblock/``). From scratch it:

1. Downloads the per-country blocks geodata (``{ISO3}_geodata.parquet``) from the
   version-qualified kblock Dataverse dataset (``doi:10.7910/DVN/DQY54U``, v2.0).
2. Extracts ``buildings_points_DJI.parquet`` (OSM building points for Djibouti) out of
   the ~2.6 GB kblock reprex ``sample-data.zip`` via HTTP-range reads, without
   downloading the whole archive.
3. Downloads + filters the Google Open Buildings V3 tile covering Cape Town to a bbox +
   confidence threshold, emitting centroid points.
4. Selects a small, deterministic, density-dense subset of blocks per city (+ two
   pinned validation blocks, force-included) and writes the four committed fixture
   parquets, printing each one's SHA256.

All three fetches are check-if-exists-else-download against ``--raw-dir``: point it at
a directory that already has the four raw files (see ``RAW_FILENAMES`` below) to skip
the network entirely; any file missing from ``--raw-dir`` is fetched into it.

Usage:
    pixi run python scripts/fetch_kblock_fixtures.py --out tests/data/kblock \\
        --raw-dir /path/to/already-downloaded/raw/parquets
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, shape

# ---------------------------------------------------------------------------
# Source endpoints (verified live against the real services while writing this
# script -- see tests/data/kblock/PROVENANCE.md for the retrieval date + checksums).
# ---------------------------------------------------------------------------

USER_AGENT = "reblock-fetch-kblock-fixtures/1.0"

DATAVERSE_DOI = "doi:10.7910/DVN/DQY54U"
DATAVERSE_VERSION = "2.0"
DATAVERSE_META_URL = "https://dataverse.harvard.edu/api/datasets/:persistentId/"
DATAVERSE_ACCESS_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"

# kblock's own "minimal reproducible example" archive (README.md "Download data
# here"). No DOI -- a bare CloudFront/S3 URL -- so this script IS the durable,
# regenerable record of where DJI's OSM building points came from.
SAMPLE_DATA_ZIP_URL = "https://dsbprylw7ncuq.cloudfront.net/_sampledata/sample-data.zip"
DJI_BUILDINGS_ZIP_ENTRY_SUFFIX = "buildings/osm/points/buildings_points_DJI.parquet"

OPEN_BUILDINGS_TILES_URL = (
    "https://openbuildings-public-dot-gweb-research.uw.r.appspot.com/public/tiles.geojson"
)
# The tile index only lists each S2 tile's polygon CSV; Open Buildings V3 publishes a
# point-centroid CSV under the same tile id in a parallel prefix.
OB_POLYGON_PREFIX = "polygons_s2_level_4_gzip"
OB_POINT_PREFIX = "points_s2_level_4_gzip"

CT_BBOX = (18.3, -34.4, 19.0, -33.5)  # lon_min, lat_min, lon_max, lat_max
OB_MIN_CONFIDENCE = 0.7

# Pinned validation blocks (Task 4 will pin exact peel-k / geometric-access values on
# these): a dense *interior* DJI block (verified this session -- see PROVENANCE.md for
# the coastal/GADM-edge check) and the Cape Town block spiked end-to-end this session.
PINNED_DJI_BLOCK = "DJI.1_2_602"
PINNED_CAPETOWN_BLOCK = "ZAF.9.3.1_1_44882"

# Fixture-selection predicate (explicit, deterministic -- see PROVENANCE.md).
MIN_DENSITY_PER_HA = 10.0
MAX_AREA_KM2 = 0.5
DENSEST_CAP = 300

RAW_FILENAMES = {
    "dji_blocks": "DJI_geodata.parquet",
    "zaf_blocks": "ZAF_geodata.parquet",
    "dji_buildings": "buildings_points_DJI.parquet",
    "capetown_buildings": "ob_capetown_points.parquet",
}


def _request(
    url: str, *, headers: dict[str, str] | None = None, method: str | None = None
) -> urllib.request.Request:
    merged = {"User-Agent": USER_AGENT}
    merged.update(headers or {})
    return urllib.request.Request(url, headers=merged, method=method)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_to(url: str, out_path: Path, *, timeout: int = 300) -> None:
    """Stream `url` to `out_path`, atomically (write to a .part file then rename)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with urllib.request.urlopen(_request(url), timeout=timeout) as resp, open(tmp_path, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    os.replace(tmp_path, out_path)


# ---------------------------------------------------------------------------
# 1. Dataverse: per-country blocks geodata (version-qualified -- bare numeric
#    Dataverse file ids drift across dataset versions, so we always resolve the id
#    from the version-qualified dataset metadata rather than hardcoding it).
# ---------------------------------------------------------------------------


def _dataverse_file_id(iso3: str, *, version: str) -> int:
    filename = f"{iso3}_geodata.parquet"
    url = f"{DATAVERSE_META_URL}?persistentId={DATAVERSE_DOI}&version={version}"
    with urllib.request.urlopen(_request(url), timeout=60) as resp:
        meta = json.load(resp)
    files: list[dict[str, Any]] = meta["data"]["latestVersion"]["files"]
    for entry in files:
        if entry["dataFile"]["filename"] == filename:
            return int(entry["dataFile"]["id"])
    raise ValueError(f"{filename!r} not found in Dataverse dataset {DATAVERSE_DOI} v{version}")


def download_dataverse_blocks(
    iso3: str, out_path: Path, *, version: str = DATAVERSE_VERSION
) -> None:
    """Download `{iso3}_geodata.parquet` from the kblock Dataverse dataset (DVN/DQY54U),
    resolving its file id from the version-qualified dataset metadata."""
    file_id = _dataverse_file_id(iso3, version=version)
    _download_to(DATAVERSE_ACCESS_URL.format(file_id=file_id), out_path)


# ---------------------------------------------------------------------------
# 2. DJI buildings: HTTP-range extraction of one entry from the ~2.6 GB
#    sample-data.zip, without downloading the whole archive.
# ---------------------------------------------------------------------------


class _HTTPRangeFile(io.RawIOBase):
    """Minimal seekable/readable file-like object backed by HTTP Range requests, so
    `zipfile.ZipFile` can read a remote zip's central directory and one entry's
    compressed bytes without downloading the whole archive."""

    def __init__(self, url: str) -> None:
        self.url = url
        with urllib.request.urlopen(_request(url, method="HEAD"), timeout=30) as resp:
            self.size = int(resp.headers["Content-Length"])
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self.size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readinto(self, b: bytearray) -> int:
        n = len(b)
        if n == 0 or self._pos >= self.size:
            return 0
        end = min(self._pos + n, self.size) - 1
        req = _request(self.url, headers={"Range": f"bytes={self._pos}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        b[: len(data)] = data
        self._pos += len(data)
        return len(data)


def extract_dji_buildings(out_path: Path, *, zip_url: str = SAMPLE_DATA_ZIP_URL) -> None:
    """Extract `buildings_points_DJI.parquet` from the kblock reprex `sample-data.zip`
    via HTTP range reads, without downloading the ~2.6 GB whole archive."""
    remote = _HTTPRangeFile(zip_url)
    with zipfile.ZipFile(remote) as zf:
        matches = [n for n in zf.namelist() if n.endswith(DJI_BUILDINGS_ZIP_ENTRY_SUFFIX)]
        if not matches:
            raise ValueError(f"no entry ending in {DJI_BUILDINGS_ZIP_ENTRY_SUFFIX!r} in {zip_url}")
        data = zf.read(matches[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    tmp_path.write_bytes(data)
    os.replace(tmp_path, out_path)


# ---------------------------------------------------------------------------
# 3. Cape Town buildings: Google Open Buildings V3, tile covering the bbox,
#    filtered to bbox + confidence, emitted as centroid points.
# ---------------------------------------------------------------------------


def _open_buildings_points_url(bbox: tuple[float, float, float, float]) -> str:
    with urllib.request.urlopen(_request(OPEN_BUILDINGS_TILES_URL), timeout=60) as resp:
        tiles = json.load(resp)
    center = Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    for feat in tiles["features"]:
        if shape(feat["geometry"]).contains(center):
            return str(feat["properties"]["tile_url"]).replace(OB_POLYGON_PREFIX, OB_POINT_PREFIX)
    raise ValueError(f"no Open Buildings tile contains bbox centroid {center}")


def download_capetown_buildings(
    bbox: tuple[float, float, float, float],
    out_path: Path,
    *,
    min_confidence: float = OB_MIN_CONFIDENCE,
) -> None:
    """Download the Open Buildings V3 points tile covering `bbox`, filter to
    `bbox` + `confidence >= min_confidence`, and write centroid points as GeoParquet
    (columns: geometry, area_in_meters, confidence)."""
    points_url = _open_buildings_points_url(bbox)
    with tempfile.TemporaryDirectory() as tmp_dir:
        gz_path = Path(tmp_dir) / "points.csv.gz"
        _download_to(points_url, gz_path, timeout=900)

        lon_min, lat_min, lon_max, lat_max = bbox
        kept: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            gz_path,
            usecols=["latitude", "longitude", "area_in_meters", "confidence"],
            chunksize=500_000,
        ):
            mask = (
                chunk["longitude"].between(lon_min, lon_max)
                & chunk["latitude"].between(lat_min, lat_max)
                & (chunk["confidence"] >= min_confidence)
            )
            if mask.any():
                kept.append(chunk[mask])
        df = (
            pd.concat(kept, ignore_index=True)
            if kept
            else pd.DataFrame(columns=["latitude", "longitude", "area_in_meters", "confidence"])
        )

    gdf = gpd.GeoDataFrame(
        df[["area_in_meters", "confidence"]],
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out_path)


# ---------------------------------------------------------------------------
# check-if-exists-else-fetch against --raw-dir
# ---------------------------------------------------------------------------


def load_blocks(iso3: str, raw_dir: Path, *, version: str = DATAVERSE_VERSION) -> gpd.GeoDataFrame:
    key = "dji_blocks" if iso3 == "DJI" else "zaf_blocks"
    path = raw_dir / RAW_FILENAMES[key]
    if path.exists():
        print(f"[cache] using existing {path}")
    else:
        print(f"[fetch] downloading {iso3}_geodata.parquet from Dataverse "
              f"({DATAVERSE_DOI} v{version})...")
        download_dataverse_blocks(iso3, path, version=version)
    return gpd.read_parquet(
        path, columns=["block_id", "k_complexity", "building_count", "block_area_m2", "geometry"])


def load_dji_buildings(raw_dir: Path) -> gpd.GeoDataFrame:
    path = raw_dir / RAW_FILENAMES["dji_buildings"]
    if path.exists():
        print(f"[cache] using existing {path}")
    else:
        print("[fetch] extracting buildings_points_DJI.parquet via HTTP-range "
              "from sample-data.zip...")
        extract_dji_buildings(path)
    return gpd.read_parquet(path, columns=["geometry"]).reset_index(drop=True)


def load_capetown_buildings(
    bbox: tuple[float, float, float, float], raw_dir: Path,
    *, min_confidence: float = OB_MIN_CONFIDENCE,
) -> gpd.GeoDataFrame:
    path = raw_dir / RAW_FILENAMES["capetown_buildings"]
    if path.exists():
        print(f"[cache] using existing {path}")
    else:
        print(f"[fetch] downloading + filtering Open Buildings tile for bbox {bbox}...")
        download_capetown_buildings(bbox, path, min_confidence=min_confidence)
    return gpd.read_parquet(path, columns=["geometry"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fixture selection: density predicate + pinned force-include
# ---------------------------------------------------------------------------


def select_dense_blocks(
    blocks: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    *,
    pinned_ids: set[str],
    min_density_per_ha: float = MIN_DENSITY_PER_HA,
    max_area_km2: float = MAX_AREA_KM2,
    cap: int = DENSEST_CAP,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Deterministic fixture-subset predicate.

    Spatial-join `buildings` into `blocks` (`sjoin(..., predicate="within")` in the
    estimated UTM CRS), keep blocks with >= `min_density_per_ha` joined
    buildings-per-hectare AND block area <= `max_area_km2`, cap to the densest `cap`
    blocks by density, then force-include every id in `pinned_ids` regardless of the
    cutoff. Returns `(kept_blocks, kept_buildings)` in the input CRS -- `kept_buildings`
    restricted to points that spatially joined into a kept block.
    """
    missing_pins = pinned_ids - set(blocks["block_id"])
    if missing_pins:
        raise ValueError(f"pinned block id(s) not found in blocks: {sorted(missing_pins)}")

    utm = blocks.estimate_utm_crs()
    blocks_utm = blocks.to_crs(utm)
    bld_utm = buildings.to_crs(utm)

    joined = gpd.sjoin(
        bld_utm, blocks_utm[["block_id", "geometry"]], predicate="within", how="inner"
    )

    area_km2 = blocks_utm.geometry.area / 1e6
    area_ha = blocks_utm.geometry.area / 1e4
    n_bld = blocks_utm["block_id"].map(joined.groupby("block_id").size()).fillna(0.0)
    density = n_bld / area_ha

    eligible = (density >= min_density_per_ha) & (area_km2 <= max_area_km2)
    ranked = blocks_utm.loc[eligible].assign(_density=density[eligible])
    ranked = ranked.sort_values("_density", ascending=False)
    kept_ids = set(ranked["block_id"].head(cap)) | pinned_ids

    kept_blocks = blocks[blocks["block_id"].isin(kept_ids)].reset_index(drop=True)
    kept_bld_positions = joined.index[joined["block_id"].isin(kept_ids)].unique()
    kept_buildings = buildings.loc[buildings.index.isin(kept_bld_positions)].reset_index(drop=True)
    return kept_blocks, kept_buildings


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("tests/data/kblock"),
                        help="directory to write the four fixture parquets into")
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="directory holding already-downloaded raw source files "
                             f"({', '.join(RAW_FILENAMES.values())}); any file missing "
                             "from it is fetched into it. Defaults to a temp cache dir.")
    parser.add_argument("--dataverse-version", default=DATAVERSE_VERSION)
    parser.add_argument("--ct-bbox", type=float, nargs=4, default=list(CT_BBOX),
                        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    parser.add_argument("--ob-min-confidence", type=float, default=OB_MIN_CONFIDENCE)
    parser.add_argument("--min-density-per-ha", type=float, default=MIN_DENSITY_PER_HA)
    parser.add_argument("--max-area-km2", type=float, default=MAX_AREA_KM2)
    parser.add_argument("--cap", type=int, default=DENSEST_CAP)
    parser.add_argument("--pinned-dji", default=PINNED_DJI_BLOCK)
    parser.add_argument("--pinned-capetown", default=PINNED_CAPETOWN_BLOCK)
    args = parser.parse_args()

    raw_dir = (
        args.raw_dir if args.raw_dir is not None
        else Path(tempfile.gettempdir()) / "kblock_fixtures_raw"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    bbox = (args.ct_bbox[0], args.ct_bbox[1], args.ct_bbox[2], args.ct_bbox[3])

    print(f"raw source cache: {raw_dir}")

    print("\n=== DJI ===")
    dji_blocks = load_blocks("DJI", raw_dir, version=args.dataverse_version)
    dji_bld = load_dji_buildings(raw_dir)
    print(f"DJI: {len(dji_blocks)} blocks, {len(dji_bld)} building points")
    dji_kept_blocks, dji_kept_bld = select_dense_blocks(
        dji_blocks, dji_bld, pinned_ids={args.pinned_dji},
        min_density_per_ha=args.min_density_per_ha, max_area_km2=args.max_area_km2, cap=args.cap,
    )
    print(f"DJI: kept {len(dji_kept_blocks)} blocks, {len(dji_kept_bld)} building points "
          f"(pinned block {args.pinned_dji} force-included)")

    print("\n=== Cape Town ===")
    zaf_blocks = load_blocks("ZAF", raw_dir, version=args.dataverse_version)
    ct_blocks = zaf_blocks.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].reset_index(drop=True)
    print(f"Cape Town bbox filter: {len(zaf_blocks)} ZAF blocks -> {len(ct_blocks)} in bbox {bbox}")
    ct_bld = load_capetown_buildings(bbox, raw_dir, min_confidence=args.ob_min_confidence)
    print(f"Cape Town: {len(ct_bld)} building points (confidence >= {args.ob_min_confidence})")
    ct_kept_blocks, ct_kept_bld = select_dense_blocks(
        ct_blocks, ct_bld, pinned_ids={args.pinned_capetown},
        min_density_per_ha=args.min_density_per_ha, max_area_km2=args.max_area_km2, cap=args.cap,
    )
    print(f"Cape Town: kept {len(ct_kept_blocks)} blocks, {len(ct_kept_bld)} building points "
          f"(pinned block {args.pinned_capetown} force-included)")

    outputs = {
        "blocks_dji_sample.parquet": dji_kept_blocks,
        "buildings_dji_sample.parquet": dji_kept_bld,
        "blocks_capetown_sample.parquet": ct_kept_blocks,
        "buildings_capetown_sample.parquet": ct_kept_bld,
    }
    print("\n=== fixtures written ===")
    for name, gdf in outputs.items():
        path = args.out / name
        gdf.to_parquet(path)
        print(f"{name}: {len(gdf)} rows, {path.stat().st_size} bytes, sha256={sha256_of(path)}")


if __name__ == "__main__":
    main()
