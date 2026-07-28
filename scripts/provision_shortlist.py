"""Provision Open Buildings points for the census shortlist (Phase 1, unit 1b).

Chains the pieces Task 7 shipped but never wired together: census -> shortlist -> `tiles_for` ->
per-tile download -> `filter_to_shortlist` -> one blocks parquet + one buildings parquet that
`KblockSource` can build real `Block`s from.

Why this is the critical path: no building points means no Voronoi parcels, which means
`Block.__post_init__` raises, which means no permeability, no GW features and no clearance
baseline. Every downstream question is blocked on it.

The shortlist is the 2026-07-28 census's qualified band, and DELIBERATELY NOTHING ELSE.

An earlier version gated it on a density floor as well, justified by "cuts the download 3.1x".
That was wrong: the download is TILE-granular (S2 level-4 cells), and the gated and ungated
shortlists need 18 and 20 tiles respectively. The floor cut the blocks RETAINED, not the bytes
fetched. Its substance is still right -- a 2,000 km2 polygon holding 258 buildings is not a
settlement and would be a poor donor -- but `area_m2` is already a census column, so that belongs
at retrieval time where it is free and reversible. Provisioning is a ONE-WAY DOOR: a block left
un-provisioned cannot be reconsidered without another multi-GB fetch, so it should filter as
little as possible. Same reasoning rules out a geographic filter, which was separately measured to
be uninformative (geographic distance is uncorrelated with GW distance).

Two caches, and the split matters. `ob_tiles_raw/` keeps each tile's raw .csv.gz -- the expensive
step, minutes and ~250 MB apiece -- and `ob_tiles/` keeps the filtered parquet, the cheap step.
Caching only the filtered result (which the first version did) means every change to the shortlist
costs another 4 GB fetch, which is exactly the coupling that made the density floor feel
irreversible. With the raw tiles kept, `--refilter` re-derives everything for free.

    pixi run python -m scripts.provision_shortlist --dry-run     # tiles + sizes, no download
    pixi run python -m scripts.provision_shortlist
"""
from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import shapely

from reblock.data.provision import filter_to_shortlist, tiles_for
from scripts.fetch_kblock_fixtures import (
    OB_MIN_CONFIDENCE,
    OB_POINT_PREFIX,
    OB_POLYGON_PREFIX,
    OPEN_BUILDINGS_TILES_URL,
    _download_to,
    _request,
)

CACHE = Path.home() / ".cache" / "reblock"
ISOS = ("ZAF", "KEN")
BLOCK_COLS = ["block_id", "k_complexity", "building_count", "block_area_m2", "geometry"]
CHUNK_ROWS = 500_000


def shortlist_ids(min_density: float, min_k: int, min_buildings: int,
                  max_buildings: int) -> set[str]:
    """Block ids clearing the qualified band and the density floor, from the census output."""
    ids: set[str] = set()
    for iso in ISOS:
        path = CACHE / f"osm_coverage_{iso}.parquet"
        if not path.exists():
            raise SystemExit(f"missing {path} -- run `python -m scripts.osm_census --iso {iso}`")
        df = pd.read_parquet(path)
        df = df[~df["census_failed"]]
        density = df["building_count"] / (df["area_m2"] / 1e6)
        keep = (df["building_count"].between(min_buildings, max_buildings)
                & (df["k_complexity"] >= min_k)
                & (density >= min_density))
        ids |= set(df.loc[keep, "block_id"].astype(str))
    return ids


def shortlist_blocks(ids: set[str]) -> gpd.GeoDataFrame:
    """The shortlist's block polygons, streamed out of the country parquets.

    Same GeoParquet-1.0 decode as the census: geometry is a plain Arrow binary field whose `geo`
    metadata is file-level, so `GeoDataFrame.from_arrow` cannot see it.
    """
    frames: list[gpd.GeoDataFrame] = []
    for iso in ISOS:
        pf = pq.ParquetFile(CACHE / f"{iso}_geodata.parquet")
        for batch in pf.iter_batches(batch_size=50_000, columns=BLOCK_COLS):
            frame = batch.to_pandas()
            frame["block_id"] = frame["block_id"].astype(str)
            hit = frame[frame["block_id"].isin(ids)]
            if hit.empty:
                continue
            frames.append(gpd.GeoDataFrame(
                hit.drop(columns=["geometry"]),
                geometry=shapely.from_wkb(hit["geometry"]), crs=4326))
    if not frames:
        raise SystemExit("no shortlist blocks located in the country parquets")
    return cast(gpd.GeoDataFrame, pd.concat(frames, ignore_index=True))


def point_tiles() -> gpd.GeoDataFrame:
    """The Open Buildings tile index, with `tile_url` rewritten to the POINT-centroid prefix.

    The published index lists each S2 level-4 cell's polygon CSV; V3 publishes point centroids
    under the same tile id in a parallel prefix, and points are ~4x smaller (3.78 GB vs 14.09 GB
    for ZAF+KEN).
    """
    with urllib.request.urlopen(_request(OPEN_BUILDINGS_TILES_URL), timeout=120) as resp:
        tiles = cast(gpd.GeoDataFrame, gpd.read_file(resp))
    tiles["tile_url"] = tiles["tile_url"].str.replace(OB_POLYGON_PREFIX, OB_POINT_PREFIX)
    return tiles


def _tile_name(url: str) -> str:
    return url.rsplit("/", 1)[-1].replace(".csv.gz", "")


def ensure_raw_tile(url: str, raw_dir: Path) -> Path:
    """The tile's raw .csv.gz, downloaded once and KEPT.

    Caching the raw download rather than only the filtered result is the whole point: the download
    is the expensive step (minutes, ~250 MB average) and the shortlist filter is the cheap one
    (seconds), so discarding the raw file would make every change to the shortlist definition cost
    another 4 GB fetch. That is precisely the coupling that turned the density floor into a
    one-way door. `_download_to` already writes via a .part file and renames, so a killed download
    leaves no half-file behind.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / f"{_tile_name(url)}.csv.gz"
    if not raw.exists():
        _download_to(url, raw, timeout=3600)
    return raw


def filter_tile(raw: Path, shortlist: gpd.GeoDataFrame, out_path: Path,
                min_confidence: float) -> int:
    """Keep confident points inside a shortlist block; write a parquet.

    Filtering happens per CSV chunk rather than after concatenation: a single tile holds tens of
    millions of rows and the shortlist retains a fraction, so materializing the whole tile first
    would cost gigabytes of RAM for data that is about to be discarded.
    """
    kept: list[gpd.GeoDataFrame] = []
    for chunk in pd.read_csv(
        raw, usecols=["latitude", "longitude", "area_in_meters", "confidence"],
        chunksize=CHUNK_ROWS,
    ):
        chunk = chunk[chunk["confidence"] >= min_confidence]
        if chunk.empty:
            continue
        pts = gpd.GeoDataFrame(
            chunk[["area_in_meters", "confidence"]],
            geometry=gpd.points_from_xy(chunk["longitude"], chunk["latitude"]),
            crs=4326)
        inside = filter_to_shortlist(pts, shortlist)
        if len(inside):
            kept.append(inside[["area_in_meters", "confidence", "geometry"]])

    out = (cast(gpd.GeoDataFrame, pd.concat(kept, ignore_index=True)) if kept
           else gpd.GeoDataFrame({"area_in_meters": [], "confidence": []},
                                 geometry=[], crs=4326))
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    out.to_parquet(tmp_path)
    os.replace(tmp_path, out_path)
    return len(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--min-density", type=float, default=0.0,
        help="buildings/km2 floor. Default 0 (OFF): provisioning is a one-way door, and the "
             "density filter is a free, reversible predicate at retrieval time because area_m2 "
             "is already a census column. See the module docstring.")
    ap.add_argument("--min-k", type=int, default=4)
    ap.add_argument("--min-buildings", type=int, default=60)
    ap.add_argument("--max-buildings", type=int, default=300)
    ap.add_argument("--min-confidence", type=float, default=OB_MIN_CONFIDENCE)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the shortlist and the tiles it needs, download nothing")
    ap.add_argument("--refilter", action="store_true",
                    help="re-derive every tile's parquet from the cached raw .csv.gz (use after "
                         "changing the shortlist definition; downloads nothing already cached)")
    args = ap.parse_args()

    ids = shortlist_ids(args.min_density, args.min_k, args.min_buildings, args.max_buildings)
    print(f"shortlist: {len(ids):,} block ids from the census", flush=True)
    blocks = shortlist_blocks(ids)
    print(f"located {len(blocks):,} block polygons", flush=True)

    tiles = point_tiles()
    urls = tiles_for(blocks, tiles)
    print(f"tiles covering the shortlist: {len(urls)} of {len(tiles)}", flush=True)
    for u in urls:
        print(f"  {_tile_name(u)}")
    if args.dry_run:
        return

    blocks_out = CACHE / "blocks_shortlist.parquet"
    blocks.to_parquet(blocks_out)
    print(f"\nwrote {blocks_out}", flush=True)

    tile_dir = CACHE / "ob_tiles"
    raw_dir = CACHE / "ob_tiles_raw"
    tile_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, url in enumerate(urls, 1):
        name = _tile_name(url)
        tile_path = tile_dir / f"{name}.parquet"
        if tile_path.exists() and not args.refilter:
            n = len(pd.read_parquet(tile_path))
            print(f"[{i}/{len(urls)}] {name}: cached, {n:,} points", flush=True)
        else:
            raw = raw_dir / f"{name}.csv.gz"
            verb = "filtering (raw cached)" if raw.exists() else "downloading"
            print(f"[{i}/{len(urls)}] {name}: {verb}...", flush=True)
            raw = ensure_raw_tile(url, raw_dir)
            n = filter_tile(raw, blocks, tile_path, args.min_confidence)
            print(f"[{i}/{len(urls)}] {name}: {n:,} points kept", flush=True)
        total += n

    # gpd.read_parquet, NOT pd.read_parquet: the pandas reader hands back the geometry column as
    # raw WKB bytes, so the concatenated frame writes out with no `geo` metadata and the artifact
    # is unreadable by every geo consumer downstream -- including KblockSource, which is the only
    # reason this file exists.
    parts = [gpd.read_parquet(p) for p in sorted(tile_dir.glob("*.parquet"))]
    merged = cast(gpd.GeoDataFrame, pd.concat([p for p in parts if len(p)], ignore_index=True))
    buildings_out = CACHE / "buildings_shortlist.parquet"
    merged.to_parquet(buildings_out)
    print(f"\nwrote {buildings_out}  ({len(merged):,} points, {total:,} across tiles)")
    print(f"blocks: {blocks_out}  ({len(blocks):,})")


if __name__ == "__main__":
    main()
