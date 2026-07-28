"""Country-wide OSM footpath census driver (Phase 1, unit 1a).

Streams the ZAF/KEN blocks parquet, batches blocks by UTM zone, reads the country footpath layer
ONCE per batch, and writes one row per block to ~/.cache/reblock/osm_coverage_{iso}.parquet.

Budget, measured: 3.31 ms/block for clip + corridor difference + filter, so ~1.67 single-core
hours per tolerance over 1.81M blocks -- about 5 h for the 0.5/2/5 m sweep and ~10 h once the
near-miss tag set is included. Use --limit for a smoke run first.

Usage (module form -- `python scripts/osm_census.py` fails at import once anything imports
transitively through `reblock.data.provision`, which needs the repo root on `sys.path`; see
`scripts/pair_matrix.py` for the same convention):
    pixi run python -m scripts.osm_census --iso ZAF --limit 5000
    pixi run python -m scripts.osm_census --iso ZAF
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from geopandas import GeoDataFrame

from reblock.data.osm_extract import (
    FOOTPATH_TAGS,
    NEAR_MISS_TAGS,
    assert_zone_fit,
    census_rows,
    read_pbf_lines,
    utm_zone_epsg,
)

CACHE = Path.home() / ".cache" / "reblock"
PBF = {"ZAF": "south-africa-latest.osm.pbf", "KEN": "kenya-latest.osm.pbf"}


def _decode_batch(batch: pa.RecordBatch) -> GeoDataFrame:
    """Decode one Arrow batch of the country blocks parquet into a GeoDataFrame.

    ~/.cache/reblock/{ZAF,KEN}_geodata.parquet are GeoParquet 1.0 files: geometry is a plain
    `binary` Arrow field, and the `geo` metadata describing its encoding/CRS lives in the
    file-level key-value store -- which `pq.ParquetFile.iter_batches` does not turn into a
    GeoArrow extension type on the field. `gpd.GeoDataFrame.from_arrow(batch)` looks for that
    extension type, not the file-level metadata, so it raises `ValueError: No geometry column
    found in the Arrow table` (passing `geometry="geometry"` fails identically -- the lookup is
    by Arrow extension type, not column name). Decode the WKB column directly instead; both
    country files are EPSG:4326 (per their `geo` metadata).
    """
    df = batch.to_pandas()
    return GeoDataFrame(df, geometry=shapely.from_wkb(df["geometry"]), crs=4326)


def _read_footpath_and_near_miss_lines(pbf_path: Path) -> tuple[GeoDataFrame, GeoDataFrame]:
    """One read over the union of FOOTPATH_TAGS + NEAR_MISS_TAGS, split in pandas on `highway` --
    not two separate `read_pbf_lines` calls. The `where` filter only reduces what crosses into
    pandas; it does NOT avoid the GDAL OSM driver's multi-GB temp SQLite build, which happens
    regardless -- calling `read_pbf_lines` twice therefore pays for that build twice."""
    combined = read_pbf_lines(pbf_path, (*FOOTPATH_TAGS, *NEAR_MISS_TAGS))
    footpaths = cast(GeoDataFrame, combined[combined["highway"].isin(FOOTPATH_TAGS)])
    near_miss = cast(GeoDataFrame, combined[combined["highway"].isin(NEAR_MISS_TAGS)])
    return footpaths.reset_index(drop=True), near_miss.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso", choices=sorted(PBF), required=True)
    ap.add_argument(
        "--limit", type=int, default=None,
        help="stop after exactly this many blocks read from the source (smoke run)")
    ap.add_argument("--batch-size", type=int, default=50_000)
    args = ap.parse_args()

    pbf_path = CACHE / "osm_pbf" / PBF[args.iso]
    if not pbf_path.exists():
        raise SystemExit(
            f"missing {pbf_path}\n"
            f"download it from https://download.geofabrik.de/ "
            f"(south-africa 417 MB, kenya 349 MB)")

    print(f"reading footpath + near-miss layers from {pbf_path.name} ...", flush=True)
    t0 = time.time()
    footpaths, near_miss = _read_footpath_and_near_miss_lines(pbf_path)
    print(f"  {len(footpaths):,} footpath ways, {len(near_miss):,} near-miss "
          f"({time.time()-t0:.0f}s)", flush=True)

    # Single-row-group parquets (833 MB / 386 MB): gpd.read_parquet will not stream a column.
    src = CACHE / f"{args.iso}_geodata.parquet"
    pf = pq.ParquetFile(src)

    out = CACHE / f"osm_coverage_{args.iso}.parquet"
    rows: list[dict[str, object]] = []
    done_ids: set[str] = set()
    if out.exists():
        # Resume support: a killed 5-10 hour run must not have to restart from zero. The parquet
        # is single-row-group (see above), so a resumed run still streams and decodes every block
        # from the start -- cheap (~0.3-0.6s/50k-batch) -- but skips the expensive per-block
        # `census_rows` work (the 3.31ms/block budget) for anything already on disk. Same
        # discipline as scripts/pair_matrix.py's checkpoint/resume pattern.
        existing = pd.read_parquet(out)
        rows = cast(list[dict[str, object]], existing.to_dict("records"))
        done_ids = {str(b) for b in existing["block_id"]}
        print(f"resuming from {out}: {len(rows):,} blocks already censused", flush=True)

    seen = 0
    rows_at_last_checkpoint = len(rows)
    t0 = time.time()

    for batch in pf.iter_batches(batch_size=args.batch_size,
                                 columns=["block_id", "building_count", "k_complexity",
                                          "geometry"]):
        blocks = _decode_batch(batch)
        if args.limit is not None:
            remaining = args.limit - seen
            if remaining <= 0:
                break
            if len(blocks) > remaining:
                # Trim to the exact remaining count so --limit means what it says, rather than
                # rounding up to the next --batch-size multiple.
                blocks = blocks.iloc[:remaining].reset_index(drop=True)

        by_zone: dict[int, list[int]] = defaultdict(list)
        reps = blocks.geometry.representative_point()
        for i, pt in enumerate(reps):
            by_zone[utm_zone_epsg(pt.x, pt.y)].append(i)

        for epsg, idx in by_zone.items():
            sub: GeoDataFrame = cast(GeoDataFrame, blocks.iloc[idx])
            assert_zone_fit(float(reps.iloc[idx[0]].x), epsg)
            sub = cast(GeoDataFrame, sub[~sub["block_id"].astype(str).isin(done_ids)])
            if sub.empty:
                continue
            bounds = sub.total_bounds
            fp = footpaths.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
            nm = near_miss.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
            batch_rows = census_rows(sub, fp, nm, epsg)
            for r, bc, kc in zip(batch_rows, sub["building_count"], sub["k_complexity"],
                                 strict=True):
                r["building_count"] = int(bc) if pd.notna(bc) else 0
                r["k_complexity"] = int(kc) if pd.notna(kc) else 0
            rows.extend(batch_rows)
            done_ids.update(str(b) for b in sub["block_id"])

        seen += len(blocks)
        rate = seen / max(time.time() - t0, 1e-9)
        print(f"  {seen:,} blocks  {rate:.0f}/s  ({len(rows):,} rows total)", flush=True)

        # Checkpoint after every batch that actually added rows (not just once at the end of a
        # 5-10 hour run): a kill loses at most one --batch-size worth of newly censused blocks.
        # Skip the write when nothing changed -- on a resumed run, early batches replay blocks
        # already on disk (see `done_ids` above) and contribute nothing new, so rewriting the
        # whole (potentially multi-GB, near the end of a country) output for them is pure waste.
        if len(rows) != rows_at_last_checkpoint:
            pd.DataFrame(rows).to_parquet(out)
            rows_at_last_checkpoint = len(rows)

        if args.limit is not None and seen >= args.limit:
            print(f"stopping at --limit {args.limit} ({seen:,} blocks processed)", flush=True)
            break

    covered = sum(1 for r in rows if int(r["n_interior_segments_0.5"]) > 0)
    print(f"\nwrote {out}  ({len(rows):,} rows)")
    print(f"blocks with >=1 interior footpath segment: {covered:,} "
          f"({covered/max(len(rows),1)*100:.1f}%)")


if __name__ == "__main__":
    main()
