"""Country-wide OSM footpath census driver (Phase 1, unit 1a).

Streams the ZAF/KEN blocks parquet, batches blocks by UTM zone, reads the country footpath layer
ONCE per batch, and writes one row per block to ~/.cache/reblock/osm_coverage_{iso}.parquet.

Budget, measured: 3.31 ms/block for clip + corridor difference + filter, so ~1.67 single-core
hours per tolerance over 1.81M blocks -- about 5 h for the 0.5/2/5 m sweep and ~10 h once the
near-miss tag set is included. Use --limit for a smoke run first.

Usage:
    pixi run python scripts/osm_census.py --iso ZAF --limit 5000
    pixi run python scripts/osm_census.py --iso ZAF
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso", choices=sorted(PBF), required=True)
    ap.add_argument("--limit", type=int, default=None, help="stop after N blocks (smoke run)")
    ap.add_argument("--batch-size", type=int, default=50_000)
    args = ap.parse_args()

    pbf_path = CACHE / "osm_pbf" / PBF[args.iso]
    if not pbf_path.exists():
        raise SystemExit(
            f"missing {pbf_path}\n"
            f"download it from https://download.geofabrik.de/ "
            f"(south-africa 417 MB, kenya 349 MB)")

    print(f"reading footpath layer from {pbf_path.name} ...", flush=True)
    t0 = time.time()
    footpaths = read_pbf_lines(pbf_path, FOOTPATH_TAGS)
    near_miss = read_pbf_lines(pbf_path, NEAR_MISS_TAGS)
    print(f"  {len(footpaths):,} footpath ways, {len(near_miss):,} near-miss "
          f"({time.time()-t0:.0f}s)", flush=True)

    # Single-row-group parquets (833 MB / 386 MB): gpd.read_parquet will not stream a column.
    src = CACHE / f"{args.iso}_geodata.parquet"
    pf = pq.ParquetFile(src)
    rows: list[dict[str, object]] = []
    seen = 0
    t0 = time.time()

    for batch in pf.iter_batches(batch_size=args.batch_size,
                                 columns=["block_id", "building_count", "k_complexity",
                                          "geometry"]):
        blocks = gpd.GeoDataFrame.from_arrow(batch)
        if blocks.crs is None:
            blocks = blocks.set_crs(4326)
        by_zone: dict[int, list[int]] = defaultdict(list)
        reps = blocks.geometry.representative_point()
        for i, pt in enumerate(reps):
            by_zone[utm_zone_epsg(pt.x, pt.y)].append(i)

        for epsg, idx in by_zone.items():
            sub = blocks.iloc[idx]
            assert_zone_fit(float(reps.iloc[idx[0]].x), epsg)
            bounds = sub.total_bounds
            fp = footpaths.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
            nm = near_miss.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
            batch_rows = census_rows(sub, fp, nm, epsg)
            for r, bc, kc in zip(batch_rows, sub["building_count"], sub["k_complexity"],
                                 strict=True):
                r["building_count"] = int(bc) if pd.notna(bc) else 0
                r["k_complexity"] = int(kc) if pd.notna(kc) else 0
            rows.extend(batch_rows)

        seen += len(blocks)
        rate = seen / max(time.time() - t0, 1e-9)
        print(f"  {seen:,} blocks  {rate:.0f}/s", flush=True)
        if args.limit and seen >= args.limit:
            print(f"stopping at --limit {args.limit} ({seen:,} blocks processed)", flush=True)
            break

    out = CACHE / f"osm_coverage_{args.iso}.parquet"
    pd.DataFrame(rows).to_parquet(out)
    covered = sum(1 for r in rows if int(r["n_interior_segments_0.5"]) > 0)
    print(f"\nwrote {out}  ({len(rows):,} rows)")
    print(f"blocks with >=1 interior footpath segment: {covered:,} "
          f"({covered/max(len(rows),1)*100:.1f}%)")


if __name__ == "__main__":
    main()
