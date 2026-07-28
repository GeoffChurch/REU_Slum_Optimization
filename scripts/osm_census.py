"""Country-wide OSM footpath census driver (Phase 1, unit 1a).

Streams the ZAF/KEN blocks parquet, batches blocks by UTM zone, reads the country footpath layer
ONCE per batch, and writes one row per block to ~/.cache/reblock/osm_coverage_{iso}.parquet.

Budget, measured: 3.31 ms/block per interiority pass (clip + corridor difference + filter). Run
naively -- 1.81M blocks x 3 tolerances x 2 tag sets -- that is ~10 hours. Two defaults cut it to
minutes without losing anything downstream:

  * `--min-k` / `--min-buildings` prefilter. Everything downstream uses qualified blocks only and
    both columns are already in the parquet, so the filter is free and drops ~96% of the corpus.
  * `--tolerances 0.5`, one tolerance instead of three. The 400-block spike settled this: 0.5 -> 5 m
    moves the coverage gate by 2.6 points while total interior length drops 17.8%, so the sweep is
    about donor-quality ranking, not coverage. Run it on a sample.

That leaves ~65k qualified blocks x 2 passes ~= 7 minutes of per-block work, at which point GDAL's
temp-SQLite PBF ingest (~5-20 min for the 417 MB South Africa extract) dominates the wall clock.

Usage (module form -- `python scripts/osm_census.py` fails at import once anything imports
transitively through `reblock.data.provision`, which needs the repo root on `sys.path`; see
`scripts/pair_matrix.py` for the same convention):
    pixi run python -m scripts.osm_census --iso ZAF --limit 5000
    pixi run python -m scripts.osm_census --iso ZAF
    pixi run python -m scripts.osm_census --iso ZAF --limit 20000 --tolerances 0.5,2,5
"""
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from geopandas import GeoDataFrame
from shapely.geometry import Polygon

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


def _tolerance_list(raw: str) -> tuple[float, ...]:
    try:
        tols = tuple(float(part) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a comma-separated float list: {raw!r}") from exc
    if not tols:
        raise argparse.ArgumentTypeError("--tolerances must name at least one tolerance")
    return tols


def _prefilter(blocks: GeoDataFrame, min_k: int, min_buildings: int) -> GeoDataFrame:
    """Drop blocks no downstream consumer can use.

    Everything downstream works on qualified blocks only, and both filter columns are already in
    the parquet, so this costs nothing and removes ~96% of the corpus -- the difference between a
    multi-hour run and a few minutes. Missing values fail the filter: a block with no recorded
    `building_count` cannot be shown to clear the floor.
    """
    keep = pd.Series(True, index=blocks.index)
    if min_k > 0:
        keep &= blocks["k_complexity"].fillna(-1) >= min_k
    if min_buildings > 0:
        keep &= blocks["building_count"].fillna(-1) >= min_buildings
    return cast(GeoDataFrame, blocks[keep])


def _checkpoint(rows: list[dict[str, object]], out: Path) -> None:
    """Write the checkpoint atomically: same-directory temp file, then `os.replace`.

    `to_parquet` straight onto `out` leaves a truncated file if the process dies mid-write, and
    the resume path reads `out` unconditionally -- so a kill during a checkpoint would not merely
    lose that batch, it would poison every subsequent resume. `os.replace` is atomic within a
    filesystem, which is why the temp file must be a sibling rather than in /tmp.
    """
    tmp = out.with_name(out.name + ".tmp")
    pd.DataFrame(rows).to_parquet(tmp)
    os.replace(tmp, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso", choices=sorted(PBF), required=True)
    ap.add_argument(
        "--limit", type=int, default=None,
        help="stop after exactly this many blocks read from the source (smoke run)")
    ap.add_argument("--batch-size", type=int, default=50_000)
    ap.add_argument(
        "--tolerances", type=_tolerance_list, default=(0.5,),
        help="comma-separated street-corridor tolerances in metres (default: 0.5). The full "
             "0.5,2,5 sweep is diagnostic -- run it on a sample, not the corpus")
    ap.add_argument(
        "--min-k", type=int, default=3,
        help="skip blocks with k_complexity below this (default: 3, 0 disables). Deliberately "
             "looser than the qualified k>=4 band: k_complexity is kblock's metric, not this "
             "repo's BFS-peel screen, so the census keeps headroom")
    ap.add_argument(
        "--min-buildings", type=int, default=40,
        help="skip blocks with building_count below this (default: 40, 0 disables)")
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
        # A resumed run must produce the SAME columns as the rows already on disk, or the
        # concatenated output silently carries NaN for whichever columns each half is missing --
        # reachable both from --tolerances and from a schema change between runs, so check the
        # whole expected column set rather than trusting either.
        wanted = set(census_rows(
            cast(GeoDataFrame, GeoDataFrame(
                {"block_id": ["probe"]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1)])], crs=4326)),
            gpd.GeoDataFrame(geometry=[], crs=4326),
            gpd.GeoDataFrame(geometry=[], crs=4326),
            32734, tolerances=args.tolerances)[0])
        missing = wanted - set(existing.columns)
        if missing:
            raise SystemExit(
                f"{out} lacks columns this run would produce: {sorted(missing)}.\n"
                f"Either it was written with different --tolerances, or its schema predates a "
                f"change to census_rows. Migrate it or delete it to start over.")
        rows = cast(list[dict[str, object]], existing.to_dict("records"))
        done_ids = {str(b) for b in existing["block_id"]}
        print(f"resuming from {out}: {len(rows):,} blocks already censused", flush=True)

    seen = 0
    kept_total = 0
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

        # --limit counts blocks READ from the source, so it is applied before the prefilter:
        # "smoke-run the first 5,000 blocks" should mean the same thing whatever the thresholds.
        seen += len(blocks)
        blocks = cast(GeoDataFrame,
                      _prefilter(blocks, args.min_k, args.min_buildings).reset_index(drop=True))
        kept_total += len(blocks)
        if blocks.empty:
            if args.limit is not None and seen >= args.limit:
                break
            continue

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
            batch_rows = census_rows(sub, fp, nm, epsg, tolerances=args.tolerances)
            for r, bc, kc in zip(batch_rows, sub["building_count"], sub["k_complexity"],
                                 strict=True):
                r["building_count"] = int(bc) if pd.notna(bc) else 0
                r["k_complexity"] = int(kc) if pd.notna(kc) else 0
            rows.extend(batch_rows)
            done_ids.update(str(b) for b in sub["block_id"])

        rate = seen / max(time.time() - t0, 1e-9)
        print(f"  {seen:,} read  {kept_total:,} kept  {rate:.0f}/s "
              f"({len(rows):,} rows total)", flush=True)

        # Checkpoint after every batch that actually added rows (not just once at the end of a
        # 5-10 hour run): a kill loses at most one --batch-size worth of newly censused blocks.
        # Skip the write when nothing changed -- on a resumed run, early batches replay blocks
        # already on disk (see `done_ids` above) and contribute nothing new, so rewriting the
        # whole (potentially multi-GB, near the end of a country) output for them is pure waste.
        if len(rows) != rows_at_last_checkpoint:
            _checkpoint(rows, out)
            rows_at_last_checkpoint = len(rows)

        if args.limit is not None and seen >= args.limit:
            print(f"stopping at --limit {args.limit} ({seen:,} blocks processed)", flush=True)
            break

    if len(rows) != rows_at_last_checkpoint:
        _checkpoint(rows, out)

    gate = f"n_interior_segments_{min(args.tolerances)}"
    covered = sum(1 for r in rows if int(r[gate]) > 0)
    failed = sum(1 for r in rows if r.get("census_failed"))
    dropped = seen - kept_total
    if failed:
        print(f"\n{failed:,} blocks defeated GEOS even after make_valid and are recorded with "
              f"census_failed=True -- exclude them, do not read them as uncovered")
    print(f"\nwrote {out}  ({len(rows):,} rows)")
    print(f"prefilter (k>={args.min_k}, buildings>={args.min_buildings}): "
          f"read {seen:,}, kept {kept_total:,}, dropped {dropped:,} "
          f"({dropped/max(seen,1)*100:.1f}%)")
    print(f"blocks with >=1 interior footpath segment (at tol={min(args.tolerances)}): "
          f"{covered:,} ({covered/max(len(rows),1)*100:.1f}%)")


if __name__ == "__main__":
    main()
