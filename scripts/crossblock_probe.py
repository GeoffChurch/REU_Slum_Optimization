"""Cross-block Phase-0 probe: over a stratified sample of adjacent Cape Town clusters,
score the boundary-reconciled block-local peel baseline and the heuristic spine-merge
reference on the metric basis, validate the basis's orthogonality with a correlation
matrix, and report the cross-block headroom (metric distributions vs their floors +
baseline->reference improvement) for a documented go/no-go. No pre-registered bar.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
from shapely import STRtree, make_valid

from reblock.contracts import Block, Proposal, Region
from reblock.data.kblock import KblockSource
from reblock.derive.cluster import merge_cluster
from reblock.derive.crossblock import reconciled_baseline, spine_merge_reference
from reblock.eval.kcomplexity import KComplexityEval
from reblock.eval.structure import StructureEval

ROOT = Path(__file__).resolve().parents[1]
CT_BLOCKS = str(ROOT / "tests" / "data" / "kblock" / "blocks_capetown_sample.parquet")
CT_BLD = str(ROOT / "tests" / "data" / "kblock" / "buildings_capetown_sample.parquet")
_BASIS = ["geometric_access_max_m", "geometric_access_p95_m", "circuity", "throughput_ratio",
          "meshedness", "four_way_fraction", "dead_end_fraction",
          "added_road_length_per_parcel", "n_cross_block_streets",
          "boundary_redundant_road_fraction"]


def enumerate_adjacent_pairs(blocks: gpd.GeoDataFrame) -> list[tuple[str, str]]:
    geoms = [make_valid(g) for g in blocks.geometry]
    ids = [str(b) for b in blocks["block_id"]]
    tree = STRtree(geoms)
    pairs: set[tuple[str, str]] = set()
    left, right = tree.query(geoms, predicate="intersects")
    for i, j in zip(left.tolist(), right.tolist(), strict=True):
        if i < j and geoms[i].intersection(geoms[j]).length > 0:
            pairs.add((ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i]))
    return sorted(pairs)


def _score(merged: Block, proposal: Proposal, prefix: str) -> dict[str, float]:
    sv = StructureEval().score(merged, proposal).values
    kv = KComplexityEval().score(merged, proposal).values
    row = {f"{prefix}_{k}": float(sv[k]) for k in _BASIS if k in sv}
    row[f"{prefix}_k_after"] = float(kv["k_after"])
    return row


def probe_cluster(
    block_ids: list[str], blocks_path: str, buildings_path: str) -> dict[str, float | str]:
    src = KblockSource(blocks_path, buildings_path, region_id="capetown", block_ids=block_ids)
    region: Region = src.region()
    merged = merge_cluster(region)
    base = reconciled_baseline(region, merged)
    ref = spine_merge_reference(merged, base)
    row: dict[str, float | str] = {
        "block_ids": "+".join(block_ids), "n_parcels": float(len(merged.parcels))}
    row.update(_score(merged, base, "base"))
    row.update(_score(merged, ref, "ref"))
    return row


def main(n_sample: int = 30) -> None:
    blocks = gpd.read_parquet(CT_BLOCKS, columns=["block_id", "k_complexity", "geometry"])
    blocks = blocks.to_crs(blocks.estimate_utm_crs())
    kmap = {str(b): float(k) for b, k in
            zip(blocks["block_id"], blocks["k_complexity"], strict=True)}
    pairs = enumerate_adjacent_pairs(blocks)
    # stratify by min(kblock_k) across the pair, sampled deterministically (sorted, strided)
    pairs.sort(key=lambda p: (min(kmap[p[0]], kmap[p[1]]), p))
    step = max(1, len(pairs) // n_sample)
    sample = pairs[::step][:n_sample]

    rows: list[dict[str, float | str]] = []
    for a, b in sample:
        try:
            rows.append(probe_cluster([a, b], CT_BLOCKS, CT_BLD))
        except (ValueError, KeyError) as exc:      # non-contiguous / sparse cluster -> skip, log
            print(f"skip {a}+{b}: {exc}", file=sys.stderr)
    df = pd.DataFrame(rows)

    print("\n=== orthogonality: correlation matrix (base_* basis) ===")
    base_cols = [f"base_{k}" for k in _BASIS if f"base_{k}" in df]
    # pandas-stubs' DataFrame.__getitem__(list[str]) overload resolves to Series[Any] here
    # rather than DataFrame (a stub-overload-resolution quirk, not a real runtime ambiguity --
    # a list-of-columns selection is always a DataFrame at runtime); cast to the true runtime
    # type, matching the existing `cast(gpd.GeoDataFrame, ...)` precedent in kblock.py.
    base_df = cast(pd.DataFrame, df[base_cols])
    print(base_df.corr().round(2).to_string())

    print("\n=== headroom: baseline metric distributions (vs floors) ===")
    print(base_df.describe(percentiles=[0.5]).round(3).to_string())

    print("\n=== baseline -> spine-merge reference improvement (median) ===")
    for k in ("circuity", "meshedness", "n_cross_block_streets", "added_road_length_per_parcel"):
        if f"base_{k}" in df and f"ref_{k}" in df:
            print(f"  {k}: {df[f'base_{k}'].median():.3f} -> {df[f'ref_{k}'].median():.3f}")


if __name__ == "__main__":
    main()
