"""Can flow accumulation reproduce a REGION's street network?

The per-block street test was mis-framed: streets run *between* blocks, so scoring them inside one
block's interior finds almost nothing (street IoU was 0.000 at the median, nonzero in 60 of 140
rows). Streets are a region-scale object, which is where the high-flow cut was always supposed to
apply -- road hierarchy is a flow-volume classification, so one accumulated field should give
footpaths at a low cut and streets at a high one.

The test has to strip the answer out first. `region.region_block` sets `streets` to the union of
every member block's streets, which ALREADY contains the inter-block network -- reblock it as-is
and every parcel is served before a method starts, so nothing has to re-derive anything. Here the
region is rebuilt with only its OUTER boundary as existing street, and the method is asked to
generate the interior network from scratch. The reference is the real OSM street network (service /
residential / unclassified) clipped to the region, minus that same outer corridor.

Length-matched to the reference, because the reported metrics are IoU and Chamfer and their
confound is how much line you draw.

    pixi run python -m scripts.score_street_mimicry --regions 3
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from shapely.ops import unary_union

from reblock.contracts import Block, Method, Screen, Source
from reblock.data.osm_extract import NEAR_MISS_TAGS, PbfDesireLines
from reblock.eval.agreement import buffered_iou, directional_chamfer
from reblock.methods.clearance import ClearanceReblocker
from reblock.methods.demand_greedy import DemandGreedyReblocker
from reblock.methods.flow_paths import FlowPathsReblocker
from reblock.methods.osm_footpaths import interior_desire_lines
from reblock.pipeline import build_regions
from reblock.region import DenseClusterRegionBuilder, region_block
from scripts.pair_matrix import DEFAULT_CACHE, PBF_BY_ISO, displacement_fraction


def stripped_region(blocks: list[Block]) -> Block:
    """The region with ONLY its outer boundary as existing street.

    `region_block` would hand back the inter-block network as already-built, which is the thing
    this test asks a method to reproduce -- leaving it in would score every method against an
    answer it was given.
    """
    fused = region_block(blocks)
    outer = gpd.GeoDataFrame(geometry=[fused.boundary.boundary], crs=fused.crs)
    return Block(block_id=fused.block_id + ":stripped", crs=fused.crs, boundary=fused.boundary,
                 parcels=fused.parcels, streets=outer,
                 building_points=fused.building_points)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", type=int, default=3)
    ap.add_argument("--max-buildings", type=int, default=3000)
    ap.add_argument("--out", type=Path,
                    default=Path("data/benchmarks/street_mimicry.parquet"))
    args = ap.parse_args()

    overrides = ["metric=density_compactness", "data=capetown_full", "screen=dense_compact",
                 "region_builder=dense_cluster", "max_blocks=1"]
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config", overrides=overrides)
    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    builder = DenseClusterRegionBuilder(max_buildings=args.max_buildings)
    regions = build_regions(source, screen, builder, None, args.regions)
    print(f"  {len(regions)} regions", flush=True)

    street_src = PbfDesireLines(pbf_path=DEFAULT_CACHE / "osm_pbf" / PBF_BY_ISO["ZAF"],
                                tags=NEAR_MISS_TAGS)
    methods: dict[str, Method] = {
        "flow_paths_q90": FlowPathsReblocker(flow_quantile=0.90),
        "flow_paths_q97": FlowPathsReblocker(flow_quantile=0.97),
        "flow_paths_q99": FlowPathsReblocker(flow_quantile=0.99),
        "flow_paths_gateway_q97": FlowPathsReblocker(destination="gateway", flow_quantile=0.97),
        "clearance": ClearanceReblocker(depth_target=1, max_roads=3000),
        "demand_greedy_uniform": DemandGreedyReblocker(desire_source=None, depth_target=1,
                                                       max_roads=3000),
    }

    rows: list[dict[str, object]] = []
    for n, region in enumerate(regions, 1):
        blk = stripped_region(region)
        b = gpd.GeoSeries([blk.boundary], crs=blk.crs).to_crs(4326).total_bounds
        lines = street_src.desire_lines(
            (float(b[0]), float(b[1]), float(b[2]), float(b[3])), blk.crs)
        ref = interior_desire_lines(lines, blk.boundary,
                                    unary_union(list(blk.streets.geometry)), blk.crs)
        target = float(ref.geometry.length.sum())
        print(f"  [{n}/{len(regions)}] {len(region)} blocks / {len(blk.parcels)} parcels; "
              f"real interior streets: {len(ref)} segs, {target:.0f} m", flush=True)
        if ref.empty:
            continue
        for name, method in methods.items():
            roads = method.propose(blk).roads
            if roads is None or roads.empty:
                continue
            cum = roads.geometry.length.cumsum()
            roads = roads[cum <= target] if target > 0 else roads.iloc[:0]
            if roads.empty:
                continue
            prec, rec = directional_chamfer(roads, ref)
            rows.append({
                "region": blk.block_id[:60], "method": name, "n_blocks": len(region),
                "ref_len_m": target, "road_len_m": float(roads.geometry.length.sum()),
                "iou_10m": buffered_iou(roads, ref, r=10.0),
                "iou_20m": buffered_iou(roads, ref, r=20.0),
                "chamfer_precision_m": prec, "chamfer_recall_m": rec,
                "displacement": displacement_fraction(blk, roads),
            })
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(args.out)

    df = pd.DataFrame(rows)
    print(f"\nwrote {args.out} ({len(df)} rows)\n")
    if df.empty:
        return
    print(f"{'method':24} {'n':>3} {'IoU@10m':>8} {'IoU@20m':>8} {'recall m':>9} "
          f"{'precision m':>12} {'disp':>6}")
    for name, g in df.groupby("method"):
        print(f"{name:24} {len(g):>3} {g.iou_10m.median():>8.3f} {g.iou_20m.median():>8.3f} "
              f"{g.chamfer_recall_m.median():>9.1f} {g.chamfer_precision_m.median():>12.1f} "
              f"{g.displacement.median():>6.3f}")
    print("\nrecall = mean distance from a REAL street to the nearest proposed road")


if __name__ == "__main__":
    main()
