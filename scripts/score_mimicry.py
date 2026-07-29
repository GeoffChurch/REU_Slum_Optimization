"""How closely does a synthesized network reproduce the REAL one?

Every comparison in this repo so far scores methods on permeability and displacement. For mimicry
that is the wrong target: real footpaths *lose* on permeability (they leave people walking twice as
far as a clearance tree, measured at matched displacement). If the goal is to generate
footpath-like networks, the objective has to be agreement with the real thing, and that is a
supervised problem -- 16,497 blocks have a known network, and `eval.agreement` already ships the
scorers.

Two targets, because the same flow field is meant to yield both:

  footpaths  the block's own interior OSM footpaths (FOOTPATH_TAGS). The low-flow web.
  streets    the region's real street network (NEAR_MISS_TAGS: service/residential/unclassified),
             which at multiblock scale is what a HIGH flow cut should reproduce -- road hierarchy
             is a flow-volume classification, so one field should give both under different cuts.

Scored with buffered IoU at two radii and directional Chamfer, kept directional per its contract:
precision is path drawn where none exists, recall is real path missed, and blending them hides
which way a generator fails.

    pixi run python -m scripts.score_mimicry --recipients 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from reblock.contracts import Block, Method
from reblock.data.osm_extract import FOOTPATH_TAGS, NEAR_MISS_TAGS, PbfDesireLines
from reblock.eval.agreement import buffered_iou, directional_chamfer
from reblock.methods.clearance import ClearanceReblocker
from reblock.methods.demand_greedy import DemandGreedyReblocker
from reblock.methods.flow_paths import FlowPathsReblocker
from reblock.methods.loop_closure import LoopClosureRefiner
from reblock.methods.osm_footpaths import interior_desire_lines
from scripts.pair_matrix import (
    DEFAULT_CACHE,
    PBF_BY_ISO,
    displacement_fraction,
    evenly_spaced,
    iso_of,
    load_pools,
)


def _reference(block: Block, source: PbfDesireLines) -> gpd.GeoDataFrame:
    b = gpd.GeoSeries([block.boundary], crs=block.crs).to_crs(4326).total_bounds
    lines = source.desire_lines((float(b[0]), float(b[1]), float(b[2]), float(b[3])), block.crs)
    from shapely.ops import unary_union
    streets = unary_union(list(block.streets.geometry))
    return interior_desire_lines(lines, block.boundary, streets, block.crs)


def agreement(proposal: gpd.GeoDataFrame, reference: gpd.GeoDataFrame) -> dict[str, float]:
    if proposal.empty or reference.empty:
        return {"iou_3m": 0.0, "iou_10m": 0.0, "chamfer_precision_m": float("nan"),
                "chamfer_recall_m": float("nan")}
    prec, rec = directional_chamfer(proposal, reference)
    return {"iou_3m": buffered_iou(proposal, reference, r=3.0),
            "iou_10m": buffered_iou(proposal, reference, r=10.0),
            "chamfer_precision_m": prec, "chamfer_recall_m": rec}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipients", type=int, default=20)
    ap.add_argument("--out", type=Path,
                    default=Path("data/benchmarks/mimicry_scores.parquet"))
    args = ap.parse_args()

    pools = load_pools()
    blocks = pools.blocks
    iso = iso_of(blocks)
    pbf = DEFAULT_CACHE / "osm_pbf" / PBF_BY_ISO[iso]
    foot_src = PbfDesireLines(pbf_path=pbf, tags=FOOTPATH_TAGS)
    street_src = PbfDesireLines(pbf_path=pbf, tags=NEAR_MISS_TAGS)

    usable = sorted(set(pools.recipients) & set(pools.donors))
    counts = [float(len(b.parcels)) for b in blocks]
    chosen = evenly_spaced(usable, counts, args.recipients)
    print(f"  scoring {len(chosen)} blocks against real footpaths and real streets", flush=True)

    methods: dict[str, Method] = {
        "flow_paths": FlowPathsReblocker(),
        "flow_paths_noreinforce": FlowPathsReblocker(iterations=1, reinforcement=0.0),
        "flow_paths_gateway": FlowPathsReblocker(destination="gateway"),
        "flow_paths_q99": FlowPathsReblocker(flow_quantile=0.99),
        "clearance": ClearanceReblocker(depth_target=1),
        "clearance_looped": LoopClosureRefiner(base=ClearanceReblocker(depth_target=1)),
        "demand_greedy_uniform": DemandGreedyReblocker(desire_source=None, depth_target=1),
    }

    rows: list[dict[str, object]] = []
    for n, i in enumerate(chosen, 1):
        block = blocks[i]
        foot = _reference(block, foot_src)
        street = _reference(block, street_src)
        if foot.empty:
            continue
        target_len = float(foot.geometry.length.sum())
        for name, method in methods.items():
            roads = method.propose(block).roads
            if roads is None:
                continue
            # Length-matched to the REFERENCE, the same convention every other comparison here
            # uses. Without it a method that simply draws more road scores better recall by
            # construction: the pilot had clearance at 3x flow_paths' length.
            cum = roads.geometry.length.cumsum()
            roads = roads[cum <= target_len] if target_len > 0 else roads.iloc[:0]
            row: dict[str, object] = {
                "block": block.block_id, "method": name,
                "road_len_m": float(roads.geometry.length.sum()) if len(roads) else 0.0,
                "displacement": displacement_fraction(block, roads),
                "n_segments": len(roads),
            }
            for label, ref in (("foot", foot), ("street", street)):
                for k, v in agreement(roads, ref).items():
                    row[f"{label}_{k}"] = v
            rows.append(row)
        print(f"  [{n}/{len(chosen)}] {block.block_id}  foot={len(foot)} street={len(street)}",
              flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(args.out)

    df = pd.DataFrame(rows)
    print(f"\nwrote {args.out} ({len(df)} rows)\n")
    if df.empty:
        return
    print(f"{'method':24} {'n':>3} {'FOOT iou10':>11} {'recall m':>9} | "
          f"{'STREET iou10':>13} {'recall m':>9} | {'road m':>8} {'disp':>6}")
    for name, g in df.groupby("method"):
        print(f"{name:24} {len(g):>3} {g.foot_iou_10m.median():>11.3f} "
              f"{g.foot_chamfer_recall_m.median():>9.1f} | "
              f"{g.street_iou_10m.median():>13.3f} {g.street_chamfer_recall_m.median():>9.1f} | "
              f"{g.road_len_m.median():>8.0f} {g.displacement.median():>6.3f}")
    print("\nrecall = mean distance from a REAL path to the nearest proposed one (lower is better)")


if __name__ == "__main__":
    main()
