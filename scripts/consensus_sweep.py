"""Two follow-ups to the consensus benchmark: the k-sweep, and a displacement-matched comparison.

**k-sweep.** `k=15` was inherited from the 2026-07-23 spike and never calibrated. The
single-donor -> consensus gain was +0.412 permeability in 100% of blocks, so the curve is steep at
the low end; where it saturates decides how many donors a deployment actually needs, and whether
the 14,189-block donor pool is being used or wasted. The sweep is NESTED -- k=3's donors are the
first 3 of k=30's -- so a difference between rungs is the extra donors, not a different draw. GW
fits are paid once at the largest k and reused.

**Displacement-matched.** Everything so far is length-matched, which flattered consensus: it
reached 1.04x clearance's permeability while displacing 0.248 vs 0.205 -- more permeable AND more
destructive, with neither dominating. Matching on displacement instead asks the question the
length-matched comparison could not: at the SAME cost in homes, which network moves more people?
Permeability and displacement are the repo's paired primary metrics, so a claim resting on one
while the other drifts is only half a result.

Scored with `scripts/consensus_matrix`'s scorer, so the same caveat holds: the 2026-07-28 numbers
in notes/2026-07-28-consensus-k-sweep-and-displacement.md predate the recalibrated metric, the
footprint-overlap displacement and the Open Buildings screen, and a re-run will not reproduce them.

    pixi run python -m scripts.consensus_sweep --recipients 20
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import TypedDict, cast

import geopandas as gpd
import numpy as np
import pandas as pd

from reblock.contracts import Block
from reblock.data.pools import capetown_pool, evenly_spaced, iso_of, load_pools, pbf_footpaths
from reblock.data.settlements import exclusion_holdout
from reblock.emit import pct_displaced
from reblock.eval.agreement import buffered_iou
from reblock.permeability import EgressContext, permeability
from reblock.transplant.consensus import extract_consensus, fit_donors, length_matched_prefix
from reblock.transplant.operating_points import SIGNATURE
from reblock.transplant.signature import signature
from reblock.transplant.transport import parcel_xy
from scripts.consensus_matrix import (
    CONF,
    Material,
    consensus_scorer,
    nearest_donors,
    within_length,
)

K_LADDER = (1, 2, 3, 5, 8, 12, 20, 30)


class SweepRow(TypedDict):
    """One (recipient, k) rung, exactly as written to the parquet."""

    recipient: str
    k: int
    perm_own: float
    disp_own: float
    perm_consensus_lenmatch: float
    disp_consensus_lenmatch: float
    perm_direct_lenmatch: float
    disp_direct_lenmatch: float
    perm_consensus_dispmatch: float
    perm_direct_dispmatch: float
    len_consensus_dispmatch: float
    len_direct_dispmatch: float
    perm_single: float
    iou_10m: float
    mean_gw_dist: float


def displacement_matched_prefix(
    block: Block, roads: gpd.GeoDataFrame, target_disp: float,
) -> gpd.GeoDataFrame:
    """The longest leading prefix of `roads` whose displacement stays within `target_disp`.

    `roads` arrives in the greedy construction order, so a prefix is a coherent partial network
    rather than an arbitrary subset -- the same convention `length_matched_prefix` relies on.
    Displacement is monotone non-decreasing in the prefix (adding road can only put more buildings
    inside a corridor), so this binary-searches instead of walking every prefix: ~9 evaluations
    rather than one per segment.
    """
    if len(roads) == 0:
        return roads
    lo, hi = 0, len(roads)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        prefix = cast(gpd.GeoDataFrame, roads.iloc[:mid])
        if pct_displaced(prefix, block.buildings) <= target_disp:
            lo = mid
        else:
            hi = mid - 1
    return cast(gpd.GeoDataFrame, roads.iloc[:lo])


def _length(roads: gpd.GeoDataFrame) -> float:
    return float(roads.geometry.length.sum()) if len(roads) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipients", type=int, default=20)
    ap.add_argument("--exclusion-radius-m", type=float, default=2000.0)
    ap.add_argument("--out", type=Path, default=Path("scratchpad/ot/consensus_sweep.parquet"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    pools = load_pools(capetown_pool(CONF))
    blocks, gdf = pools.blocks, pools.blocks_gdf
    signatures = {b.block_id: signature(parcel_xy(b), SIGNATURE) for b in blocks}
    scorer = consensus_scorer()
    material = Material(pbf_footpaths(iso_of(blocks)), scorer)

    usable = sorted(set(pools.recipients) & set(pools.donors))
    counts = [float(len(b.parcels)) for b in blocks]
    chosen = evenly_spaced(usable, counts, args.recipients)
    donor_set = set(pools.donors)
    k_max = max(K_LADDER)
    print(f"  {len(usable):,} recipients with own OSM; running {len(chosen)} at k<={k_max}",
          flush=True)

    rows: list[SweepRow] = []
    done: set[tuple[str, int]] = set()
    if args.out.exists():
        # This script's own output, so its records are the `SweepRow`s it wrote.
        rows = cast(list[SweepRow], pd.read_parquet(args.out).to_dict("records"))
        done = {(r["recipient"], int(r["k"])) for r in rows}
        print(f"resuming from {args.out}: {len(rows)} rows", flush=True)

    for n, i in enumerate(chosen, 1):
        recipient = blocks[i]
        own = material.of(recipient)
        if own is None:
            continue
        eligible = [j for j in exclusion_holdout(gdf, i, radius_m=args.exclusion_radius_m)
                    if j in donor_set]
        picked = nearest_donors(recipient, eligible, blocks, signatures, material.of, k_max)
        if len(picked) < 3:
            continue
        if all((recipient.block_id, k) in done for k in K_LADDER if k <= len(picked)):
            continue

        t0 = time.time()
        fits = fit_donors(recipient, picked, material.roads, scorer.transport)
        target_len = _length(own)
        disp_own = pct_displaced(own, recipient.buildings)
        ctx = EgressContext.of(recipient, scorer.permeability)
        perm_own = permeability(ctx, own)

        direct_full = scorer.direct_roads(recipient)
        direct_len = within_length(direct_full, target_len)
        # Matched on DISPLACEMENT to the block's own network: same cost in homes, so the
        # permeability comparison is finally like-for-like on the metric pair.
        direct_disp = displacement_matched_prefix(recipient, direct_full, disp_own)

        for k in K_LADDER:
            if k > len(picked) or (recipient.block_id, k) in done:
                continue
            cons_full = extract_consensus(recipient, fits[:k], material.quality,
                                          scorer.consensus)
            cons_len = length_matched_prefix(recipient, cons_full, target_len)
            cons_disp = displacement_matched_prefix(recipient, cons_full, disp_own)
            rows.append(SweepRow(
                recipient=recipient.block_id, k=k,
                perm_own=perm_own, disp_own=disp_own,
                # length-matched (comparable to the n=20 benchmark)
                perm_consensus_lenmatch=permeability(ctx, cons_len),
                disp_consensus_lenmatch=pct_displaced(cons_len, recipient.buildings),
                perm_direct_lenmatch=permeability(ctx, direct_len),
                disp_direct_lenmatch=pct_displaced(direct_len, recipient.buildings),
                # displacement-matched to the block's own network
                perm_consensus_dispmatch=permeability(ctx, cons_disp),
                perm_direct_dispmatch=permeability(ctx, direct_disp),
                len_consensus_dispmatch=_length(cons_disp),
                len_direct_dispmatch=_length(direct_disp),
                perm_single=permeability(ctx, scorer.single_donor(recipient, fits[:k])),
                iou_10m=buffered_iou(cons_len, own, r=10.0) if len(cons_len) else 0.0,
                mean_gw_dist=float(np.mean([f.gw_dist for f in fits[:k]])),
            ))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(args.out)
        print(f"  [{n}/{len(chosen)}] {recipient.block_id}: {len(picked)} donors, "
              f"{len([k for k in K_LADDER if k <= len(picked)])} rungs [{time.time()-t0:.1f}s]",
              flush=True)

    df = pd.DataFrame(rows)
    print(f"\nwrote {args.out} ({len(df)} rows)\n")
    if len(df):
        print("k-sweep (median over recipients):")
        print(f"{'k':>4} {'n':>4} {'perm/own':>9} {'perm/direct':>12} {'IoU@10m':>8} "
              f"{'DISPMATCH perm/direct':>22}")
        for rung, g in df.groupby("k"):
            print(f"{rung:>4} {len(g):>4} "
                  f"{(g.perm_consensus_lenmatch / g.perm_own).median():>9.3f} "
                  f"{(g.perm_consensus_lenmatch / g.perm_direct_lenmatch).median():>12.3f} "
                  f"{g.iou_10m.median():>8.3f} "
                  f"{(g.perm_consensus_dispmatch / g.perm_direct_dispmatch).median():>22.3f}")


if __name__ == "__main__":
    main()
