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

    pixi run python -m scripts.consensus_sweep --recipients 20
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from reblock.contracts import Block
from reblock.data.settlements import exclusion_holdout
from reblock.eval.agreement import buffered_iou
from reblock.methods.clearance import ClearanceReblocker
from reblock.permeability import permeability
from scripts.consensus_matrix import _bc, donor_quality, extract_consensus, fit_donors
from scripts.pair_matrix import (
    desire_source,
    displacement_fraction,
    evenly_spaced,
    fetch_donor_lines,
    iso_of,
    load_pools,
)

K_LADDER = (1, 2, 3, 5, 8, 12, 20, 30)


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
        if displacement_fraction(block, roads.iloc[:mid]) <= target_disp:
            lo = mid
        else:
            hi = mid - 1
    return roads.iloc[:lo]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipients", type=int, default=20)
    ap.add_argument("--exclusion-radius-m", type=float, default=2000.0)
    ap.add_argument("--out", type=Path, default=Path("scratchpad/ot/consensus_sweep.parquet"))
    args = ap.parse_args()

    pools = load_pools()
    blocks, gdf = pools.blocks, pools.blocks_gdf
    source = desire_source("pbf", iso_of(blocks))
    bc = _bc()

    usable = sorted(set(pools.recipients) & set(pools.donors))
    counts = [float(len(b.parcels)) for b in blocks]
    chosen = evenly_spaced(usable, counts, args.recipients)
    donor_set = set(pools.donors)
    k_max = max(K_LADDER)
    print(f"  {len(usable):,} recipients with own OSM; running {len(chosen)} at k<={k_max}",
          flush=True)

    roads_cache: dict[str, gpd.GeoDataFrame] = {}
    quality: dict[str, float] = {}

    def material(b: Block) -> gpd.GeoDataFrame | None:
        if b.block_id not in roads_cache:
            status, lines = fetch_donor_lines(source, b)
            if status != "ok" or lines is None:
                return None
            roads_cache[b.block_id] = lines
            quality[b.block_id] = donor_quality(b, lines)
        return roads_cache[b.block_id]

    rows: list[dict[str, object]] = []
    if args.out.exists():
        rows = pd.read_parquet(args.out).to_dict("records")
        print(f"resuming from {args.out}: {len(rows)} rows", flush=True)
    done = {(str(r["recipient"]), int(r["k"])) for r in rows}

    for n, i in enumerate(chosen, 1):
        recipient = blocks[i]
        own = material(recipient)
        if own is None:
            continue
        eligible = [j for j in exclusion_holdout(gdf, i, radius_m=args.exclusion_radius_m)
                    if j in donor_set]
        ranked = sorted(eligible, key=lambda j: float(np.linalg.norm(
            pools.signatures[blocks[j].block_id] - pools.signatures[recipient.block_id])))
        picked: list[Block] = []
        for j in ranked:
            if len(picked) >= k_max:
                break
            if material(blocks[j]) is not None:
                picked.append(blocks[j])
        if len(picked) < 3:
            continue
        if all((recipient.block_id, k) in done for k in K_LADDER if k <= len(picked)):
            continue

        t0 = time.time()
        transported, dists = fit_donors(recipient, picked, roads_cache)
        target_len = float(own.geometry.length.sum())
        disp_own = displacement_fraction(recipient, own)
        perm_own = float(permeability(recipient, own))

        direct_full = ClearanceReblocker().propose(recipient).roads
        cum = direct_full.geometry.length.cumsum()
        direct_len = direct_full[cum <= target_len] if target_len > 0 else direct_full.iloc[:0]
        # Matched on DISPLACEMENT to the block's own network: same cost in homes, so the
        # permeability comparison is finally like-for-like on the metric pair.
        direct_disp = displacement_matched_prefix(recipient, direct_full, disp_own)

        for k in K_LADDER:
            if k > len(picked) or (recipient.block_id, k) in done:
                continue
            cons_full, single = extract_consensus(
                recipient, picked[:k], transported[:k], dists[:k], quality)
            cons_len = (bc.length_matched_prefix(recipient, cons_full, target_len)
                        if len(cons_full) else cons_full)
            cons_disp = displacement_matched_prefix(recipient, cons_full, disp_own)
            rows.append({
                "recipient": recipient.block_id, "k": k,
                "perm_own": perm_own, "disp_own": disp_own,
                # length-matched (comparable to the n=20 benchmark)
                "perm_consensus_lenmatch": float(permeability(recipient, cons_len)),
                "disp_consensus_lenmatch": displacement_fraction(recipient, cons_len),
                "perm_direct_lenmatch": float(permeability(recipient, direct_len)),
                "disp_direct_lenmatch": displacement_fraction(recipient, direct_len),
                # displacement-matched to the block's own network
                "perm_consensus_dispmatch": float(permeability(recipient, cons_disp)),
                "perm_direct_dispmatch": float(permeability(recipient, direct_disp)),
                "len_consensus_dispmatch": (float(cons_disp.geometry.length.sum())
                                            if len(cons_disp) else 0.0),
                "len_direct_dispmatch": (float(direct_disp.geometry.length.sum())
                                         if len(direct_disp) else 0.0),
                "perm_single": float(permeability(recipient, single)),
                "iou_10m": buffered_iou(cons_len, own, r=10.0) if len(cons_len) else 0.0,
                "mean_gw_dist": float(np.mean(dists[:k])),
            })
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
        for k, g in df.groupby("k"):
            print(f"{k:>4} {len(g):>4} "
                  f"{(g.perm_consensus_lenmatch / g.perm_own).median():>9.3f} "
                  f"{(g.perm_consensus_lenmatch / g.perm_direct_lenmatch).median():>12.3f} "
                  f"{g.iou_10m.median():>8.3f} "
                  f"{(g.perm_consensus_dispmatch / g.perm_direct_dispmatch).median():>22.3f}")


if __name__ == "__main__":
    main()
