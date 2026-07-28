"""Barycenter-consensus benchmark: predict a block's real footpaths from its mapped neighbours.

The single-donor question is answered and the answer is no -- transplant fidelity does not
measurably depend on donor GW distance (beta = -0.18, 95% [-4.09, +3.85]; see
notes/2026-07-28-no-detectable-distance-effect.md), which corroborates the 2026-07-23 finding that
single-donor transplant is Pareto-dominated. What that study ALSO found, and what has never been
tested beyond n=1, is that a weighted CONSENSUS of several similar blocks' real OSM footpaths
reaches ~94% of a recipient's own network. This measures that at scale.

Mechanism, unchanged from `scratchpad/ot/osm_barycenter.py`: fit GW+UOT from each donor's parcel
cloud to the recipient's, transport the donor's real footpaths through it, weight donors by
`quality_i * exp(-gw_dist_i / tau)` (quality = the donor's own permeability x (1 - displacement),
tau = median GW distance), buffer the transported networks into a demand field, and extract a
network gap-aware along the recipient's own ChordSubstrate. Everything is then length-matched to
the recipient's OWN footpath length, so "matched budget" means what the real network actually
spent.

TWO ARMS, always run as a pair. With the exclusion radius, donors must be >2 km away; without it,
the nearest donors are admitted. The gap between the arms is the LEAKAGE estimate, and it is not
hypothetical: a median 26.7% of a recipient's nearest 15 donors sit inside 2 km, and for 24.5% of
recipients all 15 do (`scripts/donor_availability.py`). The 94% figure was measured with no
distance constraint at all, so the held-out arm is the one that can be believed.

NOT comparable to that 94% numerically. `GW_FIT_KW` uses eps=0.01, and since the Prop.-2 gradient
factor was fixed (notes/2026-07-27-gw-pot-crossvalidation.md) that is HALF the regularization the
original run actually had. Same mechanism, tighter coupling.

    pixi run python -m scripts.consensus_matrix --recipients 5 --k 15   # pilot
    pixi run python -m scripts.consensus_matrix --out data/benchmarks/consensus_matrix.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd

from reblock.budget import building_radii
from reblock.contracts import Block
from reblock.data.settlements import exclusion_holdout
from reblock.eval.agreement import buffered_iou, directional_chamfer
from reblock.methods.clearance import ClearanceReblocker
from reblock.methods.substrates import ChordSubstrate
from reblock.permeability import permeability
from scripts.pair_matrix import (
    CORRIDOR_M,
    _ot,
    desire_source,
    displacement_fraction,
    evenly_spaced,
    fetch_donor_lines,
    iso_of,
    load_pools,
    zone_source,
)

_BC_DIR = Path("scratchpad/ot")
_bc_ns: SimpleNamespace | None = None


def _bc() -> SimpleNamespace:
    """The consensus half of the 2026-07-23 spike, lazily imported from gitignored scratchpad.

    Same deferral discipline as `pair_matrix._ot`: this is spike code that predates the test suite
    and does not travel with a fresh checkout, so the import and its error message live at the
    call site rather than at module import time.
    """
    global _bc_ns
    if _bc_ns is not None:
        return _bc_ns
    if not _BC_DIR.is_dir():
        raise SystemExit(
            "scratchpad/ot/ is missing -- it holds the 2026-07-23 barycenter spike "
            "(barycenter_amortization.py, amortization_test.py, gap_snap_fix.py). Rebuild from "
            "docs/superpowers/notes/2026-07-23-ot-road-transplant.md before running this.")
    if str(_BC_DIR) not in sys.path:
        sys.path.insert(0, str(_BC_DIR))
    from amortization_test import PARAMS, length_matched_prefix, parcel_xy, score
    from barycenter_amortization import (
        GW_FIT_KW,
        build_demand_field,
        demand_edge_weights,
        demand_greedy_reblock,
    )
    from gap_snap_fix import gap_snap_routed

    _bc_ns = SimpleNamespace(
        PARAMS=PARAMS, GW_FIT_KW=GW_FIT_KW, parcel_xy=parcel_xy, score=score,
        length_matched_prefix=length_matched_prefix, build_demand_field=build_demand_field,
        demand_edge_weights=demand_edge_weights, demand_greedy_reblock=demand_greedy_reblock,
        gap_snap_routed=gap_snap_routed)
    return _bc_ns


def _perm_disp(block: Block, roads: gpd.GeoDataFrame) -> tuple[float, float]:
    radii = building_radii(block.building_points, CORRIDOR_M)
    del radii
    return (float(permeability(block, roads)), displacement_fraction(block, roads))


def donor_quality(donor: Block, roads: gpd.GeoDataFrame) -> float:
    """`own_perm * (1 - own_disp)` -- how good the donor's real network is ON ITS OWN BLOCK.

    A donor with an excellent network is worth more in the consensus than a close-but-poorly-served
    one, which is why weight is quality x proximity rather than proximity alone.
    """
    perm, disp = _perm_disp(donor, roads)
    return float(perm * (1.0 - disp))


def fit_donors(
    recipient: Block, donor_blocks: list[Block], donor_roads: dict[str, gpd.GeoDataFrame],
) -> tuple[list[gpd.GeoDataFrame], list[float]]:
    """(transported networks, GW distances) -- the expensive half, one GW fit per donor.

    Split out from extraction so a k-sweep pays for the fits ONCE at the largest k and reuses the
    first k of them at every smaller k. That also makes the sweep properly nested: the k=3 donor
    set is a subset of the k=15 one, so a difference between rungs is the extra donors and not a
    different draw.
    """
    ot, bc = _ot(), _bc()
    r_xy = bc.parcel_xy(recipient)
    c2 = ot.normalized_dist_matrix(r_xy)
    dists, transported = [], []
    for d in donor_blocks:
        d_xy = bc.parcel_xy(d)
        fit = ot.fit_transport(d_xy, r_xy, **bc.GW_FIT_KW)
        dists.append(ot.gw_cost(fit.pi, ot.normalized_dist_matrix(d_xy), c2))
        transported.append(ot.transport_lines(donor_roads[d.block_id], fit,
                                              out_crs=recipient.crs))
    return transported, dists


def extract_consensus(
    recipient: Block, donor_blocks: list[Block], transported: list[gpd.GeoDataFrame],
    dists: list[float], quality: dict[str, float],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """(consensus network, best single donor's transplant) -- the cheap half."""
    bc = _bc()
    tau = float(np.median(dists)) or 1.0
    weights = [quality[d.block_id] * float(np.exp(-gw / tau))
               for d, gw in zip(donor_blocks, dists, strict=True)]
    field = bc.build_demand_field(transported, weights)
    graph = ChordSubstrate().build(recipient)
    consensus = bc.demand_greedy_reblock(recipient, graph, bc.demand_edge_weights(graph, field))
    best = int(np.argmin(dists))
    single = bc.gap_snap_routed(transported[best], recipient, substrate=ChordSubstrate())
    return consensus, single


def consensus_for(
    recipient: Block, own_roads: gpd.GeoDataFrame, donor_blocks: list[Block],
    donor_roads: dict[str, gpd.GeoDataFrame], quality: dict[str, float],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list[float]]:
    """(consensus network, best single donor's transplant, per-donor GW distances)."""
    transported, dists = fit_donors(recipient, donor_blocks, donor_roads)
    consensus, single = extract_consensus(
        recipient, donor_blocks, transported, dists, quality)
    return consensus, single, dists


def score_recipient(
    recipient: Block, own_roads: gpd.GeoDataFrame, donor_blocks: list[Block],
    donor_roads: dict[str, gpd.GeoDataFrame], quality: dict[str, float], *, arm: str, k: int,
) -> dict[str, object]:
    bc = _bc()
    target = float(own_roads.geometry.length.sum())
    consensus, single, dists = consensus_for(
        recipient, own_roads, donor_blocks, donor_roads, quality)
    cons = bc.length_matched_prefix(recipient, consensus, target) if len(consensus) else consensus
    sing = bc.length_matched_prefix(recipient, single, target) if len(single) else single
    direct_full = ClearanceReblocker().propose(recipient).roads
    cum = direct_full.geometry.length.cumsum()
    direct = direct_full[cum <= target] if target > 0 else direct_full.iloc[:0]

    perm_own, disp_own = _perm_disp(recipient, own_roads)
    perm_cons, disp_cons = _perm_disp(recipient, cons)
    perm_sing, _ = _perm_disp(recipient, sing)
    perm_dir, disp_dir = _perm_disp(recipient, direct)
    chamfer = (directional_chamfer(cons, own_roads) if len(cons)
               else (float("nan"), float("nan")))
    return {
        "recipient": recipient.block_id, "arm": arm, "k": len(donor_blocks), "k_requested": k,
        "own_len_m": target,
        # The headline the 2026-07-23 study reported as "94% of a block's own OSM".
        "perm_ratio_own": perm_cons / perm_own if perm_own > 0 else float("nan"),
        "perm_ratio_direct": perm_cons / perm_dir if perm_dir > 0 else float("nan"),
        "perm_consensus": perm_cons, "perm_own": perm_own,
        "perm_single": perm_sing, "perm_direct": perm_dir,
        "disp_consensus": disp_cons, "disp_own": disp_own, "disp_direct": disp_dir,
        # Geometric agreement with the ground truth, the prediction branch's own scorer.
        # IoU at two radii because buffers stop overlapping past 2r -- at 3 m it reads 0 for
        # anything more than 6 m off, which a predicted network easily is, so a single radius
        # would report a flat zero and hide the gradient. Chamfer is kept DIRECTIONAL, per its
        # own contract: precision is paths drawn that are not there, recall is real paths missed,
        # and blending them hides which way the prediction fails.
        "iou_3m": buffered_iou(cons, own_roads, r=3.0) if len(cons) else 0.0,
        "iou_10m": buffered_iou(cons, own_roads, r=10.0) if len(cons) else 0.0,
        "chamfer_precision_m": chamfer[0], "chamfer_recall_m": chamfer[1],
        "mean_gw_dist": float(np.mean(dists)), "min_gw_dist": float(np.min(dists)),
        "consensus_len_m": float(cons.geometry.length.sum()) if len(cons) else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipients", type=int, default=20)
    ap.add_argument("--k", type=int, default=15, help="consensus donors per recipient")
    ap.add_argument("--exclusion-radius-m", type=float, default=2000.0)
    ap.add_argument("--utm-zone", type=int, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path("data/benchmarks/consensus_matrix.parquet"))
    args = ap.parse_args()

    pools = load_pools(source=zone_source(args.utm_zone) if args.utm_zone else None)
    blocks, gdf = pools.blocks, pools.blocks_gdf
    source = desire_source("pbf", iso_of(blocks))

    # A recipient needs its OWN footpaths as ground truth, so it must itself be donatable.
    usable = sorted(set(pools.recipients) & set(pools.donors))
    parcel_counts = [float(len(b.parcels)) for b in blocks]
    chosen = evenly_spaced(usable, parcel_counts, args.recipients)
    print(f"  {len(usable):,} recipients with own OSM; running {len(chosen)}", flush=True)

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
    done: set[tuple[str, str]] = set()
    if args.out.exists():
        prior = pd.read_parquet(args.out)
        rows = prior.to_dict("records")
        done = {(str(r["recipient"]), str(r["arm"])) for r in rows}
        print(f"resuming from {args.out}: {len(rows)} rows", flush=True)

    donor_set = set(pools.donors)
    for n, i in enumerate(chosen, 1):
        recipient = blocks[i]
        own = material(recipient)
        if own is None:
            continue
        for arm, radius in (("held_out", args.exclusion_radius_m), ("leaky", 0.0)):
            if (recipient.block_id, arm) in done:
                continue
            eligible = [j for j in exclusion_holdout(gdf, i, radius_m=radius) if j in donor_set]
            ranked = sorted(eligible, key=lambda j: float(np.linalg.norm(
                pools.signatures[blocks[j].block_id] - pools.signatures[recipient.block_id])))
            picked: list[Block] = []
            for j in ranked:
                if len(picked) >= args.k:
                    break
                if material(blocks[j]) is not None:
                    picked.append(blocks[j])
            if len(picked) < 3:
                print(f"  [{n}] {recipient.block_id} {arm}: only {len(picked)} donors, skipping",
                      flush=True)
                continue
            t0 = time.time()
            rows.append(score_recipient(recipient, own, picked, roads_cache, quality,
                                        arm=arm, k=args.k))
            r = rows[-1]
            print(f"  [{n}/{len(chosen)}] {recipient.block_id} {arm:8s} k={len(picked):2d}  "
                  f"perm/own={r['perm_ratio_own']:.3f}  iou10={r['iou_10m']:.3f}  "
                  f"[{time.time()-t0:.1f}s]", flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(args.out)

    df = pd.DataFrame(rows)
    print(f"\nwrote {args.out} ({len(df)} rows)")
    if len(df):
        for arm, g in df.groupby("arm"):
            print(f"  {arm:8s} n={len(g):2d}  perm/own median {g.perm_ratio_own.median():.3f}  "
                  f"perm/direct {g.perm_ratio_direct.median():.3f}  "
                  f"iou10 {g.iou_10m.median():.3f}  "
                  f"chamfer recall {g.chamfer_recall_m.median():.1f}m")


if __name__ == "__main__":
    main()
