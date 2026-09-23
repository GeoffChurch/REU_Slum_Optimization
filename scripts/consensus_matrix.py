"""Barycenter-consensus benchmark: predict a block's real footpaths from its mapped neighbours.

The single-donor question is answered and the answer is no -- transplant fidelity does not
measurably depend on donor GW distance (beta = -0.18, 95% [-4.09, +3.85]; see
notes/2026-07-28-no-detectable-distance-effect.md), which corroborates the 2026-07-23 finding that
single-donor transplant is Pareto-dominated. What that study ALSO found, and what had never been
tested beyond n=1, is that a weighted CONSENSUS of several similar blocks' real OSM footpaths
reaches ~94% of a recipient's own network. This measures that at scale.

The mechanism is `reblock.transplant.consensus` at its operating point: fit GW+UOT from each
donor's parcel cloud to the recipient's, transport the donor's real footpaths through it, weight
donors by `quality_i * exp(-gw_dist_i / tau)` (quality = the donor's own permeability x (1 -
displacement), tau = median GW distance), buffer the transported networks into a demand field, and
extract a network gap-aware along the recipient's own ChordSubstrate. Everything is then
length-matched to the recipient's OWN footpath length, so "matched budget" means what the real
network actually spent. A footpath is scored as the street `osm_footpaths` would build on it.

TWO ARMS, always run as a pair. With the exclusion radius, donors must be >2 km away; without it,
the nearest donors are admitted. The gap between the arms is the LEAKAGE estimate, and it is not
hypothetical: a median 26.7% of a recipient's nearest 15 donors sit inside 2 km, and for 24.5% of
recipients all 15 do (`scripts/donor_availability.py`). The 94% figure was measured with no
distance constraint at all, so the held-out arm is the one that can be believed.

NOT comparable to that 94% numerically. The GW solve uses eps=0.01, and since the Prop.-2 gradient
factor was fixed (notes/2026-07-27-gw-pot-crossvalidation.md) that is HALF the regularization the
original run actually had. Same mechanism, tighter coupling.

Nor does a re-run reproduce the committed `data/benchmarks/consensus_matrix.parquet`: that ran on
2026-07-28, before the permeability metric was recalibrated (per-road widths, a 7 m buildable
floor, the per-parcel footpath clearance), before displacement became footprint overlap, and on a
pool screened with the kblock building count rather than Open Buildings.

    pixi run python -m scripts.consensus_matrix --recipients 5 --k 15   # pilot
    pixi run python -m scripts.consensus_matrix --out data/benchmarks/consensus_matrix.parquet
"""
from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import geopandas as gpd
import numpy as np
import pandas as pd

from reblock.compare import load_permeability_config
from reblock.contracts import Block, Method
from reblock.data.pools import (
    DonorSkip,
    capetown_pool,
    evenly_spaced,
    fetch_donor_lines,
    iso_of,
    load_pools,
    pbf_footpaths,
    zone_pool,
)
from reblock.data.settlements import exclusion_holdout
from reblock.emit import pct_displaced
from reblock.eval.agreement import buffered_iou, directional_chamfer
from reblock.methods.clearance import ClearanceReblocker
from reblock.methods.desire_lines import DesireLineSource
from reblock.methods.substrates import ChordSubstrate
from reblock.permeability import (
    DEFAULT_ROAD_WIDTH_M,
    EgressContext,
    PermeabilityParams,
    permeability,
    with_width,
)
from reblock.transplant.consensus import (
    ConsensusParams,
    DonorFit,
    extract_consensus,
    fit_donors,
    length_matched_prefix,
)
from reblock.transplant.gw import Arr
from reblock.transplant.operating_points import CONSENSUS, SIGNATURE, TRANSPORT
from reblock.transplant.signature import signature, signature_distance
from reblock.transplant.snap import GapSnap, RoutedSnap
from reblock.transplant.transport import TransportParams, parcel_xy

CONF = Path("conf")


def within_length(roads: gpd.GeoDataFrame, target_m: float) -> gpd.GeoDataFrame:
    """The longest construction-order prefix of `roads` within `target_m` -- how the direct
    baseline is length-matched."""
    cum = roads.geometry.length.cumsum()
    return cast(gpd.GeoDataFrame, roads[cum <= target_m] if target_m > 0 else roads.iloc[:0])


class ConsensusRow(TypedDict):
    """One (recipient, arm) row, exactly as written to the parquet."""

    recipient: str
    arm: str
    k: int
    k_requested: int
    own_len_m: float
    perm_ratio_own: float
    perm_ratio_direct: float
    perm_consensus: float
    perm_own: float
    perm_single: float
    perm_direct: float
    disp_consensus: float
    disp_own: float
    disp_direct: float
    iou_3m: float
    iou_10m: float
    chamfer_precision_m: float
    chamfer_recall_m: float
    mean_gw_dist: float
    min_gw_dist: float
    consensus_len_m: float


@dataclass(frozen=True)
class ConsensusScorer:
    """Everything the benchmark scores with, resolved once."""

    transport: TransportParams
    consensus: ConsensusParams
    single: GapSnap             # how the best single donor's transplant reaches the substrate
    direct: Method              # the from-scratch baseline, truncated to a length prefix
    permeability: PermeabilityParams
    road_width_m: float         # stamped on footpaths and on the single-donor transplant

    def as_roads(self, footpaths: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """A block's own footpaths as roads: a street built along each, as `osm_footpaths`
        proposes them."""
        return with_width(footpaths, self.road_width_m)

    def donor_quality(self, donor: Block, roads: gpd.GeoDataFrame) -> float:
        """`own_perm * (1 - own_disp)` -- how good the donor's real network is ON ITS OWN BLOCK.

        A donor with an excellent network is worth more in the consensus than a close-but-poorly-
        served one, which is why weight is quality x proximity rather than proximity alone.
        """
        perm = permeability(EgressContext.of(donor, self.permeability), roads)
        return float(perm * (1.0 - pct_displaced(roads, donor.buildings)))

    def single_donor(self, recipient: Block, fits: Sequence[DonorFit]) -> gpd.GeoDataFrame:
        """The closest donor's transplant alone, snapped: the single-donor arm."""
        best = min(fits, key=lambda f: f.gw_dist)
        return with_width(self.single.snap(best.transported, recipient), self.road_width_m)

    def direct_roads(self, recipient: Block) -> gpd.GeoDataFrame:
        roads = self.direct.propose(recipient).roads
        assert roads is not None, "the direct baseline always proposes a road frame"
        return roads

    def score(self, recipient: Block, own: gpd.GeoDataFrame, donors: Sequence[Block],
              networks: Mapping[str, gpd.GeoDataFrame], quality: Mapping[str, float], *,
              arm: str, k: int) -> ConsensusRow:
        target = float(own.geometry.length.sum())
        fits = fit_donors(recipient, donors, networks, self.transport)
        consensus = extract_consensus(recipient, fits, quality, self.consensus)
        cons = length_matched_prefix(recipient, consensus, target)
        sing = length_matched_prefix(recipient, self.single_donor(recipient, fits), target)
        direct = within_length(self.direct_roads(recipient), target)

        ctx = EgressContext.of(recipient, self.permeability)
        perm_own, perm_cons = permeability(ctx, own), permeability(ctx, cons)
        perm_dir = permeability(ctx, direct)
        precision_m, recall_m = (directional_chamfer(cons, own) if len(cons)
                                 else (float("nan"), float("nan")))
        dists = [f.gw_dist for f in fits]
        return ConsensusRow(
            recipient=recipient.block_id, arm=arm, k=len(donors), k_requested=k,
            own_len_m=target,
            # The headline the 2026-07-23 study reported as "94% of a block's own OSM".
            perm_ratio_own=perm_cons / perm_own if perm_own > 0 else float("nan"),
            perm_ratio_direct=perm_cons / perm_dir if perm_dir > 0 else float("nan"),
            perm_consensus=perm_cons, perm_own=perm_own,
            perm_single=permeability(ctx, sing), perm_direct=perm_dir,
            disp_consensus=pct_displaced(cons, recipient.buildings),
            disp_own=pct_displaced(own, recipient.buildings),
            disp_direct=pct_displaced(direct, recipient.buildings),
            # Geometric agreement with the ground truth, the prediction branch's own scorer.
            # IoU at two radii because buffers stop overlapping past 2r -- at 3 m it reads 0 for
            # anything more than 6 m off, which a predicted network easily is, so a single radius
            # would report a flat zero and hide the gradient. Chamfer is kept DIRECTIONAL, per its
            # own contract: precision is paths drawn that are not there, recall is real paths
            # missed, and blending them hides which way the prediction fails.
            iou_3m=buffered_iou(cons, own, r=3.0) if len(cons) else 0.0,
            iou_10m=buffered_iou(cons, own, r=10.0) if len(cons) else 0.0,
            chamfer_precision_m=precision_m, chamfer_recall_m=recall_m,
            mean_gw_dist=float(np.mean(dists)), min_gw_dist=float(np.min(dists)),
            consensus_len_m=float(cons.geometry.length.sum()) if len(cons) else 0.0,
        )


def consensus_scorer() -> ConsensusScorer:
    """The benchmark's scorer: the transplant and extraction at their operating points, the
    routed snap for the single-donor arm, and the shipped clearance preset as the baseline."""
    return ConsensusScorer(
        transport=TRANSPORT, consensus=CONSENSUS, single=RoutedSnap(ChordSubstrate()),
        direct=ClearanceReblocker(substrate=ChordSubstrate(), repulsion=0.0, depth_target=2,
                                  max_roads=400, road_width_m=DEFAULT_ROAD_WIDTH_M),
        permeability=load_permeability_config(CONF).params, road_width_m=DEFAULT_ROAD_WIDTH_M)


class Material:
    """Each block's own footpaths as roads, and their quality as a donor -- fetched once per block
    and shared across recipients, arms and rungs."""

    def __init__(self, source: DesireLineSource, scorer: ConsensusScorer) -> None:
        self._source, self._scorer = source, scorer
        self.roads: dict[str, gpd.GeoDataFrame] = {}
        self.quality: dict[str, float] = {}
        self._skipped: set[str] = set()

    def of(self, block: Block) -> gpd.GeoDataFrame | None:
        """The block's footpaths as roads, or None when it has none to give."""
        if block.block_id in self._skipped:
            return None
        if block.block_id not in self.roads:
            fetched = fetch_donor_lines(self._source, block)
            if isinstance(fetched, DonorSkip):
                self._skipped.add(block.block_id)
                return None
            roads = self._scorer.as_roads(fetched)
            self.roads[block.block_id] = roads
            self.quality[block.block_id] = self._scorer.donor_quality(block, roads)
        return self.roads[block.block_id]


def nearest_donors(recipient: Block, eligible: Sequence[int], blocks: Sequence[Block],
                   signatures: Mapping[str, Arr],
                   material: Callable[[Block], gpd.GeoDataFrame | None],
                   k: int) -> list[Block]:
    """Up to `k` eligible donors that carry material, nearest first by shape signature."""
    r_sig = signatures[recipient.block_id]
    ranked = sorted(eligible,
                    key=lambda j: signature_distance(signatures[blocks[j].block_id], r_sig))
    picked: list[Block] = []
    for j in ranked:
        if len(picked) >= k:
            break
        if material(blocks[j]) is not None:
            picked.append(blocks[j])
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipients", type=int, default=20)
    ap.add_argument("--k", type=int, default=15, help="consensus donors per recipient")
    ap.add_argument("--exclusion-radius-m", type=float, default=2000.0)
    ap.add_argument("--utm-zone", type=int, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path("data/benchmarks/consensus_matrix.parquet"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    pools = load_pools(zone_pool(CONF, args.utm_zone) if args.utm_zone else capetown_pool(CONF))
    blocks, gdf = pools.blocks, pools.blocks_gdf
    signatures = {b.block_id: signature(parcel_xy(b), SIGNATURE) for b in blocks}
    scorer = consensus_scorer()
    material = Material(pbf_footpaths(iso_of(blocks)), scorer)

    # A recipient needs its OWN footpaths as ground truth, so it must itself be donatable.
    usable = sorted(set(pools.recipients) & set(pools.donors))
    parcel_counts = [float(len(b.parcels)) for b in blocks]
    chosen = evenly_spaced(usable, parcel_counts, args.recipients)
    print(f"  {len(usable):,} recipients with own OSM; running {len(chosen)}", flush=True)

    rows: list[ConsensusRow] = []
    done: set[tuple[str, str]] = set()
    if args.out.exists():
        # This script's own output, so its records are the `ConsensusRow`s it wrote.
        rows = cast(list[ConsensusRow], pd.read_parquet(args.out).to_dict("records"))
        done = {(r["recipient"], r["arm"]) for r in rows}
        print(f"resuming from {args.out}: {len(rows)} rows", flush=True)

    donor_set = set(pools.donors)
    for n, i in enumerate(chosen, 1):
        recipient = blocks[i]
        own = material.of(recipient)
        if own is None:
            continue
        for arm, radius in (("held_out", args.exclusion_radius_m), ("leaky", 0.0)):
            if (recipient.block_id, arm) in done:
                continue
            eligible = [j for j in exclusion_holdout(gdf, i, radius_m=radius) if j in donor_set]
            picked = nearest_donors(recipient, eligible, blocks, signatures, material.of, args.k)
            if len(picked) < 3:
                print(f"  [{n}] {recipient.block_id} {arm}: only {len(picked)} donors, skipping",
                      flush=True)
                continue
            t0 = time.time()
            row = scorer.score(recipient, own, picked, material.roads, material.quality,
                               arm=arm, k=args.k)
            rows.append(row)
            print(f"  [{n}/{len(chosen)}] {recipient.block_id} {arm:8s} k={len(picked):2d}  "
                  f"perm/own={row['perm_ratio_own']:.3f}  iou10={row['iou_10m']:.3f}  "
                  f"[{time.time()-t0:.1f}s]", flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(args.out)

    df = pd.DataFrame(rows)
    print(f"\nwrote {args.out} ({len(df)} rows)")
    if len(df):
        for arm_name, g in df.groupby("arm"):
            print(f"  {arm_name!s:8s} n={len(g):2d}  "
                  f"perm/own median {g.perm_ratio_own.median():.3f}  "
                  f"perm/direct {g.perm_ratio_direct.median():.3f}  "
                  f"iou10 {g.iou_10m.median():.3f}  "
                  f"chamfer recall {g.chamfer_recall_m.median():.1f}m")


if __name__ == "__main__":
    main()
