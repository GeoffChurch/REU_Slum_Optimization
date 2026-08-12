"""Does tier 2's RANKING earn its place, or would any k candidates have done?

This is the control the tier-2 work did not have. Established so far:

  * the exhaustive per-candidate search buys no measurable outcome quality -- every shortlist arm's
    median matched or beat it, and k=128 beat it on 6 of 8 blocks;
  * the exact argmax is not a stable target, flipping under a 1e-10 perturbation;
  * the first-order estimate ranks the exact benefit at Spearman +0.937.

The third fact is the one nobody has stress-tested. If candidate gains are densely near-tied, then
best-of-k on a RANDOM k is already close to best-of-everything by order statistics alone -- no
geometry required. In that case the +0.937 ranking is decoration, the real finding is "this greedy
only ever needed a subsample", and tier 2's two bulk STRtree queries per step are wasted work.

So: identical machinery, identical k, identical per-step peel (`RandomSample` pays for the depths it
ignores, so the timing comparison is not flattered by skipping work the ranking does). The only
difference is which k candidates get scored. Random arms run several seeds, because a single draw
cannot be distinguished from a lucky one.

Read the result as: does FirstOrder(k) land outside the spread of RandomSample(k) across seeds?
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from reblock.budget import building_radii, prefix_to_displacement
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.eval.access_burden import burden
from reblock.methods.arterial import SnapToBoundary
from reblock.methods.arterial.engines import _greedy_shortlist
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, permeability
from scripts.pair_matrix import evenly_spaced, load_pools
from scripts.perf.selectors import CandidateSelector, FirstOrder, RandomSample, ScoreAll

KS = (128, 32)
SEEDS = (1, 2, 3, 4, 5)
N_BLOCKS = 8
MAX_ROADS = 8
OUT = Path("scripts/perf/null_model.json")


def arms() -> list[tuple[str, CandidateSelector]]:
    out: list[tuple[str, CandidateSelector]] = [("exact", ScoreAll())]
    for k in KS:
        out.append((f"fo-{k}", FirstOrder(k)))
        out += [(f"rand-{k}-s{s}", RandomSample(k, s)) for s in SEEDS]
    return out


def main() -> None:
    pools = load_pools()
    blocks = pools.blocks
    counts = [float(len(b.parcels)) for b in blocks]
    sel = [i for i in pools.recipients if len(blocks[i].parcels) <= 110]
    the_arms = arms()

    rows: dict[str, dict[str, dict[str, float]]] = {}
    for i in evenly_spaced(sorted(sel), counts, N_BLOCKS):
        b = blocks[i]
        adj = parcel_adjacency(list(b.parcels.geometry), STREET_TOL)
        radii = building_radii(b.building_points)
        n = len(b.parcels)
        b0 = burden(parcel_access_layers(b, None, tol=STREET_TOL, adj=adj, unreached_depth=n + 1))
        rec: dict[str, dict[str, float]] = {}
        for name, selector in the_arms:
            t0 = time.perf_counter()
            r = _greedy_shortlist(b, realizer=SnapToBoundary(), objective="access",
                                  cost="displacement",
                                  half_width_m=DEFAULT_ROAD_WIDTH_M / 2.0, workers=8,
                                  max_roads=MAX_ROADS, selector=selector)
            dt = time.perf_counter() - t0
            if r is None or len(r) == 0:
                continue
            pre = prefix_to_displacement(b, r, radii, 0.10)
            if len(pre) == 0:
                continue
            b1 = burden(parcel_access_layers(b, pre, tol=STREET_TOL, adj=adj,
                                             unreached_depth=n + 1))
            rec[name] = {"burden_red": (1.0 - b1 / b0) if b0 > 0 else 0.0,
                         "perm": float(permeability(b, pre)),
                         "road_m": float(pre.geometry.length.sum()), "secs": dt}
        if len(rec) == len(the_arms):
            rows[b.block_id] = rec
            fo = rec["fo-128"]["burden_red"]
            rnd = [rec[f"rand-128-s{s}"]["burden_red"] for s in SEEDS]
            print(f"  {b.block_id:<22} n={n:<4} exact={rec['exact']['burden_red']:.4f}  "
                  f"fo-128={fo:.4f}  rand-128={np.mean(rnd):.4f}+-{np.std(rnd):.4f} "
                  f"[{min(rnd):.4f}..{max(rnd):.4f}]", flush=True)
    OUT.write_text(json.dumps(rows, indent=1))
    if not rows:
        print("no blocks completed")
        return

    def col(name: str, metric: str = "burden_red") -> np.ndarray:
        return np.array([v[name][metric] for v in rows.values()])

    print(f"\n{'=' * 84}\nDOES THE RANKING BEAT A COIN FLIP? -- {len(rows)} blocks, "
          f"max_roads={MAX_ROADS}\n")
    print(f"  {'arm':<14}{'burden_red':>12}{'perm':>9}{'road_m':>9}{'secs':>8}"
          f"{'vs exact':>11}{'beats exact':>13}")
    ex = col("exact")

    # `ref` is explicit, not closed over: the random arm's br/pm/rm/sc pool len(SEEDS) draws per
    # block (a DISTRIBUTION, not a point), so its "vs exact"/"beats exact" columns need `ex` tiled
    # to match -- comparing a (blocks * seeds)-length array against a blocks-length `ex` directly
    # is a shape mismatch (numpy refuses to broadcast (40,) against (8,) at n=8 blocks, 5 seeds).
    def line(label: str, br: np.ndarray, pm: np.ndarray, rm: np.ndarray, sc: np.ndarray,
             ref: np.ndarray) -> None:
        print(f"  {label:<14}{np.median(br):>12.4f}{np.median(pm):>9.4f}{np.median(rm):>9.1f}"
              f"{np.median(sc):>8.1f}{np.median(br) - np.median(ref):>+11.4f}"
              f"{(br > ref).sum():>9}/{len(br):<3}")

    line("exact", ex, col("exact", "perm"), col("exact", "road_m"), col("exact", "secs"), ex)
    for k in KS:
        line(f"fo-{k}", col(f"fo-{k}"), col(f"fo-{k}", "perm"), col(f"fo-{k}", "road_m"),
             col(f"fo-{k}", "secs"), ex)
        # pool every (block, seed) draw: the random arm is a DISTRIBUTION, not a point
        br = np.concatenate([col(f"rand-{k}-s{s}") for s in SEEDS])
        pm = np.concatenate([col(f"rand-{k}-s{s}", "perm") for s in SEEDS])
        rm = np.concatenate([col(f"rand-{k}-s{s}", "road_m") for s in SEEDS])
        sc = np.concatenate([col(f"rand-{k}-s{s}", "secs") for s in SEEDS])
        # `ex` tiled len(SEEDS) times matches the seed-major concatenation above exactly (each
        # block's `col(...)` preserves `rows`' iteration order, so every seed-block contributes
        # in the same block order): tile, don't repeat-per-element.
        line(f"rand-{k} (x{len(SEEDS)})", br, pm, rm, sc, np.tile(ex, len(SEEDS)))

    print("\n  PER BLOCK: where does the ranking sit inside the random arm's own spread?\n")
    for k in KS:
        inside = beat = 0
        gaps = []
        for v in rows.values():
            fo = v[f"fo-{k}"]["burden_red"]
            rnd = np.array([v[f"rand-{k}-s{s}"]["burden_red"] for s in SEEDS])
            inside += bool(rnd.min() <= fo <= rnd.max())
            beat += bool(fo > rnd.max())
            gaps.append(fo - rnd.mean())
        print(f"    k={k:<5} ranking inside the random spread on {inside}/{len(rows)} blocks; "
              f"strictly above every random draw on {beat}/{len(rows)}")
        print(f"    {'':10}mean(fo - mean(random)) = {np.mean(gaps):+.4f}   "
              f"median = {np.median(gaps):+.4f}")
    print("\n  If the ranking sits inside the random spread on most blocks, the +0.937\n"
          "  estimate is not what makes the shortlist work -- near-tied gains are, and a\n"
          "  subsample suffices.")


if __name__ == "__main__":
    main()
