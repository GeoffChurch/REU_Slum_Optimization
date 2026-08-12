"""Does the first-order local gain RANK the exact winner highly enough to shortlist on?

Tier 2 replaces a full 11k-parcel BFS peel per candidate with a local estimate:

    delta_burden ~= sum over parcels the road DIRECTLY FRONTS of (d_i^2 - 1)

one peel per STEP, then an STRtree `dwithin` query per candidate. It ignores the ripple to
neighbours whose depth also drops, so it UNDERSTATES -- fine for ranking, as long as it ranks.

Note the weights are `d^2 - 1`, NOT the `(d - 1)^2` the backlog sketched. The greedy optimizes
`budget.access_burden` = **sum of d^2**, while the REPORTED metric is `eval.access_burden.burden` =
sum of (d-1)^2 / n. Two different functions with confusingly similar names, and this estimate has to
predict the greedy's own argmax, so it must linearize the greedy's objective: a parcel the new road
fronts drops to depth 1 and contributes 1 instead of d_i^2. Parcels already fronting a street score
0 under either form, which is the only case where the two agree.

That is the whole question, and it is not a timing question. A shortlist heuristic is only sound if
the candidate the exact objective would have picked survives the cut. `resistance_greedy` shortlists
on `linearized_gain` at k=6; whether the same k works here is measured, not assumed.

## Why the ranking must happen BEFORE `_snap`

Measured on the real region block (`snap_vs_peel.py`): the peel is 88% of per-candidate cost and
`_snap` 12% -- so tier 2 aims at the right term, correcting the backlog's guess that snapping
dominated. But step 0 enumerates **468,968 candidates** (961 street vertices -> ~C(961,2)), and at
28.3 ms of snapping each that is still 3.5 h over 15 steps with the peel made entirely free. So the
shortlist has to be formed on the UNSNAPPED chord, and the exact pass snaps only the survivors.

That is the approximation this script is really testing. The estimate is scored on the straight
chord while the exact score lands on `_snap`'s boundary-graph path, which hugs the chord but wanders
and is longer. Ranking error therefore comes from two sources, not one, and only an end-to-end
comparison against the exact winner separates "understates uniformly" (harmless -- rank is
preserved) from "understates unevenly" (fatal -- rank is not).

## Method

Run the EXACT greedy, unmodified, and at every step compute the first-order ranking over the same
candidate list. Record where the exact winner lands in it. This measures the thing that matters --
required shortlist depth -- without any of the confounding that an A/B of two diverging trajectories
would introduce (the two runs would take different steps and become incomparable after the first
disagreement).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

import reblock.methods.arterial.engines as engines
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.arterial import GreedyArterialReblocker, SnapToBoundary
from reblock.methods.arterial.primitives import _planarize
from scripts.pair_matrix import evenly_spaced, load_pools

N_BLOCKS = 10
MAX_ROADS = 8
OUT = Path("scripts/perf/first_order_rank.json")


def first_order_gain(chords: list[LineString], weights: np.ndarray, tree: STRtree,
                     half_w: float) -> np.ndarray:
    """Sum of `weights` over the parcels each chord comes within `half_w` of.

    ONE bulk `dwithin` query for every chord at once -- `STRtree.query` over an array returns the
    flat (input_index, tree_index) pair list, and `np.bincount` sums the weights per input. No
    Python loop and no buffering (`dwithin` is the same predicate a buffer-then-intersects would
    test, minus building 469k polygons).
    """
    arr = np.asarray(chords, dtype=object)
    src, tgt = tree.query(arr, predicate="dwithin", distance=half_w)
    return np.bincount(src, weights=weights[tgt], minlength=len(chords))


RADII = (STREET_TOL, 3.0)         # peel seed tolerance vs the road corridor half-width

_CHORDS: list[LineString] = []
_ROWS: list[dict[str, float]] = []
_ORIG_CHORDS = engines._candidate_chords
_ORIG_BEST = engines._best_candidate

# Per-step context, tracked HERE rather than read off `scoring._STEP_STATE`. `_greedy_arterials`
# clears that global in a `finally` BEFORE it calls `_best_candidate`, so a hook on the reduce
# always sees None -- the first version of this script silently recorded zero steps for that reason.
# The committed list is reconstructed from the winners the reduce itself returns.
_BLOCK: object = None
_ADJ: list[set[int]] = []
_TREE: STRtree | None = None
_COMMITTED: list[BaseGeometry] = []
_HALF_W = 3.0


def _chords_hook(anchors: list[tuple[float, float]],
                 targets: list[tuple[float, float]]) -> list[LineString]:
    global _CHORDS
    _CHORDS = _ORIG_CHORDS(anchors, targets)
    return _CHORDS


def _best_hook(results: object) -> tuple[float, BaseGeometry | None]:
    """Let the exact greedy pick, then score the same candidates with the first-order estimate and
    record where the winner sits. `ex.map`/the serial comprehension both preserve candidate order,
    so `results[i]` corresponds to `_CHORDS[i]`."""
    res = list(results)                             # type: ignore[call-overload]
    gain, real = _ORIG_BEST(res)
    blk = _BLOCK
    if real is None or blk is None or _TREE is None or len(res) != len(_CHORDS):
        return gain, real

    # depths under the roads COMMITTED so far -- one peel per step, which is the whole point
    base = _planarize(list(_COMMITTED), blk.crs, 2.0 * _HALF_W)         # type: ignore[attr-defined]
    depths = parcel_access_layers(blk, base if len(base) else None,     # type: ignore[arg-type]
                                  tol=STREET_TOL, adj=_ADJ,
                                  unreached_depth=len(blk.parcels) + 1)  # type: ignore[attr-defined]
    # `.loc[parcel_id]` reindexes the id-indexed Series into the POSITIONAL order of
    # `block.parcels`, which is the order `STRtree` indexes -- the two must agree or the weights
    # land on the wrong parcels.
    order = depths.loc[blk.parcels["parcel_id"]].to_numpy(dtype=float)   # type: ignore[attr-defined]
    weights = order ** 2 - 1.0                    # the greedy's objective is sum d^2 (see above)

    # EVERY candidate achieving the winning gain, not just the one `_best_candidate` returned.
    # A shortlist greedy makes the same move as the exact greedy if ANY optimal candidate survives
    # the cut -- it does not need this particular chord. That distinction is not academic here:
    # thousands of chords snap to the same boundary path (hence the identical gain), and this
    # greedy additionally has exact ties between genuinely DIFFERENT roads
    # (notes/2026-08-09-greedy-arterial-is-near-tie-sensitive.md: 9 candidates, 6 distinct roads,
    # bit-identical gain). Scoring the rank of one arbitrary representative would count its own
    # equals as competitors and overstate the required depth, possibly by orders of magnitude.
    opt = np.array([i for i, (g, r) in enumerate(res) if g == gain and r is not None])
    if not len(opt):
        return gain, real
    lengths = shapely.length(np.asarray(_CHORDS, dtype=object))
    row: dict[str, float] = {"n_cand": float(len(_CHORDS)), "exact_gain": float(gain),
                             "n_optimal": float(len(opt))}
    parts = []
    for radius in RADII:
        est = first_order_gain(_CHORDS, weights, _TREE, radius)
        per_m = est / np.maximum(lengths, 1e-12)
        tag = f"{radius:g}"
        # rank = the shallowest shortlist that still contains SOME optimal candidate: over the
        # optimal set, the fewest competitors any of them sits behind. `rank < k` <=> a shortlist
        # of k preserves this step's choice exactly.
        for kind, sc in (("raw", est), ("per_m", per_m)):
            row[f"rank_{kind}_{tag}"] = float(min(int((sc > sc[i]).sum()) for i in opt))
        row[f"est_{tag}"] = float(est[opt].max())
        parts.append(f"r={tag}: raw {row[f'rank_raw_{tag}']:>6,.0f} "
                     f"per-m {row[f'rank_per_m_{tag}']:>6,.0f}")
    _ROWS.append(row)
    _COMMITTED.append(real)
    print(f"      step {len(_COMMITTED):>2}: {len(_CHORDS):>6,} cand   " + "   ".join(parts),
          flush=True)
    return gain, real


def main() -> None:
    engines._candidate_chords = _chords_hook          # type: ignore[assignment]
    engines._best_candidate = _best_hook              # type: ignore[assignment]
    pools = load_pools()
    blocks = pools.blocks
    counts = [float(len(b.parcels)) for b in blocks]
    sel = [i for i in pools.recipients if len(blocks[i].parcels) <= 110]

    global _BLOCK, _ADJ, _TREE
    by_block: dict[str, list[dict[str, float]]] = {}
    for i in evenly_spaced(sorted(sel), counts, N_BLOCKS):
        b = blocks[i]
        _ROWS.clear()
        _COMMITTED.clear()
        _BLOCK = b
        _ADJ = parcel_adjacency(list(b.parcels.geometry), STREET_TOL)
        _TREE = STRtree(list(b.parcels.geometry))
        print(f"  {b.block_id}  ({len(b.parcels)} parcels)", flush=True)
        GreedyArterialReblocker(realizer=SnapToBoundary(), objective="access", cost="displacement",
                                workers=8, max_roads=MAX_ROADS).propose(b)
        by_block[b.block_id] = list(_ROWS)
    OUT.write_text(json.dumps(by_block, indent=1))

    rows = [r for v in by_block.values() for r in v]
    if not rows:
        print("no steps recorded -- the hooks did not fire")
        return
    nc = np.array([r["n_cand"] for r in rows])
    print(f"\n{'=' * 78}\nFIRST-ORDER RANK OF THE EXACT WINNER -- "
          f"{len(rows)} steps over {len(by_block)} blocks\n")
    print(f"  candidates per step: median {np.median(nc):,.0f}, max {nc.max():,.0f}")
    print("  rank = candidates the ESTIMATE puts strictly ahead of the exact winner, so a\n"
          "  shortlist of k catches it when rank < k.\n")
    print(f"  {'ranking':<20}{'median':>9}{'p90':>9}{'max':>9}"
          f"{'top-8':>8}{'top-32':>8}{'top-128':>9}{'top-512':>9}")
    for radius in RADII:
        tag = f"{radius:g}"
        for kind in ("raw", "per_m"):
            r = np.array([row[f"rank_{kind}_{tag}"] for row in rows])
            name = f"r={tag} {'raw gain' if kind == 'raw' else 'gain / m'}"
            print(f"  {name:<20}{np.median(r):>9,.0f}{np.percentile(r, 90):>9,.0f}"
                  f"{r.max():>9,.0f}{(r < 8).mean():>8.0%}{(r < 32).mean():>8.0%}"
                  f"{(r < 128).mean():>9.0%}{(r < 512).mean():>9.0%}")
    for radius in RADII:
        tag = f"{radius:g}"
        zero = np.array([r[f"est_{tag}"] for r in rows]) == 0.0
        print(f"\n  r={tag}: winner's estimate was ZERO on {zero.sum()}/{len(rows)} steps"
              "   (fronts nothing -- its gain is ALL ripple, invisible to tier 2)")
    # A shortlist is a fixed k, so what matters operationally is the worst step in a block, not the
    # average: one step needing k=400 forces k=400 for the whole run.
    for radius in RADII:
        tag = f"{radius:g}"
        worst = [max(r[f"rank_per_m_{tag}"] for r in v) for v in by_block.values() if v]
        print(f"  r={tag}: worst rank(gain/m) within a block: "
              + ", ".join(f"{w:,.0f}" for w in sorted(worst, reverse=True)))


if __name__ == "__main__":
    main()
