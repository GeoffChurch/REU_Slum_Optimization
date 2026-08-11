"""Can the first-order gain actually be COMPUTED for 469k candidates in a reasonable time?

`first_order_rank.py` asks whether the estimate ranks well. This asks whether it ranks FAST. Both
have to hold: an estimate that ranks perfectly but costs 50 ms per candidate buys nothing over the
214 ms peel it replaces.

The implementation is one bulk `STRtree.query(chords, predicate="dwithin")` for the whole candidate
list, then `np.bincount` to sum each chord's fronted-parcel weights -- no Python loop, no buffering.
The risk is memory, not time: the query returns one (chord, parcel) pair per hit, and a chord
spanning an 11k-parcel region can front hundreds of parcels. This measures both, and chunks the
query so peak memory stays bounded regardless of hit count.

Also measures the two costs tier 2 does NOT remove, because they set the floor on what any
per-candidate speedup can achieve at region scale:

  * `_candidate_chords` -- builds and WKT-sorts every chord, once per step
  * anchor growth -- `_anchor_points` uncapped uses every network VERTEX, and each committed road
    is a boundary-graph path contributing tens of new vertices, so the candidate count grows
    quadratically across steps. If enumeration alone dominates, tier 2 needs `max_anchors` beside
    it and neither is sufficient alone.
"""
from __future__ import annotations

import time

import numpy as np
from shapely import STRtree

from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.arterial import _anchor_points, _candidate_chords, _deep_targets
from scripts.perf.snap_vs_peel import region_block_cached

CHUNK = 20_000          # chords per bulk query -- bounds the pair array, not the total work
HALF_W = 3.0


def ranked_gain(chords: list, weights: np.ndarray, tree: STRtree, radius: float
                ) -> tuple[np.ndarray, int]:
    """First-order gain per chord, chunked. Returns (gain, total hit pairs)."""
    out = np.zeros(len(chords))
    hits = 0
    for lo in range(0, len(chords), CHUNK):
        block = np.asarray(chords[lo:lo + CHUNK], dtype=object)
        src, tgt = tree.query(block, predicate="dwithin", distance=radius)
        hits += len(src)
        out[lo:lo + len(block)] = np.bincount(src, weights=weights[tgt], minlength=len(block))
    return out, hits


def main() -> None:
    block = region_block_cached()
    n = len(block.parcels)
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    depths = parcel_access_layers(block, None, tol=STREET_TOL, adj=adj, unreached_depth=n + 1)
    order = depths.loc[block.parcels["parcel_id"]].to_numpy(dtype=float)
    weights = order ** 2 - 1.0
    tree = STRtree(list(block.parcels.geometry))
    print(f"\nregion block: {n:,} parcels   depth max {order.max():.0f}   "
          f"unreached {(order > n).sum():,}\n")

    t0 = time.perf_counter()
    anchors = _anchor_points(list(block.streets.geometry), 32, 0)
    targets = _deep_targets(block, None, 8, adj)
    chords = _candidate_chords(anchors, targets)
    t_enum = time.perf_counter() - t0
    print(f"  enumerate {len(chords):,} chords from {len(anchors):,} anchors: {t_enum:6.1f} s")

    for radius in (STREET_TOL, HALF_W):
        t0 = time.perf_counter()
        gain, hits = ranked_gain(chords, weights, tree, radius)
        dt = time.perf_counter() - t0
        nz = int((gain > 0).sum())
        print(f"  rank at radius {radius:>4.1f} m: {dt:6.1f} s  "
              f"({dt / len(chords) * 1e6:5.1f} us/chord, {hits:,} hits, "
              f"{hits / len(chords):.1f}/chord, {nz:,} chords score > 0)")

    print(f"\n  vs EXACT per-candidate cost measured at 242.5 ms "
          f"-> speedup ~{242.5e-3 / (dt / len(chords)):,.0f}x per candidate\n")

    # What one step would cost end to end under tier 2: enumerate + rank everything, then snap and
    # exactly peel only the shortlist (28.3 ms + 214.2 ms each, measured).
    for k in (32, 128, 512):
        exact_s = k * 0.2425
        print(f"  ONE STEP, shortlist k={k:>4}: {t_enum:5.1f} s enumerate + {dt:5.1f} s rank + "
              f"{exact_s:6.1f} s exact  = {t_enum + dt + exact_s:6.1f} s serial")
    print("\n  (serial figures. The exact tail already parallelizes across the fork pool, and the\n"
          "   rank loop is chunk-independent so it should too -- measured in rank_parallel.py,\n"
          "   because whether it scales depends on shapely releasing the GIL, not on the algebra.)")


if __name__ == "__main__":
    main()
