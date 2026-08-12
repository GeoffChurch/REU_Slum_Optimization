"""Does capping `max_anchors` buy affordability, and what does it cost the ACCESS objective?

Enumeration is now the binding cost. Tier 2 removed per-candidate scoring, and across the 15-step
region run the candidate set still grew 2.52x (469k -> 1,180k) because uncapped `_anchor_points`
takes every network vertex and each committed road is a boundary-graph path adding tens more. Two
thirds of the 79.6 min is that growth.

`max_anchors > 0` caps exactly this. The reason it cannot simply be switched on is visible in
`_anchor_points` itself (arterial.py:53): the capped branch takes arc-length samples and returns
EARLY, before the vertex loop. Its own docstring says vertices-as-anchors is what makes
"committed-segment endpoints always anchors -> continuations come for free". So the cap does not
bias the candidate mix toward long chords -- it removes continuations outright. A spur can never
complete into a through-road, and crossings cannot planarize into true intersections. For an
objective that is *about* reaching deep parcels, that is a change to what the method can build.

Which of those two effects dominates is not derivable, so it is measured. One variable: the cap.
Every arm runs the same tier-2 selector, so nothing else differs.

  cost      candidates and seconds per STEP. The cap makes the anchor set constant, so the
            prediction is a FLAT candidate count, not merely a lower slope -- a sharp test of
            whether committed-road vertices really are what drives the 2.52x.
  quality   burden reduction and permeability at a MATCHED displacement budget, so a cap cannot
            win by simply spending more road. Permeability is the independent check: it is not
            what the greedy optimizes, so if the cap costs structure rather than noise, it should
            show there first.

The uncapped region baseline (79.6 min, the step 1/8/15 candidate table) is already measured and is
NOT re-run here; this is block scale, where all arms are affordable and the quality question lives.

Prior art: `anchors.py` asked the timing half of this question and never answered it. Its log holds
a header and zero rows -- the first `propose` never returned, so the "66-minute max_anchors=24
observation" is wall-clock-until-killed, not a completed measurement. It also used the exact greedy,
which is the thing tier 2 exists to avoid.
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
from scripts.perf.selectors import FirstOrder

K = 128                       # tier-2 shortlist, held fixed across arms
CAPS = (0, 32, 64, 128, 256)  # 0 == uncapped == the shipped default
N_BLOCKS = 12
MAX_ROADS = 8
DISP = 0.10                   # the displacement budget every arm is truncated to
OUT = Path("scripts/perf/anchor_cap.json")


def main() -> None:
    pools_ = load_pools()
    blocks = pools_.blocks
    counts = [float(len(b.parcels)) for b in blocks]
    sel = [i for i in pools_.recipients if len(blocks[i].parcels) <= 110]

    rows: dict[str, dict[str, dict[str, float | list[int]]]] = {}
    for i in evenly_spaced(sorted(sel), counts, N_BLOCKS):
        b = blocks[i]
        adj = parcel_adjacency(list(b.parcels.geometry), STREET_TOL)
        radii = building_radii(b.building_points)
        n = len(b.parcels)
        b0 = burden(parcel_access_layers(b, None, tol=STREET_TOL, adj=adj, unreached_depth=n + 1))
        rec: dict[str, dict[str, float | list[int]]] = {}
        for cap in CAPS:
            per_step: list[int] = []
            t0 = time.perf_counter()
            r = _greedy_shortlist(b, realizer=SnapToBoundary(), objective="access",
                                  cost="displacement",
                                  half_width_m=DEFAULT_ROAD_WIDTH_M / 2.0, workers=8,
                                  max_roads=MAX_ROADS, max_anchors=cap, selector=FirstOrder(K),
                                  on_step=lambda s, c, nr, acc=per_step: acc.append(c))
            dt = time.perf_counter() - t0
            if r is None or len(r) == 0:
                continue
            pre = prefix_to_displacement(b, r, radii, DISP)
            if len(pre) == 0:
                continue
            b1 = burden(parcel_access_layers(b, pre, tol=STREET_TOL, adj=adj,
                                             unreached_depth=n + 1))
            rec[str(cap)] = {"burden_red": (1.0 - b1 / b0) if b0 > 0 else 0.0,
                             "perm": float(permeability(b, pre)),
                             "road_m": float(pre.geometry.length.sum()),
                             "n_roads": float(len(pre)), "secs": dt, "cand": per_step}
        if len(rec) == len(CAPS):
            rows[b.block_id] = rec
            print(f"  {b.block_id:<22} n={n:<4} "
                  + "  ".join(f"{'uncap' if c == 0 else c}={rec[str(c)]['burden_red']:.4f}"
                              for c in CAPS), flush=True)
            # Written per block, not once at the end: long runs on this machine get killed (four
            # so far, cause unknown) and an end-only write turns a kill into total data loss --
            # the printed lines carry burden_red but not perm, secs or the candidate counts.
            OUT.write_text(json.dumps(rows, indent=1))
    if not rows:
        print("no blocks completed")
        return

    def col(cap: int, key: str) -> np.ndarray:
        return np.array([float(v[str(cap)][key]) for v in rows.values()])  # type: ignore[arg-type]

    ref_b, ref_p, ref_s = col(0, "burden_red"), col(0, "perm"), col(0, "secs")

    print(f"\n{'=' * 96}\nANCHOR CAP -- {len(rows)} blocks, k={K}, max_roads={MAX_ROADS}, "
          f"displacement budget {DISP}\n")
    print(f"  {'max_anchors':<14}{'burden_red':>12}{'perm':>10}{'road_m':>9}{'secs':>8}"
          f"{'vs uncapped: burden':>22}{'perm':>9}{'speedup':>10}{'beats':>8}")
    for cap in CAPS:
        bb, pp, ss = col(cap, "burden_red"), col(cap, "perm"), col(cap, "secs")
        label = "0 (uncapped)" if cap == 0 else str(cap)
        print(f"  {label:<14}{np.median(bb):>12.4f}{np.median(pp):>10.4f}"
              f"{np.median(col(cap, 'road_m')):>9.0f}{np.median(ss):>8.1f}"
              f"{np.median(bb) - np.median(ref_b):>+22.4f}"
              f"{np.median(pp) - np.median(ref_p):>+9.4f}"
              f"{float(np.median(ref_s / ss)):>9.2f}x{(bb > ref_b).sum():>6}/{len(bb):<3}")

    print("\n  CANDIDATE GROWTH -- median candidates at step 1 vs the last step.\n"
          "  The cap makes the anchor set constant, so capped arms should be FLAT (ratio ~1.00).\n"
          "  If uncapped is also flat, committed-road vertices are NOT what drives the region's\n"
          "  2.52x and the cap is aimed at the wrong term.\n")
    print(f"  {'max_anchors':<14}{'step 1':>10}{'last step':>12}{'growth':>9}")
    for cap in CAPS:
        first, last = [], []
        for v in rows.values():
            c = v[str(cap)]["cand"]
            assert isinstance(c, list)
            if c:
                first.append(c[0])
                last.append(c[-1])
        if not first:
            continue
        f, ln = float(np.median(first)), float(np.median(last))
        label = "0 (uncapped)" if cap == 0 else str(cap)
        print(f"  {label:<14}{f:>10,.0f}{ln:>12,.0f}{ln / max(f, 1.0):>8.2f}x")

    print("\n  Quality is the gate: a cap that is fast and worse is not a win. Permeability is\n"
          "  the tell -- it is not selected on, so a burden-only drop is tie-break noise while a\n"
          "  drop in BOTH is the lost-continuations mechanism showing up.")


if __name__ == "__main__":
    main()
