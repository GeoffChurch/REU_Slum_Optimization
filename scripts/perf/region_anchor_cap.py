"""`max_anchors` at REGION scale -- the only scale where it can pay, and the only one that decides.

`anchor_cap.py` settled block scale: no cap measurably moves burden or permeability (7 of 8 paired
bootstrap CIs span zero at n=12), and every cap costs wall clock -- 0.24x at cap=256, because
uncapped needs 1,272 candidates there and cap=256 enumerates 34,688. At block scale the cap is
dominated and should never be used.

That result does NOT transfer, and assuming it does would repeat exactly the inference error the
handoff flags. Block scale has ~35-50 anchors; the region block has ~961, overwhelmingly network
VERTICES rather than arc-length samples. The cap deletes precisely the vertex family (arterial.py:53
returns early, before the vertex loop), so what it gives up scales with how much of the anchor set
was vertices -- which is the thing that differs by two orders of magnitude between the scales. The
same cap that is pure overhead at block scale cuts step-1 candidates 468,968 -> 34,688 here, and
flattens the 2.52x growth that is two thirds of the 79.6 min.

So: does the region network survive losing continuations?

## Two things this run fixes about its predecessor

The 79.6-min uncapped run (`region_shortlist.py`) reported roads and metres but NOT burden
reduction or permeability, and did not persist the geometry -- so it left no quality baseline and
no way to compare without paying the 80 minutes again. Here every arm's roads are written as WKT
alongside its metrics.

Arms run CHEAPEST FIRST (128, 256, then uncapped). Four background runs on this machine have been
killed mid-flight for reasons still unknown, and the uncapped arm is ~80 of the ~95 minutes; if
this one dies too, it dies holding the capped evidence rather than losing everything. The JSON is
rewritten after each arm for the same reason.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from reblock.budget import building_radii, prefix_to_displacement
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.eval.access_burden import burden
from reblock.methods.arterial import SnapToBoundary
from reblock.methods.arterial.engines import _greedy_shortlist
from reblock.methods.arterial.shortlist import FirstOrder
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, permeability
from scripts.perf.snap_vs_peel import region_block_cached

SHORTLIST = 512        # matches region_shortlist.py, so the uncapped arm IS the 79.6-min baseline
MAX_ROADS = 15
THREADS = 8
WORKERS = 16
DISP = 0.10
CAPS = (128, 256, 0)   # cheapest first; 0 == uncapped == the shipped default, ~80 min of the ~95
OUT = Path("scripts/perf/region_anchor_cap.json")


def main() -> None:
    block = region_block_cached()
    n = len(block.parcels)
    half_w = DEFAULT_ROAD_WIDTH_M / 2.0
    print(f"\nregion block: {n:,} parcels, {len(block.building_points):,} buildings\n", flush=True)

    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    radii = building_radii(block.building_points)
    b0 = burden(parcel_access_layers(block, None, tol=STREET_TOL, adj=adj, unreached_depth=n + 1))
    print(f"  baseline burden {b0:.4f}\n", flush=True)

    out: dict[str, dict[str, object]] = {}
    for cap in CAPS:
        label = "uncapped" if cap == 0 else str(cap)
        print(f"  --- max_anchors={label} ---", flush=True)
        per_step: list[int] = []
        t0 = time.perf_counter()
        last = [t0]

        def tick(step: int, n_cand: int, n_roads: int, acc: list[int] = per_step,
                 mark: list[float] = last, start: float = t0) -> None:
            now = time.perf_counter()
            acc.append(n_cand)
            print(f"    step {step:>2}: {n_cand:>9,} cand  {now - mark[0]:6.1f} s  "
                  f"(total {(now - start) / 60:5.1f} min)", flush=True)
            mark[0] = now

        roads = _greedy_shortlist(block, realizer=SnapToBoundary(), objective="access",
                                  cost="displacement",
                                  half_width_m=half_w, workers=WORKERS, max_roads=MAX_ROADS,
                                  max_anchors=cap, selector=FirstOrder(SHORTLIST, threads=THREADS),
                                  on_step=tick)
        dt = time.perf_counter() - t0
        if roads is None or len(roads) == 0:
            print("    no roads -- skipped", flush=True)
            continue
        pre = prefix_to_displacement(block, roads, radii, DISP)
        if len(pre) == 0:
            print("    empty displacement prefix -- skipped", flush=True)
            continue
        b1 = burden(parcel_access_layers(block, pre, tol=STREET_TOL, adj=adj,
                                         unreached_depth=n + 1))
        red = (1.0 - b1 / b0) if b0 > 0 else 0.0
        perm = float(permeability(block, pre))
        out[label] = {"burden_red": red, "perm": perm, "secs": dt,
                      "road_m": float(pre.geometry.length.sum()), "n_roads": len(pre),
                      "cand": per_step,
                      # persisted so the next comparison never has to re-run 80 minutes
                      "roads_wkt": [g.wkt for g in pre.geometry]}
        print(f"    burden_red {red:.4f}   perm {perm:.4f}   "
              f"{len(pre)} roads, {float(pre.geometry.length.sum()):,.0f} m   "
              f"{dt / 60:.1f} min\n", flush=True)
        OUT.write_text(json.dumps(out, indent=1))

    if not out:
        print("no arms completed")
        return

    print(f"\n{'=' * 92}\nREGION ANCHOR CAP -- {n:,} parcels, shortlist={SHORTLIST}, "
          f"max_roads={MAX_ROADS}, displacement {DISP}\n")
    print(f"  {'max_anchors':<14}{'burden_red':>12}{'perm':>10}{'road_m':>10}{'min':>8}"
          f"{'cand step1':>13}{'last':>11}{'growth':>9}")
    for label, v in out.items():
        cand = v["cand"]
        assert isinstance(cand, list)
        f, ln = (cand[0], cand[-1]) if cand else (0, 0)
        print(f"  {label:<14}{float(v['burden_red']):>12.4f}{float(v['perm']):>10.4f}"  # type: ignore[arg-type]
              f"{float(v['road_m']):>10.0f}{float(v['secs']) / 60:>8.1f}"  # type: ignore[arg-type]
              f"{f:>13,}{ln:>11,}{ln / max(f, 1):>8.2f}x")

    if "uncapped" in out:
        ref = out["uncapped"]
        print("\n  vs uncapped (the shipped default):")
        for label, v in out.items():
            if label == "uncapped":
                continue
            db = float(v["burden_red"]) - float(ref["burden_red"])  # type: ignore[arg-type]
            dp = float(v["perm"]) - float(ref["perm"])              # type: ignore[arg-type]
            sp = float(ref["secs"]) / float(v["secs"])              # type: ignore[arg-type]
            print(f"    {label:<8} burden {db:+.4f}   perm {dp:+.4f}   {sp:.1f}x faster")
        print("\n  n=1 block, so these are differences, not intervals. Block scale put the paired\n"
              "  noise band at roughly +-0.03-0.09 burden; a region delta inside that is not a\n"
              "  finding. What would be a finding is a delta well outside it, or the two metrics\n"
              "  moving together -- permeability is not selected on, so burden and perm dropping\n"
              "  in step is the lost-continuations mechanism rather than tie-break scatter.")


if __name__ == "__main__":
    main()
