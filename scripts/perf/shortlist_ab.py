"""Does the tier-2 shortlist greedy land where the exact greedy lands -- and how much faster?

The per-step test ("is the exact winner in the top k?") was measured and is the wrong question: the
exact greedy's argmax flips under a 1e-10 perturbation, so the winner is one arbitrary draw from a
set of near-ties and no approximation can reproduce it. See `tie_sensitivity.py`.

The right question is whether the OUTCOME matches, and there is already a calibrated band to judge
it against. `tie_sensitivity.py` perturbed candidate gains by 1e-10 -- a change with no meaning
whatsoever -- and measured what the published numbers did:

    burden_red   spread across seeds: median 0.0000, max 0.1356   (relative median 0.0%, max 15.5%)
    perm         spread across seeds: median 0.0000, max 0.1297
    road_m       spread across seeds: median 15.8,   max 58.7     (relative median 11.2%)

That is this method's own reproducibility. A shortlist whose deviation from exact sits inside that
band is not degrading anything -- it is landing on a different, equally arbitrary tie-break. A
shortlist whose deviation sits OUTSIDE it is a real loss and tier 2 fails.

Same blocks, same `max_roads`, same D=0.10 prefix and the same metrics as that run, so the two
tables are read side by side.
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
from reblock.methods.arterial.shortlist import FirstOrder
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, permeability
from scripts.pair_matrix import evenly_spaced, load_pools

ARMS = [("exact", 0), ("k=512", 512), ("k=128", 128), ("k=32", 32)]
N_BLOCKS = 8
MAX_ROADS = 8
OUT = Path("scripts/perf/shortlist_ab.json")

# the 1e-10 tie-perturbation band from tie_sensitivity.py -- this method's own reproducibility
TIE_BAND = {"burden_red": (0.0000, 0.1356), "perm": (0.0000, 0.1297), "road_m": (15.82, 58.73)}


def main() -> None:
    pools = load_pools()
    blocks = pools.blocks
    counts = [float(len(b.parcels)) for b in blocks]
    sel = [i for i in pools.recipients if len(blocks[i].parcels) <= 110]

    rows: dict[str, dict[str, dict[str, float]]] = {}
    for i in evenly_spaced(sorted(sel), counts, N_BLOCKS):
        b = blocks[i]
        adj = parcel_adjacency(list(b.parcels.geometry), STREET_TOL)
        radii = building_radii(b.building_points)
        n = len(b.parcels)
        b0 = burden(parcel_access_layers(b, None, tol=STREET_TOL, adj=adj, unreached_depth=n + 1))
        rec: dict[str, dict[str, float]] = {}
        for name, k in ARMS:
            t0 = time.perf_counter()
            r = _greedy_shortlist(b, realizer=SnapToBoundary(), objective="access",
                                  cost="displacement",
                                  half_width_m=DEFAULT_ROAD_WIDTH_M / 2.0, workers=8,
                                  max_roads=MAX_ROADS, selector=FirstOrder(k))
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
                         "road_m": float(pre.geometry.length.sum()),
                         "secs": dt}
        if len(rec) == len(ARMS):
            rows[b.block_id] = rec
            print(f"  {b.block_id:<22} n={n:<4} " + "  ".join(
                f"{k}={v['burden_red']:.4f}({v['secs']:.0f}s)" for k, v in rec.items()),
                flush=True)
    OUT.write_text(json.dumps(rows, indent=1))
    if not rows:
        print("no blocks completed")
        return

    print(f"\n{'=' * 82}\nSHORTLIST vs EXACT -- {len(rows)} blocks, max_roads={MAX_ROADS}\n")
    print(f"  {'arm':<8}{'burden_red':>12}{'perm':>9}{'road_m':>9}{'secs':>8}{'speedup':>9}"
          f"{'|d burden| vs exact: median':>30}{'max':>9}")
    ex_secs = np.median([v["exact"]["secs"] for v in rows.values()])
    for name, _k in ARMS:
        br = np.array([v[name]["burden_red"] for v in rows.values()])
        pm = np.array([v[name]["perm"] for v in rows.values()])
        rm = np.array([v[name]["road_m"] for v in rows.values()])
        sc = np.array([v[name]["secs"] for v in rows.values()])
        d = np.abs(br - np.array([v["exact"]["burden_red"] for v in rows.values()]))
        print(f"  {name:<8}{np.median(br):>12.4f}{np.median(pm):>9.4f}{np.median(rm):>9.1f}"
              f"{np.median(sc):>8.1f}{ex_secs / np.median(sc):>8.1f}x"
              f"{np.median(d):>30.4f}{d.max():>9.4f}")

    print("\n  REFERENCE -- the same numbers under a MEANINGLESS 1e-10 gain perturbation")
    print("  (tie_sensitivity.py; a deviation inside this band is tie-breaking, not degradation)")
    for metric, (med, mx) in TIE_BAND.items():
        print(f"    {metric:<12} median {med:.4f}   max {mx:.4f}")

    print("\n  Per-metric deviation from exact, against that band:")
    for metric in ("burden_red", "perm", "road_m"):
        base = np.array([v["exact"][metric] for v in rows.values()])
        for name, _k in ARMS[1:]:
            got = np.array([v[name][metric] for v in rows.values()])
            d = np.abs(got - base)
            band_max = TIE_BAND[metric][1]
            flag = "within" if d.max() <= band_max else "OUTSIDE"
            print(f"    {metric:<11} {name:<7} median {np.median(d):>8.4f}  max {d.max():>8.4f}"
                  f"   [{flag} the 1e-10 band]")

    wins = {name: int(sum(1 for v in rows.values() if v[name]["burden_red"]
                          > v["exact"]["burden_red"])) for name, _ in ARMS[1:]}
    print("\n  blocks where the shortlist BEAT exact on burden reduction: "
          + ", ".join(f"{k} {v}/{len(rows)}" for k, v in wins.items())
          + "\n  (a shortlist cannot beat an exhaustive search on the step it shares; when it does,"
            "\n   that is the greedy's own path-dependence, and it cuts both ways.)")


if __name__ == "__main__":
    main()
