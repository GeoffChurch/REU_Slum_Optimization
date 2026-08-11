"""Best-of-R restarts over a STOCHASTIC first-order selector -- cost-matched against exact.

Composes the two findings. The ranking earns its place (`null_model.py`: `FirstOrder(128)` lands at
the top of the uniform-random spread, not in the middle), and this greedy's outcome scatters widely
and *bidirectionally* between arbitrary choices. Deterministic top-k banks the first and throws away
the second: every restart is the same network.

`StochasticFirstOrder(k, pool, seed)` draws k from the top `pool` by score, so each run keeps the
ranking's signal but is genuinely independent of the others -- which is what makes best-of-R
possible at all. Tier 2 already made one run ~5x cheaper than exact at block scale, so several runs
fit inside one exact run's wall clock.

## Reading the result

Selecting the best of R by burden reduction and then reporting burden reduction is biased upward by
construction -- the max of R draws exceeds their mean regardless of whether the procedure is any
good. So:

  * **permeability is the independent check.** Nothing selects on it. A burden lift with flat
    permeability means the selection is harvesting noise, not finding a better network.
  * **the comparison is cost-matched.** R restarts are charged their real wall clock against one
    exact run's.

Selecting on burden is optimization rather than overfitting here, because burden reduction IS this
method's declared objective. Permeability is what would expose the difference.

`pool` is swept because it is the whole trade: `pool = k` is deterministic top-k (no diversity, so
restarts are worthless), and `pool` large approaches uniform random (diversity, but the ranking's
signal is thrown away). The useful setting, if there is one, is in between.
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
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, permeability
from scripts.pair_matrix import evenly_spaced, load_pools
from scripts.perf.selectors import FirstOrder, ScoreAll, StochasticFirstOrder
from scripts.perf.shortlist_greedy import greedy_shortlist

K = 128
POOLS = (256, 1024)
R = 4
N_BLOCKS = 8
MAX_ROADS = 8
OUT = Path("scripts/perf/stochastic_restarts.json")


def main() -> None:
    pools_ = load_pools()
    blocks = pools_.blocks
    counts = [float(len(b.parcels)) for b in blocks]
    sel = [i for i in pools_.recipients if len(blocks[i].parcels) <= 110]

    arms = [("exact", ScoreAll()), (f"fo-{K}", FirstOrder(K))]
    for pool in POOLS:
        arms += [(f"sfo-{pool}-r{r}", StochasticFirstOrder(K, pool, r)) for r in range(R)]

    rows: dict[str, dict[str, dict[str, float]]] = {}
    for i in evenly_spaced(sorted(sel), counts, N_BLOCKS):
        b = blocks[i]
        adj = parcel_adjacency(list(b.parcels.geometry), STREET_TOL)
        radii = building_radii(b.building_points)
        n = len(b.parcels)
        b0 = burden(parcel_access_layers(b, None, tol=STREET_TOL, adj=adj, unreached_depth=n + 1))
        rec: dict[str, dict[str, float]] = {}
        for name, selector in arms:
            t0 = time.perf_counter()
            r = greedy_shortlist(b, mode="buildable", objective="access", cost="displacement",
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
        if len(rec) == len(arms):
            rows[b.block_id] = rec
            best = {p: max(rec[f"sfo-{p}-r{r}"]["burden_red"] for r in range(R)) for p in POOLS}
            print(f"  {b.block_id:<22} n={n:<4} exact={rec['exact']['burden_red']:.4f}  "
                  f"fo={rec[f'fo-{K}']['burden_red']:.4f}  "
                  + "  ".join(f"best-of-{R}(pool={p})={v:.4f}" for p, v in best.items()),
                  flush=True)
    OUT.write_text(json.dumps(rows, indent=1))
    if not rows:
        print("no blocks completed")
        return

    ex_b = np.array([v["exact"]["burden_red"] for v in rows.values()])
    ex_p = np.array([v["exact"]["perm"] for v in rows.values()])
    ex_s = np.array([v["exact"]["secs"] for v in rows.values()])

    print(f"\n{'=' * 88}\nSTOCHASTIC RESTARTS -- {len(rows)} blocks, k={K}, R={R}\n")
    print(f"  {'arm':<22}{'burden_red':>12}{'perm':>10}{'secs':>9}"
          f"{'vs exact: burden':>19}{'perm':>9}{'beats exact':>13}")

    def line(label: str, b: np.ndarray, p: np.ndarray, s: np.ndarray) -> None:
        print(f"  {label:<22}{np.median(b):>12.4f}{np.median(p):>10.4f}{np.median(s):>9.1f}"
              f"{np.median(b) - np.median(ex_b):>+19.4f}"
              f"{np.median(p) - np.median(ex_p):>+9.4f}{(b > ex_b).sum():>9}/{len(b):<3}")

    line("exact", ex_b, ex_p, ex_s)
    line(f"fo-{K} (deterministic)", np.array([v[f"fo-{K}"]["burden_red"] for v in rows.values()]),
         np.array([v[f"fo-{K}"]["perm"] for v in rows.values()]),
         np.array([v[f"fo-{K}"]["secs"] for v in rows.values()]))

    for pool in POOLS:
        for r in (1, 2, R):
            sel_b, sel_p, cost = [], [], []
            for v in rows.values():
                draws = [(v[f"sfo-{pool}-r{j}"]["burden_red"], v[f"sfo-{pool}-r{j}"]["perm"])
                         for j in range(r)]
                best = max(draws, key=lambda t: t[0])
                sel_b.append(best[0])
                sel_p.append(best[1])
                cost.append(sum(v[f"sfo-{pool}-r{j}"]["secs"] for j in range(r)))
            line(f"best-of-{r} pool={pool}", np.array(sel_b), np.array(sel_p), np.array(cost))

    print("\n  IS THE LIFT REAL? best-of-R minus the mean of its own draws\n"
          "  (burden must rise -- it is the max. perm is not selected on: if it does not rise\n"
          "   too, the selection is harvesting noise rather than finding a better network.)\n")
    for pool in POOLS:
        lb, lp = [], []
        for v in rows.values():
            draws = [(v[f"sfo-{pool}-r{j}"]["burden_red"], v[f"sfo-{pool}-r{j}"]["perm"])
                     for j in range(R)]
            best = max(draws, key=lambda t: t[0])
            lb.append(best[0] - float(np.mean([d[0] for d in draws])))
            lp.append(best[1] - float(np.mean([d[1] for d in draws])))
        print(f"    pool={pool:<6} burden {np.mean(lb):+.4f}   perm {np.mean(lp):+.4f}")

    one = np.array([np.median([v[f"sfo-{POOLS[0]}-r{j}"]["secs"] for j in range(R)])
                    for v in rows.values()])
    print(f"\n  One exact run costs the wall clock of {float(np.median(ex_s / one)):.1f} restarts.")


if __name__ == "__main__":
    main()
