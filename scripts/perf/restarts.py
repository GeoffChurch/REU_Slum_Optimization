"""Turn the path-dependence into a resource: best-of-R restarts.

The scatter this method shows between arbitrary choices is large (up to 0.25 burden reduction) and
crucially **bidirectional** -- the shortlist arms beat the exhaustive search about as often as they
lost. A spread that wide around no systematic difference is not only a reliability problem. It is
also unexploited range: if one greedy run is a draw from a wide distribution, R draws and keep the
best is a strictly better estimator of what the method can do, and tier 2 made a draw cheap enough
to take several.

Reads `null_model.json` rather than running anything -- that experiment already produced R=5
independent draws per block (the random-selector seeds), so best-of-R needs no new compute.

## The honest accounting

Selecting the best of R **by burden reduction** and then reporting burden reduction is biased
upward by construction: the maximum of R draws exceeds their mean whether or not the procedure is
any good. Two things keep this honest.

  * **Permeability is reported as an independent check.** Nothing selects on it. If best-of-R lifts
    burden but leaves permeability flat, the "gain" is selection noise being harvested, not a better
    network.
  * **The comparison is cost-matched.** R restarts at k=128 are compared against ONE exact run at
    the wall-clock each actually took, so "better" has to mean better per second, not just better.

Selecting on burden is legitimate here in a way it would not be for a held-out metric: burden
reduction IS this method's declared objective, so picking the best-scoring network of several is
optimization, not overfitting. Permeability is the one that would reveal the difference.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SRC = Path("scripts/perf/null_model.json")
SEEDS = (1, 2, 3, 4, 5)
K = 128


def main() -> None:
    if not SRC.exists():
        print(f"{SRC} not found -- run scripts.perf.null_model first")
        return
    rows = json.loads(SRC.read_text())
    print(f"\n{'=' * 82}\nBEST-OF-R RESTARTS -- {len(rows)} blocks, k={K}, "
          f"draws are the {len(SEEDS)} random-selector seeds\n")
    print("  Selection is on burden_red (the method's objective). perm is NOT selected on and is\n"
          "  the independent check that the lift is a better network, not harvested noise.\n")

    ex_b = np.array([v["exact"]["burden_red"] for v in rows.values()])
    ex_p = np.array([v["exact"]["perm"] for v in rows.values()])
    ex_s = np.array([v["exact"]["secs"] for v in rows.values()])
    one_s = np.array([np.median([v[f"rand-{K}-s{s}"]["secs"] for s in SEEDS])
                      for v in rows.values()])

    print(f"  {'arm':<16}{'burden_red':>12}{'perm':>10}{'secs':>9}{'vs exact: burden':>19}"
          f"{'perm':>9}{'beats exact':>13}")
    print(f"  {'exact':<16}{np.median(ex_b):>12.4f}{np.median(ex_p):>10.4f}"
          f"{np.median(ex_s):>9.1f}{'':>19}{'':>9}{'':>13}")

    for r in range(1, len(SEEDS) + 1):
        sel_b, sel_p = [], []
        for v in rows.values():
            draws = [(v[f"rand-{K}-s{s}"]["burden_red"], v[f"rand-{K}-s{s}"]["perm"])
                     for s in SEEDS[:r]]
            best = max(draws, key=lambda t: t[0])       # select on burden, carry perm along
            sel_b.append(best[0])
            sel_p.append(best[1])
        b, p = np.array(sel_b), np.array(sel_p)
        cost = one_s * r
        print(f"  {f'best-of-{r}':<16}{np.median(b):>12.4f}{np.median(p):>10.4f}"
              f"{np.median(cost):>9.1f}{np.median(b) - np.median(ex_b):>+19.4f}"
              f"{np.median(p) - np.median(ex_p):>+9.4f}{(b > ex_b).sum():>9}/{len(b):<3}")

    # cost-matched: how many restarts fit in one exact run's wall clock?
    fits = float(np.median(ex_s / one_s))
    print(f"\n  One exact run costs the wall clock of {fits:.1f} restarts at k={K}.")
    r_fit = max(1, int(fits))
    sel = []
    for v in rows.values():
        sel.append(max(v[f"rand-{K}-s{s}"]["burden_red"] for s in SEEDS[:min(r_fit, len(SEEDS))]))
    sel_arr = np.array(sel)
    print(f"  COST-MATCHED (best-of-{min(r_fit, len(SEEDS))} vs exact): "
          f"burden {np.median(sel_arr):.4f} vs {np.median(ex_b):.4f} "
          f"({np.median(sel_arr) - np.median(ex_b):+.4f}), "
          f"wins on {(sel_arr > ex_b).sum()}/{len(sel_arr)} blocks")

    # does the lift come with a permeability lift, or is it selection noise?
    r = len(SEEDS)
    lift_b, lift_p = [], []
    for v in rows.values():
        draws = [(v[f"rand-{K}-s{s}"]["burden_red"], v[f"rand-{K}-s{s}"]["perm"])
                 for s in SEEDS[:r]]
        best = max(draws, key=lambda t: t[0])
        lift_b.append(best[0] - np.mean([d[0] for d in draws]))
        lift_p.append(best[1] - np.mean([d[1] for d in draws]))
    print(f"\n  Best-of-{r} minus the MEAN of its own draws:")
    print(f"    burden {np.mean(lift_b):+.4f}   perm {np.mean(lift_p):+.4f}")
    print("    (burden must rise -- it is the max. If perm does NOT rise with it, the selection\n"
          "     is harvesting noise rather than finding a better network.)")


if __name__ == "__main__":
    main()
