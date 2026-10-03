"""SIMP at one budget (relax.py `one` rows) and greedy pickers at the same Lens A budget, paired
per block: perm at D (a greedy's linear between its steps, as picker_compare), the share of
blocks above 0.9 (saturation), median time (SIMP's whole run; a greedy's time to D), and for
each pair b/a the median of b - a [bootstrap CI], mean, share where b is ahead, and worst.

Blocks: `tuning` (the 13 large blocks the SIMP and sightline settings were tuned on) or `held`
(the other large blocks that fit the greedy: 46, of which 44 have every method of the held-out
confirmation; 20023 and 30796 lack greedy S0.005cat). Times are not comparable across machines:
the D 0.05 SIMP rows ran on the cluster's V100 / RTX 8000 cards, the greedies here on an RTX 6000
Ada.

    PYTHONPATH=. pixi run python research/roadless/simp_compare.py <tuning|held> <budget> <name>=<simp|greedy>,<along>,<plan|picker>... <b>/<a>...
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clear import rows_dir  # noqa: E402
from picker_compare import _at  # noqa: E402
from relax import plan_of, rows_of  # noqa: E402
from study import _ci  # noqa: E402

TUNING = ("ZAF.9.3.1_1_1558", "ZAF.9.3.1_1_18895", "ZAF.9.3.1_1_20269", "ZAF.9.3.1_1_20543",
          "ZAF.9.3.1_1_22422", "ZAF.9.3.1_1_23597", "ZAF.9.3.1_1_30848", "ZAF.9.3.1_1_33717",
          "ZAF.9.3.1_1_38616", "ZAF.9.3.1_1_46841", "ZAF.9.3.1_1_5810", "ZAF.9.5.4_1_9712",
          "ZAF.9.5.4_1_9717")
POWER = 2.0


def blocks_of(which: str) -> list[str]:
    large = pd.read_parquet(HERE / "large82_simp_vs_greedy.parquet").block    # the 59 that fit
    return sorted(TUNING) if which == "tuning" else sorted(set(large) - set(TUNING))


def simp(along: str, plan: str, budget: float) -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(p) for p in rows_of(plan_of(plan), along, budget)
                   .glob("*.parquet")], ignore_index=True)
    return d.set_index("block").rename(columns={"simp_perm": "perm"})[["perm", "t"]]


def greedy(along: str, picker: str, budget: float) -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(p) for p in rows_dir(picker, 0.5, "area", POWER, along)
                   .glob("*.parquet")], ignore_index=True)
    return pd.DataFrame({col: {b: _at(gg, col, budget) for b, gg in g.groupby("block")}
                         for col in ("perm", "t")})


def main(which: str, budget: float, specs: list[str], pairs: list[str]) -> None:
    blocks = blocks_of(which)
    per = {}
    for spec in specs:
        name, rest = spec.split("=")
        kind, along, what = rest.split(",")
        d = (simp if kind == "simp" else greedy)(along, what, budget)
        per[name] = d.reindex(blocks).dropna()
    print(f"{which}: {len(blocks)} blocks, Lens A at D {budget:g}")
    for name, d in per.items():
        print(f"  {name:10s} [{len(d):2d}] median {d.perm.median():.3f}  above 0.9 "
              f"{np.mean(d.perm > 0.9):.0%}  median time {d.t.median():.0f} s")
    for pair in pairs:
        b, a = pair.split("/")
        both = per[a].index.intersection(per[b].index)
        s = per[b].perm[both] - per[a].perm[both]
        print(f"  {b} vs {a} [{len(s)}]: median {s.median():+.3f} {_ci(s)}  mean {s.mean():+.3f}"
              f"  ahead {np.mean(s > 0):.0%}  worst {s.min():+.3f}"
              f" ({s.idxmin().split('_')[-1]})  time x{(per[b].t[both] / per[a].t[both]).median():.1f}")


if __name__ == "__main__":
    args = sys.argv[3:]
    main(sys.argv[1], float(sys.argv[2]), [a for a in args if "=" in a],
         [a for a in args if "/" in a and "=" not in a])
