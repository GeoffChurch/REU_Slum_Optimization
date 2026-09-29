"""Paired lens comparison of solver arms against the lineup on the 220 (score_rows/).

    pixi run python paired.py <arm> [<arm> ...]      (arm names as in score_rows, e.g. gst_2rp200_k1_lam30)
Lens A: permeability at 10% homes displaced, blocks where both arms reach the budget. Lens B: displacement
to P* = 0.60. Median paired difference, 95% bootstrap interval of the median, share of blocks better.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REFS = ["cycle_native_betweenness_contrast", "cycle_native", "greedy_arterial_access_displacement",
        "resistance_lp", "clearance_looped"]
d = pd.concat([pd.read_parquet(p) for p in (HERE / "score_rows").glob("*.parquet")], ignore_index=True)
d = d[d.error.fillna("") == ""]
P = d.pivot_table(index="block", columns="arm", values="a_P")
R = d.pivot_table(index="block", columns="arm", values="a_reached")
M = d.pivot_table(index="block", columns="arm", values="a_m")
B = d.pivot_table(index="block", columns="arm", values="b_D")
BR = d.pivot_table(index="block", columns="arm", values="b_reached")
F = d.pivot_table(index="block", columns="arm", values="total_m")
rng = np.random.default_rng(0)


def ci(x: np.ndarray) -> str:
    if len(x) < 3:
        return "n/a"
    bs = [np.median(rng.choice(x, len(x))) for _ in range(2000)]
    return f"[{np.quantile(bs, .025):+.3f},{np.quantile(bs, .975):+.3f}]"


for arm in sys.argv[1:]:
    print(f"{arm}: whole network median {F[arm].median():.0f} m; reaches the Lens A budget on "
          f"{int((R[arm] > 0).sum())}/{int(R[arm].notna().sum())}, P* on {int((BR[arm] > 0).sum())}")
    for ref in REFS:
        ok = (R[arm] > 0) & (R[ref] > 0)
        x = (P[arm] - P[ref])[ok].dropna().to_numpy()
        m = (M[arm] / M[ref])[ok].dropna()
        okb = (BR[arm] > 0) & (BR[ref] > 0)
        y = (B[arm] - B[ref])[okb].dropna().to_numpy()
        print(f"   vs {ref[:24]:24s} A n={len(x):3d} {np.median(x):+.3f} {ci(x)} road x{m.median():.2f}"
              f" | B n={len(y):3d} {np.median(y):+.4f} {ci(y)} better {np.mean(y < 0):.0%}")
