"""Per-block Pareto frontiers on the 220 blocks, from score_all's rows.

k-access views (k = 0, 1, 2): among arms whose street-first prefix reaches peel depth k + 1, minimize
(road m, displacement). Lens A: among arms reaching the 10% displacement budget, maximize
permeability and minimize road; depth at the budget is reported beside. Only blocks where every
arm in the comparison has a row are used, so every arm is judged on the same blocks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
df = pd.concat([pd.read_parquet(p) for p in (HERE / "score_rows").glob("*.parquet")],
               ignore_index=True)
df = df[df.error.fillna("") == ""]
ARMS = sorted(df.arm.unique())
NAMES = {"greedy_arterial_access_displacement": "greedy_arterial",
         "cycle_native_betweenness_contrast": "cycle_native_desire"}
exclude = set(sys.argv[1:])              # e.g. topology_best to judge on more blocks


def pareto(points: pd.DataFrame, cols: list[str], senses: list[int]) -> set[str]:
    """Arms not strictly dominated (all >=, one >) -- ties are all kept."""
    v = points[cols].to_numpy() * np.array(senses)        # larger is better after sign flip
    keep = set()
    for i, arm in enumerate(points.index):
        dom = ((v >= v[i]).all(axis=1) & (v > v[i]).any(axis=1)).any()
        if not dom:
            keep.add(arm)
    return keep


arms = [a for a in ARMS if a not in exclude]
have = df[df.arm.isin(arms)].groupby("block").arm.nunique()
blocks = have[have == len(arms)].index
d = df[df.block.isin(blocks) & df.arm.isin(arms)].set_index(["block", "arm"])
print(f"{len(blocks)} blocks with every one of {len(arms)} arms "
      f"(excluded: {sorted(exclude) or 'none'})\n")

for k in (0, 1, 2):
    need = d.xs(arms[0], level="arm").depth_before > k + 1
    bl = need[need].index
    tally = {a: 0 for a in arms}
    reach = {a: 0 for a in arms}
    road_ratio: dict[str, list[float]] = {a: [] for a in arms}
    disp_diff: dict[str, list[float]] = {a: [] for a in arms}
    for b in bl:
        pts = d.loc[b]
        ok = pts[pts[f"k{k}_reached"].astype(bool)]
        for a in ok.index:
            reach[a] += 1
        if ok.empty:
            continue
        for a in pareto(ok, [f"k{k}_m", f"k{k}_disp"], [-1, -1]):
            tally[a] += 1
        best_m, best_d = ok[f"k{k}_m"].min(), ok[f"k{k}_disp"].min()
        for a in ok.index:
            road_ratio[a].append(ok.loc[a, f"k{k}_m"] / best_m if best_m > 0 else np.nan)
            disp_diff[a].append(ok.loc[a, f"k{k}_disp"] - best_d)
    print(f"k = {k} (peel depth <= {k + 1}): non-trivial on {len(bl)} of {len(blocks)} blocks")
    for a in sorted(arms, key=lambda a: -tally[a]):
        if reach[a] == 0:
            continue
        print(f"  {NAMES.get(a, a):28s} frontier {tally[a]:3d}  reaches {reach[a]:3d}  "
              f"road / least road median {np.nanmedian(road_ratio[a]):5.2f}  "
              f"displacement over least median {np.nanmedian(disp_diff[a]):+.3f}")
    never = [NAMES.get(a, a) for a in arms if reach[a] == 0]
    print(f"  never reaches: {', '.join(never) or '-'}\n")

tally = {a: 0 for a in arms}
reach = {a: 0 for a in arms}
perm, road, depth = ({a: [] for a in arms} for _ in range(3))
for b in blocks:
    pts = d.loc[b]
    ok = pts[pts.a_reached.fillna(False).astype(bool)]
    for a in ok.index:
        reach[a] += 1
        perm[a].append(ok.loc[a, "a_P"])
        road[a].append(ok.loc[a, "a_m"])
        depth[a].append(ok.loc[a, "a_depth"])
    if ok.empty:
        continue
    for a in pareto(ok, ["a_P", "a_m"], [1, -1]):
        tally[a] += 1
print(f"Lens A (10% displacement): {len(blocks)} blocks; frontier = max permeability, min road")
for a in sorted(arms, key=lambda a: -tally[a]):
    if reach[a] == 0:
        continue
    print(f"  {NAMES.get(a, a):28s} frontier {tally[a]:3d}  reaches {reach[a]:3d}  "
          f"P median {np.median(perm[a]):.3f}  road median {np.median(road[a]):6.0f} m  "
          f"max depth median {np.median(depth[a]):.0f}")
never = [NAMES.get(a, a) for a in arms if reach[a] == 0]
print(f"  never reach the budget: {', '.join(never) or '-'}")

# paired Lens A permeability vs the best-median arm, both at budget
pv = d.a_P.where(d.a_reached.fillna(False).astype(bool)).unstack("arm")
lead = pv.median().idxmax()
print(f"\nLens A permeability, paired vs {NAMES.get(lead, lead)} (both at budget):")
for a in pv.columns:
    if a == lead:
        continue
    x = (pv[a] - pv[lead]).dropna()
    if len(x):
        print(f"  {NAMES.get(a, a):28s} n={len(x):3d} median {x.median():+.4f} better on "
              f"{int((x > 0).sum())}")
bv = d.b_D.where(d.b_reached.fillna(False).astype(bool)).unstack("arm")
print("\nLens B displacement to P* (lower better), median over blocks reaching it:")
for a in bv.median().sort_values().index:
    print(f"  {NAMES.get(a, a):28s} {bv[a].median():.4f}  (reaches {int(bv[a].notna().sum())})")
