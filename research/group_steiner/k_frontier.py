"""The k-access frontier (road m, displacement) with the displacement-weighted solver arms.

    pixi run python k_frontier.py <k> [blocks_file]

Counts, per arm, the blocks where it is on the frontier, over blocks where every solver lambda arm
and both topology arms have a row. For topology, names what dominates it where it is not.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
k = int(sys.argv[1])
only = set(Path(sys.argv[2]).read_text().split()) if len(sys.argv) > 2 else None
df = pd.concat([pd.read_parquet(p) for p in (HERE / "score_rows").glob("*.parquet")],
               ignore_index=True)
df = df[df.error.fillna("") == ""]
lam_arms = sorted({a for a in df.arm if a.startswith(f"gst_k{k}_lam")},
                  key=lambda a: float(a.split("lam")[1]))
need = [f"gst_k{k}", *lam_arms, "topology_best", "topology_shipped"]
have = df[df.arm.isin(need)].groupby("block").arm.nunique()
blocks = set(have[have == len(need)].index)
if only is not None:
    blocks &= only
d = df[df.block.isin(blocks)]
m, x = f"k{k}_m", f"k{k}_disp"
tally, reach, dom_by = Counter(), Counter(), Counter()
for b, g in d.groupby("block"):
    ok = g[g[f"k{k}_reached"].astype(bool)].set_index("arm")
    v = -ok[[m, x]].to_numpy()
    for i, a in enumerate(ok.index):
        reach[a] += 1
        better = (v >= v[i]).all(1) & (v > v[i]).any(1)
        if not better.any():
            tally[a] += 1
        elif a.startswith("topology"):
            for j in np.flatnonzero(better):
                dom_by[(a, ok.index[j])] += 1
print(f"k = {k}: {len(blocks)} blocks with every solver arm and both topology arms"
      + (" (restricted to the given blocks)" if only is not None else ""))
for a, n in sorted(reach.items(), key=lambda t: -tally[t[0]]):
    sub = d[(d.arm == a) & d[f"k{k}_reached"].astype(bool)]
    print(f"  {a:38s} frontier {tally[a]:3d} / reaches {n:3d}   road median {sub[m].median():6.0f} m"
          f"   displacement median {sub[x].median():.3f}")
print("\nwhat dominates topology where it is off the frontier (block counts):")
for (a, by), n in sorted(dom_by.items(), key=lambda t: (t[0][0], -t[1])):
    print(f"  {a:18s} by {by:38s} {n}")
