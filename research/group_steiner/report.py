"""Solver vs topology vs the lineup on topology's objective, over the blocks both studies finished."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TOPO = HERE / "rows"


def load(d: Path) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))], ignore_index=True)


g = load(HERE / "rows_small")
t = load(TOPO)
g["best"] = g["opt"].where(g["opt"].notna(), g["ub"])
g["proven"] = g["opt"].notna()
g["gap"] = (g["ub"] - g["lb"]) / g["lb"]
ok = g[g.get("error").isna()] if "error" in g else g
print(f"solver: {len(g)} blocks, {len(ok)} solved, {g['error'].notna().sum() if 'error' in g else 0} "
      f"infeasible/errors; proven optimal {int(ok.proven.sum())}; "
      f"heuristic gap median {ok.gap.median():.2%} p90 {ok.gap.quantile(.9):.2%} max {ok.gap.max():.2%}")
print(f"  time: median {ok.total_s.median():.0f}s p90 {ok.total_s.quantile(.9):.0f}s "
      f"max {ok.total_s.max():.0f}s; unproven max gap "
      f"{((ok.best - ok.lb) / ok.lb)[~ok.proven].max() if (~ok.proven).any() else 0:.2%}")
print(f"  heuristic alone (SPH+improve) vs optimum: median +{(ok.ub_improve / ok.best - 1).median():.2%}")
print(f"  canonical check: universal access reached {int(ok.canon_reached.sum())}/{len(ok)}; "
      f"canon access_m / tree_m median {(ok.canon_access_m / ok.tree_m).median():.4f}")

t["alpha"] = t.arm.str.extract(r"topology_a([\d.]+)_s")[0]
topo = t[t.alpha.notna()].copy()
topo_ok = topo[topo.error.fillna("") == ""]
fails = topo.groupby("block").error.apply(lambda e: (e.fillna("") != "").all())
print(f"\ntopology: {topo.block.nunique()} blocks; every run failed/timed out on {int(fails.sum())}")
agg = topo_ok.groupby(["block", "alpha"]).total_m.agg(["min", "median"]).unstack("alpha")
agg.columns = [f"{s}_{a}" for s, a in agg.columns]
ship = topo_ok[topo_ok.arm == "topology_a2_s0"].set_index("block").total_m
secs = topo.groupby(["block", "alpha"]).secs.sum().unstack("alpha")
j = ok.set_index("block")[["best", "proven", "lb", "n_parcels", "total_s"]].join(
    agg, how="inner").join(ship.rename("shipped"), how="left")
print(f"blocks with both: {len(j)}")
rows = [("shipped (a2, seed 0)", j["shipped"])]
for a in ("2", "10", "100"):
    rows.append((f"a{a} median of 10", j[f"median_{a}"]))
    rows.append((f"a{a} best of 10", j[f"min_{a}"]))
rows.append(("best of all 30", j[[f"min_{a}" for a in ("2", "10", "100")]].min(axis=1)))
for name, col in rows:
    ex = col / j["best"] - 1
    print(f"  {name:22s} excess over optimum: median {ex.median():+.2%} p90 {ex.quantile(.9):+.2%} "
          f"max {ex.max():+.2%}; ties optimum (<0.1%) on {int((ex < 1e-3).sum())}/{ex.notna().sum()}; "
          f"beats solver on {int((ex < -1e-6).sum())}")
tt = secs.loc[j.index]
print(f"  time per block: topology 30 runs median {tt.sum(axis=1).median():.0f}s "
      f"(one run median {(tt.sum(axis=1) / 30).median():.1f}s); solver median {j.total_s.median():.0f}s")

# every method on the objective: road length and displacement at universal access (canonical)
lin = t[t.alpha.isna()].copy()
lin = pd.concat([lin, ok.assign(arm="group_steiner", reached=ok.canon_reached,
                                access_m=ok.canon_access_m, access_disp=ok.canon_access_disp)
                 [["block", "arm", "reached", "access_m", "access_disp"]],
                 topo_ok[topo_ok.arm == "topology_a2_s0"][["block", "arm", "reached", "access_m",
                                                          "access_disp"]]], ignore_index=True)
both = set(j.index)
lin = lin[lin.block.isin(both)]
ref = lin[lin.arm == "group_steiner"].set_index("block")
print("\nuniversal access, canonical prefix (blocks with both studies):")
for arm, sub in lin.groupby("arm"):
    s = sub.set_index("block")
    r = s.reached.astype(bool)
    m = (s.access_m / ref.access_m.reindex(s.index))[r]
    d = (s.access_disp - ref.access_disp.reindex(s.index))[r]
    print(f"  {arm:38s} reaches on {int(r.sum()):3d}/{len(s)}; road vs solver median "
          f"{m.median():5.2f}x; displacement at access vs solver median {d.median():+.4f} "
          f"(lower on {int((d < 0).sum())})")
