"""How load-bearing are the shipped metric's footpath floor (FOOTPATH_EPS) and its length-free
footpath conductance? Sentinel blocks, the six road methods' count prefixes at D 0.05 (NOTES,
"The shipped metric's footpath floor and edge length").

    PYTHONPATH=. uv run python research/roadless/perm_floor.py <road_rescore's scratch parent>
"""
import functools
import glob
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import reblock.mesh as M
from reblock.compare import load_permeability_config
from reblock.permeability import EgressContext, permeability

sys.path.insert(0, "research/roadless")
import common  # noqa: E402

SCR = sys.argv[1]
params = load_permeability_config(common.REPO / "conf").params
ORIG = M._footpath_conductance
pre = pd.concat([pd.read_parquet(f) for f in glob.glob(
    "research/roadless/road_rows/prefixes_D0.02-0.05-0.1-0.15-0.2/*.parquet")])
METHODS = ["clearance", "clearance_looped", "cycle_native", "cycle_native_betweenness_contrast",
           "resistance_lp", "euclidean_grid"]

def length_aware(dist, r_sum, g_walk, eps=M.FOOTPATH_EPS):
    shape = np.maximum(eps, (dist - r_sum) / dist) / dist        # clearance per unit length
    return (g_walk * float(np.median(1.0 / dist)) / float(np.median(shape))) * shape

VARIANTS = {"eps0.02 (shipped)": ORIG, "eps0.005": functools.partial(ORIG, eps=0.005),
            "eps0.08": functools.partial(ORIG, eps=0.08), "length-aware": length_aware}
for bid in ["ZAF.9.5.4_1_9712", "ZAF.9.3.1_1_22422", "ZAF.9.3.1_1_30848", "ZAF.9.3.1_1_5810"]:
    [b] = common.build_blocks([bid])
    nets = {m: gpd.read_parquet(f"{SCR}/road_rescore/networks/{bid}/{m}.ordered.parquet")
            for m in METHODS}
    mlen = {m: int(pre[(pre.block == bid) & (pre.method == m) & (pre["pop"] == "count")
                       & (pre.budget == 0.05)].m.iloc[0]) for m in METHODS}
    stored = {m: float(pre[(pre.block == bid) & (pre.method == m) & (pre["pop"] == "count")
                           & (pre.budget == 0.05)].P_old.iloc[0]) for m in METHODS}
    res = {}
    for name, fn in VARIANTS.items():
        M._footpath_conductance = fn
        ctx = EgressContext.of(b, params)
        if name.startswith("eps0.02"):
            mesh = ctx.mesh
            rs = ctx.radii[mesh.rows] + ctx.radii[mesh.cols]
            shape = (mesh.dist - rs) / mesh.dist
            road = params.g_road_per_m * (7.0 - params.road_margin_m) / 2.0 / mesh.dist
            ratio = road / mesh.footpath_g
            def q(a):
                return " ".join(f"{x:.3g}" for x in np.percentile(a, [5, 25, 50, 75, 95]))
            print(f"{bid}: {len(shape)} edges; clearance < 0.02 (floored) on "
                  f"{(shape < 0.02).mean():.1%}, < 0 on {(shape < 0).mean():.1%}; "
                  f"edge length p5..p95 {q(mesh.dist)} m; road/footpath ratio p5..p95 {q(ratio)}")
        res[name] = {m: permeability(ctx, nets[m].iloc[:mlen[m]]) for m in METHODS}
    M._footpath_conductance = ORIG
    base = np.array([res["eps0.02 (shipped)"][m] for m in METHODS])
    worst = np.abs(base - np.array([stored[m] for m in METHODS])).max()
    print(f"  check vs stored: {worst:.1e}")
    for name in VARIANTS:
        v = np.array([res[name][m] for m in METHODS])
        print(f"  {name:18s} " + " ".join(f"{x:.3f}" for x in v)
              + f"   tau vs shipped {kendalltau(base, v).statistic:+.2f}")
print("methods:", METHODS)
