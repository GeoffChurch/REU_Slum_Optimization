"""Score the solver's large-block trees exactly as gaps/large_study.one scores the lineup, then
compare paired with the recorded lineup rows (large_rows) and desire-routed cycle_native
(cycfield/rows, cyc_contrast_g0.5 = the shipped cycle_native_betweenness_contrast).

    pixi run python score_large.py run <workers>
    pixi run python score_large.py report
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GAPS = HERE
sys.path.insert(0, str(GAPS))
OUT = HERE / "large_score_rows"
ARMS = sorted(p.name[len("trees_"):] for p in HERE.glob("trees_*") if p.is_dir())
_TASKS: list = []


def one(task) -> None:
    import large_blocks as LS
    from reblock.compare import lens_prefixes
    from reblock.emit import pct_displaced
    from reblock.permeability import EgressContext, permeability
    bid, arm = task
    out = OUT / f"{bid}__gst_{arm}.parquet"
    tree = HERE / f"trees_{arm}" / f"{bid}.parquet"
    if out.exists() or not tree.exists():
        return
    city, block = LS._BLOCKS[bid]
    roads = gpd.read_parquet(tree)
    t0 = time.time()
    ctx = EgressContext.of(block, LS.PCFG.params)
    lp = lens_prefixes(ctx, roads, LS.PCFG)

    def length(g):
        return float(g.geometry.length.sum()) if g is not None and len(g) else 0.0

    a_d = pct_displaced(lp.displacement, block.buildings)
    row = dict(city=city, block_id=bid, buildings=len(block.buildings), parcels=len(block.parcels),
               arm=f"gst_{arm}", full_m=length(roads), full_D=pct_displaced(roads, block.buildings),
               full_P=permeability(ctx, roads), a_m=length(lp.displacement), a_D=a_d,
               a_P=permeability(ctx, lp.displacement),
               a_reached=bool(a_d >= LS.PCFG.matched_displacement - 1e-9),
               b_m=length(lp.permeability), b_D=pct_displaced(lp.permeability, block.buildings),
               b_reached=bool(lp.reached), total_s=time.time() - t0)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    pd.DataFrame([row]).to_parquet(tmp)
    os.replace(tmp, out)
    print(f"{time.strftime('%H:%M:%S')} {bid} {arm}: {row['total_s']:.0f}s", flush=True)


def run(workers: int) -> None:
    import large_blocks as LS
    OUT.mkdir(exist_ok=True)
    LS.build()
    tasks = [(b, a) for b in LS.by_size()[::-1] for a in ARMS]
    with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=1) as pool:
        list(pool.imap_unordered(one, tasks))
    print("done", flush=True)


def report() -> None:
    d = pd.concat([pd.read_parquet(p) for p in OUT.glob("*.parquet")], ignore_index=True)
    P = d.pivot_table(index="block_id", columns="arm", values="a_P")
    R = d.pivot_table(index="block_id", columns="arm", values="a_reached")
    M = d.pivot_table(index="block_id", columns="arm", values="a_m")
    B = d.pivot_table(index="block_id", columns="arm", values="b_D")
    BR = d.pivot_table(index="block_id", columns="arm", values="b_reached")
    city = d.groupby("block_id").city.first()
    rng = np.random.default_rng(0)

    def ci(x):
        if len(x) < 3:
            return "n/a"
        bs = [np.median(rng.choice(x, len(x))) for _ in range(4000)]
        return f"[{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]"

    refs = ["cycle_native_betweenness_contrast", "cycle_native", "greedy_arterial_access_displacement",
            "resistance_lp", "clearance_looped"]
    want = sys.argv[2:] or [f"gst_{a}" for a in ARMS]
    for arm in want:
        if arm not in P:
            continue
        print(f"\n{arm}: blocks {int(P[arm].notna().sum())}; reaches the Lens A budget on "
              f"{int((R[arm] > 0).sum())}, P* on {int((BR[arm] > 0).sum())}")
        for ref in refs:
            for scope in ("all", "capetown", "nairobi"):
                sel = P.index if scope == "all" else city[city == scope].index
                ok = ((R[arm] > 0) & (R[ref] > 0)).reindex(sel).fillna(False)
                x = (P[arm] - P[ref]).reindex(sel)[ok].dropna().to_numpy()
                m = (M[arm] / M[ref]).reindex(sel)[ok].dropna()
                okb = ((BR[arm] > 0) & (BR[ref] > 0)).reindex(sel).fillna(False)
                y = (B[arm] - B[ref]).reindex(sel)[okb].dropna().to_numpy()
                if not len(x) and not len(y):
                    continue
                print(f"  vs {ref[:26]:26s} {scope:8s} Lens A n={len(x):2d} "
                      f"{np.median(x) if len(x) else np.nan:+.4f} {ci(x)} "
                      f"better {np.mean(x > 0) if len(x) else 0:.0%} road x{m.median():.2f} | "
                      f"Lens B n={len(y):2d} {np.median(y) if len(y) else np.nan:+.4f} {ci(y)} "
                      f"better {np.mean(y < 0) if len(y) else 0:.0%}")


if __name__ == "__main__":
    run(int(sys.argv[2])) if sys.argv[1] == "run" else report()
