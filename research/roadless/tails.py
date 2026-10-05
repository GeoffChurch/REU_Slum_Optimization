"""Escape-time tail at Lens A (10% displaced): per building u_i relative to baseline -- median,
95th percentile, max -- for the road lineup (from the study rows) and the greedy (re-solved).

    PYTHONPATH=. uv run python research/roadless/tails.py <pop> <power> <workers> [picker]
"""
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, 'research/roadless')
import numpy as np
import pandas as pd
import shapely

import common
import lifted
from clear import cleared_through
from clear import rows_dir as clear_dir
from study import SHORT, _ci, rows_dir

HERE = Path('research/roadless')
_B: dict = {}
_CFG: dict = {}


def one(bid: str) -> dict:
    b = _B[bid]
    g = pd.read_parquet(clear_dir(_CFG["picker"], 0.5, _CFG["pop"], _CFG["power"])
                        / f"{bid}.parquet")
    g = g.sort_values("step")
    stop = g[g.D >= 0.10 - 1e-9].step.iloc[0]
    rem = cleared_through(g, int(stop))
    sc = common.Scorer(b, 0.5, lifted.Params(ell_m=3.0, K=8),
                       population=common.POPULATIONS[_CFG["pop"]])
    gr = sc.grid
    op = (gr.isub & (~gr.bsub | gr.sub_of(shapely.union_all(sc.polys[rem])))).mean(axis=-1)
    u = sc.home_u(op)
    return dict(block=bid, arm="clear_bldg", umed=np.nanmedian(u) / np.nanmedian(sc.u0),
                u95=np.nanpercentile(u, 95) / np.nanpercentile(sc.u0, 95),
                umax=np.nanmax(u) / np.nanmax(sc.u0), P1=1 - sc.J(u, 1) / sc.J(sc.u0, 1),
                P2=sc.perm_p(u, 2.0))


def main(pop: str, power: float, workers: int, picker: str) -> None:
    global _B, _CFG
    _CFG = dict(pop=pop, power=power, picker=picker)
    done = sorted(p.stem for p in clear_dir(picker, 0.5, pop, power).glob("*.parquet"))
    _B = {b.block_id: b for b in common.build_blocks(common.recipients()) if b.block_id in set(done)}
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        G = pd.DataFrame(list(pool.imap_unordered(one, done)))
    lin = pd.concat([pd.read_parquet(p) for p in rows_dir(0.5, 3.0, 8, pop).glob("*.parquet")],
                    ignore_index=True)
    lin["arm"] = lin.arm.map(SHORT)
    A = lin[(lin.budget == 0.10) & (lin.D >= 0.10 - 1e-9) & lin.block.isin(done)]
    A = A.rename(columns={"P_carve": "P1", "P2_carve": "P2"})
    d = pd.concat([A[["block", "arm", "umed", "u95", "umax", "P1", "P2"]], G], ignore_index=True)
    d.to_parquet(HERE / f"tails_{picker}_{pop}_p{power:g}.parquet")
    print(f"{len(done)} blocks, greedy optimized pop={pop} p={power:g}. Lens A, u relative to "
          "baseline (lower is better): median over blocks of the median / p95 / max home; "
          "perm p1, p2")
    for arm, g in d.groupby("arm"):
        print(f"  {arm:13s} {len(g):4d}  umed {g.umed.median():.3f}  u95 {g.u95.median():.3f}  "
              f"umax {g.umax.median():.3f}   P1 {g.P1.median():.3f}  P2 {g.P2.median():.3f}")
    piv = d.pivot_table(index="block", columns="arm", values=["u95", "umax"])
    for col in ("u95", "umax"):
        best_road = piv[col].drop(columns="clear_bldg").min(axis=1)
        s = (piv[col].clear_bldg - best_road).dropna()
        print(f"  greedy {col} minus the best road arm's: {s.median():+.3f} {_ci(s)}, "
              f"greedy better on {np.mean(s < 0):.2f}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]),
         sys.argv[4] if len(sys.argv) > 4 else "M4")
