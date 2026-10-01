"""Search coarse, score fine: does a greedy run on a coarser grid (fast) pick as well as one on the
scoring grid? Re-score the coarse run's clearing on the fine grid at the two steps either side of
D = 0.10 and interpolate (D does not depend on the grid), then pair with the fine run's own score.

    PYTHONPATH=. pixi run python research/roadless/rescore_grid.py <picker> <h coarse> <workers>
"""
from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402
from clear import Clearing, cleared_through, rows_dir  # noqa: E402
from study import _ci  # noqa: E402

H_FINE = 0.5
POWER = 2.0
_B: dict = {}
_CFG: dict = {}


def one(bid: str) -> dict:
    g = pd.read_parquet(rows_dir(_CFG["picker"], _CFG["h"], "area", POWER)
                        / f"{bid}.parquet").sort_values("step")
    hi = int(g[g.D >= 0.10 - 1e-9].step.iloc[0])
    lo = hi - 1
    c = Clearing(_B[bid], H_FINE, lifted.Params(3.0, 8), population=common.POPULATIONS["area"])
    sc = c.sc
    J0 = sc.J(sc.u0, POWER)
    vals = {}
    for st in (lo, hi):
        r = np.zeros(c.n, dtype=bool)
        r[cleared_through(g, st)] = True
        op = c.open_(r)
        sol = lifted.solve(sc.grid, op, sc.f, c.p)
        vals[st] = (1 - (sc.J(sc.home_u_of(sol, op), POWER) / J0) ** (1 / POWER),
                    1 - sol.P / sc.P0)
    D = g.set_index("step").D
    t = (0.10 - D[lo]) / (D[hi] - D[lo])
    out = dict(block=bid, A=(1 - t) * vals[lo][0] + t * vals[hi][0],
               A1=(1 - t) * vals[lo][1] + t * vals[hi][1])
    f = pd.read_parquet(rows_dir(_CFG["picker"], H_FINE, "area", POWER)
                        / f"{bid}.parquet").sort_values("D")
    out.update(fine_A=float(np.interp(0.10, f.D, f.perm)),
               fine_A1=float(np.interp(0.10, f.D, f.perm1)))
    print(f"{bid} done", flush=True)
    return out


def main(picker: str, h: float, workers: int) -> None:
    global _B, _CFG
    _CFG = dict(picker=picker, h=h)
    done = sorted(p.stem for p in rows_dir(picker, h, "area", POWER).glob("*.parquet"))
    fine = {p.stem for p in rows_dir(picker, H_FINE, "area", POWER).glob("*.parquet")}
    ids = [i for i in done if i in fine]
    _B = {b.block_id: b for b in common.build_blocks(ids)}
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        d = pd.DataFrame(list(pool.imap_unordered(one, ids)))
    d.to_parquet(HERE / f"rescore_grid_{picker}_h{h:g}.parquet")
    print(f"\n{len(d)} blocks: {picker} searched at h {h:g}, scored at h {H_FINE:g}, vs searched "
          f"at h {H_FINE:g}; Lens A at D = 0.10")
    for col in ("A", "A1"):
        s = d[col] - d[f"fine_{col}"]
        print(f"  {col:3s} coarse search {d[col].median():.4f}  fine search "
              f"{d[f'fine_{col}'].median():.4f}  diff {s.median():+.4f} {_ci(s)}  coarse "
              f"better {np.mean(s > 0):.2f}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]))
