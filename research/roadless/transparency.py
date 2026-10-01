"""Does a more translucent SEARCH (tension ranked with buildings at lower kappa; scoring unchanged)
clear deeper buildings and open longer counterfactual corridors? Per block and search kappa, at
D = 0.10 (the first step reaching it):
  depth    median over cleared buildings of their distance to the nearest street, as a
           percentile among all the block's buildings (50 = typical, 100 = deepest)
  long     m^2 of previously open space whose best straight line (max over 48 headings of the
           soft line length F + B, kappa 2) grew by >= 30 m
  perm     the run's own score (perm', p 2) at D = 0.10, interpolated

    PYTHONPATH=. pixi run python research/roadless/transparency.py <ids,> <metric> <kappas,>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402
from clear import cleared_through, rows_dir  # noqa: E402

FIELD = lifted.SoftSightline(beta=1.0, kappa=2.0)
GROW_M = 30.0


def best_line(g: lifted.Grid, open_: np.ndarray) -> np.ndarray:
    L = np.zeros(open_.shape)
    for _i, R in FIELD.runs(open_, g.h):
        L = np.maximum(L, R)
    return L


def main(ids: list[str], metric: str, kappas: list[str]) -> None:
    rows = []
    for b in common.build_blocks(ids):
        polys = np.asarray(b.buildings.outlines)
        g = lifted.Grid.of(b.boundary, polys, list(b.streets.geometry), 0.5)
        lab = g.label_sub(polys)
        streets = shapely.union_all(np.asarray(list(b.streets.geometry)))
        depth = shapely.distance(shapely.centroid(polys), streets)
        pct = pd.Series(depth).rank(pct=True).to_numpy() * 100
        was_open = g.ff0 >= 0.5
        L0 = best_line(g, g.ff0)
        for k in kappas:
            along = metric if k == "base" else f"{metric}@{metric.replace('k2', f'k{k}', 1)}"
            gdf = pd.read_parquet(rows_dir("S0.01cat", 0.5, "area", 2.0, along)
                                  / f"{b.block_id}.parquet").sort_values("step")
            stop = int(gdf[gdf.D >= 0.10 - 1e-9].step.iloc[0])
            cl = cleared_through(gdf, stop)
            after = (g.isub & (~g.bsub | np.isin(lab, cl))).mean(axis=-1)
            grew = was_open & (best_line(g, after) - L0 >= GROW_M)
            rows.append(dict(block=b.block_id, n=len(polys), kappa=k, cleared=len(cl),
                             depth_pct=float(np.median(pct[cl])),
                             depth_m=float(np.median(depth[cl])),
                             long_m2=float(grew.sum() * g.h ** 2),
                             perm=float(np.interp(0.10, gdf.sort_values("D").D,
                                                  gdf.sort_values("D").perm))))
            print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(HERE / f"transparency_{metric}.parquet")
    pd.set_option("display.width", 200)
    for col in ("depth_pct", "long_m2", "perm", "cleared"):
        print(f"\n{col}")
        print(d.pivot_table(index=["block", "n"], columns="kappa", values=col)
              [kappas].round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1].split(","), sys.argv[2], sys.argv[3].split(","))
