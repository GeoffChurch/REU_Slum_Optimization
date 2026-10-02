"""The greedy's Lens A clearing next to SIMP's, same block and budget (D = 0.10): which buildings
each clears (greedy coloured by round, SIMP by its final x), with a zoom on big blocks.

    PYTHONPATH=. pixi run python research/roadless/render_simp.py <id> <plan> <along> [p] [greedy picker]
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from clear import cleared_through, rows_dir  # noqa: E402
from relax import D_LENS, OUT  # noqa: E402


def main(bid: str, plan: str, along: str, power: float, picker: str) -> None:
    [b] = common.build_blocks([bid])
    polys = gpd.GeoSeries(list(b.buildings.outlines))
    w = common.POPULATIONS["area"].weights(np.asarray(b.buildings.outlines))
    cost = w / w.sum()
    g = pd.read_parquet(rows_dir(picker, 0.5, "area", power, along)
                        / f"{bid}.parquet").sort_values("step")
    order = cleared_through(g, int(g.step.max()))
    rnd = np.concatenate([np.full(len(js), st) for st, js in zip(g.step, g.cleared, strict=True)
                          if st >= 1])
    take, left = [], D_LENS + 1e-12                 # the greedy's prefix within the budget
    for j, r in zip(order, rnd, strict=True):
        if cost[j] <= left:
            take.append((j, r))
            left -= cost[j]
    gj = np.array([j for j, _r in take])
    gr = np.array([r for _j, r in take])
    s = pd.read_parquet(OUT / along / plan / f"{bid}_p{power:g}.parquet").iloc[0]
    sj = np.asarray(s.simp_cleared, dtype=int)
    sx = np.asarray(s.simp_x)
    big = len(polys) > 1500
    rows_n = 2 if big else 1
    fig, axes = plt.subplots(rows_n, 2, figsize=(22, 11 * rows_n), squeeze=False)
    cx = float(np.median([q.centroid.x for q in polys.iloc[np.union1d(gj, sj)]]))
    cy = float(np.median([q.centroid.y for q in polys.iloc[np.union1d(gj, sj)]]))
    minx, miny, maxx, maxy = b.boundary.bounds
    half = 0.125 * max(maxx - minx, maxy - miny)
    window = (cx - half, cx + half, cy - half, cy + half)
    both = np.intersect1d(gj, sj)
    for col, (label, js, vals, cmap, vmin, vmax) in enumerate((
            (f"greedy {picker}: {len(gj)} cleared, colour = round", gj, gr, "plasma", 1,
             gr.max()),
            (f"SIMP {plan}: {len(sj)} cleared, colour = final x", sj, sx[sj], "viridis", 0, 1))):
        for row in range(rows_n):
            ax = axes[row, col]
            gpd.GeoSeries([b.boundary]).boundary.plot(ax=ax, color="k", lw=1)
            gpd.GeoSeries(list(b.streets.geometry)).plot(ax=ax, color="r", lw=2.5)
            polys.plot(ax=ax, color="0.8", ec="0.5", lw=0.2)
            gpd.GeoDataFrame(dict(v=vals), geometry=list(polys.iloc[js])).plot(
                ax=ax, column="v", cmap=cmap, ec="k", lw=0.3, vmin=vmin, vmax=vmax)
            ax.set_axis_off()
            ax.set_aspect("equal")
            if row == 1:
                ax.set_xlim(window[0], window[1])
                ax.set_ylim(window[2], window[3])
            elif big:
                ax.add_patch(plt.Rectangle((window[0], window[2]), 2 * half, 2 * half,
                                           fill=False, ec="#2ca02c", lw=2))
        axes[0, col].set_title(label, fontsize=14)
    gperm = float(np.interp(D_LENS, g.sort_values("D").D, g.sort_values("D").perm))
    fig.suptitle(f"{bid}, {along}, J_{power:g} at D {D_LENS:g}: greedy {gperm:.4f} vs SIMP "
                 f"{s.simp_perm:.4f}; {len(both)} buildings in both", fontsize=16)
    fig.tight_layout()
    out = HERE / f"simp_vs_greedy_{bid}_{along}_{plan}.png"
    fig.savefig(out, dpi=80)
    print(out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 2.0,
         sys.argv[5] if len(sys.argv) > 5 else "S0.01cat")
