"""Where does the flow go, and how did the greedy's 10% clearing reroute it? Per cell, the total
current: the sum over headings of |flux| along that heading's edges (the walkers passing through
it), solved under each run's own scoring conductance before and after its clearing. Unlike the
straight-run gain (opened_map), the current change is global: a cleared gap that links into a
corridor lights the whole route it feeds, out to the street, and dims the routes it relieves.

    CUDA_PATH=/usr PYTHONPATH=. pixi run python research/roadless/current_map.py <block id> [picker] [alongs,] [cpu|gpu]
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402
from clear import cleared_through, rows_dir  # noqa: E402

D_LENS = 0.10
H = 0.5


def current(sc: common.Scorer, open_: np.ndarray, p: lifted.Params) -> np.ndarray:
    """(ny, nx) host: per open cell, half the |flux| of every along edge touching it, summed
    over headings (turning edges move no one in space). Zero elsewhere."""
    xp = p.solver.xp
    g = sc.grid
    sol = lifted.solve(g, open_, sc.f, p, rtol=1e-5, host=False)
    nf = int((open_ > 0).sum())
    uk = xp.zeros((nf, p.K))
    uk[sol.unk_cell >= 0] = sol.u.reshape(-1, p.K)
    S = xp.zeros(nf)
    for k, a, b, w in lifted.along_edges(g, open_, p):
        F = xp.abs(w * (uk[a, k] - uk[b, k]))
        S += 0.5 * (xp.bincount(a, F, minlength=nf) + xp.bincount(b, F, minlength=nf))
    out = np.zeros(open_.shape)
    out[open_ > 0] = p.solver.to_host(S)
    return out


def main(bid: str, picker: str, alongs: list[str], device: str) -> None:
    [b] = common.build_blocks([bid])
    scans, solver = lifted.scans_of(device), lifted.solver_of(device)
    base = lifted.Params(3.0, 8, solver=solver)
    sc = common.Scorer(b, H, base, population=common.POPULATIONS["area"])
    g, polys = sc.grid, sc.polys
    lab = g.label_sub(polys)
    ext = (g.x0 - g.h / 2, g.x0 + (g.inside.shape[1] - 0.5) * g.h,
           g.y0 - g.h / 2, g.y0 + (g.inside.shape[0] - 0.5) * g.h)
    big = len(polys) > 1500
    fig, axes = plt.subplots(len(alongs), 4, figsize=(44, 11 * len(alongs)), squeeze=False)
    window = None
    for row, along in enumerate(alongs):
        p = dataclasses.replace(base, along=lifted.along_of(along.split("@")[0], scans))
        rows = pd.read_parquet(rows_dir(picker, H, "area", 2.0, along)
                               / f"{bid}.parquet").sort_values("step")
        stop = rows[rows.D >= D_LENS - 1e-9].iloc[0]
        cl = cleared_through(rows, int(stop.step))
        after = (g.isub & (~g.bsub | np.isin(lab, cl))).mean(axis=-1)
        c0, c1 = current(sc, g.ff0, p), current(sc, after, p)
        if window is None:
            cx = np.median([q.centroid.x for q in polys[cl]])
            cy = np.median([q.centroid.y for q in polys[cl]])
            minx, miny, maxx, maxy = b.boundary.bounds
            half = 0.125 * max(maxx - minx, maxy - miny)
            window = (cx - half, cx + half, cy - half, cy + half)
        on = (c0 > 0) | (c1 > 0)
        hi = np.quantile(np.maximum(c0, c1)[on], 0.995)
        d = c1 - c0
        dl = np.quantile(np.abs(d[on]), 0.995)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.log2(np.where(c0 > 0, c1 / c0, np.nan))
        panels = [
            (c0, "current before", dict(cmap="magma", norm=mcolors.LogNorm(hi * 1e-4, hi))),
            (c1, "current after", dict(cmap="magma", norm=mcolors.LogNorm(hi * 1e-4, hi))),
            (d, "after - before (symlog)",
             dict(cmap="RdBu_r", norm=mcolors.SymLogNorm(dl * 1e-3, vmin=-dl, vmax=dl))),
            (ratio, "log2(after / before) on space open before",
             dict(cmap="RdBu_r", vmin=-3, vmax=3)),
        ]
        for col, (v, label, kw) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(np.where(on, v, np.nan), origin="lower", extent=ext,
                           interpolation="nearest", **kw)
            fig.colorbar(im, ax=ax, shrink=0.5)
            gpd.GeoSeries(list(np.delete(polys, cl))).plot(ax=ax, color="0.55", lw=0)
            gpd.GeoSeries(list(polys[cl])).boundary.plot(ax=ax, color="#2ca02c", lw=0.8)
            gpd.GeoSeries([b.boundary]).boundary.plot(ax=ax, color="k", lw=1)
            gpd.GeoSeries(list(b.streets.geometry)).plot(ax=ax, color="k", lw=2)
            ax.set_axis_off()
            ax.set_aspect("equal")
            if big:
                ax.set_xlim(window[0], window[1])
                ax.set_ylim(window[2], window[3])
            ax.set_title(f"{along}: {len(cl)} cleared (green outline), D {stop.D:.3f}\n{label}",
                         fontsize=13)
        print(f"{along}: total current {c0.sum():.4g} -> {c1.sum():.4g}", flush=True)
    fig.suptitle(f"{bid}, greedy {picker}: current (sum over headings of |flux|) under each "
                 f"run's own conductance, before and after its 10% clearing", fontsize=16)
    fig.tight_layout()
    out = HERE / f"current_{bid}_{picker}_h{H:g}.png"
    fig.savefig(out, dpi=80)
    print(out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "S0.01cat",
         sys.argv[3].split(",") if len(sys.argv) > 3 else ["uni", "ss100k2"],
         sys.argv[4] if len(sys.argv) > 4 else "gpu")
