"""Where does the flow go, and how did the greedy's 10% clearing reroute it? Per cell, the total
current: the sum over headings of |flux| along that heading's edges (the walkers passing through
it), solved under each run's own scoring conductance before and after its clearing. Unlike the
straight-run gain (opened_map), the current change is global: a cleared gap that links into a
corridor lights the whole route it feeds, out to the street, and dims the routes it relieves.

    CUDA_PATH=/usr PYTHONPATH=. pixi run python research/roadless/current_map.py <block id> <arms,> [cpu|gpu]

arm: g:<picker>:<along> (the greedy's prefix within D 0.10, current under <along>) or
s:<plan>:<p>:<along> (relax.py's SIMP clearing for that plan, power and conductance).
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import NamedTuple

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
from relax import OUT as RELAX_OUT  # noqa: E402

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


class Arm(NamedTuple):
    """One clearing to map: a label, the conductance its current runs under, the buildings."""
    label: str
    along: str
    cleared: np.ndarray


def arm_of(spec: str, bid: str, cost: np.ndarray) -> Arm:
    kind, name, extra, *rest = spec.split(":")
    if kind == "g":
        rows = pd.read_parquet(rows_dir(name, H, "area", 2.0, extra)
                               / f"{bid}.parquet").sort_values("step")
        take, left = [], D_LENS + 1e-12             # the greedy's prefix within the budget
        for j in cleared_through(rows, int(rows.step.max())):
            if cost[j] <= left:
                take.append(j)
                left -= cost[j]
        return Arm(f"greedy {name} ({extra})", extra, np.array(take, dtype=int))
    if kind == "s":
        [along] = rest
        r = pd.read_parquet(RELAX_OUT / along / name
                            / f"{bid}_p{float(extra):g}.parquet").iloc[0]
        return Arm(f"SIMP {name} ({along})", along, np.asarray(r.simp_cleared, dtype=int))
    raise ValueError(f"unknown arm {spec!r}")


def main(bid: str, arms: list[str], device: str) -> None:
    [b] = common.build_blocks([bid])
    scans, solver = lifted.scans_of(device), lifted.solver_of(device)
    base = lifted.Params(3.0, 8, solver=solver)
    sc = common.Scorer(b, H, base, population=common.POPULATIONS["area"])
    g, polys = sc.grid, sc.polys
    lab = g.label_sub(polys)
    cost = sc.w / sc.w.sum()
    ext = (g.x0 - g.h / 2, g.x0 + (g.inside.shape[1] - 0.5) * g.h,
           g.y0 - g.h / 2, g.y0 + (g.inside.shape[0] - 0.5) * g.h)
    arms_ = [arm_of(a, bid, cost) for a in arms]
    allc = np.concatenate([a.cleared for a in arms_])
    cx = float(np.median([q.centroid.x for q in polys[allc]]))
    cy = float(np.median([q.centroid.y for q in polys[allc]]))
    minx, miny, maxx, maxy = b.boundary.bounds
    half = 0.125 * max(maxx - minx, maxy - miny)
    window = (cx - half, cx + half, cy - half, cy + half)
    fig, axes = plt.subplots(len(arms_), 4, figsize=(44, 11 * len(arms_)), squeeze=False)
    base_c: dict[str, np.ndarray] = {}
    for row, arm in enumerate(arms_):
        p = dataclasses.replace(base, along=lifted.along_of(arm.along.split("@")[0], scans))
        if arm.along not in base_c:
            base_c[arm.along] = current(sc, g.ff0, p)
        c0 = base_c[arm.along]
        after = (g.isub & (~g.bsub | np.isin(lab, arm.cleared))).mean(axis=-1)
        c1 = current(sc, after, p)
        on = (c0 > 0) | (c1 > 0)
        hi = np.quantile(np.maximum(c0, c1)[on], 0.995)
        d = c1 - c0
        dl = np.quantile(np.abs(d[on]), 0.995)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.log2(np.where(c0 > 0, c1 / c0, np.nan))
        D = float(cost[arm.cleared].sum())
        panels = [
            (c1, "current after", False, dict(cmap="magma",
                                               norm=mcolors.LogNorm(hi * 1e-4, hi))),
            (d, "after - before (symlog)", False,
             dict(cmap="RdBu_r", norm=mcolors.SymLogNorm(dl * 1e-3, vmin=-dl, vmax=dl))),
            (d, "after - before, zoom", True,
             dict(cmap="RdBu_r", norm=mcolors.SymLogNorm(dl * 1e-3, vmin=-dl, vmax=dl))),
            (ratio, "log2(after / before), zoom", True, dict(cmap="RdBu_r", vmin=-3, vmax=3)),
        ]
        for col, (v, label, zoom, kw) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(np.where(on, v, np.nan), origin="lower", extent=ext,
                           interpolation="nearest", **kw)
            fig.colorbar(im, ax=ax, shrink=0.5)
            gpd.GeoSeries(list(np.delete(polys, arm.cleared))).plot(ax=ax, color="0.55", lw=0)
            gpd.GeoSeries(list(polys[arm.cleared])).boundary.plot(ax=ax, color="#2ca02c",
                                                                  lw=0.8)
            gpd.GeoSeries([b.boundary]).boundary.plot(ax=ax, color="k", lw=1)
            gpd.GeoSeries(list(b.streets.geometry)).plot(ax=ax, color="k", lw=2)
            ax.set_axis_off()
            ax.set_aspect("equal")
            if zoom:
                ax.set_xlim(window[0], window[1])
                ax.set_ylim(window[2], window[3])
            elif col == 1:
                ax.add_patch(plt.Rectangle((window[0], window[2]), 2 * half, 2 * half,
                                           fill=False, ec="#2ca02c", lw=2))
            ax.set_title(f"{arm.label}: {len(arm.cleared)} cleared (green outline), D {D:.3f}"
                         f"\n{label}", fontsize=13)
        print(f"{arm.label}: total current {c0.sum():.4g} -> {c1.sum():.4g}", flush=True)
    fig.suptitle(f"{bid}: current (sum over headings of |flux|) before and after each "
                 f"clearing at D <= {D_LENS:g}", fontsize=16)
    fig.tight_layout()
    out = HERE / f"current_{bid}_{'_'.join(a.replace(':', '-') for a in arms)}.png"
    fig.savefig(out, dpi=80)
    print(out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","), sys.argv[3] if len(sys.argv) > 3 else "gpu")
