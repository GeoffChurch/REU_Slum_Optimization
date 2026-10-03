"""What did a clearing do to the flow? Solved under the arm's own scoring conductance before and
after its clearing, per cell: the FLUX difference (sum over headings of |change in signed flux|:
the rerouted flow, source-free since the injections and the streets are fixed, so a corridor
that gains Q lights up ~Q along its whole length), and the DISSIPATION difference (w du^2 of
every edge, split between its cells: the map sums exactly to the change in P, so it is where the
score's improvement comes from), and the relative change in the ESCAPE TIME from each cell (who is
better off). Dropped: the current difference and the slowness (time per metre for the walkers
in a cell): outside the cleared cells the conductance is fixed, so slowness = current / w and
both say what the flux and dissipation maps already say.

    CUDA_PATH=/usr PYTHONPATH=. pixi run python research/roadless/current_map.py <block id> <arms,> [cpu|gpu] [zoom|full]

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
from relax import D_LENS, plan_of, rows_of  # noqa: E402

H = 0.5


class Flow(NamedTuple):
    """A field's flow, per grid cell (flat, host): phi (K, cells), the signed flux along each
    heading (+v_k), half of every along edge credited to each end; power, the dissipation
    w (du)^2 of every edge (along, turning, to the street), half to each end of an along edge,
    all of a turning edge to its cell, so it sums to P."""
    phi: np.ndarray
    power: np.ndarray
    P: float
    u: np.ndarray        # escape time from the cell: its axes' potentials by angular share


def flow(sc: common.Scorer, open_: np.ndarray, p: lifted.Params) -> Flow:
    xp = p.solver.xp
    g = sc.grid
    sol = lifted.solve(g, open_, sc.f, p, rtol=1e-9, host=False)
    free = xp.asarray(np.flatnonzero(open_ > 0))         # free-cell id -> flat cell
    nf, N = len(free), open_.size
    uk = xp.zeros((nf, p.K))
    uk[sol.unk_cell >= 0] = sol.u.reshape(-1, p.K)
    phi = xp.zeros((p.K, N))
    power = xp.zeros(N)
    _v, _th, m, gap = lifted.axes(p.K)
    for k, a, b, w in lifted.along_edges(g, open_, p):
        du = uk[a, k] - uk[b, k]
        fa, fb = free[a], free[b]
        phi[k] += 0.5 * (xp.bincount(fa, w * du, minlength=N) + xp.bincount(fb, w * du, minlength=N))
        e = w * du * du
        power += 0.5 * (xp.bincount(fa, e, minlength=N) + xp.bincount(fb, e, minlength=N))
    u = xp.zeros(N)
    u[free] = uk @ xp.asarray(m / np.pi)
    vol = xp.asarray(open_.ravel()[np.flatnonzero(open_ > 0)])
    for k in range(p.K):
        k2 = (k + 1) % p.K
        w = vol * (g.h * g.h / (p.ell_m ** 2 * gap[k]))
        power[free] += w * (uk[:, k] - uk[:, k2]) ** 2
    th = p.solver.to_host
    return Flow(phi=th(phi), power=th(power), P=sol.P, u=th(u))


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
        r = pd.read_parquet(rows_of(plan_of(name), along, D_LENS)
                            / f"{bid}_p{float(extra):g}.parquet").iloc[0]
        return Arm(f"SIMP {name} ({along})", along, np.asarray(r.simp_cleared, dtype=int))
    raise ValueError(f"unknown arm {spec!r}")


def main(bid: str, arms: list[str], device: str, zoom: bool) -> None:
    [b] = common.build_blocks([bid])
    scans, solver = lifted.scans_of(device), lifted.solver_of(device)
    base = lifted.Params(3.0, 8, solver=solver)
    sc = common.Scorer(b, H, base, population=common.POPULATIONS["area"])
    g, polys = sc.grid, sc.polys
    lab = g.label_sub(polys)
    cost = sc.w / sc.w.sum()
    shape = g.inside.shape
    ext = (g.x0 - g.h / 2, g.x0 + (shape[1] - 0.5) * g.h,
           g.y0 - g.h / 2, g.y0 + (shape[0] - 0.5) * g.h)
    arms_ = [arm_of(a, bid, cost) for a in arms]
    allc = np.concatenate([a.cleared for a in arms_])
    cx = float(np.median([q.centroid.x for q in polys[allc]]))
    cy = float(np.median([q.centroid.y for q in polys[allc]]))
    minx, miny, maxx, maxy = b.boundary.bounds
    half = 0.125 * max(maxx - minx, maxy - miny)
    window = (cx - half, cx + half, cy - half, cy + half)
    big = zoom and len(polys) > 1500
    fig, axes = plt.subplots(len(arms_), 3, figsize=(36, 12 * len(arms_)), squeeze=False)
    before: dict[str, Flow] = {}
    for row, arm in enumerate(arms_):
        p = dataclasses.replace(base, along=lifted.along_of(arm.along.split("@")[0], scans))
        if arm.along not in before:
            before[arm.along] = flow(sc, g.ff0, p)
        f0 = before[arm.along]
        after = (g.isub & (~g.bsub | np.isin(lab, arm.cleared))).mean(axis=-1)
        f1 = flow(sc, after, p)
        dflux = np.abs(f1.phi - f0.phi).sum(axis=0)
        dpow = f1.power - f0.power
        on = (np.abs(f0.phi).sum(axis=0) > 0) | (np.abs(f1.phi).sum(axis=0) > 0)
        print(f"{arm.label}: P {f0.P:.6g} -> {f1.P:.6g} (dP {f1.P - f0.P:.6g}); sum of the "
              f"dissipation map {f0.power.sum():.6g} -> {f1.power.sum():.6g} (d "
              f"{dpow.sum():.6g})", flush=True)

        def sym(v):
            lim = float(np.quantile(np.abs(v[on]), 0.995))
            return dict(cmap="RdBu_r", norm=mcolors.SymLogNorm(lim * 1e-3, vmin=-lim, vmax=lim))
        hi = float(np.quantile(dflux[on], 0.995))
        was = g.ff0.ravel() > 0                         # open before: u is comparable
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_u = np.where(was & (f0.u > 0), (f1.u - f0.u) / f0.u, np.nan)
        panels = [
            (dflux, "flux difference: sum_k |phi_k after - phi_k before| (linear)",
             dict(cmap="magma", vmin=0, vmax=hi)),
            (dpow, f"dissipation difference (sums to dP = {f1.P - f0.P:.4g}; blue = P falls)",
             sym(dpow)),
            (rel_u, "escape time from here: (after - before) / before (blue = faster out)",
             dict(cmap="RdBu_r", vmin=-0.6, vmax=0.6)),
        ]
        D = float(cost[arm.cleared].sum())
        for col, (v, label, kw) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(np.where(on, v, np.nan).reshape(shape), origin="lower", extent=ext,
                           interpolation="nearest", **kw)
            fig.colorbar(im, ax=ax, shrink=0.5)
            gpd.GeoSeries(list(np.delete(polys, arm.cleared))).plot(ax=ax, color="0.55", lw=0)
            gpd.GeoSeries(list(polys[arm.cleared])).boundary.plot(ax=ax, color="#2ca02c",
                                                                  lw=0.8)
            gpd.GeoSeries([b.boundary]).boundary.plot(ax=ax, color="k", lw=1)
            gpd.GeoSeries(list(b.streets.geometry)).plot(ax=ax, color="k", lw=2)
            ax.set_axis_off()
            ax.set_aspect("equal")
            if big:
                ax.set_xlim(window[0], window[1])
                ax.set_ylim(window[2], window[3])
            ax.set_title(f"{arm.label}: {len(arm.cleared)} cleared (green outline), D {D:.3f}"
                         f"\n{label}", fontsize=13)
    fig.suptitle(f"{bid}: what each clearing at D <= {D_LENS:g} did to the flow"
                 + (" (zoom)" if big else ""), fontsize=16)
    fig.tight_layout()
    out = HERE / (f"flowdiff_{bid}_{'_'.join(a.replace(':', '-') for a in arms)}"
                  + ("" if zoom else "_full") + ".png")
    fig.savefig(out, dpi=80)
    print(out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","), sys.argv[3] if len(sys.argv) > 3 else "gpu",
         {"zoom": True, "full": False}[sys.argv[4] if len(sys.argv) > 4 else "zoom"])
