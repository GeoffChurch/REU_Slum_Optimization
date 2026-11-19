"""The routers compared (ears.Router; owner 2026-11-19: "why do we need to discretize the directions
at all?"): Grid8, Pulled, Theta and Eikonal, each run by the crossings study's recorder
(atlas/crossings/crossings.py: its worlds, its rows) on the same worlds at b0 and b20.

Per run (crossings rows: <world>/<plan>/):
  directions: the built ears' centrelines in CHORD_M chords (a line resampled every CHORD_M m of
    its length), each chord's direction mod 180 degrees in the raster's frame, weighted by its
    length; `grid45` the share within AXIS_TOL of a multiple of 45 degrees (uniform: 4 x 2
    AXIS_TOL / 180), `block` the share within AXIS_TOL of the world's own axes (rect 0 / 90,
    tilt crossings.TILT_DEG / + 90; none on a real block).
  junctions: per end of a built ear, the angle (0 -- 90) between the ear's first JCT_M m (a
    chord from the end) and the road it joins at the end's foot (an earlier ear's centreline, or
    the street: the world's boundary; its tangent from TAN_M either side); `right` the share
    within AXIS_TOL of 90.
  crossings: analyze.analyze (4-way crossings at analyze.TOL_M against the uniform and lattice
    nulls), pooled per condition and router.
  objective: L_P and the built length (the ears' centrelines summed) at each of MATCHED_D, the
    run's states interpolated linearly in D (a run that stops short of a D is not read there).
  time: the run's wall time to each matched D (steps' t; the world's build excluded) and per
    ear.

    R=research/roadless/atlas/routers/routers.py       # from the repo root, PYTHONPATH=.:src
    python $R table                                     # results/*.csv and the tables printed
    python $R plot <world> <b> <out.png>                # the four routers side by side
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import NamedTuple, TypeAlias

import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray
from shapely.geometry import LineString

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "crossings"))
import analyze as an  # noqa: E402
import crossings as cx  # noqa: E402
import ears  # noqa: E402

ROWS = cx.ROWS
OUT = HERE / "results"
CHORD_M = 10.0           # m: directions: a centreline's chords this long
AXIS_TOL = 5.0           # degrees
JCT_M = 10.0             # m: junctions: the ear's direction from its end over this much of it
TAN_M = 3.0              # m: junctions: the road's tangent from this far either side of the foot
MATCHED_D = (0.05, 0.10, 0.15)
ROUTERS = ("grid8", "pulled", "theta", "eikonal")
F64: TypeAlias = NDArray[np.float64]


class Meta(NamedTuple):
    world: str
    kind: str                  # block | rect | tilt | same | lattice
    router: str
    b: float
    dir: Path


def meta_of(d: Path) -> Meta | None:
    """The run in `d` if its plan is a paths plan with a router (else None)."""
    w = json.loads((d / "world.json").read_text())
    plan = ears.Plan.of(re.sub(r"\.v[0-9]+$", "", d.name))     # the directory's: a migrated name
    c = plan.candidates
    if not isinstance(c, ears.Paths):
        return None
    world = w["world"]
    kind = world.split(":")[0] if not world.startswith("fabric") else world.split(":")[2]
    return Meta(world=world, kind=kind, router=c.router.name, b=c.bld, dir=d)


def runs() -> list[Meta]:
    out = []
    for p in sorted(ROWS.glob("*/*/states.npz")):
        if (p.parent / "world.json").exists():
            m = meta_of(p.parent)
            if m is not None:
                out.append(m)
    return out


def chords(line: F64, step: float) -> F64:
    """The polyline resampled every `step` m of its length (its end kept): chords' ends."""
    ls = LineString(line)
    s = np.append(np.arange(0.0, ls.length, step), ls.length)
    pts = shapely.line_interpolate_point(ls, s)
    return np.asarray(np.column_stack([shapely.get_x(pts), shapely.get_y(pts)]))


def near_axes(deg: F64, axes: list[float]) -> NDArray[np.bool_]:
    """Per direction (mod 180) whether it lies within AXIS_TOL of one of `axes`."""
    out = np.zeros(len(deg), dtype=bool)
    for a in axes:
        d = np.abs((deg - a + 90.0) % 180.0 - 90.0)
        out |= d <= AXIS_TOL
    return out


def block_axes(kind: str) -> list[float]:
    return {"rect": [0.0, 90.0], "lattice": [], "tilt": [cx.TILT_DEG, cx.TILT_DEG + 90.0]}.get(
        kind, [])


def directions(run: an.Run, kind: str) -> tuple[dict[str, float], F64, F64]:
    """(the run's direction shares, every chord's direction, its length)."""
    deg, w = [], []
    for p in run.paths:
        c = chords(p, CHORD_M)
        d = np.diff(c, axis=0)
        L = np.hypot(*d.T)
        k = L > 1e-9
        deg.append(np.degrees(np.arctan2(d[k, 1], d[k, 0])) % 180.0)
        w.append(L[k])
    D, W = np.concatenate(deg), np.concatenate(w)
    out = dict(grid45=float(W[near_axes(D, [0.0, 45.0, 90.0, 135.0])].sum() / W.sum()))
    ax = block_axes(kind)
    out["block"] = float(W[near_axes(D, ax)].sum() / W.sum()) if ax else np.nan
    return out, D, W


def tangent(line: LineString, s: float) -> F64:
    a = line.interpolate(max(s - TAN_M, 0.0))
    b = line.interpolate(min(s + TAN_M, line.length))
    t = np.array([b.x - a.x, b.y - a.y])
    return np.asarray(t / max(np.hypot(*t), 1e-12))


def junctions(run: an.Run) -> F64:
    """Per end of every built ear joining a road (an earlier ear or the street) the angle (deg,
    0 -- 90) between the ear's first JCT_M m from that end and the road at the end's foot."""
    street = shapely.from_wkt(json.loads((run.dir / "world.json").read_text())["boundary"])
    ring = LineString(street.exterior.coords)
    lines = [LineString(p) for p in run.paths]
    out = []
    for i, p in enumerate(run.paths):
        ls = lines[i]
        for end, s_in in ((p[0], min(JCT_M, ls.length)), (p[-1], max(ls.length - JCT_M, 0.0))):
            k = int(run.ra.road[i][run.ra.pix(end[None])[0]])
            if k < -1:
                continue
            road = ring if k == -1 else lines[k]
            q = ls.interpolate(s_in)
            e = np.array([q.x - end[0], q.y - end[1]])
            if np.hypot(*e) < 1e-9:
                continue
            e /= np.hypot(*e)
            s = road.project(shapely.Point(end))
            t = tangent(road, s)
            out.append(float(np.degrees(np.arccos(min(1.0, abs(float(e @ t)))))))
    return np.asarray(out)


def at_D(steps: pd.DataFrame, col: F64, D: float) -> float:
    """`col` (per state) at D, linear between the states around it; nan past the run's last."""
    d = steps.D.to_numpy()
    if D > d.max() or D < d.min():
        return np.nan
    return float(np.interp(D, d, col))


def objective(d: Path) -> dict[str, float]:
    steps = pd.read_parquet(d / "steps.parquet")
    picks = pd.read_parquet(d / "picks.parquet")
    length = np.concatenate([[0.0], np.cumsum(picks.length.to_numpy())])[:len(steps)]
    out: dict[str, float] = dict(ears=len(picks), D_end=float(steps.D.max()),
                                 t_end=float(steps.t.max()))
    for D in MATCHED_D:
        out[f"LP{D:g}"] = at_D(steps, steps.L_P.to_numpy(), D)
        out[f"len{D:g}"] = at_D(steps, length, D)
        out[f"t{D:g}"] = at_D(steps, steps.t.to_numpy(), D)
    out["t_per_ear"] = float(steps.t.max() / max(len(picks), 1))
    return out


def table() -> None:
    OUT.mkdir(exist_ok=True)
    rows, nulls, dirs = [], {}, []
    for m in runs():
        run = an.load(m.dir, budget=None)
        sh, D, W = directions(run, m.kind)
        jx = junctions(run)
        cr, nl = an.analyze(m.dir, budget=None)
        cond = f"{m.kind} b{m.b:g}"
        rows.append(dict(world=m.world, kind=m.kind, b=m.b, router=m.router, condition=cond,
                         **sh, junctions=len(jx), right=float((jx >= 90 - AXIS_TOL).mean())
                         if len(jx) else np.nan, jct_mean=float(jx.mean()) if len(jx) else np.nan,
                         obs=int(cr[f"obs{an.TOL_M:g}"]), null=float(cr[f"null{an.TOL_M:g}"]),
                         lnull=float(cr[f"lnull{an.TOL_M:g}"]), **objective(m.dir)))
        nulls[(m.world, m.router, m.b)] = (cr, nl)
        hist, _ = np.histogram(D, bins=np.arange(0, 181, 5), weights=W)
        dirs.append(dict(world=m.world, kind=m.kind, b=m.b, router=m.router,
                         **{f"h{a}": float(v) for a, v in zip(range(0, 180, 5), hist / W.sum(),
                                                              strict=True)}))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "runs.csv", index=False, float_format="%.5g")
    pd.DataFrame(dirs).to_csv(OUT / "directions.csv", index=False, float_format="%.4g")
    # crossings pooled per condition and router
    pooled = []
    for (cond, r), g in t.groupby(["condition", "router"]):
        sel = [nulls[(w, r, b)] for w, b in zip(g.world, g.b, strict=True)]
        p = an.pooled([s[1] for s in sel], [s[0] for s in sel])
        pooled.append(dict(condition=cond, router=r, runs=len(g), ears=int(g.ears.sum()),
                           obs=p[f"obs{an.TOL_M:g}"], null=p[f"null{an.TOL_M:g}"],
                           p=p[f"p{an.TOL_M:g}"], lnull=p[f"lnull{an.TOL_M:g}"],
                           lp=p[f"lp{an.TOL_M:g}"],
                           grid45=float(np.average(g.grid45, weights=g.ears)),
                           block=float(np.average(g.block, weights=g.ears)),
                           right=float(np.nanmean(g.right)),
                           jct_mean=float(np.nanmean(g.jct_mean))))
    P = pd.DataFrame(pooled)
    P.to_csv(OUT / "pooled.csv", index=False, float_format="%.4g")
    # paired against grid8 on the same world and b
    base = t[t.router == "grid8"].set_index(["world", "b"])
    pair = []
    for r in ROUTERS[1:]:
        o = t[t.router == r].set_index(["world", "b"])
        common = o.index.intersection(base.index)
        for D in MATCHED_D:
            for col in (f"LP{D:g}", f"len{D:g}", f"t{D:g}"):
                x, y = o.loc[common, col], base.loc[common, col]
                k = x.notna() & y.notna()
                if not k.any():
                    continue
                if col.startswith("t"):
                    ratio = np.log(x[k].to_numpy() / y[k].to_numpy())
                    pair.append(dict(router=r, what=col, n=int(k.sum()),
                                     geo_mean_ratio=float(np.exp(ratio.mean())),
                                     lo=float(np.exp(ratio.min())), hi=float(np.exp(ratio.max()))))
                else:
                    dd = (x[k] - y[k]).to_numpy()
                    se = dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else np.nan
                    pair.append(dict(router=r, what=col, n=int(k.sum()), mean_diff=float(dd.mean()),
                                     se=float(se), wins=int((dd > 0).sum()),
                                     losses=int((dd < 0).sum())))
    Q = pd.DataFrame(pair)
    Q.to_csv(OUT / "paired.csv", index=False, float_format="%.5g")
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.max_rows", 200):
        print(P.round(3).to_string(index=False))
        print(Q.round(4).to_string(index=False))


def plot(world: str, b: float, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ms = {m.router: m for m in runs() if m.world == world and m.b == b}
    fig, axs = plt.subplots(1, len(ROUTERS), figsize=(5 * len(ROUTERS), 5.6), dpi=130)
    for ax, r in zip(axs, ROUTERS, strict=True):
        if r not in ms:
            ax.set_title(f"{r}: no run")
            ax.axis("off")
            continue
        run = an.load(ms[r].dir, budget=None)
        ra = run.ra
        nx, ny = ra.shape
        img = np.ones((ny, nx, 3))
        img[~ra.inside.reshape(nx, ny).T] = (0.97, 0.97, 0.96)
        img[(ra.inside & (ra.bld > 0)).reshape(nx, ny).T] = matplotlib.colors.to_rgb(cx_bld())
        ext = (ra.lo[0], ra.lo[0] + nx * an.PIX, ra.lo[1], ra.lo[1] + ny * an.PIX)
        ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
        n = len(run.paths)
        for i, p in enumerate(run.paths):
            for g in shapely.get_parts(LineString(p).buffer(an.ROAD_W / 2)):
                x, y = g.exterior.xy
                ax.fill(x, y, color="#52514e", alpha=0.95 - 0.5 * i / max(n - 1, 1), lw=0)
        d = ms[r].dir
        ends = pd.read_parquet(d / "ends.parquet") if (d / "ends.parquet").exists() else None
        pairs = pd.read_parquet(d / "pairs.parquet") if (d / "pairs.parquet").exists() else None
        nx_ = 0
        if ends is not None and pairs is not None:
            for q in pairs.itertuples():
                if q.ds > an.TOL_M:
                    continue
                a = ends[(ends.ear == q.a_ear) & (ends.road == q.road)]
                c = ends[(ends.ear == q.b_ear) & (ends.road == q.road)]
                ax.scatter([(a.x.mean() + c.x.mean()) / 2], [(a.y.mean() + c.y.mean()) / 2],
                           s=200, facecolors="none", edgecolors="#eb6834", linewidths=2, zorder=5)
                nx_ += 1
        st = pd.read_parquet(ms[r].dir / "steps.parquet")
        ax.set_title(f"{r}: {n} ears, D {st.D.iloc[-1]:.2f}, L_P {st.L_P.iloc[-1]:.3f}, "
                     f"{nx_} crossings", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle(f"{world}, b{b:g}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def cx_bld() -> str:
    return "#d6d5d0"


def main(argv: list[str]) -> None:
    if argv == ["table"]:
        table()
    elif len(argv) == 4 and argv[0] == "plot":
        plot(argv[1], float(argv[2]), Path(argv[3]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
