"""The corridor pictures on the gap complex (phase 2b, item 4; NOTES, "The complex as the runtime:
the ports and the cross-checks"): what a clearing or a road prefix does to the flow, under the
uniform law on the fixed mesh (the metric's own: decision (3), "uniform for the corridor
pictures"), drawn from the triangles' fluxes. They replace the raster's pictures (current_map.py's
flux and escape-time maps, opened_map.py's corridors opened, render_clear.py's clearing beside a
road corridor, the atlas's layers): the same questions, asked of the metric that scores.

THE FLUX (`flow_of`). A state's potential per cell under an injection per home (gap_clear.Model:
u, the escape time from the cell). Each static passage q between cells a and b conducts Tp_q (its
weight times its factor: gap_clear.Reach.Tp) and carries F_q = Tp_q (u_a - u_b) from a to b; each
finite street face f of a cell c carries Gf_f u_c to the street; a cell with a leak (the region
model's, gap_exits) drains l_c u_c to ground; a ground cell (an obtuse exit triangle's: u = 0)
drains whatever reaches it. Kirchhoff's law holds at every open cell: at an unknown cell the
passages' net outflow, its street faces' and its leak's equal its injection (the operator's row,
A u = b, refined to the backward-error floor), at a ground cell the drainage equals the inflow
(`conservation`, the `flux` guard of gap_clear_check).

THE CURRENT. Each cell's mean current density (people per metre per unit time), by the divergence
theorem with each passage's normal flux constant along it: |C| j_C = sum_e F_e m_e - sum_p s_p x_p
+ l_C u_C x_C over its outflows (m_e a passage's or a street face's midpoint; inflows negative),
the injections s_p at its demand points x_p and its leak's drainage at its centroid x_C. Its
magnitude is the picture's flow; the escape time of a cell is its potential.

THE PICTURE (`paint`). The triangles painted onto a pixel grid (a Frame), a pixel the mean over its
SUB x SUB sub-samples of the triangles that hold them (NaN where none does: footprints, the band,
walled-in ground), so a lane narrower than a pixel keeps its share of the pixel.

    S=research/roadless/gap_maps.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S map <id> <arm,...> <full|zoom> <out.png>

An arm is `[<label>=]<what>`: a store run (clearings.Store: `<run>`, its clearing on the block) or
a road network's area prefix (`road:<network>:<budget>`, road_rescore's prefixes: the network's
own carved mesh). Each arm's row: the flow before and after (log scale from the before map's
median to its 99.5th percentile, as the atlas's layers), its change (the corridors opened in red,
the flow drawn away in blue) and the change in the escape time from each place.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
import shapely
from numba import njit
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_factor  # noqa: E402
import metrics  # noqa: E402

if TYPE_CHECKING:
    from reblock.contracts import Block

SUB = 4                  # sub-samples a pixel's side (paint): a lane of 1/SUB of a pixel shows


class CellFlow(NamedTuple):
    """A state's flow (module docstring): `u` per cell its potential; per static passage
    (FullCells.pe order) its flux `F` from cell ea to eb; per finite street face (FullCells.pf) its
    flux `Ff` to the street; per cell its injection `inj` (the right-hand side), its drainage to
    the street `drain` (its street faces', or a ground cell's inflow) and its mean current density
    `j` (cells x 2); per triangle its flow `jt` (its cell's |j|, 0 off the state's cells). The
    drainage is the street's and the leaks' (gap_exits): `leak` per cell the leak's share of it."""
    u: NDArray[np.float64]
    F: NDArray[np.float64]
    Ff: NDArray[np.float64]
    inj: NDArray[np.float64]
    drain: NDArray[np.float64]
    j: NDArray[np.float64]
    jt: NDArray[np.float64]
    leak: NDArray[np.float64]


def _passage_flux(Tp: NDArray[np.float64], u: NDArray[np.float64], ea: NDArray[np.int64],
                  eb: NDArray[np.int64]) -> NDArray[np.float64]:
    """Per passage its flux from ea to eb, Tp (u_a - u_b)."""
    return Tp * (u[ea] - u[eb])


def flow_of(m: gcl.Model, st: gcl.State, inj: NDArray[np.float64]) -> CellFlow:
    """The flow of the factored state `st` under the injection `inj` per home."""
    c, fm, r = st.sys.cells, st.fm, st.reach
    u = m.solve(st, inj)
    F = _passage_flux(r.Tp, u, c.ea, c.eb)
    Ff = r.Gf * u[c.fc]
    rhs = m._rhs(st, inj)
    n = c.n
    out = (np.bincount(c.ea, weights=F, minlength=n) - np.bincount(c.eb, weights=F, minlength=n))
    leak = leak_drain(st, u)
    drain = np.bincount(c.fc, weights=Ff, minlength=n) + leak
    drain = np.where(r.ground, -out, drain)              # a ground cell drains its inflow
    # the mean current density: outflows at their midpoints, less the injections at their points
    mid = 0.5 * (fm.xy[fm.pu[c.pe]] + fm.xy[fm.pv[c.pe]])
    fmid = 0.5 * (fm.xy[fm.fu[c.pf]] + fm.xy[fm.fv[c.pf]])
    gmid = 0.5 * (fm.xy[fm.fu[c.pg]] + fm.xy[fm.fv[c.pg]])
    mom = np.zeros((n, 2))
    for k in range(2):
        mom[:, k] = (np.bincount(c.ea, weights=F * mid[:, k], minlength=n)
                     - np.bincount(c.eb, weights=F * mid[:, k], minlength=n)
                     + np.bincount(c.fc, weights=Ff * fmid[:, k], minlength=n))
    # a ground cell's drainage leaves through its exit faces (shared equally among them)
    gcnt = np.bincount(c.gcell, minlength=n)
    gshare = np.where(gcnt[c.gcell] > 0, drain[c.gcell] / np.maximum(gcnt[c.gcell], 1), 0.0)
    for k in range(2):
        mom[:, k] += np.bincount(c.gcell, weights=gshare * gmid[:, k], minlength=n)
    pts = m.dem
    on = m.fx.ptri >= 0
    pcell = c.cell[m.fx.ptri[on]]
    s_pt = np.asarray(pts.S @ inj)[on] * r.unknown[pcell]
    for k in range(2):
        mom[:, k] -= np.bincount(pcell, weights=s_pt * pts.xy[on, k], minlength=n)
    cg = cell_geometry(fm, c)
    # the leaks' drainage, at each cell's centroid
    for k in range(2):
        mom[:, k] += leak * cg.xy[:, k]
    j = mom / np.maximum(cg.area, 1e-300)[:, None]
    jt = np.where(r.live[c.cell], np.hypot(j[c.cell, 0], j[c.cell, 1]), 0.0)
    return CellFlow(u=u, F=F, Ff=Ff, inj=rhs, drain=drain, j=j, jt=jt, leak=leak)


def _cross(xy: NDArray[np.float64], tri: NDArray[np.int64]) -> NDArray[np.float64]:
    a, b, c = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


def leak_drain(st: gcl.State, u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per cell the current to ground through its leak (gap_exits) at the potential `u`, 0 off the
    state's unknowns: the leaks' share of the drainage (CellFlow.leak), the street's apart."""
    return np.where(st.reach.unknown, st.sys.graph.leak * u, 0.0)


class CellGeometry(NamedTuple):
    """Per cell of a cell structure its `area` (its triangles', m^2) and its centroid `xy` (their
    centroids', area-weighted)."""
    area: NDArray[np.float64]
    xy: NDArray[np.float64]


def cell_geometry(fm: gcl.FullMesh, c: gcl.FullCells) -> CellGeometry:
    tri_area = 0.5 * np.abs(_cross(fm.xy, fm.tri))
    area = np.bincount(c.cell, weights=tri_area, minlength=c.n)
    tcen = fm.xy[fm.tri].mean(axis=1)
    return CellGeometry(area=area, xy=np.column_stack([
        np.bincount(c.cell, weights=tri_area * tcen[:, k], minlength=c.n)
        / np.maximum(area, 1e-300) for k in range(2)]))


def conservation(st: gcl.State, fl: CellFlow) -> NDArray[np.float64]:
    """Per open cell (the state's live cells) its imbalance, relative to the largest current
    through a cell: at an unknown cell the net outflow through its passages, street faces and leak
    less its injection, at a ground cell its drainage less its inflow (0 by `flow_of`'s
    construction of the drainage: the guard reads the unknown cells, Kirchhoff's law as
    solved)."""
    c, r = st.sys.cells, st.reach
    n = c.n
    out = (np.bincount(c.ea, weights=fl.F, minlength=n)
           - np.bincount(c.eb, weights=fl.F, minlength=n)
           + np.bincount(c.fc, weights=fl.Ff, minlength=n) + fl.leak)
    through = (np.bincount(c.ea, weights=np.abs(fl.F), minlength=n)
               + np.bincount(c.eb, weights=np.abs(fl.F), minlength=n)
               + np.bincount(c.fc, weights=np.abs(fl.Ff), minlength=n) + np.abs(fl.leak))
    scale = max(float(through.max()), 1e-300)
    bal = np.where(r.unknown, out - fl.inj, 0.0)
    return np.where(r.live, bal, 0.0) / scale


# ------------------------------------------------------------------ painting

class Frame(NamedTuple):
    """A pixel grid in the block's local coordinates (gap_complex.geometry): pixel (row, col)'s
    centre at (x0 + col h, y0 + row h), row 0 the southmost."""
    x0: float
    y0: float
    h: float
    ny: int
    nx: int

    @staticmethod
    def around(geo: gc.Geometry, h: float, pad: float) -> Frame:
        """The block's bounds padded by `pad`, at pixels of h."""
        x0, y0, x1, y1 = shapely.bounds(cast(BaseGeometry, geo.boundary))
        nx = int(np.ceil((x1 - x0 + 2 * pad) / h))
        ny = int(np.ceil((y1 - y0 + 2 * pad) / h))
        return Frame(x0=float(x0 - pad + h / 2), y0=float(y0 - pad + h / 2), h=h, ny=ny, nx=nx)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(left, right, bottom, top) of the pixels' edges (matplotlib's imshow extent)."""
        return (self.x0 - self.h / 2, self.x0 + (self.nx - 0.5) * self.h,
                self.y0 - self.h / 2, self.y0 + (self.ny - 0.5) * self.h)


@njit(cache=True)
def _paint(xy, tri, val, x0, y0, h, ny, nx, sub):  # type: ignore[no-untyped-def]
    """Per pixel the sum of the values of the triangles holding its sub-samples, and how many
    are held."""
    acc = np.zeros((ny, nx))
    cnt = np.zeros((ny, nx))
    d = h / sub
    sx0 = x0 - h / 2 + d / 2
    sy0 = y0 - h / 2 + d / 2
    for t in range(tri.shape[0]):
        v = val[t]
        if not np.isfinite(v):
            continue
        ax, ay = xy[tri[t, 0], 0], xy[tri[t, 0], 1]
        bx, by = xy[tri[t, 1], 0], xy[tri[t, 1], 1]
        cx, cy = xy[tri[t, 2], 0], xy[tri[t, 2], 1]
        det = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if det == 0.0:
            continue
        lo_i = int(np.ceil((min(ax, bx, cx) - sx0) / d))
        hi_i = int(np.floor((max(ax, bx, cx) - sx0) / d))
        lo_j = int(np.ceil((min(ay, by, cy) - sy0) / d))
        hi_j = int(np.floor((max(ay, by, cy) - sy0) / d))
        for jj in range(max(lo_j, 0), min(hi_j, ny * sub - 1) + 1):
            py = sy0 + jj * d
            for ii in range(max(lo_i, 0), min(hi_i, nx * sub - 1) + 1):
                px = sx0 + ii * d
                w1 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / det
                w2 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / det
                w3 = 1.0 - w1 - w2
                # half-open edges: a sub-sample on a shared edge is held once
                if w1 >= 0.0 and w2 >= 0.0 and w3 > 0.0:
                    r_, c_ = jj // sub, ii // sub
                    acc[r_, c_] += v
                    cnt[r_, c_] += 1.0
    return acc, cnt


def paint(m: gcl.Model, values: NDArray[np.float64], frame: Frame) -> NDArray[np.float64]:
    """A value per triangle of `m`'s mesh (NaN: not drawn) as a pixel image on `frame`: each
    pixel the mean over its sub-samples held by a drawn triangle, NaN where none is."""
    acc, cnt = _paint(m.fm.xy, m.fm.tri, values, frame.x0, frame.y0, frame.h, frame.ny,
                      frame.nx, SUB)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, acc / cnt, np.nan)


def flow_image(m: gcl.Model, st: gcl.State, fl: CellFlow, frame: Frame) -> NDArray[np.float64]:
    """The flow (|j|) on the state's open ground as an image (NaN elsewhere)."""
    open_ = st.reach.live[st.sys.cells.cell]
    return paint(m, np.where(open_, fl.jt, np.nan), frame)


def escape_image(m: gcl.Model, st: gcl.State, fl: CellFlow, frame: Frame) -> NDArray[np.float64]:
    """The escape time from each place (the cell's potential) on the state's open ground."""
    c = st.sys.cells
    open_ = st.reach.live[c.cell]
    return paint(m, np.where(open_, fl.u[c.cell], np.nan), frame)


# ------------------------------------------------------------------ the arms

class Arm(NamedTuple):
    """One state to draw: a label, its model (the plain mesh, or a network's carved one) and its
    state, factored."""
    label: str
    model: gcl.Model
    state: gcl.State
    D: float


def arms_of(b: Block, specs: list[str], metric: gcl.ComplexMetric, plain: gcl.Model) -> list[Arm]:
    """The arms `specs` names (module docstring) on block `b`."""
    import geopandas as gpd
    import pandas as pd

    import road_rescore as rr
    store = clearings.Store.committed()
    out = []
    for arm in specs:
        label, _, spec = arm.rpartition("=")
        if spec.startswith("road:"):
            _, net, budget = spec.split(":")
            every = pd.read_parquet(rr.prefix_dir(gcl.HERE / "road_rows")
                                    / f"{b.block_id}__{net}.parquet")
            pre = every[(every["pop"] == rr.CX_POP) & np.isclose(every.budget, float(budget))]
            [mk] = pre.m.astype(int)
            ordered = gpd.read_parquet(rr.ordered_path(gcl.HERE / "road_rows" / "scratch",
                                                       b.block_id, net))
            geo = gc.geometry(b)
            carving = gcl.Carving.of(geo, [[geo.local(rr.corridor_of(ordered, mk))]])
            with tempfile.TemporaryDirectory() as tmp:
                fx = metric.fixed(b, cache=Path(tmp), carving=carving)
            mm = metric.model(b, fx, demand=common.PopulationDemand())
            out.append(Arm(label=label or f"{net} at area D {budget} ({mk} roads)", model=mm,
                           state=mm.road_state(gcl.Prefix(net=0, stop=1)),
                           D=float(pre.D_area.iloc[0])))
        else:
            ids = store.cleared(spec, b, step=1)
            x = np.zeros(plain.n)
            x[ids] = 1.0
            out.append(Arm(label=label or spec, model=plain, state=plain.state(x),
                           D=float(plain.cost[ids].sum())))
    return out


def main_map(bid: str, specs: list[str], zoom: bool, out: Path) -> None:
    """The corridor picture of `specs` on `bid` (module docstring) to `out`."""
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    metric = gcl.complex_of(metrics.RUNTIME, factorization=gap_factor.SuperLU())
    [b] = common.build_blocks([bid])
    plain = metric.clearing(b, demand=common.PopulationDemand())
    geo = gc.geometry(b)
    frame = Frame.around(geo, 0.25, 2.0)
    arms = arms_of(b, specs, metric, plain)
    st0 = plain.state(np.zeros(plain.n))
    f0 = flow_of(plain, st0, plain.pop)
    s0, e0 = flow_image(plain, st0, f0, frame), escape_image(plain, st0, f0, frame)
    on0 = np.isfinite(s0) & (s0 > 0)
    lo, hi = np.quantile(s0[on0], [0.5, 0.995])
    lognorm = mcolors.LogNorm(float(lo), float(hi), clip=True)
    polys = geo.polys
    if zoom:
        cen = np.array([[q.centroid.x, q.centroid.y] for q in polys])
        x0, y0, x1, y1 = shapely.bounds(cast(BaseGeometry, geo.boundary))
        half = 0.125 * max(x1 - x0, y1 - y0)
        mx, my = np.median(cen, axis=0)
    fig, axes = plt.subplots(len(arms), 4, figsize=(40, 10.5 * len(arms)), squeeze=False)
    for row, arm in enumerate(arms):
        mm, st = arm.model, arm.state
        f1 = flow_of(mm, st, mm.pop)
        s1, e1 = flow_image(mm, st, f1, frame), escape_image(mm, st, f1, frame)
        d = np.where(np.isfinite(s1), s1, 0.0) - np.where(np.isfinite(s0), s0, 0.0)
        d = np.where(np.isfinite(s1) | np.isfinite(s0), d, np.nan)
        lim = float(np.nanquantile(np.abs(d), 0.995))
        with np.errstate(invalid="ignore", divide="ignore"):
            ru = np.where(np.isfinite(e0) & (e0 > 0), (e1 - e0) / e0, np.nan)
        L_P = 1.0 - float(mm._rhs(st, mm.pop) @ f1.u) / float(mm.base.P)
        panels = [
            (s0, "flow before: current density |j| (log)", dict(cmap="inferno", norm=lognorm)),
            (s1, "flow after (log)", dict(cmap="inferno", norm=lognorm)),
            (d, "change in flow: red the corridors it opened, blue the flow drawn away",
             dict(cmap="RdBu_r", norm=mcolors.SymLogNorm(lim * 1e-2, vmin=-lim, vmax=lim))),
            (ru, "escape time from here: (after - before) / before (blue: faster out)",
             dict(cmap="RdBu_r", vmin=-0.6, vmax=0.6)),
        ]
        for col, (v, title, kw) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(v, origin="lower", extent=frame.extent, interpolation="nearest", **kw)
            fig.colorbar(im, ax=ax, shrink=0.5)
            standing = mm.carving.polys[st.k < 1.0]
            gpd.GeoSeries(list(standing)).plot(ax=ax, color="0.55", lw=0)
            gpd.GeoSeries([geo.boundary]).boundary.plot(ax=ax, color="k", lw=1)
            gpd.GeoSeries([geo.streets]).plot(ax=ax, color="k", lw=2)
            ax.set_axis_off()
            ax.set_aspect("equal")
            if zoom:
                ax.set_xlim(mx - half, mx + half)
                ax.set_ylim(my - half, my + half)
            ax.set_title(f"{arm.label}: D {arm.D:.3f}, L_P {L_P:.4f}\n{title}", fontsize=13)
    fig.suptitle(f"{bid}: the flow on the gap complex ({metric.name})", fontsize=16)
    fig.tight_layout()
    fig.savefig(out, dpi=60)
    print(out, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["map"] and len(a) == 5:
        main_map(a[1], a[2].split(","), {"zoom": True, "full": False}[a[3]], Path(a[4]))
    else:
        raise SystemExit(__doc__)
