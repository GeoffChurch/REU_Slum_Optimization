"""The complex as the runtime, phase 2b: the raster's side of the cross-checks against the complex,
and their report (NOTES, "The complex as the runtime: the ports and the cross-checks"). The raster
(RASTER: uni+apwfs at h 0.5, ell 3 m, K 8, the area population, B's placement; its metric as
phase 1's acceptance scored it) scores what the complex scored on the 60 large blocks: the road
networks' area prefixes at CROSS_BUDGETS, the stored clearings and the complex's presets
(road_rescore.cx_stored, cx_presets), each under P, P_{1/2} and J_2 with B (i); and each home's
P_{1/2} weight and escape time at baseline. Phase 3 deletes this module with the raster; its
summaries (ROWS/summary_*.csv, committed) are phase 3's record of the cross-checks.

    S=research/roadless/crosscheck.py     # from the repo root, PYTHONPATH=., single-threaded
    CUDA_PATH=/usr uv run python $S raster <id> <cpu|gpu>   # a cluster `blocks` task
    CUDA_PATH=/usr uv run python $S pictures <id> <cpu|gpu> # the flows, both models
    uv run python $S report

`pictures`: the corridor pictures' cross-check. On the raster's own cells (h 0.5) the raster's
flow (current_map.flow under RASTER: the flux through a cell, sum over headings of |phi_k|) and
the complex's (gap_maps: each cell's current density times h, the triangles painted onto the
raster's cells), at the baseline and after SIMP's default clearing as each metric optimized it
(PICTURE_ARMS: the store's runs under P on the 13). Per state: Spearman's rho over the cells both
models hold open; the main lanes (each model's top LANE_SHARE of its cells by flow) and their
overlap; the raster's flow on cells the complex holds closed or nearly dry, classified
(dry_kinds) as in the street band, inside a standing footprint, in the shy distance's zero zone
(within w / 2 of one) or in a pocket cut off by a gap narrower than w (its bottleneck). Rows
ROWS/pictures/<id>.parquet, images ROWS/pictures/<id>.npz.

Rows: ROWS/raster/<RASTER>/<id>__{homes,roads,clearings}.parquet (ignored); the complex's are
road_rescore's (road_rows/<metric>/<device>/...).
"""
from __future__ import annotations

import socket
import sys
import time
from pathlib import Path
from typing import NamedTuple, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clear  # noqa: E402
import clearings  # noqa: E402
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_factor  # noqa: E402
import gap_maps  # noqa: E402
import lifted  # noqa: E402
import metrics  # noqa: E402
import road_rescore as rr  # noqa: E402
import search  # noqa: E402

RASTER = "uni+apwfs"                 # the raster's metric (lifted.conductance_of)
MESH = lifted.UniformMesh(0.5, offset=lifted.OFFSET)
ELL_M, K = 3.0, 8                    # the lifted operator's turning length and headings
CROSS_BUDGETS = (0.05, 0.10)         # the road prefixes the raster scores (item 4's report's)
ROWS = HERE / "cross_rows"
ROAD_ROWS = HERE / "road_rows"
LINEUP = ("clearance", "clearance_looped", "cycle_native", "cycle_native_betweenness_contrast",
          "resistance_lp", "euclidean_grid")     # item 4's six networks on the 60
PQ = common.PriorityDemand(0.5)
POP = common.PopulationDemand()
AREA = common.POPULATIONS["area"]


def raster_dir(name: str) -> Path:
    """The raster's rows, under its metric's name."""
    return ROWS / "raster" / name


def clearing_runs() -> dict[str, str]:
    """The clearings both metrics score: the stored ones (road_rescore.cx_stored) and the
    complex's presets as the 60-block run made them on a core (their polish scored exactly)."""
    presets = rr.cx_presets(metrics.RUNTIME, search.Exactly())
    return {**rr.cx_stored(metrics.RUNTIME),
            "simp_cxE": rr.store_run(presets.simp), "cheap_cxE": rr.store_run(presets.cheap)}


def raster(bid: str, device: str) -> None:
    """One block on the raster (module docstring)."""
    rr._check_single_threaded()
    out = raster_dir(RASTER)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    [b] = common.build_blocks([bid])
    cond = lifted.conductance_of(RASTER, lifted.scans_of(device))
    p = lifted.Params(ELL_M, K, along=cond.along, cells=cond.cells,
                      solver=lifted.solver_of(device))
    sc = common.Scorer(b, MESH, p, population=AREA, demand=POP)
    scq = common.Scorer.on_grid(b, sc.grid, p, AREA, demand=PQ)
    g = sc.grid
    t_setup = time.perf_counter() - t0
    rr.write_parquet(pd.DataFrame(dict(block=bid, home=np.arange(len(sc.w)), pop=sc.w, u0=sc.u0,
                                       a_q=scq.inj / sc.w, enclosed=sc.enclosed)),
                     out / f"{bid}__homes.parquet")

    def lenses(free: np.ndarray) -> dict[str, float]:
        sy = lifted.System(g, free, p)
        s1 = lifted.solve(g, free, sc.f, p, system=sy, rtol=clear.RTOL_SCORE)
        sq = lifted.solve(g, free, scq.f, p, system=sy, rtol=clear.RTOL_SCORE)
        r = dict(L_P=1.0 - s1.P / sc.P0, L_Pq=1.0 - sq.P / scq.P0,
                 L_J2=sc.perm_p(sc.home_u_of(s1, free), 2.0), enclosed=sc.enclosed_left(free))
        del sy
        p.solver.release()
        return r

    base = dict(block=bid, n=len(sc.w), metric=RASTER, P0=sc.P0, Pq0=scq.P0,
                J2_0=sc.J(sc.u0, 2.0), enclosed0=sc.enclosed0, cells=int(g.inside.sum()),
                device=device, host=socket.gethostname())
    # the stored clearings and the complex's presets
    store = clearings.Store.committed()
    runs = store.runs()
    lab = g.label_sub(sc.polys)
    crows = []
    for key, run in clearing_runs().items():
        if run not in runs:
            raise SystemExit(f"{run}: not in the store")
        t = store.table(run)
        if not (t["block"] == bid).any():
            continue
        ids = store.cleared(run, b, step=1)
        ts = time.perf_counter()
        crows.append(dict(clearing=key, run=run, n_cleared=len(ids),
                          D=float(sc.w[ids].sum() / sc.w.sum()),
                          **lenses(common.carved(g, np.isin(lab, ids))),
                          t=time.perf_counter() - ts))
        print(f"{bid} raster {key}: L_P {crows[-1]['L_P']:.4f} L_Pq {crows[-1]['L_Pq']:.4f} "
              f"L_J2 {crows[-1]['L_J2']:.4f} enclosed {crows[-1]['enclosed']:.5f}", flush=True)
    rr.write_parquet(pd.DataFrame(crows).assign(**base), out / f"{bid}__clearings.parquet")
    # the road prefixes (Carve's rule, road by road: road_rescore.Corridors)
    X, Y = g._sub_xy()
    cor = rr.Corridors(g, X, Y)
    rrows = []
    for name in LINEUP:
        every = pd.read_parquet(rr.prefix_dir(ROAD_ROWS) / f"{bid}__{name}.parquet")
        pre = every[(every["pop"] == rr.CX_POP)
                    & every.budget.apply(lambda v: bool(np.isclose(v, CROSS_BUDGETS).any()))]
        ordered = gpd.read_parquet(rr.ordered_path(ROAD_ROWS / "scratch", bid, name))
        stops = sorted({int(mk) for mk in pre.m})
        got = {}
        for mk, _mask, free in cor.masks(ordered, stops):
            ts = time.perf_counter()
            got[mk] = dict(lenses(free), t=time.perf_counter() - ts)
        for _, r in pre.iterrows():
            rrows.append(dict(method=name, budget=float(r.budget), m=int(r.m),
                              D_area=float(r.D_area), **got[int(r.m)]))
        print(f"{bid} raster {name}: " + ", ".join(
            f"D {v['budget']:g} L_P {v['L_P']:.4f}" for v in rrows[-len(pre):]), flush=True)
    rr.write_parquet(pd.DataFrame(rrows).assign(**base, t_setup=t_setup,
                                                t_block=time.perf_counter() - t0),
                     out / f"{bid}__roads.parquet")
    print(f"{time.strftime('%H:%M:%S')} {bid} raster done in {time.perf_counter() - t0:.0f}s",
          flush=True)


# ------------------------------------------------------------------ the pictures

class Dry(NamedTuple):
    """Where the complex holds a cell dry: in the street band (its centre within band_m of a
    street: the street itself, both models' ground), inside a standing footprint (its centre in
    it), in the shy distance's zero zone (within w / 2 of one), or beyond it in a pocket the
    street reaches only through a gap narrower than w (`pocket`, with its bottleneck: the widest
    path's to the street on the complex not grown, gap_complex.exit_widths); the rest `other`."""
    band: NDArray[np.bool_]
    inside: NDArray[np.bool_]
    zone: NDArray[np.bool_]
    pocket: NDArray[np.bool_]
    other: NDArray[np.bool_]
    bottleneck: NDArray[np.float64]


def dry_kinds(geo: gc.Geometry, spec: gc.Spec, standing: NDArray[np.bool_],
              fr: gap_maps.Frame, w: float, dry: NDArray[np.bool_]) -> Dry:
    """Classify the dry cells `dry` (flat, on frame `fr`) of the state standing `standing`, w the
    shy distance's width (twice its growth)."""
    xs = fr.x0 + (np.arange(fr.nx)) * fr.h
    ys = fr.y0 + (np.arange(fr.ny)) * fr.h
    X, Y = np.meshgrid(xs, ys)
    pts = shapely.points(X.ravel()[dry], Y.ravel()[dry])
    polys = geo.polys[standing]
    d = np.full(len(pts), np.inf)
    if len(polys):
        _i, dist = shapely.STRtree(polys).query_nearest(pts, return_distance=True,
                                                         all_matches=False)
        d = dist
    cx = gc.build(geo, spec, standing, carve=None, grow=0.0)
    ex = gc.exit_widths(cx)
    qi, qt = shapely.STRtree(cx.tri_geom).query(pts, predicate="intersects")
    bott = np.full(len(pts), np.nan)
    bott[qi] = ex.width[qt]
    street = shapely.buffer(cast(BaseGeometry, geo.streets), spec.band_m)
    shapely.prepare(street)
    band = np.asarray(shapely.contains(street, pts))
    inside = ~band & (d <= 0.0)
    zone = ~band & (d > 0.0) & (d < w / 2.0)
    pocket = ~band & (d >= w / 2.0) & (bott < w)

    def spread(v: NDArray[np.bool_]) -> NDArray[np.bool_]:
        a = np.zeros(dry.shape, dtype=bool)
        a[dry] = v
        return a
    b = np.full(dry.shape, np.nan)
    b[dry] = bott
    return Dry(band=spread(band), inside=spread(inside), zone=spread(zone),
               pocket=spread(pocket), other=spread(~(band | inside | zone | pocket)),
               bottleneck=b)


PICTURE_ARMS = {"simp_fws": f"relax_rows/{RASTER}/{rr.CX_PLAN}/D0.05.p1",
                "simp_cx": f"relax_rows/{metrics.RUNTIME}/{rr.CX_PLAN}/D0.05.p1"}
LANE_SHARE = 0.10        # a model's main lanes: its top tenth of the open cells by flow
DRY = 0.01               # a cell the complex holds nearly dry: under this share of its median


def pictures(bid: str, device: str) -> None:
    """One block's flows under both models (module docstring)."""
    import current_map
    from scipy.stats import spearmanr
    rr._check_single_threaded()
    out = ROWS / "pictures"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    [b] = common.build_blocks([bid])
    cond = lifted.conductance_of(RASTER, lifted.scans_of(device))
    p = lifted.Params(ELL_M, K, along=cond.along, cells=cond.cells,
                      solver=lifted.solver_of(device))
    sc = common.Scorer(b, MESH, p, population=AREA, demand=POP)
    g = sc.grid
    lab = g.label_sub(sc.polys)
    metric = gcl.complex_of(metrics.RUNTIME, factorization=gap_factor.factorization_of(device))
    m = metric.clearing(b, demand=POP)
    geo = gc.geometry(b)
    ny, nx = g.inside.shape
    fr = gap_maps.Frame(x0=g.x0 - geo.origin[0], y0=g.y0 - geo.origin[1], h=g.h, ny=ny, nx=nx)
    store = clearings.Store.committed()
    states = {"baseline": np.zeros(m.n, dtype=bool)}
    for key, run in PICTURE_ARMS.items():
        x = np.zeros(m.n, dtype=bool)
        x[store.cleared(run, b, step=1)] = True
        states[key] = x
    rows, images = [], {}
    for key, x in states.items():
        free = common.carved(g, np.isin(lab, np.flatnonzero(x)))
        fr_r = current_map.flow(sc, free, p)
        s_r = np.abs(fr_r.phi).sum(axis=0)
        st = m.state(x.astype(np.float64))
        fl = gap_maps.flow_of(m, st, m.pop)
        s_c = (gap_maps.flow_image(m, st, fl, fr) * g.h).ravel()
        open_r, open_c = s_r > 0, np.isfinite(s_c)
        both = open_r & open_c
        rho = float(spearmanr(s_r[both], s_c[both]).statistic)
        top_r = open_r & (s_r >= np.quantile(s_r[open_r], 1 - LANE_SHARE))
        top_c = open_c & (np.nan_to_num(s_c) >= np.quantile(s_c[open_c], 1 - LANE_SHARE))
        dry = open_r & (~open_c | (np.nan_to_num(s_c) < DRY * np.median(s_c[open_c])))
        k = dry_kinds(geo, metric.spec, ~x, fr, 2.0 * metric.form.grow, dry)
        tot = float(s_r[open_r].sum())
        share = {name: float(s_r[getattr(k, name)].sum() / tot)
                 for name in ("band", "inside", "zone", "pocket", "other")}
        bott = k.bottleneck[k.pocket]
        rows.append(dict(
            block=bid, state=key, n_cleared=int(x.sum()), cells_r=int(open_r.sum()),
            cells_c=int(open_c.sum()), cells_both=int(both.sum()), rho=rho,
            lanes_jaccard=float((top_r & top_c).sum() / (top_r | top_c).sum()),
            lanes_c_in_r=float((top_r & top_c).sum() / top_c.sum()),
            lanes_r_dry=float((top_r & dry).sum() / top_r.sum()),
            flow_r_dry=float(s_r[dry].sum() / tot),
            **{f"flow_r_dry_{name}": v for name, v in share.items()},
            pocket_cells=int(k.pocket.sum()),
            pocket_bottleneck_q10=float(np.quantile(bott, 0.1)) if len(bott) else np.nan,
            pocket_bottleneck_q50=float(np.quantile(bott, 0.5)) if len(bott) else np.nan,
            pocket_bottleneck_q90=float(np.quantile(bott, 0.9)) if len(bott) else np.nan,
            P_r=float(fr_r.P), P_c=float(m._rhs(st, m.pop) @ fl.u)))
        images[f"r_{key}"] = s_r.astype("float32")
        images[f"c_{key}"] = s_c.astype("float32")
        images[f"b_{key}"] = k.bottleneck.astype("float32")
        print(f"{bid} {key}: rho {rho:.3f}, lanes Jaccard {rows[-1]['lanes_jaccard']:.3f}, the "
              f"raster's flow on cells the complex holds dry {rows[-1]['flow_r_dry']:.4f}: "
              + ", ".join(f"{n} {v:.4f}" for n, v in share.items())
              + f"; pockets' bottlenecks median {rows[-1]['pocket_bottleneck_q50']:.3f} m",
              flush=True)
        del fr_r
        p.solver.release()
    images["shape"] = np.array([ny, nx])
    images["frame"] = np.array([fr.x0, fr.y0, fr.h])
    np.savez_compressed(out / f"{bid}.npz", **images)
    rr.write_parquet(pd.DataFrame(rows).assign(device=device, host=socket.gethostname(),
                                               t=time.perf_counter() - t0),
                     out / f"{bid}.parquet")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["raster"] and len(a) == 3:
        raster(a[1], a[2])
    elif a[:1] == ["pictures"] and len(a) == 3:
        pictures(a[1], a[2])
    else:
        raise SystemExit(__doc__)
