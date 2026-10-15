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
    return {**rr.cx_stored(metrics.RUNTIME)._asdict(),
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


STORED = rr.cx_stored(metrics.RUNTIME)
PICTURE_ARMS = {"simp_fws": STORED.simp_fws, "simp_cx": STORED.simp_cx}
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
        share = {name: float(s_r[cells].sum() / tot)
                 for name, cells in (("band", k.band), ("inside", k.inside), ("zone", k.zone),
                                     ("pocket", k.pocket), ("other", k.other))}
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


# ------------------------------------------------------------------ the report

OLD_RASTER = "uni"       # item 4's rows (the step rule before the face aperture): the order's
                         # third reading, road_rows/h0.5_ell3_K8_uni_area_<budgets>/
LENS_OF_OLD = {"L_P": "P1", "L_J2": "P2"}    # item 4's columns for the same lenses (area rows)


def _complex() -> gcl.ComplexMetric:
    """The runtime's metric as the 60-block run scored it (on a core)."""
    return gcl.complex_of(metrics.RUNTIME, factorization=gap_factor.factorization_of("cpu"))


def _tau(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    from scipy.stats import kendalltau
    return float(kendalltau(a, b).statistic) if len(a) >= 3 else np.nan


def _q(v: pd.Series) -> str:
    return (f"median {v.median():+.2f} [{v.quantile(0.25):+.2f}, {v.quantile(0.75):+.2f}] "
            f"over {v.notna().sum()}")


def roads_report() -> pd.DataFrame:
    """The road methods' order, the complex's against the raster's (uni+apwfs, and item 4's uni)
    at each CROSS_BUDGET, in B's order per lens: Kendall tau per block over the methods reaching
    the budget, and whether the leader is the same."""
    cx = rr.cx_lens_table
    t = rr.cx_rows(ROAD_ROWS, _complex(), LINEUP)
    ras = rr._read_all(sorted(raster_dir(RASTER).glob("*__roads.parquet")))
    old_dir = ROAD_ROWS / f"h0.5_ell{rr.ELL_M:g}_K{rr.K}_{OLD_RASTER}_area_{rr.budgets_tag()}"
    old = rr._read_all([f for name in LINEUP for f in sorted(old_dir.glob(f"*__{name}.parquet"))]
                       ).rename(columns={v: k for k, v in LENS_OF_OLD.items()})
    rows = []
    for budget in CROSS_BUDGETS:
        c = cx(t, budget)
        r = ras[np.isclose(ras.budget, budget)].copy()
        o = old[np.isclose(old.budget, budget)].copy()
        for bid, g in c[c.reach].groupby("block"):
            gr = r[(r.block == bid) & r.method.isin(g.method)].set_index("method")
            go = o[(o.block == bid) & o.method.isin(g.method)].set_index("method")
            gc_ = g.set_index("method")
            for name, other, lenses in (("raster", gr, rr.CX_LENSES),
                                        ("uni", go, tuple(LENS_OF_OLD))):
                if set(other.index) != set(gc_.index):
                    raise ValueError(f"{bid} D {budget:g}: {name}'s rows hold "
                                     f"{sorted(other.index)}, the complex's {sorted(gc_.index)}")
                other = other.loc[gc_.index]
                for lens in lenses:
                    rank_o = rr._b_rank(other.reset_index(), "enclosed", lens)
                    rank_c = gc_[f"rank_{lens}"].to_numpy()
                    rows.append(dict(block=bid, budget=budget, lens=lens, against=name,
                                     methods=len(gc_), tau=_tau(-rank_c, -rank_o),
                                     same_leader=bool(gc_.index[np.argmin(rank_c)]
                                                      == gc_.index[np.argmin(rank_o)]),
                                     leader_cx=str(gc_.index[np.argmin(rank_c)]),
                                     leader_other=str(gc_.index[np.argmin(rank_o)])))
    df = pd.DataFrame(rows)
    print("THE ROAD METHODS' ORDER, the complex against the raster, Kendall tau per block in B's "
          "order")
    for budget in CROSS_BUDGETS:
        for lens in rr.CX_LENSES:
            for against in ("raster", "uni"):
                g = df[np.isclose(df.budget, budget) & (df.lens == lens) & (df.against == against)]
                if len(g):
                    print(f"  D {budget:g} {lens:5s} vs {against:6s}: {_q(g.tau)}; the same "
                          f"leader on {int(g.same_leader.sum())} of {len(g)}")
    return df


def clearings_report() -> pd.DataFrame:
    """The stored clearings and the complex's presets on both metrics: per block and lens, each
    clearing's B (i) and lens on each, the clearings' order in B's order on each metric (Kendall
    tau between them), and each pair's lead (the complex's SIMP and cheap preset against the
    raster's of the same preset)."""
    every = rr.cx_plain_rows(ROAD_ROWS, _complex())
    pl = every[every.clearing != "baseline"]
    ras = rr._read_all(sorted(raster_dir(RASTER).glob("*__clearings.parquet")))
    m: pd.DataFrame = pl.merge(ras, on=["block", "clearing"], suffixes=("_cx", "_r"))
    rows = []
    for bid in m.block.unique():
        g = m[m.block == bid]
        for lens in rr.CX_LENSES:
            rc = rr._b_rank(g.rename(columns={"enclosed_cx": "e", f"{lens}_cx": "v"}), "e", "v")
            rk = rr._b_rank(g.rename(columns={"enclosed_r": "e", f"{lens}_r": "v"}), "e", "v")
            rows.append(dict(block=bid, lens=lens, clearings=len(g), tau=_tau(-rc, -rk)))
    tau = pd.DataFrame(rows)
    print("\nTHE STORED CLEARINGS ON BOTH METRICS")
    for lens in rr.CX_LENSES:
        g = tau[tau.lens == lens]
        print(f"  {lens}: the clearings' order, complex against raster, Kendall tau {_q(g.tau)}")
    lev = m.groupby("clearing")[[f"{ln}_{x}" for ln in rr.CX_LENSES for x in ("cx", "r")]].median()
    print("  medians over the blocks each clearing has:")
    print(lev.round(4).to_string())
    enc = m.groupby("clearing").agg(blocks=("block", "size"),
                                    walled_cx=("enclosed_cx", lambda v: int((v > 0).sum())),
                                    walled_r=("enclosed_r", lambda v: int((v > 0).sum())))
    print("  blocks where each leaves people walled in, on the complex | the raster:")
    print(enc.to_string())
    pairs = (("simp_cxE", "simp_fws"), ("cheap_cxE", "cheap_fws"), ("simp_cxE", "simp_uni"),
             ("cheap_cxE", "cheap_uni"))
    for a_, b_ in pairs:
        for x in ("cx", "r"):
            for lens in rr.CX_LENSES:
                wa = m[m.clearing == a_].set_index("block")
                wb = m[m.clearing == b_].set_index("block")
                both = wa.index.intersection(wb.index)
                if not len(both):
                    raise ValueError(f"{a_} and {b_}: no block holds both")
                ea, eb = wa.loc[both, f"enclosed_{x}"], wb.loc[both, f"enclosed_{x}"]
                va, vb = wa.loc[both, f"{lens}_{x}"], wb.loc[both, f"{lens}_{x}"]
                ahead = (ea < eb) | ((ea == eb) & (va > vb))
                tie = ea == eb
                d = (va - vb)[tie]
                print(f"  {a_} against {b_} on the {'complex' if x == 'cx' else 'raster'}, "
                      f"{lens}: ahead on {int(ahead.sum())} of {len(both)}; where (i) ties "
                      f"({int(tie.sum())}) {d.median():+.4f} median ({d.min():+.4f} .. "
                      f"{d.max():+.4f})")
    return m


def weights_report() -> pd.DataFrame:
    """P_{1/2}'s weights per home, the complex's against the raster's: Spearman per block over
    the homes with demand on both, and the escape times' too."""
    from scipy.stats import spearmanr
    d = rr.cx_dir(ROAD_ROWS, _complex())
    rows = []
    for f in sorted(raster_dir(RASTER).glob("*__homes.parquet")):
        bid = f.name.split("__")[0]
        hr, hc = pd.read_parquet(f), pd.read_parquet(d / f"{bid}__homes.parquet")
        ok = (~hr.enclosed.to_numpy()) & (~hc.nodem.to_numpy())
        rows.append(dict(block=bid, homes=int(ok.sum()),
                         walled_apart=int((hr.enclosed.to_numpy() != hc.nodem.to_numpy()).sum()),
                         rho_a=float(spearmanr(hr.a_q[ok], hc.a_q[ok]).statistic),
                         rho_u0=float(spearmanr(hr.u0[ok], hc.u0[ok]).statistic),
                         a_ratio_med=float(np.median(hc.a_q[ok] / hr.a_q[ok])),
                         a_ratio_p05=float(np.quantile(hc.a_q[ok] / hr.a_q[ok], 0.05)),
                         a_ratio_p95=float(np.quantile(hc.a_q[ok] / hr.a_q[ok], 0.95))))
    df = pd.DataFrame(rows)
    print("\nP_{1/2}'S WEIGHTS, the complex's against the raster's, Spearman per block: "
          f"{_q(df.rho_a)} (the escape times {_q(df.rho_u0)}); the complex's weight over the "
          f"raster's at the median home {df.a_ratio_med.median():.3f}, p05 -- p95 "
          f"{df.a_ratio_p05.median():.3f} -- {df.a_ratio_p95.median():.3f}; homes walled in on "
          f"one model only: {int(df.walled_apart.sum())} on {int((df.walled_apart > 0).sum())} "
          "blocks")
    return df


def pictures_report() -> pd.DataFrame:
    df = rr._read_all(sorted((ROWS / "pictures").glob("*.parquet")))
    pd.set_option("display.width", 250)
    print("\nTHE CORRIDOR PICTURES, the raster's flow against the complex's on the raster's cells")
    cols = ["block", "state", "rho", "lanes_jaccard", "lanes_c_in_r", "lanes_r_dry", "flow_r_dry",
            "flow_r_dry_band", "flow_r_dry_inside", "flow_r_dry_zone", "flow_r_dry_pocket",
            "flow_r_dry_other", "pocket_bottleneck_q50"]
    print(df[cols].round(4).to_string(index=False))
    return df


def report() -> None:
    """Every cross-check's summary (ROWS/summary_*.csv, committed)."""
    for name, fn in (("roads", roads_report), ("clearings", clearings_report),
                     ("weights", weights_report), ("pictures", pictures_report)):
        fn().to_csv(ROWS / f"summary_{name}.csv", index=False)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["raster"] and len(a) == 3:
        raster(a[1], a[2])
    elif a[:1] == ["pictures"] and len(a) == 3:
        pictures(a[1], a[2])
    elif a == ["report"]:
        report()
    else:
        raise SystemExit(__doc__)
