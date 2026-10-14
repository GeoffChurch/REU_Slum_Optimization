"""SIMP on the gap complex under the passable width's forms (the brief's addendum: "SIMP's default
under closing against the shy distance, and whether the shy distance concentrates the clearing and
the flow into fewer, wider corridors"; NOTES, "The gap complex").

SIMP base (relax.simp with PLAN's stages and updates: fw0.q3.i10.t0.001.e0.0001.k1s1e-06) on the
complex's clearing model (gap_clear.Relaxation: meshed interiors, the form's convex factors),
objective P (the population's demand, power 1), area D BUDGET, free first (gap_complex.free_first
at the form's w, B's first stage, held at 1); every iterate rounded by x (relax.round_by_x) and the
roundings scored in the eps world at SCORE_EPS (the incumbent, as the plan's .k1s1e-06), the
winner scored exactly on the complex rebuilt (gap_compare.ComplexModel, the form's own geometry).

The corridors (`corridors`), on the winner's rebuilt complex under its form: the flow-weighted
clear width (per triangle the flux magnitude times its area, the clear width 2 d at its centroid, d
to the nearest standing footprint) and the flow's share in lanes at least LANES wide, the cleared
ground's clear width (per sub-point of the cleared footprints, area-weighted, after clearing), and
how many clusters the cleared buildings form (touching or within CLUSTER_M). `cross` measures every
form's clearing of a block under each of CROSS, so two clearings compare under one flow.

    S=research/roadless/gap_simp.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S run <ids,|@file> <procs> <forms,>
    uv run python $S cross <ids,|@file> <procs>       # every form's clearing under each of CROSS
"""
from __future__ import annotations

import multiprocessing
import os
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_compare as gcmp  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_flow as gf  # noqa: E402
import gap_study  # noqa: E402
import relax  # noqa: E402

# ------------------------------------------------------------------ the run's configuration
PLAN = relax.plan_of("fw0.q3.i10.t0.001.e0.0001.k1s1e-06")    # SIMP base
BUDGET = relax.D_LENS
POWER = 1.0
R = 0.5
LEVELS = 4
SCORE_EPS = 1e-6
CLUSTER_M = 0.5                 # cleared buildings closer than this are one cluster
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
LANES = (1.0, 2.0, 4.0)          # the flow's share in lanes at least this wide (m)
CROSS = ("close0.5", "shy0.5")   # each clearing scored under these forms too
ROWS = HERE / "gap_rows" / "simp"


def _peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def _wq(v: np.ndarray, w: np.ndarray, q) -> np.ndarray:
    """Weighted quantiles."""
    o = np.argsort(v)
    c = np.cumsum(w[o])
    c = c / c[-1]
    return np.interp(q, c, v[o])


def corridors(geo: gc.Geometry, cm: gcmp.ComplexModel, cleared: np.ndarray) -> dict:
    """The corridor measures of a clearing (module docstring) on its rebuilt complex."""
    standing = ~cleared
    m = gf.mesh(geo, cm.spec, cm.fs, cm.form, standing, carve=None)
    c = gf.cells(m)
    sc = gf.Scores(m, c, cm.dem)
    f_pt = cm.dem.S @ cm.pop
    on = sc.pcell >= 0
    uk = sc.unk[np.maximum(sc.pcell, 0)]
    live = on & (uk >= 0)
    b = np.bincount(uk[live], weights=f_pt[live], minlength=sc.A.shape[0])
    u = sc.lu.solve(b)
    ucell = np.zeros(c.n)
    ucell[~c.ground] = u
    # per triangle the flux magnitude: the cell's net |flux| through its passages, half to each
    fl = np.abs(c.T * (ucell[c.ea] - ucell[c.eb]))
    per_cell = 0.5 * (np.bincount(c.ea, weights=fl, minlength=c.n)
                      + np.bincount(c.eb, weights=fl, minlength=c.n))
    tri_cell = c.cell
    ok = tri_cell >= 0
    area = m.area
    cen = m.xy[m.tri].mean(axis=1)
    _idx, polys = gc.obstacles(geo, standing, carve=None, grow=0.0)
    _i, d = shapely.STRtree(polys).query_nearest(shapely.points(cen[ok, 0], cen[ok, 1]),
                                                 return_distance=True, all_matches=False)
    # a cell's flux spread over its triangles by area
    carea = np.bincount(tri_cell[ok], weights=area[ok], minlength=c.n)
    wt = per_cell[tri_cell[ok]] * area[ok] / np.maximum(carea[tri_cell[ok]], 1e-300)
    out = {f"flow_width_q{q:g}": float(v) for q, v in zip(QUANTILES, _wq(2 * d, wt, QUANTILES),
                                                          strict=True)}
    out["flow_width_mean"] = float((2 * d * wt).sum() / wt.sum())
    for lane in LANES:
        out[f"flow_share_ge{lane:g}"] = float(wt[2 * d >= lane].sum() / wt.sum())
    # the cleared ground's clear width
    if cleared.any():
        ff = np.flatnonzero(cleared)
        bt = gc._triangulate(shapely.union_all(geo.polys[ff]), cm.spec.s)
        bc = shapely.get_coordinates(bt).reshape(-1, 4, 2)[:, :3]
        bxy, binv = np.unique(bc.reshape(-1, 2), axis=0, return_inverse=True)
        P, w, _t = gf._points_of(bxy, binv.reshape(-1, 3), 4)
        _i2, d2 = shapely.STRtree(polys).query_nearest(shapely.points(P[:, 0], P[:, 1]),
                                                       return_distance=True, all_matches=False)
        out.update({f"cleared_width_q{q:g}": float(v)
                    for q, v in zip(QUANTILES, _wq(2 * d2, w, QUANTILES), strict=True)})
        tree = shapely.STRtree(geo.polys[ff])
        i, j = tree.query(geo.polys[ff], predicate="dwithin", distance=CLUSTER_M)
        nc, _lab = connected_components(sp.csr_matrix((np.ones(len(i)), (i, j)),
                                                      shape=(len(ff), len(ff))), directed=False)
        out["clusters"] = int(nc)
    out["n_cleared"] = int(cleared.sum())
    return out


def run_one(job: tuple[str, str]) -> str | None:
    bid, token = job
    form = gcmp.form_of(token)
    out = ROWS / f"{PLAN.name}_p{POWER:g}_{gcmp.flow_spec(R, LEVELS).name}" / form.name / \
        f"{bid}.parquet"
    if out.exists():
        return None
    try:
        t0 = time.time()
        [b] = common.build_blocks([bid])
        geo = gc.geometry(b)
        pop = gcmp.AREA.weights(geo.polys)
        cost = pop / pop.sum()
        fs = gcmp.flow_spec(R, LEVELS)
        spec = gap_study.spec_of(gcmp.S_M)
        m = gcl.Model(geo, spec, fs, form, lumped=False, pop=pop)
        t_model = time.time() - t0
        fr = gc.free_first(geo, spec, w=form.w, budget=BUDGET, cost=cost, pop=pop,
                           held0=np.zeros(len(pop), dtype=bool), cx0=None)
        held = fr.held
        xlo = np.where(held, 1.0, relax.XMIN)
        rel = gcl.Relaxation(m, POWER, eps=PLAN.eps, inj=pop, cost=cost)
        scorer = gcl.Relaxation(m, POWER, eps=SCORE_EPS, inj=pop, cost=cost)
        best = {"J": np.inf, "r": None}

        def watch(x, stage):
            r = relax.round_by_x(x, cost, BUDGET, first=held.astype(float))
            J = scorer.value(r)
            if J < best["J"]:
                best["J"], best["r"] = J, r

        rest = BUDGET - float(cost[held].sum())
        x0 = np.where(held, 1.0, rest / float(cost[~held].sum()))
        x = relax.simp(rel, x0, BUDGET, PLAN.stages, PLAN.iters, PLAN.update,
                       log=lambda s: print(f"{bid} {form.name} {s}", flush=True), xlo=xlo,
                       watch=watch)
        watch(x, PLAN.stages[-1])
        cleared = best["r"] > 0
        t_simp = time.time() - t0 - t_model
        cm = gcmp.ComplexModel(b, fs, form)
        score, _u = cm.score(~cleared, None)
        cor = corridors(geo, cm, cleared)
        # the clearing under the other forms too (each its own baseline and demand)
        cross = {}
        for other in CROSS:
            if other == token:
                continue
            co = gcmp.ComplexModel(b, fs, gcmp.form_of(other))
            r2, _u2 = co.score(~cleared, None)
            cross.update({f"{k}@{other}": r2[k] for k in ("L_P", "L_Pq0.5", "L_J2", "enclosed")})
        row = dict(block=bid, form=form.name, plan=PLAN.name, power=POWER, flow=fs.name,
                   D=float(cost[cleared].sum()), free_D=float(cost[held].sum()),
                   free_cleared=" ".join(map(str, np.flatnonzero(held))),
                   cleared=" ".join(map(str, np.flatnonzero(cleared))),
                   grey=float(np.mean((x > 0.05) & (x < 0.95))), J_eps=best["J"], **score,
                   **cor, **cross, t_model=t_model, t_simp=t_simp, t_block=time.time() - t0,
                   nt_full=m.fm.nt, cells_full=m.fc.n, host=socket.gethostname(),
                   peak_rss_gb=_peak_gb())
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_parquet(out)
        print(f"{bid} {form.name}: SIMP L_P {score['L_P']:.4f} L_Pq {score['L_Pq0.5']:.4f} L_J2 "
              f"{score['L_J2']:.4f} D {row['D']:.4f} cleared {row['n_cleared']} in "
              f"{cor.get('clusters', 0)} clusters, flow width median "
              f"{cor['flow_width_q0.5']:.2f} m ({time.time() - t0:.0f}s)", flush=True)
        return None
    except Exception as e:     # noqa: BLE001 -- a pool's task reports its block and goes on
        import traceback
        traceback.print_exc()
        return f"{bid} {token}: {e!r}"


def cross_one(bid: str) -> str | None:
    """Every form's clearing of a block (run_one's rows) measured by `corridors` under each of
    CROSS, so two clearings' corridors compare under one flow; rows gap_rows/simp_cross/."""
    fs = gcmp.flow_spec(R, LEVELS)
    base = ROWS / f"{PLAN.name}_p{POWER:g}_{fs.name}"
    out = HERE / "gap_rows" / "simp_cross" / base.name / f"{bid}.parquet"
    if out.exists():
        return None
    try:
        t0 = time.time()
        [b] = common.build_blocks([bid])
        geo = gc.geometry(b)
        clearings = {}
        for d in sorted(base.iterdir()):
            f = d / f"{bid}.parquet"
            if f.exists():
                [c] = pd.read_parquet(f).cleared
                x = np.zeros(len(geo.polys), dtype=bool)
                x[[int(i) for i in c.split()]] = True
                clearings[d.name] = x
        rows = []
        for token in CROSS:
            cm = gcmp.ComplexModel(b, fs, gcmp.form_of(token))
            for name, x in clearings.items():
                rows.append(dict(block=bid, clearing=name, scored=cm.form.name,
                                 **corridors(geo, cm, x)))
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out)
        print(f"{bid}: {len(clearings)} clearings under {len(CROSS)} forms "
              f"({time.time() - t0:.0f}s)", flush=True)
        return None
    except Exception as e:     # noqa: BLE001 -- a pool's task reports its block and goes on
        import traceback
        traceback.print_exc()
        return f"{bid}: {e!r}"


def main(argv: list[str]) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMBA_NUM_THREADS"):
        if os.environ.get(v) != "1":
            raise SystemExit(f"{v}=1 is required (single-threaded workers)")
    if argv[0] == "run":
        fn, jobs = run_one, [(i, t) for i in gap_study.ids_of(argv[1]) for t in argv[3].split(",")]
    elif argv[0] == "cross":
        fn, jobs = cross_one, gap_study.ids_of(argv[1])
    else:
        raise SystemExit(__doc__)
    procs = int(argv[2])
    if procs == 1:
        failed = [f for f in map(fn, jobs) if f]
    else:
        with multiprocessing.get_context("fork").Pool(procs, maxtasksperchild=1) as pool:
            failed = [f for f in pool.imap_unordered(fn, jobs) if f]
    if failed:
        print("FAILED:\n" + "\n".join(failed), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
