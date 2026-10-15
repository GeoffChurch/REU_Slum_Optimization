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
    uv run python $S report

Rows: ROWS/raster/<RASTER>/<id>__{homes,roads,clearings}.parquet (ignored); the complex's are
road_rescore's (road_rows/<metric>/<device>/...).
"""
from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clear  # noqa: E402
import clearings  # noqa: E402
import common  # noqa: E402
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


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["raster"] and len(a) == 3:
        raster(a[1], a[2])
    else:
        raise SystemExit(__doc__)
