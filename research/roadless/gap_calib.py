"""The complex as the runtime: calibration, the acceptance runs and the timings (NOTES, "The complex
as the runtime: one fixed mesh" and "... its speed and a GPU path"; the briefs
.superpowers/sdd/2026-10-14-complex-runtime and 2026-10-14-complex-speed).

The metric is the runtime's (METRIC: the shy distance at 0.5 m on the gap complex, its name
carrying every constant that can change a score, the code's digest among them). The raster is
uni+apwfs at h 0.5 (RASTER).

    S=research/roadless/gap_calib.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S warm <id>                      # the block's fixed mesh into gap_clear.CACHE
    uv run python $S preset <id> <simp|cheap>[E][L] <cpu|gpu>  # a preset on the complex
    uv run python $S presets <id> <items,> <cpu|gpu>   # several, one after another (one GPU run)
    uv run python $S baseline <ids,|@file> <procs>  # the fixed mesh's baseline against the rebuild
    uv run python $S clearings <ids,|@file> <procs> # every clearing on the fixed mesh and rebuilt
    uv run python $S rasterscore <id> <cpu|gpu>     # every clearing on the raster (cluster GPU)
    uv run python $S speed <ids,|@file> <procs> <cpu|gpu>  # the breakdowns on a device
    uv run python $S rescore <ids,|@file> <procs>   # the stored runs re-scored under their name
    uv run python $S report                         # gap_rows/summary_runtime_*.csv

`preset`: the cheap preset (polish_greedy, S0.01cat P64w8) or SIMP's default (relax.one, PLAN)
at area D BUDGET, power 1 (P), free first, on the complex, on the device's factorization; `E` the
polish's candidates scored exactly (search.Exactly), `L` the metric's eps world with one cell per
building's interior (+relaxlumped). Rows ROWS/preset/<device>/<item>/<id>.parquet (the run's row,
its item, metric, host, peak memory and wall time; on a GPU the device's memory in use, its most
over the run, `gpu_gb`): the
acceptance's own, so that a run on one device never overwrites another's or the stored rows
(relax_rows/<METRIC>/, polish_rows/<METRIC>/, phase 1's, which `report` compares them with).

`baseline`: per block the fixed mesh (built afresh: its time and peak memory) against the
complex rebuilt (gap_compare.ComplexModel, gap_flow's mesh): P0, P_{1/2},0, J_2,0, the walled-in
and no-demand homes, each home's escape time; rows ROWS/baseline/<id>.parquet.

`clearings`: per block every clearing of SOURCES present (the raster's presets under P and the
complex's), scored on the fixed mesh and on the complex rebuilt (P, P_{1/2}, J_2, the enclosed
population and count), and the homes each reaches against the complex grown by w / 2 at width 0
(gap_complex's), home for home; rows ROWS/clearings/<id>.parquet.

`rasterscore`: every clearing of SOURCES present scored on the raster (RASTER, h 0.5: P,
P_{1/2}, J_2, B (i), as width.cross scores); rows ROWS/raster/<id>.parquet.

`speed`: per block on the device, the fixed mesh's load and the model's baseline (with the
factorization's plan), then the breakdown (factors, reach, assembly, factorization, solves,
gradient) of an exact score (a random clearing at BUDGET, under the population's and P_{1/2}'s
demands, as the scorer does), of a SIMP iteration (value_sgrad in the eps world at PLAN's eps and
q 3) and of the tension, each SPEED_REPS times, and the SIMP iteration and the tension of the
+relaxlumped metric's eps world; a region (<id>@major) its baseline only. Rows
ROWS/speed/<device>/<id>.parquet. (Phase 1's `timing`, BEFORE and AFTER, is in
summary_runtime_timing.csv.)

`rescore`: the stored runs of the complex's presets (SOURCES' cx_*), each clearing re-scored
under the metric's name (its rows' directories migrated to it), against the score its row holds;
rows ROWS/rescore/<id>.parquet.
"""
from __future__ import annotations

import multiprocessing
import os
import re
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clear  # noqa: E402
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_compare as gcmp  # noqa: E402
import gap_factor  # noqa: E402
import gap_study  # noqa: E402
import lifted  # noqa: E402
import metrics  # noqa: E402
import polish_greedy  # noqa: E402
import relax  # noqa: E402
import search  # noqa: E402

BASE = "cx+shy0.5+s0.125r1b1+R0.5g4x2q4"                     # owner 2026-10-14
METRIC = gcl.complex_of(f"{BASE}+v{gcl.CODE}", factorization=gap_factor.SuperLU())
RASTER = "uni+apwfs"
MESH = lifted.UniformMesh(0.5, offset=lifted.OFFSET)
PLAN = relax.plan_of("fw0.q3.i10.t0.001.e0.0001.k1s1e-06.p256w8x2")     # SIMP's default
CHEAP = ("S0.01cat", 64, 8, False)      # the cheap preset: picker, tries, width, pairs
BUDGET = relax.D_LENS                   # area D 0.05
POWER = 1.0                             # P, the headline (owner 2026-10-12)
POP = common.PopulationDemand()
PQ = common.PriorityDemand(0.5)
SPEED_REPS = 2
GPU_SAMPLE_S = 0.5          # _GpuPeak's sampling (measurement only)
ROWS = HERE / "gap_rows" / "runtime"
ITEM = re.compile(r"(simp|cheap)(E?)(L?)")    # a preset item: the polish scoring, the eps world


def _peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


# ------------------------------------------------------------------ the clearings

def sources(metric_name: str) -> dict[str, Path]:
    """Per preset its rows' directory under a metric (relax.rows_of, polish_greedy.rows_of)."""
    picker, tries, width, pairs = CHEAP
    return {"simp": relax.rows_of(PLAN, metric_name, BUDGET),
            "cheap": polish_greedy.rows_of(picker, tries, width, pairs, search.InEps(), MESH,
                                           metric_name, BUDGET)}


SOURCES = {f"{who}_{preset}": (name, preset) for who, name in (("raster", RASTER),
                                                               ("cx", METRIC.name))
           for preset in ("simp", "cheap")}


def clearings_of(bid: str, n: int) -> dict[str, np.ndarray]:
    """Every clearing of SOURCES present for the block (bool per building)."""
    out = {}
    for key, (name, preset) in SOURCES.items():
        f = sources(name)[preset] / f"{bid}_p{POWER:g}.parquet"
        if not f.exists():
            continue
        r = pd.read_parquet(f).iloc[0]
        x = np.zeros(n, dtype=bool)
        x[np.asarray(r.simp_cleared if preset == "simp" else r.cleared, dtype=np.int64)] = True
        out[key] = x
    return out


# ------------------------------------------------------------------ preset

def _gpu_gb() -> float:
    """The device's memory in use (one process a card: this run's), GB."""
    import cupy
    free, total = cupy.cuda.runtime.memGetInfo()
    return float(total - free) / 2 ** 30


class _GpuPeak:
    """The device's memory in use, its most over a run, sampled every GPU_SAMPLE_S seconds by a
    thread (the factorizations' plans live and die with their models, inside the run)."""

    def __init__(self) -> None:
        import threading
        self.peak = 0.0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self) -> None:
        while not self.done.wait(GPU_SAMPLE_S):
            self.peak = max(self.peak, _gpu_gb())

    def __enter__(self) -> _GpuPeak:
        self.peak = _gpu_gb()
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.done.set()
        self.thread.join()
        self.peak = max(self.peak, _gpu_gb())


def warm(bid: str) -> None:
    """The block's fixed mesh built into the cache (gap_clear.CACHE: a node's own disk), before a
    run's tasks for the block load it at once."""
    [b] = common.build_blocks([bid])
    t0 = time.time()
    fx = METRIC.fixed(b, cache=gcl.CACHE)
    print(f"{bid}: the fixed mesh, {fx.fm.nt} triangles, {time.time() - t0:.0f}s (built in "
          f"{fx.t_build:.0f}s); peak {_peak_gb():.1f} GB", flush=True)


def preset(bid: str, item: str, device: str) -> None:
    m = ITEM.fullmatch(item)
    if m is None:
        raise SystemExit(f"unknown preset item {item!r}: <simp|cheap>[E][L]")
    which, exact, lumped = m.groups()
    metric = metrics.metric_of(f"{BASE}{'+relaxlumped' if lumped else ''}+v{gcl.CODE}",
                               device=device, mesh=MESH)
    scoring = search.polish_scoring_of(exact)
    out = ROWS / "preset" / device / item / f"{bid}.parquet"
    t0 = time.time()

    def run() -> dict[str, object]:
        if which == "simp":
            return relax.one(bid, POWER, metric, PLAN._replace(polish_scoring=scoring), BUDGET,
                             demand=POP)
        picker, tries, width, pairs = CHEAP
        return polish_greedy.run(bid, POWER, metric, picker, tries, width, BUDGET, MESH, pairs,
                                 scoring=scoring)
    if device == "gpu":
        with _GpuPeak() as peak:
            row = run()
        gpu_gb = peak.peak
    else:
        row, gpu_gb = run(), np.nan
    row.update(item=item, metric=metric.name, host=socket.gethostname(), peak_rss_gb=_peak_gb(),
               gpu_gb=gpu_gb, wall=time.time() - t0)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_parquet(out)
    print(f"{bid} {item} on {device}: peak {_peak_gb():.1f} GB, {row['wall']:.0f}s", flush=True)


# ------------------------------------------------------------------ baseline

def baseline_one(bid: str) -> str | None:
    out = ROWS / "baseline" / f"{bid}.parquet"
    if out.exists():
        return None
    try:
        [b] = common.build_blocks([bid])
        t0 = time.time()
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            fx = METRIC.fixed(b, cache=Path(tmp))
            t_build = time.time() - t0
            peak_build = _peak_gb()
        m = METRIC.model(b, fx, demand=POP)
        t_model = time.time() - t0
        cm = gcmp.ComplexModel(b, METRIC.fs, METRIC.form)
        live = ~m.nodem & ~cm.walled
        ratio = m.u_pop0[live] / cm.u0[live]
        fm = m.fm
        row = dict(block=bid, n=m.n,
                   P0=m.base.P, P0_rebuild=cm.P0, Pq0=m.base.Pq, Pq0_rebuild=cm.Pq0,
                   J2_0=m.base.J2, J2_0_rebuild=cm.J2_0,
                   walled=int(m.walled0.sum()), walled_rebuild=int(cm.walled.sum()),
                   walled_alike=bool(np.array_equal(m.nodem, cm.walled)),
                   home_rel_max=float(np.nanmax(np.abs(ratio - 1))) if live.any() else 0.0,
                   home_rel_med=float(np.nanmedian(np.abs(ratio - 1))) if live.any() else 0.0,
                   walled_apart=" ".join(map(str, np.flatnonzero(m.nodem ^ cm.walled))),
                   nt=fm.nt, cells=m.c.n, unknowns0=m.cells0, cells_rebuild=cm.cells0,
                   nt_rebuild=cm.nt0, unflat=fm.unflat, splits=fm.splits, rounds=fm.rounds,
                   flat=fm.flat, skipped=fm.skipped, t_build=t_build, t_model=t_model,
                   t_rebuild=cm.t_base, peak_build_gb=peak_build, peak_gb=_peak_gb(),
                   host=socket.gethostname())
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_parquet(out)
        print(f"{bid}: P0 {row['P0'] / row['P0_rebuild'] - 1:+.1e} Pq0 "
              f"{row['Pq0'] / row['Pq0_rebuild'] - 1:+.1e} J2 "
              f"{row['J2_0'] / row['J2_0_rebuild'] - 1:+.1e} walled {row['walled']} / "
              f"{row['walled_rebuild']} homes {row['home_rel_max']:.1e}; build {t_build:.0f}s "
              f"{peak_build:.1f} GB", flush=True)
        return None
    except Exception as e:      # noqa: BLE001 -- a pool's task reports its block and goes on
        import traceback
        traceback.print_exc()
        return f"{bid}: {e!r}"


# ------------------------------------------------------------------ clearings

def clearings_one(bid: str) -> str | None:
    out = ROWS / "clearings" / f"{bid}.parquet"
    try:
        [b] = common.build_blocks([bid])
        m = METRIC.clearing(b, demand=POP)
        picks = clearings_of(bid, m.n)
        done = pd.read_parquet(out) if out.exists() else None
        if done is not None:
            picks = {k: x for k, x in picks.items() if k not in set(done.clearing)}
        if not picks:
            return None
        cm = gcmp.ComplexModel(b, METRIC.fs, METRIC.form)
        rows = []
        for key, x in {"baseline": np.zeros(m.n, dtype=bool), **picks}.items():
            if done is not None and key in set(done.clearing):
                continue
            t0 = time.time()
            L = m.lenses(x)
            t_fixed = time.time() - t0
            r, u = cm.score(~x, None)
            live_cx = m.complex_live(x)
            Pq_r = r["Pq"]
            J2_r = float(np.nansum(cm.pop * u ** 2))
            rows.append(dict(
                block=bid, clearing=key, n_cleared=int(x.sum()), D=float(m.cost[x].sum()),
                cleared=" ".join(map(str, np.flatnonzero(x))),
                L_P=1 - L.P / m.base.P, L_Pq=1 - L.Pq / m.base.Pq,
                L_J2=1 - (L.J2 / m.base.J2) ** 0.5, enclosed=L.enclosed, n_enclosed=L.n_enclosed,
                L_P_rebuild=r["L_P"], L_Pq_rebuild=r["L_Pq0.5"], L_J2_rebuild=r["L_J2"],
                enclosed_rebuild=r["enclosed"], n_enclosed_rebuild=r["n_enclosed"],
                reach_apart=int((L.live ^ live_cx).sum()),
                reach_apart_homes=" ".join(map(str, np.flatnonzero(L.live ^ live_cx))),
                P=L.P, Pq=L.Pq, J2=L.J2, P_rebuild=r["P"], Pq_rebuild=Pq_r, J2_rebuild=J2_r,
                unknowns=L.unknowns, t_fixed=t_fixed, t_rebuild=r["t"]))
            print(f"{bid} {key}: L_P {rows[-1]['L_P']:.4f} / rebuilt {r['L_P']:.4f}, L_Pq "
                  f"{rows[-1]['L_Pq']:.4f} / {r['L_Pq0.5']:.4f}, L_J2 {rows[-1]['L_J2']:.4f} / "
                  f"{r['L_J2']:.4f}, enclosed {L.enclosed:.5f} / {r['enclosed']:.5f}, reach apart "
                  f"{rows[-1]['reach_apart']} ({t_fixed:.1f}s against {r['t']:.0f}s)", flush=True)
        df = pd.DataFrame(rows)
        if done is not None:
            df = pd.concat([done, df], ignore_index=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        return None
    except Exception as e:      # noqa: BLE001
        import traceback
        traceback.print_exc()
        return f"{bid}: {e!r}"


# ------------------------------------------------------------------ the raster's scores

def rasterscore(bid: str, device: str) -> None:
    """Every clearing of SOURCES present on the raster (width.cross's way)."""
    import width
    out = ROWS / "raster" / f"{bid}.parquet"
    t0 = time.time()
    [b] = common.build_blocks([bid])
    p = width.params(RASTER, device)
    sc = common.Scorer(b, MESH, p, population=common.POPULATIONS["area"], demand=POP)
    g, n = sc.grid, len(sc.polys)
    picks = clearings_of(bid, n)
    inj_q = PQ.injection(sc.w, sc.u0, sc.enclosed)
    f_q = sc.pl.field(inj_q, g.inside.shape)
    sol_q = lifted.solve(g, sc.free0, f_q, p)
    lab = g.label_sub(sc.polys)
    rows = []
    for key, x in picks.items():
        o = width._open_of(g, lab, np.flatnonzero(x), n)
        sy = lifted.System(g, o, p)
        s1 = lifted.solve(g, o, sc.f, p, system=sy, rtol=clear.RTOL_SCORE)
        sq = lifted.solve(g, o, f_q, p, system=sy, rtol=clear.RTOL_SCORE)
        rows.append(dict(block=bid, clearing=key, n_cleared=int(x.sum()),
                         D=float(sc.w[x].sum() / sc.w.sum()), enclosed=sc.enclosed_left(o),
                         L_P=1.0 - s1.P / sc.P0, L_J2=sc.perm_p(sc.home_u_of(s1, o), 2.0),
                         L_Pq=1.0 - sq.P / sol_q.P))
        del sy
        p.solver.release()
        print(f"{bid} raster {key}: L_P {rows[-1]['L_P']:.4f} L_Pq {rows[-1]['L_Pq']:.4f} "
              f"L_J2 {rows[-1]['L_J2']:.4f} enclosed {rows[-1]['enclosed']:.5f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).assign(t=time.time() - t0, host=socket.gethostname()).to_parquet(out)


# ------------------------------------------------------------------ speed

def _measure(m: gcl.Model, what: str, fn) -> dict:
    m.timing = gcl.Timing()
    t0 = time.time()
    fn()
    return dict(what=what, wall=time.time() - t0, **{f"t_{k}": v for k, v in m.timing.t.items()},
                **{f"n_{k}": v for k, v in m.timing.n.items()})


_SPEED_DEVICE: list[str] = []      # the pool's device (speed's argument), set before it forks


def speed_one(bid: str) -> str | None:
    [device] = _SPEED_DEVICE
    out = ROWS / "speed" / device / f"{bid}.parquet"
    if out.exists():
        return None
    try:
        [b] = common.build_blocks([bid])
        rows = []
        t0 = time.time()
        fx = METRIC.fixed(b, cache=gcl.CACHE)
        rows.append(dict(what="load", wall=time.time() - t0, peak_gb=_peak_gb()))
        models = {}
        for lumped in ("", "+relaxlumped"):
            if "@" in bid and lumped:
                continue
            metric = metrics.metric_of(f"{BASE}{lumped}+v{gcl.CODE}", device=device, mesh=MESH)
            t0 = time.time()
            assert isinstance(metric, gcl.ComplexMetric)
            mdl = metric.model(b, fx, demand=POP)
            models[lumped] = mdl
            rows.append(dict(what=f"model{lumped} (baseline: the plan, 4 solves)",
                             wall=time.time() - t0, peak_gb=_peak_gb(), unknowns=mdl.cells0,
                             cells=mdl.relax_sys.n))
        m = models[""]
        rng = np.random.default_rng(20261014)
        x = np.zeros(m.n, dtype=bool)
        order = rng.permutation(m.n)
        x[order[np.cumsum(m.cost[order]) <= BUDGET]] = True
        x_frac = np.full(m.n, BUDGET)
        cases = []
        if "@" in bid:
            cases.append(("baseline score", m, lambda mm: mm.lenses(np.zeros(mm.n, dtype=bool))))
        else:
            cases.append(("exact score", m, lambda mm: mm.lenses(x)))
            for lumped, mm in models.items():
                rel = mm.relaxation(POWER, q=PLAN.qmax, rtol=PLAN.rtol, eps=PLAN.eps)

                def tension(mm=mm):
                    mm.removed[:] = x
                    mm.tension(POWER, restore=False)
                    mm.removed[:] = False
                cases.append((f"SIMP iteration{lumped}", mm,
                              lambda _mm, rel=rel: rel.value_sgrad(x_frac)))
                cases.append((f"tension{lumped}", mm, lambda _mm, f=tension: f()))
        for rep in range(SPEED_REPS):
            for what, mm, fn in cases:
                r = _measure(mm, what, lambda fn=fn, mm=mm: fn(mm))
                rows.append(dict(r, rep=rep, peak_gb=_peak_gb()))
                print(f"{bid} {device} {what}: {r['wall']:.2f}s (factorization "
                      f"{r['t_factorization']:.2f}s, solves {r['t_solves']:.2f}s); peak "
                      f"{_peak_gb():.1f} GB", flush=True)
        df = pd.DataFrame(rows).assign(block=bid, n=m.n, unknowns0=m.cells0, cells=m.c.n,
                                       device=device, host=socket.gethostname(),
                                       gpu_gb=_gpu_gb() if device == "gpu" else np.nan)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        return None
    except Exception as e:      # noqa: BLE001
        import traceback
        traceback.print_exc()
        return f"{bid}: {e!r}"


# ------------------------------------------------------------------ the migration's check

def rescore_one(bid: str) -> str | None:
    """Each stored run of the complex's presets on `bid` (its rows under METRIC's name), its
    clearing re-scored: Lens A under P (perm, at power 1) against the row's."""
    out = ROWS / "rescore" / f"{bid}.parquet"
    try:
        [b] = common.build_blocks([bid])
        m = METRIC.clearing(b, demand=POP)
        rows = []
        for preset, d in sources(METRIC.name).items():
            f = d / f"{bid}_p{POWER:g}.parquet"
            if not f.exists():
                continue
            r = pd.read_parquet(f).iloc[0]
            cleared = r.simp_cleared if preset == "simp" else r.cleared
            stored = float(r.simp_perm if preset == "simp" else r.perm)
            x = np.zeros(m.n, dtype=bool)
            x[np.asarray(cleared, dtype=np.int64)] = True
            J, _P = m.exact(x, POWER)
            perm = 1.0 - J / m.baseline(POWER)[0]
            rows.append(dict(block=bid, preset=preset, stored=stored, rescored=perm,
                             diff=perm - stored, host=socket.gethostname()))
            print(f"{bid} {preset}: stored {stored:.10f}, re-scored {perm:.10f} "
                  f"({perm - stored:+.1e})", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out)
        return None
    except Exception as e:      # noqa: BLE001
        import traceback
        traceback.print_exc()
        return f"{bid}: {e!r}"


# ------------------------------------------------------------------ the summaries

LENSES = ("L_P", "L_Pq", "L_J2")


def _read(sub: str) -> pd.DataFrame:
    fs = sorted((ROWS / sub).glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def _read_nested(sub: str) -> pd.DataFrame:
    """Every row under ROWS/<sub>/, its subdirectories included."""
    fs = sorted((ROWS / sub).rglob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def _kendall(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import kendalltau
    return float(kendalltau(a, b).statistic)


def _b_rank(enclosed: pd.Series, lens: pd.Series) -> np.ndarray:
    """Each clearing's place in B's order: the population it leaves walled in first (less is
    better), then the lens (more is better)."""
    order = np.lexsort((-lens.to_numpy(), enclosed.to_numpy()))
    rank = np.empty(len(order))
    rank[order] = np.arange(len(order))
    return rank


def report() -> None:
    """ROWS' summaries, gap_rows/summary_runtime_<name>.csv (committed; the rows are not)."""
    out = HERE / "gap_rows"
    base = _read("baseline")
    if len(base):
        b = base.assign(dP0=base.P0 / base.P0_rebuild - 1, dPq0=base.Pq0 / base.Pq0_rebuild - 1,
                        dJ2_0=base.J2_0 / base.J2_0_rebuild - 1)
        b = b.assign(size=[gap_study.size_of(x) for x in b.block])
        b[["block", "size", "n", "dP0", "dPq0", "dJ2_0", "walled", "walled_rebuild",
           "walled_alike", "home_rel_max", "home_rel_med", "nt", "nt_rebuild", "cells",
           "unknowns0", "cells_rebuild", "unflat", "splits", "rounds", "flat", "t_build",
           "t_model", "t_rebuild", "peak_build_gb", "peak_gb"]].round(6).to_csv(
            out / "summary_runtime_baseline.csv", index=False)
    cl = _read("clearings")
    if len(cl):
        c = cl.assign(**{f"d{k}": cl[k] - cl[f"{k}_rebuild"] for k in LENSES},
                      d_enclosed=cl.enclosed - cl.enclosed_rebuild,
                      d_n_enclosed=cl.n_enclosed - cl.n_enclosed_rebuild)
        c[["block", "clearing", "n_cleared", "D", *LENSES, *(f"{k}_rebuild" for k in LENSES),
           *(f"d{k}" for k in LENSES), "enclosed", "enclosed_rebuild", "n_enclosed",
           "n_enclosed_rebuild", "d_n_enclosed", "reach_apart", "unknowns", "t_fixed",
           "t_rebuild"]].round(6).to_csv(out / "summary_runtime_clearings.csv", index=False)
    ras = _read("raster")
    if len(cl) and len(ras):
        rows = []
        for bid, g in cl[cl.clearing != "baseline"].groupby("block"):
            r = ras[ras.block == bid].set_index("clearing")
            g = g.set_index("clearing")
            keys = [k for k in g.index if k in r.index]
            row = dict(block=bid, clearings=len(keys))
            for lens in LENSES:
                for k in keys:
                    row[f"{k}_{lens}_cx"] = g.loc[k, lens]
                    row[f"{k}_{lens}_raster"] = r.loc[k, lens]
                if len(keys) >= 2:      # the clearings' order in B's order, metric against metric
                    row[f"tau_{lens}"] = _kendall(
                        _b_rank(g.loc[keys, "enclosed"], g.loc[keys, lens]),
                        _b_rank(r.loc[keys, "enclosed"], r.loc[keys, lens]))
            for k in keys:
                row[f"{k}_enclosed_cx"] = g.loc[k, "enclosed"]
                row[f"{k}_enclosed_raster"] = r.loc[k, "enclosed"]
                row[f"{k}_D"] = g.loc[k, "D"]
            for preset in ("simp", "cheap"):
                a, z = f"raster_{preset}", f"cx_{preset}"
                if a in g.index and z in g.index:
                    xa = set(map(int, str(g.loc[a, "cleared"]).split()))
                    xz = set(map(int, str(g.loc[z, "cleared"]).split()))
                    row[f"jaccard_{preset}"] = len(xa & xz) / max(len(xa | xz), 1)
            rows.append(row)
        pd.DataFrame(rows).round(6).to_csv(out / "summary_runtime_acceptance.csv", index=False)
    tim = _read("timing")
    if len(tim):
        tim.round(4).to_csv(out / "summary_runtime_timing.csv", index=False)
    cud = _read("cudss")
    if len(cud):
        # six significant digits: the residuals are 1e-9
        cud.to_csv(out / "summary_runtime_cudss.csv", index=False, float_format="%.6g")
    runs = []
    for key, (name, preset) in SOURCES.items():
        for f in sorted(sources(name)[preset].glob(f"*_p{POWER:g}.parquet")):
            r = pd.read_parquet(f).iloc[0]
            runs.append(dict(block=r.block, clearing=key,
                             perm=float(r.simp_perm if preset == "simp" else r.perm),
                             D=float(r.simp_D if preset == "simp" else r.D),
                             enclosed=float(r.enclosed), free_D=float(r.free_D),
                             t=float(r.t)))
    if runs:
        pd.DataFrame(runs).round(6).to_csv(out / "summary_runtime_runs.csv", index=False)
    resc = _read("rescore")
    if len(resc):
        resc.to_csv(out / "summary_runtime_rescore.csv", index=False, float_format="%.12g")
    spd = _read_nested("speed")
    if len(spd):
        spd.round(4).to_csv(out / "summary_runtime_speed.csv", index=False)
    pre = _read_nested("preset")
    if len(pre):
        ref = {preset: pd.DataFrame(
            [dict(block=r.block, ref_perm=float(r.simp_perm if preset == "simp" else r.perm),
                  ref_cleared=" ".join(map(str, sorted(
                      r.simp_cleared if preset == "simp" else r.cleared))),
                  ref_t=float(r.t))
             for f in sorted(sources(METRIC.name)[preset].glob(f"*_p{POWER:g}.parquet"))
             for r in [pd.read_parquet(f).iloc[0]]]) for preset in ("simp", "cheap")}
        rows = []
        for _i, r in pre.iterrows():
            preset = "simp" if r["item"].startswith("simp") else "cheap"
            cl = r.simp_cleared if preset == "simp" else r.cleared
            rf = ref[preset]
            rf = rf[rf.block == r.block]
            simp = preset == "simp"
            rows.append(dict(
                block=r.block, item=r["item"], device=r.device, host=r.host,
                perm=r.simp_perm if simp else r.perm, D=r.simp_D if simp else r.D,
                enclosed=r.enclosed, free_D=r.free_D, t=r.t, wall=r.wall,
                peak_rss_gb=r.peak_rss_gb, gpu_gb=r.gpu_gb,
                cleared=" ".join(map(str, sorted(cl))),
                phase1_perm=float(rf.ref_perm.iloc[0]) if len(rf) else np.nan,
                phase1_same=bool(len(rf) and rf.ref_cleared.iloc[0] == " ".join(
                    map(str, sorted(cl)))),
                phase1_t=float(rf.ref_t.iloc[0]) if len(rf) else np.nan))
        pd.DataFrame(rows).sort_values(["item", "device", "block"]).to_csv(
            out / "summary_runtime_presets.csv", index=False, float_format="%.10g")


# ------------------------------------------------------------------ the run

def _pool(fn, jobs: list, procs: int) -> list[str]:
    if procs == 1:
        return [f for f in map(fn, jobs) if f]
    with multiprocessing.get_context("fork").Pool(procs, maxtasksperchild=1) as pool:
        return [f for f in pool.imap_unordered(fn, jobs) if f]


def main(argv: list[str]) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMBA_NUM_THREADS"):
        if os.environ.get(v) != "1":
            raise SystemExit(f"{v}=1 is required (single-threaded workers)")
    cmd = argv[0]
    failed: list[str] = []
    if cmd == "warm":
        warm(argv[1])
    elif cmd == "preset":
        preset(argv[1], argv[2], argv[3])
    elif cmd == "presets":
        for item in argv[2].split(","):
            preset(argv[1], item, argv[3])
    elif cmd == "rasterscore":
        rasterscore(argv[1], argv[2])
    elif cmd == "speed":
        _SPEED_DEVICE.append(argv[3])
        failed = _pool(speed_one, gap_study.ids_of(argv[1]), int(argv[2]))
    elif cmd in ("baseline", "clearings", "rescore"):
        fn = {"baseline": baseline_one, "clearings": clearings_one, "rescore": rescore_one}[cmd]
        failed = _pool(fn, gap_study.ids_of(argv[1]), int(argv[2]))
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)
    if failed:
        print("FAILED:\n" + "\n".join(failed), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
