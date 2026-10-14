"""The complex as the runtime, phase 1: calibration, the acceptance run and the timings (NOTES, "The
complex as the runtime: one fixed mesh"; the brief .superpowers/sdd/2026-10-14-complex-runtime).

The metric is the runtime's (METRIC: the shy distance at 0.5 m on the gap complex, its name
carrying every constant that can change a score). The raster is uni+apwfs at h 0.5 (RASTER).

    S=research/roadless/gap_calib.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S preset <id> <simp|cheap>      # one preset on the complex (cluster.py cpu)
    uv run python $S baseline <ids,|@file> <procs>  # the fixed mesh's baseline against the rebuild
    uv run python $S clearings <ids,|@file> <procs> # every clearing on the fixed mesh and rebuilt
    uv run python $S rasterscore <id> <cpu|gpu>     # every clearing on the raster (cluster GPU)
    uv run python $S timing <ids,|@file> <procs>    # the breakdowns, before and after
    uv run python $S report                         # gap_rows/summary_runtime_*.csv

`preset`: the cheap preset (polish_greedy, S0.01cat P64w8) or SIMP's default (relax.one, PLAN)
at area D BUDGET, power 1 (P), free first, on the complex; their own rows (polish_rows/<METRIC>/,
relax_rows/<METRIC>/).

`baseline`: per block the fixed mesh (built afresh: its time and peak memory) against the
complex rebuilt (gap_compare.ComplexModel, gap_flow's mesh): P0, P_{1/2},0, J_2,0, the walled-in
and no-demand homes, each home's escape time; rows ROWS/baseline/<id>.parquet.

`clearings`: per block every clearing of SOURCES present (the raster's presets under P and the
complex's), scored on the fixed mesh and on the complex rebuilt (P, P_{1/2}, J_2, the enclosed
population and count), and the homes each reaches against the complex grown by w / 2 at width 0
(gap_complex's), home for home; rows ROWS/clearings/<id>.parquet.

`rasterscore`: every clearing of SOURCES present scored on the raster (RASTER, h 0.5: P,
P_{1/2}, J_2, B (i), as width.cross scores); rows ROWS/raster/<id>.parquet.

`timing`: per block, the fixed mesh's build and load, then the breakdown (factors, reach,
assembly, factorization, solves, gradient) of an exact score (a random clearing at BUDGET, under
the population's and P_{1/2}'s demands, as the scorer does), of a SIMP iteration (value_sgrad in
the eps world at PLAN's eps and q 3), and of the tension, each TIMING_REPS times, AFTER (the
runtime: one nested-dissection order restricted to each state's unknowns, solves refined to the
backward-error floor) and BEFORE (as the mesh was first run: SuperLU's COLAMD at every
factorization, solves refined toward a 1e-12 residual, eight steps where the floor is above it);
a region (<id>@major) its baseline only. Rows ROWS/timing/<id>.parquet.
"""
from __future__ import annotations

import multiprocessing
import os
import resource
import socket
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clear  # noqa: E402
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_compare as gcmp  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_flow as gf  # noqa: E402
import gap_study  # noqa: E402
import lifted  # noqa: E402
import metrics  # noqa: E402
import polish_greedy  # noqa: E402
import relax  # noqa: E402

METRIC = gcl.complex_of("cx+shy0.5+s0.125r1b1+R0.5g4x2q4")    # owner 2026-10-14
RASTER = "uni+apwfs"
MESH = lifted.UniformMesh(0.5, offset=lifted.OFFSET)
PLAN = relax.plan_of("fw0.q3.i10.t0.001.e0.0001.k1s1e-06.p256w8x2")     # SIMP's default
CHEAP = ("S0.01cat", 64, 8, False)      # the cheap preset: picker, tries, width, pairs
BUDGET = relax.D_LENS                   # area D 0.05
POWER = 1.0                             # P, the headline (owner 2026-10-12)
POP = common.PopulationDemand()
PQ = common.PriorityDemand(0.5)
TIMING_REPS = 2
ROWS = HERE / "gap_rows" / "runtime"


def _peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


# ------------------------------------------------------------------ the clearings

def sources(metric_name: str) -> dict[str, Path]:
    """Per preset its rows' directory under a metric (relax.rows_of, polish_greedy.rows_of)."""
    picker, tries, width, pairs = CHEAP
    return {"simp": relax.rows_of(PLAN, metric_name, BUDGET),
            "cheap": polish_greedy.rows_of(picker, tries, width, pairs, MESH, metric_name,
                                           BUDGET)}


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

def preset(bid: str, which: str) -> None:
    metric = metrics.metric_of(METRIC.name, device="cpu", mesh=MESH)
    if which == "simp":
        out = relax.rows_of(PLAN, metric.name, BUDGET) / f"{bid}_p{POWER:g}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        row = relax.one(bid, POWER, metric, PLAN, BUDGET, demand=POP)
        row.update(host=socket.gethostname(), peak_rss_gb=_peak_gb(), wall=time.time() - t0)
        pd.DataFrame([row]).to_parquet(out)
    elif which == "cheap":
        picker, tries, width, pairs = CHEAP
        polish_greedy.main(bid, POWER, metric, picker, tries, width, BUDGET, MESH, pairs)
    else:
        raise SystemExit(f"unknown preset {which!r}: simp or cheap")
    print(f"{bid} {which}: peak {_peak_gb():.1f} GB", flush=True)


# ------------------------------------------------------------------ baseline

def _rebuild_ref(b, cm: gcmp.ComplexModel | None) -> gcmp.ComplexModel:
    return cm if cm is not None else gcmp.ComplexModel(b, METRIC.fs, METRIC.form)


def baseline_one(bid: str) -> str | None:
    out = ROWS / "baseline" / f"{bid}.parquet"
    if out.exists():
        return None
    try:
        [b] = common.build_blocks([bid])
        t0 = time.time()
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            fx = gcl.Fixed.of(b, METRIC.spec, METRIC.fs, METRIC.form, lumped=False,
                              cache=Path(tmp))
            t_build = time.time() - t0
            peak_build = _peak_gb()
        m = gcl.Model(b, fx, spec=METRIC.spec, fs=METRIC.fs, form=METRIC.form,
                      pop=common.POPULATIONS["area"].weights(np.asarray(b.buildings.outlines)),
                      demand=POP)
        t_model = time.time() - t0
        cm = _rebuild_ref(b, None)
        live = ~m.nodem & ~cm.walled
        ratio = m.u_pop0[live] / cm.u0[live]
        fm = m.fm
        row = dict(block=bid, size=gap_study.size_of(bid), n=m.n,
                   P0=m.base.P, P0_rebuild=cm.P0, Pq0=m.base.Pq, Pq0_rebuild=cm.Pq0,
                   J2_0=m.base.J2, J2_0_rebuild=cm.J2_0,
                   walled=int(m.walled0.sum()), walled_rebuild=int(cm.walled.sum()),
                   walled_alike=bool(np.array_equal(m.nodem, cm.walled)),
                   home_rel_max=float(np.abs(ratio - 1).max()) if live.any() else 0.0,
                   home_rel_med=float(np.median(np.abs(ratio - 1))) if live.any() else 0.0,
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

def _complex_live(m: gcl.Model, removed: np.ndarray) -> np.ndarray:
    cx = gc.build(m.geo, m.spec, ~removed, carve=None, grow=m.form.grow)
    return ~gc.walled(gc.home_exits(gc.homes_of(m.geo, cx), gc.exit_widths(cx)),
                      m.form.filtration_w)


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
            live_cx = _complex_live(m, x)
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


# ------------------------------------------------------------------ timing

def _legacy_factor(A):
    """The factorization as the fixed mesh was first run (the measurement's BEFORE): SuperLU's
    COLAMD at every factorization, each solve refined toward a residual 1e-12 of the right-hand
    side, at most eight steps (the floor sits near 1e-9: all eight taken)."""
    from scipy.sparse.linalg import splu
    Ac = A.tocsr()
    lu = splu(A.tocsc(), permc_spec="COLAMD")

    class _Legacy:
        def solve(self, b):
            x = lu.solve(b)
            nb = float(np.abs(b).max())
            if nb == 0.0:
                return x
            for _ in range(8):
                r = b - Ac @ x
                if float(np.abs(r).max()) <= 1e-12 * nb:
                    return x
                x = x + lu.solve(r)
            return x
    return _Legacy()


@contextmanager
def _before(m: gcl.Model):
    """The model's factorizations as first run (_legacy_factor)."""
    old = gf.factor_ordered
    gf.factor_ordered = lambda A, p: _legacy_factor(A)
    try:
        yield
    finally:
        gf.factor_ordered = old


def _measure(m: gcl.Model, what: str, fn) -> dict:
    m.timing = gcl.Timing()
    t0 = time.time()
    fn()
    return dict(what=what, wall=time.time() - t0, **{f"t_{k}": v for k, v in m.timing.t.items()},
                **{f"n_{k}": v for k, v in m.timing.n.items()})


def timing_one(bid: str) -> str | None:
    out = ROWS / "timing" / f"{bid}.parquet"
    if out.exists():
        return None
    try:
        [b] = common.build_blocks([bid])
        import tempfile
        rows = []
        with tempfile.TemporaryDirectory() as tmp:
            t0 = time.time()
            fx = gcl.Fixed.of(b, METRIC.spec, METRIC.fs, METRIC.form, lumped=False,
                              cache=Path(tmp))
            rows.append(dict(what="build", wall=time.time() - t0, peak_gb=_peak_gb()))
            t0 = time.time()
            fx = gcl.Fixed.of(b, METRIC.spec, METRIC.fs, METRIC.form, lumped=False,
                              cache=Path(tmp))
            rows.append(dict(what="load", wall=time.time() - t0, peak_gb=_peak_gb()))
        t0 = time.time()
        m = gcl.Model(b, fx, spec=METRIC.spec, fs=METRIC.fs, form=METRIC.form,
                      pop=common.POPULATIONS["area"].weights(np.asarray(b.buildings.outlines)),
                      demand=POP)
        rows.append(dict(what="model (baseline: the order, 4 solves)", wall=time.time() - t0,
                         peak_gb=_peak_gb(), unknowns=m.cells0, cells=m.c.n))
        region = "@" in bid
        rng = np.random.default_rng(20261014)
        x = np.zeros(m.n, dtype=bool)
        order = rng.permutation(m.n)
        x[order[np.cumsum(m.cost[order]) <= BUDGET]] = True
        rel = m.relaxation(POWER, q=PLAN.qmax, rtol=PLAN.rtol, eps=PLAN.eps)
        x_frac = np.full(m.n, BUDGET)
        for when, ctx in (("after", None), ("before", _before)):
            for rep in range(TIMING_REPS):
                def score():
                    m.lenses(x)

                def iteration():
                    rel.value_sgrad(x_frac)

                def tension():
                    m.removed[:] = x
                    m.tension(POWER, restore=False)
                    m.removed[:] = False

                def baseline():
                    m.lenses(np.zeros(m.n, dtype=bool))
                cases = ([("baseline score", baseline)] if region else
                         [("exact score", score), ("SIMP iteration", iteration),
                          ("tension", tension)])
                for what, fn in cases:
                    if ctx is None:
                        r = _measure(m, what, fn)
                    else:
                        with ctx(m):
                            r = _measure(m, what, fn)
                    rows.append(dict(r, when=when, rep=rep, peak_gb=_peak_gb()))
                    print(f"{bid} {when} {what}: {r['wall']:.2f}s (factorization "
                          f"{r['t_factorization']:.2f}s, solves {r['t_solves']:.2f}s); peak "
                          f"{_peak_gb():.1f} GB", flush=True)
        df = pd.DataFrame(rows).assign(block=bid, n=m.n, unknowns0=m.cells0, cells=m.c.n,
                                       host=socket.gethostname())
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        return None
    except Exception as e:      # noqa: BLE001
        import traceback
        traceback.print_exc()
        return f"{bid}: {e!r}"


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
    if cmd == "preset":
        preset(argv[1], argv[2])
    elif cmd == "rasterscore":
        rasterscore(argv[1], argv[2])
    elif cmd in ("baseline", "clearings", "timing"):
        fn = {"baseline": baseline_one, "clearings": clearings_one, "timing": timing_one}[cmd]
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
