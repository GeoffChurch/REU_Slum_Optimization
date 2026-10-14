"""The gap complex, stage 2: the uniform law on the complex against the raster (NOTES, "The gap
complex: passability, enclosure and flow without the raster").

Three models of the same clearings, each scored in B's order (the population left enclosed, then
the lens) on the lenses P, P_{1/2} (common.PriorityDemand(0.5), weights from the model's own
baseline) and J_2:
- `complex`: the complex's isotropic two-point scheme (gap_flow), every 0/1 clearing by the
  complex rebuilt (gap_complex.build on the standing footprints less a road's corridor), the
  demand fixed at baseline (gap_flow.point_demand), under a width form (`form_of`: close<w>,
  shy<w>, ramp<w> with RAMP_S);
- `iso`: an isotropic model on the raster under Fws (`RasterIso`: one unknown a cell, the steps of
  lifted's pattern at their face apertures, lifted.step_widths, each axis k weighing m_k / |v_k|^2,
  no heading and no turning: the same open geometry and placement as the metric, conductivity
  sum_k m_k e_k e_k^T = pi / 2 I);
- `lifted`: the metric (uni+apwfs, ell 3 m, K 8), its rows from aperture.py's `score` under Fws
  (aperture_rows/score/o0/Fws/, the 13 tuning blocks; not recomputed).

    S=research/roadless/gap_compare.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S complex <ids,|@file> <procs> <R> <levels> <forms,>  # compare/complex_*
    uv run python $S iso <ids,|@file> <procs> [mesh]         # rows gap_rows/compare/iso_<mesh>/
    uv run python $S converge <id> <R,R,...> <levels> <form>  # P against R on one block
    uv run python $S flowsweep <ids,|@file> <procs> <forms,>   # the sweep's flow part
    uv run python $S interiors <ids,|@file> <form>        # one node per building against meshed
    uv run python $S region <id@major> <forms,>          # a region's stored clearings (REGION)
    uv run python $S lanes                     # the raster's width laws on the walled channel
    uv run python $S gates <ids,>              # FFT homogenization's gate test, re-checked

Fields (as aperture.py's): the baseline; the stored clearings (width.STORED: SIMP's default and the
cheap preset, clearings.Store), each with the homes the model leaves enclosed freed first where it
leaves any (`+free`: the model's own free first, the clearing held, no budget); item 4's road
prefixes at area D BUDGET (width.ROADS, road_rescore's prefixes, common.Carve's rule).
"""
from __future__ import annotations

import multiprocessing
import os
import resource
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clear  # noqa: E402
import clearings  # noqa: E402
import common  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_flow as gf  # noqa: E402
import gap_study  # noqa: E402
import lifted  # noqa: E402
import road_rescore as rr  # noqa: E402
import width  # noqa: E402

# ------------------------------------------------------------------ the run's configuration
S_M = 0.125                       # the complex's outline spacing (stage 1's choice)
SIGMA = gf.SIGMA_ISO              # gap_flow.FlowSpec.sigma: lifted's isotropic limit
QUAD = 4                          # demand sub-points per triangle
BUDGET = width.BUDGET             # area D 0.05
STORED = width.STORED
ROADS = width.ROADS
AREA = common.POPULATIONS["area"]
POP = common.PopulationDemand()
PQ = common.PriorityDemand(0.5)
LIFTED_ROWS = HERE / "aperture_rows" / "score" / "o0" / "Fws"
ROWS = HERE / "gap_rows" / "compare"


GRADE = 2.0                       # gap_flow.FlowSpec.grade: a coarser level's clearance per unit
                                  # of its spacing


def flow_spec(R: float, levels: int) -> gf.FlowSpec:
    return gf.FlowSpec(R=R, levels=levels, grade=GRADE, sigma=SIGMA, quad=QUAD,
                       offset=gf.OFFSET)


RAMP_S = 0.5                      # the ramp's length (m): full conductance 0.75 m from a wall


def form_of(token: str) -> gf.Width:
    """close<w> | shy<w> | ramp<w> (its s RAMP_S): gap_flow's width forms."""
    for prefix, make in (("close", lambda w: gf.Closing(w=w)), ("shy", lambda w: gf.Shy(w=w)),
                         ("ramp", lambda w: gf.Ramp(w=w, s=RAMP_S))):
        if token.startswith(prefix):
            return make(float(token[len(prefix):]))
    raise ValueError(f"unknown width form {token!r}: close<w>, shy<w> or ramp<w>")


def _peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


# ------------------------------------------------------------------ the clearings

def fields_of(b, n: int) -> dict[str, tuple[str, np.ndarray, object]]:
    """Per field: (kind, the buildings cleared (bool), a road's corridor in UTM or None)."""
    import geopandas as gpd

    from reblock.budget import road_corridor
    store = clearings.Store.committed()
    out: dict[str, tuple[str, np.ndarray, object]] = {
        "baseline": ("baseline", np.zeros(n, dtype=bool), None)}
    for name, run in STORED.items():
        x = np.zeros(n, dtype=bool)
        x[store.cleared(run, b, step=1)] = True
        out[name] = ("stored", x, None)
    for method in ROADS:
        pre = pd.read_parquet(rr.prefix_dir(width.ROAD_ROWS) / f"{b.block_id}__{method}.parquet")
        row = pre[(pre["pop"] == "area") & np.isclose(pre.budget, BUDGET)]
        [m] = row.m.astype(int)
        ordered = gpd.read_parquet(rr.ordered_path(width.ROAD_ROWS / "scratch", b.block_id,
                                                   method))
        out[method] = ("road", np.zeros(n, dtype=bool), road_corridor(ordered.iloc[:m]))
    return out


def _road_D(b, method: str) -> float:
    pre = pd.read_parquet(rr.prefix_dir(width.ROAD_ROWS) / f"{b.block_id}__{method}.parquet")
    row = pre[(pre["pop"] == "area") & np.isclose(pre.budget, BUDGET)]
    [D] = row.D_area.astype(float)
    return float(D)


# ------------------------------------------------------------------ the complex

class ComplexModel:
    """One block's complex under a width form (module docstring): the baseline's demand, escape
    times and lenses' references; `score` a clearing by the complex rebuilt."""

    def __init__(self, b, fs: gf.FlowSpec, form: gf.Width):
        self.b, self.form = b, form
        self.geo = gc.geometry(b)
        self.spec = gap_study.spec_of(S_M)
        self.fs = fs
        n = len(self.geo.polys)
        self.n = n
        self.pop = AREA.weights(self.geo.polys)
        self.cost = self.pop / self.pop.sum()
        t0 = time.time()
        m0 = gf.mesh(self.geo, self.spec, self.fs, form, np.ones(n, dtype=bool), carve=None)
        c0 = gf.cells(m0)
        self.dem = gf.point_demand(self.geo, m0, c0)
        self.walled = self.dem.walled
        self.t_mesh = time.time() - t0
        sc = gf.Scores(m0, c0, self.dem)
        self.P0, self.u0 = sc.u_home(self.pop)
        self.inj_q = PQ.injection(self.pop, self.u0, self.walled)
        self.Pq0, _ = sc.u_home(self.inj_q)
        self.J2_0 = float(np.nansum(self.pop * self.u0 ** 2))
        self.cells0, self.nt0 = c0.n, m0.nt
        self.u_q0 = sc.u_home(self.inj_q)[1]
        self.diag0 = dict(merged=c0.merged, ground_obtuse=c0.ground_obtuse,
                          negative=c0.negative, skipped=m0.skipped, nt=m0.nt, cells=c0.n)
        self.t_base = time.time() - t0

    def score(self, standing: np.ndarray, carve) -> tuple[dict, np.ndarray]:
        """(the lenses of a clearing, its escape times per home)."""
        t0 = time.time()
        m = gf.mesh(self.geo, self.spec, self.fs, self.form, standing, carve=carve)
        c = gf.cells(m)
        sc = gf.Scores(m, c, self.dem)
        P, u = sc.u_home(self.pop)
        Pq, _ = sc.u_home(self.inj_q)
        left = self.walled & ~gf.live_homes(self.geo, m, c)
        J2 = float(np.nansum(self.pop * u ** 2))
        return dict(P=P, Pq=Pq, L_P=1.0 - P / self.P0, L_J2=1.0 - (J2 / self.J2_0) ** 0.5,
                    **{"L_Pq0.5": 1.0 - Pq / self.Pq0},
                    enclosed=float(self.cost[left].sum()), unknowns=int((~c.ground).sum()),
                    nt=m.nt, dropped=sc.dropped, t=time.time() - t0), u

    def free_first(self, held: np.ndarray) -> np.ndarray:
        """gap_complex.free_first at the form's w from `held` (no budget): the held set grown (the
        three forms share stage 1's topology)."""
        fr = gc.free_first(self.geo, self.spec, w=self.form.w, grow=0.0, budget=1.0,
                           cost=self.cost, pop=self.pop, held0=held, cx0=None)
        return fr.held


def _local(geo: gc.Geometry, g):
    if g is None:
        return None
    return shapely.transform(g, lambda xy: xy - np.array(geo.origin))


def _homes_path(rows: Path) -> Path:
    out = rows.parent / "homes" / rows.name
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def complex_path(fs: gf.FlowSpec, form: gf.Width, bid: str) -> Path:
    return ROWS / f"complex_{fs.name}_{form.name}" / f"{bid}.parquet"


def complex_one(job: tuple[str, float, int, str]) -> str | None:
    bid, R, levels, token = job
    form = form_of(token)
    fs = flow_spec(R, levels)
    out = complex_path(fs, form, bid)
    if out.exists():
        return None
    try:
        t0 = time.time()
        [b] = common.build_blocks([bid])
        cm = ComplexModel(b, fs, form)
        rows, per_home = [], {"u0": cm.u0, "uq0": cm.u_q0}
        for name, (kind, x, corridor) in fields_of(b, cm.n).items():
            carve = _local(cm.geo, corridor)
            D = float(cm.cost[x].sum()) if kind != "road" else _road_D(b, name)
            r, u = cm.score(~x, carve)
            rows.append(dict(field=name, kind=kind, D=D, added="", **r))
            if name == "simp_default":
                per_home["u_default"] = u
            if kind == "stored" and r["enclosed"] > 0:
                held = cm.free_first(x)
                add = np.flatnonzero(held & ~x)
                r2, _u2 = cm.score(~held, None)
                rows.append(dict(field=name + "+free", kind="stored+free",
                                 D=float(cm.cost[held].sum()),
                                 added=" ".join(map(str, add)), **r2))
            print(f"{bid} complex {fs.name} {form.name} {name}: L_P {r['L_P']:.4f} L_J2 "
                  f"{r['L_J2']:.4f} L_Pq {r['L_Pq0.5']:.4f} enclosed {r['enclosed']:.4f} "
                  f"({r['t']:.0f}s)", flush=True)
        homes = pd.DataFrame(dict(block=bid, home=np.arange(cm.n), w=cm.pop,
                                  walled=cm.walled, inj_q=cm.inj_q, **per_home))
        out.parent.mkdir(parents=True, exist_ok=True)
        homes.to_parquet(_homes_path(out))
        df = pd.DataFrame(rows).assign(
            block=bid, model="complex", spec=cm.spec.name, flow=cm.fs.name, form=form.name,
            w=form.w, P0=cm.P0,
            Pq0=cm.Pq0, J2_0=cm.J2_0, walled0=float(cm.cost[cm.walled].sum()),
            n_walled0=int(cm.walled.sum()), shadowed=int(cm.dem.shadowed.sum()),
            t_mesh=cm.t_mesh, t_base=cm.t_base, t_block=time.time() - t0,
            host=socket.gethostname(), peak_rss_gb=_peak_gb(), **cm.diag0)
        df.to_parquet(out)
        return None
    except Exception as e:     # noqa: BLE001 -- a pool's task reports its block and goes on
        import traceback
        traceback.print_exc()
        return f"{bid}: {e!r}"


# ------------------------------------------------------------------ the raster, isotropic

@dataclass(frozen=True)
class IsoSystem:
    """The raster's isotropic operator on the open field o (RasterIso's): one unknown per free
    cell that reaches the street and is not ground, the steps of lifted's pattern at their face
    apertures (lifted.along_edges)."""
    A: sp.csr_matrix
    free: np.ndarray            # (cells,) the free cells
    on: np.ndarray              # per free cell: it reaches the street
    unk: np.ndarray             # per free cell its unknown (-1 none)
    live: np.ndarray            # per free cell: an unknown

    @classmethod
    def of(cls, g, o: np.ndarray, p) -> IsoSystem:
        free = o.ravel() > 0
        nf = int(free.sum())
        on = lifted.reach(g, o.ravel(), p).ravel()[free]           # per free cell
        ground = g.ground.ravel()[free]
        unk = -np.ones(nf, dtype=np.int64)
        live = on & ~ground
        unk[live] = np.arange(int(live.sum()))
        N = int(live.sum())
        rows, cols, vals = [], [], []
        diag = np.zeros(N)
        for _k, a, bb, w in lifted.along_edges(g, o, p):
            ua, ub = unk[a], unk[bb]
            both = (ua >= 0) & (ub >= 0)
            rows += [ua[both], ub[both]]
            cols += [ub[both], ua[both]]
            vals += [-w[both], -w[both]]
            diag += np.bincount(ua[ua >= 0], weights=w[ua >= 0], minlength=N)
            diag += np.bincount(ub[ub >= 0], weights=w[ub >= 0], minlength=N)
        A = sp.csr_matrix((np.concatenate(vals + [diag]),
                           (np.concatenate(rows + [np.arange(N)]),
                            np.concatenate(cols + [np.arange(N)]))), shape=(N, N))
        return cls(A=A, free=free, on=on, unk=unk, live=live)

    def rhs(self, f: np.ndarray) -> np.ndarray:
        """The right-hand side of an injection field f (the grid's shape)."""
        fc = f.ravel()[self.free]
        if (fc[~self.on] > 0).any():
            raise ValueError("injection on a cell with no path to the street")
        return np.bincount(self.unk[self.live], weights=fc[self.live],
                           minlength=self.A.shape[0])

    def cell_u(self, u: np.ndarray, size: int) -> np.ndarray:
        """The potential per cell of the grid (0 off the system)."""
        out = np.zeros(size)
        idx = np.flatnonzero(self.free)
        out[idx[self.live]] = u[self.unk[self.live]]
        return out


class RasterIso:
    """The isotropic model on the raster (module docstring) on one block: the metric's grid and
    placement (gap_study.raster_view, B's), each field solved on its free cells."""

    def __init__(self, b, mesh_token: str):
        self.c = gap_study.raster_view(b, mesh_token)
        self.sc = self.c.sc
        self.p = self.c.p
        g = self.sc.grid
        self.f = self.sc.pl.field(self.sc.w, g.inside.shape)
        [(P0, u0)] = self.solve(self.sc.free0, [self.f])
        self.P0, self.u0 = P0, u0
        self.inj_q = PQ.injection(self.sc.w, u0, self.sc.enclosed)
        self.f_q = self.sc.pl.field(self.inj_q, g.inside.shape)
        [(self.Pq0, self.u_q0)] = self.solve(self.sc.free0, [self.f_q])
        self.J2_0 = float(np.nansum(self.sc.w * u0 ** 2))

    def solve(self, o: np.ndarray, fs: list[np.ndarray]) -> list[tuple[float, np.ndarray]]:
        """Per injection field f of `fs`: (P, per home its escape time) in the open field o."""
        g = self.sc.grid
        sy = IsoSystem.of(g, o, self.p)
        lu = _amg(sy.A)
        out = []
        for f in fs:
            b = sy.rhs(f)
            u = lu.solve(b)
            uh = self.sc.pl.home_u(sy.cell_u(u, g.inside.size))
            out.append((float(b @ u), np.where(self.sc.enclosed, np.nan, uh)))
        return out

    def score(self, o: np.ndarray) -> tuple[dict, np.ndarray]:
        t0 = time.time()
        [(P, u), (Pq, _uq)] = self.solve(o, [self.f, self.f_q])
        J2 = float(np.nansum(self.sc.w * u ** 2))
        return dict(P=P, Pq=Pq, L_P=1.0 - P / self.P0, L_J2=1.0 - (J2 / self.J2_0) ** 0.5,
                    **{"L_Pq0.5": 1.0 - Pq / self.Pq0},
                    enclosed=self.sc.enclosed_left(o), t=time.time() - t0), u


ISO_RTOL = 1e-10                  # the raster's isotropic solves (P to ~1e-15 relative)


class _AMGSolve:
    """pyamg's smoothed aggregation as CG's preconditioner, set up once for any right-hand side
    (the raster's 8-axis stencil fills SuperLU past memory on the large blocks)."""

    def __init__(self, A):
        import pyamg
        self.A = A
        self.ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric", max_coarse=2000)

    def solve(self, b):
        res: list = []
        x = self.ml.solve(b, tol=ISO_RTOL, accel="cg", maxiter=1000, residuals=res)
        if res[-1] > ISO_RTOL * np.linalg.norm(b) * 10:
            raise RuntimeError(f"AMG-CG stalled at {res[-1] / np.linalg.norm(b):.1e}")
        return x


def _amg(A):
    return _AMGSolve(A.tocsr())


def iso_path(mesh_token: str, bid: str) -> Path:
    return ROWS / f"iso_{mesh_token}" / f"{bid}.parquet"


def iso_one(job: tuple[str, str | None]) -> str | None:
    """The raster's isotropic model on a block, on `mesh_token` (None: the metric's mesh,
    gap_study.mesh_token_of)."""
    bid, token = job
    mt = gap_study.mesh_token_of(bid) if token is None else token
    out = iso_path(mt, bid)
    if out.exists():
        return None
    try:
        t0 = time.time()
        [b] = common.build_blocks([bid])
        ri = RasterIso(b, mt)
        c, sc = ri.c, ri.sc
        g = sc.grid
        n = c.n
        rows, per_home = [], {"u0": ri.u0, "uq0": ri.u_q0}
        for name, (kind, x, corridor) in fields_of(b, n).items():
            if kind == "road":
                o = common.carved(g, g.sub_of(corridor))
                D = _road_D(b, name)
            else:
                o, D = c.open_(x), float(c.cost[x].sum())
            r, u = ri.score(o)
            if name == "simp_default":
                per_home["u_default"] = u
            rows.append(dict(field=name, kind=kind, D=D, added="", **r))
            if kind == "stored" and r["enclosed"] > 0:
                held = clear.free_first(c, budget=1.0, held=x)
                add = np.flatnonzero(held & ~x)
                r2, _u2 = ri.score(c.open_(held))
                rows.append(dict(field=name + "+free", kind="stored+free",
                                 D=float(c.cost[held].sum()), added=" ".join(map(str, add)),
                                 **r2))
            print(f"{bid} iso {name}: L_P {r['L_P']:.4f} L_J2 {r['L_J2']:.4f} L_Pq "
                  f"{r['L_Pq0.5']:.4f} enclosed {r['enclosed']:.4f} ({r['t']:.0f}s)", flush=True)
        df = pd.DataFrame(rows).assign(
            block=bid, model="iso", mesh=mt, P0=ri.P0, Pq0=ri.Pq0, J2_0=ri.J2_0,
            cells=int(g.inside.sum()), t_block=time.time() - t0, host=socket.gethostname(),
            peak_rss_gb=_peak_gb())
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        pd.DataFrame(dict(block=bid, home=np.arange(n), w=sc.w, walled=sc.enclosed,
                          inj_q=ri.inj_q, **per_home)).to_parquet(_homes_path(out))
        return None
    except Exception as e:     # noqa: BLE001
        import traceback
        traceback.print_exc()
        return f"{bid}: {e!r}"


SWEEP_R = 0.5                     # flowsweep: the mesh (graded, SWEEP_LEVELS)
SWEEP_LEVELS = 4
SWEEP_BUDGET = 1.0                # free first frees every walled-in home
TOP = 3                           # the share of a lens its top homes hold


def _top_share(terms: np.ndarray) -> float:
    t = np.sort(np.nan_to_num(terms))[::-1]
    return float(t[:TOP].sum() / t.sum()) if t.sum() > 0 else np.nan


def flowsweep_one(job: tuple[str, str]) -> str | None:
    """One block under one width form (`form_of`): its homes walled in at baseline, free first
    (gap_complex.free_first at the form's w, budget SWEEP_BUDGET), and at baseline and after free
    first P, P_{1/2} and J_2 (B's: over the homes live at baseline) with the share of each its top
    TOP homes hold. Rows gap_rows/compare/flowsweep_<flow>/<form>/<id>.parquet."""
    bid, token = job
    form = form_of(token)
    fs = flow_spec(SWEEP_R, SWEEP_LEVELS)
    out = ROWS / f"flowsweep_{fs.name}" / form.name / f"{bid}.parquet"
    if out.exists():
        return None
    try:
        t0 = time.time()
        [b] = common.build_blocks([bid])
        cm = ComplexModel(b, fs, form)
        fr = gc.free_first(cm.geo, cm.spec, w=form.w, grow=0.0, budget=SWEEP_BUDGET,
                           cost=cm.cost, pop=cm.pop, held0=np.zeros(cm.n, dtype=bool), cx0=None)
        r_free, u_free = cm.score(~fr.held, None)
        live = ~cm.walled
        rows = []
        for state, u, P, Pq in (("baseline", cm.u0, cm.P0, cm.Pq0),
                                ("free", u_free, r_free["P"], r_free["Pq"])):
            uq = cm.u_q0 if state == "baseline" else None
            rows.append(dict(
                state=state, P=P, Pq=Pq, J2=float(np.nansum(cm.pop * u ** 2)),
                top_P=_top_share(np.where(live, cm.pop * u, 0.0)),
                top_J2=_top_share(np.where(live, cm.pop * u ** 2, 0.0)),
                top_Pq=_top_share(np.where(live, cm.inj_q * u, 0.0)) if uq is None
                else _top_share(np.where(live, cm.inj_q * uq, 0.0)),
                u_med=float(np.nanmedian(u)), u_max=float(np.nanmax(u))))
        df = pd.DataFrame(rows).assign(
            block=bid, form=form.name, w=form.w, flow=fs.name,
            walled=int(cm.walled.sum()), walled_pop=float(cm.cost[cm.walled].sum()),
            free_n=int(fr.held.sum()), free_D=float(cm.cost[fr.held].sum()),
            left_pop=float(cm.cost[fr.left].sum()), enclosed_after=r_free["enclosed"],
            shadowed=int(cm.dem.shadowed.sum()), cells0=cm.cells0, t_block=time.time() - t0,
            host=socket.gethostname(), peak_rss_gb=_peak_gb())
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        print(f"{bid} {form.name}: walled {int(cm.walled.sum())}, free D {df.free_D.iloc[0]:.4f}; "
              f"P {cm.P0:.4g} -> {r_free['P']:.4g}; top{TOP} J2 {rows[0]['top_J2']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        return None
    except Exception as e:     # noqa: BLE001
        import traceback
        traceback.print_exc()
        return f"{bid} {token}: {e!r}"


INTERIOR_RANDOM = 3               # interiors: random clearings at BUDGET besides the stored ones
INTERIOR_SEED = 20261013


def interiors(bids: list[str], token: str) -> None:
    """Whether one node per building is materially as good as meshing its interior: per block,
    the stored clearings (where the block has them) and INTERIOR_RANDOM random clearings at
    BUDGET, scored (P's lens, J_2's) by the complex rebuilt, by the clearing model with interiors
    meshed and by the same with every building's interior one node (gap_clear's fixed mesh, the
    0/1 clearing; the study's first run scored it in the eps world at 1e-9). Rows
    gap_rows/compare/interiors/<form>.parquet."""
    import gap_clear as gcl
    form = form_of(token)
    fs = flow_spec(0.5, 4)
    rows = []
    for bid in bids:
        [b] = common.build_blocks([bid])
        cm = ComplexModel(b, fs, form)
        n = cm.n
        picks: dict[str, np.ndarray] = {}
        store = clearings.Store.committed()
        for name, run in STORED.items():
            try:
                x = np.zeros(n, dtype=bool)
                x[store.cleared(run, b, step=1)] = True
                picks[name] = x
            except (KeyError, clearings.StaleClearing):
                pass
        rng = np.random.default_rng(INTERIOR_SEED)
        for k in range(INTERIOR_RANDOM):
            order = rng.permutation(n)
            x = np.zeros(n, dtype=bool)
            x[order[np.cumsum(cm.cost[order]) <= BUDGET]] = True
            picks[f"random{k}"] = x
        models = {}
        for lumped in (False, True):
            t0 = time.time()
            m = gcl.ComplexMetric(spec=cm.spec, fs=fs, form=form, lumped=lumped).clearing(
                b, demand=POP)
            models["lumped" if lumped else "meshed"] = (m, time.time() - t0)
        for name, x in picks.items():
            r, _u = cm.score(~x, None)
            row = dict(block=bid, form=form.name, clearing=name, D=float(cm.cost[x].sum()),
                       L_P_rebuild=r["L_P"], L_J2_rebuild=r["L_J2"])
            for key, (m, tm) in models.items():
                L = m.lenses(x)
                row[f"L_P_{key}"] = 1.0 - L.P / m.base.P
                row[f"L_J2_{key}"] = 1.0 - (L.J2 / m.base.J2) ** 0.5
                row[f"t_{key}"] = tm
                row[f"cells_{key}"] = m.c.n
            rows.append(row)
            print(f"{bid} {name}: L_P rebuild {row['L_P_rebuild']:.4f} meshed "
                  f"{row['L_P_meshed']:.4f} lumped {row['L_P_lumped']:.4f}", flush=True)
    out = ROWS / "interiors" / f"{form.name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)


CONVERGE_FIELDS = ("baseline", "simp_default", "resistance_lp")


def converge(bid: str, Rs: list[float], levels: int, token: str) -> None:
    """P0 and the lenses of CONVERGE_FIELDS against R on one block, under one form; rows
    gap_rows/compare/converge/<id>_<form>.parquet."""
    form = form_of(token)
    [b] = common.build_blocks([bid])
    rows = []
    for R in Rs:
        t0 = time.time()
        fs = flow_spec(R, levels)
        cm = ComplexModel(b, fs, form)
        fields = fields_of(b, cm.n)
        for name in CONVERGE_FIELDS:
            kind, x, corridor = fields[name]
            r, _u = cm.score(~x, _local(cm.geo, corridor))
            rows.append(dict(block=bid, form=form.name, R=R, flow=fs.name, field=name, P0=cm.P0,
                             cells0=cm.cells0, nt0=cm.nt0, t_all=time.time() - t0, **r))
            print(f"{bid} {form.name} {fs.name}: {name} P0 {cm.P0:.2f} L_P {r['L_P']:.4f} L_J2 "
                  f"{r['L_J2']:.4f} L_Pq {r['L_Pq0.5']:.4f} cells {cm.cells0} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    out = ROWS / "converge" / f"{bid}_{form.name}_g{levels}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)


LANE_WIDTHS = (0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0)
LANE_ANGLES = (0.0, 22.5, 45.0)
LANE_OFFSETS = 4                  # lattice offsets per width and angle (seeded)
LANE_SEED = 20261014
LANE_H = 0.5                      # the metric's cell


def lanes() -> None:
    """The raster's width laws on width.walled_channel (W x L_C, walls footprints, demand within 1
    m of the dead end): P of the isotropic model (IsoSystem) and of the lifted metric (uni+apwfs)
    over the exact (L_C - 2 + 1/3) / (SIGMA W) of a channel conducting SIGMA (the complex's: its
    angle and channel guards, within 3e-4). Rows gap_rows/compare/lanes.parquet."""
    from scipy.sparse.linalg import splu
    p = gap_study._params()
    offs = np.random.default_rng(LANE_SEED).random((LANE_OFFSETS, 2))
    rows = []
    for W in LANE_WIDTHS:
        exact = (width.L_C - 2.0 + 1.0 / 3.0) / (SIGMA * W)
        for a in LANE_ANGLES:
            for k, off in enumerate(offs):
                g, f = width.walled_channel(a, W, LANE_H, (float(off[0]), float(off[1])))
                sy = IsoSystem.of(g, g.ff0, p)
                b = sy.rhs(f)
                P_iso = float(b @ splu(sy.A.tocsc()).solve(b))
                P_lift = lifted.solve(g, g.ff0, f, p).P
                rows.append(dict(W=W, angle=a, offset=k, iso=P_iso / exact,
                                 lifted=P_lift / exact))
            r = pd.DataFrame(rows[-LANE_OFFSETS:])
            print(f"W {W:g} angle {a:g}: P / exact iso {r.iso.median():.3f} lifted "
                  f"{r.lifted.median():.3f}", flush=True)
    out = ROWS / "lanes.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)


GATE_RUN = HERE / "clear_rows_S0.01cat_h0.5_area_p2"   # the greedy FFT homogenization re-scored
GATE_STEPS = (1, 2, 5)
GATE_FORMS = ("close0.125", "shy0.5")


def gates(bids: list[str]) -> None:
    """FFT homogenization's gate test re-checked (NOTES, "FFT homogenization": the fast-turning
    scalar model 0.16 below the lifted on 22422's gate, old step rule): the greedy's clearings
    through GATE_STEPS (fft_homog/extract_gated.py's), L_J2 under the lifted metric (uni+apwfs),
    the raster's isotropic model and the complex under GATE_FORMS. Rows
    gap_rows/compare/gates.parquet."""
    out = []
    for bid in bids:
        [b] = common.build_blocks([bid])
        rows = pd.read_parquet(GATE_RUN / f"{bid}.parquet")
        n = len(b.buildings.outlines)
        sc = common.Scorer(b, common.mesh_of("0.5"), gap_study._params(), population=AREA,
                           demand=common.PopulationDemand())
        c = clear.Clearing.on_scorer(sc, rtol=clear.RTOL_TENSION, search=None)
        ri = RasterIso(b, "0.5")
        cms = {t: ComplexModel(b, flow_spec(SWEEP_R, SWEEP_LEVELS), form_of(t))
               for t in GATE_FORMS}
        for step in GATE_STEPS:
            upto = rows[(rows.step >= 1) & (rows.step <= step)]
            ids = np.unique(np.concatenate([np.asarray(x, dtype=np.int64) for x in upto.cleared]))
            x = np.zeros(n, dtype=bool)
            x[ids] = True
            o = c.open_(x)
            r = dict(block=bid, step=step, n=int(x.sum()), D=float(c.cost[x].sum()),
                     old_lifted=float(rows[rows.step == step].perm.iloc[0]),
                     lifted=sc.perm_p(sc.home_u(o), 2.0), lifted_enc=sc.enclosed_left(o))
            r["iso"] = ri.score(o)[0]["L_J2"]
            for t, cm in cms.items():
                cr, _u = cm.score(~x, None)
                r[t], r[f"{t}_enc"] = cr["L_J2"], cr["enclosed"]
            out.append(r)
            print(r, flush=True)
    path = ROWS / "gates.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_parquet(path)


REGION = {   # NOTES, "A region whose only exit is its edge: 5810@major"
    "simp_default": "relax_rows/uni/fw0.q3.i10.t0.001.e0.0001.k1s1e-06.p256w8x2.a5x8/D0.05.p2",
    "cheap": "polish_rows/uni/S0.01cat.P64w8.a5x8/D0.05.p2"}


def region(bid: str, tokens: list[str]) -> None:
    """A region's baseline and its stored clearings (REGION, optimized under the old step rule on
    the lifted metric's a5x8 mesh) on the complex under each form, with time and peak memory; rows
    gap_rows/compare/region/<id>_<form>.parquet."""
    [b] = common.build_blocks([bid])
    store = clearings.Store.committed()
    fs = flow_spec(SWEEP_R, SWEEP_LEVELS)
    for token in tokens:
        t0 = time.time()
        cm = ComplexModel(b, fs, form_of(token))
        rows = [dict(field="baseline", D=0.0, P=cm.P0, Pq=cm.Pq0, L_P=0.0, L_J2=0.0,
                     **{"L_Pq0.5": 0.0}, enclosed=0.0, t=cm.t_base, peak_rss_gb=_peak_gb())]
        print(f"{bid} {cm.form.name}: baseline P0 {cm.P0:.6g} walled {int(cm.walled.sum())} "
              f"cells {cm.cells0} ({cm.t_base:.0f}s, {_peak_gb():.1f} GB)", flush=True)
        for name, run in REGION.items():
            x = np.zeros(cm.n, dtype=bool)
            x[store.cleared(run, b, step=1)] = True
            r, _u = cm.score(~x, None)
            rows.append(dict(field=name, D=float(cm.cost[x].sum()), **{k: r[k] for k in (
                "P", "Pq", "L_P", "L_J2", "L_Pq0.5", "enclosed", "t")}, peak_rss_gb=_peak_gb()))
            print(f"{bid} {cm.form.name} {name}: L_P {r['L_P']:.4f} L_J2 {r['L_J2']:.4f} L_Pq "
                  f"{r['L_Pq0.5']:.4f} enclosed {r['enclosed']:.4f} ({r['t']:.0f}s)", flush=True)
        out = ROWS / "region" / f"{bid}_{cm.form.name}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).assign(block=bid, form=cm.form.name, flow=fs.name, nt0=cm.nt0,
                                  cells0=cm.cells0, n_walled0=int(cm.walled.sum()),
                                  t_mesh=cm.t_mesh, t_block=time.time() - t0,
                                  host=socket.gethostname()).to_parquet(out)


def _pool(fn, jobs: list, procs: int) -> list[str]:
    if procs == 1:
        return [f for f in map(fn, jobs) if f]
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(procs, maxtasksperchild=1) as pool:
        return [f for f in pool.imap_unordered(fn, jobs) if f]


def main(argv: list[str]) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMBA_NUM_THREADS"):
        if os.environ.get(v) != "1":
            raise SystemExit(f"{v}=1 is required (single-threaded workers)")
    cmd = argv[0]
    if cmd == "complex":
        R, levels, tokens = float(argv[3]), int(argv[4]), argv[5].split(",")
        failed = _pool(complex_one, [(i, R, levels, t) for t in tokens
                                     for i in gap_study.ids_of(argv[1])], int(argv[2]))
    elif cmd == "interiors":
        interiors(gap_study.ids_of(argv[1]), argv[2])
        failed = []
    elif cmd == "flowsweep":
        failed = _pool(flowsweep_one, [(i, t) for t in argv[3].split(",")
                                       for i in gap_study.ids_of(argv[1])], int(argv[2]))
    elif cmd == "gates":
        gates(gap_study.ids_of(argv[1]))
        failed = []
    elif cmd == "lanes":
        lanes()
        failed = []
    elif cmd == "region":
        region(argv[1], argv[2].split(","))
        failed = []
    elif cmd == "converge":
        converge(argv[1], [float(x) for x in argv[2].split(",")], int(argv[3]), argv[4])
        failed = []
    elif cmd == "iso":
        token = argv[3] if len(argv) > 3 else None
        failed = _pool(iso_one, [(i, token) for i in gap_study.ids_of(argv[1])], int(argv[2]))
    else:
        raise SystemExit(__doc__)
    if failed:
        print("FAILED:\n" + "\n".join(failed), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
