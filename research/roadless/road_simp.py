"""A road-placing SIMP (BACKLOG, "Try everything"): roads as a density on the block's fixed mesh,
driven by the clearing model's eps world and its adjoint.

The design is a density x on a raster of PIXEL m over the mesh, each triangle reading the pixel
under its centroid. A cone filter of a road's half-width (Plan.radius: road_corridor's 7 m roads)
and the Heaviside projection (relax.project, beta continued) make it a road density r_t in [0, 1]
per triangle, SIMP's s_t = r_t^q (q continued). A triangle's factors move toward a road's: its
openness from the eps world's kt0 (a standing footprint EPS, the width form's terms on open ground)
to 1, kt0 + s (1 - kt0); its boost from the ways' b to max(b, road) (gap_exits.Boost.road, as
Model.mesh_of boosts a corridor), b + s (max(b, road) - b). Every cell is live in the eps world, so
every state is one pattern's.

Objective: P under the population's demand (the lens L_P = 1 - P / P0), plus a price on paving
(Plan.per_km: the L_P a kilometre of road, ROAD_W wide, must buy back; paving open ground costs no
displacement, so without a price the optimum paves it all). Constraint: the displacement D <=
Plan.D, per building its covered share X_b (its triangles' area times r_t over its own) through
common.MEASURE's ramp, weighted by population (the real measure reads the largest standing piece,
which a cut through a long building's middle makes larger than the covered share). The update is
relax.simp's OC, B = -dP / (price dA + nu dD) per pixel, nu bisected on D linearised at the iterate.

The answer: the triangles at r_t >= 1/2, their union the road's paved ground, scored as a road
network is (Carving.of with one network of one stop: P on its own mesh by road_state; D by
common.MEASURE on the world-frame ground).

    PYTHONPATH=. uv run python research/roadless/road_simp.py run <block> <plan> cpu|gpu [--replace]

plan: rs.D<D>.k<per_km>.r<radius>.q<q,...>.b<beta,...>.i<iters> (the stages pair q and beta in
order), e.g. rs.D0.05.k0.01.r3.5.q1,2,3,3,3.b1,2,4,8,16.i30; its name (Plan.name) adds the raster's
pixel and the eps world's eps (.p<PIXEL>e<EPS>). Rows in road_simp_rows/<metric>/<name>/
<block>.parquet beside its field (.npz) and its ground (<block>_ground.parquet).
"""
from __future__ import annotations

import re
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, TypeAlias

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely
from numba import njit
from numpy.typing import NDArray
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import common  # noqa: E402
import gap_calib  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_exits as gx  # noqa: E402
import metrics  # noqa: E402
import relax  # noqa: E402

from reblock.destroyed import IncrementalDestroyed  # noqa: E402

PIXEL = 1.0          # m: the design raster's pixel
ROAD_W = 7.0         # m: a road's width (the lineup's roads', road_corridor's buffer)
EPS = 1e-4           # the eps world's standing footprint (relax's default plan, `e0.0001`)
XMIN = relax.XMIN    # the design's floor
MOVE = 0.2           # OC's move limit (relax.simp's)
HALF = 0.5           # the answer: the triangles whose road density reaches it
OUT = HERE / "road_simp_rows"
MESHES = HERE / "road_simp_rows" / ".meshes"     # the answers' carved meshes


@dataclass(frozen=True, kw_only=True)
class Plan:
    """A run's settings, all in its name (module docstring)."""
    D: float                                   # the displacement budget
    per_km: float                              # the L_P a km of road must buy back
    radius: float                              # m: the cone filter's
    stages: tuple[tuple[float, float], ...]    # (q, beta) in order
    iters: int                                 # OC updates per stage

    def __post_init__(self) -> None:
        if not (self.D > 0 and self.per_km > 0 and self.radius > 0 and self.stages
                and self.iters > 0):
            raise ValueError(f"{self}: a budget, a price, a radius, stages and updates, each "
                             "positive")

    @property
    def name(self) -> str:
        q = ",".join(f"{q:g}" for q, _ in self.stages)
        b = ",".join(f"{b:g}" for _, b in self.stages)
        return (f"rs.D{self.D:g}.k{self.per_km:g}.r{self.radius:g}.q{q}.b{b}.i{self.iters}"
                f".p{PIXEL:g}e{EPS:g}")

    @staticmethod
    def of(spec: str) -> Plan:
        m = re.fullmatch(r"rs\.D([0-9.e-]+)\.k([0-9.e-]+)\.r([0-9.]+)\.q([0-9.,]+)\.b([0-9.,]+)"
                         r"\.i([0-9]+)", spec)
        if m is None:
            raise SystemExit(f"{spec}: a plan is rs.D<D>.k<per_km>.r<radius>.q<q,..>.b<beta,..>"
                             ".i<iters>")
        qs = [float(v) for v in m[4].split(",")]
        bs = [float(v) for v in m[5].split(",")]
        if len(qs) != len(bs):
            raise SystemExit(f"{spec}: as many q as beta")
        return Plan(D=float(m[1]), per_km=float(m[2]), radius=float(m[3]),
                    stages=tuple(zip(qs, bs, strict=True)), iters=int(m[6]))


class Raster(NamedTuple):
    """The design's raster over the mesh: per triangle its pixel (`pix`, flat: the one under its
    centroid), the design pixels (`on`: any triangle's), the cone filter's kernel and per pixel
    the kernel's sum over the design pixels (`norm`)."""
    shape: tuple[int, int]
    pix: NDArray[np.int64]
    on: NDArray[np.bool_]
    kernel: NDArray[np.float64]
    norm: NDArray[np.float64]

    @staticmethod
    def of(cen: NDArray[np.float64], *, radius: float) -> Raster:
        ij = np.floor((cen - cen.min(axis=0)) / PIXEL).astype(np.int64)
        shape = (int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1)
        pix = ij[:, 0] * shape[1] + ij[:, 1]
        on = np.zeros(shape[0] * shape[1], dtype=bool)
        on[pix] = True
        r = radius / PIXEL
        k = int(np.ceil(r))
        d = np.hypot(*np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1), indexing="ij"))
        kernel = np.maximum(0.0, r - d)
        norm = ndimage.correlate(on.reshape(shape).astype(np.float64), kernel,
                                 mode="constant").ravel()
        return Raster(shape=shape, pix=pix, on=on, kernel=kernel, norm=norm)

    def filter(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """The cone filter over the design pixels (a pixel off them reads 0)."""
        y = ndimage.correlate(np.where(self.on, x, 0.0).reshape(self.shape), self.kernel,
                              mode="constant").ravel()
        return np.where(self.on, y / np.where(self.on, self.norm, 1.0), 0.0)

    def filter_T(self, g: NDArray[np.float64]) -> NDArray[np.float64]:
        """The filter's transpose (its kernel symmetric)."""
        y = ndimage.correlate(np.where(self.on, g / np.where(self.on, self.norm, 1.0), 0.0)
                              .reshape(self.shape), self.kernel, mode="constant").ravel()
        return np.where(self.on, y, 0.0)


I64: TypeAlias = NDArray[np.int64]
F64: TypeAlias = NDArray[np.float64]


@njit(cache=True)
def _grad(pe: I64, ea: I64, eb: I64, pa: I64, pb: I64, T: F64, kt: F64, kp: F64, b: F64, pf: I64,
          fc: I64, fa: I64, Tf: F64, u: F64, nt: int) -> tuple[F64, F64]:
    """(per triangle dP / d(its openness kt), dP / d(its boost b)) by the adjoint (P self-adjoint:
    the adjoint is u): dP / dw = -du^2 per conductance w, a passage's T kp (T the boosted mesh's:
    the unboosted one's times the harmonic mean of its triangles' boosts; kp the harmonic mean of
    their openness), a street face's T_f kt (T_f times its triangle's boost)."""
    gk = np.zeros(nt)
    gb = np.zeros(nt)
    for q in range(len(pe)):
        p = pe[q]
        du = u[ea[q]] - u[eb[q]]
        g = -T[p] * du * du
        i, j = pa[p], pb[p]
        va, vb = kt[i], kt[j]
        s = va + vb
        if s > 0:
            s2 = s * s
            gk[i] += g * 2.0 * vb * vb / s2
            gk[j] += g * 2.0 * va * va / s2
        ba, bb = b[i], b[j]
        if ba > 0 and bb > 0:            # a boost of 0 holds T at 0 (g 0): no derivative read
            t = ba + bb
            gh = g * kp[p] * t / (2.0 * ba * bb)
            t2 = t * t
            gb[i] += gh * 2.0 * bb * bb / t2
            gb[j] += gh * 2.0 * ba * ba / t2
    for q in range(len(pf)):
        f = pf[q]
        g = -Tf[f] * u[fc[q]] * u[fc[q]]
        gk[fa[f]] += g
        if b[fa[f]] > 0:
            gb[fa[f]] += g * kt[fa[f]] / b[fa[f]]
    return gk, gb


class Roads:
    """The block's road density problem on the model's fixed mesh (module docstring)."""

    def __init__(self, m: gcl.Model, *, radius: float) -> None:
        if len(m.fill.e):
            raise ValueError("a width form that fills gaps: a road's density would not open them")
        self.m = m
        fm = m.fx.fm
        self.raster = Raster.of(m.cen_, radius=radius)
        a, b, c = (fm.xy[fm.tri[:, k]] for k in range(3))
        self.area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                                 - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        self.ka = np.full(m.carving.n, EPS)
        self.kt0 = np.asarray(m.terms.value(self.ka))
        self.bw = np.asarray(m.b_ways, dtype=np.float64)
        self.db = np.maximum(self.bw, m.exits.boost.road) - self.bw
        # per building its triangles' share of its area (rows: buildings)
        cov = fm.tcov.T.tocsr().astype(np.float64) @ sp.diags(self.area)
        self.cover = sp.diags(1.0 / np.maximum(np.asarray(cov.sum(axis=1)).ravel(), 1e-300)) @ cov
        self.w = m.pop / m.pop.sum()
        self.P0 = float(m.base.P)
        if (self.bw <= 0).any() or (self.kt0 <= 0).any():
            print(f"boost 0 on {int((self.bw <= 0).sum())} triangles, openness 0 on "
                  f"{int((self.kt0 <= 0).sum())}", flush=True)

    # -- the design's chain: x per pixel -> r per triangle
    def chain(self, x: NDArray[np.float64], beta: float) -> tuple[NDArray[np.float64], ...]:
        """(the filtered design per pixel, the road density per triangle)."""
        xf = self.raster.filter(x)
        return xf, relax.project(xf, beta)[self.raster.pix]

    def back(self, xf: NDArray[np.float64], beta: float,
             g_r: NDArray[np.float64]) -> NDArray[np.float64]:
        """A per-triangle derivative in r back to the design per pixel."""
        g = np.bincount(self.raster.pix, weights=g_r, minlength=len(xf))
        return self.raster.filter_T(g * relax.dproject(xf, beta))

    # -- the measures
    def D(self, r: NDArray[np.float64]) -> float:
        return float(self.w @ common.MEASURE.ramp(self.cover @ r))

    def dD(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """dD / dr per triangle (the ramp's slope where a building is under it, else 0)."""
        X = self.cover @ r
        return np.asarray(self.cover.T @ (self.w * (X < common.MEASURE.tau)
                                          / common.MEASURE.tau))

    def state(self, s: NDArray[np.float64]) -> gcl.State:
        """The eps world's state with SIMP's share s per triangle of a road."""
        m = self.m
        kt = self.kt0 + s * (1.0 - self.kt0)
        fm = m.exits.edge.mesh(gx.boosted(m.fx.fm, self.bw + s * self.db))
        kp, binds = gcl._passage_factors(fm, kt, self.ka, m.fill)
        r = m.sys.graph.reach(fm, kt, kp)
        f = m.sys.planned.factor(m.sys.graph.values(r), r.unknown)
        return gcl.State(sys=m.sys, prefix=gcl._NO_ROADS, fm=fm, k=self.ka, kt=kt, kp=kp,
                         binds=binds, reach=r, A=None, factor=f)

    def P_grad(self, s: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """P and dP / ds per triangle."""
        m = self.m
        st = self.state(s)
        u = m.solve(st, m.pop)
        P = float(m._rhs(st, m.pop) @ u)
        c, fm = st.sys.cells, st.fm
        gk, gb = _grad(c.pe, c.ea, c.eb, fm.pa, fm.pb, fm.T, st.kt, st.kp,
                       self.bw + s * self.db, c.pf, c.fc, fm.fa, fm.Tf, u, fm.nt)
        return P, gk * (1.0 - self.kt0) + gb * self.db

    def price(self, per_km: float) -> float:
        """P per square metre paved: per_km of L_P per km of road ROAD_W wide."""
        return per_km * self.P0 / (1000.0 * ROAD_W)


def _oc(x: NDArray[np.float64], on: NDArray[np.bool_], gain: NDArray[np.float64],
        paving: NDArray[np.float64], gD: NDArray[np.float64], *, D: float,
        budget: float) -> tuple[NDArray[np.float64], float]:
    """One OC update of the design pixels `on`: x (gain / (paving + nu gD))^1/2 within MOVE of x
    and [XMIN, 1], nu 0 if D linearised at x stays within the budget, else bisected to it."""
    gain = np.maximum(gain, 0.0)
    lo_x, hi_x = np.maximum(XMIN, x - MOVE), np.minimum(1.0, x + MOVE)

    def trial(nu: float) -> NDArray[np.float64]:
        B = gain / np.maximum(paving + nu * gD, 1e-300)
        return np.where(on, np.clip(x * np.sqrt(B), lo_x, hi_x), x)

    nu = 0.0
    xn = trial(nu)
    if D + float(gD @ (xn - x)) > budget:
        lo, hi = 1e-300, 1e300
        for _ in range(200):
            nu = float(np.sqrt(lo * hi))
            xn = trial(nu)
            lo, hi = (nu, hi) if D + float(gD @ (xn - x)) > budget else (lo, nu)
            if hi / lo < 1 + 1e-9:
                break
    return xn, nu


def simp(rd: Roads, plan: Plan, *,
         log: Any = print) -> tuple[NDArray[np.float64], list[dict[str, float]]]:
    """OC from XMIN everywhere (no road)."""
    x = np.full(len(rd.raster.on), XMIN)
    lam = rd.price(plan.per_km)
    on = rd.raster.on
    rows = []
    t0 = time.time()
    for q, beta in plan.stages:
        for it in range(plan.iters):
            xf, r = rd.chain(x, beta)
            s = r ** q
            P, gs = rd.P_grad(s)
            g_r = gs * q * np.maximum(r, 1e-300) ** (q - 1.0)
            gP = rd.back(xf, beta, g_r)
            gA = rd.back(xf, beta, rd.area)
            gD = rd.back(xf, beta, rd.dD(r))
            A, D = float(rd.area @ r), rd.D(r)
            xn, nu = _oc(x, on, -gP, lam * gA, gD, D=D, budget=plan.D)
            change = float(np.abs(xn - x)[on].max())
            x = xn
            rows.append(dict(q=q, beta=beta, it=it, P=P, L_P=1.0 - P / rd.P0, A=A, D=D, nu=nu,
                             change=change, t=time.time() - t0))
            log(f"q {q:g} beta {beta:g} it {it}: L_P {1 - P / rd.P0:.4f} + price "
                f"{lam * A / rd.P0:.4f}, D {D:.4f}, paved {A:.0f} m^2, change {change:.3f} "
                f"({time.time() - t0:.0f}s)")
    return x, rows


def answer(rd: Roads, x: NDArray[np.float64], beta: float) -> Any:
    """The paved ground (the triangles at r >= HALF, unioned; the model's local frame)."""
    _, r = rd.chain(x, beta)
    fm = rd.m.fx.fm
    t = np.flatnonzero(r >= HALF)
    return shapely.coverage_union_all(shapely.polygons(fm.xy[fm.tri[t]]))


def score(b: Any, metric: gcl.ComplexMetric, ground_world: Any) -> dict[str, float]:
    """The paved ground scored as a road network (module docstring)."""
    geo = gc.geometry(b)
    carving = gcl.Carving.of(geo, [[shapely.make_valid(geo.local(ground_world))]])
    MESHES.mkdir(parents=True, exist_ok=True)
    fx = metric.fixed(b, cache=MESHES, carving=carving)
    m = metric.model(b, fx, demand=common.PopulationDemand())
    lens = m.road_lenses(gcl.Prefix(net=0, stop=1))
    W = common.AreaPopulation().weights(np.asarray(b.buildings.outlines))
    inc = IncrementalDestroyed(common.MEASURE, b.buildings)
    inc.add(ground_world)
    return dict(P=lens.P, L_P=1.0 - lens.P / m.base.P, L_Pq=1.0 - lens.Pq / m.base.Pq,
                L_J2=1.0 - lens.J2 / m.base.J2, D_exact=float((inc.c() * W).sum() / W.sum()),
                paved=float(shapely.area(ground_world)), enclosed=lens.enclosed)


def run(bid: str, plan: Plan, device: str, *, standing: clearings.Standing) -> None:
    metric = metrics.metric_of(metrics.RUNTIME, device=device)
    out = OUT / metrics.RUNTIME / plan.name / f"{bid}.parquet"
    if not standing.todo(out):
        return
    t0 = time.time()
    [b] = common.build_blocks([bid])
    m = metric.clearing(b, demand=common.PopulationDemand())
    rd = Roads(m, radius=plan.radius)
    print(f"{bid}: {m.fx.fm.nt} triangles, {int(rd.raster.on.sum())} design pixels, P0 "
          f"{rd.P0:.6e} ({time.time() - t0:.0f}s)", flush=True)
    x, rows = simp(rd, plan, log=lambda s: print(s, flush=True))
    beta = plan.stages[-1][1]
    ground = m.geo.utm(answer(rd, x, beta))
    sc = score(b, metric, ground)
    print(f"{bid} {plan.name}: L_P {sc['L_P']:.4f}, D {sc['D_exact']:.4f}, paved "
          f"{sc['paved']:.0f} m^2 ({time.time() - t0:.0f}s)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), x=x)
    gpd.GeoDataFrame(geometry=[ground], crs=b.crs).to_parquet(
        out.with_name(f"{bid}_ground.parquet"))
    df = pd.DataFrame(rows).assign(block=bid, plan=plan.name, **sc, device=device,
                                   host=socket.gethostname(), peak_rss_gb=gap_calib.peak_gb(),
                                   t_total=time.time() - t0)
    standing.write(df, out)


def main(argv: list[str]) -> None:
    argv, standing = clearings.standing_of(argv)
    if len(argv) == 4 and argv[0] == "run" and argv[3] in ("cpu", "gpu"):
        run(argv[1], Plan.of(argv[2]), argv[3], standing=standing)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
