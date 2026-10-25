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
relax.simp's OC, B = -dP / (price dA + nu dD) per pixel, nu bisected on D at the trial update
(the projection at a large beta makes D linearised at the iterate overshoot: rev 1 ended 20954's
D 0.02 runs at 0.11).

Two answers, each scored as a road network is (Carving.of with one network of one stop: P on its
own mesh by road_state; D by common.MEASURE on the world-frame ground):
  paved: the triangles at r_t >= 1/2, unioned (closed and opened by SMOOTH_M and simplified to it,
    so the carved mesh meets no sliver: rev 1's raw union failed to mesh on 5810 three times);
  roads: the skeleton of the pixels at r >= 1/2 (reblock's Zhang-Suen thinning), its 8-neighbour
    steps merged into lines, each a road ROAD_W wide (road_corridor's rule): the trees' lines at a
    road's width, where `paved` keeps their trunks' widths. Thinning eats a strip from its ends, so
    the skeleton stops about a half-width short of the street where the paved strip meets it (rev
    3: 7 -- 61% of the length on 5810 in pieces more than 4 m from it): each end whose inscribed
    disk reaches past the block (the raster's pixels off the design) is joined to the nearest such
    pixel. Pieces still apart from the street are kept, their length reported (roads_floating_m).

    PYTHONPATH=. uv run python research/roadless/road_simp.py run <block> <plan> cpu|gpu [--replace]

plan: rs.D<D>.k<per_km>.r<radius>.q<q,...>.b<beta,...>.i<iters> (the stages pair q and beta in
order), e.g. rs.D0.05.k0.01.r3.5.q1,2,3,3,3.b1,2,4,8,16.i30; its name (Plan.name) adds the raster's
pixel and the eps world's eps (.p<PIXEL>e<EPS>). Rows in road_simp_rows/<metric>/<name>/
<block>.parquet beside its design (<block>_x.parquet: x per pixel, parquet so the cluster's pull
brings it), its paved ground (<block>_ground.parquet) and its roads (<block>_roads.parquet).
"""
from __future__ import annotations

import re
import socket
import sys
import time
from collections.abc import Callable
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

from reblock.budget import road_corridor  # noqa: E402
from reblock.derive.access import STREET_TOL  # noqa: E402
from reblock.destroyed import IncrementalDestroyed  # noqa: E402
from reblock.methods.betweenness.thinning import skeletonize  # noqa: E402

PIXEL = 1.0          # m: the design raster's pixel
ROAD_W = 7.0         # m: a road's width (the lineup's roads', road_corridor's buffer)
EPS = 1e-4           # the eps world's standing footprint (relax's default plan, `e0.0001`)
XMIN = relax.XMIN    # the design's floor
MOVE = 0.2           # OC's move limit (relax.simp's)
NU_TOL = 1e-3        # OC: nu bisected to this factor (cannot change the answer beyond it)
HALF = 0.5           # the answer: the triangles whose road density reaches it
SMOOTH_M = 0.1       # m: the paved answer's closing, opening and simplification
REV = 4              # the code's revision, in the plan's name (2: D bisected at the trial update,
                     # the answers smoothed and doubled; 3: the roads' mask unholed; 4: the
                     # skeleton's ends joined to the street)
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
                f".p{PIXEL:g}e{EPS:g}v{REV}")

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
    the kernel's sum over the design pixels (`norm`); `lo` the corner of pixel (0, 0)."""
    shape: tuple[int, int]
    lo: NDArray[np.float64]
    pix: NDArray[np.int64]
    on: NDArray[np.bool_]
    kernel: NDArray[np.float64]
    norm: NDArray[np.float64]

    @staticmethod
    def of(cen: NDArray[np.float64], *, radius: float) -> Raster:
        lo = cen.min(axis=0)
        ij = np.floor((cen - lo) / PIXEL).astype(np.int64)
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
        return Raster(shape=shape, lo=lo, pix=pix, on=on, kernel=kernel, norm=norm)

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
        paving: NDArray[np.float64], gD: NDArray[np.float64], *,
        D_of: Callable[[NDArray[np.float64]], float], budget: float,
        nu0: float) -> tuple[NDArray[np.float64], float]:
    """One OC update of the design pixels `on`: x (gain / (paving + nu gD))^1/2 within MOVE of x
    and [XMIN, 1]; nu 0 if the update's D (D_of) is within the budget, else bisected (from nu0,
    the last update's, bracketed by factors of 10) to the least nu whose update's D is, to a
    factor of 1 + NU_TOL (D falls as nu rises: every pixel's update does)."""
    gain = np.maximum(gain, 0.0)
    lo_x, hi_x = np.maximum(XMIN, x - MOVE), np.minimum(1.0, x + MOVE)

    def trial(nu: float) -> NDArray[np.float64]:
        B = gain / np.maximum(paving + nu * gD, 1e-300)
        return np.where(on, np.clip(x * np.sqrt(B), lo_x, hi_x), x)

    xn = trial(0.0)
    if D_of(xn) <= budget:
        return xn, 0.0
    nu = nu0 if nu0 > 0 else 1.0
    if D_of(trial(nu)) > budget:
        lo = nu
        while nu < 1e300 and D_of(trial(nu := nu * 10.0)) > budget:
            lo = nu                      # (at 1e300 the move limit holds D over: the best kept)
        hi = nu
    else:
        hi = nu
        while nu > 1e-300 and D_of(trial(nu := nu / 10.0)) <= budget:
            hi = nu
        lo = nu
    while hi / max(lo, 1e-300) > 1.0 + NU_TOL:
        nu = float(np.sqrt(max(lo, 1e-300) * hi))
        lo, hi = (nu, hi) if D_of(trial(nu)) > budget else (lo, nu)
    return trial(hi), hi


def simp(rd: Roads, plan: Plan, *,
         log: Any = print) -> tuple[NDArray[np.float64], list[dict[str, float]]]:
    """OC from XMIN everywhere (no road)."""
    x = np.full(len(rd.raster.on), XMIN)
    lam = rd.price(plan.per_km)
    on = rd.raster.on
    rows = []
    nu = 0.0
    t0 = time.time()
    for q, beta in plan.stages:
        def D_of(xn: NDArray[np.float64], beta: float = beta) -> float:
            return rd.D(rd.chain(xn, beta)[1])

        for it in range(plan.iters):
            xf, r = rd.chain(x, beta)
            s = r ** q
            P, gs = rd.P_grad(s)
            g_r = gs * q * np.maximum(r, 1e-300) ** (q - 1.0)
            gP = rd.back(xf, beta, g_r)
            gA = rd.back(xf, beta, rd.area)
            gD = rd.back(xf, beta, rd.dD(r))
            A, D = float(rd.area @ r), rd.D(r)
            xn, nu = _oc(x, on, -gP, lam * gA, gD, D_of=D_of, budget=plan.D, nu0=nu)
            change = float(np.abs(xn - x)[on].max())
            x = xn
            rows.append(dict(q=q, beta=beta, it=it, P=P, L_P=1.0 - P / rd.P0, A=A, D=D, nu=nu,
                             change=change, t=time.time() - t0))
            log(f"q {q:g} beta {beta:g} it {it}: L_P {1 - P / rd.P0:.4f} + price "
                f"{lam * A / rd.P0:.4f}, D {D:.4f}, paved {A:.0f} m^2, change {change:.3f} "
                f"({time.time() - t0:.0f}s)")
    return x, rows


def paved(rd: Roads, x: NDArray[np.float64], beta: float) -> Any:
    """The paved answer (module docstring; the model's local frame)."""
    _, r = rd.chain(x, beta)
    fm = rd.m.fx.fm
    t = np.flatnonzero(r >= HALF)
    g = shapely.coverage_union_all(shapely.polygons(fm.xy[fm.tri[t]]))
    g = shapely.buffer(shapely.buffer(g, SMOOTH_M, quad_segs=1), -SMOOTH_M, quad_segs=1)
    return shapely.make_valid(shapely.simplify(g, SMOOTH_M))


def roads(rd: Roads, x: NDArray[np.float64], beta: float) -> Any:
    """The roads answer's lines (module docstring; the model's local frame)."""
    ra = rd.raster
    # the filter read wherever its kernel reaches a design pixel: where a triangle is larger than
    # a pixel, the pixels between centroids hold no design and Raster.filter's 0 there holed the
    # mask (the skeleton wrapped every hole: a lattice)
    y = ndimage.correlate(np.where(ra.on, x, 0.0).reshape(ra.shape), ra.kernel, mode="constant")
    norm = ra.norm.reshape(ra.shape)
    on = (norm > 0) & (relax.project(y / np.where(norm > 0, norm, 1.0), beta) >= HALF)
    skel = skeletonize(on)
    i, j = np.nonzero(skel)
    keep = set(zip(i.tolist(), j.tolist(), strict=True))
    segs = []
    for a, b in keep:
        for da, db in ((0, 1), (1, -1), (1, 0), (1, 1)):
            if (a + da, b + db) in keep:
                segs.append(((a + 0.5, b + 0.5), (a + da + 0.5, b + db + 0.5)))
    # the ends joined to the street (module docstring): an end's inscribed disk (its distance off
    # the mask) reaching the nearest pixel off the design, padded so the raster's edge is off it
    block = np.pad(ra.on.reshape(ra.shape), 1)
    off, (oi, oj) = ndimage.distance_transform_edt(block, return_indices=True)
    inside = ndimage.distance_transform_edt(np.pad(on, 1))
    deg = ndimage.correlate(np.pad(skel, 1).astype(np.int64), np.ones((3, 3), dtype=np.int64),
                            mode="constant")
    ends = np.pad(skel, 1) & (deg == 2) & (off <= inside + np.sqrt(2.0))
    for a, b in zip(*np.nonzero(ends), strict=True):
        segs.append(((a - 0.5, b - 0.5), (oi[a, b] - 0.5, oj[a, b] - 0.5)))
    xy = ra.lo + PIXEL * np.asarray(segs, dtype=np.float64).reshape(-1, 2, 2)
    return shapely.line_merge(shapely.multilinestrings(shapely.linestrings(xy)))


def floating_m(net: gpd.GeoDataFrame, streets: gpd.GeoDataFrame) -> float:
    """The length of a network's pieces (its lines within a pixel of each other) that come no
    nearer the street than budget.STREET_TOL."""
    lines = net.geometry.to_numpy()
    street = shapely.union_all(streets.geometry.to_numpy())
    pieces = shapely.get_parts(shapely.union_all(shapely.buffer(lines, PIXEL)))
    apart = [p for p in pieces if shapely.distance(p, street) > STREET_TOL + PIXEL]
    return float(shapely.length(shapely.intersection(shapely.union_all(lines),
                                                     shapely.union_all(apart)))) if apart else 0.0


def score(b: Any, metric: gcl.ComplexMetric, ground_world: Any) -> dict[str, float]:
    """A paved ground scored as a road network (module docstring)."""
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
    ground = m.geo.utm(paved(rd, x, beta))
    lines = m.geo.utm(roads(rd, x, beta))
    del rd, m                                  # the eps world's model, before two more are built
    net = gpd.GeoDataFrame(geometry=list(shapely.get_parts(lines)), crs=b.crs).assign(
        width_m=ROAD_W)
    sc = {f"paved_{k}": v for k, v in score(b, metric, ground).items()}
    sc |= {f"roads_{k}": v for k, v in score(b, metric, road_corridor(net)).items()}
    sc["roads_m"] = float(net.length.sum())
    sc["roads_floating_m"] = floating_m(net, b.streets)
    print(f"{bid} {plan.name}: paved L_P {sc['paved_L_P']:.4f}, D {sc['paved_D_exact']:.4f}, "
          f"{sc['paved_paved']:.0f} m^2; roads L_P {sc['roads_L_P']:.4f}, D "
          f"{sc['roads_D_exact']:.4f}, {sc['roads_m']:.0f} m ({time.time() - t0:.0f}s)",
          flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": x}).to_parquet(out.with_name(f"{bid}_x.parquet"))
    gpd.GeoDataFrame(geometry=[ground], crs=b.crs).to_parquet(
        out.with_name(f"{bid}_ground.parquet"))
    net.to_parquet(out.with_name(f"{bid}_roads.parquet"))
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
