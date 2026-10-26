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

The update is a Strategy (Update, the plan's `.u` token). OC: x B^(1/2), the paving cost linear in
the density (price A), q continued to 3 (SIMP's penalty). Adaptive(gamma) (docs/superpowers/plans/
2026-10-25-roads-from-the-vote.md: the adaptive transport network of Hu and Cai, Bohn and
Magnasco): the cost price sum_t a_t r_t^gamma (a fully paved triangle costs price a_t at every
gamma, so per_km keeps its meaning), q 1 throughout (a road's conductance is linear in its paved
width; gamma alone sets the regime, as SIMP's q would through s = r^q, so the two are not stacked),
and B's exponent 1 / (1 + gamma): B = -dP/dr / (price gamma a r^(gamma-1) + nu dD/dr). Where the
road dominates a triangle's conductance, -dP/dr = |j|^2 a / (dsigma r^2) = C / r^2 (C fixed by the
current), so with nu 0 one step lands on x B^(1/(1+gamma)) = (C / (price gamma))^(1/(1+gamma)): the
density proportional to |j|^(2/(1+gamma)), Hu and Cai's fixed point (trees for gamma < 1, loops
for gamma > 1), the current recomputed at every step. At gamma 1 it is OC at q 1. The design keeps
OC's move limit (MOVE: the damping) and its bisection of nu on D.

The starting world is a Strategy too (Clearing, the plan's `.c` token): none (`c0`), or a listed
clearing's buildings (road_simp_clearings/<name>/<block>.csv, e.g. U's at D 0.05 on 5810) opened
in the eps world (their factor 1 in place of EPS) before the first step, so the flow the update
reads is the cleared world's. Its role is required: `real` (demolished: its buildings count in D
from the start, a building it cleared reading as covered, and the answers are scored with it) or
`device` (only the roads are built: D counts the roads' cover alone and the answers are scored
without the clearing).

Two answers, each scored as a road network is (Carving.of with one network of one stop: P on its
own mesh by road_state; D by common.MEASURE on the world-frame ground):
  paved: the triangles at r_t >= 1/2, unioned (closed and opened by SMOOTH_M and simplified to it,
    so the carved mesh meets no sliver: rev 1's raw union failed to mesh on 5810 three times);
  roads: the skeleton of the pixels at r >= 1/2 (reblock's Zhang-Suen thinning), its 8-neighbour
    steps merged into lines, each a road ROAD_W wide (road_corridor's rule): the trees' lines at a
    road's width, where `paved` keeps their trunks' widths. Thinning eats a strip from its ends, so
    the skeleton stops about a half-width short of the street where the paved strip meets it (rev
    3: 7 -- 61% of the length on 5810 in pieces more than 4 m from it): each end whose inscribed
    disk (its distance off the mask, and a pixel's diagonal) reaches the street is joined to it by
    the shortest line (rev 4 joined it to the nearest pixel off the design, up to a pixel from the
    street, and off-design pixels also sit inside the block between large triangles' centroids).
    Only the pieces linked to the street are kept, the length dropped reported (roads_apart_m):
    the paved answer has islands inside the block (on 5810 at D 0.05, k 0.08, r 3.5, 42 of its 50
    parts, 47% of its area), which conduct as paving does but are no road a vehicle or ΔP greedy
    can reach (rev 5 kept them: 54 -- 58% of two 5810 answers' length).

    PYTHONPATH=. uv run python research/roadless/road_simp.py run <block> <plan> cpu|gpu [--replace]
    PYTHONPATH=. uv run python research/roadless/road_simp.py rescore <block> <plan> cpu|gpu <rev>

`rescore` builds this revision's answers from revision <rev>'s design of the same plan (its
<block>_x.parquet and <block>_it.parquet) and writes them as this revision's, the iterates carried
over and `x_rev` recording where the design came from (`run` writes its own REV); a design from rev
7 on (the plan's grammar since; rev 6's and earlier are records under their own names).

plan: rs.D<D>.k<per_km>.r<radius>.q<q,...>.b<beta,...>.i<iters>.u<OC|A<gamma>>.c<0|<name>real|
<name>device> (the stages pair q and beta in order), e.g.
rs.D0.05.k0.01.r7.q1,1,1,1,1.b1,2,4,8,16.i60.uA0.5.cUdevice; its name (Plan.name) adds the raster's
pixel and the eps world's eps (.p<PIXEL>e<EPS>). Rows in road_simp_rows/<metric>/<name>/
<block>.parquet beside its design (<block>_x.parquet: x per pixel, parquet so the cluster's pull
brings it) and iterates (<block>_it.parquet), both written before the answers are scored, its
paved ground (<block>_ground.parquet) and its roads (<block>_roads.parquet).
"""
from __future__ import annotations

import re
import socket
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Protocol, TypeAlias

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely
from numba import njit
from numpy.typing import NDArray
from scipy import ndimage
from scipy.sparse.csgraph import connected_components

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
REV = 7              # the code's revision, in the plan's name (2: D bisected at the trial update,
                     # the answers smoothed and doubled; 3: the roads' mask unholed; 4: the
                     # skeleton's ends joined toward the street; 5: joined to it; 6: the
                     # pieces apart from the street dropped; 7: the update and the starting
                     # clearing Strategies, `.u` and `.c` in the plan)
X_FROM = 7           # the first revision whose designs rescore reads (the grammar since)
OUT = HERE / "road_simp_rows"
CLEARINGS = HERE / "road_simp_clearings"         # <name>/<block>.csv: a listed clearing's buildings
MESHES = HERE / "road_simp_rows" / ".meshes"     # the answers' carved meshes


class Update(Protocol):
    """How the design moves each step (module docstring; the plan's `.u` token)."""

    @property
    def name(self) -> str: ...

    @property
    def eta(self) -> float: ...                # B's exponent

    def paving(self, price: float, area: NDArray[np.float64],
               r: NDArray[np.float64]) -> NDArray[np.float64]:
        """The paving cost's derivative per triangle in its road density."""
        ...

    def cost(self, price: float, area: NDArray[np.float64], r: NDArray[np.float64]) -> float: ...

    def check(self, stages: tuple[tuple[float, float], ...]) -> None: ...


@dataclass(frozen=True)
class OC:
    """relax.simp's optimality criteria: the cost linear in the density, B^(1/2)."""
    name = "OC"
    eta = 0.5

    def paving(self, price: float, area: NDArray[np.float64],
               r: NDArray[np.float64]) -> NDArray[np.float64]:
        return price * area

    def cost(self, price: float, area: NDArray[np.float64], r: NDArray[np.float64]) -> float:
        return float(price * area @ r)

    def check(self, stages: tuple[tuple[float, float], ...]) -> None:
        return None


@dataclass(frozen=True, kw_only=True)
class Adaptive:
    """The adaptive transport network's update (module docstring): the cost price a r^gamma,
    B^(1/(1+gamma)), q 1."""
    gamma: float

    def __post_init__(self) -> None:
        if not self.gamma > 0:
            raise ValueError(f"gamma {self.gamma}: positive")

    @property
    def name(self) -> str:
        return f"A{self.gamma:g}"

    @property
    def eta(self) -> float:
        return 1.0 / (1.0 + self.gamma)

    def paving(self, price: float, area: NDArray[np.float64],
               r: NDArray[np.float64]) -> NDArray[np.float64]:
        return price * self.gamma * area * np.maximum(r, XMIN) ** (self.gamma - 1.0)

    def cost(self, price: float, area: NDArray[np.float64], r: NDArray[np.float64]) -> float:
        return float(price * area @ np.maximum(r, 0.0) ** self.gamma)

    def check(self, stages: tuple[tuple[float, float], ...]) -> None:
        if any(q != 1.0 for q, _ in stages):
            raise ValueError(f"{self.name}: q 1 at every stage (gamma sets the regime)")


def update_of(token: str) -> Update:
    """`token` as `OC` or `A<gamma>`."""
    if token == "OC":
        return OC()
    m = re.fullmatch(r"A([0-9.]+)", token)
    if m is None:
        raise SystemExit(f"{token}: an update is OC or A<gamma>")
    return Adaptive(gamma=float(m[1]))


class Clearing(Protocol):
    """The starting world (module docstring; the plan's `.c` token)."""

    @property
    def name(self) -> str: ...

    @property
    def real(self) -> bool: ...

    def cleared(self, bid: str, n: int) -> NDArray[np.bool_]: ...


@dataclass(frozen=True)
class NoClearing:
    name = "0"
    real = False

    def cleared(self, bid: str, n: int) -> NDArray[np.bool_]:
        return np.zeros(n, dtype=bool)


@dataclass(frozen=True, kw_only=True)
class Listed:
    """A clearing's buildings listed in CLEARINGS/<source>/<block>.csv (column `building`)."""
    source: str
    real: bool

    @property
    def name(self) -> str:
        return f"{self.source}{'real' if self.real else 'device'}"

    def cleared(self, bid: str, n: int) -> NDArray[np.bool_]:
        idx = pd.read_csv(CLEARINGS / self.source / f"{bid}.csv")["building"].to_numpy()
        out = np.zeros(n, dtype=bool)
        out[idx] = True
        return out


def clearing_of(token: str) -> Clearing:
    """`token` as `0`, `<source>real` or `<source>device`."""
    if token == "0":
        return NoClearing()
    m = re.fullmatch(r"([A-Za-z0-9]+?)(real|device)", token)
    if m is None:
        raise SystemExit(f"{token}: a clearing is 0, <source>real or <source>device")
    return Listed(source=m[1], real=m[2] == "real")


@dataclass(frozen=True, kw_only=True)
class Plan:
    """A run's settings, all in its name (module docstring)."""
    D: float                                   # the displacement budget
    per_km: float                              # the L_P a km of road must buy back
    radius: float                              # m: the cone filter's
    stages: tuple[tuple[float, float], ...]    # (q, beta) in order
    iters: int                                 # updates per stage
    update: Update
    clearing: Clearing

    def __post_init__(self) -> None:
        if not (self.D > 0 and self.per_km > 0 and self.radius > 0 and self.stages
                and self.iters > 0):
            raise ValueError(f"{self}: a budget, a price, a radius, stages and updates, each "
                             "positive")
        self.update.check(self.stages)

    @property
    def name(self) -> str:
        return self.name_at(REV)

    def name_at(self, rev: int) -> str:
        """The plan's name as revision `rev` wrote it."""
        q = ",".join(f"{q:g}" for q, _ in self.stages)
        b = ",".join(f"{b:g}" for _, b in self.stages)
        return (f"rs.D{self.D:g}.k{self.per_km:g}.r{self.radius:g}.q{q}.b{b}.i{self.iters}"
                f".u{self.update.name}.c{self.clearing.name}.p{PIXEL:g}e{EPS:g}v{rev}")

    @staticmethod
    def of(spec: str) -> Plan:
        m = re.fullmatch(r"rs\.D([0-9.e-]+)\.k([0-9.e-]+)\.r([0-9.]+)\.q([0-9.,]+)\.b([0-9.,]+)"
                         r"\.i([0-9]+)\.u([A-Za-z0-9.]+?)\.c([A-Za-z0-9]+)", spec)
        if m is None:
            raise SystemExit(f"{spec}: a plan is rs.D<D>.k<per_km>.r<radius>.q<q,..>.b<beta,..>"
                             ".i<iters>.u<OC|A<gamma>>.c<0|<source>real|<source>device>")
        qs = [float(v) for v in m[4].split(",")]
        bs = [float(v) for v in m[5].split(",")]
        if len(qs) != len(bs):
            raise SystemExit(f"{spec}: as many q as beta")
        return Plan(D=float(m[1]), per_km=float(m[2]), radius=float(m[3]),
                    stages=tuple(zip(qs, bs, strict=True)), iters=int(m[6]),
                    update=update_of(m[7]), clearing=clearing_of(m[8]))


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

    def __init__(self, m: gcl.Model, *, radius: float, cleared: NDArray[np.bool_],
                 real: bool) -> None:
        if len(m.fill.e):
            raise ValueError("a width form that fills gaps: a road's density would not open them")
        if m.carving.n != len(cleared):
            raise ValueError(f"a clearing of {len(cleared)} buildings on a mesh of "
                             f"{m.carving.n} atoms: the plain mesh's, one atom a building")
        self.m = m
        fm = m.fx.fm
        self.raster = Raster.of(m.cen_, radius=radius)
        a, b, c = (fm.xy[fm.tri[:, k]] for k in range(3))
        self.area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                                 - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        # the starting world (module docstring): the clearing's buildings open, the rest at EPS
        self.cleared = cleared
        self.real = real
        self.ka = np.where(cleared, 1.0, EPS)
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
    def covered(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per building its covered share: the roads' (cover r), a real clearing's buildings 1."""
        X = self.cover @ r
        return np.maximum(X, self.cleared.astype(np.float64)) if self.real else X

    def D(self, r: NDArray[np.float64]) -> float:
        return float(self.w @ common.MEASURE.ramp(self.covered(r)))

    def dD(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """dD / dr per triangle (the ramp's slope where a building is under it, else 0; a real
        clearing's buildings saturated)."""
        X = self.covered(r)
        live = (X < common.MEASURE.tau) & ~(self.cleared & self.real)
        return np.asarray(self.cover.T @ (self.w * live / common.MEASURE.tau))

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


def _update(x: NDArray[np.float64], on: NDArray[np.bool_], gain: NDArray[np.float64],
            paving: NDArray[np.float64], gD: NDArray[np.float64], *, eta: float,
            D_of: Callable[[NDArray[np.float64]], float], budget: float,
            nu0: float) -> tuple[NDArray[np.float64], float]:
    """One update of the design pixels `on`: x (gain / (paving + nu gD))^eta within MOVE of x and
    [XMIN, 1] (OC's eta 1/2, Adaptive's 1 / (1 + gamma): module docstring); nu 0 if the update's
    D (D_of) is within the budget, else bisected (from nu0, the last update's, bracketed by
    factors of 10) to the least nu whose update's D is, to a factor of 1 + NU_TOL (D falls as nu
    rises: every pixel's update does)."""
    gain = np.maximum(gain, 0.0)
    lo_x, hi_x = np.maximum(XMIN, x - MOVE), np.minimum(1.0, x + MOVE)

    def trial(nu: float) -> NDArray[np.float64]:
        B = gain / np.maximum(paving + nu * gD, 1e-300)
        return np.where(on, np.clip(x * B ** eta, lo_x, hi_x), x)

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


class Iterate(NamedTuple):
    """One update's row (`cost`: the paving cost's price times sum a r^gamma, OC's gamma 1)."""
    q: float
    beta: float
    it: int
    P: float
    L_P: float
    A: float
    cost: float
    D: float
    nu: float
    change: float
    t: float


def simp(rd: Roads, plan: Plan, *,
         log: Any = print) -> tuple[NDArray[np.float64], list[Iterate]]:
    """The plan's update from XMIN everywhere (no road)."""
    x = np.full(len(rd.raster.on), XMIN)
    lam = rd.price(plan.per_km)
    on = rd.raster.on
    rows: list[Iterate] = []
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
            gC = rd.back(xf, beta, plan.update.paving(lam, rd.area, r))
            gD = rd.back(xf, beta, rd.dD(r))
            A, D = float(rd.area @ r), rd.D(r)
            cost = plan.update.cost(lam, rd.area, r)
            xn, nu = _update(x, on, -gP, gC, gD, eta=plan.update.eta, D_of=D_of, budget=plan.D,
                             nu0=nu)
            change = float(np.abs(xn - x)[on].max())
            x = xn
            rows.append(Iterate(q=q, beta=beta, it=it, P=P, L_P=1.0 - P / rd.P0, A=A, cost=cost,
                                D=D, nu=nu, change=change, t=time.time() - t0))
            log(f"q {q:g} beta {beta:g} it {it}: L_P {1 - P / rd.P0:.4f} + cost "
                f"{cost / rd.P0:.4f}, D {D:.4f}, paved {A:.0f} m^2, change {change:.3f} "
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
    xy = ra.lo + PIXEL * np.asarray(segs, dtype=np.float64).reshape(-1, 2, 2)
    # the ends joined to the street (module docstring): an end whose inscribed disk reaches it
    inside = ndimage.distance_transform_edt(on) * PIXEL
    deg = ndimage.correlate(skel.astype(np.int64), np.ones((3, 3), dtype=np.int64),
                            mode="constant")
    ea, eb = np.nonzero(skel & (deg == 2))
    ends = shapely.points(ra.lo + PIXEL * (np.stack([ea, eb], axis=1) + 0.5))
    street = rd.m.geo.streets
    near = shapely.distance(ends, street) <= inside[ea, eb] + np.sqrt(2.0) * PIXEL
    joins = shapely.shortest_line(ends[near], street)
    return shapely.line_merge(shapely.multilinestrings(
        np.concatenate([shapely.linestrings(xy), joins])))


def linked(lines: NDArray[np.object_], street: Any) -> NDArray[np.bool_]:
    """Per line whether its piece (the lines that touch it, and theirs) reaches the street within
    budget.STREET_TOL."""
    tree = shapely.STRtree(lines)
    a, b = tree.query(lines, predicate="dwithin", distance=1e-6)
    n = len(lines)
    comp = connected_components(sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)))[1]
    near = shapely.distance(lines, street) <= STREET_TOL
    return np.isin(comp, np.unique(comp[near]))


def score(b: Any, metric: gcl.ComplexMetric, ground_world: Any, *,
          cleared: NDArray[np.bool_]) -> dict[str, float]:
    """A paved ground scored as a road network (module docstring), beside the buildings `cleared`
    (a real clearing's; none for a device's or no clearing): in the state and in D."""
    geo = gc.geometry(b)
    carving = gcl.Carving.of(geo, [[shapely.make_valid(geo.local(ground_world))]])
    MESHES.mkdir(parents=True, exist_ok=True)
    fx = metric.fixed(b, cache=MESHES, carving=carving)
    m = metric.model(b, fx, demand=common.PopulationDemand())
    prefix = gcl.Prefix(net=0, stop=1)
    lens = m.lenses_of(m.cleared_road_state(m.k_of(cleared), prefix) if cleared.any()
                       else m.road_state(prefix))
    W = common.AreaPopulation().weights(np.asarray(b.buildings.outlines))
    inc = IncrementalDestroyed(common.MEASURE, b.buildings)
    inc.add(ground_world)
    if cleared.any():
        inc.add(shapely.union_all(np.asarray(b.buildings.outlines)[cleared]))
    return dict(P=lens.P, L_P=1.0 - lens.P / m.base.P, L_Pq=1.0 - lens.Pq / m.base.Pq,
                L_J2=1.0 - lens.J2 / m.base.J2, D_exact=float((inc.c() * W).sum() / W.sum()),
                paved=float(shapely.area(ground_world)), enclosed=lens.enclosed)


def roads_of(b: Any, metric: gcl.ComplexMetric, plan: Plan) -> Roads:
    """The block's problem in the plan's starting world."""
    m = metric.clearing(b, demand=common.PopulationDemand())
    return Roads(m, radius=plan.radius, cleared=plan.clearing.cleared(b.block_id, m.n),
                 real=plan.clearing.real)


def run(bid: str, plan: Plan, device: str, *, standing: clearings.Standing) -> None:
    metric = metrics.metric_of(metrics.RUNTIME, device=device)
    out = OUT / metrics.RUNTIME / plan.name / f"{bid}.parquet"
    if not standing.todo(out):
        return
    t0 = time.time()
    [b] = common.build_blocks([bid])
    rd = roads_of(b, metric, plan)
    print(f"{bid}: {rd.m.fx.fm.nt} triangles, {int(rd.raster.on.sum())} design pixels, P0 "
          f"{rd.P0:.6e}, {int(rd.cleared.sum())} buildings cleared at the start "
          f"({time.time() - t0:.0f}s)", flush=True)
    x, rows = simp(rd, plan, log=lambda s: print(s, flush=True))
    answer(b, metric, rd, x, plan, pd.DataFrame(rows).assign(x_rev=REV), device=device, out=out,
           t0=t0, standing=standing)


def rescore(bid: str, plan: Plan, device: str, *, rev: int,
            standing: clearings.Standing) -> None:
    """This revision's answers from revision `rev`'s design (module docstring)."""
    if not X_FROM <= rev < REV:
        raise SystemExit(f"rev {rev}: a design to rescore is written by revisions {X_FROM} .. "
                         f"{REV - 1}")
    metric = metrics.metric_of(metrics.RUNTIME, device=device)
    out = OUT / metrics.RUNTIME / plan.name / f"{bid}.parquet"
    if not standing.todo(out):
        return
    t0 = time.time()
    src = OUT / metrics.RUNTIME / plan.name_at(rev) / f"{bid}.parquet"
    x = pd.read_parquet(src.with_name(f"{bid}_x.parquet"))["x"].to_numpy()
    rows = pd.read_parquet(src.with_name(f"{bid}_it.parquet")).assign(x_rev=rev)
    [b] = common.build_blocks([bid])
    rd = roads_of(b, metric, plan)
    if len(x) != len(rd.raster.on):
        raise SystemExit(f"{src}: {len(x)} pixels, this raster {len(rd.raster.on)}")
    answer(b, metric, rd, x, plan, rows, device=device, out=out, t0=t0, standing=standing)


def answer(b: Any, metric: gcl.ComplexMetric, rd: Roads, x: NDArray[np.float64], plan: Plan,
           rows: pd.DataFrame, *, device: str, out: Path, t0: float,
           standing: clearings.Standing) -> None:
    """The design's two answers scored (module docstring), written beside it to `out`."""
    bid = b.block_id
    beta = plan.stages[-1][1]
    geo = rd.m.geo
    # the design and its iterates first, so a failure while scoring (two of rev 4's twelve 5810
    # runs were killed meshing their answers) leaves them for `rescore`
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": x}).to_parquet(out.with_name(f"{bid}_x.parquet"))
    rows[list(Iterate._fields)].to_parquet(out.with_name(f"{bid}_it.parquet"))
    ground = geo.utm(paved(rd, x, beta))
    lines = shapely.get_parts(geo.utm(roads(rd, x, beta)))
    real = rd.cleared if rd.real else np.zeros_like(rd.cleared)
    del rd                                     # the eps world's model, before two more are built
    on = linked(lines, shapely.union_all(b.streets.geometry.to_numpy()))
    net = gpd.GeoDataFrame(geometry=list(lines[on]), crs=b.crs).assign(width_m=ROAD_W)
    sc = {f"paved_{k}": v for k, v in score(b, metric, ground, cleared=real).items()}
    sc |= {f"roads_{k}": v for k, v in score(b, metric, road_corridor(net), cleared=real).items()}
    sc["roads_m"] = float(net.length.sum())
    sc["roads_apart_m"] = float(shapely.length(lines[~on]).sum())
    print(f"{bid} {plan.name}: paved L_P {sc['paved_L_P']:.4f}, D {sc['paved_D_exact']:.4f}, "
          f"{sc['paved_paved']:.0f} m^2; roads L_P {sc['roads_L_P']:.4f}, D "
          f"{sc['roads_D_exact']:.4f}, {sc['roads_m']:.0f} m ({sc['roads_apart_m']:.0f} m apart "
          f"dropped) ({time.time() - t0:.0f}s)", flush=True)
    gpd.GeoDataFrame(geometry=[ground], crs=b.crs).to_parquet(
        out.with_name(f"{bid}_ground.parquet"))
    net.to_parquet(out.with_name(f"{bid}_roads.parquet"))
    df = rows.assign(block=bid, plan=plan.name, **sc, device=device, host=socket.gethostname(),
                     peak_rss_gb=gap_calib.peak_gb(), t_total=time.time() - t0)
    standing.write(df, out)


def main(argv: list[str]) -> None:
    argv, standing = clearings.standing_of(argv)
    if len(argv) == 4 and argv[0] == "run" and argv[3] in ("cpu", "gpu"):
        run(argv[1], Plan.of(argv[2]), argv[3], standing=standing)
    elif len(argv) == 5 and argv[0] == "rescore" and argv[3] in ("cpu", "gpu"):
        rescore(argv[1], Plan.of(argv[2]), argv[3], rev=int(argv[4]), standing=standing)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
