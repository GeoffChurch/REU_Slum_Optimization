"""Reblocking top-down: a multi-phase partition SIMP on the block's fixed mesh (owner 2026-10-27:
"methods for literal reblocking, where we partition the region (so every road is a loop)";
.superpowers/sdd/2026-10-27-partition/brief.md, part 1).

The design is K label fields psi_k per design pixel (road_simp's raster and cone filter of radius
R), phi = the softmax of the filtered fields at temperature T. The roads are where the labels
change: g = 1 - sum_k phi_k^2, read as a road density rho = (g - g_pure) / (g_pair - g_pure)
clipped to [0, 1], g_pure its value on a pure label (one field 1, the rest 0: the floor of g over
the design's box) and g_pair at the middle of a sharp two-label interface (two fields 1/2): the
interior of a label reads 0 at any T, an interface's middle 1, so the projection's threshold sits
between them (the raw 1 - sum phi^2 leaves 0.08 -- 0.7 of road density inside a label at the
temperatures that make a 7 m road, every building past the ramp's knee). Then road_simp's eps
world: per triangle r = proj(rho, beta) of its pixel, s = r^q, kt = kt0 + s (1 - kt0), b = bw +
s db (road_simp.Roads.world_of: kappa 10, the metric's own road); D the ramp over the buildings'
covered shares, C the sheet's conductance added (road_simp.Roads.D, .C). The contrast is kappa 10
only: the grounded leak (growth.GroundedLeak) needs a System per state with the leak on its
graph, where road_simp.Roads.state reads the model's own, so a second state path; not built.

The width of an interface is set by R and T: across a sharp step in the design the band rho >=
1/2 widens with T and saturates near 0.8 R. A stage's T is a multiple of T_w (`Plan.tw`: the T at
which a sharp straight interface's band is ROAD_W wide, by bisection on a synthetic step through
the plan's own filter), so the continuation runs toward ROAD_W from softer, wider interfaces.
q and beta are continued as road_simp's.

No dead ends by construction: a label boundary closes, meets the street or meets others at a
junction. The 0/1 answers can still have them (a band thinned below the threshold breaks): they
are counted on each answer's skeleton.

Objective: P + lambda C (road_simp.SheetPrice: lambda = price P0 / SHEET_M2) under D <= Plan.D.
Gradient: road_simp's adjoint (dP/ds per triangle, through kt and b), back through the
projection, the normalization, the softmax (dphi_k / dpsi~_j = phi_k (delta_kj - phi_j) / T) and
the filter's transpose. The update is MMA (Svanberg 1987), one constraint: OC needs a gain of one
sign, and a label field's has none (moving an interface helps one side and hurts the other); the
multiplier mu of D is bisected on the true D of the trial update (road_simp.least_nu: the
projection makes a linearised D overshoot), as road_simp's nu.

The start is a Strategy (Start, the plan's last token): `vor<n>s<seed>`, the Voronoi cells of n
seeds placed by k-means on the homes' centroids; `spec<n>s<seed>`, the shelved spectral normalized
cut (memory "Voronoi-adjacency partition method": a gap-aware affinity exp(-gap^2 / 2 sigma^2)
between each building and its KNN nearest, sigma the median gap, the n smallest eigenvectors of
the normalized Laplacian row-normalized and k-means'd; each pixel takes its nearest building's
cluster); `nodal<n>s<seed>`, a Dirichlet spectral partition aiming at the spectral minimal
partition (Helffer, Hoffmann-Ostenhof, Terracini: the pieces' smallest first eigenvalue the
largest, a piece's P at most its demand over its lambda_1; Nodal); each cell then coloured into
the K labels (DSatur on the cells' adjacency: K labels make any number of sub-blocks, four
colouring every planar partition, DSatur not always finding it). `ears` (the bottom-up arm's
answer) and `divides` (its Morse-Smale basins of u, research/roadless/ears.py's generator, to be
imported) are refused until they exist. Each start is scored as its own cut answer before the
update moves it (`start_*` in the row).

Answers, each scored as a road network (road_simp.scored: its own carved mesh, RUNTIME), with the
measures `measures` gives (sub-blocks, road length, dead ends, the depth measures, the damage
term's cut share):
  start: the start's own cut answer (below), before the update;
  band: the triangles at r >= 1/2 (road_simp.paved_of);
  cut: the label boundaries (argmax phi on the raster, each pixel edge between two labels, merged
    and simplified) as roads ROAD_W wide (road_corridor), clipped to the block.

    P=research/roadless/partition.py                   # from the repo root, PYTHONPATH=.
    uv run python $P run <block> <plan> cpu|gpu [--replace]
    uv run python $P start <block> <plan>              # the start's D, C and cells, no solve

plan: pt.K<K>.R<radius>.t<T/T_w,...>.b<beta,...>.q<q,...>.i<iters>.D<D>.s<price>.<start> (the
stages pair t, beta and q in order), e.g. pt.K4.R10.t2,1.5,1.2,1,1.b1,2,4,8,16.q1,2,3,3,3.i40
.D0.05.s0.1.vor12s0; its name adds the raster's pixel and the eps world's eps and the code's
revision (.p<PIXEL>e<EPS>v<REV>). Rows in partition_rows/<metric>/<name>/<block>.parquet (the
iterates and the answers' scores), beside the design (<block>_psi.parquet, the fields on the
design pixels), the answers' grounds and lines (<block>_<answer>_ground.parquet, _lines.parquet).
"""
from __future__ import annotations

import re
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Protocol, TypeAlias

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import shapely
from numpy.typing import NDArray
from scipy import ndimage, signal
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import common  # noqa: E402
import damage  # noqa: E402
import gap_calib  # noqa: E402
import gap_clear as gcl  # noqa: E402
import metrics  # noqa: E402
import relax  # noqa: E402
import road_simp as rs  # noqa: E402

from reblock.budget import road_corridor  # noqa: E402
from reblock.methods.betweenness.thinning import skeletonize  # noqa: E402

F64: TypeAlias = NDArray[np.float64]
I64: TypeAlias = NDArray[np.int64]
B: TypeAlias = NDArray[np.bool_]

ROAD_W = rs.ROAD_W       # m: a road's width (the interfaces' continuation target)
EPS = rs.EPS             # the eps world's standing footprint
PIXEL = rs.PIXEL         # m: the design raster's pixel
HALF = rs.HALF           # the answers' threshold on r
MOVE = 0.2               # MMA's move limit (road_simp's OC's)
ASYM_INIT = 0.5          # MMA: the asymptotes' first distance from x (the box is [0, 1])
ASYM_GROW, ASYM_SHRINK = 1.2, 0.7    # MMA: the asymptotes' update (Svanberg's)
ASYM_MIN, ASYM_MAX = 0.01, 10.0      # MMA: the asymptotes' distance from x, bounds
RAA = 1e-6               # MMA: p and q's floor, relative to the largest |derivative| (keeps a
                         # variable with a zero derivative from moving to an asymptote)
TW_TOL = 1e-4            # Plan.tw: T_w bisected to this, relative
TW_N = 200               # Plan.tw: the synthetic step's raster (pixels a side)
KNN = 8                  # Spectral: each building's nearest neighbours (by centroid)
NODAL_FLOOR = 0.01       # Nodal: the mass's share spread by area (the open ground's cells
                         # hold no demand: M positive definite)
KMEANS_ITER = 50         # Voronoi, Spectral: k-means iterations
SNAP_TOL = 1.0           # m: the cut answer's boundaries simplified to this
FACE_M2 = 100.0          # m^2: a sub-block is a face of the block less the roads this large
TIP_STREET_M = 5.0       # m: a skeleton end farther than this from the street is a dead end
REACH_M = 50.0           # m: a home "reached": within this of a road (the growth pilot's)
DEEP_Q = 0.75            # the deepest quarter: homes at depth quantile >= this
DAMAGE_STEP_M = 10.0     # the damage term's piece length (NOTES, "Roads from the vote, stage 2")
DAMAGE_CARRY_M = 100.0   # the damage term's carry
REV = 1
OUT = HERE / "partition_rows"


# ---------------------------------------------------------------- the starts


class Start(Protocol):
    """The starting partition (module docstring; the plan's last token): per design pixel a cell,
    coloured into labels by `colour`."""

    @property
    def name(self) -> str: ...

    def cells(self, ra: rs.Raster, m: gcl.Model) -> I64:
        """Per pixel (the raster's, on and off) its cell, 0 .. n-1."""
        ...


@dataclass(frozen=True, kw_only=True)
class Voronoi:
    """The Voronoi cells of n seeds, k-means on the homes' centroids (seeded)."""
    n: int
    seed: int

    @property
    def name(self) -> str:
        return f"vor{self.n}s{self.seed}"

    def cells(self, ra: rs.Raster, m: gcl.Model) -> I64:
        cen = shapely.get_coordinates(shapely.centroid(np.asarray(m.geo.polys)))
        seeds, _ = kmeans2(cen, self.n, iter=KMEANS_ITER, minit="++", seed=self.seed)
        return np.asarray(cKDTree(seeds).query(_pixel_xy(ra))[1], dtype=np.int64)


@dataclass(frozen=True, kw_only=True)
class Spectral:
    """The spectral normalized cut of the buildings into n clusters (module docstring), each pixel
    its nearest building's (seeded k-means on the embedding)."""
    n: int
    seed: int

    @property
    def name(self) -> str:
        return f"spec{self.n}s{self.seed}"

    def cells(self, ra: rs.Raster, m: gcl.Model) -> I64:
        polys = np.asarray(m.geo.polys)
        cen = shapely.get_coordinates(shapely.centroid(polys))
        _, nb = cKDTree(cen).query(cen, k=KNN + 1)
        i = np.repeat(np.arange(len(polys)), KNN)
        j = nb[:, 1:].ravel()
        gap = shapely.distance(polys[i], polys[j])
        sigma = float(np.median(gap[gap > 0])) if (gap > 0).any() else 1.0
        w = np.exp(-gap ** 2 / (2.0 * sigma ** 2))
        W = sp.coo_matrix((w, (i, j)), shape=(len(polys),) * 2).tocsr()
        W = W.maximum(W.T)
        d = np.asarray(W.sum(axis=1)).ravel()
        Dm = sp.diags(1.0 / np.sqrt(d))
        A = Dm @ W @ Dm                     # the normalized Laplacian's smallest: A's largest
        _, vec = spla.eigsh(A, k=self.n, which="LA")
        emb = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-300)
        _, lab = kmeans2(emb, self.n, iter=KMEANS_ITER, minit="++", seed=self.seed)
        # each pixel its nearest building's cluster (the buildings rasterized at their pixels)
        owner = _building_raster(ra, polys)
        _, idx = ndimage.distance_transform_edt(owner < 0, return_indices=True)
        near = owner[idx[0], idx[1]].ravel()
        return np.asarray(lab[near], dtype=np.int64)


@dataclass(frozen=True, kw_only=True)
class Nodal:
    """A Dirichlet spectral partition (module docstring): the n lowest eigenfunctions of the
    baseline's grounded operator (the street ground, the exits' leak on its diagonal, every
    footprint standing at its conductance), A v = lambda M v with the mass M the population per
    cell plus NODAL_FLOOR of it spread by area (a piece's P is at most its demand over its own
    lambda_1); shift-invert about 0 on the model's own factorization; the cells k-means'd
    (seeded) on the row-normalized eigenvectors, each pixel its triangle's cell's cluster (an
    off or ground pixel its nearest labelled one's)."""
    n: int
    seed: int

    @property
    def name(self) -> str:
        return f"nodal{self.n}s{self.seed}"

    def cells(self, ra: rs.Raster, m: gcl.Model) -> I64:
        st = m.state(np.zeros(m.n))
        assert st.A is not None and st.factor is not None
        idx = np.flatnonzero(st.reach.unknown)
        A = st.A[idx][:, idx]
        f = np.asarray(st.sys.Pc @ m.pop)[idx]
        area = np.asarray(m.leak_site.area)[idx]
        mass = f / f.sum() + NODAL_FLOOR * area / area.sum()
        full = np.zeros(st.A.shape[0])
        factor = st.factor

        def inv(v: F64) -> F64:
            full[idx] = np.ravel(v)
            return np.asarray(factor.solve(full)[idx])

        op = spla.LinearOperator(A.shape, matvec=inv, dtype=np.float64)
        _, vec = spla.eigsh(A, k=self.n, M=sp.diags(mass), sigma=0.0, which="LM", OPinv=op)
        emb = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-300)
        _, lab = kmeans2(emb, self.n, iter=KMEANS_ITER, minit="++", seed=self.seed)
        cell_lab = np.full(st.A.shape[0], -1, dtype=np.int64)
        cell_lab[idx] = lab
        tri = cell_lab[m.c.cell]
        px = np.full(ra.shape[0] * ra.shape[1], -1, dtype=np.int64)
        px[ra.pix[tri >= 0]] = tri[tri >= 0]
        grid = px.reshape(ra.shape)
        _, near = ndimage.distance_transform_edt(grid < 0, return_indices=True)
        return np.asarray(grid[near[0], near[1]].ravel(), dtype=np.int64)


def start_of(token: str) -> Start:
    m = re.fullmatch(r"(vor|spec|nodal)([0-9]+)s([0-9]+)", token)
    if token == "ears":
        raise SystemExit("ears: the bottom-up arm's answer does not exist yet "
                         "(.superpowers/sdd/2026-10-27-ears/brief.md)")
    if token == "divides":
        raise SystemExit("divides: the ears arm's generator (the Morse-Smale basins of u) is not "
                         "in research/roadless/ears.py yet; it is to be imported, not duplicated")
    if m is None:
        raise SystemExit(f"{token}: a start is vor<n>s<seed>, spec<n>s<seed> or nodal<n>s<seed>")
    n, seed = int(m[2]), int(m[3])
    return {"vor": Voronoi, "spec": Spectral, "nodal": Nodal}[m[1]](n=n, seed=seed)


def _pixel_xy(ra: rs.Raster) -> F64:
    i, j = np.unravel_index(np.arange(ra.shape[0] * ra.shape[1]), ra.shape)
    return ra.lo + PIXEL * (np.stack([i, j], axis=1) + 0.5)


def _building_raster(ra: rs.Raster, polys: NDArray[np.object_]) -> I64:
    """Per pixel (raster shape) the building whose outline holds its centre, -1 none."""
    xy = _pixel_xy(ra)
    owner = np.full(len(xy), -1, dtype=np.int64)
    tree = shapely.STRtree(polys)
    pts = shapely.points(xy)
    p, b = tree.query(pts, predicate="within")
    owner[p] = b
    return owner.reshape(ra.shape)


def colour(cells: I64, shape: tuple[int, int], k: int) -> I64:
    """Per cell a label 0 .. k-1, no two cells that share a pixel edge alike (DSatur); raises if
    the cells' adjacency needs more than k."""
    c = cells.reshape(shape)
    pairs = np.concatenate([np.stack([c[1:, :].ravel(), c[:-1, :].ravel()], axis=1),
                            np.stack([c[:, 1:].ravel(), c[:, :-1].ravel()], axis=1)])
    pairs = np.unique(np.sort(pairs[pairs[:, 0] != pairs[:, 1]], axis=1), axis=0)
    n = int(cells.max()) + 1
    nbr: list[set[int]] = [set() for _ in range(n)]
    for a, b in pairs:
        nbr[a].add(int(b))
        nbr[b].add(int(a))
    lab = np.full(n, -1, dtype=np.int64)
    for _ in range(n):
        free = np.flatnonzero(lab < 0)
        sat = [len({lab[v] for v in nbr[u] if lab[v] >= 0}) for u in free]
        u = int(free[np.lexsort((-np.array([len(nbr[u]) for u in free]), -np.array(sat)))[0]])
        used = {int(lab[v]) for v in nbr[u] if lab[v] >= 0}
        options = [x for x in range(k) if x not in used]
        if not options:
            raise SystemExit(f"colouring the start's {n} cells needs more than K {k} labels")
        lab[u] = options[0]
    return lab


# ---------------------------------------------------------------- the plan


@dataclass(frozen=True, kw_only=True)
class Plan:
    """A run's settings, all in its name (module docstring)."""
    K: int
    radius: float
    stages: tuple[tuple[float, float, float], ...]     # (T / T_w, beta, q) in order
    iters: int
    D: float
    price: rs.SheetPrice
    start: Start

    def __post_init__(self) -> None:
        if not (self.K >= 2 and self.radius > 0 and self.stages and self.iters > 0
                and self.D > 0):
            raise ValueError(f"{self}: K at least 2; a radius, stages, updates and a budget, "
                             "each positive")

    @property
    def name(self) -> str:
        t = ",".join(f"{t:g}" for t, _, _ in self.stages)
        b = ",".join(f"{b:g}" for _, b, _ in self.stages)
        q = ",".join(f"{q:g}" for _, _, q in self.stages)
        return (f"pt.K{self.K}.R{self.radius:g}.t{t}.b{b}.q{q}.i{self.iters}.D{self.D:g}"
                f".{self.price.name}.{self.start.name}.p{PIXEL:g}e{EPS:g}v{REV}")

    @staticmethod
    def of(spec: str) -> Plan:
        m = re.fullmatch(r"pt\.K([0-9]+)\.R([0-9.]+)\.t([0-9.,]+)\.b([0-9.,]+)\.q([0-9.,]+)"
                         r"\.i([0-9]+)\.D([0-9.e-]+)\.s([0-9.e-]+)\.([a-z]+[0-9]+s[0-9]+|ears|divides)",
                         spec)
        if m is None:
            raise SystemExit(f"{spec}: a plan is pt.K<K>.R<radius>.t<T/T_w,..>.b<beta,..>"
                             ".q<q,..>.i<iters>.D<D>.s<price>.<vor<n>s<seed>|spec<n>s<seed>>")
        ts, bs, qs = ([float(v) for v in m[k].split(",")] for k in (3, 4, 5))
        if not len(ts) == len(bs) == len(qs):
            raise SystemExit(f"{spec}: as many t as beta as q")
        return Plan(K=int(m[1]), radius=float(m[2]),
                    stages=tuple(zip(ts, bs, qs, strict=True)), iters=int(m[6]),
                    D=float(m[7]), price=rs.SheetPrice(price=float(m[8])),
                    start=start_of(m[9]))


# ---------------------------------------------------------------- the design's chain


def _g(L: F64) -> F64:
    """1 - sum phi^2 of the softmax of logits L (labels on axis 0)."""
    phi = np.exp(L - L.max(axis=0))
    phi /= phi.sum(axis=0)
    return np.asarray(1.0 - (phi ** 2).sum(axis=0))


def g_ends(T: float, K: int) -> tuple[float, float]:
    """(g on a pure label, g at the middle of a sharp two-label interface) at temperature T."""
    pure = np.zeros((K, 1))
    pure[0] = 1.0
    pair = np.zeros((K, 1))
    pair[:2] = 0.5
    return float(_g(pure / T)[0]), float(_g(pair / T)[0])


class Labels:
    """The K label fields' chain on the raster (module docstring): fields (K, pixels) -> the
    filtered fields -> phi -> rho per pixel, and its way back."""

    def __init__(self, ra: rs.Raster, *, K: int) -> None:
        self.ra, self.K = ra, K
        self.kern = ra.kernel                    # symmetric: correlation is convolution

    def filter(self, f: F64) -> F64:
        """Raster.filter by FFT, every field at once."""
        ra = self.ra
        img = np.where(ra.on, f, 0.0).reshape(self.K, *ra.shape)
        y = signal.fftconvolve(img, self.kern[None], mode="same", axes=(1, 2))
        y = y.reshape(self.K, -1)
        return np.where(ra.on, y / np.where(ra.on, ra.norm, 1.0), 0.0)

    def filter_T(self, g: F64) -> F64:
        ra = self.ra
        img = np.where(ra.on, g / np.where(ra.on, ra.norm, 1.0), 0.0).reshape(self.K, *ra.shape)
        y = signal.fftconvolve(img, self.kern[None], mode="same", axes=(1, 2))
        return np.where(ra.on, y.reshape(self.K, -1), 0.0)

    def rho(self, f: F64, T: float) -> tuple[F64, F64, F64]:
        """(rho per pixel, phi (K, pixels), rho's raw value before the clip)."""
        lo, hi = g_ends(T, self.K)
        L = self.filter(f) / T
        phi = np.exp(L - L.max(axis=0))
        phi /= phi.sum(axis=0)
        raw = ((1.0 - (phi ** 2).sum(axis=0)) - lo) / (hi - lo)
        return np.clip(raw, 0.0, 1.0), phi, raw

    def back(self, g_rho: F64, phi: F64, raw: F64, T: float) -> F64:
        """A per-pixel derivative in rho back to the fields (K, pixels)."""
        lo, hi = g_ends(T, self.K)
        live = (raw > 0.0) & (raw < 1.0)
        g_phi = np.where(live, g_rho, 0.0) * (-2.0 * phi) / (hi - lo)
        g_L = phi * (g_phi - (g_phi * phi).sum(axis=0))       # the softmax's Jacobian
        return self.filter_T(g_L / T)

    def argmax(self, f: F64, T: float) -> I64:
        """Per pixel its label (argmax phi), an off pixel its nearest on pixel's."""
        lab = np.argmax(self.filter(f), axis=0)
        on = self.ra.on.reshape(self.ra.shape)
        _, idx = ndimage.distance_transform_edt(~on, return_indices=True)
        return np.asarray(lab.reshape(self.ra.shape)[idx[0], idx[1]].ravel(), dtype=np.int64)


def tw(radius: float, K: int) -> float:
    """T_w: the temperature at which a sharp straight interface's band rho >= 1/2 is ROAD_W wide,
    through the plan's own cone filter on a synthetic TW_N x TW_N raster (the band's width read
    by linear interpolation of the profile across the step); raises if no T reaches it (the band
    saturates near 0.8 R)."""
    on = np.ones(TW_N * TW_N, dtype=bool)
    ra = rs.Raster.of(np.array([[0.5, 0.5], [TW_N - 0.5, TW_N - 0.5]]), radius=radius)
    ra = ra._replace(on=on, norm=ndimage.correlate(np.ones((TW_N, TW_N)), ra.kernel,
                                                   mode="constant").ravel())
    lab = Labels(ra, K=K)
    f = np.zeros((K, TW_N, TW_N))
    f[0, : TW_N // 2] = 1.0
    f[1, TW_N // 2:] = 1.0
    f = f.reshape(K, -1)

    def width(T: float) -> float:
        row = lab.rho(f, T)[0].reshape(TW_N, TW_N)[:, TW_N // 2]
        x = np.arange(TW_N) + 0.5
        up = np.flatnonzero((row[:-1] < HALF) & (row[1:] >= HALF))
        dn = np.flatnonzero((row[:-1] >= HALF) & (row[1:] < HALF))
        if not len(up) or not len(dn):
            return 0.0

        def cross(i: int) -> float:
            return float(x[i] + (HALF - row[i]) / (row[i + 1] - row[i]))
        return cross(int(dn[-1])) - cross(int(up[0]))

    lo, hi = 1e-3, 10.0
    if width(hi) < ROAD_W:
        raise SystemExit(f"R {radius:g}: no temperature makes a {ROAD_W:g} m band (at T {hi:g} "
                         f"it is {width(hi):.2f} m)")
    while hi / lo > 1.0 + TW_TOL:
        mid = float(np.sqrt(lo * hi))
        lo, hi = (mid, hi) if width(mid) < ROAD_W else (lo, mid)
    return hi


# ---------------------------------------------------------------- MMA


class Mma:
    """Svanberg's MMA with one constraint (module docstring), its asymptotes the instance's state:
    the subproblem's x(mu) in closed form per variable, mu bisected on the true D
    (road_simp.least_nu)."""

    def __init__(self, n: int) -> None:
        self.x1: F64 | None = None
        self.x2: F64 | None = None
        self.L = np.zeros(n)
        self.U = np.ones(n)

    def step(self, x: F64, g0: F64, g1: F64, *, free: B, D_of: Any, budget: float,
             mu0: float) -> tuple[F64, float, float]:
        """(the new x, mu, the share of the free variables at their move limit)."""
        if self.x1 is None or self.x2 is None:
            L, U = x - ASYM_INIT, x + ASYM_INIT
        else:
            turn = (x - self.x1) * (self.x1 - self.x2)
            gam = np.where(turn < 0, ASYM_SHRINK, np.where(turn > 0, ASYM_GROW, 1.0))
            L, U = x - gam * (self.x1 - self.L), x + gam * (self.U - self.x1)
        L = np.clip(L, x - ASYM_MAX, x - ASYM_MIN)
        U = np.clip(U, x + ASYM_MIN, x + ASYM_MAX)
        self.x2, self.x1, self.L, self.U = self.x1, x.copy(), L, U
        lo = np.maximum.reduce([np.zeros_like(x), L + 0.1 * (x - L), x - MOVE])
        hi = np.minimum.reduce([np.ones_like(x), U - 0.1 * (U - x), x + MOVE])

        def pq(g: F64) -> tuple[F64, F64]:
            r = RAA * max(float(np.abs(g[free]).max(initial=0.0)), 1e-300)
            gp, gm = np.maximum(g, 0.0), np.maximum(-g, 0.0)
            return ((U - x) ** 2 * (1.001 * gp + 0.001 * gm + r),
                    (x - L) ** 2 * (0.001 * gp + 1.001 * gm + r))

        p0, q0 = pq(g0)
        p1, q1 = pq(g1)

        def trial(mu: float) -> F64:
            a, b = np.sqrt(p0 + mu * p1), np.sqrt(q0 + mu * q1)
            return np.where(free, np.clip((L * a + U * b) / (a + b), lo, hi), x)

        xn, mu = rs.least_nu(trial, D_of=D_of, budget=budget, nu0=mu0)
        at = np.abs(xn - x) >= MOVE - 1e-9
        return xn, mu, float(at[free].mean())


# ---------------------------------------------------------------- the problem


class Iterate(NamedTuple):
    """One update's row (`J` = P + the price's cost; `grey` the share of the design pixels with
    0.05 < rho < 0.95; `at_limit` the share of the free variables MMA moved by MOVE; `labels`
    the pixels each label holds, argmax, joined)."""
    T: float
    beta: float
    q: float
    it: int
    P: float
    L_P: float
    J: float
    C: float
    D: float
    A: float
    mu: float
    change: float
    at_limit: float
    grey: float
    labels: str
    t: float


class Partition:
    """The block's label-field problem (module docstring) on road_simp's eps world."""

    def __init__(self, rd: rs.Roads, *, K: int, T_w: float) -> None:
        self.rd, self.K, self.T_w = rd, K, T_w
        self.lab = Labels(rd.raster, K=K)
        self.y = np.zeros(rd.m.n)               # no clearing
        self.npx = len(rd.raster.on)

    def world(self, f: F64, T: float, beta: float, q: float) -> tuple[rs.World, F64, F64]:
        """(the eps world, phi, rho's raw value) at the fields f (K, pixels)."""
        rho, phi, raw = self.lab.rho(f, T)
        r = relax.project(rho, beta)[self.rd.raster.pix]
        return self.rd.world_of(rho, r, self.y, q, beta), phi, raw

    def back_r(self, w: rs.World, g_r: F64) -> F64:
        """A per-triangle derivative in r to the pixels' rho (w.xf is rho)."""
        g = np.bincount(self.rd.raster.pix, weights=g_r, minlength=self.npx)
        return g * relax.dproject(w.xf, w.beta)

    def D_of(self, f: F64, T: float, beta: float) -> float:
        rho = self.lab.rho(f, T)[0]
        return self.rd.D(relax.project(rho, beta)[self.rd.raster.pix], self.rd.held)

    def derivs(self, f: F64, T: float, beta: float, q: float,
               price: rs.SheetPrice) -> tuple[rs.World, float, float, F64, F64]:
        """(the world, P, the price's cost, d(P + cost)/df, dD/df), the derivatives (K,
        pixels)."""
        w, phi, raw = self.world(f, T, beta, q)
        P, gs, _ = self.rd.P_grad(w)
        ch = price.charge(self.rd, w, rs.OC())          # (the sheet price reads no update)
        g_r = gs * q * np.maximum(w.r, 1e-300) ** (q - 1.0) + ch.d_r
        g0 = self.lab.back(self.back_r(w, g_r), phi, raw, T)
        g1 = self.lab.back(self.back_r(w, self.rd.dD(w)[0]), phi, raw, T)
        return w, P, ch.cost, g0, g1


def simp(pt: Partition, f: F64, plan: Plan, *, log: Any) -> tuple[F64, list[Iterate]]:
    """The plan's MMA from the start's fields f (K, pixels)."""
    K, npx = pt.K, pt.npx
    free = np.tile(pt.rd.raster.on, K)
    mma = Mma(K * npx)
    rows: list[Iterate] = []
    mu = 0.0
    t0 = time.time()
    P0 = pt.rd.P0
    for tm, beta, q in plan.stages:
        T = tm * pt.T_w

        def D_of(z: F64, T: float = T, beta: float = beta) -> float:
            return pt.D_of(z.reshape(K, npx), T, beta)

        for it in range(plan.iters):
            w, P, cost, g0, g1 = pt.derivs(f, T, beta, q, plan.price)
            z, mu, at = mma.step(f.ravel(), g0.ravel() / P0, g1.ravel(), free=free, D_of=D_of,
                                 budget=plan.D, mu0=mu)
            fn = z.reshape(K, npx)
            change = float(np.abs(fn - f)[:, pt.rd.raster.on].max())
            f = fn
            rho = w.xf[pt.rd.raster.on]
            lab = np.bincount(np.argmax(pt.lab.filter(f), axis=0)[pt.rd.raster.on], minlength=K)
            rows.append(Iterate(T=T, beta=beta, q=q, it=it, P=P, L_P=1.0 - P / P0,
                                J=(P + cost) / P0, C=pt.rd.C(w), D=pt.rd.D(w.r, w.c),
                                A=float(pt.rd.area @ w.r), mu=mu, change=change, at_limit=at,
                                grey=float(((rho > 0.05) & (rho < 0.95)).mean()),
                                labels=",".join(str(int(v)) for v in lab), t=time.time() - t0))
            r = rows[-1]
            log(f"T {T:.3f} beta {beta:g} q {q:g} it {it}: L_P {r.L_P:.4f}, J {r.J:.4f}, D "
                f"{r.D:.4f}, C {r.C:.0f} m^2, paved {r.A:.0f} m^2, mu {mu:.3g}, change "
                f"{change:.3f}, at limit {at:.3f}, grey {r.grey:.4f}, labels {r.labels} "
                f"({r.t:.0f}s)")
    return f, rows


# ---------------------------------------------------------------- answers and measures


def cut_lines(pt: Partition, f: F64, T: float) -> Any:
    """The label boundaries (module docstring), merged and simplified (local frame)."""
    ra = pt.rd.raster
    lab = pt.lab.argmax(f, T).reshape(ra.shape)
    segs = []
    i, j = np.nonzero(lab[1:, :] != lab[:-1, :])           # an edge at x = i + 1
    segs.append(np.stack([np.stack([i + 1, j], 1), np.stack([i + 1, j + 1], 1)], 1))
    i, j = np.nonzero(lab[:, 1:] != lab[:, :-1])           # an edge at y = j + 1
    segs.append(np.stack([np.stack([i, j + 1], 1), np.stack([i + 1, j + 1], 1)], 1))
    xy = ra.lo + PIXEL * np.concatenate(segs).astype(np.float64)
    merged = shapely.line_merge(shapely.multilinestrings(shapely.linestrings(xy)))
    return shapely.simplify(merged, SNAP_TOL)


def cut_of(b: Any, pt: Partition, f: F64, T: float) -> tuple[Any, Any]:
    """(the cut answer's lines, its ground), world frame, clipped to the block."""
    cut = shapely.intersection(pt.rd.m.geo.utm(cut_lines(pt, f, T)), b.boundary)
    cut = shapely.line_merge(shapely.union_all(shapely.get_parts(cut)))
    ground = shapely.intersection(
        road_corridor(gpd.GeoDataFrame(geometry=list(shapely.get_parts(cut)), crs=b.crs)
                      .assign(width_m=ROAD_W)), b.boundary)
    return cut, ground


def depth_quantiles(u: F64, area: F64) -> F64:
    """Each home's depth: the area-weighted share of homes with a shorter escape time (nan with no
    trip of its own): atlas/roads5810/depth5810's definition."""
    ok = np.isfinite(u)
    o = np.argsort(u[ok])
    cw = np.cumsum(area[ok][o])
    q = np.full(len(u), np.nan)
    q[np.flatnonzero(ok)[o]] = (cw - 0.5 * area[ok][o]) / cw[-1]
    return q


def skeleton(b: Any, ground: Any) -> tuple[Any, int, F64]:
    """The ground's (world frame) Zhang-Suen skeleton on a PIXEL raster: (its lines, its dead ends:
    the ends farther than TIP_STREET_M from the street whose branch reaches ROAD_W, each skeleton
    pixel's distance to the street)."""
    x0, y0, x1, y1 = shapely.bounds(ground)
    nx, ny = int(np.ceil((x1 - x0) / PIXEL)) + 2, int(np.ceil((y1 - y0) / PIXEL)) + 2
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cx, cy = x0 - PIXEL + PIXEL * (i + 0.5), y0 - PIXEL + PIXEL * (j + 0.5)
    shapely.prepare(ground)
    mask = shapely.contains_xy(ground, cx, cy)
    sk = skeletonize(mask)
    deg = ndimage.correlate(sk.astype(np.int64), np.ones((3, 3), dtype=np.int64),
                            mode="constant")
    street = shapely.union_all(b.streets.geometry.to_numpy())
    si, sj = np.nonzero(sk)
    d = shapely.distance(shapely.points(cx[si, sj], cy[si, sj]), street)
    keep = set(zip(si.tolist(), sj.tolist(), strict=True))
    dist = dict(zip(zip(si.tolist(), sj.tolist(), strict=True), d.tolist(), strict=True))
    ends = sum(1 for a, c in keep if deg[a, c] == 2 and dist[(a, c)] > TIP_STREET_M
               and _branch(keep, deg, (a, c)) >= ROAD_W / PIXEL)
    segs = [((a, c), (a + da, c + dc)) for a, c in keep
            for da, dc in ((0, 1), (1, -1), (1, 0), (1, 1)) if (a + da, c + dc) in keep]
    xy = np.array([[x0 - PIXEL, y0 - PIXEL]]) + PIXEL * (np.asarray(segs, dtype=np.float64)
                                                          .reshape(-1, 2, 2) + 0.5)
    lines = shapely.line_merge(shapely.multilinestrings(shapely.linestrings(xy)))
    return lines, int(ends), np.asarray(d, dtype=np.float64)


def _branch(keep: set[tuple[int, int]], deg: I64, end: tuple[int, int]) -> int:
    """The skeleton pixels from `end` along its branch to the first junction (a pixel with more
    than two neighbours), at most ROAD_W / PIXEL + 1: a branch shorter than a road's width is
    the thinning's stub at a corner, not a dead end."""
    prev, cur, n = None, end, 0
    while n <= ROAD_W / PIXEL:
        nxt = [(cur[0] + da, cur[1] + dc) for da in (-1, 0, 1) for dc in (-1, 0, 1)
               if (da or dc) and (cur[0] + da, cur[1] + dc) in keep
               and (cur[0] + da, cur[1] + dc) != prev]
        if len(nxt) != 1 or deg[nxt[0]] > 3:
            return n + 1
        prev, cur, n = cur, nxt[0], n + 1
    return n


def measures(b: Any, m: gcl.Model, lens: gcl.Lenses, ground: Any, lines: Any) -> dict[str, Any]:
    """An answer's network measures (module docstring): `ground` its carved ground and `lines`
    its centrelines (world frame), `m` and `lens` its scored model and state's lenses."""
    faces = shapely.get_parts(shapely.difference(b.boundary, ground))
    fa = np.sort(shapely.area(faces)[shapely.area(faces) >= FACE_M2])[::-1]
    sk_lines, ends, d = skeleton(b, ground)
    street = shapely.union_all(b.streets.geometry.to_numpy())
    parts = shapely.get_parts(lines)
    floating = parts[~rs.linked(parts, street)] if len(parts) else parts
    outlines = np.asarray(b.buildings.outlines)
    u0, u = m.base.u, lens.u
    q = depth_quantiles(u0, shapely.area(outlines))
    deep = q >= DEEP_Q
    near = shapely.distance(outlines, ground) <= REACH_M
    ok = deep & np.isfinite(u) & np.isfinite(u0)
    net = gpd.GeoDataFrame(geometry=list(parts), crs=b.crs).assign(width_m=ROAD_W)
    dm = damage.damage(damage.Graph.of(b, net, step_m=DAMAGE_STEP_M, carry_m=DAMAGE_CARRY_M))
    return dict(faces=int(len(fa)), face_largest_m2=float(fa[0]) if len(fa) else 0.0,
                face_median_m2=float(np.median(fa)) if len(fa) else 0.0,
                face_p90_m2=float(np.quantile(fa, 0.9)) if len(fa) else 0.0,
                road_m=float(shapely.length(parts).sum()),
                floating_m=float(shapely.length(floating).sum()),
                skeleton_m=float(shapely.length(sk_lines)), dead_ends=ends,
                depth_max=float(d.max(initial=0.0)),
                homes_within50=float(near[np.isfinite(q)].mean()),
                deep_within50=float(near[deep].mean()),
                L_P_deep=1.0 - float(m.pop[ok] @ u[ok]) / float(m.pop[ok] @ u0[ok]),
                damage_cut=dm.cut, damage_detour=dm.detour)


def score_answer(b: Any, metric: gcl.ComplexMetric, ground: Any,
                 lines: Any) -> dict[str, Any]:
    """An answer (world frame) scored (road_simp.scored, no clearing) with its measures."""
    m, lens, row = rs.scored(b, metric, ground, cleared=np.zeros(len(b.buildings.outlines),
                                                                    dtype=bool))
    return row | measures(b, m, lens, ground, lines)


# ---------------------------------------------------------------- the run


def partition_of(b: Any, metric: gcl.ComplexMetric, plan: Plan) -> Partition:
    m = metric.clearing(b, demand=common.PopulationDemand())
    none = np.zeros(m.n, dtype=bool)
    rd = rs.Roads(m, radius=plan.radius, cleared=none, real=False, free=none)
    return Partition(rd, K=plan.K, T_w=tw(plan.radius, plan.K))


def fields_of(pt: Partition, start: Start) -> tuple[F64, int]:
    """The start's fields (K, pixels): each pixel's label 1, the rest 0; and its cells' count."""
    cells = start.cells(pt.rd.raster, pt.rd.m)
    lab = colour(cells, pt.rd.raster.shape, pt.K)[cells]
    f = np.zeros((pt.K, pt.npx))
    f[lab, np.arange(pt.npx)] = 1.0
    return np.where(pt.rd.raster.on, f, 0.0), int(cells.max()) + 1


def start(bid: str, plan: Plan) -> None:
    """The start's eps-world D, C and paved area at each stage's (T, beta), no solve."""
    metric = metrics.metric_of(metrics.RUNTIME, device="cpu")
    [b] = common.build_blocks([bid])
    pt = partition_of(b, metric, plan)
    f, n = fields_of(pt, plan.start)
    print(f"{bid} {plan.start.name}: {n} cells, T_w {pt.T_w:.4f}", flush=True)
    for tm, beta, q in plan.stages:
        w = pt.world(f, tm * pt.T_w, beta, q)[0]
        print(f"  T {tm * pt.T_w:.3f} beta {beta:g} q {q:g}: D {pt.rd.D(w.r, w.c):.4f}, C "
              f"{pt.rd.C(w):.0f} m^2, paved {float(pt.rd.area @ w.r):.0f} m^2", flush=True)


def run(bid: str, plan: Plan, device: str, *, standing: clearings.Standing) -> None:
    metric = metrics.metric_of(metrics.RUNTIME, device=device)
    out = OUT / metrics.RUNTIME / plan.name / f"{bid}.parquet"
    if not standing.todo(out):
        return
    t0 = time.time()
    [b] = common.build_blocks([bid])
    pt = partition_of(b, metric, plan)
    f, n = fields_of(pt, plan.start)
    print(f"{bid}: {pt.rd.m.fx.fm.nt} triangles, {int(pt.rd.raster.on.sum())} design pixels, "
          f"K {plan.K}, T_w {pt.T_w:.4f}, start {plan.start.name}: {n} cells "
          f"({time.time() - t0:.0f}s)", flush=True)
    start_cut = cut_of(b, pt, f, plan.stages[-1][0] * pt.T_w)
    f, rows = simp(pt, f, plan, log=lambda s: print(s, flush=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    on = pt.rd.raster.on
    pd.DataFrame({f"psi{k}": f[k, on].astype(np.float32) for k in range(plan.K)}).to_parquet(
        out.with_name(f"{bid}_psi.parquet"))
    tm, beta, q = plan.stages[-1]
    T = tm * pt.T_w
    w = pt.world(f, T, beta, q)[0]
    geo = pt.rd.m.geo
    band = geo.utm(rs.paved_of(pt.rd, w.r))
    band_lines = skeleton(b, band)[0] if not band.is_empty else shapely.MultiLineString()
    cut, cut_ground = cut_of(b, pt, f, T)
    del pt
    sc: dict[str, Any] = {}
    for name, ground, lines in (("start", start_cut[1], start_cut[0]), ("band", band, band_lines),
                                ("cut", cut_ground, cut)):
        gpd.GeoDataFrame(geometry=[ground], crs=b.crs).to_parquet(
            out.with_name(f"{bid}_{name}_ground.parquet"))
        gpd.GeoDataFrame(geometry=list(shapely.get_parts(lines)), crs=b.crs).to_parquet(
            out.with_name(f"{bid}_{name}_lines.parquet"))
        if ground.is_empty:
            print(f"{name}: empty", flush=True)
            continue
        s = score_answer(b, metric, ground, lines)
        sc |= {f"{name}_{k}": v for k, v in s.items()}
        print(f"{bid} {plan.name} {name}: L_P {s['L_P']:.4f}, L_J2 {s['L_J2']:.4f}, D "
              f"{s['D_exact']:.4f}, C {s['C']:.0f} m^2, {s['faces']} sub-blocks (largest "
              f"{s['face_largest_m2']:.0f} m^2), road {s['road_m']:.0f} m, dead ends "
              f"{s['dead_ends']}, deep within 50 m {s['deep_within50']:.2f}, L_P deep "
              f"{s['L_P_deep']:.4f}, cut {s['damage_cut']:.4f} ({time.time() - t0:.0f}s)",
              flush=True)
    df = pd.DataFrame(rows).assign(block=bid, plan=plan.name, **sc, device=device,
                                   host=socket.gethostname(), peak_rss_gb=gap_calib.peak_gb(),
                                   t_total=time.time() - t0)
    standing.write(df, out)


def main(argv: list[str]) -> None:
    argv, standing = clearings.standing_of(argv)
    if len(argv) == 4 and argv[0] == "run" and argv[3] in ("cpu", "gpu"):
        run(argv[1], Plan.of(argv[2]), argv[3], standing=standing)
    elif len(argv) == 3 and argv[0] == "start":
        start(argv[1], Plan.of(argv[2]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
