"""Reblocking top-down: a multi-phase partition SIMP on the block's fixed mesh (owner 2026-10-27:
"methods for literal reblocking, where we partition the region (so every road is a loop)";
.superpowers/sdd/2026-10-27-partition/brief.md, part 1).

The design is K label fields psi_k per design pixel (road_simp's raster and cone filter of radius
R), phi = the softmax of the filtered fields at temperature T. The roads are where the labels
change, read from a field `mix` per label as g = 1 - sum_k mix_k^2, a road density rho = (g -
g_pure) / (g_pair - g_pure) clipped to [0, 1], g_pure its value on a pure label (one field 1, the
rest 0: the floor of g over the design's box) and g_pair at the middle of a sharp two-label
interface: the interior of a label reads 0 at any T, an interface's middle 1, so the projection's
threshold sits between them. `mix` is a Strategy (Band, the plan's `.band` token or none):
  Mixed (rev 2's, no token): phi itself, the road where the labels mix (the raw 1 - sum phi^2
    leaves 0.08 -- 0.7 of road density inside a label at the temperatures that make a 7 m road,
    hence the normalization). Its band rho >= 1/2 across a sharp step in the design widens with T
    and saturates near 0.8 R; a stage's T is the plan's t times T_w (`tw`: the T at which a sharp
    straight interface's band is ROAD_W wide, by bisection on a synthetic step through the plan's
    own filter), so the continuation runs toward ROAD_W from softer, wider interfaces. Two fields
    tied inside one label read as road too: a ridge, a road with the same label on both sides,
    which can end anywhere (NOTES, "Reblocking top-down": the rev 2 band answers' 4 -- 42 dead
    ends, and most of their gain over the cut).
  Banded (rev 3, `.band`; owner 2026-10-28, the design -> filtered -> projected scheme): phi
    nearly one-hot (T continued sharp: T is the plan's t), then filtered by a second cone (the
    band's, `band_radius`: a sharp straight interface's band rho >= 1/2 is ROAD_W wide, rho = 4 a
    (1 - a) with a the band's share of one side), so the road is a band of fixed width around the
    sharp labels' boundaries: its width comes from the band's cone, not from T, and a tie inside
    a label is a band of width ~T R, which the band's cone averages away (partition_check's
    `ridges`).
Then road_simp's eps world: per triangle r = proj(rho, beta) of its pixel, s = r^q, kt = kt0 + s
(1 - kt0), b = bw + s db (road_simp.Roads.world_of: kappa 10, the metric's own road); D the ramp
over the buildings' covered shares, C the sheet's conductance added (road_simp.Roads.D, .C). The
contrast is kappa 10 only: the grounded leak (growth.GroundedLeak) needs a System per state with
the leak on its graph, where road_simp.Roads.state reads the model's own, so a second state path;
not built. q and beta are continued as road_simp's.

No dead ends by construction: a label boundary closes, meets the street or meets others at a
junction. The 0/1 answers can still have them (a band thinned below the threshold breaks; Mixed's
ridges): they are counted on each answer's skeleton.

Objective: P + lambda C (road_simp.SheetPrice: lambda = price P0 / SHEET_M2) under D <= Plan.D.
Gradient: road_simp's adjoint (dP/ds per triangle, through kt and b), back through the
projection, the normalization, the band's cone (its transpose; Banded), the softmax (dphi_k /
dpsi~_j = phi_k (delta_kj - phi_j) / T) and the filter's transpose. The update is MMA (Svanberg
1987), one constraint: OC needs a gain of one sign, and a label field's has none (moving an
interface helps one side and hurts the other); the multiplier mu of D is found on the true D of
the trial update (the projection makes a linearised D overshoot): least_mu, bracketed by squaring
factors and closed by the Illinois rule (rev 3; rev 2 bisected it, road_simp.least_nu, 12 -- 16
evaluations an update once the budget binds and 300 at the first update of an over-budget start).

The ramp's tail (Partition.D, the plan's `.e` token, one eps per stage, the last 0; owner
2026-10-28): past the knee tau a building's covered share X counts min(1, X / tau) + eps (X - tau)
/ (1 - tau), so D's slope there is eps / (1 - tau), not 0: a start over its budget can shed D by
moving its interfaces off the buildings it covers (rev 2's nodal and Voronoi-10 starts never got
under, their multiplier at its ceiling throughout), and a building at the margin is traded. The
tail is in the D the budget holds during the continuation (D_relaxed in the rows, at least the
true D), and gone at the last stage; the rows' D and every answer's score use the true ramp.

The start is a Strategy (Start, the plan's last token): `vor<n>s<seed>`, the Voronoi cells of n
seeds placed by k-means on the homes' centroids; `spec<n>s<seed>`, the shelved spectral normalized
cut (memory "Voronoi-adjacency partition method": a gap-aware affinity exp(-gap^2 / 2 sigma^2)
between each building and its KNN nearest, sigma the median gap, the n smallest eigenvectors of the
normalized Laplacian row-normalized and k-means'd; each pixel takes its nearest building's cluster);
`nodal<n>s<seed>`, a Dirichlet spectral partition aiming at the spectral minimal partition (Helffer,
Hoffmann-Ostenhof, Terracini: the pieces' smallest first eigenvalue the largest, a piece's P at most
its demand over its lambda_1; Nodal); `div<n>p<tau>e<ear_m>`, the flow's own divides: the
persistence-simplified Morse-Smale cuts of u, their ends at least ear_m apart (the ears arm's
generator, imported: ears.divides) until about n pieces (Divides); `ears`, the ears' `paths` answer
at the plan's budget, its faces the cells (Ears: EARS_RUN's frame there, as the 5810 page reads it).
Each cell then coloured into the K labels (DSatur on the cells' adjacency: K labels make any number
of sub-blocks, four colouring every planar partition, DSatur not always finding it). Each start is
scored as its own cut answer before the update moves it (`start_*` in the row).

Answers, each scored as a road network (road_simp.scored: its own carved mesh, RUNTIME), with the
measures `measures` gives (sub-blocks, road length, dead ends, the depth measures, the damage
term's cut share):
  start: the start's own cut answer (below), before the update;
  band: the triangles at r >= 1/2 (road_simp.paved_of);
  cut: the label boundaries (argmax phi on the raster, each region narrower than a road merged
    into a neighbour, `thick`; each pixel edge between two labels, merged and simplified) as roads
    ROAD_W wide (road_corridor), clipped to the block.
Each answer's ground is snapped to a 1 mm grid in the block's local frame (`gridded`) before it
is stored and scored.

Speed (5810: 7.2M triangles, 6.8M cells, 519k design pixels of 2.2M, K 5; this machine, one
thread): an update is the factorization (SuperLU 23 s, cuDSS 0.1 s after its 36 s plan), the
refined solve (2 s; 0.4 s), the chain and its way back (FFT filters, the kernels' transforms
computed once: 0.2 s a K-field filter), and the search for mu (a trial and a D, 0.3 s each); on
`gpu` (the run's device: gap_factor.CuDSS) the factorization drops out.

    P=research/roadless/partition.py                   # from the repo root, PYTHONPATH=.
    uv run python $P run <block> <plan> cpu|gpu [--replace]
    uv run python $P start <block> <plan>              # the start's D, C and cells, no solve

plan: pt.K<K>.R<radius>[.band].t<t,...>.b<beta,...>.q<q,...>[.e<eps,...>].i<iters>.D<D>
.s<price>.<start> (the stages pair t, beta, q and eps in order; no `.e`, the true ramp
throughout), e.g. pt.K5.R10.band.t0.2,0.1,0.05,0.03,0.02.b1,2,4,8,16.q1,2,3,3,3
.e0.2,0.1,0.05,0.02,0.i40.D0.1.s0.1.spec40s0; its name adds the raster's pixel and the eps
world's eps and the code's revision (.p<PIXEL>e<EPS>v<REV>). Rows in
partition_rows/<metric>/<name>/<block>.parquet (the iterates and the answers' scores), beside the
design (<block>_psi.parquet, the fields on the design pixels), the answers' grounds and lines
(<block>_<answer>_ground.parquet, _lines.parquet).
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
import scipy.sparse.linalg as spla
import shapely
from numpy.typing import NDArray
from scipy import fft as sp_fft
from scipy import ndimage
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree
from skimage.measure import label

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import common  # noqa: E402
import damage  # noqa: E402
import ears  # noqa: E402
import gap_calib  # noqa: E402
import gap_clear as gcl  # noqa: E402
import growth as gr  # noqa: E402
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
MU_TOL = 1e-4           # least_mu: D within this of the budget (relative)
MU_MIN, MU_MAX = 1e-300, 1e300       # least_mu: mu's range
TW_TOL = 1e-4            # Plan.tw: T_w bisected to this, relative
TW_N = 200               # Plan.tw: the synthetic step's raster (pixels a side)
KNN = 8                  # Spectral: each building's nearest neighbours (by centroid)
MIN_REGION_M2 = 400.0    # m^2: a start's region under this is merged into a neighbour (merged)
NODAL_FLOOR = 0.01       # Nodal: the mass's share spread by area (the open ground's cells
                         # hold no demand: M positive definite)
KMEANS_ITER = 50         # Voronoi, Spectral: k-means iterations
SNAP_TOL = 1.0           # m: the cut answer's boundaries simplified to this
GRID_M = 1e-3            # m: every answer's ground snapped to this grid in the block's local
                         # frame before it is stored and scored (`gridded`: the 5810 page's
                         # networks.GRID_M)
FACE_M2 = 100.0          # m^2: a sub-block is a face of the block less the roads this large
TIP_STREET_M = 5.0       # m: a skeleton end farther than this from the street is a dead end
REACH_M = 50.0           # m: a home "reached": within this of a road (the growth pilot's)
DEEP_Q = 0.75            # the deepest quarter: homes at depth quantile >= this
DAMAGE_STEP_M = 10.0     # the damage term's piece length (NOTES, "Roads from the vote, stage 2")
DAMAGE_CARRY_M = 100.0   # the damage term's carry
REV = 4                  # 2: the start's cells split into connected regions before colouring,
                         # those under MIN_REGION_M2 merged (`merged`); 3: the banded road
                         # (`.band`), the ramp's tail (`.e`), the ears start, mu by least_mu;
                         # 4: the answers' grounds on the GRID_M grid (`gridded`), the cut's
                         # regions narrower than a road merged (`thick`): the update is rev 3's
TAU = common.MEASURE.tau                 # the ramp's knee (the ramp's tail, Partition.D)
EARS_RUN = "ears.paths.b20.gl0.25Croad.n2s0.1.M8.K4.D0.15.v1"   # the ears start's run
                         # (growth_rows/<block>/: the ears arm's paths on 5810)
OUT = HERE / "partition_rows"


# ---------------------------------------------------------------- the starts


class Start(Protocol):
    """The starting partition (module docstring; the plan's last token): per design pixel a cell,
    coloured into labels by `colour`."""

    @property
    def name(self) -> str: ...

    def cells(self, b: Any, ra: rs.Raster, m: gcl.Model) -> I64:
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

    def cells(self, b: Any, ra: rs.Raster, m: gcl.Model) -> I64:
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

    def cells(self, b: Any, ra: rs.Raster, m: gcl.Model) -> I64:
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

    def cells(self, b: Any, ra: rs.Raster, m: gcl.Model) -> I64:
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


@dataclass(frozen=True, kw_only=True)
class Divides:
    """The flow's own divides (module docstring): ears.divides on the whole block as one face
    (ears.EarWorld's PIX raster, pixel graph and sinks, the street's faces, from a growth.Growth
    of the block), u the eps world's baseline potential under the population's demand (its
    set_u), at persistence threshold `tau`;
    its cuts (their ends at least `ear_m` apart) added most persistent first until the block
    falls into at least n regions; each pixel of this raster its nearest coarse pixel's region (a
    cut pixel its nearest region's). The rev 2 div runs (`div<n>p<tau>`, before ears rev 3 made
    the floor a parameter) had ear_m 30."""
    n: int
    tau: float
    ear_m: float

    @property
    def name(self) -> str:
        return f"div{self.n}p{self.tau:g}e{self.ear_m:g}"

    def cells(self, b: Any, ra: rs.Raster, m: gcl.Model) -> I64:
        g = gr.Growth(m, contrast=gr.Road(kappa=gr.ROAD_KAPPA))
        e = ears.EarWorld(g)
        er = e.ra
        e.set_u(g.solve(g.state(np.zeros(g.nt, dtype=bool)))[0])
        face = np.where(er.tri >= 0, 0, -1)
        cuts = ears.divides(e.u_pix, face, e.sinks(), er.ea, er.eb, er.xy, er.bld, face=0,
                            tau=self.tau, stretch_m=ears.STRETCH_M, ear_m=self.ear_m)
        inside = (face == 0).reshape(er.shape)
        cut = np.zeros(er.shape, dtype=bool)
        reg = np.zeros(er.shape, dtype=np.int64)
        for d in cuts:
            cut.ravel()[d.pixels] = True
            reg = np.asarray(label(inside & ~cut, background=0, connectivity=1), dtype=np.int64)
            if reg.max() >= self.n:
                break
        if not cuts:
            reg = inside.astype(np.int64)
        _, near = ndimage.distance_transform_edt(reg == 0, return_indices=True)
        reg = reg[near[0], near[1]] - 1
        ij = np.floor((_pixel_xy(ra) - er.lo) / ears.PIX).astype(np.int64)
        ij = np.clip(ij, 0, np.array(er.shape) - 1)
        return np.asarray(reg[ij[:, 0], ij[:, 1]], dtype=np.int64)


@dataclass(frozen=True, kw_only=True)
class Ears:
    """The ears' `paths` answer (the bottom-up arm, EARS_RUN) at the plan's budget: the frame the
    5810 page shows there (atlas/roads5810/networks.Grown's reading: the shortest prefix whose
    exact D reaches it, the whole run where none does), its faces the cells (the pixels whose
    centres lie off its region, 4-connected; a road pixel its nearest face's)."""
    budget: float
    name = "ears"

    def cells(self, b: Any, ra: rs.Raster, m: gcl.Model) -> I64:
        sys.path.insert(0, str(HERE / "atlas" / "roads5810"))
        import networks  # type: ignore[import-not-found]  # noqa: PLC0415  (the page's reader)
        net = networks.Grown(b, gr.OUT / b.block_id / EARS_RUN, targets=(self.budget,))
        s = net.prefix_len(self.budget)
        reg = net.region(s)
        print(f"ears: prefix {s} of {net.size} at D {self.budget:g} (its region's D "
              f"{net.D_region(s):.4f})", flush=True)
        xy = _pixel_xy(ra)
        shapely.prepare(reg)
        road = shapely.contains_xy(reg, xy[:, 0], xy[:, 1]).reshape(ra.shape)
        faces = np.asarray(label(ra.on.reshape(ra.shape) & ~road, background=0, connectivity=1),
                           dtype=np.int64)
        _, near = ndimage.distance_transform_edt(faces == 0, return_indices=True)
        return np.asarray(faces[near[0], near[1]].ravel() - 1, dtype=np.int64)


def start_of(token: str, *, budget: float) -> Start:
    m = re.fullmatch(r"(vor|spec|nodal)([0-9]+)s([0-9]+)", token)
    d = re.fullmatch(r"div([0-9]+)p([0-9.]+)e([0-9.]+)", token)
    if token == "ears":
        return Ears(budget=budget)
    if d is not None:
        return Divides(n=int(d[1]), tau=float(d[2]), ear_m=float(d[3]))
    if m is None:
        raise SystemExit(f"{token}: a start is vor<n>s<seed>, spec<n>s<seed>, nodal<n>s<seed>, "
                         "div<n>p<tau>e<ear_m> or ears")
    n, seed = int(m[2]), int(m[3])
    kinds: dict[str, type[Voronoi] | type[Spectral] | type[Nodal]] = {
        "vor": Voronoi, "spec": Spectral, "nodal": Nodal}
    return kinds[m[1]](n=n, seed=seed)


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


class Stage(NamedTuple):
    """A stage of the continuation: the plan's t (the band's temperature: Mixed's T / T_w,
    Banded's T), beta, q and the ramp's tail (`eps`, Partition.D)."""
    t: float
    beta: float
    q: float
    eps: float


class BandOf(Protocol):
    """The plan's road reading (module docstring; `.band` in the name, or none), built once on
    the block's raster."""

    @property
    def name(self) -> str: ...

    def of(self, ra: rs.Raster, *, radius: float, K: int) -> Band: ...


@dataclass(frozen=True)
class MixedOf:
    name = ""

    def of(self, ra: rs.Raster, *, radius: float, K: int) -> Band:
        return Mixed(T_w=tw(radius, K))


@dataclass(frozen=True)
class BandedOf:
    name = "band"

    def of(self, ra: rs.Raster, *, radius: float, K: int) -> Band:
        return Banded(ra, radius=band_radius())


@dataclass(frozen=True, kw_only=True)
class Plan:
    """A run's settings, all in its name (module docstring)."""
    K: int
    radius: float
    band: BandOf
    stages: tuple[Stage, ...]
    iters: int
    D: float
    price: rs.SheetPrice
    start: Start

    def __post_init__(self) -> None:
        if not (self.K >= 2 and self.radius > 0 and self.stages and self.iters > 0
                and self.D > 0):
            raise ValueError(f"{self}: K at least 2; a radius, stages, updates and a budget, "
                             "each positive")
        if any(s.eps < 0 for s in self.stages) or self.stages[-1].eps != 0:
            raise ValueError(f"{self}: the ramp's tail non-negative, 0 at the last stage (the "
                             "true ramp)")

    @property
    def name(self) -> str:
        def joined(k: int) -> str:
            return ",".join(f"{s[k]:g}" for s in self.stages)
        band = f".{self.band.name}" if self.band.name else ""
        eps = f".e{joined(3)}" if any(s.eps for s in self.stages) else ""
        return (f"pt.K{self.K}.R{self.radius:g}{band}.t{joined(0)}.b{joined(1)}.q{joined(2)}{eps}"
                f".i{self.iters}.D{self.D:g}.{self.price.name}.{self.start.name}"
                f".p{PIXEL:g}e{EPS:g}v{REV}")

    @staticmethod
    def of(spec: str) -> Plan:
        m = re.fullmatch(r"pt\.K([0-9]+)\.R([0-9.]+)(\.band)?\.t([0-9.,]+)\.b([0-9.,]+)"
                         r"\.q([0-9.,]+)(?:\.e([0-9.,]+))?\.i([0-9]+)\.D([0-9.e-]+)\.s([0-9.e-]+)"
                         r"\.([a-z]+[0-9]+s[0-9]+|div[0-9]+p[0-9.]+|ears)", spec)
        if m is None:
            raise SystemExit(f"{spec}: a plan is pt.K<K>.R<radius>[.band].t<t,..>.b<beta,..>"
                             ".q<q,..>[.e<eps,..>].i<iters>.D<D>.s<price>.<start>")
        ts, bs, qs = ([float(v) for v in m[k].split(",")] for k in (4, 5, 6))
        es = [float(v) for v in m[7].split(",")] if m[7] else [0.0] * len(ts)
        if not len(ts) == len(bs) == len(qs) == len(es):
            raise SystemExit(f"{spec}: as many t as beta as q (as eps)")
        D = float(m[9])
        return Plan(K=int(m[1]), radius=float(m[2]), band=BandedOf() if m[3] else MixedOf(),
                    stages=tuple(Stage(t=t, beta=b, q=q, eps=e)
                                 for t, b, q, e in zip(ts, bs, qs, es, strict=True)),
                    iters=int(m[8]), D=D, price=rs.SheetPrice(price=float(m[10])),
                    start=start_of(m[11], budget=D))


# ---------------------------------------------------------------- the design's chain


class Conv:
    """A symmetric kernel's correlation (its convolution) over a raster by FFT, every field at
    once, the kernel's transform computed once: signal.fftconvolve's `same`, bit for bit (its
    padded size, next_fast_len of the full convolution's, and its centred crop)."""

    def __init__(self, shape: tuple[int, int], kernel: F64) -> None:
        self.shape = shape
        self.h = kernel.shape[0] // 2
        self.s = tuple(sp_fft.next_fast_len(n + 2 * self.h, True) for n in shape)
        self.kf = sp_fft.rfft2(kernel, self.s)

    def __call__(self, img: F64) -> F64:
        """img (m, pixels) -> (m, pixels)."""
        m = img.shape[0]
        x = img.reshape(m, *self.shape)
        y = sp_fft.irfft2(sp_fft.rfft2(x, self.s, axes=(1, 2)) * self.kf, self.s, axes=(1, 2))
        h, (a, b) = self.h, self.shape
        return np.ascontiguousarray(y[:, h:h + a, h:h + b]).reshape(m, -1)


class Filter:
    """A kernel's normalized filter over the design pixels (road_simp.Raster.filter's form: an off
    pixel reads 0 and is read as 0), every field at once, and its transpose."""

    def __init__(self, ra: rs.Raster, kernel: F64, norm: F64) -> None:
        self.on = ra.on
        self.conv = Conv(ra.shape, kernel)
        self.den = np.where(ra.on, norm, 1.0)

    def __call__(self, f: F64) -> F64:
        return np.where(self.on, self.conv(np.where(self.on, f, 0.0)) / self.den, 0.0)

    def T(self, g: F64) -> F64:
        return np.where(self.on, self.conv(np.where(self.on, g / self.den, 0.0)), 0.0)


def cone(radius: float) -> F64:
    """road_simp.Raster.of's cone kernel of `radius` m on the PIXEL grid."""
    r = radius / PIXEL
    k = int(np.ceil(r))
    d = np.hypot(*np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1), indexing="ij"))
    return np.maximum(0.0, r - d)


def softmax(L: F64) -> F64:
    phi = np.exp(L - L.max(axis=0))
    phi /= phi.sum(axis=0)
    return phi


class Band(Protocol):
    """How the labels' softmax phi makes the field the road is read from (`mix`, per label and
    design pixel), the road density rho = (g - g_pure) / (g_pair - g_pure) with g = 1 - sum_k
    mix_k^2 (`ends`: g on a pure label and at the middle of a sharp two-label interface, at T),
    and a stage's temperature from the plan's t (module docstring; the plan's `.band` token)."""

    @property
    def name(self) -> str: ...

    def T(self, t: float) -> float: ...

    def ends(self, T: float, K: int) -> tuple[float, float]: ...

    def mix(self, phi: F64) -> F64:
        """phi (m, pixels) -> mix (m, pixels)."""
        ...

    def mix_T(self, g: F64) -> F64: ...


class Mixed:
    """Rev 2's road (module docstring): phi itself, the road where the labels mix; T = t T_w."""
    name = ""

    def __init__(self, *, T_w: float) -> None:
        self.T_w = T_w

    def T(self, t: float) -> float:
        return t * self.T_w

    def ends(self, T: float, K: int) -> tuple[float, float]:
        pure = np.zeros((K, 1))
        pure[0] = 1.0
        pair = np.zeros((K, 1))
        pair[:2] = 0.5
        return _g(softmax(pure / T)), _g(softmax(pair / T))

    def mix(self, phi: F64) -> F64:
        return phi

    def mix_T(self, g: F64) -> F64:
        return g


class Banded:
    """The road a band around the sharp labels' boundaries (module docstring): phi filtered by a
    cone of `radius` (`band_radius`: a sharp straight interface's band rho >= 1/2 is ROAD_W wide);
    T = t."""
    name = "band"

    def __init__(self, ra: rs.Raster, *, radius: float) -> None:
        self.radius = radius
        k = cone(radius)
        norm = ndimage.correlate(ra.on.reshape(ra.shape).astype(np.float64), k,
                                 mode="constant").ravel()
        self.filter = Filter(ra, k, norm)

    def T(self, t: float) -> float:
        return t

    def ends(self, T: float, K: int) -> tuple[float, float]:
        a = np.zeros((K, 1))
        a[0] = 1.0
        b = np.zeros((K, 1))
        b[1] = 1.0
        pa, pb = softmax(a / T), softmax(b / T)
        return _g(pa), _g(0.5 * (pa + pb))

    def mix(self, phi: F64) -> F64:
        return self.filter(phi)

    def mix_T(self, g: F64) -> F64:
        return self.filter.T(g)


def _g(mix: F64) -> float:
    return float(1.0 - (mix[:, 0] ** 2).sum())


class Chain(NamedTuple):
    """The chain's values at a design: per pixel rho and its raw value before the clip, per label
    and pixel phi and the field the road is read from (`mix`); off the design pixels all 0."""
    rho: F64
    raw: F64
    phi: F64
    mix: F64


class Labels:
    """The K label fields' chain on the raster (module docstring): fields (K, pixels) -> the
    filtered fields (the plan's cone) -> phi (softmax at T) -> the band's mix -> rho per pixel,
    and its way back. The per-pixel steps run on the design pixels alone (off them every value
    is 0: none is read)."""

    def __init__(self, ra: rs.Raster, *, K: int, band: Band) -> None:
        self.ra, self.K, self.band = ra, K, band
        self.cone = Filter(ra, ra.kernel, ra.norm)
        self.idx = np.flatnonzero(ra.on)

    def filter(self, f: F64) -> F64:
        """Raster.filter by FFT, every field at once."""
        return self.cone(f)

    def filter_T(self, g: F64) -> F64:
        return self.cone.T(g)

    def chain(self, f: F64, T: float) -> Chain:
        lo, hi = self.band.ends(T, self.K)
        n = len(self.ra.on)
        phi = np.zeros((self.K, n))
        phi[:, self.idx] = softmax(self.filter(f)[:, self.idx] / T)
        mix = self.band.mix(phi)
        raw = np.zeros(n)
        raw[self.idx] = ((1.0 - (mix[:, self.idx] ** 2).sum(axis=0)) - lo) / (hi - lo)
        return Chain(rho=np.clip(raw, 0.0, 1.0), raw=raw, phi=phi, mix=mix)

    def back(self, g_rho: F64, ch: Chain, T: float) -> F64:
        """Per-pixel derivatives in rho (m, pixels) back to the fields (m, K, pixels)."""
        lo, hi = self.band.ends(T, self.K)
        m, K = g_rho.shape[0], self.K
        live = (ch.raw > 0.0) & (ch.raw < 1.0)
        g_mix = (np.where(live, g_rho, 0.0)[:, None, :] * (-2.0 * ch.mix)[None] / (hi - lo))
        g_phi = self.band.mix_T(g_mix.reshape(m * K, -1)).reshape(m, K, -1)
        g_L = ch.phi[None] * (g_phi - (g_phi * ch.phi[None]).sum(axis=1, keepdims=True))
        return self.filter_T((g_L / T).reshape(m * K, -1)).reshape(m, K, -1)

    def argmax(self, f: F64) -> I64:
        """Per pixel its label (argmax phi), an off pixel its nearest on pixel's."""
        lab = np.argmax(self.filter(f), axis=0)
        on = self.ra.on.reshape(self.ra.shape)
        _, idx = ndimage.distance_transform_edt(~on, return_indices=True)
        return np.asarray(lab.reshape(self.ra.shape)[idx[0], idx[1]].ravel(), dtype=np.int64)


def _step_raster(radius: float) -> rs.Raster:
    """A TW_N x TW_N raster, every pixel on, the cone of `radius` (tw's and band_radius's)."""
    on = np.ones(TW_N * TW_N, dtype=bool)
    ra = rs.Raster.of(np.array([[0.5, 0.5], [TW_N - 0.5, TW_N - 0.5]]), radius=radius)
    return ra._replace(on=on, norm=ndimage.correlate(np.ones((TW_N, TW_N)), ra.kernel,
                                                     mode="constant").ravel())


def _width(rho: F64) -> float:
    """The band rho >= HALF across the synthetic step (a TW_N x TW_N raster, the step at its
    middle row), by linear interpolation of the profile."""
    row = rho.reshape(TW_N, TW_N)[:, TW_N // 2]
    x = np.arange(TW_N) + 0.5
    up = np.flatnonzero((row[:-1] < HALF) & (row[1:] >= HALF))
    dn = np.flatnonzero((row[:-1] >= HALF) & (row[1:] < HALF))
    if not len(up) or not len(dn):
        return 0.0

    def cross(i: int) -> float:
        return float(x[i] + (HALF - row[i]) / (row[i + 1] - row[i]))
    return cross(int(dn[-1])) - cross(int(up[0]))


def _step(K: int) -> F64:
    f = np.zeros((K, TW_N, TW_N))
    f[0, : TW_N // 2] = 1.0
    f[1, TW_N // 2:] = 1.0
    return f.reshape(K, -1)


def _bisect(width: Any, lo: float, hi: float, what: str) -> float:
    """The least x in [lo, hi] (geometric bisection to TW_TOL) whose width reaches ROAD_W
    (width rises with x)."""
    if width(hi) < ROAD_W:
        raise SystemExit(f"{what}: no value makes a {ROAD_W:g} m band (at {hi:g} it is "
                         f"{width(hi):.2f} m)")
    while hi / lo > 1.0 + TW_TOL:
        mid = float(np.sqrt(lo * hi))
        lo, hi = (mid, hi) if width(mid) < ROAD_W else (lo, mid)
    return hi


def tw(radius: float, K: int) -> float:
    """T_w (Mixed): the temperature at which a sharp straight interface's band rho >= 1/2 is
    ROAD_W wide, through the plan's own cone filter on a synthetic TW_N x TW_N raster; raises if
    no T reaches it (the band saturates near 0.8 R)."""
    ra = _step_raster(radius)
    lab = Labels(ra, K=K, band=Mixed(T_w=1.0))
    f = _step(K)
    return _bisect(lambda T: _width(lab.chain(f, T).rho), 1e-3, 10.0, f"R {radius:g}")


def band_radius() -> float:
    """Banded's radius: the cone whose band rho >= 1/2 across a sharp straight interface (phi a
    one-hot step: the T -> 0 limit, rho = 4 a (1 - a) for a the band's share of one side) is
    ROAD_W wide."""
    def width(radius: float) -> float:
        ra = _step_raster(radius)
        b = Banded(ra, radius=radius)
        phi = _step(2)
        mix = b.mix(phi)
        return _width(np.clip(2.0 * (1.0 - (mix ** 2).sum(axis=0)), 0.0, 1.0))
    return _bisect(width, PIXEL, 10.0 * ROAD_W, "the band's cone")


# ---------------------------------------------------------------- MMA


def least_mu(trial: Callable[[float], F64], *, D_of: Callable[[F64], float], budget: float,
             mu0: float) -> tuple[F64, float, int]:
    """(trial(mu), mu, D's evaluations): mu 0 if trial(0)'s D is within the budget, else the
    least mu whose trial's D is, to MU_TOL of the budget (D in [budget (1 - MU_TOL), budget]) or
    a factor 1 + road_simp.NU_TOL in mu, whichever comes first (D falls as mu rises). Bracketed
    from mu0 (the last update's) by factors that grow as their squares (10, 100, 10^4, ..: to
    MU_MAX, where the move limit can hold D over the budget, in nine evaluations, where
    road_simp.least_nu's tenfold steps take 300), then the Illinois rule on log mu (false
    position, the end kept twice in a row its residual halved for the interpolation)."""
    n = 0

    def at(mu: float) -> tuple[F64, float]:
        nonlocal n
        n += 1
        x = trial(mu)
        return x, D_of(x) - budget

    x, d = at(0.0)
    if d <= 0:
        return x, 0.0, n
    mu = min(max(mu0, MU_MIN), MU_MAX) if mu0 > 0 else 1.0
    x, d = at(mu)
    step = 10.0
    if d > 0:                                       # over the budget: climb
        while d > 0:
            if mu >= MU_MAX:
                return x, mu, n                     # the move limit holds D over: the most kept
            lo, dlo = mu, d
            mu = min(mu * step, MU_MAX)
            step *= step
            x, d = at(mu)
        hi, xhi, dhi = mu, x, d
    else:                                           # within it: descend
        hi, xhi, dhi = mu, x, d
        while True:
            if mu <= MU_MIN:
                return xhi, hi, n
            mu = max(mu / step, MU_MIN)
            step *= step
            x, d = at(mu)
            if d > 0:
                lo, dlo = mu, d
                break
            hi, xhi, dhi = mu, x, d
    flo, fhi, kept = dlo, dhi, 0
    while dhi < -MU_TOL * budget and hi / lo > 1.0 + rs.NU_TOL:
        a, b = float(np.log(lo)), float(np.log(hi))
        s = b - fhi * (b - a) / (fhi - flo)
        s = min(max(s, a + 0.01 * (b - a)), b - 0.01 * (b - a))     # strictly inside
        mu = float(np.exp(s))
        x, d = at(mu)
        if d > 0:
            lo, dlo, flo = mu, d, d
            fhi = 0.5 * fhi if kept == 1 else fhi
            kept = 1
        else:
            hi, xhi, dhi, fhi = mu, x, d, d
            flo = 0.5 * flo if kept == -1 else flo
            kept = -1
    return xhi, hi, n


class Mma:
    """Svanberg's MMA with one constraint (module docstring), its asymptotes the instance's state:
    the subproblem's x(mu) in closed form per variable, mu bisected on the true D
    (road_simp.least_nu). Its variables are the free ones alone (each moves on its own: a fixed
    variable's asymptotes never reach a free one's)."""

    def __init__(self, n: int) -> None:
        self.x1: F64 | None = None
        self.x2: F64 | None = None
        self.L = np.zeros(n)
        self.U = np.ones(n)

    def step(self, x: F64, g0: F64, g1: F64, *, D_of: Callable[[F64], float], budget: float,
             mu0: float) -> tuple[F64, float, float, int]:
        """(the new x, mu, the share of the variables at their move limit, D's evaluations)."""
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
            r = RAA * max(float(np.abs(g).max(initial=0.0)), 1e-300)
            gp, gm = np.maximum(g, 0.0), np.maximum(-g, 0.0)
            return ((U - x) ** 2 * (1.001 * gp + 0.001 * gm + r),
                    (x - L) ** 2 * (0.001 * gp + 1.001 * gm + r))

        p0, q0 = pq(g0)
        p1, q1 = pq(g1)

        def trial(mu: float) -> F64:
            a, b = np.sqrt(p0 + mu * p1), np.sqrt(q0 + mu * q1)
            return np.clip((L * a + U * b) / (a + b), lo, hi)

        xn, mu, n = least_mu(trial, D_of=D_of, budget=budget, mu0=mu0)
        at = np.abs(xn - x) >= MOVE - 1e-9
        return xn, mu, float(at.mean()), n


# ---------------------------------------------------------------- the problem


class Iterate(NamedTuple):
    """One update's row (`J` = P + the price's cost; D the true ramp's, `D_relaxed` the stage's
    (Partition.D: the one held to the budget); `grey` the share of the design pixels with 0.05 <
    rho < 0.95; `at_limit` the share of the free variables MMA moved by MOVE; `labels` the pixels
    each label holds, argmax, joined; `n_D` D's evaluations in the search for mu)."""
    T: float
    beta: float
    q: float
    eps: float
    it: int
    P: float
    L_P: float
    J: float
    C: float
    D: float
    D_relaxed: float
    A: float
    mu: float
    n_D: int
    change: float
    at_limit: float
    grey: float
    labels: str
    t: float


class Partition:
    """The block's label-field problem (module docstring) on road_simp's eps world."""

    def __init__(self, rd: rs.Roads, *, K: int, band: Band) -> None:
        if rd.held.any() or rd.free.any():
            raise ValueError("the partition's eps world clears no building")
        self.rd, self.K = rd, K
        self.lab = Labels(rd.raster, K=K, band=band)
        self.y = np.zeros(rd.m.n)               # no clearing
        self.npx = len(rd.raster.on)

    def world(self, f: F64, T: float, beta: float, q: float) -> tuple[rs.World, Chain]:
        """(the eps world, the chain's values) at the fields f (K, pixels)."""
        ch = self.lab.chain(f, T)
        r = relax.project(ch.rho, beta)[self.rd.raster.pix]
        return self.rd.world_of(ch.rho, r, self.y, q, beta), ch

    def back_r(self, w: rs.World, g_r: F64) -> F64:
        """A per-triangle derivative in r to the pixels' rho (w.xf is rho)."""
        g = np.bincount(self.rd.raster.pix, weights=g_r, minlength=self.npx)
        return g * relax.dproject(w.xf, w.beta)

    def D(self, r: F64, eps: float) -> float:
        """D with the ramp's tail eps (module docstring): road_simp's D (the true ramp) plus eps
        times the population-weighted share of each building covered past the knee, (X - tau) /
        (1 - tau) (a fully covered building counts 1 + eps)."""
        rd = self.rd
        return rd.D(r, rd.held) + eps * float(rd.w @ _tail(rd.cover @ r))

    def dD(self, w: rs.World, eps: float) -> F64:
        """dD / dr per triangle, the tail's slope eps / (1 - tau) past the knee."""
        rd = self.rd
        X = rd.cover @ w.r
        return rd.dD(w)[0] + eps * np.asarray(rd.cover.T @ (rd.w * (X > TAU) / (1.0 - TAU)))

    def D_of(self, f: F64, T: float, beta: float, eps: float) -> float:
        rho = self.lab.chain(f, T).rho
        return self.D(relax.project(rho, beta)[self.rd.raster.pix], eps)

    def derivs(self, f: F64, T: float, beta: float, q: float, eps: float,
               price: rs.SheetPrice) -> tuple[rs.World, float, float, F64, F64]:
        """(the world, P, the price's cost, d(P + cost)/df, dD/df), the derivatives (K,
        pixels)."""
        w, ch = self.world(f, T, beta, q)
        P, gs, _ = self.rd.P_grad(w)
        cost = price.charge(self.rd, w, rs.OC())          # (the sheet price reads no update)
        g_r = gs * q * np.maximum(w.r, 1e-300) ** (q - 1.0) + cost.d_r
        g = self.lab.back(np.stack([self.back_r(w, g_r), self.back_r(w, self.dD(w, eps))]), ch, T)
        return w, P, cost.cost, g[0], g[1]


def _tail(X: F64) -> F64:
    return np.maximum(0.0, X - TAU) / (1.0 - TAU)


def simp(pt: Partition, f: F64, plan: Plan, *, log: Any) -> tuple[F64, list[Iterate]]:
    """The plan's MMA from the start's fields f (K, pixels), on the free variables (every field on
    the design pixels; off them 0)."""
    K, npx = pt.K, pt.npx
    on = pt.rd.raster.on
    free = np.flatnonzero(np.tile(on, K))
    mma = Mma(len(free))
    rows: list[Iterate] = []
    mu = 0.0
    t0 = time.time()
    P0 = pt.rd.P0

    def full(z: F64) -> F64:
        out = np.zeros(K * npx)
        out[free] = z
        return out.reshape(K, npx)

    for st in plan.stages:
        T, beta, q, eps = pt.lab.band.T(st.t), st.beta, st.q, st.eps

        def D_of(z: F64, T: float = T, beta: float = beta, eps: float = eps) -> float:
            return pt.D_of(full(z), T, beta, eps)

        for it in range(plan.iters):
            w, P, cost, g0, g1 = pt.derivs(f, T, beta, q, eps, plan.price)
            z, mu, at, nD = mma.step(f.ravel()[free], g0.ravel()[free] / P0,
                                     g1.ravel()[free], D_of=D_of, budget=plan.D, mu0=mu)
            fn = full(z)
            change = float(np.abs(fn - f)[:, on].max())
            f = fn
            rho = w.xf[on]
            lab = np.bincount(np.argmax(pt.lab.filter(f), axis=0)[on], minlength=K)
            rows.append(Iterate(T=T, beta=beta, q=q, eps=eps, it=it, P=P, L_P=1.0 - P / P0,
                                J=(P + cost) / P0, C=pt.rd.C(w), D=pt.rd.D(w.r, w.c),
                                D_relaxed=pt.D(w.r, eps), A=float(pt.rd.area @ w.r), mu=mu,
                                n_D=nD, change=change, at_limit=at,
                                grey=float(((rho > 0.05) & (rho < 0.95)).mean()),
                                labels=",".join(str(int(v)) for v in lab), t=time.time() - t0))
            r = rows[-1]
            log(f"T {T:.3f} beta {beta:g} q {q:g} eps {eps:g} it {it}: L_P {r.L_P:.4f}, J "
                f"{r.J:.4f}, D {r.D:.4f} (relaxed {r.D_relaxed:.4f}), C {r.C:.0f} m^2, paved "
                f"{r.A:.0f} m^2, mu {mu:.3g}, change {change:.3f}, at limit {at:.3f}, grey "
                f"{r.grey:.4f}, labels {r.labels}, {nD} D ({r.t:.0f}s)")
    return f, rows


# ---------------------------------------------------------------- answers and measures


def thick(lab: I64) -> I64:
    """The label raster (2-D) with each connected region (4-connected) narrower than a road merged
    into the neighbour it shares the most pixel edges with, the narrowest first, until none is.
    A region's width is twice its largest distance to another region (a pixel's distance to the
    nearest pixel on a region's edge, plus the half pixel to the boundary): under ROAD_W, every
    pixel of it is road once its boundary is built ROAD_W wide, so it is no sub-block. The argmax
    leaves such islands and slivers where fields nearly tie; their boundaries' corridors are blobs
    whose skeletons are spurs off the network (rev 3's cuts' 1 -- 4 dead ends, every one) or
    floating loops."""
    lab = lab.copy()
    reg = np.asarray(label(lab + 1, background=0, connectivity=1), dtype=np.int64) - 1
    n = int(reg.max()) + 1
    while True:
        edge = np.zeros(reg.shape, dtype=bool)
        dx, dy = reg[1:, :] != reg[:-1, :], reg[:, 1:] != reg[:, :-1]
        edge[1:, :] |= dx
        edge[:-1, :] |= dx
        edge[:, 1:] |= dy
        edge[:, :-1] |= dy
        depth = ndimage.distance_transform_edt(~edge)
        half = np.full(n, -1.0)
        live = np.flatnonzero(np.bincount(reg.ravel(), minlength=n) > 0)
        half[live] = np.asarray(ndimage.maximum(depth, reg, index=live)) + 0.5
        thin = live[2.0 * half[live] * PIXEL < ROAD_W]
        if not len(thin) or len(live) == 1:
            return lab
        r = int(thin[np.argmin(half[thin])])
        m = reg == r
        ring = ndimage.binary_dilation(m) & ~m
        nb = np.bincount(reg[ring], minlength=n)
        nb[r] = 0
        to = int(np.argmax(nb))
        lab[m] = lab[ring & (reg == to)][0]
        reg[m] = to


def boundary_lines(ra: rs.Raster, lab: I64) -> Any:
    """The boundaries of a label raster (2-D): each pixel edge between two labels, merged and
    simplified (local frame)."""
    segs = []
    i, j = np.nonzero(lab[1:, :] != lab[:-1, :])           # an edge at x = i + 1
    segs.append(np.stack([np.stack([i + 1, j], 1), np.stack([i + 1, j + 1], 1)], 1))
    i, j = np.nonzero(lab[:, 1:] != lab[:, :-1])           # an edge at y = j + 1
    segs.append(np.stack([np.stack([i, j + 1], 1), np.stack([i + 1, j + 1], 1)], 1))
    xy = ra.lo + PIXEL * np.concatenate(segs).astype(np.float64)
    merged = shapely.line_merge(shapely.multilinestrings(shapely.linestrings(xy)))
    return shapely.simplify(merged, SNAP_TOL)


def cut_lines(pt: Partition, f: F64) -> Any:
    """The label boundaries (module docstring) of the argmax labels less their regions narrower
    than a road (`thick`), merged and simplified (local frame)."""
    ra = pt.rd.raster
    return boundary_lines(ra, thick(pt.lab.argmax(f).reshape(ra.shape)))


def gridded(geo: Any, ground: Any) -> Any:
    """An answer's ground (world frame) with its vertices on the GRID_M grid of the block's local
    frame, the carved mesh's, its polygons kept (a collapsed sliver dropped). Off the grid the
    band's smoothed outline left two vertices 7.6e-10 m apart, which the carved mesh's conformity
    refused (rev 3's ears start at D 0.15; the parked "A carved atom has no width floor"); the 5810
    page's islands run unioned its stops on the same grid for the same refusal."""
    g = shapely.set_precision(geo.local(ground), GRID_M)
    parts = shapely.get_parts(g)
    return geo.utm(shapely.union_all(parts[np.isin(shapely.get_type_id(parts), (3, 6))]))


def cut_of(b: Any, pt: Partition, f: F64) -> tuple[Any, Any]:
    """(the cut answer's lines, its ground on the grid: `gridded`), world frame, clipped to the
    block."""
    geo = pt.rd.m.geo
    cut = shapely.intersection(geo.utm(cut_lines(pt, f)), b.boundary)
    cut = shapely.line_merge(shapely.union_all(shapely.get_parts(cut)))
    ground = shapely.intersection(
        road_corridor(gpd.GeoDataFrame(geometry=list(shapely.get_parts(cut)), crs=b.crs)
                      .assign(width_m=ROAD_W)), b.boundary)
    return cut, gridded(geo, ground)


def band_of(pt: Partition, w: rs.World) -> Any:
    """The band answer's ground (the triangles at r >= 1/2, road_simp.paved_of), world frame, on
    the grid (`gridded`)."""
    geo = pt.rd.m.geo
    return gridded(geo, geo.utm(rs.paved_of(pt.rd, w.r)))


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
    return faces_of(b, ground) | dict(
        road_m=float(shapely.length(parts).sum()),
        floating_m=float(shapely.length(floating).sum()),
        skeleton_m=float(shapely.length(sk_lines)), dead_ends=ends,
        depth_max=float(d.max(initial=0.0)),
        homes_within50=float(near[np.isfinite(q)].mean()),
        deep_within50=float(near[deep].mean()),
        L_P_deep=1.0 - float(m.pop[ok] @ u[ok]) / float(m.pop[ok] @ u0[ok]),
        damage_cut=dm.cut, damage_detour=dm.detour)


def faces_of(b: Any, ground: Any) -> dict[str, Any]:
    """The sub-blocks of an answer's ground (world frame): the faces of the block less it of
    FACE_M2 and more, their count and areas. The ground is grown by the grid's step first: on the
    grid (`gridded`) a road's end at the block's edge can stop half a millimetre short of it,
    which joined two sub-blocks through the gap (spec20's start: 15 faces for its 21)."""
    faces = shapely.get_parts(shapely.difference(b.boundary, shapely.buffer(ground, GRID_M)))
    fa = np.sort(shapely.area(faces)[shapely.area(faces) >= FACE_M2])[::-1]
    return dict(faces=int(len(fa)), face_largest_m2=float(fa[0]) if len(fa) else 0.0,
                face_median_m2=float(np.median(fa)) if len(fa) else 0.0,
                face_p90_m2=float(np.quantile(fa, 0.9)) if len(fa) else 0.0)


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
    return Partition(rd, K=plan.K, band=plan.band.of(rd.raster, radius=plan.radius, K=plan.K))


def merged(cells: I64) -> I64:
    """The cells' connected regions (4-connected), each under MIN_REGION_M2 merged into the
    neighbour it shares the most pixel edges with, smallest first, until none is (the nodal
    start's row-normalized eigenvectors scatter fragments where every eigenfunction is near 0,
    beside the street); renumbered 0 .. n-1."""
    reg = np.asarray(label(cells + 1, background=0, connectivity=1), dtype=np.int64) - 1
    small = MIN_REGION_M2 / PIXEL ** 2
    while True:
        size = np.bincount(reg.ravel())
        live = np.flatnonzero((size > 0) & (size < small))
        if not len(live) or (size > 0).sum() == 1:
            break
        r = int(live[np.argmin(size[live])])
        m = reg == r
        ring = ndimage.binary_dilation(m) & ~m
        nb = np.bincount(reg[ring], minlength=len(size))
        nb[r] = 0
        reg[m] = int(np.argmax(nb))
    _, out = np.unique(reg, return_inverse=True)
    return np.asarray(out.reshape(cells.shape), dtype=np.int64)


def fields_of(b: Any, pt: Partition, start: Start) -> tuple[F64, int]:
    """The start's fields (K, pixels): each pixel's label 1, the rest 0; and its cells' count
    (each start's cell split into its connected regions first: a k-means cluster can lie in
    several, and coloured whole their adjacencies add up; `merged`)."""
    cells = merged(start.cells(b, pt.rd.raster, pt.rd.m).reshape(pt.rd.raster.shape)).ravel()
    lab = colour(cells, pt.rd.raster.shape, pt.K)[cells]
    f = np.zeros((pt.K, pt.npx))
    f[lab, np.arange(pt.npx)] = 1.0
    return np.where(pt.rd.raster.on, f, 0.0), int(cells.max()) + 1


def start(bid: str, plan: Plan) -> None:
    """The start's eps-world D, C and paved area at each stage's (T, beta), no solve."""
    metric = metrics.metric_of(metrics.RUNTIME, device="cpu")
    [b] = common.build_blocks([bid])
    pt = partition_of(b, metric, plan)
    f, n = fields_of(b, pt, plan.start)
    print(f"{bid} {plan.start.name}: {n} cells", flush=True)
    for st in plan.stages:
        T = pt.lab.band.T(st.t)
        w = pt.world(f, T, st.beta, st.q)[0]
        print(f"  T {T:.3f} beta {st.beta:g} q {st.q:g} eps {st.eps:g}: D "
              f"{pt.rd.D(w.r, w.c):.4f} (relaxed {pt.D(w.r, st.eps):.4f}), C {pt.rd.C(w):.0f} "
              f"m^2, paved {float(pt.rd.area @ w.r):.0f} m^2", flush=True)


def run(bid: str, plan: Plan, device: str, *, standing: clearings.Standing) -> None:
    metric = metrics.metric_of(metrics.RUNTIME, device=device)
    out = OUT / metrics.RUNTIME / plan.name / f"{bid}.parquet"
    if not standing.todo(out):
        return
    t0 = time.time()
    [b] = common.build_blocks([bid])
    pt = partition_of(b, metric, plan)
    f, n = fields_of(b, pt, plan.start)
    print(f"{bid}: {pt.rd.m.fx.fm.nt} triangles, {int(pt.rd.raster.on.sum())} design pixels, "
          f"K {plan.K}, start {plan.start.name}: {n} cells ({time.time() - t0:.0f}s)",
          flush=True)
    f0 = f
    f, rows = simp(pt, f0, plan, log=lambda s: print(s, flush=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    on = pt.rd.raster.on
    pd.DataFrame({f"psi{k}": f[k, on].astype(np.float32) for k in range(plan.K)}).to_parquet(
        out.with_name(f"{bid}_psi.parquet"))
    ans = answers(b, pt, plan, start=f0, f=f)
    del pt
    sc = scores(b, metric, ans, out=out, log=lambda s: print(f"{s} ({time.time() - t0:.0f}s)",
                                                            flush=True))
    df = pd.DataFrame(rows).assign(block=bid, plan=plan.name, **sc, device=device,
                                   host=socket.gethostname(), peak_rss_gb=gap_calib.peak_gb(),
                                   t_total=time.time() - t0)
    standing.write(df, out)


class Answer(NamedTuple):
    """An answer's ground and lines (world frame)."""
    ground: Any
    lines: Any


def answers(b: Any, pt: Partition, plan: Plan, *, start: F64, f: F64) -> dict[str, Answer]:
    """The run's three answers (module docstring) from the start's fields and the last ones."""
    last = plan.stages[-1]
    w = pt.world(f, pt.lab.band.T(last.t), last.beta, last.q)[0]
    band = band_of(pt, w)
    band_lines = skeleton(b, band)[0] if not band.is_empty else shapely.MultiLineString()
    s_lines, s_ground = cut_of(b, pt, start)
    c_lines, c_ground = cut_of(b, pt, f)
    return dict(start=Answer(s_ground, s_lines), band=Answer(band, band_lines),
                cut=Answer(c_ground, c_lines))


def scores(b: Any, metric: gcl.ComplexMetric, ans: dict[str, Answer], *, out: Path,
           log: Callable[[str], None]) -> dict[str, Any]:
    """Each answer stored beside the rows `out` (<block>_<answer>_ground and _lines) and scored
    (`score_answer`); the scores as the rows' columns, <answer>_<measure>."""
    bid = b.block_id
    sc: dict[str, Any] = {}
    for name, a in ans.items():
        gpd.GeoDataFrame(geometry=[a.ground], crs=b.crs).to_parquet(
            out.with_name(f"{bid}_{name}_ground.parquet"))
        gpd.GeoDataFrame(geometry=list(shapely.get_parts(a.lines)), crs=b.crs).to_parquet(
            out.with_name(f"{bid}_{name}_lines.parquet"))
        if a.ground.is_empty:
            log(f"{name}: empty")
            continue
        s = score_answer(b, metric, a.ground, a.lines)
        sc |= {f"{name}_{k}": v for k, v in s.items()}
        log(f"{bid} {out.parent.name} {name}: L_P {s['L_P']:.4f}, L_J2 {s['L_J2']:.4f}, D "
            f"{s['D_exact']:.4f}, C {s['C']:.0f} m^2, {s['faces']} sub-blocks (largest "
            f"{s['face_largest_m2']:.0f} m^2), road {s['road_m']:.0f} m, dead ends "
            f"{s['dead_ends']}, deep within 50 m {s['deep_within50']:.2f}, L_P deep "
            f"{s['L_P_deep']:.4f}, cut {s['damage_cut']:.4f}")
    return sc


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
