"""Roadless, heading-aware conduction on a block's free space.

A walker moves forward or back along its current heading and turns slowly: the state is
(cell, axis), the along-axis edges join a cell to the next cell along that axis's lattice vector,
and the turning edges join adjacent axes in the same cell. The operator is symmetric, so the
escape-time identity and Rayleigh monotonicity both hold: P = f^T L^-1 f is the total expected
escape time AND the power of the system-optimal linear-congestion flow, and adding free space can
only lower it.

Roads do not exist here. A network enters only through the space its corridor frees: `free |=
corridor` (CARVE, the geometry the displacement charge already pays for) or every building the
corridor touches is removed whole (OBLITERATE). Demand stays where the buildings were
(rehoused in place), so only the operator changes and monotonicity is exact.

Continuum it approximates:  E(u) = int int (d_s u)^2 + (1/ell^2)(d_theta u)^2  dx dtheta
    along edge (x, x+v_k), layer k:   w = m_k / |v_k|^2        (h^2 cancels)
    turning edge (x,k)-(x,k+1):       w = h^2 / (ell^2 dtheta_{k,k+1})
m_k is axis k's share of [0, pi); a straight open channel of width W and length L conducts
m_k W / L in its own layer at every lattice angle (checked in checks.py).
"""
from __future__ import annotations

import dataclasses
import hashlib
import re
import sys
from dataclasses import dataclass
from typing import Callable, NamedTuple, Protocol
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import shapely
from numba import njit
from numpy.typing import NDArray
from scipy import ndimage
from scipy.sparse.linalg import factorized

sys.path.insert(0, str(Path.home() / ".cache/reblock-research/pydeps"))
import pyamg  # noqa: E402

# axis lattice vectors (dx = column, dy = row), angles in [0, pi)
AXES = {
    8: [(1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1)],
    16: [(1, 0), (3, 1), (2, 1), (3, 2), (1, 1), (2, 3), (1, 2), (1, 3),
         (0, 1), (-1, 3), (-1, 2), (-2, 3), (-1, 1), (-3, 2), (-2, 1), (-3, 1)],
}


def axes(K: int) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64],
                          NDArray[np.float64]]:
    """(vectors, angles, m_k, gap to the next axis cyclically)."""
    v = np.array(AXES[K], dtype=np.int64)
    th = np.arctan2(v[:, 1], v[:, 0]) % np.pi
    order = np.argsort(th)
    assert (order == np.arange(K)).all()
    nxt = np.roll(th, -1)
    nxt[-1] += np.pi
    gap = nxt - th
    m = 0.5 * (gap + np.roll(gap, 1))
    return v, th, m, gap


def _line_cells(dx: int, dy: int) -> list[tuple[int, int]]:
    """Intermediate cells a step (dx, dy) passes through (supercover, endpoints excluded)."""
    n = 64
    ts = (np.arange(1, n) + 0.0) / n
    cells = set()
    for t in ts:
        x, y = t * dx, t * dy
        # every cell whose square the segment enters (corner contacts count as both neighbours)
        for cx in {int(np.floor(x + 0.5 - 1e-9)), int(np.ceil(x - 0.5 + 1e-9))}:
            for cy in {int(np.floor(y + 0.5 - 1e-9)), int(np.ceil(y - 0.5 + 1e-9))}:
                if (cx, cy) not in {(0, 0), (dx, dy)}:
                    cells.add((cx, cy))
    return sorted(cells)


@dataclass(frozen=True, eq=False)
class Grid:
    x0: float
    y0: float
    h: float
    inside: NDArray[np.bool_]            # some sub-sample of the cell is inside the block
    building: NDArray[np.bool_]          # most of the cell is footprint (for the EDT only)
    ground: NDArray[np.bool_]            # centre within `band_m` of Block.streets
    dist_b: NDArray[np.float64]          # metres to the nearest building cell (EDT)
    xy: NDArray[np.float64]              # (ny, nx, 2) cell centres
    isub: NDArray[np.bool_]              # (ny, nx, S*S) sub-samples inside the block
    bsub: NDArray[np.bool_]              # (ny, nx, S*S) sub-samples inside a footprint
    S: int
    # per-grid derived structure (edge pattern, master aggregates), built on first use
    cache: dict = dataclasses.field(default_factory=dict, compare=False, repr=False)

    @property
    def ff0(self) -> NDArray[np.float64]:
        """Open fraction of each cell: inside the block and not footprint."""
        return (self.isub & ~self.bsub).mean(axis=-1)

    def _sub_xy(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        t = ((np.arange(self.S) + 0.5) / self.S - 0.5) * self.h
        ox, oy = np.meshgrid(t, t)
        X = self.xy[..., 0][..., None] + ox.ravel()[None, None, :]
        Y = self.xy[..., 1][..., None] + oy.ravel()[None, None, :]
        return X, Y

    def sub_of(self, geom) -> NDArray[np.bool_]:
        """(ny, nx, S*S): sub-samples inside `geom` (only those inside the block)."""
        out = np.zeros_like(self.isub)
        if geom is None or geom.is_empty:
            return out
        shapely.prepare(geom)
        X, Y = self._sub_xy()
        out[self.isub] = shapely.contains_xy(geom, X[self.isub], Y[self.isub])
        return out

    @classmethod
    def of(cls, boundary, footprints, streets, h: float, band_m: float = 1.0,
           offset: tuple[float, float] = (0.3713, 0.1931)) -> Grid:
        """`offset` (fractions of h) keeps cell centres off exactly-aligned geometry. Sub-sample
        (i, j) of the fine lattice sits at (x0 - h/2 + (j + 1/2) h/S, y0 - h/2 + (i + 1/2) h/S):
        inside the block by a scanline fill of its rings, inside a footprint by
        `_rasterize`."""
        minx, miny, maxx, maxy = boundary.bounds
        xs = np.arange(minx - h + offset[0] * h, maxx + 2 * h, h)
        ys = np.arange(miny - h + offset[1] * h, maxy + 2 * h, h)
        X, Y = np.meshgrid(xs, ys)
        S = 4
        ny, nx = X.shape
        d = h / S
        fx = xs[0] - h / 2 + (np.arange(nx * S) + 0.5) * d
        fy = ys[0] - h / 2 + (np.arange(ny * S) + 0.5) * d
        isub = _to_cells(_scanfill(boundary, fx, fy), S)
        bsub = isub & _to_cells(_rasterize(np.asarray(footprints), fx, fy) >= 0, S)
        inside = isub.any(axis=-1)
        building = bsub.mean(axis=-1) > 0.5
        st = shapely.union_all(np.asarray(streets)).buffer(band_m)
        shapely.prepare(st)
        ground = np.zeros_like(inside)
        ground[inside] = shapely.contains_xy(st, X[inside], Y[inside])
        dist_b = ndimage.distance_transform_edt(~building) * h
        return cls(x0=float(xs[0]), y0=float(ys[0]), h=h, inside=inside, building=building,
                   ground=ground, dist_b=dist_b, xy=np.stack([X, Y], axis=-1),
                   isub=isub, bsub=bsub, S=S)

    def label_sub(self, polys) -> NDArray[np.int64]:
        """(ny, nx, S*S): index of the footprint holding each footprint sub-sample (-1 none;
        overlaps go to the later footprint)."""
        S, h = self.S, self.h
        ny, nx = self.inside.shape
        d = h / S
        fx = self.x0 - h / 2 + (np.arange(nx * S) + 0.5) * d
        fy = self.y0 - h / 2 + (np.arange(ny * S) + 0.5) * d
        return np.where(self.bsub, _to_cells(_rasterize(np.asarray(polys), fx, fy), S), -1)

    def mask_of(self, geom) -> NDArray[np.bool_]:
        out = np.zeros_like(self.inside)
        if geom is None or geom.is_empty:
            return out
        shapely.prepare(geom)
        X, Y = self.xy[..., 0], self.xy[..., 1]
        out[self.inside] = shapely.contains_xy(geom, X[self.inside], Y[self.inside])
        return out


def _to_cells(fine, S: int):
    """(ny S, nx S) fine lattice -> (ny, nx, S*S), sub-sample index (row in cell) S + column."""
    ny, nx = fine.shape[0] // S, fine.shape[1] // S
    return fine.reshape(ny, S, nx, S).transpose(0, 2, 1, 3).reshape(ny, nx, S * S)


def _scanfill(geom, fx: NDArray[np.float64], fy: NDArray[np.float64]) -> NDArray[np.bool_]:
    """(len(fy), len(fx)): lattice points inside `geom` (a polygon or multipolygon), by the
    even-odd rule along each row: a point is inside when an odd number of ring edges cross its
    row to its left. Exact except for points on an edge, which the grid offset avoids."""
    segs = []
    for ring in shapely.get_rings(shapely.get_parts(geom)):
        c = shapely.get_coordinates(ring)
        segs.append(np.hstack([c[:-1], c[1:]]))
    x1, y1, x2, y2 = np.vstack(segs).T
    out = np.zeros((len(fy), len(fx)), dtype=bool)
    for i, y in enumerate(fy):
        hit = (y1 <= y) != (y2 <= y)                       # the edge spans the row (half-open)
        xc = np.sort(x1[hit] + (y - y1[hit]) * (x2[hit] - x1[hit]) / (y2[hit] - y1[hit]))
        out[i] = np.searchsorted(xc, fx) % 2 == 1
    return out


def _rasterize(polys, fx: NDArray[np.float64], fy: NDArray[np.float64]) -> NDArray[np.int64]:
    """(len(fy), len(fx)): index of the polygon holding each lattice point, -1 none (overlaps go
    to the later polygon). Each polygon is tested only on the lattice points inside its
    bounding box (fx, fy evenly spaced)."""
    fine = -np.ones((len(fy), len(fx)), dtype=np.int64)
    if len(polys) == 0:
        return fine
    d = fx[1] - fx[0]
    for k, (bx0, by0, bx1, by1) in enumerate(shapely.bounds(polys)):
        j0 = max(int(np.floor((bx0 - fx[0]) / d)), 0)
        j1 = min(int(np.ceil((bx1 - fx[0]) / d)) + 1, len(fx))
        i0 = max(int(np.floor((by0 - fy[0]) / d)), 0)
        i1 = min(int(np.ceil((by1 - fy[0]) / d)) + 1, len(fy))
        if j1 <= j0 or i1 <= i0:
            continue
        X, Y = np.meshgrid(fx[j0:j1], fy[i0:i1])
        fine[i0:i1, j0:j1][shapely.contains_xy(polys[k], X, Y)] = k
    return fine


def demand(grid: Grid, footprints, allowed: NDArray[np.bool_], weights: NDArray[np.float64],
           ring_m: float = 1.0
           ) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.int64]]:
    """Per-cell injection (ny, nx): building j injects weights[j] (its population), uniformly
    over the `allowed` free cells within `ring_m` of it whose nearest building it is (`allowed` =
    cells with a path to the street, so a ring partly in a sealed nook puts all its demand on the
    open side). A building with no allowed ring cell is STRANDED: it injects nothing. Returns
    (field, stranded mask per building, owner building per cell or -1)."""
    ring = allowed & (grid.dist_b <= ring_m + 1e-9)
    rr, cc = np.nonzero(ring)
    polys = np.asarray(footprints)
    n = len(polys)
    f = np.zeros(grid.inside.shape)
    own = -np.ones(grid.inside.shape, dtype=np.int64)
    if len(rr) == 0:
        return f, np.ones(n, dtype=bool), own
    tree = shapely.STRtree(polys)
    pts = shapely.points(grid.xy[rr, cc, 0], grid.xy[rr, cc, 1])
    idx, _ = tree.query_nearest(pts, return_distance=True, all_matches=False)
    owner = np.full(len(rr), -1)
    owner[idx[0]] = idx[1]
    cnt = np.bincount(owner, minlength=n).astype(float)
    np.add.at(f, (rr, cc), weights[owner] / cnt[owner])
    own[rr, cc] = owner
    return f, cnt == 0, own


@njit(cache=True)
def _run_count(clear, dx, dy, offs):
    """Per cell: the number of clear cells in the straight lattice run through it along
    (dx, dy), a step counting only when its far end and every cell it crosses (`offs`, the
    offsets `_line_cells` gives) are clear. Requires dy > 0, or dy == 0 and dx > 0, so that
    visiting rows (then columns) in decreasing order settles each successor first."""
    ny, nx = clear.shape
    fwd = np.zeros((ny, nx), dtype=np.int64)
    bwd = np.zeros((ny, nx), dtype=np.int64)
    for r in range(ny - 1, -1, -1):
        for c in range(nx - 1, -1, -1):
            if clear[r, c]:
                fwd[r, c] = 1
                if _step_ok(clear, r, c, dx, dy, offs):
                    fwd[r, c] += fwd[r + dy, c + dx]
    for r in range(ny):
        for c in range(nx):
            if clear[r, c]:
                bwd[r, c] = 1
                if _step_ok(clear, r, c, -dx, -dy, -offs):
                    bwd[r, c] += bwd[r - dy, c - dx]
    return fwd + bwd - 1


@njit(cache=True)
def _step_ok(clear, r, c, dx, dy, offs):
    ny, nx = clear.shape
    r2, c2 = r + dy, c + dx
    if not (0 <= r2 < ny and 0 <= c2 < nx and clear[r2, c2]):
        return False
    for i in range(offs.shape[0]):
        r3, c3 = r + offs[i, 1], c + offs[i, 0]
        if not (0 <= r3 < ny and 0 <= c3 < nx and clear[r3, c3]):
            return False
    return True


def directions(nmax: int) -> list[tuple[int, int]]:
    """Primitive lattice directions (dx, dy) with max(|dx|, |dy|) <= nmax, one per line
    (dy > 0, or dy == 0 and dx > 0)."""
    return [(dx, dy) for dy in range(0, nmax + 1) for dx in range(-nmax, nmax + 1)
            if (dy > 0 or dx > 0) and np.gcd(dx, dy) == 1]


class AlongConductance(Protocol):
    """Per-heading multipliers on the along-axis edges: layers(open_, h, K)[k] is layer k's
    (ny, nx) factor field (>= 1, non-decreasing as space is freed, so monotonicity survives) or
    the scalar 1.0. vjp(open_, h, K, A) is the gradient with respect to open_ of
    sum_k sum_x A[k, x] layers[k][x] (what clearing a cell does to the factors elsewhere)."""
    name: str

    def layers(self, open_: NDArray[np.float64], h: float,
               K: int) -> list[float] | NDArray[np.float32]: ...

    def vjp(self, open_: NDArray[np.float64], h: float, K: int,
            A: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class Uniform:
    """Every open cell conducts alike (the original model)."""
    name: str = "uni"

    def layers(self, open_, h, K):
        return [1.0] * K

    def vjp(self, open_, h, K, A):
        return np.zeros(open_.shape)


@dataclass(frozen=True)
class Sightline:
    """Layer k conducts better where long straight clear runs exist at headings near k. Over the
    fine lattice directions d (max(|dx|, |dy|) <= nmax), R_d is the capped clear run through the
    cell, as a fraction of r_max; layer k's factor is 1 + beta x the mean of R_d weighted by a
    triangle in angle, width one axis gap either side of axis k. A cell is clear when at least
    `clear_frac` of it is open. Integrated across a channel this measures the straight segments
    it holds: a corridor of width W admits headings within ~W/L of its axis at every offset, so
    its conductance grows faster than W."""
    beta: float
    r_max_m: float = 50.0
    nmax: int = 6
    clear_frac: float = 0.5

    @property
    def name(self) -> str:
        return f"sl{self.beta:g}"

    def runs(self, open_, h) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
        """(angle of each fine direction in [0, pi), capped run fraction (n_dir, ny, nx))."""
        clear = open_ >= self.clear_frac
        dirs = directions(self.nmax)
        ang = np.array([np.arctan2(dy, dx) % np.pi for dx, dy in dirs])
        R = np.empty((len(dirs), *open_.shape), dtype=np.float32)
        for i, (dx, dy) in enumerate(dirs):
            offs = np.array(_line_cells(dx, dy), dtype=np.int64).reshape(-1, 2)
            run = _run_count(clear, dx, dy, offs) * (np.hypot(dx, dy) * h)
            R[i] = np.minimum(run, self.r_max_m) / self.r_max_m
        return ang, R

    def layers(self, open_, h, K):
        ang, R = self.runs(open_, h)
        W = _angle_weights(K, ang)
        return (1.0 + self.beta * np.tensordot(W, R, axes=1)).astype(np.float32)

    def vjp(self, open_, h, K, A):
        """Hard runs are piecewise constant in open_: zero almost everywhere."""
        return np.zeros(open_.shape)


def _angle_weights(K: int, ang: NDArray[np.float64]) -> NDArray[np.float64]:
    """(K, n_dir): direction d's share in layer k, a triangle in angle one axis gap either side
    of axis k, normalised over d per layer."""
    _v, th, _m, gap = axes(K)
    W = np.zeros((K, len(ang)))
    for k in range(K):
        d = (ang - th[k] + np.pi / 2) % np.pi - np.pi / 2     # signed, in [-pi/2, pi/2)
        half = np.where(d >= 0, gap[k], np.roll(gap, 1)[k])   # to the next / previous axis
        wt = np.clip(1.0 - np.abs(d) / half, 0.0, None)
        W[k] = wt / wt.sum()
    return W


@njit(cache=True)
def _soft_fb(open_, dx, dy, offs, s, kappa):
    """Expected free path forward (F) and back (B) along (dx, dy) from each cell, the ray's
    intensity falling as exp(-kappa x the closed length it crosses): each step of length s takes
    optical depth kappa s mean(1 - open) over the cells it enters (its far end and `offs`;
    outside the grid counts closed). Tf, Tb: each cell's step transmittance forward and back."""
    ny, nx = open_.shape
    m = offs.shape[0] + 1
    F = np.zeros((ny, nx))
    B = np.zeros((ny, nx))
    Tf = np.zeros((ny, nx))
    Tb = np.zeros((ny, nx))
    for r in range(ny - 1, -1, -1):
        for c in range(nx - 1, -1, -1):
            r2, c2 = r + dy, c + dx
            if 0 <= r2 < ny and 0 <= c2 < nx:
                tau = 1.0 - open_[r2, c2]
                for i in range(offs.shape[0]):
                    r3, c3 = r + offs[i, 1], c + offs[i, 0]
                    tau += 1.0 - open_[r3, c3] if (0 <= r3 < ny and 0 <= c3 < nx) else 1.0
                Tf[r, c] = np.exp(-kappa * s / m * tau)
                F[r, c] = Tf[r, c] * (s + F[r2, c2])
    for r in range(ny):
        for c in range(nx):
            r2, c2 = r - dy, c - dx
            if 0 <= r2 < ny and 0 <= c2 < nx:
                tau = 1.0 - open_[r2, c2]
                for i in range(offs.shape[0]):
                    r3, c3 = r - offs[i, 1], c - offs[i, 0]
                    tau += 1.0 - open_[r3, c3] if (0 <= r3 < ny and 0 <= c3 < nx) else 1.0
                Tb[r, c] = np.exp(-kappa * s / m * tau)
                B[r, c] = Tb[r, c] * (s + B[r2, c2])
    return F, B, Tf, Tb


@njit(cache=True)
def _soft_vjp(dx, dy, offs, s, kappa, F, B, Tf, Tb, G, out):
    """Add to `out` the gradient over open_ of sum_x G[x] (F[x] + B[x]): reverse-mode through
    the two scans (each cotangent flows back along the ray, damped by the transmittance)."""
    ny, nx = F.shape
    m = offs.shape[0] + 1
    bar = np.zeros((ny, nx))
    # forward rays: F(p) = Tf(p) (s + F(x)) with x = p + v, so visit p before x
    for r in range(ny):
        for c in range(nx):
            b = G[r, c]
            rp, cp = r - dy, c - dx
            if 0 <= rp < ny and 0 <= cp < nx:
                b += Tf[rp, cp] * bar[rp, cp]
            bar[r, c] = b
            r2, c2 = r + dy, c + dx
            if b != 0.0 and 0 <= r2 < ny and 0 <= c2 < nx:
                coef = b * (s + F[r2, c2]) * Tf[r, c] * kappa * s / m
                out[r2, c2] += coef
                for i in range(offs.shape[0]):
                    r3, c3 = r + offs[i, 1], c + offs[i, 0]
                    if 0 <= r3 < ny and 0 <= c3 < nx:
                        out[r3, c3] += coef
    bar[:, :] = 0.0
    for r in range(ny - 1, -1, -1):
        for c in range(nx - 1, -1, -1):
            b = G[r, c]
            rp, cp = r + dy, c + dx
            if 0 <= rp < ny and 0 <= cp < nx:
                b += Tb[rp, cp] * bar[rp, cp]
            bar[r, c] = b
            r2, c2 = r - dy, c - dx
            if b != 0.0 and 0 <= r2 < ny and 0 <= c2 < nx:
                coef = b * (s + B[r2, c2]) * Tb[r, c] * kappa * s / m
                out[r2, c2] += coef
                for i in range(offs.shape[0]):
                    r3, c3 = r - offs[i, 1], c - offs[i, 0]
                    if 0 <= r3 < ny and 0 <= c3 < nx:
                        out[r3, c3] += coef


class Scans(Protocol):
    """Where SoftSightline's 1-D scans run. `xp` is the array module (numpy or cupy); fb returns
    (F, B, Tf, Tb) and vjp_add accumulates into `out`, all `xp` arrays; to_host / to_dev move
    arrays across."""
    xp: object

    def fb(self, open_, dx: int, dy: int, offs, s: float, kappa: float): ...

    def vjp_add(self, dx: int, dy: int, offs, s: float, kappa: float, F, B, Tf, Tb, G,
                out) -> None: ...

    def to_dev(self, a): ...

    def to_host(self, a) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class CpuScans:
    """numba, one core."""
    xp: object = np

    def fb(self, open_, dx, dy, offs, s, kappa):
        return _soft_fb(open_, dx, dy, offs, s, kappa)

    def vjp_add(self, dx, dy, offs, s, kappa, F, B, Tf, Tb, G, out):
        _soft_vjp(dx, dy, offs, s, kappa, F, B, Tf, Tb, G, out)

    def to_dev(self, a):
        return np.asarray(a)

    def to_host(self, a):
        return np.asarray(a)


_SCAN_CU = r"""
__device__ inline double opac(const double* o, int ny, int nx, int r, int c) {
    return (r >= 0 && r < ny && c >= 0 && c < nx) ? 1.0 - o[r * nx + c] : 1.0;
}
// tau of the step from (r, c) to (r + sy*dy, c + sy*dx): its far end and the cells it crosses
__device__ inline double step_tau(const double* o, int ny, int nx, int r, int c, int dx, int dy,
                                  const int* offs, int noffs, int sy) {
    double tau = opac(o, ny, nx, r + sy * dy, c + sy * dx);
    for (int i = 0; i < noffs; i++) tau += opac(o, ny, nx, r + sy * offs[2*i+1], c + sy * offs[2*i]);
    return tau;
}
extern "C" __global__ void soft_fb(const double* o, int ny, int nx, int dx, int dy,
        const int* offs, int noffs, double s, double kappa, const int* sr, const int* sc,
        int nstart, double* F, double* B, double* Tf, double* Tb) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nstart) return;
    double k = kappa * s / (noffs + 1);
    int r = sr[t], c = sc[t], n = 0;
    while (r >= 0 && r < ny && c >= 0 && c < nx) {            // B along the line, from its start
        int x = r * nx + c;
        if (n == 0) { Tb[x] = 0.0; B[x] = 0.0; }
        else {
            double T = exp(-k * step_tau(o, ny, nx, r, c, dx, dy, offs, noffs, -1));
            Tb[x] = T; B[x] = T * (s + B[(r - dy) * nx + (c - dx)]);
        }
        r += dy; c += dx; n++;
    }
    r -= dy; c -= dx;
    for (int j = 0; j < n; j++) {                             // F from its end, back to the start
        int x = r * nx + c;
        if (j == 0) { Tf[x] = 0.0; F[x] = 0.0; }
        else {
            double T = exp(-k * step_tau(o, ny, nx, r, c, dx, dy, offs, noffs, 1));
            Tf[x] = T; F[x] = T * (s + F[(r + dy) * nx + (c + dx)]);
        }
        r -= dy; c -= dx;
    }
}
__device__ inline void addto(double* out, int ny, int nx, int r, int c, double v) {
    if (r >= 0 && r < ny && c >= 0 && c < nx) atomicAdd(&out[r * nx + c], v);
}
extern "C" __global__ void soft_vjp(int ny, int nx, int dx, int dy, const int* offs, int noffs,
        double s, double kappa, const int* sr, const int* sc, int nstart, const double* F,
        const double* B, const double* Tf, const double* Tb, const double* G, double* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nstart) return;
    double k = kappa * s / (noffs + 1);
    // forward rays: F(p) = Tf(p) (s + F(x)), x = p + v: walk from the start, carrying bar F
    int r = sr[t], c = sc[t], n = 0;
    double bar = 0.0, Tprev = 0.0;
    while (r >= 0 && r < ny && c >= 0 && c < nx) {
        int x = r * nx + c;
        bar = G[x] + Tprev * bar;
        int r2 = r + dy, c2 = c + dx;
        if (bar != 0.0 && r2 >= 0 && r2 < ny && c2 >= 0 && c2 < nx) {
            double coef = bar * (s + F[r2 * nx + c2]) * Tf[x] * k;
            addto(out, ny, nx, r2, c2, coef);
            for (int i = 0; i < noffs; i++) addto(out, ny, nx, r + offs[2*i+1], c + offs[2*i], coef);
        }
        Tprev = Tf[x];
        r += dy; c += dx; n++;
    }
    // backward rays: B(p) = Tb(p) (s + B(x)), x = p - v: walk from the end
    r -= dy; c -= dx;
    bar = 0.0; Tprev = 0.0;
    for (int j = 0; j < n; j++) {
        int x = r * nx + c;
        bar = G[x] + Tprev * bar;
        int r2 = r - dy, c2 = c - dx;
        if (bar != 0.0 && r2 >= 0 && r2 < ny && c2 >= 0 && c2 < nx) {
            double coef = bar * (s + B[r2 * nx + c2]) * Tb[x] * k;
            addto(out, ny, nx, r2, c2, coef);
            for (int i = 0; i < noffs; i++) addto(out, ny, nx, r - offs[2*i+1], c - offs[2*i], coef);
        }
        Tprev = Tb[x];
        r -= dy; c -= dx;
    }
}
"""


@dataclass(frozen=True)
class GpuScans:
    """cupy: one GPU thread per lattice line along the direction (each line is an independent
    forward / backward recurrence); the vjp scatters with atomicAdd (a crossed cell belongs to
    other lines too). Needs cupy and CUDA_PATH."""
    _memo: dict = dataclasses.field(default_factory=dict, compare=False, repr=False)

    @property
    def xp(self):
        import cupy as cp
        return cp

    def _kernels(self):
        if "kernels" not in self._memo:
            mod = self.xp.RawModule(code=_SCAN_CU)
            self._memo["kernels"] = (mod.get_function("soft_fb"), mod.get_function("soft_vjp"))
        return self._memo["kernels"]

    def _starts(self, ny: int, nx: int, dx: int, dy: int):
        """Cells whose predecessor (r - dy, c - dx) is off the grid: one per line. They are
        the cells within dy of the top or dx of the near side edge."""
        key = (ny, nx, dx, dy)
        if key not in self._memo:
            cp = self.xp
            st = np.zeros((ny, nx), dtype=bool)
            st[:dy] = True                                    # dy >= 0
            if dx > 0:
                st[:, :dx] = True
            elif dx < 0:
                st[:, nx + dx:] = True
            r, c = np.nonzero(st)
            self._memo[key] = (cp.asarray(r.astype(np.int32)), cp.asarray(c.astype(np.int32)),
                               len(r))
        return self._memo[key]

    def fb(self, open_, dx, dy, offs, s, kappa):
        cp = self.xp
        ny, nx = open_.shape
        f_fb, _ = self._kernels()
        sr, sc, ns = self._starts(ny, nx, dx, dy)
        F, B, Tf, Tb = (cp.zeros((ny, nx)) for _ in range(4))
        og = cp.ascontiguousarray(open_, dtype=cp.float64)
        oo = cp.asarray(np.asarray(offs, dtype=np.int32).ravel())
        f_fb(((ns + 127) // 128,), (128,),
             (og, np.int32(ny), np.int32(nx), np.int32(dx), np.int32(dy), oo,
              np.int32(len(offs)), np.float64(s), np.float64(kappa), sr, sc, np.int32(ns),
              F, B, Tf, Tb))
        return F, B, Tf, Tb

    def vjp_add(self, dx, dy, offs, s, kappa, F, B, Tf, Tb, G, out):
        cp = self.xp
        ny, nx = F.shape
        _, f_vjp = self._kernels()
        sr, sc, ns = self._starts(ny, nx, dx, dy)
        oo = cp.asarray(np.asarray(offs, dtype=np.int32).ravel())
        f_vjp(((ns + 127) // 128,), (128,),
              (np.int32(ny), np.int32(nx), np.int32(dx), np.int32(dy), oo, np.int32(len(offs)),
               np.float64(s), np.float64(kappa), sr, sc, np.int32(ns), F, B, Tf, Tb,
               cp.ascontiguousarray(G, dtype=cp.float64), out))

    def to_dev(self, a):
        return self.xp.asarray(a)

    def to_host(self, a):
        return self.xp.asnumpy(a)


@dataclass(frozen=True)
class SoftSightline:
    """Sightline with translucent buildings: a ray's intensity falls as exp(-kappa x the closed
    length it crosses), and R_d is its expected free path (forward + back) along direction d. A
    line blocked only by a small building still scores, so clearing that building has a
    gradient (the counterfactual corridor is visible); as kappa grows this tends to the hard
    runs. Layer k's factor is 1 + beta x the angle-weighted mean of g(R_d) = R^n / (R^n + r0^n):
    n 1 is concave (short gaps already earn a lot), n 2 is S-shaped (gaps well under r0 earn
    almost nothing, lines past r0 nearly all; joining two runs across r0 pays most).
    Differentiable: `vjp` is exact. (Turning walkers were tried and dropped: NOTES, "Turning
    sightline".)"""
    beta: float
    kappa: float = 2.0         # per metre of building crossed
    r0_m: float = 20.0
    hill: float = 1.0          # the exponent n
    nmax: int = 6
    scans: Scans = CpuScans()
    # the last open field's layers (one round builds the same operator more than once)
    _cache: dict = dataclasses.field(default_factory=dict, compare=False, repr=False)

    @property
    def name(self) -> str:
        return f"ss{self.beta:g}k{self.kappa:g}" + (
            "" if self.hill == 1.0 and self.r0_m == 20.0 else f"n{self.hill:g}r{self.r0_m:g}")

    def g(self, R):
        x = (R / self.r0_m) ** self.hill
        return x / (1.0 + x)

    def dg(self, R):
        x = (R / self.r0_m) ** self.hill
        return self.hill * x / ((R + 1e-12) * (1.0 + x) ** 2)

    def _dirs(self):
        dirs = directions(self.nmax)
        ang = np.array([np.arctan2(dy, dx) % np.pi for dx, dy in dirs])
        return dirs, ang

    def _steps(self, h):
        """Per fine direction: (index, dx, dy, crossed-cell offsets, step length)."""
        for i, (dx, dy) in enumerate(self._dirs()[0]):
            yield (i, dx, dy, np.array(_line_cells(dx, dy), dtype=np.int64).reshape(-1, 2),
                   float(np.hypot(dx, dy) * h))

    def runs(self, open_, h):
        """Yield (d, R_d) per fine direction, R_d the expected free path in metres."""
        for i, F, B in self.paths(open_, h):
            yield i, F + B

    def paths(self, open_, h):
        """Yield (d, F_d, B_d) on the host: the expected free path forward and back. F x B is
        the measure of the segments along d that pass through the cell (pairs of a start behind
        and an end ahead); its sum along a line of length L is L^2 / 2."""
        od = self.scans.to_dev(open_)
        for i, dx, dy, offs, s in self._steps(h):
            F, B, _tf, _tb = self.scans.fb(od, dx, dy, offs, s, self.kappa)
            yield i, self.scans.to_host(F), self.scans.to_host(B)

    def layers(self, open_, h, K):
        """(K, ny, nx) on the scans' device."""
        key = (hashlib.blake2b(np.ascontiguousarray(self.scans.to_host(open_)).tobytes(),
                               digest_size=16).digest(), h, K)
        if key in self._cache:
            return self._cache[key]
        xp = self.scans.xp
        W = _angle_weights(K, self._dirs()[1])
        od = self.scans.to_dev(open_)
        out = xp.ones((K, *od.shape))
        for i, dx, dy, offs, s in self._steps(h):
            F, B, _tf, _tb = self.scans.fb(od, dx, dy, offs, s, self.kappa)
            g = self.g(F + B)
            for k in np.flatnonzero(W[:, i]):
                out[k] += (self.beta * W[k, i]) * g
        self._cache.clear()
        self._cache[key] = out
        return out

    def vjp(self, open_, h, K, A):
        """(ny, nx) on the scans' device."""
        xp = self.scans.xp
        W = _angle_weights(K, self._dirs()[1])
        od = self.scans.to_dev(open_)
        Ad = self.scans.to_dev(A)
        out = xp.zeros(od.shape)
        for i, dx, dy, offs, s in self._steps(h):
            F, B, Tf, Tb = self.scans.fb(od, dx, dy, offs, s, self.kappa)
            G = self.beta * self.dg(F + B) * xp.tensordot(xp.asarray(W[:, i]), Ad, axes=1)
            self.scans.vjp_add(dx, dy, offs, s, self.kappa, F, B, Tf, Tb, G, out)
        return out


def along_of(spec: str, scans: Scans = CpuScans()) -> AlongConductance:
    """`uni` -> Uniform(); `sl3` -> Sightline(beta 3); `ss10k2` -> SoftSightline(beta 10,
    kappa 2); `ss30k2n2r30` -> SoftSightline(beta 30, kappa 2, hill 2, r0 30 m)."""
    if spec == "uni":
        return Uniform()
    if spec.startswith("sl"):
        return Sightline(float(spec[2:]))
    m = re.fullmatch(r"ss([0-9.]+)k([0-9.]+)(?:n([0-9.]+)r([0-9.]+))?", spec)
    if m:
        shape = {} if m.group(3) is None else dict(hill=float(m.group(3)),
                                                   r0_m=float(m.group(4)))
        return SoftSightline(float(m.group(1)), float(m.group(2)), **shape, scans=scans)
    raise ValueError(f"unknown along conductance {spec!r}")


class Solver(Protocol):
    """Where the operator lives and how a grounded SPD system on it is solved. `xp`, `sparse`
    and `ndimage` are the device's array, sparse-matrix and image modules (numpy / scipy, or
    cupy / cupyx). prepare(A, coarsening) does the setup once and returns solve(b, rtol, x0) ->
    x on the device, CG from x0 (None: zero) to a relative residual of rtol (exact below SMALL
    unknowns). `coarsening`
    lets a solver reuse one aggregation across a block's systems."""

    @property
    def xp(self): ...

    @property
    def sparse(self): ...

    @property
    def ndimage(self): ...

    def to_host(self, a) -> NDArray: ...

    def prepare(self, A, coarsening: Coarsening) -> Callable: ...


SMALL = 20000
MAXITER = 2000      # rtol 1e-9 takes ~350 K-cycle iterations on a 1M-unknown block


class Coarsening(NamedTuple):
    """What a system offers a solver that coarsens once per block: `mid()` its unknowns' ids
    in the master numbering (every inside non-ground cell x K), `master()` the operator on every
    inside cell (on the device, uniform along-factors), and `cache`, a per-grid dict for what
    the solver derives from it."""
    mid: Callable
    master: Callable
    cache: dict


@dataclass(frozen=True)
class CpuAMG:
    """pyamg smoothed aggregation as a CG preconditioner (Gauss-Seidel smoothing), one core,
    set up afresh for every system."""

    @property
    def xp(self):
        return np

    @property
    def sparse(self):
        return sp

    @property
    def ndimage(self):
        return ndimage

    def to_host(self, a):
        return np.asarray(a)

    def prepare(self, A, coarsening):
        if A.shape[0] < SMALL:
            lu = factorized(A.tocsc())
            return lambda b, rtol, x0=None: lu(b)
        ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric", max_coarse=2000)

        def solve1(b, rtol, x0):
            res: list[float] = []
            x = ml.solve(b, x0=x0, tol=rtol, accel="cg", maxiter=500, residuals=res)
            nb = float(np.linalg.norm(b))
            if res[-1] > rtol * nb * 10:         # relative to b, as pyamg's tol (x0 may be close)
                raise RuntimeError(f"AMG-CG did not converge: {res[-1] / nb:.2e} after "
                                   f"{len(res)}")
            return x

        def solve(b, rtol, x0=None):
            if b.ndim == 1:
                return solve1(b, rtol, x0)
            return np.stack([solve1(b[:, i], rtol, None if x0 is None else x0[:, i])
                             for i in range(b.shape[1])], axis=1)
        return solve


@dataclass(frozen=True)
class GpuAMG:
    """Flexible CG on the GPU preconditioned by an unsmoothed-aggregation K-cycle (Notay: each
    coarse correction is two flexible Krylov steps preconditioned by the next level): damped
    Jacobi smoothing, `sweeps` each side (Gauss-Seidel is sequential), omega = 4 / (3 rho(D^-1
    A)) per level (rho by power iteration, padded 5%), a dense inverse on the coarsest level.
    The aggregates are the block's MASTER coarsening (pyamg's standard aggregation of the
    operator on every inside cell, once per grid; strength theta 0, so the sparsity pattern
    alone) restricted to the system's unknowns, so a system's setup is GPU Galerkin sums only.
    Measured and dropped (NOTES, "Speed"): V-cycles (2.2x the time), strength theta 0.25
    (3-7x), Notay's pairwise aggregation (3-10x), smoothed prolongators (cuSPARSE's products
    need more memory than the card on 5810). Needs cupy and CUDA_PATH (the toolkit
    headers)."""
    sweeps: int = 1
    single: bool = True

    @property
    def xp(self):
        import cupy
        return cupy

    @property
    def sparse(self):
        import cupyx.scipy.sparse
        return cupyx.scipy.sparse

    @property
    def ndimage(self):
        import cupyx.scipy.ndimage
        return cupyx.scipy.ndimage

    def to_host(self, a):
        return self.xp.asnumpy(a)

    def _dinv(self, A, v0=None):
        """(The damped inverse diagonal omega D^-1 of one level, the power iteration's last
        vector): 30 iterations from random, 5 from v0 (the last system's: the same pattern, values
        moved a little)."""
        cp = self.xp
        dinv = 1.0 / A.diagonal()
        v = cp.random.default_rng(0).random(A.shape[0]) if v0 is None else v0
        rho = 1.0
        for _ in range(30 if v0 is None else 5):
            w = dinv * (A @ v)
            rho = float(cp.linalg.norm(w)) / float(cp.linalg.norm(v))
            v = w / cp.linalg.norm(w)
        return dinv * (4.0 / (3.0 * 1.05 * rho)), v

    def _structure(self, A, coarsening: Coarsening) -> list:
        """Per level, everything about the Galerkin coarsening but the values: R, P, and the
        coarse operator's CSR structure with `map`, each fine nonzero's coarse nonzero (A_c =
        T^T A T with T the 0/1 aggregation sums entries by aggregate pair)."""
        cp, cs = self.xp, self.sparse
        levels, ids, ip, ix, n = [], coarsening.mid(), A.indptr, A.indices, A.shape[0]
        for agg in self._aggregates(coarsening):
            # the master aggregates of this level's unknowns, renumbered over those present
            uniq, local = cp.unique(agg[ids], return_inverse=True)
            local = local.ravel()
            nc = len(uniq)
            R = cs.csr_matrix((cp.ones(n), (local, cp.arange(n))), shape=(nc, n))
            rows = cp.searchsorted(ip, cp.arange(len(ix)), side="right") - 1
            keys = local[rows].astype(cp.int64) * nc + local[ix]
            del rows
            ukeys, inv = cp.unique(keys, return_inverse=True)
            del keys
            indptr = cp.zeros(nc + 1, dtype=cp.int32)
            indptr[1:] = cp.cumsum(cp.bincount(ukeys // nc, minlength=nc))
            indices = (ukeys % nc).astype(cp.int32)
            levels.append(dict(R=R, P=R.T.tocsr(), map=inv.ravel().astype(cp.int32),
                               indices=indices, indptr=indptr, nc=nc, v=None))
            ip, ix, n, ids = indptr, indices, nc, uniq
        return levels

    def _aggregates(self, co: Coarsening) -> list:
        """Per level, the master aggregate of each node (device), cached per grid."""
        key = "aggregates"
        if key not in co.cache:
            from pyamg.aggregation.aggregate import standard_aggregation
            from pyamg.strength import symmetric_strength_of_connection
            A = co.master().get().tocsr()
            aggs = []
            while A.shape[0] > 2000:
                Ag = sp.csr_matrix(standard_aggregation(
                    symmetric_strength_of_connection(A, theta=0.0))[0])
                assert (np.diff(Ag.indptr) == 1).all(), "every node is aggregated"
                agg = Ag.indices.astype(np.int64)
                nc = Ag.shape[1]
                co_ = A.tocoo()
                A = sp.csr_matrix((co_.data, (agg[co_.row], agg[co_.col])), shape=(nc, nc))
                aggs.append(self.xp.asarray(agg))
            co.cache[key] = aggs
        return co.cache[key]

    def prepare(self, A, coarsening):
        cp, cs = self.xp, self.sparse
        # cupy's pool keeps every freed block; each system builds new matrices, so without this a
        # process only grows (~20 GB per 5810 run; two big blocks then exhaust the 48 GB card)
        cp.get_default_memory_pool().free_all_blocks()
        if A.shape[0] < SMALL:
            lu = factorized(A.get().tocsc())
            return lambda b, rtol, x0=None: cp.asarray(lu(cp.asnumpy(b)))
        # the hierarchy's structure depends only on A's pattern, which an eps-world system
        # repeats on every call (SIMP's updates, the greedy's tension): kept once it repeats
        key = (A.shape[0], A.nnz, int(A.indices.sum()), int(A.indptr.sum()))
        store = _repeated(coarsening.cache, "galerkin")
        if key in store:
            levels = store[key]
        else:
            levels = self._structure(A, coarsening)
            _remember(coarsening.cache, "galerkin", key, levels)
        lv, Ai = [], A
        for L in levels:
            dinv, L["v"] = self._dinv(Ai, L["v"])
            lv.append(dict(A=Ai, dinv=dinv, R=L["R"], P=L["P"]))
            Ai = cs.csr_matrix((cp.bincount(L["map"], weights=Ai.data,
                                            minlength=len(L["indices"])),
                                L["indices"], L["indptr"]), shape=(L["nc"], L["nc"]))
        coarse = cp.linalg.inv(Ai.toarray())
        if self.single:
            # the cycle in single precision (sums above in double); the outer CG's residual
            # stays in double, so the answer meets rtol as before, only the preconditioner is
            # cheaper (the solves are memory-bound: 8 -> 4 bytes per value)
            f4 = cp.float32
            lv = [dict(A=cs.csr_matrix((L["A"].data.astype(f4), L["A"].indices, L["A"].indptr),
                                       shape=L["A"].shape),
                       dinv=L["dinv"].astype(f4), R=L["R"].astype(f4), P=L["P"].astype(f4))
                  for L in lv]
            coarse = coarse.astype(f4)
        kc = _KCycle(lv, coarse, self.sweeps)

        def solve(b, rtol, x0=None):
            return _pcg(A, lambda r: kc.cycle(0, r.astype(kc.dt)).astype(cp.float64), b, rtol,
                        cp, x0)
        return solve


class _KCycle:
    """A GpuAMG system's preconditioner: the K-cycle over its levels (each a dict of A, dinv,
    R, P) down to the dense inverse `coarse`. A class and not nested closures: `cycle` and
    `krylov2` call each other, and as closures they form a reference cycle that kept every
    system's levels alive until the cycle collector ran (2.6 GB per system on 30796, piling up
    over greedy steps until the 48 GB card ran out; NOTES, "Translucent greedy on 30796")."""

    def __init__(self, lv: list, coarse, sweeps: int):
        self.lv, self.coarse, self.sweeps = lv, coarse, sweeps
        self.dt = coarse.dtype

    @staticmethod
    def dot(a, b):
        """Accumulated in double: in single the K-cycle's two-step coefficients lose
        everything to cancellation (rho2 = bet - gam^2 / rho1) and the solve goes NaN."""
        return (a * b).sum(axis=0, dtype=np.float64)

    def cycle(self, i, b):
        lv = self.lv
        if i == len(lv):
            return self.coarse @ b
        L = lv[i]
        d = L["dinv"] if b.ndim == 1 else L["dinv"][:, None]
        x = d * b
        for _ in range(self.sweeps - 1):
            x = x + d * (b - L["A"] @ x)
        rc = L["R"] @ (b - L["A"] @ x)
        ec = self.krylov2(i + 1, rc) if i + 1 < len(lv) else self.cycle(i + 1, rc)
        x = x + L["P"] @ ec
        for _ in range(self.sweeps):
            x = x + d * (b - L["A"] @ x)
        return x

    def krylov2(self, j, b):
        """Two flexible CG steps on level j from zero, preconditioned by its cycle (Notay)."""
        dot, dt = self.dot, self.dt
        Aj = self.lv[j]["A"]
        c1 = self.cycle(j, b)
        v1 = Aj @ c1
        rho1, a1 = dot(c1, v1), dot(c1, b)
        r1 = b - (a1 / rho1).astype(dt) * v1
        c2 = self.cycle(j, r1)
        v2 = Aj @ c2
        gam, bet, a2 = dot(c2, v1), dot(c2, v2), dot(c2, r1)
        rho2 = bet - gam * gam / rho1
        return ((a1 / rho1 - gam * a2 / (rho1 * rho2)).astype(dt) * c1
                + (a2 / rho2).astype(dt) * c2)


def _pcg(A, M, b, rtol: float, xp, x0=None):
    """Flexible (Polak-Ribiere) preconditioned CG from x0 (None: zero) to ||r|| <= rtol ||b||;
    b may be (n,) or (n, m) (columns solved independently, in lockstep). M may vary between
    calls."""
    nb = xp.linalg.norm(b, axis=0)
    if not bool((nb > 0).any()):
        return xp.zeros_like(b)
    x = xp.zeros_like(b) if x0 is None else x0.copy()
    r = b.copy() if x0 is None else b - A @ x
    if bool((xp.linalg.norm(r, axis=0) <= rtol * nb).all()):
        return x
    z = M(r)
    pdir = z.copy()
    rz = (r * z).sum(axis=0)
    for _ in range(MAXITER):
        Ap = A @ pdir
        pAp = (pdir * Ap).sum(axis=0)
        alpha = xp.where(pAp > 0, rz / xp.where(pAp > 0, pAp, 1.0), 0.0)
        x += alpha * pdir
        r_old = r.copy()
        r -= alpha * Ap
        if bool((xp.linalg.norm(r, axis=0) <= rtol * nb).all()):
            return x
        z = M(r)
        num = (z * (r - r_old)).sum(axis=0)
        pdir = z + xp.where(rz > 0, num / xp.where(rz > 0, rz, 1.0), 0.0) * pdir
        rz = (r * z).sum(axis=0)
    worst = float((xp.linalg.norm(r, axis=0) / xp.where(nb > 0, nb, 1.0)).max())
    raise RuntimeError(f"AMG-CG did not converge: {worst:.2e} after {MAXITER}")


def scans_of(spec: str) -> Scans:
    """`cpu` -> CpuScans(); `gpu` -> GpuScans() (chosen with the solver)."""
    if spec == "cpu":
        return CpuScans()
    if spec == "gpu":
        return GpuScans()
    raise ValueError(f"unknown scans {spec!r}")


def solver_of(spec: str) -> Solver:
    """`cpu` -> CpuAMG(); `gpu` -> GpuAMG()."""
    if spec == "cpu":
        return CpuAMG()
    if spec == "gpu":
        return GpuAMG()
    raise ValueError(f"unknown solver {spec!r}")


@dataclass(frozen=True)
class Params:
    ell_m: float = 2.0       # turning length: metres travelled per radian of heading change
    K: int = 8
    along: AlongConductance = Uniform()
    solver: Solver = CpuAMG()


class Pattern(NamedTuple):
    """Every edge the operator can have on a grid: those among its `inside` cells (any field's
    open cells are inside ones). Per axis k, along[k] = (a, b, line): flat cell indices of each
    edge's ends (b = a + v_k) and of the cells the step crosses (n_line, E), in row-major order
    of a; `cells`: the inside cells, flat, row-major. Arrays live on the solver's device."""
    along: list
    cells: object


def pattern(grid: Grid, K: int, xp) -> Pattern:
    """The grid's Pattern for K axes on device `xp`, built once per grid."""
    key = ("pattern", K, xp.__name__)
    if key not in grid.cache:
        v, _th, _m, _gap = axes(K)
        inside = grid.inside
        ny, nx = inside.shape
        rr, cc = np.nonzero(inside)
        along = []
        for k in range(K):
            dx, dy = int(v[k, 0]), int(v[k, 1])
            lines = [(dx, dy), *_line_cells(dx, dy)]
            ok = np.ones(len(rr), dtype=bool)
            for ix, iy in lines:
                r3, c3 = rr + iy, cc + ix
                o3 = (r3 >= 0) & (r3 < ny) & (c3 >= 0) & (c3 < nx)
                o3[o3] &= inside[r3[o3], c3[o3]]
                ok &= o3
            a = rr[ok] * nx + cc[ok]
            along.append((xp.asarray(a), xp.asarray(a + dy * nx + dx),
                          xp.asarray(np.stack([a + iy * nx + ix for ix, iy in lines[1:]])
                                     if len(lines) > 1 else np.zeros((0, len(a)), np.int64))))
        grid.cache[key] = Pattern(along=along, cells=xp.asarray(rr * nx + cc))
    return grid.cache[key]


def _free_ids(o, xp):
    """Flat (ny * nx,) -> free-cell id (row-major order of o > 0) or -1, and the count."""
    free = o.ravel() > 0
    nf = int(free.sum())
    cell = -xp.ones(free.shape, dtype=xp.int64)
    cell[free] = xp.arange(nf)
    return cell, nf


def along_edges(grid: Grid, open_, p: Params):
    """Per axis k: (k, a, b, w) with a, b open-cell ids (row-major order of `open_ > 0`),
    b = a + v_k, every cell the step crosses open; w is scaled by the SMALLEST open fraction
    among those cells (a gap narrower than a cell conducts in proportion to its width) and by
    the smaller of the two ends' `p.along` factors. Device arrays."""
    xp = p.solver.xp
    v, _th, m, _gap = axes(p.K)
    o = xp.asarray(open_, dtype=xp.float64).ravel()
    cell, _nf = _free_ids(o, xp)
    pat = pattern(grid, p.K, xp)
    F = p.along.layers(open_, grid.h, p.K)
    for k in range(p.K):
        a, b, line = pat.along[k]
        frac = xp.minimum(o[a], o[b])
        for li in line:
            frac = xp.minimum(frac, o[li])
        ok = frac > 0
        a, b, frac = a[ok], b[ok], frac[ok]
        dx, dy = int(v[k, 0]), int(v[k, 1])
        w = frac * (m[k] / float(dx * dx + dy * dy))
        if not np.isscalar(F[k]):
            Fk = xp.asarray(F[k]).ravel()
            w = w * xp.minimum(Fk[a], Fk[b])
        elif F[k] != 1.0:
            w = w * F[k]
        yield k, cell[a], cell[b], w


def crossing(grid: Grid, open_: NDArray[np.float64], sol: Solution,
             p: Params) -> dict[str, float]:
    """How much of the along-heading power flows where streams of different headings share a
    cell. Per free cell, J = sum_k F_k e_k (the net flux vector) and S = sum_k |F_k|;
    cancellation 1 - |J|/S is ~0 for one stream (a little above 0 for a stream spread over
    neighbouring axes), 0.29 for two equal perpendicular streams, 1 for opposed ones. Reported
    power-weighted over cells. (Host arrays: a CPU-solver `p`.)"""
    v, _th, _m, _gap = axes(p.K)
    e = v / np.linalg.norm(v, axis=1)[:, None]
    nc = int((open_ > 0).sum())
    uk = np.zeros((nc, p.K))
    live = sol.unk_cell >= 0
    uk[live] = sol.u.reshape(-1, p.K)
    J = np.zeros((nc, 2))
    S = np.zeros(nc)
    pw = np.zeros(nc)
    for k, a, b, w in along_edges(grid, open_, p):
        F = w * (uk[a, k] - uk[b, k])            # flux from a to b along +v_k
        pe = w * (uk[a, k] - uk[b, k]) ** 2
        for c in (a, b):
            np.add.at(J, c, 0.5 * F[:, None] * e[k][None, :])
            np.add.at(S, c, 0.5 * np.abs(F))
            np.add.at(pw, c, 0.5 * pe)
    ok = S > 0
    canc = np.zeros(nc)
    canc[ok] = 1.0 - np.linalg.norm(J[ok], axis=1) / S[ok]
    tot = pw.sum()
    return dict(mean=float((canc * pw).sum() / tot),
                share_gt_0_2=float(pw[canc > 0.2].sum() / tot),
                share_gt_0_29=float(pw[canc > 0.29].sum() / tot),
                share_gt_0_5=float(pw[canc > 0.5].sum() / tot))


def edges(grid: Grid, open_, p: Params):
    """Every edge of the operator as families (ca, cb, ka, kb, w): free-cell ids and axes of the
    two ends. Along-axis edges per axis, then the turning edges (k, k+1) inside every free cell,
    scaled by its open fraction (its volume). Ground is applied by `operator`. Device arrays."""
    xp = p.solver.xp
    _v, _th, _m, gap = axes(p.K)
    for k, a, b, w in along_edges(grid, open_, p):
        yield a, b, xp.full(len(a), k), xp.full(len(a), k), w
    o = xp.asarray(open_, dtype=xp.float64).ravel()
    vol = o[o > 0]
    c = xp.arange(len(vol))
    h = grid.h
    for k in range(p.K):
        yield (c, c, xp.full(len(c), k), xp.full(len(c), (k + 1) % p.K),
               vol * (h * h / (p.ell_m ** 2 * gap[k])))


def unknown_of(ground, open_, xp=np):
    """Per free cell: its unknown block index (x K + axis), or -1 on ground (u = 0 there)."""
    is_g = xp.asarray(ground)[xp.asarray(open_) > 0]
    unk_cell = -xp.ones(len(is_g), dtype=xp.int64)
    unk_cell[~is_g] = xp.arange(int((~is_g).sum()))
    return unk_cell


def _assemble(grid: Grid, open_, p: Params):
    """(off-diagonal rows, cols, vals; diagonal; free-cell id per grid cell (ny, nx); unknown
    index per free cell or -1) of L over every unknown. An edge with a ground end only adds to
    the other end's diagonal."""
    xp = p.solver.xp
    K = p.K
    o = xp.asarray(open_, dtype=xp.float64)
    cell = _free_ids(o, xp)[0].reshape(o.shape)
    unk_cell = unknown_of(grid.ground, o, xp)
    N = int((unk_cell >= 0).sum()) * K
    rows, cols, vals = [], [], []
    diag = xp.zeros(N)
    for ca, cb, ka, kb, w in edges(grid, o, p):
        ua, ub = unk_cell[ca], unk_cell[cb]
        ia, ib = ua * K + ka, ub * K + kb
        ga, gb = ua < 0, ub < 0
        both = ~ga & ~gb
        rows += [ia[both], ib[both]]
        cols += [ib[both], ia[both]]
        vals += [-w[both], -w[both]]
        diag += xp.bincount(ia[~ga], weights=w[~ga], minlength=N)
        diag += xp.bincount(ib[~gb], weights=w[~gb], minlength=N)
    return xp.concatenate(rows), xp.concatenate(cols), xp.concatenate(vals), diag, cell, unk_cell


def operator(grid: Grid, open_, p: Params):
    """(L restricted to unknowns, free-cell id per grid cell, unknown index per free cell or -1).
    Cells are those with open fraction > 0."""
    xp = p.solver.xp
    r, c, v, diag, cell, unk_cell = _assemble(grid, open_, p)
    N = len(diag)
    ar = xp.arange(N)
    L = p.solver.sparse.coo_matrix((xp.concatenate([v, diag]),
                                    (xp.concatenate([r, ar]), xp.concatenate([c, ar]))),
                                   shape=(N, N)).tocsr()
    return L, cell, unk_cell


def _reach(grid: Grid, o, p: Params):
    """(ny, nx) device mask: open cells in a 4-connected component of open space holding a
    ground cell. The operator's cell graph is exactly 4-connectivity (a diagonal or longer step
    needs every cell it crosses open), so these are the cells with a path to the street."""
    xp = p.solver.xp
    free = o > 0
    lab, n = p.solver.ndimage.label(free)
    ok = xp.zeros(int(n) + 1, dtype=bool)
    ok[lab[free & xp.asarray(grid.ground)]] = True
    ok[0] = False
    return ok[lab]


def grounded(grid: Grid, open_: NDArray[np.float64], p: Params) -> NDArray[np.bool_]:
    """Open cells with a path to the street (ground cells included), on the host."""
    return p.solver.to_host(_reach(grid, p.solver.xp.asarray(open_, dtype=float), p))


def master_operator(grid: Grid, p: Params):
    """The operator on every inside cell (uniform along-factors), on the solver's device: the
    pattern every system's is a restriction of. Unknowns: inside non-ground cells x K,
    row-major."""
    return operator(grid, grid.isub.mean(axis=-1), dataclasses.replace(p, along=Uniform()))[0]


def _csr(solver, vals, rows, cols, n: int, cache: dict):
    """The (n x n) CSR matrix of triplets, duplicates summed. Its structure (a sort of every
    nonzero) is kept for a pattern seen twice (an eps-world system repeats its pattern on every
    call, only the values move; a scoring system's pattern is new each time and is not kept:
    holding it ran a 1 km^2 block out of memory)."""
    xp = solver.xp
    key = (n, len(rows), int(rows.sum()), int(cols.sum()), int((rows * 1000003 ^ cols).sum()))
    store = _repeated(cache, "csr")
    if key in store:
        inv, indices, indptr = store[key]
    else:
        ukeys, inv = xp.unique(rows.astype(xp.int64) * n + cols, return_inverse=True)
        indptr = xp.zeros(n + 1, dtype=xp.int32)
        indptr[1:] = xp.cumsum(xp.bincount(ukeys // n, minlength=n))
        inv, indices = inv.ravel().astype(xp.int32), (ukeys % n).astype(xp.int32)
        _remember(cache, "csr", key, (inv, indices, indptr))
    return solver.sparse.csr_matrix(
        (xp.bincount(inv, weights=vals, minlength=len(indices)), indices, indptr), shape=(n, n))


def _repeated(cache: dict, kind: str) -> dict:
    """The structures of `kind` kept in `cache` (at most one: the pattern that repeats)."""
    return cache.setdefault(kind, {})


def _remember(cache: dict, kind: str, key, value) -> None:
    """Keep `value` for `key` if `key` was seen before (it repeats), replacing what was kept."""
    seen = cache.setdefault(kind + " seen", set())
    if key in seen:
        store = _repeated(cache, kind)
        store.clear()
        store[key] = value
    else:
        seen.add(key)


class System:
    """L on the unknowns that reach ground, set up ONCE for any number of right-hand sides, on
    the solver's device. Unknowns in a component with no ground are dropped (u = 0 there);
    injecting into one is an error."""

    def __init__(self, grid: Grid, open_, p: Params):
        xp = self.xp = p.solver.xp
        self.K = K = p.K
        o = xp.asarray(open_, dtype=xp.float64)
        r, c, v, diag, self.cell, self.unk_cell = _assemble(grid, o, p)
        N = len(diag)
        reach = _reach(grid, o, p)[o > 0]                    # per free cell
        self.keep = xp.repeat(reach[self.unk_cell >= 0], K)
        new = xp.cumsum(self.keep) - 1
        nk = int(self.keep.sum())
        both = self.keep[r]                                   # an edge never leaves its component
        ar = xp.arange(N)[self.keep]
        cache = grid.cache.setdefault(("solver", K, p.ell_m), {})
        self.A = _csr(p.solver, xp.concatenate([v[both], diag[self.keep]]),
                      xp.concatenate([new[r[both]], new[ar]]),
                      xp.concatenate([new[c[both]], new[ar]]), nk, cache)

        def mid():
            mc = -xp.ones(o.size, dtype=xp.int64)
            live = xp.asarray((grid.inside & ~grid.ground).ravel())
            mc[live] = xp.arange(int(live.sum()))
            flat = xp.flatnonzero(o.ravel() > 0)[self.unk_cell >= 0]
            return (mc[flat][:, None] * K + xp.arange(K)[None, :]).ravel()[self.keep]
        self._solver = p.solver.prepare(self.A, Coarsening(
            mid=mid, master=lambda: master_operator(grid, p),
            cache=cache))

    @property
    def n(self) -> int:
        return len(self.keep)

    def solve(self, b, rtol: float = 1e-9, x0=None):
        """Device b (n,) or (n, m) -> device u, zero on dropped unknowns; x0 a starting guess
        over the same unknowns (e.g. the last solve of a nearby field: same numbering when the
        free cells are the same)."""
        if bool((b[~self.keep] != 0).any()):
            raise ValueError("loaded unknowns have no path to ground")
        u = self.xp.zeros(b.shape)
        u[self.keep] = self._solver(b[self.keep], rtol, None if x0 is None else x0[self.keep])
        return u

    def load(self, f_cell, p: Params):
        """Device right-hand side of a per-cell injection, spread over the axes by angular
        share."""
        xp = self.xp
        _, _, m, _ = axes(p.K)
        fc = xp.asarray(f_cell, dtype=xp.float64).ravel()[self.cell.ravel() >= 0]
        live = self.unk_cell >= 0
        b = xp.zeros(self.n)
        idx = (self.unk_cell[live][:, None] * p.K + xp.arange(p.K)[None, :]).ravel()
        b[idx] = (fc[live][:, None] * xp.asarray(m / np.pi)[None, :]).ravel()
        return b


@dataclass(frozen=True, eq=False)
class Solution:
    P: float
    u: NDArray[np.float64]          # per unknown
    cell: NDArray[np.int64]
    unk_cell: NDArray[np.int64]
    n_unknowns: int


def solve(grid: Grid, open_, f_cell: NDArray[np.float64], p: Params,
          system: System | None = None, rtol: float = 1e-9, host: bool = True,
          x0=None) -> Solution:
    """Total escape power. Injection on a closed cell would be a bug: demand cells are open in
    the original geometry and freeing only opens more. Pass `system` (built for this `open_`)
    to reuse its setup. P is quadratic in u, so rtol 1e-5 already gives it to ~1e-10; a ranking
    (tension) is unchanged at 1e-3. `host` False keeps u, cell and unk_cell on the device; x0
    warm-starts the solve (System.solve)."""
    assert not (np.asarray(f_cell)[~(p.solver.to_host(open_) > 0)] > 0).any()
    sy = System(grid, open_, p) if system is None else system
    b = sy.load(f_cell, p)
    u = sy.solve(b, rtol=rtol, x0=x0)
    P = float(b @ u)
    if not host:
        return Solution(P=P, u=u, cell=sy.cell, unk_cell=sy.unk_cell, n_unknowns=sy.n)
    th = p.solver.to_host
    return Solution(P=P, u=th(u), cell=th(sy.cell), unk_cell=th(sy.unk_cell), n_unknowns=sy.n)


def cell_mean_u(sol: Solution, open_: NDArray[np.float64], p: Params) -> NDArray[np.float64]:
    """(ny, nx): each open cell's potential averaged over axes by angular share (the weights the
    injection uses), 0 on ground and closed cells. Host arrays."""
    _, _, m, _ = axes(p.K)
    free = open_ > 0
    nc = int(free.sum())
    uk = np.zeros((nc, p.K))
    live = sol.unk_cell >= 0
    uk[live] = sol.u.reshape(-1, p.K)
    out = np.zeros(free.shape)
    out[free] = uk @ (m / np.pi)
    return out
