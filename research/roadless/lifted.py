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
from typing import Callable, Protocol
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
        """`offset` (fractions of h) keeps cell centres off exactly-aligned geometry."""
        minx, miny, maxx, maxy = boundary.bounds
        xs = np.arange(minx - h + offset[0] * h, maxx + 2 * h, h)
        ys = np.arange(miny - h + offset[1] * h, maxy + 2 * h, h)
        X, Y = np.meshgrid(xs, ys)
        S = 4
        t = ((np.arange(S) + 0.5) / S - 0.5) * h
        ox, oy = np.meshgrid(t, t)
        XS = X[..., None] + ox.ravel()[None, None, :]
        YS = Y[..., None] + oy.ravel()[None, None, :]
        shapely.prepare(boundary)
        isub = shapely.contains_xy(boundary, XS, YS)
        bsub = np.zeros_like(isub)
        if len(footprints):
            fp = shapely.union_all(np.asarray(footprints))
            shapely.prepare(fp)
            bsub[isub] = shapely.contains_xy(fp, XS[isub], YS[isub])
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
        overlaps go to the later footprint). Each footprint is tested only on the sub-sample
        lattice inside its own bounding box: sub-sample column j sits at
        x0 - h/2 + (j + 1/2) h/S (rows likewise), the points `_sub_xy` lays out."""
        S, h = self.S, self.h
        ny, nx = self.inside.shape
        d = h / S
        ox, oy = self.x0 - h / 2, self.y0 - h / 2
        fine = -np.ones((ny * S, nx * S), dtype=np.int64)
        for k, (bx0, by0, bx1, by1) in enumerate(shapely.bounds(polys)):
            j0 = max(int(np.floor((bx0 - ox) / d - 0.5)), 0)
            j1 = min(int(np.ceil((bx1 - ox) / d - 0.5)) + 1, nx * S)
            i0 = max(int(np.floor((by0 - oy) / d - 0.5)), 0)
            i1 = min(int(np.ceil((by1 - oy) / d - 0.5)) + 1, ny * S)
            if j1 <= j0 or i1 <= i0:
                continue
            X, Y = np.meshgrid(ox + (np.arange(j0, j1) + 0.5) * d,
                               oy + (np.arange(i0, i1) + 0.5) * d)
            fine[i0:i1, j0:j1][shapely.contains_xy(polys[k], X, Y)] = k
        lab = fine.reshape(ny, S, nx, S).transpose(0, 2, 1, 3).reshape(ny, nx, S * S)
        return np.where(self.bsub, lab, -1)

    def mask_of(self, geom) -> NDArray[np.bool_]:
        out = np.zeros_like(self.inside)
        if geom is None or geom.is_empty:
            return out
        shapely.prepare(geom)
        X, Y = self.xy[..., 0], self.xy[..., 1]
        out[self.inside] = shapely.contains_xy(geom, X[self.inside], Y[self.inside])
        return out


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

    @property
    def xp(self):
        import cupy as cp
        return cp

    def _kernels(self):
        import cupy as cp
        mod = cp.RawModule(code=_SCAN_CU)
        return mod.get_function("soft_fb"), mod.get_function("soft_vjp")

    def _starts(self, ny: int, nx: int, dx: int, dy: int):
        """Cells whose predecessor (r - dy, c - dx) is off the grid: one per line."""
        cp = self.xp
        r, c = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        rp, cpv = r - dy, c - dx
        st = ~((rp >= 0) & (rp < ny) & (cpv >= 0) & (cpv < nx))
        return (cp.asarray(r[st].astype(np.int32)), cp.asarray(c[st].astype(np.int32)),
                int(st.sum()))

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
    Differentiable: `vjp` is exact."""
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

    def runs(self, open_, h):
        """Yield (d, R_d) per fine direction, R_d the expected free path in metres."""
        for i, F, B in self.paths(open_, h):
            yield i, F + B

    def paths(self, open_, h):
        """Yield (d, F_d, B_d): the expected free path forward and back. F x B is the measure of
        the segments along d that pass through the cell (pairs of a start behind and an end
        ahead); its sum along a line of length L is L^2 / 2."""
        dirs, _ang = self._dirs()
        od = self.scans.to_dev(open_)
        for i, (dx, dy) in enumerate(dirs):
            offs = np.array(_line_cells(dx, dy), dtype=np.int64).reshape(-1, 2)
            F, B, _tf, _tb = self.scans.fb(od, dx, dy, offs, float(np.hypot(dx, dy) * h),
                                           self.kappa)
            yield i, self.scans.to_host(F), self.scans.to_host(B)

    def layers(self, open_, h, K):
        key = (hashlib.blake2b(np.ascontiguousarray(open_).tobytes(), digest_size=16).digest(),
               h, K)
        if key in self._cache:
            return self._cache[key]
        xp = self.scans.xp
        dirs, ang = self._dirs()
        W = _angle_weights(K, ang)
        od = self.scans.to_dev(open_)
        out = xp.ones((K, *open_.shape))
        for i, (dx, dy) in enumerate(dirs):
            offs = np.array(_line_cells(dx, dy), dtype=np.int64).reshape(-1, 2)
            F, B, _tf, _tb = self.scans.fb(od, dx, dy, offs, float(np.hypot(dx, dy) * h),
                                           self.kappa)
            g = self.g(F + B)
            for k in np.flatnonzero(W[:, i]):
                out[k] += (self.beta * W[k, i]) * g
        res = self.scans.to_host(out)
        self._cache.clear()
        self._cache[key] = res
        return res

    def vjp(self, open_, h, K, A):
        xp = self.scans.xp
        dirs, ang = self._dirs()
        W = _angle_weights(K, ang)
        od = self.scans.to_dev(open_)
        Ad = self.scans.to_dev(A)
        out = xp.zeros(open_.shape)
        for i, (dx, dy) in enumerate(dirs):
            offs = np.array(_line_cells(dx, dy), dtype=np.int64).reshape(-1, 2)
            s = float(np.hypot(dx, dy) * h)
            F, B, Tf, Tb = self.scans.fb(od, dx, dy, offs, s, self.kappa)
            G = self.beta * self.dg(F + B) * xp.tensordot(xp.asarray(W[:, i]), Ad, axes=1)
            self.scans.vjp_add(dx, dy, offs, s, self.kappa, F, B, Tf, Tb, G, out)
        return self.scans.to_host(out)


def along_of(spec: str, scans: Scans = CpuScans()) -> AlongConductance:
    """`uni` -> Uniform(); `sl3` -> Sightline(beta 3); `ss10k2` -> SoftSightline(beta 10,
    kappa 2); `ss30k2n2r30` -> SoftSightline(beta 30, kappa 2, hill 2, r0 30 m)."""
    if spec == "uni":
        return Uniform()
    if spec.startswith("sl"):
        return Sightline(float(spec[2:]))
    m = re.fullmatch(r"ss([0-9.]+)k([0-9.]+)(?:n([0-9.]+)r([0-9.]+))?", spec)
    if m:
        return SoftSightline(float(m.group(1)), float(m.group(2)),
                             *(() if m.group(3) is None else (float(m.group(4)),
                                                              float(m.group(3)))),
                             scans=scans)
    raise ValueError(f"unknown along conductance {spec!r}")


class Solver(Protocol):
    """How a grounded SPD system is solved: prepare(A) does the setup once and returns
    solve(b, rtol) -> x, CG to a relative residual of rtol (exact below SMALL unknowns)."""

    def prepare(self, A) -> Callable[[NDArray[np.float64], float], NDArray[np.float64]]: ...


SMALL = 20000


def _direct(A):
    lu = factorized(A.tocsc())
    return lambda b, rtol: lu(b)


@dataclass(frozen=True)
class CpuAMG:
    """pyamg smoothed aggregation as a CG preconditioner (Gauss-Seidel smoothing), one core."""

    def prepare(self, A):
        if A.shape[0] < SMALL:
            return _direct(A)
        ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric", max_coarse=2000)

        def solve(b, rtol):
            res: list[float] = []
            x = ml.solve(b, tol=rtol, accel="cg", maxiter=500, residuals=res)
            if res[-1] > rtol * res[0] * 10:
                raise RuntimeError(f"AMG-CG did not converge: {res[-1] / res[0]:.2e} after "
                                   f"{len(res)}")
            return x
        return solve


@dataclass(frozen=True)
class GpuAMG:
    """The same smoothed-aggregation hierarchy (built by pyamg on the CPU), run on the GPU: CG
    preconditioned by a symmetric V-cycle with damped-Jacobi smoothing (Gauss-Seidel is
    sequential), omega = 4 / (3 rho(D^-1 A)) per level (rho by power iteration on the GPU, padded
    5%), a dense inverse on the coarsest level. The prolongators use pyamg's 'local' weighting.
    Needs cupy and CUDA_PATH (the toolkit headers)."""
    sweeps: int = 2
    smooth: object = None      # prolongator smoothing: None (unsmoothed aggregation) or a pyamg spec

    def prepare(self, A):
        if A.shape[0] < SMALL:
            return _direct(A)
        import cupy as cp
        import cupyx.scipy.sparse as cs
        # the pool keeps every freed block for reuse; rounds build new hierarchies, so without
        # this a process only grows (~20 GB per 5810 run by round 5)
        cp.get_default_memory_pool().free_all_blocks()
        # unsmoothed by default: smoothing the prolongators costs scipy's slow BSR
        # sum_duplicates (7 of 11 s on 5618 at h 0.5), and the extra CG iterations it would save
        # cost milliseconds on the GPU
        ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric", max_coarse=2000,
                                               smooth=self.smooth,
                                               presmoother=None, postsmoother=None)
        lv = []
        for i, L in enumerate(ml.levels):
            Ai = L.A.tocsr()
            Ag = cs.csr_matrix(Ai)
            dinv = cp.asarray(1.0 / Ai.diagonal())
            # rho(D^-1 A) by power iteration on the GPU
            v = cp.random.default_rng(0).random(Ai.shape[0])
            for _ in range(30):
                w = dinv * (Ag @ v)
                rho = float(cp.linalg.norm(w)) / float(cp.linalg.norm(v))
                v = w / cp.linalg.norm(w)
            lv.append(dict(A=Ag, dinv=dinv * (4.0 / (3.0 * 1.05 * rho)),
                           P=cs.csr_matrix(L.P.tocsr()) if i < len(ml.levels) - 1 else None,
                           R=cs.csr_matrix(L.R.tocsr()) if i < len(ml.levels) - 1 else None))
        coarse = cp.asarray(np.linalg.inv(ml.levels[-1].A.toarray()))
        sweeps = self.sweeps

        def vcycle(i, b):
            L = lv[i]
            if L["P"] is None:
                return coarse @ b
            x = L["dinv"] * b
            for _ in range(sweeps - 1):
                x = x + L["dinv"] * (b - L["A"] @ x)
            xc = vcycle(i + 1, L["R"] @ (b - L["A"] @ x))
            x = x + L["P"] @ xc
            for _ in range(sweeps):
                x = x + L["dinv"] * (b - L["A"] @ x)
            return x

        A0 = lv[0]["A"]

        def solve(b, rtol):
            bg = cp.asarray(b)
            x = cp.zeros_like(bg)
            r = bg.copy()
            nb = float(cp.linalg.norm(bg))
            if nb == 0.0:
                return np.zeros_like(b)
            z = vcycle(0, r)
            pdir = z.copy()
            rz = float(r @ z)
            for it in range(500):
                Ap = A0 @ pdir
                alpha = rz / float(pdir @ Ap)
                x += alpha * pdir
                r -= alpha * Ap
                if float(cp.linalg.norm(r)) <= rtol * nb:
                    return cp.asnumpy(x)
                z = vcycle(0, r)
                rz_new = float(r @ z)
                pdir = z + (rz_new / rz) * pdir
                rz = rz_new
            raise RuntimeError(f"GPU AMG-CG did not converge: {float(cp.linalg.norm(r)) / nb:.2e}"
                               f" after 500")
        return solve


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


def along_edges(open_: NDArray[np.float64], h: float, p: Params):
    """Per axis k: (k, a, b, w) with a, b open-cell ids (row-major order of `open_ > 0`),
    b = a + v_k, every cell the step crosses open; w is scaled by the SMALLEST open fraction
    among those cells (a gap narrower than a cell conducts in proportion to its width) and by
    the smaller of the two ends' `p.along` factors."""
    v, _th, m, _gap = axes(p.K)
    free = open_ > 0
    ny, nx = free.shape
    cell = -np.ones((ny, nx), dtype=np.int64)
    cell[free] = np.arange(int(free.sum()))
    rr, cc = np.nonzero(free)
    F = p.along.layers(open_, h, p.K)
    for k in range(p.K):
        dx, dy = int(v[k, 0]), int(v[k, 1])
        r2, c2 = rr + dy, cc + dx
        ok = (r2 >= 0) & (r2 < ny) & (c2 >= 0) & (c2 < nx)
        ok[ok] &= free[r2[ok], c2[ok]]
        frac = np.minimum(open_[rr, cc], open_[np.clip(r2, 0, ny - 1), np.clip(c2, 0, nx - 1)])
        for (ix, iy) in _line_cells(dx, dy):
            r3, c3 = rr + iy, cc + ix
            o3 = (r3 >= 0) & (r3 < ny) & (c3 >= 0) & (c3 < nx)
            o3[o3] &= free[r3[o3], c3[o3]]
            ok &= o3
            frac = np.minimum(frac, open_[np.clip(r3, 0, ny - 1), np.clip(c3, 0, nx - 1)])
        a = cell[rr[ok], cc[ok]]
        b = cell[r2[ok], c2[ok]]
        fk = np.broadcast_to(F[k], free.shape)
        boost = np.minimum(fk[rr[ok], cc[ok]], fk[r2[ok], c2[ok]])
        yield k, a, b, frac[ok] * boost * (m[k] / float(dx * dx + dy * dy))


def crossing(open_: NDArray[np.float64], ground: NDArray[np.bool_], sol: Solution,
             h: float, p: Params) -> dict[str, float]:
    """How much of the along-heading power flows where streams of different headings share a
    cell. Per free cell, J = sum_k F_k e_k (the net flux vector) and S = sum_k |F_k|;
    cancellation 1 - |J|/S is ~0 for one stream (a little above 0 for a stream spread over
    neighbouring axes), 0.29 for two equal perpendicular streams, 1 for opposed ones. Reported
    power-weighted over cells."""
    v, _th, _m, _gap = axes(p.K)
    e = v / np.linalg.norm(v, axis=1)[:, None]
    nc = int((open_ > 0).sum())
    uk = np.zeros((nc, p.K))
    live = sol.unk_cell >= 0
    uk[live] = sol.u.reshape(-1, p.K)
    J = np.zeros((nc, 2))
    S = np.zeros(nc)
    pw = np.zeros(nc)
    for k, a, b, w in along_edges(open_, h, p):
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


def edges(open_: NDArray[np.float64], h: float, p: Params):
    """Every edge of the operator as families (ca, cb, ka, kb, w): free-cell ids and axes of the
    two ends. Along-axis edges per axis, then the turning edges (k, k+1) inside every free cell,
    scaled by its open fraction (its volume). Ground is applied by `operator`."""
    _v, _th, _m, gap = axes(p.K)
    for k, a, b, w in along_edges(open_, h, p):
        yield a, b, np.full(len(a), k), np.full(len(a), k), w
    c = np.arange(int((open_ > 0).sum()))
    vol = open_[open_ > 0]
    for k in range(p.K):
        yield (c, c, np.full(len(c), k), np.full(len(c), (k + 1) % p.K),
               vol * (h * h / (p.ell_m ** 2 * gap[k])))


def unknown_of(ground: NDArray[np.bool_], open_: NDArray[np.float64]) -> NDArray[np.int64]:
    """Per free cell: its unknown block index (x K + axis), or -1 on ground (u = 0 there)."""
    is_g = ground[open_ > 0]
    unk_cell = -np.ones(len(is_g), dtype=np.int64)
    unk_cell[~is_g] = np.arange(int((~is_g).sum()))
    return unk_cell


def operator(open_: NDArray[np.float64], ground: NDArray[np.bool_], h: float, p: Params):
    """(L restricted to unknowns, free-cell id per grid cell, unknown index per free cell or -1).
    Cells are those with open fraction > 0; an edge with a ground end only adds to the other
    end's diagonal."""
    K = p.K
    free = open_ > 0
    cell = -np.ones(free.shape, dtype=np.int64)
    cell[free] = np.arange(int(free.sum()))
    unk_cell = unknown_of(ground, open_)
    N = int((unk_cell >= 0).sum()) * K
    rows, cols, vals = [], [], []
    diag = np.zeros(N)
    for ca, cb, ka, kb, w in edges(open_, h, p):
        ua, ub = unk_cell[ca], unk_cell[cb]
        ia, ib = ua * K + ka, ub * K + kb
        ga, gb = ua < 0, ub < 0
        both = ~ga & ~gb
        rows.append(ia[both]); cols.append(ib[both]); vals.append(-w[both])
        rows.append(ib[both]); cols.append(ia[both]); vals.append(-w[both])
        diag += np.bincount(ia[~ga], weights=w[~ga], minlength=N)
        diag += np.bincount(ib[~gb], weights=w[~gb], minlength=N)
    rows.append(np.arange(N)); cols.append(np.arange(N)); vals.append(diag)
    L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N, N))
    return L, cell, unk_cell


def grounded(grid: Grid, open_: NDArray[np.float64], p: Params) -> NDArray[np.bool_]:
    """Open cells with a path to the street (ground cells included)."""
    free = open_ > 0
    L, cell, unk_cell = operator(open_, grid.ground, grid.h, p)
    n_comp, lab = sp.csgraph.connected_components(L, directed=False)
    dg = np.asarray(L.sum(axis=1)).ravel()
    ok_comp = np.zeros(n_comp, bool)
    ok_comp[np.unique(lab[dg > 1e-12])] = True
    rr, cc = np.nonzero(free)
    out = np.zeros_like(free)
    live = unk_cell >= 0
    good = np.ones(len(rr), bool)
    good[live] = ok_comp[lab[unk_cell[live] * p.K]]
    out[rr[good], cc[good]] = True
    return out


class System:
    """L on the unknowns that reach ground, factorized or AMG-preconditioned ONCE, for any number
    of right-hand sides. Unknowns in a component with no ground are dropped (u = 0 there);
    injecting into one is an error."""

    def __init__(self, grid: Grid, open_: NDArray[np.float64], p: Params):
        self.L, self.cell, self.unk_cell = operator(open_, grid.ground, grid.h, p)
        n_comp, lab = sp.csgraph.connected_components(self.L, directed=False)
        grounded = np.zeros(n_comp, bool)
        dg = np.asarray(self.L.sum(axis=1)).ravel()     # > 0 where an edge leaves to ground
        grounded[np.unique(lab[dg > 1e-12])] = True
        self.keep = grounded[lab]
        A = self.L[self.keep][:, self.keep].tocsr() if n_comp > 1 else self.L
        self.A = A
        self._solver = p.solver.prepare(A)

    @property
    def n(self) -> int:
        return self.L.shape[0]

    def solve(self, b: NDArray[np.float64], rtol: float = 1e-9) -> NDArray[np.float64]:
        if (b[~self.keep] != 0).any():
            raise ValueError(f"{(b[~self.keep] != 0).sum()} loaded unknowns have no path to ground")
        u = np.zeros(self.n)
        u[self.keep] = self._solver(b[self.keep], rtol)
        return u

    def load(self, f_cell: NDArray[np.float64], p: Params) -> NDArray[np.float64]:
        """Right-hand side of a per-cell injection, spread over the axes by angular share."""
        _, _, m, _ = axes(p.K)
        free = self.cell >= 0
        fc = f_cell[free]
        live = self.unk_cell >= 0
        b = np.zeros(self.n)
        idx = (self.unk_cell[live][:, None] * p.K + np.arange(p.K)[None, :]).ravel()
        b[idx] = (fc[live][:, None] * (m / np.pi)[None, :]).ravel()
        return b


@dataclass(frozen=True, eq=False)
class Solution:
    P: float
    u: NDArray[np.float64]          # per unknown
    cell: NDArray[np.int64]
    unk_cell: NDArray[np.int64]
    n_unknowns: int


def solve(grid: Grid, open_: NDArray[np.float64], f_cell: NDArray[np.float64], p: Params,
          system: System | None = None, rtol: float = 1e-9) -> Solution:
    """Total escape power. Injection on a closed cell would be a bug: demand cells are open in
    the original geometry and freeing only opens more. Pass `system` (built for this `open_`)
    to reuse its factorization. P is quadratic in u, so rtol 1e-5 already gives it to ~1e-10;
    a ranking (tension) is unchanged at 1e-3."""
    assert not (f_cell[~(open_ > 0)] > 0).any()
    sy = System(grid, open_, p) if system is None else system
    b = sy.load(f_cell, p)
    u = sy.solve(b, rtol=rtol)
    return Solution(P=float(b @ u), u=u, cell=sy.cell, unk_cell=sy.unk_cell, n_unknowns=sy.n)


def cell_mean_u(sol: Solution, open_: NDArray[np.float64], p: Params) -> NDArray[np.float64]:
    """(ny, nx): each open cell's potential averaged over axes by angular share (the weights the
    injection uses), 0 on ground and closed cells."""
    _, _, m, _ = axes(p.K)
    free = open_ > 0
    nc = int(free.sum())
    uk = np.zeros((nc, p.K))
    live = sol.unk_cell >= 0
    uk[live] = sol.u.reshape(-1, p.K)
    out = np.zeros(free.shape)
    out[free] = uk @ (m / np.pi)
    return out
