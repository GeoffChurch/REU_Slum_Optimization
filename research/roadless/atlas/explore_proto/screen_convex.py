"""The screen's most anomalous convex regions, blocks allowed to be cut (owner 2026-11-03), for the
Demo Lineup artifact's height map: where screen_clusters.py scans clusters of whole blocks, this
scans regions of the plane, a region any set of raster cells that is convex in the sense below.

The field. Each city's blocks (the site's ScreenMap bundle) carry a weight, their buildings n, and
a value, their log10 score (depth_density_proxy sqrt(nA)/P n/A, or density_compactness n/P^2; a
score of 0 at the city's 1st percentile), as in screen_clusters.py. Rasterised at PIX m (per city,
so a mean block spans ~15 cells: Cape Town's blocks average 0.15 km^2, 100 m; Nairobi's 0.04 km^2,
50 m; the same PIX for the regions and their null) in a frame rotated by each of ANGLES,
re-rasterised from the polygons at each angle (a cell is the block containing its centre,
nearest-neighbour sampling of the vector data, not a resampled raster): a cell carries its
block's buildings over the block's cells (W) and that times the block's log score (S); a block no
cell centre falls in puts all its buildings in the cell of its point_on_surface, so every angle's
raster holds every building. Empty ground (streets, the sea,
land no block covers) carries W = S = 0: it neither helps nor hurts a region.

Regions. An orthogonally convex region: every row and every column of the raster meets it in one
run, and it is connected; equivalently a run of rows, each an interval [l, r], consecutive
intervals overlapping, the left ends non-increasing then non-decreasing and the right ends
non-decreasing then non-increasing (Yoda, Fukuda, Morishita and Tokuyama 1997's rectilinear
convex regions). True convexity is approximated by solving at each angle of ANGLES (orthogonal
convexity is invariant under a quarter turn, so [0, 90) covers every orientation) and keeping the
best: the cells whose centres a convex region holds are orthogonally convex at any angle (when
connected), so each angle's optimum is at least any convex region's raster at that angle, and an
orthogonally convex region is a staircase-sided approximation of a convex one.

The exact solver (numba, `forward`): maximise the sum of a cell value g over orthogonally convex
regions, by a DP down the rows. Its state at row i: the region's last row is i with interval
[l, r], in phase q = (left, right), left 0 while the left ends may still move left (non-increasing)
and 1 once one moved right, right 0 while the right ends may still move right and 1 once one moved
left. F_i[q][l][r] = the best region so ended = rowsum_i(l, r) + max(0, the best F_{i-1}[p][l'][r']
with p before q in the automaton TRANS (p <= q componentwise) and [l', r'] related to [l, r] as q
says: q=00 [l', r'] inside [l, r]; q=11 [l, r] inside [l', r']; q=10 l' <= l <= r' <= r; q=01
l <= l' <= r <= r'). Each of the four range-maxima is one pass over the l <= r pairs with a
row buffer, so a row is O(C^2) and a solve O(R C^2) (the raster transposed so C <= R), memory
O(C^2); branch-free selects, ~6 ns per (row, interval, phase). Beside each value the DP carries
the argmax region's W, all a sweep needs. The region itself is recovered by `trace`: the forward
pass keeping every k = ceil(sqrt(R))-th row's F, then back from the best end state, each row's
predecessor the argmax of the previous row's F over the states its phase allows, a checkpoint's
segment of rows recomputed as the walk reaches it (twice the work, O(sqrt(R) C^2) memory).

Two objectives:
  excess: S - lambda0 W with lambda0 = log10 of the metric's shipped floor, the buildings'
    excess in decades over the floor (screen_shapes.py's excess weights by area; this weights by
    buildings, the scan's weights). One solve per angle.
  scan: the Gaussian upper-level-set scan statistic of screen_clusters.py, llr = W_Z W_O / W
    (m_Z - m_O)^2 / (2 sigma^2) for m_Z > m_O, a region at most MAX_SHARE of the city's buildings.
    llr is convex in a region's (W, S) and, for m_Z > m_O, increasing in S, so its maximiser is a
    vertex of the upper boundary of the achievable (W, S) set, which maximises S - lambda W for some
    lambda: a sweep of lambda over the building-weighted quantiles LAM_Q of the blocks' log scores
    (1 - geomspace(0.95, 0.002, 19): the 0.05 to 0.998 quantiles, denser at the top) and lambda0,
    20 lambdas (not 30: a Cape Town sweep is 120 solves of 6-15 s), llr of each lambda's optimum
    at each angle, the best kept. An optimum over the cap is
    dropped at its lambda (scores 0), so under the cap the sweep is a heuristic: the capped
    maximiser need not maximise S - lambda W for any lambda (the check measures the gap).
Top TOP regions per objective, disjoint: after each, every cell of every angle's raster whose
square overlaps the region (positive area) is forbidden (an interval holding one scores -inf),
and the whole search re-run. A reported region's outline is traced with a tie-break of EPS per
cell (the smallest of equal regions, so empty ground does not ride along); its stat is its own.

Significance: the blocks' log scores shuffled among the blocks (their weights and geometry kept;
NPERM shuffles from SeedSequence(SEED): Cape Town 19, so its p is at least 0.05, Nairobi 99; at
load ~20 a solve took 2-3.4 s on Nairobi's 50 m rasters and 9-60 s on Cape Town's 100 m ones, the
run 74 min and 159 min on 16 workers), the whole sweep re-run (every angle and lambda, sigma^2
recomputed), each shuffle's largest llr and largest excess kept; a region's p = (1 + #null
maxima >= its stat) / (1 + NPERM), Kulldorff's rule for secondary clusters too (screen_clusters's
rule).

    S=research/roadless/atlas/explore_proto/screen_convex.py   # from the repo root, PYTHONPATH=.
    uv run python $S ~/src/wt_demo <out dir>     # convex_<city>.json per city, both metrics
    uv run python $S check [fault]               # the DP and the sweep against brute force;
                                                 # faults: re-widen, no-switch (the DP), s-sign
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"           # the pool's workers single-threaded (the machine is shared)

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import NamedTuple  # noqa: E402

import numpy as np  # noqa: E402
import shapely  # noqa: E402
from numba import njit  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy import sparse  # noqa: E402

PIX = {"capetown": 100.0, "nairobi": 50.0}     # m: the raster, per city (the solve is O(R C^2))
ANGLES = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0)   # deg: the raster frames
LAM_Q = 1.0 - np.geomspace(0.95, 0.002, 19)     # the sweep's lambdas: these quantiles (+ lambda0)
MAX_SHARE = 0.5        # a scanned region's buildings at most this share of the city's
TOP = 3                # regions per objective
NPERM = {"capetown": 19, "nairobi": 99}        # shuffles for p (one sweep each)
SEED = 0
WORKERS = 16
EPS = 1e-6             # per cell (decades x buildings): the traced outline's tie-break only
CITIES = ("nairobi", "capetown")
METRICS = {"depth_density_proxy": lambda n, A, P: np.sqrt(n * A) / P * n / A,
           "density_compactness": lambda n, A, P: n / P ** 2}
# the phase automaton: TRANS[p, q], phase p (0 = 00, 1 = 01, 2 = 10, 3 = 11; left, right) may
# precede phase q: a phase's ends never return to moving outwards
TRANS = np.array([[1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.bool_)

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
B1 = NDArray[np.bool_]
NEG = -np.inf


# ---------------------------------------------------------------- the DP

@njit(cache=True, inline="always")
def _src(Fv: F64, Fw: F64, trans: B1, q: int, lo: int, r: int, v: float, w: float
         ) -> tuple[float, float]:
    """(v, w) improved by the best previous-row state at [l, r] that may precede q: its value
    and W. Selects, not branches: the comparisons are data-dependent (branches cost 5x)."""
    for p in range(4):
        if trans[p, q]:
            x = Fv[p, lo, r]
            c = x > v
            v = x if c else v
            w = Fw[p, lo, r] if c else w
    return v, w


@njit(cache=True, inline="always")
def _max(v: float, w: float, x: float, xw: float) -> tuple[float, float]:
    c = x > v
    return x if c else v, xw if c else w


@njit(cache=True, inline="always")
def _emit(q: int, lo: int, r: int, dv: float, dw: float, Nv: F64, Nw: F64, P: F64, PW: F64,
          PF: I64) -> float:
    """F_i[q][l][r] from its best admissible predecessor (value dv, W dw); -inf over a
    forbidden cell. No early return (an inlined return costs 10x)."""
    s, w = P[r + 1] - P[lo], PW[r + 1] - PW[lo]
    c = dv > 0.0
    v = s + dv if c else s
    v = NEG if PF[r + 1] - PF[lo] > 0 else v
    Nv[q, lo, r] = v
    Nw[q, lo, r] = w + dw if c else w
    return v


@njit(cache=True)
def _row(Fv: F64, Fw: F64, Nv: F64, Nw: F64, P: F64, PW: F64, PF: I64, i: int, trans: B1,
         bv: F64, bw: F64, rec: F64, irec: I64) -> None:
    """Row i of the DP: N (this row's states) from F (the previous row's). Each target phase
    one pass over l <= r; its range maximum carried with its W in a row buffer (b*) and a
    running register (r*). The row's best state into rec, irec if it beats theirs."""
    C = Fv.shape[1]
    best, bq, bl, br = rec[0], -1, -1, -1
    # q = 0, [l', r'] inside [l, r]: D[l][r] = max(G[l][r], D[l+1][r], D[l][r-1])
    for lo in range(C - 1, -1, -1):
        rv, rw = NEG, 0.0
        for r in range(lo, C):
            dv, dw = _src(Fv, Fw, trans, 0, lo, r, NEG, 0.0)
            if r > lo:
                dv, dw = _max(dv, dw, rv, rw)
                dv, dw = _max(dv, dw, bv[r], bw[r])
            bv[r], bw[r] = dv, dw
            rv, rw = dv, dw
            v = _emit(0, lo, r, dv, dw, Nv, Nw, P, PW, PF)
            if v > best:
                best, bq, bl, br = v, 0, lo, r
    # q = 3, [l, r] inside [l', r']: D[l][r] = max(G[l][r], D[l-1][r], D[l][r+1])
    for lo in range(C):
        rv, rw = NEG, 0.0
        for r in range(C - 1, lo - 1, -1):
            dv, dw = _src(Fv, Fw, trans, 3, lo, r, NEG, 0.0)
            dv, dw = _max(dv, dw, rv, rw)
            if lo > 0:
                dv, dw = _max(dv, dw, bv[r], bw[r])
            bv[r], bw[r] = dv, dw
            rv, rw = dv, dw
            v = _emit(3, lo, r, dv, dw, Nv, Nw, P, PW, PF)
            if v > best:
                best, bq, bl, br = v, 3, lo, r
    # q = 2, l' <= l <= r' <= r: A[l][r'] = max(A[l-1][r'], G[l][r']); D = max_{r' in [l, r]} A
    for lo in range(C):
        rv, rw = NEG, 0.0
        for r in range(lo, C):
            av, aw = _src(Fv, Fw, trans, 2, lo, r, NEG, 0.0)
            if lo > 0:
                av, aw = _max(av, aw, bv[r], bw[r])
            bv[r], bw[r] = av, aw
            rv, rw = _max(rv, rw, av, aw)
            v = _emit(2, lo, r, rv, rw, Nv, Nw, P, PW, PF)
            if v > best:
                best, bq, bl, br = v, 2, lo, r
    # q = 1, l <= l' <= r <= r': B[l'][r] = max(B[l'][r+1], G[l'][r]); D = max(D[l+1][r], B)
    for lo in range(C - 1, -1, -1):
        sv, sw = NEG, 0.0
        for r in range(C - 1, lo - 1, -1):
            sv, sw = _src(Fv, Fw, trans, 1, lo, r, sv, sw)
            dv, dw = sv, sw
            if r > lo:
                dv, dw = _max(dv, dw, bv[r], bw[r])
            bv[r], bw[r] = dv, dw
            v = _emit(1, lo, r, dv, dw, Nv, Nw, P, PW, PF)
            if v > best:
                best, bq, bl, br = v, 1, lo, r
    if bq >= 0:
        rec[0], rec[1] = best, Nw[bq, bl, br]
        irec[0], irec[1], irec[2], irec[3] = i, bq, bl, br


@njit(cache=True)
def _prefix(gv: F64, gw: F64, forb: B1, i: int, P: F64, PW: F64, PF: I64) -> None:
    C = gv.shape[1]
    for c in range(C):
        P[c + 1] = P[c] + gv[i, c]
        PW[c + 1] = PW[c] + gw[i, c]
        PF[c + 1] = PF[c] + (1 if forb[i, c] else 0)


@njit(cache=True)
def _buffers(C: int) -> tuple[F64, F64, F64, F64, I64, F64, I64]:
    return (np.full(C, NEG), np.zeros(C), np.zeros(C + 1), np.zeros(C + 1),
            np.zeros(C + 1, np.int64), np.array([NEG, 0.0]), np.full(4, -1, np.int64))


@njit(cache=True)
def forward(gv: F64, gw: F64, forb: B1, trans: B1) -> tuple[F64, I64]:
    """The best orthogonally convex region's value and W (rec), and (irec) its last row,
    phase and last interval [l, r]."""
    R, C = gv.shape
    Fv, Nv = np.full((4, C, C), NEG), np.full((4, C, C), NEG)
    Fw, Nw = np.zeros((4, C, C)), np.zeros((4, C, C))
    bv, bw, P, PW, PF, rec, irec = _buffers(C)
    for i in range(R):
        _prefix(gv, gw, forb, i, P, PW, PF)
        _row(Fv, Fw, Nv, Nw, P, PW, PF, i, trans, bv, bw, rec, irec)
        Fv, Nv = Nv, Fv
        Fw, Nw = Nw, Fw
    return rec, irec


@njit(cache=True)
def _back(Fv: F64, trans: B1, q: int, lo: int, r: int) -> tuple[float, int, int, int]:
    """The best state of the previous row that may precede (q, l, r)."""
    C = Fv.shape[1]
    a0, a1 = (lo, r) if q < 2 else (0, lo)
    best, bp_, ba_, bb_ = NEG, -1, -1, -1
    for a in range(a0, a1 + 1):
        if q == 0:
            b0, b1 = a, r
        elif q == 1:
            b0, b1 = r, C - 1
        elif q == 2:
            b0, b1 = lo, r
        else:
            b0, b1 = r, C - 1
        for b in range(max(a, b0), b1 + 1):
            for p in range(4):
                if trans[p, q] and Fv[p, a, b] > best:
                    best, bp_, ba_, bb_ = Fv[p, a, b], p, a, b
    return best, bp_, ba_, bb_


def trace(gv: F64, gw: F64, forb: B1) -> tuple[B1, float, float]:
    """The best orthogonally convex region (a mask), its value and W: the forward pass keeping
    every k = ceil(sqrt(R))-th row's F, then back from the best end state, each row's
    predecessor the argmax of the previous row's F over the states its phase allows, the rows
    of a checkpoint's segment recomputed (twice the work, (R/k + k) F's of memory)."""
    R, C = gv.shape
    k = max(1, int(np.ceil(np.sqrt(R))))
    Fv, Nv = np.full((4, C, C), NEG), np.full((4, C, C), NEG)
    Fw, Nw = np.zeros((4, C, C)), np.zeros((4, C, C))
    bv, bw, P, PW, PF, rec, irec = _buffers(C)
    chk: dict[int, F64] = {}

    def run(i0: int, i1: int, start: F64, keep: bool) -> list[F64]:
        nonlocal Fv, Nv, Fw, Nw
        Fv[:] = start
        out = []
        for i in range(i0, i1):
            if not keep and i % k == 0:
                chk[i] = Fv.copy()
            _prefix(gv, gw, forb, i, P, PW, PF)
            _row(Fv, Fw, Nv, Nw, P, PW, PF, i, TRANS, bv, bw, rec, irec)
            Fv, Nv = Nv, Fv
            Fw, Nw = Nw, Fw
            if keep:
                out.append(Fv.copy())
        return out

    run(0, R, np.full((4, C, C), NEG), keep=False)
    value, W = float(rec[0]), float(rec[1])
    i, q, lo, r = (int(x) for x in irec)
    for j in [j for j in chk if j > i]:
        del chk[j]
    mask = np.zeros((R, C), dtype=bool)
    seg0, seg = -1, []
    while True:
        mask[i, lo:r + 1] = True
        if i == 0:
            break
        s0 = ((i - 1) // k) * k
        if s0 != seg0:
            seg0, seg = s0, run(s0, min(s0 + k, R), chk[s0], keep=True)
        best, p, a, b = _back(seg[i - 1 - s0], TRANS, q, lo, r)
        if best <= 0.0:
            break
        i, q, lo, r = i - 1, p, a, b
    return mask, value, W


# ---------------------------------------------------------------- the rasters

@dataclass(frozen=True)
class Grid:
    """One angle's raster in the DP's orientation (rows R >= columns C)."""
    angle: float                 # deg: the frame's x axis from the bundle's +x
    x0: float                    # m: the rotated frame's raster lower-left
    y0: float
    pix: float
    transposed: bool             # the DP's rows are the frame's x (else its y)
    M: sparse.csr_matrix         # DP cell (flat) x block: the block's buildings share there
    shape: tuple[int, int]

    def frame_ij(self, mask: B1) -> tuple[I64, I64]:
        """The mask's cells as frame (row j along y, column i along x)."""
        a, b = np.nonzero(mask)
        return (b, a) if self.transposed else (a, b)

    def outline(self, mask: B1) -> shapely.Polygon:
        """The region's outline in the bundle's metre frame (row runs: one polygon)."""
        j, i = self.frame_ij(mask)
        rows = sorted(set(j.tolist()))
        lo = {jj: int(i[j == jj].min()) for jj in rows}
        hi = {jj: int(i[j == jj].max()) for jj in rows}
        P = self.pix
        right = [(self.x0 + (hi[jj] + 1) * P, self.y0 + y * P) for jj in rows
                 for y in (jj, jj + 1)]
        left = [(self.x0 + lo[jj] * P, self.y0 + y * P) for jj in reversed(rows)
                for y in (jj + 1, jj)]
        return shapely.simplify(rotated(shapely.Polygon(right + left), self.angle), 0.0)


def rotated(geom: shapely.Geometry, angle: float) -> shapely.Geometry:
    """geom rotated by angle (deg) about the bundle's origin: a frame's coordinates to the
    bundle's for +angle, the bundle's to the frame's for -angle."""
    t = np.deg2rad(angle)
    R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    return shapely.transform(geom, lambda xy: xy @ R)


def grid_of(polys: list, angle: float, pix: float) -> Grid:
    """The blocks rasterised in the frame rotated by angle: a cell the block containing its
    centre; a block containing no centre, its point_on_surface's cell."""
    rot = [rotated(p, -angle) for p in polys]
    x0, y0, x1, y1 = shapely.total_bounds(rot)
    nx, ny = int(np.ceil((x1 - x0) / pix)) + 1, int(np.ceil((y1 - y0) / pix)) + 1
    lab = np.full((ny, nx), -1, dtype=np.int64)
    for b, p in enumerate(rot):
        bx0, by0, bx1, by1 = p.bounds
        i0, i1 = int((bx0 - x0) / pix), int(np.ceil((bx1 - x0) / pix)) + 1
        j0, j1 = int((by0 - y0) / pix), int(np.ceil((by1 - y0) / pix)) + 1
        X, Y = np.meshgrid(x0 + (np.arange(i0, i1) + 0.5) * pix,
                           y0 + (np.arange(j0, j1) + 0.5) * pix)
        lab[j0:j1, i0:i1][shapely.contains_xy(p, X, Y)] = b
    transposed = nx > ny
    if transposed:
        lab = lab.T
    R, C = lab.shape
    nb = len(polys)
    flat = np.flatnonzero(lab.ravel() >= 0)
    blk = lab.ravel()[flat]
    cnt = np.bincount(blk, minlength=nb)
    tiny = np.flatnonzero(cnt == 0)
    pt = shapely.get_coordinates(shapely.point_on_surface(np.asarray(rot, dtype=object)[tiny]))
    ti = np.minimum((pt[:, 0] - x0) // pix, nx - 1).astype(np.int64)
    tj = np.minimum((pt[:, 1] - y0) // pix, ny - 1).astype(np.int64)
    tflat = ti * C + tj if transposed else tj * C + ti
    rows = np.concatenate([flat, tflat])
    cols = np.concatenate([blk, tiny])
    vals = np.concatenate([1.0 / cnt[blk], np.ones(len(tiny))])
    M = sparse.csr_matrix((vals, (rows, cols)), shape=(R * C, nb))
    return Grid(angle=angle, x0=float(x0), y0=float(y0), pix=pix, transposed=transposed, M=M,
                shape=(R, C))


# ---------------------------------------------------------------- the statistic

class City(NamedTuple):
    """The scan's totals: every building, its log score, the weighted variance."""
    W: float
    S: float
    var: float
    cap: float


def city_of(w: F64, v: F64, *, max_share: float) -> City:
    W, S = float(w.sum()), float((w * v).sum())
    return City(W=W, S=S, var=float((w * (v - S / W) ** 2).sum() / W), cap=max_share * W)


def llr(Wz: float, Sz: float, c: City) -> float:
    """screen_clusters.py's llr: one-sided, capped."""
    if Wz <= 0.0 or Wz > c.cap or Wz >= c.W:
        return 0.0
    mz, mo = Sz / Wz, (c.S - Sz) / (c.W - Wz)
    if mz <= mo:
        return 0.0
    return Wz * (c.W - Wz) / c.W * (mz - mo) ** 2 / (2.0 * c.var)


def sums(V: float, W: float, lam: float) -> tuple[float, float]:
    """A lambda-optimum's (W, S) from its value V = S - lambda W."""
    return W, V + lam * W


def wquantile(v: F64, w: F64, q: F64) -> F64:
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return v[o][np.searchsorted(cw, q * cw[-1])]


def log_scores(city: dict, metric: str) -> F64:
    n, A, P = (np.asarray(city[k], dtype=float) for k in ("n", "area_m2", "perimeter_m"))
    s = METRICS[metric](n, A, P)
    v = np.log10(np.where(s > 0, s, np.nan))
    return np.where(np.isfinite(v), v, np.nanpercentile(v, 1))


class Fields(NamedTuple):
    gw: F64          # per DP cell its W
    gs: F64          # ... its S


def fields(g: Grid, w: F64, v: F64) -> Fields:
    R, C = g.shape
    return Fields(gw=(g.M @ w).reshape(R, C), gs=(g.M @ (w * v)).reshape(R, C))


class Best(NamedTuple):
    stat: float
    angle: int       # index into the grids
    lam: float


def sweep(fs: list[Fields], forbs: list[B1], lams: F64, c: City,
          solve: Callable[[F64, F64, B1], tuple[float, float]]) -> tuple[Best, F64]:
    """The best llr over every angle's lambda-optima; and per (angle, lambda) the llr."""
    tab = np.zeros((len(fs), len(lams)))
    for a, (f, fb) in enumerate(zip(fs, forbs, strict=True)):
        for k, lam in enumerate(lams):
            V, Wv = solve(f.gs - lam * f.gw, f.gw, fb)
            tab[a, k] = llr(*sums(V, Wv, lam), c)
    a, k = np.unravel_index(int(np.argmax(tab)), tab.shape)
    return Best(stat=float(tab[a, k]), angle=int(a), lam=float(lams[k])), tab


def solve_value(gv: F64, gw: F64, forb: B1) -> tuple[float, float]:
    rec, _ = forward(gv, gw, forb, TRANS)
    return float(rec[0]), float(rec[1])


# ---------------------------------------------------------------- the pool

_JOB: dict = {}


def _solve_job(job: tuple[int, float]) -> tuple[float, float]:
    a, lam = job
    f, fb = _JOB["fs"][a], _JOB["forbs"][a]
    return solve_value(f.gs - lam * f.gw, f.gw, fb)


def _null(job: tuple[int, int]) -> tuple[float, float, float]:
    """One shuffle at one angle: the largest llr over the lambdas, the largest excess, and the
    seconds per solve."""
    seed, a = job
    g, w, v, lams, lam0 = (_JOB["grids"][job[1]], _JOB["w"], _JOB["v"], _JOB["lams"],
                           _JOB["lam0"])
    vp = np.random.default_rng(seed).permutation(v)
    c = city_of(w, vp, max_share=MAX_SHARE)
    f = fields(g, w, vp)
    fb = np.zeros(g.shape, dtype=bool)
    best_llr, best_ex = 0.0, NEG
    t0 = time.time()
    for lam in lams:
        V, Wv = solve_value(f.gs - lam * f.gw, f.gw, fb)
        best_llr = max(best_llr, llr(*sums(V, Wv, lam), c))
        if lam == lam0:
            best_ex = V
    return best_llr, best_ex, (time.time() - t0) / len(lams)


# ---------------------------------------------------------------- the regions

def region(f: Fields, forb: B1, lam: float) -> B1:
    """The lambda-optimum as a DP mask, traced with the EPS tie-break."""
    return trace(f.gs - lam * f.gw - EPS, f.gw, forb)[0]


def forbid(grids: list[Grid], forbs: list[B1], poly: shapely.Polygon) -> None:
    """Every cell whose square overlaps the region (positive area) forbidden, at every angle."""
    for g, fb in zip(grids, forbs, strict=True):
        fr = rotated(poly, -g.angle)
        bx0, by0, bx1, by1 = fr.bounds
        P = g.pix
        ny, nx = g.shape[::-1] if g.transposed else g.shape
        i0, i1 = max(int((bx0 - g.x0) // P), 0), min(int((bx1 - g.x0) // P) + 1, nx)
        j0, j1 = max(int((by0 - g.y0) // P), 0), min(int((by1 - g.y0) // P) + 1, ny)
        jj, ii = (a.ravel() for a in np.mgrid[j0:j1, i0:i1])
        boxes = shapely.box(g.x0 + ii * P, g.y0 + jj * P, g.x0 + (ii + 1) * P,
                            g.y0 + (jj + 1) * P)
        hit = shapely.area(shapely.intersection(boxes, fr)) > 1e-6 * P ** 2
        if g.transposed:
            fb[ii[hit], jj[hit]] = True
        else:
            fb[jj[hit], ii[hit]] = True


def describe(g: Grid, f: Fields, mask: B1, *, lam: float, stat: float, nmax: F64,
             floor: float) -> dict:
    poly = g.outline(mask)
    Wz, Sz = float(f.gw[mask].sum()), float(f.gs[mask].sum())
    touched = np.unique(g.M[np.flatnonzero(mask.ravel())].indices)
    cen = poly.centroid
    return dict(polygon=np.round(shapely.get_coordinates(poly.exterior), 2).tolist(),
                angle=g.angle, lam=lam, cells=int(mask.sum()),
                area_km2=float(mask.sum() * g.pix ** 2 / 1e6), buildings=Wz,
                blocks_touched=int(len(touched)), mean_score=float(10 ** (Sz / Wz)),
                x_floor=float(10 ** (Sz / Wz) / floor), stat=stat,
                p=float((1 + (nmax >= stat).sum()) / (1 + len(nmax))),
                cx=float(cen.x), cy=float(cen.y))


def scanned(city: dict, metric: str, grids: list[Grid], *, nperm: int) -> dict:
    w = np.asarray(city["n"], dtype=float)
    v = log_scores(city, metric)
    floor = next(fl["value"] for fl in city["floors"] if fl["metric"] == metric)
    lam0 = float(np.log10(floor))
    lams = np.unique(np.concatenate([wquantile(v, w, LAM_Q), [lam0]]))
    c = city_of(w, v, max_share=MAX_SHARE)
    fs = [fields(g, w, v) for g in grids]
    out: dict = {"metric": metric, "floor": floor, "pix": grids[0].pix, "angles": list(ANGLES),
                 "lambdas": lams.tolist(), "lambda0": lam0, "nperm": nperm, "seed": SEED,
                 "max_share": MAX_SHARE, "eps": EPS}
    t0 = time.time()
    _JOB.update(grids=grids, w=w, v=v, lams=lams, lam0=lam0)
    seeds = [int(x) for x in np.random.SeedSequence(SEED).generate_state(nperm)]
    cost = [g.shape[0] * g.shape[1] ** 2 for g in grids]
    jobs = sorted(((k, a) for k in range(nperm) for a in range(len(grids))),
                  key=lambda j: -cost[j[1]])
    with get_context("fork").Pool(WORKERS) as pool:
        res = pool.map(_null, [(seeds[k], a) for k, a in jobs], chunksize=1)
    per = np.zeros((nperm, len(grids), 3))
    for (k, a), r in zip(jobs, res, strict=True):
        per[k, a] = r
    null = np.column_stack([per[:, :, 0].max(axis=1), per[:, :, 1].max(axis=1)])
    out["null_llr"], out["null_excess"] = null[:, 0].tolist(), null[:, 1].tolist()
    out["solve_s"] = per[:, :, 2].mean(axis=0).tolist()          # per angle, one solve
    out["null_s"] = time.time() - t0
    print(f"    null: {nperm} sweeps {time.time() - t0:.0f}s; one solve "
          f"{np.round(out['solve_s'], 1).tolist()}s by angle; null max llr "
          f"{np.round(np.sort(null[:, 0])[::-1][:3], 1).tolist()}..., excess "
          f"{np.round(np.sort(null[:, 1])[::-1][:3], 1).tolist()}...", flush=True)

    def top(objective: str) -> list[dict]:
        forbs = [np.zeros(g.shape, dtype=bool) for g in grids]
        found = []
        jl = lams if objective == "scan" else np.array([lam0])
        for _ in range(TOP):
            jobs = [(a, float(lam)) for a in range(len(grids)) for lam in jl]
            _JOB.update(fs=fs, forbs=forbs)
            with get_context("fork").Pool(WORKERS) as pool:
                res = pool.map(_solve_job, jobs, chunksize=1)
            if objective == "scan":
                stats = [llr(*sums(V, Wv, lam), c) for (_, lam), (V, Wv) in
                         zip(jobs, res, strict=True)]
            else:
                stats = [V for V, _ in res]
            j = int(np.argmax(stats))
            if not stats[j] > 0:
                break
            a, lam = jobs[j]
            mask = region(fs[a], forbs[a], lam)
            Wz, Sz = float(fs[a].gw[mask].sum()), float(fs[a].gs[mask].sum())
            stat = llr(Wz, Sz, c) if objective == "scan" else Sz - lam0 * Wz
            d = describe(grids[a], fs[a], mask, lam=lam, stat=stat,
                         nmax=null[:, 0 if objective == "scan" else 1], floor=floor)
            d["sweep_stat"] = float(stats[j])
            found.append(d)
            forbid(grids, forbs, shapely.Polygon(d["polygon"]))
        return found

    t1 = time.time()
    out["regions"] = top("scan")
    out["excess"] = top("excess")
    out["observed_s"] = time.time() - t1
    return out


# ---------------------------------------------------------------- the checks

@njit(cache=True)
def _orthoconvex_masks(R: int, C: int) -> I64:
    """Every nonempty cell set of an R x C grid (bit r*C+c) meeting every row and column in one
    run and 4-connected, by definition (no row-interval DP): the brute force's regions."""
    out = []
    full = (1 << C) - 1
    stack = np.empty(R * C, np.int64)
    for m in range(1, 1 << (R * C)):
        ok = True
        for r in range(R):
            b = (m >> (r * C)) & full
            if b and ((b // (b & -b)) + 1) & (b // (b & -b)):
                ok = False
                break
        if not ok:
            continue
        for c in range(C):
            b = 0
            for r in range(R):
                b |= ((m >> (r * C + c)) & 1) << r
            if b and ((b // (b & -b)) + 1) & (b // (b & -b)):
                ok = False
                break
        if not ok:
            continue
        first = 0
        while not (m >> first) & 1:
            first += 1
        seen = 1 << first
        stack[0] = first
        top = 1
        while top:
            top -= 1
            x = stack[top]
            r, c = divmod(x, C)
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < R and 0 <= cc < C:
                    y = rr * C + cc
                    if (m >> y) & 1 and not (seen >> y) & 1:
                        seen |= 1 << y
                        stack[top] = y
                        top += 1
        if seen == m:
            out.append(m)
    return np.array(out, dtype=np.int64)


def _orthoconvex(mask: B1) -> bool:
    """By definition: every row and column one run, 4-connected."""
    from scipy.ndimage import label
    runs = [np.flatnonzero(x) for x in list(mask) + list(mask.T)]
    return (mask.any() and all(len(x) == 0 or x[-1] - x[0] + 1 == len(x) for x in runs)
            and label(mask)[1] == 1)


def _bits(masks: I64, n: int) -> NDArray[np.bool_]:
    return ((masks[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)


def _llr_max(Wz: F64, Sz: F64, c: City) -> float:
    """llr's maximum over the given regions, vectorised (the brute force's)."""
    ok = (Wz > 0) & (Wz <= c.cap) & (Wz < c.W)
    Wz, Sz = Wz[ok], Sz[ok]
    mz, mo = Sz / Wz, (c.S - Sz) / (c.W - Wz)
    r = np.where(mz > mo, Wz * (c.W - Wz) / c.W * (mz - mo) ** 2 / (2.0 * c.var), 0.0)
    return float(r.max(initial=0.0))


def check(fault: str | None) -> bool:
    """The DP's optimum (and the traced region) against brute force over every orthogonally
    convex region of small random grids; the lambda-sweep's best llr against brute-force llr."""
    rng = np.random.default_rng(SEED)
    bad = 0
    for R, C in ((5, 5), (4, 6), (6, 4)):
        t0 = time.time()
        masks = _orthoconvex_masks(R, C)
        X = _bits(masks, R * C)
        valid = set(masks.tolist())
        print(f"  {R}x{C}: {len(masks)} orthogonally convex regions ({time.time() - t0:.0f}s)")
        dp_bad = tr_bad = 0
        for trial in range(200):
            g = rng.normal(size=(R, C)) - 0.3
            gw = rng.exponential(size=(R, C)) * (rng.random((R, C)) > 0.2)
            forb = rng.random((R, C)) < (0.1 if trial % 2 else 0.0)
            vals = X.astype(float) @ g.ravel()
            vals[(X & forb.ravel()[None, :]).any(axis=1)] = NEG
            want = float(vals.max())
            rec, irec = forward(g, gw, forb, TRANS)
            if not np.isclose(rec[0], want, atol=1e-9):
                dp_bad += 1
                continue
            mask, value, W = trace(g, gw, forb)
            bit = int((mask.ravel().astype(np.int64) << np.arange(R * C)).sum())
            if not (bit in valid and np.isclose(g[mask].sum(), want, atol=1e-9)
                    and not (mask & forb).any() and np.isclose(gw[mask].sum(), rec[1], atol=1e-9)
                    and np.isclose(W, rec[1], atol=1e-9)
                    and int(irec[0]) == np.flatnonzero(mask.any(axis=1))[-1]):
                tr_bad += 1
        print(f"    DP optimum != brute force: {dp_bad} of 200; traced region wrong "
              f"(not orthogonally convex, value, W or last row): {tr_bad}")
        bad += dp_bad + tr_bad
    # the trace on grids past brute force (several checkpoint segments, forbidden cells)
    tr_bad = 0
    for _trial in range(20):
        R, C = int(rng.integers(20, 60)), int(rng.integers(5, 25))
        g = rng.normal(size=(R, C)) + 0.15 * np.add.outer(np.sin(np.arange(R) / 5),
                                                         np.cos(np.arange(C) / 3)) - 0.1
        gw = rng.exponential(size=(R, C))
        forb = rng.random((R, C)) < 0.02
        rec, _ = forward(g, gw, forb, TRANS)
        mask, value, W = trace(g, gw, forb)
        if not (_orthoconvex(mask) and np.isclose(g[mask].sum(), rec[0], atol=1e-9)
                and np.isclose(value, rec[0]) and not (mask & forb).any()
                and np.isclose(gw[mask].sum(), rec[1], atol=1e-9)):
            tr_bad += 1
    print(f"  trace on 20 grids 20-60 x 5-25: region not orthogonally convex, or its value or W "
          f"not the DP's: {tr_bad}")
    bad += tr_bad
    # the sweep: cells with weights, the city the grid; llr's maximiser over every region
    R, C = 5, 5
    masks = _orthoconvex_masks(R, C)
    X = _bits(masks, R * C).astype(float)
    sw_bad, gaps, gaps_q = 0, [], []
    for _trial in range(60):
        w = rng.exponential(size=R * C) * (rng.random(R * C) > 0.15)
        v = rng.normal(size=R * C) + (np.arange(R * C) % C < 2) * rng.normal() * 0.8
        fine = np.linspace(v.min() - 1, v.max() + 1, 2001)
        coarse = np.unique(np.concatenate([wquantile(v, w, LAM_Q), [0.0]]))
        forb = np.zeros((R, C), dtype=bool)
        f = Fields(gw=w.reshape(R, C), gs=(w * v).reshape(R, C))
        Wm, Sm = X @ w, X @ (w * v)
        for share in (1.0, MAX_SHARE):
            c = city_of(w, v, max_share=share)
            want = _llr_max(Wm, Sm, c)
            got = sweep([f], [forb], fine, c, solve_value)[0].stat
            got_q = sweep([f], [forb], coarse, c, solve_value)[0].stat
            if share == 1.0 and not np.isclose(got, want, rtol=1e-9, atol=1e-12):
                sw_bad += 1
            if got > want * (1 + 1e-9) + 1e-12 or got_q > want * (1 + 1e-9) + 1e-12:
                sw_bad += 1
            if share < 1.0:
                gaps.append((want - got) / want if want > 0 else 0.0)
                gaps_q.append((want - got_q) / want if want > 0 else 0.0)
    print(f"  sweep: 60 5x5 grids; best llr != brute force (no cap, 2001 lambdas) or above it "
          f"(either cap, either grid): {sw_bad}")
    for lab, gp in (("2001 lambdas", gaps), (f"the production's {len(LAM_Q) + 1}", gaps_q)):
        ga = np.asarray(gp)
        print(f"    under the cap {MAX_SHARE}, {lab}: the sweep's shortfall of the capped "
              f"maximum median {np.median(ga):.3f}, max {ga.max():.3f}, none in "
              f"{(ga < 1e-9).mean():.0%}")
    bad += sw_bad
    return bad == 0


FAULTS = ("re-widen", "no-switch", "s-sign")


def main(argv: list[str]) -> None:
    if argv and argv[0] == "check":
        fault = argv[1] if len(argv) > 1 else None
        global TRANS, sums
        if fault == "re-widen":         # the left ends may move left again after moving right
            TRANS = TRANS.copy()
            TRANS[2, 0] = TRANS[3, 1] = True
        elif fault == "no-switch":      # the left ends never turn while the right still widen
            TRANS = TRANS.copy()
            TRANS[0, 2] = False
        elif fault == "s-sign":         # a lambda-optimum's S read with lambda W's sign flipped
            sums = lambda V, W, lam: (W, V - lam * W)   # noqa: E731
        elif fault is not None:
            raise SystemExit(f"faults: {', '.join(FAULTS)}")
        ok = check(fault)
        print(f"check{'' if fault is None else f' under the fault {fault}'}: "
              f"{'pass' if ok else 'FAIL'}")
        if not ok and fault is None:
            raise SystemExit(1)
        if ok and fault is not None:
            raise SystemExit(f"the fault {fault} went unseen: the check guards nothing")
        return
    if len(argv) != 2:
        raise SystemExit(__doc__)
    root, out = Path(argv[0]).expanduser(), Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    z = np.zeros((3, 3))
    trace(z - 1.0, z, z > 0)            # compiled once, before the pools fork
    for name in CITIES:
        t0 = time.time()
        city = json.loads((root / "examples" / "screen-map" / f"{name}.json").read_text())
        polys = [shapely.Polygon(r[0], r[1:]) for r in city["rings"]]
        grids = [grid_of(polys, a, PIX[name]) for a in ANGLES]
        print(f"{name}: rasters {[g.shape for g in grids]} ({time.time() - t0:.0f}s)", flush=True)
        res = {}
        for m in METRICS:
            t1 = time.time()
            r = scanned(city, m, grids, nperm=NPERM[name])
            res[m] = r
            for kind in ("regions", "excess"):
                top = [(round(x["stat"], 1), round(x["p"], 3), round(x["buildings"]),
                        round(x["area_km2"], 2), round(x["x_floor"], 2),
                        (round(x["cx"] / 1e3, 1), round(x["cy"] / 1e3, 1))) for x in r[kind]]
                print(f"  {m} {kind}: {top}", flush=True)
            print(f"  ({time.time() - t1:.0f}s)", flush=True)
        (out / f"convex_{name}.json").write_text(json.dumps(res))
        print(f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
