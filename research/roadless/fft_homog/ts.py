"""Two-scale model of the scalar 'uni' escape metric on a block raster.

Fine reference: cell-centred finite volumes at h (the raster), face conductance min(o_a, o_b)
between 4-neighbours, ground cells u = 0, injection f, unknowns = open non-ground cells in a
4-connected component of open space that holds a ground cell (lifted.Grid.reach's rule).

Two-scale: (1) the tensor field sigma(x) from periodic FFT cell problems on moving windows (size
n px, centres every s px; cells outside the block are open: street), each also giving the
correctors chi_x, chi_y and the local source corrector w (A w = f - g, g the window's injection
spread over each conducting component by open fraction), kept on the window's central s x s
block; (2) the macro problem -div(sigma grad U) = f on Q1 finite elements of side k px, U = 0 at
nodes outside the block or on ground; (3) per cell u ~ U + chi . grad U + w, per home the
injection-weighted mean over its ring cells.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import fftk
import numpy as np
import pyamg
import scipy.sparse as sp
from scipy import ndimage
from scipy.sparse.csgraph import connected_components


# ------------------------------------------------------------------------- fine reference
def reach_of(o: np.ndarray, ground: np.ndarray) -> np.ndarray:
    free = o > 0
    lab, n = ndimage.label(free)
    ok = np.zeros(n + 1, dtype=bool)
    ok[lab[free & ground]] = True
    ok[0] = False
    return ok[lab]


def fine_matrix(o: np.ndarray, ground: np.ndarray):
    """(A over unknowns, unknown index per cell or -1)."""
    reach = reach_of(o, ground)
    unk = reach & ~ground
    idx = -np.ones(o.shape, dtype=np.int64)
    idx[unk] = np.arange(int(unk.sum()))
    N = int(unk.sum())
    rows, cols, vals = [], [], []
    diag = np.zeros(N)
    for axis in (0, 1):
        a = [slice(None), slice(None)]
        b = [slice(None), slice(None)]
        a[axis], b[axis] = slice(0, -1), slice(1, None)
        oa, ob = o[tuple(a)], o[tuple(b)]
        c = np.minimum(oa, ob)
        ra, rb = reach[tuple(a)], reach[tuple(b)]
        ia, ib = idx[tuple(a)], idx[tuple(b)]
        live = (c > 0) & ra & rb
        ia, ib, c = ia[live], ib[live], c[live]
        both = (ia >= 0) & (ib >= 0)
        rows += [ia[both], ib[both]]
        cols += [ib[both], ia[both]]
        vals += [-c[both], -c[both]]
        diag += np.bincount(ia[ia >= 0], weights=c[ia >= 0], minlength=N)
        diag += np.bincount(ib[ib >= 0], weights=c[ib >= 0], minlength=N)
    ar = np.arange(N)
    A = sp.coo_matrix((np.concatenate(vals + [diag]),
                       (np.concatenate(rows + [ar]), np.concatenate(cols + [ar]))),
                      shape=(N, N)).tocsr()
    return A, idx


def amg_solve(A, b, tol: float):
    ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric", max_coarse=2000)
    res: list[float] = []
    x = ml.solve(b, tol=tol, accel="cg", maxiter=2000, residuals=res)
    if not res[-1] <= 10 * tol * np.linalg.norm(b):
        raise RuntimeError(f"AMG-CG failed: {res[-1] / np.linalg.norm(b):.1e} after {len(res)}")
    return x, ml, len(res)


@dataclass
class FineSolution:
    u: np.ndarray        # (ny, nx), 0 on ground / closed / unreached
    seconds: float
    iters: int
    n: int


def fine_solve(o, ground, f, tol=1e-10) -> FineSolution:
    t = time.perf_counter()
    A, idx = fine_matrix(o, ground)
    on = idx >= 0
    assert not (f[~on & ~ground] > 0).any() or True
    b = f[on]
    x, _, it = amg_solve(A, b, tol)
    u = np.zeros(o.shape)
    u[on] = x
    return FineSolution(u, time.perf_counter() - t, it, A.shape[0])


def home_u(ucell, f, owner, w, live):
    on = owner >= 0
    num = np.bincount(owner[on], weights=(f * ucell)[on], minlength=len(w))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(live, num / w, np.nan)


def J(u, w, live, p=2.0):
    return float((w[live] * u[live] ** p).sum())


# ------------------------------------------------------------------------- tensor field
_G: dict = {}


def _init(o_pad, f_pad, n, s, pad, i_pad=None):
    _G.update(o=o_pad, f=f_pad, n=n, s=s, pad=pad, i=i_pad)


def _window(job):
    """job = (bi, bj): central block rows [bi s, bi s + s), cols [bj s, ...). Returns
    (bi, bj, sigma(2,2), iters, chi_x, chi_y, w (s x s each))."""
    bi, bj = job
    o, f, n, s, pad = _G["o"], _G["f"], _G["n"], _G["s"], _G["pad"]
    r0 = pad + bi * s + s // 2 - n // 2
    c0 = pad + bj * s + s // 2 - n // 2
    ow = o[r0:r0 + n, c0:c0 + n]
    fw = f[r0:r0 + n, c0:c0 + n]
    P = fftk.Problem.of(ow)
    # source corrector rhs: f minus its component mean spread by open fraction
    A = fftk.periodic_matrix(P.cx, P.cy)
    ncomp, lab = connected_components(A, directed=False)
    lab = lab.reshape(ow.shape)
    live = (P.cx > 0) | (P.cy > 0) | (np.roll(P.cx, 1, 1) > 0) | (np.roll(P.cy, 1, 0) > 0)
    fsum = np.bincount(lab[live], weights=fw[live], minlength=ncomp)
    osum = np.bincount(lab[live], weights=ow[live], minlength=ncomp)
    g = np.where(live, ow * fsum[lab] / np.where(osum[lab] > 0, osum[lab], 1), 0.0)
    bf = (np.where(live, fw, 0.0) - g)[None]
    u, it, _, res = fftk.solve(P, np.eye(2), "cg", 1e-8, 5000, 1.0, b_extra=bf)
    S, _ = P.sigma_of(u[:2])
    if _G.get("i") is not None:
        # inside-averaged apparent tensor: <q>_in <e>_in^-1 over the window's inside cells
        iw = _G["i"][r0:r0 + n, c0:c0 + n] > 0
        gx, gy = fftk.grad(u[:2])
        if iw.all() or not iw.any():
            S_in = S
        else:
            Q = np.array([[np.mean((P.cx * (gx[j] + (j == 0)))[iw]) for j in range(2)],
                          [np.mean((P.cy * (gy[j] + (j == 1)))[iw]) for j in range(2)]])
            Eg = np.array([[np.mean((gx[j] + (j == 0))[iw]) for j in range(2)],
                           [np.mean((gy[j] + (j == 1))[iw]) for j in range(2)]])
            S_in = Q @ np.linalg.inv(Eg)
            S_in = 0.5 * (S_in + S_in.T)
        S = np.stack([S, S_in])
    q0 = n // 2 - s // 2
    cen = (slice(q0, q0 + s), slice(q0, q0 + s))
    oc = ow[cen]
    wgt = oc / oc.sum() if oc.sum() > 0 else np.zeros_like(oc)
    out = []
    for m in range(3):
        blk = u[m][cen]
        out.append((blk - (blk * wgt).sum()).astype(np.float32))
    return bi, bj, S, it, out[0], out[1], out[2]


@dataclass
class TensorField:
    sigma: np.ndarray       # (nbi, nbj, 2, 2) at block centres
    inside: np.ndarray      # (nbi, nbj) inside fraction of each window
    have: np.ndarray        # (nbi, nbj) computed
    chi: np.ndarray         # (2, ny, nx)
    wf: np.ndarray          # (ny, nx)
    s: int
    n: int
    seconds: float
    iters: np.ndarray


def tensor_field(o, inside, f, n: int, s: int, workers: int, need=None) -> TensorField:
    """need: (nbi, nbj) bool, the blocks to compute (default: blocks holding an inside cell)."""
    ny, nx = o.shape
    nbi, nbj = -(-ny // s), -(-nx // s)
    pad = n
    o_pad = np.ones((nbi * s + 2 * pad, nbj * s + 2 * pad))
    f_pad = np.zeros_like(o_pad)
    o_pad[pad:pad + ny, pad:pad + nx] = np.where(inside, o, 1.0)
    f_pad[pad:pad + ny, pad:pad + nx] = f
    ins_pad = np.zeros_like(o_pad)
    ins_pad[pad:pad + ny, pad:pad + nx] = inside
    if need is None:
        blk = np.zeros((nbi * s, nbj * s), dtype=bool)
        blk[:ny, :nx] = inside
        need = blk.reshape(nbi, s, nbj, s).any(axis=(1, 3))
    jobs = [tuple(x) for x in np.argwhere(need)]
    t = time.perf_counter()
    sig = np.zeros((nbi, nbj, 2, 2))
    sig_in = np.zeros((nbi, nbj, 2, 2))
    its = np.zeros((nbi, nbj), dtype=int)
    chi = np.zeros((2, nbi * s, nbj * s), dtype=np.float32)
    wf = np.zeros((nbi * s, nbj * s), dtype=np.float32)
    with ProcessPoolExecutor(workers, initializer=_init,
                             initargs=(o_pad, f_pad, n, s, pad, ins_pad)) as ex:
        for bi, bj, S, it, cx, cy, w in ex.map(_window, jobs, chunksize=16):
            sig[bi, bj], sig_in[bi, bj] = S[0], S[1]
            its[bi, bj] = it
            chi[0, bi * s:(bi + 1) * s, bj * s:(bj + 1) * s] = cx
            chi[1, bi * s:(bi + 1) * s, bj * s:(bj + 1) * s] = cy
            wf[bi * s:(bi + 1) * s, bj * s:(bj + 1) * s] = w
    sec = time.perf_counter() - t
    # inside fraction of each window (integral image)
    integ = np.pad(ins_pad.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    bi_, bj_ = np.meshgrid(np.arange(nbi), np.arange(nbj), indexing="ij")
    r0 = pad + bi_ * s + s // 2 - n // 2
    c0 = pad + bj_ * s + s // 2 - n // 2
    insf = (integ[r0 + n, c0 + n] - integ[r0, c0 + n] - integ[r0 + n, c0] + integ[r0, c0]) / (n * n)
    tf = TensorField(sig, insf, need, chi[:, :ny, :nx], wf[:ny, :nx], s, n, sec, its)
    tf.sigma_in = sig_in
    return tf


def update_field(tf: TensorField, o, inside, f, changed: np.ndarray, workers: int) -> TensorField:
    """Recompute only the windows that overlap a changed cell."""
    s, n = tf.s, tf.n
    nbi, nbj = tf.have.shape
    pad = n
    ch = np.zeros((nbi * s + 2 * pad, nbj * s + 2 * pad), dtype=bool)
    ch[pad:pad + o.shape[0], pad:pad + o.shape[1]] = changed
    integ = np.pad(ch.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    bi_, bj_ = np.meshgrid(np.arange(nbi), np.arange(nbj), indexing="ij")
    r0 = pad + bi_ * s + s // 2 - n // 2
    c0 = pad + bj_ * s + s // 2 - n // 2
    hit = (integ[r0 + n, c0 + n] - integ[r0, c0 + n] - integ[r0 + n, c0] + integ[r0, c0]) > 0
    need = hit & tf.have
    new = tensor_field(o, inside, f, n, s, workers, need=need)
    sig = tf.sigma.copy()
    sig[need] = new.sigma[need]
    chi, wf = tf.chi.copy(), tf.wf.copy()
    m = np.repeat(np.repeat(need, s, 0), s, 1)[:o.shape[0], :o.shape[1]]
    chi[:, m] = new.chi[:, m]
    wf[m] = new.wf[m]
    its = tf.iters.copy()
    its[need] = new.iters[need]
    out = TensorField(sig, tf.inside, tf.have, chi, wf, s, n, new.seconds, its)
    if hasattr(tf, "sigma_in"):
        sig_in = tf.sigma_in.copy()
        sig_in[need] = new.sigma_in[need]
        out.sigma_in = sig_in
    return out


def fill(tf: TensorField, min_inside: float | None, inside_avg: bool = False) -> np.ndarray:
    """sigma per block; blocks not computed, or (min_inside set) whose window is less than
    min_inside inside the block, take the nearest valid block's tensor. inside_avg: the
    inside-averaged apparent tensor <q>_in <e>_in^-1 instead of the periodic one."""
    valid = tf.have.copy()
    if min_inside is not None:
        valid &= tf.inside >= min_inside
    _, (ii, jj) = ndimage.distance_transform_edt(~valid, return_indices=True)
    return (tf.sigma_in if inside_avg else tf.sigma)[ii, jj]


# ------------------------------------------------------------------------- macro Q1 FE
def _q1_mats():
    g = np.array([0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)])
    # local nodes: (di, dj) = (0,0), (0,1), (1,0), (1,1); a along columns (x), b along rows (y)
    M = {k: np.zeros((4, 4)) for k in ("xx", "yy", "xy")}
    for a in g:
        for b in g:
            da = np.array([-(1 - b), (1 - b), -b, b])
            db = np.array([-(1 - a), -a, (1 - a), a])
            M["xx"] += 0.25 * np.outer(da, da)
            M["yy"] += 0.25 * np.outer(db, db)
            M["xy"] += 0.25 * (np.outer(da, db) + np.outer(db, da))
    return M


Q1 = _q1_mats()


@dataclass
class Macro:
    k: int                  # element side in px
    U: np.ndarray           # (NI, NJ) nodal values
    seconds: float
    n: int
    sig_e: np.ndarray       # (NI-1, NJ-1, 2, 2)
    A: object
    unk: np.ndarray
    F: np.ndarray


def element_tensors(sig_blocks: np.ndarray, s: int, k: int, NI: int, NJ: int) -> np.ndarray:
    """Bilinear interpolation of the block-centre tensors at element centres."""
    yc = ((np.arange(NI - 1) + 0.5) * k - (s / 2 - 0.5)) / s
    xc = ((np.arange(NJ - 1) + 0.5) * k - (s / 2 - 0.5)) / s
    Y, X = np.meshgrid(yc, xc, indexing="ij")
    out = np.zeros((NI - 1, NJ - 1, 2, 2))
    for a in range(2):
        for b in range(2):
            out[..., a, b] = ndimage.map_coordinates(sig_blocks[..., a, b], [Y, X], order=1,
                                                     mode="nearest")
    return out


FLOOR = 1e-3   # sigma + FLOOR I on every element: a fully blocked window (sigma = 0) would
               # leave its nodes without equations (met with 25 m windows; reported)


def macro_solve(sig_blocks, s: int, k: int, inside, ground, f, tol=1e-10,
                load=None) -> Macro:
    t = time.perf_counter()
    ny, nx = inside.shape
    NI, NJ = -(-ny // k) + 1, -(-nx // k) + 1
    sig_e = element_tensors(sig_blocks, s, k, NI, NJ) + FLOOR * np.eye(2)
    # node status from the fine cell at the node position
    ri = np.clip(np.arange(NI) * k, 0, ny - 1)
    cj = np.clip(np.arange(NJ) * k, 0, nx - 1)
    free = inside[np.ix_(ri, cj)] & ~ground[np.ix_(ri, cj)]
    unk = -np.ones((NI, NJ), dtype=np.int64)
    unk[free] = np.arange(int(free.sum()))
    N = int(free.sum())
    Ir, Jn = np.meshgrid(np.arange(NI - 1), np.arange(NJ - 1), indexing="ij")
    nodes = np.stack([unk[Ir, Jn], unk[Ir, Jn + 1], unk[Ir + 1, Jn], unk[Ir + 1, Jn + 1]], -1)
    Ke = (sig_e[..., 0, 0, None, None] * Q1["xx"] + sig_e[..., 1, 1, None, None] * Q1["yy"]
          + sig_e[..., 0, 1, None, None] * Q1["xy"])
    rr = np.broadcast_to(nodes[..., :, None], Ke.shape).ravel()
    cc = np.broadcast_to(nodes[..., None, :], Ke.shape).ravel()
    vv = Ke.ravel()
    ok = (rr >= 0) & (cc >= 0)
    A = sp.coo_matrix((vv[ok], (rr[ok], cc[ok])), shape=(N, N)).tocsr()
    F = load_vector(f if load is None else load, k, unk, N)
    x, _, _ = amg_solve(A, F, tol)
    U = np.zeros((NI, NJ))
    U[free] = x
    return Macro(k, U, time.perf_counter() - t, N, sig_e, A, unk, F)


def _local(shape, k):
    ny, nx = shape
    i = np.arange(ny)[:, None] * np.ones((1, nx))
    j = np.ones((ny, 1)) * np.arange(nx)[None, :]
    Ir, Jn = np.floor(i / k).astype(int), np.floor(j / k).astype(int)
    b, a = i / k - Ir, j / k - Jn
    return Ir, Jn, a, b


def load_vector(f, k, unk, N):
    Ir, Jn, a, b = _local(f.shape, k)
    on = f > 0
    Ir, Jn, a, b, fv = Ir[on], Jn[on], a[on], b[on], f[on]
    F = np.zeros(N)
    for di, dj, wgt in ((0, 0, (1 - a) * (1 - b)), (0, 1, a * (1 - b)), (1, 0, (1 - a) * b),
                        (1, 1, a * b)):
        n = unk[Ir + di, Jn + dj]
        m = n >= 0
        F += np.bincount(n[m], weights=(fv * wgt)[m], minlength=N)
    return F


def evaluate(mac: Macro, shape):
    """(U, dU/dx, dU/dy) per fine cell (per px)."""
    k, Un = mac.k, mac.U
    Ir, Jn, a, b = _local(shape, k)
    u00, u01, u10, u11 = Un[Ir, Jn], Un[Ir, Jn + 1], Un[Ir + 1, Jn], Un[Ir + 1, Jn + 1]
    U = u00 * (1 - a) * (1 - b) + u01 * a * (1 - b) + u10 * (1 - a) * b + u11 * a * b
    Ux = ((u01 - u00) * (1 - b) + (u11 - u10) * b) / k
    Uy = ((u10 - u00) * (1 - a) + (u11 - u01) * a) / k
    return U, Ux, Uy


def load_field(path: str) -> TensorField:
    z = np.load(path)
    tf = TensorField(z["sigma"], z["inside"], z["have"], z["chi"], z["wf"], int(z["s"]),
                     int(z["n"]), float(z["seconds"]) if "seconds" in z else np.nan, z["iters"])
    if "sigma_in" in z:
        tf.sigma_in = z["sigma_in"]
    return tf


def windows_hit(tf: TensorField, changed: np.ndarray) -> np.ndarray:
    s, n = tf.s, tf.n
    nbi, nbj = tf.have.shape
    pad = n
    ch = np.zeros((nbi * s + 2 * pad, nbj * s + 2 * pad), dtype=bool)
    ch[pad:pad + changed.shape[0], pad:pad + changed.shape[1]] = changed
    integ = np.pad(ch.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    bi_, bj_ = np.meshgrid(np.arange(nbi), np.arange(nbj), indexing="ij")
    r0 = pad + bi_ * s + s // 2 - n // 2
    c0 = pad + bj_ * s + s // 2 - n // 2
    return ((integ[r0 + n, c0 + n] - integ[r0, c0 + n] - integ[r0 + n, c0]
             + integ[r0, c0]) > 0) & tf.have


def sigma_serial(tf: TensorField, o, inside, f, changed) -> tuple[np.ndarray, int]:
    """Block tensors with the windows overlapping `changed` recomputed in this process."""
    ny, nx = o.shape
    nbi, nbj = tf.have.shape
    s, n, pad = tf.s, tf.n, tf.n
    o_pad = np.ones((nbi * s + 2 * pad, nbj * s + 2 * pad))
    f_pad = np.zeros_like(o_pad)
    o_pad[pad:pad + ny, pad:pad + nx] = np.where(inside, o, 1.0)
    f_pad[pad:pad + ny, pad:pad + nx] = f
    _init(o_pad, f_pad, n, s, pad)
    sig = tf.sigma.copy()
    need = windows_hit(tf, changed)
    for bi, bj in np.argwhere(need):
        sig[bi, bj] = _window((bi, bj))[2]
    return sig, int(need.sum())
