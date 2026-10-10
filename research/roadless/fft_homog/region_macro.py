"""The two-scale macro problem on 5810@major (ts.macro_solve's Q1 model, unchanged: bilinear
elements of side k px, one tensor per element interpolated from the block tensors, + FLOOR I,
U = 0 at nodes outside the region or on ground, the injection loaded with bilinear weights),
with what the tile study needs kept: the SA hierarchy, the homes' readout as a matrix (each
home's injection-weighted mean of U over its ring cells, ground cells 0, as ts.home_u of
ts.evaluate's U), the J_2 adjoint and the element sensitivities S (dJ ~ -sum_e S_e : dsigma_e),
and a local re-solve after a change of the block tensors (the fine model's multiplicative
Schwarz: the base hierarchy plus an exact solve on the nodes within `margin` of the change).
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pyamg
import region_common as rc
import scipy.sparse as sp
import ts
from region_fine import pcg
from scipy import ndimage
from scipy.sparse.linalg import splu

MACRO_TOL = 1e-11
DELTA_RTOL = 1e-9
MARGIN = 8            # macro nodes (16 m at H 2 m)


def assemble(sig_e: np.ndarray, unk: np.ndarray, N: int) -> sp.csr_matrix:
    """ts.macro_solve's stiffness matrix from element tensors sig_e (NI-1, NJ-1, 2, 2) (FLOOR
    included by the caller)."""
    NI, NJ = unk.shape
    Ir, Jn = np.meshgrid(np.arange(NI - 1), np.arange(NJ - 1), indexing="ij")
    nodes = np.stack([unk[Ir, Jn], unk[Ir, Jn + 1], unk[Ir + 1, Jn], unk[Ir + 1, Jn + 1]], -1)
    return _scatter(sig_e, nodes, N)


def assemble_elems(s: np.ndarray, elems: np.ndarray, unk: np.ndarray, N: int) -> sp.csr_matrix:
    """The same over the elements `elems` (flat indices) with tensors s (m, 2, 2)."""
    NJm = unk.shape[1] - 1
    Ir, Jn = elems // NJm, elems % NJm
    nodes = np.stack([unk[Ir, Jn], unk[Ir, Jn + 1], unk[Ir + 1, Jn], unk[Ir + 1, Jn + 1]], -1)
    return _scatter(s, nodes, N)


def _scatter(s: np.ndarray, nodes: np.ndarray, N: int) -> sp.csr_matrix:
    Ke = (s[..., 0, 0, None, None] * ts.Q1["xx"] + s[..., 1, 1, None, None] * ts.Q1["yy"]
          + s[..., 0, 1, None, None] * ts.Q1["xy"])
    rr = np.broadcast_to(nodes[..., :, None], Ke.shape).ravel()
    cc = np.broadcast_to(nodes[..., None, :], Ke.shape).ravel()
    vv = Ke.ravel()
    ok = (rr >= 0) & (cc >= 0)
    return sp.coo_matrix((vv[ok], (rr[ok], cc[ok])), shape=(N, N)).tocsr()


class MacroBase:
    def __init__(self, sig_blocks: np.ndarray, s: int, k: int, fab: rc.Fabric,
                 tol: float = MACRO_TOL, adjoint: bool = False):
        t0 = time.perf_counter()
        self.s, self.k, self.fab = s, k, fab
        inside, ground = fab.inside, fab.ground
        ny, nx = inside.shape
        self.NI, self.NJ = NI, NJ = -(-ny // k) + 1, -(-nx // k) + 1
        self.sig_blocks = sig_blocks
        self.sig_e = ts.element_tensors(sig_blocks, s, k, NI, NJ) + ts.FLOOR * np.eye(2)
        ri = np.clip(np.arange(NI) * k, 0, ny - 1)
        cj = np.clip(np.arange(NJ) * k, 0, nx - 1)
        free = inside[np.ix_(ri, cj)] & ~ground[np.ix_(ri, cj)]
        self.unk = -np.ones((NI, NJ), dtype=np.int64)
        self.unk[free] = np.arange(int(free.sum()))
        self.N = N = int(free.sum())
        self.A = assemble(self.sig_e, self.unk, N)
        self.F = ts.load_vector(fab.f, k, self.unk, N)
        self.H = self._readout()
        t1 = time.perf_counter()
        self.ml = pyamg.smoothed_aggregation_solver(self.A, symmetry="symmetric",
                                                    max_coarse=2000)
        t2 = time.perf_counter()
        self.U0, self.iters, _ = pcg(lambda x: self.A @ x, self.F, self.vcycle, tol, 2000)
        self.t_assemble, self.t_setup, self.t_solve = t1 - t0, t2 - t1, time.perf_counter() - t2
        self.uh0 = self.home(self.U0)
        lv = fab.live
        self.J0 = float((fab.w[lv] * self.uh0[lv] ** 2).sum())
        if adjoint:
            g = self.H.T @ np.where(lv, 2 * fab.w * np.nan_to_num(self.uh0), 0.0)
            self.lam, _, _ = pcg(lambda x: self.A @ x, g, self.vcycle, tol, 2000)
            self.S = self.sensitivity(self.U0, self.lam)

    def vcycle(self, r: np.ndarray) -> np.ndarray:
        x = np.zeros_like(r)
        self.ml._MultilevelSolver__solve(0, x, np.ascontiguousarray(r), "V", 1)
        return x

    def _readout(self) -> sp.csr_matrix:
        """(buildings x unknowns): home i's u = sum over its ring cells c (off the ground) of
        f_c U(c) / w_i, U(c) ts.evaluate's bilinear interpolation of the nodes."""
        fab, k = self.fab, self.k
        fc = np.flatnonzero((fab.f.ravel() > 0) & ~fab.ground.ravel())
        nx = fab.o.shape[1]
        i, j = fc // nx, fc % nx
        Ir, Jn = i // k, j // k
        b, a = i / k - Ir, j / k - Jn
        own, fv = fab.owner.ravel()[fc], fab.f.ravel()[fc]
        rows, cols, vals = [], [], []
        for di, dj, wt in ((0, 0, (1 - a) * (1 - b)), (0, 1, a * (1 - b)), (1, 0, (1 - a) * b),
                           (1, 1, a * b)):
            n = self.unk[Ir + di, Jn + dj]
            m = n >= 0
            rows.append(own[m])
            cols.append(n[m])
            vals.append((fv * wt / fab.w[own])[m])
        return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(fab.w), self.N)).tocsr()

    def home(self, U: np.ndarray) -> np.ndarray:
        return np.where(self.fab.live, self.H @ U, np.nan)

    def nodal(self, U: np.ndarray) -> np.ndarray:
        out = np.zeros((self.NI, self.NJ))
        out[self.unk >= 0] = U
        return out

    def evaluate(self, U: np.ndarray):
        """(U, dU/dx, dU/dy) per fine cell (ts.evaluate)."""
        return ts.evaluate(SimpleNamespace(k=self.k, U=self.nodal(U)), self.fab.o.shape)

    def sensitivity(self, U: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """(NI-1, NJ-1, 3): lam_e^T M U_e for M = Q1 xx, yy, xy (tiles_clear.py's S)."""
        Un, Ln = self.nodal(U), self.nodal(lam)
        NI, NJ = self.NI, self.NJ
        Ir, Jn = np.meshgrid(np.arange(NI - 1), np.arange(NJ - 1), indexing="ij")
        ue = np.stack([Un[Ir, Jn], Un[Ir, Jn + 1], Un[Ir + 1, Jn], Un[Ir + 1, Jn + 1]], -1)
        le = np.stack([Ln[Ir, Jn], Ln[Ir, Jn + 1], Ln[Ir + 1, Jn], Ln[Ir + 1, Jn + 1]], -1)
        return np.stack([np.einsum("...i,ij,...j->...", le, ts.Q1[q], ue)
                         for q in ("xx", "yy", "xy")], -1)

    def local_dk(self, dsig_blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(flat element indices, their dk (m, 2, 2)) where ts.element_tensors(dsig_blocks)
        is nonzero, evaluated only at the elements whose bilinear stencil reaches a changed
        block (pointwise the same map_coordinates)."""
        nz = np.argwhere(np.abs(dsig_blocks).reshape(*dsig_blocks.shape[:2], 4).max(-1) > 0)
        if len(nz) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 2, 2))
        (bi0, bj0), (bi1, bj1) = nz.min(0), nz.max(0)
        nbi, nbj = dsig_blocks.shape[:2]
        s, k = self.s, self.k
        yc = ((np.arange(self.NI - 1) + 0.5) * k - (s / 2 - 0.5)) / s
        xc = ((np.arange(self.NJ - 1) + 0.5) * k - (s / 2 - 0.5)) / s
        rows = np.flatnonzero((yc > (bi0 - 1 if bi0 > 0 else -np.inf))
                              & (yc < (bi1 + 1 if bi1 < nbi - 1 else np.inf)))
        cols = np.flatnonzero((xc > (bj0 - 1 if bj0 > 0 else -np.inf))
                              & (xc < (bj1 + 1 if bj1 < nbj - 1 else np.inf)))
        Y, X = np.meshgrid(yc[rows], xc[cols], indexing="ij")
        dk = np.zeros((len(rows), len(cols), 2, 2))
        for a in range(2):
            for b in range(2):
                dk[..., a, b] = ndimage.map_coordinates(dsig_blocks[..., a, b], [Y, X], order=1,
                                                        mode="nearest")
        flat = dk.reshape(-1, 4)
        keep = np.abs(flat).max(axis=1) > 0
        elems = (rows[:, None] * (self.NJ - 1) + cols[None, :]).ravel()[keep]
        return elems, dk.reshape(-1, 2, 2)[keep]

    def first_order(self, dsig_blocks: np.ndarray) -> float:
        """-sum_e S_e : dk_e (needs adjoint=True)."""
        elems, dk = self.local_dk(dsig_blocks)
        S = self.S.reshape(-1, 3)[elems]
        return -float((S[:, 0] * dk[:, 0, 0] + S[:, 1] * dk[:, 1, 1]
                       + S[:, 2] * dk[:, 0, 1]).sum())

    def delta(self, dsig_blocks: np.ndarray, rtol: float = DELTA_RTOL,
              margin: int = MARGIN) -> dict:
        """dJ_2 after the block tensors change by dsig_blocks: the macro re-solve (two-scale)
        and the first-order prediction (needs adjoint=True)."""
        t0 = time.perf_counter()
        elems, dk = self.local_dk(dsig_blocks)
        if len(elems) == 0:
            return dict(dJ=0.0, dJ_lin=0.0, iters=0, n_elems=0, n_loc=0, seconds=0.0)
        S = self.S.reshape(-1, 3)[elems]
        dJ_lin = -float((S[:, 0] * dk[:, 0, 0] + S[:, 1] * dk[:, 1, 1]
                         + S[:, 2] * dk[:, 0, 1]).sum())
        dA = assemble_elems(dk, elems, self.unk, self.N)
        dA.eliminate_zeros()
        if dA.nnz == 0:
            return dict(dJ=0.0, dJ_lin=dJ_lin, iters=0, n_elems=len(elems), n_loc=0,
                        seconds=time.perf_counter() - t0)
        A, N = self.A, self.N

        def Aop(x):
            return A @ x + dA @ x

        NJm = self.NJ - 1
        ei, ej = elems // NJm, elems % NJm
        lo_i, hi_i = max(ei.min() - margin, 0), min(ei.max() + 2 + margin, self.NI)
        lo_j, hi_j = max(ej.min() - margin, 0), min(ej.max() + 2 + margin, self.NJ)
        box = self.unk[lo_i:hi_i, lo_j:hi_j]
        ids = np.sort(box[box >= 0])
        A_rows = (A[ids] + dA[ids]).tocsr()
        lu = splu(A_rows[:, ids].tocsc())
        vcycle = self.vcycle

        def M(r):
            z = np.zeros(N)
            z[ids] = lu.solve(r[ids])
            q = r - A_rows.T @ z[ids]
            z += vcycle(q)
            q = r - Aop(z)
            z[ids] += lu.solve(q[ids])
            return z

        r0 = -(dA @ self.U0)
        d, it, _ = pcg(Aop, r0, M, rtol, 1000)
        dh = self.H @ d
        lv = self.fab.live
        dJ = float((self.fab.w[lv] * (2 * self.uh0[lv] + dh[lv]) * dh[lv]).sum())
        return dict(dJ=dJ, dJ_lin=dJ_lin, iters=it, n_elems=len(elems), n_loc=len(ids),
                    seconds=time.perf_counter() - t0)
