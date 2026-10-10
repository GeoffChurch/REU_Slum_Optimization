"""FFT homogenization of the scalar 'uni' conductance on a periodic raster window.

Discretization (the one the direct solver also uses): cell-centred potential u on an (ny, nx)
torus, one unknown per cell; an x-face joins (i, j) and (i, j+1), a y-face (i, j) and (i+1, j)
(indices mod n); a face conducts c = min(o_a, o_b), o the cell's open fraction (the uni rule for
an axis step: the smallest open fraction among the cells it crosses). Buildings (o = 0) conduct
nothing: infinite contrast. Equilibrium for a macroscopic gradient E:

    A u = b,   A = D^T C D,   b = -D^T C E,

D the forward differences (periodic), C = diag(face conductances). The flux on a face is
q = C (E + D u) and the effective (apparent, periodic) tensor is sigma_ij = <q_i> for E = e_j.
In open space sigma = I. h cancels (a face's conductance is c * h / h).

The Green operator of the reference medium (c0 = 1) is the inverse of the periodic 5-point
Laplacian L = D^T D, diagonal under the FFT with symbol 4 sin^2(pi k_x / n_x) + 4 sin^2(pi k_y
/ n_y): the staggered-grid / finite-volume-consistent Green operator (Schneider, Ospald & Kabel
2016; for conductivity, Willot, Abdallah & Pellegrini 2014 discuss the equivalent modified
operator). With it every scheme below solves exactly the finite-volume system the sparse direct
solver solves, so they can be compared to round-off.

Schemes (all on the same discrete problem):
  ms      Moulinec-Suquet basic scheme = preconditioned Richardson u += L^+ r / c0
  cg      conjugate gradients preconditioned by L^+ (Zeman et al. 2010; Brisard & Dormieux 2010:
          the Galerkin/variational formulation, which tolerates zero-conductance phases)
  al      augmented Lagrangian (Michel, Moulinec & Suquet 2001) written as ADMM with the staggered
          projection, relaxation 1
  em      Eyre-Milton = the same splitting with relaxation 2 (Peaceman-Rachford)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import scipy.fft as sfft
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve


def faces(o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cx, cy): x-face (i, j)-(i, j+1) and y-face (i, j)-(i+1, j) conductances, periodic."""
    o = np.asarray(o, dtype=np.float64)
    return np.minimum(o, np.roll(o, -1, axis=1)), np.minimum(o, np.roll(o, -1, axis=0))


def grad(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward differences on the torus, over the last two axes."""
    return np.roll(u, -1, axis=-1) - u, np.roll(u, -1, axis=-2) - u


def div_T(qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
    """D^T q (minus the divergence): (D^T q)_ij = q_x[i,j-1] - q_x[i,j] + q_y[i-1,j] - q_y[i,j]."""
    return np.roll(qx, 1, axis=-1) - qx + np.roll(qy, 1, axis=-2) - qy


class Green:
    """L^+ on an (ny, nx) torus by real FFTs (zero mode dropped)."""

    def __init__(self, ny: int, nx: int):
        ky = np.fft.fftfreq(ny)[:, None]
        kx = np.fft.rfftfreq(nx)[None, :]
        lam = 4 * np.sin(np.pi * ky) ** 2 + 4 * np.sin(np.pi * kx) ** 2
        lam[0, 0] = np.inf
        self.inv = 1.0 / lam
        self.shape = (ny, nx)

    def __call__(self, r: np.ndarray) -> np.ndarray:
        # workers=1: the machine is shared (bit-identical results, only speed)
        R = sfft.rfft2(r, axes=(-2, -1), workers=1)
        return sfft.irfft2(R * self.inv, s=self.shape, axes=(-2, -1), workers=1)


class Result(NamedTuple):
    sigma: np.ndarray        # (2, 2) effective tensor, symmetrized
    asym: float              # |s01 - s10| / ||sigma|| before symmetrizing
    iters: int
    seconds: float
    resid: float             # final ||A u - b|| / ||b|| (max over the two loads)
    u: np.ndarray            # (2, ny, nx) fluctuation fields chi_x, chi_y (zero mean)
    history: list            # residual per iteration (max over loads)


@dataclass(frozen=True)
class Problem:
    cx: np.ndarray
    cy: np.ndarray

    @classmethod
    def of(cls, o: np.ndarray) -> Problem:
        cx, cy = faces(o)
        return cls(cx, cy)

    @property
    def shape(self) -> tuple[int, int]:
        return self.cx.shape

    def A(self, u: np.ndarray) -> np.ndarray:
        gx, gy = grad(u)
        return div_T(self.cx * gx, self.cy * gy)

    def rhs(self, E: np.ndarray) -> np.ndarray:
        """b for each load E (m, 2) -> (m, ny, nx)."""
        E = np.atleast_2d(E)
        return np.stack([-div_T(self.cx * e[0], self.cy * e[1]) for e in E])

    def flux_mean(self, u: np.ndarray, E: np.ndarray) -> np.ndarray:
        """(m, 2): <q> for each load."""
        gx, gy = grad(u)
        E = np.atleast_2d(E)
        return np.stack([[np.mean(self.cx * (gx[m] + E[m, 0])),
                          np.mean(self.cy * (gy[m] + E[m, 1]))] for m in range(len(E))])

    def sigma_of(self, u: np.ndarray) -> tuple[np.ndarray, float]:
        """Tensor from the two unit loads' fluctuations: column j = <q> under E = e_j."""
        S = self.flux_mean(u, np.eye(2)).T
        asym = abs(S[0, 1] - S[1, 0]) / max(np.linalg.norm(S), 1e-300)
        return 0.5 * (S + S.T), float(asym)


def _resid(P: Problem, u: np.ndarray, b: np.ndarray, nb: np.ndarray) -> np.ndarray:
    r = b - P.A(u)
    return np.sqrt((r ** 2).sum(axis=(-2, -1))) / nb


def solve(P: Problem, E: np.ndarray, scheme: str, tol: float, maxiter: int,
          c0: float, b_extra: np.ndarray | None = None):
    """Fluctuations u (m, ny, nx) for loads E (m, 2) [plus extra right-hand sides b_extra (k, ny,
    nx), e.g. a source corrector's]. Returns (u, iterations, residual history)."""
    b = P.rhs(E)
    if b_extra is not None:
        b = np.concatenate([b, b_extra])
    nb = np.sqrt((b ** 2).sum(axis=(-2, -1)))
    nb = np.where(nb > 0, nb, 1.0)
    G = Green(*P.shape)
    hist = []
    if scheme == "cg":
        u = np.zeros_like(b)
        r = b.copy()
        z = G(r)
        p = z.copy()
        rz = (r * z).sum(axis=(-2, -1))
        for _it in range(1, maxiter + 1):
            Ap = P.A(p)
            pAp = (p * Ap).sum(axis=(-2, -1))
            alpha = np.where(pAp > 0, rz / np.where(pAp > 0, pAp, 1), 0.0)
            u += alpha[:, None, None] * p
            r -= alpha[:, None, None] * Ap
            res = np.sqrt((r ** 2).sum(axis=(-2, -1))) / nb
            hist.append(float(res.max()))
            if res.max() < tol:
                break
            z = G(r)
            rz_new = (r * z).sum(axis=(-2, -1))
            beta = np.where(rz > 0, rz_new / np.where(rz > 0, rz, 1), 0.0)
            p = z + beta[:, None, None] * p
            rz = rz_new
        # recompute the true residual (recurrence drift)
        return u, _it, hist, float(_resid(P, u, b, nb).max())
    if scheme == "ms":
        u = np.zeros_like(b)
        for _it in range(1, maxiter + 1):
            r = b - P.A(u)
            res = np.sqrt((r ** 2).sum(axis=(-2, -1))) / nb
            hist.append(float(res.max()))
            if res.max() < tol:
                break
            u += G(r) / c0
        return u, _it, hist, hist[-1]
    if scheme in ("al", "em"):
        if b_extra is not None:
            raise ValueError("ADMM schemes here take gradient loads only")
        rho = 1.0 if scheme == "al" else 2.0
        E = np.atleast_2d(E)
        m = len(E)
        cx, cy = P.cx, P.cy
        # local fields on faces: eps (where the constitutive law holds), lam (multiplier)
        ex = np.repeat(E[:, 0, None, None], P.shape[0], 1).repeat(P.shape[1], 2).astype(float)
        ey = np.repeat(E[:, 1, None, None], P.shape[0], 1).repeat(P.shape[1], 2).astype(float)
        epx, epy = ex.copy(), ey.copy()
        lx, ly = cx * ex, cy * ey
        u = np.zeros((m, *P.shape))
        for _it in range(1, maxiter + 1):
            # compatible field: e = E + D u, u = L^+ D^T (eps - lam / c0)
            u = G(div_T(epx - lx / c0, epy - ly / c0))
            gx, gy = grad(u)
            cpx, cpy = ex + gx, ey + gy
            res = _resid(P, u, b, nb)
            hist.append(float(res.max()))
            if res.max() < tol:
                break
            hx, hy = rho * cpx + (1 - rho) * epx, rho * cpy + (1 - rho) * epy
            epx = (lx + c0 * hx) / (cx + c0)
            epy = (ly + c0 * hy) / (cy + c0)
            lx = lx + c0 * (hx - epx)
            ly = ly + c0 * (hy - epy)
        return u, _it, hist, hist[-1]
    raise ValueError(scheme)


def homogenize(o: np.ndarray, scheme: str = "cg", tol: float = 1e-8, maxiter: int = 20000,
               c0: float = 0.5) -> Result:
    P = Problem.of(o)
    t = time.perf_counter()
    u, it, hist, res = solve(P, np.eye(2), scheme, tol, maxiter, c0)
    sec = time.perf_counter() - t
    S, asym = P.sigma_of(u)
    return Result(S, asym, it, sec, res, u, hist)


# ---------------------------------------------------------------- direct sparse reference
def periodic_matrix(cx: np.ndarray, cy: np.ndarray) -> sp.csr_matrix:
    ny, nx = cx.shape
    idx = np.arange(ny * nx).reshape(ny, nx)
    a_x, b_x = idx.ravel(), np.roll(idx, -1, axis=1).ravel()
    a_y, b_y = idx.ravel(), np.roll(idx, -1, axis=0).ravel()
    a = np.concatenate([a_x, a_y])
    b = np.concatenate([b_x, b_y])
    w = np.concatenate([cx.ravel(), cy.ravel()])
    keep = w > 0
    a, b, w = a[keep], b[keep], w[keep]
    n = ny * nx
    W = sp.coo_matrix((np.concatenate([-w, -w]), (np.concatenate([a, b]), np.concatenate([b, a]))),
                      shape=(n, n)).tocsr()
    d = -np.asarray(W.sum(axis=1)).ravel()
    return (W + sp.diags(d)).tocsr()


def direct(o: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Same discretization, sparse direct (SuperLU): one pinned cell per conducting component,
    cells with no conducting face dropped. Returns (sigma, seconds, u)."""
    P = Problem.of(o)
    t = time.perf_counter()
    A = periodic_matrix(P.cx, P.cy)
    n = A.shape[0]
    deg = A.diagonal()
    live = deg > 0
    ncomp, lab = connected_components(A, directed=False)
    pin = np.zeros(n, dtype=bool)
    for c in np.unique(lab[live]):
        pin[np.flatnonzero((lab == c) & live)[0]] = True
    unk = live & ~pin
    Au = A[unk][:, unk].tocsc()
    b = P.rhs(np.eye(2)).reshape(2, -1)
    u = np.zeros((2, n))
    for m in range(2):
        u[m, unk] = spsolve(Au, b[m, unk])
    sec = time.perf_counter() - t
    u = u.reshape(2, *P.shape)
    S, _ = P.sigma_of(u)
    return S, sec, u


def principal(S: np.ndarray) -> tuple[float, float, float]:
    """(major, minor, angle of the major axis in degrees in [0, 180), from +x = east,
    counter-clockwise; rows are northing)."""
    w, v = np.linalg.eigh(S)
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1])) % 180.0
    return float(w[1]), float(w[0]), float(ang)
