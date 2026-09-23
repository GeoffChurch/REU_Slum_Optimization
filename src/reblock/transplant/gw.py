"""Entropic unbalanced Gromov-Wasserstein between two clouds' intra-cloud distance matrices.

Peyre-Cuturi-Solomon (2016) entropic GW: an OUTER loop linearizes the square-loss GW objective
around the current coupling, and an INNER loop solves that linear problem by Sinkhorn. The inner
solve is the UNBALANCED (KL-marginal-relaxed) generalized Sinkhorn, so two clouds of different
sizes need not be matched mass-for-mass and the coupling can concentrate on genuinely well-matched
points instead of spreading thin. The inner solve is LOG-STABILIZED: potentials absorbed into the
kernel keep every exponential in range at any eps, while the iterations themselves run in the
scaling domain (`sinkhorn_unbalanced`).

Hand-rolled because POT has no unbalanced entropic GW at all, and cross-validated against POT
(docs/superpowers/notes/2026-07-27-gw-pot-crossvalidation.md): the inner solver reaches the lowest
primal objective of three solvers in all nine configurations tried, `gw_cost` matches a brute-force
einsum to machine precision, the outer loop matches POT's balanced GW once the gradient carries its
factor of 2 (`gw_gradient`), and it runs ~5x faster than POT over the same 100 pairs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Arr: TypeAlias = NDArray[np.float64]

# A scaling past this (in log units) is folded back into the potentials and the kernel rebuilt, so
# neither the scalings nor the kernel can over- or underflow. Far above anything the operating point
# reaches (costs stay under ~70 eps), so it is a safeguard, not a schedule.
_ABSORB_LOG = 100.0

# A donor row carrying less coupling mass than this has been zeroed out by the unbalanced solve;
# its barycentric image is undefined, so it takes the recipient centroid instead of a 0/0.
_MASS_FLOOR = 1e-12


@dataclass(frozen=True)
class GWParams:
    """One entropic unbalanced GW solve, in POT's convention for `eps` and `tau`.

    That convention is what the gradient's factor of 2 buys (`gw_gradient`). Before 2026-07-27 the
    factor was missing and every solve silently ran at twice the `eps` and `tau` it was given, so
    halve any value quoted from an earlier note to read it here.
    """

    eps: float          # entropic regularization, on distance matrices normalized to [0, 1]
    tau: float          # marginal KL relaxation; tau -> inf recovers balanced Sinkhorn
    outer_iters: int    # linearizations of the GW objective
    inner_iters: int    # Sinkhorn iterations per linearization


def _logsumexp(a: Arr, axis: int) -> Arr:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)          # an all -inf row/col -> logsumexp = -inf cleanly
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def sinkhorn_unbalanced(cost: Arr, p: Arr, q: Arr, *, eps: float, tau: float,
                        n_iter: int) -> Arr:
    """Unbalanced entropic OT:

        min_pi <cost, pi> + eps*KL(pi | p q^T) + tau*KL(pi 1 | p) + tau*KL(pi^T 1 | q)

    Finite `tau` lets a row's or column's total mass drift from `p`/`q` when that lowers the
    transport cost. Dual potentials f (rows), g (cols); each update is the balanced one damped by
    tau/(tau+eps). Returns the primal coupling exp((f_i + g_j - cost_ij) / eps), shape (n, m).

    The iterates of the plain log-domain update

        f <- fw * eps * (log p - logsumexp_j((g_j - cost_ij) / eps)),   likewise g,

    computed in the scaling domain: with the potentials split as f = fa + eps*log u (and g
    likewise) and fa, ga absorbed into the kernel K_ij = exp((fa_i + ga_j - cost_ij) / eps), each
    half step is ONE matrix-vector product instead of n*m exponentials -- measured 6.4x faster on
    real block pairs, and equal to the log-domain iterates to ~1e-13 relative. The first half step
    runs in the log domain so the kernel starts centred, and a scaling that grows past
    `_ABSORB_LOG` is absorbed and the kernel rebuilt, so no exponential over- or underflows.
    """
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    n, m = cost.shape
    logp, logq = np.log(p), np.log(q)
    fw = tau / (tau + eps)
    # Iteration 1 exactly as the log-domain update (g starts at 0), then absorbed.
    fa = fw * eps * (logp - _logsumexp(-cost / eps, axis=1))
    ga = fw * eps * (logq - _logsumexp((fa[:, None] - cost) / eps, axis=0))
    kernel = np.exp((fa[:, None] + ga[None, :] - cost) / eps)
    log_u = np.zeros(n, dtype=np.float64)
    log_v = np.zeros(m, dtype=np.float64)
    for _ in range(n_iter - 1):
        # f/eps = fw*(log p - log(K v) + fa/eps), with K carrying fa and ga.
        log_u = fw * (logp - np.log(kernel @ np.exp(log_v)) + fa / eps) - fa / eps
        log_v = fw * (logq - np.log(kernel.T @ np.exp(log_u)) + ga / eps) - ga / eps
        if max(np.abs(log_u).max(), np.abs(log_v).max()) > _ABSORB_LOG:
            fa, ga = fa + eps * log_u, ga + eps * log_v
            kernel = np.exp((fa[:, None] + ga[None, :] - cost) / eps)
            log_u = np.zeros(n, dtype=np.float64)
            log_v = np.zeros(m, dtype=np.float64)
    f, g = fa + eps * log_u, ga + eps * log_v
    return np.exp((f[:, None] + g[None, :] - cost) / eps)


def gw_gradient(c1: Arr, c2: Arr, pi: Arr, p: Arr, q: Arr) -> Arr:
    """The square-loss GW objective's GRADIENT at `pi` (Peyre-Cuturi-Solomon Prop. 2):

        grad(pi) = 2 * (constC - 2 * c1 @ pi @ c2^T),
        constC[i, j] = sum_k c1[i,k]^2 p[k] + sum_l c2[j,l]^2 q[l]

    The leading 2 is load-bearing. Eq. 6 defines the tensor product; Prop. 2's gradient carries a
    factor of 2 the paper's own statement omits (POT's `gwggrad` ships the same correction, with a
    comment saying so). Sinkhorn at (cost, eps, tau) and (2*cost, 2*eps, 2*tau) share an argmin, so
    dropping it raises no error: it silently doubles the regularization.
    """
    const_c1 = (c1 ** 2) @ p
    const_c2 = (c2 ** 2) @ q
    const_c = const_c1[:, None] + const_c2[None, :]
    return 2.0 * (const_c - 2.0 * (c1 @ pi) @ c2.T)


def entropic_gw_unbalanced(c1: Arr, c2: Arr, p: Arr, q: Arr, params: GWParams) -> Arr:
    """The (n, m) coupling between two clouds with distance matrices `c1` (n, n) and `c2` (m, m),
    which the caller has already normalized (GW itself does not rescale). `p`, `q` are the target
    marginals, which the unbalanced solve need not bind exactly.

    Starts from the independent coupling. Each outer step re-linearizes at the current coupling and
    shifts the gradient to a zero minimum, which keeps Sinkhorn's costs non-negative. That shift is
    exactly invariant under balanced Sinkhorn but not under the unbalanced one, where a constant
    trades against created mass; measured, it moves the objective by ~0.05% relative with mixed
    sign across shapes, so it stays.
    """
    pi = np.outer(p, q)
    for _ in range(params.outer_iters):
        cost = gw_gradient(c1, c2, pi, p, q)
        cost = cost - cost.min()
        pi = sinkhorn_unbalanced(cost, p, q, eps=params.eps, tau=params.tau,
                                 n_iter=params.inner_iters)
    return pi


def gw_cost(pi: Arr, c1: Arr, c2: Arr) -> float:
    """The EXACT square-loss GW objective of a fitted coupling -- not the outer loop's linearized
    surrogate -- sum_ijkl pi[i,j] pi[k,l] (c1[i,k] - c2[j,l])^2, expanded over `pi`'s own realized
    (possibly non-normalized, under UOT) marginals r = pi 1 and c = pi^T 1:

        r^T (c1^2) r  +  c^T (c2^2) c  -  2 * sum(pi * (c1 @ pi @ c2))

    The cross term uses c2's symmetry and costs two matrix products, not the O(n^2 m^2) einsum.
    """
    row_m = pi.sum(axis=1)
    col_m = pi.sum(axis=0)
    term1 = float(row_m @ (c1 ** 2) @ row_m)
    term3 = float(col_m @ (c2 ** 2) @ col_m)
    term2 = 2.0 * float(np.sum(pi * (c1 @ pi @ c2)))
    return term1 + term3 - term2


def barycentric_projection(pi: Arr, y: Arr) -> Arr:
    """Each donor point's image: the coupling-weighted centroid of the recipient points `y`,
    T(x_i) = sum_j pi[i,j] y[j] / sum_j pi[i,j]. A row the unbalanced solve left with no mass
    takes the unweighted recipient centroid."""
    row_mass = pi.sum(axis=1, keepdims=True)
    safe = row_mass > _MASS_FLOOR
    out = np.divide(pi @ y, row_mass, out=np.zeros((pi.shape[0], y.shape[1])), where=safe)
    out[~safe.ravel()] = y.mean(axis=0)
    return out
