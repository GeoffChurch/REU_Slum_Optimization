"""The GW core: the outer loop's gradient, the exact objective, the unbalanced inner solve, and the
coupling it all produces.

The gradient test is the one that matters most. The factor of 2 it checks was missing for the
spike's whole first life and nothing failed -- Sinkhorn at (2*cost, 2*eps, 2*tau) has the same
argmin as at (cost, eps, tau), so the solver quietly ran at twice the regularization it was asked
for (docs/superpowers/notes/2026-07-27-gw-pot-crossvalidation.md). Only a check against the
objective's own derivative can see it.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from reblock.transplant.gw import (
    GWParams,
    barycentric_projection,
    entropic_gw_unbalanced,
    gw_cost,
    gw_gradient,
    sinkhorn_unbalanced,
)
from reblock.transplant.transport import normalized_dist_matrix


def _clouds(seed: int, n: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (normalized_dist_matrix(rng.normal(size=(n, 2))),
            normalized_dist_matrix(rng.normal(size=(m, 2))))


def test_gradient_is_the_objectives_own_derivative() -> None:
    """At a coupling whose marginals are exactly `p` and `q`, `gw_gradient` must equal the
    derivative of `gw_cost` -- factor of 2 included.

    FAULT INJECTION: dropping the leading `2.0 *` from `gw_gradient` halves every entry and fails
    this at ratio 0.5.
    """
    c1, c2 = _clouds(1, 6, 5)
    p, q = np.full(6, 1 / 6), np.full(5, 1 / 5)
    pi = np.outer(p, q)
    grad = gw_gradient(c1, c2, pi, p, q)
    h = 1e-6
    numeric = np.empty_like(pi)
    for i in range(6):
        for j in range(5):
            up, down = pi.copy(), pi.copy()
            up[i, j] += h
            down[i, j] -= h
            numeric[i, j] = (gw_cost(up, c1, c2) - gw_cost(down, c1, c2)) / (2 * h)
    np.testing.assert_allclose(grad, numeric, rtol=1e-6, atol=1e-9)


def test_gw_cost_is_the_quadratic_objective() -> None:
    """The two-matmul expansion against the O(n^2 m^2) definition, on a coupling with arbitrary
    (unbalanced) marginals -- the expansion must use pi's realized marginals, not p and q."""
    c1, c2 = _clouds(2, 7, 4)
    pi = np.random.default_rng(3).uniform(size=(7, 4)) * 0.1
    brute = float(np.einsum("ij,kl,ijkl->", pi, pi,
                            (c1[:, None, :, None] - c2[None, :, None, :]) ** 2))
    assert gw_cost(pi, c1, c2) == pytest.approx(brute, rel=1e-12)


def test_large_tau_binds_the_marginals_and_small_tau_releases_them() -> None:
    """tau -> inf is balanced Sinkhorn (exact marginals); a finite tau lets mass drift toward the
    cheap cells instead. Both directions, so neither can pass by the solver doing nothing."""
    cost = np.random.default_rng(4).uniform(size=(8, 6))
    p, q = np.full(8, 1 / 8), np.full(6, 1 / 6)
    tight = sinkhorn_unbalanced(cost, p, q, eps=0.05, tau=1e6, n_iter=2000)
    np.testing.assert_allclose(tight.sum(axis=1), p, atol=1e-5)
    np.testing.assert_allclose(tight.sum(axis=0), q, atol=1e-5)
    loose = sinkhorn_unbalanced(cost, p, q, eps=0.05, tau=0.05, n_iter=2000)
    assert np.abs(loose.sum(axis=1) - p).max() > 1e-3


def test_gw_recovers_a_rigid_motion_from_distances_alone() -> None:
    """A rotated, translated and reshuffled copy of a cloud has the same distance matrix up to the
    shuffle, so the coupling's argmax must undo the shuffle. Index correspondence is erased; only
    the intrinsic geometry can recover it."""
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 100, size=(25, 2))
    theta = 0.9
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    perm = rng.permutation(25)
    y = (x @ rot.T + [500.0, -200.0])[perm]
    p = q = np.full(25, 1 / 25)
    pi = entropic_gw_unbalanced(normalized_dist_matrix(x), normalized_dist_matrix(y), p, q,
                                GWParams(eps=0.01, tau=1.0, outer_iters=30, inner_iters=100))
    assert (perm[pi.argmax(axis=1)] == np.arange(25)).mean() >= 0.9


def test_a_row_with_no_mass_projects_to_the_recipient_centroid() -> None:
    y = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 20.0]])
    pi = np.array([[0.2, 0.1, 0.0], [0.0, 0.0, 0.0]])
    out = barycentric_projection(pi, y)
    np.testing.assert_allclose(out[0], [10.0 * 0.1 / 0.3, 0.0])
    np.testing.assert_allclose(out[1], y.mean(axis=0))


def _log_domain_reference(cost: np.ndarray, p: np.ndarray, q: np.ndarray, *, eps: float,
                          tau: float, n_iter: int) -> np.ndarray:
    """The plain log-domain update, written out: what `sinkhorn_unbalanced` must reproduce."""
    fw = tau / (tau + eps)
    f, g = np.zeros(len(p)), np.zeros(len(q))
    for _ in range(n_iter):
        f = fw * eps * (np.log(p) - logsumexp((g[None, :] - cost) / eps, axis=1))
        g = fw * eps * (np.log(q) - logsumexp((f[:, None] - cost) / eps, axis=0))
    return np.exp((f[:, None] + g[None, :] - cost) / eps)


@pytest.mark.parametrize("absorb_log", [100.0, 1e-9])   # the shipped threshold; absorb every step
@pytest.mark.parametrize("n_iter", [1, 2, 100])
def test_scaling_iterates_are_the_log_domain_iterates(
        monkeypatch: pytest.MonkeyPatch, n_iter: int, absorb_log: float) -> None:
    """The scaling-domain solver is a faster way to compute the SAME iterates, not a different
    solver: the GW fits, and everything measured from them, rest on that. Holds with absorption
    off and forced on every step, and at an eps small enough that an unabsorbed kernel would
    underflow (cost/eps reaches 900 here, past exp's ~745 floor).

    FAULT INJECTION: dropping the `+ fa / eps` term from the u update fails every case that
    reaches the scaling loop (n_iter > 1); rebuilding the absorbed kernel without `ga` fails every
    case that absorbs (the forced-absorption ones)."""
    import reblock.transplant.gw as gw_mod
    monkeypatch.setattr(gw_mod, "_ABSORB_LOG", absorb_log)
    rng = np.random.default_rng(7)
    for eps in (0.01, 0.001):
        cost = rng.uniform(0.0, 0.9, size=(23, 31))
        p, q = rng.dirichlet(np.ones(23)), rng.dirichlet(np.ones(31))
        want = _log_domain_reference(cost, p, q, eps=eps, tau=1.0, n_iter=n_iter)
        got = sinkhorn_unbalanced(cost, p, q, eps=eps, tau=1.0, n_iter=n_iter)
        assert np.all(np.isfinite(got))
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-300)


def test_sinkhorn_needs_an_iteration() -> None:
    with pytest.raises(ValueError, match="n_iter"):
        sinkhorn_unbalanced(np.zeros((2, 2)), np.full(2, 0.5), np.full(2, 0.5), eps=0.01,
                            tau=1.0, n_iter=0)
