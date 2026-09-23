"""The donor signature is a shape descriptor: blind to where a cloud is, how it is turned, how big
it is and how its points are numbered -- and it refuses a cloud too small to sample."""
from __future__ import annotations

import numpy as np
import pytest

from reblock.transplant.signature import SignatureParams, signature, signature_distance

EXACT = SignatureParams(n_sub=30, n_boot=3, seed=0)     # n_sub == cloud size: no subsampling


def test_invariant_to_rigid_motion_scale_and_numbering() -> None:
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 50, size=(30, 2))
    theta = 1.1
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    moved = (3.5 * xy @ rot.T + [900.0, -40.0])[rng.permutation(30)]
    assert signature_distance(signature(xy, EXACT), signature(moved, EXACT)) < 1e-9


def test_separates_different_shapes() -> None:
    """Without this the invariance test passes for a constant function."""
    rng = np.random.default_rng(1)
    blob = rng.normal(size=(30, 2))
    strip = np.c_[rng.uniform(0, 100, 30), rng.uniform(0, 5, 30)]
    assert signature_distance(signature(blob, EXACT), signature(strip, EXACT)) > 0.5


def test_bootstraps_over_consecutive_seeds() -> None:
    """A cloud above n_sub is subsampled once per seed and the spectra averaged, so the result is
    the mean of the single-draw signatures at seeds seed .. seed + n_boot - 1."""
    xy = np.random.default_rng(2).uniform(0, 50, size=(80, 2))
    singles = [signature(xy, SignatureParams(n_sub=30, n_boot=1, seed=s)) for s in (4, 5, 6)]
    np.testing.assert_allclose(signature(xy, SignatureParams(n_sub=30, n_boot=3, seed=4)),
                               np.mean(singles, axis=0))


def test_refuses_a_cloud_smaller_than_the_subsample() -> None:
    with pytest.raises(ValueError, match="need >= 30"):
        signature(np.random.default_rng(3).uniform(size=(29, 2)), EXACT)
