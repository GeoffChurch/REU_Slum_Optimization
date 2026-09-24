from __future__ import annotations

import numpy as np

from reblock.methods.betweenness.contrast import deviance_field, raw_share_field


def test_raw_share_ignores_nan_outside_the_block() -> None:
    o1 = np.array([[np.nan, 1.0, 2.0]], np.float32)
    o2 = np.array([[np.nan, 10.0, 5.0]], np.float32)
    f = raw_share_field(o1, o2)
    assert np.isnan(f[0, 0]) and np.allclose(f[0, 1:], [0.5 + 1.0, 1.0 + 0.5])


def test_deviance_is_zero_where_observation_equals_prior_and_positive_where_it_exceeds() -> None:
    o = np.array([[0.0, 5.0, 50.0, 1.0]], np.float32)
    e = np.array([[0.0, 5.0, 2.0, 10.0]], np.float32)
    z = deviance_field(o, np.zeros_like(o), e, np.zeros_like(e), floor=1e-9)
    assert np.isclose(z[0, 0], -np.sqrt(2e-9)) and z[0, 1] == 0.0 and z[0, 2] > 5 and z[0, 3] < -1
