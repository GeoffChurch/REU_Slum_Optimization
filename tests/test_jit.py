"""The typed njit wrapper compiles and keeps the function's signature for mypy."""
from __future__ import annotations

import numba.core.registry
import numpy as np
from numpy.typing import NDArray

from reblock._jit import njit


@njit
def _total(a: NDArray[np.float64]) -> float:
    s = 0.0
    for i in range(a.shape[0]):
        s += a[i]
    return s


def test_a_jitted_function_runs_compiled() -> None:
    assert _total(np.arange(10.0)) == 45.0
    # What `numba.njit` returns, and compiled for the call above -- not the plain function.
    assert isinstance(_total, numba.core.registry.CPUDispatcher)
    assert _total.signatures
