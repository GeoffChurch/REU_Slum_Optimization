"""numba's `njit`, typed: the decorated function keeps its own signature for mypy.

numba has no stubs, so `numba.njit` is `Any` and would erase every kernel's types under
--strict (`disallow_untyped_decorators`). This is the one place that cast happens."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

import numba

F = TypeVar("F", bound=Callable[..., Any])


def njit(fn: F) -> F:
    """`numba.njit(cache=True)`: compiled on first call, the machine code cached on disk."""
    return cast(F, numba.njit(cache=True)(fn))
