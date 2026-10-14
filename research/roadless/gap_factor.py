"""The gap complex's factorization, a Strategy (NOTES, "The complex as the runtime: its speed and a
GPU path"): what the fixed mesh's model (gap_clear.System) needs from a sparse direct solver, and
its two implementations, chosen once where a run is configured (metrics.metric_of: the run's
device) and handed down; below that edge no code asks which it has.

THE PROTOCOL. A `Pattern` is a fixed sparsity pattern: a symmetric CSR structure over n rows (the
cells of one cell structure of the fixed mesh), its diagonal stored, with a point per row (the
cells' centroids, for a geometric order). `Factorization.plan(pattern)` makes the pattern's plan
once (`Planned`); `Planned.factor(data, unknown)` refactors the pattern with new values (`data`,
aligned with the pattern's indices: a row off the state's unknowns an identity row, its
off-diagonals explicit zeros) and returns the state's `Factor`, whose `solve(b)` gives x over the
pattern's n rows (0 off the unknowns), refined to the floating-point floor (`Refined`). A plan
holds one factorization at a time (cuDSS's): a factor used after a later one on its plan raises.

SuperLU (`SuperLU`, on a core): the plan is one geometric nested-dissection order of the pattern
(`dissection` at the rows' points) and the pattern permuted into it; a factorization restricts the
permuted values to the state's unknowns, which keeps the order restricted to them, and drops the
explicit zeros (`_restrict`), then SuperLU factors that with no ordering of its own, pivoting on
the diagonal (the system is SPD: a weighted Laplacian with conductances to ground). Phase 1's
runtime, its order computed once and reused.

cuDSS (`CuDSS`, on a GPU, through nvmath-python): the plan is cuDSS's reordering and symbolic
factorization of the whole pattern (an SPD system), made at the first factorization; every state
is then a refactorization with new values on it (the operands on the host, nvmath moving them):
every row factored, a row off the unknowns an identity row.

The device changes scores only at the floating-point level: it is not in the metric's name, and
each run's rows record it (seam.Metric.device).

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/gap_factor.py
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, NamedTuple, Protocol

import numpy as np
import scipy.sparse as sp
from numba import njit
from numpy.typing import NDArray

REFINE_BERR = 4 * float(np.finfo(np.float64).eps)  # a solve is refined until its componentwise
                                                   # backward error max |r| / (|A| |x| + |b|) is
                                                   # this,
REFINE_ACCEPT = 1e-8  # ... or stops falling by half (refinement can do no better), when it must
                      # be at most this (a solve worse than that raises)
REFINE_STEPS = 8
ND_LEAF = 8           # dissection: subgraphs of at most this many nodes are not split (their own
                      # order kept: larger leaves fill more, measured on 20269)


class Refined:
    """A factorization's solves with iterative refinement (LAPACK's xGERFS stopping rule): a cell
    joined to the rest only through a hairline passage has a diagonal 1e-9 of the others', and
    SuperLU's partial pivoting then left residuals of 0.7% (relative) that the clearing model's
    finite differences saw; each solve is refined while its backward error falls. `once` solves
    A x = r with the factorization as it is."""

    def __init__(self, A: sp.csr_matrix, once: Callable[[NDArray[np.float64]], NDArray[np.float64]]):
        self.A = A
        self.once = once

    def _berr(self, b: NDArray[np.float64], x: NDArray[np.float64]
              ) -> tuple[float, NDArray[np.float64]]:
        berr, r = _residual(self.A.indptr, self.A.indices, self.A.data, b, x)
        return float(berr), r

    def solve(self, b: NDArray[np.float64]) -> NDArray[np.float64]:
        x = self.once(b)
        berr, r = self._berr(b, x)
        for _ in range(REFINE_STEPS):
            if berr <= REFINE_BERR:
                break
            x2 = x + self.once(r)
            berr2, r2 = self._berr(b, x2)
            if berr2 > 0.5 * berr:
                if berr2 < berr:
                    x, berr = x2, berr2
                break
            x, berr, r = x2, berr2, r2
        if berr > REFINE_ACCEPT:
            raise RuntimeError(f"refinement stalled at a backward error of {berr:.1e}")
        return x


@njit(cache=True)
def _residual(indptr, indices, data, b, x):  # type: ignore[no-untyped-def]
    """(the componentwise backward error max_i |r_i| / (|A| |x| + |b|)_i, 0 where that is 0; the
    residual r = b - A x), one pass over A (CSR)."""
    n = len(b)
    r = np.empty(n)
    berr = 0.0
    for i in range(n):
        s = b[i]
        d = abs(b[i])
        for p in range(indptr[i], indptr[i + 1]):
            v = data[p] * x[indices[p]]
            s -= v
            d += abs(v)
        r[i] = s
        if d > 0.0:
            e = abs(s) / d
            if e > berr:
                berr = e
    return berr, r


class Pattern(NamedTuple):
    """A fixed symmetric sparsity pattern over n rows (CSR, int32, each row's indices sorted, the
    diagonal stored), with a point per row (`xy`: the geometric order's)."""
    indptr: NDArray[np.int32]
    indices: NDArray[np.int32]
    xy: NDArray[np.float64]

    @property
    def n(self) -> int:
        return len(self.indptr) - 1

    @property
    def nnz(self) -> int:
        return len(self.indices)


class Factor(Protocol):
    """One state's factorization (module docstring)."""

    def solve(self, b: NDArray[np.float64]) -> NDArray[np.float64]:
        """x over the pattern's rows, 0 off the unknowns (b is 0 there), refined."""
        ...


class Planned(Protocol):
    """A pattern's plan (module docstring)."""

    def factor(self, data: NDArray[np.float64], unknown: NDArray[np.bool_]) -> Factor: ...


class Factorization(Protocol):
    """The factorization Strategy (module docstring): its device, recorded with each run's rows,
    and a plan per pattern."""

    @property
    def device(self) -> str: ...

    def plan(self, pattern: Pattern) -> Planned: ...


# ------------------------------------------------------------------ SuperLU, on a core

@njit(cache=True)
def dissection(indptr, indices, x, y, leaf):  # type: ignore[no-untyped-def]
    """A geometric nested dissection of the graph (CSR, symmetric) whose node v lies at (x[v],
    y[v]): an elimination order (perm[k] the node eliminated k-th). Each subgraph is split at the
    median of its longer side, its separator the left nodes with a neighbour on the right,
    ordered after both halves, recursively to `leaf` nodes. On a planar mesh the separators are
    of the order of the square root of the subgraph's size: fill O(n log n)."""
    n = len(x)
    perm = np.arange(n)
    stamp = -np.ones(n, dtype=np.int64)
    side = np.zeros(n, dtype=np.int8)
    stack_s = [0]
    stack_e = [n]
    seg = 0
    buf = np.empty(n, dtype=np.int64)
    while len(stack_s):
        s = stack_s.pop()
        e = stack_e.pop()
        m = e - s
        if m <= leaf:
            continue
        seg += 1
        xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf
        for k in range(s, e):
            v = perm[k]
            xmin, xmax = min(xmin, x[v]), max(xmax, x[v])
            ymin, ymax = min(ymin, y[v]), max(ymax, y[v])
        c = np.empty(m)
        if xmax - xmin >= ymax - ymin:
            for k in range(m):
                c[k] = x[perm[s + k]]
        else:
            for k in range(m):
                c[k] = y[perm[s + k]]
        o = np.argsort(c)
        for k in range(m):
            buf[k] = perm[s + o[k]]
        half = m // 2
        for k in range(m):
            v = buf[k]
            stamp[v] = seg
            side[v] = 0 if k < half else 1
        nl = 0
        ns = 0
        sep = np.empty(half, dtype=np.int64)
        for k in range(half):
            v = buf[k]
            cut = False
            for q in range(indptr[v], indptr[v + 1]):
                w = indices[q]
                if w != v and stamp[w] == seg and side[w] == 1:
                    cut = True
                    break
            if cut:
                sep[ns] = v
                ns += 1
            else:
                perm[s + nl] = v
                nl += 1
        for k in range(half, m):
            perm[s + nl + (k - half)] = buf[k]
        for k in range(ns):
            perm[e - ns + k] = sep[k]
        stack_s.append(s)
        stack_e.append(s + nl)
        stack_s.append(s + nl)
        stack_e.append(s + nl + (m - half))
    return perm


@njit(cache=True)
def _restrict(indptr, indices, data, keep):  # type: ignore[no-untyped-def]
    """The rows and columns `keep` of a CSR matrix, its zero entries dropped: (indptr, indices,
    data) of the submatrix, its rows and columns numbered in their order."""
    n = len(keep)
    new = -np.ones(n, dtype=np.int64)
    N = 0
    for i in range(n):
        if keep[i]:
            new[i] = N
            N += 1
    cnt = 0
    for i in range(n):
        if keep[i]:
            for p in range(indptr[i], indptr[i + 1]):
                if keep[indices[p]] and data[p] != 0.0:
                    cnt += 1
    ip = np.zeros(N + 1, dtype=np.int32)
    ix = np.empty(cnt, dtype=np.int32)
    dx = np.empty(cnt)
    k = 0
    for i in range(n):
        if keep[i]:
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                if keep[j] and data[p] != 0.0:
                    ix[k] = new[j]
                    dx[k] = data[p]
                    k += 1
            ip[new[i] + 1] = k
    return ip, ix, dx


class SuperLU:
    """SuperLU on a core (module docstring)."""

    @property
    def device(self) -> str:
        return "cpu"

    def plan(self, pattern: Pattern) -> _SuperLUPlan:
        return _SuperLUPlan(pattern)


class _SuperLUPlan:
    """The pattern in one nested-dissection order: per permuted entry the position of its value
    in the pattern's data (`gather`)."""

    def __init__(self, pattern: Pattern):
        n = pattern.n
        self.n = n
        self.perm = np.asarray(dissection(pattern.indptr, pattern.indices, pattern.xy[:, 0],
                                          pattern.xy[:, 1], ND_LEAF), dtype=np.int64)
        pos = sp.csr_matrix((np.arange(1, pattern.nnz + 1, dtype=np.int64), pattern.indices,
                             pattern.indptr), shape=(n, n))
        P = pos[self.perm][:, self.perm].tocsr()
        P.sort_indices()
        self.indptr = P.indptr.astype(np.int32)
        self.indices = P.indices.astype(np.int32)
        self.gather = P.data - 1

    def factor(self, data: NDArray[np.float64], unknown: NDArray[np.bool_]) -> Factor:
        from scipy.sparse.linalg import splu
        keep = unknown[self.perm]
        ip, ix, dx = _restrict(self.indptr, self.indices, data[self.gather], keep)
        N = len(ip) - 1
        # symmetric: the restricted CSR's arrays are its CSC's too
        lu = splu(sp.csc_matrix((dx, ix, ip), shape=(N, N)), permc_spec="NATURAL",
                  diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
        A = sp.csr_matrix((dx, ix, ip), shape=(N, N))
        return _SuperLUFactor(n=self.n, rows=self.perm[keep],
                              refined=Refined(A, lambda r: np.asarray(lu.solve(r))))


class _SuperLUFactor:
    def __init__(self, *, n: int, rows: NDArray[np.int64], refined: Refined):
        self.n, self.rows, self.refined = n, rows, refined

    def solve(self, b: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.zeros(self.n)
        x[self.rows] = self.refined.solve(np.ascontiguousarray(b[self.rows]))
        return x


# ------------------------------------------------------------------ cuDSS, on a GPU

CUDSS_LOG = logging.ERROR   # nvmath's logger: its warning at every reset of the operands (the
                            # values change on the plan's pattern, by design) silenced; logging
                            # only


class CuDSS:
    """cuDSS on a GPU through nvmath-python (module docstring)."""

    @property
    def device(self) -> str:
        return "gpu"

    def plan(self, pattern: Pattern) -> _CuDSSPlan:
        return _CuDSSPlan(pattern)


class _CuDSSPlan:
    """The pattern's cuDSS solver: planned at its first factorization (with that state's
    values), every later state a refactorization with the operands reset on it."""

    def __init__(self, pattern: Pattern):
        self.pattern = pattern
        self.solver: Any = None
        self.generation = 0

    def factor(self, data: NDArray[np.float64], unknown: NDArray[np.bool_]) -> Factor:
        p = self.pattern
        A = sp.csr_matrix((data, p.indices, p.indptr), shape=(p.n, p.n))
        if self.solver is None:
            from nvmath.sparse.advanced import (DirectSolver, DirectSolverMatrixType,
                                                DirectSolverOptions)
            log = logging.getLogger("gap_factor.cudss")
            log.setLevel(CUDSS_LOG)
            self.solver = DirectSolver(
                A, np.zeros(p.n), execution="cuda",
                options=DirectSolverOptions(sparse_system_type=DirectSolverMatrixType.SPD,
                                            blocking=True, logger=log))
            self.solver.plan()
        else:
            self.solver.reset_operands(a=A)
        self.solver.factorize()
        self.generation += 1
        return _CuDSSFactor(plan=self, generation=self.generation, A=A)

    def __del__(self) -> None:
        if self.solver is not None:
            self.solver.free()


class _CuDSSFactor:
    def __init__(self, *, plan: _CuDSSPlan, generation: int, A: sp.csr_matrix):
        self.plan, self.generation = plan, generation
        self.refined = Refined(A, self._once)

    def _once(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.generation != self.plan.generation:
            raise RuntimeError("a cuDSS factorization solved after a later one on its plan")
        s = self.plan.solver
        s.reset_operands(b=np.ascontiguousarray(r))
        return np.asarray(s.solve())

    def solve(self, b: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.refined.solve(b)


def factorization_of(device: str) -> Factorization:
    """The run's device (its configuration's `cpu|gpu`), resolved once."""
    if device == "cpu":
        return SuperLU()
    if device == "gpu":
        return CuDSS()
    raise ValueError(f"unknown device {device!r}: cpu or gpu")
