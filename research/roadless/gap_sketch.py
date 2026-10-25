"""Effective resistances of one state's network from its factor, by a random sketch (Spielman and
Srivastava 2008, "Graph sparsification by effective resistances").

A state's operator A (gap_clear.Graph.values: cells x cells, its unknowns' rows) is M^T M, M a row
sqrt(W) (e_i - e_j) per pair of unknown cells and a row sqrt(x_i) e_i per unknown cell, x_i its
excess (its diagonal less its pairs' to other unknowns: its street faces, its leak and its
passages to ground cells). The resistance of an electrode pair, a weight vector y on the cells
summing to 0 (two electrodes, each's weights summing to 1, one with a minus sign), or of one
electrode to the ground (its weights summing to 1), is y^T A^-1 y = |M A^-1 y|^2; with Q a k x rows
Gaussian matrix over sqrt(k), |Q M A^-1 y|^2 estimates it, its relative spread about sqrt(2 / k)
(a chi-square's with k degrees). Z = Q M A^-1 (k x cells) costs k solves with the state's factor,
Z^T's columns A^-1 M^T q, after which every pair's estimate is a dot product.

    PYTHONPATH=. uv run python research/roadless/gap_sketch.py check <block> none|<fault>

check: on the block's baseline state, the sketch at CHECK_K against the exact resistance (a solve
each) of CHECK_PAIRS random pairs of unknown cells and as many cells to the ground; the mean
ratio within CHECK_TOL of 1 and no pair's ratio outside 1 -+ 6 sqrt(2 / k). Faults: `no-excess`
(the cells' rows left out: the paths to the ground unseen), `no-sqrt` (W in place of sqrt(W)).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gap_clear as gcl  # noqa: E402

CHECK_K = 400
CHECK_PAIRS = 40
CHECK_TOL = 0.05
FAULTS = ("none", "no-excess", "no-sqrt")


class Sketch(NamedTuple):
    """Z = Q M A^-1 (module docstring), k x cells."""
    Z: NDArray[np.float64]

    @staticmethod
    def of(st: gcl.State, *, k: int, seed: int) -> Sketch:
        """The sketch of `st` (its operator assembled and factored) at k rows, Q drawn from
        `seed`."""
        if st.factor is None:
            raise ValueError("a sketch needs the state's factor")
        rows = _rows(st)
        n, unknown = len(rows.x), st.reach.unknown
        rng = np.random.default_rng(seed)
        Z = np.empty((k, n))
        for r in range(k):
            qp = rng.standard_normal(len(rows.w)) * rows.w
            y = (np.bincount(rows.i, weights=qp, minlength=n)
                 - np.bincount(rows.j, weights=qp, minlength=n) + rng.standard_normal(n) * rows.x)
            Z[r] = st.factor.solve(np.where(unknown, y, 0.0)) / np.sqrt(k)
        return Sketch(Z=Z)

    def resistance(self, y: NDArray[np.float64]) -> float:
        """y^T A^-1 y for a weight vector y on the cells (module docstring)."""
        z = self.Z @ y
        return float(z @ z)


class Rows(NamedTuple):
    """M's rows (module docstring): per pair of unknown cells its two cells and sqrt(W), per cell
    sqrt(its excess) (0 off the unknowns)."""
    i: NDArray[np.int64]
    j: NDArray[np.int64]
    w: NDArray[np.float64]
    x: NDArray[np.float64]


def _rows(st: gcl.State) -> Rows:
    if st.A is None:
        raise ValueError("a sketch needs the state's operator")
    A = st.A.tocoo()
    unknown = st.reach.unknown
    i, j, a = A.row.astype(np.int64), A.col.astype(np.int64), A.data
    both = unknown[i] & unknown[j] & (i != j)
    pair = both & (i < j) & (a != 0)
    n = A.shape[0]
    excess = np.where(unknown, A.diagonal() - np.bincount(i[both], weights=-a[both], minlength=n),
                      0.0)
    return Rows(i=i[pair], j=j[pair], w=np.sqrt(-a[pair]), x=np.sqrt(np.maximum(excess, 0.0)))


def check(bid: str, fault: str | None) -> None:
    import common
    import metrics
    metric = metrics.metric_of(metrics.RUNTIME, device="cpu")
    [b] = common.build_blocks([bid])
    m = metric.clearing(b, demand=common.PopulationDemand())
    st = m.state(np.zeros(m.n))
    if fault is not None:
        global _rows
        clean = _rows

        def _rows(st: gcl.State) -> Rows:      # the fault: one part of M dropped or mis-scaled
            r = clean(st)
            return (r._replace(x=np.zeros_like(r.x)) if fault == "no-excess"
                    else r._replace(w=r.w ** 2))
    sk = Sketch.of(st, k=CHECK_K, seed=0)
    cells = np.flatnonzero(st.reach.unknown)
    rng = np.random.default_rng(1)
    ratios = []
    for t in range(2 * CHECK_PAIRS):
        y = np.zeros(m.c.n)
        p, q = rng.choice(cells, 2, replace=False)
        y[p] = 1.0
        if t < CHECK_PAIRS:                  # a pair; then a cell to the ground
            y[q] = -1.0
        exact = float(y @ st.factor.solve(y)) if st.factor is not None else np.nan
        ratios.append(sk.resistance(y) / exact)
    r = np.asarray(ratios)
    spread = 6.0 * np.sqrt(2.0 / CHECK_K)
    ok = abs(r.mean() - 1.0) <= CHECK_TOL and bool((np.abs(r - 1.0) <= spread).all())
    print(f"{bid}: {len(cells)} unknowns, k {CHECK_K}: sketch / exact mean {r.mean():.3f}, "
          f"range {r.min():.3f} .. {r.max():.3f} (pairs {r[:CHECK_PAIRS].mean():.3f}, to ground "
          f"{r[CHECK_PAIRS:].mean():.3f}) -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "check" and sys.argv[3] in FAULTS:
        check(sys.argv[2], None if sys.argv[3] == "none" else sys.argv[3])
    else:
        raise SystemExit(__doc__)
