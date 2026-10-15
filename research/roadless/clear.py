"""The optimizers' shared parts (phase 3, NOTES "The complex as the runtime: the raster retired"):
the tension's eps world (EPS, the gap complex's Model.tension ranks in it), the catchment's view
of a round (Catchment, over a tension's Flow: how alike the candidates' effects are, for
search.Diverse's batches), the batch rule (`_spread`), the next population step (`_next_target`)
and the rows directory of the greedy's format (`rows_dir`).

This module was the raster's greedy (clear.Clearing: the open-fraction tension, free first on its
region graph, the greedy's runs over the 220); that went with the raster. The tension and free
first are the complex's (gap_clear.Model.tension, gap_complex.free_first), the greedy's rounds
search.py's.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import scipy.sparse as sp
from numba import njit
from numpy.typing import NDArray

if TYPE_CHECKING:
    import gap_clear

HERE = Path(__file__).resolve().parent
EPS = 0.01
CATCH_COLUMNS = 16      # the sweep's candidates a pass: memory only (the same shares in any chunks)


class Flow(NamedTuple):
    """The catchment's view of a tension's state (gap_clear.Tension.flow): its operator over its
    nodes (the cells), their potential, each node's conductance to ground (A's row sums), each
    home's injection on each node, and a set of candidates' member nodes (`rows`, each with its
    candidate, a column of the candidates: `cols`)."""
    A: sp.csr_matrix
    u: NDArray[np.float64]
    rowsum: NDArray[np.float64]
    load: sp.csr_matrix
    rows: NDArray[np.int64]
    cols: NDArray[np.int64]


def _next_target(c: gap_clear.Model, delta: float) -> tuple[float, float]:
    """(D now, the next multiple of delta above it)."""
    D = float(c.cost[c.removed].sum())
    return D, (np.floor(D / delta + 1e-9) + 1) * delta


@njit(cache=True)
def _catch(indptr: NDArray[np.int32], indices: NDArray[np.int32], data: NDArray[np.float64],
           u: NDArray[np.float64], rowsum: NDArray[np.float64], order: NDArray[np.int64],
           inA: NDArray[np.bool_], out: NDArray[np.float64]) -> None:
    """out[x, c] = share of node x's outflow that passes through candidate c's nodes (inA) on the
    way to ground, sweeping nodes by increasing potential (flow only runs downhill)."""
    C = inA.shape[1]
    acc = np.zeros(C)
    for x in order:
        acc[:] = 0.0
        Q = rowsum[x] * u[x]                                  # straight to ground
        for p in range(indptr[x], indptr[x + 1]):
            y = indices[p]
            if y != x and u[y] < u[x]:
                q = -data[p] * (u[x] - u[y])
                Q += q
                for c in range(C):
                    acc[c] += q * out[y, c]
        for c in range(C):
            if inA[x, c]:
                out[x, c] = 1.0
            elif Q > 0:
                out[x, c] = acc[c] / Q


class Catchment:
    """How alike the candidates' effects are (search.Diverse's batch): no solves, the round's flow
    traced downhill, each candidate given the share of each home's flow that passes through its
    nodes. H_ij = sum_h c_h v_i(h) v_j(h), with c_h the home's weight in J_power (inj_h
    u_h^power, inj the demand's injection): a PSD matrix whose normalised entries are the
    correlations rho_ij. The sweep (`_catch`) runs on the host, one core, every node in order of
    increasing potential, CATCH_COLUMNS candidates a pass."""

    @property
    def name(self) -> str:
        return "cat"

    def gram(self, t: gap_clear.Tension, cand: list[int], power: float) -> NDArray[np.float64]:
        f = t.flow(cand)
        V = self.shares(f, len(cand))
        wt = t.weight(power)
        return np.asarray((V * wt[:, None]).T @ V)

    def shares(self, f: Flow, ncand: int) -> NDArray[np.float64]:
        """v_j(h), (homes x candidates): the share of home h's injected flow in the tension's
        state (`f`, its Flow with the candidates' members) that passes through candidate j's nodes
        on its way to ground (0 for a home none of whose injection is on a node)."""
        inj = f.load @ np.ones(f.load.shape[1])
        order = np.argsort(f.u, kind="stable")
        V = np.zeros((f.load.shape[0], ncand))
        for s0 in range(0, ncand, CATCH_COLUMNS):
            C = min(CATCH_COLUMNS, ncand - s0)
            sel = (f.cols >= s0) & (f.cols < s0 + C)
            inA = np.zeros((len(f.u), C), dtype=np.bool_)
            inA[f.rows[sel], f.cols[sel] - s0] = True
            out = np.zeros((len(f.u), C))
            _catch(f.A.indptr, f.A.indices, f.A.data, f.u, f.rowsum, order, inA, out)
            V[:, s0:s0 + C] = f.load @ out
        return np.where(inj[:, None] > 0, V / np.where(inj > 0, inj, 1.0)[:, None], 0.0)


def _spread(gain: np.ndarray, cost: np.ndarray, H: np.ndarray, need: float,
            first: int | None = None) -> list[int]:
    """Greedy batch under a normalised quadratic model: with rho = H's correlations, candidate i
    is worth g_i - sum_{j in batch} rho_ij sqrt(g_i g_j) once the batch is chosen (a duplicate of
    a chosen one is worth 0, an independent one keeps its g_i, a complementary one, rho < 0,
    gains). Take the best worth per unit cost until the batch's cost reaches `need`; `first`, if
    given, is taken first whatever its worth (Diverse's screening width)."""
    d = np.sqrt(np.clip(np.diag(H), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(np.outer(d, d) > 0, H / np.outer(d, d), 0.0)
    sg = np.sqrt(np.clip(gain, 0.0, None))
    worth = gain.astype(float).copy()
    avail = np.ones(len(gain), dtype=bool)
    chosen: list[int] = []
    taken = 0.0
    while taken < need - 1e-12 and avail.any():
        i = (first if first is not None and not chosen
             else int(np.argmax(np.where(avail, worth / cost, -np.inf))))
        chosen.append(i)
        avail[i] = False
        taken += float(cost[i])
        worth -= rho[:, i] * sg * sg[i]
    return chosen


def rows_dir(picker: str, *, power: float, metric: str) -> Path:
    """The rows of a greedy-format run (search.py's schedules, relax.py `path`): `picker` its spec,
    `power` J's, `metric` the complex metric's name (its population the area's, its mesh none). The
    retired raster's greedy-format rows stand beside them as records, clear_rows_<picker>_h<h>
    [_area][_p<power>][_<conductance>]: no complex run's directory reads as one of theirs (no
    `_h`)."""
    return HERE / f"clear_rows_{picker}_p{power:g}_{metric}"


