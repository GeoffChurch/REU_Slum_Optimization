"""The optimizers' shared parts (phase 3, NOTES "The complex as the runtime: the raster retired"):
the tension's eps world (EPS, the gap complex's Model.tension ranks in it), the exact scores'
tolerance (RTOL_SCORE), the catchment's view of a round (Catchment: how alike the candidates'
effects are, for search.Diverse's batches), the batch rule (`_spread`), the next population step
(`_next_target`) and the rows directory of the greedy's format (`rows_dir`).

This module was the raster's greedy (clear.Clearing: the open-fraction tension, free first on its
region graph, the greedy's runs over the 220); that went with the raster. The tension and free
first are the complex's (gap_clear.Model.tension, gap_complex.free_first), the greedy's rounds
search.py's.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seam import Flow  # noqa: E402

if TYPE_CHECKING:
    from seam import ClearingModel
    from seam import Tension as TensionLike

EPS = 0.01
RTOL_SCORE = 1e-5       # P is quadratic in u: this gives it to ~1e-10


def _next_target(c: ClearingModel, delta: float) -> tuple[float, float]:
    """(D now, the next multiple of delta above it)."""
    D = float(c.cost[c.removed].sum())
    return D, (np.floor(D / delta + 1e-9) + 1) * delta


class GramSource(Protocol):
    """How alike the candidates' effects are: a PSD matrix over `cand` whose normalised entries
    are the correlations rho_ij that `search.Diverse` builds its batch from (a seam.Tension's:
    the raster's or the complex's)."""
    @property
    def name(self) -> str: ...

    def gram(self, t: TensionLike, cand: list[int], power: float) -> np.ndarray: ...


class Sweep(Protocol):
    """Where Catchment's downhill sweep runs. prepare(A, u, rowsum) -> run(rows, cols, C): the
    (n x C) device array out[x, c] = share of node x's outflow that passes through column c's
    member nodes (the (rows, cols) pairs) on the way to ground. Flow only runs downhill, so out
    at x needs out at x's lower neighbours first. columns(n): columns per run on n nodes, so
    that the (n x C) result fits in memory."""

    def columns(self, n: int) -> int: ...

    def prepare(self, A, u, rowsum) -> Callable: ...


@njit(cache=True)
def _catch(indptr, indices, data, u, rowsum, order, inA, out):
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


@dataclass(frozen=True)
class CpuSweep:
    """numba, one core: every node in order of increasing potential."""
    chunk: int = 16

    def columns(self, n: int) -> int:
        return self.chunk

    def prepare(self, A, u, rowsum):
        order = np.argsort(u, kind="stable")

        def run(rows, cols, C):
            inA = np.zeros((len(u), C), dtype=np.bool_)
            inA[rows, cols] = True
            out = np.zeros((len(u), C))
            _catch(A.indptr, A.indices, A.data, u, rowsum, order, inA, out)
            return out
        return run


@dataclass(frozen=True)
class Catchment:
    """No solves: trace the round's flow downhill and give each candidate the share of each
    home's flow that passes through its cells. H_ij = sum_h c_h v_i(h) v_j(h), with c_h the
    home's weight in J_power (inj_h u_h^power, inj the demand's injection). The sweep runs where
    `sweep` says."""
    sweep: Sweep

    @property
    def name(self) -> str:
        return "cat"

    def gram(self, t: TensionLike, cand: list[int], power: float) -> np.ndarray:
        f = t.flow(cand)
        V = self.shares(f, len(cand))
        wt = t.weight(power)
        return f.to_host((V * wt[:, None]).T @ V)

    def shares(self, f: Flow, ncand: int) -> Any:
        """v_j(h) on the device, (homes x candidates): the share of home h's injected flow in
        the tension's state (`f`, its Flow with the candidates' members) that passes through
        candidate j's nodes on its way to ground (0 for a home none of whose injection is on a
        node)."""
        xp = f.xp
        inj = f.load @ xp.ones(f.load.shape[1])
        run = self.sweep.prepare(f.A, f.u, f.rowsum)
        V = xp.zeros((f.load.shape[0], ncand))
        step = self.sweep.columns(len(f.u))
        for s0 in range(0, ncand, step):
            C = min(step, ncand - s0)
            sel = (f.cols >= s0) & (f.cols < s0 + C)
            V[:, s0:s0 + C] = f.load @ run(f.rows[sel], f.cols[sel] - s0, C)
        return xp.where(inj[:, None] > 0, V / xp.where(inj > 0, inj, 1.0)[:, None], 0.0)


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


