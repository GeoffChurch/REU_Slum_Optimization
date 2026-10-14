"""The optimizers' seam: what search.py, relax.py, polish_greedy.py and free first consume from a
clearing model, written down as Protocols (NOTES, "The complex as the runtime: one fixed mesh").

Two implementations, both live until phase 2 measures which dominates: the raster's
(clear.Clearing, its eps world relax.Relaxation) and the gap complex's (gap_clear.Model, its eps
world gap_clear.Relaxation). Which one a run uses is its metric (`Metric`), resolved once where the
run's configuration is read (metrics.metric_of) and handed down; below that edge no code asks which
it has.

What the consumers read, and where:
- the state and its price: `removed` (the search's clearing; search.apply and the callers that
  seed a search change it), `n`, `cost` (each building's population share: D);
- the footprints: `polys`, `tree` (search.Spaced's spacing);
- scores: `baseline` (J_power and P with nothing cleared), `exact` (clear.exact: J_power and P of
  a 0/1 clearing, real geometry), `enclosed_left` (B (i): the population share a clearing leaves
  walled in);
- first order: `tension(power, restore=...)` (search.AddTension and RestoreTension: each
  building's gain from clearing it, or from putting it back), and through it the catchment's view
  of the tension's flow (`Tension.flow`, `Tension.weight`: clear.Catchment);
- free first (B's first stage: clear.free_first, gap_complex.free_first);
- the eps worlds (`relaxation`: SIMP's and the restore screen's; `Relaxed.metric_world`: the
  incumbent's and the polish's scorer, under the metric itself).

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/seam.py
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import shapely

    import clear
    import common
    import search


class Flow(NamedTuple):
    """The catchment's view of a tension's state (clear.Catchment.shares): its operator over its
    nodes, their potential, each node's conductance to ground (A's row sums), each home's
    injection on each node, and a set of candidates' member nodes, all on the device `xp`."""
    xp: Any                     # numpy, or cupy
    A: Any                      # (nodes x nodes) CSR
    u: Any                      # (nodes,)
    rowsum: Any                 # (nodes,)
    load: Any                   # (homes x nodes) CSR
    rows: Any                   # the candidates' member nodes ...
    cols: Any                   # ... and their candidate (a column of `cand`)
    to_host: Callable[[Any], NDArray[np.float64]]


class Tension(Protocol):
    """A first-order ranking from one eps-world solve (clear.Clearing.tension's)."""

    @property
    def g(self) -> NDArray[np.float64]:
        """Per building its gain (lower J) from clearing it, or (restore) from putting it back
        (negative: J rises); -inf where the move does not apply."""
        ...

    def weight(self, power: float) -> Any:
        """Per home (on the device) its weight in J_power, inj_h u_h^power in the tension's state
        (the catchment's c_h)."""
        ...

    def flow(self, cand: list[int]) -> Flow:
        """The catchment's view of this state, with `cand`'s member nodes."""
        ...


class Relaxed(Protocol):
    """A clearing model's eps world as relax.simp, relax.Incumbent and search's worlds read it
    (relax.Relaxation's): x per building in [0, 1], a closed building conducting eps; q and beta
    set by simp's continuation."""
    q: float
    beta: float

    @property
    def c(self) -> ClearingModel: ...

    @property
    def power(self) -> float: ...

    @property
    def J0(self) -> float: ...

    def proj(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def dproj(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def value(self, x: NDArray[np.float64]) -> float: ...

    def value_sgrad(self, x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]: ...

    def value_grad(self, x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]: ...

    def exact(self, x: NDArray[np.float64]) -> float: ...

    def exact_JP(self, x: NDArray[np.float64]) -> tuple[float, float]: ...

    def perm(self, J: float) -> float: ...

    def score(self, s: search.SearchState, removed: NDArray[np.bool_]) -> search.Score: ...

    def metric_world(self, eps: float) -> Relaxed:
        """The same model's eps world at `eps` under the metric itself (q 1, no projection): the
        incumbent's and the polish's scorer."""
        ...


class ClearingModel(Protocol):
    """One block's clearing model (module docstring)."""
    removed: NDArray[np.bool_]

    @property
    def n(self) -> int: ...

    @property
    def cost(self) -> NDArray[np.float64]: ...

    @property
    def polys(self) -> NDArray[np.object_]: ...

    @property
    def tree(self) -> shapely.STRtree: ...

    @property
    def cells(self) -> int:
        """Its unknowns at baseline (for the logs)."""
        ...

    def release(self) -> None:
        """Return what its solves hold on a device (the raster's GPU pool; the complex: nothing)."""
        ...

    def baseline(self, power: float) -> tuple[float, float]:
        """(J_power, P) with nothing cleared."""
        ...

    def exact(self, removed: NDArray[np.bool_], power: float) -> tuple[float, float]:
        """(J_power, P) of the 0/1 clearing `removed`."""
        ...

    def enclosed_left(self, removed: NDArray[np.bool_]) -> float: ...

    def tension(self, power: float, *, restore: bool) -> Tension: ...

    def free_first(self, *, budget: float, held: NDArray[np.bool_]) -> NDArray[np.bool_]: ...

    def relaxation(self, power: float, *, q: float, rtol: float, eps: float) -> Relaxed:
        """The eps world the optimizers search in (the raster's: under its search
        conductance)."""
        ...


class Metric(Protocol):
    """A run's metric: its name (the rows' directories: relax_rows/<name>/..., polish_rows/<name>/
    ...), where the catchment's sweep runs (its tensions' device), a block's clearing model under
    it, and the same metric on a coarser uniform grid (a SIMP plan's .c / .C: the raster's; the
    complex has no grid and refuses)."""

    @property
    def name(self) -> str: ...

    @property
    def sweep(self) -> clear.Sweep: ...

    def clearing(self, block: Any, *, demand: common.Demand) -> ClearingModel: ...

    def coarse(self, h: float) -> Metric: ...
