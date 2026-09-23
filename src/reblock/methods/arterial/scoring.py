"""Per-candidate scoring for the arterial engines: `_StepState` is the frozen per-greedy-step
snapshot the fork process pool inherits via copy-on-write, and `eval_candidate` is the pure
per-candidate evaluation every engine calls (serially, or as the parallel-map unit of work).
`_best_candidate` is the shared argmax reduce over `eval_candidate`'s results. What a candidate
gains and what it costs are the injected objective's and cost's business -- see `objectives.py`
and `costs.py`; this module only divides one by the other.

`_STEP_STATE` is the module-level holder itself. Callers outside this module (the engines) MUST
write it as `scoring._STEP_STATE = ...` -- a qualified module-attribute assignment via
`from reblock.methods.arterial import scoring` -- never via a `from ... import _STEP_STATE`
binding, which would rebind an independent local copy that `eval_candidate` (which reads
`_STEP_STATE` as its own module global, right here) would never see updated, and that a forked
worker -- which inherits THIS module's globals via copy-on-write -- would see as permanently
`None`.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from reblock.contracts import Block
from reblock.methods.arterial.costs import ArterialCost, StepCost
from reblock.methods.arterial.objectives import BlockObjective, CommittedNetwork, StepGain
from reblock.methods.arterial.primitives import _SnapGraph
from reblock.methods.arterial.realize import ChordRealizer


@dataclass(frozen=True)
class _StepState:
    """Frozen per-greedy-step state, module-level so the fork process pool can inherit it via
    copy-on-write instead of pickling the CSR/graph context per task -- only the small chord goes
    in and `(gain, geometry)` comes out. Set once per step by the engine, read by
    `eval_candidate`, cleared in a `finally`. `gain` and `cost` close over everything their scoring
    reads (the committed roads, the step's scoring context, the corridor's pieces), built in the
    PARENT before the pool forks. `frozen=True` makes the read-only invariant real: the workers
    only READ this holder, and the committed roads the parent goes on to extend are snapshotted into
    it (`CommittedNetwork.lines` is a tuple), so no worker can observe a later step's roads."""
    sg: _SnapGraph
    realizer: ChordRealizer
    gain: StepGain
    cost: StepCost


def step_state(block: Block, *, sg: _SnapGraph, realizer: ChordRealizer,
               objective: BlockObjective, cost: ArterialCost,
               committed: CommittedNetwork) -> _StepState:
    """One step's `_StepState`: the objective's and the cost's per-step scorers over `committed`.
    Shared by every engine, so a step can never be set up two ways."""
    return _StepState(sg=sg, realizer=realizer, gain=objective.at_step(committed, realizer),
                      cost=cost.at_step(block, committed))


_STEP_STATE: _StepState | None = None

# Below this many candidates in a step, the fork/pool overhead (spawn + IPC round-trips) is not
# worth it, so the step runs the serial path over `eval_candidate` instead of the process pool.
# Module-level so tests can monkeypatch it low to force the pool path on small, fast blocks.
_PARALLEL_THRESHOLD = 128


def eval_candidate(chord: LineString) -> tuple[float, BaseGeometry | None]:
    """Pure per-candidate evaluation, module-level so it doubles as the parallel-map unit of work:
    realize the chord, then its gain over its cost, with the infinite-gain zero-denominator escape
    (a beneficial road that costs nothing ranks above every priced one -- take the free
    navigability first). Reads the frozen per-step state stashed in `_STEP_STATE` (see
    `_StepState`). Returns `(0.0, None)` for a None/zero-length realization. Returns the shapely
    GEOMETRY (not wkt) -- `_best_candidate` compares `.wkt` only for its tie-break, and returning
    the geometry keeps the process pool's pickled round-trip (WKB, lossless) bit-identical to the
    serial path's `real`, unlike a lossy default-precision `to_wkt()`."""
    st = _STEP_STATE
    assert st is not None, "eval_candidate called with no _STEP_STATE set"
    real = st.realizer.realize(chord, st.sg)
    if real is None or real.length == 0:
        return 0.0, None
    raw = st.gain.of(real)
    denom = st.cost.of(real)
    gain = float("inf") if (denom <= 0 and raw > 0) else (raw / denom if denom > 0 else 0.0)
    return gain, real



def _best_candidate(results: Iterable[tuple[float, BaseGeometry | None]]
                    ) -> tuple[float, BaseGeometry | None]:
    """The greedy's candidate selection as ONE shared reduce -- used by both the serial path below
    and a future parallel-collect path (task 2). NOT a plain argmax: `(0.0, None)` init, and the
    wkt tie-break is additionally gated on `best_real is not None`, so a candidate with `gain <=
    0` can NEVER win -- this IS the "no candidate improves -> stop" termination. A naive `best =
    None` argmax would instead let a zero/negative-gain candidate win on the terminating step and
    change the geometry. Order-independent (so parallel-collect order doesn't matter): the
    tie-break is a total order over distinct `wkt`, and `gain > best_gain` alone is order-free."""
    best_gain, best_real = 0.0, None
    for gain, real in results:
        if gain > best_gain or (best_real is not None and real is not None
                                and gain == best_gain and real.wkt < best_real.wkt):
            best_gain, best_real = gain, real
    return best_gain, best_real
