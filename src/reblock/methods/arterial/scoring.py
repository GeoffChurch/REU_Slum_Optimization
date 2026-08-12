"""Per-candidate scoring for the arterial engines: `_score` mirrors the budget metrics for one
road set, `_StepState` is the frozen per-greedy-step snapshot the fork process pool inherits via
copy-on-write, and `eval_candidate` is the pure per-candidate evaluation both the exact and lazy
engines call (serially, or as the parallel-map unit of work). `_best_candidate` is the shared
argmax reduce over `eval_candidate`'s results.

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

import numpy as np
import shapely
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from pyproj import CRS
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from reblock.budget import (
    _BlockScoringContext,
    _StepContext,
    access_burden,
    displacement,
    displacement_from_distance,
    repulsion,
)
from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.methods.arterial.primitives import _explode, _planarize, _SnapGraph, _union_with
from reblock.methods.arterial.realize import ChordRealizer


def _score(objective: str, block: Block, roads: GeoDataFrame, adj: list[set[int]],
           base_burden: float, ctx: _BlockScoringContext | None) -> float:
    """Objective value of the road set (higher = better). Mirrors the budget metrics with a
    cached parcel-adjacency for the greedy's inner loop. For efficiency/directness, scores through
    the per-block `ctx` (frozen constants, one context per block, full entry re-derivation per
    candidate) -- equivalent to `network_efficiency(block, roads)`."""
    if objective == "access":
        if base_burden == 0.0:
            return 0.0
        depths = parcel_access_layers(block, roads, tol=STREET_TOL, adj=adj,
                                      unreached_depth=len(block.parcels) + 1)
        return 1.0 - access_burden(depths) / base_burden
    assert ctx is not None
    e, direct = ctx.score(roads)
    return e if objective == "efficiency" else direct


@dataclass(frozen=True)
class _StepState:
    """Frozen per-greedy-step state, module-level so the fork process pool (task 2 of the
    process-parallel-arterial design) can inherit it via copy-on-write instead of pickling the
    CSR/graph context per task -- only the small chord goes in and `(gain, geometry)` comes out.
    Set once per step by `_greedy_arterials`, read by `eval_candidate`, cleared in a `finally`.
    `committed` is needed by the aspirational branch's full `_planarize(committed + [real])`;
    `base_merged`/`adj`/`base_burden`/`ctx` by the access-objective-buildable and displacement
    branches' `_score`/incremental-`_union_with` calls -- omitting any of these breaks a scoring
    branch silently (wrong values, not a crash). `radii` (per-building disk radius, r=NN/2,
    constant across a block's steps) feeds the `cost="displacement"` denominator's disk
    `displacement` call. `frozen=True` makes the
    read-only invariant real:
    the workers only READ this holder, and its mutable members (`committed`, `base_merged`) are
    mutated only in the PARENT and only AFTER the holder is cleared (a fresh `_StepState` is built
    per step), so no worker ever observes a mid-mutation copy."""
    step: _StepContext | None
    sg: _SnapGraph
    base_val: float
    base_merged: BaseGeometry | None
    committed: list[LineString]
    realizer: ChordRealizer
    objective: str
    cost: str
    half_width_m: float
    committed_disp: float
    # For cost="displacement_fast": per-building distance to the COMMITTED corridor, fixed for the
    # step, plus the building geometries to measure a candidate against. None for other costs.
    committed_dist: NDArray[np.float64] | None
    building_xy: NDArray[np.object_] | None
    block: Block
    radii: NDArray[np.float64]
    crs: CRS
    adj: list[set[int]]
    base_burden: float
    ctx: _BlockScoringContext | None


_STEP_STATE: _StepState | None = None

# Below this many candidates in a step, the fork/pool overhead (spawn + IPC round-trips) is not
# worth it, so the step runs the serial path over `eval_candidate` instead of the process pool.
# Module-level so tests can monkeypatch it low to force the pool path on small, fast blocks.
_PARALLEL_THRESHOLD = 128


def eval_candidate(chord: LineString) -> tuple[float, BaseGeometry | None]:
    """Pure per-candidate evaluation, module-level so it doubles as the parallel-map unit of work
    in a later task. Mirrors `_greedy_arterials`' former inline loop body EXACTLY (realizer/
    objective/cost routing, the `cost="displacement"` denominator, the infinite-gain
    zero-denominator escape) reading the frozen per-step state stashed in `_STEP_STATE` (see
    `_StepState`). Returns `(0.0, None)` for a None/zero-length realization. Returns the shapely
    GEOMETRY (not wkt) -- `_best_candidate` compares `.wkt` only for its tie-break, and returning
    the geometry keeps a future process-pool's pickled round-trip (WKB, lossless) bit-identical to
    this serial path's `real`, unlike a lossy default-precision `to_wkt()`."""
    st = _STEP_STATE
    assert st is not None, "eval_candidate called with no _STEP_STATE set"
    real = st.realizer.realize(chord, st.sg)
    if real is None or real.length == 0:
        return 0.0, None
    trial: GeoDataFrame | None = None
    if st.step is not None:
        e, direct = st.step.score_candidate(real)
        raw = (e if st.objective == "efficiency" else direct) - st.base_val
    elif st.realizer.snaps:
        trial = _explode(_union_with(st.base_merged, real), st.crs, 2.0 * st.half_width_m)
        raw = _score(st.objective, st.block, trial, st.adj, st.base_burden, st.ctx) - st.base_val
    else:
        trial = _planarize(st.committed + [real], st.crs, 2.0 * st.half_width_m)
        raw = _score(st.objective, st.block, trial, st.adj, st.base_burden, st.ctx) - st.base_val
    if st.cost == "displacement_fast":
        # dist(p, committed u cand) == min(dist(p, committed), dist(p, cand)), so only the
        # candidate's own corridor distance is new work -- no union over the committed set. Agrees
        # with `displacement` to ~1e-10, NOT bit-exactly (GEOS measures distance to a unioned
        # polygon over a different vertex set than to the parts), and this greedy's argmax turns
        # that into a different trajectory on ~29% of runs. See
        # notes/2026-08-09-greedy-arterial-is-tie-sensitive.md -- the divergence is large when it
        # lands (up to 11 points of burden reduction) but shows no systematic direction.
        assert st.building_xy is not None
        cand_d = shapely.distance(st.building_xy, real.buffer(st.half_width_m))
        d = cand_d if st.committed_dist is None else np.minimum(st.committed_dist, cand_d)
        denom = float(displacement_from_distance(st.radii, d) - st.committed_disp)
    elif st.cost == "displacement":
        if trial is None:
            # step -> buildable
            trial = _explode(_union_with(st.base_merged, real), st.crs,
                             2.0 * st.half_width_m)
        denom = float(displacement(st.block.building_points, st.radii, trial)
                     - st.committed_disp)
    elif st.cost == "repulsion":
        denom = repulsion(st.block.building_points, st.radii, real)
    else:
        denom = real.length
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
