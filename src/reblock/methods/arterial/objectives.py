"""What the arterial greedy maximizes: an injected `ArterialObjective` -- `Access`, `Efficiency` or
`Directness` -- rather than an `objective` string every engine switches on.

The configured objective is a frozen, fieldless dataclass (its own cache identity, like the
engines'). Everything it scores with is per-block state, built ONCE by `for_block` and closed over
by the `BlockObjective` it returns: the frozen `_BlockScoringContext` for the two network-metric
objectives, the block's `ParcelAdjacency` and no-roads access burden for `Access`. Each greedy
step then asks that for a `StepGain` over the roads committed so far, which is what
`scoring.eval_candidate` calls per candidate -- so no engine, and no candidate, ever asks which
objective it has.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
import shapely
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from pyproj import CRS
from shapely import STRtree
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from reblock.budget import _BlockScoringContext, _StepContext, access_burden
from reblock.derive.access import (
    ParcelAdjacency,
    _line_parts,
    parcel_access_layers,
    past_every_parcel,
    street_connectivity,
)
from reblock.methods.arterial.primitives import _planarize
from reblock.methods.arterial.realize import ChordRealizer


@dataclass(frozen=True)
class CommittedNetwork:
    """The roads committed before a greedy step, in each form a step's scoring reads: the realized
    lines in commit order, their `unary_union` (`merged`, None before the first commit), and that
    union exploded into the planarized road frame (`roads`). Built once per step by the engine."""
    lines: tuple[LineString, ...]
    merged: BaseGeometry | None
    roads: GeoDataFrame
    crs: CRS
    half_width_m: float


class StepGain(Protocol):
    """How much committing `real` raises the objective, over one step's committed roads."""

    def of(self, real: LineString) -> float: ...


class BlockObjective(Protocol):
    """An objective bound to one block's frozen scoring constants."""

    def value(self, roads: GeoDataFrame) -> float:
        """The objective (higher = better) of the block's streets plus `roads`."""

    def at_step(self, committed: CommittedNetwork, realizer: ChordRealizer) -> StepGain:
        """The per-candidate gain for the step whose committed roads are `committed`."""


@runtime_checkable
class ArterialObjective(Protocol):
    """What the greedy maximizes. `for_block` builds the per-block scorer once per proposal, from
    the block's adjacency -- which carries the block itself."""

    @property
    def identity(self) -> ObjectiveIdentity: ...

    def for_block(self, adjacency: ParcelAdjacency) -> BlockObjective: ...


@dataclass(frozen=True)
class _Replanarized:
    """Score `committed + [real]` re-unioned from scratch. The only exact path for a free
    (non-snapping) chord, which can cross a committed edge at a float interior point where the
    incremental union does not node identically ("Bug 2" -- see `budget._StepContext`)."""
    objective: BlockObjective
    committed: CommittedNetwork
    base: float

    def of(self, real: LineString) -> float:
        c = self.committed
        trial = _planarize([*c.lines, real], c.crs, 2.0 * c.half_width_m)
        return self.objective.value(trial) - self.base


@dataclass(frozen=True)
class _Incremental:
    """Score a snapped `real` through the step's frozen `_StepContext`, reprojecting parcels onto
    only its own edges -- bit-exact to a full re-derivation for snapped roads, and the reason the
    network-metric objectives are affordable per candidate."""
    step: _StepContext
    pick: Callable[[float, float], float]
    base: float

    def of(self, real: LineString) -> float:
        e, direct = self.step.score_candidate(real)
        return self.pick(e, direct) - self.base


@dataclass(frozen=True)
class _AccessStep:
    """A snapped road's access gain over one step, lowered from the step's own peel instead of
    re-peeling the block per candidate: exact, because burdens are integers and the peel's depth
    with one more road is `min(base depth, 1 + hops from the parcels the road fronts)` whenever
    the road reaches the street.

    Liveness is one distance test: a snapped road's touch-component reaches a street iff the road
    lies within `tol` of a street or of a committed road, since every committed road is itself live
    (it had a positive gain when committed, and adding roads only merges components).
    `at_step` asserts that premise for the committed network it is given. A road that is not live
    seeds nothing, so its gain is exactly 0.0, as the re-peel's is.

    `live`, `seeded` and `relaxed_burden` are the seams the guard's faults patch
    (tests/methods/test_arterial_access_step.py, research/roadless/arterial_check.py)."""
    neighbours: list[set[int]]
    parcel_tree: STRtree             # positional over the block's parcels, the peel's order
    seed: BaseGeometry | None        # streets + committed roads (all live), prepared
    tol: float
    depths: NDArray[np.int64]        # the step's peel, positional; stranded = n_parcels + 1
    burden: int                      # sum of depths ** 2
    base_burden: float               # the no-roads burden -- the objective's denominator
    base: float                      # the objective over the committed roads

    def live(self, real: LineString) -> bool:
        return self.seed is not None and bool(shapely.dwithin(real, self.seed, self.tol))

    def seeded(self, real: LineString) -> NDArray[np.intp]:
        """The parcels `real` fronts (within `tol`, the peel's seeding) that it lowers to 1."""
        near = self.parcel_tree.query(real, predicate="dwithin", distance=self.tol)
        return np.sort(near[self.depths[near] > 1])

    def relaxed_burden(self, seeds: NDArray[np.intp]) -> int:
        """The burden once `seeds` sit at depth 1: breadth-first from them, a parcel lowered only
        when one more hop beats its current depth (and nothing past it then can improve)."""
        # A list, not the array: the walk reads one element at a time, where numpy's per-element
        # access costs several times a list's.
        depths: list[int] = self.depths.tolist()
        burden = self.burden
        frontier: deque[int] = deque()
        for s in seeds.tolist():
            old = depths[s]
            depths[s] = 1
            burden -= old * old - 1
            frontier.append(s)
        neighbours = self.neighbours
        while frontier:
            i = frontier.popleft()
            nxt = depths[i] + 1
            for j in neighbours[i]:
                old = depths[j]
                if nxt < old:
                    depths[j] = nxt
                    burden -= old * old - nxt * nxt
                    frontier.append(j)
        return burden

    def of(self, real: LineString) -> float:
        if self.base_burden == 0.0 or not self.live(real):
            return 0.0
        # The same float arithmetic as `value(trial) - base`: the burden is an exact integer.
        return (1.0 - float(self.relaxed_burden(self.seeded(real))) / self.base_burden) - self.base


@dataclass(frozen=True)
class _AccessBlock:
    adjacency: ParcelAdjacency
    base_burden: float          # access burden with no roads at all -- the denominator
    parcel_tree: STRtree        # positional over the block's parcels, the peel's order

    def value(self, roads: GeoDataFrame) -> float:
        if self.base_burden == 0.0:
            return 0.0
        # Prefix-stable unreached depth: the burden is compared across the greedy's steps.
        depths = parcel_access_layers(self.adjacency, roads, unreached=past_every_parcel)
        return 1.0 - access_burden(depths) / self.base_burden

    def at_step(self, committed: CommittedNetwork, realizer: ChordRealizer) -> StepGain:
        if not realizer.snaps:
            return _Replanarized(self, committed, self.value(committed.roads))
        adj = self.adjacency
        depths = parcel_access_layers(adj, committed.roads, unreached=past_every_parcel
                                      ).to_numpy(dtype=np.int64)
        burden = int((depths ** 2).sum())
        base = 0.0 if self.base_burden == 0.0 else 1.0 - burden / self.base_burden
        seed = street_connectivity(adj.block.streets, committed.roads, adj.tol).seed_geom
        if seed is not None:
            shapely.prepare(seed)
        parts = _line_parts(committed.roads)
        # The gain's exactness rests on every committed road being live (see `_AccessStep`).
        assert not parts or (seed is not None and bool(shapely.dwithin(
            np.asarray(parts, dtype=object), seed, adj.tol).all())), (
            f"{adj.block.block_id}: a committed road does not reach the street")
        return _AccessStep(neighbours=adj.neighbours, parcel_tree=self.parcel_tree, seed=seed,
                           tol=adj.tol, depths=depths, burden=burden,
                           base_burden=self.base_burden, base=base)


@dataclass(frozen=True)
class _MetricBlock:
    ctx: _BlockScoringContext
    pick: Callable[[float, float], float]     # (E, directness) -> the objective's one number

    def value(self, roads: GeoDataFrame) -> float:
        e, direct = self.ctx.score(roads)
        return self.pick(e, direct)

    def at_step(self, committed: CommittedNetwork, realizer: ChordRealizer) -> StepGain:
        base = self.value(committed.roads)
        if realizer.snaps:
            return _Incremental(self.ctx.step(committed.roads), self.pick, base)
        return _Replanarized(self, committed, base)


def _efficiency(e: float, direct: float) -> float:
    del direct
    return e


def _directness(e: float, direct: float) -> float:
    del e
    return direct


@dataclass(frozen=True)
class Access:
    """`1 - access_burden(roads) / access_burden(no roads)`: the share of the summed squared
    access depth the roads remove. NOT submodular, so never pair it with `LazyEngine` -- see
    there, and use `ShortlistEngine` at scale."""

    @property
    def identity(self) -> ObjectiveIdentity:
        return self

    def for_block(self, adjacency: ParcelAdjacency) -> BlockObjective:
        base_burden = access_burden(parcel_access_layers(
            adjacency, None, unreached=past_every_parcel))
        return _AccessBlock(adjacency, base_burden,
                            STRtree(list(adjacency.block.parcels.geometry)))


@dataclass(frozen=True)
class Efficiency:
    """Sampled network efficiency E of streets + roads (`budget.network_efficiency`)."""

    @property
    def identity(self) -> ObjectiveIdentity:
        return self

    def for_block(self, adjacency: ParcelAdjacency) -> BlockObjective:
        return _MetricBlock(_BlockScoringContext(adjacency.block), _efficiency)


@dataclass(frozen=True)
class Directness:
    """Sampled directness of streets + roads (`budget.network_efficiency`'s second value)."""

    @property
    def identity(self) -> ObjectiveIdentity:
        return self

    def for_block(self, adjacency: ParcelAdjacency) -> BlockObjective:
        return _MetricBlock(_BlockScoringContext(adjacency.block), _directness)


ObjectiveIdentity: TypeAlias = Access | Efficiency | Directness
