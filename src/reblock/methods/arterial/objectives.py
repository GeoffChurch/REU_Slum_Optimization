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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

from geopandas import GeoDataFrame
from pyproj import CRS
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from reblock.budget import _BlockScoringContext, _StepContext, access_burden
from reblock.derive.access import ParcelAdjacency, parcel_access_layers, past_every_parcel
from reblock.methods.arterial.primitives import _explode, _planarize, _union_with
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
class _Unioned:
    """Score `real` noded against the committed union incrementally -- bit-exact for a snapped
    road, which meets the network only at shared boundary-graph vertices (see `_union_with`)."""
    objective: BlockObjective
    committed: CommittedNetwork
    base: float

    def of(self, real: LineString) -> float:
        c = self.committed
        trial = _explode(_union_with(c.merged, real), c.crs, 2.0 * c.half_width_m)
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
class _AccessBlock:
    adjacency: ParcelAdjacency
    base_burden: float          # access burden with no roads at all -- the denominator

    def value(self, roads: GeoDataFrame) -> float:
        if self.base_burden == 0.0:
            return 0.0
        # Prefix-stable unreached depth: the burden is compared across the greedy's steps.
        depths = parcel_access_layers(self.adjacency, roads, unreached=past_every_parcel)
        return 1.0 - access_burden(depths) / self.base_burden

    def at_step(self, committed: CommittedNetwork, realizer: ChordRealizer) -> StepGain:
        base = self.value(committed.roads)
        if realizer.snaps:
            return _Unioned(self, committed, base)
        return _Replanarized(self, committed, base)


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
        return _AccessBlock(adjacency, base_burden)


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
