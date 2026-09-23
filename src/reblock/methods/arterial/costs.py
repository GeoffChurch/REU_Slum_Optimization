"""What the arterial greedy divides a candidate's gain by: an injected `ArterialCost` -- `Length`,
`Displacement` or `Repulsion` -- rather than a `cost` string every engine switches on.

The configured cost is a frozen, fieldless dataclass (its own cache identity). Whatever a cost needs
per greedy step it builds in `at_step` and closes over in the `StepCost` it returns -- for
`Displacement`, the committed corridor's per-building pieces, fixed for the step so each candidate
is priced locally. That object rides the frozen `scoring._StepState` into the fork pool by
copy-on-write, so the engines never pre-build a cost's state or ask which cost they have.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

from shapely.geometry import LineString

from reblock.budget import repulsion, road_corridor
from reblock.buildings import Extents, IncrementalOverlap
from reblock.contracts import Block
from reblock.methods.arterial.objectives import CommittedNetwork


class StepCost(Protocol):
    """The price of committing `real` at one step -- the denominator of its gain."""

    def of(self, real: LineString) -> float: ...


@runtime_checkable
class ArterialCost(Protocol):
    """What a candidate's gain is divided by. `at_step` builds the step's pricing once."""

    @property
    def identity(self) -> CostIdentity: ...

    def at_step(self, block: Block, committed: CommittedNetwork) -> StepCost: ...


@dataclass(frozen=True)
class _PerMetre:
    def of(self, real: LineString) -> float:
        return real.length


@dataclass(frozen=True)
class _NewlyDisplaced:
    overlap: IncrementalOverlap     # the committed corridor's per-building pieces, fixed per step
    half_width_m: float

    def of(self, real: LineString) -> float:
        # Priced against the step's committed pieces, touching only the buildings the candidate
        # reaches -- never a union of the candidate into the committed corridor. Recomputing
        # `displacement(buildings, committed + candidate)` instead is the same number to ~1e-8 and
        # MEASURED 11x slower per candidate with 5 committed roads and 210x with 60, on
        # ZAF.9.3.1_1_5810's footprints: it re-unions the whole network every time, so its cost
        # grows with the network while this one stays ~5 ms. See IncrementalOverlap.
        return self.overlap.delta(real.buffer(self.half_width_m))


@dataclass(frozen=True)
class _Proximity:
    buildings: Extents

    def of(self, real: LineString) -> float:
        return repulsion(self.buildings, real)


@dataclass(frozen=True)
class Length:
    """Delta-benefit per metre of road."""

    @property
    def identity(self) -> CostIdentity:
        return self

    def at_step(self, block: Block, committed: CommittedNetwork) -> StepCost:
        del block, committed
        return _PerMetre()


@dataclass(frozen=True)
class Displacement:
    """Delta-benefit per expected home newly displaced within the road's half-width of the
    committed network (`budget.displacement`, at the block's building tier). A beneficial road
    that displaces nobody ranks above every priced one -- see `scoring.eval_candidate`."""

    @property
    def identity(self) -> CostIdentity:
        return self

    def at_step(self, block: Block, committed: CommittedNetwork) -> StepCost:
        overlap = IncrementalOverlap(block.buildings)
        if len(committed.roads):
            overlap.add(road_corridor(committed.roads))
        return _NewlyDisplaced(overlap, committed.half_width_m)


@dataclass(frozen=True)
class Repulsion:
    """Delta-benefit per the road's OWN quadratic-tail proximity to the building field
    (`budget.repulsion`): constant per candidate and never zero, so CELF-safe and never
    degenerate the way a zero-displacement gap road is."""

    @property
    def identity(self) -> CostIdentity:
        return self

    def at_step(self, block: Block, committed: CommittedNetwork) -> StepCost:
        del committed
        return _Proximity(block.buildings)


CostIdentity: TypeAlias = Length | Displacement | Repulsion
