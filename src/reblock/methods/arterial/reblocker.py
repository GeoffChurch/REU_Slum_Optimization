"""GreedyArterialReblocker: greedily insert the single straight arterial with the best
objective gain per meter, one at a time, until a road budget runs out. How a candidate chord
becomes a road is an injected `ChordRealizer` -- SnapToBoundary (snapped to the parcel-boundary
graph; the shippable navigability method) or IdealChord (ideal chords; a diagnostic isolating the
effect of frontage-snapping, NOT a universal directness ceiling -- see the design doc's correction
note). Which candidates get scored exactly, each step, is an injected `ArterialEngine` --
ExactEngine (every candidate, every step) or LazyEngine (CELF lazy-greedy, valid only for
submodular objectives). Candidates are through-roads (network<->network) + spurs (network->deep
pocket); continuations are through-roads from committed-segment endpoints (always anchors), so a
spur completes into a through-road for free and crossings planarize into true intersections. See
docs/superpowers/specs/2026-07-09-greedy-arterial-reblocker-design.md.
"""
from __future__ import annotations

import hashlib
from collections.abc import Hashable
from dataclasses import dataclass

from reblock.contracts import Block, Proposal
from reblock.derive_graph import config_identity
from reblock.methods.arterial.costs import ArterialCost, Length
from reblock.methods.arterial.engines import ArterialEngine, ExactEngine
from reblock.methods.arterial.objectives import ArterialObjective, Directness
from reblock.methods.arterial.realize import ChordRealizer, SnapToBoundary
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, with_width


@dataclass
class GreedyArterialReblocker:
    # How a candidate chord becomes the road that is scored and committed -- SnapToBoundary (the
    # shippable navigability method) or IdealChord (a diagnostic isolating the effect of
    # frontage-snapping, NOT a universal directness ceiling -- see the design doc's correction
    # note).
    realizer: ChordRealizer = SnapToBoundary()
    # What a road gains -- Access, Efficiency or Directness (see objectives.py).
    objective: ArterialObjective = Directness()
    n_anchors: int = 32
    top_k: int = 8
    max_roads: int = 15
    # What a road is charged -- Length (per metre), Displacement (per home newly displaced) or
    # Repulsion (soft quadratic-tail proximity, never zero, CELF-safe). See costs.py.
    cost: ArterialCost = Length()
    # Total width of the roads this method emits; also the displacement corridor it
    # scores against (half-width each side). Stamped on every road it returns.
    road_width_m: float = DEFAULT_ROAD_WIDTH_M
    workers: int = 16         # fork-pool size for per-step candidate scoring; 1 == serial no-op
    # Which candidates get scored exactly, each step -- ExactEngine (default, byte-identical) or
    # LazyEngine (CELF lazy-greedy, valid only for submodular objectives). Injected rather than
    # selected by lazy/candidate_policy/rescore_every flags, matching `realizer` above.
    engine: ArterialEngine = ExactEngine()
    # A CAP, not a mode switch: 0 = uncapped (every network vertex + arc-length samples,
    # byte-identical); >0 only ever REDUCES that uncapped anchor count, falling back to
    # ~max_anchors arc-length samples when the uncapped set does not already fit -- see
    # _anchor_points.
    max_anchors: int = 0

    @property
    def identity(self) -> Hashable | None:
        # `workers` sizes the fork pool that scores candidates; the pool and the serial path
        # produce the same roads (tests/methods/test_arterial.py pins pool-vs-serial).
        return config_identity(self, exempt=frozenset({"workers"}))

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior
        roads = self.engine.run(
            block, objective=self.objective, cost=self.cost, realizer=self.realizer,
            n_anchors=self.n_anchors, top_k=self.top_k, max_roads=self.max_roads,
            half_width_m=self.road_width_m / 2.0, workers=self.workers,
            max_anchors=self.max_anchors)
        realizer_name = type(self.realizer).__name__
        objective_name, cost_name = type(self.objective).__name__, type(self.cost).__name__
        # `proposal_id` is half of `Proposal.identity`, which keys the eval caches
        # (`access_after`, `geometric_after`), so it must tell apart every configuration `identity`
        # does -- a readable head, then a digest of the whole identity (it also names render
        # files, which a raw repr would fill with spaces and brackets).
        digest = hashlib.sha256(repr(self.identity).encode()).hexdigest()[:10]
        return Proposal(
            block_id=block.block_id, crs=block.crs, edges=None,
            roads=with_width(roads, self.road_width_m),
            proposal_id=f"greedy_arterial:{realizer_name}:{objective_name}:{cost_name}:{digest}",
            method="greedy_arterial",
            params={"segments": len(roads), "realizer": realizer_name,
                    "objective": objective_name,
                    "cost": cost_name, "road_width_m": self.road_width_m,
                    "engine": type(self.engine).__name__},
            block_identity=block.identity)
