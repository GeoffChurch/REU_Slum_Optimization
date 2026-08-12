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

from dataclasses import dataclass

from reblock.contracts import Block, Proposal
from reblock.methods.arterial.engines import ArterialEngine, EngineIdentity, ExactEngine
from reblock.methods.arterial.realize import ChordRealizer, RealizerIdentity, SnapToBoundary
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, with_width


@dataclass(frozen=True)
class ArterialIdentity:
    """Cache-key identity for GreedyArterialReblocker: one named field per proposal-affecting
    knob. The dataclass type itself discriminates the method (no string tag needed -- a frozen
    dataclass compares type before fields, so this never equals another identity type). Frozen ->
    hashable, so it works as an L1 dict key and pickles into the joblib L2 key."""
    realizer: RealizerIdentity
    objective: str
    cost: str
    # road_width_m when cost in {displacement, repulsion} else 0.0 -- DERIVED (see the property).
    corridor_key: float
    max_roads: int
    n_anchors: int
    top_k: int
    engine: EngineIdentity
    max_anchors: int


@dataclass
class GreedyArterialReblocker:
    # How a candidate chord becomes the road that is scored and committed -- SnapToBoundary (the
    # shippable navigability method) or IdealChord (a diagnostic isolating the effect of
    # frontage-snapping, NOT a universal directness ceiling -- see the design doc's correction
    # note).
    realizer: ChordRealizer = SnapToBoundary()
    objective: str = "directness"    # "access" | "efficiency" | "directness"
    n_anchors: int = 32
    top_k: int = 8
    max_roads: int = 15
    # "length" (Delta-benefit/metre) | "displacement" (Delta-benefit/building, see budget.py)
    # | "repulsion" (Delta-benefit / soft quadratic-tail proximity cost, never-zero & CELF-safe)
    # "displacement_fast" is "displacement" computed incrementally -- 1.43x, agrees to ~1e-10 but
    # not bit-exactly, so it takes a different trajectory on ~29% of runs. Kept as a VARIANT until
    # measured to win or lose; if it always wins it replaces `displacement` and this note goes away.
    cost: str = "length"
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
    def identity(self) -> ArterialIdentity:
        # Every field that changes the proposed roads must be in the derive-cache key. road_width_m
        # changes which roads win only under cost="displacement"/"repulsion"; hold it fixed so
        # length-cost methods stay corridor-independent (two methods differing only in road_width_m
        # must NOT share a cached proposal when it matters). max_roads / n_anchors / top_k all
        # change the greedy search, so they belong in the key too -- otherwise a budget/candidate
        # sweep silently returns another setting's cached proposal. `realizer.identity` (not
        # `realizer` itself) so a non-snapping realizer's irrelevant fields -- none exist today, but
        # the seam is the same one `SnapToBoundary.identity`/`IdealChord.identity` already use --
        # can never leak into the key. `engine.identity` for the identical reason -- ExactEngine
        # has no fields, and LazyEngine's policy/rescore_every only matter when the engine IS lazy.
        corridor_key = self.road_width_m if self.cost in ("displacement", "repulsion") else 0.0
        return ArterialIdentity(
            realizer=self.realizer.identity, objective=self.objective, cost=self.cost,
            corridor_key=corridor_key,
            max_roads=self.max_roads, n_anchors=self.n_anchors, top_k=self.top_k,
            engine=self.engine.identity, max_anchors=self.max_anchors)

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior
        roads = self.engine.run(
            block, objective=self.objective, cost=self.cost, realizer=self.realizer,
            n_anchors=self.n_anchors, top_k=self.top_k, max_roads=self.max_roads,
            half_width_m=self.road_width_m / 2.0, workers=self.workers,
            max_anchors=self.max_anchors)
        realizer_name = type(self.realizer).__name__
        return Proposal(
            block_id=block.block_id, crs=block.crs, edges=None,
            roads=with_width(roads, self.road_width_m),
            proposal_id=f"greedy_arterial_{realizer_name}_{self.objective}",
            method="greedy_arterial",
            params={"segments": len(roads), "realizer": realizer_name,
                    "objective": self.objective,
                    "cost": self.cost, "road_width_m": self.road_width_m,
                    "engine": type(self.engine).__name__},
            block_identity=block.identity)
