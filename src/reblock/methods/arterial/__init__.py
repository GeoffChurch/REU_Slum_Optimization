"""Greedy arterial reblocking: insert straight arterials one at a time by best gain per cost.

Split into one module per concern -- primitives (geometry and candidate generation), realize (how
a chord becomes a road), objectives (what a road gains), costs (what it is charged), scoring
(per-candidate evaluation), policies (which candidates the lazy engine keeps alive), engines (the
search strategies), reblocker (the public method). Re-exported here so
`reblock.methods.arterial.GreedyArterialReblocker` (and the realizer/objective/cost/engine/policy
types a config's `realizer:`/`objective:`/`cost:`/`engine:`/`policy:` block targets, e.g.
`reblock.methods.arterial.SnapToBoundary`, `reblock.methods.arterial.Access`,
`reblock.methods.arterial.Repulsion`, `reblock.methods.arterial.ShortlistEngine`,
`reblock.methods.arterial.Grow` and the `anchors:` samplers, `reblock.methods.arterial.HardCap`
and its siblings) keep resolving from config.
"""
from __future__ import annotations

from reblock.methods.arterial.anchors import (
    AnchorIdentity,
    AnchorSampler,
    HardCap,
    PerLineFallback,
    Uncapped,
)
from reblock.methods.arterial.costs import (
    ArterialCost,
    CostIdentity,
    Displacement,
    Length,
    Repulsion,
)
from reblock.methods.arterial.engines import (
    ArterialEngine,
    EngineIdentity,
    ExactEngine,
    LazyEngine,
    ShortlistEngine,
    ShortlistIdentity,
)
from reblock.methods.arterial.objectives import (
    Access,
    ArterialObjective,
    Directness,
    Efficiency,
    ObjectiveIdentity,
)
from reblock.methods.arterial.policies import (
    CandidatePolicySpec,
    Faithful,
    Fixed,
    Grow,
)
from reblock.methods.arterial.realize import (
    ChordRealizer,
    IdealChord,
    RealizerIdentity,
    SnapToBoundary,
)
from reblock.methods.arterial.reblocker import GreedyArterialReblocker
from reblock.methods.arterial.stop import DisplacementMeter, GreedyRun, Stop

__all__ = ["Access", "AnchorIdentity", "AnchorSampler", "ArterialCost", "ArterialEngine",
           "ArterialObjective", "CandidatePolicySpec", "ChordRealizer", "CostIdentity",
           "Directness", "Displacement", "DisplacementMeter", "Efficiency", "EngineIdentity",
           "ExactEngine", "Faithful", "Fixed", "GreedyArterialReblocker", "GreedyRun", "Grow",
           "HardCap", "IdealChord", "LazyEngine", "Length", "ObjectiveIdentity",
           "PerLineFallback", "RealizerIdentity", "Repulsion", "ShortlistEngine",
           "ShortlistIdentity", "SnapToBoundary", "Stop", "Uncapped"]
