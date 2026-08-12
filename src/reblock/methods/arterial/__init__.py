"""Greedy arterial reblocking: insert straight arterials one at a time by best gain per cost.

Split into one module per concern -- primitives (geometry and candidate generation), realize (how
a chord becomes a road), scoring (per-candidate evaluation), policies (which candidates the lazy
engine keeps alive), engines (the search strategies), reblocker (the public method). Re-exported
here so `reblock.methods.arterial.GreedyArterialReblocker` (and the realizer/engine/policy types a
config's `realizer:`/`engine:`/`policy:` block targets, e.g.
`reblock.methods.arterial.SnapToBoundary`, `reblock.methods.arterial.ShortlistEngine` and
`reblock.methods.arterial.Grow`) keep resolving from config.
"""
from __future__ import annotations

from reblock.methods.arterial.engines import (
    ArterialEngine,
    EngineIdentity,
    ExactEngine,
    LazyEngine,
    ShortlistEngine,
    ShortlistIdentity,
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
from reblock.methods.arterial.reblocker import ArterialIdentity, GreedyArterialReblocker

__all__ = ["ArterialEngine", "ArterialIdentity", "CandidatePolicySpec", "ChordRealizer",
          "EngineIdentity", "ExactEngine", "Faithful", "Fixed", "GreedyArterialReblocker", "Grow",
          "IdealChord", "LazyEngine", "RealizerIdentity", "ShortlistEngine",
          "ShortlistIdentity", "SnapToBoundary"]
