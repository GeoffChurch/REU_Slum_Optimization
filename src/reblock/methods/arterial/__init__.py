"""Greedy arterial reblocking: insert straight arterials one at a time by best gain per cost.

Split into one module per concern -- primitives (geometry and candidate generation), realize (how
a chord becomes a road), scoring (per-candidate evaluation), policies (which candidates the lazy
engine keeps alive), engines (the search strategies), reblocker (the public method). Re-exported
here so `reblock.methods.arterial.GreedyArterialReblocker` (and the realizer/engine types a
config's `realizer:`/`engine:` block targets, e.g. `reblock.methods.arterial.SnapToBoundary` and
`reblock.methods.arterial.LazyEngine`) keep resolving from config.
"""
from __future__ import annotations

from reblock.methods.arterial.engines import ArterialEngine, EngineIdentity, ExactEngine, LazyEngine
from reblock.methods.arterial.realize import (
    ChordRealizer,
    IdealChord,
    RealizerIdentity,
    SnapToBoundary,
)
from reblock.methods.arterial.reblocker import ArterialIdentity, GreedyArterialReblocker

__all__ = ["ArterialEngine", "ArterialIdentity", "ChordRealizer", "EngineIdentity", "ExactEngine",
          "GreedyArterialReblocker", "IdealChord", "LazyEngine", "RealizerIdentity",
          "SnapToBoundary"]
