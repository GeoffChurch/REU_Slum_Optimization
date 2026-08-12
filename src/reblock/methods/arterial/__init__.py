"""Greedy arterial reblocking: insert straight arterials one at a time by best gain per cost.

Split into one module per concern -- primitives (geometry and candidate generation), realize (how
a chord becomes a road), scoring (per-candidate evaluation), policies (which candidates the lazy
engine keeps alive), engines (the search strategies), reblocker (the public method). Re-exported
here so `reblock.methods.arterial.GreedyArterialReblocker` (and the realizer types a config's
`realizer:` block targets, e.g. `reblock.methods.arterial.SnapToBoundary`) keep resolving from
config.
"""
from __future__ import annotations

from reblock.methods.arterial.realize import (
    ChordRealizer,
    IdealChord,
    RealizerIdentity,
    SnapToBoundary,
)
from reblock.methods.arterial.reblocker import ArterialIdentity, GreedyArterialReblocker

__all__ = ["ArterialIdentity", "ChordRealizer", "GreedyArterialReblocker", "IdealChord",
          "RealizerIdentity", "SnapToBoundary"]
