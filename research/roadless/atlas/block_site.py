"""Where a block's build-out lives (owner 2026-10-28: the Demo page's second block): one Site per
block, resolved once where a script meets its configuration (its command line, or a page's own
block) and handed down; below that edge no code names a block.

A block's frames are scratch (git-ignored, gigabytes; SCRATCH/<block id>/):
  data      the plain mesh's baseline (baseline.npz, .json, _times.npz: compute.baseline), the
            lineup's networks' frames, and the carved meshes (.meshes);
  networks  networks.py's frames (compute.frames_of per run: <key>_<run>.npz, .json, _times.npz,
            _sheet.json) and their meshes;
and what is small and kept is committed beside this file (atlas/blocks/<id>/: a build-out's per
frame table, a topology run's roads and timing).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCRATCH = ROOT / ".superpowers" / "sdd" / "2026-10-22-site" / "blocks"
KEPT = HERE / "blocks"


@dataclass(frozen=True, kw_only=True)
class Site:
    """A block's build-out directories (module docstring)."""
    bid: str
    data: Path
    networks: Path
    kept: Path

    @staticmethod
    def of(bid: str) -> Site:
        return Site(bid=bid, data=SCRATCH / bid, networks=SCRATCH / bid / "networks",
                    kept=KEPT / bid)
