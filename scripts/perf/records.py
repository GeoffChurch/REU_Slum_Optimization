"""Typed records for the region-cap experiments, parsed once at the JSON boundary.

These results were written as nested `dict[str, object]` and read back with string subscripts --
`out[ri]["arms"][lb]["at"]["all"]["perm"]` -- which cost sixteen `# type: ignore` comments across
three harnesses and, more to the point, cannot fail loudly. A typo in any of those keys is a
`KeyError` at best; a renamed field silently reshapes a report. That already bit once: the resumed
replication merged two record shapes and died on `KeyError('all')` only *after* three hours of
search had completed, because nothing checked the shape until the report ran.

So the key sets that are fixed at authoring time are spelled as frozen dataclasses, and exactly one
place deals in strings: `load_*` below. Everything downstream uses attribute access, which mypy
checks.

What stays a mapping, deliberately, because these are genuinely open at authoring time:

  * arm labels (`"uncapped"`, `"128"`, `"256"`) -- parameterised by each harness's `CAPS`;
  * budget-fraction keys (`"0.25"`...) -- parameterised by `FRACTIONS`.

Both are loop variables over a declared schema rather than names known while writing the line, and
both are looked up with `[]` and no default, so an unknown key raises instead of substituting.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Metrics:
    """What one road set scores. Shared by whole-network and truncated-prefix results."""

    burden_red: float
    perm: float
    road_m: float
    n_roads: float

    @staticmethod
    def parse(d: Mapping[str, float]) -> Metrics:
        return Metrics(burden_red=d["burden_red"], perm=d["perm"],
                       road_m=d["road_m"], n_roads=d["n_roads"])


@dataclass(frozen=True)
class Arm:
    """One (region, max_anchors) run: its cost, its enumeration, and its whole network.

    `displaced_frac` is a property of the whole network rather than of a `Metrics`, which is why it
    sits here: a truncated prefix's displacement is set by the budget that produced it and carries
    no information.
    """

    secs: float
    cand: tuple[int, ...]
    whole: Metrics
    displaced_frac: float
    roads_wkt: tuple[str, ...]

    @property
    def growth(self) -> float:
        """Last step's candidate count over the first -- the thing capping is meant to flatten."""
        return self.cand[-1] / max(self.cand[0], 1) if self.cand else 1.0


@dataclass(frozen=True)
class RegionRun:
    """Every arm measured on one region block."""

    parcels: int
    arms: Mapping[str, Arm]


@dataclass(frozen=True)
class MatchedRegion:
    """One region's arms compared at equal displacement.

    `reach` is each arm's achievable displacement fraction and `dmax` the largest all of them can
    reach; `at` maps a fraction-of-`dmax` label to that budget's per-arm metrics. Keeping `reach`
    alongside the comparison is deliberate -- a matched result is only meaningful next to evidence
    that the budget actually bound, which is exactly what an earlier version of this work lacked.
    """

    parcels: int
    reach: Mapping[str, float]
    dmax: float
    at: Mapping[str, Mapping[str, Metrics]]


def load_runs(path: Path) -> dict[str, RegionRun]:
    raw = json.loads(path.read_text())
    return {
        ri: RegionRun(
            parcels=rec["parcels"],
            arms={lb: Arm(secs=a["secs"], cand=tuple(a["cand"]),
                          whole=Metrics.parse(a["at"]["all"]),
                          displaced_frac=a["at"]["all"]["displaced_frac"],
                          roads_wkt=tuple(a["roads_wkt"]))
                  for lb, a in rec["arms"].items()})
        for ri, rec in raw.items()
    }


def load_matched(path: Path) -> dict[str, MatchedRegion]:
    raw = json.loads(path.read_text())
    return {
        ri: MatchedRegion(
            parcels=rec["parcels"], reach=rec["reach"], dmax=rec["dmax"],
            at={f: {lb: Metrics.parse(m) for lb, m in row.items()}
                for f, row in rec["at"].items()})
        for ri, rec in raw.items()
    }


def by_size(runs: Mapping[str, RegionRun] | Mapping[str, MatchedRegion]) -> list[str]:
    """Region ids ascending by parcel count -- the order every report here presents."""
    return sorted(runs, key=lambda r: runs[r].parcels)
