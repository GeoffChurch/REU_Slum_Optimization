"""Where a block's building COUNT comes from -- the number every screen metric ranks on.

`kblock` ships a `building_count` column derived from Ecopia, the Mansueto Institute's
commercial footprint vendor. This project's parcels are built from Google Open Buildings
points. The two normally agree within a fifth (median ratio 1.10, p95 1.59 over 16,451 Cape
Town blocks), and where they do not, they disagree in one direction and for one reason.

## Why the default is Open Buildings

Measured 2026-09-16 over the whole Cape Town corpus. The undercount tracks footprint size,
monotonically:

    OB/kblock       blocks    median footprint    survey-informal
     0-1.5x         14,396        60.3 m2              3.9%
     1.5-3x          1,229        41.8 m2              8.7%
     3-5x               25        28.7 m2              4.0%
     >=5x              15        17.0 m2              0.0%

That gradient alone is ambiguous -- Ecopia might simply fail to resolve sub-20 m2 structures.
The ASYMMETRY settles it: blocks the City's 2018 survey confirms as informal are shack-scale
too, and Ecopia counts them fine (median ratio 1.20). It is only the blocks the survey does
NOT know about that explode to 5-20x, and the survey-informal share in that top band is 0.0%.
Ecopia can see shacks; it cannot see shacks that were not there yet.

Three sources, three dates, one story -- on `ZAF.9.3.1_1_5810`, which seeds the flagship
`multiblock_depth` region: the City survey (Feb 2018) finds 0 structures, Ecopia ~714, Open
Buildings ~6,619. The residuals across the worst blocks (33, 52, 74, 714) read as the seed of
a settlement at Ecopia's date.

**So the screens were ranking on a count that is stale exactly where it matters** -- on new
informal settlements, the population this project exists to find. `5810` ranks 8,887th of
16,451 by the shipped proxy on the Ecopia count, and reached the flagship example only because
`metric=depth` re-ranks survivors by the real peel, which uses the Open Buildings parcels.

*Untested:* the exact imagery dates for Ecopia and for Open Buildings v3's Cape Town tile.
The growth reading is inference from three consistent absences, not from timestamps.

## Why it is a Strategy and not a constant

A better count may appear -- a newer Open Buildings release, a national cadastre, a
hand-corrected layer. The choice is resolved ONCE, where config is read, and what flows
downstream is an instance; no metric asks which count it has, because the resolved count is
written into `building_count` before any metric sees the frame.
"""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from shapely import STRtree


@runtime_checkable
class BuildingCount(Protocol):
    """Per-block building count. `identity` keys the derivation cache, so two sources that
    would produce different counts must never share one."""

    @property
    def identity(self) -> Hashable: ...

    def counts(self, blocks: GeoDataFrame, buildings_path: str | Path) -> pd.Series: ...


@dataclass(frozen=True)
class KblockCount:
    """The `building_count` column exactly as kblock ships it (Ecopia).

    Kept because it is what every published number before 2026-09-16 was computed on, and
    because a like-for-like comparison against the Mansueto figures needs it -- not as a
    fallback for missing data.
    """

    @property
    def identity(self) -> Hashable:
        return ("count", "kblock")

    def counts(self, blocks: GeoDataFrame, buildings_path: str | Path) -> pd.Series:
        del buildings_path
        return blocks["building_count"].astype(float).reset_index(drop=True)


@dataclass(frozen=True)
class OpenBuildingsCount:
    """Count the building POINTS this project already builds its parcels from.

    Counting the same layer the parcels come from also removes a standing inconsistency: the
    screen ranked a block on Ecopia's count while the pipeline tessellated it into Open
    Buildings parcels, and on `ZAF.9.3.1_1_5810` those two differed by 9.3x.

    `contains`, not `within`: `STRtree.query(geom, predicate=P)` applies `geom.P(tree_geom)`,
    so `within` asks whether the BLOCK sits inside a point and silently returns nothing.
    """

    @property
    def identity(self) -> Hashable:
        return ("count", "open_buildings")

    def counts(self, blocks: GeoDataFrame, buildings_path: str | Path) -> pd.Series:
        pts = gpd.read_parquet(buildings_path, columns=["geometry"])
        if blocks.crs is not None and pts.crs != blocks.crs:
            pts = pts.to_crs(blocks.crs)
        hit, _ = STRtree(list(pts.geometry)).query(
            np.asarray(blocks.geometry), predicate="contains")
        return pd.Series(np.bincount(hit, minlength=len(blocks)).astype(float))


# The closed set of count sources, by name. Hydra has its own copy of this choice in
# `conf/building_count/`; this is the equivalent for the argparse scripts that bake examples, and
# it exists so the name -> instance conversion happens in ONE place rather than once per script.
# Consumers take a `BuildingCount`; only a CLI boundary is allowed to hold the string.
COUNTERS: dict[str, BuildingCount] = {
    "open_buildings": OpenBuildingsCount(),
    "kblock": KblockCount(),
}


def resolved(blocks: GeoDataFrame, buildings_path: str | Path,
             counter: BuildingCount) -> GeoDataFrame:
    """`blocks` with `building_count` replaced by `counter`'s answer.

    Overwriting the column rather than adding one is what keeps the choice UPSTREAM: every
    metric, gate and region builder reads `building_count` and none of them has to know, or
    ask, which source produced it.
    """
    out = blocks.copy()
    out["building_count"] = counter.counts(blocks, buildings_path).to_numpy()
    return out
