"""Blocks, the lineup, and the roadless score of a road set.

The 220 consensus recipients on real footprints (the group Steiner study's sample), the
method-comparison lineup through the derive cache, and `Scorer`: one block's grid and demand,
built once, scoring any road set by the free space its corridor opens.
"""
from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import lifted  # noqa: E402

LINEUP = ["clearance", "clearance_looped", "cycle_native", "cycle_native_betweenness_contrast",
          "resistance_lp", "euclidean_grid", "greedy_arterial_access_displacement"]


def recipients() -> list[str]:
    rows = pd.read_parquet(REPO / "data/benchmarks/consensus_matrix.parquet",
                           columns=["recipient"])
    return sorted(rows["recipient"].unique())


def build_blocks(ids: list[str]) -> list:
    from hydra import compose, initialize_config_dir

    from reblock.presets import load_stages
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "conf")):
        cfg = compose(config_name="compare_config",
                      overrides=["data=capetown_full", "buildings=footprints"])
    source = load_stages(cfg).source
    blocks = sorted(source.restricted(ids).region().blocks, key=lambda b: len(b.buildings))
    assert all(type(b.buildings).__name__ == "Footprints" for b in blocks)
    return blocks


def arms() -> dict:
    from hydra import compose, initialize_config_dir

    from reblock.presets import load_methods
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "conf")):
        cfg = compose(config_name="compare_config", overrides=["buildings=footprints"])
    reg = load_methods(cfg.all_methods)
    out = {}
    for name in LINEUP:
        m = reg[name]
        if name == "cycle_native_betweenness_contrast":
            sub = m.substrate
            m = dataclasses.replace(m, substrate=dataclasses.replace(
                sub, desire_source=dataclasses.replace(sub.desire_source, workers=1)))
        if name == "greedy_arterial_access_displacement":
            m = dataclasses.replace(m, workers=1)
        out[name] = m
    return out


@dataclass(frozen=True)
class Obliterate:
    """Every building the corridor touches is removed whole."""
    name: str = "obliterate"

    def freed(self, sc: Scorer, corridor) -> np.ndarray:
        hit = sc.tree.query(corridor, predicate="intersects")
        if len(hit) == 0:
            return np.zeros_like(sc.grid.inside)
        return sc.grid.mask_of(shapely.union_all(sc.polys[hit])) | sc.grid.mask_of(corridor)


@dataclass(frozen=True)
class Carve:
    """Exactly the corridor is freed: the geometry the displacement charge pays for."""
    name: str = "carve"

    def freed(self, sc: Scorer, corridor) -> np.ndarray:
        return sc.grid.mask_of(corridor)


class Scorer:
    def __init__(self, block, h: float, p: lifted.Params, rule=Carve()):
        self.block, self.h, self.p, self.rule = block, h, p, rule
        self.polys = np.asarray(block.buildings.outlines)
        self.tree = shapely.STRtree(self.polys)
        streets = list(block.streets.geometry)
        self.grid = lifted.Grid.of(block.boundary, self.polys, streets, h)
        self.f, self.n_fallback = lifted.demand(self.grid, self.polys)
        self.free0 = self.grid.inside & ~self.grid.building
        # homes in pockets sealed at baseline are left out of the demand (and counted), so the
        # demand is fixed and freeing only ever adds conductance
        reach = lifted.grounded(self.grid, self.free0, p)
        self.stranded = float(self.f[~reach].sum()) / max(len(self.polys), 1)
        self.f = np.where(reach, self.f, 0.0)
        self.P0 = self.P_free(self.free0)

    def P_free(self, free) -> float:
        return lifted.solve(self.grid, free, self.f, self.p).P

    def free_of(self, roads) -> np.ndarray:
        if roads is None or len(roads) == 0:
            return self.free0
        from reblock.budget import road_corridor
        return self.free0 | (self.rule.freed(self, road_corridor(roads)) & self.grid.inside)

    def perm(self, roads) -> float:
        free = self.free_of(roads)
        if free is self.free0:
            return 0.0
        return 1.0 - self.P_free(free) / self.P0
