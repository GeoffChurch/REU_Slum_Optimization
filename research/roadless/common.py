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
class CountPopulation:
    """One unit per building."""
    name: str = "count"

    def weights(self, polys: np.ndarray) -> np.ndarray:
        return np.ones(len(polys))


@dataclass(frozen=True)
class AreaPopulation:
    """Population proportional to footprint area, normalised to mean 1 (so the block total is
    still the building count)."""
    name: str = "area"

    def weights(self, polys: np.ndarray) -> np.ndarray:
        a = shapely.area(polys)
        return a / a.mean() if len(a) and a.mean() > 0 else np.ones(len(polys))


POPULATIONS = {"count": CountPopulation(), "area": AreaPopulation()}


def displacement(block, roads, w: np.ndarray) -> float:
    """Population-weighted share displaced: sum_i c_i w_i / sum_i w_i, c_i the share of building
    i's outline the corridor takes (the project's overlap fraction)."""
    if roads is None or len(roads) == 0 or len(w) == 0:
        return 0.0
    from reblock.budget import road_corridor
    c = block.buildings.displacement(road_corridor(roads))
    return float((c * w).sum() / w.sum())


def prefix_to(block, roads, target: float, w: np.ndarray):
    """The shortest street-first prefix whose weighted displacement reaches `target` (all roads,
    in canonical order, if none does): `prefix_to_displacement` with a population."""
    from reblock.budget import STREET_TOL, street_first_ordered
    if len(roads) == 0:
        return roads
    ordered = street_first_ordered(block, roads, STREET_TOL)
    if displacement(block, ordered, w) < target:
        return ordered
    lo, hi = 0, len(ordered)
    while lo < hi:
        mid = (lo + hi) // 2
        if displacement(block, ordered.iloc[:mid], w) >= target:
            hi = mid
        else:
            lo = mid + 1
    return ordered.iloc[:lo]


@dataclass(frozen=True)
class Obliterate:
    """Every building the corridor touches is removed whole."""
    name: str = "obliterate"

    def freed(self, sc: Scorer, corridor) -> np.ndarray:
        hit = sc.tree.query(corridor, predicate="intersects")
        if len(hit) == 0:
            return sc.grid.sub_of(corridor)
        return sc.grid.sub_of(shapely.union_all([*sc.polys[hit], corridor]))


@dataclass(frozen=True)
class Carve:
    """Exactly the corridor is freed: the geometry the displacement charge pays for."""
    name: str = "carve"

    def freed(self, sc: Scorer, corridor) -> np.ndarray:
        return sc.grid.sub_of(corridor)


class Scorer:
    def __init__(self, block, h: float, p: lifted.Params, rule=None,
                 offset: tuple[float, float] = (0.3713, 0.1931), population=None):
        rule = Carve() if rule is None else rule
        population = CountPopulation() if population is None else population
        self.block, self.h, self.p, self.rule = block, h, p, rule
        self.polys = np.asarray(block.buildings.outlines)
        self.tree = shapely.STRtree(self.polys)
        streets = list(block.streets.geometry)
        self.grid = lifted.Grid.of(block.boundary, self.polys, streets, h, offset=offset)
        self.free0 = self.grid.ff0
        # demand only where the street can be reached at baseline: fixed from here on, so
        # freeing space only ever adds conductance. `stranded` = share of buildings with no
        # reachable ring cell (they inject nothing).
        reach = lifted.grounded(self.grid, self.free0, p)   # bool mask
        self.w = population.weights(self.polys)
        self.f, stranded, self.owner = lifted.demand(self.grid, self.polys, reach, self.w)
        self.live_home = ~stranded
        self.stranded = float(self.w[stranded].sum() / max(self.w.sum(), 1e-12))
        sol = lifted.solve(self.grid, self.free0, self.f, p)
        self.P0 = sol.P
        self.u0 = self.home_u_of(sol, self.free0)

    def P_free(self, free) -> float:
        return lifted.solve(self.grid, free, self.f, self.p).P

    def home_u_of(self, sol: lifted.Solution, free) -> np.ndarray:
        """Per building: its congested escape time u_i, the injection-weighted mean potential
        over its ring cells (so sum_i w_i u_i = P). nan for stranded buildings."""
        ub = lifted.cell_mean_u(sol, free, self.p)
        on = self.owner >= 0
        num = np.bincount(self.owner[on], weights=(self.f * ub)[on], minlength=len(self.polys))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.live_home, num / self.w, np.nan)

    def home_u(self, free) -> np.ndarray:
        return self.home_u_of(lifted.solve(self.grid, free, self.f, self.p), free)

    def J(self, u: np.ndarray, power: float) -> float:
        """J_p = sum_i w_i u_i^p over reachable buildings (J_1 = P)."""
        ok = self.live_home
        return float((self.w[ok] * u[ok] ** power).sum())

    def perm_p(self, u: np.ndarray, power: float) -> float:
        return 1.0 - (self.J(u, power) / self.J(self.u0, power)) ** (1.0 / power)

    def free_of(self, roads) -> np.ndarray:
        if roads is None or len(roads) == 0:
            return self.free0
        from reblock.budget import road_corridor
        g = self.grid
        return (g.isub & (~g.bsub | self.rule.freed(self, road_corridor(roads)))).mean(axis=-1)

    def perm(self, roads) -> float:
        free = self.free_of(roads)
        if free is self.free0:
            return 0.0
        return 1.0 - self.P_free(free) / self.P0
