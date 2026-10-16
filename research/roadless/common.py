"""Blocks, the lineup, the populations and the demands.

The 220 consensus recipients on real footprints (the group Steiner study's sample) and the 82
large blocks, the blocks themselves (from the Cape Town source, each with its OSM features, or a
bank shipped to a cluster node), the method-comparison lineup through the derive cache, the populations (a building's
weight), the demands (what each home injects) and a road prefix's displacement. (`Scorer`, the
raster's scorer of a road set, and its grids went with the raster: NOTES, "The complex as the
runtime: the raster retired".)
"""
from __future__ import annotations

import dataclasses
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray

if TYPE_CHECKING:
    from geopandas import GeoDataFrame

    from reblock.contracts import Block

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

LINEUP = ["clearance", "clearance_looped", "cycle_native", "cycle_native_betweenness_contrast",
          "resistance_lp", "euclidean_grid", "greedy_arterial_access_displacement"]


def recipients() -> list[str]:
    rows = pd.read_parquet(REPO / "data/benchmarks/consensus_matrix.parquet",
                           columns=["recipient"])
    return sorted(rows["recipient"].unique())


LARGE = HERE / "large82_area.parquet"     # the 82 large blocks (block, n, M_unknowns, ...)


def large_blocks() -> list[str]:
    """The 82 large blocks (LARGE), sorted."""
    return sorted(pd.read_parquet(LARGE, columns=["block"])["block"])


CHECK_BATCH = 40     # ids_of's check builds the blocks this many at a time: memory only


def ids_of(arg: str) -> list[str]:
    """Block ids as a command line names them, `@<file>` (whitespace-separated), `all` (the 220
    recipients and the 82 large blocks) or comma-separated, each checked at this boundary: the
    blocks are built (the bank's, or the source's: build_blocks), so an id it cannot build raises
    here, before any run's work, every unknown one named (phase 2b's ZAF.9.3.1_1_7851 for
    ZAF.9.5.4_1_7851: the source holds it under its building floor and yielded nothing)."""
    if arg.startswith("@"):
        ids = Path(arg[1:]).read_text().split()
    elif arg == "all":
        ids = sorted(set(recipients()) | set(large_blocks()))
    else:
        ids = arg.split(",")
    for k in range(0, len(ids), CHECK_BATCH):
        build_blocks(ids[k:k + CHECK_BATCH])
    return ids


def _known(ids: list[str], have: set[str], *, where: str) -> None:
    """Raises naming every id of `ids` not in `have`, the blocks `where` yields: a block id is
    checked against what builds it (the bank, or the source with its building floor), never taken
    on trust."""
    missing = sorted(set(ids) - have)
    if missing:
        raise ValueError(f"{len(missing)} block id(s) {where} does not yield: "
                         f"{', '.join(missing)} (mistyped, absent, or under its building floor)")


BANK_ENV = "REBLOCK_BLOCK_BANK"


def build_blocks(ids: list[str]) -> list:
    """The blocks, fewest buildings first: from the Cape Town source, or, when $REBLOCK_BLOCK_BANK
    names a bank (a cluster node has no source data: cluster.py ships the blocks a run needs),
    from it. An id the bank or the source does not yield raises (`_known`)."""
    bank = os.environ.get(BANK_ENV)
    if bank:
        with open(bank, "rb") as f:
            held = pickle.load(f)
        _known(ids, set(held), where=f"the bank {bank}")
        blocks = [held[i] for i in ids]
    else:
        blocks = _from_source(ids)
    assert all(type(b.buildings).__name__ == "Footprints" for b in blocks)
    return sorted(blocks, key=lambda b: len(b.buildings))


def _source(*overrides: str):
    from hydra import compose, initialize_config_dir

    from reblock.presets import load_stages
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "conf")):
        cfg = compose(config_name="compare_config",
                      overrides=["data=capetown_full", "buildings=footprints", *overrides])
    return load_stages(cfg).source


def _from_source(ids: list[str]) -> list:
    """kblocks from the Cape Town source, each with its OSM features (`_with_osm`); an id
    `<kblock>@<network>` is a region (regions.py), built from that source's blocks at any
    building count, its features its face's."""
    import regions

    named = {i: regions.parse(i) for i in ids}
    plain = [i for i, r in named.items() if r is None]
    src = _source()
    out = _with_osm(list(src.restricted(plain).region().blocks), src.blocks_path) if plain else []
    faces = [r for r in named.values() if r is not None]
    if faces:
        out += [regions.build(r, _every().blocks_path, member_blocks) for r in faces]
    _known(ids, {b.block_id for b in out}, where="the Cape Town source")
    return out


def _with_osm(blocks: list[Block], blocks_path: Any) -> list[Block]:
    """Each block with its OSM features (regions.with_osm: the features of its country's extract
    meeting its polygon, attrs["osm"]), which the metric's barriers and street boost read: one
    metric for every block and region (NOTES, "The consolidated pass"). The extract is read once,
    over the source's blocks (regions.extract_of, cached), and each block's features are a query
    of it; a bank (write_bank) carries them to a cluster node, which has no extract."""
    import regions

    isos = {b.block_id.split(".", 1)[0] for b in blocks}
    extract = {iso: regions.extract_of(blocks_path, iso) for iso in isos}
    return [regions.with_osm(b, extract[b.block_id.split(".", 1)[0]]) for b in blocks]


def own_osm(b: Block) -> Any:
    """`b`'s OSM features read afresh from the cached extract (regions.Extract.meeting), whatever
    the block carries: the guards' premise, independent of what its build attached."""
    import regions

    return regions.extract_of(_source().blocks_path, b.block_id.split(".", 1)[0]).meeting(
        b.boundary)


def _every() -> Any:
    """The Cape Town source at any building count (its floor dropped)."""
    return _source("data.min_buildings=1")


def member_blocks(ids: list[str]) -> list[Block]:
    """The kblocks of `ids` the Cape Town source holds at any building count, each with its OSM
    features (`_with_osm`): a region's members (regions.build, from every kblock whose interior
    point its face holds: one with no building yields none; the per-block model of a region's
    homes, region_leaks, which checks it got every member it asked for)."""
    src = _every()
    return _with_osm(list(src.restricted(ids).region().blocks), src.blocks_path)


def write_bank(ids: list[str], path: Path) -> dict:
    """Pickle the blocks `ids` (built from the source, every id checked: `_known`) for
    build_blocks on a machine without it; returns them by id."""
    bank = {b.block_id: b for b in _from_source(ids)}
    with open(path, "wb") as f:
        pickle.dump(bank, f)
    return bank


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


class Demand(Protocol):
    """What each home injects, the escape demand the solve carries: resolved once, where a run
    is configured, and handed to the scorer, which asks nothing else of it (owner 2026-10-12, a
    priority-weighted P; NOTES, "Priority-weighted P"). The population still sets the
    displacement (a clearing's population share, gap_clear.Model.cost) and B (i) (the population
    share a clearing leaves walled in): only the demand is reweighted."""

    @property
    def name(self) -> str: ...

    def injection(self, w: NDArray[np.float64], u0: NDArray[np.float64],
                  enclosed: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Per home, what it injects, from its population `w` and its escape time at baseline
        `u0` under the population's own demand (nan where `enclosed`: a walled-in home injects
        nothing whatever this returns, as the demand gives it no point)."""
        ...


@dataclass(frozen=True)
class PopulationDemand:
    """Each home injects its population: J_p = sum_i w_i u_i^p, and J_1 = P."""

    @property
    def name(self) -> str:
        return "w"

    def injection(self, w: NDArray[np.float64], u0: NDArray[np.float64],
                  enclosed: NDArray[np.bool_]) -> NDArray[np.float64]:
        return w


@dataclass(frozen=True)
class PriorityDemand:
    """P_q (owner 2026-10-12: "first set the current emitted by deeper buildings quadratically
    higher and then did P"): home i injects w_i a_i, a_i = u0_i^q / mean_w(u0^q), u0_i its escape
    time at baseline under the population's demand and mean_w the population-weighted mean over
    the homes not enclosed, so the total injection is the population's. The weights are fixed at
    baseline, so P_q = f_q^T L^-1 f_q keeps P's properties: one self-adjoint solve, monotone in
    the open space, convex in the clearing fractions. On independent paths (u_i = R_i w_i)
    dP_q / dR_i goes as w_i^2 u0_i^(2q) and dJ_p / dR_i as w_i^2 u_i^(p - 1), so q = (p - 1) / 2
    matches J_p to first order: q 1/2 J_2, q 1 J_3. At q 0 it is the population's demand,
    injection for injection. (The brief's a_i = (u0_i / mean_w(u0))^q is this times a constant,
    the same at q 0 and 1: the constant moves no lens and no clearing, but the total.) `q` is
    required: it names the run (`name`)."""
    q: float

    @property
    def name(self) -> str:
        return f"pq{self.q:g}"

    def injection(self, w: NDArray[np.float64], u0: NDArray[np.float64],
                  enclosed: NDArray[np.bool_]) -> NDArray[np.float64]:
        live = ~enclosed
        uq = np.where(live, np.nan_to_num(u0), 1.0) ** self.q
        mean = float((w[live] * uq[live]).sum() / w[live].sum())
        return np.where(live, w * uq / mean, w)


def displacement(block, roads, w: np.ndarray) -> float:
    """Population-weighted share displaced: sum_i c_i w_i / sum_i w_i, c_i the share of building
    i's outline the corridor takes (the project's overlap fraction)."""
    if roads is None or len(roads) == 0 or len(w) == 0:
        return 0.0
    from reblock.budget import road_corridor
    c = block.buildings.displacement(road_corridor(roads))
    return float((c * w).sum() / w.sum())


_ROW = "__canonical_row__"      # canonical_order's scratch column (each road's row in the network)


def canonical_order(block: Block, roads: GeoDataFrame) -> GeoDataFrame:
    """THE prefix order of a network: its held roads first, in their own order (the network says
    which they are: its leading rows, `reblock.free_first.held_count`), then the rest in the
    order `street_first_ordered` gives the whole network. So the held roads lead every prefix
    whatever the street-first order would do with them, and every prefix still reaches the
    street: a held road starts at the street or on an earlier held road, and the rest keep the
    street-first order, each prefix of which is connected (held roads added to it change nothing
    there). With nothing held it is `street_first_ordered`, road for road."""
    from reblock.budget import STREET_TOL, street_first_ordered
    from reblock.free_first import held_count
    n = held_count(roads)
    rows = street_first_ordered(block, roads.assign(**{_ROW: np.arange(len(roads))}),
                                STREET_TOL)[_ROW].to_numpy()
    return cast("GeoDataFrame",
                roads.iloc[np.concatenate([np.arange(n), rows[rows >= n]])].reset_index(drop=True))


def prefix_to(block, roads, target: float, w: np.ndarray):
    """The shortest canonical prefix (canonical_order: the held roads, then street-first) whose
    weighted displacement reaches `target` (all roads, in canonical order, if none does):
    `prefix_to_displacement` with a population, and with the network's held roads first."""
    if len(roads) == 0:
        return roads
    ordered = canonical_order(block, roads)
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
