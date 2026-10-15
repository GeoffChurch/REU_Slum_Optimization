"""Blocks, the lineup, and the roadless score of a road set.

The 220 consensus recipients on real footprints (the group Steiner study's sample), the
method-comparison lineup through the derive cache, and `Scorer`: one block's grid and demand,
built once, scoring any road set by the free space its corridor opens.
"""
from __future__ import annotations

import dataclasses
import math
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

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
import lifted  # noqa: E402

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


def ids_of(arg: str) -> list[str]:
    """Block ids as a command line names them: `@<file>` (whitespace-separated), `all` (the 220
    recipients and the 82 large blocks) or comma-separated."""
    if arg.startswith("@"):
        return Path(arg[1:]).read_text().split()
    if arg == "all":
        return sorted(set(recipients()) | set(large_blocks()))
    return arg.split(",")


BANK_ENV = "REBLOCK_BLOCK_BANK"


def build_blocks(ids: list[str]) -> list:
    """The blocks, fewest buildings first: from the Cape Town source, or, when $REBLOCK_BLOCK_BANK
    names a bank (a cluster node has no source data: cluster.py ships the blocks a run needs),
    from it -- an id it lacks is an error."""
    bank = os.environ.get(BANK_ENV)
    if bank:
        with open(bank, "rb") as f:
            held = pickle.load(f)
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
    """kblocks from the Cape Town source; an id `<kblock>@<network>` is a region (regions.py),
    built from that source's blocks at any building count."""
    import regions

    named = {i: regions.parse(i) for i in ids}
    plain = [i for i, r in named.items() if r is None]
    out = list(_source().restricted(plain).region().blocks) if plain else []
    faces = [r for r in named.values() if r is not None]
    if faces:
        every = _source("data.min_buildings=1")
        out += [regions.build(r, every.blocks_path,
                              lambda m: list(every.restricted(m).region().blocks))
                for r in faces]
    return out


def write_bank(ids: list[str], path: Path) -> dict:
    """Pickle the blocks `ids` (built from the source) for build_blocks on a machine without it;
    returns them by id."""
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
    displacement (a clearing's population share, clear.Clearing.cost) and B (i) (the population
    share a clearing leaves enclosed): only the demand is reweighted."""

    @property
    def name(self) -> str: ...

    def injection(self, w: NDArray[np.float64], u0: NDArray[np.float64],
                  enclosed: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Per home, what it injects, from its population `w` and its escape time at baseline
        `u0` under the population's own demand (nan where `enclosed`: an enclosed home injects
        nothing whatever this returns, as lifted.Placement gives it no cell)."""
        ...

    def baseline(self, sc: Scorer, sol: lifted.Solution
                 ) -> tuple[NDArray[np.float64], lifted.Solution]:
        """(the injection per home, the baseline's solution under it), from `sol`, the
        baseline's solution under the population's demand."""
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

    def baseline(self, sc: Scorer, sol: lifted.Solution
                 ) -> tuple[NDArray[np.float64], lifted.Solution]:
        return sc.w, sol


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

    def baseline(self, sc: Scorer, sol: lifted.Solution
                 ) -> tuple[NDArray[np.float64], lifted.Solution]:
        inj = self.injection(sc.w, sc.home_u_of(sol, sc.free0), sc.enclosed)
        return inj, lifted.solve(sc.grid, sc.free0, sc.pl.field(inj, sc.grid.inside.shape),
                                 sc.p)


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


# B's test, one definition for the scorer (Scorer) and the road methods' free-first stage
# (free_roads.FreeEnclosed): which homes are enclosed at baseline, the open field a road set leaves,
# and which of the enclosed homes it frees.

def placement(grid: lifted.Grid | lifted.CompositeGrid, polys: NDArray[np.object_],
              p: lifted.Params) -> lifted.Placement:
    """The metric's demand placement on `grid` (lifted.Placement), against the open space that
    reaches the street at baseline (lifted.placeable, the step rule's): its `enclosed` homes are
    B's enclosed homes."""
    return lifted.Placement.of(grid, polys, lifted.placeable(grid, grid.ff0, p),
                               ring_m=lifted.RING_M)


def carved(grid: lifted.Grid | lifted.CompositeGrid,
           freed: NDArray[np.bool_]) -> NDArray[np.float64]:
    """The open field (per cell) with the sub-samples `freed` cleared: what a rule's `freed`
    (Carve's: exactly the corridor) leaves of the block."""
    return (grid.isub & (~grid.bsub | freed)).mean(axis=-1)


def reached(pl: lifted.Placement, grid: lifted.Grid | lifted.CompositeGrid,
            free: NDArray[np.float64], p: lifted.Params) -> NDArray[np.bool_]:
    """Per home: enclosed at baseline and freed in the field `free`, some cell within RING_M of
    its outline open and street-connected there (lifted.Placement.reached; lifted.placeable,
    the step rule's)."""
    return pl.reached(lifted.placeable(grid, free, p))


class Scorer:
    def __init__(self, block, mesh: lifted.MeshSpec, p: lifted.Params, rule=None,
                 population=None, *, demand: Demand):
        population = CountPopulation() if population is None else population
        self._on(block, mesh.build(block, p, population), p, rule, population, demand)

    @classmethod
    def on_grid(cls, block, grid, p: lifted.Params, population, *, demand: Demand) -> Scorer:
        """A scorer on a grid already built (GoalMesh's rounds)."""
        sc = cls.__new__(cls)
        sc._on(block, grid, p, None, population, demand)
        return sc

    def _on(self, block, grid, p: lifted.Params, rule, population, demand: Demand) -> None:
        rule = Carve() if rule is None else rule
        self.block, self.grid, self.p, self.rule = block, grid, p, rule
        self.polys = np.asarray(block.buildings.outlines)
        self.tree = shapely.STRtree(self.polys)
        self.free0 = self.grid.ff0
        # demand only where the street can be reached at baseline (lifted.Placement): fixed from
        # here on, so freeing space only ever adds conductance. The enclosed homes inject
        # nothing and are out of J; a clearing is scored first by the population it leaves
        # enclosed (`enclosed_left`), then by J over the rest (owner 2026-10-11, "B SGTM").
        self.w = population.weights(self.polys)
        self.pl = placement(self.grid, self.polys, p)
        self.enclosed = self.pl.enclosed
        self.enclosed0 = float(self.w[self.enclosed].sum() / self.w.sum())
        sol = lifted.solve(self.grid, self.free0, self.pl.field(self.w, self.grid.inside.shape),
                           p)
        # what each home injects (Demand; the population's own, or reweighted from this
        # baseline): `inj` per home, `f` per cell; u0 and P0 the baseline's under it
        self.demand = demand
        self.inj, sol = demand.baseline(self, sol)
        self.f = self.pl.field(self.inj, self.grid.inside.shape)
        self.P0 = sol.P
        self.u0 = self.home_u_of(sol, self.free0)

    def P_free(self, free) -> float:
        return lifted.solve(self.grid, free, self.f, self.p).P

    def home_u_of(self, sol: lifted.Solution, free) -> np.ndarray:
        """Per building: its congested escape time u_i, the injection-weighted mean potential
        over its cells (so sum_i inj_i u_i = P). nan for the enclosed buildings."""
        ub = lifted.cell_mean_u(sol, free, self.p)
        return np.where(self.enclosed, np.nan, self.pl.home_u(ub.ravel()))

    def home_u(self, free) -> np.ndarray:
        return self.home_u_of(lifted.solve(self.grid, free, self.f, self.p), free)

    def J(self, u: np.ndarray, power: float) -> float:
        """J_p = sum_i inj_i u_i^p over the buildings not enclosed, inj the demand's injection
        (J_1 = P; under the population's demand inj = w, the population)."""
        ok = ~self.enclosed
        return float((self.inj[ok] * u[ok] ** power).sum())

    def adjoint_load(self, u_home: np.ndarray, power: float) -> np.ndarray:
        """The load of J_power's adjoint, dJ/du per cell (in the grid's shape): home i's
        inj_i power u_i^(power - 1) injected as its demand is (the enclosed inject nothing)."""
        return self.pl.field(self.inj * power * np.nan_to_num(u_home) ** (power - 1.0),
                             self.grid.inside.shape)

    def enclosed_left(self, free) -> float:
        """The share of the population still enclosed in the field `free`: the baseline's
        enclosed homes with no cell within RING_M of the outline open and street-connected
        there (B (i), scored before J). With none enclosed, 0 without the reach (on a composite
        mesh a host pass over every cell: 14 -- 20 s on the oversized blocks)."""
        if not self.enclosed.any():
            return 0.0
        freed = reached(self.pl, self.grid, free, self.p)
        return float(self.w[self.enclosed & ~freed].sum() / self.w.sum())

    def perm_p(self, u: np.ndarray, power: float) -> float:
        return 1.0 - (self.J(u, power) / self.J(self.u0, power)) ** (1.0 / power)

    def free_of(self, roads) -> np.ndarray:
        if roads is None or len(roads) == 0:
            return self.free0
        from reblock.budget import road_corridor
        return carved(self.grid, self.rule.freed(self, road_corridor(roads)))

    def perm(self, roads) -> float:
        free = self.free_of(roads)
        if free is self.free0:
            return 0.0
        return 1.0 - self.P_free(free) / self.P0


GOAL_RTOL = 1e-3     # its solves: a ranking (as the tension's)
GOAL_ALPHA = 1.0     # eta = E (s / h)^alpha: 1 and 2 within 1.4x of each other (NOTES)
GOAL_STEP = 6        # a round splits ceil(gap / GOAL_STEP) cells (each adds 3): half the gap
GOAL_REACH = 0.98    # done at this share of the target


@dataclass(frozen=True)
class GoalMesh:
    """The distance mesh a<d0>x<smax> (lifted.AdaptiveMesh) refined where J_power is sensitive,
    to `factor` x its cells. Each round solves the baseline with the buildings `opening` open (1:
    as if every one were cleared, the flow any clearing could start) and its J_power adjoint,
    scores every coarse cell eta = E (s / h)^GOAL_ALPHA (E its share of |w du dlam| over its
    edges, s its side), splits the largest ceil(gap / GOAL_STEP) and balances (lifted.balance).
    NOTES, "Goal-oriented refinement"."""
    h: float
    d0: float
    smax: float
    factor: float
    power: float
    opening: float
    offset: tuple[float, float]

    @property
    def name(self) -> str:
        return (f"h{self.h:g}g{self.d0:g}x{self.smax:g}f{self.factor:g}p{self.power:g}"
                f"o{self.opening:g}")

    @property
    def suffix(self) -> str:
        return (("" if self.h == 0.5 else f".h{self.h:g}")
                + f".g{self.d0:g}x{self.smax:g}f{self.factor:g}p{self.power:g}o{self.opening:g}")

    @property
    def pilot(self) -> lifted.AdaptiveMesh:
        return lifted.AdaptiveMesh(self.h, self.d0, self.smax, offset=self.offset)

    def cells(self, block) -> float:
        """Its target (it stops within GOAL_REACH of it, or at the first round past it)."""
        return float(round(self.factor * self.pilot.cells(block)))

    def pixels(self, block) -> int:
        return self.pilot.pixels(block)

    def build(self, block, p: lifted.Params, population) -> lifted.CompositeGrid:
        lay = self.pilot.layout(block)
        target = self.factor * lifted._cells_of(lay)
        _, footprints, streets = lifted._geometry(block)
        while True:
            grid = lifted.CompositeGrid.from_layout(lay, footprints, streets, self.h,
                                                    band_m=lifted.BAND_M)
            coarse = grid.level > 0
            if len(grid.level) >= GOAL_REACH * target or not coarse.any():
                return grid
            eta = goal_indicator(block, grid, p, population, self.power, self.opening)[coarse]
            mask = np.zeros(len(eta), dtype=bool)
            mask[np.argsort(-eta)[:math.ceil((target - len(grid.level)) / GOAL_STEP)]] = True
            lay = lifted.balance(lifted.split_cells(lay, mask))
            p.solver.release()


def goal_indicator(block, grid, p: lifted.Params, population, power: float,
                   opening: float) -> np.ndarray:
    """Per cell: eta = E (s / h)^GOAL_ALPHA, E the cell's half of |w du dlam| over each of its
    edges (u the baseline with the buildings `opening` open, lam its J_power adjoint), under
    the population's own demand (the mesh is built before any run's demand)."""
    sc = Scorer.on_grid(block, grid, p, population, demand=PopulationDemand())
    xp, K = p.solver.xp, p.K
    op = sc.free0 + opening * (grid.isub.mean(axis=-1) - sc.free0)
    sol = lifted.solve(grid, op, sc.f, p, rtol=GOAL_RTOL, host=False)
    host = dataclasses.replace(sol, u=p.solver.to_host(sol.u), cell=p.solver.to_host(sol.cell),
                               unk_cell=p.solver.to_host(sol.unk_cell))
    lam = lifted.solve(grid, op, sc.adjoint_load(sc.home_u_of(host, op), power), p,
                       rtol=GOAL_RTOL, host=False)

    def per_axis(s: lifted.Solution):
        out = xp.zeros((len(s.unk_cell), K))
        out[s.unk_cell >= 0] = s.u.reshape(-1, K)
        return out

    uk, lk = per_axis(sol), per_axis(lam)
    n = uk.shape[0]
    if n != len(grid.size):
        raise ValueError(f"{n} free cells of {len(grid.size)}: the eps world frees every cell")
    E = xp.zeros(n)
    for ca, cb, ka, kb, w in lifted.edges(grid, op, p):
        q = xp.abs(w * (uk[ca, ka] - uk[cb, kb]) * (lk[ca, ka] - lk[cb, kb]))
        E += 0.5 * (xp.bincount(ca, q, minlength=n) + xp.bincount(cb, q, minlength=n))
    return p.solver.to_host(E) * (grid.size / grid.h) ** GOAL_ALPHA


def mesh_of(token: str) -> lifted.MeshSpec:
    """A mesh named on a command line, at lifted.OFFSET: <h> (lifted.UniformMesh),
    <h>a<d0>x<smax> (lifted.AdaptiveMesh) or <h>g<d0>x<smax>f<factor>p<power>o<opening>
    (GoalMesh)."""
    num = r"([0-9]+(?:\.[0-9]+)?)"
    if m := re.fullmatch(num, token):
        return lifted.UniformMesh(float(m[1]), offset=lifted.OFFSET)
    if m := re.fullmatch(f"{num}a{num}x{num}", token):
        return lifted.AdaptiveMesh(float(m[1]), float(m[2]), float(m[3]), offset=lifted.OFFSET)
    if m := re.fullmatch(f"{num}g{num}x{num}f{num}p{num}o{num}", token):
        return GoalMesh(float(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5]),
                        float(m[6]), offset=lifted.OFFSET)
    raise ValueError(f"unknown mesh {token!r}: <h>, <h>a<d0>x<smax> or "
                     "<h>g<d0>x<smax>f<factor>p<power>o<opening>")
