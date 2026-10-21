"""Walls that leak (W): every home's source inside it, each wall priced as a length (owner
2026-10-21: "can't they go through other buildings, just slowly?"; "the interior of a building
conducts at some rate ... and the boundary has higher resistance"; "everyone would pay to get out
of their own home"; "A middle-ground could have walls be (imperfect/leaky) diodes"; NOTES, "Walls
that leak: every home's source inside it, each wall priced as a length"; the briefs
.superpowers/sdd/2026-10-21-leaky-walls). An option beside T (gap_clear.Model, untouched): a scorer
of 0/1 clearings on a base metric's fixed mesh, and its relaxation, SIMP's world under W (stage 2:
THE RELAXATION below).

THE MODEL, at a 0/1 clearing on the base's fixed mesh (gap_clear: the shy distance, the runtime's
form; the plain mesh, no road carved). Each standing building OWNS its grown footprint, the faces
within w / 2 of it (gap_clear.FullMesh.fzone: its footprint, its shy zone, and a gap narrower than
w that two zones close): a triangle is INSIDE the set S(t) of standing buildings whose grown
footprint holds it, OPEN with none (exactly T's open ground: the shy distance's factor 1 there).
- Interiors conduct: an inside triangle's factor is `k_in` (no street boost: a building is no
  street), an open triangle's the metric's (1, times the street boost on a paved way, as T); k_in
  at most open ground's 1, so clearing a building never slows its ground and P stays monotone.
- Walls are priced as a length: a passage e between triangles a and b with S(a) != S(b) crosses
  walls; from a to b it leaves the buildings S(a) less S(b) and enters S(b) less S(a), costing
  c = ell_out [it leaves one] + ell_in [it enters one], and conducts, in series with what it
  conducts with no wall (T_e = sigma |e| / d_e times its two triangles' factors' harmonic mean),
  1 / (1 / T_e + c / (sigma |e|)): sigma |e| / (d_e + c) at factor 1. Open ground into a building:
  ell_in; out of one: ell_out; a party wall (one building's interior to another's): ell_out +
  ell_in both ways, a linear resistor; a zone two buildings share entered from one of them: ell_in
  (out of it into the other: ell_out). A street face of an inside triangle is its building's wall
  onto the street: c = ell_out (the potential is never negative, so its current only leaves), an
  obtuse exit of one (gap_clear's ground cell) conducting sigma |f| / ell_out, no ground.
- A cleared building is open ground: no walls, its interior and zone the metric's open ground.
- Every home's source is inside it: its population over its own footprint's triangles (its cover,
  not its zone) by area; its escape time the injection-weighted mean potential there; every home
  with a triangle has demand, so none is walled out of P (`nodem`: a footprint the band swallows).
- The leaks are the base metric's, to the cell (gap_exits: on the open ground its sites name, each
  building's destination where the base's placement puts its home's points, at its doors).
- B (i) is the base model's: the street's reach through open ground only, interiors closed
  (gap_clear.Model.live_homes_of on its own factors), never through a wall.

THE SOLVE (`Walls`, a Strategy resolved where the name is read: `walls_of`). `SymmetricWalls`
(ell_out == ell_in): one conductance per passage, a linear system, one factorization a state.
`DiodeWalls` (ell_out != ell_in): a wall's current-voltage curve is increasing and steeper one way,
so the forward problem stays convex with one solution and P = q . u is still twice the minimum
energy (each side quadratic): monotone under clearing, convex in the conductances. An active-set
loop (`DiodeSolver`): each wall passage's side guessed (out of its building first), the system
solved, each side taken from the sign of its current, the same pattern refactored, until no
passage changes side; a passage whose potential difference is within SIDE_REL of the largest
potential keeps its side (its current is roundoff, either conductance the same solution to that
level). Not settled within `max_passes`: raises. The passes are recorded per solve. Where the loop
starts is a Strategy of the model's (`ColdStart`: out of each building; `WarmStart`: every state
after the baseline from the baseline's directions), in no name: the loop settles on the state's
one solution from any start.

THE RELAXATION (`WallRelaxation`, SIMP's world under W: relax.simp, relax.Incumbent and the
polish read it as they read gap_clear.Relaxation). A design x per building in [0, 1] stands for
its cleared density rho = proj(x)^q (relax.project; q and beta set by SIMP's continuation). Every
element a clearing re-weights (a static passage each way, a finite street face, a cell's obtuse
exits) depends on its OWNERS only, the buildings whose grown footprint holds one of its triangles
(19570: of 495,812 passages 96,451 have none, 284,623 one, 105,500 two, 8,589 three, 649 four).
At a 0/1 clearing its value is W's (WallModel.state's arithmetic, per subset of its owners
cleared: `Interp`); at a design it is the multilinear extension of those values in its owners'
densities, sum over the subsets C of v(C) prod_{j in C} rho_j prod_{j not in C} (1 - rho_j): a
conductance moves from its standing value to its cleared one linearly in each owner's density,
v_s + (v_c - v_s) rho with one owner, SIMP's own eps + (1 - eps) proj(x)^q with eps the
standing conductance's share of the cleared one (a wall's length fades from ell to none, its
interior from k_in to the open ground's factor, as the conductance they give rises linearly). So
the relaxation is W exactly at every 0/1 clearing (no eps world: a standing building conducts by
its walls), and monotone (W's every value only rises as a building clears, and the multilinear
extension of a monotone function on the cube is monotone). A wall's LENGTH linear in rho is the
rejected reading: its conductance's slope at standing is d_e / ell of the clearing's gain (5e-4 at
ell 1,000 m), so the gradient would not see walls. An obtuse exit's cleared value is ground (an
infinite conductance), so there its resistance is interpolated instead: ell_out / (sigma |f|)
times 1 - prod rho (the multilinear extension of "an owner stands"), the cell ground once every
owner is cleared. The gradient by the adjoint (a diode's at its settled sides: P is twice the
minimum energy, its envelope the settled sides' conductances), through each element's slopes in
its owners' densities; at a grounded owned obtuse cell its limit, the cell's drain current under
u and under the adjoint over its exits' standing conductance. A diode's loop at each design
starts from the previous design's sides (the relaxation's own warm start: SIMP's iterates move
little), and settles on the design's one solution from any start.

THE NAME. A part `+W...` after the placement's in gap_clear's grammar (gap_clear._NAME's `walls`
group; gap_clear.complex_of refuses it: only W's own relaxation optimizes W): `Wk<k_in>l<ell>`
symmetric, `Wk<k_in>o<ell_out>i<ell_in>n<max_passes>` a diode (a diode with ell_out == ell_in
refused: the symmetric name is that model's). `walls_of` reads a name back to its WallMetric. A
SIMP run's plan (relax.Plan) names no eps world under W (`.e`, `s`: T's worlds of a standing
building at eps; WallModel.relaxation and WallRelaxation.scorer refuse them). This module's one
public constant, SIDE_REL, moves a score at the floating-point level only (exempt from
gap_clear.CODE's digest, which walks this module: a constant added here renames the metrics).

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/gap_walls.py
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple, Protocol

import clear
import common
import gap_clear as gcl
import gap_exits as gx
import gap_factor
import gap_flow as gf
import numpy as np
import relax
import scipy.sparse as sp
import search
from numpy.typing import NDArray

SIDE_REL = 1e-12        # DiodeSolver: a wall passage whose potential difference is within this
                        # share of the largest potential keeps its side (roundoff: either side's
                        # conductance gives the same potentials to that level)


# ------------------------------------------------------------------ a state

class WallState(NamedTuple):
    """A 0/1 clearing's walls (module docstring): `standing` per building, `inside` per triangle
    (in a standing building's grown footprint), per static passage (the cells' `pe` order) its
    conductance from its first cell to its second (`g_ab`) and back (`g_ba`: equal off the walls),
    the static passages crossing a wall (`wall`, positions in `pe`) with, per wall passage,
    whether its first-to-second direction leaves a building (`leave`) and enters one (`enter`);
    per finite street face (the cells' `pf` order) its conductance (`Gf`), per obtuse exit (`pg`)
    an inside one's (`Gg`: an open one is ground), per cell its conductance to the street (`G`, the
    two summed), the ground cells, the cells reached from the ground, the street and the leaks
    (`live`) and the unknowns."""
    standing: NDArray[np.bool_]
    inside: NDArray[np.bool_]
    g_ab: NDArray[np.float64]
    g_ba: NDArray[np.float64]
    wall: NDArray[np.int64]
    leave: NDArray[np.bool_]
    enter: NDArray[np.bool_]
    Gf: NDArray[np.float64]
    Gg: NDArray[np.float64]
    G: NDArray[np.float64]
    ground: NDArray[np.bool_]
    live: NDArray[np.bool_]
    unknown: NDArray[np.bool_]


class Network(Protocol):
    """What a state's solves read: a 0/1 clearing's WallState or a design's Relaxed. Per static
    passage its conductance each way, the passages that may conduct differently each way (`wall`)
    with whether each, first to second, leaves a building and enters one (the diode loop's outward
    start), per cell its conductance to ground (the street's and the obtuse exits') and the
    unknowns."""

    @property
    def g_ab(self) -> NDArray[np.float64]: ...

    @property
    def g_ba(self) -> NDArray[np.float64]: ...

    @property
    def wall(self) -> NDArray[np.int64]: ...

    @property
    def leave(self) -> NDArray[np.bool_]: ...

    @property
    def enter(self) -> NDArray[np.bool_]: ...

    @property
    def G(self) -> NDArray[np.float64]: ...

    @property
    def unknown(self) -> NDArray[np.bool_]: ...


class Relaxed(NamedTuple):
    """A design's network (WallModel.relaxed; module docstring, THE RELAXATION): per static passage
    its conductance first to second (`g_ab`) and back (`g_ba`), the passages that can be walls
    (`wall`: WallModel.cand) with whether each leaves a building and enters one with every
    building standing (`leave`, `enter`), per finite street face its conductance (`Gf`), per cell
    its owned obtuse exits' conductance to ground (`Go`: 0 where it has none or they are ground)
    and its share of their standing resistance (`Ro`: 1 - prod rho, 1 where it has none), its
    conductance to ground (`G`), the ground cells, the live cells and the unknowns; the owners'
    densities it stands for (`rho`, per building)."""
    g_ab: NDArray[np.float64]
    g_ba: NDArray[np.float64]
    wall: NDArray[np.int64]
    leave: NDArray[np.bool_]
    enter: NDArray[np.bool_]
    Gf: NDArray[np.float64]
    Go: NDArray[np.float64]
    Ro: NDArray[np.float64]
    G: NDArray[np.float64]
    ground: NDArray[np.bool_]
    live: NDArray[np.bool_]
    unknown: NDArray[np.bool_]
    rho: NDArray[np.float64]


class Solved(NamedTuple):
    """One right-hand side solved on a state: the potential per cell (0 off the unknowns), the
    conductance each static passage carried (its current's side's), the passes it took."""
    u: NDArray[np.float64]
    g: NDArray[np.float64]
    passes: int


class Solver(Protocol):
    """A state's solves (`Walls.solver`)."""

    def solve(self, rhs: NDArray[np.float64]) -> Solved: ...


# ------------------------------------------------------------------ the walls' Strategy

class Walls(Protocol):
    """How walls conduct and are solved (module docstring): the name's part, the interiors'
    factor, each wall passage's cost both ways from what it leaves and enters, and a state's
    solver."""

    @property
    def name(self) -> str: ...

    @property
    def k_in(self) -> float: ...

    @property
    def ell_out(self) -> float:
        """Out of a building (onto open ground, or the street through a street face)."""
        ...

    @property
    def ell_in(self) -> float:
        """Into a building."""
        ...

    def costs(self, leave: NDArray[np.bool_], enter: NDArray[np.bool_]
              ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    def solver(self, wm: WallModel, st: Network, prior: NDArray[np.int8]) -> Solver:
        """The state's solver, a diode's loop starting each wall passage on `prior`'s side (per
        static passage: 1 first to second, -1 back, 0 none: out of its building)."""
        ...


def _check_lengths(k_in: float, *ells: float) -> None:
    """k_in in (0, 1] (a cleared building's ground conducting no slower than a standing interior,
    so clearing never raises P), each length positive (a wall of length 0 is no wall) and finite."""
    if not (0.0 < k_in <= 1.0 and all(0.0 < e < math.inf for e in ells)):
        raise ValueError(f"walls: k_in in (0, 1] and each wall's length positive and finite: "
                         f"k_in {k_in}, lengths {ells}")


@dataclass(frozen=True, kw_only=True)
class SymmetricWalls:
    """A wall costs `ell` each way (ell_out == ell_in): a party wall 2 ell; linear."""
    k_in: float
    ell: float

    def __post_init__(self) -> None:
        _check_lengths(self.k_in, self.ell)

    @property
    def name(self) -> str:
        return f"Wk{gx.num(self.k_in)}l{gx.num(self.ell)}"

    @property
    def ell_out(self) -> float:
        return self.ell

    @property
    def ell_in(self) -> float:
        return self.ell

    def costs(self, leave: NDArray[np.bool_], enter: NDArray[np.bool_]
              ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        c = self.ell * (leave.astype(np.float64) + enter.astype(np.float64))
        return c, c

    def solver(self, wm: WallModel, st: Network, prior: NDArray[np.int8]) -> Solver:
        return LinearSolver(wm, st)


@dataclass(frozen=True, kw_only=True)
class DiodeWalls:
    """Out through a wall `ell_out`, in `ell_in` (they differ), a party wall the two summed both
    ways; the active-set loop settles within `max_passes` or raises."""
    k_in: float
    ell_out: float
    ell_in: float
    max_passes: int

    def __post_init__(self) -> None:
        _check_lengths(self.k_in, self.ell_out, self.ell_in)
        if self.ell_out == self.ell_in:
            raise ValueError(f"a diode's lengths differ (ell {self.ell_out} both ways is "
                             "SymmetricWalls)")
        if self.max_passes < 1:
            raise ValueError(f"a diode's passes at most: {self.max_passes}")

    @property
    def name(self) -> str:
        return (f"Wk{gx.num(self.k_in)}o{gx.num(self.ell_out)}i{gx.num(self.ell_in)}"
                f"n{self.max_passes}")

    def costs(self, leave: NDArray[np.bool_], enter: NDArray[np.bool_]
              ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lv, en = leave.astype(np.float64), enter.astype(np.float64)
        return self.ell_out * lv + self.ell_in * en, self.ell_out * en + self.ell_in * lv

    def solver(self, wm: WallModel, st: Network, prior: NDArray[np.int8]) -> Solver:
        return DiodeSolver(wm, st, max_passes=self.max_passes, prior=prior)


class LinearSolver:
    """SymmetricWalls' solves: the state's operator factored once."""

    def __init__(self, wm: WallModel, st: Network):
        if not np.array_equal(st.g_ab, st.g_ba):
            raise ValueError("a linear solve of walls that conduct differently each way")
        self.wm, self.st = wm, st
        self.factor = wm.factor(st, st.g_ab)

    def solve(self, rhs: NDArray[np.float64]) -> Solved:
        return Solved(u=self.wm.timed_solve(self.factor, rhs), g=self.st.g_ab, passes=1)


class DiodeSolver:
    """DiodeWalls' solves: the active-set loop (module docstring), per right-hand side."""

    def __init__(self, wm: WallModel, st: Network, *, max_passes: int,
                 prior: NDArray[np.int8]):
        self.wm, self.st, self.max_passes = wm, st, max_passes
        c = wm.sys.cells
        w = st.wall
        self.ea, self.eb = c.ea[w], c.eb[w]
        self.asym = st.g_ab[w] != st.g_ba[w]
        # each wall passage on the prior's side, else out of its building (forward where it
        # leaves one and enters none)
        outward = np.where(st.enter & ~st.leave, -1, 1).astype(np.int8)
        self.side0 = np.where(prior[w] != 0, prior[w], outward).astype(np.int8)

    def conductance(self, side: NDArray[np.int8]) -> NDArray[np.float64]:
        """Per static passage its conductance on `side` (per wall passage: 1 first to second,
        -1 back)."""
        g = self.st.g_ab.copy()
        w = self.st.wall
        g[w] = np.where(side > 0, self.st.g_ab[w], self.st.g_ba[w])
        return g

    def sides(self, u: NDArray[np.float64], side: NDArray[np.int8]) -> NDArray[np.int8]:
        """Per wall passage the side of its current under `u` (within SIDE_REL of the largest
        potential: `side` kept)."""
        du = u[self.ea] - u[self.eb]
        tol = SIDE_REL * float(np.abs(u).max(initial=0.0))
        return np.where(du > tol, 1, np.where(du < -tol, -1, side)).astype(np.int8)

    def solve(self, rhs: NDArray[np.float64]) -> Solved:
        side = self.side0
        changed = 0
        for k in range(1, self.max_passes + 1):
            g = self.conductance(side)
            u = self.wm.timed_solve(self.wm.factor(self.st, g), rhs)
            new = self.sides(u, side)
            changed = int(((new != side) & self.asym).sum())
            if changed == 0:
                return Solved(u=u, g=g, passes=k)
            side = new
        raise RuntimeError(f"{self.wm.base.block.block_id}: the walls' active set did not settle "
                           f"in {self.max_passes} passes ({changed} passages changed side in the "
                           "last)")


# ------------------------------------------------------------------ the scorer

class WallLenses(NamedTuple):
    """A 0/1 clearing's scores under W (`lenses`: P, P_{1/2}, J_2 over every home with demand,
    B (i) the base's, the homes' escape times under the population's demand, the unknowns), its
    state and the population's and P_{1/2}'s solves."""
    lenses: gcl.Lenses
    state: WallState
    pop: Solved
    q: Solved


class ColdStart:
    """Every state's diode loop from the walls' outward sides (out of each building)."""

    def prior(self, wm: WallModel) -> NDArray[np.int8]:
        return np.zeros(len(wm.sys.cells.pe), dtype=np.int8)


class WarmStart:
    """Every state's diode loop after the baseline from the baseline's directions (its P solve's
    currents' sides; the owner 2026-10-21: yes to a warm start): a clearing turns few of them."""

    def prior(self, wm: WallModel) -> NDArray[np.int8]:
        return wm.sides0


class WallModel:
    """W on a base model's fixed mesh (module docstring): `base` a gap_clear.Model on the plain
    mesh (its mesh, cells, pattern and plan, leaks, populations and B (i)), `walls` the Strategy,
    `start` where a state's diode loop starts (ColdStart, WarmStart: the loop settles on the
    state's one solution from any start, so it is in no name; the passes are recorded). The
    baseline (`base_`) is scored at construction from the outward sides, P_{1/2}'s weights from
    its escape times, the baseline's directions (`sides0`) from its P solve."""

    def __init__(self, base: gcl.Model, *, walls: SymmetricWalls | DiodeWalls,
                 start: ColdStart | WarmStart):
        if base.carving.stops:
            raise ValueError("walls on a carved mesh: the plain mesh only (no road prefix)")
        self.base, self.walls = base, walls
        self.sys = base.sys
        self.timing = gcl.Timing()
        fx, fm, c = base.fx, base.fx.fm, base.sys.cells
        self.fm = fm
        self.sigma = fm.sigma
        self.n = base.n
        self.zone = fm.tzone.tocsr()
        self.cover = fm.tcov.tocsr()
        # the static passages whose two triangles' grown-footprint sets differ: the only ones a
        # state can make a wall
        pe = c.pe
        pa, pb = fm.pa[pe], fm.pb[pe]
        za, zb = self.zone[pa], self.zone[pb]
        self.cand = np.flatnonzero((za != zb).getnnz(axis=1) > 0).astype(np.int64)
        self.za, self.zb = za[self.cand].tocsr(), zb[self.cand].tocsr()
        self.pa, self.pb = pa, pb
        self.T0 = fx.fm.T[pe]                       # sigma |e| / d_e, unboosted
        self.L = np.hypot(*(fm.xy[fm.pu[pe]] - fm.xy[fm.pv[pe]]).T)
        self.b_ways = base.b_ways
        # the street faces: unboosted, the edge as the exits have it
        tf0 = base.exits.edge.mesh(fx.fm).Tf
        self.f_tri, self.f_tf = fm.fa[c.pf], tf0[c.pf]
        self.f_len = np.hypot(*(fm.xy[fm.fu[c.pf]] - fm.xy[fm.fv[c.pf]]).T)
        self.f_open = base.fm.Tf[c.pf]              # the open ground's (boosted, edged)
        self.g_tri = fm.fa[c.pg]                     # the obtuse exits (gap_clear's ground)
        self.g_len = np.hypot(*(fm.xy[fm.fu[c.pg]] - fm.xy[fm.fv[c.pg]]).T)
        self.Pc, self.nodem = self.sources()
        self.PcT = self.Pc.T.tocsr()
        self.pop = base.pop
        self.cost = base.cost
        self.walled0 = base.walled0
        st0 = self.state(np.zeros(self.n, dtype=bool))
        off = (np.asarray(self.PcT @ (~st0.unknown).astype(np.float64)).ravel() > 0) & ~self.nodem
        if off.any():
            raise RuntimeError(f"{base.block.block_id}: homes {np.flatnonzero(off).tolist()} "
                               "have sources off the unknowns at baseline (an inside cell is "
                               "never ground, and every one is reached through its walls)")
        solver0 = walls.solver(self, st0, np.zeros(len(c.pe), dtype=np.int8))
        s0 = solver0.solve(self.rhs(st0, self.pop))
        self.sides0 = self.directions(s0.u)
        self.prior = start.prior(self)
        self.u_pop0 = self.home_u(s0.u)
        self.inj_q = gcl.PQ.injection(self.pop, self.u_pop0, self.nodem)
        # J_power's injection under the base's demand (common.Demand: the population's for P),
        # its weights from W's own escape times at baseline, as the base's are from T's
        self.inj = base.demand.injection(self.pop, self.u_pop0, self.nodem)
        self.base_ = self._score(st0, solver0, s0)
        # the optimizers' clearing (search.SearchState's: only search.apply and the polish change
        # it), as gap_clear.Model's
        self.removed = np.zeros(self.n, dtype=bool)

    def directions(self, u: NDArray[np.float64]) -> NDArray[np.int8]:
        """Per static passage the side of its current under `u` (1 first to second, -1 back, 0
        within SIDE_REL of the largest potential)."""
        c = self.sys.cells
        du = u[c.ea] - u[c.eb]
        tol = SIDE_REL * float(np.abs(u).max(initial=0.0))
        return np.where(du > tol, 1, np.where(du < -tol, -1, 0)).astype(np.int8)

    def sources(self) -> tuple[sp.csr_matrix, NDArray[np.bool_]]:
        """(cells x homes: each home's population over its own footprint's triangles (its cover)
        by area; per home whether it has none: a footprint the band swallows)."""
        fm, c = self.fm, self.sys.cells
        a, b, d = (fm.xy[fm.tri[:, k]] for k in range(3))
        area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (d[:, 1] - a[:, 1])
                            - (b[:, 1] - a[:, 1]) * (d[:, 0] - a[:, 0]))
        Wt = sp.csr_matrix(self.cover.multiply(area[:, None]))
        A_h = np.asarray(Wt.sum(axis=0)).ravel()
        nodem = A_h <= 0.0
        C = sp.csr_matrix((np.ones(fm.nt), (np.arange(fm.nt), c.cell)), shape=(fm.nt, c.n))
        Pc = (C.T @ Wt @ sp.diags(np.where(nodem, 0.0, 1.0 / np.maximum(A_h, 1e-300)))).tocsr()
        return Pc, nodem

    # -- the state

    def open_factor(self, cleared: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Per triangle its factor where it is open (not inside a standing building), `cleared`
        per triangle whether it lies in a cleared building's grown footprint: the metric's, the
        street boost, cleared or not (a cleared building's ground is the metric's open ground; the
        flag is the guards' seam, leaky_check's `cleared-slow`). Elementwise in `cleared`: the
        relaxation's tables read it at all-cleared and at none (Interp)."""
        return np.asarray(self.b_ways, dtype=np.float64)

    def standing_sets(self, standing: NDArray[np.bool_]
                      ) -> tuple[NDArray[np.bool_], NDArray[np.int64], NDArray[np.bool_],
                                 NDArray[np.bool_]]:
        """(per triangle inside a standing building's grown footprint; the static passages
        crossing a wall; per wall passage whether it leaves a building, enters one)."""
        s = standing.astype(np.float64)
        inside = np.asarray(self.zone @ s).ravel() > 0
        D = sp.diags(s)
        za, zb = (self.za @ D).tocsr(), (self.zb @ D).tocsr()
        za.eliminate_zeros()
        zb.eliminate_zeros()
        both = za.multiply(zb).getnnz(axis=1)
        leave = za.getnnz(axis=1) > both
        enter = zb.getnnz(axis=1) > both
        crosses = leave | enter
        return inside, self.cand[crosses], leave[crosses], enter[crosses]

    def state(self, removed: NDArray[np.bool_]) -> WallState:
        """The walls of the 0/1 clearing `removed` (module docstring)."""
        t = time.time()
        standing = ~np.asarray(removed, dtype=bool)
        if len(standing) != self.n:
            raise ValueError(f"a clearing of {len(standing)} buildings, the block has {self.n}")
        inside, wall, leave, enter = self.standing_sets(standing)
        k_in = self.walls.k_in
        cleared = np.asarray(self.zone @ (~standing).astype(np.float64)).ravel() > 0
        f = np.where(inside, k_in, self.open_factor(cleared))
        fa, fb = f[self.pa], f[self.pb]
        T = self.T0 * (2.0 * fa * fb / (fa + fb))
        c_ab, c_ba = self.walls.costs(leave, enter)
        g_ab, g_ba = T.copy(), T.copy()
        sL = self.sigma * self.L[wall]
        g_ab[wall] = 1.0 / (1.0 / T[wall] + c_ab / sL)
        g_ba[wall] = 1.0 / (1.0 / T[wall] + c_ba / sL)
        c = self.sys.cells
        ell_out = self.walls.ell_out
        in_f, in_g = inside[self.f_tri], inside[self.g_tri]
        with np.errstate(divide="ignore"):
            fin = np.where(in_f, 1.0 / (1.0 / (self.f_tf * k_in)
                                        + ell_out / (self.sigma * self.f_len)), self.f_open)
        ground = np.zeros(c.n, dtype=bool)
        ground[c.gcell[~in_g]] = True                 # an open obtuse exit: ground, as T's
        gin = np.where(in_g, self.sigma * self.g_len / ell_out, 0.0)
        G = np.bincount(c.fc, weights=fin, minlength=c.n) + np.bincount(c.gcell, weights=gin,
                                                                         minlength=c.n)
        graph = self.sys.graph
        Wp = np.bincount(graph.pair, weights=g_ab, minlength=graph.npair)
        p = graph.pattern
        seed = ground | (G > 0) | (graph.leak > 0)
        live = gcl._bfs(p.indptr, p.indices, graph.epair, Wp, seed)
        self.timing.add("factors", time.time() - t)
        return WallState(standing=standing, inside=inside, g_ab=g_ab, g_ba=g_ba, wall=wall,
                         leave=leave, enter=enter, Gf=fin, Gg=gin, G=G, ground=ground, live=live,
                         unknown=live & ~ground)

    def factor(self, st: Network, g: NDArray[np.float64]) -> gap_factor.Factor:
        """The state's operator with each static passage conducting `g`, factored on the base's
        pattern and plan."""
        t = time.time()
        graph = self.sys.graph
        Wp = np.bincount(graph.pair, weights=g, minlength=graph.npair)
        p = graph.pattern
        data = np.asarray(gcl._values(p.indptr, p.indices, graph.epair, Wp, st.G + graph.leak,
                                      st.unknown))
        self.timing.add("assembly", time.time() - t)
        t = time.time()
        f = self.sys.planned.factor(data, st.unknown)
        self.timing.add("factorization", time.time() - t)
        return f

    def timed_solve(self, f: gap_factor.Factor, rhs: NDArray[np.float64]) -> NDArray[np.float64]:
        t = time.time()
        u = f.solve(rhs)
        self.timing.add("solves", time.time() - t)
        return u

    def rhs(self, st: Network, inj: NDArray[np.float64]) -> NDArray[np.float64]:
        """The right-hand side per cell of the injection `inj` per home (a source in a ground
        cell, a cleared home's on an obtuse exit, escapes at once)."""
        return np.asarray(self.Pc @ inj) * st.unknown

    def home_u(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per home its escape time (nan with no demand)."""
        return np.where(self.nodem, np.nan, np.asarray(self.PcT @ u))

    # -- the scorer

    def _score(self, st: WallState, solver: Solver, sp_: Solved) -> WallLenses:
        sq = solver.solve(self.rhs(st, self.inj_q))
        uh = self.home_u(sp_.u)
        ok = ~self.nodem
        live = self.street_reach(st)
        left = self.walled0 & ~live
        L = gcl.Lenses(P=float(self.rhs(st, self.pop) @ sp_.u),
                       Pq=float(self.rhs(st, self.inj_q) @ sq.u),
                       J2=float((self.pop[ok] * uh[ok] ** 2).sum()),
                       enclosed=float(self.cost[left].sum()), n_enclosed=int(left.sum()), u=uh,
                       live=live, unknowns=int(st.unknown.sum()))
        return WallLenses(lenses=L, state=st, pop=sp_, q=sq)

    def street_reach(self, st: WallState) -> NDArray[np.bool_]:
        """Per home: the street reaches it (B (i)): the base model's reach through open ground
        only, on its own factors (interiors closed), never through a wall."""
        bm = self.base
        return bm.live_homes_of(bm.factors(bm.k_of(~st.standing)))

    def lenses(self, removed: NDArray[np.bool_]) -> WallLenses:
        """The 0/1 clearing `removed` scored (module docstring)."""
        st = self.state(removed)
        solver = self.walls.solver(self, st, self.prior)
        return self._score(st, solver, solver.solve(self.rhs(st, self.pop)))

    def power(self, removed: NDArray[np.bool_]) -> tuple[WallState, Solved, float]:
        """The 0/1 clearing `removed`'s state, its population's solve and P (P alone: no
        P_{1/2}, no B (i))."""
        st = self.state(removed)
        s = self.walls.solver(self, st, self.prior).solve(self.rhs(st, self.pop))
        return st, s, float(self.rhs(st, self.pop) @ s.u)

    # -- what the optimizers read (gap_clear.Model's names: relax.one, relax.Incumbent, search)

    def J(self, uh: NDArray[np.float64], power: float) -> float:
        """J_power of the homes' escape times `uh` under the run's demand (`inj`), over every
        home with demand."""
        ok = ~self.nodem
        return float((self.inj[ok] * uh[ok] ** power).sum())

    def exact(self, removed: NDArray[np.bool_], power: float) -> tuple[float, float]:
        """(J_power, P under the run's demand) of the 0/1 clearing `removed`: W's discrete scorer
        (gap_clear.Model.exact's pair)."""
        st = self.state(removed)
        s = self.walls.solver(self, st, self.prior).solve(self.rhs(st, self.inj))
        return self.J(self.home_u(s.u), power), float(self.rhs(st, self.inj) @ s.u)

    def baseline(self, power: float) -> tuple[float, float]:
        return self.exact(np.zeros(self.n, dtype=bool), power)

    def enclosed_left(self, removed: NDArray[np.bool_]) -> float:
        """B (i): the population share the 0/1 clearing `removed` leaves walled in, the base's
        (its street reach through open ground only, never through a wall)."""
        return self.base.enclosed_left(removed)

    def relaxation(self, power: float, *, q: float, eps: float) -> WallRelaxation:
        """SIMP's world under W (WallRelaxation). `eps` is the plan's (relax.Plan.eps, handed to
        every model by relax.one): it names T's eps world, a standing building at eps, which W
        has none of (a standing building conducts by its walls, and the relaxation is W itself
        at 0/1), so a plan naming one (an eps but clear.EPS, the plan's unnamed default) is
        refused."""
        if eps != clear.EPS:
            raise ValueError(f"W has no eps world: a plan naming eps {eps:g} (`.e`) is T's")
        return WallRelaxation(self, power, q=q)

    # -- the relaxation's network (module docstring, THE RELAXATION)

    @cached_property
    def interp(self) -> Interp:
        """Every element's value at every 0/1 clearing of its owners (Interp), built once: each
        by WallModel.state's arithmetic, so the relaxation is W's state bit for bit at 0/1."""
        t = time.time()
        c, fm, walls = self.sys.cells, self.fm, self.walls
        nt = fm.nt
        fnc = self.open_factor(np.zeros(nt, dtype=bool))
        fcl = self.open_factor(np.ones(nt, dtype=bool))
        k_in, ell_out, sigma = walls.k_in, walls.ell_out, self.sigma
        # the static passages: owned by the buildings whose grown footprint holds either triangle
        za, zb = self.zone[self.pa], self.zone[self.pb]
        fa0, fb0 = fnc[self.pa], fnc[self.pb]
        g0 = self.T0 * (2.0 * fa0 * fb0 / (fa0 + fb0))
        g_ab, g_ba = g0.copy(), g0.copy()
        passages = []
        for k, rows, own in _groups((za + zb).tocsr()):
            inA, inB = _member(self.zone, self.pa[rows], own), _member(self.zone, self.pb[rows],
                                                                         own)
            pa, pb, T0, L = self.pa[rows], self.pb[rows], self.T0[rows], self.L[rows]
            bits = _bits(k)
            ab, ba = np.empty((len(rows), 1 << k)), np.empty((len(rows), 1 << k))
            for sub in range(1 << k):
                cl = bits[sub]
                st = ~cl
                in_a, in_b = (inA & st).any(axis=1), (inB & st).any(axis=1)
                cl_a, cl_b = (inA & cl).any(axis=1), (inB & cl).any(axis=1)
                fa = np.where(in_a, k_in, np.where(cl_a, fcl[pa], fnc[pa]))
                fb = np.where(in_b, k_in, np.where(cl_b, fcl[pb], fnc[pb]))
                T = T0 * (2.0 * fa * fb / (fa + fb))
                leave = (inA & ~inB & st).any(axis=1)
                enter = (inB & ~inA & st).any(axis=1)
                c_ab, c_ba = walls.costs(leave, enter)
                wl = leave | enter
                ab[:, sub] = np.where(wl, 1.0 / (1.0 / T + c_ab / (sigma * L)), T)
                ba[:, sub] = np.where(wl, 1.0 / (1.0 / T + c_ba / (sigma * L)), T)
            same = np.array_equal(ab, ba)
            passages.append(Owned(pos=rows, own=own, ab=ab, ba=ab if same else ba))
        # the finite street faces: an inside one its building's wall onto the street
        Gf = self.f_open.copy()
        faces = []
        for k, rows, own in _groups(self.zone[self.f_tri].tocsr()):
            tab = np.empty((len(rows), 1 << k))
            with np.errstate(divide="ignore"):
                inside = 1.0 / (1.0 / (self.f_tf[rows] * k_in)
                                + ell_out / (sigma * self.f_len[rows]))
            for sub in range(1 << k):
                tab[:, sub] = inside if sub < (1 << k) - 1 else self.f_open[rows]
            faces.append(Owned(pos=rows, own=own, ab=tab, ba=tab))
        # the obtuse exits, by cell (a cell lies in one face: its exits share their owners)
        Ug = self.zone[self.g_tri].tocsr()
        owned_g = Ug.getnnz(axis=1) > 0
        ground = np.zeros(c.n, dtype=bool)
        ground[c.gcell[~owned_g]] = True
        ocell = np.zeros(c.n, dtype=bool)
        ocell[c.gcell[owned_g]] = True
        if (ground & ocell).any():
            raise RuntimeError("a cell with an owned and an ownerless obtuse exit")
        E = sp.csr_matrix((np.ones(len(c.gcell)), (c.gcell, np.arange(len(c.gcell)))),
                          shape=(c.n, len(c.gcell)))
        M = (E @ Ug).tocsr()
        n_ex = np.asarray(E @ owned_g.astype(np.float64)).ravel()
        if (M.data != np.repeat(n_ex, np.diff(M.indptr))).any():
            raise RuntimeError("a cell whose obtuse exits have different owners")
        Gs = np.bincount(c.gcell, weights=np.where(owned_g, sigma * self.g_len / ell_out, 0.0),
                         minlength=c.n)
        exits = []
        for k, rows, own in _groups(M):
            tab = np.ones((len(rows), 1 << k))
            tab[:, -1] = 0.0                     # every owner cleared: no wall, the cell ground
            exits.append(Owned(pos=rows, own=own, ab=tab, ba=tab))
        _inside, wall, leave, enter = self.standing_sets(np.ones(self.n, dtype=bool))
        if not np.array_equal(wall, self.cand):
            raise RuntimeError("a wall candidate that is no wall with every building standing")
        self.timing.add("factors", time.time() - t)
        return Interp(g_ab=g_ab, g_ba=g_ba, passages=tuple(passages), Gf=Gf, faces=tuple(faces),
                      ground=ground, ocell=ocell, Gs=Gs, exits=tuple(exits), leave=leave,
                      enter=enter)

    def relaxed(self, rho: NDArray[np.float64]) -> Relaxed:
        """The network of the cleared densities `rho` per building in [0, 1] (module docstring,
        THE RELAXATION): every owned element its values' multilinear extension in its owners'
        densities, an owned obtuse exit's resistance likewise (its cell ground with every owner
        cleared)."""
        if len(rho) != self.n or not ((rho >= 0.0) & (rho <= 1.0)).all():
            raise ValueError(f"densities in [0, 1] per building ({self.n}), not {rho}")
        it = self.interp
        t = time.time()
        g_ab, g_ba = it.g_ab.copy(), it.g_ba.copy()
        for o in it.passages:
            w = _weights(rho[o.own])
            g_ab[o.pos] = (o.ab * w).sum(axis=1)
            g_ba[o.pos] = g_ab[o.pos] if o.ba is o.ab else (o.ba * w).sum(axis=1)
        Gf = it.Gf.copy()
        for o in it.faces:
            Gf[o.pos] = (o.ab * _weights(rho[o.own])).sum(axis=1)
        c = self.sys.cells
        Ro = np.ones(c.n)
        for o in it.exits:
            Ro[o.pos] = (o.ab * _weights(rho[o.own])).sum(axis=1)
        ground = it.ground | (it.ocell & (Ro == 0.0))
        Go = np.where(it.ocell & (Ro > 0.0), it.Gs / np.where(Ro > 0.0, Ro, 1.0), 0.0)
        G = np.bincount(c.fc, weights=Gf, minlength=c.n) + Go
        graph = self.sys.graph
        Wp = np.bincount(graph.pair, weights=g_ab, minlength=graph.npair)
        p = graph.pattern
        live = gcl._bfs(p.indptr, p.indices, graph.epair, Wp, ground | (G > 0) | (graph.leak > 0))
        self.timing.add("factors", time.time() - t)
        return Relaxed(g_ab=g_ab, g_ba=g_ba, wall=self.cand, leave=it.leave, enter=it.enter,
                       Gf=Gf, Go=Go, Ro=Ro, G=G, ground=ground, live=live,
                       unknown=live & ~ground, rho=rho)

    def grad(self, net: Relaxed, s: Solved, lam: NDArray[np.float64],
             b_lam: NDArray[np.float64]) -> NDArray[np.float64]:
        """dJ / drho per building at the design `net` by the adjoint (module docstring, THE
        RELAXATION): dJ / dg = -du dlam per passage on its current's side (`s`: the settled
        solve, its potentials u), -u lam per street face, through each element's slopes in its
        owners' densities; an owned obtuse cell's resistance share Ro by (G u)(G lam) / Gs, at a
        ground one its limit, the drain currents under u and lam (`b_lam` the adjoint's
        right-hand side per cell, unmasked; the population's injection the u's)."""
        t = time.time()
        it, c, u = self.interp, self.sys.cells, s.u
        du = u[c.ea] - u[c.eb]
        wp = -du * (lam[c.ea] - lam[c.eb])
        fwd = du > 0.0
        rho = net.rho
        out = np.zeros(self.n)
        for o in it.passages:
            sl = _slopes(o.ab, rho[o.own])
            if o.ba is not o.ab:
                sl = np.where(fwd[o.pos][:, None], sl, _slopes(o.ba, rho[o.own]))
            out += np.bincount(o.own.ravel(), weights=(wp[o.pos][:, None] * sl).ravel(),
                               minlength=self.n)
        wf = -u[c.fc] * lam[c.fc]
        for o in it.faces:
            out += np.bincount(o.own.ravel(), weights=(wf[o.pos][:, None]
                                                       * _slopes(o.ab, rho[o.own])).ravel(),
                               minlength=self.n)
        if it.exits:
            Iu, Il = net.Go * u, net.Go * lam
            gr = it.ocell & net.ground
            if gr.any():
                Iu = np.where(gr, self.drain(s.g, u, np.asarray(self.Pc @ self.inj)), Iu)
                Il = np.where(gr, self.drain(s.g, lam, b_lam), Il)
            we = Iu * Il / np.where(it.ocell, it.Gs, 1.0)
            for o in it.exits:
                out += np.bincount(o.own.ravel(), weights=(we[o.pos][:, None]
                                                           * _slopes(o.ab, rho[o.own])).ravel(),
                                   minlength=self.n)
        self.timing.add("gradient", time.time() - t)
        return out

    def drain(self, g: NDArray[np.float64], u: NDArray[np.float64],
              b: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per cell the current its injection `b` (unmasked) and its passages (conducting `g`)
        bring it under the potentials `u`: at a ground cell (u 0) what drains there."""
        c = self.sys.cells
        return (b + np.bincount(c.eb, weights=g * u[c.ea], minlength=c.n)
                + np.bincount(c.ea, weights=g * u[c.eb], minlength=c.n))


# ------------------------------------------------------------------ the relaxation

class Owned(NamedTuple):
    """Elements with the same number k of owners (Interp): their positions among their kind
    (static passages, finite street faces, cells), their owners (elements x k, buildings), and
    per subset s of the owners cleared (bit i of s: owner i) the element's value at that 0/1
    clearing first to second (`ab`) and back (`ba`: `ab` itself where they agree)."""
    pos: NDArray[np.int64]
    own: NDArray[np.int64]
    ab: NDArray[np.float64]
    ba: NDArray[np.float64]


class Interp(NamedTuple):
    """W's elements at every 0/1 clearing of their owners (WallModel.interp; module docstring,
    THE RELAXATION): per static passage its value each way with no owner (`g_ab`, `g_ba`; the
    owned ones' set per design), the owned ones by their owners' count (`passages`); per finite
    street face likewise (`Gf`, `faces`); per cell whether an ownerless obtuse exit grounds it
    (`ground`), whether it has owned ones (`ocell`), their conductance with an owner standing
    (`Gs`) and their standing resistance's share per subset (`exits`: 1, 0 with every owner
    cleared); per wall candidate (WallModel.cand) whether it leaves a building and enters one with
    every building standing (`leave`, `enter`)."""
    g_ab: NDArray[np.float64]
    g_ba: NDArray[np.float64]
    passages: tuple[Owned, ...]
    Gf: NDArray[np.float64]
    faces: tuple[Owned, ...]
    ground: NDArray[np.bool_]
    ocell: NDArray[np.bool_]
    Gs: NDArray[np.float64]
    exits: tuple[Owned, ...]
    leave: NDArray[np.bool_]
    enter: NDArray[np.bool_]


def _bits(k: int) -> NDArray[np.bool_]:
    """(2^k, k): whether owner i is cleared in subset s."""
    s = np.arange(1 << k)
    return ((s[:, None] >> np.arange(k)[None, :]) & 1).astype(bool)


def _weights(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per element (row of `r`, its owners' densities) per subset s its multilinear weight,
    prod over owners of rho where cleared in s, 1 - rho where standing (exactly 0 or 1 at 0/1)."""
    m, k = r.shape
    bits = _bits(k)
    w = np.ones((m, 1 << k))
    for i in range(k):
        w *= np.where(bits[None, :, i], r[:, i:i + 1], 1.0 - r[:, i:i + 1])
    return w


def _slopes(table: NDArray[np.float64], r: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per element per owner i the slope of its multilinear extension in rho_i: over the subsets
    s without i, (table[s + i] - table[s]) times the other owners' weights."""
    m, k = r.shape
    s = np.arange(1 << k)
    out = np.empty((m, k))
    for i in range(k):
        lo = s[((s >> i) & 1) == 0]
        w = np.ones((m, len(lo)))
        for j in range(k):
            if j != i:
                w *= np.where((((lo >> j) & 1) == 1)[None, :], r[:, j:j + 1], 1.0 - r[:, j:j + 1])
        out[:, i] = ((table[:, lo | (1 << i)] - table[:, lo]) * w).sum(axis=1)
    return out


def _groups(U: sp.csr_matrix) -> list[tuple[int, NDArray[np.int64], NDArray[np.int64]]]:
    """Per owner count k > 0 of the rows of `U` (elements x buildings, nonzero where it owns):
    (k, the rows with k owners, their owners: rows x k, ascending)."""
    U = U.tocsr()
    U.sum_duplicates()
    U.eliminate_zeros()
    U.sort_indices()
    nnz = np.diff(U.indptr)
    out = []
    for k in np.unique(nnz[nnz > 0]):
        rows = np.flatnonzero(nnz == k).astype(np.int64)
        own = U.indices[U.indptr[rows][:, None] + np.arange(k)[None, :]].astype(np.int64)
        out.append((int(k), rows, own))
    return out


def _member(Z: sp.csr_matrix, tri: NDArray[np.int64], own: NDArray[np.int64]) -> NDArray[np.bool_]:
    """Per element e and owner i whether building own[e, i]'s grown footprint holds triangle
    tri[e] (Z: triangles x buildings)."""
    sub = Z[tri].tocsr()
    sub.eliminate_zeros()
    sub.sort_indices()
    nb = Z.shape[1]
    row = np.repeat(np.arange(len(tri), dtype=np.int64), np.diff(sub.indptr))
    keys = row * nb + sub.indices.astype(np.int64)
    q = (np.arange(len(tri), dtype=np.int64)[:, None] * nb + own).ravel()
    pos = np.minimum(np.searchsorted(keys, q), max(len(keys) - 1, 0))
    hit = (keys[pos] == q) if len(keys) else np.zeros(len(q), dtype=bool)
    return np.asarray(hit).reshape(own.shape)


class WallRelaxation:
    """W's relaxation (module docstring, THE RELAXATION), as relax.simp, relax.Incumbent and
    search's worlds read gap_clear.Relaxation: J_power at a design x (rho = proj(x)^q, q and beta
    set by SIMP's continuation), its gradient in the projected clearing by the adjoint; the exact
    scores W's discrete scorer's (WallModel.exact). A diode's loop at each design starts from the
    last design's sides (`sides`: the baseline's first)."""

    def __init__(self, model: WallModel, power: float, *, q: float):
        self.m, self.power_, self.q = model, power, q
        self.beta = 0.0
        self.J0_ = model.baseline(power)[0]
        self.sides = model.sides0.copy()

    @property
    def c(self) -> WallModel:
        return self.m

    @property
    def power(self) -> float:
        return self.power_

    @property
    def J0(self) -> float:
        return self.J0_

    def proj(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(relax.project(x, self.beta))

    def dproj(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(relax.dproject(x, self.beta))

    def rho(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """The cleared densities a design stands for."""
        return np.asarray(self.proj(x) ** self.q)

    def solved(self, x: NDArray[np.float64]) -> tuple[Relaxed, Solved, NDArray[np.float64]]:
        """The design's network, its run's demand's solve (a diode's loop from the last design's
        sides, which it then holds) and the homes' escape times."""
        m = self.m
        net = m.relaxed(self.rho(x))
        s = m.walls.solver(m, net, self.sides).solve(m.rhs(net, m.inj))
        self.sides = m.directions(s.u)
        return net, s, m.home_u(s.u)

    def value(self, x: NDArray[np.float64]) -> float:
        return self.m.J(self.solved(x)[2], self.power_)

    def value_sgrad(self, x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        m, p = self.m, self.power_
        net, s, uh = self.solved(x)
        b = m.inj if p == 1.0 else np.where(m.nodem, 0.0,
                                             m.inj * p * np.nan_to_num(uh) ** (p - 1.0))
        lam = s.u if p == 1.0 else m.timed_solve(m.factor(net, s.g), m.rhs(net, b))
        g = m.grad(net, s, lam, np.asarray(m.Pc @ b))
        if self.q != 1.0:
            g = g * self.q * np.maximum(self.proj(x), 1e-300) ** (self.q - 1.0)
        return m.J(uh, p), g

    def value_grad(self, x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        J, gs = self.value_sgrad(x)
        return J, gs * self.dproj(x)

    def exact(self, x: NDArray[np.float64]) -> float:
        return self.exact_JP(x)[0]

    def exact_JP(self, x: NDArray[np.float64]) -> tuple[float, float]:
        return self.m.exact(x > 0, self.power_)

    def perm(self, J: float) -> float:
        return float(1.0 - (J / self.J0_) ** (1.0 / self.power_))

    def score(self, s: search.SearchState, removed: NDArray[np.bool_]) -> search.Score:
        """The relaxation as a search.World: J of the 0/1 clearing (W's own; no P here)."""
        return search.Score(self.value(removed.astype(np.float64)), np.nan)

    def scorer(self, eps: float) -> WallRelaxation:
        """The incumbent's and the polish's scorer (relax.Incumbent): W's relaxation at q 1, no
        projection, W itself at every 0/1 clearing. `eps` is the plan's scoring eps
        (relax.Plan.kscore), T's eps world: refused unless 0 (unnamed)."""
        if eps != 0.0:
            raise ValueError(f"W has no eps world: a plan scoring at eps {eps:g} (`.k<n>s`) is "
                             "T's")
        return WallRelaxation(self.m, self.power_, q=1.0)


# ------------------------------------------------------------------ the metric, named

@dataclass(frozen=True, kw_only=True)
class WallMetric:
    """W on `base`'s fixed mesh, its leaks, its doors and its B (i) (a gap_clear.ComplexMetric
    under the shy distance, not lumped), the walls `walls`; its name the base's with the walls'
    part after the placement's (module docstring)."""
    base: gcl.ComplexMetric
    walls: SymmetricWalls | DiodeWalls

    def __post_init__(self) -> None:
        if not isinstance(self.base.form, gf.Shy) or self.base.lumped:
            raise ValueError(f"walls on the shy distance's unlumped mesh (its zones are the "
                             f"walls' grown footprints), not {self.base.name}")

    @property
    def name(self) -> str:
        head, sep, tail = self.base.name.partition(f"+v{gcl.CODE}")
        return f"{head}+{self.walls.name}{sep}{tail}"

    def model(self, block: object, *, demand: common.Demand,
              start: ColdStart | WarmStart) -> WallModel:
        """The block's W scorer on the base's cached plain mesh (its base model's demand
        `demand`: the base's own scores; W's lenses read the population's and P_{1/2}'s), its
        diode loops from `start`."""
        return WallModel(self.base.clearing(block, demand=demand), walls=self.walls, start=start)

    @property
    def device(self) -> str:
        return self.base.device

    def clearing(self, block: object, *, demand: common.Demand) -> WallModel:
        """The block's W model for an optimizer (relax.one's `metric.clearing`; J_power under
        `demand`), its exact scores' diode loops warm (WarmStart: the owner's; in no name, the
        loop settling on the state's one solution from any start)."""
        return self.model(block, demand=demand, start=WarmStart())


_N = r"([0-9]+(?:\.[0-9]+)?(?:e-?[0-9]+)?)"
_WALLS = re.compile(rf"\+Wk{_N}(?:l{_N}|o{_N}i{_N}n([0-9]+))")


def walls_part_of(part: str) -> SymmetricWalls | DiodeWalls:
    """The walls a name's part `+W...` spells (symmetric `+Wk<k_in>l<ell>`, a diode
    `+Wk<k_in>o<out>i<in>n<passes>`)."""
    m = _WALLS.fullmatch(part)
    if m is None:
        raise ValueError(f"unknown walls {part!r}: +Wk<k_in>l<ell> or "
                         "+Wk<k_in>o<ell_out>i<ell_in>n<max passes>")
    k_in, ell, out, in_, passes = m.groups()
    if ell is not None:
        return SymmetricWalls(k_in=float(k_in), ell=float(ell))
    return DiodeWalls(k_in=float(k_in), ell_out=float(str(out)), ell_in=float(str(in_)),
                      max_passes=int(str(passes)))


def walls_of(name: str, *, factorization: gap_factor.Factorization) -> WallMetric:
    """The WallMetric `name` names (gap_clear's grammar, its `walls` group the walls'; the rest the
    base's, gap_clear.complex_of), refused unless it spells it back."""
    m = gcl._NAME.fullmatch(name)
    if m is None or m.group("walls") is None:
        raise ValueError(f"{name!r} names no walls (gap_clear's grammar with a +W part)")
    a, b = m.span("walls")
    metric = WallMetric(base=gcl.complex_of(name[:a] + name[b:], factorization=factorization),
                        walls=walls_part_of(m.group("walls")))
    if metric.name != name:
        raise ValueError(f"{name!r} spells the wall metric {metric.name!r}")
    return metric

