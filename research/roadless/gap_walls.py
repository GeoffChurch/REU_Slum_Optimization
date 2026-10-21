"""Walls that leak (W): every home's source inside it, each wall priced as a length (owner
2026-10-21: "can't they go through other buildings, just slowly?"; "the interior of a building
conducts at some rate ... and the boundary has higher resistance"; "everyone would pay to get out
of their own home"; "A middle-ground could have walls be (imperfect/leaky) diodes"; NOTES, "Walls
that leak: every home's source inside it, each wall priced as a length"; the brief
.superpowers/sdd/2026-10-21-leaky-walls). An option beside T (gap_clear.Model, untouched): a scorer
of 0/1 clearings on a base metric's fixed mesh, no optimizer (SIMP's interpolation of a partly
cleared building's walls is stage 2).

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
level). Not settled within `max_passes`: raises. The passes are recorded per solve.

THE NAME. A part `+W...` after the placement's in gap_clear's grammar (gap_clear._NAME's `walls`
group; gap_clear.complex_of refuses it: no optimizer runs on W): `Wk<k_in>l<ell>` symmetric,
`Wk<k_in>o<ell_out>i<ell_in>n<max_passes>` a diode (a diode with ell_out == ell_in refused: the
symmetric name is that model's). `walls_of` reads a name back to its WallMetric. This module's one
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
from typing import NamedTuple, Protocol

import common
import gap_clear as gcl
import gap_exits as gx
import gap_factor
import gap_flow as gf
import numpy as np
import scipy.sparse as sp
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

    def solver(self, wm: WallModel, st: WallState) -> Solver: ...


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

    def solver(self, wm: WallModel, st: WallState) -> Solver:
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

    def solver(self, wm: WallModel, st: WallState) -> Solver:
        return DiodeSolver(wm, st, max_passes=self.max_passes)


class LinearSolver:
    """SymmetricWalls' solves: the state's operator factored once."""

    def __init__(self, wm: WallModel, st: WallState):
        if not np.array_equal(st.g_ab, st.g_ba):
            raise ValueError("a linear solve of walls that conduct differently each way")
        self.wm, self.st = wm, st
        self.factor = wm.factor(st, st.g_ab)

    def solve(self, rhs: NDArray[np.float64]) -> Solved:
        return Solved(u=self.wm.timed_solve(self.factor, rhs), g=self.st.g_ab, passes=1)


class DiodeSolver:
    """DiodeWalls' solves: the active-set loop (module docstring), per right-hand side."""

    def __init__(self, wm: WallModel, st: WallState, *, max_passes: int):
        self.wm, self.st, self.max_passes = wm, st, max_passes
        c = wm.sys.cells
        w = st.wall
        self.ea, self.eb = c.ea[w], c.eb[w]
        self.asym = st.g_ab[w] != st.g_ba[w]
        # out of its building first: forward where the passage leaves one and enters none
        self.side0 = np.where(st.enter & ~st.leave, -1, 1).astype(np.int8)

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
        for k in range(1, self.max_passes + 1):
            g = self.conductance(side)
            u = self.wm.timed_solve(self.wm.factor(self.st, g), rhs)
            new = self.sides(u, side)
            if not (new != side)[self.asym].any():
                return Solved(u=u, g=g, passes=k)
            side = new
        raise RuntimeError(f"{self.wm.base.block.block_id}: the walls' active set did not settle "
                           f"in {self.max_passes} passes "
                           f"({int(((new != side) & self.asym).sum())} passages still changing)")


# ------------------------------------------------------------------ the scorer

class WallLenses(NamedTuple):
    """A 0/1 clearing's scores under W (`lenses`: P, P_{1/2}, J_2 over every home with demand,
    B (i) the base's, the homes' escape times under the population's demand, the unknowns), its
    state and the population's and P_{1/2}'s solves."""
    lenses: gcl.Lenses
    state: WallState
    pop: Solved
    q: Solved


class WallModel:
    """W on a base model's fixed mesh (module docstring): `base` a gap_clear.Model on the plain
    mesh (its mesh, cells, pattern and plan, leaks, populations and B (i)), `walls` the Strategy.
    The baseline (`base_`) is scored at construction, P_{1/2}'s weights from its escape times."""

    def __init__(self, base: gcl.Model, *, walls: SymmetricWalls | DiodeWalls):
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
        solver0 = walls.solver(self, st0)
        s0 = solver0.solve(self.rhs(st0, self.pop))
        self.u_pop0 = self.home_u(s0.u)
        self.inj_q = gcl.PQ.injection(self.pop, self.u_pop0, self.nodem)
        self.base_ = self._score(st0, solver0, s0)

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

    def open_factor(self, standing: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Per triangle its factor where it is open (not inside a standing building): the
        metric's, the street boost (a cleared building's ground among it)."""
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
        f = np.where(inside, k_in, self.open_factor(standing))
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

    def factor(self, st: WallState, g: NDArray[np.float64]) -> gap_factor.Factor:
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

    def rhs(self, st: WallState, inj: NDArray[np.float64]) -> NDArray[np.float64]:
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
        solver = self.walls.solver(self, st)
        return self._score(st, solver, solver.solve(self.rhs(st, self.pop)))


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

    def model(self, block: object, *, demand: common.Demand) -> WallModel:
        """The block's W scorer on the base's cached plain mesh (its base model's demand
        `demand`: the base's own scores; W's lenses read the population's and P_{1/2}'s)."""
        return WallModel(self.base.clearing(block, demand=demand), walls=self.walls)


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

