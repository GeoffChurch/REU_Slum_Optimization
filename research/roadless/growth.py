"""Roads grown from the street as a front: two-phase Laplacian growth on the block's eps world
(.superpowers/sdd/2026-10-27-growth/brief.md; owner 2026-10-27).

The model. The grown region is a proposed road's corridor (open, its boost the contrast's: kappa
10, gap_exits.Boost.road, as Model.mesh_of boosts a carved corridor), not ground; the exits stay
the metric's. The world is road_simp's eps world (every standing footprint at EPS, every cell
live, one mesh and one pattern for every state): a triangle the region covers takes openness 1 and
boost max(b_ways, kappa). For an inclusion of contrast kappa the shape derivative of P is
-(kappa - 1) sigma (|grad_t u|^2 + |j_n|^2 / (kappa sigma^2)); on the mesh both terms are what
road_simp._grad returns, per triangle dP/dkt and dP/db at the state, so a candidate triangle's
first-order gain is -(dP/dkt (1 - kt0) + dP/db (boost - b_ways)): an upper bound on its exact gain
(P is convex in the conductances, which are concave in kt and b).

The contrast is a Strategy (Contrast): Road (kappa 10, the metric's own road) or GroundedLeak (the
corridor a leak to ground through the Graph's leak, so the street itself moves inward: one-phase
growth; its C charged as the same corridor's as a road). The connectivity is a Strategy
(Connectivity): Attached (only the front of the street, its connected paved ways and the region
grown from them grows: vehicle roads) or Free (a patch may also nucleate anywhere, an island,
at a fixed charge each: paving and plazas, or with GroundedLeak leaky islands, destinations;
the owner asked for them, 2026-10-27, after the brief's refusal).

The method. A step solves the state once (u, P, the per-triangle first-order gains) and grows:
  patches: a road-width patch per front site (the triangles within PATCH_R of the site's centroid,
    not yet grown; the sites the connectivity's, thinned to one per FRONT_M square), its cost
    nu P0 dD + lambda dC (Cost: dD by common.MEASURE's ramp over each building's covered share,
    as road_simp.Roads.D; dC its added sheet, gap_clear.Model.sheet's; lambda = price P0 /
    road_simp.SHEET_M2), its score the first-order gain over the cost; the plan's top `exact`
    patches rescored by an exact solve each (with Free, the top islands, and an island is taken
    only so: its first-order gain overstates it by its leak times its own resistance to ground;
    its exact score put on the batch's scale, times the last batch's first-order over realized
    drop);
  the batch: `batch` patches drawn one at a time by the pick (Pick: Greedy, the best; or Dbm(eta,
    seed), dielectric breakdown's dial, probability proportional to score^eta, seeded), each
    draw removing every site within `sep` of it (patches apart: a batch grows that many fingers
    by one patch each, not one blob);
  bridges (Bridges: none, or Straight): between two points of the attached set's open edge (on
    the road, never inside a building at EPS, where u is no road's) within `reach` m, a straight
    road-width bridge across a gap (GAP_SHARE of its interior off the attached set); a
    conductance g = sigma (kappa - 1) ROAD_W / L between the cells at its ends lowers P by
    exactly g (u_a - u_b)^2 / (1 + g R_ab) (rank one, Sherman-Morrison;
    R_ab by one dipole solve with the state's factor): the pairs screened by the R 0 bound g du^2
    per cost, the best `screen` by this estimate, the best `top` of those whose estimate beats the
    bar rescored exactly (the corridor itself carved in a solve), and built if the exact gain per
    cost beats the bar: the previous step's realized gain per cost (its batch's).
Steps continue until D (the eps world's) reaches the plan's; every step is a prefix (the step
each triangle was grown in). Loops are counted as the region's faces: the components of the
ungrown triangles of FACE_M2 and more, less one; after each step's patches and again after its
bridges, so the loops the bridges close are told from growth's.

    G=research/roadless/growth.py                      # from the repo root, PYTHONPATH=.
    uv run python $G run <block> <plan>

The bridge's rank-one gain is guarded by growth_check.py.

plan: gr.<k<kappa>|gl<h>Croad>.<att|free<island>>.<greedy|e<eta>s<seed>>.N<batch>.sep<m>
.n<nu>s<price>.<B0|B<reach>m<screen>k<top>>.K<exact>.D<D>, e.g.
gr.k10.att.greedy.N32.sep14.n2s0.1.B21m8k2.K0.D0.15; its name (Plan.name) adds the code's
revision (.v<REV>). Rows in growth_rows/<block>/<name>/: steps.parquet (a row per state),
picks.parquet (the patches built), bridges.parquet (the bridges offered past the screen, built or
not), tri_step.parquet (per triangle the step it was grown in, -1 never: parquet, so the
cluster's pull brings it), written after every step.
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Protocol, TypeAlias

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_exits as gx  # noqa: E402
import metrics  # noqa: E402
import road_simp as rs  # noqa: E402

I64: TypeAlias = NDArray[np.int64]
F64: TypeAlias = NDArray[np.float64]
B: TypeAlias = NDArray[np.bool_]

ROAD_W = rs.ROAD_W       # m: a road's width
PATCH_R = ROAD_W / 2.0   # m: a patch's radius about its site
EPS = rs.EPS             # the eps world's standing footprint
FRONT_M = 1.0            # m: the front's sites, one per square this wide
BRIDGE_FRONT_M = 2.0     # m: the bridges' ends, one per square this wide (each end's site)
BRIDGE_STEP_M = 1.0      # m: a bridge corridor's sample spacing along its centreline
GAP_SHARE = 0.5          # Straight: a bridge's interior samples off the attached set, at least
GAP_SAMPLES = 9          # Straight: interior samples a pair is tested at
OPEN_KT = 0.5            # a triangle on open ground: its eps-world openness at least this
REV = 2                  # the code's revision, in the plan's name (2: the bridges' ends on the
                         # road's open edge, across a gap; 1: from the front, its u read there)
MIN_NEW_M2 = 2.0         # m^2: a patch adds at least this much ground not yet grown
FACE_M2 = 100.0          # m^2: a loop is a face of the region this large (road SIMP's count)
ROAD_KAPPA = 10.0        # GroundedLeak: its C counted as a road of this contrast's adds
                         # (gap_exits.Boost.road's)
FREE_M = PATCH_R         # m: Free's nucleation sites, one per square this wide
SCREEN_PAIRS = 4         # Straight: pairs costed exactly per screened estimate (x screen)
OUT = HERE / "growth_rows"


# ---------------------------------------------------------------- the Strategies


class Contrast(Protocol):
    """The grown corridor's conductance (module docstring): the state's factors where it is
    grown, the boost its conductance added C is charged at, and its first-order gains."""

    @property
    def name(self) -> str: ...

    @property
    def grounded(self) -> bool: ...

    def world(self, g: Growth, on: B) -> tuple[F64, F64, F64 | None]:
        """(per triangle openness kt, boost b; per cell the grown leak to ground, or None)."""
        ...

    def c_boost(self, bw: F64) -> F64:
        """Per triangle the boost a grown triangle's conductance added is counted at."""
        ...

    def gains(self, g: Growth, st: gcl.State, u: F64) -> F64:
        """Per triangle the first-order drop in P from growing it."""
        ...


@dataclass(frozen=True, kw_only=True)
class Road:
    """The metric's own road: the corridor open and boosted to kappa."""
    kappa: float

    def __post_init__(self) -> None:
        if not self.kappa > 1.0:
            raise ValueError(f"kappa {self.kappa}: above the ground's 1")

    @property
    def name(self) -> str:
        return f"k{self.kappa:g}"

    @property
    def grounded(self) -> bool:
        return False

    def world(self, g: Growth, on: B) -> tuple[F64, F64, F64 | None]:
        return (np.where(on, 1.0, g.kt0), np.where(on, self.c_boost(g.bw), g.bw), None)

    def c_boost(self, bw: F64) -> F64:
        return np.maximum(bw, self.kappa)

    def gains(self, g: Growth, st: gcl.State, u: F64) -> F64:
        gk, gb = g.grad(st, u)
        return -(gk * (1.0 - g.kt0) + gb * (self.c_boost(g.bw) - g.bw))


@dataclass(frozen=True, kw_only=True)
class GroundedLeak:
    """The corridor a leak to ground, so the street itself moves inward: each grown triangle
    open (openness 1, its boost the ways') and joined to ground by sigma / h^2 per m^2 of it on
    its cell's diagonal (gap_exits' screened form, as InternalLeak's sigma / open_m^2: here h a
    fraction of a metre, the corridor held near u 0 across a road's width). Its own sheet
    conductance is not its conductance, so C is charged as the same corridor would add as a
    kappa ROAD_KAPPA road (the road-equivalent C: `Croad` in the name). First order: the opening
    (dP/dkt) and the leak, dP/d(leak_c) = -u_c^2."""
    h: float

    def __post_init__(self) -> None:
        if not self.h > 0:
            raise ValueError(f"h {self.h}: positive")

    @property
    def name(self) -> str:
        return f"gl{self.h:g}Croad"

    @property
    def grounded(self) -> bool:
        return True

    def leak_t(self, g: Growth) -> F64:
        """Per triangle its leak once grown: sigma / h^2 per m^2."""
        return g.sigma * g.area / self.h ** 2

    def world(self, g: Growth, on: B) -> tuple[F64, F64, F64 | None]:
        leak = np.bincount(g.m.c.cell[on], weights=self.leak_t(g)[on], minlength=g.m.c.n)
        return np.where(on, 1.0, g.kt0), g.bw, leak

    def c_boost(self, bw: F64) -> F64:
        return np.maximum(bw, ROAD_KAPPA)

    def gains(self, g: Growth, st: gcl.State, u: F64) -> F64:
        gk, _ = g.grad(st, u)
        uc = u[g.m.c.cell]
        return -gk * (1.0 - g.kt0) + uc * uc * self.leak_t(g)


class Connectivity(Protocol):
    """Where a patch may grow (module docstring), and its charge for starting an island."""

    @property
    def name(self) -> str: ...

    def sites(self, g: Growth) -> I64:
        """The patches' sites (triangles), thinned."""
        ...

    def nucleating(self, g: Growth, Pm: sp.csr_matrix) -> B:
        """Per patch (row of Pm) whether it starts an island."""
        ...

    def charge(self, g: Growth, Pm: sp.csr_matrix) -> F64:
        """Per patch (row of Pm) its extra cost in P."""
        ...

    def check(self, exact: int) -> None: ...


@dataclass(frozen=True)
class Attached:
    """The front of the attached set (the street's faces' triangles, the paved ways joined to them
    and the region grown from them): the ungrown triangles beside it."""
    name = "att"

    def sites(self, g: Growth) -> I64:
        return g.thin(g.front(), FRONT_M)

    def nucleating(self, g: Growth, Pm: sp.csr_matrix) -> B:
        return np.zeros(Pm.shape[0], dtype=bool)

    def charge(self, g: Growth, Pm: sp.csr_matrix) -> F64:
        return np.zeros(Pm.shape[0])

    def check(self, exact: int) -> None:
        return None


@dataclass(frozen=True, kw_only=True)
class Free:
    """Attached's front and a nucleation site anywhere ungrown, one per FREE_M square; a patch
    touching nothing grown or attached starts an island and pays `island` (L_P, times P0) on top
    of its cost (owner 2026-10-27: leaky islands allowed, priced)."""
    island: float

    def __post_init__(self) -> None:
        if not self.island >= 0:
            raise ValueError(f"island {self.island}: non-negative")

    @property
    def name(self) -> str:
        return f"free{self.island:g}"

    def sites(self, g: Growth) -> I64:
        return np.union1d(g.thin(g.front(), FRONT_M),
                          g.thin(np.flatnonzero(g.step < 0), FREE_M))

    def nucleating(self, g: Growth, Pm: sp.csr_matrix) -> B:
        s = g.attached()
        near = (np.asarray(g.adj @ s.astype(np.int8)).ravel() > 0) | s
        return np.asarray(Pm @ near.astype(np.float64)).ravel() == 0

    def charge(self, g: Growth, Pm: sp.csr_matrix) -> F64:
        return np.where(self.nucleating(g, Pm), self.island * g.P0, 0.0)

    def check(self, exact: int) -> None:
        if exact < 1:
            raise ValueError("free: an island is taken only on an exact solve (K >= 1): a lone "
                             "patch's first-order gain overstates it by its leak times its own "
                             "resistance to ground")


class Pick(Protocol):
    """One draw of a batch from the remaining patches' scores (all positive)."""

    @property
    def name(self) -> str: ...

    def draw(self, score: F64) -> int: ...


@dataclass(frozen=True)
class Greedy:
    """The best patch: dielectric breakdown's eta -> infinity."""
    name = "greedy"

    def draw(self, score: F64) -> int:
        return int(np.argmax(score))


class Dbm:
    """Dielectric breakdown's dial: a patch with probability proportional to score^eta, from the
    seed's generator (its state the instance's)."""

    def __init__(self, *, eta: float, seed: int) -> None:
        if not eta > 0:
            raise ValueError(f"eta {eta}: positive")
        self.eta, self.seed = eta, seed
        self.rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return f"e{self.eta:g}s{self.seed}"

    def draw(self, score: F64) -> int:
        w = (score / score.max()) ** self.eta
        return int(self.rng.choice(len(w), p=w / w.sum()))


@dataclass(frozen=True, kw_only=True)
class Cost:
    """A move's cost in P: nu P0 per unit of D (its budget's multiplier, L_P per unit D) and
    lambda = price P0 / road_simp.SHEET_M2 per m^2 of conductance added (road_simp's sheet price).
    A price of 0 leaves every patch on open ground free (dD 0), so the front floods the attached
    open ground before it spends any D: refused."""
    nu: float
    price: float

    def __post_init__(self) -> None:
        if not (self.nu > 0 and self.price > 0):
            raise ValueError(f"nu {self.nu}, price {self.price}: both positive (a price of 0 "
                             "makes open ground free: the front floods it)")

    @property
    def name(self) -> str:
        return f"n{self.nu:g}s{self.price:g}"

    def of(self, dD: F64, dC: F64, *, P0: float) -> F64:
        return self.nu * P0 * dD + self.price * P0 / rs.SHEET_M2 * dC


def rank_one(g: float, du: float, R: float) -> float:
    """The drop in P from a conductance g joined between two cells whose potentials differ by du
    and whose effective resistance is R: g du^2 / (1 + g R) (Sherman-Morrison)."""
    return g * du * du / (1.0 + g * R)


class Bridge(NamedTuple):
    """A bridge offered past the screen: its ends (the front site a, the attached triangle b),
    length, corridor (new triangles), dD and dC, the R 0 bound, the rank-one estimate, the exact
    drop (nan if not rescored), its cost, and whether it was built."""
    a: int
    b: int
    L: float
    tris: I64
    dD: float
    dC: float
    bound: float
    sm: float
    exact: float
    cost: float
    built: bool


class Bridges(Protocol):
    @property
    def name(self) -> str: ...

    def offer(self, g: Growth, st: gcl.State, u: F64, *, P: float, bar: float,
              cost: Cost) -> list[Bridge]: ...

    def check(self, contrast: Contrast) -> None: ...


@dataclass(frozen=True)
class NoBridges:
    name = "B0"

    def check(self, contrast: Contrast) -> None:
        return None

    def offer(self, g: Growth, st: gcl.State, u: F64, *, P: float, bar: float,
              cost: Cost) -> list[Bridge]:
        return []


@dataclass(frozen=True, kw_only=True)
class Straight:
    """Straight road-width bridges between two points of the attached set's open edge within
    `reach` m, across a gap (GAP_SHARE of the segment off the attached set) (module docstring):
    `screen` estimated by Sherman-Morrison, the best `top` past the bar rescored."""
    reach: float
    screen: int
    top: int

    def __post_init__(self) -> None:
        if not (self.reach > ROAD_W and self.screen >= self.top >= 1):
            raise ValueError(f"{self}: reach past a road's width, screen >= top >= 1")

    @property
    def name(self) -> str:
        return f"B{self.reach:g}m{self.screen}k{self.top}"

    def check(self, contrast: Contrast) -> None:
        if contrast.grounded:
            raise ValueError("bridges between grounded roads: every road sits at u 0, a bridge "
                             "gains nothing (brief): B0")

    def offer(self, g: Growth, st: gcl.State, u: F64, *, P: float, bar: float,
              cost: Cost) -> list[Bridge]:
        if not np.isfinite(bar) or bar <= 0:
            return []
        # both ends on the road: open attached triangles beside the unattached ground (a front
        # triangle may lie inside a building at EPS, where u is no road's: rev 1 read it there
        # and its estimates ran 30x the exact gain)
        E = g.thin(g.attached_open_edge(), BRIDGE_FRONT_M)
        if len(E) < 2:
            return []
        cen = g.cen
        nb = cKDTree(cen[E]).query_ball_point(cen[E], r=self.reach)
        ia = np.repeat(np.arange(len(E)), [len(x) for x in nb])
        ib = np.concatenate([np.asarray(x, dtype=np.int64) for x in nb])
        keep = ia < ib
        a, b = E[ia[keep]], E[ib[keep]]
        L = np.hypot(*(cen[a] - cen[b]).T)
        ok = L > ROAD_W
        a, b, L = a[ok], b[ok], L[ok]
        if not len(a):
            return []
        ok = g.gap_share(a, b) >= GAP_SHARE        # across a gap, not along a road
        a, b, L = a[ok], b[ok], L[ok]
        if not len(a):
            return []
        cell = g.m.c.cell
        du = u[cell[a]] - u[cell[b]]
        gab = g.sigma * (g.kc - 1.0) * ROAD_W / L
        bound = gab * du * du
        rough = bound / cost.of(np.zeros(len(L)), (g.kc - 1.0) * ROAD_W * L,
                                P0=g.P0)
        # the best pair per end, then the best pairs overall
        o = np.lexsort((-rough, a))
        first = np.ones(len(o), dtype=bool)
        first[1:] = a[o][1:] != a[o][:-1]
        o = o[first]
        o = o[np.argsort(-rough[o])][:SCREEN_PAIRS * self.screen]
        cands = []
        for i in o:
            tris = g.corridor(cen[a[i]], cen[b[i]])
            if not len(tris):
                continue
            dD, dC = g.costs_of(tris)
            c = float(cost.of(np.array([dD]), np.array([dC]), P0=g.P0)[0])
            cands.append((float(bound[i] / c), int(i), tris, dD, dC, c))
        cands.sort(key=lambda t: -t[0])
        out = []
        for _, i, tris, dD, dC, c in cands[:self.screen]:
            R = g.resistance(st, int(cell[a[i]]), int(cell[b[i]]))
            sm = rank_one(float(gab[i]), float(du[i]), R)
            out.append(Bridge(a=int(a[i]), b=int(b[i]), L=float(L[i]), tris=tris, dD=dD, dC=dC,
                              bound=float(bound[i]), sm=sm, exact=np.nan, cost=c, built=False))
        out.sort(key=lambda br: -br.sm / br.cost)
        done = []
        n = 0
        for br in out:
            if n < self.top and br.sm / br.cost >= bar:
                n += 1
                ex = P - g.P_of(np.union1d(g.grown(), br.tris))
                br = br._replace(exact=ex, built=ex / br.cost >= bar)
            done.append(br)
        return done


@dataclass(frozen=True, kw_only=True)
class Plan:
    """A run's settings, all in its name (module docstring)."""
    contrast: Contrast
    connectivity: Connectivity
    pick: Pick
    batch: int                 # patches a step
    sep: float                 # m: a batch's sites apart
    cost: Cost
    bridges: Bridges
    exact: int                 # patches rescored exactly a step
    D: float                   # the budget: growth stops once D reaches it

    def __post_init__(self) -> None:
        if not (self.batch >= 1 and self.sep >= 0 and self.exact >= 0 and self.D > 0):
            raise ValueError(f"{self}: a batch, sep >= 0, exact >= 0 and a budget")
        self.bridges.check(self.contrast)
        self.connectivity.check(self.exact)

    @property
    def name(self) -> str:
        return (f"gr.{self.contrast.name}.{self.connectivity.name}.{self.pick.name}.N{self.batch}"
                f".sep{self.sep:g}.{self.cost.name}.{self.bridges.name}.K{self.exact}"
                f".D{self.D:g}.v{REV}")

    @staticmethod
    def of(spec: str) -> Plan:
        m = re.fullmatch(r"gr\.(k[0-9.e+]+|gl[0-9.]+Croad)\.(att|free[0-9.e-]+)"
                         r"\.(greedy|e[0-9.]+s[0-9]+)"
                         r"\.N([0-9]+)\.sep([0-9.]+)\.n([0-9.e-]+)s([0-9.e-]+)"
                         r"\.(B0|B[0-9.]+m[0-9]+k[0-9]+)\.K([0-9]+)\.D([0-9.]+)", spec)
        if m is None:
            raise SystemExit(f"{spec}: a plan is gr.<k<kappa>|gl<h>Croad>.<att|free<island>>."
                             "<greedy|"
                             "e<eta>s<seed>>.N<batch>.sep<m>.n<nu>s<price>.<B0|B<reach>m<screen>"
                             "k<top>>.K<exact>.D<D>")
        contrast: Contrast = (GroundedLeak(h=float(m[1][2:-5])) if m[1].startswith("gl")
                              else Road(kappa=float(m[1][1:])))
        conn: Connectivity = (Attached() if m[2] == "att" else Free(island=float(m[2][4:])))
        if m[3] == "greedy":
            pick: Pick = Greedy()
        else:
            e = re.fullmatch(r"e([0-9.]+)s([0-9]+)", m[3])
            assert e is not None
            pick = Dbm(eta=float(e[1]), seed=int(e[2]))
        if m[8] == "B0":
            bridges: Bridges = NoBridges()
        else:
            bb = re.fullmatch(r"B([0-9.]+)m([0-9]+)k([0-9]+)", m[8])
            assert bb is not None
            bridges = Straight(reach=float(bb[1]), screen=int(bb[2]), top=int(bb[3]))
        return Plan(contrast=contrast, connectivity=conn, pick=pick, batch=int(m[4]),
                    sep=float(m[5]), cost=Cost(nu=float(m[6]), price=float(m[7])),
                    bridges=bridges, exact=int(m[9]), D=float(m[10]))


# ---------------------------------------------------------------- the growing region


class Growth:
    """The block's eps world (road_simp's: every footprint at EPS) and the region grown on it:
    per triangle the step it was grown in (`step`, -1 never)."""

    def __init__(self, m: gcl.Model, *, contrast: Contrast) -> None:
        if len(m.fill.e):
            raise ValueError("a width form that fills gaps: a grown triangle would not open them")
        self.m = m
        self.contrast = contrast
        fm = m.fx.fm
        self.nt = fm.nt
        self.cen = m.cen_
        self.area = m.tri_area
        self.sigma = float(m.fs.sigma)
        self.k = np.full(m.carving.n, EPS)
        self.kt0 = np.asarray(m.terms.value(self.k), dtype=np.float64)
        self.bw = np.asarray(m.b_ways, dtype=np.float64)
        self.cb = contrast.c_boost(self.bw)                          # a grown triangle's, in C
        self.kc = float(contrast.c_boost(np.ones(1))[0])             # a road's contrast
        self.dC_t = self.area * (self.cb - self.kt0 * self.bw)      # a triangle grown: its dC
        self.leak0 = m.sys.graph.leak
        cov = fm.tcov.T.tocsr().astype(np.float64) @ sp.diags(self.area)
        self.cover = (sp.diags(1.0 / np.maximum(np.asarray(cov.sum(axis=1)).ravel(), 1e-300))
                      @ cov).tocsr()
        self.coverT = self.cover.T.tocsr()
        self.w = m.pop / m.pop.sum()
        self.P0 = float(m.base.P)
        self.sheet0 = m.sheet(m.factors(np.zeros(m.n)))             # the exact baseline's
        n = fm.nt
        self.adj = sp.csr_matrix((np.ones(2 * len(fm.pa), dtype=np.int8),
                                  (np.concatenate([fm.pa, fm.pb]),
                                   np.concatenate([fm.pb, fm.pa]))), shape=(n, n))
        self.pa, self.pb = fm.pa, fm.pb
        # the attached base: the street faces' triangles and the paved ways joined to them
        street = np.zeros(n, dtype=bool)
        street[fm.fa[m.c.pf]] = True
        way = (self.bw > 1.0) | street
        sub = self.adj[way][:, way]
        comp = connected_components(sub, directed=False)[1]
        widx = np.flatnonzero(way)
        keep = np.isin(comp, np.unique(comp[street[widx]]))
        self.base = np.zeros(n, dtype=bool)
        self.base[widx[keep]] = True
        self.tree = cKDTree(self.cen)
        self.step = np.full(n, -1, dtype=np.int32)

    # -- the region
    def grown(self) -> I64:
        return np.flatnonzero(self.step >= 0)

    def attached(self) -> B:
        return self.base | (self.step >= 0)

    def front(self) -> I64:
        """The ungrown triangles beside the attached set (not in its base)."""
        s = self.attached()
        near = np.asarray(self.adj @ s.astype(np.int8)).ravel() > 0
        return np.flatnonzero(near & ~s)

    def attached_edge(self) -> I64:
        """The attached set's triangles beside an unattached one."""
        s = self.attached()
        near = np.asarray(self.adj @ (~s).astype(np.int8)).ravel() > 0
        return np.flatnonzero(near & s)

    def attached_open_edge(self) -> I64:
        """attached_edge's triangles on open ground (kt0 >= OPEN_KT)."""
        e = self.attached_edge()
        return e[self.kt0[e] >= OPEN_KT]

    def gap_share(self, a: I64, b: I64) -> F64:
        """Per pair of triangles the share of GAP_SAMPLES points strictly inside the segment
        between their centroids whose nearest triangle is not attached."""
        t = np.arange(1, GAP_SAMPLES + 1) / (GAP_SAMPLES + 1)
        pts = self.cen[a][:, None, :] + t[None, :, None] * (self.cen[b] - self.cen[a])[:, None, :]
        _, near = self.tree.query(pts.reshape(-1, 2), k=1)
        off = ~self.attached()[near].reshape(len(a), GAP_SAMPLES)
        return np.asarray(off.mean(axis=1), dtype=np.float64)

    def thin(self, t: I64, cell_m: float) -> I64:
        """One of `t` per square `cell_m` wide (the first by index)."""
        if not len(t):
            return t
        key = np.floor(self.cen[t] / cell_m).astype(np.int64)
        _, first = np.unique(key[:, 0] * 10_000_000 + key[:, 1], return_index=True)
        return t[np.sort(first)]

    def islands(self) -> tuple[int, float]:
        """(the grown components joined to no base triangle, their area in m^2)."""
        s = self.attached()
        idx = np.flatnonzero(s)
        k, lab = connected_components(self.adj[idx][:, idx], directed=False)
        based = np.zeros(k, dtype=bool)
        based[lab[self.base[idx]]] = True
        isl = ~based[lab]
        return int(k - based.sum()), float(self.area[idx[isl]].sum())

    def faces(self) -> int:
        """The region's faces of FACE_M2 and more: components of the ungrown triangles."""
        off = self.step < 0
        e = off[self.pa] & off[self.pb]
        n = self.nt
        g = sp.coo_matrix((np.ones(int(e.sum()), dtype=np.int8), (self.pa[e], self.pb[e])),
                          shape=(n, n))
        k, lab = connected_components(g, directed=False)
        a = np.bincount(lab[off], weights=self.area[off], minlength=k)
        return int((a >= FACE_M2).sum())

    # -- the state
    def state(self, on: B) -> gcl.State:
        """The eps world with the triangles `on` grown, its operator assembled and factored."""
        m = self.m
        kt, b, leak = self.contrast.world(self, on)
        sys = (m.sys if leak is None else
               m.sys.with_graph(m.sys.graph.with_leak(self.leak0 + leak, live0=m.leak_site.live)))
        fm = m.exits.edge.mesh(gx.boosted(m.fx.fm, b))
        kp, binds = gcl._passage_factors(fm, kt, self.k, m.fill)
        r = sys.graph.reach(fm, kt, kp)
        data = sys.graph.values(r)
        p = sys.graph.pattern
        A = sp.csr_matrix((data, p.indices, p.indptr), shape=(p.n, p.n))
        f = sys.planned.factor(data, r.unknown)
        return gcl.State(sys=sys, prefix=gcl._NO_ROADS, b=b, fm=fm, k=self.k, kt=kt, kp=kp,
                         binds=binds, reach=r, A=A, factor=f)

    def solve(self, st: gcl.State) -> tuple[F64, float]:
        u = self.m.solve(st, self.m.pop)
        return u, float(self.m._rhs(st, self.m.pop) @ u)

    def P_of(self, tris: I64) -> float:
        on = np.zeros(self.nt, dtype=bool)
        on[tris] = True
        return self.solve(self.state(on))[1]

    def grad(self, st: gcl.State, u: F64) -> tuple[F64, F64]:
        """Per triangle (dP/dkt, dP/db) at the state (road_simp._grad)."""
        c, fm = st.sys.cells, st.fm
        gk, gb = rs._grad(c.pe, c.ea, c.eb, fm.pa, fm.pb, fm.T, st.kt, st.kp, st.b, c.pf, c.fc,
                          fm.fa, fm.Tf, u, fm.nt)
        return np.asarray(gk), np.asarray(gb)

    def resistance(self, st: gcl.State, i: int, j: int) -> float:
        """The effective resistance between cells i and j (a unit dipole's solve)."""
        assert st.factor is not None
        e = np.zeros(self.m.c.n)
        e[i], e[j] = 1.0, -1.0
        x = st.factor.solve(e * st.reach.unknown)
        return float(x[i] - x[j])

    # -- the measures
    def X(self) -> F64:
        return np.asarray(self.cover @ (self.step >= 0).astype(np.float64))

    def D(self) -> float:
        return float(self.w @ common.MEASURE.ramp(self.X()))

    def C(self) -> float:
        """The conductance the eps world adds (m^2): its sheet less the exact baseline's, a grown
        triangle's counted at the contrast's c_boost (a grounded corridor's as a road's)."""
        on = self.step >= 0
        return float(self.area @ (np.where(on, 1.0, self.kt0) * np.where(on, self.cb, self.bw))
                     - self.sheet0)

    def costs(self, Pm: sp.csr_matrix) -> tuple[F64, F64]:
        """Per row of Pm (a 0/1 matrix of new triangles): (dD, dC)."""
        X = self.X()
        ramp = common.MEASURE.ramp
        dX = (self.cover @ Pm.T).tocoo()            # buildings x rows
        d = self.w[dX.row] * (ramp(X[dX.row] + dX.data) - ramp(X[dX.row]))
        dD = np.bincount(dX.col, weights=d, minlength=Pm.shape[0])
        return dD, np.asarray(Pm @ self.dC_t)

    def costs_of(self, tris: I64) -> tuple[float, float]:
        Pm = sp.csr_matrix((np.ones(len(tris)), (np.zeros(len(tris), dtype=np.int64), tris)),
                           shape=(1, self.nt))
        dD, dC = self.costs(Pm)
        return float(dD[0]), float(dC[0])

    # -- the moves
    def patches(self, sites: I64) -> sp.csr_matrix:
        """Per site its patch's ungrown triangles (rows: sites)."""
        lists = self.tree.query_ball_point(self.cen[sites], r=PATCH_R)
        rows = np.repeat(np.arange(len(sites)), [len(x) for x in lists])
        cols = np.concatenate([np.asarray(x, dtype=np.int64) for x in lists])
        keep = self.step[cols] < 0
        return sp.csr_matrix((np.ones(int(keep.sum())), (rows[keep], cols[keep])),
                             shape=(len(sites), self.nt))

    def corridor(self, a: F64, b: F64) -> I64:
        """The ungrown triangles within PATCH_R of the segment a -> b."""
        L = float(np.hypot(*(b - a)))
        k = max(2, int(np.ceil(L / BRIDGE_STEP_M)) + 1)
        pts = a + np.linspace(0.0, 1.0, k)[:, None] * (b - a)
        lists = self.tree.query_ball_point(pts, r=PATCH_R)
        t = np.unique(np.concatenate([np.asarray(x, dtype=np.int64) for x in lists]))
        return t[self.step[t] < 0]


# ---------------------------------------------------------------- the run


class Row(NamedTuple):
    """A state's row: the steps taken, P, L_P (1 - P / the exact baseline's P0), D, C (m^2), the
    region's area (m^2), its faces after the step's patches and after its bridges (the state's),
    the patches and bridges the step built, its realized gain per cost (the next step's bar),
    seconds."""
    step: int
    P: float
    L_P: float
    D: float
    C: float
    area: float
    faces_patches: int
    faces: int
    n_patches: int
    n_bridges: int
    islands: int
    island_m2: float
    bar: float
    t: float


def grow(g: Growth, plan: Plan, *, out: Path, log: Any) -> None:
    """The plan's growth until D reaches plan.D, every step written to `out`."""
    out.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    picks: list[dict[str, float]] = []
    offered: list[dict[str, float]] = []
    t0 = time.time()
    faces0 = g.faces()
    faces_p = faces0
    bar = np.nan
    last_cost = np.nan
    last_fo = np.nan
    n_p = n_b = 0
    P_prev = np.nan
    step = 0
    while True:
        st = g.state(g.step >= 0)
        u, P = g.solve(st)
        if np.isfinite(last_cost):
            bar = (P_prev - P) / last_cost
        D = g.D()
        faces = g.faces()
        isl, isl_m2 = g.islands()
        rows.append(Row(step=step, P=P, L_P=1.0 - P / g.P0, D=D, C=g.C(),
                        area=float(g.area[g.step >= 0].sum()), faces_patches=faces_p,
                        faces=faces, n_patches=n_p, n_bridges=n_b, islands=isl,
                        island_m2=isl_m2, bar=bar, t=time.time() - t0))
        r = rows[-1]
        log(f"state {step}: L_P {r.L_P:.4f}, D {D:.4f}, C {r.C:.0f} m^2, area {r.area:.0f} m^2, "
            f"loops {faces - faces0} (patches {faces_p - faces0}), built {n_p} + {n_b} bridges, "
            f"islands {isl} ({isl_m2:.0f} m^2), "
            f"bar {bar:.3g} ({r.t:.0f}s)")
        pd.DataFrame(rows).to_parquet(out / "steps.parquet")
        pd.DataFrame(picks).to_parquet(out / "picks.parquet")
        pd.DataFrame(offered).to_parquet(out / "bridges.parquet")
        pd.DataFrame({"step": g.step}).to_parquet(out / "tri_step.parquet")
        if D >= plan.D:
            break
        gain = plan.contrast.gains(g, st, u)
        sites = plan.connectivity.sites(g)
        Pm = g.patches(sites)
        new = np.asarray(Pm @ g.area)
        dD, dC = g.costs(Pm)
        fo = np.asarray(Pm @ gain)
        cost = plan.cost.of(dD, dC, P0=g.P0) + plan.connectivity.charge(g, Pm)
        score = np.where(new >= MIN_NEW_M2, fo / cost, 0.0)
        exact = np.full(len(sites), np.nan)
        isl = plan.connectivity.nucleating(g, Pm)
        # the top `exact` rescored: with islands, the islands' (and only theirs): a lone patch's
        # first-order gain overstates it by its leak times its own resistance to ground, far more
        # than a front patch's, so an island is taken only on an exact solve, its gain put on the
        # batch's first-order scale (times the last batch's first-order over realized drop)
        pool = np.flatnonzero(isl if isl.any() else score > 0)
        top = pool[np.argsort(-score[pool])[:plan.exact]]
        infl = last_fo / (P_prev - P) if np.isfinite(last_fo) and P_prev > P else np.nan
        score[isl] = 0.0
        for i in top:
            exact[i] = P - g.P_of(np.union1d(g.grown(), Pm[i].indices))
            score[i] = exact[i] / cost[i] * (infl if isl[i] else 1.0)
        score = np.nan_to_num(score, nan=0.0)
        # the batch
        live = score > 0
        tree = cKDTree(g.cen[sites])
        chosen: list[int] = []
        while len(chosen) < plan.batch and live.any():
            idx = np.flatnonzero(live)
            i = int(idx[plan.pick.draw(score[idx])])
            chosen.append(i)
            live[tree.query_ball_point(g.cen[sites[i]], r=plan.sep)] = False
            live[i] = False
        # the bridges, at the state before the batch
        brs = plan.bridges.offer(g, st, u, P=P, bar=bar, cost=plan.cost)
        for i in chosen:
            g.step[Pm[i].indices] = step
            picks.append(dict(step=step, site=int(sites[i]), x=float(g.cen[sites[i], 0]),
                              y=float(g.cen[sites[i], 1]), score=float(score[i]),
                              first_order=float(fo[i]), exact=float(exact[i]), dD=float(dD[i]),
                              dC=float(dC[i]), cost=float(cost[i]), new_m2=float(new[i]),
                              island=bool(isl[i])))
        faces_p = g.faces()
        built = [br for br in brs if br.built]
        for br in built:
            g.step[br.tris[g.step[br.tris] < 0]] = step
        for br in brs:
            offered.append(dict(step=step, ax=float(g.cen[br.a, 0]), ay=float(g.cen[br.a, 1]),
                                bx=float(g.cen[br.b, 0]), by=float(g.cen[br.b, 1]), L=br.L,
                                n_tris=len(br.tris), dD=br.dD, dC=br.dC, bound=br.bound,
                                sm=br.sm, exact=br.exact, cost=br.cost, built=br.built))
        last_cost = float(cost[chosen].sum()) + sum(br.cost for br in built)
        last_fo = float(fo[chosen].sum())
        P_prev = P
        n_p, n_b = len(chosen), len(built)
        if not chosen and not built:
            log("no patch or bridge left: stopped short of the budget")
            break
        step += 1


def growth_of(b: Any, metric: gcl.ComplexMetric, *, contrast: Contrast) -> Growth:
    return Growth(metric.clearing(b, demand=common.PopulationDemand()), contrast=contrast)


def run(bid: str, plan: Plan) -> None:
    metric = metrics.metric_of(metrics.RUNTIME, device="cpu")
    out = OUT / bid / plan.name
    t0 = time.time()
    [b] = common.build_blocks([bid])
    g = growth_of(b, metric, contrast=plan.contrast)
    print(f"{bid} {plan.name}: {g.nt} triangles, P0 {g.P0:.6e}, base {int(g.base.sum())} "
          f"triangles ({time.time() - t0:.0f}s)", flush=True)
    grow(g, plan, out=out, log=lambda s: print(s, flush=True))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


def main(argv: list[str]) -> None:
    if len(argv) == 3 and argv[0] == "run":
        run(argv[1], Plan.of(argv[2]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
