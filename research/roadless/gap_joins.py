"""Joins: moves that clear several buildings together, offered to the polish as its last tiers
(search.Swaps, relax.Plan.joins; NOTES, "Walled-in homes, round 2: T, U and W, each with its best
optimizer"). A crossing that needs several buildings opened together gains almost nothing from
any one of them, so SIMP's gradient and the polish's single and paired moves miss it (the uniform
rule's 32 blocks where its optimizer ends below T's clearing by its own measure); a join offers
the crossing whole, scored exactly, kept only if it pays at the budget (search._joined gives back
the cleared buildings worth least to fit it).

Two sources, each a search.Joiner built per block where the plan meets its model (relax.one):

CROSSINGS (`CrossingJoiner`, relax.Crossings, `jF<width>`): free first's crossings
(gap_complex.free_first through gap_clear.Model.freed, computed as free first computes them, at
baseline within the budget), each offered as one move of every building its path needs from the
baseline (gap_complex.Crossing.path), never held; the first `width` not yet cleared, in free
first's order, a round.

CHAINS (`ChainJoiner`, relax.Chains, `jV<width>s<sources>`; path moves): from one exact solve at
the polish's clearing, chains of standing buildings whose clearing would join two regions at
different potentials. A region is a component of the state's live cells (the street's, or a
sealed pocket or closed cell live through its leak alone); a building's contacts are the live
cells its clearing alone would join (open cells across one of its passages, and closed cells
inside its grown footprint that a leak keeps live: HomeDemand's), per region at the highest and
lowest potential there. Adding a conductance between two points at potentials u_a, u_b changes P
by -(u_a - u_b)^2 / (R_new + R_ab), R_ab the network's resistance between them: here R_new the
chain's (each building's footprint a sheet of its own aspect, L^2 / (16 A) squares: one for a
square) and R_ab each end's region's own (a region joined to the street none: a well-grounded
sheet; a sealed region the resistance of its leak, 1 / its conductance to ground, which a
pocket's escape time is set by). The search is a shortest path over the buildings (their grown
footprints' adjacency on the fixed mesh: a passage between two buildings' own triangles, or a
triangle in both their grown footprints), each building's resistance its node's weight, from the
`sources` buildings whose contacts reach the highest potentials; a chain's estimate sums over the
source's regions the change each would make joined to the target's lowest-potential region. The
best `width` chains a round, distinct, each offered as one move. Free first's crossings are the
case where one side is a sealed pocket (the walled-in homes' report says how often a chain at
baseline is one).

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/gap_joins.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import shapely
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components, dijkstra

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import search  # noqa: E402

# ------------------------------------------------------------------ crossings

class CrossingJoiner:
    """Free first's crossings at baseline within `budget` (gap_clear.Model.freed from `held`),
    each one move of its path's buildings; the first `width` not yet cleared a round."""

    def __init__(self, m: gcl.Model, *, budget: float, width: int, held: NDArray[np.bool_]):
        self.width = width
        self.held_ = held.copy()
        self.crossings: list[gc.Crossing] = m.freed(budget=budget, held=held).crossings

    @property
    def held(self) -> NDArray[np.bool_]:
        return self.held_

    def joins(self, s: search.SearchState) -> list[search.Join]:
        out: list[search.Join] = []
        for cr in self.crossings:
            if not s.c.removed[cr.path].all():
                out.append(search.Join(add=tuple(cr.path), est=0.0))
            if len(out) == self.width:
                break
        return out


# ------------------------------------------------------------------ chains

class Chain(NamedTuple):
    """A chain of standing buildings (`buildings`, source first) and its estimated change in P
    (`gain`, positive: P falls)."""
    buildings: tuple[int, ...]
    gain: float


class Contacts(NamedTuple):
    """Per (building, region) the live cells the building's clearing alone would join, in that
    region: `b`, `reg`, their potentials' largest `hi` and smallest `lo`."""
    b: NDArray[np.int64]
    reg: NDArray[np.int64]
    hi: NDArray[np.float64]
    lo: NDArray[np.float64]


def _regions(m: gcl.Model, st: gcl.State) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Per cell its region (a component of the state's live cells over its conducting pairs; -1
    off them) and per region its own resistance: 0 joined to the street (a ground cell or a street
    face in it), else 1 / its leak's conductance to ground."""
    g, r = m.sys.graph, st.reach
    p = g.pattern
    rows = np.repeat(np.arange(p.n), np.diff(p.indptr))
    cols = np.asarray(p.indices, dtype=np.int64)
    ep = np.asarray(g.epair, dtype=np.int64)
    on = ep >= 0
    on[on] = r.W[ep[on]] > 0.0
    on &= r.live[rows] & r.live[cols]
    A = sp.csr_matrix((np.ones(int(on.sum())), (rows[on], cols[on])), shape=(p.n, p.n))
    _k, lab = connected_components(A, directed=False)
    lab = np.where(r.live, lab, -1).astype(np.int64)
    k = int(lab.max()) + 1
    live = np.flatnonzero(r.live)
    grounded = np.bincount(lab[live], weights=(r.ground | (r.G > 0.0))[live].astype(np.float64),
                           minlength=k) > 0
    leak = np.bincount(lab[live], weights=g.leak[live], minlength=k)
    with np.errstate(divide="ignore"):
        R = np.where(grounded, 0.0, 1.0 / leak)
    if not np.isfinite(R[lab[live]]).all():
        raise RuntimeError("a live region joined to neither the street nor a leak")
    return lab, R


def _owners(m: gcl.Model, standing: NDArray[np.bool_]) -> tuple[NDArray[np.int64],
                                                                  sp.csr_matrix]:
    """Per triangle of the fixed mesh its one standing owner (the building whose grown footprint
    holds it, alone among those standing; -1 none or several) and the standing owners' matrix
    (triangles x buildings)."""
    zone = m.fx.fm.tzone.tocsr()
    if zone.shape[1] != m.n:
        raise ValueError("chains run on the plain mesh: a building per atom")
    Z = (zone @ sp.diags(standing.astype(np.float64))).tocsr()
    Z.eliminate_zeros()
    nown = np.diff(Z.indptr)
    one = np.full(Z.shape[0], -1, dtype=np.int64)
    k = np.flatnonzero(nown == 1)
    one[k] = Z.indices[Z.indptr[k]]
    return one, Z


def contacts_of(m: gcl.Model, st: gcl.State, u: NDArray[np.float64], lab: NDArray[np.int64],
                one: NDArray[np.int64], Z: sp.csr_matrix) -> Contacts:
    """Each standing building's contacts (Contacts): across a static passage from one of its own
    triangles (`one`) to an open one (no standing owner) in a live cell, and its own triangles in
    a live cell (closed ground a leak keeps live: HomeDemand's)."""
    fm, c = m.fx.fm, m.sys.cells
    nown = np.diff(Z.indptr)
    ta, tb = fm.pa[c.pe], fm.pb[c.pe]
    bs, cs = [], []
    for x, y in ((ta, tb), (tb, ta)):
        k = (one[x] >= 0) & (nown[y] == 0) & st.reach.live[c.cell[y]]
        bs.append(one[x][k])
        cs.append(c.cell[y][k])
    k = np.flatnonzero((one >= 0) & st.reach.live[c.cell])
    bs.append(one[k])
    cs.append(c.cell[k])
    b, cell = np.concatenate(bs), np.concatenate(cs)
    reg = lab[cell]
    key = b * (int(lab.max()) + 1) + reg
    uk, inv = np.unique(key, return_inverse=True)
    hi = np.full(len(uk), -np.inf)
    lo = np.full(len(uk), np.inf)
    np.maximum.at(hi, inv, u[cell])
    np.minimum.at(lo, inv, u[cell])
    nreg = int(lab.max()) + 1
    return Contacts(b=(uk // nreg).astype(np.int64), reg=(uk % nreg).astype(np.int64), hi=hi,
                    lo=lo)


def adjacency(m: gcl.Model, one: NDArray[np.int64], Z: sp.csr_matrix) -> sp.csr_matrix:
    """The standing buildings' adjacency (n x n, 0/1): a static passage between two buildings' own
    triangles, or a triangle in exactly two standing buildings' grown footprints."""
    fm, c = m.fx.fm, m.sys.cells
    ta, tb = one[fm.pa[c.pe]], one[fm.pb[c.pe]]
    k = (ta >= 0) & (tb >= 0) & (ta != tb)
    a, b = [ta[k]], [tb[k]]
    two = np.flatnonzero(np.diff(Z.indptr) == 2)
    if len(two):
        a.append(Z.indices[Z.indptr[two]])
        b.append(Z.indices[Z.indptr[two] + 1])
    a_, b_ = np.concatenate(a), np.concatenate(b)
    A = sp.csr_matrix((np.ones(2 * len(a_)), (np.concatenate([a_, b_]), np.concatenate([b_, a_]))),
                      shape=(m.n, m.n))
    A.data[:] = 1.0
    return A


def squares(m: gcl.Model) -> NDArray[np.float64]:
    """Per building its footprint's resistance across in sheet squares: L^2 / (16 A), L its
    outline's length, A its area (1 for a square, more for a long or ragged footprint)."""
    polys = m.polys
    return np.asarray(shapely.length(polys) ** 2 / (16.0 * np.maximum(shapely.area(polys),
                                                                       1e-12)))


def chains(m: gcl.Model, removed: NDArray[np.bool_], *, sources: int, width: int
           ) -> list[Chain]:
    """The best `width` distinct chains at the clearing `removed` (module docstring), from the
    `sources` buildings whose contacts reach the highest potentials; one exact solve (P's)."""
    st = m.state(m.k_of(removed))
    u = m.solve(st, m.inj)
    lab, Rreg = _regions(m, st)
    standing = ~removed
    one, Z = _owners(m, standing)
    con = contacts_of(m, st, u, lab, one, Z)
    if len(con.b) == 0:
        return []
    n = m.n
    R = squares(m) / float(m.fx.fm.sigma)
    # per building its lowest contact (potential, region) and the largest one, as sources
    order = np.lexsort((con.lo, con.b))
    first = np.r_[True, con.b[order][1:] != con.b[order][:-1]]
    lo_u = np.full(n, np.inf)
    lo_reg = np.full(n, -1, dtype=np.int64)
    lo_u[con.b[order][first]] = con.lo[order][first]
    lo_reg[con.b[order][first]] = con.reg[order][first]
    lo_R = np.where(lo_reg >= 0, Rreg[np.maximum(lo_reg, 0)], np.inf)
    hi_u = np.full(n, -np.inf)
    np.maximum.at(hi_u, con.b, con.hi)
    cand = np.flatnonzero(np.isfinite(hi_u) & standing)
    src = cand[np.argsort(-hi_u[cand], kind="stable")[:sources]]
    A = adjacency(m, one, Z)
    A = (sp.diags(standing.astype(np.float64)) @ A @ sp.diags(R * standing)).tocsr()
    A.eliminate_zeros()
    dist, pred = dijkstra(A, directed=True, indices=src, return_predecessors=True)
    by_b: dict[int, list[int]] = {}
    for i, b in enumerate(con.b):
        by_b.setdefault(int(b), []).append(i)
    found: list[tuple[float, int, int]] = []
    reach = np.isfinite(lo_u)
    for i, s in enumerate(src):
        d = R[s] + dist[i]
        on = reach & np.isfinite(d)
        gain = np.zeros(n)
        for j in by_b[int(s)]:
            k, uk = int(con.reg[j]), float(con.hi[j])
            t = np.flatnonzero(on & (lo_reg != k))
            gain[t] += np.maximum(uk - lo_u[t], 0.0) ** 2 / (d[t] + Rreg[k] + lo_R[t])
        for t in np.argsort(-gain, kind="stable")[:width]:
            if gain[t] > 0.0:
                found.append((float(gain[t]), int(s), int(t)))
    found.sort(key=lambda f: -f[0])
    out: list[Chain] = []
    seen: set[tuple[int, ...]] = set()
    row = {int(s): i for i, s in enumerate(src)}
    for gain_t, s, t in found:
        path = [t]
        while path[-1] != s:
            path.append(int(pred[row[s], path[-1]]))
        key = tuple(sorted(path))
        if key in seen:
            continue
        seen.add(key)
        out.append(Chain(buildings=tuple(reversed(path)), gain=gain_t))
        if len(out) == width:
            break
    return out


class ChainJoiner:
    """Path moves (module docstring): the best `width` chains from the `sources` highest-potential
    buildings at the polish's clearing, each its estimated change in P (negated: lower first)."""

    def __init__(self, *, width: int, sources: int, held: NDArray[np.bool_]):
        self.width, self.sources = width, sources
        self.held_ = held.copy()

    @property
    def held(self) -> NDArray[np.bool_]:
        return self.held_

    def joins(self, s: search.SearchState) -> list[search.Join]:
        m = search.complex_model(s)
        return [search.Join(add=ch.buildings, est=-ch.gain)
                for ch in chains(m, m.removed.copy(), sources=self.sources, width=self.width)]
