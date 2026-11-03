"""Reblocking by ears: a partition of the block where every road is on a loop
(.superpowers/sdd/2026-10-27-ears/brief.md; owner 2026-10-27, a third arm of the growth pilot).

A network with no dead ends (every road on two faces, the street ring included) is a partition of
the block into sub-blocks, and every such network has an ear decomposition from the street ring:
each move adds an ear, a road leaving the network at one point and rejoining it at another, which
splits one face in two. Growing by ears keeps every state a valid reblocking; the D budget stops
it.

The world is growth.py's: the block's eps world, the grown roads grounded through the Graph's leak
(growth.GroundedLeak), their C charged as kappa-10 roads, the cost nu P0 dD + lambda dC
(growth.Cost), every step a prefix (tri_step). With the roads grounded, current reaching a road
leaves there, so the faces nearly decouple: P is about the sum of the faces' own P, each face its
cells with every cell outside it held at 0 (the plan's factorization restricted to them:
gap_factor's restriction of the pattern), exact up to the leak's finite conductance (check
`decouple`). An ear's exact gain is then its face's P less the face's P with the ear grown (its two
halves, one restricted solve), and its first order the sum over its corridor of growth's
first-order gains (dP per unit leak at a cell is -u_c^2; the opening's dP/dkt).

Candidate ears are a Strategy (Candidates): `paths` (Paths), cheapest paths on a PIX m raster of
the faces (a metre on open ground costs 1, through a building 1 + `bld`; grown ground closed)
from the faces' sinks (EarWorld.sinks: pixels on the street or a grounded road, not a block edge
with no exit nor a paved way inside the block, where an ear would dead-end), sampled every SRC_M
m, to any other sink pixel of the same face at least `ear_m` away in a straight line: per
source the target with the most first-order gain along the path per path cost (the gains summed
down the shortest-path tree), the best `pairs` sources' ears built as ROAD_W corridors
(EarWorld.corridor) and costed exactly (dD, dC); and `divides` (Divides,
owner 2026-10-28), the separatrices of the persistence-simplified Morse-Smale complex of u on each
face (the function `divides`, pure in its pixel inputs so other optimizers can seed from it; the
persistence threshold `tau` in the name; the harmonic-measure split where the strict basins
degenerate along the continuous street ring), near-duplicates dropped, across faces the `pairs`
with the most first-order gain per metre costed (rev 2; rev 1's most persistent starved the
large faces: Divides); the brief's
`ridge` is `divides` at its strictest. Each step: every candidate's first-order gain per cost; in
that order each must split its face (its faces of FACE_M2 and more rise by at least one), an ear
that does not refused and logged (the no-dead-end invariant; check `ends`), until `exact` split;
those rescored by the restricted solve, the best exact gain per cost built (greedy, one ear a
step). Every DEC_EVERY ears the decoupling gap (the global P less the faces' sum, over P) goes into
the state's row.

Where a path may end is a Strategy of `paths` (Ends): `loop` (Loop), the above; `spur<min>-<max>`
(Spur, owner 2026-11-03: Idea A, spurs in Shock Reblock), dead ends allowed, so spurs and loops
compete at one budget: a path may also end on the face's open ground (no building, no sink)
between min and max m from its source in a straight line, a spur (Ear.spur), its corridor joined
to the network at its start only and built without splitting its face. A spur lies in one face
and leaves its cells as they were, so its exact gain is the same restricted solve. `divides` cuts
run sink to sink by construction: their ears are loops, and their names carry no ends.

A term added to the escape objective of `paths` is a Strategy (Attract, owner 2026-11-03: Idea C,
detour-saving attraction): NoDetour, the objective alone; Detour(lam) (`.dt<lam>`), each ear
credited lam P0 S / S0, S its detour saving (`savings`: over pairs of homes, each at the road
point nearest it, the mean metres by which the trip between their road points along the network
gets shorter with the ear's path a link; the network the state's sinks, its road points one per
SRC_M square, `roads_of`), S0 the mean such trip on the street alone. A spur shortens no trip and
earns nothing; the ear that later closes it is credited the whole loop. An ear is scored
(gain + credit) / cost at every stage, its credit times the last step's first order over exact
gain where the first order stands in for the exact gain (build).

    E=research/roadless/ears.py                        # from the repo root, PYTHONPATH=.
    uv run python $E run <block> <plan>
    uv run python $E check decouple|first|ends|spur|detour|offer <block> [fault]
    uv run python $E check horseshoe [fault]

plan: ears.<paths.b<bld>.f<face_m2>.e<ear_m>.<loop|spur<min_m>-<max_m>>[.dt<lam>]|divides.p<tau>
.f<face_m2>.e<ear_m>>.gl<h>Croad.n<nu>s<price>.M<pairs>.K<exact>.D<D> (a face under face_m2 not
split, an ear's ends at least ear_m apart, a spur's end min_m to max_m from its start, `.dt<lam>`
Detour, none NoDetour: every run before it, whose names stand), its name adding the code's
revision (.v<REV>); rows in growth_rows/<block>/<name>/ in growth.py's layout (steps.parquet,
picks.parquet one row per ear with whether it is a spur and its detour saving and credit,
bridges.parquet empty, tri_step.parquet), so the growth runs' frames and plates read them.

Checks, each run both ways:
decouple: a state with a few ears on the block, the global P against the sum of its faces' P
  (one solve restricted to the ungrown triangles' cells: the faces, with the roads' cells held
  at 0, are its diagonal blocks), within TOL_DEC relative. Fault `weak-leak` (h 25 m: roads
  barely grounded, the faces coupled) fails it.
first: on that state, for the top ears (their open triangles only, kt0 >= OPEN_KT, so the
  opening's term is nil) the exact drop at a small leak (h FIRST_H: dP linear in it) against the
  first order (sum over the corridor of u_c^2 l_t), within TOL_FIRST relative. Fault `abs-u`
  (|u| in place of u^2) fails it.
ends: the top candidate ears each split their face (its faces of FACE_M2 and more rise). Fault
  `spur` (an ear's last third dropped: a dead end) fails it.
spur: the spur check plan's top SPUR_EARS candidate spurs each a corridor in one piece, touching
  the network, that splits nothing (its end on open ground). Fault `both-ends` (a spur's
  corridor extended at its end too, as a loop's is) fails it.
detour: after DETOUR_EARS loop ears, the detour plan's costed ears' savings (Ear.saving) and
  DETOUR_TARGETS links from the first one's source at once (`savings`' many targets) against all
  pairs' distances again with the link an edge of the network's graph, within TOL_DETOUR of the
  largest. Fault `one-way` (the trips counted from a's side only: savings' factor 2 dropped) fails
  it.
horseshoe: synthetic horseshoes (no block; Experiment 1 of the owner's doc): the closing link's
  saving rises strictly with the arms at a fixed gap, and its saving per metre of link strictly as
  the gap closes at fixed arms. Fault `crow` (straight-line distances for the road distances: no
  link shortens a trip) fails it.
offer: after OFFER_EARS ears of the divides check plan (faces of very different sizes), the
  `pairs` divides Divides costs hold a splitting one within TOL_OFFER of the best first order
  per cost among every face's divides. Fault `persistence` (rev 1's order, the most persistent
  across faces: persistence is relative to the face's largest u) fails it.
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
import shapely
from numba import njit
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_exits as gx  # noqa: E402
import growth as gr  # noqa: E402
import metrics  # noqa: E402

I64: TypeAlias = NDArray[np.int64]
F64: TypeAlias = NDArray[np.float64]
B: TypeAlias = NDArray[np.bool_]

PIX = 2.0              # m: the paths' raster
SRC_M = 20.0           # m: ear sources along each face's boundary, one per square this wide
STRETCH_M = 40.0       # m: divides' harmonic split: the face's boundary in stretches this wide
DUP_M = 2 * PIX        # m: Divides: a divide's pixel this near a kept divide's is shared
DUP_SHARE = 0.5        # Divides: a divide sharing this much of its pixels with a kept one dropped
BLD_G = 0.01           # divides' harmonic split: a pixel edge's conductance through a building
STEP_M = 0.25          # m: an ear's centreline sampled this finely (EarWorld.corridor)
LOCATE_K = 12          # EarWorld.locate: candidate triangles per point
CHUNK = 32             # sources per Dijkstra call (its distance and predecessor rows held)
REV = 3                # 2: Divides offers by first-order gain per metre, near-duplicates dropped
                       # (rev 1: the most persistent across faces); 3: the two floors are the
                       # plan's (`.f<face_m2>.e<ear_m>`, required); every rev 1 and 2 run had
                       # f2000 e30, the constants they replace
TOL_DEC = 1e-2         # decouple: relative (19570, 3 ears at h 0.25: 2.4e-3; at h 25, 0.24)
DEC_EVERY = 10         # build: the decoupling gap measured every this many ears
TOL_FIRST = 2e-3       # first: relative (19570: 3.3e-4 at FIRST_H; 3.3e-2 at h 250, the drop
                       # not yet linear)
WEAK_H = 25.0          # m: decouple's fault: a weak leak
FIRST_H = 2500.0       # m: first's leak (the drop linear in it: an ear's leak times its own
                       # resistance to ground far below 1)
OPEN_KT = 0.999        # first: the ears' open triangles only (no opening term to speak of)
SPUR_EARS = 4          # spur: the top candidate spurs judged
DETOUR_EARS = 3        # detour: loop ears built before the links are judged
DETOUR_TARGETS = 8     # detour: links from the first candidate's source judged at once
DETOUR_SEED = 0        # detour: their targets' draw
DETOUR_STRETCH = 1.2   # detour: their lengths, this times the straight line (plus PIX)
TOL_DETOUR = 1e-3      # detour: |S - all pairs again| over the largest S (float32 distances)
HS_W = 40.0            # m: horseshoe: the arms' spacing
HS_HOME_M = 4.0        # m: horseshoe: the homes' grid
HS_GAP_M = 14.0        # m: horseshoe: the gap the arms are compared at (its bars' ends this apart)
HS_ARMS = (20.0, 40.0, 80.0, 160.0)   # m: horseshoe: the arms compared
HS_ARM_M = 80.0        # m: horseshoe: the arms the gaps are compared at
HS_GAPS = (24.0, 16.0, 8.0)           # m: horseshoe: the gaps compared, closing
OFFER_EARS = 8         # offer: ears built by the divides check plan before the offers are judged
TOL_OFFER = 0.9        # offer: the costed divides' best splitting first order per cost over all
                       # divides' (5810, rev 2's states 6 / 8 / 12: 1.0 each; rev 1's persistence
                       # order 0.73 / 0.49 / 0.64, and 0.15 / 0.07 at rev 1's own states 4 / 8)


class Face(NamedTuple):
    """The ungrown triangles' components: per triangle its face (-1 grown), per face its area; a
    face counts once its area reaches growth.FACE_M2 (slivers a corridor leaves are not faces)."""
    lab: I64
    area: F64

    @property
    def n(self) -> int:
        return int((self.area >= gr.FACE_M2).sum())


class Ear(NamedTuple):
    """A candidate: its face, its path's pixels, whether it is a spur (its path ends on the face's
    open ground, not at a sink: its corridor joins the network at its start only, and it need not
    split its face), its corridor's new triangles, dD, dC, its cost, first-order gain, its detour
    saving (m: Detour; 0 without one, and for a spur, which shortens no trip) and the credit it
    earns in P (added to its first-order and exact gains wherever an ear is scored), exact gain
    (nan unless rescored)."""
    face: int
    path: I64
    spur: bool
    tris: I64
    dD: float
    dC: float
    cost: float
    first: float
    saving: float
    credit: float
    exact: float


class Raster(NamedTuple):
    """The block on a PIX raster: per pixel (flat) its nearest triangle and building share; the
    8-neighbour grid's edges (a, b) and their lengths."""
    shape: tuple[int, int]
    lo: F64
    xy: F64
    tri: I64
    bld: F64
    ea: I64
    eb: I64
    el: F64


def raster_of(g: gr.Growth) -> Raster:
    geo = g.m.geo
    boundary: Any = geo.boundary
    x0, y0, x1, y1 = boundary.bounds
    lo = np.array([x0, y0])
    shape = (int(np.ceil((x1 - x0) / PIX)), int(np.ceil((y1 - y0) / PIX)))
    i, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    xy = np.column_stack([lo[0] + (i.ravel() + 0.5) * PIX, lo[1] + (j.ravel() + 0.5) * PIX])
    inside = shapely.contains_xy(boundary, xy[:, 0], xy[:, 1])
    tri = np.where(inside, g.tree.query(xy)[1], -1)
    blds = shapely.union_all(np.asarray(geo.polys))
    shapely.prepare(blds)
    bld = shapely.contains_xy(blds, xy[:, 0], xy[:, 1]).astype(np.float64)
    ea, eb, el = _grid_edges(shape)
    return Raster(shape=shape, lo=lo, xy=xy, tri=tri, bld=bld, ea=ea, eb=eb, el=el)


def _grid_edges(shape: tuple[int, int]) -> tuple[I64, I64, F64]:
    """A PIX raster's 8-neighbour edges (a, b) and their lengths, pixels flat (row-major)."""
    ea, eb, el = [], [], []
    idx = np.arange(shape[0] * shape[1]).reshape(shape)
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        j0, j1 = max(0, -dj), shape[1] - max(0, dj)
        a = idx[0:shape[0] - di, j0:j1]
        b = idx[di:shape[0], j0 + dj:j1 + dj]
        ea.append(a.ravel())
        eb.append(b.ravel())
        el.append(np.full(a.size, PIX * np.hypot(di, dj)))
    return np.concatenate(ea), np.concatenate(eb), np.concatenate(el)


@njit(cache=True)
def _tree_sum(pred: NDArray[np.int32], order: I64, w: F64) -> F64:
    """Per node the sum of `w` down its shortest-path tree from the source (pred -9999 at it)."""
    acc = np.zeros(len(w))
    for k in range(len(order)):
        v = order[k]
        p = pred[v]
        acc[v] = w[v] + (acc[p] if p >= 0 else 0.0)
    return acc


class Candidates(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def pairs(self) -> int: ...                # candidates costed a step

    def ears(self, e: EarWorld, faces: Face, gain: F64, *, cost: gr.Cost,
             scale: float) -> list[Ear]:
        """The step's costed candidates; `scale` puts an ear's credit (in P) on its first-order
        gain's scale wherever the first order stands in for the exact gain (build)."""
        ...


class Ends(Protocol):
    """Where a path may end (Paths): per pixel of the source's face whether it is a target."""
    @property
    def name(self) -> str: ...

    def targets(self, sink: B, ground: B, dist: F64, *, ear_m: float) -> B: ...


@dataclass(frozen=True, kw_only=True)
class Loop:
    """Every road on a loop: a path ends at a sink pixel (`sink`) at least ear_m from its source
    in a straight line, and an ear that splits nothing is refused (build)."""

    @property
    def name(self) -> str:
        return "loop"

    def targets(self, sink: B, ground: B, dist: F64, *, ear_m: float) -> B:
        return np.asarray(sink & (dist >= ear_m))


@dataclass(frozen=True, kw_only=True)
class Spur:
    """Dead ends allowed (Idea A, owner 2026-11-03): Loop's targets, and also the face's open
    ground (`ground`: an ungrown pixel off every building and off every sink) between min_m and
    max_m from the source in a straight line, where the path is a spur (Ear.spur): its corridor
    joins the network at its start only and it is built whether or not it splits its face. The
    loop ears keep ear_m, the spurs min_m (a spur ends nowhere near the source's road by
    construction, so the two floors guard different things: an ear's ends not crowding one road,
    a spur not a stub)."""
    min_m: float
    max_m: float

    def __post_init__(self) -> None:
        if not 0 < self.min_m < self.max_m:
            raise ValueError(f"{self}: 0 < min_m < max_m")

    @property
    def name(self) -> str:
        return f"spur{self.min_m:g}-{self.max_m:g}"

    def targets(self, sink: B, ground: B, dist: F64, *, ear_m: float) -> B:
        return np.asarray((sink & (dist >= ear_m))
                          | (ground & (dist >= self.min_m) & (dist <= self.max_m)))


# ---------------------------------------------------------------- detour saving (Idea C)


class Roads(NamedTuple):
    """The network on the PIX raster at a state (`roads_of`): per pixel its column (-1 off the
    network), the road points (network pixels, one per SRC_M square, the first by index), per
    road point the homes' population share nearest it in a straight line (summing to 1), and per
    road point its distance along the network to every network pixel (rows road points, columns
    the network's pixels; inf where the network inside the block does not join them)."""
    col: I64
    pts: I64
    w: F64
    D: NDArray[np.float32]

    def among(self) -> F64:
        """The road points' distances to each other (float64)."""
        return np.asarray(self.D[:, self.col[self.pts]], dtype=np.float64)

    def mean(self) -> float:
        """The mean distance along the network between two homes' road points (each drawn by
        w), over the pairs it joins."""
        DRR = self.among()
        fin = np.isfinite(DRR)
        ww = np.outer(self.w, self.w)
        return float((ww * np.where(fin, DRR, 0.0)).sum() / (ww * fin).sum())


def _net_graph(net: B, ea: I64, eb: I64, el: F64) -> tuple[I64, sp.csr_matrix]:
    """(per pixel its column, -1 off the network; the network's graph on its columns)."""
    idx = np.flatnonzero(net)
    col = -np.ones(len(net), dtype=np.int64)
    col[idx] = np.arange(len(idx))
    ok = net[ea] & net[eb]
    m = len(idx)
    return col, sp.coo_matrix((el[ok], (col[ea[ok]], col[eb[ok]])), shape=(m, m)).tocsr()


def roads_of(net: B, xy: F64, ea: I64, eb: I64, el: F64, *, homes: F64, share: F64) -> Roads:
    """The network `net` (per pixel whether it lies on a road: the faces' sinks) as Roads; pure in
    its pixel inputs (xy per pixel its centre, (ea, eb, el) the 8-neighbour edges and lengths),
    `homes` the homes' points and `share` their population shares."""
    col, G = _net_graph(net, ea, eb, el)
    idx = np.flatnonzero(net)
    key = np.floor(xy[idx] / SRC_M).astype(np.int64)
    _, first = np.unique(key[:, 0] * 1_000_000 + key[:, 1], return_index=True)
    pts = idx[np.sort(first)]
    w = np.bincount(cKDTree(xy[pts]).query(homes)[1], weights=share, minlength=len(pts))
    D = np.empty((len(pts), len(idx)), dtype=np.float32)
    for c0 in range(0, len(pts), CHUNK):
        D[c0:c0 + CHUNK] = dijkstra(G, directed=False, indices=col[pts[c0:c0 + CHUNK]])
    return Roads(col=col, pts=pts, w=w / w.sum(), D=D)


@njit(cache=True)
def _hinge(c: F64, w: F64, z: F64) -> F64:
    """Per row t of z: sum over (p, q) of w_p w_q (c[p, q] - z[t, q])^+. Per column q, c sorted
    (descending) with its prefix sums of w and w c, so each (t, q) is one binary search: the p
    with c[p, q] > z[t, q] are a prefix."""
    n, m = len(w), z.shape[0]
    out = np.zeros(m)
    for q in range(n):
        if w[q] == 0.0:
            continue
        o = np.argsort(-c[:, q])
        cs = c[o, q]
        W = np.cumsum(w[o])
        WC = np.cumsum(w[o] * np.where(np.isfinite(cs), cs, 0.0))
        top = cs[0]
        for t in range(m):
            zz = z[t, q]
            if not top > zz:
                continue
            lo, hi = 0, n
            while lo < hi:
                mid = (lo + hi) // 2
                if cs[mid] > zz:
                    lo = mid + 1
                else:
                    hi = mid
            out[t] += w[q] * (WC[lo - 1] - zz * W[lo - 1])
    return out


def savings(r: Roads, a: int, tg: I64, length: F64) -> F64:
    """Per target pixel t of `tg` (network pixels) the detour saving of a new link from the
    network pixel `a` to t of `length[t]` m: the mean, over pairs of homes (each drawn by the
    road points' w), of how much shorter the trip between their road points gets, S = sum_pq
    w_p w_q (D_pq - D'_pq) with D' = min(D_pq, D_pa + l + D_tq, D_pt + l + D_aq), the link used
    at most once. At most one orientation shortens a pair (the two together would make
    2 D_pq > D_pa + D_aq + D_pt + D_tq + 2 l >= 2 D_pq + 2 l, by the triangle inequality), and
    relabelling p, q maps the one onto the other, so S = 2 sum_pq w_p w_q (D_pq - D_pa - l -
    D_tq)^+ (`_hinge`). A pair the network inside the block does not join (D_pq inf) counts
    nothing."""
    DRR = r.among()
    Da = np.asarray(r.D[:, r.col[a]], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        c = np.where(np.isfinite(DRR) & np.isfinite(Da)[:, None], DRR - Da[:, None], -np.inf)
    z = length[:, None] + np.asarray(r.D[:, r.col[tg]], dtype=np.float64).T
    return 2.0 * _hinge(c, r.w, z)


class Field(Protocol):
    """The attraction at one state (Attract.field)."""

    def credits(self, a: int, tg: I64, pred: NDArray[np.int32], order: I64) -> F64:
        """Per target pixel of `tg` the credit in P of the path from `a` to it down the
        shortest-path tree `pred` (its nodes `order`, nearest first)."""
        ...

    def of(self, path: I64, *, spur: bool) -> tuple[float, float]:
        """A costed path's (detour saving in m, credit in P)."""
        ...


class Attract(Protocol):
    """A term added to the escape objective of `paths` (owner 2026-11-03, Idea C): an ear's score
    is (gain + credit) / cost at every stage (each source's target, the costed ears' order, the
    built ear's exact gain)."""

    @property
    def suffix(self) -> str: ...                 # its token in the plan's name ('' for none)

    def field(self, e: EarWorld, *, P0: float) -> Field: ...


class _NoField:
    def credits(self, a: int, tg: I64, pred: NDArray[np.int32], order: I64) -> F64:
        return np.zeros(len(tg))

    def of(self, path: I64, *, spur: bool) -> tuple[float, float]:
        return 0.0, 0.0


@dataclass(frozen=True, kw_only=True)
class NoDetour:
    """The escape objective alone (every ears-paths run before Idea C): no credit. Its name adds
    no token, so those runs' names stand."""

    @property
    def suffix(self) -> str:
        return ""

    def field(self, e: EarWorld, *, P0: float) -> Field:
        return _NoField()


@dataclass(frozen=True, kw_only=True)
class _DetourField:
    roads: Roads
    xy: F64
    kappa: float                                 # P per metre of detour saving

    def credits(self, a: int, tg: I64, pred: NDArray[np.int32], order: I64) -> F64:
        out = np.zeros(len(tg))
        on = self.roads.col[tg] >= 0             # sinks; a spur's open ground shortens nothing
        if not on.any():
            return out
        p = pred[order]
        inc = np.zeros(len(pred))
        inc[order] = np.where(p >= 0, np.hypot(*(self.xy[order] - self.xy[np.maximum(p, 0)]).T),
                              0.0)
        length = _tree_sum(pred, order, inc)
        out[on] = self.kappa * savings(self.roads, a, tg[on], length[tg[on]])
        return out

    def of(self, path: I64, *, spur: bool) -> tuple[float, float]:
        if spur:
            return 0.0, 0.0
        L = float(np.hypot(*np.diff(self.xy[path], axis=0).T).sum())
        S = float(savings(self.roads, int(path[0]), path[-1:], np.array([L]))[0])
        return S, self.kappa * S


@dataclass(frozen=True, kw_only=True)
class Detour:
    """Detour-saving attraction (owner 2026-11-03, Idea C): a loop ear's credit is lam P0 S /
    S0, S its detour saving (`savings`: the mean metres a trip between two homes' road points
    saves, the network the state's sinks, `roads_of`) and S0 the mean such trip on the street
    alone (the block's own length scale, `Roads.mean`): lam is the credit, as a share of P0, of a
    saving as long as the mean street-only trip. A spur shortens no trip and earns nothing: the
    loop it sets up is credited once, when an ear closes it (its saving then large against its
    short length), which is what makes a nearly finished loop finish.

    Runs (2026-11-03, spur10-80, ears_compare.py): it ends spur takeover (dead-end share of
    length, 41785 f100 at D 0.4: .92 plain, .07 / .05 / .02 at lam 0.1 / 0.3 / 1; 5810 f2000 at
    D 0.15: .34 plain, .04 / .05 at lam 0.1 / 0.3), but buys no L_P: 41785 L_P at D 0.1 / 0.2 /
    0.4 plain .748 / .894 / .978, lam 0.1 .740 / .880 / .978, loop .656 / .855 / .980; 5810 at D
    0.08 / 0.10 / 0.15 plain .949 / .960 / .978, lam 0.1 .950 / .962 / .978, lam 0.3 .952 / .960
    / .977, loop .951 / .968 / .983, at 1.7 -- 1.9 times plain spurs' wall time."""
    lam: float

    def __post_init__(self) -> None:
        if not self.lam > 0:
            raise ValueError(f"lam {self.lam}: positive (NoDetour is the term off)")

    @property
    def suffix(self) -> str:
        return f".dt{self.lam:g}"

    def field(self, e: EarWorld, *, P0: float) -> Field:
        homes, share = e.homes()
        ra = e.ra
        r = roads_of(e.sinks(), ra.xy, ra.ea, ra.eb, ra.el, homes=homes, share=share)
        r0 = roads_of(e.sinks_of(e.street), ra.xy, ra.ea, ra.eb, ra.el, homes=homes,
                      share=share)
        return _DetourField(roads=r, xy=ra.xy, kappa=self.lam * P0 / r0.mean())


@dataclass(frozen=True, kw_only=True)
class Paths:
    """Cheapest paths on the raster (module docstring); `bld` the extra cost of a metre through a
    building, `pairs` the sources whose best ears are built and costed; a face under `face_m2` is
    not split, an ear's ends at least `ear_m` apart in a straight line; `ends` where a path may
    end (Loop: at a sink; Spur: there or on open ground); `attract` the term added to the escape
    objective (NoDetour: none; Detour: detour saving)."""
    bld: float
    pairs: int
    face_m2: float
    ear_m: float
    ends: Ends
    attract: Attract

    @property
    def name(self) -> str:
        return (f"paths.b{self.bld:g}.f{self.face_m2:g}.e{self.ear_m:g}.{self.ends.name}"
                f"{self.attract.suffix}")

    def ears(self, e: EarWorld, faces: Face, gain: F64, *, cost: gr.Cost,
             scale: float) -> list[Ear]:
        g, ra = e.g, e.ra
        pf = np.where(ra.tri >= 0, faces.lab[np.maximum(ra.tri, 0)], -1)    # pixel's face
        big = faces.area >= self.face_m2
        live = (pf >= 0) & big[np.maximum(pf, 0)]
        # the ends: sink pixels of the face (a road or the street, not a wall)
        same = pf[ra.ea] == pf[ra.eb]
        bnd = live & e.sinks()
        ground = live & (ra.bld == 0) & ~bnd       # open ground: where a spur may end
        # per pixel its first-order gain density (the triangles under it)
        gp = np.bincount(_pix_of(ra, g.cen), weights=gain, minlength=len(pf))
        fld = self.attract.field(e, P0=g.P0)
        ok = live[ra.ea] & live[ra.eb] & same
        w = ra.el[ok] * (1.0 + self.bld * 0.5 * (ra.bld[ra.ea[ok]] + ra.bld[ra.eb[ok]]))
        n = len(pf)
        G = sp.coo_matrix((w, (ra.ea[ok], ra.eb[ok])), shape=(n, n)).tocsr()
        src = np.flatnonzero(bnd)
        key = np.floor(ra.xy[src] / SRC_M).astype(np.int64)
        _, first = np.unique(key[:, 0] * 1_000_000 + key[:, 1], return_index=True)
        src = src[np.sort(first)]
        best: list[tuple[float, int, int, NDArray[np.int32]]] = []
        for c0 in range(0, len(src), CHUNK):
            s = src[c0:c0 + CHUNK]
            dist, pred = dijkstra(G, directed=False, indices=s, return_predecessors=True)
            for k, a in enumerate(s):
                d = dist[k]
                fin = np.isfinite(d)
                order = np.flatnonzero(fin)[np.argsort(d[fin])]
                acc = _tree_sum(pred[k], order, gp)
                crow = np.hypot(*(ra.xy - ra.xy[a]).T)     # straight-line, from the source
                tg = np.flatnonzero(self.ends.targets(bnd, ground, crow, ear_m=self.ear_m)
                                    & fin & (pf == pf[a]))
                if not len(tg):
                    continue
                cr = fld.credits(int(a), tg, pred[k], order)
                r = (acc[tg] + scale * cr) / np.maximum(d[tg], 1e-9)
                t = int(tg[np.argmax(r)])
                best.append((float(r.max()), int(a), t, pred[k].copy()))
        best.sort(key=lambda q: -q[0])
        out = []
        for _, a, t, pr in best[:self.pairs]:
            path = [t]
            while path[-1] != a:
                path.append(int(pr[path[-1]]))
            path_a = np.asarray(path[::-1], dtype=np.int64)
            spur = not bool(bnd[t])
            tris = e.corridor(ra.xy[path_a], spur=spur)
            if not len(tris):
                continue
            dD, dC = g.costs_of(tris)
            c = float(cost.of(np.array([dD]), np.array([dC]), P0=g.P0)[0])
            sv, cr = fld.of(path_a, spur=spur)
            out.append(Ear(face=int(pf[a]), path=path_a, spur=spur, tris=tris, dD=dD, dC=dC,
                           cost=c, first=float(gain[tris].sum()), saving=sv, credit=cr,
                           exact=np.nan))
        return out


class Divide(NamedTuple):
    """A candidate cut of one face: its pixels in order from one boundary end to the other, its
    persistence (the strict basins' pass over the face's largest u; the harmonic split's median
    u along it over the largest), and which construction gave it."""
    pixels: I64
    persistence: float
    kind: str


def _ms_cut(u: F64, on: B, edge: B, a: I64, b: I64, *, tau: float) -> B:
    """The strict Morse-Smale divides (module docstring of `divides`): per pixel whether it
    lies on an edge between two surviving basins."""
    n = len(u)
    nxt = np.arange(n)
    best = u.copy()
    for x, y in ((a, b), (b, a)):
        o = np.argsort(u[y])[::-1]                 # the lowest neighbour written last wins
        xs, ys = x[o], y[o]
        lower = u[ys] < best[xs]
        nxt[xs[lower]] = ys[lower]
        best[xs[lower]] = u[ys[lower]]
    nxt = np.where(edge | ~on, np.arange(n), nxt)
    for _ in range(64):                            # pointer jumping to the sinks
        nn = nxt[nxt]
        if np.array_equal(nn, nxt):
            break
        nxt = nn
    roots, basin = np.unique(nxt[on], return_inverse=True)
    bas = -np.ones(n, dtype=np.int64)
    bas[on] = basin
    k = len(roots)
    umax = float(u[on].max())
    pa, pb = bas[a], bas[b]
    diff = pa != pb
    key = np.minimum(pa[diff], pb[diff]) * k + np.maximum(pa[diff], pb[diff])
    h = np.maximum(u[a[diff]], u[b[diff]])
    o = np.lexsort((h, key))
    first = np.ones(len(o), dtype=bool)
    first[1:] = key[o][1:] != key[o][:-1]
    pk, ph = key[o][first], h[o][first]
    parent = np.arange(k)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    for q in np.argsort(ph):
        if ph[q] >= tau * umax:
            break
        r1, r2 = find(int(pk[q] // k)), find(int(pk[q] % k))
        if r1 != r2:
            parent[max(r1, r2)] = min(r1, r2)
    top = np.array([find(i) for i in range(k)])
    fb = np.where(bas >= 0, top[np.maximum(bas, 0)], -1)
    cut = np.zeros(n, dtype=bool)
    d = fb[a] != fb[b]
    cut[a[d]] = True
    cut[b[d]] = True
    return cut


def _harmonic_cut(xy: F64, bld: F64, on: B, edge: B, a: I64, b: I64, *,
                  stretch_m: float) -> B:
    """The harmonic-measure split: the face's sinks in stretches (its sink pixels on a
    stretch_m grid), per interior pixel the stretch taking most of its current (the harmonic
    measure of each stretch, from one factorization of the face's pixel Laplacian: an edge
    conducts 1 on open ground, BLD_G through a building), and the edges between pixels of
    different stretches."""
    n = len(xy)
    bkey = np.floor(xy / stretch_m).astype(np.int64)
    bkey = bkey[:, 0] * 1_000_000 + bkey[:, 1]
    st_ids, st = np.unique(bkey[edge], return_inverse=True)
    ns = len(st_ids)
    if ns < 2:
        return np.zeros(n, dtype=bool)
    inner = on & ~edge
    ii = -np.ones(n, dtype=np.int64)
    ii[inner] = np.arange(int(inner.sum()))
    bi = -np.ones(n, dtype=np.int64)
    bi[edge] = st
    w = np.where((bld[a] > 0) | (bld[b] > 0), BLD_G, 1.0)
    rows, cols, vals = [], [], []
    deg = np.zeros(n)
    np.add.at(deg, a, w)
    np.add.at(deg, b, w)
    m = int(inner.sum())
    rhs = np.zeros((m, ns))
    for x, y in ((a, b), (b, a)):
        both = inner[x] & inner[y]
        rows.append(ii[x[both]])
        cols.append(ii[y[both]])
        vals.append(-w[both])
        tob = inner[x] & edge[y]
        np.add.at(rhs, (ii[x[tob]], bi[y[tob]]), w[tob])
    L = sp.csc_matrix((np.concatenate(vals + [deg[inner]]),
                       (np.concatenate(rows + [np.arange(m)]),
                        np.concatenate(cols + [np.arange(m)]))), shape=(m, m))
    omega = splu(L).solve(rhs)
    lab = -np.ones(n, dtype=np.int64)
    lab[inner] = np.argmax(omega, axis=1)
    lab[edge] = st
    cut = np.zeros(n, dtype=bool)
    d = on[a] & on[b] & (lab[a] != lab[b]) & inner[a] & inner[b]
    cut[a[d]] = True
    cut[b[d]] = True
    return cut


def divides(u: F64, lab: I64, sink: B, ea: I64, eb: I64, xy: F64, bld: F64, *, face: int,
            tau: float, stretch_m: float, ear_m: float) -> list[Divide]:
    """Candidate cuts of one face of a pixel graph from its potential, most persistent first
    (brief, `divides`); pure in its inputs, so other optimizers can seed from it.

    u: per pixel the potential (held near 0 at the street and the grounded roads); lab: per pixel
    its face (-1 off every face); sink: per pixel whether it lies on the street or a grounded
    road (the face's sinks: the rest of its boundary, a block edge with no exit, is a wall the
    current does not cross, and a cut ending there would be a dead end); (ea, eb): the graph's
    edges (8-neighbour pixels); xy: per pixel its centre; bld: per pixel its building share.
    First the strict Morse-Smale
    divides: every other pixel descends to its lowest neighbour, a basin is the pixels draining
    to one sink, and basins are merged across their passes (the lowest max(u_a, u_b) between
    them) below tau times the face's largest u. Along a continuous street ring every boundary
    pixel is a sink and the passes between neighbouring sinks are low, so the basins degenerate
    into one; then the harmonic-measure split (`_harmonic_cut`, stretches stretch_m wide) gives
    the divides. Either way the divide pixels form a network; its ends are the divide pixels
    beside a sink, and each end's cut is the path through the network to another end at
    least ear_m away that keeps highest in u (the network's edges weighted by length over
    u, a shortest path), so every cut runs sink to sink (no dead end)."""
    on = lab == face
    if not on.any():
        return []
    both = on[ea] & on[eb]
    a, b = ea[both], eb[both]
    edge = on & sink
    umax = float(u[on].max())
    out = _cuts(u, xy, _ms_cut(u, on, edge, a, b, tau=tau), edge, a, b, umax=umax, kind="ms",
                ear_m=ear_m)
    if not out:
        out = _cuts(u, xy, _harmonic_cut(xy, bld, on, edge, a, b, stretch_m=stretch_m), edge,
                    a, b, umax=umax, kind="harmonic", ear_m=ear_m)
    out.sort(key=lambda d: -d.persistence)
    return out


def _cuts(u: F64, xy: F64, cut: B, edge: B, a: I64, b: I64, *, umax: float,
          kind: str, ear_m: float) -> list[Divide]:
    """The boundary-to-boundary paths through a divide network (`divides`), one per end, each
    path once."""
    if not cut.any():
        return []
    near = np.zeros(len(u), dtype=bool)
    near[a[cut[b] & edge[a]]] = True
    near[b[cut[a] & edge[b]]] = True
    net = cut | near
    ends = np.flatnonzero(near)
    if len(ends) < 2:
        return []
    key = np.floor(xy[ends] / SRC_M).astype(np.int64)
    _, first = np.unique(key[:, 0] * 1_000_000 + key[:, 1], return_index=True)
    ends = ends[np.sort(first)]
    k = net[a] & net[b] & ~(edge[a] & edge[b])
    n = len(u)
    w = np.hypot(*(xy[a[k]] - xy[b[k]]).T) / (1e-3 * umax + 0.5 * (u[a[k]] + u[b[k]]))
    G = sp.coo_matrix((w, (a[k], b[k])), shape=(n, n)).tocsr()
    out: list[Divide] = []
    seen: set[bytes] = set()
    for c0 in range(0, len(ends), CHUNK):
        src = ends[c0:c0 + CHUNK]
        dist, pred = dijkstra(G, directed=False, indices=src, return_predecessors=True)
        for r, s0 in enumerate(src):
            far = np.hypot(*(xy[ends] - xy[s0]).T) >= ear_m
            tg = ends[far & np.isfinite(dist[r, ends])]
            if not len(tg):
                continue
            t = int(tg[np.argmin(dist[r, tg])])
            path = [t]
            while path[-1] != s0:
                path.append(int(pred[r, path[-1]]))
            px = np.asarray(path[::-1], dtype=np.int64)
            h = np.sort(px).tobytes()
            if h in seen:
                continue
            seen.add(h)
            out.append(Divide(pixels=px, persistence=float(np.median(u[px])) / umax, kind=kind))
    return out


class Offer(NamedTuple):
    """A divide offered for costing: its face, the divide, and its first-order gain per metre
    (the state's first-order gains binned to the PIX raster, summed along its pixels, over its
    length: comparable across faces, as `paths`' gain per path cost is)."""
    face: int
    divide: Divide
    per_m: float


def _distinct(ds: list[Divide], xy: F64) -> list[Divide]:
    """`ds` (most persistent first) less the near-duplicates: a divide with at least DUP_SHARE of
    its pixels within DUP_M of a kept one's (ends one SRC_M cell apart trace the same divide)."""
    kept: list[Divide] = []
    trees: list[cKDTree] = []
    for d in ds:
        p = xy[d.pixels]
        if any((t.query(p, k=1, distance_upper_bound=DUP_M)[0] <= DUP_M).mean() >= DUP_SHARE
               for t in trees):
            continue
        kept.append(d)
        trees.append(cKDTree(p))
    return kept


@dataclass(frozen=True, kw_only=True)
class Divides:
    """The divides of u (`divides`, ends at least `ear_m` apart), per face of at least `face_m2`
    on its own u (the state's,
    the roads grounded), at persistence threshold `tau`, the harmonic-measure split where the
    strict basins degenerate, near-duplicates dropped (`_distinct`); across faces the `pairs`
    with the most first-order gain per metre costed (`offers`). Rev 1 took the `pairs` most
    persistent across faces, but persistence is relative to its face's largest u, so on 5810 the
    small faces' divides (u 2 -- 60) filled the pairs and the large faces' (u 160 -- 230, 7 -- 1000
    times the costed ones' first-order gain per cost) were never costed."""
    tau: float
    pairs: int
    face_m2: float
    ear_m: float

    @property
    def name(self) -> str:
        return f"divides.p{self.tau:g}.f{self.face_m2:g}.e{self.ear_m:g}"

    def offers(self, e: EarWorld, faces: Face, gain: F64) -> list[Offer]:
        """Every face's distinct divides, the most first-order gain per metre first."""
        g, ra = e.g, e.ra
        pf = np.where(ra.tri >= 0, faces.lab[np.maximum(ra.tri, 0)], -1)
        gp = np.bincount(_pix_of(ra, g.cen), weights=gain, minlength=len(pf))
        sink = e.sinks()
        out: list[Offer] = []
        for f in np.flatnonzero(faces.area >= self.face_m2):
            ds = divides(e.u_pix, pf, sink, ra.ea, ra.eb, ra.xy, ra.bld, face=int(f),
                         tau=self.tau, stretch_m=STRETCH_M, ear_m=self.ear_m)
            for d in _distinct(ds, ra.xy):
                L = float(np.hypot(*np.diff(ra.xy[d.pixels], axis=0).T).sum())
                out.append(Offer(face=int(f), divide=d,
                                 per_m=float(gp[d.pixels].sum()) / max(L, PIX)))
        out.sort(key=lambda o: -o.per_m)
        return out

    def ears(self, e: EarWorld, faces: Face, gain: F64, *, cost: gr.Cost,
             scale: float) -> list[Ear]:
        return self.costed(e, self.offers(e, faces, gain)[:self.pairs], gain, cost=cost)

    @staticmethod
    def costed(e: EarWorld, offers: list[Offer], gain: F64, *, cost: gr.Cost) -> list[Ear]:
        g, ra = e.g, e.ra
        out = []
        for o in offers:
            f, d = o.face, o.divide
            tris = e.corridor(ra.xy[d.pixels], spur=False)
            if not len(tris):
                continue
            dD, dC = g.costs_of(tris)
            c = float(cost.of(np.array([dD]), np.array([dC]), P0=g.P0)[0])
            out.append(Ear(face=f, path=d.pixels, spur=False, tris=tris, dD=dD, dC=dC, cost=c,
                           first=float(gain[tris].sum()), saving=0.0, credit=0.0, exact=np.nan))
        return out


def _pix_of(ra: Raster, xy: F64) -> I64:
    ij = np.floor((xy - ra.lo) / PIX).astype(np.int64)
    ij[:, 0] = np.clip(ij[:, 0], 0, ra.shape[0] - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, ra.shape[1] - 1)
    return ij[:, 0] * ra.shape[1] + ij[:, 1]


@dataclass(frozen=True, kw_only=True)
class Plan:
    candidates: Candidates
    contrast: gr.GroundedLeak
    cost: gr.Cost
    exact: int
    D: float

    def __post_init__(self) -> None:
        if not (self.exact >= 1 and self.D > 0):
            raise ValueError(f"{self}: exact >= 1 (an ear is built on its exact gain), a budget")

    @property
    def name(self) -> str:
        return (f"ears.{self.candidates.name}.{self.contrast.name}.{self.cost.name}"
                f".M{self.candidates.pairs}"
                f".K{self.exact}.D{self.D:g}.v{REV}")

    @staticmethod
    def of(spec: str) -> Plan:
        m = re.fullmatch(r"ears\.(?:paths\.b(?P<bld>[0-9.]+)\.f(?P<pf>[0-9.]+)\.e(?P<pe>[0-9.]+)"
                         r"\.(?P<ends>loop|spur(?P<lo>[0-9.]+)-(?P<hi>[0-9.]+))"
                         r"(?:\.dt(?P<lam>[0-9.]+))?"
                         r"|divides\.p(?P<tau>[0-9.]+)\.f(?P<df>[0-9.]+)\.e(?P<de>[0-9.]+))"
                         r"\.gl(?P<h>[0-9.]+)Croad\.n(?P<nu>[0-9.e-]+)s(?P<price>[0-9.e-]+)"
                         r"\.M(?P<pairs>[0-9]+)\.K(?P<exact>[0-9]+)\.D(?P<D>[0-9.]+)", spec)
        if m is None:
            raise SystemExit(f"{spec}: a plan is ears.<paths.b<bld>.f<face_m2>.e<ear_m>"
                             ".<loop|spur<min_m>-<max_m>>[.dt<lam>]|divides.p<tau>.f<face_m2>"
                             ".e<ear_m>>"
                             ".gl<h>Croad.n<nu>s<price>.M<pairs>.K<exact>.D<D>")
        cands: Candidates
        if m["bld"] is not None:
            ends: Ends = (Loop() if m["ends"] == "loop" else
                          Spur(min_m=float(m["lo"]), max_m=float(m["hi"])))
            att: Attract = NoDetour() if m["lam"] is None else Detour(lam=float(m["lam"]))
            cands = Paths(bld=float(m["bld"]), pairs=int(m["pairs"]), face_m2=float(m["pf"]),
                          ear_m=float(m["pe"]), ends=ends, attract=att)
        else:
            cands = Divides(tau=float(m["tau"]), pairs=int(m["pairs"]), face_m2=float(m["df"]),
                            ear_m=float(m["de"]))
        return Plan(candidates=cands, contrast=gr.GroundedLeak(h=float(m["h"])),
                    cost=gr.Cost(nu=float(m["nu"]), price=float(m["price"])),
                    exact=int(m["exact"]), D=float(m["D"]))


class EarWorld:
    """growth.Growth's grounded world, its faces and their restricted solves."""

    def __init__(self, g: gr.Growth) -> None:
        self.g = g
        self.ra = raster_of(g)
        self.u_pix = np.zeros(len(self.ra.tri))
        self.street = np.zeros(g.nt, dtype=bool)
        self.street[g.m.fx.fm.fa[g.m.c.pf]] = True

    def bounding(self) -> B:
        """Per triangle whether it bounds a face as a road does: the street's faces' triangles
        and the grown roads (not the paved ways inside the block: ungrown, open ground of the
        face, a cut ending on one splits nothing)."""
        return self.street | (self.g.step >= 0)

    def sinks(self) -> B:
        """Per pixel whether a bounding triangle (`bounding`) lies within PIX of its centre: the
        faces' sinks."""
        return self.sinks_of(self.bounding())

    def sinks_of(self, bounding: B) -> B:
        """`sinks` with the triangles `bounding` the roads (Detour: the street's alone)."""
        att = np.flatnonzero(bounding)
        d, _ = cKDTree(self.g.cen[att]).query(self.ra.xy, k=1, distance_upper_bound=PIX)
        return np.asarray(np.isfinite(d) & (self.ra.tri >= 0))

    def homes(self) -> tuple[F64, F64]:
        """(per home its footprint's centroid, its population share)."""
        c = shapely.centroid(np.asarray(self.g.m.geo.polys))
        if len(c) != len(self.g.w):
            raise ValueError(f"{len(c)} footprints, {len(self.g.w)} homes' shares")
        return np.column_stack([shapely.get_x(c), shapely.get_y(c)]), self.g.w

    def set_u(self, u: F64) -> None:
        """The state's potential per pixel (its nearest triangle's cell's)."""
        t = np.maximum(self.ra.tri, 0)
        self.u_pix = np.where(self.ra.tri >= 0, u[self.g.m.c.cell[t]], 0.0)

    def faces(self) -> Face:
        g = self.g
        off = g.step < 0
        e = off[g.pa] & off[g.pb]
        n = g.nt
        G = sp.coo_matrix((np.ones(int(e.sum()), dtype=np.int8), (g.pa[e], g.pb[e])),
                          shape=(n, n))
        k, lab = connected_components(G, directed=False)
        lab = np.where(off, lab, -1)
        area = np.bincount(lab[off], weights=g.area[off], minlength=k)
        return Face(lab=lab, area=area)

    def corridor(self, pts: F64, *, spur: bool) -> I64:
        """An ear's corridor along the polyline `pts`, extended at both ends to the nearest
        bounding triangle (the road or street it joins; a spur's at its start only, its end on
        open ground): the ungrown triangles within growth.PATCH_R of its points, and every
        triangle its centreline crosses (sampled every STEP_M): on 5810's open ground a triangle
        can be larger than a road, and one whose centroid lies off the corridor but which
        straddles it joins the two halves (the first runs' ears split nothing)."""
        g = self.g
        att = np.flatnonzero(self.bounding())
        _, k = cKDTree(g.cen[att]).query(pts[[0, -1]], k=1)
        line = np.vstack([g.cen[att[k[0]]], pts] + ([] if spur else [g.cen[att[k[1]]]]))
        seg = np.diff(line, axis=0)
        L = np.hypot(*seg.T)
        dense = [line[:1]]
        for p0, d, n in zip(line[:-1], seg, np.ceil(L / STEP_M).astype(int), strict=True):
            dense.append(p0 + np.linspace(0.0, 1.0, max(n, 1) + 1)[1:, None] * d)
        P = np.vstack(dense)
        lists = g.tree.query_ball_point(pts, r=gr.PATCH_R)
        t = np.unique(np.concatenate([np.asarray(x, dtype=np.int64) for x in lists]
                                     + [self.locate(P)]))
        return t[(t >= 0) & (g.step[t] < 0)]

    def locate(self, P: F64) -> I64:
        """Per point the triangle containing it (among its LOCATE_K nearest by centroid; the
        nearest where none does)."""
        g = self.g
        fm = g.m.fx.fm
        _, cand = g.tree.query(P, k=LOCATE_K)
        v = fm.xy[fm.tri[cand]]                    # points x K x 3 x 2
        a, b, c = v[..., 0, :], v[..., 1, :], v[..., 2, :]
        p = P[:, None, :]

        def cross(o: F64, x: F64, y: F64) -> F64:
            return np.asarray((x[..., 0] - o[..., 0]) * (y[..., 1] - o[..., 1])
                              - (x[..., 1] - o[..., 1]) * (y[..., 0] - o[..., 0]))

        d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
        inside = ((d1 >= 0) & (d2 >= 0) & (d3 >= 0)) | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0))
        first = np.where(inside.any(axis=1), np.argmax(inside, axis=1), 0)
        return np.asarray(cand[np.arange(len(P)), first], dtype=np.int64)

    def values(self, on: B) -> tuple[Any, Any, F64]:
        """(the system, its reach, its operator's values) with `on` grown, unfactored."""
        g, m = self.g, self.g.m
        kt, b, leak = g.contrast.world(g, on)
        sys = (m.sys if leak is None else
               m.sys.with_graph(m.sys.graph.with_leak(g.leak0 + leak, live0=m.leak_site.live)))
        fm = m.exits.edge.mesh(gx.boosted(m.fx.fm, b))
        kp, _ = gcl._passage_factors(fm, kt, g.k, m.fill)
        r = sys.graph.reach(fm, kt, kp)
        return sys, r, sys.graph.values(r)

    def P_face(self, on: B, cells: B) -> float:
        """P of the cells `cells` alone, every other cell held at 0 (the plan's factorization
        restricted to them)."""
        sys, r, data = self.values(on)
        keep = r.unknown & cells
        rhs = np.asarray(sys.Pc @ self.g.m.pop) * keep
        x = sys.planned.factor(data, keep).solve(rhs)
        return float(rhs @ x)

    def face_cells(self, faces: Face, f: int) -> B:
        c = np.zeros(self.g.m.c.n, dtype=bool)
        c[self.g.m.c.cell[faces.lab == f]] = True
        return c


def build(e: EarWorld, plan: Plan, *, out: Path, log: Any, steps: int | None = None) -> None:
    """Ears greedily until D reaches plan.D (or `steps` ears: the checks' short runs)."""
    g = e.g
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []
    picks: list[dict[str, float]] = []
    t0 = time.time()
    faces = e.faces()
    n0 = faces.n
    # a credit's weight where the first order stands in for the exact gain: the last step's
    # rescored ears' first order over their exact gain (the grounded leak's first order
    # overstates it 10^3 -- 10^5 times); 0 before any exact gain (step 0: credit in the exact
    # stage only), as growth.grow's islands at its first step
    scale = 0.0
    step = 0
    while True:
        st = g.state(g.step >= 0)
        u, P = g.solve(st)
        D = g.D()
        dec = np.nan
        if step and step % DEC_EVERY == 0:
            cells = np.zeros(g.m.c.n, dtype=bool)
            cells[g.m.c.cell[faces.lab >= 0]] = True
            dec = (P - e.P_face(g.step >= 0, cells)) / P
        rows.append(dict(step=step, P=P, L_P=1.0 - P / g.P0, D=D, C=g.C(), decouple=dec,
                         area=float(g.area[g.step >= 0].sum()),
                         faces_patches=faces.n, faces=faces.n,
                         n_patches=0, n_bridges=0, islands=0, island_m2=0.0,
                         largest_face=float(faces.area.max()), t=time.time() - t0))
        log(f"state {step}: L_P {1 - P / g.P0:.4f}, D {D:.4f}, C {rows[-1]['C']:.0f} m^2, faces "
            f"{faces.n} (largest {faces.area.max():.0f} m^2) ({time.time() - t0:.0f}s)")
        pd.DataFrame(rows).to_parquet(out / "steps.parquet")
        pd.DataFrame(picks).to_parquet(out / "picks.parquet")
        pd.DataFrame({"step": pd.Series(dtype=np.int64)}).iloc[:0].to_parquet(
            out / "bridges.parquet")
        pd.DataFrame({"step": g.step}).to_parquet(out / "tri_step.parquet")
        if D >= plan.D or (steps is not None and step >= steps):
            break
        gain = plan.contrast.gains(g, st, u)
        e.set_u(u)
        cands = plan.candidates.ears(e, faces, gain, cost=plan.cost, scale=scale)
        if not cands:
            log("no ear left: stopped short of the budget")
            break
        cands.sort(key=lambda q: -(q.first + scale * q.credit) / q.cost)
        on = g.step >= 0
        # the splitting ones only (no dead end) and the spurs (a dead end by design: Spur), best
        # first order per cost first, `exact` of them
        split: list[tuple[Ear, Face]] = []
        for q in cands:
            if len(split) == plan.exact:
                break
            g.step[q.tris] = step
            nf = e.faces()
            g.step[q.tris] = -1
            if nf.n > faces.n or q.spur:
                split.append((q, nf))
            else:
                log(f"  refused an ear in face {q.face}: it splits nothing")
        before: dict[int, float] = {}
        scored = []
        for q, nf in split:
            cells = e.face_cells(faces, q.face)
            if q.face not in before:
                before[q.face] = e.P_face(on, cells)
            on2 = on.copy()
            on2[q.tris] = True
            scored.append((q._replace(exact=before[q.face] - e.P_face(on2, cells)), nf))
        if not scored:
            log("no candidate ear splits a face: stopped")
            break
        built, faces = max(scored, key=lambda qn: (qn[0].exact + qn[0].credit) / qn[0].cost)
        ex = sum(q.exact for q, _ in scored)
        if ex > 0:
            scale = sum(q.first for q, _ in scored) / ex
        g.step[built.tris] = step
        xy = e.ra.xy[built.path]
        picks.append(dict(step=step, face=built.face, spur=built.spur,
                          ax=float(xy[0, 0]), ay=float(xy[0, 1]),
                          bx=float(xy[-1, 0]), by=float(xy[-1, 1]),
                          length=float(np.hypot(*np.diff(xy, axis=0).T).sum()),
                          n_tris=len(built.tris), dD=built.dD, dC=built.dC, cost=built.cost,
                          first=built.first, exact=built.exact, saving=built.saving,
                          credit=built.credit, offered=len(cands)))
        step += 1
    log(f"{faces.n - n0} faces added by {step} ears")


def world_of(bid: str, h: float) -> EarWorld:
    metric = metrics.metric_of(metrics.RUNTIME, device="cpu")
    [b] = common.build_blocks([bid])
    return EarWorld(gr.growth_of(b, metric, contrast=gr.GroundedLeak(h=h)))


def run(bid: str, plan: Plan) -> None:
    t0 = time.time()
    e = world_of(bid, plan.contrast.h)
    print(f"{bid} {plan.name}: {e.g.nt} triangles, raster {e.ra.shape} ({time.time() - t0:.0f}s)",
          flush=True)
    build(e, plan, out=gr.OUT / bid / plan.name, log=lambda s: print(s, flush=True))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- the checks

CHECK_PLAN = "ears.paths.b20.f2000.e30.loop.gl0.25Croad.n2s0.1.M8.K2.D1"
CHECK_SPUR = "ears.paths.b20.f2000.e30.spur10-80.gl0.25Croad.n2s0.1.M16.K2.D1"
CHECK_DIVIDES = "ears.divides.p0.1.f2000.e30.gl0.25Croad.n2s0.1.M16.K4.D1"
CHECK_DETOUR = "ears.paths.b20.f2000.e30.loop.dt1.gl0.25Croad.n2s0.1.M8.K2.D1"


def _horseshoe(arm_m: float, gap_m: float, *, crow: bool) -> tuple[float, float]:
    """(detour saving S in m, the closing link's length) of the link across a horseshoe's gap:
    a street along y = 10 m, two arms HS_W m apart rising arm_m from it, joined at the top by two
    bars that stop gap_m apart (the network the pixels within PIX of these segments), homes on a
    HS_HOME_M grid over the whole raster; the link joins the network's two tips (on the raster
    about gap_m apart: its length is returned). crow: the `horseshoe` check's fault (the road
    points' distances straight lines)."""
    W, H = 200.0, arm_m + 40.0
    shape = (int(W / PIX), int(H / PIX))
    i, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    xy = np.column_stack([(i.ravel() + 0.5) * PIX, (j.ravel() + 0.5) * PIX])
    x0, x1, top = 100.0 - HS_W / 2, 100.0 + HS_W / 2, 10.0 + arm_m
    segs = shapely.MultiLineString([[(0.0, 10.0), (W, 10.0)], [(x0, 10.0), (x0, top)],
                                    [(x1, 10.0), (x1, top)], [(x0, top), (100.0 - gap_m / 2, top)],
                                    [(x1, top), (100.0 + gap_m / 2, top)]])
    net = shapely.distance(segs, shapely.points(xy)) <= PIX
    ea, eb, el = _grid_edges(shape)
    hx, hy = np.meshgrid(np.arange(HS_HOME_M / 2, W, HS_HOME_M),
                         np.arange(HS_HOME_M / 2, H, HS_HOME_M), indexing="ij")
    homes = np.column_stack([hx.ravel(), hy.ravel()])
    r = roads_of(net, xy, ea, eb, el, homes=homes, share=np.full(len(homes), 1.0 / len(homes)))
    on = np.flatnonzero(net)
    if crow:
        d = xy[r.pts][:, None, :] - xy[on][None, :, :]
        r = r._replace(D=np.hypot(d[..., 0], d[..., 1]).astype(np.float32))
    mid = np.hypot(*(xy[on] - [100.0, top]).T)          # the tips: nearest the gap's middle
    left = xy[on, 0] < 100.0
    a = int(on[left][np.argmin(mid[left])])
    t = int(on[~left][np.argmin(mid[~left])])
    L = float(np.hypot(*(xy[t] - xy[a])))
    return float(savings(r, a, np.array([t]), np.array([L]))[0]), L


def horseshoe(fault: str | None) -> None:
    """Experiment 1 (owner's doc, Idea C) as a check, on synthetic horseshoes (`_horseshoe`):
    with the gap fixed (HS_GAP_M) the closing link's saving S rises strictly with the arms
    (HS_ARMS: closure more likely the longer the horseshoe), and with the arms fixed (HS_ARM_M)
    its saving per metre of link S / l rises strictly as the gap closes (HS_GAPS: a spur growing
    toward the far arm makes the closing ear ever better per cost, so a nearly finished loop
    finishes). Fault `crow` (straight-line distances in place of distances along the roads: no
    trip is shorter by a link, by the triangle inequality) fails it."""
    if fault not in (None, "crow"):
        raise SystemExit("horseshoe's fault: crow")
    crow = fault == "crow"
    by_arm = [_horseshoe(A, HS_GAP_M, crow=crow) for A in HS_ARMS]
    by_gap = [_horseshoe(HS_ARM_M, g, crow=crow) for g in HS_GAPS]
    for A, (S, L) in zip(HS_ARMS, by_arm, strict=True):
        print(f"  arms {A:g} m, gap {HS_GAP_M:g} m: link {L:.1f} m, S {S:.3f} m")
    for g_, (S, L) in zip(HS_GAPS, by_gap, strict=True):
        print(f"  arms {HS_ARM_M:g} m, gap {g_:g} m: link {L:.1f} m, S {S:.3f} m, "
              f"S / l {S / L:.4f}")
    ok = (all(b[0] > a[0] for a, b in zip(by_arm, by_arm[1:], strict=False))
          and all(b[0] / b[1] > a[0] / a[1] for a, b in zip(by_gap, by_gap[1:], strict=False)))
    print(f"horseshoe {fault or 'none'}: -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


def _short(bid: str, h: float, n: int, *, spec: str) -> EarWorld:
    """A world with `n` ears built by the check plan `spec` at leak h."""
    plan = Plan.of(spec)
    e = world_of(bid, h)
    build(e, plan, out=gr.OUT / bid / f"check.{plan.name}", log=lambda s: print(s, flush=True),
          steps=n)
    return e


def check(verb: str, bid: str, fault: str | None) -> None:
    if verb == "decouple":
        if fault not in (None, "weak-leak"):
            raise SystemExit("decouple's fault: weak-leak")
        e = _short(bid, gr.GroundedLeak(h=0.25).h, 3, spec=CHECK_PLAN)
        if fault == "weak-leak":
            e.g.contrast = gr.GroundedLeak(h=WEAK_H)
        on = e.g.step >= 0
        P = e.g.solve(e.g.state(on))[1]
        fc = e.faces()
        cells = np.zeros(e.g.m.c.n, dtype=bool)
        cells[e.g.m.c.cell[fc.lab >= 0]] = True
        Ps = e.P_face(on, cells)
        rel = abs(P - Ps) / P
        ok = rel <= TOL_DEC
        print(f"decouple {bid} {fault or 'none'}: {fc.n} faces, global P {P:.6e}, faces' sum "
              f"{Ps:.6e}, rel {rel:.2e} (tol {TOL_DEC:g}) -> {'PASS' if ok else 'FAIL'}")
    elif verb == "first":
        if fault not in (None, "abs-u"):
            raise SystemExit("first's fault: abs-u")
        e = _short(bid, 0.25, 2, spec=CHECK_PLAN)
        g = e.g
        g.contrast = gr.GroundedLeak(h=FIRST_H)
        on = g.step >= 0
        st = g.state(on)
        u, P = g.solve(st)
        gain = g.contrast.gains(g, st, u)
        if fault == "abs-u":
            gk, _ = g.grad(st, u)
            gain = -gk * (1.0 - g.kt0) + np.abs(u[g.m.c.cell]) * g.contrast.leak_t(g)
        fc = e.faces()
        cands = Paths(bld=20.0, pairs=6, face_m2=2000.0, ear_m=30.0, ends=Loop(),
                      attract=NoDetour()).ears(
            e, fc, gain, cost=gr.Cost(nu=2.0, price=0.1), scale=0.0)
        worst = 0.0
        for q in cands[:4]:
            tris = q.tris[g.kt0[q.tris] >= OPEN_KT]
            on2 = on.copy()
            on2[tris] = True
            ex = P - g.solve(g.state(on2))[1]
            first = float(gain[tris].sum())
            rel = abs(first - ex) / abs(ex)
            worst = max(worst, rel)
            print(f"  ear of {len(tris)} open triangles: first order {first:.6e}, exact {ex:.6e}, "
                  f"rel {rel:.2e}")
        ok = worst <= TOL_FIRST
        print(f"first {bid} {fault or 'none'}: worst rel {worst:.2e} (tol {TOL_FIRST:g}) -> "
              f"{'PASS' if ok else 'FAIL'}")
    elif verb == "ends":
        if fault not in (None, "spur"):
            raise SystemExit("ends' fault: spur")
        e = _short(bid, 0.25, 0, spec=CHECK_PLAN)
        g = e.g
        st = g.state(g.step >= 0)
        u, _ = g.solve(st)
        fc = e.faces()
        cands = Paths(bld=20.0, pairs=4, face_m2=2000.0, ear_m=30.0, ends=Loop(),
                      attract=NoDetour()).ears(
            e, fc, g.contrast.gains(g, st, u), cost=gr.Cost(nu=2.0, price=0.1), scale=0.0)
        ok = bool(cands)
        for q in cands[:4]:
            path = q.path[: max(1, 2 * len(q.path) // 3)] if fault == "spur" else q.path
            tris = e.corridor(e.ra.xy[path], spur=False)
            n0 = fc.n
            g.step[tris] = 0
            n1 = e.faces().n
            g.step[tris] = -1
            split = n1 > n0
            ok &= split
            print(f"  ear of {len(path)} pixels: faces {n0} -> {n1}: "
                  f"{'splits' if split else 'a dead end'}")
        print(f"ends {bid} {fault or 'none'}: -> {'PASS' if ok else 'FAIL'}")
    elif verb == "spur":
        if fault not in (None, "both-ends"):
            raise SystemExit("spur's fault: both-ends")
        plan = Plan.of(CHECK_SPUR)
        e = _short(bid, plan.contrast.h, 0, spec=CHECK_SPUR)
        g = e.g
        st = g.state(g.step >= 0)
        u, _ = g.solve(st)
        fc = e.faces()
        spurs = [q for q in plan.candidates.ears(e, fc, g.contrast.gains(g, st, u),
                                                 cost=plan.cost, scale=0.0) if q.spur]
        bounding = e.bounding()
        ok = bool(spurs)
        for q in spurs[:SPUR_EARS]:
            tris = e.corridor(e.ra.xy[q.path], spur=fault != "both-ends")
            on = np.zeros(g.nt, dtype=bool)
            on[tris] = True
            inner = on[g.pa] & on[g.pb]
            pieces = connected_components(sp.coo_matrix(
                (np.ones(int(inner.sum())), (g.pa[inner], g.pb[inner])), shape=(g.nt, g.nt)),
                directed=False)[1][tris]
            joined = bool(((on[g.pa] & bounding[g.pb]) | (on[g.pb] & bounding[g.pa])).any())
            g.step[tris] = 0
            n1 = e.faces().n
            g.step[tris] = -1
            good = len(np.unique(pieces)) == 1 and joined and n1 == fc.n
            ok &= good
            L = float(np.hypot(*np.diff(e.ra.xy[q.path], axis=0).T).sum())
            print(f"  spur of {L:.0f} m: {len(np.unique(pieces))} piece(s), "
                  f"{'joined' if joined else 'detached'}, faces {fc.n} -> {n1}")
        print(f"spur {bid} {fault or 'none'}: {len(spurs)} spurs among the candidates -> "
              f"{'PASS' if ok else 'FAIL'}")
    elif verb == "detour":
        if fault not in (None, "one-way"):
            raise SystemExit("detour's fault: one-way")
        plan = Plan.of(CHECK_DETOUR)
        pa = plan.candidates
        assert isinstance(pa, Paths)
        e = _short(bid, plan.contrast.h, DETOUR_EARS, spec=CHECK_PLAN)
        g, ra = e.g, e.ra
        st = g.state(g.step >= 0)
        u, _ = g.solve(st)
        fld = pa.attract.field(e, P0=g.P0)
        assert isinstance(fld, _DetourField)
        rd = fld.roads
        net = e.sinks()
        col, G = _net_graph(net, ra.ea, ra.eb, ra.el)
        DRR = rd.among()
        fin = np.isfinite(DRR)
        ww = np.outer(rd.w, rd.w)

        def brute(a: int, t: int, L: float) -> float:
            """All pairs again, the link an edge of the network's graph."""
            link = sp.coo_matrix(([L], ([col[a]], [col[t]])), shape=G.shape).tocsr()
            D2 = dijkstra(G + link, directed=False, indices=col[rd.pts])[:, col[rd.pts]]
            return float((ww * np.where(fin, DRR - D2, 0.0)).sum())

        half = 0.5 if fault == "one-way" else 1.0  # the trips counted from a's side only
        cands = [q for q in pa.ears(e, e.faces(), g.contrast.gains(g, st, u), cost=plan.cost,
                                    scale=0.0) if not q.spur]
        judged: list[tuple[str, float, float]] = []
        for q in cands:
            L = float(np.hypot(*np.diff(ra.xy[q.path], axis=0).T).sum())
            judged.append((f"ear of {L:.0f} m (credit {q.credit / g.P0:.3g} P0)", half * q.saving,
                           brute(int(q.path[0]), int(q.path[-1]), L)))
        a0 = int(cands[0].path[0])
        on = np.flatnonzero(net & np.isfinite(rd.D[0, np.maximum(col, 0)]) & (col >= 0))
        tg = np.random.default_rng(DETOUR_SEED).choice(on, size=DETOUR_TARGETS, replace=False)
        Ls = DETOUR_STRETCH * np.hypot(*(ra.xy[tg] - ra.xy[a0]).T) + PIX
        fast = savings(rd, a0, tg, Ls)
        for t, L, f in zip(tg, Ls, fast, strict=True):
            judged.append((f"link of {L:.0f} m from the first ear's source", half * float(f),
                           brute(a0, int(t), float(L))))
        top = max(b for _, _, b in judged)
        worst = max(abs(f - b) for _, f, b in judged) / top
        for what, f, b in judged:
            print(f"  {what}: S {f:.4f} m, all pairs again {b:.4f} m")
        ok = top > 0 and worst <= TOL_DETOUR
        print(f"detour {bid} {fault or 'none'}: {len(rd.pts)} road points, worst |S - all pairs| "
              f"{worst:.2e} of the largest S (tol {TOL_DETOUR:g}) -> {'PASS' if ok else 'FAIL'}")
    elif verb == "offer":
        if fault not in (None, "persistence"):
            raise SystemExit("offer's fault: persistence")
        plan = Plan.of(CHECK_DIVIDES)
        dv = plan.candidates
        assert isinstance(dv, Divides)
        e = _short(bid, plan.contrast.h, OFFER_EARS, spec=CHECK_DIVIDES)
        g = e.g
        st = g.state(g.step >= 0)
        u, _ = g.solve(st)
        e.set_u(u)
        gain = g.contrast.gains(g, st, u)
        fc = e.faces()
        every = dv.offers(e, fc, gain)
        if fault == "persistence":                 # rev 1's order: relative persistence
            every = sorted(every, key=lambda o: -o.divide.persistence)
        best = [0.0, 0.0]                          # splitting first order per cost: costed, all
        for k, q in enumerate(Divides.costed(e, every, gain, cost=plan.cost)):
            g.step[q.tris] = 0
            split = e.faces().n > fc.n
            g.step[q.tris] = -1
            if split:
                r = q.first / q.cost
                best[1] = max(best[1], r)
                if k < dv.pairs:
                    best[0] = max(best[0], r)
        ratio = best[0] / best[1]
        ok = ratio >= TOL_OFFER
        print(f"offer {bid} {fault or 'none'}: {fc.n} faces, {len(every)} divides; the costed "
              f"{dv.pairs}' best splitting first order per cost {best[0]:.4g}, all divides' "
              f"{best[1]:.4g}: ratio {ratio:.2f} (tol {TOL_OFFER:g}) -> {'PASS' if ok else 'FAIL'}")
    else:
        raise SystemExit(__doc__)
    if not ok:
        raise SystemExit(1)


def main(argv: list[str]) -> None:
    if len(argv) == 3 and argv[0] == "run":
        run(argv[1], Plan.of(argv[2]))
    elif len(argv) in (2, 3) and argv[:2] == ["check", "horseshoe"]:
        horseshoe(argv[2] if len(argv) == 3 else None)
    elif len(argv) in (3, 4) and argv[0] == "check":
        check(argv[1], argv[2], argv[3] if len(argv) == 4 else None)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
