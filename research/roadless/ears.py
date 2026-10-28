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
m, to any other sink pixel of the same face at least MIN_EAR_M away in a straight line: per
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

    E=research/roadless/ears.py                        # from the repo root, PYTHONPATH=.
    uv run python $E run <block> <plan>
    uv run python $E check decouple|first|ends|offer <block> [fault]

plan: ears.<paths.b<bld>|divides.p<tau>>.gl<h>Croad.n<nu>s<price>.M<pairs>.K<exact>.D<D>, its
name adding the
code's revision (.v<REV>); rows in growth_rows/<block>/<name>/ in growth.py's layout
(steps.parquet, picks.parquet one row per ear, bridges.parquet empty, tri_step.parquet), so the
growth runs' frames and plates read them.

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
MIN_EAR_M = 30.0       # m: an ear's ends at least this far apart in a straight line
MIN_FACE_M2 = 2000.0   # m^2: a face this small is not split further
STRETCH_M = 40.0       # m: divides' harmonic split: the face's boundary in stretches this wide
DUP_M = 2 * PIX        # m: Divides: a divide's pixel this near a kept divide's is shared
DUP_SHARE = 0.5        # Divides: a divide sharing this much of its pixels with a kept one dropped
BLD_G = 0.01           # divides' harmonic split: a pixel edge's conductance through a building
STEP_M = 0.25          # m: an ear's centreline sampled this finely (EarWorld.corridor)
LOCATE_K = 12          # EarWorld.locate: candidate triangles per point
CHUNK = 32             # sources per Dijkstra call (its distance and predecessor rows held)
REV = 2                # 2: Divides offers by first-order gain per metre, near-duplicates dropped
                       # (rev 1: the most persistent across faces)
TOL_DEC = 1e-2         # decouple: relative (19570, 3 ears at h 0.25: 2.4e-3; at h 25, 0.24)
DEC_EVERY = 10         # build: the decoupling gap measured every this many ears
TOL_FIRST = 2e-3       # first: relative (19570: 3.3e-4 at FIRST_H; 3.3e-2 at h 250, the drop
                       # not yet linear)
WEAK_H = 25.0          # m: decouple's fault: a weak leak
FIRST_H = 2500.0       # m: first's leak (the drop linear in it: an ear's leak times its own
                       # resistance to ground far below 1)
OPEN_KT = 0.999        # first: the ears' open triangles only (no opening term to speak of)
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
    """A candidate: its face, its path's pixels, its corridor's new triangles, dD, dC, its cost,
    first-order gain, exact gain (nan unless rescored)."""
    face: int
    path: I64
    tris: I64
    dD: float
    dC: float
    cost: float
    first: float
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
    ea, eb, el = [], [], []
    idx = np.arange(shape[0] * shape[1]).reshape(shape)
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        j0, j1 = max(0, -dj), shape[1] - max(0, dj)
        a = idx[0:shape[0] - di, j0:j1]
        b = idx[di:shape[0], j0 + dj:j1 + dj]
        ea.append(a.ravel())
        eb.append(b.ravel())
        el.append(np.full(a.size, PIX * np.hypot(di, dj)))
    return Raster(shape=shape, lo=lo, xy=xy, tri=tri, bld=bld, ea=np.concatenate(ea),
                  eb=np.concatenate(eb), el=np.concatenate(el))


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

    def ears(self, e: EarWorld, faces: Face, gain: F64, *, cost: gr.Cost) -> list[Ear]: ...


@dataclass(frozen=True, kw_only=True)
class Paths:
    """Cheapest paths on the raster (module docstring); `bld` the extra cost of a metre through a
    building, `pairs` the sources whose best ears are built and costed."""
    bld: float
    pairs: int

    @property
    def name(self) -> str:
        return f"paths.b{self.bld:g}"

    def ears(self, e: EarWorld, faces: Face, gain: F64, *, cost: gr.Cost) -> list[Ear]:
        g, ra = e.g, e.ra
        pf = np.where(ra.tri >= 0, faces.lab[np.maximum(ra.tri, 0)], -1)    # pixel's face
        big = faces.area >= MIN_FACE_M2
        live = (pf >= 0) & big[np.maximum(pf, 0)]
        # the ends: sink pixels of the face (a road or the street, not a wall)
        same = pf[ra.ea] == pf[ra.eb]
        bnd = live & e.sinks()
        # per pixel its first-order gain density (the triangles under it)
        gp = np.bincount(_pix_of(ra, g.cen), weights=gain, minlength=len(pf))
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
                far = np.hypot(*(ra.xy - ra.xy[a]).T) >= MIN_EAR_M
                tg = np.flatnonzero(bnd & fin & far & (pf == pf[a]))
                if not len(tg):
                    continue
                r = acc[tg] / np.maximum(d[tg], 1e-9)
                t = int(tg[np.argmax(r)])
                best.append((float(r.max()), int(a), t, pred[k].copy()))
        best.sort(key=lambda q: -q[0])
        out = []
        for _, a, t, pr in best[:self.pairs]:
            path = [t]
            while path[-1] != a:
                path.append(int(pr[path[-1]]))
            path_a = np.asarray(path[::-1], dtype=np.int64)
            tris = e.corridor(ra.xy[path_a])
            if not len(tris):
                continue
            dD, dC = g.costs_of(tris)
            c = float(cost.of(np.array([dD]), np.array([dC]), P0=g.P0)[0])
            out.append(Ear(face=int(pf[a]), path=path_a, tris=tris, dD=dD, dC=dC, cost=c,
                           first=float(gain[tris].sum()), exact=np.nan))
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
            tau: float, stretch_m: float) -> list[Divide]:
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
    least MIN_EAR_M away that keeps highest in u (the network's edges weighted by length over
    u, a shortest path), so every cut runs sink to sink (no dead end)."""
    on = lab == face
    if not on.any():
        return []
    both = on[ea] & on[eb]
    a, b = ea[both], eb[both]
    edge = on & sink
    umax = float(u[on].max())
    out = _cuts(u, xy, _ms_cut(u, on, edge, a, b, tau=tau), edge, a, b, umax=umax, kind="ms")
    if not out:
        out = _cuts(u, xy, _harmonic_cut(xy, bld, on, edge, a, b, stretch_m=stretch_m), edge,
                    a, b, umax=umax, kind="harmonic")
    out.sort(key=lambda d: -d.persistence)
    return out


def _cuts(u: F64, xy: F64, cut: B, edge: B, a: I64, b: I64, *, umax: float,
          kind: str) -> list[Divide]:
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
            far = np.hypot(*(xy[ends] - xy[s0]).T) >= MIN_EAR_M
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
    """The divides of u (`divides`), per face of at least MIN_FACE_M2 on its own u (the state's,
    the roads grounded), at persistence threshold `tau`, the harmonic-measure split where the
    strict basins degenerate, near-duplicates dropped (`_distinct`); across faces the `pairs`
    with the most first-order gain per metre costed (`offers`). Rev 1 took the `pairs` most
    persistent across faces, but persistence is relative to its face's largest u, so on 5810 the
    small faces' divides (u 2 -- 60) filled the pairs and the large faces' (u 160 -- 230, 7 -- 1000
    times the costed ones' first-order gain per cost) were never costed."""
    tau: float
    pairs: int

    @property
    def name(self) -> str:
        return f"divides.p{self.tau:g}"

    def offers(self, e: EarWorld, faces: Face, gain: F64) -> list[Offer]:
        """Every face's distinct divides, the most first-order gain per metre first."""
        g, ra = e.g, e.ra
        pf = np.where(ra.tri >= 0, faces.lab[np.maximum(ra.tri, 0)], -1)
        gp = np.bincount(_pix_of(ra, g.cen), weights=gain, minlength=len(pf))
        sink = e.sinks()
        out: list[Offer] = []
        for f in np.flatnonzero(faces.area >= MIN_FACE_M2):
            ds = divides(e.u_pix, pf, sink, ra.ea, ra.eb, ra.xy, ra.bld, face=int(f),
                         tau=self.tau, stretch_m=STRETCH_M)
            for d in _distinct(ds, ra.xy):
                L = float(np.hypot(*np.diff(ra.xy[d.pixels], axis=0).T).sum())
                out.append(Offer(face=int(f), divide=d,
                                 per_m=float(gp[d.pixels].sum()) / max(L, PIX)))
        out.sort(key=lambda o: -o.per_m)
        return out

    def ears(self, e: EarWorld, faces: Face, gain: F64, *, cost: gr.Cost) -> list[Ear]:
        return self.costed(e, self.offers(e, faces, gain)[:self.pairs], gain, cost=cost)

    @staticmethod
    def costed(e: EarWorld, offers: list[Offer], gain: F64, *, cost: gr.Cost) -> list[Ear]:
        g, ra = e.g, e.ra
        out = []
        for o in offers:
            f, d = o.face, o.divide
            tris = e.corridor(ra.xy[d.pixels])
            if not len(tris):
                continue
            dD, dC = g.costs_of(tris)
            c = float(cost.of(np.array([dD]), np.array([dC]), P0=g.P0)[0])
            out.append(Ear(face=f, path=d.pixels, tris=tris, dD=dD, dC=dC, cost=c,
                           first=float(gain[tris].sum()), exact=np.nan))
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
        m = re.fullmatch(r"ears\.(paths\.b|divides\.p)([0-9.]+)\.gl([0-9.]+)Croad"
                         r"\.n([0-9.e-]+)s([0-9.e-]+)\.M([0-9]+)\.K([0-9]+)\.D([0-9.]+)", spec)
        if m is None:
            raise SystemExit(f"{spec}: a plan is ears.<paths.b<bld>|divides.p<tau>>.gl<h>Croad"
                             ".n<nu>s<price>.M<pairs>.K<exact>.D<D>")
        cands: Candidates = (Paths(bld=float(m[2]), pairs=int(m[6])) if m[1] == "paths.b"
                             else Divides(tau=float(m[2]), pairs=int(m[6])))
        return Plan(candidates=cands, contrast=gr.GroundedLeak(h=float(m[3])),
                    cost=gr.Cost(nu=float(m[4]), price=float(m[5])), exact=int(m[7]),
                    D=float(m[8]))


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
        g = self.g
        att = np.flatnonzero(self.bounding())
        d, _ = cKDTree(g.cen[att]).query(self.ra.xy, k=1, distance_upper_bound=PIX)
        return np.asarray(np.isfinite(d) & (self.ra.tri >= 0))

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

    def corridor(self, pts: F64) -> I64:
        """An ear's corridor along the polyline `pts`, extended at both ends to the nearest
        bounding triangle (the road or street it joins): the ungrown triangles within
        growth.PATCH_R of its points, and every triangle its centreline crosses (sampled every
        STEP_M): on 5810's open ground a triangle can be larger than a road, and one whose
        centroid lies off the corridor but which straddles it joins the two halves (the first
        runs' ears split nothing)."""
        g = self.g
        att = np.flatnonzero(self.bounding())
        _, k = cKDTree(g.cen[att]).query(pts[[0, -1]], k=1)
        line = np.vstack([g.cen[att[k[0]]], pts, g.cen[att[k[1]]]])
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
        cands = plan.candidates.ears(e, faces, gain, cost=plan.cost)
        if not cands:
            log("no ear left: stopped short of the budget")
            break
        cands.sort(key=lambda q: -q.first / q.cost)
        on = g.step >= 0
        # the splitting ones only (no dead end), best first order per cost first, `exact` of them
        split: list[tuple[Ear, Face]] = []
        for q in cands:
            if len(split) == plan.exact:
                break
            g.step[q.tris] = step
            nf = e.faces()
            g.step[q.tris] = -1
            if nf.n > faces.n:
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
        built, faces = max(scored, key=lambda qn: qn[0].exact / qn[0].cost)
        g.step[built.tris] = step
        xy = e.ra.xy[built.path]
        picks.append(dict(step=step, face=built.face, ax=float(xy[0, 0]), ay=float(xy[0, 1]),
                          bx=float(xy[-1, 0]), by=float(xy[-1, 1]),
                          length=float(np.hypot(*np.diff(xy, axis=0).T).sum()),
                          n_tris=len(built.tris), dD=built.dD, dC=built.dC, cost=built.cost,
                          first=built.first, exact=built.exact,
                          offered=len(cands)))
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

CHECK_PLAN = "ears.paths.b20.gl0.25Croad.n2s0.1.M8.K2.D1"
CHECK_DIVIDES = "ears.divides.p0.1.gl0.25Croad.n2s0.1.M16.K4.D1"


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
        cands = Paths(bld=20.0, pairs=6).ears(e, fc, gain, cost=gr.Cost(nu=2.0, price=0.1))
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
        cands = Paths(bld=20.0, pairs=4).ears(e, fc, g.contrast.gains(g, st, u),
                                              cost=gr.Cost(nu=2.0, price=0.1))
        ok = bool(cands)
        for q in cands[:4]:
            path = q.path[: max(1, 2 * len(q.path) // 3)] if fault == "spur" else q.path
            tris = e.corridor(e.ra.xy[path])
            n0 = fc.n
            g.step[tris] = 0
            n1 = e.faces().n
            g.step[tris] = -1
            split = n1 > n0
            ok &= split
            print(f"  ear of {len(path)} pixels: faces {n0} -> {n1}: "
                  f"{'splits' if split else 'a dead end'}")
        print(f"ends {bid} {fault or 'none'}: -> {'PASS' if ok else 'FAIL'}")
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
    elif len(argv) in (3, 4) and argv[0] == "check":
        check(argv[1], argv[2], argv[3] if len(argv) == 4 else None)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
