"""The gap complex: a triangulation of the open ground between footprints, its passages measured
once from the outlines (NOTES, "The gap complex: passability, enclosure and flow without the
raster"; BACKLOG 4, decision (3), the passable width).

THE COMPLEX. The open ground is the block less its footprints and less the street band (the
raster's ground: within `band_m` of Block.streets, lifted.Grid.of's `ground`). Its outline, every
footprint's and the band's inner line, is densified to at most `s` between vertices
(shapely.segmentize) and triangulated by GEOS's constrained Delaunay triangulation
(shapely.constrained_delaunay_triangles): every vertex lies on an outline, every outline segment
is a triangle's edge, so a triangle is a pocket of open ground and an edge two triangles share is
a passage, spanning open ground from one outline to another. Two triangles meet across such an
edge or not at all: a phantom passage (the raster's old defect) cannot occur.

A passage's APERTURE is its length when both its ends lie on footprints: the width of the gap
there (an edge shorter than w blocks a disk of diameter w, whose centre crossing it would come
within w / 2 of an end). A passage with an end on the band's line (a street vertex, on no
footprint) is no constriction: beyond the line is the street, open ground, and its aperture is
inf. A triangle with an edge on the band's line is an EXIT: the walker there has reached the
street. A building's own concavities are passages like any other (their ends on one outline).

THE FILTRATION. At width w a passage is open iff its aperture >= w. A triangle's EXIT WIDTH is
the widest path's bottleneck to the street, max over paths of the narrowest passage on it (the
maximum spanning forest, `exit_widths`): it is walled in from the street exactly for w above it
(-inf: no path at any width; inf: a street triangle). Each triangle's FILL width is its own largest
clearance (2 x its circumradius if acute, else its longest edge: the Delaunay radius function),
and a walled-in pocket is a BAR from its exit width to its fill, the largest fill over the
triangles cut off together (`Bars`).

HOMES (lifted.Placement's test on the complex). A home is live at w when some open ground within
`ring_m` of its outline reaches the street at w: a triangle within ring_m of it whose exit width
is at least w, or the open street band within ring_m of it (the raster's ground cells, open
ground there, are its street). Distances are the polygons' own (shapely), so as in the raster a
ring reaches through a thin wall.

FREE FIRST (`free_first`, clear.free_first on the complex). At width w: the region graph's nodes
the components of the triangles joined by open passages (the street's the one holding the exits)
and the standing buildings; a building is adjacent to every component with a triangle touching its
outline at a vertex on no other footprint's (clearing it opens its interior and frees the passages
that end on it; a vertex where two footprints meet stays an obstacle while either stands), to the
buildings it intersects, and to the street where it reaches into the band. Each round one
Dijkstra from the street, entering a building costing its population share, picks the crossing
with the least D per unit of the home's population, as clear.free_first does. A pass ends when the
graph frees every walled-in home or none it can afford; the complex is then rebuilt without the
held buildings and the metric's own test (`walled`) re-run, another pass taking any home the graph
freed wrongly. Clearing a building can open a passage no triangle of the old complex shows (two
buildings close across its former footprint), so the graph is generous and the rebuild decides.

The run's identity is `Spec` (s, ring_m, band_m: required, keyword-only, its name in every row);
the width w and the budget are each call's, required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import shapely
from numba import njit
from numpy.typing import NDArray
from scipy.sparse.csgraph import breadth_first_order, connected_components, dijkstra

TOL = 1e-6           # metres: a vertex on an outline (in the block's local coordinates the
                     # overlay's and segmentize's vertices sit within 1e-14 of it; in UTM
                     # segmentize's sat up to 9e-6 off, so every geometry is moved first)
FREE_STEP = 1e-12    # clear.FREE_STEP: every step through the region graph costs this more
DUST_M2 = 1e-6       # a part of the open ground smaller than this (m^2) is the overlay's dust
                     # (slivers 1e-15 -- 1e-8 m^2 between footprints that share a wall, where
                     # GEOS's triangles fall inside the footprints): dropped, holding no one
DELAUNAY_TOL = 1e-9  # a passage is flipped when its two triangles' circumcentres lie in the wrong
                     # order along its normal by more than this share of its length: (cot a +
                     # cot b) / 2 < -DELAUNAY_TOL, a and b the angles facing it (an angle sum's
                     # tolerance would let a sliver's centre run far back: d = L / 2 (cot a + cot
                     # b) = -L sin(eps) / (2 sin a sin b))
GROW_TOL_REL = 0.02  # a grown outline lies within this share of the growth of its offset: GEOS
                     # simplifies a buffer's input by 0.01 of the distance (a grown edge's
                     # midpoint was found 0.2507 m off a footprint grown by 0.25)
SNAP_M = 1e-9        # vertices snapped to this grid (m): GEOS repeats a vertex where two
                     # outlines meet with the last bits apart, which made zero-area triangles;
                     # a triangle left with a repeated vertex is dropped


@dataclass(frozen=True, kw_only=True)
class Spec:
    """The complex's identity: the outlines' spacing `s` (m), the demand ring `ring_m`
    (lifted.RING_M in the metric) and the street band `band_m` (lifted.BAND_M)."""
    s: float
    ring_m: float
    band_m: float

    @property
    def name(self) -> str:
        return f"gc.s{self.s:g}.r{self.ring_m:g}.b{self.band_m:g}"


class Geometry(NamedTuple):
    """A block's geometry in local coordinates (`origin` subtracted): its polygon, its
    footprints in the block's outline order, and its streets' union."""
    boundary: object
    polys: NDArray[np.object_]
    streets: object
    origin: tuple[float, float]


def geometry(block) -> Geometry:
    x0, y0 = block.boundary.bounds[:2]

    def local(g):
        return shapely.transform(g, lambda xy: xy - np.array([x0, y0]))
    return Geometry(boundary=local(block.boundary),
                    polys=local(np.asarray(block.buildings.outlines)),
                    streets=local(shapely.union_all(list(block.streets.geometry))),
                    origin=(float(x0), float(y0)))


@dataclass(frozen=True, eq=False)
class Complex:
    """One block's gap complex (`build`), for the standing buildings `standing` (positions into
    the block's outlines). Arrays on the host; `tri_geom` the triangles as shapely polygons."""
    spec: Spec
    xy: NDArray[np.float64]          # (nv, 2) vertices
    tri: NDArray[np.int64]           # (nt, 3) vertex ids, counter-clockwise
    tri_geom: NDArray[np.object_]    # (nt,) shapely triangles
    area: NDArray[np.float64]        # (nt,)
    fill: NDArray[np.float64]        # (nt,) the triangle's own largest clearance (m)
    pa: NDArray[np.int64]            # (ne,) passages: the two triangles
    pb: NDArray[np.int64]
    pu: NDArray[np.int64]            # (ne,) the passage's two vertices
    pv: NDArray[np.int64]
    ap: NDArray[np.float64]          # (ne,) aperture (m), inf with an end on the street line
    exit: NDArray[np.bool_]          # (nt,) an edge on the street line
    street_faces: NDArray[np.int64]  # (nf, 2) the edges on the street line, sorted vertex pairs
    street_v: NDArray[np.bool_]      # (nv,) on the street line and no footprint
    vb: sp.csr_matrix                # (nv, n buildings) 0/1: the vertex on building j's outline
    standing: NDArray[np.bool_]      # (n buildings,)
    band_open: object                # the open street band (shapely), the raster's ground

    @property
    def nt(self) -> int:
        return len(self.tri)


def obstacles(geo: Geometry, standing: NDArray[np.bool_], *, carve,
              grow: float) -> tuple[NDArray[np.int64], NDArray[np.object_]]:
    """(the standing buildings' positions, their footprints less `carve`, the empty ones left
    out): what bounds the open ground before growing by `grow`."""
    idx = np.flatnonzero(standing)
    polys = geo.polys[idx] if carve is None else shapely.difference(geo.polys[idx], carve)
    keep = ~shapely.is_empty(polys)
    return idx[keep], polys[keep]


def _open_ground(geo: Geometry, standing: NDArray[np.bool_], band_m: float, *, carve,
                 grow: float) -> tuple[object, object]:
    """(the open ground beyond the band, the open band) of a block with the buildings
    `standing` less `carve` (a corridor, common.Carve's rule: exactly it freed; None: nothing),
    grown by `grow` (the shy distance's w / 2; 0: the footprints as they are): the block less
    them, split at band_m from the streets (the raster's ground: cell centres within band_m of
    Block.streets, lifted.Grid.of; the band is not grown into: it is the street)."""
    _idx, polys = obstacles(geo, standing, carve=carve, grow=grow)
    fu0 = shapely.union_all(polys) if len(polys) else shapely.Polygon()
    fu = fu0.buffer(grow) if grow > 0 else fu0
    band = geo.streets.buffer(band_m)
    inner = geo.boundary.difference(band).difference(fu)
    band_open = geo.boundary.intersection(band).difference(fu0)
    return inner, band_open


def _attempts(part, s: float):
    """The polygons a part's triangulation tries in turn: its rings densified to s; then snapped
    to SNAP_M after densifying, or before (GEOS's ear clipping fails on some near-collinear
    corners the overlay leaves a hair apart, "Unable to find a convex corner", and its hole
    joiner on some snapped ones, "Unable to find shell join index")."""
    yield shapely.segmentize(part, s)
    yield shapely.set_precision(shapely.segmentize(part, s), SNAP_M)
    yield shapely.segmentize(shapely.set_precision(part, SNAP_M), s)


def triangulated(inner, s: float) -> list[tuple[object, NDArray[np.object_]]]:
    """Per polygon of `inner` of DUST_M2 and above: (the densified polygon triangulated, GEOS's
    constrained Delaunay triangles of it), the first of `_attempts` GEOS takes. GEOS stops
    improving after a fixed number of scans, so a few passages may remain non-Delaunay
    (`_flips` finishes the job)."""
    out = []
    for part in shapely.get_parts(inner):
        if part.area < DUST_M2:
            continue
        err = None
        for q in _attempts(part, s):
            try:
                tris = shapely.get_parts(shapely.constrained_delaunay_triangles(q))
            except shapely.errors.GEOSException as e:
                err = e
                continue
            out += [(qq, tris) for qq in [q]]
            err = None
            break
        if err is not None:
            raise err
    return out


def _triangulate(inner, s: float) -> NDArray[np.object_]:
    out = [tris for _q, tris in triangulated(inner, s)]
    return np.concatenate(out) if out else np.zeros(0, dtype=object)


@njit(cache=True)
def _fill_of(p0, p1, p2):
    """Per triangle: 2 x circumradius if acute, else its longest edge (the Delaunay radius
    function's value, as a width)."""
    n = p0.shape[0]
    out = np.empty(n)
    for i in range(n):
        a2 = (p1[i, 0] - p2[i, 0]) ** 2 + (p1[i, 1] - p2[i, 1]) ** 2
        b2 = (p0[i, 0] - p2[i, 0]) ** 2 + (p0[i, 1] - p2[i, 1]) ** 2
        c2 = (p0[i, 0] - p1[i, 0]) ** 2 + (p0[i, 1] - p1[i, 1]) ** 2
        m = max(a2, b2, c2)
        if m >= a2 + b2 + c2 - m:            # right or obtuse: the longest edge
            out[i] = math.sqrt(m)
        else:
            cross = abs((p1[i, 0] - p0[i, 0]) * (p2[i, 1] - p0[i, 1])
                        - (p1[i, 1] - p0[i, 1]) * (p2[i, 0] - p0[i, 0]))
            out[i] = math.sqrt(a2 * b2 * c2) / cross      # 2R = abc / (2 area) = abc / cross
    return out


def _neighbours(tri: NDArray[np.int64], nv: int) -> NDArray[np.int64]:
    """(nt, 3): the triangle across the edge opposite each vertex (-1 on the outline)."""
    nt = len(tri)
    e = np.stack([tri[:, [1, 2]], tri[:, [2, 0]], tri[:, [0, 1]]], axis=1).reshape(-1, 2)
    es = np.sort(e, axis=1)
    key = es[:, 0] * nv + es[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    same = ks[1:] == ks[:-1]
    if (ks[2:] == ks[:-2]).any():
        raise ValueError("an edge in three triangles")
    i0, i1 = order[:-1][same], order[1:][same]
    nbr = -np.ones(3 * nt, dtype=np.int64)
    nbr[i0] = i1 // 3
    nbr[i1] = i0 // 3
    return nbr.reshape(nt, 3)


@njit(cache=True)
def _cot(o, u, v):
    """The cotangent of the angle at o of the counter-clockwise triangle (o, u, v) (-inf for a
    flat one with o between u and v, inf with o beyond them)."""
    ax, ay, bx, by = u[0] - o[0], u[1] - o[1], v[0] - o[0], v[1] - o[1]
    cr = ax * by - ay * bx
    dot = ax * bx + ay * by
    if cr <= 0.0:
        return -math.inf if dot < 0.0 else math.inf
    return dot / cr


@njit(cache=True)
def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


@njit(cache=True)
def flips_all(xy, tri, nbr, con, tol):
    """`_flips` with the constraint flags given (`legalize`'s con)."""
    stack = [(0, 0)]
    stack.pop()
    for t in range(tri.shape[0]):
        for i in range(3):
            if nbr[t, i] > t:
                stack.append((t, i))
    return legalize(xy, tri, nbr, con, stack, tol)


@njit(cache=True)
def _flips(xy, tri, nbr, tol):
    """Lawson's flips, in place, until no passage's circumcentres lie in the wrong order by more
    than tol of its length (DELAUNAY_TOL) where its quadrilateral is strictly convex (counter-clockwise triangles; nbr[t, i] the triangle across
    the edge opposite tri[t, i], -1 on the outline: the constraints, never flipped). Returns the
    number of flips."""
    nt = tri.shape[0]
    stack = [(0, 0)]
    stack.pop()
    for t in range(nt):
        for i in range(3):
            if nbr[t, i] > t:
                stack.append((t, i))
    return legalize(xy, tri, nbr, nbr < 0, stack, tol)


@njit(cache=True)
def legalize(xy, tri, nbr, con, stack, tol):
    """_flips's loop from the edges on `stack` ((triangle, local edge) pairs; consumed), never
    flipping an edge flagged in `con` (nt, 3: per local edge, as nbr; the outline, and any edge
    to be kept), the flags carried through the flips."""
    nflip = 0
    while len(stack):
        t, i = stack.pop()
        u = nbr[t, i]
        if u < 0 or con[t, i]:
            continue
        j = -1
        for k in range(3):
            if nbr[u, k] == t:
                j = k
        if j < 0:
            continue
        p, b, c = tri[t, i], tri[t, (i + 1) % 3], tri[t, (i + 2) % 3]
        q = tri[u, j]
        if 0.5 * (_cot(xy[p], xy[b], xy[c]) + _cot(xy[q], xy[c], xy[b])) >= -tol:
            continue
        # strictly convex quadrilateral p, b, q, c (counter-clockwise)
        if not (_orient(xy[p], xy[b], xy[q]) > 0 and _orient(xy[q], xy[c], xy[p]) > 0):
            continue
        # t = (p, b, c), u = (q, c, b): the outer neighbours
        n_pb = nbr[t, (i + 2) % 3]      # t's edge (p, b), opposite c
        n_cp = nbr[t, (i + 1) % 3]      # t's edge (c, p), opposite b
        jb = -1
        jc = -1
        for k in range(3):
            if tri[u, k] == b:
                jb = k
            if tri[u, k] == c:
                jc = k
        n_qc = nbr[u, jb]               # u's edge (q, c), opposite b
        n_bq = nbr[u, jc]               # u's edge (b, q), opposite c
        c_pb, c_cp = con[t, (i + 2) % 3], con[t, (i + 1) % 3]
        c_qc, c_bq = con[u, jb], con[u, jc]
        # t' = (p, b, q): opposite p (b, q) n_bq, opposite b (q, p) u', opposite q (p, b) n_pb
        tri[t, 0], tri[t, 1], tri[t, 2] = p, b, q
        nbr[t, 0], nbr[t, 1], nbr[t, 2] = n_bq, u, n_pb
        con[t, 0], con[t, 1], con[t, 2] = c_bq, False, c_pb
        # u' = (p, q, c): opposite p (q, c) n_qc, opposite q (c, p) n_cp, opposite c (p, q) t'
        tri[u, 0], tri[u, 1], tri[u, 2] = p, q, c
        nbr[u, 0], nbr[u, 1], nbr[u, 2] = n_qc, n_cp, t
        con[u, 0], con[u, 1], con[u, 2] = c_qc, c_cp, False
        # the outer neighbours' pointers: n_bq was u's, now t's; n_cp was t's, now u's
        if n_bq >= 0:
            for k in range(3):
                if nbr[n_bq, k] == u:
                    nbr[n_bq, k] = t
        if n_cp >= 0:
            for k in range(3):
                if nbr[n_cp, k] == t:
                    nbr[n_cp, k] = u
        nflip += 1
        stack.append((t, 0))
        stack.append((t, 2))
        stack.append((u, 0))
        stack.append((u, 1))
    return nflip


def delaunay_flips(xy: NDArray[np.float64], tri: NDArray[np.int64]) -> NDArray[np.int64]:
    """The triangulation made constrained Delaunay by Lawson's flips (`_flips`), on a copy."""
    tri = tri.copy()
    _flips(xy, tri, _neighbours(tri, len(xy)), DELAUNAY_TOL)
    return tri


def build(geo: Geometry, spec: Spec, standing: NDArray[np.bool_], *, carve,
          grow: float) -> Complex:
    """The gap complex of a block (`geometry`) with the buildings `standing` (a cleared
    building's interior is open ground), less `carve` (a corridor in the block's local
    coordinates, freed: a vertex on its cut lies inside its footprint, an obstacle; None:
    nothing), grown by `grow` (_open_ground; a vertex within grow + TOL of a building is on its
    outline)."""
    inner, band_open = _open_ground(geo, standing, spec.band_m, carve=carve, grow=grow)
    tris = _triangulate(inner, spec.s)
    c = shapely.get_coordinates(tris).reshape(-1, 4, 2)[:, :3]
    xy, inv = np.unique(np.round(c.reshape(-1, 2) / SNAP_M) * SNAP_M, axis=0,
                        return_inverse=True)
    tri = inv.reshape(-1, 3).astype(np.int64)
    tri = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 2] != tri[:, 0])]
    # counter-clockwise
    p0, p1, p2 = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (
        p2[:, 0] - p0[:, 0])
    cw = cross < 0
    tri[cw] = tri[cw][:, [0, 2, 1]]
    tri = delaunay_flips(xy, tri)
    tris = shapely.polygons(xy[tri])
    p0, p1, p2 = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (
        p2[:, 0] - p0[:, 0])
    nt, nv = len(tri), len(xy)
    # the vertices on each standing building's (grown) outline
    idx, opolys = obstacles(geo, standing, carve=carve, grow=grow)
    tree = shapely.STRtree(opolys)
    pts = shapely.points(xy[:, 0], xy[:, 1])
    near = grow * (1.0 + GROW_TOL_REL) + TOL
    qv, qb = tree.query(pts, predicate="dwithin", distance=near)
    vb = sp.csr_matrix((np.ones(len(qv)), (qv, idx[qb])), shape=(nv, len(geo.polys)))
    on_b = np.zeros(nv, dtype=bool)
    on_b[qv] = True
    # edges: (triangle, local edge) -> sorted vertex pair
    e = np.stack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=1).reshape(-1, 2)
    es = np.sort(e, axis=1)
    key = es[:, 0] * nv + es[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    same = ks[1:] == ks[:-1]
    if (ks[2:] == ks[:-2]).any():
        raise ValueError("an edge in three triangles")
    i0, i1 = order[:-1][same], order[1:][same]
    pa, pb = i0 // 3, i1 // 3
    pu, pv = es[i0, 0], es[i0, 1]
    street_v = ~on_b
    L = np.hypot(*(xy[pu] - xy[pv]).T)
    ap = np.where(street_v[pu] | street_v[pv], np.inf, L)
    # boundary edges (in one triangle): on a footprint's outline or the street line
    single = np.ones(len(key), dtype=bool)
    single[order[:-1][same]] = False
    single[order[1:][same]] = False
    bi = np.flatnonzero(single)
    mid = 0.5 * (xy[es[bi, 0]] + xy[es[bi, 1]])
    mpts = shapely.points(mid[:, 0], mid[:, 1])
    qm, _ = tree.query(mpts, predicate="dwithin", distance=near)
    on_fp = np.zeros(len(bi), dtype=bool)
    on_fp[qm] = True
    # a street face lies on the band's line (a grown outline's edge, up to GEOS's buffer
    # simplification off its offset, is never one)
    band_line = shapely.boundary(geo.streets.buffer(spec.band_m))
    shapely.prepare(band_line)
    on_fp |= ~shapely.dwithin(band_line, mpts, TOL)
    exit_ = np.zeros(nt, dtype=bool)
    exit_[bi[~on_fp] // 3] = True
    return Complex(spec=spec, xy=xy, tri=tri, tri_geom=tris, area=np.abs(cross) / 2,
                   fill=_fill_of(p0, p1, p2), pa=pa, pb=pb, pu=pu, pv=pv, ap=ap, exit=exit_,
                   street_faces=es[bi[~on_fp]],
                   street_v=street_v, vb=vb, standing=standing.copy(), band_open=band_open)


# ------------------------------------------------------------------ the filtration

@njit(cache=True)
def _propagate(order, pred, wpred, exitw, bar):
    """Along a BFS order of the maximum spanning forest from the street node (order[0]): each
    node's bottleneck to the street, the narrowest tree edge on its path (wpred: the aperture of
    the edge to its predecessor), and its bar, the node whose edge is that bottleneck (the
    nearest to the street on ties: the pocket cut off with it)."""
    for i in range(1, len(order)):
        v = order[i]
        u = pred[v]
        if wpred[v] < exitw[u]:
            exitw[v] = wpred[v]
            bar[v] = v
        else:
            exitw[v] = exitw[u]
            bar[v] = bar[u]


class Exits(NamedTuple):
    """Per triangle: its exit width (the widest path's bottleneck to the street: inf a street
    triangle, -inf none at any width) and its bar (the triangle whose tree edge is that
    bottleneck: the triangles cut off together share it; -1 street or none)."""
    width: NDArray[np.float64]
    bar: NDArray[np.int64]


def exit_widths(cx: Complex) -> Exits:
    nt = cx.nt
    street = nt
    # the maximum spanning forest: scipy's minimum one on the apertures' ranks (only the order
    # matters), the street node joined to every exit triangle below every rank
    order = np.argsort(-cx.ap, kind="stable")
    rank = np.empty(len(order))
    rank[order] = np.arange(1, len(order) + 1, dtype=float)
    ex = np.flatnonzero(cx.exit)
    r = np.concatenate([cx.pa, ex])
    c = np.concatenate([cx.pb, np.full(len(ex), street)])
    w = np.concatenate([rank, np.full(len(ex), 0.5)])
    G = sp.csr_matrix((w, (r, c)), shape=(nt + 1, nt + 1))
    from scipy.sparse.csgraph import minimum_spanning_tree
    T = minimum_spanning_tree(G)
    T = (T + T.T).tocsr()
    width = np.full(nt + 1, -np.inf)
    bar = np.full(nt + 1, -1, dtype=np.int64)
    if len(ex) == 0:
        return Exits(width=width[:nt], bar=bar[:nt])
    bfs, pred = breadth_first_order(T, street, directed=False, return_predecessors=True)
    # the aperture of each node's tree edge to its predecessor
    ap_of_rank = np.concatenate([[np.inf], cx.ap[order]])      # rank 0.5 -> inf (street edge)
    wpred = np.full(nt + 1, np.inf)
    v = bfs[1:]
    rk = np.asarray(T[pred[v], v]).ravel()
    wpred[v] = np.where(rk == 0.5, np.inf, ap_of_rank[np.round(rk).astype(np.int64)])
    width[street] = np.inf
    _propagate(bfs, pred, wpred, width, bar)
    bar[bar == street] = -1
    bar[~np.isfinite(width)] = -1
    return Exits(width=width[:nt], bar=bar[:nt])


class Bars(NamedTuple):
    """The walled-in pockets as bars: per bar (a triangle id, `exit_widths`' bar), its exit width
    and its fill (the largest fill over the triangles cut off with it), and its area."""
    id: NDArray[np.int64]
    exit: NDArray[np.float64]
    fill: NDArray[np.float64]
    area: NDArray[np.float64]


def bars(cx: Complex, ex: Exits) -> Bars:
    ok = ex.bar >= 0
    ids, inv = np.unique(ex.bar[ok], return_inverse=True)
    fill = np.zeros(len(ids))
    np.maximum.at(fill, inv, cx.fill[ok])
    area = np.bincount(inv, weights=cx.area[ok], minlength=len(ids))
    return Bars(id=ids, exit=ex.width[ids], fill=fill, area=area)


# ------------------------------------------------------------------ homes

@dataclass(frozen=True, eq=False)
class Homes:
    """Each home's open ground within ring_m of its outline on a complex: `ring` (homes x
    triangles) 0/1, `band` the open street band within ring_m."""
    ring: sp.csr_matrix
    band: NDArray[np.bool_]


def homes_of(geo: Geometry, cx: Complex) -> Homes:
    polys = geo.polys
    tree = shapely.STRtree(cx.tri_geom)
    qh, qt = tree.query(polys, predicate="dwithin", distance=cx.spec.ring_m)
    ring = sp.csr_matrix((np.ones(len(qh)), (qh, qt)), shape=(len(polys), cx.nt))
    band = np.zeros(len(polys), dtype=bool)
    parts = shapely.get_parts(cx.band_open)
    parts = parts[shapely.area(parts) > 0]
    if len(parts):
        bh, _ = shapely.STRtree(parts).query(polys, predicate="dwithin",
                                              distance=cx.spec.ring_m)
        band[bh] = True
    return Homes(ring=ring, band=band)


def home_exits(hm: Homes, ex: Exits) -> NDArray[np.float64]:
    """Per home: the widest of its ring's exits (inf with the street band in its ring, -inf
    with no ring triangle reaching the street at any width)."""
    R = hm.ring.tocsr()
    out = np.full(R.shape[0], -np.inf)
    rows = np.repeat(np.arange(R.shape[0]), np.diff(R.indptr))
    np.maximum.at(out, rows, ex.width[R.indices])
    out[hm.band] = np.inf
    return out


def walled(hexit: NDArray[np.float64], w: float) -> NDArray[np.bool_]:
    """Walled in at width w: no exit of the ring at least w wide (`home_exits`)."""
    return hexit < w


# ------------------------------------------------------------------ free first

def components(cx: Complex, w: float) -> tuple[NDArray[np.int64], int]:
    """(each triangle's component at width w, the street's): passages with aperture >= w open,
    the exits joined to the street node."""
    nt = cx.nt
    ok = cx.ap >= w
    ex = np.flatnonzero(cx.exit)
    r = np.concatenate([cx.pa[ok], ex])
    c = np.concatenate([cx.pb[ok], np.full(len(ex), nt)])
    _n, lab = connected_components(sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(nt + 1, nt + 1)),
                                   directed=False)
    return lab[:nt], int(lab[nt])


class Graph(NamedTuple):
    """free_first's region graph at one width (`region_graph`): nodes the components (0 ..
    ncomp - 1, the street's `street`) then the buildings (ncomp + j); `adj` symmetric 0/1."""
    ncomp: int
    street: int
    comp: NDArray[np.int64]          # per triangle
    adj: sp.csr_matrix


def region_graph(geo: Geometry, cx: Complex, w: float) -> Graph:
    polys = geo.polys
    n = len(polys)
    comp, street = components(cx, w)
    ncomp = int(comp.max()) + 1
    # a building and every component with a triangle touching its outline at a vertex on its
    # outline alone (a vertex where two footprints meet stays an obstacle while either stands)
    tv = sp.csr_matrix((np.ones(3 * cx.nt), (np.repeat(np.arange(cx.nt), 3), cx.tri.ravel())),
                       shape=(cx.nt, len(cx.xy)))
    vb = cx.vb.tocsr()
    alone = sp.diags((np.diff(vb.indptr) == 1).astype(float)) @ vb
    tb = (tv @ alone).tocoo()
    a = [comp[tb.row]]
    b = [ncomp + tb.col]
    # buildings that intersect (overlap or touch), both standing
    idx = np.flatnonzero(cx.standing)
    tree = shapely.STRtree(polys[idx])
    i, j = tree.query(polys[idx], predicate="intersects")
    keep = i < j
    a.append(ncomp + idx[i[keep]])
    b.append(ncomp + idx[j[keep]])
    # a building reaching into the band: its interior, cleared, is street
    band = geo.streets.buffer(cx.spec.band_m)
    shapely.prepare(band)
    touch = idx[shapely.intersects(band, polys[idx])]
    a.append(np.full(len(touch), street))
    b.append(ncomp + touch)
    a_, b_ = np.concatenate(a), np.concatenate(b)
    N = ncomp + n
    adj = sp.csr_matrix((np.ones(2 * len(a_)), (np.concatenate([a_, b_]),
                                                 np.concatenate([b_, a_]))), shape=(N, N))
    adj.data[:] = 1.0
    return Graph(ncomp=ncomp, street=street, comp=comp, adj=adj)


def _sources(geo: Geometry, cx: Complex, hm: Homes, g: Graph, homes: NDArray[np.int64]
             ) -> dict[int, NDArray[np.int64]]:
    """Per home: the regions it is freed through: its ring triangles' components and the
    standing buildings within ring_m of its outline (itself among them if standing: clearing a
    building opens its interior, within ring_m of the home)."""
    polys = geo.polys
    idx = np.flatnonzero(cx.standing)
    tree = shapely.STRtree(polys[idx])
    qh, qb = tree.query(polys[homes], predicate="dwithin", distance=cx.spec.ring_m)
    R = hm.ring.tocsr()
    out = {}
    for k, h in enumerate(homes):
        tris = R.indices[R.indptr[h]:R.indptr[h + 1]]
        out[int(h)] = np.unique(np.concatenate([g.comp[tris], g.ncomp + idx[qb[qh == k]]]))
    return out


class Freed(NamedTuple):
    """free_first's answer at one width: `held` the buildings cleared; `left` the homes still
    walled in (by the complex rebuilt without `held`); `crossings` in order (home, buildings);
    `passes` the rebuilds; `misses` the homes a pass's graph freed that its rebuild did not."""
    held: NDArray[np.bool_]
    left: NDArray[np.bool_]
    crossings: list
    passes: int
    misses: int


def free_first(geo: Geometry, spec: Spec, *, w: float, budget: float, cost: NDArray[np.float64],
               pop: NDArray[np.float64], held0: NDArray[np.bool_],
               cx0: Complex | None = None) -> Freed:
    """clear.free_first on the complex at width w: `held0` (a clearing, held) plus crossings
    freeing the homes walled in at baseline at w that it leaves walled in, the least D (`cost`,
    each building's population share) per unit of the home's population (`pop`) first, while the
    held set's D stays within `budget` (module docstring). `cx0` the baseline complex if built."""
    held = held0.copy()
    base = (build(geo, spec, np.ones(len(cost), dtype=bool), carve=None, grow=0.0)
            if cx0 is None else cx0)
    target = walled(home_exits(homes_of(geo, base), exit_widths(base)), w)
    cx = base if not held.any() else build(geo, spec, ~held, carve=None, grow=0.0)
    hm = homes_of(geo, cx)
    left = target & walled(home_exits(hm, exit_widths(cx)), w)
    n = len(cost)
    crossings: list = []
    passes = misses = 0
    while left.any():
        passes += 1
        g = region_graph(geo, cx, w)
        src = _sources(geo, cx, hm, g, np.flatnonzero(left))
        N = g.adj.shape[0]
        claimed = np.zeros(n, dtype=bool)
        progress = False
        hold = np.zeros(n, dtype=bool)        # held in this pass (still standing in cx)
        while True:
            enter = np.concatenate([np.zeros(g.ncomp), np.where(hold, 0.0, cost)])
            co = g.adj.tocoo()
            G = sp.csr_matrix((enter[co.col] + FREE_STEP, (co.row, co.col)), shape=(N, N))
            dist, pred = dijkstra(G, directed=True, indices=g.street, return_predecessors=True)
            zero = _zero_reach(g, hold)
            room = budget - float(cost[held | hold].sum())
            best = None
            key = np.inf
            for h in np.flatnonzero(left & ~claimed):
                s = src[int(h)]
                if zero[s].any():
                    claimed[h] = True
                    continue
                if not np.isfinite(dist[s]).any():
                    continue
                t, path = int(s[np.argmin(dist[s])]), []
                while t != g.street:
                    path.append(t)
                    t = int(pred[t])
                new = sorted({t - g.ncomp for t in path if t >= g.ncomp} - set(
                    np.flatnonzero(hold | held).tolist()))
                D = float(cost[new].sum())
                if D <= room and D / pop[h] < key:
                    best, key = (int(h), new, D), D / pop[h]
            if best is None:
                break
            h, new, D = best
            hold[new] = True
            crossings.append((h, new, D))
            claimed[h] = True
            progress = True
        if not progress or not (hold & ~held).any():
            break
        held |= hold
        cx = build(geo, spec, ~held, carve=None, grow=0.0)
        hm = homes_of(geo, cx)
        now = walled(home_exits(hm, exit_widths(cx)), w)
        misses += int((claimed & left & now).sum())
        left = target & now
    return Freed(held=held, left=left, crossings=crossings, passes=passes, misses=misses)


def _zero_reach(g: Graph, hold: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """The region-graph nodes joined to the street through components and buildings held (cost
    0)."""
    N = g.adj.shape[0]
    on = np.concatenate([np.ones(g.ncomp, dtype=bool), hold])
    co = g.adj.tocoo()
    k = on[co.row] & on[co.col]
    _n, lab = connected_components(sp.csr_matrix((np.ones(int(k.sum())), (co.row[k], co.col[k])),
                                                 shape=(N, N)), directed=False)
    return (lab == lab[g.street]) & on
