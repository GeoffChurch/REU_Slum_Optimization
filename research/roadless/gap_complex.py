"""The gap complex: a triangulation of the open ground between footprints, its passages measured
once from the outlines (NOTES, "The gap complex: passability, enclosure and flow without the
raster"; BACKLOG 4, decision (3), the passable width).

THE COMPLEX. The open ground is the block less its footprints and less the street band (the
raster's ground: within `band_m` of Block.streets, the raster's `ground`). Its outline, every
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

HOMES (the raster's Placement test on the complex, its own ground: owner 2026-10-21, "No, its own
pocket must open"; NOTES, "A home is freed only when its own ground opens"). A home's OWN GROUND
is the open ground within `ring_m` of its outline at baseline, every footprint standing (live or
in a sealed pocket), and the open street band within ring_m then (the raster's ground cells, open
ground there, are its street). A home is live at w when some of its own ground reaches the street
at w: ground a clearing opens within ring_m of it (a neighbour's interior) frees it only by
opening its own pocket. A GROUNDLESS home, with no open ground within ring_m at baseline (every
side against another footprint or its zone: decided by Placement's points, gap_flow.groundless,
as gap_clear.PocketDemand leaves it without demand), has no pocket to open: any ground within
ring_m of it reaching the street frees it, cleared ground included (`homes_of`, its RING). On the
baseline complex a home's own ground is its ring; on a complex rebuilt without some footprints,
the triangles holding the centroids of its baseline ring's triangles (`Own`, `homes_own`): a
baseline triangle is open ground in every rebuild and convex, so at width 0 one centroid reads its
whole region. Distances are the polygons' own (shapely), so as in the raster a ring reaches
through a thin wall.

FREE FIRST (`free_first`, the raster's free first on the complex). At width w on the complex built
on the footprints grown by `grow` (the study's: 0, the filtration read at w; the shy distance's
runtime: w / 2 and the filtration at 0, its own geometry, gap_clear): the region graph's nodes
the components of the triangles joined by open passages (the street's the one holding the exits)
and the standing buildings; a building is adjacent to every component with a triangle touching its
(grown) outline at a vertex on no other footprint's (clearing it opens its interior and frees the
passages that end on it; a vertex where two footprints meet stays an obstacle while either
stands), to the buildings whose grown footprints meet its own (within 2 grow), and to the street
where its grown footprint reaches into the band. A home is freed through the components of its
own ground (a groundless home's: of its ring, and the standing buildings within ring_m of it).
Each round one Dijkstra from the street, entering a building costing its population share, picks
the crossing with the least D per unit of the home's population, as the raster's did. A pass ends
when the graph frees every walled-in home or none it can afford; the complex is then rebuilt
without the held buildings and the metric's own test (`walled` of `homes_own`) re-run, another
pass taking any home the graph freed wrongly. Clearing a building
can open a passage no triangle of the old complex shows (two buildings close across its former
footprint), so the graph is generous and the rebuild decides.

The run's identity is `Spec` (s, ring_m, band_m: required, keyword-only, its name in every row);
the width w, the growth, the budget and the groundless homes are each call's, required; the test
of a walled-in home is ENCLOSURE_REV's, in the code's digest.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import scipy.sparse as sp
import shapely
from numba import njit
from numpy.typing import NDArray
from scipy.sparse.csgraph import breadth_first_order, connected_components, dijkstra
from scipy.spatial import cKDTree

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
ENCLOSURE_REV = 2    # the test of a walled-in home (HOMES: B (i)'s, gap_clear.Model.live_homes_of,
                     # and free first's), in the code's digest (gap_clear.CODE) and so in every
                     # complex metric's name: raised by any change to the test (2: freed only when
                     # its own ground reaches the street, owner 2026-10-21)
MERGE_M = 1e-7       # vertices closer than this (m) are one (`vertices`): where two grown
                     # outlines nearly touch their offset arcs cross twice a hair apart (1.4e-9
                     # m on 18912's shy 0.5), a passage that short has needles either side whose
                     # centres no flip can order, and the snap grid splits a pair across a line


@dataclass(frozen=True, kw_only=True)
class Spec:
    """The complex's identity: the outlines' spacing `s` (m), the demand ring `ring_m`
    (the metric's name's r) and the street band `band_m` (its b)."""
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

    def local(self, g: Any) -> Any:
        """`g` (the block's CRS: a road's corridor, say) in these local coordinates."""
        return _shifted(g, self.origin)

    def utm(self, g: Any) -> Any:
        """`g` (local coordinates) back in the block's CRS (`local`'s inverse)."""
        return _shifted(g, (-self.origin[0], -self.origin[1]))


def _shifted(g: Any, origin: tuple[float, float]) -> Any:
    return shapely.transform(g, lambda xy: xy - np.array(origin))


def origin_of(block) -> tuple[float, float]:
    """The block's local frame's origin: its polygon's lower bounds."""
    return (float(block.boundary.bounds[0]), float(block.boundary.bounds[1]))


def geometry(block) -> Geometry:
    origin = origin_of(block)
    return Geometry(boundary=_shifted(block.boundary, origin),
                    polys=_shifted(np.asarray(block.buildings.outlines), origin),
                    streets=_shifted(shapely.union_all(list(block.streets.geometry)), origin),
                    origin=origin)


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
    grow: float                      # the footprints' growth it was built on (0: as they are)

    @property
    def nt(self) -> int:
        return len(self.tri)


_ARRAYS = ("xy", "tri", "area", "fill", "pa", "pb", "pu", "pv", "ap", "exit", "street_faces",
           "street_v", "standing")


def complex_arrays(cx: Complex) -> dict[str, NDArray[Any]]:
    """A complex's arrays, to store (gap_clear.Fixed keeps the baseline's for free first): every
    field but its spec and its triangles' geometries (`complex_of_arrays` makes them again)."""
    vb = cx.vb.tocsr()
    out: dict[str, NDArray[Any]] = {name: getattr(cx, name) for name in _ARRAYS}
    out.update(vb_data=vb.data, vb_indices=vb.indices, vb_indptr=vb.indptr,
               vb_shape=np.array(vb.shape),
               band_open=np.frombuffer(shapely.to_wkb(cx.band_open), dtype=np.uint8),
               grow=np.array(cx.grow))
    return out


def complex_of_arrays(z: Mapping[str, NDArray[Any]], spec: Spec) -> Complex:
    """The complex `complex_arrays` stored (on `spec`, the one it was built on)."""
    xy, tri = z["xy"], z["tri"]
    return Complex(spec=spec, **{name: z[name] for name in _ARRAYS},
                   tri_geom=shapely.polygons(xy[tri]),
                   vb=sp.csr_matrix((z["vb_data"], z["vb_indices"], z["vb_indptr"]),
                                    shape=tuple(z["vb_shape"])),
                   band_open=shapely.from_wkb(z["band_open"].tobytes()),
                   grow=float(z["grow"]))


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
    `standing` less `carve` (a corridor, the road rule: exactly it freed; None: nothing),
    grown by `grow` (the shy distance's w / 2; 0: the footprints as they are): the block less
    them, split at band_m from the streets (the raster's ground: cell centres within band_m of
    Block.streets, as the retired raster's ground; the band is not grown into: it is the street)."""
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
    than tol of its length (DELAUNAY_TOL) where its quadrilateral is strictly convex
    (counter-clockwise triangles; nbr[t, i] the triangle across the edge opposite tri[t, i], -1
    on the outline: the constraints, never flipped). Returns the number of flips."""
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


def vertices(c: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """(the distinct vertices, each input point's) of points `c` (k x 2): snapped to SNAP_M, then
    every cluster closer than MERGE_M (single linkage) one vertex, its lowest's coordinates."""
    xy, inv = np.unique(np.round(c / SNAP_M) * SNAP_M, axis=0, return_inverse=True)
    pairs = cKDTree(xy).query_pairs(MERGE_M, output_type="ndarray")
    if len(pairs) == 0:
        return xy, inv.ravel()
    nv = len(xy)
    _n, lab = connected_components(sp.csr_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(nv, nv)), directed=False)
    first = np.full(lab.max() + 1, nv)
    np.minimum.at(first, lab, np.arange(nv))
    keep, new = np.unique(first, return_inverse=True)
    return xy[keep], new[lab][inv.ravel()]


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
    xy, inv = vertices(c.reshape(-1, 2))
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
                   street_v=street_v, vb=vb, standing=standing.copy(), band_open=band_open,
                   grow=grow)


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
    """Each home's ground on a complex: `ring` (homes x triangles) 0/1, its open ground within
    ring_m of its outline (homes_of) or the ground that frees it (homes_own); `band` the open
    street band within ring_m."""
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


class Own(NamedTuple):
    """Each home's own ground on a baseline triangulation (module docstring, HOMES): its triangles
    there within ring_m of its outline (`tri`, homes x samples, 0/1: every triangle of the
    baseline open at baseline), sampled by their centroids (`xy`), the open band within ring_m
    then (`band`), and the groundless homes (`groundless`), whose ring on a state's triangulation
    frees them instead. The complex's (`own_of`), and the rebuilt flow mesh's
    (gap_compare.ComplexModel)."""
    xy: NDArray[np.float64]
    tri: sp.csr_matrix
    band: NDArray[np.bool_]
    groundless: NDArray[np.bool_]


def own_of(geo: Geometry, base: Complex, groundless: NDArray[np.bool_]) -> Own:
    """The homes' own ground on the baseline complex `base` (every footprint standing):
    `groundless` the caller's (gap_flow.groundless)."""
    if not base.standing.all():
        raise ValueError("own ground is the baseline's: every footprint standing")
    hm = homes_of(geo, base)
    return Own(xy=base.xy[base.tri].mean(axis=1), tri=hm.ring.tocsr(), band=hm.band,
               groundless=groundless)


def located(tri_geom: NDArray[np.object_], xy: NDArray[np.float64]) -> sp.csr_matrix:
    """(points x triangles) 0/1: the triangles `tri_geom` holding each point of `xy` (a point on a
    shared edge in both; none outside them)."""
    qi, qt = shapely.STRtree(tri_geom).query(shapely.points(xy[:, 0], xy[:, 1]),
                                             predicate="intersects")
    return sp.csr_matrix((np.ones(len(qi)), (qi, qt)), shape=(len(xy), len(tri_geom)))


def homes_own(geo: Geometry, own: Own, cx: Complex) -> Homes:
    """Each home's ground that frees it on `cx` (module docstring, HOMES), as Homes: a home with
    own ground the triangles of cx holding its own ground's samples and its baseline band; a
    groundless home its ring on cx (homes_of)."""
    hm = homes_of(geo, cx)
    mine = (sp.diags((~own.groundless).astype(np.float64)) @ own.tri).tocsr()
    used = np.unique(mine.indices)
    mine = ((mine[:, used] @ located(cx.tri_geom, own.xy[used])) > 0).astype(np.float64)
    ring = (mine + sp.diags(own.groundless.astype(np.float64)) @ hm.ring).tocsr()
    ring.eliminate_zeros()
    return Homes(ring=ring, band=np.where(own.groundless, hm.band, own.band))


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
    # buildings whose (grown) footprints meet, both standing (dwithin 0: they intersect)
    idx = np.flatnonzero(cx.standing)
    tree = shapely.STRtree(polys[idx])
    i, j = tree.query(polys[idx], predicate="dwithin", distance=2.0 * cx.grow)
    keep = i < j
    a.append(ncomp + idx[i[keep]])
    b.append(ncomp + idx[j[keep]])
    # a building whose (grown) footprint reaches into the band: its interior, cleared, is street
    band = geo.streets.buffer(cx.spec.band_m)
    shapely.prepare(band)
    touch = idx[shapely.dwithin(band, polys[idx], cx.grow)]
    a.append(np.full(len(touch), street))
    b.append(ncomp + touch)
    a_, b_ = np.concatenate(a), np.concatenate(b)
    N = ncomp + n
    adj = sp.csr_matrix((np.ones(2 * len(a_)), (np.concatenate([a_, b_]),
                                                 np.concatenate([b_, a_]))), shape=(N, N))
    adj.data[:] = 1.0
    return Graph(ncomp=ncomp, street=street, comp=comp, adj=adj)


def _sources(geo: Geometry, cx: Complex, hm: Homes, g: Graph, homes: NDArray[np.int64],
             groundless: NDArray[np.bool_]) -> dict[int, NDArray[np.int64]]:
    """Per home: the regions it is freed through (`hm`: homes_own): its own ground's components;
    a groundless home's ring's, and the standing buildings within ring_m of its outline (itself
    among them if standing: clearing a building opens its interior, within ring_m of the home)."""
    polys = geo.polys
    idx = np.flatnonzero(cx.standing)
    tree = shapely.STRtree(polys[idx])
    bare = homes[groundless[homes]]
    qh, qb = tree.query(polys[bare], predicate="dwithin", distance=cx.spec.ring_m)
    near = {int(h): idx[qb[qh == k]] for k, h in enumerate(bare)}
    R = hm.ring.tocsr()
    out = {}
    for h in homes:
        tris = R.indices[R.indptr[h]:R.indptr[h + 1]]
        parts = [g.comp[tris], *([g.ncomp + near[int(h)]] if groundless[h] else [])]
        out[int(h)] = np.unique(np.concatenate(parts))
    return out


class Crossing(NamedTuple):
    """One of free first's crossings: the `home` it frees, the buildings it holds `new` (those its
    path crosses not held before it) and their D, and the buildings that free it from the
    baseline alone (`path`: every building its path crosses, the held ones among them, and every
    building held before its pass, which a later pass's complex has made open ground)."""
    home: int
    new: list[int]
    D: float
    path: list[int]


class Freed(NamedTuple):
    """free_first's answer at one width: `target` the homes walled in at baseline; `held` the
    buildings cleared; `left` the homes still walled in (by the complex rebuilt without `held`);
    `crossings` in order (Crossing); `passes` the rebuilds; `misses` the homes a pass's graph
    freed that its rebuild did not."""
    target: NDArray[np.bool_]
    held: NDArray[np.bool_]
    left: NDArray[np.bool_]
    crossings: list[Crossing]
    passes: int
    misses: int


def free_first(geo: Geometry, spec: Spec, *, w: float, grow: float, budget: float,
               cost: NDArray[np.float64], pop: NDArray[np.float64], held0: NDArray[np.bool_],
               cx0: Complex | None, groundless: NDArray[np.bool_]) -> Freed:
    """Free first on the complex grown by `grow` at width w: `held0` (a clearing, held)
    plus crossings freeing the homes walled in at baseline at w that it leaves walled in, the
    least D (`cost`, each building's population share) per unit of the home's population (`pop`)
    first, while the held set's D stays within `budget` (module docstring). `cx0` the baseline
    complex (grown by `grow`) if built, else None; `groundless` the homes with no open ground
    within ring_m at baseline (HOMES: gap_flow.groundless)."""
    held = held0.copy()
    base = (build(geo, spec, np.ones(len(cost), dtype=bool), carve=None, grow=grow)
            if cx0 is None else cx0)
    if base.grow != grow:
        raise ValueError(f"the baseline complex is grown by {base.grow:g}, free first by {grow:g}")
    own = own_of(geo, base, groundless)
    target = walled(home_exits(homes_own(geo, own, base), exit_widths(base)), w)
    cx = base if not held.any() else build(geo, spec, ~held, carve=None, grow=grow)
    hm = homes_own(geo, own, cx)
    left = target & walled(home_exits(hm, exit_widths(cx)), w)
    n = len(cost)
    crossings: list[Crossing] = []
    passes = misses = 0
    while left.any():
        passes += 1
        g = region_graph(geo, cx, w)
        src = _sources(geo, cx, hm, g, np.flatnonzero(left), groundless)
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
                every = sorted({t - g.ncomp for t in path if t >= g.ncomp})
                new = sorted(set(every) - set(np.flatnonzero(hold | held).tolist()))
                D = float(cost[new].sum())
                if D <= room and D / pop[h] < key:
                    best, key = Crossing(home=int(h), new=new, D=D, path=sorted(
                        set(every) | set(np.flatnonzero(held).tolist()))), D / pop[h]
            if best is None:
                break
            hold[best.new] = True
            crossings.append(best)
            h = best.home
            claimed[h] = True
            progress = True
        if not progress or not (hold & ~held).any():
            break
        held |= hold
        cx = build(geo, spec, ~held, carve=None, grow=grow)
        hm = homes_own(geo, own, cx)
        now = walled(home_exits(hm, exit_widths(cx)), w)
        misses += int((claimed & left & now).sum())
        left = target & now
    return Freed(target=target, held=held, left=left, crossings=crossings, passes=passes,
                 misses=misses)


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
