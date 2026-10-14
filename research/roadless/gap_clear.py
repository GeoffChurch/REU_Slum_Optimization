"""The gap complex's clearing model: the roadless metric's runtime on one fixed mesh (NOTES, "The
complex as the runtime: one fixed mesh"; its fractional clearing first in "The gap complex:
passability, enclosure and flow without the raster"). The owner's decision (2026-10-14): the shy
distance at w 0.5 m on the complex, isotropic, as the metric and its runtime.

THE MESH (`full_mesh`; `Fixed.of` caches it on disk by its identity). Every footprint's outline,
its w / 2 offset (the width form's growth: the shy distance's zero zone; Closing has none) and the
street band's line, noded together (shapely's union of lines, the region's boundary whole),
clipped to the block beyond the band, densified to s once and polygonized: faces, each covered by
a set of footprints (none: open ground; one: a building's interior; several: an overlap) and lying
within w / 2 of a set (its zone). Each face is triangulated by GEOS (constrained Delaunay) on that
shared linework, so faces meet vertex for vertex; Lawson's flips within each face (the face
boundaries flagged, never flipped: a triangle lies in one face); Steiner points on gap_flow's
graded lattice, laid over the box of the baseline's open ground (the rebuild's own lattice there),
kept R / 2 clear of every line of the noding (and in the open faces by the form's clearance),
inserted with Lawson's flips. Then CONFORMITY (`_conform`): every face-boundary edge two triangles
share whose circumcentres lie in the wrong order or within MERGE_REL of its length of each other
((cot a + cot b) / 2 <= MERGE_REL, a and b the angles facing it: a constrained edge not Delaunay,
or two faces sharing a circumcircle) is split at its midpoint and the faces re-flipped, round after
round until none is. The mesh is then Delaunay outright, so whichever faces a clearing opens, every
passage's two-point weight sigma |e| / d_e is positive and consistent: no clip (the study's
CROSS_MIN clipped the 0.3% of passages that were not). Faces under DUST_M2 hold no triangles (the
overlay's slivers, as the complex's); a passage beside a flat triangle (a hairline crack's, no
centre) conducts nothing (gap_flow's rule). Cells: triangles sharing a circumcircle within a face
are one cell (their factors are equal at every state).

THE STATE (`Model.state`). A clearing k (per building in [0, 1]: 0/1 for an exact score; eps +
(1 - eps) x^q in SIMP's relaxation; eps standing and 1 cleared in the tension) only re-weights
passages: a triangle's factor is the width form's (gap_flow.Width.terms: under the shy distance
the least k_b of the footprints whose zone holds it), a passage's the harmonic mean of its two
triangles' (the half-cells in series: concave and non-decreasing, so P stays convex in the
clearing), a street face its triangle's. A passage conducts its weight times its factor; the
cells reached from the street through conducting passages are the state's (the rest walled in at
w), an obtuse exit triangle's cell is ground once it conducts. Scoring a state is assembly,
factorization and solves: no GEOS call and no meshing after the baseline.

THE SCORER (`Model`). The demand is gap_flow's (lifted.Placement's rule on the complex), fixed at
baseline on the triangles reaching the street then; each home's escape time the injection-weighted
mean potential over its points. P (each home injecting its population), P_{1/2}
(common.PriorityDemand(0.5), its weights from the baseline's escape times on this mesh) and J_2,
over the homes with demand; the enclosed population: the homes walled in at baseline (no ring
triangle reaching the street: Placement's test) that a clearing leaves walled in, by the flow
model's own reach (`live_homes`). The optimizers' objective is J_power under the run's demand
(common.Demand; J_1 is P).

FREE FIRST (`Model.free_first`): gap_complex.free_first on the complex grown as the form grows its
footprints, read at the form's filtration width (the shy distance: grown by w / 2, at 0: its own
geometry exactly, where the unrown filtration at w lets a gap a few millimetres under w through by
its apertures' error), its walled-in homes and the homes it frees checked home for home against
the flow model's reach.

THE SEAM. `Model` is a seam.ClearingModel, `Relaxation` a seam.Relaxed, `Tension` a
seam.Tension, `ComplexMetric` a seam.Metric: what search.py, relax.py and polish_greedy.py run
on.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import scipy.sparse as sp
import shapely
from numba import njit
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components

import clear
import clearings
import common
import gap_complex as gc
import gap_flow as gf
import lifted
import relax
import search
from gap_complex import legalize
from seam import Flow

HERE = Path(__file__).resolve().parent
CACHE = HERE / "gap_cache"              # the fixed meshes, by identity (`Fixed.of`; git-ignored)
CONFORM_ROUNDS = 64     # conformity: rounds at most (each splits every bad edge it finds; a mesh
                        # still holding one after them raises)
MESH_REV = 1            # the fixed mesh's construction, in its cache key with every meshing
                        # constant: raised by any change to the code that moves a mesh
LINE_SNAP_M = 1e-4      # the linework's precision (m): a crack narrower than this conducts
                        # nothing under any passable width while either footprint stands (it is
                        # in their zero zone, or flat: gap_flow.FLAT_REL s is 1.25e-5 m), and with
                        # both cleared it is nothing; the vertices move by at most half of it


# ------------------------------------------------------------------ the mesh

def _faces(geo: gc.Geometry, spec: gc.Spec, grow: float
           ) -> tuple[NDArray[np.object_], NDArray[np.object_], NDArray[np.object_], Any]:
    """(the faces, densified, as polygons; per face its covering footprints; per face the
    footprints within `grow` of it, its zone: the offset curves at `grow` are in the linework, so
    a face lies wholly inside or outside each footprint's zero zone; the noded linework)."""
    band = geo.streets.buffer(spec.band_m)
    region = geo.boundary.difference(band)
    # the outlines clipped to the region, then noded with the region's own boundary whole (a
    # noded set clipped afterwards lost pieces of that boundary on 19570, and with them the open
    # faces it closes)
    inside = [shapely.boundary(geo.polys)]
    if grow > 0:
        inside.append(shapely.boundary(shapely.buffer(geo.polys, grow)))
    clipped = shapely.intersection(np.concatenate(inside), region)
    noded = shapely.union_all(np.concatenate([clipped, [shapely.boundary(region)]]))
    # snap-rounded (GEOS's fixed-precision noding): lines closer than LINE_SNAP_M are one, so a
    # hairline crack between two footprints is a shared boundary, not a sliver face of flat
    # triangles (a wall between the two interiors once both are cleared)
    noded = shapely.segmentize(shapely.set_precision(noded, LINE_SNAP_M), spec.s)
    faces = shapely.get_parts(shapely.polygonize(shapely.get_parts(noded)))
    faces = faces[shapely.area(faces) >= gc.DUST_M2]
    rep = shapely.point_on_surface(faces)
    tree = shapely.STRtree(geo.polys)
    qi, qb = tree.query(rep, predicate="intersects")
    cover = [tuple(sorted(qb[qi == k].tolist())) for k in range(len(faces))]
    if grow > 0:
        zi, zb = tree.query(rep, predicate="dwithin", distance=grow)
        dz = shapely.distance(rep[zi], geo.polys[zb])
        zi, zb = zi[dz < grow], zb[dz < grow]
        zone = [tuple(sorted(zb[zi == k].tolist())) for k in range(len(faces))]
    else:
        zone = cover
    return (faces, np.array(cover + [()], dtype=object)[:-1],
            np.array(zone + [()], dtype=object)[:-1], noded)


@njit(cache=True)
def _cot(o, u, v):
    """The cotangent of the angle at o of the counter-clockwise triangle (o, u, v)."""
    ax, ay, bx, by = u[0] - o[0], u[1] - o[1], v[0] - o[0], v[1] - o[1]
    cr = ax * by - ay * bx
    if cr <= 0.0:
        return -np.inf
    return (ax * bx + ay * by) / cr


@njit(cache=True)
def _area2(a, b, c):
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


@njit(cache=True)
def _bad(xy, tri, nbr, con, t, i, merge_rel, flat_rel):
    """Whether the edge opposite tri[t, i] is a face boundary two triangles share whose
    circumcentres lie in the wrong order or within merge_rel of its length (module docstring),
    neither triangle flat (area under flat_rel L^2: a hairline crack's, left to drop)."""
    u = nbr[t, i]
    if u < 0 or not con[t, i]:
        return False
    j = -1
    for k in range(3):
        if nbr[u, k] == t:
            j = k
    if j < 0:
        return False
    p, b, c = tri[t, i], tri[t, (i + 1) % 3], tri[t, (i + 2) % 3]
    q = tri[u, j]
    L2 = (xy[b, 0] - xy[c, 0]) ** 2 + (xy[b, 1] - xy[c, 1]) ** 2
    if (min(_area2(xy[p], xy[b], xy[c]), _area2(xy[q], xy[c], xy[b]))
            <= 2.0 * flat_rel * L2):
        return False
    return 0.5 * (_cot(xy[p], xy[b], xy[c]) + _cot(xy[q], xy[c], xy[b])) <= merge_rel


@njit(cache=True)
def _split(xy, tri, nbr, con, face, nt, nv, t, i, tol):
    """The constrained edge opposite tri[t, i] split at its midpoint m: t = (p, b, c) and its
    neighbour u = (q, c, b) become (p, b, m), (p, m, c), (q, c, m), (q, m, b), the halves
    constrained, each child in its parent's face; then Lawson's flips from the four outer edges
    (gap_complex.legalize: never across a constrained edge). Returns (nt, nv)."""
    u = nbr[t, i]
    p, b, c = tri[t, i], tri[t, (i + 1) % 3], tri[t, (i + 2) % 3]
    j = -1
    jb = -1
    jc = -1
    for k in range(3):
        if nbr[u, k] == t:
            j = k
        if tri[u, k] == b:
            jb = k
        if tri[u, k] == c:
            jc = k
    q = tri[u, j]
    n_cp, n_pb = nbr[t, (i + 1) % 3], nbr[t, (i + 2) % 3]
    c_cp, c_pb = con[t, (i + 1) % 3], con[t, (i + 2) % 3]
    n_bq, n_qc = nbr[u, jc], nbr[u, jb]
    c_bq, c_qc = con[u, jc], con[u, jb]
    m = nv
    xy[m, 0] = 0.5 * (xy[b, 0] + xy[c, 0])
    xy[m, 1] = 0.5 * (xy[b, 1] + xy[c, 1])
    nv += 1
    t2 = nt
    u2 = nt + 1
    nt += 2
    tri[t, 0], tri[t, 1], tri[t, 2] = p, b, m
    nbr[t, 0], nbr[t, 1], nbr[t, 2] = u2, t2, n_pb
    con[t, 0], con[t, 1], con[t, 2] = True, False, c_pb
    tri[t2, 0], tri[t2, 1], tri[t2, 2] = p, m, c
    nbr[t2, 0], nbr[t2, 1], nbr[t2, 2] = u, n_cp, t
    con[t2, 0], con[t2, 1], con[t2, 2] = True, c_cp, False
    tri[u, 0], tri[u, 1], tri[u, 2] = q, c, m
    nbr[u, 0], nbr[u, 1], nbr[u, 2] = t2, u2, n_qc
    con[u, 0], con[u, 1], con[u, 2] = True, False, c_qc
    tri[u2, 0], tri[u2, 1], tri[u2, 2] = q, m, b
    nbr[u2, 0], nbr[u2, 1], nbr[u2, 2] = t, n_bq, u
    con[u2, 0], con[u2, 1], con[u2, 2] = True, c_bq, False
    face[t2] = face[t]
    face[u2] = face[u]
    if n_cp >= 0:
        for k in range(3):
            if nbr[n_cp, k] == t:
                nbr[n_cp, k] = t2
    if n_bq >= 0:
        for k in range(3):
            if nbr[n_bq, k] == u:
                nbr[n_bq, k] = u2
    stack = [(t, 2), (t2, 1), (u, 2), (u2, 1)]
    legalize(xy[:nv], tri[:nt], nbr[:nt], con[:nt], stack, tol)
    return nt, nv


@njit(cache=True)
def _unflat(xy, tri, nbr, con, face, nt, flat_rel, tol):
    """Each flat triangle (area under flat_rel L^2, L its longest edge: three outline vertices in
    a row, or a hairline crack's) whose longest edge is no face boundary dissolved into its
    neighbour across that edge: t = (b, x, y), its middle vertex b between x and y, and u = (d,
    y, x) become (d, y, b) and (d, b, x), the edges b-x and y-b kept, then Lawson's flips from the
    outer edges. In the rebuild such a triangle lines a wall and its passages are dropped
    harmlessly; on the fixed mesh it lies on a face boundary that a clearing opens, where dropped
    passages would be a wall the geometry has not. Returns (dissolved, left: flat triangles kept,
    their longest edge a face boundary or the region's)."""
    done = 0
    for t in range(nt):
        best = -1.0
        i = -1
        for k in range(3):
            a, c = tri[t, (k + 1) % 3], tri[t, (k + 2) % 3]
            L2 = (xy[a, 0] - xy[c, 0]) ** 2 + (xy[a, 1] - xy[c, 1]) ** 2
            if L2 > best:
                best, i = L2, k
        if _area2(xy[tri[t, 0]], xy[tri[t, 1]], xy[tri[t, 2]]) > 2.0 * flat_rel * best:
            continue
        u = nbr[t, i]
        if u < 0 or con[t, i]:
            continue
        b, x, y = tri[t, i], tri[t, (i + 1) % 3], tri[t, (i + 2) % 3]
        jx = -1
        jy = -1
        jd = -1
        for k in range(3):
            if tri[u, k] == x:
                jx = k
            elif tri[u, k] == y:
                jy = k
            else:
                jd = k
        d = tri[u, jd]
        n_bx, c_bx = nbr[t, (i + 2) % 3], con[t, (i + 2) % 3]    # t's edge (b, x), opposite y
        n_yb, c_yb = nbr[t, (i + 1) % 3], con[t, (i + 1) % 3]    # t's edge (y, b), opposite x
        n_dy, c_dy = nbr[u, jx], con[u, jx]                      # u's edge (d, y), opposite x
        n_xd, c_xd = nbr[u, jy], con[u, jy]                      # u's edge (x, d), opposite y
        # u <- (d, y, b), t <- (d, b, x)
        tri[u, 0], tri[u, 1], tri[u, 2] = d, y, b
        nbr[u, 0], nbr[u, 1], nbr[u, 2] = n_yb, t, n_dy
        con[u, 0], con[u, 1], con[u, 2] = c_yb, False, c_dy
        tri[t, 0], tri[t, 1], tri[t, 2] = d, b, x
        nbr[t, 0], nbr[t, 1], nbr[t, 2] = n_bx, n_xd, u
        con[t, 0], con[t, 1], con[t, 2] = c_bx, c_xd, False
        face[t] = face[u]
        if n_yb >= 0:
            for k in range(3):
                if nbr[n_yb, k] == t:
                    nbr[n_yb, k] = u
        if n_xd >= 0:
            for k in range(3):
                if nbr[n_xd, k] == u:
                    nbr[n_xd, k] = t
        stack = [(u, 0), (u, 2), (t, 0), (t, 1)]
        legalize(xy, tri, nbr, con, stack, tol)
        done += 1
    left = 0
    for t in range(nt):
        best = 0.0
        for k in range(3):
            a, c = tri[t, (k + 1) % 3], tri[t, (k + 2) % 3]
            best = max(best, (xy[a, 0] - xy[c, 0]) ** 2 + (xy[a, 1] - xy[c, 1]) ** 2)
        if _area2(xy[tri[t, 0]], xy[tri[t, 1]], xy[tri[t, 2]]) <= 2.0 * flat_rel * best:
            left += 1
    return done, left


@njit(cache=True)
def _conform_round(xy, tri, nbr, con, face, nt, nv, cand, merge_rel, flat_rel, tol):
    """One round: each candidate edge (t, i), still bad when its turn comes (an earlier split or
    flip may have changed it), split. Returns (nt, nv, splits)."""
    splits = 0
    for k in range(cand.shape[0]):
        t, i = cand[k, 0], cand[k, 1]
        if _bad(xy, tri, nbr, con, t, i, merge_rel, flat_rel):
            nt, nv = _split(xy, tri, nbr, con, face, nt, nv, t, i, tol)
            splits += 1
    return nt, nv, splits


@njit(cache=True)
def _find_bad(xy, tri, nbr, con, nt, merge_rel, flat_rel):
    out = []
    for t in range(nt):
        for i in range(3):
            if nbr[t, i] > t and _bad(xy, tri, nbr, con, t, i, merge_rel, flat_rel):
                out.append((t, i))
    res = np.empty((len(out), 2), dtype=np.int64)
    for k in range(len(out)):
        res[k, 0], res[k, 1] = out[k]
    return res


class Arrays(NamedTuple):
    """A triangulation being built, with room: the first nv / nt rows are live."""
    xy: NDArray[np.float64]
    tri: NDArray[np.int64]
    nbr: NDArray[np.int64]
    con: NDArray[np.bool_]
    face: NDArray[np.int64]
    nt: int
    nv: int


def _room(a: Arrays, k: int) -> Arrays:
    """`a` with room for k more vertices and 2 k more triangles."""
    def grow(x: NDArray[Any], n: int, fill: Any) -> NDArray[Any]:
        if len(x) >= n:
            return x
        out = np.full((n, *x.shape[1:]), fill, dtype=x.dtype)
        out[:len(x)] = x
        return out
    return Arrays(xy=grow(a.xy, a.nv + k, 0.0), tri=grow(a.tri, a.nt + 2 * k, 0),
                  nbr=grow(a.nbr, a.nt + 2 * k, -1), con=grow(a.con, a.nt + 2 * k, False),
                  face=grow(a.face, a.nt + 2 * k, 0), nt=a.nt, nv=a.nv)


class Conformed(NamedTuple):
    arrays: Arrays
    unflat: int             # flat triangles dissolved
    flat_left: int          # flat triangles left (their longest edge a face boundary)
    splits: int
    rounds: int


UNFLAT_PASSES = 8           # _unflat's passes at most (a dissolution's flips can expose another)


def _conform(a: Arrays) -> Conformed:
    """Conformity (module docstring): the flat triangles dissolved (`_unflat`), then bad
    face-boundary edges split until none is left."""
    unflat = left = 0
    for _p in range(UNFLAT_PASSES):
        k, left = _unflat(a.xy, a.tri, a.nbr, a.con, a.face, a.nt, gf.FLAT_REL, gc.DELAUNAY_TOL)
        unflat += k
        if k == 0:
            break
    splits = 0
    for r in range(CONFORM_ROUNDS):
        cand = _find_bad(a.xy, a.tri, a.nbr, a.con, a.nt, gf.MERGE_REL, gf.FLAT_REL)
        if len(cand) == 0:
            return Conformed(arrays=a, unflat=unflat, flat_left=left, splits=splits, rounds=r)
        a = _room(a, len(cand))
        nt, nv, k = _conform_round(a.xy, a.tri, a.nbr, a.con, a.face, a.nt, a.nv, cand,
                                   gf.MERGE_REL, gf.FLAT_REL, gc.DELAUNAY_TOL)
        a = a._replace(nt=nt, nv=nv)
        splits += k
    left = _find_bad(a.xy, a.tri, a.nbr, a.con, a.nt, gf.MERGE_REL, gf.FLAT_REL)
    if len(left):
        raise RuntimeError(f"conformity: {len(left)} face-boundary edges still not Delaunay after "
                           f"{CONFORM_ROUNDS} rounds ({splits} splits)")
    return Conformed(arrays=a, unflat=unflat, flat_left=left, splits=splits,
                     rounds=CONFORM_ROUNDS)


@dataclass(frozen=True, eq=False)
class FullMesh:
    """The block's fixed mesh beyond the band, building interiors included (module docstring)."""
    xy: NDArray[np.float64]
    tri: NDArray[np.int64]
    face: NDArray[np.int64]          # per triangle
    fcov: sp.csr_matrix              # (faces, buildings) 0/1: the footprints covering the face
    fzone: sp.csr_matrix             # (faces, buildings) 0/1: the footprints within grow of it
    pa: NDArray[np.int64]            # passages: their two triangles and two vertices
    pb: NDArray[np.int64]
    pu: NDArray[np.int64]
    pv: NDArray[np.int64]
    T: NDArray[np.float64]           # per passage sigma |e| / d_e (inf: one circumcentre, within
                                     # a face; 0 beside a flat triangle)
    fa: NDArray[np.int64]            # street faces: triangle, vertices
    fu: NDArray[np.int64]
    fv: NDArray[np.int64]
    Tf: NDArray[np.float64]          # per street face sigma |f| / d_f (inf: its centre beyond it,
                                     # the cell ground)
    vb: sp.csr_matrix                # (vertices, buildings) 0/1: on the outline
    obstacle: NDArray[np.bool_]      # per vertex: on a footprint's outline
    sigma: float
    unflat: int                      # flat triangles dissolved (conformity's first step)
    splits: int                      # conformity's
    rounds: int
    flat: int                        # passages beside a flat triangle (no weight)
    skipped: int                     # Steiner points left out

    @property
    def nt(self) -> int:
        return len(self.tri)

    @property
    def tcov(self) -> sp.csr_matrix:
        return self.fcov[self.face]

    @property
    def tzone(self) -> sp.csr_matrix:
        return self.fzone[self.face]

    @property
    def open_tri(self) -> NDArray[np.bool_]:
        return np.diff(self.tcov.indptr) == 0


def _incidence(sets: NDArray[np.object_], n: int) -> sp.csr_matrix:
    rows = np.repeat(np.arange(len(sets)), [len(c) for c in sets])
    cols = (np.concatenate([np.asarray(c, dtype=np.int64) for c in sets]) if len(rows)
            else np.zeros(0, dtype=np.int64))
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(sets), n))


def full_mesh(geo: gc.Geometry, spec: gc.Spec, fs: gf.FlowSpec, form: gf.Width) -> FullMesh:
    faces, cover, zone, noded = _faces(geo, spec, form.grow)
    tris, tface = [], []
    for k, f in enumerate(faces):
        err = None
        for q in (f, shapely.set_precision(f, gc.SNAP_M)):
            try:
                t = shapely.get_parts(shapely.constrained_delaunay_triangles(q))
            except shapely.errors.GEOSException as e:
                err = e
                continue
            tris.append(t)
            tface.append(np.full(len(t), k))
            err = None
            break
        if err is not None:
            raise err
    tris_ = np.concatenate(tris)
    face = np.concatenate(tface)
    c = shapely.get_coordinates(tris_).reshape(-1, 4, 2)[:, :3]
    xy, inv = gc.vertices(c.reshape(-1, 2))
    tri = inv.reshape(-1, 3).astype(np.int64)
    good = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 2] != tri[:, 0])
    tri, face = tri[good], face[good]
    cr = gf._cross(xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]])
    tri[cr < 0] = tri[cr < 0][:, [0, 2, 1]]
    nv0, nt = len(xy), len(tri)
    nbr = gc._neighbours(tri, nv0)
    con = (nbr < 0) | (face[np.maximum(nbr, 0)] != face[:, None])
    gc.flips_all(xy, tri, nbr, con, gc.DELAUNAY_TOL)
    # Steiner points on the baseline's lattice (the rebuild's, gap_flow.mesh at every footprint
    # standing), kept clear of every line of the noding: the open faces' by the form's clearance
    # too, the interiors' by R / 2
    base, _band = gc._open_ground(geo, np.ones(len(geo.polys), dtype=bool), spec.band_m,
                                  carve=None, grow=form.grow)
    box = base.bounds
    open_f = np.array([len(cv) == 0 for cv in cover])
    region_open = shapely.union_all(faces[open_f]) if open_f.any() else shapely.Polygon()
    region_bld = shapely.union_all(faces[~open_f]) if (~open_f).any() else shapely.Polygon()
    pts = np.concatenate([
        gf.steiner_points(region_open, fs, form.steiner_clear, box=box, lines=noded),
        gf.steiner_points(region_bld, fs, 0.0, box=box, lines=noded)])
    a = _room(Arrays(xy=xy, tri=tri, nbr=nbr, con=con, face=face, nt=nt, nv=nv0), len(pts))
    # a split triangle's children keep its face (gap_flow._insert), and flips stay within faces
    nt2, nv2, skipped = gf.insert_points(a.xy, a.tri, a.nbr, a.con, a.face, nt, nv0, pts)
    conf = _conform(a._replace(nt=nt2, nv=nv2))
    a = conf.arrays
    xy, tri, face = a.xy[:a.nv], a.tri[:a.nt], a.face[:a.nt]
    n = len(geo.polys)
    # passages, their two-point weights
    nvv = len(xy)
    e = np.stack([tri[:, [1, 2]], tri[:, [2, 0]], tri[:, [0, 1]]], axis=1).reshape(-1, 2)
    es = np.sort(e, axis=1)
    key = es[:, 0] * nvv + es[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    same = ks[1:] == ks[:-1]
    if (ks[2:] == ks[:-2]).any():
        raise ValueError("an edge in three triangles")
    i0, i1 = order[:-1][same], order[1:][same]
    pa, pb = i0 // 3, i1 // 3
    pu, pv = es[i0, 0], es[i0, 1]
    cc = gf.circumcentres(xy, tri)
    L = np.hypot(*(xy[pu] - xy[pv]).T)
    tv = (xy[pv] - xy[pu]) / L[:, None]
    nrm = np.column_stack([-tv[:, 1], tv[:, 0]])
    p3 = gf._third(tri, pa, pu, pv)
    flip = ((xy[p3] - xy[pu]) * nrm).sum(1) > 0
    nrm[flip] *= -1.0
    d = ((cc[pb] - cc[pa]) * nrm).sum(1)
    area = 0.5 * np.abs(gf._cross(xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]))
    flat = np.minimum(area[pa], area[pb]) <= gf.FLAT_REL * L * L
    cross_f = face[pa] != face[pb]
    wrong = ((d < -gf.MERGE_REL * L) | (cross_f & (d <= gf.MERGE_REL * L))) & ~flat
    if wrong.any():
        w_ = np.flatnonzero(wrong)
        raise ValueError(f"{len(w_)} passages ({int(cross_f[w_].sum())} across faces) with their "
                         f"centres in the wrong order or one: the mesh is not Delaunay "
                         f"(d / L {np.round(d[w_] / L[w_], 6)[:5].tolist()}, at "
                         f"{np.round(xy[pu[w_[:3]]], 3).tolist()})")
    drop = flat & (d <= gf.MERGE_REL * L)
    with np.errstate(divide="ignore"):
        T = np.where(drop, 0.0, np.where(d <= gf.MERGE_REL * L, np.inf,
                                         fs.sigma * L / np.maximum(d, 1e-300)))
    # street faces: boundary edges on the band's line
    single = np.ones(len(key), dtype=bool)
    single[i0] = False
    single[i1] = False
    bi = np.flatnonzero(single)
    mid = 0.5 * (xy[es[bi, 0]] + xy[es[bi, 1]])
    shapely.prepare(band_line := shapely.boundary(geo.streets.buffer(spec.band_m)))
    on_band = shapely.dwithin(band_line, shapely.points(mid[:, 0], mid[:, 1]), LINE_SNAP_M)
    fb = bi[on_band]
    fa_, fu_, fv_ = fb // 3, es[fb, 0], es[fb, 1]
    fl = np.hypot(*(xy[fv_] - xy[fu_]).T)
    ftv = (xy[fv_] - xy[fu_]) / fl[:, None]
    fn = np.column_stack([-ftv[:, 1], ftv[:, 0]])
    fp = gf._third(tri, fa_, fu_, fv_)
    inward = ((xy[fp] - xy[fu_]) * fn).sum(1) > 0
    fn[inward] *= -1.0
    df = ((xy[fu_] - cc[fa_]) * fn).sum(1)
    Tf = np.where(df > gf.MERGE_REL * fl, fs.sigma * fl / np.maximum(df, 1e-300), np.inf)
    # vertices on outlines (Closing's filled gaps)
    tree = shapely.STRtree(geo.polys)
    qv, qb = tree.query(shapely.points(xy[:, 0], xy[:, 1]), predicate="dwithin",
                        distance=LINE_SNAP_M)
    vb = sp.csr_matrix((np.ones(len(qv)), (qv, qb)), shape=(nvv, n))
    obstacle = np.zeros(nvv, dtype=bool)
    obstacle[qv] = True
    obstacle[nv0:] = False
    return FullMesh(xy=xy, tri=tri, face=face, fcov=_incidence(cover, n),
                    fzone=_incidence(zone, n), pa=pa, pb=pb, pu=pu, pv=pv, T=T, fa=fa_, fu=fu_,
                    fv=fv_, Tf=Tf, vb=vb, obstacle=obstacle, sigma=fs.sigma, unflat=conf.unflat,
                    splits=conf.splits, rounds=conf.rounds, flat=int(drop.sum()),
                    skipped=int(skipped))


# ------------------------------------------------------------------ cells

@dataclass(frozen=True, eq=False)
class FullCells:
    """The fixed mesh's cells: triangles merged across zero-distance passages within a face
    (and, `lumped`, every building's interior triangles into one cell: one node per building,
    the study's alternative); per passage between two cells of weight at least T_MIN sigma its
    cell pair (`ea`, `eb`) and passage id (`pe`); per street face its cell, finite ones (`fc`,
    ids `pf`) and ground ones (`gcell`, ids `pg`)."""
    cell: NDArray[np.int64]
    n: int
    ea: NDArray[np.int64]
    eb: NDArray[np.int64]
    pe: NDArray[np.int64]
    fc: NDArray[np.int64]
    pf: NDArray[np.int64]
    gcell: NDArray[np.int64]
    pg: NDArray[np.int64]


def full_cells(fm: FullMesh, *, lumped: bool) -> FullCells:
    nt = fm.nt
    merge = np.isinf(fm.T) & (fm.face[fm.pa] == fm.face[fm.pb])
    a, b = [fm.pa[merge]], [fm.pb[merge]]
    if lumped:
        cov = fm.fcov.tocsr()
        single = np.diff(cov.indptr) == 1
        tri_b = np.flatnonzero(single[fm.face])
        key = cov.indices[cov.indptr[fm.face[tri_b]]]
        order = np.argsort(key, kind="stable")
        tb_, kb = tri_b[order], key[order]
        same = kb[1:] == kb[:-1]
        a.append(tb_[1:][same])
        b.append(tb_[:-1][same])
    a_, b_ = np.concatenate(a), np.concatenate(b)
    _n, lab = connected_components(sp.csr_matrix((np.ones(len(a_)), (a_, b_)), shape=(nt, nt)),
                                   directed=False)
    keep = (lab[fm.pa] != lab[fm.pb]) & (fm.T >= gf.T_MIN * fm.sigma)
    if np.isinf(fm.T[keep]).any():
        raise ValueError("a passage of one circumcentre between two cells: a face boundary the "
                         "conformity left co-circular")
    finite = np.isfinite(fm.Tf)
    return FullCells(cell=lab.astype(np.int64), n=int(lab.max()) + 1, ea=lab[fm.pa[keep]],
                     eb=lab[fm.pb[keep]], pe=np.flatnonzero(keep), fc=lab[fm.fa[finite]],
                     pf=np.flatnonzero(finite), gcell=lab[fm.fa[~finite]],
                     pg=np.flatnonzero(~finite))


# ------------------------------------------------------------------ the fixed mesh, cached

class Fixed(NamedTuple):
    """A block's fixed mesh and what its baseline fixes (module docstring): the mesh, its cells,
    the demand on the triangles reaching the street with every footprint standing
    (gap_flow.demand_on; per point its triangle `ptri`, -1 in the band) and each home's ring
    (gap_flow.ring_of)."""
    fm: FullMesh
    cells: FullCells
    dem: gf.PointDemand
    ptri: NDArray[np.int64]
    ring: gf.Ring
    t_build: float

    @property
    def pcell(self) -> NDArray[np.int64]:
        """Per demand point its cell (-1 in the band)."""
        return np.where(self.ptri >= 0, self.cells.cell[np.maximum(self.ptri, 0)], -1)

    @staticmethod
    def key(block, spec: gc.Spec, fs: gf.FlowSpec, form: gf.Width) -> str:
        """The identity: the block's footprints (clearings.fingerprint), its streets and boundary,
        and every constant that can move the mesh."""
        geo = gc.geometry(block)
        ident = dict(
            block=block.block_id, footprints=clearings.fingerprint(block.buildings.outlines),
            streets=hashlib.sha256(shapely.to_wkb(geo.streets)).hexdigest(),
            boundary=hashlib.sha256(shapely.to_wkb(geo.boundary)).hexdigest(),
            spec=spec.name, flow=fs.name, sigma=fs.sigma, offset=fs.offset, form=form.name,
            rev=MESH_REV, conform=CONFORM_ROUNDS, snap=LINE_SNAP_M,
            consts=[gc.TOL, gc.DUST_M2, gc.DELAUNAY_TOL, gc.GROW_TOL_REL, gc.SNAP_M, gc.MERGE_M,
                    gf.MERGE_REL, gf.T_MIN, gf.FLAT_REL, gf.STEINER_W, gf.INSERT_PASSES])
        return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:24]

    @classmethod
    def of(cls, block, spec: gc.Spec, fs: gf.FlowSpec, form: gf.Width, *, lumped: bool,
           cache: Path) -> Fixed:
        """The block's fixed mesh: from `cache` if built before, else built and stored there
        (the cells, cheap, are made at each load: `lumped` is not in the key)."""
        path = cache / f"{block.block_id}__{cls.key(block, spec, fs, form)}.npz"
        if path.exists():
            return cls._load(path, lumped=lumped)
        t0 = time.time()
        geo = gc.geometry(block)
        fm = full_mesh(geo, spec, fs, form)
        cells = full_cells(fm, lumped=lumped)
        # the baseline's reach: every footprint standing
        k0 = np.zeros(len(geo.polys))
        kt = form.terms(geo, fm, fs.quad).value(k0)
        fe, fb = form.fill_pairs(fm)
        kp, _binds = _passage_factors(fm, kt, k0, Fill(e=fe, b=fb))
        live0 = _reach(fm, cells, kt, kp).live[cells.cell]
        _ib, band_open = gc._open_ground(geo, np.ones(len(geo.polys), dtype=bool), spec.band_m,
                                         carve=None, grow=0.0)
        ids = np.flatnonzero(live0)
        dem = gf.demand_on(geo, spec, fs.quad, fm.xy, fm.tri[ids], band_open)
        nin = len(ids) * fs.quad
        ptri = -np.ones(len(dem.xy), dtype=np.int64)
        ptri[:nin] = ids[np.arange(nin) // fs.quad]
        ring = gf.ring_of(geo, spec.ring_m, fm.xy, fm.tri, band_open)
        fx = cls(fm=fm, cells=cells, dem=dem, ptri=ptri, ring=ring, t_build=time.time() - t0)
        cls._store(fx, path)
        return fx

    @staticmethod
    def _store(fx: Fixed, path: Path) -> None:
        fm, dem, ring = fx.fm, fx.dem, fx.ring
        arrs: dict[str, NDArray[Any]] = {}
        for name in ("xy", "tri", "face", "pa", "pb", "pu", "pv", "T", "fa", "fu", "fv", "Tf",
                     "obstacle"):
            arrs[f"fm_{name}"] = getattr(fm, name)
        for name in ("fcov", "fzone", "vb"):
            M = getattr(fm, name).tocsr()
            arrs[f"fm_{name}_data"], arrs[f"fm_{name}_indices"] = M.data, M.indices
            arrs[f"fm_{name}_indptr"], arrs[f"fm_{name}_shape"] = M.indptr, np.array(M.shape)
        arrs["fm_scalars"] = np.array([fm.sigma, fm.unflat, fm.splits, fm.rounds, fm.flat,
                                       fm.skipped, fx.t_build])
        arrs["ptri"] = fx.ptri
        for name in ("xy", "band", "walled", "shadowed"):
            arrs[f"dem_{name}"] = getattr(dem, name)
        for name, M in (("dem_S", dem.S.tocsr()), ("ring_tri", ring.tri.tocsr())):
            arrs[f"{name}_data"], arrs[f"{name}_indices"] = M.data, M.indices
            arrs[f"{name}_indptr"], arrs[f"{name}_shape"] = M.indptr, np.array(M.shape)
        arrs["ring_band"] = ring.band
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, **arrs)
        os.replace(tmp, path)

    @classmethod
    def _load(cls, path: Path, *, lumped: bool) -> Fixed:
        z = np.load(path)

        def csr(name: str) -> sp.csr_matrix:
            return sp.csr_matrix((z[f"{name}_data"], z[f"{name}_indices"], z[f"{name}_indptr"]),
                                 shape=tuple(z[f"{name}_shape"]))
        sig, unflat, splits, rounds, flat, skipped, t_build = z["fm_scalars"]
        fm = FullMesh(**{n: z[f"fm_{n}"] for n in ("xy", "tri", "face", "pa", "pb", "pu", "pv",
                                                   "T", "fa", "fu", "fv", "Tf", "obstacle")},
                      fcov=csr("fm_fcov"), fzone=csr("fm_fzone"), vb=csr("fm_vb"),
                      sigma=float(sig), unflat=int(unflat), splits=int(splits),
                      rounds=int(rounds), flat=int(flat), skipped=int(skipped))
        dem = gf.PointDemand(xy=z["dem_xy"], band=z["dem_band"], S=csr("dem_S"),
                             walled=z["dem_walled"], shadowed=z["dem_shadowed"])
        return cls(fm=fm, cells=full_cells(fm, lumped=lumped), dem=dem, ptri=z["ptri"],
                   ring=gf.Ring(tri=csr("ring_tri"), band=z["ring_band"]),
                   t_build=float(t_build))


# ------------------------------------------------------------------ the state

class Fill(NamedTuple):
    """The passages a filled gap holds shut (Width.fill_pairs), each with a building at its ends."""
    e: NDArray[np.int64]
    b: NDArray[np.int64]


def _passage_factors(fm: FullMesh, kt: NDArray[np.float64], k: NDArray[np.float64],
                     fill: Fill) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """(per passage the harmonic mean of its two triangles' factors (the half-cells in series;
    0 if either is), a filled one the sum of its buildings' k where that is less; per passage
    whether its fill binds)."""
    va, vb_ = kt[fm.pa], kt[fm.pb]
    s_ = va + vb_
    kp = np.where(s_ > 0, 2.0 * va * vb_ / np.where(s_ > 0, s_, 1.0), 0.0)
    binds = np.zeros(len(kp), dtype=bool)
    if len(fill.e):
        ssum = np.bincount(fill.e, weights=k[fill.b], minlength=len(kp))
        filled = np.zeros(len(kp), dtype=bool)
        filled[fill.e] = True
        binds = filled & (ssum < kp)
        kp = np.where(filled, np.minimum(kp, ssum), kp)
    return kp, binds


class Reach(NamedTuple):
    """A state's cells: `live` (reached from the street through conducting passages), `ground`
    (a conducting obtuse exit's), `unk` per cell its unknown (-1: not live, or ground); per
    static passage (FullCells.pe) its conductance `Tp`, per finite street face its `Gf`."""
    live: NDArray[np.bool_]
    ground: NDArray[np.bool_]
    unk: NDArray[np.int64]
    Tp: NDArray[np.float64]
    Gf: NDArray[np.float64]

    @property
    def N(self) -> int:
        return int((self.unk >= 0).sum())


def _reach(fm: FullMesh, c: FullCells, kt: NDArray[np.float64], kp: NDArray[np.float64]) -> Reach:
    Tp = fm.T[c.pe] * kp[c.pe]
    kf = kt[fm.fa]
    Gf = fm.Tf[c.pf] * kf[c.pf]
    if (Tp < 0).any() or (Gf < 0).any():
        raise ValueError("a negative conductance")
    ground = np.zeros(c.n, dtype=bool)
    ground[c.gcell[kf[c.pg] > 0]] = True
    on = Tp > 0
    street = c.n
    touch = np.unique(np.concatenate([c.fc[Gf > 0], np.flatnonzero(ground)]))
    r = np.concatenate([c.ea[on], touch])
    cc_ = np.concatenate([c.eb[on], np.full(len(touch), street)])
    _k, comp = connected_components(sp.csr_matrix((np.ones(len(r)), (r, cc_)),
                                                  shape=(c.n + 1, c.n + 1)), directed=False)
    live = comp[:c.n] == comp[street]
    unk = -np.ones(c.n, dtype=np.int64)
    free = live & ~ground
    unk[free] = np.arange(int(free.sum()))
    return Reach(live=live, ground=ground, unk=unk, Tp=Tp, Gf=Gf)


def _assemble(c: FullCells, r: Reach) -> sp.csr_matrix:
    """The state's operator over its unknowns (a passage to a ground cell on the diagonal only)."""
    N = r.N
    ua, ub = r.unk[c.ea], r.unk[c.eb]
    T = r.Tp
    diag = (np.bincount(ua[ua >= 0], weights=T[ua >= 0], minlength=N)
            + np.bincount(ub[ub >= 0], weights=T[ub >= 0], minlength=N))
    uf = r.unk[c.fc]
    diag += np.bincount(uf[uf >= 0], weights=r.Gf[uf >= 0], minlength=N)
    both = (ua >= 0) & (ub >= 0) & (T > 0)
    rows = np.concatenate([ua[both], ub[both], np.arange(N)])
    cols = np.concatenate([ub[both], ua[both], np.arange(N)])
    vals = np.concatenate([-T[both], -T[both], diag])
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def _rank(fm: FullMesh, c: FullCells) -> NDArray[np.int64]:
    """Per cell its position in one elimination order of the whole cell graph (every static
    passage; gap_flow.dissection at the cells' centroids), every state's unknowns factored in
    the order it restricts to them: one ordering for all, its separators separating any
    subgraph."""
    cen = fm.xy[fm.tri].mean(axis=1)
    cnt = np.bincount(c.cell, minlength=c.n)
    x = np.bincount(c.cell, weights=cen[:, 0], minlength=c.n) / cnt
    y = np.bincount(c.cell, weights=cen[:, 1], minlength=c.n) / cnt
    G = sp.csr_matrix((np.ones(2 * len(c.ea)), (np.concatenate([c.ea, c.eb]),
                                                 np.concatenate([c.eb, c.ea]))),
                      shape=(c.n, c.n))
    perm = gf.dissection(G.indptr, G.indices, x, y, gf.ND_LEAF)
    rank = np.empty(c.n, dtype=np.int64)
    rank[perm] = np.arange(c.n)
    return rank


class State(NamedTuple):
    """A clearing's state on the fixed mesh: the factors (per building `k`, per triangle `kt`,
    per passage `kp`, whether its fill binds), its reach and, once factored, its operator and
    the operator's factorization."""
    k: NDArray[np.float64]
    kt: NDArray[np.float64]
    kp: NDArray[np.float64]
    binds: NDArray[np.bool_]
    reach: Reach
    A: sp.csr_matrix | None
    lu: Any


class Timing:
    """Where a model's time goes, summed over its calls: the factors (the terms and the passages'
    means), the reach, the assembly, the factorization, the solves (each refined), the
    gradient's back-propagation."""
    KEYS = ("factors", "reach", "assembly", "factorization", "solves", "gradient")

    def __init__(self) -> None:
        self.t = {k: 0.0 for k in self.KEYS}
        self.n = {k: 0 for k in self.KEYS}

    def add(self, key: str, t: float) -> None:
        self.t[key] += t
        self.n[key] += 1


# ------------------------------------------------------------------ the scorer and the seam

class Lenses(NamedTuple):
    """A 0/1 clearing's scores (module docstring): P (each home injecting its population),
    P_{1/2}, J_2, the enclosed population's share and count, the homes' escape times under the
    population's demand, the homes the flow model reaches, the state's unknowns."""
    P: float
    Pq: float
    J2: float
    enclosed: float
    n_enclosed: int
    u: NDArray[np.float64]
    live: NDArray[np.bool_]
    unknowns: int


PQ = common.PriorityDemand(0.5)         # the equity view (owner 2026-10-13)


class Model:
    """One block's clearing model on its fixed mesh (module docstring): the baseline, the scorer,
    free first and the seam's ClearingModel. `pop` each building's population (area: the
    owner's), `demand` what each home injects in J_power (common.Demand)."""

    def __init__(self, block, fx: Fixed, *, spec: gc.Spec, fs: gf.FlowSpec, form: gf.Width,
                 pop: NDArray[np.float64], demand: common.Demand):
        self.block, self.fx, self.spec, self.fs, self.form = block, fx, spec, fs, form
        self.geo = gc.geometry(block)
        self.fm, self.c = fx.fm, fx.cells
        self.n_ = len(self.geo.polys)
        self.terms = form.terms(self.geo, self.fm, fs.quad)
        fe, fb = form.fill_pairs(self.fm)
        self.fill = Fill(e=fe, b=fb)
        self.pop = pop
        self.cost_ = pop / pop.sum()
        self.removed = np.zeros(self.n_, dtype=bool)
        self.polys_ = np.asarray(block.buildings.outlines)
        self.tree_ = shapely.STRtree(self.polys_)
        self.timing = Timing()
        self.rank = _rank(self.fm, self.c)
        self.dem, self.pcell = fx.dem, fx.pcell
        self.nodem = fx.dem.walled            # the homes with no demand point: out of every J
        # the baseline, every footprint standing: the walled-in homes, the escape times under the
        # population's demand (P_{1/2}'s weights) and under the run's (J_power's), the lenses'
        # references
        st0 = self.state(np.zeros(self.n_))
        self.walled0 = ~self.live_homes_of(st0)
        self.u_pop0 = self.home_u(st0, self.solve(st0, self.pop))
        self.inj = demand.injection(pop, self.u_pop0, self.nodem)
        self.inj_q = PQ.injection(pop, self.u_pop0, self.nodem)
        self.demand = demand
        u = self.solve(st0, self.inj)
        self.u0 = self.home_u(st0, u)
        self.P0 = float(self._rhs(st0, self.inj) @ u)
        self.base = self.lenses_of(st0)
        self.cells0 = st0.reach.N

    # -- the state

    def factors(self, k: NDArray[np.float64]) -> State:
        """The state's factors and reach (no operator: the enclosed homes need no solve)."""
        t = time.time()
        kt = self.terms.value(k)
        kp, binds = _passage_factors(self.fm, kt, k, self.fill)
        self.timing.add("factors", time.time() - t)
        t = time.time()
        r = _reach(self.fm, self.c, kt, kp)
        self.timing.add("reach", time.time() - t)
        return State(k=k, kt=kt, kp=kp, binds=binds, reach=r, A=None, lu=None)

    def state(self, k: NDArray[np.float64]) -> State:
        """The state of the clearing k, its operator assembled and factored."""
        st = self.factors(k)
        t = time.time()
        A = _assemble(self.c, st.reach)
        self.timing.add("assembly", time.time() - t)
        t = time.time()
        on = st.reach.unk >= 0
        lu = gf.factor_ordered(A, np.argsort(self.rank[np.flatnonzero(on)][
            np.argsort(st.reach.unk[on])], kind="stable"))
        self.timing.add("factorization", time.time() - t)
        return st._replace(A=A, lu=lu)

    def _unk_pt(self, st: State) -> NDArray[np.int64]:
        return np.where(self.pcell >= 0, st.reach.unk[np.maximum(self.pcell, 0)], -1)

    def _rhs(self, st: State, inj: NDArray[np.float64]) -> NDArray[np.float64]:
        """The right-hand side of the injection `inj` per home (a point in a ground cell or the
        band escapes at once)."""
        f_pt = self.dem.S @ inj
        uk = self._unk_pt(st)
        on = uk >= 0
        return np.bincount(uk[on], weights=f_pt[on], minlength=st.reach.N)

    def solve(self, st: State, inj: NDArray[np.float64]) -> NDArray[np.float64]:
        """The potential per unknown under the injection `inj` per home."""
        t = time.time()
        u = st.lu.solve(self._rhs(st, inj))
        self.timing.add("solves", time.time() - t)
        return np.asarray(u)

    def home_u(self, st: State, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per home its escape time (nan with no demand) from the potential per unknown."""
        uk = self._unk_pt(st)
        u_pt = np.where(uk >= 0, u[np.maximum(uk, 0)], 0.0)
        uh = np.asarray(self.dem.S.T @ u_pt).ravel()
        return np.where(self.nodem, np.nan, uh)

    def J(self, u_home: NDArray[np.float64], inj: NDArray[np.float64], power: float) -> float:
        ok = ~self.nodem
        return float((inj[ok] * u_home[ok] ** power).sum())

    def live_homes_of(self, st: State) -> NDArray[np.bool_]:
        """Per home: some triangle within ring_m of its outline reaches the street in the state
        (or the open band does): the flow model's own reach."""
        return np.asarray(self.fx.ring.live(st.reach.live[self.c.cell]))

    def grad(self, st: State, u: NDArray[np.float64],
             lam: NDArray[np.float64]) -> NDArray[np.float64]:
        """dJ / dk per building by the adjoint (lam: J's adjoint; u for P): dJ / dw = -du dlam
        per conductance, through each passage's harmonic mean and each triangle's terms
        (Width.terms.vjp: ties shared)."""
        t = time.time()
        fm, c, r = self.fm, self.c, st.reach
        uc = np.zeros(c.n)
        lc = np.zeros(c.n)
        on = r.unk >= 0
        uc[on], lc[on] = u[r.unk[on]], lam[r.unk[on]]
        gp = np.zeros(len(fm.pa))
        gp[c.pe] = -fm.T[c.pe] * (uc[c.ea] - uc[c.eb]) * (lc[c.ea] - lc[c.eb])
        gf_ = np.zeros(len(fm.fa))
        gf_[c.pf] = -fm.Tf[c.pf] * uc[c.fc] * lc[c.fc]
        va, vb_ = st.kt[fm.pa], st.kt[fm.pb]
        s2 = (va + vb_) ** 2
        free = ~st.binds
        ga = np.where(s2 > 0, 2.0 * vb_ ** 2 / np.where(s2 > 0, s2, 1.0), 0.0) * free
        gb = np.where(s2 > 0, 2.0 * va ** 2 / np.where(s2 > 0, s2, 1.0), 0.0) * free
        nt = fm.nt
        gt = (np.bincount(fm.pa, weights=gp * ga, minlength=nt)
              + np.bincount(fm.pb, weights=gp * gb, minlength=nt)
              + np.bincount(fm.fa, weights=gf_, minlength=nt))
        dk = self.terms.vjp(st.k, gt)
        if len(self.fill.e):
            sel = st.binds[self.fill.e]
            dk += np.bincount(self.fill.b[sel], weights=gp[self.fill.e[sel]], minlength=self.n_)
        self.timing.add("gradient", time.time() - t)
        return np.asarray(dk)

    # -- the scorer

    def lenses_of(self, st: State) -> Lenses:
        u = self.solve(st, self.pop)
        uh = self.home_u(st, u)
        P = float(self._rhs(st, self.pop) @ u)
        Pq = float(self._rhs(st, self.inj_q) @ self.solve(st, self.inj_q))
        J2 = self.J(uh, self.pop, 2.0)
        live = self.live_homes_of(st)
        left = self.walled0 & ~live
        return Lenses(P=P, Pq=Pq, J2=J2, enclosed=float(self.cost_[left].sum()),
                      n_enclosed=int(left.sum()), u=uh, live=live, unknowns=st.reach.N)

    def lenses(self, removed: NDArray[np.bool_]) -> Lenses:
        return self.lenses_of(self.state(self.k_of(removed)))

    @staticmethod
    def k_of(removed: NDArray[np.bool_]) -> NDArray[np.float64]:
        return removed.astype(np.float64)

    # -- seam.ClearingModel

    @property
    def n(self) -> int:
        return self.n_

    @property
    def cost(self) -> NDArray[np.float64]:
        return self.cost_

    @property
    def polys(self) -> NDArray[np.object_]:
        return self.polys_

    @property
    def tree(self) -> shapely.STRtree:
        return self.tree_

    @property
    def cells(self) -> int:
        return self.cells0

    def release(self) -> None:
        """Nothing on a device."""

    def baseline(self, power: float) -> tuple[float, float]:
        return self.J(self.u0, self.inj, power), self.P0

    def exact(self, removed: NDArray[np.bool_], power: float) -> tuple[float, float]:
        st = self.state(self.k_of(removed))
        u = self.solve(st, self.inj)
        return self.J(self.home_u(st, u), self.inj, power), float(self._rhs(st, self.inj) @ u)

    def enclosed_left(self, removed: NDArray[np.bool_]) -> float:
        left = self.walled0 & ~self.live_homes_of(self.factors(self.k_of(removed)))
        return float(self.cost_[left].sum())

    def tension(self, power: float, *, restore: bool) -> Tension:
        """clear.Clearing.tension on the fixed mesh: the gradient of J_power at the eps world
        (clear.EPS) of the current clearing, a building's gain the first-order change of moving
        its k from where it is to where the move takes it (adding: eps to 1; putting back: 1 to
        eps). One solve, two for power != 1."""
        eps = clear.EPS
        k = np.where(self.removed, 1.0, eps)
        st = self.state(k)
        u = self.solve(st, self.inj)
        uh = self.home_u(st, u)
        lam = u if power == 1.0 else self.solve(
            st, np.where(self.nodem, 0.0, self.inj * power * np.nan_to_num(uh) ** (power - 1.0)))
        dk = self.grad(st, u, lam)
        g = -(np.where(self.removed, eps, 1.0) - k) * dk
        g[~self.removed if restore else self.removed] = -np.inf
        return Tension(model=self, st=st, u=u, home_u=uh, g_=g)

    def free_first(self, *, budget: float, held: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """gap_complex.free_first on the complex grown as the form grows its footprints, read at
        its filtration width (module docstring), checked home for home by the flow model's own
        reach: its walled-in homes at baseline, and those it leaves walled in."""
        fr = gc.free_first(self.geo, self.spec, w=self.form.filtration_w, grow=self.form.grow,
                           budget=budget, cost=self.cost_, pop=self.pop, held0=held, cx0=None)
        bid = self.block.block_id
        if not np.array_equal(fr.target, self.walled0):
            raise RuntimeError(f"{bid}: the complex walls in homes "
                               f"{np.flatnonzero(fr.target ^ self.walled0).tolist()} at baseline "
                               "where the flow model does not, or the other way round")
        left = self.walled0 & ~self.live_homes_of(self.factors(self.k_of(fr.held)))
        if not np.array_equal(left, fr.left):
            raise RuntimeError(f"{bid}: free first and the flow model part on homes "
                               f"{np.flatnonzero(left ^ fr.left).tolist()}")
        n0 = int(self.walled0.sum())
        print(f"{bid} free: {n0 - int(left.sum())} of {n0} walled-in homes freed, D "
              f"{float(self.cost_[fr.held & ~held].sum()):.4f} on {len(fr.crossings)} crossings "
              f"({fr.passes} passes), {int(left.sum())} left walled in", flush=True)
        return np.asarray(fr.held)

    def relaxation(self, power: float, *, q: float, rtol: float, eps: float) -> Relaxation:
        return Relaxation(self, power, q=q, rtol=rtol, eps=eps)


class Tension:
    """A seam.Tension of the fixed mesh (Model.tension)."""

    def __init__(self, *, model: Model, st: State, u: NDArray[np.float64],
                 home_u: NDArray[np.float64], g_: NDArray[np.float64]):
        self.m, self.st, self.u, self.home_u, self.g_ = model, st, u, home_u, g_

    @property
    def g(self) -> NDArray[np.float64]:
        return self.g_

    def weight(self, power: float) -> NDArray[np.float64]:
        return np.asarray(self.m.inj * np.nan_to_num(self.home_u) ** power)

    def flow(self, cand: list[int]) -> Flow:
        """The tension's system, its unknowns the nodes; a candidate's members the unknowns of
        the triangles its clearing re-weights (its terms: its zone under the shy distance)."""
        m, st = self.m, self.st
        A = st.A
        N = st.reach.N
        uk = m._unk_pt(st)
        S = m.dem.S.tocoo()
        on = uk[S.row] >= 0
        load = sp.csr_matrix((S.data[on] * m.inj[S.col[on]], (S.col[on], uk[S.row[on]])),
                             shape=(m.n, N))
        terms = m.terms
        tri_of = terms.row // terms.G
        rows, cols = [], []
        for col, j in enumerate(cand):
            nodes = np.unique(st.reach.unk[m.c.cell[tri_of[terms.b == j]]])
            nodes = nodes[nodes >= 0]
            rows.append(nodes)
            cols.append(np.full(len(nodes), col, dtype=np.int64))
        return Flow(xp=np, A=A, u=self.u, rowsum=np.asarray(A @ np.ones(N)), load=load,
                    rows=np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64),
                    cols=np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64),
                    to_host=np.asarray)


class Relaxation:
    """A seam.Relaxed of the fixed mesh: J_power in the eps world, k_b = eps + (1 - eps)
    proj(x_b)^q (relax.project), its gradient in the projected clearing by the adjoint. `rtol`
    is the raster's iterative tolerance, kept for the seam: every solve here is direct, refined
    to the floating-point floor (gap_flow._Refined)."""

    def __init__(self, model: Model, power: float, *, q: float, rtol: float, eps: float):
        self.m, self.power_, self.q, self.rtol, self.eps = model, power, q, rtol, eps
        self.beta = 0.0
        self.J0_ = model.baseline(power)[0]

    @property
    def c(self) -> Model:
        return self.m

    @property
    def power(self) -> float:
        return self.power_

    @property
    def J0(self) -> float:
        return self.J0_

    def proj(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return relax.project(x, self.beta)

    def dproj(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return relax.dproject(x, self.beta)

    def k_of(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self.eps + (1.0 - self.eps) * self.proj(x) ** self.q)

    def _J(self, st: State) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        u = self.m.solve(st, self.m.inj)
        uh = self.m.home_u(st, u)
        return self.m.J(uh, self.m.inj, self.power_), u, uh

    def value(self, x: NDArray[np.float64]) -> float:
        return self._J(self.m.state(self.k_of(x)))[0]

    def value_sgrad(self, x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        m = self.m
        st = m.state(self.k_of(x))
        J, u, uh = self._J(st)
        p = self.power_
        lam = u if p == 1.0 else m.solve(
            st, np.where(m.nodem, 0.0, m.inj * p * np.nan_to_num(uh) ** (p - 1.0)))
        g = m.grad(st, u, lam) * (1.0 - self.eps)
        if self.q != 1.0:
            g = g * self.q * np.maximum(self.proj(x), 1e-300) ** (self.q - 1.0)
        return J, g

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
        """The eps world as a search.World: J of the clearing (no P here)."""
        return search.Score(self.value(removed.astype(float)), np.nan)

    def metric_world(self, eps: float) -> Relaxation:
        return Relaxation(self.m, self.power_, q=1.0, rtol=self.rtol, eps=eps)


# ------------------------------------------------------------------ the run's metric

@dataclass(frozen=True, kw_only=True)
class ComplexMetric:
    """The metric on the complex (seam.Metric): its identity the complex's Spec (s, ring, band),
    the FlowSpec (R, its grading, sigma, the demand's quadrature, the lattice offset), the width
    form and `lumped` (one node per building: the study's alternative), every one in its name, which
    no raster conductance's can be (`check_disjoint`): its rows never read as the raster's."""
    spec: gc.Spec
    fs: gf.FlowSpec
    form: gf.Width
    lumped: bool

    def __post_init__(self) -> None:
        check_disjoint(self.name)

    @property
    def name(self) -> str:
        return (f"cx+{self.form.name}+s{self.spec.s:g}r{self.spec.ring_m:g}b{self.spec.band_m:g}"
                f"+{self.fs.name}" + ("+lumped" if self.lumped else ""))

    @property
    def sweep(self) -> clear.Sweep:
        """The catchment's sweep: on the host, as every solve here."""
        return clear.CpuSweep()

    def clearing(self, block, *, demand: common.Demand) -> Model:
        fx = Fixed.of(block, self.spec, self.fs, self.form, lumped=self.lumped, cache=CACHE)
        return Model(block, fx, spec=self.spec, fs=self.fs, form=self.form,
                     pop=common.POPULATIONS["area"].weights(np.asarray(block.buildings.outlines)),
                     demand=demand)

    def coarse(self, h: float) -> ComplexMetric:
        raise ValueError(f"a coarse grid (h {h:g}) is the raster's: the complex has none")


def check_disjoint(name: str) -> None:
    """Raises if `name` reads as a raster conductance (lifted.conductance_of, or an along-
    conductance of the rows before the face aperture, lifted.along_of): a complex's rows would
    sit in a raster run's directories."""
    for parse in (lambda: lifted.conductance_of(name, lifted.CpuScans()),
                  lambda: lifted.along_of(name)):
        try:
            parse()
        except ValueError:
            continue
        raise ValueError(f"{name!r} names a raster conductance: the complex's rows would be read "
                         "as the raster's")


_NUM = r"([0-9]+(?:\.[0-9]+)?(?:e-?[0-9]+)?)"
_NAME = re.compile(rf"cx\+(close|shy|ramp){_NUM}(?:s{_NUM})?\+s{_NUM}r{_NUM}b{_NUM}"
                   rf"\+R{_NUM}(?:g([0-9]+)x{_NUM})?q([0-9]+)(?:k{_NUM})?"
                   rf"(?:o(-?{_NUM[1:-1]}),(-?{_NUM[1:-1]}))?(\+lumped)?")


def complex_of(name: str) -> ComplexMetric:
    """The ComplexMetric `name` names (its name's inverse: refused unless it spells it back)."""
    m = _NAME.fullmatch(name)
    if m is None:
        raise ValueError(f"unknown complex metric {name!r}: cx+<form><w>+s<s>r<ring>b<band>+"
                         "R<R>[g<levels>x<grade>]q<quad>[k<sigma>][o<x>,<y>][+lumped]")
    (kind, w, ramp_s, s_, ring, band, R, levels, grade, quad, sigma, ox, oy,
     lumped) = m.groups()
    form: gf.Width
    if kind == "close":
        form = gf.Closing(w=float(w))
    elif kind == "shy":
        form = gf.Shy(w=float(w))
    else:
        form = gf.Ramp(w=float(w), s=float(ramp_s))
    fs = gf.FlowSpec(R=float(R), levels=0 if levels is None else int(levels),
                     grade=0.0 if grade is None else float(grade),
                     sigma=gf.SIGMA_ISO if sigma is None else float(sigma), quad=int(quad),
                     offset=gf.OFFSET if ox is None else (float(ox), float(oy)))
    metric = ComplexMetric(spec=gc.Spec(s=float(s_), ring_m=float(ring), band_m=float(band)),
                           fs=fs, form=form, lumped=lumped is not None)
    if metric.name != name:
        raise ValueError(f"{name!r} spells the complex metric {metric.name!r}")
    return metric
