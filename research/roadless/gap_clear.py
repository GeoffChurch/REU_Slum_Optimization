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

THE SYSTEM (`System`, `Graph`; NOTES, "The complex as the runtime: its speed and a GPU path"). A
cell structure's operator lives on one fixed pattern: every pair of cells a static passage joins
and the diagonal, every cell in every state. A state's reach is a search from the street through
the pairs that conduct; its values are one pass over the pattern: an unknown row its conductances,
a closed passage an explicit zero, a cell off the unknowns (walled in, or ground) an identity row.
The run's factorization (gap_factor, a Strategy resolved where the run is configured: SuperLU on
a core, restricting the pattern to the state's unknowns in one nested-dissection order; cuDSS on
a GPU, refactoring the whole pattern) makes one plan per pattern, so exact scores, tensions,
SIMP's iterations and the polish share it. Potentials, right-hand sides and the adjoint are per
cell (0 off the unknowns).

THE IDENTITY (`ComplexMetric.name`): every configurable constant, and the code's own: every
public module-level constant of this module, gap_complex, gap_flow and gap_factor (and clear.EPS,
the tension's) digested at import (CODE, `code_constants`), the few that cannot change a result
exempt with their reasons (`_EXEMPT`). Editing a constant renames the metric, and the mesh cache's
key carries the digest too.

THE SCORER (`Model`). The demand is gap_flow's (the raster's Placement rule on the complex), fixed
at baseline on the triangles reaching the street then; each home's escape time the
injection-weighted mean potential over its points. P (each home injecting its population), P_{1/2}
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

THE OPTIMIZERS run on these types themselves: `Model` (search.py's state, relax.py's and
polish_greedy.py's model), its eps world `Relaxation`, its `Tension` and `ComplexMetric`, the run's
metric. (They were the seam's Protocols while the raster stood beside them, seam.py; folded when it
was retired: NOTES, "The complex as the runtime: the raster retired".)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from collections.abc import Callable
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
import digests
import gap_complex as gc
import gap_exits as gx
import gap_factor
import gap_flow as gf
import relax
import search
from gap_complex import legalize

HERE = Path(__file__).resolve().parent
# the fixed meshes, by identity (`Fixed.of`; git-ignored): $GAP_CACHE where a machine sets it
# (cluster_env.sh: a node's own disk, not the shared home's quota), else beside this file
CACHE = Path(os.environ["GAP_CACHE"]) if "GAP_CACHE" in os.environ else HERE / "gap_cache"
CONFORM_ROUNDS = 64     # conformity: rounds at most (each splits every bad edge it finds; a mesh
                        # still holding one after them raises)
MESH_REV = 3            # the fixed mesh's construction, in its cache key with every meshing
                        # constant: raised by any change to the code that moves a mesh
LINE_SNAP_M = 1e-4      # the linework's precision (m): a crack narrower than this conducts
                        # nothing under any passable width while either footprint stands (it is
                        # in their zero zone, or flat: gap_flow.FLAT_REL s is 1.25e-5 m), and with
                        # both cleared it is nothing; the vertices move by at most half of it
BAND_TOL_M = LINE_SNAP_M    # a boundary edge whose midpoint lies this near the band's line is a
                            # street face (its vertices snapped: gap_complex.TOL lost the exits)


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
    # noded by snap-rounding (OverlayNG at a grid of LINE_SNAP_M): lines closer than that are
    # one, so a hairline crack between two footprints is a shared boundary, not a sliver face of
    # flat triangles (a wall between the two interiors once both are cleared). Snapping the noded
    # set afterwards (set_precision) left dangles where offset arcs nearly coincide, and
    # polygonize lost the faces they bound (38136: a crack between two footprints joined to the
    # open ground, its homes' escape times 3% low)
    noded = shapely.union_all(np.concatenate([clipped, [shapely.boundary(region)]]),
                              grid_size=LINE_SNAP_M)
    noded = shapely.segmentize(noded, spec.s)
    faces = shapely.get_parts(shapely.polygonize(shapely.get_parts(noded)))
    faces = faces[shapely.area(faces) >= gc.DUST_M2]
    rep = shapely.point_on_surface(faces)
    # the rings of the region's holes polygonize too: a street inside the block (its band an
    # island of the region) came back as a face, meshed over the band, its line no exit (23597:
    # 8,685 m^2 of band meshed, 470 m of exits lost)
    shapely.prepare(region)
    inside_region = shapely.contains(region, rep)
    faces, rep = faces[inside_region], rep[inside_region]
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


def full_mesh(geo: gc.Geometry, spec: gc.Spec, fs: gf.FlowSpec, form: gf.Width, *,
              box: tuple[float, float, float, float]) -> FullMesh:
    """The fixed mesh of `geo`'s footprints (a carving's atoms), its Steiner lattice laid over
    `box` (`lattice_box`: the baseline open ground's, the buildings' own)."""
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
    T, drop = _weights(xy, tri, face, cc, pa, pb, pu, pv, fs.sigma)
    # street faces: boundary edges on the band's line
    single = np.ones(len(key), dtype=bool)
    single[i0] = False
    single[i1] = False
    bi = np.flatnonzero(single)
    mid = 0.5 * (xy[es[bi, 0]] + xy[es[bi, 1]])
    shapely.prepare(band_line := shapely.boundary(geo.streets.buffer(spec.band_m)))
    on_band = shapely.dwithin(band_line, shapely.points(mid[:, 0], mid[:, 1]), BAND_TOL_M)
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


def lattice_box(geo: gc.Geometry, spec: gc.Spec, form: gf.Width) -> tuple[float, float, float,
                                                                          float]:
    """The box of the baseline's open ground (every footprint of `geo`, the buildings, standing,
    grown as the form grows them): the Steiner lattice's frame, the rebuild's own. From the
    buildings, not a carving's atoms: the atoms' union is the same ground, and GEOS's union of
    pieces sharing their cuts can fail (38988: a TopologyException)."""
    base, _band = gc._open_ground(geo, np.ones(len(geo.polys), dtype=bool), spec.band_m,
                                  carve=None, grow=form.grow)
    return base.bounds


def _weights(xy, tri, face, cc, pa, pb, pu, pv, sigma: float
             ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """(per passage sigma |e| / d_e, inf where its two triangles share a circumcentre within a
    face, 0 beside a flat triangle; whether it is dropped so). Refuses a passage whose centres
    are in the wrong order (or, across faces, one): the mesh not Delaunay."""
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
                                         sigma * L / np.maximum(d, 1e-300)))
    return T, drop


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


# ------------------------------------------------------------------ the footprints' atoms: carving

class Prefix(NamedTuple):
    """A road network's prefix as a state of a carving (Carving): `net` the network (its position
    among the carving's), `stop` its stop (1 .. that network's stops; 0 carves nothing)."""
    net: int
    stop: int


_NO_ROADS = Prefix(net=0, stop=0)      # a clearing's state: no road carved


class Carving(NamedTuple):
    """The footprints a fixed mesh is built on, its ATOMS, and how road networks' prefixes carve
    them (the road rule, budget.road_corridor: exactly a prefix's corridor freed; NOTES, "The
    complex as the runtime: the ports and the cross-checks"). A network's scored prefixes are its
    STOPS, their corridors nested, C_1 within C_2 ... within C_K (each the union of the prefix's
    roads buffered by half their width: reblock.budget.road_corridor, in the block's local
    coordinates). Every footprint is cut by every network's corridors: its atoms are the pieces of
    their arrangement, each carved by a network from the first of its stops whose corridor holds
    it (0: never; pieces under gap_complex.DUST_M2 dropped, as the open ground's dust); a building
    no corridor touches is one atom, itself. A state at a network's stop k stands the atoms of
    every building it does not clear less those that network has carved by then: b less C_k, the
    footprint the complex rebuilt with the prefix carved stands (gap_complex.obstacles), and a
    footprint's zone (within w / 2 of it) is the union of its standing atoms' zones, so the shy
    distance's factor is exact on the atoms. Homes (the demand, the ring, the population and B (i))
    stay the buildings. No networks: every building one atom, the plain mesh. `polys` per atom its
    footprint, `parent` its building, `stop` per atom and network the stop it is carved from;
    `stops` per network its stops; `digest` the corridors' identity (the mesh cache's file name;
    '' for none)."""
    polys: NDArray[np.object_]
    parent: NDArray[np.int64]
    stop: NDArray[np.int64]               # (atoms, networks)
    stops: tuple[int, ...]
    digest: str

    @property
    def n(self) -> int:
        return len(self.polys)

    @staticmethod
    def none(geo: gc.Geometry) -> Carving:
        """No roads: every building one atom (the plain fixed mesh)."""
        n = len(geo.polys)
        return Carving(polys=geo.polys, parent=np.arange(n, dtype=np.int64),
                       stop=np.zeros((n, 0), dtype=np.int64), stops=(), digest="")

    @staticmethod
    def of(geo: gc.Geometry, networks: list[list[Any]]) -> Carving:
        """The atoms of `geo`'s footprints cut by each network's nested stop corridors (local
        coordinates, in order); refuses a network whose corridors are not nested (a prefix's
        corridor holds every shorter one's, but for slivers under the linework's precision,
        LINE_SNAP_M along the shorter one's outline: each corridor is unioned afresh, as the
        rebuild and the raster carve it, and two unions part by such slivers, 3.4e-6 m^2 on
        18895)."""
        if not networks or not all(networks):
            raise ValueError("a carving needs a network with a stop: Carving.none is the plain "
                             "mesh")
        nets = [[shapely.make_valid(c) for c in cs] for cs in networks]
        for i, cs in enumerate(nets):
            for k in range(1, len(cs)):
                out = shapely.area(shapely.difference(cs[k - 1], cs[k]))
                if out > LINE_SNAP_M * shapely.length(cs[k - 1]):
                    raise ValueError(f"network {i}: stop {k + 1}'s corridor misses {out:.2e} m^2 "
                                     f"of stop {k}'s: not one network's nested prefixes")
        polys = np.asarray(geo.polys, dtype=object)
        parent = np.arange(len(polys), dtype=np.int64)
        tag = np.zeros((len(polys), len(nets)), dtype=np.int64)
        for i, cs in enumerate(nets):
            for k, c in enumerate(cs, start=1):
                hit = np.unique(shapely.STRtree(polys).query(c, predicate="intersects"))
                hit = hit[tag[hit, i] == 0]
                if not len(hit):
                    continue
                inside = shapely.intersection(polys[hit], c)
                outside = shapely.difference(polys[hit], c)
                new_p, new_par, new_tag = [], [], []
                for j, a, z in zip(hit, inside, outside, strict=True):
                    for part in _polygons(a):
                        new_p.append(part)
                        new_par.append(parent[j])
                        t = tag[j].copy()
                        t[i] = k
                        new_tag.append(t)
                    for part in _polygons(z):
                        new_p.append(part)
                        new_par.append(parent[j])
                        new_tag.append(tag[j].copy())
                keep = np.ones(len(polys), dtype=bool)
                keep[hit] = False
                polys = np.concatenate([polys[keep], np.asarray(new_p + [None], dtype=object)[:-1]])
                parent = np.concatenate([parent[keep], np.asarray(new_par, dtype=np.int64)])
                tag = np.concatenate([tag[keep], np.asarray(new_tag, dtype=np.int64).reshape(
                    -1, len(nets))])
        o = np.lexsort((*tag.T[::-1], parent))
        h = hashlib.sha256()
        for cs in nets:
            h.update(b"".join(shapely.to_wkb(np.asarray(cs, dtype=object), hex=False,
                                             output_dimension=2, byte_order=1)))
            h.update(b"|")
        return Carving(polys=polys[o], parent=parent[o], stop=tag[o],
                       stops=tuple(len(cs) for cs in nets), digest=h.hexdigest()[:16])

    def carved(self, prefix: Prefix) -> NDArray[np.bool_]:
        """Per atom: carved by `prefix` (stop 0: none)."""
        if prefix.stop == 0:
            return np.zeros(self.n, dtype=bool)
        if not 1 <= prefix.stop <= self.stops[prefix.net]:
            raise ValueError(f"{prefix}: network {prefix.net} has stops 1 .. "
                             f"{self.stops[prefix.net]}")
        s = self.stop[:, prefix.net]
        return (s >= 1) & (s <= prefix.stop)

    def atom_k(self, k: NDArray[np.float64], *, prefix: Prefix) -> NDArray[np.float64]:
        """Per atom its clearing: its building's k, or 1 where `prefix` carves it."""
        return np.where(self.carved(prefix), 1.0, k[self.parent])

    def to_buildings(self, g: NDArray[np.float64], n: int, *,
                     prefix: Prefix) -> NDArray[np.float64]:
        """A per-atom derivative summed to the buildings (the atoms `prefix` carves held at 1)."""
        return np.bincount(self.parent, weights=np.where(self.carved(prefix), 0.0, g),
                           minlength=n)


def _polygons(g: Any) -> list[Any]:
    """The polygons of a piece (an intersection's or difference's result), each of at least
    gap_complex.DUST_M2."""
    parts = shapely.get_parts(g)
    return [p for p in parts if shapely.get_type_id(p) == 3 and shapely.area(p) >= gc.DUST_M2]


# ------------------------------------------------------------------ the fixed mesh, cached

_CACHE_FORMAT = 2      # what a cached mesh's file holds (in its key; a storage detail, no score's):
                       # 2 the baseline's grown complex beside the mesh


class Fixed(NamedTuple):
    """A block's fixed mesh and what its baseline fixes (module docstring): the mesh, built on the
    footprints' atoms (`carving`: the buildings themselves, or a road network's pieces), the
    demand on the triangles reaching the street with every footprint standing (gap_flow.demand_on;
    per point its triangle `ptri`, -1 in the band), each home's ring (gap_flow.ring_of) and the
    complex grown as the form grows its footprints with every footprint standing (`cx0`, its
    arrays: gap_complex.complex_arrays), free first's baseline. The demand, the rings and cx0 are
    the buildings', whatever the atoms."""
    fm: FullMesh
    dem: gf.PointDemand
    ptri: NDArray[np.int64]
    ring: gf.Ring
    cx0: dict[str, NDArray[Any]]
    t_build: float
    carving: Carving

    def complex0(self, spec: gc.Spec) -> gc.Complex:
        """Free first's baseline complex (gap_complex.free_first's cx0)."""
        return gc.complex_of_arrays(self.cx0, spec)

    @staticmethod
    def key(block, spec: gc.Spec, fs: gf.FlowSpec, form: gf.Width) -> str:
        """The identity: the block's footprints (clearings.fingerprint), its streets and boundary,
        the configurable constants and the code's (CODE: every constant that can move a mesh
        among them)."""
        geo = gc.geometry(block)
        ident = dict(
            block=block.block_id, footprints=clearings.fingerprint(block.buildings.outlines),
            streets=hashlib.sha256(shapely.to_wkb(geo.streets)).hexdigest(),
            boundary=hashlib.sha256(shapely.to_wkb(geo.boundary)).hexdigest(),
            spec=spec.name, flow=fs.name, sigma=fs.sigma, offset=fs.offset, form=form.name,
            code=CODE, format=_CACHE_FORMAT)
        return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:24]

    @classmethod
    def of(cls, block, spec: gc.Spec, fs: gf.FlowSpec, form: gf.Width, *, cache: Path,
           carving: Carving) -> Fixed:
        """The block's fixed mesh on the atoms `carving` (Carving.none: the buildings): from
        `cache` if built before, else built and stored there (a carved mesh's file named by its
        corridors' digest beside the block's identity)."""
        tag = f"__c{carving.digest}" if carving.stops else ""
        path = cache / f"{block.block_id}{tag}__{cls.key(block, spec, fs, form)}.npz"
        if path.exists():
            return cls._load(path, carving)
        t0 = time.time()
        geo = gc.geometry(block)
        atoms = geo._replace(polys=carving.polys)
        fm = full_mesh(atoms, spec, fs, form, box=lattice_box(geo, spec, form))
        cells = full_cells(fm, lumped=False)
        # the baseline's reach: every footprint standing
        k0 = np.zeros(carving.n)
        kt = form.terms(atoms, fm, fs.quad).value(k0)
        fe, fb = form.fill_pairs(fm)
        kp, _binds = _passage_factors(fm, kt, k0, Fill(e=fe, b=fb))
        live0 = Graph(fm, cells, leak=np.zeros(cells.n)).reach(fm, kt, kp).live[cells.cell]
        _ib, band_open = gc._open_ground(geo, np.ones(len(geo.polys), dtype=bool), spec.band_m,
                                         carve=None, grow=0.0)
        ids = np.flatnonzero(live0)
        dem = gf.demand_on(geo, spec, fs.quad, fm.xy, fm.tri[ids], band_open)
        nin = len(ids) * fs.quad
        ptri = -np.ones(len(dem.xy), dtype=np.int64)
        ptri[:nin] = ids[np.arange(nin) // fs.quad]
        ring = gf.ring_of(geo, spec.ring_m, fm.xy, fm.tri, band_open)
        cx0 = gc.complex_arrays(gc.build(geo, spec, np.ones(len(geo.polys), dtype=bool),
                                         carve=None, grow=form.grow))
        fx = cls(fm=fm, dem=dem, ptri=ptri, ring=ring, cx0=cx0, t_build=time.time() - t0,
                 carving=carving)
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
        arrs.update({f"cx_{name}": a for name, a in fx.cx0.items()})
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, **arrs)
        os.replace(tmp, path)

    @classmethod
    def _load(cls, path: Path, carving: Carving) -> Fixed:
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
        if fm.fcov.shape[1] != carving.n:
            raise ValueError(f"{path.name}: a mesh of {fm.fcov.shape[1]} atoms, the carving has "
                             f"{carving.n}")
        return cls(fm=fm, dem=dem, ptri=z["ptri"],
                   ring=gf.Ring(tri=csr("ring_tri"), band=z["ring_band"]),
                   cx0={name[3:]: z[name] for name in z.files if name.startswith("cx_")},
                   t_build=float(t_build), carving=carving)


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
    kp = _harmonic(kt, fm.pa, fm.pb)
    binds = np.zeros(len(kp), dtype=bool)
    if len(fill.e):
        ssum = np.bincount(fill.e, weights=k[fill.b], minlength=len(kp))
        filled = np.zeros(len(kp), dtype=bool)
        filled[fill.e] = True
        binds = filled & (ssum < kp)
        kp = np.where(filled, np.minimum(kp, ssum), kp)
    return kp, binds


@njit(cache=True)
def _harmonic(kt, pa, pb):
    """Per passage 2 a b / (a + b) of its triangles' factors a, b (0 where a + b is)."""
    out = np.empty(len(pa))
    for p in range(len(pa)):
        a, b = kt[pa[p]], kt[pb[p]]
        s = a + b
        out[p] = 2.0 * a * b / s if s > 0 else 0.0
    return out


class Reach(NamedTuple):
    """A state's cells: `live` (reached from the street through conducting passages), `ground`
    (a conducting obtuse exit's), `unknown` (live and not ground: the system's unknowns); per
    static passage (FullCells.pe) its conductance `Tp`, per finite street face its `Gf`, per cell
    its conductance to the street `G` (its faces'), per pair of cells their conductance `W` (the
    sum of their passages')."""
    live: NDArray[np.bool_]
    ground: NDArray[np.bool_]
    unknown: NDArray[np.bool_]
    Tp: NDArray[np.float64]
    Gf: NDArray[np.float64]
    G: NDArray[np.float64]
    W: NDArray[np.float64]

    @property
    def N(self) -> int:
        return int(self.unknown.sum())


@njit(cache=True)
def _conduct(pe, pair, T, kp, npair):
    """(per static passage its conductance T kp, per pair the sum of its passages')."""
    Tp = np.empty(len(pe))
    W = np.zeros(npair)
    for q in range(len(pe)):
        p = pe[q]
        t = T[p] * kp[p]
        Tp[q] = t
        W[pair[q]] += t
    return Tp, W


@njit(cache=True)
def _bfs(indptr, indices, epair, W, seed):
    """The rows reached from `seed` through the pattern's entries whose pair conducts (W > 0)."""
    n = len(seed)
    live = seed.copy()
    stack = np.empty(n, dtype=np.int64)
    top = 0
    for i in range(n):
        if seed[i]:
            stack[top] = i
            top += 1
    while top > 0:
        top -= 1
        i = stack[top]
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            if not live[j] and W[epair[p]] > 0.0:
                live[j] = True
                stack[top] = j
                top += 1
    return live


@njit(cache=True)
def _values(indptr, indices, epair, W, G, unknown):
    """The operator's values on the pattern: an unknown row conducts -W to each unknown
    neighbour (an explicit zero to any other: a passage to a ground cell is on its diagonal
    only) and its diagonal the sum of its pairs' W and its G; a row off the unknowns the
    identity's."""
    n = len(unknown)
    data = np.empty(len(indices))
    for i in range(n):
        s = 0.0
        dpos = -1
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            if j == i:
                dpos = p
                continue
            w = W[epair[p]]
            s += w
            data[p] = -w if unknown[i] and unknown[j] else 0.0
        data[dpos] = s + G[i] if unknown[i] else 1.0
    return data


class Graph:
    """A cell structure's fixed pattern (gap_factor.Pattern: every pair of cells a static passage
    joins, and the diagonal, at the cells' centroids), per off-diagonal entry its pair (`epair`;
    the diagonal's -1), per static passage its pair (`pair`), and per cell its fixed conductance
    to ground (`leak`: the region model's leaks, gap_exits; zero for the per-block exits). A
    state's reach is a search through the conducting pairs from the cells joined to ground (the
    street's faces, the ground cells, the leaking cells), its operator's values one pass over the
    pattern (`_values`): every state on the one pattern, a closed passage an explicit zero, a cell
    off the unknowns an identity row, a leak on its cell's diagonal."""

    def __init__(self, fm: FullMesh, c: FullCells, *, leak: NDArray[np.float64]):
        n = c.n
        self.c = c
        ea, eb = c.ea.astype(np.int64), c.eb.astype(np.int64)
        lo, hi = np.minimum(ea, eb), np.maximum(ea, eb)
        uniq, pair = np.unique(lo * n + hi, return_inverse=True)
        self.pair = pair.astype(np.int64)
        self.npair = len(uniq)
        pi, pj = uniq // n, uniq % n
        diag = np.arange(n)
        rows = np.concatenate([pi, pj, diag])
        cols = np.concatenate([pj, pi, diag])
        tag = np.concatenate([np.arange(self.npair), np.arange(self.npair),
                              -np.ones(n, dtype=np.int64)])
        o = np.lexsort((cols, rows))
        indptr = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
        cen = fm.xy[fm.tri].mean(axis=1)
        cnt = np.bincount(c.cell, minlength=n)
        xy = np.column_stack([np.bincount(c.cell, weights=cen[:, 0], minlength=n) / cnt,
                              np.bincount(c.cell, weights=cen[:, 1], minlength=n) / cnt])
        self.pattern = gap_factor.Pattern(indptr=indptr, indices=cols[o].astype(np.int32),
                                          xy=xy)
        self.epair = tag[o].astype(np.int32)
        # the cells the street reaches when every pair conducts and every street face does (the
        # eps world's every state: its reach without the search)
        street = np.zeros(n, dtype=bool)
        street[c.fc] = True
        street[c.gcell] = True
        self.street = street
        self._grounded(leak)

    def _grounded(self, leak: NDArray[np.float64]) -> None:
        if len(leak) != self.c.n or (leak < 0).any():
            raise ValueError("a leak is a non-negative conductance per cell")
        self.leak = leak
        self.seed_all = self.street | (leak > 0)
        self.live_all = _bfs(self.pattern.indptr, self.pattern.indices, self.epair,
                             np.ones(self.npair), self.seed_all)

    def with_leak(self, leak: NDArray[np.float64]) -> Graph:
        """This pattern with `leak` per cell (one search for the eps world's reach; the pattern,
        and so a factorization's plan of it, unchanged)."""
        g = copy.copy(self)
        g._grounded(leak)
        return g

    def reach(self, fm: FullMesh, kt: NDArray[np.float64], kp: NDArray[np.float64]) -> Reach:
        c = self.c
        Tp, W = _conduct(c.pe, self.pair, fm.T, kp, self.npair)
        kf = kt[fm.fa]
        Gf = fm.Tf[c.pf] * kf[c.pf]
        if (Tp < 0).any() or (Gf < 0).any():
            raise ValueError("a negative conductance")
        ground = np.zeros(c.n, dtype=bool)
        ground[c.gcell[kf[c.pg] > 0]] = True
        G = np.bincount(c.fc, weights=Gf, minlength=c.n)
        p = self.pattern
        seed = ground | (G > 0) | (self.leak > 0)
        if (self.npair == 0 or W.min() > 0) and np.array_equal(seed, self.seed_all):
            live = self.live_all            # every pair, street face and leak conducts
        else:
            live = _bfs(p.indptr, p.indices, self.epair, W, seed)
        return Reach(live=live, ground=ground, unknown=live & ~ground, Tp=Tp, Gf=Gf, G=G, W=W)

    def values(self, r: Reach) -> NDArray[np.float64]:
        """The state's operator on the pattern: its passages, its street faces and the leaks."""
        p = self.pattern
        return np.asarray(_values(p.indptr, p.indices, self.epair, r.W, r.G + self.leak,
                                  r.unknown))


class System:
    """A cell structure of the fixed mesh as each state's linear system: its cells (FullCells),
    their Graph (its leaks among it), the demand's points on them (`Pc`, cells x homes: per home
    its injection's weight on each cell; transposed, `PcT`, a home's escape time from the
    potential per cell) and the factorization's plan of the pattern (gap_factor)."""

    def __init__(self, cells: FullCells, *, graph: Graph, dem: gf.PointDemand,
                 ptri: NDArray[np.int64], factorization: gap_factor.Factorization):
        self.cells = cells
        self.graph = graph
        pcell = np.where(ptri >= 0, cells.cell[np.maximum(ptri, 0)], -1)
        on = np.flatnonzero(pcell >= 0)
        M = sp.csr_matrix((np.ones(len(on)), (pcell[on], on)), shape=(cells.n, len(pcell)))
        self.Pc = (M @ dem.S).tocsr()
        self.PcT = self.Pc.T.tocsr()
        self.planned = factorization.plan(self.graph.pattern)

    @property
    def n(self) -> int:
        return self.cells.n

    def with_graph(self, graph: Graph) -> System:
        """This system on `graph` (the same pattern: its plan kept; a leak set on it)."""
        if graph.pattern is not self.graph.pattern:
            raise ValueError("another pattern: the plan is this pattern's")
        out = copy.copy(self)
        out.graph = graph
        return out


class State(NamedTuple):
    """A clearing's state on one System of the fixed mesh: the road prefix carved (`prefix`: none,
    or a Carving's network at a stop), the factors (per atom `k`, per triangle `kt`, per passage
    `kp`, whether its fill binds), its reach and, once factored, its operator on the pattern
    (cells x cells) and the operator's factorization."""
    sys: System
    prefix: Prefix
    k: NDArray[np.float64]
    kt: NDArray[np.float64]
    kp: NDArray[np.float64]
    binds: NDArray[np.bool_]
    reach: Reach
    A: sp.csr_matrix | None
    factor: gap_factor.Factor | None


@njit(cache=True)
def _grad_tri(pe, ea, eb, pa, pb, T, kt, binds, pf, fc, fa, Tf, u, lam, nt):
    """(per triangle dJ / d(its factor), per static passage dJ / d(its factor)) by the adjoint:
    dJ / dw = -du dlam per conductance, through each passage's harmonic mean (a filled one's
    binding fill takes it instead) and each street face's triangle."""
    gt = np.zeros(nt)
    gp = np.empty(len(pe))
    for q in range(len(pe)):
        p = pe[q]
        g = -T[p] * (u[ea[q]] - u[eb[q]]) * (lam[ea[q]] - lam[eb[q]])
        gp[q] = g
        if binds[p]:
            continue
        va, vb = kt[pa[p]], kt[pb[p]]
        s = va + vb
        if s > 0:
            s2 = s * s
            gt[pa[p]] += g * 2.0 * vb * vb / s2
            gt[pb[p]] += g * 2.0 * va * va / s2
    for q in range(len(pf)):
        f = pf[q]
        gt[fa[f]] += -Tf[f] * u[fc[q]] * lam[fc[q]]
    return gt, gp


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


# ------------------------------------------------------------------ the scorer, the optimizers

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
    free first and what the optimizers read. `pop` each building's population (area: the
    owner's), `demand` what each home injects in J_power (common.Demand); `factorization` the
    run's (gap_factor: its device); `lumped` every building's interior one cell; `tension_eps`
    the eps world the tension ranks in (clear.EPS, or the translucent search's); `exits` the
    region model's (gap_exits: the street boost on the mesh's weights, a closed edge on its street
    faces', the leaks on the cells live at baseline with the edge open; the barriers are in
    `block` and its mesh already: ComplexMetric.fixed and .model). The mesh's atoms (fx.carving: the buildings, or a road network's pieces) carry the
    width form's factors; a clearing k is per building and each atom takes its building's k, or 1
    where the state's road prefix (`stop`) carves it (`road_state`); the homes are the
    buildings."""

    def __init__(self, block, fx: Fixed, *, spec: gc.Spec, fs: gf.FlowSpec, form: gf.Width,
                 pop: NDArray[np.float64], demand: common.Demand,
                 factorization: gap_factor.Factorization, lumped: bool, tension_eps: float,
                 exits: gx.Exits):
        self.block, self.fx, self.spec, self.fs, self.form = block, fx, spec, fs, form
        self.tension_eps = tension_eps
        self.exits = exits
        self.geo = gc.geometry(block)
        fm_open = exits.boost.mesh(fx.fm, self.geo, block)
        self.fm = exits.edge.mesh(fm_open)
        self.n_ = len(self.geo.polys)
        self.carving = fx.carving
        if self.carving.parent.max(initial=-1) >= self.n_:
            raise ValueError("the carving's atoms belong to buildings the block lacks")
        self.terms = form.terms(self.geo._replace(polys=self.carving.polys), self.fm, fs.quad)
        fe, fb = form.fill_pairs(self.fm)
        self.fill = Fill(e=fe, b=fb)
        self.pop = pop
        self.cost_ = pop / pop.sum()
        self.removed = np.zeros(self.n_, dtype=bool)
        self.polys_ = np.asarray(block.buildings.outlines)
        self.tree_ = shapely.STRtree(self.polys_)
        self.timing = Timing()
        cells = full_cells(self.fm, lumped=lumped)
        bare = System(cells, graph=Graph(self.fm, cells, leak=np.zeros(cells.n)), dem=fx.dem,
                      ptri=fx.ptri, factorization=factorization)
        # the leaks on the cells that are unknowns at baseline (every footprint standing) with the
        # edge open (a closed edge grounds no face: its leaks lie where the open edge's street
        # reaches, the cells ground there among them), fixed
        a, b, c = (self.fm.xy[self.fm.tri[:, k]] for k in range(3))
        tri_area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        f0 = self._factors(bare, np.zeros(self.n_), prefix=_NO_ROADS)
        self.leak_site = gx.LeakSite(
            area=np.bincount(cells.cell, weights=tri_area, minlength=cells.n),
            live=bare.graph.reach(fm_open, f0.kt, f0.kp).unknown,
            foot=np.asarray(bare.Pc @ shapely.area(self.polys_)), sigma=fs.sigma)
        self.sys = bare.with_graph(bare.graph.with_leak(exits.leak.per_cell(self.leak_site)))
        self.dem = fx.dem
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

    @property
    def c(self) -> FullCells:
        return self.sys.cells

    # -- the state

    def _factors(self, sys: System, k: NDArray[np.float64], *, prefix: Prefix) -> State:
        t = time.time()
        ka = self.carving.atom_k(k, prefix=prefix)
        kt = self.terms.value(ka)
        kp, binds = _passage_factors(self.fm, kt, ka, self.fill)
        self.timing.add("factors", time.time() - t)
        t = time.time()
        r = sys.graph.reach(self.fm, kt, kp)
        self.timing.add("reach", time.time() - t)
        return State(sys=sys, prefix=prefix, k=ka, kt=kt, kp=kp, binds=binds, reach=r, A=None,
                     factor=None)

    def _state(self, sys: System, k: NDArray[np.float64], *, prefix: Prefix) -> State:
        st = self._factors(sys, k, prefix=prefix)
        t = time.time()
        data = sys.graph.values(st.reach)
        p = sys.graph.pattern
        A = sp.csr_matrix((data, p.indices, p.indptr), shape=(p.n, p.n))
        self.timing.add("assembly", time.time() - t)
        t = time.time()
        f = sys.planned.factor(data, st.reach.unknown)
        self.timing.add("factorization", time.time() - t)
        return st._replace(A=A, factor=f)

    def factors(self, k: NDArray[np.float64]) -> State:
        """The state of the clearing k (no road carved): its factors and reach (no operator: the
        enclosed homes need no solve)."""
        return self._factors(self.sys, k, prefix=_NO_ROADS)

    def state(self, k: NDArray[np.float64]) -> State:
        """The state of the clearing k (no road carved), its operator assembled and factored."""
        return self._state(self.sys, k, prefix=_NO_ROADS)

    def leaked_state(self, leak: gx.Leak, k: NDArray[np.float64]) -> State:
        """The state of the clearing k (no road carved) with `leak` in place of the exits' own,
        on the same cells (leak_site), pattern and plan: a leak's other lam without the model
        built again (region_leaks' trip calibration)."""
        sys = self.sys.with_graph(self.sys.graph.with_leak(leak.per_cell(self.leak_site)))
        return self._state(sys, k, prefix=_NO_ROADS)

    def road_factors(self, prefix: Prefix) -> State:
        """A road network's prefix's factors and reach (no operator: the homes it reaches need no
        solve)."""
        if prefix.stop < 1:
            raise ValueError(f"{prefix}: a road prefix has a stop from 1")
        return self._factors(self.sys, np.zeros(self.n_), prefix=prefix)

    def road_state(self, prefix: Prefix) -> State:
        """The state of a road network's prefix (the carving's), no building cleared: the atoms it
        carves open (the road rule: exactly its corridor), its operator assembled and factored."""
        if prefix.stop < 1:
            raise ValueError(f"{prefix}: a road prefix has a stop from 1")
        return self._state(self.sys, np.zeros(self.n_), prefix=prefix)

    def _rhs(self, st: State, inj: NDArray[np.float64]) -> NDArray[np.float64]:
        """The right-hand side per cell of the injection `inj` per home (a point in a ground cell
        or the band escapes at once)."""
        return np.asarray(st.sys.Pc @ inj) * st.reach.unknown

    def solve(self, st: State, inj: NDArray[np.float64]) -> NDArray[np.float64]:
        """The potential per cell (0 off the unknowns) under the injection `inj` per home."""
        t = time.time()
        assert st.factor is not None
        u = st.factor.solve(self._rhs(st, inj))
        self.timing.add("solves", time.time() - t)
        return u

    def home_u(self, st: State, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per home its escape time (nan with no demand) from the potential per cell."""
        uh = np.asarray(st.sys.PcT @ u)
        return np.where(self.nodem, np.nan, uh)

    def J(self, u_home: NDArray[np.float64], inj: NDArray[np.float64], power: float) -> float:
        ok = ~self.nodem
        return float((inj[ok] * u_home[ok] ** power).sum())

    def live_homes_of(self, st: State) -> NDArray[np.bool_]:
        """Per home: some triangle within ring_m of its outline reaches the street in the state
        (or the open band does): the flow model's own reach."""
        return np.asarray(self.fx.ring.live(st.reach.live[st.sys.cells.cell]))

    def complex_live(self, removed: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Per home: it reaches the street on the complex rebuilt without `removed`, grown as the
        form grows its footprints, read at the form's filtration width (gap_complex's test, free
        first's): the reference the flow model's reach is checked against, home for home."""
        cx = gc.build(self.geo, self.spec, ~removed, carve=None, grow=self.form.grow)
        return np.asarray(~gc.walled(gc.home_exits(gc.homes_of(self.geo, cx), gc.exit_widths(cx)),
                                     self.form.filtration_w))

    def grad(self, st: State, u: NDArray[np.float64],
             lam: NDArray[np.float64]) -> NDArray[np.float64]:
        """dJ / dk per building by the adjoint (lam: J's adjoint, per cell; u for P): dJ / dw =
        -du dlam per conductance, through each passage's harmonic mean and each triangle's terms
        (Width.terms.vjp: ties shared)."""
        t = time.time()
        fm, c = self.fm, st.sys.cells
        gt, gp = _grad_tri(c.pe, c.ea, c.eb, fm.pa, fm.pb, fm.T, st.kt, st.binds, c.pf, c.fc,
                           fm.fa, fm.Tf, u, lam, fm.nt)
        dk = self.terms.vjp(st.k, gt)
        if len(self.fill.e):
            gp_all = np.zeros(len(fm.pa))
            gp_all[c.pe] = gp
            sel = st.binds[self.fill.e]
            dk += np.bincount(self.fill.b[sel], weights=gp_all[self.fill.e[sel]],
                              minlength=self.carving.n)
        dk = self.carving.to_buildings(dk, self.n_, prefix=st.prefix)
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

    def road_lenses(self, prefix: Prefix) -> Lenses:
        """The lenses of a road network's prefix (road_state)."""
        return self.lenses_of(self.road_state(prefix))

    @staticmethod
    def k_of(removed: NDArray[np.bool_]) -> NDArray[np.float64]:
        return removed.astype(np.float64)

    # -- what the optimizers read

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
        """The greedy's tension on the fixed mesh: the gradient of J_power at the eps world
        (`tension_eps`: clear.EPS, or more translucent for the translucent search) of the current
        clearing, a building's gain the first-order change of moving its k from where it is to
        where the move takes it (adding: eps to 1; putting back: 1 to eps). One solve, two for
        power != 1."""
        eps = self.tension_eps
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
                           budget=budget, cost=self.cost_, pop=self.pop, held0=held,
                           cx0=self.fx.complex0(self.spec))
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

    def relaxation(self, power: float, *, q: float, eps: float) -> Relaxation:
        return Relaxation(self, power, q=q, eps=eps)


class Tension:
    """The greedy's tension on the fixed mesh (Model.tension): per building its first-order gain
    `g`, each home's weight in J_power, and the catchment's view of its state (`flow`)."""

    def __init__(self, *, model: Model, st: State, u: NDArray[np.float64],
                 home_u: NDArray[np.float64], g_: NDArray[np.float64]):
        self.m, self.st, self.u, self.home_u, self.g_ = model, st, u, home_u, g_

    @property
    def g(self) -> NDArray[np.float64]:
        return self.g_

    def weight(self, power: float) -> NDArray[np.float64]:
        return np.asarray(self.m.inj * np.nan_to_num(self.home_u) ** power)

    def flow(self, cand: list[int]) -> clear.Flow:
        """The tension's system, its nodes the cells (a cell off the unknowns an identity row at
        potential 0: no flow); a candidate's members the unknowns of the triangles its clearing
        re-weights (its terms: its zone under the shy distance)."""
        m, st = self.m, self.st
        A = st.A
        assert A is not None
        unk = st.reach.unknown
        load = (sp.diags(m.inj) @ st.sys.PcT @ sp.diags(unk.astype(np.float64))).tocsr()
        load.eliminate_zeros()
        terms = m.terms
        tri_of = terms.row // terms.G
        cell = st.sys.cells.cell
        rows, cols = [], []
        for col, j in enumerate(cand):
            nodes = np.unique(cell[tri_of[m.carving.parent[terms.b] == j]])
            nodes = nodes[unk[nodes]]
            rows.append(nodes)
            cols.append(np.full(len(nodes), col, dtype=np.int64))
        return clear.Flow(A=A, u=self.u, rowsum=np.asarray(A @ np.ones(A.shape[0])), load=load,
                          rows=np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64),
                          cols=np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64))


class Relaxation:
    """The model's eps world, as relax.simp, relax.Incumbent and search's worlds read it: J_power,
    k_b = eps + (1 - eps) proj(x_b)^q (relax.project; q and beta set by simp's continuation), its
    gradient in the projected clearing by the adjoint. Every solve is direct, refined to the
    floating-point floor (gap_factor.Refined): no tolerance (the raster's iterative one went with
    it)."""

    def __init__(self, model: Model, power: float, *, q: float, eps: float):
        self.m, self.power_, self.q, self.eps = model, power, q, eps
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
        """The same model's eps world at `eps` under the metric itself (q 1, no projection): the
        incumbent's and the polish's scorer."""
        return Relaxation(self.m, self.power_, q=1.0, eps=eps)


# ------------------------------------------------------------------ the run's metric

@dataclass(frozen=True, kw_only=True)
class ComplexMetric:
    """The metric on the complex: its identity the complex's Spec (s, ring, band),
    the FlowSpec (R, its grading, sigma, the demand's quadrature, the lattice offset), the width
    form, `lumped` (one node per building: the study's alternative) and the code's constants
    (CODE), every one in its name, which no raster conductance's can be (`check_disjoint`): its rows
    never read as the raster's. `search_eps`: the eps world the tension ranks in (the greedy's
    and the polish's ranking: clear.EPS, the metric's own, or a larger one, the translucent search,
    the raster's `@<search>` on the complex), its name's `@e<eps>` where it differs from clear.EPS;
    every score is under the metric. `exits` the region model's (gap_exits: barriers, the edge, leaks
    and a street boost; the per-block exits, the block's whole street ring, name nothing), its name's
    `+x...` where it has any. `factorization` is the run's device (gap_factor): it moves scores
    only at the floating-point level, so it is not in the name; the rows record it."""
    spec: gc.Spec
    fs: gf.FlowSpec
    form: gf.Width
    lumped: bool
    search_eps: float
    exits: gx.Exits
    factorization: gap_factor.Factorization

    def __post_init__(self) -> None:
        check_disjoint(self.name)

    @property
    def name(self) -> str:
        return (f"cx+{self.form.name}+s{self.spec.s:g}r{self.spec.ring_m:g}b{self.spec.band_m:g}"
                f"+{self.fs.name}" + ("+lumped" if self.lumped else "") + f"+v{CODE}"
                + self.exits.suffix
                + ("" if self.search_eps == clear.EPS else f"@e{self.search_eps:g}"))

    @property
    def device(self) -> str:
        return self.factorization.device

    def model(self, block, fx: Fixed, *, demand: common.Demand) -> Model:
        """The block's model on its fixed mesh `fx` (`clearing`'s, or one meshed afresh: `fixed`,
        of the same block), the block as the exits' barriers cut it."""
        block = self.exits.barriers.block(block)
        return Model(block, fx, spec=self.spec, fs=self.fs, form=self.form,
                     pop=common.POPULATIONS["area"].weights(np.asarray(block.buildings.outlines)),
                     demand=demand, factorization=self.factorization, lumped=self.lumped,
                     tension_eps=self.search_eps, exits=self.exits)

    def fixed(self, block, *, cache: Path, carving: Carving) -> Fixed:
        """The block's fixed mesh on `carving`'s atoms: Carving.none, the metric's own mesh (every
        clearing a state); a road network's (Carving.of), its prefixes states too. The block as
        the exits' barriers cut it (its polygon less them: the mesh's identity carries them)."""
        return Fixed.of(self.exits.barriers.block(block), self.spec, self.fs, self.form,
                        cache=cache, carving=carving)

    def clearing(self, block, *, demand: common.Demand) -> Model:
        return self.model(block, self.fixed(block, cache=CACHE,
                                            carving=Carving.none(gc.geometry(block))),
                          demand=demand)


# The retired raster's metric names, a fixed grammar (its rows stay on disk as records under them):
# an along-conductance (`uni`, Sightline `sl<beta>`, SoftSightline `ss<beta>k<kappa>[n<hill>r<r0>]`,
# turn-boosted with a trailing `t`), bare as the rows before the face aperture name it, or as a
# conductance `<along>[+<cells>]+apwfs` (the cells B_r `b<r>` or D_s `d<s>`); either with a search's
# `@<along>`. Private: it is no constant of the complex's code (`code_constants`).
_RASTER_ALONG = r"(?:uni|sl[^+@]+|ss[0-9.]+k[0-9.]+(?:n[0-9.]+r[0-9.]+)?t?)"
_RASTER_NAME = re.compile(rf"{_RASTER_ALONG}(?:(?:\+[bd][0-9]+(?:\.[0-9]+)?)?\+apwfs)?")


def check_disjoint(name: str) -> None:
    """Raises if `name`, less a search's `@...`, reads as one of the retired raster's metric names
    (_RASTER_NAME): a complex's rows would sit in a raster run's directories, records of another
    metric."""
    if _RASTER_NAME.fullmatch(name.split("@")[0]):
        raise ValueError(f"{name!r} names a raster conductance: the complex's rows would be read "
                         "as the raster's")


_NUM = r"([0-9]+(?:\.[0-9]+)?(?:e-?[0-9]+)?)"
_NAME = re.compile(rf"cx\+(close|shy|ramp){_NUM}(?:s{_NUM})?\+s{_NUM}r{_NUM}b{_NUM}"
                   rf"\+R{_NUM}(?:g([0-9]+)x{_NUM})?q([0-9]+)(?:k{_NUM})?"
                   rf"(?:o(-?{_NUM[1:-1]}),(-?{_NUM[1:-1]}))?(\+lumped)?"
                   rf"\+v([0-9a-f]+){gx.SUFFIX_RE}(?:@e{_NUM})?")


def complex_of(name: str, *, factorization: gap_factor.Factorization) -> ComplexMetric:
    """The ComplexMetric `name` names, on `factorization` (its name's inverse: refused unless it
    spells it back, its code digest this code's)."""
    m = _NAME.fullmatch(name)
    if m is None:
        raise ValueError(f"unknown complex metric {name!r}: cx+<form><w>+s<s>r<ring>b<band>+"
                         "R<R>[g<levels>x<grade>]q<quad>[k<sigma>][o<x>,<y>][+lumped]+v<code>"
                         "[+x[B<line>c<cross>][E0][L[o<open>][b<foot>]][S<boost>]v<exits code>]"
                         "[@e<search eps>]")
    (kind, w, ramp_s, s_, ring, band, R, levels, grade, quad, sigma, ox, oy,
     lumped, code, *exits, search) = m.groups()
    if code != CODE:
        raise ValueError(f"{name!r}: code {code}, but this code's constants digest to {CODE} "
                         "(its rows are another code's)")
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
                           fs=fs, form=form, lumped=lumped is not None,
                           search_eps=clear.EPS if search is None else float(search),
                           exits=gx.exits_of(tuple(exits)), factorization=factorization)
    if metric.name != name:
        raise ValueError(f"{name!r} spells the complex metric {metric.name!r}")
    return metric


# ------------------------------------------------------------------ the code's identity

# The modules whose module-level constants (public UPPER_CASE names) can change what the complex
# scores, and the constants it reads from elsewhere. Every such constant is in the code's digest
# (CODE), which every complex metric's name carries, so editing one renames the metric: no row of
# the old constants reads as the new code's. Exempt, each because it cannot change a result:
_EXEMPT = {
    "gap_clear.HERE": "a path: where this file lies",
    "gap_clear.CACHE": "a path: where the meshes are cached",
    "gap_clear.CONFORM_ROUNDS": "a cap: a mesh still not conformed after it raises, so every "
                                "mesh made is the same under any cap it fits in",
    "gap_clear.CODE": "the digest itself",
    "gap_flow.RING_CHUNK": "memory only: the same query in chunks",
    "gap_flow.TYPE_CHECKING": "typing's flag, imported: False when run",
    "gap_factor.CUDSS_LOG": "logging",
}
_EXTERNAL: dict[str, Callable[[], object]] = {
    "clear.EPS": lambda: clear.EPS,     # the tension's eps world (Model.tension)
}


def _identity_modules() -> tuple[Any, ...]:
    import sys
    return (sys.modules[__name__], gc, gf, gap_factor)


def code_constants() -> dict[str, str]:
    """Every constant of the complex's code (module comment above) by `<module>.<NAME>`, as its
    repr (digests.constants_of: a kind it does not know refused, decided here or in _EXEMPT). The
    exits' code (gap_exits: the region model's leaks, barriers and boost) is not in this walk: its
    own digest names the metrics that use it (gap_exits.EXITS_CODE), so the per-block metric's name
    does not move with it."""
    return digests.constants_of(_identity_modules(), exempt=_EXEMPT, external=_EXTERNAL)


def code_digest() -> str:
    """The code's constants (code_constants) digested: 8 hex digits."""
    return digests.digest_of(code_constants())


CODE = code_digest()
