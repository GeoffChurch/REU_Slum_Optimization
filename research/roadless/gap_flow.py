"""The gap complex, stage 2: the uniform law on the complex (NOTES, "The gap complex: passability,
enclosure and flow without the raster").

THE MESH (`mesh`). The complex of gap_complex.build (outlines densified to s, constrained
Delaunay, Lawson-flipped; the footprints grown by the width form's `grow`), refined: Steiner points
on a triangular lattice of spacing R (FlowSpec.R, the largest pocket size) over the open ground,
those at least max(R / 2, Width.steiner_clear) from every outline (footprints and the band's
line), inserted one by one with Lawson's flips (`_insert`), the outlines never flipped. Under
Closing the clearance keeps the passable width honest: a Steiner point is no obstacle, so a
passage with one at an end has aperture inf, and a gap of width a <= w keeps its crossing edge
between the two outlines only while no Steiner point falls in that edge's diametral disk, which a
point clearing both ends by more than a / sqrt(2) cannot (STEINER_W w > w / sqrt(2)).

THE PASSABLE WIDTH (`Width`, the brief's addendum): Closing (a passage narrower than w closed),
Shy (every footprint grown by w / 2: a lane of width W conducts W - w) and Ramp (the same zero
zone, then phi = (d - w / 2) / s up to 1): each a factor on every passage and street face of the
mesh at a 0/1 state.

THE LAW (`cells`): the two-point finite-volume scheme, each triangle a pocket centred at its
circumcentre (`circumcentres`): a passage e between pockets a and b conducts sigma |e| / d_e, d_e
the distance between the centres along e's normal (both centres lie on e's perpendicular
bisector, so the scheme is consistent for isotropic diffusion). On a Delaunay mesh d_e >= 0;
d_e = 0 where the two triangles share their circumcircle (a rectangle split by a diagonal: common
along densified parallel walls): such pockets are one cell (merged, union-find, d_e <=
MERGE_REL |e|), centred at their common circumcentre. An obtuse triangle's centre lies outside it
(beyond its longest edge); across a passage that is harmless (the order of the centres along each
normal is what consistency needs, and Delaunay keeps it); beyond an outline (the longest edge a
wall) the centre lies in a footprint and only its potential's location is off. Walls conduct
nothing. The street: an exit face (an edge on the band's line, beyond which u = 0) conducts
sigma |e| / d_f, d_f the centre's distance to the face; where the centre lies beyond the face (an
obtuse exit triangle) the pocket is ground itself (counted: `Cells.ground_obtuse`). A passage
beside a flat triangle (a hairline crack's, 1e-11 m) cannot be flipped and has no centre: dropped
(`Cells.negative`).

DEMAND (`Demand`), lifted.Placement's rule on the complex: each triangle's QUAD sub-points (the
centroids of its 4^k midpoint subdivision) and the open band's (its own constrained Delaunay
triangles, u = 0) carry area / QUAD each; a sub-point within ring_m of a home's outline whose
nearest building it is belongs to that home's ring; a home injects its population over its ring's
sub-points by area, those that reach the street at width w (`allowed`). A home with no allowed
ring point but an allowed point within ring_m of its outline is SHADOWED and injects over those;
with none it is walled in (gap_complex's test, counted apart).

The run's identity: gap_complex.Spec (s, ring_m, band_m), FlowSpec (R, sigma, quad, offset) and
the Width (its w, and the ramp's s), required, keyword-only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scipy.sparse as sp
import shapely
from numba import njit
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components

import gap_complex as gc

MERGE_REL = 1e-4      # pockets whose centres lie within this share of their passage's length along
                      # its normal are merged (d_e 0: a shared circumcircle; below 1e-4 the weight
                      # |e| / d_e passes 1e4 and the system's conditioning suffers for nothing)
T_MIN = 1e-6          # a passage whose weight kappa |e| / d_e is below this (a hairline crack's:
                      # an aperture 1e-7 m across cells s apart; a ramp's edge) is dropped: the
                      # cells it alone joins left the system near-singular (SuperLU's residuals
                      # 0.5%)
FLAT_REL = 1e-4       # a triangle of area under this x L^2 (L a passage's length) is flat: a
                      # hairline crack's, 25 um high on a 12.5 cm base (a 5 cm crack's triangles
                      # are ~0.2; 19570's full mesh held one at 6.6e-6, its centre 3e8 L away)
STEINER_W = 0.75      # Closing's Steiner clearance, x w (> 1 / sqrt(2))
INSERT_PASSES = 8     # mesh: passes locating the Steiner points left by blocked walks
SIGMA_ISO = math.pi / 2   # the raster's isotropic limit (lifted's sum_k m_k e_k e_k^T): a
                          # FlowSpec's name omits sigma at this value (as the offset at OFFSET)
OFFSET = (0.3141592653, 0.2718281828)   # the Steiner lattice's offset, x R (irrational: no row
                                        # on an outline's)


@dataclass(frozen=True, kw_only=True)
class FlowSpec:
    """R the lattice spacing of the Steiner points (the largest pocket size near the outlines,
    m; with `levels` > 0 it doubles with the clearance, `steiner_points`); sigma the
    conductivity (pi / 2: lifted's operator's isotropic limit, sum_k m_k e_k e_k^T, for the
    levels; the lenses do not see it); quad the demand's sub-points per triangle (4^k); offset the
    lattice's (OFFSET; the checks shift it)."""
    R: float
    levels: int                      # coarser lattice levels (0: R everywhere)
    grade: float                     # a level's clearance per unit of its spacing
    sigma: float
    quad: int
    offset: tuple[float, float]      # the lattice's offset from the open ground's box, x R

    @property
    def name(self) -> str:
        o = "" if self.offset == OFFSET else f"o{self.offset[0]:g},{self.offset[1]:g}"
        g = f"g{self.levels}x{self.grade:g}" if self.levels > 0 else ""
        k = "" if self.sigma == SIGMA_ISO else f"k{self.sigma:g}"
        return f"R{self.R:g}{g}q{self.quad}{k}{o}"


# ------------------------------------------------------------------ refinement

@njit(cache=True)
def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


@njit(cache=True)
def _locate(xy, tri, nbr, t, p, maxsteps):
    """The triangle holding p, walking from t (-1 if the walk leaves the mesh or stalls)."""
    off = 0
    for _ in range(maxsteps):
        moved = False
        for i in range(3):
            j = (i + off) % 3
            a = tri[t, (j + 1) % 3]
            b = tri[t, (j + 2) % 3]
            if _orient(xy[a], xy[b], p) < 0.0:
                n = nbr[t, j]
                if n < 0:
                    return -1
                t = n
                moved = True
                off = (off + 1) % 3
                break
        if not moved:
            return t
    return -1


@njit(cache=True)
def _insert(xy, tri, nbr, con, face, nt, nv, pts, start, tol):
    """Each of pts into the triangulation (xy, tri, nbr, con, in place, with room for them):
    located by a walk from start[k] (across any edge), split in three (the children keep the
    triangle's `face`), legalized by gap_complex.legalize (never across an edge flagged in con,
    so a flip stays within a face). Returns (nt, nv, skipped, done: per point, inserted or dropped
    (on an edge), not left for another pass)."""
    skipped = 0
    done = np.zeros(pts.shape[0], dtype=np.bool_)
    for k in range(pts.shape[0]):
        p = pts[k]
        t = _locate(xy, tri, nbr, start[k], p, 100000)
        if t < 0:
            skipped += 1
            continue
        a, b, c = tri[t, 0], tri[t, 1], tri[t, 2]
        # strictly inside, away from its edges (a point on an edge would make a flat triangle)
        if (_orient(xy[a], xy[b], p) <= 0.0 or _orient(xy[b], xy[c], p) <= 0.0
                or _orient(xy[c], xy[a], p) <= 0.0):
            skipped += 1
            done[k] = True                      # dropped for good
            continue
        done[k] = True
        v = nv
        xy[v, 0] = p[0]
        xy[v, 1] = p[1]
        nv += 1
        na, nb, nc = nbr[t, 0], nbr[t, 1], nbr[t, 2]
        ca, cb, cc = con[t, 0], con[t, 1], con[t, 2]
        t1 = nt
        t2 = nt + 1
        nt += 2
        tri[t, 0], tri[t, 1], tri[t, 2] = a, b, v
        nbr[t, 0], nbr[t, 1], nbr[t, 2] = t1, t2, nc
        con[t, 0], con[t, 1], con[t, 2] = False, False, cc
        tri[t1, 0], tri[t1, 1], tri[t1, 2] = b, c, v
        nbr[t1, 0], nbr[t1, 1], nbr[t1, 2] = t2, t, na
        con[t1, 0], con[t1, 1], con[t1, 2] = False, False, ca
        tri[t2, 0], tri[t2, 1], tri[t2, 2] = c, a, v
        nbr[t2, 0], nbr[t2, 1], nbr[t2, 2] = t, t1, nb
        con[t2, 0], con[t2, 1], con[t2, 2] = False, False, cb
        face[t1] = face[t]
        face[t2] = face[t]
        if na >= 0:
            for q in range(3):
                if nbr[na, q] == t:
                    nbr[na, q] = t1
        if nb >= 0:
            for q in range(3):
                if nbr[nb, q] == t:
                    nbr[nb, q] = t2
        stack = [(t, 2), (t1, 2), (t2, 2)]
        gc.legalize(xy[:nv], tri[:nt], nbr[:nt], con[:nt], stack, tol)
    return nt, nv, skipped, done


def _lattice(inner, h: float, offset: tuple[float, float]) -> NDArray[np.float64]:
    """The triangular lattice of spacing h over `inner`'s box, inside it (its offset irrational, so
    no point falls on an outline's vertex row; perturbed by 1e-6 h, seeded: a lattice's rows are
    exactly collinear, and a point would fall on the edge between two of its row inserted before
    it)."""
    x0, y0, x1, y1 = inner.bounds
    dy = h * math.sqrt(3.0) / 2.0
    ys = np.arange(y0 + offset[1] * dy, y1, dy)
    pts = []
    for k, y in enumerate(ys):
        xs = np.arange(x0 + (offset[0] + 0.5 * (k % 2)) * h, x1, h)
        pts.append(np.column_stack([xs, np.full(len(xs), y)]))
    P = np.concatenate(pts) if pts else np.zeros((0, 2))
    P = P + np.random.default_rng(len(P)).uniform(-1e-6, 1e-6, P.shape) * h
    shapely.prepare(inner)
    return P[shapely.contains_xy(inner, P[:, 0], P[:, 1])]


def steiner_points(inner, fs: FlowSpec, clear: float) -> NDArray[np.float64]:
    """The Steiner points: on lattice level l (spacing R 2^l, l = 0 .. fs.levels) the points whose
    clearance d from `inner`'s boundary is at least max(clear, R 2^l / 2), and for l > 0 at least
    fs.grade R 2^l, and below fs.grade R 2^(l+1) unless l is the last: R near the outlines,
    doubling with the clearance (levels 0: R everywhere)."""
    if inner.is_empty:
        return np.zeros((0, 2))
    bnd = shapely.boundary(inner)
    shapely.prepare(bnd)
    out = []
    for lv in range(fs.levels + 1):
        h = fs.R * 2 ** lv
        P = _lattice(inner, h, fs.offset)
        pts = shapely.points(P[:, 0], P[:, 1])
        lo = max(clear, h / 2.0, fs.grade * h if lv > 0 else 0.0)
        keep = ~shapely.dwithin(bnd, pts, lo)
        if lv < fs.levels:
            keep &= shapely.dwithin(bnd, pts, fs.grade * 2.0 * h)
        out.append(P[keep])
    return np.concatenate(out)


# ------------------------------------------------------------------ the passable width's forms

class Width(Protocol):
    """The passable width's form (the brief's addendum): resolved once where a run is configured,
    part of its name. `grow` the footprints' growth before meshing, `steiner_clear` the Steiner
    points' clearance from every outline, `kappa` the factor in [0, 1] of each passage and street
    face at a 0/1 state (the standing footprints less `carve`), from which the form's width law
    follows: a lane of width W conducts G(W) = 2 int_0^{W/2} phi."""

    @property
    def name(self) -> str: ...

    @property
    def grow(self) -> float: ...

    @property
    def steiner_clear(self) -> float: ...

    def kappa(self, geo: gc.Geometry, standing: NDArray[np.bool_], carve, xy: NDArray[np.float64],
              passages: tuple, faces: tuple) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ...


@dataclass(frozen=True, kw_only=True)
class Closing:
    """Gaps narrower than w filled into the footprints beside them: a passage conducts wholly or,
    with aperture < w, not at all (a lane at least w wide keeps its full width: G(W) = W for W >=
    w). The Steiner points kept STEINER_W w clear (module docstring); w 0: every passage open."""
    w: float

    @property
    def name(self) -> str:
        return f"close{self.w:g}"

    @property
    def grow(self) -> float:
        return 0.0

    @property
    def steiner_clear(self) -> float:
        return STEINER_W * self.w

    def kappa(self, geo, standing, carve, xy, passages, faces):
        _pu, _pv, ap = passages
        return (ap >= self.w).astype(float), np.ones(len(faces[0]))


@dataclass(frozen=True, kw_only=True)
class Shy:
    """The shy distance: every footprint grown by w / 2 (the step kernel phi = [d >= w / 2]), so
    the open ground itself shrinks and a lane of width W conducts W - w (0 below w): G(W) =
    max(0, W - w). Every passage of the grown complex conducts."""
    w: float

    @property
    def name(self) -> str:
        return f"shy{self.w:g}"

    @property
    def grow(self) -> float:
        return self.w / 2.0

    @property
    def steiner_clear(self) -> float:
        return 0.0

    def kappa(self, geo, standing, carve, xy, passages, faces):
        return np.ones(len(passages[0])), np.ones(len(faces[0]))


GAUSS_N = 16          # Ramp: Gauss-Legendre points along a passage (phi has kinks where the
                      # ramp starts and ends: 4 points left a lane wholly on the ramp 4% off)
GAUSS = np.polynomial.legendre.leggauss(GAUSS_N)


@dataclass(frozen=True, kw_only=True)
class Ramp:
    """The shy distance's zero zone with a ramp: phi(d) = clip((d - w / 2) / s, 0, 1), d the
    distance to the nearest standing footprint; the footprints grown by w / 2 (phi's zero zone)
    and each passage and street face conducting by phi's mean along it (GAUSS_N-point Gauss):
    G(W) = W - w - s for W >= w + 2 s."""
    w: float
    s: float

    @property
    def name(self) -> str:
        return f"ramp{self.w:g}s{self.s:g}"

    @property
    def grow(self) -> float:
        return self.w / 2.0

    @property
    def steiner_clear(self) -> float:
        return 0.0

    def kappa(self, geo, standing, carve, xy, passages, faces):
        _idx, polys = gc.obstacles(geo, standing, carve=carve, grow=0.0)
        tree = shapely.STRtree(polys)
        out = []
        for u, v in (passages[:2], faces):
            if not len(u):
                out.append(np.zeros(0))
                continue
            t = 0.5 * (GAUSS[0] + 1.0)
            P = xy[u][:, None, :] + t[None, :, None] * (xy[v] - xy[u])[:, None, :]
            P = P.reshape(-1, 2)
            if len(polys):
                _i, d = tree.query_nearest(shapely.points(P[:, 0], P[:, 1]),
                                           return_distance=True, all_matches=False)
            else:
                d = np.full(len(P), np.inf)
            phi = np.clip((d - self.w / 2.0) / self.s, 0.0, 1.0).reshape(len(u), -1)
            out.append(phi @ (0.5 * GAUSS[1]))
        return out[0], out[1]


@dataclass(frozen=True, eq=False)
class Mesh:
    """The refined complex: `cx` the complex itself (unrefined: its passages' apertures, the
    passable width), xy / tri the refined triangulation (vertices: cx's, then the Steiner points),
    `ap` per refined passage its aperture (its ends' outlines; inf with a Steiner or street end),
    fa / fu / fv its street faces; `kp`, `kf` the width form's factor (Width.kappa) per passage
    and per street face."""
    cx: gc.Complex
    fs: FlowSpec
    width: Width
    xy: NDArray[np.float64]
    tri: NDArray[np.int64]
    nv0: int
    pa: NDArray[np.int64]
    pb: NDArray[np.int64]
    pu: NDArray[np.int64]
    pv: NDArray[np.int64]
    ap: NDArray[np.float64]
    fa: NDArray[np.int64]            # street faces: the triangle, and the face's two vertices
    fu: NDArray[np.int64]
    fv: NDArray[np.int64]
    kp: NDArray[np.float64]
    kf: NDArray[np.float64]
    skipped: int

    @property
    def nt(self) -> int:
        return len(self.tri)

    @property
    def area(self) -> NDArray[np.float64]:
        p0, p1, p2 = (self.xy[self.tri[:, i]] for i in range(3))
        return 0.5 * np.abs(_cross(p0, p1, p2))


def _cross(p0, p1, p2):
    return (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (
        p2[:, 0] - p0[:, 0])


def insert_points(xy, tri, nbr, con, face, nt: int, nv: int, pts) -> tuple[int, int, int]:
    """`pts` into the triangulation (arrays with room for them, in place, _insert): each
    located on the current mesh (STRtree), then walked to from there as earlier insertions change
    it; a walk a wall blocks (a concave pocket) is retried in the next pass. Returns (nt, nv, the
    points left)."""
    for _pass in range(INSERT_PASSES):
        if not len(pts):
            break
        geom = shapely.polygons(xy[tri[:nt]])
        qi, qt = shapely.STRtree(geom).query(shapely.points(pts[:, 0], pts[:, 1]),
                                             predicate="intersects")
        start = np.full(len(pts), -1, dtype=np.int64)
        start[qi[::-1]] = qt[::-1]
        pts, start = pts[start >= 0], start[start >= 0]
        order = np.argsort(start, kind="stable")
        pts, start = pts[order], start[order]
        before = nv
        nt, nv, _skipped, done = _insert(xy, tri, nbr, con, face, nt, nv, pts, start,
                                         gc.DELAUNAY_TOL)
        pts = pts[~done]
        if nv == before:
            break
    return nt, nv, len(pts)


def mesh(geo: gc.Geometry, spec: gc.Spec, fs: FlowSpec, width: Width,
         standing: NDArray[np.bool_], *, carve) -> Mesh:
    """The complex of `geo` with the buildings `standing` (less `carve`, a corridor: Carve's rule;
    None: nothing), grown by width.grow, refined (module docstring), and the width form's
    factors."""
    cx = gc.build(geo, spec, standing, carve=carve, grow=width.grow)
    inner, _band = gc._open_ground(geo, standing, spec.band_m, carve=carve, grow=width.grow)
    pts = steiner_points(inner, fs, width.steiner_clear)
    nt, nv = cx.nt, len(cx.xy)
    nv0 = nv
    xy = np.zeros((nv + len(pts), 2))
    xy[:nv] = cx.xy
    tri = np.zeros((nt + 2 * len(pts), 3), dtype=np.int64)
    tri[:nt] = cx.tri
    nbr = -np.ones_like(tri)
    nbr[:nt] = gc._neighbours(cx.tri, nv)
    con = nbr < 0
    nt, nv, skipped = insert_points(xy, tri, nbr, con, np.zeros(len(tri), dtype=np.int64), nt,
                                    nv, pts)
    xy, tri = xy[:nv], tri[:nt]
    # passages and street faces of the refined mesh
    e = np.stack([tri[:, [1, 2]], tri[:, [2, 0]], tri[:, [0, 1]]], axis=1).reshape(-1, 2)
    es = np.sort(e, axis=1)
    key = es[:, 0] * nv + es[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    same = ks[1:] == ks[:-1]
    i0, i1 = order[:-1][same], order[1:][same]
    pa, pb = i0 // 3, i1 // 3
    pu, pv = es[i0, 0], es[i0, 1]
    obstacle = np.zeros(nv, dtype=bool)
    obstacle[:nv0] = ~cx.street_v
    L = np.hypot(*(xy[pu] - xy[pv]).T)
    ap = np.where(obstacle[pu] & obstacle[pv], L, np.inf)
    single = np.ones(len(key), dtype=bool)
    single[i0] = False
    single[i1] = False
    bi = np.flatnonzero(single)
    # a boundary edge is a street face where cx had one: its ends are cx's vertices (the outline
    # is never split), so look the pair up among cx's street faces
    sf = cx.street_faces
    bkey = es[bi, 0] * nv + es[bi, 1]
    street = np.isin(bkey, sf[:, 0] * nv + sf[:, 1])
    fb = bi[street]
    fu, fv = es[fb, 0], es[fb, 1]
    kp, kf = width.kappa(geo, standing, carve, xy, (pu, pv, ap), (fu, fv))
    return Mesh(cx=cx, fs=fs, width=width, xy=xy, tri=tri, nv0=nv0, pa=pa, pb=pb, pu=pu, pv=pv,
                ap=ap, fa=fb // 3, fu=fu, fv=fv, kp=kp, kf=kf, skipped=int(skipped))


# ------------------------------------------------------------------ the law

def circumcentres(xy: NDArray[np.float64], tri: NDArray[np.int64]) -> NDArray[np.float64]:
    a, b, c = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    bx, by = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    cx_, cy = c[:, 0] - a[:, 0], c[:, 1] - a[:, 1]
    d = 2.0 * (bx * cy - by * cx_)
    b2, c2 = bx * bx + by * by, cx_ * cx_ + cy * cy
    with np.errstate(divide="ignore", invalid="ignore"):   # a flat triangle: inf / nan, and its
        ux = (cy * b2 - by * c2) / d                       # passages are dropped (no weight)
        uy = (bx * c2 - cx_ * b2) / d
    return np.column_stack([a[:, 0] + ux, a[:, 1] + uy])


def _third(tri, t, u, v):
    T = tri[t]
    return np.where((T[:, 0] != u) & (T[:, 0] != v), T[:, 0],
                    np.where((T[:, 1] != u) & (T[:, 1] != v), T[:, 1], T[:, 2]))


@dataclass(frozen=True, eq=False)
class Cells:
    """The mesh's cells at width w: `cell` per triangle its cell (-1 walled in at w), `n`;
    `ground` per cell (u = 0); the conductances: `ea`, `eb`, `T` between cells, `g` per cell to
    the street; `area` per cell; diagnostics."""
    cell: NDArray[np.int64]
    n: int
    ground: NDArray[np.bool_]
    ea: NDArray[np.int64]
    eb: NDArray[np.int64]
    T: NDArray[np.float64]
    g: NDArray[np.float64]
    merged: int
    ground_obtuse: int
    negative: int          # passages dropped: their centres in the wrong order, beside a flat
                           # triangle


def cells(m: Mesh) -> Cells:
    """The two-point scheme's cells and conductances (module docstring), each passage and street
    face times its width form's factor (kp, kf: a passage with 0 closed); the triangles not
    joined to the street through open passages (walled in at the form's width) dropped."""
    xy, tri = m.xy, m.tri
    nt = m.nt
    cc = circumcentres(xy, tri)
    open_ = m.kp > 0
    pa, pb, pu, pv = m.pa[open_], m.pb[open_], m.pu[open_], m.pv[open_]
    kp = m.kp[open_]
    L = np.hypot(*(xy[pu] - xy[pv]).T)
    tvec = (xy[pv] - xy[pu]) / L[:, None]
    n = np.column_stack([-tvec[:, 1], tvec[:, 0]])
    p = _third(tri, pa, pu, pv)
    flip = ((xy[p] - xy[pu]) * n).sum(1) > 0          # n from a towards b: away from a's third
    n[flip] *= -1.0
    d = ((cc[pb] - cc[pa]) * n).sum(1)
    neg = d < -MERGE_REL * L
    # a flat triangle (three outline vertices in a row, or the two walls of a hairline crack
    # 1e-11 m apart) cannot be flipped (its quadrilateral is not convex) and has no centre: its
    # passages conduct nothing (dropped, counted)
    p0, p1, p2 = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    area = 0.5 * np.abs(_cross(p0, p1, p2))
    flat = np.minimum(area[pa], area[pb]) <= FLAT_REL * L * L
    if (neg & ~flat).any():
        raise ValueError(f"{int((neg & ~flat).sum())} passages with their centres in the wrong "
                         "order between triangles that are not flat: the mesh is not Delaunay")
    negative = int((neg & flat).sum())
    drop = neg & flat
    pa, pb, L, d, kp = pa[~drop], pb[~drop], L[~drop], d[~drop], kp[~drop]
    merge = np.abs(d) <= MERGE_REL * L
    # cells: triangles merged across zero-distance passages
    _k, lab = connected_components(sp.csr_matrix(
        (np.ones(int(merge.sum())), (pa[merge], pb[merge])), shape=(nt, nt)), directed=False)
    # the street: faces
    fl = np.hypot(*(xy[m.fv] - xy[m.fu]).T)
    ftv = (xy[m.fv] - xy[m.fu]) / fl[:, None]
    fn = np.column_stack([-ftv[:, 1], ftv[:, 0]])
    fp = _third(tri, m.fa, m.fu, m.fv)
    inward = ((xy[fp] - xy[m.fu]) * fn).sum(1) > 0
    fn[inward] *= -1.0                                  # outward
    df = ((xy[m.fu] - cc[m.fa]) * fn).sum(1)
    # components: open passages and street faces
    keep = ~merge & (kp * L / np.maximum(d, 1e-300) >= T_MIN)
    ea, eb, T = lab[pa[keep]], lab[pb[keep]], kp[keep] * L[keep] / d[keep]
    ncell = int(lab.max()) + 1
    fo = m.kf > 0
    obt = fo & (df <= MERGE_REL * fl)
    ground = np.zeros(ncell, dtype=bool)
    ground[lab[m.fa[obt]]] = True
    ok = fo & ~obt
    g = np.bincount(lab[m.fa[ok]], weights=m.kf[ok] * fl[ok] / df[ok], minlength=ncell)
    # reach: cells joined (through conductances) to a ground cell or a face
    street = ncell
    touch = np.flatnonzero(ground | (g > 0))
    r = np.concatenate([ea, touch])
    c = np.concatenate([eb, np.full(len(touch), street)])
    _k2, comp = connected_components(sp.csr_matrix((np.ones(len(r)), (r, c)),
                                                   shape=(ncell + 1, ncell + 1)), directed=False)
    live = comp[:ncell] == comp[street]
    newid = -np.ones(ncell, dtype=np.int64)
    newid[live] = np.arange(int(live.sum()))
    k = live[ea] & live[eb] & (ea != eb)
    sig = m.fs.sigma
    return Cells(cell=newid[lab], n=int(live.sum()), ground=ground[live],
                 ea=newid[ea[k]], eb=newid[eb[k]], T=sig * T[k], g=sig * g[live],
                 merged=int(merge.sum()), ground_obtuse=int(obt.sum()), negative=negative)


def laplacian(c: Cells, kappa: NDArray[np.float64] | None = None) -> sp.csr_matrix:
    """The cells' operator on the unknowns (the non-ground cells), as a CSR matrix; `kappa` per
    conductance (ea, eb) a factor (None: 1)."""
    T = c.T if kappa is None else c.T * kappa
    unk = -np.ones(c.n, dtype=np.int64)
    free = ~c.ground
    unk[free] = np.arange(int(free.sum()))
    N = int(free.sum())
    ua, ub = unk[c.ea], unk[c.eb]
    both = (ua >= 0) & (ub >= 0)
    diag = np.bincount(ua[ua >= 0], weights=T[ua >= 0], minlength=N) + np.bincount(
        ub[ub >= 0], weights=T[ub >= 0], minlength=N) + c.g[free]
    rows = np.concatenate([ua[both], ub[both], np.arange(N)])
    cols = np.concatenate([ub[both], ua[both], np.arange(N)])
    vals = np.concatenate([-T[both], -T[both], diag])
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


REFINE_RTOL = 1e-12   # factor's solves: iterative refinement to this residual (relative), and
REFINE_ACCEPT = 1e-5  # ... at worst this after REFINE_STEPS (the weights span ten decades; P's
                      # error is of this order, far below any lens's resolution)
REFINE_STEPS = 8


class _Refined:
    """A's SuperLU factorization with iterative refinement: a cell joined to the rest only through
    a hairline passage has a diagonal 1e-9 of the others', and SuperLU's partial pivoting then
    left residuals of 0.7% (relative) that the clearing model's finite differences saw; each
    solve is refined until its residual is below REFINE_RTOL of the right-hand side."""

    def __init__(self, A: sp.csr_matrix):
        from scipy.sparse.linalg import splu
        self.A = A.tocsr()
        self.lu = splu(A.tocsc(), permc_spec="COLAMD")

    def solve(self, b: NDArray[np.float64]) -> NDArray[np.float64]:
        x = self.lu.solve(b)
        nb = float(np.abs(b).max())
        if nb == 0.0:
            return x
        for _ in range(REFINE_STEPS):
            r = b - self.A @ x
            if float(np.abs(r).max()) <= REFINE_RTOL * nb:
                return x
            x = x + self.lu.solve(r)
        r = b - self.A @ x
        if float(np.abs(r).max()) > REFINE_ACCEPT * nb:
            raise RuntimeError(f"refinement stalled at {float(np.abs(r).max()) / nb:.1e}")
        return x


def factor(A: sp.csr_matrix) -> _Refined:
    """A's LU factorization (SuperLU, COLAMD: on these planar meshes it fills little, 1.5 s for
    690k unknowns against pyamg's 16 -- 69 s per solve to 1e-8, the conductances spanning six
    decades), its `solve` refined (_Refined)."""
    return _Refined(A)


# ------------------------------------------------------------------ demand

def quad_bary(quad: int) -> NDArray[np.float64]:
    """Barycentric coordinates of the centroids of a triangle's 4^k midpoint subdivision (quad
    = 4^k sub-points)."""
    tris = [np.eye(3)]
    while len(tris) < quad:
        nxt = []
        for T in tris:
            a, b, c = T
            ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
            nxt += [np.array([a, ab, ca]), np.array([ab, b, bc]), np.array([ca, bc, c]),
                    np.array([ab, bc, ca])]
        tris = nxt
    if len(tris) != quad:
        raise ValueError(f"quad {quad} is not a power of 4")
    return np.array([T.mean(axis=0) for T in tris])


@dataclass(frozen=True, eq=False)
class PointDemand:
    """lifted.Placement on the complex (module docstring), fixed at baseline: sub-points `xy`
    (the open ground's triangles' and the open band's, `band` True for the latter: u = 0), `S`
    (points x homes) each home's injection share per point (a live home's column sums to 1),
    `walled` the homes walled in at baseline at the run's width, `shadowed`."""
    xy: NDArray[np.float64]
    band: NDArray[np.bool_]
    S: sp.csr_matrix
    walled: NDArray[np.bool_]
    shadowed: NDArray[np.bool_]


def _points_of(xy, tri, quad: int) -> tuple[NDArray[np.float64], NDArray[np.float64],
                                            NDArray[np.int64]]:
    """(sub-points, their weight area / quad, their triangle)."""
    B = quad_bary(quad)
    P = np.einsum("qk,tkd->tqd", B, xy[tri]).reshape(-1, 2)
    p0, p1, p2 = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    a = 0.5 * np.abs(_cross(p0, p1, p2))
    return P, np.repeat(a / quad, quad), np.repeat(np.arange(len(tri)), quad)


def point_demand(geo: gc.Geometry, m: Mesh, c: Cells) -> PointDemand:
    """The baseline's demand on mesh m (its cells at the run's width, c)."""
    spec, fs = m.cx.spec, m.fs
    P, wt, t = _points_of(m.xy, m.tri, fs.quad)
    ok = c.cell[t] >= 0                                    # reaches the street at w
    P, wt = P[ok], wt[ok]
    # the open band's points (u = 0)
    btris = gc._triangulate(m.cx.band_open, spec.s)
    if len(btris):
        bc = shapely.get_coordinates(btris).reshape(-1, 4, 2)[:, :3]
        bxy, binv = np.unique(bc.reshape(-1, 2), axis=0, return_inverse=True)
        BP, bwt, _bt = _points_of(bxy, binv.reshape(-1, 3), fs.quad)
    else:
        BP, bwt = np.zeros((0, 2)), np.zeros(0)
    xy = np.concatenate([P, BP])
    w = np.concatenate([wt, bwt])
    band = np.concatenate([np.zeros(len(P), dtype=bool), np.ones(len(BP), dtype=bool)])
    polys = geo.polys
    n = len(polys)
    pts = shapely.points(xy[:, 0], xy[:, 1])
    tree = shapely.STRtree(polys)
    (pi, hi), dist = tree.query_nearest(pts, return_distance=True, all_matches=False)
    ring = dist <= spec.ring_m
    pi, hi = pi[ring], hi[ring]
    has = np.bincount(hi, minlength=n) > 0
    rows, cols = [pi], [hi]
    # shadowed: no ring point of its own, but allowed points within ring_m of its outline
    lack = np.flatnonzero(~has)
    shadowed = np.zeros(n, dtype=bool)
    if len(lack):
        qi, qh = shapely.STRtree(pts).query(polys[lack], predicate="dwithin",
                                            distance=spec.ring_m)
        shadowed[lack[np.unique(qi)]] = True
        rows.append(qh)
        cols.append(lack[qi])
    r, cl = np.concatenate(rows), np.concatenate(cols)
    tot = np.bincount(cl, weights=w[r], minlength=n)
    S = sp.csr_matrix((w[r] / tot[cl], (r, cl)), shape=(len(xy), n))
    walled = tot == 0
    return PointDemand(xy=xy, band=band, S=S, walled=walled, shadowed=shadowed)


def locate(m: Mesh, xy: NDArray[np.float64]) -> NDArray[np.int64]:
    """Per point, the mesh's triangle holding it (-1 none: the band, or outside)."""
    geom = shapely.polygons(m.xy[m.tri])
    qi, qt = shapely.STRtree(geom).query(shapely.points(xy[:, 0], xy[:, 1]),
                                         predicate="intersects")
    out = -np.ones(len(xy), dtype=np.int64)
    out[qi[::-1]] = qt[::-1]
    return out


class Scores:
    """A field's escape times per home (nan walled in) and P, from the mesh's cells, the demand
    and an injection per home."""

    def __init__(self, m: Mesh, c: Cells, dem: PointDemand):
        self.m, self.c, self.dem = m, c, dem
        t = locate(m, dem.xy)
        cell = np.where(t >= 0, c.cell[np.maximum(t, 0)], -1)
        cell[dem.band] = -1
        # a demand point the field does not reach (a clearing only opens space, so none should:
        # counted, its injection dropped)
        self.dropped = int(((cell < 0) & ~dem.band
                            & (np.asarray(dem.S.sum(axis=1)).ravel() > 0)).sum())
        self.pcell = cell
        free = ~c.ground
        unk = -np.ones(c.n, dtype=np.int64)
        unk[free] = np.arange(int(free.sum()))
        self.unk = unk
        self.A = laplacian(c)
        self.lu = factor(self.A)

    def u_home(self, inj: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """(P, per home its escape time: the injection-weighted mean potential over its points;
        nan walled in) of the injection `inj` per home."""
        S = self.dem.S
        f_pt = S @ inj
        on = self.pcell >= 0
        uk = self.unk[np.maximum(self.pcell, 0)]
        live = on & (uk >= 0)
        b = np.bincount(uk[live], weights=f_pt[live], minlength=self.A.shape[0])
        u = self.lu.solve(b)
        u_pt = np.where(live, u[np.maximum(uk, 0)], 0.0)
        uh = np.asarray(S.T @ u_pt).ravel()
        uh[self.dem.walled] = np.nan
        return float(b @ u), uh


def live_homes(geo: gc.Geometry, m: Mesh, c: Cells) -> NDArray[np.bool_]:
    """Per home: some open ground within ring_m of its outline reaches the street on the mesh's
    cells (a triangle with a cell, or the open band): Placement's test under the width form (its
    complement the homes walled in)."""
    ring = m.cx.spec.ring_m
    geom = shapely.polygons(m.xy[m.tri])
    qh, qt = shapely.STRtree(geom).query(geo.polys, predicate="dwithin", distance=ring)
    out = np.zeros(len(geo.polys), dtype=bool)
    out[qh[c.cell[qt] >= 0]] = True
    parts = shapely.get_parts(m.cx.band_open)
    parts = parts[shapely.area(parts) > 0]
    if len(parts):
        bh, _ = shapely.STRtree(parts).query(geo.polys, predicate="dwithin", distance=ring)
        out[bh] = True
    return out
