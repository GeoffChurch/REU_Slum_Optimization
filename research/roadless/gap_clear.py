"""The gap complex's clearing model: fractional clearings on one fixed mesh, for SIMP's relaxation
(NOTES, "The gap complex: passability, enclosure and flow without the raster"; the brief's stage 2,
"Clearing", and its addendum's convex forms).

THE MESH (`full_mesh`). Every footprint's outline and the street band's line, noded together
(shapely's union of lines), clipped to the block beyond the band, densified to s once and
polygonized: faces, each covered by a set of footprints (none: open ground; one: a building's
interior; several: an overlap). Each face is triangulated by GEOS (constrained Delaunay) on that
shared linework, so faces meet vertex for vertex; Lawson's flips within each face (the face
boundaries flagged, never flipped: a triangle lies in one face); Steiner points inserted as in
gap_flow (open faces kept clear of the outlines by the width form's clearance, interiors by R / 2).
The open faces are the open-ground complex's polygons, densified alike.

THE STATE. A building's clearing x_b in [0, 1] (SIMP's q and the eps world: k_b = eps + (1 - eps)
x_b^q). A triangle covered by C conducts by min_{b in C} k_b (open: 1). Each passage conducts its
two-point weight sigma |e| / d_e (gap_flow) times a factor (`Factors`), concave in x so P stays
convex (the addendum: per point, min over buildings b of [phi(d_b) + k_b (1 - phi(d_b))], each
term affine in one building's clearing):
- Closing(w): min of its two triangles' conductances; an open passage narrower than w (both ends
  on outlines) min(1, the sum of the k_b of the buildings at its ends): a filled gap opens as soon
  as a flanking building is cleared;
- Shy(w): the mean over GAUSS_N points along the passage of min over the buildings within w / 2 of
  the point of k_b (1 if none; a covering building is at 0);
- Ramp(w, s): the same over the buildings within w / 2 + s, each term phi(d) + k_b (1 - phi(d)).
At a 0/1 clearing each is the form's own (Shy: the passage's share beyond w / 2 of every standing
footprint, sampled). Street faces (on the band's line) likewise. The passages across a face
boundary were triangulated from its two sides apart: where their circumcentres lie in the wrong
order (d_e < 0, a constrained edge not Delaunay) the weight takes the two centres' distances to
the edge each clipped at CROSS_MIN |e| (counted, `FullMesh.crossed`).

THE OBJECTIVE (`Relaxation`): J_p = sum_h inj_h u_h^p over the homes live at baseline (gap_flow's
point demand on the open faces, fixed at baseline), its gradient by the adjoint (dJ/dkappa_e =
-T_e du_e dlam_e) through each factor's argmin (ties shared). Duck-typed for relax.simp (c.cost,
q, beta, proj, dproj, value_sgrad, value_grad).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import shapely
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components

import gap_complex as gc
import gap_flow as gf
import relax

CROSS_MIN = 0.05       # a passage across a face boundary whose circumcentres are in the wrong
                       # order: each centre's distance to the edge clipped at this x |e|


# ------------------------------------------------------------------ the mesh

def _faces(geo: gc.Geometry, spec: gc.Spec, grow: float
           ) -> tuple[NDArray[np.object_], NDArray[np.object_], NDArray[np.object_]]:
    """(the faces, densified, as polygons; per face its covering footprints; per face the
    footprints within `grow` of it, its zone: the offset curves at `grow` are in the linework, so
    a face lies wholly inside or outside each footprint's zero zone)."""
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
    noded = shapely.segmentize(noded, spec.s)
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
    return faces, np.array(cover, dtype=object), np.array(zone, dtype=object)


@dataclass(frozen=True, eq=False)
class FullMesh:
    """The block's mesh beyond the band, building interiors included (module docstring)."""
    xy: NDArray[np.float64]
    tri: NDArray[np.int64]
    face: NDArray[np.int64]          # per triangle
    cover: NDArray[np.object_]       # per face: its covering footprints (tuple)
    tcov: sp.csr_matrix              # (triangles, buildings) 0/1: the covering footprints
    tzone: sp.csr_matrix             # (triangles, buildings) 0/1: footprints within grow of it
    pa: NDArray[np.int64]
    pb: NDArray[np.int64]
    pu: NDArray[np.int64]
    pv: NDArray[np.int64]
    T: NDArray[np.float64]           # per passage sigma |e| / d_e (before cells merge)
    fa: NDArray[np.int64]            # street faces: triangle, vertices
    fu: NDArray[np.int64]
    fv: NDArray[np.int64]
    Tf: NDArray[np.float64]          # per street face sigma |f| / d_f (inf: ground)
    vb: sp.csr_matrix                # (vertices, buildings) 0/1: on the outline
    obstacle: NDArray[np.bool_]      # per vertex: on a footprint's outline
    sigma: float
    crossed: int
    flat: int
    skipped: int

    @property
    def nt(self) -> int:
        return len(self.tri)

    @property
    def open_tri(self) -> NDArray[np.bool_]:
        return np.asarray(self.tcov.sum(axis=1)).ravel() == 0


def full_mesh(geo: gc.Geometry, spec: gc.Spec, fs: gf.FlowSpec, form: gf.Width) -> FullMesh:
    faces, cover, zone = _faces(geo, spec, form.grow)
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
    xy, inv = np.unique(np.round(c.reshape(-1, 2) / gc.SNAP_M) * gc.SNAP_M, axis=0,
                        return_inverse=True)
    tri = inv.reshape(-1, 3).astype(np.int64)
    good = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 2] != tri[:, 0])
    tri, face = tri[good], face[good]
    cr = gf._cross(xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]])
    tri[cr < 0] = tri[cr < 0][:, [0, 2, 1]]
    nv0, nt = len(xy), len(tri)
    nbr = gc._neighbours(tri, nv0)
    con = (nbr < 0) | (face[np.maximum(nbr, 0)] != face[:, None])
    gc.flips_all(xy, tri, nbr, con, gc.DELAUNAY_TOL)
    # Steiner points: the open faces' kept clear by the form, the interiors' by R / 2
    open_f = np.array([len(cv) == 0 for cv in cover])
    region_open = shapely.union_all(faces[open_f]) if open_f.any() else shapely.Polygon()
    region_bld = shapely.union_all(faces[~open_f]) if (~open_f).any() else shapely.Polygon()
    pts = np.concatenate([gf.steiner_points(region_open, fs, form.steiner_clear),
                          gf.steiner_points(region_bld, fs, 0.0)])
    XY = np.zeros((nv0 + len(pts), 2))
    XY[:nv0] = xy
    TRI = np.zeros((nt + 2 * len(pts), 3), dtype=np.int64)
    TRI[:nt] = tri
    NBR = -np.ones_like(TRI)
    NBR[:nt] = nbr
    CON = np.zeros(TRI.shape, dtype=bool)
    CON[:nt] = con
    FACE = np.zeros(len(TRI), dtype=np.int64)
    FACE[:nt] = face
    # a split triangle's children keep its face (gap_flow._insert), and flips stay within faces
    nt2, nv2, skipped = gf.insert_points(XY, TRI, NBR, CON, FACE, nt, nv0, pts)
    xy, tri, face = XY[:nv2], TRI[:nt2], FACE[:nt2]
    nbr = NBR[:nt2]
    # covering footprints per triangle
    rows = np.repeat(np.arange(nt2), [len(cover[f]) for f in face])
    cols = np.concatenate([np.asarray(cover[f], dtype=np.int64) for f in face]) if len(rows) \
        else np.zeros(0, dtype=np.int64)
    n = len(geo.polys)
    tcov = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nt2, n))
    zr = np.repeat(np.arange(nt2), [len(zone[f]) for f in face])
    zc = np.concatenate([np.asarray(zone[f], dtype=np.int64) for f in face]) if len(zr) \
        else np.zeros(0, dtype=np.int64)
    tzone = sp.csr_matrix((np.ones(len(zr)), (zr, zc)), shape=(nt2, n))
    # passages, their two-point weights
    nvv = len(xy)
    e = np.stack([tri[:, [1, 2]], tri[:, [2, 0]], tri[:, [0, 1]]], axis=1).reshape(-1, 2)
    es = np.sort(e, axis=1)
    key = es[:, 0] * nvv + es[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    same = ks[1:] == ks[:-1]
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
    da = ((xy[pu] - cc[pa]) * nrm).sum(1)              # a's centre to the edge (positive: a's side)
    db = ((cc[pb] - xy[pu]) * nrm).sum(1)
    d = da + db
    area = 0.5 * np.abs(gf._cross(xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]))
    flat = np.minimum(area[pa], area[pb]) <= gf.FLAT_REL * L * L
    cross_f = face[pa] != face[pb]
    bad = cross_f & (d < gf.MERGE_REL * L)
    d = np.where(bad, np.maximum(da, CROSS_MIN * L) + np.maximum(db, CROSS_MIN * L), d)
    wrong = (d < -gf.MERGE_REL * L) & ~flat
    if wrong.any():
        e = np.flatnonzero(wrong)
        raise ValueError(f"{len(e)} passages within a face with their centres in the wrong order "
                         f"(d / L {np.round(d[e] / L[e], 6)[:5].tolist()}, at "
                         f"{np.round(xy[pu[e[:3]]], 3).tolist()}, areas "
                         f"{np.round(np.minimum(area[pa[e]], area[pb[e]]) / L[e] ** 2, 8)[:5].tolist()})")
    with np.errstate(divide="ignore"):
        T = np.where(flat & (d < gf.MERGE_REL * L), 0.0,
                     np.where(d <= gf.MERGE_REL * L, np.inf,     # inf: merged (one circumcentre)
                              fs.sigma * L / np.maximum(d, 1e-300)))
    # street faces: boundary edges on the band's line (not on a footprint's outline)
    single = np.ones(len(key), dtype=bool)
    single[i0] = False
    single[i1] = False
    bi = np.flatnonzero(single)
    mid = 0.5 * (xy[es[bi, 0]] + xy[es[bi, 1]])
    shapely.prepare(band_line := shapely.boundary(geo.streets.buffer(spec.band_m)))
    on_band = shapely.dwithin(band_line, shapely.points(mid[:, 0], mid[:, 1]), gc.TOL)
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
    # vertices on outlines
    tree = shapely.STRtree(geo.polys)
    qv, qb = tree.query(shapely.points(xy[:, 0], xy[:, 1]), predicate="dwithin",
                        distance=gc.TOL)
    vb = sp.csr_matrix((np.ones(len(qv)), (qv, qb)), shape=(nvv, n))
    obstacle = np.zeros(nvv, dtype=bool)
    obstacle[qv] = True
    obstacle[nv0:] = False
    return FullMesh(xy=xy, tri=tri, face=face, cover=cover, tcov=tcov, tzone=tzone, pa=pa,
                    pb=pb, pu=pu, pv=pv, T=T, fa=fa_, fu=fu_, fv=fv_, Tf=Tf, vb=vb,
                    obstacle=obstacle,
                    sigma=fs.sigma, crossed=int(bad.sum()),
                    flat=int((flat & (d < gf.MERGE_REL * L)).sum()),
                    skipped=skipped)




# ------------------------------------------------------------------ the factors

@dataclass(frozen=True, eq=False)
class Terms:
    """A family of factors (one per triangle): each the mean over its G points of the minimum over
    the point's terms (building b, phi) of phi + k_b (1 - phi), 1 with no term; `row` per term its
    (item, point) row (rows item * G + g), sorted; `b`, `phi`; `G`."""
    n: int
    G: int
    row: NDArray[np.int64]
    b: NDArray[np.int64]
    phi: NDArray[np.float64]

    def value_grad(self, k: NDArray[np.float64]) -> tuple[NDArray[np.float64], sp.csr_matrix]:
        """(per item its factor, d factor / d k as (items x buildings) sparse)."""
        n, G = self.n, self.G
        v = self.phi + k[self.b] * (1.0 - self.phi)
        nrow = n * G
        mins = np.ones(nrow)
        if len(self.row):
            np.minimum.at(mins, self.row, v)
        # the argmin terms (ties shared)
        at = v <= mins[self.row] + 1e-15
        cnt = np.bincount(self.row[at], minlength=nrow)
        wt = (1.0 - self.phi[at]) / cnt[self.row[at]] / G
        item = self.row[at] // G
        fac = mins.reshape(n, G).mean(axis=1)
        D = sp.csr_matrix((wt, (item, self.b[at])), shape=(n, len(k)))
        return fac, D.tocsr()


def _pairs(M: sp.csr_matrix) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    co = M.tocoo()
    o = np.lexsort((co.col, co.row))
    return co.row[o].astype(np.int64), co.col[o].astype(np.int64)


def tri_terms(geo: gc.Geometry, fm: FullMesh, form: gf.Width, quad: int) -> Terms:
    """Each triangle's factor terms (module docstring): Closing its covering footprints, Shy the
    footprints whose zero zone holds it (its zone: exact, the offset curves being outlines of the
    mesh), Ramp at each of its `quad` sub-points the footprints within w / 2 + s, phi(d)."""
    nt = fm.nt
    if isinstance(form, gf.Closing):
        r, b = _pairs(fm.tcov)
        return Terms(n=nt, G=1, row=r, b=b, phi=np.zeros(len(b)))
    if isinstance(form, gf.Shy):
        r, b = _pairs(fm.tzone)
        return Terms(n=nt, G=1, row=r, b=b, phi=np.zeros(len(b)))
    if isinstance(form, gf.Ramp):
        P, _wt, _t = gf._points_of(fm.xy, fm.tri, quad)     # triangle-major, quad points each
        pts = shapely.points(P[:, 0], P[:, 1])
        qi, qb = shapely.STRtree(geo.polys).query(pts, predicate="dwithin",
                                                  distance=form.w / 2.0 + form.s)
        d = shapely.distance(pts[qi], geo.polys[qb])
        phi = np.clip((d - form.w / 2.0) / form.s, 0.0, 1.0)
        keep = phi < 1.0
        o = np.argsort(qi[keep], kind="stable")
        return Terms(n=nt, G=quad, row=qi[keep][o].astype(np.int64),
                     b=qb[keep][o].astype(np.int64), phi=phi[keep][o])
    raise TypeError(f"no fractional factor for {form!r}")


def _harmonic_of(kt, Dt: sp.csr_matrix, a, b):
    """Per pair (a, b) of triangles their factors' harmonic mean 2 k_a k_b / (k_a + k_b), the two
    half-cells in series (concave and non-decreasing in both, so the passage's factor stays
    concave; 0 if either is), and its gradient."""
    va, vb_ = kt[a], kt[b]
    s_ = va + vb_
    ok = s_ > 0
    fac = np.where(ok, 2.0 * va * vb_ / np.where(ok, s_, 1.0), 0.0)
    ga = np.where(ok, 2.0 * vb_ ** 2 / np.where(ok, s_ ** 2, 1.0), 0.0)
    gb = np.where(ok, 2.0 * va ** 2 / np.where(ok, s_ ** 2, 1.0), 0.0)
    D = sp.diags(ga) @ Dt[a] + sp.diags(gb) @ Dt[b]
    return fac, D.tocsr()


class Factors:
    """The factor of every passage and street face from the triangles' (`tri_terms`): a passage
    the harmonic mean of its two triangles' (the half-cells in series), a street face its
    triangle's; under Closing a passage
    between two open triangles, both ends on outlines and shorter than w, min(1, the sum of the
    k_b of the footprints at its ends) (the filled gap)."""

    def __init__(self, geo: gc.Geometry, fm: FullMesh, form: gf.Width, quad: int):
        self.fm, self.form = fm, form
        self.tt = tri_terms(geo, fm, form, quad)
        self.fill_e, self.fill_b = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        if isinstance(form, gf.Closing) and form.w > 0:
            op = fm.open_tri
            L = np.hypot(*(fm.xy[fm.pu] - fm.xy[fm.pv]).T)
            narrow = op[fm.pa] & op[fm.pb] & fm.obstacle[fm.pu] & fm.obstacle[fm.pv] & (
                L < form.w)
            e = np.flatnonzero(narrow)
            vb = fm.vb.tocsr()
            ends = [np.zeros((0, 2), dtype=np.int64)]
            for x in (fm.pu[e], fm.pv[e]):
                sub = vb[x].tocoo()
                ends.append(np.stack([e[sub.row], sub.col], axis=1))
            pr = np.unique(np.concatenate(ends), axis=0)
            self.fill_e, self.fill_b = pr[:, 0], pr[:, 1]

    def values(self, k: NDArray[np.float64]):
        """(kp, Dp, kf, Df): the passages' and street faces' factors and gradients in k."""
        fm = self.fm
        kt, Dt = self.tt.value_grad(k)
        kp, Dp = _harmonic_of(kt, Dt, fm.pa, fm.pb)
        if len(self.fill_e):
            ne = len(fm.pa)
            ssum = np.bincount(self.fill_e, weights=k[self.fill_b], minlength=ne)
            filled = np.zeros(ne, dtype=bool)
            filled[self.fill_e] = True
            binds = filled & (ssum < kp)
            kp = np.where(filled, np.minimum(kp, ssum), kp)
            keep = sp.diags((~binds).astype(float))
            sel = binds[self.fill_e]
            Dp = (keep @ Dp + sp.csr_matrix(
                (np.ones(int(sel.sum())), (self.fill_e[sel], self.fill_b[sel])),
                shape=Dp.shape)).tocsr()
        return kp, Dp, kt[fm.fa], Dt[fm.fa].tocsr()


# ------------------------------------------------------------------ cells and the objective

@dataclass(frozen=True, eq=False)
class FullCells:
    """The full mesh's cells: triangles merged across zero-distance passages within a face (and,
    `lumped`, every building's interior triangles into one cell: one node per building);
    per passage between two cells its cell pair (`ea`, `eb`, the passage ids `pe`), per street
    face its cell (`fc`, ids `pf`); `ground` the cells grounded by an obtuse street face."""
    cell: NDArray[np.int64]
    n: int
    ea: NDArray[np.int64]
    eb: NDArray[np.int64]
    pe: NDArray[np.int64]
    fc: NDArray[np.int64]
    pf: NDArray[np.int64]


def fs_sigma(fm: FullMesh) -> float:
    return fm.sigma


def full_cells(fm: FullMesh, *, lumped: bool) -> FullCells:
    nt = fm.nt
    merge = np.isinf(fm.T) & (fm.face[fm.pa] == fm.face[fm.pb])
    a, b = [fm.pa[merge]], [fm.pb[merge]]
    if lumped:
        single = np.array([len(c) == 1 for c in fm.cover])
        tri_b = np.flatnonzero(single[fm.face])
        key = np.array([fm.cover[f][0] for f in fm.face[tri_b]], dtype=np.int64)
        order = np.argsort(key, kind="stable")
        tb_, kb = tri_b[order], key[order]
        same = kb[1:] == kb[:-1]
        a.append(tb_[1:][same])
        b.append(tb_[:-1][same])
    a_, b_ = np.concatenate(a), np.concatenate(b)
    _n, lab = connected_components(sp.csr_matrix((np.ones(len(a_)), (a_, b_)), shape=(nt, nt)),
                                   directed=False)
    keep = (lab[fm.pa] != lab[fm.pb]) & (fm.T >= gf.T_MIN * fs_sigma(fm))
    # the cells joined to the street through the kept passages (any factor above 0: the eps
    # world's); the rest (pockets joined only by a hairline) leave the system
    ncell = int(lab.max()) + 1
    ea, eb = lab[fm.pa[keep]], lab[fm.pb[keep]]
    street = ncell
    fcell = lab[fm.fa]
    r = np.concatenate([ea, fcell])
    c = np.concatenate([eb, np.full(len(fcell), street)])
    _k, comp = connected_components(sp.csr_matrix((np.ones(len(r)), (r, c)),
                                                  shape=(ncell + 1, ncell + 1)), directed=False)
    live = comp[:ncell] == comp[street]
    newid = -np.ones(ncell, dtype=np.int64)
    newid[live] = np.arange(int(live.sum()))
    k2 = live[ea] & live[eb]
    fk = live[fcell]
    return FullCells(cell=newid[lab], n=int(live.sum()), ea=newid[ea[k2]], eb=newid[eb[k2]],
                     pe=np.flatnonzero(keep)[k2], fc=newid[fcell[fk]],
                     pf=np.flatnonzero(fk))


class Model:
    """The complex's fractional clearing model on one block (module docstring) under a form: the
    full mesh, its cells, the factors, the baseline's demand (fixed, gap_flow's rule on the open
    triangles reaching the street at the 0/1 baseline). `J(k, power)` and its gradient in k."""

    def __init__(self, geo: gc.Geometry, spec: gc.Spec, fs: gf.FlowSpec, form: gf.Width, *,
                 lumped: bool, pop: NDArray[np.float64]):
        self.geo, self.spec, self.fs, self.form = geo, spec, fs, form
        self.fm = full_mesh(geo, spec, fs, form)
        self.fc = full_cells(self.fm, lumped=lumped)
        self.fac = Factors(geo, self.fm, form, fs.quad)
        n = len(geo.polys)
        self.n = n
        # the baseline's demand: k = 0 (every footprint standing, eps 0)
        k0 = np.zeros(n)
        kp, _, kf, _ = self.fac.values(k0)
        live = self._reach(kp, kf)
        cell = self.fc.cell
        tri_ok = (cell >= 0) & live[np.maximum(cell, 0)] & self.fm.open_tri
        self.dem, self.pcell = self._demand(tri_ok, pop)
        self.pop = pop

    def _reach(self, kp, kf) -> NDArray[np.bool_]:
        fc = self.fc
        on = kp[fc.pe] > 0
        g = np.zeros(fc.n, dtype=bool)
        g[fc.fc[kf[fc.pf] > 0]] = True
        street = fc.n
        touch = np.flatnonzero(g)
        r = np.concatenate([fc.ea[on], touch])
        c = np.concatenate([fc.eb[on], np.full(len(touch), street)])
        _k, comp = connected_components(sp.csr_matrix((np.ones(len(r)), (r, c)),
                                                      shape=(fc.n + 1, fc.n + 1)), directed=False)
        return comp[:fc.n] == comp[street]

    def _demand(self, tri_ok, pop):
        """gap_flow.point_demand's rule on the open triangles `tri_ok`: (the demand, each point's
        cell, -1 in the band)."""
        fm, spec = self.fm, self.spec
        tri = fm.tri[tri_ok]
        P, wt, t = gf._points_of(fm.xy, tri, self.fs.quad)
        cellp = self.fc.cell[np.flatnonzero(tri_ok)[t]]
        if self.form.grow > 0:
            # the form's zero zone takes no demand (as the grown complex has no ground there)
            _i, d0 = shapely.STRtree(self.geo.polys).query_nearest(
                shapely.points(P[:, 0], P[:, 1]), return_distance=True, all_matches=False)
            out = d0 >= self.form.grow
            P, wt, cellp = P[out], wt[out], cellp[out]
        _ib, band_open = gc._open_ground(self.geo, np.ones(self.n, dtype=bool), spec.band_m,
                                         carve=None, grow=0.0)
        btris = gc._triangulate(band_open, spec.s)
        if len(btris):
            bc = shapely.get_coordinates(btris).reshape(-1, 4, 2)[:, :3]
            bxy, binv = np.unique(bc.reshape(-1, 2), axis=0, return_inverse=True)
            BP, bwt, _bt = gf._points_of(bxy, binv.reshape(-1, 3), self.fs.quad)
        else:
            BP, bwt = np.zeros((0, 2)), np.zeros(0)
        xy = np.concatenate([P, BP])
        w = np.concatenate([wt, bwt])
        pcell = np.concatenate([cellp, -np.ones(len(BP), dtype=np.int64)])
        polys = self.geo.polys
        n = len(polys)
        pts = shapely.points(xy[:, 0], xy[:, 1])
        (pi, hi), dist = shapely.STRtree(polys).query_nearest(pts, return_distance=True,
                                                              all_matches=False)
        ring = dist <= spec.ring_m
        pi, hi = pi[ring], hi[ring]
        has = np.bincount(hi, minlength=n) > 0
        rows, cols = [pi], [hi]
        lack = np.flatnonzero(~has)
        if len(lack):
            qi, qh = shapely.STRtree(pts).query(polys[lack], predicate="dwithin",
                                                distance=spec.ring_m)
            rows.append(qh)
            cols.append(lack[qi])
        r, cl = np.concatenate(rows), np.concatenate(cols)
        tot = np.bincount(cl, weights=w[r], minlength=n)
        S = sp.csr_matrix((w[r] / np.where(tot[cl] > 0, tot[cl], 1.0), (r, cl)),
                          shape=(len(xy), n))
        return gf.PointDemand(xy=xy, band=pcell < 0, S=S, walled=tot == 0,
                              shadowed=np.zeros(n, dtype=bool)), pcell

    def system(self, k: NDArray[np.float64]):
        """(A over the cells, the factors' values and gradients)."""
        fm, fc = self.fm, self.fc
        kp, Dp, kf, Df = self.fac.values(k)
        Tp = fm.T[fc.pe] * kp[fc.pe]
        Tf = (np.where(np.isinf(fm.Tf), self.fs.sigma / CROSS_MIN, fm.Tf) * kf)[fc.pf]
        n = fc.n
        diag = np.bincount(fc.ea, weights=Tp, minlength=n) + np.bincount(
            fc.eb, weights=Tp, minlength=n) + np.bincount(fc.fc, weights=Tf, minlength=n)
        A = sp.csr_matrix((np.concatenate([-Tp, -Tp, diag]),
                           (np.concatenate([fc.ea, fc.eb, np.arange(n)]),
                            np.concatenate([fc.eb, fc.ea, np.arange(n)]))), shape=(n, n))
        return A, (kp, Dp, kf, Df)

    def J(self, k: NDArray[np.float64], inj: NDArray[np.float64], power: float
          ) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        """(J_power of the injection `inj` per home, per home its escape time, dJ / dk)."""
        fm, fc, dem = self.fm, self.fc, self.dem
        A, (kp, Dp, kf, Df) = self.system(k)
        lu = gf.factor(A)
        f_pt = dem.S @ inj
        on = self.pcell >= 0
        b = np.bincount(self.pcell[on], weights=f_pt[on], minlength=fc.n)
        u = lu.solve(b)
        u_pt = np.where(on, u[np.maximum(self.pcell, 0)], 0.0)
        uh = np.asarray(dem.S.T @ u_pt).ravel()
        live = ~dem.walled
        J = float((inj[live] * uh[live] ** power).sum())
        if power == 1.0:
            lam = u
        else:
            g_h = np.where(live, inj * power * np.maximum(uh, 0.0) ** (power - 1.0), 0.0)
            gp = dem.S @ g_h
            lam = lu.solve(np.bincount(self.pcell[on], weights=gp[on], minlength=fc.n))
        du = u[fc.ea] - u[fc.eb]
        dl = lam[fc.ea] - lam[fc.eb]
        dJ_dkp = np.zeros(len(fm.pa))
        dJ_dkp[fc.pe] = -fm.T[fc.pe] * du * dl
        Tf0 = np.where(np.isinf(fm.Tf), self.fs.sigma / CROSS_MIN, fm.Tf)
        dJ_dkf = np.zeros(len(fm.fa))
        dJ_dkf[fc.pf] = -Tf0[fc.pf] * u[fc.fc] * lam[fc.fc]
        grad = np.asarray(Dp.T @ dJ_dkp).ravel() + np.asarray(Df.T @ dJ_dkf).ravel()
        uh = np.where(live, uh, np.nan)
        return J, uh, grad


@dataclass
class _C:
    """What relax.simp reads of a Clearing: the cost and the building count."""
    cost: NDArray[np.float64]
    n: int


class Relaxation:
    """relax.Relaxation's duck for relax.simp on the complex: J_power in the eps world, k_b = eps +
    (1 - eps) proj(x_b)^q, its gradient in the projected clearing."""

    def __init__(self, model: Model, power: float, *, eps: float, inj: NDArray[np.float64],
                 cost: NDArray[np.float64]):
        self.m, self.power, self.eps, self.inj = model, power, eps, inj
        self.c = _C(cost=cost, n=len(cost))
        self.q, self.beta = 1.0, 0.0

    def proj(self, x):
        if self.beta == 0.0:
            return x
        b, E = self.beta, relax.ETA
        return (np.tanh(b * E) + np.tanh(b * (x - E))) / (np.tanh(b * E) + np.tanh(b * (1 - E)))

    def dproj(self, x):
        if self.beta == 0.0:
            return np.ones_like(x)
        b, E = self.beta, relax.ETA
        return b * (1 - np.tanh(b * (x - E)) ** 2) / (np.tanh(b * E) + np.tanh(b * (1 - E)))

    def k_of(self, x):
        return self.eps + (1.0 - self.eps) * self.proj(x) ** self.q

    def value(self, x) -> float:
        return self.m.J(self.k_of(x), self.inj, self.power)[0]

    def value_sgrad(self, x):
        s = self.proj(x)
        J, _uh, g = self.m.J(self.k_of(x), self.inj, self.power)
        g = g * (1.0 - self.eps)
        if self.q != 1.0:
            g = g * self.q * np.maximum(s, 1e-300) ** (self.q - 1.0)
        return J, g

    def value_grad(self, x):
        J, gs = self.value_sgrad(x)
        return J, gs * self.dproj(x)
