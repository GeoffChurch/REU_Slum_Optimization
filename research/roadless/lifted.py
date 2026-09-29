"""Roadless, heading-aware conduction on a block's free space.

A walker moves forward or back along its current heading and turns slowly: the state is
(cell, axis), the along-axis edges join a cell to the next cell along that axis's lattice vector,
and the turning edges join adjacent axes in the same cell. The operator is symmetric, so the
escape-time identity and Rayleigh monotonicity both hold: P = f^T L^-1 f is the total expected
escape time AND the power of the system-optimal linear-congestion flow, and adding free space can
only lower it.

Roads do not exist here. A network enters only through the space its corridor frees: `free |=
corridor` (CARVE, the geometry the displacement charge already pays for) or every building the
corridor touches is removed whole (OBLITERATE). Demand stays where the buildings were
(rehoused in place), so only the operator changes and monotonicity is exact.

Continuum it approximates:  E(u) = int int (d_s u)^2 + (1/ell^2)(d_theta u)^2  dx dtheta
    along edge (x, x+v_k), layer k:   w = m_k / |v_k|^2        (h^2 cancels)
    turning edge (x,k)-(x,k+1):       w = h^2 / (ell^2 dtheta_{k,k+1})
m_k is axis k's share of [0, pi); a straight open channel of width W and length L conducts
m_k W / L in its own layer at every lattice angle (checked in checks.py).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import shapely
from numpy.typing import NDArray
from scipy import ndimage
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path.home() / ".cache/reblock-research/pydeps"))
import pyamg  # noqa: E402

# axis lattice vectors (dx = column, dy = row), angles in [0, pi)
AXES = {
    8: [(1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1)],
    16: [(1, 0), (3, 1), (2, 1), (3, 2), (1, 1), (2, 3), (1, 2), (1, 3),
         (0, 1), (-1, 3), (-1, 2), (-2, 3), (-1, 1), (-3, 2), (-2, 1), (-3, 1)],
}


def axes(K: int) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64],
                          NDArray[np.float64]]:
    """(vectors, angles, m_k, gap to the next axis cyclically)."""
    v = np.array(AXES[K], dtype=np.int64)
    th = np.arctan2(v[:, 1], v[:, 0]) % np.pi
    order = np.argsort(th)
    assert (order == np.arange(K)).all()
    nxt = np.roll(th, -1)
    nxt[-1] += np.pi
    gap = nxt - th
    m = 0.5 * (gap + np.roll(gap, 1))
    return v, th, m, gap


def _line_cells(dx: int, dy: int) -> list[tuple[int, int]]:
    """Intermediate cells a step (dx, dy) passes through (supercover, endpoints excluded)."""
    n = 64
    ts = (np.arange(1, n) + 0.0) / n
    cells = set()
    for t in ts:
        x, y = t * dx, t * dy
        # every cell whose square the segment enters (corner contacts count as both neighbours)
        for cx in {int(np.floor(x + 0.5 - 1e-9)), int(np.ceil(x - 0.5 + 1e-9))}:
            for cy in {int(np.floor(y + 0.5 - 1e-9)), int(np.ceil(y - 0.5 + 1e-9))}:
                if (cx, cy) not in {(0, 0), (dx, dy)}:
                    cells.add((cx, cy))
    return sorted(cells)


@dataclass(frozen=True, eq=False)
class Grid:
    x0: float
    y0: float
    h: float
    inside: NDArray[np.bool_]
    building: NDArray[np.bool_]          # cell centre inside some footprint (original)
    ground: NDArray[np.bool_]            # within `band_m` of Block.streets
    dist_b: NDArray[np.float64]          # metres to the nearest building cell (EDT)
    xy: NDArray[np.float64]              # (ny, nx, 2) cell centres

    @classmethod
    def of(cls, boundary, footprints, streets, h: float, band_m: float = 1.0,
           offset: tuple[float, float] = (0.3713, 0.1931)) -> Grid:
        """`offset` (fractions of h) keeps cell centres off exactly-aligned geometry."""
        minx, miny, maxx, maxy = boundary.bounds
        xs = np.arange(minx - h + offset[0] * h, maxx + 2 * h, h)
        ys = np.arange(miny - h + offset[1] * h, maxy + 2 * h, h)
        X, Y = np.meshgrid(xs, ys)
        inside = shapely.contains_xy(boundary, X, Y)
        building = np.zeros_like(inside)
        if len(footprints):
            fp = shapely.union_all(np.asarray(footprints))
            shapely.prepare(fp)
            building[inside] = shapely.contains_xy(fp, X[inside], Y[inside])
        st = shapely.union_all(np.asarray(streets)).buffer(band_m)
        shapely.prepare(st)
        ground = np.zeros_like(inside)
        ground[inside] = shapely.contains_xy(st, X[inside], Y[inside])
        dist_b = ndimage.distance_transform_edt(~building) * h
        return cls(x0=float(xs[0]), y0=float(ys[0]), h=h, inside=inside, building=building,
                   ground=ground, dist_b=dist_b, xy=np.stack([X, Y], axis=-1))

    def mask_of(self, geom) -> NDArray[np.bool_]:
        out = np.zeros_like(self.inside)
        if geom is None or geom.is_empty:
            return out
        shapely.prepare(geom)
        X, Y = self.xy[..., 0], self.xy[..., 1]
        out[self.inside] = shapely.contains_xy(geom, X[self.inside], Y[self.inside])
        return out


def demand(grid: Grid, footprints, ring_m: float = 1.0) -> tuple[NDArray[np.float64], int]:
    """Per-cell injection (ny, nx): each building injects 1, uniformly over the free cells within
    `ring_m` of it whose nearest building it is. Buildings with no such cell (enclosed) fall back
    to their nearest free cell. Returns (field, n_fallback)."""
    free0 = grid.inside & ~grid.building
    ring = free0 & (grid.dist_b <= ring_m + 1e-9)
    rr, cc = np.nonzero(ring)
    polys = np.asarray(footprints)
    tree = shapely.STRtree(polys)
    pts = shapely.points(grid.xy[rr, cc, 0], grid.xy[rr, cc, 1])
    idx, _ = tree.query_nearest(pts, return_distance=True, all_matches=False)
    owner = np.full(len(rr), -1)
    owner[idx[0]] = idx[1]
    n = len(polys)
    cnt = np.bincount(owner, minlength=n).astype(float)
    f = np.zeros(grid.inside.shape)
    np.add.at(f, (rr, cc), 1.0 / cnt[owner])
    missing = np.nonzero(cnt == 0)[0]
    if len(missing):
        fr, fc = np.nonzero(free0)
        from scipy.spatial import cKDTree
        t = cKDTree(np.c_[grid.xy[fr, fc, 0], grid.xy[fr, fc, 1]])
        cen = shapely.get_coordinates(shapely.centroid(polys[missing]))
        _, k = t.query(cen)
        np.add.at(f, (fr[k], fc[k]), 1.0)
    return f, len(missing)


@dataclass(frozen=True)
class Params:
    ell_m: float = 2.0       # turning length: metres travelled per radian of heading change
    K: int = 8


def along_edges(free: NDArray[np.bool_], p: Params):
    """Per axis k: (k, a, b, w) with a, b free-cell ids (row-major order of `free`), b = a + v_k,
    every cell the step crosses free."""
    v, _th, m, _gap = axes(p.K)
    ny, nx = free.shape
    cell = -np.ones((ny, nx), dtype=np.int64)
    cell[free] = np.arange(int(free.sum()))
    rr, cc = np.nonzero(free)
    for k in range(p.K):
        dx, dy = int(v[k, 0]), int(v[k, 1])
        r2, c2 = rr + dy, cc + dx
        ok = (r2 >= 0) & (r2 < ny) & (c2 >= 0) & (c2 < nx)
        ok[ok] &= free[r2[ok], c2[ok]]
        for (ix, iy) in _line_cells(dx, dy):
            r3, c3 = rr + iy, cc + ix
            o3 = (r3 >= 0) & (r3 < ny) & (c3 >= 0) & (c3 < nx)
            o3[o3] &= free[r3[o3], c3[o3]]
            ok &= o3
        a = cell[rr[ok], cc[ok]]
        b = cell[r2[ok], c2[ok]]
        yield k, a, b, np.full(len(a), m[k] / float(dx * dx + dy * dy))


def crossing(free: NDArray[np.bool_], ground: NDArray[np.bool_], sol: Solution,
             p: Params) -> dict[str, float]:
    """How much of the along-heading power flows where streams of different headings share a
    cell. Per free cell, J = sum_k F_k e_k (the net flux vector) and S = sum_k |F_k|;
    cancellation 1 - |J|/S is ~0 for one stream (a little above 0 for a stream spread over
    neighbouring axes), 0.29 for two equal perpendicular streams, 1 for opposed ones. Reported
    power-weighted over cells."""
    v, _th, _m, _gap = axes(p.K)
    e = v / np.linalg.norm(v, axis=1)[:, None]
    nc = int(free.sum())
    is_g = ground[free]
    uk = np.zeros((nc, p.K))
    live = sol.unk_cell >= 0
    uk[live] = sol.u.reshape(-1, p.K)
    J = np.zeros((nc, 2))
    S = np.zeros(nc)
    pw = np.zeros(nc)
    for k, a, b, w in along_edges(free, p):
        F = w * (uk[a, k] - uk[b, k])            # flux from a to b along +v_k
        pe = w * (uk[a, k] - uk[b, k]) ** 2
        for c in (a, b):
            np.add.at(J, c, 0.5 * F[:, None] * e[k][None, :])
            np.add.at(S, c, 0.5 * np.abs(F))
            np.add.at(pw, c, 0.5 * pe)
    ok = S > 0
    canc = np.zeros(nc)
    canc[ok] = 1.0 - np.linalg.norm(J[ok], axis=1) / S[ok]
    tot = pw.sum()
    return dict(mean=float((canc * pw).sum() / tot),
                share_gt_0_2=float(pw[canc > 0.2].sum() / tot),
                share_gt_0_29=float(pw[canc > 0.29].sum() / tot),
                share_gt_0_5=float(pw[canc > 0.5].sum() / tot))


def operator(free: NDArray[np.bool_], ground: NDArray[np.bool_], h: float, p: Params):
    """(L restricted to unknowns, unknown index per (cell, axis) or -1)."""
    _v, _th, _m, gap = axes(p.K)
    K = p.K
    ny, nx = free.shape
    cell = -np.ones((ny, nx), dtype=np.int64)
    cell[free] = np.arange(int(free.sum()))
    nc = int(free.sum())
    is_g = ground[free]
    # unknowns: non-ground free cells x K
    unk_cell = -np.ones(nc, dtype=np.int64)
    unk_cell[~is_g] = np.arange(int((~is_g).sum()))
    N = int((~is_g).sum()) * K
    rows, cols, vals = [], [], []
    diag = np.zeros(N)

    def add_edge(a_cell, b_cell, ka, kb, w):
        """a/b are free-cell ids; ground ends only add to the other end's diagonal."""
        ga, gb = is_g[a_cell], is_g[b_cell]
        ia = unk_cell[a_cell] * K + ka
        ib = unk_cell[b_cell] * K + kb
        both = ~ga & ~gb
        rows.append(ia[both]); cols.append(ib[both]); vals.append(-w[both])
        rows.append(ib[both]); cols.append(ia[both]); vals.append(-w[both])
        np.add.at(diag, ia[~ga], w[~ga])
        np.add.at(diag, ib[~gb], w[~gb])

    for k, a, b, w in along_edges(free, p):
        add_edge(a, b, np.full(len(a), k), np.full(len(a), k), w)
    # turning edges inside every non-ground cell (ground cells are u = 0 in every layer)
    live = np.nonzero(~is_g)[0]
    for k in range(K):
        k2 = (k + 1) % K
        w = np.full(len(live), h * h / (p.ell_m ** 2 * gap[k]))
        add_edge(live, live, np.full(len(live), k), np.full(len(live), k2), w)
    rows.append(np.arange(N)); cols.append(np.arange(N)); vals.append(diag)
    L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N, N))
    return L, cell, unk_cell


def grounded(grid: Grid, free: NDArray[np.bool_], p: Params) -> NDArray[np.bool_]:
    """Free cells with a path to the street (ground cells included)."""
    L, cell, unk_cell = operator(free, grid.ground, grid.h, p)
    n_comp, lab = sp.csgraph.connected_components(L, directed=False)
    dg = np.asarray(L.sum(axis=1)).ravel()
    ok_comp = np.zeros(n_comp, bool)
    ok_comp[np.unique(lab[dg > 1e-12])] = True
    rr, cc = np.nonzero(free)
    out = np.zeros_like(free)
    live = unk_cell >= 0
    good = np.ones(len(rr), bool)
    good[live] = ok_comp[lab[unk_cell[live] * p.K]]
    out[rr[good], cc[good]] = True
    return out


def _spd_solve(L, b, rtol: float = 1e-9):
    if L.shape[0] < 20000:
        return spsolve(L.tocsc(), b)
    ml = pyamg.smoothed_aggregation_solver(L, symmetry="symmetric", max_coarse=2000)
    res: list[float] = []
    u = ml.solve(b, tol=rtol, accel="cg", maxiter=500, residuals=res)
    if res[-1] > rtol * res[0] * 10:
        raise RuntimeError(f"AMG-CG did not converge: {res[-1] / res[0]:.2e} after {len(res)}")
    return u


@dataclass(frozen=True, eq=False)
class Solution:
    P: float
    u: NDArray[np.float64]          # per unknown
    cell: NDArray[np.int64]
    unk_cell: NDArray[np.int64]
    n_unknowns: int


def solve(grid: Grid, free: NDArray[np.bool_], f_cell: NDArray[np.float64], p: Params,
          ) -> Solution:
    """Total escape power. Injection on a building/non-free cell would be a bug: demand cells
    are free in the original geometry and freeing only adds cells."""
    assert not (f_cell[~free] > 0).any()
    L, cell, unk_cell = operator(free, grid.ground, grid.h, p)
    _, _, m, _ = axes(p.K)
    rr, cc = np.nonzero(free)
    fc = f_cell[rr, cc]
    live = unk_cell >= 0
    b = np.zeros(L.shape[0])
    # each cell's injection spread over the axes by angular share
    idx = (unk_cell[live][:, None] * p.K + np.arange(p.K)[None, :]).ravel()
    b[idx] = (fc[live][:, None] * (m / np.pi)[None, :]).ravel()
    # isolated components with injection but no ground would be singular: detect
    n_comp, lab = sp.csgraph.connected_components(L, directed=False)
    if n_comp > 1:
        grounded = np.zeros(n_comp, bool)
        dg = np.asarray(L.sum(axis=1)).ravel()          # > 0 where an edge leaves to ground
        grounded[np.unique(lab[dg > 1e-12])] = True
        bad = ~grounded[lab]
        if (b[bad] > 0).any():
            raise ValueError(f"{(b[bad] > 0).sum()} injected unknowns have no path to ground")
        keep = ~bad
        u = np.zeros(L.shape[0])
        u[keep] = _spd_solve(L[keep][:, keep].tocsr(), b[keep])
    else:
        u = _spd_solve(L, b)
    return Solution(P=float(b @ u), u=u, cell=cell, unk_cell=unk_cell, n_unknowns=L.shape[0])
