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

A cell partly covered by footprints conducts in part: its turning edges by its open fraction (its
open area), an along step by the share of its width open all the way from one cell's centre to
the next, read on the S x S sub-samples (step_widths: the face aperture, Fws; NOTES, "The grid
fix: steps that conduct by their open width"). The operator reads the footprints only through
the sub-samples (Grid.bsub, label_sub), so a change to the geometry they rasterize needs no
change here.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import math
import re
import weakref
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, NamedTuple, Protocol

import numpy as np
import scipy.sparse as sp
import shapely
from numba import njit
from numpy.typing import NDArray
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree

import pyamg

# axis lattice vectors (dx = column, dy = row), angles in [0, pi)
AXES = {
    8: [(1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1)],
    16: [(1, 0), (3, 1), (2, 1), (3, 2), (1, 1), (2, 3), (1, 2), (1, 3),
         (0, 1), (-1, 3), (-1, 2), (-2, 3), (-1, 1), (-3, 2), (-2, 1), (-3, 1)],
}


# cell centres off exactly-aligned geometry: the lattice's offset from the block's bounds, in
# fractions of a cell
OFFSET = (0.3713, 0.1931)
BAND_M = 1.0              # ground: cell centres within this many metres of a street
RING_M = 1.0              # the demand ring (Placement): cell centres within this of a home


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


def lattice(boundary, h: float,
            offset: tuple[float, float]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The fine lattice's cell-centre columns xs and rows ys over a block (every mesh's)."""
    minx, miny, maxx, maxy = boundary.bounds
    return (np.arange(minx - h + offset[0] * h, maxx + 2 * h, h),
            np.arange(miny - h + offset[1] * h, maxy + 2 * h, h))


@dataclass(frozen=True, eq=False)
class Grid:
    x0: float
    y0: float
    h: float
    inside: NDArray[np.bool_]            # some sub-sample of the cell is inside the block
    building: NDArray[np.bool_]          # most of the cell is footprint (for the EDT only)
    ground: NDArray[np.bool_]            # centre within `band_m` of Block.streets
    dist_b: NDArray[np.float64]          # metres to the nearest building cell (EDT)
    xy: NDArray[np.float64]              # (ny, nx, 2) cell centres
    isub: NDArray[np.bool_]              # (ny, nx, S*S) sub-samples inside the block
    bsub: NDArray[np.bool_]              # (ny, nx, S*S) sub-samples inside a footprint
    S: int
    # per-grid derived structure (edge pattern, master aggregates), built on first use
    cache: dict = dataclasses.field(default_factory=dict, compare=False, repr=False)

    @property
    def ff0(self) -> NDArray[np.float64]:
        """Open fraction of each cell: inside the block and not footprint."""
        return (self.isub & ~self.bsub).mean(axis=-1)

    # A field is already the (ny, nx) raster an along-conductance scans: the maps between them
    # (CompositeGrid's) are the identity.
    def paint(self, field, xp):
        """The raster of a field: each cell's value over its pixels."""
        return field

    def mean(self, R, xp):
        """(..., ny, nx) raster -> (..., field): each cell's mean over its pixels."""
        return R

    def mean_T(self, A, xp):
        """The adjoint of `mean`: (..., field) -> (..., ny, nx)."""
        return A

    def gather(self, G, xp):
        """The adjoint of `paint`: (ny, nx) -> field, each cell's sum over its pixels."""
        return G

    def pattern(self, K: int, xp) -> Pattern:
        """The grid's Pattern for K axes on device `xp`, built once per grid."""
        key = ("pattern", K, xp.__name__)
        if key not in self.cache:
            v, _th, m, _gap = axes(K)
            inside = self.inside
            ny, nx = inside.shape
            rr, cc = np.nonzero(inside)
            along = []
            for k in range(K):
                dx, dy = int(v[k, 0]), int(v[k, 1])
                lines = [(dx, dy), *_line_cells(dx, dy)]
                ok = np.ones(len(rr), dtype=bool)
                for ix, iy in lines:
                    r3, c3 = rr + iy, cc + ix
                    o3 = (r3 >= 0) & (r3 < ny) & (c3 >= 0) & (c3 < nx)
                    o3[o3] &= inside[r3[o3], c3[o3]]
                    ok &= o3
                a = rr[ok] * nx + cc[ok]
                along.append((xp.asarray(a), xp.asarray(a + dy * nx + dx),
                              xp.asarray(np.stack([a + iy * nx + ix for ix, iy in lines[1:]])
                                         if len(lines) > 1 else np.zeros((0, len(a)), np.int64)),
                              m[k] / float(dx * dx + dy * dy)))
            self.cache[key] = Pattern(along=along, cells=xp.asarray(rr * nx + cc))
        return self.cache[key]

    def track_axes(self, K: int, xp) -> list[_AxisTracks]:
        """The steps' tracks on this grid (_AxisTracks), per axis: a slot's cell by its offset
        from each edge's first cell (the pattern's own a, b and line; others computed, the dummy
        cell where off the raster)."""
        v, _th, _m, _gap = axes(K)
        pat = self.pattern(K, xp)
        ny, nx = self.inside.shape
        n, S2 = ny * nx, self.S * self.S
        out = []
        for k in range(K):
            a, b, line, _wk = pat.along[k]
            tr = axis_tracks(int(v[k, 0]), int(v[k, 1]), self.S)
            cells = [a, b, *list(line)]
            r, c = a // nx, a % nx
            for ox, oy in tr.cells[len(cells):]:
                rr, cc = r + oy, c + ox
                ok = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx)
                cells.append(xp.where(ok, rr * nx + cc, n))
            out.append(_AxisTracks(base=[q * S2 for q in cells], tracks=tr.tracks, coarse=None))
        return out

    def cell_area(self, xp, free) -> float:
        """The area of each of the `free` cells (one size: a number)."""
        return self.h * self.h

    def _sub_xy(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        t = ((np.arange(self.S) + 0.5) / self.S - 0.5) * self.h
        ox, oy = np.meshgrid(t, t)
        X = self.xy[..., 0][..., None] + ox.ravel()[None, None, :]
        Y = self.xy[..., 1][..., None] + oy.ravel()[None, None, :]
        return X, Y

    def sub_of(self, geom) -> NDArray[np.bool_]:
        """(ny, nx, S*S): sub-samples inside `geom` (only those inside the block)."""
        out = np.zeros_like(self.isub)
        if geom is None or geom.is_empty:
            return out
        shapely.prepare(geom)
        X, Y = self._sub_xy()
        out[self.isub] = shapely.contains_xy(geom, X[self.isub], Y[self.isub])
        return out

    @classmethod
    def of(cls, boundary, footprints, streets, h: float, band_m: float = BAND_M,
           offset: tuple[float, float] = OFFSET) -> Grid:
        """`offset` (fractions of h) keeps cell centres off exactly-aligned geometry. Sub-sample
        (i, j) of the fine lattice sits at (x0 - h/2 + (j + 1/2) h/S, y0 - h/2 + (i + 1/2) h/S):
        inside the block by a scanline fill of its rings, inside a footprint by
        `_rasterize`."""
        xs, ys = lattice(boundary, h, offset)
        X, Y = np.meshgrid(xs, ys)
        S = 4
        ny, nx = X.shape
        d = h / S
        fx = xs[0] - h / 2 + (np.arange(nx * S) + 0.5) * d
        fy = ys[0] - h / 2 + (np.arange(ny * S) + 0.5) * d
        isub = _to_cells(_scanfill(boundary, fx, fy), S)
        bsub = isub & _to_cells(_rasterize(np.asarray(footprints), fx, fy) >= 0, S)
        inside = isub.any(axis=-1)
        building = bsub.mean(axis=-1) > 0.5
        st = shapely.union_all(np.asarray(streets)).buffer(band_m)
        shapely.prepare(st)
        ground = np.zeros_like(inside)
        ground[inside] = shapely.contains_xy(st, X[inside], Y[inside])
        dist_b = ndimage.distance_transform_edt(~building) * h
        return cls(x0=float(xs[0]), y0=float(ys[0]), h=h, inside=inside, building=building,
                   ground=ground, dist_b=dist_b, xy=np.stack([X, Y], axis=-1),
                   isub=isub, bsub=bsub, S=S)

    def label_sub(self, polys) -> NDArray[np.int64]:
        """(ny, nx, S*S): index of the footprint holding each footprint sub-sample (-1 none;
        overlaps go to the later footprint)."""
        S, h = self.S, self.h
        ny, nx = self.inside.shape
        d = h / S
        fx = self.x0 - h / 2 + (np.arange(nx * S) + 0.5) * d
        fy = self.y0 - h / 2 + (np.arange(ny * S) + 0.5) * d
        return np.where(self.bsub, _to_cells(_rasterize(np.asarray(polys), fx, fy), S), -1)

    def mask_of(self, geom) -> NDArray[np.bool_]:
        out = np.zeros_like(self.inside)
        if geom is None or geom.is_empty:
            return out
        shapely.prepare(geom)
        X, Y = self.xy[..., 0], self.xy[..., 1]
        out[self.inside] = shapely.contains_xy(geom, X[self.inside], Y[self.inside])
        return out

    def cells_near(self, geom, dist: float) -> NDArray[np.int64]:
        """Flat indices of the inside cells whose centres lie within `dist` of `geom` (shapely's
        distance: a centre inside it is at 0), tested in its bounding box padded by dist + h
        only."""
        xs, ys = self.xy[0, :, 0], self.xy[:, 0, 1]
        x0, y0, x1, y1 = geom.bounds
        pad = dist + self.h
        c0, c1 = np.searchsorted(xs, x0 - pad), np.searchsorted(xs, x1 + pad, side="right")
        r0, r1 = np.searchsorted(ys, y0 - pad), np.searchsorted(ys, y1 + pad, side="right")
        rr, cc = np.nonzero(self.inside[r0:r1, c0:c1])
        rr, cc = rr + r0, cc + c0
        ok = shapely.dwithin(geom, shapely.points(self.xy[rr, cc, 0], self.xy[rr, cc, 1]), dist)
        return (rr * self.inside.shape[1] + cc)[ok]


def _to_cells(fine, S: int):
    """(ny S, nx S) fine lattice -> (ny, nx, S*S), sub-sample index (row in cell) S + column."""
    ny, nx = fine.shape[0] // S, fine.shape[1] // S
    return fine.reshape(ny, S, nx, S).transpose(0, 2, 1, 3).reshape(ny, nx, S * S)


def _scanfill(geom, fx: NDArray[np.float64], fy: NDArray[np.float64]) -> NDArray[np.bool_]:
    """(len(fy), len(fx)): lattice points inside `geom` (a polygon or multipolygon), by the
    even-odd rule along each row: a point is inside when an odd number of ring edges cross its
    row to its left. Exact except for points on an edge, which the grid offset avoids."""
    segs = []
    for ring in shapely.get_rings(shapely.get_parts(geom)):
        c = shapely.get_coordinates(ring)
        segs.append(np.hstack([c[:-1], c[1:]]))
    x1, y1, x2, y2 = np.vstack(segs).T
    out = np.zeros((len(fy), len(fx)), dtype=bool)
    for i, y in enumerate(fy):
        hit = (y1 <= y) != (y2 <= y)                       # the edge spans the row (half-open)
        xc = np.sort(x1[hit] + (y - y1[hit]) * (x2[hit] - x1[hit]) / (y2[hit] - y1[hit]))
        out[i] = np.searchsorted(xc, fx) % 2 == 1
    return out


def _rasterize(polys, fx: NDArray[np.float64], fy: NDArray[np.float64]) -> NDArray[np.int64]:
    """(len(fy), len(fx)): index of the polygon holding each lattice point, -1 none (overlaps go
    to the later polygon). Each polygon is tested only on the lattice points inside its
    bounding box (fx, fy evenly spaced)."""
    fine = -np.ones((len(fy), len(fx)), dtype=np.int64)
    if len(polys) == 0:
        return fine
    d = fx[1] - fx[0]
    for k, (bx0, by0, bx1, by1) in enumerate(shapely.bounds(polys)):
        j0 = max(int(np.floor((bx0 - fx[0]) / d)), 0)
        j1 = min(int(np.ceil((bx1 - fx[0]) / d)) + 1, len(fx))
        i0 = max(int(np.floor((by0 - fy[0]) / d)), 0)
        i1 = min(int(np.ceil((by1 - fy[0]) / d)) + 1, len(fy))
        if j1 <= j0 or i1 <= i0:
            continue
        X, Y = np.meshgrid(fx[j0:j1], fy[i0:i1])
        fine[i0:i1, j0:j1][shapely.contains_xy(polys[k], X, Y)] = k
    return fine


def _scanfill_at(geom, fx: NDArray[np.float64], fy: NDArray[np.float64], R: NDArray[np.int64],
                 C: NDArray[np.int64]) -> NDArray[np.bool_]:
    """`_scanfill` at the lattice points (fy[R], fx[C]) only: the same rule and arithmetic, so
    the same answer, without the whole lattice. A row's crossings come from the ring edges
    spanning it (min(y1, y2) <= y < max(y1, y2), `_scanfill`'s half-open test)."""
    segs = []
    for ring in shapely.get_rings(shapely.get_parts(geom)):
        c = shapely.get_coordinates(ring)
        segs.append(np.hstack([c[:-1], c[1:]]))
    x1, y1, x2, y2 = np.vstack(segs).T
    r0 = np.searchsorted(fy, np.minimum(y1, y2), side="left")
    r1 = np.searchsorted(fy, np.maximum(y1, y2), side="left")
    seg = np.repeat(np.arange(len(x1)), r1 - r0)
    row = np.arange(len(seg)) - np.repeat(np.cumsum(r1 - r0) - (r1 - r0), r1 - r0) + r0[seg]
    so = np.argsort(row, kind="stable")
    seg, row = seg[so], row[so]
    out = np.zeros(len(R), dtype=bool)
    order = np.argsort(R, kind="stable")
    rows, starts = np.unique(R[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    lo = np.searchsorted(row, rows, side="left")
    hi = np.searchsorted(row, rows, side="right")
    for r, s0, e0, a, b in zip(rows, starts, ends, lo, hi):
        y = fy[r]
        hit = np.sort(seg[a:b])                            # _scanfill's order (no matter: sorted)
        xc = np.sort(x1[hit] + (y - y1[hit]) * (x2[hit] - x1[hit]) / (y2[hit] - y1[hit]))
        idx = order[s0:e0]
        out[idx] = np.searchsorted(xc, fx[C[idx]]) % 2 == 1
    return out


def _rasterize_cells(polys, fx: NDArray[np.float64], fy: NDArray[np.float64], S: int,
                     keys: NDArray[np.int64], nx: int) -> NDArray[np.int64]:
    """`_rasterize` on the sub-samples of the fine cells with flat indices `keys` (i nx + j,
    sorted) only, as (len(keys), S*S): the same rule (each polygon on the lattice points in its
    bounding box, overlaps to the later polygon)."""
    out = -np.ones((len(keys), S * S), dtype=np.int64)
    if len(polys) == 0 or len(keys) == 0:
        return out
    d = fx[1] - fx[0]
    for k, (bx0, by0, bx1, by1) in enumerate(shapely.bounds(polys)):
        j0 = max(int(np.floor((bx0 - fx[0]) / d)), 0)
        j1 = min(int(np.ceil((bx1 - fx[0]) / d)) + 1, len(fx))
        i0 = max(int(np.floor((by0 - fy[0]) / d)), 0)
        i1 = min(int(np.ceil((by1 - fy[0]) / d)) + 1, len(fy))
        if j1 <= j0 or i1 <= i0:
            continue
        X, Y = np.meshgrid(fx[j0:j1], fy[i0:i1])
        I, J = np.nonzero(shapely.contains_xy(polys[k], X, Y))
        I, J = I + i0, J + j0
        key = (I // S) * nx + J // S
        pos = np.minimum(np.searchsorted(keys, key), len(keys) - 1)
        ok = keys[pos] == key
        out[pos[ok], ((I % S) * S + J % S)[ok]] = k
    return out


def _pieces(geoms) -> NDArray[np.object_]:
    """Geometries as small pieces for an STRtree: lines split into their segments (a long
    street or ring's box covers half the block), polygons and points whole."""
    out = []
    for g in shapely.get_parts(np.array(geoms, dtype=object)):
        if g.geom_type in ("LineString", "LinearRing"):
            c = shapely.get_coordinates(g)
            out.extend(shapely.linestrings(np.stack([c[:-1], c[1:]], axis=1)))
        elif g.geom_type == "Polygon":
            out.append(g)
        elif g.geom_type == "Point":
            out.append(g)
        else:
            raise ValueError(f"unexpected geometry {g.geom_type}")
    return np.asarray(out, dtype=object)


NEAR_CHUNK = 100_000     # boxes per STRtree query: bounds the pair arrays, not the answer
NEAR_STAGES = (0.25, 1.0)    # the radii _near tries in turn, as fractions of its distance: speed
                             # only (the last is the distance itself)


def _near(tree: shapely.STRtree, boxes: NDArray[np.object_], dist: float) -> NDArray[np.bool_]:
    """Whether some piece in `tree` is within `dist` of each box: STRtree `dwithin` pairs,
    reduced to any per box. A box with a piece within dist / 4 is settled there, among few
    pairs, and only the rest are queried at `dist`, where a dense block's boxes would each pair
    with hundreds of pieces. 0.34 -- 0.76x query_nearest's time at d0 5 -- 20 (NOTES)."""
    near = np.zeros(len(boxes), dtype=bool)
    rest = np.arange(len(boxes))
    for frac in NEAR_STAGES:
        hit = np.zeros(len(rest), dtype=bool)
        for s0 in range(0, len(rest), NEAR_CHUNK):
            hit[s0 + tree.query(boxes[rest[s0:s0 + NEAR_CHUNK]], predicate="dwithin",
                                distance=frac * dist)[0]] = True
        near[rest[hit]] = True
        rest = rest[~hit]
    return near


class _Layout(NamedTuple):
    """CompositeGrid's cells before their attributes: the fine lattice (centres xs, ys; S x S
    sub-samples per cell at fx, fy), the coarse cells (level, first fine row, first fine
    column), and the fine cells' flat indices (i nx + j, sorted) with their inside sub-samples."""
    xs: NDArray[np.float64]
    ys: NDArray[np.float64]
    fx: NDArray[np.float64]
    fy: NDArray[np.float64]
    S: int
    coarse: list[tuple[int, NDArray[np.int64], NDArray[np.int64]]]
    key: NDArray[np.int64]
    isub: NDArray[np.bool_]


def _layout(boundary, footprints: NDArray[np.object_], streets, h: float, *, d0: float,
            smax: float, band_m: float, offset: tuple[float, float]) -> _Layout:
    """Grid.of's lattice and sub-samples, refined top-down: a cell of side h 2^l (l >= 1) stays
    whole when no footprint, street or block-edge piece is within d0 2^(l-1) of it (and its
    centre is inside the block: then all of it is), else it splits in four; the h cells are kept
    where some sub-sample is inside the block, as Grid's."""
    L = int(round(np.log2(smax / h)))
    if L < 1 or h * 2 ** L != smax:
        raise ValueError(f"smax {smax:g} is not h {h:g} x a power of two >= 2")
    if d0 <= band_m:
        raise ValueError(f"d0 {d0:g} must exceed the ground band {band_m:g} (no coarse "
                         "cell on the ground)")
    xs, ys = lattice(boundary, h, offset)
    ny, nx = len(ys), len(xs)
    S = 4
    d = h / S
    fx = xs[0] - h / 2 + (np.arange(nx * S) + 0.5) * d
    fy = ys[0] - h / 2 + (np.arange(ny * S) + 0.5) * d
    rings = shapely.get_rings(shapely.get_parts(boundary))
    tree = shapely.STRtree(np.concatenate([_pieces(footprints), _pieces(list(streets)),
                                           _pieces(rings)]))
    shapely.prepare(boundary)
    lx, ly = xs[0] - h / 2, ys[0] - h / 2              # the lattice's lower-left corner
    r = 1 << L
    ti, tj = np.meshgrid(np.arange(-(-ny // r)), np.arange(-(-nx // r)), indexing="ij")
    ti, tj = ti.ravel(), tj.ravel()
    coarse = []                                       # (level, i0, j0) per coarse cell
    for lev in range(L, 0, -1):
        r = 1 << lev
        boxes = shapely.box(lx + tj * r * h, ly + ti * r * h, lx + (tj + 1) * r * h,
                            ly + (ti + 1) * r * h)
        near = _near(tree, boxes, d0 * 2 ** (lev - 1))
        far = np.flatnonzero(~near)
        keep = far[shapely.contains_xy(boundary, lx + (tj[far] + 0.5) * r * h,
                                       ly + (ti[far] + 0.5) * r * h)]
        coarse.append((lev, ti[keep] * r, tj[keep] * r))
        ti = (2 * ti[near])[:, None] + np.array([0, 0, 1, 1])[None, :]
        tj = (2 * tj[near])[:, None] + np.array([0, 1, 0, 1])[None, :]
        ti, tj = ti.ravel(), tj.ravel()
        ok = (ti * (r // 2) < ny) & (tj * (r // 2) < nx)
        ti, tj = ti[ok], tj[ok]
    # the fine cells: Grid's, where some sub-sample is inside the block
    key = np.sort(ti * nx + tj)
    fi, fj = key // nx, key % nx
    sub_r = np.repeat(np.arange(S), S)                # sub-sample (row in cell) S + column
    sub_c = np.tile(np.arange(S), S)
    R = (fi * S)[:, None] + sub_r[None, :]
    C = (fj * S)[:, None] + sub_c[None, :]
    isub = _scanfill_at(boundary, fx, fy, R.ravel(), C.ravel()).reshape(-1, S * S)
    ins = isub.any(axis=1)
    return _Layout(xs=xs, ys=ys, fx=fx, fy=fy, S=S, coarse=coarse, key=key[ins], isub=isub[ins])


@dataclass(eq=False)
class CompositeGrid:
    """A block's cells at several sizes (AdaptiveMesh): the metric's h near footprints,
    streets and the block edge, h 2^l further out. Every attribute and method of Grid, over
    flat cells (n,) where Grid has the (ny, nx) raster: the fine cells are exactly the uniform
    grid's own (same lattice, sub-samples, footprint labels, ground, dist_b), in its row-major
    order, then the coarse cells by level and row-major. A coarse cell is wholly inside the
    block and open and off the ground (its sub-samples all inside, none footprint).

    The operator on it: a cell steps along each axis by its own size x v_k to the cell holding
    the point it lands on (a step between fine cells conducting by its open width, as on the
    uniform grid; one with a coarse end by 1: step_widths). A step to a cell of its own size is
    the uniform edge
    (forward only, m_k / |v_k|^2: the weight is scale-free); one to a LARGER cell (forward or
    back) carries the flux-consistent weight m_k (s / |v_k|) / d, the source's strip width over
    the projected distance between the centres; a step to a smaller cell is none (the smaller
    cells' own steps reach this one). Turning edges scale with each cell's area. Checked
    against the uniform h 0.5 grid (NOTES, "Local coarsening")."""
    h: float
    S: int
    x0: float                            # the fine lattice's first cell centre (Grid's x0, y0)
    y0: float
    shape: tuple[int, int]               # the fine lattice (ny, nx)
    level: NDArray[np.int64]             # (n,) cell side h 2^level
    i0: NDArray[np.int64]                # (n,) fine-lattice row, column of the cell's first fine
    j0: NDArray[np.int64]                #      cell
    inside: NDArray[np.bool_]            # (n,) all True
    building: NDArray[np.bool_]
    ground: NDArray[np.bool_]
    dist_b: NDArray[np.float64]          # metres from the centre to the nearest building cell's
    xy: NDArray[np.float64]              # (n, 2) cell centres
    isub: NDArray[np.bool_]              # (n, S*S)
    bsub: NDArray[np.bool_]
    fx: NDArray[np.float64]              # the sub-sample lattice (Grid.of's), for label_sub
    fy: NDArray[np.float64]
    cache: dict = dataclasses.field(default_factory=dict, compare=False, repr=False)

    @property
    def ff0(self) -> NDArray[np.float64]:
        return (self.isub & ~self.bsub).mean(axis=-1)

    # The maps between a field (n,) and the fine-lattice raster (ny, nx) an along-conductance
    # scans: a coarse cell covers 4^level pixels, a fine cell one, a pixel outside every cell
    # (outside the block) is 0. Per level, chunked so a gather of PIXEL_CHUNK pixels is the
    # largest transient.
    PIXEL_CHUNK = 1 << 24

    def _by_level(self, xp) -> list:
        """[(level, cell ids, rows (m, 2^level), cols (m, 2^level))] on device `xp`."""
        key = ("by_level", xp.__name__)
        if key not in self.cache:
            out = []
            for lev in np.unique(self.level).tolist():
                ids = np.flatnonzero(self.level == lev)
                ar = np.arange(1 << lev)
                for c0 in range(0, len(ids), max(1, self.PIXEL_CHUNK >> (2 * lev))):
                    c = ids[c0:c0 + max(1, self.PIXEL_CHUNK >> (2 * lev))]
                    out.append((lev, xp.asarray(c), xp.asarray(self.i0[c][:, None] + ar),
                                xp.asarray(self.j0[c][:, None] + ar)))
            self.cache[key] = out
        return self.cache[key]

    def paint(self, field, xp):
        """The raster of a field: each cell's value over its pixels."""
        R = xp.zeros(self.shape)
        for _lev, c, r, q in self._by_level(xp):
            R[r[:, :, None], q[:, None, :]] = field[c][:, None, None]
        return R

    def mean(self, R, xp):
        """(..., ny, nx) raster -> (..., n): each cell's mean over its pixels."""
        out = xp.zeros((*R.shape[:-2], len(self.level)))
        for _lev, c, r, q in self._by_level(xp):
            out[..., c] = R[..., r[:, :, None], q[:, None, :]].mean(axis=(-2, -1))
        return out

    def mean_T(self, A, xp):
        """The adjoint of `mean`: (..., n) -> (..., ny, nx)."""
        R = xp.zeros((*A.shape[:-1], *self.shape))
        for lev, c, r, q in self._by_level(xp):
            share = A[..., c] / float(1 << (2 * lev))
            R[..., r[:, :, None], q[:, None, :]] = share[..., None, None]
        return R

    def gather(self, G, xp):
        """The adjoint of `paint`: (ny, nx) -> (n,), each cell's sum over its pixels."""
        out = xp.zeros(len(self.level))
        for _lev, c, r, q in self._by_level(xp):
            out[c] = G[r[:, :, None], q[:, None, :]].sum(axis=(-2, -1))
        return out

    @property
    def size(self) -> NDArray[np.float64]:
        """(n,) each cell's side in metres."""
        return (1 << self.level) * self.h

    @classmethod
    def from_layout(cls, lay: _Layout, footprints, streets, h: float, *,
                    band_m: float) -> CompositeGrid:
        """`lay`'s cells, with Grid's attributes on the fine ones (footprint sub-samples and
        labels, ground within band_m of a street, dist_b)."""
        footprints = np.asarray(footprints)
        xs, ys, fx, fy, S = lay.xs, lay.ys, lay.fx, lay.fy, lay.S
        ny, nx = len(ys), len(xs)
        lx, ly = xs[0] - h / 2, ys[0] - h / 2              # the lattice's lower-left corner
        key, isub_f = lay.key, lay.isub
        fi, fj = key // nx, key % nx
        bsub_f = isub_f & (_rasterize_cells(footprints, fx, fy, S, key, nx) >= 0)
        building_f = bsub_f.mean(axis=-1) > 0.5
        st = shapely.union_all(np.asarray(streets)).buffer(band_m)
        shapely.prepare(st)
        ground_f = shapely.contains_xy(st, xs[fj], ys[fi])
        lev_c, i0_c, j0_c = _flat(lay)
        nf, nc = len(key), len(lev_c)
        level = np.concatenate([np.zeros(nf, dtype=np.int64), lev_c.astype(np.int64)])
        i0 = np.concatenate([fi, i0_c]).astype(np.int64)
        j0 = np.concatenate([fj, j0_c]).astype(np.int64)
        rc = (1 << level[nf:]).astype(float)
        xy = np.concatenate([np.stack([xs[fj], ys[fi]], axis=1),
                             np.stack([lx + (j0_c + rc / 2) * h, ly + (i0_c + rc / 2) * h],
                                      axis=1)])
        # dist_b: Grid's EDT (centre to the nearest building cell's centre, in cells, x h)
        bc = np.stack([fi[building_f], fj[building_f]], axis=1).astype(float)
        centre = np.stack([i0 + ((1 << level) - 1) / 2, j0 + ((1 << level) - 1) / 2], axis=1)
        dist_b = (cKDTree(bc).query(centre)[0] * h if len(bc) else np.full(nf + nc, np.inf))
        return cls(h=h, S=S, x0=float(xs[0]), y0=float(ys[0]), shape=(ny, nx), level=level,
                   i0=i0, j0=j0, inside=np.ones(nf + nc, dtype=bool),
                   building=np.concatenate([building_f, np.zeros(nc, dtype=bool)]),
                   ground=np.concatenate([ground_f, np.zeros(nc, dtype=bool)]),
                   dist_b=dist_b, xy=xy,
                   isub=np.concatenate([isub_f, np.ones((nc, S * S), dtype=bool)]),
                   bsub=np.concatenate([bsub_f, np.zeros((nc, S * S), dtype=bool)]),
                   fx=fx, fy=fy)

    def _sub_xy(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        t = (np.arange(self.S) + 0.5) / self.S - 0.5
        ox, oy = np.meshgrid(t, t)
        s = self.size[:, None]
        return (self.xy[:, 0][:, None] + ox.ravel()[None, :] * s,
                self.xy[:, 1][:, None] + oy.ravel()[None, :] * s)

    def sub_of(self, geom) -> NDArray[np.bool_]:
        """(n, S*S): sub-samples inside `geom` (a coarse cell's spread over its area)."""
        out = np.zeros_like(self.isub)
        if geom is None or geom.is_empty:
            return out
        shapely.prepare(geom)
        X, Y = self._sub_xy()
        out[self.isub] = shapely.contains_xy(geom, X[self.isub], Y[self.isub])
        return out

    def label_sub(self, polys) -> NDArray[np.int64]:
        """(n, S*S): Grid.label_sub on the fine cells; -1 on the coarse ones (no footprint)."""
        ny, nx = self.shape
        fine = np.flatnonzero(self.level == 0)
        lab = -np.ones(self.isub.shape, dtype=np.int64)
        lab[fine] = _rasterize_cells(np.asarray(polys), self.fx, self.fy, self.S,
                                     self.i0[fine] * nx + self.j0[fine], nx)
        return np.where(self.bsub, lab, -1)

    def mask_of(self, geom) -> NDArray[np.bool_]:
        out = np.zeros_like(self.inside)
        if geom is None or geom.is_empty:
            return out
        shapely.prepare(geom)
        out[:] = shapely.contains_xy(geom, self.xy[:, 0], self.xy[:, 1])
        return out

    def cells_near(self, geom, dist: float) -> NDArray[np.int64]:
        """Grid.cells_near: the cells holding a fine-lattice point of `geom`'s bounding box
        padded by dist + h (every cell whose centre is in the box holds the points beside its
        centre), tested."""
        ny, nx = self.shape
        x0, y0, x1, y1 = geom.bounds
        pad = dist + self.h
        j0 = max(int(np.floor((x0 - pad - self.x0) / self.h)), 0)
        j1 = min(int(np.ceil((x1 + pad - self.x0) / self.h)) + 1, nx)
        i0 = max(int(np.floor((y0 - pad - self.y0) / self.h)), 0)
        i1 = min(int(np.ceil((y1 + pad - self.y0) / self.h)) + 1, ny)
        py, px = np.meshgrid(np.arange(i0, i1, dtype=float), np.arange(j0, j1, dtype=float),
                             indexing="ij")
        cells = np.unique(self._locate(py.ravel(), px.ravel()))
        cells = cells[cells >= 0]
        ok = shapely.dwithin(geom, shapely.points(self.xy[cells, 0], self.xy[cells, 1]), dist)
        return cells[ok]

    def _locate(self, py: NDArray[np.float64], px: NDArray[np.float64]) -> NDArray[np.int64]:
        """The cell holding each fine-lattice point (row py, column px; fine cell (i, j)'s centre
        at (i, j)), -1 where none."""
        if "locate" not in self.cache:
            nx = self.shape[1]
            index = []
            for lev in range(int(self.level.max()) + 1):
                ids = np.flatnonzero(self.level == lev)
                k = (self.i0[ids] >> lev) * ((nx >> lev) + 1) + (self.j0[ids] >> lev)
                o = np.argsort(k)
                index.append((k[o], ids[o]))
            self.cache["locate"] = index
        ny, nx = self.shape
        iy = np.floor(py + 0.5).astype(np.int64)
        ix = np.floor(px + 0.5).astype(np.int64)
        out = -np.ones(len(iy), dtype=np.int64)
        todo = np.flatnonzero((iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx))
        for lev, (keys, ids) in enumerate(self.cache["locate"]):
            if len(todo) == 0 or len(keys) == 0:
                continue
            k = (iy[todo] >> lev) * ((nx >> lev) + 1) + (ix[todo] >> lev)
            pos = np.minimum(np.searchsorted(keys, k), len(keys) - 1)
            hit = keys[pos] == k
            out[todo[hit]] = ids[pos[hit]]
            todo = todo[~hit]
        return out

    def pattern(self, K: int, xp) -> Pattern:
        """The mesh's Pattern for K axes on device `xp`, built once (the class docstring's
        steps)."""
        key = ("pattern", K, xp.__name__)
        if key not in self.cache:
            v, _th, m, _gap = axes(K)
            r = 1 << self.level                            # fine cells per side
            cy = self.i0 + (r - 1) / 2
            cx = self.j0 + (r - 1) / 2
            along = []
            for k in range(K):
                dx, dy = int(v[k, 0]), int(v[k, 1])
                lc = _line_cells(dx, dy)
                parts = []
                for sign in (1, -1):
                    t = self._locate(cy + sign * r * dy, cx + sign * r * dx)
                    s = np.flatnonzero(t >= 0)
                    t = t[s]
                    keep = (r[t] > r[s]) | ((r[t] == r[s]) & (sign == 1))
                    s, t = s[keep], t[keep]
                    line = (np.stack([self._locate(cy[s] + sign * r[s] * iy,
                                                   cx[s] + sign * r[s] * ix) for ix, iy in lc])
                            if lc else np.zeros((0, len(s)), dtype=np.int64))
                    ok = (line >= 0).all(axis=0)
                    s, t, line = s[ok], t[ok], line[:, ok]
                    # projected distance between the centres, x |v| (fine cells)
                    proj = sign * ((cx[t] - cx[s]) * dx + (cy[t] - cy[s]) * dy)
                    if not (proj > 0).all():
                        raise ValueError(f"axis {k}: a step to a larger cell whose centre is "
                                         "not ahead (the mesh grades too fast: raise d0)")
                    wk = np.where(r[t] == r[s], m[k] / float(dx * dx + dy * dy),
                                  m[k] * r[s] / proj)
                    a, b = (s, t) if sign == 1 else (t, s)
                    parts.append((a, b, line, wk))
                along.append((xp.asarray(np.concatenate([q[0] for q in parts])),
                              xp.asarray(np.concatenate([q[1] for q in parts])),
                              xp.asarray(np.concatenate([q[2] for q in parts], axis=1)),
                              xp.asarray(np.concatenate([q[3] for q in parts]))))
            self.cache[key] = Pattern(along=along, cells=xp.arange(len(self.level)))
        return self.cache[key]

    def track_axes(self, K: int, xp) -> list[_AxisTracks]:
        """The steps' tracks on this mesh (_AxisTracks), per axis: a step between two fine cells
        reads the tracks, a slot's cell the one holding the fine-lattice position at its offset
        from the step's first cell (the pattern's own a, b and line; others located, the dummy
        cell where none); a step with a coarse end conducts by 1 (`coarse`)."""
        v, _th, _m, _gap = axes(K)
        pat = self.pattern(K, np)                          # on the host: _locate is numpy
        n, S2 = len(self.level), self.S * self.S
        out = []
        for k in range(K):
            a, b, line, _wk = pat.along[k]
            tr = axis_tracks(int(v[k, 0]), int(v[k, 1]), self.S)
            cells = [a, b, *list(line)]
            for ox, oy in tr.cells[len(cells):]:
                q = self._locate((self.i0[a] + oy).astype(float), (self.j0[a] + ox).astype(float))
                cells.append(np.where(q >= 0, q, n))
            coarse = (self.level[a] > 0) | (self.level[b] > 0)
            out.append(_AxisTracks(base=[xp.asarray(q * S2) for q in cells], tracks=tr.tracks,
                                   coarse=xp.asarray(coarse)))
        return out

    def cell_area(self, xp, free):
        """The area of each of the `free` cells (device)."""
        key = ("area", xp.__name__)
        if key not in self.cache:
            s = self.size
            self.cache[key] = xp.asarray(s * s)
        return self.cache[key][free]


class MeshSpec(Protocol):
    """How a block is discretized: resolved once, where the scorer is built, and part of the
    run's name. `build` gets the run's physics (`p`, the population): a mesh refined for the
    score (common.GoalMesh) solves on the way; the others use only the block's geometry."""

    @property
    def name(self) -> str: ...

    @property
    def suffix(self) -> str:
        """Its part of a SIMP plan's name (relax.Plan; the metric's uniform h 0.5 unnamed)."""
        ...

    def build(self, block, p: Params, population) -> Grid | CompositeGrid: ...

    def cells(self, block) -> float:
        """How many cells it has on a block (what a run's memory scales with), without a
        solve."""
        ...

    def pixels(self, block) -> int:
        """Its fine lattice's pixels on a block: the raster a scanning along-conductance runs
        on (SoftSightline: what its memory scales with)."""
        ...


def _geometry(block) -> tuple:
    """A block's boundary, footprints and streets, as the grids take them."""
    return (block.boundary, np.asarray(block.buildings.outlines), list(block.streets.geometry))


@dataclass(frozen=True)
class UniformMesh:
    """One cell size everywhere (Grid)."""
    h: float
    offset: tuple[float, float]

    @property
    def name(self) -> str:
        return f"h{self.h:g}"

    @property
    def suffix(self) -> str:
        return "" if self.h == 0.5 else f".h{self.h:g}"

    def build(self, block, p: Params, population) -> Grid:
        return Grid.of(*_geometry(block), self.h, offset=self.offset)

    def cells(self, block) -> float:
        return block.boundary.area / self.h ** 2

    def pixels(self, block) -> int:
        return _pixels(block, self.h, self.offset)


@dataclass(frozen=True)
class AdaptiveMesh:
    """h within d0 of every footprint, street and the block edge, h 2^l beyond d0 2^(l-1), up
    to smax (CompositeGrid)."""
    h: float
    d0: float
    smax: float
    offset: tuple[float, float]

    @property
    def name(self) -> str:
        return f"h{self.h:g}a{self.d0:g}x{self.smax:g}"

    @property
    def suffix(self) -> str:
        return ("" if self.h == 0.5 else f".h{self.h:g}") + f".a{self.d0:g}x{self.smax:g}"

    def layout(self, block) -> _Layout:
        return _layout(*_geometry(block), self.h, d0=self.d0, smax=self.smax, band_m=BAND_M,
                       offset=self.offset)

    def build(self, block, p: Params, population) -> CompositeGrid:
        return CompositeGrid.from_layout(self.layout(block), *_geometry(block)[1:], self.h,
                                         band_m=BAND_M)

    def cells(self, block) -> float:
        return float(_cells_of(self.layout(block)))

    def pixels(self, block) -> int:
        return _pixels(block, self.h, self.offset)


def _pixels(block, h: float, offset: tuple[float, float]) -> int:
    xs, ys = lattice(block.boundary, h, offset)
    return len(xs) * len(ys)


def _cells_of(lay: _Layout) -> int:
    return len(lay.key) + sum(len(i) for _, i, _ in lay.coarse)


def _flat(lay: _Layout) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """The coarse cells' (level, i0, j0) in `lay.coarse`'s order: CompositeGrid's order after
    its fine cells."""
    if not lay.coarse:
        e = np.zeros(0, dtype=np.int64)
        return e, e, e
    return (np.concatenate([np.full(len(i), lv, dtype=np.int64) for lv, i, _ in lay.coarse]),
            np.concatenate([i for _, i, _ in lay.coarse]).astype(np.int64),
            np.concatenate([j for _, _, j in lay.coarse]).astype(np.int64))


def split_cells(lay: _Layout, mask: NDArray[np.bool_]) -> _Layout:
    """`lay` with the coarse cells flagged in `mask` (`_flat`'s order) split in four. A coarse
    cell is wholly inside the block, so a child of side h is a fine cell with every sub-sample
    inside."""
    lev, i0, j0 = _flat(lay)
    sl, si, sj = lev[mask], i0[mask], j0[mask]
    half = np.repeat(1 << (sl - 1), 4)
    cl = np.repeat(sl - 1, 4)
    ci = np.repeat(si, 4) + np.tile([0, 0, 1, 1], len(si)) * half
    cj = np.repeat(sj, 4) + np.tile([0, 1, 0, 1], len(sj)) * half
    co = cl >= 1
    lev, i0, j0 = (np.concatenate([lev[~mask], cl[co]]), np.concatenate([i0[~mask], ci[co]]),
                   np.concatenate([j0[~mask], cj[co]]))
    coarse = [(int(lv), i0[lev == lv], j0[lev == lv])
              for lv in sorted(set(lev.tolist()), reverse=True)]
    key = np.concatenate([lay.key, ci[~co] * len(lay.xs) + cj[~co]])
    isub = np.concatenate([lay.isub, np.ones(((~co).sum(), lay.isub.shape[1]), dtype=bool)])
    o = np.argsort(key, kind="stable")
    return lay._replace(coarse=coarse, key=key[o], isub=isub[o])


def _leaf_level(lay: _Layout, py: NDArray[np.int64], px: NDArray[np.int64]) -> NDArray[np.int64]:
    """The level of the cell holding each fine-lattice point (row py, column px), -1 where none
    (outside the block)."""
    nx = len(lay.xs)
    out = -np.ones(len(py), dtype=np.int64)
    inb = (py >= 0) & (px >= 0) & (py < len(lay.ys)) & (px < nx)
    k = py * nx + px
    pos = np.minimum(np.searchsorted(lay.key, k), max(len(lay.key) - 1, 0))
    out[inb & (lay.key[pos] == k)] = 0
    lev, i0, j0 = _flat(lay)
    for lv in np.unique(lev):
        sel = lev == lv
        keys = np.sort((i0[sel] >> lv) * (nx + 1) + (j0[sel] >> lv))
        kk = (py >> lv) * (nx + 1) + (px >> lv)
        pos = np.minimum(np.searchsorted(keys, kk), len(keys) - 1)
        out[inb & (out < 0) & (keys[pos] == kk)] = lv
    return out


def balance(lay: _Layout) -> _Layout:
    """Split coarse cells until none has a neighbour (across an edge or a corner) under half its
    size, as the distance rule's meshes already are (CompositeGrid's interface weights assume
    sizes within 2x). A level-l cell's neighbour of level <= l - 2 is caught by probing just
    outside it every 2^(l-2) fine cells."""
    while True:
        lev, i0, j0 = _flat(lay)
        cand = np.flatnonzero(lev >= 2)
        bad = np.zeros(len(lev), dtype=bool)
        for s0 in range(0, len(cand), BALANCE_CHUNK):
            c = cand[s0:s0 + BALANCE_CHUNK]
            lv, i, j = lev[c], i0[c], j0[c]
            n, q = 1 << lv, 1 << (lv - 2)
            probes = [(i - 1, j - 1), (i - 1, j + n), (i + n, j - 1), (i + n, j + n)]
            for k in range(4):
                probes += [(i - 1, j + k * q), (i + n, j + k * q), (i + k * q, j - 1),
                           (i + k * q, j + n)]
            ll = _leaf_level(lay, np.concatenate([py for py, _ in probes]),
                             np.concatenate([px for _, px in probes]))
            own = np.tile(np.arange(len(c)), len(probes))
            hit = (ll >= 0) & (ll <= lv[own] - 2)
            bad[c[np.unique(own[hit])]] = True
        if not bad.any():
            return lay
        lay = split_cells(lay, bad)


BALANCE_CHUNK = 200_000      # cells probed at once: bounds memory, not the answer


@dataclass(frozen=True, eq=False)
class Placement:
    """Where each home injects: the metric's demand (owner 2026-10-11, "B SGTM": the placement
    fix, and the enclosed homes scored apart; insulator_report.md, "The door link", Task 1, and
    "Uniform walls"). Only cells `allowed` (with a path to the street at baseline) take demand,
    so a ring partly in a sealed nook puts all of it on the open side, and freeing space only
    ever adds conductance. A home's RING is the allowed cells within ring_m of a building
    (dist_b) whose nearest building it is; it injects its population uniformly over them. A
    home with no ring cell is SHADOWED when free cells within ring_m of its own outline reach the
    street (the nearest-building rule gave them to its neighbours): it injects uniformly over
    those, whoever's ring they are in, so a cell can take several homes' demand. The rest are
    ENCLOSED: no free cell within ring_m of the outline reaches the street; they inject nothing
    and are out of J, and a clearing reports the population it leaves enclosed (`reached`),
    scored before J. A population w injects S w, and home j's escape time is (S^T u_cell)_j,
    the injection-weighted mean potential over its cells (common.Scorer.home_u_of)."""
    S: sp.csr_matrix                 # (grid cells flat, homes): home j's share of its injection
                                     # per cell; a live home's column sums to 1, an enclosed
                                     # home's is empty
    enclosed: NDArray[np.bool_]      # per home: no free cell within ring_m of its outline
                                     # reaches the street
    near: sp.csr_matrix              # (grid cells flat, homes), 0/1: per ENCLOSED home, the
                                     # inside cells whose centres lie within ring_m of its outline,
                                     # its own footprint's included (a clearing can open a covered
                                     # cell); empty columns for the live homes

    @classmethod
    def of(cls, grid: Grid | CompositeGrid, footprints, allowed: NDArray[np.bool_], *,
           ring_m: float) -> Placement:
        """The ring by each allowed cell's nearest building (STRtree.query_nearest); the outline
        by exact polygon distance (a centre inside the footprint is at 0), tested only near the
        few homes with no ring cell (grid.cells_near)."""
        polys = np.asarray(footprints)
        n, ncell = len(polys), grid.inside.size
        ok = np.asarray(allowed).ravel()
        at = np.flatnonzero(ok & (grid.dist_b.ravel() <= ring_m + 1e-9))
        xy = grid.xy.reshape(-1, 2)[at]
        idx = shapely.STRtree(polys).query_nearest(shapely.points(xy[:, 0], xy[:, 1]),
                                                   all_matches=False)
        owner = np.full(len(at), -1)
        owner[idx[0]] = idx[1]
        isub, bsub = grid.isub.reshape(ncell, -1), grid.bsub.reshape(ncell, -1)
        enclosed = np.zeros(n, dtype=bool)
        cells, homes = [at], [owner]
        near_cells, near_homes = [np.zeros(0, dtype=np.int64)], [np.zeros(0, dtype=np.int64)]
        for j in np.flatnonzero(np.bincount(owner, minlength=n) == 0):
            c = grid.cells_near(polys[j], ring_m + 1e-9)
            door = c[(isub[c] & ~bsub[c]).any(axis=1) & ok[c]]     # free (ff0 > 0), allowed
            if len(door):                       # shadowed: on its open side
                cells.append(door)
                homes.append(np.full(len(door), j))
            else:
                enclosed[j] = True
                near_cells.append(c)
                near_homes.append(np.full(len(c), j))
        c, h = np.concatenate(cells), np.concatenate(homes)
        k = np.bincount(h, minlength=n).astype(float)
        nc, nh = np.concatenate(near_cells), np.concatenate(near_homes)
        return cls(S=sp.csr_matrix((1.0 / k[h], (c, h)), shape=(ncell, n)), enclosed=enclosed,
                   near=sp.csr_matrix((np.ones(len(nc)), (nc, nh)), shape=(ncell, n)))

    def field(self, w: NDArray[np.float64], shape) -> NDArray[np.float64]:
        """The injection of population w (per home), S w, in the grid's shape."""
        return (self.S @ w).reshape(shape)

    def home_u(self, u_cell) -> NDArray[np.float64]:
        """Per home, S^T u_cell: from per-cell potentials (flat; columns for several fields)."""
        return np.asarray(self.S.T @ u_cell)

    def reached(self, reach: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Per home: enclosed, and some cell within ring_m of its outline in `reach` (open and
        street-connected in a clearing's field: `grounded`). The live homes' columns of `near`
        are empty: never."""
        nz = self.near.tocoo()
        out = np.zeros(len(self.enclosed), dtype=bool)
        out[nz.col[np.asarray(reach).ravel()[nz.row]]] = True
        return out


@njit(cache=True)
def _run_count(clear, dx, dy, offs):
    """Per cell: the number of clear cells in the straight lattice run through it along
    (dx, dy), a step counting only when its far end and every cell it crosses (`offs`, the
    offsets `_line_cells` gives) are clear. Requires dy > 0, or dy == 0 and dx > 0, so that
    visiting rows (then columns) in decreasing order settles each successor first."""
    ny, nx = clear.shape
    fwd = np.zeros((ny, nx), dtype=np.int64)
    bwd = np.zeros((ny, nx), dtype=np.int64)
    for r in range(ny - 1, -1, -1):
        for c in range(nx - 1, -1, -1):
            if clear[r, c]:
                fwd[r, c] = 1
                if _step_ok(clear, r, c, dx, dy, offs):
                    fwd[r, c] += fwd[r + dy, c + dx]
    for r in range(ny):
        for c in range(nx):
            if clear[r, c]:
                bwd[r, c] = 1
                if _step_ok(clear, r, c, -dx, -dy, -offs):
                    bwd[r, c] += bwd[r - dy, c - dx]
    return fwd + bwd - 1


@njit(cache=True)
def _step_ok(clear, r, c, dx, dy, offs):
    ny, nx = clear.shape
    r2, c2 = r + dy, c + dx
    if not (0 <= r2 < ny and 0 <= c2 < nx and clear[r2, c2]):
        return False
    for i in range(offs.shape[0]):
        r3, c3 = r + offs[i, 1], c + offs[i, 0]
        if not (0 <= r3 < ny and 0 <= c3 < nx and clear[r3, c3]):
            return False
    return True


def directions(nmax: int) -> list[tuple[int, int]]:
    """Primitive lattice directions (dx, dy) with max(|dx|, |dy|) <= nmax, one per line
    (dy > 0, or dy == 0 and dx > 0)."""
    return [(dx, dy) for dy in range(0, nmax + 1) for dx in range(-nmax, nmax + 1)
            if (dy > 0 or dx > 0) and np.gcd(dx, dy) == 1]


class AlongConductance(Protocol):
    """Per-heading multipliers on the along-axis edges: layers(grid, open_, K)[k] is layer k's
    factor per cell (>= 1, non-decreasing as space is freed, so monotonicity survives) or the
    scalar 1.0. turning(grid, open_, K)[k] is the same for the turning edges (x, k)-(x, k+1):
    the scalar 1.0 but for TurnBoosted. vjp(grid, open_, K, A) is the gradient with respect to
    open_ of sum_k sum_x A[k, x] layers[k][x] (what clearing a cell does to the factors
    elsewhere). open_, A and the results are the grid's fields; a conductance that scans lines
    does so on the fine lattice's raster (grid.paint), a cell's factor its mean over its
    pixels."""
    name: str

    def layers(self, grid: Grid | CompositeGrid, open_,
               K: int) -> list[float] | NDArray[np.float64]: ...

    def turning(self, grid: Grid | CompositeGrid, open_,
                K: int) -> list[float] | NDArray[np.float64]: ...

    def vjp(self, grid: Grid | CompositeGrid, open_, K: int, A) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class Uniform:
    """Every open cell conducts alike (the original model)."""
    name: str = "uni"

    def layers(self, grid, open_, K):
        return [1.0] * K

    def turning(self, grid, open_, K):
        return [1.0] * K

    def vjp(self, grid, open_, K, A):
        return np.zeros(open_.shape)


@dataclass(frozen=True)
class Sightline:
    """Layer k conducts better where long straight clear runs exist at headings near k. Over the
    fine lattice directions d (max(|dx|, |dy|) <= nmax), R_d is the capped clear run through the
    cell, as a fraction of r_max; layer k's factor is 1 + beta x the mean of R_d weighted by a
    triangle in angle, width one axis gap either side of axis k. A cell is clear when at least
    `clear_frac` of it is open. Integrated across a channel this measures the straight segments
    it holds: a corridor of width W admits headings within ~W/L of its axis at every offset, so
    its conductance grows faster than W."""
    beta: float
    r_max_m: float = 50.0
    nmax: int = 6
    clear_frac: float = 0.5

    @property
    def name(self) -> str:
        return f"sl{self.beta:g}"

    def runs(self, open_, h) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
        """(angle of each fine direction in [0, pi), capped run fraction (n_dir, ny, nx))."""
        clear = open_ >= self.clear_frac
        dirs = directions(self.nmax)
        ang = np.array([np.arctan2(dy, dx) % np.pi for dx, dy in dirs])
        R = np.empty((len(dirs), *open_.shape), dtype=np.float32)
        for i, (dx, dy) in enumerate(dirs):
            offs = np.array(_line_cells(dx, dy), dtype=np.int64).reshape(-1, 2)
            run = _run_count(clear, dx, dy, offs) * (np.hypot(dx, dy) * h)
            R[i] = np.minimum(run, self.r_max_m) / self.r_max_m
        return ang, R

    def layers(self, grid, open_, K):
        ang, R = self.runs(grid.paint(open_, np), grid.h)
        W = _angle_weights(K, ang)
        return grid.mean((1.0 + self.beta * np.tensordot(W, R, axes=1)).astype(np.float32), np)

    def turning(self, grid, open_, K):
        return [1.0] * K

    def vjp(self, grid, open_, K, A):
        """Hard runs are piecewise constant in open_: zero almost everywhere."""
        return np.zeros(open_.shape)


def _angle_weights(K: int, ang: NDArray[np.float64]) -> NDArray[np.float64]:
    """(K, n_dir): direction d's share in layer k, a triangle in angle one axis gap either side
    of axis k, normalised over d per layer."""
    _v, th, _m, gap = axes(K)
    W = np.zeros((K, len(ang)))
    for k in range(K):
        d = (ang - th[k] + np.pi / 2) % np.pi - np.pi / 2     # signed, in [-pi/2, pi/2)
        half = np.where(d >= 0, gap[k], np.roll(gap, 1)[k])   # to the next / previous axis
        wt = np.clip(1.0 - np.abs(d) / half, 0.0, None)
        W[k] = wt / wt.sum()
    return W


@njit(cache=True)
def _step_t(open_, r, c, sy, dx, dy, offs, s, kappa, m):
    """The transmittance of the step from (r, c) to (r + sy dy, c + sy dx) (sy = +-1; its far
    end on the grid): optical depth kappa s mean(1 - open) over its far end and the cells it
    crosses, `offs` (outside the grid counts closed)."""
    ny, nx = open_.shape
    tau = 1.0 - open_[r + sy * dy, c + sy * dx]
    for i in range(offs.shape[0]):
        r3, c3 = r + sy * offs[i, 1], c + sy * offs[i, 0]
        tau += 1.0 - open_[r3, c3] if (0 <= r3 < ny and 0 <= c3 < nx) else 1.0
    return np.exp(-kappa * s / m * tau)


@njit(cache=True)
def _soft_fb(open_, dx, dy, offs, s, kappa):
    """Expected free path forward (F) and back (B) along (dx, dy) from each cell, the ray's
    intensity falling as exp(-kappa x the closed length it crosses): each step of length s takes
    optical depth kappa s mean(1 - open) over the cells it enters (`_step_t`)."""
    ny, nx = open_.shape
    m = offs.shape[0] + 1
    F = np.zeros((ny, nx))
    B = np.zeros((ny, nx))
    for r in range(ny - 1, -1, -1):
        for c in range(nx - 1, -1, -1):
            r2, c2 = r + dy, c + dx
            if 0 <= r2 < ny and 0 <= c2 < nx:
                F[r, c] = _step_t(open_, r, c, 1, dx, dy, offs, s, kappa, m) * (s + F[r2, c2])
    for r in range(ny):
        for c in range(nx):
            r2, c2 = r - dy, c - dx
            if 0 <= r2 < ny and 0 <= c2 < nx:
                B[r, c] = _step_t(open_, r, c, -1, dx, dy, offs, s, kappa, m) * (s + B[r2, c2])
    return F, B


@njit(cache=True)
def _soft_vjp(open_, dx, dy, offs, s, kappa, F, B, G, out):
    """Add to `out` the gradient over open_ of sum_x G[x] (F[x] + B[x]): reverse-mode through
    the two scans (each cotangent flows back along the ray, damped by the transmittance, which
    is recomputed rather than stored: two rasters fewer)."""
    ny, nx = F.shape
    m = offs.shape[0] + 1
    bar = np.zeros((ny, nx))
    # forward rays: F(p) = T(p) (s + F(x)) with x = p + v, so visit p before x; p's cotangent
    # is G[p] + T(p - v) bar[p - v], pushed to p by p - v
    for r in range(ny):
        for c in range(nx):
            rp, cp = r - dy, c - dx
            b = bar[r, c] if (0 <= rp < ny and 0 <= cp < nx) else G[r, c]
            r2, c2 = r + dy, c + dx
            if 0 <= r2 < ny and 0 <= c2 < nx:
                T = _step_t(open_, r, c, 1, dx, dy, offs, s, kappa, m)
                bar[r2, c2] = G[r2, c2] + T * b
                if b != 0.0:
                    coef = b * (s + F[r2, c2]) * T * kappa * s / m
                    out[r2, c2] += coef
                    for i in range(offs.shape[0]):
                        r3, c3 = r + offs[i, 1], c + offs[i, 0]
                        if 0 <= r3 < ny and 0 <= c3 < nx:
                            out[r3, c3] += coef
    for r in range(ny - 1, -1, -1):
        for c in range(nx - 1, -1, -1):
            rp, cp = r + dy, c + dx
            b = bar[r, c] if (0 <= rp < ny and 0 <= cp < nx) else G[r, c]
            r2, c2 = r - dy, c - dx
            if 0 <= r2 < ny and 0 <= c2 < nx:
                T = _step_t(open_, r, c, -1, dx, dy, offs, s, kappa, m)
                bar[r2, c2] = G[r2, c2] + T * b
                if b != 0.0:
                    coef = b * (s + B[r2, c2]) * T * kappa * s / m
                    out[r2, c2] += coef
                    for i in range(offs.shape[0]):
                        r3, c3 = r - offs[i, 1], c - offs[i, 0]
                        if 0 <= r3 < ny and 0 <= c3 < nx:
                            out[r3, c3] += coef


class Scans(Protocol):
    """Where SoftSightline's 1-D scans run. `xp` is the array module (numpy or cupy); fb returns
    (F, B) and vjp_add accumulates into `out`, all `xp` arrays; response and response_weight are
    SoftSightline's g(F + B) and beta g'(F + B) M, elementwise (on the GPU fused and in place:
    a raster each, not one per operation); to_host / to_dev move arrays across."""
    xp: object

    def fb(self, open_, dx: int, dy: int, offs, s: float, kappa: float): ...

    def vjp_add(self, open_, dx: int, dy: int, offs, s: float, kappa: float, F, B, G,
                out) -> None: ...

    def response(self, F, B, r0: float, hill: float): ...

    def response_weight(self, F, B, M, beta: float, r0: float, hill: float): ...

    def to_dev(self, a): ...

    def to_host(self, a) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class CpuScans:
    """numba, one core."""
    xp: object = np

    def fb(self, open_, dx, dy, offs, s, kappa):
        return _soft_fb(open_, dx, dy, offs, s, kappa)

    def vjp_add(self, open_, dx, dy, offs, s, kappa, F, B, G, out):
        _soft_vjp(open_, dx, dy, offs, s, kappa, F, B, G, out)

    def response(self, F, B, r0, hill):
        x = ((F + B) / r0) ** hill
        return x / (1.0 + x)

    def response_weight(self, F, B, M, beta, r0, hill):
        R = F + B
        x = (R / r0) ** hill
        return beta * (hill * x / ((R + 1e-12) * (1.0 + x) ** 2)) * M

    def to_dev(self, a):
        return np.asarray(a)

    def to_host(self, a):
        return np.asarray(a)


_SCAN_CU = r"""
__device__ inline double opac(const double* o, int ny, int nx, int r, int c) {
    return (r >= 0 && r < ny && c >= 0 && c < nx) ? 1.0 - o[r * nx + c] : 1.0;
}
// tau of the step from (r, c) to (r + sy*dy, c + sy*dx): its far end and the cells it crosses
__device__ inline double step_tau(const double* o, int ny, int nx, int r, int c, int dx, int dy,
                                  const int* offs, int noffs, int sy) {
    double tau = opac(o, ny, nx, r + sy * dy, c + sy * dx);
    for (int i = 0; i < noffs; i++) tau += opac(o, ny, nx, r + sy * offs[2*i+1], c + sy * offs[2*i]);
    return tau;
}
extern "C" __global__ void soft_fb(const double* o, int ny, int nx, int dx, int dy,
        const int* offs, int noffs, double s, double kappa, const int* sr, const int* sc,
        int nstart, double* F, double* B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nstart) return;
    double k = kappa * s / (noffs + 1);
    int r = sr[t], c = sc[t], n = 0;
    while (r >= 0 && r < ny && c >= 0 && c < nx) {            // B along the line, from its start
        int x = r * nx + c;
        if (n == 0) B[x] = 0.0;
        else B[x] = exp(-k * step_tau(o, ny, nx, r, c, dx, dy, offs, noffs, -1))
                    * (s + B[(r - dy) * nx + (c - dx)]);
        r += dy; c += dx; n++;
    }
    r -= dy; c -= dx;
    for (int j = 0; j < n; j++) {                             // F from its end, back to the start
        int x = r * nx + c;
        if (j == 0) F[x] = 0.0;
        else F[x] = exp(-k * step_tau(o, ny, nx, r, c, dx, dy, offs, noffs, 1))
                    * (s + F[(r + dy) * nx + (c + dx)]);
        r -= dy; c -= dx;
    }
}
__device__ inline void addto(double* out, int ny, int nx, int r, int c, double v) {
    if (r >= 0 && r < ny && c >= 0 && c < nx) atomicAdd(&out[r * nx + c], v);
}
// the transmittances recomputed, not stored (soft_fb's, bit for bit)
extern "C" __global__ void soft_vjp(const double* o, int ny, int nx, int dx, int dy,
        const int* offs, int noffs, double s, double kappa, const int* sr, const int* sc,
        int nstart, const double* F, const double* B, const double* G, double* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nstart) return;
    double k = kappa * s / (noffs + 1);
    // forward rays: F(p) = T(p) (s + F(x)), x = p + v: walk from the start, carrying bar F
    int r = sr[t], c = sc[t], n = 0;
    double bar = 0.0, Tprev = 0.0;
    while (r >= 0 && r < ny && c >= 0 && c < nx) {
        int x = r * nx + c;
        bar = G[x] + Tprev * bar;
        int r2 = r + dy, c2 = c + dx;
        double T = 0.0;
        if (r2 >= 0 && r2 < ny && c2 >= 0 && c2 < nx) {
            T = exp(-k * step_tau(o, ny, nx, r, c, dx, dy, offs, noffs, 1));
            if (bar != 0.0) {
                double coef = bar * (s + F[r2 * nx + c2]) * T * k;
                addto(out, ny, nx, r2, c2, coef);
                for (int i = 0; i < noffs; i++)
                    addto(out, ny, nx, r + offs[2*i+1], c + offs[2*i], coef);
            }
        }
        Tprev = T;
        r += dy; c += dx; n++;
    }
    // backward rays: B(p) = T(p) (s + B(x)), x = p - v: walk from the end
    r -= dy; c -= dx;
    bar = 0.0; Tprev = 0.0;
    for (int j = 0; j < n; j++) {
        int x = r * nx + c;
        bar = G[x] + Tprev * bar;
        int r2 = r - dy, c2 = c - dx;
        double T = 0.0;
        if (r2 >= 0 && r2 < ny && c2 >= 0 && c2 < nx) {
            T = exp(-k * step_tau(o, ny, nx, r, c, dx, dy, offs, noffs, -1));
            if (bar != 0.0) {
                double coef = bar * (s + B[r2 * nx + c2]) * T * k;
                addto(out, ny, nx, r2, c2, coef);
                for (int i = 0; i < noffs; i++)
                    addto(out, ny, nx, r - offs[2*i+1], c - offs[2*i], coef);
            }
        }
        Tprev = T;
        r -= dy; c -= dx;
    }
}
"""


@dataclass(frozen=True)
class GpuScans:
    """cupy: one GPU thread per lattice line along the direction (each line is an independent
    forward / backward recurrence); the vjp scatters with atomicAdd (a crossed cell belongs to
    other lines too). Needs cupy and CUDA_PATH."""
    _memo: dict = dataclasses.field(default_factory=dict, compare=False, repr=False)

    @property
    def xp(self):
        import cupy as cp
        return cp

    def _kernels(self):
        if "kernels" not in self._memo:
            cp = self.xp
            mod = cp.RawModule(code=_SCAN_CU)
            self._memo["kernels"] = (
                mod.get_function("soft_fb"), mod.get_function("soft_vjp"),
                cp.ElementwiseKernel("float64 B, float64 r0, float64 hill", "float64 F",
                                     "double x = pow((F + B) / r0, hill); F = x / (1.0 + x)",
                                     "soft_response"),
                cp.ElementwiseKernel(
                    "float64 F, float64 B, float64 beta, float64 r0, float64 hill", "float64 M",
                    "double R = F + B; double x = pow(R / r0, hill); double q = 1.0 + x; "
                    "M = beta * (hill * x / ((R + 1e-12) * (q * q))) * M",
                    "soft_response_weight"))
        return self._memo["kernels"]

    def _starts(self, ny: int, nx: int, dx: int, dy: int):
        """Cells whose predecessor (r - dy, c - dx) is off the grid: one per line. They are
        the cells within dy of the top or dx of the near side edge."""
        key = (ny, nx, dx, dy)
        if key not in self._memo:
            cp = self.xp
            st = np.zeros((ny, nx), dtype=bool)
            st[:dy] = True                                    # dy >= 0
            if dx > 0:
                st[:, :dx] = True
            elif dx < 0:
                st[:, nx + dx:] = True
            r, c = np.nonzero(st)
            self._memo[key] = (cp.asarray(r.astype(np.int32)), cp.asarray(c.astype(np.int32)),
                               len(r))
        return self._memo[key]

    def fb(self, open_, dx, dy, offs, s, kappa):
        cp = self.xp
        ny, nx = open_.shape
        f_fb = self._kernels()[0]
        sr, sc, ns = self._starts(ny, nx, dx, dy)
        F, B = cp.zeros((ny, nx)), cp.zeros((ny, nx))
        og = cp.ascontiguousarray(open_, dtype=cp.float64)
        oo = cp.asarray(np.asarray(offs, dtype=np.int32).ravel())
        f_fb(((ns + 127) // 128,), (128,),
             (og, np.int32(ny), np.int32(nx), np.int32(dx), np.int32(dy), oo,
              np.int32(len(offs)), np.float64(s), np.float64(kappa), sr, sc, np.int32(ns),
              F, B))
        return F, B

    def vjp_add(self, open_, dx, dy, offs, s, kappa, F, B, G, out):
        cp = self.xp
        ny, nx = F.shape
        f_vjp = self._kernels()[1]
        sr, sc, ns = self._starts(ny, nx, dx, dy)
        oo = cp.asarray(np.asarray(offs, dtype=np.int32).ravel())
        f_vjp(((ns + 127) // 128,), (128,),
              (cp.ascontiguousarray(open_, dtype=cp.float64), np.int32(ny), np.int32(nx),
               np.int32(dx), np.int32(dy), oo, np.int32(len(offs)), np.float64(s),
               np.float64(kappa), sr, sc, np.int32(ns), F, B,
               cp.ascontiguousarray(G, dtype=cp.float64), out))

    def response(self, F, B, r0, hill):
        """g(F + B), into F."""
        self._kernels()[2](B, r0, hill, F)
        return F

    def response_weight(self, F, B, M, beta, r0, hill):
        """beta g'(F + B) M, into M."""
        self._kernels()[3](F, B, beta, r0, hill, M)
        return M

    def to_dev(self, a):
        return self.xp.asarray(a)

    def to_host(self, a):
        return self.xp.asnumpy(a)


@dataclass(frozen=True)
class SoftSightline:
    """Sightline with translucent buildings: a ray's intensity falls as exp(-kappa x the closed
    length it crosses), and R_d is its expected free path (forward + back) along direction d. A
    line blocked only by a small building still scores, so clearing that building has a
    gradient (the counterfactual corridor is visible); as kappa grows this tends to the hard
    runs. Layer k's factor is 1 + beta x the angle-weighted mean of g(R_d) = R^n / (R^n + r0^n):
    n 1 is concave (short gaps already earn a lot), n 2 is S-shaped (gaps well under r0 earn
    almost nothing, lines past r0 nearly all; joining two runs across r0 pays most).
    Differentiable: `vjp` is exact. (Turning walkers were tried and dropped: NOTES, "Turning
    sightline".)"""
    beta: float
    kappa: float = 2.0         # per metre of building crossed
    r0_m: float = 20.0
    hill: float = 1.0          # the exponent n
    nmax: int = 6
    scans: Scans = CpuScans()
    # the last open field's layers (one round builds the same operator more than once)
    _cache: dict = dataclasses.field(default_factory=dict, compare=False, repr=False)

    @property
    def name(self) -> str:
        return f"ss{self.beta:g}k{self.kappa:g}" + (
            "" if self.hill == 1.0 and self.r0_m == 20.0 else f"n{self.hill:g}r{self.r0_m:g}")

    def g(self, R):
        x = (R / self.r0_m) ** self.hill
        return x / (1.0 + x)

    def dg(self, R):
        x = (R / self.r0_m) ** self.hill
        return self.hill * x / ((R + 1e-12) * (1.0 + x) ** 2)

    def _dirs(self):
        dirs = directions(self.nmax)
        ang = np.array([np.arctan2(dy, dx) % np.pi for dx, dy in dirs])
        return dirs, ang

    def _steps(self, h):
        """Per fine direction: (index, dx, dy, crossed-cell offsets, step length)."""
        for i, (dx, dy) in enumerate(self._dirs()[0]):
            yield (i, dx, dy, np.array(_line_cells(dx, dy), dtype=np.int64).reshape(-1, 2),
                   float(np.hypot(dx, dy) * h))

    def runs(self, open_, h):
        """Yield (d, R_d) per fine direction, R_d the expected free path in metres."""
        for i, F, B in self.paths(open_, h):
            yield i, F + B

    def paths(self, open_, h):
        """Yield (d, F_d, B_d) on the host: the expected free path forward and back. F x B is
        the measure of the segments along d that pass through the cell (pairs of a start behind
        and an end ahead); its sum along a line of length L is L^2 / 2."""
        od = self.scans.to_dev(open_)
        for i, dx, dy, offs, s in self._steps(h):
            F, B = self.scans.fb(od, dx, dy, offs, s, self.kappa)
            yield i, self.scans.to_host(F), self.scans.to_host(B)

    def layers(self, grid, open_, K):
        """(K, field) on the scans' device."""
        key = (id(grid), hashlib.blake2b(np.ascontiguousarray(self.scans.to_host(open_)).tobytes(),
                                         digest_size=16).digest(), K)
        if key in self._cache and self._cache[key][0]() is grid:
            return self._cache[key][1]
        xp = self.scans.xp
        W = _angle_weights(K, self._dirs()[1])
        od = grid.paint(self.scans.to_dev(open_), xp)
        out = xp.ones((K, *open_.shape))
        for i, dx, dy, offs, s in self._steps(grid.h):
            F, B = self.scans.fb(od, dx, dy, offs, s, self.kappa)
            g = grid.mean(self.scans.response(F, B, self.r0_m, self.hill), xp)
            del F, B                                    # rasters: free them before the next
            for k in np.flatnonzero(W[:, i]):
                out[k] += (self.beta * W[k, i]) * g
        self._cache.clear()
        self._cache[key] = (weakref.ref(grid), out)    # an id outlives its grid: check it
        return out

    def turning(self, grid, open_, K):
        return [1.0] * K

    def vjp(self, grid, open_, K, A):
        """(field) on the scans' device."""
        xp = self.scans.xp
        W = _angle_weights(K, self._dirs()[1])
        od = grid.paint(self.scans.to_dev(open_), xp)
        Ad = self.scans.to_dev(A)
        out = xp.zeros(od.shape)
        for i, dx, dy, offs, s in self._steps(grid.h):
            F, B = self.scans.fb(od, dx, dy, offs, s, self.kappa)
            G = self.scans.response_weight(
                F, B, grid.mean_T(xp.tensordot(xp.asarray(W[:, i]), Ad, axes=1), xp),
                self.beta, self.r0_m, self.hill)
            self.scans.vjp_add(od, dx, dy, offs, s, self.kappa, F, B, G, out)
            del F, B, G
        return grid.gather(out, xp)


@dataclass(frozen=True)
class TurnBoosted:
    """An along-conductance whose boost scales the turning edges too: the turning edge
    (x, k)-(x, k+1) times min(F_k(x), F_k+1(x)), F the inner conductance's layers (NOTES, "The
    sightline metric's angle error": the boost then scales the whole (x, theta) metric, so a
    walker in a boosted lane keeps the persistence length ell instead of going ballistic and
    zig-zagging between headings). For scoring and maps only: no gradient (`vjp` raises, so
    neither SIMP nor the greedy's tension can run under it). The layers' device is the inner
    conductance's (np.minimum dispatches to cupy's)."""
    inner: AlongConductance

    @property
    def name(self) -> str:
        return self.inner.name + "t"

    def layers(self, grid, open_, K):
        return self.inner.layers(grid, open_, K)

    def turning(self, grid, open_, K):
        F = self.inner.layers(grid, open_, K)
        return [np.minimum(F[k], F[(k + 1) % K]) for k in range(K)]

    def vjp(self, grid, open_, K, A):
        raise NotImplementedError(f"{self.name}: the turning boost is not in the gradient (for "
                                  "scoring and maps only)")


def along_of(spec: str, scans: Scans = CpuScans()) -> AlongConductance:
    """`uni` -> Uniform(); `sl3` -> Sightline(beta 3); `ss10k2` -> SoftSightline(beta 10,
    kappa 2); `ss30k2n2r30` -> SoftSightline(beta 30, kappa 2, hill 2, r0 30 m); a trailing `t`
    on an `ss` spec (`ss100k2n2r30t`) -> TurnBoosted of it."""
    if spec == "uni":
        return Uniform()
    if spec.startswith("sl"):
        return Sightline(float(spec[2:]))
    m = re.fullmatch(r"ss([0-9.]+)k([0-9.]+)(?:n([0-9.]+)r([0-9.]+?))?(t?)", spec)
    if m:
        shape = {} if m.group(3) is None else dict(hill=float(m.group(3)),
                                                   r0_m=float(m.group(4)))
        ss = SoftSightline(float(m.group(1)), float(m.group(2)), **shape, scans=scans)
        return TurnBoosted(ss) if m.group(5) else ss
    raise ValueError(f"unknown along conductance {spec!r}")


# ----------------------------------------------------------------- what an open cell conducts by

class CellConductance(Protocol):
    """What an open cell conducts by (Params.cells; NOTES, "Uniform conductance with an effective
    width"): factor(grid, open_, solver), per cell on the solver's device, in place of the cell's
    open fraction in its turning edges' weight. Which cells are in the operator does not change
    (open fraction above 0), and every factor is above 0 there and at most the open fraction, and
    it grows as space is freed, so Rayleigh monotonicity holds.
    vjp(grid, open_, A, solver): the gradient with respect to open_ of sum_x A[x] factor[x]
    (relax.Relaxation's chain rule), or NotImplementedError where the factor has none.
    `pointwise`: a cell's factor depends on its own open fraction alone, which the greedy's
    secant tension assumes (clear.Clearing.tension). `name`: its part of a run's conductance
    (conductance_of), "" for the open fraction itself.
    sub_factor(grid, open_, solver): (cells, S*S) on the solver's device, what each sub-sample
    conducts by, of which the cell's factor is the mean: what the steps' tracks read
    (step_widths; NOTES, "The grid fix"), so a factor without one has no operator.
    sub_vjp(grid, open_, A, solver): the gradient with respect to open_ (per cell, flat) of
    sum A[c, s] sub_factor[c, s], or NotImplementedError."""

    @property
    def name(self) -> str: ...

    @property
    def pointwise(self) -> bool: ...

    def factor(self, grid: Grid | CompositeGrid, open_, solver: Solver): ...

    def vjp(self, grid: Grid | CompositeGrid, open_, A, solver: Solver): ...

    def sub_factor(self, grid: Grid | CompositeGrid, open_, solver: Solver): ...

    def sub_vjp(self, grid: Grid | CompositeGrid, open_, A, solver: Solver): ...


class _SubState(NamedTuple):
    """A grid's sub-samples per cell (flat cells, S*S), on one device, for the per-sub-sample
    factors: `inside` and `cov` (inside a footprint) as 1.0 / 0.0, `n` each cell's covered count
    (0 without any), `o0` its open fraction at baseline."""
    inside: object
    cov: object
    n: object
    o0: object


def _sub_state(grid: Grid | CompositeGrid, xp) -> _SubState:
    key = ("sub state", xp.__name__)
    if key not in grid.cache:
        ncell = grid.inside.size
        isub = grid.isub.reshape(ncell, -1)
        cov = isub & grid.bsub.reshape(ncell, -1)
        grid.cache[key] = _SubState(inside=xp.asarray(isub, dtype=xp.float64),
                                    cov=xp.asarray(cov, dtype=xp.float64),
                                    n=xp.asarray(cov.sum(axis=1).astype(np.float64)),
                                    o0=xp.asarray(grid.ff0.ravel()))
    return grid.cache[key]


def _openness(grid: Grid | CompositeGrid, o, xp):
    """(cells, S*S) on the device: each sub-sample's openness in the flat open field `o`: 1 where
    open at baseline, 0 outside the block, and on a footprint its cell's cleared share of its
    footprint sub-samples, (o - o0) S^2 / n clipped to [0, 1] (`_closed`'s rule: the cleared part
    of a cell spread evenly over its footprint, exact for a cell under one building; written so a
    cell cleared whole gives exactly 1). Affine in o between the clips, and its cell mean is o."""
    st = _sub_state(grid, xp)
    S2 = st.inside.shape[1]
    t = xp.clip((o - st.o0) * S2 / xp.maximum(st.n, 1.0), 0.0, 1.0)
    return st.inside - st.cov * (1.0 - t)[:, None]


def _openness_vjp(grid: Grid | CompositeGrid, A, xp):
    """The gradient with respect to the flat open field of sum A[c, s] _openness[c, s] (the clip
    taken as inactive: a one-sided derivative where a cell is cleared whole)."""
    st = _sub_state(grid, xp)
    S2 = st.inside.shape[1]
    return (st.cov * A).sum(axis=1) * (S2 / xp.maximum(st.n, 1.0))


@dataclass(frozen=True)
class OpenFraction:
    """U: a cell conducts by its open fraction, so a gap narrower than a cell conducts in
    proportion to its width and a channel in proportion to its width (today's model, the
    sub-cell fractions' fix for the flicker: NOTES, "Results so far"). Per sub-sample, its
    openness (`_openness`)."""

    @property
    def name(self) -> str:
        return ""

    @property
    def pointwise(self) -> bool:
        return True

    def factor(self, grid, open_, solver):
        return open_

    def vjp(self, grid, open_, A, solver):
        return A

    def sub_factor(self, grid, open_, solver):
        return _openness(grid, open_.ravel(), solver.xp)

    def sub_vjp(self, grid, open_, A, solver):
        return _openness_vjp(grid, A, solver.xp)


class _FineWalls(NamedTuple):
    """A uniform grid's footprints on its fine lattice, for the factors that look past a cell
    (Blurred, WallDistance), on one device. `cov` (ny S, nx S): 1.0 at an inside sub-sample a
    footprint covers, 0 elsewhere (a sub-sample outside the block counts open: every ring of a
    kblock, and a region's outer ring, is a street, and a hole is open space); `inside` the
    inside sub-samples, as 1.0; `spread` (ny, nx): S^2 / the cell's covered sub-samples (0
    without any), so one unit of the cell's open fraction above its baseline `o0` clears that
    share of each of its footprint sub-samples."""
    cov: object
    inside: object
    spread: object
    o0: object


def _fine(sub: NDArray, S: int) -> NDArray:
    """(ny, nx, S*S) -> (ny S, nx S): `_to_cells`'s inverse."""
    ny, nx = sub.shape[:2]
    return sub.reshape(ny, nx, S, S).transpose(0, 2, 1, 3).reshape(ny * S, nx * S)


def _uniform(grid: Grid | CompositeGrid, who: str) -> Grid:
    if not isinstance(grid, Grid):
        raise NotImplementedError(f"{who}: built on the uniform grid's fine lattice; a "
                                  f"{type(grid).__name__} has cells of several sizes")
    return grid


def _fine_walls(grid: Grid, xp) -> _FineWalls:
    key = ("fine walls", xp.__name__)
    if key not in grid.cache:
        cov = grid.isub & grid.bsub
        n = cov.sum(axis=-1)
        grid.cache[key] = _FineWalls(
            cov=xp.asarray(_fine(cov, grid.S), dtype=xp.float64),
            inside=xp.asarray(_fine(grid.isub, grid.S), dtype=xp.float64),
            spread=xp.asarray(np.where(n > 0, grid.S * grid.S / np.maximum(n, 1), 0.0)),
            o0=xp.asarray(grid.ff0))
    return grid.cache[key]


def _closed(grid: Grid, o, xp):
    """(ny S, nx S) on the device: each sub-sample's closed share in the open field `o`
    (ny, nx): a footprint sub-sample's is 1 - t, t its cell's cleared share of its footprint
    sub-samples, (o - o0) spread, clipped to [0, 1] (the cleared part of a cell spread evenly
    over its footprint: exact for a cell under one building, cleared or not); every other
    sub-sample's 0."""
    fw = _fine_walls(grid, xp)
    t = xp.clip((o - fw.o0) * fw.spread, 0.0, 1.0)
    S = grid.S
    return fw.cov * xp.repeat(xp.repeat(1.0 - t, S, axis=0), S, axis=1)


class _Disk(NamedTuple):
    """A disk of radius r about a sub-sample of the fine lattice, as rows: (dr, L) the sub-samples
    dr rows away and up to L columns either side; `n` how many, `pad` the margin a raster needs
    for every sub-sample's disk to fall inside it."""
    rows: tuple[tuple[int, int], ...]
    n: int
    pad: int


def _disk(r_m: float, h: float, S: int) -> _Disk:
    rho = r_m * S / h                       # in sub-samples
    R = int(np.floor(rho + 1e-9))
    rows = tuple((dr, int(np.floor(np.sqrt(rho * rho - dr * dr) + 1e-9)))
                 for dr in range(-R, R + 1))
    return _Disk(rows=rows, n=sum(2 * L + 1 for _, L in rows), pad=R + 1)


def _disk_sum(Z, disk: _Disk, xp):
    """The sum of the fine raster Z over the disk about each of its sub-samples (0 beyond Z), by
    row prefix sums. The disk is symmetric, so this is its own adjoint."""
    P = disk.pad
    ny, nx = Z.shape
    C = xp.zeros((ny + 2 * P, nx + 2 * P + 1))
    C[P:P + ny, P + 1:P + 1 + nx] = Z
    C = xp.cumsum(C, axis=1)                # C[:, k]: the sum of the padded row left of k
    out = xp.zeros((ny, nx))
    for dr, L in disk.rows:
        rows = slice(P + dr, P + dr + ny)
        out += C[rows, P + L + 1:P + L + 1 + nx] - C[rows, P - L:P - L + nx]
    return out


def _cell_sum(F, S: int):
    """(ny S, nx S) -> (ny, nx): each cell's sum over its sub-samples."""
    ny, nx = F.shape[0] // S, F.shape[1] // S
    return F.reshape(ny, S, nx, S).sum(axis=(1, 3))


@dataclass(frozen=True)
class Blurred:
    """B_r: each sub-sample of the fine lattice conducts by min(its openness, B), B the open share
    of the sub-samples within r_m of it (a disk; outside the block counts open: _FineWalls), and
    a cell by the sum of that over its sub-samples inside the block, over S^2 (as its open
    fraction counts them). At a 0/1 state min(open, B) is open x B: open ground conducts by how
    open its neighbourhood is. In a channel narrower than 2 r, B grows with the width everywhere
    in it, so the channel conducts like width^2, and like width beyond, less 2 r / 3 pi at each
    straight wall. Integrated over the cell's own sub-samples, so a cell partly open conducts in
    proportion: B sampled at the cell's centre instead lets a cell half in a narrow channel
    conduct about as much as one wholly in it, and how many cells a channel touches sets its
    conductance (NOTES: the flicker the sub-cell fractions removed). B is affine in the open
    field (the cleared part of a cell spread evenly over its footprint sub-samples, `_closed`),
    so the factor, a sum of minimums of affine functions, is concave in the clearing fractions
    and P stays convex (relax.py). Above 0 wherever a cell has an open sub-sample (its own disk
    holds it). r_m is required: it names the conductance (`b<r>`)."""
    r_m: float

    @property
    def name(self) -> str:
        return f"b{self.r_m:g}"

    @property
    def pointwise(self) -> bool:
        return False

    def _parts(self, g: Grid, o, xp):
        """(Z the closed share per sub-sample, B the open share of its disk, the disk)."""
        Z = _closed(g, o, xp)
        disk = _disk(self.r_m, g.h, g.S)
        return Z, 1.0 - _disk_sum(Z, disk, xp) / disk.n, disk

    def factor(self, grid, open_, solver):
        g, xp = _uniform(grid, self.name), solver.xp
        Z, B, _ = self._parts(g, open_.reshape(g.inside.shape), xp)
        k = _fine_walls(g, xp).inside * xp.minimum(1.0 - Z, B)
        return (_cell_sum(k, g.S) / (g.S * g.S)).reshape(open_.shape)

    def vjp(self, grid, open_, A, solver):
        """A sub-sample's term is its openness where that is at most B (its own cell's
        footprint), else B, whose gradient is the disk's (its own adjoint); each through the
        cell footprint sub-samples' share cleared (the clip in `_closed` taken as inactive: a
        one-sided derivative where a cell is fully cleared)."""
        g, xp = _uniform(grid, self.name), solver.xp
        ny, nx = g.inside.shape
        S = g.S
        a = xp.repeat(xp.repeat(A.reshape(ny, nx) / (S * S), S, axis=0), S, axis=1)
        return self._vjp_fine(g, open_, a, xp)

    def _vjp_fine(self, g: Grid, open_, a, xp):
        """`vjp` from the weight per sub-sample `a` (the fine raster, ny S x nx S)."""
        ny, nx = g.inside.shape
        S = g.S
        Z, B, disk = self._parts(g, open_.reshape(ny, nx), xp)
        fw = _fine_walls(g, xp)
        a = fw.inside * a
        own = (1.0 - Z) <= B
        H = xp.where(own, a, 0.0) + _disk_sum(xp.where(own, 0.0, a), disk, xp) / disk.n
        return (_cell_sum(fw.cov * H, S) * fw.spread).reshape(open_.shape)

    def sub_factor(self, grid, open_, solver):
        """Per sub-sample min(its openness, B), inside the block (`factor` before its sum)."""
        g, xp = _uniform(grid, self.name), solver.xp
        Z, B, _ = self._parts(g, open_.reshape(g.inside.shape), xp)
        k = _fine_walls(g, xp).inside * xp.minimum(1.0 - Z, B)
        return _to_cells(k, g.S).reshape(g.inside.size, g.S * g.S)

    def sub_vjp(self, grid, open_, A, solver):
        g, xp = _uniform(grid, self.name), solver.xp
        ny, nx = g.inside.shape
        return self._vjp_fine(g, open_, _fine(A.reshape(ny, nx, -1), g.S), xp).ravel()


@dataclass(frozen=True)
class WallDistance:
    """D_s (the owner's distance form): each open sub-sample conducts by min(1, d / s_m), d its
    distance to the nearest footprint sub-sample still standing, measured from its centre to the
    footprint's edge (the fine lattice's Euclidean distance transform less half a sub-sample;
    outside the block counts open: _FineWalls), and a cell by the mean of that over its
    sub-samples, each weighted by its openness, so a channel conducts like width^2 below 2 s and
    like width less s beyond. A footprint sub-sample stands when none of its cell's footprint is
    cleared (`_closed` exactly 1): a cell under one building is exact, a cell two buildings share
    counts both cleared once either is. The distances move with every clearing, so the factor is
    not affine in the clearing fractions and has no gradient here: for scoring (and a greedy's
    exact solves), not SIMP. s_m is required: it names the conductance (`d<s>`)."""
    s_m: float

    @property
    def name(self) -> str:
        return f"d{self.s_m:g}"

    @property
    def pointwise(self) -> bool:
        return False

    def factor(self, grid, open_, solver):
        g = _uniform(grid, self.name)
        S = g.S
        return (_cell_sum(self._fine_factor(g, open_, solver), S) / (S * S)).reshape(open_.shape)

    def _fine_factor(self, g: Grid, open_, solver):
        """(ny S, nx S): each sub-sample's openness x min(1, d / s_m), inside the block."""
        xp = solver.xp
        ny, nx = g.inside.shape
        S = g.S
        Z = _closed(g, open_.reshape(ny, nx), xp)
        stand = Z >= 1.0
        if bool(stand.any()):
            d = solver.ndimage.distance_transform_edt(~stand) * (g.h / S) - g.h / (2 * S)
            ramp = xp.clip(d / self.s_m, 0.0, 1.0)
        else:
            ramp = xp.ones(Z.shape)
        return _fine_walls(g, xp).inside * (1.0 - Z) * ramp

    def vjp(self, grid, open_, A, solver):
        raise NotImplementedError(f"{self.name}: the distances move with every clearing, so "
                                  "the factor is not affine in the clearing fractions; no "
                                  "gradient (for scoring, not SIMP)")

    def sub_factor(self, grid, open_, solver):
        g = _uniform(grid, self.name)
        return _to_cells(self._fine_factor(g, open_, solver), g.S).reshape(g.inside.size,
                                                                            g.S * g.S)

    def sub_vjp(self, grid, open_, A, solver):
        return self.vjp(grid, open_, A, solver)


def cells_of(token: str) -> CellConductance:
    """`b<r>` -> Blurred(r metres); `d<s>` -> WallDistance(s metres)."""
    if m := re.fullmatch(r"b([0-9]+(?:\.[0-9]+)?)", token):
        return Blurred(float(m[1]))
    if m := re.fullmatch(r"d([0-9]+(?:\.[0-9]+)?)", token):
        return WallDistance(float(m[1]))
    raise ValueError(f"unknown cell conductance {token!r}: b<r> or d<s>")


# ------------------------------------------------------------- how much of a step's width conducts
#
# The face aperture (Fws; owner 2026-10-13, "We can adopt Fws and delete the losing methods";
# NOTES, "The grid fix: steps that conduct by their open width"): a step conducts by the share of
# its width open all the way from one cell's centre to the next, read on the S x S sub-samples.
# Tracks parallel to v_k (axis_tracks: across the face for an axis-aligned step, centre to centre
# across the cell's width h for an oblique one), each the 4-connected chain of sub-samples its
# segment passes through, each conducting by the smallest of p.cells' per-sub-sample factors on
# it (U: the openness); the step by the mean over its tracks, so an open field gives every step
# its whole weight, a crack one sub-sample wide on an axis-aligned step conducts 1/S of it, and
# two cells whose open sub-samples do not touch conduct nothing: U computed faithfully, a cut-cell
# scheme's face apertures, no new parameter. Its reach is the 4-connected flood fill of the
# sub-samples it reads (aperture_check's `connect`). Under fractional clearing a sub-sample's
# factor is affine in the clearing (U's openness, `_openness`) or concave (B_r's), so each track,
# a minimum of them, and the step, a mean of tracks, are concave, and P stays convex and
# monotone; the gradient shares each track's coefficient among the sub-samples tied at its
# minimum. On a CompositeGrid a step with a coarse end conducts by 1: every cell within a step of
# a coarse cell is wholly open (AdaptiveMesh's d0 > the ground band). A split cell, whose open
# sub-samples form parts that do not touch, conducts through its anchor's part only (below). The
# rule is the operator's cell graph, so B's reach (`placeable`), the enclosed test, free
# first (`regions`) and the scorer all read it. Its name, APERTURE, ends every conductance's.

APERTURE = "apwfs"      # the step rule's token, last in every conductance's name (conductance_of):
                        # the study's Fws, this rule; a name written under any other ends
                        # otherwise (the smallest open fraction of the cells a step touches,
                        # until 2026-10-13, none: uni, uni+b1; the study's A, Aw, Aws ap, apw,
                        # apws), so no row of theirs is read as this rule's


class Regions(NamedTuple):
    """Free first's graph at baseline (clear.free_first, clear.RegionGraph; `regions`): regions
    0 .. ncomp - 1 the components of open space, then the regions a clearing enters, each with its
    `owner` building (owner[r - ncomp]: pieces of buildings, and gadgets); (a, b) the adjacency,
    both ways; `entry` the regions the street steps into; `R` (grid cells flat, regions) the
    regions a home is freed through at each cell (RegionGraph.sources): a cell that can take demand
    at baseline its component (and the pieces covering it), any other the pieces whose building,
    cleared alone, lets it."""
    ncomp: int
    owner: NDArray[np.int64]
    a: NDArray[np.int64]
    b: NDArray[np.int64]
    entry: NDArray[np.int64]
    R: sp.csr_matrix


class Tracks(NamedTuple):
    """One axis's tracks on the sub-sample raster, relative to a step's first cell a
    (axis_tracks): `cells` the cell offsets (dx, dy) the tracks pass through, in slot order: a, b,
    the cells the step crosses as _line_cells gives them (Pattern.along's `line`), then any other;
    `tracks`, per track, the (slot, sub-sample index) of each sub-sample it passes through, in
    order."""
    cells: tuple[tuple[int, int], ...]
    tracks: tuple[tuple[tuple[int, int], ...], ...]


def _segment_cover(p0: tuple[Fraction, Fraction],
                   p1: tuple[Fraction, Fraction]) -> list[tuple[int, int]]:
    """The squares (row I, column J; square (I, J) is [J, J + 1) x [I, I + 1)) whose interior the
    open segment p0 -> p1 (points (x, y), exact) passes through, in order. Raises where it runs
    along a grid line or through a lattice corner (it would touch squares only at a point)."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    tx, ty = set(), set()
    if dx != 0:
        for X in range(math.floor(min(x0, x1)), math.ceil(max(x0, x1)) + 1):
            t = (X - x0) / dx
            if 0 < t < 1:
                tx.add(t)
    if dy != 0:
        for Y in range(math.floor(min(y0, y1)), math.ceil(max(y0, y1)) + 1):
            t = (Y - y0) / dy
            if 0 < t < 1:
                ty.add(t)
    if tx & ty:
        raise ValueError(f"segment {p0} -> {p1} passes through a lattice corner")
    ts = sorted({Fraction(0), Fraction(1)} | tx | ty)
    out: list[tuple[int, int]] = []
    for t0, t1 in zip(ts[:-1], ts[1:], strict=True):
        tm = (t0 + t1) / 2
        xm, ym = x0 + tm * dx, y0 + tm * dy
        if xm == math.floor(xm) or ym == math.floor(ym):
            raise ValueError(f"segment {p0} -> {p1} runs along a grid line")
        out.append((math.floor(ym), math.floor(xm)))
    return out


@functools.cache
def axis_tracks(dx: int, dy: int, S: int) -> Tracks:
    """A step (dx, dy)'s tracks on a raster of S x S sub-samples a cell (S even, so a cell's
    centre is a lattice corner). Coordinates in sub-samples: cell a's sub-sample (i, j) is the
    square (I, J) = (i, j), its centre at (S/2, S/2), b's at (S/2 + S dx, S/2 + S dy). An
    axis-aligned step's tracks are the S pairs of sub-samples facing each other across the face
    (the cut-cell face aperture: a one-sub-sample sliver along a wall touches its neighbour there,
    where a track to its cell's centre would cross the wall; within a cell its node joins its open
    part, which the anchor rule keeps one part). An oblique step's track is the segment between
    the two centres moved across by an offset u (u = dx y - dy x, the distance across times |v|),
    u at the half-integers with |u| < S |v| / 2, the cell's width h (six tracks on a diagonal
    step, eight on a knight step, overlapping their neighbours'). A lattice corner has an integer
    u, so no track passes through one: each is a 4-connected chain of sub-samples."""
    if S % 2:
        raise ValueError(f"S {S} is odd: a cell's centre is not a lattice corner")
    if dx == 0 or dy == 0:
        if (dx, dy) == (1, 0):
            return Tracks(cells=((0, 0), (1, 0)), tracks=tuple(
                ((0, i * S + S - 1), (1, i * S)) for i in range(S)))
        if (dx, dy) == (0, 1):
            return Tracks(cells=((0, 0), (0, 1)), tracks=tuple(
                ((0, (S - 1) * S + j), (1, j)) for j in range(S)))
        raise ValueError(f"axis ({dx}, {dy}): the lattice's axis vectors are (1, 0) and (0, 1)")
    v2 = dx * dx + dy * dy
    umax = S * math.sqrt(v2) / 2
    us = [Fraction(2 * i + 1, 2) for i in range(-S * v2, S * v2) if abs((2 * i + 1) / 2) < umax]
    cells: list[tuple[int, int]] = [(0, 0), (dx, dy), *_line_cells(dx, dy)]
    tracks = []
    for u in us:
        ox, oy = u * Fraction(-dy, v2), u * Fraction(dx, v2)
        p0 = (Fraction(S, 2) + ox, Fraction(S, 2) + oy)
        p1 = (p0[0] + S * dx, p0[1] + S * dy)
        tr = []
        for row, col in _segment_cover(p0, p1):
            cell = (col // S, row // S)
            if cell not in cells:
                cells.append(cell)
            tr.append((cells.index(cell), (row % S) * S + col % S))
        tracks.append(tuple(tr))
    return Tracks(cells=tuple(cells), tracks=tuple(tracks))


# ------------------------------------------------------------------ split cells
#
# A split cell, whose open sub-samples form parts that do not touch (two buildings meeting at a
# corner inside the cell), conducts through one part only, the one holding its anchor
# (part_anchors: a sub-sample of its largest part at baseline), its other parts closed to every
# track; and a cell split in the field takes no demand nor frees a home (its node would speak for
# parts on the far side of a wall). Opening space only grows the anchor's part, so the operator
# stays monotone in the clearing; at baseline and every 0/1 state the parts are the field's own
# (a clearing that joins them makes them one again), and in the eps world (relax.py: every
# sub-sample above 0) each cell is one part, so the widths' concavity is untouched. In place of a
# node per part, which would make the mesh's cells parts that merge as clearings open footprints.
# A cell with no open sub-sample at baseline has no anchor: its parts, if a clearing opens
# several, join.

@functools.cache
def _parts_lut(S: int) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Over every S x S pattern of open sub-samples m (bit s: the sub-sample (s // S, s % S)):
    how many 4-connected parts it holds; per sub-sample s, the bits of the part holding it (0
    where s is not open), flat (m S^2 + s); and m's popcount."""
    S2 = S * S
    n = 1 << S2
    weights = 1 << np.arange(S2)
    bits = (np.arange(n)[:, None] >> np.arange(S2)[None, :]) & 1
    nparts = np.zeros(n, dtype=np.int64)
    part_of = np.zeros((n, S2), dtype=np.int64)
    for m in range(1, n):
        lab, k = ndimage.label(bits[m].reshape(S, S))
        lab = lab.ravel()
        nparts[m] = k
        comp = np.bincount(lab, weights=weights, minlength=k + 1).astype(np.int64)
        part_of[m] = np.where(lab > 0, comp[lab], 0)
    return nparts, part_of.ravel(), bits.sum(axis=1)


def _open_code(F, xp):
    """(cells,) each cell's pattern of open sub-samples (factor above 0), as _parts_lut's index."""
    S2 = F.shape[1]
    return ((F > 0).astype(xp.int64) << xp.arange(S2, dtype=xp.int64)[None, :]).sum(axis=1)


def part_anchors(grid: Grid | CompositeGrid) -> NDArray[np.int64]:
    """(cells flat) on the host, kept per grid: each cell's anchor, the first sub-sample of its
    largest part of open sub-samples at baseline (-1 where none is open)."""
    key = ("part anchors",)
    if key not in grid.cache:
        S2 = grid.S * grid.S
        nparts, part_of, popc = _parts_lut(grid.S)
        code = _open_code((grid.isub & ~grid.bsub).reshape(-1, S2), np)
        size = popc[part_of[code[:, None] * S2 + np.arange(S2)[None, :]]]
        grid.cache[key] = np.where(code > 0, np.argmax(size, axis=1), -1)
    return grid.cache[key]


def _part_luts(grid: Grid | CompositeGrid, xp) -> tuple:
    """(_parts_lut's part count and part bits, part_anchors) on device `xp`, kept per grid."""
    key = ("part luts", xp.__name__)
    if key not in grid.cache:
        nparts, part_of, _popc = _parts_lut(grid.S)
        grid.cache[key] = (xp.asarray(nparts), xp.asarray(part_of),
                           xp.asarray(part_anchors(grid)))
    return grid.cache[key]


def _anchor_keep(F, grid: Grid | CompositeGrid, xp):
    """(cells, S*S) 1.0 at the sub-samples the tracks read, 0.0 elsewhere (the same map on their
    gradient), from the per-sub-sample factors F of the field: in a cell with an anchor its
    anchor's part, in one without every open sub-sample."""
    S2 = F.shape[1]
    _nparts, part_of, anchor = _part_luts(grid, xp)
    code = _open_code(F, xp)
    kept = xp.where(anchor >= 0, part_of[code * S2 + xp.maximum(anchor, 0)], code)
    return ((kept[:, None] >> xp.arange(S2, dtype=xp.int64)[None, :]) & 1).astype(xp.float64)


def _one_part(F: NDArray[np.float64], grid: Grid | CompositeGrid) -> NDArray[np.bool_]:
    """(cells flat) on the host, F host: the cells whose open sub-samples form one part at most,
    the cells that may take a home's demand and free an enclosed home."""
    nparts, _part_of, _anchor = _part_luts(grid, np)
    return nparts[_open_code(F, np)] < 2


def _never_open(code: NDArray[np.int64], F: NDArray[np.float64],
                grid: Grid | CompositeGrid) -> NDArray[np.int64]:
    """Free first's per-sub-sample codes (building, -1 open, -2 never open; the dummy cell's
    after the cells') at baseline (F host), the open sub-samples the tracks do not read made never
    open."""
    n, S2 = F.shape
    drop = (F > 0) & (_anchor_keep(F, grid, np) == 0)
    out = code.copy()
    out[:n * S2][drop.ravel()] = -2
    return out


def _cleared_alone(cells: NDArray[np.int64], jbits: NDArray[np.int64], F: NDArray[np.float64],
                   grid: Grid | CompositeGrid) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Free first's view of one building cleared alone in a cell, per incidence (a cell, the bits
    of that building's sub-samples there, F host at baseline): whether its sub-samples then join
    the part the cell's node conducts through (the part then holding the anchor takes some of
    them; a cell with no anchor: whatever opens joins), and whether the cell can then take demand
    (what is then open is one part)."""
    nparts, part_of, anchor = _part_luts(grid, np)
    S2 = F.shape[1]
    code = _open_code(F[cells], np) | jbits
    a = anchor[cells]
    joins = np.where(a >= 0, (part_of[code * S2 + np.maximum(a, 0)] & jbits) != 0, True)
    return joins, nparts[code] < 2


# ------------------------------------------------------------------ the steps' widths

class _AxisTracks(NamedTuple):
    """One axis's tracks on a grid (track_axes): `base` per slot (Tracks.cells) the cell of every
    edge of Pattern.along[k], times S*S (the dummy cell, past the last, where a slot falls outside
    every cell); `tracks` Tracks.tracks; `coarse` (CompositeGrid) the edges with a coarse end,
    which conduct by 1 (every cell near a coarse one is wholly open), else None."""
    base: list
    tracks: tuple[tuple[tuple[int, int], ...], ...]
    coarse: object


def _track_axes_of(grid: Grid | CompositeGrid, K: int, xp) -> list[_AxisTracks]:
    key = ("tracks", K, xp.__name__)
    if key not in grid.cache:
        grid.cache[key] = grid.track_axes(K, xp)
    return grid.cache[key]


def _track_means(flat, ax: _AxisTracks, xp):
    """Per edge, the mean over ax's tracks of the smallest sub-sample factor on each (`flat`: the
    factors (cells + 1) x S*S, flat, the dummy cell's 0)."""
    acc = None
    for tr in ax.tracks:
        m = None
        for slot, sub in tr:
            v = flat[ax.base[slot] + sub]
            m = v if m is None else xp.minimum(m, v)
        acc = m if acc is None else acc + m
    w = acc / len(ax.tracks)
    return w if ax.coarse is None else xp.where(ax.coarse, 1.0, w)


def _track_factors(grid: Grid | CompositeGrid, o, p: Params):
    """(the factors the tracks read, (cells + 1) x S*S flat, the dummy cell's 0; the anchor
    rule's keep, by which they and their gradient are scaled), from the flat open field `o`."""
    xp, S2 = p.solver.xp, grid.S * grid.S
    F = p.cells.sub_factor(grid, o.reshape(grid.inside.shape), p.solver)
    keep = _anchor_keep(F, grid, xp)
    return xp.concatenate([(F * keep).ravel(), xp.zeros(S2)]), keep


def step_widths(grid: Grid | CompositeGrid, o, p: Params):
    """Per axis k in turn, the share, in [0, 1], of its weight m_k / |v_k|^2 (times the along
    factors) every edge of grid.pattern's along[k] carries (device), from the flat open field
    `o`: the mean over its tracks of the smallest factor on each (the section's comment). A step
    with 0 is not in the operator."""
    xp = p.solver.xp
    flat, _keep = _track_factors(grid, o, p)
    for ax in _track_axes_of(grid, p.K, xp):
        yield _track_means(flat, ax, xp)


class StepGrad:
    """The widths' gradient (step_gradient): add(k, coef), coef dJ / d(share) per edge of axis k,
    shares each edge's coefficient over its tracks, each track's share split equally among the
    sub-samples tied at its minimum, into dJ / d(sub-sample factor); open_grad() takes that
    through p.cells.sub_vjp to dJ / d o per cell (flat)."""

    def __init__(self, grid, o, p: Params, axes_: list[_AxisTracks], flat, keep):
        self.grid, self.o, self.p, self.axes, self.flat = grid, o, p, axes_, flat
        self.keep = keep
        self.G = p.solver.xp.zeros(flat.size)

    def add(self, k: int, coef) -> None:
        xp, ax = self.p.solver.xp, self.axes[k]
        c = coef / len(ax.tracks)
        if ax.coarse is not None:
            c = xp.where(ax.coarse, 0.0, c)
        for tr in ax.tracks:
            idx = [ax.base[slot] + sub for slot, sub in tr]
            vals = xp.stack([self.flat[i] for i in idx])
            act = vals <= vals.min(axis=0)[None, :]
            share = c / act.sum(axis=0)
            self.G += xp.bincount(xp.concatenate([i[a_] for i, a_ in zip(idx, act, strict=True)]),
                                  xp.concatenate([share[a_] for a_ in act]),
                                  minlength=self.G.size)

    def open_grad(self):
        g, S2 = self.grid, self.grid.S * self.grid.S
        n = g.inside.size
        A = self.G[:n * S2].reshape(n, S2) * self.keep
        return self.p.cells.sub_vjp(g, self.o.reshape(g.inside.shape), A, self.p.solver).ravel()


def step_gradient(grid: Grid | CompositeGrid, o, p: Params) -> StepGrad:
    """An accumulator of the widths' gradient at the flat open field `o` (StepGrad)."""
    flat, keep = _track_factors(grid, o, p)
    return StepGrad(grid, o, p, _track_axes_of(grid, p.K, p.solver.xp), flat, keep)


def reach(grid: Grid | CompositeGrid, o, p: Params):
    """(the grid's shape) device mask: the open cells (o > 0) in a component of the operator's
    cell graph -- its along edges, those with a width above 0 -- holding a ground cell: the cells
    with a path to the street. On the host (scipy); kept for a free set that repeats (the eps
    world's)."""
    xp = p.solver.xp
    o = o.ravel()
    free = o > 0
    ab = []
    for (a, b, _line, _wk), frac in zip(grid.pattern(p.K, xp).along, step_widths(grid, o, p),
                                        strict=True):
        ok = frac > 0
        ab.append((a[ok], b[ok]))
    key = (int(free.sum()), sum(len(a) for a, _ in ab), sum(int(a.sum()) for a, _ in ab),
           sum(int(b.sum()) for _, b in ab))
    store = _repeated(grid.cache, "edge reach")
    if key in store:
        return store[key]
    th = p.solver.to_host
    a = np.concatenate([th(a) for a, _ in ab])
    b = np.concatenate([th(b) for _, b in ab])
    n = grid.inside.size
    _nc, lab = connected_components(
        sp.coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n)), directed=False)
    fh = th(free)
    ok = np.zeros(int(lab.max()) + 1, dtype=bool)
    ok[lab[fh & grid.ground.ravel()]] = True
    out = xp.asarray((ok[lab] & fh).reshape(grid.inside.shape))
    _remember(grid.cache, "edge reach", key, out)
    return out


def components(grid: Grid | CompositeGrid, o, p: Params) -> NDArray[np.int64]:
    """(cells flat) on the host: each open cell's component of the operator's cell graph in the
    flat open field `o` (its steps with a width above 0), numbered 0 .. from the first open cell;
    -1 where closed."""
    xp, th = p.solver.xp, p.solver.to_host
    free = th(o.ravel() > 0)
    a, b = [], []
    for (ea, eb, _line, _wk), frac in zip(grid.pattern(p.K, xp).along, step_widths(grid, o, p),
                                          strict=True):
        ok = frac > 0
        a.append(th(ea[ok]))
        b.append(th(eb[ok]))
    n = grid.inside.size
    _nc, lab = connected_components(
        sp.coo_matrix((np.ones(sum(len(x) for x in a), dtype=np.int8),
                       (np.concatenate(a), np.concatenate(b))), shape=(n, n)), directed=False)
    uniq, comp = np.unique(lab[free], return_inverse=True)
    out = -np.ones(n, dtype=np.int64)
    out[free] = comp
    return out


def demand_cells(grid: Grid | CompositeGrid, open0: NDArray[np.bool_],
                 F0: NDArray[np.float64]) -> NDArray[np.bool_]:
    """(cells flat) on the host, from the open cells `open0` (flat) and their sub-samples' factors
    F0 (host): the cells that can take demand, whatever the reach: open, and not split. Free first
    frees a home through these cells' components, through pieces elsewhere (`regions`)."""
    return open0 & _one_part(F0, grid)


def regions(grid: Grid | CompositeGrid, open0: NDArray[np.bool_], bf: sp.csr_matrix,
            lab: NDArray[np.int64], p: Params) -> Regions:
    """Free first's Regions at baseline, on the host (NOTES, "The grid fix"), from open0 (flat,
    the cells open at baseline), bf (buildings x inside cells, the share of each cell each building
    covers) and lab ((cells, S*S) the building holding each sub-sample, -1 none): the open
    components at baseline by the operator's own steps; the pieces of a building its incidences
    (building, cell) joined by steps some track of which is blocked by it alone (or by nothing);
    and adjacency only where a clearing really joins two regions, read off each step's tracks at
    baseline, so the graph claims no path the operator lacks:
    - a cell is one node: its component (if it has an open sub-sample) and the piece of every
      building covering part of it are adjacent (RegionGraph's shared cell), in a split cell
      only the pieces whose building, cleared alone, joins the anchor's part;
    - a track whose closed sub-samples all belong to one building j joins, once j is cleared, the
      nodes of the step's two end cells, so every region of the two (their components, j's
      pieces there) is made adjacent to j's piece in an end cell, or, where j covers neither end
      (only cells the step crosses), to a gadget owned by j that stands for this crossing alone;
    - a track blocked by exactly two buildings, each with a piece in an end cell, joins those two
      pieces (entering both pays for both);
    - a track through a sub-sample outside the block, or blocked by three or more, joins nothing.
    The street steps into the components holding a ground cell and the pieces covering one. A home
    is freed through a near cell's component only where the cell can take demand at baseline,
    else through the pieces there whose building, cleared alone, lets it (R): a split cell's node
    reaching the street frees no home until its parts are one."""
    th = p.solver.to_host
    n, S2 = grid.inside.size, grid.S * grid.S
    xp = p.solver.xp
    o0 = xp.asarray(grid.ff0.ravel())
    comp = components(grid, o0, p)
    ncomp = int(comp.max()) + 1 if (comp >= 0).any() else 0
    # each sub-sample's building (>= 0), open (-1) or outside the block (-2); the dummy cell -2
    isub = grid.isub.reshape(n, S2)
    code = np.where(isub, np.where(lab >= 0, lab, -1), -2).astype(np.int64)
    F0 = th(p.cells.sub_factor(grid, o0.reshape(grid.inside.shape), p.solver))
    code = _never_open(np.concatenate([code.ravel(), np.full(S2, -2, dtype=np.int64)]), F0, grid)
    big = np.iinfo(np.int64).max
    # incidences (building, cell): one per pair, from bf (buildings x inside cells)
    bc = bf.tocoo()
    inc_cell = np.flatnonzero(grid.inside.ravel())[bc.col]
    inc_bld = bc.row.astype(np.int64)
    o = np.lexsort((inc_cell, inc_bld))
    inc_bld, inc_cell = inc_bld[o], inc_cell[o]
    keep = np.ones(len(o), dtype=bool)
    keep[1:] = (np.diff(inc_bld) != 0) | (np.diff(inc_cell) != 0)
    inc_bld, inc_cell = inc_bld[keep], inc_cell[keep]
    ninc = len(inc_bld)
    inc_key = inc_bld * (n + 1) + inc_cell                 # sorted
    # one building cleared alone in a cell it covers part of (_cleared_alone): whether its
    # sub-samples then join the part the cell's node conducts through (only then is its piece
    # there joined to the cell's component), and whether the cell can then take demand (a home
    # is freed through a cell that cannot at baseline, closed or split, only by such a piece)
    jbits = ((lab[inc_cell] == inc_bld[:, None]).astype(np.int64)
             << np.arange(S2, dtype=np.int64)[None, :]).sum(axis=1)
    joins, frees = _cleared_alone(inc_cell, jbits, F0, grid)
    takes = demand_cells(grid, open0, F0)

    def inc_of(j: NDArray[np.int64], x: NDArray[np.int64]) -> NDArray[np.int64]:
        """The incidence (j, x) for each pair, -1 where j covers no part of x."""
        if ninc == 0:
            return -np.ones(len(j), dtype=np.int64)
        key = j * (n + 1) + x
        pos = np.minimum(np.searchsorted(inc_key, key), ninc - 1)
        return np.where(inc_key[pos] == key, pos, -1)

    # per track of every step: its end cells and what blocks it (-1 nothing, j one building, or
    # the pair (j, k), j < k); a step with a coarse end is open
    single: list[tuple] = []                # (u, v, j)
    double: list[tuple] = []                # (u, v, j, k)
    opened: list[tuple] = []                # (u, v): a track open at baseline
    for (ea, eb, _line, _wk), ax in zip(grid.pattern(p.K, xp).along,
                                        _track_axes_of(grid, p.K, xp), strict=True):
        u, v = th(ea), th(eb)
        base = [th(q) for q in ax.base]
        fine = np.ones(len(u), dtype=bool) if ax.coarse is None else ~th(ax.coarse)
        for tr in ax.tracks:
            lo = np.full(len(u), big, dtype=np.int64)
            hi = np.full(len(u), -1, dtype=np.int64)
            out = np.zeros(len(u), dtype=bool)
            cs = []
            for slot, sub in tr:
                c = code[base[slot] + sub]
                cs.append(c)
                out |= c == -2
                lo = np.minimum(lo, np.where(c >= 0, c, big))
                hi = np.maximum(hi, c)
            ok = fine & ~out
            third = np.zeros(len(u), dtype=bool)
            for c in cs:
                third |= (c >= 0) & (c != lo) & (c != hi)
            is_open = ok & (hi < 0)
            one = ok & (hi >= 0) & (lo == hi)
            two = ok & (hi >= 0) & (lo < hi) & ~third
            opened.append((u[is_open], v[is_open]))
            single.append((u[one], v[one], hi[one]))
            double.append((u[two], v[two], lo[two], hi[two]))

    def cat(parts: list[tuple], i: int) -> NDArray[np.int64]:
        return np.concatenate([q[i] for q in parts]).astype(np.int64)

    su, sv, sj = cat(single, 0), cat(single, 1), cat(single, 2)
    if len(su):
        k = np.unique(np.stack([su, sv, sj], axis=1), axis=0)
        su, sv, sj = k[:, 0], k[:, 1], k[:, 2]
    du, dv, dj, dk = (cat(double, i) for i in range(4))
    ou, ov = cat(opened, 0), cat(opened, 1)
    # the pieces: incidences joined by a track blocked by their building alone or open
    pa, pb = [], []
    iu, iv = inc_of(sj, su), inc_of(sj, sv)
    both = (iu >= 0) & (iv >= 0)
    pa.append(iu[both])
    pb.append(iv[both])
    holds = np.zeros(n + 1, dtype=bool)
    holds[inc_cell] = True
    sel = holds[ou] & holds[ov]
    if sel.any():
        # an open track between two cells joins the pieces of every building covering part of
        # both: Inc (cells x buildings) holds each incidence's id + 1
        e = np.unique(np.stack([ou[sel], ov[sel]], axis=1), axis=0)
        inc = sp.csr_matrix((np.arange(1, ninc + 1), (inc_cell, inc_bld)),
                            shape=(n + 1, int(inc_bld.max()) + 1))
        Iu, Iv = inc[e[:, 0]], inc[e[:, 1]]
        A = Iu.multiply(Iv > 0).tocsr()
        Bm = Iv.multiply(Iu > 0).tocsr()
        A.sort_indices()
        Bm.sort_indices()
        pa.append(np.asarray(A.data, dtype=np.int64) - 1)
        pb.append(np.asarray(Bm.data, dtype=np.int64) - 1)
    pa_, pb_ = np.concatenate(pa), np.concatenate(pb)
    npiece, piece = connected_components(
        sp.coo_matrix((np.ones(len(pa_), dtype=np.int8), (pa_, pb_)), shape=(ninc, ninc)),
        directed=False)
    owner_piece = np.empty(npiece, dtype=np.int64)
    owner_piece[piece] = inc_bld
    reg_piece = ncomp + piece                              # each incidence's region
    links_a, links_b = [], []
    # a cell is one node: its component and the pieces there whose building, cleared, joins the
    # part the node conducts through
    open_inc = open0[inc_cell] & joins
    links_a.append(comp[inc_cell[open_inc]])
    links_b.append(reg_piece[open_inc])
    order = np.argsort(inc_cell, kind="stable")
    same = inc_cell[order][1:] == inc_cell[order][:-1]
    links_a.append(reg_piece[order][1:][same])             # consecutive pieces of one cell (a
    links_b.append(reg_piece[order][:-1][same])            # chain joins them all)
    # one building blocks the track: the end cells' regions to its piece in an end, or a gadget
    cu = np.where(open0[su], comp[su], -1)
    cv = np.where(open0[sv], comp[sv], -1)
    pu = np.where(iu >= 0, reg_piece[np.maximum(iu, 0)], -1)
    pv = np.where(iv >= 0, reg_piece[np.maximum(iv, 0)], -1)
    hub = np.where(pu >= 0, pu, pv)                        # j's piece in an end (same if both)
    has = hub >= 0
    for side in (cu, cv, pu, pv):
        m = has & (side >= 0) & (side != hub)
        links_a.append(side[m])
        links_b.append(hub[m])
    gad = ~has & (cu >= 0) & (cv >= 0) & (cu != cv)
    g_key = np.unique(np.stack([sj[gad], np.minimum(cu, cv)[gad], np.maximum(cu, cv)[gad]],
                               axis=1), axis=0) if gad.any() else np.zeros((0, 3), np.int64)
    ngad = len(g_key)
    reg_gad = ncomp + npiece + np.arange(ngad)
    links_a += [g_key[:, 1], g_key[:, 2]]
    links_b += [reg_gad, reg_gad]
    # two buildings block the track, each with a piece in an end cell
    pj = [inc_of(dj, x) for x in (du, dv)]
    pk = [inc_of(dk, x) for x in (du, dv)]
    aj = np.where(pj[0] >= 0, pj[0], pj[1])
    ak = np.where(pk[0] >= 0, pk[0], pk[1])
    m = (aj >= 0) & (ak >= 0)
    links_a.append(reg_piece[aj[m]])
    links_b.append(reg_piece[ak[m]])
    la, lb = np.concatenate(links_a), np.concatenate(links_b)
    keep = la != lb
    la, lb = la[keep], lb[keep]
    adj = np.unique(np.stack([np.concatenate([la, lb]), np.concatenate([lb, la])], axis=1), axis=0)
    ground = grid.ground.ravel()
    entry = np.concatenate([np.unique(comp[open0 & ground]),
                            np.unique(reg_piece[ground[inc_cell]])])
    oc = np.flatnonzero(takes)
    via = takes[inc_cell] | frees
    R = sp.csr_matrix((np.ones(len(oc) + int(via.sum())),
                       (np.concatenate([oc, inc_cell[via]]),
                        np.concatenate([comp[oc], reg_piece[via]]))),
                      shape=(n, ncomp + npiece + ngad))
    return Regions(ncomp=ncomp, owner=np.concatenate([owner_piece, g_key[:, 0]]),
                   a=adj[:, 0], b=adj[:, 1], entry=entry, R=R)


class Conductance(NamedTuple):
    """A run's conductance: what an open cell conducts by and the per-heading multipliers (its
    steps conducting by their open width, step_widths). `name` the run's, the spec conductance_of
    reads back."""
    cells: CellConductance
    along: AlongConductance

    @property
    def name(self) -> str:
        return (self.along.name + (f"+{self.cells.name}" if self.cells.name else "")
                + f"+{APERTURE}")


def conductance_of(spec: str, scans: Scans) -> Conductance:
    """`<along>[+<cells>]+apwfs` (APERTURE last): along_of; cells_of (none: every open cell by
    its open fraction, OpenFraction). uni+apwfs, uni+b1+apwfs, uni+d0.5+apwfs, ss100k2n2r30+apwfs,
    ss100k2n2r30t+apwfs. A name without APERTURE is the step rule's before 2026-10-13: it
    raises."""
    *head, last = spec.split("+")
    if last != APERTURE or not 1 <= len(head) <= 2 or any(not t for t in head):
        raise ValueError(f"unknown conductance {spec!r}: <along>[+<cells>]+{APERTURE}")
    return Conductance(cells=cells_of(head[1]) if len(head) == 2 else OpenFraction(),
                       along=along_of(head[0], scans))


class Solver(Protocol):
    """Where the operator lives and how a grounded SPD system on it is solved. `xp`, `sparse`
    and `ndimage` are the device's array, sparse-matrix and image modules (numpy / scipy, or
    cupy / cupyx). prepare(A, coarsening) does the setup once and returns solve(b, rtol, x0) ->
    x on the device, CG from x0 (None: zero) to a relative residual of rtol (exact below SMALL
    unknowns). `coarsening`
    lets a solver reuse one aggregation across a block's systems."""

    @property
    def xp(self): ...

    @property
    def sparse(self): ...

    @property
    def ndimage(self): ...

    def to_host(self, a) -> NDArray: ...

    def release(self) -> None: ...      # return the device pool's cached free blocks

    def prepare(self, A, coarsening: Coarsening) -> Callable: ...


SMALL = 20000
MAXITER = 2000      # rtol 1e-9 takes ~350 K-cycle iterations on a 1M-unknown block


class Coarsening(NamedTuple):
    """What a system offers a solver that coarsens once per block: `mid()` its unknowns' ids
    in the master numbering (every inside non-ground cell x K), `master()` the operator on every
    inside cell (on the device, uniform along-factors), and `cache`, a per-grid dict for what
    the solver derives from it."""
    mid: Callable
    master: Callable
    cache: dict


@dataclass(frozen=True)
class CpuAMG:
    """pyamg smoothed aggregation as a CG preconditioner (Gauss-Seidel smoothing), one core,
    set up afresh for every system."""

    @property
    def xp(self):
        return np

    @property
    def sparse(self):
        return sp

    @property
    def ndimage(self):
        return ndimage

    def to_host(self, a):
        return np.asarray(a)

    def release(self) -> None:
        """Nothing to release: numpy has no pool."""

    def prepare(self, A, coarsening):
        if A.shape[0] < SMALL:
            lu = factorized(A.tocsc())
            return lambda b, rtol, x0=None: lu(b)
        ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric", max_coarse=2000)

        def solve1(b, rtol, x0):
            res: list[float] = []
            x = ml.solve(b, x0=x0, tol=rtol, accel="cg", maxiter=500, residuals=res)
            nb = float(np.linalg.norm(b))
            if res[-1] > rtol * nb * 10:         # relative to b, as pyamg's tol (x0 may be close)
                raise RuntimeError(f"AMG-CG did not converge: {res[-1] / nb:.2e} after "
                                   f"{len(res)}")
            return x

        def solve(b, rtol, x0=None):
            if b.ndim == 1:
                return solve1(b, rtol, x0)
            return np.stack([solve1(b[:, i], rtol, None if x0 is None else x0[:, i])
                             for i in range(b.shape[1])], axis=1)
        return solve


def aggregate_level(A) -> tuple[np.ndarray, int]:
    """One level of pyamg's standard aggregation over A's (symmetric, theta 0) strength graph:
    each node's aggregate and the aggregate count. An isolated node -- no off-diagonal entry, so
    no strong neighbour -- is left in no aggregate by pyamg; it gets one of its own (decoupled,
    it is solved exactly at the coarse level). First met on 53556 at h 0.5."""
    from pyamg.aggregation.aggregate import standard_aggregation
    from pyamg.strength import symmetric_strength_of_connection
    Ag = sp.csr_matrix(standard_aggregation(symmetric_strength_of_connection(A, theta=0.0))[0])
    counts = np.diff(Ag.indptr)
    assert (counts <= 1).all(), "a node in two aggregates"
    agg = np.empty(A.shape[0], dtype=np.int64)
    agg[counts == 1] = Ag.indices
    lone = np.flatnonzero(counts == 0)
    if len(lone):
        print(f"  AMG: {len(lone)} isolated unknown(s) of {A.shape[0]}, an aggregate each",
              flush=True)
        agg[lone] = Ag.shape[1] + np.arange(len(lone))
    return agg, Ag.shape[1] + len(lone)


@dataclass(frozen=True)
class GpuAMG:
    """Flexible CG on the GPU preconditioned by an unsmoothed-aggregation K-cycle (Notay: each
    coarse correction is two flexible Krylov steps preconditioned by the next level): damped
    Jacobi smoothing, `sweeps` each side (Gauss-Seidel is sequential), omega = 4 / (3 rho(D^-1
    A)) per level (rho by power iteration, padded 5%), a dense inverse on the coarsest level.
    The aggregates are the block's MASTER coarsening (pyamg's standard aggregation of the
    operator on every inside cell, once per grid; strength theta 0, so the sparsity pattern
    alone) restricted to the system's unknowns, so a system's setup is GPU Galerkin sums only.
    Measured and dropped (NOTES, "Speed"): V-cycles (2.2x the time), strength theta 0.25
    (3-7x), Notay's pairwise aggregation (3-10x), smoothed prolongators (cuSPARSE's products
    need more memory than the card on 5810). Needs cupy and CUDA_PATH (the toolkit
    headers)."""
    sweeps: int = 1
    single: bool = True

    @property
    def xp(self):
        import cupy
        return cupy

    @property
    def sparse(self):
        import cupyx.scipy.sparse
        return cupyx.scipy.sparse

    @property
    def ndimage(self):
        import cupyx.scipy.ndimage
        return cupyx.scipy.ndimage

    def to_host(self, a):
        return self.xp.asnumpy(a)

    def release(self) -> None:
        """Hand cupy's cached free blocks back to the device. A new system's CSR build sorts a
        key per nonzero (250M on 30796): left to the pool, those temporaries are carved out of
        the freed chunks of earlier systems, which splits them, and on the largest blocks the
        split chunks ran a 48 GB card out with ~10 GB of them free (NOTES, "Translucent greedy
        on 30796")."""
        self.xp.get_default_memory_pool().free_all_blocks()

    def _dinv(self, A, v0=None):
        """(The damped inverse diagonal omega D^-1 of one level, the power iteration's last
        vector): 30 iterations from random, 5 from v0 (the last system's: the same pattern, values
        moved a little)."""
        cp = self.xp
        dinv = 1.0 / A.diagonal()
        v = cp.random.default_rng(0).random(A.shape[0]) if v0 is None else v0
        rho = 1.0
        for _ in range(30 if v0 is None else 5):
            w = dinv * (A @ v)
            rho = float(cp.linalg.norm(w)) / float(cp.linalg.norm(v))
            v = w / cp.linalg.norm(w)
        return dinv * (4.0 / (3.0 * 1.05 * rho)), v

    def _structure(self, A, coarsening: Coarsening) -> list:
        """Per level, everything about the Galerkin coarsening but the values: R, P, and the
        coarse operator's CSR structure with `map`, each fine nonzero's coarse nonzero (A_c =
        T^T A T with T the 0/1 aggregation sums entries by aggregate pair)."""
        cp, cs = self.xp, self.sparse
        levels, ids, ip, ix, n = [], coarsening.mid(), A.indptr, A.indices, A.shape[0]
        for agg in self._aggregates(coarsening):
            # the master aggregates of this level's unknowns, renumbered over those present
            uniq, local = cp.unique(agg[ids], return_inverse=True)
            local = local.ravel()
            nc = len(uniq)
            R = cs.csr_matrix((cp.ones(n), (local, cp.arange(n))), shape=(nc, n))
            rows = cp.searchsorted(ip, cp.arange(len(ix)), side="right") - 1
            keys = local[rows].astype(cp.int64) * nc + local[ix]
            del rows
            ukeys, inv = cp.unique(keys, return_inverse=True)
            del keys
            indptr = cp.zeros(nc + 1, dtype=cp.int32)
            indptr[1:] = cp.cumsum(cp.bincount(ukeys // nc, minlength=nc))
            indices = (ukeys % nc).astype(cp.int32)
            levels.append(dict(R=R, P=R.T.tocsr(), map=inv.ravel().astype(cp.int32),
                               indices=indices, indptr=indptr, nc=nc, v=None))
            ip, ix, n, ids = indptr, indices, nc, uniq
        return levels

    def _aggregates(self, co: Coarsening) -> list:
        """Per level, the master aggregate of each node (device), cached per grid."""
        key = "aggregates"
        if key not in co.cache:
            A = co.master().get().tocsr()
            aggs = []
            while A.shape[0] > 2000:
                agg, nc = aggregate_level(A)
                co_ = A.tocoo()
                A = sp.csr_matrix((co_.data, (agg[co_.row], agg[co_.col])), shape=(nc, nc))
                aggs.append(self.xp.asarray(agg))
            co.cache[key] = aggs
        return co.cache[key]

    def prepare(self, A, coarsening):
        cp, cs = self.xp, self.sparse
        # cupy's pool keeps every freed block; each system builds new matrices, so without this a
        # process only grows (~20 GB per 5810 run; two big blocks then exhaust the 48 GB card)
        cp.get_default_memory_pool().free_all_blocks()
        if A.shape[0] < SMALL:
            lu = factorized(A.get().tocsc())
            return lambda b, rtol, x0=None: cp.asarray(lu(cp.asnumpy(b)))
        # the hierarchy's structure depends only on A's pattern, which an eps-world system
        # repeats on every call (SIMP's updates, the greedy's tension): kept once it repeats
        key = (A.shape[0], A.nnz, int(A.indices.sum()), int(A.indptr.sum()))
        store = _repeated(coarsening.cache, "galerkin")
        if key in store:
            levels = store[key]
        else:
            levels = self._structure(A, coarsening)
            _remember(coarsening.cache, "galerkin", key, levels)
        lv, Ai = [], A
        for L in levels:
            dinv, L["v"] = self._dinv(Ai, L["v"])
            lv.append(dict(A=Ai, dinv=dinv, R=L["R"], P=L["P"]))
            Ai = cs.csr_matrix((cp.bincount(L["map"], weights=Ai.data,
                                            minlength=len(L["indices"])),
                                L["indices"], L["indptr"]), shape=(L["nc"], L["nc"]))
        coarse = cp.linalg.inv(Ai.toarray())
        if self.single:
            # the cycle in single precision (sums above in double); the outer CG's residual
            # stays in double, so the answer meets rtol as before, only the preconditioner is
            # cheaper (the solves are memory-bound: 8 -> 4 bytes per value)
            f4 = cp.float32
            lv = [dict(A=cs.csr_matrix((L["A"].data.astype(f4), L["A"].indices, L["A"].indptr),
                                       shape=L["A"].shape),
                       dinv=L["dinv"].astype(f4), R=L["R"].astype(f4), P=L["P"].astype(f4))
                  for L in lv]
            coarse = coarse.astype(f4)
        kc = _KCycle(lv, coarse, self.sweeps)

        def solve(b, rtol, x0=None):
            return _pcg(A, lambda r: kc.cycle(0, r.astype(kc.dt)).astype(cp.float64), b, rtol,
                        cp, x0)
        return solve


class _KCycle:
    """A GpuAMG system's preconditioner: the K-cycle over its levels (each a dict of A, dinv,
    R, P) down to the dense inverse `coarse`. A class and not nested closures: `cycle` and
    `krylov2` call each other, and as closures they form a reference cycle that kept every
    system's levels alive until the cycle collector ran (2.6 GB per system on 30796, piling up
    over greedy steps until the 48 GB card ran out; NOTES, "Translucent greedy on 30796")."""

    def __init__(self, lv: list, coarse, sweeps: int):
        self.lv, self.coarse, self.sweeps = lv, coarse, sweeps
        self.dt = coarse.dtype

    @staticmethod
    def dot(a, b):
        """Accumulated in double: in single the K-cycle's two-step coefficients lose
        everything to cancellation (rho2 = bet - gam^2 / rho1) and the solve goes NaN."""
        return (a * b).sum(axis=0, dtype=np.float64)

    def cycle(self, i, b):
        lv = self.lv
        if i == len(lv):
            return self.coarse @ b
        L = lv[i]
        d = L["dinv"] if b.ndim == 1 else L["dinv"][:, None]
        x = d * b
        for _ in range(self.sweeps - 1):
            x = x + d * (b - L["A"] @ x)
        rc = L["R"] @ (b - L["A"] @ x)
        ec = self.krylov2(i + 1, rc) if i + 1 < len(lv) else self.cycle(i + 1, rc)
        x = x + L["P"] @ ec
        for _ in range(self.sweeps):
            x = x + d * (b - L["A"] @ x)
        return x

    def krylov2(self, j, b):
        """Two flexible CG steps on level j from zero, preconditioned by its cycle (Notay)."""
        dot, dt = self.dot, self.dt
        Aj = self.lv[j]["A"]
        c1 = self.cycle(j, b)
        v1 = Aj @ c1
        rho1, a1 = dot(c1, v1), dot(c1, b)
        r1 = b - (a1 / rho1).astype(dt) * v1
        c2 = self.cycle(j, r1)
        v2 = Aj @ c2
        gam, bet, a2 = dot(c2, v1), dot(c2, v2), dot(c2, r1)
        rho2 = bet - gam * gam / rho1
        return ((a1 / rho1 - gam * a2 / (rho1 * rho2)).astype(dt) * c1
                + (a2 / rho2).astype(dt) * c2)


def _pcg(A, M, b, rtol: float, xp, x0=None):
    """Flexible (Polak-Ribiere) preconditioned CG from x0 (None: zero) to ||r|| <= rtol ||b||;
    b may be (n,) or (n, m) (columns solved independently, in lockstep). M may vary between
    calls."""
    nb = xp.linalg.norm(b, axis=0)
    if not bool((nb > 0).any()):
        return xp.zeros_like(b)
    x = xp.zeros_like(b) if x0 is None else x0.copy()
    r = b.copy() if x0 is None else b - A @ x
    if bool((xp.linalg.norm(r, axis=0) <= rtol * nb).all()):
        return x
    z = M(r)
    pdir = z.copy()
    rz = (r * z).sum(axis=0)
    for _ in range(MAXITER):
        Ap = A @ pdir
        pAp = (pdir * Ap).sum(axis=0)
        alpha = xp.where(pAp > 0, rz / xp.where(pAp > 0, pAp, 1.0), 0.0)
        x += alpha * pdir
        r_old = r.copy()
        r -= alpha * Ap
        if bool((xp.linalg.norm(r, axis=0) <= rtol * nb).all()):
            return x
        z = M(r)
        num = (z * (r - r_old)).sum(axis=0)
        pdir = z + xp.where(rz > 0, num / xp.where(rz > 0, rz, 1.0), 0.0) * pdir
        rz = (r * z).sum(axis=0)
    worst = float((xp.linalg.norm(r, axis=0) / xp.where(nb > 0, nb, 1.0)).max())
    raise RuntimeError(f"AMG-CG did not converge: {worst:.2e} after {MAXITER}")


def scans_of(spec: str) -> Scans:
    """`cpu` -> CpuScans(); `gpu` -> GpuScans() (chosen with the solver)."""
    if spec == "cpu":
        return CpuScans()
    if spec == "gpu":
        return GpuScans()
    raise ValueError(f"unknown scans {spec!r}")


def solver_of(spec: str) -> Solver:
    """`cpu` -> CpuAMG(); `gpu` -> GpuAMG()."""
    if spec == "cpu":
        return CpuAMG()
    if spec == "gpu":
        return GpuAMG()
    raise ValueError(f"unknown solver {spec!r}")


@dataclass(frozen=True)
class Params:
    ell_m: float = 2.0       # turning length: metres travelled per radian of heading change
    K: int = 8
    along: AlongConductance = Uniform()
    solver: Solver = CpuAMG()
    # what an open cell conducts by (OpenFraction: today's): required, so every run names it
    cells: CellConductance = dataclasses.field(kw_only=True)


class Pattern(NamedTuple):
    """Every edge the operator can have on a mesh: those among its `inside` cells (any field's
    open cells are inside ones). Per axis k, along[k] = (a, b, line, wk): flat cell indices of
    each edge's ends (b ahead of a along v_k) and of the cells the step crosses (n_line, E), and
    the weight of a fully open edge (a number where every edge has the same, m_k / |v_k|^2, else
    per edge); `cells`: the inside cells, flat. Arrays live on the solver's device."""
    along: list
    cells: object


def _free_ids(o, xp):
    """Flat (ny * nx,) -> free-cell id (row-major order of o > 0) or -1, and the count."""
    free = o.ravel() > 0
    nf = int(free.sum())
    cell = -xp.ones(free.shape, dtype=xp.int64)
    cell[free] = xp.arange(nf)
    return cell, nf


def cell_factor(grid: Grid | CompositeGrid, o, p: Params):
    """`p.cells`'s factor of the flat open field `o` (device), flat: what each open cell conducts
    by (OpenFraction: `o` itself)."""
    return p.cells.factor(grid, o.reshape(grid.inside.shape), p.solver).ravel()


def along_edges(grid: Grid, open_, p: Params):
    """Per axis k: (k, a, b, w) with a, b open-cell ids (cell order of `open_ > 0`), b ahead
    of a along v_k; w is m_k / |v_k|^2 scaled by the share of its width the step conducts (its
    open width, read on the sub-samples: step_widths), kept where above 0, and by the smaller of
    the two ends' `p.along` factors. Device arrays."""
    xp = p.solver.xp
    yield from _along(grid, xp.asarray(open_, dtype=xp.float64).ravel(), p)


def _along(grid: Grid, o, p: Params):
    """along_edges on the flat open field `o` (device)."""
    xp = p.solver.xp
    cell, _nf = _free_ids(o, xp)
    pat = grid.pattern(p.K, xp)
    F = p.along.layers(grid, o.reshape(grid.inside.shape), p.K)
    for k, frac in enumerate(step_widths(grid, o, p)):
        a, b, _line, wk = pat.along[k]
        ok = frac > 0
        a, b, frac = a[ok], b[ok], frac[ok]
        w = frac * (wk if np.isscalar(wk) else wk[ok])
        if not np.isscalar(F[k]):
            Fk = xp.asarray(F[k]).ravel()
            w = w * xp.minimum(Fk[a], Fk[b])
        elif F[k] != 1.0:
            w = w * F[k]
        yield k, cell[a], cell[b], w


def crossing(grid: Grid, open_: NDArray[np.float64], sol: Solution,
             p: Params) -> dict[str, float]:
    """How much of the along-heading power flows where streams of different headings share a
    cell. Per free cell, J = sum_k F_k e_k (the net flux vector) and S = sum_k |F_k|;
    cancellation 1 - |J|/S is ~0 for one stream (a little above 0 for a stream spread over
    neighbouring axes), 0.29 for two equal perpendicular streams, 1 for opposed ones. Reported
    power-weighted over cells. (Host arrays: a CPU-solver `p`.)"""
    v, _th, _m, _gap = axes(p.K)
    e = v / np.linalg.norm(v, axis=1)[:, None]
    nc = int((open_ > 0).sum())
    uk = np.zeros((nc, p.K))
    live = sol.unk_cell >= 0
    uk[live] = sol.u.reshape(-1, p.K)
    J = np.zeros((nc, 2))
    S = np.zeros(nc)
    pw = np.zeros(nc)
    for k, a, b, w in along_edges(grid, open_, p):
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


def edges(grid: Grid, open_, p: Params):
    """Every edge of the operator as families (ca, cb, ka, kb, w): free-cell ids and axes of the
    two ends. Along-axis edges per axis (along_edges: their width step_widths'), then the turning
    edges (k, k+1) inside every free cell, scaled by its cell factor (`p.cells`; OpenFraction's its
    open fraction, its open area) and area (its volume), and by `p.along`'s turning factor
    (TurnBoosted's; 1 otherwise). Ground is applied by `operator`. Device arrays."""
    xp = p.solver.xp
    _v, _th, _m, gap = axes(p.K)
    o = xp.asarray(open_, dtype=xp.float64).ravel()
    kap = cell_factor(grid, o, p)
    for k, a, b, w in _along(grid, o, p):
        yield a, b, xp.full(len(a), k), xp.full(len(a), k), w
    free = o > 0
    vol = kap[free]
    c = xp.arange(len(vol))
    area = grid.cell_area(xp, free)
    T = p.along.turning(grid, o.reshape(grid.inside.shape), p.K)
    for k in range(p.K):
        w = vol * (area / (p.ell_m ** 2 * gap[k]))
        if not np.isscalar(T[k]):
            w = w * xp.asarray(T[k]).ravel()[free]
        elif T[k] != 1.0:
            w = w * T[k]
        yield c, c, xp.full(len(c), k), xp.full(len(c), (k + 1) % p.K), w


def unknown_of(ground, open_, xp=np):
    """Per free cell: its unknown block index (x K + axis), or -1 on ground (u = 0 there)."""
    is_g = xp.asarray(ground)[xp.asarray(open_) > 0]
    unk_cell = -xp.ones(len(is_g), dtype=xp.int64)
    unk_cell[~is_g] = xp.arange(int((~is_g).sum()))
    return unk_cell


def _assemble(grid: Grid, open_, p: Params):
    """(off-diagonal rows, cols, vals; diagonal; free-cell id per grid cell (ny, nx); unknown
    index per free cell or -1) of L over every unknown. An edge with a ground end only adds to
    the other end's diagonal."""
    xp = p.solver.xp
    K = p.K
    o = xp.asarray(open_, dtype=xp.float64)
    cell = _free_ids(o, xp)[0].reshape(o.shape)
    unk_cell = unknown_of(grid.ground, o, xp)
    N = int((unk_cell >= 0).sum()) * K
    rows, cols, vals = [], [], []
    diag = xp.zeros(N)
    for ca, cb, ka, kb, w in edges(grid, o, p):
        ua, ub = unk_cell[ca], unk_cell[cb]
        ia, ib = ua * K + ka, ub * K + kb
        ga, gb = ua < 0, ub < 0
        both = ~ga & ~gb
        rows += [ia[both], ib[both]]
        cols += [ib[both], ia[both]]
        vals += [-w[both], -w[both]]
        diag += xp.bincount(ia[~ga], weights=w[~ga], minlength=N)
        diag += xp.bincount(ib[~gb], weights=w[~gb], minlength=N)
    return xp.concatenate(rows), xp.concatenate(cols), xp.concatenate(vals), diag, cell, unk_cell


def operator(grid: Grid, open_, p: Params):
    """(L restricted to unknowns, free-cell id per grid cell, unknown index per free cell or -1).
    Cells are those with open fraction > 0."""
    xp = p.solver.xp
    r, c, v, diag, cell, unk_cell = _assemble(grid, open_, p)
    N = len(diag)
    ar = xp.arange(N)
    L = p.solver.sparse.coo_matrix((xp.concatenate([v, diag]),
                                    (xp.concatenate([r, ar]), xp.concatenate([c, ar]))),
                                   shape=(N, N)).tocsr()
    return L, cell, unk_cell


def grounded(grid: Grid | CompositeGrid, open_: NDArray[np.float64],
             p: Params) -> NDArray[np.bool_]:
    """Open cells with a path to the street (ground cells included), on the host."""
    return p.solver.to_host(reach(grid, p.solver.xp.asarray(open_, dtype=float), p))


def placeable(grid: Grid | CompositeGrid, open_: NDArray[np.float64],
              p: Params) -> NDArray[np.bool_]:
    """The grounded cells that may take a home's demand and free an enclosed home, on the host:
    those not split (a split cell's node would speak for parts on the far side of a wall)."""
    o = p.solver.xp.asarray(open_, dtype=float)
    F = p.cells.sub_factor(grid, o.reshape(grid.inside.shape), p.solver)
    r = p.solver.to_host(reach(grid, o, p))
    return r & _one_part(p.solver.to_host(F), grid).reshape(r.shape)


def master_operator(grid: Grid, p: Params):
    """The operator on every inside cell, every footprint open (uniform along-factors, cells by
    their open fraction), on the solver's device: the pattern every system's is a restriction of
    (opening space only adds steps, so no field's system has a step it lacks). Unknowns: inside
    non-ground cells x K, row-major. (Its values are U's whatever `p.cells`: every footprint open,
    Blurred's and WallDistance's factors are the open fraction; the aggregation reads the pattern
    only.)"""
    return operator(grid, grid.isub.mean(axis=-1),
                    dataclasses.replace(p, along=Uniform(), cells=OpenFraction()))[0]


def _csr(solver, vals, rows, cols, n: int, cache: dict):
    """The (n x n) CSR matrix of triplets, duplicates summed. Its structure (a sort of every
    nonzero) is kept for a pattern seen twice (an eps-world system repeats its pattern on every
    call, only the values move; a scoring system's pattern is new each time and is not kept:
    holding it ran a 1 km^2 block out of memory)."""
    xp = solver.xp
    key = (n, len(rows), int(rows.sum()), int(cols.sum()), int((rows * 1000003 ^ cols).sum()))
    store = _repeated(cache, "csr")
    if key in store:
        inv, indices, indptr = store[key]
    else:
        ukeys, inv = xp.unique(rows.astype(xp.int64) * n + cols, return_inverse=True)
        indptr = xp.zeros(n + 1, dtype=xp.int32)
        indptr[1:] = xp.cumsum(xp.bincount(ukeys // n, minlength=n))
        inv, indices = inv.ravel().astype(xp.int32), (ukeys % n).astype(xp.int32)
        _remember(cache, "csr", key, (inv, indices, indptr))
    return solver.sparse.csr_matrix(
        (xp.bincount(inv, weights=vals, minlength=len(indices)), indices, indptr), shape=(n, n))


def _repeated(cache: dict, kind: str) -> dict:
    """The structures of `kind` kept in `cache` (at most one: the pattern that repeats)."""
    return cache.setdefault(kind, {})


def _remember(cache: dict, kind: str, key, value) -> None:
    """Keep `value` for `key` if `key` was seen before (it repeats), replacing what was kept."""
    seen = cache.setdefault(kind + " seen", set())
    if key in seen:
        store = _repeated(cache, kind)
        store.clear()
        store[key] = value
    else:
        seen.add(key)


class System:
    """L on the unknowns that reach ground, set up ONCE for any number of right-hand sides, on
    the solver's device. Unknowns in a component with no ground are dropped (u = 0 there);
    injecting into one is an error."""

    def __init__(self, grid: Grid, open_, p: Params):
        p.solver.release()
        xp = self.xp = p.solver.xp
        self.K = K = p.K
        o = xp.asarray(open_, dtype=xp.float64)
        r, c, v, diag, self.cell, self.unk_cell = _assemble(grid, o, p)
        N = len(diag)
        on = reach(grid, o, p)[o > 0]                         # per free cell
        self.keep = xp.repeat(on[self.unk_cell >= 0], K)
        new = xp.cumsum(self.keep) - 1
        nk = int(self.keep.sum())
        both = self.keep[r]                                   # an edge never leaves its component
        ar = xp.arange(N)[self.keep]
        cache = grid.cache.setdefault(("solver", K, p.ell_m), {})
        self.A = _csr(p.solver, xp.concatenate([v[both], diag[self.keep]]),
                      xp.concatenate([new[r[both]], new[ar]]),
                      xp.concatenate([new[c[both]], new[ar]]), nk, cache)

        def mid():
            mc = -xp.ones(o.size, dtype=xp.int64)
            live = xp.asarray((grid.inside & ~grid.ground).ravel())
            mc[live] = xp.arange(int(live.sum()))
            flat = xp.flatnonzero(o.ravel() > 0)[self.unk_cell >= 0]
            return (mc[flat][:, None] * K + xp.arange(K)[None, :]).ravel()[self.keep]
        self._solver = p.solver.prepare(self.A, Coarsening(
            mid=mid, master=lambda: master_operator(grid, p),
            cache=cache))

    @property
    def n(self) -> int:
        return len(self.keep)

    def solve(self, b, rtol: float = 1e-9, x0=None):
        """Device b (n,) or (n, m) -> device u, zero on dropped unknowns; x0 a starting guess
        over the same unknowns (e.g. the last solve of a nearby field: same numbering when the
        free cells are the same)."""
        if bool((b[~self.keep] != 0).any()):
            raise ValueError("loaded unknowns have no path to ground")
        u = self.xp.zeros(b.shape)
        u[self.keep] = self._solver(b[self.keep], rtol, None if x0 is None else x0[self.keep])
        return u

    def load(self, f_cell, p: Params):
        """Device right-hand side of a per-cell injection, spread over the axes by angular
        share."""
        xp = self.xp
        _, _, m, _ = axes(p.K)
        fc = xp.asarray(f_cell, dtype=xp.float64).ravel()[self.cell.ravel() >= 0]
        live = self.unk_cell >= 0
        b = xp.zeros(self.n)
        idx = (self.unk_cell[live][:, None] * p.K + xp.arange(p.K)[None, :]).ravel()
        b[idx] = (fc[live][:, None] * xp.asarray(m / np.pi)[None, :]).ravel()
        return b


@dataclass(frozen=True, eq=False)
class Solution:
    P: float
    u: NDArray[np.float64]          # per unknown
    cell: NDArray[np.int64]
    unk_cell: NDArray[np.int64]
    n_unknowns: int


def solve(grid: Grid, open_, f_cell: NDArray[np.float64], p: Params,
          system: System | None = None, rtol: float = 1e-9, host: bool = True,
          x0=None) -> Solution:
    """Total escape power. Injection on a closed cell would be a bug: demand cells are open in
    the original geometry and freeing only opens more. Pass `system` (built for this `open_`)
    to reuse its setup. P is quadratic in u, so rtol 1e-5 already gives it to ~1e-10; a ranking
    (tension) is unchanged at 1e-3. `host` False keeps u, cell and unk_cell on the device; x0
    warm-starts the solve (System.solve)."""
    assert not (np.asarray(f_cell)[~(p.solver.to_host(open_) > 0)] > 0).any()
    sy = System(grid, open_, p) if system is None else system
    b = sy.load(f_cell, p)
    u = sy.solve(b, rtol=rtol, x0=x0)
    P = float(b @ u)
    if not host:
        return Solution(P=P, u=u, cell=sy.cell, unk_cell=sy.unk_cell, n_unknowns=sy.n)
    th = p.solver.to_host
    return Solution(P=P, u=th(u), cell=th(sy.cell), unk_cell=th(sy.unk_cell), n_unknowns=sy.n)


def cell_mean_u(sol: Solution, open_: NDArray[np.float64], p: Params) -> NDArray[np.float64]:
    """(ny, nx): each open cell's potential averaged over axes by angular share (the weights the
    injection uses), 0 on ground and closed cells. Host arrays."""
    _, _, m, _ = axes(p.K)
    free = open_ > 0
    nc = int(free.sum())
    uk = np.zeros((nc, p.K))
    live = sol.unk_cell >= 0
    uk[live] = sol.u.reshape(-1, p.K)
    out = np.zeros(free.shape)
    out[free] = uk @ (m / np.pi)
    return out
