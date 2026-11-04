"""The screen's convex regions (owner 2026-11-03), for the Demo Lineup artifact's height map: hulls
of agglomerated blocks, each scored by the screen's formula applied to the hull as if it were one
big block.

A region is a convex polygon H, the convex hull of its member blocks (the site's ScreenMap bundle).
Its screen inputs: n(H) = sum over blocks b of n_b area(b cap H) / area(b), the buildings
prorated by overlap (density uniform within a block) over EVERY block H overlaps, members or
not; A(H) = area(H), streets and empty ground included, so empty ground lowers density; P(H) its
perimeter. Recorded beside them: the street / empty share 1 - sum area(b cap H) / A(H) (the
bundle's simplified rings overlap a little, so it can dip just below 0), the blocks touched
(overlap > 1e-6 m^2), the member blocks' buildings and n_eff = (sum w)^2 / sum w^2.

Scores (a Strategy each, Formula; per metric a Scores pair):
  as_is: the shipped formulas on (n, A, P): depth_density_proxy sqrt(nA)/P n/A, density_
    compactness n/P^2.
  intensive: size-agnostic. depth_density_proxy factors exactly as sqrt(n) (n/A) sqrt(A/P^2):
    the sqrt(n), the depth, grows with the region (meaningless for a region with internal
    streets), so it is dropped: (n/A) sqrt(A/P^2). density_compactness n/P^2 = (n/A)(A/P^2) is
    already intensive, so its as_is and intensive trees and lists coincide.

Candidates: an agglomeration per score. Every block a cluster, its hull the block's own convex
hull; then repeatedly the pair of clusters neighbouring in the Delaunay graph of the blocks'
centroids whose merged hull (the convex hull of both clusters' blocks) has the best log score, a
merge whose n(H) would exceed MAX_SHARE of the city's buildings refused (so the tree ends a
forest). Every node with >= MIN_BLOCKS (2) members is a candidate.
Exact and fast: proration is clipped exactly (Sutherland--Hodgman of each block ring against the
convex hull, numba; block rings oriented so signed areas sum to the block's), incrementally: a
merge's hull h contains the larger cluster's hull hb, and h less hb is a set of pockets, one per
run of h's edges not hb's, each a convex Q (the run closed by its chord) less hb's part R beyond
the chord, so a merge reads only the blocks meeting the Qs (a uniform grid index) and clips them
against Q and R, each a few edges, culling the edges that leave a block's bounding box whole.
48 -- 83 us a merge evaluation on Cape Town (median 26 us, a heavy tail from merges of
1,000 km^2 hulls); an intensive tree is 4 s Nairobi, 38 -- 42 s Cape Town (464,000 evaluations),
the as_is depth_density_proxy tree, which snowballs (its sqrt(n) favours the biggest cluster),
20 s and 180 s (2.2 million). shapely's own clipping was 10x slower; the cheaper route (blocks
by centroid inclusion) was not needed.

Rankings, per metric, each the TOP disjoint (hulls overlapping by < OVERLAP_MAX of the smaller's
area; greedy in rank order):
  as_is: the as_is tree's candidates by as_is score.
  intensive_raw: the intensive tree's by intensive score (MIN_BLOCKS 2: these are tiny).
  intensive_cal: the intensive tree's by p under msr, then z under msr. p as screen_concepts.py
    calibrates: per replicate the best intensive log score among its candidates with n_eff within
    a factor 1.25 (bins of log10(1.25)/8), p = (1 + #replicates >= the region's) / (1 + NPERM);
    z = (score - mean) / sd of every null candidate's at that n_eff (Gaussian kernel, bandwidth
    log10 1.25).
Nulls (a Strategy each, screen_concepts.py's), each replicate regenerating the whole tree (the
selection is in the null): the field re-dealt is the block building density d_b = n_b / area_b
(the hull scores' only data), n_b' = d_b' area_b. shuffle: the densities permuted among the
blocks. msr: Moran spectral randomization of log10 d on the street adjacency (screen_concepts.py's
MSR, one eigendecomposition a city shared by both jobs), the observed densities re-dealt by the
surrogate's ranks. Replicates seeded default_rng([SEED, null index, replicate]); one surrogate
regenerates both metrics' intensive trees (the density field is metric-free).
The bundle's followed block (Cape Town's 5810): whether each region holds it (member, and its
area's share inside the hull), and per ranking the best-ranked candidate holding it.
Each region: its hull (the exterior ring, the bundle's local metres, the ScreenMap rings' frame),
members, blocks_touched, n, A km^2, P km, street_share, both metrics' as_is and intensive scores
and x floor (as_is / the shipped block floor: a hull's A includes its streets, so a hull of
at-floor blocks scores below the floor; beside it the blocks-only variant, sum area(b cap H) for
A), n_eff, z and p under both nulls (the intensive lists), its centroid cx, cy; each reported
region's n(H) re-checked against shapely's exact clipping (an error if they differ by 1e-6).

Also clusters_msr_<city>.json: screen_clusters.py's hull-tree scan (its run, wrapped, unchanged:
its clusters, llr and shuffle p), each cluster's p_msr beside its p: its llr's rank among the
largest llr of hull trees rebuilt on MSR surrogates of the blocks' log scores (screen_concepts.
py's surrogates, the score marginal kept), as many replicates as its shuffle's (47), seeded
default_rng([SEED, 1, replicate]).

RUN PENDING

    S=research/roadless/atlas/explore_proto/screen_hulls.py  # from the repo root, PYTHONPATH=.
    uv run python $S ~/src/wt_demo <out dir> [city ...]   # hulls_<city>.json,
                                                # clusters_msr_<city>.json (default both)
    uv run python $S check [fault]              # proration, factorisation, scale invariance,
                                                # conditional p, the tree; faults:
                                                # drop-ratio, drop-sqrt, as-is-intensive,
                                                # unconditional, skip-member
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"           # the pool's workers single-threaded (the machine is shared)

import heapq  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import NamedTuple  # noqa: E402

import numpy as np  # noqa: E402
import screen_clusters as sc  # noqa: E402  (the script's directory is on the path)
import screen_concepts as cn  # noqa: E402
import shapely  # noqa: E402
import shapely.affinity  # noqa: E402
from numba import njit  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

MAX_SHARE = 0.5        # a candidate's buildings n(H) at most this share of the city's
MIN_BLOCKS = 2         # a candidate region: at least this many member blocks (one is the block)
TOP = 8                # regions per ranking
OVERLAP_MAX = 0.05     # disjoint: two hulls overlap by less than this share of the smaller's area
MIN_SUPPORT = 0.5      # p defined where at least this share of null replicates has a candidate of
                       # the region's n_eff (within WINDOW)
NPERM = {"capetown": 199, "nairobi": 999}      # null replicates per null (a tree each)
SEED = 0
WORKERS = 16
CITIES = ("nairobi", "capetown")

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]


# ---------------------------------------------------------------- the plane: blocks as rings

class Plane(NamedTuple):
    """The blocks' rings for exact clipping (numba): every ring's vertices (open, no repeat),
    oriented so a block's signed ring areas sum to its area (exterior counter-clockwise, holes
    clockwise); a uniform grid of cells over the blocks' bounding boxes (an index only: no
    effect on a result)."""
    xy: F64                  # M x 2
    ring_ptr: I64            # R + 1
    block_ring: I64          # N + 1: block b's rings block_ring[b]:block_ring[b + 1]
    area: F64                # N: the signed ring areas' sum (the block's area)
    box: F64                 # N x 4: xmin, ymin, xmax, ymax
    cells: I64               # N x 4: the block's cell range ix0, iy0, ix1, iy1 (inclusive)
    origin: F64              # 2
    cell: float
    nx: int
    ny: int
    cell_ptr: I64            # nx * ny + 1
    cell_blocks: I64


def plane_of(polys: list) -> Plane:
    xy, ring_ptr, block_ring = [], [0], [0]
    for p in polys:
        for g in shapely.get_parts(p):
            g = shapely.orient_polygons(g)
            for r in [g.exterior, *g.interiors]:
                c = np.asarray(r.coords)[:-1]
                xy.append(c)
                ring_ptr.append(ring_ptr[-1] + len(c))
        block_ring.append(len(ring_ptr) - 1)
    XY = np.ascontiguousarray(np.vstack(xy))
    rp, br = np.asarray(ring_ptr, dtype=np.int64), np.asarray(block_ring, dtype=np.int64)
    area = _areas(XY, rp, br)
    box = np.asarray(shapely.bounds(np.asarray(polys, dtype=object)), dtype=float)
    cell = float(2.0 * np.median(np.sqrt(np.abs(area))))
    origin = box[:, :2].min(axis=0)
    nx = int((box[:, 2].max() - origin[0]) // cell) + 1
    ny = int((box[:, 3].max() - origin[1]) // cell) + 1
    cells = np.column_stack([(box[:, 0] - origin[0]) // cell, (box[:, 1] - origin[1]) // cell,
                             (box[:, 2] - origin[0]) // cell, (box[:, 3] - origin[1]) // cell]
                            ).astype(np.int64)
    cid, bid = [], []
    for b, (x0, y0, x1, y1) in enumerate(cells):
        ix, iy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        cid.append((iy * nx + ix).ravel())
        bid.append(np.full(ix.size, b))
    cid_a, bid_a = np.concatenate(cid), np.concatenate(bid)
    o = np.argsort(cid_a, kind="stable")
    ptr = np.zeros(nx * ny + 1, dtype=np.int64)
    np.add.at(ptr, cid_a + 1, 1)
    return Plane(xy=XY, ring_ptr=rp, block_ring=br, area=area, box=box, cells=cells,
                 origin=origin, cell=cell, nx=nx, ny=ny, cell_ptr=np.cumsum(ptr),
                 cell_blocks=bid_a[o].astype(np.int64))


@njit(cache=True)
def _shoelace(p: F64, m: int) -> float:
    a = 0.0
    for i in range(m):
        j = i + 1 if i + 1 < m else 0
        a += p[i, 0] * p[j, 1] - p[j, 0] * p[i, 1]
    return a / 2.0


@njit(cache=True)
def _areas(xy: F64, rp: I64, br: I64) -> F64:
    n = len(br) - 1
    out = np.zeros(n)
    for b in range(n):
        for r in range(br[b], br[b + 1]):
            out[b] += _shoelace(xy[rp[r]:rp[r + 1]], rp[r + 1] - rp[r])
    return out


@njit(cache=True)
def _ring_clip(xy: F64, s: int, e: int, h: F64, cut: I64, ncut: int) -> float:
    """The signed area of ring xy[s:e] clipped to the convex counter-clockwise polygon h, by
    the edges cut[:ncut] of h only (the others leave the ring's bounding box whole):
    Sutherland--Hodgman, exact for any ring against a convex clipper (the degenerate edges it
    leaves where a concave ring re-enters carry no area)."""
    m = e - s
    cap = 2 * m + 8
    a = np.empty((cap, 2))
    bb = np.empty((cap, 2))
    for i in range(m):
        a[i, 0], a[i, 1] = xy[s + i, 0], xy[s + i, 1]
    k = len(h)
    for q in range(ncut):
        c = cut[q]
        x0, y0 = h[c, 0], h[c, 1]
        x1, y1 = h[(c + 1) % k, 0], h[(c + 1) % k, 1]
        ex, ey = x1 - x0, y1 - y0
        if 2 * m + 2 > cap:
            cap = 2 * m + 2
            a2 = np.empty((cap, 2))
            a2[:m] = a[:m]
            a = a2
            bb = np.empty((cap, 2))
        out = 0
        px, py = a[m - 1, 0], a[m - 1, 1]
        dp = ex * (py - y0) - ey * (px - x0)
        for i in range(m):
            cx, cy = a[i, 0], a[i, 1]
            dc = ex * (cy - y0) - ey * (cx - x0)
            if dc >= 0.0:
                if dp < 0.0:
                    t = dp / (dp - dc)
                    bb[out, 0], bb[out, 1] = px + t * (cx - px), py + t * (cy - py)
                    out += 1
                bb[out, 0], bb[out, 1] = cx, cy
                out += 1
            elif dp >= 0.0:
                t = dp / (dp - dc)
                bb[out, 0], bb[out, 1] = px + t * (cx - px), py + t * (cy - py)
                out += 1
            px, py, dp = cx, cy, dc
        a, bb = bb, a
        m = out
        if m < 3:
            return 0.0
    return _shoelace(a, m)


@njit(cache=True)
def _box(h: F64) -> tuple[float, float, float, float]:
    return h[:, 0].min(), h[:, 1].min(), h[:, 0].max(), h[:, 1].max()


@njit(cache=True)
def block_in(pl_xy: F64, rp: I64, br: I64, area: F64, box: F64, j: int, h: F64,
             hx0: float, hy0: float, hx1: float, hy1: float, cut: I64) -> float:
    """area(block j cap h), h convex counter-clockwise with bounding box hx0..hy1: 0 when the
    block's box lies outside an edge's half-plane, the block's area when inside every one,
    else its rings clipped by the edges that cut the box (cut: scratch, len(h) long)."""
    if box[j, 2] <= hx0 or box[j, 0] >= hx1 or box[j, 3] <= hy0 or box[j, 1] >= hy1:
        return 0.0
    k = len(h)
    if k < 3:
        return 0.0
    ncut = 0
    for c in range(k):
        x0, y0 = h[c, 0], h[c, 1]
        ex, ey = h[(c + 1) % k, 0] - x0, h[(c + 1) % k, 1] - y0
        nin = 0
        for cx, cy in ((box[j, 0], box[j, 1]), (box[j, 2], box[j, 1]), (box[j, 0], box[j, 3]),
                       (box[j, 2], box[j, 3])):
            if ex * (cy - y0) - ey * (cx - x0) >= 0.0:
                nin += 1
        if nin == 0:
            return 0.0
        if nin < 4:
            cut[ncut] = c
            ncut += 1
    if ncut == 0:
        return area[j]
    a = 0.0
    for r in range(br[j], br[j + 1]):
        a += _ring_clip(pl_xy, rp[r], rp[r + 1], h, cut, ncut)
    return a


@njit(cache=True)
def _index(hb: F64, x: float, y: float) -> int:
    for i in range(len(hb)):
        if hb[i, 0] == x and hb[i, 1] == y:
            return i
    return -1


@njit(cache=True)
def _pocket(Q: F64, R: F64, w: F64, xy: F64, rp: I64, br: I64, area: F64, box: F64,
            cells: I64, origin: F64, cell: float, nx: int, ny: int, cell_ptr: I64,
            cell_blocks: I64, cut: I64) -> tuple[float, float]:
    """Over the blocks meeting Q's bounding box: sum w_j (area(j cap Q) - area(j cap R)) /
    area(j), and the sum of the area differences (R inside Q, both convex counter-clockwise; R
    of no vertices: nothing taken off)."""
    qx0, qy0, qx1, qy1 = _box(Q)
    if len(R):
        rx0, ry0, rx1, ry1 = _box(R)
    else:
        rx0 = ry0 = rx1 = ry1 = 0.0
    ix0 = max(int((qx0 - origin[0]) // cell), 0)
    iy0 = max(int((qy0 - origin[1]) // cell), 0)
    ix1 = min(int((qx1 - origin[0]) // cell), nx - 1)
    iy1 = min(int((qy1 - origin[1]) // cell), ny - 1)
    dn = da = 0.0
    for iy in range(iy0, iy1 + 1):
        for ix in range(ix0, ix1 + 1):
            c = iy * nx + ix
            for t in range(cell_ptr[c], cell_ptr[c + 1]):
                j = cell_blocks[t]
                # each block once: in the first cell of its range within the query's
                if ix != max(cells[j, 0], ix0) or iy != max(cells[j, 1], iy0):
                    continue
                a1 = block_in(xy, rp, br, area, box, j, Q, qx0, qy0, qx1, qy1, cut)
                if a1 == 0.0:
                    continue
                a0 = (block_in(xy, rp, br, area, box, j, R, rx0, ry0, rx1, ry1, cut)
                      if len(R) >= 3 else 0.0)
                d = a1 - a0
                if d != 0.0:
                    dn += w[j] * d / area[j]
                    da += d
    return dn, da


@njit(cache=True)
def prorated(h: F64, hb: F64, w: F64, xy: F64, rp: I64, br: I64, area: F64, box: F64,
             cells: I64, origin: F64, cell: float, nx: int, ny: int, cell_ptr: I64,
             cell_blocks: I64) -> tuple[float, float]:
    """n(h) - n(hb) and the blocks' area in h less in hb (hb of no vertices: n(h) itself), for
    convex counter-clockwise hb inside h, hb's vertices exact copies of h's where they are h's
    (h the hull of a point set holding hb's): every block j adds w_j (area(j cap h) - area(j cap
    hb)) / area(j). h less hb is a set of pockets, one per maximal run of h's edges that are not
    hb's: the run from hb's vertex a to its vertex b closes with the chord ab into a convex
    polygon Q (h beyond the chord), and hb beyond the chord is R, hb's vertices from a to b
    (both run counter-clockwise), so the pocket is Q less R and only blocks meeting a Q are read.
    With no edge in common (hb touching h at most at vertices) the one pocket is h less hb."""
    k = len(h)
    cut = np.empty(max(k, len(hb)) + 1, dtype=np.int64)
    G = (w, xy, rp, br, area, box, cells, origin, cell, nx, ny, cell_ptr, cell_blocks, cut)
    if len(hb) == 0:
        return _pocket(h, hb, *G)
    shared = np.zeros(k, dtype=np.bool_)
    for i in range(k):
        ia = _index(hb, h[i, 0], h[i, 1])
        if ia >= 0:
            j = (i + 1) % k
            nb = (ia + 1) % len(hb)
            shared[i] = hb[nb, 0] == h[j, 0] and hb[nb, 1] == h[j, 1]
    if not shared.any():
        return _pocket(h, hb, *G)
    s0 = 0
    while not shared[s0]:
        s0 += 1
    dn = da = 0.0
    i = (s0 + 1) % k                     # walk from the edge after a shared one, once round
    steps = 0
    while steps < k:
        if shared[i]:
            i = (i + 1) % k
            steps += 1
            continue
        i0 = i
        while not shared[i]:
            i = (i + 1) % k
            steps += 1
        # the run is edges i0 .. i - 1: vertices i0 .. i
        m = (i - i0) % k + 1
        Q = np.empty((m, 2))
        for t in range(m):
            Q[t] = h[(i0 + t) % k]
        ia = _index(hb, h[i0, 0], h[i0, 1])
        ib = _index(hb, h[i, 0], h[i, 1])
        mr = (ib - ia) % len(hb) + 1
        R = np.empty((mr, 2))
        for t in range(mr):
            R[t] = hb[(ia + t) % len(hb)]
        x, y = _pocket(Q, R, *G)
        dn += x
        da += y
    return dn, da


NOHULL = np.empty((0, 2))


def prorate(pl: Plane, h: F64, hb: F64, w: F64) -> tuple[float, float]:
    return prorated(h, hb, w, pl.xy, pl.ring_ptr, pl.block_ring, pl.area, pl.box, pl.cells,
                    pl.origin, pl.cell, pl.nx, pl.ny, pl.cell_ptr, pl.cell_blocks)


@njit(cache=True)
def perimeter_of(h: F64) -> float:
    p = 0.0
    for i in range(len(h)):
        j = (i + 1) % len(h)
        p += np.hypot(h[j, 0] - h[i, 0], h[j, 1] - h[i, 1])
    return p


@njit(cache=True)
def merged(px: F64, py: F64, hb: F64, w: F64, xy: F64, rp: I64, br: I64, area: F64, box: F64,
           cells: I64, origin: F64, cell: float, nx: int, ny: int, cell_ptr: I64,
           cell_blocks: I64) -> tuple[F64, float, float, float, float]:
    """The hull of two clusters' hulls px, py; its area, perimeter; and n and the blocks' area
    it adds to those of hb (one of px, py: the larger)."""
    h = sc.hull(np.vstack((px, py)))
    dn, da = prorated(h, hb, w, xy, rp, br, area, box, cells, origin, cell, nx, ny, cell_ptr,
                      cell_blocks)
    return h, sc.area_of(h), perimeter_of(h), dn, da


# ---------------------------------------------------------------- the scores, a Strategy each

@dataclass(frozen=True)
class Formula:
    """A region score of its buildings n, area A and perimeter P."""
    name: str
    f: Callable[[F64, F64, F64], F64]

    def log(self, n: float, A: float, P: float) -> float:
        v = float(self.f(n, A, P))
        return math.log10(v) if v > 0 else -math.inf


class Scores(NamedTuple):
    """A metric's two scores of a region: as_is, the shipped formula; intensive, its
    size-agnostic form."""
    as_is: Formula
    intensive: Formula


# depth_density_proxy sqrt(nA)/P n/A = sqrt(n) (n/A) sqrt(A/P^2): the sqrt(n), the depth, grows
# with the region, so the intensive form drops it; density_compactness n/P^2 = (n/A)(A/P^2) is
# already intensive (both unchanged by scaling a region at fixed density)
SCORES = {
    "depth_density_proxy": Scores(
        as_is=Formula("as_is", sc.METRICS["depth_density_proxy"]),
        intensive=Formula("intensive", lambda n, A, P: n / A * np.sqrt(A / P ** 2))),
    "density_compactness": Scores(
        as_is=Formula("as_is", sc.METRICS["density_compactness"]),
        intensive=Formula("intensive", lambda n, A, P: n / A * (A / P ** 2))),
}


# ---------------------------------------------------------------- the agglomeration

@dataclass(frozen=True)
class City:
    """What every tree of a city shares: its blocks, their rings (Plane), each block's convex
    hull (counter-clockwise), and the Delaunay pairs of the blocks' centroids."""
    b: sc.Blocks
    pl: Plane
    hulls: list
    edges: I64

    def prorate(self, h: F64, hb: F64, w: F64) -> tuple[float, float]:
        return prorate(self.pl, h, hb, w)

    def merged(self, px: F64, py: F64, hb: F64, w: F64) -> tuple[F64, float, float, float,
                                                                   float]:
        pl = self.pl
        return merged(px, py, hb, w, pl.xy, pl.ring_ptr, pl.block_ring, pl.area, pl.box,
                      pl.cells, pl.origin, pl.cell, pl.nx, pl.ny, pl.cell_ptr, pl.cell_blocks)


def city_of(data: dict) -> City:
    b = sc.blocks_of(data)
    hulls = [sc.hull(np.vstack([np.asarray(g.exterior.coords)[:-1]
                                for g in shapely.get_parts(p)])) for p in b.polys]
    return City(b=b, pl=plane_of(b.polys), hulls=hulls, edges=sc.delaunay_edges(b.cen))


class Tree(NamedTuple):
    """The agglomeration's nodes: blocks 0..N-1, then each merge N, N+1, ...; per node its
    hull (counter-clockwise vertices), its children (-1 for a block), n(H), the blocks' area in
    H, A(H), P(H), its log score (the tree's), its member blocks' count, buildings and sum of
    squared buildings."""
    hulls: list
    kids: I64
    n: F64
    cover: F64
    A: F64
    P: F64
    log: F64
    blocks: I64
    W: F64
    W2: F64

    @property
    def n_eff(self) -> F64:
        return self.W ** 2 / self.W2

    def members(self, node: int) -> I64:
        N = int((self.kids[:, 0] < 0).sum())
        out, stack = [], [node]
        while stack:
            x = stack.pop()
            if x < N:
                out.append(x)
            else:
                stack.extend(int(c) for c in self.kids[x])
        return np.sort(np.asarray(out, dtype=np.int64))


def agglomerate(city: City, w: F64, score: Formula, max_share: float) -> Tree:
    """Every block a cluster, its hull the block's convex hull; then repeatedly the pair of
    clusters neighbouring in the Delaunay graph of the blocks' centroids (a cluster's
    neighbours its blocks' neighbours outside it) whose merged hull, the convex hull of both
    clusters' blocks, scores best under score, n(H) prorated over every block H overlaps; a
    merge whose n(H) exceeds max_share of the buildings is refused (the tree ends a forest).
    A pair's score is fixed while both live, so the heap is lazy: a popped pair with a merged
    member is skipped. Ties broken by the pair's node ids."""
    N = len(w)
    cap = max_share * float(w.sum())
    pts = list(city.hulls)
    A = [sc.area_of(h) for h in pts]
    P = [perimeter_of(h) for h in pts]
    n, cov = [], []
    for h in pts:
        dn, da = city.prorate(h, NOHULL, w)
        n.append(dn)
        cov.append(da)
    lg = [score.log(n[i], A[i], P[i]) for i in range(N)]
    W, W2, k = list(w), list(w * w), [1] * N
    kids = [(-1, -1)] * N
    nb: list[set[int]] = [set() for _ in range(N)]
    for x, y in city.edges:
        nb[x].add(int(y))
        nb[y].add(int(x))
    alive = [True] * N

    def pair(x: int, y: int) -> tuple[F64, float, float, float, float]:
        base = x if A[x] >= A[y] else y
        h, Ah, Ph, dn, da = city.merged(pts[x], pts[y], pts[base], w)
        return h, Ah, Ph, n[base] + dn, cov[base] + da

    heap = []
    for x, y in city.edges:
        _, Ah, Ph, nn, _ = pair(int(x), int(y))
        if nn <= cap:
            heap.append((-score.log(nn, Ah, Ph), int(x), int(y)))
    heapq.heapify(heap)
    while heap:
        _, x, y = heapq.heappop(heap)
        if not (alive[x] and alive[y]):
            continue
        h, Ah, Ph, nn, aa = pair(x, y)
        z = len(pts)
        pts.append(h)
        A.append(Ah)
        P.append(Ph)
        n.append(nn)
        cov.append(aa)
        lg.append(score.log(nn, Ah, Ph))
        W.append(W[x] + W[y])
        W2.append(W2[x] + W2[y])
        k.append(k[x] + k[y])
        kids.append((x, y))
        alive.append(True)
        alive[x] = alive[y] = False
        nz = (nb[x] | nb[y]) - {x, y}
        nb.append(nz)
        for u in nz:
            nb[u].discard(x)
            nb[u].discard(y)
            nb[u].add(z)
            _, Au, Pu, nu, _ = pair(z, u)
            if nu <= cap:
                heapq.heappush(heap, (-score.log(nu, Au, Pu), min(z, u), max(z, u)))
    return Tree(hulls=pts, kids=np.asarray(kids, dtype=np.int64), n=np.asarray(n),
                cover=np.asarray(cov), A=np.asarray(A), P=np.asarray(P), log=np.asarray(lg),
                blocks=np.asarray(k, dtype=np.int64), W=np.asarray(W), W2=np.asarray(W2))


# ---------------------------------------------------------------- the nulls: densities re-dealt

def density(city: City, w: F64) -> F64:
    return w / city.pl.area


def surrogate_buildings(city: City, null: cn.Null, rng: np.random.Generator) -> F64:
    """The blocks' building densities (n_b / area_b, uniform within a block) re-dealt among the
    blocks by null (on their log10: shuffle, or msr on the street adjacency), each block's
    buildings then its new density times its own area."""
    d = density(city, city.b.w)
    v = np.log10(d)
    x = null.surrogates(v, [rng])[0]
    sv = np.sort(v)
    return np.sort(d)[np.searchsorted(sv, x)] * city.pl.area


_JOB: dict = {}


class Rep(NamedTuple):
    """One null replicate: per metric its candidates binned by n_eff, the seconds its tree
    took; the surrogate log densities' Moran's I."""
    binned: dict
    seconds: dict
    moran: float


def candidates(t: Tree) -> I64:
    return np.flatnonzero(t.blocks >= MIN_BLOCKS)


def _rep(task: tuple[int, int]) -> Rep:
    ni, r = task
    city, nulls, A, G = _JOB["city"], _JOB["nulls"], _JOB["A"], _JOB["G"]
    w = surrogate_buildings(city, nulls[ni], np.random.default_rng([SEED, ni, r]))
    out, sec = {}, {}
    for metric, sc_ in SCORES.items():
        t0 = time.time()
        t = agglomerate(city, w, sc_.intensive, MAX_SHARE)
        sec[metric] = time.time() - t0
        c = candidates(t)
        out[metric] = cn.binned(t.n_eff[c], t.log[c], G)
    return Rep(binned=out, seconds=sec, moran=cn.moran(np.log10(density(city, w)), A))


def nulls_run(city: City, A: object, nulls: list[cn.Null], nperm: int, G: int) -> list[Rep]:
    """Every (null, replicate), each regenerating both metrics' intensive trees from one
    surrogate (the field re-dealt is the metric-free building density)."""
    _JOB.update(city=city, nulls=nulls, A=A, G=G)
    tasks = [(ni, r) for ni in range(len(nulls)) for r in range(nperm)]
    with get_context("fork").Pool(WORKERS) as pool:
        return pool.map(_rep, tasks, chunksize=1)


# ---------------------------------------------------------------- selection and the report

def disjoint(order: I64, t: Tree, top: int) -> list[int]:
    """The first top nodes of order whose hull overlaps every earlier chosen one's by less
    than OVERLAP_MAX of the smaller's area (greedy)."""
    chosen: list[tuple[int, object]] = []
    for node in order:
        H = shapely.Polygon(t.hulls[node])
        if all(shapely.area(shapely.intersection(H, G)) < OVERLAP_MAX * min(t.A[node], t.A[c])
               for c, G in chosen):
            chosen.append((int(node), H))
            if len(chosen) == top:
                break
    return [c for c, _ in chosen]


def parents(t: Tree) -> I64:
    par = np.full(len(t.kids), -1, dtype=np.int64)
    for z, (x, y) in enumerate(t.kids):
        if x >= 0:
            par[x] = par[y] = z
    return par


def ancestors(t: Tree, block: int) -> I64:
    par, out, x = parents(t), [], block
    while x >= 0:
        out.append(x)
        x = int(par[x])
    return np.asarray(out, dtype=np.int64)


def _f(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def entry(node: int, t: Tree, city: City, floors: dict[str, float], fol: int | None,
          zp: dict[str, tuple[F64, F64]] | None) -> dict:
    h = t.hulls[node]
    H = shapely.Polygon(h)
    mem = t.members(node)
    P = np.asarray(city.b.polys, dtype=object)
    idx = shapely.STRtree(P).query(H, predicate="intersects")
    a = shapely.area(shapely.intersection(P[idx], H))
    n_ref = float((city.b.w[idx] * a / city.pl.area[idx]).sum())
    if abs(n_ref - t.n[node]) > 1e-6 * max(t.n[node], 1.0):
        raise RuntimeError(f"node {node}: n(H) {t.n[node]} unlike shapely's {n_ref}")
    n, A, P_, cov = float(t.n[node]), float(t.A[node]), float(t.P[node]), float(t.cover[node])
    scores = {}
    for m, s in SCORES.items():
        as_is = float(s.as_is.f(n, A, P_))
        scores[m] = dict(as_is=as_is, intensive=float(s.intensive.f(n, A, P_)),
                         x_floor=as_is / floors[m],
                         blocks_only_x_floor=float(s.as_is.f(n, cov, P_)) / floors[m])
    c = shapely.centroid(H)
    out = dict(
        node=int(node), hull=np.round(h, 2).tolist(), members=mem.tolist(),
        blocks=int(t.blocks[node]), blocks_touched=int((a > 1e-6).sum()), buildings=n,
        member_buildings=float(t.W[node]), n_eff=float(t.n_eff[node]), area_km2=A / 1e6,
        perimeter_km=P_ / 1e3, street_share=1.0 - cov / A, log_score=float(t.log[node]),
        scores=scores, cx=float(shapely.get_x(c)), cy=float(shapely.get_y(c)))
    if zp is not None:
        for nm, (z, p) in zp.items():
            out[f"z_{nm}"] = _f(z[node])
            out[f"p_{nm}"] = _f(p[node])
    if fol is not None:
        out["follow_member"] = bool(fol in set(mem.tolist()))
        f = shapely.area(shapely.intersection(P[fol], H)) / city.pl.area[fol]
        out["follow_share_in_hull"] = float(f)
    return out


def ranked(order: I64, t: Tree, city: City, floors: dict[str, float], fol: int | None,
           zp: dict | None) -> tuple[list[dict], dict | None]:
    """The TOP disjoint regions of order, and the best-ranked node holding the followed block
    (disjointness not required)."""
    lst = [entry(node, t, city, floors, fol, zp) for node in disjoint(order, t, TOP)]
    follow = None
    if fol is not None:
        anc = set(ancestors(t, fol).tolist())
        for rank, node in enumerate(order):
            if int(node) in anc:
                follow = dict(rank=rank + 1, of=len(order),
                              **entry(int(node), t, city, floors, fol, zp))
                break
    return lst, follow


def conditional_p(cal: cn.Calibration, n_eff: F64, stat: F64) -> F64:
    """p among the replicates that HAVE a candidate within WINDOW of the region's n_eff: (1 +
    #those whose best is >= stat) / (1 + #those), undefined (nan) where fewer than
    MIN_SUPPORT of the replicates have one. screen_concepts.py counts a replicate with no
    candidate of the size as one below the region, so a size the null seldom reaches gets a
    small p whatever its score; here it would, since the null's re-dealt densities times block
    areas give heavier-tailed buildings, hence smaller n_eff for a region of a given extent."""
    g = cn.bins_of(n_eff)
    R = cal.wbest.shape[0]
    out = np.full(len(n_eff), np.nan)
    for i, (gi, s) in enumerate(zip(g, stat, strict=True)):
        col = cal.wbest[:, gi]
        have = int(np.isfinite(col).sum())
        if have >= MIN_SUPPORT * R:
            out[i] = (1.0 + R - np.searchsorted(col, s, side="left")) / (1.0 + have)
    return out


def of_null(reps: list[Rep], ni: int, nperm: int) -> list[Rep]:
    return reps[ni * nperm:(ni + 1) * nperm]


def scanned(data: dict, city: City, A: object, nulls: list[cn.Null], nperm: int) -> dict:
    floors = {f["metric"]: f["value"] for f in data["floors"]}
    fol = int(data["follow"]["index"]) if data.get("follow") else None
    N = len(city.b.w)
    G = int(np.ceil(np.log10(N) / cn.STEP)) + 2
    obs = {}
    sec = {}
    for metric, s in SCORES.items():
        for form in (s.as_is, s.intensive):
            t0 = time.time()
            obs[metric, form.name] = agglomerate(city, city.b.w, form, MAX_SHARE)
            sec[metric, form.name] = time.time() - t0
            print(f"  {metric} {form.name}: tree {sec[metric, form.name]:.0f}s, "
                  f"{len(obs[metric, form.name].kids) - N} merges", flush=True)
    t0 = time.time()
    reps = nulls_run(city, A, nulls, nperm, G)
    t_null = time.time() - t0
    print(f"  nulls: {len(reps)} replicates {t_null:.0f}s", flush=True)
    I_obs = cn.moran(np.log10(density(city, city.b.w)), A)
    res = {}
    for metric in SCORES:
        ta, ti = obs[metric, "as_is"], obs[metric, "intensive"]
        ci = candidates(ti)
        cals = [cn.calibrate([r.binned[metric] for r in of_null(reps, ni, nperm)])
                for ni in range(len(nulls))]
        zp = {}
        for nl, cal in zip(nulls, cals, strict=True):
            z, p = np.full(len(ti.kids), np.nan), np.full(len(ti.kids), np.nan)
            pc = conditional_p(cal, ti.n_eff[ci], ti.log[ci])
            ok = np.isfinite(pc)
            p[ci] = pc
            z[ci[ok]] = cal.z(ti.n_eff[ci[ok]], ti.log[ci[ok]])
            zp[nl.name] = (z, p)
        ca = candidates(ta)
        o_as = ca[np.argsort(-ta.log[ca], kind="stable")]
        o_raw = ci[np.argsort(-ti.log[ci], kind="stable")]
        z_m, p_m = zp["msr"]
        cs = ci[np.isfinite(p_m[ci])]
        o_cal = cs[np.lexsort((-np.nan_to_num(z_m[cs], nan=-np.inf), p_m[cs]))]
        l_as, f_as = ranked(o_as, ta, city, floors, fol, None)
        l_raw, f_raw = ranked(o_raw, ti, city, floors, fol, zp)
        l_cal, f_cal = ranked(o_cal, ti, city, floors, fol, zp)
        g = np.arange(G)
        keep = np.isfinite(cals[0].q(0.95)) | np.isfinite(cals[1].q(0.95))
        params = dict(
            max_share=MAX_SHARE, min_blocks=MIN_BLOCKS, top=TOP, overlap_max=OVERLAP_MAX,
            region=("H the convex hull of the member blocks; n(H) = sum over blocks b of n_b "
                    "area(b cap H) / area(b) (buildings uniform within a block, prorated by "
                    "overlap, over every block H overlaps, members or not); A(H) its whole area "
                    "(streets and empty ground included, so empty ground lowers density), P(H) "
                    "its perimeter; street_share = 1 - sum area(b cap H) / A(H) (blocks' "
                    "simplified rings overlap a little, so it can dip below 0)"),
            x_floor=("as_is / the shipped block floor. A hull's A includes its streets, so a "
                     "hull of at-floor blocks scores below the floor; blocks_only_x_floor puts "
                     "sum area(b cap H) for A (density_compactness does not read A: the same)"),
            scores={"as_is": "the shipped formula on (n, A, P)",
                    "intensive": ("depth_density_proxy sqrt(nA)/P n/A = sqrt(n) (n/A) "
                                  "sqrt(A/P^2) less its sqrt(n): (n/A) sqrt(A/P^2); "
                                  "density_compactness n/P^2 = (n/A)(A/P^2), already "
                                  "intensive (so its two trees and lists coincide)")},
            tree=("agglomerative over Delaunay neighbours of block centroids, the pair whose "
                  "merged hull's log score (the list's: as_is for as_is, intensive for the "
                  "others) is best; merges over max_share refused; every node with >= "
                  "min_blocks members a candidate"),
            candidates={"as_is": int(len(ca)), "intensive": int(len(ci)),
                        "intensive_with_msr_null_at_their_size": int(len(cs))},
            seconds_tree={k: sec[metric, k] for k in ("as_is", "intensive")},
            disjoint=f"hulls overlap < {OVERLAP_MAX} of the smaller's area (greedy, rank order)",
            rankings={"as_is": "as_is log score, descending",
                      "intensive_raw": "intensive log score, descending",
                      "intensive_cal": ("p_msr ascending, then z_msr descending, over the "
                                        "candidates with a null at their n_eff under msr")},
            nulls=[dict(name=nl.name, nperm=nperm, seed=SEED, null_index=ni,
                        rng="numpy default_rng([seed, null index, replicate])",
                        field=("block building density n_b / area_b re-dealt among the blocks "
                               "(msr on its log10 over the street adjacency, the observed "
                               "densities re-dealt by the surrogate's ranks), n_b' = d' area_b; "
                               "the whole intensive tree regenerated per replicate"),
                        moran_observed=I_obs,
                        moran_surrogates=dict(
                            mean=float(np.mean([r.moran for r in of_null(reps, ni, nperm)])),
                            sd=float(np.std([r.moran for r in of_null(reps, ni, nperm)]))),
                        seconds_tree_mean=float(np.mean(
                            [r.seconds[metric] for r in of_null(reps, ni, nperm)])),
                        candidates_without_p=int((~np.isfinite(zp[nl.name][1][ci])).sum()),
                        largest_candidate_n_eff_with_p=_f(float(ti.n_eff[ci][np.isfinite(
                            zp[nl.name][1][ci])].max())))
                   for ni, (nl, cal) in enumerate(zip(nulls, cals, strict=True))],
            seconds_nulls_all_metrics=t_null,
            size="n_eff = (sum w)^2 / sum w^2 over the member blocks, w their buildings",
            calibration=(f"as screen_concepts.py but conditional on size: p = (1 + #replicates "
                         f"whose best intensive log score among candidates with n_eff within "
                         f"x/{cn.WINDOW} is >= the region's) / (1 + #replicates with such a "
                         f"candidate), undefined where fewer than {MIN_SUPPORT} of the replicates "
                         f"have one; on log10 n_eff bins of log10({cn.WINDOW})/"
                         f"{cn.BINS_PER_WINDOW}; z = (score - mean) / sd of every null "
                         f"candidate's, Gaussian kernel in log10 n_eff, bandwidth "
                         f"log10({cn.WINDOW})"),
            null_curve=dict(
                n_eff=(10 ** ((g[keep] + 0.5) * cn.STEP)).tolist(),
                **{f"q95_best_{nl.name}": [_f(x) for x in cal.q(0.95)[keep]]
                   for nl, cal in zip(nulls, cals, strict=True)},
                **{f"mean_{nl.name}": [_f(x) for x in cal.mean[keep]]
                   for nl, cal in zip(nulls, cals, strict=True)}))
        res[metric] = dict(params=params, as_is=l_as, intensive_raw=l_raw, intensive_cal=l_cal,
                           follow=dict(as_is=f_as, intensive_raw=f_raw, intensive_cal=f_cal))
    return res


# ---------------------------------------------------------------- screen_clusters' hull scan, MSR

def _msr_hull(r: int) -> float:
    tree, v, msr = _JOB["tree"], _JOB["v"], _JOB["msr"]
    x = msr.surrogates(v, [np.random.default_rng([SEED, 1, r])])[0]
    return float(np.nanmax(tree.build(x)["scan"]))


def clusters_msr(data: dict, city: City, msr: cn.MSR) -> dict:
    """screen_clusters.py's hull-tree scan (its run, wrapped: its clusters and shuffle p), each
    cluster's p also under MSR surrogates of the blocks' log scores (screen_concepts.py's), the
    hull tree rebuilt per replicate, as many replicates as its shuffle's."""
    out: dict = {}
    b = city.b
    for metric in sc.METRICS:
        v = sc.log_scores(data, metric)
        floor = next(f["value"] for f in data["floors"] if f["metric"] == metric)
        tree = sc.Hull(b, city.edges)
        t0 = time.time()
        r = sc.scanned(tree, v, b, floor)
        _JOB.update(tree=tree, v=v, msr=msr)
        with get_context("fork").Pool(WORKERS) as pool:
            nmax = np.asarray(pool.map(_msr_hull, range(tree.nperm), chunksize=1))
        for c in r["scan"]:
            c["p_msr"] = float((1 + (nmax >= c["stat"]).sum()) / (1 + len(nmax)))
        out[metric] = dict(nperm=tree.nperm, seed=SEED,
                           p="the shuffle's p, screen_clusters.py's own (its SeedSequence(seed) "
                             "states)",
                           msr_rng="numpy default_rng([seed, 1, replicate])",
                           msr_null_max=dict(median=float(np.median(nmax)),
                                             max=float(nmax.max())),
                           seconds=time.time() - t0, scan=r["scan"])
        print(f"  clusters {metric}: " + ", ".join(
            f"{c['blocks']} blocks x{c['x_floor']:.2f} p {c['p']:.3f}/{c['p_msr']:.3f}"
            for c in r["scan"][:4]) + f" ({time.time() - t0:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------- the checks

def _partitions(box: tuple[float, float, float, float], rng: np.random.Generator) -> list:
    """Two partitions of a box into convex cells (counter-clockwise): a 5 x 4 grid of
    rectangles, and a fan of triangles from a random interior point to 16 boundary points."""
    x0, y0, x1, y1 = box
    xs, ys = np.linspace(x0, x1, 6), np.linspace(y0, y1, 5)
    grid = [np.array([[xs[i], ys[j]], [xs[i + 1], ys[j]], [xs[i + 1], ys[j + 1]],
                      [xs[i], ys[j + 1]]]) for i in range(5) for j in range(4)]
    c = np.array([x0 + (x1 - x0) * rng.uniform(0.3, 0.7), y0 + (y1 - y0) * rng.uniform(0.3, 0.7)])
    t = np.linspace(0, 1, 5)[:-1]
    ring = np.vstack([np.column_stack([x0 + (x1 - x0) * t, np.full(4, y0)]),
                      np.column_stack([np.full(4, x1), y0 + (y1 - y0) * t]),
                      np.column_stack([x1 - (x1 - x0) * t, np.full(4, y1)]),
                      np.column_stack([np.full(4, x0), y1 - (y1 - y0) * t])])
    fan = [np.array([c, ring[i], ring[(i + 1) % len(ring)]]) for i in range(len(ring))]
    return [grid, fan]


def _synthetic(k: float) -> tuple[City, F64]:
    """Four blocks (a square, an L, a square with a hole, a triangle) scaled by k, buildings
    scaled by k^2 (fixed density)."""
    polys = [shapely.Polygon([(0, 0), (40, 0), (40, 30), (0, 30)]),
             shapely.Polygon([(50, 0), (90, 0), (90, 15), (65, 15), (65, 40), (50, 40)]),
             shapely.Polygon([(0, 40), (40, 40), (40, 80), (0, 80)],
                             [[(10, 50), (10, 60), (25, 60), (25, 50)]]),
             shapely.Polygon([(50, 50), (95, 55), (60, 85)])]
    polys = [shapely.affinity.scale(p, k, k, origin=(0, 0)) for p in polys]
    data = {"rings": [[list(p.exterior.coords)] + [list(r.coords) for r in p.interiors]
                      for p in polys],
            "n": [30.0 * k * k, 55.0 * k * k, 12.0 * k * k, 70.0 * k * k],
            "area_m2": [p.area for p in polys]}
    return city_of(data), np.asarray(data["n"])


def check() -> bool:
    rng = np.random.default_rng(SEED)
    bad = 0
    root = Path("~/src/wt_demo").expanduser()
    cities = {nm: json.loads((root / "examples" / "screen-map" / f"{nm}.json").read_text())
              for nm in CITIES}
    built = {nm: city_of(d) for nm, d in cities.items()}
    # 1. proration over a partition of the plane sums to the city's buildings
    for nm, city in built.items():
        bx = city.pl.box
        box = (bx[:, 0].min() - 1.0, bx[:, 1].min() - 1.0, bx[:, 2].max() + 1.0,
               bx[:, 3].max() + 1.0)
        for lab, cells in zip(("grid", "fan"), _partitions(box, rng), strict=True):
            tot = sum(prorate(city.pl, sc.hull(c), NOHULL, city.b.w)[0] for c in cells)
            err = abs(tot - city.b.w.sum()) / city.b.w.sum()
            print(f"  proration {nm} {lab} ({len(cells)} cells): sum n(H) {tot:.3f} of "
                  f"{city.b.w.sum():.0f} buildings, rel err {err:.1e}")
            bad += err > 1e-9
    # 2. the factorisation as_is = sqrt(n) intensive (depth_density_proxy), and n/P^2 already
    # intensive
    n = 10 ** rng.uniform(0, 5, 10_000)
    A = 10 ** rng.uniform(2, 8, 10_000)
    P = np.sqrt(4 * np.pi * A) * rng.uniform(1, 25, 10_000)
    dd, dc = SCORES["depth_density_proxy"], SCORES["density_compactness"]
    e1 = float(np.max(np.abs(dd.as_is.f(n, A, P) - np.sqrt(n) * dd.intensive.f(n, A, P))
                      / dd.as_is.f(n, A, P)))
    e2 = float(np.max(np.abs(dc.as_is.f(n, A, P) - dc.intensive.f(n, A, P))
                      / dc.as_is.f(n, A, P)))
    print(f"  factorisation on 10,000 (n, A, P): depth_density_proxy as_is vs sqrt(n) intensive "
          f"rel err {e1:.1e}; density_compactness as_is vs intensive {e2:.1e}")
    bad += (e1 > 1e-12) + (e2 > 1e-12)
    # 3. scale invariance at fixed density: a synthetic region's hull, coordinates x k
    pts = np.array([[5.0, 5.0], [80.0, 10.0], [70.0, 70.0], [10.0, 75.0]])
    base = None
    for k in (1.0, 3.7, 41.0):
        city, w = _synthetic(k)
        h = sc.hull(pts * k)
        nk, _ = prorate(city.pl, h, NOHULL, w)
        Ak, Pk = sc.area_of(h), perimeter_of(h)
        v = {m: (float(s.as_is.f(nk, Ak, Pk)), float(s.intensive.f(nk, Ak, Pk)))
             for m, s in SCORES.items()}
        if base is None:
            base = v
            continue
        for m in SCORES:
            ei = abs(v[m][1] / base[m][1] - 1)
            grow = v[m][0] / base[m][0]
            want = k if m == "depth_density_proxy" else 1.0
            print(f"  scale x{k}: {m} intensive ratio {v[m][1] / base[m][1]:.12f}, as_is ratio "
                  f"{grow:.9f} (want {want})")
            bad += (ei > 1e-9) + (abs(grow / want - 1) > 1e-9)
    # 4. the size-conditional p against brute force: replicates with
    # no candidate near a size neither count for nor against a region there
    R, G = 40, int(np.ceil(3.0 / cn.STEP)) + 2
    reps = []
    for r in range(R):
        ne = 10 ** rng.uniform(0, 2.0 if r % 3 else 3.0, 50)   # a third reach the large sizes
        reps.append(cn.binned(ne, rng.normal(size=50) + np.log10(ne), G))
    cal = cn.calibrate(reps)
    ne = 10 ** rng.uniform(0, 3.0, 300)
    st = rng.normal(size=300) + np.log10(ne) + 1.0
    pc = conditional_p(cal, ne, st)
    pb = np.full(300, np.nan)
    for i, (gi, s_) in enumerate(zip(cn.bins_of(ne), st, strict=True)):
        col = cal.wbest[:, gi]
        have = col[np.isfinite(col)]
        if len(have) >= MIN_SUPPORT * R:
            pb[i] = (1 + (have >= s_).sum()) / (1 + len(have))
    pe = int((np.isnan(pc) != np.isnan(pb)).sum()
             + (np.abs(pc - pb)[np.isfinite(pb)] > 1e-12).sum())
    print(f"  size-conditional p on 300 regions over 40 synthetic replicates: "
          f"{int(np.isfinite(pb).sum())} defined; unlike brute force {pe}")
    bad += pe
    # 5. the agglomeration: every node's hull is its members' convex hull, its n(H), cover,
    # A and P the exact ones, its counts its members'
    city = built["nairobi"]
    t = agglomerate(city, city.b.w, SCORES["depth_density_proxy"].intensive, MAX_SHARE)
    N = len(city.b.w)
    P_ = np.asarray(city.b.polys, dtype=object)
    tree = shapely.STRtree(P_)
    merges = np.arange(N, len(t.kids))
    nodes = np.unique(np.concatenate([rng.choice(merges, 300, replace=False),
                                      merges[np.argsort(-t.blocks[merges])[:30]]]))
    hb = nb_ = cb = 0
    for node in nodes:
        mem = t.members(int(node))
        ref = shapely.convex_hull(shapely.union_all(P_[mem]))
        H = shapely.Polygon(t.hulls[node])
        if (shapely.area(shapely.symmetric_difference(ref, H)) > 1e-9 * ref.area
                or len(mem) != t.blocks[node] or abs(city.b.w[mem].sum() - t.W[node]) > 1e-6):
            hb += 1
        idx = tree.query(H, predicate="intersects")
        a = shapely.area(shapely.intersection(P_[idx], H))
        n_ref = float((city.b.w[idx] * a / city.pl.area[idx]).sum())
        if abs(n_ref - t.n[node]) > 1e-9 * n_ref or abs(a.sum() - t.cover[node]) > 1e-9 * a.sum():
            nb_ += 1
        if (abs(H.area - t.A[node]) > 1e-9 * H.area
                or abs(H.length - t.P[node]) > 1e-9 * H.length):
            cb += 1
    print(f"  nairobi depth_density_proxy intensive tree: {len(merges)} merges; of {len(nodes)} "
          f"nodes, hull not its members' convex hull (or counts off) {hb}, n(H) or cover unlike "
          f"shapely's exact proration {nb_}, A or P off {cb}")
    bad += hb + nb_ + cb
    return bad == 0


def _fault_drop_ratio() -> None:
    """prorate counts every block H meets whole."""
    global prorate

    def whole(pl: Plane, h: F64, hb: F64, w: F64) -> tuple[float, float]:
        meets = ~((pl.box[:, 2] <= h[:, 0].min()) | (pl.box[:, 0] >= h[:, 0].max())
                  | (pl.box[:, 3] <= h[:, 1].min()) | (pl.box[:, 1] >= h[:, 1].max()))
        idx = np.flatnonzero(meets)
        a = np.array([sum(abs(_ring_clip_full(pl, r, h)) for r in
                          range(pl.block_ring[j], pl.block_ring[j + 1])) for j in idx])
        return float(w[idx][a > 0].sum()), float(pl.area[idx][a > 0].sum())

    prorate = whole


def _ring_clip_full(pl: Plane, r: int, h: F64) -> float:
    return _ring_clip(pl.xy, int(pl.ring_ptr[r]), int(pl.ring_ptr[r + 1]), h,
                      np.arange(len(h), dtype=np.int64), len(h))


def _fault_drop_sqrt() -> None:
    """depth_density_proxy's intensive form loses the square root of A/P^2."""
    SCORES["depth_density_proxy"] = SCORES["depth_density_proxy"]._replace(
        intensive=Formula("intensive", lambda n, A, P: n / A * (A / P ** 2)))


def _fault_as_is() -> None:
    """the intensive scores are the as_is ones."""
    for m, s in list(SCORES.items()):
        SCORES[m] = s._replace(intensive=Formula("intensive", s.as_is.f))


def _fault_skip_member() -> None:
    """a merge's hull drops a vertex of the second cluster's hull (its block skipped)."""
    real = City.merged

    def skip(self: City, px: F64, py: F64, hb: F64, w: F64) -> tuple:
        return real(self, px, py[1:] if len(py) > 3 else py, hb, w)

    City.merged = skip  # type: ignore[method-assign]


def _fault_unconditional() -> None:
    """p counts replicates with no candidate of the size as below the region (screen_concepts's
    p), wherever any replicate has one."""
    global conditional_p

    def uncond(cal: cn.Calibration, n_eff: F64, stat: F64) -> F64:
        p = cal.p(n_eff, stat)
        return np.where(np.isfinite(cal.wbest[-1, cn.bins_of(n_eff)]), p, np.nan)

    conditional_p = uncond


FAULTS: dict[str, Callable[[], None]] = {"unconditional": _fault_unconditional,
                                         "drop-ratio": _fault_drop_ratio,
                                         "drop-sqrt": _fault_drop_sqrt,
                                         "as-is-intensive": _fault_as_is,
                                         "skip-member": _fault_skip_member}


# ---------------------------------------------------------------- the run

def main(argv: list[str]) -> None:
    if argv and argv[0] == "check":
        fault = argv[1] if len(argv) > 1 else None
        if fault is not None and fault not in FAULTS:
            raise SystemExit(f"faults: {', '.join(FAULTS)}")
        if fault is not None:
            FAULTS[fault]()
        ok = check()
        print(f"check{'' if fault is None else f' under the fault {fault}'}: "
              f"{'pass' if ok else 'FAIL'}")
        if not ok and fault is None:
            raise SystemExit(1)
        if ok and fault is not None:
            raise SystemExit(f"the fault {fault} went unseen: the check guards nothing")
        return
    if len(argv) < 2 or not set(argv[2:]) <= set(CITIES):
        raise SystemExit(__doc__)
    root, out = Path(argv[0]).expanduser(), Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name in argv[2:] or CITIES:
        t0 = time.time()
        data = json.loads((root / "examples" / "screen-map" / f"{name}.json").read_text())
        city = city_of(data)
        A = cn.adjacency(city.b)
        t1 = time.time()
        _, U = cn.mem_basis(A)
        print(f"{name}: {len(city.b.w)} blocks; Moran eigenvectors {time.time() - t1:.0f}s",
              flush=True)
        msr = cn.MSR(U)
        (out / f"clusters_msr_{name}.json").write_text(json.dumps(clusters_msr(data, city, msr)))
        res = scanned(data, city, A, [cn.Shuffle(), msr], NPERM[name])
        (out / f"hulls_{name}.json").write_text(json.dumps(res))
        for metric, r in res.items():
            for k, cal in (("as_is", False), ("intensive_raw", True), ("intensive_cal", True)):
                for c in r[k][:5]:
                    p = ""
                    if cal:
                        p = f" p {c['p_msr']}/{c['p_shuffle']} z {c['z_msr']}"
                    print(f"    {metric} {k}: x{c['scores'][metric]['x_floor']:.2f} (blocks "
                          f"x{c['scores'][metric]['blocks_only_x_floor']:.2f}) n "
                          f"{c['buildings']:.0f} blocks {c['blocks']} n_eff {c['n_eff']:.1f} "
                          f"A {c['area_km2']:.2f} km2 street {c['street_share']:.2f}{p} "
                          f"({c['cx'] / 1e3:.1f}, {c['cy'] / 1e3:.1f}) km", flush=True)
        del U, msr
        print(f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
