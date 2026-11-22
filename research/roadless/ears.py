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

Candidate ears are a Strategy (Candidates): `paths` (Paths), cheapest paths across the faces (a
metre of centreline at x costs 1 + `bld` s(x), s(x) the building share of the road's
cross-section there: the share of the disc of radius ROAD_W / 2 about x the footprints cover,
`road_share`, owner 2026-11-22; binf the limit b -> inf, lexicographic: the least integral of s
along the path, B, then the least length; grown ground closed), on a PIX m raster of them or on
their vector geometry (the Router), from the faces'
sinks (EarWorld.sinks: pixels on the street or a grounded road, not a block edge with no exit nor
a paved way inside the block, where an ear would dead-end), sampled every SRC_M
m, to any other sink pixel of the same face and ring (EarWorld.arcs: the face's boundary
component; a path to another, a hole's, splits nothing) at least `ear_m` away in a straight
line whose path cuts its face into two pieces each of growth.FACE_M2 or more, less CUT_ALLOW
PATCH_R a metre of path for its corridor (`cuts`, owner 2026-11-22: rank only ears that can be
built; build refuses an ear whose faces of FACE_M2 and more do not rise): the pieces by Green's
theorem, the signed area the path sweeps (Reach.area, one pass with the search: a tree's sum,
Theta's per segment, Eikonal's transported, Visibility's per piece) closed along the face's
contour on the raster (Arcs; a saddle, pixels of one label meeting at a corner only, joined as
the mesh is at that corner: a fixed rule miscounted both ways on 21634, keeping ears whose piece
a one-pixel neck of the face left joined, dropping ears across a road's corner). Without it,
at binf on 5618 every source's best was a B-free path
in the open strip along the street that cut off 41 -- 73 m^2, and the build stopped after 1 ear
(grid8) or 2 (theta): per
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

Where a path may end is a Strategy of `paths` (Ends): `loop` (Loop), the above; `spur<min>-<max>`
(Spur, owner 2026-11-03: Idea A, spurs in Shock Reblock), dead ends allowed, so spurs and loops
compete at one budget: a path may also end on the face's open ground (no building, no sink)
between min and max m from its source in a straight line, a spur (Ear.spur), its corridor joined
to the network at its start only and built without splitting its face. A spur lies in one face
and leaves its cells as they were, so its exact gain is the same restricted solve. `divides` cuts
run sink to sink by construction: their ears are loops, and their names carry no ends.

A term added to the escape objective of `paths` is a Strategy (Attract, owner 2026-11-03: Idea C,
detour-saving attraction), a required plan token after the ends: NoDetour (`.nodt`), the objective
alone; Detour(lam) (`.dt<lam>`), each ear credited lam P0 S / S0, S its detour saving (`savings`:
over pairs of homes, each at the road point nearest it, the mean metres by which the trip between
their road points along the network gets shorter with the ear's path a link; the network the state's
sinks, its road points one per SRC_M square, `roads_of`), S0 the mean such trip on the street alone.
A spur shortens no trip and earns nothing; the ear that later closes it is credited the whole loop.
An ear is scored (gain + credit) / cost at every stage, its credit times the last step's first order
over exact gain where the first order stands in for the exact gain (build).

    E=research/roadless/ears.py                        # from the repo root, PYTHONPATH=.
    uv run python $E run <block> <plan>
    uv run python $E check decouple|first|ends|spur|detour|offer <block> [fault]
    uv run python $E check horseshoe|names [fault]
    uv run python $E check router cost|straight|building|gain|area [fault]
    uv run python $E check vector|share|limit|rings|binf <block> [fault]

plan: ears.<paths.<router>.b<bld>.f<face_m2>.e<ear_m>.<loop|spur<min_m>-<max_m>>.<nodt|dt<lam>>
|divides.p<tau>.f<face_m2>.e<ear_m>>.gl<h>Croad.n<nu>s<price>.M<pairs>.K<exact>.D<D> (a face under
face_m2 not split, an ear's ends at least ear_m apart, a spur's end min_m to max_m from its start,
<router> grid8|pulled|theta|eikonal|visibility (Router; visibility at b0 only, eikonal not at
binf), <bld> a number or inf (the limit, lexicographic),
`.nodt` NoDetour, `.dt<lam>` Detour; a divides name carries neither ends nor attraction, its ears
loops with no credit by construction), its name adding the code's
revision (.v<REV>); rows in growth_rows/<block>/<name>/ in growth.py's layout (steps.parquet,
picks.parquet one row per ear with whether it is a spur and its detour saving and credit,
bridges.parquet empty, tri_step.parquet), so the growth runs' frames and plates read them.

Checks, each run both ways:
decouple: a state with a few ears on the block, the global P against the sum of its faces' P
  (one solve restricted to the ungrown triangles' cells: the faces, with the roads' cells held
  at 0, are its diagonal blocks), within TOL_DEC relative. Fault `weak-leak` (h 25 m: roads
  barely grounded, the faces coupled) fails it.
first: on that state, for the top ears (their open triangles only, kt0 >= OPEN_KT, so the
  opening's term is nil) the exact drop at a small leak (h FIRST_H: dP linear in it) against the
  first order (sum over the corridor of u_c^2 l_t), within TOL_FIRST relative. Fault `abs-u`
  (|u| in place of u^2) fails it.
ends: every router's top candidate ears, at b0, b20 and binf (`routers_at`), whose ideal
  corridor (the line buffered by PATCH_R) separates their face into pieces of ENDS_M2 and more,
  each split it (its faces of ENDS_M2 and more rise; a piece under FACE_M2 build then refuses);
  the others reported. Faults `spur` (an ear's last
  third dropped: a dead end) and `vertex-sites` (a mutation: a corridor's patches at its
  line's vertices only, which for an any-angle line leaves gaps) fail it.
spur: the spur check plan's top SPUR_EARS candidate spurs each a corridor in one piece, touching
  the network, that splits nothing (its end on open ground). Fault `both-ends` (a spur's
  corridor extended at its end too, as a loop's is) fails it.
detour: after DETOUR_EARS loop ears, the detour plan's costed ears' savings (Ear.saving) and
  DETOUR_TARGETS links from the first one's source at once (`savings`' many targets) against all
  pairs' distances again with the link an edge of the network's graph, within TOL_DETOUR of the
  largest. Fault `one-way` (the trips counted from a's side only: savings' factor 2 dropped) fails
  it.
horseshoe: synthetic horseshoes (no block; Experiment 1 of the owner's doc): the closing link's
  saving rises strictly with the arms at a fixed gap, and its saving per metre of link strictly as
  the gap closes at fixed arms. Fault `crow` (straight-line distances for the road distances: no
  link shortens a trip) fails it.
names: every check plan's name round-trips (Plan.of(name).name), every router's too at b0,
  b20 and binf, and each paths one with its attraction token dropped (the old grammar's NoDetour)
  or its router token dropped (the grammar before 2026-11-19, Grid8 implied) is refused, as are
  Visibility at b0.05, b20 and binf and Eikonal at binf. Fault `keep` (the tokens kept: the name
  parses) fails it.
router: the routers on synthetic rasters, each fault a mutation of a router's code (`router_check`:
  cost, straight, building, gain, area).
share: the road's building share (`road_share`, the raster's pixel centres) against an
  independent exact one (the disc a polygon of 4 x SHARE_CHECK_SEGS sides, intersected with each
  sampled point's footprints one by one) on SHARE_SAMPLE pixels within ROAD_W / 2 + PIX of a
  footprint, to TOL_SHARE; 0 at every pixel ROAD_W / 2 or farther from every footprint; 1 at every
  pixel whose disc lies in one footprint. Faults share-radius (the disc ROAD_W wide in radius)
  and share-raster (s the 0/1 building raster) fail it.
limit: on the block's first state, LIMIT_SOURCES sources to LIMIT_TARGETS goal pixels each,
  Grid8's and Theta's paths at every b of LIMIT_BLDS and binf, each path's B (s integrated along
  its line, `along`) and length L: Grid8 (exact) has B non-increasing and L non-decreasing in b
  (any optimum of L + b B does: (b2 - b1)(B2 - B1) <= 0), binf's B no more than any b's and
  B(b) - B(inf) <= (L(inf) - L(b)) / b (binf's path is no cheaper at b), each to TOL_LIMIT
  relative; Theta (not exact) its violations reported, not judged. Faults no-lex (Grid8's binf
  ignores B: the shortest path) and length-first (Grid8's binf orders by length, then B) fail
  it.
rings: on a block with a hole (5810: its 132 m^2 one, a ring of its own), the rings plan built
  RINGS_STEPS ears: the face's sinks on two rings or more at the first state, the build not
  stopped short, and every candidate offered on the way ending on its source's ring. Fault
  one-ring (every sink on one ring: rev 3's targets) fails it (5810, Theta at b0: after the
  first ear every source's best target is a sink on the hole, its 8 ears split nothing, and the
  build stops at D 0.020, as rev 3's runs did).
binf: on a block (5618, 5810), Grid8 and Theta at binf (BINF_PLANS) build BINF_STEPS ears, not
  stopped short (the ears refused on the way reported). Fault no-cut (the cut test dropped)
  fails it (5618: Grid8 builds 1 ear, Theta 2: "no candidate ear splits a face").
offer: after OFFER_EARS ears of the divides check plan (faces of very different sizes), the
  `pairs` divides Divides costs hold a splitting one within TOL_OFFER of the best first order
  per cost among every face's divides. Fault `persistence` (rev 1's order, the most persistent
  across faces: persistence is relative to the face's largest u) fails it.
"""
from __future__ import annotations

import heapq
import re
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple, Protocol, TypeAlias

import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely
import shapely.ops
from numba import njit
from numpy.typing import NDArray
from scipy import ndimage
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from skimage import measure

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
STRETCH_M = 40.0       # m: divides' harmonic split: the face's boundary in stretches this wide
DUP_M = 2 * PIX        # m: Divides: a divide's pixel this near a kept divide's is shared
DUP_SHARE = 0.5        # Divides: a divide sharing this much of its pixels with a kept one dropped
BLD_G = 0.01           # divides' harmonic split: a pixel edge's conductance through a building
STEP_M = 0.25          # m: an ear's centreline sampled this finely (EarWorld.corridor)
LOCATE_K = 12          # EarWorld.locate: candidate triangles per point
CHUNK = 32             # sources per Dijkstra call (its distance and predecessor rows held)
REV = 4                # 2: Divides offers by first-order gain per metre, near-duplicates dropped
                       # (rev 1: the most persistent across faces); 3: the two floors are the
                       # plan's (`.f<face_m2>.e<ear_m>`, required); every rev 1 and 2 run had
                       # f2000 e30, the constants they replace; 4 (2026-11-22): b charges for the
                       # building share of the road's cross-section (`road_share`), not for the
                       # centreline's pixel being a building's, and binf is the limit b -> inf
                       # (lexicographic), not buildings impassable; b0 unchanged (rev 3's b0
                       # rows renamed: atlas/routers/migrate_bcost.py)
Q_B = 2.0 ** -30       # m: binf: a piece's B (s integrated along it) rounded to this (`_qb`), so
                       # a path's B is an exact sum (to 2^23 m) and two paths tie in B exactly
SHARE_SEGS = 32        # road_share: the disc a regular polygon of 4 x this sides, a share its
                       # cover over the polygon's own area (1 exactly where a footprint holds it;
                       # the polygon's area 4.0e-4 under the circle's)
TOL_DEC = 1e-2         # decouple: relative (19570, 3 ears at h 0.25: 2.4e-3; at h 25, 0.24)
DEC_EVERY = 10         # build: the decoupling gap measured every this many ears
TOL_FIRST = 2e-3       # first: relative (19570: 3.3e-4 at FIRST_H; 3.3e-2 at h 250, the drop
                       # not yet linear)
WEAK_H = 25.0          # m: decouple's fault: a weak leak
FIRST_H = 2500.0       # m: first's leak (the drop linear in it: an ear's leak times its own
                       # resistance to ground far below 1)
OPEN_KT = 0.999        # first: the ears' open triangles only (no opening term to speak of)
SPUR_EARS = 4          # spur: the top candidate spurs judged
ENDS_M2 = 10.0         # m^2: ends: a face this large counts (a corridor with a gap cuts off none;
                       # binf's B-free chords along the street's margin can cut off less than
                       # FACE_M2: 5618 theta, 48 m^2)
DETOUR_EARS = 3        # detour: loop ears built before the links are judged
DETOUR_TARGETS = 8     # detour: links from the first candidate's source judged at once
DETOUR_SEED = 0        # detour: their targets' draw
DETOUR_STRETCH = 1.2   # detour: their lengths, this times the straight line (plus PIX)
TOL_DETOUR = 1e-3      # detour: |S - all pairs again| over the largest S (float32 distances)
HS_W = 40.0            # m: horseshoe: the arms' spacing
HS_HOME_M = 4.0        # m: horseshoe: the homes' grid
HS_GAP_M = 14.0        # m: horseshoe: the gap the arms are compared at (its bars' ends this apart)
HS_ARMS = (20.0, 40.0, 80.0, 160.0)   # m: horseshoe: the arms compared
HS_ARM_M = 80.0        # m: horseshoe: the arms the gaps are compared at
HS_GAPS = (24.0, 16.0, 8.0)           # m: horseshoe: the gaps compared, closing
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
    """A candidate: its face, its path's pixels and centreline (Route), whether it is a spur (its
    path ends on the face's open ground, not at a sink: its corridor joins the network at its
    start only, and it need not split its face), its corridor's new triangles, dD, dC, its cost,
    first-order gain, its detour saving (m: Detour; 0 without one, and for a spur, which shortens
    no trip) and the credit it earns in P (added to its first-order and exact gains wherever an
    ear is scored), exact gain (nan unless rescored)."""
    face: int
    path: I64
    line: F64
    spur: bool
    tris: I64
    dD: float
    dC: float
    cost: float
    first: float
    saving: float
    credit: float
    exact: float


class Raster(NamedTuple):
    """The block on a PIX raster: per pixel (flat) its nearest triangle, whether its centre is in
    a building (bld, 0 or 1: Ends' open ground, Divides' harmonic split) and the building share
    of a road's cross-section at its centre (share, `road_share`: the paths' cost); the
    8-neighbour grid's edges (a, b) and their lengths."""
    shape: tuple[int, int]
    lo: F64
    xy: F64
    tri: I64
    bld: F64
    share: F64
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
    share = road_share(np.asarray(geo.polys), xy, r=gr.ROAD_W / 2.0)
    ea, eb, el = _grid_edges(shape)
    return Raster(shape=shape, lo=lo, xy=xy, tri=tri, bld=bld, share=share, ea=ea, eb=eb, el=el)


def road_share(polys: NDArray[np.object_], xy: F64, *, r: float) -> F64:
    """Per point of xy the building share of a road's cross-section there (owner 2026-11-22: what
    b charges for): the share of the disc of radius r about it that the footprints `polys` cover,
    exactly (the disc a regular polygon of 4 SHARE_SEGS sides, its intersection with each part of
    the footprints' union summed over its own area; the pairs within r by an STRtree), so 0
    where every footprint is r or more away and 1 where one holds the disc."""
    parts = shapely.get_parts(shapely.union_all(polys))
    out = np.zeros(len(xy))
    if not len(parts):
        return out
    ip, jp = shapely.STRtree(parts).query(shapely.points(xy), predicate="dwithin", distance=r)
    ring = np.asarray(shapely.get_coordinates(
        shapely.Point(0.0, 0.0).buffer(r, quad_segs=SHARE_SEGS).exterior))
    area = float(shapely.area(shapely.polygons(ring)))
    pts, pos = np.unique(ip, return_inverse=True)
    discs = shapely.polygons(ring[None] + xy[pts][:, None])
    cover = shapely.area(shapely.intersection(discs[pos], parts[jp]))
    out[pts] = np.minimum(np.bincount(pos, weights=cover, minlength=len(pts)) / area, 1.0)
    return out


def _grid_edges(shape: tuple[int, int]) -> tuple[I64, I64, F64]:
    """A PIX raster's 8-neighbour edges (a, b) and their lengths, pixels flat (row-major)."""
    ea, eb, el = [], [], []
    idx = np.arange(shape[0] * shape[1]).reshape(shape)
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        j0, j1 = max(0, -dj), shape[1] - max(0, dj)
        a = idx[0:shape[0] - di, j0:j1]
        b = idx[di:shape[0], j0 + dj:j1 + dj]
        ea.append(a.ravel())
        eb.append(b.ravel())
        el.append(np.full(a.size, PIX * np.hypot(di, dj)))
    return np.concatenate(ea), np.concatenate(eb), np.concatenate(el)


@njit(cache=True)
def _tree_sum(pred: NDArray[np.int32], order: I64, w: F64) -> F64:
    """Per node the sum of `w` down its shortest-path tree from the source (pred -9999 at it)."""
    acc = np.zeros(len(w))
    for k in range(len(order)):
        v = order[k]
        p = pred[v]
        acc[v] = w[v] + (acc[p] if p >= 0 else 0.0)
    return acc


@njit(cache=True)
def _tri(x0: float, y0: float, x1: float, y1: float) -> float:
    """The signed area the segment (x0, y0) -- (x1, y1) sweeps about the origin, (x0 y1 - x1 y0)
    / 2: summed along a closed curve, the area it encloses (positive anticlockwise)."""
    return 0.5 * (x0 * y1 - x1 * y0)


@njit(cache=True)
def _sweep(p: int, q: int, ny: int) -> float:
    """`_tri` of the segment between the centres of pixels p and q (flat, ny columns), about the
    raster's lower corner."""
    return _tri((p // ny + 0.5) * PIX, (p % ny + 0.5) * PIX, (q // ny + 0.5) * PIX,
                (q % ny + 0.5) * PIX)


@njit(cache=True)
def _tree_area(pred: NDArray[np.int32], order: I64, ny: int) -> F64:
    """Per node the signed area its path down the shortest-path tree sweeps (`_sweep`)."""
    out = np.zeros(len(pred))
    for k in range(len(order)):
        v = order[k]
        p = pred[v]
        if p >= 0:
            out[v] = out[p] + _sweep(p, v, ny)
    return out


# ---------------------------------------------------------------- routers (owner 2026-11-19)


class Terrain(NamedTuple):
    """A state's routing field on the PIX raster (Paths.ears): per pixel its face where a path may
    run (-1 elsewhere: grown ground, off the block, a face under face_m2), the building share of a
    road's cross-section at its centre (`road_share`) and its first-order gain, `bld` what a
    metre of centreline costs per unit share (a metre in a pixel's cell costs 1 + bld * share;
    inf: the limit, lexicographic: a path's B, share integrated along it, first, then its length),
    the raster's shape, lower corner, centres and 8-neighbour edges; `goal` per pixel whether
    some source's path may end there (Ends.goal), and `vec` the state's vector geometry (Shapes:
    the vector router's faces; the raster routers read none). A path stays in its source's
    face."""
    shape: tuple[int, int]
    lo: F64
    xy: F64
    face: I64
    share: F64
    gain: F64
    bld: float
    ea: I64
    eb: I64
    el: F64
    goal: B
    vec: Shapes

    def lex(self) -> bool:
        """Whether paths are compared at the limit (binf): by B, then length."""
        return bool(np.isinf(self.bld))

    def per_m(self) -> F64:
        """Per pixel the cost of a metre through its cell, 1 + bld * share; at binf the limit's
        second key's, the length's (1 everywhere: its first, B, is the share's integral)."""
        if self.lex():
            return np.ones(len(self.share))
        return np.asarray(1.0 + self.bld * self.share)


class Route(NamedTuple):
    """A path from a source pixel to a target: `path` the pixels its centreline turns at (every
    pixel of a grid path; the vertices of an any-angle one; the pixels a descent visits, Eikonal),
    its first the source and its last the target; `line` the centreline (m), from the source's
    centre to the target's."""
    path: I64
    line: F64


class Reach(Protocol):
    """One source's search (Router.reaches): its source pixel, per pixel its path's cost (at binf
    its B, the limit's first key), the first-order gain summed along it (inf / 0 where
    unreached) and the signed area it sweeps about the raster's lower corner (the integral of
    (x dy - y dx) / 2 along it: with Arcs, the pieces its corridor cuts its face into, `cuts`),
    the path's length on demand (Detour; at binf the limit's
    second key), and the path to a pixel."""
    @property
    def src(self) -> int: ...

    @property
    def cost(self) -> F64: ...

    @property
    def gain(self) -> F64: ...

    @property
    def area(self) -> F64: ...

    def length(self) -> F64: ...

    def route(self, t: int) -> Route: ...


class Router(Protocol):
    """How a source's cheapest paths are searched and drawn (owner 2026-11-19: "why do we need to
    discretize the directions at all?"), a required plan token after `paths`: Grid8 (`grid8`),
    the 8-neighbour grid; Pulled (`pulled`), its paths string-pulled once chosen; Theta
    (`theta`), any-angle search; Eikonal (`eikonal`), fast marching, refused at binf;
    Visibility (`visibility`), exact on the vector face at b0, refused elsewhere. Steiner
    (`steiner<k>`, a (1 + eps)
    Steiner-point graph at any b) was deleted as dominated (no better L_P than Theta, 10 -- 17x
    Grid8's routing time): the wiki's reblock/steiner-router-is-dominated, commit 1a92867c for
    its code and numbers."""
    @property
    def name(self) -> str: ...

    def reaches(self, ter: Terrain, src: I64) -> Iterator[Reach]:
        """Per source pixel of `src`, in order, its search."""
        ...


@njit(cache=True)
def _seg(p0: int, p1: int, ny: int, face: I64, f: int, c: F64, sh: F64,
         gp: F64) -> tuple[bool, float, float, float]:
    """The straight segment between the centres of pixels p0 and p1 (flat, ny columns):
    (whether every cell it crosses for a positive length is of face f, the cost field c (per
    metre) integrated along it, the gain gp (per pixel) integrated along it per PIX of length,
    the share sh integrated along it: its B).
    Its cells by a DDA in integers (crossings of the cells' edges at t = (2k + 1) / (2 |d|)), so
    a segment through a corner crosses into the diagonal cell, its two side cells touched at a
    point only (the 8-neighbour grid's diagonal move)."""
    i0, j0 = p0 // ny, p0 % ny
    i1, j1 = p1 // ny, p1 % ny
    dx, dy = i1 - i0, j1 - j0
    ax, ay = abs(dx), abs(dy)
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    L = PIX * np.sqrt(float(dx * dx + dy * dy))
    i, j, kx, ky = i0, j0, 0, 0
    t0 = 0.0
    cost = 0.0
    gain = 0.0
    blk = 0.0
    while True:
        xo, yo = kx < ax, ky < ay
        if not xo and not yo:
            t1, step = 1.0, 0
        elif xo and (not yo or (2 * kx + 1) * ay < (2 * ky + 1) * ax):
            t1, step = (2 * kx + 1) / (2.0 * ax), 1
        elif yo and (not xo or (2 * ky + 1) * ax < (2 * kx + 1) * ay):
            t1, step = (2 * ky + 1) / (2.0 * ay), 2
        else:
            t1, step = (2 * kx + 1) / (2.0 * ax), 3
        q = i * ny + j
        ds = (t1 - t0) * L
        if ds > 0.0:
            if face[q] != f:
                return False, cost, gain, blk
            cost += c[q] * ds
            gain += gp[q] * ds / PIX
            blk += sh[q] * ds
        if step == 0:
            break
        if step & 1:
            i += sx
            kx += 1
        if step & 2:
            j += sy
            ky += 1
        t0 = t1
    return True, cost, gain, blk


@njit(cache=True)
def _qb(b: float) -> float:
    """A piece's B rounded to Q_B (the limit's: sums of multiples of Q_B are exact, so two paths
    tie in B exactly when their pieces' rounded B sum alike)."""
    return float(np.round(b / Q_B) * Q_B)


@njit(cache=True)
def _better(c1: float, b1: float, l1: float, c2: float, b2: float, l2: float,
            lex: bool) -> bool:
    """Whether path 1 is strictly cheaper than path 2: by cost (c), or at the limit (lex) by B
    first, then length."""
    if lex:
        return b1 < b2 or (b1 == b2 and l1 < l2)
    return c1 < c2


@njit(cache=True)
def _pull(path: I64, ny: int, face: I64, c: F64, sh: F64, gp: F64, lex: bool) -> I64:
    """`path` (a grid path's pixels) string-pulled (Pulled): from each kept vertex the farthest
    later pixel the straight segment to which stays in the face and costs (the cost field
    integrated along it) no more than the path between them (at the limit, lex: B no more, and
    on a tie in B no longer); its vertices."""
    n = len(path)
    f = face[path[0]]
    cum = np.zeros(n)
    cb = np.zeros(n)
    cl = np.zeros(n)
    for k in range(1, n):
        _, cs, _, bs = _seg(path[k - 1], path[k], ny, face, f, c, sh, gp)
        cum[k] = cum[k - 1] + cs
        cb[k] = cb[k - 1] + _qb(bs)
        a0, a1 = path[k - 1], path[k]
        cl[k] = cl[k - 1] + PIX * np.sqrt(float((a1 // ny - a0 // ny) ** 2
                                                + (a1 % ny - a0 % ny) ** 2))
    out = [path[0]]
    i = 0
    while i < n - 1:
        nxt = i + 1
        for j in range(n - 1, i + 1, -1):
            ok, cs, _, bs = _seg(path[i], path[j], ny, face, f, c, sh, gp)
            if not ok:
                continue
            if lex:
                pi_, pj_ = path[i], path[j]
                d = PIX * np.sqrt(float((pj_ // ny - pi_ // ny) ** 2 + (pj_ % ny - pi_ % ny) ** 2))
                bq, db = _qb(bs), cb[j] - cb[i]
                take = bq < db or (bq == db and d <= (cl[j] - cl[i]) * (1.0 + 1e-12))
            else:
                take = cs <= (cum[j] - cum[i]) * (1.0 + 1e-12)
            if take:
                nxt = j
                break
        out.append(path[nxt])
        i = nxt
    return np.asarray(out, dtype=np.int64)


@njit(cache=True)
def _theta(a: int, nx: int, ny: int, face: I64, c: F64, sh: F64, gp: F64,
           lex: bool) -> tuple[F64, F64, F64, F64, F64, I64]:
    """Theta* from pixel a over its face, weighted (Theta): Dijkstra on the 8-neighbour grid,
    but a pixel s2 reached from s may take s's parent as its own when the straight segment from
    it stays in the face and is cheaper (`_better`: the cost field integrated along it, `_seg`;
    at the limit, lex, its B, each segment's rounded `_qb`, then its length) than through s. Per
    pixel (cost, gain summed along its path, the path's length, its B, its swept area
    (`_sweep`), its parent; -1 unreached, a its own). A parent is settled before its children,
    so each child's gain, length, B and area add its segment's to its parent's: one pass, as
    `_tree_sum` down a grid tree."""
    n = nx * ny
    f = face[a]
    g = np.full(n, np.inf)
    acc = np.zeros(n)
    ln = np.full(n, np.inf)
    blk = np.full(n, np.inf)
    ar = np.zeros(n)
    par = -np.ones(n, dtype=np.int64)
    done = np.zeros(n, dtype=np.bool_)
    g[a] = 0.0
    ln[a] = 0.0
    blk[a] = 0.0
    par[a] = a
    heap = [(0.0, 0.0, a)]
    while len(heap):
        _, _, s = heapq.heappop(heap)
        if done[s]:
            continue
        done[s] = True
        p = par[s]
        si, sj = s // ny, s % ny
        pi, pj = p // ny, p % ny
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue
                i2, j2 = si + di, sj + dj
                if i2 < 0 or i2 >= nx or j2 < 0 or j2 >= ny:
                    continue
                s2 = i2 * ny + j2
                if done[s2] or face[s2] != f:
                    continue
                ok, cs, gn, bs = _seg(s, s2, ny, face, f, c, sh, gp)
                best = g[s] + cs
                bp = s
                bacc = acc[s] + gn
                bln = ln[s] + PIX * np.sqrt(float(di * di + dj * dj))
                bb = blk[s] + (_qb(bs) if lex else bs)
                bar = ar[s] + _sweep(s, s2, ny)
                if p != s:
                    d = PIX * np.sqrt(float((i2 - pi) ** 2 + (j2 - pj) ** 2))
                    # a metre costs at least 1 (lex: B grows by at least 0 and the length by d)
                    if (_better(g[p] + d, blk[p], ln[p] + d, best, bb, bln, lex)
                            and _better(g[p] + d, blk[p], ln[p] + d, g[s2], blk[s2], ln[s2],
                                        lex)):
                        ok, cs, gn, bs = _seg(p, s2, ny, face, f, c, sh, gp)
                        b2 = blk[p] + (_qb(bs) if lex else bs)
                        if ok and _better(g[p] + cs, b2, ln[p] + d, best, bb, bln, lex):
                            best = g[p] + cs
                            bp = p
                            bacc = acc[p] + gn
                            bln = ln[p] + d
                            bb = b2
                            bar = ar[p] + _sweep(p, s2, ny)
                if _better(best, bb, bln, g[s2], blk[s2], ln[s2], lex):
                    g[s2] = best
                    par[s2] = bp
                    acc[s2] = bacc
                    ln[s2] = bln
                    blk[s2] = bb
                    ar[s2] = bar
                    if lex:
                        heapq.heappush(heap, (bb, bln, s2))
                    else:
                        heapq.heappush(heap, (best, 0.0, s2))
    return g, acc, ln, blk, ar, par


@njit(cache=True)
def _grid8_lex(a: int, nx: int, ny: int, face: I64,
               sh: F64) -> tuple[F64, F64, NDArray[np.int32], I64]:
    """Grid8 at the limit (b inf): Dijkstra from pixel a over its face on the 8-neighbour grid
    by (B, length) lexicographically, an edge's B its length times its two pixels' mean share
    (Grid8's at finite b), rounded (`_qb`). Per pixel (B, length, predecessor: -9999 at a and
    unreached), and the settle order."""
    n = nx * ny
    f = face[a]
    B = np.full(n, np.inf)
    L = np.full(n, np.inf)
    pred = np.full(n, -9999, dtype=np.int32)
    done = np.zeros(n, dtype=np.bool_)
    order = np.empty(n, dtype=np.int64)
    k = 0
    B[a] = 0.0
    L[a] = 0.0
    heap = [(0.0, 0.0, a)]
    while len(heap):
        _, _, v = heapq.heappop(heap)
        if done[v]:
            continue
        done[v] = True
        order[k] = v
        k += 1
        vi, vj = v // ny, v % ny
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue
                i2, j2 = vi + di, vj + dj
                if i2 < 0 or i2 >= nx or j2 < 0 or j2 >= ny:
                    continue
                u = i2 * ny + j2
                if done[u] or face[u] != f:
                    continue
                el = PIX * np.sqrt(float(di * di + dj * dj))
                nb = B[v] + _qb(el * 0.5 * (sh[v] + sh[u]))
                nl = L[v] + el
                if nb < B[u] or (nb == B[u] and nl < L[u]):
                    B[u] = nb
                    L[u] = nl
                    pred[u] = v
                    heapq.heappush(heap, (nb, nl, u))
    return B, L, pred, order[:k]


FMM_INIT_PX = 6        # Eikonal: pixels within this many of the source, the straight segment to
                       # which runs on open ground only, start (and stay) at its cost: exact there,
                       # where an upwind stencil smears the point source's kink (an open 161 x 161
                       # raster: T / straight line - 1 in [-0.034, +0.002] without, [-0.004,
                       # +0.002] with)
DESCENT_STEP = 0.5     # Eikonal: the descent's step, in pixels
SIMPLIFY_M = 0.1       # m: Eikonal: the descent's line simplified to this tolerance


@njit(cache=True)
def _stencil(r: int, nx: int, ny: int, T: F64, A: F64, Lg: F64, W: F64,
             known: NDArray[np.bool_], cr: float, gr_: float) -> tuple[float, float, float, float]:
    """The upwind update of pixel r from its known neighbours (Eikonal): per stencil (the axes,
    spacing PIX; the diagonals, spacing PIX sqrt 2) and direction the smaller known neighbour,
    on the axes second order (U = (4 T1 - T2) / 3) where the next one beyond it is known and no
    larger (on the diagonals too, the open raster's T undershoots the straight line by up to
    13%, 1.6% beyond 100 m: the smaller of two second-order stencils is biased low); T
    solves sum_k alpha_k^2 (T - U_k)^2 = cr^2 (one direction alone where the two-direction root
    is not upwind of both); the smaller T of the two stencils. With it the gain and length along
    the characteristic, from the transport equations grad A . grad T = cr gr_ / PIX and grad L .
    grad T = cr discretized on the same stencil: A = (sum_k alpha_k^2 (T - U_k) V_k + cr gr_ /
    PIX) / sum_k alpha_k^2 (T - U_k), V_k the gains extrapolated as U_k is; and the swept area
    the same way, each upwind neighbour's own (W) and its segment to r's centre (`_sweep`). (T,
    A, L, W)."""
    ri, rj = r // ny, r % ny
    bestT, bestA, bestL, bestW = np.inf, 0.0, 0.0, 0.0
    for st in range(2):
        h = PIX if st == 0 else PIX * np.sqrt(2.0)
        al = np.zeros(2)
        U = np.zeros(2)
        V = np.zeros(2)
        W_ = np.zeros(2)
        X = np.zeros(2)
        m = 0
        for k in range(2):
            if st == 0:
                ui, uj = (1, 0) if k == 0 else (0, 1)
            else:
                ui, uj = (1, 1) if k == 0 else (1, -1)
            t1 = np.inf
            q1, q2 = -1, -1
            for sgn in (-1, 1):
                i1, j1 = ri + sgn * ui, rj + sgn * uj
                if 0 <= i1 < nx and 0 <= j1 < ny:
                    q = i1 * ny + j1
                    if known[q] and T[q] < t1:
                        t1 = T[q]
                        q1 = q
                        i2, j2 = i1 + sgn * ui, j1 + sgn * uj
                        q2 = (i2 * ny + j2) if (0 <= i2 < nx and 0 <= j2 < ny) else -1
            if q1 < 0:
                continue
            X[m] = W[q1] + _sweep(q1, r, ny)
            if st == 0 and q2 >= 0 and known[q2] and T[q2] <= T[q1]:
                al[m] = 1.5 / h
                U[m] = (4.0 * T[q1] - T[q2]) / 3.0
                V[m] = (4.0 * A[q1] - A[q2]) / 3.0
                W_[m] = (4.0 * Lg[q1] - Lg[q2]) / 3.0
            else:
                al[m] = 1.0 / h
                U[m] = T[q1]
                V[m] = A[q1]
                W_[m] = Lg[q1]
            m += 1
        if m == 0:
            continue
        T0 = np.inf
        use = 0
        if m == 2:
            a2 = al[0] ** 2 + al[1] ** 2
            b2 = al[0] ** 2 * U[0] + al[1] ** 2 * U[1]
            c2 = al[0] ** 2 * U[0] ** 2 + al[1] ** 2 * U[1] ** 2 - cr * cr
            disc = b2 * b2 - a2 * c2
            if disc >= 0.0:
                t = (b2 + np.sqrt(disc)) / a2
                if t >= U[0] and t >= U[1]:
                    T0, use = t, 2
        if use == 0:
            for k in range(m):
                t = U[k] + cr / al[k]
                if t < T0:
                    T0, use = t, -(k + 1)
        if T0 < bestT:
            if use == 2:
                w0 = al[0] ** 2 * (T0 - U[0])
                w1 = al[1] ** 2 * (T0 - U[1])
                ws = w0 + w1
                bestA = (w0 * V[0] + w1 * V[1] + cr * gr_ / PIX) / ws
                bestL = (w0 * W_[0] + w1 * W_[1] + cr) / ws
                bestW = (w0 * X[0] + w1 * X[1]) / ws
            else:
                k = -use - 1
                bestA = V[k] + gr_ / (al[k] * PIX)
                bestL = W_[k] + 1.0 / al[k]
                bestW = X[k]
            bestT = T0
    return bestT, bestA, bestL, bestW


@njit(cache=True)
def _fmm(a: int, nx: int, ny: int, face: I64, c: F64, gp: F64) -> tuple[F64, F64, F64, F64]:
    """Fast marching from pixel a over its face (Eikonal): arrival times T of |grad T| = c (the
    cost per metre), the gain A, length L and swept area W along each pixel's characteristic,
    each pixel's
    taken from its final upwind stencil once it is known (`_stencil`: the second pass in arrival
    order, folded into the march, since a pixel's upwind neighbours are known before it is).
    Pixels within FMM_INIT_PX of a on open ground start at their straight segment's values
    (`_seg`), which no stencil overwrites."""
    n = nx * ny
    f = face[a]
    T = np.full(n, np.inf)
    A = np.zeros(n)
    Lg = np.zeros(n)
    W = np.zeros(n)
    known = np.zeros(n, dtype=np.bool_)
    init = np.zeros(n, dtype=np.bool_)
    T[a] = 0.0
    init[a] = True
    heap = [(0.0, a)]
    ai, aj = a // ny, a % ny
    R = FMM_INIT_PX
    for i in range(max(0, ai - R), min(nx, ai + R + 1)):
        for j in range(max(0, aj - R), min(ny, aj + R + 1)):
            q = i * ny + j
            if q == a or face[q] != f or (i - ai) ** 2 + (j - aj) ** 2 > R * R:
                continue
            ok, cs, gn, _ = _seg(a, q, ny, face, f, c, c, gp)   # its B not wanted
            d = PIX * np.sqrt(float((i - ai) ** 2 + (j - aj) ** 2))
            if ok and cs <= d * (1.0 + 1e-12):     # open ground all the way: exact (c >= 1)
                T[q] = cs
                A[q] = gn
                Lg[q] = d
                W[q] = _sweep(a, q, ny)
                init[q] = True
                heapq.heappush(heap, (cs, q))
    while len(heap):
        tq, q = heapq.heappop(heap)
        if known[q] or tq > T[q]:
            continue
        if not init[q]:
            _, A[q], Lg[q], W[q] = _stencil(q, nx, ny, T, A, Lg, W, known, c[q], gp[q])
        known[q] = True
        qi, qj = q // ny, q % ny
        for di in range(-1, 2):
            for dj in range(-1, 2):
                i2, j2 = qi + di, qj + dj
                if (di == 0 and dj == 0) or i2 < 0 or i2 >= nx or j2 < 0 or j2 >= ny:
                    continue
                r = i2 * ny + j2
                if known[r] or face[r] != f:
                    continue
                t, _, _, _ = _stencil(r, nx, ny, T, A, Lg, W, known, c[r], gp[r])
                if t < T[r] and not init[r]:
                    T[r] = t
                    heapq.heappush(heap, (t, r))
    return T, A, Lg, W


@njit(cache=True)
def _interp(T: F64, nx: int, ny: int, x: float, y: float) -> tuple[float, float, float]:
    """T bilinear between the pixel centres around (x, y) (pixel units, a centre at k + 0.5):
    (T, dT/dx, dT/dy); inf where a corner is unreached or off the raster."""
    i0 = int(np.floor(x - 0.5))
    j0 = int(np.floor(y - 0.5))
    if i0 < 0 or j0 < 0 or i0 + 1 >= nx or j0 + 1 >= ny:
        return np.inf, 0.0, 0.0
    fx, fy = x - 0.5 - i0, y - 0.5 - j0
    t00, t10 = T[i0 * ny + j0], T[(i0 + 1) * ny + j0]
    t01, t11 = T[i0 * ny + j0 + 1], T[(i0 + 1) * ny + j0 + 1]
    if not (np.isfinite(t00) and np.isfinite(t10) and np.isfinite(t01) and np.isfinite(t11)):
        return np.inf, 0.0, 0.0
    v = (t00 * (1 - fx) * (1 - fy) + t10 * fx * (1 - fy) + t01 * (1 - fx) * fy + t11 * fx * fy)
    gx = (t10 - t00) * (1 - fy) + (t11 - t01) * fy
    gy = (t01 - t00) * (1 - fx) + (t11 - t10) * fx
    return v, gx, gy


@njit(cache=True)
def _descend(T: F64, nx: int, ny: int, a: int, t: int) -> F64:
    """The path from pixel t down T to its source a (Eikonal), in pixel units from t's centre to
    a's: steps of DESCENT_STEP against the bilinear gradient while T keeps falling; where it
    cannot (a corner unreached, a kink), a step to the current pixel's lowest 8-neighbour's centre
    (every reached pixel but a has a lower neighbour: its upwind one)."""
    ax_, ay_ = a // ny + 0.5, a % ny + 0.5
    x, y = t // ny + 0.5, t % ny + 0.5
    xs = [x]
    ys = [y]
    for _ in range(8 * nx * ny):
        if (x - ax_) ** 2 + (y - ay_) ** 2 <= 1.0:
            break
        v, gx, gy = _interp(T, nx, ny, x, y)
        g = np.sqrt(gx * gx + gy * gy)
        moved = False
        if np.isfinite(v) and g > 0.0:
            x2, y2 = x - DESCENT_STEP * gx / g, y - DESCENT_STEP * gy / g
            v2, _, _ = _interp(T, nx, ny, x2, y2)
            if v2 < v:
                x, y = x2, y2
                moved = True
        if not moved:
            i, j = int(np.floor(x)), int(np.floor(y))
            q = i * ny + j
            bq, bt = -1, T[q]
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    i2, j2 = i + di, j + dj
                    if 0 <= i2 < nx and 0 <= j2 < ny and T[i2 * ny + j2] < bt:
                        bq, bt = i2 * ny + j2, T[i2 * ny + j2]
            if bq < 0:          # at a's own pixel, or a flat spot of it
                break
            x, y = bq // ny + 0.5, bq % ny + 0.5
        xs.append(x)
        ys.append(y)
    xs.append(ax_)
    ys.append(ay_)
    out = np.empty((len(xs), 2))
    for k in range(len(xs)):
        out[k, 0] = xs[len(xs) - 1 - k]
        out[k, 1] = ys[len(xs) - 1 - k]
    return out


@dataclass(frozen=True, kw_only=True)
class _TreeReach:
    """A grid search's reach (Grid8): its distances, predecessor tree and settle order."""
    src: int
    cost: F64
    pred: NDArray[np.int32]
    order: I64
    gain: F64
    area: F64
    xy: F64

    def length(self) -> F64:
        p = self.pred[self.order]
        inc = np.zeros(len(self.pred))
        inc[self.order] = np.where(p >= 0, np.hypot(*(self.xy[self.order]
                                                      - self.xy[np.maximum(p, 0)]).T), 0.0)
        return _tree_sum(self.pred, self.order, inc)

    def route(self, t: int) -> Route:
        path = [t]
        while path[-1] != self.src:
            path.append(int(self.pred[path[-1]]))
        p = np.asarray(path[::-1], dtype=np.int64)
        return Route(path=p, line=self.xy[p])


@dataclass(frozen=True, kw_only=True)
class Grid8:
    """Dijkstra on the 8-neighbour grid (the router of every run before 2026-11-19): an edge
    costs its length times 1 + bld times its two pixels' mean share (the cost field integrated
    along it), at binf (B, length) lexicographically, its B its length times that mean
    (`_grid8_lex`, exact as the limit); the gain summed per pixel of the path (`_tree_sum`), the
    path the grid's staircase. Directions come in multiples of 45 degrees, so the cheapest path
    between two points on open ground is up to 8.2% longer than the straight line (at 22.5
    degrees) and every staircase between them ties (b0's right angles: the crossings study)."""

    @property
    def name(self) -> str:
        return "grid8"

    def reaches(self, ter: Terrain, src: I64) -> Iterator[Reach]:
        if ter.lex():
            nx, ny = ter.shape
            for a in src:
                B, _, pred, order = _grid8_lex(int(a), nx, ny, ter.face, ter.share)
                yield _TreeReach(src=int(a), cost=B, pred=pred, order=order,
                                 gain=_tree_sum(pred, order, ter.gain),
                                 area=_tree_area(pred, order, ny), xy=ter.xy)
            return
        ok = (ter.face[ter.ea] >= 0) & (ter.face[ter.ea] == ter.face[ter.eb])
        c = ter.per_m()
        w = ter.el[ok] * 0.5 * (c[ter.ea[ok]] + c[ter.eb[ok]])
        n = len(ter.face)
        ny = ter.shape[1]
        G = sp.coo_matrix((w, (ter.ea[ok], ter.eb[ok])), shape=(n, n)).tocsr()
        for c0 in range(0, len(src), CHUNK):
            s = src[c0:c0 + CHUNK]
            dist, pred = dijkstra(G, directed=False, indices=s, return_predecessors=True)
            for k, a in enumerate(s):
                d = dist[k]
                fin = np.isfinite(d)
                order = np.flatnonzero(fin)[np.argsort(d[fin])]
                yield _TreeReach(src=int(a), cost=d.copy(), pred=pred[k].copy(), order=order,
                                 gain=_tree_sum(pred[k], order, ter.gain),
                                 area=_tree_area(pred[k], order, ny), xy=ter.xy)


@dataclass(frozen=True, kw_only=True)
class _PulledReach:
    inner: Reach
    ter: Terrain

    @property
    def src(self) -> int:
        return self.inner.src

    @property
    def cost(self) -> F64:
        return self.inner.cost

    @property
    def gain(self) -> F64:
        return self.inner.gain

    @property
    def area(self) -> F64:
        return self.inner.area

    def length(self) -> F64:
        return self.inner.length()

    def route(self, t: int) -> Route:
        ter = self.ter
        v = _pull(self.inner.route(t).path, ter.shape[1], ter.face, ter.per_m(), ter.share,
                  ter.gain, ter.lex())
        return Route(path=v, line=ter.xy[v])


@dataclass(frozen=True, kw_only=True)
class Pulled:
    """Grid8's search and choice (its costs, gains and targets, so its octile bias in what is
    chosen), the chosen path string-pulled before its corridor is built (`_pull`): a staircase
    run replaced by the straight segment wherever that stays in the face and costs no more (the
    cost field integrated along it, not its ends averaged; at binf no more B, then no longer). A
    path shaping on Grid8's reach, not
    a search of its own."""
    inner: Grid8

    @property
    def name(self) -> str:
        return "pulled"

    def reaches(self, ter: Terrain, src: I64) -> Iterator[Reach]:
        for r in self.inner.reaches(ter, src):
            yield _PulledReach(inner=r, ter=ter)


@dataclass(frozen=True, kw_only=True)
class _ArrayReach:
    """A search's reach held as per-pixel arrays (Theta, Eikonal), its route drawn by `draw`."""
    src: int
    cost: F64
    gain: F64
    ln: F64
    area: F64
    draw: Callable[[int, int], Route]

    def length(self) -> F64:
        return self.ln

    def route(self, t: int) -> Route:
        return self.draw(self.src, t)


@dataclass(frozen=True, kw_only=True)
class Theta:
    """Any-angle search (`_theta`: Theta*, weighted): a path's vertices any pixels in line of
    sight within the face, a segment costing the cost field integrated along it (`_seg`), its
    gain the gain of the cells it crosses, each by its length in the cell over PIX. Near-optimal,
    not exact, where the cost varies: a vertex's parent is its grid neighbour's or that one's
    parent, never a pixel further back. At binf (B, length) lexicographically, each segment's B
    rounded (`_qb`)."""

    @property
    def name(self) -> str:
        return "theta"

    def reaches(self, ter: Terrain, src: I64) -> Iterator[Reach]:
        nx, ny = ter.shape
        c = ter.per_m()
        lex = ter.lex()
        for a in src:
            g, acc, ln, blk, ar, par = _theta(int(a), nx, ny, ter.face, c, ter.share, ter.gain,
                                              lex)

            def draw(s: int, t: int, par: I64 = par) -> Route:
                path = [t]
                while path[-1] != s:
                    path.append(int(par[path[-1]]))
                p = np.asarray(path[::-1], dtype=np.int64)
                return Route(path=p, line=ter.xy[p])

            yield _ArrayReach(src=int(a), cost=blk if lex else g, gain=acc, ln=ln, area=ar,
                              draw=draw)


@dataclass(frozen=True, kw_only=True)
class Eikonal:
    """Fast marching (`_fmm`): arrival times T of |grad T| = the cost per metre on the raster,
    continuous directions (a second-order upwind stencil on the axes and on the diagonals, the
    smaller), building pixels a weighted region; no tree, so the gain along each pixel's
    characteristic is carried by the same stencil as T in arrival order; the path the descent of
    T from the target to the source (`_descend`), simplified to SIMPLIFY_M. Refused at binf
    (construction, Plan.of): the limit's B is the arrival time of |grad B| = s, which is 0 on
    open ground, so its minimizers there are not one characteristic and the length that breaks
    their ties is a second eikonal on the set of B-optimal paths, which a march over the raster
    does not represent (nor does a big M: |grad T| = 1 + M s at any finite M is a finite b)."""
    bld: float

    def __post_init__(self) -> None:
        if np.isinf(self.bld):
            raise ValueError("eikonal at binf: the limit is not one eikonal (Eikonal)")

    @property
    def name(self) -> str:
        return "eikonal"

    def reaches(self, ter: Terrain, src: I64) -> Iterator[Reach]:
        if ter.bld != self.bld:
            raise ValueError(f"eikonal built for b{self.bld:g} on a b{ter.bld:g} terrain")
        nx, ny = ter.shape
        c = ter.per_m()
        for a in src:
            T, A, Lg, W = _fmm(int(a), nx, ny, ter.face, c, ter.gain)

            def draw(s: int, t: int, T: F64 = T) -> Route:
                pts = _descend(T, nx, ny, s, t) * PIX + ter.lo
                line = np.asarray(shapely.get_coordinates(
                    shapely.simplify(shapely.linestrings(pts), SIMPLIFY_M)))
                line[0], line[-1] = ter.xy[s], ter.xy[t]
                pix = _pix_of_lo(ter.lo, ter.shape, pts)
                keep = np.ones(len(pix), dtype=bool)
                keep[1:] = pix[1:] != pix[:-1]
                return Route(path=pix[keep], line=line)

            yield _ArrayReach(src=int(a), cost=T, gain=A, ln=Lg, area=W, draw=draw)


# ---------------------------------------------------------------- vector routers (2026-11-22)
#
# The routers above search the cost on the PIX raster. The one below (Visibility) searches it on
# the state's vector geometry (the wiki's
# reblock/shock-reblock-routing-is-a-weighted-region-problem): a face the polygon of its mesh
# triangles (Shapes), at b0 alone (its free space, `free_of`: the face). Before rev 4 it also ran
# at binf, the footprints holes in the face; since b charges for the share of the road's
# cross-section (`road_share`) and binf is that charge's limit, the cost at binf is not one per
# metre anywhere near a building, and Visibility is refused there. A goal pixel's centre is its
# point where the free space covers it, else the nearest
# point of the free space (`attach`), joined to its centre by a straight stub costing its length.
# Steiner<k> (a (1 + eps) Steiner-point graph on the free space cut into weighted pieces, at any
# b) was deleted as dominated: the wiki's reblock/steiner-router-is-dominated, commit 1a92867c
# for its code and numbers.
# The router builds a Net per face (once per state, on its first source there): nodes, a graph
# among them, and per goal pixel the nodes its point joins by a straight piece in the free space;
# a source's search is one Dijkstra from its own pixel's attachments (`_net_solve`), its gain the
# per-pixel gain density integrated along each piece (`_gain_xy`, as `_seg` integrates it: only
# the source's face's pixels count), so it compares with the raster routers'.

FACE_SIMPLIFY_M = 0.1    # m: a face's polygon (its mesh triangles' union, edges 0.1 -- 1 m)
                         # simplified to this (5618: 15,913 vertices to 107)
GRID_M = 1e-3            # m: the overlay's snap grid
NUDGE_M = 1e-4           # m: a goal pixel's centre off the free space: its nearest point this
                         # further in
CELL_M = PIX             # m: Visibility's grid of the free space's edges
TAN_TOL = 1e-9           # Visibility: a turn (a sine) within this is straight: no corner there,
                         # and a corner's neighbour this near a line through it is on the line
                         # (tangent). One threshold for both: a vertex neither a corner nor on a
                         # line's side loses the path along its wall (5618 binf, an exact sign
                         # test against a 1e-9 corner test: paths 4.5 % too dear); a
                         # looser one (1e-3) passes a line into a building along a snapped wall
                         # (8.8 m through one)
ON_M = 1e-6              # m: Visibility: a boundary vertex this near a segment is on it
WEDGE_TOL = 1e-6         # Visibility: a direction this far (a sine) outside a vertex's free
                         # wedge is in it (along a wall)


class Shapes(Protocol):
    """A state's vector geometry: a face's polygon."""
    def face(self, f: int) -> Any: ...


@dataclass(frozen=True, kw_only=True)
class MeshShapes:
    """A state's faces from the growth mesh (`lab` per triangle its face, -1 grown): a face the
    union of its triangles (their edges no other of its triangles shares, areas built)."""
    xy: F64
    tri: I64
    lab: I64

    def face(self, f: int) -> Any:
        t = self.tri[self.lab == f]
        e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
        n = len(self.xy)
        u, c = np.unique(e[:, 0] * n + e[:, 1], return_counts=True)
        b = u[c == 1]
        lines = shapely.linestrings(np.stack([self.xy[b // n], self.xy[b % n]], axis=1))
        return shapely.build_area(shapely.multilinestrings(lines))


@dataclass(frozen=True, kw_only=True)
class FixedShapes:
    """Faces given outright (the router checks' synthetic terrains)."""
    faces: dict[int, Any]

    def face(self, f: int) -> Any:
        return self.faces[f]


def _polygons(g: Any) -> NDArray[np.object_]:
    """g's polygons of positive area (an overlay snapped to a grid can return a collection with
    collapsed lines and points beside them, which the next overlay refuses: mixed-dimension)."""
    p = shapely.get_parts(shapely.get_parts(g))          # a collection's multipolygons too
    return p[(shapely.get_type_id(p) == 3) & (shapely.area(p) > 0)]


def _polygonal(g: Any) -> Any:
    return shapely.multipolygons(_polygons(g))


def free_of(vec: Shapes, f: int) -> Any:
    """Face f's free space at b0 (Visibility's: a metre costs 1 wherever a path may go): its
    polygon, simplified to FACE_SIMPLIFY_M and snapped to GRID_M."""
    return _polygonal(shapely.set_precision(
        _polygonal(shapely.simplify(vec.face(f), FACE_SIMPLIFY_M)), GRID_M))


def attach(domain: Any, xy: F64) -> tuple[F64, B]:
    """Per point (a goal pixel's centre) its point in `domain`: itself where the domain covers
    it, else its nearest point of the domain NUDGE_M further in (along the way from it); and
    whether that one is covered (a point no nudge brings in is unreachable)."""
    pts = shapely.points(xy)
    shapely.prepare(domain)
    inside = shapely.covers(domain, pts)
    out = xy.copy()
    if (~inside).any():
        q = shapely.get_coordinates(shapely.shortest_line(pts[~inside], domain)).reshape(-1, 2, 2)
        d = q[:, 1] - q[:, 0]
        n = np.maximum(np.hypot(*d.T), 1e-300)
        out[~inside] = q[:, 1] + d / n[:, None] * NUDGE_M
    return out, np.asarray(inside | shapely.covers(domain, shapely.points(out)))


@njit(cache=True)
def _gain_xy(x0: float, y0: float, x1: float, y1: float, lx: float, ly: float, nx: int, ny: int,
             gp: F64) -> float:
    """The per-pixel field gp integrated along the segment (x0, y0) -- (x1, y1) (m) per PIX of
    its length: each raster cell it crosses (lower corner lx, ly; nx x ny) by the length in it
    (`_seg`'s gain between any two points; a cell off the raster counts 0)."""
    dx, dy = x1 - x0, y1 - y0
    L = np.sqrt(dx * dx + dy * dy)
    if L == 0.0:
        return 0.0
    px, py = (x0 - lx) / PIX, (y0 - ly) / PIX
    ddx, ddy = dx / PIX, dy / PIX
    i, j = int(np.floor(px)), int(np.floor(py))
    sx = 1 if ddx > 0 else -1
    sy = 1 if ddy > 0 else -1
    tdx = abs(1.0 / ddx) if ddx != 0.0 else np.inf
    tdy = abs(1.0 / ddy) if ddy != 0.0 else np.inf
    tmx = ((i + (1 if sx > 0 else 0)) - px) / ddx if ddx != 0.0 else np.inf
    tmy = ((j + (1 if sy > 0 else 0)) - py) / ddy if ddy != 0.0 else np.inf
    t0 = 0.0
    acc = 0.0
    while True:
        t1 = min(tmx, tmy, 1.0)
        if 0 <= i < nx and 0 <= j < ny:
            acc += gp[i * ny + j] * (t1 - t0)
        if t1 >= 1.0:
            break
        if tmx <= tmy:
            i += sx
            tmx += tdx
        else:
            j += sy
            tmy += tdy
        t0 = t1
    return acc * L / PIX


@njit(cache=True)
def _net_solve(indptr: I64, idx: I64, cost: F64, length: F64, s_nodes: I64, s_cost: F64,
               s_len: F64) -> tuple[F64, F64, I64, I64]:
    """Dijkstra on a Net's graph (CSR) from a point joined to `s_nodes` by straight pieces of
    cost s_cost and length s_len: per node (cost, length, predecessor: -1 the point itself, -2
    unreached), and the settle order."""
    n = len(indptr) - 1
    dist = np.full(n, np.inf)
    ln = np.zeros(n)
    pred = np.full(n, -2, dtype=np.int64)
    done = np.zeros(n, dtype=np.bool_)
    order = np.empty(n, dtype=np.int64)
    m = 0
    if len(s_nodes) == 0:
        return dist, ln, pred, order[:0]
    heap = [(np.inf, np.int64(-1))]
    heap.pop()
    for k in range(len(s_nodes)):
        v = s_nodes[k]
        if s_cost[k] < dist[v]:
            dist[v] = s_cost[k]
            ln[v] = s_len[k]
            pred[v] = -1
            heapq.heappush(heap, (s_cost[k], v))
    while len(heap):
        d, u = heapq.heappop(heap)
        if done[u] or d > dist[u]:
            continue
        done[u] = True
        order[m] = u
        m += 1
        for e in range(indptr[u], indptr[u + 1]):
            v = idx[e]
            nd = d + cost[e]
            if nd < dist[v]:
                dist[v] = nd
                ln[v] = ln[u] + length[e]
                pred[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, ln, pred, order[:m]


@njit(cache=True)
def _net_gain(order: I64, pred: I64, xy: F64, sx: float, sy: float, lx: float, ly: float,
              nx: int, ny: int, gp: F64) -> F64:
    """Per node the gain along its shortest-path tree's path from the point (sx, sy)."""
    acc = np.zeros(len(xy))
    for k in range(len(order)):
        v = order[k]
        p = pred[v]
        if p == -1:
            acc[v] = _gain_xy(sx, sy, xy[v, 0], xy[v, 1], lx, ly, nx, ny, gp)
        else:
            acc[v] = acc[p] + _gain_xy(xy[p, 0], xy[p, 1], xy[v, 0], xy[v, 1], lx, ly, nx, ny, gp)
    return acc


@njit(cache=True)
def _net_area(order: I64, pred: I64, xy: F64, sx: float, sy: float, lx: float,
              ly: float) -> F64:
    """Per node the signed area (`_tri`, about (lx, ly)) its shortest-path tree's path from the
    point (sx, sy) sweeps."""
    out = np.zeros(len(xy))
    for k in range(len(order)):
        v = order[k]
        p = pred[v]
        x0, y0 = (sx, sy) if p == -1 else (xy[p, 0], xy[p, 1])
        out[v] = (0.0 if p == -1 else out[p]) + _tri(x0 - lx, y0 - ly, xy[v, 0] - lx,
                                                     xy[v, 1] - ly)
    return out


def _tris(a: F64, b: F64, lo: F64) -> F64:
    """`_tri` of each segment a[k] -- b[k] about lo."""
    return np.asarray(0.5 * ((a[:, 0] - lo[0]) * (b[:, 1] - lo[1])
                             - (b[:, 0] - lo[0]) * (a[:, 1] - lo[1])))


@njit(cache=True)
def _net_targets(dist: F64, ln: F64, acc: F64, xy: F64, aptr: I64, aidx: I64, acost: F64,
                 alen: F64, dc: F64, dl: F64, pt: F64, sx: float, sy: float, lx: float,
                 ly: float, nx: int, ny: int, gp: F64) -> tuple[F64, F64, F64, I64]:
    """Per goal point its cheapest path from the point (sx, sy): the straight piece (dc, dl; inf
    where none) or a node's path and its attachment; (cost, length, gain, the node it leaves the
    graph at: -1 the straight piece)."""
    T = len(aptr) - 1
    best = np.full(T, np.inf)
    bl = np.zeros(T)
    bg = np.zeros(T)
    via = np.full(T, -1, dtype=np.int64)
    for r in range(T):
        best[r] = dc[r]
        bl[r] = dl[r]
        for k in range(aptr[r], aptr[r + 1]):
            c = dist[aidx[k]] + acost[k]
            if c < best[r]:
                best[r] = c
                bl[r] = ln[aidx[k]] + alen[k]
                via[r] = aidx[k]
        if not np.isfinite(best[r]):
            continue
        if via[r] < 0:
            bg[r] = _gain_xy(sx, sy, pt[r, 0], pt[r, 1], lx, ly, nx, ny, gp)
        else:
            v = via[r]
            bg[r] = acc[v] + _gain_xy(xy[v, 0], xy[v, 1], pt[r, 0], pt[r, 1], lx, ly, nx, ny, gp)
    return best, bl, bg, via


@njit(cache=True)
def _stub_gain(a: F64, b: F64, lx: float, ly: float, nx: int, ny: int, gp: F64) -> F64:
    out = np.zeros(len(a))
    for k in range(len(a)):
        out[k] = _gain_xy(a[k, 0], a[k, 1], b[k, 0], b[k, 1], lx, ly, nx, ny, gp)
    return out


@dataclass(frozen=True, kw_only=True)
class Net:
    """A face's search graph (Visibility's): its nodes' points, the graph (CSR: indptr,
    idx, each edge's cost and length), the face's goal pixels (sorted) with their points in the
    free space (`attach`; `ok` whether attached) and stubs (pixel centre to point: length, gain,
    swept area about the raster's corner),
    per goal pixel the nodes its point joins (CSR: aptr, aidx, acost, alen), `direct` per goal
    row (a source's) the straight piece to every goal row (cost, length; inf where none), and the
    gain density counted (the face's pixels' only)."""
    xy: F64
    indptr: I64
    idx: I64
    cost: F64
    length: F64
    pix: I64
    pt: F64
    ok: B
    stub: F64
    stub_gain: F64
    stub_area: F64
    aptr: I64
    aidx: I64
    acost: F64
    alen: F64
    direct: Callable[[int], tuple[F64, F64]]
    gp: F64

    def reach(self, ter: Terrain, a: int) -> Reach:
        k = int(np.searchsorted(self.pix, a))
        n = len(ter.face)
        cost, gain, ln, area = np.full(n, np.inf), np.zeros(n), np.zeros(n), np.zeros(n)
        if k == len(self.pix) or self.pix[k] != a or not self.ok[k]:
            return _NetReach(src=a, cost=cost, gain=gain, ln=ln, area=area, net=self, row=-1,
                             ter=ter,
                             pred=np.zeros(0, dtype=np.int64), via=np.zeros(0, dtype=np.int64))
        s = self.pt[k]
        a0, a1 = self.aptr[k], self.aptr[k + 1]
        dist, dln, pred, order = _net_solve(self.indptr, self.idx, self.cost, self.length,
                                            self.aidx[a0:a1], self.acost[a0:a1],
                                            self.alen[a0:a1])
        nx, ny = ter.shape
        acc = _net_gain(order, pred, self.xy, s[0], s[1], ter.lo[0], ter.lo[1], nx, ny, self.gp)
        na = _net_area(order, pred, self.xy, s[0], s[1], ter.lo[0], ter.lo[1])
        dc, dl = self.direct(k)
        best, bl, bg, via = _net_targets(dist, dln, acc, self.xy, self.aptr, self.aidx,
                                         self.acost, self.alen, dc, dl, self.pt, s[0], s[1],
                                         ter.lo[0], ter.lo[1], nx, ny, self.gp)
        node = via >= 0
        frm = np.repeat(s[None], len(via), axis=0)
        frm[node] = self.xy[via[node]]
        ba = _tris(frm, self.pt, ter.lo)
        ba[node] += na[via[node]]
        fin = np.isfinite(best) & self.ok
        stubs = self.stub[k] + self.stub
        cost[self.pix[fin]] = best[fin] + stubs[fin]
        ln[self.pix[fin]] = bl[fin] + stubs[fin]
        gain[self.pix[fin]] = bg[fin] + self.stub_gain[k] + self.stub_gain[fin]
        area[self.pix[fin]] = ba[fin] + self.stub_area[k] - self.stub_area[fin]
        return _NetReach(src=a, cost=cost, gain=gain, ln=ln, area=area, net=self, row=k, ter=ter,
                         pred=pred, via=via)


@dataclass(frozen=True, kw_only=True)
class _NetReach:
    """A vector router's reach: per pixel arrays (inf off the face's goal pixels), and what
    draws a path: the search's predecessors and per goal row the node it leaves the graph at."""
    src: int
    cost: F64
    gain: F64
    ln: F64
    area: F64
    net: Net
    row: int
    ter: Terrain
    pred: I64
    via: I64

    def length(self) -> F64:
        return self.ln

    def route(self, t: int) -> Route:
        net, ter = self.net, self.ter
        r = int(np.searchsorted(net.pix, t))
        nodes = []
        v = int(self.via[r])
        while v >= 0:
            nodes.append(v)
            v = int(self.pred[v])
        line = np.vstack([ter.xy[self.src][None], net.pt[self.row][None], net.xy[nodes[::-1]],
                          net.pt[r][None], ter.xy[t][None]])
        keep = np.ones(len(line), dtype=bool)
        keep[1:] = (np.diff(line, axis=0) != 0).any(axis=1)
        line = line[keep]
        pix = np.concatenate([[self.src], _pix_of_lo(ter.lo, ter.shape, line[1:-1]), [t]])
        k = np.ones(len(pix), dtype=bool)
        k[1:] = pix[1:] != pix[:-1]
        return Route(path=np.asarray(pix[k], dtype=np.int64), line=line)


def _net_reaches(ter: Terrain, src: I64, build: Callable[[Terrain, int], Net]) -> Iterator[Reach]:
    """Per source its Net's reach, each face's Net built on its first source there."""
    nets: dict[int, Net] = {}
    for a in src:
        f = int(ter.face[a])
        if f not in nets:
            nets[f] = build(ter, f)
        yield nets[f].reach(ter, int(a))


class Goals(NamedTuple):
    """A face's goal pixels (sorted), their points in the domain, whether attached, their stubs'
    lengths, gains and swept areas (pixel centre to point), and the gain density counted (the
    face's pixels' only)."""
    pix: I64
    pt: F64
    ok: B
    stub: F64
    stub_gain: F64
    stub_area: F64
    gp: F64


def _goal_points(ter: Terrain, f: int, domain: Any) -> Goals:
    """Face f's goal pixels and their points in the domain (Goals)."""
    pix = np.flatnonzero(ter.goal & (ter.face == f))
    pt, ok = attach(domain, ter.xy[pix])
    gp = np.where(ter.face == f, ter.gain, 0.0)
    stub = np.hypot(*(pt - ter.xy[pix]).T)
    nx, ny = ter.shape
    sg = _stub_gain(ter.xy[pix], pt, ter.lo[0], ter.lo[1], nx, ny, gp)
    return Goals(pix=pix, pt=pt, ok=ok, stub=stub, stub_gain=sg,
                 stub_area=_tris(ter.xy[pix], pt, ter.lo), gp=gp)


def _csr(n: int, a: I64, b: I64, *vals: F64) -> tuple[I64, I64, tuple[F64, ...]]:
    """An undirected edge list (a, b) as CSR on n nodes (both ways), its values carried."""
    u = np.concatenate([a, b])
    v = np.concatenate([b, a])
    o = np.argsort(u, kind="stable")
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, u + 1, 1)
    return (np.cumsum(indptr), np.asarray(v[o], dtype=np.int64),
            tuple(np.concatenate([x, x])[o] for x in vals))


# -- Visibility


class Rings(NamedTuple):
    """A domain's boundary (Visibility): its rings' vertices (interior on the left), each one's
    ring neighbours, its edges (vertex pairs) and the corners (reflex vertices: where a shortest
    path may bend), and the edges' grid (CELL_M cells from g0, gn cells each way; CSR per cell)."""
    xy: F64
    prv: I64
    nxt: I64
    ev: I64
    corner: I64
    g0: F64
    gn: tuple[int, int]
    gptr: I64
    gidx: I64


@njit(cache=True)
def _register(ev: I64, xy: F64, g0x: float, g0y: float, gnx: int, gny: int) -> tuple[I64, I64]:
    """Per edge the grid cells its bounding box covers: (cell, edge) pairs."""
    cs, es = [], []
    for e in range(len(ev)):
        a, b = ev[e, 0], ev[e, 1]
        i0 = int(np.floor((min(xy[a, 0], xy[b, 0]) - g0x) / CELL_M))
        i1 = int(np.floor((max(xy[a, 0], xy[b, 0]) - g0x) / CELL_M))
        j0 = int(np.floor((min(xy[a, 1], xy[b, 1]) - g0y) / CELL_M))
        j1 = int(np.floor((max(xy[a, 1], xy[b, 1]) - g0y) / CELL_M))
        for i in range(max(i0, 0), min(i1, gnx - 1) + 1):
            for j in range(max(j0, 0), min(j1, gny - 1) + 1):
                cs.append(i * gny + j)
                es.append(e)
    return np.asarray(cs, dtype=np.int64), np.asarray(es, dtype=np.int64)


def rings_of(domain: Any) -> Rings:
    """`domain`'s Rings."""
    xs, prv, nxt, ev = [], [], [], []
    n = 0
    for p in shapely.get_parts(shapely.orient_polygons(domain, exterior_cw=False)):
        for ring in [p.exterior, *p.interiors]:
            c = np.asarray(ring.coords)[:-1]
            c = c[(c != np.roll(c, 1, axis=0)).any(axis=1)]      # no zero-length edge
            m = len(c)
            if m < 3:
                continue
            k = np.arange(m)
            xs.append(c)
            prv.append(n + (k - 1) % m)
            nxt.append(n + (k + 1) % m)
            ev.append(np.column_stack([n + k, n + (k + 1) % m]))
            n += m
    xy = np.vstack(xs)
    pv, nv = np.concatenate(prv), np.concatenate(nxt)
    e = np.vstack(ev)
    a, b = xy - xy[pv], xy[nv] - xy
    turn = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    scale = np.hypot(*a.T) * np.hypot(*b.T)
    corner = np.flatnonzero(turn < -TAN_TOL * scale)       # a right turn: reflex
    g0 = xy.min(axis=0) - CELL_M
    gn = tuple(int(x) for x in np.ceil((xy.max(axis=0) - g0) / CELL_M) + 2)
    cs, es = _register(e, xy, g0[0], g0[1], gn[0], gn[1])
    o = np.argsort(cs, kind="stable")
    gptr = np.zeros(gn[0] * gn[1] + 1, dtype=np.int64)
    np.add.at(gptr, cs + 1, 1)
    return Rings(xy=xy, prv=pv, nxt=nv, ev=e, corner=corner, g0=g0, gn=(gn[0], gn[1]),
                 gptr=np.cumsum(gptr), gidx=es[o])


@njit(cache=True)
def _cr(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


@njit(cache=True)
def _tangent(x: float, y: float, c: int, xy: F64, prv: I64, nxt: I64) -> bool:
    """Whether the line from (x, y) through corner c leaves both of c's ring neighbours on one
    side (a shortest path bends at a corner only so), a neighbour within TAN_TOL (the sine of
    its angle off the line) on it: a wall the line runs along."""
    a = np.hypot(x - xy[c, 0], y - xy[c, 1])
    if a == 0.0:                    # (x, y) at c: no line (a pinch's two corners: no link)
        return False
    s1 =_cr(x, y, xy[c, 0], xy[c, 1], xy[prv[c], 0], xy[prv[c], 1]) / (
        a * np.hypot(xy[prv[c], 0] - xy[c, 0], xy[prv[c], 1] - xy[c, 1]))
    s2 = _cr(x, y, xy[c, 0], xy[c, 1], xy[nxt[c], 0], xy[nxt[c], 1]) / (
        a * np.hypot(xy[nxt[c], 0] - xy[c, 0], xy[nxt[c], 1] - xy[c, 1]))
    return abs(s1) <= TAN_TOL or abs(s2) <= TAN_TOL or s1 * s2 > 0.0


@njit(cache=True)
def _in_wedge(w: int, dx: float, dy: float, xy: F64, prv: I64, nxt: I64) -> bool:
    """Whether direction (dx, dy) from vertex w lies in its free wedge (the free space on each
    edge's left: counter-clockwise from the next vertex's direction to the previous one's), to
    WEDGE_TOL (a sine)."""
    ax, ay = xy[nxt[w], 0] - xy[w, 0], xy[nxt[w], 1] - xy[w, 1]
    cx, cy = xy[prv[w], 0] - xy[w, 0], xy[prv[w], 1] - xy[w, 1]
    la, lc, ld = np.hypot(ax, ay), np.hypot(cx, cy), np.hypot(dx, dy)
    ad = (ax * dy - ay * dx) / (la * ld)          # sine from a to d
    dc = (dx * cy - dy * cx) / (ld * lc)          # sine from d to c
    if ax * cy - ay * cx >= 0.0:                  # a convex wedge: d between a and c
        return ad >= -WEDGE_TOL and dc >= -WEDGE_TOL
    return not (ad < -WEDGE_TOL and dc < -WEDGE_TOL)   # reflex: d not in the convex rest


@njit(cache=True)
def _sees(x0: float, y0: float, x1: float, y1: float, u: int, v: int, xy: F64, ev: I64,
          prv: I64, nxt: I64, g0x: float, g0y: float, gnx: int, gny: int, gptr: I64,
          gidx: I64, stamp: I64, tick: int) -> bool:
    """Whether the segment (x0, y0) -- (x1, y1) properly crosses no boundary edge (those at
    vertex u or v skipped, -1 none) and, through a boundary vertex on it (other than u, v: an
    edge's first, within ON_M), leaves that vertex's free wedge neither way (`_in_wedge`; a
    proper-crossing test misses a segment that enters an obstacle at a vertex and leaves at
    another, its edges there collinear with it): the grid cells it passes through, each edge
    once (stamp)."""
    L = np.hypot(x1 - x0, y1 - y0)
    if L == 0.0:
        return True
    dx, dy = (x1 - x0) / CELL_M, (y1 - y0) / CELL_M
    px, py = (x0 - g0x) / CELL_M, (y0 - g0y) / CELL_M
    i, j = int(np.floor(px)), int(np.floor(py))
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    tdx = abs(1.0 / dx) if dx != 0.0 else np.inf
    tdy = abs(1.0 / dy) if dy != 0.0 else np.inf
    tmx = ((i + (1 if sx > 0 else 0)) - px) / dx if dx != 0.0 else np.inf
    tmy = ((j + (1 if sy > 0 else 0)) - py) / dy if dy != 0.0 else np.inf
    while True:
        if 0 <= i < gnx and 0 <= j < gny:
            c = i * gny + j
            for k in range(gptr[c], gptr[c + 1]):
                e = gidx[k]
                if stamp[e] == tick:
                    continue
                stamp[e] = tick
                a, b = ev[e, 0], ev[e, 1]
                d1 = _cr(x0, y0, x1, y1, xy[a, 0], xy[a, 1])
                if a != u and a != v and abs(d1) <= ON_M * L:
                    s = ((xy[a, 0] - x0) * (x1 - x0) + (xy[a, 1] - y0) * (y1 - y0)) / (L * L)
                    if 0.0 < s < 1.0 and not (_in_wedge(a, x1 - x0, y1 - y0, xy, prv, nxt)
                                              and _in_wedge(a, x0 - x1, y0 - y1, xy, prv, nxt)):
                        return False
                if a == u or a == v or b == u or b == v:
                    continue
                d2 = _cr(x0, y0, x1, y1, xy[b, 0], xy[b, 1])
                if d1 * d2 >= 0.0:
                    continue
                d3 = _cr(xy[a, 0], xy[a, 1], xy[b, 0], xy[b, 1], x0, y0)
                d4 = _cr(xy[a, 0], xy[a, 1], xy[b, 0], xy[b, 1], x1, y1)
                if d3 * d4 < 0.0:
                    return False
        if min(tmx, tmy) >= 1.0:
            break
        if tmx <= tmy:
            i += sx
            tmx += tdx
        else:
            j += sy
            tmy += tdy
    return True


@njit(cache=True)
def _vis_graph(corner: I64, xy: F64, prv: I64, nxt: I64, ev: I64, g0x: float, g0y: float,
               gnx: int, gny: int, gptr: I64, gidx: I64) -> tuple[I64, I64, F64]:
    """The corners' reduced visibility graph: pairs (as corner rows) tangent at both ends whose
    segment crosses no edge, and their lengths."""
    stamp = np.zeros(len(ev), dtype=np.int64)
    tick = 0
    ia, ib, il = [], [], []
    K = len(corner)
    for p in range(K):
        a = corner[p]
        for q in range(p + 1, K):
            b = corner[q]
            if not (_tangent(xy[a, 0], xy[a, 1], b, xy, prv, nxt)
                    and _tangent(xy[b, 0], xy[b, 1], a, xy, prv, nxt)):
                continue
            tick += 1
            if _sees(xy[a, 0], xy[a, 1], xy[b, 0], xy[b, 1], a, b, xy, ev, prv, nxt, g0x, g0y,
                     gnx, gny, gptr, gidx, stamp, tick):
                ia.append(p)
                ib.append(q)
                il.append(np.hypot(xy[a, 0] - xy[b, 0], xy[a, 1] - xy[b, 1]))
    return (np.asarray(ia, dtype=np.int64), np.asarray(ib, dtype=np.int64),
            np.asarray(il, dtype=np.float64))


@njit(cache=True)
def _vis_attach(pt: F64, ok: NDArray[np.bool_], corner: I64, xy: F64, prv: I64, nxt: I64,
                ev: I64, g0x: float, g0y: float, gnx: int, gny: int, gptr: I64,
                gidx: I64) -> tuple[I64, I64, F64]:
    """Per point (CSR) the corners (rows) it sees with the line tangent at the corner."""
    stamp = np.zeros(len(ev), dtype=np.int64)
    tick = 0
    T = len(pt)
    ptr = np.zeros(T + 1, dtype=np.int64)
    ix, il = [], []
    for r in range(T):
        if ok[r]:
            x, y = pt[r, 0], pt[r, 1]
            for p in range(len(corner)):
                c = corner[p]
                if not _tangent(x, y, c, xy, prv, nxt):
                    continue
                tick += 1
                if _sees(x, y, xy[c, 0], xy[c, 1], -1, c, xy, ev, prv, nxt, g0x, g0y, gnx, gny,
                         gptr, gidx, stamp, tick):
                    ix.append(p)
                    il.append(np.hypot(x - xy[c, 0], y - xy[c, 1]))
        ptr[r + 1] = len(ix)
    return ptr, np.asarray(ix, dtype=np.int64), np.asarray(il, dtype=np.float64)


@njit(cache=True)
def _vis_direct(k: int, pt: F64, ok: NDArray[np.bool_], xy: F64, prv: I64, nxt: I64, ev: I64,
                g0x: float, g0y: float, gnx: int, gny: int, gptr: I64, gidx: I64) -> F64:
    """Per point the length of the straight segment from point k to it (inf where it crosses
    the boundary)."""
    stamp = np.zeros(len(ev), dtype=np.int64)
    out = np.full(len(pt), np.inf)
    x, y = pt[k, 0], pt[k, 1]
    for r in range(len(pt)):
        if ok[r] and _sees(x, y, pt[r, 0], pt[r, 1], -1, -1, xy, ev, prv, nxt, g0x, g0y, gnx,
                           gny, gptr, gidx, stamp, r + 1):
            out[r] = np.hypot(x - pt[r, 0], y - pt[r, 1])
    return out


@dataclass(frozen=True, kw_only=True)
class Visibility:
    """Exact where a metre costs one thing everywhere a path may go: b0 (buildings free, the
    face's own walls only). A shortest path in a polygonal
    domain bends only at its reflex corners, tangent there (the wiki's
    reblock/shock-reblock-routing-is-a-weighted-region-problem), so Dijkstra on the reduced
    visibility graph of the free space's corners (`_vis_graph`), each goal point joined to the
    corners it sees tangentially (`_vis_attach`) and the source to the goal points it sees
    (`_vis_direct`), is exact on the vector face (Shapes, `free_of`); a segment's visibility by
    the boundary edges it could cross (a grid of CELL_M cells). Any other b is refused at
    construction (Plan.of): no finite candidate set is exact there, binf included (its first
    key, B, varies continuously with the distance to the footprints)."""
    bld: float

    def __post_init__(self) -> None:
        if self.bld != 0.0:
            raise ValueError(f"visibility at b{self.bld:g}: exact at b0 only")

    @property
    def name(self) -> str:
        return "visibility"

    def reaches(self, ter: Terrain, src: I64) -> Iterator[Reach]:
        if ter.bld != self.bld:
            raise ValueError(f"visibility built for b{self.bld:g} on a b{ter.bld:g} terrain")
        yield from _net_reaches(ter, src, self.net)

    def net(self, ter: Terrain, f: int) -> Net:
        dom = free_of(ter.vec, f)
        rg = rings_of(dom)
        gl = _goal_points(ter, f, dom)
        pt, ok = gl.pt, gl.ok
        args = (rg.ev, rg.g0[0], rg.g0[1], rg.gn[0], rg.gn[1], rg.gptr, rg.gidx)
        ia, ib, il = _vis_graph(rg.corner, rg.xy, rg.prv, rg.nxt, *args)
        indptr, idx, (cost, length) = _csr(len(rg.corner), ia, ib, il, il)
        aptr, aidx, alen = _vis_attach(pt, ok, rg.corner, rg.xy, rg.prv, rg.nxt, *args)

        def direct(k: int) -> tuple[F64, F64]:
            d = _vis_direct(k, pt, ok, rg.xy, rg.prv, rg.nxt, *args)
            return d, d

        return Net(xy=rg.xy[rg.corner], indptr=indptr, idx=idx, cost=cost, length=length,
                   pix=gl.pix, pt=pt, ok=ok, stub=gl.stub, stub_gain=gl.stub_gain,
                   stub_area=gl.stub_area, aptr=aptr, aidx=aidx, acost=alen, alen=alen,
                   direct=direct, gp=gl.gp)


ROUTER_TOKENS = "grid8|pulled|theta|eikonal|visibility"


def router_of(tok: str, *, bld: float) -> Router:
    """The plan token's router (Plan.of), at the plan's building cost (Visibility refuses any
    but b0, Eikonal binf)."""
    match tok:
        case "grid8":
            return Grid8()
        case "pulled":
            return Pulled(inner=Grid8())
        case "theta":
            return Theta()
        case "eikonal":
            return Eikonal(bld=bld)
        case "visibility":
            return Visibility(bld=bld)
    raise SystemExit(f"{tok}: a router is {ROUTER_TOKENS}")


class Candidates(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def pairs(self) -> int: ...                # candidates costed a step

    def ears(self, e: EarWorld, faces: Face, gain: F64, *, cost: gr.Cost,
             scale: float) -> list[Ear]:
        """The step's costed candidates; `scale` puts an ear's credit (in P) on its first-order
        gain's scale wherever the first order stands in for the exact gain (build)."""
        ...


class Ends(Protocol):
    """Where a path may end (Paths): per pixel of the source's face whether it is a target."""
    @property
    def name(self) -> str: ...

    def targets(self, sink: B, ground: B, dist: F64, *, ear_m: float) -> B: ...

    def goal(self, sink: B, ground: B) -> B:
        """Per pixel whether some source's path may end there (`targets` at some distance: the
        vector routers' goal points, Terrain.goal)."""
        ...


@dataclass(frozen=True, kw_only=True)
class Loop:
    """Every road on a loop: a path ends at a sink pixel (`sink`) at least ear_m from its source
    in a straight line, and an ear that splits nothing is refused (build)."""

    @property
    def name(self) -> str:
        return "loop"

    def targets(self, sink: B, ground: B, dist: F64, *, ear_m: float) -> B:
        return np.asarray(sink & (dist >= ear_m))

    def goal(self, sink: B, ground: B) -> B:
        return sink


@dataclass(frozen=True, kw_only=True)
class Spur:
    """Dead ends allowed (Idea A, owner 2026-11-03): Loop's targets, and also the face's open
    ground (`ground`: an ungrown pixel off every building and off every sink) between min_m and
    max_m from the source in a straight line, where the path is a spur (Ear.spur): its corridor
    joins the network at its start only and it is built whether or not it splits its face. The
    loop ears keep ear_m, the spurs min_m (a spur ends nowhere near the source's road by
    construction, so the two floors guard different things: an ear's ends not crowding one road,
    a spur not a stub)."""
    min_m: float
    max_m: float

    def __post_init__(self) -> None:
        if not 0 < self.min_m < self.max_m:
            raise ValueError(f"{self}: 0 < min_m < max_m")

    @property
    def name(self) -> str:
        return f"spur{self.min_m:g}-{self.max_m:g}"

    def targets(self, sink: B, ground: B, dist: F64, *, ear_m: float) -> B:
        return np.asarray((sink & (dist >= ear_m))
                          | (ground & (dist >= self.min_m) & (dist <= self.max_m)))

    def goal(self, sink: B, ground: B) -> B:
        return np.asarray(sink | ground)


# ---------------------------------------------------------------- detour saving (Idea C)


class Roads(NamedTuple):
    """The network on the PIX raster at a state (`roads_of`): per pixel its column (-1 off the
    network), the road points (network pixels, one per SRC_M square, the first by index), per
    road point the homes' population share nearest it in a straight line (summing to 1), and per
    road point its distance along the network to every network pixel (rows road points, columns
    the network's pixels; inf where the network inside the block does not join them)."""
    col: I64
    pts: I64
    w: F64
    D: NDArray[np.float32]

    def among(self) -> F64:
        """The road points' distances to each other (float64)."""
        return np.asarray(self.D[:, self.col[self.pts]], dtype=np.float64)

    def mean(self) -> float:
        """The mean distance along the network between two homes' road points (each drawn by
        w), over the pairs it joins."""
        DRR = self.among()
        fin = np.isfinite(DRR)
        ww = np.outer(self.w, self.w)
        return float((ww * np.where(fin, DRR, 0.0)).sum() / (ww * fin).sum())


def _net_graph(net: B, ea: I64, eb: I64, el: F64) -> tuple[I64, sp.csr_matrix]:
    """(per pixel its column, -1 off the network; the network's graph on its columns)."""
    idx = np.flatnonzero(net)
    col = -np.ones(len(net), dtype=np.int64)
    col[idx] = np.arange(len(idx))
    ok = net[ea] & net[eb]
    m = len(idx)
    return col, sp.coo_matrix((el[ok], (col[ea[ok]], col[eb[ok]])), shape=(m, m)).tocsr()


def roads_of(net: B, xy: F64, ea: I64, eb: I64, el: F64, *, homes: F64, share: F64) -> Roads:
    """The network `net` (per pixel whether it lies on a road: the faces' sinks) as Roads; pure in
    its pixel inputs (xy per pixel its centre, (ea, eb, el) the 8-neighbour edges and lengths),
    `homes` the homes' points and `share` their population shares."""
    col, G = _net_graph(net, ea, eb, el)
    idx = np.flatnonzero(net)
    key = np.floor(xy[idx] / SRC_M).astype(np.int64)
    _, first = np.unique(key[:, 0] * 1_000_000 + key[:, 1], return_index=True)
    pts = idx[np.sort(first)]
    w = np.bincount(cKDTree(xy[pts]).query(homes)[1], weights=share, minlength=len(pts))
    D = np.empty((len(pts), len(idx)), dtype=np.float32)
    for c0 in range(0, len(pts), CHUNK):
        D[c0:c0 + CHUNK] = dijkstra(G, directed=False, indices=col[pts[c0:c0 + CHUNK]])
    return Roads(col=col, pts=pts, w=w / w.sum(), D=D)


@njit(cache=True)
def _hinge(c: F64, w: F64, z: F64) -> F64:
    """Per row t of z: sum over (p, q) of w_p w_q (c[p, q] - z[t, q])^+. Per column q, c sorted
    (descending) with its prefix sums of w and w c, so each (t, q) is one binary search: the p
    with c[p, q] > z[t, q] are a prefix."""
    n, m = len(w), z.shape[0]
    out = np.zeros(m)
    for q in range(n):
        if w[q] == 0.0:
            continue
        o = np.argsort(-c[:, q])
        cs = c[o, q]
        W = np.cumsum(w[o])
        WC = np.cumsum(w[o] * np.where(np.isfinite(cs), cs, 0.0))
        top = cs[0]
        for t in range(m):
            zz = z[t, q]
            if not top > zz:
                continue
            lo, hi = 0, n
            while lo < hi:
                mid = (lo + hi) // 2
                if cs[mid] > zz:
                    lo = mid + 1
                else:
                    hi = mid
            out[t] += w[q] * (WC[lo - 1] - zz * W[lo - 1])
    return out


def savings(r: Roads, a: int, tg: I64, length: F64) -> F64:
    """Per target pixel t of `tg` (network pixels) the detour saving of a new link from the
    network pixel `a` to t of `length[t]` m: the mean, over pairs of homes (each drawn by the
    road points' w), of how much shorter the trip between their road points gets, S = sum_pq
    w_p w_q (D_pq - D'_pq) with D' = min(D_pq, D_pa + l + D_tq, D_pt + l + D_aq), the link used
    at most once. At most one orientation shortens a pair (the two together would make
    2 D_pq > D_pa + D_aq + D_pt + D_tq + 2 l >= 2 D_pq + 2 l, by the triangle inequality), and
    relabelling p, q maps the one onto the other, so S = 2 sum_pq w_p w_q (D_pq - D_pa - l -
    D_tq)^+ (`_hinge`). A pair the network inside the block does not join (D_pq inf) counts
    nothing."""
    DRR = r.among()
    Da = np.asarray(r.D[:, r.col[a]], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        c = np.where(np.isfinite(DRR) & np.isfinite(Da)[:, None], DRR - Da[:, None], -np.inf)
    z = length[:, None] + np.asarray(r.D[:, r.col[tg]], dtype=np.float64).T
    return 2.0 * _hinge(c, r.w, z)


class Field(Protocol):
    """The attraction at one state (Attract.field)."""

    def credits(self, a: int, tg: I64, reach: Reach) -> F64:
        """Per target pixel of `tg` the credit in P of the path from `a` to it (`reach`, a's
        search)."""
        ...

    def of(self, route: Route, *, spur: bool) -> tuple[float, float]:
        """A costed path's (detour saving in m, credit in P)."""
        ...


class Attract(Protocol):
    """A term added to the escape objective of `paths` (owner 2026-11-03, Idea C): an ear's score
    is (gain + credit) / cost at every stage (each source's target, the costed ears' order, the
    built ear's exact gain)."""

    @property
    def name(self) -> str: ...                   # its token in the plan's name

    def field(self, e: EarWorld, *, P0: float) -> Field: ...


class _NoField:
    def credits(self, a: int, tg: I64, reach: Reach) -> F64:
        return np.zeros(len(tg))

    def of(self, route: Route, *, spur: bool) -> tuple[float, float]:
        return 0.0, 0.0


@dataclass(frozen=True, kw_only=True)
class NoDetour:
    """The escape objective alone: no credit (`nodt`)."""

    @property
    def name(self) -> str:
        return "nodt"

    def field(self, e: EarWorld, *, P0: float) -> Field:
        return _NoField()


@dataclass(frozen=True, kw_only=True)
class _DetourField:
    roads: Roads
    xy: F64
    kappa: float                                 # P per metre of detour saving

    def credits(self, a: int, tg: I64, reach: Reach) -> F64:
        out = np.zeros(len(tg))
        on = self.roads.col[tg] >= 0             # sinks; a spur's open ground shortens nothing
        if not on.any():
            return out
        length = reach.length()
        out[on] = self.kappa * savings(self.roads, a, tg[on], length[tg[on]])
        return out

    def of(self, route: Route, *, spur: bool) -> tuple[float, float]:
        if spur:
            return 0.0, 0.0
        L = float(np.hypot(*np.diff(route.line, axis=0).T).sum())
        S = float(savings(self.roads, int(route.path[0]), route.path[-1:], np.array([L]))[0])
        return S, self.kappa * S


@dataclass(frozen=True, kw_only=True)
class Detour:
    """Detour-saving attraction (owner 2026-11-03, Idea C): a loop ear's credit is lam P0 S /
    S0, S its detour saving (`savings`: the mean metres a trip between two homes' road points
    saves, the network the state's sinks, `roads_of`) and S0 the mean such trip on the street
    alone (the block's own length scale, `Roads.mean`): lam is the credit, as a share of P0, of a
    saving as long as the mean street-only trip. A spur shortens no trip and earns nothing: the
    loop it sets up is credited once, when an ear closes it (its saving then large against its
    short length), which is what makes a nearly finished loop finish.

    Runs (2026-11-03, spur10-80, ears_compare.py): it ends spur takeover (dead-end share of
    length, 41785 f100 at D 0.4: .92 plain, .07 / .05 / .02 at lam 0.1 / 0.3 / 1; 5810 f2000 at
    D 0.15: .34 plain, .04 / .05 at lam 0.1 / 0.3), but buys no L_P: 41785 L_P at D 0.1 / 0.2 /
    0.4 plain .748 / .894 / .978, lam 0.1 .740 / .880 / .978, loop .656 / .855 / .980; 5810 at D
    0.08 / 0.10 / 0.15 plain .949 / .960 / .978, lam 0.1 .950 / .962 / .978, lam 0.3 .952 / .960
    / .977, loop .951 / .968 / .983, at 1.7 -- 1.9 times plain spurs' wall time."""
    lam: float

    def __post_init__(self) -> None:
        if not self.lam > 0:
            raise ValueError(f"lam {self.lam}: positive (NoDetour is the term off)")

    @property
    def name(self) -> str:
        return f"dt{self.lam:g}"

    def field(self, e: EarWorld, *, P0: float) -> Field:
        homes, share = e.homes()
        ra = e.ra
        r = roads_of(e.sinks(), ra.xy, ra.ea, ra.eb, ra.el, homes=homes, share=share)
        r0 = roads_of(e.sinks_of(e.street), ra.xy, ra.ea, ra.eb, ra.el, homes=homes,
                      share=share)
        return _DetourField(roads=r, xy=ra.xy, kappa=self.lam * P0 / r0.mean())


class Routing(NamedTuple):
    """A state's routing (Paths.routing): the Terrain, the sources (sink pixels, one per SRC_M
    square), per pixel its face (pf: before Terrain's own, which drops the faces under face_m2),
    whether it is a sink of a live face (bnd) and open ground (ground), and the faces' Arcs
    (EarWorld.arcs: its sink's ring, -1 off every sink, and place on its face's contour)."""
    ter: Terrain
    src: I64
    pf: I64
    bnd: B
    ground: B
    arcs: Arcs


@dataclass(frozen=True, kw_only=True)
class Paths:
    """Cheapest paths (module docstring); `router` how they are searched and drawn (Router:
    Grid8, Pulled, Theta, Eikonal on the raster; Visibility on the vector geometry);
    `bld` what a metre of centreline costs per unit of the building share of the road's
    cross-section there (`road_share`; inf the limit, lexicographic), `pairs` the sources whose
    best ears are built and costed; a face under `face_m2` is
    not split, an ear's ends at least `ear_m` apart in a straight line; `ends` where a path may
    end (Loop: at a sink; Spur: there or on open ground); `attract` the term added to the escape
    objective (NoDetour: none; Detour: detour saving)."""
    router: Router
    bld: float
    pairs: int
    face_m2: float
    ear_m: float
    ends: Ends
    attract: Attract

    @property
    def name(self) -> str:
        return (f"paths.{self.router.name}.b{self.bld:g}.f{self.face_m2:g}.e{self.ear_m:g}"
                f".{self.ends.name}"
                f".{self.attract.name}")

    def routing(self, e: EarWorld, faces: Face, gain: F64) -> Routing:
        """The state's routing field (Terrain) and its sources."""
        g, ra = e.g, e.ra
        pf = np.where(ra.tri >= 0, faces.lab[np.maximum(ra.tri, 0)], -1)    # pixel's face
        big = faces.area >= self.face_m2
        live = (pf >= 0) & big[np.maximum(pf, 0)]
        # the ends: sink pixels of the face (a road or the street, not a wall), on their rings
        arcs = e.arcs(faces)
        bnd = live & (arcs.ring >= 0)
        ground = live & (ra.bld == 0) & ~bnd       # open ground: where a spur may end
        # per pixel its first-order gain density (the triangles under it)
        gp = np.bincount(_pix_of(ra, g.cen), weights=gain, minlength=len(pf))
        ter = Terrain(shape=ra.shape, lo=ra.lo, xy=ra.xy, face=np.where(live, pf, -1),
                      share=ra.share, gain=gp, bld=self.bld, ea=ra.ea, eb=ra.eb, el=ra.el,
                      goal=self.ends.goal(bnd, ground),
                      vec=MeshShapes(xy=g.m.fx.fm.xy, tri=g.m.fx.fm.tri, lab=faces.lab))
        src = np.flatnonzero(bnd)
        key = np.floor(ra.xy[src] / SRC_M).astype(np.int64)
        _, first = np.unique(key[:, 0] * 1_000_000 + key[:, 1], return_index=True)
        return Routing(ter=ter, src=src[np.sort(first)], pf=pf, bnd=bnd, ground=ground,
                       arcs=arcs)

    def ears(self, e: EarWorld, faces: Face, gain: F64, *, cost: gr.Cost,
             scale: float) -> list[Ear]:
        g, ra = e.g, e.ra
        ter, src, pf, bnd, ground, arcs = self.routing(e, faces, gain)
        ring = arcs.ring
        fld = self.attract.field(e, P0=g.P0)
        # the best `pairs` sources so far, best first (a stable sort truncated: the same as
        # every source's sorted and cut once, ties in insertion order), each with its reach
        best: list[tuple[tuple[int, float], int, int, Reach]] = []
        lex = ter.lex()
        for reach in self.router.reaches(ter, src):
            a = reach.src
            d = reach.cost
            fin = np.isfinite(d)
            acc = reach.gain
            crow = np.hypot(*(ra.xy - ra.xy[a]).T)     # straight-line, from the source
            own = bnd & (ring == ring[a])              # a loop ends on its source's ring
            tg = np.flatnonzero(self.ends.targets(own, ground, crow, ear_m=self.ear_m)
                                & fin & (pf == pf[a]))
            ln = reach.length()[tg]
            X, Y = cuts(arcs, int(pf[a]), int(a), tg, reach.area[tg])
            big = np.minimum(X, Y) - CUT_ALLOW * gr.PATCH_R * ln >= gr.FACE_M2
            keep = ~bnd[tg] | big                      # a loop must split its face; a spur not
            tg, ln = tg[keep], ln[keep]
            if not len(tg):
                continue
            cr = fld.credits(int(a), tg, reach)
            tier, r = rank(acc[tg] + scale * cr, d[tg], ln if lex else None)
            k = int(np.lexsort((-r, -tier))[0])
            t = int(tg[k])
            best.append(((int(tier[k]), float(r[k])), int(a), t, reach))
            if len(best) > 2 * self.pairs:
                best.sort(key=lambda q: (-q[0][0], -q[0][1]))
                del best[self.pairs:]
        best.sort(key=lambda q: (-q[0][0], -q[0][1]))
        out = []
        for _, a, t, reach in best[:self.pairs]:
            rt = reach.route(t)
            spur = not bool(bnd[t])
            tris = e.corridor(rt.line, spur=spur)
            if not len(tris):
                continue
            dD, dC = g.costs_of(tris)
            c = float(cost.of(np.array([dD]), np.array([dC]), P0=g.P0)[0])
            sv, cr = fld.of(rt, spur=spur)
            out.append(Ear(face=int(pf[a]), path=rt.path, line=rt.line, spur=spur, tris=tris,
                           dD=dD, dC=dC, cost=c, first=float(gain[tris].sum()), saving=sv,
                           credit=cr, exact=np.nan))
        return out


def rank(num: F64, cost: F64, length: F64 | None) -> tuple[I64, F64]:
    """Per target its rank (a tier, then a value, both the higher the better) by the escape
    objective num (gain and credit) per path cost: at finite b one tier, num / cost; at binf
    (length given, cost the path's B) the limit of num / (length + b B) as b -> inf: a path with
    B 0 ranks above any with B > 0 (whose ratio falls as 1 / b), the first by num / length, the
    second by num / B."""
    if length is None:
        return np.zeros(len(num), dtype=np.int64), num / np.maximum(cost, 1e-9)
    free = cost == 0.0
    return (free.astype(np.int64),
            np.where(free, num / np.maximum(length, 1e-9), num / np.where(free, 1.0, cost)))


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
            tau: float, stretch_m: float, ear_m: float) -> list[Divide]:
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
    least ear_m away that keeps highest in u (the network's edges weighted by length over
    u, a shortest path), so every cut runs sink to sink (no dead end)."""
    on = lab == face
    if not on.any():
        return []
    both = on[ea] & on[eb]
    a, b = ea[both], eb[both]
    edge = on & sink
    umax = float(u[on].max())
    out = _cuts(u, xy, _ms_cut(u, on, edge, a, b, tau=tau), edge, a, b, umax=umax, kind="ms",
                ear_m=ear_m)
    if not out:
        out = _cuts(u, xy, _harmonic_cut(xy, bld, on, edge, a, b, stretch_m=stretch_m), edge,
                    a, b, umax=umax, kind="harmonic", ear_m=ear_m)
    out.sort(key=lambda d: -d.persistence)
    return out


def _cuts(u: F64, xy: F64, cut: B, edge: B, a: I64, b: I64, *, umax: float,
          kind: str, ear_m: float) -> list[Divide]:
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
            far = np.hypot(*(xy[ends] - xy[s0]).T) >= ear_m
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
    """The divides of u (`divides`, ends at least `ear_m` apart), per face of at least `face_m2`
    on its own u (the state's,
    the roads grounded), at persistence threshold `tau`, the harmonic-measure split where the
    strict basins degenerate, near-duplicates dropped (`_distinct`); across faces the `pairs`
    with the most first-order gain per metre costed (`offers`). Rev 1 took the `pairs` most
    persistent across faces, but persistence is relative to its face's largest u, so on 5810 the
    small faces' divides (u 2 -- 60) filled the pairs and the large faces' (u 160 -- 230, 7 -- 1000
    times the costed ones' first-order gain per cost) were never costed."""
    tau: float
    pairs: int
    face_m2: float
    ear_m: float

    @property
    def name(self) -> str:
        return f"divides.p{self.tau:g}.f{self.face_m2:g}.e{self.ear_m:g}"

    def offers(self, e: EarWorld, faces: Face, gain: F64) -> list[Offer]:
        """Every face's distinct divides, the most first-order gain per metre first."""
        g, ra = e.g, e.ra
        pf = np.where(ra.tri >= 0, faces.lab[np.maximum(ra.tri, 0)], -1)
        gp = np.bincount(_pix_of(ra, g.cen), weights=gain, minlength=len(pf))
        sink = e.sinks()
        out: list[Offer] = []
        for f in np.flatnonzero(faces.area >= self.face_m2):
            ds = divides(e.u_pix, pf, sink, ra.ea, ra.eb, ra.xy, ra.bld, face=int(f),
                         tau=self.tau, stretch_m=STRETCH_M, ear_m=self.ear_m)
            for d in _distinct(ds, ra.xy):
                L = float(np.hypot(*np.diff(ra.xy[d.pixels], axis=0).T).sum())
                out.append(Offer(face=int(f), divide=d,
                                 per_m=float(gp[d.pixels].sum()) / max(L, PIX)))
        out.sort(key=lambda o: -o.per_m)
        return out

    def ears(self, e: EarWorld, faces: Face, gain: F64, *, cost: gr.Cost,
             scale: float) -> list[Ear]:
        return self.costed(e, self.offers(e, faces, gain)[:self.pairs], gain, cost=cost)

    @staticmethod
    def costed(e: EarWorld, offers: list[Offer], gain: F64, *, cost: gr.Cost) -> list[Ear]:
        g, ra = e.g, e.ra
        out = []
        for o in offers:
            f, d = o.face, o.divide
            tris = e.corridor(ra.xy[d.pixels], spur=False)
            if not len(tris):
                continue
            dD, dC = g.costs_of(tris)
            c = float(cost.of(np.array([dD]), np.array([dC]), P0=g.P0)[0])
            out.append(Ear(face=f, path=d.pixels, line=ra.xy[d.pixels], spur=False, tris=tris,
                           dD=dD, dC=dC, cost=c,
                           first=float(gain[tris].sum()), saving=0.0, credit=0.0, exact=np.nan))
        return out


def _pix_of(ra: Raster, xy: F64) -> I64:
    return _pix_of_lo(ra.lo, ra.shape, xy)


SITE_M = PIX * np.sqrt(2.0)   # m: a corridor's patch sites at most this far apart (`sites`)


def sites(line: F64) -> F64:
    """The polyline's points with each segment longer than SITE_M cut evenly into pieces no
    longer (EarWorld.corridor's patch sites: an any-angle line's vertices can lie a road's length
    apart, and a ROAD_W band needs its patches overlapping). A grid path's segments are no longer
    than SITE_M, so its points come back as they were."""
    out = [line[:1]]
    for p, q in zip(line[:-1], line[1:], strict=True):
        n = max(1, int(np.ceil(np.hypot(*(q - p)) / SITE_M - 1e-9)))
        out.append(p + (np.arange(1, n + 1) / n)[:, None] * (q - p))
    return np.vstack(out)


def line_pixels(lo: F64, shape: tuple[int, int], line: F64) -> I64:
    """The pixels (flat) the polyline `line` (m) crosses, in order, each run of one pixel once:
    each segment sampled at (k + 1/2) / m of its length, m one per PIX / 4 (a grid path's line,
    centre to neighbouring centre, gives back its pixels: no sample on a corner)."""
    pts = [line[:1]]
    for p, q in zip(line[:-1], line[1:], strict=True):
        m = max(1, int(np.ceil(4 * np.hypot(*(q - p)) / PIX)))
        pts.append(p + ((np.arange(m) + 0.5) / m)[:, None] * (q - p))
    pts.append(line[-1:])
    pix = _pix_of_lo(lo, shape, np.vstack(pts))
    keep = np.ones(len(pix), dtype=bool)
    keep[1:] = pix[1:] != pix[:-1]
    return np.asarray(pix[keep])


def _pix_of_lo(lo: F64, shape: tuple[int, int], xy: F64) -> I64:
    """Per point (m) its pixel (flat) on the PIX raster with lower corner lo, clipped."""
    ij = np.floor((xy - lo) / PIX).astype(np.int64)
    ij[:, 0] = np.clip(ij[:, 0], 0, shape[0] - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, shape[1] - 1)
    return np.asarray(ij[:, 0] * shape[1] + ij[:, 1])


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
        m = re.fullmatch(r"ears\.(?:paths\.(?P<router>[a-z0-9]+)\.b(?P<bld>[0-9.]+|inf)"
                         r"\.f(?P<pf>[0-9.]+)\.e(?P<pe>[0-9.]+)"
                         r"\.(?P<ends>loop|spur(?P<lo>[0-9.]+)-(?P<hi>[0-9.]+))"
                         r"\.(?:nodt|dt(?P<lam>[0-9.]+))"
                         r"|divides\.p(?P<tau>[0-9.]+)\.f(?P<df>[0-9.]+)\.e(?P<de>[0-9.]+))"
                         r"\.gl(?P<h>[0-9.]+)Croad\.n(?P<nu>[0-9.e-]+)s(?P<price>[0-9.e-]+)"
                         r"\.M(?P<pairs>[0-9]+)\.K(?P<exact>[0-9]+)\.D(?P<D>[0-9.]+)", spec)
        if m is None:
            raise SystemExit(f"{spec}: a plan is ears.<paths.<router>.b<bld>.f<face_m2>.e<ear_m>"
                             ".<loop|spur<min_m>-<max_m>>.<nodt|dt<lam>>|divides.p<tau>.f<face_m2>"
                             ".e<ear_m>>"
                             ".gl<h>Croad.n<nu>s<price>.M<pairs>.K<exact>.D<D>")
        cands: Candidates
        if m["bld"] is not None:
            ends: Ends = (Loop() if m["ends"] == "loop" else
                          Spur(min_m=float(m["lo"]), max_m=float(m["hi"])))
            att: Attract = NoDetour() if m["lam"] is None else Detour(lam=float(m["lam"]))
            bld = float(m["bld"])
            cands = Paths(router=router_of(m["router"], bld=bld), bld=bld,
                          pairs=int(m["pairs"]), face_m2=float(m["pf"]),
                          ear_m=float(m["pe"]), ends=ends, attract=att)
        else:
            cands = Divides(tau=float(m["tau"]), pairs=int(m["pairs"]), face_m2=float(m["df"]),
                            ear_m=float(m["de"]))
        return Plan(candidates=cands, contrast=gr.GroundedLeak(h=float(m["h"])),
                    cost=gr.Cost(nu=float(m["nu"]), price=float(m["price"])),
                    exact=int(m["exact"]), D=float(m["D"]))


class Arcs(NamedTuple):
    """A state's faces' boundaries on the raster (`arcs_of`): per pixel (flat) the ring its sink
    lies on (-1 off the sinks), and per sink the contour it lies on (cid; -1 none), its place
    there (k: the nearest of the contour's vertices, v, at (vx, vy) m about the raster's lower
    corner), h, the signed area (`_tri`) the contour sweeps from its first vertex to v plus the
    segment from v to the sink's centre, and the contour's signed area (tot); per face (by its
    label) the sign of its outer contours' areas (sign: the side the face is on; a hole's
    contour has the other) and the area its outer contours enclose (outer: its holes'
    included)."""
    ring: I64
    cid: I64
    k: I64
    h: F64
    tot: F64
    vx: F64
    vy: F64
    sign: F64
    outer: F64


def _chain(cs: list[F64]) -> F64:
    """One ring's contours in one face (closed vertex loops, the largest first after sorting)
    as one loop: each next one entered and left at its vertex nearest the loop so far, the loop
    turned to start there (the contours of a ring a face's pixels touch along lobes the raster
    joins at a diagonal only: in the mesh one boundary component). The bridges, there and back,
    sweep no area net."""
    cs = sorted(cs, key=len, reverse=True)
    out = cs[0]
    for c in cs[1:]:
        d, j = cKDTree(c).query(out, k=1)
        i = int(np.argmin(d))
        out = np.vstack([np.roll(out, -i, axis=0), np.roll(c, -int(j[i]), axis=0)])
    return out


def saddles(pfp: I64) -> B:
    """Per corner of the padded pixel faces' 2 x 2 windows (shape one less each way) whether it
    is a saddle: one label on a diagonal, neither of the other diagonal's that one. Which way
    the raster joins there is the mesh's to say (EarWorld.arcs: the label at the corner)."""
    a, b, c, d = pfp[:-1, :-1], pfp[1:, 1:], pfp[1:, :-1], pfp[:-1, 1:]
    return np.asarray(((a == b) & (c != a) & (d != a)) | ((c == d) & (a != c) & (b != c)))


def face_loops(pfp: I64, corner: I64, f: int) -> list[F64]:
    """Face f's contours (closed vertex loops, m about the raster's lower corner; its first
    vertex not repeated) on the padded pixel faces `pfp`, each saddle (`saddles`) joined the way
    `corner` (the label at each window's corner) says: skimage's marching squares at 0.5 on the
    raster split in two each way, the four quarter pixels at a saddle's corner f where the corner
    is f's, not f where not. A contour's vertices run mid-way between quarter pixels; with no
    saddle the loop is the pixels' boundary less a corner's (PIX / 4)^2 / 2 at each turn."""
    pfp0 = pfp
    sl = ndimage.find_objects((pfp0 == f).astype(np.int64))[0]
    r0, c0 = max(sl[0].start - 1, 0), max(sl[1].start - 1, 0)
    r1, c1 = sl[0].stop + 1, sl[1].stop + 1
    m = pfp0[r0:r1, c0:c1] == f
    M = np.repeat(np.repeat(m, 2, axis=0), 2, axis=1)
    win = pfp0[r0:r1, c0:c1]
    a, b, c, d = win[:-1, :-1], win[1:, 1:], win[1:, :-1], win[:-1, 1:]
    sd = ((a == f) & (b == f) & (c != f) & (d != f)) | ((c == f) & (d == f) & (a != f) & (b != f))
    for i, j in np.argwhere(sd):
        M[2 * i + 1:2 * i + 3, 2 * j + 1:2 * j + 3] = corner[r0 + i, c0 + j] == f
    out = []
    for cv in measure.find_contours(M.astype(np.float64), 0.5, fully_connected="low"):
        out.append((cv[:-1] / 2.0 + [r0 - 0.75, c0 - 0.75]) * PIX)
    return out


def arcs_of(pf: I64, sink: B, corner: I64) -> Arcs:
    """Arcs of the raster's pixel faces `pf` (2-D; -1 in no face: the network, the raster padded
    by one such pixel so the street's outside is one), sinks (flat) and, per corner of the
    padded raster's 2 x 2 windows, the label there where it is a saddle (`saddles`; the mesh's,
    EarWorld.arcs). The network in pieces: 4-connected, and joined at a saddle of two network
    pixels whose corner is network; a sink's ring the piece of the network pixel nearest it.
    Each face's contours (`face_loops`: the saddles joined the same way), each one ring's (the
    piece nearest the non-face pixels its vertices border, most of them), an outer one where its
    area has the largest one's sign (a face the raster cuts at a neck has more than one), a
    hole's else, a ring's contours in a face one loop (`_chain`); a sink's vertex the nearest of
    its ring's loop's in its face. A path between a face's pixels that crosses a saddle whose
    corner is not the face's leaves it there (it crosses a road, its pieces miscounted:
    Paths.ears drops it where they come out under FACE_M2)."""
    nx, ny = pf.shape
    pfp = np.pad(pf, 1, constant_values=-1)
    net = pfp < 0
    lab, nl = ndimage.label(net)
    # network saddles whose corner is network: their two diagonal pieces one
    sd = saddles(pfp) & (corner < 0)
    i1, j1 = np.nonzero(sd & net[:-1, :-1] & net[1:, 1:])
    i2, j2 = np.nonzero(sd & net[1:, :-1] & net[:-1, 1:])
    u = np.concatenate([lab[i1, j1], lab[i2 + 1, j2]])
    v = np.concatenate([lab[i1 + 1, j1 + 1], lab[i2, j2 + 1]])
    _, comp = connected_components(
        sp.coo_matrix((np.ones(len(u)), (u, v)), shape=(nl + 1, nl + 1)), directed=False)
    lab = np.where(net, comp[lab], -1)
    _, (ii, jj) = ndimage.distance_transform_edt(~net, return_indices=True)
    near = lab[ii, jj]                               # padded: the nearest network piece
    n = nx * ny
    flat = pf.ravel()
    ring = np.where(sink, near[1:-1, 1:-1].ravel(), -1)
    cid = np.full(n, -1, dtype=np.int64)
    kk = np.zeros(n, dtype=np.int64)
    h, tot, vx, vy = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    nf = int(flat.max()) + 1 if flat.max() >= 0 else 0
    sign, outer = np.zeros(nf), np.zeros(nf)
    xy = np.column_stack([(np.arange(n) // ny + 0.5) * PIX, (np.arange(n) % ny + 0.5) * PIX])
    c_id = 0
    for f in np.unique(flat[sink & (flat >= 0)]):
        by: dict[int, list[F64]] = {}
        for P in face_loops(pfp, corner, int(f)):
            # the padded pixel either side of each vertex (mid-way between quarter pixels): the
            # one not in f, its nearest network piece
            g_ = P / PIX + 1.0                       # padded pixel units, centres at k + 0.5
            nb = np.floor(np.stack([g_ - 0.125, g_ + 0.125])).astype(np.int64)   # 2 x V x 2
            nb[..., 0] = np.clip(nb[..., 0], 0, nx + 1)
            nb[..., 1] = np.clip(nb[..., 1], 0, ny + 1)
            inf_ = pfp[nb[0, :, 0], nb[0, :, 1]] == f
            o = np.where(inf_[:, None], nb[1], nb[0])
            rl = int(np.bincount(np.maximum(near[o[:, 0], o[:, 1]], 0)).argmax())
            by.setdefault(rl, []).append(P)
        verts, rings_c, ids, cum, tots = [], [], [], [], []
        for rl, cs in by.items():
            P = _chain(cs)
            Q = np.roll(P, -1, axis=0)
            seg = 0.5 * (P[:, 0] * Q[:, 1] - Q[:, 0] * P[:, 1])
            verts.append(P)
            rings_c.append(np.full(len(P), rl))
            ids.append(np.full(len(P), c_id))
            cum.append(np.concatenate([[0.0], np.cumsum(seg)[:-1]]))
            tots.append(np.full(len(P), seg.sum()))
            c_id += 1
        V_, R_, I_, C_, T_ = (np.concatenate(x) for x in (verts, rings_c, ids, cum, tots))
        ts = np.array([t[0] for t in tots])
        sign[f] = np.sign(ts[np.argmax(np.abs(ts))])
        outer[f] = float(np.abs(ts[sign[f] * ts > 0]).sum())
        K_ = np.concatenate([np.arange(len(v)) for v in verts])
        own = np.flatnonzero(sink & (flat == f))
        for r in np.unique(ring[own]):
            on = np.flatnonzero(R_ == r)
            q = own[ring[own] == r]
            if not len(on):
                continue
            _, j = cKDTree(V_[on]).query(xy[q], k=1)
            v = on[j]
            cid[q], kk[q], tot[q] = I_[v], K_[v], T_[v]
            vx[q], vy[q] = V_[v, 0], V_[v, 1]
            h[q] = C_[v] + 0.5 * (V_[v, 0] * xy[q, 1] - xy[q, 0] * V_[v, 1])
    return Arcs(ring=ring, cid=cid, k=kk, h=h, tot=tot, vx=vx, vy=vy, sign=sign, outer=outer)


CUT_ALLOW = 1.0        # cuts: the corridor's share of each piece, in PATCH_R per metre of path
                       # (a corridor 2 PATCH_R wide about its centreline: PATCH_R a side)


def cuts(ar: Arcs, f: int, a: int, tg: I64, area: F64) -> tuple[F64, F64]:
    """The two pieces (m^2) the path from sink a to each of its face f's sinks tg (on a's ring;
    `area` each path's swept area, Reach.area) cuts the face into, gross of the face's other
    holes (a hole within a piece counts in it), by Green's theorem: X the signed area the path
    and the contour from the target forward to a bound (the path's swept area and the
    contour's, Arcs.h and tot), times the face's side (sign). On the outer contour X is a piece
    and the face's outer area less X the other. On a hole's, X is the piece the path cuts off
    when positive (the arc on the face's side) and that piece with the hole when negative (the
    arc round the hole's far side), the other the outer area less both. nan where a target
    lies on another contour than a's."""
    s1 = (area + ar.h[a] - ar.h[tg] + np.where(ar.k[a] < ar.k[tg], ar.tot[tg], 0.0))
    X = ar.sign[f] * s1
    hole = np.abs(ar.tot[tg]) * (ar.sign[f] * ar.tot[tg] < 0)
    P = np.where(X < 0, -X - hole, X)
    other = ar.cid[tg] != ar.cid[a]
    return np.where(other, np.nan, P), np.where(other, np.nan, ar.outer[f] - P - hole)


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
        return self.sinks_of(self.bounding())

    def arcs(self, faces: Face) -> Arcs:
        """The faces' boundaries on the raster (`arcs_of`; pixels in no face, grown or off the
        block, the network): per sink the ring it lies on (`sinks`), the network's piece nearest it
        (the raster's diagonal saddles joined as the mesh is at their corners), and its place on
        its face's contour. A face's boundary components are
        rings: the street's, and a hole's where the block has one (5810, 132 m^2; 6310, 21634,
        23743), until a road joins it to the street. A cut between two rings splits nothing (in
        a plane region a cut splits only between two points of one boundary component), so a
        loop ends on its source's ring
        (Paths.ears). Rev 3 offered such cuts: on 5810 after Theta's first ear at b0 every
        source's best target was on the hole, its 8 ears were refused and the run stopped. (The
        bounding triangles are no network to label: the street's are one per street face, not
        edge-connected.) Paths.ears' cut test reads the contours (`cuts`)."""
        ra = self.ra
        pf = np.where(ra.tri >= 0, faces.lab[np.maximum(ra.tri, 0)], -1).reshape(ra.shape)
        # each saddle's corner: the label of the triangle there (as a pixel's, `raster_of`)
        sd = saddles(np.pad(pf, 1, constant_values=-1))
        corner = np.full(sd.shape, -1, dtype=np.int64)
        ij = np.argwhere(sd)
        xy = ra.lo + ij * PIX
        boundary: Any = self.g.m.geo.boundary
        inside = shapely.contains_xy(boundary, xy[:, 0], xy[:, 1])
        corner[ij[:, 0], ij[:, 1]] = np.where(inside, faces.lab[self.g.tree.query(xy)[1]], -1)
        return arcs_of(pf, self.sinks(), corner)

    def sinks_of(self, bounding: B) -> B:
        """`sinks` with the triangles `bounding` the roads (Detour: the street's alone)."""
        att = np.flatnonzero(bounding)
        d, _ = cKDTree(self.g.cen[att]).query(self.ra.xy, k=1, distance_upper_bound=PIX)
        return np.asarray(np.isfinite(d) & (self.ra.tri >= 0))

    def homes(self) -> tuple[F64, F64]:
        """(per home its footprint's centroid, its population share)."""
        c = shapely.centroid(np.asarray(self.g.m.geo.polys))
        if len(c) != len(self.g.w):
            raise ValueError(f"{len(c)} footprints, {len(self.g.w)} homes' shares")
        return np.column_stack([shapely.get_x(c), shapely.get_y(c)]), self.g.w

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

    def corridor(self, pts: F64, *, spur: bool) -> I64:
        """An ear's corridor along the polyline `pts`, extended at both ends to the nearest
        bounding triangle (the road or street it joins; a spur's at its start only, its end on
        open ground): the ungrown triangles within growth.PATCH_R of its points (`sites`: at most
        a diagonal pixel apart, as a grid path's are), and every triangle its centreline crosses
        (sampled every STEP_M): on 5810's open ground a triangle can be larger than a road, and
        one whose centroid lies off the corridor but which straddles it joins the two halves (the
        first runs' ears split nothing)."""
        g = self.g
        att = np.flatnonzero(self.bounding())
        _, k = cKDTree(g.cen[att]).query(pts[[0, -1]], k=1)
        line = np.vstack([g.cen[att[k[0]]], pts] + ([] if spur else [g.cen[att[k[1]]]]))
        seg = np.diff(line, axis=0)
        L = np.hypot(*seg.T)
        dense = [line[:1]]
        for p0, d, n in zip(line[:-1], seg, np.ceil(L / STEP_M).astype(int), strict=True):
            dense.append(p0 + np.linspace(0.0, 1.0, max(n, 1) + 1)[1:, None] * d)
        P = np.vstack(dense)
        lists = g.tree.query_ball_point(sites(pts), r=gr.PATCH_R)
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


class Scored(NamedTuple):
    """A rescored candidate (`offer`): the ear, its exact gain filled in, and the faces with it
    built."""
    ear: Ear
    faces: Face


class Offered(NamedTuple):
    """A state's step (`offer`): its costed candidates, best first order per cost first, and the
    `exact` splitting ones (and spurs) among them rescored by the restricted solve."""
    cands: list[Ear]
    scored: list[Scored]


def ratio(q: Ear) -> float:
    """An ear's exact score: (exact gain + credit) / cost (build's pick)."""
    return (q.exact + q.credit) / q.cost


def scale_of(scored: list[Scored], *, scale: float) -> float:
    """The credit's weight after a step (build): its rescored ears' first order over their exact
    gain, `scale` (the last step's) where they have none."""
    ex = sum(s.ear.exact for s in scored)
    return sum(s.ear.first for s in scored) / ex if ex > 0 else scale


def offer(e: EarWorld, plan: Plan, faces: Face, st: gcl.State, u: F64, *, scale: float,
          log: Any) -> Offered:
    """The state's step (g.step the grown roads, (st, u) its solve, `faces` its faces): the
    candidates, and the `exact` splitting ones rescored; build picks the best `ratio`, a search
    (atlas/lookahead) may look further. Leaves g.step as it found it."""
    g = e.g
    gain = plan.contrast.gains(g, st, u)
    e.set_u(u)
    cands = plan.candidates.ears(e, faces, gain, cost=plan.cost, scale=scale)
    cands.sort(key=lambda q: -(q.first + scale * q.credit) / q.cost)
    on = g.step >= 0
    mark = int(g.step.max()) + 1
    # the splitting ones only (no dead end) and the spurs (a dead end by design: Spur), best
    # first order per cost first, `exact` of them
    split: list[tuple[Ear, Face]] = []
    for q in cands:
        if len(split) == plan.exact:
            break
        g.step[q.tris] = mark
        nf = e.faces()
        g.step[q.tris] = -1
        if nf.n > faces.n or q.spur:
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
        scored.append(Scored(ear=q._replace(exact=before[q.face] - e.P_face(on2, cells)),
                             faces=nf))
    return Offered(cands=cands, scored=scored)


def build(e: EarWorld, plan: Plan, *, out: Path, log: Any, steps: int | None = None) -> None:
    """Ears greedily until D reaches plan.D (or `steps` ears: the checks' short runs)."""
    g = e.g
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []
    picks: list[dict[str, float]] = []
    t0 = time.time()
    faces = e.faces()
    n0 = faces.n
    # a credit's weight where the first order stands in for the exact gain: the last step's
    # rescored ears' first order over their exact gain (the grounded leak's first order
    # overstates it 10^3 -- 10^5 times); 0 before any exact gain (step 0: credit in the exact
    # stage only), as growth.grow's islands at its first step
    scale = 0.0
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
        off = offer(e, plan, faces, st, u, scale=scale, log=log)
        if not off.cands:
            log("no ear left: stopped short of the budget")
            break
        if not off.scored:
            log("no candidate ear splits a face: stopped")
            break
        built, faces = max(off.scored, key=lambda s: ratio(s.ear))
        scale = scale_of(off.scored, scale=scale)
        g.step[built.tris] = step
        xy = built.line
        picks.append(dict(step=step, face=built.face, spur=built.spur,
                          ax=float(xy[0, 0]), ay=float(xy[0, 1]),
                          bx=float(xy[-1, 0]), by=float(xy[-1, 1]),
                          length=float(np.hypot(*np.diff(xy, axis=0).T).sum()),
                          n_tris=len(built.tris), dD=built.dD, dC=built.dC, cost=built.cost,
                          first=built.first, exact=built.exact, saving=built.saving,
                          credit=built.credit, offered=len(off.cands)))
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

CHECK_PLAN = "ears.paths.grid8.b20.f2000.e30.loop.nodt.gl0.25Croad.n2s0.1.M8.K2.D1"
CHECK_SPUR = "ears.paths.grid8.b20.f2000.e30.spur10-80.nodt.gl0.25Croad.n2s0.1.M16.K2.D1"
CHECK_DIVIDES = "ears.divides.p0.1.f2000.e30.gl0.25Croad.n2s0.1.M16.K4.D1"
CHECK_DETOUR = "ears.paths.grid8.b20.f2000.e30.loop.dt1.gl0.25Croad.n2s0.1.M8.K2.D1"


def _horseshoe(arm_m: float, gap_m: float, *, crow: bool) -> tuple[float, float]:
    """(detour saving S in m, the closing link's length) of the link across a horseshoe's gap:
    a street along y = 10 m, two arms HS_W m apart rising arm_m from it, joined at the top by two
    bars that stop gap_m apart (the network the pixels within PIX of these segments), homes on a
    HS_HOME_M grid over the whole raster; the link joins the network's two tips (on the raster
    about gap_m apart: its length is returned). crow: the `horseshoe` check's fault (the road
    points' distances straight lines)."""
    W, H = 200.0, arm_m + 40.0
    shape = (int(W / PIX), int(H / PIX))
    i, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    xy = np.column_stack([(i.ravel() + 0.5) * PIX, (j.ravel() + 0.5) * PIX])
    x0, x1, top = 100.0 - HS_W / 2, 100.0 + HS_W / 2, 10.0 + arm_m
    segs = shapely.MultiLineString([[(0.0, 10.0), (W, 10.0)], [(x0, 10.0), (x0, top)],
                                    [(x1, 10.0), (x1, top)], [(x0, top), (100.0 - gap_m / 2, top)],
                                    [(x1, top), (100.0 + gap_m / 2, top)]])
    net = shapely.distance(segs, shapely.points(xy)) <= PIX
    ea, eb, el = _grid_edges(shape)
    hx, hy = np.meshgrid(np.arange(HS_HOME_M / 2, W, HS_HOME_M),
                         np.arange(HS_HOME_M / 2, H, HS_HOME_M), indexing="ij")
    homes = np.column_stack([hx.ravel(), hy.ravel()])
    r = roads_of(net, xy, ea, eb, el, homes=homes, share=np.full(len(homes), 1.0 / len(homes)))
    on = np.flatnonzero(net)
    if crow:
        d = xy[r.pts][:, None, :] - xy[on][None, :, :]
        r = r._replace(D=np.hypot(d[..., 0], d[..., 1]).astype(np.float32))
    mid = np.hypot(*(xy[on] - [100.0, top]).T)          # the tips: nearest the gap's middle
    left = xy[on, 0] < 100.0
    a = int(on[left][np.argmin(mid[left])])
    t = int(on[~left][np.argmin(mid[~left])])
    L = float(np.hypot(*(xy[t] - xy[a])))
    return float(savings(r, a, np.array([t]), np.array([L]))[0]), L


def horseshoe(fault: str | None) -> None:
    """Experiment 1 (owner's doc, Idea C) as a check, on synthetic horseshoes (`_horseshoe`):
    with the gap fixed (HS_GAP_M) the closing link's saving S rises strictly with the arms
    (HS_ARMS: closure more likely the longer the horseshoe), and with the arms fixed (HS_ARM_M)
    its saving per metre of link S / l rises strictly as the gap closes (HS_GAPS: a spur growing
    toward the far arm makes the closing ear ever better per cost, so a nearly finished loop
    finishes). Fault `crow` (straight-line distances in place of distances along the roads: no
    trip is shorter by a link, by the triangle inequality) fails it."""
    if fault not in (None, "crow"):
        raise SystemExit("horseshoe's fault: crow")
    crow = fault == "crow"
    by_arm = [_horseshoe(A, HS_GAP_M, crow=crow) for A in HS_ARMS]
    by_gap = [_horseshoe(HS_ARM_M, g, crow=crow) for g in HS_GAPS]
    for A, (S, L) in zip(HS_ARMS, by_arm, strict=True):
        print(f"  arms {A:g} m, gap {HS_GAP_M:g} m: link {L:.1f} m, S {S:.3f} m")
    for g_, (S, L) in zip(HS_GAPS, by_gap, strict=True):
        print(f"  arms {HS_ARM_M:g} m, gap {g_:g} m: link {L:.1f} m, S {S:.3f} m, "
              f"S / l {S / L:.4f}")
    ok = (all(b[0] > a[0] for a, b in zip(by_arm, by_arm[1:], strict=False))
          and all(b[0] / b[1] > a[0] / a[1] for a, b in zip(by_gap, by_gap[1:], strict=False)))
    print(f"horseshoe {fault or 'none'}: -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


# ---------------------------------------------------------------- the routers' checks

def routers_at(b: float) -> tuple[Router, ...]:
    """The routers the checks run at building cost b: Grid8, Pulled and Theta at every b, Eikonal
    at finite b, Visibility at b0 (each refused elsewhere)."""
    eik: tuple[Router, ...] = (Eikonal(bld=b),) if np.isfinite(b) else ()
    vis: tuple[Router, ...] = (Visibility(bld=b),) if b == 0 else ()
    return (Grid8(), Pulled(inner=Grid8()), Theta(), *eik, *vis)


OCTILE = float(np.cos(np.pi / 8) + (np.sqrt(2) - 1) * np.sin(np.pi / 8)) - 1   # 0.0824
RC_N = 161             # router checks: the open raster's side (pixels)
RC_NEAR_M = 20.0       # m: cost: pixels this near the source not judged
RC_R_PX = 60           # straight, gain: the targets this many pixels from the source
RC_ANGLES = (10.0, 22.5, 30.0, 37.0, 60.0, 80.0)   # straight: the targets' bearings (degrees)
TOL_EXACT = 1e-9       # relative: a cost or gain the router computes exactly (Theta, Pulled's line)
TOL_EIK = 5e-3         # relative: Eikonal's cost on open ground (an open 161 x 161 raster:
                       # [-0.004, +0.002])
TOL_STRAIGHT_M = 0.5   # m: Eikonal's line off the chord at most this, a quarter pixel (Theta and
                       # Pulled: TOL_EXACT)
RC_BLD_W = 20          # building: a square building this many pixels wide between the ends
RC_BLDS = (0.05, 20.0, np.inf)   # building: through it cheaper, around it cheaper, the limit
TOL_AROUND = 0.02      # building: a router's path cost over the cheaper of the two, less 1 (the
                       # raster's corner: the way around hugs pixel centres, not the corner)
TOL_GAIN_EIK = 0.02    # gain: Eikonal's one-pass gain against the integral along its drawn line,
                       # relative: the largest on open ground, the median with buildings (where a
                       # line clipping a building's corner can part from the characteristic: up
                       # to 0.095 at bld 20, its T 7% over the line's cost)
RC_GAIN_BLDS = (0.0, 20.0, np.inf)   # gain: the buildings free, dear, and the limit
TOL_GAIN_VEC = 1e-6    # gain: Visibility's one-pass gain against `along` its line, relative
                       # (a piece along a pixel's edge counts the pixel `along` picks, to rounding)
RC_SEED = 0            # gain: the smooth gain field's and the buildings' draw
RC_BUMPS = 12          # gain: Gaussian bumps in the gain field
RC_TARGETS = 24        # gain: targets judged
RC_HOLE_PX = 10        # area: the hole's half-width (pixels) in the middle of the open raster
RC_AREA_FROM_PX = 12   # area: targets this many pixels or more from the source (straight line)
TOL_AREA = 1e-9        # area: an exact router's pieces against the split, over the face's area
TOL_AREA_EIK = 2e-3    # area: Eikonal's (transported) median, over the face's area
# a fault is a mutation of this file's routers (`_mutant`): (the text, its replacement)
MUTATIONS: dict[str, tuple[str, str]] = {
    "no-parent": ("                if p != s:\n", "                if False:\n"),
    "no-pull": ("            if take:\n", "            if False:\n"),
    "no-start": ("FMM_INIT_PX = 6 ", "FMM_INIT_PX = 0 "),
    "discrete-descent": ("        if np.isfinite(v) and g > 0.0:\n", "        if False:\n"),
    "endpoint-cost": ("            cost += c[q] * ds\n",
                      "            cost += 0.5 * (c[p0] + c[p1]) * ds\n"),
    "no-cut": ("keep = ~bnd[tg] | big ", "keep = ~bnd[tg] | np.isfinite(ln) "),
    "area-wrap": ("np.where(ar.k[a] < ar.k[tg], ar.tot[tg], 0.0)", "0.0 * ar.tot[tg]"),
    "area-from-s": ("                            bar = ar[p] + _sweep(p, s2, ny)",
                    "                            bar = ar[s] + _sweep(s, s2, ny)"),
    "area-node": ("        out[v] = (0.0 if p == -1 else out[p]) + _tri(",
                  "        out[v] = 0.0 + _tri("),
    "flat-field": ("_stencil(r, nx, ny, T, A, Lg, W, known, c[r], gp[r])",
                   "_stencil(r, nx, ny, T, A, Lg, W, known, 1.0, gp[r])"),
    "gain-from-s": ("                            bacc = acc[p] + gn\n",
                    "                            bacc = acc[s] + gn\n"),
    "vertex-sites": ("query_ball_point(sites(pts), r=gr.PATCH_R)",
                     "query_ball_point(pts, r=gr.PATCH_R)"),
    "gain-per-pixel": ("                bestA = (w0 * V[0] + w1 * V[1] + cr * gr_ / PIX) / ws\n",
                       "                bestA = (w0 * V[0] + w1 * V[1]) / ws + gr_\n"),
    "no-walls": ("                if d3 * d4 < 0.0:\n                    return False\n",
                 "                if False:\n                    return False\n"),
    "half-corners": ("    corner = np.flatnonzero(turn < -TAN_TOL * scale)",
                     "    corner = np.flatnonzero(turn < -TAN_TOL * scale)[::2]"),
    "gain-unit": ("    return acc * L / PIX\n", "    return acc * L\n"),
    "one-ring": ("    lab = np.where(net, comp[lab], -1)\n", "    lab = np.where(net, 0, -1)\n"),
    "share-radius": ("share = road_share(np.asarray(geo.polys), xy, r=gr.ROAD_W / 2.0)",
                     "share = road_share(np.asarray(geo.polys), xy, r=gr.ROAD_W)"),
    "share-raster": ("tri=tri, bld=bld, share=share,", "tri=tri, bld=bld, share=bld,"),
    "no-lex": ("                nb = B[v] + _qb(el * 0.5 * (sh[v] + sh[u]))\n",
               "                nb = B[v] + 0.0 * _qb(el * 0.5 * (sh[v] + sh[u]))\n"),
    "length-first": ("""                if nb < B[u] or (nb == B[u] and nl < L[u]):
                    B[u] = nb
                    L[u] = nl
                    pred[u] = v
                    heapq.heappush(heap, (nb, nl, u))""",
                     """                if nl < L[u] or (nl == L[u] and nb < B[u]):
                    B[u] = nb
                    L[u] = nl
                    pred[u] = v
                    heapq.heappush(heap, (nl, nb, u))"""),
}


def _mutant(fault: str) -> Any:
    """This module with the fault's mutation (MUTATIONS) applied once, loaded afresh: a guard's
    fault is its code broken, not its inputs."""
    import importlib.util
    import tempfile
    if fault not in MUTATIONS:
        raise SystemExit(f"a router check's fault: {'|'.join(MUTATIONS)}")
    old, new = MUTATIONS[fault]
    src = Path(__file__).read_text()
    a, b = src.index("MUTATIONS: dict["), src.index("def _mutant(")   # the table spells it too
    head, table, tail = src[:a], src[a:b], src[b:]
    if (head + tail).count(old) != 1:
        raise ValueError(f"{fault}: its text occurs {(head + tail).count(old)} times, not once")
    p = Path(tempfile.mkdtemp(prefix="ears_mutant_")) / f"ears_{fault.replace('-', '_')}.py"
    p.write_text(head.replace(old, new) + table + tail.replace(old, new))
    spec = importlib.util.spec_from_file_location(p.stem, p)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[p.stem] = m
    spec.loader.exec_module(m)
    return m


def open_terrain(nx: int, ny: int, *, share: F64, gain: F64, bld: float) -> Terrain:
    """A synthetic one-face terrain nx x ny pixels (the router checks), its corner at 0."""
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    xy = np.column_stack([(i.ravel() + 0.5) * PIX, (j.ravel() + 0.5) * PIX])
    ea, eb, el = _grid_edges((nx, ny))
    vec = FixedShapes(faces={0: shapely.box(0.0, 0.0, nx * PIX, ny * PIX)})
    return Terrain(shape=(nx, ny), lo=np.zeros(2), xy=xy, face=np.zeros(nx * ny, dtype=np.int64),
                   share=share, gain=gain, bld=bld, ea=ea, eb=eb, el=el,
                   goal=np.ones(nx * ny, dtype=bool), vec=vec)


def pieces(lo: F64, shape: tuple[int, int], line: F64) -> tuple[I64, F64]:
    """The polyline `line` cut by brute force, independently of `_seg` (the checks), wherever it
    crosses a cell edge of the PIX raster at lo (every x = lo + k PIX, y = lo + m PIX): per piece
    its cell and its length."""
    cells, lens = [], []
    for p, q in zip(line[:-1], line[1:], strict=True):
        d = q - p
        L = float(np.hypot(*d))
        ts = [0.0, 1.0]
        for k in range(2):
            if d[k] != 0:
                a, b = sorted((p[k] - lo[k], q[k] - lo[k]))
                for m in range(int(np.ceil(a / PIX)), int(np.floor(b / PIX)) + 1):
                    ts.append((m * PIX + lo[k] - p[k]) / d[k])
        t = np.unique(np.clip(ts, 0.0, 1.0))
        mid = p + ((t[:-1] + t[1:]) / 2)[:, None] * d
        cells.append(_pix_of_lo(lo, shape, mid))
        lens.append(np.diff(t) * L)
    return np.concatenate(cells), np.concatenate(lens)


def along(ter: Terrain, line: F64, w: F64) -> float:
    """The per-pixel field w integrated along the polyline `line` (`pieces`: each piece its
    cell's value times its length; the router checks)."""
    cells, lens = pieces(ter.lo, ter.shape, line)
    return float((w[cells] * lens).sum())


def _one(r: Router, ter: Terrain, a: int) -> Reach:
    [x] = list(r.reaches(ter, np.array([a], dtype=np.int64)))
    return x


def router_check(test: str, fault: str | None) -> None:
    """The routers on synthetic open rasters (`routers_at`: Grid8, Pulled, Theta, Eikonal but at
    binf, Visibility at b0), each judged against what it claims; a fault is a mutation of its
    code (MUTATIONS, `_mutant`), which fails it:
    cost: the cost of the cheapest path from the centre of an open RC_N raster to every pixel
      RC_NEAR_M or more away, and its drawn line's length, against the straight line: Grid8's cost
      up to OCTILE more (and reaching it); Pulled's line, Theta's cost and line exact; Eikonal's
      cost and line within TOL_EIK; Visibility's exact. Faults no-parent (Theta never tries its
      parent), no-pull, no-start (Eikonal from the source's pixel alone, no exact start) fail
      it.
    straight: targets RC_R_PX pixels off at bearings RC_ANGLES: the drawn line's largest distance
      from the chord (Grid8 reported, not judged; Pulled and Theta exact; Eikonal within
      TOL_STRAIGHT_M; Visibility exact). Faults no-parent, no-pull, discrete-descent (Eikonal's
      descent steps from pixel to pixel) fail it.
    building: a square of share 1, RC_BLD_W pixels wide, midway between the ends, at each bld of
      RC_BLDS: every router's path goes through it when that is cheaper and around it when that
      is, its cost (the field integrated along its line, `along`) within TOL_AROUND of the cheaper
      way (Grid8: OCTILE; Pulled: no more than the grid path it pulls); at binf its B (the share
      along its line) 0 and its length so against the way around. Faults endpoint-cost (a
      segment costed by its ends' mean, not integrated: Theta and Pulled), flat-field (Eikonal's
      stencil blind to the cost) fail it.
    gain: a smooth gain field (RC_BUMPS Gaussian bumps) over the open raster with buildings
      (at each bld of RC_GAIN_BLDS): each router's one-pass gain at RC_TARGETS targets against the
      gain integrated along its drawn path (Grid8 and Pulled: the gain of the grid path's pixels;
      Theta: `along`, exact; Eikonal: `along`, within TOL_GAIN_EIK; Visibility: `along`, within
      TOL_GAIN_VEC). Faults gain-from-s (Theta credits a segment from the parent on top of the
      child's gain), gain-per-pixel (Eikonal adds a pixel's gain once however its characteristic
      crosses it), gain-unit (Visibility's gain per metre, not per PIX) fail it.
    area: the open raster less a square hole in its middle (RC_HOLE_PX; off the face and a hole
      in Visibility's polygon), the gain test's buildings at each bld of RC_GAIN_BLDS, its sinks
      the pixels beside the outside or the hole (`arcs_of`): from a source on the outer ring and
      one on the hole's, to RC_TARGETS of the same ring's sinks RC_AREA_FROM_PX or more away,
      the two pieces (`cuts`: each router's one-pass swept area, Reach.area, and the contours)
      against the face's contour polygon split by the drawn line (its ends joined to their
      contour vertices; Grid8 and Pulled: the grid path; from the outer ring the polygon with
      its hole filled: `cuts` counts the hole in the piece it lies in), a split into other than
      two pieces not judged: exact (TOL_AREA of the face's area) but Eikonal's (transported:
      its median within TOL_AREA_EIK). Faults area-wrap (the contour's own area never added
      where the arc wraps past its first vertex), area-from-s (Theta's segment from the parent
      swept from s), area-node (Visibility's node areas not summed down its tree) fail it."""
    if fault is not None:
        print(f"router {test}: mutant {fault}")
        _mutant(fault).router_check(test, None)
        return
    n = RC_N
    c0 = (n // 2) * n + n // 2
    zero = np.zeros(n * n)
    ok = True
    if test == "cost":
        ter = open_terrain(n, n, share=zero, gain=zero, bld=0.0)
        eu = np.hypot(*(ter.xy - ter.xy[c0]).T)
        far = eu >= RC_NEAR_M
        ang = np.radians(np.arange(0.0, 90.0, 2.5))
        tg = ((n // 2 + np.round(RC_R_PX * np.cos(ang))) * n
              + n // 2 + np.round(RC_R_PX * np.sin(ang))).astype(np.int64)
        for r in routers_at(0.0):
            x = _one(r, ter, c0)
            cost = x.cost[far] / eu[far] - 1
            line = np.array([np.hypot(*np.diff(x.route(int(t)).line, axis=0).T).sum() / eu[t] - 1
                             for t in tg])
            match r:
                case Grid8():
                    good = cost.max() <= OCTILE + TOL_EXACT and cost.max() >= OCTILE - 2e-3
                case Pulled():
                    good = abs(line).max() <= TOL_EXACT
                case Theta() | Visibility():
                    good = max(abs(cost).max(), abs(line).max()) <= TOL_EXACT
                case _:
                    good = max(abs(cost).max(), abs(line).max()) <= TOL_EIK
            ok &= bool(good)
            print(f"  {r.name}: cost / straight line - 1 in [{cost.min():+.2e}, {cost.max():+.2e}]"
                  f", drawn line's length in [{line.min():+.2e}, {line.max():+.2e}] -> "
                  f"{'ok' if good else 'BAD'}")
    elif test == "straight":
        ter = open_terrain(n, n, share=zero, gain=zero, bld=0.0)
        a = 10 * n + 10
        for r in routers_at(0.0):
            x = _one(r, ter, a)
            worst = 0.0
            for deg in RC_ANGLES:
                t = int((10 + round(RC_R_PX * np.cos(np.radians(deg)))) * n
                        + 10 + round(RC_R_PX * np.sin(np.radians(deg))))
                line = x.route(t).line
                d = line[-1] - line[0]
                rel = line - line[0]
                off = np.abs(d[0] * rel[:, 1] - d[1] * rel[:, 0]) / np.hypot(*d)
                worst = max(worst, float(off.max()))
            match r:
                case Grid8():
                    tol = np.inf
                case Pulled() | Theta() | Visibility():
                    tol = TOL_EXACT * RC_R_PX * PIX
                case _:
                    tol = TOL_STRAIGHT_M
            good = worst <= tol
            ok &= good
            print(f"  {r.name}: the line at most {worst:.3g} m off its chord "
                  f"({'not judged' if tol == np.inf else 'ok' if good else 'BAD'})")
    elif test == "building":
        nx, ny = 121, 61
        a, t = 10 * ny + 30, 110 * ny + 30
        i, j = np.divmod(np.arange(nx * ny), ny)
        w = RC_BLD_W // 2
        share = ((abs(i - 60) <= w) & (abs(j - 30) <= w)).astype(np.float64)
        side = PIX * (2 * w + 1)
        D = PIX * 100
        for bld in RC_BLDS:
            through = D + bld * side
            around = 2 * float(np.hypot((D - side) / 2, side / 2)) + side
            ter = open_terrain(nx, ny, share=share, gain=np.zeros(nx * ny), bld=bld)
            best = min(through, around)
            for r in routers_at(bld):
                x = _one(r, ter, a)
                line = x.route(t).line
                blk = along(ter, line, ter.share)
                cost = (float(np.hypot(*np.diff(line, axis=0).T).sum()) if ter.lex() else
                        along(ter, line, ter.per_m()))
                went = "around" if np.abs(line[:, 1] - line[0, 1]).max() > side / 4 else "through"
                want = "around" if around < through else "through"
                match r:
                    case Pulled():                # no dearer than the grid path it pulls
                        grid = x.inner.route(t).line
                        bar = (float(np.hypot(*np.diff(grid, axis=0).T).sum()) if ter.lex() else
                               along(ter, grid, ter.per_m())) * (1 + TOL_EXACT)
                    case Grid8():
                        bar = best * (1 + OCTILE)
                    case _:
                        bar = best * (1 + TOL_AROUND)
                good = went == want and cost <= bar and (not ter.lex() or blk <= TOL_EXACT * D)
                ok &= good
                print(f"  bld {bld:g}, {r.name}: through {through:.1f}, around {around:.1f}; went "
                      f"{went}, its line {'long' if ter.lex() else 'costs'} {cost:.1f}, B "
                      f"{blk:.2f} -> {'ok' if good else 'BAD'}")
    elif test == "gain":
        rng = np.random.default_rng(RC_SEED)
        i, j = np.divmod(np.arange(n * n), n)
        gp = np.zeros(n * n)
        for _ in range(RC_BUMPS):
            ci, cj, s = rng.uniform(0, n), rng.uniform(0, n), rng.uniform(6, 20)
            gp += rng.uniform(0.5, 2.0) * np.exp(-((i - ci) ** 2 + (j - cj) ** 2) / (2 * s * s))
        share = np.zeros(n * n)
        for _ in range(30):
            ci, cj = rng.integers(0, n, size=2)
            share[(abs(i - ci) <= 3) & (abs(j - cj) <= 3)] = 1.0
        share[c0] = 0.0
        ang = rng.uniform(0, 2 * np.pi, RC_TARGETS)
        tg = ((n // 2 + np.round(RC_R_PX * np.cos(ang))) * n
              + n // 2 + np.round(RC_R_PX * np.sin(ang))).astype(np.int64)
        for bld in RC_GAIN_BLDS:
            ter = open_terrain(n, n, share=share, gain=gp, bld=bld)
            for r in routers_at(bld):
                x = _one(r, ter, c0)
                rel = []
                for t in tg:
                    match x:
                        case _PulledReach():      # its gain is its grid path's (the choice)
                            brute = float(gp[x.inner.route(int(t)).path].sum())
                        case _TreeReach():
                            brute = float(gp[x.route(int(t)).path].sum())
                        case _:
                            brute = along(ter, x.route(int(t)).line, gp / PIX)
                    rel.append(x.gain[t] / brute - 1)
                err = np.abs(rel)
                if isinstance(r, Visibility):
                    judged, tol = float(err.max()), TOL_GAIN_VEC
                elif not isinstance(r, Eikonal):
                    judged, tol = float(err.max()), TOL_EXACT
                elif bld == 0:
                    judged, tol = float(err.max()), TOL_GAIN_EIK
                else:
                    judged, tol = float(np.median(err)), TOL_GAIN_EIK
                good = judged <= tol
                ok &= good
                print(f"  bld {bld:g}, {r.name}: |one-pass gain / along its path - 1| median "
                      f"{np.median(err):.2e}, at most {err.max():.2e} (judged "
                      f"{'the median' if isinstance(r, Eikonal) and bld else 'the largest'}, tol "
                      f"{tol:g}) -> {'ok' if good else 'BAD'}")
    elif test == "area":
        rng = np.random.default_rng(RC_SEED)
        i, j = np.divmod(np.arange(n * n), n)
        hole = (abs(i - n // 2) <= RC_HOLE_PX) & (abs(j - n // 2) <= RC_HOLE_PX)
        share = np.zeros(n * n)
        for _ in range(30):
            ci, cj = rng.integers(0, n, size=2)
            share[(abs(i - ci) <= 3) & (abs(j - cj) <= 3)] = 1.0
        pf = np.where(hole, -1, 0).reshape(n, n)
        off = np.pad(pf < 0, 1, constant_values=True)
        sink = (ndimage.binary_dilation(off, structure=np.ones((3, 3), dtype=bool))[1:-1, 1:-1]
                & (pf >= 0)).ravel()
        pfp = np.pad(pf, 1, constant_values=-1)
        corner = np.full(saddles(pfp).shape, -1, dtype=np.int64)     # no saddle here
        arcs = arcs_of(pf, sink, corner)
        share[sink] = 0.0
        rings_ = sorted(face_loops(pfp, corner, 0), key=lambda c: -len(c))
        poly = shapely.Polygon(rings_[0], [rings_[1]])
        h0, h1 = n // 2 - RC_HOLE_PX - 1, n // 2
        srcs = (n // 2, h0 * n + h1)                 # on the outer ring, on the hole's
        face = np.where(hole, -1, 0)
        dom = shapely.box(0.0, 0.0, n * PIX, n * PIX).difference(shapely.box(
            (n // 2 - RC_HOLE_PX) * PIX, (n // 2 - RC_HOLE_PX) * PIX,
            (n // 2 + RC_HOLE_PX + 1) * PIX, (n // 2 + RC_HOLE_PX + 1) * PIX))
        for bld in RC_GAIN_BLDS:
            ter = open_terrain(n, n, share=share, gain=zero, bld=bld)._replace(
                face=face, goal=sink, vec=FixedShapes(faces={0: dom}))
            for r in routers_at(bld):
                err, skipped = [], 0
                for a in srcs:
                    x = _one(r, ter, a)
                    on = np.flatnonzero(sink & (arcs.ring == arcs.ring[a])
                                        & (np.hypot(*(ter.xy - ter.xy[a]).T)
                                           >= RC_AREA_FROM_PX * PIX) & np.isfinite(x.cost))
                    for t in on[np.linspace(0, len(on) - 1, RC_TARGETS).astype(np.int64)]:
                        X, Y = cuts(arcs, 0, a, np.array([t]), x.area[[t]])
                        line = (x.inner.route(int(t)).line if isinstance(x, _PulledReach)
                                else x.route(int(t)).line)
                        cut = np.vstack([[arcs.vx[a], arcs.vy[a]], line,
                                         [arcs.vx[t], arcs.vy[t]]])
                        dom_ = shapely.Polygon(poly.exterior) if a == srcs[0] else poly
                        got = shapely.get_parts(shapely.ops.split(dom_,
                                                                  shapely.LineString(cut)))
                        if len(got) != 2 or not np.isfinite(X[0]):
                            skipped += 1
                            continue
                        want = np.sort(shapely.area(got))
                        err.append(float(np.abs(np.sort([X[0], Y[0]]) - want).max())
                                   / poly.area)
                err_ = np.asarray(err)
                if isinstance(r, Eikonal):
                    judged, tol = float(np.median(err_)), TOL_AREA_EIK
                else:
                    judged, tol = float(err_.max()), TOL_AREA
                good = judged <= tol and len(err_) >= RC_TARGETS
                ok &= good
                print(f"  bld {bld:g}, {r.name}: |pieces - the split's| / the face's area median "
                      f"{np.median(err_):.2e}, at most {err_.max():.2e} over {len(err_)} "
                      f"targets ({skipped} not split in two, not judged; judged "
                      f"{'the median' if isinstance(r, Eikonal) else 'the largest'}, tol "
                      f"{tol:g}) -> {'ok' if good else 'BAD'}")
    else:
        raise SystemExit(__doc__)
    print(f"router {test}: -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


def names(fault: str | None) -> None:
    """Every check plan's name round-trips, every router's too (each check paths plan with its
    router swapped in), and each paths name with its attraction token or its router token dropped
    is refused (the tokens are required: a name that omits one means nothing; the router's is how
    a pre-2026-11-19 name, which had none, is refused). Fault `keep` (the tokens kept) fails
    it."""
    if fault not in (None, "keep"):
        raise SystemExit("names' fault: keep")
    ok = True
    specs = [CHECK_PLAN, CHECK_SPUR, CHECK_DETOUR, CHECK_DIVIDES]
    specs += [CHECK_PLAN.replace(".grid8.b20.", f".{r.name}.b{b:g}.", 1)
              for b in (0.0, 20.0, np.inf) for r in routers_at(b)
              if (r.name, b) != ("grid8", 20.0)]
    for spec in specs:
        p = Plan.of(spec)
        back = p.name == f"{spec}.v{REV}"
        ok &= back
        print(f"  {spec}: {'round-trips' if back else f'named {p.name}'}")
        if isinstance(p.candidates, Paths):
            for tok in (f".{p.candidates.attract.name}.", f".{p.candidates.router.name}."):
                bare = spec if fault == "keep" else spec.replace(tok, ".", 1)
                try:
                    Plan.of(bare)
                    refused = False
                except SystemExit:
                    refused = True
                ok &= refused
                print(f"    {bare}: {'refused' if refused else 'PARSES'}")
    for tok, b in (("visibility", "0.05"), ("visibility", "20"), ("visibility", "inf"),
                   ("eikonal", "inf")):         # Visibility exact at b0 only; Eikonal not at binf
        bare = CHECK_PLAN.replace(".grid8.b20.", f".{'grid8' if fault == 'keep' else tok}"
                                  f".b{b}.", 1)
        try:
            Plan.of(bare)
            refused = False
        except ValueError:
            refused = True
        ok &= refused
        print(f"    {bare}: {'refused' if refused else 'PARSES'}")
    print(f"names {fault or 'none'}: -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


VEC_SOURCES = 12       # vector: sources judged (evenly through the state's)
TOL_VEC_LINE_M = 1e-2  # m: vector: a Visibility line's length outside its free space (the line's
                       # vertices on walls, the domain snapped to GRID_M)
TOL_VEC_P1 = 0.05      # vector: its p1 at least minus this (5618: theta -0.014, Eikonal -0.011; a
                       # Visibility missing corners, half-corners, -0.69)
TOL_VEC_MED = 2e-3     # vector: a raster router's median cost over Visibility's, less 1, at least
                       # minus this (its face is a pixel off the vector one, so a pair can
                       # undercut by a per cent: 5618 theta p1 -0.010; its median not: +0.0000)


def vector_check(bid: str, fault: str | None) -> None:
    """`vector`: Visibility against the raster routers, on the block's first state at b0,
    VEC_SOURCES sources to every goal pixel ear_m or more away:
    (1) each Visibility line (its pixel stubs aside) lies in its free space, to TOL_VEC_LINE_M;
    (2) each raster router's cost over Visibility's, less 1, has its median at least -TOL_VEC_MED
      and its p1 at least -TOL_VEC_P1 (Visibility is the optimum on the vector face: a raster
      router undercuts a pair only by its pixels; its least reported).
    Faults (mutations: MUTATIONS) no-walls (1), half-corners (2) fail it."""
    if fault is not None:
        print(f"vector {bid}: mutant {fault}")
        _mutant(fault).vector_check(bid, None)
        return
    e = world_of(bid, 0.25)
    g = e.g
    st = g.state(g.step >= 0)
    u, _ = g.solve(st)
    gain = g.contrast.gains(g, st, u)
    fc = e.faces()
    ok = True
    for b in (0.0,):
        pa = Paths(router=Grid8(), bld=b, pairs=8, face_m2=2000.0, ear_m=30.0, ends=Loop(),
                   attract=NoDetour())
        ter, src = pa.routing(e, fc, gain)[:2]
        src = src[np.linspace(0, len(src) - 1, VEC_SOURCES).astype(np.int64)]
        vis = list(Visibility(bld=b).reaches(ter, src))
        far = [ter.goal & np.isfinite(x.cost) & (np.hypot(*(ter.xy - ter.xy[x.src]).T)
                                                  >= pa.ear_m) for x in vis]
        doms = {f: free_of(ter.vec, f) for f in np.unique(ter.face[src])}
        out = 0.0
        for x, k in zip(vis, far, strict=True):
            dom = shapely.buffer(doms[int(ter.face[x.src])], TOL_VEC_LINE_M / 10)
            for t in np.flatnonzero(k)[::7]:
                line = x.route(int(t)).line
                inner = line[1:-1] if len(line) > 3 else line[1:-1]
                if len(inner) > 1:
                    out = max(out, float(shapely.difference(shapely.LineString(inner), dom).length))
        good = out <= TOL_VEC_LINE_M
        ok &= good
        print(f"  b{b:g}: Visibility's lines at most {out:.2e} m outside the free space -> "
              f"{'ok' if good else 'BAD'}")
        vc = np.concatenate([x.cost[k] for x, k in zip(vis, far, strict=True)])
        for r in (Grid8(), Pulled(inner=Grid8()), Theta(), Eikonal(bld=b)):
            rc = np.concatenate([y.cost[k] for y, k in zip(r.reaches(ter, src), far, strict=True)])
            rel = rc / vc - 1
            rel = rel[np.isfinite(rel)]
            good = bool(np.median(rel) >= -TOL_VEC_MED and np.percentile(rel, 1) >= -TOL_VEC_P1)
            ok &= good
            print(f"  b{b:g} {r.name}: cost / Visibility's - 1: median {np.median(rel):+.4f}, p1 "
                  f"{np.percentile(rel, 1):+.4f}, least {rel.min():+.4f}, most {rel.max():+.4f} ("
                  f"{len(rel)} pairs) -> {'ok' if good else 'BAD'}")
    print(f"vector {bid} {fault or 'none'}: -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


SHARE_SAMPLE = 400     # share: pixels near a footprint judged against the independent share
SHARE_CHECK_SEGS = 256  # share: the independent disc a polygon of 4 x this sides, its cover over
                       # the circle's area (pi r^2): 2.5e-5 under it
TOL_SHARE = 1e-3       # share: |road_share - the independent one| (the polygon's 4.0e-4 deficit
                       # cancels in road_share's own ratio but not against the circle)
SHARE_SEED = 0         # share: the sample's draw


def share_check(bid: str, fault: str | None) -> None:
    """`share`: the raster's road share (Raster.share, `road_share`) on the block (module
    docstring). Faults share-radius, share-raster (MUTATIONS) fail it."""
    if fault is not None:
        print(f"share {bid}: mutant {fault}")
        _mutant(fault).share_check(bid, None)
        return
    e = world_of(bid, 0.25)
    ra = e.ra
    R = gr.ROAD_W / 2.0
    U = shapely.union_all(np.asarray(e.g.m.geo.polys))
    shapely.prepare(U)
    pts = shapely.points(ra.xy)
    dist = shapely.distance(U, pts)
    far = dist >= R
    inside = np.flatnonzero(dist == 0)
    deep = inside[shapely.distance(shapely.boundary(U), pts[inside]) >= R]
    near = np.flatnonzero(dist < R + PIX)
    rng = np.random.default_rng(SHARE_SEED)
    idx = rng.choice(near, size=min(SHARE_SAMPLE, len(near)), replace=False)
    exact = np.array([shapely.area(shapely.intersection(
        shapely.Point(*ra.xy[k]).buffer(R, quad_segs=SHARE_CHECK_SEGS), U)) for k in idx])
    exact /= np.pi * R * R
    err = np.abs(ra.share[idx] - exact)
    ok = True
    good = bool(err.max() <= TOL_SHARE)
    ok &= good
    print(f"  {len(idx)} pixels within {R + PIX:g} m of a footprint (share in [{exact.min():.3f}, "
          f"{exact.max():.3f}], mean {exact.mean():.3f}): |share - exact| median "
          f"{np.median(err):.2e}, at most {err.max():.2e} (tol {TOL_SHARE:g}) -> "
          f"{'ok' if good else 'BAD'}; the 0/1 raster's at most "
          f"{np.abs(ra.bld[idx] - exact).max():.2f}")
    good = bool(far.any() and ra.share[far].max() == 0.0)
    ok &= good
    print(f"  {int(far.sum())} pixels {R:g} m or more from every footprint: share at most "
          f"{ra.share[far].max():.2e} -> {'ok' if good else 'BAD'}")
    good = bool(len(deep) and ra.share[deep].min() >= 1.0 - TOL_EXACT)
    ok &= good
    print(f"  {len(deep)} pixels {R:g} m or more inside a footprint: share at least "
          f"{ra.share[deep].min() if len(deep) else np.nan:.12f} -> {'ok' if good else 'BAD'}")
    print(f"share {bid} {fault or 'none'}: -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


RINGS_PLAN = "ears.paths.theta.b0.f2000.e30.loop.nodt.gl0.25Croad.n2s0.1.M8.K4.D1"
RINGS_STEPS = 3        # rings: ears built (rev 3 stopped after 1 on 5810)


class _Offer(NamedTuple):
    """An offered candidate (_Offers): the rings of its path's two ends, at the state it was
    offered in."""
    ring0: int
    ring1: int


@dataclass(kw_only=True)
class _Offers:
    """Paths whose offers are noted (the rings check)."""
    inner: Paths
    offers: list[_Offer]

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def pairs(self) -> int:
        return self.inner.pairs

    def ears(self, e: EarWorld, faces: Face, gain: F64, *, cost: gr.Cost,
             scale: float) -> list[Ear]:
        out = self.inner.ears(e, faces, gain, cost=cost, scale=scale)
        ring = e.arcs(faces).ring
        self.offers += [_Offer(ring0=int(ring[q.path[0]]), ring1=int(ring[q.path[-1]]))
                        for q in out]
        return out


def rings_check(bid: str, fault: str | None) -> None:
    """`rings` (module docstring). Fault one-ring (MUTATIONS) fails it."""
    if fault is not None:
        print(f"rings {bid}: mutant {fault}")
        _mutant(fault).rings_check(bid, None)
        return
    plan = Plan.of(RINGS_PLAN)
    assert isinstance(plan.candidates, Paths)
    e = world_of(bid, plan.contrast.h)
    ring = e.arcs(e.faces()).ring
    n0 = len(np.unique(ring[ring >= 0]))
    rec = _Offers(inner=plan.candidates, offers=[])
    out = gr.OUT / bid / f"check.{plan.name}"
    build(e, replace(plan, candidates=rec), out=out, log=lambda s: print(s, flush=True),
          steps=RINGS_STEPS)
    built = len(pd.read_parquet(out / "picks.parquet"))
    cross = sum(o.ring0 != o.ring1 for o in rec.offers)
    ok = n0 >= 2 and built == RINGS_STEPS and cross == 0
    print(f"rings {bid} {fault or 'none'}: {n0} rings at the first state; {built} ears built of "
          f"{RINGS_STEPS}; {cross} of {len(rec.offers)} offered candidates end on another ring "
          f"than their source's -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


BINF_PLANS = tuple(f"ears.paths.{r}.binf.f2000.e30.loop.nodt.gl0.25Croad.n2s0.1.M8.K4.D1"
                   for r in ("grid8", "theta"))
BINF_STEPS = 10        # binf: ears built (before the cut test: 1 for grid8, 2 for theta on 5618)


def binf_check(bid: str, fault: str | None) -> None:
    """`binf` (module docstring). Fault no-cut (MUTATIONS) fails it."""
    if fault is not None:
        print(f"binf {bid}: mutant {fault}")
        _mutant(fault).binf_check(bid, None)
        return
    ok = True
    for spec in BINF_PLANS:
        plan = Plan.of(spec)
        assert isinstance(plan.candidates, Paths)
        e = world_of(bid, plan.contrast.h)
        out = gr.OUT / bid / f"check.{plan.name}"
        lines: list[str] = []
        build(e, plan, out=out, log=lambda s, lines=lines: (lines.append(s), print(s, flush=True)),
              steps=BINF_STEPS)
        built = len(pd.read_parquet(out / "picks.parquet"))
        refused = sum("refused" in s for s in lines)
        good = built == BINF_STEPS
        ok &= good
        print(f"  {plan.candidates.router.name}: {built} ears built of {BINF_STEPS} ({refused} "
              f"offered ears refused on the way) -> {'ok' if good else 'BAD'}")
    print(f"binf {bid}: -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


LIMIT_SOURCES = 8      # limit: sources (evenly through the state's)
LIMIT_TARGETS = 10     # limit: goal pixels per source, ear_m or more away (drawn)
LIMIT_BLDS = (0.0, 5.0, 20.0, 1e3, 1e4)   # limit: the finite b compared with binf
TOL_LIMIT = 1e-6       # limit: relative to the longer path (B and L to rounding)
LIMIT_SEED = 0         # limit: the targets' draw


def limit_check(bid: str, fault: str | None) -> None:
    """`limit`: Grid8's and Theta's paths along b to binf (module docstring). Faults no-lex,
    length-first (MUTATIONS) fail it."""
    if fault is not None:
        print(f"limit {bid}: mutant {fault}")
        _mutant(fault).limit_check(bid, None)
        return
    e = world_of(bid, 0.25)
    g = e.g
    st = g.state(g.step >= 0)
    u, _ = g.solve(st)
    gain = g.contrast.gains(g, st, u)
    fc = e.faces()
    bs = (*LIMIT_BLDS, np.inf)
    ok = True
    for router in (Grid8(), Theta()):
        BL = []                                   # per b: (B, L) per pair
        pairs: list[tuple[int, int]] = []
        for b in bs:
            pa = Paths(router=router, bld=b, pairs=8, face_m2=2000.0, ear_m=30.0, ends=Loop(),
                       attract=NoDetour())
            ter, src, pf = pa.routing(e, fc, gain)[:3]
            if not pairs:
                rng = np.random.default_rng(LIMIT_SEED)
                for a in src[np.linspace(0, len(src) - 1, LIMIT_SOURCES).astype(np.int64)]:
                    ok_t = np.flatnonzero(ter.goal & (pf == pf[a]) & (np.hypot(
                        *(ter.xy - ter.xy[a]).T) >= pa.ear_m))
                    pairs += [(int(a), int(t)) for t in rng.choice(
                        ok_t, size=min(LIMIT_TARGETS, len(ok_t)), replace=False)]
            srcs = np.unique([a for a, _ in pairs])
            reach = {x.src: x for x in router.reaches(ter, srcs)}
            row = []
            for a, t in pairs:
                line = reach[a].route(t).line
                row.append((along(ter, line, ter.share),
                            float(np.hypot(*np.diff(line, axis=0).T).sum())))
            BL.append(np.asarray(row))
        V = np.stack(BL)                          # (b, pair, (B, L))
        tol = TOL_LIMIT * V[:, :, 1].max(axis=0)
        dB = V[1:, :, 0] - V[:-1, :, 0]           # along b: B should not rise, L not fall
        dL = V[:-1, :, 1] - V[1:, :, 1]
        mono = (dB > tol).sum() + (dL > tol).sum()
        Bi, Li = V[-1, :, 0], V[-1, :, 1]
        under = (V[:-1, :, 0] < Bi - tol).sum()   # a finite b's B under binf's
        fb = np.asarray(LIMIT_BLDS)[1:, None]     # B(b) - B(inf) <= (L(inf) - L(b)) / b
        slack = (V[1:-1, :, 0] - Bi) - (Li - V[1:-1, :, 1]) / fb
        over = (slack > tol).sum()
        same = [(np.abs(V[k, :, 0] - Bi) <= tol) & (np.abs(V[k, :, 1] - Li) <= tol)
                for k in (len(bs) - 3, len(bs) - 2)]
        bad = int(mono + under + over)
        judged = isinstance(router, Grid8)
        good = bad == 0 or not judged
        ok &= good
        print(f"  {router.name}: {len(pairs)} pairs; B mean by b "
              + ", ".join(f"b{b:g} {V[k, :, 0].mean():.2f}" for k, b in enumerate(bs))
              + "; L mean " + ", ".join(f"{V[k, :, 1].mean():.1f}" for k in range(len(bs))))
        print(f"    monotone in b: {int(mono)} violations (B rising {int((dB > tol).sum())}, "
              f"L falling {int((dL > tol).sum())}, largest {max(dB.max(), dL.max()):+.3g} m); "
              f"a finite b's B under binf's {int(under)}; binf's path cheaper at b than b's own "
              f"{int(over)}; binf's path = b1e3's {int(same[0].sum())}/{len(pairs)}, b1e4's "
              f"{int(same[1].sum())}/{len(pairs)}, either {int((same[0] | same[1]).sum())} -> "
              f"{('ok' if good else 'BAD') if judged else 'reported, not judged'}")
    print(f"limit {bid} {fault or 'none'}: -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


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
        cands = Paths(router=Grid8(), bld=20.0, pairs=6, face_m2=2000.0, ear_m=30.0, ends=Loop(),
                      attract=NoDetour()).ears(
            e, fc, gain, cost=gr.Cost(nu=2.0, price=0.1), scale=0.0)
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
        if fault == "vertex-sites":
            print(f"ends {bid}: mutant {fault}")
            _mutant(fault).check(verb, bid, None)
            return
        if fault not in (None, "spur"):
            raise SystemExit("ends' fault: spur|vertex-sites")
        e = _short(bid, 0.25, 0, spec=CHECK_PLAN)
        g = e.g
        st = g.state(g.step >= 0)
        u, _ = g.solve(st)
        fc = e.faces()
        shp = MeshShapes(xy=g.m.fx.fm.xy, tri=g.m.fx.fm.tri, lab=fc.lab)
        ok = True
        n_judged = 0
        for b in (0.0, 20.0, np.inf):
            for r in routers_at(b):
                cands = Paths(router=r, bld=b, pairs=4, face_m2=2000.0, ear_m=30.0, ends=Loop(),
                              attract=NoDetour()).ears(
                    e, fc, g.contrast.gains(g, st, u), cost=gr.Cost(nu=2.0, price=0.1), scale=0.0)
                ok &= bool(cands)
                for q in cands[:4]:
                    ideal = shapely.difference(shp.face(q.face), shapely.buffer(
                        shapely.LineString(q.line), gr.PATCH_R))
                    if (shapely.area(shapely.get_parts(ideal)) >= ENDS_M2).sum() < 2:
                        print(f"  b{b:g} {r.name}: ear of {len(q.line)} points: its ideal "
                              "corridor separates nothing (not judged)")
                        continue
                    n_judged += 1
                    line = q.line
                    if fault == "spur":           # its last third dropped
                        ls = shapely.LineString(line)
                        line = np.asarray(shapely.get_coordinates(
                            shapely.ops.substring(ls, 0.0, 2 * ls.length / 3)))
                    tris = e.corridor(line, spur=False)
                    n0 = int((fc.area >= ENDS_M2).sum())
                    g.step[tris] = 0
                    n1 = int((e.faces().area >= ENDS_M2).sum())
                    g.step[tris] = -1
                    split = n1 > n0
                    ok &= split
                    print(f"  b{b:g} {r.name}: ear of {len(line)} points: faces {n0} -> {n1}: "
                          f"{'splits' if split else 'a dead end'}")
        ok &= n_judged > 0
        print(f"ends {bid} {fault or 'none'}: {n_judged} ears judged -> {'PASS' if ok else 'FAIL'}")
    elif verb == "spur":
        if fault not in (None, "both-ends"):
            raise SystemExit("spur's fault: both-ends")
        plan = Plan.of(CHECK_SPUR)
        e = _short(bid, plan.contrast.h, 0, spec=CHECK_SPUR)
        g = e.g
        st = g.state(g.step >= 0)
        u, _ = g.solve(st)
        fc = e.faces()
        spurs = [q for q in plan.candidates.ears(e, fc, g.contrast.gains(g, st, u),
                                                 cost=plan.cost, scale=0.0) if q.spur]
        bounding = e.bounding()
        ok = bool(spurs)
        for q in spurs[:SPUR_EARS]:
            tris = e.corridor(q.line, spur=fault != "both-ends")
            on = np.zeros(g.nt, dtype=bool)
            on[tris] = True
            inner = on[g.pa] & on[g.pb]
            pieces = connected_components(sp.coo_matrix(
                (np.ones(int(inner.sum())), (g.pa[inner], g.pb[inner])), shape=(g.nt, g.nt)),
                directed=False)[1][tris]
            joined = bool(((on[g.pa] & bounding[g.pb]) | (on[g.pb] & bounding[g.pa])).any())
            g.step[tris] = 0
            n1 = e.faces().n
            g.step[tris] = -1
            good = len(np.unique(pieces)) == 1 and joined and n1 == fc.n
            ok &= good
            L = float(np.hypot(*np.diff(q.line, axis=0).T).sum())
            print(f"  spur of {L:.0f} m: {len(np.unique(pieces))} piece(s), "
                  f"{'joined' if joined else 'detached'}, faces {fc.n} -> {n1}")
        print(f"spur {bid} {fault or 'none'}: {len(spurs)} spurs among the candidates -> "
              f"{'PASS' if ok else 'FAIL'}")
    elif verb == "detour":
        if fault not in (None, "one-way"):
            raise SystemExit("detour's fault: one-way")
        plan = Plan.of(CHECK_DETOUR)
        pa = plan.candidates
        assert isinstance(pa, Paths)
        e = _short(bid, plan.contrast.h, DETOUR_EARS, spec=CHECK_PLAN)
        g, ra = e.g, e.ra
        st = g.state(g.step >= 0)
        u, _ = g.solve(st)
        fld = pa.attract.field(e, P0=g.P0)
        assert isinstance(fld, _DetourField)
        rd = fld.roads
        net = e.sinks()
        col, G = _net_graph(net, ra.ea, ra.eb, ra.el)
        DRR = rd.among()
        fin = np.isfinite(DRR)
        ww = np.outer(rd.w, rd.w)

        def brute(a: int, t: int, L: float) -> float:
            """All pairs again, the link an edge of the network's graph."""
            link = sp.coo_matrix(([L], ([col[a]], [col[t]])), shape=G.shape).tocsr()
            D2 = dijkstra(G + link, directed=False, indices=col[rd.pts])[:, col[rd.pts]]
            return float((ww * np.where(fin, DRR - D2, 0.0)).sum())

        half = 0.5 if fault == "one-way" else 1.0  # the trips counted from a's side only
        cands = [q for q in pa.ears(e, e.faces(), g.contrast.gains(g, st, u), cost=plan.cost,
                                    scale=0.0) if not q.spur]
        judged: list[tuple[str, float, float]] = []
        for q in cands:
            L = float(np.hypot(*np.diff(q.line, axis=0).T).sum())
            judged.append((f"ear of {L:.0f} m (credit {q.credit / g.P0:.3g} P0)", half * q.saving,
                           brute(int(q.path[0]), int(q.path[-1]), L)))
        a0 = int(cands[0].path[0])
        on = np.flatnonzero(net & np.isfinite(rd.D[0, np.maximum(col, 0)]) & (col >= 0))
        tg = np.random.default_rng(DETOUR_SEED).choice(on, size=DETOUR_TARGETS, replace=False)
        Ls = DETOUR_STRETCH * np.hypot(*(ra.xy[tg] - ra.xy[a0]).T) + PIX
        fast = savings(rd, a0, tg, Ls)
        for t, L, f in zip(tg, Ls, fast, strict=True):
            judged.append((f"link of {L:.0f} m from the first ear's source", half * float(f),
                           brute(a0, int(t), float(L))))
        top = max(b for _, _, b in judged)
        worst = max(abs(f - b) for _, f, b in judged) / top
        for what, f, b in judged:
            print(f"  {what}: S {f:.4f} m, all pairs again {b:.4f} m")
        ok = top > 0 and worst <= TOL_DETOUR
        print(f"detour {bid} {fault or 'none'}: {len(rd.pts)} road points, worst |S - all pairs| "
              f"{worst:.2e} of the largest S (tol {TOL_DETOUR:g}) -> {'PASS' if ok else 'FAIL'}")
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
    elif len(argv) in (2, 3) and argv[:2] == ["check", "horseshoe"]:
        horseshoe(argv[2] if len(argv) == 3 else None)
    elif len(argv) in (2, 3) and argv[:2] == ["check", "names"]:
        names(argv[2] if len(argv) == 3 else None)
    elif len(argv) in (3, 4) and argv[:2] == ["check", "vector"]:
        vector_check(argv[2], argv[3] if len(argv) == 4 else None)
    elif len(argv) in (3, 4) and argv[:2] == ["check", "share"]:
        share_check(argv[2], argv[3] if len(argv) == 4 else None)
    elif len(argv) in (3, 4) and argv[:2] == ["check", "rings"]:
        rings_check(argv[2], argv[3] if len(argv) == 4 else None)
    elif len(argv) in (3, 4) and argv[:2] == ["check", "binf"]:
        binf_check(argv[2], argv[3] if len(argv) == 4 else None)
    elif len(argv) in (3, 4) and argv[:2] == ["check", "limit"]:
        limit_check(argv[2], argv[3] if len(argv) == 4 else None)
    elif len(argv) in (3, 4) and argv[:2] == ["check", "router"]:
        router_check(argv[2], argv[3] if len(argv) == 4 else None)
    elif len(argv) in (3, 4) and argv[0] == "check":
        check(argv[1], argv[2], argv[3] if len(argv) == 4 else None)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
