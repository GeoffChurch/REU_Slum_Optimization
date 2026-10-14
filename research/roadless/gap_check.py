"""Guards of the gap complex (gap_complex.py; NOTES, "The gap complex: passability, enclosure and
flow without the raster"), each with its fault injection (a patch in one process, no file
edited): a guard passes on the code as it is and must fail under its fault.

    S=research/roadless/gap_check.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S conform <ids,> [fault]
    uv run python $S delaunay <ids,> [fault]
    uv run python $S apertures - [fault]
    uv run python $S filtration - [fault]
    uv run python $S free - [fault]

`conform`: on blocks, at every S_SENS, the complex conforms to the outlines: every segment of the
open ground's densified rings is an edge of exactly one triangle (its ends read as the complex's
vertices within gap_complex.MERGE_M, a segment shorter than that merged away), no edge lies in
three, the triangles' area is the open ground's (to CONFORM_AREA of it) and no triangle's centroid
lies in a footprint by more than INSIDE_TOL (gap_complex.triangulated's polygons: the parts under
DUST_M2 left out, their area reported). Fault `unconstrained`: the Delaunay triangulation of the
same vertices (scipy), the triangles with their centroid in the open ground kept: an edge may
cross a footprint.

`delaunay`: the complex is constrained Delaunay: across every passage whose quadrilateral is
strictly convex the two triangles' circumcentres lie in order along its normal, (cot a + cot b) / 2
>= -gap_complex.DELAUNAY_TOL (a, b the angles facing it; the angles sum to at most pi), so no flip
would take the other diagonal, the property that keeps a gap's crossing edges short (the bichromatic
closest pair of two outlines is a Delaunay edge). Fault `geos`: GEOS's triangles as they come,
without gap_complex's flips (GEOS stops improving after a fixed number of scans). Fault `flip`:
every FLIP_EVERY-th flippable passage of GEOS's flipped (a valid triangulation, not Delaunay),
again without the flips.

`apertures`: two footprints facing each other across a gap g (APERTURE_GAPS) at angles
APERTURE_ANGLES, one shifted along the other by APERTURE_SHIFT (so no corners face each other):
the shortest passage between them lies in [g, sqrt(g^2 + s^2)] at every S_SENS. Fault `corners`:
no densification (the outlines' own vertices only): the corner-to-corner distance.

`filtration`: a courtyard home with two exits (gaps FILT_EXITS) has the wider exit's aperture as
its exit width; a home behind a crack alone the crack's; a sealed courtyard's -inf; a home on the
street's band inf. Fault `narrowest`: the minimum spanning forest of the apertures (each pocket's
exit read on its narrowest tree's path), in place of the maximum.

`free`: a walled-in home whose cheapest crossing in the region graph is a shed between two
footprints FREE_GAP apart (at width FREE_W, clearing it leaves the two too close): free first
frees the home by the complex rebuilt without its crossing (another pass takes a footprint
beside the shed). Fault `no-rebuild`: the region graph's claim taken as the answer.
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import shapely
from scipy.spatial import Delaunay, cKDTree
from shapely import affinity
from shapely.geometry import MultiLineString, Polygon, box

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import gap_complex as gc  # noqa: E402
import lifted  # noqa: E402

S_SENS = (0.125, 0.25, 0.5)
CONFORM_AREA = 1e-9
INSIDE_TOL = 1e-7          # m: a centroid inside a footprint by more than this is a triangle in it
FLIP_EVERY = 50


@contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    old = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


def spec_of(s: float) -> gc.Spec:
    return gc.Spec(s=s, ring_m=lifted.RING_M, band_m=lifted.BAND_M)


def synthetic_block(boundary: Polygon, footprints: list, bid: str = "synthetic"):
    """A Block's fields as gap_complex reads them: the streets the boundary's rings."""
    streets = MultiLineString([boundary.exterior, *boundary.interiors])
    return SimpleNamespace(block_id=bid, boundary=boundary,
                           buildings=SimpleNamespace(outlines=list(footprints)),
                           streets=SimpleNamespace(geometry=[streets]))


# ------------------------------------------------------------------ conform

def _vertex_ids(tree: cKDTree, c: np.ndarray) -> np.ndarray:
    """Each point's vertex: the complex's within MERGE_M (gap_complex.vertices merges closer ones),
    -1 none."""
    d, i = tree.query(c, distance_upper_bound=gc.MERGE_M)
    return np.where(np.isfinite(d), i, -1)


def conform(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        geo = gc.geometry(b)
        n = len(geo.polys)
        for s in S_SENS:
            spec = spec_of(s)
            cx = gc.build(geo, spec, np.ones(n, dtype=bool), carve=None, grow=0.0)
            inner, _band = gc._open_ground(geo, np.ones(n, dtype=bool), spec.band_m,
                                           carve=None, grow=0.0)
            nv = len(cx.xy)
            tree = cKDTree(cx.xy)
            # every densified ring segment an edge of exactly one triangle
            e = np.sort(np.stack([cx.tri[:, [0, 1]], cx.tri[:, [1, 2]], cx.tri[:, [2, 0]]],
                                 axis=1).reshape(-1, 2), axis=1)
            key, cnt = np.unique(e[:, 0] * nv + e[:, 1], return_counts=True)
            count = dict(zip(key.tolist(), cnt.tolist(), strict=True))
            missing = wrong = 0
            polys = [q for q, _t in gc.triangulated(inner, s)]
            dust = inner.area - float(sum(q.area for q in polys))
            for part in shapely.get_parts(polys):
                for ring in [part.exterior, *part.interiors]:
                    v = _vertex_ids(tree, shapely.get_coordinates(ring))
                    for i, j in zip(v[:-1].tolist(), v[1:].tolist(), strict=True):
                        if i >= 0 and i == j:
                            continue                    # a repeated vertex, merged together
                        if i < 0 or j < 0:
                            missing += 1
                            continue
                        k = min(i, j) * nv + max(i, j)
                        if count.get(k, 0) != 1:
                            wrong += 1
            three = int((cnt > 2).sum())
            area = abs(cx.area.sum() - (inner.area - dust)) / inner.area
            cen = shapely.centroid(cx.tri_geom)
            fu = shapely.union_all(geo.polys)
            shapely.prepare(fu)
            within = shapely.contains(fu, cen)
            # beyond rounding: a flat triangle along a wall has its centroid on the wall, which the
            # 1e-9 snap (gap_complex.SNAP_M) can put a hair inside
            inside = int((shapely.distance(fu.boundary, cen[within]) > INSIDE_TOL).sum())
            ok = missing == 0 and wrong == 0 and three == 0 and area <= CONFORM_AREA and inside == 0
            print(f"{bid} s {s:g}: {cx.nt} triangles; ring segments not a vertex pair {missing}, "
                  f"not in exactly one triangle {wrong}; edges in three {three}; area off "
                  f"{area:.1e} (dust dropped {dust:.1e} m^2); centroids in a footprint {inside} -> "
                  f"{'ok' if ok else 'FAILS'}", flush=True)
            if not ok:
                bad.append(f"{bid} s {s:g}")
    return bad


_TRIANGULATE = gc._triangulate


def _unconstrained(inner, s: float):
    """The Delaunay triangulation of the densified rings' vertices, the triangles whose centroid
    lies in the open ground kept (no constraint)."""
    seg = shapely.union_all([q for q, _t in gc.triangulated(inner, s)])
    xy = np.unique(shapely.get_coordinates(seg), axis=0)
    tri = Delaunay(xy).simplices
    polys = shapely.polygons(xy[tri])
    shapely.prepare(inner)
    return polys[shapely.contains(inner, shapely.centroid(polys))]


# ------------------------------------------------------------------ delaunay

def _facing_angles(cx: gc.Complex) -> tuple[np.ndarray, np.ndarray]:
    """Per passage, (cot a + cot b) / 2 of the two angles facing it (its circumcentres' distance
    along its normal over its length), and whether its quadrilateral is strictly convex (a flip
    possible)."""
    def third(t, u, v):
        T = cx.tri[t]
        return np.where((T[:, 0] != u) & (T[:, 0] != v), T[:, 0],
                        np.where((T[:, 1] != u) & (T[:, 1] != v), T[:, 1], T[:, 2]))

    def cot(o, u, v):
        a, b = cx.xy[u] - cx.xy[o], cx.xy[v] - cx.xy[o]
        cr = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        dot = (a * b).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(cr > 0, dot / cr, np.where(dot < 0, -np.inf, np.inf))
    p, q = third(cx.pa, cx.pu, cx.pv), third(cx.pb, cx.pu, cx.pv)

    def orient(a, b, c):
        A, B, C = cx.xy[a], cx.xy[b], cx.xy[c]
        return (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
    # p, u, q, v around the quadrilateral: convex iff p, q lie on opposite sides of u v and u, v
    # on opposite sides of p q, strictly
    convex = (np.sign(orient(cx.pu, cx.pv, p)) * np.sign(orient(cx.pu, cx.pv, q)) < 0) & (
        np.sign(orient(p, q, cx.pu)) * np.sign(orient(p, q, cx.pv)) < 0)
    return 0.5 * (cot(p, cx.pu, cx.pv) + cot(q, cx.pu, cx.pv)), convex


def delaunay(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        geo = gc.geometry(b)
        n = len(geo.polys)
        for s in S_SENS:
            cx = gc.build(geo, spec_of(s), np.ones(n, dtype=bool), carve=None,
                          grow=0.0)
            dl, convex = _facing_angles(cx)
            over = convex & (dl < -gc.DELAUNAY_TOL)
            ok = not over.any()
            worst = float(dl[convex].min()) if convex.any() else np.inf
            print(f"{bid} s {s:g}: {len(dl)} passages, {int(convex.sum())} flippable, "
                  f"{int(over.sum())} of them with their centres in the wrong order (the worst "
                  f"d / L {worst:+.1e}; {int((~convex & (dl < -gc.DELAUNAY_TOL)).sum())} more, "
                  f"not flippable: degenerate) -> {'ok' if ok else 'FAILS'}", flush=True)
            if not ok:
                bad.append(f"{bid} s {s:g}")
    return bad


def _flipping(inner, s: float):
    """_triangulate with every FLIP_EVERY-th flippable passage flipped."""
    tris = _TRIANGULATE(inner, s)
    c = shapely.get_coordinates(tris).reshape(-1, 4, 2)[:, :3]
    xy, inv = np.unique(c.reshape(-1, 2), axis=0, return_inverse=True)
    T = inv.reshape(-1, 3)
    nv = len(xy)
    e = np.sort(np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1).reshape(-1, 2), axis=1)
    key = e[:, 0] * nv + e[:, 1]
    order = np.argsort(key, kind="stable")
    same = key[order][1:] == key[order][:-1]
    i0, i1 = order[:-1][same], order[1:][same]
    used = np.zeros(len(T), dtype=bool)
    out = [list(t) for t in T]
    # flip: per shared edge (u, v) of triangles a = (u, v, p), b = (v, u, q) -> (p, q, u), (q, p, v)
    for idx, (ea, eb) in enumerate(zip(i0, i1, strict=True)):
        a, b = ea // 3, eb // 3
        if used[a] or used[b] or idx % FLIP_EVERY:
            continue
        u, v = e[ea]
        p = [x for x in T[a] if x not in (u, v)][0]
        q = [x for x in T[b] if x not in (u, v)][0]
        quad = Polygon(xy[[u, p, v, q]])
        if not quad.is_valid or quad.convex_hull.area > quad.area * (1 + 1e-12):
            continue                                       # not convex: no flip
        out[a], out[b] = [p, q, u], [q, p, v]
        used[a] = used[b] = True
    return shapely.polygons(xy[np.array(out)])


# ------------------------------------------------------------------ apertures

APERTURE_GAPS = (0.05, 0.125, 0.3, 1.0)
APERTURE_ANGLES = (0.0, 17.0, 45.0)
APERTURE_SHIFT = 1.37      # one footprint shifted along the gap (m): no corners face each other
WALL = 6.0                 # the footprints' facing walls (m)


def _gap_block(g: float, angle: float):
    """Two 4 x WALL footprints facing each other across a gap g, the second shifted along the gap,
    rotated by `angle` degrees, in a block with room around them."""
    a = box(0.0, 0.0, 4.0, WALL)
    b = box(4.0 + g, APERTURE_SHIFT, 8.0 + g, WALL + APERTURE_SHIFT)
    blk = box(-6.0, -6.0, 14.0 + g, WALL + APERTURE_SHIFT + 6.0)
    rot = [affinity.rotate(x, angle, origin=(0, 0)) for x in (a, b, blk)]
    return synthetic_block(rot[2], rot[:2])


def apertures(_ids: list[str]) -> list[str]:
    bad = []
    for s in S_SENS:
        for g in APERTURE_GAPS:
            for ang in APERTURE_ANGLES:
                blk = _gap_block(g, ang)
                geo = gc.geometry(blk)
                cx = gc.build(geo, spec_of(s), np.ones(2, dtype=bool), carve=None,
                              grow=0.0)
                vb = cx.vb.tocsr()
                ua = np.asarray(vb[cx.pu].todense())
                va = np.asarray(vb[cx.pv].todense())
                between = ((ua[:, 0] > 0) & (va[:, 1] > 0)) | ((ua[:, 1] > 0) & (va[:, 0] > 0))
                m = float(cx.ap[between].min()) if between.any() else np.inf
                hi = np.hypot(g, s)
                ok = g - 1e-9 <= m <= hi + 1e-9
                print(f"s {s:g} gap {g:g} angle {ang:g}: measured {m:.4f} (in [{g:g}, {hi:.4f}]: "
                      f"{'ok' if ok else 'FAILS'})", flush=True)
                if not ok:
                    bad.append(f"s {s:g} g {g:g} a {ang:g}")
    return bad


def _no_densify(inner, s: float):
    return _TRIANGULATE(inner, 1e9)


# ------------------------------------------------------------------ filtration

FILT_EXITS = (0.3, 0.8)
FILT_CRACK = 0.2


def _courtyard(exits: tuple[float, ...], sealed: bool = False):
    """A 6 x 6 m courtyard ringed by footprints 2 m deep, its home a 2 x 2 m footprint in the
    middle; on the west and east sides a gap of each width in `exits` (none if `sealed`); the
    block's edge 8 m beyond the ring, so the band (1 m) is far."""
    c0, c1 = 0.0, 6.0
    d = 2.0
    ring = box(c0 - d, c0 - d, c1 + d, c1 + d).difference(box(c0, c0, c1, c1))
    cuts = []
    if not sealed:
        for side, g in zip((-1, 1), exits, strict=False):
            x = c0 - d if side < 0 else c1
            cuts.append(box(x, 3.0 - g / 2, x + d, 3.0 + g / 2))
    walls = shapely.get_parts(ring.difference(shapely.union_all(cuts)) if cuts else ring)
    home = box(2.0, 2.0, 4.0, 4.0)
    blk = box(-10.0, -10.0, 16.0, 16.0)
    return synthetic_block(blk, [home, *walls]), 0


def filtration(_ids: list[str]) -> list[str]:
    bad = []
    s = 0.125
    cases = (("two exits", _courtyard(FILT_EXITS), max(FILT_EXITS)),
             ("a crack alone", _courtyard((FILT_CRACK,)), FILT_CRACK),
             ("sealed", _courtyard((), sealed=True), -np.inf))
    for name, (blk, home), want in cases:
        geo = gc.geometry(blk)
        n = len(geo.polys)
        cx = gc.build(geo, spec_of(s), np.ones(n, dtype=bool), carve=None,
                          grow=0.0)
        he = gc.home_exits(gc.homes_of(geo, cx), gc.exit_widths(cx))
        got = he[home]
        ok = (got == want) if not np.isfinite(want) else abs(got - want) <= 1e-9
        print(f"{name}: the home's exit width {got:.4f} (want {want:.4f}) -> "
              f"{'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(name)
    # a home whose ring reaches the band
    blk = synthetic_block(box(0, 0, 20, 20), [box(1.3, 5, 4, 8)])
    geo = gc.geometry(blk)
    cx = gc.build(geo, spec_of(s), np.ones(1, dtype=bool), carve=None, grow=0.0)
    got = gc.home_exits(gc.homes_of(geo, cx), gc.exit_widths(cx))[0]
    ok = got == np.inf
    print(f"a home 0.3 m from the band: exit width {got} (want inf) -> {'ok' if ok else 'FAILS'}",
          flush=True)
    if not ok:
        bad.append("band")
    return bad


def _narrowest(cx: gc.Complex) -> gc.Exits:
    """exit_widths on the reversed order (each finite aperture by its inverse, the street's
    passages still first): the minimum spanning forest of the apertures, each pocket's exit
    the widest passage on its narrowest tree's path."""
    flipped = gc.Complex(**{**cx.__dict__, "ap": np.where(np.isfinite(cx.ap), 1.0 / cx.ap,
                                                           np.inf)})
    ex = _EXIT_WIDTHS(flipped)
    w = np.where(np.isfinite(ex.width) & (ex.width > 0), 1.0 / np.where(ex.width > 0, ex.width,
                                                                        1.0), ex.width)
    return gc.Exits(width=w, bar=ex.bar)


_EXIT_WIDTHS = gc.exit_widths


# ------------------------------------------------------------------ free first

FREE_GAP = 0.3       # the two footprints either side of the shed (m)
FREE_W = 0.5         # the width free first works at (m)


def _shed_block():
    """A home in a courtyard sealed by a ring of footprints; on the east side the ring has a
    1 m gap holding a shed (0.9 m deep) whose two neighbours stand FREE_GAP apart across it, so
    clearing the shed opens only a FREE_GAP slot; the ring's footprint south of the gap is the
    expensive way out. Homes: 0 the home, 1 the shed (cheap), 2 -- 3 the shed's neighbours."""
    home = box(2.0, 2.0, 4.0, 4.0)
    west = box(-2.0, -2.0, 0.0, 8.0)
    north = box(0.0, 6.0, 6.0, 8.0)
    south = box(0.0, -2.0, 6.0, 0.0)
    east_lo = box(6.0, -2.0, 8.0, 2.5)                 # neighbour, its north wall at 2.5
    east_hi = box(6.0, 2.5 + FREE_GAP, 8.0, 8.0)       # neighbour, its south wall FREE_GAP up
    # east_lo and east_hi each overhang the gap's slot from x 6 to 8: the shed fills the slot
    shed = box(6.0, 2.5, 8.0, 2.5 + FREE_GAP)
    blk = box(-10.0, -10.0, 18.0, 18.0)
    return synthetic_block(blk, [home, shed, east_lo, east_hi, west, north, south])


def free(_ids: list[str]) -> list[str]:
    blk = _shed_block()
    geo = gc.geometry(blk)
    n = len(geo.polys)
    wpop = np.array([1.0, 0.05, 1.0, 1.2, 3.0, 3.0, 3.0])
    cost = wpop / wpop.sum()
    spec = spec_of(0.125)
    fr = gc.free_first(geo, spec, w=FREE_W, grow=0.0, budget=1.0, cost=cost, pop=wpop,
                       held0=np.zeros(n, dtype=bool), cx0=None)
    cx = gc.build(geo, spec, ~fr.held, carve=None, grow=0.0)
    he = gc.home_exits(gc.homes_of(geo, cx), gc.exit_widths(cx))
    ok = he[0] >= FREE_W and not fr.left[0]
    print(f"free first at w {FREE_W:g}: held {np.flatnonzero(fr.held).tolist()}, passes "
          f"{fr.passes}, the graph's misses {fr.misses}; the home's exit width rebuilt "
          f"{he[0]:.3f} -> {'ok' if ok else 'FAILS'}", flush=True)
    return [] if ok else ["shed"]


def _no_rebuild(geo, spec, *, w, grow, budget, cost, pop, held0, cx0):
    """free_first stopping after its first pass: the graph's claim is the answer."""
    real = gc.build

    calls = {"n": 0}

    def build_once(*a, **k):
        calls["n"] += 1
        if calls["n"] > 1:
            return real(a[0], a[1], np.ones_like(a[2]), **k)     # the rebuild: nothing changes
        return real(*a, **k)
    with _patched(gc, "build", build_once):
        fr = _FREE_FIRST(geo, spec, w=w, grow=grow, budget=budget, cost=cost, pop=pop,
                         held0=held0, cx0=cx0)
    return gc.Freed(target=fr.target, held=fr.held, left=np.zeros_like(fr.left),
                    crossings=fr.crossings, passes=1, misses=0)


_FREE_FIRST = gc.free_first


def _as_is(xy, tri):
    return tri


@contextmanager
def _fault_flip() -> Iterator[None]:
    with _patched(gc, "delaunay_flips", _as_is), _patched(gc, "_triangulate", _flipping):
        yield


GUARDS: dict[str, Callable[[list[str]], list[str]]] = {
    "conform": conform, "delaunay": delaunay, "apertures": apertures,
    "filtration": filtration, "free": free}
FAULTS: dict[str, dict[str, Callable]] = {
    "conform": {"unconstrained": lambda: _patched(gc, "_triangulate", _unconstrained)},
    "delaunay": {"geos": lambda: _patched(gc, "delaunay_flips", _as_is),
                 "flip": lambda: _fault_flip()},
    "apertures": {"corners": lambda: _patched(gc, "_triangulate", _no_densify)},
    "filtration": {"narrowest": lambda: _patched(gc, "exit_widths", _narrowest)},
    "free": {"no-rebuild": lambda: _patched(gc, "free_first", _no_rebuild)},
}


def main(guard: str, ids: list[str], fault: str | None) -> None:
    if fault is None:
        bad = GUARDS[guard](ids)
    else:
        with FAULTS[guard][fault]():
            bad = GUARDS[guard](ids)
    verdict = "FAILS on " + ", ".join(bad) if bad else "passes"
    print(f"{guard}{'' if fault is None else f' under the fault {fault}'}: {verdict}", flush=True)
    if bad and fault is None:
        raise SystemExit(1)
    if not bad and fault is not None:
        raise SystemExit(f"the fault {fault} went unseen: the guard guards nothing")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) not in (2, 3) or a[0] not in GUARDS:
        raise SystemExit(__doc__)
    main(a[0], [] if a[1] == "-" else a[1].split(","), a[2] if len(a) == 3 else None)
