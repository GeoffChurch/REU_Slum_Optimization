"""Routing substrates for the clearance reblocker: a pluggable graph the least-cost-path greedy
routes on. Each Substrate.build(block) returns a RoutingGraph (node coords + symmetric COO edges
+ a network-seed tolerance). The cost field + greedy (reblock.methods.clearance) are substrate-
agnostic; substrates differ only in node set + edge selection. See
docs/superpowers/specs/2026-07-12-parametric-substrate-design.md.
"""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Protocol, cast

import geopandas as gpd
import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.spatial import Delaunay
from shapely import contains_xy
from shapely.geometry import MultiPolygon, Polygon

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL


@dataclass(frozen=True)
class RoutingGraph:
    """A built substrate: node coords `pts` (M,2), symmetric COO edges `rows`/`cols` (each
    undirected edge stored both ways) with lengths `edist`, and `net_tol` (a node within this of
    a street both seeds the network and gates the final street-snap)."""

    pts: NDArray[np.float64]
    rows: NDArray[np.int64]
    cols: NDArray[np.int64]
    edist: NDArray[np.float64]
    net_tol: float


class Substrate(Protocol):
    def build(self, block: Block) -> RoutingGraph: ...
    @property
    def identity(self) -> Hashable: ...
    @property
    def tag(self) -> str: ...


def _pack_edges(
    pts: NDArray[np.float64], edges: set[frozenset[int]]
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Symmetric COO from a set of undirected {i, j} edges, sorted for determinism."""
    ordered = sorted(tuple(sorted(e)) for e in edges)
    rows: list[int] = []
    cols: list[int] = []
    dist: list[float] = []
    for i, j in ordered:
        d = float(np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1]))
        rows += [i, j]
        cols += [j, i]
        dist += [d, d]
    return (pts.astype(np.float64), np.asarray(rows, dtype=np.int64),
            np.asarray(cols, dtype=np.int64), np.asarray(dist, dtype=np.float64))


def _build_grid(
    boundary: Polygon | MultiPolygon, res: float
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """8-connected grid of points inside `boundary` at spacing `res`. Returns
    (pts (M,2), rows, cols, edist): `rows`/`cols` are symmetric COO edge endpoints (each
    undirected edge stored both ways) and `edist` their Euclidean lengths (res or res*sqrt2)."""
    minx, miny, maxx, maxy = boundary.bounds
    xs = np.arange(minx, maxx + res, res)
    ys = np.arange(miny, maxy + res, res)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.c_[gx.ravel(), gy.ravel()]
    pts = pts[contains_xy(boundary, pts[:, 0], pts[:, 1])]
    idx = {(round(float(x), 3), round(float(y), 3)): k for k, (x, y) in enumerate(pts)}
    rows: list[int] = []
    cols: list[int] = []
    dist: list[float] = []
    for k, (x, y) in enumerate(pts):
        for dx, dy in ((res, 0.0), (0.0, res), (res, res), (res, -res)):
            nb = idx.get((round(float(x + dx), 3), round(float(y + dy), 3)))
            if nb is not None:
                d = float(np.hypot(dx, dy))
                rows += [k, nb]
                cols += [nb, k]
                dist += [d, d]
    return (
        pts.astype(np.float64),
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(dist, dtype=np.float64),
    )


_GRID_NET_TOL_FACTOR = 1.5   # a grid node within res * this of the street seeds the network


@dataclass(frozen=True)
class GridSubstrate:
    """8-connected regular grid at resolution `res` (m). Faithful cost-field sampler, but node
    count scales with block AREA (∝ area/res²) and paths staircase at 45°."""

    res: float = 1.5

    @property
    def identity(self) -> Hashable:
        return ("grid", float(self.res))

    @property
    def tag(self) -> str:
        return "grid"

    def build(self, block: Block) -> RoutingGraph:
        pts, rows, cols, edist = _build_grid(block.boundary, self.res)
        return RoutingGraph(pts, rows, cols, edist, net_tol=self.res * _GRID_NET_TOL_FACTOR)


def _boundary_vertices(
    parcels: gpd.GeoDataFrame,
) -> tuple[NDArray[np.float64], dict[tuple[float, float], int], set[frozenset[int]]]:
    """The parcel-tessellation boundary graph as (node coords, coord->index map, boundary-edge
    set). Nodes are the boundary vertices (snapped to cm by dijkstra._rnd), edges the party-wall
    segments. Shared node set for the chord / spanner / cdt substrates — nodes sit in the gaps
    between buildings, never on them."""
    from reblock.methods import dijkstra as dijkstra_mod
    g = dijkstra_mod._boundary_graph(parcels)
    nodes_sorted = sorted(g.nodes())
    node_idx = {n: i for i, n in enumerate(nodes_sorted)}
    pts = np.asarray(nodes_sorted, dtype=np.float64)
    edges = {frozenset((node_idx[a], node_idx[b])) for a, b in g.edges()}
    return pts, node_idx, edges


@dataclass(frozen=True)
class ChordSubstrate:
    """Boundary-vertex graph + ALL within-cell diagonals (every non-adjacent pair in each
    parcel's exterior ring; parcels are ~convex so every diagonal is interior/valid). Node count
    ∝ parcels (not area); the winner of the substrate head-to-head. `net_tol = STREET_TOL`."""

    @property
    def identity(self) -> Hashable:
        return ("chord_diag",)

    @property
    def tag(self) -> str:
        return "chord_diag"

    def build(self, block: Block) -> RoutingGraph:
        from reblock.methods import dijkstra as dijkstra_mod
        pts, node_idx, edges = _boundary_vertices(block.parcels)
        for geom in block.parcels.geometry:
            coords = list(cast(Polygon, geom).exterior.coords)[:-1]     # drop closing duplicate
            ring = [node_idx[ni] for c in coords
                    if (ni := dijkstra_mod._rnd(cast(tuple[float, float], c))) in node_idx]
            m = len(ring)
            if m < 3:
                continue
            for a in range(m):
                for b in range(a + 2, m):
                    if a == 0 and b == m - 1:
                        continue                            # wraparound-adjacent (a boundary edge)
                    if ring[a] != ring[b]:
                        edges.add(frozenset((ring[a], ring[b])))
        r, ro, co, di = _pack_edges(pts, edges)
        return RoutingGraph(r, ro, co, di, net_tol=STREET_TOL)


@dataclass(frozen=True)
class SpannerSubstrate:
    """Theta/Yao geometric spanner on the boundary vertices: per node, partition directions into
    `cones` angular cones and connect to the nearest node in each non-empty cone — an O(n·cones)-
    edge spanner with bounded stretch. Sparsest of the tessellation substrates. `net_tol =
    STREET_TOL`."""

    cones: int = 6

    @property
    def identity(self) -> Hashable:
        return ("theta_spanner", int(self.cones))

    @property
    def tag(self) -> str:
        return "theta_spanner"

    def build(self, block: Block) -> RoutingGraph:
        pts, _node_idx, edges = _boundary_vertices(block.parcels)
        n = len(pts)
        two_pi = 2.0 * np.pi
        cone_width = two_pi / self.cones
        for i in range(n):
            dx = pts[:, 0] - pts[i, 0]
            dy = pts[:, 1] - pts[i, 1]
            dist = np.hypot(dx, dy)
            cone = np.floor(np.mod(np.arctan2(dy, dx), two_pi) / cone_width).astype(np.int64)
            for c in range(self.cones):
                mask = (cone == c) & (dist > 0.0)
                if not np.any(mask):
                    continue
                idxs = np.flatnonzero(mask)
                j = int(idxs[np.argmin(dist[idxs])])
                edges.add(frozenset((i, j)))
        r, ro, co, di = _pack_edges(pts, edges)
        return RoutingGraph(r, ro, co, di, net_tol=STREET_TOL)


@dataclass(frozen=True)
class CdtSubstrate:
    """Delaunay triangulation of the boundary vertices, edges clipped to the block (an edge is
    kept only if its whole segment stays within the boundary, so it can't cut across a concave
    notch). Delaunay-SELECTED diagonals — sparser edges than chord_diag. `net_tol = STREET_TOL`."""

    @property
    def identity(self) -> Hashable:
        return ("cdt_gap",)

    @property
    def tag(self) -> str:
        return "cdt_gap"

    def build(self, block: Block) -> RoutingGraph:
        pts, _node_idx, edges = _boundary_vertices(block.parcels)
        pts_u = np.unique(pts, axis=0)
        if len(pts_u) >= 4:                                 # Delaunay needs >=3 non-collinear pts
            tri = Delaunay(pts_u)
            tri_edges: set[frozenset[int]] = set()
            for s in tri.simplices:
                i0, i1, i2 = (int(x) for x in s)
                tri_edges |= {frozenset((i0, i1)), frozenset((i1, i2)), frozenset((i0, i2))}
            ordered = sorted(tuple(sorted(e)) for e in tri_edges)
            from shapely.geometry import LineString
            boundary_buf = block.boundary.buffer(1e-6)      # tolerate exact frontage-segment runs
            segs = np.array([LineString([pts_u[i], pts_u[j]]) for i, j in ordered], dtype=object)
            keep = shapely.covers(boundary_buf, segs)
            # remap unique-point indices back to the boundary-vertex indices via coordinate match
            uidx = {(round(float(x), 3), round(float(y), 3)): k for k, (x, y) in enumerate(pts)}
            for k, ok in enumerate(keep):
                if ok:
                    i, j = ordered[k]
                    a = uidx[(round(float(pts_u[i, 0]), 3), round(float(pts_u[i, 1]), 3))]
                    b = uidx[(round(float(pts_u[j, 0]), 3), round(float(pts_u[j, 1]), 3))]
                    edges.add(frozenset((a, b)))
        r, ro, co, di = _pack_edges(pts, edges)
        return RoutingGraph(r, ro, co, di, net_tol=STREET_TOL)


@dataclass(frozen=True)
class PrebuiltSubstrate:
    """The 'provided graph' escape hatch: `build` returns the given RoutingGraph verbatim.
    identity is None (uncacheable) — an ad-hoc graph must not key-collide with a named substrate."""

    graph: RoutingGraph

    @property
    def identity(self) -> Hashable:
        return None

    @property
    def tag(self) -> str:
        return "prebuilt"

    def build(self, block: Block) -> RoutingGraph:
        del block
        return self.graph
