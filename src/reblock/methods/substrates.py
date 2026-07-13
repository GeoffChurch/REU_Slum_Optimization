"""Routing substrates for the clearance reblocker: a pluggable graph the least-cost-path greedy
routes on. Each Substrate.build(block) returns a RoutingGraph (node coords + symmetric COO edges
+ a network-seed tolerance). The cost field + greedy (reblock.methods.clearance) are substrate-
agnostic; substrates differ only in node set + edge selection. See
docs/superpowers/specs/2026-07-12-parametric-substrate-design.md.
"""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from shapely import contains_xy
from shapely.geometry import Polygon

from reblock.contracts import Block


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
    boundary: Polygon, res: float
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
