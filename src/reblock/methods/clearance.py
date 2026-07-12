"""ClearanceReblocker: greedy least-cost-path reblocker with one physical knob (repulsion)
spanning aspirational straight roads -> buildable Voronoi-following roads.

Each road is a least-cost path from the current deepest parcel to the road+street network, on
an 8-connected grid whose edge weights come from a cost field that repels from building points:
    edge_weight = length * [(1 - t) + t / clearance],   clearance = dist(nearest building) (+ eps)
The user-facing knob is the logit s (`repulsion`); t = sigmoid(s) in (0, 1). s -> -inf: uniform
cost -> the straight line (aspirational, best directness). s -> +inf: hug the max-clearance ridges
= the Voronoi edges (equidistant from the two nearest buildings) = the buildable gaps. Access
depth is maintained incrementally (a road only lowers depth) so large regions stay fast.
See docs/superpowers/specs/2026-07-12-clearance-reblocker-design.md.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import Polygon

_CLEARANCE_EPS = 0.3       # keeps node cost finite on a grid node sitting on a building point
_NET_TOL_FACTOR = 1.5      # a grid node within res * this of the street seeds the network
_SIGMOID_EPS = 1e-15       # clamps sigmoid strictly inside (0, 1); float64 underflows to exact
                           # 0.0/1.0 well before |s| = 800 (already at |s| ~ 37), which would
                           # otherwise saturate the cost field's blend weight t


def _sigmoid(s: float) -> float:
    """t = sigmoid(s) in (0, 1) -- the logit knob's internal blend weight. Overflow-safe for
    extreme |s| (never returns exactly 0.0 or 1.0, so the cost field stays finite)."""
    if s >= 0.0:
        z = 1.0 / (1.0 + math.exp(-s))
    else:
        e = math.exp(s)
        z = e / (1.0 + e)
    return min(max(z, _SIGMOID_EPS), 1.0 - _SIGMOID_EPS)


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


def _node_clearance(
    pts: NDArray[np.float64], building_pts: NDArray[np.float64], radii: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-node clearance = distance to the nearest building minus that building's exclusion
    radius (0 in the plain case), floored at 0, plus `_CLEARANCE_EPS`. With uniform/zero radii
    this is the exact Euclidean clearance (== the Voronoi distance field); nonzero radii use the
    nearest-building approximation of the additively-weighted (Apollonius) clearance. No
    buildings -> uniform clearance, so the path is straight regardless of t."""
    if len(building_pts) == 0:
        return np.ones(len(pts), dtype=np.float64)
    dist, nearest = cKDTree(building_pts).query(pts)
    clear = np.maximum(dist - radii[nearest], 0.0) + _CLEARANCE_EPS
    return clear.astype(np.float64)


def _edge_weights(
    clear: NDArray[np.float64], t: float,
    rows: NDArray[np.int64], cols: NDArray[np.int64], edist: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Edge weight = length * average node cost, node cost = (1 - t) + t / clearance. t=0 ->
    uniform (straight); t->1 -> cost dominated by 1/clearance (hug the high-clearance gaps)."""
    node_cost = (1.0 - t) + t / clear
    return edist * 0.5 * (node_cost[rows] + node_cost[cols])


def _relax_depth(depth: NDArray[np.float64], adj: list[set[int]], served: Iterable[int]) -> None:
    """In place: given parcels `served` now front a street-connected road (depth 1), lower
    `depth` and propagate depth[j] = depth[i] + 1 outward along parcel adjacency `adj` (BFS),
    never raising a value.

    PRECONDITION: `depth` must be a proper BFS distance labelling in which any parcel with no
    adjacency path to a street is pinned to a sentinel >= every possible true in-block distance
    -- build it with `parcel_access_layers(..., unreached_depth=len(parcels)+1)`. Given that,
    this equals a full `parcel_access_layers` recompute for the post-road network: a road only
    adds street frontage (parcel adjacency is unchanged), so the post-road depth is a BFS from
    (original street seeds) union (newly served parcels), and every stale placeholder is high
    enough that the strict-decrease guard always re-propagates through it. WITHOUT that seeding,
    a disconnected component's default `max(reached)+1` placeholder can coincide with a true
    depth and halt propagation early, leaving parcels beyond it falsely shallow."""
    q: deque[int] = deque()
    for p in served:
        if depth[p] > 1.0:
            depth[p] = 1.0
            q.append(int(p))
    while q:
        i = q.popleft()
        di = depth[i]
        for j in adj[i]:
            if depth[j] > di + 1.0:
                depth[j] = di + 1.0
                q.append(j)
