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
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely import STRtree, contains_xy
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, unary_union

from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency

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


def _greedy_reblock(
    block: Block, *, t: float, res: float, depth_target: int, max_roads: int,
    radii: NDArray[np.float64],
) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    """Greedy least-cost-path reblock: repeatedly connect the deepest parcel to the growing
    road+street network by a Dijkstra path on the repulsion cost field, maintaining access
    depth incrementally, until every parcel is within `depth_target` (or `max_roads` is hit)."""
    parcels = block.parcels
    geoms = list(parcels.geometry)
    parcel_ids = np.asarray(parcels["parcel_id"])
    adj = parcel_adjacency(geoms, STREET_TOL)
    # Seed unreached (adjacency-disconnected) parcels to a sentinel above any true in-block
    # distance so `_relax_depth` stays exact as roads connect them (see its docstring precondition).
    depth = parcel_access_layers(
        block, None, adj=adj, unreached_depth=len(geoms) + 1).to_numpy().astype(np.float64)

    empty = gpd.GeoDataFrame(geometry=[], crs=block.crs)
    if depth.size == 0 or float(depth.max()) <= depth_target:
        return empty, {"roads": 0, "max_depth_after": int(depth.max()) if depth.size else 0,
                       "grid_unreachable": 0, "max_roads_hit": False}

    pts, rows, cols, edist = _build_grid(block.boundary, res)
    if len(pts) == 0:
        raise ValueError("Block.boundary yields no grid nodes at this resolution")
    building_pts = (
        shapely.get_coordinates(block.building_points.geometry.to_numpy())
        if not block.building_points.empty else np.empty((0, 2), dtype=np.float64)
    )
    clear = _node_clearance(pts, building_pts, radii)
    w = _edge_weights(clear, t, rows, cols, edist)
    csr = csr_matrix((w, (rows, cols)), shape=(len(pts), len(pts)))

    pt_tree = cKDTree(pts)
    reps = np.array([[g.representative_point().x, g.representative_point().y] for g in geoms])
    street = unary_union(list(block.streets.geometry))
    parcel_tree = STRtree(geoms)
    net = np.flatnonzero(
        shapely.dwithin(shapely.points(pts), street, res * _NET_TOL_FACTOR)).tolist()
    if not net:
        raise ValueError(
            "Block.streets yields no grid seed nodes: with no street frontage the least-cost "
            "forest has no root")

    roads: list[LineString] = []
    n_grid_unreachable = 0
    while len(roads) < max_roads:
        maxd = float(depth.max())
        if maxd <= depth_target:
            break
        cands = np.flatnonzero(depth == maxd)
        worst = int(cands[np.argmin(parcel_ids[cands])])          # deepest, ties by parcel_id
        start = int(pt_tree.query(reps[worst])[1])
        d, pred, _src = dijkstra(csr, indices=net, return_predecessors=True, min_only=True)
        if not np.isfinite(d[start]):
            depth[worst] = -np.inf                        # grid-unroutable: drop from selection
            n_grid_unreachable += 1
            continue
        pathn = [start]
        while pred[pathn[-1]] >= 0:
            pathn.append(int(pred[pathn[-1]]))
        coords: list[tuple[float, float]] = [(float(reps[worst][0]), float(reps[worst][1]))]
        coords += [(float(pts[k][0]), float(pts[k][1])) for k in pathn]
        term = Point(pts[pathn[-1]])
        if street.distance(term) <= res * _NET_TOL_FACTOR:        # bridge grid->street gap only
            sp = nearest_points(term, street)[1]                  # when we actually reached street
            coords.append((sp.x, sp.y))
        coords = [c for i, c in enumerate(coords) if i == 0 or c != coords[i - 1]]
        if len(coords) < 2:
            depth[worst] = -np.inf
            n_grid_unreachable += 1
            continue
        road = LineString(coords)
        roads.append(road)
        served = [int(p) for p in
                  parcel_tree.query(road, predicate="dwithin", distance=STREET_TOL)]
        _relax_depth(depth, adj, served)
        net.extend(pathn)

    gdf = gpd.GeoDataFrame(geometry=roads, crs=block.crs)
    final = parcel_access_layers(block, gdf, adj=adj)     # honest max over the ACTUAL network
    max_depth_after = int(final.max())                    # surfaces any grid-stranded parcel
    max_roads_hit = len(roads) >= max_roads and max_depth_after > depth_target
    params: dict[str, object] = {
        "roads": len(roads), "max_depth_after": max_depth_after,
        "grid_unreachable": n_grid_unreachable, "max_roads_hit": bool(max_roads_hit)}
    return gdf, params


@dataclass
class ClearanceReblocker:
    """Greedy least-cost-path reblocker. `repulsion` is the logit knob (s): s -> -inf straight
    (aspirational), 0 balanced, s -> +inf Voronoi-following (buildable). See module docstring."""

    repulsion: float = 0.0
    depth_target: int = 2
    res: float = 1.5
    max_roads: int = 400

    @property
    def identity(self) -> tuple[str, float, int, float, int]:
        return ("clearance", float(self.repulsion), int(self.depth_target),
                float(self.res), int(self.max_roads))

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior  # accepted for Method conformance; the routing is block-only
        t = _sigmoid(self.repulsion)
        n_b = 0 if block.building_points.empty else len(block.building_points)
        radii = np.zeros(n_b, dtype=np.float64)   # plain clearance; weighted footprints are future
        roads, params = _greedy_reblock(
            block, t=t, res=self.res, depth_target=self.depth_target,
            max_roads=self.max_roads, radii=radii)
        pid = f"clearance:r{self.repulsion:g}:d{self.depth_target}:res{self.res:g}"
        return Proposal(
            block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
            proposal_id=pid, method="clearance",
            params={**params, "repulsion": self.repulsion,
                    "depth_target": self.depth_target, "res": self.res},
            block_identity=block.identity)
