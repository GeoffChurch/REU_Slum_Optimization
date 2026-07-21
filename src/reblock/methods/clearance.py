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
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points, unary_union

from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.substrates import ChordSubstrate, RoutingGraph, Substrate

_CLEARANCE_EPS = 0.3       # keeps node cost finite on a grid node sitting on a building point
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
    pts: NDArray[np.float64], rows: NDArray[np.int64], cols: NDArray[np.int64],
    edist: NDArray[np.float64], building_pts: NDArray[np.float64],
    radii: NDArray[np.float64], t: float,
) -> NDArray[np.float64]:
    """Edge weight = length * mean(node cost) sampled at BOTH endpoints AND the midpoint, node
    cost = (1 - t) + t / clearance. 3-point (not endpoint-only) so a long edge whose midpoint
    skims a building — but whose endpoints sit in the open — still reads as expensive. Returns
    weights aligned to the symmetric COO `rows`/`cols` order."""
    n = len(pts)
    mask = rows < cols                                   # one direction per undirected edge
    ui, uj, ulen = rows[mask], cols[mask], edist[mask]
    e = len(ui)
    if e == 0:
        return np.zeros(0, dtype=np.float64)
    mid = (pts[ui] + pts[uj]) / 2.0
    sample_pts = np.vstack([pts[ui], pts[uj], mid])
    clear = _node_clearance(sample_pts, building_pts, radii)
    ci, cj, cm = clear[:e], clear[e:2 * e], clear[2 * e:]
    mean_cost = ((1.0 - t) + t / ci) + ((1.0 - t) + t / cj) + ((1.0 - t) + t / cm)
    uw = ulen * (mean_cost / 3.0)
    # scatter the per-undirected-edge weight back onto BOTH directed COO entries
    key = np.minimum(rows, cols).astype(np.int64) * n + np.maximum(rows, cols).astype(np.int64)
    ukey = ui.astype(np.int64) * n + uj.astype(np.int64)
    order = np.argsort(ukey)
    pos = np.searchsorted(ukey[order], key)
    return uw[order][pos]


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
    block: Block, graph: RoutingGraph, *, t: float, depth_target: int, max_roads: int,
    radii: NDArray[np.float64],
) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    """Greedy least-cost-path reblock on a routing substrate `graph`: repeatedly connect the
    deepest parcel to the growing road+street network by a Dijkstra path on the repulsion cost
    field, maintaining access depth incrementally, until every parcel is within `depth_target`
    (or `max_roads` is hit)."""
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

    pts, rows, cols, edist, net_tol = (
        graph.pts, graph.rows, graph.cols, graph.edist, graph.net_tol)
    if len(pts) == 0:
        raise ValueError("substrate yields no nodes for this block")
    building_pts = (
        shapely.get_coordinates(block.building_points.geometry.to_numpy())
        if not block.building_points.empty else np.empty((0, 2), dtype=np.float64)
    )
    w = _edge_weights(pts, rows, cols, edist, building_pts, radii, t)
    csr = csr_matrix((w, (rows, cols)), shape=(len(pts), len(pts)))

    pt_tree = cKDTree(pts)
    reps = np.array([[g.representative_point().x, g.representative_point().y] for g in geoms])
    street = unary_union(list(block.streets.geometry))
    parcel_tree = STRtree(geoms)
    net = np.flatnonzero(shapely.dwithin(shapely.points(pts), street, net_tol)).tolist()
    if not net:
        raise ValueError(
            "substrate net seed empty: no node within net_tol of the street -- with no street "
            "frontage the least-cost forest has no root")

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
        if street.distance(term) <= net_tol:                      # bridge grid->street gap only
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


@dataclass(frozen=True)
class ClearanceIdentity:
    """Cache-key identity for ClearanceReblocker. The dataclass type discriminates the method (no
    string tag). `substrate` holds the child substrate's own identity verbatim (whatever it
    returns -- a tuple today); frozen -> hashable, usable as an L1 dict key and joblib-picklable."""
    substrate: Hashable            # the nested Substrate.identity (not converted in this pass)
    repulsion: float
    depth_target: int
    max_roads: int


@dataclass
class ClearanceReblocker:
    """Greedy least-cost-path reblocker on a pluggable routing substrate (default chord_diag,
    the parcel-boundary graph + all within-cell diagonals). `repulsion` is the logit knob (s):
    s -> -inf straight (aspirational), 0 balanced, s -> +inf Voronoi-following (buildable)."""

    substrate: Substrate = field(default_factory=ChordSubstrate)   # chord_diag, the default winner
    repulsion: float = 0.0
    depth_target: int = 2
    max_roads: int = 400

    @property
    def identity(self) -> ClearanceIdentity | None:
        # An uncacheable substrate (PrebuiltSubstrate: identity None) makes the whole method
        # uncacheable -- propagate the None up so derive() bypasses the memoized propose, else two
        # different ad-hoc graphs would key-collide in the access_after/geometric_after caches.
        if self.substrate.identity is None:
            return None
        return ClearanceIdentity(
            substrate=self.substrate.identity, repulsion=float(self.repulsion),
            depth_target=int(self.depth_target), max_roads=int(self.max_roads))

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior  # accepted for Method conformance; the routing is block-only
        t = _sigmoid(self.repulsion)
        n_b = 0 if block.building_points.empty else len(block.building_points)
        radii = np.zeros(n_b, dtype=np.float64)   # plain clearance; weighted footprints are future
        graph = self.substrate.build(block)
        roads, params = _greedy_reblock(
            block, graph, t=t, depth_target=self.depth_target,
            max_roads=self.max_roads, radii=radii)
        pid = (f"clearance:{self.substrate.tag}:r{self.repulsion:g}"
               f":d{self.depth_target}:mr{self.max_roads}")
        # An uncacheable substrate (PrebuiltSubstrate: identity None, fixed tag "prebuilt") gives
        # ad-hoc roads proposal_id can't distinguish, so its eval must bypass the cache too: drop
        # block_identity to None -> Proposal.identity None -> uncacheable end-to-end (matching
        # Method.identity, already None). Else two prebuilt graphs collide on the eval-cache key.
        cacheable = self.substrate.identity is not None
        return Proposal(
            block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
            proposal_id=pid, method="clearance",
            params={**params, "substrate": self.substrate.tag, "repulsion": self.repulsion,
                    "depth_target": self.depth_target},
            block_identity=block.identity if cacheable else None)
