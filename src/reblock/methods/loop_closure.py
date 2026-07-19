"""Bridge-tree greedy loop-closure engine. Pure functions only -- the Method wrapper is a later
task. Scalability core: the bridge-tree (2-edge-connected components of the current road graph,
contracted to a tree over one edge per bridge) is computed ONCE per greedy step, not once per
candidate -- a candidate's bridges-removed count is then a single bridge-tree shortest-path-length
lookup, not a fresh `nx.bridges` call.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import geopandas as gpd
import networkx as nx
import numpy as np
import shapely
from geopandas import GeoDataFrame
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from reblock.budget import _explode_segments, _noded_graph
from reblock.contracts import Block, Method, Proposal
from reblock.derivations import propose
from reblock.derive.access import STREET_TOL
from reblock.methods.arterial import _snap, _snap_graph
from reblock.methods.dijkstra import _boundary_graph

Node = tuple[float, float]


def _bridge_tree(g: nx.Graph) -> tuple[dict[Node, int], nx.Graph]:
    """Map each node of `g` to its 2-edge-connected-component (2ECC) id, and return the bridge-tree
    -- a tree over those component ids with one edge per bridge of `g`. `g`'s bridges, removed,
    leave exactly the 2ECCs as the remaining connected components (a node with no incident bridge
    survives in a multi-node 2ECC; a node touched only by bridges becomes a singleton 2ECC)."""
    bridge_edges = list(nx.bridges(g))
    h = g.copy()
    h.remove_edges_from(bridge_edges)
    comp_of: dict[Node, int] = {}
    for comp_id, comp in enumerate(nx.connected_components(h)):
        for n in comp:
            comp_of[n] = comp_id
    tree: nx.Graph = nx.Graph()
    tree.add_nodes_from(comp_of.values())
    for u, v in bridge_edges:
        tree.add_edge(comp_of[u], comp_of[v])
    return comp_of, tree


def bridges_removed(comp_of: dict[Node, int], tree: nx.Graph, u: Node, v: Node) -> int:
    """How many bridges a connector between road-graph nodes `u`, `v` would eliminate: the
    bridge-tree path length between their components (0 if either node is unknown, they are
    already in the same 2ECC, or -- distinct original graph components -- unreachable)."""
    if u not in comp_of or v not in comp_of:
        return 0
    cu, cv = comp_of[u], comp_of[v]
    if cu == cv:
        return 0
    try:
        return int(nx.shortest_path_length(tree, cu, cv))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return 0


def _nearest_node(nodes: list[Node], kdt: cKDTree, point: tuple[float, float], tol: float
                  ) -> Node | None:
    dist, idx = kdt.query(point)
    return nodes[int(idx)] if dist <= tol else None


def greedy_close_loops(base_roads: GeoDataFrame, streets: GeoDataFrame,
                       candidates: list[tuple[LineString, tuple[float, float],
                                              tuple[float, float]]],
                       *, budget_m: float | None, max_loops: int) -> list[LineString]:
    """Greedily add the candidate connector with the highest bridges-removed-per-metre, one at a
    time, stopping at `budget_m` cumulative added length, `max_loops` additions, or once the best
    remaining marginal gain is <= 0 -- whichever comes first. Each step recomputes the bridge-tree
    ONCE (over the current `base_roads` + already-added connectors, noded against `streets`), then
    scores every remaining candidate with a single `bridges_removed` lookup apiece; candidate
    endpoints are snapped to the nearest road-graph node within `STREET_TOL`. Returns the full road
    list: `base_roads`'s geometries plus whichever connectors were committed."""
    current: list[LineString] = cast("list[LineString]", list(base_roads.geometry))
    crs = base_roads.crs
    remaining = list(candidates)
    added_len = 0.0
    n_added = 0
    while remaining and n_added < max_loops:
        roads_gdf = gpd.GeoDataFrame(geometry=current, crs=crs)
        g = _noded_graph(roads_gdf, streets)
        if g.number_of_nodes() == 0:
            break
        comp_of, tree = _bridge_tree(g)
        nodes = list(g.nodes())
        kdt = cKDTree(np.array(nodes, dtype=float))
        best_idx = -1
        best_score = 0.0
        best_len = 0.0
        for i, (connector, u, v) in enumerate(remaining):
            nu = _nearest_node(nodes, kdt, u, STREET_TOL)
            nv = _nearest_node(nodes, kdt, v, STREET_TOL)
            if nu is None or nv is None:
                continue
            removed = bridges_removed(comp_of, tree, nu, nv)
            if removed <= 0:
                continue
            score = removed / max(connector.length, 1e-6)
            if score > best_score:
                best_score = score
                best_idx = i
                best_len = connector.length
        if best_idx < 0:
            break
        if budget_m is not None and added_len + best_len > budget_m:
            break
        connector, _u, _v = remaining.pop(best_idx)
        current.append(connector)
        added_len += best_len
        n_added += 1
    return current


def _knn_bounded_pairs(nodes: list[Node], kdt: cKDTree, search_radius_m: float, max_candidates: int
                       ) -> list[tuple[int, int]]:
    """Bound `query_pairs(search_radius_m)`'s pair volume to roughly `max_candidates`, WITHOUT
    truncating the pair list by distance: a hard truncation would keep only the globally shortest
    pairs -- tiny same-branch loops the geometric loop-floor in `loop_candidates` already rejects,
    wasting the whole budget on candidates that can never survive. Instead every node contributes
    its own up-to-`k` nearest OTHER nodes within `search_radius_m` (`k` sized so `n * k` lands near
    `max_candidates`), so distant/cross-branch pairs -- the ones that actually close big loops --
    stay represented regardless of local mesh density."""
    n = len(nodes)
    k = max(1, max_candidates // max(n, 1))
    kk = min(k + 1, n)                                     # +1: query includes each node itself
    dists, idxs = kdt.query(np.array(nodes, dtype=float), k=kk)
    if kk == 1:                                            # scipy squeezes the k=1 case to 1-D
        dists, idxs = dists[:, None], idxs[:, None]
    out: set[tuple[int, int]] = set()
    for i in range(n):
        for d, j in zip(dists[i], idxs[i], strict=True):
            jj = int(j)
            if jj == i or d > search_radius_m:
                continue
            out.add((i, jj) if i < jj else (jj, i))
    return sorted(out)


def loop_candidates(base_roads: GeoDataFrame, block: Block, *, search_radius_m: float,
                    min_loop_len_m: float, snap_lam: float, max_candidates: int | None = None
                    ) -> list[tuple[LineString, Node, Node]]:
    """Candidate loop-closing connectors: for every pair of `base_roads`' OWN node coords
    (`_explode_segments(base_roads)`'s unique endpoints -- the few-hundred base-road nodes, NOT the
    thousands of perimeter street nodes a region's `_noded_graph` would otherwise contribute, which
    made candidate generation explode) within `search_radius_m` (`cKDTree.query_pairs`), the
    gap-following BUILDABLE connector `_snap` routes along the block's parcel-boundary graph
    (`_snap_graph(_boundary_graph(block.parcels))`, precomputed once -- not the road graph itself,
    since a real connector must be a buildable path along parcel frontage). A pair survives only if
    its connector is realizable (non-None, length >= 1 m -- filters coincident-node snap artifacts)
    AND it would close a loop of GEOMETRIC perimeter >= `min_loop_len_m`: connector length + the
    road graph's OWN shortest-path distance between the two endpoints, weighted by euclidean edge
    length ("len") -- NEVER a hop count, which would score the same physical gap differently
    depending on how finely `base_roads` happens to be noded/subdivided. This floor's shortest path
    still runs over the FULL `_noded_graph(base_roads, block.streets)` (road nodes are a subset of
    it, so both endpoints always resolve). A pair whose endpoints fall in different components of
    the base road graph has no such path -- not loop-closing on this component -- and is dropped.
    Deduplicated by rounded (0.1 m) WKB, so numerically-identical connectors reached via different
    candidate node pairs collapse to one entry.

    `max_candidates` bounds the PAIR VOLUME `query_pairs` hands to the expensive per-pair `_snap`
    Dijkstra + shortest-path below -- both scale with pair count, so this is where the volume must
    be capped, before either runs. On a dense clearance mesh, `query_pairs(search_radius_m)` can
    return tens of thousands of pairs; past `max_candidates`, `_knn_bounded_pairs` swaps in a
    k-nearest-neighbours-per-node scheme that keeps the pair count near `max_candidates` while still
    representing cross-branch loops (see `_knn_bounded_pairs`). `None` (the default) leaves
    `query_pairs`' own radius cutoff as the only bound -- unchanged behavior for callers that don't
    opt in."""
    g = _noded_graph(base_roads, block.streets)
    for u, v in g.edges():
        g[u][v]["len"] = math.hypot(u[0] - v[0], u[1] - v[1])
    nodes = sorted({n for pair in _explode_segments(base_roads.geometry) for n in pair})
    if len(nodes) < 2:
        return []
    kdt = cKDTree(np.array(nodes, dtype=float))
    pairs = sorted(kdt.query_pairs(search_radius_m))
    if max_candidates is not None and len(pairs) > max_candidates:
        pairs = _knn_bounded_pairs(nodes, kdt, search_radius_m, max_candidates)
    sg = _snap_graph(_boundary_graph(block.parcels))
    seen: set[bytes] = set()
    out: list[tuple[LineString, Node, Node]] = []
    for i, j in pairs:
        u, v = nodes[i], nodes[j]
        connector = _snap(LineString([u, v]), sg, snap_lam)
        if connector is None or connector.length < 1.0:
            continue
        try:
            gap_len = float(nx.shortest_path_length(g, u, v, weight="len"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue                                       # unreachable on this component
        if connector.length + gap_len < min_loop_len_m:
            continue
        key = shapely.to_wkb(shapely.set_precision(connector, 0.1))
        if key in seen:
            continue
        seen.add(key)
        out.append((connector, u, v))
    return out


@dataclass
class LoopClosureRefiner:
    """Method wrapper composing `loop_candidates` + `greedy_close_loops` behind the `Method.propose`
    `prior` seam: refines a base method's (e.g. `ClearanceReblocker`) tree-ish proposal by adding
    bridge-removing loop-closing connectors, greedily, up to `budget_m` added length / `max_loops`.
    NOT frozen: `base` is an arbitrary (unhashable) `Method`, and `@dataclass(frozen=True)` would
    auto-generate a `__hash__` over it -- mirrors the sibling `ClearanceReblocker`."""

    base: Method
    budget_m: float | None = 200.0
    max_loops: int = 20
    min_loop_len_m: float = 40.0
    # 20 m (down from an initial 60 m default): on a dense region-scale clearance mesh, 60 m put
    # `loop_candidates` at ~28k pairs / ~155s wall-clock end to end -- 20 m lands in the low
    # hundreds-to-low-thousands of pairs (single-digit-to-teens seconds) while still lifting
    # commute_ratio well above the bare clearance base (region-measured, see loop_closure.py's
    # module docstring companion in the task report -- ρ 0.0 -> >0.10 preserved at this radius).
    search_radius_m: float = 45.0
    snap_lam: float = 2.0
    # A second, independent bound on top of `search_radius_m`: even a modest radius can still
    # explode on a denser-than-tested mesh (finer substrate, bigger region), since pair count grows
    # with LOCAL node density, not just radius. `max_candidates` caps `loop_candidates`' pair volume
    # via `_knn_bounded_pairs` regardless of density -- a belt-and-suspenders guard, not the primary
    # lever (search_radius_m's cut is what keeps the common case fast).
    max_candidates: int | None = 4000

    @property
    def identity(self) -> tuple[object, ...] | None:
        # An uncacheable base (identity None) makes the whole refiner uncacheable -- propagate the
        # None up so derive() bypasses the memoized propose, matching ClearanceReblocker's
        # uncacheable-substrate handling.
        bid = getattr(self.base, "identity", None)
        if bid is None:
            return None
        return ("loop_closure", bid, self.budget_m, self.max_loops, self.min_loop_len_m,
                self.search_radius_m, self.snap_lam, self.max_candidates)

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        # `prior`, when given, IS the base proposal to refine -- skip recomputing/re-fetching it
        # (and, in particular, never call self.base.propose again).
        base_prop = prior if prior is not None else propose(self.base, block)
        base_roads = (base_prop.roads if base_prop.roads is not None
                     else gpd.GeoDataFrame(geometry=[], crs=block.crs))
        candidates = loop_candidates(
            base_roads, block, search_radius_m=self.search_radius_m,
            min_loop_len_m=self.min_loop_len_m, snap_lam=self.snap_lam,
            max_candidates=self.max_candidates)
        all_roads = greedy_close_loops(
            base_roads, block.streets, candidates,
            budget_m=self.budget_m, max_loops=self.max_loops)
        roads = gpd.GeoDataFrame(geometry=all_roads, crs=block.crs)
        pid = f"loop_closure:{base_prop.proposal_id}:b{self.budget_m}:ml{self.max_loops}"
        return Proposal(
            block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
            proposal_id=pid, method="loop_closure",
            params={"budget_m": self.budget_m, "max_loops": self.max_loops,
                    "min_loop_len_m": self.min_loop_len_m, "search_radius_m": self.search_radius_m,
                    "snap_lam": self.snap_lam, "max_candidates": self.max_candidates,
                    "n_added": len(all_roads) - len(base_roads)},
            block_identity=base_prop.block_identity)
