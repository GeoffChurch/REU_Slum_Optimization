"""Bridge-tree greedy loop-closure engine. Pure functions only -- the Method wrapper is a later
task. Scalability core: the bridge-tree (2-edge-connected components of the current road graph,
contracted to a tree over one edge per bridge) is computed ONCE per greedy step, not once per
candidate -- a candidate's bridges-removed count is then a single bridge-tree shortest-path-length
lookup, not a fresh `nx.bridges` call.
"""
from __future__ import annotations

import math
from collections.abc import Hashable
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
                       *, budget_m: float | None, max_loops: int,
                       min_bridges_per_m: float = 0.0) -> list[LineString]:
    """Greedily add the candidate connector with the highest bridges-removed-per-metre, one at a
    time, stopping at `budget_m` cumulative added length, `max_loops` additions, or once the best
    remaining marginal gain (bridges removed per metre) drops below `max(min_bridges_per_m, 1e-9)`
    -- whichever comes first. `min_bridges_per_m=0.0` (the default) preserves the original
    stop-when-nothing-left-to-gain behavior; a positive value is a diminishing-returns early stop
    that bails out once the best remaining loop is inefficient (e.g. removes fewer than 1 bridge
    per 100 m -> gain < 0.01), even with budget left. Each step recomputes the bridge-tree ONCE
    (over the current `base_roads` + already-added connectors, noded against `streets`), then
    scores every remaining candidate with a single `bridges_removed` lookup apiece; candidate
    endpoints are snapped to the nearest road-graph node within `STREET_TOL`. Returns the full road
    list: `base_roads`'s geometries plus whichever connectors were committed."""
    current: list[LineString] = cast("list[LineString]", list(base_roads.geometry))
    crs = base_roads.crs
    remaining = list(candidates)
    added_len = 0.0
    n_added = 0
    min_gain = max(min_bridges_per_m, 1e-9)
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
        if best_idx < 0 or best_score < min_gain:
            break
        if budget_m is not None and added_len + best_len > budget_m:
            break
        connector, _u, _v = remaining.pop(best_idx)
        current.append(connector)
        added_len += best_len
        n_added += 1
    return current


def _subsample_pairs(pairs: list[tuple[int, int]], max_candidates: int
                     ) -> list[tuple[int, int]]:
    """Uniformly subsample the index-sorted `pairs` to ~`max_candidates`, bounding the pair volume
    handed to `loop_candidates`' expensive per-pair `_snap` + shortest-path WITHOUT biasing which
    pairs survive. The previous nearest-k scheme kept each node's CLOSEST neighbours -- exactly the
    short, low-perimeter pairs the `min_loop_len_m` floor rejects -- so on a dense mesh (many nodes
    -> k collapses to 1-2) it starved the candidate pool to near-zero valid loop-closers (an
    11k-parcel region fell to 63 candidates / commute_ratio 0.06). A uniform stride over the
    index-sorted list preserves the straight-line-distance distribution, so the fraction of pairs
    that clear the loop floor is retained: a cap of C yields ~C valid candidates, not a handful
    (same region -> ~1300 candidates / commute_ratio ~0.50). `pairs` is assumed sorted (as
    `sorted(query_pairs(...))` returns), so the stride samples evenly across the node-index
    space. A non-positive `max_candidates` is clamped to 1 (never a division by zero or a
    reversed-stride slice)."""
    cap = max(max_candidates, 1)
    if len(pairs) <= cap:
        return pairs
    stride = math.ceil(len(pairs) / cap)
    return pairs[::stride]


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
    be capped, before either runs. On a dense clearance mesh `query_pairs(search_radius_m)` can
    return tens of thousands of pairs; past `max_candidates`, `_subsample_pairs` takes a uniform
    stride over the index-sorted pairs, preserving the distance distribution so real
    (floor-clearing) loop-closers survive the cap (see `_subsample_pairs`). `None` (the default)
    leaves `query_pairs`' own radius cutoff as the only bound -- unchanged behavior for callers
    that don't opt in."""
    g = _noded_graph(base_roads, block.streets)
    for u, v in g.edges():
        g[u][v]["len"] = math.hypot(u[0] - v[0], u[1] - v[1])
    nodes = sorted({n for pair in _explode_segments(base_roads.geometry) for n in pair})
    if len(nodes) < 2:
        return []
    kdt = cKDTree(np.array(nodes, dtype=float))
    pairs = sorted(kdt.query_pairs(search_radius_m))
    if max_candidates is not None and len(pairs) > max_candidates:
        pairs = _subsample_pairs(pairs, max_candidates)
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


@dataclass(frozen=True)
class LoopClosureIdentity:
    """Cache-key identity for LoopClosureRefiner. The dataclass type discriminates the refiner (no
    string tag). `base` holds the wrapped base method's own identity verbatim (whatever it returns);
    frozen -> hashable, usable as an L1 dict key and joblib-picklable."""
    base: Hashable                 # the nested base Method.identity (not converted in this pass)
    budget_frac: float
    min_bridges_per_m: float
    max_loops: int
    min_loop_len_m: float
    search_radius_m: float
    snap_lam: float
    max_candidates: int | None


@dataclass
class LoopClosureRefiner:
    """Method wrapper composing `loop_candidates` + `greedy_close_loops` behind the `Method.propose`
    `prior` seam: refines a base method's (e.g. `ClearanceReblocker`) tree-ish proposal by adding
    bridge-removing loop-closing connectors, greedily, up to `budget_frac`'s share of the base
    proposal's road length / `max_loops`, with a `min_bridges_per_m` diminishing-returns early
    stop. NOT frozen: `base` is an arbitrary (unhashable) `Method`, and `@dataclass(frozen=True)`
    would auto-generate a `__hash__` over it -- mirrors the sibling `ClearanceReblocker`."""

    base: Method
    budget_frac: float = 0.12
    # Loops may add up to this FRACTION of the base proposal's total road length, region-adaptive
    # in place of a fixed absolute budget_m: on an 11k-parcel region a 200 m absolute budget was
    # ~1% of the base road and added negligible redundancy (commute_ratio 0.009). Calibrated on a
    # 1724-parcel block (base road 8622 m): 0.05 -> ρ 0.31, 0.10 -> 0.375, 0.12 -> ~0.39, 0.15 ->
    # 0.40 (diminishing returns past ~0.12 -- the knee), all with external held ~0.949 and
    # ~8-11 s runtime.
    min_bridges_per_m: float = 0.01
    # Early-stop threshold on the greedy ranking objective (bridges removed per metre): once the
    # best remaining candidate's efficiency falls below this, stop adding loops even if budget
    # remains -- a diminishing-returns guard so a large budget_frac doesn't get spent on
    # increasingly marginal connectors. 0.01 = 1 bridge per 100 m.
    max_loops: int = 400
    # A high safety cap, not the real bound -- budget_frac (via the resolved budget_m) and
    # min_bridges_per_m now do the real work of stopping the greedy loop.
    min_loop_len_m: float = 40.0
    # 45 m: sufficient for block-scale bases with the uniform `max_candidates` cap, which bounds the
    # expensive per-pair snap volume regardless of mesh density (see `_subsample_pairs`).
    search_radius_m: float = 45.0
    snap_lam: float = 2.0
    # A second, independent bound on top of `search_radius_m`: even a modest radius can still
    # explode on a denser-than-tested mesh (finer substrate, bigger region), since pair count grows
    # with LOCAL node density, not just radius. `max_candidates` caps `loop_candidates`' pair volume
    # via `_subsample_pairs` regardless of density -- a belt-and-suspenders guard, not the primary
    # lever (search_radius_m's cut is what keeps the common case fast).
    max_candidates: int | None = 1500
    # Uniform-subsample cap on `loop_candidates`' pair volume (see `_subsample_pairs`): bounds the
    # per-pair `_snap` cost regardless of mesh density. 1500 is the commute_ratio PLATEAU -- caps
    # 1500/2500/4000 all reach ~the same ρ on an 11k-parcel region (budget-bound past ~1300 valid
    # candidates), and 1500 is ~6x faster than uncapped. search_radius_m stays 45 by default (that
    # already recovers large blocks with the uniform cap); the sparse dt=3 REGION base overrides to
    # 60 (see scripts/gen_multiblock_example.py).

    @property
    def identity(self) -> LoopClosureIdentity | None:
        # An uncacheable base (identity None) makes the whole refiner uncacheable -- propagate the
        # None up so derive() bypasses the memoized propose, matching ClearanceReblocker's
        # uncacheable-substrate handling.
        bid = getattr(self.base, "identity", None)
        if bid is None:
            return None
        return LoopClosureIdentity(
            base=bid, budget_frac=self.budget_frac, min_bridges_per_m=self.min_bridges_per_m,
            max_loops=self.max_loops, min_loop_len_m=self.min_loop_len_m,
            search_radius_m=self.search_radius_m, snap_lam=self.snap_lam,
            max_candidates=self.max_candidates)

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        # `prior`, when given, IS the base proposal to refine -- skip recomputing/re-fetching it
        # (and, in particular, never call self.base.propose again).
        base_prop = prior if prior is not None else propose(self.base, block)
        base_roads = (base_prop.roads if base_prop.roads is not None
                     else gpd.GeoDataFrame(geometry=[], crs=block.crs))
        # Region-adaptive budget: a fixed absolute length is block-scale and vanishes on a big
        # region's base road (see budget_frac's field comment); scaling to the base proposal's own
        # road length keeps the added redundancy proportional regardless of region size.
        base_len = float(base_roads.geometry.length.sum())
        budget_m = self.budget_frac * base_len
        candidates = loop_candidates(
            base_roads, block, search_radius_m=self.search_radius_m,
            min_loop_len_m=self.min_loop_len_m, snap_lam=self.snap_lam,
            max_candidates=self.max_candidates)
        all_roads = greedy_close_loops(
            base_roads, block.streets, candidates,
            budget_m=budget_m, max_loops=self.max_loops,
            min_bridges_per_m=self.min_bridges_per_m)
        roads = gpd.GeoDataFrame(geometry=all_roads, crs=block.crs)
        pid = f"loop_closure:{base_prop.proposal_id}:bf{self.budget_frac}:ml{self.max_loops}"
        return Proposal(
            block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
            proposal_id=pid, method="loop_closure",
            params={"budget_frac": self.budget_frac, "min_bridges_per_m": self.min_bridges_per_m,
                    "max_loops": self.max_loops,
                    "min_loop_len_m": self.min_loop_len_m, "search_radius_m": self.search_radius_m,
                    "snap_lam": self.snap_lam, "max_candidates": self.max_candidates,
                    "n_added": len(all_roads) - len(base_roads)},
            block_identity=base_prop.block_identity)
