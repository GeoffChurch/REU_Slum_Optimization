"""Budget-sweep and scoring primitives for reblocking. `_sweep` adds a method's roads
incrementally in `street_first_ordered` order -- drainage-descending, with each road's connectors
to the street bought first -- and samples a value at each budget, yielding a `Curve` of value
vs cost (cumulative added road length, m); `displacement_curve` and (in `permeability.py`)
`permeability_curve` ride it, and the matched-displacement / matched-permeability lens
truncations read the resulting index-aligned curves. Also holds the retained road/parcel scoring
primitives (`road_drainage`, `building_radii`, `displacement`, `repulsion`, `access_burden`,
`network_efficiency` + `_BlockScoringContext`).
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TypeVar, cast

import networkx as nx
import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.permeability import PermeabilityParams, egress_power, permeability

_Node = tuple[float, float]      # an _rnd-snapped (x, y) graph node
_Pair = tuple[_Node, _Node]      # an undirected edge as its two endpoints


def _rnd(c: tuple[float, ...]) -> tuple[float, float]:
    return (round(c[0], 2), round(c[1], 2))


def building_radii(building_points: GeoDataFrame, corridor_m: float) -> NDArray[np.float64]:
    """Per-building disk radius = HALF the nearest-neighbor distance among the building points (the
    fair, non-overlapping 'as big as possible' footprint bound). Fewer than 2 points -> no neighbor,
    so fall back to `corridor_m`. Coincident points get radius 0 (handled by `displacement`)."""
    n = len(building_points)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n < 2:
        return np.full(n, float(corridor_m), dtype=np.float64)
    xy = np.column_stack([building_points.geometry.x.to_numpy(),
                          building_points.geometry.y.to_numpy()])
    dist, _ = cKDTree(xy).query(xy, k=2)     # k=2: self (0) + nearest other
    return (dist[:, 1] * 0.5).astype(np.float64)


def displacement(building_points: GeoDataFrame, radii: NDArray[np.float64],
                 roads: GeoDataFrame, corridor_m: float) -> float:
    """Extent-aware expected homes displaced: each building is a disk (radius `radii[i]`); its
    contribution is the probability the road corridor grazes it under a uniform size prior,
    c_i = max(0, 1 - d_i/r_i), d_i = distance from the point to roads.buffer(corridor_m). r_i = 0
    (coincident points) counts iff d_i = 0. Returns Sum c_i; 0 with no roads or no points."""
    n = len(building_points)
    if n == 0 or roads is None or len(roads) == 0:
        return 0.0
    corridor = roads.geometry.buffer(corridor_m).union_all()
    d = building_points.geometry.distance(corridor).to_numpy()
    r = np.asarray(radii, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(r > 0.0, 1.0 - d / r, np.where(d <= 0.0, 1.0, 0.0))
    return float(np.clip(c, 0.0, 1.0).sum())


def repulsion(building_points: GeoDataFrame, radii: NDArray[np.float64],
              geom: BaseGeometry) -> float:
    """Soft per-road intrusion cost: a road's OWN proximity to the building field, summed over
    building points as the quadratic-tail kernel r^2/(r^2 + d^2) (d = point-to-road distance,
    r = the building's disk radius from `building_radii`). > 0 whenever any building has r>0
    (the tail never reaches zero) so no road is 'free' -- unlike `displacement`, whose hard
    0-beyond-r cutoff makes gap-hugging roads free and degenerate. Depends ONLY on `geom` and
    the fixed building field (not on other committed roads), so it is CONSTANT per candidate ->
    a well-behaved, CELF-safe greedy cost denominator (marginal displacement is not). r==0
    (coincident points): 1.0 if the road touches the point (d<=0) else 0.0, matching
    displacement's r==0 handling.
    0.0 with no building points."""
    n = len(building_points)
    if n == 0:
        return 0.0
    d = building_points.geometry.distance(geom).to_numpy()
    r = np.asarray(radii, dtype=np.float64)
    r2 = r * r
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(r > 0.0, r2 / (r2 + d * d), np.where(d <= 0.0, 1.0, 0.0))
    return float(np.clip(k, 0.0, 1.0).sum())


def access_burden(depths: pd.Series) -> float:
    """Sigma depth^2 -- q=2 severity-weighted access burden (kblock parcels = one building each)."""
    return float((depths.astype("float64") ** 2).sum())


@dataclass(frozen=True)
class _RoadNet:
    """The snap-planarized road graph, with each road's own nodes and its distance to the street.

    Shared by `road_drainage` and `street_first_ordered` so the two can never disagree about what
    the road network is. Nodes are `_rnd`-snapped segment endpoints -- deliberately NOT
    `unary_union`-noded, because that re-nodes geometry into vertices no `_rnd` key matches.
    """

    graph: nx.Graph
    edge_row: dict[frozenset[_Node], int]
    road_nodes: list[list[_Node]]
    street_nodes: list[_Node]

    def street_distance(self) -> tuple[dict[_Node, float], dict[_Node, list[_Node]]]:
        """Per-node network distance from the street, and the path back to it."""
        if not self.street_nodes:
            return {}, {}
        d, p = nx.multi_source_dijkstra(self.graph, sorted(self.street_nodes))
        return cast(dict[_Node, float], d), cast(dict[_Node, list[_Node]], p)


def _road_net(block: Block, roads: GeoDataFrame, tol: float) -> _RoadNet:
    g: nx.Graph = nx.Graph()
    edge_row: dict[frozenset[_Node], int] = {}
    road_nodes: list[list[_Node]] = []
    for i, geom in enumerate(roads.geometry):
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]   # explode Multi*
        mine: list[_Node] = []
        for part in parts:
            cs = list(part.coords)
            for a, b in zip(cs, cs[1:], strict=False):
                na, nb = _rnd(a), _rnd(b)
                if na != nb:
                    g.add_edge(na, nb, weight=Point(na).distance(Point(nb)))
                    edge_row[frozenset((na, nb))] = i
                    mine += [na, nb]
        road_nodes.append(mine)
    street = unary_union(list(block.streets.geometry))
    snodes = [node for node in g.nodes if Point(node).distance(street) <= tol]
    return _RoadNet(graph=g, edge_row=edge_row, road_nodes=road_nodes, street_nodes=snodes)


def road_drainage(block: Block, roads: GeoDataFrame, *, tol: float = STREET_TOL) -> list[int]:
    """Per-road PARCEL count: build a graph from the road segments, route each parcel to the street
    through it, and count how many parcels' routes use each road. Uniform across methods.

    Each parcel contributes AT MOST 1 to any one road, however many of that road's segments its
    route happens to traverse. Summing per-segment instead inflates vertex-dense roads -- a route
    crossing two segments of one road scored it 2 -- which is a count of geometry, not of traffic,
    and it biases every drainage-based decision toward finely-vertexed roads. Two parcels served by
    one road must read 2, whether that road is drawn with two vertices or twenty.
    """
    n = len(roads)
    if n == 0:
        return []
    net = _road_net(block, roads, tol)
    g, edge_row = net.graph, net.edge_row
    if not net.street_nodes:
        return [0] * n
    dist, paths = net.street_distance()
    nodes = list(g.nodes)
    tree = STRtree([Point(node) for node in nodes])
    counts: dict[int, int] = defaultdict(int)
    for geom in block.parcels.geometry:
        reach = [nodes[j] for j in tree.query(geom, predicate="dwithin", distance=tol)
                 if nodes[j] in dist]
        if not reach:
            continue
        entry = min(reach, key=lambda node: (dist[node], node))
        used = {row for a, b in zip(paths[entry], paths[entry][1:], strict=False)
                if (row := edge_row.get(frozenset((a, b)))) is not None}
        for row in used:
            counts[row] += 1
    return [counts.get(i, 0) for i in range(n)]


def _line_entries(geoms: list[BaseGeometry], reps: list[Point], edges: list[LineString],
                  tree: STRtree, tol: float
                  ) -> tuple[list[tuple[float, float] | None],
                             dict[int, list[tuple[float, tuple[float, float]]]]]:
    """Each parcel's entry is the nearest POINT on a road/street edge within `tol` of the parcel
    -- NOT the nearest graph VERTEX. The old nearest-vertex rule undercounted
    sparse straight chords (aspirational arterials): a 2-point chord has only its endpoints as
    vertices, so a parcel abreast of its middle snapped to nothing and the chord scored ~0,
    spuriously ranking sparse through-roads far below frontage-dense ones. Query by the parcel
    GEOMETRY, not its interior rep point:
    a parcel's roads run along its boundary, so on real (meters-wide) parcels the rep point is far
    from every edge and would spuriously find none. For each parcel take the nearest edge within
    `tol` of it, project the rep point onto that edge (`P = edge.interpolate(edge.project(rep))` =
    the boundary point where the parcel meets the road), `_rnd(P)` is the entry node, and record P
    against its edge so the CSR build (`_build_csr`) injects it as a colinear split. Returns
    (per-parcel entry node or None, {edge index -> [(proj-distance-along-edge, _rnd(P))]}).
    Deterministic (nearest edge broken by index,
    splits sorted by proj). A parcel with no edge within `tol` is unreached this round (None)."""
    entry: list[tuple[float, float] | None] = []
    splits: dict[int, list[tuple[float, tuple[float, float]]]] = defaultdict(list)
    for geom, pt in zip(geoms, reps, strict=True):
        cand = tree.query(geom, predicate="dwithin", distance=tol)
        if len(cand) == 0:
            entry.append(None)
            continue
        j = min((int(c) for c in cand), key=lambda c: (float(geom.distance(edges[c])), c))
        ls = edges[j]
        p = _rnd(ls.interpolate(ls.project(pt)).coords[0])
        entry.append(p)
        splits[j].append((ls.project(pt), p))
    return entry, splits


def _sampled_efficiency_core(csr: csr_matrix, node_index: dict[_Node, int],
                             entry: list[_Node | None], sources: list[int],
                             rep_xy: np.ndarray, src_euclid: np.ndarray) -> tuple[float, float]:
    """(E, directness) of the DOOR-TO-DOOR trip between every parcel pair, over the graph passed as
    a symmetric CSR + `node_index`. Pure numeric core: the CSR, the per-parcel `entry` nodes, the K
    sampled `sources`, the frozen `rep_xy` (N,2) and `src_euclid` (K,N) are all supplied by the
    caller (`_BlockScoringContext`), so none of the per-block constants are rebuilt per candidate.

    The effective distance is the whole journey `d = leg_i + netdist(entry_i, entry_j) + leg_j`,
    where `leg_k = euclid(rep_k, entry_k)` is the last-mile walk from a parcel to where it joins the
    road -- NOT `netdist` alone. With the walk legs included, `euclid(rep_i, rep_j) <= d` by the
    triangle inequality, so directness = mean(euclid/d) is a circuity ratio bounded in [0, 1];
    E = mean(1/d). Averaged over the FIXED all-parcel pair set (unreached pairs -- an entry missing,
    or absent/unreachable in the graph -- contribute 0); (0, 0) if there are no pairs. ONE batched
    `scipy.sparse.csgraph.dijkstra` over the CSR replaces the K per-source
    `nx.single_source_dijkstra_path_length` calls; the O(K*N) leg/euclid/accumulation arithmetic is
    masked numpy sums."""
    n = len(entry)
    if n == 0 or not sources:
        return 0.0, 0.0
    entry_xy = np.array([[np.nan, np.nan] if e is None else [e[0], e[1]] for e in entry],
                        dtype=np.float64)
    legs = np.hypot(rep_xy[:, 0] - entry_xy[:, 0], rep_xy[:, 1] - entry_xy[:, 1])  # NaN if e None
    # Column (per-target) node row in the CSR, or -1 if this parcel has no entry / its entry
    # node isn't in the graph at all -- explicit sentinel, NEVER a truthiness check, so a genuine
    # netdist == 0 (coincident entries) never gets mistaken for "missing".
    entry_row = np.array([node_index.get(e, -1) if e is not None else -1 for e in entry],
                         dtype=np.int64)
    reachable = entry_row >= 0

    # Only sources whose OWN entry exists in the graph can seed a Dijkstra run; the rest stay
    # all-inf (matching the prior per-source `dist = {} if src is None or src not in g`).
    valid_rows = [i for i, si in enumerate(sources)
                 if entry[si] is not None and entry[si] in node_index]
    dist_mat = np.full((len(sources), csr.shape[0]), np.inf, dtype=np.float64)
    if valid_rows:
        valid_indices = [node_index[cast(_Node, entry[sources[i]])] for i in valid_rows]
        dist_mat[valid_rows] = dijkstra(csr, directed=False, indices=valid_indices)

    inv_sum = dir_sum = 0.0
    pairs = 0
    for i, si in enumerate(sources):
        pairs += n - 1                            # all j != si, unchanged regardless of validity
        nd = np.full(n, np.inf, dtype=np.float64)
        nd[reachable] = dist_mat[i, entry_row[reachable]]
        d = legs[si] + nd + legs                  # door-to-door: walk + drive + walk
        mask = (entry[si] is not None) & np.isfinite(nd) & np.isfinite(legs) & (d > 0)
        mask[si] = False                          # exclude the self pair (j == si)
        inv_sum += float(np.sum(1.0 / d[mask]))
        dir_sum += float(np.sum(src_euclid[i, mask] / d[mask]))
    if pairs == 0:
        return 0.0, 0.0
    return inv_sum / pairs, dir_sum / pairs


def _explode_segments(geoms: Iterable[BaseGeometry]) -> list[_Pair]:
    """Explode geometries to `_rnd`-snapped, non-degenerate, first-seen-deduplicated undirected
    segment endpoint pairs -- the segment extraction the old nx graph builder fed `nx.add_edge`,
    minus the nx graph. Multi* geometries are exploded (a holed/courtyard block's streets)."""
    seen: set[frozenset[_Node]] = set()
    out: list[_Pair] = []
    for geom in geoms:
        if geom is None:
            continue
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for part in parts:
            cs = list(part.coords)
            for a, b in zip(cs, cs[1:], strict=False):
                na, nb = _rnd(a), _rnd(b)
                if na == nb:
                    continue
                key = frozenset((na, nb))
                if key not in seen:
                    seen.add(key)
                    out.append((na, nb))
    return out


def _edges_in_nx_order(road_segs: list[_Pair], street_segs: list[_Pair]) -> list[_Pair]:
    """The combined road+street edge set in the exact order `networkx.Graph.edges()` yields it for
    a graph built roads-first-then-streets (as the old nx graph builder did): nodes ordered by
    first appearance across the roads-then-streets scan, each node's neighbours in first-seen
    order, and every undirected edge emitted at its first-inserted endpoint. This reproduces
    `_edge_lines(g)`'s order WITHOUT building the nx graph -- and the order is load-bearing,
    because it sets `_line_entries`'s edge-index tie-break, so it must match to keep entries
    (hence the metric) identical on exact-distance ties (which grid geometry produces)."""
    node_pos: dict[_Node, int] = {}
    node_order: list[_Node] = []
    adj: dict[_Node, list[_Node]] = {}
    seen: set[frozenset[_Node]] = set()
    for na, nb in [*road_segs, *street_segs]:
        for nd in (na, nb):
            if nd not in node_pos:
                node_pos[nd] = len(node_order)
                node_order.append(nd)
                adj[nd] = []
        key = frozenset((na, nb))
        if key in seen:
            continue
        seen.add(key)
        adj[na].append(nb)
        adj[nb].append(na)
    order: list[_Pair] = []
    done: set[_Node] = set()
    for nd in node_order:
        for nbr in adj[nd]:
            if nbr not in done:
                order.append((nd, nbr))
        done.add(nd)
    return order


def _build_csr(base_pairs: Iterable[_Pair],
               splits: dict[_Pair, list[tuple[float, _Node]]]
               ) -> tuple[csr_matrix, dict[_Node, int]]:
    """Assemble a symmetric CSR + deterministic `node_index` from undirected `base_pairs` plus
    colinear split-chain injections, applying the exact-graph-parity rule (design risk I3): edges
    live in a dict keyed by the unordered endpoint pair with LAST-WRITE-WINS weights -- a repeated
    pair is ONE edge, never scipy's silent duplicate-COO summation that would double its weight and
    corrupt the distance -- and injecting a split chain DELETES its parent pair first (exactly as
    `_split_graph` did). `splits` maps a parent edge's endpoints `(u, v)` to its
    `[(proj-along-edge, split point)]` list; a parent ABSENT from `base_pairs` (a road not in this
    prefix) is skipped, so its parcels' entry nodes stay absent from the graph and contribute 0 --
    keeping the frozen-entry prefix sweep monotone."""
    edges: dict[frozenset[_Node], _Pair] = {}
    for na, nb in base_pairs:
        edges[frozenset((na, nb))] = (na, nb)
    for (u, v), pts in splits.items():
        key = frozenset((u, v))
        if key not in edges:
            continue
        del edges[key]
        chain: list[_Node] = [u]
        for _proj, p in sorted(pts):
            if p != chain[-1]:
                chain.append(p)
        if v != chain[-1]:
            chain.append(v)
        for a, b in zip(chain, chain[1:], strict=False):
            if a != b:
                edges[frozenset((a, b))] = (a, b)
    nodes = sorted({nd for pair in edges.values() for nd in pair})
    node_index = {nd: i for i, nd in enumerate(nodes)}
    ends = list(edges.values())
    if ends:
        a_xy = np.array([e[0] for e in ends], dtype=np.float64)
        b_xy = np.array([e[1] for e in ends], dtype=np.float64)
        w = np.hypot(a_xy[:, 0] - b_xy[:, 0], a_xy[:, 1] - b_xy[:, 1])
        ri = np.array([node_index[e[0]] for e in ends], dtype=np.int64)
        ci = np.array([node_index[e[1]] for e in ends], dtype=np.int64)
        rows = np.concatenate([ri, ci])
        cols = np.concatenate([ci, ri])
        data = np.concatenate([w, w])
    else:
        rows = cols = np.zeros(0, dtype=np.int64)
        data = np.zeros(0, dtype=np.float64)
    csr = csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
    return csr, node_index


def _reproject_hits(geoms_arr: np.ndarray, reps_arr: np.ndarray, lines: np.ndarray,
                    hits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray]:
    """Vectorize `_line_entries`'s per-hit geometry for a batched STRtree `dwithin` result. `hits`
    is the STRtree query's 2xM array (row 0 = parcel index, row 1 = edge index into `lines`).
    Returns per-hit arrays `(parcel_idx, edge_idx, distance, projection-along-edge, foot_xy)`:
    `distance = parcel.distance(edge)`, `foot_xy = edge.interpolate(edge.project(rep))` (the entry
    point), computed with shapely's own vectorized ufuncs (bit-identical to the scalar
    `Geometry.distance`/`.project`/`.interpolate` `_line_entries` calls) in a handful of C calls
    instead of a Python loop over every parcel x edge pair."""
    pidx, eidx = hits[0], hits[1]
    edge_geoms = lines[eidx]
    dist = shapely.distance(geoms_arr[pidx], edge_geoms)
    proj = shapely.line_locate_point(edge_geoms, reps_arr[pidx])
    foot_xy = shapely.get_coordinates(shapely.line_interpolate_point(edge_geoms, proj))
    return pidx, eidx, dist, proj, foot_xy


class _BlockScoringContext:
    """Per-block scoring constants frozen ONCE, shared by `network_efficiency` and the greedy
    arterial loop. Freezing lifts the representative points, sampled sources, the K*N euclidean
    matrix and the street edge geometry out of the per-candidate hot path (they were previously
    rebuilt for all ~7k candidates). `.score(roads)` re-derives entries against streets + `roads`
    and matches `network_efficiency` exactly."""

    def __init__(self, block: Block, *, k: int = 40, tol: float = STREET_TOL) -> None:
        self.block = block
        self.tol = tol
        self.geoms: list[BaseGeometry] = list(block.parcels.geometry)
        n = len(self.geoms)
        self.n = n
        self.reps: list[Point] = [gm.representative_point() for gm in self.geoms]
        self.rep_xy = (np.array([[p.x, p.y] for p in self.reps], dtype=np.float64)
                       if n else np.zeros((0, 2), dtype=np.float64))
        step = max(1, n // k)
        self.sources: list[int] = list(range(n))[::step][:k]
        if n >= 2:
            src_xy = self.rep_xy[self.sources]
            self.src_euclid = np.hypot(src_xy[:, 0, None] - self.rep_xy[:, 0],
                                       src_xy[:, 1, None] - self.rep_xy[:, 1])
        else:
            self.src_euclid = np.zeros((len(self.sources), n), dtype=np.float64)

        # Frozen UN-split street edge geometry (deduped segments in scan order). The greedy's
        # per-step `_StepContext` rebuilds its streets-∪-committed graph and entry base from these
        # un-split segments on each commit -- deliberately NOT a pre-split street CSR: splitting the
        # streets at streets-only projections bakes in an `_rnd`-rounded split node that goes stale
        # (perturbing distances above the 1e-9 tolerance) the moment a parcel's entry later moves
        # onto a committed road, so no such CSR is frozen here (task 4 review #5).
        self.street_segs: list[_Pair] = _explode_segments(block.streets.geometry)
        # Parcel geometry + rep points as object arrays, for the vectorized (batched STRtree +
        # shapely-ufunc) entry reprojection `_StepContext` runs per candidate.
        self.geoms_arr: np.ndarray = np.array(self.geoms, dtype=object)
        self.reps_arr: np.ndarray = np.array(self.reps, dtype=object)

    def step(self, committed: GeoDataFrame | None) -> _StepContext:
        """A per-greedy-step scoring context over streets ∪ `committed` (rebuilt on each commit,
        ~max_roads times -- cheap), off which `score_candidate(real)` scores each trial road
        incrementally. `committed` is the PLANARIZED committed road set (`_planarize(committed)`).
        See `_StepContext`."""
        return _StepContext(self, committed)

    def _derive_entries(self, roads: GeoDataFrame | None
                        ) -> tuple[list[_Node | None], dict[_Pair, list[tuple[float, _Node]]],
                                   list[_Pair]]:
        """(entry, splits_by_parent_pair, edge_pairs) over streets + `roads`, re-deriving entries
        with `_line_entries` against the combined edge set in `networkx.Graph.edges()` order -- so
        the edge-index tie-break (hence the entries) matches `network_efficiency` exactly. `splits`
        is re-keyed from edge index to the parent edge's endpoint pair so the CSR build (and the
        frozen prefix sweep) need only the pair, not the full edge list."""
        road_segs = (_explode_segments(roads.geometry)
                     if roads is not None and len(roads) else [])
        edge_pairs = _edges_in_nx_order(road_segs, self.street_segs)
        if not edge_pairs:
            return [None] * self.n, {}, edge_pairs
        edge_lines = [LineString([a, b]) for a, b in edge_pairs]
        entry, splits = _line_entries(self.geoms, self.reps, edge_lines,
                                     STRtree(edge_lines), self.tol)
        splits_uv = {edge_pairs[j]: pts for j, pts in splits.items()}
        return entry, splits_uv, edge_pairs

    def score(self, roads: GeoDataFrame | None) -> tuple[float, float]:
        """(E, directness) for streets + `roads`, re-deriving entries against the combined graph
        exactly as `network_efficiency` does (FULL re-derivation, no incremental shortcut)."""
        if self.n < 2:
            return 0.0, 0.0
        entry, splits_uv, edge_pairs = self._derive_entries(roads)
        if not edge_pairs:
            return 0.0, 0.0
        csr, node_index = _build_csr(edge_pairs, splits_uv)
        return _sampled_efficiency_core(csr, node_index, entry, self.sources,
                                        self.rep_xy, self.src_euclid)

class _StepContext:
    """The greedy arterial's per-commit scoring context: the streets ∪ committed-roads edge set and
    each parcel's nearest-edge entry base over it, frozen ONCE per commit (~max_roads times), so
    `score_candidate(real)` scores a candidate trial road by reprojecting only onto `real`'s few
    edges (+ any committed/street edges `real` splits) rather than re-deriving every parcel against
    every edge. Built via `ctx.step(committed)`.

    PRECONDITION (see `score_candidate`): `score_candidate(real)` is bit-exact to
    `ctx.score(_planarize(committed + [real]))` ONLY for boundary-snapped trials that meet the
    committed/street network at SHARED graph vertices -- the buildable mode, where `_snap` returns
    boundary-graph paths, so a new road joins committed roads at common lattice nodes and never
    splits a committed edge at a floating-point interior point. It is NOT bit-exact for aspirational
    free chords that cross a committed edge at an interior float point: there
    `unary_union([base_merged, real])` does not node identically to the reference
    `unary_union(committed + [real])` (the incremental-planarize noding diverges within `_rnd`
    rounding -- design "Bug 2"), so the greedy routes aspirational candidates through the full
    `ctx.score(_planarize(committed + [real]))` reference path instead (see
    `arterial._greedy_arterials`); `score_candidate` is used only for buildable.

    The step's per-parcel base freezes, for each parcel, ALL streets∪committed edges within `tol`,
    each as `(edge pair, parcel->edge distance, projection-along-edge, entry node)`. The final entry
    is the full `_line_entries` (distance, nx-edge-index) argmin over {frozen step edges} ∪
    {`real`/split delta edges}, resolved against the SAME `networkx.Graph.edges()` order `.score`
    uses (so exact-distance grid ties pick the same node). Freezing ALL near edges -- not only those
    at the minimum distance -- is the "Bug-1" robustness fix: when `real` cleanly SPLITS a parcel's
    nearest step edge, that edge is dropped (its pair is no longer in the noded edge set) and its
    near sub-segment -- whose `_rnd`-rounded crossing distance can rise ABOVE a 2nd-nearest step
    edge -- is recovered from the per-candidate delta; a min-distance-only freeze would have
    discarded that 2nd-nearest edge and picked the wrong entry node."""

    def __init__(self, ctx: _BlockScoringContext, committed: GeoDataFrame | None) -> None:
        self.ctx = ctx
        lines = (list(committed.geometry)
                 if committed is not None and len(committed) else [])
        # The planarized committed union, re-noded against `real` per candidate.
        self.base_merged: BaseGeometry | None = unary_union(lines) if lines else None
        committed_segs = _explode_segments(lines)
        step_pairs: list[_Pair] = [*committed_segs, *ctx.street_segs]
        self.step_set: set[frozenset[_Node]] = {frozenset(p) for p in step_pairs}
        # Per parcel: ALL streets∪committed edges within `tol`, each as
        # (edge pair, parcel->edge distance, projection-along-edge, entry node). Freezing EVERY near
        # edge (not only the minimum-distance ones) is the Bug-1 fix: when `real` splits a parcel's
        # nearest edge, that edge is dropped and its sub-segment's `_rnd`-rounded distance can
        # exceed a 2nd-nearest edge, which must therefore stay a live candidate.
        self.step_cands: list[list[tuple[_Pair, float, float, _Node]]] = [[] for _ in range(ctx.n)]
        if step_pairs and ctx.n:
            self._freeze_base(step_pairs)

    def _freeze_base(self, step_pairs: list[_Pair]) -> None:
        ctx = self.ctx
        lines = np.array([LineString([a, b]) for a, b in step_pairs], dtype=object)
        hits = STRtree(lines).query(ctx.geoms_arr, predicate="dwithin", distance=ctx.tol)
        if hits.shape[1] == 0:
            return
        pidx, eidx, dist, proj, foot_xy = _reproject_hits(ctx.geoms_arr, ctx.reps_arr, lines, hits)
        for m in range(pidx.shape[0]):
            node = (round(float(foot_xy[m, 0]), 2), round(float(foot_xy[m, 1]), 2))
            self.step_cands[int(pidx[m])].append(
                (step_pairs[int(eidx[m])], float(dist[m]), float(proj[m]), node))

    def score_candidate(self, real: LineString) -> tuple[float, float]:
        """(E, directness) for streets ∪ committed ∪ `real`, incrementally. Bit-exact to
        `ctx.score(_planarize(committed + [real]))` ONLY when `real` meets the committed/street
        network at shared graph vertices (buildable-snapped trials); NOT bit-exact for aspirational
        free chords crossing a committed edge at a float interior point ("Bug 2" -- see the class
        docstring, which is why the greedy routes aspirational through the full path). Robust to
        `real` cleanly splitting a parcel's nearest step edge ("Bug-1" fix)."""
        ctx = self.ctx
        if ctx.n < 2:
            return 0.0, 0.0
        merged = (unary_union([self.base_merged, real]) if self.base_merged is not None
                  else unary_union([real]))
        road_parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
        road_segs = _explode_segments(road_parts)
        full_pairs = _edges_in_nx_order(road_segs, ctx.street_segs)
        if not full_pairs:
            return 0.0, 0.0
        idx = {frozenset(p): i for i, p in enumerate(full_pairs)}
        # Delta = full edges absent from the frozen step set: `real`'s edges + any committed/street
        # edge `real` split at a mid-span crossing (the split sub-segments).
        delta_pairs = [p for p in full_pairs if frozenset(p) not in self.step_set]
        cands: list[list[tuple[float, int, _Pair, float, _Node]]] = [[] for _ in range(ctx.n)]
        if delta_pairs:
            lines = np.array([LineString([a, b]) for a, b in delta_pairs], dtype=object)
            hits = STRtree(lines).query(ctx.geoms_arr, predicate="dwithin", distance=ctx.tol)
            if hits.shape[1]:
                pidx, eidx, dist, proj, foot_xy = _reproject_hits(
                    ctx.geoms_arr, ctx.reps_arr, lines, hits)
                for m in range(pidx.shape[0]):
                    pair = delta_pairs[int(eidx[m])]
                    node = (round(float(foot_xy[m, 0]), 2), round(float(foot_xy[m, 1]), 2))
                    cands[int(pidx[m])].append((float(dist[m]), idx[frozenset(pair)], pair,
                                                float(proj[m]), node))
        entry: list[_Node | None] = [None] * ctx.n
        splits: dict[_Pair, list[tuple[float, _Node]]] = defaultdict(list)
        for p in range(ctx.n):
            per = cands[p]
            for pair, sdist, pr, node in self.step_cands[p]:
                fs = frozenset(pair)
                if fs in idx:                          # a split-away step edge is dropped here...
                    per.append((sdist, idx[fs], pair, pr, node))
            if not per:                                # ...and recovered from `delta_pairs` above
                continue
            _d, _i, pair, pr, node = min(per, key=lambda t: (t[0], t[1]))
            entry[p] = node
            splits[pair].append((pr, node))
        csr, node_index = _build_csr(full_pairs, splits)
        return _sampled_efficiency_core(csr, node_index, entry, ctx.sources, ctx.rep_xy,
                                        ctx.src_euclid)


def network_efficiency(block: Block, roads: GeoDataFrame | None, *, k: int = 40,
                       tol: float = STREET_TOL) -> tuple[float, float]:
    """Sampled (E, directness) of the network `roads` + block.streets. A parcel maps to the graph
    by line-proximity -- the nearest POINT on a road/street edge within `tol`, injected as a
    colinear split node (`_line_entries` + `_build_csr`), which counts sparse straight chords the
    old nearest-vertex rule undercounted. From K seeded source parcels to ALL parcels; (0, 0) if the
    graph is empty. Deterministic: sources are evenly spaced over the parcel row order.

    Delegates to `_BlockScoringContext(block).score(roads)` -- the context freezes the per-block
    constants once, though a bare `network_efficiency` call (as here) still builds one context per
    call. The arterial greedy loop instead builds the context ONCE per block and calls `.score`
    per candidate.

    Note this function alone is NOT guaranteed monotone as `roads` grows across separate calls,
    because each call re-derives entries against its own `roads` and entries can churn."""
    return _BlockScoringContext(block, k=k, tol=tol).score(roads)


@dataclass(frozen=True)
class Curve:
    cost: list[float]     # cumulative added road length (m)
    benefit: list[float]  # value per budget (permeability or displacement fraction), in [0, 1)


V = TypeVar("V")


def street_first_ordered(block: Block, roads: GeoDataFrame, tol: float) -> GeoDataFrame:
    """`roads` reindexed so EVERY PREFIX IS A CONNECTED NETWORK REACHING THE STREET, reset to a
    fresh RangeIndex.

    THE canonical prefix order: `_sweep`, every `prefix_to_*` walk and `animate` all grow the road
    set in this one sequence, so a method's curve, its lens truncations and its GIF agree by
    construction. Public (not `_`-prefixed) because `animate` imports it across a module
    boundary -- it is a real seam, and naming it private only obscured that. It is deliberately NOT
    a Hydra Strategy: the two alternatives tried are measurably WRONG rather than different (see
    below), so a plug-in point would ship one implementation and a menu nobody selects. See
    docs/superpowers/backlog.md, "Lens prefix selection", for the optimization problem that would
    earn one.

    Callers guard `len(roads) == 0` before calling.

    Drainage-descending order, with each road's CONNECTORS BOUGHT ON DEMAND: walk roads by drainage
    descending, and before emitting one, emit any not-yet-emitted roads along its own shortest path
    back to the street (street end first). Roads in no street-connected component have an empty
    chain and simply take their drainage position; they are unbuildable either way.

    Two properties make this the right order, and both matter:

    1. **Every prefix is connected**, because a road is only ever chosen when it already touches the
       built network. A pure drainage-descending order does NOT guarantee this. It is a valid
       topological order only for a drainage TREE, where a road nearer the street carries at least
       the traffic of everything beyond it; every method used to be such a tree, so it held and the
       gap went unnoticed. It fails once a network has loops (drainage stops being monotone along a
       path) or a later path branches off the MIDDLE of an earlier one, which even clearance does.
       Measured fraction of scored prefix length reaching the street under the unconstrained order:
       greedy_arterial_repulsion 0.782, resistance_greedy 0.900, clearance_looped 0.831 at region
       scale. The lens was scoring road sets that could not be built.
    2. **It preserves drainage's trunk-first preference**, and reduces EXACTLY to the old order
       whenever that order was already buildable -- so a drainage tree's numbers are unchanged, and
       the correction only touches prefixes that were actually broken.

    Two alternatives were built and measured, and both distort in the same direction -- they make a
    method look more expensive than a road set it demonstrably achieves:

    - Sorting by network DISTANCE to the street guarantees connectivity but is breadth-first: it
      completes a whole ring before reaching any deeper, penalizing fine-grained networks for their
      granularity rather than their geometry. It moved `resistance_lp` on one region from 2,236 m at
      0.0403 displacement to 5,104 m at 0.0630.
    - Taking the highest-drainage road that merely TOUCHES the built network is nearly free at
      block scale but loose at region scale: on the depth region it needed 23,490 m to reach
      P* = 0.60 where a fully connected 1,951 m prefix was measured to exist -- 12x too much.

    Buying the chain on demand avoids both: a deep high-drainage road is reachable as soon as its
    own connectors are paid for, so the walk goes deep immediately instead of sweeping outward.

    The lens's real question -- the CHEAPEST connected subnetwork reaching a target -- is an
    optimization problem that no fixed order answers exactly. This is a heuristic for it, chosen to
    be conservative in the one direction that matters (never scoring a set nobody could build)
    without inflating cost.
    """
    net = _road_net(block, roads, tol)
    drain = road_drainage(block, roads, tol=tol)
    dist, paths = net.street_distance()

    def connector_chain(i: int) -> list[int]:
        """The roads on road `i`'s own shortest path back to the street, street end first."""
        ns = [n for n in net.road_nodes[i] if n in dist]
        if not ns:
            return []
        path = paths[min(ns, key=lambda n: (dist[n], n))]
        chain: list[int] = []
        for a, b in zip(path, path[1:], strict=False):
            r = net.edge_row.get(frozenset((a, b)))
            if r is not None and r != i and r not in chain:
                chain.append(r)
        return chain

    order: list[int] = []
    taken = [False] * len(roads)
    for i in sorted(range(len(roads)), key=lambda k: (-drain[k], k)):
        if taken[i]:
            continue
        for r in connector_chain(i):        # street end first, so the prefix stays connected
            if not taken[r]:
                taken[r] = True
                order.append(r)
        taken[i] = True
        order.append(i)
    return cast(GeoDataFrame, roads.iloc[order].reset_index(drop=True))


def _sweep(block: Block, roads: GeoDataFrame, value: Callable[[GeoDataFrame | None], V],
           n_points: int, tol: float) -> tuple[list[float], list[V]]:
    """Drainage-ordered cumulative-budget sweep: returns ([road_length_m(prefix)], [value(prefix)]).
    Order roads by drainage descending, then at n_points cumulative-length budgets evaluate `value`
    on the empty-prefix baseline and each growing prefix (skipping budgets that add no new road).
    The reported x-axis is cumulative added road length in metres."""
    def _length(prefix: GeoDataFrame) -> float:
        return float(prefix.geometry.length.sum())

    costs: list[float] = [_length(cast(GeoDataFrame, roads.iloc[:0]))]
    vals: list[V] = [value(cast(GeoDataFrame, roads.iloc[:0]))]
    if len(roads) == 0 or block.boundary.area == 0.0:
        return costs, vals
    ordered = street_first_ordered(block, roads, tol)
    cum = ordered.geometry.length.to_numpy().cumsum()
    total = float(cum[-1])
    seen = 0
    for kk in range(1, n_points + 1):
        m = int((cum <= (kk / n_points) * total + 1e-9).sum())
        if m <= seen:
            continue
        seen = m
        costs.append(_length(cast(GeoDataFrame, ordered.iloc[:m])))
        vals.append(value(cast(GeoDataFrame, ordered.iloc[:m])))
    return costs, vals


def max_access_depth(block: Block, roads: GeoDataFrame | None, *, tol: float = STREET_TOL,
                     adj: list[set[int]] | None = None) -> int:
    """The block's deepest BFS access-depth (`parcel_access_layers`) given `roads` -- 1 = every
    parcel fronts a street, higher = buried. `adj` (parcel adjacency) may be passed to avoid
    rebuilding it across repeated calls on the same block."""
    return int(parcel_access_layers(block, roads, tol=tol, adj=adj).max())


def prefix_to_depth(block: Block, roads: GeoDataFrame, target_depth: int, *,
                    tol: float = STREET_TOL) -> tuple[GeoDataFrame, int]:
    """The minimal drainage-ordered prefix of `roads` whose max BFS access-depth is
    <= `target_depth`, paired with that prefix's actual max depth. Access-depth is monotone
    non-increasing as drainage-ordered roads are added (a larger street seed only shrinks depths),
    so a binary search over the prefix length finds the smallest sufficient prefix in O(log R)
    peels. If even all `roads` cannot reach `target_depth`, returns (all roads in canonical order,
    floor depth) with floor depth > `target_depth` -- the caller reports that as unreached (an
    `osm_footpaths`-style fixed input that never reaches the deep interior). Empty `roads` returns
    (empty, the no-road peel's max depth)."""
    adj = parcel_adjacency(list(block.parcels.geometry), tol)
    if len(roads) == 0:
        empty = cast(GeoDataFrame, roads.iloc[:0])
        return empty, max_access_depth(block, empty, tol=tol, adj=adj)
    ordered = street_first_ordered(block, roads, tol)

    def depth_at(m: int) -> int:
        return max_access_depth(block, cast(GeoDataFrame, ordered.iloc[:m]), tol=tol, adj=adj)

    n = len(ordered)
    full_depth = depth_at(n)
    if full_depth > target_depth:                 # unreachable: best effort is all roads
        return ordered, full_depth
    lo, hi = 0, n                                 # smallest m with depth_at(m) <= target_depth
    while lo < hi:
        mid = (lo + hi) // 2
        if depth_at(mid) <= target_depth:
            hi = mid
        else:
            lo = mid + 1
    return cast(GeoDataFrame, ordered.iloc[:lo].reset_index(drop=True)), depth_at(lo)


def prefix_to_displacement(block: Block, roads: GeoDataFrame, radii: NDArray[np.float64],
                           d_frac: float, *, corridor_m: float = 3.0,
                           tol: float = STREET_TOL) -> GeoDataFrame:
    """The minimal drainage-ordered prefix of `roads` whose displacement FRACTION
    (`displacement(block.building_points, radii, prefix, corridor_m) / n_buildings`) is
    >= `d_frac`. Displacement is monotone non-decreasing as drainage-ordered roads are added (a
    growing prefix's buffered corridor union only grows, so every building's distance to it is
    non-increasing, hence each cᵢ is non-decreasing), so a binary search over the prefix length
    finds the smallest sufficient prefix in O(log R) peels -- mirroring `prefix_to_permeability`'s
    binary search. n_buildings == 0 makes the fraction always 0.0 (matching
    `displacement_curve`'s `_disp`), so a positive `d_frac` is then unreachable. If even all
    `roads` cannot reach `d_frac`, returns all roads in canonical order. Empty `roads` returns
    empty."""
    n = len(block.building_points)
    if len(roads) == 0:
        return cast(GeoDataFrame, roads.iloc[:0])
    ordered = street_first_ordered(block, roads, tol)

    def frac_at(m: int) -> float:
        if n == 0:
            return 0.0
        return displacement(block.building_points, radii,
                            cast(GeoDataFrame, ordered.iloc[:m]), corridor_m) / n

    total = len(ordered)
    if frac_at(total) < d_frac:                   # unreachable: best effort is all roads
        return ordered
    lo, hi = 0, total                              # smallest m with frac_at(m) >= d_frac
    while lo < hi:
        mid = (lo + hi) // 2
        if frac_at(mid) >= d_frac:
            hi = mid
        else:
            lo = mid + 1
    return cast(GeoDataFrame, ordered.iloc[:lo].reset_index(drop=True))


def prefix_to_permeability(
    block: Block,
    roads: GeoDataFrame,
    p_star: float,
    params: PermeabilityParams = PermeabilityParams(),  # noqa: B008 (frozen, immutable)
    *,
    tol: float = STREET_TOL,
) -> tuple[GeoDataFrame, bool]:
    """The minimal drainage-ordered prefix of `roads` whose `permeability` is >= `p_star`, paired
    with whether that target was reached. Permeability is monotone non-decreasing as
    drainage-ordered roads are added (roads only add conductance -- never remove it -- so the
    dissipated power is monotone non-increasing by Rayleigh's monotonicity theorem; see
    permeability.py's module docstring), so a binary search over the prefix length finds the
    smallest sufficient prefix in O(log R) peels, mirroring `prefix_to_displacement`'s binary
    search. The no-roads baseline p0 is frozen ONCE via `egress_power` rather than recomputed per
    probed prefix. `adj` (parcel_adjacency, an STRtree spatial join -- costly at region scale) is
    likewise built ONCE and threaded through every `egress_power`/`permeability` call: adjacency
    is a function of `block.parcels` geometry alone, invariant across road prefixes, exactly the
    precomputed-adj pattern `prefix_to_depth` already uses. If even all `roads`
    cannot reach `p_star` (including an ungrounded block, where permeability is nan and every
    comparison is False), returns (all roads in canonical order, False). Empty `roads` returns
    (empty, False)."""
    if len(roads) == 0:
        return cast(GeoDataFrame, roads.iloc[:0]), False
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    p0, _ = egress_power(block, None, params, adj=adj)
    ordered = street_first_ordered(block, roads, tol)

    def perm_at(m: int) -> float:
        return permeability(block, cast(GeoDataFrame, ordered.iloc[:m]), params, p0=p0, adj=adj)

    n = len(ordered)
    if not (perm_at(n) >= p_star):                 # unreachable (incl. nan): best effort all roads
        return ordered, False
    lo, hi = 0, n                                  # smallest m with perm_at(m) >= p_star
    while lo < hi:
        mid = (lo + hi) // 2
        if perm_at(mid) >= p_star:
            hi = mid
        else:
            lo = mid + 1
    return cast(GeoDataFrame, ordered.iloc[:lo].reset_index(drop=True)), True


def displacement_curve(block: Block, roads: GeoDataFrame, radii: NDArray[np.float64], *,
                       corridor_m: float = 3.0, n_points: int = 20,
                       tol: float = STREET_TOL) -> Curve:
    """A Curve whose x is cumulative added road length (m) and whose y is the FRACTION of homes
    displaced, Σcᵢ / n_buildings (a rising COST in [0, 1]). Reuses the drainage-ordered _sweep.
    n_buildings = len(block.building_points) (buildings, not parcels)."""
    n = len(block.building_points)

    def _disp(prefix: GeoDataFrame | None) -> float:
        if prefix is None or len(prefix) == 0 or n == 0:
            return 0.0
        return displacement(block.building_points, radii, prefix, corridor_m) / n

    costs, vals = _sweep(block, roads, _disp, n_points, tol)
    return Curve(costs, vals)


def _noded_graph(roads: GeoDataFrame, streets: GeoDataFrame) -> nx.Graph:
    """The PLANARIZED road∪street graph: `unary_union` nodes every crossing/touch into shared
    vertices, then `_explode_segments` turns the merged geometry into `_rnd`-snapped (2-dp),
    deduped, non-degenerate undirected edges (a stray point yields no segment, so it drops out).
    Empty input -> empty graph."""
    g: nx.Graph = nx.Graph()
    geoms = list(roads.geometry) + list(streets.geometry)
    if geoms:
        g.add_edges_from(_explode_segments([unary_union(geoms)]))
    return g


