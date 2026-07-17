"""Cost-benefit curves for reblocking methods: add a method's roads incrementally in
drainage order, score access at each budget, trace benefit (fraction of Sigma depth^2
removed) vs cost (cumulative added road length, m). AUC = a 0-1 efficiency score. See the
design spec.
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


def access_burden(depths: pd.Series) -> float:
    """Sigma depth^2 -- q=2 severity-weighted access burden (kblock parcels = one building each)."""
    return float((depths.astype("float64") ** 2).sum())


def road_drainage(block: Block, roads: GeoDataFrame, *, tol: float = STREET_TOL) -> list[int]:
    """Per-road parcel count: build a graph from the road segments, route each parcel to the
    street through it, and count how many parcels use each segment. Uniform across methods."""
    n = len(roads)
    if n == 0:
        return []
    g: nx.Graph = nx.Graph()
    edge_row: dict[frozenset[tuple[float, float]], int] = {}
    for i, geom in enumerate(roads.geometry):
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]   # explode Multi*
        for part in parts:
            cs = list(part.coords)
            for a, b in zip(cs, cs[1:], strict=False):
                na, nb = _rnd(a), _rnd(b)
                if na != nb:
                    g.add_edge(na, nb, weight=Point(na).distance(Point(nb)))
                    edge_row[frozenset((na, nb))] = i
    street = unary_union(list(block.streets.geometry))
    snodes = {node for node in g.nodes if Point(node).distance(street) <= tol}
    if not snodes:
        return [0] * n
    dist, paths = nx.multi_source_dijkstra(g, sorted(snodes))
    nodes = list(g.nodes)
    tree = STRtree([Point(node) for node in nodes])
    counts: dict[int, int] = defaultdict(int)
    for geom in block.parcels.geometry:
        reach = [nodes[j] for j in tree.query(geom, predicate="dwithin", distance=tol)
                 if nodes[j] in dist]
        if not reach:
            continue
        entry = min(reach, key=lambda node: (dist[node], node))
        for a, b in zip(paths[entry], paths[entry][1:], strict=False):
            row = edge_row.get(frozenset((a, b)))
            if row is not None:
                counts[row] += 1
    return [counts.get(i, 0) for i in range(n)]


BenefitFactory = Callable[..., Callable[["GeoDataFrame | None"], float]]


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
    """Per-block scoring constants frozen ONCE, shared by `network_efficiency`, the compare's
    efficiency/directness curves (`_efficiency_factory`), and the greedy arterial loop. Freezing
    lifts the representative points, sampled sources, the K*N euclidean matrix and the street edge
    geometry out of the per-candidate hot path (they were previously rebuilt for all ~7k
    candidates). `.score(roads)` re-derives entries against streets + `roads` and matches
    `network_efficiency` exactly; `.score_frozen(prefix, ...)` scores a prefix against FROZEN
    entries/splits and matches `_efficiency_factory` (monotone across prefixes)."""

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

    def score_frozen(self, roads_prefix: GeoDataFrame | None, *,
                     entry: list[_Node | None],
                     splits: dict[_Pair, list[tuple[float, _Node]]]) -> tuple[float, float]:
        """(E, directness) over streets + `roads_prefix` using the FROZEN `entry`/`splits` (keyed
        by parent edge pair, derived once against the full road set) -- builds its own per-prefix
        CSR (streets + prefix roads, split at the frozen parents that are present) and matches
        `_efficiency_factory`. A frozen source whose entry's parent edge is absent from the prefix
        contributes 0, so distances from fixed entries are non-increasing and the sweep is
        monotone across prefixes."""
        if self.n < 2:
            return 0.0, 0.0
        prefix_segs = (_explode_segments(roads_prefix.geometry)
                       if roads_prefix is not None and len(roads_prefix) else [])
        base_pairs = [*prefix_segs, *self.street_segs]
        if not base_pairs:
            return 0.0, 0.0
        csr, node_index = _build_csr(base_pairs, splits)
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
    because each call re-derives entries against its own `roads` and entries can churn.
    `cost_benefit_curve`'s `efficiency_benefit`/`directness_benefit` factories get monotonicity
    instead by freezing entries against the FULL road set once, then only growing the edge set
    -- see `_efficiency_factory`."""
    return _BlockScoringContext(block, k=k, tol=tol).score(roads)


def _efficiency_factory(block: Block, roads_full: GeoDataFrame | None, tol: float,
                        k: int = 40) -> Callable[[GeoDataFrame | None], tuple[float, float]]:
    """Shared machinery behind `efficiency_directness_curves` (the frontier-sweep measurement of
    arterial's kept `objective=directness|efficiency`, backing the arterial-vs-dijkstra
    directness tests and the 1e-9 scoring-equivalence net) and `efficiency_benefit`/
    `directness_benefit` (fed to `cost_benefit_curve` by budget.py's own monotonicity tests, not
    the deleted cost-benefit reporting). Freezes the parcel->entry-node mapping and the K sampled
    sources against the FULL graph (`roads_full` + block.streets), built ONCE, in a
    `_BlockScoringContext`. The returned f(roads_prefix) computes (E, directness) from those
    FIXED entries via `ctx.score_frozen`, over a graph containing only `roads_prefix` +
    block.streets edges (rounded coordinates keep node identity stable across subsets). A
    source/dest whose fixed entry node's parent edge is absent from that prefix contributes 0.

    Since the entry mapping, sources, and the all-parcel pair set never change while the
    edge set only grows as `roads_prefix` grows, shortest-path distances from fixed entries
    are non-increasing -- so E and directness are non-decreasing across the frontier sweep's
    prefixes, unlike calling `network_efficiency(block, roads_prefix)` per prefix (which
    re-derives entries against each prefix and can regress, see budget.py module docstring
    history / the review this fixes)."""
    ctx = _BlockScoringContext(block, k=k, tol=tol)
    entry, splits, edge_pairs = ctx._derive_entries(roads_full)
    if ctx.n < 2 or not edge_pairs:
        return lambda _roads: (0.0, 0.0)

    def f(roads_prefix: GeoDataFrame | None) -> tuple[float, float]:
        return ctx.score_frozen(roads_prefix, entry=entry, splits=splits)
    return f


def access_benefit(block: Block, roads_full: GeoDataFrame | None, *,
                   tol: float = STREET_TOL) -> Callable[[GeoDataFrame | None], float]:
    """`roads_full` is unused here: access_burden's per-block unreached_depth cap (N+1)
    already makes this benefit monotone as roads are added. The param exists so this matches
    the shared benefit-factory signature (see efficiency_benefit/directness_benefit, which do
    need the full road set to freeze entries)."""
    adj = parcel_adjacency(list(block.parcels.geometry), tol)
    cap = len(block.parcels) + 1
    base = access_burden(parcel_access_layers(block, None, tol=tol, adj=adj, unreached_depth=cap))

    def f(roads: GeoDataFrame | None) -> float:
        if base == 0.0:
            return 0.0
        return 1.0 - access_burden(
            parcel_access_layers(block, roads, tol=tol, adj=adj, unreached_depth=cap)) / base
    return f


def efficiency_benefit(block: Block, roads_full: GeoDataFrame | None, *,
                       tol: float = STREET_TOL) -> Callable[[GeoDataFrame | None], float]:
    """The E (global efficiency) half of `_efficiency_factory`, wrapped as a `cost_benefit_curve`
    `benefit_fn`. Not part of the deleted cost-benefit reporting -- exercised directly by
    budget.py's own monotonicity tests (`tests/test_budget.py`)."""
    f = _efficiency_factory(block, roads_full, tol)
    return lambda roads: f(roads)[0]


def directness_benefit(block: Block, roads_full: GeoDataFrame | None, *,
                       tol: float = STREET_TOL) -> Callable[[GeoDataFrame | None], float]:
    """The directness half of `_efficiency_factory`, wrapped as a `cost_benefit_curve`
    `benefit_fn`. Not part of the deleted cost-benefit reporting -- exercised directly by
    budget.py's own monotonicity tests (`tests/test_budget.py`)."""
    f = _efficiency_factory(block, roads_full, tol)
    return lambda roads: f(roads)[1]


@dataclass(frozen=True)
class Curve:
    cost: list[float]     # cumulative added road length (m)
    benefit: list[float]  # fraction of Sigma depth^2 removed, in [0, 1]


V = TypeVar("V")


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
    drain = road_drainage(block, roads, tol=tol)
    order = sorted(range(len(roads)), key=lambda i: (-drain[i], i))
    ordered = roads.iloc[order].reset_index(drop=True)
    cum = ordered.geometry.length.to_numpy().cumsum()
    total = float(cum[-1])
    seen = 0
    for kk in range(1, n_points + 1):
        m = int((cum <= (kk / n_points) * total + 1e-9).sum())
        if m <= seen:
            continue
        seen = m
        costs.append(_length(ordered.iloc[:m]))
        vals.append(value(ordered.iloc[:m]))
    return costs, vals


def truncate_to_length(block: Block, roads: GeoDataFrame, budget_m: float,
                       tol: float = STREET_TOL) -> GeoDataFrame:
    """The drainage-ordered prefix of `roads` whose cumulative length <= `budget_m` (the same order
    _sweep uses). Empty for budget_m <= 0; all roads for budget_m >= total."""
    if len(roads) == 0 or budget_m <= 0.0:
        return cast(GeoDataFrame, roads.iloc[:0])
    drain = road_drainage(block, roads, tol=tol)
    order = sorted(range(len(roads)), key=lambda i: (-drain[i], i))
    ordered = roads.iloc[order].reset_index(drop=True)
    cum = ordered.geometry.length.to_numpy().cumsum()
    m = int((cum <= budget_m + 1e-9).sum())
    return cast(GeoDataFrame, ordered.iloc[:m])


def matched_budget(total_length_by_method: dict[str, float]) -> float:
    """The common render budget: the smallest method's total road length (every method can reach
    it). 0.0 if empty."""
    return min(total_length_by_method.values()) if total_length_by_method else 0.0


def displacement_curve(block: Block, roads: GeoDataFrame, radii: NDArray[np.float64], *,
                       corridor_m: float = 3.0, n_points: int = 20,
                       tol: float = STREET_TOL) -> Curve:
    """A Curve whose x is cumulative added road length (m) and whose y is Sum c_i displacement (a
    rising COST). Reuses the drainage-ordered _sweep with displacement as the value."""
    def _disp(prefix: GeoDataFrame | None) -> float:
        if prefix is None or len(prefix) == 0:
            return 0.0
        return displacement(block.building_points, radii, prefix, corridor_m)

    costs, vals = _sweep(block, roads, _disp, n_points, tol)
    return Curve(costs, vals)


def cost_benefit_curve(block: Block, roads: GeoDataFrame, *,
                       benefit_fn: BenefitFactory = access_benefit,
                       n_points: int = 20, tol: float = STREET_TOL) -> Curve:
    """Order roads by drainage descending, then at n_points cumulative-length budgets score
    benefit_fn's benefit vs the no-roads baseline. The x-axis is cumulative added road length
    (m). `corridor_m` is gone: it only fed the deleted displacement cost-mode; benefit is
    corridor-free."""
    costs, benefit = _sweep(block, roads, benefit_fn(block, roads, tol=tol), n_points, tol)
    return Curve(costs, benefit)


def _noded_graph(roads: GeoDataFrame, streets: GeoDataFrame) -> nx.Graph:
    """The PLANARIZED road∪street graph: unary_union nodes every crossing/touch into shared
    vertices, then each _rnd-snapped (2-dp) segment becomes one undirected edge (deduped). Non-
    LineString union fragments (stray points) are skipped. Empty input -> empty graph."""
    geoms = list(roads.geometry) + list(streets.geometry)
    g: nx.Graph = nx.Graph()
    if not geoms:
        return g
    merged = unary_union(geoms)
    parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    for part in parts:
        if not hasattr(part, "coords"):
            continue
        cs = [_rnd(c) for c in part.coords]
        for a, b in zip(cs, cs[1:], strict=False):
            if a != b:
                g.add_edge(a, b)
    return g


def cycle_density(block: Block, roads: GeoDataFrame | None) -> float:
    """Internal connectivity: circuit rank per parcel, (E - N + C) / P, over the noded road∪street
    graph (E/N/C = edge/node/component counts, P = parcel count). The number of independent cycles
    (redundant internal routes) per dwelling; a tree -> 0. Circuit rank is a topological invariant
    (subdivision-insensitive), so /P (fixed, exogenous) keeps the whole metric discretization-
    invariant. 0.0 with no roads / no parcels / an empty graph."""
    p = len(block.parcels)
    if roads is None or len(roads) == 0 or p < 1:
        return 0.0
    g = _noded_graph(roads, block.streets)
    n = g.number_of_nodes()
    if n == 0:
        return 0.0
    circuit_rank = int(g.number_of_edges() - n + nx.number_connected_components(g))
    return circuit_rank / p


def cycle_benefit(block: Block, roads_full: GeoDataFrame | None, *,
                  tol: float = STREET_TOL) -> Callable[[GeoDataFrame | None], float]:
    """Internal-connectivity benefit factory (shares the `access_benefit` signature so it plugs into
    `cost_benefit_curve(..., benefit_fn=cycle_benefit)` and the `_sweep` frontier).
    `roads_full`/`tol` are unused -- cycle_density is self-contained and needs no frozen entries --
    but kept for the shared BenefitFactory signature."""
    del roads_full, tol

    def f(roads: GeoDataFrame | None) -> float:
        return cycle_density(block, roads)
    return f


def efficiency_directness_curves(block: Block, roads: GeoDataFrame, *, n_points: int = 20,
                                 tol: float = STREET_TOL) -> tuple[Curve, Curve]:
    """ONE sampled shortest-path sweep yielding both E and directness curves (x = road length,
    m) -- the frontier-sweep measurement of arterial's kept `objective=directness|efficiency`,
    backing the arterial-vs-dijkstra directness margin tests and the 1e-9 scoring-equivalence
    net. Not part of the deleted cost-benefit reporting."""
    f = _efficiency_factory(block, roads, tol)
    costs, pairs = _sweep(block, roads, f, n_points, tol)
    return Curve(costs, [p[0] for p in pairs]), Curve(costs, [p[1] for p in pairs])


def auc(curve: Curve, cost_cap: float) -> float:
    """Normalized area under benefit-vs-cost over [0, cost_cap] (curve held at its terminal
    benefit beyond its own max cost) -> 0-1 efficiency; higher = more access per meter."""
    if cost_cap <= 0.0 or len(curve.cost) < 2:
        return 0.0
    cs, bs = list(curve.cost), list(curve.benefit)
    if cs[-1] < cost_cap:                       # extend the plateau to the common cap
        cs, bs = cs + [cost_cap], bs + [bs[-1]]
    area = 0.0
    pairs = zip(zip(cs, bs, strict=False), zip(cs[1:], bs[1:], strict=False), strict=False)
    for (c0, b0), (c1, b1) in pairs:
        if c0 >= cost_cap:
            break
        if c1 > cost_cap:                       # segment straddles the cap: interpolate to it
            b_cap = b0 + (b1 - b0) * (cost_cap - c0) / (c1 - c0) if c1 > c0 else b0
            area += 0.5 * (b0 + b_cap) * (cost_cap - c0)
            break
        area += 0.5 * (b0 + b1) * (c1 - c0)
    return area / cost_cap
