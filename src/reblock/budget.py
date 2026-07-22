"""Cost-benefit curves for reblocking methods: add a method's roads incrementally in
drainage order, score access at each budget, trace benefit (fraction of Sigma depth^2
removed) vs cost (cumulative added road length, m). AUC = a 0-1 efficiency score. See the
design spec.
"""
from __future__ import annotations

import math
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


def repulsion(building_points: GeoDataFrame, radii: NDArray[np.float64],
              geom: BaseGeometry) -> float:
    """Soft per-road intrusion cost: a road's OWN proximity to the building field, summed over
    building points as the quadratic-tail kernel r^2/(r^2 + d^2) (d = point-to-road distance,
    r = the building's disk radius from `building_radii`). > 0 whenever any building has r>0 (the tail never reaches
    zero) so no road is 'free' -- unlike `displacement`, whose hard 0-beyond-r cutoff makes
    gap-hugging roads free and degenerate. Depends ONLY on `geom` and the fixed building field
    (not on other committed roads), so it is CONSTANT per candidate -> a well-behaved,
    CELF-safe greedy cost denominator (marginal displacement is not). r==0 (coincident points):
    1.0 if the road touches the point (d<=0) else 0.0, matching displacement's r==0 handling.
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
    """Per-block scoring constants frozen ONCE, shared by `network_efficiency`, the arterial
    directness/efficiency measurement curves (`_efficiency_factory`), and the greedy arterial
    loop. Freezing
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


def _drainage_ordered(block: Block, roads: GeoDataFrame, tol: float) -> GeoDataFrame:
    """`roads` reindexed in drainage-descending order (ties by original index), reset to a fresh
    RangeIndex -- the single canonical prefix order shared by `_sweep`, `truncate_to_length`, and
    `prefix_to_depth`, so every budget/prefix walk grows the road set in the same sequence. Callers
    guard `len(roads) == 0` before calling (an empty road set has no drainage to order)."""
    drain = road_drainage(block, roads, tol=tol)
    order = sorted(range(len(roads)), key=lambda i: (-drain[i], i))
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
    ordered = _drainage_ordered(block, roads, tol)
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


def truncate_to_length(block: Block, roads: GeoDataFrame, budget_m: float,
                       tol: float = STREET_TOL) -> GeoDataFrame:
    """The drainage-ordered prefix of `roads` whose cumulative length <= `budget_m` (the same order
    _sweep uses). Empty for budget_m <= 0; all roads for budget_m >= total."""
    if len(roads) == 0 or budget_m <= 0.0:
        return cast(GeoDataFrame, roads.iloc[:0])
    ordered = _drainage_ordered(block, roads, tol)
    cum = ordered.geometry.length.to_numpy().cumsum()
    m = int((cum <= budget_m + 1e-9).sum())
    return cast(GeoDataFrame, ordered.iloc[:m])


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
    peels. If even all `roads` cannot reach `target_depth`, returns (all roads in drainage order,
    floor depth) with floor depth > `target_depth` -- the caller reports that as unreached (an
    `osm_footpaths`-style fixed input that never reaches the deep interior). Empty `roads` returns
    (empty, the no-road peel's max depth)."""
    adj = parcel_adjacency(list(block.parcels.geometry), tol)
    if len(roads) == 0:
        empty = cast(GeoDataFrame, roads.iloc[:0])
        return empty, max_access_depth(block, empty, tol=tol, adj=adj)
    ordered = _drainage_ordered(block, roads, tol)

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


def prefix_to_external_connectivity(block: Block, roads: GeoDataFrame, target_ext: float, *,
                                    tol: float = STREET_TOL) -> tuple[GeoDataFrame, float]:
    """The minimal drainage-ordered prefix of `roads` whose external connectivity
    (`access_benefit`, fraction of access-burden Sigma-d^2 removed) is >= `target_ext`, paired with
    that prefix's actual external connectivity. Connectivity is monotone NON-DECREASING as
    drainage-ordered roads are added (access_burden's unreached-depth cap makes access_benefit
    monotone), so a binary search over the prefix length finds the smallest sufficient prefix in
    O(log R) peels. If even all `roads` cannot reach `target_ext`, returns (all roads in drainage
    order, full connectivity) with that value < `target_ext` -- the caller reports unreached (an
    osm_footpaths-style fixed input that never reaches the target). Empty `roads` returns
    (empty, 0.0)."""
    ext = access_benefit(block, None, tol=tol)
    if len(roads) == 0:
        return cast(GeoDataFrame, roads.iloc[:0]), 0.0
    ordered = _drainage_ordered(block, roads, tol)

    def ext_at(m: int) -> float:
        return ext(cast(GeoDataFrame, ordered.iloc[:m]))

    n = len(ordered)
    full_ext = ext_at(n)
    if full_ext < target_ext:                     # unreachable: best effort is all roads
        return ordered, full_ext
    lo, hi = 0, n                                 # smallest m with ext_at(m) >= target_ext
    while lo < hi:
        mid = (lo + hi) // 2
        if ext_at(mid) >= target_ext:
            hi = mid
        else:
            lo = mid + 1
    return cast(GeoDataFrame, ordered.iloc[:lo].reset_index(drop=True)), ext_at(lo)


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
    """The PLANARIZED road∪street graph: `unary_union` nodes every crossing/touch into shared
    vertices, then `_explode_segments` turns the merged geometry into `_rnd`-snapped (2-dp),
    deduped, non-degenerate undirected edges (a stray point yields no segment, so it drops out).
    Empty input -> empty graph."""
    g: nx.Graph = nx.Graph()
    geoms = list(roads.geometry) + list(streets.geometry)
    if geoms:
        g.add_edges_from(_explode_segments([unary_union(geoms)]))
    return g


def _entry_resistance(guu: float, gvv: float, guv: float, a: float, b: float, r: float) -> float:
    """Exact grounded resistance R(p) at a point p subdividing an INTERIOR-INTERIOR edge (u, v) of
    resistance r=a+b (a from u, b from v), given only the full grounded Green's function G's
    entries at u and v (`guu`, `gvv`, `guv` = G[u,u], G[v,v], G[u,v]) -- no new matrix row needed,
    so the dense solve never grows past the topology graph's own node count. Derivation: remove
    the direct (u, v) edge via a Sherman-Morrison rank-1 downdate of the grounded Laplacian, then
    solve the two-resistor pendant p attaches via (current conservation at p). Branches to the
    exact series limit -- min(guu + a, gvv + b), matching how R_geo itself picks a route -- when
    (u, v) is a bridge: the downdate is singular exactly when u and v have NO alternate coupling
    (e.g. a plain tree/single-egress edge), where the general formula's individually-diverging
    terms cancel to precisely that limit (verified analytically and, against a slower
    node-injecting reference implementation, numerically on every test scenario below)."""
    c = 1.0 / r
    s = guu - 2.0 * guv + gvv
    denom = 1.0 - c * s
    if denom < 1e-9:                                            # (u, v) is a bridge -> series limit
        return min(guu + a, gvv + b)
    wu, wv = guu - guv, guv - gvv
    k = c / denom
    gpuu = guu + k * wu * wu
    gpvv = gvv + k * wv * wv
    gpuv = guv + k * wu * wv
    big_a = gpuu + a - gpuv
    big_b = gpvv + b - gpuv
    if big_a + big_b <= 0.0:                                    # numerical guard, same limit
        return min(guu + a, gvv + b)
    iu, iv = big_b / (big_a + big_b), big_a / (big_a + big_b)
    return (gpuu + a) * iu + gpuv * iv


def _entry_resistance_ground(gvv: float, a: float, b: float, r: float) -> float:
    """R(p) at a point p subdividing an edge from GROUND (a street node) to one interior node v,
    at distance a from ground and b from v (a+b=r). p sees two parallel routes to ground: the
    direct a-leak, or via v (b + v's own resistance to ground once this edge's leak is removed,
    a diagonal Sherman-Morrison downdate) -- the parallel-resistor formula is stable at the bridge
    limit (v's only ground path is this edge) with no branch needed: it reduces cleanly to `a`."""
    c = 1.0 / r
    denom = 1.0 - c * gvv
    if denom < 1e-9:
        return a
    gpvv = gvv + c * gvv * gvv / denom
    return a * (b + gpvv) / (a + b + gpvv)


_CommuteSetup = tuple[
    # interior node -> (idx map, its component's grounded G)
    dict[_Node, tuple[dict[_Node, int], NDArray[np.float64]]],
    dict[_Node, float],           # R_geo per node (multi-source dijkstra)
    list[tuple[_Node, _Node]],    # edges
    list[LineString],             # edge_lines (index-aligned to edges)
    STRtree,                      # STRtree over edge_lines
]


def _commute_setup(roads: GeoDataFrame | None, streets: GeoDataFrame) -> _CommuteSetup | None:
    """Build the planarized road-union-street graph and, per connected component, its grounded
    Green's function (dense inverse of the interior-node Laplacian) + R_geo (multi-source dijkstra
    from the street nodes), plus an STRtree of edges for line-proximity parcel entry. Returns None
    when there is no usable graph: no/empty roads, no graph nodes, or no street node / no interior
    node. Street nodes are those within STREET_TOL of the street geometry (GEOMETRIC test)."""
    if roads is None or len(roads) == 0:
        return None
    g = _noded_graph(roads, streets)
    if g.number_of_nodes() == 0:
        return None
    street_geom = unary_union(list(streets.geometry))
    snodes = {n for n in g.nodes if Point(n).distance(street_geom) <= STREET_TOL}
    interior = [n for n in g.nodes if n not in snodes]
    if not snodes or not interior:
        return None
    for u, v in g.edges():
        g[u][v]["len"] = max(math.hypot(u[0] - v[0], u[1] - v[1]), 1e-6)
    geo = nx.multi_source_dijkstra_path_length(g, snodes, weight="len")
    green: dict[_Node, tuple[dict[_Node, int], NDArray[np.float64]]] = {}
    for comp in nx.connected_components(g):
        comp_streets = comp & snodes
        comp_int = [n for n in comp if n not in snodes]
        if not comp_streets or not comp_int:                            # stranded -> excluded
            continue
        idx = {n: i for i, n in enumerate(comp_int)}
        m = len(comp_int)
        lg = np.zeros((m, m))
        for u, v in g.subgraph(comp).edges():
            c = 1.0 / g[u][v]["len"]
            ui, vi = idx.get(u), idx.get(v)
            if ui is not None and vi is not None:
                lg[ui, ui] += c
                lg[vi, vi] += c
                lg[ui, vi] -= c
                lg[vi, ui] -= c
            elif ui is not None:
                lg[ui, ui] += c
            elif vi is not None:
                lg[vi, vi] += c
        ginv = np.linalg.inv(lg)                                        # DENSE grounded solve
        for n in comp_int:
            green[n] = (idx, ginv)
    edges = list(g.edges())
    edge_lines = [LineString([u, v]) for u, v in edges]
    return green, geo, edges, edge_lines, STRtree(edge_lines)


def _nearest_edge_ratio(setup: _CommuteSetup, pt: Point) -> tuple[bool, float]:
    """(included, ratio) for parcel point `pt` entering via its single geometrically-nearest
    topology edge. included=False (ratio 0.0) when that edge has NO interior endpoint (the parcel's
    closest frontage is the bare street) or R_geo is non-finite/zero. The caller decides what
    False means: 'skip' (dynamic membership) or 'contribute 0.0' (frozen membership)."""
    green, geo, edges, edge_lines, tree = setup
    j = int(tree.nearest(pt))                                           # line-proximity entry
    ls = edge_lines[j]
    u, v = edges[j]
    proj = ls.project(pt)
    r = max(ls.length, 1e-6)
    a, b = proj, r - proj
    u_int, v_int = u in green, v in green
    if u_int and v_int:
        idx, ginv = green[u]
        guu, gvv, guv = ginv[idx[u], idx[u]], ginv[idx[v], idx[v]], ginv[idx[u], idx[v]]
        r_eff = _entry_resistance(guu, gvv, guv, a, b, r)
    elif u_int:                                                         # v is a street node
        idx, ginv = green[u]                                           # ground dist=b, interior=a
        r_eff = _entry_resistance_ground(ginv[idx[u], idx[u]], b, a, r)
    elif v_int:                                                         # u is a street node
        idx, ginv = green[v]                                           # ground dist=a, interior=b
        r_eff = _entry_resistance_ground(ginv[idx[v], idx[v]], a, b, r)
    else:
        return False, 0.0                                              # both ends on the street
    r_geo = min(geo.get(u, math.inf) + a, geo.get(v, math.inf) + b)
    if not (math.isfinite(r_geo) and r_geo > 1e-9):
        return False, 0.0
    return True, min(max(1.0 - r_eff / r_geo, 0.0), 1.0 - 1e-12)       # clip [0, 1)


def _commute_membership(block: Block, roads: GeoDataFrame | None) -> frozenset[int]:
    """The frozen averaged-parcel set S: indices of parcels with a valid interior entry under
    `roads` (the sweep's terminal/superset network), computed ONCE. A prefix sweep that averages
    over this fixed S has a fixed denominator, which removes the composition churn (parcels
    flickering in/out of the mean) that makes the per-prefix curve non-monotone. Empty if `roads`
    yields no usable graph."""
    setup = _commute_setup(roads, block.streets)
    if setup is None:
        return frozenset()
    return frozenset(i for i, geom in enumerate(block.parcels.geometry)
                     if _nearest_edge_ratio(setup, geom.centroid)[0])


def commute_ratio(block: Block, roads: GeoDataFrame | None, *,
                  membership: frozenset[int] | None = None) -> float:
    """Internal connectivity: mean over parcels of 1 - R(dwelling->street)/R_geodesic on the noded
    road-union-street graph. R = grounded effective resistance to the whole street (a component-wise
    DENSE solve); R_geo = single-best-route (shortest-path) resistance. A single-egress tree route
    -> 0; ->1 as parallel backup routes thicken. Clipped to [0, 1). Rewards added redundancy via
    Rayleigh monotonicity (adding a redundant connector to an existing loop can only help). A small
    tight loop can legitimately outscore a large loose one -- coverage-insensitive by design (see
    access_benefit for coverage). Task-1 corpus gate (2026-07-17): corr(rho, access)=+0.294;
    anti-gaming holds on realistic networks -- loops ADDED to clearance give rho 0.000->TINY
    0.060->BIG 0.278 (BIG >> TINY); a matched-length parallel bundle scores 0.00145/m vs a genuine
    loop's 0.00234/m and costs displacement, so corridor duplication is Pareto-dominated on the
    {external, internal, displacement} suite.

    Each parcel enters by TRUE line-proximity -- the nearest POINT on its nearest topology edge
    (via _entry_resistance/_entry_resistance_ground, computed analytically from the edge's two
    endpoints, so the dense per-component solve never grows with parcel count).

    `membership` selects the averaged set:
    - None (default): DYNAMIC -- exactly the parcels with a valid interior entry under THIS `roads`,
      each contributing its real ratio; a parcel whose nearest edge is a bare street segment, or
      that is unreachable, is skipped. Self-contained per road set. The raw metric is NON-MONOTONE
      across different road sets (a ratio of co-decreasing R/R_geo AND a changing averaged set) --
      standalone reporting ranks by terminal value, never assumes rise.
    - a frozen index set S (from `_commute_membership(block, terminal_roads)`): FROZEN -- every
      parcel in S contributes on EVERY call, its real ratio if it has an interior entry under
      `roads`, else 0.0 ('not yet connected', NOT skipped). Members are averaged in ascending index
      order so a frozen-to-self call (S from the same `roads`) is byte-identical to the dynamic
      default; freezing changes ONLY intermediate prefixes of a sweep, never the terminal value,
      and removes the composition churn so the swept curve reads monotone.

    0.0 with no roads / no parcels / no interior nodes / no usable graph / empty membership."""
    if roads is None or len(roads) == 0 or len(block.parcels) < 1:
        return 0.0
    setup = _commute_setup(roads, block.streets)
    if setup is None:
        return 0.0
    if membership is None:
        ratios = [ratio for geom in block.parcels.geometry
                  for included, ratio in [_nearest_edge_ratio(setup, geom.centroid)] if included]
        return float(np.mean(ratios)) if ratios else 0.0
    if not membership:
        return 0.0
    ratios = [_nearest_edge_ratio(setup, block.parcels.geometry.iloc[i].centroid)[1]
              for i in sorted(membership)]
    return float(np.mean(ratios)) if ratios else 0.0


def commute_ratio_benefit(block: Block, roads_full: GeoDataFrame | None, *,
                          tol: float = STREET_TOL) -> Callable[[GeoDataFrame | None], float]:
    """Internal-connectivity benefit factory (shares the access_benefit signature so it plugs into
    cost_benefit_curve(..., benefit_fn=commute_ratio_benefit) and the _sweep frontier). Freezes the
    averaged parcel set S to `roads_full` -- the terminal network of the sweep -- via
    _commute_membership, so every prefix scores commute_ratio over the SAME denominator. This
    removes the composition churn that made the per-prefix curve non-monotone; the terminal value
    is unchanged (frozen-to-self == the dynamic metric). `tol` is unused, kept for the shared
    BenefitFactory signature."""
    del tol
    membership = _commute_membership(block, roads_full)

    def f(roads: GeoDataFrame | None) -> float:
        return commute_ratio(block, roads, membership=membership)
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
