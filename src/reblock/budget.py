"""Cost-benefit curves for reblocking methods: add a method's roads incrementally in
drainage order, score access at each budget, trace benefit (fraction of Sigma depth^2
removed) vs cost (road density, m/ha). AUC = a 0-1 efficiency score. See the design spec.
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
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import factorized
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


def displacement_count(building_points: GeoDataFrame, roads: GeoDataFrame,
                       corridor_m: float) -> int:
    """Buildings whose site lies in the road corridor (union of `roads.buffer(corridor_m)`).
    0 if there are no points or no roads."""
    if building_points is None or building_points.empty or roads is None or len(roads) == 0:
        return 0
    corridor = roads.geometry.buffer(corridor_m).union_all()
    return int(building_points.geometry.within(corridor).sum())


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


def _resistance_core(csr: csr_matrix, node_index: dict[_Node, int],
                     entry: list[_Node | None], rep_xy: np.ndarray,
                     ground_idx: np.ndarray, cap: float) -> float:
    """Mean per-parcel grounded resistance-to-egress `R_i` (metres; lower = easier egress) over the
    graph `csr` (symmetric CSR, `data` = edge lengths) + `node_index` ((x, y) -> row) -- the
    resistance analogue of `_sampled_efficiency_core`'s door-to-door distance. Edge conductance
    `c_e = 1/length_e`, so a single wire's resistance equals its length, and resistance strictly
    drops below shortest-path length wherever loops give redundant paths (the whole point of the
    metric -- see the design spec's "The metric" section). `ground_idx` are the node rows already
    on the block's street network (potential 0, the egress current's sink); the weighted Laplacian
    `L = diag(deg) - C` is reduced to the FREE (non-ground) nodes reachable from ground
    (`connected_components`), factorized ONCE (`factorized`), then `R_i = (L_G^-1)_{k,k} + leg_i`
    for parcel i's entry node's reduced index k -- solved once per DISTINCT entry node, not once
    per parcel. `leg_i = euclid(rep_xy[i], entry_i)` is the last-mile walk from the parcel to its
    road-entry point (mirrors `_sampled_efficiency_core`'s door-to-door legs). An entry ON a ground
    node -> `R_i = leg_i` (drive term 0, no solve needed); an entry missing / absent from the graph
    / stuck in a component that never reaches ground -> `R_i = cap` (the block bbox diagonal,
    analogous to `access_burden`'s unreached-depth cap).

    Reachability is computed from the POSITIVE-conductance graph (`conductance.eliminate_zeros()`
    before `connected_components`), NOT the raw structural CSR: a non-positive-length edge carries
    zero conductance, so it must not be able to "ground" a component. A component whose only path
    to ground runs through such a zero-length edge is correctly treated as unreachable (degrades
    to `cap`) instead of handing a singular `L_G` to `factorized` (structural reach let this
    through and raised `MatrixRankWarning`/a singular-factorization error).

    Returns the INTENSIVE per-parcel mean `mean_i R_i`, NOT the extensive single-solve aggregate
    `w^T L_G^-1 w` or the raw Kirchhoff index -- those grow with node/parcel count and rank road
    sets WRONGLY (the investigation's caveat: they would call a road set worse purely for adding
    nodes). `mean_i R_i` is invariant to extraneous ungrounded/unreferenced nodes added to `csr`
    (see `test_intensive_mean`), because such nodes never become any parcel's entry and so never
    enter the per-parcel sum.

    Guards: 0 parcels -> 0.0 (nothing to average); 0 graph nodes, or an empty `free` set (every
    node is either grounded or unreachable) -> every parcel falls to `cap` (or `leg_i` if its entry
    happens to sit ON a ground node) and no linear solve runs."""
    n = len(entry)
    if n == 0:
        return 0.0
    num_nodes = csr.shape[0]

    ground_mask = np.zeros(num_nodes, dtype=bool)
    if ground_idx.size and num_nodes:
        ground_mask[ground_idx] = True

    # Conductance built up front (not just inside the `if free.size` solve branch): a zero- or
    # negative-length edge maps to zero conductance, and `eliminate_zeros` drops it from the
    # sparsity structure entirely, so `connected_components` below sees only positive-conductance
    # edges -- see the zero-conductance-reach guard in the docstring above.
    conductance = csr.copy()
    with np.errstate(divide="ignore"):
        conductance.data = np.where(conductance.data > 0, 1.0 / conductance.data, 0.0)
    conductance.eliminate_zeros()

    reach = np.zeros(num_nodes, dtype=bool)
    if num_nodes:
        _n_comp, labels = connected_components(conductance, directed=False)
        grounded_labels = set(labels[ground_idx].tolist()) if ground_idx.size else set()
        if grounded_labels:
            reach = np.isin(labels, list(grounded_labels))

    free = np.flatnonzero(reach & ~ground_mask)
    free_pos = {int(gi): k for k, gi in enumerate(free)}

    solve: Callable[[np.ndarray], np.ndarray] | None = None
    if free.size:
        deg = np.asarray(conductance.sum(axis=1)).ravel()
        laplacian = (diags(deg) - conductance).tocsc()
        lg = laplacian[free][:, free].tocsc()
        solve = factorized(lg)

    entry_xy = np.array([[np.nan, np.nan] if e is None else [e[0], e[1]] for e in entry],
                        dtype=np.float64)
    legs = np.hypot(rep_xy[:, 0] - entry_xy[:, 0], rep_xy[:, 1] - entry_xy[:, 1])
    gi_arr = np.array([node_index.get(e, -1) if e is not None else -1 for e in entry],
                      dtype=np.int64)

    need = sorted({free_pos[int(gi)] for gi in gi_arr if int(gi) in free_pos})
    diag: dict[int, float] = {}
    if solve is not None:
        for k in need:
            unit = np.zeros(free.size, dtype=np.float64)
            unit[k] = 1.0
            diag[k] = float(solve(unit)[k])

    R = np.full(n, cap, dtype=np.float64)
    for i in range(n):
        gi = int(gi_arr[i])
        leg = legs[i]
        if gi < 0 or not np.isfinite(leg):
            continue                                   # no entry / not in graph -> cap
        if ground_mask[gi]:
            R[i] = leg                                 # entry ON the street: drive term 0
        else:
            pos = free_pos.get(gi)
            # invariant: `need` (hence `diag`) is built from every `free_pos` value that any
            # entry's gi maps to, so `pos in diag` always holds when `pos is not None` (given
            # `solve is not None`, guaranteed here since `free_pos` is only ever nonempty when
            # `free.size` -- the same condition that built `solve`).
            if pos is not None:
                R[i] = diag[pos] + leg
            # else: entry not reachable to ground -> stays cap
    return float(np.mean(R))


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

        # For `resistance_frozen`/`_resistance_core`: the block bbox diagonal (the unreached-egress
        # cap, analogous to `access_burden`'s unreached-depth cap) and the streets union (the
        # ground-test geometry for `_ground_indices`), each frozen once per block.
        bounds = np.array(block.boundary.bounds, dtype=np.float64)
        self.cap: float = float(np.hypot(*(bounds[2:] - bounds[:2])))
        self.streets_geom: BaseGeometry = unary_union(list(block.streets.geometry))

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

    def _ground_indices(self, node_index: dict[_Node, int]) -> np.ndarray:
        """Node rows (into a CSR built with `node_index`) whose (x, y) lies within `self.tol` of
        `self.streets_geom` -- the ground set S (potential 0) `resistance_frozen` grounds its
        Laplacian solve on. Vectorized: one batched `shapely.distance` call over every node, not a
        per-node python loop."""
        if not node_index:
            return np.zeros(0, dtype=np.int64)
        nodes = list(node_index.keys())
        pts = shapely.points(np.array(nodes, dtype=np.float64))
        dist = shapely.distance(pts, self.streets_geom)
        rows = np.array([node_index[nd] for nd in nodes], dtype=np.int64)
        return rows[dist <= self.tol]

    def resistance_frozen(self, roads_prefix: GeoDataFrame | None, *,
                          entry: list[_Node | None],
                          splits: dict[_Pair, list[tuple[float, _Node]]]) -> float:
        """Mean per-parcel grounded resistance-to-egress over streets + `roads_prefix`, using the
        FROZEN `entry`/`splits` (derived once against the full road set) -- mirrors
        `score_frozen`'s CSR build exactly (same `base_pairs`/`_build_csr` call), but solves the
        grounded Laplacian (`_resistance_core`) instead of running `_sampled_efficiency_core`'s
        Dijkstra. An empty base edge set (no streets, no prefix roads) -> `self.cap` (every parcel
        unreached), matching `_resistance_core`'s own unreached-cap convention. (No separate
        `self.n < 1` guard: the only caller, `resistance_benefit`, already returns a constant 0.0
        function when `ctx.n < 2`, and `_resistance_core` itself returns 0.0 -- not `self.cap` --
        for 0 parcels, so a guard here would be both dead and wrong.)"""
        prefix_segs = (_explode_segments(roads_prefix.geometry)
                       if roads_prefix is not None and len(roads_prefix) else [])
        base_pairs = [*prefix_segs, *self.street_segs]
        if not base_pairs:
            return self.cap
        csr, node_index = _build_csr(base_pairs, splits)
        return _resistance_core(csr, node_index, entry, self.rep_xy,
                                self._ground_indices(node_index), self.cap)


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
    """Freeze the parcel->entry-node mapping and the K sampled sources against the FULL
    graph (`roads_full` + block.streets), built ONCE, in a `_BlockScoringContext`. The returned
    f(roads_prefix) computes (E, directness) from those FIXED entries via `ctx.score_frozen`, over
    a graph containing only `roads_prefix` + block.streets edges (rounded coordinates keep node
    identity stable across subsets). A source/dest whose fixed entry node's parent edge is absent
    from that prefix contributes 0.

    Since the entry mapping, sources, and the all-parcel pair set never change while the
    edge set only grows as `roads_prefix` grows, shortest-path distances from fixed entries
    are non-increasing -- so E and directness are non-decreasing across cost_benefit_curve's
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
    f = _efficiency_factory(block, roads_full, tol)
    return lambda roads: f(roads)[0]


def directness_benefit(block: Block, roads_full: GeoDataFrame | None, *,
                       tol: float = STREET_TOL) -> Callable[[GeoDataFrame | None], float]:
    f = _efficiency_factory(block, roads_full, tol)
    return lambda roads: f(roads)[1]


def resistance_benefit(block: Block, roads_full: GeoDataFrame | None, *,
                       tol: float = STREET_TOL,
                       k: int = 40) -> Callable[[GeoDataFrame | None], float]:
    """Grounded egress-resistance benefit -- mirrors `_efficiency_factory` exactly (freeze the
    parcel->entry-node mapping against the FULL road set ONCE, in a `_BlockScoringContext`, so
    edges only ever get ADDED across `roads_prefix` growth): by Rayleigh monotonicity, adding
    conductance to a network can only lower every grounded resistance, so `benefit` is
    non-decreasing across `cost_benefit_curve`'s prefixes, same guarantee as
    efficiency_benefit/directness_benefit.

    `R0` is the no-roads (streets-only) score under those frozen entries; `benefit(prefix) =
    (R0 - R(prefix)) / R0`, so `benefit(empty) == 0` and it rises toward 1 as resistance drops
    with added roads. `ctx.n < 2` or no edges at all -> constant 0.0 (degenerate, matches
    `_efficiency_factory`); `R0 <= 0` (every parcel already ON the street) -> constant 0.0 too,
    since there is no resistance left to remove."""
    ctx = _BlockScoringContext(block, k=k, tol=tol)
    entry, splits, edge_pairs = ctx._derive_entries(roads_full)
    if ctx.n < 2 or not edge_pairs:
        return lambda _roads: 0.0
    r0 = ctx.resistance_frozen(None, entry=entry, splits=splits)

    def f(roads_prefix: GeoDataFrame | None) -> float:
        if r0 <= 0:
            return 0.0
        return (r0 - ctx.resistance_frozen(roads_prefix, entry=entry, splits=splits)) / r0
    return f


@dataclass(frozen=True)
class Curve:
    cost: list[float]     # cumulative road density (m/ha) OR cumulative buildings displaced
    benefit: list[float]  # fraction of Sigma depth^2 removed, in [0, 1]


V = TypeVar("V")


def _sweep(block: Block, roads: GeoDataFrame, value: Callable[[GeoDataFrame | None], V],
           n_points: int, tol: float,
           cost_fn: Callable[[GeoDataFrame], float] | None = None) -> tuple[list[float], list[V]]:
    """Drainage-ordered cumulative-budget sweep: returns ([cost_fn(prefix)], [value(prefix)]).
    Order roads by drainage descending, then at n_points cumulative-length budgets evaluate
    `value` on the empty-prefix baseline and each growing prefix (skipping budgets that add no
    new road). `value` maps a road prefix -> any V (a float for a single metric, a tuple for
    several -- run the whole sweep ONCE for a batched multi-metric value). `cost_fn` maps a
    road prefix -> the reported x-axis cost; default is road density (m/ha). The road ORDER and
    length-budget SAMPLING are unaffected by `cost_fn` -- only the reported cost changes."""
    area_ha = block.boundary.area / 1e4

    def _density(prefix: GeoDataFrame) -> float:
        return float(prefix.geometry.length.sum()) / area_ha if area_ha > 0 else 0.0

    fn = cost_fn if cost_fn is not None else _density
    costs: list[float] = [fn(cast(GeoDataFrame, roads.iloc[:0]))]
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
        costs.append(fn(ordered.iloc[:m]))
        vals.append(value(ordered.iloc[:m]))
    return costs, vals


def _cost_fn_for(block: Block, cost: str,
                 corridor_m: float) -> Callable[[GeoDataFrame], float] | None:
    """The `_sweep` cost_fn for `cost` ("length" | "displacement"): None (density default) for
    "length", else cumulative buildings displaced (via block.building_points) for "displacement"."""
    if cost == "length":
        return None
    if cost == "displacement":
        return lambda prefix: float(displacement_count(block.building_points, prefix, corridor_m))
    raise ValueError(f"cost must be 'length' or 'displacement', got {cost!r}")


def cost_benefit_curve(block: Block, roads: GeoDataFrame, *,
                       benefit_fn: BenefitFactory = access_benefit,
                       cost: str = "length", corridor_m: float = 3.0,
                       n_points: int = 20, tol: float = STREET_TOL) -> Curve:
    """Order roads by drainage descending, then at n_points cumulative-length budgets score
    benefit_fn's benefit vs the no-roads baseline. `benefit_fn` is a factory:
    (block, roads_full) -> f(roads_prefix), given the FULL road set so it can freeze any
    graph/entry state against it (see efficiency_benefit/directness_benefit); it is then
    called with growing prefixes of `roads` (starting with the empty prefix as baseline).
    `cost` selects the x-axis: "length" (road density, m/ha, default) or "displacement"
    (cumulative buildings whose site lies in the union of committed roads' `corridor_m`
    buffer, via block.building_points -- see `displacement_count`). Only the reported cost
    changes; benefit is computed identically either way."""
    cost_fn = _cost_fn_for(block, cost, corridor_m)
    costs, benefit = _sweep(block, roads, benefit_fn(block, roads, tol=tol), n_points, tol, cost_fn)
    return Curve(costs, benefit)


def efficiency_directness_curves(block: Block, roads: GeoDataFrame, *, cost: str = "length",
                                 corridor_m: float = 3.0, n_points: int = 20,
                                 tol: float = STREET_TOL) -> tuple[Curve, Curve]:
    """ONE sampled shortest-path sweep yielding both E and directness curves. The efficiency
    factory returns (E, directness) per prefix, so a single `_sweep` yields both -- avoids the
    doubled ~n_points x K Dijkstra pass of scoring efficiency_benefit and directness_benefit
    separately. `cost`/`corridor_m`: see `cost_benefit_curve`."""
    f = _efficiency_factory(block, roads, tol)          # prefix -> (E, directness)
    cost_fn = _cost_fn_for(block, cost, corridor_m)
    costs, pairs = _sweep(block, roads, f, n_points, tol, cost_fn)
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
