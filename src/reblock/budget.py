"""Cost-benefit curves for reblocking methods: add a method's roads incrementally in
drainage order, score access at each budget, trace benefit (fraction of Sigma depth^2
removed) vs cost (road density, m/ha). AUC = a 0-1 efficiency score. See the design spec.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, cast

import networkx as nx
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency


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


def _road_street_graph(block: Block, roads: GeoDataFrame | None, tol: float) -> nx.Graph:
    """Graph over the road segments PLUS block.streets (so inter-parcel trips can use the
    street), nodes = snapped endpoints. (Shared with road_drainage's graph build.)"""
    g: nx.Graph = nx.Graph()
    lines = [] if roads is None else list(roads.geometry)
    lines += list(block.streets.geometry)
    for geom in lines:
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for part in parts:
            cs = list(part.coords)
            for a, b in zip(cs, cs[1:], strict=False):
                na, nb = _rnd(a), _rnd(b)
                if na != nb:
                    g.add_edge(na, nb, weight=Point(na).distance(Point(nb)))
    return g


def _edge_lines(g: nx.Graph) -> list[LineString]:
    """The graph's edges as single-segment LineStrings, one per undirected edge, in a stable
    order (the STRtree over these is the line-proximity entry index)."""
    return [LineString([u, v]) for u, v in g.edges()]


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
    against its edge so `_split_graph` injects it. Returns (per-parcel entry node or None, {edge
    index -> [(proj-distance-along-edge, _rnd(P))]}). Deterministic (nearest edge broken by index,
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


def _split_graph(g: nx.Graph, edges: list[LineString],
                 splits: dict[int, list[tuple[float, tuple[float, float]]]]) -> nx.Graph:
    """A copy of `g` with each edge that carries frozen line-entry projection points replaced by
    consecutive colinear sub-edges through those points (length-weighted). An edge absent from `g`
    (a road not in this prefix) is skipped -- its parcels' entry nodes then stay absent from the
    graph and contribute 0, exactly like a missing vertex entry, so freezing the splits against the
    full road set and evaluating prefixes stays monotone (edges only appear as roads are added;
    distances from fixed entries are non-increasing)."""
    h: nx.Graph = g.copy()
    for j, pts in splits.items():
        ls = edges[j]
        u, v = _rnd(ls.coords[0]), _rnd(ls.coords[-1])
        if not h.has_edge(u, v):
            continue
        h.remove_edge(u, v)
        chain: list[tuple[float, float]] = [u]
        for _proj, p in sorted(pts):
            if p != chain[-1]:
                chain.append(p)
        if v != chain[-1]:
            chain.append(v)
        for a, b in zip(chain, chain[1:], strict=False):
            if a != b:
                h.add_edge(a, b, weight=Point(a).distance(Point(b)))
    return h


def _dist_or_inf(dist: dict[tuple[float, float], float], e: tuple[float, float] | None) -> float:
    """`dist[e]` (a `nx.single_source_dijkstra_path_length` result) or +inf if `e` is None / not
    reached -- the sentinel `_sampled_efficiency` uses so a missing/unreached netdist drops out of
    a `np.isfinite` mask instead of needing a separate None check."""
    return math.inf if e is None else dist.get(e, math.inf)


def _sampled_efficiency(g: nx.Graph, entry: list[tuple[float, float] | None],
                        reps: list[Point], sources: list[int]) -> tuple[float, float]:
    """(E, directness) of the DOOR-TO-DOOR trip between every parcel pair, over graph `g`. The
    effective distance is the whole journey `d = leg_i + netdist(entry_i, entry_j) + leg_j`, where
    `leg_k = euclid(rep_k, entry_k)` is the last-mile walk from a parcel to where it joins the road
    -- NOT `netdist` alone. With the walk legs included, `euclid(rep_i, rep_j) <= d` by the triangle
    inequality (rep_i->entry_i->entry_j->rep_j, and netdist >= euclid(entry_i, entry_j)), so
    directness = mean(euclid/d) is a true circuity ratio bounded in [0, 1]; E = mean(1/d). Averaged
    over the FIXED all-parcel pair set (unreached pairs -- an entry missing, or absent/unreachable
    in `g` -- contribute 0); (0, 0) if there are no pairs. The legs are fixed once entries are
    frozen and `netdist` is non-increasing as roads grow, so both metrics stay monotone.

    Numpy-vectorized: the per-source Dijkstra call stays (networkx, over the graph `g`), but the
    O(K*N) leg/euclid/accumulation arithmetic that used to call shapely's `Point.distance` per
    pair is precomputed once as numpy arrays and reduced with masked sums."""
    n = len(entry)
    if n == 0 or not sources:
        return 0.0, 0.0
    rep_xy = np.array([[p.x, p.y] for p in reps], dtype=np.float64)
    entry_xy = np.array([[np.nan, np.nan] if e is None else [e[0], e[1]] for e in entry],
                        dtype=np.float64)
    legs = np.hypot(rep_xy[:, 0] - entry_xy[:, 0], rep_xy[:, 1] - entry_xy[:, 1])  # NaN if e None
    src_xy = rep_xy[sources]                                                     # (K, 2)
    src_euclid = np.hypot(src_xy[:, 0, None] - rep_xy[:, 0], src_xy[:, 1, None] - rep_xy[:, 1])

    inv_sum = dir_sum = 0.0
    pairs = 0
    for i, si in enumerate(sources):
        pairs += n - 1                            # all j != si, unchanged regardless of validity
        src = entry[si]
        dist = nx.single_source_dijkstra_path_length(g, src) if src is not None and src in g else {}
        nd = np.array([_dist_or_inf(dist, e) for e in entry], dtype=np.float64)
        d = legs[si] + nd + legs                  # door-to-door: walk + drive + walk
        mask = (entry[si] is not None) & np.isfinite(nd) & np.isfinite(legs) & (d > 0)
        mask[si] = False                          # exclude the self pair (j == si)
        inv_sum += float(np.sum(1.0 / d[mask]))
        dir_sum += float(np.sum(src_euclid[i, mask] / d[mask]))
    if pairs == 0:
        return 0.0, 0.0
    return inv_sum / pairs, dir_sum / pairs


def network_efficiency(block: Block, roads: GeoDataFrame | None, *, k: int = 40,
                       tol: float = STREET_TOL) -> tuple[float, float]:
    """Sampled (E, directness) of the network `roads` + block.streets. A parcel maps to the graph
    by line-proximity -- the nearest POINT on a road/street edge within `tol`, injected as a node
    by splitting that edge (`_line_entries`/`_split_graph`), which counts sparse straight chords the
    old nearest-vertex rule undercounted. From K seeded source parcels to ALL parcels; (0, 0) if the
    graph is empty. Deterministic: sources are evenly spaced over the parcel row order.

    Note this function alone is NOT guaranteed monotone as `roads` grows across separate calls,
    because each call re-derives entries against its own `roads` and entries can churn.
    `cost_benefit_curve`'s `efficiency_benefit`/`directness_benefit` factories get monotonicity
    instead by freezing entries against the FULL road set once, then only growing the edge set
    -- see `_efficiency_factory`."""
    g = _road_street_graph(block, roads, tol)
    n = len(block.parcels)
    if n < 2 or g.number_of_nodes() == 0:
        return 0.0, 0.0
    geoms = list(block.parcels.geometry)
    reps = [gm.representative_point() for gm in geoms]
    edges = _edge_lines(g)
    entry, splits = _line_entries(geoms, reps, edges, STRtree(edges), tol)
    g = _split_graph(g, edges, splits)
    step = max(1, n // k)
    sources = list(range(n))[::step][:k]
    return _sampled_efficiency(g, entry, reps, sources)


def _efficiency_factory(block: Block, roads_full: GeoDataFrame | None, tol: float,
                        k: int = 40) -> Callable[[GeoDataFrame | None], tuple[float, float]]:
    """Freeze the parcel->entry-node mapping and the K sampled sources against the FULL
    graph (`roads_full` + block.streets), built ONCE. The returned f(roads_prefix) computes
    (E, directness) from those FIXED entries, but over a subgraph containing only
    `roads_prefix` + block.streets edges (rounded coordinates keep node identity stable
    across subsets, so this is a true subgraph on the same node set). A source/dest whose
    fixed entry node is absent or unreachable in that subgraph contributes 0.

    Since the entry mapping, sources, and the all-parcel pair set never change while the
    edge set only grows as `roads_prefix` grows, shortest-path distances from fixed entries
    are non-increasing -- so E and directness are non-decreasing across cost_benefit_curve's
    prefixes, unlike calling `network_efficiency(block, roads_prefix)` per prefix (which
    re-derives entries against each prefix and can regress, see budget.py module docstring
    history / the review this fixes)."""
    g_full = _road_street_graph(block, roads_full, tol)
    n = len(block.parcels)
    if n < 2 or g_full.number_of_nodes() == 0:
        return lambda _roads: (0.0, 0.0)
    geoms = list(block.parcels.geometry)
    reps = [gm.representative_point() for gm in geoms]
    step = max(1, n // k)
    sources = list(range(n))[::step][:k]
    # Freeze the line entries + edge splits against the FULL graph once; each prefix's graph is
    # split at those FROZEN points (an edge missing from a prefix is skipped -> its entries stay
    # absent until its road appears), so E/directness are non-decreasing across prefixes.
    edges = _edge_lines(g_full)
    entry, splits = _line_entries(geoms, reps, edges, STRtree(edges), tol)

    def f(roads_prefix: GeoDataFrame | None) -> tuple[float, float]:
        g = _split_graph(_road_street_graph(block, roads_prefix, tol), edges, splits)
        return _sampled_efficiency(g, entry, reps, sources)
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
