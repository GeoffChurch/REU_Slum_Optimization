"""Cost-benefit curves for reblocking methods: add a method's roads incrementally in
drainage order, score access at each budget, trace benefit (fraction of Sigma depth^2
removed) vs cost (road density, m/ha). AUC = a 0-1 efficiency score. See the design spec.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, cast

import networkx as nx
import pandas as pd
from geopandas import GeoDataFrame
from shapely import STRtree
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency


def _rnd(c: tuple[float, ...]) -> tuple[float, float]:
    return (round(c[0], 2), round(c[1], 2))


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


def _entry_nodes(geoms: list[BaseGeometry], nodes: list[tuple[float, float]], tree: STRtree,
                 tol: float) -> list[tuple[float, float] | None]:
    """Each parcel's entry: its nearest graph node that its own geometry actually touches
    (dwithin `tol`, matching the touch tolerance `parcel_access_layers` uses) -- NOT the
    nearest node to an interior representative point within a generous multiple of `tol`.
    That looser variant let interior parcels several parcel-widths from any street snap
    straight onto the boundary ring "as the crow flies" even with zero roads, so most
    parcels already looked reachable before any road existed; adding roads then only
    lengthened those already-short ring hops into ring+spur+ring detours, so E and
    directness fell as roads were added -- the opposite of what the metric is meant to
    measure. Parcels touching no node are unreached this round (still counted as targets
    below, contributing 0)."""
    entry: list[tuple[float, float] | None] = []
    for geom in geoms:
        near = tree.query(geom, predicate="dwithin", distance=tol)
        entry.append(min((nodes[j] for j in near), key=lambda node: geom.distance(Point(node)),
                         default=None))
    return entry


def _sampled_efficiency(g: nx.Graph, entry: list[tuple[float, float] | None],
                        reps: list[Point], sources: list[int]) -> tuple[float, float]:
    """(E, directness) from each source parcel's entry node to every other parcel's entry
    node, over graph `g`. E = mean(1/d), directness = mean(euclid/d), each averaged over the
    FIXED set of all-parcel pairs (unreached pairs -- entry missing, or absent/unreachable in
    `g` -- contribute 0, the standard global-efficiency definition); (0, 0) if there are no
    pairs."""
    inv_sum = dir_sum = 0.0
    pairs = 0
    for si in sources:
        src = entry[si]
        dist = nx.single_source_dijkstra_path_length(g, src) if src is not None and src in g else {}
        for j in range(len(entry)):
            if j == si:
                continue
            pairs += 1
            d = dist.get(entry[j]) if entry[j] is not None else None
            if d and d > 0:
                inv_sum += 1.0 / d
                dir_sum += reps[si].distance(reps[j]) / d
    if pairs == 0:
        return 0.0, 0.0
    return inv_sum / pairs, dir_sum / pairs


def network_efficiency(block: Block, roads: GeoDataFrame | None, *, k: int = 40,
                       tol: float = STREET_TOL) -> tuple[float, float]:
    """Sampled (E, directness) of the network `roads` + block.streets, with entries fixed to
    THIS graph (i.e. `roads` serves as both the entry-node mapping and the edge set): from K
    seeded source parcels to ALL parcels. (0, 0) if the graph is empty. Deterministic: sources
    are evenly spaced over the parcel row order.

    Note this function alone is NOT guaranteed monotone as `roads` grows across separate
    calls, because each call re-derives entries against its own `roads` and entries can churn.
    `cost_benefit_curve`'s `efficiency_benefit`/`directness_benefit` factories get monotonicity
    instead by freezing entries against the FULL road set once, then only growing the edge set
    -- see `_entry_nodes`/`_sampled_efficiency` above, which this function also uses."""
    g = _road_street_graph(block, roads, tol)
    n = len(block.parcels)
    if n < 2 or g.number_of_nodes() == 0:
        return 0.0, 0.0
    geoms = list(block.parcels.geometry)
    reps = [gm.representative_point() for gm in geoms]
    nodes = list(g.nodes)
    tree = STRtree([Point(node) for node in nodes])
    entry = _entry_nodes(geoms, nodes, tree, tol)
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
    nodes = list(g_full.nodes)
    tree = STRtree([Point(node) for node in nodes])
    entry = _entry_nodes(geoms, nodes, tree, tol)
    step = max(1, n // k)
    sources = list(range(n))[::step][:k]

    def f(roads_prefix: GeoDataFrame | None) -> tuple[float, float]:
        g = _road_street_graph(block, roads_prefix, tol)
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
    cost: list[float]     # cumulative road density, m/ha
    benefit: list[float]  # fraction of Sigma depth^2 removed, in [0, 1]


V = TypeVar("V")


def _sweep(block: Block, roads: GeoDataFrame, value: Callable[[GeoDataFrame | None], V],
           n_points: int, tol: float) -> tuple[list[float], list[V]]:
    """Drainage-ordered cumulative-budget sweep: returns (costs m/ha, [value(prefix)]).
    Order roads by drainage descending, then at n_points cumulative-length budgets evaluate
    `value` on the empty-prefix baseline and each growing prefix (skipping budgets that add no
    new road). `value` maps a road prefix -> any V (a float for a single metric, a tuple for
    several -- run the whole sweep ONCE for a batched multi-metric value)."""
    costs: list[float] = [0.0]
    vals: list[V] = [value(cast(GeoDataFrame, roads.iloc[:0]))]
    if len(roads) == 0 or block.boundary.area == 0.0:
        return costs, vals
    drain = road_drainage(block, roads, tol=tol)
    order = sorted(range(len(roads)), key=lambda i: (-drain[i], i))
    ordered = roads.iloc[order].reset_index(drop=True)
    cum = ordered.geometry.length.to_numpy().cumsum()
    total, area_ha = float(cum[-1]), block.boundary.area / 1e4
    seen = 0
    for kk in range(1, n_points + 1):
        m = int((cum <= (kk / n_points) * total + 1e-9).sum())
        if m <= seen:
            continue
        seen = m
        costs.append(float(cum[m - 1]) / area_ha)
        vals.append(value(ordered.iloc[:m]))
    return costs, vals


def cost_benefit_curve(block: Block, roads: GeoDataFrame, *,
                       benefit_fn: BenefitFactory = access_benefit,
                       n_points: int = 20, tol: float = STREET_TOL) -> Curve:
    """Order roads by drainage descending, then at n_points cumulative-length budgets score
    benefit_fn's benefit vs the no-roads baseline. `benefit_fn` is a factory:
    (block, roads_full) -> f(roads_prefix), given the FULL road set so it can freeze any
    graph/entry state against it (see efficiency_benefit/directness_benefit); it is then
    called with growing prefixes of `roads` (starting with the empty prefix as baseline)."""
    costs, benefit = _sweep(block, roads, benefit_fn(block, roads, tol=tol), n_points, tol)
    return Curve(costs, benefit)


def efficiency_directness_curves(block: Block, roads: GeoDataFrame, *, n_points: int = 20,
                                 tol: float = STREET_TOL) -> tuple[Curve, Curve]:
    """ONE sampled shortest-path sweep yielding both E and directness curves. The efficiency
    factory returns (E, directness) per prefix, so a single `_sweep` yields both -- avoids the
    doubled ~n_points x K Dijkstra pass of scoring efficiency_benefit and directness_benefit
    separately."""
    f = _efficiency_factory(block, roads, tol)          # prefix -> (E, directness)
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
