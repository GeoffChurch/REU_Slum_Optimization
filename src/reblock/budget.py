"""Cost-benefit curves for reblocking methods: add a method's roads incrementally in
drainage order, score access at each budget, trace benefit (fraction of Sigma depth^2
removed) vs cost (road density, m/ha). AUC = a 0-1 efficiency score. See the design spec.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import pandas as pd
from geopandas import GeoDataFrame
from shapely import STRtree
from shapely.geometry import Point
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


@dataclass(frozen=True)
class Curve:
    cost: list[float]     # cumulative road density, m/ha
    benefit: list[float]  # fraction of Sigma depth^2 removed, in [0, 1]


def cost_benefit_curve(block: Block, roads: GeoDataFrame, *, n_points: int = 20,
                       tol: float = STREET_TOL) -> Curve:
    """Order roads by drainage descending, then at n_points cumulative-length budgets score
    benefit = fraction of Sigma depth^2 removed vs the no-roads baseline. Adjacency is built
    once and reused across prefixes."""
    adj = parcel_adjacency(list(block.parcels.geometry), tol)
    base = access_burden(parcel_access_layers(block, None, tol=tol, adj=adj))
    cost, benefit = [0.0], [0.0]
    if len(roads) == 0 or base == 0.0 or block.boundary.area == 0.0:
        return Curve(cost, benefit)
    drain = road_drainage(block, roads, tol=tol)
    order = sorted(range(len(roads)), key=lambda i: (-drain[i], i))
    ordered = roads.iloc[order].reset_index(drop=True)
    lengths = ordered.geometry.length.to_numpy()
    cum = lengths.cumsum()
    total = float(cum[-1])
    area_ha = block.boundary.area / 1e4
    seen = 0
    for k in range(1, n_points + 1):
        m = int((cum <= (k / n_points) * total + 1e-9).sum())
        if m <= seen:
            continue
        seen = m
        depths = parcel_access_layers(block, ordered.iloc[:m], tol=tol, adj=adj)
        b = 1.0 - access_burden(depths) / base
        cost.append(float(cum[m - 1]) / area_ha)
        benefit.append(b)
    return Curve(cost, benefit)


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
