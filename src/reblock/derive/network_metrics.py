"""Network-quality metrics on a noded planar graph of (proposed roads ∪ existing
streets). Correct noding is load-bearing: set_precision(grid≈STREET_TOL) on every line
BEFORE union_all, so real ~0.5 m cadastral drift nodes at true intersections instead of
being missed (a post-hoc round only relabels, it cannot split an edge).
"""
from __future__ import annotations

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely import STRtree, line_merge, set_precision, union_all
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL
from reblock.derive.geometric_access import geometric_access_distances

_Node = tuple[float, float]


def _snap(x: float, y: float, tol: float) -> _Node:
    return (round(x / tol) * tol, round(y / tol) * tol)


def _iter_lines(geom: BaseGeometry) -> list[LineString]:
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    return []


def node_network(
    roads: gpd.GeoDataFrame | None, streets: gpd.GeoDataFrame, tol: float = STREET_TOL,
) -> nx.Graph:
    road_lines = list(roads.geometry) if roads is not None and not roads.empty else []
    street_lines = list(streets.geometry) if not streets.empty else []
    tagged: list[tuple[LineString, bool]] = []
    for is_road, src in ((True, road_lines), (False, street_lines)):
        snapped = union_all([set_precision(g, tol) for g in src]) if src else None
        if snapped is None or snapped.is_empty:
            continue
        for ls in _iter_lines(line_merge(snapped)):
            tagged.append((ls, is_road))

    graph: nx.Graph = nx.Graph()
    for ls, is_road in tagged:
        coords = list(ls.coords)
        for (x0, y0), (x1, y1) in zip(coords, coords[1:], strict=False):
            u, v = _snap(x0, y0, tol), _snap(x1, y1, tol)
            if u == v:
                continue
            seg = LineString([u, v])
            if graph.has_edge(u, v):
                graph[u][v]["is_road"] = graph[u][v]["is_road"] or is_road
            else:
                graph.add_edge(u, v, length=seg.length, is_road=is_road)
    return graph


def meshedness(graph: nx.Graph) -> float:
    """(E - N + C) / (2N - 5): 0 for a tree/forest, up to 1 for a maximal planar mesh."""
    n, e = graph.number_of_nodes(), graph.number_of_edges()
    if n < 3:
        return 0.0
    c = nx.number_connected_components(graph)
    denom = 2 * n - 5
    return max(0.0, (e - n + c) / denom) if denom > 0 else 0.0


def degree_fractions(graph: nx.Graph) -> dict[str, float]:
    n = graph.number_of_nodes()
    if n == 0:
        return {"four_way_fraction": 0.0, "dead_end_fraction": 0.0, "t_fraction": 0.0}
    degs = [d for _, d in graph.degree()]
    return {
        "four_way_fraction": sum(d >= 4 for d in degs) / n,
        "dead_end_fraction": sum(d == 1 for d in degs) / n,
        "t_fraction": sum(d == 3 for d in degs) / n,
    }


def crossing_counts(graph: nx.Graph) -> dict[str, int]:
    """Bare-degree node taxonomy (the through-going/collinear refinement is deferred to
    Phase 1): degree>=4 -> crossing, degree 3 -> T, degree-1 road tip -> dead-end."""
    n_cross = n_t = n_dead = 0
    for node, deg in graph.degree():
        if deg >= 4:
            n_cross += 1
        elif deg == 3:
            n_t += 1
        elif deg == 1 and any(graph[node][nb]["is_road"] for nb in graph[node]):
            n_dead += 1
    return {"n_crossings": n_cross, "n_t_junctions": n_t, "n_dead_ends": n_dead}


def _road_lines(roads: gpd.GeoDataFrame | None) -> list[LineString]:
    if roads is None or roads.empty:
        return []
    out: list[LineString] = []
    for g in roads.geometry:
        out.extend(_iter_lines(g))
    return out


def _side(b: LineString, x: float, y: float) -> int:
    """Which side of `b`'s NEAREST segment the point (x, y) lies on (+1/-1), or 0 if within
    `STREET_TOL` of it. Using the nearest segment (not the endpoint-chord) is robust to long
    jagged real frontages: a boundary-hugging stub's on-frontage vertex reads 0 and its
    in-block vertex reads its true side, so the stub is not miscounted as crossing.

    The "on it" test compares the point's actual (shapely-computed) distance to the nearest
    segment against `STREET_TOL` -- the same real-world "same feature" precision
    `_crosses_boundary` already uses for its own distance gate -- rather than a raw
    un-normalized cross-product threshold. Two empirical failure modes on real Cape Town
    clusters drove this: (1) the raw cross product scales with segment length, so a
    length-blind `cross > 1e-9` is far tighter than double-precision noise at UTM-scale
    coordinates (~1e6-1e7 m) once a chord runs 100+ m -- on ZAF.9.3.1_1_16951+_17068 (449 m
    frontage split into four ~100-170 m straight sub-chords, so the nearest-segment search
    alone is a no-op there) a point sitting exactly on a 170 m chord raised a raw cross of
    ~2e-8 from coordinate noise alone, past the 1e-9 floor; (2) even with an
    infinite-precision cross product, a block-local peel stub can run genuinely ~0.1-0.3 m
    off an interior chord that is itself a straight-line idealization of a slightly wiggly
    real cadastral edge (verified on ZAF.9.3.1_1_23732+_23733 and _46298+_46299) -- still
    "on" that boundary in every real-world sense the rest of this module treats as one
    feature. Both produced false "both sides" crossings on the reconciled baseline (spec
    says this MUST be 0 -- block-local roads can't cross a block boundary by construction).
    Gating on real distance against `STREET_TOL` (not a bare sign) fixes both to 0."""
    pt = Point(x, y)
    coords = list(b.coords)
    best: tuple[tuple[float, float], tuple[float, float]] | None = None
    best_d = float("inf")
    for i in range(len(coords) - 1):
        ax, ay = coords[i][0], coords[i][1]
        cx, cy = coords[i + 1][0], coords[i + 1][1]
        d = pt.distance(LineString([(ax, ay), (cx, cy)]))
        if d < best_d:
            best_d, best = d, ((ax, ay), (cx, cy))
    if best is None or best_d <= STREET_TOL:
        return 0
    (x0, y0), (x1, y1) = best
    cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
    return 1 if cross > 0 else -1


def _crosses_boundary(line: LineString, interior: MultiLineString, tol: float) -> bool:
    """True iff `line` has vertices strictly on both sides of some interior boundary
    segment AND runs within `tol` of it — robust where shapely's `.crosses` is not
    (runs-along -> side 0 only; kiss-and-bounce -> one side only)."""
    for b in interior.geoms:
        sides = {_side(b, x, y) for x, y in line.coords}
        if {-1, 1} <= sides and line.distance(b) <= tol:
            return True
    return False


def n_cross_block_streets(
    roads: gpd.GeoDataFrame | None, interior: MultiLineString, tol: float = STREET_TOL) -> int:
    if interior.is_empty:
        return 0
    return sum(_crosses_boundary(ls, interior, tol) for ls in _road_lines(roads))


def cross_block_trunk_length_m(
    roads: gpd.GeoDataFrame | None, interior: MultiLineString, tol: float = STREET_TOL) -> float:
    if interior.is_empty:
        return 0.0
    return float(sum(ls.length for ls in _road_lines(roads)
                     if _crosses_boundary(ls, interior, tol)))


def boundary_redundant_road_fraction(
    roads: gpd.GeoDataFrame | None, interior: MultiLineString, tol: float = STREET_TOL,
    band: float = 20.0) -> float:
    """Fraction of road length running within `band` of an interior boundary WITHOUT
    crossing it — the boundary-parallel spine road a shared through-trunk would merge."""
    lines = _road_lines(roads)
    total = sum(ls.length for ls in lines)
    if total == 0 or interior.is_empty:
        return 0.0
    corridor = interior.buffer(band)
    redundant = sum(ls.intersection(corridor).length for ls in lines
                    if not _crosses_boundary(ls, interior, tol))
    return float(redundant / total)


def circuity(block: Block, roads: gpd.GeoDataFrame | None, tol: float = STREET_TOL) -> float:
    """mean(network distance / straight-line distance) from each off-street parcel to the
    nearest street-frontage parcel, on the parcel-adjacency graph. Numerator and
    denominator share endpoints (parcel centroid -> frontage-parcel centroid), so the
    floor is 1.0 (direct) and detours score higher. geometric_access_distances anchors a
    street-touching parcel's network distance at 0, so the euclidean baseline is measured
    to the frontage-parcel centroid the network path actually reaches, not to the street
    line (which would make the ratio spuriously < 1)."""
    net = geometric_access_distances(block, roads, tol=tol)
    cents = {pid: g.representative_point()
             for pid, g in zip(block.parcels["parcel_id"], block.parcels.geometry, strict=True)}
    frontage = [pid for pid in net.index if float(net.loc[pid]) <= tol]
    if not frontage:
        return 1.0
    frontage_pts = [cents[pid] for pid in frontage]
    tree = STRtree(frontage_pts)
    ratios: list[float] = []
    for pid in net.index:
        nd = float(net.loc[pid])
        if nd <= tol:
            continue
        idx = int(tree.nearest(cents[pid]))
        euc = cents[pid].distance(frontage_pts[idx])
        if euc > tol and nd < float("inf"):
            ratios.append(nd / euc)
    return float(pd.Series(ratios).mean()) if ratios else 1.0


def throughput_ratio(graph: nx.Graph, block: Block, tol: float = STREET_TOL) -> float:
    """Max-flow from a unit-demand super-source over parcel access-nodes to a
    super-sink at the perimeter, normalized by parcel count: 1.0 = no bottleneck,
    lower = the network chokes. Each undirected graph edge becomes two directed
    edges of unit capacity; multiple parcels nearest the same node accumulate
    demand on that node's super-source edge rather than overwriting it."""
    if graph.number_of_nodes() == 0:
        return 0.0
    nodes = list(graph.nodes)
    node_pts = [Point(n) for n in nodes]
    tree = STRtree(node_pts)

    flow: nx.DiGraph = nx.DiGraph()
    for u, v in graph.edges:
        flow.add_edge(u, v, capacity=1.0)
        flow.add_edge(v, u, capacity=1.0)

    perim = block.boundary.boundary   # ring(s); .boundary (not .exterior) handles a MultiPolygon
    sink = "__SINK__"
    for n, pt in zip(nodes, node_pts, strict=True):
        if pt.distance(perim) <= tol:
            flow.add_edge(n, sink, capacity=float("inf"))

    src = "__SRC__"
    demand = 0
    for geom in block.parcels.geometry:
        idx = int(tree.nearest(geom.representative_point()))
        node = nodes[idx]
        if flow.has_edge(src, node):
            flow[src][node]["capacity"] += 1.0
        else:
            flow.add_edge(src, node, capacity=1.0)
        demand += 1

    if demand == 0 or src not in flow or sink not in flow:
        return 0.0
    return float(nx.maximum_flow_value(flow, src, sink)) / demand
