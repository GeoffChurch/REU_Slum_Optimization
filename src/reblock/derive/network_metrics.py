"""Network-quality metrics on a noded planar graph of (proposed roads ∪ existing
streets). Correct noding is load-bearing: set_precision(grid≈STREET_TOL) on every line
BEFORE union_all, so real ~0.5 m cadastral drift nodes at true intersections instead of
being missed (a post-hoc round only relabels, it cannot split an edge).
"""
from __future__ import annotations

import geopandas as gpd
import networkx as nx
from shapely import line_merge, set_precision, union_all
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry

from reblock.derive.access import STREET_TOL

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
