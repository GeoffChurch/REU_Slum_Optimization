"""How a candidate chord becomes a buildable road: `_snap` routes it along the parcel-boundary
graph, hugging the ideal line as closely as the graph allows.
"""
from __future__ import annotations

import networkx as nx
import shapely
from shapely.geometry import LineString, Point

from reblock.methods.arterial.primitives import _SnapGraph, _xy
from reblock.methods.boundary_graph import _rnd


def _snap(chord: LineString, sg: _SnapGraph, lam: float) -> LineString | None:
    """Buildable realization: the boundary-graph path between the chord endpoints' nearest
    nodes that hugs the ideal line (edge cost = length + lam * dist(edge midpoint, chord)).
    None if the endpoints snap to the same node or no path exists.

    The per-edge weight is computed ONCE per chord via shapely's vectorized `shapely.distance`
    ufunc over the block's precomputed `edge_midpoints` (bit-identical to the scalar
    `Point.distance(chord)` the naive per-edge loop would call -- both call into the same GEOS
    routine) instead of a Python-loop shapely call per edge per Dijkstra relaxation. The result is
    folded into an `{(u, v): weight}` dict (both directions, since the graph is undirected and
    `nx.shortest_path`'s callback may see either); the weight callback only looks it up, so the
    same `nx.shortest_path` call, weights, and (hence) equal-cost tie-break as before -- identical
    path geometry, minus the per-edge shapely cost."""
    p, q = _rnd(_xy(chord.coords[0])), _rnd(_xy(chord.coords[-1]))
    np_ = sg.nodes[int(sg.node_tree.nearest(Point(p)))]
    nq_ = sg.nodes[int(sg.node_tree.nearest(Point(q)))]
    if np_ == nq_:
        return None

    dists = shapely.distance(sg.mid_points, chord)
    weights = sg.lengths + lam * dists
    weight_map: dict[tuple[tuple[float, float], tuple[float, float]], float] = {}
    for (u, v), wt in zip(sg.edges, weights, strict=True):
        fwt = float(wt)
        weight_map[(u, v)] = fwt
        weight_map[(v, u)] = fwt

    def w(u: tuple[float, float], v: tuple[float, float], d: dict[str, float]) -> float:
        del d
        return weight_map[(u, v)]

    try:
        path = nx.shortest_path(sg.g, np_, nq_, weight=w)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return LineString([tuple(node) for node in path])
