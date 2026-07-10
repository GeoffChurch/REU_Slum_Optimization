"""GreedyArterialReblocker: greedily insert the single straight arterial with the best
objective gain per meter, one at a time, until a road budget runs out. Two modes -- buildable
(snapped to the parcel-boundary graph) and aspirational (ideal chords) -- so the compare reports
the price of buildability. Candidates are through-roads (network<->network) + spurs
(network->deep pocket); continuations are through-roads from committed-segment endpoints (always
anchors), so a spur completes into a through-road for free and crossings planarize into true
intersections. See docs/superpowers/specs/2026-07-09-greedy-arterial-reblocker-design.md.
"""
from __future__ import annotations

import geopandas as gpd
import networkx as nx
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.methods.dijkstra import _rnd


def _xy(c: tuple[float, ...]) -> tuple[float, float]:
    """First two components of a coordinate tuple (shapely yields 3-tuples for Z-aware
    geometry; every geometry here is 2-D, so drop anything past x, y)."""
    return (c[0], c[1])


def _anchor_points(network: list[LineString], n: int) -> list[tuple[float, float]]:
    """`n` points sampled evenly by arc-length along the merged network, plus every network
    vertex (so committed-segment endpoints are always anchors -> continuations come for free).
    _rnd-snapped, de-duplicated, sorted for determinism."""
    merged = unary_union(network)
    lines = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    pts: set[tuple[float, float]] = set()
    for ln in lines:
        pts.update(_rnd(_xy(c)) for c in ln.coords)                  # vertices
    total = sum(ln.length for ln in lines)
    if total > 0 and n > 0:
        step = total / n
        for ln in lines:
            d = 0.0
            while d <= ln.length:
                pts.add(_rnd(_xy(ln.interpolate(d).coords[0])))
                d += step
    return sorted(pts)


def _deep_targets(block: Block, roads: GeoDataFrame | None, k: int,
                   adj: list[set[int]]) -> list[tuple[float, float]]:
    """Representative points of the k deepest-access parcels (spur targets), _rnd-snapped."""
    depths = parcel_access_layers(block, roads, tol=STREET_TOL, adj=adj)
    order = depths.sort_values(ascending=False, kind="stable")
    id_to_pos = {pid: i for i, pid in enumerate(block.parcels["parcel_id"])}
    geoms = list(block.parcels.geometry)
    out: list[tuple[float, float]] = []
    for pid in list(order.index)[:k]:
        rep = geoms[id_to_pos[pid]].representative_point()
        out.append(_rnd(_xy(rep.coords[0])))
    return out


def _candidate_chords(anchors: list[tuple[float, float]],
                       targets: list[tuple[float, float]]) -> list[LineString]:
    """Through-roads (anchor pairs) + spurs (anchor -> deep target). De-duplicated, sorted."""
    seen: set[frozenset[tuple[float, float]]] = set()
    chords: list[LineString] = []
    for i, a in enumerate(anchors):
        for b in anchors[i + 1:]:
            key = frozenset((a, b))
            if a != b and key not in seen:
                seen.add(key)
                pair: list[tuple[float, float]] = sorted((a, b))
                chords.append(LineString(pair))
        for t in targets:
            key = frozenset((a, t))
            if a != t and key not in seen:
                seen.add(key)
                spur: list[tuple[float, float]] = sorted((a, t))
                chords.append(LineString(spur))
    return sorted(chords, key=lambda ls: ls.wkt)


def _snap(chord: LineString, g: nx.Graph, node_tree: STRtree,
          nodes: list[tuple[float, float]], lam: float) -> LineString | None:
    """Buildable realization: the boundary-graph path between the chord endpoints' nearest
    nodes that hugs the ideal line (edge cost = length + lam * dist(edge midpoint, chord)).
    None if the endpoints snap to the same node or no path exists."""
    p, q = _rnd(_xy(chord.coords[0])), _rnd(_xy(chord.coords[-1]))
    np_ = nodes[int(node_tree.nearest(Point(p)))]
    nq_ = nodes[int(node_tree.nearest(Point(q)))]
    if np_ == nq_:
        return None

    def w(u: tuple[float, float], v: tuple[float, float], d: dict[str, float]) -> float:
        mid = Point((u[0] + v[0]) / 2, (u[1] + v[1]) / 2)
        return float(d["weight"]) + lam * mid.distance(chord)

    try:
        path = nx.shortest_path(g, np_, nq_, weight=w)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return LineString([tuple(node) for node in path])


def _planarize(lines: list[LineString], crs: CRS) -> GeoDataFrame:
    """unary_union the lines (nodes crossings), explode to LineStrings, one row each."""
    if not lines:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
    merged = unary_union(lines)
    parts: list[BaseGeometry] = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    rows = [ln for ln in parts if "LineString" in ln.geom_type and ln.length > 0]
    return gpd.GeoDataFrame({"geometry": rows}, geometry="geometry", crs=crs)
