"""Shared parcel-boundary graph helpers: nodes = boundary vertices (cm-snapped so shared
vertices coincide), edges = parcel boundary segments. Used by the arterial/loop-closure
scoring machinery and the substrate builders as the underlying planar tessellation graph.
"""
from __future__ import annotations

import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
from shapely.ops import unary_union


def _rnd(c: tuple[float, float]) -> tuple[float, float]:
    return (round(c[0], 2), round(c[1], 2))   # snap to cm so shared vertices coincide


def _boundary_graph(parcels: gpd.GeoDataFrame) -> nx.Graph:
    """Planar graph of the tessellation: nodes = boundary vertices, edges = parcel
    boundary segments (shared party-walls dedup via unary_union), weight = length.
    Edges are added in sorted order for determinism."""
    noded = unary_union([g.boundary for g in parcels.geometry])
    lines = list(noded.geoms) if hasattr(noded, "geoms") else [noded]
    edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for ln in lines:
        cs = list(ln.coords)
        for a, b in zip(cs, cs[1:], strict=False):
            na, nb = _rnd(a), _rnd(b)
            if na != nb:
                edges.add((min(na, nb), max(na, nb)))
    g = nx.Graph()
    for na, nb in sorted(edges):
        g.add_edge(na, nb, weight=Point(na).distance(Point(nb)))
    return g
