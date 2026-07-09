"""DijkstraReblocker: route roads along the parcel-boundary graph as a shortest-path
forest rooted at the street (drainage-weighted, coverage-complete).

Deterministic, network-forming alternative to PeelReblocker's center-to-center descent:
roads follow parcel frontages (buildable) instead of cutting through parcels, shared
route-prefixes coalesce into arterials, and every segment reaches the street (so the
k-metric's street_connectivity grants each fronted parcel depth-1 access).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL

INF = float("inf")


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
        for a, b in zip(cs, cs[1:]):
            na, nb = _rnd(a), _rnd(b)
            if na != nb:
                edges.add((min(na, nb), max(na, nb)))
    g = nx.Graph()
    for na, nb in sorted(edges):
        g.add_edge(na, nb, weight=Point(na).distance(Point(nb)))
    return g


def _reblock_dijkstra(block: Block) -> gpd.GeoDataFrame:
    parcels = block.parcels
    g = _boundary_graph(parcels)
    street = unary_union(list(block.streets.geometry))
    corridor = street.buffer(STREET_TOL)
    snodes = {n for n in g.nodes if Point(n).distance(street) <= STREET_TOL}
    dist, paths = (nx.multi_source_dijkstra(g, snodes) if snodes else ({}, {}))

    drain: dict[frozenset[tuple[float, float]], int] = defaultdict(int)
    info: list[tuple[list[tuple[tuple[float, float], tuple[float, float]]],
                     tuple[float, float] | None]] = []
    # 1. shortest-path forest: route each non-street parcel's nearest node to the street.
    for geom in parcels.geometry:
        coords = [_rnd(c) for c in geom.exterior.coords]
        pes = [(a, b) for a, b in zip(coords, coords[1:]) if g.has_edge(a, b)]
        if not pes or any(LineString([a, b]).within(corridor) for a, b in pes):
            info.append((pes, None))                        # street-fronting -> served
            continue
        pn = [n for e in pes for n in e if n in dist]
        if not pn:
            info.append((pes, None))
            continue
        entry = min(pn, key=lambda n: (dist[n], n))          # deterministic
        for a, b in zip(paths[entry], paths[entry][1:]):
            drain[frozenset((a, b))] += 1
        info.append((pes, entry))

    forest = set(drain)
    # 2. coverage spurs: a parcel served only at a vertex gets its boundary edge incident
    #    to its routing node (so the spur attaches to the forest -- never floating).
    for pes, entry in info:
        if entry is None or any(frozenset(e) in forest for e in pes):
            continue
        incident = [e for e in pes if entry in e]
        if not incident:
            continue
        spur = min(incident, key=lambda e: (dist.get(e[0] if e[1] == entry else e[1], INF), e))
        drain[frozenset(spur)] += 1

    items = sorted(drain.items(), key=lambda kv: (-kv[1], sorted(kv[0])))
    rows = [{"geometry": LineString(sorted(e)), "drain": d} for e, d in items]
    return gpd.GeoDataFrame(rows, columns=["geometry", "drain"], geometry="geometry",
                            crs=block.crs)
