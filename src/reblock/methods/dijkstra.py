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
from typing import cast

import geopandas as gpd
import networkx as nx
from shapely import STRtree
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL
from reblock.methods.boundary_graph import _boundary_graph

INF = float("inf")


def _reblock_dijkstra(block: Block) -> gpd.GeoDataFrame:
    if block.streets.empty:
        raise ValueError(
            "Block.streets must be non-empty: with no street frontage the forest has no "
            "root -- every parcel would be a false depth-1 and the proposal an empty no-op")
    parcels = block.parcels
    g = _boundary_graph(parcels)
    edges: list[tuple[tuple[float, float], tuple[float, float]]] = sorted(g.edges())
    edge_geoms = [LineString([a, b]) for a, b in edges]
    tree = STRtree(edge_geoms)
    street = unary_union(list(block.streets.geometry))
    corridor = street.buffer(STREET_TOL)
    snodes = {n for n in g.nodes if Point(n).distance(street) <= STREET_TOL}
    dist, paths = nx.multi_source_dijkstra(g, sorted(snodes)) if snodes else ({}, {})

    drain: dict[frozenset[tuple[float, float]], int] = defaultdict(int)
    info: list[tuple[list[tuple[tuple[float, float], tuple[float, float]]],
                     tuple[float, float] | None]] = []
    n_unreachable = 0
    for geom in parcels.geometry:
        ext = cast(Polygon, geom).exterior
        ring = ext.buffer(STREET_TOL)
        # boundary edges of THIS parcel = graph edges lying along its exterior (robust to
        # T-junctions, where union-noding split an edge the parcel's raw coords no longer match).
        pes = [edges[i] for i in sorted(tree.query(ext, predicate="dwithin", distance=STREET_TOL))
               if edge_geoms[i].within(ring)]
        if pes and any(LineString(list(e)).within(corridor) for e in pes):
            info.append((pes, None))                 # street-fronting: already served
            continue
        pn = [n for e in pes for n in e if n in dist]
        if not pn:
            n_unreachable += 1                        # non-street parcel, no route to a street
            info.append((pes, None))
            continue
        entry = min(pn, key=lambda n: (dist[n], n))
        for a, b in zip(paths[entry], paths[entry][1:], strict=False):
            drain[frozenset((a, b))] += 1
        info.append((pes, entry))

    forest = set(drain)
    for cov_edges, node in info:
        if node is None or any(frozenset(e) in forest for e in cov_edges):
            continue
        incident = [e for e in cov_edges if node in e]
        spur = min(incident, key=lambda e: (dist.get(e[0] if e[1] == node else e[1], INF), e))
        drain[frozenset(spur)] += 1

    items = sorted(drain.items(), key=lambda kv: (-kv[1], sorted(kv[0])))
    rows = [{"geometry": LineString(sorted(e)), "drain": d} for e, d in items]  # type: ignore[arg-type]
    gdf = gpd.GeoDataFrame(rows, columns=["geometry", "drain"], geometry="geometry",
                           crs=block.crs)
    gdf.attrs["unreachable"] = n_unreachable
    return gdf


@dataclass
class DijkstraReblocker:
    @property
    def identity(self) -> tuple[str]:
        return ("dijkstra",)

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior  # accepted for Method conformance; the routing is block-only
        roads = _reblock_dijkstra(block)
        spurs = int((roads["drain"] == 1).sum()) if len(roads) else 0
        return Proposal(block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
                        proposal_id="dijkstra", method="dijkstra",
                        params={"segments": len(roads), "leaf_roads": spurs,
                                "unreachable": int(roads.attrs.get("unreachable", 0))},
                        block_identity=block.identity)
