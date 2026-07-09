"""MeshReblocker: the dijkstra forest plus crossing roads. Closes boundary-graph loops in
descending shortcut-ratio order (forest-path-distance / edge-length -- a one-BFS proxy for
circuity reduction), so the network gains through-roads and redundancy the tree lacks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL
from reblock.methods.dijkstra import _boundary_graph, _reblock_dijkstra, _rnd


def _mesh_roads(block: Block) -> gpd.GeoDataFrame:
    forest = _reblock_dijkstra(block)
    g = _boundary_graph(block.parcels)
    forest_edges = {
        frozenset((_rnd(cast("tuple[float, float]", a)), _rnd(cast("tuple[float, float]", b))))
        for line in forest.geometry
        for a, b in zip(list(line.coords), list(line.coords)[1:], strict=False)}
    fg = nx.Graph()
    for e in forest_edges:
        u, v = tuple(e)
        fg.add_edge(u, v, weight=Point(u).distance(Point(v)))
    street = unary_union(list(block.streets.geometry))
    corridor = street.buffer(STREET_TOL)
    # candidate loops: graph edges NOT in the forest with both endpoints already on the forest
    cands = []
    for u, v, w in g.edges(data="weight"):
        if frozenset((u, v)) in forest_edges or u not in fg or v not in fg:
            continue
        if LineString(sorted((u, v))).within(corridor):   # coincides with a street: not a new road
            continue
        try:
            fp = nx.shortest_path_length(fg, u, v, weight="weight")
        except nx.NetworkXNoPath:
            continue
        cands.append((fp / w, w, frozenset((u, v))))   # (shortcut ratio, len, edge)
    cands.sort(key=lambda c: (-c[0], sorted(c[2])))
    loops = [LineString(sorted(e)) for ratio, _w, e in cands if ratio > 1.0]   # a real detour

    rows = [{"geometry": geom, "drain": d} for geom, d in
            zip(forest.geometry, forest["drain"], strict=True)]
    rows += [{"geometry": geom, "drain": 0} for geom in loops]   # loops: not tree carriers
    gdf = gpd.GeoDataFrame(rows, columns=["geometry", "drain"], geometry="geometry",
                           crs=block.crs)
    gdf.attrs["loops"] = len(loops)
    return gdf


@dataclass
class MeshReblocker:
    @property
    def identity(self) -> tuple[str]:
        return ("mesh",)

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior
        roads = _mesh_roads(block)
        loops = int(roads.attrs.get("loops", 0))
        return Proposal(block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
                        proposal_id="mesh", method="mesh",
                        params={"segments": len(roads), "loops": loops},
                        block_identity=block.identity)
