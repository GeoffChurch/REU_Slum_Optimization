"""TopologyMethod: wrap topology's greedy road-builder into a Proposal."""
from __future__ import annotations

import random
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from shapely import union_all
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from topology import MyEdge, MyGraph, build_all_roads

from reblock.contracts import Block, Proposal
from reblock.derive.parcel_graph import to_parcel_graph

# `to_parcel_graph`'s `clean_up_geometry(0.5, byblock=False)` step merges
# near-duplicate parcel vertices within this many units of each other, so a
# surviving graph node can sit up to this far from the raw vertex `Block.streets`
# was built from (`ShapefileSource` derives it from the *un-cleaned* parcel
# union). Matching edges to streets must tolerate that same drift, or real
# (noisy) boundary geometry silently under-matches -- see `_mark_streets_as_roads`.
_STREET_MATCH_TOL = 0.5


def _edge_line(edge: MyEdge, origin: tuple[float, float]) -> LineString:
    a, b = edge.nodes
    return LineString([(a.x + origin[0], a.y + origin[1]),
                       (b.x + origin[0], b.y + origin[1])])


def _streets_local_geometry(streets: gpd.GeoDataFrame,
                            origin: tuple[float, float]) -> BaseGeometry | None:
    """`Block.streets`, origin-shifted into the parcel graph's local frame."""
    lines = [LineString([(x - origin[0], y - origin[1]) for x, y in geom.coords])
             for geom in streets.geometry if isinstance(geom, LineString)]
    return union_all(lines) if lines else None


def _mark_streets_as_roads(graph: MyGraph, streets: gpd.GeoDataFrame,
                           origin: tuple[float, float]) -> None:
    """Set `edge.road = True` for graph edges coincident with `Block.streets`.

    "Coincident" means the *whole edge* lies within `_STREET_MATCH_TOL` of the
    street network -- tested as the edge line being `within` the street's
    `buffer(tol)` corridor, not merely its two endpoints being near a street.
    Endpoint-only proximity over-matches: an interior party line whose two
    endpoints happen to sit on the boundary (a chord across a notch/reflex
    corner) would be wrongly marked a road even though it detours far from any
    street. The buffer tolerance absorbs the sub-`tol` coordinate drift between
    `Block.streets` and the cleaned parcel graph (see `_STREET_MATCH_TOL`), so
    genuine boundary edges still match despite that drift.
    """
    street_geom = _streets_local_geometry(streets, origin)
    if street_geom is None:
        return
    corridor = street_geom.buffer(_STREET_MATCH_TOL)
    for edge in graph.myedges():
        a, b = edge.nodes
        if LineString([(a.x, a.y), (b.x, b.y)]).within(corridor):
            edge.road = True


@dataclass
class TopologyMethod:
    alpha: float = 2.0
    seed: int = 0

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        ppg = to_parcel_graph(block)
        graph = ppg.graph
        # `prior` is unused: topology's greedy road-builder is block-independent
        # (accepted only so TopologyMethod structurally satisfies Method).
        del prior
        _mark_streets_as_roads(graph, block.streets, ppg.origin)
        graph.define_interior_parcels()
        initial = {e for e in graph.myedges() if e.road}

        # build_all_roads is probabilistic: choose_path -> WeightedPick draws
        # from numpy's global RNG (np.random.choice), so np.random.seed is what
        # actually pins the road layout; seed random too for any stdlib draws.
        random.seed(self.seed)
        np.random.seed(self.seed)
        build_all_roads(graph, alpha=self.alpha, vquiet=True)

        # `e not in initial` is a value-equality set-difference: MyEdge.__eq__/__hash__ are
        # VALUE-based on the unordered pair of node locations (see MyEdge in
        # ext/topology/topology/graph/my_graph.py), not identity-based. So this correctly
        # finds edges newly marked as roads regardless of whether build_all_roads mutates
        # edges in place or topology otherwise preserves the original edge objects.
        new_edges = [e for e in graph.myedges() if e.road and e not in initial]
        roads = gpd.GeoDataFrame(geometry=[_edge_line(e, ppg.origin) for e in new_edges],
                                 crs=block.crs)
        all_edges = list(graph.myedges())
        edges = gpd.GeoDataFrame(
            {"road": [e.road for e in all_edges],
             "interior": [e.interior for e in all_edges],
             "barrier": [e.barrier for e in all_edges]},
            geometry=[_edge_line(e, ppg.origin) for e in all_edges], crs=block.crs)
        return Proposal(block_id=block.block_id, crs=block.crs, roads=roads, edges=edges,
                        proposal_id=f"topology_a{self.alpha}_s{self.seed}",
                        method="topology", params={"alpha": self.alpha, "seed": self.seed})
