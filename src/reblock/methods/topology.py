"""TopologyMethod: wrap topology's greedy road-builder into a Proposal."""
from __future__ import annotations

import random
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from topology import MyEdge, build_all_roads

from reblock.contracts import Block, Proposal
from reblock.derive.parcel_graph import to_parcel_graph


def _edge_line(edge: MyEdge, origin: tuple[float, float]) -> LineString:
    a, b = edge.nodes
    return LineString([(a.x + origin[0], a.y + origin[1]),
                       (b.x + origin[0], b.y + origin[1])])


@dataclass
class TopologyMethod:
    alpha: float = 2.0
    seed: int = 0

    def propose(self, block: Block) -> Proposal:
        ppg = to_parcel_graph(block)
        graph = ppg.graph
        # NOTE (Slice 1, decision 10 gap): Block.streets == the block boundary here, so
        # define_roads() (outer-face detection) is used as the initial road set instead of
        # deriving it from Block.streets. Slice 2 must map Block.streets -> initial road
        # edges instead, so real OSM streets that are interior frontage are honored.
        graph.define_roads()                 # boundary edges = initial streets
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
                        method="topology", params={"alpha": self.alpha, "seed": self.seed})
