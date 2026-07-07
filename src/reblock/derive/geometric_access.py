"""Geometric access: shortest-path distance (metres) from each parcel to the nearest
street, on the parcel-adjacency graph weighted by centroid distance. Morphology-
sensitive where the topological peel (hops on a Voronoi tiling) is not.

`geometric_access_max_m` / the per-parcel `pd.Series` returned here may be non-finite
(`inf`) for a parcel with no street-connected path in the adjacency graph -- unlike the
peel, which uses a finite sentinel one layer past the deepest reached layer. Downstream
aggregation (e.g. the planned cross-block rollup) must handle non-finite values.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd
from geopandas import GeoDataFrame

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, street_connectivity
from reblock.derive.adjacency import parcel_adjacency


def geometric_access_distances(
    block: Block, roads: GeoDataFrame | None = None, *, tol: float = STREET_TOL
) -> pd.Series:
    ids = list(block.parcels["parcel_id"])
    geoms = list(block.parcels.geometry)
    cents = [g.representative_point() for g in geoms]
    adj = parcel_adjacency(geoms, tol)

    street = street_connectivity(block.streets, roads, tol).seed_geom

    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(range(len(geoms)))
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            if i < j:
                graph.add_edge(i, j, weight=cents[i].distance(cents[j]))
    SRC = -1
    if street is not None:
        for i, g in enumerate(geoms):
            if g.distance(street) <= tol:
                graph.add_edge(SRC, i, weight=0.0)
    lengths = nx.single_source_dijkstra_path_length(graph, SRC) if SRC in graph else {}
    d = [float(lengths.get(i, float("inf"))) for i in range(len(geoms))]
    return pd.Series(d, index=pd.Index(ids, name="parcel_id"), dtype="float64")
