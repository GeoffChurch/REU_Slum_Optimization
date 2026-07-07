"""Geometric access: shortest-path distance (metres) from each parcel to the nearest
street, on the parcel-adjacency graph weighted by centroid distance. Morphology-
sensitive where the topological peel (hops on a Voronoi tiling) is not.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd
from geopandas import GeoDataFrame
from shapely import union_all

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL
from reblock.derive.adjacency import parcel_adjacency


def geometric_access_distances(
    block: Block, roads: GeoDataFrame | None = None, *, tol: float = STREET_TOL
) -> pd.Series:
    ids = list(block.parcels["parcel_id"])
    geoms = list(block.parcels.geometry)
    cents = [g.representative_point() for g in geoms]
    adj = parcel_adjacency(geoms, tol)

    seed_geoms = list(block.streets.geometry)
    if roads is not None and not roads.empty:
        seed_geoms += list(roads.geometry)
    street = union_all(seed_geoms) if seed_geoms else None

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
