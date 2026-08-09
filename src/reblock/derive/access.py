"""Access-depth via a BFS parcel peel (kblock's k-complexity definition).

Layer 1 = touches a street; layer L = L-1 parcels from the nearest street.
Runs on parcels directly (STRtree adjacency + BFS) -- no topology graph, so
it is robust to the weak-dual degeneracies of `reblock.eval.kcomplexity`
(a single-file corridor scoring k=1 regardless of length; k silently
capping at 8) and is native per-parcel by `parcel_id`.
"""
from __future__ import annotations

from collections import deque
from typing import NamedTuple

import networkx as nx
import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from shapely import STRtree, union_all
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

from reblock.contracts import Block
from reblock.derive.adjacency import parcel_adjacency

# The `Block.streets` seam tolerance. `to_parcel_graph`'s
# `clean_up_geometry(0.5, byblock=False)` step merges parcel vertices within
# this many units, so anything gating that seam -- the peel's parcel-adjacency
# and street-seeding here, and `TopologyMethod`'s street-edge matching (which
# imports this same constant) -- must tolerate the same drift, or noisy real
# boundary geometry silently under-matches. One constant so the two never
# disagree.
STREET_TOL = 0.5


class StreetConnectivity(NamedTuple):
    seed_geom: BaseGeometry | None      # streets + road segments that reach a street
    n_components: int                   # touch-components among the road segments
    connected_frac: float               # road length in street-connected components / total


def _line_parts(roads: GeoDataFrame | None) -> list[BaseGeometry]:
    """Positive-length line parts of `roads` (explode Multi*; drop non-lines)."""
    if roads is None or roads.empty:
        return []
    parts: list[BaseGeometry] = []
    for g in roads.geometry:
        if g is None or g.is_empty:
            continue
        geoms: list[BaseGeometry] = list(g.geoms) if isinstance(g, BaseMultipartGeometry) else [g]
        parts.extend(p for p in geoms if "LineString" in p.geom_type and p.length > 0)
    return parts


def street_connectivity(
    streets: GeoDataFrame, roads: GeoDataFrame | None, tol: float
) -> StreetConnectivity:
    """Seed geometry = streets plus only those road segments whose touch-component
    reaches a street (variant a). Floating interior roads grant no access."""
    street_geom = union_all(list(streets.geometry)) if len(streets) else None
    segs = _line_parts(roads)
    if not segs:
        return StreetConnectivity(street_geom, 0, 0.0)
    tree = STRtree(segs)
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(range(len(segs)))
    for i, g in enumerate(segs):
        for j in tree.query(g, predicate="dwithin", distance=tol):
            jj = int(j)
            if i < jj:
                graph.add_edge(i, jj)
    comps = list(nx.connected_components(graph))
    live: list[int] = []
    for comp in comps:
        if street_geom is not None and any(segs[i].distance(street_geom) <= tol for i in comp):
            live.extend(comp)
    total = sum(segs[i].length for i in range(len(segs)))
    live_len = sum(segs[i].length for i in live)
    frac = live_len / total if total > 0 else 0.0
    parts = ([street_geom] if street_geom is not None else []) + [segs[i] for i in live]
    seed = union_all(parts) if parts else None
    return StreetConnectivity(seed, len(comps), frac)


def parcel_access_layers(
    block: Block, roads: GeoDataFrame | None, *, tol: float = STREET_TOL,
    adj: list[set[int]] | None = None, unreached_depth: int | None = None,
) -> pd.Series:
    """BFS-peel access depth per parcel: 1 = touches a street, L = L-1 parcels deep.

    Seeds the frontier with parcels within `tol` of the street network
    (`block.streets`, plus any additional `roads`), then peels outward one
    parcel-adjacency hop at a time. A parcel with no path to any street
    (disconnected from the rest of the block) gets one layer past the
    deepest reached layer, so it sorts last honestly instead of silently
    capping like the old weak-dual `k`.

    Returned `pd.Series` is indexed by `parcel_id` (not position), so it
    survives reordering of `block.parcels`.
    """
    parcels = block.parcels
    ids = list(parcels["parcel_id"])
    geoms = list(parcels.geometry)

    adj = adj if adj is not None else parcel_adjacency(geoms, tol)

    street = street_connectivity(block.streets, roads, tol).seed_geom

    layer = [0] * len(geoms)
    # VECTORIZED frontier seed. This was a Python loop calling `g.distance(street)` once per
    # parcel, which is one GEOS round trip per parcel per call -- and this function is called once
    # per CANDIDATE inside the arterial greedy's inner loop, so it dominated: profiled at 278,833
    # scalar `shapely.measurement` calls (2.54 s of 28.8 s) on a 50-parcel block scoring 5,228
    # candidates. `shapely.dwithin` is the same predicate (`distance <= tol`) evaluated over the
    # whole array in one call. Bit-identical, verified against the scalar form over every
    # parcel/prefix pair on real blocks -- the same swap `permeability.edge_conductances` records
    # as worth 4.4x at region scale.
    if street is None or not geoms:
        frontier: deque[int] = deque()
    else:
        near = shapely.dwithin(np.asarray(geoms, dtype=object), street, tol)
        frontier = deque(int(i) for i in np.flatnonzero(near))
    seen = set(frontier)
    for i in frontier:
        layer[i] = 1

    while frontier:
        i = frontier.popleft()
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                layer[j] = layer[i] + 1
                frontier.append(j)

    unreached = [i for i, depth in enumerate(layer) if depth == 0]
    if unreached:
        # `unreached_depth` pins unreached parcels to a caller-supplied, prefix-stable
        # value (the budget curve needs this for cross-prefix comparability); default is
        # the historical "one past the deepest reached layer".
        far = unreached_depth if unreached_depth is not None else max(layer) + 1
        for i in unreached:
            layer[i] = far

    return pd.Series(layer, index=pd.Index(ids, name="parcel_id"), dtype="int64")
