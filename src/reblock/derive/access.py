"""Access-depth via a BFS parcel peel (kblock's k-complexity definition).

Layer 1 = touches a street; layer L = L-1 parcels from the nearest street.
Runs on parcels directly (STRtree adjacency + BFS) -- no topology graph, so
it is robust to the weak-dual degeneracies of `reblock.eval.kcomplexity`
(a single-file corridor scoring k=1 regardless of length; k silently
capping at 8) and is native per-parcel by `parcel_id`.
"""
from __future__ import annotations

from collections import deque

import pandas as pd
from geopandas import GeoDataFrame
from shapely import union_all

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


def parcel_access_layers(
    block: Block, roads: GeoDataFrame | None, *, tol: float = STREET_TOL
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

    adj = parcel_adjacency(geoms, tol)

    seed_geoms = list(block.streets.geometry)
    if roads is not None and not roads.empty:
        seed_geoms += list(roads.geometry)
    street = union_all(seed_geoms) if seed_geoms else None

    layer = [0] * len(geoms)
    frontier = deque(
        i for i, g in enumerate(geoms) if street is not None and g.distance(street) <= tol
    )
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
        far = max(layer) + 1
        for i in unreached:
            layer[i] = far

    return pd.Series(layer, index=pd.Index(ids, name="parcel_id"), dtype="int64")
