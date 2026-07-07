"""PeelReblocker: route roads by steepest descent down the access-depth peel.

Deterministic, graph-free second method. Each interior parcel links (via its
centroid) to a descent parent one layer shallower; every root (a street-adjacent
parcel that is some parcel's parent) is tied to the nearest street point. The
union is a connected centerline network reaching the street (full access).
"""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
from shapely import union_all
from shapely.geometry import LineString
from shapely.ops import nearest_points

from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency


@dataclass
class PeelReblocker:
    tol: float = STREET_TOL

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior  # accepted for Method conformance; steepest-descent is block-only
        ids = list(block.parcels["parcel_id"])
        geoms = list(block.parcels.geometry)
        pos = {pid: i for i, pid in enumerate(ids)}
        layer = parcel_access_layers(block, None, tol=self.tol)
        depth = {pid: int(layer.loc[pid]) for pid in ids}
        adj = parcel_adjacency(geoms, self.tol)
        cent = [g.centroid for g in geoms]
        street = union_all(list(block.streets.geometry))

        segments: list[LineString] = []
        roots: set[int] = set()
        unreachable = 0
        for i, pid in enumerate(ids):
            if depth[pid] < 2:
                continue
            # descent parent: adjacent parcel one layer shallower, min parcel_id.
            # Keyed by parcel_id (not row position) so row order can't change it.
            cands = [ids[q] for q in adj[i] if depth[ids[q]] == depth[pid] - 1]
            if not cands:
                unreachable += 1          # disconnected island: no gradient to a street
                continue
            parent = min(cands)
            segments.append(LineString([cent[i], cent[pos[parent]]]))
            if depth[parent] == 1:
                roots.add(parent)
        for pid in sorted(roots):         # sorted -> deterministic street stubs
            on_street = nearest_points(cent[pos[pid]], street)[1]
            segments.append(LineString([cent[pos[pid]], on_street]))

        roads = gpd.GeoDataFrame(geometry=segments, crs=block.crs)
        return Proposal(block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
                        proposal_id="peel", method="peel",
                        params={"unreachable": unreachable})
