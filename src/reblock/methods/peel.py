"""PeelReblocker: route roads by steepest descent down the access-depth peel.

Deterministic, graph-free second method. Each interior parcel links (via its
representative point, guaranteed inside the polygon even when non-convex) to a
descent parent one layer shallower; every root (a street-adjacent parcel that
is some parcel's parent) is tied to the nearest street point. The union is a
connected centerline network reaching the street (full access).
"""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import geopandas as gpd
from shapely import union_all
from shapely.geometry import LineString
from shapely.ops import nearest_points

from reblock.contracts import Block, Proposal
from reblock.derive.access import ParcelAdjacency, one_past_deepest, parcel_access_layers
from reblock.derive_graph import config_identity
from reblock.permeability import with_width


@dataclass
class PeelReblocker:
    tol: float
    # Total width of the roads this method emits; stamped on every one. The metric has no
    # global corridor to fall back on.
    road_width_m: float

    @property
    def identity(self) -> Hashable | None:
        return config_identity(self)

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior  # accepted for Method conformance; steepest-descent is block-only
        ids = list(block.parcels["parcel_id"])
        if len(set(ids)) != len(ids):
            raise ValueError(
                "Block.parcels['parcel_id'] must be unique: deterministic descent-parent "
                "tie-breaks (min parcel_id) require unambiguous ids"
            )
        if block.streets.empty:
            raise ValueError(
                "Block.streets must be non-empty: with no street frontage the peel would "
                "collapse every parcel to a false depth-1 and emit an empty no-op proposal"
            )
        geoms = list(block.parcels.geometry)
        pos = {pid: i for i, pid in enumerate(ids)}
        # One adjacency for both the peel and the descent below, so the two cannot disagree about
        # which parcels are neighbours.
        adjacency = ParcelAdjacency.of(block, self.tol)
        layer = parcel_access_layers(adjacency, None, unreached=one_past_deepest)
        depth = {pid: int(layer.loc[pid]) for pid in ids}
        adj = adjacency.neighbours
        # representative_point() (not centroid) is guaranteed to lie inside the
        # polygon for arbitrary/non-convex shapes, so every served parcel's link
        # actually touches it -- a reflex/L-shaped parcel's centroid can fall outside.
        cent = [g.representative_point() for g in geoms]
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
        return Proposal(block_id=block.block_id, crs=block.crs,
                        roads=with_width(roads, self.road_width_m), edges=None,
                        proposal_id=f"peel_tol{self.tol}", method="peel",
                        params={"unreachable": unreachable},
                        block_identity=block.identity)
