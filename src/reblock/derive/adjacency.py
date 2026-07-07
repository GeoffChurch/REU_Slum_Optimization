"""Parcel adjacency: shared-boundary neighbour sets, robust to invalid geometry.

Two parcels are adjacent iff their boundaries share a positive-length run within
`tol`. Snapping bridges sub-`tol` digitization gaps; on real cadastral data
`snap()` can raise a GEOS side-location conflict for a messy touching pair, so
fall back to a direct intersection on validated operands. Positional index in,
neighbour sets out.
"""
from __future__ import annotations

from shapely import STRtree, make_valid, snap
from shapely.errors import GEOSException
from shapely.geometry.base import BaseGeometry


def _shared_len(gi: BaseGeometry, gj: BaseGeometry, tol: float) -> float:
    """Length of the boundary run parcels `gi`, `gj` share (0 if only a point)."""
    try:
        return float(snap(gi, gj, tol).intersection(gj).length)
    except GEOSException:
        return float(make_valid(gi).intersection(make_valid(gj)).length)


def parcel_adjacency(geoms: list[BaseGeometry], tol: float) -> list[set[int]]:
    """Positional neighbour sets: adjacent iff a positive-length shared boundary
    within `tol`. Candidate pairs come from an STRtree `dwithin` query; a snap
    bridges sub-`tol` gaps, and the shared run must be a line (not just a point),
    which excludes diagonally-touching parcels."""
    tree = STRtree(geoms)
    left, right = tree.query(geoms, predicate="dwithin", distance=tol)
    adj: list[set[int]] = [set() for _ in geoms]
    for i, j in zip(left.tolist(), right.tolist(), strict=True):
        if i >= j:
            continue  # each unordered pair once; also drops self-pairs (i == j)
        if _shared_len(geoms[i], geoms[j], tol) > 0:
            adj[i].add(j)
            adj[j].add(i)
    return adj
