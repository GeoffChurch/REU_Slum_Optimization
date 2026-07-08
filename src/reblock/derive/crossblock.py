"""Probe baselines: the boundary-reconciled block-local peel union (the fair myopia
baseline), and a heuristic automatic spine-merge cross-block reference (replace
boundary-flanking parallel spines with a single through-trunk) — no optimizer, no hand
drawing; just enough to isolate the cross-block-specific gain.
"""
from __future__ import annotations

import geopandas as gpd
from shapely import snap, union_all
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry

from reblock.contracts import Block, Proposal, Region
from reblock.derive.access import STREET_TOL
from reblock.derive.network_metrics import _crosses_boundary, _road_lines
from reblock.methods.peel import PeelReblocker


def reconciled_baseline(region: Region, merged: Block, tol: float = STREET_TOL) -> Proposal:
    blocks = sorted(region.blocks, key=lambda b: b.block_id)
    segments: list[BaseGeometry] = []
    for b in blocks:
        prop = PeelReblocker(tol=tol).propose(b)
        if prop.roads is not None and not prop.roads.empty:
            segments.extend(prop.roads.geometry)
    if not segments:
        return Proposal(block_id=merged.block_id, crs=merged.crs, method="peel_reconciled",
                        proposal_id="peel_reconciled",
                        roads=gpd.GeoDataFrame(geometry=[], crs=merged.crs))
    # snap co-located endpoints together (reconcile stubs meeting across a boundary)
    reference = union_all(segments)
    reconciled = [snap(g, reference, tol) for g in segments]
    roads = gpd.GeoDataFrame(geometry=reconciled, crs=merged.crs)
    return Proposal(block_id=merged.block_id, crs=merged.crs, method="peel_reconciled",
                    proposal_id="peel_reconciled", roads=roads)


def _midline(a: LineString, b: LineString) -> LineString:
    """A trunk from a's start-ish to b's end-ish — a crude through-trunk replacing two
    boundary-parallel spines."""
    pa, pb = a.interpolate(0.5, normalized=True), b.interpolate(0.5, normalized=True)
    return LineString([Point(a.coords[0]), pa, pb, Point(b.coords[-1])])


def spine_merge_reference(
    merged: Block, baseline: Proposal, tol: float = STREET_TOL, band: float = 20.0) -> Proposal:
    interior = merged.attrs.get("interior_boundaries")
    if not isinstance(interior, MultiLineString) or interior.is_empty:
        return baseline
    lines = _road_lines(baseline.roads)
    corridor = interior.buffer(band)
    flanking = [ls for ls in lines
                if not _crosses_boundary(ls, interior, tol) and ls.intersects(corridor)]
    others = [ls for ls in lines if ls not in flanking]
    trunks: list[BaseGeometry] = list(others)
    used = [False] * len(flanking)
    for i in range(len(flanking)):
        if used[i]:
            continue
        for j in range(i + 1, len(flanking)):
            if used[j]:
                continue
            trunk = _midline(flanking[i], flanking[j])
            if _crosses_boundary(trunk, interior, tol):
                trunks.append(trunk)
                used[i] = used[j] = True
                break
        else:
            trunks.append(flanking[i])
            used[i] = True
    roads = gpd.GeoDataFrame(geometry=trunks, crs=merged.crs)
    return Proposal(block_id=merged.block_id, crs=merged.crs, method="spine_merge_ref",
                    proposal_id="spine_merge_ref", roads=roads)
