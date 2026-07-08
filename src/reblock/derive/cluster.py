"""merge_cluster: fold a cluster of adjacent blocks into one synthetic super-block so
every per-block derivation (parcel_access_layers, geometric_access_distances,
KComplexityEval, ...) runs on it unchanged. Interior former-boundaries are kept as
real streets (Decision A); the shared frontage lines are exposed via
attrs["interior_boundaries"] for the cross-block metrics.
"""
from __future__ import annotations

from typing import cast

import geopandas as gpd
import pandas as pd
from shapely import make_valid, union_all
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.geometry.base import BaseGeometry

from reblock.contracts import Block, Region


def _blocks_sorted(region: Region) -> list[Block]:
    return sorted(region.blocks, key=lambda b: b.block_id)


def _interior_boundaries(blocks: list[Block]) -> MultiLineString:
    """The shared frontage lines between adjacent blocks (each pair's boundary
    intersection, kept only where it is a positive-length line)."""
    lines: list[BaseGeometry] = []
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            shared = make_valid(blocks[i].boundary).intersection(make_valid(blocks[j].boundary))
            if shared.length > 0:
                lines.append(shared)
    merged = union_all(lines) if lines else MultiLineString([])
    if isinstance(merged, MultiLineString):
        return merged
    # A single shared frontage line unions to a bare LineString, not a MultiLineString;
    # `shared.length > 0` filtering above rules out points, so this is always a line.
    return MultiLineString([cast(LineString, merged)])


def merge_cluster(region: Region) -> Block:
    blocks = _blocks_sorted(region)
    if not blocks:
        raise ValueError(f"{region.region_id}: cluster has no blocks")
    crs = blocks[0].crs

    boundary = make_valid(union_all([b.boundary for b in blocks]))
    if not isinstance(boundary, Polygon):
        raise ValueError(
            f"{region.region_id}: blocks are not contiguous (union is "
            f"{boundary.geom_type}, not a single Polygon): {[b.block_id for b in blocks]}")

    parcels = pd.concat([b.parcels[["geometry"]] for b in blocks], ignore_index=True)
    parcels = gpd.GeoDataFrame(
        {"parcel_id": list(range(len(parcels)))}, geometry=parcels.geometry.to_numpy(), crs=crs)

    streets = gpd.GeoDataFrame(
        geometry=[union_all([g for b in blocks for g in b.streets.geometry])], crs=crs)

    interior = _interior_boundaries(blocks)
    return Block(
        block_id="+".join(b.block_id for b in blocks), crs=crs, boundary=boundary,
        parcels=parcels, streets=streets,
        attrs={"block_ids": [b.block_id for b in blocks], "interior_boundaries": interior})
