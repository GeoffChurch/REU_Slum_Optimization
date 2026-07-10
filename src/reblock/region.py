"""region_block: union a list of Blocks into one region-level Block, so the existing
single-block Methods can reblock a whole region jointly (roads spanning old block
boundaries). See docs/superpowers/specs/2026-07-10-multi-block-reblocking-design.md.

The one load-bearing decision: `streets` is the outer perimeter of the unioned region
only -- interior shared block-boundary edges vanish in the union. This reframes interior
roads as removable/re-plannable, which is what makes cross-block reblocking "joint"
rather than independent-per-block.
"""
from __future__ import annotations

import hashlib
from typing import cast

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union

from reblock.contracts import Block


def region_block(blocks: list[Block]) -> Block:
    if not blocks:
        raise ValueError("region_block requires a non-empty list of blocks")
    crs = blocks[0].crs
    if any(b.crs != crs for b in blocks):
        raise ValueError("region_block requires all blocks to share one CRS")

    parcels = pd.concat([b.parcels for b in blocks], ignore_index=True)
    parcels["parcel_id"] = range(len(parcels))
    parcels = gpd.GeoDataFrame(parcels, geometry="geometry", crs=crs)

    union = unary_union([b.boundary for b in blocks])
    boundary = union if isinstance(union, Polygon) else cast(Polygon, union.convex_hull)

    streets = gpd.GeoDataFrame(geometry=[union.boundary], crs=crs)

    block_id = "region:" + "+".join(sorted(b.block_id for b in blocks))
    hashes = sorted(f"{b.source_content_hash}:{b.block_id}" for b in blocks)
    source_content_hash = (
        "" if any(b.source_content_hash == "" for b in blocks)
        else hashlib.sha256("|".join(hashes).encode()).hexdigest()
    )

    return Block(block_id=block_id, crs=crs, boundary=boundary, parcels=parcels,
                streets=streets, source_content_hash=source_content_hash)
