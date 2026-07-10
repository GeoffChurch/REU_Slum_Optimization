"""region_block/region_reblock: union a list of Blocks into one region-level Block, so the
existing single-block Methods can reblock a whole region jointly (roads spanning old block
boundaries). See docs/superpowers/specs/2026-07-10-multi-block-reblocking-design.md.

The seed model: existing inter-block roads are a pre-added "seed" road network that the
methods *extend* (routed on, not demolished) -- treated the same as roads we add, counted as
first-added road; true egress is the region's outer perimeter. Concretely:

- `region_block(blocks).streets` = the FULL existing road network (union of every block's own
  streets, perimeter + interior), so a method routes on it and adds only complementary roads.
- `region_perimeter(blocks)` = the outer perimeter only -- the eval egress.
- `region_seed_roads(blocks)` = the interior existing roads (full existing minus perimeter) --
  the counted seed, emitted first (highest-drain) alongside whatever the method adds.
- `region_reblock(blocks, method, evals)` ties the three together: the method sees the full
  existing network (perimeter + seed) as already-street; the eval scores seed + added against
  the perimeter-only egress, so parcels must reach the city THROUGH the network.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import cast

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import Polygon
from shapely.geometry.base import BaseMultipartGeometry
from shapely.ops import unary_union

from reblock.budget import road_drainage
from reblock.contracts import Block, Eval, Method, Proposal, Result
from reblock.derive.access import STREET_TOL


def _shared_parts(blocks: list[Block]) -> tuple[gpd.GeoDataFrame, Polygon, CRS, str]:
    """Common region pieces every builder needs: parcels (unioned + re-parcel_id'ed), boundary
    (union, or its convex hull if the union is a MultiPolygon), crs (asserted shared), and
    source_content_hash (a deterministic hash of the sorted constituent identities, or "" if
    any is uncacheable)."""
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

    hashes = sorted(f"{b.source_content_hash}:{b.block_id}" for b in blocks)
    source_content_hash = (
        "" if any(b.source_content_hash == "" for b in blocks)
        else hashlib.sha256("|".join(hashes).encode()).hexdigest()
    )
    return parcels, boundary, crs, source_content_hash


def region_block(blocks: list[Block]) -> Block:
    """The block a method reblocks. `streets` = the union of every block's existing streets
    (perimeter + inter-block = the full existing road network): routing on this means
    seed-adjacent parcels are already served, so the method adds only complementary roads."""
    parcels, boundary, crs, source_content_hash = _shared_parts(blocks)

    all_existing = unary_union([g for b in blocks for g in b.streets.geometry])
    streets = gpd.GeoDataFrame(geometry=[all_existing], crs=crs)

    block_id = "region:" + "+".join(sorted(b.block_id for b in blocks))
    return Block(block_id=block_id, crs=crs, boundary=boundary, parcels=parcels,
                streets=streets, source_content_hash=source_content_hash)


def region_perimeter(blocks: list[Block]) -> gpd.GeoDataFrame:
    """The outer perimeter lines of the unioned region -- the eval egress (this is what
    `region_block.streets` used to be, pre-seed-model)."""
    crs = blocks[0].crs
    union = unary_union([b.boundary for b in blocks])
    return gpd.GeoDataFrame(geometry=[union.boundary], crs=crs)


def region_seed_roads(blocks: list[Block]) -> gpd.GeoDataFrame:
    """The interior existing roads = all existing streets MINUS the perimeter -- the counted
    seed, emitted first (highest-drain) in `region_reblock`. Exploded to one LineString per
    row; an empty GeoDataFrame if there is no interior seed."""
    crs = blocks[0].crs
    all_existing = unary_union([g for b in blocks for g in b.streets.geometry])
    perim = unary_union(list(region_perimeter(blocks).geometry))
    seed = all_existing.difference(perim.buffer(STREET_TOL))

    parts = list(seed.geoms) if isinstance(seed, BaseMultipartGeometry) else [seed]
    rows = [g for g in parts if "LineString" in g.geom_type and g.length > 0]
    if not rows:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
    return gpd.GeoDataFrame({"geometry": rows}, geometry="geometry", crs=crs)


def region_reblock(blocks: list[Block], method: Method, evals: list[Eval]) -> Result:
    """Reblock a region jointly: the method routes on the full existing network (seed +
    perimeter), the eval scores the seed's + the method's added roads against the perimeter-
    only egress. See the module docstring / design spec for why this is correct: the method
    sees the seed as existing street (routes/extends it); `cost_benefit_curve` (used by the
    evals) recomputes drainage on the full road set, so the high-drainage seed trunk roads are
    sliced first automatically ("counted, added first"); and scoring against `eval_block`
    (egress = perimeter) makes parcels reach the city THROUGH seed + added."""
    rb = region_block(blocks)                                  # streets = full existing network
    proposal: Proposal = method.propose(rb)
    seed = region_seed_roads(blocks)
    added = proposal.roads
    parts = [g for g in (seed, added) if g is not None and len(g) > 0]
    full_geom = (
        pd.concat([g[["geometry"]] for g in parts], ignore_index=True) if parts
        else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=rb.crs)
    )
    full = gpd.GeoDataFrame(full_geom, geometry="geometry", crs=rb.crs)
    eval_block = replace(rb, streets=region_perimeter(blocks))
    full["drain"] = road_drainage(eval_block, full) if len(full) else []
    full_proposal: Proposal = replace(proposal, roads=full, block_id=eval_block.block_id,
                                      block_identity=eval_block.identity)
    metrics = tuple(ev.score(eval_block, full_proposal) for ev in evals)
    return Result(block=eval_block, proposal=full_proposal, metrics=metrics)
