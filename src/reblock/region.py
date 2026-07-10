"""region_block/region_reblock: union a list of Blocks into one region-level Block, so the
existing single-block Methods can reblock a whole region jointly (roads spanning old block
boundaries). See docs/superpowers/specs/2026-07-10-multi-block-reblocking-design.md.

The seed model: existing inter-block roads are a pre-added "seed" road network that the
methods *extend* (routed on, not demolished) -- treated the same as roads we add, counted as
first-added road; true egress is the region's outer perimeter. Concretely:

- `region_block(blocks).streets` = the FULL existing road network (union of every block's own
  streets, perimeter + interior), so a method routes on it and adds only complementary roads.
- `region_perimeter(blocks)` = the existing streets that sit on the region's outer boundary --
  the eval egress.
- `region_seed_roads(blocks)` = the existing streets that do NOT -- the counted seed, emitted
  first (highest-drain) alongside whatever the method adds.
- Both are derived from the STREETS (not just the region's outline geometry), so they are
  consistent partitions of `region_block.streets`: `region_perimeter` union `region_seed_roads`
  reconstructs it (a block with only partial frontage isn't credited with egress along a
  stretch of its outline that has no actual street).
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
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import unary_union

from reblock.contracts import Block, Eval, Method, Proposal, Result
from reblock.derive.access import STREET_TOL


def _check(blocks: list[Block], name: str) -> CRS:
    """Shared empty-list / CRS-mismatch guard for every region.py builder."""
    if not blocks:
        raise ValueError(f"{name} requires a non-empty list of blocks")
    crs = blocks[0].crs
    if any(b.crs != crs for b in blocks):
        raise ValueError(f"{name} requires all blocks to share one CRS")
    return crs


def _union_streets(blocks: list[Block]) -> BaseGeometry:
    """Union of every block's OWN existing streets -- the full routing street network
    (perimeter + inter-block). `region_perimeter` and `region_seed_roads` are both partitions
    of this: egress = the part on the region's outer boundary, seed = the rest."""
    return unary_union([g for b in blocks for g in b.streets.geometry])


def _explode_lines(geom: BaseGeometry, crs: CRS) -> gpd.GeoDataFrame:
    """Explode a (possibly multi-part) geometry to one row per LineString, dropping
    degenerate (zero-length / non-line) parts; an empty GeoDataFrame if none survive."""
    parts = list(geom.geoms) if isinstance(geom, BaseMultipartGeometry) else [geom]
    rows = [g for g in parts if "LineString" in g.geom_type and g.length > 0]
    if not rows:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
    return gpd.GeoDataFrame({"geometry": rows}, geometry="geometry", crs=crs)


def _shared_parts(blocks: list[Block]) -> tuple[gpd.GeoDataFrame, Polygon, CRS, str]:
    """Common region pieces every builder needs: parcels (unioned + re-parcel_id'ed), boundary
    (union, or its convex hull if the union is a MultiPolygon), crs (asserted shared), and
    source_content_hash (a deterministic hash of the sorted constituent identities, or "" if
    any is uncacheable)."""
    crs = _check(blocks, "region_block")

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

    streets = gpd.GeoDataFrame(geometry=[_union_streets(blocks)], crs=crs)

    block_id = "region:" + "+".join(sorted(b.block_id for b in blocks))
    return Block(block_id=block_id, crs=crs, boundary=boundary, parcels=parcels,
                streets=streets, source_content_hash=source_content_hash)


def region_perimeter(blocks: list[Block]) -> gpd.GeoDataFrame:
    """The eval egress: the routing STREETS that sit on the region's outer geometric boundary
    (not the boundary itself -- a block with only partial street frontage must not be credited
    with egress along a stretch of its own outline that has no actual street). Streets-derived,
    so `region_perimeter` and `region_seed_roads` are consistent partitions of `_union_streets`
    (== `region_block.streets`): together they union back to it."""
    crs = _check(blocks, "region_perimeter")
    outer = unary_union([b.boundary for b in blocks]).boundary
    perim = _union_streets(blocks).intersection(outer.buffer(STREET_TOL))
    return _explode_lines(perim, crs)


def region_seed_roads(blocks: list[Block]) -> gpd.GeoDataFrame:
    """The counted seed: the routing streets NOT on the region's outer boundary (all existing
    streets minus `region_perimeter`) -- emitted first (highest-drain) in `region_reblock`.
    Exploded to one LineString per row; an empty GeoDataFrame if there is no interior seed."""
    crs = _check(blocks, "region_seed_roads")
    outer = unary_union([b.boundary for b in blocks]).boundary
    seed = _union_streets(blocks).difference(outer.buffer(STREET_TOL))
    return _explode_lines(seed, crs)


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
    full_proposal: Proposal = replace(proposal, roads=full, block_id=eval_block.block_id,
                                      block_identity=eval_block.identity)
    metrics = tuple(ev.score(eval_block, full_proposal) for ev in evals)
    return Result(block=eval_block, proposal=full_proposal, metrics=metrics)
