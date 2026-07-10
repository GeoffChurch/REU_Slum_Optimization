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
import logging
from dataclasses import dataclass, replace
from typing import Protocol, cast

import geopandas as gpd
import networkx as nx
import pandas as pd
from pyproj import CRS
from shapely import STRtree
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import unary_union

from reblock.contracts import Block, Eval, Method, Proposal, Result
from reblock.derive.access import STREET_TOL

logger = logging.getLogger(__name__)


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


class RegionBuilder(Protocol):
    """Maps user seed groups to expanded region member groups, on cheap block GEOMETRIES (no
    Voronoi) -- so members are chosen before the expensive full-Block build. `groups` is a list
    of seed groups (block_ids); returns the expanded groups (block_ids), each sorted for
    determinism, group order preserved."""

    def build(self, block_geoms: gpd.GeoDataFrame, groups: list[list[str]]) -> list[list[str]]: ...


def _validate_group_ids(block_geoms: gpd.GeoDataFrame, groups: list[list[str]]) -> None:
    """Raise a clear `ValueError` naming any region-group block_id absent from
    `block_geoms`, so a typo'd `block_ids` entry fails with an actionable message instead of
    an opaque `KeyError` from a builder's `by_id[b]` lookup."""
    known = set(block_geoms["block_id"])
    missing = sorted({b for group in groups for b in group if b not in known})
    if missing:
        raise ValueError(f"unknown block_id(s) in region group(s): {missing}")


def _touch_adjacent(geoms: list[BaseGeometry]) -> bool:
    """True iff `geoms` form a single connected component under boundary-touch (within
    STREET_TOL of each other) -- an STRtree `dwithin` query feeds a networkx connectivity
    check. A group of 0 or 1 blocks is trivially connected."""
    if len(geoms) <= 1:
        return True
    tree = STRtree(geoms)
    left, right = tree.query(geoms, predicate="dwithin", distance=STREET_TOL)
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(range(len(geoms)))
    graph.add_edges_from(zip(left.tolist(), right.tolist(), strict=True))
    return bool(nx.is_connected(graph))


@dataclass
class IdentityRegionBuilder:
    """Passthrough RegionBuilder (the default): each seed group IS the region -- unchanged,
    apart from sorting for determinism. Reduces to today's per-block behavior exactly when
    every group is a singleton.

    Warns (naming `convex_hull` as the fix) when a group's blocks are not mutually
    touch-adjacent: a disjoint group's boundary graph splits into disconnected components, so
    every method reblocks each block's interior independently and produces no cross-gap road
    (a buildable road can't span land outside the region) -- correct, but almost always a user
    mistake. A warning, not a hard error, so a deliberate aggregate over scattered blocks still
    runs."""

    def build(self, block_geoms: gpd.GeoDataFrame, groups: list[list[str]]) -> list[list[str]]:
        _validate_group_ids(block_geoms, groups)
        by_id = dict(zip(block_geoms["block_id"], block_geoms.geometry, strict=True))
        result: list[list[str]] = []
        for group in groups:
            if len(group) > 1 and not _touch_adjacent([by_id[b] for b in group]):
                logger.warning(
                    "region group %s is not mutually touch-adjacent -- each block will be "
                    "reblocked independently with no road spanning the gap; pass "
                    "region_builder=convex_hull to fill the gap into one contiguous region",
                    sorted(group),
                )
            result.append(sorted(group))
        return result


@dataclass
class ConvexHullRegionBuilder:
    """Expands each seed group to every candidate block whose geometry intersects the convex
    hull of the group's own block polygons (inclusive) -- fills the gap of a disjoint group
    into one contiguous region where cross-block roads are meaningful. A singleton group's hull
    is its own block's shape, so for a CONVEX block it returns just that block (plus any block
    genuinely overlapping it -- for road-bounded blocks that is only itself); this "singleton
    reduces to identity" reduction is not guaranteed in general, though -- a concave kblock face
    has a hull that bulges past its own outline, and a neighbor poking into that bulge is pulled
    in too. Overlap between different groups' expansions is fine: regions are independent, with
    no partition/merge across groups."""

    def build(self, block_geoms: gpd.GeoDataFrame, groups: list[list[str]]) -> list[list[str]]:
        _validate_group_ids(block_geoms, groups)
        by_id = dict(zip(block_geoms["block_id"], block_geoms.geometry, strict=True))
        result: list[list[str]] = []
        for group in groups:
            hull = unary_union([by_id[b] for b in group]).convex_hull
            matched = cast(gpd.GeoDataFrame, block_geoms[block_geoms.intersects(hull)])
            result.append(sorted(cast(list[str], list(matched["block_id"]))))
        return result
