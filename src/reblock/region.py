"""region_block/region_reblock: union a list of Blocks into one region-level Block, so the
existing single-block Methods can reblock a whole region jointly (roads spanning old block
boundaries). See docs/superpowers/specs/2026-07-10-multi-block-reblocking-design.md.

A region is just a single Block whose `streets` are the FULL existing road network -- the union
of every member block's own streets (outer perimeter + the inter-block streets between adjacent
members). Those existing inter-block streets are existing egress, not part of the intervention:
`region_reblock` reblocks this region-Block exactly like any single block -- the method routes
on the existing network and adds only complementary roads, and the evals score those added
roads against it. A parcel already served by an inter-block street therefore reads as shallow
in the 'before', and only the method's new roads count toward the cost-benefit curve.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Protocol, cast

import geopandas as gpd
import networkx as nx
import pandas as pd
from pyproj import CRS
from shapely import STRtree
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from reblock.contracts import Block, Eval, Method, Result
from reblock.derivations import propose
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
    """Union of every block's OWN existing streets -- the full existing road network (outer
    perimeter + inter-block streets). This is `region_block.streets`: the network a method
    routes on, and the egress the evals score the method's added roads against."""
    return unary_union([g for b in blocks for g in b.streets.geometry])


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
    already-served parcels stay served, so the method adds only complementary roads, and it is
    the egress the evals score the method's added roads against."""
    parcels, boundary, crs, member_hash = _shared_parts(blocks)

    streets = gpd.GeoDataFrame(geometry=[_union_streets(blocks)], crs=crs)

    member_pts = [b.building_points for b in blocks if not b.building_points.empty]
    building_points = (
        gpd.GeoDataFrame(pd.concat(member_pts, ignore_index=True), crs=crs) if member_pts
        else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
    )

    # The identity folds in the region model version. derive() caches on the block's identity
    # (source_content_hash, block_id); the region's streets ARE the full existing network the
    # access/curve derivations consume, so a change in what the region means -- here the move to
    # existing-egress, superseding the old perimeter-egress eval-swap that scored a
    # perimeter-streets block under this same identity -- must yield a FRESH key, not a stale hit.
    source_content_hash = (
        "" if member_hash == ""
        else hashlib.sha256(("region-existing-egress|" + member_hash).encode()).hexdigest()
    )
    block_id = "region:" + "+".join(sorted(b.block_id for b in blocks))
    return Block(block_id=block_id, crs=crs, boundary=boundary, parcels=parcels,
                streets=streets, source_content_hash=source_content_hash,
                building_points=building_points)


def region_reblock(blocks: list[Block], method: Method, evals: list[Eval]) -> Result:
    """Reblock a region jointly. The region-Block's `streets` ARE the full existing network
    (outer perimeter + inter-block streets, from `region_block`); the method routes on it and
    adds only complementary roads, and the evals score those added roads against it -- exactly
    like reblocking any single Block. The existing inter-block streets are existing egress, not
    part of the intervention, so the cost-benefit curve reflects only the method's added roads."""
    rb = region_block(blocks)
    # Route through the memoized derivations.propose (keyed on method.identity + the region
    # block's deterministic identity) so a region's proposal is computed once and shared by
    # compare + render, and re-runs hit the L2 disk cache. Uncacheable regions (empty member
    # hash -> identity None) transparently fall back to a direct compute.
    proposal = propose(method, rb)
    metrics = tuple(ev.score(rb, proposal) for ev in evals)
    return Result(block=rb, proposal=proposal, metrics=metrics)


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


def _block_adjacency(geoms: list[BaseGeometry]) -> list[set[int]]:
    """Undirected block-adjacency graph over ALL of `geoms`, as an index -> neighbor-indices
    adjacency list -- an STRtree `dwithin` query within STREET_TOL (the same touch-adjacency
    `_touch_adjacent` checks), excluding self-adjacency."""
    adj: list[set[int]] = [set() for _ in geoms]
    if len(geoms) <= 1:
        return adj
    tree = STRtree(geoms)
    left, right = tree.query(geoms, predicate="dwithin", distance=STREET_TOL)
    for i, j in zip(left.tolist(), right.tolist(), strict=True):
        if i != j:
            adj[i].add(j)
    return adj


def _density(count: float, area: float) -> float:
    """Building density (`count / area`), with a zero-area guard: a degenerate block
    (`area == 0`) has undefined density -- by convention this returns 0.0, the lowest possible
    density, so a degenerate frontier candidate sorts LAST (never preferred over a real block)
    instead of raising `ZeroDivisionError`."""
    return 0.0 if area == 0 else count / area


@dataclass
class DenseClusterRegionBuilder:
    """Grows each seed group into ONE contiguous region by block adjacency, up to a buildings
    budget (a parcel proxy) -- turns "plump a single block (or a screen's flagged block) into a
    right-sized region" into a one-knob operation, no hand-listing neighbors.

    Per group: `cluster` starts as the seed group's own block(s) -- always included, even alone
    over budget (no growth, and never dropped). It then grows greedily, one block at a time:
    among the blocks adjacent to the cluster but not in it (the "frontier"), pick the one with
    the highest building DENSITY (`building_count / area`, `_density`'s zero-area-safe) -- ties
    broken by higher `building_count`, then by `block_id` ascending (determinism) -- add it, and
    repeat until the cluster's total `building_count` reaches `max_buildings` (the last block may
    push it slightly over) or the frontier is exhausted (the seed's whole connected component is
    smaller than the budget).

    Densest-first: density is the closest geometry-available proxy for reblocking need (dense
    blocks are where buried parcels concentrate), so growth reaches toward the neediest
    surrounding fabric rather than sprawling into sparse edges -- true need (access depth) isn't
    available at block-geometry level, but the seed already carries it (via `block_ids`, or a
    screen's worst-first ranking) and growth stays local + dense.

    Graceful without building counts: if `block_geoms` lacks a `building_count` column (a
    non-kblock source), every block counts as 1 -- the budget becomes a block-count budget, and
    density falls back to `1 / area` (smaller blocks first) -- so the builder still produces a
    contiguous, deterministic region. A NaN `building_count` (a degenerate source row) is treated
    as 0, so it can't poison the budget sum or win the density argmax.

    Grows CONTIGUOUSLY from a mutually adjacent seed -- it does not BRIDGE a disjoint one: like
    `IdentityRegionBuilder`, if a seed group's own blocks are not mutually touch-adjacent, this
    warns (naming `convex_hull` as the fix) and still grows each fragment locally, so the output
    stays disjoint too. A warning, not a hard error, so a deliberate aggregate over scattered
    seeds still runs.
    """

    max_buildings: int = 150

    def build(self, block_geoms: gpd.GeoDataFrame, groups: list[list[str]]) -> list[list[str]]:
        _validate_group_ids(block_geoms, groups)
        ids = cast(list[str], list(block_geoms["block_id"]))
        geoms = list(block_geoms.geometry)
        areas = [float(g.area) for g in geoms]
        has_count = "building_count" in block_geoms.columns
        counts = (
            [0.0 if pd.isna(c) else float(c) for c in block_geoms["building_count"]] if has_count
            else [1.0] * len(ids)
        )
        idx_by_id = {b: i for i, b in enumerate(ids)}
        adj = _block_adjacency(geoms)

        result: list[list[str]] = []
        for group in groups:
            if len(group) > 1 and not _touch_adjacent([geoms[idx_by_id[b]] for b in group]):
                logger.warning(
                    "region group %s is not mutually touch-adjacent -- dense_cluster grows each "
                    "fragment locally and won't bridge the gap between them, so the region stays "
                    "disjoint; pass adjacent seeds, or region_builder=convex_hull to fill the gap "
                    "into one contiguous region",
                    sorted(group),
                )
            cluster = {idx_by_id[b] for b in group}
            size = sum(counts[i] for i in cluster)
            while size < self.max_buildings:
                frontier = {j for i in cluster for j in adj[i]} - cluster
                if not frontier:
                    break
                best = min(
                    frontier,
                    key=lambda j: (-_density(counts[j], areas[j]), -counts[j], ids[j]),
                )
                cluster.add(best)
                size += counts[best]
            result.append(sorted(ids[i] for i in cluster))
        return result
