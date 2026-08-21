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
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, cast

import geopandas as gpd
import networkx as nx
import pandas as pd
from pyproj import CRS
from shapely import STRtree
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from reblock.contracts import Block, Eval, Method, Result, Source
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


def _projected(block_geoms: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """`block_geoms` in a METRIC frame, reprojected from a geographic one if needed.

    Every builder needs this for two separate reasons that used to be handled separately and
    inconsistently. The growing builders reprojected for their METRIC (`sqrt(n*A)/P` and the shape
    objectives are meaningless where area and length are anisotropic) but deliberately left
    ADJACENCY on the caller's frame -- and adjacency is `_block_adjacency`'s
    `dwithin(STREET_TOL)`, with `STREET_TOL = 0.5`. That is 0.5 metres in UTM and about 55 km in
    lon/lat, so a geographic frame made every block in a metro adjacent to every other and growth
    silently assembled regions from blocks kilometres apart.

    A no-op on every shipped path -- `KblockSource.block_geometries()` already returns UTM -- so
    this changes no production output. It exists because `scripts/pair_matrix.py:304` reads its
    frame straight out of the parquet (lon/lat) and reaches `build()` through a
    `cast(gpd.GeoDataFrame, ...)`, which is a type-checker assertion and not a runtime conversion.
    """
    if block_geoms.crs is not None and block_geoms.crs.is_geographic:
        return block_geoms.to_crs(block_geoms.estimate_utm_crs())
    return block_geoms


def _union_streets(blocks: list[Block]) -> BaseGeometry:
    """Union of every block's OWN existing streets -- the full existing road network (outer
    perimeter + inter-block streets). This is `region_block.streets`: the network a method
    routes on, and the egress the evals score the method's added roads against."""
    return unary_union([g for b in blocks for g in b.streets.geometry])


def _shared_parts(blocks: list[Block]) -> tuple[gpd.GeoDataFrame, Polygon | MultiPolygon, CRS, str]:
    """Common region pieces every builder needs: parcels (unioned + re-parcel_id'ed), boundary
    (the true union of member boundaries -- a MultiPolygon when members are separated by street
    gaps, NOT a convex hull, which would enclose the empty gaps and inflate the area), crs
    (asserted shared), and source_content_hash (a deterministic hash of the sorted constituent
    identities, or "" if any is uncacheable).

    MEMBERS ARE SORTED BY block_id FIRST, and that is load-bearing rather than tidiness: the
    `parcel_id` assignment below numbers parcels by member order, while `block_id` and
    `source_content_hash` are both built from `sorted(...)` and would NOT change. Handed members in
    a different order, this would renumber every parcel under an identity the derivation cache
    treats as unchanged.

    Load-bearing, not inert, for the two growing builders. `pipeline.build_regions` returns each
    region's Blocks in whichever order `RegionBuilder.build` produced -- accretion order for
    `DenseClusterRegionBuilder` / `ShapeStandardizingRegionBuilder` -- and nothing re-sorts them
    before `pipeline.run` passes that straight to `region_reblock(rblocks, ...)`
    (`pipeline.py:216`), which calls `region_block(blocks)` (`region.py:154`) -> here: one hop
    from `build_regions` to this sort. For `region_builder=dense_cluster` /
    `shape_standardizing` (`conf/region_builder/*.yaml`, selectable today, not hypothetical) that
    member order is genuinely unsorted, and this line is what keeps `parcel_id` numbering
    consistent with the unchanged `block_id` / `source_content_hash`. It is inert only under the
    default `identity` builder, whose output is already sorted. Do not delete it as dead
    insurance.
    """
    crs = _check(blocks, "region_block")
    blocks = sorted(blocks, key=lambda b: b.block_id)

    parcels = pd.concat([b.parcels for b in blocks], ignore_index=True)
    parcels["parcel_id"] = range(len(parcels))
    parcels = gpd.GeoDataFrame(parcels, geometry="geometry", crs=crs)

    boundary = cast("Polygon | MultiPolygon", unary_union([b.boundary for b in blocks]))

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
    of seed groups (block_ids); returns the expanded groups (block_ids), each in BUILD ORDER,
    group order preserved.

    Build order means accretion order where there is one: `DenseClusterRegionBuilder` and
    `ShapeStandardizingRegionBuilder` return the seed group (sorted) followed by each block in the
    order it was added, which is what the site's RegionGrow widget replays and what pins its
    browser-side greedy to this code. `IdentityRegionBuilder` and `ConvexHullRegionBuilder` have no
    accretion, so sorted IS their build order. Determinism is unchanged either way -- accretion
    order is fixed by the tie-break rule.

    Consumers that need a SET must sort: `region._shared_parts` does, because it numbers parcels by
    member order and a renumbering under an unchanged `source_content_hash` would corrupt cached
    derivations.
    """

    def build(self, block_geoms: gpd.GeoDataFrame, groups: list[list[str]],
              depth_fn: Callable[[str], float] | None = None) -> list[list[str]]: ...


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

    def build(self, block_geoms: gpd.GeoDataFrame, groups: list[list[str]],
              depth_fn: Callable[[str], float] | None = None) -> list[list[str]]:
        del depth_fn   # these builders don't rank by depth
        _validate_group_ids(block_geoms, groups)
        block_geoms = _projected(block_geoms)
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

    def build(self, block_geoms: gpd.GeoDataFrame, groups: list[list[str]],
              depth_fn: Callable[[str], float] | None = None) -> list[list[str]]:
        del depth_fn   # these builders don't rank by depth
        _validate_group_ids(block_geoms, groups)
        block_geoms = _projected(block_geoms)
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


def _depth_proxy(count: float, area: float, perim: float) -> float:
    """Cheap depth proxy `sqrt(count * area) / perimeter` -- a block-geometry estimate of parcel
    access depth in rings (inradius / parcel-width), the region builder's growth metric. It ranks
    true access depth ~5x better than building density (`count / area`), which is nearly
    uncorrelated with depth -- see `metric.Depth` (the canonical closed form) and
    docs/superpowers/notes/2026-07-14-depth-proxy-screen-gate.md. A degenerate block (`area == 0`
    or `perim == 0`) returns 0.0, the lowest score, so it sorts LAST as a frontier candidate
    instead of raising. `area`/`perim` must be in metres (the caller reprojects)."""
    if area <= 0 or perim <= 0:
        return 0.0
    return math.sqrt(count * area) / perim


def block_depths(source: Source, block_ids: list[str]) -> dict[str, float]:
    """True max BFS access-depth (parcel rings from a street) for each of `block_ids`, built in ONE
    `KblockSource(block_ids=...).region()` call, peeled with the memoized `access_before` ->
    `{block_id: max_depth}`. BATCHING is load-bearing: `KblockSource.region()` reads and
    spatial-joins the WHOLE ~49 MB buildings parquet on every call regardless of `block_ids`, so a
    per-block accessor pays that ~2.7 s read PER block (profiled). One batched call amortizes it
    across all `block_ids`, as the screen's fine pass peels ~900 blocks per read. Blocks the screen
    already peeled are cache hits (`_voronoi_impl`/`access_before` are memoized). A block that can't
    be built/peeled (below `min_buildings`, bad geometry) is simply ABSENT from the returned
    dict -- callers default a missing id to 0.0, so it never wins a `deepest` argmax. Returns `{}`
    for a non-peel-capable source (no `blocks_path`) or an empty `block_ids`."""
    from reblock.data.kblock import KblockSource
    from reblock.derivations import access_before
    if not isinstance(source, KblockSource) or not block_ids:
        return {}
    sub = KblockSource(source.blocks_path, source.buildings_path, "depth",
                       min_buildings=getattr(source, "min_buildings", 10),
                       block_ids=list(block_ids))
    return {str(b.block_id): float(access_before(b).max()) for b in sub.region().blocks}


@dataclass
class DenseClusterRegionBuilder:
    """Grows each seed group into ONE contiguous region by block adjacency, up to a buildings
    budget (a parcel proxy) -- turns "plump a single block (or a screen's flagged block) into a
    right-sized region" into a one-knob operation, no hand-listing neighbors.

    Per group: `cluster` starts as the seed group's own block(s) -- always included, even alone
    over budget (no growth, and never dropped). It then grows greedily, one block at a time:
    among the blocks adjacent to the cluster but not in it (the "frontier"), pick the one with the
    highest DEPTH PROXY (`sqrt(building_count * area) / perimeter`, `_depth_proxy`'s zero-safe) --
    ties broken by higher `building_count`, then by `block_id` ascending (determinism) -- add it,
    and repeat until the cluster's total `building_count` reaches `max_buildings` (the last block
    may push it slightly over) or the frontier is exhausted (the seed's whole connected component
    is smaller than the budget).

    Deepest-first: the depth proxy sqrt(n*A)/P is a cheap block-geometry estimate of parcel access
    depth (frontage-starvation), so growth reaches toward the deepest surrounding fabric -- the
    informal core -- rather than wandering into shallow formal housing the way building density
    does (density is nearly uncorrelated with true depth, even within one neighborhood; the proxy
    ranks it ~5x better). True access depth isn't available at block-geometry level, and the seed
    already carries it (via `block_ids`, or a screen's worst-first ranking); the proxy keeps growth
    local AND deep. Area/perimeter are measured in metres (reprojected from a geographic CRS).

    Graceful without building counts: if `block_geoms` lacks a `building_count` column (a
    non-kblock source), every block counts as 1 -- the budget becomes a block-count budget, and
    the proxy falls back to `sqrt(area) / perimeter` (bigger, compacter blocks first) -- so the
    builder still produces a contiguous, deterministic region. A NaN `building_count` (a degenerate
    source row) is treated as 0, so it can't poison the budget sum or win the proxy argmax.

    Grows CONTIGUOUSLY from a mutually adjacent seed -- it does not BRIDGE a disjoint one: like
    `IdentityRegionBuilder`, if a seed group's own blocks are not mutually touch-adjacent, this
    warns (naming `convex_hull` as the fix) and still grows each fragment locally, so the output
    stays disjoint too. A warning, not a hard error, so a deliberate aggregate over scattered
    seeds still runs.
    """

    max_buildings: int = 150

    def build(self, block_geoms: gpd.GeoDataFrame, groups: list[list[str]],
              depth_fn: Callable[[str], float] | None = None) -> list[list[str]]:
        _validate_group_ids(block_geoms, groups)
        metric = _projected(block_geoms)
        ids = cast(list[str], list(block_geoms["block_id"]))
        # ONE frame drives everything: adjacency, the touch-adjacency warning AND the metric. The
        # previous split -- metric reprojected, adjacency on the caller's frame -- is the §1.5 bug.
        geoms = list(metric.geometry)
        areas = [float(g.area) for g in metric.geometry]
        perims = [float(g.length) for g in metric.geometry]
        has_count = "building_count" in block_geoms.columns
        counts = (
            [0.0 if pd.isna(c) else float(c) for c in block_geoms["building_count"]] if has_count
            else [1.0] * len(ids)
        )
        idx_by_id = {b: i for i, b in enumerate(ids)}
        adj = _block_adjacency(geoms)

        depth_cache: dict[str, float] = {}

        def _score(j: int) -> float:
            if depth_fn is None:
                return _depth_proxy(counts[j], areas[j], perims[j])
            bid = ids[j]
            if bid not in depth_cache:
                depth_cache[bid] = depth_fn(bid)
            return depth_cache[bid]

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
            order = sorted(cluster, key=lambda i: ids[i])   # the seed group, deterministically
            size = sum(counts[i] for i in cluster)
            while size < self.max_buildings:
                frontier = {j for i in cluster for j in adj[i]} - cluster
                if not frontier:
                    break
                best = min(
                    frontier,
                    key=lambda j: (-_score(j), -counts[j], ids[j]),
                )
                cluster.add(best)
                order.append(best)
                size += counts[best]
            result.append([ids[i] for i in order])
        return result


class ShapeObjective(Protocol):
    """Scores the OUTLINE of a candidate region union. Higher is better; scale-free.

    Scale-free matters: the accretion compares unions of different sizes at every step, so an
    objective that grows with area would just pick the biggest block every time.
    """

    # read-only: the implementations are frozen dataclasses, and a plain `name: str` in a Protocol
    # demands a SETTABLE attribute, which a frozen field is not
    @property
    def name(self) -> str: ...

    def score(self, union: BaseGeometry) -> float: ...


@dataclass(frozen=True)
class Isoperimetric:
    """4*pi*A / P^2 -- 1 for a circle, lower for anything else. The obvious first guess.

    The spec that asked for this builder warns against assuming it is the right target: the
    requirement is that outline variance be small enough not to dominate GW distance, NOT that
    regions be maximally circular. It is here as a baseline to beat, not as the default answer.
    """

    name: str = "isoperimetric"

    def score(self, union: BaseGeometry) -> float:
        p = float(union.length)
        return 0.0 if p <= 0 else float(4.0 * math.pi * union.area / (p * p))


@dataclass(frozen=True)
class Rectangularity:
    """A / area(minimum rotated rectangle) -- 1 for any rectangle, at any orientation.

    Unlike `Isoperimetric` this does not punish elongation, and it keeps whatever dominant
    orientation the fabric has rather than discarding it toward a circle.
    """

    name: str = "rectangularity"

    def score(self, union: BaseGeometry) -> float:
        mrr = union.minimum_rotated_rectangle
        a = float(mrr.area)
        return 0.0 if a <= 0 else float(union.area / a)


@dataclass(frozen=True)
class Squareness:
    """Rectangularity times the minimum-rotated-rectangle's aspect ratio -- 1 only for a square.

    The spec's argument for preferring this over a circle: squares tile, they are FFT-native if
    retrieval goes that way, and a square still admits the fabric's own orientation because the
    rectangle is rotated, not axis-aligned.
    """

    name: str = "squareness"

    def score(self, union: BaseGeometry) -> float:
        mrr = union.minimum_rotated_rectangle
        a = float(mrr.area)
        if a <= 0:
            return 0.0
        xs, ys = zip(*list(cast(Polygon, mrr).exterior.coords)[:4], strict=True)
        sides = [math.dist((xs[i], ys[i]), (xs[(i + 1) % 4], ys[(i + 1) % 4])) for i in range(4)]
        short, long_ = min(sides), max(sides)
        aspect = short / long_ if long_ > 0 else 0.0
        return float(union.area / a) * aspect


@dataclass
class ShapeStandardizingRegionBuilder:
    """Accretes blocks into a region whose OUTLINE is standardized, scoring the union as it grows.

    The distinction from `DenseClusterRegionBuilder` is the whole point of this builder, and it is
    one line of the loop: dense-cluster ranks each frontier block by `sqrt(n*A)/P` computed on that
    block ALONE and never looks at the shape being assembled, so its regions came out as 150-900
    parcel tendrils whose outline is a growth artifact. That uncontrolled outline is a confound the
    Phase 3 donor-material test cannot tolerate -- street-form donors force accretion (a kblock
    block is a street-bounded face, so a single block has no internal streets to copy), so material
    can only be compared against outline held fixed.

    Here the frontier block chosen is the one maximizing `objective.score(union u candidate)`.

    ## The objective is deliberately pluggable, and deliberately not defaulted to compactness

    The originally-specified builder was never built and a substitute shipped in its place. The spec
    is explicit that the objective is open -- isoperimetric compactness is "only the obvious first
    guess", squareness and rectangularity are live alternatives, and the choice should be made
    empirically against the outline's share of inter-region GW distance variance rather than by
    assuming the familiar quotient is right. So this takes a `ShapeObjective`.

    `Squareness` is the default, and NOT by assumption -- `Isoperimetric` is disqualified on a
    necessary condition before the GW criterion is even reached. Polyomino perimeters tie
    constantly (a 1x3 strip and an L-tromino both have area 3 and perimeter 8, so identical
    quotient), so on grid-like fabric the greedy cannot discriminate, falls back to the `block_id`
    tie-break, and walks into shapes from which the compact option is unreachable. Growing a
    4-block region from the centre of a uniform 5x5 grid:

        isoperimetric  -> staircase, quotient 0.503   (its OWN metric, and it misses 0.785)
        rectangularity -> 1x4 strip, quotient 0.503   (blind to elongation by construction)
        squareness     -> the 2x2,   quotient 0.785

    An objective that ties everywhere standardizes nothing -- it reproduces exactly the
    growth-artifact outline this builder exists to remove. The GW-variance criterion the spec names
    is still the one that should settle squareness vs alternatives on real fabric; this only rules
    out the familiar quotient.

    Growth stops on the same `max_buildings` budget as dense-cluster, and shares its conventions:
    the seed group is always included (even alone over budget), ties break by higher
    `building_count` then `block_id` ascending, and a non-adjacent seed group grows each fragment
    locally with a warning rather than bridging.
    """

    objective: ShapeObjective = field(default_factory=lambda: Squareness())
    max_buildings: int = 150

    def build(self, block_geoms: gpd.GeoDataFrame, groups: list[list[str]],
              depth_fn: Callable[[str], float] | None = None) -> list[list[str]]:
        del depth_fn                      # shape is scored on geometry; access depth plays no part
        _validate_group_ids(block_geoms, groups)
        metric = _projected(block_geoms)
        ids = cast(list[str], list(block_geoms["block_id"]))
        # ONE frame drives everything: adjacency, the touch-adjacency warning AND the shape
        # objective. The previous split -- shape reprojected, adjacency on the caller's frame --
        # is the §1.5 bug.
        geoms = list(metric.geometry)
        shape_geoms = geoms
        counts = (
            [0.0 if pd.isna(c) else float(c) for c in block_geoms["building_count"]]
            if "building_count" in block_geoms.columns else [1.0] * len(ids)
        )
        idx_by_id = {b: i for i, b in enumerate(ids)}
        adj = _block_adjacency(geoms)

        result: list[list[str]] = []
        for group in groups:
            if len(group) > 1 and not _touch_adjacent([geoms[idx_by_id[b]] for b in group]):
                logger.warning(
                    "region group %s is not mutually touch-adjacent -- shape_standardizing grows "
                    "each fragment locally and won't bridge the gap, so the region stays disjoint "
                    "and its outline is not standardized; pass adjacent seeds, or "
                    "region_builder=convex_hull to fill the gap into one contiguous region",
                    sorted(group),
                )
            cluster = {idx_by_id[b] for b in group}
            order = sorted(cluster, key=lambda i: ids[i])   # the seed group, deterministically
            size = sum(counts[i] for i in cluster)
            union = unary_union([shape_geoms[i] for i in cluster])
            while size < self.max_buildings:
                frontier = {j for i in cluster for j in adj[i]} - cluster
                if not frontier:
                    break
                # One binary union per candidate against the running union -- NOT a fresh
                # unary_union of the whole cluster each time, which would make growth quadratic.
                scored = {j: self.objective.score(union.union(shape_geoms[j])) for j in frontier}
                best = min(frontier, key=lambda j: (-scored[j], -counts[j], ids[j]))
                cluster.add(best)
                order.append(best)
                size += counts[best]
                union = union.union(shape_geoms[best])
            result.append([ids[i] for i in order])
        return result
