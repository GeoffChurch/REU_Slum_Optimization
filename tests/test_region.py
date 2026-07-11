import hashlib
import logging
from dataclasses import replace
from pathlib import Path
from typing import cast

import geopandas as gpd
import networkx as nx
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from reblock.budget import auc, efficiency_directness_curves
from reblock.contracts import Block, Result
from reblock.data.kblock import KblockSource
from reblock.eval.kcomplexity import KComplexityEval
from reblock.methods.arterial import GreedyArterialReblocker
from reblock.methods.dijkstra import DijkstraReblocker
from reblock.region import (
    ConvexHullRegionBuilder,
    IdentityRegionBuilder,
    region_block,
    region_reblock,
)

UTM = CRS.from_epsg(32643)
DJI_BLOCKS = Path(__file__).resolve().parent / "data" / "kblock" / "blocks_dji_sample.parquet"
DJI_BLD = Path(__file__).resolve().parent / "data" / "kblock" / "buildings_dji_sample.parquet"


def _block_geoms(*specs: tuple[str, float, float]) -> gpd.GeoDataFrame:
    """A block_id + geometry GeoDataFrame -- one unit-square "block" per (block_id, x, y)
    offset -- the cheap geometry input a RegionBuilder operates on."""
    ids = [block_id for block_id, _, _ in specs]
    polys = [Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]) for _, x, y in specs]
    return gpd.GeoDataFrame({"block_id": ids}, geometry=polys, crs=UTM)

_SIDES = {
    "bottom": lambda x0, y0, w, h: LineString([(x0, y0), (x0 + w, y0)]),
    "top": lambda x0, y0, w, h: LineString([(x0, y0 + h), (x0 + w, y0 + h)]),
    "left": lambda x0, y0, w, h: LineString([(x0, y0), (x0, y0 + h)]),
    "right": lambda x0, y0, w, h: LineString([(x0 + w, y0), (x0 + w, y0 + h)]),
}


def _grid_block(x0: int, y0: int, w: int, h: int, streets_side: str = "all",
                block_id: str = "grid", points: gpd.GeoDataFrame | None = None) -> Block:
    """A w x h grid of unit parcels at (x0, y0). `streets_side="all"` (the default) gives the
    full block-perimeter frontage (as if every existing road around the block is already
    street); a side name ("bottom"/"top"/"left"/"right") gives frontage on only that outer
    edge, for building a deep block/region. `points`, if given, becomes the block's
    `building_points` (default: the empty default -- most fixtures don't need real sites)."""
    polys, ids = [], []
    for i in range(w):
        for j in range(h):
            polys.append(Polygon([
                (x0 + i, y0 + j), (x0 + i + 1, y0 + j),
                (x0 + i + 1, y0 + j + 1), (x0 + i, y0 + j + 1)]))
            ids.append(i * h + j)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, unary_union(polys))
    line = boundary.boundary if streets_side == "all" else _SIDES[streets_side](x0, y0, w, h)
    streets = gpd.GeoDataFrame(geometry=[line], crs=UTM)
    if points is None:
        return Block(block_id=block_id, crs=UTM, boundary=boundary, parcels=parcels,
                    streets=streets)
    return Block(block_id=block_id, crs=UTM, boundary=boundary, parcels=parcels, streets=streets,
                building_points=points)


def _spans_both_sides(roads: gpd.GeoDataFrame, x_split: float, margin: float) -> bool:
    """True if `roads`, viewed as a touch-graph on rounded endpoints, has a connected
    component reaching both x <= x_split - margin and x >= x_split + margin -- i.e. some road
    (or chain of touching roads) genuinely spans across the original block boundary at
    x=x_split, proving joint (not per-block) reblocking."""
    g: nx.Graph = nx.Graph()
    for geom in roads.geometry:
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for part in parts:
            cs = [(round(x, 2), round(y, 2)) for x, y in part.coords]
            for p, q in zip(cs, cs[1:], strict=False):
                if p != q:
                    g.add_edge(p, q)
    return any(
        min(n[0] for n in comp) <= x_split - margin and max(n[0] for n in comp) >= x_split + margin
        for comp in nx.connected_components(g)
    )


def test_region_block_streets_are_the_full_existing_network() -> None:
    # region_block.streets = the union of every block's OWN existing streets (perimeter +
    # inter-block), so the interior shared edge -- present in both a's and b's own streets --
    # is INCLUDED. This full existing network is what a method routes on and what the evals
    # score the method's added roads against.
    a = _grid_block(0, 0, 3, 3, block_id="a")
    b = _grid_block(3, 0, 3, 3, block_id="b")
    rb = region_block([a, b])

    assert len(rb.parcels) == 18
    assert sorted(rb.parcels["parcel_id"]) == list(range(18))
    assert rb.crs == UTM

    street_union = unary_union(rb.streets.geometry).buffer(1e-6)
    shared_edge = LineString([(3, 0), (3, 3)])
    outer_edge = LineString([(0, 0), (0, 3)])
    assert shared_edge.within(street_union)
    assert outer_edge.within(street_union)


def test_region_block_identity_folds_the_existing_egress_model_tag() -> None:
    # The cache-identity fix: derive() caches on Block.identity = (source_content_hash, block_id),
    # which EXCLUDES streets, so region_block folds a model-version tag into source_content_hash --
    # otherwise a region scored under a different streets/egress model collides on the same key
    # (the bug: the old perimeter-egress eval-swap's cached access reused under the new model).
    # Needs CACHEABLE members (non-empty source_content_hash) so the tagged branch runs; the
    # _grid_block fixtures elsewhere have "" hashes and take the uncacheable "" branch.
    a = replace(_grid_block(0, 0, 3, 3, block_id="a"), source_content_hash="srcA")
    b = replace(_grid_block(3, 0, 3, 3, block_id="b"), source_content_hash="srcB")
    rb = region_block([a, b])

    assert rb.identity is not None                       # cacheable: the tagged else-branch ran
    member_hash = hashlib.sha256(
        "|".join(sorted(f"{blk.source_content_hash}:{blk.block_id}" for blk in (a, b))).encode()
    ).hexdigest()
    # NOT the bare member hash the old (perimeter-egress) model keyed on ...
    assert rb.source_content_hash != member_hash
    # ... but exactly that member hash folded under the existing-egress version tag.
    assert rb.source_content_hash == hashlib.sha256(
        ("region-existing-egress|" + member_hash).encode()).hexdigest()
    assert region_block([a, b]).source_content_hash == rb.source_content_hash   # deterministic


def test_region_block_unions_member_building_points() -> None:
    # Two blocks with explicit building_points -- region_block.building_points is their union
    # (concatenated, not deduped: overlapping points from different sources are both real sites).
    a_pts = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5), Point(1.5, 1.5)], crs=UTM)
    b_pts = gpd.GeoDataFrame(geometry=[Point(3.5, 0.5)], crs=UTM)
    a = _grid_block(0, 0, 3, 3, block_id="a", points=a_pts)
    b = _grid_block(3, 0, 3, 3, block_id="b", points=b_pts)
    rb = region_block([a, b])

    assert len(rb.building_points) == len(a_pts) + len(b_pts)
    assert rb.building_points.crs == UTM
    assert (rb.building_points.geometry.geom_type == "Point").all()


def test_region_block_rejects_empty_list() -> None:
    with pytest.raises(ValueError):
        region_block([])


def test_region_block_rejects_crs_mismatch() -> None:
    a = _grid_block(0, 0, 3, 3, block_id="a")
    other_crs = CRS.from_epsg(32644)
    b = Block(block_id="b", crs=other_crs, boundary=a.boundary,
              parcels=a.parcels.set_crs(other_crs, allow_override=True),
              streets=a.streets.set_crs(other_crs, allow_override=True))
    with pytest.raises(ValueError):
        region_block([a, b])


def test_region_reblock_reblocks_the_region_block_against_its_existing_network() -> None:
    # Two adjacent 3x3 blocks, each with its own full-perimeter streets (so the shared edge is
    # already existing inter-block street on both sides). region_reblock reblocks the region-
    # Block directly: the Result's block IS the region-block (streets = full existing network,
    # incl. the shared edge), the proposal is exactly the method's added roads (nothing is
    # pre-added), and it is deterministic.
    a = _grid_block(0, 0, 3, 3, block_id="a")
    b = _grid_block(3, 0, 3, 3, block_id="b")
    rb = region_block([a, b])

    result = region_reblock([a, b], DijkstraReblocker(), [KComplexityEval()])

    assert isinstance(result, Result)
    result_streets = unary_union(result.block.streets.geometry)
    assert result_streets.equals(unary_union(rb.streets.geometry))
    assert LineString([(3, 0), (3, 3)]).within(result_streets.buffer(1e-6))  # the shared edge

    assert result.proposal.roads is not None
    direct = DijkstraReblocker().propose(rb).roads      # region_reblock == propose on region-block
    assert direct is not None
    assert result.proposal.roads.geometry.equals(direct.geometry)


def test_region_reblock_arterial_beats_dijkstra_with_a_margin_on_a_wide_region() -> None:
    # Three 4x3 blocks in a row spanning a wide 12x3 region, with street frontage only at the
    # far ENDS (A's left edge, C's right edge; B a left stub) -- so the region's interior is deep
    # in the cross-region direction and a long cross-block arterial reaches it directly, where a
    # per-block tree can't. This is the design's headline hypothesis. NB it must be a DEEP region:
    # under the door-to-door directness basis, on a *well-served* region (full frontage) the walk
    # legs dominate and arterial's straight roads buy ~nothing over dijkstra -- the margin is real
    # only where a through-road genuinely shortens buried-parcel trips.
    a = _grid_block(0, 0, 4, 3, streets_side="left", block_id="A")
    b = _grid_block(4, 0, 4, 3, streets_side="left", block_id="B")
    c = _grid_block(8, 0, 4, 3, streets_side="right", block_id="C")
    blocks = [a, b, c]

    dij_result = region_reblock(blocks, DijkstraReblocker(), [])
    art_result = region_reblock(
        blocks,
        GreedyArterialReblocker(mode="buildable", objective="directness",
                                n_anchors=12, max_roads=6),
        [],
    )

    eval_block = dij_result.block
    assert unary_union(eval_block.streets.geometry).equals(
        unary_union(art_result.block.streets.geometry))
    assert dij_result.proposal.roads is not None and art_result.proposal.roads is not None

    _, dij_directness = efficiency_directness_curves(eval_block, dij_result.proposal.roads)
    _, art_directness = efficiency_directness_curves(eval_block, art_result.proposal.roads)
    cap = max(dij_directness.cost[-1], art_directness.cost[-1])
    auc_dij = auc(dij_directness, cap)
    auc_art = auc(art_directness, cap)

    # Recorded numbers (door-to-door directness basis): AUC dijkstra ~0.071, arterial ~0.39
    # (ratio ~5.5). On this DEEP wide region the arterial's cross-region through-road reaches the
    # buried middle directly while dijkstra's per-block trees detour, so arterial wins by a wide
    # margin -- and honestly so (the door-to-door basis, unlike the old entry-denominator one,
    # only credits directness a resident actually experiences).
    assert auc_art > 1.2 * auc_dij


def test_greedy_arterial_beats_dijkstra_directness_auc_on_a_deep_region() -> None:
    # Two 3x6 blocks side by side (sharing the interior edge x=3), each with street frontage
    # only on its OUTER short end -- opposite ends (a: bottom y=0, b: top y=6) so the joined
    # interior is deep and reaching the far corners genuinely benefits from a road that crosses
    # the old block boundary, not just a per-block tree. This is the design's headline
    # hypothesis: greedy_arterial should beat dijkstra even more decisively at region scale
    # than per-block, because a region has more room for long cross-block through-roads.
    a = _grid_block(0, 0, 3, 6, streets_side="bottom", block_id="a")
    b = _grid_block(3, 0, 3, 6, streets_side="top", block_id="b")

    dij_result = region_reblock([a, b], DijkstraReblocker(), [])
    art_result = region_reblock(
        [a, b],
        GreedyArterialReblocker(mode="buildable", objective="directness",
                                n_anchors=12, max_roads=6),
        [],
    )

    eval_block = dij_result.block
    # This fixture has partial frontage (a: bottom stub, b: top stub) and no existing street on
    # the shared edge x=3, so region_block.streets is just the two declared stubs -- the egress
    # baseline. Compare the whole network (not row 0 alone, which would pass vacuously either way).
    assert unary_union(eval_block.streets.geometry).equals(
        unary_union(art_result.block.streets.geometry))
    assert dij_result.proposal.roads is not None and art_result.proposal.roads is not None

    _, dij_directness = efficiency_directness_curves(eval_block, dij_result.proposal.roads)
    _, art_directness = efficiency_directness_curves(eval_block, art_result.proposal.roads)
    cap = max(dij_directness.cost[-1], art_directness.cost[-1])
    auc_dij = auc(dij_directness, cap)
    auc_art = auc(art_directness, cap)

    # Recorded numbers on this fixture: AUC dijkstra ~0.094, AUC arterial ~0.403. The seed is
    # empty here (no interior existing street), so this fixture's egress is unchanged by the
    # existing-egress model. Arterial wins comfortably, confirming the hypothesis. Kept as >=
    # (not >) per guidance: the real signal is the recorded numbers, not a brittle margin.
    assert auc_art >= auc_dij

    # Cross-block roads (design Scope): the arterial's own proposal -- not the seed, which is
    # empty on this fixture since no interior street exists yet -- genuinely spans from block
    # a's territory into block b's, proving joint (not per-block) reblocking.
    assert _spans_both_sides(art_result.proposal.roads, x_split=3.0, margin=1.0)


def test_identity_region_builder_sorts_each_group_for_determinism() -> None:
    geoms = _block_geoms(("B", 1, 0), ("A", 0, 0))   # touching squares, input order B, A
    assert IdentityRegionBuilder().build(geoms, [["B", "A"]]) == [["A", "B"]]


def test_identity_region_builder_no_warning_for_a_touch_adjacent_group(
    caplog: pytest.LogCaptureFixture,
) -> None:
    geoms = _block_geoms(("A", 0, 0), ("B", 1, 0))   # touching squares
    with caplog.at_level(logging.WARNING, logger="reblock.region"):
        result = IdentityRegionBuilder().build(geoms, [["A", "B"]])
    assert result == [["A", "B"]]
    assert caplog.records == []


def test_identity_region_builder_warns_for_a_disjoint_group(
    caplog: pytest.LogCaptureFixture,
) -> None:
    geoms = _block_geoms(("A", 0, 0), ("C", 5, 0))   # far apart -- gap 4 >> STREET_TOL
    with caplog.at_level(logging.WARNING, logger="reblock.region"):
        result = IdentityRegionBuilder().build(geoms, [["A", "C"]])
    assert result == [["A", "C"]]                     # still passed through unchanged
    assert any("convex_hull" in r.message for r in caplog.records)


def test_convex_hull_region_builder_fills_gaps_respects_singletons_and_allows_overlap() -> None:
    geoms = _block_geoms(
        ("A", 0, 0), ("B", 1.2, 0), ("C", 2.4, 0),        # a row with small (< STREET_TOL) gaps
        ("E", 200, 0), ("F", 201, 0),                      # overlap-test pair 1
        ("G", 200.5, 0.5), ("H", 201.5, 0.5),              # overlap-test pair 2 (overlaps E/F)
    )
    result = ConvexHullRegionBuilder().build(
        geoms, [["A", "C"], ["B"], ["E", "F"], ["G", "H"]])

    assert result[0] == ["A", "B", "C"]   # B sits inside hull(A, C) -- the gap is filled
    assert result[1] == ["B"]             # a singleton's hull is its own shape -- just itself
    # Two groups whose hulls genuinely overlap in space: each keeps (at least) its own seed
    # members, and overlap between the two expansions is allowed, not merged/deduped away.
    assert set(result[2]) >= {"E", "F"}
    assert set(result[3]) >= {"G", "H"}
    assert set(result[2]) & set(result[3])


def test_identity_region_builder_raises_clear_error_for_unknown_block_id() -> None:
    geoms = _block_geoms(("A", 0, 0), ("B", 1, 0))
    with pytest.raises(ValueError, match="ZZZ"):
        IdentityRegionBuilder().build(geoms, [["A", "ZZZ"]])


def test_convex_hull_region_builder_raises_clear_error_for_unknown_block_id() -> None:
    geoms = _block_geoms(("A", 0, 0), ("B", 1, 0))
    with pytest.raises(ValueError, match="ZZZ"):
        ConvexHullRegionBuilder().build(geoms, [["A", "ZZZ"]])


def test_kblock_source_block_geometries_is_cheap_and_wellformed() -> None:
    src = KblockSource(DJI_BLOCKS, DJI_BLD, region_id="dji")
    geoms = src.block_geometries()

    assert not geoms.empty
    assert set(geoms.columns) >= {"block_id", "geometry"}
    assert {"DJI.1_2_1267", "DJI.1_2_602"} <= set(geoms["block_id"])


def test_kblock_building_points_are_points_in_region_utm() -> None:
    src = KblockSource(DJI_BLOCKS, DJI_BLD, region_id="dji")
    pts = src.building_points()
    assert not pts.empty and (pts.geometry.geom_type == "Point").all()
    assert pts.crs == src.block_geometries().crs                 # same UTM -> overlays align


def test_kblock_building_points_bbox_windows() -> None:
    src = KblockSource(DJI_BLOCKS, DJI_BLD, region_id="dji")
    allpts = src.building_points()
    minx, miny, maxx, maxy = allpts.total_bounds
    # Bottom-left quadrant of the extent -- a strict, non-empty subset (the DJI points
    # cluster near the extent, so the geometric *middle* is empty; a corner has ~100).
    sub = (minx, miny, (minx + maxx) / 2, (miny + maxy) / 2)
    assert 0 < len(src.building_points(sub)) < len(allpts)


def test_kblock_block_carries_building_points() -> None:
    block = next(iter(KblockSource(DJI_BLOCKS, DJI_BLD, region_id="dji").region().blocks))
    assert not block.building_points.empty
    assert (block.building_points.geometry.geom_type == "Point").all()
    assert block.building_points.crs == block.crs


def test_kblock_block_geometries_bbox_windows() -> None:
    src = KblockSource(DJI_BLOCKS, DJI_BLD, region_id="dji")
    allg = src.block_geometries()
    minx, miny, maxx, maxy = allg.total_bounds
    sub = (minx, miny, (minx + maxx) / 2, (miny + maxy) / 2)
    # Strict, non-empty subset: `> 0` guards against a regression that windows to empty
    # (e.g. swapped x/y or an inverted .cx slice), which a bare `< len(allg)` would miss.
    assert 0 < len(src.block_geometries(sub)) < len(allg)
