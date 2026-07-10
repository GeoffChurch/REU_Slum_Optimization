from typing import cast

import geopandas as gpd
import networkx as nx
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from reblock.budget import auc, efficiency_directness_curves
from reblock.contracts import Block, Result
from reblock.eval.kcomplexity import KComplexityEval
from reblock.methods.arterial import GreedyArterialReblocker
from reblock.methods.dijkstra import DijkstraReblocker
from reblock.region import region_block, region_perimeter, region_reblock, region_seed_roads

UTM = CRS.from_epsg(32643)

_SIDES = {
    "bottom": lambda x0, y0, w, h: LineString([(x0, y0), (x0 + w, y0)]),
    "top": lambda x0, y0, w, h: LineString([(x0, y0 + h), (x0 + w, y0 + h)]),
    "left": lambda x0, y0, w, h: LineString([(x0, y0), (x0, y0 + h)]),
    "right": lambda x0, y0, w, h: LineString([(x0 + w, y0), (x0 + w, y0 + h)]),
}


def _grid_block(x0: int, y0: int, w: int, h: int, streets_side: str = "all",
                block_id: str = "grid") -> Block:
    """A w x h grid of unit parcels at (x0, y0). `streets_side="all"` (the default) gives the
    full block-perimeter frontage (as if every existing road around the block is already
    street); a side name ("bottom"/"top"/"left"/"right") gives frontage on only that outer
    edge, for building a deep block/region."""
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
    return Block(block_id=block_id, crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


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
    # Under the seed model, region_block.streets = the union of every block's OWN existing
    # streets (perimeter + inter-block), so the interior shared edge -- present in both a's and
    # b's own streets -- is now INCLUDED, not dropped (contrast the old perimeter-only model,
    # now `region_perimeter`, tested below).
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


def test_region_perimeter_drops_the_shared_interior_edge() -> None:
    a = _grid_block(0, 0, 3, 3, block_id="a")
    b = _grid_block(3, 0, 3, 3, block_id="b")
    perim = region_perimeter([a, b])

    assert len(perim) == 1
    assert perim.crs == UTM
    perim_union = unary_union(perim.geometry).buffer(1e-6)
    shared_edge = LineString([(3, 0), (3, 3)])
    outer_edge = LineString([(0, 0), (0, 3)])
    assert not shared_edge.within(perim_union)
    assert outer_edge.within(perim_union)


def test_region_seed_roads_is_the_interior_existing_roads() -> None:
    a = _grid_block(0, 0, 3, 3, block_id="a")
    b = _grid_block(3, 0, 3, 3, block_id="b")
    seed = region_seed_roads([a, b])

    assert len(seed) >= 1
    assert seed.crs == UTM
    seed_union = unary_union(seed.geometry).buffer(1e-6)
    # The shared edge's midpoint (away from its corners, which the perimeter-buffer erosion
    # legitimately eats into -- STREET_TOL either side of y=0/y=3) IS in the seed; a genuine
    # outer-edge midpoint is NOT.
    shared_midpoint = Point(3, 1.5)
    outer_midpoint = Point(0, 1.5)
    assert shared_midpoint.within(seed_union)
    assert not outer_midpoint.within(seed_union)


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


def test_region_reblock_scores_seed_and_added_roads_against_the_perimeter_egress() -> None:
    # Two adjacent 3x3 blocks, each with its own full-perimeter streets (so the shared edge is
    # already "existing road" on both sides) -- the seed model's simplest case.
    a = _grid_block(0, 0, 3, 3, block_id="a")
    b = _grid_block(3, 0, 3, 3, block_id="b")
    seed = region_seed_roads([a, b])
    perim = region_perimeter([a, b])
    assert len(seed) >= 1

    result = region_reblock([a, b], DijkstraReblocker(), [KComplexityEval()])

    assert isinstance(result, Result)
    assert result.proposal.roads is not None
    assert len(result.proposal.roads) >= len(seed)
    assert result.block.streets.geometry.iloc[0].equals(perim.geometry.iloc[0])

    again = region_reblock([a, b], DijkstraReblocker(), [KComplexityEval()])
    assert again.proposal.roads is not None
    assert result.proposal.roads.geometry.equals(again.proposal.roads.geometry)


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
    assert eval_block.streets.geometry.iloc[0].equals(art_result.block.streets.geometry.iloc[0])
    assert dij_result.proposal.roads is not None and art_result.proposal.roads is not None

    _, dij_directness = efficiency_directness_curves(eval_block, dij_result.proposal.roads)
    _, art_directness = efficiency_directness_curves(eval_block, art_result.proposal.roads)
    cap = max(dij_directness.cost[-1], art_directness.cost[-1])
    auc_dij = auc(dij_directness, cap)
    auc_art = auc(art_directness, cap)

    # Recorded numbers on this fixture: AUC dijkstra ~0.314, AUC arterial ~0.476 -- arterial
    # wins comfortably, confirming the hypothesis. Kept as >= (not >) per the task's guidance:
    # the real signal is the recorded numbers, not a brittle margin.
    assert auc_art >= auc_dij

    # Cross-block roads (design Scope): the arterial's own proposal -- not the seed, which is
    # empty on this fixture since no interior street exists yet -- genuinely spans from block
    # a's territory into block b's, proving joint (not per-block) reblocking.
    assert _spans_both_sides(art_result.proposal.roads, x_split=3.0, margin=1.0)
