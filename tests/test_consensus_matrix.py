"""scripts/consensus_matrix.py and consensus_sweep.py: a benchmark row is scored end to end on the
current metric, and the two matching conventions do what they claim."""
from __future__ import annotations

from typing import cast

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from reblock.buildings import SpacingDiscs
from reblock.contracts import Block
from reblock.emit import pct_displaced
from reblock.permeability import DEFAULT_ROAD_WIDTH_M
from scripts.consensus_matrix import ConsensusRow, Material, consensus_scorer, within_length
from scripts.consensus_sweep import displacement_matched_prefix

UTM = CRS.from_epsg(32734)


def _slab(w: int, h: int, block_id: str) -> Block:
    """`w` x `h` 10 m parcels, one building each, street frontage along the bottom edge only."""
    polys = [Polygon([(10 * i, 10 * j), (10 * i + 10, 10 * j), (10 * i + 10, 10 * j + 10),
                      (10 * i, 10 * j + 10)]) for j in range(h) for i in range(w)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    return Block(block_id=block_id, crs=UTM, boundary=cast(Polygon, unary_union(polys)),
                 parcels=parcels,
                 streets=gpd.GeoDataFrame(geometry=[LineString([(0, 0), (10 * w, 0)])], crs=UTM),
                 building_geometries=gpd.GeoDataFrame(
                     geometry=[Point(p.centroid.x + 2, p.centroid.y - 1) for p in polys],
                     crs=UTM),
                 source_content_hash=None, building_tier=SpacingDiscs)


def _paths(x: float, top: float) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[LineString([(x, 0), (x, top), (5, top)])], crs=UTM)


def test_a_row_scores_consensus_single_and_direct_against_the_blocks_own_network() -> None:
    """Every column, on today's metric, for a recipient with three donors. The footpaths are
    scored as streets -- without a width the metric refuses them."""
    scorer = consensus_scorer()
    recipient = _slab(6, 5, "r")
    donors = [_slab(6, 6, "a"), _slab(7, 5, "b"), _slab(5, 6, "c")]
    own = scorer.as_roads(_paths(25, 45))
    networks = {d.block_id: scorer.as_roads(_paths(25, 45)) for d in donors}
    quality = {d.block_id: scorer.donor_quality(d, networks[d.block_id]) for d in donors}
    assert all(0 < q <= 1 for q in quality.values())

    row = scorer.score(recipient, own, donors, networks, quality, arm="held_out", k=15)
    assert set(row) == set(ConsensusRow.__annotations__)
    assert (row["k"], row["k_requested"], row["arm"]) == (3, 15, "held_out")
    assert row["own_len_m"] == pytest.approx(float(own.geometry.length.sum()))
    assert row["perm_ratio_own"] == pytest.approx(row["perm_consensus"] / row["perm_own"])
    assert row["consensus_len_m"] >= row["own_len_m"]     # the prefix reaches the budget
    assert 0 <= row["iou_10m"] <= 1 and row["min_gw_dist"] <= row["mean_gw_dist"]


def test_material_stamps_each_blocks_footpaths_once() -> None:
    scorer = consensus_scorer()
    block = _slab(4, 4, "m")

    class Fixed:
        identity = None
        calls = 0

        def footpaths(self, bbox: tuple[float, float, float, float],
                      crs: CRS) -> gpd.GeoDataFrame:
            del bbox
            Fixed.calls += 1
            return _paths(15, 35).to_crs(crs)

    material = Material(Fixed(), scorer)
    roads = material.of(block)
    assert roads is not None and (roads["width_m"] == DEFAULT_ROAD_WIDTH_M).all()
    assert material.of(block) is roads and Fixed.calls == 1
    assert block.block_id in material.quality


def test_the_direct_prefix_stays_within_length_and_the_displacement_prefix_within_cost() -> None:
    """Both are 'largest leading prefix within budget' -- the direct baseline's conventions."""
    block = _slab(6, 5, "r")
    roads = consensus_scorer().direct_roads(block)
    total = float(roads.geometry.length.sum())
    half = within_length(roads, total / 2)
    assert float(half.geometry.length.sum()) <= total / 2 < float(
        roads.iloc[:len(half) + 1].geometry.length.sum())
    assert within_length(roads, 0.0).empty

    budget = pct_displaced(roads, block.buildings) / 2
    prefix = displacement_matched_prefix(block, roads, budget)
    assert pct_displaced(prefix, block.buildings) <= budget
    assert pct_displaced(cast(gpd.GeoDataFrame, roads.iloc[:len(prefix) + 1]),
                         block.buildings) > budget
