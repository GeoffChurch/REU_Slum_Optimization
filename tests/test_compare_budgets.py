from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.contracts import Block

UTM = CRS.from_epsg(32643)


def _deep_column_block_with_two_roads() -> tuple[Block, gpd.GeoDataFrame]:
    # Same fixture shape as tests/test_budget.py: a 4-deep column fronting a street at y=0. No roads
    # -> max depth 4; road A (right edge, bottom half) -> depth 2; {A,B} -> depth 1. One building
    # point per parcel so displacement/benefit are exercised.
    from shapely.geometry import Point
    polys = [Polygon([(0, j), (1, j), (1, j + 1), (0, j + 1)]) for j in range(4)]
    parcels = gpd.GeoDataFrame({"parcel_id": [0, 1, 2, 3]}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs=UTM)
    points = gpd.GeoDataFrame(geometry=[Point(0.5, j + 0.5) for j in range(4)], crs=UTM)
    block = Block(block_id="deep_col", crs=UTM, boundary=boundary, parcels=parcels,
                  streets=streets, building_points=points)
    roads = gpd.GeoDataFrame(geometry=[LineString([(1, 0), (1, 2)]), LineString([(1, 2), (1, 4)])],
                             crs=UTM)
    return block, roads


def test_two_lens_rows_reports_reached_target_and_matched_budget() -> None:
    from reblock.budget import truncate_to_length
    from scripts.compare_budgets import two_lens_rows

    block, roads = _deep_column_block_with_two_roads()
    budget_a = float(roads.geometry.iloc[0].length)          # room for road A only
    lens_a, lens_b = two_lens_rows(block, {"m": roads}, {"m": 0.5}, target_depth=3,
                                   budget_m=budget_a, corridor_m=3.0)
    assert len(lens_a) == 1 and len(lens_b) == 1
    (a,) = lens_a
    assert a.method == "m" and a.reached is True and a.reached_depth == 2
    assert a.propose_seconds == 0.5                          # timing passed through, not remeasured
    assert a.road_length_m > 0.0 and a.displacement >= 0.0
    (b,) = lens_b
    assert b.budget_m == budget_a
    assert 0.0 <= b.external_connectivity <= 1.0
    assert b.internal_connectivity >= 0.0
    # Lens B scores the matched-budget prefix (road A only), matching truncate_to_length.
    assert len(truncate_to_length(block, roads, budget_a)) == 1


def test_two_lens_rows_reports_floor_when_depth_target_unreachable() -> None:
    from scripts.compare_budgets import two_lens_rows

    block, roads = _deep_column_block_with_two_roads()
    lens_a, _ = two_lens_rows(block, {"m": roads}, {"m": 0.1}, target_depth=0,
                              budget_m=1.0, corridor_m=3.0)
    (a,) = lens_a
    assert a.reached is False and a.reached_depth == 1       # floor depth (> target 0)


def _street_block(x0: int, block_id: str) -> Block:
    # A 3x3 grid of unit parcels fronting a street on its bottom edge, offset to x0 so two of them
    # tile into a small 2-block region.
    polys = [Polygon([(x0 + i, j), (x0 + i + 1, j), (x0 + i + 1, j + 1), (x0 + i, j + 1)])
             for i in range(3) for j in range(3)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(9))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[LineString([(x0, 0), (x0 + 3, 0)])], crs=UTM)
    return Block(block_id=block_id, crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_run_two_lens_writes_tables_and_renders(tmp_path) -> None:
    # End-to-end glue smoke test on a tiny region with a real reblocker (DijkstraReblocker paves
    # everything, so it reaches a shallow depth). Asserts the two CSVs + a render per lens are
    # written and wall-clock propose time is captured.
    from reblock.methods.dijkstra import DijkstraReblocker
    from scripts.compare_budgets import run_two_lens

    region = [_street_block(0, "a"), _street_block(4, "b")]
    lens_a, lens_b = run_two_lens(region, {"dijkstra": DijkstraReblocker()}, target_depth=2,
                                  out_dir=tmp_path)
    assert (tmp_path / "lens_a_depth.csv").exists()
    assert (tmp_path / "lens_b_matched.csv").exists()
    assert (tmp_path / "after_dijkstra_depth2.jpg").exists()
    assert (tmp_path / "after_dijkstra_matched.jpg").exists()
    assert len(lens_a) == 1 and lens_a[0].propose_seconds > 0.0
    assert len(lens_b) == 1
