from typing import cast

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.budget import Curve, access_burden, auc, cost_benefit_curve, road_drainage
from reblock.contracts import Block
from reblock.methods.dijkstra import DijkstraReblocker
from reblock.methods.peel import PeelReblocker

UTM = CRS.from_epsg(32643)


def _grid_block(n: int) -> Block:
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(n) for j in range(n)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(n * n))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_access_burden_is_sum_of_squared_depths() -> None:
    assert access_burden(pd.Series([1, 2, 3])) == 1 + 4 + 9


def test_road_drainage_trunks_exceed_leaves() -> None:
    # dijkstra's roads on a 5x5 grid: a segment near the street carries more parcels than a leaf.
    block = _grid_block(5)
    roads = DijkstraReblocker().propose(block).roads
    assert roads is not None
    drain = road_drainage(block, roads)
    assert len(drain) == len(roads) and max(drain) > min(drain) and max(drain) >= 2


def test_cost_benefit_curve_is_monotonic_and_reaches_full() -> None:
    block = _grid_block(5)
    roads = DijkstraReblocker().propose(block).roads
    assert roads is not None
    curve = cost_benefit_curve(block, roads, n_points=10)
    assert curve.cost[0] == 0.0 and curve.benefit[0] == 0.0
    assert curve.benefit == sorted(curve.benefit)          # monotonic non-decreasing
    assert curve.benefit[-1] > 0.5                         # flattens the block substantially


def test_auc_rewards_reaching_benefit_at_lower_cost() -> None:
    cheap = Curve(cost=[0.0, 1.0], benefit=[0.0, 1.0])     # full benefit by cost 1
    dear = Curve(cost=[0.0, 4.0], benefit=[0.0, 1.0])      # full benefit only by cost 4
    assert auc(cheap, cost_cap=4.0) > auc(dear, cost_cap=4.0)
    assert 0.0 <= auc(dear, cost_cap=4.0) <= 1.0


def test_auc_interpolates_a_cap_straddling_segment() -> None:
    # A curve whose data crosses the cap BETWEEN points must interpolate the partial area,
    # not drop the whole segment (regression: dropped it -> 0.30 instead of 0.5125).
    c = Curve(cost=[0.0, 3.0, 5.0], benefit=[0.0, 0.8, 1.0])
    assert abs(auc(c, cost_cap=4.0) - 0.5125) < 1e-6


def test_cost_benefit_curve_monotonic_with_disconnected_pocket() -> None:
    # One street-fronting parcel T + a chain P0..P3 across a GAP from T (adjacency-disconnected),
    # reachable only via a road to P0 -- then P1..P3 chain DEEP via parcel adjacency. Bridging the
    # pocket makes those deep parcels exceed the moving unreached-placeholder, so benefit would DIP
    # without a stable per-block depth cap; with the cap the curve is monotonic.
    from shapely.geometry import LineString

    t = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    pocket = [Polygon([(0, 3 + y), (2, 3 + y), (2, 4 + y), (0, 4 + y)]) for y in range(4)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(5))}, geometry=[t, *pocket], crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (2, 0)])], crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    block = Block(block_id="pk", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)
    roads = gpd.GeoDataFrame(geometry=[
        LineString([(1, 0), (1, 3)]),    # bridge street -> P0 (P0=depth1; P1..P3 chain deep)
        LineString([(1, 3), (1, 4)]),    # a leaf inside the pocket
    ], crs=UTM)
    curve = cost_benefit_curve(block, roads, n_points=6)
    assert curve.benefit == sorted(curve.benefit)    # monotonic (would dip without the cap)
    assert curve.benefit[-1] > 0.0


def test_road_drainage_floating_roads_get_zero() -> None:
    # roads with no street-connected component grant no access -> all-zero drainage.
    from shapely.geometry import LineString
    block = _grid_block(3)
    floating = gpd.GeoDataFrame(geometry=[LineString([(1.2, 1.2), (1.8, 1.8)])], crs=UTM)
    assert road_drainage(block, floating) == [0]


def test_efficiency_and_directness_rise_with_roads() -> None:
    from reblock.budget import network_efficiency
    block = _grid_block(5)
    roads = DijkstraReblocker().propose(block).roads
    assert roads is not None
    e_none, d_none = network_efficiency(block, cast(gpd.GeoDataFrame, roads.iloc[:0]))   # no roads
    e_full, d_full = network_efficiency(block, roads)
    assert e_full > e_none and d_full > d_none


def test_cost_benefit_curve_accepts_a_benefit_fn() -> None:
    from reblock.budget import cost_benefit_curve, efficiency_benefit
    block = _grid_block(5)
    roads = DijkstraReblocker().propose(block).roads
    assert roads is not None
    curve = cost_benefit_curve(block, roads, benefit_fn=efficiency_benefit, n_points=8)
    assert curve.benefit[-1] >= curve.benefit[0]      # efficiency non-decreasing with roads
    assert len(curve.cost) == len(curve.benefit)


def test_efficiency_and_directness_are_monotone_across_the_full_curve() -> None:
    # Regression for the review finding: efficiency_benefit/directness_benefit re-derived each
    # parcel's entry node against the CURRENT road subset, so entries churned as roads were
    # added and E/directness could FALL mid-curve (reproduced ~9% drops for PeelReblocker, and
    # mid-curve dips even for DijkstraReblocker). The fix freezes entries against the FULL road
    # set once; only edge availability grows across prefixes, so both metrics must be
    # non-decreasing over every point of the curve, for both methods' road layouts.
    from reblock.budget import directness_benefit, efficiency_benefit
    block = _grid_block(5)
    for method in (DijkstraReblocker(), PeelReblocker()):
        roads = method.propose(block).roads
        assert roads is not None
        for benefit_fn in (efficiency_benefit, directness_benefit):
            curve = cost_benefit_curve(block, roads, benefit_fn=benefit_fn, n_points=40)
            assert curve.benefit == sorted(curve.benefit), (
                f"{type(method).__name__} + {benefit_fn.__name__} not monotone: {curve.benefit}"
            )
