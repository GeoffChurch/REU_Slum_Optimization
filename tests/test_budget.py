from typing import cast

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon

from reblock.budget import (
    Curve,
    access_burden,
    auc,
    cost_benefit_curve,
    displacement_count,
    road_drainage,
)
from reblock.contracts import Block
from reblock.methods.dijkstra import DijkstraReblocker
from reblock.methods.mesh import MeshReblocker
from reblock.methods.peel import PeelReblocker

UTM = CRS.from_epsg(32643)


def _points(coords: list[tuple[float, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[Point(c) for c in coords], crs=UTM)


def _roads(lines: list[LineString]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=lines, crs=UTM)


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


def test_line_proximity_scores_a_sparse_straight_chord() -> None:
    # A single 2-point straight chord has only its endpoints as graph vertices, so the OLD
    # nearest-VERTEX entry rule scored parcels abreast of its middle as ~unreachable (~0
    # directness) -- undercounting sparse through-roads. Line-proximity projects each parcel onto
    # the nearest POINT on the chord, so the sparse chord genuinely serves the parcels it runs past.
    from shapely.geometry import LineString

    from reblock.budget import network_efficiency
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(3) for j in range(7)]     # 3x7 block, deep
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[LineString([(0.0, 0.0), (3.0, 0.0)])], crs=UTM)  # bottom
    block = Block(block_id="deep", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)
    chord = gpd.GeoDataFrame(geometry=[LineString([(1.5, 0.0), (1.5, 7.0)])], crs=UTM)  # spine
    _, d_none = network_efficiency(block, cast(gpd.GeoDataFrame, chord.iloc[:0]))
    _, d_chord = network_efficiency(block, chord)
    assert d_chord > d_none            # the chord helps
    assert d_chord > 0.05              # ... non-trivially -- the 2-point chord IS counted, not ~0


def test_directness_is_a_bounded_circuity_ratio() -> None:
    # Door-to-door directness = euclid(homes) / (walk + network + walk) is bounded in [0, 1] by the
    # triangle inequality -- >1 was the old rep-numerator / entry-denominator basis, which
    # line-proximity entries most amplified. A bare straight chord (the worst case) must stay <= 1.
    from shapely.geometry import LineString

    from reblock.budget import network_efficiency
    block = _grid_block(5)
    roads = DijkstraReblocker().propose(block).roads
    assert roads is not None
    _, d_dijkstra = network_efficiency(block, roads)
    chord = gpd.GeoDataFrame(geometry=[LineString([(2.5, 0.0), (2.5, 5.0)])], crs=UTM)
    _, d_chord = network_efficiency(block, chord)
    assert 0.0 <= d_dijkstra <= 1.0
    assert 0.0 <= d_chord <= 1.0


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
    for method in (DijkstraReblocker(), PeelReblocker(), MeshReblocker()):
        roads = method.propose(block).roads
        assert roads is not None
        for benefit_fn in (efficiency_benefit, directness_benefit):
            curve = cost_benefit_curve(block, roads, benefit_fn=benefit_fn, n_points=40)
            assert curve.benefit == sorted(curve.benefit), (
                f"{type(method).__name__} + {benefit_fn.__name__} not monotone: {curve.benefit}"
            )


def test_displacement_count_only_sites_within_the_corridor() -> None:
    road = _roads([LineString([(0.0, 0.0), (10.0, 0.0)])])
    pts = _points([(5.0, 0.5), (5.0, 0.99), (5.0, 2.0)])   # last one is outside a 1m corridor
    assert displacement_count(pts, road, corridor_m=1.0) == 2


def test_displacement_count_overlapping_corridors_count_a_shared_site_once() -> None:
    road_a = LineString([(0.0, 0.0), (5.0, 0.0)])
    road_b = LineString([(4.0, 0.0), (10.0, 0.0)])          # overlaps road_a's corridor near x=4-5
    roads = _roads([road_a, road_b])
    pts = _points([(1.0, 0.5), (9.0, 0.5), (4.5, 0.5)])     # last one sits in BOTH corridors
    assert displacement_count(pts, roads, corridor_m=1.0) == 3   # each site counted once, not 4


def test_displacement_count_zero_when_no_points_or_no_roads() -> None:
    road = _roads([LineString([(0.0, 0.0), (10.0, 0.0)])])
    pts = _points([(5.0, 0.0)])
    empty_pts = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=UTM)
    empty_roads = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=UTM)
    assert displacement_count(empty_pts, road, corridor_m=1.0) == 0
    assert displacement_count(pts, empty_roads, corridor_m=1.0) == 0


def test_cost_axis_is_cumulative_road_length_metres() -> None:
    block = _grid_block(5)
    roads = DijkstraReblocker().propose(block).roads
    assert roads is not None
    curve = cost_benefit_curve(block, roads)
    # x is cumulative added road length in METRES, non-decreasing, ending at total road length
    assert curve.cost[0] == 0.0
    assert curve.cost == sorted(curve.cost)
    assert abs(curve.cost[-1] - float(roads.geometry.length.sum())) < 1e-6


def test_cost_benefit_curve_has_no_cost_param() -> None:
    import inspect
    assert "cost" not in inspect.signature(cost_benefit_curve).parameters


def test_building_radii_are_half_nearest_neighbor():
    import geopandas as gpd
    from shapely.geometry import Point
    from reblock.budget import building_radii
    # three collinear points 10 m apart -> NN dist 10 for the ends, 10 for the middle -> r = 5
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(10, 0), Point(30, 0)], crs="EPSG:32734")
    r = building_radii(pts, corridor_m=3.0)
    assert list(r) == [5.0, 5.0, 10.0]      # 3rd point's NN is the 2nd, 20 m away -> r = 10


def test_building_radii_fallback_when_fewer_than_two_points():
    import geopandas as gpd
    from shapely.geometry import Point
    from reblock.budget import building_radii
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:32734")
    assert list(building_radii(pts, corridor_m=3.0)) == [3.0]     # fallback = corridor_m


def test_displacement_is_linear_ramp_in_distance_to_corridor():
    import geopandas as gpd, numpy as np
    from shapely.geometry import Point, LineString
    from reblock.budget import displacement
    crs = "EPSG:32734"
    # one road along y=0; corridor_m=1 -> corridor is the strip |y|<=1
    roads = gpd.GeoDataFrame(geometry=[LineString([(-50, 0), (50, 0)])], crs=crs)
    # point A on the corridor edge-ish (y=1 -> d=0 -> c=1); B at y=3 with r=4 -> d=2 -> c=0.5;
    # C at y=10 with r=4 -> d=9 -> c=0 (far)
    pts = gpd.GeoDataFrame(geometry=[Point(0, 1), Point(0, 3), Point(0, 10)], crs=crs)
    radii = np.array([4.0, 4.0, 4.0])
    # d_A = dist(A, strip|y|<=1) = 0 ; d_B = 3-1 = 2 ; d_C = 10-1 = 9
    got = displacement(pts, radii, roads, corridor_m=1.0)
    assert abs(got - (1.0 + 0.5 + 0.0)) < 1e-6


def test_displacement_zero_without_roads_or_points():
    import geopandas as gpd, numpy as np
    from shapely.geometry import Point
    from reblock.budget import displacement
    crs = "EPSG:32734"
    empty = gpd.GeoDataFrame(geometry=[], crs=crs)
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=crs)
    assert displacement(pts, np.array([3.0]), empty, 1.0) == 0.0
    assert displacement(empty, np.array([]), empty, 1.0) == 0.0


def _straight_block_with_two_roads() -> tuple[Block, gpd.GeoDataFrame]:
    # Two 10m-wide parcels side by side, both fronting a street along y=0; road1 (x=5) serves
    # the left parcel, road2 (x=15) serves the right one -- disjoint 3m corridors, so a
    # building point only picks up displacement once ITS road is in the drainage-ordered prefix.
    left = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    right = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])
    parcels = gpd.GeoDataFrame({"parcel_id": [0, 1]}, geometry=[left, right], crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (20, 0)])], crs=UTM)
    points = _points([(5.0, 5.0), (15.0, 5.0)])
    block = Block(block_id="two_roads", crs=UTM, boundary=boundary, parcels=parcels,
                 streets=streets, building_points=points)
    roads = _roads([LineString([(5, 0), (5, 10)]), LineString([(15, 0), (15, 10)])])
    return block, roads


def test_truncate_to_length_keeps_drainage_prefix():
    from reblock.budget import truncate_to_length
    block, roads = _straight_block_with_two_roads()   # or the shared curve fixture
    total = float(roads.geometry.length.sum())
    assert float(truncate_to_length(block, roads, total).geometry.length.sum()) == total
    assert len(truncate_to_length(block, roads, 0.0)) == 0
    half = truncate_to_length(block, roads, total / 2.0)
    assert 0.0 < float(half.geometry.length.sum()) <= total / 2.0 + 1e-6


def test_displacement_curve_is_monotonic_and_ends_at_full():
    import numpy as np
    from reblock.budget import displacement, displacement_curve
    block, roads = _straight_block_with_two_roads()
    radii = np.full(len(block.building_points), 3.0)
    curve = displacement_curve(block, roads, radii, corridor_m=3.0)
    assert curve.cost[0] == 0.0 and curve.benefit[0] == 0.0
    assert curve.benefit == sorted(curve.benefit)     # non-decreasing displacement
    assert abs(curve.benefit[-1]
               - displacement(block.building_points, radii, roads, 3.0)) < 1e-6
