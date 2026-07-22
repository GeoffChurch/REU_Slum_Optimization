from typing import cast

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from reblock.budget import (
    Curve,
    access_burden,
    auc,
    cost_benefit_curve,
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
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import LineString, Point

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
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import Point

    from reblock.budget import displacement
    crs = "EPSG:32734"
    empty = gpd.GeoDataFrame(geometry=[], crs=crs)
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=crs)
    assert displacement(pts, np.array([3.0]), empty, 1.0) == 0.0
    assert displacement(empty, np.array([]), empty, 1.0) == 0.0


def test_displacement_counts_a_shared_site_once_under_overlapping_corridors():
    # A building whose disk sits in the OVERLAP of two roads' corridors must contribute once, not
    # once per overlapping road -- guaranteed by `displacement`'s design (one `union_all` corridor,
    # one `distance` per building), but worth a direct regression test since this exact scenario
    # used to be covered by the now-deleted `displacement_count` overlap test.
    from reblock.budget import building_radii, displacement
    crs = "EPSG:32734"
    road_a = LineString([(0.0, 0.0), (5.0, 0.0)])
    road_b = LineString([(4.0, 0.0), (10.0, 0.0)])          # overlaps road_a's corridor near x=4-5
    roads = gpd.GeoDataFrame(geometry=[road_a, road_b], crs=crs)
    pts = gpd.GeoDataFrame(geometry=[Point(1.0, 0.5), Point(9.0, 0.5), Point(4.5, 0.5)], crs=crs)
    # the 3rd point sits in BOTH corridors; all 3 are >=1.75 m from every other point (r >= 1.75)
    # and sit right on the (y=0) road line (d=0) -- each c_i = 1.0, so the sum must be exactly 3.0.
    radii = building_radii(pts, corridor_m=1.0)
    assert displacement(pts, radii, roads, corridor_m=1.0) == 3.0


def test_repulsion_is_positive_even_far_from_all_buildings():
    from shapely.geometry import LineString, Point

    from reblock.budget import building_radii, displacement, repulsion
    crs = "EPSG:32734"
    # three buildings clustered near the origin
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(0, 5), Point(5, 0)], crs=crs)
    radii = building_radii(pts, corridor_m=3.0)
    far_road = LineString([(1000.0, 1000.0), (1000.0, 1010.0)])   # nowhere near any building
    # the quadratic tail r^2/(r^2+d^2) never reaches zero -> repulsion stays strictly positive even
    # for a road far from every building (the key non-degeneracy property)...
    assert repulsion(pts, radii, far_road) > 0.0
    # ... whereas displacement's hard 0-beyond-r cutoff makes the very same far road cost 0 -- the
    # degeneracy repulsion is designed to avoid.
    far_roads = gpd.GeoDataFrame(geometry=[far_road], crs=crs)
    assert displacement(pts, radii, far_roads, corridor_m=3.0) == 0.0


def test_repulsion_higher_for_a_road_closer_to_buildings():
    from shapely.geometry import LineString, Point

    from reblock.budget import building_radii, repulsion
    crs = "EPSG:32734"
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(0, 10), Point(0, 20)], crs=crs)
    radii = building_radii(pts, corridor_m=3.0)
    near = LineString([(2.0, 0.0), (2.0, 20.0)])      # 2 m from the building column
    far = LineString([(50.0, 0.0), (50.0, 20.0)])     # 50 m away
    r_near, r_far = repulsion(pts, radii, near), repulsion(pts, radii, far)
    assert r_near > r_far > 0.0                        # closer road intrudes strictly more


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


def _trunk_leaf_block_with_two_roads() -> tuple[Block, gpd.GeoDataFrame]:
    # Road 0 (lower index) is a leaf that drains ONE parcel; road 1 (higher index) is a trunk with
    # a Y-branch off a shared stem that drains TWO -- so drainage order (road 1 first) DISAGREES
    # with input-index order (road 0 first). `_straight_block_with_two_roads` can't discriminate
    # this: both its roads score drain=0, so sort key (-drain[i], i) degenerates to index order
    # there and a sort that silently dropped the drainage term would still pass.
    leaf_parcel = Polygon([(5, 10), (9, 10), (9, 14), (5, 14)])       # corner at road 0's tip
    trunk_left = Polygon([(16, 10), (20, 10), (20, 14), (16, 14)])    # corner at trunk's left tip
    trunk_right = Polygon([(30, 10), (34, 10), (34, 14), (30, 14)])   # corner at trunk's right tip
    parcels = gpd.GeoDataFrame({"parcel_id": [0, 1, 2]},
                               geometry=[leaf_parcel, trunk_left, trunk_right], crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (40, 0)])], crs=UTM)
    block = Block(block_id="trunk_leaf", crs=UTM, boundary=boundary, parcels=parcels,
                 streets=streets)
    roads = gpd.GeoDataFrame(geometry=[
        LineString([(5, 0), (5, 10)]),                                # road 0: leaf, drains 1
        MultiLineString([
            LineString([(25, 0), (25, 5)]),
            LineString([(25, 5), (20, 10)]),
            LineString([(25, 5), (30, 10)]),
        ]),                                                           # road 1: trunk, drains 4
    ], crs=UTM)
    return block, roads


def test_truncate_to_length_orders_by_drainage_not_index():
    # Discriminator for the coverage gap above: with drainage genuinely differing across roads,
    # a budget wide enough for exactly one road must keep the HIGHER-drainage trunk (road 1) even
    # though it has the LARGER index -- an index-order (or drainage-ascending) sort would instead
    # wrongly keep the lower-drainage leaf (road 0).
    from reblock.budget import road_drainage, truncate_to_length
    block, roads = _trunk_leaf_block_with_two_roads()
    drain = road_drainage(block, roads)
    assert drain[1] > drain[0]                          # drainage order disagrees with index order
    lengths = roads.geometry.length.to_numpy()
    budget = float(lengths.max()) + 0.5                 # room for exactly one road, whichever first
    assert budget < float(lengths.sum())                # ... but not both
    result = truncate_to_length(block, roads, budget)
    assert len(result) == 1
    assert result.geometry.iloc[0].equals(roads.geometry.iloc[1])   # kept the higher-drainage trunk


def _deep_column_block_with_two_roads() -> tuple[Block, gpd.GeoDataFrame]:
    # A 1-wide, 4-deep column of unit parcels fronting a street at y=0. With no roads the peel is
    # 1,2,3,4 (max depth 4). Road A runs up the right edge for the bottom half (touches the street
    # at (1,0)); it seeds parcels 0,1 directly, AND its tip at (1,2) exactly meets parcel 2's
    # corner vertex (distance 0, so within any tol), seeding parcel 2 too -- the peel becomes
    # 1,1,1,2 (max depth 2). Road B extends the right edge to the top; it only reaches the street
    # THROUGH road A (touch-component), so {A,B} seeds all four parcels as layer 1 (max depth 1).
    # Drainage: A is the trunk (every parcel routes through it), so A sorts before B.
    polys = [Polygon([(0, j), (1, j), (1, j + 1), (0, j + 1)]) for j in range(4)]
    parcels = gpd.GeoDataFrame({"parcel_id": [0, 1, 2, 3]}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs=UTM)
    block = Block(block_id="deep_col", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)
    road_a = LineString([(1, 0), (1, 2)])
    road_b = LineString([(1, 2), (1, 4)])
    roads = gpd.GeoDataFrame(geometry=[road_a, road_b], crs=UTM)
    return block, roads


def _permeability_grid_block_and_roads() -> tuple[Block, gpd.GeoDataFrame]:
    # `_deep_column_block_with_two_roads`'s 1-unit-cell spacing corridor-SATURATES at
    # PermeabilityParams()'s default corridor_m=3.0 (a 3m corridor blankets the whole 1m-spaced
    # column regardless of road extent -- see test_permeability.py's test_monotone_under_added_roads
    # docstring for the measured trap), so road B there adds ZERO marginal permeability and can't
    # discriminate `prefix_to_permeability`'s binary search. Reuse that same test's proven-thin-
    # corridor fixture instead: a 15x15 grid of 10m parcels (large spacing keeps corridor_m=3.0 a
    # local band) with a drainage-trunk spur (x=15, depth 135) + a short cross-connector near its
    # top (y=115, x=0..30) that upgrades additional footpath edges the spur alone doesn't reach --
    # MEASURED (not assumed) strict marginal gain: 10.00% -> 11.19% footpath-edge coverage.
    k, cell = 15, 10.0
    polys, ids = [], []
    for r in range(k):
        for c in range(k):
            x0, x1, y0, y1 = c * cell, (c + 1) * cell, r * cell, (r + 1) * cell
            polys.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
            ids.append(r * k + c)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (k * cell, 0)])], crs=UTM)
    boundary = Polygon([(0, 0), (k * cell, 0), (k * cell, k * cell), (0, k * cell)])
    block = Block(block_id="perm_grid", crs=UTM, boundary=boundary, parcels=parcels,
                 streets=streets)
    roads = gpd.GeoDataFrame(geometry=[
        LineString([(15, 0), (15, 135)]),           # the drainage trunk (spur)
        LineString([(0, 115), (30, 115)]),          # cross-connector, grounded only via the spur
    ], crs=UTM)
    return block, roads


def test_max_access_depth_matches_the_peel() -> None:
    from reblock.budget import max_access_depth
    block, roads = _deep_column_block_with_two_roads()
    assert max_access_depth(block, gpd.GeoDataFrame(geometry=[], crs=UTM)) == 4   # no roads
    assert max_access_depth(block, roads) == 1                                     # both roads


def test_prefix_to_depth_returns_minimal_prefix_that_reaches_target() -> None:
    from reblock.budget import prefix_to_depth
    block, roads = _deep_column_block_with_two_roads()
    prefix, reached = prefix_to_depth(block, roads, 2)
    assert reached == 2                        # road A alone brings max depth to 2
    assert len(prefix) == 1                    # the MINIMAL prefix, not both roads
    assert prefix.geometry.iloc[0].equals(roads.geometry.iloc[0])   # road A (the drainage trunk)


def test_prefix_to_depth_reaches_a_deeper_target_only_with_all_roads() -> None:
    from reblock.budget import prefix_to_depth
    block, roads = _deep_column_block_with_two_roads()
    prefix, reached = prefix_to_depth(block, roads, 1)
    assert reached == 1
    assert len(prefix) == 2                    # needs both roads to reach depth 1


def test_prefix_to_depth_reports_floor_when_target_unreachable() -> None:
    from reblock.budget import prefix_to_depth
    block, roads = _deep_column_block_with_two_roads()
    prefix, reached = prefix_to_depth(block, roads, 0)   # depth 0 is impossible (min is 1)
    assert reached == 1                        # the floor depth (> target), reported honestly
    assert len(prefix) == len(roads)           # best effort = all roads in drainage order


def test_prefix_to_external_connectivity_returns_minimal_prefix_that_reaches_target() -> None:
    from reblock.budget import access_benefit, prefix_to_external_connectivity
    block, roads = _deep_column_block_with_two_roads()
    prefix, reached = prefix_to_external_connectivity(block, roads, 0.7)
    full = access_benefit(block, None)(roads)
    assert reached >= 0.7                      # meets the target
    assert reached <= full                     # never exceeds the full-road connectivity
    assert len(prefix) <= len(roads)           # no longer than all roads
    assert len(prefix) == 1                    # road A alone clears 0.7 (the MINIMAL prefix)
    assert prefix.geometry.iloc[0].equals(roads.geometry.iloc[0])   # road A (the drainage trunk)


def test_prefix_to_external_connectivity_reaches_a_higher_target_only_with_all_roads() -> None:
    from reblock.budget import prefix_to_external_connectivity
    block, roads = _deep_column_block_with_two_roads()
    prefix, reached = prefix_to_external_connectivity(block, roads, 0.8)
    assert reached >= 0.8
    assert len(prefix) == len(roads)           # needs both roads to reach 0.8


def test_prefix_to_external_connectivity_reports_floor_when_target_unreachable() -> None:
    from reblock.budget import prefix_to_external_connectivity
    block, roads = _deep_column_block_with_two_roads()
    prefix, reached = prefix_to_external_connectivity(block, roads, 1.5)   # 1.5 exceeds max (1.0)
    assert reached < 1.5                       # unreached, reported honestly
    assert len(prefix) == len(roads)           # best effort = all roads in drainage order
    assert prefix.geometry.iloc[0].equals(roads.geometry.iloc[0])
    assert prefix.geometry.iloc[1].equals(roads.geometry.iloc[1])


def test_prefix_to_external_connectivity_empty_roads_returns_empty() -> None:
    from reblock.budget import prefix_to_external_connectivity
    block, _roads = _deep_column_block_with_two_roads()
    empty_roads = gpd.GeoDataFrame(geometry=[], crs=UTM)
    prefix, reached = prefix_to_external_connectivity(block, empty_roads, 0.5)
    assert len(prefix) == 0
    assert reached == 0.0


def test_prefix_to_permeability_returns_minimal_prefix_that_reaches_target() -> None:
    from reblock.budget import prefix_to_permeability
    from reblock.permeability import permeability
    block, roads = _permeability_grid_block_and_roads()
    p1 = permeability(block, cast(gpd.GeoDataFrame, roads.iloc[:1]))
    p2 = permeability(block, roads)
    assert 0.0 < p1 < p2                       # spur alone helps but the pair does strictly more
    prefix, reached = prefix_to_permeability(block, roads, p1)
    assert reached
    assert len(prefix) == 1                    # the MINIMAL prefix, not both roads
    assert prefix.geometry.iloc[0].equals(roads.geometry.iloc[0])   # the spur (the drainage trunk)


def test_prefix_to_permeability_reaches_a_higher_target_only_with_all_roads() -> None:
    from reblock.budget import prefix_to_permeability
    from reblock.permeability import permeability
    block, roads = _permeability_grid_block_and_roads()
    p1 = permeability(block, cast(gpd.GeoDataFrame, roads.iloc[:1]))
    p2 = permeability(block, roads)
    target = (p1 + p2) / 2.0                   # strictly between: needs both roads
    prefix, reached = prefix_to_permeability(block, roads, target)
    assert reached
    assert len(prefix) == 2


def test_prefix_to_permeability_reports_unreached_when_target_unreachable() -> None:
    from reblock.budget import prefix_to_permeability
    block, roads = _permeability_grid_block_and_roads()
    prefix, reached = prefix_to_permeability(block, roads, 1.5)   # 1.5 exceeds max (permeability<1)
    assert not reached
    assert len(prefix) == len(roads)           # best effort = all roads in drainage order
    assert prefix.geometry.iloc[0].equals(roads.geometry.iloc[0])
    assert prefix.geometry.iloc[1].equals(roads.geometry.iloc[1])


def test_prefix_to_permeability_empty_roads_returns_empty_unreached() -> None:
    from reblock.budget import prefix_to_permeability
    block, _roads = _permeability_grid_block_and_roads()
    empty_roads = gpd.GeoDataFrame(geometry=[], crs=UTM)
    prefix, reached = prefix_to_permeability(block, empty_roads, 0.1)
    assert len(prefix) == 0
    assert not reached


def test_displacement_curve_is_monotonic_and_ends_at_full():
    import numpy as np

    from reblock.budget import displacement, displacement_curve
    block, roads = _straight_block_with_two_roads()
    radii = np.full(len(block.building_points), 3.0)
    curve = displacement_curve(block, roads, radii, corridor_m=3.0)
    n = len(block.building_points)
    assert curve.cost[0] == 0.0 and curve.benefit[0] == 0.0
    assert curve.benefit == sorted(curve.benefit)     # non-decreasing displacement
    assert abs(curve.benefit[-1]
               - displacement(block.building_points, radii, roads, 3.0) / n) < 1e-6


def test_displacement_curve_is_home_fraction() -> None:
    from reblock.budget import building_radii, displacement, displacement_curve
    block, roads = _straight_block_with_two_roads()   # existing helper with building_points
    radii = building_radii(block.building_points, 3.0)
    curve = displacement_curve(block, roads, radii, corridor_m=3.0)
    n = len(block.building_points)
    assert all(0.0 <= b <= 1.0 for b in curve.benefit)          # fraction, not a count
    # terminal fraction == displacement(full roads)/n_buildings
    assert abs(curve.benefit[-1]
               - displacement(block.building_points, radii, roads, 3.0) / n) < 1e-9


def test_prefix_to_displacement_returns_minimal_prefix_that_reaches_fraction() -> None:
    import numpy as np

    from reblock.budget import displacement, prefix_to_displacement
    block, roads = _straight_block_with_two_roads()
    radii = np.full(len(block.building_points), 3.0)
    n = len(block.building_points)
    frac1 = displacement(block.building_points, radii,
                         cast(gpd.GeoDataFrame, roads.iloc[:1]), 3.0) / n
    frac2 = displacement(block.building_points, radii, roads, 3.0) / n
    assert 0.0 < frac1 < frac2                  # road 0 alone displaces only its own building
    prefix = prefix_to_displacement(block, roads, radii, frac1, corridor_m=3.0)
    assert len(prefix) == 1                     # the MINIMAL prefix, not both roads
    assert prefix.geometry.iloc[0].equals(roads.geometry.iloc[0])


def test_prefix_to_displacement_needs_all_roads_for_a_higher_fraction() -> None:
    import numpy as np

    from reblock.budget import displacement, prefix_to_displacement
    block, roads = _straight_block_with_two_roads()
    radii = np.full(len(block.building_points), 3.0)
    n = len(block.building_points)
    frac1 = displacement(block.building_points, radii,
                         cast(gpd.GeoDataFrame, roads.iloc[:1]), 3.0) / n
    frac2 = displacement(block.building_points, radii, roads, 3.0) / n
    target = (frac1 + frac2) / 2.0              # strictly between: needs both roads
    prefix = prefix_to_displacement(block, roads, radii, target, corridor_m=3.0)
    assert len(prefix) == 2


def test_prefix_to_displacement_returns_all_roads_when_fraction_unreachable() -> None:
    import numpy as np

    from reblock.budget import prefix_to_displacement
    block, roads = _straight_block_with_two_roads()
    radii = np.full(len(block.building_points), 3.0)
    prefix = prefix_to_displacement(block, roads, radii, 1.5, corridor_m=3.0)   # > 1.0, impossible
    assert len(prefix) == len(roads)            # best effort = all roads in drainage order


def test_prefix_to_displacement_empty_roads_returns_empty() -> None:
    import numpy as np

    from reblock.budget import prefix_to_displacement
    block, _roads = _straight_block_with_two_roads()
    radii = np.full(len(block.building_points), 3.0)
    empty_roads = gpd.GeoDataFrame(geometry=[], crs=UTM)
    prefix = prefix_to_displacement(block, empty_roads, radii, 0.5, corridor_m=3.0)
    assert len(prefix) == 0


def test_external_internal_displacement_curves_share_cost_samples():
    # emit.compare_report re-bases the two benefit curves' plotted x-axis onto the displacement
    # curve's cumulative Σcᵢ/n_buildings, which is only valid if all three curves are
    # INDEX-ALIGNED: same drainage-ordered _sweep, same n_points=20, over the same roads, so their
    # `.cost` samples (still cumulative added road length, m, for all three) land at identical
    # budgets. A future change making _sweep's sampling value-dependent would silently misalign
    # every plot.
    from reblock.budget import (
        access_benefit,
        building_radii,
        commute_ratio_benefit,
        displacement_curve,
    )
    block, roads = _straight_block_with_two_roads()
    radii = building_radii(block.building_points, corridor_m=3.0)
    external = cost_benefit_curve(block, roads, benefit_fn=access_benefit)
    internal = cost_benefit_curve(block, roads, benefit_fn=commute_ratio_benefit)
    disp = displacement_curve(block, roads, radii, corridor_m=3.0)
    assert list(external.cost) == list(internal.cost) == list(disp.cost)
