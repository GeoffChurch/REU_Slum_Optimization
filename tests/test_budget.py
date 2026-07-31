from typing import cast

import geopandas as gpd
import networkx as nx
import pandas as pd
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union

from reblock.budget import (
    access_burden,
    road_drainage,
    street_first_ordered,
)
from reblock.contracts import Block
from reblock.derive.access import STREET_TOL
from reblock.methods.clearance import ClearanceReblocker
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, with_width

UTM = CRS.from_epsg(32643)


def _points(coords: list[tuple[float, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[Point(c) for c in coords], crs=UTM)


def _roads(lines: list[LineString]) -> gpd.GeoDataFrame:
    return with_width(gpd.GeoDataFrame(geometry=lines, crs=UTM), DEFAULT_ROAD_WIDTH_M)


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
    # clearance's roads on a 5x5 grid: a segment near the street carries more parcels than a leaf.
    # depth_target=1 (not the default 2): the grid's only depth>2 parcel is the single center
    # cell, so the default target is satisfied by ONE road (no trunk/leaf branching to measure);
    # depth_target=1 forces every parcel to the street, producing a genuine branching tree.
    block = _grid_block(5)
    roads = ClearanceReblocker(depth_target=1).propose(block).roads
    assert roads is not None
    drain = road_drainage(block, roads)
    assert len(drain) == len(roads) and max(drain) > min(drain) and max(drain) >= 2


def test_road_drainage_floating_roads_get_zero() -> None:
    # roads with no street-connected component grant no access -> all-zero drainage.
    from shapely.geometry import LineString
    block = _grid_block(3)
    floating = with_width(
        gpd.GeoDataFrame(geometry=[LineString([(1.2, 1.2), (1.8, 1.8)])], crs=UTM),
        DEFAULT_ROAD_WIDTH_M)
    assert road_drainage(block, floating) == [0]


def test_efficiency_and_directness_rise_with_roads() -> None:
    from reblock.budget import network_efficiency
    block = _grid_block(5)
    roads = ClearanceReblocker().propose(block).roads
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
    chord = with_width(gpd.GeoDataFrame(geometry=[LineString([(1.5, 0.0), (1.5, 7.0)])], crs=UTM),
                       DEFAULT_ROAD_WIDTH_M)  # spine
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
    roads = ClearanceReblocker().propose(block).roads
    assert roads is not None
    _, d_roads = network_efficiency(block, roads)
    chord = with_width(gpd.GeoDataFrame(geometry=[LineString([(2.5, 0.0), (2.5, 5.0)])], crs=UTM),
                       DEFAULT_ROAD_WIDTH_M)
    _, d_chord = network_efficiency(block, chord)
    assert 0.0 <= d_roads <= 1.0
    assert 0.0 <= d_chord <= 1.0


def test_building_radii_are_half_nearest_neighbor():
    import geopandas as gpd
    from shapely.geometry import Point

    from reblock.budget import building_radii
    # three collinear points 10 m apart -> NN dist 10 for the ends, 10 for the middle -> r = 5
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(10, 0), Point(30, 0)], crs="EPSG:32734")
    r = building_radii(pts)
    assert list(r) == [5.0, 5.0, 10.0]      # 3rd point's NN is the 2nd, 20 m away -> r = 10


def test_building_radii_fallback_when_fewer_than_two_points():
    import geopandas as gpd
    from shapely.geometry import Point

    from reblock.budget import building_radii
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:32734")
    assert list(building_radii(pts)) == [3.0]     # fallback = corridor_m


def test_displacement_is_linear_ramp_in_distance_to_corridor():
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import LineString, Point

    from reblock.budget import displacement
    crs = "EPSG:32734"
    # one road along y=0; width 2 m -> corridor is the strip |y|<=1
    roads = with_width(gpd.GeoDataFrame(geometry=[LineString([(-50, 0), (50, 0)])], crs=crs),
                       2.0)
    # point A on the corridor edge-ish (y=1 -> d=0 -> c=1); B at y=3 with r=4 -> d=2 -> c=0.5;
    # C at y=10 with r=4 -> d=9 -> c=0 (far)
    pts = gpd.GeoDataFrame(geometry=[Point(0, 1), Point(0, 3), Point(0, 10)], crs=crs)
    radii = np.array([4.0, 4.0, 4.0])
    # d_A = dist(A, strip|y|<=1) = 0 ; d_B = 3-1 = 2 ; d_C = 10-1 = 9
    got = displacement(pts, radii, roads)
    assert abs(got - (1.0 + 0.5 + 0.0)) < 1e-6


def test_displacement_zero_without_roads_or_points():
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import Point

    from reblock.budget import displacement
    crs = "EPSG:32734"
    empty = gpd.GeoDataFrame(geometry=[], crs=crs)
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=crs)
    assert displacement(pts, np.array([3.0]), empty) == 0.0
    assert displacement(empty, np.array([]), empty) == 0.0


def test_displacement_counts_a_shared_site_once_under_overlapping_corridors():
    # A building whose disk sits in the OVERLAP of two roads' corridors must contribute once, not
    # once per overlapping road -- guaranteed by `displacement`'s design (one `union_all` corridor,
    # one `distance` per building), but worth a direct regression test since this exact scenario
    # used to be covered by the now-deleted `displacement_count` overlap test.
    from reblock.budget import building_radii, displacement
    crs = "EPSG:32734"
    road_a = LineString([(0.0, 0.0), (5.0, 0.0)])
    road_b = LineString([(4.0, 0.0), (10.0, 0.0)])          # overlaps road_a's corridor near x=4-5
    roads = with_width(gpd.GeoDataFrame(geometry=[road_a, road_b], crs=crs), 2.0)
    pts = gpd.GeoDataFrame(geometry=[Point(1.0, 0.5), Point(9.0, 0.5), Point(4.5, 0.5)], crs=crs)
    # the 3rd point sits in BOTH corridors; all 3 are >=1.75 m from every other point (r >= 1.75)
    # and sit right on the (y=0) road line (d=0) -- each c_i = 1.0, so the sum must be exactly 3.0.
    radii = building_radii(pts)
    assert displacement(pts, radii, roads) == 3.0


def test_repulsion_is_positive_even_far_from_all_buildings():
    from shapely.geometry import LineString, Point

    from reblock.budget import building_radii, displacement, repulsion
    crs = "EPSG:32734"
    # three buildings clustered near the origin
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(0, 5), Point(5, 0)], crs=crs)
    radii = building_radii(pts)
    far_road = LineString([(1000.0, 1000.0), (1000.0, 1010.0)])   # nowhere near any building
    # the quadratic tail r^2/(r^2+d^2) never reaches zero -> repulsion stays strictly positive even
    # for a road far from every building (the key non-degeneracy property)...
    assert repulsion(pts, radii, far_road) > 0.0
    # ... whereas displacement's hard 0-beyond-r cutoff makes the very same far road cost 0 -- the
    # degeneracy repulsion is designed to avoid.
    far_roads = with_width(gpd.GeoDataFrame(geometry=[far_road], crs=crs), DEFAULT_ROAD_WIDTH_M)
    assert displacement(pts, radii, far_roads) == 0.0


def test_repulsion_higher_for_a_road_closer_to_buildings():
    from shapely.geometry import LineString, Point

    from reblock.budget import building_radii, repulsion
    crs = "EPSG:32734"
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(0, 10), Point(0, 20)], crs=crs)
    radii = building_radii(pts)
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
    roads = with_width(gpd.GeoDataFrame(geometry=[road_a, road_b], crs=UTM), DEFAULT_ROAD_WIDTH_M)
    return block, roads


def _permeability_grid_block_and_roads() -> tuple[Block, gpd.GeoDataFrame]:
    # `_deep_column_block_with_two_roads`'s 1-unit-cell spacing corridor-SATURATES at a
    # default 6 m road's 3 m half-width (that corridor blankets the whole 1m-spaced
    # column regardless of road extent -- see test_permeability.py's test_monotone_under_added_roads
    # docstring for the measured trap), so road B there adds ZERO marginal permeability and can't
    # discriminate `prefix_to_permeability`'s binary search. Reuse that same test's proven-thin-
    # corridor fixture instead: a 15x15 grid of 10m parcels (large spacing keeps the 3 m
    # half-width a local band) with a drainage-trunk spur (x=15, depth 135) + a
    # short cross-connector near its top (y=115, x=0..30) that upgrades additional
    # footpath edges the spur alone doesn't reach --
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
    return block, with_width(roads, DEFAULT_ROAD_WIDTH_M)


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
    curve = displacement_curve(block, roads, radii)
    n = len(block.building_points)
    assert curve.cost[0] == 0.0 and curve.benefit[0] == 0.0
    assert curve.benefit == sorted(curve.benefit)     # non-decreasing displacement
    assert abs(curve.benefit[-1]
               - displacement(block.building_points, radii, roads) / n) < 1e-6
    # cost axis = cumulative added road length in METRES, non-decreasing, ending at the full
    # road length -- a `_sweep` property formerly pinned only by the retired
    # test_cost_axis_is_cumulative_road_length_metres (via cost_benefit_curve); migrated here
    # onto displacement_curve (the lightest live _sweep vehicle) to keep it covered.
    assert curve.cost == sorted(curve.cost)
    assert abs(curve.cost[-1] - float(roads.geometry.length.sum())) < 1e-6


def test_displacement_curve_is_home_fraction() -> None:
    from reblock.budget import building_radii, displacement, displacement_curve
    block, roads = _straight_block_with_two_roads()   # existing helper with building_points
    radii = building_radii(block.building_points)
    curve = displacement_curve(block, roads, radii)
    n = len(block.building_points)
    assert all(0.0 <= b <= 1.0 for b in curve.benefit)          # fraction, not a count
    # terminal fraction == displacement(full roads)/n_buildings
    assert abs(curve.benefit[-1]
               - displacement(block.building_points, radii, roads) / n) < 1e-9


def test_prefix_to_displacement_returns_minimal_prefix_that_reaches_fraction() -> None:
    import numpy as np

    from reblock.budget import displacement, prefix_to_displacement
    block, roads = _straight_block_with_two_roads()
    radii = np.full(len(block.building_points), 3.0)
    n = len(block.building_points)
    frac1 = displacement(block.building_points, radii,
                         cast(gpd.GeoDataFrame, roads.iloc[:1])) / n
    frac2 = displacement(block.building_points, radii, roads) / n
    assert 0.0 < frac1 < frac2                  # road 0 alone displaces only its own building
    prefix = prefix_to_displacement(block, roads, radii, frac1)
    assert len(prefix) == 1                     # the MINIMAL prefix, not both roads
    assert prefix.geometry.iloc[0].equals(roads.geometry.iloc[0])


def test_prefix_to_displacement_needs_all_roads_for_a_higher_fraction() -> None:
    import numpy as np

    from reblock.budget import displacement, prefix_to_displacement
    block, roads = _straight_block_with_two_roads()
    radii = np.full(len(block.building_points), 3.0)
    n = len(block.building_points)
    frac1 = displacement(block.building_points, radii,
                         cast(gpd.GeoDataFrame, roads.iloc[:1])) / n
    frac2 = displacement(block.building_points, radii, roads) / n
    target = (frac1 + frac2) / 2.0              # strictly between: needs both roads
    prefix = prefix_to_displacement(block, roads, radii, target)
    assert len(prefix) == 2


def test_prefix_to_displacement_returns_all_roads_when_fraction_unreachable() -> None:
    import numpy as np

    from reblock.budget import prefix_to_displacement
    block, roads = _straight_block_with_two_roads()
    radii = np.full(len(block.building_points), 3.0)
    prefix = prefix_to_displacement(block, roads, radii, 1.5)   # > 1.0, impossible
    assert len(prefix) == len(roads)            # best effort = all roads in drainage order


def test_prefix_to_displacement_empty_roads_returns_empty() -> None:
    import numpy as np

    from reblock.budget import prefix_to_displacement
    block, _roads = _straight_block_with_two_roads()
    radii = np.full(len(block.building_points), 3.0)
    empty_roads = gpd.GeoDataFrame(geometry=[], crs=UTM)
    prefix = prefix_to_displacement(block, empty_roads, radii, 0.5)
    assert len(prefix) == 0


def test_permeability_and_displacement_curves_share_cost_samples():
    # emit.compare_report pairs (displacement[i], permeability[i]) on the plotted frontier, which
    # is only valid if both curves are INDEX-ALIGNED: same drainage-ordered _sweep, same
    # n_points=20, over the same roads, so their `.cost` samples (cumulative added road length, m,
    # for both) land at identical budgets. A future change making _sweep's sampling
    # value-dependent would silently misalign every plot.
    from reblock.budget import building_radii, displacement_curve
    from reblock.permeability import PermeabilityParams, permeability_curve
    block, roads = _straight_block_with_two_roads()
    radii = building_radii(block.building_points)
    perm = permeability_curve(block, roads, PermeabilityParams())
    disp = displacement_curve(block, roads, radii)
    assert list(perm.cost) == list(disp.cost)


def test_every_scored_prefix_reaches_the_street() -> None:
    """`street_first_ordered` must make every prefix a connected network reaching the street.

    The lenses score a PREFIX of a method's roads, so a prefix that does not reach the street is a
    road set nobody could build -- and permeability still credits it, because an isolated corridor
    upgrades local adjacency conductance. Under the previous drainage-descending order this failed
    for every loop-bearing method (measured fraction of prefix length connected:
    greedy_arterial_repulsion 0.782, resistance_greedy 0.900, clearance_looped 0.831 at region
    scale).

    The fixture inverts drainage against street distance, which is the only way that order can
    fail. A LOOP -- two branches up from the street joined by a far bar -- with every parcel on the
    BAR and none beside the branches. Each branch carries only the parcels that route down its own
    side, so the bar's drainage exceeds either branch's, and drainage-descending puts the bar
    (which touches nothing) first.

    FAULT INJECTION: dropping the street-distance term from the sort key -- i.e. restoring
    `(-drain[i], i)` -- orders the bar first and this fails with a disconnected prefix.
    """
    street = LineString([(0.0, 0.0), (100.0, 0.0)])
    xs = [20.0, 35.0, 50.0, 65.0, 80.0]
    roads = gpd.GeoDataFrame(geometry=[
        # Far bar, vertexed at each parcel: `road_drainage` attaches parcels to NODES, and a
        # two-point LineString has nodes only at its ends.
        LineString([(10.0, 20.0), *[(x, 20.0) for x in xs], (90.0, 20.0)]),
        LineString([(10.0, 0.0), (10.0, 20.0)]),      # left branch
        LineString([(90.0, 0.0), (90.0, 20.0)]),      # right branch
    ], crs=UTM)
    # Parcels sit ON the bar and nowhere near the branches, so each branch drains only its share.
    parcels = gpd.GeoDataFrame(
        {"parcel_id": [str(k) for k in range(len(xs))]},
        geometry=[Polygon([(x - 5, 20), (x + 5, 20), (x + 5, 30), (x - 5, 30)]) for x in xs],
        crs=UTM)
    block = Block(
        block_id="loop", crs=UTM,
        boundary=Polygon([(0, 0), (100, 0), (100, 40), (0, 40)]),
        parcels=parcels,
        streets=gpd.GeoDataFrame(geometry=[street], crs=UTM),
        building_points=gpd.GeoDataFrame(
            geometry=[Point(x, 25) for x in xs], crs=UTM))

    drain = road_drainage(block, roads)
    assert drain[0] > max(drain[1], drain[2]), (
        f"fixture is vacuous: the far bar must out-drain both branches, got {drain}")

    ordered = street_first_ordered(block, roads, STREET_TOL)
    for k in range(1, len(ordered) + 1):
        prefix = ordered.iloc[:k]
        # prefix + street must be ONE connected component. Checking each noded piece's own distance
        # to the street would be wrong -- the bar is legitimately reached THROUGH a branch.
        merged = unary_union([*prefix.geometry, street])
        pieces = list(merged.geoms) if isinstance(merged, MultiLineString) else [merged]
        g: nx.Graph = nx.Graph()
        g.add_nodes_from(range(len(pieces)))
        for i, a in enumerate(pieces):
            for j, b in enumerate(pieces[:i]):
                if a.distance(b) <= STREET_TOL:
                    g.add_edge(i, j)
        assert nx.number_connected_components(g) == 1, (
            f"prefix of {k} road(s) is disconnected from the street: {list(prefix.geometry)}")


def test_ordering_reduces_to_plain_drainage_when_that_is_already_buildable() -> None:
    """On a drainage TREE the constrained order must equal plain drainage-descending, exactly.

    This is what makes the constraint a fix rather than a new bias. A tree's drainage order is
    already buildable, so constraining it must change nothing -- otherwise every existing
    tree-method number would shift for no reason.

    It also pins the CHOICE of constraint. Ordering by network distance to the street also
    guarantees connectivity, but it is breadth-first: it completes a whole ring before going
    deeper, penalizing fine-grained networks for granularity rather than geometry. The fixture
    separates the two -- two branches off the street, the BUSIER one ordered first by drainage but
    tied with the other by distance -- so a distance key gives a different answer.

    FAULT INJECTION: replacing the heap key `-drain[i]` with a street-distance key orders
    `[A, B, A2, B2]` instead of drainage's `[B, B2, A, A2]` and fails here.
    """
    street = LineString([(0.0, 0.0), (100.0, 0.0)])
    roads = gpd.GeoDataFrame(geometry=[
        LineString([(20.0, 0.0), (20.0, 10.0)]),                     # A  -- quiet branch stem
        LineString([(20.0, 10.0), (20.0, 20.0)]),                    # A2 -- 1 parcel
        LineString([(80.0, 0.0), (80.0, 10.0)]),                     # B  -- busy branch stem
        # Single segment on purpose: `road_drainage` counts segment TRAVERSALS, so a
        # multi-vertex road accumulates inflated drainage and would muddle this fixture.
        LineString([(80.0, 10.0), (80.0, 20.0)]),                    # B2 -- 2 parcels
    ], crs=UTM)
    parcels = gpd.GeoDataFrame(
        {"parcel_id": ["a", "b", "c"]},
        geometry=[Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]),
                  Polygon([(80, 20), (90, 20), (90, 30), (80, 30)]),
                  Polygon([(70, 20), (80, 20), (80, 30), (70, 30)])],
        crs=UTM)
    block = Block(
        block_id="tree", crs=UTM,
        boundary=Polygon([(0, 0), (100, 0), (100, 45), (0, 45)]),
        parcels=parcels,
        streets=gpd.GeoDataFrame(geometry=[street], crs=UTM),
        building_points=gpd.GeoDataFrame(
            geometry=[Point(25, 25), Point(85, 25), Point(75, 25)], crs=UTM))

    drain = road_drainage(block, roads)
    plain = sorted(range(len(roads)), key=lambda i: (-drain[i], i))
    # Only meaningful if the busy branch outranks the quiet one -- i.e. if drainage and street
    # distance disagree. Distance puts A (index 0) first; drainage puts B (index 2).
    assert plain[0] == 2, f"busy stem must lead plain drainage order, got {plain} for {drain}"

    ordered = street_first_ordered(block, roads, STREET_TOL)
    got = [list(roads.geometry).index(g) for g in ordered.geometry]
    assert got == plain, f"constrained order {got} differs from plain drainage {plain}"


def test_drainage_counts_parcels_not_segment_traversals() -> None:
    """A road's drainage must not depend on how many vertices it is drawn with.

    Drainage is documented as a per-road PARCEL count and is the ranking key for the lens prefix
    order, so inflating vertex-dense roads biases every prefix walk toward them -- a count of
    geometry rather than of traffic.

    Two identical geometries, one drawn with a single segment and one subdivided into four, each
    serving the same single parcel from the same street. Their drainage must be equal.

    FAULT INJECTION: summing per traversed segment (`counts[row] += 1` inside the path loop, instead
    of collecting the road set first) scores the subdivided road 4 against the plain road's 1.
    """
    street = LineString([(0.0, 0.0), (100.0, 0.0)])

    def block_with(spur: LineString, parcel_at: tuple[float, float]) -> Block:
        x, y = parcel_at
        return Block(
            block_id="d", crs=UTM,
            boundary=Polygon([(0, 0), (100, 0), (100, 60), (0, 60)]),
            parcels=gpd.GeoDataFrame(
                {"parcel_id": ["a"]},
                geometry=[Polygon([(x, y), (x + 10, y), (x + 10, y + 10), (x, y + 10)])],
                crs=UTM),
            streets=gpd.GeoDataFrame(geometry=[street], crs=UTM),
            building_points=gpd.GeoDataFrame(geometry=[Point(x + 5, y + 5)], crs=UTM))

    plain = LineString([(20.0, 0.0), (20.0, 40.0)])
    subdivided = LineString([(20.0, 0.0), (20.0, 10.0), (20.0, 20.0), (20.0, 30.0), (20.0, 40.0)])
    d_plain = road_drainage(block_with(plain, (20.0, 40.0)),
                            gpd.GeoDataFrame(geometry=[plain], crs=UTM))
    d_sub = road_drainage(block_with(subdivided, (20.0, 40.0)),
                          gpd.GeoDataFrame(geometry=[subdivided], crs=UTM))
    assert d_plain == [1], f"one parcel on one road must read 1, got {d_plain}"
    assert d_sub == d_plain, (
        f"subdividing a road changed drainage {d_plain} -> {d_sub}: counting segments, not parcels")
