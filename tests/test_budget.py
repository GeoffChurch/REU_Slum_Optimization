from typing import cast

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union

from reblock.budget import (
    _rnd,
    access_burden,
    road_drainage,
    street_first_ordered,
)
from reblock.buildings import ANCHOR_COL, SpacingDiscs
from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, ParcelAdjacency
from reblock.methods.clearance import ClearanceReblocker
from reblock.methods.substrates import ChordSubstrate
from reblock.permeability import (
    DEFAULT_ROAD_WIDTH_M,
    EgressContext,
    permeability,
    with_width,
)
from tests.block_fixtures import no_buildings
from tests.permeability_fixtures import SHIPPED

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
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets,
                 source_content_hash=None, building_geometries=no_buildings(UTM),
                 building_tier=SpacingDiscs)


def test_access_burden_is_sum_of_squared_depths() -> None:
    assert access_burden(pd.Series([1, 2, 3])) == 1 + 4 + 9


def test_road_drainage_trunks_exceed_leaves() -> None:
    # clearance's roads on a 5x5 grid: a segment near the street carries more parcels than a leaf.
    # depth_target=1 (not the default 2): the grid's only depth>2 parcel is the single center
    # cell, so the default target is satisfied by ONE road (no trunk/leaf branching to measure);
    # depth_target=1 forces every parcel to the street, producing a genuine branching tree.
    block = _grid_block(5)
    roads = ClearanceReblocker(depth_target=1, substrate=ChordSubstrate(), repulsion=0.0,
                               max_roads=400,
                               road_width_m=DEFAULT_ROAD_WIDTH_M).propose(block).roads
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
    roads = ClearanceReblocker(substrate=ChordSubstrate(), repulsion=0.0, depth_target=2,
                               max_roads=400,
                               road_width_m=DEFAULT_ROAD_WIDTH_M).propose(block).roads
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
    block = Block(block_id="deep", crs=UTM, boundary=boundary, parcels=parcels, streets=streets,
                  source_content_hash=None, building_geometries=no_buildings(UTM),
                  building_tier=SpacingDiscs)
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
    roads = ClearanceReblocker(substrate=ChordSubstrate(), repulsion=0.0, depth_target=2,
                               max_roads=400,
                               road_width_m=DEFAULT_ROAD_WIDTH_M).propose(block).roads
    assert roads is not None
    _, d_roads = network_efficiency(block, roads)
    chord = with_width(gpd.GeoDataFrame(geometry=[LineString([(2.5, 0.0), (2.5, 5.0)])], crs=UTM),
                       DEFAULT_ROAD_WIDTH_M)
    _, d_chord = network_efficiency(block, chord)
    assert 0.0 <= d_roads <= 1.0
    assert 0.0 <= d_chord <= 1.0


def test_spacing_discs_are_half_nearest_neighbor():
    import geopandas as gpd
    from shapely.geometry import Point


    # three collinear points 10 m apart -> NN dist 10 for the ends, 10 for the middle -> r = 5
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(10, 0), Point(30, 0)], crs="EPSG:32734")
    r = SpacingDiscs(pts).radii
    assert list(r) == [5.0, 5.0, 10.0]      # 3rd point's NN is the 2nd, 20 m away -> r = 10


def test_spacing_discs_fallback_when_fewer_than_two_points():
    import geopandas as gpd
    from shapely.geometry import Point


    pts = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:32734")
    assert list(SpacingDiscs(pts).radii) == [3.0]     # fallback = corridor_m


def _band_fraction(y0: float, r: float, lo: float, hi: float) -> float:
    """EXACT share of a disc (centre height y0, radius r) lying in the band lo <= y <= hi.

    The area of a disc below a line at offset u from its centre is
    r^2 arccos(-u/r) + u sqrt(r^2 - u^2), so a band is the difference of two of those.
    """
    import math

    def below(u: float) -> float:
        u = max(-r, min(r, u))
        return r * r * math.acos(-u / r) + u * math.sqrt(r * r - u * u)
    return (below(hi - y0) - below(lo - y0)) / (math.pi * r * r)


def test_displacement_is_the_share_of_each_disc_the_corridor_covers() -> None:
    """Displacement is overlap, checked against GEOMETRY rather than against the formula's own
    earlier output: a disc cut by a straight band has an exact analytic area.

    A long road along y=0, 2 m wide, is the band |y| <= 1. Three discs of radius 4 centred at
    y = 1, 3, 10. The first sits with its CENTRE on the band's edge; the retired
    `clip(1 - d/r)` scored it c = 1 there (d = 0), where the true share is 0.3045. Over all three
    that formula summed to 1.5 against a true 0.50 -- the over-count, in miniature.
    """
    import numpy as np

    from reblock.budget import displacement, road_corridor
    from reblock.buildings import Discs
    crs = "EPSG:32734"
    roads = with_width(gpd.GeoDataFrame(geometry=[LineString([(-500, 0), (500, 0)])], crs=crs),
                       2.0)
    ys = [1.0, 3.0, 10.0]
    b = Discs(gpd.GeoDataFrame(geometry=[Point(0, y) for y in ys], crs=crs), np.full(3, 4.0))
    c = b.displacement(road_corridor(roads))
    exact = [_band_fraction(y, 4.0, -1.0, 1.0) for y in ys]
    # the disc outline is a 64-gon (shapely's default buffer), inscribed, so a partial slice is off
    # by a few parts in 1e3 -- the only reason this is not exact
    np.testing.assert_allclose(c, exact, rtol=5e-3, atol=1e-9)
    assert abs(exact[0] - 0.3045) < 1e-4 and abs(sum(exact) - 0.5) < 1e-3
    assert c[2] == 0.0                                       # never touched: exactly zero
    assert abs(displacement(b, roads) - float(c.sum())) < 1e-12


def test_a_footprint_half_covered_is_half_displaced() -> None:
    """Real outlines, no approximation anywhere: a 4 x 4 square whose lower half the corridor
    covers exactly is displaced exactly 0.5 -- a corner clip costs a corner, not a home."""
    from reblock.budget import road_corridor
    from reblock.buildings import Footprints
    crs = "EPSG:32734"
    sq = Polygon([(0, -2), (4, -2), (4, 2), (0, 2)])
    fp = Footprints(gpd.GeoDataFrame({ANCHOR_COL: [sq.centroid]}, geometry=[sq], crs=crs))
    # along y = -1, 2 m wide: covers y in [-2, 0], exactly the square's lower half
    road = with_width(gpd.GeoDataFrame(geometry=[LineString([(-50, -1), (50, -1)])], crs=crs),
                      2.0)
    assert abs(float(fp.displacement(road_corridor(road))[0]) - 0.5) < 1e-12


def test_displacement_zero_without_roads_or_buildings() -> None:
    import numpy as np

    from reblock.budget import displacement
    from reblock.buildings import Discs
    crs = "EPSG:32734"
    empty = gpd.GeoDataFrame(geometry=[], crs=crs)
    one = Discs(gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=crs), np.array([3.0]))
    assert displacement(one, empty) == 0.0
    assert displacement(Discs(empty, np.zeros(0)), empty) == 0.0


def test_displacement_charges_a_shared_building_once_under_overlapping_corridors() -> None:
    """A building two roads both cover is displaced by their UNION: charged once, never once per
    road. So scoring the pair must come out strictly below scoring each road alone and adding --
    which is exactly the double-count a per-road sum would commit."""
    import numpy as np

    from reblock.budget import displacement
    from reblock.buildings import Discs
    crs = "EPSG:32734"
    road_a = LineString([(0.0, 0.0), (5.0, 0.0)])
    road_b = LineString([(4.0, 0.0), (10.0, 0.0)])          # overlaps road_a's corridor, x 4-5
    both = with_width(gpd.GeoDataFrame(geometry=[road_a, road_b], crs=crs), 2.0)
    only_a = with_width(gpd.GeoDataFrame(geometry=[road_a], crs=crs), 2.0)
    only_b = with_width(gpd.GeoDataFrame(geometry=[road_b], crs=crs), 2.0)
    shared = Discs(gpd.GeoDataFrame(geometry=[Point(4.5, 0.0)], crs=crs), np.array([1.5]))
    union = displacement(shared, both)
    assert 0.0 < union <= 1.0
    assert union < displacement(shared, only_a) + displacement(shared, only_b)


def test_a_building_without_extent_is_displaced_iff_the_corridor_covers_its_point() -> None:
    """r = 0 (coincident points) has no area to take a share of, so the convention stands: c = 1
    iff the corridor covers the point, else 0. Pinned per building, because a 0-or-1 term moves a
    total either way and the sum would hide which way it went."""
    import numpy as np

    from reblock.budget import road_corridor
    from reblock.buildings import Discs
    crs = "EPSG:32734"
    road = with_width(gpd.GeoDataFrame(geometry=[LineString([(-50, 0), (50, 0)])], crs=crs), 2.0)
    b = Discs(gpd.GeoDataFrame(geometry=[Point(0, 0), Point(0, 5), Point(0, 0.5)], crs=crs),
              np.array([0.0, 0.0, 3.0]))
    c = b.displacement(road_corridor(road))
    assert c[0] == 1.0 and c[1] == 0.0                       # on the corridor / off it, no radius
    assert 0.0 < c[2] < 1.0                                  # an ordinary disc, partly covered


def test_repulsion_is_positive_even_far_from_all_buildings() -> None:
    from reblock.budget import displacement, repulsion
    crs = "EPSG:32734"
    b = SpacingDiscs(gpd.GeoDataFrame(geometry=[Point(0, 0), Point(0, 5), Point(5, 0)], crs=crs))
    far_road = LineString([(1000.0, 1000.0), (1000.0, 1010.0)])   # nowhere near any building
    # the quadratic tail r^2/(r^2+d^2) never reaches zero -> repulsion stays strictly positive even
    # for a road far from every building (the key non-degeneracy property)...
    assert repulsion(b, far_road) > 0.0
    # ... whereas displacement is exactly 0 for a road that touches no building -- the degeneracy
    # repulsion is designed to avoid.
    far_roads = with_width(gpd.GeoDataFrame(geometry=[far_road], crs=crs), DEFAULT_ROAD_WIDTH_M)
    assert displacement(b, far_roads) == 0.0


def test_repulsion_higher_for_a_road_closer_to_buildings() -> None:
    from reblock.budget import repulsion
    crs = "EPSG:32734"
    b = SpacingDiscs(gpd.GeoDataFrame(geometry=[Point(0, 0), Point(0, 10), Point(0, 20)],
                                      crs=crs))
    near = LineString([(2.0, 0.0), (2.0, 20.0)])      # 2 m from the building column
    far = LineString([(50.0, 0.0), (50.0, 20.0)])     # 50 m away
    assert repulsion(b, near) > repulsion(b, far) > 0.0


def test_repulsion_is_measured_from_centres_whatever_the_tier() -> None:
    """Measured from a polygon, d becomes an EDGE distance and repulsion silently changes meaning
    the moment a block moves to footprints. From the centre, a footprint and the disc sharing its
    centre and equivalent radius must cost exactly the same. Watched failing with repulsion
    measuring from `building_geometries` again."""
    from reblock.budget import repulsion
    from reblock.buildings import Discs, Footprints
    crs = "EPSG:32734"
    squares = [Polygon([(x, 0), (x + 4, 0), (x + 4, 4), (x, 4)]) for x in (0.0, 10.0)]
    fp = Footprints(gpd.GeoDataFrame({ANCHOR_COL: [s.centroid for s in squares]},
                                     geometry=squares, crs=crs))
    discs = Discs(gpd.GeoDataFrame(geometry=[Point(float(x), float(y)) for x, y in fp.xy],
                                   crs=crs), fp.radii)
    road = LineString([(-5, 6), (20, 6)])
    assert abs(repulsion(fp, road) - repulsion(discs, road)) < 1e-12


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
                 streets=streets, building_geometries=points,
                 source_content_hash=None, building_tier=SpacingDiscs)
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
    block = Block(block_id="deep_col", crs=UTM, boundary=boundary, parcels=parcels, streets=streets,
                  source_content_hash=None, building_geometries=no_buildings(UTM),
                  building_tier=SpacingDiscs)
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
                 streets=streets,
                 source_content_hash=None, building_geometries=no_buildings(UTM),
                 building_tier=SpacingDiscs)
    roads = gpd.GeoDataFrame(geometry=[
        LineString([(15, 0), (15, 135)]),           # the drainage trunk (spur)
        LineString([(0, 115), (30, 115)]),          # cross-connector, grounded only via the spur
    ], crs=UTM)
    return block, with_width(roads, DEFAULT_ROAD_WIDTH_M)


def test_max_access_depth_matches_the_peel() -> None:
    from reblock.budget import max_access_depth
    block, roads = _deep_column_block_with_two_roads()
    adjacency = ParcelAdjacency.of(block, STREET_TOL)
    assert max_access_depth(adjacency, gpd.GeoDataFrame(geometry=[], crs=UTM)) == 4   # no roads
    assert max_access_depth(adjacency, roads) == 1                                     # both roads


def test_prefix_to_depth_returns_minimal_prefix_that_reaches_target() -> None:
    from reblock.budget import prefix_to_depth
    block, roads = _deep_column_block_with_two_roads()
    prefix, reached = prefix_to_depth(ParcelAdjacency.of(block, STREET_TOL), roads, 2)
    assert reached == 2                        # road A alone brings max depth to 2
    assert len(prefix) == 1                    # the MINIMAL prefix, not both roads
    assert prefix.geometry.iloc[0].equals(roads.geometry.iloc[0])   # road A (the drainage trunk)


def test_prefix_to_depth_reaches_a_deeper_target_only_with_all_roads() -> None:
    from reblock.budget import prefix_to_depth
    block, roads = _deep_column_block_with_two_roads()
    prefix, reached = prefix_to_depth(ParcelAdjacency.of(block, STREET_TOL), roads, 1)
    assert reached == 1
    assert len(prefix) == 2                    # needs both roads to reach depth 1


def test_prefix_to_depth_reports_floor_when_target_unreachable() -> None:
    from reblock.budget import prefix_to_depth
    block, roads = _deep_column_block_with_two_roads()
    # depth 0 is impossible (min is 1)
    prefix, reached = prefix_to_depth(ParcelAdjacency.of(block, STREET_TOL), roads, 0)
    assert reached == 1                        # the floor depth (> target), reported honestly
    assert len(prefix) == len(roads)           # best effort = all roads in drainage order


def test_prefix_to_permeability_returns_minimal_prefix_that_reaches_target() -> None:
    from reblock.budget import prefix_to_permeability
    block, roads = _permeability_grid_block_and_roads()
    ctx = EgressContext.of(block, SHIPPED)
    p1 = permeability(ctx, cast(gpd.GeoDataFrame, roads.iloc[:1]))
    p2 = permeability(ctx, roads)
    assert 0.0 < p1 < p2                       # spur alone helps but the pair does strictly more
    prefix, reached = prefix_to_permeability(ctx, roads, p1)
    assert reached
    assert len(prefix) == 1                    # the MINIMAL prefix, not both roads
    assert prefix.geometry.iloc[0].equals(roads.geometry.iloc[0])   # the spur (the drainage trunk)


def test_prefix_to_permeability_reaches_a_higher_target_only_with_all_roads() -> None:
    from reblock.budget import prefix_to_permeability
    block, roads = _permeability_grid_block_and_roads()
    ctx = EgressContext.of(block, SHIPPED)
    p1 = permeability(ctx, cast(gpd.GeoDataFrame, roads.iloc[:1]))
    p2 = permeability(ctx, roads)
    target = (p1 + p2) / 2.0                   # strictly between: needs both roads
    prefix, reached = prefix_to_permeability(ctx, roads, target)
    assert reached
    assert len(prefix) == 2


def test_prefix_to_permeability_reports_unreached_when_target_unreachable() -> None:
    from reblock.budget import prefix_to_permeability
    block, roads = _permeability_grid_block_and_roads()
    ctx = EgressContext.of(block, SHIPPED)
    prefix, reached = prefix_to_permeability(ctx, roads, 1.5)   # 1.5 exceeds max (permeability<1)
    assert not reached
    assert len(prefix) == len(roads)           # best effort = all roads in drainage order
    assert prefix.geometry.iloc[0].equals(roads.geometry.iloc[0])
    assert prefix.geometry.iloc[1].equals(roads.geometry.iloc[1])


def test_prefix_to_permeability_empty_roads_returns_empty_unreached() -> None:
    from reblock.budget import prefix_to_permeability
    block, _roads = _permeability_grid_block_and_roads()
    empty_roads = gpd.GeoDataFrame(geometry=[], crs=UTM)
    prefix, reached = prefix_to_permeability(EgressContext.of(block, SHIPPED),
                                             empty_roads, 0.1)
    assert len(prefix) == 0
    assert not reached


def test_displacement_curve_is_monotonic_and_ends_at_full():

    from reblock.budget import displacement, displacement_curve
    block, roads = _straight_block_with_two_roads()
    curve = displacement_curve(block, roads)
    n = len(block.building_geometries)
    assert curve.cost[0] == 0.0 and curve.benefit[0] == 0.0
    assert curve.benefit == sorted(curve.benefit)     # non-decreasing displacement
    assert abs(curve.benefit[-1]
               - displacement(block.buildings, roads) / n) < 1e-6
    # cost axis = cumulative added road length in METRES, non-decreasing, ending at the full
    # road length -- a `_sweep` property formerly pinned only by the retired
    # test_cost_axis_is_cumulative_road_length_metres (via cost_benefit_curve); migrated here
    # onto displacement_curve (the lightest live _sweep vehicle) to keep it covered.
    assert curve.cost == sorted(curve.cost)
    assert abs(curve.cost[-1] - float(roads.geometry.length.sum())) < 1e-6


def test_displacement_curve_is_home_fraction() -> None:
    from reblock.budget import displacement, displacement_curve
    block, roads = _straight_block_with_two_roads()   # existing helper with building_geometries
    curve = displacement_curve(block, roads)
    n = len(block.building_geometries)
    assert all(0.0 <= b <= 1.0 for b in curve.benefit)          # fraction, not a count
    # terminal fraction == displacement(full roads)/n_buildings
    assert abs(curve.benefit[-1]
               - displacement(block.buildings, roads) / n) < 1e-9


def test_prefix_to_displacement_returns_minimal_prefix_that_reaches_fraction() -> None:

    from reblock.budget import displacement, prefix_to_displacement
    block, roads = _straight_block_with_two_roads()
    n = len(block.building_geometries)
    frac1 = displacement(block.buildings, cast(gpd.GeoDataFrame, roads.iloc[:1])) / n
    frac2 = displacement(block.buildings, roads) / n
    assert 0.0 < frac1 < frac2                  # road 0 alone displaces only its own building
    prefix = prefix_to_displacement(block, roads, frac1)
    assert len(prefix) == 1                     # the MINIMAL prefix, not both roads
    assert prefix.geometry.iloc[0].equals(roads.geometry.iloc[0])


def test_prefix_to_displacement_needs_all_roads_for_a_higher_fraction() -> None:

    from reblock.budget import displacement, prefix_to_displacement
    block, roads = _straight_block_with_two_roads()
    n = len(block.building_geometries)
    frac1 = displacement(block.buildings, cast(gpd.GeoDataFrame, roads.iloc[:1])) / n
    frac2 = displacement(block.buildings, roads) / n
    target = (frac1 + frac2) / 2.0              # strictly between: needs both roads
    prefix = prefix_to_displacement(block, roads, target)
    assert len(prefix) == 2


def test_prefix_to_displacement_returns_all_roads_when_fraction_unreachable() -> None:

    from reblock.budget import prefix_to_displacement
    block, roads = _straight_block_with_two_roads()
    prefix = prefix_to_displacement(block, roads, 1.5)   # > 1.0, impossible
    assert len(prefix) == len(roads)            # best effort = all roads in drainage order


def test_prefix_to_displacement_empty_roads_returns_empty() -> None:

    from reblock.budget import prefix_to_displacement
    block, _roads = _straight_block_with_two_roads()
    empty_roads = gpd.GeoDataFrame(geometry=[], crs=UTM)
    prefix = prefix_to_displacement(block, empty_roads, 0.5)
    assert len(prefix) == 0


def test_permeability_and_displacement_curves_share_cost_samples():
    # emit.compare_report pairs (displacement[i], permeability[i]) on the plotted frontier, which
    # is only valid if both curves are INDEX-ALIGNED: same drainage-ordered _sweep, same
    # n_points=20, over the same roads, so their `.cost` samples (cumulative added road length, m,
    # for both) land at identical budgets. A future change making _sweep's sampling
    # value-dependent would silently misalign every plot.
    from reblock.budget import displacement_curve
    from reblock.permeability import permeability_curve
    block, roads = _straight_block_with_two_roads()
    perm = permeability_curve(EgressContext.of(block, SHIPPED), roads)
    disp = displacement_curve(block, roads)
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
        building_geometries=gpd.GeoDataFrame(
            geometry=[Point(x, 25) for x in xs], crs=UTM),
        source_content_hash=None, building_tier=SpacingDiscs)

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
        building_geometries=gpd.GeoDataFrame(
            geometry=[Point(25, 25), Point(85, 25), Point(75, 25)], crs=UTM),
        source_content_hash=None, building_tier=SpacingDiscs)

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
            building_geometries=gpd.GeoDataFrame(geometry=[Point(x + 5, y + 5)], crs=UTM),
            source_content_hash=None, building_tier=SpacingDiscs)

    plain = LineString([(20.0, 0.0), (20.0, 40.0)])
    subdivided = LineString([(20.0, 0.0), (20.0, 10.0), (20.0, 20.0), (20.0, 30.0), (20.0, 40.0)])
    d_plain = road_drainage(block_with(plain, (20.0, 40.0)),
                            gpd.GeoDataFrame(geometry=[plain], crs=UTM))
    d_sub = road_drainage(block_with(subdivided, (20.0, 40.0)),
                          gpd.GeoDataFrame(geometry=[subdivided], crs=UTM))
    assert d_plain == [1], f"one parcel on one road must read 1, got {d_plain}"
    assert d_sub == d_plain, (
        f"subdividing a road changed drainage {d_plain} -> {d_sub}: counting segments, not parcels")


def _crossing_block() -> tuple[Block, gpd.GeoDataFrame]:
    """A stem up from the street, crossed mid-segment by a crossbar every parcel fronts.

    The crossing at (50, 40) is a vertex of NEITHER road, which is the whole point: it is a real
    junction that a graph keyed on segment endpoints cannot see.
    """
    xs = [20.0, 35.0, 65.0, 80.0]          # 50.0 deliberately absent -- no vertex at the crossing
    stem = LineString([(50.0, 0.0), (50.0, 80.0)])
    crossbar = LineString([(10.0, 40.0), *[(x, 40.0) for x in xs], (90.0, 40.0)])
    # Parcels sit ON the crossbar and nowhere near the stem or the street; `road_drainage` attaches
    # them to graph NODES, hence the per-parcel vertices above.
    parcels = gpd.GeoDataFrame(
        {"parcel_id": [str(k) for k in range(len(xs))]},
        geometry=[Polygon([(x - 5, 40), (x + 5, 40), (x + 5, 50), (x - 5, 50)]) for x in xs],
        crs=UTM)
    block = Block(
        block_id="cross", crs=UTM,
        boundary=Polygon([(0, 0), (100, 0), (100, 80), (0, 80)]),
        parcels=parcels,
        streets=gpd.GeoDataFrame(geometry=[LineString([(0.0, 0.0), (100.0, 0.0)])], crs=UTM),
        building_geometries=gpd.GeoDataFrame(geometry=[Point(x, 45) for x in xs], crs=UTM),
        source_content_hash=None, building_tier=SpacingDiscs)
    return block, _roads([stem, crossbar])


def test_road_drainage_routes_through_an_unnoded_crossing() -> None:
    """A crossing is a junction even when neither road carries a vertex there.

    Two roads crossing mid-segment share no coordinate, so a graph keyed on raw segment endpoints
    puts them in separate components. The crossbar -- which every parcel actually walks out along --
    then reads as floating and scores 0, and the prefix order drainage drives is ranking roads by an
    artifact rather than by traffic.

    FAULT INJECTION: keying nodes on raw segment endpoints without splitting at crossings (the
    pre-2026-09-14 `_road_net`) leaves the crossbar street-disconnected and this reads [0, 0].
    """
    block, roads = _crossing_block()
    stem, crossbar = roads.geometry
    assert not {(50.0, 40.0)} & set(stem.coords) | {(50.0, 40.0)} & set(crossbar.coords), (
        "fixture is vacuous: the crossing must not be a vertex of either road")

    drain = road_drainage(block, roads)
    assert drain == [4, 4], (
        f"every parcel should route crossbar -> crossing -> stem -> street, got {drain}")


def test_network_efficiency_ignores_whether_a_crossing_is_drawn_as_a_vertex() -> None:
    """Two roads crossing in an X are one network, however the geometry happens to be vertexed.

    The same X is built twice -- once as two bare 2-point lines that cross at (50, 50), once with
    that point spelled as an explicit vertex of each. The geometry is identical; only the
    coordinate lists differ. `_build_csr` explodes roads to `_rnd`-snapped endpoint pairs and never
    splits at crossings, so the bare version routes as two disconnected arms and understates
    door-to-door travel between them.

    Same defect as `test_road_drainage_routes_through_an_unnoded_crossing`, in the other graph
    builder (`_explode_segments` -> `_build_csr`), reached through `network_efficiency`.

    FAULT INJECTION: dropping the crossing vertex from `noded` below makes the two arms equal by
    construction and the test passes vacuously -- so the assertion that they DIFFER today is what
    proves the fixture bites.
    """
    from reblock.budget import network_efficiency

    polys = [Polygon([(x, y), (x + 20, y), (x + 20, y + 20), (x, y + 20)])
             for x, y in [(0, 40), (80, 40), (40, 0), (40, 80)]]   # one parcel at each arm tip
    parcels = gpd.GeoDataFrame({"parcel_id": [str(k) for k in range(4)]}, geometry=polys, crs=UTM)
    block = Block(
        block_id="x", crs=UTM,
        boundary=Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
        parcels=parcels,
        streets=gpd.GeoDataFrame(geometry=[LineString([(0.0, 0.0), (100.0, 0.0)])], crs=UTM),
        building_geometries=gpd.GeoDataFrame(
            geometry=[Point(10, 50), Point(90, 50), Point(50, 10), Point(50, 90)], crs=UTM),
        source_content_hash=None, building_tier=SpacingDiscs)

    bare = _roads([LineString([(0.0, 50.0), (100.0, 50.0)]),
                   LineString([(50.0, 0.0), (50.0, 100.0)])])
    noded = _roads([LineString([(0.0, 50.0), (50.0, 50.0), (100.0, 50.0)]),
                    LineString([(50.0, 0.0), (50.0, 50.0), (50.0, 100.0)])])
    assert bare.geometry[0].equals(noded.geometry[0]), "the two spellings must be the same geometry"
    assert bare.geometry[1].equals(noded.geometry[1]), "the two spellings must be the same geometry"

    e_bare, d_bare = network_efficiency(block, bare)
    e_noded, d_noded = network_efficiency(block, noded)
    assert (e_bare, d_bare) == pytest.approx((e_noded, d_noded)), (
        f"the crossing is a junction either way: bare {e_bare:.6f}/{d_bare:.6f} vs "
        f"noded {e_noded:.6f}/{d_noded:.6f}")


def test_every_prefix_is_connected_by_the_same_test_the_peel_uses() -> None:
    """`street_first_ordered` and `street_connectivity` must agree on what reaches the street.

    `_road_net` decided street-adjacency on `_rnd`-ROUNDED nodes while `street_connectivity` -- the
    predicate the peel, access depth and burden actually use -- decides on RAW segments. A road
    sitting within a rounding step of `STREET_TOL` is accepted by one and refused by the other, so
    the ordering leads with a road the peel scores as floating: a prefix that grants no access while
    the lens reports it as buildable.

    Not hypothetical. `euclidean_grid` trims to `street_buffer: 0.5`, exactly `STREET_TOL`, so on
    the pinned block six of nine grid roads land within 5e-10 m of the boundary and the two
    predicates disagree on three of them.

    The fixture reproduces that deliberately: `false_front` lies at y = 0.504, which `_rnd` snaps to
    0.500 (accepted, 0.500 <= 0.500) while its true distance 0.504 is refused. `real_front` is
    unambiguously connected at y = 0.2, and the two cross so the ordering CAN reach the first
    through the second.

    FAULT INJECTION: restoring the rounded-node predicate makes `false_front` a street node, so it
    leads the order and prefix[:1] reads connected_frac 0.000.
    """
    from reblock.derive.access import street_connectivity

    xs = [20.0, 35.0, 65.0, 80.0]
    false_front = LineString([(10.0, 0.504), *[(x, 0.504) for x in xs], (90.0, 0.504)])
    real_front = LineString([(50.0, 0.2), (50.0, 40.0)])
    parcels = gpd.GeoDataFrame(
        {"parcel_id": [str(k) for k in range(len(xs))]},
        geometry=[Polygon([(x - 5, 0.6), (x + 5, 0.6), (x + 5, 10), (x - 5, 10)]) for x in xs],
        crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0.0, 0.0), (100.0, 0.0)])], crs=UTM)
    block = Block(
        block_id="knife", crs=UTM,
        boundary=Polygon([(0, 0), (100, 0), (100, 40), (0, 40)]),
        parcels=parcels, streets=streets,
        building_geometries=gpd.GeoDataFrame(geometry=[Point(x, 5) for x in xs], crs=UTM),
        source_content_hash=None, building_tier=SpacingDiscs)

    street_geom = unary_union(list(streets.geometry))
    raw = false_front.distance(street_geom)
    rounded = Point(_rnd((10.0, 0.504))).distance(street_geom)
    assert raw > STREET_TOL >= rounded, (
        f"fixture is vacuous: it must straddle the tolerance, got raw {raw} vs rounded {rounded}")

    roads = _roads([false_front, real_front])
    ordered = street_first_ordered(block, roads, STREET_TOL)
    for k in range(1, len(ordered) + 1):
        prefix = cast(gpd.GeoDataFrame, ordered.iloc[:k])
        frac = street_connectivity(block.streets, prefix, STREET_TOL).connected_frac
        assert frac == 1.0, (
            f"prefix[:{k}] ({prefix.geometry.length.sum():.1f} m) is {frac:.3f} street-connected "
            f"by the peel's own test -- the lens would score road nobody could build")


@pytest.mark.parametrize("tier", ["discs", "footprints"])
def test_incremental_overlap_is_the_full_recompute_one_road_at_a_time(tier: str) -> None:
    """`greedy_arterial` and `cycle_native` price every candidate with `IncrementalOverlap.delta`
    instead of recomputing `displacement` over the whole grown network -- measured 11-210x faster
    on a 6,619-building block -- so the two must be the SAME number. Grown one road at a time,
    through roads that cross, run parallel into a building from both sides, and repeat exactly,
    because union-by-area is where an incremental rule goes wrong: a slice covered twice must count
    once, and two different slices of one building must both count.

    FAULT INJECTION: `_pieces` returning `new_part` without the union with `old` (a later piece
    overwriting an earlier one) fails this at the second road."""
    import numpy as np

    from reblock.budget import displacement
    from reblock.buildings import Discs, Footprints, IncrementalOverlap
    crs = "EPSG:32734"
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 60, size=(120, 2))
    pts = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in xy], crs=crs)
    b: Discs | Footprints = (
        Discs(pts, rng.uniform(1.0, 3.0, len(xy))) if tier == "discs" else
        Footprints(gpd.GeoDataFrame(
            {ANCHOR_COL: pts.geometry.to_numpy()},
            geometry=[Point(x, y).buffer(1.5, cap_style="square").union(
                Point(x + 1.2, y + 1.2).buffer(0.9)) for x, y in xy], crs=crs)))
    lines = [LineString([(0, 30), (60, 30)]), LineString([(30, 0), (30, 60)]),
             LineString([(0, 33), (60, 33)]), LineString([(0, 30), (60, 30)]),
             LineString([(5, 5), (55, 50)]), LineString([(10, 0), (10, 60)])]
    tracker = IncrementalOverlap(b)
    for k, line in enumerate(lines):
        roads = with_width(gpd.GeoDataFrame(geometry=lines[:k + 1], crs=crs), 4.0)
        full = displacement(b, roads)
        before = tracker.total()
        delta = tracker.delta(line.buffer(2.0))
        assert delta == pytest.approx(full - before, abs=1e-9), f"road {k}: delta"
        tracker.add(line.buffer(2.0))
        assert tracker.total() == pytest.approx(full, abs=1e-9), f"road {k}: total"
    assert displacement(b, roads) > 0.0
