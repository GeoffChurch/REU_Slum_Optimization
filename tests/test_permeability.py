import math
from dataclasses import replace

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon

from reblock.contracts import Block
from reblock.permeability import (
    DEFAULT_ROAD_WIDTH_M,
    PermeabilityParams,
    _footpath_conductance,
    edge_conductances,
    egress_power,
    lane_width,
    parcel_radii,
    permeability,
    permeability_curve,
    road_conductance,
    with_width,
)

UTM = CRS.from_epsg(32734)

def _grid_block(k=4, cell=1.0):
    # k x k `cell`-sized parcels tiling a k*cell x k*cell square; south edge (y=0) is the
    # street. `cell` defaults to 1.0 (unit parcels, unchanged from the brief) for every test
    # except test_monotone_under_added_roads and test_loop_beats_spur_at_equal_length, which
    # need a larger cell so the default corridor_m=3.0 is thin relative to parcel spacing (see
    # those tests' comments).
    polys, ids = [], []
    for r in range(k):
        for c in range(k):
            x0, x1, y0, y1 = c*cell, (c+1)*cell, r*cell, (r+1)*cell
            polys.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
            ids.append(r*k+c)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (k*cell, 0)])], crs=UTM)
    boundary = Polygon([(0, 0), (k*cell, 0), (k*cell, k*cell), (0, k*cell)])
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)

def _roads(lines): return with_width(gpd.GeoDataFrame(geometry=lines, crs=UTM),
                                     DEFAULT_ROAD_WIDTH_M)

def test_no_roads_permeability_is_zero():
    b = _grid_block()
    assert permeability(b, None) == 0.0 and permeability(b, _roads([])) == 0.0

def test_permeability_in_unit_interval_and_positive_with_a_road():
    b = _grid_block()
    p = permeability(b, _roads([LineString([(2, 0), (2, 4)])]))   # a spine road to the interior
    assert 0.0 < p < 1.0

def test_monotone_under_added_roads():
    # Same corridor-saturation trap as test_loop_beats_spur_at_equal_length: on the 6x6
    # 1m-unit grid, corridor_m=3.0 blankets 100% (60/60) of footpath edges for BOTH r1 and its
    # superset r2, so the added road has zero marginal coverage and the old `>=` assertion was
    # an exact tie -- passing vacuously, not exercising monotonicity at all. Reuse the
    # thin-corridor 15x15/10m grid from the loop-vs-spur test instead, and pick a superset road
    # pair whose second road genuinely adds coverage: r1 = the same spur (x=15, depth 135);
    # r2 = r1 plus a short cross-road near its top (y=115, x=0..30) that upgrades additional
    # footpath edges neither road alone covers. Coverage: r1 42/420=10.00%, r2 47/420=11.19%
    # (measured, not both 100%) -- confirms the added road has real marginal effect, so a
    # STRICT increase is the honest assertion (not just `>=`).
    b = _grid_block(15, cell=10.0)
    r1 = _roads([LineString([(15, 0), (15, 135)])])
    r2 = _roads([LineString([(15, 0), (15, 135)]), LineString([(0, 115), (30, 115)])])
    assert permeability(b, r2) > permeability(b, r1)   # adding a road with real coverage helps

def test_loop_beats_spur_at_equal_length():
    # A 15x15 grid of 10m parcels (225 parcels; centroid spacing 10m) -- large enough, at the
    # PermeabilityParams() default that a road's buffered corridor stays THIN
    # (covers a local band, not the whole grid): measured coverage is 10.0% of footpath edges
    # for spur, 8.6% for loop -- both well under 100% (see task-1-report.md for the full
    # measurement). (A 6x6 grid of 1m unit parcels -- this repo's earlier attempt -- fails this:
    # corridor_m=3.0 blankets ~all footpath edges of a 1m-spaced grid regardless of road shape,
    # so spur and loop tie to floating-point noise; see task-1-report.md.)
    #
    # spur: a single straight road up column x=15 from the street to depth 135 -- ONE egress
    # route the whole way; deep parcels' only fast path to ground.
    #
    # loop: the SAME total length (135 m), split as a short loop-closing connector near the
    # street (the road climbs to y=22, jogs over to the adjacent column x=5, and back down to
    # the street -- a closed cycle touching the street at x=15 AND x=5, i.e. TWO egress routes
    # for everything below y=22) followed by a single trunk continuing up to depth 103 (matching
    # spur's one-route-only regime beyond the loop). A naive SYMMETRIC two-arm fork (no trunk)
    # was tried first and consistently LOST to the spur once the grid extends past the fork's
    # reach (see task-1-report.md's search) -- total-power-over-ALL-parcels rewards a single
    # path's raw reach enough to usually beat an equal-length fork. What robustly wins is adding
    # a SHORT redundant loop at the base (where every deep parcel's current must funnel through
    # regardless of depth) and spending the rest of the budget on reach, exactly like this
    # repo's own LoopClosureRefiner (src/reblock/methods/loop_closure.py) does post-hoc.
    #
    # Uses an explicit g_walk=1.0 (this test's ORIGINAL calibration level, pre-2026-07-23 metric
    # change): this is a topological property (loop redundancy beats single-arm reach at equal
    # road length) that empirically FLIPS at the new production default g_walk=0.1/g_road_per_m=8 (a
    # 200x ratio makes REACH dominate REDUNDANCY on this single-arm grid -- verified directly, not
    # assumed). That is not a regression in the r0-corridor formula itself: this fixture has no
    # `building_points`, so r0=0 and `_footpath_conductance` reduces exactly to the pre-change
    # g_walk/dist baseline (see `_adaptive_r0`'s docstring) -- it is the production g_walk LEVEL
    # this task lowers, independent of the r0-corridor change, that this particular topological
    # demonstration is sensitive to. Pin the level explicitly so this test keeps demonstrating the
    # principle it is named for, independent of whatever g_walk the production default is tuned to.
    b = _grid_block(15, cell=10.0)
    spur = _roads([LineString([(15, 0), (15, 135)])])
    loop = _roads([LineString([(15, 0), (15, 103)]),
                    LineString([(15, 22), (5, 22), (5, 0)])])
    assert spur.geometry.length.sum() == loop.geometry.length.sum() == 135.0
    params = PermeabilityParams(g_walk=1.0)
    assert permeability(b, loop, params) > permeability(b, spur, params)

def test_ungrounded_returns_zero_benefit_or_guarded():
    b = _grid_block()
    b.streets.geometry = gpd.GeoSeries([], crs=UTM)   # no street -> no ground
    P, v = egress_power(b, None)
    assert not np.isfinite(P)     # +inf; permeability() guards this (returns nan) -- assert guard
    assert math.isnan(permeability(b, None))          # exercise the actual nan guard, not just P

def test_permeability_curve_terminal_matches_full_permeability():
    # Two segments (a spur + a cross-connector) on the thin-corridor 15x15/10m grid, so the
    # drainage sweep has real intermediate prefixes (see test_monotone_under_added_roads for why
    # this grid, not the 1m-unit default, is needed to avoid corridor saturation).
    b = _grid_block(15, cell=10.0)
    roads = _roads([LineString([(15, 0), (15, 135)]), LineString([(0, 115), (30, 115)])])
    curve = permeability_curve(b, roads, n_points=10)
    assert abs(curve.benefit[-1] - permeability(b, roads)) < 1e-9

def test_permeability_curve_starts_at_zero_and_is_bounded():
    b = _grid_block(15, cell=10.0)
    roads = _roads([LineString([(15, 0), (15, 135)]), LineString([(0, 115), (30, 115)])])
    curve = permeability_curve(b, roads, n_points=10)
    assert curve.cost[0] == 0.0 and curve.benefit[0] == 0.0
    assert all(0.0 <= v < 1.0 for v in curve.benefit)

def test_permeability_curve_is_monotone_non_decreasing():
    b = _grid_block(15, cell=10.0)
    roads = _roads([LineString([(15, 0), (15, 135)]), LineString([(0, 115), (30, 115)])])
    curve = permeability_curve(b, roads, n_points=10)
    assert curve.benefit == sorted(curve.benefit)

def test_permeability_curve_freezes_p0_matching_a_manual_baseline():
    # p0 is frozen once via egress_power(block, None, params)[0] -- every sample must equal
    # permeability(block, prefix, params, p0=p0_manual) computed against that SAME baseline.
    b = _grid_block(15, cell=10.0)
    roads = _roads([LineString([(15, 0), (15, 135)])])
    p0, _ = egress_power(b, None)
    curve = permeability_curve(b, roads, n_points=4)
    assert abs(curve.benefit[-1] - permeability(b, roads, p0=p0)) < 1e-9

def test_permeability_at_displacement_first_crossing_and_unreached():
    from reblock.budget import Curve
    from scripts.calibrate_permeability import permeability_at_displacement
    perm = Curve([0, 1, 2, 3], [0.0, 0.20, 0.35, 0.50])
    disp = Curve([0, 1, 2, 3], [0.0, 0.10, 0.25, 0.45])
    # first sample with disp >= 0.20 is i=2 (disp .25) -> perm .35
    assert permeability_at_displacement(perm, disp, 0.2) == 0.35
    assert permeability_at_displacement(perm, disp, 0.45) == 0.50
    assert permeability_at_displacement(perm, disp, 0.5) == float("-inf")

# --- clearance-fraction footpath conductance + per-parcel radii ---------------------------

def test_footpath_conductance_cramped_edge_lower_than_open_edge():
    # Two edges whose footprints sum to 6 m of the centroid line. dist=4m is CRAMPED (the disks
    # overlap, so the clearance is negative and it hits the eps floor); dist=40m is OPEN. Passing
    # r_sum = 6 on both reproduces the block-median r0=3 case exactly, so the hand-computed values
    # below are unchanged from the previous model -- the estimator generalises it rather than
    # replacing it. Hand-computed:
    #   shape = [max(0.02, (4-6)/4), max(0.02, (40-6)/40)] = [0.02, 0.85]
    #   median(shape) = 0.435; median(1/dist) = median([0.25, 0.025]) = 0.1375
    #   target_median = g_walk * median(1/dist) = 0.1 * 0.1375 = 0.01375
    #   scale = target_median / median(shape) = 0.01375 / 0.435 = 0.0316091...
    #   g = scale * shape = [0.000632184, 0.026868]
    dist = np.array([4.0, 40.0])
    g = _footpath_conductance(dist, np.array([6.0, 6.0]), g_walk=0.1)
    assert g[0] < g[1]                            # cramped strictly less permeable than open
    assert g[0] > 0.0                              # eps floor: never literally zero
    assert g[0] == pytest.approx(0.000632184, rel=1e-5)
    assert g[1] == pytest.approx(0.026868, rel=1e-5)

def test_footpath_conductance_empty_dist_returns_empty():
    assert _footpath_conductance(np.zeros(0), np.zeros(0), g_walk=0.1).shape == (0,)

def _points_block(n_points: int, spacing: float) -> Block:
    # A trivial valid Block whose PARCEL geometry is irrelevant -- only `building_points` (a row
    # of `n_points` points `spacing` apart, so every point's nearest-neighbour distance is exactly
    # `spacing`, including the two endpoints) matters for the radii.
    boundary = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    parcels = gpd.GeoDataFrame({"parcel_id": [0]}, geometry=[boundary], crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (10, 0)])], crs=UTM)
    pts = gpd.GeoDataFrame(geometry=[Point(float(i) * spacing, 0.0) for i in range(n_points)],
                           crs=UTM)
    block = Block(block_id="pts", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)
    return replace(block, building_points=pts)

def test_parcel_radii_are_PER_PARCEL_and_scale_with_local_spacing():
    """The point of the change: each parcel gets its OWN footprint radius, not a block median.

    The single-parcel fixture puts every point in one parcel, so containment resolves to that
    parcel and the radius is a real per-parcel quantity rather than an average. Radii scale 1:1 with
    point spacing, as `building_radii` (NN/2) must.

    FAULT INJECTION: return `np.full(n, radii.mean())` from `parcel_radii` and the dense/sparse
    ratio survives but `test_footpath_clearance_is_LOCAL_not_a_block_median` below fails.
    """
    params = PermeabilityParams()
    dense = parcel_radii(_points_block(6, spacing=4.0), params)
    sparse = parcel_radii(_points_block(6, spacing=12.0), params)
    assert dense.shape == (1,) and sparse.shape == (1,)
    assert sparse[0] == pytest.approx(3.0 * dense[0])   # NN scales 1:1 with point spacing


def test_footpath_clearance_is_LOCAL_not_a_block_median():
    """Two edges with the SAME length but different footprints must now differ.

    Under the old block-median r0 both edges got `1 - 2*r0/dist` with one r0, so they were
    identical. This is the whole content of the change.
    """
    dist = np.array([20.0, 20.0])
    g = _footpath_conductance(dist, np.array([2.0, 16.0]), g_walk=0.1)
    assert g[0] > g[1], "an edge between small footprints must beat one between large ones"
    # a block median would have assigned both the mean gap and returned them equal
    same = _footpath_conductance(dist, np.array([9.0, 9.0]), g_walk=0.1)
    assert same[0] == pytest.approx(same[1])

def test_parcel_radii_fall_back_to_zero_without_enough_building_points():
    params = PermeabilityParams()
    b = _grid_block()   # building_points defaults to empty
    assert not parcel_radii(b, params).any()

def test_a_road_upgrade_never_lowers_an_edges_conductance():
    # The monotonicity guarantee, checked where it is actually enforced: `edge_conductances` takes
    # `max(footpath, road)`, so covering an edge can only raise it. This replaced an explicit clamp
    # (every footpath edge capped at its own would-be upgrade) that the `max` makes unnecessary.
    #
    # The fixture is chosen so the OLD replace-outright rule would have FAILED it: the raw
    # r0-corridor shape asymptotes to a nonzero constant as dist -> infinity while a road's g/dist
    # decays, so on a long enough edge the footpath genuinely beats the road. Hand-verified with
    # r0=3 (2*r0=6), g_walk=0.1, dist=[1, 10, 1000]:
    #   shape = [max(0.02, 1-6/1), max(0.02, 1-6/10), max(0.02, 1-6/1000)] = [0.02, 0.4, 0.994]
    #   median(shape) = 0.4; median(1/dist) = 0.1; scale = (0.1 * 0.1) / 0.4 = 0.025
    #   raw = scale * shape = [0.0005, 0.01, 0.02485]; road_g = 20/dist = [20, 2, 0.02]
    #   at dist=1000: raw = 0.02485 > road_g = 0.02 -- the footpath EXCEEDS the road.
    params = PermeabilityParams()
    dist = np.array([1.0, 10.0, 1000.0])
    foot = _footpath_conductance(dist, np.full(dist.size, 6.0), g_walk=params.g_walk)
    road = road_conductance(params, np.full(3, lane_width(params, DEFAULT_ROAD_WIDTH_M)), dist)
    assert foot[2] > road[2]                 # the long edge: the footpath really does beat the road
    assert (foot[:2] < road[:2]).all()       # the shorter edges never come close

    # Four collinear parcels at x = 0, 1, 11, 1011 -- edge lengths exactly the [1, 10, 1000] above,
    # so the per-block normalization is the one hand-verified there -- all spanned by one long road.
    cent = np.array([0.0, 1.0, 11.0, 1011.0])
    rows, cols = np.array([0, 1, 2]), np.array([1, 2, 3])
    fp = _footpath_conductance(dist, np.full(dist.size, 6.0), g_walk=params.g_walk)
    roads = with_width(gpd.GeoDataFrame(
        geometry=[LineString([(0.0, 0.0), (1011.0, 0.0)])], crs=UTM), DEFAULT_ROAD_WIDTH_M)
    gf, gb = edge_conductances(cent, np.zeros(4), rows, cols, dist, fp, roads, params)

    assert (gf >= fp - 1e-12).all() and (gb >= fp - 1e-12).all()   # the guarantee itself
    # ...and it is not vacuous: on the two short edges the road wins, on the long one the FOOTPATH
    # does, so a `max` that silently took the road would drop that edge from 0.02485 to 0.02.
    assert gf[:2] == pytest.approx(road[:2]) and gf[2] == pytest.approx(fp[2])
    assert fp[2] > road[2]
