"""Guards for the width/direction solver: it must respect the budget, and climb a real lattice."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.contracts import Block
from reblock.orient import one_way_width
from reblock.permeability import (
    DEFAULT_ROAD_WIDTH_M,
    ONEWAY_COL,
    WIDTH_COL,
    PermeabilityParams,
    with_width,
)
from reblock.width_solver import BOTTOM, ONE_WAY, TWO_WAY, WidthSolver

UTM = CRS.from_epsg(32734)
PARAMS = PermeabilityParams()


def _block(k: int = 10, cell: float = 10.0) -> Block:
    polys, ids = [], []
    for r in range(k):
        for c in range(k):
            x0, y0 = c * cell, r * cell
            polys.append(Polygon([(x0, y0), (x0 + cell, y0), (x0 + cell, y0 + cell),
                                  (x0, y0 + cell)]))
            ids.append(r * k + c)
    return Block(
        block_id="ws", crs=UTM,
        boundary=Polygon([(0, 0), (k * cell, 0), (k * cell, k * cell), (0, k * cell)]),
        parcels=gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM),
        streets=gpd.GeoDataFrame(geometry=[LineString([(0, 0), (k * cell, 0)])], crs=UTM),
        building_points=gpd.GeoDataFrame(geometry=[p.centroid for p in polys], crs=UTM))


def _roads() -> gpd.GeoDataFrame:
    # a street-to-street loop (orientable) plus two spurs off it (bridges, so never one-way)
    return with_width(gpd.GeoDataFrame(geometry=[
        LineString([(20.0, 0.0), (20.0, 60.0)]),
        LineString([(20.0, 60.0), (70.0, 60.0)]),
        LineString([(70.0, 60.0), (70.0, 0.0)]),
        LineString([(45.0, 60.0), (45.0, 90.0)]),
        LineString([(20.0, 30.0), (0.0, 30.0)]),
    ], crs=UTM), DEFAULT_ROAD_WIDTH_M)


def test_lattice_heights_are_ordered():
    # The greedy only ever climbs, so the encoding must actually be an order.
    assert BOTTOM < ONE_WAY < TWO_WAY


def test_it_respects_the_displacement_budget():
    block, roads = _block(), _roads()
    for cap in (0.02, 0.05, 0.10, 0.25):
        got = WidthSolver(params=PARAMS).solve(block, roads, max_displacement=cap)
        assert got.displacement_frac <= cap + 1e-9, f"overspent at cap={cap}"


def test_a_bigger_budget_never_buys_less_permeability():
    # Monotone in the budget: the feasible set only grows, and the solver climbs a lattice on which
    # permeability is monotone, so a larger cap cannot do worse.
    block, roads = _block(), _roads()
    caps = [0.02, 0.06, 0.12, 0.30]
    perms = [WidthSolver(params=PARAMS).solve(block, roads, max_displacement=c).permeability
             for c in caps]
    assert all(b >= a - 1e-9 for a, b in zip(perms, perms[1:], strict=False)), perms


def test_every_emitted_road_is_buildable_and_scorable():
    block, roads = _block(), _roads()
    got = WidthSolver(params=PARAMS).solve(block, roads, max_displacement=0.20)
    assert len(got.roads) > 0
    widths = got.roads[WIDTH_COL].to_numpy(dtype=float)
    one = got.roads[ONEWAY_COL].to_numpy(dtype=bool)
    floors = np.where(one, PARAMS.min_one_way_width_m, PARAMS.min_two_way_width_m)
    assert (widths >= floors).all(), "emitted a road too narrow for its own direction"
    assert set(np.unique(widths)) <= {one_way_width(PARAMS, DEFAULT_ROAD_WIDTH_M),
                                      DEFAULT_ROAD_WIDTH_M}


def test_a_bridge_is_never_made_one_way():
    # Robbins is not a feasibility constraint here (the reverse falls back to footpath, so nothing
    # strands) but it IS what makes a one-way bridge pointless, and `strong_orientation` refuses to
    # orient one -- so the solver must have no one-way option for it.
    block = _block()
    spur = with_width(gpd.GeoDataFrame(
        geometry=[LineString([(50.0, 0.0), (50.0, 60.0)])], crs=UTM), DEFAULT_ROAD_WIDTH_M)
    got = WidthSolver(params=PARAMS).solve(block, spur, max_displacement=0.30)
    assert not got.roads[ONEWAY_COL].any()
    assert got.one_way == 0


def test_it_builds_a_subnetwork_not_the_whole_proposal():
    # `don't build` is the bottom of the lattice, so a tight budget must leave roads UNBUILT rather
    # than shrink every road -- that is the property that makes this comparable to a lens prefix.
    block, roads = _block(), _roads()
    tight = WidthSolver(params=PARAMS).solve(block, roads, max_displacement=0.03)
    loose = WidthSolver(params=PARAMS).solve(block, roads, max_displacement=0.40)
    assert tight.built < loose.built, (tight.built, loose.built)


def test_empty_roads_solve_to_empty():
    block = _block()
    empty = with_width(gpd.GeoDataFrame(geometry=[], crs=UTM), DEFAULT_ROAD_WIDTH_M)
    got = WidthSolver(params=PARAMS).solve(block, empty, max_displacement=0.10)
    assert len(got.roads) == 0
    assert got.displacement_frac == 0.0 and got.built == 0


def test_a_zero_budget_builds_nothing():
    block, roads = _block(), _roads()
    got = WidthSolver(params=PARAMS).solve(block, roads, max_displacement=0.0)
    assert got.built == 0 and got.permeability == pytest.approx(0.0, abs=1e-9)


def test_building_points_are_not_required():
    # A synthetic block without building points still has to solve rather than divide by zero.
    block = _block()
    bare = Block(block_id=block.block_id, crs=block.crs, boundary=block.boundary,
                 parcels=block.parcels, streets=block.streets)
    got = WidthSolver(params=PARAMS).solve(bare, _roads(), max_displacement=0.10)
    assert got.displacement_frac >= 0.0
