"""Robbins' rule: bridgeless <=> strongly orientable, and nothing else gets made one-way."""
from __future__ import annotations

import geopandas as gpd
import networkx as nx
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.contracts import Block
from reblock.orient import (
    STREET_NODE,
    bridge_fraction,
    one_way_width,
    strong_orientation,
)
from reblock.permeability import (
    DEFAULT_ROAD_WIDTH_M,
    ONEWAY_COL,
    WIDTH_COL,
    PermeabilityParams,
    lane_width,
    with_width,
)

UTM = CRS.from_epsg(32734)
PARAMS = PermeabilityParams()


def _block(streets: list[LineString]) -> Block:
    # One big parcel is enough: orientation reads the ROAD graph and the street geometry, never the
    # parcel mesh. The boundary is generous so nothing is clipped.
    poly = Polygon([(-10, -10), (110, -10), (110, 110), (-10, 110)])
    return Block(block_id="o", crs=UTM,
                 boundary=poly,
                 parcels=gpd.GeoDataFrame({"parcel_id": [0]}, geometry=[poly], crs=UTM),
                 streets=gpd.GeoDataFrame(geometry=streets, crs=UTM))


def _roads(*lines: list[tuple[float, float]]) -> gpd.GeoDataFrame:
    return with_width(gpd.GeoDataFrame(geometry=[LineString(c) for c in lines], crs=UTM),
                      DEFAULT_ROAD_WIDTH_M)


def test_one_way_width_equalizes_per_direction_conductance():
    # The derived rule: a one-way road matches a two-way road's PER-DIRECTION capacity at
    # (W + margin)/2 -- 3.5 m, not the naive 3.0 m, because both pay the margin once.
    w_one = one_way_width(PARAMS, DEFAULT_ROAD_WIDTH_M)
    assert w_one == pytest.approx(4.0)                # re-based with the floors
    assert w_one > DEFAULT_ROAD_WIDTH_M / 2.0        # wider than half; the margin is why
    assert lane_width(PARAMS, w_one, oneway=True) == pytest.approx(
        lane_width(PARAMS, DEFAULT_ROAD_WIDTH_M))


def test_a_spur_is_a_bridge_and_stays_two_way():
    # A dead-end off the street: orienting it strands whatever it serves, so Robbins forbids it.
    block = _block([LineString([(0.0, 0.0), (100.0, 0.0)])])
    roads = _roads([(50.0, 0.0), (50.0, 40.0)])
    out = strong_orientation(block, roads, params=PARAMS)
    assert not out[ONEWAY_COL].any()
    assert out[WIDTH_COL].iloc[0] == DEFAULT_ROAD_WIDTH_M     # no discount for an unorientable road
    assert bridge_fraction(block, roads) == pytest.approx(1.0)


def test_a_street_to_street_loop_is_orientable():
    # Out at x=20, across at y=40, back at x=80. Read alone that is a PATH -- every edge a bridge --
    # but its two ends are joined by the street, which is why the street contracts to one node.
    block = _block([LineString([(0.0, 0.0), (100.0, 0.0)])])
    roads = _roads([(20.0, 0.0), (20.0, 40.0)],
                   [(20.0, 40.0), (80.0, 40.0)],
                   [(80.0, 40.0), (80.0, 0.0)])
    assert bridge_fraction(block, roads) == pytest.approx(0.0)

    out = strong_orientation(block, roads, params=PARAMS)
    assert out[ONEWAY_COL].all()
    assert list(out[WIDTH_COL]) == pytest.approx([4.0, 4.0, 4.0])

    # ...and the orientation is strongly connected: every node reaches every other. Street-touching
    # nodes contract to one, the same way `strong_orientation` models them.
    dg = nx.DiGraph()
    for geom in out.geometry:
        cs = [(round(x, 2), round(y, 2)) for x, y in geom.coords]
        for a, b in zip(cs, cs[1:], strict=False):
            dg.add_edge(STREET_NODE if a[1] == 0.0 else a, STREET_NODE if b[1] == 0.0 else b)
    assert nx.is_strongly_connected(dg)


def test_a_loop_with_a_spur_orients_the_loop_and_spares_the_spur():
    # The mixed case that matters: partial orientability must be reported per road, not all-or-none.
    block = _block([LineString([(0.0, 0.0), (100.0, 0.0)])])
    roads = _roads([(20.0, 0.0), (20.0, 40.0)],
                   [(20.0, 40.0), (80.0, 40.0)],
                   [(80.0, 40.0), (80.0, 0.0)],
                   [(50.0, 40.0), (50.0, 70.0)])       # dead-end hanging off the loop
    out = strong_orientation(block, roads, params=PARAMS)
    assert list(out[ONEWAY_COL]) == [True, True, True, False]
    assert out[WIDTH_COL].iloc[3] == DEFAULT_ROAD_WIDTH_M
    assert 0.0 < bridge_fraction(block, roads) < 1.0


def test_orientation_reverses_geometry_so_coordinate_order_is_the_permitted_direction():
    # `edge_conductances` reads direction off coordinate order, so a road oriented against its own
    # drawing must come back REVERSED -- not merely flagged.
    block = _block([LineString([(0.0, 0.0), (100.0, 0.0)])])
    roads = _roads([(20.0, 0.0), (20.0, 40.0)],
                   [(20.0, 40.0), (80.0, 40.0)],
                   [(80.0, 0.0), (80.0, 40.0)])        # drawn street->interior, like the first
    out = strong_orientation(block, roads, params=PARAMS)
    assert out[ONEWAY_COL].all()
    # the three roads must form a consistent circulation, so exactly one of the two radials runs
    # away from the street and the other toward it
    ends_at_street = [geom.coords[-1][1] == 0.0 for geom in out.geometry]
    assert sum(ends_at_street) == 1


def test_orientation_needs_a_width_column():
    block = _block([LineString([(0.0, 0.0), (100.0, 0.0)])])
    bare = gpd.GeoDataFrame(geometry=[LineString([(50.0, 0.0), (50.0, 40.0)])], crs=UTM)
    with pytest.raises(ValueError, match=WIDTH_COL):
        strong_orientation(block, bare, params=PARAMS)


def test_empty_roads_orient_to_empty():
    block = _block([LineString([(0.0, 0.0), (100.0, 0.0)])])
    empty = with_width(gpd.GeoDataFrame(geometry=[], crs=UTM), DEFAULT_ROAD_WIDTH_M)
    out = strong_orientation(block, empty, params=PARAMS)
    assert len(out) == 0 and ONEWAY_COL in out.columns
    assert bridge_fraction(block, empty) == 0.0
