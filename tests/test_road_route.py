"""The planarized road graph. Planarized, not raw: two roads that CROSS must let a walker turn at
the crossing, and the raw `_rnd` graph leaves them disconnected unless they happen to share an
endpoint (measured: 521 raw components against 35 planarized on the LP)."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import LineString

from reblock.permeability import PermeabilityParams, with_width
from reblock.road_route import build_roadnet

UTM = CRS.from_epsg(32734)
P = PermeabilityParams()


def _roads(*pairs):
    geoms = [LineString(c) for c, _w in pairs]
    gdf = with_width(gpd.GeoDataFrame(geometry=geoms, crs=UTM), 7.0)
    gdf["width_m"] = [w for _c, w in pairs]
    return gdf


def test_crossing_roads_are_connected_after_planarization():
    # An X. Raw endpoint keys would leave these two roads disjoint; planarizing must node the
    # crossing so a walker can turn there.
    net = build_roadnet(_roads(([(0, 10), (20, 10)], 7.0), ([(10, 0), (10, 20)], 7.0)), P)
    import networkx as nx
    g = nx.Graph()
    g.add_edges_from(zip(net.seg_a.tolist(), net.seg_b.tolist(), strict=True))
    assert nx.number_connected_components(g) == 1
    assert len(net.seg_a) >= 4, "the crossing must split both roads"


def test_segment_widths_survive_planarization():
    # The union destroys row identity; widths must be recovered by midpoint. The wide road's
    # segments must carry 12.0, the narrow one's 7.0.
    net = build_roadnet(_roads(([(0, 10), (40, 10)], 12.0), ([(0, 30), (40, 30)], 7.0)), P)
    wide = net.seg_width[np.isclose(net.nodes[net.seg_a][:, 1], 10.0)]
    narrow = net.seg_width[np.isclose(net.nodes[net.seg_a][:, 1], 30.0)]
    assert wide.size and np.allclose(wide, 12.0)
    assert narrow.size and np.allclose(narrow, 7.0)


def test_overlapping_roads_take_the_WIDEST_width():
    net = build_roadnet(_roads(([(0, 10), (40, 10)], 7.0), ([(0, 10), (40, 10)], 12.0)), P)
    assert np.allclose(net.seg_width, 12.0)


def test_segment_lengths_sum_to_the_road_length():
    net = build_roadnet(_roads(([(0, 10), (40, 10)], 7.0)), P)
    assert net.seg_len.sum() == pytest.approx(40.0)


def test_empty_roads_give_an_empty_net():
    net = build_roadnet(with_width(gpd.GeoDataFrame(geometry=[], crs=UTM), 7.0), P)
    assert len(net.seg_a) == 0 and len(net.nodes) == 0
