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


def _roads_ow(*pairs):
    """Like `_roads`, but each item is `(coords, width_m, oneway)` -- an explicit per-road
    direction, needed to test that `seg_oneway` is recovered from the same governing road as
    `seg_width`, not independently."""
    geoms = [LineString(c) for c, _w, _o in pairs]
    gdf = with_width(gpd.GeoDataFrame(geometry=geoms, crs=UTM), 7.0)
    gdf["width_m"] = [w for _c, w, _o in pairs]
    gdf["oneway"] = [o for _c, _w, o in pairs]
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


def test_a_wide_oneway_road_governs_seg_oneway_over_a_narrower_two_way_one():
    # Same corridor, two overlapping roads: a WIDE one-way (12.0, clears the 4.0 one-way floor)
    # and a NARROW two-way (7.0, at the 7.0 two-way floor). Widest wins on width -- so it must ALSO
    # win on direction. An independent AND across every covering road (the brief's original logic)
    # would flip this to False, because it ANDs in the two-way road's False regardless of which
    # road is widest.
    net = build_roadnet(
        _roads_ow(([(0, 10), (40, 10)], 12.0, True), ([(0, 10), (40, 10)], 7.0, False)), P)
    assert np.allclose(net.seg_width, 12.0)
    assert net.seg_oneway.all(), "the widest (one-way) road must govern direction too"


def test_a_wide_two_way_road_governs_seg_oneway_over_a_narrower_oneway_one():
    # The mirror case: a WIDE two-way (12.0) and a NARROW one-way (5.0, clears the 4.0 floor).
    # Widest wins on width -- and it is the two-way road, so seg_oneway must be False.
    net = build_roadnet(
        _roads_ow(([(0, 10), (40, 10)], 12.0, False), ([(0, 10), (40, 10)], 5.0, True)), P)
    assert np.allclose(net.seg_width, 12.0)
    assert not net.seg_oneway.any(), "the widest (two-way) road must govern direction too"


def test_a_midpoint_no_corridor_covers_falls_back_to_the_narrowest_road_for_BOTH_fields(
        monkeypatch):
    """The documented fallback (a midpoint no corridor covers) is a defensive guard for
    buffer-approximation edge cases -- every real segment's midpoint lies ON the road it came
    from, so it is always covered by that road's own buffer at any width clearing the buildable
    floors. There is no real geometry that reaches this branch, so force it: stub the STRtree so
    every lookup misses, then check the intent stated in `build_roadnet`'s docstring -- width AND
    oneway BOTH come from the single narrowest road, not from independent defaults."""
    class _NoHitsTree:
        def __init__(self, *_args, **_kwargs):
            pass

        def query(self, *_args, **_kwargs):
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    monkeypatch.setattr("reblock.road_route.STRtree", _NoHitsTree)
    # Narrowest road (7.0, at its two-way floor) is TWO-WAY while the wider one (12.0) is one-way
    # -- chosen so this fails under any scratch-value default for seg_oneway, whichever way that
    # default leans (in particular the brief's own `seg_o = np.ones(...)`, which an all-miss query
    # never touches and so leaves at True). A narrower two-way road isn't buildable at all (floor
    # 7.0), so 7.0 is the tightest two-way width available here.
    net = build_roadnet(
        _roads_ow(([(0, 10), (40, 10)], 12.0, True), ([(0, 30), (40, 30)], 7.0, False)), P)
    assert np.allclose(net.seg_width, 7.0), "every segment falls back to the narrowest road's width"
    assert not net.seg_oneway.any(), "...and that SAME road's direction, not an independent default"
