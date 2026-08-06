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
from reblock.road_route import build_roadnet, route_resistance

UTM = CRS.from_epsg(32734)
P = PermeabilityParams()
G = P.g_road_per_m
# Usable corridor a DEFAULT 7 m two-way road gives one direction: lane_width(7) - margin = 3.0.
# Every expected value below is in these units, because that is the capacity convention
# `road_conductance` already uses -- see `_usable_widths`.
U7 = 3.0


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

# ---------------------------------------------------------------------------
# route_resistance
# ---------------------------------------------------------------------------
# Resistance of ONE METRE of walking. Travel along a default road costs 1/(G*U7) = 0.05 per metre,
# so walking here is 20x dearer; on the pinned block the real ratio is ~500x. The gap is what makes
# the joint minimization safe -- an entry point far from the centroid is not worth walking to --
# and every expected value below turns on it.
WALK = 1.0
ROAD_M = 1.0 / (G * U7)      # 0.05, resistance per metre of a default two-way road


def _w(k: int) -> np.ndarray:
    return np.full(k, WALK, dtype=np.float64)


def test_a_zigzag_costs_more_than_a_straight_road_of_the_same_endpoints():
    """D1: today these score BIT-IDENTICALLY at detour ratios to 3.07x."""
    straight = build_roadnet(_roads(([(0, 0), (40, 0)], 7.0)), P)
    zig = build_roadnet(_roads(([(0, 0), (10, 12), (20, 0), (30, 12), (40, 0)], 7.0)), P)
    a = np.array([[0.0, 0.0]])
    b = np.array([[40.0, 0.0]])
    cut = np.array([np.inf])
    r_straight = route_resistance(straight, a, b, P, cut, _w(1))[0]
    r_zig = route_resistance(zig, a, b, P, cut, _w(1))[0]
    assert r_zig > r_straight * 1.5


def test_a_zigzag_is_not_short_circuited_by_WALKING_to_its_far_end():
    """The reason D1 survives a joint minimization, pinned. Every node of the network is an entry
    candidate, so the far end of the zigzag is one -- reachable by a 40 m straight line, exactly
    the straight road's own length. Priced at ROAD rate that leg makes the two nets score
    bit-identically (measured: it does). Priced at WALKING rate it costs 40.0 against the zigzag's
    own 3.12, and never binds."""
    zig = build_roadnet(_roads(([(0, 0), (10, 12), (20, 0), (30, 12), (40, 0)], 7.0)), P)
    got = route_resistance(zig, np.array([[0.0, 0.0]]), np.array([[40.0, 0.0]]), P,
                           np.array([np.inf]), _w(1))[0]
    zig_len = 4.0 * np.hypot(10.0, 12.0)
    assert got == pytest.approx(zig_len * ROAD_M, rel=1e-9), "the route follows the road"
    assert got < 40.0 * WALK, "...and beats the straight walk, which is what makes it worth taking"


def test_resistance_is_series_over_mixed_widths():
    # 20 m then 20 m, at USABLE widths 3.0 and 5.5 (two-way 7 m and 12 m roads, margin 1 m):
    # resistance is the SUM of len/(g*u), not len/(g*mean).
    net = build_roadnet(_roads(([(0, 0), (20, 0)], 7.0), ([(20, 0), (40, 0)], 12.0)), P)
    got = route_resistance(net, np.array([[0.0, 0.0]]), np.array([[40.0, 0.0]]), P,
                           np.array([np.inf]), _w(1))[0]
    g = P.g_road_per_m
    assert got == pytest.approx(20.0 / (g * 3.0) + 20.0 / (g * 5.5), rel=1e-6)


def test_a_road_too_far_to_reach_gives_no_benefit_and_is_discarded():
    """A3, under walking legs. The old form of this test asserted `inf` for a disconnected
    component; that came from the entry being ASSIGNED to the nearest segment at road rate. With a
    walking leg every component is reachable, so the fallback is no longer a special value -- a
    road you would have to walk 100 m to reach simply loses to walking straight there, and the
    caller's own cutoff discards it. Still no gate and no rule."""
    net = build_roadnet(_roads(([(0, 100), (40, 100)], 7.0)), P)
    a, b = np.array([[0.0, 0.0]]), np.array([[40.0, 0.0]])
    footpath = np.array([40.0 * WALK])           # what walking the edge direct costs
    assert not np.isfinite(route_resistance(net, a, b, P, footpath, _w(1))[0])
    assert route_resistance(net, a, b, P, np.array([np.inf]), _w(1))[0] > footpath[0]


def test_disconnected_components_are_not_traversed_as_though_joined():
    """The disconnection property that still bites. Two collinear stubs with a 20 m gap: the route
    may ride ONE of them and walk the rest, but must not cross the gap along the network.

    Riding the first stub 10 m and walking the remaining 30 m is genuinely better than walking all
    40 m, so the answer is finite -- that is correct, not a regression. What it must never be is
    the 40 m of riding a joined road would give."""
    split = build_roadnet(_roads(([(0, 0), (10, 0)], 7.0), ([(30, 0), (40, 0)], 7.0)), P)
    joined = build_roadnet(_roads(([(0, 0), (40, 0)], 7.0)), P)
    a, b = np.array([[0.0, 0.0]]), np.array([[40.0, 0.0]])
    cut = np.array([np.inf])
    r_split = route_resistance(split, a, b, P, cut, _w(1))[0]
    r_joined = route_resistance(joined, a, b, P, cut, _w(1))[0]
    assert r_joined == pytest.approx(40.0 * ROAD_M, rel=1e-9)
    assert r_split == pytest.approx(10.0 * ROAD_M + 30.0 * WALK, rel=1e-9), "ride 10 m, walk 30"
    assert r_split > r_joined


def test_the_early_exit_is_EXACT_not_approximate():
    """The monotonicity proof depends on the cutoff returning the SAME answer, not a close one --
    it only ever discards values that `max(footpath, road)` would drop anyway."""
    net = build_roadnet(_roads(([(0, 0), (40, 0)], 7.0), ([(40, 0), (40, 40)], 7.0)), P)
    a = np.array([[0.0, 0.0], [0.0, 0.0]])
    b = np.array([[40.0, 40.0], [40.0, 0.0]])
    exact = route_resistance(net, a, b, P, np.array([np.inf, np.inf]), _w(2))
    generous = route_resistance(net, a, b, P, exact * 2.0, _w(2))
    assert np.allclose(exact, generous), "a cutoff above the true value must not change it"
    # A7 asks for BIT-identity, not closeness -- `allclose` above would pass an early exit that
    # rounded differently, and the Loewner argument needs the two to be the same function.
    assert np.array_equal(exact, generous)


def test_the_early_exit_is_EXACT_on_a_route_whose_MIDDLE_carries_it():
    """The real guard on the bounded search; the two-segment case above cannot be one.

    There, every route runs end to end of a single segment at each side, so a truncated
    node-to-node distance is simply absorbed: the search cuts the same route at the OTHER endpoint
    and pays the difference as a partial-segment offset, reaching the identical answer through a
    shorter one. Fault injection proved it -- shrinking `limit` to 0.4x left that test green.

    A real route is tens of metres over ~5 m segments (`topology`'s median is 4.83 m), so the
    middle is nearly all of the resistance and there is no shorter cut to fall back on.
    """
    coords = [(float(x), 0.0) for x in range(0, 201, 5)]
    net = build_roadnet(_roads((coords, 7.0)), P)
    a, b = np.array([[2.5, 0.0]]), np.array([[197.5, 0.0]])
    exact = route_resistance(net, a, b, P, np.array([np.inf]), _w(1))
    assert exact[0] == pytest.approx(195.0 * ROAD_M, rel=1e-9)
    generous = route_resistance(net, a, b, P, exact * 2.0, _w(1))
    assert np.array_equal(exact, generous), "a cutoff above the true value must not change it"


def test_a_cutoff_below_the_true_resistance_returns_inf():
    """The other half of the contract: the caller takes `max(footpath, 1/R)`, so a route it would
    discard may come back as `inf` -- that is what makes the bounded search legitimate."""
    net = build_roadnet(_roads(([(0, 0), (40, 0)], 7.0)), P)
    a, b = np.array([[0.0, 0.0]]), np.array([[40.0, 0.0]])
    true = 40.0 * ROAD_M
    assert route_resistance(net, a, b, P, np.array([true * 1.5]), _w(1))[0] == pytest.approx(true)
    assert not np.isfinite(route_resistance(net, a, b, P, np.array([true * 0.5]), _w(1))[0])


def test_projections_attach_where_they_land_not_at_the_nearest_node():
    """Two 50 m segments meeting at (50, 0); the points sit 10 m in from each far end. The route
    is 40 + 40 = 80 m of road, NOT the 100 m that snapping each point to its nearest NODE would
    charge. `topology`'s median segment is 4.83 m against sub-metre access legs, so snapping is
    not a rounding error -- it is the dominant term."""
    net = build_roadnet(_roads(([(0, 0), (50, 0)], 7.0), ([(50, 0), (100, 0)], 7.0)), P)
    got = route_resistance(net, np.array([[10.0, 0.0]]), np.array([[90.0, 0.0]]), P,
                           np.array([np.inf]), _w(1))[0]
    assert got == pytest.approx(80.0 * ROAD_M, rel=1e-9)
    assert got < 100.0 * ROAD_M, "snapping to the nearest node would charge the whole 100 m"


def test_the_access_legs_are_charged_at_the_WALKING_rate_in_series():
    """A point OFF the network pays `|c - projection| * walk_res_per_m` on top of the on-network
    route -- the spec's `r_leg`, in series, but at the FOOTPATH rate, because the leg is walking.

    Pair 0 crosses a width boundary, pair 1 stays on ONE segment: the leg is charged in BOTH
    branches, which is what keeps `R` continuous as a point slides across a node (drop it from the
    same-segment branch and `R` jumps by `r_leg` the instant the two projections part company).
    """
    net = build_roadnet(_roads(([(0, 0), (20, 0)], 7.0), ([(20, 0), (40, 0)], 12.0)), P)
    a = np.array([[10.0, 3.0], [5.0, 3.0]])
    b = np.array([[40.0, 0.0], [15.0, 0.0]])
    got = route_resistance(net, a, b, P, np.array([np.inf, np.inf]), _w(2))
    g = P.g_road_per_m
    # pair 0: walk 3 m to the projection at (10,0), ride 10 m at usable 3.0, then 20 m at usable 5.5
    assert got[0] == pytest.approx(3.0 * WALK + 10.0 / (g * 3.0) + 20.0 / (g * 5.5), rel=1e-9)
    # pair 1: walk 3 m, then |0.25 - 0.75| * 20 m along the SAME segment, never reaching a node
    assert got[1] == pytest.approx(3.0 * WALK + 10.0 / (g * 3.0), rel=1e-9)


def test_a_leg_at_ROAD_rate_would_let_a_long_straight_leg_beat_a_short_one():
    """The fault this rate choice exists to prevent, pinned as a property rather than a value.

    Entry is minimized over the WHOLE network, so a long straight leg to a distant node is always
    a candidate. Walking must make it lose. Here the near entry is 3 m off a road that then runs
    the wrong way; the far node is 50 m away in a straight line. At walking rate the near entry
    wins outright; at road rate (1/20th the price per metre) the 50 m leg would win and the route
    would cut across the block.
    """
    net = build_roadnet(_roads(([(0, 0), (0, 40)], 7.0), ([(50, 0), (50, 40)], 7.0)), P)
    a, b = np.array([[3.0, 0.0]]), np.array([[3.0, 40.0]])
    got = route_resistance(net, a, b, P, np.array([np.inf]), _w(1))[0]
    near = 3.0 * WALK + 40.0 * ROAD_M + 3.0 * WALK          # walk on, ride the near road, walk off
    far = np.hypot(47.0, 0.0) * WALK * 2 + 40.0 * ROAD_M    # walk to the far road and back
    assert near < far, "the fixture must actually discriminate the two rates"
    assert got == pytest.approx(near, rel=1e-9), "the NEAR entry wins when the leg is walking"


def test_a_one_way_segment_carries_travel_in_one_direction_only():
    """`seg_oneway` is a hard gate on ROAD travel, exactly as `edge_conductances` treats it. The
    permitted direction costs what the road costs; the forbidden one gets no road benefit at all
    and is left walking the whole way. Which of the two the planarizer calls forward is its own
    business, so assert the PAIR."""
    net = build_roadnet(_roads_ow(([(0, 0), (40, 0)], 7.0, True)), P)
    assert net.seg_oneway.all()
    a, b = np.array([[0.0, 0.0]]), np.array([[40.0, 0.0]])
    cut = np.array([np.inf])
    both = sorted([route_resistance(net, a, b, P, cut, _w(1))[0],
                   route_resistance(net, b, a, P, cut, _w(1))[0]])
    g = P.g_road_per_m
    assert both[0] == pytest.approx(40.0 / (g * 6.0), rel=1e-9)   # one-way: usable = 7 - margin
    assert both[1] == pytest.approx(40.0 * WALK, rel=1e-9), "the forbidden direction just walks"


def test_a_two_way_net_is_symmetric():
    """The control for the one-way test: with nothing one-way, both directions must price alike.

    The route needs a MIDDLE for this to guard anything -- with the two points on the two end
    segments of a two-segment net, both directions are paid entirely in partial-segment offsets
    (which are two-way by construction) and never traverse the graph at all, so even a graph
    searched as directed comes back symmetric. Four segments with the points mid-way along the
    outer two puts 20 of the 30 m into the graph.
    """
    net = build_roadnet(_roads(([(0, 0), (10, 0), (20, 0), (30, 0), (40, 0)], 7.0)), P)
    assert not net.seg_oneway.any()
    a, b = np.array([[5.0, 0.0]]), np.array([[35.0, 0.0]])
    cut = np.array([np.inf])
    fwd = route_resistance(net, a, b, P, cut, _w(1))[0]
    bwd = route_resistance(net, b, a, P, cut, _w(1))[0]
    assert fwd == pytest.approx(30.0 * ROAD_M, rel=1e-9)
    assert bwd == fwd


def test_an_empty_net_gives_infinite_resistance():
    """`build_roadnet` returns an empty net for empty roads, and the caller asks for a route
    anyway -- there is nothing to route over, so the road term is zero via `1/inf`."""
    net = build_roadnet(with_width(gpd.GeoDataFrame(geometry=[], crs=UTM), 7.0), P)
    got = route_resistance(net, np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]), P,
                           np.array([np.inf]), _w(1))
    assert got.shape == (1,) and not np.isfinite(got[0])


def test_segment_resistance_uses_road_conductance_s_OWN_capacity_convention():
    """`_usable_widths` re-derives `lane_width` + `road_conductance`'s `usable` per SEGMENT, so it
    can drift from them. Pin it against the real functions.

    This is what keeps the change honest: the crow-flies term being replaced is
    `road_conductance(params, lane_width(w), d)`, so a route of length `d` must price IDENTICALLY
    to it. Only then does `L >= d` mean road conductance strictly falls (A2) rather than a capacity
    re-base hiding inside a geometry fix.
    """
    from reblock.permeability import lane_width, road_conductance
    from reblock.road_route import _usable_widths

    net = build_roadnet(
        _roads_ow(([(0, 0), (40, 0)], 7.0, False), ([(0, 30), (40, 30)], 12.0, True)), P)
    got = _usable_widths(net, P)
    for w, oneway in ((7.0, False), (12.0, True)):
        pick = np.isclose(net.seg_width, w)
        assert pick.any()
        want = lane_width(P, w, oneway=oneway) - P.road_margin_m
        assert np.allclose(got[pick], want), f"width {w} oneway={oneway}"

    # ...and end to end: a straight road of length d must give exactly today's crow-flies term.
    straight = build_roadnet(_roads(([(0, 0), (40, 0)], 7.0)), P)
    r = route_resistance(straight, np.array([[0.0, 0.0]]), np.array([[40.0, 0.0]]), P,
                         np.array([np.inf]), _w(1))[0]
    today = road_conductance(P, np.array([lane_width(P, 7.0)]), np.array([40.0]))[0]
    assert 1.0 / r == pytest.approx(today, rel=1e-12), "a route of length d must price like crow"
