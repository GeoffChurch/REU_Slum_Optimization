# tests/test_commute_ratio.py
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.budget import _noded_graph, commute_ratio, commute_ratio_benefit, cost_benefit_curve
from reblock.contracts import Block

UTM = CRS.from_epsg(32734)


def _block(n_parcels: int, parcel_geoms: list[Polygon] | None = None) -> Block:
    boundary = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    geoms = parcel_geoms or [boundary] * n_parcels
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(n_parcels))}, geometry=geoms, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (100, 0)])], crs=UTM)
    return Block(block_id="b", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _roads(lines):
    return gpd.GeoDataFrame(geometry=lines, crs=UTM)


def _parcels_at(pts):
    # small distinct parcels so each maps to a nearby entry node
    return [Polygon([(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1)])
           for x, y in pts]


def test_single_egress_tree_is_zero() -> None:
    # one path from an interior point down to the single street: no parallel route -> rho = 0
    block = _block(3, _parcels_at([(50, 40), (50, 30), (50, 20)]))
    roads = _roads([LineString([(50, 0), (50, 50)])])
    assert commute_ratio(block, roads) == 0.0


def test_loop_gives_positive_rho() -> None:
    block = _block(3, _parcels_at([(40, 40), (50, 40), (60, 40)]))
    loop = _roads([LineString([(30, 0), (30, 50), (70, 50), (70, 0)])])  # two arms to the street
    assert commute_ratio(block, loop) > 0.0


def test_denser_loop_beats_sparser_loop() -> None:
    # REPLACES a brief-given "big loop beats tiny loop" test whose specific dimensions (big:
    # w=80/h=60 top-to-arm ratio 1.33; tiny: w=4/h=30 ratio 0.13) turned out to assert something
    # false: for a symmetric 2-arm loop the corner's rho has the closed form 1/(w/h + 2) --
    # SCALE-INVARIANT, a function of aspect ratio alone -- so the narrow/tall "tiny" loop
    # (ratio 0.13, near the 0.5 ceiling as ratio->0) provably beats the wide/short "big" one
    # (ratio 1.33) regardless of parcel-entry method; verified both analytically and empirically
    # (the original vertex-snap code and this file's point-projection code both give tiny > big).
    # Mean-over-reachable-parcels is coverage-insensitive by design (a road's benefit to the
    # parcels it does NOT reach is a different metric's job -- see access_benefit), so "spans more
    # parcels" alone can't be asserted to score higher. What IS always true (Rayleigh's
    # monotonicity law for resistor networks: adding conductance never increases effective
    # resistance to ground) is that adding a redundant connector to an existing loop can only
    # help -- never hurt -- the parcels it shortens a route for.
    parcels = _parcels_at([(20, 40), (40, 40), (60, 40), (80, 40)])
    block = _block(4, parcels)
    sparse = _roads([LineString([(10, 0), (10, 60), (90, 60), (90, 0)])])
    denser = _roads([LineString([(10, 0), (10, 60), (90, 60), (90, 0)]),
                     LineString([(30, 0), (30, 60)]), LineString([(70, 0), (70, 60)])])
    assert commute_ratio(block, denser) > commute_ratio(block, sparse)


def test_range_and_empty_guards() -> None:
    block = _block(4, _parcels_at([(40, 40), (50, 40), (60, 40), (30, 40)]))
    loop = _roads([LineString([(30, 0), (30, 50), (70, 50), (70, 0)])])
    r = commute_ratio(block, loop)
    assert 0.0 <= r < 1.0
    assert commute_ratio(block, _roads([])) == 0.0
    assert commute_ratio(block, None) == 0.0
    # NOTE: the "no parcels" guard (len(block.parcels) < 1) is not separately exercised here --
    # contracts.Block.__post_init__ raises ValueError on an empty parcels GeoDataFrame, so a
    # Block with 0 parcels cannot be constructed at all; the guard is defensive/unreachable via
    # the public contract.


def test_stranded_spur_excluded_no_blowup() -> None:
    # a road that never reaches the street: its parcels are excluded (reachable-conditioned), no
    # crash
    block = _block(2, _parcels_at([(50, 80), (50, 70)]))
    spur = _roads([LineString([(50, 60), (50, 90)])])  # detached from the south street
    assert commute_ratio(block, spur) == 0.0


def test_subdivision_invariance() -> None:
    # effective resistance is subdivision-invariant -> rho unchanged by an added mid-vertex
    block = _block(3, _parcels_at([(40, 40), (50, 40), (60, 40)]))
    loop = _roads([LineString([(30, 0), (30, 50), (70, 50), (70, 0)])])
    sub = _roads([LineString([(30, 0), (30, 25), (30, 50), (50, 50), (70, 50), (70, 0)])])
    assert abs(commute_ratio(block, loop) - commute_ratio(block, sub)) < 1e-9


def test_interior_interior_bridge_branch() -> None:
    # A vertical tree-stem road with a mid-vertex (50, 30), and a single parcel centered at
    # (50, 50): its nearest edge is the INTERIOR-INTERIOR span (50, 30)-(50, 60) (both endpoints
    # are interior nodes -- neither touches the street), and that span is a bridge (the road is a
    # tree), so this drives _entry_resistance's singular (denom < 1e-9) branch -- returning
    # min(guu + a, gvv + b) -- which no other test in this file reaches (the other bridge tests
    # only hit _entry_resistance_ground's ground-to-interior branch). Still a single-egress tree,
    # so rho = 0 regardless.
    block = _block(1, _parcels_at([(50, 50)]))
    roads = _roads([LineString([(50, 0), (50, 30), (50, 60)])])
    assert commute_ratio(block, roads) == 0.0


def test_crossing_is_noded_into_a_shared_vertex() -> None:  # RE-HOMED (deleted cycle-density suite)
    block = _block(4)
    roads = _roads([LineString([(20, 20), (80, 80)]), LineString([(20, 80), (80, 20)])])
    g = _noded_graph(roads, block.streets)
    assert (50.0, 50.0) in g.nodes
    assert g.degree[(50.0, 50.0)] == 4


def test_benefit_factory_terminal_matches_metric() -> None:
    block = _block(4, _parcels_at([(20, 40), (40, 40), (60, 40), (80, 40)]))
    roads = _roads([LineString([(10, 0), (10, 60), (90, 60), (90, 0)])])
    f = commute_ratio_benefit(block, roads)
    assert f(roads) == commute_ratio(block, roads)          # factory delegates to the metric
    curve = cost_benefit_curve(block, roads, benefit_fn=commute_ratio_benefit)
    assert curve.benefit[-1] == commute_ratio(block, roads)  # terminal == full-roads metric
    assert all(0.0 <= b < 1.0 for b in curve.benefit)        # do NOT assert monotone (rho isn't)


def test_no_interior_nodes_returns_zero() -> None:
    # roads lying ENTIRELY on the street -> every graph node is a street node -> no interior nodes
    # -> the `not interior` guard returns 0.0 (spec §3.6 guarded case).
    block = _block(3, _parcels_at([(30, 40), (50, 40), (70, 40)]))
    on_street = _roads([LineString([(10, 0), (90, 0)])])     # on the south street (y=0)
    assert commute_ratio(block, on_street) == 0.0
