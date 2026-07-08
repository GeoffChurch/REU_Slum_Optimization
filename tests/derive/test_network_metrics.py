import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString

from reblock.derive.network_metrics import (
    boundary_redundant_road_fraction,
    circuity,
    cross_block_trunk_length_m,
    crossing_counts,
    degree_fractions,
    meshedness,
    n_cross_block_streets,
    node_network,
)

UTM = CRS.from_epsg(32734)


def _lines(*coords: list[tuple[float, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[LineString(c) for c in coords], crs=UTM)


def test_plus_is_one_crossing() -> None:
    # a "+" of two lines crossing at the origin -> one degree-4 node, 4 dead-end tips
    roads = _lines([(-1, 0), (1, 0)], [(0, -1), (0, 1)])
    empty = gpd.GeoDataFrame(geometry=[], crs=UTM)
    g = node_network(roads, empty)
    cc = crossing_counts(g)
    assert cc["n_crossings"] == 1
    assert cc["n_dead_ends"] == 4
    assert degree_fractions(g)["four_way_fraction"] > 0


def test_near_miss_nodes_via_set_precision() -> None:
    # two stubs 0.4 m apart (< STREET_TOL=0.5) crossed by a through-line: set_precision
    # snaps them so the crossing is a single degree-4 node, not a missed near-miss.
    roads = _lines([(-1, 0), (-0.2, 0)], [(0.2, 0), (1, 0)], [(0, -1), (0, 1)])
    empty = gpd.GeoDataFrame(geometry=[], crs=UTM)
    g = node_network(roads, empty)
    assert crossing_counts(g)["n_crossings"] == 1


def test_tree_has_zero_meshedness_grid_positive() -> None:
    # a path (tree) has no cycles; a 2x2 grid of squares has cycles
    tree = _lines([(0, 0), (1, 0)], [(1, 0), (2, 0)], [(1, 0), (1, 1)])
    empty = gpd.GeoDataFrame(geometry=[], crs=UTM)
    assert meshedness(node_network(tree, empty)) == 0.0
    grid = _lines([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])   # one closed loop
    assert meshedness(node_network(grid, empty)) > 0.0


VBOUND = MultiLineString([[(0, -5), (0, 5)]])   # the interior boundary line x=0


def test_cross_block_counts_both_sides_only() -> None:
    crossing = _lines([(-2, 0), (2, 0)])          # vertices strictly on both sides of x=0
    along = _lines([(0, -3), (0, 3)])             # runs ALONG the boundary -> not a crossing
    kiss = _lines([(-2, 1), (0, 0), (-2, -1)])    # touches x=0 but stays on the left -> not
    assert n_cross_block_streets(crossing, VBOUND) == 1
    assert n_cross_block_streets(along, VBOUND) == 0
    assert n_cross_block_streets(kiss, VBOUND) == 0
    assert cross_block_trunk_length_m(crossing, VBOUND) == 4.0


def test_boundary_redundant_road_fraction_parallel_vs_crossing() -> None:
    import pytest
    # a road parallel to and within `band` of the interior boundary, that never crosses it,
    # is entirely "redundant" (the spine a shared through-trunk would merge) -> fraction ~1.0
    parallel = _lines([(2, -3), (2, 3)])
    assert boundary_redundant_road_fraction(parallel, VBOUND, band=20.0) == pytest.approx(1.0)

    # a road that crosses the boundary is excluded from the "redundant" numerator entirely
    # (even though it also runs within band) -> fraction 0.0
    crossing = _lines([(-2, 0), (2, 0)])
    assert boundary_redundant_road_fraction(crossing, VBOUND, band=20.0) == 0.0


def test_circuity_straight_is_one_detour_is_more() -> None:
    import geopandas as gpd
    from shapely.geometry import Polygon, box

    from reblock.contracts import Block
    # a 1x5 strip; street on the left edge. Parcel k sits ~k m along the adjacency chain
    # but only ~k m straight-line -> circuity ~1 for a strip (adjacency ~ euclidean here).
    polys = [box(i, 0, i + 1, 1) for i in range(5)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(5))}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])], crs=UTM)
    block = Block(block_id="s", crs=UTM, boundary=Polygon([(0, 0), (5, 0), (5, 1), (0, 1)]),
                  parcels=parcels, streets=streets)
    c = circuity(block, None)
    assert c >= 1.0 and c < 1.5


def test_throughput_bottleneck_single_vs_branched() -> None:
    # 3 parcels clustered at interior node (7,2), > tol from every perimeter edge, so their
    # only egress is along the road corridor(s) to the left perimeter. A single unit-capacity
    # corridor bottlenecks all 3 to min-cut 1 -> 1/3; two edge-disjoint corridors -> min-cut 2
    # -> 2/3. Both < 1.0 (a genuine bottleneck the metric detects), branched > single.
    import pytest
    from shapely.geometry import Polygon, box

    from reblock.contracts import Block
    from reblock.derive.network_metrics import throughput_ratio

    parcels = gpd.GeoDataFrame(
        {"parcel_id": [0, 1, 2]},
        geometry=[box(7, 1, 8, 2), box(7, 2, 8, 3), box(6, 1.5, 7, 2.5)], crs=UTM)
    boundary = Polygon([(0, 0), (10, 0), (10, 4), (0, 4)])
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 4)])], crs=UTM)
    block = Block(block_id="s", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)

    single = _lines([(7, 2), (0, 2)])                        # one corridor to the perimeter
    branched = _lines([(7, 2), (0, 1)], [(7, 2), (0, 3)])    # two edge-disjoint corridors
    t_single = throughput_ratio(node_network(single, streets), block)
    t_branched = throughput_ratio(node_network(branched, streets), block)

    assert t_single == pytest.approx(1 / 3)
    assert t_branched == pytest.approx(2 / 3)
    assert t_single < 1.0 and t_branched > t_single  # a real bottleneck, and branching relieves it
