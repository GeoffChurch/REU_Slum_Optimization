import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString

from reblock.derive.network_metrics import (
    crossing_counts,
    degree_fractions,
    meshedness,
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
