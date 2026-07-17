# tests/test_cycle_density.py
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.budget import _noded_graph, cycle_density
from reblock.contracts import Block

UTM = CRS.from_epsg(32734)


def _block(n_parcels: int) -> Block:
    # A square block whose `streets` is the south edge; parcel count controls the /P denominator.
    boundary = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(n_parcels))},
                               geometry=[boundary] * n_parcels, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (100, 0)])], crs=UTM)
    return Block(block_id="b", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _roads(lines: list[LineString]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=lines, crs=UTM)


def test_tree_has_zero_cycles() -> None:
    # A path touching the street: no loop -> circuit rank 0.
    block = _block(4)
    roads = _roads([LineString([(50, 0), (50, 40), (70, 40)])])
    assert cycle_density(block, roads) == 0.0


def test_single_loop_is_one_over_parcels() -> None:
    # A closed square interior road = one independent cycle; /P with P=4 -> 0.25.
    block = _block(4)
    loop = LineString([(20, 20), (60, 20), (60, 60), (20, 60), (20, 20)])
    assert cycle_density(block, _roads([loop])) == 0.25


def test_two_disjoint_loops_have_circuit_rank_two() -> None:
    block = _block(8)   # P=8 -> 2/8 = 0.25
    a = LineString([(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)])
    b = LineString([(60, 60), (80, 60), (80, 80), (60, 80), (60, 60)])
    assert cycle_density(block, _roads([a, b])) == 0.25


def test_crossing_is_noded_into_a_shared_vertex() -> None:
    # Two crossing roads that individually have no loop: planarizing nodes the X into 4 edges
    # sharing the centre. With the street edge they still form no cycle here, but the crossing MUST
    # create a degree-4 centre node (5 nodes, 4 edges, 1 component -> circuit rank 0), proving that
    # noding happened.
    block = _block(4)
    roads = _roads([LineString([(20, 20), (80, 80)]), LineString([(20, 80), (80, 20)])])
    g = _noded_graph(roads, block.streets)
    assert (50.0, 50.0) in g.nodes            # the crossing became a shared vertex
    assert g.degree[(50.0, 50.0)] == 4


def test_subdivision_invariance() -> None:
    # Circuit rank is a topological invariant: adding a mid-vertex to a loop edge won't change it.
    block = _block(4)
    loop = LineString([(20, 20), (60, 20), (60, 60), (20, 60), (20, 20)])
    subdivided = LineString([(20, 20), (40, 20), (60, 20), (60, 60), (20, 60), (20, 20)])
    assert cycle_density(block, _roads([loop])) == cycle_density(block, _roads([subdivided]))


def test_empty_roads_zero() -> None:
    assert cycle_density(_block(4), _roads([])) == 0.0
    assert cycle_density(_block(4), None) == 0.0
