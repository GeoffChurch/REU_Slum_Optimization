import math
from typing import cast

import geopandas as gpd
import networkx as nx
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.budget import _noded_graph
from reblock.contracts import Block
from reblock.methods.loop_closure import (
    _bridge_tree,
    bridges_removed,
    greedy_close_loops,
    loop_candidates,
)

UTM = CRS.from_epsg(32734)


def _path_graph() -> nx.Graph:
    g: nx.Graph = nx.Graph()
    g.add_edges_from([((0.0, 0.0), (1.0, 0.0)), ((1.0, 0.0), (2.0, 0.0)),
                      ((2.0, 0.0), (3.0, 0.0))])
    return g


def _cycle_graph() -> nx.Graph:
    g: nx.Graph = nx.Graph()
    g.add_edges_from([((0.0, 0.0), (10.0, 0.0)), ((10.0, 0.0), (10.0, 10.0)),
                      ((10.0, 10.0), (0.0, 10.0)), ((0.0, 10.0), (0.0, 0.0))])
    return g


# --- _bridge_tree ------------------------------------------------------------------------------

def test_bridge_tree_path_is_all_bridges() -> None:
    g = _path_graph()
    comp_of, tree = _bridge_tree(g)
    assert len(set(comp_of.values())) == g.number_of_nodes()  # every node its own component
    assert tree.number_of_edges() == g.number_of_nodes() - 1


def test_bridge_tree_cycle_is_one_component_no_bridges() -> None:
    g = _cycle_graph()
    comp_of, tree = _bridge_tree(g)
    assert len(set(comp_of.values())) == 1
    assert tree.number_of_edges() == 0


# --- bridges_removed -----------------------------------------------------------------------

def test_bridges_removed_counts_bridges_on_path() -> None:
    g = _path_graph()
    comp_of, tree = _bridge_tree(g)
    assert bridges_removed(comp_of, tree, (0.0, 0.0), (3.0, 0.0)) == 3


def test_bridges_removed_is_zero_within_same_2ecc() -> None:
    g = _cycle_graph()
    comp_of, tree = _bridge_tree(g)
    assert bridges_removed(comp_of, tree, (0.0, 0.0), (10.0, 0.0)) == 0


# --- greedy_close_loops --------------------------------------------------------------------
# Fixture: a street (0,0)-(100,0) with 4 tree "teeth" hanging off it (T1 x=10, T2 x=12, T3 x=50,
# T4 x=54, all height 40). No loops in base_roads -- every edge is a bridge. Three candidate
# connectors across the tooth-tops:
#   cand1: (10,40)-(12,40), len=2   -- closes T1/street[10-12]/T2, removes 3 bridges, score 1.5
#   cand2: (50,40)-(54,40), len=4   -- closes T3/street[50-54]/T4, removes 3 bridges, score 0.75
#   cand3: (12,40)-(50,40), len=38  -- long-range, initially removes 3 bridges, score ~0.079
# cand1 has the highest bridges-removed-per-metre and is independent of cand2/cand3, so it is
# always picked first regardless of input order.

def _streets() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[LineString([(0, 0), (100, 0)])], crs=UTM)


def _teeth_roads() -> gpd.GeoDataFrame:
    teeth = [(10, 0, 10, 40), (12, 0, 12, 40), (50, 0, 50, 40), (54, 0, 54, 40)]
    return gpd.GeoDataFrame(
        geometry=[LineString([(x0, y0), (x1, y1)]) for x0, y0, x1, y1 in teeth], crs=UTM)


def _candidates() -> list[tuple[LineString, tuple[float, float], tuple[float, float]]]:
    cand1 = (LineString([(10, 40), (12, 40)]), (10.0, 40.0), (12.0, 40.0))
    cand2 = (LineString([(50, 40), (54, 40)]), (50.0, 40.0), (54.0, 40.0))
    cand3 = (LineString([(12, 40), (50, 40)]), (12.0, 40.0), (50.0, 40.0))
    return [cand2, cand3, cand1]     # deliberately NOT in score order


def test_greedy_close_loops_result_is_superset_of_base_roads() -> None:
    base = _teeth_roads()
    out = greedy_close_loops(base, _streets(), _candidates(), budget_m=None, max_loops=2)
    out_wkt = {ls.wkt for ls in out}
    assert {g.wkt for g in base.geometry}.issubset(out_wkt)


def test_greedy_close_loops_tiny_budget_adds_only_the_best_scoring_loop() -> None:
    base = _teeth_roads()
    out = greedy_close_loops(base, _streets(), _candidates(), budget_m=2.5, max_loops=20)
    added = [ls for ls in out if ls.wkt not in {g.wkt for g in base.geometry}]
    assert len(added) == 1
    assert added[0].equals(LineString([(10, 40), (12, 40)]))   # cand1: highest score (1.5)


def test_greedy_close_loops_budget_caps_total_added_length() -> None:
    base = _teeth_roads()
    out = greedy_close_loops(base, _streets(), _candidates(), budget_m=6.0, max_loops=20)
    added = [ls for ls in out if ls.wkt not in {g.wkt for g in base.geometry}]
    assert sum(ls.length for ls in added) <= 6.0
    # cand1 (len 2) + cand2 (len 4) fit exactly; cand3 (len 38) would blow the budget.
    assert len(added) == 2


def test_greedy_close_loops_max_loops_caps_the_count() -> None:
    base = _teeth_roads()
    out = greedy_close_loops(base, _streets(), _candidates(), budget_m=None, max_loops=1)
    added = [ls for ls in out if ls.wkt not in {g.wkt for g in base.geometry}]
    assert len(added) == 1
    out2 = greedy_close_loops(base, _streets(), _candidates(), budget_m=None, max_loops=2)
    added2 = [ls for ls in out2 if ls.wkt not in {g.wkt for g in base.geometry}]
    assert len(added2) == 2


# --- loop_candidates -------------------------------------------------------------------------
# Fixture: a 4x1 row of unit-square parcels [0,4]x[0,1], street along the bottom (0,0)-(4,0).
# base_roads is a TREE (every edge a bridge): two spurs (1,0)-(1,1) and (3,0)-(3,1). The parcel
# boundary graph (from block.parcels) has grid nodes at every integer (x,y) in [0,4]x{0,1}, so
# _snap can route a buildable connector straight across the tooth-tops (1,1)-(2,1)-(3,1), len=2,
# closing a loop with the base tree path (1,1)-(1,0)-(3,0)-(3,1), len=4 -> perimeter 6.

def _gap_parcels() -> gpd.GeoDataFrame:
    polys = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i in range(4)]
    return gpd.GeoDataFrame({"parcel_id": list(range(4))}, geometry=polys, crs=UTM)


def _gap_streets() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[LineString([(0, 0), (4, 0)])], crs=UTM)


def _gap_block() -> Block:
    parcels = _gap_parcels()
    boundary = cast(Polygon, parcels.geometry.union_all())
    return Block(block_id="gap", crs=UTM, boundary=boundary, parcels=parcels,
                streets=_gap_streets())


def _gap_roads() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        geometry=[LineString([(1, 0), (1, 1)]), LineString([(3, 0), (3, 1)])], crs=UTM)


def _weighted_road_graph(base_roads: gpd.GeoDataFrame, streets: gpd.GeoDataFrame) -> nx.Graph:
    g = _noded_graph(base_roads, streets)
    for u, v in g.edges():
        g[u][v]["len"] = math.hypot(u[0] - v[0], u[1] - v[1])
    return g


def test_loop_candidates_returns_gap_closing_connectors_meeting_the_perimeter_floor() -> None:
    block = _gap_block()
    base = _gap_roads()
    min_len = 5.0
    cands = loop_candidates(base, block, search_radius_m=2.5, min_loop_len_m=min_len, snap_lam=2.0)
    assert cands != []
    g = _weighted_road_graph(base, block.streets)
    node_set = set(g.nodes())
    for connector, u, v in cands:
        assert u in node_set and v in node_set     # endpoints are base road-graph nodes
        gap_len = nx.shortest_path_length(g, u, v, weight="len")
        assert connector.length + gap_len >= min_len - 1e-9   # implied loop perimeter >= floor


def test_loop_candidates_min_loop_len_past_block_size_returns_empty() -> None:
    block = _gap_block()
    base = _gap_roads()
    cands = loop_candidates(base, block, search_radius_m=2.5, min_loop_len_m=1000.0, snap_lam=2.0)
    assert cands == []


def test_loop_candidates_no_pair_within_radius_returns_empty() -> None:
    block = _gap_block()
    base = _gap_roads()
    cands = loop_candidates(base, block, search_radius_m=0.05, min_loop_len_m=1.0, snap_lam=2.0)
    assert cands == []
