import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import geopandas as gpd
import networkx as nx
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock import derive_graph
from reblock.budget import _noded_graph, commute_ratio
from reblock.contracts import Block, Proposal
from reblock.methods.loop_closure import (
    LoopClosureIdentity,
    LoopClosureRefiner,
    _bridge_tree,
    _subsample_pairs,
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


def test_greedy_close_loops_min_bridges_per_m_forces_early_stop() -> None:
    # cand3 (len 38) only ever removes 3 bridges -> score ~0.079, well under a 0.1 floor; cand1
    # (score 1.5) and cand2 (score 0.75) both clear it. budget_m=None/max_loops=20 so nothing but
    # the efficiency floor itself can stop the greedy early.
    base = _teeth_roads()
    baseline = greedy_close_loops(base, _streets(), _candidates(), budget_m=None, max_loops=20)
    added_baseline = [ls for ls in baseline if ls.wkt not in {g.wkt for g in base.geometry}]
    assert len(added_baseline) == 3   # sanity: all three candidates go in with no floor

    out = greedy_close_loops(base, _streets(), _candidates(), budget_m=None, max_loops=20,
                             min_bridges_per_m=0.1)
    added = [ls for ls in out if ls.wkt not in {g.wkt for g in base.geometry}]
    assert len(added) < len(added_baseline)
    assert len(added) == 2   # cand1 + cand2 only; cand3's ~0.079 score falls below the 0.1 floor


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


# Fixture: the SAME 4x1 block as above, but base_roads is a CLOSED RING around it -- left side
# (0,0)-(0,1), the top (0,1)-(1,1)-(2,1)-(3,1)-(4,1), and the right side (4,1)-(4,0) -- which,
# noded together with the bottom street (0,0)-(4,0), forms one 7-node cycle with NO bridges (every
# node is 2-edge-connected to every other). At search_radius_m=2.5, `query_pairs` finds plenty of
# node pairs and `_snap` realizes connectors for all of them -- so the radius/snap path is not what
# empties the result. But because the base is already a closed, short ring, every pair's implied
# loop perimeter (connector length + the ring's own shortest gap) tops out at 6.0 m (verified via
# `_weighted_road_graph` below across all candidates at a permissive floor); a `min_loop_len_m`
# past that (6.5) rejects every one of them on the geometric-loop-floor check specifically.

def _closed_loop_roads() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0), (0, 1)]),
            LineString([(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]),
            LineString([(4, 1), (4, 0)]),
        ], crs=UTM)


def test_loop_candidates_closed_loop_rejected_by_perimeter_floor_not_radius() -> None:
    block = _gap_block()
    base = _closed_loop_roads()
    g = _weighted_road_graph(base, block.streets)
    assert list(nx.bridges(g)) == []              # fully 2-edge-connected: no gaps to bridge
    assert nx.number_connected_components(g) == 1
    # Tiny floor: query_pairs finds pairs and _snap succeeds -> non-empty. This pins down that the
    # radius/snap path is working, so the empty result below can only come from the floor check.
    tiny = loop_candidates(base, block, search_radius_m=2.5, min_loop_len_m=1.0, snap_lam=2.0)
    assert tiny != []
    # Raised floor, same radius and base: every candidate found above is rejected because its own
    # perimeter falls short of min_loop_len_m -- not because no pairs were found.
    raised = loop_candidates(base, block, search_radius_m=2.5, min_loop_len_m=6.5, snap_lam=2.0)
    assert raised == []


# --- _subsample_pairs / max_candidates cap ------------------------------------------------------
# A uniform stride bounds pair volume WITHOUT the nearest-k bias that starved dense meshes (kept
# only the short, floor-rejected pairs). It must preserve the distance SPREAD -- keep long pairs,
# not just short ones -- so real (min_loop_len-clearing) loop-closers survive the cap.

def test_subsample_pairs_noop_when_within_cap() -> None:
    pairs = [(0, 1), (0, 2), (1, 2)]
    assert _subsample_pairs(pairs, max_candidates=10) == pairs


def test_subsample_pairs_shrinks_volume_to_about_cap() -> None:
    pairs = [(0, j) for j in range(1, 101)]           # 100 sorted pairs
    bounded = _subsample_pairs(pairs, max_candidates=10)
    assert 0 < len(bounded) <= 10
    for p in bounded:
        assert p in pairs                             # subset, no fabricated pairs


def test_subsample_pairs_preserves_distance_spread_not_just_shortest() -> None:
    # Regression guard for the kNN starvation bug: the cap must retain pairs from ACROSS the list
    # (which, index-sorted, spans the node-coordinate space), not collapse to the shortest/first
    # few.
    pairs = [(0, j) for j in range(1, 1001)]          # 1000 sorted pairs
    bounded = _subsample_pairs(pairs, max_candidates=50)
    seconds = [j for _i, j in bounded]
    assert min(seconds) < 100 and max(seconds) > 900  # spans low AND high, not just the head


def test_subsample_pairs_nonpositive_cap_clamps_to_one_no_crash() -> None:
    # max_candidates=0 must not ZeroDivisionError, and a negative must not reverse-stride -- both
    # are clamped to a cap of 1 (the floor the old nearest-k helper carried).
    pairs = [(0, j) for j in range(1, 11)]
    assert _subsample_pairs(pairs, max_candidates=0) == [pairs[0]]
    assert _subsample_pairs(pairs, max_candidates=-5) == [pairs[0]]


def test_loop_candidates_max_candidates_caps_pairs_and_stays_valid() -> None:
    block = _gap_block()
    base = _gap_roads()
    uncapped = loop_candidates(base, block, search_radius_m=2.5, min_loop_len_m=1.0, snap_lam=2.0)
    capped = loop_candidates(
        base, block, search_radius_m=2.5, min_loop_len_m=1.0, snap_lam=2.0, max_candidates=4)
    assert len(capped) <= len(uncapped)
    assert capped != []                         # the cap still lets the one real loop through
    g = _weighted_road_graph(base, block.streets)
    node_set = set(g.nodes())
    for connector, u, v in capped:
        assert u in node_set and v in node_set
        gap_len = nx.shortest_path_length(g, u, v, weight="len")
        assert connector.length + gap_len >= 1.0 - 1e-9


def test_loop_candidates_max_candidates_none_is_unbounded_default() -> None:
    block = _gap_block()
    base = _gap_roads()
    explicit_none = loop_candidates(
        base, block, search_radius_m=2.5, min_loop_len_m=1.0, snap_lam=2.0, max_candidates=None)
    default = loop_candidates(base, block, search_radius_m=2.5, min_loop_len_m=1.0, snap_lam=2.0)
    assert {c.wkt for c, _u, _v in explicit_none} == {c.wkt for c, _u, _v in default}


# --- LoopClosureRefiner ------------------------------------------------------------------------
# Reuses the "gap" fixtures above: `_gap_block()`/`_gap_roads()` is a TREE (street + two spurs,
# every edge a bridge) with one admissible loop-closing connector across the tooth-tops
# ((1,1)-(2,1)-(3,1), len=2, perimeter 6 against the floor of 5.0); `_closed_loop_roads()` is
# already fully 2-edge-connected (no bridges to remove -- no admissible gap).

class _RefinerKw(TypedDict):
    """A precise key/type map for `**_REFINER_KW` call sites below -- a plain `dict[str, float]`
    would widen its value type and make mypy treat the unpacked kwargs as a possible (invalid)
    source for `LoopClosureRefiner`'s `int | None`-typed `max_candidates` field too."""

    search_radius_m: float
    min_loop_len_m: float
    snap_lam: float


_REFINER_KW: _RefinerKw = {"search_radius_m": 2.5, "min_loop_len_m": 5.0, "snap_lam": 2.0}

# A dedicated fixture for the commute_ratio assertion: taller (1x2, not 1x1) parcels than
# `_gap_parcels()` so each parcel's centroid is strictly closer to its spur than to the street
# (0.5 m vs 1.0 m) -- `_gap_parcels()`'s 1x1 squares are EXACTLY tied (0.5 m to both), and
# `commute_ratio`'s `STRtree.nearest` line-proximity pick breaks that tie in an
# implementation-defined way that (here) always lands on a street-only edge, making every parcel
# skip (both endpoints already street nodes) and the metric read 0.0 for every road set.


def _ratio_parcels() -> gpd.GeoDataFrame:
    polys = [Polygon([(i, 0), (i + 1, 0), (i + 1, 2), (i, 2)]) for i in range(4)]
    return gpd.GeoDataFrame({"parcel_id": list(range(4))}, geometry=polys, crs=UTM)


def _ratio_streets() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[LineString([(0, 0), (4, 0)])], crs=UTM)


def _ratio_block() -> Block:
    parcels = _ratio_parcels()
    boundary = cast(Polygon, parcels.geometry.union_all())
    return Block(block_id="ratio-gap", crs=UTM, boundary=boundary, parcels=parcels,
                streets=_ratio_streets())


def _ratio_base_roads() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        geometry=[LineString([(1, 0), (1, 2)]), LineString([(3, 0), (3, 2)])], crs=UTM)


@dataclass
class _FakeBase:
    """A minimal `Method` stand-in that always returns a fixed `Proposal`, spying on call count so
    the `prior` pass-through test can assert `propose` is bypassed."""

    proposal: Proposal
    ident: object = ("fake", 1)
    calls: int = 0

    @property
    def identity(self) -> object:
        return self.ident

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del block, prior
        self.calls += 1
        return self.proposal


def _base_proposal(block: Block, roads: gpd.GeoDataFrame, *,
                   proposal_id: str = "tree-base") -> Proposal:
    return Proposal(block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
                    proposal_id=proposal_id, method="tree",
                    block_identity=("test", block.block_id))


# budget_frac large enough that budget_m (= budget_frac * base road length) never binds on any
# fixture below -- the refiner-level stand-in for the old budget_m=None ("unlimited") calls.
_UNLIMITED_BUDGET_FRAC = 10.0


def test_loop_closure_refiner_default_max_candidates_is_the_plateau() -> None:
    # The ρ-plateau default -- documents the calibrated cap so a silent regression to the old
    # starving value is caught.
    block = _gap_block()
    refiner = LoopClosureRefiner(base=_FakeBase(_base_proposal(block, _gap_roads())))
    assert refiner.max_candidates == 1500


def test_loop_closure_refiner_adds_loop_reduces_bridges_and_raises_commute_ratio() -> None:
    block = _ratio_block()
    base_roads = _ratio_base_roads()
    base_prop = _base_proposal(block, base_roads)
    refiner = LoopClosureRefiner(
        base=_FakeBase(base_prop), budget_frac=_UNLIMITED_BUDGET_FRAC, max_loops=5, **_REFINER_KW)
    out = refiner.propose(block)
    assert out.roads is not None
    before_bridges = len(list(nx.bridges(_noded_graph(base_roads, block.streets))))
    after_bridges = len(list(nx.bridges(_noded_graph(out.roads, block.streets))))
    assert after_bridges < before_bridges
    assert commute_ratio(block, out.roads) > commute_ratio(block, base_roads)


def test_loop_closure_refiner_budget_frac_caps_added_length() -> None:
    block = _gap_block()
    base_roads = _gap_roads()
    base_prop = _base_proposal(block, base_roads)
    base_wkts = {g.wkt for g in base_roads.geometry}
    base_len = float(base_roads.geometry.length.sum())

    tight = LoopClosureRefiner(base=_FakeBase(base_prop), budget_frac=0.5, max_loops=5,
                               **_REFINER_KW)
    out_tight = tight.propose(block)
    assert out_tight.roads is not None
    assert {g.wkt for g in out_tight.roads.geometry} == base_wkts   # connector (len 2) doesn't fit
                                                                     # (0.5 * base_len == 1.0 < 2)

    loose = LoopClosureRefiner(base=_FakeBase(base_prop), budget_frac=5.0, max_loops=5,
                               **_REFINER_KW)
    out_loose = loose.propose(block)
    assert out_loose.roads is not None
    added_len = float(out_loose.roads.geometry.length.sum() - base_roads.geometry.length.sum())
    assert 0.0 < added_len <= loose.budget_frac * base_len


def test_loop_closure_refiner_max_loops_caps_the_count() -> None:
    block = _gap_block()
    base_roads = _gap_roads()
    base_prop = _base_proposal(block, base_roads)
    base_wkts = {g.wkt for g in base_roads.geometry}

    zero = LoopClosureRefiner(base=_FakeBase(base_prop), budget_frac=_UNLIMITED_BUDGET_FRAC,
                              max_loops=0, **_REFINER_KW)
    out_zero = zero.propose(block)
    assert out_zero.roads is not None
    assert {g.wkt for g in out_zero.roads.geometry} == base_wkts

    one = LoopClosureRefiner(base=_FakeBase(base_prop), budget_frac=_UNLIMITED_BUDGET_FRAC,
                             max_loops=1, **_REFINER_KW)
    out_one = one.propose(block)
    assert out_one.roads is not None
    assert len(out_one.roads) - len(base_roads) == 1


def test_loop_closure_refiner_min_bridges_per_m_forces_early_stop() -> None:
    # The gap fixture's winning candidate ((1,1)-(2,1)-(3,1), len 2.0) removes 3 bridges -> score
    # 1.5 (verified directly against `greedy_close_loops`' ranking). A floor comfortably above that
    # (2.0) must reject it outright -- no loop added at all -- while a 0.0 floor accepts it, same
    # as `test_loop_closure_refiner_roads_are_superset_of_base_roads` below.
    block = _gap_block()
    base_roads = _gap_roads()
    base_prop = _base_proposal(block, base_roads)
    base_wkts = {g.wkt for g in base_roads.geometry}

    permissive = LoopClosureRefiner(
        base=_FakeBase(base_prop), budget_frac=_UNLIMITED_BUDGET_FRAC, max_loops=5,
        min_bridges_per_m=0.0, **_REFINER_KW)
    out_permissive = permissive.propose(block)
    assert out_permissive.roads is not None
    assert {g.wkt for g in out_permissive.roads.geometry} != base_wkts   # the one loop goes in

    strict = LoopClosureRefiner(
        base=_FakeBase(base_prop), budget_frac=_UNLIMITED_BUDGET_FRAC, max_loops=5,
        min_bridges_per_m=2.0, **_REFINER_KW)
    out_strict = strict.propose(block)
    assert out_strict.roads is not None
    assert {g.wkt for g in out_strict.roads.geometry} == base_wkts   # floor rejects it: no loop


def test_loop_closure_refiner_no_admissible_candidate_returns_base_unchanged() -> None:
    block = _gap_block()
    base_roads = _closed_loop_roads()
    base_prop = _base_proposal(block, base_roads)
    refiner = LoopClosureRefiner(
        base=_FakeBase(base_prop), budget_frac=_UNLIMITED_BUDGET_FRAC, max_loops=5,
        search_radius_m=2.5, min_loop_len_m=1.0, snap_lam=2.0)
    out = refiner.propose(block)
    assert out.roads is not None
    assert {g.wkt for g in out.roads.geometry} == {g.wkt for g in base_roads.geometry}


def test_loop_closure_refiner_prior_bypasses_base_propose() -> None:
    block = _gap_block()
    base_roads = _gap_roads()
    prior_prop = _base_proposal(block, base_roads, proposal_id="prior-base")
    unused_prop = Proposal(
        block_id=block.block_id, crs=block.crs,
        roads=gpd.GeoDataFrame(geometry=[], crs=block.crs), proposal_id="should-not-be-used",
        block_identity=("test", block.block_id))
    fake = _FakeBase(unused_prop)
    refiner = LoopClosureRefiner(base=fake, budget_frac=_UNLIMITED_BUDGET_FRAC, max_loops=5,
                                 **_REFINER_KW)
    out = refiner.propose(block, prior=prior_prop)
    assert fake.calls == 0
    assert out.roads is not None
    assert {g.wkt for g in base_roads.geometry}.issubset({g.wkt for g in out.roads.geometry})


def test_loop_closure_refiner_roads_are_superset_of_base_roads() -> None:
    block = _gap_block()
    base_roads = _gap_roads()
    base_prop = _base_proposal(block, base_roads)
    refiner = LoopClosureRefiner(
        base=_FakeBase(base_prop), budget_frac=_UNLIMITED_BUDGET_FRAC, max_loops=5, **_REFINER_KW)
    out = refiner.propose(block)
    assert out.roads is not None
    assert {g.wkt for g in base_roads.geometry}.issubset({g.wkt for g in out.roads.geometry})


def test_loop_closure_refiner_identity_folds_in_base_identity() -> None:
    base_prop = Proposal(block_id="b", crs=UTM, block_identity=("t", "b"))
    fake = _FakeBase(base_prop, ident=("fake", 1))
    refiner = LoopClosureRefiner(base=fake)
    ident = refiner.identity
    assert ident is not None
    assert isinstance(ident, LoopClosureIdentity)
    assert ident.base == fake.identity


def test_loop_closure_refiner_identity_none_when_base_identity_none() -> None:
    base_prop = Proposal(block_id="b", crs=UTM)
    fake = _FakeBase(base_prop, ident=None)
    refiner = LoopClosureRefiner(base=fake)
    assert refiner.identity is None


def test_loop_closure_refiner_identity_changes_with_params() -> None:
    base_prop = Proposal(block_id="b", crs=UTM, block_identity=("t", "b"))
    fake = _FakeBase(base_prop, ident=("fake", 1))
    r1 = LoopClosureRefiner(base=fake, budget_frac=0.10)
    r2 = LoopClosureRefiner(base=fake, budget_frac=0.20)
    assert r1.identity != r2.identity
    r3 = LoopClosureRefiner(base=fake, min_loop_len_m=10.0)
    r4 = LoopClosureRefiner(base=fake, min_loop_len_m=20.0)
    assert r3.identity != r4.identity
    r5 = LoopClosureRefiner(base=fake, max_candidates=1000)
    r6 = LoopClosureRefiner(base=fake, max_candidates=2000)
    assert r5.identity != r6.identity
    r7 = LoopClosureRefiner(base=fake, min_bridges_per_m=0.01)
    r8 = LoopClosureRefiner(base=fake, min_bridges_per_m=0.02)
    assert r7.identity != r8.identity


def test_loop_closure_registered_in_derivation_modules() -> None:
    import reblock.methods.loop_closure as lc
    expected = Path(lc.__file__).resolve()
    assert any(Path(p).resolve() == expected for p in derive_graph._DERIVATION_MODULES)
