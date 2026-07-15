import math
from dataclasses import replace
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from pyproj import CRS
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.clearance import (
    ClearanceReblocker,
    _edge_weights,
    _greedy_reblock,
    _node_clearance,
    _relax_depth,
    _sigmoid,
)
from reblock.methods.substrates import (
    GridSubstrate,
    PrebuiltSubstrate,
    RoutingGraph,
    _build_grid,
)


def test_sigmoid_is_bounded_and_symmetric() -> None:
    assert _sigmoid(0.0) == pytest.approx(0.5)
    assert _sigmoid(6.0) == pytest.approx(1.0, abs=1e-2)
    assert _sigmoid(-6.0) == pytest.approx(0.0, abs=1e-2)
    assert _sigmoid(3.0) + _sigmoid(-3.0) == pytest.approx(1.0)
    # never saturates to exactly 0/1 (weights stay finite), and no overflow at extreme s
    assert 0.0 < _sigmoid(-800.0) < _sigmoid(800.0) < 1.0


def test_build_grid_is_8_connected_and_inside_boundary() -> None:
    # contains_xy is strict (excludes the boundary), so a 4x4 box at res=1 gives interior nodes
    # {1,2,3}x{1,2,3}; the center (2,2) is a true interior node with 8 neighbours.
    boundary = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    pts, rows, cols, edist = _build_grid(boundary, 1.0)
    assert len(pts) > 0
    assert all(boundary.contains(Point(p)) for p in pts)   # strict, matches contains_xy
    # edges are symmetric (both directions present) and lengths are 1 or sqrt(2)
    assert len(rows) == len(cols) == len(edist)
    assert set(np.round(np.unique(edist), 6)) <= {1.0, round(math.sqrt(2.0), 6)}
    undirected = {frozenset((int(a), int(b))) for a, b in zip(rows, cols, strict=True)}
    assert len(undirected) * 2 == len(rows)  # every undirected edge stored both ways
    # an interior node has 8 neighbours
    tree = cKDTree(pts)
    center = int(tree.query([2.0, 2.0])[1])
    assert int((rows == center).sum()) == 8


def test_node_clearance_is_euclidean_when_unweighted() -> None:
    pts = np.array([[0.0, 0.0], [5.0, 0.0]])
    buildings = np.array([[0.0, 0.0]])
    radii = np.zeros(1)
    clear = _node_clearance(pts, buildings, radii)
    # node ON the building -> eps; node 5 away -> 5 + eps
    assert clear[0] == pytest.approx(0.3)
    assert clear[1] == pytest.approx(5.3)


def test_node_clearance_weighted_radius_shrinks_clearance() -> None:
    pts = np.array([[5.0, 0.0]])
    buildings = np.array([[0.0, 0.0]])
    plain = _node_clearance(pts, buildings, np.zeros(1))
    weighted = _node_clearance(pts, buildings, np.array([3.0]))  # radius-3 footprint
    assert weighted[0] < plain[0]
    assert weighted[0] == pytest.approx(5.0 - 3.0 + 0.3)


def test_node_clearance_no_buildings_is_uniform() -> None:
    pts = np.array([[0.0, 0.0], [10.0, 10.0]])
    clear = _node_clearance(pts, np.empty((0, 2)), np.zeros(0))
    assert clear[0] == clear[1]  # uniform -> straight regardless of t


def test_repulsion_bends_the_path_around_buildings() -> None:
    # A straight route (t≈0) crosses a vertical wall of buildings; a repelled route (t≈1)
    # bows away and stays farther from them, at >= the straight length.
    boundary = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    buildings = np.array([[5.0, 3.0], [5.0, 5.0], [5.0, 7.0]])
    pts, rows, cols, edist = _build_grid(boundary, 0.5)
    radii = np.zeros(len(buildings))
    tree = cKDTree(pts)
    src = int(tree.query([5.0, 9.0])[1])
    dst = int(tree.query([5.0, 1.0])[1])

    def route(t: float) -> tuple[float, float]:
        w = _edge_weights(pts, rows, cols, edist, buildings, radii, t)
        csr = csr_matrix((w, (rows, cols)), shape=(len(pts), len(pts)))
        _d, pred, _s = dijkstra(csr, indices=[src], return_predecessors=True, min_only=True)
        node, path = dst, [dst]
        while pred[node] >= 0:
            node = int(pred[node])
            path.append(node)
        line = LineString([tuple(pts[k]) for k in path])
        min_clear = min(
            Point(cast(tuple[float, float], tuple(b))).distance(line) for b in buildings
        )
        return float(line.length), float(min_clear)

    len_straight, clear_straight = route(_sigmoid(-6.0))
    len_repelled, clear_repelled = route(_sigmoid(6.0))
    assert clear_straight < clear_repelled           # repelled path keeps farther from buildings
    assert len_repelled >= len_straight - 1e-9        # ...at no less than the straight length


UTM = CRS.from_epsg(32643)


def _column_block(h: int) -> Block:
    """A 1-wide, h-tall column of unit parcels with street frontage only on the bottom edge ->
    access depth 1..h from the street upward. parcel_id == row index (bottom = 0)."""
    polys = [Polygon([(0, j), (1, j), (1, j + 1), (0, j + 1)]) for j in range(h)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(h))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, unary_union(polys))
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs=UTM)
    return Block(block_id="col", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_relax_depth_matches_full_recompute() -> None:
    # A single street-connected road up the column should, incrementally, reproduce exactly what
    # parcel_access_layers computes from scratch for that road.
    block = _column_block(6)
    geoms = list(block.parcels.geometry)
    adj = parcel_adjacency(geoms, STREET_TOL)
    depth = parcel_access_layers(block, None, adj=adj).to_numpy().astype(float)
    assert list(depth) == [1, 2, 3, 4, 5, 6]  # sanity: deep column

    road = gpd.GeoDataFrame(geometry=[LineString([(0.5, 0.0), (0.5, 6.0)])], crs=UTM)
    served = [int(p) for p in STRtree(geoms).query(
        road.geometry.iloc[0], predicate="dwithin", distance=STREET_TOL)]
    _relax_depth(depth, adj, served)

    naive = parcel_access_layers(block, road, adj=adj).to_numpy().astype(float)
    assert list(depth) == list(naive)
    assert max(depth) == 1.0  # every parcel now fronts the street-connected road


def test_relax_depth_matches_recompute_on_disconnected_component() -> None:
    # The relax equals a full recompute ONLY when the base array pins unreached parcels to a
    # high sentinel (len+1). This locks in that precondition and shows the default-seeded base
    # (unreached = max(reached)+1) diverges -- which is exactly why Task 3's greedy seeds with
    # unreached_depth=len+1. Row A (3 parcels) fronts the street; column B (5 parcels, disjoint
    # from A) is unreached until a road connects its near end.
    a = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i in range(3)]
    b = [Polygon([(0, y), (1, y), (1, y + 1), (0, y + 1)]) for y in range(5, 10)]  # gap at y=1..5
    polys = a + b
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, unary_union(polys))
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (3, 0)])], crs=UTM)
    block = Block(block_id="disc", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)
    adj = parcel_adjacency(cast(list[BaseGeometry], polys), STREET_TOL)
    n = len(polys)

    # street-connected road reaching ONLY B's near end (top at y=5.4 -> >0.5 from B[1] at y=6)
    road = gpd.GeoDataFrame(geometry=[LineString([(0.5, 0.0), (0.5, 5.4)])], crs=UTM)
    served = [int(p) for p in STRtree(polys).query(
        road.geometry.iloc[0], predicate="dwithin", distance=STREET_TOL)]
    naive = parcel_access_layers(block, road, adj=adj).to_numpy().astype(float)

    seeded = parcel_access_layers(
        block, None, adj=adj, unreached_depth=n + 1).to_numpy().astype(float)
    _relax_depth(seeded, adj, served)
    assert list(seeded) == list(naive)                 # correct precondition -> exact

    default_base = parcel_access_layers(block, None, adj=adj).to_numpy().astype(float)
    _relax_depth(default_base, adj, served)
    assert list(default_base) != list(naive)           # default seeding -> falsely shallow


def _column_block_with_buildings(h: int) -> Block:
    block = _column_block(h)
    pts = [g.representative_point() for g in block.parcels.geometry]
    block_bp = gpd.GeoDataFrame(geometry=pts, crs=UTM)
    return Block(block_id="colb", crs=UTM, boundary=block.boundary, parcels=block.parcels,
                 streets=block.streets, building_points=block_bp)


def test_default_substrate_is_chord_diag() -> None:
    m = ClearanceReblocker(repulsion=0.0)
    assert m.substrate.tag == "chord_diag"
    p = m.propose(_column_block_with_buildings(6))
    assert p.proposal_id == "clearance:chord_diag:r0:d2:mr400"
    assert p.params["substrate"] == "chord_diag"
    after = parcel_access_layers(_column_block_with_buildings(6), p.roads).to_numpy()
    assert int(after.max()) <= 2


def test_greedy_reblock_achieves_depth_target() -> None:
    block = _column_block_with_buildings(8)  # depth 1..8
    graph = GridSubstrate(res=0.5).build(block)
    roads, params = _greedy_reblock(block, graph, t=0.5, depth_target=2, max_roads=400,
                                    radii=np.zeros(len(block.building_points)))
    assert len(roads) > 0
    after = parcel_access_layers(block, roads).to_numpy()
    assert int(after.max()) <= 2
    assert params["max_roads_hit"] is False


def test_greedy_reblock_returns_empty_when_already_shallow() -> None:
    block = _column_block_with_buildings(2)  # depth 1..2, target 2 -> nothing to do
    graph = GridSubstrate(res=0.5).build(block)
    roads, params = _greedy_reblock(block, graph, t=0.5, depth_target=2, max_roads=400,
                                    radii=np.zeros(len(block.building_points)))
    assert len(roads) == 0
    assert params["roads"] == 0


def test_propose_is_deterministic_and_leaves_rng_untouched() -> None:
    block = _column_block_with_buildings(8)
    np.random.seed(123)
    state = np.random.get_state()[1].tolist()
    p1 = ClearanceReblocker(depth_target=2, substrate=GridSubstrate(res=0.5)).propose(block)
    p2 = ClearanceReblocker(depth_target=2, substrate=GridSubstrate(res=0.5)).propose(block)
    assert np.random.get_state()[1].tolist() == state
    assert p1.roads is not None and p2.roads is not None and len(p1.roads) > 0
    assert [g.wkt for g in p1.roads.geometry] == [g.wkt for g in p2.roads.geometry]


def test_propose_metadata_and_identity() -> None:
    m = ClearanceReblocker(substrate=GridSubstrate(res=0.75), repulsion=2.0,
                           depth_target=3, max_roads=50)
    assert m.identity == ("clearance", ("grid", 0.75), 2.0, 3, 50)
    p = m.propose(_column_block_with_buildings(4))
    assert p.method == "clearance"
    assert p.proposal_id == "clearance:grid:r2:d3:mr50"
    assert p.block_identity == _column_block_with_buildings(4).identity
    assert p.params["repulsion"] == 2.0 and p.params["depth_target"] == 3


def test_distinct_repulsions_get_distinct_proposal_identity() -> None:
    # so access_after / geometric_after (keyed on the proposal) never collide across the knob
    # (res=0.5: the fixture's unit-width column needs a sub-1 grid resolution, like every other
    # _column_block_with_buildings test here -- the default res=1.5 is for real meter-scale blocks.
    # source_content_hash gives the block a non-None identity, matching the real (Source-loaded)
    # blocks this collision concern is actually about -- Block.identity is None for the bare
    # synthetic fixture, which would make Proposal.identity collapse to None regardless of
    # proposal_id and the second assertion vacuously fail.)
    block = replace(_column_block_with_buildings(6), source_content_hash="test-hash")
    a = ClearanceReblocker(repulsion=-6.0, substrate=GridSubstrate(res=0.5)).propose(block)
    b = ClearanceReblocker(repulsion=6.0, substrate=GridSubstrate(res=0.5)).propose(block)
    assert a.proposal_id != b.proposal_id
    assert a.identity != b.identity


def test_prebuilt_substrate_makes_proposal_uncacheable_even_on_a_real_block() -> None:
    # A PrebuiltSubstrate is an ad-hoc graph (identity None, fixed tag "prebuilt"), so proposal_id
    # can't distinguish two prebuilt graphs -- its eval must bypass the cache too, else the second
    # graph's metrics get served from the first's cache entry. Proposal.identity must be None even
    # on a block with a real (Source) identity.
    block = replace(_column_block_with_buildings(6), source_content_hash="test-hash")
    assert block.identity is not None
    graph = GridSubstrate(res=0.5).build(block)
    method = ClearanceReblocker(substrate=PrebuiltSubstrate(graph), depth_target=2)
    proposal = method.propose(block)
    assert method.identity is None       # Method already uncacheable (substrate identity None)
    assert proposal.identity is None     # ...and now the Proposal too (block_identity dropped)


def test_propose_achieves_target_on_real_block() -> None:
    # bare (not `tests.scoring_fixtures`): pyproject.toml deliberately keeps tests/ without an
    # __init__.py (a tests.__init__ would collide with ext/topology's own "tests" package under
    # mypy), so this file is a top-level module -- matching test_scoring_equivalence.py /
    # test_arterial.py's existing imports of the same fixture.
    from scoring_fixtures import _block_1808

    block = _block_1808()
    m = ClearanceReblocker(depth_target=2, substrate=GridSubstrate(res=0.75))
    roads = m.propose(block).roads
    assert roads is not None
    after = parcel_access_layers(block, roads).to_numpy()
    assert int(after.max()) <= 2  # invariant holds whether or not roads were needed


def test_clearance_module_is_in_derivation_modules() -> None:
    # so a change to the algorithm busts the memoized propose() cache (like dijkstra/mesh/arterial)
    from reblock.derive_graph import _DERIVATION_MODULES
    assert any(p.name == "clearance.py" and p.parent.name == "methods"
               for p in _DERIVATION_MODULES)


def test_clearance_method_yaml_instantiates_with_defaults() -> None:
    conf_dir = str(Path(__file__).resolve().parents[2] / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config", overrides=["method=clearance"])
    method = instantiate(cfg.method)
    assert isinstance(method, ClearanceReblocker)
    assert method.identity == ("clearance", ("chord_diag",), 0.0, 2, 400)


def test_clearance_registered_in_compare_all_methods() -> None:
    conf_dir = str(Path(__file__).resolve().parents[2] / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="compare_config")
    method = instantiate(cfg.all_methods["clearance"])
    assert isinstance(method, ClearanceReblocker)


def test_edge_weights_3point_sees_a_midspan_building() -> None:
    # A long edge whose two endpoints sit in the clear but whose MIDPOINT skims a building must
    # read as more expensive than a plain endpoint-only average would make it. Endpoints at
    # (0,0) and (10,0) are far from the single building at (5,3); the midpoint (5,0) is close.
    pts = np.array([[0.0, 0.0], [10.0, 0.0]])
    rows = np.array([0, 1])
    cols = np.array([1, 0])
    edist = np.array([10.0, 10.0])
    buildings = np.array([[5.0, 3.0]])          # nearest to the midpoint, far from endpoints
    radii = np.zeros(1)
    t = _sigmoid(6.0)                            # high repulsion -> clearance dominates cost
    w = _edge_weights(pts, rows, cols, edist, buildings, radii, t)
    # endpoint-only weight for comparison: mean of the two endpoint node costs * length
    clear_ends = _node_clearance(pts, buildings, radii)
    node_cost_ends = (1.0 - t) + t / clear_ends
    endpoint_only = 10.0 * 0.5 * (node_cost_ends[0] + node_cost_ends[1])
    assert w[0] == pytest.approx(w[1])          # symmetric COO
    assert w[0] > endpoint_only                 # the midpoint building raised the cost


def test_propose_routes_through_memoized_derivation() -> None:
    # region_reblock / compare call derivations.propose; a cacheable block returns identical roads
    from scoring_fixtures import _block_1808  # bare: see the note above on the same import

    from reblock.derivations import propose
    block = _block_1808()
    m = ClearanceReblocker(depth_target=2, substrate=GridSubstrate(res=0.75))
    r1 = propose(m, block).roads
    r2 = propose(m, block).roads
    assert r1 is not None and r2 is not None
    assert [g.wkt for g in r1.geometry] == [g.wkt for g in r2.geometry]


def test_substrate_config_group_instantiates_each() -> None:
    conf_dir = str(Path(__file__).resolve().parents[2] / "conf")
    expected = {"grid": ("grid", 1.5), "chord_diag": ("chord_diag",),
                "theta_spanner": ("theta_spanner", 6), "cdt_gap": ("cdt_gap",)}
    for name, ident in expected.items():
        with initialize_config_dir(version_base=None, config_dir=conf_dir):
            cfg = compose(config_name="config", overrides=[f"substrate={name}"])
        sub = instantiate(cfg.substrate)
        assert sub.identity == ident


def test_clearance_method_defaults_to_chord_diag_substrate() -> None:
    conf_dir = str(Path(__file__).resolve().parents[2] / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config", overrides=["method=clearance"])
    method = instantiate(cfg.method)
    assert isinstance(method, ClearanceReblocker)
    assert method.substrate.tag == "chord_diag"


def test_compare_registers_clearance_and_grid_variant() -> None:
    conf_dir = str(Path(__file__).resolve().parents[2] / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="compare_config")
    assert instantiate(cfg.all_methods["clearance"]).substrate.tag == "chord_diag"
    assert instantiate(cfg.all_methods["clearance_grid"]).substrate.tag == "grid"


def test_default_chord_diag_propose_is_deterministic() -> None:
    # The committed gallery default -- its determinism is load-bearing.
    block = _column_block_with_buildings(8)
    p1 = ClearanceReblocker().propose(block)
    p2 = ClearanceReblocker().propose(block)
    assert p1.roads is not None and p2.roads is not None and len(p1.roads) > 0
    assert p1.proposal_id.startswith("clearance:chord_diag:")
    assert [g.wkt for g in p1.roads.geometry] == [g.wkt for g in p2.roads.geometry]


def test_prebuilt_substrate_makes_the_method_uncacheable() -> None:
    # PrebuiltSubstrate.identity is None -> ClearanceReblocker.identity must propagate None so
    # derive() bypasses the memoized propose (distinct ad-hoc graphs must not key-collide).
    g = RoutingGraph(pts=np.array([[0.0, 0.0], [1.0, 0.0]]), rows=np.array([0, 1]),
                     cols=np.array([1, 0]), edist=np.array([1.0, 1.0]), net_tol=0.5)
    assert ClearanceReblocker(substrate=PrebuiltSubstrate(g)).identity is None
    # a named substrate stays cacheable (non-None identity)
    assert ClearanceReblocker().identity is not None
