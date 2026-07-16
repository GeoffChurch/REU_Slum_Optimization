from typing import cast

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon

from reblock.budget import building_radii, displacement
from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, street_connectivity
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods import arterial
from reblock.methods.arterial import (
    GreedyArterialReblocker,
    _anchor_points,
    _best_candidate,
    _candidate_chords,
    _deep_targets,
    _greedy_arterials,
    _planarize,
    _snap,
    _snap_graph,
)
from reblock.methods.dijkstra import _boundary_graph

UTM = CRS.from_epsg(32643)


def _grid_block(n: int) -> Block:
    polys, ids = [], []
    for i in range(n):
        for j in range(n):
            polys.append(Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]))
            ids.append(i * n + j)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="grid", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_anchor_points_sample_the_network_and_include_vertices() -> None:
    net = [LineString([(0.0, 0.0), (4.0, 0.0)])]
    pts = _anchor_points(net, n=4)
    assert (0.0, 0.0) in pts and (4.0, 0.0) in pts        # endpoints/vertices kept
    assert len(pts) >= 4 and pts == sorted(pts)           # sampled + deterministic order


def test_anchor_points_max_anchors_caps_and_default_matches_uncapped() -> None:
    # A network with many vertices (mimics a large block's boundary graph, e.g. block 5810's ~229
    # vertices) so max_anchors actually bounds the count instead of coincidentally landing on it.
    coords = [(float(i), 0.0 if i % 2 == 0 else 1.0) for i in range(40)]
    net = [LineString(coords)]
    uncapped = _anchor_points(net, n=8)
    # max_anchors=0 (default) must be byte-identical to today's behavior (every vertex + samples).
    assert _anchor_points(net, n=8, max_anchors=0) == uncapped
    assert _anchor_points(net, n=8) == uncapped
    capped = _anchor_points(net, n=8, max_anchors=8)
    assert len(capped) <= 15
    assert len(capped) < len(uncapped)


def test_deep_targets_are_the_deepest_parcels() -> None:
    block = _grid_block(5)                       # center parcel is deepest, full-boundary streets
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    targets = _deep_targets(block, None, k=1, adj=adj)
    assert len(targets) == 1
    assert Point(targets[0]).distance(Point(2.5, 2.5)) < 1.0   # near the 5x5 center


def test_candidate_chords_include_through_roads_and_spurs() -> None:
    anchors = [(0.0, 0.0), (4.0, 0.0)]
    targets = [(2.0, 2.0)]
    chords = _candidate_chords(anchors, targets)
    assert LineString([(0.0, 0.0), (4.0, 0.0)]) in chords            # through-road
    assert any(t == (2.0, 2.0) for c in chords for t in c.coords)    # a spur reaches the target
    assert chords == sorted(chords, key=lambda ls: ls.wkt)           # deterministic order


def test_snap_returns_a_boundary_following_street_connected_path() -> None:
    block = _grid_block(5)
    g = _boundary_graph(block.parcels)
    sg = _snap_graph(g)
    chord = LineString([(0.0, 2.5), (5.0, 2.5)])          # a horizontal cut across the grid
    path = _snap(chord, sg, lam=2.0)
    assert path is not None
    # every vertex of the snapped path is a boundary-graph node (buildable)
    assert all(_round(c) in set(g.nodes) for c in path.coords)


def _round(c: tuple[float, ...]) -> tuple[float, float]:
    return (round(c[0], 2), round(c[1], 2))


def test_best_candidate_reduce() -> None:
    # Direct, geometry-free test of the shared reduce -- guards the critical subtlety: this is
    # NOT a plain argmax. `(0.0, None)` init + the wkt tie-break gated on `best_real is not None`
    # means a candidate with gain <= 0 can never win (the greedy's termination condition).
    def L(a: tuple[float, float], b: tuple[float, float]) -> LineString:
        return LineString([a, b])
    # (a) all non-positive gains -> (0.0, None): a naive `best=None` argmax would wrongly let the
    # least-negative (or the 0.0) candidate win here.
    assert _best_candidate(
        [(-0.1, L((0, 0), (1, 1))), (0.0, L((0, 0), (2, 2))), (-0.05, L((0, 0), (3, 3)))]
    ) == (0.0, None)
    # (b) positive-gain tie -> deterministic min-wkt winner, order-independent.
    r1, r2 = L((0, 0), (1, 0)), L((0, 0), (0, 1))
    lo = r1 if r1.wkt < r2.wkt else r2
    fwd = _best_candidate([(0.4, r1), (0.4, r2)])
    rev = _best_candidate([(0.4, r2), (0.4, r1)])
    assert fwd[1] is not None and rev[1] is not None
    assert fwd[1].wkt == lo.wkt and rev[1].wkt == lo.wkt and fwd[0] == 0.4
    # (c) inf gain wins.
    inf = float("inf")
    assert _best_candidate([(0.9, r1), (inf, r2)])[0] == inf
    # (d) a (0.0, None) skipped-candidate entry never displaces a later positive-gain real.
    result = _best_candidate([(0.0, None), (0.5, r1)])
    assert result[0] == 0.5 and result[1] is not None and result[1].wkt == r1.wkt


def test_arterial_serial_refactor_identical() -> None:
    # The task-1 acceptance gate: the extraction of `eval_candidate` + the shared `_best_candidate`
    # reduce must be behavior-preserving. `_grid_block(3)` with a small `n_anchors` terminates
    # (no candidate improves) well before the default `max_roads=15` for BOTH modes -- verified by
    # comparing against a higher `max_roads` cap -- so this exercises the all-non-positive-gain
    # reduce path (the critical sentinel), not just positive-gain steps. Expected WKT captured from
    # the pre-refactor implementation.
    expected = {
        "buildable": sorted([
            "LINESTRING (1 0, 1 1)", "LINESTRING (1 1, 1 2)", "LINESTRING (1 1, 2 1)",
            "LINESTRING (1 2, 2 2, 2 1)", "LINESTRING (2 0, 2 1)",
        ]),
        "aspirational": sorted([
            "LINESTRING (0 1, 1 2)", "LINESTRING (0 3, 1 2)", "LINESTRING (1 2, 1.5 2.5)",
            "LINESTRING (1 2, 2.5 0.5)", "LINESTRING (1.5 2.5, 2 3)", "LINESTRING (2 0, 2.06 0)",
        ]),
    }
    block = _grid_block(3)
    for mode, want in expected.items():
        roads = GreedyArterialReblocker(mode=mode, objective="directness", n_anchors=6
                                        ).propose(block).roads
        assert roads is not None
        assert sorted(g.wkt for g in roads.geometry) == want


def test_arterial_parallel_identical_to_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    # The acceptance gate for the fork process pool: parallel roads must be WKT-identical to serial.
    # Force the threshold to 1 so EVERY step genuinely dispatches the fork pool (>1 candidate per
    # step here, so the 16-worker pool is truly exercised -- confirmed by the ~2x wall-clock speedup
    # over workers=1) on this small, fast, EARLY-TERMINATING block. n_anchors=6 (the same known-fast
    # config as test_arterial_serial_refactor_identical) keeps candidate counts modest so each step
    # forks quickly, while the block still terminates well before max_roads=15 in BOTH modes -- so
    # the final all-non-positive-gain step also runs through the pool. Without the forced-low
    # threshold, candidates < 128 and the "parallel" run would silently fall back to serial, proving
    # nothing. Repeat a few times because fork races are low-probability per run.
    monkeypatch.setattr(arterial, "_PARALLEL_THRESHOLD", 1)
    for _ in range(3):
        for mode in ("buildable", "aspirational"):
            block = _grid_block(3)
            serial = GreedyArterialReblocker(mode=mode, n_anchors=6, workers=1).propose(block).roads
            par = GreedyArterialReblocker(mode=mode, n_anchors=6, workers=16).propose(block).roads
            assert serial is not None and par is not None
            # Non-vacuity guard: both modes commit roads on this block before terminating (5
            # buildable, 6 aspirational -- see test_arterial_serial_refactor_identical), so an
            # empty-vs-empty comparison here would mean the fixture regressed, not that the pool
            # matched serial.
            assert len(serial.geometry) > 0
            assert sorted(g.wkt for g in serial.geometry) == sorted(g.wkt for g in par.geometry)


def test_arterial_parallel_geometry_bit_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    # The winner geometry returned across the process boundary must be COORDINATE-exact to serial's,
    # not merely wkt-equal (.wkt rounds at precision 6). This fails fast if a future change reverts
    # eval_candidate's return from the shapely geometry (pickled lossless via WKB) to a lossy wkt.
    monkeypatch.setattr(arterial, "_PARALLEL_THRESHOLD", 1)
    block = _grid_block(3)
    serial = GreedyArterialReblocker(mode="buildable", n_anchors=6, workers=1).propose(block).roads
    par = GreedyArterialReblocker(mode="buildable", n_anchors=6, workers=16).propose(block).roads
    assert serial is not None and par is not None
    s = sorted(serial.geometry, key=lambda g: g.wkt)
    p = sorted(par.geometry, key=lambda g: g.wkt)
    assert len(s) == len(p) and len(s) > 0
    for gs, gp in zip(s, p, strict=True):
        assert list(gs.coords) == list(gp.coords)


def test_arterial_parallel_matches_reference_1808(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tie the pool path directly to the committed ground truth: forced-low threshold so the fork
    # pool actually runs on the real 1808 block, and its buildable roads must WKT-set-equal the
    # pinned reference (same guarantee the serial test_arterial_proposal_wkt_unchanged asserts).
    from scoring_fixtures import _REF, _block_1808
    monkeypatch.setattr(arterial, "_PARALLEL_THRESHOLD", 1)
    roads = GreedyArterialReblocker(mode="buildable", workers=16).propose(_block_1808()).roads
    assert roads is not None
    assert sorted(g.wkt for g in roads.geometry) == sorted(_REF["arterial_buildable"]["wkt"])


def test_arterial_parallel_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two identical workers=16 proposes must produce identical sorted WKT. Forced-low threshold so
    # the pool path (not serial) is what's being checked for determinism.
    monkeypatch.setattr(arterial, "_PARALLEL_THRESHOLD", 1)
    block = _grid_block(3)
    a = GreedyArterialReblocker(mode="buildable", n_anchors=6, workers=16).propose(block).roads
    b = GreedyArterialReblocker(mode="buildable", n_anchors=6, workers=16).propose(block).roads
    assert a is not None and b is not None
    assert sorted(g.wkt for g in a.geometry) == sorted(g.wkt for g in b.geometry)


def test_arterial_parallel_soak(monkeypatch: pytest.MonkeyPatch) -> None:
    # NOT a correctness check (see test_arterial_parallel_identical_to_serial for that) -- guards
    # against the fork pool leaking/accumulating semaphores or resource-tracker handles, or hanging,
    # across many short-lived pools (the ~pools-per-propose x many-proposes churn a real multi-block
    # pipeline run produces). Force the threshold to 1 so EVERY step of EVERY propose genuinely
    # dispatches a fork pool: at the default threshold, _grid_block(3)'s modest candidate counts
    # (n_anchors=6, the same known-fast config as the other parallel tests) stay under 128 and the
    # soak would take the serial path every time, spawning zero pools and proving nothing. 30
    # proposes x a handful of greedy steps/propose = a few hundred pool create/teardown cycles.
    monkeypatch.setattr(arterial, "_PARALLEL_THRESHOLD", 1)
    block = _grid_block(3)
    for _ in range(30):
        roads = GreedyArterialReblocker(mode="buildable", objective="directness", n_anchors=6,
                                        workers=16).propose(block).roads
        assert roads is not None
        assert len(roads) > 0


def test_planarize_nodes_two_crossing_chords() -> None:
    a = LineString([(0.0, 1.0), (2.0, 1.0)])
    b = LineString([(1.0, 0.0), (1.0, 2.0)])             # crosses a at (1, 1)
    gdf = _planarize([a, b], UTM)
    coords = {c for geom in gdf.geometry for c in geom.coords}
    assert (1.0, 1.0) in coords                    # crossing became a shared vertex
    assert len(gdf) == 4                           # each chord split into two at the crossing


def test_greedy_first_arterial_cuts_the_deep_block() -> None:
    # A long 3x9 block with street frontage on ONE short end only (not full-boundary: with
    # streets on all 4 sides every interior parcel is at most 2 hops from a street, so a
    # length-1 stub that merely touches the shared corner of two such parcels already "unlocks"
    # them for network_efficiency's per-pair entry-reachability test -- under the greedy's
    # gain-per-meter ranking that corner-touch stub always beats a length-9 spine, no matter how
    # many parcels the spine would serve, because the spine's benefit is diluted by its length.
    # A real deep pocket -- depth growing down the spine, only reachable by routing its length --
    # is what actually forces a spanning arterial to be the best first move.
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(3) for j in range(9)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[LineString([(0.0, 0.0), (0.0, 9.0)])], crs=UTM)
    block = Block(block_id="long", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)
    roads = _greedy_arterials(block, mode="buildable", objective="directness", max_roads=3,
                              n_anchors=12)
    assert len(roads) >= 1
    assert roads.geometry.length.max() >= 6.0                        # a real lengthwise arterial
    conn = street_connectivity(block.streets, roads, STREET_TOL)
    assert conn.connected_frac == 1.0                                # buildable + street-connected


def test_greedy_is_deterministic() -> None:
    block = _grid_block(5)
    r1 = _greedy_arterials(block, mode="buildable", objective="directness", max_roads=4,
                           n_anchors=12)
    r2 = _greedy_arterials(block, mode="buildable", objective="directness", max_roads=4,
                           n_anchors=12)
    assert [g.wkt for g in r1.geometry] == [g.wkt for g in r2.geometry]


def test_greedy_roads_carry_drainage_and_slice_into_a_curve() -> None:
    from reblock.budget import cost_benefit_curve, road_drainage
    block = _grid_block(6)
    roads = _greedy_arterials(block, mode="buildable", objective="directness",
                              max_roads=5, n_anchors=12)
    assert len(roads) >= 1
    assert list(roads["drain"]) == road_drainage(block, roads)   # drain IS the actual drainage
    curve = cost_benefit_curve(block, roads)                     # integrates with budget machinery
    assert len(curve.cost) >= 2                                  # multiple budget points, not stub
    assert curve.benefit[-1] >= curve.benefit[0]                 # benefit doesn't regress w/ budget


def test_arterial_proposal_wkt_unchanged() -> None:
    # The perf-refactor gate for `_snap`/`_planarize` (task 5 of the arterial-frozen-context perf
    # design): buildable arterial roads on the real 1808 block must match the pinned reference WKT
    # EXACTLY -- not merely score-equivalent. `_snap`'s shapely-ufunc weights and `_planarize`'s
    # incremental union are perf-only changes; any drift here means a path or noding decision
    # actually changed.
    from scoring_fixtures import _REF, _block_1808
    roads = GreedyArterialReblocker(mode="buildable", objective="directness").propose(
        _block_1808()).roads
    assert roads is not None
    assert sorted(g.wkt for g in roads.geometry) == sorted(_REF["arterial_buildable"]["wkt"])


def test_aspirational_planarizes_crossings_into_true_intersections() -> None:
    block = _grid_block(6)
    roads = _greedy_arterials(block, mode="aspirational", objective="directness", max_roads=6,
                              n_anchors=12)
    # Build the road+street graph as a CSR (the nx-free scoring path's own builders) and read
    # node degree off the CSR's per-row nnz -- a real crossroads noded by `_planarize` shows up
    # as one shared node where >= 4 segments meet, vs two overlapping lines that never node.
    from reblock.budget import _build_csr, _explode_segments
    segs = _explode_segments([*roads.geometry, *block.streets.geometry])
    csr, node_index = _build_csr(segs, {})
    degree = csr.getnnz(axis=1)
    # at least one interior node has degree >= 4 -> a real crossroads, not overlapping lines
    interior = [nd for nd in node_index
               if Point(nd).distance(block.boundary.boundary) > STREET_TOL]
    assert any(degree[node_index[nd]] >= 4 for nd in interior)


def test_identity_and_proposal_metadata() -> None:
    m = GreedyArterialReblocker(mode="buildable", objective="directness")
    assert m.identity == (
        "greedy_arterial", "buildable", "directness", "length", 0.0,
        15, 32, 8, 2.0, False, "grow", 0, 0)
    # max_roads / n_anchors / top_k / lam change the proposed roads -> must change the cache key,
    # else a budget/candidate sweep silently returns another setting's cached proposal.
    assert GreedyArterialReblocker(max_roads=3).identity != m.identity
    assert GreedyArterialReblocker(n_anchors=16).identity != m.identity
    assert GreedyArterialReblocker(lazy=True).identity != m.identity
    assert GreedyArterialReblocker(candidate_policy="fixed").identity != m.identity
    assert GreedyArterialReblocker(rescore_every=2).identity != m.identity
    assert GreedyArterialReblocker(max_anchors=48).identity != m.identity
    proposal = GreedyArterialReblocker(objective="directness").propose(_grid_block(5))
    assert proposal.block_identity == _grid_block(5).identity


def test_both_modes_produce_roads() -> None:
    block = _grid_block(6)
    for mode in ("buildable", "aspirational"):
        p = GreedyArterialReblocker(mode=mode, objective="directness", max_roads=4).propose(block)
        assert p.roads is not None and len(p.roads) > 0


def test_config_and_derivation_wiring() -> None:
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from reblock.derive_graph import _DERIVATION_MODULES
    assert any(p.name == "arterial.py" for p in _DERIVATION_MODULES)
    conf_dir = str(Path("conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="compare_config",
                      overrides=["shapefile=x", "methods=[greedy_arterial_buildable]"])
    m = instantiate(cfg.all_methods["greedy_arterial_buildable"])
    # NOTE: lazy=True here (unlike other goldens in this file) matches compare_config.yaml's
    # inline greedy_arterial_buildable entry, which sets lazy: true -- a pre-existing golden/config
    # mismatch (this assertion previously asserted False) found and fixed incidentally while
    # updating these tuples for max_anchors; unrelated to the anchor-cap feature itself.
    assert m.identity == (
        "greedy_arterial", "buildable", "directness", "length", 0.0, 15, 32, 8, 2.0, True, "grow",
        0, 0)


def test_displacement_config_instantiates_with_right_params_and_identity() -> None:
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    conf_dir = str(Path("conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="compare_config",
                      overrides=["shapefile=x", "methods=[greedy_arterial_displacement]"])
    m = instantiate(cfg.all_methods["greedy_arterial_displacement"])
    assert (m.mode, m.objective, m.cost, m.corridor_m) == (
        "aspirational", "directness", "displacement", 3.0)
    assert m.identity == (
        "greedy_arterial", "aspirational", "directness", "displacement", 3.0, 15, 32, 8, 2.0, False,
        "grow", 0, 0)

    # The standalone conf/method/greedy_arterial_displacement.yaml config group (config.yaml's
    # `method=` default group), separate from compare_config's inline `all_methods` entry above.
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        method_cfg = compose(config_name="config",
                             overrides=["shapefile=x", "method=greedy_arterial_displacement"])
    assert instantiate(method_cfg.method).identity == m.identity


def _deep_block() -> Block:
    # A 3x9 block with street frontage on one short end only -- a deep pocket where an arterial
    # genuinely helps (same shape as test_greedy_first_arterial_cuts_the_deep_block).
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(3) for j in range(9)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[LineString([(0.0, 0.0), (0.0, 9.0)])], crs=UTM)
    return Block(block_id="deep", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_buildable_arterial_more_direct_than_dijkstra() -> None:
    # The buildable arterial's raison d'etre: on a deep block it beats the tree method's tendrils
    # on directness (demonstrated ~10x on a real DJI block; asserted here on a fast synthetic
    # deep block). This is the sub-project's headline value claim.
    from reblock.budget import auc, efficiency_directness_curves
    from reblock.methods.dijkstra import DijkstraReblocker
    block = _deep_block()
    art = GreedyArterialReblocker(mode="buildable", objective="directness",
                                  n_anchors=12, max_roads=4).propose(block).roads
    dij = DijkstraReblocker().propose(block).roads
    assert art is not None and dij is not None
    _, dc_art = efficiency_directness_curves(block, art)
    _, dc_dij = efficiency_directness_curves(block, dij)
    cap = max(dc_art.cost[-1], dc_dij.cost[-1])
    assert auc(dc_art, cap) > auc(dc_dij, cap)


def _holed_block() -> Block:
    # A 5x5 grid with the center parcel removed -> the block boundary is a square with a square
    # hole, so `boundary.boundary` is a MultiLineString (outer ring + inner ring).
    polys, ids = [], []
    for i in range(5):
        for j in range(5):
            if (i, j) == (2, 2):
                continue
            polys.append(Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]))
            ids.append(i * 5 + j)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="holed", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_greedy_handles_multilinestring_streets() -> None:
    # A holed/courtyard block's streets are a MultiLineString; anchor sampling must explode Multi*
    # rather than filter it out, or the greedy sees no anchors and returns an empty proposal.
    block = _holed_block()
    assert "Multi" in block.streets.geometry.iloc[0].geom_type    # precondition: streets ARE Multi
    roads = _greedy_arterials(block, mode="buildable", objective="directness",
                              max_roads=3, n_anchors=12)
    assert len(roads) >= 1


def test_cost_displacement_in_identity() -> None:
    m = GreedyArterialReblocker(mode="aspirational", objective="directness", cost="displacement")
    assert m.identity == (
        "greedy_arterial", "aspirational", "directness", "displacement", 3.0, 15, 32, 8, 2.0, False,
        "grow", 0, 0)


def _two_arm_block(building_points: gpd.GeoDataFrame, h: int = 9, gap_x1: int = 10) -> Block:
    # Two disjoint 1-wide x h-tall columns ("arms"), each with its own street frontage at its
    # bottom edge, separated by an empty gap -- two independent deep pockets, mirror images of
    # each other (translated by `gap_x1` in x), so a straight arterial spanning either arm's full
    # height has IDENTICAL objective benefit and length. building_points lets the two arms'
    # corridors differ in displacement while their raw benefit stays tied.
    polys = [Polygon([(0, j), (1, j), (1, j + 1), (0, j + 1)]) for j in range(h)]
    polys += [Polygon([(gap_x1, j), (gap_x1 + 1, j), (gap_x1 + 1, j + 1), (gap_x1, j + 1)])
             for j in range(h)]
    ids = list(range(2 * h))
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[
        LineString([(0.0, 0.0), (1.0, 0.0)]),
        LineString([(float(gap_x1), 0.0), (float(gap_x1 + 1), 0.0)]),
    ], crs=UTM)
    return Block(block_id="two_arm", crs=UTM, boundary=boundary, parcels=parcels, streets=streets,
                building_points=building_points)


def test_cost_displacement_avoids_the_denser_corridor() -> None:
    # cost="displacement" must select differently from cost="length": dense building points along
    # the LEFT arm, sparse (one, far-flung) on the RIGHT. The arms are length- and benefit-tied, so
    # cost="length" is blind to the points (it falls to the wkt tie-break, landing on the LEFT arm);
    # cost="displacement" steers away from the dense corridor to a road that displaces far fewer
    # buildings under the disk measure. NOTE the picked road is NOT a zero-displacement escape here
    # -- under the OLD centroid-count rule the greedy's optimum was a diagonal chord that grazed no
    # CENTROID (count=0, an infinite-gain "free" pick); but that diagonal actually passes close
    # enough to the lone sparse point's large disk (its radius is NN/2=5.0, since it is far from
    # every other point) to score WORSE under the disk measure (~1.30) than the arm-serving road the
    # disk-based greedy picks instead (1.0, the sparse point fully inside the corridor) -- exactly
    # the degenerate "denom=0 for candidates that graze footprint edges but miss centroids" escape
    # this migration closes. The FINITE raw/denom ranking (two candidates both displacing >0, pick
    # the cheaper) is covered by the next test.
    left_pts = [Point(0.5, y) for y in range(1, 8)]     # dense: 7 sites astride the left arm
    right_pts = [Point(10.5, 4)]                        # sparse: 1 site astride the right arm
    pts = gpd.GeoDataFrame(geometry=left_pts + right_pts, crs=UTM)
    block = _two_arm_block(pts)

    roads_length = _greedy_arterials(block, mode="aspirational", objective="access", max_roads=1,
                                     n_anchors=8, top_k=2, corridor_m=1.0, cost="length")
    roads_disp = _greedy_arterials(block, mode="aspirational", objective="access", max_roads=1,
                                   n_anchors=8, top_k=2, corridor_m=1.0, cost="displacement")

    assert len(roads_length) == 1 and len(roads_disp) == 1
    # the cost switch changed WHICH road is proposed:
    assert roads_length.geometry.iloc[0].wkt != roads_disp.geometry.iloc[0].wkt
    # ... to one that displaces fewer buildings (disk measure) than the length-optimal pick:
    radii = building_radii(pts, corridor_m=1.0)
    d_length = displacement(pts, radii, roads_length, corridor_m=1.0)
    d_disp = displacement(pts, radii, roads_disp, corridor_m=1.0)
    assert d_disp < d_length
    assert d_disp == 1.0    # right arm's single (large-radius) point, fully inside its corridor


def _grid_block_with_points(building_points: gpd.GeoDataFrame, w: int = 8, h: int = 3) -> Block:
    # A w x h unit-parcel grid with bottom-edge street frontage -- a shallow block whose only
    # deep parcels are the top row, so straight arterials from the bottom compete on reaching them.
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(w) for j in range(h)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[LineString([(0.0, 0.0), (float(w), 0.0)])], crs=UTM)
    return Block(block_id="grid", crs=UTM, boundary=boundary, parcels=parcels, streets=streets,
                building_points=building_points)


def test_cost_displacement_finite_ranking_prefers_the_sparser_corridor() -> None:
    # The FINITE-vs-FINITE case the greedy denominator exists for: every candidate here displaces
    # >0 buildings (points tile the whole block at corridor_m=1.0, so no free/zero-displacement
    # escape chord exists), so the pick is decided by the raw/denom comparison, NOT the inf branch.
    # A heavy cluster sits astride the access-optimal corridor (x~2): cost="length" drives straight
    # through it (many displaced); cost="displacement" must swerve to a sparser corridor that
    # displaces far fewer (disk measure) while giving up a little access benefit.
    base = [Point(i + 0.5, j + 0.5) for i in range(8) for j in range(3)]     # tile: no free chord
    cluster = [Point(2.0 + dx, 1.5 + dy) for dx in (-0.4, -0.2, 0.0, 0.2, 0.4)
               for dy in (-0.6, -0.3, 0.0, 0.3, 0.6)]                        # dense at x~2
    pts = gpd.GeoDataFrame(geometry=base + cluster, crs=UTM)
    block = _grid_block_with_points(pts)

    roads_length = _greedy_arterials(block, mode="aspirational", objective="access", max_roads=1,
                                     n_anchors=10, top_k=4, corridor_m=1.0, cost="length")
    roads_disp = _greedy_arterials(block, mode="aspirational", objective="access", max_roads=1,
                                   n_anchors=10, top_k=4, corridor_m=1.0, cost="displacement")
    radii = building_radii(pts, corridor_m=1.0)
    d_length = displacement(pts, radii, roads_length, corridor_m=1.0)
    d_disp = displacement(pts, radii, roads_disp, corridor_m=1.0)

    assert roads_length.geometry.iloc[0].wkt != roads_disp.geometry.iloc[0].wkt   # cost changed it
    assert d_disp > 0.0              # the pick DISPLACES -> the finite raw/denom branch, not inf
    assert d_disp < d_length         # ... and displaces strictly fewer (disk) than the length pick


def test_cost_displacement_is_deterministic() -> None:
    left_pts = [Point(0.5, y) for y in range(1, 8)]
    right_pts = [Point(10.5, 4)]
    pts = gpd.GeoDataFrame(geometry=left_pts + right_pts, crs=UTM)
    block = _two_arm_block(pts)
    r1 = _greedy_arterials(block, mode="aspirational", objective="access", cost="displacement",
                           max_roads=1, n_anchors=8, top_k=2, corridor_m=1.0)
    r2 = _greedy_arterials(block, mode="aspirational", objective="access", cost="displacement",
                           max_roads=1, n_anchors=8, top_k=2, corridor_m=1.0)
    assert [g.wkt for g in r1.geometry] == [g.wkt for g in r2.geometry]


def test_cost_displacement_commits_a_zero_displacement_beneficial_road() -> None:
    # A beneficial straight road whose corridor holds no building sites must still be committed
    # -- the zero-marginal-displacement candidate ranks as infinite gain (strictly above any
    # positive-denominator candidate), not skipped by an attempted division by zero. The
    # building_points are real (non-empty) but sited far outside the block, so disk `displacement`
    # exercises its actual buffer/union/distance path (not the empty-input short circuit) and still
    # returns 0 for every candidate here.
    plain = _deep_block()
    far_points = gpd.GeoDataFrame(geometry=[Point(1000.0, 1000.0)], crs=UTM)
    block = Block(block_id=plain.block_id, crs=plain.crs, boundary=plain.boundary,
                 parcels=plain.parcels, streets=plain.streets, building_points=far_points)
    roads = _greedy_arterials(block, mode="aspirational", objective="access", cost="displacement",
                              max_roads=1, n_anchors=12, corridor_m=1.0)
    assert len(roads) == 1
    assert roads.geometry.iloc[0].length > 1.0                 # a real, non-degenerate candidate
    far_radii = building_radii(far_points, corridor_m=1.0)
    assert displacement(far_points, far_radii, roads, corridor_m=1.0) == 0.0


def test_displacement_objective_is_extent_aware_unlike_the_old_centroid_rule() -> None:
    # WHY this migration matters: a road whose corridor GRAZES a building's footprint disk but
    # misses its centroid entirely displaces 0 buildings under the old centroid rule
    # (displacement_count's `within(corridor)` is False for every point) -- the exact degeneracy
    # the spike found (denom=0 for candidates that graze footprint EDGES but miss centroids,
    # block 40972: disk 2.84 -> 0.24). The disk `displacement` still credits it partial
    # displacement because the footprint's disk (r=NN/2) reaches into the corridor even though
    # the centroid does not.
    road = gpd.GeoDataFrame(geometry=[LineString([(0.0, 0.0), (10.0, 0.0)])], crs=UTM)
    # two buildings 2 m apart -> NN dist 2 -> r = 1.0 each. Point A sits just past the 1 m
    # corridor edge (y=1.5 -> d=0.5 < r=1.0 -> its disk grazes the corridor); point B is out of
    # disk range entirely (y=3.5 -> d=2.5 > r=1.0 -> contributes 0 either way).
    pts = gpd.GeoDataFrame(geometry=[Point(5.0, 1.5), Point(5.0, 3.5)], crs=UTM)
    radii = building_radii(pts, corridor_m=1.0)

    corridor = road.geometry.buffer(1.0).union_all()
    assert not pts.geometry.within(corridor).any()          # OLD centroid rule: nobody displaced
    d = displacement(pts, radii, road, corridor_m=1.0)
    assert d > 0.0                                          # disk rule: A is partially displaced
