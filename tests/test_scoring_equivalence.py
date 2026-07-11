"""Equivalence harness pinning `network_efficiency`/`efficiency_directness_curves`/`auc`'s
CURRENT output on the 1808 sample block, to 1e-9. This is the correctness safety net for the
arterial frozen-context perf refactor: every task that touches scoring must keep these green."""
import json
from pathlib import Path

import networkx as nx
from scipy.sparse.csgraph import dijkstra
from scoring_fixtures import _block_1808, _grid_block, _region_deep, _roads, sampled_fixtures
from shapely.geometry import LineString

from reblock.budget import (
    _BlockScoringContext,
    _graph_to_csr,
    auc,
    efficiency_directness_curves,
    network_efficiency,
)
from reblock.methods.arterial import _planarize


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_network_efficiency_matches_reference() -> None:
    for name, block, roads, exp in sampled_fixtures():
        e, d = network_efficiency(block, roads)
        assert _close(e, exp["E"]), (name, "E", e, exp["E"])
        assert _close(d, exp["directness"]), (name, "directness", d, exp["directness"])


def test_context_score_matches_network_efficiency() -> None:
    # `_BlockScoringContext(block).score(roads)` is the migration target of `network_efficiency`;
    # they must agree bit-for-bit on every fixture (and hence both match the pinned reference).
    for name, block, roads, _exp in sampled_fixtures():
        ctx = _BlockScoringContext(block)
        e_ctx, d_ctx = ctx.score(roads)
        e_ref, d_ref = network_efficiency(block, roads)
        assert _close(e_ctx, e_ref), (name, "E", e_ctx, e_ref)
        assert _close(d_ctx, d_ref), (name, "directness", d_ctx, d_ref)


def test_one_context_scores_many_road_sets() -> None:
    # The arterial greedy builds ONE context per block and scores every candidate road set through
    # it: a single reused context must still match the pinned reference for each of the 1808 block's
    # three road sets (nothing road-specific may leak into the frozen state).
    block = _block_1808()
    ref = json.loads((Path(__file__).resolve().parent
                      / "data/scoring/ref_values_1808.json").read_text())
    ctx = _BlockScoringContext(block)
    for key in ("no_roads", "dijkstra", "arterial_buildable"):
        roads = _roads(block, ref, key)
        e, d = ctx.score(roads)
        assert _close(e, ref[key]["E"]), (key, "E", e, ref[key]["E"])
        assert _close(d, ref[key]["directness"]), (key, "directness", d, ref[key]["directness"])


def test_curves_and_auc_match_reference() -> None:
    for name, block, roads, exp in sampled_fixtures():
        if roads is None or "E_auc" not in exp:
            continue
        ec, dc = efficiency_directness_curves(block, roads)
        for got, want in zip(ec.benefit, exp["E_curve_benefit"], strict=True):
            assert _close(got, want), (name, "E_curve", got, want)
        for got, want in zip(dc.benefit, exp["dir_curve_benefit"], strict=True):
            assert _close(got, want), (name, "dir_curve", got, want)
        cap = min(ec.cost[-1], dc.cost[-1])
        assert _close(auc(ec, cap), exp["E_auc"]), (name, "E_auc")
        assert _close(auc(dc, cap), exp["dir_auc"]), (name, "dir_auc")


def test_incremental_scorer_matches_full_rederivation() -> None:
    # The arterial greedy's per-candidate incremental scorer (`ctx.step(committed).score_candidate`)
    # must equal a FULL re-derivation `network_efficiency(block, _planarize(committed + [real]))` to
    # 1e-9 on a block WITH committed roads -- the public harness above only ever scores full road
    # sets through full re-derivation, so it cannot catch an incremental entry/graph bug (a stale
    # committed road, a missed crossing node, a diverging tie-break). This gate MUST cover BOTH:
    #   (R1) an aspirational trial that crosses a committed road MID-SPAN (a diagonal meeting it
    #        away from a shared vertex): the road subgraph re-nodes the crossing, so a bare
    #        "append the trial's edges" would leave the committed road unsplit and mis-distance it.
    #   (#4) a grid/region block whose parcels sit at EXACT distance ties (a parcel abuts several
    #        edges at the identical distance), where the entry node is decided purely by the
    #        `_line_entries` (distance, nx-edge-index) tie-break -- so the incremental min(step,
    #        trial) must resolve ties against the SAME nx order as `.score`, or it picks a different
    #        entry node. (Block 1808 alone would pass even with the #4 bug -- it has no exact ties.)
    b1808 = _block_1808()
    committed_1808 = [LineString([(297630.0, 1280300.0), (297648.0, 1280300.0)])]
    trials_1808 = [
        LineString([(297630.0, 1280300.0), (297635.0, 1280290.0)]),    # meets committed at a vertex
        LineString([(297635.0, 1280290.0), (297642.0, 1280312.0)]),    # crosses committed MID-SPAN
        LineString([(297626.0, 1280295.0), (297650.0, 1280308.0)]),    # another mid-span diagonal
        LineString([(297628.48, 1280309.62), (297644.49, 1280299.48)]),  # buildable-style chord
    ]
    region = _region_deep()                       # deep 3x6|3x6 grid region -- exact distance ties
    committed_region = [LineString([(1.0, 0.0), (1.0, 6.0)])]
    trials_region = [
        LineString([(2.0, 0.0), (2.0, 6.0)]),      # grid-aligned, meets committed at vertices
        LineString([(0.0, 3.0), (6.0, 3.0)]),      # grid-aligned crossbar over the committed road
        LineString([(4.0, 0.0), (4.0, 6.0)]),
        LineString([(0.0, 1.0), (6.0, 5.0)]),      # diagonal crossing the committed road mid-span
    ]
    grid = _grid_block(0, 0, 6, 6, "bottom", "grid6")   # exact ties, exercised from empty committed
    trials_grid = [
        LineString([(3.0, 0.0), (3.0, 6.0)]),
        LineString([(0.0, 3.0), (6.0, 3.0)]),
    ]
    cases = [
        ("1808", b1808, committed_1808, trials_1808),
        ("deep_region", region, committed_region, trials_region),
        ("grid6", grid, [], trials_grid),
    ]
    for name, block, committed, trials in cases:
        ctx = _BlockScoringContext(block)
        step = ctx.step(_planarize(committed, block.crs))
        for real in trials:
            got = step.score_candidate(real)
            want = network_efficiency(block, _planarize(committed + [real], block.crs))
            assert _close(got[0], want[0]), (name, real.wkt, "E", got, want)
            assert _close(got[1], want[1]), (name, real.wkt, "directness", got, want)


def test_csgraph_matches_networkx_distances() -> None:
    g = nx.Graph()
    for a, b, w in [((0.0, 0.0), (1.0, 0.0), 1.0), ((1.0, 0.0), (2.0, 0.0), 1.0),
                    ((0.0, 0.0), (2.0, 0.0), 3.0), ((1.0, 0.0), (1.0, 0.0), 0.0)]:
        if a != b:
            g.add_edge(a, b, weight=w)
    csr, idx = _graph_to_csr(g)
    src = idx[(0.0, 0.0)]
    d = dijkstra(csr, directed=False, indices=src)
    ref = nx.single_source_dijkstra_path_length(g, (0.0, 0.0))
    for node, i in idx.items():
        rv = ref.get(node, float("inf"))
        assert abs(d[i] - rv) <= 1e-12, (node, d[i], rv)
