"""Equivalence harness pinning `network_efficiency`/`efficiency_directness_curves`/`auc`'s
CURRENT output on the 1808 sample block, to 1e-9. This is the correctness safety net for the
arterial frozen-context perf refactor: every task that touches scoring must keep these green."""
import json
from pathlib import Path

import networkx as nx
from scipy.sparse.csgraph import dijkstra
from scoring_fixtures import _block_1808, _roads, sampled_fixtures

from reblock.budget import (
    _BlockScoringContext,
    _graph_to_csr,
    auc,
    efficiency_directness_curves,
    network_efficiency,
)


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
