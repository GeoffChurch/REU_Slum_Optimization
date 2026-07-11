"""Equivalence harness pinning `network_efficiency`/`efficiency_directness_curves`/`auc`'s
CURRENT output on the 1808 sample block, to 1e-9. This is the correctness safety net for the
arterial frozen-context perf refactor: every task that touches scoring must keep these green."""
from scoring_fixtures import sampled_fixtures

from reblock.budget import auc, efficiency_directness_curves, network_efficiency


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_network_efficiency_matches_reference() -> None:
    for name, block, roads, exp in sampled_fixtures():
        e, d = network_efficiency(block, roads)
        assert _close(e, exp["E"]), (name, "E", e, exp["E"])
        assert _close(d, exp["directness"]), (name, "directness", d, exp["directness"])


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
