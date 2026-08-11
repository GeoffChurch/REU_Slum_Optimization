"""Equivalence harness pinning `network_efficiency`'s CURRENT output on the 1808 sample block, to
1e-9. This is the correctness safety net for the arterial frozen-context perf refactor: every task
that touches scoring must keep these green."""
import json
from pathlib import Path

import pytest
from scoring_fixtures import _block_1808, _grid_block, _region_deep, _roads, sampled_fixtures
from shapely.geometry import LineString

from reblock.budget import (
    _BlockScoringContext,
    _StepContext,
    network_efficiency,
)
from reblock.methods.arterial.engines import _greedy_arterials
from reblock.methods.arterial.primitives import _planarize
from reblock.permeability import DEFAULT_ROAD_WIDTH_M


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
    for key in ("no_roads", "least_cost", "arterial_buildable"):
        roads = _roads(block, ref, key)
        e, d = ctx.score(roads)
        assert _close(e, ref[key]["E"]), (key, "E", e, ref[key]["E"])
        assert _close(d, ref[key]["directness"]), (key, "directness", d, ref[key]["directness"])


def test_incremental_scorer_matches_full_rederivation() -> None:
    # The arterial greedy's per-candidate incremental scorer (`ctx.step(committed).score_candidate`)
    # must equal a FULL re-derivation `network_efficiency(block, _planarize(committed + [real]))` to
    # 1e-9 for BUILDABLE-valid trials -- boundary-snapped roads that meet the committed/street
    # network at SHARED graph vertices (never crossing a committed edge at a float interior point).
    # `score_candidate` is bit-exact ONLY under that precondition: an aspirational free chord that
    # crosses a committed edge at a float point is NOT bit-exact (the incremental planarize noding
    # diverges from the reference -- "Bug 2"), which is exactly why the greedy routes aspirational
    # through the full path (see `test_greedy_routes_aspirational_to_full_rederivation`). The public
    # harness above only scores full road sets through full re-derivation, so it cannot catch an
    # incremental entry/graph bug (a stale committed road, a missed crossing node, a diverging
    # tie-break). This gate MUST cover:
    #   (#4) a grid/region block whose parcels sit at EXACT distance ties (a parcel abuts several
    #        edges at the identical distance), where the entry node is decided purely by the
    #        `_line_entries` (distance, nx-edge-index) tie-break -- so the incremental min(step,
    #        trial) must resolve ties against the SAME nx order as `.score`, or it picks a different
    #        entry node. (Block 1808 alone would pass even with the #4 bug -- it has no exact ties.)
    #   (Bug-1) a trial that CLEANLY SPLITS a parcel's nearest committed step edge (on crossing-
    #        committed grid/region fixtures): the split drops that step edge and its near sub-seg
    #        -- whose `_rnd`-rounded crossing distance can rise ABOVE a 2nd-nearest step edge --
    #        must be recovered from the delta while the 2nd-nearest stays a live candidate.
    #        `_freeze_base` freezes ALL near step edges (not only the min-distance ones) precisely
    #        for this; a min-only freeze would discard the 2nd-nearest and pick the wrong entry.
    b1808 = _block_1808()
    committed_1808 = [LineString([(297630.0, 1280300.0), (297648.0, 1280300.0)])]
    trials_1808 = [
        LineString([(297630.0, 1280300.0), (297635.0, 1280290.0)]),   # meets committed at a vertex
        LineString([(297635.0, 1280290.0), (297642.0, 1280312.0)]),   # splits the committed edge
        LineString([(297626.0, 1280295.0), (297650.0, 1280308.0)]),   # splits it again (Bug-1)
        LineString([(297628.48, 1280309.62), (297644.49, 1280299.48)]),  # buildable-style chord
    ]
    region = _region_deep()                       # deep 3x6|3x6 grid region -- exact distance ties
    committed_region = [LineString([(1.0, 0.0), (1.0, 6.0)])]
    trials_region = [
        LineString([(2.0, 0.0), (2.0, 6.0)]),      # grid-aligned, meets committed at vertices
        LineString([(0.0, 3.0), (6.0, 3.0)]),      # grid-aligned crossbar over the committed road
        LineString([(4.0, 0.0), (4.0, 6.0)]),
    ]
    grid = _grid_block(0, 0, 6, 6, "bottom", "grid6")   # exact ties, exercised from empty committed
    trials_grid = [
        LineString([(3.0, 0.0), (3.0, 6.0)]),
        LineString([(0.0, 3.0), (6.0, 3.0)]),
    ]
    # Bug-1 focus: committed roads that CROSS EACH OTHER (so the step base is already noded), then
    # buildable-valid grid trials that CROSS one committed sub-edge at a lattice point -> split a
    # frozen step edge. Integer lattice crossings node identically both ways, so these stay
    # bit-exact and specifically exercise the "freeze ALL step edges" robustness path.
    committed_cross = [LineString([(2.0, 0.0), (2.0, 6.0)]),
                       LineString([(0.0, 3.0), (6.0, 3.0)])]         # cross at (2, 3)
    trials_cross = [
        LineString([(4.0, 0.0), (4.0, 6.0)]),   # crosses (2,3)-(6,3) at (4,3): splits a sub-edge
        LineString([(0.0, 4.0), (6.0, 4.0)]),   # crosses (2,3)-(2,6) at (2,4): splits a sub-edge
        LineString([(0.0, 2.0), (6.0, 2.0)]),   # crosses (2,0)-(2,3) at (2,2): splits a sub-edge
        LineString([(5.0, 0.0), (5.0, 6.0)]),   # crosses (2,3)-(6,3) at (5,3)
    ]
    region_cross_committed = [LineString([(1.0, 0.0), (1.0, 6.0)]),
                              LineString([(0.0, 3.0), (6.0, 3.0)])]  # cross at (1, 3)
    trials_region_cross = [
        LineString([(2.0, 0.0), (2.0, 6.0)]),   # crosses (1,3)-(6,3) at (2,3): splits a sub-edge
        LineString([(0.0, 4.0), (6.0, 4.0)]),   # crosses (1,3)-(1,6) at (1,4): splits a sub-edge
    ]
    cases = [
        ("1808", b1808, committed_1808, trials_1808),
        ("deep_region", region, committed_region, trials_region),
        ("grid6", grid, [], trials_grid),
        ("grid6_cross", grid, committed_cross, trials_cross),
        ("region_cross", region, region_cross_committed, trials_region_cross),
    ]
    for name, block, committed, trials in cases:
        ctx = _BlockScoringContext(block)
        step = ctx.step(_planarize(committed, block.crs, 6.0))
        for real in trials:
            got = step.score_candidate(real)
            want = network_efficiency(block, _planarize(committed + [real], block.crs, 6.0))
            assert _close(got[0], want[0]), (name, real.wkt, "E", got, want)
            assert _close(got[1], want[1]), (name, real.wkt, "directness", got, want)


def test_greedy_routes_aspirational_to_full_rederivation(monkeypatch: pytest.MonkeyPatch) -> None:
    # The arterial greedy MUST score BUILDABLE candidates through the incremental
    # `_StepContext.score_candidate` (fast, bit-exact for boundary-snapped trials) and ASPIRATIONAL
    # candidates through the full `ctx.score(_planarize(committed + [real]))` reference path --
    # because `score_candidate` is NOT bit-exact for aspirational float-crossing chords ("Bug 2").
    # This is the gate that FAILS if someone routes aspirational back through `score_candidate`:
    # spy on the method and assert an aspirational run never calls it, while a buildable run does.
    # Pin `workers=1` (serial): this routing invariant is parallelism-independent, but the spy is an
    # in-PROCESS counter -- the default fork process pool (`workers=16`) would run `eval_candidate`
    # (and its `score_candidate` calls) in child processes on this 833-candidates-per-step region,
    # where the parent's `calls["n"]` never sees them. Serial keeps the scorer calls observable.
    calls = {"n": 0}
    orig = _StepContext.score_candidate

    def spy(self: _StepContext, real: LineString) -> tuple[float, float]:
        calls["n"] += 1
        return orig(self, real)

    monkeypatch.setattr(_StepContext, "score_candidate", spy)
    region = _region_deep()

    calls["n"] = 0
    roads_a1 = _greedy_arterials(
        region, half_width_m=DEFAULT_ROAD_WIDTH_M / 2.0,
        mode="aspirational", objective="directness",
                                          max_roads=2, workers=1)
    assert calls["n"] == 0, "aspirational must NOT use the incremental scorer (Bug 2)"

    calls["n"] = 0
    _greedy_arterials(
        region, half_width_m=DEFAULT_ROAD_WIDTH_M / 2.0,
        mode="buildable", objective="directness", max_roads=2,
                               workers=1)
    assert calls["n"] > 0, "buildable must score candidates through the incremental scorer"

    # Aspirational proposed geometry is deterministic/unchanged across runs (it scores through the
    # full path, which equals network_efficiency -- verified elsewhere -- so the argmax is stable).
    roads_a2 = _greedy_arterials(
        region, half_width_m=DEFAULT_ROAD_WIDTH_M / 2.0,
        mode="aspirational", objective="directness",
                                          max_roads=2, workers=1)
    assert [g.wkt for g in roads_a1.geometry] == [g.wkt for g in roads_a2.geometry]
