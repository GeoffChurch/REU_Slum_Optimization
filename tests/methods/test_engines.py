import dataclasses
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from reblock.budget import EFFICIENCY_SOURCES
from reblock.derive.access import STREET_TOL, ParcelAdjacency
from reblock.methods.arterial import (
    Access,
    Directness,
    Displacement,
    GreedyArterialReblocker,
    IdealChord,
    Length,
    SnapToBoundary,
)
from reblock.methods.arterial.engines import (
    ArterialEngine,
    ExactEngine,
    LazyEngine,
    ShortlistEngine,
)
from reblock.methods.arterial.policies import Faithful, Fixed, Grow
from reblock.permeability import DEFAULT_ROAD_WIDTH_M
from tests.methods.test_arterial import (  # reuse fast fixtures
    ARTERIAL,
    UTM,
    _grid_block,
    _two_arm_block,
)


def _policy(spec, block):
    return spec.build(ParcelAdjacency.of(block, STREET_TOL), list(block.streets.geometry),
                      n_anchors=6, top_k=4, max_anchors=0)


def test_engines_are_their_own_identity_and_discriminate() -> None:
    # Reflexive equality (identity must be usable as a cache key: two separate constructions with
    # the same field values share it) plus discrimination on every field LazyEngine has --
    # rescore_every and policy -- not just the engine-type switch. A narrower check (e.g. only the
    # engine-type switch) would let a copy-paste `return self` bug -- LazyEngine.identity silently
    # ignoring `policy` or `rescore_every` -- through undetected.
    assert ExactEngine().identity == ExactEngine().identity
    lazy = LazyEngine(policy=Grow(), rescore_every=0)
    assert lazy.identity == LazyEngine(policy=Grow(), rescore_every=0).identity
    assert (LazyEngine(policy=Fixed(), rescore_every=3).identity
            == LazyEngine(policy=Fixed(), rescore_every=3).identity)
    assert ExactEngine().identity != lazy.identity
    assert lazy.identity != replace(lazy, rescore_every=1).identity
    assert lazy.identity != replace(lazy, policy=Faithful()).identity


def test_engines_satisfy_the_protocol() -> None:
    """All three engines conform to ArterialEngine; a non-conformer doesn't."""
    assert isinstance(ExactEngine(), ArterialEngine)
    assert isinstance(LazyEngine(policy=Grow(), rescore_every=0), ArterialEngine)
    assert isinstance(ShortlistEngine(k=512, threads=8), ArterialEngine)
    assert not isinstance(object(), ArterialEngine)


def test_reblocker_has_no_engine_flags_left() -> None:
    """lazy + candidate_policy + rescore_every jointly picked an engine; a fourth would have made
    it worse. They are one injected instance now."""
    fields = {f.name for f in dataclasses.fields(GreedyArterialReblocker)}
    assert {"lazy", "candidate_policy", "rescore_every", "mode", "lam"} & fields == set()
    assert {"engine", "realizer"} <= fields


def test_shortlist_with_non_binding_k_is_the_exact_engine() -> None:
    """The shortlist re-states the exact step loop so an injected ranking can cut the candidate
    list mid-loop. With k above every step's candidate count it must reduce to the exact greedy
    EXACTLY -- the two have separate copies of a dozen per-step setup lines, and dropping one
    (the committed overlap, base_val, the step context) changes scores silently rather than
    crashing.

    Uses the access objective with cost=displacement because that is the combination the shortlist
    exists for, and _two_arm_block supplies the building points displacement needs.

    Bisected 2026-11-04 against handing the shortlist's step an empty committed network: a second
    road is the least that can see it, which the two-arm block commits from arms 6 tall at any
    anchor count from 2 (5-tall arms stop at one road). Chosen 7-tall arms, 3 anchors, 3 roads
    (the run stops at 2, so the terminating step is compared too). The 9-tall, 32-anchor run it
    replaced cost 20x the CPU and caught the same fault."""
    pts = gpd.GeoDataFrame(geometry=[Point(0.5, y) for y in range(1, 6)] + [Point(10.5, 3)],
                           crs=UTM)
    block = _two_arm_block(pts, h=7)
    base = replace(ARTERIAL, objective=Access(), cost=Displacement(), n_anchors=3, max_roads=3,
                   workers=1)
    want = replace(base, engine=ExactEngine()).propose(block)
    got = replace(base, engine=ShortlistEngine(k=10_000_000, threads=8)).propose(block)
    assert want.roads is not None and got.roads is not None
    assert len(want.roads) >= 2, "one road never meets a committed network"
    assert [g.wkt for g in got.roads.geometry] == [g.wkt for g in want.roads.geometry]


def test_shortlist_threads_do_not_enter_identity() -> None:
    """threads is a parallelism knob, same category as workers -- it cannot change the roads, so
    it must not split the cache key."""
    assert ShortlistEngine(k=512, threads=1).identity == ShortlistEngine(k=512, threads=8).identity
    assert ShortlistEngine(k=512, threads=8).identity != ShortlistEngine(k=256, threads=8).identity


@pytest.mark.parametrize("engine", [
    ExactEngine(), LazyEngine(policy=Grow(), rescore_every=0), ShortlistEngine(k=8, threads=1)])
def test_pool_matches_serial_on_every_engine(engine: ArterialEngine,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """`workers` is the one engine parameter allowed a default, because it cannot change the roads.
    That holds only while every engine's fork pool returns what its serial path does -- pinned here
    for all three, not just the exact engine test_arterial.py covers. The threshold is forced to 1
    so every step really forks (at the shipped 128 this block's steps would all run serial), and
    the pool constructor is counted so a run that silently stayed serial fails instead of passing.
    """
    from reblock.methods.arterial import engines
    monkeypatch.setattr(engines, "_PARALLEL_THRESHOLD", 1)
    pools = {"n": 0}

    class CountingPool(ProcessPoolExecutor):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pools["n"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(engines, "ProcessPoolExecutor", CountingPool)
    block = _grid_block(3)
    method = replace(ARTERIAL, n_anchors=6, engine=engine)
    serial = replace(method, workers=1).propose(block).roads
    assert pools["n"] == 0
    pooled = replace(method, workers=2).propose(block).roads
    assert pools["n"] > 0, "the pooled run never forked -- it compared serial with serial"
    assert serial is not None and pooled is not None and len(serial) > 0
    assert [g.wkt for g in pooled.geometry] == [g.wkt for g in serial.geometry]


def test_lazy_fixed_and_faithful_run_and_differ_from_exact_is_ok():
    # Every policy driven through the engine end to end: each policy's initial/after_commit
    # round-trip runs inside the engine and commits roads. A 2x2 grid commits one road and then
    # asks after_commit for the next step's candidates, which is the whole round trip; 2 roads is
    # the least cap that reaches it. Chosen 3 anchors, 3 roads.
    block = _grid_block(2)
    for spec in (Fixed(), Grow(), Faithful()):
        roads = replace(
            ARTERIAL, objective=Directness(), n_anchors=3,
            max_roads=3, engine=LazyEngine(policy=spec, rescore_every=0), workers=1,
        ).propose(block).roads
        assert roads is not None
        assert len(roads) > 0, spec       # every policy commits roads
    # The fixed/rescore_every=1 determinism check that ended this test is gone: both heap pushes
    # are shared by every policy and rescore setting, and a coin-flip tie-break at either one fails
    # test_lazy_dispatch_and_determinism 30 times of 30 (2026-11-04), as it failed this.


def test_fixed_policy_never_changes_after_initial():
    block = _grid_block(5)
    pol = _policy(Fixed(), block)
    assert len(pol.initial()) > 0
    added, removed = pol.after_commit([LineString([(0, 0), (10, 10)])], 1)
    assert added == [] and removed == []


def test_grow_policy_only_adds():
    block = _grid_block(5)
    pol = _policy(Grow(), block)
    base = pol.initial()
    added, removed = pol.after_commit([LineString([(0, 0), (10, 10)])], 1)
    assert removed == []                       # grow removes nothing
    base_keys = {ls.wkt for ls in base}
    assert all(ls.wkt not in base_keys for ls in added)   # only genuinely new candidates


@pytest.mark.parametrize("grid_n,n_anchors,max_roads,realizer", [
    (4, 3, 7, SnapToBoundary(lam=2.0)),   # an equal-gain tie at road 6: real.wkt vs chord.wkt
    (2, 3, 3, SnapToBoundary(lam=2.0)),   # runs out of positive gains after one road: termination
    (2, 3, 3, IdealChord()),              # termination again, on the aspirational realizer
], ids=["tie-snap", "stop-snap", "stop-ideal"])
def test_lazy_faithful_rescore1_equals_exact(grid_n, n_anchors, max_roads, realizer):
    # The engine-correctness gate (the oracle): rescore_every=1 + faithful policy re-scores every
    # candidate every step, so the heap's top IS the true per-step argmax over arterial's OWN
    # candidate set -- byte-identical road sequence to the exact greedy. A failure here is a
    # bookkeeping bug (a _StepState field, the heap tie-break, or termination), NOT submodularity.
    #
    # One case per bug, each bisected to the smallest config that catches it and then one step
    # past it on every parameter (fault-injected 2026-11-04):
    #   - ordering the heap by chord.wkt instead of real.wkt fails `tie-snap`. It needs a snapping
    #     realizer (whose `real` differs from its chord) and a candidate set with an equal-gain tie:
    #     the smallest is a 4x4 grid at road 6, with any anchor count down to 2 (3x3 grids and
    #     5-road runs never tie). Chosen: 3 anchors, 7 roads.
    #   - continuing on zero gain (`<= 0.0` -> `< 0.0`) fails both `stop-*` cases: a 2x2 grid runs
    #     out of positive gains after one road, so max_roads=2 is the least that reaches the
    #     termination step, at any anchor count down to 2. Chosen: 3 anchors, 3 roads.
    #   - dropping this step's `after_commit` additions on a full re-score fails `tie-snap` (it
    #     needs a 3x3 grid or larger; the 2x2 cases commit too few roads to re-score).
    # The 10 cases this replaced (5 configs x 2 realizers) cost 232 s of CPU and caught only the
    # first and last of these three bugs.
    from reblock.methods.arterial.engines import _greedy_arterials, _greedy_arterials_lazy
    block = _grid_block(grid_n)
    exact = _greedy_arterials(
        block, half_width_m=DEFAULT_ROAD_WIDTH_M / 2.0,
        realizer=realizer, objective=Directness(), cost=Length(), n_anchors=n_anchors,
        max_roads=max_roads, workers=1, top_k=8, max_anchors=0)
    lazy = _greedy_arterials_lazy(block, realizer=realizer, objective=Directness(),
                                  n_anchors=n_anchors,
                                  top_k=8, max_roads=max_roads, cost=Length(),
                                  half_width_m=DEFAULT_ROAD_WIDTH_M / 2.0, workers=1,
                                  policy_spec=Faithful(), rescore_every=1, max_anchors=0)
    assert [g.wkt for g in exact.geometry] == [g.wkt for g in lazy.geometry]


def test_faithful_policy_matches_arterial_candidate_set():
    # faithful's set after committing a road must equal arterial's own regeneration for that network
    from reblock.methods.arterial.primitives import _anchor_points, _candidate_chords
    block = _grid_block(5)
    pol = _policy(Faithful(), block)
    base = pol.initial()
    road = LineString([(5.0, 0.0), (5.0, 40.0)])
    live = {ls.wkt for ls in base}
    added, removed = pol.after_commit([road], 1)
    live = (live - set(removed)) | {ls.wkt for ls in added}
    network = list(block.streets.geometry) + [road]
    # deep_targets change with roads; compare through-road structure via arterial's own generator
    anchors = _anchor_points(network, n=6, max_anchors=0)
    expect_roads = {ls.wkt for ls in _candidate_chords(anchors, [])}
    assert expect_roads <= live


def test_lazy_dispatch_and_determinism():
    # A 2x2 grid, where a coin-flip heap tie-break (fault-injected 2026-11-04) lands on one of 4
    # road sets, none above 0.3 of 60 draws: 10 runs agree by chance under 3e-5 of the time (7 is
    # the least for 1e-3). The 5x5 / 6-anchor / 4-road block it ran on before produced one road
    # set in 60 coin-flip runs: it could not fail.
    block = _grid_block(2)
    m = replace(ARTERIAL, objective=Directness(), n_anchors=3, max_roads=3,
                engine=LazyEngine(policy=Grow(), rescore_every=0), workers=1)
    runs = []
    for _ in range(10):
        roads = m.propose(block).roads
        assert roads is not None
        runs.append([g.wkt for g in roads.geometry])
    assert runs[0], "no roads: nothing left to be deterministic about"
    assert all(r == runs[0] for r in runs[1:])   # deterministic


def test_lazy_grow_with_max_anchors_runs_end_to_end():
    # max_anchors bounds anchor generation (see test_arterial.py's _anchor_points tests); this is
    # the end-to-end check that the cap threads through the lazy/grow path without breaking the
    # proposal shape.
    # A 2x2 grid's 8 vertices plus 32 samples are well over the cap of 8, so the cap binds.
    block = _grid_block(2)
    roads = replace(ARTERIAL, engine=LazyEngine(policy=Grow(), rescore_every=0),
                    max_anchors=8, max_roads=2, workers=1).propose(block).roads
    assert roads is not None
    assert len(roads) >= 0
    assert "drain" in roads.columns


def test_lazy_far_fewer_scorings_than_exact(monkeypatch):
    # instrument eval_candidate call count on a real block where arterial runs
    from scoring_fixtures import _block_1808

    from reblock.methods.arterial import engines
    block = _block_1808()
    calls = {"n": 0}
    real_eval = engines.eval_candidate
    def counting(chord):
        calls["n"] += 1
        return real_eval(chord)
    # `_greedy_arterials` (exact) and `_greedy_arterials_lazy` now live in the same module and
    # share this one imported `eval_candidate` binding, so a single patch covers both engines.
    monkeypatch.setattr(engines, "eval_candidate", counting)
    # exact
    # 3 anchors, 4 roads. Re-scoring every live candidate on every step (the lazy engine
    # degenerating to exact) fails this at every count tried from 3 anchors / 3 roads; with
    # fewer than 4 roads, or 4 or more anchors at 3 roads, lazy does not reach half on this block
    # even when it is right, because step 0 scores everything in both engines.
    calls["n"] = 0
    replace(ARTERIAL, n_anchors=3, max_roads=4, workers=1).propose(block)
    exact_calls = calls["n"]
    # lazy grow
    calls["n"] = 0
    replace(
        ARTERIAL, n_anchors=3, max_roads=4, workers=1,
        engine=LazyEngine(policy=Grow(), rescore_every=0)).propose(block)
    lazy_calls = calls["n"]
    # A lower bound, not just an upper one: without it, a future change that moves the lazy engine
    # off `engines.eval_candidate` would silently drive lazy_calls to 0 and this assertion would
    # keep passing vacuously -- 0 < exact_calls / 2 is true for any positive exact_calls.
    assert lazy_calls > 0
    assert lazy_calls < exact_calls / 2, (lazy_calls, exact_calls)


def test_lazy_roads_carry_drain_column_like_exact():
    # The lazy engine's tail must match the exact path's tail (`_planarize` + `road_drainage`),
    # not just produce equivalent geometry -- `.roads` is a schema, not just a geometry column,
    # and downstream consumers (e.g. rendering) read `drain`. Regression test for the schema
    # divergence where the lazy engine ended on `_explode(_merge(committed))` with no `drain`.
    # A schema check, so the block size does not matter: dropping the lazy engine's `drain` line
    # fails it on one road on a 2x2 grid.
    block = _grid_block(2)
    roads = replace(ARTERIAL, objective=Directness(), n_anchors=3, max_roads=1,
                    engine=LazyEngine(policy=Grow(), rescore_every=0), workers=1,
                    ).propose(block).roads
    assert roads is not None
    assert "drain" in roads.columns
    if len(roads):
        assert len(roads["drain"]) == len(roads)
        assert (roads["drain"] >= 0).all()


def test_lazy_quality_within_tolerance():
    from scoring_fixtures import _block_1808

    from reblock.budget import network_efficiency
    block = _block_1808()
    # 4 anchors, 3 roads. Fault-injected 2026-11-04: a lazy engine that never re-scores a stale
    # heap top (`at == step` -> `True`) fails this from 3 anchors / 2 roads; the 8-anchor /
    # 4-road run it replaced passed it.
    exact = replace(
        ARTERIAL, n_anchors=4, max_roads=3, workers=1).propose(block).roads
    lazy = replace(ARTERIAL, n_anchors=4, max_roads=3, workers=1,
                             engine=LazyEngine(policy=Grow(),
                                               rescore_every=0)).propose(block).roads
    _e0, d_exact = network_efficiency(block, exact, tol=STREET_TOL, k=EFFICIENCY_SOURCES)
    _e1, d_lazy = network_efficiency(block, lazy, tol=STREET_TOL, k=EFFICIENCY_SOURCES)
    assert d_lazy >= d_exact - 0.02, (d_lazy, d_exact)  # comparable-or-better (beats exact)
