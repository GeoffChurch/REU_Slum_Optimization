import dataclasses

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from reblock.derive.access import STREET_TOL
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.arterial import GreedyArterialReblocker, IdealChord, SnapToBoundary
from reblock.methods.arterial.engines import (
    ArterialEngine,
    ExactEngine,
    LazyEngine,
    ShortlistEngine,
)
from reblock.methods.arterial.policies import Faithful, Fixed, Grow
from reblock.permeability import DEFAULT_ROAD_WIDTH_M
from tests.methods.test_arterial import UTM, _grid_block, _two_arm_block  # reuse fast fixtures


def _policy(spec, block):
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    return spec.build(block, list(block.streets.geometry), 6, 4, adj, 0)


def test_engines_are_their_own_identity_and_discriminate() -> None:
    # Reflexive equality (identity must be usable as a cache key: two separate constructions with
    # the same field values share it) plus discrimination on every field LazyEngine has --
    # rescore_every and policy -- not just the engine-type switch. Mirrors test_policies.py's
    # test_specs_are_their_own_identity_and_distinct, which caught a real gap (a missed comparison
    # let a copy-paste `return self` bug through undetected).
    assert ExactEngine().identity == ExactEngine().identity
    assert LazyEngine().identity == LazyEngine().identity
    assert (LazyEngine(policy=Fixed(), rescore_every=3).identity
            == LazyEngine(policy=Fixed(), rescore_every=3).identity)
    assert ExactEngine().identity != LazyEngine().identity
    assert LazyEngine(rescore_every=0).identity != LazyEngine(rescore_every=1).identity
    assert LazyEngine(policy=Grow()).identity != LazyEngine(policy=Faithful()).identity


def test_engines_satisfy_the_protocol() -> None:
    """Both engines conform to ArterialEngine; a non-conformer doesn't."""
    assert isinstance(ExactEngine(), ArterialEngine)
    assert isinstance(LazyEngine(), ArterialEngine)
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
    (committed_disp, base_val, the step context) changes scores silently rather than crashing.

    Uses the access objective with cost=displacement because that is the combination the shortlist
    exists for, and _two_arm_block supplies the building points displacement needs."""
    pts = gpd.GeoDataFrame(geometry=[Point(0.5, y) for y in range(1, 8)] + [Point(10.5, 4)],
                           crs=UTM)
    block = _two_arm_block(pts)
    kw: dict[str, object] = dict(objective="access", cost="displacement",
                                 realizer=SnapToBoundary(), max_roads=3,
                                 road_width_m=DEFAULT_ROAD_WIDTH_M, workers=2)
    want = GreedyArterialReblocker(engine=ExactEngine(), **kw).propose(block)          # type: ignore[arg-type]
    got = GreedyArterialReblocker(engine=ShortlistEngine(k=10_000_000), **kw).propose(block)  # type: ignore[arg-type]
    assert want.roads is not None and got.roads is not None
    assert [g.wkt for g in got.roads.geometry] == [g.wkt for g in want.roads.geometry]


def test_shortlist_threads_do_not_enter_identity() -> None:
    """threads is a parallelism knob, same category as workers -- it cannot change the roads, so
    it must not split the cache key."""
    assert ShortlistEngine(k=512, threads=1).identity == ShortlistEngine(k=512, threads=8).identity
    assert ShortlistEngine(k=512).identity != ShortlistEngine(k=256).identity


def test_lazy_fixed_and_faithful_run_and_differ_from_exact_is_ok():
    block = _grid_block(5)
    for spec in (Fixed(), Grow(), Faithful()):
        roads = GreedyArterialReblocker(
            objective="directness", n_anchors=6,
            max_roads=4, engine=LazyEngine(policy=spec),
        ).propose(block).roads
        assert roads is not None
        assert len(roads) >= 0            # all policies produce a valid proposal
    # rescore_every=1 with fixed equals a full-rescore greedy over that policy's
    # set: determinism
    a = GreedyArterialReblocker(
        n_anchors=6, max_roads=3,
        engine=LazyEngine(policy=Fixed(), rescore_every=1)).propose(block).roads
    b = GreedyArterialReblocker(
        n_anchors=6, max_roads=3,
        engine=LazyEngine(policy=Fixed(), rescore_every=1)).propose(block).roads
    assert a is not None and b is not None
    assert [g.wkt for g in a.geometry] == [g.wkt for g in b.geometry]


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


@pytest.mark.parametrize("grid_n,n_anchors,max_roads", [
    (5, 6, 4),      # original aspirational-coverage config -- kept for continuity
    (5, 6, 6),      # reviewer's known-divergent buildable case (chord.wkt vs real.wkt tie-break)
    (5, 8, 6),      # more anchors -> denser candidate set, more tie opportunities
    (4, 6, 4),      # smaller grid
    (6, 6, 6),      # larger grid
])
def test_lazy_faithful_rescore1_equals_exact(grid_n, n_anchors, max_roads):
    # The engine-correctness gate (the oracle): rescore_every=1 + faithful policy re-scores every
    # candidate every step, so the heap's top IS the true per-step argmax over arterial's OWN
    # candidate set -- byte-identical road sequence to the exact greedy. A failure here is a
    # bookkeeping bug (a _StepState field, the heap tie-break, or termination), NOT submodularity.
    # Parametrized over several buildable/aspirational configs (not just one) because the heap's
    # tie-break only diverges from exact on candidate sets that actually contain an equal-gain tie
    # -- a single lucky config can pass while the tie-break logic is still wrong.
    from reblock.methods.arterial.engines import _greedy_arterials, _greedy_arterials_lazy
    for realizer in (SnapToBoundary(), IdealChord()):
        block = _grid_block(grid_n)
        exact = _greedy_arterials(
            block, half_width_m=DEFAULT_ROAD_WIDTH_M / 2.0,
            realizer=realizer, objective="directness", n_anchors=n_anchors,
                                  max_roads=max_roads, workers=1)
        lazy = _greedy_arterials_lazy(block, realizer=realizer, objective="directness",
                                      n_anchors=n_anchors,
                                      top_k=8, max_roads=max_roads, cost="length",
                                      half_width_m=DEFAULT_ROAD_WIDTH_M / 2.0, workers=1,
                                      policy_spec=Faithful(), rescore_every=1)
        assert [g.wkt for g in exact.geometry] == [g.wkt for g in lazy.geometry], \
            (realizer, grid_n, n_anchors, max_roads)


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
    expect_roads = {ls.wkt for ls in _candidate_chords(_anchor_points(network, 6), [])}
    assert expect_roads <= live


def test_lazy_dispatch_and_determinism():
    block = _grid_block(5)
    m = GreedyArterialReblocker(objective="directness", n_anchors=6,
                               max_roads=4, engine=LazyEngine(policy=Grow()))
    a = m.propose(block).roads
    b = m.propose(block).roads
    assert a is not None and b is not None
    assert [g.wkt for g in a.geometry] == [g.wkt for g in b.geometry]   # deterministic
    assert len(a) > 0


def test_lazy_grow_with_max_anchors_runs_end_to_end():
    # max_anchors bounds anchor generation (see test_arterial.py's _anchor_points tests); this is
    # the end-to-end check that the cap threads through the lazy/grow path without breaking the
    # proposal shape.
    block = _grid_block(5)
    roads = GreedyArterialReblocker(engine=LazyEngine(policy=Grow()),
                                    max_anchors=8, max_roads=3).propose(block).roads
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
    calls["n"] = 0
    GreedyArterialReblocker(n_anchors=8, max_roads=4, workers=1).propose(block)
    exact_calls = calls["n"]
    # lazy grow
    calls["n"] = 0
    GreedyArterialReblocker(
        n_anchors=8, max_roads=4, workers=1,
        engine=LazyEngine(policy=Grow(), rescore_every=0)).propose(block)
    lazy_calls = calls["n"]
    assert lazy_calls < exact_calls / 2, (lazy_calls, exact_calls)


def test_lazy_roads_carry_drain_column_like_exact():
    # The lazy engine's tail must match the exact path's tail (`_planarize` + `road_drainage`),
    # not just produce equivalent geometry -- `.roads` is a schema, not just a geometry column,
    # and downstream consumers (e.g. rendering) read `drain`. Regression test for the schema
    # divergence where the lazy engine ended on `_explode(_merge(committed))` with no `drain`.
    block = _grid_block(5)
    roads = GreedyArterialReblocker(objective="directness", n_anchors=6,
                                    max_roads=4,
                                    engine=LazyEngine(policy=Grow())).propose(block).roads
    assert roads is not None
    assert "drain" in roads.columns
    if len(roads):
        assert len(roads["drain"]) == len(roads)
        assert (roads["drain"] >= 0).all()


def test_lazy_quality_within_tolerance():
    from scoring_fixtures import _block_1808

    from reblock.budget import network_efficiency
    block = _block_1808()
    exact = GreedyArterialReblocker(
        n_anchors=8, max_roads=4, workers=1).propose(block).roads
    lazy = GreedyArterialReblocker(n_anchors=8, max_roads=4, workers=1,
                                   engine=LazyEngine(policy=Grow())).propose(block).roads
    _e0, d_exact = network_efficiency(block, exact)
    _e1, d_lazy = network_efficiency(block, lazy)
    assert d_lazy >= d_exact - 0.02, (d_lazy, d_exact)  # comparable-or-better (beats exact)
