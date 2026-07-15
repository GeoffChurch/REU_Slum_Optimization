import pytest
from shapely.geometry import LineString

from reblock.derive.access import STREET_TOL
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.arterial_lazy import _make_policy
from tests.methods.test_arterial import _grid_block  # reuse the fast grid fixture


def _policy(name, block):
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    return _make_policy(name, block, list(block.streets.geometry), 6, 4, adj)


def test_fixed_policy_never_changes_after_initial():
    block = _grid_block(5)
    pol = _policy("fixed", block)
    assert len(pol.initial()) > 0
    added, removed = pol.after_commit([LineString([(0, 0), (10, 10)])], 1)
    assert added == [] and removed == []


def test_grow_policy_only_adds():
    block = _grid_block(5)
    pol = _policy("grow", block)
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
    from reblock.methods.arterial import _greedy_arterials
    from reblock.methods.arterial_lazy import _greedy_arterials_lazy
    for mode in ("buildable", "aspirational"):
        block = _grid_block(grid_n)
        exact = _greedy_arterials(block, mode=mode, objective="directness", n_anchors=n_anchors,
                                  max_roads=max_roads, workers=1)
        lazy = _greedy_arterials_lazy(block, mode=mode, objective="directness", n_anchors=n_anchors,
                                      top_k=8, lam=2.0, max_roads=max_roads, cost="length",
                                      corridor_m=3.0, workers=1,
                                      candidate_policy="faithful", rescore_every=1)
        assert [g.wkt for g in exact.geometry] == [g.wkt for g in lazy.geometry], \
            (mode, grid_n, n_anchors, max_roads)


def test_faithful_policy_matches_arterial_candidate_set():
    # faithful's set after committing a road must equal arterial's own regeneration for that network
    from reblock.methods.arterial import _anchor_points, _candidate_chords
    block = _grid_block(5)
    pol = _policy("faithful", block)
    base = pol.initial()
    road = LineString([(5.0, 0.0), (5.0, 40.0)])
    live = {ls.wkt for ls in base}
    added, removed = pol.after_commit([road], 1)
    live = (live - set(removed)) | {ls.wkt for ls in added}
    network = list(block.streets.geometry) + [road]
    # deep_targets change with roads; compare through-road structure via arterial's own generator
    expect_roads = {ls.wkt for ls in _candidate_chords(_anchor_points(network, 6), [])}
    assert expect_roads <= live
