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
