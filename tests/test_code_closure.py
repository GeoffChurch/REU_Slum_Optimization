"""A code change must invalidate the derivations it can reach, and no others.

`derive_graph` hashed every derivation module into ONE `_CODE_HASH`, so editing any of them
invalidated every cached proposal. Measured on the `depth` region: `cycle_native` (74 min) and
the arterial (64 min) are 93% of that variant's proposal time, and a change touching only the
scoring half of `budget.py` cannot affect `cycle_native` -- which imports `displacement` and
nothing else from it -- yet invalidated it anyway.

These tests pin the property that decides invalidation: which module files a derivation's key
is computed over.
"""
from __future__ import annotations

from pathlib import Path

from reblock.derive_graph import _closure_paths
from reblock.methods.substrates import ChordSubstrate
from reblock.permeability import DEFAULT_ROAD_WIDTH_M
from tests.permeability_fixtures import SHIPPED

SRC = Path(__file__).resolve().parent.parent / "src" / "reblock"


def test_a_methods_closure_contains_itself() -> None:
    # FAULT INJECTION: a closure that stops at the entry module (no transitive walk) still
    # passes this one; the sibling-exclusion test below is what needs the walk to be correct.
    paths = _closure_paths("reblock.methods.cycle_native")
    assert SRC / "methods" / "cycle_native.py" in paths


def test_a_methods_closure_excludes_its_siblings() -> None:
    """The whole point: `cycle_native` and `resistance_lp` cannot reach each other, so an edit
    to one must not invalidate the other.

    FAULT INJECTION: returning every derivation module (today's global `_CODE_HASH` set) makes
    this fail, since that set contains every sibling.
    """
    paths = _closure_paths("reblock.methods.cycle_native")
    assert SRC / "methods" / "resistance_lp.py" not in paths
    assert SRC / "methods" / "euclidean_grid.py" not in paths
    assert SRC / "methods" / "topology.py" not in paths


def test_the_closure_is_transitive() -> None:
    """`cycle_native` imports `permeability` and `budget` (via `displacement`), so an edit to
    either must still reach it.

    FAULT INJECTION: a non-transitive closure (direct imports only) drops `mesh.py`, which
    `permeability` imports and `cycle_native` does not.
    """
    paths = _closure_paths("reblock.methods.cycle_native")
    assert SRC / "permeability.py" in paths
    assert SRC / "budget.py" in paths
    assert SRC / "mesh.py" in paths, "transitive: permeability imports mesh, cycle_native does not"


def test_nested_and_local_imports_are_followed() -> None:
    """`derivations._screen_selection_impl` imports `screen.dense_compact` INSIDE the function
    body to dodge a cycle, and that module is what the screen's selection logic lives in.

    A module-level-only scan misses it, and `ScreenSelectionInput`'s own docstring records that
    the selection logic is hashed in "because screen/dense_compact.py and metric.py are in
    _DERIVATION_MODULES" -- i.e. it was relying on the global hash exactly here.

    FAULT INJECTION: walking only module-level `Import`/`ImportFrom` nodes fails this.
    """
    paths = _closure_paths("reblock.derivations")
    assert SRC / "screen" / "dense_compact.py" in paths
    assert SRC / "metric.py" in paths


def test_a_methods_code_version_is_blind_to_its_siblings() -> None:
    """The behaviour the whole change exists for, at the level the cache key uses.

    `_propose_impl` never imports any method module -- the method arrives as an argument -- so
    only `type(method).__module__` reveals which code actually runs. Two different methods must
    therefore hash over two different file sets, neither containing the other's module.

    FAULT INJECTION: dropping the input type-modules from `_code_version` (hashing only `fn`'s
    own closure) makes both sets identical and this fails on the first assertion -- and would
    have been the silent-staleness bug, since editing a method would then not invalidate it.
    """
    from reblock.derivations import _propose_impl
    from reblock.derive_graph import _closure_paths, _code_version
    from reblock.methods.clearance import ClearanceReblocker
    from reblock.methods.cycle_native import CycleNativeReblocker

    def modules(inst: object) -> frozenset[Path]:
        return _closure_paths(type(inst).__module__) | _closure_paths(_propose_impl.__module__)

    clearance = modules(ClearanceReblocker(substrate=ChordSubstrate(), repulsion=0.0,
                                           depth_target=2, max_roads=400,
                                           road_width_m=DEFAULT_ROAD_WIDTH_M))
    cycle = modules(CycleNativeReblocker(substrate=ChordSubstrate(), max_displacement=0.10,
                                         max_cycles=400, shortlist=8, params=SHIPPED,
                                         road_width_m=DEFAULT_ROAD_WIDTH_M))
    assert clearance != cycle
    assert SRC / "methods" / "cycle_native.py" not in clearance
    assert SRC / "methods" / "clearance.py" not in cycle

    a = _code_version(_propose_impl, (ClearanceReblocker(substrate=ChordSubstrate(), repulsion=0.0,
                                                         depth_target=2, max_roads=400,
                                                         road_width_m=DEFAULT_ROAD_WIDTH_M),))
    b = _code_version(_propose_impl, (CycleNativeReblocker(substrate=ChordSubstrate(),
                                                           max_displacement=0.10, max_cycles=400,
                                                           shortlist=8, params=SHIPPED,
                                                           road_width_m=DEFAULT_ROAD_WIDTH_M),))
    assert a != b, "two methods must not share a code version"


def _target_modules() -> set[str]:
    """Every `_target_` class's module named anywhere in `conf/`, straight from the yaml."""
    import re
    conf = Path(__file__).resolve().parent.parent / "conf"
    out: set[str] = set()
    for y in conf.rglob("*.yaml"):
        for m in re.findall(r"_target_:\s*([A-Za-z_][\w.]*)", y.read_text()):
            out.add(m.rsplit(".", 1)[0])
    return out


def test_every_configurable_strategy_can_invalidate_something() -> None:
    """No `_target_` class may be invisible to every cache key.

    This is the guard for the case neither runtime mechanism can see: a NEW strategy whose
    module is outside every import closure AND which carries no explicit code hash. Its code
    could then change with no key moving -- silent staleness, the exact bug that made the
    global `_CODE_HASH` a deliberate retreat in the first place.

    Enumerated from `conf/` rather than from a list here, so adding a strategy to the config
    without covering it fails HERE rather than in a stale artifact months later.

    FAULT INJECTION: a `_closure_paths` that stops at the entry module drops most of these.
    """
    from reblock.derivations import ScreenSelectionInput, VoronoiInput, _propose_impl
    from reblock.derive_graph import _closure_paths

    # Everything a key is computed over: the derivation bodies, plus every type that reaches
    # `derive` as a top-level input (`_code_version` adds those), plus the carriers' own
    # closures (their fields' strategies ride the identity).
    covered: set[Path] = set()
    for mod in (_propose_impl.__module__, VoronoiInput.__module__,
                ScreenSelectionInput.__module__, "reblock.pipeline", "reblock.run"):
        covered |= _closure_paths(mod)
    # A Method reaches `derive` as a TOP-LEVEL input, so `_code_version` folds in its own
    # closure -- and with it anything it holds as a field (a desire-line source, a substrate).
    # Not circular: this models what `_code_version` actually does at runtime.
    covered |= {f for m in _target_modules() if m.startswith("reblock.methods.")
                for f in _closure_paths(m)}

    uncovered = sorted(m for m in _target_modules()
                       if (f := _closure_paths(m)) and not (f & covered))
    assert not uncovered, (
        "these configurable strategies are in no cache key's closure, so editing them would "
        f"invalidate nothing: {uncovered}")
