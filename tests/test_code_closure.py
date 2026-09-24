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

import importlib
import re
from pathlib import Path
from typing import Protocol, runtime_checkable
from unittest import mock

from reblock import derive_graph
from reblock.contracts import Block, Proposal, Source
from reblock.data.shapefile import ShapefileSource
from reblock.derive_graph import Identified, _closure_paths, _module_file, closure_hash
from reblock.methods.substrates import ChordSubstrate
from reblock.permeability import DEFAULT_ROAD_WIDTH_M
from tests.data.test_shapefile_source import PHULE
from tests.methods.betweenness.test_source import BASE as BETWEENNESS
from tests.permeability_fixtures import SHIPPED
from tests.transplant.test_isolation import ENTRY_POINTS

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


def _targets() -> list[object]:
    """Every `_target_` named anywhere in `conf/`, straight from the yaml, imported. The names
    are an open set parsed from files, so each is resolved here, once, with no default: a
    `_target_` that no longer imports fails this module instead of dropping out of the check."""
    conf = Path(__file__).resolve().parent.parent / "conf"
    names = {t for y in conf.rglob("*.yaml")
             for t in re.findall(r"_target_:\s*([A-Za-z_][\w.]*)", y.read_text())}
    out: list[object] = []
    for name in sorted(names):
        module, _, attr = name.rpartition(".")
        out.append(getattr(importlib.import_module(module), attr))
    return out


@runtime_checkable
class _Proposes(Protocol):
    """`contracts.Method` less its `identity` property, which `issubclass` cannot test."""

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal: ...


# Strategies a Method holds as a FIELD whose own identity leads with their closure hash, so that
# closure rides the Method's key. Credited below only while the hash is actually there.
SELF_HASHED_FIELDS: tuple[Identified, ...] = (BETWEENNESS,)


def _leads_with_its_own_closure_hash(strategy: Identified) -> bool:
    identity = strategy.identity
    return isinstance(identity, tuple) and identity[0] == closure_hash(type(strategy).__module__)


# Sources whose Blocks carry their reader's closure in their identity (`derive_graph.reader_hash`),
# so that closure rides every key over those Blocks. Credited below only while it actually does.
READER_HASHED_SOURCES: tuple[Source, ...] = (
    ShapefileSource(PHULE, "phule", assumed_crs=3857, block_ids=["phule_0"]),)


def _identity_moves_with_its_reader(source: Source) -> bool:
    """Does a Block's identity change when its reader's code does? Simulated by dropping the
    reader's own file from its closure, which changes the hashed bytes as an edit would."""
    module = type(source).__module__
    real, own = _closure_paths, _module_file(module)

    def edited(m: str) -> frozenset[Path]:
        return real(m) - {own} if m == module else real(m)

    before = next(iter(source.region().blocks)).identity
    with mock.patch.object(derive_graph, "_closure_paths", edited):
        after = next(iter(source.region().blocks)).identity
    return before is not None and before != after


def test_every_configurable_strategy_can_invalidate_something() -> None:
    """No `_target_` class may be invisible to every cache key.

    This is the guard for the case neither runtime mechanism can see: a NEW strategy whose
    module is outside every import closure AND which carries no explicit code hash. Its code
    could then change with no key moving -- silent staleness, the exact bug that made the
    global `_CODE_HASH` a deliberate retreat in the first place.

    Enumerated from `conf/` rather than from a list here, so adding a strategy to the config
    without covering it fails HERE rather than in a stale artifact months later.

    FAULT INJECTION: returning `config_identity(self, exempt=...)` alone from
    `BetweennessDesire.identity` fails this, naming `reblock.methods.betweenness.source` and
    `.contrast` -- a field strategy no key's closure reaches. So does a `_closure_paths` that
    stops at the entry module. So does `ShapefileSource` hashing its file alone
    (`source_hash(self.path)`), naming `reblock.data.shapefile`.
    """
    from reblock.derivations import ScreenSelectionInput, VoronoiInput, _propose_impl

    targets = _targets()
    # Everything a key is computed over: the derivation bodies, plus every type that reaches
    # `derive` as a top-level input (`_code_version` adds those), plus the carriers' own
    # closures (their fields' strategies ride the identity).
    covered: set[Path] = set()
    for mod in (_propose_impl.__module__, VoronoiInput.__module__,
                ScreenSelectionInput.__module__, "reblock.pipeline", "reblock.run"):
        covered |= _closure_paths(mod)
    # A Method reaches `derive` as a TOP-LEVEL input (a refiner's `base` through its own nested
    # `propose`), so `_code_version` folds in its closure. A strategy it holds as a field does
    # NOT: `_code_version` sees only `type(method).__module__`, so the field is covered only if
    # that closure imports it or the field's own identity carries a code hash (next).
    covered |= {f for t in targets if isinstance(t, type) and issubclass(t, _Proposes)
                for f in _closure_paths(t.__module__)}
    covered |= {f for s in SELF_HASHED_FIELDS if _leads_with_its_own_closure_hash(s)
                for f in _closure_paths(type(s).__module__)}
    # The research code is in no shipped closure by design. It reaches a derivation as a
    # configured field, through an entry point whose identity leads with its own closure hash
    # (`tests/transplant/test_isolation.py`), so that closure rides the key instead.
    covered |= {f for cls in ENTRY_POINTS for f in _closure_paths(cls.__module__)}
    # A Source's code shapes every Block it yields, and reaches a key through the Block's identity.
    covered |= {f for s in READER_HASHED_SOURCES if _identity_moves_with_its_reader(s)
                for f in _closure_paths(type(s).__module__)}

    modules = {t.__module__ for t in targets}
    sources = {t.__module__ for t in targets if isinstance(t, type) and issubclass(t, Source)}
    # STRICT for the Methods, the strategies they hold, and the Sources: the module that DEFINES
    # the class must itself be in a key's closure. Sharing any file with one (the check below) is
    # not enough -- every such module imports `contracts`, which every key holds.
    strict = sorted(m for m in modules if (m.startswith("reblock.methods.") or m in sources)
                    and _module_file(m) not in covered)
    # Elsewhere the original, weaker check stands. Made strict everywhere, it would also flag the
    # evals and `screen.identity`, which are never cached, so no hole.
    weak = sorted(m for m in modules if not m.startswith("reblock.methods.") and m not in sources
                  and (f := _closure_paths(m)) and not (f & covered))
    assert not strict + weak, (
        "these configurable strategies are in no cache key's closure, so editing them would "
        f"invalidate nothing: {strict + weak}")
