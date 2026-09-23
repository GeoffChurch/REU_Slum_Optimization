"""The research code reaches a derivation only by configuration, and carries its own code into the
cache key when it does.

A derivation's cache key hashes its import closure (`derive_graph._closure_paths`). If any `reblock`
module outside `reblock.transplant` and `reblock.data.pools` imported them, a tweak to a GW constant
would silently invalidate every cached proposal of that module -- and a full examples regeneration
costs hours. Checked over EVERY other module, not just the derivation ones, because a closure is
transitive: any module a derivation might reach later has to be clean now.

The research code does reach shipped derivations now -- the consensus is `demand_greedy` with a
research desire source, and the study's ground truth is `osm_footpaths` reading the pool's
footpaths -- but only as a CONFIGURED FIELD (Hydra `_target_`, typed by `presets.load_research`),
which no import closure sees. Its code enters the key instead through the identity of the
outermost research object in the field, which leads with that object's own closure hash. That is
the boundary: nothing shipped imports the research code, and every entry point it has into a
shipped derivation hashes the code behind it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import reblock
from reblock.data.pools import PoolFootpaths
from reblock.derive_graph import Identified, _closure_paths, _module_file, closure_hash
from reblock.methods.substrates import ChordSubstrate
from reblock.permeability import DEFAULT_ROAD_WIDTH_M
from reblock.transplant.consensus import ConsensusDesireSource
from reblock.transplant.donor_transplant import DonorTransplantReblocker
from reblock.transplant.donors import Donors
from reblock.transplant.snap import RoutedSnap
from tests.permeability_fixtures import SHIPPED
from tests.transplant.pool_fixtures import SIGNATURE, TRANSPORT, pool, slab

SRC = Path(reblock.__file__).resolve().parent
CONF = SRC.parent.parent / "conf"
POOLS = SRC / "data" / "pools.py"
TRANSPLANT = SRC / "transplant"

# Where research code enters a shipped derivation: each is the outermost research object of a
# configured field of a shipped Method (a desire source, a footpath source), or a Method itself.
ENTRY_POINTS: tuple[type, ...] = (ConsensusDesireSource, PoolFootpaths, DonorTransplantReblocker)


def _research(path: Path) -> bool:
    return path == POOLS or TRANSPLANT in path.parents


def _module(path: Path) -> str:
    parts = path.relative_to(SRC).with_suffix("").parts
    return ".".join(("reblock", *(parts[:-1] if parts[-1] == "__init__" else parts)))


def test_no_shipped_module_imports_the_research_code() -> None:
    """FAULT INJECTION: adding `from reblock.transplant.gw import GWParams` to
    `reblock/methods/demand_greedy.py` fails this, naming that module."""
    shipped = [p for p in SRC.rglob("*.py") if not _research(p)]
    assert len(shipped) > 50, "the walk found almost nothing -- src/reblock moved"
    leaks = {_module(p): sorted(str(q.relative_to(SRC)) for q in _closure_paths(_module(p))
                                if _research(q))
             for p in shipped}
    assert {m: r for m, r in leaks.items() if r} == {}


def test_the_research_modules_are_where_this_test_looks() -> None:
    """Otherwise a move would leave the checks here vacuously green."""
    assert POOLS.is_file()
    assert (TRANSPLANT / "gw.py").is_file()
    assert any(_research(q) for q in _closure_paths("reblock.transplant.snap"))


def _configured_research_modules() -> set[str]:
    """The module of every research `_target_` anywhere in conf/, straight from the yaml."""
    out: set[str] = set()
    for y in CONF.rglob("*.yaml"):
        for target in re.findall(r"_target_:\s*([A-Za-z_][\w.]*)", y.read_text()):
            module = target.rsplit(".", 1)[0]
            f = _module_file(module)
            if f is not None and _research(f):
                out.add(module)
    return out


def test_every_configured_research_class_lies_behind_an_entry_point() -> None:
    """A research class conf/ names must be an entry point or sit in one's closure -- or a new one
    could be configured straight into a shipped Method with nothing hashing its code.

    FAULT INJECTION: dropping `DonorTransplantReblocker` from ENTRY_POINTS fails this, naming
    `reblock.transplant.donor_transplant` and `reblock.transplant.snap`, which no other entry
    point's closure reaches.
    """
    configured = _configured_research_modules()
    assert "reblock.transplant.consensus" in configured, "the walk found no research _target_"
    covered = set().union(*(_closure_paths(cls.__module__) for cls in ENTRY_POINTS))
    uncovered = sorted(m for m in configured if _module_file(m) not in covered)
    assert uncovered == []


def _entry_points() -> list[object]:
    blocks = (slab(6, 5, "r", x=0.0, source_hash="iso"), slab(6, 6, "a", x=3000.0,
                                                              source_hash="iso"))
    donors = Donors(pool=pool(blocks, with_paths=blocks, content=("test", "isolation")),
                    exclusion_radius_m=2000.0, k=1, min_donors=1, signature=SIGNATURE,
                    transport=TRANSPORT)
    return [ConsensusDesireSource(donors=donors, permeability=SHIPPED,
                                  road_width_m=DEFAULT_ROAD_WIDTH_M),
            PoolFootpaths(pool=donors.pool),
            DonorTransplantReblocker(donors=donors, snap=RoutedSnap(ChordSubstrate()),
                                     road_width_m=DEFAULT_ROAD_WIDTH_M)]


@pytest.mark.parametrize("entry", _entry_points(), ids=lambda e: type(e).__name__)
def test_an_entry_point_leads_its_identity_with_its_own_closure_hash(entry: object) -> None:
    """What makes a research edit a cache miss for the shipped Method configured with it.

    FAULT INJECTION: returning `config_identity(self)` alone from
    `ConsensusDesireSource.identity` fails this; the consensus's cached proposals would then
    survive any edit to the GW solver.
    """
    assert type(entry) in ENTRY_POINTS and isinstance(entry, Identified)
    identity = entry.identity
    assert isinstance(identity, tuple)
    assert identity[0] == closure_hash(type(entry).__module__)


def test_the_consensus_hashes_the_solver_it_runs() -> None:
    """Its closure -- what its identity's hash is over -- holds every module the extraction
    depends on, the GW solve and the pool included."""
    paths = _closure_paths("reblock.transplant.consensus")
    for f in ("gw.py", "transport.py", "signature.py", "donors.py", "consensus.py"):
        assert TRANSPLANT / f in paths, f
    assert POOLS in paths
