from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import joblib
import pytest

import reblock.derive_graph as dg


@dataclass(frozen=True)
class _Datum:
    tag: str
    @property
    def identity(self) -> str:
        return self.tag


class _NoIdentity:
    pass


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(dg, "memory", joblib.Memory(location=str(tmp_path), verbose=0))
    monkeypatch.setattr(dg, "_l2", dg.memory.cache(dg._l2_impl, ignore=["fn", "inputs"]))
    dg.clear_l1()
    yield
    dg.clear_l1()


def _count(box: dict[str, int]) -> Callable[[_Datum], str]:
    def f(x: _Datum) -> str:
        box["n"] += 1
        return x.identity.upper()
    return f


def test_derive_hits_l1_on_repeat() -> None:
    box = {"n": 0}
    fn = _count(box)
    a = _Datum("a")
    assert dg.derive(fn, a) == "A"
    assert dg.derive(fn, a) == "A"   # L1 hit
    assert box["n"] == 1


def test_derive_serves_from_l2_after_l1_cleared() -> None:
    box = {"n": 0}
    fn = _count(box)
    a = _Datum("a")
    dg.derive(fn, a)
    dg.clear_l1()                    # drop memory layer; L2 disk remains
    assert dg.derive(fn, a) == "A"   # L2 hit -> no recompute
    assert box["n"] == 1


def test_distinct_identity_is_distinct_key() -> None:
    box = {"n": 0}
    fn = _count(box)
    dg.derive(fn, _Datum("a"))
    dg.derive(fn, _Datum("b"))       # different identity -> recompute
    assert box["n"] == 2


def test_missing_identity_bypasses_cache(tmp_path: Path) -> None:
    box = {"n": 0}

    def fn(x: _NoIdentity) -> int:
        box["n"] += 1
        return 42
    dg.derive(fn, _NoIdentity())
    dg.derive(fn, _NoIdentity())     # no identity -> never cached
    assert box["n"] == 2
    assert not dg._L1                 # nothing stored in L1


def test_version_bump_forces_a_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    box = {"n": 0}
    fn = _count(box)
    a = _Datum("a")
    dg.derive(fn, a)
    # simulate a derivation-logic / lib change: version() returns a new tag
    monkeypatch.setattr(dg, "version", lambda: ("CHANGED", "g", "p"))
    dg.clear_l1()
    dg.derive(fn, a)                 # new version -> new key -> recompute
    assert box["n"] == 2


def test_source_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    a.write_bytes(b"hello")
    h1 = dg.source_hash(a)
    h2 = dg.source_hash(a)
    assert h1 == h2 and h1 != ""
    a.write_bytes(b"HELLO")
    assert dg.source_hash(a) != h1


def test_source_hash_covers_all_paths_order_independent(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    a.write_bytes(b"aaa")
    b = tmp_path / "b.bin"
    b.write_bytes(b"bbb")
    assert dg.source_hash(a, b) == dg.source_hash(b, a)   # sorted internally
    assert dg.source_hash(a, b) != dg.source_hash(a)


def test_every_method_module_is_hashed_into_the_cache_key() -> None:
    """`derivations.propose` caches ANY method, so every method module must be in the key.

    This used to be a hand-maintained list and it went stale silently: it named exactly the methods
    that existed when it was written, so edits to every later one were invisible. A real
    `segment_displacement` fix in resistance_lp changed its output on a direct call and changed
    nothing through the cache -- a full examples regeneration wrote 0 new entries and republished
    pre-fix results without a word.

    FAULT INJECTION: drop the `methods/*.py` glob back to a hand-list and this fails, naming every
    module the list forgot.
    """
    from pathlib import Path

    from reblock.derive_graph import _DERIVATION_MODULES

    methods_dir = Path(__file__).resolve().parents[1] / "src" / "reblock" / "methods"
    on_disk = {p.name for p in methods_dir.glob("*.py")}
    hashed = {p.name for p in _DERIVATION_MODULES}
    missing = sorted(on_disk - hashed)
    assert not missing, f"method modules absent from the cache key: {missing}"
