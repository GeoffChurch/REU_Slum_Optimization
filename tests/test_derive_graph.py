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
