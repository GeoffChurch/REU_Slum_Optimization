import importlib.metadata
import re
import tomllib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import pytest

import reblock.derive_graph as dg

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _Datum:
    tag: str
    @property
    def identity(self) -> str:
        return self.tag


class _Uncacheable:
    """What a synthetic block or a live source is: an input that answers `identity` with None."""

    @property
    def identity(self) -> None:
        return None


class _NoIdentity:
    pass


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(dg, "memory", joblib.Memory(location=str(tmp_path), verbose=0))
    monkeypatch.setattr(dg, "_l2", dg.memory.cache(dg._l2_derive, ignore=["fn", "inputs"]))
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


def test_none_identity_bypasses_cache(tmp_path: Path) -> None:
    box = {"n": 0}

    def fn(x: _Uncacheable) -> int:
        box["n"] += 1
        return 42
    dg.derive(fn, _Uncacheable())
    dg.derive(fn, _Uncacheable())    # identity None -> never cached
    assert box["n"] == 2
    assert not dg._L1                 # nothing stored in L1


def test_an_input_that_declares_no_identity_is_an_error() -> None:
    """Every input `derive` receives declares `identity`, so one that does not is a bug in the
    caller -- not an uncacheable input to recompute quietly, forever, while looking cached."""
    with pytest.raises(AttributeError, match="identity"):
        dg.derive(lambda x: x, _NoIdentity())     # type: ignore[arg-type]


@pytest.mark.parametrize("change", [   # each field by name, so a field renamed or dropped fails
    pytest.param(lambda e: replace(e, python="CHANGED"), id="python"),
    pytest.param(lambda e: replace(e, geos="CHANGED"), id="geos"),
    pytest.param(lambda e: replace(e, proj="CHANGED"), id="proj"),
    pytest.param(lambda e: replace(e, gdal="CHANGED"), id="gdal"),
])
def test_a_runtime_or_native_library_change_forces_a_miss(
        monkeypatch: pytest.MonkeyPatch,
        change: Callable[[dg.EnvVersions], dg.EnvVersions]) -> None:
    """Python, GEOS, PROJ and GDAL (the files read through pyogrio) are each in the key alone."""
    box = {"n": 0}
    fn = _count(box)
    a = _Datum("a")
    dg.derive(fn, a)
    # simulate one library change. (A derivation-LOGIC change is no longer global -- see
    # tests/test_code_closure.py.)
    changed = change(dg.env_version())
    monkeypatch.setattr(dg, "env_version", lambda: changed)
    dg.clear_l1()
    dg.derive(fn, a)                 # new version -> new key -> recompute
    assert box["n"] == 2


def test_upgrading_a_keyed_package_forces_a_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scipy upgrade (any keyed package) changes the key: the cache never serves what the old
    version computed."""
    box = {"n": 0}
    fn = _count(box)
    a = _Datum("a")
    dg.derive(fn, a)
    env = dg.env_version()
    assert ("scipy", importlib.metadata.version("scipy")) in env.packages
    upgraded = tuple((n, "99.0" if n == "scipy" else v) for n, v in env.packages)
    monkeypatch.setattr(dg, "env_version", lambda: replace(env, packages=upgraded))
    dg.clear_l1()
    dg.derive(fn, a)
    assert box["n"] == 2


# The runtime packages NOT in the key, each with why it cannot change a derivation's result.
# Spelled out, like KEYED_PACKAGES: exempting a package is a decision, made here.
KEY_EXEMPT = {
    "hydra-core",   # composes configuration before any derivation runs; the values it produces
                    # reach keys through `config_identity`
    "joblib",       # the cache itself
    "matplotlib",   # rendering only: imported by run/render/animate, in no derivation's closure
    "segno",        # QR codes for the site
    "topology",     # covered as SOURCE: the closure walk follows topology.* (test_code_closure)
}


def test_every_runtime_package_is_keyed_or_exempt() -> None:
    """A new package in pyproject.toml's `runtime` group fails here until it is classified: in
    derive_graph.KEYED_PACKAGES, or in KEY_EXEMPT above with the reason."""
    def norm(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = {norm(re.split(r"[\s<>=!~;\[]", spec, maxsplit=1)[0])    # the name, sans specifier
               for spec in pyproject["dependency-groups"]["runtime"]}
    keyed = {norm(n) for n in dg.KEYED_PACKAGES}
    assert not keyed & KEY_EXEMPT, f"both keyed and exempt: {sorted(keyed & KEY_EXEMPT)}"
    assert runtime == keyed | KEY_EXEMPT, (
        f"unclassified: {sorted(runtime - keyed - KEY_EXEMPT)}; "
        f"not in the runtime group: {sorted((keyed | KEY_EXEMPT) - runtime)}")


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


def test_every_method_module_can_be_reached_by_a_cache_key() -> None:
    """`derivations.propose` caches ANY method, so every method module must be in the key.

    This used to be a hand-maintained list and it went stale silently: it named exactly the methods
    that existed when it was written, so edits to every later one were invisible. A real
    `segment_displacement` fix in resistance_lp changed its output on a direct call and changed
    nothing through the cache -- a full examples regeneration wrote 0 new entries and republished
    pre-fix results without a word.

    FAULT INJECTION: a `_closure_paths` that returns only its entry module makes this fail,
    naming every method module nothing else imports.
    """
    import re
    from pathlib import Path

    from reblock.derive_graph import _closure_paths

    root = Path(__file__).resolve().parents[1]
    configured = {m.rsplit(".", 1)[0]
                  for y in (root / "conf").rglob("*.yaml")
                  for m in re.findall(r"_target_:\s*([A-Za-z_][\w.]*)", y.read_text())
                  if m.startswith("reblock.methods.")}
    # A Method reaches `derive` as a top-level input, so `_code_version` hashes over its own
    # closure. Every method module on disk must therefore sit inside SOME configured method's
    # closure, or nothing that runs can invalidate it.
    reachable = {f for m in configured for f in _closure_paths(m)}
    on_disk = set((root / "src" / "reblock" / "methods").glob("*.py")) - {
        (root / "src" / "reblock" / "methods" / "__init__.py")}
    missing = sorted(p.name for p in on_disk - reachable)
    assert not missing, f"method modules no cache key can reach: {missing}"
