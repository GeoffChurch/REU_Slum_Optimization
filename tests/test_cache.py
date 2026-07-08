from pathlib import Path

import pytest

import reblock.cache as cache


def test_source_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    a.write_bytes(b"hello")
    h1 = cache.source_hash(a)
    h2 = cache.source_hash(a)
    assert h1 == h2 and h1 != ""
    a.write_bytes(b"HELLO")
    assert cache.source_hash(a) != h1


def test_source_hash_covers_all_paths_order_independent(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    a.write_bytes(b"aaa")
    b = tmp_path / "b.bin"
    b.write_bytes(b"bbb")
    assert cache.source_hash(a, b) == cache.source_hash(b, a)  # sorted internally
    assert cache.source_hash(a, b) != cache.source_hash(a)


def test_key_parts_reports_live_versions() -> None:
    geos, proj, code = cache.key_parts()
    assert geos and proj and code  # all non-empty strings
    assert isinstance(geos, str) and isinstance(code, str)


def test_cached_wrapper_hits_and_key_invalidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Point the joblib Memory at a temp dir so the test never touches ~/.cache.
    import joblib

    monkeypatch.setattr(cache, "memory", joblib.Memory(location=str(tmp_path), verbose=0))
    calls = {"n": 0}

    def _impl(heavy: str, *, key: str) -> int:
        calls["n"] += 1
        return len(heavy) + calls["n"] * 0  # value depends only on heavy, keyed on `key`

    fn = cache.cached(_impl, ignore=["heavy"])
    r1 = fn("abcd", key="k1")
    r2 = fn("XXXX", key="k1")  # same key, different (ignored) heavy -> cache HIT, stale-by-design
    assert calls["n"] == 1 and r1 == r2 == 4  # heavy ignored: 2nd call returns cached r1
    fn("abcd", key="k2")  # different key -> recompute
    assert calls["n"] == 2
