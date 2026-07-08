from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
import pytest
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely.geometry import Polygon

import reblock.cache as cache
from reblock.contracts import Block
from reblock.derive.access import parcel_access_layers

_UTM = CRS.from_epsg(32643)


def _grid_block(hash_: str) -> Block:
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(3) for j in range(3)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(9))}, geometry=polys, crs=_UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=_UTM)
    return Block(block_id="g", crs=_UTM, boundary=boundary, parcels=parcels,
                 streets=streets, source_content_hash=hash_)


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


def test_cached_access_before_after_use_distinct_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import joblib

    monkeypatch.setattr(cache, "memory", joblib.Memory(location=str(tmp_path), verbose=0))
    calls = {"n": 0}

    def spy(block: Block, roads: GeoDataFrame | None = None, **kw: object) -> pd.Series:
        calls["n"] += 1
        return parcel_access_layers(block, roads)

    monkeypatch.setattr(cache, "parcel_access_layers", spy)
    # Rebind the memoized wrapper onto the (now-patched) memory. `_access_impl`
    # resolves `parcel_access_layers` via a module-global lookup at call time,
    # so it picks up `spy` regardless of when this rebind happens relative to
    # the patch above -- but we still need *a* rebind post-patch-memory so the
    # wrapper is backed by tmp_path's Memory rather than the real ~/.cache one.
    monkeypatch.setattr(
        cache, "_access_impl_cached", cache.cached(cache._access_impl, ignore=["block", "roads"])
    )

    block = _grid_block("deadbeef")
    before1 = cache.cached_access_layers(block, None, "__before__")
    before2 = cache.cached_access_layers(block, None, "__before__")  # HIT
    after = cache.cached_access_layers(block, block.streets, "peel")  # distinct key -> MISS
    assert calls["n"] == 2  # before computed once, after computed once
    assert before1.equals(before2)
    assert isinstance(after, pd.Series)
    # before and after must NOT collapse onto the same cache key: the
    # recompute count (2, not 3) proves the "before" hit was a real cache hit
    # and the "after" call was a real (separately-keyed) miss.


def test_cached_access_bypasses_when_hash_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import joblib

    monkeypatch.setattr(cache, "memory", joblib.Memory(location=str(tmp_path), verbose=0))
    block = _grid_block(cache.SOURCE_HASH_UNSET)  # ""
    out = cache.cached_access_layers(block, None, "__before__")
    assert isinstance(out, pd.Series)
    # Bypass path: an unset source hash never writes to the joblib store.
    assert not any(tmp_path.glob("**/*.pkl"))


def test_cached_propose_hits_and_bypasses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import joblib

    monkeypatch.setattr(cache, "memory", joblib.Memory(location=str(tmp_path), verbose=0))
    monkeypatch.setattr(cache, "_propose_impl_cached",
                        cache.cached(cache._propose_impl, ignore=["method", "block"]))

    from reblock.methods.topology import TopologyMethod
    m = TopologyMethod(alpha=2.0, seed=0)
    block = _grid_block("cafe1234")
    p1 = cache.cached_propose(m, block)
    p2 = cache.cached_propose(m, block)          # HIT
    assert p1.proposal_id == p2.proposal_id == "topology_a2.0_s0"
    # bypass path when hash unset writes nothing
    monkeypatch.setattr(cache, "memory", joblib.Memory(location=str(tmp_path / "b"), verbose=0))
    monkeypatch.setattr(cache, "_propose_impl_cached",
                        cache.cached(cache._propose_impl, ignore=["method", "block"]))
    cache.cached_propose(m, _grid_block(cache.SOURCE_HASH_UNSET))
    assert not any((tmp_path / "b").glob("**/*.pkl"))
