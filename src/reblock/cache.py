"""L2 per-block derivation cache: a content-addressed joblib.Memory.

Cached derivations are pure functions of a block (and optionally roads/params).
We never hash the heavy geometry (slow + GEOS-fragile); instead each cached
wrapper passes the heavy objects via joblib `ignore=` and keys ONLY on
lightweight strings: (block_id, source_content_hash, geos, proj, code_version,
roads_key | method_repr). An empty source_content_hash bypasses the cache
(synthetic/test blocks). See
docs/superpowers/specs/2026-07-07-atomic-flow-and-sweep-architecture-design.md §6.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd
import pyproj
import shapely
from geopandas import GeoDataFrame

from reblock.contracts import Block
from reblock.derive.access import parcel_access_layers
from reblock.derive.geometric_access import geometric_access_distances

SOURCE_HASH_UNSET = ""

_CACHE_DIR = Path.home() / ".cache" / "reblock" / "derivations"
memory = joblib.Memory(location=str(_CACHE_DIR), verbose=0)

# Modules whose source defines the cached derivations; hashed into every key so
# an edit to derivation logic auto-invalidates (joblib alone only hashes the
# thin wrapper, not its callees). Coarse but safe: any edit to these files
# invalidates all cached derivations.
_DERIVATION_MODULE_FILES = (
    Path(__file__).with_name("cache.py"),
    Path(__file__).parent / "derive" / "access.py",
    Path(__file__).parent / "derive" / "geometric_access.py",
    Path(__file__).parent / "data" / "kblock.py",
    Path(__file__).parent / "methods" / "topology.py",
    Path(__file__).parent / "methods" / "peel.py",
)


def source_hash(*paths: Path) -> str:
    """sha256 over the sorted paths' names + bytes. Stable, content-sensitive,
    order-independent. Used both for a Source's data files and for code_version."""
    h = hashlib.sha256()
    for p in sorted(paths, key=str):
        h.update(str(Path(p).name).encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()


_CODE_VERSION = source_hash(*_DERIVATION_MODULE_FILES)


def key_parts() -> tuple[str, str, str]:
    """(geos_version, proj_version, code_version) — read live so tests can
    monkeypatch and force a clean miss."""
    geos = ".".join(str(x) for x in shapely.geos_version)
    return geos, pyproj.proj_version_str, _CODE_VERSION


def cached(impl: Callable[..., Any], ignore: list[str]) -> Callable[..., Any]:
    """Wrap `impl` with the module joblib.Memory, ignoring the named heavy args
    when computing the cache key."""
    # joblib ships no py.typed marker (see the mypy override in pyproject.toml),
    # so memory.cache(...) is untyped Any at the type-checker's eyes; cast back
    # to the declared signature rather than let strict mypy flag no-any-return.
    return cast("Callable[..., Any]", memory.cache(impl, ignore=ignore))


def _access_impl(block: Block, roads: GeoDataFrame | None, *, block_id: str,
                  src_hash: str, geos: str, proj: str, code: str, roads_key: str) -> pd.Series:
    return parcel_access_layers(block, roads)


def _geometric_impl(block: Block, roads: GeoDataFrame | None, *, block_id: str,
                     src_hash: str, geos: str, proj: str, code: str,
                     roads_key: str) -> pd.Series:
    return geometric_access_distances(block, roads)


_access_impl_cached = cached(_access_impl, ignore=["block", "roads"])
_geometric_impl_cached = cached(_geometric_impl, ignore=["block", "roads"])


def cached_access_layers(block: Block, roads: GeoDataFrame | None, roads_key: str) -> pd.Series:
    """`parcel_access_layers`, memoized on (block_id, source_content_hash, geos,
    proj, code_version, roads_key). `roads_key` must be `"__before__"` when
    `roads is None` and the proposal's id (or another value distinct from
    `"__before__"`) otherwise, so before/after never collapse onto one key.
    Blocks with an unset `source_content_hash` bypass the cache entirely."""
    if block.source_content_hash == SOURCE_HASH_UNSET:
        return parcel_access_layers(block, roads)
    geos, proj, code = key_parts()
    # _access_impl_cached is Callable[..., Any] (see `cached`'s joblib-typing note);
    # cast back to the declared return type rather than let strict mypy flag it.
    return cast("pd.Series", _access_impl_cached(
        block, roads, block_id=block.block_id, src_hash=block.source_content_hash,
        geos=geos, proj=proj, code=code, roads_key=roads_key))


def cached_geometric(block: Block, roads: GeoDataFrame | None, roads_key: str) -> pd.Series:
    """`geometric_access_distances`, memoized the same way as `cached_access_layers`
    (see its docstring for the `roads_key` convention and the bypass path)."""
    if block.source_content_hash == SOURCE_HASH_UNSET:
        return geometric_access_distances(block, roads)
    geos, proj, code = key_parts()
    return cast("pd.Series", _geometric_impl_cached(
        block, roads, block_id=block.block_id, src_hash=block.source_content_hash,
        geos=geos, proj=proj, code=code, roads_key=roads_key))
