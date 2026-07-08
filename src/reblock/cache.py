"""L2 per-block derivation cache: a content-addressed joblib.Memory.

Cached derivations are pure functions of a block (and optionally roads/params).
We never hash the heavy geometry (slow + GEOS-fragile); instead each cached
wrapper passes the heavy objects via joblib `ignore=` and keys ONLY on
lightweight strings: (block_id, source_content_hash, geos, proj, code_version,
roads_key | method_repr). An empty source_content_hash bypasses the cache
(synthetic/test blocks). See docs/.../2026-07-07-atomic-flow-...-design.md §6.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import joblib
import pyproj
import shapely

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
