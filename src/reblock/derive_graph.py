"""derive_graph: one memoization primitive for the content-addressed dataflow.

`derive(fn, *inputs)` computes `fn(*inputs)` with L1 (in-process) + L2 (joblib
disk) caching, keyed on `(fn.identity, tuple(input identities))` -- heavy inputs
are never hashed (passed via joblib `ignore=`). A missing input identity bypasses
both layers. `fn.identity = (qualified-name, version)` where `version` is a
content hash of the derivation modules + GEOS + PROJ, so any derivation-logic
edit (or native-lib upgrade) is a clean miss. See
docs/superpowers/specs/2026-07-08-content-addressed-dataflow-redesign.md.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Hashable
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

import joblib
import pyproj
import shapely

T = TypeVar("T")

_CACHE_DIR = Path(os.environ.get(
    "REBLOCK_CACHE_DIR", str(Path.home() / ".cache" / "reblock" / "derivations")))
memory = joblib.Memory(location=str(_CACHE_DIR), verbose=0)

_L1: dict[tuple[Any, ...], Any] = {}

# Derivation modules whose source defines cached logic; hashed into `version`
# so an edit to any of them is a clean miss (D2: centralized + complete). Grows
# as later layers add derivation modules.
_DERIVATION_MODULES: tuple[Path, ...] = (
    Path(__file__).with_name("derive_graph.py"),
    Path(__file__).parent / "derive" / "access.py",
    Path(__file__).parent / "derive" / "geometric_access.py",
    Path(__file__).parent / "derive" / "adjacency.py",
    Path(__file__).parent / "derive" / "parcel_graph.py",
    Path(__file__).parent / "methods" / "topology.py",
    Path(__file__).parent / "methods" / "peel.py",
    Path(__file__).parent / "methods" / "dijkstra.py",
    Path(__file__).with_name("derivations.py"),      # derive()-wrapper bodies
    Path(__file__).parent / "data" / "kblock.py",    # _voronoi_parcels (the voronoi derivation)
)


@runtime_checkable
class Identified(Protocol):
    @property
    def identity(self) -> Hashable: ...


def source_hash(*paths: Path) -> str:
    """sha256 over the sorted paths' names + bytes. Stable, content-sensitive,
    order-independent. Used for a Source's data files (Block.source_content_hash)
    and for the derivation-module code hash below."""
    h = hashlib.sha256()
    for p in sorted(paths, key=str):
        h.update(str(Path(p).name).encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()


_CODE_HASH = source_hash(*_DERIVATION_MODULES)


def version() -> tuple[str, str, str]:
    """(code_hash, geos, proj) -- read live so tests can monkeypatch and miss."""
    geos = ".".join(str(x) for x in shapely.geos_version)
    return _CODE_HASH, geos, pyproj.proj_version_str


def clear_l1() -> None:
    """Drop the in-process L1 cache (call between independent runs/tests)."""
    _L1.clear()


def _fn_identity(fn: Callable[..., Any]) -> tuple[str, tuple[str, str, str]]:
    return (f"{fn.__module__}.{fn.__qualname__}", version())


def _l2_impl(key: tuple[Any, ...], fn: Callable[..., Any], inputs: tuple[Any, ...]) -> Any:
    return fn(*inputs)


_l2 = memory.cache(_l2_impl, ignore=["fn", "inputs"])


def derive(fn: Callable[..., T], *inputs: object) -> T:
    """Memoized compute of `fn(*inputs)`, keyed on (fn.identity, input identities).
    Bypasses (computes directly) if any input lacks a usable `.identity`."""
    ids: list[Hashable] = []
    for i in inputs:
        ident = getattr(i, "identity", None)
        if ident is None:
            return fn(*inputs)          # bypass: uncacheable input
        ids.append(ident)
    key = (_fn_identity(fn), tuple(ids))
    if key in _L1:
        return cast(T, _L1[key])
    out = cast(T, _l2(key, fn, inputs))  # joblib keys on `key`, ignores fn+inputs
    _L1[key] = out
    return out
