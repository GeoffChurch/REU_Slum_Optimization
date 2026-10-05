"""derive_graph: one memoization primitive for the content-addressed dataflow.

`derive(fn, *inputs)` computes `fn(*inputs)` with L1 (in-process) + L2 (joblib
disk) caching, keyed on a `DerivationKey`: the derivation's name, a content hash of the code it
can reach (reblock's and topology's), the versions of everything else its results depend on
(`env_version`), and its inputs' identities -- heavy inputs are never hashed (passed via joblib
`ignore=`). An input whose identity is None bypasses both layers. So any edit to that code, and
any upgrade of a library or native library under it, is a clean miss. See
docs/superpowers/specs/2026-07-08-content-addressed-dataflow-redesign.md.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import os
import platform
from collections.abc import Callable, Hashable
from functools import cache
from pathlib import Path
from typing import Protocol, TypeVar, cast, runtime_checkable

import joblib
import pyogrio
import pyproj
import shapely

T = TypeVar("T")

_CACHE_DIR = Path(os.environ.get(
    "REBLOCK_CACHE_DIR", str(Path.home() / ".cache" / "reblock" / "derivations")))
memory = joblib.Memory(location=str(_CACHE_DIR), verbose=0)

_L1: dict[DerivationKey, object] = {}

def source_hash(*paths: Path) -> str:
    """sha256 over the sorted paths' names + bytes. Stable, content-sensitive,
    order-independent. Used for a Source's data files (inside `reader_hash`) and for the
    derivation-module code hash below."""
    h = hashlib.sha256()
    for p in sorted(paths, key=str):
        h.update(str(Path(p).name).encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()


# A derivation's code version is its own IMPORT CLOSURE, walked from source -- never a list of
# modules, and never one hash over all of them.
#
# The list came first and went stale silently: it named the methods that existed when it was
# written, while `derivations.propose` caches ANY method, so every later one was invisible to the
# key. A `segment_displacement` fix in `resistance_lp` then changed its output on a direct call and
# changed NOTHING through the cache -- a full examples regeneration wrote 0 new entries and
# republished pre-fix results. The retreat was to hash EVERY derivation module together, which
# cannot go stale but invalidates everything: measured on the `depth` region, `cycle_native`
# (74 min) and the arterial (64 min) are 93% of that variant's proposal time, and a change to the
# scoring half of `budget.py` cannot affect `cycle_native` -- which takes only `buildings.radii` and
# `displacement` from it -- yet invalidated it anyway.
#
# A walked closure is neither: automatic, so there is no list to maintain, and per-derivation, so
# an edit reaches what it can actually affect. It over-approximates (an imported-but-unused module
# still invalidates), which is the safe direction. `tests/test_code_closure.py` holds the guard
# that no configurable strategy falls outside every key.


def _source_root(package: str) -> Path:
    """Where an installed package's source lives, found from its import spec: locating a package
    does not run it, so nothing executes while a cache key is built."""
    spec = importlib.util.find_spec(package)
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError(f"{package} is not an installed package")
    return Path(list(spec.submodule_search_locations)[0]).resolve()


# The packages whose SOURCE a derivation's code version covers: reblock, and topology -- the
# vendored package (ext/topology, a git submodule) that reblock's methods.topology,
# derive.parcel_graph and eval.kcomplexity call into, so an edit to it can change their results.
# Every other library a derivation uses is covered by its installed version (`env_version`).
_SOURCE_ROOTS: dict[str, Path] = {"reblock": Path(__file__).resolve().parent,
                                  "topology": _source_root("topology")}


def _module_file(name: str) -> Path | None:
    """`reblock.a.b` or `topology.a.b` -> its source file, or None if it is not one of ours.

    Tries the module then the package, so `reblock.methods.arterial` resolves to that
    package's `__init__.py`. Resolution is by PATH, never by import: importing to find a
    file would run module-level code, and this is called while building a cache key.
    """
    top, *rel = name.split(".")
    root = _SOURCE_ROOTS.get(top)    # absent: a library, which the key covers by version
    if root is None:
        return None
    for cand in (root.joinpath(*rel).with_suffix(".py"), root.joinpath(*rel, "__init__.py")):
        if cand.is_file():
            return cand
    return None


def _imports_of(path: Path, module: str) -> set[str]:
    """Every module of ours (`reblock.*`, `topology.*`) that `path` imports, at ANY depth in its
    AST.

    Nested and function-local imports count: `derivations._screen_selection_impl` imports
    `screen.dense_compact` inside the function body to dodge a cycle, and that module holds the
    selection logic the screen's key must cover. A module-level-only scan misses it -- which is
    precisely what `ScreenSelectionInput`'s docstring meant by relying on a global module list.

    `from reblock.methods import arterial` names a SUBMODULE, not an attribute, so both the
    package and `package.name` are offered as candidates and `_module_file` keeps whichever
    resolves.
    """
    out: set[str] = set()
    tree = ast.parse(path.read_text(), filename=str(path))
    own = _module_file(module)
    pkg = module.rsplit(".", 1)[0] if own is not None and own.name != "__init__.py" else module
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:                       # relative: climb `level-1` from the package
                parts = pkg.split(".")
                parts = parts[:len(parts) - (node.level - 1)] if node.level > 1 else parts
                base = ".".join([*parts, base]) if base else ".".join(parts)
            out.add(base)
            out.update(f"{base}.{a.name}" for a in node.names)
    return {n for n in out if n.split(".")[0] in _SOURCE_ROOTS}


@cache
def _closure_paths(module: str) -> frozenset[Path]:
    """Every source file of ours (reblock's and topology's) reachable from `module` by imports,
    transitively.

    This is the set a derivation's code hash is taken over, and therefore the exact set whose
    edits invalidate it. Static, so it OVER-approximates -- the safe direction: a module that
    is imported but unused still invalidates, where a module that is used but unseen would not.
    """
    start = _module_file(module)
    if start is None:
        return frozenset()
    seen: dict[str, Path] = {module: start}
    queue = [module]
    while queue:
        name = queue.pop()
        for dep in _imports_of(seen[name], name):
            if dep in seen:
                continue
            f = _module_file(dep)
            if f is not None:
                seen[dep] = f
                queue.append(dep)
    return frozenset(seen.values())


@cache
def closure_hash(module: str) -> str:
    """Content hash of `_closure_paths(module)` -- a derivation's own code version."""
    return source_hash(*sorted(_closure_paths(module)))


def reader_hash(reader: str, *paths: Path) -> str:
    """A Source's `Block.source_content_hash`: its data files' bytes AND the import closure of
    `reader`, the module that turns them into Blocks.

    The files alone name the data, not what was made of it. A reader edit -- how parcels group
    into components, which rings count as streets -- changes every Block under an unchanged file
    hash, and a derivation over a Block is keyed on the Block's identity plus its OWN closure,
    which need not import the reader. The shapefile reader was in no key's closure at all.
    """
    return source_hash(*paths, *_closure_paths(reader))


# The installed libraries a derivation's results can depend on, keyed by version. Every package in
# pyproject.toml's `runtime` group is here or exempt in tests/test_derive_graph.py, which says why
# and holds the two lists against the group, so a new runtime dependency has to be classified.
KEYED_PACKAGES: tuple[str, ...] = (
    "geopandas", "networkx", "numba", "numpy", "pandas", "pyarrow", "pyogrio", "pyproj", "pyshp",
    "scipy", "shapely")


@cache
def _package_versions() -> tuple[tuple[str, str], ...]:
    """(distribution, installed version) for each of KEYED_PACKAGES. Cached: what is installed
    does not change under a running process."""
    return tuple((name, importlib.metadata.version(name)) for name in KEYED_PACKAGES)


@dataclasses.dataclass(frozen=True)
class EnvVersions:
    """Everything outside the code a derivation runs that its results depend on, by version. (That
    code, reblock's and topology's, is the key's `code`.)"""
    python: str
    geos: str    # every geometry operation (shapely's C library)
    proj: str    # every reprojection (pyproj's)
    gdal: str    # every file read through pyogrio: the shapefile Block reader, footprints,
                 # informal structures, desire-line snapshots, OSM footpaths
    packages: tuple[tuple[str, str], ...]    # (distribution, version) for KEYED_PACKAGES


@dataclasses.dataclass(frozen=True)
class DerivationKey:
    """What a derivation's result is cached under.

    Built in `derive` and only ever hashed -- as the L1 dict's key, and by joblib, which pickles
    it -- so it is a frozen dataclass: fields by name only, no position to index or unpack, and
    never equal to a plain tuple of the same values."""
    fn: str                         # the derivation, as module.qualname
    code: str                       # `_code_version(fn, inputs)`: the code it actually runs
    env: EnvVersions                # `env_version()`
    inputs: tuple[Hashable, ...]    # each input's `identity`, in order


def env_version() -> EnvVersions:
    """The versions a derivation's results depend on, beyond its own code.

    Read live so a test can monkeypatch it and force a miss. The CODE version is not here: it is
    per-derivation (`_code_version`), because one global hash over every derivation module meant
    an edit to any of them invalidated all of them.
    """
    return EnvVersions(python=platform.python_version(),
                       geos=".".join(str(x) for x in shapely.geos_version),
                       proj=pyproj.proj_version_str, gdal=pyogrio.__gdal_version_string__,
                       packages=_package_versions())


@cache
def _code_version_of(modules: frozenset[str]) -> str:
    """Content hash over the UNION of `modules`' import closures. Cached: the same module set
    recurs for every block, and hashing ~30 files per `derive` call would be real I/O."""
    paths: set[Path] = set()
    for m in modules:
        paths |= _closure_paths(m)
    return source_hash(*sorted(paths))


def _code_version(fn: Callable[..., object], inputs: tuple[object, ...]) -> str:
    """The code a derivation actually runs, as a hash.

    The UNION of two things, because neither covers the other:

    * `fn`'s own import closure -- the derivation body and everything it reaches;
    * every INPUT's type module closure -- because dispatch is invisible to a static walk of
      `fn`. `derivations._propose_impl` never imports `methods.cycle_native`; the method
      arrives as an argument, so only `type(method).__module__` reveals it.

    A strategy held as a FIELD of an input (the screen's `BlockMetric`) is reached by neither,
    and is covered instead by its own `identity` carrying a code hash -- which rides the input
    identity already in the key. Union semantics throughout: the two mechanisms cannot drift,
    because disagreement is not an error state, and both over-approximate in the safe direction.
    """
    return _code_version_of(frozenset({fn.__module__,
                                       *(type(i).__module__ for i in inputs)}))


def clear_l1() -> None:
    """Drop the in-process L1 cache (call between independent runs/tests)."""
    _L1.clear()


def _l2_derive(key: DerivationKey, fn: Callable[..., object],
               inputs: tuple[object, ...]) -> object:
    return fn(*inputs)


# joblib keeps a cached function's results in a directory named after it, and WIPES that directory
# when the function's source text changes. So this function is never edited: a new key format gets
# a new function, and with it a directory of its own, which no checkout on another format wipes.
_l2 = memory.cache(_l2_derive, ignore=["fn", "inputs"])


@runtime_checkable
class Identified(Protocol):
    """An input `derive` can key on. Every input declares its `identity`; None is the explicit
    answer of one that has no content address -- a synthetic block, a live data source, an ad-hoc
    routing graph -- and makes the derivation uncacheable, so two such inputs can never share a
    key."""

    @property
    def identity(self) -> Hashable | None: ...


class _Uncacheable:
    """A nested input answered `identity is None`: the whole configuration has no address."""


_UNCACHEABLE = _Uncacheable()


def config_identity(config: object, *, exempt: frozenset[str] = frozenset()) -> Hashable | None:
    """The cache key of a configured object (a Method, and anything it is configured with): its
    class, then EVERY dataclass field by name except those in `exempt`.

    Derived rather than listed: a hand-written identity is a second copy of the field list, and the
    copies drift -- most methods' omitted `road_width_m`, two their `PermeabilityParams`, so a sweep
    over those returned another setting's cached roads, silently. A field is in the key unless its
    class NAMES it in `exempt`, which is for fields that cannot change the output (a worker count).

    A field holding something `Identified` contributes that `identity`, and if that is None the
    whole configuration is uncacheable (returns None) -- a live data source, an ad-hoc graph. A
    nested plain dataclass (`PermeabilityParams`) contributes its own fields. Anything else that is
    not a plain value raises: a key must never silently omit what it cannot describe."""
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        raise TypeError(f"config_identity needs a dataclass instance, got {type(config).__name__}")
    fields = dataclasses.fields(config)
    stale = exempt - {f.name for f in fields}
    if stale:
        raise ValueError(
            f"{type(config).__name__} exempts fields it does not have: {sorted(stale)}")
    parts: list[tuple[str, Hashable]] = []
    for f in fields:
        if f.name in exempt:
            continue
        # By name over the declared schema -- the one place dynamic access is correct -- and with
        # no default, so a field that is not there raises.
        value = _value_identity(getattr(config, f.name), f"{type(config).__name__}.{f.name}")
        if value is _UNCACHEABLE:
            return None
        parts.append((f.name, value))
    return (type(config).__qualname__, tuple(parts))


def _value_identity(value: object, where: str) -> Hashable:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, tuple | list | frozenset):
        items = [_value_identity(v, where) for v in value]
        if any(i is _UNCACHEABLE for i in items):
            return _UNCACHEABLE
        return tuple(sorted(items, key=repr)) if isinstance(value, frozenset) else tuple(items)
    if isinstance(value, Identified):
        ident = value.identity
        return _UNCACHEABLE if ident is None else ident
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        nested = config_identity(value)
        return _UNCACHEABLE if nested is None else nested
    raise TypeError(f"{where} holds a {type(value).__name__}, which has no identity: give it one, "
                    f"or exempt the field if it cannot change the output")


def derive(fn: Callable[..., T], *inputs: Identified) -> T:
    """Memoized compute of `fn(*inputs)`, keyed on (fn.identity, input identities).
    Bypasses (computes directly) if any input's `identity` is None."""
    ids: list[Hashable] = []
    for i in inputs:
        ident = i.identity
        if ident is None:
            return fn(*inputs)          # bypass: uncacheable input
        ids.append(ident)
    key = DerivationKey(fn=f"{fn.__module__}.{fn.__qualname__}", code=_code_version(fn, inputs),
                        env=env_version(), inputs=tuple(ids))
    if key in _L1:
        return cast(T, _L1[key])
    out = cast(T, _l2(key, fn, inputs))  # joblib keys on `key`, ignores fn+inputs
    _L1[key] = out
    return out
