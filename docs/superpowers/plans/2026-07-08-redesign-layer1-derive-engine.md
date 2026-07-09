# Redesign Layer 1 — the `derive` engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `reblock.derive_graph` — one uniform memoization primitive `derive(fn, *inputs)` keyed on `(fn.identity, input identities)`, with L1 (in-process) + L2 (joblib disk), replacing F2's four hand-rolled `cached_*` wrappers.

**Architecture:** A datum is `Identified` if it has a hashable `.identity`. `derive(fn, *inputs)` content-addresses on `(fn's identity, each input's identity)` — heavy inputs never hashed. A missing identity → bypass. `fn.identity` = `(qualified name, version)` where `version` = a content hash of the derivation modules + GEOS + PROJ (D2: centralized + complete). Self-contained (no import of the F2 `cache` module, which is deleted in the final layer).

**Tech Stack:** Python 3.12, joblib, shapely (geos_version), pyproj (proj_version_str), pixi, pytest, `mypy --strict`, ruff.

## Global Constraints

- `pixi run check` stays green — `ruff check` + `ruff format --check` + `mypy --strict src tests scripts/crossblock_probe.py` + pytest. Suite is currently 119 tests.
- **Additive only** — Layer 1 introduces `reblock.derive_graph` + its tests; it does NOT touch the existing `reblock.cache`, `run`, Source, or contracts (those change in later layers; `cache` is deleted in the final layer). Nothing existing breaks.
- **`derive` forms cache keys in exactly one place** — `(fn.identity, tuple(i.identity for i in inputs))`. Heavy inputs are passed through joblib `ignore=`, never hashed.
- **Bypass on missing identity** — if any input lacks a usable `.identity`, compute directly (no L1/L2 store touch). This is how synthetic/test data (and later, unhashable inputs) stay uncached.
- **`version` (in `fn.identity`) read LIVE** — so a test can monkeypatch it and force a clean miss. `version` = `source_hash(<derivation module files>) + geos_version + proj_version` (D2).
- **Tests never touch the real `~/.cache/reblock`** — point the joblib store at a tmp dir (per-test monkeypatch in Task 1; a session conftest fixture in Task 2) and honor `REBLOCK_CACHE_DIR`.
- Commit trailers on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
  ```

---

### Task 1: `reblock.derive_graph` — the `derive` primitive

**Files:**
- Create: `src/reblock/derive_graph.py`
- Test: `tests/test_derive_graph.py`

**Interfaces:**
- Produces:
  - `Identified` (Protocol): a `.identity` property returning a hashable.
  - `version() -> tuple[str, str, str]` — `(code_hash, geos, proj)`, read live.
  - `derive(fn: Callable[..., T], *inputs: object) -> T` — memoized compute; bypass if any input has no `.identity`.
  - `clear_l1() -> None`; `memory` (the joblib.Memory); an internal `_L1` dict.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_derive_graph.py`:

```python
from dataclasses import dataclass

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
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(dg, "memory", joblib.Memory(location=str(tmp_path), verbose=0))
    monkeypatch.setattr(dg, "_l2", dg.memory.cache(dg._l2_impl, ignore=["fn", "inputs"]))
    dg.clear_l1()
    yield
    dg.clear_l1()


def _count(box):
    def f(x):
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


def test_missing_identity_bypasses_cache(tmp_path) -> None:
    box = {"n": 0}

    def fn(x):
        box["n"] += 1
        return 42
    dg.derive(fn, _NoIdentity())
    dg.derive(fn, _NoIdentity())     # no identity -> never cached
    assert box["n"] == 2
    assert not dg._L1                 # nothing stored in L1
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_derive_graph.py -v`
Expected: FAIL — `No module named 'reblock.derive_graph'`.

- [ ] **Step 3: Implement `src/reblock/derive_graph.py`**

```python
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
from pathlib import Path
from typing import Any, Callable, Hashable, Protocol, TypeVar, cast, runtime_checkable

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
)


@runtime_checkable
class Identified(Protocol):
    @property
    def identity(self) -> Hashable: ...


def _source_hash(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=str):
        h.update(str(Path(p).name).encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()


_CODE_HASH = _source_hash(*_DERIVATION_MODULES)


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
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run pytest tests/test_derive_graph.py -v`
Expected: PASS (4 tests). Then `pixi run check` — ruff + `mypy --strict` clean (joblib is already in the `[[tool.mypy.overrides]]` from F2; the `cast`s cover joblib's untyped returns). 123 tests green.

- [ ] **Step 5: Commit**

```bash
git add src/reblock/derive_graph.py tests/test_derive_graph.py
git commit -m "$(cat <<'EOF'
feat: reblock.derive_graph -- the derive() memoization primitive (redesign L1)

derive(fn, *inputs) content-addresses on (fn.identity, input identities) with
L1 in-process + L2 joblib disk; heavy inputs ignored, missing identity bypasses.
fn.identity folds in a complete derivation-module code hash + GEOS + PROJ, so a
logic edit is a clean miss -- one place, replacing F2's four cached_* wrappers.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

### Task 2: session test-isolation + version invalidation

**Files:**
- Modify: `tests/conftest.py` (isolate `derive_graph`'s store alongside `cache`'s)
- Test: `tests/test_derive_graph.py` (version-bump forces a miss)

**Interfaces:**
- Consumes: `derive_graph.memory`, `derive_graph._l2`, `derive_graph._l2_impl`, `derive_graph.clear_l1`, `derive_graph.version`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_derive_graph.py`:

```python
def test_version_bump_forces_a_miss(monkeypatch) -> None:
    box = {"n": 0}
    fn = _count(box)
    a = _Datum("a")
    dg.derive(fn, a)
    # simulate a derivation-logic / lib change: version() returns a new tag
    monkeypatch.setattr(dg, "version", lambda: ("CHANGED", "g", "p"))
    dg.clear_l1()
    dg.derive(fn, a)                 # new version -> new key -> recompute
    assert box["n"] == 2
```

- [ ] **Step 2: Run to verify it fails or passes as-is**

Run: `pixi run pytest tests/test_derive_graph.py::test_version_bump_forces_a_miss -v`
Expected: PASS already IF `version()` is read live inside `_fn_identity` (Task 1 did this). If it FAILS (version captured at import), fix `_fn_identity` to call `version()` live. Either way, this test locks the behavior in.

- [ ] **Step 3: Isolate `derive_graph`'s store in conftest**

In `tests/conftest.py`, the existing session fixture already repoints `cache.memory` + sets `REBLOCK_CACHE_DIR`. Extend it (or add a sibling session-autouse fixture) to also isolate `derive_graph`: after setting `REBLOCK_CACHE_DIR`, repoint `derive_graph.memory` to the same tmp dir and rebind `derive_graph._l2`:

```python
import reblock.derive_graph as _dg
# inside the session fixture, after the env + cache.memory setup:
mp.setattr(_dg, "memory", joblib.Memory(location=str(tmp_dir), verbose=0))
mp.setattr(_dg, "_l2", _dg.memory.cache(_dg._l2_impl, ignore=["fn", "inputs"]))
```

And in the existing function-autouse `_clear_l1` fixture, also clear derive_graph's L1:

```python
@pytest.fixture(autouse=True)
def _clear_l1() -> Iterator[None]:
    cache.clear_l1()
    _dg.clear_l1()
    yield
    cache.clear_l1()
    _dg.clear_l1()
```

(If Layer 0's F3 `_clear_l1` fixture isn't present yet on this branch — it was F3, which was dropped — add the `_clear_l1` function-autouse fixture now for `derive_graph` only.)

- [ ] **Step 4: Run + full check**

Run: `pixi run pytest tests/test_derive_graph.py -v` then `pixi run check`
Expected: PASS; and confirm the real `~/.cache/reblock/derivations` file count is unchanged after a full suite run (the isolation holds) — `find ~/.cache/reblock/derivations -type f | wc -l` before/after `pixi run check`.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_derive_graph.py
git commit -m "$(cat <<'EOF'
test: isolate derive_graph store + lock version-bump invalidation (redesign L1)

conftest repoints derive_graph.memory to a tmp dir and clears its L1 per test
(suite stays hermetic); a version-bump test locks that a derivation-logic/lib
change is a clean miss.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

## Self-Review

**Spec coverage (Layer 1 of the migration):** the `derive` primitive (spec Layer 3 / "one memoization primitive") with composed-identity keying (D1), complete centralized `version` (D2), L1+L2, bypass — Tasks 1–2. Additive; touches nothing existing. ✓

**Placeholder scan:** every code step is complete; the Task-2 note about the `_clear_l1` fixture possibly not existing yet is a real branch-state contingency (F3 was dropped), handled explicitly. No TBD.

**Type consistency:** `derive`, `version`, `clear_l1`, `_l2`/`_l2_impl`, `_L1`, `Identified` are used identically between the module (Task 1), the tests (Tasks 1–2), and the conftest (Task 2). `_l2` is rebound with the exact `ignore=["fn", "inputs"]` everywhere it's constructed (module, conftest, test fixture).
