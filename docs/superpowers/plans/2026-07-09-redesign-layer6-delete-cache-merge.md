# Redesign Layer 6 — delete the orphaned cache + consolidate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the now-orphaned `reblock.cache` (the F2 four-`cached_*`-wrapper module, dead since L3 rewired everything through `derive()`), promoting its one still-used symbol — `source_hash` — into `reblock.derive_graph`, so the merged redesign carries **no dual caching system** (the "no warts / no dual path" directive on the shipped state).

**Architecture:** `reblock.derive_graph` already defines a byte-identical private `_source_hash`; make it public (`source_hash`), repoint the two Source importers (`kblock.py`, `shapefile.py`) at it, migrate the `source_hash` unit tests onto it, and `git rm` `src/reblock/cache.py` + `tests/test_cache.py`. Everything else in `cache.py` (the four `cached_*` wrappers, `key_parts`, `cached`, `memory`, `_CODE_VERSION`, `SOURCE_HASH_UNSET`) has no remaining `src` importer and dies with the file.

**Tech Stack:** Python 3.12, joblib, geopandas/shapely, pixi, pytest, `mypy --strict`, ruff.

## Global Constraints

- `pixi run check` stays green — `ruff check` + `mypy --strict src tests scripts/crossblock_probe.py` + pytest. Suite is currently **144 tests**.
- **Byte-identical `source_hash`** — `derive_graph._source_hash` and `cache.source_hash` are verified character-for-character identical (sha256 over sorted paths' `name` + `read_bytes()`). Promoting one and repointing the Sources MUST leave every `Block.source_content_hash` (hence `Block.identity`, hence every `derive()` cache key) unchanged. The pinned kblock/Phule values in `tests/test_run.py` are the guard — they must still pass unchanged.
- **No dual path / no compat shim** (owner directive): `reblock.cache` is **deleted**, not left orphaned or aliased. No `source_hash` shim left behind in a `cache` module; no re-export. After this task, `grep -rn "reblock.cache" src tests` returns nothing.
- **Scope:** this layer only removes the orphaned module + relocates `source_hash`. The standalone screen app and `RunConfig` were already deleted in L5. No behavior change.
- Commit trailers on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
  ```

---

### Task 1: promote `source_hash` into `derive_graph`; repoint Sources; delete `reblock.cache` + its test

**Files:**
- Modify: `src/reblock/derive_graph.py` (rename `_source_hash` → public `source_hash`)
- Modify: `src/reblock/data/kblock.py` (import from `derive_graph`)
- Modify: `src/reblock/data/shapefile.py` (import from `derive_graph`)
- Modify: `tests/test_derive_graph.py` (add the migrated `source_hash` tests)
- Modify: `tests/conftest.py` (drop the stale `cache.py` / `test_cache.py` mentions in its module docstring)
- Delete: `src/reblock/cache.py`, `tests/test_cache.py`

**Interfaces:**
- Produces: `reblock.derive_graph.source_hash(*paths: Path) -> str` (was the private `_source_hash`; identical body).
- Consumes: nothing new.

- [ ] **Step 1: Migrate the `source_hash` tests onto `derive_graph`**

In `tests/test_derive_graph.py` (which already does `import reblock.derive_graph as dg`), add the two `source_hash` tests, retargeted from `cache.source_hash` to `dg.source_hash`. Append them after the existing tests:

```python
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
```

(`Path` is already imported in `tests/test_derive_graph.py`.)

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_derive_graph.py -k source_hash -v`
Expected: FAIL — `AttributeError: module 'reblock.derive_graph' has no attribute 'source_hash'` (it is still the private `_source_hash`).

- [ ] **Step 3: Make `source_hash` public in `derive_graph`**

In `src/reblock/derive_graph.py`, rename the private `_source_hash` to a public `source_hash` (add a docstring; keep the body identical), and update its one internal caller:

```python
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
```

(That `_CODE_HASH = source_hash(*_DERIVATION_MODULES)` line currently reads `_source_hash(...)` — update it. Confirm no other `_source_hash` reference remains: `grep -n "_source_hash" src/reblock/derive_graph.py` → none.)

- [ ] **Step 4: Repoint the two Source importers**

In `src/reblock/data/kblock.py`, change:
```python
from reblock.cache import source_hash
```
to:
```python
from reblock.derive_graph import source_hash
```

In `src/reblock/data/shapefile.py`, make the identical change. (Both call sites — `source_hash(self.blocks_path, self.buildings_path)` and `source_hash(self.path)` — are unchanged; only the import module changes.)

- [ ] **Step 5: Delete `reblock.cache` + its test; fix the stale conftest comment**

```bash
git rm src/reblock/cache.py tests/test_cache.py
```

In `tests/conftest.py`, the module docstring still names the deleted files. Update the two stale mentions:
- `"...it too binds to this tmp dir when it freshly imports cache.py/derive_graph.py."` → `"...when it freshly imports derive_graph.py."`
- `"Per-test/per-module monkeypatches (tests/test_cache.py, tests/test_derive_graph.py) still work..."` → `"Per-test monkeypatches (tests/test_derive_graph.py) still work..."`

(These are comment-only edits; do not change any fixture logic.)

- [ ] **Step 6: Run the full suite — no behavior change**

Run: `pixi run check`
Expected: PASS, 144 tests (test_cache.py's ~11 tests removed; 2 `source_hash` tests re-added to test_derive_graph → net drop, exact count reported by the implementer). Verify:
- `grep -rn "reblock.cache\|reblock\.cache\|from reblock import cache\|import reblock.cache" src tests` → **no matches** (the module is gone and unreferenced).
- The pinned values in `tests/test_run.py` still pass unchanged (byte-identical `source_hash` → identical `Block.identity` → identical cache keys and metrics).

- [ ] **Step 7: Commit**

```bash
git add -A -- src/reblock/derive_graph.py src/reblock/data/kblock.py src/reblock/data/shapefile.py tests/test_derive_graph.py tests/conftest.py src/reblock/cache.py tests/test_cache.py
git commit -m "$(cat <<'EOF'
refactor: promote source_hash into derive_graph; delete orphaned reblock.cache (redesign L6)

reblock.cache (F2's four cached_* wrappers) has been dead since L3 rewired every
derivation through derive(). Its one live symbol, source_hash, moves to
derive_graph (which already had a byte-identical private _source_hash, now public);
kblock + shapefile import it from there. Deletes src/reblock/cache.py +
tests/test_cache.py (source_hash tests migrated to test_derive_graph). No behavior
change -- source_hash is identical, so every Block.identity and derive() cache key
is unchanged. The redesign now carries a single caching system.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

(Do NOT `git add -A` the whole tree — there is an untracked, unrelated `.vscode/` dir; the explicit pathspec above adds only this task's files, including the two deletions.)

---

## Self-Review

**Spec coverage (L6 = migration step 6, "Delete the old + consolidate"):** `reblock.cache` deleted (Task 1); its one live symbol `source_hash` relocated to `derive_graph` and both importers repointed (Task 1); `test_cache.py` deleted with its `source_hash` coverage preserved on `derive_graph` (Task 1). The screen app + `RunConfig` were already deleted in L5. The merge to `main` is the finishing step (superpowers:finishing-a-development-branch), not a plan task.

**Placeholder scan:** every step has concrete code / exact commands. No TBD.

**Type consistency:** `source_hash(*paths: Path) -> str` keeps the exact signature it had as `_source_hash` / `cache.source_hash`; both call sites (`kblock`, `shapefile`) pass `Path` args and receive `str`, unchanged. `_CODE_HASH = source_hash(*_DERIVATION_MODULES)` keeps its type. `mypy --strict` sees a public name where a private one was — no signature change.

**No-behavior-change** is guarded by the byte-identical `source_hash` body (verified before planning) plus the `tests/test_run.py` pinned kblock/Phule assertions passing unchanged, and the post-delete grep proving no dangling `reblock.cache` reference survives.
