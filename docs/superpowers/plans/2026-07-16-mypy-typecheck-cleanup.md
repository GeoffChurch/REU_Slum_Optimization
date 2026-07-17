# mypy --strict Test Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get `pixi run typecheck` (`mypy --strict src tests scripts/crossblock_probe.py`) to report `Success: no issues found`, by fixing ~22 pre-existing test type-errors that a now-fixed mypy config bug (duplicate-module fatal error) was silently hiding. All 22 errors are benign strictness nits, not bugs — no test assertion or runtime behavior should change.

**Architecture:** No production code changes. Four independent, mechanical fixes: (1) a `pyproject.toml` mypy override for the deliberately-bare `scoring_fixtures` import, (2) deletion of now-unused `# type: ignore` comments in two topology-adjacent test files, (3) a one-line `str`→`CRS` fix in `tests/test_render.py`, (4) `assert ... is not None` narrowing for `GeoDataFrame | None` in `tests/methods/test_arterial_lazy.py`.

**Tech Stack:** Python, mypy --strict, pytest, ruff, pixi.

## Global Constraints

- **Every fix is the minimal correct one.** Do NOT relax more mypy error codes than the spec names, do NOT add blanket `# type: ignore`, do NOT weaken any test assertion or change test logic/behavior.
- **Do NOT touch the `from scoring_fixtures import ...` / `from scoring_fixtures import _block_1808` bare-import lines themselves** — they are a deliberate pattern (defended by comments already in the test files) that must keep resolving at runtime via pytest's sys.path. Fix the mypy error via a `pyproject.toml` override, never by rewriting the import.
- **Gates for every task:** `pixi run typecheck`, `pixi run pytest`, `pixi run ruff check` — all run from `/home/gchurchill/src/reblock`. `pixi run ruff check` forbids semicolons (E702), lines > 100 chars (E501), and `zip()` without `strict=` (B905).
- **Do not run the full `pixi run typecheck`/`pixi run pytest` gates as "done" until ALL 22 errors are gone** — but each task should independently verify its own slice doesn't regress (e.g. `mypy --strict <touched files>` for a quick check is fine mid-task; the final task confirms the whole-repo gates).
- **Branch:** `fix-mypy-typecheck` (already checked out). `pyproject.toml` already carries the `explicit_package_bases`/`mypy_path`/`tests.*` override config fix — do not touch that part of the file, only ADD a new override block for `scoring_fixtures`.
- If ANY of these "benign" errors turns out to guard a real bug (not just a strictness nit), STOP and flag it — do not paper over it with an ignore.

## File Structure

- `pyproject.toml` — add one `[[tool.mypy.overrides]]` block for `module = ["scoring_fixtures"]`.
- `tests/derive/test_parcel_graph.py`, `tests/methods/test_topology_method.py` — delete unused `# type: ignore[no-untyped-call]` comments.
- `tests/test_render.py` — change one local `crs = "EPSG:32734"` string to `crs = CRS.from_epsg(32734)` (module already imports `pyproj.CRS`).
- `tests/methods/test_arterial_lazy.py` — add `assert ... is not None` narrowing before 6 use-sites of `.roads`.

---

### Task 1: `scoring_fixtures` import — mypy override

**Files:**
- Modify: `pyproject.toml`

**Context:** `tests/test_scoring_equivalence.py:8`, `tests/methods/test_arterial.py:190`, `tests/methods/test_clearance.py:280`, `tests/methods/test_arterial_lazy.py:117` all do a bare `from scoring_fixtures import ...` (resolved at runtime via pytest's rootdir-relative sys.path, NOT as `tests.scoring_fixtures`.) mypy reports:
```
error: Cannot find implementation or library stub for module named "scoring_fixtures"  [import-not-found]
```
This is expected — `scoring_fixtures` isn't a package mypy can resolve from `mypy_path`/`explicit_package_bases` the way `tests.*` submodules are, and it doesn't need type-checking (it's the test's own fixture helper, effectively test infrastructure).

**Fix:** Add a new mypy override block to `pyproject.toml`, near the existing `[[tool.mypy.overrides]]` blocks (e.g. right after the `tests.*` block added for the duplicate-module fix):

```toml
[[tool.mypy.overrides]]
# tests/scoring_fixtures.py is imported bare (`from scoring_fixtures import ...`), not as
# `tests.scoring_fixtures` — a deliberate pattern so pytest's sys.path (not a package name)
# resolves it at runtime. mypy can't resolve the bare name from mypy_path/explicit_package_bases;
# this override silences that import-not-found without touching the import itself.
module = ["scoring_fixtures"]
ignore_missing_imports = true
```

Match the existing style/comment density of the surrounding `[tool.mypy]` section (read it first).

**Verify:**
```
pixi run mypy --strict tests/test_scoring_equivalence.py tests/methods/test_arterial.py tests/methods/test_clearance.py tests/methods/test_arterial_lazy.py 2>&1 | grep scoring_fixtures
```
should show zero `scoring_fixtures` import-not-found lines (other unrelated errors in `test_arterial_lazy.py` are Task 4's, not yours — ignore them here).

**Do NOT:**
- Change any `from scoring_fixtures import ...` line.
- Add `ignore_missing_imports` more broadly (e.g. to `tests.*` or a wildcard).

---

### Task 2: Remove unused `# type: ignore` comments

**Files:**
- Modify: `tests/derive/test_parcel_graph.py` (lines 86, 161, 168)
- Modify: `tests/methods/test_topology_method.py` (lines 83, 84, 86, 87)

**Context:** Now that mypy actually type-checks these files (the duplicate-module fatal error is fixed), mypy reports these 7 lines as:
```
error: Unused "type: ignore" comment  [unused-ignore]
```
Each currently ends in `# type: ignore[no-untyped-call]`, suppressing an "untyped call" error against `ext/topology`'s `graphFromMyFaces`/`graphFromShapes`/`.define_roads()`/`.define_interior_parcels()` methods — but `mypy_path` now includes `src` so topology (which ships `py.typed`) resolves with full types, and the calls are no longer flagged. The ignore comments are dead weight.

**Fix:** For each of these 7 lines, delete the trailing `  # type: ignore[no-untyped-call]` (and any leading whitespace before it that was only there for the comment), leaving the statement otherwise identical. Example (`tests/derive/test_parcel_graph.py:86`):
```python
# before
raw = graphFromMyFaces(_myfaces_from_parcels(block.parcels, origin))  # type: ignore[no-untyped-call]
# after
raw = graphFromMyFaces(_myfaces_from_parcels(block.parcels, origin))
```
Watch line length after removal — ruff's E501 (>100 chars) still applies; these lines get *shorter* by removing the comment, so this should be a non-issue, just don't reflow unnecessarily.

**After removing all 7:** re-run `pixi run mypy --strict tests/derive/test_parcel_graph.py tests/methods/test_topology_method.py`. If removing any ONE of them surfaces a NEW error at that line (meaning the ignore was silencing something real, just not what its old error-code claimed), do not blanket-ignore it — report it in your task report as a surprise and pick the minimal correct fix (or stop and ask if it looks like a real bug, per the plan's global constraint). Expect this NOT to happen (spec says "most will just delete cleanly") but verify rather than assume.

**Do NOT:**
- Remove any OTHER `# type: ignore` comments in these files that mypy is NOT flagging as unused — only the 7 flagged lines.

---

### Task 3: `test_render.py` — `str` → `CRS`

**Files:**
- Modify: `tests/test_render.py` (line 246)

**Context:** `test_displaced_points_carry_fraction_and_radius` (around line 239) builds its own local CRS instead of reusing the module-level `UTM = CRS.from_epsg(32643)` (line 18) — it uses a different EPSG code (32734) for this test. Currently:
```python
crs = "EPSG:32734"
```
This `str` is passed to `gpd.GeoDataFrame(..., crs=crs)` calls (fine, geopandas accepts strings) but ALSO to `Block(..., crs=crs, ...)` and `Proposal(..., crs=crs, ...)` at lines 251/254, whose dataclasses type `crs` as `pyproj.CRS` — mypy reports:
```
tests/test_render.py:251: error: Argument "crs" to "Block" has incompatible type "str"; expected "CRS"  [arg-type]
tests/test_render.py:254: error: Argument "crs" to "Proposal" has incompatible type "str"; expected "CRS"  [arg-type]
```
`pyproj.CRS` is already imported at the top of this file (`from pyproj import CRS`, line 11).

**Fix:** change line 246 from:
```python
crs = "EPSG:32734"
```
to:
```python
crs = CRS.from_epsg(32734)
```
This matches the file's own existing idiom (`UTM = CRS.from_epsg(32643)` at line 18) and requires no other changes — the same `crs` variable is reused for the `gpd.GeoDataFrame(..., crs=crs)` calls below, which accept a `CRS` object just as well as a string.

**Do NOT:**
- Touch the module-level `UTM` constant or any other test in this file.
- Add a second import of `CRS` (it's already imported).

---

### Task 4: `test_arterial_lazy.py` — narrow `GeoDataFrame | None`

**Files:**
- Modify: `tests/methods/test_arterial_lazy.py`

**Context:** `Proposal.roads` is typed `GeoDataFrame | None`. These tests call `.propose(block).roads` and then use the result (`len(...)`, `.geometry`, `.columns`, `[...]`) knowing it's non-None at runtime (arterial's `propose` always returns roads in these code paths), but mypy --strict can't know that. It reports:
```
tests/methods/test_arterial_lazy.py:21: error: Argument 1 to "len" has incompatible type "GeoDataFrame | None"; expected "Sized"  [arg-type]
tests/methods/test_arterial_lazy.py:27: error: Item "None" of "GeoDataFrame | None" has no attribute "geometry"  [union-attr]
tests/methods/test_arterial_lazy.py:99: error: Item "None" of "GeoDataFrame | None" has no attribute "geometry"  [union-attr]
tests/methods/test_arterial_lazy.py:100: error: Argument 1 to "len" has incompatible type "GeoDataFrame | None"; expected "Sized"  [arg-type]
tests/methods/test_arterial_lazy.py:150: error: Item "None" of "GeoDataFrame | None" has no attribute "columns"  [union-attr]
tests/methods/test_arterial_lazy.py:151: error: Argument 1 to "len" has incompatible type "GeoDataFrame | None"; expected "Sized"  [arg-type]
tests/methods/test_arterial_lazy.py:152: error: Value of type "GeoDataFrame | None" is not indexable  [index]
tests/methods/test_arterial_lazy.py:152: error: Argument 1 to "len" has incompatible type "GeoDataFrame | None"; expected "Sized"  [arg-type]
tests/methods/test_arterial_lazy.py:153: error: Value of type "GeoDataFrame | None" is not indexable  [index]
```
(Line 117's `scoring_fixtures` import-not-found in this same file is Task 1's, already fixed by then — ignore it.)

Fix each by adding `assert ... is not None` narrowing immediately after the `.roads` assignment, WITHOUT changing any existing assertion or test logic. Four call sites, verbatim:

1. `test_lazy_fixed_and_faithful_run_and_differ_from_exact_is_ok` (~lines 16-27): inside the `for pol in (...)` loop,
```python
        roads = GreedyArterialReblocker(mode="buildable", objective="directness", n_anchors=6,
                                        max_roads=4, lazy=True, candidate_policy=pol).propose(block).roads
        assert len(roads) >= 0            # all policies produce a valid proposal
```
add `assert roads is not None` on its own line right after the `roads = ...` assignment (before the existing `assert len(roads) >= 0` line). Then below, for the `a`/`b` determinism check:
```python
    a = GreedyArterialReblocker(mode="buildable", n_anchors=6, max_roads=3, lazy=True,
                                candidate_policy="fixed", rescore_every=1).propose(block).roads
    b = GreedyArterialReblocker(mode="buildable", n_anchors=6, max_roads=3, lazy=True,
                                candidate_policy="fixed", rescore_every=1).propose(block).roads
    assert [g.wkt for g in a.geometry] == [g.wkt for g in b.geometry]
```
add `assert a is not None and b is not None` on its own line right after the `b = ...` assignment, before the existing `assert [...]` line (this covers both line 21's and line 27's errors).

2. `test_lazy_dispatch_and_determinism` (~lines 93-100):
```python
    a = m.propose(block).roads
    b = m.propose(block).roads
    assert [g.wkt for g in a.geometry] == [g.wkt for g in b.geometry]   # deterministic
    assert len(a) > 0
```
add `assert a is not None and b is not None` right after the `b = ...` line, before the existing `assert [...]` line (covers lines 99 and 100).

3. `test_lazy_roads_carry_drain_column_like_exact` (~lines 141-153):
```python
    roads = GreedyArterialReblocker(mode="buildable", objective="directness", n_anchors=6,
                                    max_roads=4, lazy=True,
                                    candidate_policy="grow").propose(block).roads
    assert "drain" in roads.columns
    if len(roads):
        assert len(roads["drain"]) == len(roads)
        assert (roads["drain"] >= 0).all()
```
add `assert roads is not None` right after the `roads = ...` assignment, before the existing `assert "drain" in roads.columns` line (covers lines 150, 151, 152, 153 — a single narrowing above the block covers all four since they're all in the same function scope / `if` block that mypy tracks from the assert).

**Do NOT** touch `test_lazy_grow_with_max_anchors_runs_end_to_end` (~line 103-112) or `test_faithful_policy_matches_arterial_candidate_set`/other tests in this file — they already have `assert roads is not None` (or don't touch `.roads` on a `Proposal`) and are not in the error list.

**Verify:** `pixi run mypy --strict tests/methods/test_arterial_lazy.py` shows zero errors (assuming Task 1's `scoring_fixtures` override already landed — if run standalone before Task 1, the only remaining error should be the `scoring_fixtures` one at line 117, not any of the 9 above).

**Also fix (pre-existing, unrelated to mypy, but in this same file and blocking the `pixi run ruff check` gate):** `pixi run ruff check` currently reports 2 pre-existing `E501` (line too long, >100 chars) violations in this file, at lines 20 and 22 (before your edits shift line numbers):
```
tests/methods/test_arterial_lazy.py:20:101: E501 Line too long (106 > 100)
tests/methods/test_arterial_lazy.py:22:101: E501 Line too long (102 > 100)
```
These predate this whole cleanup effort (confirmed present on `main`/merge-base `943bbef`) and are unrelated to the type-narrowing work, but since you're already editing this file, reflow both to ≤100 chars as part of this task (e.g. wrap the `roads = GreedyArterialReblocker(...)` call's arguments across an extra line; shorten or wrap the long comment on line 22) — do not change their semantics, just re-wrap. Re-run `pixi run ruff check tests/methods/test_arterial_lazy.py` to confirm both are gone.

**Do NOT:**
- Change any existing assertion's condition or comparison.
- Add narrowing to functions/lines not in the error list above.

---

### Task 5: Whole-repo gate verification

**Files:** none (verification only — but if the gates below are NOT green, this task must identify and either fix directly or escalate whatever residual is preventing green, staying within the Global Constraints).

**Do, in order, from `/home/gchurchill/src/reblock`:**
1. `pixi run typecheck` — must print `Success: no issues found in <N> source files` with exit code 0. Zero errors anywhere (not just the 22 originally listed).
2. `pixi run pytest` — must pass with the SAME test count as before this branch's changes (no test added, removed, skipped, or newly failing). Record the pass count. Baseline recorded by the controller before Task 1: **357 passed**.
3. `pixi run ruff check` — must be clean. Note: at the start of this whole cleanup effort (before any task), `pixi run ruff check` already reported 3 PRE-EXISTING violations unrelated to mypy: 2 in `tests/methods/test_arterial_lazy.py` (lines 20, 22 — Task 4 fixes these) and 1 in `scripts/fetch_desire_lines_snapshot.py:5` (a docstring line >100 chars, confirmed present on `main`/merge-base `943bbef`, untouched by any of Tasks 1-4). If that script's E501 is still present at this point, fix it too (reflow the docstring comment, no semantic change) so the gate is genuinely clean — it is in scope for this task specifically because the plan's gate demands `ruff check` be clean, not merely "no new violations."
   The module docstring currently reads (lines 1-12 of `scripts/fetch_desire_lines_snapshot.py`):
   ```python
   """One-off: fetch OSM desire-lines for a region/block and write a committed GeoJSON snapshot so the
   examples reproduce osm_footpaths offline + byte-stable (no live Overpass call at example time).

   Run (module form -- puts the repo root on sys.path so the data source's `from scripts...` import
   resolves): `pixi run python -m scripts.fetch_desire_lines_snapshot <out.geojson> <hydra override>...`
   ...
   """
   ```
   Line 5 (`resolves): ...`) is 101 chars. Reflow lines 4-5 into three shorter lines instead of two, e.g.:
   ```python
   Run (module form -- puts the repo root on sys.path so the data source's `from scripts...`
   import resolves): `pixi run python -m scripts.fetch_desire_lines_snapshot <out.geojson>
   <hydra override>...`
   ```
   (exact wrap points flexible — just keep every line ≤100 chars and the sentence meaning unchanged).

If anything is not green, diagnose within the constraints above (no relaxing beyond what's needed, no changing test logic) and fix, or report precisely what's blocking and stop rather than papering over it.

**Report:** this task's report doubles as the final validation evidence for the whole plan — include full `pixi run typecheck` and `pixi run pytest` tail output.
