# External-Connectivity Target (Lens A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retarget the two-lens comparison's Lens A (fixed OUTCOME) from a max-depth target (depth ≤ 3) to an external-connectivity target (external connectivity ≥ 0.70), demoting depth to informational.

**Architecture:** Add `prefix_to_external_connectivity` to `budget.py` (mirrors `prefix_to_depth` but binary-searches the monotone-non-decreasing external-connectivity/`access_benefit` axis for the smallest prefix reaching the target). Rewire `run_two_lens`/`two_lens_rows`/`main` in `compare_budgets.py` to use it; migrate the artifact names (`lens_a_depth.csv` → `lens_a_external.csv`, `after_<m>_depth3.jpg` → `after_<m>_ext70.jpg`). Point the orchestrator and README generator at the new outcome. Regenerate the six examples.

**Tech Stack:** Python, geopandas/shapely, matplotlib, Hydra; `pixi run check` (ruff + mypy --strict + pytest).

## Global Constraints

- **Migrate, never accommodate:** rename the Lens A CSV and after-image tag; delete the old names. NO dual depth/connectivity path, NO back-compat shim, NO `target_depth` kept "just in case". The depth target BECOMES the connectivity target.
- **X = 0.70** external connectivity is the absolute bar (calibrated from the committed `frontier_external_connectivity.csv`: osm never reaches it, clearance reaches it in the four depth-family regions, dense `density_compactness` regions are reported unreached like osm-at-depth-3).
- **Depth stays informational:** the depth-vs-road curve (`depth_vs_road_report`) and screening/region-growth are UNCHANGED. Only Lens A's outcome target moves.
- `pixi run check` green every task. ruff forbids E702 semicolons, E501 >100-char lines, B905 bare zip.
- Commit trailers: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` + `Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x`.

---

### Task 1: `prefix_to_external_connectivity` in budget.py

**Files:**
- Modify: `src/reblock/budget.py` (add function next to `prefix_to_depth`)
- Test: `tests/test_budget.py`

**Interfaces:**
- Consumes: `_drainage_ordered(block, roads, tol)`, `access_benefit(block, None, tol=tol)` (returns `f(roads)->float` external connectivity, monotone non-decreasing as roads are added — see its docstring), `STREET_TOL`.
- Produces: `prefix_to_external_connectivity(block: Block, roads: GeoDataFrame, target_ext: float, *, tol: float = STREET_TOL) -> tuple[GeoDataFrame, float]` — the minimal drainage-ordered prefix whose external connectivity ≥ `target_ext`, paired with that prefix's actual external connectivity. If even all roads can't reach it, returns `(all roads in drainage order, full_ext)` with `full_ext < target_ext` (caller reports unreached). Empty roads → `(empty, 0.0)`.

- [ ] **Step 1: Write the failing test** in `tests/test_budget.py`. Reuse that file's existing deep-column fixture pattern (a 4-deep column fronting a street, two stacked roads that drain it). Assert: (a) a reachable target returns a prefix whose returned external connectivity ≥ target and ≤ the full-road connectivity, and no longer than all roads; (b) an unreachable target (e.g. 1.5) returns all roads in drainage order with returned connectivity < target (unreached); (c) empty roads returns an empty GeoDataFrame and 0.0.

- [ ] **Step 2: Run it, verify it fails** (`pixi run pytest tests/test_budget.py -k external_connectivity -q`) — NameError / not defined.

- [ ] **Step 3: Implement**, mirroring `prefix_to_depth` but on the connectivity axis (monotone NON-DECREASING, so we want the smallest `m` with `ext_at(m) >= target_ext`):

```python
def prefix_to_external_connectivity(block: Block, roads: GeoDataFrame, target_ext: float, *,
                                    tol: float = STREET_TOL) -> tuple[GeoDataFrame, float]:
    """The minimal drainage-ordered prefix of `roads` whose external connectivity
    (`access_benefit`, fraction of access-burden Sigma-d^2 removed) is >= `target_ext`, paired with
    that prefix's actual external connectivity. Connectivity is monotone NON-DECREASING as
    drainage-ordered roads are added (access_burden's unreached-depth cap makes access_benefit
    monotone), so a binary search over the prefix length finds the smallest sufficient prefix in
    O(log R) peels. If even all `roads` cannot reach `target_ext`, returns (all roads in drainage
    order, full connectivity) with that value < `target_ext` -- the caller reports unreached (an
    osm_footpaths-style fixed input that never reaches the target). Empty `roads` returns
    (empty, 0.0)."""
    ext = access_benefit(block, None, tol=tol)
    if len(roads) == 0:
        return cast(GeoDataFrame, roads.iloc[:0]), 0.0
    ordered = _drainage_ordered(block, roads, tol)

    def ext_at(m: int) -> float:
        return ext(cast(GeoDataFrame, ordered.iloc[:m]))

    n = len(ordered)
    full_ext = ext_at(n)
    if full_ext < target_ext:                     # unreachable: best effort is all roads
        return ordered, full_ext
    lo, hi = 0, n                                 # smallest m with ext_at(m) >= target_ext
    while lo < hi:
        mid = (lo + hi) // 2
        if ext_at(mid) >= target_ext:
            hi = mid
        else:
            lo = mid + 1
    return cast(GeoDataFrame, ordered.iloc[:lo].reset_index(drop=True)), ext_at(lo)
```

- [ ] **Step 4: Run the test, verify it passes.** Then `pixi run check`.

- [ ] **Step 5: Commit** (`feat: prefix_to_external_connectivity — smallest drainage prefix reaching an external-connectivity target`).

---

### Task 2: Retarget Lens A in compare_budgets.py

**Files:**
- Modify: `scripts/compare_budgets.py`
- Test: `tests/test_compare_budgets.py`

**Interfaces:**
- Consumes: `prefix_to_external_connectivity` (Task 1).
- Produces: `two_lens_rows(..., target_ext: float, budget_m, ...)` and `run_two_lens(region, methods, target_ext: float, out_dir, ...)`; `LensARow` with `reached_ext: float` (replacing `reached_depth: int`); writes `lens_a_external.csv`; after-image tag `ext{int(round(target_ext*100))}` (e.g. `after_clearance_ext70.jpg`).

- [ ] **Step 1: Update `tests/test_compare_budgets.py`** to the new API (these are the failing tests first):
  - `two_lens_rows(..., target_ext=0.5, ...)` calls; assert `a.reached is True` and `a.reached_ext >= 0.5` (drop `reached_depth`).
  - The unreachable case: `target_ext=1.5` → `a.reached is False and a.reached_ext < 1.5`.
  - `run_two_lens(..., target_ext=0.3, out_dir=tmp_path)`; assert `(tmp_path / "lens_a_external.csv").exists()` and `(tmp_path / "after_dijkstra_ext30.jpg").exists()`.
  - The reblock-once spy test: `target_ext=0.3`.

- [ ] **Step 2: Run those tests, verify they fail** (old names gone / signature mismatch).

- [ ] **Step 3: Implement in `scripts/compare_budgets.py`:**
  - Import `prefix_to_external_connectivity` (remove `prefix_to_depth` import if now unused — check; it is used only by Lens A, so remove it).
  - `LensARow`: replace `reached_depth: int` with `reached_ext: float` (update the field comment: the prefix's actual external connectivity; the floor when not reached).
  - `two_lens_rows`: signature `target_depth: int` → `target_ext: float`. Lens A: `prefix_a, reached_ext = prefix_to_external_connectivity(block, roads, target_ext)`; `reached = reached_ext >= target_ext`. Keep everything else (displacement, pct_displaced, propose_seconds, Lens B) identical.
  - `run_two_lens`: signature `target_depth: int` → `target_ext: float`. In the after-image loop, `prefix_a, _ = prefix_to_external_connectivity(block, roads_by_method[name], target_ext)` and the tag becomes `f"ext{int(round(target_ext * 100))}"`. Write `lens_a_external.csv` with header `["method", "target_ext", "reached", "reached_ext", "road_length_m", "displacement", "pct_displaced", "propose_seconds"]` and rows using `f"{target_ext:.2f}"`, `r.reached`, `f"{r.reached_ext:.4f}"`, etc.
  - `main`: `target_ext = float(sys.argv[2])`; pass through; update the two `print` lines (Lens A: `reached ext {a.reached_ext:.2f}` vs `target_ext`; the unreached branch prints `FLOOR ext {a.reached_ext:.2f}`).
  - Update all docstrings (module docstring Lens A description, `two_lens_rows` docstring) to describe the connectivity target instead of the depth target.

- [ ] **Step 4: Run the tests, verify they pass.** Then `pixi run check`.

- [ ] **Step 5: Commit** (`feat: retarget Lens A to an external-connectivity outcome (>=0.70), migrating the CSV + after-image names`).

---

### Task 3: Orchestrator + README generator + fixture

**Files:**
- Modify: `scripts/gen_multiblock_example.py`, `scripts/gen_example_readme.py`
- Rename in fixture: `tests/data/example_fixture/after_clearance_depth3.jpg` → `after_clearance_ext70.jpg`, `after_greedy_arterial_buildable_depth3.jpg` → `after_greedy_arterial_buildable_ext70.jpg`, `lens_a_depth.csv` → `lens_a_external.csv`
- Test: `tests/test_gen_example_readme.py`

**Interfaces:**
- Consumes: `run_two_lens(..., target_ext, ...)` (Task 2).

- [ ] **Step 1: Update `tests/test_gen_example_readme.py`**: the §4 heading assertion `"Matched access target"` → `"Matched external-connectivity target"`; the matched-access after-image glob is now `after_<m>_ext70.jpg` (the fixture rename handles the files). Keep the `after_clearance_matched.jpg` assertion.

- [ ] **Step 2: Rename the fixture files** (git mv the three files listed above). The `lens_a_external.csv` keeps the fixture's existing row content (its header text isn't asserted).

- [ ] **Step 3: Run the README test, verify it fails** (heading text mismatch / glob finds nothing until code updated).

- [ ] **Step 4: Implement:**
  - `gen_multiblock_example.py`: the `run_two_lens(region, methods, 3, out, label=seed, extend=extend)` call → `run_two_lens(region, methods, 0.70, out, label=seed, extend=extend)`.
  - `gen_example_readme.py`: in §4, `depth_imgs = sorted(run_dir.glob("after_*_depth*.jpg"))` → `sorted(run_dir.glob("after_*_ext*.jpg"))`; the heading text `"**Matched access target** — every method truncated where access-depth reaches the target, so this compares the *road each takes* for the same outcome:"` → `"**Matched external-connectivity target** — every method truncated where external connectivity (access-burden removed) reaches 0.70, so this compares the *road each takes* for the same outcome:"`. In §3's connectivity-curve note, the CSV reference `lens_a_depth.csv` → `lens_a_external.csv`. In §3's depth-curve note, add that the depth curve is now shown **for reference** (the method budget is the external-connectivity outcome in Lens A), keeping the existing continued-past-depth-target sentence.

- [ ] **Step 5: Run the README test, verify it passes.** Then `pixi run check`.

- [ ] **Step 6: Commit** (`feat: point the example orchestrator + README at the external-connectivity outcome`).

---

## Regeneration (controller, after all tasks + final review)

Not a subagent task. Regenerate all six examples with the new outcome:
`pixi run python -m scripts.gen_multiblock_example <metric> [nairobi]` for depth/depth_density/density_compactness × {capetown, nairobi}. Verify each `lens_a_external.csv` has sensible reached/unreached rows and the `after_<m>_ext70.jpg` images render; sanity-check X=0.70 against the calibration (clearance reaches in the depth-family regions, osm/unreached elsewhere). Revert only spurious timing churn. Commit the regenerated artifacts.
