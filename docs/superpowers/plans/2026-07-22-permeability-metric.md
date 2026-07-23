# Permeability Metric — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. **Task 3 ends at a human checkpoint** — the controller runs the calibration probe and gets sign-off on `(D, P*)` before Task 4.

**Goal:** Replace external+internal connectivity with a single flow-based **permeability** metric; report only permeability + displacement-fraction, as one frontier curve + two lenses; retire `commute_ratio`/freeze/`access_benefit`-as-metric.

**Architecture:** A validated prototype exists at `/tmp/claude-1641171234/-home-gchurchill-src-reblock/27c82570-a74d-47e6-9e87-e53987507f6d/scratchpad/contention_flow.py` — port it. Permeability = normalized total dissipated power of an all-parcels grounded egress flow on a Voronoi footpath mesh that roads upgrade; monotone (Rayleigh), one sparse solve (~4 s @ 11k parcels).

**Spec:** `docs/superpowers/specs/2026-07-22-permeability-metric-design.md` (read for rationale).

**On this branch already:** displacement-as-fraction + percent display (dual-target Task 1) — KEEP. The dual-target `prefix_to_joint_target`/`JointTargetOutcome` and `scripts/calibrate_joint_target.py` are SUPERSEDED by this plan (retired in Tasks 2–3).

## Global Constraints

- **No back-compat shims (owner directive).** Migrate call sites, delete retired code. Retire: `commute_ratio`, `commute_ratio_benefit`, `_commute_membership`, `_commute_setup`, `_nearest_edge_ratio`, the freeze; `access_benefit`/`access_burden`/`commute_ratio` **as reported metrics/curves**; `prefix_to_external_connectivity`; `prefix_to_joint_target`/`JointTargetOutcome`; `scripts/calibrate_joint_target.py`; `run_two_lens`/`two_lens_rows`/`LensARow`/`LensBRow`. **KEEP** `parcel_access_layers` + the `depth` proxy (the SCREEN uses them) and the k-complexity `access_after` field (one of the two heatmap colorings).
- **permeability ∈ [0,1)**, `= 1 − P(roads)/P(no_roads)`, monotone non-decreasing in the road set. `g_road/g_walk = 20` (validated). Params in `conf/permeability.yaml`.
- **Report permeability + displacement ONLY.** One frontier curve (permeability y vs displacement x). Two lenses: matched-displacement `D` (compare permeability) + matched-permeability `P*` (compare displacement; a method never reaching `P*` reads "unreached"). No kill machinery.
- **Rendering:** before + per-method-after images, EACH in BOTH colorings (access-depth `_depth`, permeability-potential `_perm`); screen map with no colorbar/title; no `block_id` on any plot; title-less curves labeled `displacement`/`permeability`; all static images 300 dpi + poster-grade (figsize giving ≳3000 px long edge, line/font/marker sizes scaled up). GIFs unchanged.
- `(D, P*)` are set ONCE from the calibration probe at the Task 3 checkpoint; `i_min: null`-style placeholders until then.

## File Structure

- `src/reblock/permeability.py` — NEW: the metric (port of the prototype), `PermeabilityParams`, `egress_power` (returns P + per-parcel potentials `v`), `permeability`, `permeability_curve`.
- `src/reblock/budget.py` — `prefix_to_displacement`, `prefix_to_permeability` (lenses); retire `prefix_to_external_connectivity`, `prefix_to_joint_target`/`JointTargetOutcome`, and (Task 7) `commute_ratio`+freeze.
- `conf/permeability.yaml` — metric params + `(D, P*)` thresholds.
- `scripts/calibrate_permeability.py` — NEW probe (isolated per region). Retire `scripts/calibrate_joint_target.py`.
- `src/reblock/render.py`, `src/reblock/emit.py` — heatmap both-colorings + poster sizing + curve/screen cleanups.
- `scripts/compare_budgets.py` — `run_two_lens` → `run_permeability_lenses`.
- `scripts/gen_multiblock_example.py`, `scripts/gen_method_comparison.py`, `scripts/gen_example_readme.py`, `scripts/regenerate_examples.sh` — wire + README story.
- Tests: `tests/test_permeability.py` (new), `tests/test_budget.py`, `tests/test_compare_budgets.py`, `tests/test_emit.py`, `tests/test_gen_example_readme.py`.

---

### Task 1: The permeability metric

**Files:** Create `src/reblock/permeability.py`; Test `tests/test_permeability.py`.

**Interfaces (Produces):**
- `@dataclass(frozen=True) class PermeabilityParams: g_walk=1.0; g_road=20.0; g_street=20.0; corridor_m=3.0`
- `egress_power(block, roads, params=PermeabilityParams(), *, adj=None) -> tuple[float, NDArray]` — returns `(P, v)`: total dissipated power `P = bᵀv` and per-parcel potentials `v` (for the heatmap). `(+inf, zeros)` if no street-fronting parcel.
- `permeability(block, roads, params=PermeabilityParams(), *, p0=None, adj=None) -> float` — `1 − P(roads)/P(no_roads)` ∈ [0,1); the reported metric.
- `parcel_potentials(block, roads, params=PermeabilityParams()) -> pd.Series` — `v` indexed by `parcel_id`, for the `_perm` heatmap coloring.

- [ ] **Step 1: Port + failing tests.** Port the prototype's `contention_power` graph assembly + sparse solve into `egress_power` (return `(P, v)` — capture `v = spsolve(...)` and `P = b @ v`; the prototype already computes `v`, just also return it). Reuse `reblock.derive.access.parcel_adjacency`, `STREET_TOL`, scipy.sparse. Write `tests/test_permeability.py`:

```python
import numpy as np, geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon
from reblock.permeability import PermeabilityParams, egress_power, permeability
from reblock.contracts import Block
UTM = CRS.from_epsg(32734)

def _grid_block(k=4):
    # k x k unit parcels tiling a k x k square; south edge (y=0) is the street.
    polys, ids = [], []
    for r in range(k):
        for c in range(k):
            polys.append(Polygon([(c, r), (c+1, r), (c+1, r+1), (c, r+1)])); ids.append(r*k+c)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (k, 0)])], crs=UTM)
    boundary = Polygon([(0, 0), (k, 0), (k, k), (0, k)])
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)

def _roads(lines): return gpd.GeoDataFrame(geometry=lines, crs=UTM)

def test_no_roads_permeability_is_zero():
    b = _grid_block()
    assert permeability(b, None) == 0.0 and permeability(b, _roads([])) == 0.0

def test_permeability_in_unit_interval_and_positive_with_a_road():
    b = _grid_block()
    p = permeability(b, _roads([LineString([(2, 0), (2, 4)])]))   # a spine road to the interior
    assert 0.0 < p < 1.0

def test_monotone_under_added_roads():
    b = _grid_block(6)
    r1 = _roads([LineString([(3, 0), (3, 6)])])
    r2 = _roads([LineString([(3, 0), (3, 6)]), LineString([(0, 3), (6, 3)])])   # superset
    assert permeability(b, r2) >= permeability(b, r1) - 1e-12    # adding roads never lowers it

def test_loop_beats_spur_at_equal_length():
    # a closed loop reaching the street twice vs a single spur of equal length -> loop lower P
    b = _grid_block(6)
    spur = _roads([LineString([(3, 0), (3, 5)])])                       # 5 m single egress
    loop = _roads([LineString([(2, 0), (2, 2.5), (4, 2.5), (4, 0)])])   # ~5 m, two egresses
    assert permeability(b, loop) > permeability(b, spur)

def test_ungrounded_returns_zero_benefit_or_guarded():
    b = _grid_block()
    b.streets.geometry = gpd.GeoSeries([], crs=UTM)   # no street -> no ground
    P, v = egress_power(b, None)
    assert not np.isfinite(P)     # +inf; permeability() guards this (returns nan) -- assert its guard
```

Run: `pixi run pytest tests/test_permeability.py -v` → FAIL (import).

- [ ] **Step 2: Implement `src/reblock/permeability.py`** — port `contention_power` verbatim (it is validated), renamed to `egress_power` returning `(P, v)`; add `permeability` (`p0` = `egress_power(block, None)`; guard non-finite/≤0 → the tests expect 0.0 at no-roads, nan if ungrounded); add `parcel_potentials` (solve for `v`, return `pd.Series(v, index=block.parcels["parcel_id"])`). NO scratchpad imports — port the code. Confirm imports (`scipy.sparse`, `numpy`, `parcel_adjacency`, `STREET_TOL`, `unary_union`, `LineString`).

- [ ] **Step 3: Run** `pixi run pytest tests/test_permeability.py -v` → PASS; ruff + mypy --strict clean on the new module.
- [ ] **Step 4: Commit** — `git commit -m "Add permeability metric: all-parcels grounded egress flow (sparse, monotone)"`

---

### Task 2: permeability_curve + the two lens truncations

**Files:** Modify `src/reblock/permeability.py` (curve), `src/reblock/budget.py` (lenses; retire `prefix_to_external_connectivity`, `prefix_to_joint_target`/`JointTargetOutcome`); Test `tests/test_permeability.py`, `tests/test_budget.py`.

**Interfaces (Produces):**
- `permeability_curve(block, roads, params=PermeabilityParams(), *, n_points=20, tol=STREET_TOL) -> Curve` — x = cumulative road length; y = permeability per drainage-ordered prefix (freeze `p0` once; monotone). Reuses `_sweep`.
- `prefix_to_displacement(block, roads, radii, d_frac, *, corridor_m=3.0, tol) -> GeoDataFrame` — the minimal drainage prefix whose displacement-fraction ≥ `d_frac` (binary search; displacement monotone). If full roads < `d_frac`, return all.
- `prefix_to_permeability(block, roads, p_star, params, *, tol) -> tuple[GeoDataFrame, bool]` — minimal drainage prefix with permeability ≥ `p_star` (binary search; permeability monotone), + `reached` flag (False + full roads if never reached).

- [ ] **Step 1: Failing tests** for `permeability_curve` (terminal == `permeability(block, roads)`; all in [0,1); monotone non-decreasing), `prefix_to_displacement` (prefix hits the fraction; monotone), `prefix_to_permeability` (reached vs unreached). Model these on `tests/test_budget.py`'s existing curve/truncation tests.
- [ ] **Step 2: Implement.** `permeability_curve` mirrors `displacement_curve` (a `_sweep` over `permeability(block, prefix, params, p0=<frozen>)`). The two lenses mirror `prefix_to_external_connectivity`'s binary search (both monotone). RETIRE `prefix_to_external_connectivity`, `prefix_to_joint_target`, `JointTargetOutcome` (delete + migrate any imports — `test_joint_target.py` retires with them).
- [ ] **Step 3: Run** `pixi run pytest tests/test_permeability.py tests/test_budget.py -q` → PASS (delete the retired-symbol tests).
- [ ] **Step 4: Commit** — `git commit -m "permeability_curve + matched-displacement/matched-permeability lenses; retire joint-target"`

---

### Task 3: Calibration probe (ends at human checkpoint)

**Files:** Create `conf/permeability.yaml`, `scripts/calibrate_permeability.py`; delete `scripts/calibrate_joint_target.py`; Test `tests/test_permeability.py`.

- [ ] **Step 1: `conf/permeability.yaml`** — metric params (`g_walk: 1.0, g_road: 20.0, g_street: 20.0, corridor_m: 3.0`) + thresholds `matched_displacement: null`, `matched_permeability: null` (checkpoint placeholders).
- [ ] **Step 2: Probe** `scripts/calibrate_permeability.py` — for each method (arterial_repulsion, clearance_looped, euclidean_grid; osm_footpaths reference), NATURAL config (NO over-provisioning), on each of the 6 multiblock regions + the method-comparison block, build the permeability-vs-displacement frontier and print a per-(method, region) table of {permeability at a few displacement %s; displacement to reach a few permeability levels}. **Load each region in a FRESH SUBPROCESS** (isolation — the earlier in-process probe suffered cross-region cache bleed; run one region per `subprocess.run([...python -m scripts.calibrate_permeability, <region-arg>])` invocation, or fork per region, and aggregate). Propose `(D, P*)` from dynamic range + a humane home budget. Add a unit test for its frontier-extraction helper (e.g. `permeability_at_displacement(perm_curve, disp_curve, d)` — analogous to the retired `max_internal_within`).
- [ ] **Step 3: Run the helper test** → PASS. Commit — `git commit -m "Permeability calibration probe (per-region isolated) + config"`.
- [ ] **CHECKPOINT (controller):** run the probe, present the frontiers + proposed `(D, P*)` to the human, get sign-off, write them into `conf/permeability.yaml`, commit. THEN dispatch Task 4.

---

### Task 4: Rendering primitives — both heatmap colorings, poster sizing, curve/screen cleanups

**Files:** Modify `src/reblock/render.py`, `src/reblock/emit.py`; Test `tests/test_emit.py`, `tests/test_render.py`.

- [ ] **Step 1:** `render_before`/`render_after`/`_draw_heatmap` gain a coloring mode: they already color parcels by a `layers: pd.Series` with `_CMAP`. Generalize so a caller passes EITHER the access-depth layers (existing, integer `vmin=1..vmax`, `_CMAP`) OR the permeability potentials (continuous; a sequential cmap normalized to `[0, v.max()]`, dark = high potential = hard escape). Keep the signature backward-friendly via a `field: Literal["depth","perm"]` + the corresponding series/vmax. The caller (Task 5 driver) renders each image twice.
- [ ] **Step 2: Poster sizing.** Bump heatmap `figsize=(10,10)`→`(12,12)` and curve `figsize=(7,5)`→`(12,9)`; scale line widths, font sizes, and marker sizes proportionally so they read at poster scale; keep `save_render`'s `dpi=300`. (Target ≳3000 px long edge after `bbox_inches="tight"`.)
- [ ] **Step 3: Curve + screen cleanups in `compare_report`/`region_map`.** ONE frontier curve: permeability (y) vs displacement (x) — drop title, x-label `displacement`, y-label `permeability`, no `block_id` in filename-title. Remove the colorbar + title from the screen map (`region_map`), AND remove the region boundary-following outline (the per-member black outline drawn at ~emit.py:202, `edgecolor="black"`) so the metric colors show unoccluded — keep ONLY the thick black bounding box. Strip `block_id`/parcel-name titles from all plots. (The old external/internal curve plotting is removed in Task 5 with the driver; here, make `compare_report` render the single permeability-vs-displacement frontier.)
- [ ] **Step 4:** tests updated (smoke: the frontier PNG renders; both heatmap colorings produce files; no title/colorbar on screen map). Run `pixi run pytest tests/test_emit.py tests/test_render.py -q` → PASS. Commit.

---

### Task 5: Reporting driver + wire generators (retire the two-lens/external/internal surface)

**Files:** Modify `scripts/compare_budgets.py` (`run_two_lens`→`run_permeability_lenses`), `scripts/gen_multiblock_example.py`, `scripts/gen_method_comparison.py`, `src/reblock/compare.py`; Test `tests/test_compare_budgets.py`.

- [ ] **Step 1:** `run_permeability_lenses(region, methods, out_dir, *, matched_displacement, matched_permeability, params, corridor_m=3.0, label=None) -> list[OutcomeRow]`: reblock each method once (natural config); build the **permeability_curve** + **displacement_curve** per method → the single frontier (`compare_report`); compute Lens A (`prefix_to_displacement`) + Lens B (`prefix_to_permeability`) prefixes; render the **before** image (both colorings) + a per-method **after** image per lens (both colorings) via the Task-4 render modes (compute `parcel_potentials` for the `_perm` coloring, `KComplexityEval` `access_after` for the `_depth` coloring); write two outcome tables (`lens_displacement.csv`, `lens_permeability.csv`); keep the per-method GIF. Retire `run_two_lens`, `two_lens_rows`, `LensARow`, `LensBRow`, and the external/internal curve building.
- [ ] **Step 2:** wire `gen_multiblock_example.py` + `gen_method_comparison.py` to `run_permeability_lenses` with the `conf/permeability.yaml` params + thresholds. `src/reblock/compare.py`: drop the external/internal/displacement three-curve build; if it still has a role, build only the permeability frontier (else fold into the driver).
- [ ] **Step 3:** `pixi run pytest tests/test_compare_budgets.py -q` then `pixi run pytest -q` → PASS (migrate/delete tests bound to the retired lens surface). Commit.

---

### Task 6: README generators → permeability story

**Files:** Modify `scripts/gen_example_readme.py`, `examples/method-comparison/README.md`; Test `tests/test_gen_example_readme.py`.

- [ ] **Step 1:** replace the two-lens/external/internal README sections with: the single **permeability-vs-displacement frontier**; the **before** image (both colorings); the **two lens** sections (matched-displacement table + after-images; matched-permeability table + after-images), each showing both colorings; permeability + displacement described as the only two axes. Read the new `lens_*.csv`.
- [ ] **Step 2:** migrate `tests/test_gen_example_readme.py` assertions onto the permeability strings (drop external/internal/matched-budget). Run → PASS. Commit.

---

### Task 7: Retire commute_ratio / freeze / access_benefit-as-metric

**Files:** Modify `src/reblock/budget.py` (delete `commute_ratio`, `commute_ratio_benefit`, `_commute_membership`, `_commute_setup`, `_nearest_edge_ratio`, `access_benefit`, `access_burden` and helpers no longer used as reported metrics); Test files.

- [ ] **Step 1:** confirm (grep) nothing outside the screen path + tests imports these; the SCREEN uses `parcel_access_layers` + the `depth` proxy (KEEP those) and the k-complexity `access_after` field (KEEP — it's a heatmap coloring). Delete the retired functions + `loop_closure.py`/`test` references to `commute_ratio` as an eval (the LoopClosureRefiner still builds loops — it just no longer scores via `commute_ratio`; if it needs an internal score, use `permeability` or leave it geometry-only per its own tests).
- [ ] **Step 2:** delete/adapt `tests/test_commute_ratio.py`, and the `commute_ratio` assertions in `tests/test_loop_closure.py`/`tests/test_compare.py`. Run `pixi run pytest -q` → PASS.
- [ ] **Step 3: Commit** — `git commit -m "Retire commute_ratio, freeze, and access_benefit as reported metrics (permeability replaces them)"`

---

### Task 8 (controller): Regenerate all examples + verify

Not an SDD task: `pixi run bash scripts/regenerate_examples.sh`; verify the frontier curves read monotone, both heatmap colorings render, the two lens tables/images are present, before-images exist, poster sizing looks crisp on a sample; confirm the survivor/permeability pattern matches the probe. Rewrite `examples/method-comparison/README.md` prose to its regenerated numbers. Commit the regenerated `examples/**`.

## Self-Review

- **Spec coverage:** metric (T1) ✓; curve + lenses (T2) ✓; calibration + checkpoint (T3) ✓; both colorings + poster + cleanups (T4) ✓; driver + retire two-lens (T5) ✓; READMEs (T6) ✓; retire commute_ratio/freeze/access_benefit (T7) ✓; regen (T8) ✓.
- **Placeholder scan:** the `null` thresholds in `conf/permeability.yaml` are the deliberate checkpoint placeholder; all code steps concrete (the metric is a port of validated prototype code).
- **Type/name consistency:** `PermeabilityParams`, `egress_power`→`(P,v)`, `permeability`, `permeability_curve`, `parcel_potentials`, `prefix_to_displacement`, `prefix_to_permeability`, `run_permeability_lenses`, `OutcomeRow` consistent across tasks; displacement-fraction (kept) unchanged.
- **Retirement safety (T7):** the screen path (`parcel_access_layers`/`depth`) and the `access_after` heatmap field are explicitly KEPT; only the *reported-metric* functions are deleted.
