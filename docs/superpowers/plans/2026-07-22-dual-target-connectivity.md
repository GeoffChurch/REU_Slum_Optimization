# Dual-Target Connectivity Outcome — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. **Task 3 ends at a human checkpoint** — the controller runs the calibration probe and gets sign-off on the derived `(I_min, E_min, D_max)` before Task 4.

**Goal:** Replace the external-only example calibration (two lenses) with a single joint-target outcome: run each synthetic method until it first attains `external ≥ E_min` **and** `internal ≥ I_min` at `displacement ≤ D_max`, else it is killed. Universal, empirically-probed `(I_min, E_min, D_max)`. Displacement becomes a fraction of homes everywhere.

**Architecture:** The three per-method curves (external `access_benefit`, internal frozen `commute_ratio`, displacement) already share the same drainage-ordered `_sweep` sample grid (invariant locked by `test_curves_share_cost_samples`). The outcome is therefore a cheap post-hoc scan of the first sample index meeting all three conditions — no extra reblock or resistance solve. Methods are over-provisioned past natural convergence so the scan can reach `D_max`.

**Tech Stack:** Python, numpy, geopandas, shapely, matplotlib, Hydra config, pytest, pixi.

**Design spec:** `docs/superpowers/specs/2026-07-22-dual-target-connectivity-outcome-design.md` (read for rationale; this plan is the build).

## Global Constraints

- **No back-compat shims / no dual code paths (owner directive).** Retire `prefix_to_external_connectivity`, `two_lens_rows`, `LensARow`, `LensBRow`, the `lens_a_external.csv`/`lens_b_matched.csv` outputs — migrate call sites, delete the old path. `displacement_curve` changes meaning (now a fraction); migrate, don't branch.
- **Displacement is the home fraction everywhere it is plotted, tabulated, or thresholded**: `Σcᵢ / n_buildings`, `n_buildings = len(block.building_points)` (buildings, not parcels). The raw `displacement()` stays as the `Σcᵢ` primitive.
- **Outcome = first sample index** `i*` with `external.benefit[i*] ≥ E_min` and `internal.benefit[i*] ≥ I_min` and `displacement.benefit[i*] ≤ D_max`. Touch-and-go: stop at the first qualifying index even if internal later dips. The three curves are index-aligned — rely on `test_curves_share_cost_samples`.
- **Killed** = no qualifying index within `displacement ≤ D_max`; reasons: `internal_below` (never reaches I_min) or `over_budget` (floors only past D_max). `clearance_looped` may be killed — accepted.
- `(I_min, E_min, D_max)` are universal constants in `conf/joint_target.yaml`, set once from the calibration probe (Task 3 checkpoint). All are home-fraction/benefit floats.
- All existing tests stay green except those asserting the retired two-lens surface (migrate those to the outcome surface). Do not weaken assertions to pass.

---

## File Structure

- `src/reblock/budget.py` — `displacement_curve` returns the fraction (Task 1); new `JointTargetOutcome` + `prefix_to_joint_target` (Task 2); retire `prefix_to_external_connectivity` (Task 5).
- `src/reblock/emit.py` — `compare_report` displacement labels → fraction + `displacement_table` consolidation (Task 1); outcome-marker overlay (Task 4).
- `conf/joint_target.yaml` — universal constants + per-method over-provisioning knobs (Task 3, filled at checkpoint).
- `scripts/calibrate_joint_target.py` — calibration probe + `max_internal_within` helper (Task 3).
- `scripts/compare_budgets.py` — `run_two_lens` → `run_dual_target`; retire the lens types/rows (Task 5).
- `scripts/gen_multiblock_example.py`, `scripts/gen_method_comparison.py` — call the dual-target path with config constants (Task 5).
- `scripts/gen_example_readme.py` + `examples/method-comparison/README.md` — joint-target outcome story (Task 6).
- Tests: `tests/test_budget.py`, `tests/test_joint_target.py` (new), `tests/test_compare_budgets.py`, `tests/test_emit.py`.

---

### Task 1: Displacement as a fraction of homes

**Files:**
- Modify: `src/reblock/budget.py` (`displacement_curve`, ~L785-796)
- Modify: `src/reblock/emit.py` (`compare_report` labels ~L350-351, `disp_x` label; `displacement_table` writer ~L358-365)
- Test: `tests/test_budget.py`

**Interfaces:**
- Produces: `displacement_curve(...) -> Curve` (signature unchanged; `benefit` values now `Σcᵢ/n_buildings ∈ [0,1]`).

- [ ] **Step 1: Failing test — displacement_curve is a fraction**

Add to `tests/test_budget.py` (reuse that file's Block+roads fixtures; find one with `building_points`):

```python
def test_displacement_curve_is_home_fraction() -> None:
    block, roads = _block_with_buildings_and_roads()   # existing/adapted helper with building_points
    radii = building_radii(block.building_points, 3.0)
    curve = displacement_curve(block, roads, radii, corridor_m=3.0)
    n = len(block.building_points)
    assert all(0.0 <= b <= 1.0 for b in curve.benefit)          # fraction, not a count
    # terminal fraction == displacement(full roads)/n_buildings
    assert abs(curve.benefit[-1] - displacement(block.building_points, radii, roads, 3.0) / n) < 1e-9
```

Run: `pixi run pytest tests/test_budget.py::test_displacement_curve_is_home_fraction -v` → FAIL (values are counts > 1).

- [ ] **Step 2: Make `displacement_curve` fractional**

Replace `_disp` and the docstring's "y is Sum c_i" line:

```python
def displacement_curve(block: Block, roads: GeoDataFrame, radii: NDArray[np.float64], *,
                       corridor_m: float = 3.0, n_points: int = 20,
                       tol: float = STREET_TOL) -> Curve:
    """A Curve whose x is cumulative added road length (m) and whose y is the FRACTION of homes
    displaced, Σcᵢ / n_buildings (a rising COST in [0, 1]). Reuses the drainage-ordered _sweep.
    n_buildings = len(block.building_points) (buildings, not parcels)."""
    n = len(block.building_points)

    def _disp(prefix: GeoDataFrame | None) -> float:
        if prefix is None or len(prefix) == 0 or n == 0:
            return 0.0
        return displacement(block.building_points, radii, prefix, corridor_m) / n

    costs, vals = _sweep(block, roads, _disp, n_points, tol)
    return Curve(costs, vals)
```

- [ ] **Step 3: Update `compare_report` labels + consolidate `displacement_table`**

In `src/reblock/emit.py`:
- The x-label branch (~L350) and the `disp_x` benefit-metric x-label (~L351): the displacement metric plot's x-axis stays "added road length (m)"; its Y-label (`_METRIC_YLABELS["displacement"]`, L270) changes from `"buildings displaced (Σ disk-graze probability)"` to `"fraction of homes displaced"`. The two benefit metrics' x-label (currently `"buildings displaced (Σ disk-graze probability)"`) changes to `"fraction of homes displaced"`.
- `displacement_table.csv` writer (~L358-365): `r.curve.benefit[-1]` is now the terminal fraction, which equals `r.pct_displaced`. Drop the redundant column — header becomes `["method", "displaced_fraction", "n_blocks"]`, value `mean(fraction)` (keep sorting by it). Update the docstring line mentioning the columns.

- [ ] **Step 4: Run tests**

Run: `pixi run pytest tests/test_budget.py tests/test_emit.py -v` → PASS (update any test asserting the old count-valued displacement curve or the old `displacement_table` columns — migrate to the fraction).

- [ ] **Step 5: Commit** — `git commit -m "Displacement curves/tables/labels as fraction of homes (Σcᵢ/n_buildings)"`

---

### Task 2: `prefix_to_joint_target` + tests

**Files:**
- Modify: `src/reblock/budget.py` (add near `prefix_to_external_connectivity`, ~L744)
- Test: `tests/test_joint_target.py` (new)

**Interfaces:**
- Consumes: `Curve` (has `.cost`, `.benefit` lists), `truncate_to_length(block, roads, budget_m)`.
- Produces:
  - `@dataclass(frozen=True) class JointTargetOutcome: prefix: GeoDataFrame; reached: bool; reason: str; external: float; internal: float; displacement: float; sample_index: int; road_m: float`
  - `prefix_to_joint_target(block, roads, external, internal, displacement, *, i_min, e_min, d_max) -> JointTargetOutcome`

- [ ] **Step 1: Failing tests**

Create `tests/test_joint_target.py`:

```python
from dataclasses import dataclass
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon
from reblock.budget import Curve, prefix_to_joint_target
from reblock.contracts import Block

UTM = CRS.from_epsg(32734)


def _block() -> Block:
    boundary = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    parcels = gpd.GeoDataFrame({"parcel_id": [0]}, geometry=[boundary], crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (100, 0)])], crs=UTM)
    return Block(block_id="b", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _roads(n: int):   # n unit road segments so truncate_to_length has lengths to cut at
    return gpd.GeoDataFrame(geometry=[LineString([(i, 0), (i, 1)]) for i in range(n)], crs=UTM)


def _curves(ext, inte, disp):
    cost = list(range(len(ext)))
    return Curve(cost, ext), Curve(cost, inte), Curve(cost, disp)


def test_first_qualifying_index() -> None:
    ext, inte, disp = _curves([0.4, 0.6, 0.75, 0.8], [0.0, 0.1, 0.30, 0.35],
                              [0.05, 0.10, 0.20, 0.30])
    o = prefix_to_joint_target(_block(), _roads(4), ext, inte, disp,
                               i_min=0.25, e_min=0.70, d_max=0.45)
    assert o.reached and o.sample_index == 2                    # first index with ext>=.70 AND int>=.25


def test_touch_and_go_stops_at_first_cross() -> None:
    # internal crosses at i=1 then dips below; still stop at i=1
    ext, inte, disp = _curves([0.7, 0.72, 0.74], [0.30, 0.20, 0.31], [0.1, 0.2, 0.3])
    o = prefix_to_joint_target(_block(), _roads(3), ext, inte, disp,
                               i_min=0.25, e_min=0.70, d_max=0.45)
    assert o.reached and o.sample_index == 0


def test_killed_internal_below() -> None:
    ext, inte, disp = _curves([0.8, 0.9, 0.95], [0.0, 0.02, 0.03], [0.1, 0.2, 0.3])
    o = prefix_to_joint_target(_block(), _roads(3), ext, inte, disp,
                               i_min=0.25, e_min=0.70, d_max=0.45)
    assert not o.reached and o.reason == "internal_below"


def test_killed_over_budget() -> None:
    # floors met only at i=2 where displacement 0.5 > d_max 0.45
    ext, inte, disp = _curves([0.6, 0.68, 0.75], [0.1, 0.2, 0.30], [0.2, 0.4, 0.5])
    o = prefix_to_joint_target(_block(), _roads(3), ext, inte, disp,
                               i_min=0.25, e_min=0.70, d_max=0.45)
    assert not o.reached and o.reason == "over_budget"
```

Run: `pixi run pytest tests/test_joint_target.py -v` → FAIL (import error).

- [ ] **Step 2: Implement**

Add to `src/reblock/budget.py`:

```python
@dataclass(frozen=True)
class JointTargetOutcome:
    """The road prefix at which a method first meets the joint connectivity target, or the
    displacement-capped prefix if it never does. `reason`: "reached" | "internal_below" |
    "over_budget". external/internal/displacement are the values at the chosen sample; displacement
    is the home fraction. `sample_index` indexes the shared curve sample grid."""
    prefix: GeoDataFrame
    reached: bool
    reason: str
    external: float
    internal: float
    displacement: float
    sample_index: int
    road_m: float


def prefix_to_joint_target(block: Block, roads: GeoDataFrame, external: Curve, internal: Curve,
                           displacement: Curve, *, i_min: float, e_min: float,
                           d_max: float) -> JointTargetOutcome:
    """First sample index i with external[i] >= e_min AND internal[i] >= i_min AND
    displacement[i] <= d_max (displacement = home fraction). The three curves are index-aligned
    (same drainage-ordered _sweep grid; see test_curves_share_cost_samples), so this is a pure scan.
    Touch-and-go: the first qualifying index wins even if internal later dips. If no index qualifies
    within d_max, the outcome is killed at the last index with displacement <= d_max, with reason
    "internal_below" (internal never reached i_min anywhere within budget) or "over_budget" (external
    and internal both reachable, but only past d_max). The outcome prefix is `roads` truncated to that
    sample's cumulative road length."""
    n = len(external.cost)
    last_in_budget = -1
    ever_int_ok_in_budget = False
    for i in range(n):
        if displacement.benefit[i] > d_max:
            break
        last_in_budget = i
        if internal.benefit[i] >= i_min:
            ever_int_ok_in_budget = True
        if external.benefit[i] >= e_min and internal.benefit[i] >= i_min:
            prefix = truncate_to_length(block, roads, external.cost[i])
            return JointTargetOutcome(prefix, True, "reached", external.benefit[i],
                                      internal.benefit[i], displacement.benefit[i], i, external.cost[i])
    j = last_in_budget if last_in_budget >= 0 else 0
    reason = "over_budget" if ever_int_ok_in_budget else "internal_below"
    prefix = truncate_to_length(block, roads, external.cost[j]) if n else roads.iloc[:0]
    return JointTargetOutcome(prefix, False, reason,
                              external.benefit[j] if n else 0.0, internal.benefit[j] if n else 0.0,
                              displacement.benefit[j] if n else 0.0, j, external.cost[j] if n else 0.0)
```

(Confirm `dataclass`, `Curve`, `truncate_to_length`, `Block`, `GeoDataFrame` are importable in scope.)

- [ ] **Step 3: Run tests** — `pixi run pytest tests/test_joint_target.py -v` → PASS.

- [ ] **Step 4: Commit** — `git commit -m "Add prefix_to_joint_target: first prefix meeting the joint connectivity target within a displacement budget"`

---

### Task 3: Over-provisioning config + calibration probe (ends at human checkpoint)

**Files:**
- Create: `conf/joint_target.yaml`
- Create: `scripts/calibrate_joint_target.py`
- Test: `tests/test_joint_target.py` (append a helper test)

**Interfaces:**
- Produces: `max_internal_within(external, internal, displacement, *, e_min, d_max) -> float` (the max internal a method reaches at any sample with external ≥ e_min and displacement ≤ d_max; `-inf` if none) — the calibration primitive.

- [ ] **Step 1: `conf/joint_target.yaml`** — the universal constants (placeholders, filled at the checkpoint) + per-method over-provisioning knobs:

```yaml
# Universal joint-target thresholds (home fractions / benefit floats). Set once from
# scripts/calibrate_joint_target.py — see the plan's Task 3 checkpoint. PLACEHOLDERS until then.
i_min: null
e_min: 0.70
d_max: 0.45
# Over-provision each synthetic method past natural convergence so the joint-target scan can reach
# d_max displacement. Values are hydra overrides applied to all_methods.<name> for the probe + the
# example generators.
over_provision:
  greedy_arterial_repulsion: {max_roads: 40}
  clearance_looped: {base.max_roads: 3000, budget_frac: 0.30}
  euclidean_grid: {spacing: 120}
```

- [ ] **Step 2: Helper test**

Append to `tests/test_joint_target.py`:

```python
def test_max_internal_within() -> None:
    from reblock.budget import Curve
    from scripts.calibrate_joint_target import max_internal_within
    ext = Curve([0, 1, 2, 3], [0.5, 0.72, 0.8, 0.9])
    inte = Curve([0, 1, 2, 3], [0.1, 0.30, 0.45, 0.50])
    disp = Curve([0, 1, 2, 3], [0.1, 0.2, 0.40, 0.60])
    # samples with ext>=.70 and disp<=.45: i=1 (int .30), i=2 (int .45); i=3 disp .60 excluded
    assert max_internal_within(ext, inte, disp, e_min=0.70, d_max=0.45) == 0.45
    assert max_internal_within(ext, inte, disp, e_min=0.99, d_max=0.45) == float("-inf")
```

- [ ] **Step 3: Implement the probe**

Create `scripts/calibrate_joint_target.py`: for each internal-capable method (arterial_repulsion, clearance_looped, euclidean_grid), over-provisioned per `conf/joint_target.yaml`, on each example region (the 6 multiblock regions + the method-comparison block), build the three aligned curves (`access_benefit`, `commute_ratio_benefit`, fractional `displacement_curve`) and:

```python
def max_internal_within(external, internal, displacement, *, e_min, d_max):
    """Max internal benefit at any sample with external >= e_min and displacement <= d_max; -inf if
    none. The largest internal a method can deliver while clearing the external floor within budget."""
    best = float("-inf")
    for i in range(len(external.cost)):
        if external.benefit[i] >= e_min and displacement.benefit[i] <= d_max:
            best = max(best, internal.benefit[i])
    return best
```

The `main()`: print a per-(method, region) table of `max_internal_within(...)` at `e_min=0.70, d_max=0.45`, then propose `i_min = min over regions of arterial's max_internal_within × 0.95` (the reference method's reliable floor), and, at that `i_min`, report which methods reach the joint target on which regions (survivors/killed). Region loading mirrors `scripts/gen_multiblock_example.py` (screen + region-build per metric/city) and `gen_method_comparison.py` (the pinned block). Self-log to stdout; no repo constants written by the script — the human bakes them.

- [ ] **Step 4: Run the helper test** — `pixi run pytest tests/test_joint_target.py::test_max_internal_within -v` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "Calibration probe + over-provisioning config for the joint target"`

- [ ] **CHECKPOINT (controller, not a subagent):** run `pixi run python -m scripts.calibrate_joint_target`, present the derived `(I_min, E_min, D_max)` + survivor/killed table to the human, get sign-off, then write the agreed constants into `conf/joint_target.yaml` (`i_min`, and adjust `e_min`/`d_max` if the human revises them) and commit. Only then dispatch Task 4.

---

### Task 4: Outcome-marker overlay in `compare_report`

**Files:**
- Modify: `src/reblock/emit.py` (`compare_report`, ~L293)
- Test: `tests/test_emit.py`

**Interfaces:**
- Consumes: a per-`(block_id, method, metric)` outcome sample index.
- Produces: `compare_report(results, out_dir, *, method_order, outcomes=None)` where `outcomes: dict[tuple[str, str], int] | None` maps `(block_id, method) -> sample_index` (the outcome index on the shared grid); when given, draw a star marker at that index on each of the three metric curves for that method. `None` (default) draws no markers — keeps the function usable without outcomes.

- [ ] **Step 1: Failing test** — add to `tests/test_emit.py` a test that calls `compare_report(results, tmp_path, method_order=..., outcomes={(block, method): 1})` and asserts the curve PNGs still render (smoke) and that passing `outcomes=None` behaves as before (no error). (Marker geometry isn't pixel-asserted; the test guards the new param + backward-compatible default.)

- [ ] **Step 2: Implement** — in the per-metric plotting loop, after `ax.plot(xs, mc.curve.benefit, ...)`, if `outcomes` has `(block_id, mc.method)`, look up `i*`, and `ax.plot([xs[i*]], [mc.curve.benefit[i*]], marker="*", markersize=16, color=colors[mc.method], zorder=6)`. Add `outcomes` to the signature (default `None`).

- [ ] **Step 3: Run** — `pixi run pytest tests/test_emit.py -v` → PASS.

- [ ] **Step 4: Commit** — `git commit -m "compare_report: optional outcome-marker overlay on the metric curves"`

---

### Task 5: `run_dual_target` — the outcome pipeline (retire the two lenses)

**Files:**
- Modify: `scripts/compare_budgets.py` (retire `run_two_lens`/`two_lens_rows`/`LensARow`/`LensBRow`; add `run_dual_target`)
- Modify: `src/reblock/budget.py` (retire `prefix_to_external_connectivity`)
- Modify: `scripts/gen_multiblock_example.py`, `scripts/gen_method_comparison.py` (call `run_dual_target`)
- Test: `tests/test_compare_budgets.py`

**Interfaces:**
- Consumes: `prefix_to_joint_target`, `JointTargetOutcome`, `access_benefit`, `commute_ratio_benefit`, `displacement_curve`, `cost_benefit_curve`, `compare_report(..., outcomes=...)`, `render_after`, `_displaced_points`, `KComplexityEval`, `frame_bbox`, the `conf/joint_target.yaml` constants.
- Produces: `run_dual_target(region, methods, out_dir, *, i_min, e_min, d_max, corridor_m=3.0, label=None) -> list[OutcomeRow]` where `@dataclass(frozen=True) class OutcomeRow: method: str; reached: bool; reason: str; road_m: float; external: float; internal: float; displacement: float`.

- [ ] **Step 1: Failing/updated smoke test** — adapt `tests/test_compare_budgets.py`'s `test_run_two_lens_writes_tables_and_renders` into `test_run_dual_target_writes_outcome_and_renders`: call `run_dual_target(region, {"dijkstra": DijkstraReblocker()}, tmp_path, i_min=0.0, e_min=0.0, d_max=1.0)`; assert `outcome.csv` exists, one `after_<method>.jpg` per method exists, `curve_*` + `displacement_*` PNGs exist, and the returned `OutcomeRow` has `reached is True` (trivial floors). Delete the assertions/tests bound to `lens_a_external.csv`/`lens_b_matched.csv`/`two_lens_rows`.

- [ ] **Step 2: Implement `run_dual_target`** — mirror the structure of the current `run_two_lens` (reblock each method once — over-provisioned config already applied by the caller; the region block scores every method), but replace the two-lens body with:
  1. Build the three curves per method (`external`, `internal`, `disp`) exactly as `run_two_lens` already does for the frontier (`cost_benefit_curve(..., access_benefit)`, `cost_benefit_curve(..., commute_ratio_benefit)`, `displacement_curve`).
  2. `outcome = prefix_to_joint_target(block, roads, external, internal, disp, i_min=i_min, e_min=e_min, d_max=d_max)` per method.
  3. Render ONE after-image per method at `outcome.prefix` via the existing `render_after` + `KComplexityEval` + `_displaced_points` + `frame_bbox` pattern; title suffix from `outcome.reason` when `not reached` (e.g. `— failed joint target (internal {int:.2f} < {i_min})` or `(displaced {disp:.0%} > {d_max:.0%})`). File `after_<method>.jpg`.
  4. Keep the per-method reblock GIF (`reblock_gif`) unchanged.
  5. `compare_report(curves, out_dir, method_order=..., outcomes={(label, name): outcome.sample_index for reached methods})`.
  6. Write `outcome.csv` (header: `method, reached, reason, road_m, external, internal, displacement`).
  7. Keep the depth-vs-road report call.
  Retire `prefix_to_external_connectivity`, `matched_budget` usage, `two_lens_rows`, `LensARow`, `LensBRow`, and the lens CSV writers.

- [ ] **Step 3: Wire the generators** — `gen_multiblock_example.py`: replace `run_two_lens(region, methods, 0.70, out, label=seed)` with loading `conf/joint_target.yaml`, applying `over_provision` overrides to the synthetic methods, and `run_dual_target(region, methods, out, i_min=..., e_min=..., d_max=..., label=seed)`. `gen_method_comparison.py`: replace its bespoke curve+render block with a `run_dual_target([block], methods, OUT, i_min=..., e_min=..., d_max=...)` call (single-block region), keeping its config compose + method set.

- [ ] **Step 4: Run** — `pixi run pytest tests/test_compare_budgets.py tests/test_joint_target.py tests/test_budget.py -v` then `pixi run pytest -q` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "run_dual_target: single joint-target outcome per method; retire the two-lens surface"`

---

### Task 6: README generators → joint-target outcome story

**Files:**
- Modify: `scripts/gen_example_readme.py` (retire the lens sections; use `outcome.csv`)
- Modify: `examples/method-comparison/README.md` (hand-written; rewrite to the outcome story) — done at Task 7 regeneration when numbers exist, but the structural rewrite lands here with placeholders keyed to `run.log`/`outcome.csv`.
- Test: `tests/test_gen_example_readme.py`

- [ ] **Step 1: Update `gen_example_readme.py`** — replace the "Matched road budget"/"Matched external-connectivity target" sections (reading `lens_a_external.csv`/`lens_b_matched.csv`, `after_*_matched.jpg`/`after_*_ext*.jpg`) with a single "Joint-target outcome" section: one after-image per method (`after_<method>.jpg`, via a simplified `_after_method`), and an outcome table read from `outcome.csv` (method, reached, reason, road_m, external, internal, displacement-fraction). Update the frontier-section prose to mention the star marks the outcome and the displacement axis is a home fraction.
- [ ] **Step 2: Update `tests/test_gen_example_readme.py`** — migrate assertions off the retired lens strings (`"Matched road budget"`, `after_*_matched.jpg`, `"Lens A"`) onto the outcome strings; keep the structural assertions (screen stats, frontier embeds, GIF row).
- [ ] **Step 3: Run** — `pixi run pytest tests/test_gen_example_readme.py -v` → PASS.
- [ ] **Step 4: Commit** — `git commit -m "README generator: joint-target outcome section (retire two-lens sections)"`

---

### Task 7: Regenerate all examples + verify (controller-run, after final review)

Not an SDD task — controller steps after the whole-branch review:
1. `pixi run bash scripts/regenerate_examples.sh` (all 6 multiblock + method-comparison, now via `run_dual_target`).
2. Confirm the survivor/killed pattern matches the probe (arterial survives; trees killed `internal_below`; `clearance_looped` per D_max). Inspect a couple of outcome after-images + a curve with the star marker.
3. Rewrite `examples/method-comparison/README.md` prose to the outcome numbers from its `outcome.csv`/`run.log`.
4. Commit the regenerated `examples/**`.

## Self-Review

- **Spec coverage:** displacement fraction (T1) ✓; joint-target scan (T2) ✓; universal constants + probe + checkpoint (T3) ✓; outcome marker (T4) ✓; run_dual_target single-outcome + retire lenses (T5) ✓; READMEs (T6) ✓; regen (T7) ✓; osm shown as-is — handled in T5 (osm is a non-synthetic method: no over-provision, its curves + outcome computed and reported, reached/killed like any method, never truncated to a target since its outcome is just where its fixed curve lands).
- **Placeholder scan:** `i_min: null` in `conf/joint_target.yaml` is a deliberate checkpoint placeholder (filled by the human), not a code placeholder; every code step has concrete content.
- **Type consistency:** `JointTargetOutcome`/`OutcomeRow` fields, `prefix_to_joint_target` signature, `compare_report(..., outcomes=...)`, and `run_dual_target` signature are consistent across tasks; `displacement_curve` signature is unchanged (only values change).
- **Ambiguity resolved:** touch-and-go = first-qualifying-index (T2 test); kill reason precedence (`over_budget` iff internal was reachable in budget, else `internal_below`) is defined in code + tested.
