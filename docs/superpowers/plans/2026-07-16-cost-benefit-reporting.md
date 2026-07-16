# Cost-Benefit Reporting Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make added road length (m) the single cost x-axis, add an extent-aware disk-based displacement metric emitted as its own curve, and give the multiblock example per-method renders at a matched road budget.

**Architecture:** `budget.py` already orders roads by drainage and samples by cumulative length; we switch the *reported* x to metres, add `building_radii`/`displacement`/`truncate_to_length`, and delete the `cost`-mode machinery. `emit.py`/`compare.py` gain a `metric="displacement"` curve + two CSVs and drop the density/displacement-cost branches. `render.py` shades building disks by their displacement fraction. Both flagship examples regenerate once at the end.

**Tech Stack:** Python, geopandas/shapely, scipy (cKDTree for nearest-neighbor), matplotlib, Hydra, pixi, pytest, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-07-16-cost-benefit-reporting-design.md` (read it — it carries the derivation and every locked decision).

## Global Constraints

- **migrate, never accommodate — DELETE, no shims:** remove the m/ha density axis, the `cost` config key + every `cost` parameter, `budget._cost_fn_for`, `budget._density`, `budget.displacement_count`, every `cost == "displacement"` branch, and the old `tradeoff_table_*.csv` + `curve_*_displacement.png` example artifacts. Nothing keeps the old path alive.
- **Displacement is a RISING COST**, never inverted to "preservation".
- **Road length is metres (m)**, not m/ha density.
- **Disk displacement:** `cᵢ = max(0, 1 − dᵢ/rᵢ)`, `rᵢ = ½·NN_distᵢ` (uniform prior, **NO cap**). `dᵢ` = distance from point to `roads.buffer(corridor_m)` (0 if inside). `displacement = Σcᵢ`. Edge cases: `< 2` building points → `rᵢ = corridor_m`; coincident points (`rᵢ = 0`) → counts iff `dᵢ = 0`.
- **`corridor_m`** default stays 3.0 and remains the road half-width.
- **Green gates every task:** `pixi run pytest`, `pixi run ruff check`, `pixi run mypy --strict`. ruff forbids semicolons (E702), lines > 100 chars (E501), and `zip()` without `strict=` (B905).
- **Branch:** work on `osm-footpaths` (already carries the `osm_footpaths` rename). Fold everything in so the two flagship examples regenerate exactly once (final task).

## File Structure

- `src/reblock/budget.py` — cost axis → metres; `building_radii`, `displacement`, `displacement_curve`, `truncate_to_length`; delete `_density`, `_cost_fn_for`, `displacement_count`, all `cost` params.
- `src/reblock/emit.py` — `compare_report` (length x, displacement curve + 2 CSVs, drop `cost`/`tradeoff_table`); `pct_displaced` fractional; `_displaced_points` disk/`cᵢ`-aware; `_METRIC_YLABELS["displacement"]`.
- `src/reblock/compare.py` — drop `cost`; compute region radii once; add the displacement `MethodCurve`; log displacement terminals.
- `src/reblock/render.py` — `_point_disks` per-point radii + `cᵢ` colour ramp; `render_after` disk shading.
- `conf/compare_config.yaml` — remove the `cost` key.
- `scripts/render_methods_matched.py` — NEW one-off: per-method multiblock renders at the matched budget.
- `examples/method-comparison/`, `examples/multiblock/` — regenerate + README rewrites (final task).

Interface signatures every task shares (define once, reuse verbatim):

```python
# budget.py
def _length(prefix: GeoDataFrame) -> float: ...                       # cumulative metres
def cost_benefit_curve(block, roads, *, benefit_fn=access_benefit,
                       n_points=20, tol=STREET_TOL) -> Curve: ...   # no `cost`, no `corridor_m`
def efficiency_directness_curves(block, roads, *, n_points=20,
                                 tol=STREET_TOL) -> tuple[Curve, Curve]: ...  # no `cost`/`corridor_m`
def building_radii(building_points: GeoDataFrame, corridor_m: float) -> NDArray[np.float64]: ...
def displacement(building_points: GeoDataFrame, radii: NDArray[np.float64],
                 roads: GeoDataFrame, corridor_m: float) -> float: ...
def displacement_curve(block, roads, radii, *, corridor_m=3.0,
                       n_points=20, tol=STREET_TOL) -> Curve: ...     # x=metres, y=Σcᵢ
def truncate_to_length(block, roads: GeoDataFrame, budget_m: float,
                       tol: float = STREET_TOL) -> GeoDataFrame: ...  # drainage-ordered prefix
```

---

### Task 1: Road length (metres) as the single cost axis

**Files:**
- Modify: `src/reblock/budget.py` (`_sweep` ~772-805, `cost_benefit_curve` ~819-834, `efficiency_directness_curves` ~837-847; delete `_cost_fn_for` ~808-816)
- Test: `tests/test_budget.py`

**Interfaces:**
- Produces: `_length(prefix)->float`; `cost_benefit_curve`/`efficiency_directness_curves` **without** a `cost` param; `_sweep` **without** a `cost_fn` param (always reports `_length`).

- [ ] **Step 1: Write the failing test** — append to `tests/test_budget.py`:

```python
def test_cost_axis_is_cumulative_road_length_metres():
    from reblock.budget import cost_benefit_curve
    block, roads = _straight_block_with_two_roads()   # existing helper; else build a 100m block
    curve = cost_benefit_curve(block, roads)
    # x is cumulative added road length in METRES, non-decreasing, ending at total road length
    assert curve.cost[0] == 0.0
    assert curve.cost == sorted(curve.cost)
    assert abs(curve.cost[-1] - float(roads.geometry.length.sum())) < 1e-6


def test_cost_benefit_curve_has_no_cost_param():
    import inspect
    from reblock.budget import cost_benefit_curve
    assert "cost" not in inspect.signature(cost_benefit_curve).parameters
```

(If `_straight_block_with_two_roads` doesn't exist, reuse whatever block/road fixture `tests/test_budget.py` already builds for curve tests — grep the file first.)

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_budget.py -k "road_length or no_cost_param" -v`
Expected: FAIL (curve.cost still m/ha; `cost` param still present).

- [ ] **Step 3: Implement** — in `budget.py`:

Replace `_sweep`'s density cost with road-length metres and drop the `cost_fn` parameter:

```python
def _sweep(block: Block, roads: GeoDataFrame, value: Callable[[GeoDataFrame | None], V],
           n_points: int, tol: float) -> tuple[list[float], list[V]]:
    """Drainage-ordered cumulative-budget sweep: returns ([road_length_m(prefix)], [value(prefix)]).
    Order roads by drainage descending, then at n_points cumulative-length budgets evaluate `value`
    on the empty-prefix baseline and each growing prefix (skipping budgets that add no new road).
    The reported x-axis is cumulative added road length in metres."""
    def _length(prefix: GeoDataFrame) -> float:
        return float(prefix.geometry.length.sum())

    costs: list[float] = [_length(cast(GeoDataFrame, roads.iloc[:0]))]
    vals: list[V] = [value(cast(GeoDataFrame, roads.iloc[:0]))]
    if len(roads) == 0 or block.boundary.area == 0.0:
        return costs, vals
    drain = road_drainage(block, roads, tol=tol)
    order = sorted(range(len(roads)), key=lambda i: (-drain[i], i))
    ordered = roads.iloc[order].reset_index(drop=True)
    cum = ordered.geometry.length.to_numpy().cumsum()
    total = float(cum[-1])
    seen = 0
    for kk in range(1, n_points + 1):
        m = int((cum <= (kk / n_points) * total + 1e-9).sum())
        if m <= seen:
            continue
        seen = m
        costs.append(_length(ordered.iloc[:m]))
        vals.append(value(ordered.iloc[:m]))
    return costs, vals
```

Delete `_cost_fn_for` entirely. Drop the `cost` param + `cost_fn` call from both curve functions:

```python
def cost_benefit_curve(block: Block, roads: GeoDataFrame, *,
                       benefit_fn: BenefitFactory = access_benefit,
                       n_points: int = 20, tol: float = STREET_TOL) -> Curve:
    """Order roads by drainage descending, then at n_points cumulative-length budgets score
    benefit_fn's benefit vs the no-roads baseline. The x-axis is cumulative added road length (m).
    `corridor_m` is gone: it only fed the deleted displacement cost-mode; benefit is corridor-free."""
    costs, benefit = _sweep(block, roads, benefit_fn(block, roads, tol=tol), n_points, tol)
    return Curve(costs, benefit)


def efficiency_directness_curves(block: Block, roads: GeoDataFrame, *,
                                 n_points: int = 20, tol: float = STREET_TOL) -> tuple[Curve, Curve]:
    """ONE sampled shortest-path sweep yielding both E and directness curves (x = road length, m)."""
    f = _efficiency_factory(block, roads, tol)
    costs, pairs = _sweep(block, roads, f, n_points, tol)
    return Curve(costs, [p[0] for p in pairs]), Curve(costs, [p[1] for p in pairs])
```

Update `Curve.cost`'s comment to "cumulative added road length (m)". (Leave `auc` alone; it's unused by the frontier path and out of scope — but if `mypy`/`ruff` flags it as dead, that's a separate cleanup, not this task.)

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run pytest tests/test_budget.py -v && pixi run ruff check src/reblock/budget.py && pixi run mypy --strict src/reblock/budget.py`
Expected: PASS / clean. (Callers in `compare.py` still pass `cost=` and `corridor_m=` to these two functions; that breaks `compare.py` — Task 4 fixes it. If any *test* imports `_cost_fn_for`/`_density`, update it here.)

- [ ] **Step 5: Commit**

```bash
git add src/reblock/budget.py tests/test_budget.py
git commit -m "refactor(budget): road length (m) is the single cost axis; drop density + cost mode"
```

---

### Task 2: Extent-aware disk displacement metric

**Files:**
- Modify: `src/reblock/budget.py` (delete `displacement_count` ~37-44; add `building_radii`, `displacement`)
- Test: `tests/test_budget.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `building_radii(building_points, corridor_m)->NDArray[float64]`, `displacement(building_points, radii, roads, corridor_m)->float`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_budget.py`:

```python
def test_building_radii_are_half_nearest_neighbor():
    import geopandas as gpd
    from shapely.geometry import Point
    from reblock.budget import building_radii
    # three collinear points 10 m apart -> NN dist 10 for the ends, 10 for the middle -> r = 5
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(10, 0), Point(30, 0)], crs="EPSG:32734")
    r = building_radii(pts, corridor_m=3.0)
    assert list(r) == [5.0, 5.0, 10.0]      # 3rd point's NN is the 2nd, 20 m away -> r = 10


def test_building_radii_fallback_when_fewer_than_two_points():
    import geopandas as gpd
    from shapely.geometry import Point
    from reblock.budget import building_radii
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:32734")
    assert list(building_radii(pts, corridor_m=3.0)) == [3.0]     # fallback = corridor_m


def test_displacement_is_linear_ramp_in_distance_to_corridor():
    import geopandas as gpd, numpy as np
    from shapely.geometry import Point, LineString
    from reblock.budget import displacement
    crs = "EPSG:32734"
    # one road along y=0; corridor_m=1 -> corridor is the strip |y|<=1
    roads = gpd.GeoDataFrame(geometry=[LineString([(-50, 0), (50, 0)])], crs=crs)
    # point A on the corridor edge-ish (y=1 -> d=0 -> c=1); B at y=3 with r=4 -> d=2 -> c=0.5;
    # C at y=10 with r=4 -> d=9 -> c=0 (far)
    pts = gpd.GeoDataFrame(geometry=[Point(0, 1), Point(0, 3), Point(0, 10)], crs=crs)
    radii = np.array([4.0, 4.0, 4.0])
    # d_A = dist(A, strip|y|<=1) = 0 ; d_B = 3-1 = 2 ; d_C = 10-1 = 9
    got = displacement(pts, radii, roads, corridor_m=1.0)
    assert abs(got - (1.0 + 0.5 + 0.0)) < 1e-6


def test_displacement_zero_without_roads_or_points():
    import geopandas as gpd, numpy as np
    from shapely.geometry import Point
    from reblock.budget import displacement
    crs = "EPSG:32734"
    empty = gpd.GeoDataFrame(geometry=[], crs=crs)
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=crs)
    assert displacement(pts, np.array([3.0]), empty, 1.0) == 0.0
    assert displacement(empty, np.array([]), empty, 1.0) == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_budget.py -k "radii or displacement" -v`
Expected: FAIL (`building_radii`/`displacement` not defined).

- [ ] **Step 3: Implement** — in `budget.py` (add `from scipy.spatial import cKDTree` and `import numpy as np` / `from numpy.typing import NDArray` if not already imported; grep the header first). Delete `displacement_count`. Add:

```python
def building_radii(building_points: GeoDataFrame, corridor_m: float) -> NDArray[np.float64]:
    """Per-building disk radius = HALF the nearest-neighbor distance among the building points (the
    fair, non-overlapping 'as big as possible' footprint bound). Fewer than 2 points -> no neighbor,
    so fall back to `corridor_m`. Coincident points get radius 0 (handled by `displacement`)."""
    n = len(building_points)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n < 2:
        return np.full(n, float(corridor_m), dtype=np.float64)
    xy = np.column_stack([building_points.geometry.x.to_numpy(),
                          building_points.geometry.y.to_numpy()])
    dist, _ = cKDTree(xy).query(xy, k=2)     # k=2: self (0) + nearest other
    return (dist[:, 1] * 0.5).astype(np.float64)


def displacement(building_points: GeoDataFrame, radii: NDArray[np.float64],
                 roads: GeoDataFrame, corridor_m: float) -> float:
    """Extent-aware expected homes displaced: each building is a disk (radius `radii[i]`); its
    contribution is the probability the road corridor grazes it under a uniform size prior,
    c_i = max(0, 1 - d_i/r_i), d_i = distance from the point to roads.buffer(corridor_m). r_i = 0
    (coincident points) counts iff d_i = 0. Returns Sum c_i; 0 with no roads or no points."""
    n = len(building_points)
    if n == 0 or roads is None or len(roads) == 0:
        return 0.0
    corridor = roads.geometry.buffer(corridor_m).union_all()
    d = building_points.geometry.distance(corridor).to_numpy()
    r = np.asarray(radii, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(r > 0.0, 1.0 - d / r, np.where(d <= 0.0, 1.0, 0.0))
    return float(np.clip(c, 0.0, 1.0).sum())
```

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run pytest tests/test_budget.py -k "radii or displacement" -v && pixi run ruff check src/reblock/budget.py && pixi run mypy --strict src/reblock/budget.py`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/reblock/budget.py tests/test_budget.py
git commit -m "feat(budget): extent-aware disk displacement (c=max(0,1-d/r), r=NN/2)"
```

---

### Task 3: `truncate_to_length` + `displacement_curve`

**Files:**
- Modify: `src/reblock/budget.py` (add both after `_sweep`)
- Test: `tests/test_budget.py`

**Interfaces:**
- Consumes: `_sweep`, `road_drainage` (Task 1); `displacement` (Task 2).
- Produces: `truncate_to_length(block, roads, budget_m, tol)->GeoDataFrame`; `displacement_curve(block, roads, radii, *, corridor_m, n_points, tol)->Curve`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_budget.py`:

```python
def test_truncate_to_length_keeps_drainage_prefix():
    from reblock.budget import truncate_to_length
    block, roads = _straight_block_with_two_roads()   # or the shared curve fixture
    total = float(roads.geometry.length.sum())
    assert float(truncate_to_length(block, roads, total).geometry.length.sum()) == total
    assert len(truncate_to_length(block, roads, 0.0)) == 0
    half = truncate_to_length(block, roads, total / 2.0)
    assert 0.0 < float(half.geometry.length.sum()) <= total / 2.0 + 1e-6


def test_displacement_curve_is_monotonic_and_ends_at_full():
    import numpy as np
    from reblock.budget import displacement, displacement_curve
    block, roads = _straight_block_with_two_roads()
    radii = np.full(len(block.building_points), 3.0)
    curve = displacement_curve(block, roads, radii, corridor_m=3.0)
    assert curve.cost[0] == 0.0 and curve.benefit[0] == 0.0
    assert curve.benefit == sorted(curve.benefit)     # non-decreasing displacement
    assert abs(curve.benefit[-1]
               - displacement(block.building_points, radii, roads, 3.0)) < 1e-6
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_budget.py -k "truncate or displacement_curve" -v`
Expected: FAIL (not defined).

- [ ] **Step 3: Implement** — in `budget.py`:

```python
def truncate_to_length(block: Block, roads: GeoDataFrame, budget_m: float,
                       tol: float = STREET_TOL) -> GeoDataFrame:
    """The drainage-ordered prefix of `roads` whose cumulative length <= `budget_m` (the same order
    _sweep uses). Empty for budget_m <= 0; all roads for budget_m >= total."""
    if len(roads) == 0 or budget_m <= 0.0:
        return cast(GeoDataFrame, roads.iloc[:0])
    drain = road_drainage(block, roads, tol=tol)
    order = sorted(range(len(roads)), key=lambda i: (-drain[i], i))
    ordered = roads.iloc[order].reset_index(drop=True)
    cum = ordered.geometry.length.to_numpy().cumsum()
    m = int((cum <= budget_m + 1e-9).sum())
    return cast(GeoDataFrame, ordered.iloc[:m])


def displacement_curve(block: Block, roads: GeoDataFrame, radii: NDArray[np.float64], *,
                       corridor_m: float = 3.0, n_points: int = 20,
                       tol: float = STREET_TOL) -> Curve:
    """A Curve whose x is cumulative added road length (m) and whose y is Sum c_i displacement (a
    rising COST). Reuses the drainage-ordered _sweep with displacement as the value."""
    def _disp(prefix: GeoDataFrame | None) -> float:
        if prefix is None or len(prefix) == 0:
            return 0.0
        return displacement(block.building_points, radii, cast(GeoDataFrame, prefix), corridor_m)

    costs, vals = _sweep(block, roads, _disp, n_points, tol)
    return Curve(costs, vals)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run pytest tests/test_budget.py -k "truncate or displacement_curve" -v && pixi run ruff check src/reblock/budget.py && pixi run mypy --strict src/reblock/budget.py`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/reblock/budget.py tests/test_budget.py
git commit -m "feat(budget): truncate_to_length + displacement_curve"
```

---

### Task 4: Displacement curve + compare/emit/config wiring; delete cost mode

**Files:**
- Modify: `src/reblock/compare.py` (drop `cost`; region radii; displacement `MethodCurve`; terminal logging), `src/reblock/emit.py` (`compare_report`, `pct_displaced`, `_METRIC_YLABELS`), `conf/compare_config.yaml` (remove `cost`)
- Test: `tests/test_emit.py`, `tests/test_compare.py` (if present)

**Interfaces:**
- Consumes: `building_radii`, `displacement`, `displacement_curve` (Tasks 2-3); `cost_benefit_curve`/`efficiency_directness_curves` without `cost` (Task 1).
- Produces: results now include `MethodCurve(metric="displacement", curve=<x=length,y=Σcᵢ>)`; `compare_report(results, out_dir, *, method_order)` (no `cost`) emitting `frontier_{metric}.csv` (col `road_length_m`) for the four benefit metrics, `displacement_{block}.png`, `displacement_vs_length.csv`, `displacement_table.csv`.

- [ ] **Step 1: Write the failing test** — add to `tests/test_emit.py`:

```python
def test_compare_report_emits_length_frontier_and_displacement_artifacts(tmp_path):
    from reblock.emit import compare_report
    from reblock.compare import MethodCurve
    from reblock.budget import Curve
    curves = [
        MethodCurve("clearance", "B", "access", Curve([0.0, 100.0], [0.0, 0.8]), 0.1, 0.2),
        MethodCurve("clearance", "B", "displacement", Curve([0.0, 100.0], [0.0, 42.0]), 0.1, 0.2),
    ]
    compare_report(curves, tmp_path, method_order=["clearance"])
    assert (tmp_path / "frontier_access.csv").exists()
    assert "road_length_m" in (tmp_path / "frontier_access.csv").read_text()
    assert (tmp_path / "displacement_vs_length.csv").exists()
    assert (tmp_path / "displacement_table.csv").exists()
    assert (tmp_path / "displacement_B.png").exists()
    # migrated away:
    assert not list(tmp_path.glob("tradeoff_table_*.csv"))
    assert "road_density_m_per_ha" not in (tmp_path / "frontier_access.csv").read_text()


def test_compare_report_has_no_cost_param():
    import inspect
    from reblock.emit import compare_report
    assert "cost" not in inspect.signature(compare_report).parameters
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_emit.py -k "displacement_artifacts or no_cost_param" -v`
Expected: FAIL.

- [ ] **Step 3: Implement.**

**`conf/compare_config.yaml`:** delete the `cost: length` line and its comment block (the `# Cost axis ...` paragraph). Keep `corridor_m`.

**`emit.py`:**
- Add `"displacement": "buildings displaced (Σ disk-graze probability)"` to `_METRIC_YLABELS` (grep for the dict).
- `pct_displaced(roads, corridor_m, building_points, radii)` — new `radii` param; return `displacement(building_points, radii, roads, corridor_m) / n`:

```python
def pct_displaced(roads: gpd.GeoDataFrame | None, corridor_m: float,
                  building_points: gpd.GeoDataFrame,
                  radii: "NDArray[np.float64]") -> float:
    """Fraction of buildings-equivalent displaced: Σcᵢ / n_buildings (see budget.displacement)."""
    from reblock.budget import displacement
    n = len(building_points)
    if roads is None or len(roads) == 0 or n == 0:
        return 0.0
    return displacement(building_points, radii, roads, corridor_m) / n
```

- `compare_report(results, out_dir, *, method_order)` — remove the `cost` param and the whole `if cost == "displacement": ... else: ...` split. Always write `frontier_{metric}.csv` with header `["method", "block", "road_length_m", "benefit"]` for the four benefit metrics; for `metric == "displacement"` write `displacement_vs_length.csv` (header `["method", "block", "road_length_m", "displacement"]`) and accumulate a terminal table `displacement_table.csv` (header `["method", "terminal_displacement", "pct_displaced", "n_blocks"]`, mean over blocks). Plot every metric (including displacement) with `curve.cost` (x, "added road length (m)") vs `curve.benefit` (y, `_METRIC_YLABELS[metric]`); write benefit plots as `curve_{metric}_{block}.png` and the displacement plot as `displacement_{block}.png`. Curve legend label: `f"{mc.method} ({int(mc.curve.cost[-1])} m)"`.

Concretely, the metric loop becomes (replacing lines ~283-329):

```python
    from collections import defaultdict
    disp_terminal: dict[str, list[tuple[float, float]]] = defaultdict(list)  # method -> [(disp, pct)]
    for metric, metric_results in by_metric.items():
        by_block: dict[str, list[MethodCurve]] = {}
        for r in metric_results:
            by_block.setdefault(r.block_id, []).append(r)
        if metric == "displacement":
            with (out_dir / "displacement_vs_length.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["method", "block", "road_length_m", "displacement"])
                for r in metric_results:
                    for c, b in zip(r.curve.cost, r.curve.benefit, strict=True):
                        w.writerow([r.method, r.block_id, f"{c:.4f}", f"{b:.4f}"])
                    disp_terminal[r.method].append((r.curve.benefit[-1], r.pct_displaced))
        else:
            with (out_dir / f"frontier_{metric}.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["method", "block", "road_length_m", "benefit"])
                for r in metric_results:
                    for c, b in zip(r.curve.cost, r.curve.benefit, strict=True):
                        w.writerow([r.method, r.block_id, f"{c:.4f}", f"{b:.4f}"])
        ylabel = _METRIC_YLABELS[metric]
        for block_id, curves in by_block.items():
            fig, ax = plt.subplots(figsize=(7, 5))
            for mc in curves:
                ax.plot(mc.curve.cost, mc.curve.benefit, marker="o",
                        label=f"{mc.method} ({int(mc.curve.cost[-1])} m)", color=colors[mc.method])
            ax.set_xlabel("added road length (m)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"cost-benefit ({metric}): {block_id}")
            ax.legend()
            stem = "displacement" if metric == "displacement" else f"curve_{metric}"
            save_render(fig, out_dir / f"{stem}_{block_id}.png")
            plt.close(fig)
    if disp_terminal:
        from statistics import mean
        with (out_dir / "displacement_table.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "terminal_displacement", "pct_displaced", "n_blocks"])
            for m, rows in sorted(disp_terminal.items(), key=lambda kv: -mean(d for d, _ in kv[1])):
                w.writerow([m, f"{mean(d for d, _ in rows):.1f}",
                            f"{mean(p for _, p in rows):.4f}", len(rows)])
```

**`compare.py`:**
- Delete `cost = str(cfg.get("cost", "length"))` in `compare()` and `main()`. Remove `cost=cost` from the `cost_benefit_curve`/`efficiency_directness_curves` calls.
- Compute radii once per region and thread to `pct_displaced` + the displacement curve. Inside the method loop, after computing `roads`:

```python
            radii = building_radii(block.building_points, corridor_m)
            pp = pct_paved(roads, corridor_m, block_area)
            pd_ = pct_displaced(roads, corridor_m, block.building_points, radii)
            access = cost_benefit_curve(block, roads, benefit_fn=access_benefit)
            eff, direct = efficiency_directness_curves(block, roads)
            resistance = cost_benefit_curve(block, roads, benefit_fn=resistance_benefit)
            disp = displacement_curve(block, roads, radii, corridor_m=corridor_m)
            raw.append((name, label, "access", access, pp, pd_))
            raw.append((name, label, "efficiency", eff, pp, pd_))
            raw.append((name, label, "directness", direct, pp, pd_))
            raw.append((name, label, "resistance", resistance, pp, pd_))
            raw.append((name, label, "displacement", disp, pp, pd_))
```

Add imports: `from reblock.budget import building_radii, displacement_curve` (plus the existing `cost_benefit_curve`, etc.).
- In `main()`, replace the `if cost == "displacement": ... else: ...` logging with: log the four benefit terminals as before (`benefit=%.3f at %.0f m (%.1f%% paved)`, cost now metres), and log each displacement-metric row as `%s %s: %.1f displaced (%.1f%% of homes)`.

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run pytest tests/test_emit.py tests/test_compare.py -v && pixi run ruff check src/reblock/emit.py src/reblock/compare.py && pixi run mypy --strict src/reblock/emit.py src/reblock/compare.py`
Expected: PASS / clean. Then the FULL suite: `pixi run pytest -q` (fixes any test still referencing `cost=`/`tradeoff_table`/`displacement_count`).

- [ ] **Step 5: Commit**

```bash
git add src/reblock/compare.py src/reblock/emit.py conf/compare_config.yaml tests/
git commit -m "feat(compare): displacement curve + table; road-length frontier; delete cost mode"
```

---

### Task 5: Renders shade building disks by displacement fraction

**Files:**
- Modify: `src/reblock/render.py` (`_point_disks` ~98-107, `_draw_heatmap` ~110-163, `render_after` ~183-213), `src/reblock/emit.py` (`_displaced_points` ~51-63)
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `building_radii`, `displacement`-style `cᵢ` (Task 2).
- Produces: `_point_disks(points, radius_m=None)` accepting a per-point `radius` column; `render_after` shading a `displaced` frame (carrying `c` + `radius` columns) grey→red by `c`; `_displaced_points(block, proposal)` returning building points with `c` and `radius` columns (no `cost` gate).

- [ ] **Step 1: Write the failing test** — add to `tests/test_render.py`:

```python
def test_displaced_points_carry_fraction_and_radius(tmp_path):
    # a proposal with roads over a couple of building points -> _displaced_points has c in (0,1]
    import geopandas as gpd
    from shapely.geometry import Point, LineString, Polygon
    from reblock.contracts import Block, Proposal
    from reblock.emit import _displaced_points
    crs = "EPSG:32734"
    boundary = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    parcels = gpd.GeoDataFrame({"parcel_id": [0]}, geometry=[boundary], crs=crs)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (20, 0)])], crs=crs)
    pts = gpd.GeoDataFrame(geometry=[Point(10, 10), Point(10, 12)], crs=crs)
    block = Block(block_id="b", crs=crs, boundary=boundary, parcels=parcels,
                  streets=streets, building_points=pts)
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 10), (20, 10)])], crs=crs)
    prop = Proposal(block_id="b", crs=crs, roads=roads, edges=None,
                    proposal_id="x", method="m", params={"corridor_m": 1.0}, block_identity=None)
    disp = _displaced_points(block, prop)
    assert "c" in disp.columns and "radius" in disp.columns
    assert (disp["c"] > 0).any() and (disp["c"] <= 1).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_render.py -k displaced_points -v`
Expected: FAIL (`_displaced_points` gates on `cost`, returns no `c`/`radius`).

- [ ] **Step 3: Implement.**

**`emit.py` `_displaced_points`** — drop the `cost` gate; return points with `c` (displacement fraction) and `radius` columns:

```python
def _displaced_points(block: Block, proposal: Proposal) -> gpd.GeoDataFrame:
    """`block.building_points` with a per-point displacement fraction `c` = max(0, 1 - d/r)
    (r = NN/2, see budget) and its disk `radius`, for the render to shade. Empty when there are no
    points or no proposed roads."""
    import numpy as np
    from reblock.budget import building_radii
    pts = block.building_points
    if pts.empty or proposal.roads is None or proposal.roads.empty:
        return cast(gpd.GeoDataFrame, pts.iloc[:0])
    corridor_m = cast(float, proposal.params.get("corridor_m", 3.0))
    radii = building_radii(pts, corridor_m)
    corridor = proposal.roads.geometry.buffer(corridor_m).union_all()
    d = pts.geometry.distance(corridor).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(radii > 0.0, 1.0 - d / radii, np.where(d <= 0.0, 1.0, 0.0))
    out = pts.copy()
    out["c"] = np.clip(c, 0.0, 1.0)
    out["radius"] = radii
    return cast(gpd.GeoDataFrame, out[out["c"] > 0.0])
```

**`render.py` `_point_disks`** — accept an optional per-point `radius` column:

```python
def _point_disks(points: gpd.GeoDataFrame, radius_m: float | None = None) -> gpd.GeoDataFrame:
    """Points as geographic-size disks. If a `radius` column is present, each disk uses it (the
    building-footprint disks); else all disks share `radius_m`. A `weight` column still scales the
    radius by sqrt(weight)."""
    if "radius" in points.columns:
        radii = points["radius"].to_numpy()
    elif "weight" in points.columns:
        radii = (radius_m or 0.0) * (points["weight"].to_numpy() ** 0.5)
    else:
        radii = radius_m or 0.0
    return gpd.GeoDataFrame(geometry=points.geometry.buffer(radii), crs=points.crs)
```

**`render.py` `_draw_heatmap` / `render_after`** — replace the fixed red `displaced_points` block with a grey→red shading by `c`. In `_draw_heatmap`, where `displaced_points` is drawn (~154-156):

```python
    if displaced_points is not None and not displaced_points.empty:
        disks = _point_disks(displaced_points)                 # uses the `radius` column
        disks["c"] = displaced_points["c"].to_numpy()
        disks.plot(ax=ax, column="c", cmap="Reds", vmin=0.0, vmax=1.0, zorder=5, linewidth=0)
```

(Keep `own_points` drawn as before with `_POINT_RADIUS_M`; only the displaced overlay changes.)

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run pytest tests/test_render.py -v && pixi run ruff check src/reblock/render.py src/reblock/emit.py && pixi run mypy --strict src/reblock/render.py src/reblock/emit.py`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/reblock/render.py src/reblock/emit.py tests/test_render.py
git commit -m "feat(render): shade building disks grey->red by displacement fraction"
```

---

### Task 6: Per-method multiblock renders at a matched budget

**Files:**
- Create: `scripts/render_methods_matched.py`
- Test: `tests/test_render.py` (unit-test the truncation+render helper it calls)

**Interfaces:**
- Consumes: `truncate_to_length` (Task 3); `region_reblock` (`reblock.region`); the kcomplexity eval; `render_after` + `_displaced_points` (Task 5); `build_regions` (`reblock.pipeline`).
- Produces: a runnable module `python -m scripts.render_methods_matched <out_dir> <method,...> <hydra overrides>` writing `after_{method}.jpg` for each method, every method's roads truncated to `min(total road length over the methods)`.

- [ ] **Step 1: Write the failing test** — add to `tests/test_render.py` a unit test of the budget-side truncation-to-matched-budget helper the script uses. Put the pure helper in `budget.py`:

```python
def test_matched_budget_is_min_total_over_methods():
    from reblock.budget import matched_budget
    lengths = {"a": 100.0, "b": 40.0, "c": 61.0}
    assert matched_budget(lengths) == 40.0
    assert matched_budget({}) == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_render.py -k matched_budget -v`
Expected: FAIL (`matched_budget` not defined).

- [ ] **Step 3: Implement.**

Add to `budget.py`:

```python
def matched_budget(total_length_by_method: dict[str, float]) -> float:
    """The common render budget: the smallest method's total road length (every method can reach
    it). 0.0 if empty."""
    return min(total_length_by_method.values()) if total_length_by_method else 0.0
```

Create `scripts/render_methods_matched.py` (module form; mirrors `scripts/fetch_desire_lines_snapshot.py`'s Hydra bootstrapping):

```python
"""One-off: render one after-heatmap per method for a region, every method's roads truncated to a
MATCHED added-road-length budget (the sparsest method's total) so the comparison is fair.

Run: pixi run python -m scripts.render_methods_matched <out_dir> <m1,m2,...> <hydra overrides>...
  e.g. examples/multiblock clearance,greedy_arterial_buildable,osm_footpaths \
       data=capetown_full region_builder=dense_cluster region_builder.max_buildings=3000 \
       block_ids=[[ZAF.9.3.1_1_5810]] all_methods.clearance.max_roads=3000 \
       all_methods.greedy_arterial_buildable.candidate_policy=fixed \
       +all_methods.greedy_arterial_buildable.max_anchors=64 \
       desire_source.snapshot=examples/multiblock/desire_lines_osm_5810.geojson
"""
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from reblock.budget import matched_budget, truncate_to_length
from reblock.eval.kcomplexity import ...      # the eval that yields access_after layers + vmax
from reblock.pipeline import build_regions
from reblock.region import region_reblock
from reblock.render import render_after, save_render, short_label, frame_bbox
# plus emit._displaced_points

def main() -> None:
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    method_names = sys.argv[2].split(",")
    overrides = ["max_blocks=1", *sys.argv[3:]]
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config", overrides=overrides)
    source, screen = instantiate(cfg.data), instantiate(cfg.screen)
    region_builder = instantiate(cfg.region_builder)
    groups = [[str(b) for b in g] for g in cfg.block_ids]
    region = build_regions(source, screen, region_builder, groups, 1)[0]
    methods = {n: instantiate(cfg.all_methods[n]) for n in method_names}
    # 1) reblock each method, collect roads + eval'd access-depth
    results = {n: region_reblock(region, m, []) for n, m in methods.items()}
    lengths = {n: float(r.proposal.roads.geometry.length.sum()) for n, r in results.items()}
    budget = matched_budget(lengths)
    # 2) per method: truncate to budget, RE-EVAL access-depth on the truncated roads, render
    for n, r in results.items():
        block = r.block
        roads_t = truncate_to_length(block, r.proposal.roads, budget)
        # rebuild a Proposal with the truncated roads, re-run the kcomplexity peel to get
        # access_after + vmax, then render_after with _displaced_points(block, truncated_proposal).
        ...  # (implementer: reuse the exact eval call region_reblock/run uses; frame=frame_bbox(block.parcels))
        save_render(fig, out_dir / f"after_{n}.jpg")
    print(f"rendered {len(results)} methods at matched budget {budget:.0f} m -> {out_dir}")

if __name__ == "__main__":
    main()
```

The implementer resolves the `...` by reading how `reblock.run` / `region_reblock` produce `access_after` + `vmax` for a proposal (the kcomplexity eval fields `access_before`/`access_after`), and reuses that exact call on the truncated proposal. Keep the render frame + context consistent with `emit._render_block_group` (`frame_bbox(block.parcels)`). This is a one-off generator, not imported by the package — no unit test beyond `matched_budget`; its output is eyeballed in Task 7.

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run pytest tests/test_render.py -k matched_budget -v && pixi run ruff check src/reblock/budget.py scripts/render_methods_matched.py && pixi run mypy --strict src/reblock/budget.py`
Expected: PASS / clean. (Do NOT run the full script here — that's compute-heavy and belongs to Task 7.)

- [ ] **Step 5: Commit**

```bash
git add src/reblock/budget.py scripts/render_methods_matched.py tests/test_render.py
git commit -m "feat: matched-budget per-method render script (scripts/render_methods_matched)"
```

---

### Task 7: Regenerate both flagship examples + rewrite READMEs

**Files:**
- Modify/replace: `examples/method-comparison/*`, `examples/multiblock/*` (curves, CSVs, renders, run.logs, READMEs)
- Delete: old `tradeoff_table_*.csv`, `curve_*_displacement.png`

**Interfaces:** consumes the finished `reblock.compare` (Tasks 1-5) + `scripts/render_methods_matched` (Task 6).

This task is compute-heavy (topology on the deep block; region-scale clearance/arterial). Run each compare to a fixed `hydra.run.dir` and copy artifacts, matching the existing example filenames (see how the current examples map: `curve_{metric}.png` is the length curve, `curve_{metric}_{block}.png` its suffixed copy; there is no per-block displacement mode anymore).

- [ ] **Step 1: method-comparison — one length run now emits benefit + displacement**

```bash
pixi run python -m reblock.compare data=capetown_full "block_ids=[[ZAF.9.3.1_1_40972]]" \
  methods='[topology,clearance,greedy_arterial_buildable,osm_footpaths]' max_blocks=1 \
  all_methods.greedy_arterial_buildable.max_roads=8 \
  desire_source.snapshot=examples/method-comparison/desire_lines_40972.geojson \
  hydra.run.dir=/tmp/cbr_mc
```

Copy into `examples/method-comparison/`: `frontier_{metric}.csv` (4), `curve_{metric}_ZAF.9.3.1_1_40972.png` → also copy to `curve_{metric}.png` (4×2), `displacement_ZAF.9.3.1_1_40972.png` → `displacement.png`, `displacement_vs_length.csv`, `displacement_table.csv`. Rebuild `run.log` from `/tmp/cbr_mc/compare.log` (single `### method-comparison` section now). **Delete** the old `tradeoff_table_*.csv` and `curve_*_displacement.png`. Re-render the per-method afters (shaded by cᵢ) via the existing per-method `reblock.run` render commands used before (unchanged — they now show cᵢ shading automatically).

- [ ] **Step 2: multiblock — §4 length run + the matched-budget per-method renders**

```bash
pixi run python -m reblock.compare data=capetown_full region_builder=dense_cluster \
  region_builder.max_buildings=3000 "block_ids=[[ZAF.9.3.1_1_5810]]" \
  methods='[clearance,greedy_arterial_buildable,osm_footpaths]' max_blocks=1 \
  all_methods.clearance.max_roads=3000 all_methods.greedy_arterial_buildable.candidate_policy=fixed \
  +all_methods.greedy_arterial_buildable.max_anchors=64 \
  desire_source.snapshot=examples/multiblock/desire_lines_5810.geojson hydra.run.dir=/tmp/cbr_mb

pixi run python -m scripts.render_methods_matched examples/multiblock \
  clearance,greedy_arterial_buildable,osm_footpaths \
  data=capetown_full region_builder=dense_cluster region_builder.max_buildings=3000 \
  "block_ids=[[ZAF.9.3.1_1_5810]]" all_methods.clearance.max_roads=3000 \
  all_methods.greedy_arterial_buildable.candidate_policy=fixed \
  +all_methods.greedy_arterial_buildable.max_anchors=64 \
  desire_source.snapshot=examples/multiblock/desire_lines_5810.geojson
```

Copy `compare_{metric}.png` (from `curve_{metric}_<regionlabel>.png`), `displacement.png` (from `displacement_<regionlabel>.png`), `frontier_{metric}.csv`, `displacement_vs_length.csv`, `displacement_table.csv`. The script writes `after_{method}.jpg` ×3 directly. Rebuild `run.log` preserving §3/§5/§6 (unchanged) and replacing §4/§4b with the single new §4 run (see how the osm_footpaths rename rebuilt it). **Delete** old `tradeoff_table_*.csv` and `compare_directness_displacement.png`.

- [ ] **Step 3: Rewrite both READMEs**

Update every `m/ha` → `m`; replace the displacement tables with the new fractional `Σcᵢ` figures from `displacement_table.csv`; add the displacement-vs-length plot (`displacement.png`) with a one-paragraph explanation of the disk model (radius = NN/2, cᵢ = P(corridor grazes the uncertain-size footprint), a rising cost); in the multiblock, add the §4 per-method render grid (`after_{method}.jpg`, note the matched budget). Drop the separate `cost=displacement` reproduce command (one length run now emits both). Verify all `![...](...)` image paths resolve.

- [ ] **Step 4: Verify**

Run: `pixi run pytest -q && pixi run ruff check && pixi run mypy --strict` (green), then eyeball each regenerated PNG's legend/axis reads "added road length (m)", the displacement plot rises, and the multiblock per-method renders show shaded disks. `grep -rl "m/ha\|dream_come_true\|tradeoff_table" examples/` returns nothing.

- [ ] **Step 5: Commit**

```bash
git add examples/
git commit -m "docs(examples): regenerate for road-length axis + disk displacement + matched renders"
```

---

## Notes for the executor

- Tasks 1-6 are code + unit tests (fast); Task 7 is the only compute-heavy one — leave it last and run its compares in the background.
- After Task 7, the whole branch (`osm_footpaths` rename + this feature) is ready to finish via `superpowers:finishing-a-development-branch`.
- If a subagent finds an existing `tests/test_budget.py` fixture for blocks/roads, reuse it rather than the sketched `_straight_block_with_two_roads`; grep first.
