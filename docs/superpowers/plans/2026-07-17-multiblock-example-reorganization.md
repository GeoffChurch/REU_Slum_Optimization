# Multiblock Example Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `examples/multiblock` into a two-lens (fixed-depth-target vs fixed-road-budget) method comparison with per-method timing, and recolor the screen choropleth by the squared depth proxy `nA/P²`.

**Architecture:** Two small pure additions to `src/reblock` (a squared-proxy color helper in `emit.py`; a `prefix_to_depth` budget-walker + a shared drainage-order helper in `budget.py`), one new example driver script (`scripts/compare_budgets.py`) that reblocks each method once, drives it to a depth target (Lens A) and to a matched road budget (Lens B), and emits both tables + renders — superseding and replacing `scripts/render_methods_matched.py`. The final task rewrites the README and regenerates all figures.

**Tech Stack:** Python 3, Hydra, geopandas/shapely, scipy/networkx, matplotlib, pytest, ruff, mypy --strict, pixi.

## Global Constraints

- **Migrate, never accommodate:** no dual `√`-vs-squared path (the squared color is the new default); `scripts/render_methods_matched.py` is deleted, its matched-budget render folded into the new driver. No legacy fallback branches.
- **`pixi run check` (ruff lint + mypy --strict + pytest) must stay green** at the end of every task.
- **ruff forbids:** semicolons (E702), lines >100 chars (E501), `zip()` without `strict=` (B905).
- **mypy --strict:** every function typed; `cast(GeoDataFrame, ...)` for GeoDataFrame slices (the codebase convention).
- **Depth proxy formula:** `depth² = n·A/P² = (building count) × (compactness)`, where compactness `A/P²` is Polsby-Popper up to `4π`. The screen SELECTOR (`screen/dense_compact.py`) is unchanged and still ranks by depth `√(nA)/P`; only the region_map CHOROPLETH color changes to the squared form. Squaring is monotone, so the flagged/ranked set is identical.
- **Access-depth is monotone non-increasing** as drainage-ordered roads are added (a larger street seed only shrinks BFS depths) — `prefix_to_depth` relies on this for its binary search; do NOT assume the internal-connectivity metric is monotone.
- **`osm_footpaths` is a fixed input** — it lands where it lands; if its full road set cannot reach depth `D`, Lens A reports it unreached with its floor depth (honest ✗), which is itself the informative result.
- **Drainage order is canonical:** roads are always walked in `road_drainage`-descending order (ties by original index), the exact order `_sweep`/`truncate_to_length` already use.
- **Commit trailers** on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
  ```
- **Reproducible-by-CLI:** the example runs from `capetown_full` + the committed OSM snapshot; no manual data steps beyond the auto-download.

**Spec:** `docs/superpowers/specs/2026-07-17-multiblock-example-reorganization-design.md`

---

## File Structure

- `src/reblock/emit.py` — add `_screen_proxy(...)`, switch `region_map`'s `proxy` column + title to the squared form. (Task 1)
- `src/reblock/budget.py` — add `_drainage_ordered(...)` (extracted from `_sweep`/`truncate_to_length`), `max_access_depth(...)`, `prefix_to_depth(...)`. (Task 2)
- `scripts/compare_budgets.py` — NEW two-lens driver (`LensARow`, `LensBRow`, `two_lens_rows`, `run_two_lens`, `main`). (Task 3)
- `scripts/render_methods_matched.py` — DELETED (folded into `compare_budgets.py`). (Task 3)
- `tests/test_emit.py`, `tests/test_budget.py`, `tests/test_compare_budgets.py` — tests. (Tasks 1–3)
- `examples/multiblock/README.md`, `examples/README.md`, `examples/multiblock/run.log` + figures/CSVs — rewrite + regenerate. (Task 4)

---

## Task 1: Squared screen coloring in `region_map`

**Files:**
- Modify: `src/reblock/emit.py` (add `_screen_proxy`; edit `region_map` lines ~170–206)
- Test: `tests/test_emit.py`

**Interfaces:**
- Produces: `_screen_proxy(building_count: NDArray[np.float64], area: NDArray[np.float64], perim: NDArray[np.float64]) -> NDArray[np.float64]` returning `n·A/P²` (nan where `perim <= 0`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_emit.py`:

```python
def test_screen_proxy_is_squared_depth_and_rank_preserving() -> None:
    # region_map colors by the SQUARED depth proxy n*A/P^2 (= building-count x compactness A/P^2),
    # which is monotone in the depth proxy sqrt(nA)/P the screen ranks by -- so the flagged/ranked
    # set is identical, only the color contrast sharpens. Non-positive perimeter -> nan (drawn as
    # missing, never a divide-by-zero).
    import numpy as np

    from reblock.emit import _screen_proxy
    n = np.array([100.0, 400.0, 50.0])
    a = np.array([1000.0, 4000.0, 500.0])
    p = np.array([100.0, 200.0, 80.0])
    proxy = _screen_proxy(n, a, p)
    assert np.allclose(proxy, (n * a) / (p ** 2))
    depth = np.sqrt(n * a) / p                       # the screen's depth proxy
    assert list(np.argsort(proxy)) == list(np.argsort(depth))   # squaring preserves rank
    assert np.isnan(_screen_proxy(np.array([1.0]), np.array([1.0]), np.array([0.0]))[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_emit.py::test_screen_proxy_is_squared_depth_and_rank_preserving -v`
Expected: FAIL with `ImportError: cannot import name '_screen_proxy'`

- [ ] **Step 3: Add `_screen_proxy` to `src/reblock/emit.py`**

Insert above `region_map` (after `flagged_map`, before `region_map`):

```python
def _screen_proxy(building_count: NDArray[np.float64], area: NDArray[np.float64],
                  perim: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-block depth-squared proxy `n·A/P²` = building-count × compactness (`A/P²`, the
    Polsby-Popper measure up to `4π`). The screen RANKS blocks by depth `√(nA)/P`; `region_map`
    COLORS by this squared form -- monotone in the depth proxy (so the flagged/ranked set is
    identical) but with far better slum-vs-formal contrast on the metro choropleth (the `√` form
    saturates most dense fabric at max colour). `perim <= 0` -> nan (drawn as `missing`, never a
    divide-by-zero)."""
    safe = np.where(perim > 0.0, perim, np.nan)
    return cast(NDArray[np.float64], (building_count * area) / (safe ** 2))
```

- [ ] **Step 4: Switch `region_map` to the squared proxy**

In `region_map`, replace the proxy block (currently):

```python
        utm = geoms.to_crs(geoms.estimate_utm_crs())
        area = geoms["block_area_m2"] if "block_area_m2" in geoms.columns else utm.geometry.area
        perim = utm.geometry.length
        geoms["proxy"] = np.sqrt(geoms["building_count"] * area) / perim.where(perim > 0)
        # Cap high (p99, not p97): the proxy is heavy-tailed, so a lower cap saturates most of the
        # metro at max colour; p99 keeps the deep fabric standing out against a paler background.
        vmax = float(geoms["proxy"].quantile(0.99)) or 1.0
```

with:

```python
        utm = geoms.to_crs(geoms.estimate_utm_crs())
        area = (geoms["block_area_m2"].to_numpy(dtype=float) if "block_area_m2" in geoms.columns
                else utm.geometry.area.to_numpy())
        perim = utm.geometry.length.to_numpy()
        geoms["proxy"] = _screen_proxy(geoms["building_count"].to_numpy(dtype=float), area, perim)
        # Cap high (p99, not p97): the squared proxy is heavy-tailed, so a lower cap saturates most
        # of the metro at max colour; p99 keeps the deep fabric standing out against a paler
        # background.
        vmax = float(geoms["proxy"].quantile(0.99)) or 1.0
```

- [ ] **Step 5: Update `region_map`'s docstring + screen title to the squared form**

In the `region_map` docstring, change `the city depth-proxy choropleth (sqrt(n*A)/P -- what the screen keys on to find deep fabric)` to `the city depth-proxy choropleth (n·A/P², the SQUARED depth proxy -- monotone in what the screen ranks by, sharper slum contrast)`.

Change the screen title line:

```python
    ax_s.set_title(f"depth proxy √(n·A)/P; {len(all_member_ids)} blocks reblocked")
```

to:

```python
    ax_s.set_title(f"depth² proxy n·A/P²; {len(all_member_ids)} blocks reblocked")
```

- [ ] **Step 6: Run the test + existing emit tests**

Run: `pixi run pytest tests/test_emit.py -v`
Expected: PASS (new test + all existing `region_map`/`render_results`/`compare_report` tests).

- [ ] **Step 7: Lint + type-check the file**

Run: `pixi run ruff check src/reblock/emit.py tests/test_emit.py && pixi run mypy --strict src/reblock/emit.py`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add src/reblock/emit.py tests/test_emit.py
git commit -m "feat: color the region_map screen choropleth by the squared depth proxy nA/P^2

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

## Task 2: `prefix_to_depth` + shared drainage-order helper in `budget.py`

**Files:**
- Modify: `src/reblock/budget.py` (`_sweep`, `truncate_to_length`; add `_drainage_ordered`, `max_access_depth`, `prefix_to_depth`)
- Test: `tests/test_budget.py`

**Interfaces:**
- Consumes: `road_drainage`, `parcel_access_layers`, `parcel_adjacency` (already imported in `budget.py`), `STREET_TOL`.
- Produces:
  - `_drainage_ordered(block: Block, roads: GeoDataFrame, tol: float) -> GeoDataFrame`
  - `max_access_depth(block: Block, roads: GeoDataFrame | None, *, tol: float = STREET_TOL, adj: list[set[int]] | None = None) -> int`
  - `prefix_to_depth(block: Block, roads: GeoDataFrame, target_depth: int, *, tol: float = STREET_TOL) -> tuple[GeoDataFrame, int]` — the minimal drainage-ordered prefix whose max access-depth ≤ `target_depth`, paired with that prefix's actual max depth; if unreachable, returns `(all roads in drainage order, floor depth)` with floor depth > `target_depth`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_budget.py` (near the other `truncate_to_length` fixtures):

```python
def _deep_column_block_with_two_roads() -> tuple[Block, gpd.GeoDataFrame]:
    # A 1-wide, 4-deep column of unit parcels fronting a street at y=0. With no roads the peel is
    # 1,2,3,4 (max depth 4). Road A runs up the right edge for the bottom half (touches the street
    # at (1,0)); it seeds parcels 0,1 as layer 1, so the peel becomes 1,1,2,3 (max depth 3). Road B
    # extends the right edge to the top; it only reaches the street THROUGH road A (touch-component),
    # so {A,B} seeds all four parcels as layer 1 (max depth 1). Drainage: A is the trunk (every
    # parcel routes through it), so A sorts before B.
    polys = [Polygon([(0, j), (1, j), (1, j + 1), (0, j + 1)]) for j in range(4)]
    parcels = gpd.GeoDataFrame({"parcel_id": [0, 1, 2, 3]}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs=UTM)
    block = Block(block_id="deep_col", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)
    road_a = LineString([(1, 0), (1, 2)])
    road_b = LineString([(1, 2), (1, 4)])
    roads = gpd.GeoDataFrame(geometry=[road_a, road_b], crs=UTM)
    return block, roads


def test_max_access_depth_matches_the_peel() -> None:
    from reblock.budget import max_access_depth
    block, roads = _deep_column_block_with_two_roads()
    assert max_access_depth(block, gpd.GeoDataFrame(geometry=[], crs=UTM)) == 4   # no roads
    assert max_access_depth(block, roads) == 1                                     # both roads


def test_prefix_to_depth_returns_minimal_prefix_that_reaches_target() -> None:
    from reblock.budget import prefix_to_depth
    block, roads = _deep_column_block_with_two_roads()
    prefix, reached = prefix_to_depth(block, roads, 3)
    assert reached == 3                        # road A alone brings max depth to 3
    assert len(prefix) == 1                    # the MINIMAL prefix, not both roads
    assert prefix.geometry.iloc[0].equals(roads.geometry.iloc[0])   # road A (the drainage trunk)


def test_prefix_to_depth_reaches_a_deeper_target_only_with_all_roads() -> None:
    from reblock.budget import prefix_to_depth
    block, roads = _deep_column_block_with_two_roads()
    prefix, reached = prefix_to_depth(block, roads, 1)
    assert reached == 1
    assert len(prefix) == 2                    # needs both roads to reach depth 1


def test_prefix_to_depth_reports_floor_when_target_unreachable() -> None:
    from reblock.budget import prefix_to_depth
    block, roads = _deep_column_block_with_two_roads()
    prefix, reached = prefix_to_depth(block, roads, 0)   # depth 0 is impossible (min is 1)
    assert reached == 1                        # the floor depth (> target), reported honestly
    assert len(prefix) == len(roads)           # best effort = all roads in drainage order
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run pytest tests/test_budget.py -k "prefix_to_depth or max_access_depth" -v`
Expected: FAIL with `ImportError: cannot import name 'max_access_depth'` / `'prefix_to_depth'`.

- [ ] **Step 3: Extract `_drainage_ordered` and route `_sweep`/`truncate_to_length` through it**

Add above `_sweep` in `src/reblock/budget.py`:

```python
def _drainage_ordered(block: Block, roads: GeoDataFrame, tol: float) -> GeoDataFrame:
    """`roads` reindexed in drainage-descending order (ties by original index), reset to a fresh
    RangeIndex -- the single canonical prefix order shared by `_sweep`, `truncate_to_length`, and
    `prefix_to_depth`, so every budget/prefix walk grows the road set in the same sequence. Callers
    guard `len(roads) == 0` before calling (an empty road set has no drainage to order)."""
    drain = road_drainage(block, roads, tol=tol)
    order = sorted(range(len(roads)), key=lambda i: (-drain[i], i))
    return cast(GeoDataFrame, roads.iloc[order].reset_index(drop=True))
```

In `_sweep`, replace:

```python
    drain = road_drainage(block, roads, tol=tol)
    order = sorted(range(len(roads)), key=lambda i: (-drain[i], i))
    ordered = roads.iloc[order].reset_index(drop=True)
```

with:

```python
    ordered = _drainage_ordered(block, roads, tol)
```

(The `if len(roads) == 0 or block.boundary.area == 0.0: return costs, vals` guard already sits ABOVE these lines — leave it in place.)

In `truncate_to_length`, replace:

```python
    if len(roads) == 0 or budget_m <= 0.0:
        return cast(GeoDataFrame, roads.iloc[:0])
    drain = road_drainage(block, roads, tol=tol)
    order = sorted(range(len(roads)), key=lambda i: (-drain[i], i))
    ordered = roads.iloc[order].reset_index(drop=True)
    cum = ordered.geometry.length.to_numpy().cumsum()
```

with:

```python
    if len(roads) == 0 or budget_m <= 0.0:
        return cast(GeoDataFrame, roads.iloc[:0])
    ordered = _drainage_ordered(block, roads, tol)
    cum = ordered.geometry.length.to_numpy().cumsum()
```

- [ ] **Step 4: Add `max_access_depth` and `prefix_to_depth`**

Add after `truncate_to_length` in `src/reblock/budget.py`:

```python
def max_access_depth(block: Block, roads: GeoDataFrame | None, *, tol: float = STREET_TOL,
                     adj: list[set[int]] | None = None) -> int:
    """The block's deepest BFS access-depth (`parcel_access_layers`) given `roads` -- 1 = every
    parcel fronts a street, higher = buried. `adj` (parcel adjacency) may be passed to avoid
    rebuilding it across repeated calls on the same block."""
    return int(parcel_access_layers(block, roads, tol=tol, adj=adj).max())


def prefix_to_depth(block: Block, roads: GeoDataFrame, target_depth: int, *,
                    tol: float = STREET_TOL) -> tuple[GeoDataFrame, int]:
    """The minimal drainage-ordered prefix of `roads` whose max BFS access-depth is
    <= `target_depth`, paired with that prefix's actual max depth. Access-depth is monotone
    non-increasing as drainage-ordered roads are added (a larger street seed only shrinks depths),
    so a binary search over the prefix length finds the smallest sufficient prefix in O(log R)
    peels. If even all `roads` cannot reach `target_depth`, returns (all roads in drainage order,
    floor depth) with floor depth > `target_depth` -- the caller reports that as unreached (an
    `osm_footpaths`-style fixed input that never reaches the deep interior). Empty `roads` returns
    (empty, the no-road peel's max depth)."""
    adj = parcel_adjacency(list(block.parcels.geometry), tol)
    if len(roads) == 0:
        empty = cast(GeoDataFrame, roads.iloc[:0])
        return empty, max_access_depth(block, empty, tol=tol, adj=adj)
    ordered = _drainage_ordered(block, roads, tol)

    def depth_at(m: int) -> int:
        return max_access_depth(block, cast(GeoDataFrame, ordered.iloc[:m]), tol=tol, adj=adj)

    n = len(ordered)
    full_depth = depth_at(n)
    if full_depth > target_depth:                 # unreachable: best effort is all roads
        return ordered, full_depth
    lo, hi = 0, n                                 # smallest m with depth_at(m) <= target_depth
    while lo < hi:
        mid = (lo + hi) // 2
        if depth_at(mid) <= target_depth:
            hi = mid
        else:
            lo = mid + 1
    return cast(GeoDataFrame, ordered.iloc[:lo].reset_index(drop=True)), depth_at(lo)
```

- [ ] **Step 5: Run the new tests + the existing budget suite**

Run: `pixi run pytest tests/test_budget.py -v`
Expected: PASS — the new tests, and every existing `truncate_to_length`/`_sweep`/`cost_benefit_curve`/`displacement_curve` test still green (the `_drainage_ordered` extraction is behavior-preserving).

- [ ] **Step 6: Lint + type-check**

Run: `pixi run ruff check src/reblock/budget.py tests/test_budget.py && pixi run mypy --strict src/reblock/budget.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/reblock/budget.py tests/test_budget.py
git commit -m "feat: prefix_to_depth budget-walker + shared _drainage_ordered helper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

## Task 3: Two-lens driver `scripts/compare_budgets.py` (replaces `render_methods_matched.py`)

**Files:**
- Create: `scripts/compare_budgets.py`
- Delete: `scripts/render_methods_matched.py`
- Test: `tests/test_compare_budgets.py`

**Interfaces:**
- Consumes: `prefix_to_depth`, `truncate_to_length`, `matched_budget`, `building_radii`, `displacement`, `access_benefit`, `commute_ratio` (budget.py); `pct_displaced`, `_displaced_points` (emit.py); `KComplexityEval` (eval.kcomplexity); `region_reblock` (region.py); `render_after`, `frame_bbox`, `save_render` (render.py); `build_regions` (pipeline.py).
- Produces:
  - `@dataclass(frozen=True) LensARow{method: str, reached: bool, reached_depth: int, road_length_m: float, displacement: float, pct_displaced: float, propose_seconds: float}`
  - `@dataclass(frozen=True) LensBRow{method: str, budget_m: float, external_connectivity: float, internal_connectivity: float, displacement: float, pct_displaced: float}`
  - `two_lens_rows(block: Block, roads_by_method: dict[str, GeoDataFrame], propose_seconds: dict[str, float], target_depth: int, budget_m: float, *, corridor_m: float = 3.0) -> tuple[list[LensARow], list[LensBRow]]` — the pure table logic (no I/O, no rendering, deterministic).
  - `run_two_lens(region: list[Block], methods: dict[str, Method], target_depth: int, out_dir: Path, *, corridor_m: float = 3.0) -> tuple[list[LensARow], list[LensBRow]]` — reblocks each method once (timed), calls `two_lens_rows`, writes `lens_a_depth.csv` + `lens_b_matched.csv`, renders `after_{method}_depth{D}.jpg` (Lens A) + `after_{method}_matched.jpg` (Lens B).
  - `main() -> None` — the `python -m scripts.compare_budgets <out_dir> <target_depth> <m1,m2,...> <hydra overrides>...` edge.

- [ ] **Step 1: Write the failing test**

Create `tests/test_compare_budgets.py`:

```python
from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.contracts import Block

UTM = CRS.from_epsg(32643)


def _deep_column_block_with_two_roads() -> tuple[Block, gpd.GeoDataFrame]:
    # Same fixture shape as tests/test_budget.py: a 4-deep column fronting a street at y=0. No roads
    # -> max depth 4; road A (right edge, bottom half) -> depth 3; {A,B} -> depth 1. One building
    # point per parcel so displacement/benefit are exercised.
    from shapely.geometry import Point
    polys = [Polygon([(0, j), (1, j), (1, j + 1), (0, j + 1)]) for j in range(4)]
    parcels = gpd.GeoDataFrame({"parcel_id": [0, 1, 2, 3]}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs=UTM)
    points = gpd.GeoDataFrame(geometry=[Point(0.5, j + 0.5) for j in range(4)], crs=UTM)
    block = Block(block_id="deep_col", crs=UTM, boundary=boundary, parcels=parcels,
                  streets=streets, building_points=points)
    roads = gpd.GeoDataFrame(geometry=[LineString([(1, 0), (1, 2)]), LineString([(1, 2), (1, 4)])],
                             crs=UTM)
    return block, roads


def test_two_lens_rows_reports_reached_target_and_matched_budget() -> None:
    from reblock.budget import truncate_to_length
    from scripts.compare_budgets import two_lens_rows

    block, roads = _deep_column_block_with_two_roads()
    budget_a = float(roads.geometry.iloc[0].length)          # room for road A only
    lens_a, lens_b = two_lens_rows(block, {"m": roads}, {"m": 0.5}, target_depth=3,
                                   budget_m=budget_a, corridor_m=3.0)
    assert len(lens_a) == 1 and len(lens_b) == 1
    (a,) = lens_a
    assert a.method == "m" and a.reached is True and a.reached_depth == 3
    assert a.propose_seconds == 0.5                          # timing passed through, not remeasured
    assert a.road_length_m > 0.0 and a.displacement >= 0.0
    (b,) = lens_b
    assert b.budget_m == budget_a
    assert 0.0 <= b.external_connectivity <= 1.0
    assert b.internal_connectivity >= 0.0
    # Lens B scores the matched-budget prefix (road A only), matching truncate_to_length.
    assert len(truncate_to_length(block, roads, budget_a)) == 1


def test_two_lens_rows_reports_floor_when_depth_target_unreachable() -> None:
    from scripts.compare_budgets import two_lens_rows

    block, roads = _deep_column_block_with_two_roads()
    lens_a, _ = two_lens_rows(block, {"m": roads}, {"m": 0.1}, target_depth=0,
                              budget_m=1.0, corridor_m=3.0)
    (a,) = lens_a
    assert a.reached is False and a.reached_depth == 1       # floor depth (> target 0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_compare_budgets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.compare_budgets'`.

- [ ] **Step 3: Create `scripts/compare_budgets.py` — the pure table logic first**

```python
"""Two-lens method comparison for the multiblock example (replaces render_methods_matched.py).

Reblocks each method once over the region (timed), then reports it under two budgets:

  Lens A -- fixed OUTCOME (depth target D): the drainage-ordered road prefix that first brings every
    parcel within access-depth <= D (`budget.prefix_to_depth`); reports the road length, displacement
    and wall-clock propose time it took. A fixed input that never reaches D (osm_footpaths) is
    reported unreached with its floor depth.

  Lens B -- fixed COST (matched road budget): every method truncated to the sparsest method's total
    added road length (`budget.matched_budget` + `truncate_to_length`); reports benefit on both axes
    (external + internal connectivity) + displacement.

Both lenses render one after-heatmap per method (Lens A at the depth-D prefix, Lens B at the matched
budget), re-scoring access-depth on the truncated roads exactly as render_methods_matched did.

Run (module form -- mirrors scripts/render_methods_matched.py's Hydra bootstrapping):
  pixi run python -m scripts.compare_budgets <out_dir> <target_depth> <m1,m2,...> <hydra override>...

  e.g. examples/multiblock 3 clearance,greedy_arterial_buildable,osm_footpaths \
       data=capetown_full region_builder=dense_cluster region_builder.max_buildings=3000 \
       block_ids=[[ZAF.9.3.1_1_5810]] all_methods.clearance.max_roads=3000 \
       all_methods.clearance.depth_target=3 \
       all_methods.greedy_arterial_buildable.candidate_policy=fixed \
       +all_methods.greedy_arterial_buildable.max_anchors=64 \
       desire_source.snapshot=examples/multiblock/desire_lines_5810.geojson
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from reblock.budget import (
    access_benefit,
    building_radii,
    commute_ratio,
    displacement,
    matched_budget,
    prefix_to_depth,
    truncate_to_length,
)
from reblock.contracts import Block, Method, Proposal, Screen, Source
from reblock.emit import _displaced_points, pct_displaced
from reblock.eval.kcomplexity import KComplexityEval
from reblock.pipeline import build_regions
from reblock.region import RegionBuilder, region_reblock
from reblock.render import frame_bbox, render_after, save_render


@dataclass(frozen=True)
class LensARow:
    method: str
    reached: bool           # did the method reach access-depth <= target_depth?
    reached_depth: int      # the prefix's actual max access-depth (the floor when not reached)
    road_length_m: float
    displacement: float     # Sigma disk-graze probability at the depth-D prefix
    pct_displaced: float
    propose_seconds: float  # wall-clock to reblock the method (overprovisioned), passed through


@dataclass(frozen=True)
class LensBRow:
    method: str
    budget_m: float
    external_connectivity: float
    internal_connectivity: float
    displacement: float
    pct_displaced: float


def two_lens_rows(block: Block, roads_by_method: dict[str, GeoDataFrame],
                  propose_seconds: dict[str, float], target_depth: int, budget_m: float, *,
                  corridor_m: float = 3.0) -> tuple[list[LensARow], list[LensBRow]]:
    """Pure two-lens table logic (no I/O, no rendering). For each method's full road set:
    Lens A truncates to the depth-`target_depth` prefix (`prefix_to_depth`); Lens B truncates to the
    shared `budget_m` (`truncate_to_length`) and scores external (`access_benefit`) + internal
    (`commute_ratio`) connectivity. `propose_seconds` is the caller-measured reblock time per method,
    reported verbatim (kept out of this function so it stays deterministic)."""
    radii = building_radii(block.building_points, corridor_m)
    ext_factory = access_benefit(block, None)
    lens_a: list[LensARow] = []
    lens_b: list[LensBRow] = []
    for name, roads in roads_by_method.items():
        prefix_a, reached_depth = prefix_to_depth(block, roads, target_depth)
        lens_a.append(LensARow(
            method=name, reached=reached_depth <= target_depth, reached_depth=reached_depth,
            road_length_m=float(prefix_a.geometry.length.sum()),
            displacement=displacement(block.building_points, radii, prefix_a, corridor_m),
            pct_displaced=pct_displaced(prefix_a, corridor_m, block.building_points, radii),
            propose_seconds=propose_seconds[name]))
        prefix_b = truncate_to_length(block, roads, budget_m)
        lens_b.append(LensBRow(
            method=name, budget_m=budget_m,
            external_connectivity=ext_factory(prefix_b),
            internal_connectivity=commute_ratio(block, prefix_b),
            displacement=displacement(block.building_points, radii, prefix_b, corridor_m),
            pct_displaced=pct_displaced(prefix_b, corridor_m, block.building_points, radii)))
    return lens_a, lens_b
```

- [ ] **Step 4: Run the two `two_lens_rows` tests**

Run: `pixi run pytest tests/test_compare_budgets.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Add `run_two_lens` (timing, rendering, CSV) + `main` to `scripts/compare_budgets.py`**

Append:

```python
def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def run_two_lens(region: list[Block], methods: dict[str, Method], target_depth: int,
                 out_dir: Path, *, corridor_m: float = 3.0) -> tuple[list[LensARow], list[LensBRow]]:
    """Reblock each method once over `region` (timed), compute both lens tables, write the two CSVs,
    and render one after-heatmap per method per lens. The region block is method-independent (same
    parcels/streets every reblock), so the first method's block scores every method and fixes the
    shared render `vmax`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    roads_by_method: dict[str, GeoDataFrame] = {}
    proposals: dict[str, Proposal] = {}
    propose_seconds: dict[str, float] = {}
    block: Block | None = None
    for name, method in methods.items():
        t0 = time.perf_counter()
        result = region_reblock(region, method, [])
        propose_seconds[name] = time.perf_counter() - t0
        block = result.block
        proposals[name] = result.proposal
        roads_by_method[name] = cast(GeoDataFrame, result.proposal.roads)
    assert block is not None
    budget = matched_budget({n: float(r.geometry.length.sum()) for n, r in roads_by_method.items()})
    lens_a, lens_b = two_lens_rows(block, roads_by_method, propose_seconds, target_depth, budget,
                                   corridor_m=corridor_m)

    kc_eval = KComplexityEval()
    vmax: int | None = None
    for name in methods:
        prefix_a, _ = prefix_to_depth(block, roads_by_method[name], target_depth)
        prefix_b = truncate_to_length(block, roads_by_method[name], budget)
        for prefix, tag in ((prefix_a, f"depth{target_depth}"), (prefix_b, "matched")):
            truncated = replace(proposals[name], roads=prefix, block_identity=None)
            kc = kc_eval.score(block, truncated)
            if vmax is None:
                vmax = int(kc.fields["access_before"].max())
            fig = render_after(block, truncated, kc.fields["access_after"], vmax=vmax, metrics=kc,
                               frame=frame_bbox(block.parcels),
                               displaced_points=_displaced_points(block, truncated))
            save_render(fig, out_dir / f"after_{name}_{tag}.jpg")
            plt.close(fig)

    _write_csv(out_dir / "lens_a_depth.csv",
               ["method", "target_depth", "reached", "reached_depth", "road_length_m",
                "displacement", "pct_displaced", "propose_seconds"],
               [[r.method, target_depth, r.reached, r.reached_depth, f"{r.road_length_m:.1f}",
                 f"{r.displacement:.1f}", f"{r.pct_displaced:.4f}", f"{r.propose_seconds:.1f}"]
                for r in lens_a])
    _write_csv(out_dir / "lens_b_matched.csv",
               ["method", "budget_m", "external_connectivity", "internal_connectivity",
                "displacement", "pct_displaced"],
               [[r.method, f"{r.budget_m:.1f}", f"{r.external_connectivity:.6g}",
                 f"{r.internal_connectivity:.6g}", f"{r.displacement:.1f}", f"{r.pct_displaced:.4f}"]
                for r in lens_b])
    return lens_a, lens_b


def main() -> None:
    out_dir = Path(sys.argv[1])
    target_depth = int(sys.argv[2])
    method_names = sys.argv[3].split(",")
    overrides = ["max_blocks=1", *sys.argv[4:]]
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config", overrides=overrides)
    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    region_builder = cast(RegionBuilder, instantiate(cfg.region_builder))
    groups = [[str(b) for b in g] for g in cfg.block_ids]
    region = build_regions(source, screen, region_builder, groups, 1)[0]
    methods = {n: cast(Method, instantiate(cfg.all_methods[n])) for n in method_names}
    lens_a, lens_b = run_two_lens(region, methods, target_depth, out_dir)
    for a in lens_a:
        mark = "reached" if a.reached else f"FLOOR depth {a.reached_depth}"
        print(f"[lens A d<={target_depth}] {a.method}: {mark} at {a.road_length_m:.0f} m, "
              f"{a.displacement:.0f} displaced, {a.propose_seconds:.1f} s")
    for b in lens_b:
        print(f"[lens B {b.budget_m:.0f} m] {b.method}: ext={b.external_connectivity:.3f} "
              f"int={b.internal_connectivity:.3f} {b.displacement:.0f} displaced")


if __name__ == "__main__":
    main()
```

**Note:** the render loop is `render_methods_matched.py`'s exact proven chain — `replace(proposal, roads=prefix, block_identity=None)` → `kc_eval.score(block, truncated)` → `render_after(...)` — with `block_identity=None` so the derive-memo cache never hands back the untruncated access-depth. `proposals: dict[str, Proposal]` is typed (import `Proposal` from `reblock.contracts`), so no `# type: ignore` is needed. If `render_after`'s keyword names differ, mirror `render_methods_matched.py`'s call verbatim.

- [ ] **Step 6: Delete `scripts/render_methods_matched.py`**

```bash
git rm scripts/render_methods_matched.py
```

(Its matched-budget render is now Lens B in `run_two_lens`. The README reference is fixed in Task 4. Confirm no `.py` imports it: `grep -rn render_methods_matched src tests scripts` returns nothing.)

- [ ] **Step 7: Add a `run_two_lens` smoke test**

Append to `tests/test_compare_budgets.py`:

```python
def _street_block(x0: int, block_id: str) -> Block:
    # A 3x3 grid of unit parcels fronting a street on its bottom edge, offset to x0 so two of them
    # tile into a small 2-block region.
    polys = [Polygon([(x0 + i, j), (x0 + i + 1, j), (x0 + i + 1, j + 1), (x0 + i, j + 1)])
             for i in range(3) for j in range(3)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(9))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[LineString([(x0, 0), (x0 + 3, 0)])], crs=UTM)
    return Block(block_id=block_id, crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_run_two_lens_writes_tables_and_renders(tmp_path) -> None:
    # End-to-end glue smoke test on a tiny region with a real reblocker (DijkstraReblocker paves
    # everything, so it reaches a shallow depth). Asserts the two CSVs + a render per lens are
    # written and wall-clock propose time is captured.
    from reblock.methods.dijkstra import DijkstraReblocker
    from scripts.compare_budgets import run_two_lens

    region = [_street_block(0, "a"), _street_block(4, "b")]
    lens_a, lens_b = run_two_lens(region, {"dijkstra": DijkstraReblocker()}, target_depth=2,
                                  out_dir=tmp_path)
    assert (tmp_path / "lens_a_depth.csv").exists()
    assert (tmp_path / "lens_b_matched.csv").exists()
    assert (tmp_path / "after_dijkstra_depth2.jpg").exists()
    assert (tmp_path / "after_dijkstra_matched.jpg").exists()
    assert len(lens_a) == 1 and lens_a[0].propose_seconds > 0.0
    assert len(lens_b) == 1
```

- [ ] **Step 8: Run the full compare_budgets test module**

Run: `pixi run pytest tests/test_compare_budgets.py -v`
Expected: PASS (all three tests). If `render_after`'s keyword names differ from `render_methods_matched.py`'s call, mirror that file's exact call (it is the proven reference).

- [ ] **Step 9: Lint + type-check**

Run: `pixi run ruff check scripts/compare_budgets.py tests/test_compare_budgets.py && pixi run mypy --strict scripts/compare_budgets.py`
Expected: no errors (resolve any typing wrinkle via the typed-`Proposal`-dict note in Step 5).

- [ ] **Step 10: Commit**

```bash
git add scripts/compare_budgets.py tests/test_compare_budgets.py
git rm scripts/render_methods_matched.py
git commit -m "feat: two-lens (depth-target + matched-budget) compare driver, with timing

Replaces scripts/render_methods_matched.py; Lens B is the matched-budget render.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

## Task 4: README rewrite + figure regeneration

**Files:**
- Modify: `examples/multiblock/README.md`, `examples/multiblock/run.log`
- Modify (if the multiblock row description shifts): `examples/README.md`
- Regenerate: `screen.jpg`, `region.jpg`, the Lens A/B renders, the frontier PNGs/CSVs, `lens_a_depth.csv`, `lens_b_matched.csv`

**Interfaces:**
- Consumes: Task 1 (squared screen coloring), Task 3 (`scripts.compare_budgets`).

- [ ] **Step 1: Determine the working depth target D**

Run the two-lens driver at D=3 over the region (deepest block `ZAF.9.3.1_1_5810`, `dense_cluster` at `max_buildings=3000`):

```bash
pixi run python -m scripts.compare_budgets examples/multiblock 3 \
  clearance,greedy_arterial_buildable,osm_footpaths \
  data=capetown_full region_builder=dense_cluster region_builder.max_buildings=3000 \
  "block_ids=[[ZAF.9.3.1_1_5810]]" \
  all_methods.clearance.max_roads=3000 all_methods.clearance.depth_target=3 \
  all_methods.greedy_arterial_buildable.candidate_policy=fixed \
  +all_methods.greedy_arterial_buildable.max_anchors=64 \
  desire_source.snapshot=examples/multiblock/desire_lines_5810.geojson \
  2>&1 | tee -a examples/multiblock/run.log
```

Inspect `examples/multiblock/lens_a_depth.csv`. If `clearance` shows `reached=True`, **use D=3** everywhere below. If `clearance` shows `reached=False`, **rerun the command with `3`→`4` and `depth_target=3`→`depth_target=4`** and use D=4 everywhere below (README prose, figure filenames `after_*_depth4.jpg`, table headers). Record which D you used.

- [ ] **Step 2: Regenerate the screen + region maps (squared coloring)**

```bash
pixi run python -m reblock.run \
  data=capetown_full screen=dense_compact max_blocks=1 \
  region_builder=dense_cluster region_builder.max_buildings=3000 \
  method=clearance method.depth_target=3 method.max_roads=2000 \
  eval=kcomplexity render.enabled=true region_map.enabled=true \
  2>&1 | tee -a examples/multiblock/run.log
```

Copy the run's `screen.png`/`region.png` (and the `region:…_before.png`/`…_after.png` heatmaps) from the Hydra run dir into `examples/multiblock/`, converting to the gallery `.jpg` the same way the current figures were made (downsize to JPEG — e.g. `pixi run python -c "from PIL import Image; Image.open('screen.png').convert('RGB').save('screen.jpg', quality=85)"`). The screen map is now colored by `nA/P²` and titled `depth² proxy n·A/P²`.

- [ ] **Step 3: Regenerate the frontier curves + CSVs (backdrop)**

```bash
pixi run python -m reblock.compare \
  data=capetown_full region_builder=dense_cluster region_builder.max_buildings=3000 \
  "block_ids=[[ZAF.9.3.1_1_5810]]" \
  methods=[clearance,greedy_arterial_buildable,osm_footpaths] max_blocks=1 \
  all_methods.clearance.max_roads=3000 all_methods.clearance.depth_target=3 \
  all_methods.greedy_arterial_buildable.candidate_policy=fixed \
  +all_methods.greedy_arterial_buildable.max_anchors=64 \
  desire_source.snapshot=examples/multiblock/desire_lines_5810.geojson \
  2>&1 | tee -a examples/multiblock/run.log
```

Copy `frontier_external_connectivity.csv`, `frontier_internal_connectivity.csv`, `displacement_vs_length.csv`, `displacement_table.csv`, and the `compare_*`/`displacement` PNGs into `examples/multiblock/`, as the current README already does. (These are the frontier backdrop that Lens A slices horizontally and Lens B slices vertically.)

- [ ] **Step 4: Rewrite `examples/multiblock/README.md`**

Rewrite the prose to the reorganized structure (use the real numbers from `lens_a_depth.csv`, `lens_b_matched.csv`, the frontier CSVs, and `displacement_table.csv` — do NOT invent figures). Sections:

- **§1 Screen the metro** — present the proxy as `depth² = n × compactness` (compactness `A/P²`, Polsby-Popper up to `4π`; derived: parcels ÷ frontage-parcels ⇒ `depth ≈ √(nA)/P`). State the screen ranks by depth `√(nA)/P` but the choropleth (now `screen.jpg`) is **colored by the squared proxy `nA/P²`** for slum-vs-formal contrast (validated: Nairobi Spearman(proxy, k_complexity)=0.72 vs density −0.12). Keep the flagged-count and locator link.
- **§2 Grow the deep core** — unchanged (23-block core, ~10,706 homes).
- **§3 The two-lens comparison** — replaces the clearance-only "Reblock to depth 3":
  - Explain the two budgets and that the frontier curves contain both (Lens A = horizontal slice at fixed depth/benefit → read road; Lens B = vertical slice at fixed road → read benefit).
  - **Lens A — fixed outcome (every parcel ≤ depth D):** a table from `lens_a_depth.csv` — per method: road length, displacement (Σ graze-probability + % of homes), and **wall-clock propose time**; `osm_footpaths` reported ✗ with its floor depth if it can't reach D. Renders: `after_{method}_depth{D}.jpg`.
  - **Lens B — fixed cost (matched road budget = sparsest method's total):** a table from `lens_b_matched.csv` — per method: external + internal connectivity + displacement at the matched budget. Renders: `after_{method}_matched.jpg`. Reproduce command: the `scripts.compare_budgets` invocation from Step 1.
- **§4 Why it's tractable** — the former §5, unchanged content, renumbered.

Point the reproduce/`run.log` note at `run.log` (regenerated). Remove the old §3 "Reblock to depth 3" single-command block and the old §4 "matched budget"/"displacement at scale" prose that Lens A/B now subsume; fold their still-true displacement explanation (disk graze-probability) into Lens A/B where displacement is reported.

- [ ] **Step 5: Update `examples/README.md` if needed**

Read `examples/README.md`; if the multiblock row still describes it as "reblock to depth 3 / matched budget render", update that one line to "two-lens comparison (depth target + matched budget) with per-method timing". Otherwise leave it.

- [ ] **Step 6: Verify the whole check suite is green**

Run: `pixi run check`
Expected: ruff + mypy --strict + pytest all pass.

- [ ] **Step 7: Verify README internal consistency**

Re-read `examples/multiblock/README.md` end to end: every figure it references exists in `examples/multiblock/`; every number matches the regenerated CSVs; the depth `D` is consistent throughout (headline, table, filenames); no dangling reference to `render_methods_matched.py` or "Reblock to depth 3". Fix any drift.

- [ ] **Step 8: Commit**

```bash
git add examples/multiblock examples/README.md
git commit -m "docs: two-lens multiblock example (depth target + matched budget) + squared screen

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

## Self-Review

**1. Spec coverage:**
- Spec §3.1 formula `depth² = n × compactness` → Task 4 §1 prose. ✓
- Spec §3.1 squared screen coloring (emit.region_map) → Task 1. ✓
- Spec §3.3 Lens A (prefix-to-depth-D + road/displacement/time + renders; osm honest ✗; D=3 or 4) → Task 2 (`prefix_to_depth`) + Task 3 (`two_lens_rows` Lens A + timing) + Task 4 Steps 1/4. ✓
- Spec §3.3 Lens B (matched budget, external+internal+displacement + renders) → Task 3 (`two_lens_rows` Lens B + `run_two_lens` renders) + Task 4 §3. ✓
- Spec §3.3 frontier backdrop → Task 4 Step 3 + §3 prose. ✓
- Spec §3.4 §4 renumber → Task 4 §4. ✓
- Spec §4 delete render_methods_matched, fold into new driver → Task 3 Steps 5–6. ✓
- Spec §6 testing: emit column = squared + rank invariance (Task 1 test); prefix is first ≤ D, fixed-input floor, timing captured (Task 2 + Task 3 tests). ✓
- Spec §5 scope boundaries (no Nairobi, no density, no native arterial knob) → nothing in the plan adds them. ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code; the one conditional (D=3 vs 4) is an explicit empirical branch with both arms specified. The `# type: ignore` markers in Task 3 Step 5 carry an explicit "prefer the typed dict" resolution.

**3. Type consistency:** `LensARow`/`LensBRow` fields, `two_lens_rows`/`run_two_lens`/`prefix_to_depth`/`max_access_depth`/`_drainage_ordered`/`_screen_proxy` signatures are identical everywhere they appear (Interfaces blocks ↔ code steps ↔ tests). `prefix_to_depth` returns `tuple[GeoDataFrame, int]` consistently. `region_reblock(blocks: list[Block], method, evals)` matches the verified signature. `render_after`/`save_render`/`frame_bbox` calls mirror the proven `render_methods_matched.py` reference.

## Execution Handoff

Execution is **subagent-driven-development** (owner's standing preference — not asked). Fresh implementer per task + task review (spec + quality) between tasks + a final whole-branch review. Task 4 is compute-heavy (region-scale, multiple methods) and doc-heavy — run it last, after Tasks 1–3 are green.
