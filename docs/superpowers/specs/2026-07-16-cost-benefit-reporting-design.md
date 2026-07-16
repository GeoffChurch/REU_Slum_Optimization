# Cost-benefit reporting improvements (Design)

**Date:** 2026-07-16
**Status:** approved design, ready for implementation plan
**Depends on:** the frontier metric + `osm_footpaths` rename (both on `main`/branch `osm-footpaths`).

## Goal

Three coupled improvements to how `reblock.compare` reports the cost-benefit of a reblock:

1. **Added road length (m) is the single cost x-axis** on every curve — retiring the m/ha density
   axis and the `cost=displacement` mode.
2. **Displacement becomes a curve** (buildings displaced vs added road length) plus a terminal table,
   instead of an x-axis mode.
3. **Displacement is extent-aware** — each building is a disk sized from its nearest-neighbor
   distance, and its contribution is the *probability* the corridor grazes it, not a binary
   centroid-in-corridor test.

Plus: the **multiblock example gains one after-render per method**, all at a matched road-length
budget, and both flagship examples regenerate. `migrate, never accommodate`: no back-compat shims —
the old density axis and displacement cost-mode are deleted, not kept alongside.

## A. Road length is the only cost axis

`budget._sweep` already **orders roads by drainage and samples at cumulative-length budgets** — only
the *reported* x-value (`cost_fn`) varies. Change the reported x to cumulative added road length in
metres:

- `_sweep`'s default cost becomes `_length(prefix) = prefix.geometry.length.sum()` (raw metres),
  replacing `_density` (which divided by `area_ha`).
- **Delete** `_cost_fn_for`, the `cost` parameter threaded through `cost_benefit_curve` /
  `efficiency_directness_curves` / `compare()` / `compare_report()`, the `cost` key in
  `conf/compare_config.yaml`, and every `cost == "displacement"` branch.
- `Curve.cost`'s comment becomes "cumulative added road length (m)".
- Frontier CSV column `road_density_m_per_ha` → `road_length_m`. Curve legend `f"{method} ({int
  cost[-1]} m/ha)"` → `f"{method} ({int cost[-1]} m)"`. The `main()` terminal log line drops the
  `%.0f m/ha (%.1f%% paved)` density and reads road length + `% paved` instead.
- `pct_paved` (road area ÷ block area, in `emit.py`) is **unaffected** — it is a coverage fraction,
  independent of the x-axis — and stays in the frontier terminal reporting.

Single-region plots are unchanged in *shape* (density and length are proportional within one region);
the axis label/scale and the CSV column change. Cross-region normalization is not needed — each plot
is one region, and the frontier is read per-region, not as a shared-cap AUC (already retired).

## B. Extent-aware disk displacement

Replace `budget.displacement_count(building_points, roads, corridor_m) -> int` with a float measure.

**Building radii (computed once per block, independent of roads):**
`building_radii(building_points) -> NDArray[float]` where `rᵢ = ½ · NN_distᵢ`, `NN_distᵢ` = distance
from point `i` to its nearest *other* building point (via `scipy.spatial.cKDTree` k=2, or the
geopandas spatial index). No cap. Edge cases:
- `n_points < 2` (no neighbor): `rᵢ = corridor_m` (a minimal fallback footprint).
- coincident points (`NN_dist = 0` → `rᵢ = 0`): contributes 1 iff `dᵢ = 0`, else 0 (guard the
  division).

**Displacement of a road set:**
`displacement(building_points, radii, roads, corridor_m) -> float`:
- `corridor = roads.buffer(corridor_m).union_all()`; `dᵢ = pointᵢ.distance(corridor)` (0 if inside).
- `cᵢ = max(0, 1 − dᵢ/rᵢ)` — the uniform-size-prior contribution (derivation below).
- return `Σ cᵢ`.
- 0 if no roads or no points.

`pct_displaced` (emit.py) becomes `Σcᵢ / n_buildings`.

### Derivation (why `cᵢ = max(0, 1 − dᵢ/rᵢ)`)

Model building `i` as a disk of *true* radius `ρ = X·rᵢ` centered at the point, with `X` a size prior
on `[0,1]` and `rᵢ = NN/2` the disk radius. The corridor grazes (destroys) the building iff `ρ ≥ dᵢ`.
Because "intersects" is monotonic in `ρ`, the contribution is a **survival function**:
`P(destroyed) = P(ρ ≥ dᵢ) = 1 − F_X(dᵢ/rᵢ)`. For the chosen **uniform** prior `X ~ U[0,1]`
(`F_X(u)=u`): `cᵢ = max(0, 1 − dᵢ/rᵢ)` — a linear ramp (center in corridor → 1; `rᵢ` beyond → 0).
This is *not* the area-fraction of a clipped disk (a circular-segment function, ½ at `d=0`); it is the
probability the uncertain-size footprint is grazed, which honestly encodes "even a graze destroys it".
No cap on `rᵢ`: isolated buildings get large `rᵢ`, but roads are built to drain *dense* fabric and
never run near isolated parcels, so their `dᵢ` is large and `cᵢ ≈ 0` regardless.

## C. Displacement-vs-length curve + terminal table

Displacement, being metric-independent, is emitted as its own curve per region:
- A drainage-ordered `_sweep` with `value = displacement(prefix)` yields a `Curve` whose `cost` is
  added road length (m) and whose `y` values are `Σcᵢ` at each budget — a `displacement` curve
  carried in the results alongside the four benefit metrics (e.g. a `MethodCurve` with
  `metric="displacement"`, so `compare_report`'s existing per-(metric, region) plotting draws it).
- `compare_report` emits, per region: the four benefit curves (y = benefit vs x = road length) **and**
  the displacement curve (`displacement_{region}.png`, y-label "buildings displaced (Σ disk-graze
  probability)").
- CSVs: `displacement_vs_length.csv` (method, block, road_length_m, displacement) — the full curve —
  and a terminal `displacement_table.csv` (method, terminal_displacement, pct_displaced, n_blocks).
- `main()` logs each method's terminal displacement + `% displaced`.

## D. Per-method multiblock renders at a matched budget

The multiblock §4 gains one after-render per compared method (`clearance`,
`greedy_arterial_buildable`, `osm_footpaths`), each at a **matched added-road-length budget** = the
**smallest method's total road length** (every method can reach it):
- `truncate_to_length(roads, block, budget_m) -> GeoDataFrame`: drainage-order the roads (reuse
  `road_drainage` + the `_sweep` ordering), take the longest prefix whose cumulative length `≤
  budget_m`.
- Compute each method's roads, find `budget_m = min(total length over methods)`, render each
  method's truncated roads as an after-heatmap → `after_{method}.jpg` ×3 in `examples/multiblock/`,
  shown as a render grid in §4 beside the compare curves.
- §3's headline `after.jpg` (clearance at depth 3, full build — a different purpose) and the §5/§6
  clearance sweeps are untouched. The new §4 renders are the equal-budget head-to-head the single
  `after.jpg` couldn't provide.

## E. Renders visualize the disk-displacement metric

In `render.render_after`, draw each building as a disk of radius `rᵢ = NN/2` shaded on a grey→red ramp
by its displacement fraction `cᵢ` (reusing `_point_disks`, extended to accept per-point radii + a
`c` array driving colour), replacing the current binary red `displaced_points` overlay. The road
corridor is drawn over it, so the reader sees which footprints the road grazes and how strongly. The
`before` render keeps plain building disks (no `c`).

## F. Regenerate both flagship examples + READMEs

Both examples regenerate:
- **method-comparison**: benefit curves re-x-axised to road length (m); a new displacement-vs-length
  plot; fractional displacement figures in the displacement table; renders shaded by `cᵢ`.
- **multiblock**: same, plus the three per-method matched-budget after-renders in §4.
- READMEs: update every "m/ha" → "m", the displacement tables to the new fractional figures, add the
  displacement-vs-length plot + (multiblock) the per-method render grid, and describe the disk model
  in a sentence. The `cost=displacement` reproduce commands collapse into the single length run
  (displacement is now always emitted).

## Files

- `src/reblock/budget.py` — `_length` (replaces `_density`); delete `_cost_fn_for` + `cost` params;
  `building_radii`, `displacement` (replaces `displacement_count`); a `displacement`-valued sweep;
  `truncate_to_length`.
- `src/reblock/emit.py` — `compare_report`: road-length x, the displacement curve + two CSVs, drop the
  `cost`/`tradeoff_table`/displacement-mode branches; `pct_displaced` → fractional; `_displaced_points`
  → disk/`cᵢ`-aware.
- `src/reblock/compare.py` — drop `cost`; add the displacement curve to results; per-method
  matched-budget render wiring for the multiblock; terminal displacement logging.
- `src/reblock/render.py` — `_point_disks` per-point radii + `cᵢ` colour ramp; `render_after` disk
  shading; a matched-budget per-method render entry.
- `conf/compare_config.yaml` — remove `cost`.
- `examples/method-comparison/`, `examples/multiblock/` — regenerate + README rewrites.

## Testing

- `building_radii`: known point set → correct NN/2 radii; `n<2` fallback; coincident points.
- `displacement`/`cᵢ`: point on corridor → 1; point `rᵢ` beyond → 0; halfway (`d=rᵢ/2`) → 0.5;
  isolated far point → ~0; empty roads/points → 0.
- `_length` cost fn: cumulative metres, monotonic, matches `geometry.length.sum()`.
- `truncate_to_length`: budget between road lengths yields the right drainage-ordered prefix; budget
  ≥ total → all roads; budget 0 → none.
- displacement curve: monotonic non-decreasing in road length; terminal equals full-roads
  displacement.
- `compare_report`: emits `displacement_{region}.png`, `displacement_vs_length.csv`,
  `displacement_table.csv`; frontier CSV has `road_length_m`; no `tradeoff_table`/`cost` artifacts.

## Migration (delete, don't shim)

Removed outright: the `cost` config key + parameter, `_cost_fn_for`, `displacement_count`, the m/ha
density cost, `tradeoff_table_{metric}.csv` (displacement-mode), and every `cost == "displacement"`
branch. Existing example artifacts named `tradeoff_table_*` and `curve_*_displacement.png` are
regenerated into the new scheme (`displacement_vs_length.csv`, `displacement_table.csv`,
`displacement_{region}.png`); the old files are deleted.
