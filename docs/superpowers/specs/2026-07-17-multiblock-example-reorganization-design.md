# Multiblock Example Reorganization — Design

**Status:** design approved (2026-07-17, incrementally), pending spec review
**Author:** owner + Claude
**Scope:** `examples/multiblock` (Cape Town) + a small `emit.py` screen-coloring change. Cross-city
(Nairobi/Kibera) generalization is explicitly OUT of scope — a separate future example.

## 1. Goal

Turn `examples/multiblock` from a clearance-centric walkthrough into a **systematic method comparison
at settlement scale**, structured around the two distinct budgets a reader actually cares about:
"same *outcome*, what does it cost?" and "same *cost*, what outcome?". Plus a sharper depth-proxy
presentation (better coloring + a principled formula) now that the proxy is validated against
ground truth.

## 2. Why (what's wrong today)

- **§3 "Reblock to depth 3" shows only clearance** — the other methods never appear at a common
  operating point, so the reader can't compare them head-to-head there.
- The example conflates two different "budgets" — a **fixed depth target** (what §3 does, clearance-
  only) and a **fixed road budget** (what §4's matched-budget render does) — without ever naming the
  distinction or showing all methods under both.
- **No per-method runtime** is reported, though "which methods scale" is the multiblock example's
  whole thesis (§5, currently asserted anecdotally).
- The **screen map is colored by `√(nA)/P`**, which saturates most of a dense settlement dark (poor
  slum-vs-formal contrast), and the **depth-proxy formula is presented opaquely**.

Validation done during design (`spike_*` in scratchpad, and the Kenya-blocks cross-check):
- The proxy tracks ground-truth access-depth: **Spearman(√(nA)/P, k_complexity) = 0.72 on Nairobi**
  (≈ Cape Town's 0.76), while **density n/A = −0.12** (uncorrelated). So depth (count `n`, perimeter
  `P`), not density, is the right signal — validated on a fresh city.
- The **squared proxy `nA/P²`** separates informal fabric far better than `√(nA)/P` (Khayelitsha
  within-settlement spread **10.2× vs 3.2×**; visually it fades the formal grid and pops the informal
  clusters). Squaring is monotone, so the *ranking* and flagged blocks are unchanged — only the
  color scale improves.

## 3. Design

The region is unchanged (the screen's deepest 23-block core, ~10,706 homes, Cape Town). Section arc:

### 3.1 §1 Screen the metro — formula + coloring

- **Formula.** Present the depth proxy as its clean decomposition, derived not asserted:
  parcels ÷ frontage-parcels ⇒ `depth ≈ √(nA)/P`, and **`depth² = n · (A/P²) = (building count) ×
  (compactness)`**, where `A/P²` is the compactness (the Polsby-Popper measure `4πA/P²` up to the
  irrelevant `4π`). This reads as principled and separates the two drivers of depth (how many
  buildings, how compact the shape). §1 prose is rewritten around this.
- **Coloring — `emit.py` change.** `region_map` (`emit.py:149`) colors the `screen.png`/`region.png`
  choropleths by the `proxy` column, currently `√(building_count·area)/perimeter` (line 177).
  **Change the coloring to the squared proxy `nA/P²`** (i.e. square the existing expression), keeping
  the `YlOrRd` cmap and the p99-cap pattern (recompute `vmax` as the p99 of the squared values).
  Migrate-not-accommodate: change the default; no `√`-vs-squared config flag. This is a strictly
  better slum detector (validated), so all `region_map` outputs benefit. Ranking/flagging is
  unchanged (monotone); only the color scale sharpens.

### 3.2 §2 Grow the deep core — unchanged.

### 3.3 §3 The two-lens method comparison (the core rework)

Replace the clearance-only "reblock to depth 3" with a systematic comparison of the scalable methods
(`clearance`, `greedy_arterial`, `osm_footpaths`; `topology` stays region-excluded) under **two named
lenses**, with the frontier curves as the connecting backdrop.

**Lens A — fixed *outcome* (depth target), varying cost.** Drive each method to "every parcel within
access-depth ≤ D" and report what it cost:

- **Target D.** Prefer **D = 3** (matching the current headline). Verify empirically that the one
  coverage-capable method in the region set (`clearance`) reaches it; if it cannot, **fall back to
  D = 4** for the whole lens. Pick the largest D that `clearance` hits, preferring 3.
- **Per-method mechanic — prefix-to-depth-D.** Run each method **overprovisioned** (a generous
  `max_roads`), then walk its drainage-ordered road prefix (the `_sweep`/`truncate_to_length`
  ordering) until the block's **max access-depth first drops to ≤ D**; that prefix is the method's
  depth-D solution. `clearance` reaches D natively via `depth_target`; `greedy_arterial` reaches it
  (if at all) by overprovisioning; `osm_footpaths` is a **fixed input** — it lands where it lands,
  reported as its actual floor depth (honest ✗ if it never reaches D, which is itself the
  informative "as-built paths don't cover the deep interior" result).
- **Report, per method:** road length + displacement (Σ graze-probability) + **wall-clock propose
  time** at the depth-D prefix, plus an after-render. A table + renders. `✗ (floor depth k)` where a
  method genuinely can't reach D.
- **Access-depth at a prefix** is computed with the existing access-depth machinery
  (`parcel_access_layers`/`access_burden` in `budget.py` / the `kcomplexity` eval), evaluated on the
  growing prefix — the same depth the pipeline already computes for the headline run.

**Lens B — fixed *cost* (matched road budget), varying outcome.** Truncate every method to the same
added-road-length (the sparsest method's total, as today's matched-budget render does via
`scripts/render_methods_matched.py`), and report **benefit on both axes** (external + internal
connectivity) + displacement + the renders. This is today's §4 "matched budget" render, promoted to
a systematic table with the metric numbers alongside.

**Frontier backdrop.** Keep the benefit-vs-road-length frontier curves (§4 today) as the continuous
object that *contains* both lenses: Lens A is a horizontal slice (fix benefit/depth, read road), Lens
B a vertical slice (fix road, read benefit). One or two sentences make this explicit so the reader
sees the lenses aren't ad hoc.

### 3.4 §4 Why it's tractable — unchanged content, renumbered as the closing section.

## 4. Components & interfaces

- **`emit.py` `region_map`** — square the `proxy` column expression + p99 `vmax` recompute. ~2 lines.
- **Lens A driver** — a script (e.g. `scripts/compare_to_depth.py`) OR an extension of the compare
  path that, per method: proposes overprovisioned, sweeps the drainage-ordered prefix, finds the
  first prefix with max access-depth ≤ D, and records (road_length_m, displacement, wall-clock,
  after-render). Reuses `truncate_to_length`, the access-depth machinery, `displacement`,
  `building_radii`, and the render path. Emits the Lens-A table (CSV) + per-method renders.
- **Lens B** — reuse `scripts/render_methods_matched.py` (matched-budget renders) + read the frontier
  CSVs for the benefit numbers at the matched budget. Assemble the Lens-B table.
- **`examples/multiblock/README.md`** — rewrite §1 (formula), §3 (two lenses replacing the
  clearance-only depth-3 section), keep §2/§4; regenerate all affected figures (screen recolored,
  Lens-A renders/table, Lens-B renders/table, frontier).
- **`examples/README.md`** — one-line update if the multiblock row's description shifts.

## 5. Scope boundaries (YAGNI)

- **No Nairobi/Kibera** in this example (validated the proxy; a cross-city example is separate future
  work needing the data-layer plumbing — add `KEN`, generalize the CT-hardcoded buildings fetch).
- **No density metric** — rejected (uncorrelated with ground-truth depth).
- **No native `depth_target` knob for arterial/topology** — use overprovision + prefix-to-depth-D
  instead. (A native depth-targeted arterial is a possible separate method feature, not this.)
- **No new metric or reblocking method** — this is presentation + a comparison harness over existing
  methods.

## 6. Testing / validation

- `emit.py` coloring change: an assertion that `region_map`'s choropleth column equals `nA/P²`
  (squared), and that existing `region_map` tests (if any) still pass; the flagged/ranked set is
  unchanged (monotone) — assert ranking invariance on a small fixture.
- Lens A driver: on a small fixture region, assert (a) the reported prefix is the first with max
  access-depth ≤ D, (b) a fixed-input method that can't reach D reports ✗ with its floor depth, (c)
  timing is captured (> 0). Determinism: same inputs → same prefix.
- The example regeneration is compute-heavy (region-scale, multiple methods, overprovisioned) — the
  **final task**; reproduce commands in the README match what generated the figures.
- `pixi run check` (ruff + mypy --strict + pytest) stays green.

## 7. Global constraints

- Migrate-not-accommodate: change the screen coloring default to squared; no `√`-vs-squared flag.
- `pixi run check` green; ruff (no semicolons E702, ≤100-char E501, `zip(strict=)` B905); mypy --strict.
- Commit trailers + PR-body footer per repo convention.
- The example must remain reproducible-by-CLI (Cape Town, `capetown_full`, committed OSM snapshot).

## 8. Implementation phasing

1. **`emit.py` squared coloring** + its test (small, self-contained; unblocks the screen figure).
2. **Lens A driver** (`compare_to_depth`): the prefix-to-depth-D + timing harness + its tests;
   determine the working D (3, or 4 if needed) on the real region.
3. **Lens B assembly** (matched-budget table from existing renders + frontier CSVs).
4. **README rewrite + figure regeneration** (§1 formula, §3 two lenses; screen recolor; Lens A/B
   figures + tables; frontier). Compute-heavy — last. Ends `pixi run check` green.
