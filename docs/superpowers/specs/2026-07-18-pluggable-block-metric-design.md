# Pluggable Block Metric — Design

**Status:** design approved in principle (2026-07-18, incrementally), pending spec review
**Author:** owner + Claude
**Generalizes / subsumes:** `docs/superpowers/specs/2026-07-17-true-depth-everywhere-design.md` (the
hand-rolled `depth` behavior on branch `true-depth-everywhere`). Its `block_depths` batched peel and
screen-depth exposure are reused verbatim; its hardcoded-depth screen gate, region growth, and
coloring become the `depth` preset of this framework.

## 1. Goal

Make the per-block scoring metric — the thing the screen gates/ranks on, the region builder grows by,
and the maps color by — a **Hydra-swappable `conf/metric/` config group**. Ship three presets
(`depth`, `depth_density`, `density_compactness`) plus the knobs to tune them, so different
settlement-targeting lenses (deepest access vs crowded-and-deep vs dense-compact) are a one-flag
change, each with a **fast batch proxy** and a **true fine score**.

## 2. Why

These are legitimately different use cases, not competing approximations of one truth:
- **`depth`** — deepest street-access fabric (the reblocking objective). Flags deep-by-rings blocks
  regardless of crowding.
- **`depth_density`** — deep *and* crowded (visually isolates the real informal settlements; sparse
  deep-by-geometry blocks fade).
- **`density_compactness`** — dense, compact fabric from geometry alone (no peel needed → scores the
  whole metro from columns, very fast).

Today the metric is hardcoded (`√(nA)/P` proxy + true peel depth) and threaded by hand through the
screen, region builder, and `region_map`. Making it a config group turns "try a different lens" from
a code change into `metric=…`, and lets each metric carry its own fast proxy + gate.

## 3. Design

### 3.1 The metric abstraction — a composable `BlockMetric` algebra

A metric is a small expression tree of **primitives** combined by **higher-order combinators**. Every
node implements the same three-part interface — a fast batch **proxy**, a true per-block **fine**, and
**needs_peel** — so the rest of the system (§3.3) only ever talks to the top node and never knows
whether it's a leaf or a deep composition.

```python
class BlockMetric(Protocol):
    needs_peel: bool                                              # does `fine` need the peel?
    def proxy(self, blocks: GeoDataFrame) -> pd.Series: ...       # vectorized, from columns (n,A,P)
    def fine(self, depth: float, count: float, area: float, perim: float) -> float: ...  # per block
```

**Primitives** (leaves — `n` = building count, `A` = area m², `P` = perimeter m):

| primitive | `proxy` (batch, columns) | `fine` (per block) | needs_peel |
|---|---|---|---|
| `Depth()` | `√(nA)/P` (the depth *proxy*) | `depth` (true peel `access_before(block).max()`) | ✔ |
| `Density()` | `n/A` | `n/A` | ✗ |
| `Compactness()` | `A/P²` | `A/P²` | ✗ |

`Depth` is the only primitive whose `proxy` and `fine` differ — `proxy` estimates depth from columns,
`fine` uses the real peel. Density/Compactness are closed forms, identical in both. (Raw `Count()` /
`Area()` / `Perimeter()` leaves are trivial to add later; YAGNI for now.)

**Combinators** (internal nodes) — each folds the interface over its children:

```python
Power(base, k):    proxy = base.proxy ** k;   fine = base.fine ** k;   needs_peel = base.needs_peel
Product(*terms):   proxy = ∏ t.proxy;          fine = ∏ t.fine;         needs_peel = any(t.needs_peel)
```

`needs_peel` is just an OR over the tree, so the screen peels **iff the expression contains a
`Depth`**. The `proxy`/`fine` duality propagates cleanly (each combinator applies itself to both
evaluators). Start with **`Power` + `Product` only**; `Sum`/`Ratio`/`Log`/… are added when a real use
case appears (YAGNI *within* the algebra).

**Presets are shallow trees:**

| `metric=` | expression | needs_peel |
|---|---|---|
| `depth` | `Depth()` | ✔ |
| `depth_density` | `Product(Depth(), Density())` | ✔ |
| `density_compactness` | `Product(Density(), Compactness())` | ✗ |

Tuning ("square/sqrt something") is *composition*, not a special field: `Product(Power(Density(), 2),
Compactness())` for density²×compactness, or `Power(Product(Depth(), Density()), 0.5)` for
`√(depth·density)` — a power over a whole sub-expression, which a flat exponent triple can't express.
The **gate (§3.2) is not part of the algebra** — it wraps the finished score at the top level, so
metric *shape* and *selection rule* stay orthogonal.

### 3.2 The gate is part of the metric (per-metric), not a global percentile

The gate has two distinct roles (per the design discussion), handled separately:

- **Selection gate (semantic) — per-metric, carried by the metric.** A small object
  `Gate{kind: "absolute" | "percentile", value: float}`. `absolute` = "keep score ≥ value" (portable
  across cities — a formal city flags few blocks, an informal one many); `percentile` = "keep the top
  value % by score" (when a fixed fraction genuinely is the intent). Each preset picks what fits:
  `depth` → `absolute` in rings; a geometry metric may want an absolute floor or a percentile. A
  **single global percentile is rejected**: it isn't portable (always flags a fixed fraction
  regardless of whether the fabric is deep) and can't express the floors a peel-free metric needs as
  its *final* selection.
- **Cheap recall pre-filter (non-semantic) — shared, only for `needs_peel` metrics.** Its job is to
  cut 83k → ~Nk so the peel runs on fewer, with high recall (never drop a block the selection gate
  would keep). A generous percentile or a survivor cap (e.g. `keep top 30% by proxy` or `cap at N`),
  configured once at the screen level, not per metric. Metrics that don't peel skip it.

### 3.3 Wiring (generalizing the hardcoded `depth` seams that already exist)

- **Screen (`dense_compact`)** takes a `BlockMetric`:
  - Cheap gate = `metric.proxy(blocks)` + the recall pre-filter → survivors (skipped when
    `not needs_peel`).
  - Fine pass = (if `needs_peel`: batch-peel survivors → true depth) → `metric.fine(...)` per block →
    apply the metric's **selection gate** → flagged set, ranked by the fine score. For
    `needs_peel=False`, the fine score *is* the proxy (no Voronoi/peel step at all — the whole screen
    is a vectorized column pass).
  - `selection_depths` generalizes to **`selection_scores(source) -> dict[str, float]`** (block_id →
    fine score) — what the coloring keys on.
- **Region growth (`pipeline._region_depth_map` → `_region_score_map`):** for the seed's reachable
  neighbourhood, compute `metric.fine`; when `needs_peel`, get the depth factor from the one batched
  `block_depths` call (already built); else from columns. `depth_fn = score_map.get` (unchanged
  region-builder seam).
- **Coloring (`emit.region_map`):** color flagged blocks by `metric.fine` (from `selection_scores`),
  backdrop/region members likewise; colorbar label from `metric.name`. `block_depths` still supplies
  any un-mapped member's depth factor.
- **`block_depths`** (the batched peel) is unchanged — it's the depth-factor supplier, used only when
  `needs_peel`.

### 3.4 Efficiency the abstraction buys

`density_compactness` (`needs_peel=False`) skips the Voronoi+peel entirely — the screen becomes a
single vectorized pass over the columns for the whole metro (sub-second), and region growth needs no
`block_depths` call. Peel cost is paid *only* by metrics that actually use depth.

### 3.5 Example variants (one metric end-to-end each) + a generated README

Because the metric now drives the screen, region growth, AND coloring, the multiblock example is
**re-cut as one variant per metric** — `examples/multiblock_depth/` and
`examples/multiblock_depthdensity/` — each run with a SINGLE metric end-to-end (no decoupling). Put
side by side they *are* the demonstration of swappability (depth ships its true-depth region; the
denser depth_density region is the visibly-better slum target). This replaces the single
`examples/multiblock/`.

Each variant's `README.md` is **machine-generated from the run outputs** so its numbers can never
drift from the CSVs (the class of error hand-editing kept introducing), via a two-layer split:

- **Generator (pure, dir-reader):** `gen_example_readme(run_dir, *, metric_name, formula, blurb) ->
  README.md` reads the artifacts already on disk (the two-lens `lens_*.csv`, `frontier_*.csv`,
  `displacement_table.csv`, a small `meta.json` of structured stats the orchestrator writes, and the
  figure filenames). **Sections are data-gated** — a section is emitted iff its artifacts are present
  (screen section if `meta.json` has screen stats; two-lens section if the lens CSVs exist; frontier
  section if those exist). Pure function of the directory → tested on a fixture dir with zero compute,
  and re-emittable from cached outputs without re-running. Templated for a real report (headers, a
  metric formula + `blurb` line, captioned tables, honest stat callouts), not a data dump.
- **Orchestrator (end-to-end):** `gen_multiblock_example(metric)` runs the example's commands
  (`reblock.run` for screen/region/maps, `scripts.compare_budgets` for the two lenses,
  `reblock.compare` for the frontier) into `examples/multiblock_<metric>/`, writes `meta.json`
  (flagged count, region members/parcels, mean depth + density), then calls the generator. One command
  per variant.

A per-run README emitter (a `readme.enabled` emitter on `reblock.run` with flag-gated sections) is a
*different artifact* (an ad-hoc single-run summary, not the multi-command example) and is **out of
scope** here (YAGNI) — the dir-reader covers the example. The short authored `blurb` (2–3 sentences of
framing per variant) is the only hand-written prose; it's supplied to the orchestrator per metric
(a small authored map — NOT a field on the metric node, which is a pure scorer).

## 4. Config

`conf/metric/{depth,depth_density,density_compactness}.yaml` — each a Hydra-instantiated expression
tree (nested `_target_`), plus a top-level `gate` and `name`. Presets are shallow, e.g.:

```yaml
# conf/metric/depth_density.yaml
_target_: reblock.metric.Product
terms:
  - {_target_: reblock.metric.Depth}
  - {_target_: reblock.metric.Density}
name: depth_density
gate: {_target_: reblock.metric.Gate, kind: absolute, value: 2.0}
```

`conf/config.yaml` + `conf/compare_config.yaml` gain a `metric` default of **`depth`** (preserves
today's behavior; the others are opt-in via `metric=…`). The screen carries the shared
recall-pre-filter knob (`proxy_keep_pct` or `survivor_cap`). Deep custom expressions nest further, but
the shipped presets stay one or two levels.

## 5. Relationship to the `true-depth-everywhere` branch + sequencing

The framework is the generalization of that branch, so **evolve the branch into it** rather than land
then immediately refactor:
- **Reuse as-is:** `block_depths` (batched peel), the screen's `screen_selection` pairs + depth
  exposure, `region_map`'s blank-deselected + continuous-true-scale coloring.
- **Generalize:** the screen cheap gate `_depth_proxy` → `metric.proxy`; the fine pass →
  `metric.fine` + `needs_peel` skip; `selection_depths` → `selection_scores`; region growth
  `depth_fn` → `metric.fine`; `region_map` color → `metric.fine` + metric label.
- **Default `metric=depth`** reproduces the branch's depth-targeting behavior (including the
  true-depth region growth the owner accepted), so the multiblock example is regenerated **once** on
  the default; `depth_density` / `density_compactness` are opt-in and unlock the denser regions we saw
  without a second rebuild. **Caveat:** today's fine pass drops on *mean* depth but ranks on *max*
  depth; a single fine scalar can't carry both, so the `depth` preset's fine = max depth (rank +
  color) and its selection gate approximates the mean-depth drop — the flagged *count* may shift
  slightly (the "13,906 of 83,192" headline). The plan pins the `depth` preset's gate to stay as
  close as practical; this is a deliberate, bounded change, not exact byte-reproduction.

The prior true-depth spec/plan are marked superseded-by-this; the branch's completed tasks stand as
the `depth` preset's implementation.

## 6. Scope boundaries (YAGNI)

- **Minimal algebra** — primitives `Depth`/`Density`/`Compactness` + combinators `Power`/`Product`
  only. No `Sum`/`Ratio`/`Log`/extra primitives until a real use case appears.
- **Cheap gate keeps a proxy/percentile pre-filter** — not switched to true depth over 83k.
- **No k-correlation gating** — these are different-purpose lenses by design (owner directive); we do
  not reject a metric for tracking access-depth less well.
- **Default stays `depth`** — no change to which fabric ships as the example's target unless the owner
  flips the default.

## 7. Testing

- Primitives: `Depth`/`Density`/`Compactness` `proxy` + `fine` equal their closed forms on a fixture
  (`Depth.proxy` = `√(nA)/P`, `Depth.fine` = the passed peel depth; Density/Compactness identical in
  both). Combinators: `Power`/`Product` fold `proxy`, `fine`, and `needs_peel` correctly (e.g.
  `Product(Depth, Density).needs_peel` is True; `Product(Density, Compactness).needs_peel` is False and
  its `fine` ignores depth). Composition: `Power(Product(Depth, Density), 0.5)` equals
  `√(depth·density)`.
- `Gate`: absolute keeps `score ≥ value`; percentile keeps the top `value%`; per-metric gates select
  the expected blocks on a fixture.
- Screen: with `metric=density_compactness` the fine pass performs no peel (assert via a spy/cache) and
  still flags/ranks; with `metric=depth` behavior matches the current `depth` selection.
- Region growth / coloring: `_region_score_map` and `region_map` use `metric.fine`; the depth preset
  grows/colors by max peel depth (the true-depth branch's region + colors, modulo the mean-vs-max
  gate caveat in §5).
- README generator: on a fixture directory (sample `meta.json` + `lens_*.csv` + figure files) emits a
  README whose numbers match the CSVs; a section is present iff its artifacts are (data-gated); a dir
  missing the lens CSVs omits the two-lens section without erroring.
- `pixi run check` green. The heavy metro regeneration (the two example variants) is the final task.

## 8. Global constraints

- Migrate-not-accommodate: the hardcoded depth path becomes the `depth` preset — no dual metric path.
- Continuous colormap only (no scheme/binning); `pixi run check` (ruff + mypy --strict + pytest) green;
  ruff E702/E501/B905; commit trailers + PR footer per repo convention.
- Reproducible-by-CLI (Cape Town `capetown_full` + committed OSM snapshot).

## 9. Implementation phasing

1. The `BlockMetric` algebra (`Depth`/`Density`/`Compactness` + `Power`/`Product`) + `Gate` (+ tests):
   the pure scorers and gate, no wiring.
2. `conf/metric/` group + the `metric` default wired into `run`/`compare` config edges.
3. Screen consumes the metric (cheap proxy gate + recall pre-filter; fine pass with `needs_peel` skip;
   `selection_scores`).
4. Region growth + `region_map` consume the metric (`_region_score_map`, coloring + label).
5. The pure dir-reader README generator (`gen_example_readme`) — data-gated sections — tested on a
   fixture directory (no compute).
6. The per-variant orchestrator (`gen_multiblock_example`) — run `depth` and `depth_density`
   end-to-end into `examples/multiblock_depth/` + `examples/multiblock_depthdensity/`, delete
   `examples/multiblock/`, update `examples/README.md`. Compute-heavy — last.
