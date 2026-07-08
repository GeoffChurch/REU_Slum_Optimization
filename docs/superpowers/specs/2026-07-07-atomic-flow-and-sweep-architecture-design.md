# reblock — Atomic flow + externalized sweeps

**Status:** revised for execution (sliced F1–F4) · **Date:** 2026-07-08 · **Branch:** `flow-refactor` (to be cut)

> **Revision note (2026-07-08):** §1's `block_ids` early-filter has since **shipped** in the kblock
> `Source`, and the slum-detection S1 slice shipped the standalone `reblock.screen` app this spec
> folds away. This refactor is now executed in four ordered slices (see *Slicing & sequencing*
> below), and two requirements are added: a **one-command end-to-end** (detect → reblock → render)
> and a **binary city flagged-map** visual — both land in **F3**.

## Why this exists

The pipeline conflates two things inside `run()`: the **atomic** work (one data source, one method,
one eval, per block) and a **sweep/comparison** (lists of methods/evals, "one before / N afters"
render). The planned peel-reblocker Slice 2 would have piled more sweeping *into the method itself*
(`Method.propose → Iterable[Proposal]` + a budget/disjunction DSL — the heaviest thing on the
roadmap, and one the red-team already flagged as over-built).

This refactor **fully externalizes sweeps**: `run()` becomes a pure atomic function; a separate,
thin **aggregate** app owns every cross-run concern (the sweep, the head-to-head scorecard, the
shared-scale before/after). `joblib` caching of the expensive derivations makes an external sweep
as cheap as an in-process family — so the atomic flow stays trivial and the Slice-2 DSL disappears.

## Slicing & sequencing (F1–F4)

**Already shipped** (this spec's §1 block selection): the `block_ids` early-filter at the `Source`
— a top-level interpolated scalar → constructor → filter before sjoin/build. The rest is delivered
in four ordered, individually-shippable slices, each with its own plan + execution (F2–F4 designs
refined just-in-time):

- **F1 — Atomic pure `run()` + render emitter.** §1 (drop the `method` **list → single**; remove
  render coupling from `run()`; `TopologyMethod` **local RNG**; `run()` pure, writes nothing) + the
  §2 **render emitter only** (`RenderConfig`, `emit(results, out_dir, cfg)`, `png`/`separate` —
  today's behavior) + §9 migration. **No emitter registry yet** — `main` calls the one render
  emitter directly; the `enabled_emitters` fan-out arrives in F4 with a second emitter.
  `format=webpage` / `side_by_side` deferred until needed.
- **F2 — L2 per-block persistent cache.** §6: joblib, content-addressed key
  `(block_id, source_content_hash, geos_version, proj_version[, params])`, separate before/after
  keys, `Block` carries `source_content_hash`. The "no double-build" enabler for F3.
- **F3 — Screen as a `run()` stage + city flagged-map + delete the standalone app.** §1's
  screen-stage subsection: a `Screen` stage before block iteration, default `IdentityScreen`
  passthrough, the S1 `DenseCompactScreen` plugged in; a **binary city flagged-map emitter** (all
  metro blocks drawn light, flagged blocks highlighted); **delete the standalone `reblock.screen`
  app** (migrate its detect + flagged-ids/visual output into the stage + emitters). F2's L2 cache
  makes the screen's fine-pass builds cache hits, so `run()`'s reblock build is a hit —
  screen-then-reblock is free of double-building. Delivers the **one-command end-to-end** (detect →
  reblock → render) + its README recipe.
- **F4 — Compare aggregate + sweep.** The §2 **emitter registry** + §3/§4 `reblock.compare` +
  scorecard emitter + L1 in-process reuse (§6). The topology-vs-peel head-to-head.

## 1. Atomic `run()` — pure, single of each

```python
def run(cfg: RunConfig | DictConfig) -> list[Result]: ...   # single data + method + eval (+ params)
```

- Loads one `Source`, and for each block (up to `max_blocks`) runs the one `Method` and scores it with
  **each** configured `Eval` → one `Result` per block, `Result.metrics` = tuple (one `Metrics` per
  eval).
- **The `method` LIST is dropped — single method** (the sweep axis, §4). The **`eval` list stays**:
  evals are cheap and computed together on the one proposal, and — per review — `Result` must carry
  the kcomplexity metrics for the render emitter *regardless* of which eval a scorecard tabulates, so
  `Result.metrics` **stays a tuple** (the earlier "collapse to a single `Metrics`" is reverted;
  `WeakDualKEval` emits no `fields`, so a single-metrics `Result` under `weakdual_k` would be
  un-renderable).
- **No rendering, no output-dir coupling.** `render_base`/`render_dir`/`_render_block` are removed
  from `run()` (rendering is an emitter, §2).
- **No RNG-of-its-own, and deterministic on repeat.** `TopologyMethod` currently seeds the *global*
  numpy RNG (`np.random.seed`); refactor it to a **local** `np.random.default_rng(seed)` passed into
  the builder so `run()` has no global side effect and repeats are bit-identical (needed for the
  purity claim; also removes a latent cross-method nondeterminism).
- **Block selection — filter early, at the source. (SHIPPED — kblock `Source`.)** A `block_ids` list (a top-level interpolated
  Hydra scalar, exactly like `${shapefile}`/`${alpha}` today) flows into the `Source` constructor;
  `region()` filters the blocks frame **before** the sjoin + per-block build loop, so a targeted run
  does O(k) work for k requested blocks (and the sjoin touches only their buildings) instead of
  building the whole region and discarding. **No `Source`-protocol change** — it is another
  constructor param, Hydra-set; `block_ids=None` means "all blocks" (today's behaviour). This early
  split is also what makes the L2 per-block cache **O(n)-not-2ⁿ** (§6): the split boundary is the
  individual block, so any `block_ids` subset reuses the same per-block cache entries.

  ```bash
  python -m reblock.run data=capetown method=peel block_ids=[ZAF.9.3.1_1_44882] render.enabled=true
  ```

- **Screen becomes a `run()` stage** (folds in the standalone `reblock.screen` app from the
  slum-detection S1 slice — `specs/2026-07-08-screen-slum-detection-design.md`). Add a `Screen` stage
  before block iteration: `run()` instantiates `cfg.screen` (default **`IdentityScreen`**, a
  passthrough that selects the configured `block_ids` / all blocks), calls `select() -> block_ids`,
  and those become the source's `block_ids`. This makes it **one entrypoint** — `Source → Screen →
  Method → Eval`, screen mandatory-but-defaulted — instead of a separate detect app. The reason it
  belongs *here* and not in S1: a real Screen's fine pass already builds its survivors (Voronoi+peel
  for `mean_depth`), and `run()` then builds the selected blocks to reblock them — the **L2 per-block
  cache (§6) makes that second build a cache hit**, so the screen-then-reblock unification is free of
  double-building only once L2 exists. S1 ships the standalone app as the interim; this stage
  supersedes it (delete the standalone app when this lands — migrate, don't keep both).

## 2. Outputs — pluggable emitters that consume `Result`s

Both kinds of output (the quantitative **scorecard** and the **visualizations**) are just consumers
of the same `Result`s. So an output is an **emitter**: `emit(results, out_dir, cfg) -> None`. The
aggregate computes the `Result`s once and fans them out to whichever emitters are enabled. `run()`
never emits — it returns `Result`s; emitters are called by the entrypoints on the objects they hold.

Each emitter is a **structured config** (dataclass) with its own `enabled` toggle and options — not a
flat list, because every output has real knobs:

```python
@dataclass
class ScorecardConfig:
    enabled: bool = True
    evals: list[str] = field(default_factory=lambda: ["kcomplexity"])   # which evals to tabulate
    sort_by: str = "delta_k"

@dataclass
class RenderConfig:
    enabled: bool = False            # opt-in: rendering is expensive (matplotlib, per block × proposal)
    format: str = "png"              # "png" | "webpage"
    layout: str = "separate"         # "separate" | "side_by_side"
```

- `render` (enabled) is today's `_render_block` logic lifted out of `run()`: per block a shared-`vmax`
  `{block}_before.png` + one `{block}_{proposal}_after.png` per proposal, laid out per `layout`, and
  `format=webpage` bundles them into one self-contained HTML page (the artifact style used this
  session).
- `scorecard` tabulates the chosen evals' metrics across the swept `Result`s (one row per
  block×method), the quantitative head-to-head.
- New emitters (CSV export, a budget-vs-k curve) are a ~20-line `emit` fn + a config dataclass —
  the extension point stays cheap and uncluttered.

## 3. Entrypoints — Hydra at the edges, pure functions in the middle

`run()` is a **pure function that ingests a config** — it is *not* a Hydra app per invocation. Hydra
lives only at the two entrypoints; the aggregate calls the `run()` *function* directly:

```python
def run(cfg: RunConfig) -> list[Result]: ...            # single data + single method + eval list. PURE.

# reblock.run:main    (@hydra.main)  — standalone: score/inspect one config, its own dir
results = run(cfg); [emit(results, out_dir, cfg) for emit in enabled_emitters(cfg)]

# reblock.compare:main (@hydra.main)  — the aggregate
results = [r for c in expand(cfg.sweep) for r in run(c)]
for emit in enabled_emitters(cfg):                       # scorecard + render (if enabled)
    emit(results, out_dir, cfg)
```

- **Only the two `@hydra.main` entrypoints get a Hydra output dir.** The `run()` calls the aggregate
  makes are plain function calls with **no dir of their own** — they return `Result`s. A skeleton
  per-atomic Hydra dir (just a log) would be a *symptom* that the atomic runner shouldn't be a Hydra
  app; it isn't.
- **Provenance lives at the sweep level:** the aggregate's config *is* the reproducible description
  of every point (the `sweep` spec + axis values), so swept runs need no per-run config snapshot.
- **Per-run diagnostics don't need Hydra:** the aggregate wraps each `run()` in try/except and logs
  failures with the offending config — better for debugging a sweep than hunting through N per-run
  dirs.

## 4. The aggregate (`reblock.compare`)

A thin second Hydra app whose config declares the sweep + the enabled emitters; orchestration is
plain Python (an ergonomic dataclass populated from `conf/compare.yaml`, §5):

```python
@dataclass
class CompareConfig:
    data: Any          = MISSING     # fixed base data source (a _target_ group)
    eval: list[Any]    = ...          # fixed eval list (scored on every point; Result.metrics is a tuple)
    points: list[Any]  = ...          # the sweep: an explicit LIST of per-method configs
    max_blocks: int    = 3
    scorecard: ScorecardConfig = ScorecardConfig()
    render:    RenderConfig     = RenderConfig()   # enabled=False by default
```

**The sweep is an explicit list of atomic method-configs, NOT a cartesian of a method-group against a
flat param dict** — because a flat `params` dict can't express "`budget` belongs to `peel`, `alpha`
to `topology`," and `topology × budget` would `TypeError` or silently double-count. Each point names
its method *and that method's own params* together:

```yaml
# conf/compare.yaml
points:
  - {_target_: reblock.methods.topology.TopologyMethod, alpha: 2}
  - {_target_: reblock.methods.peel.PeelReblocker, budget: 400}
  - {_target_: reblock.methods.peel.PeelReblocker, budget: 800}
```

- For each point, builds an atomic `RunConfig(data, method=point, eval=cfg.eval)`, calls `run()`
  **in-process**, collects the returned `Result`s, then runs the enabled emitters.
- **It reads return values, never artifacts from any Hydra output dir.** A `Result` carries `block`,
  `proposal`, and every eval's `metrics` (incl. kcomplexity `fields`), so nothing is scraped from
  disk and the render emitter always has the layers it needs — the robustness win.

## 5. Configuration — keep the current YAML + dataclass hybrid (structured-config migration deferred)

**Decision (post-review):** do **not** migrate to full Hydra structured configs / ConfigStore in this
slice, and do **not** delete the `conf/*.yaml` groups. The red-team showed the migration is premature
(it freezes a ~40-line surface right as it's about to churn — new source, emitter, sweep shapes) and
oversold (the fields most worth typing — `assumed_crs: CRS|int|None`, interpolated `${...}` values —
fall back to `Any`/`str` under OmegaConf, so it *relocates* the untyped island rather than closing
it). It also created the only cross-spec churn (kblock adds `conf/data/*.yaml` that this slice would
delete). Deferred to `docs/superpowers/backlog.md`, to be done once the surface stabilizes.

For now: keep today's pattern — `conf/{data,method,eval}/*.yaml` groups with `_target_` +
`hydra.utils.instantiate`, and the new `CompareConfig`/`ScorecardConfig`/`RenderConfig` as
**ergonomic dataclasses** (like the current `RunConfig`), populated from a `conf/compare.yaml`. The
one cleanup worth doing *within* this refactor (no full migration needed): simplify the
`RunConfig.__post_init__` flat-field→`_target_`-dict translation glue now that `run()` is atomic.

**Block selection wires as a top-level interpolated scalar** (§1): add `block_ids: null` to
`conf/config.yaml` (and `conf/compare.yaml`), referenced as `block_ids: ${block_ids}` in each
`conf/data/*.yaml` group that supports it — the same interpolation pattern `${shapefile}`/`${alpha}`
already use, so a CLI `block_ids=[...]` flows into the selected `Source` with no group-qualified
override.

## 6. Efficiency — in-process reuse (L1) + per-block persistent cache (L2)

Two complementary layers. L1 is the within-one-process fast path (no serialization); L2 persists the
expensive **pure** derivations across sweep invocations. Both key on the **individual block**, never
on the filter or the run — the O(n)-not-2ⁿ principle.

**L1 — in-process reuse (within one sweep).** Because §3/§4 run the whole sweep in one process, the
aggregate **loads the `Source` once**, materializes each `Block` once (iterate `region.blocks`
a single time), and computes the shared **method-independent** derivations once per block —
`parcel_adjacency`, the `roads=None` "before" access layers, the "before" geometric distances, the
topology graph build — into a plain in-process `dict` keyed by the block, then fans out across
methods/evals. No pickling, no disk. This is the fast path for repeated touches within a single sweep.

**L2 — per-block persistent cache (across sweeps / invocations).** The measured cost breakdown (Cape
Town, 700–900-parcel blocks) shows the tessellation is **not** the bottleneck: `_voronoi_parcels`
≈ 20 ms (~6% of a peel block-run), while the eval (`parcel_access_layers` ×2 + geometric Dijkstra +
connectivity ≈ 210 ms) and the method (`propose` — 110 ms peel; **minutes** for topology) dominate.
So the persistent cache targets the expensive **pure** derivations, keyed per block:

- block build / Voronoi (≈20 ms each; ≈6–7 s across a ~300-block region)
- the method-independent **before**-derivations (`access_before`, `geometric_before`) — recomputed
  for *every* method/param on the same block, so the biggest sweep win
- the **method proposal** per `(block, method-params)` — trivial for peel, but turns a topology rerun
  from minutes into an instant hit

All are pure and deterministic given a fixed GEOS, so all are safely cacheable.

**The key is per-block and content-addressed, never per-filter** — cache-unit key =
`(block_id, source_content_hash, geos_version, proj_version[, method-params])`:

- **`block_id` + `source_content_hash`** (a hash of the source parquet(s), computed once at load and
  carried on the `Block`) — so a block built under filter `{X,Y,Z}` and under filter `{X}` hit the
  *same* entry. n blocks → n keys, reused across all 2ⁿ filters. This is why block selection filters
  *early* (§1): the split boundary is the block, so the cache unit is the block. (Owner's lean:
  `block_id` + whole-source hash over per-geometry-WKB hashing — simpler, and kblock ids are stable;
  trade-off is coarse invalidation, below.)
- **`geos_version` AND `proj_version` in the key** — parcel counts are GEOS-sensitive (the
  pinned-value test is GEOS-fragile) and the cached derivations run on **already-reprojected**
  geometry, so a GEOS *or* PROJ upgrade changes the result; joblib keys on args + func-source but
  **not** on native-library versions. Because the owner's key is a **raw source-file hash** (not the
  reprojected geometry), neither library version folds in on its own — **both must be explicit**.
  (Keying on the reprojected geometry-WKB instead would auto-fold PROJ — a change reprojects to new
  coords → new hash → clean miss — but that is the finer/costlier alternative rejected above.)
- **separate keys for `before` vs `after` derivations** — the original design collapsed
  `parcel_access_layers(block, None)` and `(block, roads)` to one key (→ `delta_k=0`); keying the
  before-derivations on the block alone and the after-derivations on `(block, proposal)` keeps them
  distinct.

**This supersedes the earlier "drop joblib" decision.** That decision was right for the *original*
design (a positional `block_id = f"{region_id}_{k}"` key that served stale geometry, the before/after
collapse, a hand-bumped version tag) and for a seconds-fast toy pipeline. With real two-city data
(region builds cost seconds, sweeps are real), the corrected content-addressed / version-keyed design
above, and the owner's "cache anything that can be safely cached" directive, the per-block **L2 cache
is in-scope for this slice**. Trade-off, stated: keying on the whole `source_content_hash` invalidates
*all* blocks when the source file changes (coarse but always safe); per-geometry-WKB hashing would
invalidate only changed blocks (finer, at a WKB-hash-per-lookup cost) — we take the coarse, simpler
key. Revisit only if partial-source edits become common.

## 7. What this deletes / changes

- `run()`'s **`method` list → single**; `render_base`/`render_dir`/`_render_block` **removed from
  `run()`**. `run()` becomes a pure function. `Result.metrics` **stays a tuple** (eval list retained).
- `TopologyMethod` global `np.random.seed` → a **local `default_rng`** (§1).
- **Kept as-is:** `conf/*.yaml` config groups, `_target_`+`instantiate` (structured-config migration
  deferred, §5). Persistent caching dropped for in-process reuse (§6). The `RunConfig.__post_init__`
  glue is simplified (not fully deleted) within the atomic refactor.
- **The peel-reblocker Slice 2 `Method.propose → Iterable[Proposal]` migration and the budget /
  disjunction DSL are deleted from the roadmap.** A budget becomes an ordinary `PeelReblocker`
  param; the aggregate sweeps it via an explicit per-method point; a budget-vs-k *curve* is a
  post-hoc assembly from the swept `Result`s. `Method.propose` stays `-> Proposal`. **Note (from the
  peel red-team):** the hard part — downward-closed / monotonic-prefix budget truncation — is
  *retained work inside `PeelReblocker.propose`*, not deleted; only the `Iterable` return type + the
  `any_of` DSL go away. A curve is only coherent if budget points nest (roads@400 ⊂ roads@800), so
  `propose` must truncate a deterministic prefix — add a monotonic-nesting test.
- **New:** an emitter layer (`ScorecardConfig`/`RenderConfig` + `emit(results, out_dir, cfg)` fns,
  render lifted from `run()`); `src/reblock/compare.py` (the aggregate) + `conf/compare.yaml`.
- **New:** `block_ids` selection — a top-level interpolated scalar → `Source` constructor → early
  filter before sjoin/build (§1); `Block` carries a `source_content_hash` computed once at load.
- **New:** the L2 per-block persistent cache (§6), keyed
  `(block_id, source_content_hash, geos_version, proj_version[, params])` — supersedes the earlier
  drop-joblib stance for the corrected, content-addressed design.

## 8. Testing

- `run()` is pure: a given `cfg` yields deterministic `Result`s and **writes no files**; repeat runs
  are bit-identical (guards the topology local-RNG fix).
- Emitters: `scorecard` produces the expected table from hand-built `Result`s; `render` (enabled)
  produces the expected PNGs / HTML per `format`+`layout`; a disabled emitter produces nothing.
- `conf/compare.yaml` composes and instantiates (the `points` list → typed methods; the `eval` list
  → typed evals) — a wiring test like today's `test_hydra_compose_wires_config_groups`.
- Aggregate: a 2-point sweep on a fixture → a scorecard with 2 rows + one shared-`vmax` before and
  2 afters; assert the shared per-block derivation is **computed once** across the two `run()` calls
  (in-process reuse — e.g. a call counter on the derivation).
- Block selection: `block_ids=[b]` yields only block `b`'s `Result`(s), and the source **builds only
  that block** (assert via a build/Voronoi call counter, not just that others are absent from the
  output) — the early-filter guarantee, not a post-filter. `block_ids=None` → all blocks (unchanged).
- L2 cache: two independent invocations over overlapping blocks return **identical** `Result`s and the
  second **skips** the cached derivations (a recompute counter); a filter `{X}` reuses the entry built
  under `{X,Y}` (same key → one compute total); a simulated `geos_version`/`proj_version` bump forces a
  **clean miss** (recompute, no stale hit); before- and after-derivations do **not** collapse to one key.

## 9. Migration note (owner's "migrate, don't accommodate")

Drop the `method` list (single method), extract rendering into emitters, delete the Slice-2 DSL plan,
localize topology's RNG — no dual path, no compat shim. `test_run.py`'s multi-method tests migrate:
the "one before, N afters" comparison behaviour moves to aggregate/emitter tests; the single-method
atomic behaviour stays a `run()` test. `Result.metrics` **stays a tuple** (eval list retained), so
its `.metric(eval, key)` API and consumers are unchanged. `conf/*.yaml` and the structured-config
migration are **kept/deferred** (§5), so there is no config-consumer churn this slice.

## Sequencing

The kblock source shipped first (as recommended), so the real two-city, two-method data is already
in place — this refactor lands with an immediate payoff (the Cape Town topology-vs-peel head-to-head
as a `reblock.compare` run at F4). Execution order is the four slices above: **F1 → F2 → F3 → F4**
(F2 precedes F3 so the screen's fine-pass builds are L2 cache hits when `run()` reblocks — no
double-building; F4 last, orthogonal to the F3 end-to-end/visual payoff).
