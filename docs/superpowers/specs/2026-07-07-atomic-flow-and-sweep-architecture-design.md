# reblock — Atomic flow + externalized sweeps

**Status:** draft for review · **Date:** 2026-07-07 · **Branch:** `flow-refactor` (to be cut)

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

## 6. Efficiency — in-process reuse, no disk cache

**Decision (post-review):** drop persistent `joblib` caching. Three review lenses converged: the
spec's design could return silently-wrong results two ways (a positional `block_id = f"{region_id}_{k}"`
key that serves stale geometry when a fixture is regenerated; `parcel_access_layers(block, None)` vs
`(block, roads)` collapsing to one key → `delta_k=0`), the human-bumped code-version tag is
unreliable (even joblib's own func-hash misses transitive edits + the `STREET_TOL` constant), and for
a pipeline that runs a handful of blocks in seconds it's a dependency + correctness surface solving a
problem the project doesn't have.

Because §3/§4 run the whole sweep **in one process**, the "as cheap as an in-process family" goal is
achieved directly, with none of joblib's hazards:

- The aggregate **loads the `Source` once** and materializes each `Block` once (iterate the
  `region.blocks` generator a single time), rather than re-reading per atomic call.
- Per block, compute the shared derivations once — `parcel_adjacency`, the `roads=None` "before"
  layers, the topology graph build — into a plain in-process `dict` keyed by the block, then fan out
  across methods/evals. No pickling, no disk, no cache dir, no version tag, no content-vs-id hazard.

If cross-process / resumable sweeps are ever genuinely needed, add caching *then*, with
**content-addressed keys** (hash the block geometry once at load, carry it on the `Block`) and an
**automatic source-hash version** — not a positional id and a hand-bumped tag. Captured in the backlog.

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

## 9. Migration note (owner's "migrate, don't accommodate")

Drop the `method` list (single method), extract rendering into emitters, delete the Slice-2 DSL plan,
localize topology's RNG — no dual path, no compat shim. `test_run.py`'s multi-method tests migrate:
the "one before, N afters" comparison behaviour moves to aggregate/emitter tests; the single-method
atomic behaviour stays a `run()` test. `Result.metrics` **stays a tuple** (eval list retained), so
its `.metric(eval, key)` API and consumers are unchanged. `conf/*.yaml` and the structured-config
migration are **kept/deferred** (§5), so there is no config-consumer churn this slice.

## Sequencing

Independent of the kblock source (a `Source` is unaffected — it works with atomic `run()` unchanged).
**Recommended order: kblock source first, this refactor second** — kblock provides the real
two-city, two-method data that gives the aggregate/comparison something worth sweeping and rendering,
so the refactor lands with an immediate real payoff (the Cape Town topology-vs-peel head-to-head as a
`reblock.compare` run). Either order is safe.
