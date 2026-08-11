# Arterial engine productionization: injected engines, a real anchor cap, access at region scale

**Date:** 2026-08-11
**Branch:** `continuum-permeability`
**Status:** design, approved section by section; not yet implemented
**Supersedes:** handoff §4 ("The productionization decision (not made — deliberately)")

## Why

`GreedyArterialReblocker` cannot run the access objective at region scale, so
`greedy_arterial_access_displacement` is absent from all three multiblock example variants. The
tier-2 shortlist greedy that fixes this lives in `scripts/perf/shortlist_greedy.py`, outside the
shipped method.

Measured basis (`notes/2026-08-11-max-anchors-is-a-region-scale-win.md`, six regions):

* tier-2 shortlist makes region-scale access finish at all — 11.6 h unfinished → ~80 min uncapped;
* `max_anchors` gives a further **7.6× median** (2.5–12.2×, 6/6 regions), no detectable quality
  difference at matched displacement;
* together, roughly **330×** on the original problem.

## Scope

**In.** The engine layer and the candidate-generation layer: engine selection, candidate policy,
chord realization, the `max_anchors` semantics, config migration, and the access rollout.

**Out, deliberately.**

* `objective` and `cost` stay strings. They branch inside `eval_candidate` — the per-candidate
  function executed across the fork pool — and `cost` fans into `budget.py` (6 + 5 branch points).
  Converting them is a scoring-core change needing bit-identity validation across every
  objective × cost combination, and `cost="displacement_fast"` is explicitly provisional ("Kept as
  a VARIANT until measured to win or lose"), so typing it now may bake in something that gets
  deleted.
* **Stochastic restarts** (handoff §3) — the highest-value open thread, but unvalidated (n=8
  blocks, two `pool` points, one displacement budget, and no decision about what a restart selects
  on when the method reports two metrics). The `ArterialEngine` Protocol leaves room for a
  `StochasticShortlistEngine` without pre-building it.
* **`max_anchors` for the directness/CELF methods.** The cap changes which candidates exist, so it
  changes directness outcomes too, and every measurement here is `objective=access`. Directness
  keeps `max_anchors: 0`.
* **Tier 2 for directness.** Not applicable: `first_order_score` computes
  `weights = depths**2 - 1`, which *is* the access objective's benefit (`budget.access_burden` is
  Σd²). Directness already has CELF, which is valid there because directness is submodular — the
  reason access needed tier 2 at all is that CELF is invalid for it (measured diverging 6/6,
  slower in 4/6).

---

## §1. Architecture

`arterial.py` (549 lines, mixing primitives, the exact engine and the public method) and
`arterial_lazy.py` become one package, one module per concern:

```
reblock/methods/arterial/
  __init__.py     re-exports the public surface
  primitives.py   _anchor_points, _candidate_chords, _deep_targets,
                  _explode, _merge, _planarize, _snap_graph
  realize.py      ChordRealizer · SnapToBoundary(lam) · IdealChord   (+ _snap)
  scoring.py      _score, _StepState, _STEP_STATE, eval_candidate, _best_candidate
  policies.py     CandidatePolicySpec/CandidatePolicy · Grow · Fixed · Faithful
  engines.py      ArterialEngine · ExactEngine · LazyEngine · ShortlistEngine
  reblocker.py    ArterialIdentity · GreedyArterialReblocker
```

Dependencies flow one way: `primitives → realize → scoring → {policies, engines} → reblocker`.
`arterial_lazy.py` disappears, its contents splitting along that seam.

`__init__.py` re-exports, so `_target_: reblock.methods.arterial.GreedyArterialReblocker` keeps
resolving and engines/realizers get short config paths.

### Three injected strategies

All are frozen dataclasses: they pickle into the fork pool and hash into the cache key.

```python
class ArterialEngine(Protocol):                  # replaces `lazy`
    def run(self, block, *, objective, cost, realizer, n_anchors, top_k,
            max_roads, half_width_m, workers, max_anchors) -> GeoDataFrame: ...
    @property
    def identity(self) -> EngineIdentity: ...

class ChordRealizer(Protocol):                   # replaces `mode` + `lam`
    def realize(self, chord, sg) -> LineString | None: ...
    @property
    def identity(self) -> RealizerIdentity: ...

class CandidatePolicySpec(Protocol):             # replaces `candidate_policy`
    def build(self, block, streets, n_anchors, top_k, adj, max_anchors) -> CandidatePolicy: ...
```

| Protocol | implementations |
|---|---|
| `ArterialEngine` | `ExactEngine()` · `LazyEngine(policy, rescore_every=0)` · `ShortlistEngine(k=512, threads=8)` |
| `ChordRealizer` | `SnapToBoundary(lam=2.0)` · `IdealChord()` |
| `CandidatePolicySpec` | `Grow()` · `Fixed()` · `Faithful()` |

`GreedyArterialReblocker` then keeps only what is shared and unconditional: `objective`, `cost`,
`road_width_m`, `n_anchors`, `top_k`, `max_roads`, `max_anchors`, `workers`, plus `engine` and
`realizer`. **Deleted:** `lazy`, `candidate_policy`, `rescore_every`, `mode`, `lam`.

### Why a spec/instance split for policies

`_GrowPolicy` closes over the block, adjacency and seed candidates — per-block state, not
configuration. Config injects a *spec* carrying only config-level parameters; the engine calls
`build(block, …)` once per proposal. The three existing policies already share an identical
interface (`initial()` / `after_commit(committed, step)`), so `_make_policy`'s string factory (with
its `ValueError` fallback over a closed set of three) converts almost for free.

### Two `mode` reads that must become behaviour, not a predicate

`mode` is read outside `_snap` at `arterial.py:301` (the buildable-only benefit branch) and `:419`
(`step = ctx.step(base) if … mode == "buildable"`). These must become realizer *methods* — the
realizer supplies the step context — not a `realizer.is_snapping` query. A predicate would move the
string dispatch rather than remove it.

---

## §2. Identity and cache invalidation

```python
@dataclass(frozen=True)
class ArterialIdentity:
    objective: str
    cost: str
    corridor_key: float          # road_width_m when cost ∈ {displacement, repulsion} else 0.0
    max_roads: int
    n_anchors: int
    top_k: int
    max_anchors: int
    realizer: RealizerIdentity   # SnapToBoundary(lam) | IdealChord()
    engine: EngineIdentity       # ExactEngine() | LazyEngine(policy, rescore_every)
                                 #              | ShortlistIdentity(k)
```

Both are **closed union aliases**, not `object`: the set is known at authoring time, so per the
static-checkability directive it gets a declared type a checker can verify exhaustively.

**The identity rule, stated once.** Every strategy exposes `.identity`. It returns **`self` when
every field it carries affects the proposal**, and a reduced frozen dataclass when some field does
not. Exactly one strategy needs the reduced form: `ShortlistEngine.threads` is a parallelism knob —
the same category as the existing `workers` — so `ShortlistEngine.identity` returns
`ShortlistIdentity(k)`. Everything else is pure proposal-affecting config and is its own identity.

This is why there are no mirror classes. A `SnapIdentity(lam)` shadowing `SnapToBoundary(lam)` would
be parallel structure that has to be kept in sync by hand, which is a drift hazard for no gain.

The alternative — hoisting `threads` onto the reblocker next to `workers` — would reintroduce
exactly the conditionally-relevant field this work removes, since only one engine uses it. So
`threads` stays where it is used.

**This fixes a live defect.** `lam` is used only in `_snap`, which runs only when the mode snaps —
yet `lam` is unconditionally in `ArterialIdentity` today, so two aspirational configs differing only
in `lam` get different cache keys for provably identical output. `IdealChord()` carries no `lam`, so
they now collide correctly. The code already has precedent: `corridor_key` derives to `0.0` when
`cost` makes `road_width_m` irrelevant.

**What invalidates.** The field change alters the pickled key, so every cached arterial proposal
misses and examples fully recompute. Output changes only where behaviour changes:

| method | key changes | roads change |
|---|---|---|
| directness (CELF), aspirational | yes | **no** — identical roads, recomputed |
| access | yes | **yes** — moves to `ShortlistEngine` + cap |

`max_anchors` needs no data migration: every shipped config sets `0`, which means "off" under both
the old and new readings.

---

## §3. `max_anchors` becomes a real cap; `k` defaults to 512

### The cap

Today `max_anchors > 0` *replaces* the per-vertex anchor family with arc-length samples and returns
early, before the vertex loop. It is a mode switch wearing the costume of a maximum: at `cap=128`
you get 129 anchors where uncapped gives 35–48, so it **pessimises** at block scale (measured 1.69×
at cap=128, 4.19× at cap=256).

```python
def _anchor_points(network, n, max_anchors=0):
    lines, total = _merged_lines(network)
    full = _vertices_and_samples(lines, total, n)              # today's uncapped branch, verbatim
    if max_anchors <= 0 or len(full) <= max_anchors:
        return full
    sampled = _arclength_samples(lines, total, max_anchors)    # today's capped branch, verbatim
    return sampled if len(sampled) < len(full) else full
```

Both helper bodies are today's code unchanged, so **the binding behaviour is exactly what was
measured**. The composition adds the guarantee `len(result) ≤ len(full)` — never a pessimisation,
including near the threshold where the sampled family can come out larger than the uncapped one.

Building `full` even when the cap binds costs 961 points at region scale against the ~462k candidate
enumeration that follows: negligible.

Consequences, both measured:

| scale | uncapped anchors | cap=128 | effect |
|---|---|---|---|
| region 0 (11k parcels) | 961 | binds (186) | as measured — 7.6× median across 6 regions |
| blocks (n=50–110) | 39–48 | no-op | the 1.69× pessimisation disappears |

This is why the **access methods need only one preset** rather than a block-scale and a region-scale
pair: `max_anchors: 128` is now correct at both scales, because the parameter finally means what its
name says. (Directness still keeps `max_anchors: 0` — see Scope.) The old early-return path is
deleted, not kept behind a flag.

### `k = 512`, `threads = 8`

`k` is already honest — `FirstOrder` returns everything when `len(chords) ≤ k`, so it no-ops rather
than pessimising.

512 is the value every *region* result was measured at, and the saturation check bounds it only from
above: uncapped at 512/1024/2048/4096 produced a bit-identical network (perm 0.4536 at all four
rungs), so overshooting costs nothing in quality. The unmeasured direction is downward. Choosing 128
would optimise block scale — already cheap — by extrapolating on the scale where cost matters.

**Open, and backlogged** (`backlog.md`, "Shortlist `k` at multiblock scale"): capping removes ~40× of
the enumeration term, so `k` becomes the dominant lever *after* this ships. A `k` sweep with the cap
on, at multiblock scale, is the follow-up.

`threads=8` is the measured optimum: 354.9 s at 1 thread, 104.3 s at 8, degrading to 134.0 s at 16
(memory-bandwidth bound). At block scale it is a no-op by construction — a few thousand candidates
is one chunk.

**Note against an earlier claim in this work:** the "~5× at block scale" figure for tier 2 was
measured at `k=128`. At `k=512` the shortlist still binds (block candidates run 1,272 → 4,158 across
steps) so it buys speedup, but less. The implementation plan should measure it rather than quote a
number.

---

## §4. Config surface and rollout

Migration is **9 sites**: 3 files in `conf/method/` and 6 inline entries in
`conf/compare_config.yaml`. The example configs define their own `all_methods` but none construct an
arterial method.

| before | after |
|---|---|
| `mode: buildable` | `realizer: {_target_: …arterial.SnapToBoundary}` |
| `mode: aspirational` | `realizer: {_target_: …arterial.IdealChord}` |
| `lazy: true, candidate_policy: grow, rescore_every: 0` | `engine: {_target_: …arterial.LazyEngine, policy: {_target_: …arterial.Grow}}` |
| *(field absent ⇒ exact)* | `engine: {_target_: …arterial.ExactEngine}` |
| access methods | `engine: {_target_: …arterial.ShortlistEngine, k: 512}` **+ `max_anchors: 128`** |

Defaults stay conservative — `engine=ExactEngine()`, `realizer=SnapToBoundary()` — matching today's
`lazy: false` / `mode: buildable`.

**The rollout.** The three multiblock variants currently run
`[clearance_looped, euclidean_grid, resistance_lp, cycle_native]` — no arterial method at all. The
unblock is adding `greedy_arterial_access_displacement` to those three lists.
`method_comparison` already runs it and switches engine in place.
`greedy_arterial_access_repulsion` is migrated but is in no variant's list, so no rollout change.

**Regeneration cost.** Everything arterial recomputes; directness and aspirational reproduce
identical roads. The new cost is the access method appearing in 6 multiblock examples (3 variants ×
2 cities). At `max_roads: 15` with the cap, region 0 measured 9.7 min, but region cost varied
2.5–12 min at 8 roads across six regions — budget **roughly 1–2 hours added**, not a precise figure.

Published numbers are free to move; no before/after announcement is required.

---

## §5. Testing

### Behaviour preservation: the net already exists

The refactor must not change any road. These existing tests are that oracle and need only
import/construction updates:

* `test_arterial_proposal_wkt_unchanged` — pinned WKT regression
* `test_arterial_parallel_matches_reference_1808` — pinned reference block
* `test_arterial_parallel_geometry_bit_identical`, `test_greedy_is_deterministic`
* `test_lazy_faithful_rescore1_equals_exact` — the existing bit-identity oracle between the lazy and
  exact paths, which carries over to `LazyEngine` ≡ `ExactEngine`

### The semantics change touches only untested ground

Checked, not assumed. `test_anchor_points_max_anchors_caps_and_default_matches_uncapped` uses a
40-vertex fixture with `cap=8`: uncapped gives 47 anchors, so the cap binds and the new semantics
returns the sampled set unchanged. **The test passes as written** — its author chose the fixture "so
max_anchors actually bounds the count instead of coincidentally landing on it."

The non-binding case is what changes, and nothing covers it. Today it pessimises silently: on that
same fixture, cap=64 → 65 anchors, cap=128 → 128, cap=256 → 256, against uncapped's 47.

### New guards, each with its break-it proof

| guard | proof it guards something |
|---|---|
| never-pessimise: `len(anchors(cap)) ≤ len(anchors(0))` across a cap sweep | **fails on `main` today** |
| cap above the anchor count is a no-op: `anchors(net, n, 10_000) == anchors(net, n, 0)` | **fails on `main` today** |
| `ShortlistEngine(k)` with `k` above every step's candidate count ≡ `ExactEngine`, WKT-for-WKT | delete one per-step setup line (`committed_disp`, `base_val`) from the shortlist path → diverges. This is what `control_check.py` was built to catch, promoted from script to test |
| `lam` must not enter identity under `IdealChord` | **fails on `main` today** — `lam` is unconditionally in `ArterialIdentity` |
| `ShortlistEngine.threads` must not enter identity | add `threads` to `ShortlistIdentity` → fails |
| swapping realizer changes the roads | if `realize()` ignored its config, both realizers would agree → fails |

Three of six fail on `main` right now, which is the strongest available proof; run them against
`main` before writing the implementation.

### Not tested, deliberately

That shortlist quality matches exact on real blocks. That is a measurement, it lives in the notes
with its intervals, and asserting it in CI would be flaky — this greedy's argmax flips under a 1e-10
perturbation of its own gains.

---

## Suggested phasing

Six phases, each independently verifiable, so the plan has real checkpoints rather than one big
landing. Only the last changes any output.

| # | phase | gate |
|---|---|---|
| 1 | Package split, pure motion. Move code, update 16 files' imports, delete `arterial_lazy.py`. No signature changes. | Full suite passes **unchanged**. The pinned WKT and reference-block oracles are what prove nothing moved. |
| 2 | `max_anchors` cap semantics. Independent of the injection work. | The two never-pessimise / no-op guards (currently failing on `main`) pass; the existing `max_anchors` test still passes as written. |
| 3 | Inject `ChordRealizer`; delete `mode` and `lam`. | The `lam`-under-`IdealChord` identity guard (currently failing on `main`) passes; realizer-swap sensitivity guard passes; buildable WKT oracles unchanged. |
| 4 | Inject `ArterialEngine` + `CandidatePolicySpec`; delete `lazy`, `candidate_policy`, `rescore_every`. | `test_lazy_faithful_rescore1_equals_exact` still passes — the existing lazy≡exact oracle. |
| 5 | Add `ShortlistEngine`; delete `scripts/perf/shortlist_greedy.py` and repoint the harnesses. | Non-binding-`k` ≡ `ExactEngine`, WKT-for-WKT (`control_check.py`'s oracle, promoted). |
| 6 | Config migration (9 sites), access rollout into the 3 multiblock variants, examples regeneration. | Directness and aspirational examples byte-identical; access examples change; `pixi run check` green. |

Phase 3 is where `ArterialIdentity` first changes, so the derive cache starts missing there. That is
harmless for tests — the cache is content-addressed, so a key change is a miss, not a failure — and
regeneration is deferred to phase 6 where it is the point.

Measure `k=512`'s block-scale speedup during phase 5, since the "~5×" figure elsewhere in this work
was taken at `k=128`.

## Consequences for `scripts/perf`

The split **shrinks** the harness tree rather than growing it:

* `shortlist_greedy.py` is **deleted** — once `ShortlistEngine` exists it is a duplicate of
  production logic, the same drift hazard that kept `arterial_incremental.py` untracked. The
  measurement harnesses call the production engine instead.
* `control_check.py`'s oracle becomes a test.
* `selectors.py` keeps the research arms (`RandomSample`, `StochasticFirstOrder`) that are not
  production; `FirstOrder`'s scoring moves into `ShortlistEngine`.
* 10 harnesses need import updates.

## Decisions log

| decision | rationale |
|---|---|
| Injected `ArterialEngine`, not a 4th flag | the standing injection directive; three flags already jointly pick an engine |
| Engine owns only its own parameters | shared search parameters stay shared; `max_roads` is a budget, not an engine concern |
| Package split rather than a `TYPE_CHECKING` import dodge | owner: "always fine with more churn if the end result is cleaner" |
| A strategy is its own identity unless it carries a non-proposal-affecting field | avoids mirror classes (`SnapIdentity(lam)` shadowing `SnapToBoundary(lam)`) that must be hand-synced; only `ShortlistEngine.threads` forces the reduced form |
| `mode` + `lam` resolved now | `lam` is a conditionally-relevant flag that corrupts the cache key |
| `objective` + `cost` deferred | scoring-core change, fork-pool hot path, and `displacement_fast` is provisional |
| Per-step cap decision, take the smaller set | literal cap semantics, provably never a pessimisation, bounds cost exactly when cost grows |
| Binding behaviour left exactly as measured | subsampling the vertex set instead would preserve continuations and might be better, but it is unmeasured — not slipped in under a semantics fix |
| `k=512` | measured at region scale; saturation bounds it from above only |
| Cap on access methods only | every measurement is `objective=access` |
