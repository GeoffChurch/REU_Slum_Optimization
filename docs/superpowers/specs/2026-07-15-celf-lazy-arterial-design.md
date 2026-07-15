# CELF / Lazy-Greedy Arterial — Design

## Goal

Make `GreedyArterialReblocker` tractable at regional road budgets by adding an opt-in **lazy
(CELF) greedy** that re-scores only the heap-top candidate per commit instead of re-scoring all
~4,600 candidates every step — without changing the scoring itself, and keeping the existing exact
path byte-identical.

## Motivation & validated prototype

Arterial's cost is `≈ (candidates) × (max_roads steps) × (per-score cost)`. Prior attempts to cut
the per-score cost (reduced-K, geometric shortcut proxies) all failed — directness gain is
graph-coupled and not cheaply approximable. CELF (Leskovec) attacks the *number of scorings*
instead, exploiting that marginal gains mostly diminish: keep a max-heap of candidate gains, and on
each commit pop the top and re-score only it (and any now-stale runners-up) until the top is fresh.

Prototype (`scratchpad/arterial_celf.py`, block 40972, fixed candidate set, 8 roads):

| | score_candidate calls | wall | directness |
|---|---|---|---|
| exact | 36,864 | 207 s | 0.390 |
| CELF | 4,685 | 19 s | **0.410** |

→ **7.9× fewer scorings, 10.9× faster, +5.4% directness.** Directness gain is *not* submodular
(the sequences differ), so the exact greedy is itself only a heuristic — CELF's path landed higher.
The speedup is `N·steps / (N + r·steps)` with `r ≈ 10` re-scores/step, so it **grows with the road
budget**: ~15× at max_roads=15, ~37× at 40, ~80× at 100 — precisely the regional regime where
arterial is currently intractable. It composes with the existing fork pool (parallelize the one
expensive initial full pass).

## Global constraints

- **Exact path byte-identical.** `lazy=False` must run today's `_greedy_arterials` unchanged; all
  existing byte-identical arterial goldens must still pass. This is a new *mode*, not a
  back-compat shim (no dual-path accommodation of old data — the owner's no-legacy directive is
  satisfied: one exact path, one new lazy path, both first-class).
- **No scoring change.** Lazy re-scores through the *same* `eval_candidate` / `_STEP_STATE` /
  frozen-context (`_BlockScoringContext`, `ctx.step`, `score_candidate`) machinery. CELF changes
  *which* candidates get scored each step, never *how* one is scored.
- **Compose with `workers`.** The fork pool parallelizes the initial full pass and any eager
  scoring of newly-added candidates; the per-commit pop-re-score loop is serial (too few to pool).
- **Determinism.** Lazy is deterministic; heap ties break by the same `wkt` total order arterial's
  `_best_candidate` already uses.

## Architecture

A new `_greedy_arterials_lazy(...)` sits beside the existing `_greedy_arterials(...)`.
`GreedyArterialReblocker.propose` dispatches on the `lazy` flag; the exact function is not edited.

Two collaborating pieces, both policy-agnostic:

1. **CELF core (shared).** A max-heap of entries `(-gain, wkt_key, candidate, scored_at_step)`.
   Per commit: rebuild the per-step frozen context (`base = merge(committed)`, `base_val`,
   `ctx.step(base)` / `_STEP_STATE`) exactly as the exact loop does; then pop-and-re-score the top
   until `scored_at_step == current_step`; commit that candidate; increment the step counter. The
   `wkt_key` in the heap tuple reproduces arterial's existing tie-break so a tie resolves
   identically to exact.
2. **Candidate policy (pluggable, 3 impls).** Called once per commit to evolve the candidate set +
   heap for the *next* step. This is the only thing that differs between the three modes; it fits
   the existing Strategy pattern (cf. `conf/substrate/`).

## The CELF algorithm & non-submodularity

Per commit at step `s` (with the step context frozen against the current `committed`):

```
while heap:
    (-g, key, cand, at) = heap.peek()
    if at == s: break                 # top already fresh under this committed set → true max
    heap.pop()
    g' , real = eval_candidate(cand)  # fresh gain under current context (same machinery as exact)
    heap.push((-g', key, cand, s))
if heap.empty() or heap.peek().gain <= 0: stop   # no improving candidate → terminate (== exact stop)
commit(heap.pop())
```

**Correctness caveat (documented, not hidden):** the heap short-circuit `at == s ⇒ true max` is
exact **iff** gains are monotone non-increasing as roads are added (submodularity), because a stale
gain is then an upper bound and a fresh top ≥ every stale entry. Arterial's directness gain is
*not* strictly submodular, so a candidate whose gain *rose* after a commit can stay buried under
its stale (lower) key and be missed. In practice this is benign-to-beneficial (the prototype beat
exact), but `rescore_every` (below) bounds it for the risk-averse. This behavior is spec'd, not a
bug: lazy mode is a distinct heuristic, not a bit-equivalent of exact (except at `rescore_every=1`).

## Candidate policies

Each policy exposes `initial(streets, targets0) -> candidates` and `after_commit(committed, step)
-> (added, removed)`; the engine eager-scores `added`, drops `removed`, and lazily carries the
rest.

- **`fixed`** — anchors are the street network only (vertices + arc-length samples computed once);
  deep-target spurs frozen at the initial deepest parcels. `after_commit` returns `([], [])`; the
  heap never changes after step 1. Biggest speedup, largest semantic departure.
- **`grow` (default)** — street anchors fixed as in `fixed`, but `after_commit` **adds** the new
  committed road's vertices as anchors (matching arterial's branching rule) → the resulting new
  through-road + spur candidates are eager-scored, and the fresh `_deep_targets(block, committed)`
  spurs are added; nothing is removed. The candidate set grows monotonically (no churn), so the
  ~4,600 street-anchor bulk stays lazy while only the *new* candidates are eager each step. Speedup
  sits between `fixed` (max) and `faithful`, near `fixed`, while preserving
  branching-off-committed-roads and live deep-targeting — the recommended sweet spot. (At very large
  `max_roads`, accumulated committed-vertex candidates raise the eager cost; `fixed` is the escape
  hatch for maximum speed there.)
- **`faithful`** — `after_commit` regenerates arterial's *exact* per-step candidate set
  (`_anchor_points(streets+committed, n)` including redistributed arc-length samples +
  `_deep_targets`), diffs it against the heap's current set: `added` = candidates now present but
  not in the heap (eager-scored), `removed` = heap candidates no longer generated (dropped),
  survivors stay lazy. Reproduces exact arterial's candidate semantics; the churn means ~half the
  candidates are eager each step, so the speedup is roughly halved.

## Config surface & identity

On `GreedyArterialReblocker` and `conf/method/greedy_arterial.yaml`:

```
lazy: bool = False                                  # False → exact path (byte-identical)
candidate_policy: "grow" | "fixed" | "faithful" = "grow"   # only used when lazy
rescore_every: int = 0                              # 0 = pure lazy; N = full re-score every N commits
```

All three are appended to `GreedyArterialReblocker.identity` (proposal-affecting → cache-key
correctness). When `lazy=False` the latter two are inert but remain in identity (harmless extra key
dimensions; keeps identity a pure function of the fields).

## Safety: `rescore_every`

`rescore_every = N > 0` forces, every `N`-th commit, a full re-score of the entire current
candidate set (heap rebuilt with fresh gains) — recovering the exact per-step argmax on that step
and erasing any accumulated non-submodularity drift. `0` trusts laziness (default; prototype-backed).
`1` re-scores fully every step and is therefore, per policy, identical to the exact greedy over that
policy's candidate set — the correctness oracle used in testing.

## Composition with the fork pool

- **Initial full pass** (step 1, score every candidate to seed the heap): pooled when
  `workers > 1` and candidate count ≥ `_PARALLEL_THRESHOLD`, reusing the exact loop's fork-context
  `ProcessPoolExecutor` path over `eval_candidate`.
- **Eager scoring of `added`** candidates (grow/faithful `after_commit`): pooled by the same rule.
- **Per-commit pop-re-score loop**: serial (a handful of candidates; pool spawn/IPC would dominate).

## Testing

1. **Exact untouched.** All existing byte-identical arterial goldens pass with `lazy=False`
   (the exact function is not edited). Strongest guard.
2. **CELF engine correctness.** `lazy=True, rescore_every=1, candidate_policy="faithful"` produces
   the **identical** road sequence to the exact path on a small block (40972) — proving the heap
   bookkeeping + faithful reconciliation replicate exact, isolating engine bugs from
   submodularity effects.
3. **Speedup + quality regression.** `lazy=True, rescore_every=0` on 40972: assert materially fewer
   `score_candidate` calls than exact (instrument a counter) and final directness within a
   tolerance (e.g. ≥ exact − small ε; the prototype exceeded exact) — a guard, not byte-exact.
4. **Policy behavior.** `fixed` heap size constant after step 1; `grow` heap size
   non-decreasing; both yield valid proposals.
5. **Determinism.** Two runs of the same lazy config give identical output.
6. **Identity.** Golden identity tests updated for the three new fields (as done previously for
   max_roads/n_anchors/top_k/lam).

## Non-goals

- No change to arterial's scoring, objectives, modes, or the exact path.
- No new approximation of the per-candidate score (that direction is a settled dead end).
- Not attempting to make lazy bit-identical to exact (except the `rescore_every=1` oracle) — lazy
  is an intentionally distinct, faster heuristic.
