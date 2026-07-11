# Arterial frozen-context performance refactor — Design

**Status:** draft for review · **Date:** 2026-07-11

**Goal:** Make greedy-arterial scoring — and the shared efficiency metric and the compare
cost-benefit curves that share its core — **10–50× faster**, by freezing the per-block constants
that are currently recomputed on every candidate and migrating the hot distance path from
networkx to `scipy.sparse.csgraph`, **with every metric value preserved to floating-point
tolerance.** This unblocks the `dense_cluster` flagship (a real ~150-parcel region is currently
intractable — 83 parcels times out at 15 min) and makes the compare curves fast.

**Architecture:** A single shared `_BlockScoringContext` in `budget.py`, built once per block,
that freezes the base street graph (as a scipy CSR + node→index map), the street edge geometry
and each parcel's projection onto it, the representative points, the sampled sources, and the K×N
euclidean distance matrix. `network_efficiency`, the compare's `_efficiency_factory`, and
`GreedyArterialReblocker`'s greedy loop all score through this one context; the greedy loop builds
it **once per block** and evaluates each candidate by appending only the trial road's edges and
entries. All shortest-path computation moves to `scipy.sparse.csgraph.dijkstra`; networkx is
removed from the hot path.

**Tech stack:** Python, numpy, scipy (`sparse`, `csgraph`), geopandas, shapely, Hydra, pixi,
pytest, mypy --strict, ruff.

---

## Global Constraints

- **Metric values are preserved exactly.** After every task, `network_efficiency`, the
  directness/efficiency curves, and their AUCs must equal the pre-refactor values to within
  `1e-9` (relative) on the equivalence fixtures below. This is a pure performance refactor; the
  metric definition does not change.
- **Monotonicity is preserved.** The compare's frozen-entry sweep (`_efficiency_factory`) must
  stay non-decreasing across cost-benefit prefixes (the property `_split_graph`'s docstring and
  the existing tests guarantee).
- **The recorded numbers stand.** `tests/test_budget.py` and `tests/test_region.py`'s recorded
  values — the compact-block AUC ordering (buildable 0.44 > aspirational 0.39 > dijkstra/mesh
  0.24), the `directness ∈ [0,1]` bound, and the deep-region ratio (~5.5×, assertion `>1.2×`) —
  must all still pass unchanged.
- **No dual code paths.** Per the project's "migrate, never accommodate" rule: the networkx graph
  build + K single-source dijkstra in the *metric-scoring* path are *replaced* by CSR +
  `csgraph.dijkstra`, not kept alongside. Two networkx uses are **explicitly exempt** and remain:
  `road_drainage` (one call per proposal, cold), and `_snap`'s `shortest_path` — kept because it
  must reproduce networkx's exact equal-cost-path tie-break so proposed geometry stays identical
  (see §4 / risk I4). The metric `_sampled_efficiency` + graph-build path must not import or call
  networkx.
- **Determinism.** Same block + roads → same score, byte-stable across runs (sorted sources,
  stable node indexing, deterministic entry tie-breaks).

---

## The problem (measured)

`cProfile` of one `GreedyArterialReblocker(mode="buildable", objective="directness").propose` on
block `DJI.3_1_1808` (**10 parcels**) — **43 seconds** under the profiler, 8 roads placed
(un-profiled wall time is **~19.5s** — cProfile inflates ~2×; the *proportions* below hold):

| component | time | share | called | why |
|---|---|---|---|---|
| `network_efficiency` (metric re-score) | 31.1s | 74% | **7,220×** | once per candidate |
| ├ graph construction (`_road_street_graph`+`_split_graph`+`_line_entries`+`_edge_lines`) | 21.3s | 51% | 7,220× | rebuilds the whole graph + re-projects entries every candidate |
| ├ `_sampled_efficiency` geometry (shapely legs + rep distances) | ~4s | 10% | 7,220× | recomputes constant home-to-home distances (1.48M shapely `.distance()` calls) |
| └ networkx dijkstra | 4.6s | 11% | 71,726× | K sampled single-source passes |
| `_planarize` | 6.8s | 16% | 7,221× | re-planarizes the trial road set every candidate |
| `_snap` (buildable only) | 3.8s | 9% | 7,463× | another networkx `shortest_path` per candidate |

Root cause: **the per-block constants (street graph, representative points, euclidean distances,
street-edge geometry) are recomputed for all 7,220 candidates**, and **51% of the time is graph
*construction*, not pathfinding.** networkx is only ~15–20% of the total — so a naive
"swap networkx for csgraph" is a minor win. Freezing the constants is the fix.

Scaling is worse than linear in parcels (10 parcels: 43s; 83 parcels: >15 min timeout; 236
parcels: >2h45m), because candidate count grows with anchors and each candidate's cost grows with
graph size.

---

## Design

### 1. `_BlockScoringContext` — the frozen core (in `budget.py`)

Built once from a `Block` (+ `tol`, `k`). Freezes, computed exactly once:

- `reps: list[Point]`, `rep_xy: np.ndarray (N,2)` — parcel representative points.
- `sources: list[int]` — the K deterministic sampled source parcels (`range(N)[::max(1,N//k)][:k]`,
  unchanged from `network_efficiency`).
- `src_euclid: np.ndarray (K, N)` — `euclid(rep_source_i, rep_j)`, the directness numerator.
- **Base street graph as CSR:** the `block.streets` edges only, `_rnd`-snapped, with **every
  parcel's street-edge projection point already injected as a split node** (so the street graph is
  fully "entry-ready"). Stored as: `node_index: dict[(x,y)->int]`, and the base edge arrays
  (`rows`, `cols`, `weights`) over those indices. A parcel that projects onto no street edge
  within `tol` has `base_street_entry[p] = None`.
- **Street entry base:** for each parcel, `(nearest_street_edge_distance, street_entry_node or
  None)` — the constant part of `_line_entries`, computed against street edges only.
- `street_edge_lines: list[LineString]`, `street_tree: STRtree` — for the incremental entry check.

Interface:

```python
class _BlockScoringContext:
    def __init__(self, block: Block, *, k: int = 40, tol: float = STREET_TOL) -> None: ...

    def score(self, roads: GeoDataFrame | None) -> tuple[float, float]:
        """(E, directness) for streets + `roads`, re-deriving entries against streets+roads
        (matches network_efficiency exactly)."""

    def score_frozen(self, roads_prefix, *, entry, splits) -> tuple[float, float]:
        """(E, directness) over streets + `roads_prefix` using FROZEN entries/splits
        (matches _efficiency_factory exactly — monotone across prefixes)."""
```

### 2. Fast `_sampled_efficiency` (csgraph + numpy)

Rewrite the inner engine to take the CSR graph, the source node indices, the destination entry
node indices (or `None`), the frozen `src_euclid`, and the per-parcel `legs`:

- **Batched dijkstra:** `dist = scipy.sparse.csgraph.dijkstra(csr, directed=False,
  indices=source_node_indices)` → a `(K, Nnodes)` distance matrix in one C call, replacing the
  `40×` networkx `single_source_dijkstra_path_length`. Unreachable = `inf`.
- **Vectorized metric:** for each sampled source `si`, gather `nd = dist[si, entry_node_index[j]]`
  as a length-N vector (destinations whose entry node is `None` or unreachable → masked out);
  `d = legs[si] + nd + legs` (vectorized); `inv_sum += Σ 1/d`, `dir_sum += Σ src_euclid[si]/d`
  over the valid mask. `legs[k] = euclid(rep_k, entry_xy_k)` computed once with numpy
  (`np.hypot`), replacing the shapely `Point`/`.distance()` per pair.
- Same pair set, same "unreached → 0", same coincident-entry (`nd=0`) handling, same
  `d>0` guard → identical results, `(0,0)` when no pairs.

The `[0,1]` bound and within-prefix monotonicity are unchanged (same math; distances from fixed
entries over a growing edge set are still non-increasing).

**CSR construction (exact-graph parity — see risk I3).** networkx dedups parallel edges, and
`_split_graph` *removes* a split edge's parent `(u,v)` before adding its colinear chain; a
COO→CSR build must replicate both or scipy's duplicate-summation doubles a weight (verified: a
repeated edge yields distance 10 where networkx gives 5). Rule: assemble the undirected edge set
as a dict keyed by the unordered node-index pair with **last-write-wins** weights, and when
injecting a split chain **delete the parent pair first**; then one symmetric `csr_matrix`.

### 3. Two-level incrementalism (the arterial win)

The set scored inside `_greedy_arterials` is `_planarize(committed + [real])`, and `committed`
grows by one road **per greedy step**. So the "constant" base is not the streets alone — it is
**streets + committed roads**, which changes once per commit (~`max_roads` ≈ 8–15 times), NOT per
candidate. Three layers:

- **Per block (frozen once):** the `_BlockScoringContext` — reps, `rep_xy`, sources, `src_euclid`,
  the street CSR + street-entry base + street edge geometry/tree.
- **Per greedy step (rebuilt on each commit, ~`max_roads` times — cheap):** a `StepContext` that
  extends the block base with the **committed roads** — node them into the graph (planarized, so
  crossings with streets and each other are noded), inject their entry splits, and re-derive each
  parcel's `(nearest_edge_distance, entry_node)` **against streets ∪ committed**. Cost
  O(N × (street+committed) edges), paid 8–15 times total, not 7,220.
- **Per candidate (incremental within a step):** only the single trial road `real` is new.
  **Entries:** for each parcel compare its `StepContext` nearest-edge distance with its distance to
  `real`'s few edges and take the closer → the candidate entries (the entry *node* is invariant to
  whether a committed edge is later split at a crossing — projection foot-point + `_rnd` unchanged
  — so the StepContext base stays valid). **Graph (R1 — must re-node crossings):** build the road
  subgraph from the incremental-planarize explode `unary_union([base_merged, real])` (§4), which
  re-nodes **committed×trial mid-span crossings** — a bare "append `real`'s edges to the step CSR"
  leaves the committed edge unsplit at the crossing and gives a wrong distance (measured: aspirational
  diagonals `dir=0.333` vs the true `0.444`). Harmless for buildable (snapped roads meet only at
  shared lattice vertices) but wrong for aspirational chords and real cadastral geometry. The
  per-candidate graph = frozen street CSR + this freshly planarized road CSR + entry splits; then
  `_sampled_efficiency`.

This is exactly equivalent to `network_efficiency(block, _planarize(committed+[real]))` because
streets ∪ committed ∪ trial is the full edge set the current `_line_entries` sees. Exact distance
ties between a `StepContext` edge and a `real` edge are measure-zero on continuous geometry and
the entry point is `_rnd`-rounded; the **incremental-scorer equivalence test** (Correctness) pins
parity against the full re-derivation on a block *with committed roads* — the single highest-risk
item (I2/C1).

### 4. `_snap` and `_planarize`

- **`_snap` (buildable) — keep networkx, kill the shapely.** `_snap` returns the *path geometry*,
  and on symmetric grids csgraph predecessor-reconstruction and networkx `shortest_path` pick
  *different equal-cost paths* (both valid), which would change proposed roads on grid fixtures
  (risk I4). To keep geometry **identical**, retain `nx.shortest_path` (same algorithm → same
  tie-break) but eliminate its dominant cost — the per-edge shapely `mid.distance(chord)` in the
  weight callback (the 2.38s `w`). Precompute all boundary-graph edge midpoints once as a
  shapely `Point` array (`edge_midpoints`, constant per block); per chord compute every edge's
  `dist(midpoint, chord)` in one call to **shapely's own vectorized ufunc**
  `shapely.distance(edge_midpoints, chord)` — verified **bit-identical** to the per-point
  `Point.distance(chord)` (a numpy point-to-segment reimplementation differs by ~50 ulp and flips
  ~1 path in 220, changing geometry — R2). Fold into an edge→weight dict and give
  `nx.shortest_path` a weight callback that only looks it up. Same weights, same path, identical
  geometry, minus the Python-loop shapely. (A csgraph `_snap` is deferred — it needs a tie-break
  matching networkx, which is fragile.)
- **`_planarize` (incremental):** planarize `committed` once per greedy step
  (`base_merged = unary_union(committed)`); per candidate compute `unary_union([base_merged,
  real])` and explode, instead of re-unioning the whole `committed + [real]` list each time. This
  also yields `real` noded against the existing network for the graph update in §3.

### 5. Caller migration + networkx removal

- `network_efficiency(block, roads)` → build a context, return `ctx.score(roads)`.
- `_efficiency_factory(block, roads_full, tol, k)` → build a context once, freeze entries/splits
  against the full graph exactly as today, return `lambda prefix: ctx.score_frozen(prefix, …)`.
- `_road_street_graph` / `_split_graph` / `_edge_lines` are absorbed into the context (street CSR
  build + split injection). Remove them once no caller remains, or keep private helpers the
  context uses — but no networkx `Graph` in the scoring path.
- Remove networkx from the **metric-scoring** path (`_sampled_efficiency` graph build + dijkstra).
  `road_drainage` (cold) and `_snap` (identical-geometry tie-break, §4) keep networkx explicitly.
  `tests/methods/test_arterial.py::test_aspirational_planarizes_crossings_into_true_intersections`
  imports `_road_street_graph` directly (asserts a degree-≥4 crossroads) — migrate it to whatever
  graph accessor survives, or to the context's CSR, as part of task 6.

---

## Correctness strategy

**Equivalence harness (new test module `tests/test_scoring_equivalence.py`):** capture the CURRENT
`network_efficiency` and `efficiency_directness_curves` outputs on a battery of fixtures BEFORE
refactoring (record them as literal expected values in the test, or via a pinned reference
implementation kept in the test), and assert the refactored path matches to `1e-9`. Fixtures:
- block `DJI.3_1_1808` (compact, 10 parcels) with: no roads; a dijkstra output; an arterial
  buildable output.
- the deep 3×4 / 4×3 synthetic region fixture from `tests/test_region.py`.
- a bare 2-point straight chord (the sparse-chord line-proximity case).
- a coincident-entry case (two parcels projecting to the same node).
Assert on: `E`, `directness`, every point of both curves, and both AUCs. Pin the expected values
as literals captured from the pre-refactor code (reference: `scratchpad/ref_values_1808.json` —
no-roads `E=0.026106 dir=0.326370`; dijkstra `E=0.023619 dir=0.239429 E_auc=0.015844
dir_auc=0.172189`; arterial-buildable `E=0.052905 dir=0.643840 E_auc=0.024198 dir_auc=0.273571`;
the road WKT is stored so the harness reloads the exact road sets without re-running `propose`).

**Incremental-scorer parity (pins the arterial's per-candidate path — the public-metric harness
above does NOT exercise it, so C1/I2 would ship silently otherwise):** for a block with **≥1
committed road**, assert the arterial's incremental per-candidate score equals
`network_efficiency(block, _planarize(committed + [real]))` for a battery of trial roads `real`.
The public harness only ever scores *full* road sets through full re-derivation, so it cannot
catch an incremental-entry bug (e.g. omitting committed roads); this test must. **The battery MUST
include an aspirational trial that crosses a committed road mid-span** (not just buildable grid
trials, which meet at shared vertices), or the R1 crossing-node bug slips.

**Invariants (existing tests must stay green, unchanged):** `directness ∈ [0,1]`, within-prefix
monotonicity, the recorded AUC ordering and region ratio.

**Per-task gate:** every task ends by running the equivalence harness + the touched existing tests,
AND records a fresh timing of the `DJI.3_1_1808` buildable propose (the 43s baseline) so the
speedup is measured, not assumed.

---

## Task decomposition (staged so each task independently preserves equivalence + is measured)

1. **Equivalence harness + numpy `_sampled_efficiency`.** Land the harness (pinning current
   values). Vectorize the inner loop with numpy (precomputed `src_euclid` + numpy `legs`), still
   on the networkx graph. Gate: harness parity + [0,1] + monotonicity; re-time.
2. **csgraph batched dijkstra.** Replace the `40×` networkx `single_source_dijkstra` in
   `_sampled_efficiency` with one `csgraph.dijkstra(indices=sources)` on a CSR built from the
   current graph. Gate: harness parity; re-time.
3. **`_BlockScoringContext` + migrate `network_efficiency` / `_efficiency_factory`.** Freeze the
   street CSR + reps + `src_euclid` + street-entry base + sources (using the dedup + split-removal
   CSR rules from §2). `network_efficiency` → `ctx.score` (re-derives entries); `_efficiency_factory`
   → `ctx.score_frozen` (builds its own per-prefix CSR from the frozen entries/splits; an isolated
   frozen source contributes 0). Arterial builds ONE context per block and, for now, scores each
   candidate by **full entry re-derivation over streets ∪ committed ∪ trial** (correct, not yet
   incremental). Gate: public harness parity + full suite; re-time (the 51% graph-construction cost
   collapses to once-per-block + a per-candidate CSR build).
4. **Two-level incremental scorer (the C1/I2 task — highest risk).** Add the per-step `StepContext`
   (streets ∪ committed, rebuilt on each commit) and the per-candidate trial-road delta: re-project
   only onto `real`'s edges, take the min vs the StepContext nearest edge, append `real`'s
   edges/entries to the step CSR. Gate: the **incremental-scorer parity test** (block with ≥1
   committed road, battery of trials) must match `network_efficiency(block, _planarize(committed +
   [real]))` byte-for-byte, PLUS public harness + full suite; re-time (the big drop —
   `_line_entries` is no longer O(N × all-edges) per candidate).
5. **`_snap` shapely-ufunc weights + incremental `_planarize`.** `_snap` keeps `nx.shortest_path`
   but computes per-chord edge weights via shapely's vectorized `shapely.distance` ufunc
   (bit-identical to `Point.distance`, so path unchanged). Planarize `committed` once per step; per
   candidate union only `[base_merged, real]`. Gate:
   arterial proposed roads **identical** on the fixtures (WKT match) + full suite; re-time.
6. **Remove networkx from the metric path + final measurement.** Delete the dead nx graph builders
   from the scoring path (nx remains only in `road_drainage` + `_snap`); migrate the
   `_road_street_graph` test import (§5). Confirm no nx in `_sampled_efficiency`/the context. Final
   re-measure on `DJI.3_1_1808` (19.5s real baseline) and the ~80-parcel calibration cluster
   `DJI.3_1_2914,2923,2925,2930` (currently >15-min timeout — the real proof); report speedups;
   whole-branch review.

---

## Alternatives considered

- **Naive csgraph swap only (no freezing).** Rejected: the profile shows networkx is ~15–20%;
  this leaves the 51% graph-construction and 16% planarize untouched. ~2× at best.
- **Memoize candidate scores across greedy steps.** The base changes each step, so most scores are
  invalid across steps; low yield versus the freezing approach, and fragile.
- **Grounded effective-resistance metric (north-star Piece 2).** A *different metric* (Laplacian
  solve + rank-1 marginals). The parallel investigation
  (`docs/superpowers/notes/2026-07-11-spectral-metric-investigation.md`) **concluded**: it is 3–45×
  faster than `network_efficiency`, its rank-1 (Sherman-Morrison) candidate marginals match a full
  re-solve to ~1e-14 (10–547× faster greedy scoring), and it *correctly* credits connecting
  stranded deep parcels — an action the current directness metric actually **mis-scores** (verdict:
  *augment, not replace* — resistance is the egress axis, directness the internal-circulation axis).
  It is **out of scope here** because it changes metric semantics and is a pending owner decision;
  this refactor makes the *existing* directness/E metric fast without changing any value (needed
  regardless — the compare grades every method on directness/E). If adopted, a resistance-objective
  arterial with rank-1 marginals would be a separate, even larger speedup than this refactor.
- **GPU.** Parked. At these graph sizes (hundreds–low-thousands of nodes) the win is in batching,
  and CPU freezing + csgraph should reach the target. The task-6 measurement decides if GPU is
  ever warranted.

## Risks

- **Committed roads in the incremental base (C1 — highest risk).** The scored set is
  `committed + [real]`, so a parcel's true nearest edge can be a *committed* road, not only a
  street or the trial road (measured: 22/32 parcels after 2 commits). The `StepContext`
  (streets ∪ committed, rebuilt per commit) folds committed roads into both the graph and the
  entry base; only `real` is per-candidate incremental. Guarded by the incremental-scorer parity
  test on a block with ≥1 committed road.
- **Committed×trial mid-span crossings (R1).** Appending only `real`'s edges to the step CSR leaves
  a committed edge unsplit where `real` crosses it mid-span (aspirational diagonals) → a missing
  crossing node and wrong distance (`dir=0.333` vs true `0.444`). Build the road subgraph from the
  `unary_union([base_merged, real])` explode, which re-nodes the crossing. Guarded by an
  aspirational-crossing trial in the parity battery (buildable grids meet only at shared vertices).
- **CSR duplicate-edge summation (I3).** scipy *sums* duplicate `(i,j)` COO entries (repeated edge
  → doubled weight → wrong distance), whereas networkx dedups and `_split_graph` removes a split
  edge's parent before adding its chain. The CSR build must dedup undirected pairs
  (last-write-wins) and delete the parent pair on split injection. Verified by harness parity.
- **`_snap` path tie-break (I4/R2).** csgraph and networkx pick different equal-cost paths on
  symmetric grids → different geometry; and even with `nx.shortest_path`, a numpy weight
  reimplementation differs from shapely by ~50 ulp and flips a path. Resolved by keeping
  `nx.shortest_path` AND computing weights via shapely's own `shapely.distance` ufunc
  (bit-identical to `Point.distance`). Gate: WKT-identical proposals.
- **Entry tie-breaks (task 4).** `_line_entries` breaks exact-distance ties by edge index; the
  incremental `min(step-base, real)` could pick a different edge on an exact tie. Exact ties are
  measure-zero on continuous coordinates and the entry point is `_rnd`-rounded; parity is verified
  on the fixtures. Fallback: re-derive over all edges for that parcel (still fast).
- **CSR node indexing.** Appending trial-road nodes must not collide with frozen indices; use a
  copy-on-write index map per candidate; deterministic ordering for byte-stable output.
- **`csgraph.dijkstra` vs networkx float parity.** Confirmed by the reviewer: identical to `0.0`
  on distinct-edge graphs; differences are float-summation-order only. Verified by the harness.
- **Speedup is block-size-dependent (M5).** Per-candidate `csr_matrix` construction is *new* work;
  on a 10-parcel block (cheap constants) it partly offsets the freezing savings, so expect a
  smaller multiple there. The large wins are on the big, currently-intractable blocks — the goal.
  The task-6 measurement on the ~80-parcel cluster is the real proof, not the 10-parcel block.
- **`road_drainage` keeps networkx.** Acceptable (one call per proposal); must not be on the
  scoring hot path.
