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
  build + dijkstra in the hot path are *replaced*, not kept alongside a csgraph path. `networkx`
  may remain only in genuinely cold code (e.g. `road_drainage`, run once per proposal) if
  migrating it is out of scope; the hot scoring path must not import or call it.
- **Determinism.** Same block + roads → same score, byte-stable across runs (sorted sources,
  stable node indexing, deterministic entry tie-breaks).

---

## The problem (measured)

`cProfile` of one `GreedyArterialReblocker(mode="buildable", objective="directness").propose` on
block `DJI.3_1_1808` (**10 parcels**) — **43 seconds**, 8 roads placed:

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

### 3. Per-candidate incrementalism (the arterial win)

`GreedyArterialReblocker` builds **one `_BlockScoringContext` per block** (not per candidate). Per
candidate road set (`committed + [real]`):

- **Entries (incremental):** start from the frozen street-entry base; for each parcel, compare its
  nearest-street-edge distance with its distance to the *trial road's* few edges (via the trial
  road's own small STRtree, or the parcels-within-`tol`-of-`real` query), and take the closer as
  the entry — equivalent to `_line_entries` over all edges, but O(N × road-edges) instead of
  O(N × all-edges). Exact ties between a street edge and a road edge are measure-zero on
  continuous geometry (see Risks); the equivalence harness confirms parity.
- **Graph (incremental):** append the trial road's edges (noded against the frozen street+committed
  network — see planarize below) and the road-entry split nodes to the frozen base CSR arrays
  (new nodes get appended indices); build the per-candidate CSR by concatenation.
- **Score:** `_sampled_efficiency` on that CSR.

### 4. `_snap` and `_planarize`

- **`_snap` (buildable):** the boundary graph `g = _boundary_graph(block.parcels)` is already
  constant per block. Freeze it as a CSR + node→index + `edge_midpoint_xy (E,2)` (constant). Per
  candidate the weight is `length + lam·dist(edge_midpoint, chord)`; compute the `dist(midpoints,
  chord)` term for all edges with a **vectorized numpy point-to-segment distance**, form the
  per-chord weighted CSR, and run `csgraph.dijkstra` between the two endpoint indices, then
  reconstruct the path from predecessors. Replaces the networkx `shortest_path` with a Python
  weight callback (the 2.38s `w`).
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
- Delete `import networkx` from the scoring path. `road_drainage` (one call per proposal, cold)
  may keep networkx for now; note it explicitly as out of scope.

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
Assert on: `E`, `directness`, every point of both curves, and both AUCs.

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
   street CSR + reps + entry base + sources; both callers score through it. Arterial builds ONE
   context per block and scores each candidate with full entry re-derivation against the frozen
   street CSR (no incremental-entry risk yet). Gate: harness parity + full suite; re-time (expect
   the 51% graph-construction cost to collapse).
4. **Incremental entries.** Freeze street-entries; per candidate re-project only onto the trial
   road's edges and take the min. Gate: harness parity (this is the tie-break-sensitive step —
   verify byte parity on the fixtures); re-time.
5. **`_snap` (numpy weight + csgraph) and incremental `_planarize`.** Gate: arterial output
   geometry unchanged on the fixtures (the proposed roads must be identical), full suite; re-time.
6. **Remove networkx from the hot path + final measurement.** Delete the dead nx graph builders
   from the scoring path; confirm no nx import in `budget.py`'s scoring path or arterial scoring.
   Final re-measure on `DJI.3_1_1808` and a ~80-parcel region (the calibration cluster
   `DJI.3_1_2914,2923,2925,2930`), report the speedup, and a whole-branch review.

---

## Alternatives considered

- **Naive csgraph swap only (no freezing).** Rejected: the profile shows networkx is ~15–20%;
  this leaves the 51% graph-construction and 16% planarize untouched. ~2× at best.
- **Memoize candidate scores across greedy steps.** The base changes each step, so most scores are
  invalid across steps; low yield versus the freezing approach, and fragile.
- **Grounded effective-resistance metric (north-star Piece 2).** A *different metric* (Laplacian
  solve + rank-1 marginals) that would make greedy scoring algorithmically cheap (O(1)-ish
  marginals) and is the natural GPU target. It is being **investigated in parallel**
  (`docs/superpowers/notes/2026-07-11-spectral-metric-investigation.md`). It is out of scope here
  because it changes the metric semantics (just adopted door-to-door directness); this refactor
  makes the *existing* metric fast without changing any value. If the investigation recommends it,
  it becomes a separate follow-on.
- **GPU.** Parked. At these graph sizes (hundreds–low-thousands of nodes) the win is in batching,
  and CPU freezing + csgraph should reach the target. The task-6 measurement decides if GPU is
  ever warranted.

## Risks

- **Entry tie-breaks (task 4).** `_line_entries` breaks exact-distance ties by edge index; the
  incremental min(street, road) approach could pick a different edge on an exact tie. Exact ties
  are measure-zero on continuous coordinates and the entry point is `_rnd`-rounded; the
  equivalence harness verifies parity on the fixtures. If a real divergence appears, fall back to
  re-deriving entries over all edges for that parcel (still fast).
- **CSR node indexing.** Appending trial-road nodes must not collide with frozen indices; use a
  copy-on-write index map per candidate. Deterministic ordering required for byte-stable output.
- **`csgraph.dijkstra` vs networkx float parity.** Both compute exact shortest paths on the same
  weights; differences are float-summation-order only, well within `1e-9`. Verified by the harness.
- **`road_drainage` still uses networkx.** Acceptable (one call per proposal); explicitly out of
  scope, but must not be on the scoring hot path.
