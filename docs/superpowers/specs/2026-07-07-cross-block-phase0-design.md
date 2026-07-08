# reblock — Cross-block reblocking, Phase 0: super-block merge + structural metrics

**Status:** draft for review · **Date:** 2026-07-07 · **Branch:** `cross-block-phase0` (to be cut)

## Why this exists

The pipeline is **block-local**: every method proposes roads inside one `Block`, bounded by that
block's frontage. The goal (from the five-approach exploration) is **non-myopic** street design — a
single new street that runs through *several* blocks, stays smooth, and *crosses* other streets
rather than dead-ending at them. But all five candidate placement methods converged on two findings
that must land **before** any of them is worth building:

1. **The cross-block plumbing is nearly free.** Merge a cluster of adjacent blocks into one
   *super-block* and every existing derivation (`parcel_adjacency`, `parcel_access_layers`,
   `street_connectivity`, `geometric_access_distances`, `KComplexityEval`) runs on it **unchanged** —
   adjacency spans the former block boundaries automatically. The contract already anticipates a
   region-level method (`RegionMethod`, `contracts.py:94`, currently unused).
2. **peel-k is blind to the very thing we now care about.** It counts topological hops — it cannot
   tell a clean straight 4-way grid from a jagged spine of equal depth. Without **structural metrics**
   (crossings, dead-ends, smoothness, displacement, cross-block continuity), no placement method's
   advantage is measurable.

**Phase 0 builds exactly that foundation and nothing more:** the super-block merge, the structural
metrics, and a re-scoring of *today's* block-local methods on merged clusters to **quantify the
myopia**. No new placement method. The point is ground truth — stand up the scorecard and measure the
baseline *before* betting on any Phase-1 method.

## Scope

**In:** `merge_cluster(region) -> Block`; a `StructureEval` computing structural metrics from one
noded planar graph; a myopia scorecard that re-scores block-local `peel`/`topology` (as a
union-of-per-block baseline) on merged Cape Town clusters.

**Out (deferred, explicit):** every Phase-1 placement method and the `RegionMethod` contract change;
automatic cluster selection (seed + screened adjacency-growth) — the screen *is* slum-detection, its
own slice; arc-*emitting* methods and arc noding; real building footprints and the
displacement↔tessellation fix; the `streets`-model B/C variants; region choropleth / viz.

## Global constraints

- Reuse the existing derivations **unchanged** on the merged block.
- **Additive:** new `derive/cluster.py`, `derive/structure.py`, `eval/structure.py`,
  `conf/eval/structure.yaml`, `scripts/myopia_scorecard.py`. No edits to `contracts.py`, `run.py`,
  `ShapefileSource`, or the existing methods/evals.
- **Deterministic:** sorted `block_id` order, sequential `parcel_id`, no RNG.
- **Decision A (settled):** interior former-boundaries are **kept as real street seeds** on the merged
  block. Myopia surfaces in the *structural* metrics, not a manufactured access delta.
- **Smoothness = curvature *variation*, not straightness (settled):** the metric admits straight lines
  *and* constant-curvature arcs equally; only changing curvature is penalized.
- **Crossings = through-going (settled):** a crossing is a high-degree node where two edges pass
  roughly straight through (collinear pair), not a bare degree-4.
- `pixi run check` green (ruff + `mypy --strict src tests` + pytest).

---

## 1. The super-block merge — `src/reblock/derive/cluster.py`

`merge_cluster(region: Region) -> Block` folds the region's blocks (already filtered to the cluster
via the existing `block_ids` knob — **cluster *selection* is upstream and out of scope**; it just
emits `block_ids`) into one `Block`:

- **`parcels`** = `pd.concat` of every constituent block's `parcels`, re-indexed with a fresh
  **globally-unique sequential `parcel_id`**, plus a `source_block_id` column mapping each parcel to
  its origin block (needed by no eval, but cheap provenance and useful for render/debug).
- **`streets`** = `union_all` of every constituent block's `streets` (Decision A: interior
  former-boundaries stay). A parcel fronting an interior street is still depth-1 — correct.
- **`boundary`** = `union_all` of the block polygons. Must be a single `Polygon`; a `MultiPolygon`
  means the blocks are **not contiguous** → fail loud (see below).
- **`attrs`** carries `{"block_ids": [...], "kblock_k": {bid: k}, "interior_boundaries": <MultiLineString>}`
  where `interior_boundaries` = the former inter-block boundary lines (the shared frontages), which
  `StructureEval` needs for `n_cross_block_streets`.

**Contiguity check (fail loud).** The selected blocks must form **one** connected component under
shared-boundary adjacency (two blocks adjacent iff `b.boundary.intersection(c.boundary).length > 0`
within `STREET_TOL`, reusing the robust `_shared_len` pattern from `derive/adjacency.py`). If they
form >1 component (or `boundary` is a `MultiPolygon`), raise `ValueError` naming the disconnected
groups — merging disconnected fabric yields a meaningless multi-component graph.

**Reuse, unchanged.** The merged `Block` is a normal `Block`; `parcel_adjacency`,
`parcel_access_layers`, `street_connectivity`, `geometric_access_distances`, and `KComplexityEval`
all operate on it with no modification. That is the whole "cross-block plumbing is nearly free" claim,
made concrete.

## 2. Structural metrics — `derive/structure.py` + `eval/structure.py`

### 2a. One noded planar graph (`derive/structure.py`)

`node_network(roads: GeoDataFrame, streets: GeoDataFrame) -> nx.Graph`: `union_all` the proposed
roads with the block's streets so every line splits at every intersection (`unary_union` on the line
set nodes them), then build a `networkx` graph whose **nodes** are endpoints/intersection points
(rounded to a `SNAP` grid to merge coincident points) and **edges** are the segments between them,
each carrying its geometry, length, and a `is_road`/`is_street` tag. Guard messy real geometry with
`make_valid` (same GEOS-robustness lesson as `parcel_adjacency`).

Pure functions over that graph + the road geometry:

- **`n_cross_block_streets(roads, interior_boundaries) -> int`** — count of proposed road features
  whose geometry crosses ≥1 interior former-boundary line. **0 for any block-local method by
  construction** — the myopia signature.
- **`crossing_counts(graph, delta_deg) -> {n_crossings, n_t_junctions, n_dead_ends}`** — per node,
  by degree: degree-1 road endpoint not on a street → **dead-end**; degree-3 → **T-junction**;
  degree ≥ 4 with a **collinear incident pair** (two edges whose bearings differ < `delta_deg`,
  default 30°) → a **through-going crossing**. A degree-4 node with *no* collinear pair (four kinked
  stubs) is *not* counted as a crossing.
- **`curvature_variation(roads) -> float`** — per road polyline, discrete curvature at interior
  vertex *i* is `kappa_i = turn_angle_i / mean_adjacent_segment_len`; the road's roughness is the
  total variation `sum |kappa_{i+1} - kappa_i|`; report the **road-length-weighted mean** across
  roads. A straight line (all `kappa≈0`) and a constant-curvature arc (all `kappa≈const`) both score
  ≈ 0; a jagged centroid-chain scores high. Guard near-zero-length segments.
- **Raw descriptors** (read straight-vs-curved separately from smooth-vs-jagged):
  `total_turning_deg` (Σ|Δbearing|), `total_abs_curvature` (Σ|kappa|·len), `vertices_per_km`.
- **`dwellings_displaced(roads, buildings, half_width) -> int`** — building points within
  `union_all(roads).buffer(half_width)`. **Documented approximation:** buildings are *points* not
  footprints, so this under-counts real sweep; and displacing a building would change the Voronoi
  tessellation the eval scores — that circular dependency is acknowledged and **deferred** (needs
  real footprints).

### 2b. `StructureEval` (`eval/structure.py`)

A normal `Eval`: `score(block, proposal) -> Metrics`. It nodes `(proposal.roads ∪ block.streets)`,
reads `block.attrs.get("interior_boundaries")` (absent → no interior boundaries →
`n_cross_block_streets = 0`, so it degrades gracefully on an ordinary single block), and emits all of
2a into `Metrics(eval="structure", values={...})` (no `fields`). Because it satisfies the `Eval`
protocol, it slots into any eval list — so **Phase-1 methods get scored on it automatically** via
`run()`/`compare` with no extra wiring. `conf/eval/structure.yaml` exposes it.

## 3. The myopia scorecard — `scripts/myopia_scorecard.py`

The re-scoring driver (a script, so **no `run.py` / contract change** — `run()` iterates per block;
we need merge-then-union-then-score, a different control flow that Phase 1's `RegionMethod` path will
formalize later):

1. Take a manual cluster: `block_ids=[...]` over `data=capetown` → `KblockSource(block_ids=...)` →
   `region` of the cluster's blocks.
2. `merge_cluster(region)` → the super-block.
3. For each method in `{PeelReblocker, TopologyMethod}`: run it **per original constituent block**,
   `pd.concat` the per-block `Proposal.roads` into one **union proposal** on the super-block. This is
   the faithful *block-local myopia baseline* — the stubs die at the boundaries.
4. Score each union proposal on the super-block with **`KComplexityEval` + `StructureEval`**.
5. Emit a table (one row per method): `k_before/k_after/delta_k`, `geometric_access_max_m`,
   `added_road_length_m`, and the structural metrics. Optionally render the merged cluster
   before/after via the existing `render_before`/`render_after`.

**Expected result (the myopia, quantified):** `n_cross_block_streets = 0` (by construction — a
block-local road cannot cross a boundary), block-local roads **T-into** the interior boundaries
instead of crossing them (elevated `n_t_junctions` there, since a road reaching its block's frontage
now meets an *interior* street), plus interior spur-tip `n_dead_ends` and high `curvature_variation`
for peel — a concrete baseline the Phase-1 methods must beat.

## 4. Testing

- **Merge:** two adjacent synthetic blocks → super-block with concatenated parcels (unique global
  `parcel_id`), the shared boundary retained in `streets`, single-`Polygon` boundary, contiguity
  passes; two **non-adjacent** blocks → `ValueError`. A real 2-block Cape Town cluster yields a
  well-formed super-block whose parcel count = sum of constituents.
- **Structural metrics on known geometry:** a `+` of two crossing lines → `n_crossings = 1`; a `T` →
  `n_t_junctions = 1`, `n_dead_ends` accounts for the stub; a straight polyline **and** a
  densely-sampled circular arc → both `curvature_variation ≈ 0`; a zigzag → high; a road segment
  crossing an interior boundary → `n_cross_block_streets = 1`; points inside/outside a road buffer →
  exact `dwellings_displaced`.
- **Myopia signature (real data):** union-of-per-block `peel` on a real Cape Town 2-block cluster →
  assert `n_cross_block_streets == 0` (the by-construction myopia signature — no block-local road
  crosses an interior boundary) and that ≥1 road segment T-terminates *on* an interior boundary.

## 5. Deferred (carry to Phase 1 / other slices)

Placement methods + the `RegionMethod` path; automatic cluster selection (seed + screened
adjacency-growth) and the density/k screen (= slum-detection); arc-emitting methods and arc-aware
noding; real footprints and the displacement↔tessellation-circularity fix; `streets`-model B/C;
region-scale viz. The five-approach exploration (planner-hierarchy, tensor-field, medial-axis,
network-design ILP, variational) and its cross-cutting risks (the dominant-orientation/grid
assumption; points-not-footprints) are the Phase-1 brainstorm's input — and the plan is to red-team
Phase 1 once its design is concrete, using this scorecard as ground truth.
