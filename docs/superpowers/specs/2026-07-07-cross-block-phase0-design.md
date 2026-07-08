# reblock — Cross-block reblocking, Phase 0: super-block merge + a falsifiable cross-block probe

**Status:** draft for review (revised after red-team) · **Date:** 2026-07-07 · **Branch:** `cross-block-phase0`

## Why this exists

The pipeline is **block-local** — every method proposes roads inside one `Block`, bounded by that
block's frontage. The goal is **non-myopic** street design: a single new street that runs through
several blocks, stays smooth, and *crosses* other streets rather than dead-ending at them. Phase 0
is the foundation that must land before any placement method is worth building.

**This spec was rewritten after a four-angle red-team killed its first draft.** The original plan —
"merge blocks into a super-block, then re-score today's block-local methods to *quantify the
myopia*" — was shown to be a **self-confirming non-experiment** (verified on the real flagship
cluster):

- **Access is inert on the merged block.** With interior streets kept as seeds (Decision A, retained
  — it is physically honest), a parcel's merged access depth equals its own-block depth: a
  cross-block adjacency edge only ever links two parcels that both touch the shared boundary, so both
  are already depth-1. Confirmed: **0 of 1158 parcels change depth** on the flagship pair;
  `k_before(super) = max(k_A, k_B)`. So `delta_k` carries zero cross-block information.
- **The structural "signature" is ≈0 on real data.** `n_cross_block_streets = 0` is a *tautology*
  (block-local roads can't cross a boundary by definition), and the expected "roads T-into interior
  boundaries" signal is empirically absent (the block *perimeter* dominates a 66,000 m² face).
- **Block-local peel already reaches k=1** on the flagship (the Δk=6 from the hosted artifact). So the
  cost of myopia was never in *access* — both block-local and cross-block reach k=1. It lives in
  **road efficiency and network structure**: block-local reaches k=1 with *more* road, *more*
  dead-ends, *zero* crossings, and no shared cross-block trunks.

So Phase 0's real job is the **cheapest falsifiable probe**: *does a cross-block network actually beat
boundary-reconciled block-local reblocking on road-efficiency + structure?* Answer that — on real
clusters, with a pre-registered kill criterion — **before** building Phase 1. If even a hand-authored
cross-block network can't beat the reconciled block-local baseline, the whole direction is
reconsidered in an afternoon instead of after shipping five modules.

## Scope

**In:** `merge_cluster(region) -> Block` (the confirmed-sound primitive); a **minimal, correctly-noded**
`StructureEval` (only the metrics the probe needs); and the **positive-control probe** — a
boundary-reconciled block-local baseline vs a hand-authored cross-block reference network, compared on
road-efficiency + structure across a few representative clusters, with a pre-registered falsification
criterion.

**Out (deferred, with reasons):**
- **`curvature_variation` + raw descriptors (`total_turning_deg`, `total_abs_curvature`,
  `vertices_per_km`)** — its whole point is scoring *arcs*, but arc-emitting methods are deferred two
  slices out, so it has no real consumer here; and as drafted it reads **0 for both baseline methods**
  (they emit 2-point `LineString`s with no interior vertex) and is sampling-density × micro-noise
  dominated (a 90° arc with 1 mm jitter swings from 0.008 to 36.3 across sampling densities). Build it
  *with* the first arc-emitting method, where it can be calibrated.
- **`dwellings_displaced`** — knowingly wrong (points, not footprints; a road clipping a parcel corner
  displaces 0 while destroying the home), its correctness fix (real footprints) is out of scope, and
  no Phase-0 method optimizes it. Defer to the real-footprints slice.
- **The through-going/collinear crossing refinement** — `n_crossings` is ≈0 by construction for the
  baseline, so the collinearity threshold gets its first real workout (and tuning) in Phase 1 anyway;
  Phase 0 counts **bare degree-≥4 nodes** (enough to show the reference creates crossings the baseline
  can't).
- Automatic cluster selection (= slum-detection); the `RegionMethod` path + Phase-1 placement methods;
  the full stratified sweep of all ~267 adjacent pairs; footprints; arcs.

## Global constraints

- **Reuse the existing derivations unchanged** on the merged block (verified: `parcel_adjacency`,
  `parcel_access_layers`, `street_connectivity`, `geometric_access_distances`, `KComplexityEval` all
  key on `parcel_id`/geometry, not `block_id`).
- **Deterministic baseline.** Use **`PeelReblocker` (deterministic)** as the block-local baseline;
  `TopologyMethod` seeds an RNG that moves exactly the structural metrics, so if it's included at all
  it must be aggregated over seeds — not in this probe.
- **Correct noding (settled fixes):** `shapely.set_precision(geom, grid≈STREET_TOL)` on the line set
  **before** `union_all` (raw `union_all` does not node sub-tolerance gaps; a post-hoc SNAP-round only
  relabels). Dead-end / T classification is `STREET_TOL`-aware. `n_cross_block_streets` is defined as
  "the road has vertices strictly on **both sides** of the boundary line" — not `.crosses()` (which
  false-negatives on run-along and false-positives on kiss-and-bounce).
- **Additive:** new `derive/cluster.py`, `derive/structure.py`, `eval/structure.py`,
  `conf/eval/structure.yaml`, `scripts/crossblock_probe.py`. No edits to `contracts.py`, `run.py`,
  existing methods/evals.
- `pixi run check` green.

---

## 1. The super-block merge — `src/reblock/derive/cluster.py`

`merge_cluster(region: Region) -> Block` folds the region's blocks (already filtered to the cluster
via the existing `block_ids` knob; cluster *selection* is upstream and out of scope) into one `Block`:

- **`parcels`** = `pd.concat` of every block's `parcels`, re-indexed with a fresh globally-unique
  sequential `parcel_id` (peel *asserts* uniqueness, so this is required, not decorative).
- **`streets`** = `union_all` of every block's `streets` — **Decision A, retained:** interior
  former-boundaries stay, because they are real walkable streets. Its role is now understood: it makes
  *access* inert across the merge (correctly), so the probe's signal lives in **road-efficiency and
  structure**, not an access delta.
- **`boundary`** = `union_all` of the block polygons; must be a single `Polygon`.
- **`attrs`** carries `{"block_ids": [...], "interior_boundaries": <MultiLineString>}` — the former
  inter-block boundary lines, which `StructureEval` reads for `n_cross_block_streets`. (Drop the
  speculative `source_block_id` column and per-block `kblock_k` map — no Phase-0 consumer.)

**Contiguity check (fail loud).** The selected blocks must share positive-length boundary and their
polygon union must be a single `Polygon`; otherwise raise `ValueError`. **Caveat to verify on target
data:** this bets on **zero-width inter-block streets** (kblock faces abut along a shared line). It
holds on the Cape Town fixture (adjacent blocks share 60–120 m); a source with positive-width
carriageways would separate genuinely-adjacent blocks by a gap and fail this check — check before
relying on it. (The full connected-component naming is deferred to the auto-selector slice that needs
it; the single-`Polygon` assertion suffices for hand-picked 2–4 block clusters.)

## 2. Minimal structural core — `derive/structure.py` + `eval/structure.py`

One correctly-noded planar graph, then only the metrics the probe needs.

`node_network(roads, streets) -> nx.Graph`: `set_precision(line, grid≈STREET_TOL)` on every road and
street line, `union_all` to node them at true and near-miss intersections, build a `networkx` graph
(nodes = endpoints/intersections, edges = segments with geometry + length + road/street tag).

Metrics (`StructureEval.score(block, proposal) -> Metrics`, `eval="structure"`):

- **`n_cross_block_streets`** — road features with vertices strictly on both sides of an
  `interior_boundaries` line (0 for any block-local method — the Phase-1 discriminator).
- **`n_crossings`** — bare degree-≥4 nodes (collinear "through-going" refinement deferred to Phase 1).
- **`n_t_junctions`** — degree-3 nodes; **`n_dead_ends`** — degree-1 road endpoints not within
  `STREET_TOL` of a street.
- **`added_road_length_m`** and **`k_after`** come from `KComplexityEval` (already emitted) — together
  they are the **road-efficiency** headline: total road to reach a given access level. Compared at
  equal `k_after` (ideally both = 1), *less road wins*; if `k_after` differs, use `delta_k /
  added_road_length_m`.

`StructureEval` degrades gracefully on an ordinary single block (no `interior_boundaries` →
`n_cross_block_streets = 0`) and, being an `Eval`, is reused unchanged when Phase-1 methods are scored.

## 3. The falsifiable probe — `scripts/crossblock_probe.py`

The deliverable and the point of Phase 0. For each of **3–5 representative clusters** hand-picked to
span morphology (at least two *deep* clusters, `min(kblock_k) ≥ 5`, and one moderate), by `block_ids`:

1. `merge_cluster(region)` → super-block.
2. **Baseline (boundary-reconciled block-local peel).** Run `PeelReblocker` per constituent block;
   union the roads; then **reconcile the boundary**: snap road endpoints that land within ~2·STREET_TOL
   of each other across an interior boundary into a shared node. This removes the naive-union
   strawman — a competent practitioner *would* stitch co-located stubs — so the comparison measures
   method myopia, not an un-reconciled artifact.
3. **Reference (hand-authored cross-block network).** By hand, author a cross-block network that
   reaches the same access (k=1) — typically by replacing the reconciled baseline's redundant
   *boundary-parallel* spines with shared **through-trunks** that cross the interior boundary, keeping
   short feeders. Stored as a small WKT/GeoJSON fixture per cluster for reproducibility. This is an
   **existence proof**: if a human can't beat the reconciled baseline, a Phase-1 algorithm won't.
4. **Score both** on the super-block with `KComplexityEval` + `StructureEval`, and tabulate:
   `k_after`, `added_road_length_m`, `n_cross_block_streets`, `n_crossings`, `n_t_junctions`,
   `n_dead_ends`. Optionally render both via the existing `render_before`/`render_after`.

**Pre-registered falsification criterion (write the number before running).** Default proposal:
*on the deep clusters, the hand-authored cross-block reference must reach k=1 with ≥ 15% less total
road than the boundary-reconciled baseline* **and** *introduce ≥ 1 real crossing where the baseline
has none.* If it does not clear that bar, cross-block reblocking's benefit is below the threshold and
Phase 1 is **not** pursued as specced. The whole probe exists so that a result *can* say "don't build
Phase 1" — the first draft could not.

## 4. Testing

- **Merge:** two adjacent synthetic blocks → super-block with concatenated parcels (unique global
  `parcel_id`), shared boundary retained in `streets`, single-`Polygon` boundary; two **non-adjacent**
  blocks → `ValueError`. A real 2-block Cape Town cluster → well-formed super-block, parcel count =
  sum of constituents.
- **Noding correctness on near-miss geometry:** two lines that pass within `STREET_TOL` but don't
  exactly meet → after `set_precision`+`union_all`, a single degree-4 node (crossing recovered), and a
  road ending 0.3 m short of a street → a **T-junction, not a dead-end**. A `+` → `n_crossings = 1`; a
  `T` → `n_t_junctions = 1`. A road with vertices on both sides of an interior boundary →
  `n_cross_block_streets = 1`; a road merely running *along* it → 0.
- **Probe invariant (real data):** the boundary-reconciled block-local peel union on a real Cape Town
  2-block cluster → `n_cross_block_streets == 0` (by construction) and `k_after == 1` (block-local peel
  already flattens each block), establishing the baseline the reference must beat on *road length*.

## 5. Deferred (with reasons) — carry to Phase 1 / other slices

`curvature_variation` + raw descriptors and `dwellings_displaced` (built *with* their first real
consumer — an arc-emitting method / real footprints — where they can be calibrated, per §Scope); the
collinear through-going crossing refinement; automatic cluster selection + the density/k screen (=
slum-detection); the `RegionMethod` contract path and Phase-1 placement methods; the full stratified
sweep of all adjacent pairs; the displacement↔tessellation-circularity fix. The five-approach
placement exploration (planner-hierarchy, tensor-field, medial-axis, network-design ILP, variational)
and its cross-cutting risks (the dominant-orientation/grid assumption; points-not-footprints) are the
Phase-1 brainstorm's input — to be red-teamed once concrete, using this probe's result as ground truth.

## 6. Sequencing note

The sibling **atomic-flow spec** (same date) builds `reblock.compare` with reusable **scorecard** and
**render** emitters. The probe's tabulation/render half is a hand-rolled instance of that machinery;
the genuinely novel, non-reusable part is the merge → per-block → reconciled-union baseline harness
(itself throwaway once `RegionMethod` lands — and that's fine; its job is one measurement). If
atomic-flow lands first, make `crossblock_probe.py` a thin driver that emits `Result`s into the shared
emitters rather than printing its own table.
