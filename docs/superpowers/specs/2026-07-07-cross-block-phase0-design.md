# reblock — Cross-block reblocking, Phase 0: super-block merge + a falsifiable proxy-metric probe

**Status:** draft for review (revised twice: red-team, then proxy-metric basis) · **Date:** 2026-07-07 · **Branch:** `cross-block-phase0`

## Why this exists

The pipeline is **block-local** — every method proposes roads inside one `Block`, bounded by that
block's frontage. The goal is **non-myopic** street design: a single new street that runs through
several blocks, stays smooth, and *crosses* other streets rather than dead-ending at them. Phase 0 is
the foundation that must land before any placement method is worth building.

**This spec was rewritten after a four-angle red-team.** The original plan — "merge blocks, re-score
today's methods to *quantify the myopia*" — was a **self-confirming non-experiment** (verified on the
real flagship cluster): access is inert on the merged block (block-local peel already reaches k=1;
0/1158 parcels change depth on merge), `n_cross_block_streets = 0` is a tautology, and the drafted
`curvature_variation` reads 0 for the 2-point-`LineString` baseline methods. The cost of myopia was
never in *access* — both block-local and cross-block reach k=1. It lives in **network quality**:
block-local reblocking produces a **tree** (all T's and dead-ends, no crossings, no loops, redundant
boundary-parallel spines), and the question is *how costly that is*.

So Phase 0 is the **cheapest falsifiable probe**, built on **proxy network-quality metrics** measured
on real clusters. Crucially, welfare proxies like **circuity** and **throughput** have *absolute
floors*, so they reveal the block-local baseline's headroom **without any hand-drawn or reference
network** — solving both the "can't procure a reference" problem and the red-team's "no control"
problem at once. If block-local networks already run near the welfare optimum across a sample, cross-
block reblocking has little to add and Phase 1 is not built — a result the first draft could not
produce.

## Scope

**In:** `merge_cluster(region) -> Block`; a **correctly-noded planar graph** + an **orthogonal metric
basis** (below) computed on the boundary-reconciled block-local baseline; and the **probe** — run the
basis over a stratified sample of real clusters, **validate the basis's orthogonality with a
correlation matrix**, and assess cross-block headroom against a **pre-registered falsification bar**.

**Out (deferred, with reasons):**
- **`curvature_variation` + raw descriptors** — its point is scoring *arcs*, but arc-emitting methods
  are deferred; as drafted it reads 0 for 2-point-`LineString` baselines and is sampling-noise
  dominated. Build it *with* the first arc-emitting method (Axis I).
- **`dwellings_displaced`** — knowingly wrong (points, not footprints), fix out of scope, no Phase-0
  consumer. Defer to the real-footprints slice.
- **The collinear "through-going" crossing refinement** — bare degree-≥4 suffices for the probe.
- **A spine-merge *optimizer*** — the probe uses at most a *heuristic* spine-merge reference; the real
  optimizer is Phase 1.
- Automatic cluster selection (= slum-detection); the `RegionMethod` path + Phase-1 placement methods;
  footprints; arcs.

## Global constraints

- **Reuse existing derivations unchanged** on the merged block (`parcel_adjacency`,
  `parcel_access_layers`, `street_connectivity`, `geometric_access_distances`, `KComplexityEval` key
  on `parcel_id`/geometry, not `block_id` — verified).
- **Deterministic baseline:** `PeelReblocker` (deterministic). `TopologyMethod` seeds an RNG that
  moves the structural metrics — excluded from this probe.
- **Correct noding:** `shapely.set_precision(geom, grid≈STREET_TOL)` on the line set **before**
  `union_all` (raw union does not node sub-tolerance gaps; a post-hoc round only relabels). Dead-end/T
  classification is `STREET_TOL`-aware. `n_cross_block_streets` = "road has vertices strictly on
  **both sides** of the boundary" (not `.crosses()`).
- **Throughput demand model:** unit egress demand per parcel → nearest perimeter access; unit edge
  capacity (a width-agnostic permeability proxy; width-weighting is a later refinement).
- **Additive:** new `derive/cluster.py`, `derive/structure.py`, `derive/network_metrics.py`,
  `eval/structure.py`, `scripts/crossblock_probe.py`. No edits to `contracts.py`, `run.py`, existing
  methods/evals.
- `pixi run check` green.

## 1. The super-block merge — `derive/cluster.py`

`merge_cluster(region) -> Block` folds the `block_ids`-selected blocks (selection is upstream, out of
scope) into one `Block`: `parcels` = `pd.concat` with a fresh globally-unique sequential `parcel_id`;
`streets` = `union_all` of all block `streets` (**Decision A retained** — interior streets are real;
its consequence, inert merge-access, is now expected, and the signal lives in the proxy metrics);
`boundary` = `union_all` of block polygons (a single `Polygon`); `attrs` carries `block_ids` and the
`interior_boundaries` `MultiLineString`. **Contiguity fail-loud** (shared positive-length boundary,
single-`Polygon` union). **Caveat to verify on target data:** this bets on **zero-width inter-block
streets** — holds on the Cape Town fixture (blocks abut 60–120 m); a positive-width-carriageway source
would fail the check.

## 2. Noded graph + the orthogonal metric basis

`node_network(roads, streets)` — `set_precision` on every line, `union_all` to node at true and
near-miss intersections, build a length-weighted `networkx` graph tagged road/street. The basis is
computed by `derive/network_metrics.py` + `eval/structure.py` (`StructureEval.score -> Metrics`), one
representative per axis; some reuse existing derivations directly:

| Axis (question) | Metric(s) | Source |
|---|---|---|
| **A. Reachability** — can everyone reach a street? | `peel_k`, `geometric_access_max_m` | existing evals |
| **B. Equity** — who's *worst*-served? | `geometric_access_p95_m` | `geometric_access_distances` tail |
| **C. Directness** — how circuitous? | `circuity` = mean(network dist / straight-line dist to street); floor 1.0 | reuses `geometric_access_distances` + euclidean |
| **D. Throughput** — does it bottleneck? | `throughput` = max-flow interior→perimeter (unit demand/capacity), normalized by perimeter capacity | `networkx.maximum_flow` |
| **E. Redundancy** — alternate routes? | `meshedness` = (E−N+C)/(2N−5); tree = 0 | `networkx` |
| **F. Permeability** — grid vs cul-de-sac? | `four_way_fraction`, `dead_end_fraction` | degree histogram |
| **G. Cost** — how much to build? | `added_road_length_per_parcel` | road length ÷ parcels |
| **H. Cross-block** — non-myopia | `n_cross_block_streets`, `cross_block_trunk_length_m`, `boundary_redundant_road_fraction` | geometry |
| **I. Smoothness** — buildability | `curvature_variation` | **deferred** (arc slice) |

**Two tiers.** *Falsification metrics* (reference-free, welfare-interpretable, absolute floors):
**circuity (C), throughput (D), `geometric_access_p95_m` (B).** *Structural descriptors* (want a
reference / are near-tautological alone): redundancy (E), permeability (F), cross-block (H). Cost (G)
and reachability (A) are context. `boundary_redundant_road_fraction` (H) — the fraction of baseline
road running near-parallel and close on **both** sides of an interior boundary (the road a shared
through-trunk would merge) — is the one cross-block-*specific* quantity measurable on the baseline
alone, and is the bridge between "there is headroom" and "the headroom is cross-block."

## 3. The probe — `scripts/crossblock_probe.py`

Fully automatic; no hand-drawn network.

1. **Sample.** Enumerate adjacent-block clusters in the Cape Town fixture (blocks sharing positive-
   length boundary), **stratify by `min(kblock_k)`** across the cluster, and draw ≥ ~30 (spanning
   shallow→deep). This replaces the first draft's indefensible n=1 flagship — automatic metrics make a
   real sample free.
2. **Per cluster:** `merge_cluster` → run `PeelReblocker` per constituent block → union the roads →
   **boundary-reconcile** (snap co-located stubs across an interior boundary within ~2·STREET_TOL, so
   the baseline is a competent practitioner's stitched output, not a strawman) → compute the full
   metric basis.
3. **Validate the basis (orthogonality).** Assemble the sample × metric table and compute the
   **correlation matrix**; any pair with |r| ≳ 0.9 is a redundant direction — keep the more
   interpretable member, drop the other, and report the pruned empirically-orthogonal basis. (Don't
   assert independence — demonstrate it. Likely collapses to watch: peel-k vs geometric-max;
   throughput vs meshedness; dead-end-fraction vs meshedness.)
4. **Assess headroom.** Report the **distributions** of the falsification metrics (circuity,
   throughput, `geometric_access_p95_m`) and `boundary_redundant_road_fraction` across the sample.
5. **Optional relative check.** A *heuristic* automatic spine-merge reference (replace near-parallel
   boundary-flanking spine pairs with a single through-trunk crossing the boundary) scored on the same
   basis — isolates the cross-block-specific gain without any optimizer.
6. **Pre-registered falsification bar** (write the numbers before running; defaults to refine):
   *cross-block reblocking is **not** worth Phase 1 if, across the sample, median `circuity < 1.3`
   **and** median `boundary_redundant_road_fraction < 0.10` (the baseline is already direct and shares
   little redundant boundary road).* If instead circuity is high and boundary-redundant road is large,
   the cross-block headroom is real and Phase 1 proceeds.

## 4. Testing

- **Merge:** two adjacent synthetic blocks → super-block with concatenated unique-`parcel_id` parcels,
  shared boundary retained, single-`Polygon` boundary; non-adjacent → `ValueError`; a real 2-block
  Cape Town cluster → parcel count = sum of constituents.
- **Noding on near-miss geometry:** two lines within `STREET_TOL` but not exactly meeting → one
  degree-4 node after `set_precision`+`union_all`; a road ending 0.3 m short of a street → a
  **T-junction, not a dead-end**.
- **Metric basis on known graphs:** a tree → `meshedness = 0`, high `dead_end_fraction`; a 3×3 grid →
  `meshedness > 0`, high `four_way_fraction`; a straight corridor → `circuity ≈ 1.0`; an L-detour →
  `circuity > 1`; a known min-cut graph → expected `throughput`; a road with vertices on both sides of
  an interior boundary → `n_cross_block_streets = 1` (a road running *along* it → 0).
- **Probe invariant (real data):** boundary-reconciled block-local peel on a real Cape Town cluster →
  `n_cross_block_streets == 0` and `meshedness ≈ 0` (it's a forest), with finite, sane circuity/
  throughput — establishing the baseline the headroom is read against.

## 5. Deferred (with reasons)

`curvature_variation` + raw descriptors (Axis I — build with the first arc-emitting method);
`dwellings_displaced` (real-footprints slice); the collinear through-going crossing refinement; the
spine-merge *optimizer* and all Phase-1 placement methods + the `RegionMethod` path; automatic cluster
selection + density/k screen (= slum-detection); the displacement↔tessellation-circularity fix. The
five-approach placement exploration and its cross-cutting risks (dominant-orientation/grid assumption;
points-not-footprints) are the Phase-1 brainstorm's input — to be red-teamed once concrete, using this
probe's headroom result as ground truth.

## 6. Sequencing note

The sibling **atomic-flow spec** builds `reblock.compare` with reusable scorecard/render emitters; the
probe's tabulation is a hand-rolled instance. The genuinely novel, non-reusable part is the
merge→per-block→reconciled-union baseline harness (throwaway once `RegionMethod` lands — fine, its job
is one measurement). If atomic-flow lands first, make `crossblock_probe.py` a thin driver emitting
`Result`s into the shared emitters.
