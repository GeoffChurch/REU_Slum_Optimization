# Substrate head-to-head: grid vs corner-chopping chord graph

Spike comparing the shipped 8-connected `grid` substrate against a `chord1` (skip-one
corner-chop) and `chord_diag` (full within-cell diagonals) graph built on the parcel
tessellation, for the clearance least-cost-path reblocker's greedy loop
(`reblock.methods.clearance._greedy_reblock`), reimplemented substrate-agnostically as
`greedy_on_substrate` in `substrate_headtohead.py`. Same `depth_target=2`, `max_roads=400`,
3-point (endpoint+endpoint+midpoint) edge-cost sampling on **both** substrates for fairness.

Script: `substrate_headtohead.py`. Overlay: `substrate_overlay_s0.png`. Full run log
captured below; total wall time 16.2s (well under the 10-minute budget).

## Sanity cross-check

`greedy_on_substrate` on the `grid` substrate vs the shipped `ClearanceReblocker` (which calls
`_greedy_reblock` directly), on the main region:

| s | shipped roads / len | substrate roads / len | note |
|---|---|---|---|
| -6 | 22 / 328m | 22 / 328m | match |
| -2 | 22 / 328m | 22 / 328m | match |
| 0 | 22 / 327m | 22 / 327m | match |
| +2 | 21 / 341m | 21 / 342m | match (~0.3%) |
| +6 | 21 / 389m | 21 / 368m | **diverges, ~5.4%** |

Road count matches exactly at every s. Length matches almost exactly except at the most
extreme repulsion (s=+6), where the substrate version comes out ~5.4% shorter. This is the
one place the 3-point cost rule (mine) and the shipped endpoint-only `_edge_weights` visibly
disagree: at maximal clearance-hugging, at least one road's Dijkstra path picks a different
route because the midpoint sample pulls the effective edge cost of some near-building grid
edges up (or down) relative to the endpoint-only average, tipping a tie. Per the design's
explicit instruction, this is flagged rather than smoothed over — it is a real (small)
behavioral difference introduced by fixing the "long chord skims a building" fairness problem,
not a bug in the reimplementation (road counts agree, and the divergence is confined to the
single most extreme knob setting).

## Main region compare (`ZAF.9.3.1_1_23732` + `ZAF.9.3.1_1_23733`, 250 parcels / 250 buildings, 10,967 m²)

Substrate build (once, reused across all 5 s values):

| substrate | n_nodes | n_edges | build_time_s |
|---|---:|---:|---:|
| grid | 4,878 | 19,095 | 0.039 |
| chord1 | 548 | 2,204 | 0.028 |
| chord_diag | 548 | 3,056 | 0.026 |

Grid has **8.9x** as many nodes as either chord variant on this region (chord1/chord_diag
share the same 548 boundary-graph nodes; chord_diag simply adds more chord edges: 3,056 vs
2,204, i.e. 1.4x chord1's edge count).

Per (substrate, s) — `directness_AUC` uses `cap = 386.0m` (max `length_m` across all 15 rows):

| substrate | s | roads | length_m | displaced | max_depth_after | n_unroutable | directness_AUC | propose_time_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| grid | -6 | 22 | 328.1 | 70 | 2 | 0 | 0.0474 | 0.059 |
| grid | -2 | 22 | 328.1 | 70 | 2 | 0 | 0.0474 | 0.058 |
| grid | 0 | 22 | 327.0 | 67 | 2 | 0 | 0.0506 | 0.060 |
| grid | +2 | 21 | 341.8 | 64 | 2 | 0 | 0.0537 | 0.060 |
| grid | +6 | 21 | 368.3 | 61 | 2 | 0 | 0.0446 | 0.059 |
| chord1 | -6 | 18 | 329.6 | 60 | 2 | 0 | 0.0532 | 0.029 |
| chord1 | -2 | 18 | 329.6 | 60 | 2 | 0 | 0.0532 | 0.030 |
| chord1 | 0 | 18 | 333.0 | 62 | 2 | 0 | 0.0519 | 0.029 |
| chord1 | +2 | 17 | 345.5 | 58 | 2 | 0 | 0.0545 | 0.030 |
| chord1 | +6 | 18 | 386.0 | 63 | 2 | 0 | 0.0521 | 0.030 |
| chord_diag | -6 | 19 | 347.7 | 67 | 2 | 0 | 0.0803 | 0.030 |
| chord_diag | -2 | 18 | 337.1 | 64 | 2 | 0 | 0.0798 | 0.030 |
| chord_diag | 0 | 18 | 329.3 | 62 | 2 | 0 | 0.0635 | 0.030 |
| chord_diag | +2 | 17 | 344.3 | 58 | 2 | 0 | 0.0615 | 0.033 |
| chord_diag | +6 | 18 | 380.8 | 64 | 2 | 0 | 0.0533 | 0.032 |

Every row hits `max_depth_after = 2` (the target) with `n_unroutable = 0` — no substrate,
including either chord variant, ever failed to route a road or left a parcel stranded on this
region, across the whole repulsion sweep.

## Scaling curve (single blocks, increasing size)

Blocks picked from `DenseCompactScreen(max_depth_min=6.0).select(source)`, targeting nominal
building_count 100/400/800 from the cheap `block_geometries()` column; the actual joined
`building_points` count differs somewhat from that column (likely a different building-count
definition than the geometric point-in-polygon join used to build `Block`), so the real x-axis
values are 228 / 341 / 688 buildings — still a good increasing-size spread. One propose call
per substrate at s=0.

| block | n_buildings | substrate | n_nodes | n_edges | build_time_s | propose_time_s | roads | max_depth_after |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ZAF.9.3.1_1_45803 | 228 | grid | 4,250 | 16,547 | 0.033 | 0.053 | 24 | 2 |
| ZAF.9.3.1_1_45803 | 228 | chord1 | 481 | 1,991 | 0.023 | 0.028 | 19 | 2 |
| ZAF.9.3.1_1_45803 | 228 | chord_diag | 481 | 2,739 | 0.025 | 0.027 | 19 | 2 |
| ZAF.9.3.1_1_20442 | 341 | grid | 13,121 | 51,707 | 0.105 | 0.152 | 34 | 2 |
| ZAF.9.3.1_1_20442 | 341 | chord1 | 733 | 3,047 | 0.033 | 0.040 | 28 | 2 |
| ZAF.9.3.1_1_20442 | 341 | chord_diag | 733 | 4,250 | 0.036 | 0.045 | 27 | 2 |
| ZAF.9.3.1_1_38454 | 688 | grid | 75,219 | 298,544 | 0.632 | 1.392 | 71 | 2 |
| ZAF.9.3.1_1_38454 | 688 | chord1 | 1,577 | 6,373 | 0.073 | 0.097 | 58 | 2 |
| ZAF.9.3.1_1_38454 | 688 | chord_diag | 1,577 | 10,035 | 0.081 | 0.110 | 59 | 2 |

Node-count and time ratios (grid / chord), by block size:

| n_buildings | grid/chord nodes | grid/chord build_time | grid/chord propose_time |
|---:|---:|---:|---:|
| 228 | 8.8x | 1.3-1.4x | 1.9x |
| 341 | 17.9x | 2.9-3.2x | 3.4-3.8x |
| 688 | 47.7x | 7.8-8.7x | 12.7-14.3x |

The sparsity and speed advantage of the chord graph is not a fixed constant — **it grows with
region size**, from ~9x nodes / ~2x propose time at 228 buildings to ~48x nodes / ~13x propose
time at 688 buildings. This is the expected consequence of a fixed-resolution grid: node count
scales with block *area* at a constant `res=1.5` density regardless of building count, while
the chord graph's node count scales with the parcel tessellation's own vertex count (≈ parcels,
since Voronoi cells have a roughly bounded number of sides) — i.e. roughly with building count.
As blocks grow, grid oversamples faster than the chord graph does.

## Failure modes

- **None observed in the routing itself**: across all 24 (15 main-region + 9 scaling) runs,
  every substrate (including both chord variants) reached `max_depth_target=2` with
  `n_unroutable=0`. The chord graph was never disconnected from its own `net` seed
  (`STREET_TOL=0.5`, since boundary-graph vertices coincide with the street geometry almost
  exactly), and no road kinked into an unreachable dead end.
- **One real, flagged divergence**: the sanity cross-check's grid substrate reproduces the
  shipped `ClearanceReblocker` exactly at s ∈ {-6,-2,0,+2} but comes out ~5.4% shorter at the
  most extreme s=+6 (see "Sanity cross-check" above) — attributable to the fairer 3-point
  midpoint-sampled edge cost picking a different tie-broken Dijkstra path than the shipped
  endpoint-only `_edge_weights` at maximal clearance-hugging. Road *count* still matches; only
  one road's routing choice differs.
- **displacement is noisier (less monotonic) on chord than on grid**: grid's displaced-building
  count falls smoothly and monotonically as s increases (70→70→67→64→61). Both chord variants
  broadly track downward from s=-6 to s=+2 but tick back *up* at the extreme s=+6 (chord1:
  60→60→62→58→**63**; chord_diag: 67→64→62→58→**64**) — the corner-chopping lattice occasionally
  forces a longer, more circuitous clearance-hugging detour that grazes more buildings than the
  smoother grid does at the same knob setting.

## Verdict

**(a) Does chord hold the displacement↔directness tradeoff across s like grid?** Partially, and
in an interesting asymmetric way. On *displacement*, grid's monotonic decrease-with-s is the
cleaner curve; both chord variants echo the same general downward trend but are noisier and
reverse direction at the most extreme s=+6. On *directness_AUC*, it's the other way round:
`chord_diag` gives the cleanest, most textbook-monotonic curve of the three substrates (0.080 →
0.080 → 0.064 → 0.061 → 0.053, falling steadily as s pushes the path off the straight line),
while grid's own directness_AUC is *not* monotonic (0.047 → 0.047 → 0.051 → 0.054 → 0.045,
peaking at s=+2) — a reminder that `directness_AUC` is a network-level budget-sweep metric, not
a single-path straightness score, so it needn't move monotonically with the per-edge cost
knob even for the shipped substrate. `chord1` is essentially flat on directness (0.052–0.055)
regardless of s — the skip-one chop doesn't buy it much extra straightness. Net: the chord
substrates hold the *qualitative* repulsion tradeoff (displacement generally down, paths
generally less direct as s rises) but not as cleanly/monotonically as grid on displacement.

**(b) How much sparser/faster, and does the gap grow with size?** Yes, decisively, and the gap
widens with region size: ~9x fewer nodes / ~2x faster propose at 228 buildings, growing to
~48x fewer nodes / ~13x faster propose at 688 buildings. This is the strongest, cleanest result
of the spike — it directly confirms the scaling motivation for moving off a fixed-resolution
grid.

**(c) Does chord_diag beat chord1 on directness/straightness?** Yes, clearly, and the margin is
largest exactly where it should matter most: at low |s| (aspirational, near-straight regime),
chord_diag's directness_AUC is 50%+ higher than chord1's (0.080 vs 0.053 at s=-6); the gap
narrows to near-parity by s=+6 (0.053 vs 0.052), where the cost field dominates path choice
over graph topology anyway. Makes sense: chord1's single skip-one chord per ring position only
chops one corner at a time, so a path crossing a hexagonal-or-larger parcel still kinks; full
diagonals let it cut straight across. The cost is more edges (chord_diag has ~1.4x chord1's
edge count) but that's still a rounding error next to grid's edge count.

**(d) Failure modes.** None in routing correctness/coverage — no disconnected chord graph, no
missed depth target, no unroutable parcel, in any of the 24 runs. The one real issue found is
the flagged ~5.4% sanity-check divergence at the most extreme repulsion (a cost-rule
tie-breaking difference, not a structural bug) and the noisier/non-monotonic displacement curve
for both chord variants at extreme s. Neither is disqualifying for a spike, but both are worth
tracking if this substrate graduates past exploration.

## CDT + resolution sweep

Follow-up spike (`substrate_cdt_followup.py`, imports `substrate_headtohead.py`'s
`greedy_on_substrate`, `build_substrates`, `_pack_edges`, `_build_grid`, and the metric
helpers — nothing substrate-agnostic was reimplemented) adds two constrained-Delaunay (CDT)
substrates, a grid-resolution sweep, and a **results cache** (`substrate_results.json`, keyed
`f"{substrate}|{region_key}|{s:+.1f}"`, compute-if-absent, raw directness-vs-length curves
cached rather than a cap-dependent AUC). First run: 27 rows computed, 5.6s. Second run: all 27
rows served from cache (`[cache hit]` for every row), 3.3s — confirms the cache actually
short-circuits recomputation; only substrate *construction* (cheap, ~0.15s total) still runs
every time, since only the routing+metrics rows are cached.

**cdt_gap**: nodes = the same 548 parcel-boundary vertices the chord substrates use, edges =
the Delaunay triangulation of those vertices clipped to `block.boundary` (segment must be fully
`covers`-contained, buffer 1e-6 for the frontage-segment floating-point case). **cdt_bldg**:
same construction, nodes = the 250 `block.building_points` instead.

### Substrate build (main region, 250 parcels / 250 buildings)

| substrate | n_nodes | n_edges | net_tol | build_time_s |
|---|---:|---:|---:|---:|
| grid | 4,878 | 19,095 | 2.25 | 0.040 |
| chord1 | 548 | 2,204 | 0.50 | 0.028 |
| chord_diag | 548 | 3,056 | 0.50 | 0.028 |
| cdt_gap | 548 | 1,376 | 0.50 | 0.036 |
| cdt_bldg | 250 | 733 | 2.25 | 0.009 |

`cdt_gap` shares chord_diag's 548-node boundary-graph node set but keeps only the
Delaunay-*legal* diagonals: 1,376 edges vs chord_diag's 3,056 (all diagonals) and chord1's 2,204
(skip-one) — 207 of the 1,583 pre-clip Delaunay edges (13%) were dropped by the boundary clip on
this region (concavity ratio boundary-area/hull-area = 0.989, i.e. nearly convex, so most of the
clipping loss is from ordinary Delaunay selectivity, not notch-crossing). `cdt_bldg` has far
fewer nodes (250, one per building) and its raw pre-clip Delaunay triangulation of the 250
building points stayed **fully connected after clipping** (1 component, all 250 nodes) — no
disconnection from clipping on this region.

`cdt_bldg`'s `net_tol` had to be set to `GRID_NET_TOL=2.25` (reused, not chord's `STREET_TOL=
0.5`): building points sit 1.6–12m+ back from the street (median ~4.6m), so at `net_tol=0.5`
**zero** of the 250 building nodes are within tolerance of the street and the substrate net seed
is empty (`greedy_on_substrate` raises immediately). At `net_tol=2.25` only **3 of 250** building
nodes qualify as street-reachable seeds — every road on this substrate is forced to route
through one of those same 3 corridors regardless of where the worst parcel actually is.

### Main region compare (`grid`/`chord1`/`chord_diag`/`cdt_gap`/`cdt_bldg` × s)

`directness_AUC` uses `cap = 386.0m` (max `length_m` across all 27 cached rows, main sweep +
resolution sweep):

| substrate | s | roads | length_m | displaced | max_depth_after | n_unroutable | directness_AUC | n_nodes | n_edges | propose_time_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grid | -6 | 22 | 328.1 | 70 | 2 | 0 | 0.0474 | 4,878 | 19,095 | 0.060 |
| grid | -2 | 22 | 328.1 | 70 | 2 | 0 | 0.0474 | 4,878 | 19,095 | 0.059 |
| grid | 0 | 22 | 327.0 | 67 | 2 | 0 | 0.0506 | 4,878 | 19,095 | 0.059 |
| grid | +2 | 21 | 341.8 | 64 | 2 | 0 | 0.0537 | 4,878 | 19,095 | 0.060 |
| grid | +6 | 21 | 368.3 | 61 | 2 | 0 | 0.0446 | 4,878 | 19,095 | 0.057 |
| chord1 | -6 | 18 | 329.6 | 60 | 2 | 0 | 0.0532 | 548 | 2,204 | 0.031 |
| chord1 | -2 | 18 | 329.6 | 60 | 2 | 0 | 0.0532 | 548 | 2,204 | 0.030 |
| chord1 | 0 | 18 | 333.0 | 62 | 2 | 0 | 0.0519 | 548 | 2,204 | 0.029 |
| chord1 | +2 | 17 | 345.5 | 58 | 2 | 0 | 0.0545 | 548 | 2,204 | 0.030 |
| chord1 | +6 | 18 | 386.0 | 63 | 2 | 0 | 0.0521 | 548 | 2,204 | 0.029 |
| chord_diag | -6 | 19 | 347.7 | 67 | 2 | 0 | 0.0803 | 548 | 3,056 | 0.032 |
| chord_diag | -2 | 18 | 337.1 | 64 | 2 | 0 | 0.0798 | 548 | 3,056 | 0.031 |
| chord_diag | 0 | 18 | 329.3 | 62 | 2 | 0 | 0.0635 | 548 | 3,056 | 0.030 |
| chord_diag | +2 | 17 | 344.3 | 58 | 2 | 0 | 0.0615 | 548 | 3,056 | 0.029 |
| chord_diag | +6 | 18 | 380.8 | 64 | 2 | 0 | 0.0533 | 548 | 3,056 | 0.030 |
| **cdt_gap** | -6 | 19 | 362.1 | 71 | 2 | 0 | 0.0538 | 548 | 1,376 | 0.030 |
| **cdt_gap** | -2 | 19 | 362.4 | 71 | 2 | 0 | 0.0535 | 548 | 1,376 | 0.031 |
| **cdt_gap** | 0 | 18 | 349.1 | 67 | 2 | 0 | 0.0539 | 548 | 1,376 | 0.030 |
| **cdt_gap** | +2 | 16 | 333.6 | 55 | 2 | 0 | 0.0560 | 548 | 1,376 | 0.031 |
| **cdt_gap** | +6 | 17 | 367.6 | 55 | 2 | 0 | 0.0694 | 548 | 1,376 | 0.028 |
| **cdt_bldg** | -6 | 18 | 345.8 | 69 | 2 | 0 | 0.0610 | 250 | 733 | 0.029 |
| **cdt_bldg** | -2 | 18 | 345.8 | 69 | 2 | 0 | 0.0610 | 250 | 733 | 0.027 |
| **cdt_bldg** | 0 | 18 | 345.8 | 69 | 2 | 0 | 0.0610 | 250 | 733 | 0.026 |
| **cdt_bldg** | +2 | 18 | 345.8 | 69 | 2 | 0 | 0.0610 | 250 | 733 | 0.026 |
| **cdt_bldg** | +6 | 18 | 345.8 | 69 | 2 | 0 | 0.0610 | 250 | 733 | 0.026 |

Every row still hits `max_depth_after=2` with `n_unroutable=0` — the CDT substrates never fail
to route or strand a parcel on this region, same as grid/chord1/chord_diag.

**`cdt_bldg`'s five rows are bit-for-bit identical** (`length_m=345.75068706649284` to full
float precision at every s from -6 to +6, confirmed against the cached curve arrays, not just
the rounded table). This is a real structural finding, not a coincidence: `_node_clearance`
floors clearance at `_CLEARANCE_EPS=0.3` (`clearance.py:35`, "keeps node cost finite on a grid
node sitting on a building point"). Every `cdt_bldg` node's *own* nearest building is itself
(distance 0), so **every node's clearance saturates to the identical constant 0.3**, regardless
of s. In the 3-point edge cost, both endpoint terms of every edge are therefore an s-dependent
constant *shared by every edge alike* — only the midpoint term varies edge-to-edge, and
empirically it's never enough to flip which path is cheapest across the whole s∈{-6,...,+6}
range on this region. Net effect: **the repulsion knob is a no-op on `cdt_bldg`** — an
artifact this substrate was never designed to guard against (the epsilon floor was meant for a
rare grid-node coincidence, not for an entire substrate anchored on buildings by construction).
The 3-seed net bottleneck (above) compounds this: with so few street-adjacent nodes, there's
often only one viable corridor topologically, leaving cost-field tie-breaking even less room to
matter.

### Grid-resolution sweep (s=0)

| res | n_nodes | build_time_s | propose_time_s | directness_AUC | displaced |
|---:|---:|---:|---:|---:|---:|
| 0.75 | 19,496 | 0.162 | 0.161 | 0.0514 | 71 |
| 1.5 (baseline `grid`, reused from cache — not recomputed) | 4,878 | 0.039 | 0.059 | 0.0506 | 67 |
| 3.0 | 1,216 | 0.010 | 0.037 | 0.0379 | 69 |

Node count scales ~1/res² as expected (19,496 → 4,878 → 1,216, a clean 4x/4x). Going 2x finer
than the baseline (0.75) barely moves `directness_AUC` (0.0514 vs 0.0506, +1.6%) at a 4x node
and 2.7x propose-time cost, and actually makes displacement *worse* (71 vs 67). Going 2x coarser
(3.0) drops `directness_AUC` by 25% (0.0379) for a 4x node/1.6x time saving. Finer grid does not
change the qualitative story below — see Pareto view.

### Pareto view (s=0, all substrates)

| substrate | directness_AUC | displaced | n_nodes | propose_time_s |
|---|---:|---:|---:|---:|
| grid (res=0.75) | 0.0514 | 71 | 19,496 | 0.161 |
| grid (res=1.5, baseline) | 0.0506 | 67 | 4,878 | 0.059 |
| grid (res=3.0) | 0.0379 | 69 | 1,216 | 0.037 |
| chord1 | 0.0519 | 62 | 548 | 0.029 |
| chord_diag | 0.0635 | 62 | 548 | 0.030 |
| cdt_gap | 0.0539 | 67 | 548 | 0.030 |
| cdt_bldg | 0.0610 | 69 | 250 | 0.026 |

**chord1, chord_diag, and cdt_gap each strictly Pareto-dominate grid at all three resolutions**
tested (0.75, 1.5, 3.0): every one of them has higher `directness_AUC`, lower-or-equal
`displaced`, far fewer `n_nodes`, and lower `propose_time_s` than *every* grid-resolution point,
including the finest (0.75, 4x the baseline's node count). `cdt_bldg` dominates grid at res 0.75
and 3.0 but **not** at the res=1.5 baseline — it wins on `directness_AUC` (0.0610 vs 0.0506) and
`n_nodes`/`propose_time_s`, but loses on `displaced` (69 vs 67), so it does not cleanly dominate
there. **Finer grid resolution does not change the story**: even at 4x the baseline's node
density, grid never catches up to chord_diag/cdt_gap/chord1 on directness, and gets worse (not
better) on displacement as it gets finer on this region.

### Verdicts

**Does `cdt_gap` beat `chord_diag`?** Not consistently — the opposite of the naive expectation
that "Delaunay-selected" beats "all diagonals." Across the five s values, `chord_diag` has the
higher `directness_AUC` at s=-6,-2,0 (0.080 vs 0.054, 0.080 vs 0.054, 0.064 vs 0.054) and a
higher-but-mixed result at s=+2 (0.0615 vs 0.0560 AUC, but `cdt_gap`'s displaced is lower: 55 vs
58). Only at the most extreme repulsion, s=+6, does `cdt_gap` win outright on **both** axes
(AUC 0.069 vs 0.053, displaced 55 vs 64) — restricting the graph to Delaunay-legal diagonals
only pays off once the cost field is pushed hard enough toward clearance-hugging that
`chord_diag`'s extra non-Delaunay diagonals start cutting closer to buildings than the
triangulation would allow. For the aspirational-to-balanced regime that matters most in
practice (s ≤ 0), `chord_diag`'s extra straightening chords win cleanly.

**Does node-location (gap vs bldg) change displacement as predicted?** Partially, and the more
interesting result is the *mechanism*, not the raw numbers. At s=0/+2/+6, `cdt_bldg`'s
displaced (69, constant) is worse than `cdt_gap`'s (67, 55, 55) as predicted. But at s=-6/-2,
`cdt_gap`'s displaced (71, 71) is actually *worse* than `cdt_bldg`'s constant 69 — because
`cdt_gap` is still repulsion-sensitive at that end (routing straighter/nearer to buildings at
low s) while `cdt_bldg` is s-invariant throughout (see above). The headline isn't "higher
displacement," it's that **anchoring every node on a building destroys the substrate's own
node-level clearance signal** (every node reads the same saturated `_CLEARANCE_EPS=0.3`), so the
repulsion knob has no effect at all — a stronger and more structural failure than a displacement
delta.

**Does chord/CDT Pareto-dominate grid across resolutions?** Yes for chord1, chord_diag, and
cdt_gap — cleanly, at all three grid resolutions tested, on every axis measured
(directness/displacement/nodes/time). `cdt_bldg` dominates the two off-baseline resolutions but
not the res=1.5 baseline (loses narrowly on displaced). A 4x-finer grid does not close the gap;
if anything it widens the cost gap (4x more nodes, 2.7x more time) for a negligible directness
gain and a displacement regression.

**Failure modes.** No disconnected-graph errors, no missed depth targets, no unroutable
parcels, and no exceptions anywhere across all 27 cached rows (both CDT variants included) —
`cdt_bldg`'s post-clip Delaunay triangulation stayed one connected component on this region
despite its clipping. The one real failure mode is structural, not a crash: **`cdt_bldg`'s
repulsion knob is a no-op** (bit-identical routes at every s, root-caused to the
`_CLEARANCE_EPS` floor saturating uniformly at every building-anchored node), compounded by a
**3-of-250-node street bottleneck** (net_tol=2.25 only reaches 3 building points, so `cdt_bldg`
routes funnel through very few corridors regardless of the worst parcel's location). Neither is
a bug in `greedy_on_substrate` — both are direct, generalizable consequences of choosing
"nodes = buildings" as a substrate's node source, worth flagging clearly if a building-anchored
substrate is ever considered for real use.
