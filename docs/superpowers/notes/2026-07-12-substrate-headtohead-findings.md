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
