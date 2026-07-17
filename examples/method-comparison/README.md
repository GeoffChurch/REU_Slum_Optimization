# Method comparison: four reblockers, head-to-head on one deep block

Four reblockers graded head-to-head on the **metric basis** — external connectivity, internal
connectivity, and displacement — on a single deep informal block small enough that even `topology`
runs — it's **single-block-only** (it crashes on a multi-block region: a gappy parcel graph gives it a
disconnected source node). `greedy_arterial` now runs via **CELF/lazy**, so it scales too — the
companion [`multiblock`](../multiblock/) flagship runs the scalable methods, *including arterial*, on
a whole settlement.

The block is **`ZAF.9.3.1_1_40972`** — the deepest block (by the depth proxy `√(n·A)/P`) in a
topology-tractable size window: **263 parcels, up to 7 deep**, auto-picked, no hand tuning.
📍 [See it on Google Maps](https://www.google.com/maps/@-33.97795,18.58064,18z) (every run logs this
link for its selection).

## Reproduce

One run grades both connectivity axes AND the displacement cost, all against **added road length**:
```bash
pixi run python -m reblock.compare data=capetown_full \
  "block_ids=[[ZAF.9.3.1_1_40972]]" \
  methods=[topology,clearance,greedy_arterial_buildable,osm_footpaths] max_blocks=1 \
  all_methods.greedy_arterial_buildable.max_roads=8 \
  desire_source.snapshot=examples/method-comparison/desire_lines_40972.geojson
```
`greedy_arterial_buildable` is configured `lazy: true` (CELF), so its 8 roads take seconds, not the
~14 min the exact greedy needed; `topology` is now the run's only slow pole. `osm_footpaths` loads a
committed OSM snapshot (`desire_lines_40972.geojson`, 22 mapped ways — see
`scripts/fetch_desire_lines_snapshot.py`) so it reproduces offline. The console output — each
selection's locator link plus the per-method frontier terminal points and displacement figures — is
captured in [`run.log`](run.log). (Displacement is now a curve emitted every run, not a separate
`cost=displacement` pass — see the displacement section below.)

## The roads each method builds

Before — every parcel up to 7 deep (dark = deep):

![before](before.jpg)

After, per method (blue = added roads; black = building sites; the depth heatmap goes pale as roads
reach every parcel). `topology`'s whole-graph optimizer blankets the fabric; `greedy_arterial`'s few
through-roads wind between the building clusters (tracing the gaps); `clearance` threads least-cost
roads to the deep interior; `osm_footpaths` is the real, unoptimized informal network people already
walk:

| topology (access-optimal) | clearance (least-cost) |
|---|---|
| ![topology](after_topology.jpg) | ![clearance](after_clearance.jpg) |
| **greedy_arterial** (few roads, real loops) | **osm_footpaths** (as-built, loopiest) |
| ![arterial](after_greedy_arterial.jpg) | ![osm_footpaths](after_osm_footpaths.jpg) |

## The metric basis

A spectral investigation (PCA over metric correlations across a diverse corpus of road networks — see
the [basis derivation](../../docs/superpowers/specs/2026-07-16-metric-basis-reporting-design.md) and
[ρ metric migration](../../docs/superpowers/specs/2026-07-17-redundancy-metric-and-refiner-design.md)
design docs) found road-structure quality is **two orthogonal axes** once road *quantity* is controlled
(quantity is the displacement/cost axis below). The five entangled lenses this example used to report
(access, efficiency, directness, resistance, displacement) collapse to that basis:

- **External connectivity** — reach/drainage of parcels to the *outside* street network (the fraction
  of access-burden removed). This is the former "access" metric, unchanged, just relabeled — egress
  `resistance` loaded on the same axis and is now subsumed by it.
- **Internal connectivity** — richness/redundancy of the network *itself*: backup-route redundancy,
  `commute_ratio` (mean 1 − R/R_geo, the effective-resistance ratio over the noded road∪street graph)
  — how many alternative routes a dwelling has, not just whether it has one. `directness`/`efficiency`
  (ρ=0.99 duplicates that straddled both axes) are retired from reporting in favour of this cleaner,
  orthogonal representative. (It replaced the earlier circuit-rank `cycle_density` representative,
  which was size-blind — it rewarded many tiny loops over a few big useful ones.)

Each axis is read off the **frontier** — the full `(method, block, road_length_m, benefit)` curve saved
to `frontier_{metric}.csv` — rather than a single AUC scalar. An AUC (benefit integrated across the
shared road budget) rewards absolute benefit over benefit-per-road, so a road-efficient method could
rank *below* a pave-everything one; we dropped it. Instead, each method is described by its **terminal
point**: the benefit it reached, the **added road length** it spent (metres), and `% paved`. Curve
legend labels read `method (NNN m)` — the terminal road length, not an AUC. (The x-axis is metres of
added road, not m/ha density — the same budget both the benefit and displacement curves are graded on.)

Terminal frontier point per method per axis (benefit @ added road length, % paved):

| axis | topology | clearance | greedy_arterial | osm_footpaths |
|---|---|---|---|---|
| external connectivity — access-burden removed | **0.921** @ 934 m | 0.827 @ 486 m | 0.764 @ 401 m | 0.761 @ 639 m |
| internal connectivity — backup-route redundancy (mean 1 − R/R_geo) | 0.037 @ 934 m | 0.000 @ 486 m | 0.270 @ 401 m | **0.327** @ 639 m |
| % paved | 38.7% | 20.4% | **16.2%** | 25.2% |

Each method earns a different corner of the frontier:

- **`osm_footpaths`** — the REAL informal network (mapped OSM footpaths, not an optimizer's output) —
  has the **most internal connectivity (0.327)**. That's the striking result: the worn paths people
  already walk are the *loopiest*, a genuinely redundant mesh with the most backup routes per dwelling
  — external connectivity 0.761 at 25.2% paved / 74.7 displaced. The as-built network is a real
  reblock, just not an optimized one (see its render above).
- **`greedy_arterial`** is a close second on internal connectivity (0.270) and buys it at the **least
  paving** (16.2%) and least displacement (62.0, see below): its few through-roads reconnect into the
  street perimeter, closing real loops instead of dead-ending — the most redundancy *per metre of
  road*. Runs via CELF/lazy here and scales to the region (see [`multiblock`](../multiblock/)).
- **`topology`** reaches the most external connectivity (0.921) but at the heaviest paving (38.7%),
  with only middling internal connectivity (0.037): its whole-graph optimizer builds a
  reach-everywhere tree, not a mesh. Single-block-only.
- **`clearance`** is the balanced least-cost option — near-topology external connectivity (0.827) at
  20.4% paved — but the **least internal connectivity (0.000)**: a least-cost drainage tree has no
  backup routes *by construction* (every dwelling reaches the street exactly one way). See
  [`multiblock`](../multiblock/) for its region-scale run.

![external connectivity](curve_external_connectivity.png) ![internal connectivity](curve_internal_connectivity.png)

## Displacement: the homes each road costs

Displacement is now a **curve of its own**, plotted against the same added-road-length x-axis as the
two connectivity axes (`displacement.png`, `displacement_vs_length.csv`) — a rising **cost**: as a
method lays road, how many homes does it destroy?

It's **extent-aware**. Rather than counting only buildings whose *centroid* falls in the road
corridor, each building is a disk (radius = half its nearest-neighbour distance), contributing its
**probability of being grazed**, `c = max(0, 1 − d/r)` (`d` = distance from the point to the road
corridor, `r` = the disk radius). Summed over all 263 buildings, that Σc is the "expected homes
displaced" — it catches roads that clip a footprint's *edge*, which a centroid test misses, so these
figures read higher than a raw centroid count.

Terminal displacement (Σ graze-probability at each method's full road) — a rising cost, so *lower is
better*:

| method | homes displaced | % of homes | terminal internal connectivity (backup-route redundancy) |
|---|---|---|---|
| **greedy_arterial** | **62.0** | **23.6%** | 0.270 |
| osm_footpaths | 74.7 | 28.4% | **0.327** |
| clearance | 87.8 | 33.4% | 0.000 |
| topology | 160.2 | 60.9% | 0.037 |

**`greedy_arterial` and `osm_footpaths` lead on redundancy-per-home** — and they're nearly tied on it
(0.270/62.0 ≈ 0.327/74.7 ≈ 0.0044 redundancy per home displaced). Arterial reaches high internal
connectivity at the *fewest* homes moved; osm_footpaths reaches the *highest* internal connectivity but
displaces more. `clearance` (0.000) and `topology` (0.037) build trees, not meshes, so they add little
redundancy however much road they lay. Every curve rises from 0 with road, so you read the tradeoff
directly: benefit *and* homes-displaced both climb as road is added — pick the point you can afford.

![displacement vs added road length](displacement.png)

**The takeaway:** pick the method by the axis *and* the road you can afford — external connectivity →
`topology` (single block); the most backup-route redundancy → `osm_footpaths` (or `greedy_arterial` for
the most redundancy *per metre of road*); balanced least-cost → `clearance` (see
[`multiblock`](../multiblock/) for the region-scale run).
