# Method comparison: four reblockers, head-to-head on one deep block

Four reblockers graded head-to-head on the four lenses, on a single deep informal block small enough
that even `topology` runs — it's **single-block-only** (it crashes on a multi-block region: a gappy
parcel graph gives it a disconnected source node). `greedy_arterial` now runs via **CELF/lazy**, so it
scales too — the companion [`multiblock`](../multiblock/) flagship runs the scalable methods,
*including arterial*, on a whole settlement.

The block is **`ZAF.9.3.1_1_40972`** — the deepest block (by the depth proxy `√(n·A)/P`) in a
topology-tractable size window: **263 parcels, up to 7 deep**, auto-picked, no hand tuning.
📍 [See it on Google Maps](https://www.google.com/maps/@-33.97795,18.58064,18z) (every run logs this
link for its selection).

## Reproduce

Length cost — the four lenses + the frontier:
```bash
pixi run python -m reblock.compare data=capetown_full \
  "block_ids=[[ZAF.9.3.1_1_40972]]" \
  methods=[topology,clearance,greedy_arterial_buildable,osm_footpaths] max_blocks=1 \
  all_methods.greedy_arterial_buildable.max_roads=8 \
  desire_source.snapshot=examples/method-comparison/desire_lines_40972.geojson
```
Displacement cost — buildings displaced + `% displaced` (topology dropped: it's the slow pole and a
frontage method):
```bash
pixi run python -m reblock.compare data=capetown_full \
  "block_ids=[[ZAF.9.3.1_1_40972]]" \
  methods=[clearance,greedy_arterial_buildable,osm_footpaths] max_blocks=1 \
  all_methods.greedy_arterial_buildable.max_roads=8 cost=displacement \
  desire_source.snapshot=examples/method-comparison/desire_lines_40972.geojson
```
`greedy_arterial_buildable` is configured `lazy: true` (CELF), so its 8 roads take seconds, not the
~14 min the exact greedy needed; `topology` is now the run's only slow pole. `osm_footpaths` loads a
committed OSM snapshot (`desire_lines_40972.geojson`, 22 mapped ways — see
`scripts/fetch_desire_lines_snapshot.py`) so it reproduces offline. The console output of both
commands — each selection's locator link plus the per-method frontier terminal points and
displacement counts — is captured in [`run.log`](run.log).

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
| **greedy_arterial** (directness) | **osm_footpaths** (as-built) |
| ![arterial](after_greedy_arterial.jpg) | ![osm_footpaths](after_osm_footpaths.jpg) |

## The four lenses

Each lens is now read off the **frontier** — the full `(method, block, road_density_m_per_ha,
benefit)` curve saved to `frontier_{metric}.csv` — rather than a single AUC scalar. An AUC (benefit
integrated across the shared road budget) rewards absolute benefit over benefit-per-road, so a
road-efficient method could rank *below* a pave-everything one; we dropped it. Instead, each method is
described by its **terminal point**: the benefit it reached, the road density it spent (m/ha), and
`% paved`. Curve legend labels read `method (NNN m/ha)` — the terminal road density, not an AUC.

Terminal frontier point per method per lens (benefit @ road density, % paved):

| lens | topology | clearance | greedy_arterial | osm_footpaths |
|---|---|---|---|---|
| access — burden removed | **0.921** @ 620 m/ha | 0.827 @ 323 m/ha | 0.764 @ 267 m/ha | 0.761 @ 425 m/ha |
| resistance — egress removed | **0.626** @ 620 m/ha | 0.425 @ 323 m/ha | 0.389 @ 267 m/ha | 0.365 @ 425 m/ha |
| directness — 1/circuity | 0.121 @ 620 m/ha | 0.053 @ 323 m/ha | **0.257** @ 267 m/ha | 0.069 @ 425 m/ha |
| efficiency — network E | ~0.00 | ~0.00 | ~0.01 | ~0.00 |
| **% paved** | **38.7%** | 20.4% | 16.2% | 25.2% |

Each method earns a different corner of the frontier:

- **`topology`** reaches the most access (0.921) and resistance (0.626), but at the heaviest paving
  (38.7%) — its whole-graph optimizer covers best but spends the most road. Single-block-only.
- **`greedy_arterial`** owns directness (0.257, ~2× the next) at the **least paving** (16.2%) and
  least displacement (32) — straight through-roads, maximal navigability per metre. Runs via
  CELF/lazy here and now scales to the region (see [`multiblock`](../multiblock/)).
- **`osm_footpaths`** is the REAL informal network — the mapped OSM footpaths themselves, not an
  optimizer's output. On this small block the actual paths form a genuine, moderately-direct grid:
  directness 0.069 (it *beats* clearance's 0.053) and access 0.761, at 25.2% paved / 49 displaced. A
  striking reference — the worn paths people already use are a real reblock, just not an optimized
  one (see its render above).
- **`clearance`** is the balanced least-cost option: near-topology-free access (0.827) at 20.4%
  paved, with a depth-target + repulsion knob (see [`multiblock`](../multiblock/)).

![access](curve_access.png) ![resistance](curve_resistance.png)
![directness](curve_directness.png) ![efficiency](curve_efficiency.png)

`efficiency` (network E, all-pairs mean 1/distance) is near-inert at every scale for all four methods
(~0.00–0.01) — the many far-apart parcel pairs swamp it.

## Displacement: navigability per building moved

Re-graded on the **displacement** cost axis (x = buildings whose footprint falls inside the road
corridor, 263 buildings total). Displacement scales with how much road a method builds, so the sparse
methods touch far less fabric:

| method | terminal directness | buildings displaced | % displaced |
|---|---|---|---|
| **greedy_arterial** | **0.26** | **32** | **12%** |
| osm_footpaths | 0.07 | 49 | 19% |
| clearance | 0.05 | 53 | 20% |

(Access reached on the same displacement pass: arterial 0.764/32, osm_footpaths 0.761/49, clearance
0.827/53.)

**Arterial displaces the fewest buildings (32, 12%) for its directness** — the most navigability per
building moved. `osm_footpaths`, the as-built footpath network, sits in between: 49 displaced (19%)
for a directness that already beats clearance. `clearance` displaces the most of the three (53, 20%)
for the least directness, trading displacement for its balanced access. There's no AUC on this axis —
a road-sparing method has no curve width — so this is read from the terminal points, not an integral.

![directness vs displacement](curve_directness_ZAF.9.3.1_1_40972_displacement.png)

**The takeaway:** pick the method by the lens *and* the road you can afford — access/egress →
`topology` (single block); directness at minimal paving → `greedy_arterial`; the honest as-built
baseline → `osm_footpaths`; balanced least-cost → `clearance` (see [`multiblock`](../multiblock/)
for the region-scale run and its depth/repulsion knobs).
