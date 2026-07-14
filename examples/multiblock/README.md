# Multiblock: a whole settlement reblocked, and the methods that scale to it

The headline demonstration that the substrate + screen work make **whole-settlement** reblocking
tractable: screen the entire Cape Town metro, grow its **deepest** informal core, thread roads
through it so every home lands within **3 parcels of a street** — then compare, on that same
10,700-home region, the methods that actually run at this scale, and show clearance's tuning knobs.

Reproduces from **`capetown_full`** (the full metro, auto-downloaded to `~/.cache/reblock` on first
use) via plain CLI commands. For the *comprehensive* method bake-off — all six reblockers, including
the two that don't scale to a region (`topology`, `greedy_arterial`) — see
[`method-comparison`](../method-comparison/) on a single deep block.

## 1. Screen the metro

`dense_compact` flags **13,906 of 83,192** blocks as deep informal fabric, ranked by max
access-depth. The cheap gate is the **depth proxy `√(n·A)/P`** (building count · block area ÷
perimeter) — a closed-form estimate of how many parcel-rings deep a block is, which ranks true depth
~5× better than building density (it's frontage-starvation, not crowding, that buries parcels; see
[the note](../../docs/superpowers/notes/2026-07-14-depth-proxy-screen-gate.md)). The whole ranked
selection is memoized, so the one-time metro pass is 0.1 s on every rerun.

The screen's deepest block is `ZAF.9.3.1_1_5810`, with parcels **24** deep.

![screen](screen.jpg)

## 2. Grow the deep core

`dense_cluster` grows that seed into its neighborhood by the **same depth proxy** (not density — so
growth follows the deep informal fabric instead of wandering into shallow formal housing). A
`max_buildings=3000` budget isolates a clean **23-block deep core** — **10,706 parcels / ~10,700
homes** — the densest, most buried fabric in the metro. The fine grain packed inside vs the sparse
formal grid around it is exactly what the screen is built to find.

![region](region.jpg)

## 3. Reblock to depth 3

```bash
pixi run python -m reblock.run \
  data=capetown_full screen=dense_compact max_blocks=1 \
  region_builder=dense_cluster region_builder.max_buildings=3000 \
  method=clearance method.depth_target=3 method.max_roads=2000 \
  eval=kcomplexity render.enabled=true region_map.enabled=true
```

One command: screen → grow → reblock → render. The clearance reblocker (chord-diagonal substrate)
threads **304 roads / 13,699 m** through the settlement, bringing every parcel from **depth 24 to
depth 3** and displacing **959** homes (sites inside a road corridor), in **~11 s**. `max_blocks=1`
takes the screen's deepest block as the seed; `region_map.enabled` writes `screen.png` + `region.png`
(shown above); `render.enabled` writes the before/after heatmaps below (as `region:…_before.png` /
`…_after.png`). The four dense maps are downsized to JPEG here for the gallery.

| before (depth ≤ 24) | after — depth 3 (304 roads, 959 displaced) |
|---|---|
| ![before](before.jpg) | ![after](after.jpg) |

## 4. Compare the methods that scale

The same 10,700-home region, graded on the four lenses for the methods that run at settlement scale.
`topology` (single-block only) and full `greedy_arterial` (~55 s/road) don't — for the six-method
bake-off on a small block, see [`method-comparison`](../method-comparison/).

```bash
pixi run python -m reblock.compare \
  data=capetown_full region_builder=dense_cluster region_builder.max_buildings=3000 \
  "block_ids=[[ZAF.9.3.1_1_5810]]" methods=[dijkstra,peel,mesh,clearance] max_blocks=1 \
  all_methods.clearance.max_roads=3000
```

| lens | dijkstra | peel | mesh | clearance |
|---|---|---|---|---|
| access — burden removed | 0.94 | 0.94 | 0.94 | **0.96** |
| resistance — egress removed | **0.64** | 0.50 | 0.61 | 0.42 |
| directness — 1/circuity | 0.00 | 0.00 | 0.05 | **0.08** |
| efficiency — network E | 0.00 | 0.00 | 0.00 | 0.00 |

At this scale every method blankets access near-perfectly (~0.94–0.96) — a deep region is exactly
where roads matter most. **clearance is the best all-rounder**: it edges out the coverage methods on
access *and* leads directness (its least-cost roads cut more direct interior routes). **dijkstra wins
resistance** — its frontage spanning tree gives the most redundant egress. So at scale: clearance for
a navigable, well-covered reblock; dijkstra when egress redundancy is the goal.

![access](compare_access.png) ![resistance](compare_resistance.png) ![directness](compare_directness.png)

## 5. The depth_target tradeoff — why 3

`depth_target` is the road-budget dial: the looser the target, the fewer roads. On the deep core:

| depth_target | roads | road length | displaced | reached | access AUC | resistance AUC |
|---|---|---|---|---|---|---|
| 2 | 876 | 26,326 m | 2,056 | depth 2 | **0.903** | **0.233** |
| **3** | **304** | **13,699 m** | **959** | **depth 3** | **0.903** | 0.184 |
| 4 | 137 | 8,345 m | 532 | depth 4 | 0.892 | 0.126 |

Depth 3 is the sweet spot: it removes access-burden at the **same rate per metre** as depth 2
(access AUC 0.903 for both) but with **⅓ the road and half the displacement** — depth 2 just keeps
committing road to shave the last two rings. Depth 4 saves more road still, but leaves parcels 4 deep.
(The resistance lens — egress redundancy — is the one thing depth 2's extra road buys: 0.233 vs 0.184.)

![depth access](depth_access.png) ![depth resistance](depth_resistance.png)

## 6. The repulsion knob — displacement vs directness

At depth 3, `repulsion` (the logit knob steering roads toward gaps vs straight through buildings)
trades **homes displaced** against **route directness**:

| repulsion | roads | road length | displaced | directness AUC |
|---|---|---|---|---|
| −3 (seek clearance) | 295 | 13,387 m | 1,035 | **0.045** |
| 0 (balanced) | 304 | 13,699 m | 959 | 0.029 |
| +3 (repel buildings) | 285 | 14,484 m | **543** | 0.018 |

Turning repulsion up **nearly halves displacement** (1,035 → 543) by routing roads around buildings
through the gaps — at the cost of slightly longer, less direct roads. Turning it down cuts straighter
through the fabric: more direct internal trips, but more homes cleared. Same coverage either way
(access AUC ≈ 0.86); the knob is purely *how* the roads get there.

![repulsion directness](repulsion_directness.png)

## 7. Why it's tractable — the scaling payoff

The whole settlement is reblockable at all because the **chord-diagonal substrate scales with the
fabric, not the area**. The routing graph the reblocker searches has **22,088 nodes** (one region of
parcel-boundary chords). A fixed `res = 1.5 m` grid over the settlement's **232 ha** bounding area
would be **≈ 1,033,034 nodes — 47× more** — and most of them empty space. That O(parcels) vs
O(area/res²) gap is why 10,700 homes reblock in ~11 s on the chord substrate; a grid at the same
resolution would be intractable at this scale.

## Reproduce the compare panels

```bash
# §5 depth_target tradeoff
pixi run python -m reblock.compare \
  data=capetown_full region_builder=dense_cluster region_builder.max_buildings=3000 \
  "block_ids=[[ZAF.9.3.1_1_5810]]" methods=[] max_blocks=1 all_methods.clearance.max_roads=3000 \
  'method_sweep={base: clearance, param: depth_target, values: [2, 3, 4]}'

# §6 repulsion knob (depth pinned to 3)
pixi run python -m reblock.compare \
  data=capetown_full region_builder=dense_cluster region_builder.max_buildings=3000 \
  "block_ids=[[ZAF.9.3.1_1_5810]]" methods=[] max_blocks=1 \
  all_methods.clearance.depth_target=3 all_methods.clearance.max_roads=2000 \
  'method_sweep={base: clearance, param: repulsion, values: [-3, 0, 3]}'
```

(AUCs are comparable *within* a sweep, not across the two — each normalizes to its own shared
road-density axis, and the depth sweep's axis runs much further because depth 2 commits ~3× the road.)
