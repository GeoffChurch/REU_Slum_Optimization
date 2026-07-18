# Multiblock, screened by `depth`

*The deepest street-access fabric: how many parcels a home sits from a street, regardless of crowding.*

**Metric:** `depth = √(n·A)/P  →  true peel rings from a street` — one metric drives the screen, region growth, and colouring end to end.

## 1. Screen the metro

`depth` flagged **3,313 of 16,200** blocks. Top-scoring: `KEN.30.6_1_80` (peel depth 15).

![screen](screen.jpg)

**Location:** [see the grown region on Google Maps](https://www.google.com/maps/@-1.31807,36.88120,15z).

## 2. Grow the region

The metric grows a **1-block** region (**4,365 parcels**), mean depth 15.0 rings, mean density 38 bldg/ha.

![region](region.jpg)

## 3. The method frontier (benefit vs added road)

Each method's benefit as cumulative added road grows — the full trade-off whose fixed-depth and matched-budget slices are tabulated in `lens_a_depth.csv` and `lens_b_matched.csv` (this dir). External connectivity (access burden removed), internal connectivity (backup-route redundancy), and displacement (a rising cost):

![external connectivity](curve_external_connectivity_KEN.30.6_1_80.png)

![internal connectivity](curve_internal_connectivity_KEN.30.6_1_80.png)

![displacement](displacement_KEN.30.6_1_80.png)

## 4. Each method on the ground

The same region on the same access-depth colour scale (blue = at a street, red = deep) with displaced buildings marked — so the maps are directly comparable across methods.

**Watch each method reblock** — its full road set added in drainage order, the deep interior draining as the network reaches in:

| clearance | greedy_arterial_buildable | osm_footpaths |
|---|---|---|
| ![clearance](reblock_clearance.gif) | ![greedy_arterial_buildable](reblock_greedy_arterial_buildable.gif) | ![osm_footpaths](reblock_osm_footpaths.gif) |

**Matched road budget** — every method truncated to the same total added road, so this compares the access each *buys for the same cost*:

| clearance | greedy_arterial_buildable | osm_footpaths |
|---|---|---|
| ![clearance](after_clearance_matched.jpg) | ![greedy_arterial_buildable](after_greedy_arterial_buildable_matched.jpg) | ![osm_footpaths](after_osm_footpaths_matched.jpg) |

**Matched access target** — every method truncated where access-depth reaches the target, so this compares the *road each takes* for the same outcome:

| clearance | greedy_arterial_buildable | osm_footpaths |
|---|---|---|
| ![clearance](after_clearance_depth3.jpg) | ![greedy_arterial_buildable](after_greedy_arterial_buildable_depth3.jpg) | ![osm_footpaths](after_osm_footpaths_depth3.jpg) |

