# Multiblock, screened by `depth`

*The deepest street-access fabric: how many parcels a home sits from a street, regardless of crowding.*

**Metric:** `depth = √(n·A)/P  →  true peel rings from a street` — one metric drives the screen, region growth, and colouring end to end.

## 1. Screen the metro

`depth` flagged **13,793 of 83,192** blocks. Top-scoring: `ZAF.9.3.1_1_5810` (peel depth 24).

![screen](screen.jpg)

**Location:** [see the grown region on Google Maps](https://www.google.com/maps/@-33.84562,18.74451,15z).

## 2. Grow the region

The metric grows a **12-block** region (**11,006 parcels**), mean depth 6.4 rings, mean density 99 bldg/ha.

![region](region.jpg)

## 3. The method frontier (benefit vs added road)

How far each method's road drives the region's **max access depth**, shown **for reference** (the method budget below is now the external-connectivity outcome) — a dot marks where it first reaches each new integer depth. `clearance` is **continued past its depth target** (a full-drainage run) out to the longest method's road, so every method is compared at the same budget: the as-built `osm_footpaths` network plateaus at its floor while `clearance` reaches the same depth for a fraction of the road:

![access depth vs added road](depth_vs_road_ZAF.9.3.1_1_5810.png)

Each method's benefit as cumulative added road grows — the full trade-off whose fixed-depth and matched-budget slices are tabulated in `lens_a_external.csv` and `lens_b_matched.csv` (this dir). External connectivity (access burden removed), internal connectivity (backup-route redundancy), and displacement (a rising cost):

![external connectivity](curve_external_connectivity_ZAF.9.3.1_1_5810.png)

![internal connectivity](curve_internal_connectivity_ZAF.9.3.1_1_5810.png)

![displacement](displacement_ZAF.9.3.1_1_5810.png)

## 4. Each method on the ground

The same region on the same access-depth colour scale (blue = at a street, red = deep) with displaced buildings marked — so the maps are directly comparable across methods.

**Watch each method reblock** — its full road set added in drainage order, the deep interior draining as the network reaches in:

| clearance | clearance_looped | greedy_arterial_buildable | osm_footpaths |
|---|---|---|---|
| ![clearance](reblock_clearance.gif) | ![clearance_looped](reblock_clearance_looped.gif) | ![greedy_arterial_buildable](reblock_greedy_arterial_buildable.gif) | ![osm_footpaths](reblock_osm_footpaths.gif) |

**Matched road budget** — every method truncated to the same total added road, so this compares the access each *buys for the same cost*:

| clearance_looped | clearance | greedy_arterial_buildable | osm_footpaths |
|---|---|---|---|
| ![clearance_looped](after_clearance_looped_matched.jpg) | ![clearance](after_clearance_matched.jpg) | ![greedy_arterial_buildable](after_greedy_arterial_buildable_matched.jpg) | ![osm_footpaths](after_osm_footpaths_matched.jpg) |

**Matched external-connectivity target** — every method truncated where external connectivity (access-burden removed) reaches 0.70, so this compares the *road each takes* for the same outcome:

| clearance | clearance_looped | greedy_arterial_buildable | osm_footpaths |
|---|---|---|---|
| ![clearance](after_clearance_ext70.jpg) | ![clearance_looped](after_clearance_looped_ext70.jpg) | ![greedy_arterial_buildable](after_greedy_arterial_buildable_ext70.jpg) | ![osm_footpaths](after_osm_footpaths_ext70.jpg) |

