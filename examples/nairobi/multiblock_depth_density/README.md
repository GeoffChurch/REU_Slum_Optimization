# Multiblock, screened by `depth_density`

*Deep and crowded at once — the metric that isolates the genuine informal settlements and fades the deep-but-sparse blocks.*

**Metric:** `depth × density  —  deep AND crowded` — one metric drives the screen, region growth, and colouring end to end.

## 1. Screen the metro

`depth_density` flagged **3,354 of 16,200** blocks. Top-scoring: `KEN.30.6_1_109` (peel depth 12).

![screen](screen.jpg)

**Location:** [see the grown region on Google Maps](https://www.google.com/maps/@-1.31999,36.87177,15z).

## 2. Grow the region

The metric grows a **7-block** region (**5,095 parcels**), mean depth 7.3 rings, mean density 68 bldg/ha.

![region](region.jpg)

## 3. The method frontier (benefit vs added road)

Each method's benefit as cumulative added road grows — the full trade-off whose fixed-depth and matched-budget slices are tabulated in `lens_a_depth.csv` and `lens_b_matched.csv` (this dir). External connectivity (access burden removed), internal connectivity (backup-route redundancy), and displacement (a rising cost):

![external connectivity](curve_external_connectivity_KEN.30.6_1_109.png)

![internal connectivity](curve_internal_connectivity_KEN.30.6_1_109.png)

![displacement](displacement_KEN.30.6_1_109.png)

## 4. Each method on the ground

The same region on the same access-depth colour scale (blue = at a street, red = deep) with displaced buildings marked — so the maps are directly comparable. Each row shows every method at one **matched** condition:

**Matched road budget** — every method truncated to the same total added road, so this compares the access each *buys for the same cost*:

| clearance | greedy_arterial_buildable | osm_footpaths |
|---|---|---|
| ![clearance](after_clearance_matched.jpg) | ![greedy_arterial_buildable](after_greedy_arterial_buildable_matched.jpg) | ![osm_footpaths](after_osm_footpaths_matched.jpg) |

**Matched access target** — every method truncated where access-depth reaches the target, so this compares the *road each takes* for the same outcome:

| clearance | greedy_arterial_buildable | osm_footpaths |
|---|---|---|
| ![clearance](after_clearance_depth3.jpg) | ![greedy_arterial_buildable](after_greedy_arterial_buildable_depth3.jpg) | ![osm_footpaths](after_osm_footpaths_depth3.jpg) |

