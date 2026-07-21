# Multiblock, screened by `density_compactness`

*Dense and compact from geometry alone — the tightest, most built-up blocks by building count per perimeter², found without ever peeling a single parcel ring.*

**Metric:** `density × compactness = n/P²  —  dense, compact fabric (no peel)` — one metric drives the screen, region growth, and colouring end to end.

## 1. Screen the metro

`density_compactness` flagged **2,013 of 16,200** blocks. Top-scoring: `KEN.30.9_1_3515` (peel depth 2).

![screen](screen.jpg)

**Location:** [see the grown region on Google Maps](https://www.google.com/maps/@-1.24565,36.90874,16z).


<a href="https://www.google.com/maps/@-1.24565,36.90874,16z"><img src="maps_qr.png" alt="Google Maps QR" width="120"></a>

## 2. Grow the region

The metric grows a **89-block** region (**2,809 parcels**), mean depth 2.5 rings, mean density 63 bldg/ha.

![region](region.jpg)

## 3. The method frontier (benefit vs added road)

How far each method's road drives the region's **max access depth**, shown **for reference** (the method budget below is now the external-connectivity outcome) — a dot marks where it first reaches each new integer depth. `clearance` is **continued past its depth target** (a full-drainage run) out to the longest method's road, so every method is compared at the same budget: the as-built `osm_footpaths` network plateaus at its floor while `clearance` reaches the same depth for a fraction of the road:

![access depth vs added road](depth_vs_road_KEN.30.9_1_3515.png)

Each method's benefit as cumulative added road grows — the full trade-off whose fixed-depth and matched-budget slices are tabulated in `lens_a_external.csv` and `lens_b_matched.csv` (this dir). External connectivity (access burden removed), internal connectivity (backup-route redundancy), and displacement (a rising cost):

![external connectivity](curve_external_connectivity_KEN.30.9_1_3515.png)

![internal connectivity](curve_internal_connectivity_KEN.30.9_1_3515.png)

![displacement](displacement_KEN.30.9_1_3515.png)

## 4. Each method on the ground

The same region on the same access-depth colour scale (blue = at a street, red = deep) with displaced buildings marked — so the maps are directly comparable across methods.

**Watch each method reblock** — its full road set added in drainage order, the deep interior draining as the network reaches in:

| clearance | clearance_looped | euclidean_grid | greedy_arterial_buildable |
|---|---|---|---|
| ![clearance](reblock_clearance.gif) | ![clearance_looped](reblock_clearance_looped.gif) | ![euclidean_grid](reblock_euclidean_grid.gif) | ![greedy_arterial_buildable](reblock_greedy_arterial_buildable.gif) |

**Matched road budget** — every method truncated to the same total added road, so this compares the access each *buys for the same cost*:

| clearance_looped | clearance | euclidean_grid | greedy_arterial_buildable |
|---|---|---|---|
| ![clearance_looped](after_clearance_looped_matched.jpg) | ![clearance](after_clearance_matched.jpg) | ![euclidean_grid](after_euclidean_grid_matched.jpg) | ![greedy_arterial_buildable](after_greedy_arterial_buildable_matched.jpg) |

**Matched external-connectivity target** — every method truncated where external connectivity (access-burden removed) reaches 0.70, so this compares the *road each takes* for the same outcome:

| clearance | clearance_looped | euclidean_grid | greedy_arterial_buildable |
|---|---|---|---|
| ![clearance](after_clearance_ext70.jpg) | ![clearance_looped](after_clearance_looped_ext70.jpg) | ![euclidean_grid](after_euclidean_grid_ext70.jpg) | ![greedy_arterial_buildable](after_greedy_arterial_buildable_ext70.jpg) |


## How this was generated

This example is machine-generated — one self-logging command emits the data, maps, curves, and this README:

```bash
pixi run python -m scripts.gen_multiblock_example density_compactness nairobi
```
The full run log is in [`run.log`](run.log).

