# Multiblock, screened by `density_compactness`

*Dense and compact from geometry alone — the tightest, most built-up blocks by building count per perimeter², found without ever peeling a single parcel ring.*

**Metric:** `density × compactness = n/P²  —  dense, compact fabric (no peel)` — one metric drives the screen, region growth, and colouring end to end.

## 1. Screen the metro

`density_compactness` flagged **2,865 of 83,192** blocks. Top-scoring: `ZAF.9.3.1_1_44531` (peel depth 4).

![screen](screen.png)

**Location:** [see the grown region on Google Maps](https://www.google.com/maps/@-34.01690,18.58833,16z).


<a href="https://www.google.com/maps/@-34.01690,18.58833,16z"><img src="maps_qr.png" alt="Google Maps QR" width="120"></a>

## 2. Grow the region

The metric grows a **18-block** region (**4,615 parcels**), mean depth 5.1 rings, mean density 146 bldg/ha.

![region](region.png)

## 3. The permeability frontier (benefit vs added road)

The frontier is the whole trade-off: **permeability** (benefit — the only benefit axis) on the y-axis against **displacement** (cost — the only cost axis) on the x-axis, one line per method. Pareto-dominance — which method buys more permeability for less displacement — reads straight off it (raw per-method samples are in `frontier_permeability.csv`, this dir):

![permeability vs displacement](frontier_ZAF.9.3.1_1_44531.png)

**Before any road is added**, the same region in both colorings: access-depth (blue = at a street, red = deep interior) vs permeability potential (dark = hard to escape, light = easy):

| access-depth | permeability potential |
|---|---|
| ![access-depth](before_depth.png) | ![permeability potential](before_perm.png) |

## 4. Each method on the ground

**Watch each method reblock** — its full road set added busiest-first, each road preceded by whatever it needs to reach the street, so every frame is a network you could actually build. The deep interior drains as the network reaches in:

| Looped Tree | Loop Network | Grid | Frontage (street-priced) | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|---|
| ![Looped Tree](reblock_clearance_looped.gif) | ![Loop Network](reblock_cycle_native.gif) | ![Grid](reblock_euclidean_grid.gif) | ![Frontage (street-priced)](reblock_greedy_arterial_access_displacement.gif) | ![OSM Footpaths](reblock_osm_footpaths.gif) | ![Direct Objective (LP)](reblock_resistance_lp.gif) |

### Matched permeability (primary)

Every method truncated where permeability first reaches the standard target, so this compares **what each spends to get there** — in homes displaced and in metres of road. Pinning the benefit and comparing costs is the sounder direction: both costs appear in their own units, so no exchange rate between homes and metres is needed.

| Method | Road | Displacement | Permeability | Note |
|---|---|---|---|---|
| Looped Tree | 1,301 m | 5.7% | 60.3% |  |
| Loop Network | 1,621 m | 5.9% | 60.5% |  |
| Grid | 3,466 m | 9.7% | 61.2% |  |
| Frontage (street-priced) | 1,870 m | 2.6% | 47.2% | unreached |
| OSM Footpaths | 2,506 m | 9.0% | 60.9% |  |
| Direct Objective (LP) | 2,026 m | 4.9% | 60.1% |  |


Access-depth coloring:

| Looped Tree | Loop Network | Grid | Frontage (street-priced) | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|---|
| ![Looped Tree](after_clearance_looped_perm_depth.png) | ![Loop Network](after_cycle_native_perm_depth.png) | ![Grid](after_euclidean_grid_perm_depth.png) | ![Frontage (street-priced)](after_greedy_arterial_access_displacement_perm_depth.png) | ![OSM Footpaths](after_osm_footpaths_perm_depth.png) | ![Direct Objective (LP)](after_resistance_lp_perm_depth.png) |

Permeability-potential coloring:

| Looped Tree | Loop Network | Grid | Frontage (street-priced) | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|---|
| ![Looped Tree](after_clearance_looped_perm_perm.png) | ![Loop Network](after_cycle_native_perm_perm.png) | ![Grid](after_euclidean_grid_perm_perm.png) | ![Frontage (street-priced)](after_greedy_arterial_access_displacement_perm_perm.png) | ![OSM Footpaths](after_osm_footpaths_perm_perm.png) | ![Direct Objective (LP)](after_resistance_lp_perm_perm.png) |

### Matched displacement (secondary)

Every method truncated to the same displacement %, so this compares the **permeability each buys for the same home-cost**. This lens budgets homes but **not road length**, and the two are not proportional — a metre through a gap displaces far less than a metre through the dense interior. So read `road_m` beside `permeability`: a method showing a higher number at several times the road length has not been shown to be better, only more expensive. Prefer the matched-permeability lens above, which prices both costs.

| Method | Road | Displacement | Permeability | Note |
|---|---|---|---|---|
| Looped Tree | 2,284 m | 10.1% | 74.1% |  |
| Loop Network | 3,063 m | 10.1% | 74.3% |  |
| Grid | 3,554 m | 10.1% | 61.7% |  |
| Frontage (street-priced) | 1,870 m | 2.6% | 47.2% | converged below budget |
| OSM Footpaths | 2,874 m | 10.2% | 63.0% |  |
| Direct Objective (LP) | 4,454 m | 10.0% | 80.7% |  |


Access-depth coloring:

| Looped Tree | Loop Network | Grid | Frontage (street-priced) | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|---|
| ![Looped Tree](after_clearance_looped_disp_depth.png) | ![Loop Network](after_cycle_native_disp_depth.png) | ![Grid](after_euclidean_grid_disp_depth.png) | ![Frontage (street-priced)](after_greedy_arterial_access_displacement_disp_depth.png) | ![OSM Footpaths](after_osm_footpaths_disp_depth.png) | ![Direct Objective (LP)](after_resistance_lp_disp_depth.png) |

Permeability-potential coloring:

| Looped Tree | Loop Network | Grid | Frontage (street-priced) | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|---|
| ![Looped Tree](after_clearance_looped_disp_perm.png) | ![Loop Network](after_cycle_native_disp_perm.png) | ![Grid](after_euclidean_grid_disp_perm.png) | ![Frontage (street-priced)](after_greedy_arterial_access_displacement_disp_perm.png) | ![OSM Footpaths](after_osm_footpaths_disp_perm.png) | ![Direct Objective (LP)](after_resistance_lp_disp_perm.png) |


## How this was generated

This example is machine-generated — one self-logging command emits the data, maps, curves, and this README:

```bash
pixi run python -m scripts.gen_example density_compactness
```
The full run log is in [`run.log`](run.log).

