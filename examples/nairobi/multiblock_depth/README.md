# Multiblock, screened by `depth`

*The deepest street-access fabric: how many parcels a home sits from a street, regardless of crowding.*

**Metric:** `depth = √(n·A)/P  →  true peel rings from a street` — one metric drives the screen, region growth, and colouring end to end.

## 1. Screen the metro

`depth` flagged **3,313 of 16,200** blocks. Top-scoring: `KEN.30.6_1_80` (peel depth 15).

![screen](screen.png)

**Location:** [see the grown region on Google Maps](https://www.google.com/maps/@-1.31807,36.88120,15z).


<a href="https://www.google.com/maps/@-1.31807,36.88120,15z"><img src="maps_qr.png" alt="Google Maps QR" width="120"></a>

## 2. Grow the region

The metric grows a **1-block** region (**4,365 parcels**), mean depth 15.0 rings, mean density 38 bldg/ha.

![region](region.png)

## 3. The permeability frontier (benefit vs added road)

The frontier is the whole trade-off: **permeability** (benefit — the only benefit axis) on the y-axis against **displacement** (cost — the only cost axis) on the x-axis, one line per method. Pareto-dominance — which method buys more permeability for less displacement — reads straight off it (raw per-method samples are in `frontier_permeability.csv`, this dir):

![permeability vs displacement](frontier_KEN.30.6_1_80.png)

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
| Looped Tree | 2,215 m | 3.3% | 60.1% |  |
| Loop Network | 1,247 m | 1.1% | 60.3% |  |
| Grid | 6,040 m | 6.3% | 69.2% |  |
| Frontage (street-priced) | 2,498 m | 0.5% | 60.6% |  |
| OSM Footpaths | 2,890 m | 1.9% | 46.6% | unreached |
| Direct Objective (LP) | 1,645 m | 0.9% | 60.1% |  |


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
| Looped Tree | 6,897 m | 10.1% | 77.4% |  |
| Loop Network | 10,669 m | 7.5% | 86.3% | converged below budget |
| Grid | 10,399 m | 10.3% | 82.4% |  |
| Frontage (street-priced) | 5,026 m | 0.9% | 67.6% | converged below budget |
| OSM Footpaths | 2,890 m | 1.9% | 46.6% | converged below budget |
| Direct Objective (LP) | 19,523 m | 10.0% | 95.9% |  |


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
pixi run python -m scripts.gen_example depth nairobi
```
The full run log is in [`run.log`](run.log).

