# Multiblock, screened by `depth_density`

*Deep and crowded at once — the metric that isolates the genuine informal settlements and fades the deep-but-sparse blocks.*

**Metric:** `depth × density  —  deep AND crowded` — one metric drives the screen, region growth, and colouring end to end.

## 1. Screen the metro

`depth_density` flagged **3,354 of 16,200** blocks. Top-scoring: `KEN.30.6_1_109` (peel depth 12).

![screen](screen.png)

**Location:** [see the grown region on Google Maps](https://www.google.com/maps/@-1.31999,36.87177,15z).


<a href="https://www.google.com/maps/@-1.31999,36.87177,15z"><img src="maps_qr.png" alt="Google Maps QR" width="120"></a>

## 2. Grow the region

The metric grows a **7-block** region (**5,095 parcels**), mean depth 7.3 rings, mean density 68 bldg/ha.

![region](region.png)

## 3. The permeability frontier (benefit vs added road)

The frontier is the whole trade-off: **permeability** (benefit — the only benefit axis) on the y-axis against **displacement** (cost — the only cost axis) on the x-axis, one line per method. Pareto-dominance — which method buys more permeability for less displacement — reads straight off it (raw per-method samples are in `frontier_permeability.csv`, this dir):

![permeability vs displacement](frontier_KEN.30.6_1_109.png)

**Before any road is added**, the same region in both colorings: access-depth (blue = at a street, red = deep interior) vs permeability potential (dark = hard to escape, light = easy):

| access-depth | permeability potential |
|---|---|
| ![access-depth](before_depth.png) | ![permeability potential](before_perm.png) |

## 4. Each method on the ground

**Watch each method reblock** — its full road set added busiest-first, each road preceded by whatever it needs to reach the street, so every frame is a network you could actually build. The deep interior drains as the network reaches in:

| Looped Tree | Loop Network | Grid | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|
| ![Looped Tree](reblock_clearance_looped.gif) | ![Loop Network](reblock_cycle_native.gif) | ![Grid](reblock_euclidean_grid.gif) | ![OSM Footpaths](reblock_osm_footpaths.gif) | ![Direct Objective (LP)](reblock_resistance_lp.gif) |

### Matched permeability (primary)

Every method truncated where permeability first reaches the standard target, so this compares **what each spends to get there** — in homes displaced and in metres of road. Pinning the benefit and comparing costs is the sounder direction: both costs appear in their own units, so no exchange rate between homes and metres is needed.

| Method | Road | Displacement | Permeability | Note |
|---|---|---|---|---|
| Looped Tree | 1,803 m | 3.3% | 60.7% |  |
| Loop Network | 2,024 m | 2.8% | 60.0% |  |
| Grid | 8,625 m | 9.0% | 69.1% |  |
| OSM Footpaths | 2,438 m | 1.5% | 31.2% | unreached |
| Direct Objective (LP) | 2,169 m | 1.3% | 60.0% |  |


Access-depth coloring:

| Looped Tree | Loop Network | Grid | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|
| ![Looped Tree](after_clearance_looped_perm_depth.png) | ![Loop Network](after_cycle_native_perm_depth.png) | ![Grid](after_euclidean_grid_perm_depth.png) | ![OSM Footpaths](after_osm_footpaths_perm_depth.png) | ![Direct Objective (LP)](after_resistance_lp_perm_depth.png) |

Permeability-potential coloring:

| Looped Tree | Loop Network | Grid | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|
| ![Looped Tree](after_clearance_looped_perm_perm.png) | ![Loop Network](after_cycle_native_perm_perm.png) | ![Grid](after_euclidean_grid_perm_perm.png) | ![OSM Footpaths](after_osm_footpaths_perm_perm.png) | ![Direct Objective (LP)](after_resistance_lp_perm_perm.png) |

### Matched displacement (secondary)

Every method truncated to the same displacement %, so this compares the **permeability each buys for the same home-cost**. This lens budgets homes but **not road length**, and the two are not proportional — a metre through a gap displaces far less than a metre through the dense interior. So read `road_m` beside `permeability`: a method showing a higher number at several times the road length has not been shown to be better, only more expensive. Prefer the matched-permeability lens above, which prices both costs.

| Method | Road | Displacement | Permeability | Note |
|---|---|---|---|---|
| Looped Tree | 5,986 m | 10.0% | 81.5% |  |
| Loop Network | 9,505 m | 7.8% | 75.6% | converged below budget |
| Grid | 9,620 m | 10.3% | 75.6% |  |
| OSM Footpaths | 2,438 m | 1.5% | 31.2% | converged below budget |
| Direct Objective (LP) | 17,419 m | 10.0% | 92.1% |  |


Access-depth coloring:

| Looped Tree | Loop Network | Grid | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|
| ![Looped Tree](after_clearance_looped_disp_depth.png) | ![Loop Network](after_cycle_native_disp_depth.png) | ![Grid](after_euclidean_grid_disp_depth.png) | ![OSM Footpaths](after_osm_footpaths_disp_depth.png) | ![Direct Objective (LP)](after_resistance_lp_disp_depth.png) |

Permeability-potential coloring:

| Looped Tree | Loop Network | Grid | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|
| ![Looped Tree](after_clearance_looped_disp_perm.png) | ![Loop Network](after_cycle_native_disp_perm.png) | ![Grid](after_euclidean_grid_disp_perm.png) | ![OSM Footpaths](after_osm_footpaths_disp_perm.png) | ![Direct Objective (LP)](after_resistance_lp_disp_perm.png) |


## How this was generated

This example is machine-generated — one self-logging command emits the data, maps, curves, and this README:

```bash
pixi run python -m scripts.gen_example depth_density nairobi
```
The full run log is in [`run.log`](run.log).

