# Multiblock, screened by `depth`

*The deepest street-access fabric: how many parcels a home sits from a street, regardless of crowding.*

**Metric:** `depth = √(n·A)/P  →  true peel rings from a street` — one metric drives the screen, region growth, and colouring end to end.

## 1. Screen the metro

`depth` flagged **13,793 of 83,192** blocks. Top-scoring: `ZAF.9.3.1_1_5810` (peel depth 24).

![screen](screen.png)

**Location:** [see the grown region on Google Maps](https://www.google.com/maps/@-33.84562,18.74451,15z).


<a href="https://www.google.com/maps/@-33.84562,18.74451,15z"><img src="maps_qr.png" alt="Google Maps QR" width="120"></a>

## 2. Grow the region

The metric grows a **12-block** region (**11,006 parcels**), mean depth 6.4 rings, mean density 99 bldg/ha.

![region](region.png)

## 3. The permeability frontier (benefit vs added road)

The frontier is the whole trade-off: **permeability** (benefit — the only benefit axis) on the y-axis against **displacement** (cost — the only cost axis) on the x-axis, one line per method. Pareto-dominance — which method buys more permeability for less displacement — reads straight off it (raw per-method samples are in `frontier_permeability.csv`, this dir):

![permeability vs displacement](frontier_ZAF.9.3.1_1_5810.png)

**Before any road is added**, the same region in both colorings: access-depth (blue = at a street, red = deep interior) vs permeability potential (dark = hard to escape, light = easy):

| access-depth | permeability potential |
|---|---|
| ![access-depth](before_depth.png) | ![permeability potential](before_perm.png) |

## 4. Each method on the ground

**Watch each method reblock** — its full road set added in drainage order, the deep interior draining as the network reaches in:

| Looped Tree | Grid | Throughways | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|
| ![Looped Tree](reblock_clearance_looped.gif) | ![Grid](reblock_euclidean_grid.gif) | ![Throughways](reblock_greedy_arterial_repulsion.gif) | ![OSM Footpaths](reblock_osm_footpaths.gif) | ![Direct Objective (LP)](reblock_resistance_lp.gif) |

### Matched permeability (primary)

Every method truncated where permeability first reaches the standard target, so this compares **what each spends to get there** — in homes displaced and in metres of road. Pinning the benefit and comparing costs is the sounder direction: both costs appear in their own units, so no exchange rate between homes and metres is needed.

| Method | Road | Displacement | Permeability | Note |
|---|---|---|---|---|
| Looped Tree | 2,818 m | 2.7% | 60.2% |  |
| Grid | 3,163 m | 2.8% | 61.6% |  |
| Throughways | 3,660 m | 2.8% | 60.3% |  |
| OSM Footpaths | 7,674 m | 3.5% | 20.5% | unreached |
| Direct Objective (LP) | 2,302 m | 0.9% | 60.5% |  |

Access-depth coloring:

| Looped Tree | Grid | Throughways | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|
| ![Looped Tree](after_clearance_looped_perm_depth.png) | ![Grid](after_euclidean_grid_perm_depth.png) | ![Throughways](after_greedy_arterial_repulsion_perm_depth.png) | ![OSM Footpaths](after_osm_footpaths_perm_depth.png) | ![Direct Objective (LP)](after_resistance_lp_perm_depth.png) |

Permeability-potential coloring:

| Looped Tree | Grid | Throughways | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|
| ![Looped Tree](after_clearance_looped_perm_perm.png) | ![Grid](after_euclidean_grid_perm_perm.png) | ![Throughways](after_greedy_arterial_repulsion_perm_perm.png) | ![OSM Footpaths](after_osm_footpaths_perm_perm.png) | ![Direct Objective (LP)](after_resistance_lp_perm_perm.png) |

### Matched displacement (secondary)

Every method truncated to the same displacement %, so this compares the **permeability each buys for the same home-cost**. This lens budgets homes but **not road length**, and the two are not proportional — a metre through a gap displaces far less than a metre through the dense interior. So read `road_m` beside `permeability`: a method showing a higher number at several times the road length has not been shown to be better, only more expensive. Prefer the matched-permeability lens above, which prices both costs.

| Method | Road | Displacement | Permeability | Note |
|---|---|---|---|---|
| Looped Tree | 10,040 m | 10.0% | 78.7% |  |
| Grid | 12,194 m | 10.2% | 87.8% |  |
| Throughways | 11,437 m | 5.7% | 72.3% | converged below budget |
| OSM Footpaths | 7,674 m | 3.5% | 20.5% | converged below budget |
| Direct Objective (LP) | 33,660 m | 10.0% | 93.7% |  |

Access-depth coloring:

| Looped Tree | Grid | Throughways | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|
| ![Looped Tree](after_clearance_looped_disp_depth.png) | ![Grid](after_euclidean_grid_disp_depth.png) | ![Throughways](after_greedy_arterial_repulsion_disp_depth.png) | ![OSM Footpaths](after_osm_footpaths_disp_depth.png) | ![Direct Objective (LP)](after_resistance_lp_disp_depth.png) |

Permeability-potential coloring:

| Looped Tree | Grid | Throughways | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|
| ![Looped Tree](after_clearance_looped_disp_perm.png) | ![Grid](after_euclidean_grid_disp_perm.png) | ![Throughways](after_greedy_arterial_repulsion_disp_perm.png) | ![OSM Footpaths](after_osm_footpaths_disp_perm.png) | ![Direct Objective (LP)](after_resistance_lp_disp_perm.png) |


## How this was generated

This example is machine-generated — one self-logging command emits the data, maps, curves, and this README:

```bash
pixi run python -m scripts.gen_multiblock_example depth
```
The full run log is in [`run.log`](run.log).

