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

**Watch each method reblock** — its full road set added in drainage order, the deep interior draining as the network reaches in:

| Looped Tree | Grid | Worn Paths | Throughways | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|---|
| ![Looped Tree](reblock_clearance_looped.gif) | ![Grid](reblock_euclidean_grid.gif) | ![Worn Paths](reblock_flow_paths.gif) | ![Throughways](reblock_greedy_arterial_repulsion.gif) | ![OSM Footpaths](reblock_osm_footpaths.gif) | ![Direct Objective (LP)](reblock_resistance_lp.gif) |

### Matched displacement

Every method truncated to the same displacement %, so this compares the **permeability each buys for the same home-cost**:

| Method | Road | Displacement | Permeability | Note |
|---|---|---|---|---|
| Looped Tree | 2,750 m | 10.1% | 74.3% |  |
| Grid | 3,945 m | 10.1% | 61.9% |  |
| Worn Paths | 2,976 m | 10.0% | 67.2% |  |
| Throughways | 3,158 m | 10.0% | 54.9% |  |
| OSM Footpaths | 3,196 m | 10.0% | 66.5% |  |
| Direct Objective (LP) | 8,822 m | 10.0% | 80.6% | converged below budget |

Access-depth coloring:

| Looped Tree | Grid | Worn Paths | Throughways | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|---|
| ![Looped Tree](after_clearance_looped_disp_depth.png) | ![Grid](after_euclidean_grid_disp_depth.png) | ![Worn Paths](after_flow_paths_disp_depth.png) | ![Throughways](after_greedy_arterial_repulsion_disp_depth.png) | ![OSM Footpaths](after_osm_footpaths_disp_depth.png) | ![Direct Objective (LP)](after_resistance_lp_disp_depth.png) |

Permeability-potential coloring:

| Looped Tree | Grid | Worn Paths | Throughways | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|---|
| ![Looped Tree](after_clearance_looped_disp_perm.png) | ![Grid](after_euclidean_grid_disp_perm.png) | ![Worn Paths](after_flow_paths_disp_perm.png) | ![Throughways](after_greedy_arterial_repulsion_disp_perm.png) | ![OSM Footpaths](after_osm_footpaths_disp_perm.png) | ![Direct Objective (LP)](after_resistance_lp_disp_perm.png) |

### Matched permeability

Every method truncated where permeability first reaches the standard target, so this compares the **displacement each spends** for the same permeability outcome:

| Method | Road | Displacement | Permeability | Note |
|---|---|---|---|---|
| Looped Tree | 1,474 m | 5.6% | 60.3% |  |
| Grid | 3,678 m | 9.3% | 60.4% |  |
| Worn Paths | 2,403 m | 7.9% | 60.1% |  |
| Throughways | 3,809 m | 11.8% | 60.5% |  |
| OSM Footpaths | 2,460 m | 7.8% | 60.6% |  |
| Direct Objective (LP) | 2,236 m | 4.0% | 60.0% |  |

Access-depth coloring:

| Looped Tree | Grid | Worn Paths | Throughways | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|---|
| ![Looped Tree](after_clearance_looped_perm_depth.png) | ![Grid](after_euclidean_grid_perm_depth.png) | ![Worn Paths](after_flow_paths_perm_depth.png) | ![Throughways](after_greedy_arterial_repulsion_perm_depth.png) | ![OSM Footpaths](after_osm_footpaths_perm_depth.png) | ![Direct Objective (LP)](after_resistance_lp_perm_depth.png) |

Permeability-potential coloring:

| Looped Tree | Grid | Worn Paths | Throughways | OSM Footpaths | Direct Objective (LP) |
|---|---|---|---|---|---|
| ![Looped Tree](after_clearance_looped_perm_perm.png) | ![Grid](after_euclidean_grid_perm_perm.png) | ![Worn Paths](after_flow_paths_perm_perm.png) | ![Throughways](after_greedy_arterial_repulsion_perm_perm.png) | ![OSM Footpaths](after_osm_footpaths_perm_perm.png) | ![Direct Objective (LP)](after_resistance_lp_perm_perm.png) |


## How this was generated

This example is machine-generated — one self-logging command emits the data, maps, curves, and this README:

```bash
pixi run python -m scripts.gen_multiblock_example density_compactness
```
The full run log is in [`run.log`](run.log).

