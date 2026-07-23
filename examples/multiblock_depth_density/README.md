# Multiblock, screened by `depth_density`

*Deep and crowded at once — the metric that isolates the genuine informal settlements and fades the deep-but-sparse blocks.*

**Metric:** `depth × density  —  deep AND crowded` — one metric drives the screen, region growth, and colouring end to end.

## 1. Screen the metro

`depth_density` flagged **13,822 of 83,192** blocks. Top-scoring: `ZAF.9.3.1_1_38528` (peel depth 13).

![screen](screen.jpg)

**Location:** [see the grown region on Google Maps](https://www.google.com/maps/@-34.00410,18.61263,17z).


<a href="https://www.google.com/maps/@-34.00410,18.61263,17z"><img src="maps_qr.png" alt="Google Maps QR" width="120"></a>

## 2. Grow the region

The metric grows a **3-block** region (**2,690 parcels**), mean depth 8.7 rings, mean density 117 bldg/ha.

![region](region.jpg)

## 3. The permeability frontier (benefit vs added road)

The frontier is the whole trade-off: **permeability** (benefit — the only benefit axis) on the y-axis against **displacement** (cost — the only cost axis) on the x-axis, one line per method. Pareto-dominance — which method buys more permeability for less displacement — reads straight off it (raw per-method samples are in `frontier_permeability.csv`, this dir):

![permeability vs displacement](frontier_ZAF.9.3.1_1_38528.png)

**Before any road is added**, the same region in both colorings: access-depth (blue = at a street, red = deep interior) vs permeability potential (dark = hard to escape, light = easy):

| access-depth | permeability potential |
|---|---|
| ![access-depth](before_depth.jpg) | ![permeability potential](before_perm.jpg) |

## 4. Each method on the ground

**Watch each method reblock** — its full road set added in drainage order, the deep interior draining as the network reaches in:

| clearance_looped | euclidean_grid | greedy_arterial_repulsion | osm_footpaths |
|---|---|---|---|
| ![clearance_looped](reblock_clearance_looped.gif) | ![euclidean_grid](reblock_euclidean_grid.gif) | ![greedy_arterial_repulsion](reblock_greedy_arterial_repulsion.gif) | ![osm_footpaths](reblock_osm_footpaths.gif) |

### Matched displacement

Every method truncated to the same displacement %, so this compares the **permeability each buys for the same home-cost**:

| Method | Road | Displacement | Permeability | Note |
|---|---|---|---|---|
| clearance_looped | 2,334 m | 10.0% | 47.3% |  |
| euclidean_grid | 2,696 m | 10.5% | 43.7% |  |
| greedy_arterial_repulsion | 3,431 m | 10.1% | 42.4% |  |
| osm_footpaths | 4,607 m | 10.1% | 43.2% |  |

Access-depth coloring:

| clearance_looped | euclidean_grid | greedy_arterial_repulsion | osm_footpaths |
|---|---|---|---|
| ![clearance_looped](after_clearance_looped_disp_depth.jpg) | ![euclidean_grid](after_euclidean_grid_disp_depth.jpg) | ![greedy_arterial_repulsion](after_greedy_arterial_repulsion_disp_depth.jpg) | ![osm_footpaths](after_osm_footpaths_disp_depth.jpg) |

Permeability-potential coloring:

| clearance_looped | euclidean_grid | greedy_arterial_repulsion | osm_footpaths |
|---|---|---|---|
| ![clearance_looped](after_clearance_looped_disp_perm.jpg) | ![euclidean_grid](after_euclidean_grid_disp_perm.jpg) | ![greedy_arterial_repulsion](after_greedy_arterial_repulsion_disp_perm.jpg) | ![osm_footpaths](after_osm_footpaths_disp_perm.jpg) |

### Matched permeability

Every method truncated where permeability first reaches the standard target, so this compares the **displacement each spends** for the same permeability outcome:

| Method | Road | Displacement | Permeability | Note |
|---|---|---|---|---|
| clearance_looped | 1,291 m | 6.0% | 40.3% |  |
| euclidean_grid | 2,371 m | 9.0% | 43.0% |  |
| greedy_arterial_repulsion | 3,250 m | 9.4% | 40.3% |  |
| osm_footpaths | 3,850 m | 8.7% | 40.0% |  |

Access-depth coloring:

| clearance_looped | euclidean_grid | greedy_arterial_repulsion | osm_footpaths |
|---|---|---|---|
| ![clearance_looped](after_clearance_looped_perm_depth.jpg) | ![euclidean_grid](after_euclidean_grid_perm_depth.jpg) | ![greedy_arterial_repulsion](after_greedy_arterial_repulsion_perm_depth.jpg) | ![osm_footpaths](after_osm_footpaths_perm_depth.jpg) |

Permeability-potential coloring:

| clearance_looped | euclidean_grid | greedy_arterial_repulsion | osm_footpaths |
|---|---|---|---|
| ![clearance_looped](after_clearance_looped_perm_perm.jpg) | ![euclidean_grid](after_euclidean_grid_perm_perm.jpg) | ![greedy_arterial_repulsion](after_greedy_arterial_repulsion_perm_perm.jpg) | ![osm_footpaths](after_osm_footpaths_perm_perm.jpg) |


## How this was generated

This example is machine-generated — one self-logging command emits the data, maps, curves, and this README:

```bash
pixi run python -m scripts.gen_multiblock_example depth_density
```
The full run log is in [`run.log`](run.log).

