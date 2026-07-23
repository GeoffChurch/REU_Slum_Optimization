# Multiblock, screened by `depth`

*The deepest street-access fabric: how many parcels a home sits from a street, regardless of crowding.*

**Metric:** `depth = √(n·A)/P  →  true peel rings from a street` — one metric drives the screen, region growth, and colouring end to end.

## 1. Screen the metro

`depth` flagged **13,793 of 83,192** blocks. Top-scoring: `ZAF.9.3.1_1_5810` (peel depth 24).

![screen](screen.jpg)

**Location:** [see the grown region on Google Maps](https://www.google.com/maps/@-33.84562,18.74451,15z).


<a href="https://www.google.com/maps/@-33.84562,18.74451,15z"><img src="maps_qr.png" alt="Google Maps QR" width="120"></a>

## 2. Grow the region

The metric grows a **12-block** region (**11,006 parcels**), mean depth 6.4 rings, mean density 99 bldg/ha.

![region](region.jpg)

## 3. The permeability frontier (benefit vs added road)

The frontier is the whole trade-off: **permeability** (benefit — the only benefit axis) on the y-axis against **displacement** (cost — the only cost axis) on the x-axis, one line per method. Pareto-dominance — which method buys more permeability for less displacement — reads straight off it (raw per-method samples are in `frontier_permeability.csv`, this dir):

![permeability vs displacement](frontier_ZAF.9.3.1_1_5810.png)

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
| clearance_looped | 9,878 m | 10.1% | 78.3% |  |
| euclidean_grid | 12,194 m | 10.2% | 87.8% |  |
| greedy_arterial_repulsion | 11,437 m | 5.7% | 72.3% | converged below budget |
| osm_footpaths | 7,674 m | 3.5% | 20.5% | converged below budget |

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
| clearance_looped | 2,804 m | 2.5% | 60.1% |  |
| euclidean_grid | 3,163 m | 2.8% | 61.6% |  |
| greedy_arterial_repulsion | 3,589 m | 2.7% | 60.1% |  |
| osm_footpaths | 7,674 m | 3.5% | 20.5% | unreached |

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
pixi run python -m scripts.gen_multiblock_example depth
```
The full run log is in [`run.log`](run.log).

