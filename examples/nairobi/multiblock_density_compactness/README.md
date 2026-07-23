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

## 3. The permeability frontier (benefit vs added road)

The frontier is the whole trade-off: **permeability** (benefit — the only benefit axis) on the y-axis against **displacement** (cost — the only cost axis) on the x-axis, one line per method. Pareto-dominance — which method buys more permeability for less displacement — reads straight off it (raw per-method samples are in `frontier_permeability.csv`, this dir):

![permeability vs displacement](frontier_KEN.30.9_1_3515.png)

**Before any road is added**, the same region in both colorings: access-depth (blue = at a street, red = deep interior) vs permeability potential (dark = hard to escape, light = easy):

| access-depth | permeability potential |
|---|---|
| ![access-depth](before_depth.jpg) | ![permeability potential](before_perm.jpg) |

## 4. Each method on the ground

**Watch each method reblock** — its full road set added in drainage order, the deep interior draining as the network reaches in:

| clearance_looped | euclidean_grid | greedy_arterial_repulsion |
|---|---|---|
| ![clearance_looped](reblock_clearance_looped.gif) | ![euclidean_grid](reblock_euclidean_grid.gif) | ![greedy_arterial_repulsion](reblock_greedy_arterial_repulsion.gif) |

### Matched displacement

Every method truncated to the same displacement %, so this compares the **permeability each buys for the same home-cost**:

| Method | Road | Displacement | Permeability | Note |
|---|---|---|---|---|
| clearance_looped | 485 m | 0.9% | 15.9% | converged below budget |
| euclidean_grid | 4,459 m | 10.1% | 38.7% |  |
| greedy_arterial_repulsion | 1,820 m | 2.0% | 15.4% | converged below budget |

Access-depth coloring:

| clearance_looped | euclidean_grid | greedy_arterial_repulsion |
|---|---|---|
| ![clearance_looped](after_clearance_looped_disp_depth.jpg) | ![euclidean_grid](after_euclidean_grid_disp_depth.jpg) | ![greedy_arterial_repulsion](after_greedy_arterial_repulsion_disp_depth.jpg) |

Permeability-potential coloring:

| clearance_looped | euclidean_grid | greedy_arterial_repulsion |
|---|---|---|
| ![clearance_looped](after_clearance_looped_disp_perm.jpg) | ![euclidean_grid](after_euclidean_grid_disp_perm.jpg) | ![greedy_arterial_repulsion](after_greedy_arterial_repulsion_disp_perm.jpg) |

### Matched permeability

Every method truncated where permeability first reaches the standard target, so this compares the **displacement each spends** for the same permeability outcome:

| Method | Road | Displacement | Permeability | Note |
|---|---|---|---|---|
| clearance_looped | 485 m | 0.9% | 15.9% | unreached |
| euclidean_grid | 4,670 m | 10.6% | 41.4% |  |
| greedy_arterial_repulsion | 1,820 m | 2.0% | 15.4% | unreached |

Access-depth coloring:

| clearance_looped | euclidean_grid | greedy_arterial_repulsion |
|---|---|---|
| ![clearance_looped](after_clearance_looped_perm_depth.jpg) | ![euclidean_grid](after_euclidean_grid_perm_depth.jpg) | ![greedy_arterial_repulsion](after_greedy_arterial_repulsion_perm_depth.jpg) |

Permeability-potential coloring:

| clearance_looped | euclidean_grid | greedy_arterial_repulsion |
|---|---|---|
| ![clearance_looped](after_clearance_looped_perm_perm.jpg) | ![euclidean_grid](after_euclidean_grid_perm_perm.jpg) | ![greedy_arterial_repulsion](after_greedy_arterial_repulsion_perm_perm.jpg) |


## How this was generated

This example is machine-generated — one self-logging command emits the data, maps, curves, and this README:

```bash
pixi run python -m scripts.gen_multiblock_example density_compactness nairobi
```
The full run log is in [`run.log`](run.log).

