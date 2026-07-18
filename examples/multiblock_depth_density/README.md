# Multiblock, screened by `depth_density`

*Deep and crowded at once — the metric that isolates the genuine informal settlements and fades the deep-but-sparse blocks.*

**Metric:** `depth × density  —  deep AND crowded` — one metric drives the screen, region growth, and colouring end to end.

## 1. Screen the metro

`depth_density` flagged **13,822 of 83,192** blocks. Top-scoring: `ZAF.9.3.1_1_38528` (peel depth 13).

![screen](screen.jpg)

## 2. Grow the region

The metric grows a **3-block** region (**2,690 parcels**), mean depth 8.7 rings, mean density 117 bldg/ha.

![region](region.jpg)

## 3. Compare the methods (two lenses)

**Lens A — every parcel to the depth target:**

| method | target_depth | reached | reached_depth | road_length_m | displacement | pct_displaced | propose_seconds |
|---|---|---|---|---|---|---|---|
| clearance | 3 | True | 3 | 3955.5 | 505.2 | 0.1878 | 0.9 |
| greedy_arterial_buildable | 3 | False | 7 | 4340.5 | 483.3 | 0.1797 | 8.3 |


**Lens B — matched road budget:**

| method | budget_m | external_connectivity | internal_connectivity | displacement | pct_displaced |
|---|---|---|---|---|---|
| clearance | 3955.5 | 0.875523 | 2.56168e-15 | 505.2 | 0.1878 |
| greedy_arterial_buildable | 3955.5 | 0.727197 | 0.333373 | 453.4 | 0.1685 |

