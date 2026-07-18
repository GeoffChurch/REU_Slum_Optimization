# Multiblock, screened by `density_compactness`

*Dense and compact from geometry alone — the tightest, most built-up blocks by building count per perimeter², found without ever peeling a single parcel ring.*

**Metric:** `density × compactness = n/P²  —  dense, compact fabric (no peel)` — one metric drives the screen, region growth, and colouring end to end.

## 1. Screen the metro

`density_compactness` flagged **8,293 of 83,192** blocks. Top-scoring: `ZAF.9.3.1_1_44531` (peel depth 4).

![screen](screen.jpg)

## 2. Grow the region

The metric grows a **19-block** region (**4,677 parcels**), mean depth 5.1 rings, mean density 142 bldg/ha.

![region](region.jpg)

## 3. Compare the methods (two lenses)

**Lens A — every parcel to the depth target:**

| method | target_depth | reached | reached_depth | road_length_m | displacement | pct_displaced | propose_seconds |
|---|---|---|---|---|---|---|---|
| clearance | 3 | True | 3 | 4037.8 | 736.6 | 0.1576 | 0.0 |
| greedy_arterial_buildable | 3 | False | 8 | 1576.0 | 241.9 | 0.0517 | 223.9 |


**Lens B — matched road budget:**

| method | budget_m | external_connectivity | internal_connectivity | displacement | pct_displaced |
|---|---|---|---|---|---|
| clearance | 1576.0 | 0.480567 | 2.66732e-14 | 281.5 | 0.0602 |
| greedy_arterial_buildable | 1576.0 | 0.226954 | 0.323752 | 241.9 | 0.0517 |

