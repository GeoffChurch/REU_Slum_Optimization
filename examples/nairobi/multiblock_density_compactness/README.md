# Multiblock, screened by `density_compactness`

*Dense and compact from geometry alone — the tightest, most built-up blocks by building count per perimeter², found without ever peeling a single parcel ring.*

**Metric:** `density × compactness = n/P²  —  dense, compact fabric (no peel)` — one metric drives the screen, region growth, and colouring end to end.

## 1. Screen the metro

`density_compactness` flagged **2,013 of 16,200** blocks. Top-scoring: `KEN.30.9_1_3515` (peel depth 2).

![screen](screen.jpg)

**Location:** [see the grown region on Google Maps](https://www.google.com/maps/@-1.24565,36.90874,16z).

## 2. Grow the region

The metric grows a **89-block** region (**2,809 parcels**), mean depth 2.5 rings, mean density 63 bldg/ha.

![region](region.jpg)

## 3. The method frontier (benefit vs added road)

Each method's benefit as cumulative added road grows — the full trade-off whose fixed-depth and matched-budget slices are tabulated in `lens_a_depth.csv` and `lens_b_matched.csv` (this dir). External connectivity (access burden removed), internal connectivity (backup-route redundancy), and displacement (a rising cost):

![external connectivity](curve_external_connectivity_KEN.30.9_1_3515.png)

![internal connectivity](curve_internal_connectivity_KEN.30.9_1_3515.png)

![displacement](displacement_KEN.30.9_1_3515.png)

