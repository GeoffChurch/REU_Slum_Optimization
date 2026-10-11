# The insulator world: buildings as imperfect insulators (2026-10-11)

The owner (2026-10-11): "We shouldn't exclude homes. Instead, we could maybe have buildings be imperfect insulators, which is perhaps related to transparency. That way, escape time would be finite. They should be fairly good insulators but we should get realistic results (e.g. the worst escape time should be worse, but not massively worse, than any of its neighbors that have open paths)".

Code: `insulator.py` (lifted.py and common.py are used unchanged).
Rows: `insulator_rows/` (parquet is ignored; the CSV summaries named in each section are kept).

## Answer

**No single eps meets all three criteria.**
The criteria want values an order of magnitude apart.

- (a) and (b) want eps of about 1e-3 to 3e-3.
  - (a) Walled-in homes are moderately worse than their open neighbours at 1e-3 (median r 2.5 count, 1.6 area) and 3e-3 (1.4, 1.0).
  - (b) The few homes a finer raster opens match at eps* of about 1.6e-3 (median; spread 5e-4 to 3.5e-3).
- (c) wants eps of 1e-4 or less.
  - At 1e-3, 22422's gate is worth 0.628 against 0.783 today.
  - At 1e-3, homes reached only through a sliver keep 54-63% of the relative escape time the finer raster gives them.
  - At 1e-4 those are 0.764 and 92-94%.

**Where it breaks: at the gated pockets.**
A home behind a sub-cell sliver is almost stranded.
Any eps that lets walled-in homes out also drains the slivered ones.
The finer raster says the slivers' long escape times are real: u / block median is the same at h 0.25 as at h 0.5 (median ratio 1.00-1.04 in every band, 1.02 above 8x the median).
So the leak at the gates is a distortion, not a correction, and the gates are what the optimizers must find.

**Alternative worth testing: let only the walled-in out.**
Prototyped here in four variants (three run on all 31 blocks; section "Confining the leak"), each with the leak confined to the stranded pockets.
- **Gates are kept exactly.** 30848's and 5810's open-home Lens A equals the stored value for every variant and parameter; 22422 and 9712 have no stranded homes, so they score as today. The median open home's u moves by at most 0.2% (the stranded homes' added demand).
- **Two variants meet (a) and (b) at the median.**
  - Local walls at eps 3e-3 (the footprint within a pocket's barrier + 1 m conducts eps): walled-in homes' median r 2.1 (count) / 1.7 (area), p90 8.2 / 5.3, matching the finer raster at the median (log ratio -0.05 / +0.04).
  - The leaf door at door_eps 0.1 (the cheapest crossing opened as a passage, with one closed cell per wall conducting door_eps: a cost per wall crossed): median 1.75 / 1.37, p90 11.8 / 6.6, log ratio -0.05 / -0.11.
  - The residual leak sits next to the pockets (worst open home 0.55 and 0.83).
- **Next to test: a door link.** A one-cell lattice door makes r depend on the corridor's shape: even a fully open corridor leaves a 9-home courtyard at r 37.
  The cleaner model is one conductance per pocket, joining its ring to the street-connected cell across its thinnest wall as an edge in the operator: a cost per wall crossed, leak-free by construction.

**Placement fix, free of parameters.**
Most stranding on large blocks is not walls.
- 36 of 60 stranded homes on the 60 large blocks (60%), and 25 of 88 on the 31 small blocks, are *shadowed*: street-connected open space lies within 1 m of their outline, but in a neighbour's ring.
- 22 of the 60 have no ring cell at all.
- Letting a stranded home use the free cells within 1 m of its own outline, reachable ones first, frees them with no leakage and no eps. Tier 2 below does this for the homes with no ring cell; extending it to the 14 (large) and 18 (small) shadowed tier-1 homes was not run.

**If one global eps is wanted anyway** (the optimizers already search in this world): 1e-4 is the fairly-good-insulator choice.
- It keeps the gates (22422 -0.018, 30848 -0.040 on open homes) and the slivers (92-94%).
- Ordinary blocks barely move (controls: median u ratio 0.998, J_2 shift 0.006).
- But walled-in homes are massively worse (own ring: median r 17.6 count, 10.3 area; p90 52, 33).
- Their u grows like 1/eps, so a few of them can dominate a block's J_2: on 30848 one walled-in home (u 3,104 against a median of 33) cuts SIMP's Lens A from 0.615 over open homes to 0.480 over all homes.

**Surprises.**
- A finer raster frees almost nothing: 77 of the 88 stranded homes are still stranded at h 0.25 (81 at 0.35), behind a median 3.0 m of footprint (IQR 1.8-4.2). NOTES' "0.000 at h <= 0.35" held for its three test blocks only.
- On 30848 the eps world reverses the stored clearings: cheap leads SIMP by 0.020 at eps 0, they tie at 1e-3 (0.603 / 0.604), SIMP leads by 0.022 at 3e-3.
- Stranding on large blocks is rare: 60 of 100,724 homes (0.06%), at most 0.84% of a block (41673).

## The model

**Today.** A home injects its weight over its ring: the free cells within 1 m whose nearest building it is, and only those with a path to the street through open space.
A home with no such cell is STRANDED: it injects nothing and is out of P and J_2.

**The insulator world.** Each cell's open fraction becomes o_eps = o + eps (full - o).
A footprint sub-sample then conducts eps relative to an open one, and every inside cell is an unknown.
This is clear.py's and relax.py's eps world, used here as the metric.
Clearing raises a cell from eps toward 1, so the score stays monotone.
Crossing T metres of footprint over a contact width W costs about what T / eps metres of open channel of width W would.

Placement (`Placement`), by tier:

| tier | the home | injects over |
|---|---|---|
| 0 | a ring cell reaches the street through open space | today's cells (its open side) |
| 1 | ring cells, none reaching the street (today's stranded) | all its ring cells, sealed as they are |
| 2 | no ring cell (the nearest-building rule gave every free cell near it to neighbours), but a free cell within 1 m of its outline reaches the street | those cells: its door on the open side |
| 3 | as 2, but no such cell reaches the street | every free cell within 1 m of its outline |
| 4 | no free cell within 1 m of its outline (footprint all round) | the cells its footprint covers, by sub-samples held |
| 5 | a footprint holding no sub-sample (overlapped by a later footprint) | the inside cell nearest its centroid |

Tiers 0 and 1 are the spec.
Tiers 2-5 are my answer to "a home with no ring cell at all".
A first version put tiers 2-3 inside their own footprint; a home with open space a metre away (63718 home 329) then had r 8 at eps 1e-4 from crossing its own walls.
eps 0 is today's rule (tier 0 only): the reference.

**Two diagnostics per formerly stranded home.**
- `barrier_m`: the fewest metres of footprint on a 4-connected path from street-connected open space to its cells (a step into a cell costs h (1 - o)).
- `shadowed`: a free cell within 1 m of its outline reaches the street, whoever's ring it is in. The spec places a shadowed tier-1 home in its sealed ring, although open space is a metre from its wall.

**Fixed.** h 0.5 with sub-sampled open fractions, K 8, ell 3 m, along 'uni', ground = cells within 1 m of the streets, CARVE, lifted.OFFSET.
eps grid: 0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1.
Populations: count and area (mean 1).
CG relative residual 1e-8 on the small blocks (CPU pyamg) and 1e-6 on the large ones (GPU; the stored Lens A values were scored at 1e-5).
At eps 0 the stored clearings' Lens A is reproduced on all four large blocks (to 2e-8).

## Blocks and measurements

- **Stranded set:** the 31 small blocks of the 220 with stranded > 0 at h 0.5 (`insulator_rows/stranded_ids.txt`): 6,276 homes, 88 stranded.
- **Leakage controls:** ten stranded-0 blocks of the 220 at the building-count deciles (69 -- 327 buildings; `control_ids.txt`).
- **Large, GPU, h 0.5:** 22422 and 30848 (gated), 5810 and 9712 (ordinary).
- **Census:** grid and demand only, on the 60 large blocks of road_rescore's stage 1.

Measurements:
- (a) For each formerly stranded home i, r_i = u_i / max u_j over N(i), its 6 nearest open-access homes by centroid distance, at the same eps.
  Reported over the walled-in homes: tier 2 is left out, since its u does not depend on eps.
  "Own ring" is tier 1 (74 homes); "in footprint" is tiers 3-5 (7 homes).
- (b) eps 0 at h 0.35 and 0.25: which formerly stranded homes reach open space there.
  For those, r_fine is computed with the same N(i), dropping any neighbour not open at that h.
  eps* is the eps at which r(eps) at h 0.5 equals r_fine (log-log interpolation).
- (c) Tier-0 homes: u(eps) / u(0) pooled, and P and J_2 over the same homes per block (the baseline shift 1 - (J_2(eps) / J_2(0))^(1/2)).
  u_open is the part due to the tier-0 demand alone, so it isolates the leak from the demand the stranded homes add.
  Slivers: homes binned by z = u / block median at h 0.5, eps 0, and compared with the finer raster.
- (d) Large blocks: Lens A (area, J_2, D 0.05) of the stored SIMP default and cheap clearings, against the same world with nothing cleared.
  Also shown over the tier-0 homes alone, since a stranded home's J_2 is a floor no stored clearing touches.

## Results

### How common stranding is (`summary_census.csv`)

| set | homes | stranded | share | blocks_any | max_block | tier 1 | tier 2 | tier 3 | tier 4 | tier 5 | shadowed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 60 large | 100724 | 60 | 0.06% | 26 | 0.84% | 33 | 22 | 0 | 5 | 0 | 36 |
| 31 small | 6276 | 88 | 1.40% | 31 | 6.77% | 74 | 7 | 0 | 6 | 1 | 25 |

The share per large block: median 0, p90 0.20%, max 0.84% (41673, 9 homes; area population 0.52%).
36 of the 60 stranded homes on large blocks are shadowed; 22 of them have no ring cell of their own (tier 2) and inject on the open side in the insulator world at any eps.
On the 31 small blocks: 25 of 88.
A finer raster frees few: of the 88, 11 reach open space at h 0.25 (7 walled-in, 4 tier 2) and 7 at h 0.35.
The 77 still stranded at h 0.25 sit behind a median 3.0 m of footprint (IQR 1.8 -- 4.2), the same as at h 0.5.
Some of the openings are flicker: 43365 home 6 is open at h 0.35 and walled in again at h 0.25 (4.5 m).

### (a) The owner's criterion (`summary_a_ratio.csv`, `stranded_homes.csv`)

Own ring (tier 1: 74 homes, 25 blocks), r = u_i / max u_j over its 6 nearest open-access homes:

| eps | count med | p90 | max | r<=1 | r>3 | area med | p90  | max  | r<=1  | r>3  |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0001 | 17.57 | 52.36 | 70.5 | 0% | 92% | 10.31 | 32.57 | 47.3 | 1% | 77% |
| 0.0003 | 6.22 | 18.73 | 24.0 | 0% | 73% | 3.85 | 11.34 | 16.1 | 9% | 59% |
| 0.001 | 2.47 | 6.61 | 9.0 | 11% | 34% | 1.60 | 3.99 | 5.7 | 16% | 20% |
| 0.003 | 1.36 | 2.74 | 4.4 | 28% | 8% | 1.03 | 1.74 | 2.7 | 47% | 0% |
| 0.01 | 0.99 | 1.50 | 2.2 | 51% | 0% | 0.87 | 1.22 | 1.5 | 76% | 0% |
| 0.03 | 0.86 | 1.17 | 1.4 | 66% | 0% | 0.81 | 1.05 | 1.3 | 82% | 0% |
| 0.1 | 0.82 | 1.06 | 1.2 | 73% | 0% | 0.80 | 1.00 | 1.2 | 89% | 0% |

In footprint (tiers 3-5: 7 homes, 5 blocks):

| eps | count med | p90 | max | r<=1 | r>3 | area med | p90  | max  | r<=1  | r>3  |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0001 | 79.32 | 531.13 | 601.0 | 0% | 86% | 51.80 | 242.02 | 483.5 | 0% | 86% |
| 0.0003 | 27.09 | 178.65 | 203.0 | 0% | 86% | 17.57 | 82.00 | 163.9 | 14% | 86% |
| 0.001 | 8.79 | 55.22 | 63.6 | 14% | 71% | 5.79 | 25.92 | 51.8 | 14% | 71% |
| 0.003 | 3.53 | 19.82 | 23.4 | 14% | 71% | 2.41 | 9.75 | 19.5 | 14% | 29% |
| 0.01 | 1.63 | 7.02 | 8.7 | 29% | 29% | 1.19 | 3.86 | 7.5 | 43% | 14% |
| 0.03 | 1.03 | 3.09 | 4.0 | 43% | 14% | 0.82 | 2.00 | 3.5 | 86% | 14% |
| 0.1 | 0.87 | 1.57 | 2.0 | 71% | 0% | 0.72 | 1.29 | 1.8 | 86% | 0% |

"Worse, but not massively worse" holds between 1e-3 and 3e-3.
At 1e-3 a third of the own-ring homes (count) are more than 3x their worst neighbour.
At 3e-3 half (area) are no worse than their worst neighbour.
At 1e-2 and above a walled-in home is better off than its worst open neighbour more often than not, because leakage drains the neighbours too.
r is not set by the barrier: the eps world drains a pocket through its whole perimeter, so a 0.9 m wall (46233: r 13-23 at 1e-4) can cost more than a 4.6 m one (63718: r ~4), depending on the pocket's demand and how deep its neighbours are.
The tier 3-5 homes start inside footprint material, so their r at small eps reflects their own walls; they are data artifacts (nested or overlapped footprints), and their placement is my choice.

### (b) The finer raster (`summary_b_open.csv`, `summary_b_match.csv`, `fine_homes.csv`)

Walled-in formerly stranded homes that reach open space at a finer h (eps 0):

| h | pop | stranded_at_0_5 | open_at_h | with_r_fine | blocks | r_fine_med | r_fine_min | r_fine_max | star_p10 | star_med | star_p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | area | 81 | 7 | 7 | 5 | 1.55 | 0.91 | 2.65 | 0.00026 | 0.0019 | 0.003 |
| 0.25 | count | 81 | 7 | 7 | 5 | 1.67 | 1.06 | 3.24 | 0.0005 | 0.0016 | 0.0035 |
| 0.35 | area | 81 | 5 | 5 | 3 | 1.70 | 1.10 | 2.10 | 0.00047 | 0.00072 | 0.006 |
| 0.35 | count | 81 | 5 | 5 | 3 | 1.77 | 1.15 | 4.09 | 0.00064 | 0.0013 | 0.0073 |

log(r(eps) at h 0.5 / r_fine at h 0.25), the same 7 homes:

| pop | eps | homes | logratio_med | logratio_p25 | logratio_p75 | logratio_p10 | logratio_p90 |
|---|---|---|---|---|---|---|---|
| area | 0.0001 | 7 | +1.78 | +1.05 | +2.14 | +0.61 | +2.15 |
| area | 0.0003 | 7 | +0.93 | +0.15 | +1.19 | -0.08 | +1.22 |
| area | 0.001 | 7 | +0.28 | -0.31 | +0.33 | -0.50 | +0.38 |
| area | 0.003 | 7 | -0.23 | -0.48 | -0.08 | -0.81 | -0.01 |
| area | 0.01 | 7 | -0.32 | -0.62 | -0.27 | -0.87 | -0.20 |
| area | 0.03 | 7 | -0.35 | -0.64 | -0.29 | -0.84 | -0.25 |
| area | 0.1 | 7 | -0.42 | -0.63 | -0.24 | -0.80 | -0.22 |
| count | 0.0001 | 7 | +2.14 | +1.67 | +2.35 | +1.08 | +2.54 |
| count | 0.0003 | 7 | +1.14 | +0.69 | +1.36 | +0.28 | +1.53 |
| count | 0.001 | 7 | +0.23 | -0.02 | +0.45 | -0.30 | +0.55 |
| count | 0.003 | 7 | -0.27 | -0.35 | -0.11 | -0.60 | +0.04 |
| count | 0.01 | 7 | -0.59 | -0.62 | -0.28 | -0.84 | -0.25 |
| count | 0.03 | 7 | -0.61 | -0.71 | -0.35 | -0.94 | -0.29 |
| count | 0.1 | 7 | -0.59 | -0.73 | -0.39 | -0.99 | -0.30 |

The best single eps is between 1e-3 and 3e-3 (median log ratio +0.23 and -0.27, count).
At either one the p10 -- p90 band of r / r_fine spans a factor of 1.9 -- 2.3.
The sample is small: 7 homes in 5 blocks.

### (c) Leakage (`summary_c_homes.csv`, `summary_c_blocks.csv`, `summary_c_sliver.csv`)

Area population.
Controls: 10 blocks, 1,633 open-access homes. All: 41 blocks, 7,821.
Ratios are u(eps) / u(0) per home.
Shift is the per-block 1 - (J_2(eps) / J_2(0))^(1/2) over the same homes.
Sliver is the median z(eps) / z at h 0.25, for open-access homes with z = u / block median between 4 and 8 (222 homes) and above 8 (84 homes):

| eps | ctrl_med | ctrl_p5 | ctrl_min | all_p5 | all_min | ctrl_shift_med | ctrl_shift_max | all_shift_med | all_shift_max | sliver_4_8 | sliver_8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0001 | 0.998 | 0.990 | 0.886 | 0.970 | 0.486 | 0.006 | 0.025 | 0.006 | 0.453 | 0.949 | 0.940 |
| 0.0003 | 0.994 | 0.970 | 0.743 | 0.914 | 0.243 | 0.017 | 0.062 | 0.027 | 0.651 | 0.903 | 0.827 |
| 0.001 | 0.981 | 0.913 | 0.535 | 0.782 | 0.093 | 0.053 | 0.136 | 0.086 | 0.753 | 0.802 | 0.596 |
| 0.003 | 0.947 | 0.801 | 0.377 | 0.603 | 0.038 | 0.126 | 0.245 | 0.165 | 0.791 | 0.662 | 0.392 |
| 0.01 | 0.860 | 0.605 | 0.249 | 0.395 | 0.016 | 0.261 | 0.413 | 0.318 | 0.837 | 0.510 | 0.257 |
| 0.03 | 0.715 | 0.408 | 0.140 | 0.236 | 0.009 | 0.416 | 0.587 | 0.491 | 0.886 | 0.430 | 0.186 |
| 0.1 | 0.504 | 0.235 | 0.074 | 0.125 | 0.005 | 0.608 | 0.753 | 0.684 | 0.944 | 0.359 | 0.161 |

The finer raster keeps the slivers: z at h 0.25 over z at h 0.5 is 1.00, 1.01, 1.04 and 1.03 in the bands z < 2, 2-4, 4-8 and > 8 (area; count 1.00 -- 1.04).
The long escapes behind slivers are real geometry, not raster artifacts.
The leak lands on them: the blocks with the largest shifts are the ones whose worst open-access home is 13 -- 61x the block median (43365 0.45 at 1e-4, 39257 0.18, 43817 0.11).
The added demand of the formerly stranded homes is legitimate congestion, and small next to the leak.
It adds 0.1 -- 0.2% to the median open-access home's u, more than 8 -- 16% to 1% of homes, and at most doubles a home's u (the homes on a pocket's way out).

### (d) Large blocks (`summary_d_large.csv`)

Lens A (area, J_2, D 0.05) in each world, over every home ("open": over tier-0 homes only):

| eps | 22422 SIMP | 22422 cheap | 30848 SIMP | 30848 cheap | 30848 SIMP open | 30848 cheap open | 5810 SIMP | 5810 cheap | 9712 SIMP | 9712 cheap |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.783 | 0.761 | 0.655 | 0.675 | 0.655 | 0.675 | 0.259 | 0.223 | 0.130 | 0.119 |
| 0.0001 | 0.764 | 0.741 | 0.480 | 0.492 | 0.615 | 0.635 | 0.258 | 0.222 | 0.130 | 0.119 |
| 0.0003 | 0.730 | 0.702 | 0.586 | 0.600 | 0.608 | 0.622 | 0.257 | 0.221 | 0.130 | 0.119 |
| 0.001 | 0.628 | 0.590 | 0.603 | 0.604 | 0.605 | 0.607 | 0.254 | 0.218 | 0.129 | 0.118 |
| 0.003 | 0.447 | 0.389 | 0.597 | 0.575 | 0.597 | 0.576 | 0.247 | 0.210 | 0.126 | 0.116 |
| 0.01 | 0.243 | 0.161 | 0.557 | 0.506 | 0.557 | 0.506 | 0.226 | 0.190 | 0.119 | 0.108 |
| 0.03 | 0.148 | 0.062 | 0.470 | 0.401 | 0.470 | 0.401 | 0.189 | 0.156 | 0.103 | 0.092 |
| 0.1 | 0.087 | 0.026 | 0.318 | 0.250 | 0.318 | 0.250 | 0.127 | 0.103 | 0.071 | 0.061 |

Baseline shift (nothing cleared): J_2 is 1 - (J_2 of tier-0 homes / J_2 at eps 0)^(1/2) (area); P is 1 - P / P0 (count):

| eps | 22422 J_2 | 30848 J_2 | 5810 J_2 | 9712 J_2 | 22422 P | 30848 P | 5810 P | 9712 P |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.0001 | 0.080 | 0.113 | 0.001 | 0.000 | 0.001 | 0.017 | 0.000 | 0.000 |
| 0.0003 | 0.200 | 0.146 | 0.003 | 0.001 | 0.002 | 0.034 | 0.002 | 0.001 |
| 0.001 | 0.425 | 0.192 | 0.009 | 0.003 | 0.006 | 0.075 | 0.008 | 0.003 |
| 0.003 | 0.623 | 0.272 | 0.026 | 0.010 | 0.015 | 0.156 | 0.023 | 0.009 |
| 0.01 | 0.742 | 0.416 | 0.073 | 0.029 | 0.043 | 0.310 | 0.064 | 0.028 |
| 0.03 | 0.793 | 0.578 | 0.161 | 0.072 | 0.099 | 0.489 | 0.145 | 0.071 |
| 0.1 | 0.833 | 0.741 | 0.315 | 0.165 | 0.209 | 0.677 | 0.293 | 0.162 |

22422's gate keeps its value within 0.02 at 1e-4 (SIMP 0.764 against 0.783 today) and loses 0.053 at 3e-4, 0.155 at 1e-3 and 0.540 at 1e-2.
NOTES' eps-world Lens A of 0.75 with nothing cleared at 0.01 is the 0.742 J_2 shift here.
Its area J_2 is dominated by one large home in the pocket (u 1,211 against a median of 30); the count P shift is only 0.006 at 1e-3.
30848's gated pocket leaks even at 1e-4 (J_2 shift 0.113); its stored clearings tie at 1e-3 and swap order from 3e-3.
Its one walled-in home (2523, barrier 0.9 m, shadowed) has u 3,104 at 1e-4 (area), 335 at 1e-3 and 47 at 1e-2 (median 33).
Ordinary blocks move little: at most 0.009 (J_2 shift) and 0.005 (Lens A) up to 1e-3.

### Confining the leak: doors and local walls (`summary_door_<kind>_*.csv`)

Each variant opens footprint only where a formerly stranded home needs it (`DoorKind`, `door <kind>`).
- **metre**: the cheapest 4-connected crossing (fewest footprint metres) from each sealed pocket to street-connected open space conducts door_eps (a cost per metre).
- **leaf**: the same crossing as an open passage, but for one door-leaf cell per wall (the most closed cell of each run) at door_eps (a cost per wall crossed).
- **straight**: the pocket cell nearest street-connected open space, joined to it by a 1 m capsule.
- **local**: no door. The footprint within (the pocket's largest barrier + 1 m) of its cells conducts eps: the eps world confined to each pocket's surroundings.
Tiers 4-5 also open the footprint cells they inject on.

Own ring (tier 1), r against the 6 nearest open-access homes, metre door:

| door_eps | count med | p90 | max | r<=1 | r>3 | area med | p90  | max  | r<=1  | r>3  |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 20.86 | 198.58 | 223.8 | 0% | 91% | 13.84 | 103.38 | 148.9 | 0% | 85% |
| 0.03 | 7.70 | 69.06 | 107.3 | 3% | 72% | 5.15 | 36.37 | 67.3 | 5% | 62% |
| 0.1 | 3.17 | 25.48 | 55.9 | 11% | 51% | 2.09 | 14.46 | 35.2 | 20% | 41% |
| 0.3 | 1.82 | 10.76 | 42.1 | 19% | 35% | 1.44 | 7.31 | 29.0 | 28% | 31% |
| 1 | 1.23 | 6.50 | 36.9 | 30% | 24% | 1.07 | 5.81 | 27.0 | 41% | 20% |

Local walls (here door_eps is the walls' eps):

| door_eps | count med | p90 | max | r<=1 | r>3 | area med | p90  | max  | r<=1  | r>3  |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0001 | 37.54 | 171.41 | 252.4 | 0% | 99% | 25.45 | 110.40 | 190.8 | 0% | 89% |
| 0.0003 | 13.16 | 60.29 | 84.8 | 0% | 81% | 9.01 | 39.04 | 64.2 | 3% | 76% |
| 0.001 | 4.60 | 20.77 | 26.2 | 7% | 66% | 3.33 | 13.65 | 19.9 | 16% | 53% |
| 0.003 | 2.10 | 8.20 | 10.2 | 22% | 31% | 1.66 | 5.34 | 7.2 | 32% | 24% |
| 0.01 | 1.14 | 3.09 | 5.5 | 42% | 12% | 1.03 | 2.25 | 3.7 | 49% | 5% |

Leaf door (a cost per wall crossed):

| door_eps | count med | p90 | max | r<=1 | r>3 | area med | p90  | max  | r<=1  | r>3  |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 6.73 | 54.00 | 93.4 | 3% | 73% | 3.83 | 38.32 | 51.2 | 7% | 58% |
| 0.03 | 2.98 | 27.44 | 51.5 | 8% | 49% | 1.90 | 14.76 | 32.9 | 23% | 39% |
| 0.1 | 1.75 | 11.78 | 40.9 | 18% | 32% | 1.37 | 6.58 | 28.7 | 30% | 28% |
| 0.3 | 1.37 | 7.20 | 38.0 | 24% | 24% | 1.12 | 5.97 | 27.5 | 38% | 24% |
| 1 | 1.23 | 6.50 | 36.9 | 30% | 24% | 1.07 | 5.81 | 27.0 | 41% | 20% |

Comparison (count; the value nearest the finer raster for each variant):

| (count) | (b) best value (median log r / r_fine at h 0.25) | p10 -- p90 of the log ratio there | open homes: median u / u(0) | worst open home, leak alone (u_open / u(0)) | 30848 open-home Lens A |
|---|---|---|---|---|---|
| global eps | 1e-3 / 3e-3 (+0.23 / -0.27) | -0.30 -- +0.55 / -0.60 -- +0.04 | 0.979 / 0.935 | 0.09 / 0.04 | 0.605 / 0.597 (stored 0.655) |
| metre door | 0.1 (+0.05; area +0.00) | -0.74 -- +0.98 | 1.002 | 0.89 | 0.655 (unchanged) |
| local walls | 3e-3 (-0.05; area +0.04) | -0.51 -- +0.95 | 1.002 | 0.55 | 0.655 (unchanged) |
| leaf door | 0.1 (-0.05; area -0.11) | -0.86 -- +0.09 | 1.002 | 0.83 | 0.655 (unchanged) |

- **The leak is confined.** The median open-access home's u ratio is 1.000-1.002 in every confined variant; the residue is the stranded homes' own added demand.
  30848's and 5810's open-home Lens A equals the stored value for every kind and parameter (to 5 decimals).
  22422 and 9712 have no stranded homes, so every confined variant scores them as today.
  The residual leak sits next to the pockets.
  For the worst open home it is 0.98 -- 0.83 for the metre door (door_eps 0.01 -- 1), 0.83 for the leaf door at any door_eps (from the open passage's side contacts), and 0.55 for the local walls at 3e-3.
- **(a) and (b) at the median.**
  - Local walls at 3e-3: median r 2.1 (count) / 1.7 (area), p90 8.2 / 5.3, median log ratio to the finer raster -0.05 / +0.04.
  - Leaf door at 0.1: median 1.75 / 1.37, p90 11.8 / 6.6, log ratio -0.05 / -0.11, with the narrowest band of the three.
  - Metre door at 0.3: median 1.8 / 1.4, p90 10.8 / 7.3; it matches the finer raster at 0.1, where the median r is 3.2 / 2.1.
- **The tail is geometric.** At door_eps 1 all doors are the same open corridor, and 38486's 9-home courtyard still has r 37.
  The cheapest crossing counts footprint metres only, so it winds along sealed gaps (77 cells, two corridors, about 38 m each).
  A one-cell staircase pays a 90-degree turn per step at ell 3 m.
- **straight** (tested on 38486 only): the 1 m capsule through the thinnest wall cuts the courtyard's r at door_eps 1 from 5.2 -- 37 to 1.5 -- 5.2.
  But it short-circuits open space: the worst open home drops to 0.73. Not run further.

## Recommendation

1. **Don't adopt a global eps as the metric.**
   No value meets (a), (b) and (c).
   The range (a) and (b) want (1e-3 to 3e-3) costs 22422's gate 0.155 to 0.336, and leaves real slivers 35 -- 63% of their relative escape time.
   The value (c) accepts (1e-4) leaves the walled-in homes 10 -- 18x worse than their neighbours (median).
2. **Fix the placement first: the open side for shadowed homes.**
   A stranded home with street-connected open space within 1 m of its outline injects there (tier 2, extended to shadowed tier-1 homes).
   This frees 60% of large-block and 28% of small-block stranding with no parameter and no leak.
   The spec's tier 1 keeps a shadowed home in its sealed ring, which is why some walled-in homes here (30848's 2523, 5810's 277) have open space a metre away.
3. **For the walled-in remainder (0.02% of large-block homes, 1% on the small blocks): confine the leak.**
   Local walls at eps 3e-3, or the leaf door at door_eps 0.1, do it.
   Both meet (a) at the median (r 1.4 -- 2.1) and (b) at the median.
   Both keep every gate and sliver, since the leak is confined to the pockets' surroundings (worst open home 0.55 and 0.83, next to a pocket).
   Weak points: a tail (p90 5 -- 12) from lattice geometry (necks, winding one-cell corridors), and a second parameter for the local walls (the 1 m margin).
   The cleaner model to test next is a **door link**: one conductance g per walled-in pocket (or per home) joining its ring to the street-connected cell across its thinnest wall, as an edge in the operator.
   It is a cost per wall crossed, independent of wall thickness and corridor shape, and leak-free by construction.
   Set g so that the median r is about 2, then check it against the finer raster (b) on more blocks than the 5 here.
4. **If a single eps is needed now** (the optimizers' search world already is one): 1e-4.
   It is the largest value on the grid that keeps the gates within 0.02 (22422) and 0.04 (30848, open homes), and the slivers within 6 -- 8%.
   Report the walled-in homes' J_2 share alongside Lens A, because at 1e-4 one such home can dominate a block (30848).

## Caveats

- (b) rests on 7 homes in 5 blocks (h 0.25) and 5 in 3 (h 0.35); one opening is flicker (43365 home 6). eps* spans 5e-4 -- 3.5e-3.
- N(i) is the 6 nearest open-access homes by centroid. r uses the neighbours' u at the same eps, so leakage lowers the denominator and r stays near 0.8 at large eps.
- A pocket's homes share its way out, so r grows with the pocket's demand: 38486's 9-home courtyard, 63718's 8 homes.
- Tiers 3-5 (7 small-block homes, 5 on large blocks) start inside footprint material. These are nested or overlapped footprints, and their r at small eps is about their own walls.
- One grid offset (lifted.OFFSET). NOTES found offset spread comparable to the h effect for gaps near h; the finer-raster comparison inherits it.
- The small blocks are the 220's (51 -- 759 buildings). Large blocks were checked on four, and the census covers 60.
- The 22422 gate check ran after ollama released the card (48 GB free). With ollama resident (~29 GB), 22422 does not fit at h 0.5.
- The door prototypes are geometric first cuts. Their parameters were not tuned and their corridors are one or two cells wide.
- (c) mixes the leak with the stranded homes' added demand. u_open separates them, and the tables give both.

## Files

- `insulator.py`: `small`, `fine`, `large`, `census`, `door <kind>`, `large-door <kind>`, `report` (the module docstring has the commands).
- Rows (parquet, ignored): `small_h0.5_ell3_K8/`, `fine_h0.35_ell3_K8/`, `fine_h0.25_ell3_K8/`, `large_h0.5_ell3_K8/`, `census_h0.5/`, `door_<kind>_h0.5_ell3_K8/` (straight: 38486 only, so its summaries cover one block).
- Kept: `stranded_ids.txt`, `control_ids.txt`; summaries `summary_a_ratio.csv`, `summary_b_open.csv`, `summary_b_match.csv`, `summary_c_homes.csv`, `summary_c_blocks.csv`, `summary_c_sliver.csv`, `summary_d_large.csv`, `summary_census.csv`, `summary_door_<kind>_{a,b_open,b_match,c,d}.csv`; per home `stranded_homes.csv`, `fine_homes.csv`, `door_<kind>_stranded_homes.csv`.
