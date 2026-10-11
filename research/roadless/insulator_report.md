# The insulator world: buildings as imperfect insulators (2026-10-11)

The owner (2026-10-11): "We shouldn't exclude homes. Instead, we could maybe have buildings be imperfect insulators, which is perhaps related to transparency. That way, escape time would be finite. They should be fairly good insulators but we should get realistic results (e.g. the worst escape time should be worse, but not massively worse, than any of its neighbors that have open paths)".

Code: `insulator.py` (lifted.py and common.py are used unchanged).
The owner chose B from "Uniform walls" (2026-10-11, "B SGTM"), and it is now the metric (`lifted.Placement`, `common.Scorer.enclosed_left`; NOTES, "The insulator world").
`insulator.py`, `crossing.py` and `walls.py` were then deleted; their code is at commit 9e10bbf.
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

**Let only the walled-in out.**
Prototyped in four lattice variants (three run on all 31 blocks; section "Confining the leak") and as a door link (section "The door link"), each with the leak confined to the stranded pockets.
- **Gates are kept exactly.** 30848's and 5810's open-home Lens A equals the stored value for every variant and parameter; 22422 and 9712 have no stranded homes, so they score as today. The median open home's u moves by at most 0.2% (the stranded homes' added demand).
- **Two variants meet (a) and (b) at the median.**
  - Local walls at eps 3e-3 (the footprint within a pocket's barrier + 1 m conducts eps): walled-in homes' median r 2.1 (count) / 1.7 (area), p90 8.2 / 5.3, matching the finer raster at the median (log ratio -0.05 / +0.04).
  - The leaf door at door_eps 0.1 (the cheapest crossing opened as a passage, with one closed cell per wall conducting door_eps: a cost per wall crossed): median 1.75 / 1.37, p90 11.8 / 6.6, log ratio -0.05 / -0.11.
  - The residual leak sits next to the pockets (worst open home 0.55 and 0.83).
- **The door link does it with one parameter and no leak** (section "The door link").
  It is one conductance g per walled-in home across its thinnest wall, an edge in the operator joining fully open cells on either side.
  At g = 0.01 (about 63 m of 1 m-wide straight alley) the walled-in homes' median r is 1.8 (count) / 1.5 (area) on the small blocks and 1.6 / 1.5 on the large, and no open home's own escape moves by more than 0.1%.
  Its tail is not the door's: p90 10.5 / 11.0 on the small blocks (4.9 / 3.5 on the large), set by where the thinnest wall leads and by neighbours at the street.
  A lattice door leaves 38486's 9-home courtyard at r 37 even fully open; the door link leaves it at 29 at most.

**Placement fix, free of parameters.**
Most stranding on large blocks is not walls.
- 36 of 60 stranded homes on the 60 large blocks (60%), and 25 of 88 on the 31 small blocks, are *shadowed*: street-connected open space lies within 1 m of their outline, but in a neighbour's ring.
- 22 of the 60 have no ring cell at all.
- Letting a stranded home use the free cells within 1 m of its own outline, reachable ones first, frees them with no leakage and no eps.
  Tier 2 below does this for the homes with no ring cell; the fix extends it to the 14 (large) and 18 (small) shadowed tier-1 homes (section "The door link").
  That leaves 24 walled-in homes on the 60 large blocks (0.024%) and 63 on the 31 small (1.0%).
  Against the finer raster the fix is optimistic for the homes it moves: their r at h 0.5 is about half the finer raster's (8 homes).

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
   It is optimistic for the homes it moves: about half the finer raster's r (8 homes).
3. **For the walled-in remainder (0.024% of large-block homes, 1.0% on the small blocks): a door link per home, g = 0.01.**
   One conductance per walled-in home and pocket, joining the pocket's most open cell to a street-connected cell with no footprint across the thinnest wall, heading-preserving, placed once from the baseline geometry (section "The door link").
   g = 0.01 costs about what 63 m of 1 m-wide straight alley does.
   It meets (a) at the median (r 1.8 / 1.5 small, 1.6 / 1.5 large), opens no footprint and leaks at most 0.1%, keeps the score monotone, and runs on the GPU solver as it is.
   On the 12 large blocks with walled-in homes, Lens A over all homes is within 0.022 of Lens A over open homes, and the walled-in homes hold at most 4.1% of J_2 (12% after clearing): report that share alongside Lens A.
   Weak points: the small blocks' tail (p90 10.5 / 11.0), set by what lies beyond the thinnest wall, and the three homes a finer raster opens, which want g of 0.0015 -- 0.0034.
   The next refinement is a door that weighs the street side, for courtyards whose thinnest wall leads into open space they congest (38486).
   Local walls (eps 3e-3) and the leaf door (door_eps 0.1) also meet (a) at the median, on the spec's placement, but open footprint and leak next to the pockets (worst open home 0.55 and 0.83).
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

## The door link

The two steps recommended above, tested: the placement fix extended to every shadowed home (Task 1), and a door link for the walled-in homes that remain (Task 2).
Code: `insulator.py` (`Placement.of(open_side=True)`; `pockets_of`, `DoorPlan`, `DoorEnds`, `Coupling`, `door_links`, `LinkedSystem`; the commands `link-census`, `link`, `link-large`, `corridor` and `link-report`).
Rows: `insulator_rows/door_link_*/` (parquet, ignored).
A run is named by its door plan, coupling, ends and thickness term: `door_link_<plan>_<coupling>_<ends>_k<kappa>_h0.5_ell3_K8/`, with summaries `summary_door_link_<plan>_<coupling>_<ends>_k<kappa>_*.csv`.

### Answer

**Recommended: one door link per walled-in home, joining open space to open space, heading-preserving, g = 0.01** (the cost of about 63 m of 1 m-wide straight alley at ell 3 m).
- (a) Walled-in homes' median r is 1.80 (count) / 1.51 (area) on the small blocks, p90 10.5 / 11.0, and 1.57 / 1.51 on the large blocks, p90 4.9 / 3.5.
- (b) The three walled-in homes the finer raster opens prefer a lower g (0.0015 -- 0.0034): at 0.01 they get r 0.9 -- 1.2 against 1.1 -- 3.2 at h 0.25.
- (c) Leak-free: no open home's u from the open homes' own demand moves by more than 0.1% (g up to 0.03).
  The demand the walled-in homes add is legitimate congestion: the median open home does not move, p95 rises 4 -- 8%, and one home (38486's 74, on its courtyard's way out) rises 4.9x.
- (d) On the 12 large blocks with walled-in homes, Lens A over all homes trails Lens A over open homes by at most 0.022 (20543, one home inside a footprint) and keeps the stored clearings' order on every block.
  The walled-in homes hold at most 4.1% of J_2 with nothing cleared (median 0.34%) and 12% after SIMP's clearing, which does not touch their doors.

**What breaks: "not massively worse", in the tail of the small blocks.**
At every g that keeps the median near 2, p90 stays at 10 -- 13 there, in every form tested.
The large blocks, with pockets of at most three homes, come close: p90 4.9 / 3.5 at g 0.01, 2.8 / 2.3 at g 0.02 (per home).
The door does not set the tail:
- where the thinnest wall leads: 38486's 9-home courtyard crosses into open space its own demand congests (98x the block median at the door's street end), so with g at 0.3 its homes still have r up to 29 (count);
- neighbours at the street: 21622's home 178 (inside a footprint) has neighbours with u of about 3, so any door that costs 1/g = 100 puts it at r 33;
- a pocket's homes sharing one door, which a door per home removes.

**Choices the brief left open, measured.**
- Ends: the thinnest wall's crossing as `_barrier` computes it starts and ends on the wall's face cells, which are partial (every small-block door's street end; median 1/8 open).
  The door then pours its pocket into a sliver that throttles it at any g (38486: r 95 at g 0.3).
  Joining the nearest fully open cells instead, a rule with no parameter, removes it.
- Per home, not per pocket: a door per pocket makes r grow with the pocket's homes (median 1.3, 4.0, 11.3 for pockets of 1, 2 -- 3, 4 or more at g 0.01, count); a door per home brings pockets of 2 -- 3 homes to 1.8.
- Heading-preserving edges: an isotropic door node changes r by at most 0.3%, and its extra unknown has no cell for the GPU solver's master coarsening.
- The thickness term (1 + kappa barrier_m) / g leaves the tail as wide or wider at matched medians: not kept.

### Task 1: the placement fix, extended (`summary_door_link_census_sets.csv`, `summary_door_link_census.csv`, `door_link_census_homes.csv`)

A stranded home with a free cell within 1 m of its own outline that reaches the street injects on those cells (tier 2), whether or not it also has a sealed ring.
Tiers under the spec (left of the arrow) and under the fix:

| set | homes | stranded | tier 1 | tier 2 | tier 4 | tier 5 | walled in | share | blocks | pockets | homes per pocket, max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 small | 6,276 | 88 | 74 → 56 | 7 → 25 | 6 | 1 | 63 | 1.00% | 24 | 57 | 9 |
| 60 large | 100,724 | 60 | 33 → 19 | 22 → 36 | 5 | 0 | 24 | 0.024% | 12 | 29 | 3 |

- No home is tier 3 in either set, so the walled-in homes are tier 1 (sealed in their own ring) and tiers 4 -- 5 (footprint all round: 7 on the small blocks, 5 on the large).
- On the large blocks: 41673 has 5, 41806 4, 22267 3; 21252, 23597 and 43708 2 each; 20543, 22336, 38254, 39240, 41515 and 5706 1 each.
  At most 0.47% of a block's homes (41673; 0.22% by area).
- On the small blocks (24 of the 31): 38486 has 9 (one courtyard), 63718 7, 21622 6, 19537 4; 19263, 19538, 43353, 43817, 43874 and 44837 3 each; 21708, 39257, 43365, 43876 and 46233 2 each; nine blocks 1 each.
  At most 6.8% of a block (38486).
- They sit behind a median 3.0 m of footprint (IQR 1.7 -- 4.0, max 7.5) on the small blocks and 2.3 m (1.5 -- 3.7, max 5.0) on the large.
- 30848 and 5810 have no walled-in home under the fix (2523 and 277 become tier 2), so the door link leaves them as today but for those homes' demand, now on the open side.
- A home whose ring touches several sealed nooks has a pocket in each (41673: 5 homes, 7 pockets; 22267: 3 homes, 7 pockets).
  Of the 63 small-block homes, 22 are alone in their main pocket, 11 share it with one other home, 17 with two, 4 with three, and 9 form 38486's courtyard.

The fix against the finer raster (`summary_door_link_b_fix.csv`): eight tier-2 homes reach open space at h 0.25 (four the fix moved, four tier 2 under the spec too).
For all eight, r at h 0.5 under the fix is below the finer raster's: median log(r / r_fine) -0.75 (count) / -0.45 (area); at h 0.35 (five homes), -0.32 / -0.28.
The four the fix moved: 42454's 120 -0.31, 43817's 153 -0.80, 44274's 55 -1.88, 46233's 139 -0.69 (count).
In the eps world (1e-3, their sealed ring) the same four were +0.23 to +0.69.
The fix borrows cells on a neighbour's open side; at h 0.25 the nearest-building rule gives each home its own ring, behind the sliver, so the fix is optimistic for them by about 2x.

### The door link: model

- **Pocket.** A connected set (4-connected, as the operator) of sealed free cells holding the injection cells of at least one walled-in home.
  For tiers 4 -- 5, the closed cells they inject on, opened fully (the home's own interior): no other footprint is opened.
- **Door.** One extra conductance g joining a pocket cell a to a street-connected cell b across the pocket's thinnest wall: the crossing with the fewest footprint metres, `_crossing_graph`'s Dijkstra as in `_barrier`.
  It is placed once, from the baseline geometry, and kept in every scenario: clearing only raises conductances, so the score stays monotone.
  Per pocket (`PocketDoors`): one door shared by the pocket's homes, from the pocket cell its cheapest crossing reaches.
  Per home (`HomeDoors`): one door per walled-in home and pocket it injects into, from the home's own cell its cheapest crossing reaches (a pocket's doors may share b).
  Small blocks: 57 pocket doors or 90 home doors; large blocks: 29 or 32.
- **Ends** (`DoorEnds`).
  Face (the crossing as `_barrier` computes it): b is any street-connected cell, a any pocket cell.
  The cheapest crossing then starts and ends on the wall's face cells, which are partial: every small-block door's street end (median 1/8 open; 82% under half), and 90% on the large blocks (median 1/4).
  Open: b is a street-connected cell with no footprint, a one of the pocket's (or the home's) most open cells, and the crossing between them counts the face cells' footprint as wall (median crossing 3.8 m against 3.3 m on the small blocks).
  Every open door's street end is fully open; 9% of its pocket ends are less than half open (pockets with no open cell).
- **Heading coupling** (`Coupling`).
  Heading-preserving: per axis k an edge (a, k)-(b, k) of g m_k / pi, so the door totals g and adds no unknown: `GpuAMG`'s master coarsening still maps every unknown to a cell.
  Isotropic node (the check, CPU only): one new unknown per door, joined to every axis of a and of b by 2 g m_k / pi (total g in series).
- **Thickness** (the variant): link resistance (1 + kappa barrier_m) / g, a cost per wall plus kappa / g per metre.
- **The solve** (`LinkedSystem`).
  `lifted.System` drops every unknown whose component has no ground before a door exists, so the linked system is assembled here: `lifted._assemble`, the door edges and their diagonal terms appended, `keep` recomputed on the connectivity with the links, then `lifted._csr` and the solver's setup.
  pyamg on the linked operator is unreliable: its setup starts the prolongation smoother's spectral radius from a random vector, and at strength theta 0 a door is a strong edge an aggregate can straddle.
  On 38486 at g 0.03, two of four seeds diverged, and a deterministic (Gershgorin) weighting stalled at a residual of 1.7.
  On the CPU, CG on the linked operator is therefore preconditioned by pyamg built on the same operator without the doors' off-diagonal entries (`CpuCutAMG`; 33 iterations on every seed there); on the GPU, `GpuAMG` runs unchanged, its aggregates coming from the master operator, which has no doors (`GpuMasterAMG`).
  Either way the CG residual is taken on the operator with its doors: the worst recorded is 1.0e-8 on the small blocks (rtol 1e-8) and 1.0e-6 on the large (rtol 1e-6).
  Checks: with no links, the operator equals `lifted.System`'s and u agrees to 1e-11.
  Each pocket's door current equals the demand injected into it to 1e-10 for both couplings.
  Dropping the doors' diagonal terms makes the CG fail.

g in metres (`corridor`, `summary_door_link_corridor.csv`): a 20 m-wide synthetic block, a street along one edge, 6 m of open apron, a wall L thick with a 1 m gap, and above it a 10 x 6 m room, closed but for the gap.
A unit demand is spread over the room, and u_room is its mean potential.
With the gap open, u_room rises by 1.62 per metre of corridor; walled up with one door (open ends), u_room = 3.36 + 1/g.
So g = 0.63 / L for L of 16 m and more:

| g | 0.003 | 0.005 | 0.01 | 0.02 | 0.03 | 0.1 | 0.3 |
|---|---|---|---|---|---|---|---|
| metres of 1 m corridor (open ends) | 207 | 125 | 63 | 32 | 22 | 7 | 3 |

The isotropic node matches within 1%.
With face ends even an infinite door costs as much as 3 m of corridor (u_room = 6.4 + 1/g; open ends: 1 m).
A 0.5 m wall has no closed row at h 0.5, so the shortest reference is 1 m.

### (a) The owner's criterion (`summary_door_link_<run>_a.csv`, `door_link_<run>_stranded_homes.csv`)

r = u_i / max u_j over the 6 nearest open-access (tier 0) homes, over the walled-in homes under the fix (63 homes in 24 small blocks; 24 in 12 large).

Per home, open ends (the recommended form), small blocks:

| g | metres | count med | p90 | max | r<=1 | r>3 | area med | p90 | max | r<=1 | r>3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 | - | 10.80 | 43.5 | 321.1 | 2% | 83% | 7.89 | 27.4 | 243.4 | 2% | 68% |
| 0.002 | - | 5.94 | 30.8 | 160.9 | 2% | 63% | 4.11 | 20.2 | 122.0 | 6% | 59% |
| 0.003 | 207 | 4.29 | 21.2 | 107.5 | 3% | 59% | 3.07 | 14.6 | 81.5 | 11% | 51% |
| 0.005 | 125 | 2.86 | 13.6 | 64.8 | 10% | 46% | 2.22 | 12.3 | 49.1 | 19% | 30% |
| 0.01 | 63 | 1.80 | 10.5 | 32.8 | 21% | 29% | 1.51 | 11.0 | 29.0 | 29% | 19% |
| 0.02 | 32 | 1.32 | 9.9 | 28.1 | 25% | 21% | 1.04 | 10.6 | 28.3 | 41% | 17% |
| 0.03 | 22 | 1.15 | 9.7 | 27.8 | 43% | 17% | 0.99 | 8.3 | 28.1 | 51% | 16% |
| 0.05 | 13 | 1.03 | 7.3 | 27.6 | 49% | 17% | 0.94 | 6.6 | 27.9 | 57% | 16% |
| 0.1 | 7 | 0.95 | 7.2 | 27.4 | 54% | 16% | 0.89 | 6.5 | 27.7 | 57% | 16% |
| 0.3 | 3 | 0.87 | 7.2 | 27.3 | 57% | 14% | 0.86 | 6.5 | 27.6 | 63% | 14% |

Per pocket, open ends, small blocks:

| g | metres | count med | p90 | max | r<=1 | r>3 | area med | p90 | max | r<=1 | r>3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 | - | 17.26 | 110.5 | 321.1 | 2% | 89% | 12.37 | 69.3 | 243.4 | 2% | 75% |
| 0.002 | - | 9.23 | 56.0 | 160.9 | 2% | 76% | 6.82 | 43.2 | 122.0 | 3% | 67% |
| 0.003 | 207 | 6.53 | 38.0 | 107.5 | 3% | 70% | 4.88 | 29.0 | 81.5 | 8% | 62% |
| 0.005 | 125 | 4.26 | 23.3 | 64.8 | 5% | 59% | 3.33 | 18.3 | 49.1 | 11% | 51% |
| 0.01 | 63 | 2.64 | 12.8 | 43.1 | 16% | 46% | 1.72 | 12.7 | 32.3 | 25% | 40% |
| 0.02 | 32 | 1.77 | 11.1 | 35.8 | 22% | 30% | 1.28 | 11.6 | 30.1 | 35% | 27% |
| 0.03 | 22 | 1.48 | 10.5 | 33.4 | 29% | 29% | 1.14 | 8.5 | 29.4 | 37% | 17% |
| 0.05 | 13 | 1.28 | 8.2 | 31.4 | 35% | 22% | 1.04 | 7.4 | 28.8 | 43% | 16% |
| 0.1 | 7 | 1.05 | 7.8 | 30.0 | 41% | 16% | 1.02 | 7.1 | 28.4 | 49% | 16% |
| 0.3 | 3 | 0.95 | 7.6 | 29.0 | 56% | 14% | 0.89 | 6.8 | 28.1 | 57% | 14% |

Per pocket, face ends (the crossing as `_barrier` computes it), small blocks:

| g | metres | count med | p90 | max | r<=1 | r>3 | area med | p90 | max | r<=1 | r>3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.003 | 209 | 8.38 | 39.8 | 158.8 | 3% | 70% | 5.07 | 29.6 | 99.4 | 6% | 63% |
| 0.01 | 65 | 3.30 | 14.8 | 120.2 | 11% | 52% | 2.21 | 14.1 | 75.3 | 21% | 43% |
| 0.03 | 24 | 2.01 | 13.0 | 105.5 | 17% | 43% | 1.51 | 12.4 | 66.1 | 27% | 30% |
| 0.1 | 9 | 1.30 | 12.1 | 98.1 | 27% | 27% | 1.26 | 11.5 | 61.5 | 37% | 22% |
| 0.3 | 5 | 1.18 | 11.7 | 95.1 | 32% | 25% | 1.13 | 11.2 | 59.6 | 41% | 19% |

Large blocks (GPU):

| run | g | metres | count med | p90 | max | r<=1 | r>3 | area med | p90 | max | r<=1 | r>3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| per home | 0.005 | 125 | 2.81 | 9.1 | 16.2 | 4% | 38% | 2.09 | 5.9 | 15.6 | 8% | 29% |
| per home | 0.01 | 63 | 1.57 | 4.9 | 8.7 | 17% | 25% | 1.51 | 3.5 | 8.3 | 25% | 12% |
| per home | 0.02 | 32 | 1.11 | 2.8 | 4.9 | 38% | 8% | 1.06 | 2.3 | 4.7 | 38% | 4% |
| per pocket | 0.003 | 207 | 4.50 | 17.0 | 26.3 | 0% | 62% | 2.81 | 16.1 | 25.3 | 8% | 42% |
| per pocket | 0.01 | 63 | 1.97 | 5.8 | 8.7 | 17% | 42% | 1.56 | 5.3 | 8.4 | 25% | 25% |
| per pocket | 0.03 | 22 | 1.32 | 2.8 | 3.6 | 29% | 8% | 1.20 | 2.2 | 3.5 | 29% | 4% |
| per pocket | 0.1 | 7 | 1.02 | 1.7 | 2.0 | 46% | 0% | 1.02 | 1.2 | 1.8 | 42% | 0% |

- The median r is about 2 at g 0.008 (count) and 0.006 (area) per home, at 0.015 and 0.008 per pocket, and at 0.03 and 0.01 with face ends.
- With face ends the tail does not shrink with g (p90 11.7, max 95 at g 0.3): the throttle is the door's end cells, not the door.
  Open ends cut the max at g 0.3 from 95 to 29 and the p90 from 11.7 to 7.6 (count).
- With open ends the tail stops shrinking once g is past about 0.05: from there on it is set by what lies beyond the door (below, "The tail").

r against pocket size (`summary_door_link_<run>_size.csv`; walled-in homes sharing the home's main pocket; count, g 0.01):

| homes in the pocket | homes | per pocket: med | p90 | max | per home: med | p90 | max |
|---|---|---|---|---|---|---|---|
| 1 | 22 | 1.32 | 5.9 | 32.8 | 1.26 | 5.9 | 32.8 |
| 2 -- 3 | 28 | 3.98 | 12.4 | 30.6 | 1.80 | 4.8 | 26.3 |
| 4 or more | 13 | 11.3 | 43.0 | 43.1 | 7.61 | 28.9 | 28.9 |

A door per pocket makes r grow with the pocket's demand: the door's drop is (the pocket's demand) / g.
A door per home brings pockets of 2 -- 3 homes level with single homes; the 4+ class (38486's courtyard and a 4-home pocket on 63718) keeps a high r at any g, because its homes' doors all land at the same street-side cell (below).

### (b) The finer raster (`summary_door_link_<run>_b_open.csv`, `summary_door_link_<run>_b_match.csv`)

Under the fix only three walled-in homes reach open space at h 0.25 (46233's 232 and 274, sharing a pocket behind about 1 m of footprint, and 63718's 127, behind 3.7 m), and two at h 0.35 (46233's).
r_fine at h 0.25 is 3.24, 1.78 and 1.06 (count); 2.65, 1.73 and 1.05 (area).

| run | h | homes | g* count: median (p10 -- p90) | g* area | log(r / r_fine) at g 0.01: count | area |
|---|---|---|---|---|---|---|
| per home | 0.25 | 3 | 0.0034 (0.0018 -- 0.028) | 0.0015 (0.0003 -- 0.014) | -0.67 | -1.00 |
| per home | 0.35 | 2 | 0.0027 (0.0021 -- 0.0033) | 0.0013 (0.0011 -- 0.0015) | -0.84 | -1.08 |
| per pocket | 0.25 | 3 | 0.0069 (0.0036 -- 0.030) | 0.0030 (0.0018 -- 0.015) | -0.27 | -0.73 |
| per pocket | 0.35 | 2 | 0.0054 (0.0041 -- 0.0066) | 0.0025 (0.0021 -- 0.0029) | -0.46 | -0.83 |

Per home at g 0.01, r is 0.94 and 0.91 for 46233's two homes (r_fine 3.24 and 1.78) and 1.15 for 63718's 127 (1.06).
At h 0.25 46233's homes escape through a long sub-cell gap, which costs them more than the door does: matching them needs g of about 0.002 -- 0.003, where the median walled-in home is at r 4 -- 6.
Three homes in two blocks cannot set g against (a): the homes a finer raster can open are by construction those behind a sub-cell gap, not a wall.

### (c) Leakage (`summary_door_link_<run>_c.csv`, `summary_door_link_pocket_heading_open_k0_c_worst_g*.csv`)

Tier-0 homes, u / u(eps 0); u_open is the part due to the tier-0 demand alone (the leak), u adds the walled-in and tier-2 homes' demand.

| run | worst u_open / u(0), g 0.01 | g 0.03 | g 0.3 | u / u(0): median | p95 | max |
|---|---|---|---|---|---|---|
| per home, open ends | 1.000 | 0.999 | 0.994 | 1.000 | 1.08 (count) / 1.04 (area) | 4.92 / 4.39 |
| per pocket, open ends | 1.000 | 0.999 | 0.993 | 1.000 | 1.08 / 1.04 | 4.92 / 4.39 |
| per pocket, face ends | 0.975 | 0.951 | 0.911 | 1.000 | 1.08 / 1.05 | 5.54 / 6.86 |
| per pocket, open ends, large | 1.000 | 1.000 | 0.999 | 1.000 | 1.01 / 1.01 | 2.41 / 1.42 |

- The door is leak-free up to its end cells: a pocket hangs off its street-side cell and gives the open network a little extra room to turn there.
  With face ends that cell is a sliver whose own turning is weak, so the extra room matters (0.91 at g 0.3).
- The added demand does not depend on g: the walled-in homes' demand leaves through b whatever the door costs.
  Its largest effect is on 38486's home 74 (u 30 → 148, count), 6 m from the courtyard door's street end, and its neighbour 124 (204 → 843).

### (d) Large blocks (`summary_door_link_d.csv`)

The 12 large blocks with walled-in homes under the fix, all with stored clearings at D 0.05.
Lens A (area, J_2) of the stored clearings in the door world over every home ("all") and over the tier-0 homes ("open"), against the stored value, and the walled-in homes' share of J_2; per home, open ends, g 0.01:

| block | homes | walled in | J_2 share, none | SIMP: all | open | stored | J_2 share | cheap: all | open | stored | J_2 share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20543 | 1,538 | 1 | 4.05% | 0.4804 | 0.5023 | 0.5026 | 11.97% | 0.4597 | 0.4767 | 0.4763 | 9.99% |
| 5706 | 1,347 | 1 | 1.42% | 0.5686 | 0.5816 | 0.5817 | 7.22% | 0.5572 | 0.5697 | 0.5698 | 6.88% |
| 43708 | 1,296 | 2 | 1.35% | 0.5639 | 0.5691 | 0.5702 | 3.57% | 0.5284 | 0.5337 | 0.5349 | 3.46% |
| 41806 | 1,118 | 4 | 0.88% | 0.5709 | 0.5707 | 0.5701 | 0.79% | 0.5539 | 0.5535 | 0.5507 | 0.72% |
| 22267 | 1,575 | 3 | 0.74% | 0.7050 | 0.7071 | 0.7070 | 2.15% | 0.6935 | 0.6933 | 0.6931 | 0.59% |
| 41673 | 1,074 | 5 | 0.34% | 0.5729 | 0.5732 | 0.5731 | 0.40% | 0.5484 | 0.5485 | 0.5483 | 0.35% |
| 21252 | 1,025 | 2 | 0.33% | 0.3452 | 0.3463 | 0.3467 | 0.67% | 0.3215 | 0.3213 | 0.3216 | 0.27% |
| 39240 | 1,081 | 1 | 0.31% | 0.6220 | 0.6224 | 0.6215 | 0.70% | 0.5706 | 0.5711 | 0.5705 | 0.65% |
| 22336 | 1,342 | 1 | 0.21% | 0.5708 | 0.5728 | 0.5729 | 1.14% | 0.5482 | 0.5501 | 0.5501 | 1.03% |
| 38254 | 1,220 | 1 | 0.10% | 0.5464 | 0.5473 | 0.5474 | 0.46% | 0.5246 | 0.5254 | 0.5255 | 0.42% |
| 41515 | 1,510 | 1 | 0.01% | 0.5589 | 0.5603 | 0.5606 | 0.05% | 0.5514 | 0.5512 | 0.5512 | 0.05% |
| 23597 | 1,130 | 2 | 0.01% | 0.4748 | 0.4748 | 0.4749 | 0.02% | 0.4673 | 0.4674 | 0.4674 | 0.02% |

- Lens A over open homes stays within 0.003 of the stored value (the added demand's congestion); over all homes it is lower by at most 0.022 (20543), 0.013 (5706) and 0.005 (43708), and within 0.002 elsewhere.
- SIMP leads cheap on every block, over all homes and over open homes, as stored.
- A clearing lowers every open home's u but leaves the doors as they are, so the walled-in homes' share of J_2 rises after clearing: 20543's one home, inside a footprint behind 3.5 m of wall, holds 4.05% of J_2 with nothing cleared and 12% after SIMP's clearing.
- Per pocket gives the same table within 0.0002 on every block but 41806 (a 3-home pocket), where Lens A over all homes is 0.010 lower and the walled-in homes hold 1.9% of J_2 (6.3% after SIMP's clearing).
  Per pocket at g 0.03, the largest gap over all homes falls to 0.004 and the largest share of J_2 to 1.1% (2.8% after clearing).
- With face ends, at g 0.01 the gap reaches 0.037 and the share 5.0% (41806, whose doors' street ends are 1/16 to 9/16 open).

### The tail

Per home at g 0.01, 18 of the 63 small-block homes have r above 3 (count): 7 in 38486's courtyard, 5 inside footprints (21622's 178, 42404's 65, 44837's three), 43817's 47 and 69, 38365's 4 and 19263's three.
Potentials at the doors' ends (`summary_door_link_ends_diag.csv`; per pocket, open ends, count):

- **Where the door leads.** 38486's courtyard (9 homes) crosses its thinnest wall into open space that is itself badly connected.
  At g 0.3 the door's own drop is 30, but its street-side cell sits at u 1,672: 2.0x the worst of its homes' neighbours and 98x the block median.
  The courtyard's demand, entering there, is what lifts it: open homes 4 and 6 m from it go from u 204 to 843 and from 30 to 148.
  Its homes stay at r 2.1 -- 29 at g 0.3 (count), and the 4+ class keeps a median r near 7.
  43817's largest pocket does the same (u 846 at its door's street end, 35x the block median), while that block's other two pockets' doors land at u of about 30.
- **Neighbours at the street.** 21622's home 178 (tier 4) has neighbours with u of about 3: its door's street end is at u 0.9, and the door's own drop (1/g = 100 at g 0.01) is the whole of its r of 33; at g 0.3 its r is 1.8.
- **One door per pocket.** 19263's and 44837's 3-home pockets pay three times the drop (300 at g 0.01, per pocket); one door per home removes it.

Choosing b by the thinnest wall alone ignores how well the open space behind it connects.
A rule that weighs the street side (several doors, or the crossing that minimises the escape time rather than the wall metres) is the next thing to try for the 4+ pockets; it was not run.
38486's courtyard, at r 37 with the leaf door fully open, is at 29 at most with the door link (per pocket at g 0.3, per home at g 0.01).

### Variants

- **Isotropic door node** (`door_link_pocket_iso_open_k0`; 6 blocks, 32 walled-in homes, g 0.003 -- 0.1): |log(r_iso / r)| at most 0.003; open homes' u within 0.4%; the corridor calibration within 1%.
  The coupling does not matter, so the heading-preserving edges stay: they keep the GPU solver.
- **Thickness term** (`door_link_pocket_heading_open_k1`, kappa 1 per metre): the median r is near 1.7 at g 0.1 with p90 11.2 (kappa 0: g 0.02, p90 11.1), and near 3.2 at g 0.03 with p90 23.8 (kappa 0: about 16).
  The three finer-raster homes keep their spread: the thickest-walled of them (63718's 127, 3.7 m at h 0.5) is the least affected at h 0.25, so a cost per metre moves it the wrong way.
  Not kept.
- **Face ends** are the crossing as `_barrier` computes it; the rows are kept (`door_link_pocket_heading_face_k0`) for (a) -- (d) above.

### Caveats

- (b) rests on three homes in two blocks; two share a pocket. The homes a finer raster can open are those behind sub-cell gaps, so (b) says nothing about doors through real walls.
- The small blocks are those with stranded homes; the large-block numbers (24 homes, pockets of at most three) are what the full run will see.
- Tiers 4 -- 5 (7 small-block homes, 5 large) are nested or overlapped footprints whose interior becomes their pocket; four of them are in the small blocks' tail.
- One grid offset (lifted.OFFSET), as above.
- g is calibrated on one synthetic geometry; the corridor equivalents are a reading aid, not part of the model.

### Files

- `insulator.py`: the commands `link-census`, `link <procs> <plan> <coupling> <ends> <kappa> <g,> <ids,>`, `link-large`, `corridor`, `link-report` (the module docstring).
- Rows (parquet, ignored): `door_link_census_h0.5/`; `door_link_{pocket,home}_heading_open_k0_h0.5_ell3_K8/`, `door_link_pocket_heading_face_k0_h0.5_ell3_K8/`, `door_link_pocket_iso_open_k0_h0.5_ell3_K8/`, `door_link_pocket_heading_open_k1_h0.5_ell3_K8/` (small blocks: `<block>__g<g>.parquet` and `<block>__sites.parquet`; large: `<block>__large.parquet`, `<block>__large_homes.parquet`).
- Kept: `summary_door_link_census_sets.csv`, `summary_door_link_census.csv`, `door_link_census_homes.csv`, `summary_door_link_corridor.csv`, `summary_door_link_b_fix.csv`, `summary_door_link_vs_default.csv`, `summary_door_link_d.csv`, `summary_door_link_ends_diag.csv`; per run `summary_door_link_<run>_{a,b_open,b_match,c,size}.csv`, `summary_door_link_<run>_large_{a,c}.csv` and `door_link_<run>_stranded_homes.csv`.

## The private crossing (stopped)

The owner (2026-10-11), of the door link: "Is there a way to simplify your suggested method? The if-else feels unnatural"; then "Yes, test that. But, it would be nice to also test a version that can account for traffic/contention, if that would fit into the same framework, since it would reward multiple points of egress".
Code: `crossing.py` (the commands `small`, `large`, `report`); rows `insulator_rows/crossing_*` (parquet, ignored); summaries `summary_crossing_small.csv`, `summary_crossing_size.csv`.
Stopped on the small blocks when the owner dropped the criterion: "I am fine with having fully enclosed buildings be much worse off than exposed neighbors".

### The models

Both leave the open network O as today, with no edge in the operator, and use the door's crossings (`HomeDoors` with open ends).
- **The toll.** A walled-in home's demand enters O at its crossing's street end b and pays a private resistance after the solve: u = u_O(b) + d / g per crossing (the door's physics with no edge), or u_O(b) + d T / g1 per metre of footprint crossed (d the demand crossing, T its metres).
- **Private walls.** The walled-in homes' walkers cross a private copy of the block whose footprint conducts eps relative to open ground (isotropic, its fully open street-connected cells held at 0), then go on in O from where they come out; their u is the copy's potential plus O's, averaged over their exits.
  O's own traffic cannot use the copy, so nothing leaks; homes share its walls (contention), and several thin walls conduct in parallel (several ways out).
  The copy lives on a window 25 m around the walled-in homes: at 50 m, u moves by at most 1.2e-5 (log ratio; 12 blocks, 43 walled-in homes).

### Results at matched medians (31 small blocks, 63 walled-in homes)

Each model is calibrated to the door's median r at g 0.01 (1.80, count): the toll per crossing at g 0.0104, per metre at g1 0.040, the private walls at eps 3.0e-3 (grid 3e-3).

| model | count: median | p90 | max | r>3 | area: median | p90 | max | r>3 |
|---|---|---|---|---|---|---|---|---|
| door, g 0.01 | 1.80 | 10.5 | 32.8 | 29% | 1.51 | 11.0 | 29.0 | 19% |
| toll per crossing, g 0.0104 | 1.80 | 10.2 | 31.1 | 25% | 1.32 | 10.8 | 30.1 | 21% |
| toll per metre, g1 0.040 | 1.80 | 10.8 | 28.7 | 27% | 1.27 | 10.9 | 30.5 | 21% |
| private walls, eps 3e-3 | 1.78 | 5.0 | 40.1 | 25% | 1.24 | 3.0 | 32.8 | 11% |

By pocket size (count; walled-in homes sharing the home's main pocket):

| homes in the pocket | homes | door: median | p90 | max | toll per crossing: median | p90 | private walls: median | p90 | max |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 22 | 1.26 | 5.9 | 32.8 | 1.21 | 5.5 | 1.04 | 4.8 | 40.1 |
| 2 -- 3 | 28 | 1.80 | 4.8 | 26.3 | 1.81 | 4.9 | 2.21 | 5.6 | 7.8 |
| 4 or more | 13 | 7.61 | 28.9 | 28.9 | 7.58 | 28.7 | 1.11 | 3.1 | 3.1 |

- The toll per crossing is the door's physics without its edge, and gives the same distribution: per home within 12% of the door's r for 57 of the 63 homes (count; median 3%), 46 by area (the largest gaps, up to 1.7x, on homes in several pockets and inside footprints).
  The door's matrix edge buys nothing a toll added after the solve does not.
- The toll per metre lowers the homes behind thin walls and raises those behind thick ones, and leaves the tail where it was.
- The private walls halve the p90 by letting a pocket out on every side.
  38486's 9-home courtyard leaves into open space at u of about 60 rather than through its thinnest wall into space at about 1,670; a median 19% of a walled-in home's walkers land within 5 m of that wall's street end (p10 5%, p90 32%), spread over 12.5 m (RMS, median).
  Against that, the max rises (21622's home 178, inside a footprint with neighbours at the street: 40.1 against 32.8), five of the seven homes inside a footprint get worse (42404's 65: 13.8 against 6.2), and pockets of 2 -- 3 homes go from a median of 1.8 to 2.2.
- Several doors per pocket, in the shared operator, would not give the same: they would make a courtyard a corridor between the spaces on its sides (38486's: u of about 1,670 and 60), the leak a single door avoids by ending in a dead end.
  A model with several ways out needs the private rule stated outright: invented passages serve only the homes with no visible way out.

## Uniform walls: one number against two

The owner (2026-10-11): "Actually, I think I am fine with having fully enclosed buildings be much worse off than exposed neighbors, so we can try to find a uniform resistance that buildings can have, such that we strike some sort of nice, ideally canonical/motivated balance"; then, on a single eps of 1e-4 against ordering by two numbers, "I'm not sure between a single 1e-4 and order via two numbers, so maybe we can test both".
Code: `walls.py` (the commands `small`, `small-exact`, `clearings`, `large`, `report`); rows `insulator_rows/walls_{small,large}_h0.5_ell3_K8/` (parquet, ignored); summaries `summary_walls_*.csv` and `walls_enclosed_homes.csv`.

### Answer

- No uniform resistance strikes a canonical balance.
  An enclosed home's escape time is a toll set by eps and its own walls, so its weight in J_2 grows as 1/eps^2: eps sets an exchange rate between the enclosed homes and everyone else, and the walls and each block's escape times move it.
  At eps 1e-4 one enclosed home holds 80% of J_2 on 20543 and 78% on 5706, and Lens A of SIMP's stored D 0.05 clearing falls there from 0.50 and 0.58 to 0.08 and 0.10.
- A (one number, eps 1e-4) also leaks at the gates: the gated sentinels lose 0.018 -- 0.040 of Lens A, and 3.9% of the small blocks' open homes lose more than 5% of their escape time.
- B (two numbers) is the eps -> 0 end of the same family: there any enclosed home left outweighs every other change, and among clearings that leave the same ones, the rest's J_2 decides.
  B counts the enclosed homes by population where the limit weighs them by their walls' tolls, and drops their own flow through the open network (at eps 4.2e-6 that flow and the leak together move Lens A over the open homes by at most 0.0024).
  B's second number is today's metric with the placement fix: the stored Lens A to 4e-8 on the 7 large blocks with no shadowed home moved, and to 1.6e-3 on the other 9.
- B's optimizer is two stages, not a search at the gate-keeping eps: free the enclosed homes (clearing every enclosed home's cheapest crossings costs D 0.0004 -- 0.0144 per block, median 0.0022, on the 12 large blocks that have any), then today's search with the rest of the budget.
  At eps_gate (4.2e-6) the GPU solver did not converge with nothing cleared on 5 of the 16 large blocks.
- Recommendation: B (below).

### The model

Both candidates place demand with the placement fix (`Placement.of(open_side=True)`; "The door link", Task 1), and call a home enclosed when no free cell within 1 m of its outline reaches the street (`insulator.walled`: tiers 1, 3, 4 and 5).
- **A, one number.** Every footprint sub-sample conducts eps relative to open ground (`insulator_open`, the eps world), every home injects, and Lens A is over every home; eps 1e-4 is the strength SIMP's default searches at.
- **B, two numbers.** Walls are walls (eps 0). A clearing is scored first by (i) the population still enclosed after it (no injection cell open and street-connected in its exact field), then by (ii) Lens A over the homes not enclosed at baseline, which inject alone.
- **eps_gate**, the world B's optimizer was to search in: the largest eps at which the walls carry at most 5% of every open home's escape, u_open(eps) / u_open(0) >= 0.95 for every tier-0 home, u_open the potential due to the tier-0 demand alone.
- On the large blocks, five clearings: none; the stored D 0.05 clearings of SIMP's default and of the cheap preset; `free`, every building with a footprint sub-sample on an enclosed home's cheapest crossing (one per home and pocket, `HomeDoors` with open ends); and `free+simp`, their union.
  D is the cleared buildings' share of the area population; cleared homes inject at their old cells, as in today's metric.

### Small blocks: the gate curve (`summary_walls_gate.csv`, `summary_walls_eps_gate.csv`, `summary_walls_worst_fit.csv`)

The 41 small blocks (the 31 with stranded homes and ten controls; 7,821 tier-0 homes), on the CPU at rtol 1e-8; tier-0 u matches the eps world's rows to 1.6e-9.
u_open(eps) / u_open(0) over the tier-0 homes (area; count within 0.01); gated: the 79 homes in 11 blocks whose u(0) is at least 8x their block's median.

| eps | min | p0.1 | p1 | p5 | median | below 0.95 | gated below 0.95 |
|---|---|---|---|---|---|---|---|
| 1e-6 | 0.989 | 0.995 | 0.999 | 1.000 | 1.000 | 0 | 0 |
| 3e-6 | 0.969 | 0.986 | 0.996 | 0.999 | 1.000 | 0 | 0 |
| 1e-5 | 0.903 | 0.955 | 0.987 | 0.996 | 1.000 | 6 | 5 |
| 3e-5 | 0.758 | 0.877 | 0.962 | 0.988 | 0.999 | 54 | 24 |
| 1e-4 | 0.486 | 0.687 | 0.885 | 0.962 | 0.997 | 302 (3.9%) | 60 |
| 3e-4 | 0.243 | 0.475 | 0.734 | 0.901 | 0.992 | 793 (10%) | 79 |
| 1e-3 | 0.093 | 0.250 | 0.499 | 0.764 | 0.973 | 2,299 (29%) | 79 |

- The min crosses 0.95 at 4.25e-6 (area) and 4.22e-6 (count): eps_gate 4.2e-6.
  p0.1 crosses at 1.07e-5, p1 at 3.6e-5 -- 3.7e-5.
- The worst home at every eps is 43365's home 162 (u(0) 22x its block's median); 1 / (1 + eps / eps0) fits its curve with eps0 9.9e-5 (rms log error 0.019), and puts it at 0.95 near 5.2e-6, about 20% above the grid's crossing.
- On the large blocks 30848's home 676 (u(0) 139x the median) has 0.413 at 1e-4, which puts it at 0.95 near 3.7e-6 by the same law; its solve at 4.2e-6 did not converge.
  Over every block eps_gate is therefore at most about 3.7e-6 (about 3.0e-6 if the law overstates this home's crossing as it did 43365's).

### Small blocks: the enclosed homes (`summary_walls_enclosed.csv`, `summary_walls_enclosed_share.csv`, `walls_enclosed_homes.csv`)

63 enclosed homes on 24 of the small blocks: r = u_i over the largest u among its 6 nearest tier-0 homes, z = u_i over its block's median tier-0 u (area; count in brackets for r's median), and the enclosed homes' share of their block's J_2 (area, over the 24 blocks).

| eps | r: median | p90 | max | z: median | share of J_2: median | max |
|---|---|---|---|---|---|---|
| 1e-6 | 1,197 (1,649) | 3,383 | 47,941 | 2,944 | 0.9998 | 1.000 |
| 3e-6 | 399 (550) | 1,128 | 15,983 | 986 | 0.998 | 1.000 |
| 1e-5 | 120 (165) | 339 | 4,798 | 301 | 0.981 | 1.000 |
| 3e-5 | 40 (55) | 113 | 1,602 | 103 | 0.861 | 0.9996 |
| 1e-4 | 12.3 (16.8) | 34 | 484 | 33 | 0.414 | 0.996 |
| 3e-4 | 4.4 (5.9) | 11.9 | 164 | 13.4 | 0.103 | 0.968 |
| 1e-3 | 1.7 (2.2) | 4.6 | 52 | 5.5 | 0.031 | 0.781 |

- r and z go as 1/eps over the whole range: an enclosed home's escape is a toll its walls set.
- Between keeping the gates (4.2e-6) and a moderate r (1e-3, the range of the owner's first criterion) the enclosed homes' share of J_2 runs from 99.8% to 3%; at 1e-5 they already hold 98%.

### Large blocks (`summary_walls_large_*.csv`)

The 12 large blocks of the full run with enclosed homes under the placement fix (24 homes: 19 in tier 1, and 5 in tier 4, with no free cell within 1 m, which inject on their own footprint's cells; 0.01 -- 0.23% of a block's population), and the four sentinels (22422 and 30848 gated, 5810 and 9712 ordinary), on the cluster's GPUs at rtol 1e-6 and eps 0, 4.2e-6 and 1e-4 (one task per block; 42 minutes on four GPUs).
At 1e-4 the enclosed homes have r median 8.1 (count 16.6), p90 35, max 125, and z median 34 (count 45), p90 70, max 210, as on the small blocks.
Their share of J_2 with nothing cleared is 9.5% at the median but 0.7 -- 80% by block, because each toll is set against its block's escape times (median tier-0 u 5.9 -- 301).
Lens A (area, J_2) under A over every home at 1e-4, and under B (ii):

| block | enclosed | share of J_2: none | after SIMP | A: SIMP | A: cheap | A: free | A: free+simp | B (ii): SIMP | B (ii): cheap | D of free |
|---|---|---|---|---|---|---|---|---|---|---|
| 20543 | 1 (tier 4) | 0.80 | 0.93 | 0.077 | 0.074 | 0.552 | 0.758 | 0.503 | 0.476 | 0.0025 |
| 5706 | 1 | 0.78 | 0.95 | 0.097 | 0.095 | 0.533 | 0.802 | 0.582 | 0.570 | 0.0007 |
| 41806 | 4 (one tier 4) | 0.43 | 0.69 | 0.404 | 0.387 | 0.242 | 0.669 | 0.570 | 0.551 | 0.0043 |
| 43708 | 2 (one tier 4) | 0.39 | 0.75 | 0.328 | 0.310 | 0.231 | 0.662 | 0.569 | 0.535 | 0.0046 |
| 22336 | 1 (tier 4) | 0.32 | 0.72 | 0.329 | 0.319 | 0.177 | 0.644 | 0.573 | 0.550 | 0.0019 |
| 22267 | 3 | 0.15 | 0.58 | 0.530 | 0.596 | 0.096 | 0.699 | 0.707 | 0.693 | 0.0080 |
| 38254 | 1 | 0.04 | 0.16 | 0.513 | 0.493 | 0.020 | 0.555 | 0.547 | 0.526 | 0.0011 |
| 39240 | 1 | 0.03 | 0.13 | 0.594 | 0.544 | 0.014 | 0.622 | 0.623 | 0.571 | 0.0004 |
| 41673 | 5 | 0.02 | 0.07 | 0.542 | 0.517 | 0.038 | 0.559 | 0.573 | 0.548 | 0.0100 |
| 21252 | 2 | 0.01 | 0.03 | 0.340 | 0.324 | 0.017 | 0.355 | 0.347 | 0.322 | 0.0144 |
| 41515 | 1 (tier 4) | 0.01 | 0.05 | 0.548 | 0.541 | 0.005 | 0.560 | 0.559 | 0.552 | 0.0011 |
| 23597 | 2 | 0.007 | 0.02 | 0.459 | 0.449 | 0.004 | 0.465 | 0.475 | 0.467 | 0.0005 |

- On 20543 and 5706 one home holds four-fifths of J_2, and clearing the 1 -- 3 buildings that free it (D 0.0025 and 0.0007; 20543's crossing runs through the home's own footprint) is worth 7.2x and 5.5x SIMP's whole D 0.05 clearing under A.
- The order of SIMP's and the cheap preset's clearings: SIMP first on 15 of the 16 blocks today and under B (ii), 14 under A (22267 flips: the cheap preset frees one of its three enclosed homes), 13 under B ((i) also gives 21252 to the cheap preset, which frees both of its enclosed homes); 30848 is the cheap preset's in every world.
  SIMP's and the cheap preset's clearings each free one home on 41673 and 41806, and SIMP's none elsewhere.
- A's open homes: Lens A over the non-enclosed at 1e-4 trails B (ii) by up to 0.043 (20543) and 0.039 (22267); at 4.2e-6 by at most 0.0024.
- The sentinels' leak at 1e-4 (Lens A over every home, against eps 0): 22422 -0.018 (SIMP's 0.783 to 0.764) and -0.020 (the cheap preset's), 30848 -0.040 and -0.040 (46 of its 5,395 homes lose more than 5% of their escape time), 5810 -0.0005, 9712 -0.0001.
  At 4.2e-6, 22422 loses 0.0008 and the ordinary two less than 1e-4; the two clearings keep their order on all four.
- At 4.2e-6, wherever the solve converged, the enclosed homes hold 76 -- 99.95% of J_2 with nothing cleared and 92 -- 99.99% after SIMP's clearing: Lens A there says whether a clearing frees them (SIMP's 0.0003 -- 0.23, `free`'s 0.51 -- 0.98), B (i) as one number.

### Solves (`summary_walls_solver.csv`, `summary_walls_failed.csv`)

- Small blocks (CPU, rtol 1e-8): pyamg's CG stopped at its 500-iteration cap on 7 block-eps (63718 and 43874 at 1e-6, 3e-6 and 1e-5; 19570 at 1e-6), and accepted 4 fields above rtol inside its 10x slack; all 11 were re-solved with SuperLU and one step of iterative refinement (`small-exact`; residuals at most 2.4e-9).
- Large blocks (GPU, rtol 1e-6): every system at eps 0 (72) and at 1e-4 (72) converged, medians 4.5 s and 14 s.
  At 4.2e-6, 66 converged (median 44 s, max 457 s), and 6 stopped at the 2,000-iteration cap: nothing cleared on 22267, 30848, 38254, 39240 and 41673 (residuals 2.0e-6 -- 4.3e-5), and SIMP's clearing on 30848 (5.4e-6).
  The swings' restore screens met the same on 30848 at 1e-6 (NOTES, "Swings as SIMP's finisher").
  Untested: the K-cycle's aggregates come from the sparsity pattern alone (strength theta 0), so they straddle walls whose conductance is eps.

### Recommendation

B: walls are walls, and a clearing reports the population it leaves enclosed and, after it, today's Lens A over everyone else.
- It is what the uniform resistance says at its leak-free end, so it answers "canonical" without a constant.
- Its second number keeps every stored result (to 1.6e-3 where the placement fix moved a shadowed home).
- Its optimizer frees first and then searches as today; no solve at the gate-keeping eps is needed.

A stays the one-number option, with eps 1e-4 as its constant.
Its price is a leak of 0.02 -- 0.04 at the gates, and an exchange rate set by the walls and the block, which gives one home four-fifths of a block's score.

### Caveats

- `free` is one clearing, not an optimum: each enclosed home's crossing cheapest in metres of footprint, not in D.
  On 11 of the 24 homes it runs through the home's own building (all five tier-4 homes), which (i) then counts as freed while its residents inject at their old cells, as today's metric counts every cleared home.
- `free+simp` overspends the budget (D 0.050 -- 0.064); B's optimizer would search with D 0.05 less `free`'s.
- eps_gate is 4.2e-6 on the small blocks and about 3.7e-6 with 30848's gated home, from one point and the law that put 43365's home 20% above its grid crossing.
- 16 large blocks: the 12 of the full run's 60 with enclosed homes, and four sentinels; the road methods' networks are not scored yet.

### Files

- `walls.py`: `small <procs> <eps,> <ids,>`, `small-exact <procs> <ids,>`, `clearings <out> <ids,>`, `large <clearings> <eps,> <ids,>` (on the cluster: `cluster.py submit blocks <run> --with <clearings> <ids> -- research/roadless/walls.py large <its name> <eps,> {id}`), `report`.
- Rows (parquet, ignored): `walls_small_h0.5_ell3_K8/<block>__eps<eps>.parquet`; `walls_large_h0.5_ell3_K8/<block>__eps<eps>__{homes,scenarios}.parquet` and `<block>__free.parquet`; failures under `failed/` beside them; `walls_clearings.parquet`, the large run's input.
- Kept: `summary_walls_{check_old,gate,eps_gate,worst_fit,worst_curve,worst_homes,enclosed,enclosed_share,solver,failed}.csv`, `walls_enclosed_homes.csv`, `summary_walls_large_{blocks,a,a_gate,b,order,free,leak,gate,enclosed}.csv`.
