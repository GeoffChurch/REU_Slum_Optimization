# Roadless scoring: heading-aware conduction on the space a network frees (living note)

Branch `research/roadless`, code in `research/roadless/`. Started 2026-09-29.
cupy and pyamg are in the project env (pyproject's `gpu` dependency group, installed by default on
this branch). Run scripts from the repo root with `PYTHONPATH=.`.

## The idea (owner, 2026-09-29)

Forget roads. A network matters only through the buildings its corridor destroys; score how well
people can move through the resulting open space along fairly straight / smooth paths, with no
road/walk distinction. Use it (1) to score any method, (2) to generate clearing methods.

Why it is worth doing, from the record:
- The road/walk conductance ratio DECIDES the ranking and has no physical support
  (notes 2026-08-06, 2026-08-08). Roadless has no such ratio.
- resistance_lp's 6 m mesh threads existing gaps: a road through open ground frees nothing, so
  it should earn nothing here (prediction to test).
- Continuum conduction on the free space was blocked on DISCS (eps reordered methods, 2026-08-05);
  real footprints leave one connected free space (B1, 2026-08-06), and
  `specs/2026-08-06-continuum-on-footprints-design.md` was spec'd on that and never built.

## The model (lifted.py)

State (cell, axis), axes in [0, pi). Along-axis edges join a cell to the next along the axis's
lattice vector (K = 8: king+knight; K = 16 adds (3,1), (3,2)), every crossed cell must be free,
weight m_k / |v|^2; turning edges join adjacent axes in a cell, weight h^2 / (ell^2 dtheta).
Continuum: E(u) = int (d_s u)^2 + ell^-2 (d_theta u)^2. Symmetric, so P = f^T L^-1 f is both the
total expected escape time and the power of the system-optimal LINEAR-congestion routing, and
freeing space can only lower P (Rayleigh). ell = metres travelled per radian of turning.

Demand: each building injects 1 over the free cells within 1 m of it that it is nearest to,
spread over axes by angular share; FIXED at the original footprints (rehoused in place), so only
the operator changes. Ground: cells within 1 m of Block.streets. Homes in pockets sealed at
baseline (no path to ground) are dropped from the demand and counted (`stranded`).

Rules for what a network frees: CARVE (the corridor itself -- what the displacement charge pays
for) and OBLITERATE (every building the corridor touches, whole). Score perm' = 1 - P/P0.

Contention caveat (told to the owner): linear conduction makes co-directional streams contend
(power ~ flow^2), but PERPENDICULAR streams do not (cross term J1.J2 = 0) -- true of today's
permeability too, not new here. The lifted model also weakens oblique (45 deg) interaction.
`lifted.crossing` measures the power share in cells where streams cancel.

## Results so far

**Open channel, angle invariance** (checks.py, angle_h.py; K = 16). Grid offset by
(0.37h, 0.19h) is REQUIRED: with cell centres exactly on axis-aligned channel walls the rasterized
width lost a row and 0/90 deg read 30% worse. After that, P vs angle max/min:

    W 2 m, ell 3:  h 0.5 -> 1.246, h 0.25 -> 1.075
    W 2 m, ell 10: h 0.5 -> 1.380, h 0.25 -> 1.215
    W 4 m, ell 3:  h 0.5 -> 1.074, h 0.25 -> 1.058
    W 4 m, ell 10: h 0.5 -> 1.127, h 0.25 -> 1.107

Discretization, converging in h; worse at large ell (K-limited: off-lattice headings need
turning). K = 8 at ell 10 is bad (1.9). P(W 2)/P(W 4) ~ 2.8, not 2: narrow channels are
penalized beyond width (the injection has to turn into the channel's heading within the width).
Single straight stream crossing diagnostic: power-weighted cancellation 0.11-0.16, share of power
with cancellation > 0.29 at most 1%.

**Cost.** SuperLU: 56-262 s per solve at h 0.5 (fill-in). AMG (pyamg SA + CG, rtol 1e-9): ~10 s
per solve at h 0.5 on 76-316-building blocks.

**h convergence on real blocks (K 16, ell 3) -- NOT converged at h = 1 -> 0.5:**

    18537 (76 bldg):  P0 4837 -> 2250; perm' clearance_looped 0.242 -> 0.318, cycle_native
                      0.708 -> 0.726, resistance_lp 0.675 -> 0.674, greedy_arterial 0.579 -> 0.495
    41132 (138):      P0 2170 -> 1642; 0.218 -> 0.212, 0.261 -> 0.259, 0.394 -> 0.331, 0.269 -> 0.247
    stranded demand 3.6-7.8% and moving with h.

K 8 vs K 16 at ell 3 (h 1 and 0.5): perm' within 0.01 on both blocks -- K 8 is enough at
ell 3. 52727 (316 bldg, K 8): h 1 -> 0.5 perm' clearance_looped 0.441 -> 0.359, cycle_native
0.540 -> 0.491, resistance_lp 0.516 -> 0.455, greedy_arterial 0.558 -> 0.457.

**Crossing streams carry negligible power (gap closed).** Power-weighted share with cancellation
> 0.29 on real blocks, h 0.5: 1.1-1.6% (no roads), 1.0-1.1% (cycle_native at 10%), against up to
1.2% for ONE straight stream in an open channel. Mean cancellation 0.11-0.18 vs 0.11-0.16 for the
single stream. Crossings are where catchments meet; egress flow is overwhelmingly one stream.

**Demand fix (2026-09-29).** First version spread each building's demand over its WHOLE 1 m
ring and dropped the part in sealed nooks, reading 4-8% "stranded". Now demand goes only on ring
cells with a path to the street (per building, renormalized); a building is stranded only if no
ring cell reaches the street. Stranded is then 0-2% at h 1-0.5 and 0.000 at h <= 0.35.

**h convergence, K 8, ell 3, fixed demand** (conv/b*_h*.log). perm' at the Lens A prefix:

    block (bldg)   h     P0      cl_looped  cycle_nat  resist_lp  greedy_art
    18537 (76)    1.0   4004     0.257      0.704      0.666      0.564
                  0.5   2723     0.298      0.756      0.711      0.562
                  0.35  2983     0.367      0.769      0.732      0.549
                  0.25  2601     0.392      0.756      0.717      0.520
    41132 (138)   1.0   2210     0.213      0.256      0.402      0.266
                  0.5   1916     0.187      0.316      0.310      0.232
                  0.35  1793     0.176      0.287      0.288      0.220
                  0.25  1755     0.187      0.251      0.304      0.222
    52727 (316)   1.0  25847     0.453      0.545      0.536      0.572
                  0.5  17851     0.358      0.487      0.455      0.458
                  0.35 17704     0.327      0.449      0.423      0.467
                  0.25 17247     0.341      0.467      0.428      0.459

h = 1 is unusable. From 0.5 down, values move +-0.03 and not monotonically (P0 on 18537
oscillates), and the top two swap on 41132 and 52727. clearance_looped is last everywhere.
Suspect rasterization flicker of gaps ~h (open/closed by cell-centre placement): testing grid-
offset spread at h 0.5 / 0.35 (conv/off_*.log).

**Grid-offset spread (binary cells) is as big as the h effect** (conv/off_*.log): at h 0.5, four
offsets give e.g. 52727 clearance_looped 0.358-0.403, 41132 cycle_native 0.278-0.316. It is
rasterization FLICKER -- gaps ~h open or close by cell-centre placement. Offset-AVERAGED, the
within-block ranking is identical at h 0.5 and 0.35 on all 3 blocks (18537 cn > rlp > ga > cl;
41132 rlp > cn > ga > cl; 52727 cn > ga > rlp > cl), with a residual level shift ~0.03 on 52727.

**Monotonicity: holds.** 18537 h 1, carve: 48 nested random-order steps, largest relative
increase 0.00e+00.

**Fix for the flicker: sub-cell open fractions** (4 x 4 sub-samples per cell). A cell exists if
any sub-sample is open; an along edge's weight is scaled by the smallest open fraction among the
cells it touches (a gap narrower than a cell conducts in proportion to its width), a turning
edge by its cell's open fraction. Carving/obliterating adds sub-samples, so it stays monotone.
Open channel angle max/min, K 8, ell 3: W 2: h 0.5 1.158, h 0.25 1.073; W 4: 1.059, 1.028
(binary cells K 16: 1.246 / 1.075 / 1.074 / 1.058).

**With sub-cell fractions the score converges** (conv2/, K 8, ell 3; perm' at the Lens A prefix):

    block    h (offsets)     cl_looped    cycle_nat    resist_lp    greedy_art
    18537    0.5 (4)         0.322-0.330  0.672-0.676  0.648-0.654  0.503-0.505
             0.35 (2)        0.321-0.325  0.670-0.671  0.645-0.655  0.501-0.510
             0.25            0.327        0.675        0.653        0.502
    41132    0.5 (4)         0.147-0.174  0.259-0.297  0.237-0.276  0.171-0.200
             0.35 (2)        0.154        0.252-0.256  0.246-0.251  0.177-0.180
             0.25            0.149        0.237        0.242        0.176
    52727    0.5 (4)         0.282-0.297  0.405-0.417  0.359-0.372  0.396-0.411
             0.35 (2)        0.284-0.292  0.406-0.415  0.359-0.371  0.403-0.412
             0.25            0.289        0.414        0.370        0.411

Offset spread <= 0.01 on 2 of 3 blocks; 41132 at h 0.5 has one outlier offset (+0.03, the
default one) -- a gap near 0.5 m, presumably. Ranking stable except the 41132 cycle_native /
resistance_lp near-tie (0.005 apart at h 0.25). Stranded 0.000 everywhere. Monotone under both
rules (obliterate: 48 steps, largest relative increase 1.9e-15).

Working resolution: h 0.5, K 8, ell 3, one offset; offset noise to be quantified on a subsample.

## Study: the lineup on the 220 (study.py, rows_h0.5_ell3_K8/)

h 0.5, ell 3, K 8, one grid offset, CARVE unless stated. Same canonical street-first prefixes for
both metrics. (First launch ran 40 workers each with a full BLAS pool: load ~1,240, 6 h for 28
blocks; single-threaded the other 192 took 38 min.) Stranded median 0.000, max 0.068.

Lens A (10% displaced), median perm old | roadless-carve | roadless-obliterate:

    arterial 0.891 | 0.360 | 0.697     cycle 0.855 | 0.360 | 0.593
    cycle_desire 0.864 | 0.362 | 0.619 resist_lp 0.891 | 0.368 | 0.673
    grid 0.767 | 0.324 | 0.520         clear_loop 0.761 | 0.293 | 0.495   clear 0.763 | 0.266 | 0.454

Paired vs plain cycle_native at Lens A, median [95% bootstrap CI], old | roadless:

    arterial      +0.030 [+0.025,+0.038] | -0.002 [-0.014,+0.013]   (n 206)
    resist_lp     +0.028 [+0.022,+0.034] | -0.000 [-0.014,+0.015]
    cycle_desire  +0.010 [+0.004,+0.016] | +0.004 [-0.003,+0.017]
    grid          -0.094 [-0.115,-0.068] | -0.034 [-0.048,-0.023]
    clear_loop    -0.089 [-0.100,-0.076] | -0.053 [-0.068,-0.046]
    clear         -0.084 [-0.094,-0.073] | -0.081 [-0.096,-0.071]

**Roadless, the four leaders TIE** (arterial, resistance_lp, desire cycle, plain cycle: all
within +-0.004, CIs straddle 0), where the old metric separated them by 0.02-0.03. grid and
clearance_looped close most of their gap. Per-block Kendall tau old vs roadless: median +0.43
(IQR +0.24..+0.62), negative on 11% of blocks. Best-at-Lens-A counts spread out: desire cycle
53, resist_lp 44, arterial 43, cycle 40, grid 32. Frontier (Lens A perm, Lens B displacement)
roadless at P*' 0.25: resist_lp 88, arterial 87, cycle 72, desire cycle 71, grid 41,
clear_loop 18, clear 10 (old: 133, 130, 50, 65, 5, 1, 2).

resistance_lp's mesh does NOT collapse (prediction wrong): at 10% displacement its corridors
widen real alleys; it just loses its old-metric lead. The OLD metric's separation of the leaders
is the road/walk conductance ratio at work: roads are ~200x open ground there.

Obliterate (whole buildings touched) vs carve ranking: tau +0.52; obliterate rewards clipping
many buildings slightly (arterial 0.697). Carve is the rule consistent with the fractional
displacement charge.

Owner 2026-09-29: roadless eval has no road-length axis -- the frontier is permeability vs
displacement only.

## Road-free clearing (clear.py)

Owner's idea: greedy LOO over BUILDINGS, no roads. Exact LOO = one solve per building per step.
TENSION screen: one solve with remaining buildings as eps = 0.01 conductor; first-order gain of
clearing j = sum over its covered edges of (Delta u)^2 x weight gained (dP/dw_e = -(Delta u_e)^2).
Screened greedy: top M by tension, exact solve each, clear the best.

**Tension ranks like exact LOO** (loo_b*.log, h 0.5, single-threaded solves 1.5-1.8 s):

    41132 (138): Spearman(tension, exact gain) +0.930; true best = tension's #1
    18537 (76):  +0.947; tension #1 gets 0.997 of the best gain, top 4 contains it

Top single-building gains on 18537 are 36-39% of P0 each. NOT a sealed-pocket artifact: per-home
escape time max/mean 4.0-5.8 (old metric's no-road tau: 2.4-3.8), top 1% of homes hold 3-8% of
P0 (tail.py). It is a real bottleneck: one building plugging the route many homes take.

**LOO "cancellation" (owner asked).** First order: complete -- P is self-adjoint, so u is its own
adjoint and every conductance derivative (dP/dw_e = -(Delta u_e)^2) comes from ONE solve; that
is the tension. Exact: Woodbury makes P - P_j depend only on u near j and the Green's function
on j's perimeter (~perimeter cells x K, hundreds of solves) -- not cheap. Cheap and CERTIFIED:
a local correction solve in a window around j bounds the gain from above (Dirichlet form: gain_j
<= D_j - r^T L_OO^-1 r, D_j = first-order term) and local flow rerouting bounds it from below
(Thomson); both converge to exact as the window grows. Not built: with 1.7 s solves, M = 4
screening already costs ~5 solves per step. Keep for the large blocks.

**Screened greedy (M 4) vs the lineup** (compare_clear.py; FULL 220: greedy 0.603 vs 0.360-0.368 for
the top four road arms at Lens A, +0.212..+0.218, wins on every block; first read on 139 below):
Lens A roadless perm 0.622 vs 0.37 for the best road arm (+0.216..+0.221 vs the top four, wins
on EVERY block); Lens B P*' 0.25 at D 0.016 vs 0.051 (resist_lp). Frontier: greedy on all 139.

Caveats, measured:
- It is an ORACLE: it optimizes exactly the score it is graded on (cf. the vehicle-access direct
  optimizer). Re-scoring its clearing under other settings (rescore.py: h 0.25, shifted grid,
  ell 1, ell 10) is running.
- It clears SCATTERED, LARGE, often free-standing buildings, not lanes (clear_vs_cycle_native_*.png).
  Displacement counts a building as one home whatever its size, while freed space scales with
  area. At 10% of homes the greedy clears 18.1% of footprint AREA (its buildings average 1.78x
  the block mean) vs 11.4-12.7% for cycle_native / resistance_lp / arterial (area_share.py, 40
  blocks).
- MATCHED ON FOOTPRINT AREA the gap halves but holds (area_match.py, 155 blocks): at 10 / 15 / 20%
  of area cleared the greedy leads the best road arm (cycle_native) by +0.126 / +0.131 / +0.134,
  winning on 95-99% of blocks. And the greedy optimized per home, not per m^2.

**Not a discretization or ell exploit** (rescore.py, 20 random blocks; the greedy's Lens A set
chosen at h 0.5 / ell 3, re-scored; gap = greedy minus the best of resist_lp / cycle / arterial):

    setting   greedy  best road  gap median [min, max]   greedy ahead
    base      0.630   0.437      +0.194 [+0.083, +0.248]  20/20
    h 0.25    0.636   0.445      +0.200 [+0.081, +0.241]  20/20
    shifted   0.633   0.439      +0.191 [+0.064, +0.238]  20/20
    ell 1     0.589   0.395      +0.190 [+0.092, +0.229]  20/20
    ell 10    0.700   0.501      +0.196 [+0.075, +0.277]  20/20

Even a strong straightness preference (ell 10) does not penalize the scattered pattern: the
turning cost prices how paths bend, not whether the open space forms lanes.

Open question for the owner: is a scattered pattern of plazas what the metric SHOULD reward? The
roadless score values open ground near congested homes; nothing in it asks for continuous lanes
beyond the turning cost (ell 3 m is weak).

## Population proportional to footprint area (owner, 2026-09-29)

Owner: buildings contribute (fractional) population proportional to their area. Implemented as a
population Strategy (common.CountPopulation / AreaPopulation, weights normalised to mean 1) that
sets BOTH the escape demand (building j injects w_j) and the displacement (sum_i c_i w_i / sum
w_i; `common.prefix_to` = prefix_to_displacement with weights -- the count version reproduces
the project's prefixes exactly). The greedy ranks by tension per unit population and clears the
best exact gain per unit population displaced. Also robust to Open Buildings merging adjoining
shacks (a merged blob counts by its area). Caveat: large non-residential footprints are
overweighted, and neither count nor area can tell.

Also answered: the score DOES price width (conductance ~ width/length) and straightness (turning
cost) and charges length; what it lacks is a THRESHOLD (a continuous path >= W to the street),
and contention penalizes funnelling, so it widens bottlenecks rather than cutting lanes.

Running: study.py ... area (rows_h0.5_ell3_K8_area/), clear.py ... area
(clear_rows_M4_h0.5_area/).

## Equity: J_p = sum_i w_i u_i^p (owner chose option 2, 2026-09-30)

u_i = building i's CONGESTED escape time: the injection-weighted mean potential of u = L^-1 f over
its ring cells, so sum_i w_i u_i = P and p = 1 is the current score. u_i is half the marginal
total cost of one more person leaving from i (dP/df_i = 2 u_i). Score: 1 - (J_p / J_p0)^(1/p).
- Same family as the screen's closed form (notes 2026-09-19): ring counts are linear in depth, so
  any polynomial weight collapses to n x const x depth_proxy^m; with u roughly quadratic in ring
  depth, J_p at baseline closes to n x const x depth_proxy^(2p) (untested).
- Adjoint: dJ/dw_e = -(du_e)(dlam_e), L lam = p u_i^(p-1) x injection. p = 1 is self-adjoint
  (one solve); p != 1 costs one more solve for ALL buildings. Validated at p 2, area population,
  41132: Spearman(tension, exact LOO gain) +0.936, tension's #1 is the true best.
- NOT monotone in general -- but dead ends are neutral (no steady current into a demand-free
  pocket; that worry was about hitting times). The remaining non-monotonicity is CONGESTION
  SPILLOVER: opening a route for buried homes pushes flow past the homes at its outlet.

Running (2026-09-30 00:47): study.py area rerun with P2_carve and tail columns (u95, umax, umed =
ratio to baseline) into rows_h0.5_ell3_K8_area/ (the p = 1-only area rows are in
rows_h0.5_ell3_K8_area_p1only/, obliterate dropped); clear.py area p 2 into
clear_rows_M4_h0.5_area_p2/; clear.py area p 1 still running (clear_rows_M4_h0.5_area/).

### Area population results (2026-09-30)

Greedy (area, p 1) vs the lineup re-scored with area demand and area displacement
(compare_clear.py 4 0.5 area, all 220): Lens A 0.537 vs 0.327-0.334 for the top road arms,
+0.171..+0.195, wins on every block; Lens B P*' 0.25 at D 0.021 vs 0.068 (cycle / resist_lp /
arterial). Charging by area barely dents the lead: targeting, not the size discount, is most of it.

Tails at Lens A (tails.py area 1): u relative to baseline, median over blocks of the median /
p95 / max home: greedy 0.513 / 0.411 / 0.355; cycle 0.703 / 0.613 / 0.699; resist_lp 0.722 /
0.621 / 0.703; arterial 0.713 / 0.632 / 0.688; grid 0.701 / 0.646 / 0.753. Greedy minus the best
road arm per block: p95 -0.093 [-0.103, -0.086] (better on 96%), max -0.105 [-0.122, -0.087]
(95%). The p = 1 greedy does NOT neglect the buried; it helps them most. Its P2 is 0.569 vs 0.34.

### p 2 vs p 1 greedy (area population, 219 blocks; 5618 still running)

tails.py area 2: greedy p 2 at Lens A: umed 0.531, u95 0.397, umax 0.334, P1 0.543, P2 0.582
(road arms unchanged: u95 0.61-0.65, umax 0.69-0.75). Paired p 2 minus p 1 greedy: median home
+0.018 (p 2 worse on 79%), p95 -0.014 (better on 80%), max -0.011 (76%), P1 -0.004, P2 +0.003.
p 2 does what it says -- trades a little of the median for the tail -- but the effect is small
next to the greedy-vs-roads gap (p95 -0.11, max -0.13 vs the best road arm per block).

## Scaling: batching the greedy (2026-09-30 -- 10-01)

One-at-a-time screened greedy (M4) on ZAF.9.3.1_1_5810 (6,619 buildings, 577k m^2, 14.6M
unknowns, one solve 495 s single-threaded, 11 GB) would be ~800 steps x 5-6 solves: 600-700 h.
Setup fixes: one baseline solve instead of two; building labels by per-footprint bounding-box
rasterization (2 s instead of a whole-grid point query; differs only where footprints overlap).

Pickers (clear.py, a Picker Strategy). Area population, p 2, d_max 0.15, all 220; Lens A
interpolated to exactly D = 0.10 (picker_compare.py; first-step-past-0.10 scored coarse pickers
at ~12%). Medians; diff = paired median vs M4:

    picker                         Lens A   diff     Lens B 0.35   compute (load-confounded)
    M4  one at a time, top-4 exact 0.565    --       0.039         112 h
    1%  rounds, 3 m spacing        0.552    -0.010   0.043          27 h
    1%  rounds, catchment          0.550    -0.011   0.044          21 h
    1%  rounds, impact             0.551    -0.011   0.044          44 h   DOMINATED (ties cat, 2x)
    1%  rounds, no spacing         0.548    -0.014   0.045          22 h   DOMINATED by cat
    3%  rounds, catchment          0.537    -0.022   0.052           9 h
    3%  rounds, impact             0.542    -0.018   0.049          27 h   DOMINATED by 1% cat
    3%  rounds, sketch k 32        0.526    -0.026   0.054          26 h   DOMINATED (noisy)
    3%  rounds, no spacing         0.519    -0.038   0.056           9 h   DOMINATED by 3% cat

Spacing = how alike two candidates' effects are. Impact: correlation of their impact fields
L^-1 dL_i u (energy inner product; the exact second-order cross term for p 1). Sketch: the same
from k random edge-load probes. Catchment: overlap of the homes whose downhill flow passes
through each (no solves). Batch built under a normalised quadratic model, worth_i = g_i -
sum_{j in batch} rho_ij sqrt(g_i g_j) (duplicates worth 0, complements gain). Measured on 41132:
touching pairs ranged rho +0.70 (redundant) .. +0.11 (independent) .. -0.21 (complementary), so
distance is a poor proxy; sketch(400) matched impact (corr 0.95), catchment only weakly (0.35).
- At 1% steps spacing barely matters (every rule ties; +0.0005 over none): the loss vs M4 is
  from not re-ranking after every clearing, not from interference.
- At 3% it matters: catchment recovers ~1/3 of no-spacing's loss at the same cost; impact a
  little more (+0.0025 over catchment) at 3x.
- Impact, sketch and the null were deleted (frontier rule). Kept: M4, B0.01g3 (simplest),
  S0.01cat, S0.03cat.

**5810 done** with B0.01g3: 15 rounds, 7.3 h single-threaded on a loaded machine (vs ~650 h).
Lens A: 618 of 6,619 cleared, roadless perm (p 1) 0.330 vs cycle_native 0.183
(clear_vs_cycle_native_ZAF.9.3.1_1_5810_B0.01g3_area_p2.png). Early rounds breach the dense
fabric along the street edges (all flow converges there: tension = current^2 x conductance gained
peaks at the outlets); later rounds work inward; also clears the large institutional buildings by
the southern field; leaves the sparse west half and the north-east strip nearly untouched.

## Road-favouring conductances (owner, 2026-10-01)

Why the greedy nibbles: in a linear conduction model a channel's conductance grows only as its
width -- a 3 m lane conducts like three 1 m strips -- so nothing rewards space being continuous,
wide or straight. Three candidate fixes, mapped on 5810 before any clearing
(conductance_map.py; owner: "pretty impressed by all three"):
- width: local clear width (2 x distance to buildings; the street side of the edge counts open).
- sightline: longest straight run >= 1.5 m wide over 8 headings.
- vehicle layer: width >= 4.5 m, connected to a street. On 5810 it reaches most of the block,
  so a 4.5 m threshold barely separates lane from gap there.

Segment measure (owner's lift idea): the number of straight segments of length l a W-wide
corridor holds, over positions and headings, is ~ W^2 / l -- so rewarding the FAMILY of segments
gives width^2 without assuming it. Implemented as a per-heading multiplier on the along edges
(lifted.AlongConductance Strategy in Params): layer k's factor = 1 + beta x the angle-weighted
(triangle, one axis gap) mean over 48 fine lattice directions of the run through the cell.
sightline_*.png maps the runs' mean (a connected path network) and dominant orientation (long
coherent alleys). Checks (sightline_checks.py): channel resistance ratio to the original model at
beta 3 falls 0.67 -> 0.46 from W 1 to 8 m (0.82 -> 0.44 at 30 deg); freeing never raises P;
operator build 0.2 s vs 0.1 s.
- Sightline (hard runs, cap 50 m) at beta 3 / 10, greedy S0.01cat on six blocks: SAME scattered
  pattern as the original (alongs_*.png). Road lineup re-scored under it (along_lineup.py, five
  blocks, p 1 at 10%): greedy minus best road +0.110/+0.080/+0.052 (30686, uni/3/10),
  +0.068/+0.023/+0.012 (41148), +0.161/+0.145/+0.147, +0.130/+0.114/+0.121, but +0.170/+0.194/
  +0.282 on 41132 (wide open space whose runs the greedy boosts). Roads gain more than the
  greedy as beta rises, but do not overtake it -- and the road corridors were designed for other
  objectives, so this does not show lanes cannot win.
- SoftSightline (owner: translucent buildings): a ray's intensity falls as exp(-kappa x closed
  length crossed); R_d = expected free path; saturating R / (R + r0) (r0 20 m) instead of the
  arbitrary 50 m cap. Differentiable: vjp by reverse-mode through the 1-D scans, exact to 1e-9
  vs finite differences. Tension now = local + NONLOCAL gain (clearing lengthens rays through
  the building, boosting edges elsewhere); matches finite differences of P within a few % (the
  residual is the old local term's finite step). Measured: the nonlocal term is < 1% of the gain
  at beta 10, kappa 2, and < 2% at kappa 0.3 -- at these strengths lengthening runs is worth far
  less than opening a hole, so transparency alone will not make corridors form.

Beta sweep (SoftSightline kappa 2, greedy S0.01cat, five blocks; 5618 still running): the
pattern stays SCATTERED at every strength (alongs_*.png, uni / ss10k2 / ss30k2 / ss100k2). Greedy
minus the best road arm at 10% (p 1, under each conductance; along_lineup.py):

    block   uni     ss10k2  ss30k2  ss100k2
    19570  +0.161  +0.152  +0.183  +0.198
    30686  +0.110  +0.088  +0.057  +0.068
    41132  +0.170  +0.294  +0.376  +0.438
    41148  +0.068  +0.065  +0.066  +0.070
    63718  +0.130  +0.099  +0.094  +0.099

Roads never overtake; on 41132 the greedy pulls far ahead. Reading (untested interpretation): the
open space between footprints ALREADY forms a connected network of straight-ish channels (the
conductance maps); a strong straight-run bonus makes that network fast, and the binding
constraint becomes the last few metres from homes to it -- which is what the greedy breaches.
A new lane would duplicate a channel that exists. The model's assumption doing the work: every
gap between footprints is walkable (no walls, fences or private yards).

Was running: beta sweep 10 / 30 / 100 (kappa 2) on the six blocks. Next: completion field
(smoothing along headings in the lifted space -- the owner's "smooth only toward nearby
corridors with similar angles"), and corridor-forming via a transparent SEARCH kappa annealed
toward the scoring kappa, if the sweep shows lanes can pay.

### Length response and its scale (owner, 2026-10-01)

Length was SUBLINEAR: g(R) = R / (R + r0) on R = F + B is concave (a 10 m gap earned 1/3 of the
max at r0 20 m). Owner: integrating along a corridor should give length the same quadratic weight
integration gives width. Made exact: the segments along a heading through a cell are the pairs
(start behind, end ahead), F x B of them, summing to L^2 / 2 along a line -- the natural lift
measure is int F_theta B_theta dtheta. As a conductance it chokes a line's ends (B ~ 0 at the
street), and unbounded growth is unphysical, so the metric uses an S-curve on the whole line,
g = R^n / (R^n + r0^n) (SoftSightline hill n; exact vjp kept); F x B is used for display.

Choosing r0 (owner: max entropy / dynamic range). The max-entropy g is the CDF of the block's
line lengths (histogram equalization); the Hill curve IS a log-logistic CDF, so the fit is closed
form: r0 = median L, n = pi / (sqrt 3 sd log L), L = a cell's best line (max over headings,
soft F + B, before clearing). line_scale.py:

    block   n      maxent r0  n    Otsu    percolation (as defined: not discriminating)
    41132   138    31 m      3.3   21 m     60 m
    41148   234    30 m      3.2   23 m     69 m
    19570   312    19 m      2.7   18 m     74 m
    30686   346    20 m      2.4   27 m    146 m
    63718   372    13 m      2.2   13 m     74 m
    5618    759   123 m      1.9   81 m    404 m
    5810   6619    74 m      1.9  122 m    943 m

n ~ 2-3 everywhere (the S-curve's form is what the data prefer); r0 30 m suits the small blocks
but the big ones want 75-120 m. Otsu explains ~0.6 of the variance (not sharply bimodal).
Percolation of the long-line cells to the street stays ~1 until t reaches the block's longest
lines (edge and field runs connect everything): would need restricting to the interior.

S-curve greedy (n 2, r0 30 m, beta 30 / 100), five small blocks: still scattered; greedy minus
best road +0.142/+0.150 (19570), +0.052/+0.035 (30686), +0.368/+0.516 (41132), +0.059/+0.064
(41148), +0.097/+0.119 (63718).

5618 (S0.01cat; uni / sl10 / ss10k2 / ss30k2): the clearing still targets the frontage of the
two built bands around the central field and the street edges; under ss30k2 the F x B view shows
long lines fanning ACROSS the central field between cleared buildings on its two sides -- the
field is the corridor, and the greedy opens its frontage (opened_ZAF.9.3.1_1_5618_S0.01cat.png).
Running: 5618 at the fitted scale (ss100k2n2r123), 5810 (S0.03cat) uni / ss30k2 / ss100k2 /
ss100k2n2r74.

### Speed (owner: "several hours on a big block is a lot", 2026-10-01)

Per solve on 5618 (single thread): h 0.5 setup 7.8 s, solve 29 s at rtol 1e-9 (37 it), 19 s at
1e-5 (P identical), 13.5 s at 1e-3 (P to 1e-6); h 1: setup 1.6 s, solve 5.8 / 3.4 / 2.4 s, P
+2.5%. Tension at rtol 1e-3 ranks IDENTICALLY to 1e-9 (5618, h 0.5 and 1: Spearman 1.000000,
top-100 overlap 100/100) at half the time; set RTOL_TENSION 1e-3, RTOL_SCORE 1e-5.
Search coarse, score fine (rescore_grid.py, S0.01cat, all 220): searched at h 1, re-scored at
h 0.5 at D = 0.10: Lens A 0.534 vs 0.550 searched at h 0.5, -0.0028 [-0.004, -0.002] (coarse
better on 33%); p 1 -0.0024. Compute 6.4 h vs 21.2 h (5618: 658 s vs 2,544 s). Together ~7.5x
on the tension (5618: 106 s -> 14 s).
Not done: a threaded AMG (pyamgcl does not build here -- needs Boost headers); the GPU is busy
with another user's job. 5810 relaunched at h 1, rtol 1e-3, S0.01cat: uni / ss100k2 /
ss100k2n2r74 (the old h 0.5 S0.03cat runs were stopped).

### Translucent search (owner, 2026-10-01)

Tension ranked under a more translucent SEARCH conductance (kappa_search), scoring unchanged
(ss100k2n2r30; `<along>@<search>`); greedy S0.01cat, GPU, six blocks; transparency.py at D 0.10.
depth = median street-distance percentile of the cleared buildings among all buildings; long =
m^2 of previously open space whose best line grew >= 30 m; perm = the run's score (fixed metric).

    block   kappa_s:  2      0.5    0.2    0.05     (depth / long m^2 / perm)
    19570   48.7/151/.836   47.8/284/.848   48.6/165/.840   47.0/205/.820
    30686   50.0/ 52/.732   48.8/ 94/.759   45.1/180/.765   40.8/180/.758
    41132   49.3/ 56/.861   50.0/ 57/.865   54.3/140/.830   52.9/ 45/.642
    41148   56.2/130/.632   54.7/344/.664   53.0/211/.663   54.7/211/.627
    5618    40.6/2300/.876  40.4/4393/.891  32.4/2955/.877  32.3/2322/.844
    63718   52.6/ 81/.835   58.1/159/.798   52.4/178/.652   48.9/134/.602

- NOT deeper: depth flat or SHALLOWER (5618 41 -> 32 percentile at kappa <= 0.2). Counterfactual
  corridors are anchored at the street, so the buildings opening them sit at the frontage.
- LONGER corridors: kappa_s 0.5 roughly doubles the long-line area on 5 of 6 blocks (5618 2,300
  -> 4,393 m^2); 0.05 falls back (everything looks open; the gradient stops discriminating).
- Kappa_s 0.5 also scores BETTER under the fixed metric on 5 of 6 (5618 0.876 -> 0.891, 30686
  0.732 -> 0.759, 41148 0.632 -> 0.664; 63718 worse, 0.835 -> 0.798): the translucent gradient is
  a better search, not just a different one. 0.05 is worse (41132 0.861 -> 0.642, 63718 -> 0.602).
Six blocks only; the 220-block check is cheap on the GPU.

**220-block check (2026-10-01): kappa_s 0.5 wins.** Metric ss100k2n2r30, S0.01cat, h 0.5, CPU,
paired per block (translucent220.parquet), medians:

    perm (J_2) at D 0.10   kappa_s 2 0.7669   0.5 0.7889   diff +0.0144 [+0.012, +0.017]   better on 85%
    perm1 (P)  at D 0.10             0.7048       0.7281        +0.0211 [+0.019, +0.024]             90%
    perm (J_2) at D 0.15             0.8423       0.8526        +0.0119 [+0.009, +0.014]             86%
    largest quarter (n >= 232)                                  +0.0227 [+0.015, +0.026]             87%
    smallest quarter                                            +0.0121 [+0.007, +0.016]             85%

Same cost (one extra layers evaluation per round). A more translucent gradient is a better search
for the sightline metric, most on the big blocks; ss100k2n2r30@ss100k0.5n2r30 is the preset for
that metric. Untested on 220: kappa_s 0.2 (mixed on the six), and whether the gain carries to
the uniform metric (uni has no kappa to soften; a translucent sightline SEARCH for a uni metric
would be a different question).

### 5810 at full resolution on the GPU (2026-10-01)

S0.01cat, h 0.5, GPU solver: uni 1,839 s (the CPU estimate was ~6 h); ss100k2n2r74 3,332 s with
CPU scans; ss100k2 2,446 s with GPU scans (only 27% faster: on 5810 operator assembly, catchment
and the pyamg aggregation now dominate). perm' at D 0.15: 0.385 / 0.703 / 0.628. Pictures
(alongs_ / opened_ZAF.9.3.1_1_5810_S0.01cat_h0.5.png) match the h 1 runs: under sightline,
coherent diagonal bands and several long straight lines through chains of clearings; under uni,
almost none. Three GPU runs plus 18 workers exhausted the 48 GB card (cupy's pool grew to ~20 GB
per 5810 run before the pool release was added).

### GPU-resident rounds (owner: "move the rest to the GPU", 2026-10-01)

Profile of one 5810 round (h 0.5, ss100k2, S0.01cat, GPU solver + scans; profile_round.py):
148 s = catchment sweep 76 (numba, one core) + operator assembly 24 (12 per System) + scan start
cells 16 (rebuilt every call) + pyamg setup 10 + GPU solves 14 + rest. Setup per block 69 s.
Now one round is **13 s** (11x) and setup ~40 s:
- **Edge pattern once per grid** (lifted.Pattern): every edge among the inside cells, built once;
  a field's operator keeps the edges whose cells are all open and fills in weights, on the
  solver's device (numpy or cupy, the Solver Strategy carries xp / sparse / ndimage). Identical
  matrices to the old assembly (max |diff| 7e-15, the float32 layers).
- **Grounded components by labelling** the free mask (4-connected: a diagonal or knight step
  needs every cell it crosses open, so the operator's cell graph is exactly 4-connectivity);
  identical masks.
- **One coarsening per block** (Coarsening, GpuAMG): pyamg's standard aggregation of the
  operator on every inside cell, restricted to each system's unknowns, Galerkin sums on the GPU.
  pyamg's aggregation at strength theta 0 reads only the sparsity pattern, so for the eps
  (tension) systems this is the hierarchy pyamg would build afresh. System setup 5 s -> 0.2 s.
- **K-cycle** (Notay: each coarse correction two flexible Krylov steps preconditioned by the next
  level) instead of a V-cycle: 5810 eps system to 1e-3 in 63 iterations instead of 181, score
  system to 1e-5 in 98 instead of 272 (2.2x the speed). Measured and dropped: 2-3 Jacobi sweeps
  (no gain), strength theta 0.25 (3-7x slower: knight-axis layers lose every strong link),
  Notay's pairwise aggregation (3-10x slower per solve), smoothed prolongators (cuSPARSE's
  products wanted a 52 GB buffer on 5810, still 14-31 GB in row blocks). Convergence stays linear
  (~0.94 per iteration at tight tolerance on 63718), so rtol 1e-9 takes ~350 iterations: the cap
  is now 2000. A better coarsening for this anisotropic operator (aggregates along each heading's
  chains, then across) is the remaining solver lever.
- **Catchment sweep on the GPU** (Sweep Strategy: CpuSweep numba / GpuSweep): the downhill graph's
  topological waves by Kahn's algorithm (a node is ready once all its lower neighbours are; one
  launch per wave; 5618 is 2,113 waves deep, ~1,100 nodes per wave), then one launch per wave
  for the sweep, a thread per (node, candidate). Same arithmetic per node: 76 s -> ~1 s.
- **Tension on the device**: edges, gains, the nonlocal vjp and the building attribution (one
  sparse buildings x cells matrix) never leave the GPU; only u (for the per-building escape
  times) and the per-building gains come back.
- **Grid construction**: block membership by a scanline fill of the boundary rings, footprints by
  per-footprint bounding-box rasterization (what label_sub already did): identical sub-samples
  on four blocks, 5810 58 s -> 6 s.
Checks: new CPU vs old CPU and new GPU vs old GPU on 41132, 41148, 5618: tension within the
solve tolerance (<= 1e-4 relative), identical candidates and picks, Gram matrices <= 1e-4.

**Impact spacing on the GPU: still dominated.** Block CG (16 right-hand sides) makes it 56x
faster than on the CPU (63718, 30 candidates: 6.9 s vs 387 s), but catchment on the GPU takes
0.1 s there, and on 5810 at 3% Impact's Gram had not finished after 12 min against ~1 s. Its
quality edge (+0.0025 over catchment at 3%, a tie at 1%) cannot pay for that; deleted again.

End to end (5810 and 5618, ss100k2, S0.01cat, GPU): identical picks in 14 of 15 rounds on 5810
(the other within solver tolerance) and 15 of 15 on 5618, same scores to 4 decimals; 5810 2,446 s
-> **195 s** of rounds (+ ~40 s setup), 5618 25 s.

**Current maps** (current_map.py; owner: "show the current through a point, not the
conductance"). Per cell, the sum over headings of |flux| under the run's own conductance, before
and after its 10% clearing (current_ZAF.9.3.1_1_5810_S0.01cat_h0.5.png). Before: a dendritic
drainage network toward the street. The diff is global: a cleared gap lights the whole route it
feeds, out to the street (red), and relieves the routes it replaces (blue); under sightline the
red is line-like, long straight streaks running past the cleared buildings. Not monotone per cell
and should not be: only P (and each home's resistance to the street) must fall; flow reroutes
into the new route as traffic does onto a road. Total current (person-metres) fell 1.33M ->
1.23M (uni), 1.41M -> 1.24M (ss100k2).

### Turning sightline (tried and dropped, 2026-10-01)

Owner's ask: corridors that connect and reinforce, bends still rewarded. Built as SoftSightline
with walkers that keep their heading for turn_m metres on average, then take a fresh one worth
Gbar (a power mean, exponent `sharp`, of the free paths over all headings), settled by
fixed-point passes; vjp exact (central differences 1e-6..1e-9, CPU = GPU to 1e-16). On a T
junction (a 50 m stem meeting a bar):
- uniform fresh heading (sharp 1): turning only SHORTENS (most fresh headings hit a corridor
  wall): stem 48 m straight -> 32 m at turn_m 30 -> 19 m at 10; the junction adds ~0.5-1 m.
- best fresh heading (sharp 8, 32): the reverse heading is a "long" option, so walkers
  ping-pong: the stem ALONE scores 54 m, 110 m at sharp 32.
Fixing it (no U-turns, turn only where it pays, a turn cost, a per-metre discount against loops
round a building) builds a longest-polyline reach with no destination, which is exactly what
needs the loop guards. Owner's point: that is a problem of myopia. The flow solve already rewards
connection globally (a gap that joins a corridor carrying flow to the street has a large
tension), the lifted operator already charges curvature (turning edges), and potential flow
cannot loop (it runs downhill: the catchment sweep's DAG). Dropped; the code is in 4f56c4b.
What would revive it: a measured gap that the flow cannot see, e.g. a block where the greedy
leaves a T unjoined although joining it pays under P.

### One convex program, and SIMP (owner: "a very simple single approach?", 2026-10-01)

relax.py. Clear a fraction x_j of each building; open fractions are affine in x, edge weights are
mins of them (concave), and P(w) = max_u 2 f^T u - sum w (du)^2 is convex non-increasing in w,
so min P(x) s.t. population . x <= D, x in [0, 1] is CONVEX. Frank-Wolfe on it: the gradient is
the tension (-du dlam per edge; a min's gradient shared over the cells attaining it, a valid
subgradient), the linear step is the batch pickers' fractional knapsack, and the gap certifies a
bound. Objective in the eps world (J_eps <= J, so the bound covers real clearings). Gradient vs
central differences 1e-6 (p 1 and 2).

Result, 220 blocks, p 1 (P), D 0.10, against the greedy run on the same objective (S0.01cat,
p 1, clear_rows_S0.01cat_h0.5_area); relax220_p1.parquet; medians:

    convex optimum (fractional)        0.670
    certified bound (no clearing beats) 0.675   greedy is 0.150 below it (quartiles 0.13-0.17)
    convex optimum rounded by x        0.445   -0.052 [-0.059, -0.044] vs greedy, wins on 7%
    SIMP (q 1.5 .. 3), rounded         0.516   +0.004 [+0.002, +0.005] vs greedy, wins on 60%
                                               (largest quarter +0.006)
    greedy (BESO-like batches)         0.517
    time per block: greedy 97 s (CPU), relaxation + SIMP 154 s (FW start + 80 OC steps)

- The convex optimum is GREY: 73% of buildings partly cleared (median). P is convex in the
  conductances, so spreading a little clearing everywhere (every building made porous) beats
  concentrating it (Jensen). Corridors pay only because clearing comes in whole buildings: road
  form comes from the integrality, not from the flow model. Hence the weak rounding and a loose
  bound (0.15 above the greedy: it does not say the greedy is near optimal).
- This is topology optimization of heat conduction (the volume-to-point problem; thermal
  compliance f^T K^-1 f = P, conducting material = clearing; Bejan's constructal trees). Its
  standard method, SIMP (material conductivity x^q, q -> 3, optimality-criteria updates), drives
  x to 0/1 and edges out the greedy on P. The greedy itself is that field's BESO (add whole
  elements by sensitivity, a fixed volume step per iteration).
- On J_2 (Lens A), 220 blocks (relax220_p2.parquet; not convex, so Frank-Wolfe is a local
  method and its gap no bound), paired against the J_2 greedy (S0.01cat p 2):

    convex-start optimum (fractional)  +0.141 vs greedy
    rounded by x                       -0.063 [-0.076, -0.051], wins on 3%
    SIMP rounded                       +0.002 [-0.003, +0.006], wins on 55%: a TIE
    SIMP rounded, largest quarter      +0.006 [+0.003, +0.011], wins on 67%
    time: relaxation + SIMP 377 s median per block, greedy 261 s (older run, other load)

  (5618, where SIMP rounded 0.30-0.34 against the greedy's 0.40 from every start, is an
  outlier.) Frontier: the greedy is cheaper and ties overall; SIMP is better on big blocks
  and on P. SIMP still leaves ~8% of buildings grey; its rounding (by decreasing x) is the
  obvious place it loses value.

**SIMP, cheaper** (relax.py plans `fw<FW its>.q<final q>.i<updates per q>.t<step rtol>`).
Warm-started CG (System.solve x0: the last solve of the previous x; the eps world keeps the same
unknowns), solves to 1e-3 while optimizing (scores stay exact), no Frank-Wolfe start (uniform
x = D). 40 blocks (10 largest + 30 random), J_2, D 0.10, against the 220-run plan
fw40.q3.i20.t1e-05:

    plan                  vs 220 plan   vs greedy                     wins   median time
    fw40.q3.i20.t1e-05    --            +0.009 [+0.001, +0.015]       68%    386 s
    fw0.q3.i20.t0.001     +0.000        +0.010 [+0.003, +0.015]       70%     93 s   <- default
    fw0.q5.i10.t0.001     -0.000        +0.009 [-0.003, +0.017]       60%     96 s
    fw0.q5.i20.t0.001     +0.000        +0.010 [+0.002, +0.015]       68%    189 s

Same answers at a quarter of the time. Continuing to q 5 halves the grey (8% -> 4%) but does not
raise the rounded score. (Times are under 12 parallel workers; nearly every block is over
lifted.SMALL unknowns, so these are AMG solves, not direct ones.)

**SIMP on 5810** (J_2, D 0.10, GPU, uni): greedy (S0.01cat) Lens A 0.3189; SIMP fw0.q3.i20.t0.001
**0.3468 (+0.028)** in 164 s wall including setup (the greedy is ~4 min to D 0.15 at today's
speed); fw0.q5.i20.t0.001 0.3476 in 309 s. The gain grows with block size (220: +0.002 overall,
+0.006 largest quarter, +0.028 on the biggest block of all). simp_vs_greedy_ZAF.9.3.1_1_5810_*.png:
SIMP clears 692 mostly small buildings packed into the dense band along the eastern street edge
and the south-west edge; the greedy 524, including the big institutional buildings by the
southern field, more scattered; 279 in both. Neither forms corridors under uni.
Not dominance: the greedy gives every budget in one run (nested clearings: Lens B, phasing),
SIMP one budget per run; SIMP is untested under the sightline conductances.

**SIMP under the sightline metric** (ss100k2n2r30, J_2, D 0.10, 220; the gradient now includes the
factors' response to opening via their vjp, exact against central differences):
    vs greedy, search = metric        +0.003 [-0.002, +0.008] wins 55%; largest quarter +0.010
                                      [+0.002, +0.018] wins 71%
    vs greedy, translucent search     -0.010 [-0.013, -0.006] wins 26%; largest quarter -0.005
SIMP ties the plain greedy and loses to the translucent-search greedy. Untried: SIMP with the
same trick (its gradient under the kappa 0.5 conductance, the objective under the metric).

**Block sizes.** The 220 are small blocks (median 137 buildings, max 759); the region's 28,768
blocks have median 39, but 82 have >= 1,000 buildings (145k buildings, 7.5% of all), 16 >= 2,000
and 6 >= 4,000 (5810 6,619; then 5,396, 5,023, 4,542, 4,374, 4,365). Every 220 result is a
small-block result; block_sizes.parquet lists them all. Large-block benchmark: the 82 >= 1,000.

### Large blocks (owner: "I only really care about the large blocks", 2026-10-02)

The 82 blocks with >= 1,000 buildings, J_2, uni, Lens A at D 0.10, one GPU process at a time
(large82_simp_vs_greedy.parquet; areas in large82_area.parquet).

**What fits.** Area, not building count, sets the memory: the eps world solves on every inside
cell x 8 headings. The greedy completes every block up to ~1.4 km^2 (59 of 82; 30796, 1.5 km^2,
48M unknowns, peaks at 48.2 GB of the 48 GB card), SIMP up to ~1.5 km^2 (61). The other 21-23
(1.4 -- 52 km^2: peri-urban land, 1,000 -- 4,400 scattered buildings) do not fit at h 0.5; they
need a coarser grid or the grid cropped to the built-up area, both model changes (owner's call).
Two memory fixes on the way: GpuAMG.prepare frees cupy's pool per system (the rewrite had
dropped it), and the catchment sweep's (unknowns x columns) output is sized from free memory
and written column-major (cuSPARSE's csr @ dense copied the row-major one: 2x per chunk).

**SIMP (fw0.q3.i20.t0.001) vs greedy (S0.01cat), 59 blocks:** median +0.005 [+0.002, +0.008],
SIMP wins 66%, but mean -0.031: SIMP collapses on six blocks (22422: greedy 0.814, SIMP 0.014;
30848 0.770 / 0.434; 23597 0.772 / 0.503; 20543 0.556 / 0.234; 38616 0.566 / 0.400; 46841
0.573 / 0.429). Median time 2x the greedy's to D 0.10 (5810 the exception: 142 s vs 1,168 s).

**Why it collapses: stranded pockets behind a gate.** 22422's J_0 is 1.2e8 (pockets of homes
with enormous escape times); the greedy's first round clears one building (cost 0.01) and gets
0.534. SIMP optimizes in the eps world, where closed buildings conduct EPS 0.01 and every grey
building x^q of its cells: there the pocket already drains (eps-world Lens A 0.75 with nothing
cleared), so the gate is worth little. At the uniform start x = 0.10 the gate ranks 391st of
1,977 by -gradient / cost (1st at x = XMIN); SIMP ends with it at 0.001.
  - Frank-Wolfe start (fw5): its rounding finds the gate (22422 0.796, 23597 0.751, 20543
    0.496), but the SIMP that follows lands where the uniform start did (0.014, 0.503, 0.234):
    not the start, the leaky grey intermediate states.
  - Smaller eps (plan suffix .e<eps>): 1e-4 and 1e-3 alike fix 30848 (0.434 -> 0.719) and 46841
    (0.429 -> 0.522), leave the other four, and nudge the controls up (5810 0.3468 -> 0.3479).
  - SIMP polishing the greedy's clearing (x0 = its 0/1 set, q 3, or q 2 .. 3): <= +0.007; it
    stays in the greedy's basin (5810: 0.318, where SIMP from uniform finds 0.348).
So neither contains the other: SIMP's grey continuation finds better spread-out clearings
(5810 +0.03), the greedy's exact rescoring finds gates. Best of both, scored exactly, is the
robust answer for large blocks at the cost of running both.

**eps 1e-4 on all 59** (fw0.q3.i20.t0.001.e0.0001; now the plan to use, quality-first; e 0.01
stays selectable, 15% faster):
    vs greedy                      median +0.007 [+0.005, +0.011]  mean -0.026  SIMP ahead on 73%
    vs SIMP at eps 0.01            median +0.001 [+0.001, +0.002]  mean +0.005  ahead on 90%
    best of greedy and SIMP e1e-4  median +0.007 [+0.005, +0.011]  mean +0.009  never behind
Not monotone: 1558 falls 0.523 -> 0.292 at e1e-4 (greedy 0.596). Still collapsed: 22422, 23597,
20543, 38616. Median time to D 0.10: greedy 29 s, SIMP 61 s (e 0.01) / 70 s (e 1e-4); totals
over the 59: greedy 3,681 s, SIMP e1e-4 6,354 s.

### SIMP speed (owner: "why hours and not minutes?", 2026-10-02)

Where the hours went: one SIMP run is <= 80 OC updates of two solves each (forward, adjoint) on
every inside cell x 8 headings (30796: 48M unknowns, 241M nonzeros); the nested path reruns
the whole continuation at 15 budgets (15x); batches multiply again. The greedy is minutes.
Profile of one update on 30796, 5.1 s: solves 2.5 (18 -- 33 K-cycle CG iterations warm, each
fine SpMV 5.6 ms at ~660 GB/s, near the card's bandwidth), AMG setup 1.15, the CSR sort 0.4,
gradient 0.4, host-side home_u and copies 0.5.
Same-answer engineering (0488d72), 5.1 -> 2.8 s per update, SIMP end to end 1.5 -- 1.8x
(5810 169 -> 101 s; picks identical on 3 of 4 blocks, 3 of 621 differ on the fourth):
  - the eps world has the same sparsity pattern on every update: the CSR structure and each
    level's Galerkin map are cached per pattern (setup 1.15 -> 0.13 s), the power iteration for
    the smoother's rho warm-starts (5 iterations, not 30);
  - the K-cycle in single precision under the double outer CG (solves -28%); its dot products
    must accumulate in double (rho2 = bet - gam^2/rho1 cancels to NaN in single). Costs ~1 GB
    more memory: GpuAMG(single=False) stays for blocks at the card's edge;
  - home_u on the device (the 48M potentials no longer go to the host).
Fewer updates (fw0.q3.i10.t0.001.e0.0001 vs i20, 13 blocks): 3x faster in all with the above;
ordinary blocks move by -0.0004 median (-0.004 .. +0.0004), but the gated blocks flip
chaotically: 30848 0.719 -> 0.436, 20543 0.235 -> 0.538, 46841 0.522 -> 0.556. Whether SIMP
opens a gate is basin luck under the grey leak, which is what projection is meant to fix. i10
is the fast preset, i20 the default until projection is measured.

### Projection and the incumbent (owner: "a more natural, non-Frankenstein approach?", 2026-10-02)

Projection (plan .b<b0>-<bmax>: conductance H_beta(x)^q, budget c . H(x), beta doubling per stage)
stops the grey leak: on 22422 the eps world with nothing cleared drops from perm 0.79 to 0.08,
and the gate ranks 1st-2nd by -gradient / cost at the start (8th without). But SIMP still ends
at 0.012 (13 blocks, i10: b8-32 median vs greedy -0.011, b1-32 -0.049; 1558 rescued 0.29 ->
0.53, controls -0.005 .. -0.024).
**Why: the trace** (22422, b8-32): the gate and its two neighbours are substitutes (any opens
the pocket). Half open together, each looks redundant and OC cuts all three by the move limit;
shut, each looks vital and OC raises them: they flip x 0.3 <-> 0.5 every update. The rounding
of update 3 already scores 0.810, of q 2's first updates 0.815 and 0.819 (greedy 0.814); as
beta and q rise the flip breaks the wrong way and the gate ends at x 0.001.
**Incumbent** (plan .k<every>: round every iterate, score it exactly, return the best; repeated
roundings scored once): SIMP proposes 0/1 clearings, the real objective picks.
    12 blocks (6 gated + 6 controls; 1558 OOM), vs greedy at D 0.10:
    i10 (last iterate)    median -0.004  mean -0.126  worst -0.800  ahead 6/12
    i10.k1                median +0.011  mean +0.004  worst -0.037  ahead 8/12   ~2x i10's time
    i10.b8-32.k1          median +0.007  mean +0.002  worst -0.062  ahead 8/12
22422 0.814 / 0.819, 30848 0.782 / 0.708, 38616 0.528 / 0.573, 23597 0.750 / 0.760. Projection
is no longer needed for the collapses; it wins some gated blocks and loses some controls: kept
selectable, i10.k1 is the default. The exact scoring adds a second structure (real world) to
the solver's caches: 1558 (1.38 km^2) now runs out of memory.

**SIMP + incumbent on the large blocks** (fw0.q3.i10.t0.001.e0.0001.k1, J_2, uni, Lens A at D
0.10; 56 of the 59 that fit the greedy; 1558, 20023, 30796 run out of memory with the exact
scoring's second structure in the caches):
    vs greedy          median +0.009 [+0.006, +0.012]  mean +0.008  ahead 84%  worst -0.037
    (SIMP e1e-4 alone  median +0.007 [+0.005, +0.011]  mean -0.022  ahead 75%  worst -0.800)
    by size: 1,000-1,500 bldgs +0.008 (33); 1,500-2,500 +0.009 (20); > 2,500 +0.022 (3)
Median time 50 s (greedy 25 s to D 0.10). Worst: 38616 0.528 vs 0.565, 23597 0.750 vs 0.772.
Best of greedy and SIMP adds only +0.002 in mean over SIMP + incumbent alone.

**Nested SIMP path + incumbent vs greedy** (fw0.q3.i10.t0.001.e0.0001.k1 path, J_2, uni; the 6
largest blocks, 30796 out of memory): the path is ahead at every budget on all 5 blocks.
    block (n)      Lens A at D 0.02 / 0.05 / 0.10 (greedy -> path)       Lens B D for 0.25 / 0.35
    18739 (2881)   0.257->0.284  0.375->0.413  0.513->0.531              0.019->0.016  0.044->0.033
    18985 (2476)   0.293->0.314  0.448->0.469  0.582->0.595              0.016->0.014  0.028->0.025
    30848 (5396)   0.305->0.553  0.670->0.700  0.770->0.785              0.014->0.005  0.029->0.007
    5810  (6619)   0.113->0.163  0.221->0.251  0.319->0.343              0.061->0.050  0.121->0.105
    9710  (2109)   0.073->0.079  0.123->0.128  0.173->0.178              (perm never reaches 0.25)
Nesting costs little: the path at D 0.10 is within 0.004 of single-budget SIMP + incumbent
(0.531 vs 0.535, 0.595 vs 0.600, 0.785 vs 0.782, 0.343 vs 0.348, 0.178 vs 0.179). The gain is
largest at small budgets (30848: 0.305 -> 0.553 at D 0.02, the gate found first). Cost: 9 -- 29
min per block for 15 budgets, against the greedy's 0.5 -- 2 min (before today's speedups to both).

**Held-out confirmation** (the 44 large blocks not used for tuning; Lens A at D 0.10, J_2, uni):
    greedy S0.005cat vs S0.01cat         median +0.003 [+0.002, +0.004]  mean +0.003  ahead 82%  worst -0.001
    SIMP .k1s1e-06 vs greedy S0.01cat    median +0.009 [+0.006, +0.011]  mean +0.009  ahead 89%  worst -0.013
    SIMP .k1s1e-06 vs greedy S0.005cat   median +0.006 [+0.004, +0.007]  mean +0.006  ahead 91%  worst -0.013
Greedy S0.005 also needs a little less displacement for Lens B (perm 0.35: median -0.002 D, 38
blocks). Median time to D 0.10: greedy S0.005 36 s, SIMP .k1s1e-06 36 s (same). The eps-scored
incumbent (.k1s1e-06) gives .k1's answers (max |diff| 0.0002) 15% faster, and fits 1558,
20023, 30796 (greedy / SIMP: 0.596 / 0.576, 0.337 / 0.342, 0.480 / 0.513).
Frontier: at one budget SIMP + incumbent beats the greedy at equal time; the greedy stays for
the nested curve (every budget in one run, where SIMP's path is ~15 runs).

### Warm SIMP path and grow-then-prune (owner, 2026-10-02)

**Warm-started path** (.k1s1e-06.w3: each budget after the first starts from the last x, the
free buildings lifted to the even share, 3 updates per stage instead of 10): 2.6 -- 2.9x faster
than the cold path, a little worse. Lens A greedy / cold path / warm path:
    18985  D 0.02 0.293/0.314/0.309  D 0.05 0.448/0.469/0.448  D 0.10 0.582/0.595/0.573   557 -> 215 s
    30848  D 0.02 0.305/0.553/0.551  D 0.05 0.670/0.700/0.690  D 0.10 0.770/0.785/0.779  1749 -> 663 s
    9710   D 0.02 0.073/0.079/0.077  D 0.05 0.123/0.128/0.127  D 0.10 0.173/0.178/0.175  1356 -> 462 s
Lens B (D for 0.35) equal to the cold path's (30848 0.007 both, greedy 0.029). A cheaper
preset, not a replacement; 3 blocks.

**Grow then prune** (search.py GP3xS0.01catr0.005m8: the greedy to 3 x 0.15, then restore the
cheapest per round, the 8 cheapest by first order rescored in the eps world at 1e-6):
  - first-order restore scores alone fail (rank correlation 0.57 -- 0.74 with the exact restore
    loss; Lens A 0.38 vs the greedy's 0.60 on 19421), but exact backward elimination beats the
    greedy there (0.613 vs 0.601): the backward direction is sound, the scores were not;
  - with the screened shortlist: small blocks 19421 0.608 vs 0.601, 19510 0.490 vs 0.528, 19537
    0.863 vs 0.860 (0.655 vs 0.550 at D 0.02); 9712 0.182 vs 0.176;
  - **collapses on 22422: 0.014 vs 0.814**, the substitutes failure: with 45% cleared the
    pocket has other exits, so the gate is cheap to restore; its alternatives are restored one by
    one later, each cheap given the others, and the pocket shuts;
  - slow: 23 min on 9712, 61 min on 22422 (the SIMP path is 8 -- 29 min, warm 4 -- 11); the
    eps 1e-6 screen failed to converge on 30848 (AMG-CG 2.4e-5 after 2000); 5810 stopped after
    80 min of pruning.
Dominated by the SIMP path on the blocks measured (worse on the gated block, slower on every
block). What would put it back: a substitute-aware restore (score restoring a building together
with the others that carry its flow, or re-add after a restore that raises exact J -- the
floating schedule), and screening cheap enough to compete.

### Which blocks tell methods apart (adaptive block selection, first look, 2026-10-02)

Method x block matrices of Lens A at D 0.10, double-centred (block difficulty and method mean
removed), SVD; three matrices, each under one metric: 220 small x 10 methods (J_2 uni), 220
small x 3 (J_2 sightline), 48 large x 6 (J_2 uni).
- **One axis is nearly everything (87 -- 96% of the residual variance): collapses.** It sets the
  method that sometimes collapses (SIMP without the incumbent; the old Frank-Wolfe-started SIMP
  on small blocks) against the rest, and the top 10% of blocks carry 79 -- 97% of it: the gated
  blocks (large 22422, 30848; small 5521, 38149, 45919; sightline 5613, 41275).
- On large blocks the 4 most discriminating blocks rank the 6 methods as all 48 do (Kendall tau
  0.97; 4 random blocks 0.41), because collapses drive the means. On small blocks they are no
  better than random.
- **Without the collapse blocks, method effects are consistent:** ranking methods by mean
  within-block rank, 8 -- 16 random blocks agree with all of them (tau 0.81 -- 1.00), the most
  typical blocks too (0.87 -- 1.00), and the most "discriminating" by rank residual are the worst
  (-1.00 .. +0.69: atypical blocks, whose order departs from the consensus).
So a screen = a sentinel of gated blocks (collapses) + a random sample (everything else);
choosing "informative" blocks beyond the sentinel misleads. What sequential stopping would add
is the sample size: +0.003 (batch size) needs ~40 blocks, +0.009 (SIMP + incumbent) ~13.

### The large blocks under the sightline conductance (owner, 2026-10-03)

ss100k2n2r30, J_2, Lens A at D 0.10, the 13 tuning blocks (greedy / translucent search
ss100k2n2r30@ss100k0.5n2r30 / SIMP + eps-scored incumbent / SIMP under the same translucent
search conductance, incumbent and scores under the metric: relax `<metric>@<search>`):
    SIMP vs greedy (10)                    median +0.005  mean -0.002  ahead 7/10
    SIMP vs translucent greedy (10)        median -0.003  mean -0.006  ahead 3/10
    translucent greedy vs greedy (13)      median +0.002  mean +0.003  ahead 10/13
    SIMP-translucent vs translucent (13)   median -0.001  mean -0.005  ahead 6/13  worst -0.081
Median time to D 0.10: greedies 89 -- 90 s, SIMP-translucent 198 s. SIMP-translucent wins where
the score has room (5810 0.631 vs 0.595, 20269 0.764 vs 0.744, 9712 0.439 vs 0.432) and loses
near the ceiling (23597 0.881 vs 0.961, 46841 0.913 vs 0.940, 38616 0.870 vs 0.886).
**Lens A at D 0.10 saturates under sightline:** median 0.906 on these blocks, 54% above 0.9
(uniform: 0.522, none above 0.9); 22422's gate is worth nothing here (0.99 for every method).
A lens with room (D 0.02 -- 0.05, median 0.70 -- 0.83, or Lens B at a higher target) would tell
the methods apart better. Under this lens the translucent greedy is the frontier: SIMP ties it
at twice the time.
Memory: the sweep's column count now ignores the pool's cached fragments (22422 and 20543 failed
under sightline asking for 8 GB that was not contiguous); plain SIMP still runs out on 1558.

### Lens A at D 0.05 (owner: "We can try lens A at D 0.05", 2026-10-03)

D_LENS is now 0.05 (relax.py): at D 0.10 the sightline scores crowd the ceiling. SIMP + the
eps-scored incumbent (fw0.q3.i10.t0.001.e0.0001.k1s1e-06) at D 0.05 ran on the cluster
(cluster.py; V100 and RTX 8000 cards). The greedies' times are from the RTX 6000 Ada here, and
the same SIMP takes 1.5 -- 1.7x longer there at D 0.05 than here at D 0.10, so the SIMP time
ratios below overstate its cost by about that factor. Paired per block (simp_compare.py; a
greedy's perm is linear between its steps at D 0.05):

**Uniform conductance, the held-out blocks:**
    SIMP vs greedy S0.01cat (46)     median +0.015 [+0.009, +0.019]  mean +0.016  ahead 96%  worst -0.005
    SIMP vs greedy S0.005cat (44)    median +0.010 [+0.006, +0.015]  mean +0.011  ahead 91%  worst -0.008
    S0.005cat vs S0.01cat (44)       median +0.004 [+0.003, +0.005]  mean +0.005  ahead 93%  worst -0.001
SIMP's lead over each greedy is about 1.7x its lead at D 0.10 (+0.009 / +0.006: the same script
and blocks reproduce the held-out confirmation). Median time: SIMP 64 s (cluster), greedy to D
0.05 12 s (S0.01) / 18 s (S0.005). The greedy gets to a small budget early, while SIMP's
updates do not shrink with the budget. On the tuning blocks the gated ones are harder at the
smaller budget: 23597 SIMP 0.475 vs greedy 0.594, 1558 0.466 vs 0.515 (at D 0.10 the worst
was -0.020).

**Sightline (ss100k2n2r30), the 13 tuning blocks:**
    SIMP vs greedy                     median +0.008 [+0.001, +0.022]  mean +0.011  ahead 77%  worst -0.021
    SIMP vs translucent greedy         median +0.000 [-0.001, +0.003]  mean +0.024  ahead 62%  worst -0.015
    SIMP-translucent vs translucent    median +0.005 [-0.012, +0.033]  mean +0.020  ahead 54%  worst -0.066
    translucent greedy vs greedy       median +0.005 [-0.001, +0.022]  mean -0.012  ahead 54%  worst -0.269
At D 0.05 SIMP is ahead of the plain greedy and level with the translucent one. At D 0.10 it
trailed the translucent greedy (median -0.003). The translucent greedy misses 22422's gate at
small budgets: 0.717 vs 0.986 at D 0.05, and 0.697 behind at D 0.02. SIMP and the plain greedy
find it; that one block is SIMP's mean lead over the translucent greedy. Median time: the
greedies 45 s to D 0.05, SIMP 275 s, SIMP-translucent 333 s (cluster).

**Saturation:** at D 0.05 the sightline median is 0.83, with 5 of 13 blocks above 0.9 (38%; 54% at
D 0.10). At D 0.02 the greedy's median is 0.70, with 2 of 13 above 0.9. D 0.05 already separates
the methods (SIMP vs the greedy: the interval excludes zero), so D 0.02 is not needed for that.
The greedy rows give it for free (every budget in one run) if the 44 want it.
**What follows:** the 44 under sightline at D 0.05: SIMP, the plain greedy and the translucent
greedy. Neither greedy has rows on the 44 yet (5 of them do).

### Sightline on the held-out blocks at D 0.05 (owner: "we can launch the 44-block sightline run at D 0.05", 2026-10-03)

All three methods ran on the cluster (V100 / RTX 8000), so these times can be compared with each
other: SIMP + incumbent (`.k1s1e-06`), the plain greedy S0.01cat and the translucent greedy
(`ss100k2n2r30@ss100k0.5n2r30`). The 46 held-out large blocks are the 44 plus 20023 and 30796, which
fit under the incumbent. 30796's translucent greedy ran out of memory on a 48 GB card (in `_exact`'s
CSR at step 4) and is pending on the H100, so the translucent pairs are on 45 blocks. Paired per block
(simp_compare.py held):

    SIMP vs greedy (46)              median +0.021 [+0.015, +0.026]  mean +0.021  ahead 96%  worst -0.023
    SIMP vs translucent greedy (45)  median +0.005 [+0.001, +0.013]  mean +0.013  ahead 69%  worst -0.023
    translucent vs greedy (45)       median +0.014 [+0.002, +0.022]  mean +0.008  ahead 78%  worst -0.165
    at D 0.02: translucent vs greedy median +0.010 [-0.001, +0.018]  mean -0.002  ahead 64%  worst -0.228

**SIMP's lead holds on blocks it was not tuned on, and is larger.** Against the plain greedy it is
+0.021 (+0.008 on the 13). Against the translucent greedy it is +0.005 with an interval that
excludes zero (level on the 13). The translucent greedy's gain over the plain one is real on the
median, but its tail is the same as on the 13: it misses a gate on some blocks (18910 -0.165 at
D 0.05, 7662 -0.228 at D 0.02), so its mean gain is small. SIMP has no such tail: its worst block
is -0.023.

Medians: SIMP 0.746, the translucent greedy 0.741, the plain greedy 0.720. Median time: SIMP 103 s,
either greedy 33 s to D 0.05, so SIMP costs 3.1x on the same cards.

**Saturation:** 15% of the 46 are above 0.9 for SIMP and 11% for the greedies, against 38% of the
13 tuning blocks, which are richer in gated pockets. D 0.05 leaves room on the held-out set. At
D 0.02 the greedy median is 0.55, with 4% above 0.9.

**What follows:** SIMP + incumbent is the frontier under sightline as under uni. It is on top of
both greedies at 3x their time, and it never collapses on a gate the way the translucent greedy
does. The open cost question is the greedy's speed (BACKLOG, Greedy: speed). 30796's translucent
greedy ran on 2026-10-04 after a memory fix (next section).

### Translucent greedy on 30796: an AMG hierarchy outlived its system (owner: "make it fit in 48GB", 2026-10-04)

The translucent greedy ran out of memory on 30796 (48M unknowns) on 48 GB cards, in the scoring
solve's CSR build at step 4, with cupy's pool holding 45.4 GiB. The H100 was booked for days, so
the question was whether the work really needs more than 48 GB. memprobe_greedy.py measured the
pool per greedy phase, with a high-water mark and the bytes held by each long-lived holder.

**Cause.** `GpuAMG.prepare`'s `cycle` and `krylov2` were closures that call each other. That is a
reference cycle, so a system's levels (2.6 GiB on 30796) stayed alive after the system until
Python's cycle collector happened to run.
- Live memory before each scoring solve climbed step by step: 12.0, 14.6, 17.2, 19.8, 22.4 GiB.
- The scoring peak reached 42.8 GiB.
- The pool's split chunks did the rest: at the failure, 27.9 GiB was live and 45.4 GiB held.
- The plain greedy fitted with little margin (peak 38 GiB).
- CUDA's stream-ordered allocator (`MemoryAsyncPool`) was not a fix: it took the whole card, and
  thrust's sort then failed.

**Fix.** The K-cycle is now a class (`_KCycle`), so a system's levels are freed with the system.
- With `gc.collect()` before each phase, standing in for the fix: live memory before scoring stayed
  at 9.3 GiB, the scoring peak was 31.2 GiB, and at least 11.9 GiB stayed free.
- With the fix, a system's bytes return to the pool on `del` with the collector off (1.98 GiB
  before, 2.79 during, 1.98 after).
- Results are bit-identical: on 30848, translucent, 15 steps, every D, perm, perm1 and clearing
  order is equal. The time is the same (131 s against 132 s).
- 30796's translucent greedy then ran to D 0.15 on the local 48 GB card in 410 s.

**With 30796 in**, the held-out sightline comparison, all 46 blocks:

    D 0.05  SIMP vs translucent greedy   median +0.005 [+0.002, +0.012]  mean +0.013  ahead 70%
    D 0.05  translucent vs plain greedy  median +0.015 [+0.002, +0.023]  mean +0.008  ahead 78%
    D 0.02  translucent vs plain greedy  median +0.010 [+0.000, +0.018]  mean -0.002  ahead 65%

This is the same as on 45 blocks.

**What follows.** The leak was in every GPU solve, SIMP's too, so on the biggest blocks the peaks
were about 11 GiB above what the work needs. That brings few new blocks within reach, though. At
about 0.65 GiB per million unknowns, only three of the 23 large blocks left out are near 48 GB:
- 14401: 1.36 km^2, 43M unknowns;
- 7851: 1.75 km^2, 56M unknowns;
- 32841: 2.01 km^2, 64M unknowns.

A handful at 100 -- 125M unknowns might fit the 80 GB H100. The rest run from 130M to 1.66
billion unknowns (4 -- 52 km^2) and stay out of reach at h 0.5, so the coarser-grid-or-crop
question stays open for them.

**Coverage screen (2026-10-04).** The leak fix alone was not enough on the screened blocks:
- On the cluster, 14401 (43M unknowns, on a 32 GB card) and 32841 (64M, 48 GB) still ran out of
  memory, both in a new system's CSR build. The pool held about 15% more than was live.
- 7851 (56M) fitted.

A new `System` now first hands cupy's cached free blocks back (`Solver.release`), so its CSR
sort's temporaries no longer split earlier systems' freed chunks. With that, all three fit 48 GB
for both methods:
- the greedy to D 0.15 in 388 / 439 / 497 s;
- SIMP `.k1s1e-06` at D 0.05 in 5 -- 7 min.

The release cannot change a result: it frees only memory nothing uses. Checked on the cluster
(`releaseprobe.py`): on 30848, the translucent greedy's 15 steps are bit-identical with and
without it (D, perm, perm1, clearing order).

The large set is 62 of 82 blocks now. At uni, Lens A, D 0.05, SIMP minus greedy:

    14401  greedy 0.386  SIMP 0.393  +0.007
    7851   greedy 0.111  SIMP 0.128  +0.016
    32841  greedy 0.305  SIMP 0.277  -0.028

This is within the held-out spread. Also fixed: clear.py, relax.py and search.py now exit non-zero
when a block fails. They used to catch the error, print FAILED and exit 0, so the launcher counted
an out-of-memory block as passed.

### Greedy speed (2026-10-04)

The greedy's step on 30848 (uni, V100) takes 8 -- 12 s: the ranking 3 -- 4.5 s, the gram 1 -- 7 s
and the scoring solve 2 -- 6 s.

**Scoring tolerance.** Loosening the scoring solve's rtol from 1e-5 to 1e-3 moves the reported perm
by at most 3.5e-5, and usually by 2e-6. It cannot change a pick, since `Spread.pick` ignores the
scoring J. It saves 0.3 -- 1.5 s a step.

**Gram.** gramprobe.py shows the gram's cost is the catchment sweep's runs, one per 32-candidate
chunk at about 0.9 s on 12M unknowns, so it grows linearly in candidates:

    30848  12.2M unknowns  2.7 -- 4.0K waves  prepare 0.10 -- 0.14 s  ~0.029 s/candidate  5.2 -- 6.7 s/step
    9712   15.7M unknowns  7.3 -- 7.6K waves  prepare 0.25 s          ~0.027 s/candidate  1.4 -- 1.9 s/step

Kahn's wave pass is negligible, so wave launches are not the cost. Step-to-step swings of 0.8 /
4.3 s seen earlier on 9712 came from the local card, which another job had started using. The
cluster run is steady.

**What follows.** The greedy can get perhaps 20 -- 30% faster:
- the scoring tolerance;
- a sweep kernel that reads each node's neighbour indices once per chunk rather than once per
  column;
- fewer candidates, which changes the greedy.

This is not the 3x that separates it from SIMP under sightline, so SIMP's speed items come next.

### SIMP: every 2nd candidate, coarse-to-fine, damped OC, MMA (owner: "We can try SIMP .k2 and coarse-to-fine and damped OC/MMA", 2026-10-04)

Each was screened against `fw0.q3.i10.t0.001.e0.0001.k1s1e-06` (uni, J_2, Lens A at D 0.05) on
the sentinel (22422 and 30848 gated, 5810 biggest, 9712 ordinary). The survivors also ran on the
other 9 tuning blocks. relax.py's plan grammar gained `.c<h>` (coarse-to-fine) and
`.u<rule>`, the update rule as a Strategy:
- `ocETAmMOVE`: OC with exponent eta and move limit (the classic 0.5 / 0.2 is unnamed);
- `mmaMOVE`: MMA with a per-building lower asymptote. J falls as x grows, so only the lower
  asymptote matters, and the update is L + (x - L) sqrt(B / lam), OC with L fixed at 0. The gap
  starts at x, shrinks x0.7 where a building's moves reverse and grows x1.2 where they agree.

    sentinel        22422   30848   5810    9712    time vs base
    base            0.7766  0.6541  0.2584  0.1297  1.0
    .k2s1e-06       0.0115  0.6541  0.2584  0.1297  0.88
    .c1             0.7756  0.3993  0.2575  0.1293  0.75
    .uoc0.3m0.2     0.7783  0.6526  0.2572  0.1291  1.0
    .uoc0.5m0.1     0.7766  0.6533  0.2583  0.1296  1.03
    .umma0.2        0.1836  0.6532  0.2583  0.1296  1.03

- **`.k2` (every 2nd iterate rounded and scored): dead.** It collapses on 22422. The gate there is
  held by an odd-numbered iterate only, and the other iterates' roundings score 0.01. So the
  incumbent is not a speed knob: SIMP's capture of a gate can hang on a single iterate.
- **Coarse-to-fine `.c1`:** stages q 1.5 -- 2.5 on the h 1.0 grid (4x fewer cells), q 3 at h 0.5
  from that x, with the incumbent scoring at h 0.5 throughout. It is the fastest variant (0.75x),
  but it collapses on 30848 (-0.255): the iterates that reach the fine grid have lost the gate,
  which h 1.0 presumably does not resolve. It stays selectable as the cheapest operating point.
  - What would have to be true to try it again: a coarse h that keeps 30848's gate (0.75?), or
    only the first stage coarse.
- **Damped OC: no gain.**
  - oc0.3 takes smaller steps (changes 0.06 -- 0.13 against 0.1 -- 0.2).
  - Move 0.1 keeps hitting its own limit, so the flip-flop is capped, not cured.
  - Neither converges within 10 updates a stage.
  - On the 13:

        .uoc0.3m0.2 - base   median -0.0005  mean -0.0018  ahead 2/13  worst -0.0166 (1558)
        .uoc0.5m0.1 - base   median -0.0001  mean -0.0021  ahead 3/13  worst -0.0117 (38616)

  - The incumbent already takes the best rounding of the oscillating iterates, so a calmer path
    proposes nothing better. Not run on the 44: there is nothing to confirm. (Times of about 0.9x
    are cross-card noise: the baseline ran on other cards at another time.)
- **MMA: collapses on 22422 (0.18).** Its gaps grew more than they shrank (steps of 0.16 -- 0.19
  by q 2.5 to 3), so it amplified the motion instead of damping it, and the iterate that holds
  22422's gate never came. It stays selectable (it ties OC elsewhere), but there is no reason to
  run it: damping, its other use, did not help either.

**What follows.** Two of the three collapses come from one fragile mechanism: on a gated block,
SIMP's answer is one lucky iterate's rounding (k2 and MMA each lost 22422 by perturbing the
iterate sequence). A method that keeps the gate on purpose would be sturdier than tuning the path:
- multiple starts (BACKLOG);
- several roundings per iterate, not one, e.g. the top-x rounding plus a randomized one;
- the add/remove search seeded with SIMP's incumbent.

### SIMP multi-rounding `.r<n>` (owner: "Yes, try multi-rounding", 2026-10-05)

The incumbent rounded each kept iterate once, by decreasing x. `.r<n>` adds n randomized roundings
per kept iterate (relax.py `round_sampled`): SIMP's live buildings (x above 2 XMIN) in a
Plackett-Luce order drawn proportional to x, the buildings at the floor after them as round_by_x
takes them, cleared while they fit; each distinct candidate is scored in the eps world (`s1e-06`),
the winner exactly. SIMP's path is unchanged -- only the incumbent sees more candidates -- so `.r<n>`
cannot lose to its base plan beyond the eps-world pick (-0.000 below). Seed fixed (SAMPLE_SEED).
First run on the uv env (cluster_submit Uv, cupy and pyamg from the lock); the base plan
reproduced its earlier numbers to the 4th decimal.

Sentinel, Lens A at D 0.05, J_2, uni (times: same session, same card types):

    sentinel             22422   30848   5810    9712    time vs base
    base .k1s1e-06       0.7766  0.6541  0.2584  0.1297  1.0
    .r2                  0.7766  0.6757  0.2584  0.1297  2.1
    .r4                  0.7766  0.6798  0.2584  0.1297  3.6
    .k2s1e-06.r4         0.5476  0.6845  0.2584  0.1297  2.2   (.k2 alone: 22422 0.0115)
    .r4.umma0.2          0.5506  0.6803  0.2583  0.1296  3.5   (MMA alone: 22422 0.1836)

- Random roundings find better clearings than any iterate's top-x one on 30848 (+0.022 to +0.030).
- They partly rescue the gate `.k2` and MMA lose on 22422 (0.01 -> 0.55, 0.18 -> 0.55), not to
  0.78: still collapses by the screen's rule.
- Survivor: `.r2` only (`.r4` is the best on 30848 but 3.6x the time, over the 3x rule).

`.r2` on the 13 tuning blocks (simp_compare.py tuning):

    r2 vs base [13]: median +0.000 [-0.000,+0.009]  mean +0.008  ahead 69%  worst -0.000  time x2.2

    38616 +0.0384 (84 buildings changed)   1558 +0.0307 (49)   30848 +0.0216 (113)
    20543 +0.0091 (16)   46841 +0.0077 (25)   the other 8 identical

Gains on 5 of 13, three of them about the size of SIMP's whole lead over the greedy on the
held-out blocks (median +0.021). The 8 unchanged blocks pay the 2.2x too.

**Held out (the 46): small.**

    r2 vs base [46]: median +0.000 [-0.000,+0.000]  mean +0.001  ahead 50%  worst -0.000  time x2.0

No block gains more than 0.01; 7 gain 0.001 -- 0.009 (8480 +0.0092, 30796 +0.0086, 24240 +0.0082,
43547, 20080, 44775, 5706); 39 within 0.001; the sum over 46 is +0.040. The tuning set's gains came
from its makeup: it was chosen for its gated collapses, and multi-rounding pays where SIMP's answer
hangs on a rounding, which on a typical block it does not. At 2x everywhere, not a default; never
behind, so `.r2` stays selectable, the hard-block operating point. What would make it a default: a
trigger that samples only where a rounding is in doubt (see the provenance run).

**Where `.r2`'s winners come from** (run mr-r2-where, the 5 tuning blocks it gained on; relax.py
now logs each new incumbent's iterate, rounding and greyness; all 5 reproduced their scores):

    38616  iterate 40 sample 1 (grey 0.049)   1558   iterate 31 sample 1 (grey 0.064)
    30848  iterate 35 sample 1 (grey 0.054)   20543  iterate 40 sample 2 (grey 0.046)
    46841  iterate 26 sample 1 (grey 0.074)        (of 40: q 1.5, 2, 2.5, 3 x 10 updates)

Every final winner is a random rounding from the last two stages, where 93 -- 95% of the
buildings are decided; before it, a chain of improvements through q 2.5 and 3, mostly from
samples. So the top-x rounding of a late, nearly binary iterate picks a worse subset of its few
undecided buildings than a reordering does -- the gate and its substitutes sit in that set. Not
"sample while grey" (the early grey iterates gave nothing that lasted). What follows:
- **sample only in the last two stages** (half the sampling, ~1.5x instead of 2x);
- **polish the undecided set**: from the incumbent, swap each still-grey building in or out,
  scored exactly (50 -- 100 solves, a few updates' worth) -- a targeted case of the add/remove
  search (BACKLOG).

### Late sampling `.r2l2` and the undecided-set polish `.p64w8` (owner: "Yes", 2026-10-05)

Built from where `.r2`'s winners came from. `.r<n>l<L>`: the random roundings only in the last L
stages. `.p<tries>w<width>`: after SIMP, exchange refinement of the incumbent on the undecided
buildings (grey in the final iterate, or where the incumbent and its top-x rounding differ):
each round ranks every add (budget left permitting) and swap within that set by its linearized
change from the eps-world gradient at the 0/1 incumbent, scores the best `width`, keeps the best
if it improves; stops when a round improves nothing or after `tries` scorings. Neither can score
below its base plan, so the screen is gain against time: the 13 tuning blocks, base re-run in
the same session for the times (runs pl-*):

    13 tuning, Lens A D 0.05   mean gain  ahead  median time
    .r2 (earlier session)       +0.0083    5/13   ~2.2x
    .r2l2                       +0.0085    5/13   1.58x
    .p64w8                      +0.0060    7/13   1.19x
    .r2l2.p64w8                 +0.0106    8/13   1.85x

    block   r2l2     p64      both     (polish: undecided / moves kept / scorings)
    1558    +0.0297  +0.0424  +0.0379  88 / 3 / 32
    38616   +0.0417  +0.0261  +0.0478  123 / 8 / 64 (cap)
    30848   +0.0256  +0.0009  +0.0285  257 / 2 / 24
    46841   +0.0065  +0.0030  +0.0097  60 / 1 / 16
    20543   +0.0075  0        +0.0075  71 / 0 / 8
    22422   0        +0.0039  +0.0039  146 / 8 / 64 (cap)
    18895, 5810: polish +0.0015, +0.0005; the other 5 unchanged by all three

- Late sampling keeps `.r2`'s gains at 1.58x instead of ~2.2x: the early stages' samples bought
  nothing (the provenance said so).
- The polish is cheap where it finds nothing (one round of 8 scorings: 1.05 -- 1.13x) and finds
  different gains: 1558 more than sampling, 22422 (which sampling never moved), little on 30848
  (where sampling finds +0.026). It hit its cap on 22422 and 38616 while still improving.
- Together they are the best (+0.0106, 8 of 13), the gains roughly adding.
- All three are on the gain-time frontier of the 13.

**Held out (the 46):** the polish holds, sampling does not.

    p64 vs base [46]:  median +0.001 [+0.001,+0.004]  mean +0.003  ahead 83%  worst +0.000  time x1.3
    both vs base [46]: median +0.001 [+0.001,+0.003]  mean +0.003  ahead 80%  worst +0.000  time x1.8
    r2 vs base [46]:   median +0.000 [-0.000,+0.000]  mean +0.001  ahead 50%  worst -0.000  time x2.0

    summed gain over the 46: p64 +0.153 (2 blocks above 0.01: 5706 +0.0196, 24240 +0.0151; 23 at
    0.001 -- 0.01), both +0.134, r2 +0.040

(Times against the base rows of an earlier session.) The polish finds four times `.r2`'s gain on
typical blocks in less time; sampling on top adds nothing there (the combination's sum is lower:
sampling moves the incumbent the polish starts from). Presets: `.p64w8` the new SIMP default
(+0.003 mean, never behind, 1.3x), `.r2l2.p64w8` for hard blocks (best on the 13, +0.0106).
`.r2` (every stage) is beaten by `.r2l2` on time at equal quality on the 13 and by the polish on
the 46; it is one parameter value of `.r`, nothing to delete.

### Sequential stopping replayed (BACKLOG "Sequential stopping", 2026-10-06)

seqscreen.py: the posterior on a screen's mean paired difference after each block (seeded random
orders), by the Bayesian bootstrap -- the differences are mostly exact zeros (the variant left the
clearing alone) plus a skewed tail; a parametric Student-t pinned the mean at 0 and called the
never-behind `.r2` behind after 3 blocks. For a variant that cannot score below its base the
question is not "> 0" (the first positive block settles it) but "> tau", the smallest gain worth
its extra time (the owner's call). Replayed on the held-out screens at tau 0.002, 20 orders each:

    minimum  r2 (+0.0009)                 p64 (+0.0033)                both (+0.0029)
    10       median 12 blocks, right      median 10, 2 of 20 wrong     median 17, 2 wrong
    15       median 15, right             median 15, 1 wrong           median 20, right
    20       median 20, right             median 20, right             median 21, right

At a 20-block minimum no replay mis-decides and the screens stop at about 20 of 46 blocks: a
held-out confirmation can run in two halves, the second only if the first does not decide.

### Polish cap 256, and coarse-to-fine at h 0.75 (2026-10-06)

`.p256w8` on the 13 (same-session base pl-base): the cap of 64 bound only where the polish was
still improving -- 38616 +0.0261 -> +0.0373 (13 moves, 112 scorings), 22422 +0.0039 -> +0.0043
(10, 88); the other 11 stop by themselves, identical. Mean +0.0060 -> +0.0069, median time 1.19x
-> 1.23x: the cap costs only where it pays. **`.p256w8` replaces `.p64w8` as the default.**

`.c0.75` (every stage but the last at h 0.75) on the sentinel: 30848 -0.0009 (`.c1` lost its gate,
-0.255), 9712 -0.0006, 22422 -0.0005, 5810 -0.0006, at 0.82 -- 0.90x the time. h 0.75 keeps the
gate h 1.0 loses; a cheaper operating point at a small cost. Next: `.p256w8.c0.75` (does the
polish recover it?) and the pair moves `.p256w8x2`, both on the 13.

### Greedy + polish, the every-building polish, and the methods' portfolio (2026-10-06)

polish_greedy.py: S0.01cat to D 0.05, its clearing within the budget, then the polish over every
building (P64w8). SIMP `.P64w8`: SIMP's polish over every building instead of the undecided set.
On the 13, against SIMP base (same session, pl-base):

    method                    mean vs SIMP  ahead  median time
    greedy (own clearing)     -0.0129       4/13   0.40x
    greedy + polish           -0.0039       4/13   0.81x
    SIMP .p256w8              +0.0069       7/13   1.23x
    SIMP .P64w8 (every bldg)  +0.0068       7/13   1.21x

- The polish closes ~70% of the greedy's gap (23597: -0.092 -> -0.008), still cheaper than SIMP.
- On the gated blocks the greedy beats SIMP outright: 1558 +0.049, 38616 +0.040, 30848 +0.018.
- The two polishes win on different blocks: every building 1558 +0.052, 20543 +0.014, 46841 +0.014;
  the undecided set 38616 +0.037 (every building +0.003).
- Per-block best (a portfolio): greedy+polish or p256 +0.0095; p256 or P64 +0.0096; all three
  +0.0113. Built to collect it in one run: greedy-seeded SIMP (`.g<picker>`: the greedy's
  clearing offered to SIMP's incumbent, the polish from the better) and the two-phase polish
  (`.pP`); queued on the 13.

Held out (the 46; greedy + polish ran on the local RTX 6000 Ada, so its times are not the
cluster's): greedy + polish vs SIMP base median -0.013 [-0.018,-0.008], mean -0.015, ahead 7%; the
polish over the greedy's own clearing +0.0031 mean, ahead 65%, ~40% of its time. On typical blocks
SIMP keeps its lead; greedy + polish is the cheap point of the frontier, not SIMP's replacement.

`.p256w8` held out (the 46; local RTX 6000 Ada, so times are not the cluster's): vs base median
+0.001 [+0.001,+0.004], mean +0.003, ahead 91%, worst -0.000; vs `.p64w8` identical (the cap binds
only on hard blocks). Confirmed as the default.

### Where resolution matters (res-check, the 59, 2026-10-06)

resolution_check.py scored five fixed clearings per block (SIMP base, `.r2`, `.p64w8`,
`.r2l2.p64w8`, the greedy's own within D 0.05) exactly at h 0.5, 0.75 and 1.0 (h 0.5 recomputed
equals the stored scores to 3e-9). Score at h minus score at h 0.5, over all 295 clearings:

    h      median   mean     |diff| > 0.02  > 0.05   blocks with every clearing within 0.01
    0.75   -0.0005  -0.0052  22%            3%       35 / 59
    1.0    -0.0029  -0.0192  26%            15%      28 / 59

A coarser grid understates permeability, and on gated blocks by a lot: at h 1.0 38616 -0.313,
45876 -0.209, 22267 -0.142, 1558 -0.114, 41806 -0.100 (h 0.75: 0.011, 0.075, 0.081, 0.043, 0.036).
The clearings' order mostly survives (Kendall tau median 1.00 at both; the argmax moves on ~30% of
blocks, mostly among near-tied variants). For the owner's call on the over-1.4 km^2 blocks: Lens A
at h 0.75 would read ~0.005 low on average and up to ~0.08 low on gated blocks; at h 1.0 far worse.

### Multi-start on coarse grids (the owner's question: several coarse runs vs one fine run)

On the 13, each with the fine polish `.p64w8` (same-session base pl-base):

    plan                         mean vs base  worst            median time
    fine, one start (.p64w8)     +0.0060       +0.0000          1.19x
    h 1.0, 4 starts (.C1.m4)     -0.0216       -0.254 (30848)   2.06x   (12 of 13; 1558 OOM)
    h 0.75, 2 starts (.C0.75.m2) +0.0028       -0.0028          1.36x

No: not faster and not better. h 1.0 loses 30848's gate even with four starts and a fine polish (a
gate below the cell has no value in the coarse objective: res-check), and coarse runs cost more
than their cell counts suggest (four h 1.0 runs ~2x one fine). h 0.75 with two starts is beaten by
one fine run with the polish. The coarse lead left is h 0.75 coarse-to-fine (`.c0.75`, 0.85x,
-0.0006) with the polish: queued. (1558 ran out of 32 GB in `.C`: the coarse phase's GPU blocks are
now released before the fine finish.) Fine two-start `.m2`: running.

Fine two-start `.p64w8.m2` on the 13: identical to one start on 12 (the seeded random start never
beat the uniform one), 46841 +0.0015 instead of +0.0030, at 2.12x instead of 1.19x. SIMP from the
uniform start is robust here; extra starts, fine or coarse, are dominated.

Polish pair moves (`x2`) after the greedy, the 46 held out (local): P256w8x2 vs P64w8 mean +0.0005,
median 0, ahead 30%, never behind (largest 43708 +0.0056, 41673 +0.0051); over the greedy's own
clearing +0.0036 against +0.0031; the polish's median time 14 s against 9 s (28 scorings against
16). Cap and pairs together; on typical blocks the cap rarely binds. A marginal point: a little
more for ~1.5x the polish. SIMP's `.p256w8x2` on the 13: queued.

### The polish's combinations on the 13 (2026-10-06)

Against the same base (pl-base; the runs are a day apart, so times carry cross-session noise):

    plan                            mean vs base  ahead  worst    median time
    .p256w8 (default)               +0.0069       7/13   +0.0000  1.23x
    .p256w8x2 (pair moves)          +0.0097       10/13  +0.0000  1.26x
    .pP256w8 (two-phase)            +0.0095       8/13   +0.0000  1.27x
    .p256w8.gS0.01cat (seeded)      +0.0100       7/13   +0.0000  1.57x
    .pP256w8.gS0.01cat (both)       +0.0111       8/13   +0.0000  1.42x
    .p256w8.c0.75                   +0.0042       6/13   -0.0016  1.23x
    greedy + polish (P64w8)         -0.0039       4/13   -0.0357  0.81x
    greedy w4 + polish (P256w8)     -0.0033       4/13   -0.0341  0.88x

- Pair moves help SIMP far more than the greedy (there +0.0005): 1558 +0.0511, 20543 +0.0140,
  46841 +0.0126, 22422 +0.0063, at ~no extra time -- the blocks the every-building phase reached,
  so `.p256w8x2` beats `.p256w8` and `.pP256w8` on the 13.
- Greedy seeding collects the greedy's gated wins (30848 +0.0222, 38616 +0.0445, 1558 +0.0506);
  with the two-phase polish the best mean, +0.0111.
- `.c0.75` with the polish is beaten by the polish alone at the same time: the coarse-to-fine lead
  closes. The greedy's screening width adds +0.0006 for +0.07x: marginal.
Next: pairs + seeding (`.p256w8x2.gS0.01cat`, `.pP256w8x2.gS0.01cat`) on the 13; `.p256w8x2` on
the 46; the local GPU runs `.pP256w8.gS0.01cat` on the 46.

Held out (the 46, local): `.pP256w8.gS0.01cat` vs `.p256w8` mean -0.000, ahead 9%, worst -0.012
(5706), at 1.4x. On typical blocks the combination adds nothing, and seeding can lose: on 5706 the
greedy's clearing scored better than SIMP's incumbent before the polish, became the start, and
polished to a worse end (SIMP's incumbent polishes +0.0196 there). Hence `.G<picker>`: both
polished, the better kept. `.p256w8x2.GS0.01cat` on the 13 (cluster) and the 46 (local).

Pairs on the held out, and pairs + seeding on the 13 (2026-10-06):

    the 46       .p256w8x2 vs base: median +0.002 [+0.001,+0.004]  mean +0.004  ahead 89%  worst +0.000  time x1.4
    the 13       .p256w8x2         +0.0097  10/13  1.26x
                 .p256w8x2.gS0.01cat  +0.0114  10/13  1.59x   (pairs' 22422, 46841, 9717 + the greedy's 30848, 38616)
                 .pP256w8x2.gS0.01cat +0.0114  10/13  1.99x   (the two-phase adds nothing over pairs: dominated)
                 .pP256w8.gS0.01cat   +0.0111   8/13  1.42x

**`.p256w8x2` is the new default**: never below `.p256w8`, a little above on both sets -- the 13
+0.0028 mean; held out, paired with `.p256w8`, only +0.0002 mean (sum +0.009, largest 0.0026) --
for little more time. For hard blocks, pairs + the greedy; seeding can lose
(5706), so the two-track `.p256w8x2.GS0.01cat` is running on the 13 and the 46.

### Two polished tracks, and their gate (2026-10-06)

`.p256w8x2.GS0.01cat`: SIMP's incumbent and the greedy's clearing each polished (pairs), the better
kept. The 13: +0.0119, 10/13 ahead, never behind -- on every block the better of pairs alone and
pairs + seeding (SIMP's track 1558, 46841; the greedy's 30848, 38616) -- but 2.68x. The 46 (local):
identical to `.p256w8x2` on every block (5706 included): the greedy's track never wins on a typical
block, and nothing is lost the way seeding lost 5706.

Gate, replayed on those logs: polish the greedy's track only if its raw (unpolished) clearing is
within e% of SIMP's polished one. e 0: polished on 2 of the 13 -- exactly the two it won -- and 0 of
the 46; e 1%: 3 and 0; no win missed at either. `.G<picker>e<pct>` built; `.p256w8x2.GS0.01cate1`
and `.p256w8x2` running on the 13 in one session for its time (quality as replayed: +0.0119).
Its floor cost on a typical block is the greedy itself (~0.4x).

Gated two tracks `.p256w8x2.GS0.01cate1`, the 13, same session as `.p256w8x2` (t-x2, t-x2gate):
+0.0119 (= ungated: the greedy's track polished on 3 of 13, as replayed), median time 1.30x
`.p256w8x2` (2.68x ungated). That run's greedy still scored every step; t-x2gate2 repeats it with
the greedy's unused scoring skipped (560a152). Queued: `.r2l2.p256w8x2.GS0.01cate1` (does late
sampling add on top?) and `.p256w16x2` (wider polish rounds).

With the greedy's unused per-step scoring skipped (t-x2gate2): the same scores on all 13, the
greedy's median 41 s -> 26 s, the hard-block preset 1.30x -> 1.21x the default `.p256w8x2`.

The last parameters, the 13 (t-w16, t-all): wider polish rounds `.p256w16x2` equal `.p256w8x2` on
the mean (+0.0097; ahead on 2, behind on 3) at 1.28x its time; late sampling on top of the
hard-block preset `.r2l2.p256w8x2.GS0.01cate1` +0.0120 against +0.0119 (30848, 38616 up; 1558,
20543, 46841 down -- sampling moves the incumbent the polish starts from) at 1.33x. Both dominated.

### The largest blocks on the H100 (2026-10-06)

Under the default `.p256w8x2` (first results; they never fit a 48 GB card): 19593 (4,365
buildings) 0.0955 in 217 s, 6498 (1,559) 0.7987 in 242 s. 53556 failed in a minute: pyamg's
aggregation left an isolated unknown (no off-diagonal entry) in no aggregate, which the solver's
assertion refused; lifted.aggregate_level now gives such a node its own aggregate (with none,
pyamg's aggregation exactly) and logs it; 53556 resubmitted (big3-r1). 45267 is estimated at
86.6 GB, past the 80 GB card: with the other 16 over 1.4 km^2, the owner's coarse-or-crop call.

### The oversized blocks: a coarser grid, and where local coarsening would pay (owner 2026-10-07: "We can try a coarser grid. Is local coarsening (in areas with fewer parcels, like a KDTree) possible?")

The 17 blocks past every card at h 0.5 (45267 joins at 86.6 GB) are peri-urban: 1,000 -- 4,500
buildings over 3.9 -- 52 km^2. Their share of area within d of a building (union of footprints
buffered by d, inside the boundary):

    d                      5 m      10 m     20 m     40 m
    the ~4 km^2 blocks     8-17%    10-23%   14-30%   20-38%
    the 12 -- 52 km^2      1-8%     2-12%    3-17%    5-22%    (4287, 472, 1182, 63818, 64070: 3-4% at 20 m)

So nearly all of each block is open land far from any building, where the field is smooth; the
resolution matters near buildings (gates: res-check). Local coarsening -- h 0.5 within ~20 m of a
building, coarse cells beyond -- would keep the metric's resolution where it decides the score
(4287: from 1.66 billion unknowns to ~50 -- 80 million). It is a new discretization (lifted.py's
grid is uniform: 8 headings per cell, lattice moves such as (2, 1), sightline scans along grid
lines): the fine/coarse interface needs its own stencils and conductances; validated against
uniform h 0.5 on the 59 that fit. Owner's call on building it.

Meanwhile the uniform coarse grid (`.h<h>`, the launcher's `--h`), the finest h each block fits on
a 48 GB card (4287 needs the 80 GB one at h 2), base and the default `.p256w8x2`:
h 0.75: 45267, 62385, 38142, 26061, 62403; h 1: 62523, 38190, 28051; h 1.5: 7499, 62803, 62441,
63612; h 2: 64070, 63818, 472, 1182, 4287. And res-check at h 1.5 and 2 on the 59 (how low those
read). 53556's rerun (the AMG fix) was cancelled while the H100 stayed booked -- the launcher's
budget chained every later run behind it -- and goes back when the H100 frees.

**2026-10-07 -- 08: the first coarse runs were lost to a bad node.** quadro2's CUDA does not
initialise (cudaErrorUnknown in seconds; cluster_submit e83caa8 marked it unusable on 2026-10-07
after a mycooc run). Idle with 8 free GPUs, it took 83 of this run's tasks -- res-check2's 55 and
every oversized-block task but one -- because the research branch still pinned cluster_submit
5e76c39; now 950eeed (924b730). The 48 and 80 GB cards (quadro1, h100) are held by other users'
2 -- 7 day jobs, so the oversized blocks are re-planned for the 32 GB cards, each at the finest h
under 28 GB: h 1 (45267, 62385, 38142, 26061, 62403), 1.25 (62523, 38190, 28051), 1.5 (7499), 2
(62803, 62441, 63612), 2.5 (64070), 3 (63818, 472, 1182), 4 (4287); base and `.p256w8x2`. res-check
at h 1.25 -- 4 on 58 of the 59 (30796 is sized at h 0.5 by the launcher, 34 GB: a 48 GB card).
The four res-check2 blocks that ran: h 1.5 and 2 read 0.05 -- 0.27 low (to be confirmed on 58).
cluster_submit runs the runs one after another (its GPU budget chains each behind the live ones),
so a one-task run leaves three GPUs idle: a packing improvement for the package's backlog.

### What a coarser grid does to Lens A, and the oversized blocks' first results (2026-10-08)

res-check on the 59 at every spacing used (h 1.25 -- 4 on 58: 30796 needs a 48 GB card), five
fixed clearings per block, score at h minus score at h 0.5 (mesh_bias.py; the order compared at
1e-8, since identical clearings score alike only to ~1e-12 on the GPU):

    h      median   mean     worst   |d|>0.05  blocks within 0.01  Kendall tau (median)  order kept
    0.75   -0.0005  -0.0052  -0.081   3%       35/59               1.00                  66%
    1      -0.0029  -0.0192  -0.313  15%       28/59               1.00                  66%
    1.25   -0.0190  -0.0380  -0.204  33%       13/58               1.00                  55%
    1.5    -0.0401  -0.0675  -0.382  44%       14/58               0.76                  40%
    2      -0.0599  -0.0881  -0.303  61%        6/58               0.60                  33%
    2.5    -0.1088  -0.1286  -0.624  71%        1/58               0.53                  28%
    3      -0.1503  -0.1626  -0.607  80%        0/58               0.23                  17%
    4      -0.2065  -0.2047  -0.544  89%        0/58               0.20                  19%

A coarse grid reads low and reorders the clearings: already at h 0.75 a third of the blocks swap
some pair (pairs up to 0.038 apart at h 0.5), and from h 2 most do. (Measured on these urban blocks; the sparse
peri-urban ones may suffer less near their scattered buildings, which this cannot show.)

The 17 oversized blocks on the 32 GB cards (each at the finest h under 28 GB; base and
`.p256w8x2`, the default never below the base on its own grid):

    h     blocks: base -> default
    1     26061 0.3031 +0.0002, 38142 0.0933 +0.0001, 45267 0.3974 +0.0023, 62385 0.5652 +0.0004,
          62403 0.0834 +0.0003
    1.25  28051 0.2226 +0.0187, 38190 0.3367 +0.0013, 62523 0.1191 +0.0000
    1.5   7499 0.2029 +0.0046
    2     62441 0.1958 +0.0003, 62803 0.2737 0, 63612 0.5646 +0.0330
    2.5   64070 0.2431 +0.0008
    3     1182 0.1170 0, 472 0.4397 0, 63818 0.1731 +0.0073
    4     4287 0.1857 +0.0012
    times 2 -- 5 min base, 1.2 -- 1.8x for the default.

The h 1 blocks are close to what h 0.5 would say; from h 2 on the numbers are indicative only.
That is the case for local coarsening (h 0.5 near buildings, which covers 3 -- 30% of these
blocks within 20 m): it would score them at the metric's resolution. Owner's call.

### Local coarsening: the prototype (owner 2026-10-08: "Yes, build that")

The operator's along weight w = m_k/|v_k|^2 is scale-free (the continuum's h^2 cancels: a lattice
step of h|v| for a strip of width h/|v|), so a coarse cell reuses it on its own lattice; turning
edges scale with each cell's own area (s^2). The fine/coarse interface, prototyped in
composite_proto/proto.py from a uniform fine Grid (validation only, not memory-lean):

- every cell steps along each heading to the cell containing its target point (crossed cells
  open, the min open fraction scaling the weight, as on the uniform grid);
- same size: the usual edge (forward only); a LARGER target: an edge with the flux-consistent
  weight m_k (s/|v|) / d -- the source's strip width over the projected distance between centres
  (= m_k/|v|^2 at equal sizes) -- taken in both directions from the smaller side; a smaller target:
  no edge (the smaller cells' own steps reach the larger one);
- the mesh is fixed from the baseline geometry; a clearing changes only fine cells' open fractions
  (every building is in the fine region), and the fine cells are those open in the current field.

Block 6310 (dense; coarse tiles only where open, far from buildings and streets), Lens A at D 0.05
for the five res-check clearings against exact uniform h 0.5:

    two levels                     P0 vs uniform   Lens A diff (5 clearings)
    r 4 (2 m), D 10 m              +0.04%          -0.0001
    r 4 (2 m), D 5 m               +0.12%          (P0 only)
    r 4 (2 m), D 2 m (stress)      +0.52%          -0.0012 .. -0.0013
    r 8 (4 m), D 5 m               +0.09%          -0.0003 .. -0.0004
    r 16 (8 m), D 5 m              +0.25%          -0.0009 .. -0.0010

Against a uniform h 1 (median -0.003, worst -0.31) or h 2 (-0.06): the coarse-grid bias essentially
gone, even jumping straight from 0.5 m to 8 m. Design for the real build:
- levels 0.5, 1, 2, 4, 8 m by distance to the nearest building, street or block edge: 0.5 within D0
  (10 m), doubling with each doubling of the distance; built top-down, subdividing only what needs
  it; sub-sampled geometry (inside, footprint, ground, building labels) only for 0.5 m cells, per
  tile on the uniform grid's own lattice (so the two agree where both exist); coarse cells fully
  inside and open by construction; dist_b by nearest building cell (= the EDT);
- the grid becomes a mesh (flat cells, each with its size): the uniform Grid stays bit-identical,
  a CompositeGrid gives the same attributes over flat cells plus its own pattern (point location
  by level), components (graph), cell areas and sub-sample positions; the sightline
  along-conductances refuse it (they scan grid lines); `uni` only;
- chosen upstream: a plan suffix for the mesh, built where the Clearing is.
Validation before use: the composite against exact h 0.5 on the 59 (fixed clearings, as res-check)
and SIMP on a few; then the 17 oversized blocks at h 0.5 near their buildings.

### Local coarsening: built (2026-10-08)

`lifted.CompositeGrid`, chosen by a mesh Strategy (`lifted.MeshSpec`: `UniformMesh(h)`,
`AdaptiveMesh(h, d0, smax)`) resolved where the scorer is built: SIMP plans name it `.a<d0>x<smax>`,
resolution_check.py and cluster.py `--mesh` take tokens `<h>` / `<h>a<d0>x<smax>`. The mesh:
- Grid.of's own lattice and sub-samples, refined top-down from tiles of side smax: a cell of side
  h 2^l stays whole when no footprint, street or block-edge piece is within d0 2^(l-1) of it (and
  its centre is inside the block, so all of it is), else splits in four; the h cells are Grid's
  (kept where a sub-sample is inside), with Grid's footprint labels, ground and dist_b computed on
  them alone -- no raster, so 4287 (51.7 km^2, 207M cells uniform) builds in 82 s and 6 GB;
- the prototype's interface rule on a graded mesh. A step into a much larger cell can land where
  that cell's centre is BEHIND the source along v_k (a (2, 1) step into the corner of a cell 16x
  its size: projected distance < 0, a negative conductance -- the prototype's r 16 row had some);
  doubling the distance per level keeps neighbours within 2x for d0 > ~2.5 m, and the pattern
  refuses a non-positive projected distance rather than build one;
- every consumer takes the cells flat: the pattern carries each edge's fully-open weight (one
  number on the uniform grid), the turning edges each cell's area, and the mesh answers reach
  (Grid: ndimage labels; CompositeGrid: graph components on the host, kept for a repeating free
  set). Sightline along-conductances need a raster and refuse the mesh (`needs_raster`).
Checks: the uniform path is bit-identical to before (6310: geometry, demand, labels, the baseline
and master operators, reach); an all-fine composite reproduces the uniform grid bit for bit; GPU
and CPU agree. On 6310 (dense: the mesh saves only 16% there), Lens A of the five res-check
clearings against exact h 0.5:

    a10x8  -0.0001 (307k cells)    a5x8  -0.0004 (284k)    a2.5x8  -0.0016 .. -0.0018 (256k)

GPU estimates (0.7 GB per million unknowns, cells x 8) on the oversized blocks: fine cells are
66 -- 91% of every mesh (a third of 4287's only for its 51 km street edge), so smax barely matters:

    mesh     range over the 17 (GB)   over 28 GB
    a10x8    18 -- 52                  9 blocks
    a5x8     14 -- 37                  63612 (37), 4287 (31.8)
    a5x32    13 -- 36                  63612 (36), 4287 at 28.4
    a2.5x8   10 -- 27                  none
(uniform h 0.5: 87 -- 1159 GB.) So a5 for 16 blocks (4287 at a5x32), 63612 at a2.5, if the
validation holds: res-check on the 59 at a10x8, a5x8, a5x32, a2.5x8 (run res-mesh), and on the
oversized blocks the convergence in d0 of their coarse runs' clearings (ov_clearings.parquet).

The validation (run res-mesh: res-check on the 59, the five clearings each; mesh_bias.py):

    mesh     cells (median of uniform)  median   mean     worst    blocks within 0.01  order kept
    a10x8    0.81                       -0.0004  -0.0004  -0.0019  59/59               93%
    a5x8     0.68                       -0.0009  -0.0012  -0.0038  59/59               92%
    a5x32    0.68                       -0.0009  -0.0012  -0.0038  59/59               92%
    a2.5x8   0.57                       -0.0025  -0.0031  -0.0083  59/59               92%
    (uniform h 0.75                     -0.0005  -0.0052  -0.081   35/59               66%)

Every clearing of a block reads low by nearly the same amount (the shift's spread within a block:
median 0.0001, at most 0.0010 at a5), so the order holds: the pairs a composite swaps were within
0.00005 at h 0.5 (ties, in effect), where h 0.75 swaps pairs up to 0.038 apart. a5x32 equals a5x8
here (no cell of these urban blocks is 40 m from everything). So the oversized blocks run at a5
(4287 at a5x32, 63612 at a2.5x8), base and default (ov-base-a5, ov-def-a5 and the per-block runs;
16 blocks with 53556, which no longer needs the H100); their coarse runs' clearings rescored at
several d0 (ov-mesh-*) show the convergence on peri-urban blocks themselves.

On the oversized blocks themselves (runs ov-mesh-*: their coarse runs' base and default clearings
rescored at several d0; 4.5M cells at most, 1.7 -- 5.1M), the composite converges in d0, a little
more slowly than on the urban 59: a10 - a5 mean +0.0013 (+0.0004 .. +0.0026, 14 clearings), a2.5
- a5 mean -0.0012 (-0.0044 .. +0.0026, 30). Here the far field counts a little: a5x32 - a5x8 is
-0.0036 .. +0.0012 (472, 1182 the lowest), so 4287 at a5x32 may read up to ~0.004 low. The coarse
runs' own scores were far off the composite's (coarse score minus a5): 45267 at h 1 -0.14 and
28051 at h 1.25 -0.22 (gates below the cell), 472 at h 3 +0.16, 63818 at h 3 -0.14 (base) and
+0.07 (default) -- and on 63818 the coarse grid ranked the default's clearing above the base's
(0.180 vs 0.173) where a5 puts it far below (0.111 vs 0.317): optimizing on the coarse grid found
a worse clearing. The coarse-h numbers in "What a coarser grid does to Lens A" are superseded by
the a5 runs.

### The oversized blocks at the metric's resolution (2026-10-08)

Base and `.p256w8x2` on all 18 (the 17 and 53556) at a5x8, 4287 at a5x32 and 63612 at a2.5x8, on
the 32 GB cards (runs ov-base-a5, ov-def-a5, ov-base/def-4287, ov-base/def-63612). Against them,
the coarse runs' own base and default clearings scored on the same mesh (ov-mesh-*):

    block  mesh    coarse h  base    default  gain     min base/default  coarse clearings (base, default)
    63612  a2.5x8  2         0.7695  0.7706   +0.0011  9.6 / 12.9        0.5929  0.6129
    28051  a5x8    1.25      0.5406  0.5464   +0.0059  5.8 / 8.2         0.4378  0.4356
    1182   a5x8    3         0.1178  0.1178   0        5.5 / 5.8         0.0958  0.0958
    64070  a5x8    2.5       0.2206  0.2260   +0.0054  4.0 / 5.6         0.2063  0.2007
    63818  a5x8    3         0.3292  0.3321   +0.0028  5.4 / 9.8         0.3174  0.1109
    62803  a5x8    2         0.2754  0.2785   +0.0031  5.5 / 12.7        0.2722  0.2711
    62441  a5x8    2         0.1866  0.1866   0        3.5 / 3.5         0.1819  0.1789
    472    a5x8    3         0.2881  0.2881   0        4.6 / 5.7         0.2834  0.2834
    45267  a5x8    1         0.5351  0.5391   +0.0040  4.1 / 5.9         0.5356  0.5351
    26061  a5x8    1         0.2998  0.3040   +0.0042  3.6 / 6.9         0.2993  0.3006
    4287   a5x32   2         0.1369  0.1384   +0.0015  13.4 / 23.2       0.1353  0.1281
    38190  a5x8    1.25      0.2905  0.2905   0        2.4 / 3.1         0.2881  0.2880
    7499   a5x8    1.5       0.1836  0.1872   +0.0036  4.3 / 5.5         0.1850  0.1840
    62403  a5x8    1         0.0804  0.0819   +0.0014  3.1 / 4.8         0.0807  0.0802
    62385  a5x8    1         0.5606  0.5613   +0.0006  4.1 / 5.6         0.5595  0.5601
    62523  a5x8    1.25      0.0960  0.0962   +0.0002  3.3 / 4.3         0.0955  0.0955
    38142  a5x8    1         0.0924  0.0924   0        3.4 / 3.8         0.0922  0.0922
    53556  a5x8    --        0.2242  0.2321   +0.0080  2.3 / 3.5         --

The default is never below the base (+0.0023 mean, +0.0015 median, tied on 5), at 1.0 -- 2.3x
the base's time (1.39x median); base 2.3 -- 13.4 min, default 3.1 -- 23.2 (4287 the slowest).
Optimizing at the metric's resolution beats both coarse clearings on all 17, by +0.0002 to +0.158
(median +0.0035): most where a coarse cell closed a gate (63612 at h 2 +0.158, 28051 at h 1.25
+0.109) or where the coarse grid misled the search (1182 and 63818 at h 3, 64070 at h 2.5: +0.015
-- +0.022). The h 1 runs' clearings were within 0.004 of it. The base alone trails the coarse
base clearing on three blocks by at most 0.0014 (45267, 62403, 7499), a SIMP run's own scatter.
So each of the 82 large blocks now has SIMP at h 0.5 near its buildings: the base on all but
19593 and 6498 (the default only, on the H100), the default on all but 14401, 7851 and 32841 (the
base only). The coarse-h runs are superseded (their clearings kept in ov_clearings.parquet).
Composite cells: 2.4 -- 5.1M, well within a 32 GB card.

The three blocks that had only the base (14401, 32841, 7851: they fit 48 GB at h 0.5, and those
cards are booked) ran base and default at a5x8 on the 32 GB cards (runs three-base-a5,
three-def-a5): 14401 0.3843 -> 0.3893 (+0.0050), 32841 0.2742 -> 0.3070 (+0.0328), 7851 0.1278
both; 2.2 -- 3.4 min base, 2.3 -- 5.7 default. 14401's base at a5 read 0.009 below its base at
h 0.5, so the h 0.5 clearing was scored on the composites too (three_clearings.parquet, runs
three-res, three-res2), Lens A minus its exact h 0.5 score:

    block  exact h 0.5  a20x8    a10x8    a5x8     a2.5x8
    14401  0.3932       -0.0014  -0.0044  -0.0094  -0.0141
    32841  0.2766       -0.0003  -0.0011  -0.0023  -0.0024
    7851   0.1275       -0.0001  -0.0002  -0.0002  -0.0004

So it is the mesh, not the search: 14401 reads low at a5 by 2.5x the 59's worst (-0.0038), the
error falling with d0 at about first order. It shifts the block's clearings alike (their spread
on any composite at most 0.0005), so the order holds and the default's +0.005 is the same at every
d0; only absolute scores on a composite carry the bias. 14401 at a20x8 is 3.6M cells against 5.4M
uniform. On the oversized blocks a10 - a5 was at most +0.0026, so their a5 scores are probably
within ~0.005 of h 0.5, with 14401 a reminder that a block can do worse.

### Faster submits (2026-10-08, owner: "start with faster submits")

cluster.py sized an adaptive mesh's tasks by building each block's CompositeGrid in turn: 14 min
for the 16 oversized blocks, 11.5 min for the 59 (a5x8). 73% of a build was the quadtree's near
test (`query_nearest` with a max distance, per box: is a piece within d0 2^(l-1)?). Measured per
level on 26061, 30796, 1558 (a5 -- a20), against query_nearest:

    near test                                        a5            a10           a20 (dense urban)
    all `dwithin` pairs, any per box                 0.33 -- 0.64  0.44 -- 0.48  1.17 -- 2.35
    `dwithin` at dist/4, the rest at dist            0.34 -- 0.38  0.40          0.67 -- 0.76
    `dwithin` at dist/8, /4, /2, dist                0.45 -- 0.51  0.52          0.58 -- 0.59
    `dwithin` at a small radius, query_nearest rest  0.75 -- 0.85  0.86 -- 0.94  0.79 -- 0.95

All pairs at once blows up where a dense block's top-level boxes (D 160 m at a20) each pair with
hundreds of pieces; staging settles most boxes among few pairs. lifted._near is now the
two-stage form (dist / 4, then dist), the build split into `_layout` (the leaves) and the cells'
attributes, and `CompositeGrid.count` the leaves alone (no footprint raster, street band or
KD-tree); cluster.py counts SIZING_WORKERS = 8 blocks at a time. Identical meshes to the old build
on all 139 checked (the 80 blocks at a5x8, 4287 at a5x32; the 59 at a20x8: level, i0, j0, isub,
bsub, ground, building, dist_b, xy, and the count). Under a 12-process load a full build takes
0.59x the old time at a5 (0.44 -- 0.73), 0.90x at a20 (a few small dense blocks up to 1.4x, a few
seconds each); the count alone 0.38x and 0.71x. Sizing a submit: the 16 oversized blocks 840 s
-> 51 s, the 59 691 s -> 38 s (16 -- 18x). No cache: the count is cheap enough that one would
only add a key to keep in step with the mesh code.

### The greedy and the cheap preset on the composite (2026-10-08)

The greedy under uni has no raster step (its nonlocal term is Uniform's zero vjp), so it runs on a
CompositeGrid as it is. On 6310, an all-fine composite (a100x1: 336,553 cells, every one h 0.5)
reproduces the uniform greedy to D 0.02: the same picks, J within 3e-10 at every step, J0 within
6e-16, the tension within 6e-7 (its solves are at 1e-3); a30x1, with 382 cells of 1 m, the same
picks and J within 1e-5. polish_greedy.py now takes a mesh (a required argument; its suffix in
the rows path, its name in the rows; the 118 existing rows migrated, `h0.5`).

The cheap preset (S0.01cat.P64w8, D 0.05, p 2) at a5x8 on the 59 (run cheap-a5-59) against the
same at uniform h 0.5: on each its own mesh, median -0.0011, mean -0.0016. Its clearings rescored
exactly at h 0.5 (cheap-a5-rescore, 58 blocks; 30796 needs a 48 GB card) split that into the
search and the mesh:

                                              mean     median   range
    search: a5 clearing - uniform clearing   -0.0004   0.0000   -0.0145 .. +0.0028
    mesh: a5's read of its clearing          -0.0012  -0.0009   -0.0038 .. +0.0004

The search on the composite is as good as on the uniform grid (ahead on 16, behind on 22, tied on
20), but for 39240 (-0.0145), where the greedy took another path (its unpolished clearing
-0.0115): the greedy is path-dependent and a slightly different tension can send it elsewhere.
Same time (1.02x): these urban blocks shed few cells.

### Goal-oriented refinement (owner 2026-10-08: prioritized)

14401 read 0.009 low at a5, 2.5x the 59's worst, so distance from features misses some of what
the score depends on. Refining where J_2 is sensitive instead: from the pilot a2.5x8, each round
solves the baseline with the buildings at eps 0.01 (the tension's world) and its J_2 adjoint,
gives each coarse cell eta = E (s / h)^alpha (E its half of |w du dlam| over each of its edges,
s its side), splits the largest ceil(gap / 6) (a split adds 3 cells: half the gap to the target
per round) and 2:1-balances (no neighbour, across an edge or a corner, under half a cell's size;
the distance meshes already are, as their thresholds double per level). Prototype
(goal_proto.py, scratch), the five res-check clearings (14401: its three) against exact h 0.5, at
the distance meshes' cell counts (share of uniform cells):

    block   distance mesh                     goal mesh, alpha 1 (alpha 2)
    40144   a5   0.55  -0.0037                0.54  -0.0013 (-0.0012)
            a10  0.67  -0.0018                0.66  -0.0005 (-0.0005)
            a20  0.83  -0.0007                0.82  -0.0001 (-0.0001)
    20543   a5   0.25  -0.0029                0.25  -0.0016 (-0.0020)
            a10  0.32  -0.0012                0.32  -0.0008 (-0.0011)
            a20  0.43  -0.0005                0.42  -0.0002 (-0.0003)
    14401   a5   0.33  -0.0095                0.33        (-0.0029)
            a10  0.47  -0.0044                0.46        (-0.0004)
            a20  0.65  -0.0013                0.64        (-0.000003)

1.5 -- 400x less error for the same cells, most where distance does worst (14401: the goal mesh at
0.46 beats a20 at 0.65), and a smaller spread between a block's clearings. alpha 1 and 2 within
1.4x; alpha 1 kept. (A first run's balance compared each probe with the wrong cell's level and
over-refined, 6 -- 96% more cells even on the distance meshes; fixed, the distance meshes need no
balancing and 2,000 random splits about 1,000 more cells.)

Built (common.GoalMesh, `<h>g<d0>x<smax>f<factor>p<power>`: the pilot a<d0>x<smax> refined to
`factor` x its cells for J_power; a SIMP plan's `.g...`; lifted.split_cells and balance).
MeshSpec.build now takes the block and the run's physics (Params, population), since a goal mesh
solves on the way; the others ignore them. cells(block) gives the target, so a submit sizes it
without a solve. On 40144 it reproduces the prototype (333,642 cells against 333,639, GPU noise
in the ranking at the cutoff; -0.00128 on every clearing), mesh and five scorings in 19 s.
Validation on the 59 (goal-59, factors 1.2, 1.5 and 2; 38 tasks first failed on a layout refined
to all fine cells, fixed in 860c91b and rerun). Per block, the distance curve (a2.5 .. a20, log
error linear in the cell share) at the goal mesh's share, over the goal mesh's mean |error| on the
five clearings:

    factor  all fine  share (median)  compared  ratio (median)  goal mesh ahead  worst
    1.2      6 / 59   0.68            53        1.7x            34               0.15x
    1.5     16 / 59   0.84            40        3.1x            33               0.29x
    2       38 / 59   1.00             9        2.6x             6               0.36x

Better in the median, but behind the distance meshes on a third of the blocks, down to 6.6x the
error: on 33717, refining from 0.43 to 0.51 of the uniform cells took the error from 0.0018 to
0.0020, where a5 at 0.54 reached 0.0006. The prototype's three blocks were all winners. alpha
was not it (0 and 2 within 15% of 1 on five blocks, losers and winners alike). The indicator's
world was: it solved with the buildings at eps 0.01, nothing cleared, so it never refined where a
clearing opens new flow. The five blocks at f1.2 with the buildings more open in that world:

    block  o0.01    o0.1     o0.3     o1       a5 (its share / the goal mesh's)
    33717  0.00197  0.00125  0.00055  0.00024  0.00059 (0.54 / 0.51)
    20423  0.00135  0.00088  0.00054  0.00034  0.00040 (0.15 / 0.14)
    8152   0.00245  0.00176  0.00108  0.00097  0.00078 (0.55 / 0.51)
    40144  0.00110  0.00097  0.00092  0.00093  0.00370 (0.55 / 0.56)
    22422  0.00129  0.00125  0.00137  0.00136  0.00275 (0.37 / 0.31)

Monotone in the opening on the losers, flat on the winners, the build's time the same. The
world is now part of the mesh, `o<opening>` (GoalMesh.opening, the buildings' openness; goal-59's
rows renamed o0.01), and goal-o1-59 runs the 59 with the buildings fully open:

    factor  compared  ratio (median)  goal mesh ahead  worst  mean |error|  (o0.01)
    1.2     53        2.8x            49               0.26x  0.00050       0.00092
    1.5     40        4.1x            39               0.68x  0.00012       0.00021
    2        9        4.0x             7               0.46x  0.00002       0.00003

At equal error it spends a median 0.11 of the uniform cells fewer than the distance meshes (IQR
0.06 -- 0.16, 45 blocks at f1.2; 0.11 at f1.5). Behind them still on 20952 (0.26x), 44602
(0.27x), 8235 (0.65x) and 9710 (0.93x) at f1.2 and 8152 (0.68x) at f1.5, each less than at
o0.01. The goal mesh is o1 from here.

Why those five lose. Two are the yardstick's: on 20952 and 44602 the distance meshes' bias changes
sign between a2.5 and a10 (20952 a2.5 -0.0011, a5 -0.00002, a10 +0.00013, a20 +0.00005; 44602
a2.5 +0.0005 .. +0.0013, a5 -0.0001 .. +0.0004, a10 -0.0004, a20 -0.0002), so a5 reads near zero
by cancellation and the curve through it is an envelope no mesh earns. The goal mesh there is as
good as the finer distance meshes: 20952 +0.00016 at 0.81 of the uniform cells, about a10's at
0.86; 44602 at f1.5 -0.00003 .. -0.00006 at 0.80, 5 -- 10x better than a10 and a20. 9710 is a tie
(0.93x). Two are real but mild: on 8235 f1.2 spent 12% more cells (0.65 -> 0.77) and left the
error at the pilot's -0.00045 (where the sign crosses too: a5 +0.0002), and on 8152 f1.5 reads
0.0004 at 0.63 where distance gets 0.0003 (0.68x). Nothing to fix in the mesh; a comparison that
scores a mesh by a smoothed bias rather than one mesh's |error| would not count the first two.

The goal mesh on the oversized blocks (runs ovgoal-*-r1, on the SIMP base, default and cheap
clearings of the 19 that run at a5x8). None fits uniform h 0.5 on a 32 GB card, so the reference
is the finest goal mesh that does: f2 on the 12 smaller, f1.5 on the 7 larger (2.0 -- 5.0M
cells). On 14401, the one whose uniform h 0.5 was scored, the reference reads 0.0007 low, against
a5's 0.0095 and the equal-cell goal mesh's 0.0024. Against the reference, at a5's cells:

    mesh             cells (of a5x8's)    mean |error|  per block, a5's error / the goal mesh's
    a5x8             1                    0.00257
    g2.5x8f1.4p2o1   0.98 (0.92 -- 1.07)  0.00041       median 4.9x (0.98 -- 69x), ahead on 18

a5 reads low on all 19 (-0.0001 .. -0.0090); the goal mesh reads -0.0017 .. +0.0002, within
0.0006 on 16 (not 14401, 45267, 62385), and the one tie is 62523 (0.0001 each). Both keep the
reference's order of the clearings (on six blocks base and default are the same clearing). a5
shrinks a gain a little, the goal mesh hardly: default - base moves by up to 0.0010 on a5 (32841,
a gain of 0.034) and by up to 0.0003 on the goal mesh. 4287 (a5x32 against g2.5x32f1.5, 5.1 and
5.0M cells) and 63612 (a2.5x8 against g2.5x32f1.1, 4.8 and 5.1M) have no finer mesh that fits:
there the goal mesh reads +0.0027 and +0.0043 above the distance mesh on every clearing, on the
side where the distance meshes' bias puts the truth. So the oversized blocks' composite scores
("The oversized blocks at the metric's resolution") read low by about a5's 0.003 (14401 0.009)
and their gains by at most 0.001; the goal mesh at the same cells leaves a sixth of that.

The cheap preset on the 21 oversized and base-only blocks, on the SIMP runs' meshes (cheap-ov-*:
a5x8, 4287 a5x32, 63612 a2.5x8), against SIMP base and the default on the same: cheap - default
median -0.0107 (-0.0594 .. +0.0054), cheap - base median -0.0067 (-0.0588 .. +0.0258), at 0.50x
the default's time and 0.72x the base's (median). As on the 46 held out (-0.015 against the
default), a cheaper point, not a better one: it wins only on 28051 (+0.0054 over the default) and
ties on 1182; its worst is 62385 (-0.059, a gate the SIMP presets find). Every large block now
has the default and the cheap preset but 19593 and 6498 (the default only, uniform h 0.5).

### Sightline on the composite (owner 2026-10-08: prioritized)

The sightline metric and the translucent search scan lattice lines, so they raised on a
composite. An along-conductance now takes the grid, and a grid maps its fields to and from the
fine lattice's raster: Grid by the identity (layers and vjp bit for bit as before), CompositeGrid
by painting each cell over its pixels and giving a cell the mean of the raster's factor over
them, with exact adjoints (paint and gather, mean and its transpose: 1e-16). An all-fine
composite reproduces the uniform grid's layers exactly and its vjp to 3e-16. On 40144 under
ss100k2n2r30 the five res-check clearings read, against uniform h 0.5, a5 -0.0012 .. -0.0016,
a20 -0.0004 .. -0.0005 and g2.5x8f1.5p2o1 -0.0005 .. -0.0007: as under uni.

The raster is the cost: the oversized blocks' lattices run to 260M pixels (472), where the scans'
thirteen float64 temporaries took the translucent greedy on a5x8 to 43.9 GiB. The scans now keep
five (fb no longer stores the transmittances and the vjp recomputes them, bit for bit on the CPU;
g and the vjp's weight fused and in place on the GPU): 30.0 GiB, inside a 32 GB card, the steps
as fast (38 and 21 s against 39 and 23 s). On 38190 (67M pixels) the translucent greedy's steps
take 10 -- 14 s against uni's 7 s. `cluster.py --scans` sizes for the rasters: what a run holds
between solves (0.6 GB per million unknowns) plus 40 bytes a pixel, or the solves' own estimate
if larger (472 on a5x8: 30.1).

Validation under the sightline metric (ss-res-59 and ss-res-59-r1, ss100k2n2r30, the five
clearings per block), error against uniform h 0.5 on the 59:

    mesh             cells (median share)  median   p95 |error|  worst
    a5x8             0.68                  -0.0005  0.0053       +0.0082 (7662)
    a10x8            0.81                  -0.0001  0.0018       +0.0028 (33717)
    a20x8            0.91                  -0.0000  0.0006       -0.0012 (20423)
    g2.5x8f1.5p2o1   0.84                  -0.0000  0.0018       +0.0026 (33717)

As under uni, the error falls with the share of the cells. The goal mesh's indicator solves in
the run's conductance (System applies p.along), so it is goal-oriented here too, but it gains
less: per block against the distance curve (a5 .. a20, log error linear in the share) at its
share, 40 compared (16 all fine, 3 outside a5 .. a20's range), median 1.9x (4.1x under uni at
f1.5), ahead on 38, worst 0.18x (8152) and 0.45x (22640); at equal error it spends a median 0.05
of the uniform cells fewer (IQR 0.02 -- 0.08, 24 blocks) against 0.11 under uni. 8152, a mild
loser under uni (0.68x), is a clear one here: the goal mesh reads up to 0.0018 off at 0.63 of the
cells, where a10 at 0.69 reads 0.0002. Why it gains less under the sightline conductance, and
8152, are not looked at. Every mesh now runs under either metric.

### The add/remove search as one Strategy (owner 2026-10-08: prioritized)

The greedy (pickers M, B and S), exchange (the polish), grow-then-prune and SIMP's polish were
four loops; they are now configurations of one `Round` in search.py. A round ranks the buildings
by a first-order estimate (the add or restore tension, or the eps world's gradient), builds
candidate moves from the ranking (singles, a spaced or diverse batch, swaps, a restore batch),
optionally after re-ranking a shortlist in a costlier world, then scores some in a world and
accepts one. A schedule (Do, Until, Seq, With, Rescore) runs rounds under stops, and a record
says which states the rows need scored exactly. Spec:
docs/superpowers/specs/2026-10-08-add-remove-search-design.md.

Agreement: every preset (the four pickers, polish_greedy P64w8x2, GP3xS0.01catr0.005m8, SIMP
.k1s1e-06.p256w8x2.GS0.01cate1) on 19421, 19510 and 38138 on the CPU, row for row against fresh
runs of the old code, identical within the CPU's own noise at every step of the work and after
the final review's fixes. (Two runs of the old code differ by pyamg's randomly seeded
spectral-radius estimate, so the check takes discrete columns exactly, floats to 1e-7, arrays
to 1e-4; three injected faults were each reported.) The pair tier, Spread's short batches and
the Screen's reordering fire on every check block. cut_to_budget's skip never does (a building
there is large against the batch gap), so it is covered through the pick order, which the greedy
rows pin. GPU memory (memprobe_greedy, 20543, translucent): 18.14 GB peak, old and new.

Floating search, new (`FL<grow>x<picker>r<step>[m<k>]c<cap>`): grow-then-prune where each
restore is followed by the greedy's add round from there, kept only if its exact J beats the
best archived at its budget level (Pudil's rule: an archive per level, every exact state within
d_max offered to it). It repeats until an add is refused, D passes d_max or `cap` scorings are
spent. FL3xS0.01catr0.005m8c100 on grow-then-prune's five blocks, d_max 0.15, the GPU, against
GP3xS0.01catr0.005m8 and the greedy S0.01cat (Lens A, best row at or below each budget; the
ranges over D 0.025, 0.05 .. 0.15):

    block  n     at D 0.05: FL  GP     greedy  FL - GP          FL - greedy      FL time  / GP's
    19421   112             0.472  0.433  0.464   0 .. +0.039     0 .. +0.041     2 min    1.40x
    19510    88             0.345  0.237  0.268   0 .. +0.224     0 .. +0.078     3 min    2.22x
    19537   247             0.799  0.799  0.716   0 .. +0.009  +0.008 .. +0.254   8 min    1.31x
    9712   1721             0.121  0.119  0.107   0 .. +0.002  +0.006 .. +0.014  26 min    1.12x
    22422  1977             0.782  0.011  0.534  +0.757 .. +0.828  +0.006 .. +0.269  100 min  1.64x

Never below grow-then-prune or the greedy at any budget (two zeros are -2e-8 and -2e-14: the same
clearing, solver noise). It repairs the prune's collapse on gated blocks: on 22422, where
restoring substitutes one at a time shut the gate's pocket, grow-then-prune reads 0.011 at D 0.05
and floating search 0.782. Against SIMP at D 0.05 (rows on 9712 and 22422 only): 9712 0.121
against base 0.130 and default 0.130, 22422 0.782 against 0.777 and 0.783. SIMP gives one budget
in 1.5 -- 15 min; floating search the whole curve to 0.15 in one run. At a single budget it is
behind or tied with SIMP on the two blocks with SIMP rows.

The two further rows BACKLOG names (a substitute-aware restore, screened adds) are not one
configuration each, as the spec first said: each needs two parts (spec, "What the two new rows
need"). search.py passes `mypy --strict` (the command is in its docstring), so pairing a ranking
with a builder that expects another kind of ranking is a type error.

### Stopping by the value of more blocks (owner 2026-10-09: the kernelcore model dropped, the stop built here, tau and price derived)

The kernelcore outcome model (a spec in GeoffChurch/kernelcore, dropped in cb5595a) would have
valued more blocks by reweighting Dirichlet draws over the observed differences plus a grid on
[-1, 1]. Reviewed on the held-out screens (r2, p64 and both against base), it failed where it
mattered. Reweighting collapses with the horizon even with no grid: the mean effective share is
0.15 at 4 blocks, 0.03 -- 0.07 at 8 and 0.006 at 16, under kernelcore's 0.1 guard, and the
estimates ran 1.5 -- 3x high. One to four more blocks are worth exactly 0 under the bootstrap
(they cannot carry the mean across tau), so a next-horizon stop stops at once: 4 -- 8 of 20
orders wrong on both. And the grid's alpha, not the data, sets the stop: the exact posterior sd
of mu at 20 blocks rises 2 -- 5x at alpha 0.001 and about 50x at 1, moving the stop from about
19 blocks to 42 to never. Collapses stay with the sentinel's veto.

seqscreen.py stops by the value of more blocks instead. The decision is b over a if mu > tau,
tau = rate x b's extra GPU-seconds per block, `rate` the Lens A a GPU-second is worth. The loss,
min(E(tau - mu)+, E(mu - tau)+), is linear in mu, so the value of k more blocks is exact through
the posterior mean after them, under the bootstrap a Polya urn on the observed atoms (against
enumeration at k 1 -- 3: within 1%; an urn that ignores the counts is off by up to 2x). The
screen stops when no k is worth rate / future x b's predicted GPU-seconds on the next k blocks
(a's times b's time ratio so far), `future` the block runs the decision governs; at once on a
collapse (> 0.1 below a); never before 15 blocks. It replaces the probability rule (P(mu > tau)
outside [0.05, 0.95] from 20 blocks).

rate from the decisions on record, b against a on the same blocks:

    decision                                       gain      extra GPU-s  per GPU-s
    .p256w8x2 over .p64w8, the 46: adopted         +0.00019  10.4         1.9e-5  (rate below)
    late sampling on the hard-block preset, the    +0.0001   ~76 (1.33x)  1.3e-6  (rate above;
      13: closed                                                                   3.9e-6 by the
                                                                                   rows' times)

So rate lies in about [2e-6, 1.9e-5], 6e-6 in the middle: one GPU-minute more per block must
buy 0.00036 Lens A (0.0001 -- 0.0011). tau is then per variant: against base r2 0.00053, both
0.00047, x2 0.00021, p64 0.00014; x2 over p64 0.00006. The old rule's single 0.002 would have
turned down the default itself (x2 over p64, +0.00019). price = rate / future.

Replayed on six held-out screens (r2, p64, x2 and both against base; x2 and both against p64;
none with a block below -0.007), 20 orders each, against the 46 blocks' decision at the same tau.
The realized loss is the regret of a wrong call, |mu - tau|, plus rate / future x the GPU-seconds
spent, per governed block. At rate 6e-6:

    future   blocks (mean)  GPU-h per screen  wrong of 120  loss (1e-6 per screen)
             old    value   old    value      old   value   old    value
    82       26.0   15.4    1.1    0.6        1     6       281    182
    820      26.0   16.6    1.1    0.7        1     4       29.1   26.1
    8200     26.0   18.5    1.1    0.8        1     2       3.9    4.2

The value rule settles the clear screens at the floor (p64, x2, both: right in every order) and
spends where the call is close: r2, 0.00035 over its tau, runs to a median 24 blocks (15 -- 40)
at future 8200. Its wrong calls are those near-ties, 2 of 20 on r2 and on x2 over p64. At rate
1.9e-5, x2 over p64 is a tie (0.000003 apart): the value rule calls it either way at 15 -- 21
blocks (12 of 20 "wrong", no regret), where the old rule ran 44. Over the three rates and five
futures (82 -- 8200) the value rule's loss is the lower in 11 of 15. The old rule's is lower at
large futures and small rates, by at most 1.6x on small losses (2.2 against 3.5 at 2e-6 and
8200), where its five extra blocks pay. The floor: 15 has the lower loss than 20 in 11 of the
15, than 10 in 9; below 15 the bootstrap's 90% interval misses the 46 blocks' mean 12 -- 21% of
the time at 10 blocks. future barely matters on these screens (only r2 moves, 15 -> 24 blocks
from 82 to 8200), so 820, the large blocks about ten times over, is a fair default: price 7.3e-9
per GPU-second per governed block. Caveats: the floor and the rate come from these same screens
and decisions, none has a collapse, and the truth is the 46 blocks' mean, a finite population.

The owner has no intuition for either constant (2026-10-09), so the rate is a bracket, not a
value (seqscreen.RATES, the two calls above; FUTURE 820). A screen's call is "adopt" or "keep"
when it holds at every rate in the bracket and "ask" when it turns on the rate. An ask is a
question in units one can judge, this much Lens A for this much more time, and the answer moves
an end of the bracket to the screen's break-even rate, mu / extra seconds. The stop averages the
value of more blocks and their cost over the bracket (five rates, geometric), the owner's rate
taken as log-uniform: requiring the stop at every rate instead ran r2 to a median 35 blocks
against 21, for the same call. On the six screens (future 820):

    screen         call   blocks (median, range)  GPU-h
    p64 vs base    adopt  15 (15 -- 15)            0.47
    x2 vs base     adopt  15 (15 -- 15)            0.52
    both vs base   adopt  15 (15 -- 15)            0.70
    both vs p64    keep   15 (15 -- 34)            0.71
    r2 vs base     ask    21 (15 -- 40)            0.98   +0.00088 (0.21%) for 2.03x the time
    x2 vs p64      ask    15 (15 -- 21)            0.53   +0.00019 (0.05%) for 1.10x

Neither ask is live: x2 over p64 is the call that set the bracket's top, so it sits on the edge
(at the stop 9 orders say adopt, 9 ask, 2 keep), and r2 against base is moot now that r2 on top
of p64 is a keep at every rate.

### Floating search at the cheap preset's budget: the first live screen (owner 2026-10-09)

seqscreen.py's `next` ran floating search at d_max 0.05 (FL3xS0.01catr0.005m8c100D0.05: the grow
to 0.15, the restore rounds and the conditional adds all for the cheap preset's budget; search.py
now has d_max in its rows' name) against the cheap preset (S0.01cat.P64w8), on the held-out
blocks in the replay's first order, on the local RTX 6000 Ada like the cheap preset's rows.
Stopped by hand after 15 blocks, where the stop said run on (one to 31 more blocks each worth more
than its price at the bracket's mean rate) and the call was an ask:

    floating search against   mean     median   ahead (of 15)
    the cheap preset          +0.0043  +0.0025  10
    SIMP base                 -0.0086  -0.0078   3
    SIMP .p256w8x2 (default)  -0.0109  -0.0092   2

Median time a block: floating search 527 s, the cheap preset 28 s (29x in total; 30796 alone
9000 s against 147 s), SIMP's default about 45 s on the same card (`.p256w8`'s rows, run here,
the same Lens A on these 15, pairs about 2% more time on the 13; the default's own rows ran on
the cluster's older cards, 87 s). At one budget floating search is dominated by SIMP's default,
behind on 13 of 15 at about 10x the time, so the ask
(+0.0043 Lens A, 1.2% of the cheap preset's mean, for 29x its time; break-even 3.7e-6) is moot.
It is behind the cheap preset itself on five (19161 -0.019, 44602 -0.013, three by under 0.003):
the polish finds what its restores and conditional adds do not. Its niche stays the whole curve
to d_max in one run.

The screen compared one pair and would have run on (about 10 GPU-hours) for a decision a third
preset had already made. A candidate is screened against every frontier point it could displace:
the cheapest it would sit above and the one at its cost.

### Swing schedules at the cheap preset's budget (owner 2026-10-09: "add a bunch and drop a bunch, and then add a smaller amount and drop a smaller amount, etc?")

search.py's swings (`SW<grow>-<grow>-...`): the greedy grows to grow x d_max, screened restore
rounds prune back to d_max, the next swing starts from the archive's best (every exact state at
or below d_max, the greedy's ascent included) and the cheap preset's polish (P64w8, from the
archive's best) ends the run. Two arms on the live screen's 15 held-out blocks at d_max 0.05, on
the local RTX 6000 Ada: the owner's shape, SW3-1.5-1.2xS0.01catr0.005m8.P64w8, and a cheap arm,
SW1.5-1.2 (the same otherwise).

    against                   SW1.5-1.2                  SW3-1.5-1.2
                              mean     median   ahead    mean     median   ahead (of 15)
    the cheap preset          +0.0101  +0.0099  14       +0.0106  +0.0077  14
    floating search           +0.0058  +0.0048  12       +0.0062  +0.0048  15
    SIMP base                 -0.0027  -0.0030   5       -0.0023  -0.0038   4
    SIMP .p256w8x2 (default)  -0.0050  -0.0031   3       -0.0046  -0.0049   4
    SW1.5-1.2                                            +0.0004  +0.0003   8

Median time a block: SW1.5-1.2 145 s, SW3-1.5-1.2 398 s, floating search 527 s, the cheap
preset 28 s (1.27, 4.04, 5.03 and 0.17 GPU-hours on the 15); SIMP's default about 45 s on the
same card (as in the section above).
- Swings dominate floating search at one budget: SW1.5-1.2 is +0.006 at 0.3x its time, and
  SW3-1.5-1.2 is ahead on all 15 at 0.9x.
- The large first swing buys nothing here: SW3 against SW1.5, median +0.000 [-0.003, +0.002],
  at 3.0x the time, and its grow to 3x and prune back are about 70% of its run (22659: 115 of
  166 s; 30796: 4650 of 6684 s). Per block the two differ by up to 0.009 (38988 +0.009, 18739
  +0.006, 20423 -0.005); whether that is a block's preferred amplitude or the search's path
  dependence is unknown without repeats. Judging complements together, the reason for a large
  swing, did not show at this budget.
- Both are dominated by SIMP's default: -0.005 mean, behind on 11 or 12 of 15, at 3x (SW1.5)
  and 9x its time. They beat it on 30796 (+0.004, +0.005: the largest block, n 5023, where
  floating search beats it too), 8152 and 8480 (both under +0.001), and 22640 (SW3, +0.001).
- Against the cheap preset they are behind only on 19161 (-0.006, -0.010): the polish starts
  from the archive's lowest J and ends below the greedy's polished clearing, the trap that
  SIMP's `.g` seeding fell into and `.G` (polish both, keep the better) fixed.

The swings close two-thirds of the cheap preset's gap to SIMP's default (+0.010 of +0.015) at
5x the cheap preset's time and 3x SIMP's: not a frontier point between the two. The adaptive
form's lever would be the amplitude, and the fixed arms already say the large one is wasted
here. The CPU smoke run (SW3-1.5-1.2 on two small blocks, 19510 and 19421, D 0.05) gave 0.3455
(floating search's clearing) and 0.488 (floating search 0.472).

### Swings as SIMP's finisher (owner 2026-10-10: "Let's try the 1.2x swing plus the polish as a finisher on SIMP's answer")

search.py runs any schedule from SIMP's clearing (`SIMP<relax plan>+<spec>`): the seed is the
first move, scored exactly and checked against SIMP's own score of it (within 1e-4; all 46
matched). SIMPfw0.q3.i10.t0.001.e0.0001.k1s1e-06.p256w8x2+SW1.2xS0.01catr0.005m8.P64w8 on the
46 held-out blocks at D 0.05, on the local RTX 6000 Ada: from SIMP's default answer the greedy
to 1.2x the budget, restore rounds back to it, then the cheap preset's polish. The seed stays in
the archive, so the finisher never ends below SIMP's answer.

    against SIMP's default, the 46   mean     median           ahead (> 1e-4)  behind
    SIMP's default + the finisher    +0.0010  +0.0000 [0, 0]   19              0

- Half the gain is one block: 30796 (n 5023, the largest) +0.024, 0.402 -> 0.427, above every
  other method there (floating search 0.407, the swings from the greedy 0.406 and 0.408). Then
  43708 +0.0034, 6312 +0.0028, 38988 +0.0020, 41515 +0.0019, and 14 blocks +0.0001 -- +0.0014.
- The swing finds it: the archive's best before the polish holds 95% of the summed gain (the
  polish +0.002 of +0.047).
- It costs SIMP's time again: median 44 s a block (mean 74 s; 0.94 GPU-hours on the 46) against
  `.p256w8`'s 40 s on the same card (1.02x, median per block); 30796 565 s, 24240 389 s.
- On the 15 it is +0.0071 over the swings from the greedy (ahead on 13) at about two-thirds of
  their time with SIMP's run counted; over the cheap preset +0.019, ahead on all 46.

Its gain per GPU-second is 1.4e-5 Lens A (8.0e-6 without 30796; Bayesian bootstrap over blocks,
5 -- 95%: 6.9e-6 -- 2.3e-5): above the screens' rate bracket's bottom (2e-6) with probability
1.0, its middle (6e-6) 0.98, its top (1.9e-5, the rate `.p256w8x2` was adopted at) 0.14. An ask
in seqscreen's terms: worth it at any rate below about 1.4e-5, not at the bracket's top; the
owner's answer narrows the bracket. Adopted, it is a preset above SIMP's default at twice its
time, beside the default rather than replacing it. The frontier at D 0.05 under uni is then the
cheap preset (28 s), SIMP's default (40 s) and SIMP + the finisher (84 s); swings from the
greedy are dominated by SIMP's default (configurations of the code the finisher runs: nothing to
delete).

Untried: the finisher on the 13 tuning blocks, where SIMP's gated answers are fragile (22422,
30848, 38616) and the hard-block preset `.p256w8x2.GS0.01cate1` gains +0.0119, from that
preset's answer too (on the 46 it equals the default's); the swing's size (1.1x, 1.3x) and two
swings (1.2-1.1).

What it changes (owner: "does that +0.001 gain actually look like a different topology, or is
it just swapping some buildings along the same corridors? ... is it normalized properly"):
- On 27 of the 46 blocks, nothing. On 18 it re-picks 2 -- 10 of the 60 -- 180 buildings SIMP
  cleared (2 -- 9% of the budget), each added building mostly tens to hundreds of metres from
  the nearest dropped one, each block opening one to six single-building gaps and closing as
  many: SIMP's layout, a few of its buildings moved (43708, +0.0034: six small buildings in the
  central cluster for four).
- Their gain is worth a median 0.8% more budget (the greedy's local slope of Lens A in D at
  0.05), against about 10% for SIMP's lead over the cheap preset; 0.3% of the block's Lens A.
- 30796 is structural: 39 scattered single buildings dropped (31 isolated openings closed) for
  5, among them a long one and a large one in the industrial east, 17% of the budget moved, the
  gain worth 19% more budget.

Lens A is 1 - (J/J0)^(1/2), each block against itself uncleared, so gains compare across
blocks. The finisher's gain keeps its sign and size on finer grids and shifted ones (below, "The
metric's resolution floor"), though res-check found the best of five near-tied clearings
changing on about 30% of blocks between h 0.5 and the coarser h 0.75.
The case for the finisher is SIMP's occasional large miss, not the typical re-pick.

The 13 tuning blocks, where SIMP's gated answers are fragile (owner: "You can start that up"),
from SIMP's default and from the hard-block preset `.p256w8x2.GS0.01cate1`: on 12 the finisher
gains on one, 20269 +0.0006 (22422 +0.0001), from either start; mean +0.0001, 5e-7 Lens A per
GPU-second, below the screens' bracket. The gated blocks' misses (1558, 38616, 30848) are the
greedy track's to fix, and the finisher finds nothing past it. On all 58 blocks: 1.0e-5 per
GPU-second, 5.6e-6 without 30796. A gate on SIMP's answer would not find 30796 either: by
fragmentation (patches per cleared building) it ranks 42nd of 46; only its size (the largest,
n 5023) sets it apart, one witness. Not a preset: the gain is one block's. It stays a spec
(`SIMP<plan>+SW...`), on the frontier for Lens A at twice SIMP's time.

30848 failed from both starts: the restore round's screen (search.py Screen, the eps world at
EPS_SCREEN 1e-6, RTOL_SCORE) did not converge, AMG-CG at 2000 iterations at 1.9e-5 -- 5.0e-5
against 1e-5, varying run to run; the gate closed at eps 1e-6 leaves a pocket nearly cut off.
Any restore round can meet it on a gated block (grow-then-prune, floating search, swings).

### The metric's resolution floor (owner 2026-10-10: the agenda's first item)

The clearings of the 59 large blocks (the cheap preset, SIMP base, `.p256w8`, SIMP's default,
the finisher; the hard-block preset on the 13 tuning blocks) scored exactly at h 0.35 and 0.25
(resolution_check.py; cluster runs fine25, fine35). h 0.25 covers 51: the seven largest do not
fit 80 GB there, and 38616's solve did not converge (AMG-CG 4.9e-8 after 2000 iterations).

    the same 51 blocks                 mean gap                    per-block |change| 0.5 -> 0.25
                                       h 0.5    h 0.35   h 0.25    median  90%     max
    SIMP's default - the cheap preset  +0.0181  +0.0155  +0.0149   0.0009  0.0094  0.104 (23597)
    SIMP's default - SIMP base         +0.0036  +0.0030  +0.0030   0.0007  0.0047  0.027 (43547)
    the finisher - SIMP's default      +0.0005  +0.0005  +0.0005   0.0000  0.0007  0.0013
    .p256w8 - SIMP's default           -0.0007  -0.0006  -0.0004   0.0000  0.0004  0.012 (46841)

Per block, of the gaps over 1e-3 at h 0.5 the sign flips at h 0.25 on 3 of 51 (SIMP's default
against the cheap preset), 2 of 30 (against SIMP base), 0 of 11 (the finisher), 1 of 7
(`.p256w8`). A block's level moves more than its gaps: SIMP's default's Lens A changed by a
median 0.005 from h 0.5 to 0.25, by over 0.02 on 12 of 51, up to 0.20 (20543 0.50 -> 0.70,
30848 0.65 -> 0.48), and not towards a limit: on 9 of the 18 blocks that moved over 0.005 from
0.5 to 0.35, the step to 0.25 went the other way.

Alignment alone does as much (offset_check.py: h 0.5 at the metric's lattice offset and three
shifts, on 9 of the blocks that moved most). SIMP's default's level ranges over the four offsets
by 0.004 -- 0.157: 20543 0.36 -- 0.51, 38366 0.37 -- 0.47, 17608 0.70 -- 0.77. Its gaps barely
move:

    9 blocks, h 0.5                    mean gap                 per-block range    sign against
                                       metric's   three shifts  median  max        the metric's
    SIMP's default - the cheap preset  +0.0090    +0.0096       0.0052  0.0136     0 of 27
    SIMP's default - SIMP base         +0.0051    +0.0055       0.0019  0.0201     1 of 27
    SIMP base - the cheap preset       +0.0039    +0.0041       0.0057  0.0212     2 of 27
    the finisher - SIMP's default      +0.0005    +0.0003       0.0000  0.0014     0 of 24
    .p256w8 - SIMP's default           -0.0026    -0.0028       0.0000  0.0048     0 of 27

- Paired gaps share the grid, so most of its error cancels. Every mean gap keeps its sign at
  every spacing and offset; its size moves by about 2e-4, or 20%.
- The clearings were optimized at the metric's offset and keep their gaps under the shifts (the
  shifted means are as large): no fitting to the alignment.
- Per block, a re-pick of a few buildings (the finisher, `.p256w8`, the hard-block preset against
  SIMP's default) holds to 0.002 at the 90th percentile; different layouts (SIMP against the
  cheap preset) move by a median 0.005, up to 0.02 under a shift and 0.10 under refinement.
- A block's level carries up to about +-0.08 on some large blocks, and so do per-block absolute
  claims and the share above 0.9; comparisons of levels across blocks need that margin.

No floor added to the screens' tau. Alignment is per-block noise the bootstrap already carries:
a block's lattice sits arbitrarily against its buildings, so the error is in the spread of the
paired differences. Refinement moved mean gaps by about 20% without turning one, which matters
only for a call near tau; tau stays a price (rate x extra GPU-seconds). Per-block claims need a
margin of about 0.01 between different layouts and 0.002 between re-picks. The finisher's
+0.0005 is real in sign and size; the verdict on it stands on its size.

Why a level moves (a hypothesis, untested): J_2 weights the worst-off homes, which reach the
street through gaps between buildings of the order of h; a gap conducts by its sampled open
fraction (4 x 4 sub-samples a cell, the smallest along a step), which the alignment sets. A
clearing rarely opens those gaps, so their share of J is common to J and J0 and dilutes the
ratio by an amount the alignment decides. The metric's passable width is thereby implicit in h
and the sub-sampling; an explicit one would make it a modelling choice (the agenda's third
item). The test is a map of the change in each home's term between two offsets on 20543.

### Visualization: block 5810 (owner 2026-10-10: "The block I most want to see is ZAF.9.3.1_1_5810")

Flow maps of three clearings at D 0.05, published as a private artifact (Block 5810 Flow Atlas):
the cheap preset (202 buildings), SIMP's default (312; every SIMP variant picks this set there,
base differs by 3) and the translucent-search greedy under the sightline metric (S0.01cat,
ss100k2n2r30 @ ss100k0.5n2r30, cut to the budget: 390). Each pane's flow is solved under its
method's own metric (current_map.flow). Each clearing scored under both metrics:

                       cheap preset  SIMP    sightline greedy
    uniform metric     0.223         0.259   0.202
    sightline metric   0.339         0.436   0.440
    shared with SIMP   92            312     144

- SIMP's uniform-metric clearing comes within 0.004 of the sightline greedy under the sightline
  metric, while the sightline greedy is last under the uniform one. One block.
- Under the uniform metric the flow drains to the block edge as a sheet; under the sightline
  metric it gathers into straight lanes running to the north edge, which the sightline greedy
  extends. The flow the uniform clearings redirect still runs in long corridors through existing
  lanes, most visibly where the west part meets the north-east strip.

What an exit is: a kblock's streets are its polygon's boundary, every ring (kblock.py), and every
cell within BAND_M (1 m) of one is ground. Every block's outline is an exit, so blocks are
independent problems: a region of kblocks is its blocks side by side with a pooled budget,
whatever its shape. 5810's one hole (132 m^2, 21 m inside the south edge) is an exit too.
Current OSM maps 6.8 km of residential ways and 2.6 km of paths and footways inside 5810 (its
outline is 5.6 km), which the block data treats as open ground.

Flow relative to a no-buildings prior (owner: "normalize against a prior ... visible corridors
throughout the block"): the same homes' flow to the same edge with every building gone, under
each metric's conductance, divided into each map's flow. Raw flow is not largest at the edge: by
escape-time band its median rises from 0.16 at the exit to 0.97 mid-way and falls to 0.24 in the
deepest band, and the bands explain 15% of the variance of log flow; the prior explains 36%
(uniform metric) and 18% (sightline). The ratio shows a branching corridor network through the
whole block, under the sightline metric most clearly. Where the prior's flow vanishes (the
watershed in the middle) the ratio is large: a knot. No floor (owner: "I'd rather have a knot
than a clamp, unless there's some rigorous justification"); the page shows the ratio two ways,
after Monroe, Colaresi & Quinn 2008 (the Fightin' Words estimator, bookgen's keyword layer):
delta, the log ratio of a cell's share of all walking to its share in the open field (their Eq
16, the counts large against any prior), and z = delta / sqrt(1/y + 1/y') (Eq 18, 21), reading a
cell's flow as the expected number of walkers crossing it (current as expected net crossings of
a random walk, Doyle & Snell). The knot is a ratio of two small counts, so its z is small, with
nothing clamped; the count scale multiplies every z by one constant, so the map does not depend
on it. z also raises the large corridors near the exits again (sqrt of their counts): more
walkers make the same ratio surer.

5810's north edge is a railway: 1.7 km of its 5.6 km outline runs within 15 m of OSM rail with
no OSM road within 15 m. The block data (Million Neighborhoods) bounds blocks by railways as
well as streets, and every outline is an exit, so the lanes draining to that edge drain to a
railway line.

### A region whose only exit is its edge: 5810@major (owner 2026-10-10: "just the boundary of the region would be the sink, so we still wouldn't use roads at all")

regions.py: `<kblock>@major` is the face of the OSM major-road network (motorway to tertiary,
links included) holding the kblock's interior point, every building anchored inside it (the kblock
source's, at any count), its outer ring the only street. 5810@major: 6.74 km^2 (area / convex
hull 0.82), bounded by the N1 to the north and major roads on the other sides; 12,663 buildings
from 77 kblocks (12,700 Open Buildings outlines have a point inside), 5810's 6,619 among them;
5.5M cells at 0.5a5x8 (27M uniform). On an H100 (runs face-cheap, face-simp), D 0.05:

                         Lens A   cleared  in 5810  share of cleared area in 5810  time
    the cheap preset     0.2714   548      157      20%                            170 s
    SIMP's default       0.3014   714      162      17%                            386 s

5810 holds 33% of the region's footprint area. Inside 5810 the region's clearings share 20 of
the cheap preset's 202 and 17 of SIMP's 312 per-block picks: where the exits are decides the
answer more than the method does. Most of the clearing goes to the formal grid in 5810's hook,
whose streets were exits per block and are open ground here, so its homes count as deep; the
flow the clearings add runs south through it to the diagonal road. 5810's informal west part
drains west, south and north, across the railway (open ground here, an exit per block) into the
vacant land and on to the N1. So with the region's edge as the only exit, a formal home on a
street counts as far from access as an informal one deep in a settlement, and a railway or a
freeway is ground to walk across or an exit to walk to. Published as a private artifact (5810
Region Flow Atlas), with the flow against the no-buildings prior as on the 5810 page.

### FFT homogenization (owner 2026-10-10: "look into FFT homogenization. If we could eventually run this on e.g. all of Cape Town that would be amazing")

A feasibility study, fft_homog/ (report.md, the scripts and tables; the fields, 288 MB, stayed in
the session's scratch). Homogenized: a scalar 'uni', one potential per cell, a face between
4-neighbours conducting min(o_a, o_b) (the uni rule for an axis step), the metric's ground and
demand: the lifted model's fast-turning neighbour. At 5810's baseline it ranks homes like lifted
uni (Spearman 0.997, lifted ~0.80 x scalar), but its Lens A runs 0.04 -- 0.08 lower on 5810's
clearings and 0.16 lower on 22422's gate.

- Schemes: the staggered grid is exactly the 5-point finite volumes; CG preconditioned by the FFT
  Laplacian converges at infinite contrast, 40 -- 150 iterations, 3 -- 12x faster than the basic
  scheme, the augmented Lagrangian or Eyre--Milton, within 1.2e-10 of a sparse direct solve on 20
  windows, 26 ms per 50 m window per core. Moulinec--Suquet's original collocation is unstable
  with pores; undamped Eyre--Milton is not guaranteed to converge.
- Fabric: dense informal fabric conducts 0.33 -- 0.55 as well as open ground (the densest 200 m
  window 0.33). Anisotropy median 1.27 on 12.5 m tiles, 1.10 on 100 m; the major axis follows the
  lanes (22 deg from the lane axis, 44 deg with the lane axes shuffled). No strict representative
  volume: the spread between tiles falls only 17% from 50 to 100 m; nested windows settle to
  +-10% by 50 -- 100 m.
- J_2 on 5810, two-scale against the fine solve of the same scalar model: -20% to +20% by window
  size and the treatment of windows at the exit, best +0.4% (200 m). The interior converges by
  100 m windows; what remains is a boundary layer at the exits (homes within 10 m read +27 to +97%,
  but carry under 1% of J_2). Correctors move J_2 by under 0.3%.
- Clearings (5810's, optimized under lifted uni): 50 m windows land within 0.007 of the fine
  scalar Lens A, but SIMP minus the greedy at D 0.05 falls from 0.0074 to 0.0009 (D 0.10: 0.0103
  to 0.0058).
- Gates: on 22422 two-scale J_0 is 51% low (20 pocket homes hold 83% of it), and opening the gate
  scores 0.15 against 0.37 fine. Homogenization erases gates by construction.
- Screening: over 164 tiles of 50 m and two clearing rules, two-scale (or its first-order adjoint
  map) ranks tiles by fine gain with Spearman 0.98 / 0.99 and 94% of the top decile; a depth proxy
  scores 0.04 / 0.22. The gain is where the macro flow concentrates, not where homes are deepest.
- Cost: the fine scalar solve of 5810 takes 12 s and 1.4 GB on one core (built-up Cape Town ~5
  CPU-h), lifted uni 277 s and 11.7 GB (~110 CPU-h); a city tensor field at 50 m windows, 25 m
  stride ~16 CPU-h. Under per-block exits the median built-up block is 0.017 km^2: nothing to
  homogenize.

Verdict: not a design model (it erases gates, compresses the differences the optimizers compete
on, and the fine solve is already cheap). Possibly a screening layer, for exits at region
boundaries or a city-wide pooled budget, with a separate gate detector. The deciding experiment is
5810@major: fine scalar against two-scale (100 m windows) on J_2, on the two region clearings'
Lens A and on the 50 m tile ranking; keep it if the tile Spearman stays >= 0.9 and an optimizer
restricted to the map's top areas keeps >= 90% of Lens A.

### The sightline metric's angle error (owner 2026-10-10: "Sightline greedy has some visible artifacts from angular discretization")

The 5810 flow under ss100k2n2r30 has axis-aligned streaks (flux_before_ss on the atlas page). A W x
40 m corridor rotated 0 -- 45 deg (checks.channel: demand at one end, street at the other; K 8,
ell 3, h 0.5), P's max / min over the angles:

    W      uni    sightline  uniform rays  turning boosted  both
    1 m    1.66   5.32       2.51          5.08             2.33
    1.5 m  1.33   2.40       2.26          1.73             1.58
    2 m    1.15   1.67       1.79          1.26             1.26
    4 m    1.06   1.17       1.31          1.13             1.14

sightline_angle.py. Uniform rays: 96 directions, each scanned along the rows of its own rotated
copy of the raster (bilinear there and back), the same hat weights. Turning boosted: the turning
edge (x, k)-(x, k+1) times min(F_k(x), F_k+1(x)). Also measured at 1 m: lattice directions to
nmax 12 / 24, 2.67 / 2.14; K 16, 5.30 (1.39 at 2 m); h 0.25, 3.54. Two causes:
- The ray directions, for 1 m lanes. The 48 lattice directions are uneven (9.5 deg gaps beside
  the axes, 1.1 deg near 45 deg), and a lane's long rays lie within ~W / L of its axis (1.4 deg at
  1 m x 40 m): aliasing in angle. Uniform rays halve the error.
- The walker's headings, for wider lanes. The boost multiplies the along edges (13 -- 27 in a
  40 m lane 1 -- 4 m wide) and not the turning edges, so a walker in a lane goes ~ell sqrt(a),
  11 -- 16 m, per radian of turning: it is ballistic. A lane between two of the 8 headings is
  walked by zig-zagging, each turn the bottleneck (both headings carry the lane's boost: 13.4 and
  13.3 at 2 m and 15 deg); the error peaks at 15 and 35 deg, midway between headings, and
  resolving it would take headings ~W / (ell sqrt a) apart (~10 deg at 2 m; K 8's are 18 -- 27
  deg apart). It is radiative transfer's "ray effect" of discrete ordinates. Boosting the turning
  edges by the same factor (the boost then scales the whole (x, theta) metric, and the
  persistence stays ell) brings 2 -- 4 m lanes to within 1.1x of uni (1.5 m: 1.3x, 1.2x with
  uniform rays as well).
Together the fixes leave the error the uniform metric has, a lane of two cells, which only h
removes.

The owner's alternatives:
- Fourier in angle (P_N): exactly rotation-invariant, but a ballistic lane's angular profile is a
  spike, which a truncated series rings around (negative conductances: no walker, no Rayleigh
  monotonicity) unless it keeps about as many terms as there would be headings. The visibility has
  no transform at all: a free path is a product of transmittances along a line (it solves
  e . grad F = kappa c F - 1 along each direction), nonlinear in the buildings.
- Linear reduction: exact in the fast-turning limit (ell -> 0), where the heading integrates out
  and only the boost's angular harmonics 0 and 2 survive, a 2 x 2 conductivity tensor per cell
  (one unknown instead of K; what FFT homogenization produces). The boost is what leaves that
  limit in a lane, so it is a different model. PCA over the cells' angular profiles returns the
  Fourier basis when the directions are spread evenly (the covariance is then circulant).
- Clustering (per-cell headings fitted to the lanes, like fixels in diffusion MRI): the
  discretization would change with each clearing, so P would jump instead of falling
  monotonically: no gradient for SIMP, no tension for the greedy.

Not done: the turning boost in the gradient (relax.py and the vjp), uniform rays on the GPU and
the composite mesh, and re-scoring; both fixes change the sightline metric's numbers, and the
turning boost changes the model (walkers no longer go ballistic in a lane; the straight-run reward
stays, in the per-heading boost).

### FFT homogenization on 5810@major: the deciding test (owner 2026-10-10: "Sure, let's try it on 5810")

fft_homog/region_report.md, the `region_*` scripts and tables, region_checks/ (the fields, ~4 GB,
stayed in the session's scratch; steps 1 -- 3 of the report regenerate them in ~15 min). The
fine scalar 'uni' on the whole region at h 0.5 (25.4M unknowns, 252 s and 14.4 GB on one core)
against two-scale (macro H 2 m, readout U). The stored lifted scores reproduce to 6 digits on
the local GPU (0.271361, 0.301406).

- J_2 at baseline: 100 m windows +2.2% (inside-avg), +2.8% (extrapolated), -2.3% (open-outside);
  200 m inside-avg -1.8%; 50 m +9.6 to +12.1%, growing with depth as on 5810. At 100 m the homes
  deeper than 100 m (97.8% of J_2) are within a median +0.7 to +1.9%; the exit's boundary layer
  (homes within 10 m read +56%) carries ~0% of J_2. Per-home Spearman 0.998 -- 0.999.
- The two clearings (D 0.05): the order cheap < SIMP survives every variant, the gap does not.
  SIMP minus the cheap preset: lifted 0.030, fine scalar 0.018, two-scale 50 m 0.009 -- 0.010,
  100 m 0.014 -- 0.016. Two-scale at 100 m reads 0.015 -- 0.026 below fine (a local opening
  diluted over a larger window), at 50 m within 0.012.
- Tile ranking, 160 stratified 50 m tiles per rule (R20, STRIP) against fine local re-solves:
  the two-scale re-solve at 50 or 100 m and the first-order map at 100 m reach Spearman 0.975 --
  0.989, raw and per displaced m^2 (every 95% interval >= 0.958), 12 -- 14 of the fine top 16.
  The first-order map at 50 m, the 5810 study's, fails per m^2 on STRIP: 0.882 [0.823, 0.936].
  Large buildings nearly block 78 of its 68,267 windows (sigma < 0.01; none below 0.04 at 100
  m), and there the linear prediction overshoots the fine gain 51 -- 205x. A naive proxy: 0.66 --
  0.80.
- Restriction: tiles ranked by predicted gain per displaced m^2 until they hold k x the budget's
  footprint. Truncating the stored clearings to them keeps at most 88.8% of Lens A (k 5, the 50 m
  re-solve; 50 -- 85% with the 50 m map): the truncations underspend (D 0.017 -- 0.040). Topped
  up to D 0.05 inside the same tiles by the macro adjoint's sensitivity (a one-pass restricted
  optimizer, no re-solves), k 5 keeps 90.5 -- 94.6% with the 100 m map and 91.3 -- 94.9% with
  the 50 m re-solve; k 2 -- 3 keeps 66 -- 88.5%. k 5 is ~113 of the 2,518 tiles, 25% of the
  region's footprint. The optimizers put 4 -- 9% of their cleared footprint in tiles crossing the
  edge, which no tile set here holds (all 2,518 inside tiles: 95.8% / 93.6%).
- Cost: 14.5 CPU-h and 21 GPU-min in all. The 100 m tensor field 99 s on 11 workers, the
  first-order map over every tile ~0.7 CPU-h, a fine re-solve per tile 59 s (warm start, exact
  solve within 20 m of the change around the baseline's V-cycle). A restricted optimizer still
  solves the whole region at every score, so it is barely faster: screening pays for placing a
  pooled budget across faces, not inside one.
- Cape Town's major-road faces (region_capetown_faces.py): 689 built up (>= 5 buildings/ha, >= 10
  buildings), 718 km^2, 1.33M buildings, median 0.60 km^2, 9 at least 5810@major's size, the
  largest 26 km^2. By area (x106.5): the fine scalar baselines ~7.5 CPU-h, the 100 m field ~32
  CPU-h, one lifted score per face ~4 GPU-h with setup. Our extrapolation, unmeasured: if time
  scales with area, the stored runs (170 s cheap, 386 s SIMP on an H100) come to ~5 and ~11
  H100-h for all of them; by a score's memory (34.8 GiB at 6.74 km^2) only 2 faces (26 and 16
  km^2, 35k buildings) exceed an 80 GB card at a5x8.

Verdict against the rule (tile Spearman >= 0.9, a restricted optimizer >= 90% of Lens A): met at
k 5 with the first-order map at 100 m windows, not with the 50 m map the 5810 study used, and not
below k 5 by either bound. Whether a real restricted optimizer keeps 90% at k 2 -- 3 is open. To
find out: the cheap preset needs no search change (`SearchState.movable` from a tile mask, ~20
lines in polish_greedy.py), SIMP ~40 -- 60 lines in relax.py (a per-building upper bound through
the OC and MMA steps, the rounding and the polish); ~1 GPU-h for k 2, 3, 5 and both methods. Its
use is a city-wide pooled budget across faces, with the gate detector the 5810 study asked for
(homogenization still erases gates). The fine model itself looks within reach of built-up Cape
Town without it, on the extrapolation above.
