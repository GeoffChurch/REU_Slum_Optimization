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

Demand: each building injects 1 over the free cells within 1 m of it that it is nearest to and
that reach the street at baseline, spread over axes by angular share; FIXED at the original
footprints (rehoused in place), so only the operator changes. Ground: cells within 1 m of
Block.streets. A home with no such cell, but free cells within 1 m of its own outline that reach
the street (the nearest-building rule gave them to a neighbour), injects over those, shared; the
rest are ENCLOSED: they inject nothing, are out of J, and a clearing reports the population it
leaves enclosed (`Scorer.enclosed_left`), scored before J (lifted.Placement; owner 2026-10-11,
"The insulator world").

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
ring cell reaches the street. Stranded is then 0-2% at h 1-0.5 and 0.000 at h <= 0.35 (three
test blocks; not in general: "The insulator world").

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
the default's time and 0.72x the base's (median). As on the 46 held out (-0.015 against base,
-0.017 -- -0.019 against the default), a cheaper point, not a better one: it wins only on 28051
(+0.0054 over the default) and ties on 1182; its worst is 62385 (-0.059, a gate the SIMP presets
find). Every large block now has the default and the cheap preset but 19593 and 6498 (the
default only, uniform h 0.5).

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

### Re-scoring the road methods: the large-block sentinel (owner 2026-10-10: "Go")

road_rescore.py (rows in road_rows/, not committed; the networks in the derive cache). The 220
study's lineup on the sentinel (22422 and 30848 gated, 5810, 9712), each network cut to its
street-first prefixes at D 0.02 -- 0.20 and whole, and scored three ways: (a) the shipped
permeability on count-weight prefixes; (b) roadless with only the walking model changed (count
population, p 1, the same prefixes); (c) roadless as the optimizers use it (area population, J_2,
area-weight prefixes). CARVE, h 0.5, uni, ell 3, K 8. The scorer reproduces the stored SIMP
default and cheap clearings' Lens A to 2.6e-10 on all four blocks, and study.py's rows on a small
block to 1e-11.

- Cost per block, CPU s single-threaded: clearance 1 -- 5, grid < 1, clearance_looped 20 -- 47,
  resistance_lp 42 -- 588, cycle_native 110 -- 622 (n^1.07), cycle_desire 2,560 -- 6,200 where
  its betweenness counts were not cached: ~6,000 -- 7,000 s and 6 -- 10 GB per km^2, by area
  rather than buildings (5810's 1,083 s reused cached counts). Every other proposal stayed under
  1 GB. The arterial took 2,728 -- 6,632 s and passed the 2 h cap on 30848; at `max_roads: 60`
  its whole networks displace 0.6 -- 3.5%, so it never reaches D 0.05. Scoring: ~2 s per solve; a
  block's networks, every prefix under both lenses, and the clearings take 5 -- 11 min on the GPU
  (15 -- 29 GB).
- On the composite 0.5a5x8 (22422, 9712) (c) moves by at most 0.0024 (0.0005 at D 0.05) and the
  stored clearings by -0.0007 to -0.0028; the rankings keep, but for a near-tie at D 0.20 (0.1305
  against 0.1306).
- Lens A at D 0.05 among the six methods reaching it, (a) (b) (c), best first by (c):

    9712   resist_lp .938 .068 .065     cycle .876 .050 .059         cycle_desire .886 .039 .040
           clear .716 .033 .039         clear_loop .759 .034 .039    grid .754 .032 .034
    22422  resist_lp .933 .100 .0054     cycle_desire .842 .071 .0041  cycle .819 .063 .0034
           grid .672 .046 .0025          clear_loop .732 .038 .0025    clear .660 .032 .0023
    30848  cycle .763 .385 .292         resist_lp .827 .431 .291     cycle_desire .767 .371 .282
           grid .679 .325 .262          clear_loop .705 .292 .245    clear .714 .296 .240
    5810   resist_lp .892 .139 .148     cycle .876 .118 .139         cycle_desire .887 .089 .102
           grid .805 .075 .100          clear_loop .730 .076 .093    clear .620 .061 .071

- Kendall tau over those six, D 0.05 | D 0.10: (a, b) 0.60 -- 0.87 | 0.60 -- 0.87; (a, c) 0.33 --
  0.87 | 0.33 -- 0.87; (b, c) 0.73 -- 1.00 | -0.07 -- 1.00. On the 220, (a, b) at D 0.10 had a
  median of +0.43 (IQR 0.24 -- 0.62); all four blocks here are at 0.60 or above. The leaders
  separate too: under (b) at D 0.10 resistance_lp leads cycle_native by 0.02 -- 0.08 on all four,
  where on the 220 the four leaders tied within 0.004 (the arterial among them). resistance_lp
  leads (c) at D 0.05 on 3 of 4 (behind cycle_native by 0.0014 on 30848), with 1.5 -- 2.6x
  cycle_native's road length at the same displacement.
- 30848 is where (b) and (c) part (tau -0.07 at D 0.10): from D 0.05 to 0.10 (c) jumps for
  cycle_desire (0.28 to 0.52) and clearance (0.24 to 0.48), not for resistance_lp (0.29 to 0.35),
  while under (b) all three gain 0.09 -- 0.12. A pocket opening, by the jump (not inspected):
  worth much to J_2 on area, little to P on count.
- 22422, gated: no road prefix opens the gate before D 0.15 ((c) at most 0.0104);
  clearance_looped, cycle_native, cycle_desire and resistance_lp open it at D 0.19 -- 0.20 ((c)
  0.53 -- 0.55; resistance_lp's whole network, D 0.23, 0.79), clearance never (0.006 at its whole
  network's D 0.14). SIMP's 70 buildings get 0.783 at D 0.05. The equity power shows:
  resistance_lp's (b) is 0.100 at D 0.05, its (c) 0.005.
- The arterial: (a) 0.82 -- 0.89 for whole networks displacing 0.6 -- 3.5% (on 22422 its network
  at D 0.016 outscores cycle_native's at D 0.05, 0.875 against 0.819), (c) 0.007 -- 0.060. The
  shipped metric pays for road length through open ground (a road edge conducts ~204x a footpath
  per metre); roadless pays only for the corridor a road frees, and open ground is walkable
  already.
- The price of roads at D 0.05 under (c), a lower bound (the road prefixes overshoot to D 0.050
  -- 0.052, grid's to 0.061; the clearings stay at or under 0.05): SIMP's default minus the best
  road method +0.065 (9712), +0.111 (5810), +0.363 (30848), +0.777 (22422); the cheap preset
  +0.054, +0.075, +0.383, +0.756. The best road network gets 50% and 57% of SIMP's Lens A on the
  ungated 9712 and 5810, 45% and 0.7% on the gated 30848 and 22422.
- The full run (the 82 large blocks and 5810@major). The sentinel agent's projection (236 -- 331
  CPU-h, the arterial a floor; 9.2 GPU-h) took every block at the sentinel's median and slowest
  per-block times, but cycle_desire's cost goes with area. By area it is ~45 CPU-h on the 60
  blocks that fit a 48 GB card at h 0.5 (25 km^2, the largest 1.5 km^2) and ~500 CPU-h on the 23
  that do not (276 km^2; at 6 -- 10 GB per km^2 the five of 24 -- 52 km^2 need more memory than
  this machine's 250 GB). The other five methods: ~3.5 CPU-h on the 60. Scoring ~5 GPU-h on the
  60 at h 0.5 and ~4 on 0.5a5x8 for the 23 (63612 near 47 GB even there; ollama holds 25 GB of
  the local card at times, which leaves room only for blocks under ~20M unknowns). The arterial
  at `max_roads: 60` is left out: it reaches no Lens A here, and its cost goes with its road
  count.

Stage 1 launched 2026-10-10 22:52: the six methods on the 60 that fit (56 new blocks, 336
proposals), 10 processes, a 6 h cap per proposal, then the prefixes; scoring after, on the local
GPU when there is room or on the cluster. Open: the 23 oversized blocks (cycle_desire there needs
another plan: by its memory per km^2 the largest do not fit), and the arterial (a higher cap on
the cluster, or out). Done 2026-10-11 04:26: all 336 proposals, none failed or timed out
(cycle_desire 43.2 CPU-h, longest 4.3 h on 30796; the other five 2.5 CPU-h together), and the
prefixes for all 60.

What four blocks say: the shipped metric and roadless order the road methods alike more often on
large blocks than on the 220's small ones, and roadless tells the leaders apart here. They part
on what a road is worth: (a) pays for road length (the arterial), roadless for the buildings a
road removes, and on roadless's one scale clearing beats the best road network by 0.065 -- 0.777
at D 0.05. Whether the agreement holds across blocks is the full run's question.

### The shipped metric's footpath floor and edge length (owner 2026-10-11: "Are any overcomplicated or hacky in any way?")

perm_floor.py, on the sentinel's count prefixes at D 0.05 (the six methods). The footpath
clearance floor (FOOTPATH_EPS 0.02, reblock.mesh, outside PermeabilityParams) fires on 0.6 --
10.3% of mesh edges (the disks overlap, clearance < 0, on 0.6 -- 9.2%: the radii come from
building points, the edges join parcel centroids), but 0.005 or 0.08 moves (a) by at most 0.003
and no rank. Footpath conductance has no length term (the clearance fraction times a per-block
constant) while a road's falls as 1/length, so the road/footpath ratio is ~205x only at the
median edge (p5 -- p95: 55 -- 4,740x on 30848, 82 -- 811x on 9712). Footpaths at clearance/length
(same median) raise every method by 0.01 -- 0.05, keep the order on 22422 and 5810, and swap one
pair on 9712 and 30848 (tau +0.87). Neither is what (a) turns on; the lane conductance tuned for
method spread (the road/walk ratio) is.

### The insulator world: buildings as imperfect insulators (owner 2026-10-11: "We shouldn't exclude homes. Instead, we could maybe have buildings be imperfect insulators")

insulator.py; insulator_report.md has every table, insulator_rows/ the summaries (parquet
ignored). Each cell's open fraction becomes o + eps (full - o), the eps world clear.py and
relax.py search in, made the metric: every home drains, and clearing stays monotone. A stranded
home injects over its whole ring, sealed as it is; one with no ring cell, over the free cells
within 1 m of its own outline. On the 31 small blocks with stranded homes (6,276 homes, 88
stranded), ten controls, and 22422, 30848, 5810 and 9712 on the GPU, eps 1e-4 -- 0.1; at eps 0
the stored Lens A is reproduced to 2e-8. r is a formerly stranded home's u over the largest u
among its 6 nearest open-access homes (the owner's criterion: worse, but not massively worse).

**No single eps.** r is moderate at 1e-3 -- 3e-3: median 2.47 (count) / 1.60 (area) at 1e-3 and
1.36 / 1.03 at 3e-3, over the 74 homes walled into their own ring (25 blocks); at 1e-4 it is
17.6 / 10.3, p90 52 / 33. The few homes a finer raster opens (7 in 5 blocks at h 0.25) match at
eps* ~1.6e-3 (count; 5e-4 -- 3.5e-3). Leakage wants 1e-4 or less: 22422's gate (SIMP 0.783 at
eps 0) is worth 0.764 at 1e-4, 0.628 at 1e-3 and 0.243 at 1e-2, and homes behind slivers (u
above 8x the block median) keep 92 -- 94% of their relative escape time at 1e-4, 54 -- 63% at
1e-3. The finer raster says those long escapes are real (u / block median at h 0.25 over h 0.5:
1.00 -- 1.04 in every band), so the leak at a gate is a distortion, not a correction. Controls
barely move at 1e-4 (median u ratio 0.998, J_2 shift 0.006). On 30848 the eps world reorders the
stored clearings: cheap leads SIMP by 0.020 at eps 0, they tie at 1e-3, SIMP leads by 0.022 at
3e-3.

**Most stranding is placement, not walls.** On the full run's 60 large blocks 60 of 100,724
homes are stranded (0.06%; at most 0.84% of a block, 41673). 36 of them, and 25 of the 88 on the
small blocks, are shadowed: open space with a path to the street lies within 1 m of the outline,
in a neighbour's ring under the nearest-building rule. Injecting on those cells frees them with
no parameter and no leak; run here only for the homes with no ring cell at all (22 large, 7
small), not for the shadowed homes with a sealed ring (14 and 18).

**Correction (the demand fix, 2026-09-29):** "0.000 at h <= 0.35" held for its three test blocks
only. 77 of the 88 are still stranded at h 0.25 (81 at 0.35), behind a median 3.0 m of footprint
(IQR 1.8 -- 4.2).

**Confining the leak** to the stranded pockets keeps every gate exactly (30848's and 5810's
open-home Lens A equal the stored values; 22422 and 9712 have no stranded home) and meets r at
the median: the footprint within a pocket's barrier + 1 m at eps 3e-3 (median r 2.1 / 1.7, p90
8.2 / 5.3), or the cheapest crossing opened, with one cell per wall at 0.1 (a cost per wall
crossed: 1.75 / 1.37, p90 11.8 / 6.6); both match the finer raster at the median. The tail is
lattice geometry: with that crossing fully open, 38486's 9-home courtyard, reached by a winding
one-cell corridor, still has r 37. Next: a door link, one conductance per walled-in pocket
across its thinnest wall, an edge in the operator (leak-free by construction, blind to the
corridor's shape). If one eps is wanted meanwhile, 1e-4 (SIMP's own) keeps the gates, but a few
walled-in homes then dominate J_2: on 30848 one (u 3,104 against a median of 33) cuts SIMP's
Lens A from 0.615 over open homes to 0.480 over all.

**The door link** (insulator_report.md, "The door link"). The placement fix extended to every
shadowed home leaves 63 walled-in homes on the 31 small blocks (1.0%) and 24 on the 60 large
(0.024%, 12 blocks; 30848's and 5810's are freed). Each gets one fixed conductance g across its
thinnest wall, open cell to open cell, heading-preserving, an edge in the operator: nothing else
opens, so no open home's u from the open homes' own demand moves by more than 0.1% (g up to
0.03). At g 0.01 (about 63 m of 1 m-wide straight alley at ell 3 m) the walled-in homes' median
r is 1.80 (count) / 1.51 (area) on the small blocks, p90 10.5 / 11.0, and 1.57 / 1.51 on the
large, p90 4.9 / 3.5. On the 12 large blocks Lens A over all homes trails Lens A over open homes
by at most 0.022 (20543) and keeps the stored clearings' order; the walled-in homes hold at most
4.1% of J_2 with nothing cleared and 12% after SIMP's clearing, which leaves their doors as they
are. What breaks is the small blocks' tail (p90 10 -- 13 at every g that keeps the median near
2, in every form), set beyond the door: 38486's courtyard opens into space its own demand
congests, and 21622's home 178 has neighbours at the street (u ~3), so any door costs it r 33.
Measured and worse: a door per pocket (r grows with the pocket's homes: median 1.3, 4.0, 11.3 for
1, 2 -- 3, 4+), the crossing's partial face cells as ends (a sliver that throttles the door at
any g), an isotropic door node (r within 0.3%, and an extra unknown the GPU solver cannot
coarsen), a thickness term (the tail as wide or wider at matched medians). The finer raster
prefers g 0.0015 -- 0.0034 (three homes) and finds the fix about 2x optimistic for the homes it
moves (eight homes).

**The private crossing** (crossing.py; owner 2026-10-11: "Is there a way to simplify your
suggested method? The if-else feels unnatural", then "a version that can account for
traffic/contention ... since it would reward multiple points of egress"). Two ways out for the
walled-in homes with no edge in the operator, on the door's crossings (one per walled-in home
and pocket, open ends), on the 31 small blocks (63 homes), each calibrated to the door's median
r (1.80, count). A toll: the home's demand enters O at the crossing's street end and pays a
private resistance after the solve. Per crossing (1 / g, g* 0.0104) it is the door without the
edge: median 1.80, p90 10.2 against the door's 10.5, max 31.1 against 32.8. Per metre of
footprint crossed (T / g1, g1* 0.040) it lowers thin walls and raises thick ones but keeps the
tail (p90 10.8). Private walls: the walled-in walkers cross a private copy of the block whose
footprint conducts eps (its exits held at 0), then go on in O from where they come out; O's own
traffic cannot use it, homes share its walls (contention) and parallel walls add (several ways
out). At eps* 3.0e-3 (median 1.78) the p90 halves (5.0 count, 3.0 area, against 10.5 / 11.0):
pockets of 4+ homes go from median r 7.6 to 1.1, since 38486's courtyard leaves on every side
into open space at u ~60 rather than through its thinnest wall into space at u ~1,670 (a median
19% of a home's walkers land within 5 m of that wall's street end). But the max rises (40.1,
21622's 178), five of the seven homes inside a footprint get worse (42404's 65: 13.8 against
6.2), and pockets of 2 -- 3 homes go from 1.8 to 2.2. Several shared doors per pocket instead
would make a courtyard a corridor between the spaces on its sides (38486's: u ~1,670 and ~60),
the leak a single door avoids by ending in a dead end; any model with several ways out needs the
private rule stated. Stopped when the owner dropped the criterion (2026-10-11: "I am fine with
having fully enclosed buildings be much worse off than exposed neighbors"); next, one wall
resistance for everyone, scored as one number at eps 1e-4 or as two (the population still
enclosed, then Lens A over the rest).

**Uniform walls** (walls.py; insulator_report.md, "Uniform walls: one number against two"; owner
2026-10-11: "I am fine with having fully enclosed buildings be much worse off than exposed
neighbors, so we can try to find a uniform resistance that buildings can have", then "I'm not
sure between a single 1e-4 and order via two numbers, so maybe we can test both"). A: every
footprint conducts eps 1e-4 (SIMP's), every home injects, Lens A over all. B: walls are walls;
first the population still enclosed (no injection cell open and street-connected), then Lens A
over the rest (today's metric with the placement fix: the stored Lens A to 4e-8, or 1.6e-3 where
a shadowed home moved). On the 41 small blocks the walls carry at most 5% of every open home's
escape up to eps_gate 4.2e-6 (the worst, 43365's home 162, follows 1 / (1 + eps / 9.9e-5);
30848's gated 676 puts it at 3.7e-6 or less), and at 1e-4 302 of 7,821 open homes (3.9%) lose
more than 5%. The 63 enclosed homes' r and z go as 1/eps (median r 12 at 1e-4) and their share
of J_2 with them: 99.8% at 3e-6, 98% at 1e-5, 41% at 1e-4, 3% at 1e-3. On the 12 large blocks
with enclosed homes (24) and the four sentinels (cluster, one task a block, 42 minutes): at 1e-4
the share is 0.7 -- 80% by block, each toll set against its block's escape times (median tier-0
u 5.9 -- 301). On 20543 and 5706 one home holds four-fifths, so SIMP's D 0.05 clearing scores
0.08 and 0.10 under A against 0.50 and 0.58 under B, and clearing the 1 -- 3 buildings that free
that home (D 0.0025, 0.0007) is worth 7.2x and 5.5x it. The gated sentinels leak 0.018 -- 0.040
at 1e-4 (22422, 30848). SIMP's clearing ahead of the cheap preset's: today 15 of 16, A 14 (22267
flips: cheap frees a home), B 13 (21252 too). B is the uniform resistance's eps -> 0 end: any
enclosed home left outweighs everything else there (weighed by its walls' toll, where B counts
population), then the rest's J_2 (the enclosed homes' own flow through the open network, which B
drops, and the leak together move it by at most 0.0024 at 4.2e-6); any finite eps is an exchange
rate between the two that eps, the walls and the block set. Searching at eps_gate is out: with
nothing cleared the GPU solver stopped at 2,000 iterations on 5 of the 16 (residuals 2e-6 --
4e-5), as the swings' screens did on 30848 at 1e-6. B's optimizer is two stages: free (every
enclosed home's cheapest crossings, D 0.0004 -- 0.0144 a block, median 0.0022), then today's
search with the rest. Recommended: B; the owner chose it (2026-10-11: "B SGTM").

**The port** (owner 2026-10-11, "B SGTM"). B is the metric. lifted.Placement replaces
lifted.demand and the per-cell owner map with S (cells x homes, so a cell can take several
homes' demand): the shadowed homes inject on their open side, and the enclosed ones (no free
cell within 1 m of the outline reaches the street) inject nothing and are reported by
Scorer.enclosed_left, which road_rescore.py and resolution_check.py record as `enclosed`. The
optimizers' adjoint loads (Scorer.adjoint_load) and device readouts (relax, the cat greedy's
gram, goal_indicator) run on S. Checked on the CPU against 9e10bbf where no home is shadowed
(six small blocks: f, u0, P0, J_2 and Lens A to 4.4e-13, SIMP's gradient to 6.8e-14, the tension
and gram to 9.4e-13), and against the walls rows where homes are (17 small blocks, 25 shadowed
homes: u0 to 2.2e-9; the enclosed share left exact on all 41 small blocks and all 60 large-block
clearings). B frees a home when any cell within 1 m of its outline opens to the street, so it
frees more than walls.py did (40 of 561 single-building probes, never fewer). The fix moves the
baseline where a home was shadowed: on those 17 blocks J_2 rises by a median 1.3% (29% on 43817,
from one sliver-door home at 37x the median u), and 400 of 3,515 ring homes move by more than 1%
(the cells they now share congest). insulator.py, crossing.py and walls.py are deleted (code at
9e10bbf); fft_homog's extracts (in a session scratchpad) predate the change and must be
regenerated before its scripts run again.

### Re-scoring the road methods: the 60 large blocks under B (owner 2026-10-11: "B SGTM")

road_rescore.py `score` on the cluster (run road-b-r1, one task a block, its networks, prefixes
and stored clearings shipped as a tar: cluster.py `--unpack`): the six methods of stage 1 on the
60 blocks that fit at h 0.5, every prefix scored as on the sentinel ((a) the shipped
permeability and (b) roadless count p 1 on the count prefixes, (c) roadless area J_2 on the area
prefixes) with its B (i), the population share it leaves enclosed, and the stored clearings
rescored beside them. 5.6 GPU-h (a median 3.6 min a block, the longest 25 min; GPU peak 40.5
GB), 1 h 43 min on four GPUs. `report` ranks (b) and (c) in B's order and writes
road_rows/summary_*.csv.

- The scorer: the stored clearings rescore to their stored Lens A within 3.2e-9 on the 38 blocks
  with no shadowed home (14401's are on 0.5a5x8 only) and move by a median 7e-5 (at most 1.6e-3,
  41515; down on 31 of 42 rows) on the 21 with one (36 homes, insulator_rows/summary_census.csv):
  Lens A over the rest now counts a home the nearest-building rule had stranded. Against
  walls.py's B on its 16 blocks: (ii) to 3.8e-8, (i) to 1e-16.
- Enclosed homes: on 12 of the 60 blocks (24 homes, 0.010 -- 0.225% of a block's area
  population, median 0.077%). At D 0.05 the road prefixes free some (all) of them on:
  resistance_lp 7 (5), clearance 4 (2), clearance_looped 4 (3), cycle_native 3 (1),
  cycle_desire 3 (0), grid 1 (1); some road frees every one on 7 of the 12. A road's corridor
  frees homes in passing; the stored clearings, searched before B, mostly do not: SIMP's default
  frees some on 2 (41673, 41806) and all on none, the cheap preset some on 4, all on 1 (21252).
- The price of roads at D 0.05 in B's order: SIMP's default is ahead of the best road prefix on
  51 of the 60 and behind on (i) on 9, all among the 12 (the cheap preset 52 and 8). Where (i)
  ties (51 blocks) SIMP's default leads by a median +0.148 (p10 +0.089, p90 +0.345; +0.047 --
  +0.777), the cheap preset by +0.127 (+0.042 -- +0.756), and the best road gets a median 57% of
  SIMP's Lens A (p10 43%, p90 70%). Where a road is ahead it is on (i) alone, at times with
  little Lens A: on 23597 clearance frees the enclosed home at 0.008 where SIMP's default leaves
  it at 0.475. walls.py's free+simp (the union, D 0.050 -- 0.064) was ahead of every road prefix
  on all 12 ((i) 0, (ii) 1e-5 -- 6e-3 above SIMP's alone); B's optimizer, free and then the
  search with the rest of the budget, is next (BACKLOG 3).
- The lenses: Kendall tau over the methods reaching both budgets, median [IQR] over the 60 in
  B's order, D 0.05 | D 0.10: (a, b) +0.60 [+0.47, +0.73] | +0.60 [+0.40, +0.73]; (a, c) +0.47
  [+0.20, +0.73] | +0.47 [+0.30, +0.60]; (b, c) +0.73 [+0.60, +1.00] | +0.80 [+0.57, +1.00]. The
  sentinel's agreement holds (the 220's (a, b) at D 0.10: +0.43). The 12 blocks with enclosed
  homes agree less on (ii) alone already ((a, c) +0.13 at D 0.05 against +0.60 on the other 48),
  so B's order is not what parts them.
- The leaders at D 0.05: resistance_lp under (a) on 56 (cycle_desire 4), (b) 54, (c) 41
  (cycle_native 7, cycle_desire 4, clearance_looped 4, grid 2, clearance 2; B's order moved 6 of
  the 12); at D 0.10 under (c) 39 (cycle_desire 7, cycle_native 6, grid 5). The medians of (c)
  at D 0.05: resistance_lp 0.224, cycle_native 0.192, cycle_desire 0.183, clearance_looped
  0.149, clearance 0.127, grid 0.119; resistance_lp lays 1.47x cycle_native's road (p10 1.12,
  p90 2.04) for the same displacement. clearance reaches D 0.05 on 56 of the 60 and D 0.10 on
  48, cycle_native on 59 and 56, cycle_desire on 59 and 55, the rest on all.

What the 60 say: roadless and the shipped metric order the road methods alike, resistance_lp
leads under all three lenses, and wherever B (i) ties SIMP's default leads the best road network
by at least +0.047 at D 0.05. Where (i) decides, the stored clearings lose because they were
searched without it, which B's optimizer is for.

### B's optimizer: free first (owner 2026-10-11: "B SGTM")

clear.free_first, run first by relax.py (one, path) and polish_greedy.py: the crossings that free
the homes enclosed at baseline, the least D per unit of the home's population first, while they
fit the budget; then today's search with the rest, the free stage's buildings held (SIMP at x 1
through its roundings and polish, the greedy from them, the polish never moving them). Each round
is one Dijkstra from the street over a graph of the open components at baseline and the
buildings' 4-connected pieces (14 of 40 blocks checked have a building in 2 -- 3 pieces, which as
one region would claim a path the raster lacks); the metric's own test must free the chosen home
or the run stops. Where no home is enclosed nothing is held and the run is unchanged (on the CPU
within solver noise: pyamg draws its start vectors from numpy's global RNG, up to 1e-11 in Lens
A between processes).

- Small blocks (CPU): 19263's three enclosed homes (0.0235 of the area population) are freed by
  one building (D 0.0072), 38486's courtyard (nine homes, 0.0501) by one (D 0.0042), 46295's home
  with footprint all round by a neighbour (D 0.0036; its own building 0.0074). What freeing costs
  Lens A at D 0.05: 19263 -0.063 (SIMP) and -0.074 (cheap); 38486 +0.022 and +0.020 (the freed
  courtyard is space neither search had opened); 46295 -0.001 and -0.007. 21 more small blocks
  with enclosed homes: D 0.0007 -- 0.0139 a block, the top on 42404 for a home of 0.00065 of the
  population.
- The 12 large blocks with enclosed homes (runs free-simp and free-cheap, 0.28 and 0.10 GPU-h, as
  before): both presets free every home, the free stage D 0.00008 -- 0.0026 (median 0.0008, 1 --
  4 buildings). Lens A against the stored clearings: SIMP's default a median -0.0030 (-0.0060 --
  +0.0009, up on 2), the cheap preset -0.0021 (-0.0152 -- +0.0056, up on 3); the clearings
  overlap the stored ones at a median Jaccard 0.88 and 0.83. road_rescore.py (run free-rescore)
  reproduces their Lens A to 8.9e-9, none enclosed.
- In B's order at D 0.05 both presets now lead the best road prefix on all 12 (before: SIMP's
  default on 3, the cheap preset on 4): on (i) on the 5 where no road frees every home, and where
  (i) ties (7) by a median +0.171 (+0.130 -- +0.467) and +0.140 (+0.074 -- +0.459). Over the 60
  both are ahead on all; where (i) ties (55) SIMP's default leads by a median +0.152 (p10 +0.090,
  p90 +0.358; +0.047 -- +0.777) and the best road gets a median 56% of its Lens A (p10 32%, p90
  71%), the cheap preset by +0.127 (+0.042 -- +0.756). road_rows/summary_*.csv are regenerated.
- Not free-first: the composite mesh (free_first raises on a CompositeGrid with an enclosed
  home; the 12's cheap_a5x8 clearings predate it), the plain greedy (clear.greedy_block, the
  track), and the stored rows on the other blocks with enclosed homes (the 21 small ones above
  among them).

### Transparency for the road methods: the sentinel (owner 2026-10-12: "Leave out arterial")

BACKLOG 5, the experiment the owner approved: does a road method build a better network at equal
displacement when the buildings the roadless optimum clears are made cheaper for it, to displace
(the price) and to route through (the route)? Judged where item 4 judged the lineup, lens (c) at
area D 0.05, and against random sets of buildings made as cheap.

What a scout found first (scratch scripts). No method forbids crossing a building and none prices
it while routing: cycle_native, resistance_lp and the clearance tree route by length on
chord_diag (34 -- 70% of its edges cross a footprint) and price displacement afterwards, and
clearance_looped's loop stage has no building term at all; a price discount alone barely
reroutes. Discounting every building alike is no control (a no-op for cycle_native's and the
arterial's scale-free rankings, a budget rescaling for resistance_lp, the repulsion logit for
clearance), hence random sets matched to the clearing. The arterial reaches D 0.011 -- 0.044 with
its 60 roads at 2.7 -- 6.6k CPU-s a network, short of D 0.05: left out. Headroom: on 59 large
blocks a median 6 -- 12% of a stored network's displacement at area D 0.05 falls on SIMP's set (5%
by chance), its roads touching 15 -- 27% of the set's buildings. A CPU pilot (h 1.0, (c) at area
D 0.05, 9712 and the four smallest of the 60): cycle_native price and route at f 0.1 +0.032 (up
on 5 of 5, random sets -0.001), resistance_lp price at f 0.1 +0.034 (5 of 5; random -0.015),
clearance_looped route +0.030 (4 of 5; random +0.001), against gaps to SIMP's clearing of 0.14 --
0.17; f 0.1 was picked after the fact from four levels.

What Task 1 built (bcbca3c -- f7c2952): reblock.transparency, a Transparency Strategy of
per-building factors f in (0, 1] (Opaque, every factor 1, in every preset), PricedOverlap, and
TransparentSubstrate (the base's edge length times the mean factor along the edge, the smallest
where footprints overlap); a required `price` on cycle_native (it ranks candidates by priced
displacement, its cap reads true displacement), resistance_lp (its budget row and rounding
priced, so its true displacement passes max_displacement by what the discount lets off) and
clearance (its building term times the nearest building's f); transparency_sets.py, the sets
(C, C', and the matched random sets: AreaMass, 20 strata of equal footprint area, area ratio
median 1.002 on the 59); and the guards, among them that Opaque reproduces item 4's networks.

The arms (transparency_run.py, PLAN; every value that changes a network in
transparency_rows/arms.csv), per method and block: the base, opaque, recomputed and checked
against item 4's network; the method's own knobs, opaque (cycle_native's `shortlist` 4 and 16, the
preset's 8, and the desire-routed preset cycle_native_betweenness_contrast, whose item-4 network
is reused; resistance_lp's `chunks` 4 and 16, the preset's 8; clearance_looped's tree's
`repulsion` -2, 2 and 4, the preset's 0); C, the buildings SIMP's default clears at D 0.05, in
the main form at f 0.5 and 0.1 (price and route for cycle_native and clearance_looped's tree,
price alone for resistance_lp, whose route lowered it in the pilot); C at f 0.1 ablated
(cycle_native and clearance_looped price only and route only, resistance_lp route only and both);
C', the cheap preset's clearing, in the main form at f 0.1; and the control, five random sets
matched to C (seeds 1 -- 5) in the main form at f 0.1. 41 arms, 160 networks to propose, fresh
(no derive cache), cut and scored as item 4 (h 0.5, uni, the same budgets and prefixes; (b) and
(c) on the cluster).

The screen, fixed before any result (committed and pushed before the first proposal ran):
- Lens (c) at area D 0.05 is the primary number. No home is enclosed at baseline on the sentinel
  (item 4's rows), so B's order is (c)'s value; report stops if any prefix leaves one enclosed.
  (a) at count D 0.05 and 0.10 is the guard.
- Collapse veto: an arm more than 0.1 below its method's base on (c) on any of the four blocks is
  dropped.
- Per method, the main form at the f (0.5 or 0.1) with the larger mean (c) gain over base (the
  mean over the four blocks of arm minus base) advances to the 13 tuning blocks, with its random
  control, if (i) that mean gain is above 0, (ii) its (c) is above the mean (c) of its five
  random sets on at least 3 of the 4 blocks, (iii) its (a) loss against base, the median over
  the four blocks, is at most 0.01 at count D 0.05 and at count D 0.10, and (iv) the veto does
  not drop it. The random sets run at f 0.1 only; a main form at f 0.5 is held against them as
  they are.
- An ablation advances only if it passes (i) -- (iv) ((ii) against the main form's random sets,
  the only control run) and its mean gain beats the main form's.
- C' advances beside the main form if the main form advances, the veto does not drop C', and its
  mean gain is within 0.01 of C's main form at the same f (0.1), or above it.
- Reported beside it: the share of the gap to SIMP's default closed, (arm's (c) - base) / (SIMP's
  (c) - base) per block and its mean (SIMP's (c) from item 4's rows, the same scorer and mesh; the
  stored clearings are rescored here as the check); each arm's mean (c) against the method's best
  own-knob point (the best mean (c) of its base and knob arms); per arm and block, the whole
  network's true displacement (it must reach D 0.10 for the prefixes), the share of the (c)
  prefix's true displacement that falls on C, and for a random set its area over C's and its
  Jaccard overlap with C.
- The 13 tuning blocks and the held-out blocks are not part of this run.

Run (2026-10-12), proposals at 13d6c30, the screen's commit (two canary runs on 9712 first): 160
networks, fresh, 8 processes, 41 min, 5.27 CPU-h (cycle_native 3.1, resistance_lp 1.9,
clearance_looped 0.3), none failed, peak RSS 1.1 GB; a discount changes no method's cost (the
knobs do: shortlist 16 costs 2 -- 3x). Prefixes 27 min on 7 processes. Scoring on the cluster
(run transp-sentinel, one task a block, the arms' prefixes and networks shipped as tars): 2.87
GPU-h, 1 h 23 min on four GPUs (22422 81 min, 29.6 GB).
- Checks. The 12 recomputed base networks are item 4's, WKB row for row and widths. The
  desire-routed preset is reused: Task 1 left its substrate's code untouched (desire_substrate,
  demand_greedy, desire_lines, betweenness) and changed cycle_native.py only in arithmetic that
  is the old float for float at factor 1, and recomputed at HEAD (cycle_native fresh, its
  betweenness counts from the derive cache) it is item 4's network on 9712 and 22422 (204 and
  427 s). The stored clearings rescore to item 4's rows within 8.5e-10, the base arms' (a), (b)
  and (c) within 1e-16, 2.3e-9 and 6.4e-8. No prefix leaves anyone enclosed, and every whole
  network reaches D 0.15 under both weightings (priced resistance_lp's true D 0.203 -- 0.237
  against its 0.20 cap).
- The discount moves the roads: the area-weighted share of the (c) prefix's true displacement
  that falls on C, mean over the four blocks (about 5% by chance), goes from 0.086 to 0.218 for
  cycle_native's main form at f 0.1 (price alone 0.101, route alone 0.216), from 0.124 to 0.189
  for resistance_lp's (route alone 0.178, both 0.285), from 0.072 to 0.179 for
  clearance_looped's (price alone 0.105, route alone 0.158). A random set draws the roads onto
  itself too, less (its own share 0.11 -- 0.21, 0.04 -- 0.16, 0.07 -- 0.19). The random sets'
  area matches C's within 1.3% on 30848 and 9712 and within -6% / +20% on 22422, but every seed
  falls to 0.81 -- 0.89 of C's on 5810, which biases the control there toward a smaller effect
  (Jaccard overlap with C 0.022 -- 0.050).
- Lens (c) at area D 0.05 per block, the mean gain over base, the mean share of the gap to SIMP's
  default closed, the median (a) loss against base at count D 0.05 | 0.10, the blocks above the
  random sets' mean (>r), and the share on C as above (p+r: price and route); > advances. No arm
  is vetoed (the worst is 0.037 below its base, cycle_native's desire-routed preset on 5810).
  SIMP's default (c): 0.7829, 0.6550, 0.2589, 0.1299.

    cycle_native       22422  30848   5810   9712     gain     gap  (a) .05 | .10  >r  on C
    base              0.0034 0.2921 0.1389 0.0588                                       0.086
    shortlist 4       0.0035 0.4973 0.1086 0.0465  +0.0407  +0.035 +0.0065 | +0.0010  1  0.088
    shortlist 16      0.0048 0.3072 0.1349 0.0567  +0.0026  -0.005 -0.0045 | -0.0066  4  0.088
    desire-routed     0.0041 0.2818 0.1021 0.0399  -0.0163  -0.150 -0.0109 | -0.0050  1  0.085
    C f0.5 p+r        0.0046 0.2993 0.1370 0.0561  +0.0009  -0.008 +0.0063 | +0.0016  4  0.175
    C f0.1 p+r        0.0035 0.3146 0.1557 0.0571  +0.0094  +0.045 +0.0101 | -0.0021  3  0.218
    C f0.1 price      0.0033 0.2995 0.1307 0.0509  -0.0022  -0.040 +0.0083 | -0.0002  3  0.101
    C f0.1 route      0.0056 0.3023 0.1387 0.0551  +0.0021  -0.006 +0.0061 | +0.0007  4  0.216
    C' f0.1 p+r       0.0045 0.2899 0.1405 0.0528  -0.0014  -0.019 +0.0063 | +0.0010  3  0.141
    random f0.1 mean  0.0038 0.2902 0.1252 0.0502  -0.0060  sd 0.0007 0.0093 0.0061 0.0027

    resistance_lp      22422  30848   5810   9712     gain     gap  (a) .05 | .10  >r  on C
    base              0.0054 0.2907 0.1481 0.0650                                       0.124
    chunks 4          0.0065 0.3144 0.1607 0.0557  +0.0070  +0.009 +0.0091 | +0.0027  3  0.127
    chunks 16         0.0052 0.2856 0.1388 0.0571  -0.0056  -0.055 -0.0004 | +0.0009  0  0.113
    C f0.5 price      0.0056 0.3026 0.1443 0.0657  +0.0022  +0.002 +0.0004 | -0.0019  3  0.162
  > C f0.1 price      0.0060 0.2986 0.1608 0.0694  +0.0064  +0.051 -0.0029 | -0.0009  4  0.189
    C f0.1 route      0.0056 0.3016 0.1626 0.0592  +0.0049  +0.018 +0.0074 | +0.0006  3  0.178
  > C f0.1 p+r        0.0062 0.3201 0.1702 0.0714  +0.0147  +0.095 +0.0076 | +0.0010  4  0.285
  > C' f0.1 price     0.0056 0.3023 0.1530 0.0645  +0.0040  +0.017 +0.0045 | -0.0013  3  0.136
  > random f0.1 mean  0.0055 0.2892 0.1535 0.0639  +0.0007  sd 0.0004 0.0118 0.0070 0.0022

    clearance_looped   22422  30848   5810   9712     gain     gap  (a) .05 | .10  >r  on C
    base              0.0025 0.2446 0.0925 0.0388                                       0.072
    repulsion -2      0.0031 0.2428 0.0826 0.0412  -0.0022  -0.009 +0.0038 | -0.0058  2  0.077
    repulsion 2       0.0028 0.2313 0.0882 0.0376  -0.0046  -0.018 -0.0464 | -0.0386  2  0.072
    repulsion 4       0.0033 0.2713 0.0743 0.0395  +0.0025  -0.009 -0.0698 | -0.0509  2  0.087
    C f0.5 p+r        0.0031 0.2792 0.0823 0.0380  +0.0060  +0.004 +0.0160 | +0.0035  3  0.128
  > C f0.1 p+r        0.0036 0.2815 0.0958 0.0503  +0.0132  +0.059 -0.0095 | -0.0199  4  0.179
    C f0.1 price      0.0026 0.2852 0.0980 0.0354  +0.0107  +0.024 -0.0128 | -0.0056  3  0.105
    C f0.1 route      0.0037 0.2793 0.0893 0.0440  +0.0095  +0.031 +0.0009 | -0.0037  4  0.158
  > C' f0.1 p+r       0.0030 0.2809 0.1159 0.0449  +0.0166  +0.074 -0.0196 | -0.0086  4  0.095
  > random f0.1 mean  0.0024 0.2477 0.0794 0.0419  -0.0018  sd 0.0004 0.0111 0.0060 0.0047

- The screen. resistance_lp: its main form, C priced at f 0.1 (+0.0064 against f 0.5's +0.0022),
  is above its random sets on 4 of 4 and loses no (a) (-0.0029 | -0.0009): it advances to the 13
  with its random control, with the ablation price and route beside it (+0.0147, 4 of 4, (a)
  +0.0076 | +0.0010; route alone, +0.0049, does not beat the main form) and C' (+0.0040, within
  0.01). clearance_looped: its main form, C priced and routed on its tree at f 0.1 (+0.0132
  against +0.0060), is above its random sets on 4 of 4 and gains (a) (-0.0095 | -0.0199): it
  advances with its control and C' (+0.0166, above C); price alone (+0.0107) and route alone
  (+0.0095) do not beat it. cycle_native advances nothing: its main form at f 0.1 (+0.0094,
  above its random sets on 3 of 4) loses a median 0.01015 of (a) at count D 0.05, over the
  guard's 0.01 by 0.00015 ((a) is exact here: item 4's reproduced to 1e-16), a knife edge and
  the owner's call; its f 0.5 form passes every test at +0.0009 but is not the f with the larger
  mean, route alone passes at +0.0021 below the main form, price alone -0.0022, C' -0.0014.
- Beside it. The gap to SIMP's default closed: a mean 4.5% (cycle_native), 5.1% and 9.5%
  (resistance_lp, price and price and route), 5.9% and 7.4% (clearance_looped, C and C'); on
  gated 22422 0.3% at most, as no arm opens the gate at D 0.05 ((c) at most 0.0065 against
  SIMP's 0.783). Against random sets made as cheap the main form gains +0.015 (cycle_native),
  +0.006 (resistance_lp) and +0.015 (clearance_looped), partly because a random discount costs
  cycle_native and clearance_looped (-0.006, -0.002). Against the best own-knob point: for
  cycle_native it is shortlist 4 (mean (c) 0.164), all from a pocket it opens on 30848 (0.497
  against 0.292; below base on 5810 and 9712), and its main form is 0.031 behind it; for
  resistance_lp chunks 4 (0.134), the main form 0.0006 behind and price and route 0.0077 ahead;
  for clearance_looped repulsion 4 (0.097), the main form 0.011 and C' 0.014 ahead. The pilot's
  +0.03 at h 1.0 does not hold at h 0.5 on these blocks: on 9712, common to both, the pilot's
  arms give cycle_native -0.0017, resistance_lp +0.0044, clearance_looped's route alone +0.0052
  (with the price +0.0115).
- Unlike the pilot, resistance_lp's route does not lower it here (+0.0049 alone), and with the
  price it is the largest gain of the run; clearance_looped's C' (the cheap preset's clearing, 10
  -- 255 buildings against C's 70 -- 322) beats C, cycle_native's does not.

What the sentinel says: making the buildings the roadless optimum clears cheaper moves 1.5 --
2.5x as much of a road network's displacement onto them, but at equal displacement it buys
little of the roadless benefit, +0.006 -- +0.013 of (c) on average for the main forms (+0.015
and +0.017 for resistance_lp's price and route and clearance_looped's C'), 4 -- 10% of the gap
to SIMP's default, nothing on a gated block, and only resistance_lp and clearance_looped get
past the screen. The 13 tuning blocks are next for them: resistance_lp priced at f 0.1 (price and
route and C' beside it), clearance_looped priced and routed at f 0.1 (C' beside it), each with
its random control; cycle_native only if the owner waives the 0.00015 on the guard. Committed:
transparency_rows/arms.csv and summary_{records,screen,lens}.csv (the rows and networks are in
the ignored transparency_rows/ parquet and scratch).

### Transparency for the road methods: the 13 tuning blocks (owner 2026-10-12: "run the 13- and 60-block stages on orion/quadro2")

The sentinel's next stage: the 13 tuning blocks (simp_compare.TUNING, the sentinel's four among
them), every network proposed and cut to its prefixes on the cluster's CPU nodes, orion and
quadro2, whose GPUs are unusable, and scored on its GPUs; transparency_run.py's `tuning` stage,
under transparency_rows/tuning (the sentinel's rows stay where they are).

The CPU path (cluster.py's `cpu` kind; BACKLOG, Infrastructure): one task per (block, arm),
single-threaded, CPU only, pinned to one node named on the command line, a block's arms all on one
node (results can differ by machine) and two nodes two runs; at most 64 tasks at once on orion and
40 on quadro2. Each task proposes its network fresh (or copies item 4's, for the reused arm) and
cuts its prefixes (`transparency_run.py tuning arm <id> <arm>`); the stored clearings and item 4's
desire-routed networks ship as a tar per block, and the networks and rows come back through the
run's pull. cluster_submit held any run, CPU-only ones too, to the project's four GPUs' worth of
tasks at once; GeoffChurch/cluster_submit e299a2b lifts that for a run that holds no GPU.

The arms (transparency_run.TUNING_PLAN; transparency_rows/tuning/arms.csv), every discount at f
0.1, C SIMP's default clearing at D 0.05, C' the cheap preset's, and the random sets matched to C
(AreaMass, 20 strata), seeds 1 -- 5, in each candidate's own form:
- resistance_lp: base; chunks 4 and 16; C priced (the main form), C priced and routed, C' priced;
  the random sets priced, and priced and routed (ten).
- clearance_looped: base; its tree's repulsion -2, 2 and 4; C priced and routed (the main form), C'
  priced and routed; the random sets priced and routed.
- cycle_native: base; shortlist 4 and 16; the desire-routed preset, its item-4 networks reused
  after the sentinel's check (its code path unchanged since; the preset recomputed on the nodes on
  5810 and 20269 against item 4's); C priced and routed; the random sets priced and routed. Its
  main form is in by the controller's ruling: it missed the sentinel's (a) guard by 0.00015 (a
  median loss of 0.01015 against 0.01), the screen exists to save compute, the cluster's CPUs make
  it cheap, and adding an arm to the next stage cannot flatter that stage's evidence; it is held to
  the same rule as the others.
37 arms, 36 proposed per block (468 networks), fresh. Scoring reads only what the rule reads: lens
(c) at area D 0.05 and 0.10 (h 0.5, uni, B's order; road_rescore.py `score` with the area
population alone, the prefix rows 0.05,0.1), and (a) at count D 0.05 and 0.10 from the prefixes.
Where a node's network is the sentinel's, the reduced scoring must give the sentinel's values
(checked on its blocks).

The opaque check on the nodes, before any other arm runs: the base networks computed there must
equal item 4's (road_rows/scratch/networks/) on all 13; any difference stops the stage, as machine
dependence would undo the comparison's footing. The sentinel's blocks are recomputed there too and
set against the sentinel's local networks (expected identical).

B's order on the 13. Item 4's base prefixes at area D 0.05 leave population enclosed on two of the
13 (its committed lens rows): 20543 (cycle_native's and clearance_looped's, 0.0009 of it) and 23597
(cycle_native's and resistance_lp's, 0.0003). B puts (i), the population a prefix leaves enclosed,
before (c): an arm that leaves more than its base on a block is behind it there whatever (c), which
the rule treats as the collapse veto; one that leaves less is ahead (counted among the blocks up).
The means, medians and intervals are over (c) itself.

The choice rule, fixed before any result (this commit precedes the first proposal on the 13): per
method, among its C and C' arms (resistance_lp's three, clearance_looped's two, cycle_native's
one), the one with the largest mean (c) gain over base at area D 0.05 on the 13 that also has
- no block more than 0.1 below base on (c), and none behind it on B (i);
- a median (a) loss against base of at most 0.01 at count D 0.05, and at D 0.10;
- a mean (c) gain above the best own-knob point's: the knob with the largest mean gain on the 13
  (base is not a knob; the desire-routed preset is one of cycle_native's);
- a mean (c) above its own random sets' mean (the five seeds in its form, over the 13).
The chosen arm, with base, the best own-knob point and its random sets, goes to the held-out
blocks. A method with no arm that qualifies stops there, and that is its result. The rule asks no
mean gain above 0 of its own, only one above the best knob's, which is negative if every knob
loses. Reported beside it: each arm's mean gain with a Bayesian-bootstrap 95% interval over the 13
(paired by block: 10,000 Dirichlet(1, ..., 1) weight vectors over the blocks, seed 20261012, one
matrix for every arm); the share of the gap to SIMP's default closed, (arm - base) / (SIMP's (c) -
base) per block and its mean (SIMP's (c) from item 4's rows); the per-block values; and per arm and
block, as the sentinel did, the whole network's true displacement, the share of the (c) prefix's
true displacement on C, and for a random set its area over C's and its Jaccard overlap with C.

Run (2026-10-12), one task per (block, arm), single-threaded, every network fresh, from the rule's
commit (e4dae5b) on. transp-smoke on orion: 33717's three base arms and an arm that does not exist
(the fault injection: it failed with rc 1 and the run's wait reported it; the three passed in 45 s
-- 2 min 17 s, and their networks and rows came back). The opaque check: transp-base-orion and
-quadro2, the other 36 base networks (1.30 and 0.57 task-h). transp-desire-orion and -quadro2: the
desire-routed preset recomputed on 5810 and 20269 (1.41 and 0.16 task-h). Then transp-arms-orion
and -quadro2: every block's other 34 arms, 442 tasks, all passed (19.2 and 6.9 task-h; the longest
34 min, 5810's shortlist 16). A block's arms on one node: orion took 1558, 22422, 30848, 33717,
46841 and 5810, quadro2 the smaller 18895, 20269, 20543, 23597, 38616, 9712 and 9717 (another
user's jobs held 8 of its CPUs and 320 of its 376 GB). The partition allocates whole cores
(CR_CORE_MEMORY), so a single-threaded task holds both of a core's CPUs: 36 ran at once on orion
and 27 on quadro2 (at 2 GB a task), under the throttles of 64 and 40. Proposals 15.8 CPU-h
(cycle_native 8.1, resistance_lp 6.7, clearance_looped 1.1), prefixes 10.8, none failed, peak RSS
1.0 GB; the nodes ran the base networks 1.0 -- 2.9x slower than this machine, and orion, every core
busy, ran 5810's shortlist 16 in 1,902 s against the sentinel's 788 here. Priced resistance_lp's
true D is 0.201 -- 0.237 against its 0.20 cap, as on the sentinel. Scoring on the GPUs, (c) at area
D 0.05 and 0.10 only: transp-tuning-score-a (quadro2's seven blocks) and -b (orion's other five) on
three GPUs, and transp-tuning-score-1558 on one card of at least 40 GB (it peaked at 38.1 GB, as in
item 4's scoring, where the launcher estimates 30.9): 1.64 GPU-h, all on the H100s, the longest
14.8 min (1558).
- Checks. The opaque check on the nodes: all 39 base networks computed there are item 4's, WKB row
  for row and widths (18 on orion, 21 on quadro2): neither machine moves a base network. The
  desire-routed preset recomputed on the nodes, betweenness counts and all (the cluster has no
  derive cache), is item 4's network on 20269 (quadro2, 566 s; item 4's run took 581 s here) and on
  5810 (orion, 5,063 s and 3.9 GB, where item 4's timed rerun, 1,083 s, read the counts from its
  cache): the preset's item-4 networks are reused on all 13. On the sentinel's blocks all 128
  networks of the 32 arms both stages share are the sentinel's (22422, 30848 and 5810 on orion,
  9712 on quadro2): no machine moved a priced network either. Where the networks are the same, the
  reduced scoring gives the sentinel's values: (a) and B (i) exactly, (c) within 7.3e-8 at D 0.05
  and 1.6e-7 at 0.10. The stored clearings rescore to item 4's rows within 3.2e-9, the base arms'
  (a) within 1.1e-16 and their (c) within 9.0e-8.
- Short of the budget. cycle_native's whole network falls short of D 0.05 on 33717 for its base (D
  0.045 by area, 0.037 by count, as in item 4), shortlist 4, the desire-routed preset and three of
  its random sets, and of D 0.10 there for every arm; on 38616 by area for base, both shortlists
  and four random sets; 1558's shortlist 4 stops at D 0.053 -- 0.055. There (c) and (a) read the
  whole network, at less displacement than an arm that reaches the budget. Every resistance_lp and
  clearance_looped network reaches D 0.10.
- The discount moves the roads, as on the sentinel: the area-weighted share of the (c) prefix's
  true displacement on C, mean over the 13 (about 5% by chance), goes from 0.080 to 0.206 for
  cycle_native's C priced and routed, from 0.116 to 0.194 (priced) and 0.297 (priced and routed)
  for resistance_lp (C' priced 0.144), from 0.072 to 0.192 for clearance_looped (C' 0.108). A
  random set draws the roads onto itself too, less (its own share 0.14 -- 0.16 for cycle_native,
  0.09 -- 0.10 priced and 0.15 -- 0.18 priced and routed for resistance_lp, 0.12 -- 0.15 for
  clearance_looped). The random sets' area is C's within 3% on 9 of the 13 (the ratio's median
  1.0025; Jaccard overlap with C median 0.035, at most 0.094) and within 6.3% on 46841, but 0.94 --
  1.20 of it on 22422, 0.87 -- 1.19 on 23597 and 0.81 -- 0.89 on 5810, as on the sentinel.
- Lens (c) at area D 0.05, per method: each arm's mean, its mean gain over base with the
  Bayesian-bootstrap 95% interval over the 13, the blocks up in B's order, the worst block's gain,
  the median (a) loss against base at count D 0.05 | 0.10, the mean share of the gap to SIMP's
  default closed, and the share on C as above; * vetoed (on a knob, for information: the rule holds
  candidates to it, not knobs), = chosen, + goes to the held-out blocks; the random sets in each
  form as their mean. Then each candidate's and best knob's (c) gain per block, x 100. Every value
  per block is in transparency_rows/tuning/summary_choice.csv and summary_lens.csv.

    cycle_native       mean (c)    gain     95% int.    up   worst  (a) .05 |   .10    gap  on C
    base                 0.1196 +0.0000 [+.0000,+.0000]  0 +0.0000  +0.0000 | +0.0000 +0.000 0.080
    shortlist4           0.1278 +0.0081 [-.0119,+.0489]  7 -0.0368  +0.0134 | +0.0059 -0.009 0.071
    shortlist16          0.1211 +0.0015 [-.0132,+.0116]  9 -0.0670  -0.0005 | -0.0077 +0.008 0.089
    desire               0.1149 -0.0047 [-.0150,+.0065]  5 -0.0368  -0.0117 | -0.0145 -0.050 0.088
    simp-price-route     0.1538 +0.0341 [+.0103,+.0631] 11 -0.0537  +0.0142 | +0.0003 +0.129 0.206
    random price-route   0.1220 +0.0024  (seeds -0.0100 -- +0.0212), on C 0.087

    resistance_lp      mean (c)    gain     95% int.    up   worst  (a) .05 |   .10    gap  on C
    base                 0.1648 +0.0000 [+.0000,+.0000]  0 +0.0000  +0.0000 | +0.0000 +0.000 0.116
  * chunks4              0.1968 +0.0320 [+.0132,+.0666] 11 -0.0093  +0.0061 | +0.0041 +0.134 0.132
    chunks16             0.1688 +0.0040 [-.0148,+.0399]  2 -0.0430  -0.0012 | +0.0021 -0.032 0.107
    simp-price           0.1852 +0.0204 [+.0100,+.0347] 12 -0.0032  -0.0014 | +0.0003 +0.131 0.194
  * simp-price-route     0.1997 +0.0349 [-.0215,+.0872] 12 -0.2120  +0.0127 | +0.0045 +0.102 0.297
  * cheap-price          0.1519 -0.0129 [-.0560,+.0057] 10 -0.2222  +0.0044 | +0.0012 -0.079 0.144
    random price         0.1548 -0.0100  (seeds -0.0271 -- -0.0007), on C 0.117
    random price-route   0.1571 -0.0077  (seeds -0.0324 -- +0.0180), on C 0.118

    clearance_looped   mean (c)    gain     95% int.    up   worst  (a) .05 |   .10    gap  on C
  + base                 0.1020 +0.0000 [+.0000,+.0000]  0 +0.0000  +0.0000 | +0.0000 +0.000 0.072
    repulsion-2          0.0991 -0.0029 [-.0089,+.0018]  6 -0.0288  -0.0017 | -0.0013 -0.014 0.070
    repulsion2           0.0970 -0.0050 [-.0145,+.0024]  6 -0.0479  -0.0505 | -0.0230 -0.015 0.074
  + repulsion4           0.1038 +0.0018 [-.0060,+.0108]  6 -0.0212  -0.0733 | -0.0429 +0.005 0.070
  =+simp-price-route     0.1269 +0.0249 [+.0079,+.0575] 10 -0.0112  -0.0261 | -0.0165 +0.094 0.192
    cheap-price-route    0.1247 +0.0227 [+.0033,+.0567]  9 -0.0401  -0.0172 | -0.0077 +0.080 0.107
  + random price-route   0.1053 +0.0033  (seeds -0.0046 -- +0.0081), on C 0.071

    (c) gain x 100   1558 18895 20269 20543 22422 23597 30848 33717 38616 46841  5810  9712  9717
    cyc sl 4        -3.7  -1.8  +2.2  +0.4  +0.0  +0.1 +20.5  -2.1  +0.1  +0.2  -3.0  -1.2  -1.2
    cyc C p+r       +1.3  +5.4  +6.1  +2.8  +0.0 +10.1  +2.2  +2.9  -5.4 +15.3  +1.7  -0.2  +2.1
    rlp ch 4       +19.7  +7.0  +0.6  +4.4  +0.1  +0.1  +2.4  +2.7  +1.6  +1.4  +1.3  -0.9  +1.3
    rlp C p         -0.2  +7.6  +3.1  +6.4  +0.1  -0.3  +0.8  +1.9  +1.3  +3.0  +1.3  +0.4  +1.3
    rlp C p+r      +22.9  +2.6  +2.8 -21.2  +0.1  +9.7  +2.9  +3.3  +0.5 +17.6  +2.2  +0.6  +1.4
    rlp C' p        +0.5  -2.0  +0.8 -22.2  +0.0  -0.0  +1.2  +0.5  +1.8  +1.5  +0.5  -0.1  +1.0
    clr rep 4       -0.5  -2.1  +1.6  -1.3  +0.1  -0.3  +2.7  -0.2  -0.7  +3.1  -1.8  +0.1  +1.8
    clr C p+r       +1.0  -0.1  +4.5  -0.2  +0.1  +0.1  +3.7  +2.7  -1.1 +17.9  +0.3  +1.2  +2.3
    clr C' p+r      +1.0  -4.0  +4.2  -0.2  +0.0  -0.0  +3.6  +1.7  -0.2 +18.5  +2.3  +0.6  +2.0

- The rule. cycle_native: its one candidate, C priced and routed, gains +0.0341 [+0.0103, +0.0631]
  (up on 11 of 13, median +0.022), beats its best knob (shortlist 4, +0.0081, all of it from
  30848's pocket) and its random sets (mean (c) 0.1538 against 0.1220), is vetoed on no block, but
  loses a median 0.0142 of (a) at count D 0.05 (+0.0003 at 0.10) against the guard's 0.01: it
  stops, as on the sentinel (0.01015 there), not by a hair this time. Its random sets in the same
  form lose (a) too (median 0.0055 -- 0.0187), so the loss is the form's, routing through
  buildings, not the set's. resistance_lp: C priced, its main form, gains +0.0204 [+0.0100,
  +0.0347], up on 12 of 13 (median +0.013), the worst -0.0032, (a) unmoved (-0.0014 | +0.0003), far
  above its random sets (-0.0100), but below its best knob, chunks 4 (+0.0320 [+0.0132, +0.0666],
  1558 alone +0.197). C priced and routed (+0.0349) and C' priced (-0.0129) are vetoed: 20543
  collapses under each (0.336 to 0.124 and 0.114), as it does under two of the five random sets
  priced and four of the five priced and routed; resistance_lp's base opens a gate there (0.336,
  against 0.10 -- 0.11 for the other methods) that half its discounted networks miss. No candidate
  qualifies, and resistance_lp stops. clearance_looped: C priced and routed gains +0.0249 [+0.0079,
  +0.0575] (up on 10, median +0.010), the worst -0.0112, gains (a) (-0.0261 | -0.0165), is above
  its random sets (mean (c) 0.1269 against 0.1053) and its best knob (repulsion 4, +0.0018); C'
  priced and routed passes too, at +0.0227 [+0.0033, +0.0567]. Chosen: C priced and routed, which
  goes to the held-out blocks with base, repulsion 4 and its five random sets priced and routed.
- Where the call turns on the rule's letter (the owner's). resistance_lp's best knob, chunks 4, is
  itself behind base on B (i) on 20543 (it leaves 0.0009 of the population enclosed there, which
  base frees); the rule holds candidates to the veto, not knobs. Held to it, the best knob would be
  chunks 16 (+0.0040), and C priced would qualify and go on. cycle_native's (a) loss is over the
  guard at D 0.05 and nil at 0.10.
- Beside it. The gap to SIMP's default closed, mean over the 13: cycle_native's candidate 12.9%,
  resistance_lp's 13.1% (priced) and 10.2% (priced and routed), clearance_looped's 9.4% (C) and
  8.0% (C'), against the sentinel's 4 -- 10%; on gated 22422 nothing, as no arm opens it at D 0.05.
  Much of the gain is gates the discounts open: 46841 (+0.15 -- +0.18 for each arm priced and
  routed on C or C', and for one random set of five in each method's priced-and-routed form), 23597
  (+0.10 for cycle_native's and resistance_lp's priced and routed, and +0.17 for one of
  resistance_lp's random sets), 1558 (resistance_lp's priced and routed +0.23, one of its random
  sets +0.16; chunks 4 +0.20, chunks 16 +0.18). Without 46841 clearance_looped's chosen arm gains
  +0.012 a block. The same arms gained less on the sentinel's four (cycle_native's +0.0094,
  resistance_lp's +0.0064 and +0.0147, clearance_looped's +0.0132 and, C', +0.0166); the random
  sets' mean gain here: cycle_native +0.0024, resistance_lp -0.0100 (priced) and -0.0077 (priced
  and routed), clearance_looped +0.0033.

What the 13 say: making the buildings SIMP's default clears cheaper moves 1.7 -- 2.7x as much of a
road network's displacement onto them, and at equal displacement buys more than on the sentinel,
+0.020 -- +0.035 of (c) on average for the main forms and their variants (resistance_lp's C' aside,
-0.013), 8 -- 13% of the gap to SIMP's default, well above random sets made as cheap, much of it
from gates on a few blocks. Under the rule fixed before the run, clearance_looped's main form, C
priced and routed at f 0.1, goes to the held-out blocks, with base, repulsion 4 and its random
control; cycle_native stops on the (a) guard (0.0142 at count D 0.05) and resistance_lp on its own
knob (chunks 4, which the B (i) veto would rule out were knobs held to it: the owner's call).
Committed: transparency_rows/tuning/arms.csv and summary_{records,choice,lens}.csv (the rows and
networks are in the ignored transparency_rows/tuning/ parquet and scratch).

### Transparency for the road methods: the held-out blocks (owner 2026-10-12: "run the 13- and 60-block stages on orion/quadro2"; on resistance_lp, "Yes, include it")

The stage where the claims stand or fall: what came out of the 13 runs on the held-out blocks,
transparency_run.py's `held` stage, under transparency_rows/held.

The blocks. Item 4's 60 large blocks that fit at h 0.5 (the blocks of its committed lens rows,
road_rows/summary_*_lens_D0.05.csv, the same 60 as road_rescore.py `project`'s blocks under 48 GB
and as its networks) less the 13 tuning blocks: 47. 14401 has no stored clearing at h 0.5 (neither
SIMP's default nor the cheap preset's), so no C, and is left out: 46 blocks (transparency_run.HELD;
transparency_check.py `held` derives them again). No sequential stop: every one of the 46 is run.
Proposals on the CPU nodes and the reduced scoring make the whole set cheap, and a claim on all the
held-out blocks is cleaner than one on an early stop.

The arms are data, not code (transparency_rows/held/plan.csv: per method its claim and its own
knobs, each an arm of the 13; base and the claim's random sets come with them, and a method joins
by a row; arms.csv lists them all):
- clearance_looped, chosen under the 13's rule: base; its tree's repulsion 4 (its best knob on the
  13); C priced and routed at f 0.1, the claim; the five random sets matched to C (AreaMass, 20
  strata, seeds 1 -- 5), priced and routed at f 0.1.
- resistance_lp, by the owner's amendment below: base; chunks 4 and chunks 16; C priced at f 0.1,
  the claim; the five random sets, priced at f 0.1.
17 arms a block, 782 networks, every one proposed fresh.

The amendment (owner 2026-10-12, decided after the 13's results were seen). The 13's choice rule
vetoed a candidate that leaves population enclosed (behind base on B (i)) but did not hold the
method's own knobs to the same test, and resistance_lp's best knob there, chunks 4, itself leaves
people enclosed on 20543, where base leaves none. The B (i) test now applies to knobs as it does to
candidates: under B a setting that walls people in is not a fair comparison point on that block.
Held to it, resistance_lp's bar on the 13 is chunks 16 (+0.0040), and C priced (+0.0204) passes the
13's rule. As this was decided after the 13's results were seen, the held-out blocks are where
resistance_lp's claim stands or falls, and chunks 4 stays in as a comparison point.
clearance_looped's knob gets the same treatment, set before any held-out result exists: repulsion 4
is left out of rule 2 on any block where it is behind base on B (i).

Compute, as the 13. One task per (block, arm) on orion and quadro2 (cluster.py's `cpu` kind), every
arm of a block on one node. The opaque check first, on every held-out block before its arms count:
the base networks computed on the nodes, clearance_looped's and resistance_lp's, must be item 4's,
WKB row for row and widths; any difference stops the stage. The reduced scoring on the cluster's
GPUs, four at most: lens (c) at area D 0.05 and 0.10 (h 0.5, uni, B's order) and (a) at count D
0.05 and 0.10 from the prefixes. The launcher's GPU sizing is recalibrated first, GB_PER_MILLION
0.7 -> 0.95: the 13's measured scoring peaks are above 0.7 on all 13, 0.83 -- 0.87 GB per million
unknowns on the nine over 14M (0.95 is the largest of these, 22422's 0.866, with 10% to spare) and
1.1 -- 1.4 on the four under 13M (a few GB fixed, which any card holds); at 0.7 item 4's run sent
1558, 14401 and 20023 to 32 GB cards their peaks (37.5, 37.3, 33.9 GB) would not have fit, and they
ran on bigger cards by luck (cluster_check.py `gpu`, fault-injected).

B's order, as on the 13: an arm that leaves more population enclosed than base at area D 0.05 (B
(i)) is behind it on that block whatever its (c); one that leaves less is ahead, and counts among
the blocks up. Means, medians and intervals are over (c) itself.

The claim's rules, fixed before any held-out result (this commit precedes the first proposal on the
held-out blocks). For each method, over the 46, paired by block, each interval a Bayesian
bootstrap's 95% (transparency_run.boot_weights over the 46: 10,000 Dirichlet(1, ..., 1) draws
seeded 20261012, one matrix for every arm and test); an interval "excluding 0" is read as above it,
its lower end above 0, as each rule claims a gain:
1. Primary: C's mean (c) gain over base at area D 0.05, its interval above 0, and C behind base on
   B (i) on no block.
2. Against the method's own knobs, for each knob K (clearance_looped's repulsion 4; resistance_lp's
   chunks 4 and chunks 16): on the blocks where K is not behind base on B (i), C's mean (c) gain
   over K, its interval above 0 (the weights of those blocks renormalized: a Bayesian bootstrap
   over them). Reported: the blocks each K is left out on. Rule 2 holds only if it holds against
   every knob (a knob left out on every block leaves nothing to compare, and the rule does not hold
   against it).
3. Against targeting by chance: C's mean (c) above its random sets' per-block mean (the five seeds
   in C's form), the mean difference's interval above 0.
4. The shipped metric: C's median (a) loss against base at most 0.01 at count D 0.05, and at D
   0.10.
The verdict, per method: a shift of the frontier on the held-out blocks if 1 -- 4 all hold;
otherwise the rules that fail, said. Reported beside it, as description and not part of the
verdict: the median paired gain; the mean without the three blocks of largest gain (on the 13 the
gains came largely from gates on 46841, 23597 and 1558); the blocks up; the share of the gap to
SIMP's default closed (per block, as on the 13, and its mean; SIMP's (c) from item 4's rows); (c)
at area D 0.10; the share of the (c) prefix's true displacement on C; and the random draws' area
over C's and Jaccard overlap with C. The code: transparency_run.Claim, guarded by
transparency_check.py `claim` (synthetic lens tables, one case per rule, fault-injected four ways).

Run (2026-10-12), one task per (block, arm), single-threaded, every network fresh, at the rules'
commit (e573227). The opaque check first, transp-held-base-orion and -quadro2: both methods' base
on all 46 (92 tasks, 2.47 task-h, the longest 8.6 min). Then transp-held-arms-orion and -quadro2:
every block's other 15 arms (690 tasks, 20.2 task-h, the longest 12.5 min). All 782 passed. A
block's arms on one node, the blocks split by item 4's costs: orion took 24 (30796 among them),
quadro2 22; 32 and 27 ran at once (whole cores; on quadro2 another user's jobs held 8 CPUs and 320
of its 376 GB). Proposals 8.19 CPU-h (clearance_looped 2.39, resistance_lp 5.80; the longest
network 405 s, 30796's chunks 16), prefixes 11.90 h, peak RSS 0.90 GB. Scoring on the GPUs, (c) at
area D 0.05 and 0.10 only: transp-held-score, 43 tasks on 32 GB cards or larger (tesla1 and tesla2,
three on the H100s) and 20023, 20423 and 30796 on 48 GB ones (the H100s), as the recalibrated
sizing put them; at most four at once, two for most of the run (other users' jobs were ahead in the
queue); 2.83 GPU-h, the longest task 10.8 min, the GPU peak 41.1 GB (30796).
- Checks. The opaque check on the nodes: all 92 base networks are item 4's, WKB row for row and
  widths (48 on orion, 44 on quadro2). The base arms reproduce item 4's (a) within 1.1e-16, (c)
  within 4.9e-8 and B (i) within 9.2e-17, and the stored clearings rescore to item 4's rows within
  4.8e-9. Every whole network reaches D 0.10 under both weightings (priced resistance_lp's true D
  by count 0.212 -- 0.249 against its 0.20 cap). The sizing held: cluster_check.py `gpu
  transp-held-score` finds every block on a card that holds its measured peak with 5% to spare (the
  least room 30796, 41.1 GB on 48); the scorer peaked at 0.80 -- 0.87 GB per million unknowns on
  the blocks over 14M.
- The discount moves the roads, as on the 13: the area-weighted share of the (c) prefix's true
  displacement on C, mean over the 46 (about 5% by chance), goes from 0.081 to 0.224 for
  clearance_looped's C priced and routed and from 0.124 to 0.232 for resistance_lp's C priced. A
  random set draws the roads onto itself too, less (its own share 0.148 -- 0.155 and 0.098 --
  0.107). The draws' area over C's has a median of 1.0027 (p10 0.989, p90 1.027; 0.918 -- 1.264)
  and their Jaccard overlap with C a median of 0.041 (at most 0.104).
- Lens (c) at area D 0.05 per method, as on the 13: each arm's mean, its mean gain over base with
  the Bayesian-bootstrap 95% interval over the 46, the blocks up in B's order, the worst block's
  gain, the median (a) loss against base at count D 0.05 | 0.10, the mean share of the gap to
  SIMP's default closed, and the share on C as above; the random sets as their mean. Every value
  per block is in transparency_rows/held/summary_lens.csv, per arm in summary_claim.csv, per rule
  in summary_rules.csv.

    clearance_looped   mean (c)    gain     95% int.    up   worst  (a) .05 |   .10    gap  on C
    base                 0.1817 +0.0000 [+.0000,+.0000]  0 +0.0000  +0.0000 | +0.0000 +0.000 0.081
    repulsion4           0.1906 +0.0089 [-.0133,+.0222] 30 -0.3534  -0.0420 | -0.0266 +0.030 0.088
    simp-price-route     0.2157 +0.0341 [+.0230,+.0464] 39 -0.0645  -0.0099 | -0.0027 +0.130 0.224
    random price-route   0.1809 -0.0008  (seeds -0.0087 -- +0.0095), on C 0.089

    resistance_lp      mean (c)    gain     95% int.    up   worst  (a) .05 |   .10    gap  on C
    base                 0.2425 +0.0000 [+.0000,+.0000]  0 +0.0000  +0.0000 | +0.0000 +0.000 0.124
    chunks4              0.2572 +0.0146 [+.0011,+.0373] 34 -0.1098  +0.0159 | +0.0035 +0.069 0.130
    chunks16             0.2240 -0.0185 [-.0331,-.0070] 16 -0.1978  -0.0022 | -0.0019 -0.111 0.121
    simp-price           0.2738 +0.0313 [+.0171,+.0545] 38 -0.1088  +0.0100 | +0.0036 +0.157 0.232
    random price         0.2396 -0.0030  (seeds -0.0077 -- +0.0083), on C 0.128

- The rules. clearance_looped: (1) C's mean (c) gain over base is +0.0341 [+0.0230, +0.0464], up on
  39 of the 46 in B's order, but C is behind base on B (i) on one block, 5706, where it leaves
  0.052% of the population enclosed and base none (its (c) there +0.062): rule 1 fails. (2) Against
  repulsion 4, left out on the 2 blocks where it is behind base on B (i): +0.0259 [+0.0099,
  +0.0492] over the other 44, holds (over all 46, the rule as first written, +0.0251 [+0.0098,
  +0.0474]). (3) Above its random sets' per-block mean by +0.0348 [+0.0252, +0.0466], holds. (4)
  (a) gained, a median loss of -0.0099 at count D 0.05 and -0.0027 at 0.10, holds. No shift: rule 1
  fails, on B (i) on one block. resistance_lp: (1) +0.0313 [+0.0171, +0.0545], up on 38, but behind
  base on B (i) on four blocks (22336, 39240 and 43708, where base leaves no one enclosed and C
  0.054 -- 0.161% of the population, and 41673, 0.170% to 0.225%): rule 1 fails. (2) Against chunks
  4, left out on 4 blocks, +0.0163 [+0.0091, +0.0256] over 42; against chunks 16, left out on 2,
  +0.0519 [+0.0365, +0.0750] over 44: holds (unamended, +0.0166 [+0.0096, +0.0254] and +0.0498
  [+0.0349, +0.0722]). (3) +0.0342 [+0.0220, +0.0520] above its random sets, holds. (4) A median
  (a) loss of 0.010033 at count D 0.05 (23 of the 46 lose more than 0.01) and 0.0036 at 0.10: over
  the guard by 0.00003 at D 0.05, rule 4 fails. No shift: rules 1 and 4 fail.
- Beside it (description, not verdict). The gains are broad, not a few gates as on the 13: the
  claims' median gains are +0.0272 (clearance_looped) and +0.0241 (resistance_lp), their means
  without the three blocks of largest gain +0.0274 (6310, 19161 and 7662, +0.109 -- +0.157) and
  +0.0205 (44775, +0.405, a gate C opens by the jump, not inspected; 43708, 6316), and each is up
  on (c) alone on 40 of the 46. The gap to SIMP's default closed: 13.0% and 15.7% (the mean of the
  blocks'; pooled 13.9% and 17.0%), against 9.4% and 13.1% on the 13. At area D 0.10: +0.0355
  [+0.0262, +0.0459], up on 41, and +0.0355 [+0.0226, +0.0472], up on 43. The same arms on the 13:
  +0.0249 and +0.0204. The knobs: repulsion 4 +0.0089 (+0.0018 on the 13), collapsing on 17608
  (0.413 to 0.060); chunks 4 +0.0146 (+0.0320 on the 13) and chunks 16 -0.0185 (+0.0040). The
  random sets gain nothing on average (-0.0008 and -0.0030), and four of resistance_lp's five
  collapse on 43547 (0.350 to 0.085 -- 0.090). resistance_lp's C is 0.109 below base on 44602 (by
  the drop a gate base opens and C misses, not inspected), as chunks 4 and 16 and three of the
  random sets are: beyond the 13's collapse veto, which the held-out rules do not carry. Walling in
  at D 0.05 is common: behind base on B (i) on 0 -- 3 blocks for clearance_looped's random sets and
  3 -- 5 for resistance_lp's, on 2 for repulsion 4 and on 4 and 2 for chunks 4 and 16; base leaves
  population enclosed on 8 blocks (clearance_looped) and 6 (resistance_lp), the claims on 7 and 8,
  and each claim leaves less enclosed than base on 3.

What the held-out blocks say: under the rules fixed before the run, neither claim is a shift of the
frontier. Each falls on B's first criterion, a discounted network cut at D 0.05 leaving a few homes
enclosed that base frees: clearance_looped on one block of 46 (5706, 0.05% of its population),
resistance_lp on four, and resistance_lp's (a) guard by 0.00003 at count D 0.05 besides. Everything
else holds, with room: making the buildings SIMP's default clears cheaper buys +0.034
(clearance_looped) and +0.031 (resistance_lp) of (c) at area D 0.05 on the held-out blocks, broadly
(medians +0.027 and +0.024, 13 -- 16% of the gap to SIMP's default), more than on the 13 (+0.025
and +0.020), more than the methods' own knobs (by +0.026, and by +0.016 and +0.052) and than random
sets made as cheap (by +0.035 and +0.034), at (a) gained (clearance_looped) or at the guard
(resistance_lp). Committed: transparency_rows/held/plan.csv, arms.csv and
summary_{records,rules,claim,lens}.csv (the rows and networks are in the ignored
transparency_rows/held/ parquet and scratch).

### Transparency for the road methods: free first (owner 2026-10-12: "Yes, let's try to free enclosed homes first")

The held-out stage's claims both fell on B's first criterion, a discounted network cut at area D
0.05 leaving a few homes enclosed that base frees (clearance_looped on 5706, resistance_lp on 22336,
39240, 41673 and 43708), and 12 of the other 13 knob and random arms walled people in on some block
too: road networks free enclosed homes only in passing. The owner: give the road methods what B's
optimizer got, a stage that frees the homes enclosed at baseline before anything else, then the
method as before. This stage asks whether the claims hold once every arm does that. It was decided
after the held-out results were seen (the B (i) misses above, and resistance_lp's rule 4 missed by
0.000033), so it is the claims re-tested under a changed method, not a fresh test of them.

The stage (d238f7c2). reblock.free_first: a Strategy every road method holds (`free_first` beside
`price`; HoldNothing, nothing held, in every preset), whose roads the method commits before its own
first move -- cycle_native's cap and ranking count them, resistance_lp builds them before its first
chunk, clearance's tree starts from them, and clearance_looped's loops see them as part of the tree
-- and which lead its network, marked held. The research strategy, free_roads.FreeEnclosed: the
homes enclosed at baseline as the scorer defines them (h 0.5); for each, a road on the method's own
graph from the street or a road already held, routed by what its own 7 m corridor displaces
(area-weighted), ending on a node or through an edge whose corridor reaches the home's near cells or
the sealed pocket they touch; the least true displacement per unit of the home's area population
first, within the prefix budget, D 0.05 by area and by count; each held road must free its home by
the metric's own test, or the stage raises. The prefixes put the held roads first
(common.canonical_order), so every prefix at D 0.05 holds them all. Not routed through free_first's
crossing: a road through it still pays for every building its corridor touches on the way, and
21252's home 631 is reached only through a parcel diagonal across its pocket. On the ten held-out
blocks with homes enclosed at baseline (21 homes) 18 roads free every one, D 0.0009 -- 0.0058 by
area and 0.0004 -- 0.0048 by count a block, 6 -- 304 m long. The guards (transparency_check.py
`hold-nothing`, `no-enclosed`, `free-roads`, `lead`), each fault-injected, are in that commit.

The arms and the rules. Every arm of the held-out plan (transparency_rows/held/plan.csv), both
methods -- base, the knobs (repulsion 4; chunks 4 and 16), the claims (clearance_looped's C priced
and routed, resistance_lp's C priced, f 0.1) and their five random sets -- from the free-first
stage's roads (transparency_run's `free` stage, FREE: FreeEnclosed within area and count D 0.05, the
area population, h 0.5), on the same 46 blocks. Rules 1 -- 4 of the held-out stage unchanged, with
the owner's amendment (a knob behind base on B (i) is left out of rule 2 on that block), the same
bootstrap weights (boot_weights over the 46, seed 20261012), every rule judged as written: a miss by
a hair, rule 4's 0.01 included, is a fail. The verdict per method, beside the held-out stage's.

The procedure, fixed before any free-first result (this commit precedes the first). The plan
(FreeEnclosed.plan on each method's graph, what each arm's `held` computes) runs first on the other
36 held-out blocks: where it holds nothing every arm's network is the held stage's (with nothing
held a method runs as with HoldNothing: guard 2), so that block's rows are the held stage's, copied
(`reuse`); a block where it holds roads is run as the ten, and said. The ten (and any such block) on
the CPU nodes, every arm of a block on the node that ran it in the held stage (orion 21252, 22267,
22336, 41515, 41806, 5706; quadro2 38254, 39240, 41673, 43708), the base arms first, their held
roads checked against this machine's plan (WKB row for row) before the other arms run; the reduced
scoring on the cluster's GPUs ((c) at area D 0.05 and 0.10, four at most). The report checks the
base arms against item 4 only where nothing is held. On the blocks with held roads it checks instead
that every arm's network holds the plan's roads first, that its ordered network and every D 0.05
prefix lead with them, and that the scorer leaves no more population enclosed there than the plan
does (none on the ten); guard 4 (`lead`) runs on the cluster's networks beside it.

Reported beside the verdict: what free first cost each arm on the blocks it touched, its (c) at area
D 0.05 and 0.10 and (a) at count D 0.05 and 0.10 against the same arm in the held stage, paired by
block; and whether resistance_lp's rule 4, missed by 0.000033, moves.

Run (2026-10-12), from the procedure's commit (5a0b7b1f) on. The plan on the other 36 held-out
blocks (transparency_rows/free/plans.csv, on both methods' graphs, 3.3 s a block median): no home
enclosed at baseline on any, nothing held, so each reuses the held stage's rows (87 files a block,
copied: reused.csv; their records, recomputed here, are the held stage's exactly). The ten on the
CPU nodes, each on the node that ran it in the held stage: free-base-orion and -quadro2, the base
arms (20 tasks, 0.42 task-h) at 5a0b7b1f, then free-arms-orion and -quadro2, the other 15 arms (150
tasks, 3.54 task-h) at 6aaa7d33, which adds only a local check (nothing on the arms' path changed);
all 170 passed. Proposals 1.70 CPU-h (clearance_looped 0.62, resistance_lp 1.08; 1.30 in the held
stage on these blocks), the longest 175 s, peak RSS 2.71 GB (5706, against 0.47 GB in the held
stage: the stage carves on the h 0.5 raster), prefixes 1.54 h. Scoring, (c) at area D 0.05 and 0.10
only: free-score, ten tasks on 32 GB cards (tesla1, tesla2), four at once at most, 0.80 GPU-h, the
longest 17.4 min (5706), the GPU peak 10.5 GB (22267).
- Checks. The base networks the nodes computed hold this machine's plan first, WKB row for row (20
  of 20, before the other arms ran), and so do all 170; the 51 of 41673, 21252 and 43708 are, road
  for road, the networks this machine computed for the guards (free-local). Guard 4 on the cluster's
  networks (transparency_check.py `lead free`, the scorer on the CPU): on all 170 the held roads
  lead the network and its ordered network, every held road is inside both D 0.05 prefixes, and no
  home is left enclosed there. The report's own checks: on the ten every arm's B (i) at area D 0.05
  and 0.10 is 0, as the plan leaves no one; on the 36 the base arms reproduce item 4's (a) within
  1.1e-16, (c) within 4.1e-8 and B (i) exactly; the stored clearings rescore to item 4's rows within
  4.8e-9.
- Free first frees everyone: no arm leaves population enclosed on any of the 46 blocks at area D
  0.05 or 0.10. In the held stage base had left some on 8 blocks (clearance_looped) and 6
  (resistance_lp), the claims on 7 and 8, and every knob and random arm but one on some block.
- The rules, free first beside the held-out stage (each interval a Bayesian bootstrap's 95% over the
  46, paired, boot_weights). No claim is behind base on B (i) on any block (the held-out stage: 1 for
  clearance_looped, 4 for resistance_lp), and no knob either, so rule 2 runs over all 46 (the
  held-out stage: 44 for repulsion 4, 42 and 44 for chunks 4 and 16).

                               free first                         the held-out stage
    clearance_looped
      1 C over base            +0.0417 [+0.0304, +0.0552]  holds  +0.0341 [+0.0230, +0.0464]  FAILS
      2 C over repulsion 4     +0.0314 [+0.0167, +0.0540]  holds  +0.0259 [+0.0099, +0.0492]  holds
      3 C over random sets     +0.0348 [+0.0261, +0.0460]  holds  +0.0348 [+0.0252, +0.0466]  holds
      4 (a) loss .05 | .10     -0.0125 | -0.0070           holds  -0.0099 | -0.0027           holds
      verdict                  a shift of the frontier            no shift: rule 1 fails
    resistance_lp
      1 C over base            +0.0326 [+0.0187, +0.0557]  holds  +0.0313 [+0.0171, +0.0545]  FAILS
      2 C over chunks 4        +0.0241 [+0.0153, +0.0358]  holds  +0.0163 [+0.0091, +0.0256]  holds
        C over chunks 16       +0.0503 [+0.0363, +0.0719]  holds  +0.0519 [+0.0365, +0.0750]  holds
      3 C over random sets     +0.0415 [+0.0282, +0.0603]  holds  +0.0342 [+0.0220, +0.0520]  holds
      4 (a) loss .05 | .10     +0.010978 | +0.003623       FAILS  +0.010033 | +0.003623       FAILS
      verdict                  no shift: rule 4 fails             no shift: rules 1 and 4 fail

- Under the rules fixed before the run, clearance_looped's claim, C priced and routed at f 0.1, is a
  shift of the frontier on the held-out blocks once every arm frees the enclosed homes first: all
  four rules hold, rule 1 with room (+0.0417, the interval's lower end +0.0304). resistance_lp's, C
  priced at f 0.1, holds rules 1 -- 3 but not rule 4: its median (a) loss against base at count D
  0.05 moves from 0.010033 to 0.010978, further over the guard's 0.01, so it fails as written
  (0.0036 at D 0.10, as before).
- What free first cost each arm on the ten blocks it touched, the arm with free first minus the same
  arm in the held stage, paired by block: (c) at area D 0.05, mean [median; min, max], its mean at D
  0.10, (a) at count D 0.05 mean [median], and the blocks of ten where (c) at D 0.05 rose. A
  negative is a cost; the held roads take 0.0009 -- 0.0058 of the area D 0.05 budget a block (12% of
  it on 41673), so where they open nothing the arm builds less of its own.

                      (c) area D .05                     (c) D .10   (a) count D .05     up
    clearance_looped
      base            -0.0306 [-0.0293; -0.116, +0.020]   +0.0266    -0.0211 [-0.0298]    2
      repulsion 4     -0.0245 [-0.0067; -0.163, +0.035]   -0.0062    +0.0092 [+0.0044]    4
      C p+r           +0.0044 [+0.0108; -0.095, +0.125]   -0.0102    -0.0039 [+0.0056]    7
      random p+r      +0.0047 (seeds -0.0229 -- +0.0235)  +0.0082    -0.0026              3 -- 6
    resistance_lp
      base            +0.0200 [+0.0064; -0.021, +0.138]   +0.0270    +0.0052 [+0.0042]    6
      chunks 4        -0.0083 [-0.0079; -0.029, +0.006]   -0.0022    -0.0100 [-0.0160]    3
      chunks 16       +0.0238 [-0.0078; -0.042, +0.318]   +0.0357    -0.0004 [+0.0005]    3
      C p             +0.0263 [+0.0064; -0.023, +0.166]   +0.0271    +0.0054 [+0.0069]    7
      random p        -0.0074 (seeds -0.0324 -- +0.0198)  -0.0106    +0.0008              3 -- 5

- Free first costs clearance_looped's base most (-0.031 a touched block, down on 8 of the ten,
  -0.116 on 41673, whose four held roads take 12% of the budget), and its claim little (+0.004, up
  on 7): the claim's gain over base on the 46 grows from +0.0341 to +0.0417, most of it base falling
  where it builds less. resistance_lp's base and claim both gain (+0.020 and +0.026; +0.138 and
  +0.166 on 41673, not inspected), and its gain over base moves little (+0.0313 to +0.0326). Single
  blocks swing far for the knobs and random sets (not inspected: a method rerouted from a new start,
  by the size): resistance_lp's chunks 16 +0.318 and a random set -0.353 on 22267,
  clearance_looped's repulsion 4 -0.163 and a random set +0.212 on 41673.
- Beside it (description, not verdict). The claims' medians +0.0287 (clearance_looped) and +0.0241
  (resistance_lp), their means without the three largest gains +0.0339 (41673, 6310, 19161) and
  +0.0212 (44775, 22267, 6316), up in B's order on 41 and 42 of the 46; the gap to SIMP's default
  closed 15.8% and 16.8% (pooled 16.6% and 18.2%; 13.0% and 15.7% in the held stage); at area D 0.10
  +0.0275 [+0.0134, +0.0388], up on 40, and +0.0356 [+0.0218, +0.0496], up on 44. The knobs:
  repulsion 4 +0.0102 [-0.0119, +0.0233], chunks 4 +0.0085 [-0.0072, +0.0320], chunks 16 -0.0176
  [-0.0309, -0.0046]. The random sets: clearance_looped's +0.0006 -- +0.0141 (mean +0.0069, -0.0008
  in the held stage: base fell more than they did), resistance_lp's -0.0147 -- -0.0031 (mean
  -0.0089). The share of the (c) prefix's displacement on C, 0.222 and 0.231 (0.224 and 0.232).

What free first says: holding roads that free the enclosed homes first removes what failed both
claims on the held-out blocks, B's first criterion, at a price in (c) that falls mostly on
clearance_looped's base and turns no rule that held into one that fails. With every arm freeing
everyone, clearance_looped's C priced and routed is a shift of the frontier on all four rules;
resistance_lp's C priced still misses the (a) guard, by more than before (0.010978 against 0.01).
The stage was decided after the held-out results, so this is the claims re-tested under a changed
method. Committed: transparency_rows/free/arms.csv, plans.csv, reused.csv, check_lead.csv and
summary_{records,rules,claim,lens,versus,versus_rules}.csv (the rows and networks are in the ignored
transparency_rows/free/ parquet and scratch).
