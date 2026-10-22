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

### Priority-weighted P (owner 2026-10-12: "Let's see how weighted P behaves. It would be nice if it could replace J_2")

BACKLOG 4's metric decisions: P is the roadless headline, with an equity view beside it, J_2 or, if
it can replace J_2, a priority-weighted P (owner 2026-10-12: "what would happen if you first set the
current emitted by deeper buildings quadratically higher and then did P? Would you get something
like J_2?"). pweight.py and pweight_check.py (7c91250); pweight_rows/ (the rows ignored, the
summaries committed).

The quantity. P_q: home i injects w_i a_i, a_i = u0_i^q / mean_w(u0^q), u0_i its escape time at
baseline (nothing cleared, the same grid and model, area population w) and mean_w the
population-weighted mean over the homes not enclosed, so the total injection is the population's;
P_q = f_q^T L^-1 f_q, its lens 1 - P_q / P_q0. The weights are fixed at baseline, so P_q keeps P's
properties: one self-adjoint solve, monotone in the open space, convex in the clearing fractions
(the relaxation's Frank-Wolfe gap certifies a bound). The displacement stays the population's: only
the demand is reweighted. (The brief wrote a_i = (u0_i / mean_w(u0))^q, which is this times a
constant at each q, equal at q 0 and 1. A constant scales P_q and P_q0 alike, so it moves no lens
and no clearing, only the total injection, which the brief also asks to keep, and which this keeps
at every q.) The reasoning to test, not assume: on independent paths (u_i = R_i w_i) dJ_p / dR_i
goes as w_i^2 u_i^(p - 1) and dP_q / dR_i as w_i^2 u0_i^(2q), the injection entering P twice, so q =
(p - 1) / 2 matches J_p to first order: q 1/2 is J_2's analogue, q 1 (the owner's "quadratically
higher", escape time growing about as depth^2) J_3's. Shared paths break the independence; the test
measures how much.

The code. common.Demand, a Strategy beside the population, resolved where a run is configured and
handed to the scorer (common.Scorer's `demand`, required): PopulationDemand (each home injects its
population: every run before this one, injection for injection) and PriorityDemand(q) (q required,
and in the run's name: Pq0.5, Pq1). The scorer's J, its adjoint load and its baseline read the
demand's injection; the cost (clear.Clearing.cost, the displacement), B (i) and free first read the
population; clear.Catchment's load and weights read the injection (the population's on every
existing path). Every scorer and clearing built anywhere now names its demand, the population's
everywhere but here. Editing common.py changes free_roads.CODE, and with it FreeEnclosed's identity
and tag; no held road changes. relax.frank_wolfe now backs off when neither its parabola's step nor
the full step descends (the step quartered down to 1e-3) before it stops: J is nonsmooth (an edge
weight is a minimum of open fractions, its gradient a subgradient shared among tied cells), and on
41132 at eps 1e-4 the old rule stopped at iteration 8 with a third of J in its gap (bound 0.634)
where a step of 0.05 descends; with the back-off its 40 iterations reach a bound of 0.486 (relaxed
0.476, the gap left 2%; in relax220's eps 1e-2 world, 0.512). Every iterate's lower bound is valid,
so going on can only raise the bound and lower J; the fw<n> plans run as before up to where they
used to stop.

The test. The 220 small blocks (common.recipients()), h 0.5, uni, area population, B's placement
(enclosed homes scored apart). One optimizer for every objective: SIMP base
(fw0.q3.i10.t0.001.e0.0001.k1s1e-06, the presets' base plan) with free first (relax.one), on the
CPU, at area D 0.05. Four arms: P (J_1), J_2, P_{1/2} and P_1. D 0.10 is not run: a SIMP run is one
budget, and a second budget is a second run. For the three convex objectives the certified bound:
relax.frank_wolfe from x = 0 with nothing held, so it bounds every clearing within D, free first or
not; 40 iterations (relax220_p1's count), solves to 1e-4 (the bound 1e-5's to 3e-6 on 41132, in
0.85x the time), in SIMP base's eps world (1e-4: J_eps <= J for every 0/1 clearing, and tighter than
relax220's 1e-2). Every arm's clearing is scored under every objective (pweight.py `score`: one
scorer, one system per clearing, one solve per demand): the lenses of P, J_2 (lens (c), 1 - (J_2 /
J_2,0)^(1/2)), P_{1/2} and P_1; each home's escape time under the population's demand relative to
baseline at the median, p95 and max home (umed, u95, umax: tails.py's ratios of quantiles; lower is
better); B (i). For the description: the Jaccard overlap of the cleared sets, and how deep the
cleared buildings sit, relative to the block's population-weighted mean escape time at baseline:
served (the baseline escape time of the homes whose flow passes through them in the clearing,
flow-weighted: clear.Catchment's shares in the tension's eps world) and own (their own homes',
area-weighted).

The guards (pweight_check.py), each passing on the code and failing under its fault (a patch in one
process), on 19263 (three homes enclosed at baseline) and 38136 (the smallest block):
- `identity`, P_0 is P exactly: PriorityDemand(0)'s injection, field, baseline P0 and escape times,
  and P's and J_2's values on a random clearing within D 0.05, are the population's bit for bit
  (every system seeded alike: pyamg's setup draws from numpy's global RNG, which otherwise moves a
  solve by ~1e-11). Fault `no-population`, the injection a_i alone: all six differ on both blocks.
- `total`: the injection's total and the field's are the population's within 2.2e-16 relative at q
  1/2, 1 and 2 (a from 0.105 -- 1.81 at q 1/2, 0.009 -- 2.79 at q 1 and 0.000 -- 5.27 at q 2 on
  19263). Fault `brief`, the brief's literal a_i = (u0_i / mean_w(u0))^q: the total off by 4.5 --
  7.6% at q 1/2 and 30 -- 47% at q 2, kept at q 1.
- `displacement`: Clearing.cost, the population, B (i) at baseline (19263's 0.0235) and of a random
  clearing, and that clearing's area displacement under J_2, P_{1/2} and P_1 are P's, bit for bit.
  Fault `cost-from-demand`, the cost from the injection: the cost and displacement differ under
  P_{1/2} and P_1 on both blocks (the random set's D 0.031 -- 0.065 against P's 0.050).
- `gradient`: the relaxation's gradient of P_{1/2}, P_1 and P at a random fractional clearing (x in
  [0.1, 0.4]), SIMP's q 1 and 3, eps 1e-4, against central differences (step 1e-4, solves to 1e-9
  from zero, so P is second-order accurate) on six buildings: within 4.7e-8 of the largest gradient
  on both. Fault `J-population`, J weighting the population while the solve injects the priority
  demand (lam = u is then not J's adjoint): 24 -- 45% off for P_{1/2} and P_1 (P, whose injection is
  the population, unmoved) on 38136.

The questions and their criteria, fixed before any result (this commit precedes every run of the
arms). The smoke runs before it, on 41132 here, timed the items and ran the code path end to end,
scoring and report included, on that one block: its numbers were seen before this commit, after the
criteria below were written. Over the 220, paired by block, each interval a Bayesian bootstrap's
95% (transparency_run.boot_weights over the 220: 10,000 Dirichlet(1, ..., 1) draws seeded 20261012,
one matrix for every comparison):
1. Can P_{1/2} replace J_2? P_{1/2}'s clearing against J_2's, on J_2's own terms: (a) the J_2 lens,
   the mean difference's interval inside [-0.005, +0.005] or above it, read as its lower end at
   -0.005 or above (an interval that straddles +0.005 is better than one inside, and passes); (b)
   the p95 home's relative escape time (u95), the mean difference's interval's upper end at +0.005
   or below (lower is better: the same test mirrored); (c) B's order: P_{1/2}'s clearing leaves more
   population enclosed than J_2's on no block. It can if (a) -- (c) all hold. Reported beside it:
   what P_{1/2} gives up on P against P's clearing (the P lens, mean and interval), the median and
   max homes, J_2's clearing under P_{1/2}'s own lens, and P itself (q 0) under the same test: if
   P's own clearing passes too, matching J_2 says nothing for the weighting.
2. Where does P_1 land? The same table and test.
3. What the convexity buys: per block, the bound (an upper bound on the lens of every clearing
   within D) minus the rounded clearing's own lens, for P, P_{1/2} and P_1: median and quartiles,
   with the Frank-Wolfe gap left at the end beside it (the bound's own slack, not the clearing's);
   J_2 has none.
4. Description: the pairwise Jaccard (median, quartiles); the served and own depths per arm (median
   over blocks), and each weighted arm's against P's (mean difference, interval).
pweight.report computes them.

Compute: one task per (block, item) on orion and quadro2 (cluster.py's `cpu` kind), single-threaded,
every item of a block on one node: simp-<arm> for the four arms and fw-<arm> for the three convex
ones, 1,540 tasks, the blocks split by an estimate of their cost (orion 116, quadro2 104); the
scoring here, 8 processes. Smoke on 41132 (138 buildings, the 220's median) here: SIMP base 152 --
247 s an arm, the bound 11 min, peak RSS 0.7 GB, the scoring 49 s; 5618 (759 buildings, the
largest) 34 min for SIMP on P and 47 on J_2.

Run (2026-10-13), at the criteria's commit (1d7f25e): pw-orion (116 blocks) and pw-quadro2 (104),
1,540 tasks, all passed, 406 task-h: SIMP 125.5 (P 28.3, J_2 40.3, P_{1/2} 28.7, P_1 28.3), the
bound 280.4 (P 97.2, P_{1/2} 92.1, P_1 91.2), 2.2x the four SIMP runs together; 5 h 44 min on orion
and 6 h 40 min on quadro2 (33 -- 36 and 32 at once, whole cores). The longest task, the bound for P
on 63718 (372 buildings), 5.5 h: late iterations ~750 s each, 4x its P_q bounds on the same block
(not inspected); 5618's bounds 2.6 h. Peak RSS 2.44 GB. The scoring here, 8 processes, 3.7 CPU-h
(5618 8 min).
- Checks. The scorer reproduces every clearing's own lens, the optimizer's exact score, within
  2.3e-9. The nodes' clearings are this machine's: 41132's four (quadro2) cleared set for cleared
  set, their lenses within 4.3e-13, its bound for P within 3e-7; 52727's P and J_2 (quadro2) and
  5618's (orion) within 7.2e-11. B (i): 24 of the 220 have homes enclosed at baseline, free first
  frees every one on every arm (D 0.0007 -- 0.0139 a block), and no clearing leaves anyone
  enclosed, so (c) holds for every comparison. Every rounded clearing's lens is below its bound, on
  all 220, for all three. Frank-Wolfe ran its 40 iterations on all but 4 (P), 3 (P_{1/2}) and 8
  (P_1) blocks (26 -- 29 at the least).
- Every clearing under every lens, median over the blocks (u: escape time relative to baseline,
  lower is better; means in summary_lens.csv):

      clearing   P       J_2     P_{1/2} P_1     umed    u95     umax
      P          0.3945  0.4121  0.4443  0.4710  0.654   0.558   0.571
      J_2        0.3966  0.4191  0.4516  0.4877  0.686   0.549   0.531
      P_{1/2}    0.3927  0.4206  0.4576  0.4968  0.689   0.555   0.523
      P_1        0.3846  0.4115  0.4567  0.5002  0.703   0.556   0.501

  Each objective's clearing leads its own lens's median but P's (J_2's clearing's median P lens is
  above P's own; paired, P's own leads by +0.0015 [-0.0001, +0.0032]) and J_2's (P_{1/2}'s
  clearing's median J_2 lens is above J_2's own; paired, in 1).
- 1. P_{1/2}'s clearing minus J_2's, paired means [95% interval]: the J_2 lens +0.0009 [-0.0013,
  +0.0036] (ahead on 87 blocks; the same clearing on 47): (a) holds. The p95 home's relative
  escape time +0.0011 [-0.0028, +0.0050003]: (b) fails, its upper end 3e-7 above +0.005. Never
  behind on B (i): (c) holds. **As written, P_{1/2} does not replace J_2**: it misses (b) by 3e-7
  of relative escape time. Description, not verdict: that end moves by about 5e-5 with the
  bootstrap's draws (under 20 other seeds 0.00485 -- 0.00500, above 0.005 under 1; with 200,000
  draws 0.00493), so the miss is inside the interval's own Monte Carlo error: the owner's call, as
  cycle_native's 0.00015 on the sentinel's guard was. Beside it: the median home +0.0054 [+0.0007,
  +0.0099] (P_{1/2} worse there), the max home -0.0197 [-0.0353, -0.0072] (better); on P against
  P's clearing it gives up 0.0028 [0.0010, 0.0046], where J_2's clearing gives up 0.0015 [-0.0001,
  +0.0032]; J_2's clearing sits 0.0076 [0.0051, 0.0108] below P_{1/2}'s on P_{1/2}'s own lens. P
  itself (q 0) under the same test fails both: the J_2 lens -0.0060 [-0.0089, -0.0031], u95
  +0.0156 [+0.0114, +0.0197]. The weighting does the work: q 1/2 closes P's gap to J_2 on J_2's
  lens entirely and 93% of it at the p95 home.
- 2. P_1's clearing minus J_2's: the J_2 lens -0.0093 [-0.0135, -0.0050], u95 +0.0070 [+0.0014,
  +0.0132]: (a) and (b) both fail. It overshoots toward the deepest homes: the max home -0.0293
  [-0.0467, -0.0144] (better than J_2's), the median home +0.0293 [+0.0229, +0.0363] (worse), and
  it gives up 0.0168 [0.0136, 0.0207] of P against P's clearing. The first-order matching holds in
  order and about in place on real blocks, shared paths and all: at the median home P is ahead of
  J_2 (-0.021) and P_{1/2} and P_1 behind it (+0.005, +0.029); at the max home P is behind (+0.031)
  and P_{1/2} and P_1 ahead (-0.020, -0.029); on J_2's lens and at the p95 home q 1/2 lands within
  0.0011 of J_2, where q 0 and q 1 both fall behind it (-0.006 and -0.009 on the lens, +0.016 and
  +0.007 at the p95 home).
- 3. The bound against the rounded clearing, medians [quartiles] over the 220 (the bound's own
  slack: its value minus the relaxation's at the last iterate; grey: the relaxed optimum's share of
  buildings partly cleared):

      objective  rounded  bound   gap                      slack   grey
      P          0.3945   0.5833  0.1749 [0.1473, 0.2137]  0.0111  0.65
      P_{1/2}    0.4576   0.6436  0.1619 [0.1319, 0.1996]  0.0090  0.62
      P_1        0.5002   0.6874  0.1469 [0.1074, 0.1780]  0.0079  0.59
      J_2        0.4191   none (not convex)

  The gap is 24 -- 32% of the bound and nearly all of it the relaxation's, not the Frank-Wolfe's
  (slack ~0.01): the relaxed optimum is grey, partial clearing everywhere, as relax220 found at D
  0.10 (0.15 above the greedy there). What the convexity buys: a certificate that no clearing
  within D beats SIMP's by more than 0.15 -- 0.17 of the lens on the median block, valid everywhere
  (every rounded clearing below it), but loose, since it bounds the grey relaxation, not 0/1
  clearings; J_2 has none. It cost 280 task-h.
- 4. How far apart and where. The Jaccard overlap of the cleared sets, median [quartiles]: P--J_2
  0.69 [0.56, 0.84], P--P_{1/2} 0.64 [0.50, 0.75], P--P_1 0.50 [0.38, 0.62], J_2--P_{1/2} 0.73
  [0.54, 0.88], J_2--P_1 0.57 [0.43, 0.70], P_{1/2}--P_1 0.67 [0.53, 0.84]; the same clearing as
  J_2's on 49 blocks (P), 47 (P_{1/2}) and 21 (P_1). J_2's nearest clearing is P_{1/2}'s. Depth,
  relative to the block's mean escape time at baseline, median over blocks (each against P's,
  paired mean [interval]): served (the homes whose flow crosses the cleared buildings) P 1.64, J_2
  1.69 (+0.057 [+0.041, +0.076]), P_{1/2} 1.74 (+0.140 [+0.112, +0.177]), P_1 1.84 (+0.251 [+0.212,
  +0.305]); own P 1.00, J_2 1.07 (+0.091 [+0.065, +0.123]), P_{1/2} 1.14 (+0.215 [+0.172, +0.271]),
  P_1 1.28 (+0.389 [+0.327, +0.473]). The weighted objectives clear deeper, P_{1/2} deeper than J_2
  and P_1 deeper still, with as many buildings (a median of 10; P_1 9).

What the 220 say: P_{1/2} lands on J_2. On J_2's own lens its clearing is not behind J_2's own
optimizer (+0.0009 [-0.0013, +0.0036]), and at the p95 home it is within the 0.005 margin but for
3e-7 at the interval's end, so as written it does not replace J_2, by a miss inside the bootstrap's
noise (the owner's call, made 2026-10-13: P_{1/2} is the equity view beside P). Where it differs
from J_2 it leans further into the tail (better at the max home by 0.020, worse at the median by
0.005, its clearings deeper), gives up a little more P (0.0028 against J_2's 0.0015), and alone of
the two carries a certified bound (0.16 above its clearing on the median block, nearly all of it
the grey relaxation's). P_1, the literal "quadratically higher", overshoots: worse than J_2 on
J_2's lens and the p95 home, best of all at the max home, worst at the median. P itself fails J_2's
test on both counts, so the match is the weighting's doing, and the first-order rule
q = (p - 1) / 2 holds about in place on blocks with shared paths. Committed:
pweight_rows/summary_{lens,questions,bound,describe,cost}.csv (the rows are in the ignored
pweight_rows/ parquet), and the four runs' clearings in the stored clearings (clearings_export.py
export: clearings/pweight_rows/simp/, all 220 blocks each, the store now 61,351 rows).

### The stored clearings, committed (owner 2026-10-13: "Yes to everything")

BACKLOG's "The stored clearings made durable". The optimizers' rows (relax_rows/, polish_rows/,
clear_rows_*/, pweight_rows/) are git-ignored, one parquet a block, and were rewritten in place: the
free-first runs of 2026-10-11 20:17 overwrote the default's and the cheap preset's rows of 12
blocks. research/roadless/clearings/ now commits each clearing's core (clearings.py: the format, its
writer and its reader; clearings_export.py fills it from the rows; clearings_check.py checks it
against the real blocks and the scorer; tests/test_clearings.py).

What is there. 151 runs on 303 blocks (the 220 small, the 82 large, 5810@major), 60,715 rows and
538,595 positions, from every row file of relax_rows (1,947), polish_rows (199), clear_rows_*
(3,994) and pweight_rows/simp (244). A run is one directory's rows of one power, its id the rows'
path under research/roadless (relax_rows/uni/<plan>/D0.05.p2): 70 SIMP runs and the relaxation's
rounding of the 3 Frank-Wolfe plans (`.fw`); 6 polish runs; 43 clear_rows_* runs (the greedies, the
SIMP path, grow-prune, the floating search, the swings); pweight_rows/simp's 4 objectives, in flight
(60 -- 62 of 220 blocks each when exported: re-run the export when pweight.py is done); 23 sets the
studies gathered into parquets (fine_clearings, ov_clearings, walls_clearings, ...), each equal to a
stored run's; and 2 runs of rows only the cluster launcher still holds (below). runs.csv gives each
run's settings as its name encodes them (power, population, along, the search's along, mesh, budget,
demand), its source files and bytes, when it was written, its scorer (`metric`) and its blocks, rows
and positions.

Format. CSV, the positions space-separated: a file a run, a line a (block, step) with D, `perm` (the
run's lens at that clearing as the optimizer stored it: Lens A at D = budget for the power-2 area
runs), the positions, `enclosed` and free first's `free_D` and `free_cleared` where the row has
them, and the timings. A greedy's rows are nested (a step's positions add to the last's), a swing's
each a whole clearing. 8.6 MB, 3.5 MB as git stores it: about what the rows weigh as parquet (3.3 MB
zstd against 3.0 MB zlib'd CSV in a trial), with no exception to the .gitignore that ignores
*.parquet, a rewritten row a changed line in git diff (the audit the overwrites lacked), and
readable without a library; the floats are reprs and read back exactly. Left out: SIMP's design x
(`simp_x`, the Frank-Wolfe iterate `x`: a float a building, 2.6 MB compressed), the machine's name
and memory, and the stores that hold no clearing (road_rows, transparency_rows: networks;
insulator_rows, res_rows, rows_h0.5_ell3_K8*: scores; pweight_rows/fw, /score: bounds and scores).
The export names each with its reason and raises on a store or a column it does not know.

The ids. blocks.csv pins each block by its building count and the SHA-256 of its outlines' WKB in
row order (clearings.fingerprint, through common.build_blocks); Store.cleared(run, block, step=)
hashes the block it is handed and raises StaleClearing on a count or a fingerprint that is not the
committed one, so a block rebuilt with its buildings reordered, moved, added or dropped cannot be
handed another block's positions. The guards are fault-injected in tests/test_clearings.py. That the
positions still index the buildings that made them: every row's D recomputed from its positions
(area share, or building share for the count runs) equals the row's own to 1e-9 on all 60,715
(clearings_check.py blocks), and the 159 block banks the cluster runs were shipped
(.cluster_submit/runs/*/bank.pkl, 2026-10-03 to 10-12) hold all 303 blocks 2,230 times, each with
today's count and fingerprint (`banks`).

Which rows predate which metric change (runs.csv's `metric`: preB, B or mixed; a row with `enclosed`
is B's):
- lifted.Placement (7913fe4, 2026-10-11 16:21: B is the metric). Before it a home with no ring cell,
  shadowed by the nearest-building rule or enclosed, injected nothing; since, a shadowed home
  injects on its open side and the enclosed are scored apart. 145 runs are preB, pweight_rows' 4 are
  B, and the default's and the cheap preset's D 0.05 runs are mixed (49 and 47 preB rows of
  2026-10-06 and 10-08, 12 B rows each, the free-first runs of 20:17). A preB row's Lens A is
  today's where the block has no shadowed home, and is not where it has. Rescored today
  (clearings_check.py rescore, CPU, 16 rows on each of three small blocks): on 4408 (263 buildings,
  none enclosed or shadowed) and 19537 (247, four enclosed, none shadowed) 11 of the 14 preB rows
  match their stored Lens A to 5.5e-8 (SIMP and its rounding, the sightline runs and the greedies'
  steps among them) and the rest are M4's, below; on 19570 (312, one enclosed, two shadowed) all 14
  move, 13 down by 3.7e-4 to 1.7e-2 (the sightline rows most); the pweight rows (B) match to 3.1e-11
  on all three. On the 60 large blocks the stored clearings moved by a median 7e-5, at most 1.6e-3
  ("Re-scoring the road methods: the 60 large blocks under B").
- The three M4 runs (clear_rows_M4_h0.5, _area, _area_p2: the first greedy, written or migrated to
  the `cleared` format 2026-09-30 14:12) are off today by 2e-6 -- 4.5e-4 on 4408 and 19537, though
  their P0 (to 1e-12) and D agree with later runs': the score of the clearing itself differs. The
  ten other greedy-format runs tried (B0.01g3 of 09-30 14:52 on) match there. The cause is not
  isolated; the nearest change is the batched greedy's (d88265e, 15:40 that day: bbox-rasterized
  labels, overlapping footprints to the later one). Read these runs' Lens A to no better than 5e-4.
- Nothing else is known to have moved a score: the demand Strategy (2026-10-12) is the population's
  own, injection for injection.

The 12 blocks' overwritten rows are not gone. The launcher kept the inputs of run road-b (2026-10-11
16:23: a tar a large block, its stored clearings among them,
.cluster_submit/runs/road-b/remote/inputs/<block>.tar; road-b-r1 shipped the same bytes), and in
them the default's and the cheap preset's rows of 20543, 21252, 22267, 22336, 23597, 38254, 39240,
41515, 41673, 41806, 43708 and 5706 stand as written (2026-10-06 03:24 -- 05:45 and 10-08 17:05). Of
the 801 row copies in the input tars of 24 cluster runs these 24 rows are the only ones that differ
from the local row, and none is missing locally (clearings_export.ShippedFamily takes any that does
or is). They are committed as D0.05.p2@road-b, with SIMP's grey and the timings. Their positions are
the `default` and `cheap` sets of fine_clearings and the `simp_default` and `cheap` of
walls_clearings, their Lens A walls' stored_perm to the last digit, and free first moved them by a
median -0.0030 (the default) and -0.0022 (the cheap preset) at a median Jaccard of 0.88 and 0.83, as
"B's optimizer: free first" has it.

Re-running. The export adds the rows the store lacks, keeps those the sources lost, and raises on a
row that differs (the in-place overwrite) unless its run is named in --replace, when the old row is
left in git's history; a block whose footprints changed raises too. The optimizers still write in
place (their no-overwrite guard is the other half of BACKLOG's item), so run the export after every
batch.

### Uniform conductance with an effective width (owner 2026-10-12: "conductance be some monotone function of the distance to the nearest footprint")

BACKLOG 4, decision (3): uniform or sightline conductance. Under U a cell conducts by its open
fraction, so a channel conducts in proportion to its width and a 1 m gap between shacks conducts per
metre of width what the middle of a plaza does. The owner wants corridors to form in the flow (a
cleared building can open several), not the clearing to look like lanes, and asked for a uniform
conductance justified by open ground conducting more the farther it is from buildings ("resistance
not just within buildings but radiate out from them, or just have conductance be some monotone
function of the distance to the nearest footprint"; "any promising variants can remain in
consideration, pending its results"). width.py and width_check.py (dfb5f3ea -- 13d64415);
width_rows/ (the rows and maps ignored, the summaries committed). Every lens is reported for P,
P_{1/2} (the equity view beside P, decided 2026-10-13) and J_2 (lens (c), the measure P_{1/2} stands
in for).

The variants. lifted.CellConductance, a Strategy beside the along conductance where the conductance
is built (Params.cells, required): what an open cell conducts by, in place of its open fraction in
its turning edges and, as the smallest over the cells a step touches, in its along edges. Which
cells are in the operator does not change; every factor is above 0 there, at most the open fraction
and growing as space is freed, so Rayleigh monotonicity holds.
- U (OpenFraction): today's.
- B_r (Blurred; r 0.5, 1, 1.5 m): each sub-sample of the fine lattice (h / 4) conducts by min(its
  openness, B), B the open share of the sub-samples within r of it (outside the block counts open:
  every ring of a kblock is a street), and a cell by the sum over its sub-samples, over 16. At a 0/1
  state the min is openness x B. In the relaxation B is affine in the clearing fractions (a cell's
  cleared part spread evenly over its footprint sub-samples, exact for a cell under one building),
  so the factor is a sum of minimums of affine functions, concave, and P stays convex. The form the
  brief wrote, B sampled at each cell's centre (as the factor, or min(o, B)), was built first and
  dropped: a cell half in a narrow channel then conducts about as much as one wholly in it, so how
  many cells a channel touches sets its conductance, the flicker the sub-cell fractions removed
  ("Results so far"); on the corridor below eight sub-cell shifts of the lattice move P by up to
  1.73x at W 1 m, against U's 1.12 (width.BlurCentre, kept as the diagnostic).
- D_s (WallDistance; s 0.5, 1 m): each open sub-sample conducts by min(1, d / s), d its distance to
  the nearest footprint sub-sample still standing (the fine lattice's distance transform, less half
  a sub-sample), summed over the cell. The distances move with every clearing, so the factor is not
  affine in the clearing fractions: scoring only, here. What SIMP would take: a fractional distance
  (each footprint standing with weight 1 - x_j, a soft minimum or the expected distance) and its
  vjp, run as a local method as SIMP is under sightline (not convex, no bound). B_r is its convex
  relative: the disk's open share is a smooth ramp in the distance to a straight wall, 1/2 at it.
- S: ss100k2n2r30, today's sightline. S_t (lifted.TurnBoosted): the same with every turning edge
  times min(F_k, F_k+1) ("The sightline metric's angle error"), for scoring and maps (no vjp).
lifted.conductance_of names each in a run: `<along>+<cells>` (uni+b1, uni+d0.5, ss100k2n2r30t).

The guards (width_check.py), each failing under its fault (a patch in one process):
- `identity`: U reproduces 57 numbers computed at 3f74f5fa, the commit before Params.cells, bit for
  bit, on 38136, 19263 (three homes enclosed) and 19570 (one enclosed, two shadowed): the scorer's
  baseline and placement, a random clearing's operator, P, J_2 and P_{1/2}, the relaxation's value
  and gradient (SIMP q 3 on J_2, q 1 on P, eps 1e-4), the greedy's tension, current_map.flow, and
  the sightline metric's layers and P. Fault `ulp` (U's factor one ulp up): 16 of the 19 per block
  differ.
- `operator`: U's edge families equal the open fractions' definition assembled independently, bit
  for bit (the baseline and a random clearing, 38136 and 19570). Fault `b1` (U's factor B_1's).
- `gradient`: B_r's relaxation gradient against central differences (r 0.5 -- 1.5, P and J_2, SIMP q
  1 and 3; 38136, 19263, 19570): within 2.8e-6 of the largest at step 1e-5. At 1e-4 one difference
  straddled a kink of the per-sub-sample min (building 5 of 38136: 1.1e-4 off, 4e-10 at 1e-5); the
  factor is piecewise linear, each sub-sample's openness against its disk's. Fault `no-blur-vjp`
  (the disk's term dropped): 0.2 -- 1.2 off.
- `convex`: along 4 random segments across the box, every cell factor's second difference at most
  8e-14 and P's at least 4.6e-3 of P (38136, 19263, every r). Fault `product` (openness x B, equal
  at every 0/1 state): the factors' second differences up to 0.051 and P's down to -4.1e-3 of P, so
  that form would cost SIMP its convexity and Frank-Wolfe its bound.

**Q2: the corridor.** The W x 40 m corridor (checks.channel) with its walls made footprints 3 m
thick and a dead end behind the demand (width.walled_channel, so a factor that looks past a cell
sees walls, not the open street side of a block edge; its lattice is checks.channel's, and U's P
equals the old check's bit for bit). P's max / min over the 13 angles 0 -- 45 deg (K 8, ell 3 m, h
0.5):

    W       U     B0.5  B1    B1.5  D0.5  D1    S     S_t
    1 m     1.66  1.85  1.73  1.72  2.39  2.39  5.32  5.09
    1.5 m   1.33  1.41  1.40  1.38  1.70  1.80  2.40  1.74
    2 m     1.15  1.19  1.21  1.19  1.34  1.48  1.67  1.26
    4 m     1.06  1.07  1.07  1.07  1.15  1.21  1.17  1.13

U and S reproduce "The sightline metric's angle error" (1.66 / 1.33 / 1.15 / 1.06; 5.32 / 2.40 /
1.67 / 1.17), S_t its turning-boosted column (5.08 / 1.73 / 1.26 / 1.13). B_r keeps about U's error
(+0.06 -- +0.19 at 1 m, +0.01 -- +0.08 from 1.5 m); D_s adds +0.7 at 1 m and +0.2 -- +0.3 at 2 m:
its factor ramps steeply across a narrow lane, and an oblique step's weight is the smallest over the
cells it touches, the one nearest a wall. S_t mends sightline's 1.5 -- 2 m lanes, not its 1 m ones
(the ray directions).

Conduction (1 / P) over U's at the same width, a straight corridor at 0 deg (30 deg alike; S's and
S_t's are boosts, not effective widths):

    W        0.5   0.75  1     1.5   2     3     4     6
    B0.5     0.58  0.72  0.76  0.82  0.85  0.90  0.92  0.95
    B1       0.31  0.45  0.57  0.68  0.74  0.81  0.85  0.90
    B1.5     0.21  0.31  0.40  0.56  0.65  0.74  0.79  0.86
    D0.5     0.23  0.37  0.43  0.56  0.64  0.73  0.79  0.85
    D1       0.12  0.19  0.22  0.31  0.39  0.52  0.61  0.71
    S        1.16  1.23  4.59  5.38  5.94  7.16  8.43  10.8
    S_t      1.16  1.23  4.62  5.52  6.26  8.09  10.2  14.5

Below 2 r (2 s) the ratio about doubles with the width (from 0.5 to 1 m x1.8 -- 1.9 for B1, B1.5,
D0.5 and D1): the lane conducts like width^2. Beyond, it nears 1 as 1 - 2 delta / W, delta the
margin lost at each wall: at 6 m 0.16, 0.30, 0.42 m for B0.5, B1, B1.5 (the continuum's 2 r / 3 pi:
0.11, 0.21, 0.32) and 0.46, 0.88 m for D0.5, D1 (s / 2: 0.25, 0.50); at 30 deg 0.20, 0.36, 0.48 and
0.70, 1.22. The lattice adds 35 -- 85% to the continuum's margin, the smallest-over-a-step rule
again. Pedestrian engineering's shy distance from a wall, about 0.5 m (Fruin; the HCM's effective
width), lies between B1.5's margin and D0.5's, and the two have nearly the same response at 0 deg
(0.21 / 0.40 / 0.65 / 0.86 against 0.23 / 0.43 / 0.64 / 0.85), D0.5 losing more at 30 deg. Viscous
channel flow, the brief's second justification, would grow as W^3 (plane Poiseuille); the variants'
width^2 below 2 r is a margin of fixed depth at each wall, a boundary layer, not a parabolic
profile. Sightline has no width law but a threshold: its rays need a whole row of open cells, so a
straight corridor gains x3.7 on U between 0.75 and 1 m.

Alignment on the corridor (0 deg; the lattice shifted k / 8 of a cell on both axes, k 0 -- 7; P's
max / min at W 0.75, 1, 1.5, 2 m): U 1.06, 1.12, 1.07, 1.05; B_r 1.07 -- 1.08, 1.07 -- 1.12, 1.05 --
1.07, 1.03 -- 1.04 (as U, or better from 1.5 m); D_s 1.22, 1.11, 1.10, 1.12 -- 1.16; S 4.85, 1.85,
1.56, 1.45 and S_t 4.85, 1.87, 1.58, 1.48 (the row threshold); B at the cell's centre 1.08 -- 1.78.
The shift across the corridor does it; along it P moves 1.01 -- 1.10.

**Q1: do corridors form in the flow?** On the sentinel (22422, 30848, 5810, 9712), under every
variant, the flux per cell (current_map's layers: the sum over headings of |signed flux|) at
baseline and under SIMP's default (the stored row), each against the no-buildings prior (every
inside sub-sample open, the same homes' demand to the same edge, under the same variant), mapped as
the atlas pages do (delta, the log of a cell's share of all walking over its share in the prior:
width.py `maps`, PNGs in width_rows/maps/, ignored, unpublished). Three indices beside the maps
(width._indices):
- the open-lane lift, the share of the flux in cells at least 2 m (and 1 m) from a standing
  footprint over those cells' share of the open area: 1 when walkers use open ground as they use the
  rest, above 1 when they keep to lanes and plazas. Under U a cell conducts per metre of width alike
  wherever it is, so the lift is where the geometry alone sends walkers; a variant that makes a
  gap's middle conduct more than its edge raises it by construction. It measures the intended
  effect, not its realism (the wiki's footpath-conductance page: a criterion a variant is built to
  move cannot judge it); the question here is how much of the flow reorganises, not whether it
  should.
- the coherence of the net flux over a square of half-side 2 m (sum over cells of the resultant's
  length over the sum of the lengths, flux-weighted): 1 for parallel streams, lower where they
  split, merge or cross. It saturates near 1 on every block (0.96 -- 0.98: flow in a gap is parallel
  to it).
- the concentration: the share of the flux carried by the 10% of the open area carrying the most per
  unit area. Corridors concentrate flow; so does any bottleneck.

The open-lane lift at 2 m (summary_q1_indices.csv; base: nothing cleared, SIMP: its default):

    block  field  U      B0.5   B1     B1.5   D0.5   D1     S      S_t
    22422  base   0.885  0.890  0.893  0.896  0.900  0.914  0.934  0.925
           SIMP   0.909  0.914  0.917  0.921  0.924  0.938  0.954  0.945
    30848  base   0.832  0.847  0.858  0.869  0.879  0.925  0.983  0.986
           SIMP   0.992  1.007  1.019  1.030  1.038  1.083  1.129  1.125
    5810   base   1.070  1.091  1.105  1.118  1.131  1.187  1.252  1.233
           SIMP   1.112  1.130  1.143  1.154  1.165  1.216  1.269  1.251
    9712   base   0.940  0.948  0.954  0.959  0.964  0.986  1.023  1.004
           SIMP   0.960  0.967  0.972  0.976  0.981  1.000  1.028  1.012

The variants move the flow onto open ground in their order of strength, B_r least (+0.005 -- +0.05),
D1 most of the uniforms (+0.03 -- +0.12), sightline most of all (+0.04 -- +0.18), and most on the
dense blocks (30848, 5810); SIMP's clearing raises the lift under every variant by about what it
does under U (30848 +0.16, the others +0.02 -- +0.04). The concentration follows (5810 at baseline:
U 0.270, B1.5 0.281, D1 0.308, S 0.339, S_t 0.323; on 9712 sightline lowers it, 0.303 to 0.261) and
the coherence does not move (within 0.006 of U's everywhere). The maps say why (5810's zoom,
ZAF.9.3.1_1_5810_variants_zoom.png): under U the flow against the prior already follows the main
alleys to the edge; B_r and D_s sharpen those same lanes and dim the gaps between them, D1 most, and
add none; S and S_t draw the clearest network of lanes, S with axis-aligned streaks across the open
field and the centre (its ray directions), S_t without them. So the uniforms make walkers keep to
the lanes U already uses; corridors that U's flow lacks do not form under them at these radii.

**Q3: grid alignment on blocks.** SIMP's default's level at the metric's lattice offset and at the
shift that moved it most under U in offset_check.py's run (20543 and 38366: (0.12, 0.94); 17608:
(0.62, 0.44)), every variant (width.py `score-all <id> shift`). The change from the metric's offset
to the shift (summary_q3_levels.csv):

    block  lens     U       B0.5    B1      B1.5    D0.5    D1      S       S_t
    20543  P        -0.045  -0.145  -0.148  -0.146  -0.148  -0.179  -0.213  -0.228
           P_{1/2}  -0.234  -0.488  -0.444  -0.412  -0.351  -0.261  -0.191  -0.167
           J_2      -0.147  -0.414  -0.391  -0.370  -0.335  -0.295  -0.229  -0.206
    38366  P        -0.009  -0.082  -0.077  -0.088  -0.062  -0.092  -0.120  -0.149
           P_{1/2}  -0.085  -0.577  -0.631  -0.679  -0.572  -0.663  -0.652  -0.679
           J_2      -0.093  -0.490  -0.536  -0.581  -0.520  -0.608  -0.636  -0.660
    17608  P        -0.047  -0.098  -0.117  -0.116  -0.081  -0.076  -0.057  -0.057
           P_{1/2}  -0.060  -0.059  -0.047  -0.039  -0.022  -0.017  -0.010  -0.009
           J_2      -0.062  -0.077  -0.069  -0.056  -0.033  -0.026  -0.015  -0.012

Every variant moves P more than U does (7 -- 17x on 38366, 3 -- 5x on 20543, 1.2 -- 2.5x on 17608).
The equity lenses move 5 -- 8x U's on 38366 (P_{1/2}: 0.085 under U, 0.57 -- 0.68 under every
variant, its lens at the shift down to 0.05 -- 0.27 from U's 0.46), up to 2.8x on 20543 (B_r and
D_s; sightline moves P_{1/2} there less than U, 0.17 -- 0.19 against 0.23), and less than U on
17608.

Who carries it, per home (width.fine_bottleneck: at baseline, per cell, the widest path of open
sub-samples to the street, 4-connected on the fine lattice, a sub-sample's clear width twice its
distance to the nearest footprint; a cell is behind a sub-cell gap when that path is at most 0.5 m
wide and narrower than the cell's own clear width; it is behind a phantom passage when there is no
such path at all, its open sub-samples joined to the street only through the operator's cell graph,
which links two 4-adjacent cells that each have an open sub-sample whether or not their open parts
touch). On 38366 at the shift three homes reach escape times of 1,265 -- 2,193 under B0.5 (131 --
219 under U; the median home 8.6): 22 -- 58% of each one's injection sits in cells behind a phantom
passage, at a potential of 570 -- 588 under U and 6,007 -- 6,047 under B0.5, while the same homes'
cells behind a real crack one sub-sample wide sit at 1 -- 49 under either. The pocket's potential
rises 10.3x: the phantom link runs through cells whose open sub-samples hug a wall, where B0.5's
disk is nearly all footprint, so the link conducts about a tenth of what U lets it. SIMP's default
leaves those pockets as they are, and 38366's J_2 lens at the shift is 0.03 under B0.5 against U's
0.37. Per block, the homes with over 5% of their injection behind a sub-cell gap or a phantom
passage at either alignment are 15%, 15% and 10% of the population (20543, 38366, 17608), and those
behind a phantom passage alone 2.4%, 3.9% and 4.7%. Their share of J_2's baseline, at the metric's
offset and the shift (summary_q3_homes.csv):

    block  homes behind      U            B0.5         B1.5         D1           S
    20543  sub-cell or both  0.60 / 0.33  0.94 / 0.41  0.95 / 0.52  0.98 / 0.73  1.00 / 0.90
           phantom only      0.42         0.91         0.91         0.94         0.95
    38366  sub-cell or both  0.31 / 0.43  0.44 / 0.96  0.71 / 0.96  0.84 / 0.96  0.93 / 0.98
           phantom only      0.17         0.27         0.48         0.58         0.66
    17608  sub-cell or both  0.63 / 0.61  0.84 / 0.72  0.94 / 0.88  0.75 / 0.79  0.91 / 0.91
           phantom only      0.42         0.65         0.68         0.51         0.26

(phantom only: the share at the metric's offset; the three heaviest homes hold 0.30, 0.14 and 0.52
of J_2's baseline under U, up to 0.69, 0.43 and 0.70 under B1.5.)

The resolution floor's hypothesis ("The metric's resolution floor": homes behind gaps of order h
carry the drift; BACKLOG 2's pending check): held, and sharpened. Under U the homes behind a
sub-cell gap or a phantom passage, a sixth of the population or less, hold 0.31 -- 0.63 of J_2's
baseline, and their share is what changes with the alignment (20543: 0.60 at the metric's offset,
0.33 at the shift); those behind a phantom passage alone, 2 -- 5% of the people, hold 17 -- 42% of
it. A phantom passage is the lattice's: a finer raster or a test of the open sub-samples touching
would close it, and B's enclosed test, made on the same cell graph, counts its homes reachable. The
variants do what they should at such a passage, make it conduct less, and so magnify its homes:
under them those homes hold most of J_2's baseline (94 -- 99.7% on 20543, 91 -- 95% for those behind
a phantom passage alone), and a block's level, P_{1/2}'s and J_2's most, turns on whether a clearing
happens to open their pocket, at the alignment of the day. Any conductance change needs that fixed
first: connectivity and B's enclosed test on the fine lattice (open sub-samples touching), or a
passable width below which a gap is closed, so that a home whose only way out is a passage nobody
walks is scored apart in B (i). On the corridor the variants' part of it is direct: at widths where
a shift adds or drops a column of sub-samples, P's max / min over 16 shifts is U 2.53, 1.81, 1.67 at
W 0.19, 0.30, 0.44 m, B_r 4.5 -- 4.9, 2.6 -- 2.7, 2.1 -- 2.2 and D_s 2.53, 2.82, 2.03 (back to 1.3
-- 1.4 by 1.06 m, U 1.29; summary_q2_subcell.csv): below the disk's size a gap conducts like
width^2, so a column gained or lost moves it by the square. Sightline's row threshold does worse: S
and S_t 1.79, 10.8 and 6.4 at 0.30, 0.44 and 0.69 m.

**Q4: what it changes.** On the 13 tuning blocks, the stored clearings (SIMP's default, the cheap
preset) and item 4's road prefixes at area D 0.05 (the six methods; a road that reaches D 0.05 only)
re-scored under every variant (summary_q4_*.csv). Under U the rows reproduce item 4's road prefixes
to 9e-8 (90 rows) and its B re-scoring of the stored clearings to 3.2e-9 (30; up to 5.4e-4 from
their stored preB Lens A, the shadowed homes' known shift). Kendall tau of the method order (B's
order) with U's, median [min] over the 13:

    lens     B0.5        B1          B1.5        D0.5        D1          S           S_t
    P        1.00 [0.86] 1.00 [0.81] 1.00 [0.79] 0.93 [0.71] 0.93 [0.64] 0.86 [0.57] 0.86 [0.36]
    P_{1/2}  0.93 [0.64] 0.93 [0.57] 0.90 [0.57] 0.90 [0.36] 0.86 [0.36] 0.86 [0.36] 0.71 [0.36]
    J_2      0.93 [0.79] 0.93 [0.64] 0.87 [0.64] 0.93 [0.43] 0.86 [0.43] 0.79 [0.43] 0.79 [0.43]

resistance_lp stays the best road on P on all 13 under U, B_r and D_s (11 under sightline), on
P_{1/2} on 10 -- 11. SIMP's default leads the best road by more under every variant (median, P: U
0.145, B_r 0.17 -- 0.19, D_s 0.21 -- 0.26, S 0.26, S_t 0.25; P_{1/2}: 0.24, 0.22 -- 0.31, 0.31 --
0.33, 0.34, 0.20), the best road keeping 0.48 -- 0.58 of SIMP's lens. Under the equity lenses the
roads fall while the clearings rise (median P_{1/2} of cycle_native 0.118 under U, 0.061 under B1.5,
0.032 under S_t; SIMP's default 0.52, 0.72, 0.93): Q3's pockets again, which the stored clearings
tend to open and a road's corridor mostly does not.

**Q4: SIMP under B_r.** On the sentinel, SIMP base (fw0.q3.i10.t0.001.e0.0001.k1s1e-06) with
objective P and free first (clear.free_first; none of the four has a home enclosed), area D 0.05,
under U and under all three B_r (runs w-simp1-*, one card each: a run cost less than choosing a
radius first), 1 -- 2 GPU-min a block, B_r 1.1 -- 1.6x U's time. Each clearing then scored under U
and the three B_r (width.py `cross`, runs w-cross-*; each reproduces its own lens to 1e-10):

    P lens       clearing:  U       B0.5    B1      B1.5      (rows: the conductance scored under)
    22422  U                0.5889  0.5880  0.5890  0.5873
           B1.5             0.8726  0.8723  0.8728  0.8723
    30848  U                0.5679  0.5679  0.5682  0.5672
           B1.5             0.6590  0.6606  0.6612  0.6606
    5810   U                0.2558  0.2500  0.2492  0.2486
           B1.5             0.2873  0.2836  0.2837  0.2841
    9712   U                0.1378  0.1375  0.1368  0.1373
           B1.5             0.1591  0.1589  0.1585  0.1593

SIMP under B_r finds nothing better on B_r's own lens than SIMP under U does: its clearing minus
U's, both under the same B_r, is -0.0047 -- +0.0020 over the 12 (block, radius) pairs, and on 5810
U's clearing is ahead under every radius (by 0.003 -- 0.005). The four clearings of a block lie
within 0.008 of each other under any conductance (P_{1/2} and J_2 alike), while the conductance
moves the level by up to 0.28 (22422's P lens 0.59 under U, 0.87 under B1.5). The clearings differ
more than they score: Jaccard with U's 0.84 -- 0.95 (B0.5), 0.78 -- 0.88 (B1), 0.75 -- 0.86 (B1.5),
and B_r clears more buildings for the same D (5810: 346, 350, 353 against 302; 22422: 72, 74, 76
against 70).

**What looks promising, and what does not (the owner decides).**
- B_r is the justified uniform that keeps the machinery: P convex (the guards), a gradient for SIMP,
  the angle error about U's (1.72 -- 1.85 at 1 m against 1.66), U's alignment on lanes from 0.75 m,
  a width^2 law below 2 r and a margin at each wall of 0.16, 0.30, 0.42 m on the lattice (B1.5's
  near pedestrian engineering's 0.5 m), U's method order on P (tau 1.00) and nearly on the equity
  lenses (0.87 -- 0.93). What it does to the flow is modest: walkers keep a little more to the lanes
  U's flow already uses (the lift +0.005 -- +0.05) and no new lane forms. SIMP under it clears 5 --
  25% other buildings (smaller ones; on 5810 deeper ones, farther from the street) for no gain even
  on its own lens: on these four blocks an optimizer run under U serves it. B1.5 is the most
  promising of the three, as the strongest; a radius of 2 -- 3 m is untested and is what would
  decide whether B_r can reach D1's strength convexly.
- D_s, the owner's literal form, is the strongest uniform (D1's lift +0.03 -- +0.12) but its angle
  error is worse (2.39 at 1 m), it has no gradient (SIMP would need a fractional distance, above)
  and it reorders the methods more. D0.5 is B1.5 at 0 deg with a worse angle error; D1 has no B_r as
  strong, so neither is dominated until a larger B_r is scored beside D1.
- S_t over S, for the corridor pictures (the turning-boost call): it mends sightline's 1.5 -- 2 m
  lanes (1.74, 1.26 against 2.40, 1.67), its maps lose the streaks (5810's zoom) and it keeps the
  clearest lanes of every variant (the lift +0.04 -- +0.16 on U); its 1 m lanes (5.09) and its
  alignment at sub-metre widths (4.85 at 0.75 m, 10.8 at 0.44 m) stay, it has no gradient and it
  reorders the methods most (P_{1/2}'s tau 0.71).
- Not as they stand, any of them: every variant hands the equity lenses, P_{1/2} first, to the homes
  in pockets behind phantom passages and sub-cell cracks (Q3), and makes a block's level several
  times as sensitive to the lattice as under U (on 38366 P_{1/2} 5 -- 8x, P 7 -- 17x). That is an
  effective width doing its job on a passage nobody walks, with the placement still counting the
  pocket reachable. The pockets want fixing first whatever the conductance (fine-lattice
  connectivity for the operator and B's enclosed test, or a passable width), and under U they
  already hold 17 -- 42% of J_2's baseline on the three blocks while holding 2 -- 5% of the people.

What is not tested: B_r beyond r 1.5 m; the greedy under B_r or D_s (Clearing.tension refuses a
factor that looks past its cell); the pockets' fix itself.

Compute: GPU 3.4 GPU-h of scoring (144 block-variant tasks, 22422 under S the longest at 6.4 min),
0.38 for SIMP and 0.23 for the cross-scoring, with the first SIMP run's 2 min on four cards before
it was resubmitted one card a variant: about 4.5 GPU-h. The CPU nodes' first scoring runs
(w-o0-orion, w-o0-quadro2, 67 tasks at once) were cancelled after about 50 min (~55 core-h, no row
kept): a field took 575 -- 1,094 s there (31 -- 36 memory-bound AMG solves a node) against ~50 s
here, so 1558 would have needed ~13 h a variant. Here: the guards, the corridors (about 2,500
solves, seconds in all), the smoke tests, the maps and the fine bottlenecks, about 6 CPU-h.

The owner's call (2026-10-13): uniform, for the metric and for the corridor pictures alike ("I
don't actually mind if there are no new corridors. That may be to be expected"; "LGTM, incl using
uniform for the corridor pictures"); the variants stay selectable. Next, before any re-scoring, the
pockets' fix as one rule rather than a mask on today's: a step conducts by the share of its width
open all the way from cell centre to cell centre, read on the sub-samples (U discretized
faithfully, the face apertures of a cut-cell scheme where today's rule uses the cells' volume
fractions), so a phantom passage conducts nothing and B's enclosed test and Placement, reading the
same steps, see it; a passable width stays a separate choice, after the fix if real cracks still
matter (BACKLOG 4, decision (3)).

### The grid fix: steps that conduct by their open width (owner 2026-10-13: "LGTM")

BACKLOG 4, decision (3)'s "Next", and item 2 (why a block's level drifts with the lattice). Today a
step conducts by the smallest open fraction of the cells it touches (lifted.SmallestCell): how much
of each cell is open, not whether the open parts meet, so two 4-adjacent cells whose open
sub-samples do not touch are joined (a phantom passage), and a home whose only way out is one is
scored reachable at a huge escape time ("Uniform conductance with an effective width", Q3). The fix
the owner approved: a step conducts by the share of its width open all the way from one cell's
centre to the next, read on the sub-samples (4 x 4 a cell). aperture.py and aperture_check.py
(9f131a0f -- c81cd667); aperture_rows/ (the rows ignored, the summaries committed). h 0.5, ell 3 m,
K 8, area population, B's placement; every lens for P, P_{1/2} and J_2.

The rule. lifted.StepWidth, a Strategy where a step's weight is built (Params.steps, required and
keyword-only, named in the run's conductance as `<along>[+<cells>][+<steps>]`: uni+apwfs):
SmallestCell (today's, unnamed) and Aperture. A step along v_k conducts m_k / |v_k|^2 times the
mean over its tracks, each track by the smallest per-sub-sample factor on it
(CellConductance.sub_factor: U's openness; B_r's min(openness, B) and D_s's ramped openness port as
they are, D_s still without a gradient; width.BlurCentre raises). Under fractional clearing a
sub-sample's openness is affine in the clearing (a cell's cleared share spread evenly over its
footprint sub-samples, exact for a cell under one building, as B_r has it), so a track, a minimum
of affine functions, is concave, the step too, and P stays convex and monotone; the gradient shares
a track's coefficient among the sub-samples tied at its minimum. Turning edges keep the cell's open
fraction. The rule is the operator's cell graph, so B's reach (Placement's allowed cells,
lifted.placeable), the enclosed test, free first and the scorer all read it. A field costs about
1.2x today's to set up and solve (median, small blocks 6.5 s against 5.0, large 207 s against 174,
on the CPU).

The choices (lifted.Band, lifted.PartRule), each variant selectable and named:
- The band of an oblique step: the edge's own share of the cross-section, h / |v_k| (the bands of
  parallel edges tile the plane: A, `ap`), or the cell's width h (six tracks on a diagonal step,
  eight on a knight step, overlapping their neighbours': Aw, `apw`). A is the best on an open
  oblique channel (0.71 -- 0.98 of m_k W against the wide band's 0.65 -- 0.96), but a wall sliver
  one sub-sample wide falls between its bands and is cut off: on the corridor below, P at 0 deg
  moves 1.78 -- 1.91x over eight shifts (today's 1.05 -- 1.12) and a third of the demand cells are
  not reached.
- The tracing: each track the exact supercover of its segment (the sub-samples whose interior it
  crosses, in rational arithmetic: axis_tracks), at half-integer offsets across the band, so no
  track passes a lattice corner and each is a 4-connected chain (traced through sub-sample centres,
  a diagonal track would cross two sub-samples touching at a corner: the connect guard's fault). An
  axis-aligned step's tracks ran from one centre to the next (A, Aw, Aws), as the brief has it;
  that cuts a one-sub-sample sliver along a wall off its neighbours (in a cell whose centre is
  footprint no axis-aligned track can cross the half cell on its side), which on the corridor's
  walls leaves up to 22% of the demand cells unreached and moves P at 0 deg 1.09 -- 1.22x over
  eight shifts. The face tracing (Band's `face`: Fws, `apwfs`) takes an axis-aligned step's tracks
  across the face only, the four pairs of sub-samples facing each other (the cut-cell face aperture
  itself; within a cell its node joins its open part, which AnchorPart keeps one part), oblique
  steps keeping the wide band's centre-to-centre tracks. Its reach is then exactly the 4-connected
  flood fill of the sub-samples it reads (the connect guard).
- Split cells, whose open sub-samples form parts that do not touch (two buildings meeting at a
  corner inside the cell): the cell's one node joins them whatever the tracks say (JoinParts: A,
  Aw), or only the part holding its anchor conducts, a sub-sample of its largest part at baseline,
  and the cell takes no demand nor frees a home (AnchorPart: Aws, Fws, the `s`). The parts are the
  field's own at every 0/1 state (a lookup over the 65,536 patterns of a 4 x 4 cell); opening space
  only grows the anchor's part, so the operator stays monotone, and in the eps world every cell is
  one part, so the relaxation is untouched. Few people's demand sits in them (0.38% of the demand
  on the small blocks under today's placement, 0.30% on the large), but their joins matter: a
  corner contact joined a pocket to the street (19537's J_2 baseline 5.8x today's under Aw; 3776's
  home 92 6.3x its escape time), so the joins are cut rather than the cells split into a node a
  part, which would make the mesh's cells parts that merge as clearings open footprints. A mask
  fixed at baseline was built first and dropped (the open guard caught a cleared wall's minor part
  staying closed).
- The composite mesh: direct. A step with a coarse end conducts by 1 (every cell within a step of a
  coarse cell is wholly open: AdaptiveMesh keeps h within d0 > the ground band of every footprint);
  a step between fine cells reads its tracks. On 6310, a5x8 against uniform h 0.5: P0 within 0.13%
  under both rules (today's 36,661 / 36,613; Aw 36,253 / 36,204), the same home enclosed and the
  same crossing freeing it.
- Free first's region graph under the aperture (lifted._track_regions; today's,
  Grid.regions_smallest, raises on the composite): the components by the rule's own steps; a
  building's pieces, its cells joined by steps some track of which only it blocks; adjacency only
  where a clearing really joins two regions, read off each track's blocking buildings at baseline
  (one building: its piece in an end cell, or a gadget owned by it for a crossing through cells
  neither end holds; two, each with a piece in an end cell: those pieces). In a split cell
  (PartRule.through) a piece joins the cell's component only if its building, cleared alone, joins
  the anchor's part, and a home is freed through the cell only by a piece whose clearing makes it
  one part: its node reaching the street is no way out while it takes no demand. That last was
  found when free first raised on 24240, 45267 and 63612 (a home whose only near cells reaching the
  street were split, its crossing clearing nothing); the `free` guard below.

The corridor (width.walled_channel: W x 40 m, walls footprints 3 m thick, a dead end behind the
demand), demand on the cells each rule reaches, as Placement allows; P's max / min:

    angle, 13 angles 0 -- 45 deg     T      A      Aw     Aws    Fws
    W 1 m                            1.66   1.54   1.22   1.22   1.21
    1.5 m                            1.33   1.82   1.12   1.12   1.10
    2 m                              1.15   1.83   1.11   1.11   1.09
    4 m                              1.06   1.77   1.07   1.07   1.03
    alignment, 0 deg, 8 shifts
    0.75 m                           1.06   1.91   1.22   1.22   1.06
    1 m                              1.12   1.78   1.19   1.19   1.12
    1.5 m                            1.07   1.85   1.12   1.12   1.07
    2 m                              1.05   1.86   1.09   1.09   1.05
    sub-cell, 0 deg, 16 shifts
    0.19 m                           2.53   2.05   2.05   2.05   2.05
    0.30 m                           1.81   1.54   1.54   1.54   1.54
    0.44 m                           1.67   1.51   1.65   1.65   1.49
    0.69 m                           1.34   2.38   1.42   1.42   1.35
    1.06 m                           1.29   2.09   1.39   1.39   1.30

Fws reaches every demand cell at every angle and shift, as today's rule does; A, Aw and Aws leave
7 -- 33% of them unreached (wall slivers). Fws is the best of the apertures everywhere and against
today's rule better on the angle check (1.21 against 1.66 at 1 m: today's oblique steps take the
smallest open fraction of four cells, conducting 0.27 -- 0.89 of m_k W on an open channel, the
aperture 0.65 -- 0.96) and below half a metre (2.05 against 2.53 at 0.19 m, 1.49 against 1.67 at
0.44 m), and within 0.01 of it on the alignment checks from 0.69 m. What is left at 0.19 -- 0.30 m
is the sampling itself: a channel one or two sub-samples wide reads as one or as two (half or all
of its conductance) as the lattice shifts. Relative to today's, the aperture conducts a 30 deg
channel 2.13x as well at 0.5 m, 1.46x at 1 m, 1.13x at 2 m, 1.02x at 6 m (at 0 deg alike, 0.95 at
0.5 m).

The guards (aperture_check.py), each failing under its fault (a patch in one process):
- `identity`: with today's rule selected, width_check's 57 numbers (computed at 3f74f5fa, before
  the cells strategy and the step rule) reproduce bit for bit on 38136, 19263 and 19570, the
  relaxation's value and gradient among them (relax's along term now goes through SmallestCell's
  accumulator). Fault `ulp` (SmallestCell's widths one ulp up): 15 of 19 differ on 38136.
- `open`: an open field gives today's operator: on three synthetic rectangles every sub-sample
  open, lifted.edges under each aperture equals today's family for family, bit for bit; on 38136,
  19263 and 19570 with every footprint open, every edge whose cells are wholly open (216,219,
  316,323 and 522,582) has today's weight bit for bit (the 3,774 -- 5,089 under Fws at the block's
  ragged edge differ, as the rule intends). Fault `sum` (a step by the sum of its tracks): every
  rule fails.
- `channel`: a straight open channel aligned with an axis conducts m_k W per metre in layer k
  exactly at the axis-aligned lattice angles under every rule, and at the oblique ones within 0.2 m
  a wall (W_eff in [W - 0.4, W]: 0.65 m at 1 m and 45 deg, the worst). Fault `max-track` (a step by
  its best track): W_eff up to 1.118 m on a 1 m channel.
- `cracks`: on pairs of cells, open sub-samples that do not touch conduct nothing (rows and columns
  that do not meet across a face, an open half far from it, corners away from each other on a
  diagonal step: today's rule 0.06 -- 0.50); a crack one sub-sample wide from centre to centre
  conducts 1/4. Fault `smallest` (today's widths): four of the seven cases fail.
- `connect`: on the sentinel (22422, 30848, 5810, 9712) and the drift blocks (20543, 38366, 17608),
  against the 4-connected flood fill of open sub-samples from the ground cells. Today's rule
  reaches 87 -- 2,474 cells the flood fill does not (its phantom passages; 17 -- 662 of them
  through a split cell's node). Under Fws the reach is exactly the flood fill of the sub-samples it
  reads (split cells' minor parts closed: 0 cells differ on all seven), so every disagreement with
  the plain flood fill is a cell it reaches only through a split cell's minor part (9 -- 119 a
  block). A, Aw and Aws reach no cell the flood fill does not except through a split cell's node (A
  and Aw: up to 414; Aws none) and miss 2,950 -- 17,586, the wall slivers (93 -- 99%: a cell whose
  2 x 2 centre is footprint) and cells behind a bent passage. Fault `diagonal-leak` (tracks through
  sub-sample centres, so two sub-samples touching at a corner pass): every rule fails on 17608 and
  38366 (Fws's reach 915 and 194 cells off the flood fill's).
- `convex`: along four random lines through the clearing box, every step width concave (second
  differences at most 2.6e-15) and P convex (at least 4.6e-3 of P) on 38136, 19263 and 19570, every
  aperture. Fault `square` (a track by the square of its smallest factor, equal at every 0/1
  state): P's second difference down to -5.2e-3 (the widths' up to 5.1e-2) of P. (width_check's
  `product` form left the eps world's system too ill-conditioned to solve.)
- `gradient`: the relaxation's gradient (SIMP q 1 and 3, P and J_2, eps 1e-4) against central
  differences on six buildings of each of the three blocks: within 1.8e-6 of the largest over the
  48 checks (Fws 2.1e-7). Fault `no-spread` (the openness's vjp without its spread over a cell's
  footprint sub-samples): 7.1e-2 -- 8.7e-2 of the largest off, all 16 checks on 38136.
- `free`: free first (budget 1, nothing held) under Aws and Fws on 24240 frees every home enclosed
  at baseline it takes a crossing for, by the metric's own test (home 1639, enclosed under both,
  freed by one building for D 0.0001). Fault `split-source` (every open cell taken to take demand
  in the region graph): its crossing clears nothing, and free first raises under both.

**Q1: how common the pockets are today.** Under today's rule at baseline (width.fine_bottleneck on
the uniform grid; on the 22 oversized blocks the composite mesh, phantom passages only:
aperture.composite_flood), a home is behind a pocket when over 5% of its injection sits in cells
behind a sub-cell gap (a bottleneck to the street of at most 0.5 m, narrower than the cell's own
clear width) or a phantom passage:

    today's rule, baseline     behind a pocket                 behind a phantom passage only
                               people demand P     P_1/2 J_2   people demand P     P_1/2 J_2
    220 small    median        0.22   0.044  0.31  0.35  0.37  0.030  0.006  0.036 0.040 0.040
                 90th pct      0.40   0.10   0.62  0.74  0.77  0.083  0.022  0.16  0.26  0.28
                 pooled        0.25   0.060                    0.041  0.011
    60 large     median        0.13   0.026  0.19  0.24  0.26  0.020  0.005  0.033 0.034 0.035
                 90th pct      0.22   0.056  0.42  0.63  0.65  0.038  0.009  0.084 0.22  0.22
                 pooled        0.13   0.031                    0.022  0.006
    22 oversized median                                        0.007  0.002  0.008 0.009 0.009
                 pooled                                        0.010  0.002

(people: the homes over the threshold, as a share of the block's people; demand: the share of the
block's demand that sits behind; P, P_1/2, J_2: those homes' share of each baseline; pooled: over
every block's people.) A quarter of the small blocks' people and an eighth of the large blocks'
have over 5% of their demand behind a sub-cell gap or a phantom passage, and they hold about a
third of P and of the equity lenses' baselines at the median block, three quarters at the 90th
percentile: a pocket home's escape time is long, and P_{1/2} and J_2 weight it. Behind phantom
passages alone, the defect the fix removes, are 4.1% of the small blocks' people (2.2% of the
large, 1.0% of the oversized), holding a median 3.6 -- 4.0% of the lenses and up to 0.89 of P and
0.998 of J_2 (small blocks; large 0.26 and 0.64).

**Q2: what the fix does at baseline** (Fws against today's on every block; A, Aw and Aws in
summary_q2_pooled.csv).

- Enclosed: Fws encloses 112 homes on the small blocks against today's 63 (49 new, on 19 blocks,
  0.09% of their people), 62 against 24 on the large (38 on 17 blocks, 0.02%) and 11 against 8 on
  the oversized (3 on 45267 and 63612); every home today's encloses stays enclosed. All but three
  new ones are enclosed by the geometry, every open sub-sample near them joined to the street only
  through phantom passages; the three (one small block, two large) by the split-cell rule, their
  only way out a corner contact. Free first frees every one within D 0.05: on 35 small blocks for
  64 buildings, a median D 0.0031 a block, at most 0.0164 (today's rule: 24 blocks, 38 buildings,
  0.0044, 0.0156); on 23 large blocks for 49 (0.00045, at most 0.0082; today's 12 and 20, 0.0008);
  on 45267 and 63612 for 3 and 8, D 0.0004 and 0.0024 (today's region graph raises on the
  composite).
- Demand moved (the share of the demand Fws places on other cells than today's, pooled over the
  people): 1.33% on the small blocks, 0.78% on the large, 0.38% on the oversized (median per block
  0.9, 0.7 and 0.3%; Aws 5.3% and 4.5%). Of it, on the uniform blocks, 1.02% and 0.53% comes off
  cells behind phantom passages, to the home's cells with a way out (all of Q1's phantom demand),
  and 0.31% and 0.25% off split cells, which take none (Aws also moves 4.3% and 4.0% off the wall
  slivers it cuts off).
- Split cells: 10,982 on the small blocks, 26,718 on the large, 12,973 on the oversized; today's
  placement puts 0.38%, 0.30% and 0.18% of the demand in them.
- The levels (Fws over today's, median, 10th -- 90th percentile and extremes): small blocks P0
  0.988 (0.91 -- 1.12; 0.17 -- 1.84), P_{1/2} 1.000 (0.88 -- 1.27; 0.03 -- 8.98), J_2 0.985
  (0.78 -- 1.39; 0.01 -- 11.8); large P0 0.994 (0.96 -- 1.05), P_{1/2} 0.994 (0.91 -- 1.26;
  0.49 -- 2.13), J_2 0.988 (0.89 -- 1.27; 0.44 -- 2.16); oversized 0.995, 0.998 and 0.993 (J_2
  0.67 -- 17.1). A level moves past 1.5x either way on 43 of the 302 blocks.
- What moves them (summary_q2_levels, q2_level_tops; aperture_rows/behind keeps each home's
  shares). Where a level falls past 1/1.5 (J_2 on 20 blocks), on 16 the homes behind phantom
  passages and those Fws encloses carry 69 -- 120% of the fall: the defect's own homes, scored at
  their real distance or apart (43304's J_2 baseline 0.01x, 17608's 0.44x); on the other four
  (17027, 17033, 23597, 43817) homes partly in cracks carry it, where I read the narrow lanes'
  oblique steps, which today's rule conducts at a quarter to a half of their width [inferred].
  Where a level rises past 1.5x (J_2 on 20 uniform blocks, and on 28051 and 63612 of the
  oversized), it is real cracks: on 13 of the 20 the three homes that rise most carry 64 -- 102% of
  the rise, their escape times 2 -- 100x today's (44274: 55, 164, 283 to 2,949, 2,592, 2,270;
  45876: 12 to 1,211), with 21 -- 100% (median 50%) of their injection in cells whose way out is a
  crack of at most 0.25 m, one or two sub-samples; on the other seven the rise is spread over homes
  with 10 -- 25% of it there. Today's rule hid these behind its own defect (a crack cell's steps
  reach through the walls beside it at the cell's open fraction); Fws conducts a 12.5 cm crack as
  12.5 cm, which under U is a long, slow exit.

**Q3: drift and angle.** SIMP's default at the metric's offset and at the block's shift
(width.ALIGN, the shift that moved each most), the lens's change:

    lens change      P                        P_{1/2}                  J_2
                     T       Aws     Fws      T       Aws     Fws      T       Aws     Fws
    20543 (o2)       -0.0455 +0.0010 +0.0008  -0.2339 +0.0024 +0.0018  -0.1468 +0.0012 +0.0007
    38366 (o2)       -0.0088 -0.0025 -0.0021  -0.0848 -0.0021 -0.0055  -0.0928 -0.0020 -0.0045
    17608 (o3)       -0.0469 +0.0037 +0.0094  -0.0601 +0.0144 +0.0215  -0.0621 +0.0240 +0.0266

The aperture takes the drift away: Fws's lens moves by at most 0.009 (P), 0.022 (P_{1/2}) and 0.027
(J_2) between the alignments, today's by up to 0.047, 0.234 and 0.147 (Aw on 17608: +0.11, +0.07,
+0.12, its corner contacts). Today's was carried by the homes behind phantom passages: on 20543
they held 0.41 and 0.42 of P_{1/2}'s and J_2's baselines at the metric's offset and 0.04 at the
shift; on 38366 (per home, summary_q3_homes and q3_top) 70 of them, 3.9% of the people, carry 69
and 77% of the absolute change of the homes' terms, home 1497 alone 17 -- 31%. Under Fws the
largest single home carries 4 -- 10% on 38366 and the phantom group 8 -- 17%. What is left is
spread over the block: the candidate, homes whose share of injection behind a crack of at most 0.25
m moves with the alignment by over 5 points (a crack's sampled width snapping; 8 -- 14% of the
people on the three blocks), carries 8 -- 21% of the absolute change, about its share of the
people, and the rest 65 -- 83%. So the drift that remains is the lattice's ordinary resolution
(BACKLOG 1: alignment moves a large block's level by up to 0.16 under today's rule; here 0.027 at
most), not snapping cracks, though the corridor shows the snapping where a crack is all there is (a
0.19 m channel's P moves 2.05x over 16 shifts). What the cracks carry is level: on 17608 under Fws
one home with most of its demand in a crack holds 0.52 of P_{1/2}'s and 0.56 of J_2's baseline
(0.18 and 0.19 under today's rule), and three homes 0.58 -- 0.65.

**Q4: what it changes.** On the 13 tuning blocks, SIMP's default and the cheap preset
(clearings.Store.cleared; with free first redone under the rule where a clearing leaves a home it
encloses: 1558's default 2 buildings, D +0.0005; 23597's two 1, +0.0001; 30848's two 1, +0.0002)
and item 4's road prefixes at D 0.05, scored under the aperture against today's rows (width.py's U,
the same code path; my rerun of today's matches them within 3e-7 on the three drift blocks, both
alignments), in B's order (enclosed population first, then the lens):

Kendall tau against today's order: median 1.00 on every lens (Aws 1.00, 0.94, 1.00), below 1 on
five blocks: 1558 (0.67 on P, 0.56 on P_{1/2}, 0.61 on J_2; under Aws 1.00, 0.94, 1.00), 23597 and
30848 (0.72 -- 0.83, as under Aws), 38616 (0.72 -- 0.89) and 9717 (0.94 on P_{1/2}). SIMP's default
leads on 12, 10 and 11 of the 13 (P, P_{1/2}, J_2; today 12, 11, 11), the cheap preset on the rest;
resistance_lp is the best road on 11, 9 and 11 (today 12, 10, 11); SIMP's default leads the best
road by a median 0.132, 0.131 and 0.165 (today 0.145, 0.238, 0.195), at least 0.056. 1558's reorder
is the cracks again: three homes with 70 -- 87% of their injection behind 12.5 -- 25 cm cracks rise
from escape times 82 -- 109 to 647 -- 864 and carry 92% of P_{1/2}'s rise to 2.13x, so whatever
opens them scores: clearance and clearance_looped go from 0.07 -- 0.10 to 0.62 -- 0.63 on P_{1/2}
and from 0.06 -- 0.08 to 0.42 -- 0.44 on J_2, past resistance_lp (0.16 to 0.07, 0.13 to 0.06), and
the cheap preset passes SIMP's default (0.81 against 0.76 on P_{1/2}). On 38616 every road but
euclidean_grid rises the same way on P_{1/2} (0.18 -- 0.27 to 0.55 -- 0.59).

**SIMP under the aperture** (the gradient's check, on 19537 with its 16 homes Fws encloses, 4
today; SIMP base fw0.q3.i10.t0.001.e0.0001.k1s1e-06, free first, area D 0.05, CPU): on P its
clearing scores 0.6336 (free first 6 buildings, D 0.0150), today's stored SIMP clearing under Fws
0.6093 leaving 0.16% of the people enclosed, 0.6107 with them freed first (D 0.0527); on J_2 0.5866
against 0.5700. Jaccard with today's clearing 0.73 (P) and 0.54 (J_2); 330 and 557 s.

**What is left, and the recommendation (the owner decides).**
- Adopt Fws (uni+apwfs) and delete SmallestCell. Fws removes the phantom passages exactly (its
  reach is the flood fill of the sub-samples it reads), takes the levels' lattice drift from up to
  0.23 to at most 0.027, cuts the angle error at 1 m from 1.66 to 1.21 and ties today's alignment
  on walled lanes, keeps the method order (tau median 1.00 on every lens) and SIMP's lead, and
  keeps P convex with a working gradient, at about 1.2x the time. Today's rule is ahead only on
  that time, and every stored score was computed under it: they move (Q2), and re-scoring belongs
  to the consolidated pass.
- A, Aw and Aws: Aw is dominated by Aws (alike on the corridor, its corner contacts on the blocks:
  17608's drift 0.11, 19537's J_2 5.8x); Aws by Fws everywhere but 17608's P drift (0.0037 against
  0.0094), which it buys by cutting the wall slivers off (5.3% of the demand moved against 1.3%,
  7 -- 33% of the corridor's demand cells unreached); A is the best on an open oblique channel
  (W_eff 0.71 -- 0.98 of W against 0.65 -- 0.96) and the worst with walls (1.78 -- 1.91x at 0 deg).
  Recommendation: keep Fws only; A's open-channel edge and Aws's sliver cut go to the dead-end
  record.
- The passable width still looks needed, and more than before. Fws conducts a real crack at its
  sampled width, and a home with most of its demand in a 12.5 -- 25 cm crack gets an escape time up
  to 54x today's, which the equity lenses then follow (44274's J_2 baseline 11.8x, 28051's 17x; on
  1558 the method order). Today's rule hid this behind its defect, Aws by accident. A crack nobody
  walks through is enclosure, not a slow exit: closing gaps under about 0.5 m encloses these homes
  (free first then frees them, first in B's order) or moves their demand to their open side, and
  touches Q1's pockets (a quarter of the small blocks' people have over 5% of their demand behind
  such gaps). It is a modelling choice and a convexity question: a hard opening of the open set
  moves with every clearing (scoring only, as D_s), a B_r-like soft one would keep P convex.
- Not done: SIMP re-optimized under the aperture past the one-block check (the consolidated pass);
  the composite's free first under today's rule (still raises); free first's region graph reads a
  split cell's minor part as never open (conservative: a clearing that would join it is not seen
  through it).

Compute: GPU about 4.8 GPU-h on the cluster, at most 4 cards (the baselines of the 23 large blocks
too big to run here and the 22 oversized, T with A, Aw and Aws or with Fws: ap-base-*; the drift
and tuning blocks' fields under T, Aw, Aws and Fws: ap-score*), nothing on the CPU nodes. Here,
single-threaded, at most 8 processes: the small blocks' and 37 large blocks' baselines (10.8 CPU-h,
plus about 2 lost when a pool read aperture_check as edited under it), the moves and pockets (2.7),
the corridor (900 solves, under a minute), the guards (about 5), SIMP on 19537 (0.25) and the
report: about 20 CPU-h.

**Adopted (owner 2026-10-13: "We can adopt Fws and delete the losing methods").** Fws is the
operator's one step rule, folded into it with no Strategy left (03f7eb49, dba7e320 -- afa00a02):
lifted.step_widths (a step by the mean over its tracks of the smallest per-sub-sample factor on
each), step_gradient, reach, placeable, components, demand_cells and regions (free first's graph),
axis_tracks across the face for an axis-aligned step and centre to centre across the cell's width
for an oblique one, and the anchor-part rule, as plain functions. Deleted: SmallestCell with its
gradient accumulator and the 4-connected reach and region graph (Grid.reach_smallest and
regions_smallest, the composite's raising stub), A, Aw and Aws (the edge-share band, the
centre-to-centre tracing of axis-aligned steps, JoinParts), Params.steps and the StepWidth
Strategy, width.BlurCentre (sampled at a cell's centre, it has no per-sub-sample factor, so no step
width: it cannot run) and aperture.py (the rules' comparison: its summaries stay in aperture_rows/,
its code at bd1704dc). TurnBoosted (S_t) runs as it was: it boosts the turning edges, which the
step rule does not touch (on 38136 P0 10.14 against S's 15.85; its vjp still raises). GpuAMG's
master operator is now the aperture's on the open field, so a GPU solve can move at its tolerance
from the study's (a step whose every track leaves the block is no longer in the aggregation's
pattern) [inferred: not run on a card]. free_roads' targets take the pockets of the operator's
components, through the near cells that can take demand, where they took 4-connected open cells.
The identity: every conductance name ends in `+apwfs` (lifted.APERTURE: uni+apwfs, uni+b1+apwfs,
ss100k2n2r30t+apwfs) and conductance_of raises on any other, so nothing written under the old rule
(uni, uni+b1, ss...) or the study's A, Aw and Aws (ap, apw, apws) is read as the new rule's:
relax's, search's, polish's, road_rescore's and transparency's rows take the new name in their
paths; clear.py's greedy names its rows by the conductance; pweight's and width.py's rows, keyed by
objective and variant, move under pweight_rows/uni+apwfs/ and width_rows/apwfs/; free_roads.CODE
hashes lifted.py, so FreeEnclosed's identity and tag moved. The stored clearings (the decisions)
stand; their `perm` is the old rule's (clearings_check's rescore now scores them under the aperture
and says so), and the export's along pattern matches no `+apwfs` name, so a run under the aperture
raises there rather than export as one of the old rule's (teaching it the name belongs to the
consolidated pass). The guards, re-run after the deletion, each failing under its fault: identity
(width_check's 57 numbers under uni+apwfs, recorded at bd1704dc before any edit,
width_rows/identity_reference_fws.csv: bit for bit on 38136, 19263 and 19570; fault `ulp`, every
width above 0 one ulp up: 16 of 19 differ on each; the old rule's reference retired); open (now
against the uniform operator assembled in the guard: the synthetic rectangles family for family,
and with every footprint open the steps whose tracks' cells are wholly open, 216,025 -- 522,400 a
block, of width exactly 1; fault `sum`); channel (W_eff 0.65 -- 1 of W at the oblique angles, exact
at the axis-aligned; fault `max-track`: 19 of 24 fail); cracks (fault `open-fraction`, each track
by the smallest open fraction of the cells it crosses, the phantom passage: 4 of 7 fail); connect
(the reach equals the flood fill of the sub-samples it reads on all seven blocks, the plain flood
fill reaching 9 -- 119 cells more through split cells' minor parts; fault `diagonal-leak`: 5 -- 915
cells off); convex (the widths' second differences at most 2.2e-15, P's at least 4.6e-3 of P; fault
`square`: P's down to -8.3e-3); gradient (within 3.3e-7 of the largest; fault `no-spread`: 6.6e-2
-- 1.2e-1); free (24240's home 1639 freed for D 0.0001; fault `split-source` raises). width_check,
under the aperture now: `operator` checks U's turning edges (bit for bit) and per-sub-sample
openness (within 5.6e-17) against the open fractions' definition (fault `b1`); `convex` (B_r's cell
factors' second differences at most 1.1e-13; fault `product`); `gradient` (B_r) within 4.4e-6 of
the largest over its 36 checks, against the closer of the central differences at steps 1e-5 and
1e-6 (fault `no-blur-vjp`: 3.7e-3 -- 3.8e-2): at 1e-5 alone one check straddled a kink of a track's
minimum (38136, B_1.5, J_2, q 1, building 5: 1.1e-5, 9.4e-6, 4.4e-6, 1.0e-7 off at steps 1e-5,
3e-6, 1e-6, 3e-7), at 3e-7 alone two lost digits to cancellation (up to 5.9e-5).
transparency_check's free-roads guard on its ten blocks: 41 homes enclosed at baseline under the
aperture (21 under the old rule; 41673's 5 became 18), every one freed, by 32 roads, each freeing
its home by the scorer's test (D area at most 0.019, on 41673). tests/test_clearings.py: 34 pass.
Compute here, single-threaded, at most 8 processes: about 7.5 CPU-h, nothing on the cluster. The
passable width stays open (the owner is weighing it): closing gaps under a width needs only the
sub-samples it folds into the footprints beside them (Grid.bsub and label_sub, all the operator
reads of the footprints), not the operator.

### The gap complex: passability, enclosure and flow without the raster (owner 2026-10-13: "sweep it, as a persistence filtration")

BACKLOG 4, decision (3)'s "Next", the passable width. The owner's proposal: rather than pick a
width, sweep it as a persistence filtration of the gaps between buildings, and if the structure
holds up use it as the runtime representation (footprints read once, to measure the gaps, and never
again). The brief (2026-10-13) and its addendum (the turning cost may go; three forms of the
width): stage 1 the complex, its filtration and free first, gated on agreeing with the raster under
Fws as w -> 0; stage 2 the uniform law on the complex, the forms, and the heading lift's weight.
Code (all new, research/roadless): gap_complex.py (the complex, the filtration, free first),
gap_study.py (agreement, apertures, the sweep), gap_check.py (stage 1's guards), gap_flow.py (the
mesh, the two-point scheme, the demand, the width forms), gap_clear.py (fractional clearing),
gap_compare.py (the complex against the raster's isotropic model and the lifted rows; lanes,
interiors, the region), gap_simp.py (SIMP on the complex), gap_flow_check.py (stage 2's guards),
gap_report.py (gap_rows/summary_*.csv, committed; the rows are not). No new dependency: shapely
2.1.2's constrained Delaunay, scipy's SuperLU, pyamg (already in the research group) for the
raster's isotropic solves.

**The complex** (gap_complex.build). The open ground beyond the street band less the footprints,
every outline densified to s and triangulated by GEOS's constrained Delaunay, then finished by
Lawson's flips (GEOS stops after a fixed number of scans: 74 non-Delaunay passages left on 19263),
flipping by the circumcentres' order along the passage's normal, (cot a + cot b) / 2 >= -1e-9 (an
angle-sum tolerance let a sliver's centre run far back). Every vertex lies on an outline, so a
passage two triangles share spans open ground from one outline to another and its length is the
gap's width there, its aperture; a phantom passage cannot occur. The geometry forced some
housekeeping: local coordinates (segmentize's vertices sat up to 9e-6 m off their outline in UTM),
vertices snapped to 1e-9 m and merged within 1e-7 m (GEOS repeats a vertex where outlines meet;
grown outlines cross a hair apart), the overlay's dust under 1e-6 m^2 dropped (slivers where GEOS's
triangles fell inside footprints), and three fallbacks for GEOS's ear clipping. The street: the
raster's ground is every cell centre within 1 m of the block's streets (lifted.Grid.of), so the
complex's open ground stops at that line, a passage with an end on it has aperture inf (beyond it
is street) and a triangle with an edge on it is an exit. Sampling or a constrained triangulation:
both, the constrained triangulation of outlines densified to s 0.125 m. On 20 blocks (the 13
tuning, the drift and four guard blocks; summary_apertures.csv), for every pair of footprints
closer than 2 m whose closest approach crosses open ground, the shortest passage joining them
measures the true gap from above, never below (a passage's ends lie on the outlines): gaps under
12.5 cm a median 6.6 mm over (90th percentile 2.9 cm), 0.125 -- 0.5 m 1.5 mm (5.9 mm), 0.5 -- 1 m
0.6 mm (2.2 mm); s 0.25 and 0.5 roughly 3x and 10x that. 97 -- 100% of the pairs have a passage
joining them; the rest, and a 1% tail of the gaps over 0.5 m (0.15 -- 0.42 m over), are pairs whose
closest approach the triangulation spans in two steps. Under closing a gap just under w can
therefore stay open by its error; the shy distance grows the outlines themselves and has none.

**The filtration.** A pocket's exit width is the widest path's bottleneck to the street (one
maximum spanning forest); a pocket walled in below it is a bar from its exit width to its fill (its
largest clearance, the Delaunay radius function as a width). A home is walled in at w when no open
ground within RING_M of its outline (Placement's ring: any triangle within 1 m, or the open band)
reaches the street at w. **Free first** on the complex: clear.free_first's objective on a region
graph (the components at w and the standing buildings, a building joined to a component through a
vertex on its own outline alone, since where two footprints meet the gap stays shut while either
stands), Dijkstra from the street at each building's population share, the cheapest D per person
first; each pass rebuilds the complex without the held buildings and re-runs the test, a home freed
wrongly going to the next pass.

**The agreement (the gate): passed.** All 302 blocks, s 0.125, against the raster under Fws
(summary_agree*.csv, summary_crossings.csv). At w 0 the raster encloses 185 homes on 60 blocks and
the complex walls in 73 on 30. Of the 122 only the raster encloses, 109 sit behind cracks under
12.5 cm (a sub-sample; the complex keeps them open), 8 behind gaps of 12.5 -- 17.7 cm that the
raster's 4-connected sub-samples cannot cross on a diagonal, and 5 reach a wide exit (0.35 -- 1.15
m) only through a triangle 0.95 -- 0.99 m from their outline, where the raster's ring, sampled at
cell centres, stops short. The 10 only the complex walls in (5 blocks) are sub-sample leaks: the
raster's 4-adjacency passes between footprints touching at a corner or across a wall thinner than a
sub-sample. Closing the complex at the raster's resolution (w 0.125) leaves 13 raster-only (the 8
diagonal, the 5 ring edges) and 17 complex-only (the 10 leaks, 7 behind gaps of 0.0001 -- 0.122 m
the raster samples open), on 18 of 302 blocks; at s 0.25 14 and 14, at s 0.5 39 and 14 (coarser
outlines cut corners). The sentinel and drift blocks agree exactly at w 0.125, the walled-in homes
and free first's crossings and D alike (17608 604:1003 D 0.00045; 20543 1194:812 D 0.00041; 30848
4217:3453 and 5041:575 D 0.00026; 22422, 5810, 9712 and 38366 none); at w 0 they differ by cracks
only (17608's three homes behind 5.5 -- 14.7 mm, 20543's 1194 behind 3.5 mm, 30848's 4217 behind
5.2 cm). Free first's crossings at 0.125: 47 of the raster's 61 free their home in the complex too;
of the other 14, 6 are crossings both models chose alike that free their home only with the pass's
earlier picks, 3 are the leaks, and 5 free it in the raster only through a mixed cell: lifted
spreads a cell's cleared share over all its footprint sub-samples, so clearing one building opens
part of a standing neighbour's share of a cell they share, and with each building's own sub-samples
cleared the raster does not free the home either. Of the complex's 70, 48 free their home in the
raster; the other 22 are 13 homes the raster does not enclose (the leaks and sub-resolution gaps)
and 9 conditional crossings as above. Every disagreement is the raster's resolution, so stage 2
ran. The mixed-cell spread is a small leak in today's metric, visible only through free first.

**The sweep** (302 blocks; s 0.125, 0.25 and 0.5; summary_sweep.csv, summary_exits.csv,
summary_bars.csv). Pooled over the 183,140 homes (area population, mean 1 a home), s 0.125:

    w (m)                 0       0.125   0.25    0.375   0.5     0.625   0.75    1.0     2.0
    homes walled in       73      189     279     399     581     848     1254    2710    18091
    their people (%)      0.025   0.068   0.10    0.15    0.22    0.32    0.46    1.02    7.5
    freed first: bldgs    54      -       204     277     382     536     726     1314    -
    small: blocks         19      -       45      57      76      89      101     137     -
      D median            0.0038  -       0.0048  0.0036  0.0034  0.0036  0.0045  0.0070  -
      D max               0.016   -       0.026   0.034   0.043   0.054   0.063   0.096   -
    large: blocks         11      -       29      35      40      45      46      53      -
      D median            0.0002  -       0.0009  0.0008  0.0008  0.0011  0.0014  0.0022  -
      D max               0.003   -       0.009   0.009   0.009   0.009   0.010   0.014   -

(blocks: those with a home walled in; D, free first's, over them; large: the 82 large and
oversized.)

The population behind each exit width (people per metre of width): a spike under 1 cm (2,056 /m:
walls traced a hair apart), a low floor of 300 -- 760 /m from 1 cm to 35 cm, then a steady rise
(1,060 -- 1,330 /m at 0.35 -- 0.6 m, 2,100 -- 2,500 at 0.6 -- 0.8, 4,300 -- 4,500 at 0.8 -- 1,
7,000 at 1 -- 1.25 and 15,200 at 1.5 -- 2). **No scale gap:** the walled-in count grows at every
step (x1.4 -- 2.2 a step from 0.25 m), free first's cost with it, and the trough between traced
walls and lanes (1 -- 35 cm) is shallow, not empty. The bars: the homes behind exits under 1 m sit
in real yards, not slivers (fill a median 2.4 -- 4.5 m wide; bar length, fill less exit, median 2.3
-- 3.7 m, 90th percentile 6 -- 7.5 m). Spacing: s 0.25 and 0.5 wall in 1% and 4 -- 14% fewer at w
0.25 -- 1 m, alike at 0. Free first frees every walled-in home at every width (no home left: the
homes the region graph freed wrongly, 3 -- 122 a width over the 302 blocks, every one taken by a
later pass, at most 3 passes); D over 0.05 on 1 -- 2 small blocks from 0.625 m. Cost: the complex
takes a median 4.1 s to build on a small block and 38 s on a large (at most 219 s, 1.6M triangles),
free first 0.3 -- 1.7 s and 4.5 -- 26 s a width; the s 0.125 sweep 7.4 CPU-h for all 302 blocks.

**The uniform law on the complex** (gap_flow). The complex refined by Steiner points on a
triangular lattice of spacing R (the largest pocket), graded by clearance (4 levels, 2x a level),
inserted one at a time with Lawson's flips, the outlines never flipped; under closing a Steiner
point keeps 0.75 w from every outline, so no narrow gap gains a point. Each triangle a pocket at
its circumcentre, a passage conducting sigma |e| / d, d the distance between the centres along its
normal: on a Delaunay mesh both lie on its perpendicular bisector, so the scheme is consistent for
isotropic diffusion. Pockets sharing a circumcircle (a rectangle split by its diagonal, common
along densified parallel walls) are one cell. An obtuse triangle's centre lies outside it: across a
passage harmless (consistency needs the centres' order, which Delaunay keeps), across a wall only
the potential's location moves, and at an exit whose centre lies beyond the street line the pocket
is ground. A passage beside a flat triangle (under 1e-4 |e|^2: a hairline crack's, which no flip
mends) or with a weight under 1e-6 is dropped. SuperLU (COLAMD) with iterative refinement. Demand
is Placement's: each pocket's 4 sub-points within 1 m of a home's outline whose nearest building it
is, the home's population spread over them by area, those cut off at w moving to its other ring
points, shadowed homes as in the raster. sigma pi / 2, the raster's isotropic limit. R converges at
first order (P0 at R 1, 0.5, 0.25, 0.125: 33717 under closing at 0.125 16,212, 16,128, 16,091,
16,075; 22422 under shy 0.5 86,427, 85,627, 85,347, 85,254; 20269 65,951, 64,722, 64,236 to R
0.25), the lenses faster (SIMP's default on 33717: L_P 0.1946, 0.1931, 0.1925, 0.1924). R 0.5 runs
everything below: P0 within 0.4 -- 1.5% of the limit, the lenses within 0.001 -- 0.004.

**The passable width's three forms** (gap_flow.Width, a Strategy named in every row: close<w>,
shy<w>, ramp<w>s<s>): closing (passages narrower than w closed; a filled part opens with min(1, the
sum of its flanking clearings)), the shy distance (every footprint grown by w / 2, the complex
rebuilt on the grown outlines: a lane of width W conducts W - w exactly) and the ramp (phi(d) =
clip((d - w / 2) / s, 0, 1), s 0.5 m, phi sampled by 16-point Gauss-Legendre on each passage).
Stage 1's topology is theirs alike.

**Fractional clearing** (gap_clear.Model, for SIMP). The arrangement of every outline and every w /
2 offset meshed as one, interiors included, each face carrying the buildings covering it and those
whose zone holds it; a triangle's factor is the minimum over them of phi(d_b) + x_b (1 - phi(d_b))
(the addendum's composition), a passage's the harmonic mean of its two triangles' (concave and
non-decreasing in each, so the mean stays concave), so P is convex in the clearing and its adjoint
gradient exact. Meshed interiors against one node per building (`lumped`), both at 0/1 clearings
against the complex rebuilt (gap_compare interiors, closing at 0.125, 15 clearings on 38136, 19263,
19570 and 20269): meshed within 0.0001 -- 0.007 of the rebuild on L_P (median 0.0005), one node per
building 0.013 -- 0.091 above it (SIMP's default on 20269 0.385 against 0.294, random clearings 30
-- 110% high); under the shy distance at 0.5 (9 random clearings on the first three) meshed within
0.0004, one node per building 0.006 -- 0.051 above: not materially as good, so meshed. Two defects
found on the way: polygonizing the noded outlines after clipping them to the region lost most open
faces (19570's model saw no clearing at all: faces 9,544 of the region's 16,038 m^2), fixed by
noding the region's boundary whole; grown outlines crossing a hair apart made needles no flip
orders, fixed by the 1e-7 m merge.

**The guards**, each run from the repo root and failing under its fault (exit 0 both ways; the
blocks 38136, 19263, 19570 and 19537). Stage 1 (gap_check.py): `conform` (every densified ring
segment an edge of exactly one triangle, none in three, the area to 3e-12, no centroid in a
footprint; fault `unconstrained`, scipy's Delaunay of the same vertices: edges cross footprints);
`delaunay` (no flippable passage with its centres out of order beyond 1e-9 of its length; fault
`geos`, GEOS's triangulation as it comes: 24 on 38136 at s 0.125; fault `flip`, every few flippable
passages of GEOS's flipped the wrong way and not mended); `apertures` (two footprints facing across
a gap g at 0, 17 and 45 deg, shifted so no corners face: the shortest passage within [g, sqrt(g^2 +
s^2)], 0.0502 for 0.05; fault `corners`, outlines not densified: the corner-to-corner 1.37 m);
`filtration` (a courtyard with two exits reads the wider, 0.8; a crack alone its 0.2; a sealed
courtyard -inf; the band inf; fault `narrowest`, the minimum spanning forest: 2.19 and 2.18);
`free` (a home whose cheapest crossing in the region graph is a shed leaving its two neighbours too
close: the rebuild catches it and the next pass frees the home; fault `no-rebuild`, the graph's
claim taken: the home left walled in). Stage 2 (gap_flow_check.py): `angle` (P's max / min over the
13 angles 1.0003 at W 1 -- 4 m, P / exact 0.9999 -- 1.0003; fault `centroids`, centroids for
circumcentres: 1.006 -- 1.036 and 1 -- 8% off); `shift` (8 lattice offsets: 1.0005; fault 1.05 --
1.13); `channel` (an open channel's P / exact 0.9999 -- 1.0003 at W 0.05 -- 4 m, 0 and 30 deg;
fault 0.76 -- 1.20); `field` (an open disk's P within 0.07 -- 1.4% of its closed form; fault 1.3 --
5% low); `law` (each form's width law on the channel, closing W, shy W - w and the ramp's G(W),
within 0.3%; fault `no-grow`, shy and the ramp without the growth: 0.33 -- 0.83 of it); `closed` (a
0.6 m lane conducts under closing at 0.5 and is closed at 0.7; fault `no-clearance`, Steiner points
let into the gap: it conducts at 0.7); `convex` (fractional clearing along random lines: the
factors' second differences at most 8.9e-16, P's at least 4.3e-3 of P, every form; fault
`multiply`, transmittances multiplied: the factors' second differences +0.03 -- 0.07); `gradient`
(q 1 and 3, J_1 and J_2, every form, within 4e-7 -- 1.5e-5 of the largest against the closer of the
central differences at 1e-5 and 1e-6; fault `min-grad`, a passage's gradient through the smaller of
its triangles': 6e-3 -- 0.35).

**The comparison** (gap_compare complex and iso, aperture_rows' lifted rows;
summary_compare_*.csv). The 13 tuning blocks; the stored clearings (SIMP's default, the cheap
preset; freed first where a model leaves a home walled in) and item 4's six road prefixes at D
0.05, in B's order on P, P_{1/2} and J_2:
- The discretization (the complex under closing at 0.125 against the raster's isotropic model: the
  same open geometry and placement, no heading): order tau median 1.00 on every lens (q25 0.93 --
  1.00, min 0.43 -- 0.50 on 1558); levels a median 0.010 lower (10th -- 90th percentile -0.036 --
  +0.005); the homes' escape times Spearman 0.998 (min 0.987). Where they part the raster is off:
  on the walled corridor (gap_compare lanes, 10 widths, 3 angles, 4 offsets) its isotropic model
  conducts a lane like an exact one narrower by 0.14 -- 0.29 m at each wall (a lane of 1, 2, 4, 16
  m like 0.60 -- 0.69, 0.79 -- 0.86, 0.89 -- 0.91, 0.96 -- 0.97 of its width; one of 0.25 -- 0.5 m
  like 0.28 -- 0.55 of it): beside a wall the lattice's oblique steps do not fit. The complex is
  exact within 3e-4 at every width. On 22422 the three homes holding 63% of the raster's P (692,
  1640 and 685, behind the gate the stored clearings open) leave through a 14 m lane 0.25 -- 0.4 m
  wide, which the raster resists about 4x as much as the complex (their demand's dissipation there
  73k against 19k, of 118k and 46k in all): their escape times read 2.2 -- 2.5x the complex's and
  SIMP's default 0.605 against 0.376 on L_P, the order still alike. On 1558 the clearance roads,
  which free homes whose way out is such a lane (1558's crack homes, the grid fix), score 0.11 --
  0.14 on the raster and 0.055 -- 0.084 on the complex, falling from above cycle_native and
  euclidean_grid to below them.
- The lift (the lifted metric against the raster's isotropic model, one raster): order tau median
  1.00 on P and J_2, 0.93 on P_{1/2} (q25 0.86 -- 0.93, min 0.79 -- 0.86: at most 3 of the 28 pairs
  swapped); levels a median 0.043 higher on L_P and 0.056 on L_J2 for the stored clearings (0.015
  and 0.009 for the roads); escape times Spearman 0.997 (min 0.992), the isotropic 0.86x the
  lifted. The lift's width law, on the corridor: a boundary layer at each wall that grows with the
  lane, 0.3 m in a 1 m lane, 0.5 in 2 m, 0.7 in 4 m, 0.8 -- 1.1 m in 8 -- 16 m (lanes of 1, 2, 4,
  8, 16 m conduct 0.33 -- 0.37, 0.48 -- 0.51, 0.66 -- 0.67, 0.79 -- 0.81, 0.86 -- 0.89 of their
  width's share; 0.25 -- 0.5 m lanes 0.09 -- 0.30): the runtime metric's superlinear width law is
  the lift's. The FFT homogenization study's finding (old step rule: the fast-turning scalar model
  0.04 -- 0.08 lower on 5810's clearings, 0.16 on 22422's gate, J_2) re-checked under Fws: the
  isotropic model reads 5810's SIMP and cheap clearings 0.036 and 0.022 below the lifted (L_J2
  0.230 against 0.266, 0.209 against 0.231), and the greedy's steps 1, 2 and 5 on the gated blocks
  (clear_rows_S0.01cat, J_2; gap_compare gates, summary_gates.csv) read: 22422's gate (step 1, one
  building, D 0.01) 0.575 isotropic against 0.556 lifted (the old rule's 0.534), steps 2 and 5
  within 0.005, so the 0.16 was the old step rule; on 30848 the isotropic reads steps 1 and 2 0.11
  and 0.17 higher and step 5 0.13 lower (0.280, 0.389, 0.575 against 0.168, 0.223, 0.705). On the
  complex 22422's gate scores 0.386 under closing at 0.125 and 0.004 under the shy distance at 0.5:
  its three homes' way out through the gate is the 0.25 -- 0.4 m lane, which the shy distance
  closes, their demand moving to their open sides, so 22422's gate is no gate at a passable width
  of 0.5 m.
- So the heading lift changes few decisions, fewer than the passable width does (below); a lifted
  complex was not built (the brief's step 3).
- The forms at w 0.5 (closing, shy, the ramp) against closing at 0.125: order tau median 0.86, 0.79
  -- 0.86 and 0.64 -- 0.86 (min 0.29, 0.21, 0.21); closing and shy at 0.5 order alike (median 1.00,
  min 0.81 -- 0.91), the ramp nearly (against the shy distance 0.93 -- 1.00). Levels, the stored
  clearings' L_P median: closing at 0.125 0.240 (SIMP's default) and 0.224 (cheap), closing at 0.5
  0.243 and 0.223, shy 0.277 and 0.251, the ramp 0.362 and 0.326; the raster's isotropic 0.276 and
  0.257, the lifted 0.347 and 0.333. Against the lifted metric the shy distance at 0.5 comes
  closest on levels (a median -0.003 on L_P and L_J2, 10th -- 90th -0.045 -- +0.036; closing at
  0.125 -0.033, at 0.5 -0.017, the ramp +0.022). The share of J_2 its top three homes hold at
  baseline (the dynamic range the cracks caused): the lifted median 0.19 (22422 0.96, 17608 0.60,
  1558 0.51, 38616 0.38, 30848 0.36), the raster's isotropic 0.08, the complex 0.04 under closing
  at 0.125 (22422 0.82), 0.04 -- 0.05 under closing and shy at 0.5 (22422 0.15), the ramp 0.15.
- Cost, one CPU core: the complex scores a block's 9 -- 10 fields in 340 -- 2,500 s at 1.2 -- 5.9
  GB (a field 30 -- 240 s, nearly all of it the mesh rebuilt in Python and GEOS; the baseline 52 --
  330 s), the raster's isotropic model (AMG-CG) in 130 -- 3,700 s at 1.3 -- 26 GB, the lifted
  metric 20 -- 330 s on a GPU at 2 -- 25 GB of host memory (on a core about 200 s a field on a
  large block, the grid fix). The complex has 0.47 -- 2.8M cells where the raster has 0.41 -- 5.5M
  (3 -- 42M lifted unknowns). The shy distance costs 1.0 -- 1.5x closing.
- The region 5810@major (12,663 buildings, 6.74 km^2) on one core: the complex's baseline under
  closing at 0.125 in 545 s and 11.8 GB (6.26M triangles), under shy 0.5 648 s and 12.8 GB, against
  the lifted metric's 34.8 GiB score on an H100 and the fine scalar raster's 252 s and 14.4 GB (FFT
  homogenization on 5810@major). Its two stored clearings (optimized on the lifted metric under the
  old step rule at a5x8, D 0.05) on the complex, a clearing 380 -- 470 s: L_J2 under closing at
  0.125 0.237 (SIMP's default) and 0.218 (the cheap preset), under shy 0.5 0.297 and 0.270, against
  the lifted metric's 0.301 and 0.271 and the fine scalar's 0.018 apart (NOTES, the deciding test).
  The order holds under both; the shy distance reproduces the lifted levels and gap (0.027 against
  0.030), closing at 0.125 the fine scalar's (0.018). Both clearings leave 0.06 -- 0.16% of the
  people walled in at the form's width (not freed first here).

**SIMP under the forms** (gap_simp; SIMP base fw0.q3.i10.t0.001.e0.0001.k1s1e-06, P, D 0.05, free
first, the winner rescored on the complex rebuilt; the 3 drift blocks and 10 small ones;
summary_simp*.csv). Each form's clearing scored under the others too:
- Under the shy distance its own clearing beats closing's on 9 of 11 blocks where they differ, by a
  median 0.004 on L_P (-0.022 -- +0.079; L_J2 0.006); under closing, closing's own beats the shy
  distance's on 8 of 11 by a median 0.001 (-0.075 -- +0.061): on 18918 and 40973 the shy distance's
  clearing scores 0.043 and 0.075 higher under closing than closing's own, so closing's step leaves
  SIMP a poorer landscape. Jaccard closing / shy a median 0.60 (0.29 -- 1; identical on 2 blocks),
  closing / ramp 0.41.
- Fewer, wider corridors? Under one flow (gap_simp cross: every clearing measured under both
  forms), the shy distance's clearing carries a little more of the flow in wide lanes than
  closing's: its share in lanes of 2 m or more +0.007 (median; up to +0.036), higher on 10 of the
  11 blocks where the clearings differ, the flow-weighted median lane +0.05 m (up to +0.18); the
  ramp +0.012 -- 0.016. But it clears as many buildings or more (median +1, up to +10) in as many
  clusters or more (median +0, up to +7): wider by a little, not fewer.
- Levels (each under its own form) L_P median 0.299 (closing), 0.342 (shy), 0.507 (ramp). Run time
  a median 150, 199 and 250 s a block (at most 2,100 -- 3,700 s on the drift blocks, 7.5 GB). On
  20543, the one block in both sets, the complex's SIMP base comes within 0.003 (closing) and 0.008
  (shy) of the stored SIMP default (polished on the raster) scored on the complex.

**The sweep's flow part** (gap_compare flowsweep; every block under closing at 0 -- 1 m and the shy
distance at 0.25 -- 1 m (the oversized at 0.5 only), free first at budget 1, then P, P_{1/2} and
J_2, each block's relative to its closing at 0.125, medians over blocks; summary_flowsweep.csv).
The 220 small blocks: closing moves P by -1% at w 0 (the cracks conduct), +8% at 0.5 and +21% at 1
m, P_{1/2} +13% and +42%, J_2 +22% and +65%; the shy distance moves P +17%, +40% and +92% at 0.25,
0.5 and 1 m, P_{1/2} +23%, +58% and +169%, J_2 +41%, +114% and +381%. The share of J_2 its top
three homes hold stays 0.15 -- 0.18 under closing (90th percentile 0.26 -- 0.29) and 0.17 -- 0.20
(0.29 -- 0.37) under the shy distance; on the 104 small blocks with the lifted metric's per-home
rows (aperture_rows) the lifted metric's is 0.17 (0.40) and closing at 0.125's 0.145 (0.26). The 60
large blocks: closing +4% on P at 0.5 and +14% at 1 m (P_{1/2} +8%, +26%; J_2 +12%, +42%), the shy
distance +11%, +22% and +57% at 0.25, 0.5 and 1 m (J_2 +27%, +60%, +210%); the 22 oversized barely
move (closing +0.6% and +1.7% on P at 0.5 and 1 m, the shy distance at 0.5 +5%, J_2 +11%). The top
three homes' share of J_2: on the large blocks a median 0.04 at every width and form, its 90th
percentile 0.47 at w 0, 0.19 at 0.125, 0.14 at 0.5 and 0.13 at 1 m under closing, 0.15 -- 0.20
under the shy distance; on the 35 large blocks with the lifted metric's rows the lifted's 90th
percentile is 0.46, the complex's 0.44 at w 0 and 0.16 at 0.125. So the dynamic range the cracks
caused is the cracks': the complex reproduces it with every crack open (w 0) and loses it once gaps
under the raster's own sub-sample close, the passable width beyond that moving the levels, not the
tail. Free first at every width costs a median D of at most 0.003 (the walled-in population at most
0.5% on the median block).

**Pedestrian buffers** (the shy distance's w / 2 a wall). The HCM's effective walkway width deducts
a shy distance of 1.5 ft (0.46 m) from a curb or a fence or low wall, 2 ft (0.61 m) from a building
face and 3 ft (0.91 m) from window displays; Habicht and Braaksma (J. Transp. Eng. 1984) measured
14 -- 22 cm in a tunnel against the 46 cm then usual in design, and Bosina, Weidmann and colleagues
(ETH 2016) a density-dependent distance kept from walls; single-file passage takes about 0.55 m a
person (Seyfried et al., bottleneck experiments). Shy at 0.5 (0.25 m a wall) sits between the
measured tunnel figure and the HCM's deductions; the raster's lattice loses 0.14 -- 0.29 m a wall
by itself, the lifted metric 0.3 -- 1.1 m.

**Decision support** (the owner decides).
- A natural width: none. Enclosure, free first's cost, the levels and the method order all move
  across every step of the sweep; the exit widths' only feature is a shallow trough from 1 to 35 cm
  between traced walls and lanes. A width has to be chosen on other grounds: the trough's edge
  (0.35 m) clears the traced walls and little else; a body's width (0.45 -- 0.55 m) and practice's
  buffers argue for about 0.5 m.
- The complex reproduces the raster where it should: the same enclosure and free first up to the
  raster's resolution on all 302 blocks, the same method order (tau median 1.00) and escape-time
  ranks (0.998), exact channels where the raster loses 0.14 -- 0.29 m at each wall; the levels part
  where narrow lanes carry a home's way out (22422, 1558), and there the complex is the one that
  measures the geometry. Cost: one core, 3 -- 6x less memory than the raster, about the raster's
  CPU time per block, 30 -- 240 s per clearing (the rebuild); slower than the lifted metric on a
  GPU.
- The heading lift: few decisions (order tau median 1.00 / 1.00 / 0.93, min 0.79), the levels
  (+0.04 -- 0.06) and the dynamic range (the top three homes' share of J_2 0.19 against 0.08)
  through its wall boundary layer; the shy distance at 0.5 reproduces its levels on the 13 (median
  -0.003).
- Recommendation: the complex for passability and enclosure now, with the shy distance at 0.5 m as
  the passable width (Placement's walled-in test and free first's graph read the gaps exactly where
  the raster's resolution decides the enclosure of 0.07% of the people, for 4 -- 38 s a block);
  then the runtime on the complex, isotropic, the raster and the lift retired. On the 13 it keeps
  the raster's method order (tau median 1.00) and, with the shy distance, the lifted metric's
  levels, measures lanes exactly where the raster loses 0.14 -- 1.1 m at each wall (the decisions
  that move, 22422's gate and 1558's clearance roads, move toward the geometry), is convex with an
  exact gradient (SIMP runs on it), needs 3 -- 6x less memory and runs a region on one core
  (5810@major's baseline in 9 minutes, 12 GB). What the runtime needs first: a clearing re-meshed
  locally (today the whole complex is rebuilt, 30 -- 240 s a field, which the greedy, the polish
  and the swings cannot afford), and those optimizers moved (SIMP is). Keeping the raster keeps its
  lattice's width law (0.14 -- 0.29 m a wall), its angle error (1.21 at 1 m) and the mixed-cell
  leak, and a passable width would have to be folded into its sub-samples, a sub-sample coarse.

Compute: no GPU. The CPU nodes (orion, quadro2; single-threaded tasks): the sweep 13.2 CPU-h (906
tasks), the comparison on the 13 25.2, SIMP under the forms 8.6, the flow sweep 157 (4,122
block-forms; an oversized block's 40 min and up to 64 GB), and about 15 more lost to duplicated and
cancelled tasks (a resubmission at a fix, the large blocks' sweep split across the two nodes):
about 220 CPU-h. Here (single-threaded, at most 8 processes): the agreement 6 CPU-h, the raster's
isotropic model on the 13 4.2, the gate re-check 1.9, R's convergence 1.9, the region 1.1, the
guards about 4, the rest (interiors, lanes, apertures, the corridors, the diagnoses) about 3, and
the development before them about 10 [inferred]: about 30 CPU-h.

**Decided** (owner 2026-10-14, "that SGTM"): the shy distance at 0.5 m on the complex, as the
metric and its runtime, the raster and the lift retired once the complex is measured to dominate
them. Two questions the owner asked with it. A vehicle's width as the shy distance: no, for the
walking metric; at 3 m the lanes narrower wall in 27% of the small blocks' people, 16% of the large
blocks' and 2% of the oversized blocks' (summary_exits.csv, s 0.125), so enclosure, ranked first,
would become vehicle access and P its tie-break, and free first would have to build vehicle roads;
vehicle access is the same filtration read another way (a home's exit width is its widest path's
bottleneck, the widest vehicle that reaches it, turning ignored), to be reported over W. A GPU
path: only where measured faster. The complex's systems are planar and factor directly on one core
(SuperLU 1.5 s at 690k unknowns, where pyamg takes 16 -- 69 s a solve, the conductances spanning
six decades), so the iterative solve a GPU speeds up already loses on a core; a field's 30 -- 240 s
is the mesh rebuilt in Python and GEOS, which the fixed mesh removes; and a GPU direct solver
(cuDSS) is a new dependency, piloted only if the factorization dominates once the mesh is fixed.

### The complex as the runtime: one fixed mesh (owner 2026-10-14: "that SGTM")

BACKLOG 4, decision (3): the runtime's first phase, the gap complex made the model the optimizers
run on, calibrated by equality against the rebuild and timed (the brief:
.superpowers/sdd/2026-10-14-complex-runtime). The metric is `cx+shy0.5+s0.125r1b1+R0.5g4x2q4`: the
shy distance at w 0.5 m, the complex's Spec (s 0.125, ring and band 1 m) and the FlowSpec (R 0.5, 4
levels graded 2x, 4 demand sub-points a triangle): every configurable constant that can move a
score is in the name, which names the rows' directories (relax_rows/<metric>/...,
polish_rows/<metric>/...); the code's own constants (the 0.1 mm snap, the merge and flatness
tolerances, MESH_REV) are in the mesh's cache key with them, so a change to one of those needs a
new name by hand. Code: gap_clear.py (the fixed mesh, the model, its eps world, the metric),
seam.py (the optimizers' seam), metrics.py (the metric resolved at the edge), gap_clear_check.py
(the guards), gap_calib.py (calibration, acceptance, timing; gap_rows/summary_runtime_*.csv,
committed; the rows are not), gap_cudss.py (the GPU pilot). No new dependency.

**The seam** (seam.py). What search.py, relax.py, polish_greedy.py and free first read from a
clearing model, written down as Protocols: `ClearingModel` (the clearing `removed`, `n`, `cost`,
the footprints and their tree for the spaced picker, `baseline`, `exact`, `enclosed_left`,
`tension(power, restore=)`, `free_first`, `relaxation`, `release`), `Tension` (`g`, and the
catchment's view of its state: `weight` and `flow`, the operator, the potential, the row sums, each
home's load and the candidates' member nodes), `Relaxed` (the eps world of relax.simp, the
incumbent and the polish: proj, value, value_sgrad, exact, perm, `metric_world`) and `Metric` (the
name, the catchment sweep's device, a block's model, the coarse grid a plan's .c / .C asks for,
which the complex refuses). The raster's clear.Clearing and relax.Relaxation and the complex's
gap_clear.Model and Relaxation both satisfy it; clear.Catchment sweeps a `Flow`, so the cheap
preset's picker runs on either. The metric is resolved once where a run's configuration is read
(metrics.metric_of: a name beginning `cx+` the complex's, on the host; anything else a raster
conductance on its device and grid) and handed down; below that edge no code asks which it has. The
raster stays a live alternative until phase 2 measures dominance, and its numbers did not move:
width_check.py's identity (the raster scorer's baseline, a clearing's operator, P, J_2 and P_{1/2},
the relaxation's value and gradient, the tension, current_map's flow, the sightline metric's)
passes bit for bit on its three blocks, and the catchment's gram on 19570 (40 candidates) is bit
for bit the code's before the seam. gap_clear.check_disjoint refuses a complex metric whose name
reads as a raster conductance (with or without a search's @<along>), so no row the raster wrote can
be read as the complex's; stored clearings are sets of buildings and stay valid across metrics.

**The mesh** (gap_clear.full_mesh; Fixed.of caches it). Every outline, its w / 2 offset (the shy
distance's zero zone) and the band's line noded together by snap-rounding at 0.1 mm (OverlayNG on a
grid: snapping the noded set afterwards left dangles where offset arcs nearly coincide, and a crack
between two footprints joined the open ground, 38136's homes 3% low), densified to s and
polygonized, the faces outside the region dropped (a street inside the block, its band a hole of
the region, came back as a face: 23597's 8,685 m^2 of band meshed and 470 m of exits lost); each
face triangulated by GEOS's constrained Delaunay and finished by Lawson's flips within it, Steiner
points on the rebuild's own lattice (laid over the baseline open ground's box, kept R / 2 clear of
every line of the noding), flat triangles dissolved into their neighbours. A clearing then only
re-weights passages: a triangle conducts by the least k_b of the footprints whose zone holds it, a
passage by the harmonic mean of its two triangles', so after the baseline a state's score is
assembly, factorization and solves, with no GEOS call and no meshing (0/1 for the greedy,
fractional for SIMP, one code path). The factorization uses one geometric nested-dissection order
of the whole cell graph, computed once and restricted to each state's unknowns; every solve is
refined to the backward-error floor. The cache key is the block's footprints
(clearings.fingerprint), streets and boundary, every meshing constant and MESH_REV. On the 18
calibration blocks a mesh builds in 34 -- 323 s at 2 -- 10 GB, 1.3x the rebuild's baseline, and
loads in 0.1 -- 4.4 s (the timing blocks); it holds 1.9 -- 3.0x the rebuild's triangles (the
interiors and strips) and 0.95 -- 0.98x its unknowns at baseline.

**Conformity: no clip.** A passage across a face boundary whose circumcentres lie in the wrong
order (the study's CROSS_MIN clipped the 0.3% that were) lies in open ground once a clearing opens
both faces, where a clipped weight makes the two-point scheme inconsistent. So every face-boundary
edge two triangles share whose centres are out of order, or within MERGE_REL of its length of each
other, is split at its midpoint and the faces re-flipped, round after round until none is
(`_conform`: 355 -- 6,694 splits in 4 -- 14 rounds on the 18 blocks). The mesh is then Delaunay
outright and every passage's weight sigma |e| / d_e is non-negative by construction; a passage
still out of order is refused when the weights are made (`_weights`), and a negative conductance in
any state assembled raises (`_reach`), so every state scored is checked. Its cost allows it:
against the same mesh without the splits and with the study's clip (the out-of-order passages'
centre distances clipped at 0.05 |e|: 70 -- 841 passages of 0.12 -- 2.2M on 19263, 38136, 19570 and
20269), conformity's splits cost nothing measurable (builds of 1.8 -- 47 s, at most 3 s apart
either way over two runs), and the clip itself, measured anyway on four random clearings a block,
moves L_P and L_J2 by at most 2.8e-5 where both sit within 1.3 -- 1.7e-4 of the rebuild: harmless
at these blocks, but a weight no longer consistent wherever a clearing opens it. The `weights`
guard: no negative weight and no passage across faces sharing a centre on 19570 (508k passages,
1,737 splits) and 20269 (2.2M, 1,693), every conductance non-negative at six states (the baseline,
the eps world, a fractional state, three random clearings) and u0 > 0 for every home with demand
(P_{1/2} needs it); with the splits skipped the mesh is refused (841 and 728 passages across faces
out of order).

**Calibration** (gap_calib baseline and clearings; summary_runtime_baseline.csv,
summary_runtime_clearings.csv).
- The baseline on 18 blocks (the 13, the large 17608, 30796 and 38366, the oversized 32841 and
  7851) against the rebuild (gap_compare.ComplexModel: gap_flow's mesh under the shy distance at
  0.5): P0 a median 1.1e-4 low and within 2.3e-4 on 16 (P_{1/2},0 within 4.0e-4, J_2,0 5.8e-4); the
  homes with no demand alike on all 18; each block's median home escape time within 2.4e-4. They
  cannot be equal: the fixed mesh triangulates the same open ground differently (its faces share
  their boundaries with the interiors and strips, conformity splits those boundaries, the outlines
  are snap-rounded at 0.1 mm), and 1e-4 is far below the scheme's own error (P0 within 0.4 -- 1.5%
  of its limit at R 0.5: the study).
- The outliers are necks. On 46841 (P0 -2.7e-3, P_{1/2},0 -1.0e-2, J_2,0 -1.2e-2) four homes (6,
  51, 222, 1026; 97% of P0's difference) leave through a neck 2.1 mm wide between footprints 222
  and 1026, which stand 0.50097 m apart and are each grown by 0.25 m; both meshes have it, equally
  wide. Flow through such a gap is set within a few centimetres of the neck (between two arcs of
  radius 0.25 m its width doubles 2.3 cm either side), which s 0.125 does not resolve; conformity's
  splits refine those arcs on the fixed mesh [inferred]. The four escape times read 18 -- 23% below
  the rebuild's at s 0.125, and refining the outlines moves the rebuild toward the fixed mesh: P0
  8,480.7, 8,469.9 and 8,465.3 at s 0.125, 0.0625 and 0.03125 (home 222's escape time 57.4, 51.3,
  48.4), the fixed mesh's 8,458.2 and 8,456.9 at s 0.125 and 0.0625 (44.4, 43.9), so of the two the
  fixed mesh is the nearer its limit there. On 30848 (P0 -1.1e-3) the difference is spread over
  many homes (the top five hold 18% of it): necks of the same kind (3585 and 1010, 0.5035 m apart,
  escape times 4% low) and, the other way, a few homes up to 19% high (2137, whose ring reaches the
  band; not traced: its demand points lie off the band in both models, so it is not the demand's
  placement); its clearings' lenses are within 2.3e-4 all the same.
- Clearings: every stored clearing of both presets under both metrics on the 13 (52, with the 13
  baselines) on the fixed mesh against the rebuild: L_P within 2.3e-4 (median 7e-5), L_{P_1/2}
  within 3.3e-4 and L_J2 within 2.5e-4 on 12 of the 13; 46841's neck carries its baseline
  difference into every lens (1.5 -- 1.7e-3 on L_P, 5.1 -- 5.4e-3 on L_{P_1/2}, 3.2 -- 3.4e-3 on
  L_J2). The enclosed population and count are equal at every state. The study's 0.0004 (meshed
  interiors against the rebuild, shy 0.5) holds everywhere but the neck. An exact score takes
  1.2 -- 10.6 s on the fixed mesh against the rebuild's 18 -- 164 s (16x, 10 -- 23x).
- Reach: the homes the flow model reaches equal those the complex grown by w / 2 reaches at width
  0, home for home, at all 65 states scored (and at the `reach` guard's seven states a block).
- Each optimizer's stored score is its scorer's: the complex's rows and the fixed mesh's L_P of the
  same clearing agree to 1e-15, the raster's rows and gap_calib's re-scoring of them on the raster
  (rasterscore, on a core) to 3e-8.

**The guards** (gap_clear_check.py; each passes on the code as it is and fails under its fault,
both ways on the final code; 19570 and 20269 unless named):
- `baseline`: P0, P_{1/2},0, J_2,0 within 2e-3 of the rebuild, the no-demand homes alike, every
  escape time within 5%, the three most populous buildings' single clearings' L_P within 2e-3
  (19570 -5.4e-5, +1.3e-5, -7.4e-5, homes within 1.8e-2, singles at most 9.1e-5; 20269 -1.6e-4,
  -2.1e-4, -3.5e-4, 2.8e-2, 3.9e-6). Fault `interior` (the most populous building's interior faces
  dropped): its single clearing's L_P 3.4e-2 and 4.0e-3 off. Fault `exits` (the band's faces read
  at the unsnapped tolerance): P0 11x and +68%.
- `weights`: above.
- `reach`: no home apart at the baseline, three single clearings, two random clearings and free
  first's state. Fault `threshold` (a ring with no triangle reaching the street read as reaching
  it): every walled-in home apart (7 on 19570, 1 on 20269). The brief's fault, the width threshold
  off by one comparison (gap_complex.walled read `<=`), goes unseen under the shy distance: its
  filtration is read at 0 on the grown complex, where a home's exit width is -inf or positive and
  never 0; the ring's test is the comparison that decides reach there.
- `gradient` (q 1 and 3, J_1 and J_2, every form, at a random x and at SIMP's start, against the
  closer of the central differences): within 3.9e-5 of the largest on 19570 and 38136. Fault `tie`
  (a tie's gradient given to one building): 6e-3 -- 0.10 at the start.
- `free`: free first frees every walled-in home and the flow model agrees (20269 one home, 23597
  six). Fault `drop-crossing`: the flow model leaves a home walled in (198, 45) and
  Model.free_first raises.
- `identity`: the names spell back; the 7 raster names (every directory under relax_rows/ and
  polish_rows/, and the conductances) are refused as a complex metric's. Fault `collide`
  (check_disjoint off): all 7 accepted.
- `convex` (the study's, moved here): the factors' second differences at most 7.8e-16 along random
  lines, P's at least 2.7e-3 of P (19570, 38136). Fault `multiply`: 0.023 -- 0.058.
- The study's guards (gap_check.py: conform, delaunay, apertures, filtration, free;
  gap_flow_check.py: angle, shift, channel, field, law, closed), whose modules this phase touched,
  pass and fail under their faults as before.

**The acceptance run** (gap_calib preset on orion and quadro2, one core a task;
summary_runtime_acceptance.csv, summary_runtime_runs.csv). The cheap preset (S0.01cat.P64w8) and
SIMP's default (fw0.q3.i10.t0.001.e0.0001.k1s1e-06.p256w8x2) at D 0.05, free first, optimizing P,
area population, on the 13, on the complex; against the raster's clearings of the same presets
under uni+apwfs (the Store holds no P runs of them, so they were run on the GPU), every clearing
scored in both metrics:
- Each metric prefers its own optimizer's clearings, in B's order (the population left walled in
  first, then the lens). On the complex its SIMP is ahead of the raster's SIMP clearing on 9 of the
  13 and its cheap preset on 12: on 5 blocks the raster's clearings leave 0.0001 -- 0.0012 of the
  people walled in on the complex (its free first frees the raster's walled-in homes, not the
  complex's), and where enclosure ties L_P is a median +0.0004 (-0.003 -- +0.028) for SIMP and
  +0.006 (-0.006 -- +0.058) for the cheap preset. On the raster the raster's SIMP is ahead on all
  13 (L_P a median +0.010, up to +0.53) and its cheap preset on 11 (+0.010, -0.004 -- +0.42). The
  other lenses alike (on the complex SIMP ahead on 8 and 9 of 13 on P_{1/2} and J_2, the cheap
  preset on 12 and 12).
- Levels, median over the 13, on the complex (its own clearing / the raster's): SIMP 0.287 / 0.275
  on L_P, 0.333 / 0.334 on L_{P_1/2}, 0.310 / 0.312 on L_J2; the cheap preset 0.266 / 0.256,
  0.324 / 0.314, 0.301 / 0.290. On the raster the complex's clearings read 0.280 (SIMP) and 0.313
  (cheap) on L_P against the raster's own 0.378 and 0.364. They part most on 22422 and 23597, where
  the raster's clearings open the 0.25 -- 0.4 m lanes the shy distance closes (the study's gate):
  0.61 and 0.43 on the raster against 0.16 and 0.24 on the complex.
- The method order: in B's order SIMP leads the cheap preset on the complex on 12 of 13 with the
  complex's clearings (L_P median +0.015) and on 11 of 13 with the raster's, as the raster's SIMP
  leads its cheap preset on the raster on 13 of 13 (the complex's clearings there on 9). The four
  clearings' order in B's order, metric against metric (Kendall tau per block), is a median 0.00 on
  every lens (-1.00 -- +1.00): each metric ranks its own optimizer's clearings first, while the
  presets keep their order.
- Jaccard, the complex's clearing against the raster's of the same preset: SIMP a median 0.63
  (0.37 -- 0.79), the cheap preset 0.48 (0.10 -- 0.64).
- No run leaves a home walled in on its own metric, and the complex's clearings leave none on the
  raster.
- Time and memory, one core a block: SIMP 820 -- 11,543 s (median 1,841), the cheap preset
  175 -- 1,858 s (median 699), at 2.2 -- 12.5 GB (SIMP's rows; the cheap preset's task logs within
  0.6 GB of them); the raster's on a GPU (H100 or the tesla node) 26 -- 337 s and 8 -- 166 s. The
  complex is 3.7 -- 73x slower for SIMP (median 11x) and 4.7 -- 40x for the cheap preset (median
  13x), at about half the raster's host memory (2 -- 25 GB: the study). The 26 runs took 12.7
  CPU-h. Of SIMP's 3.2 h on 5810 the continuation (four stages of at most ten iterations, each with
  its incumbent's roundings scored) took about 5,000 s and the polish's 88 scorings the other 6,500
  [inferred from the log's times], every one an eps-world factorization of all 6.8M cells.

**Search or model.** In B's order the complex's optimizers end behind the raster's clearing on the
complex's own metric five times: SIMP on 18895, 33717, 9712 and 9717 (L_P 0.003, 0.0003, 0.0006 and
0.0012 behind) and the cheap preset on 33717 (0.006). It is the search, not the model's accuracy.
The model scores those clearings as the rebuild does (within 2.3e-4), and its gradient is the
adjoint's (the guard). Started from the raster's clearing (free first's building added and held
where the complex has one), the complex's own polish holds it or climbs from it, every time above
the complex's own result: SIMP's polish (P256w8x2, every building movable) takes 18895's from
0.2997 (with free first's building; the raster's clearing alone 0.3007, the complex's own 0.2977)
to 0.3026, 33717's from 0.2224 to 0.2233 (own 0.2221), 9712's from 0.1232 to 0.1234 (own 0.1226)
and holds 9717's at 0.1549 (own 0.1537); the cheap preset's polish (P64w8) takes 33717's cheap
clearing from 0.1990 to 0.2000 (own 0.1930). The complex's own clearings are local optima of the
same polish (20269's cheap one does not move even under two-for-one exchanges with four times the
tries), so which basin a run ends in is decided by the greedy's path or SIMP's continuation, both
starting from first-order information. That information is sound: the tension's ranking of single
clearings at the greedy's start against their exact gains has Spearman +0.84 on 20269 (all 1,045
buildings; +0.80 per person; the exact best in its top ten, its own best worth 0.92 of the best per
person) and +0.98 on 33717 (its top 100 and 100 others, a sample that flatters it; +0.94 per
person; the exact best in its top ten, its own best worth 0.75 of it), against the raster's
+0.93 -- 0.95 (NOTES, "Tension ranks like exact LOO": 41132 and 18537, another metric and other
blocks); it reads a single clearing's gain a median 3.4 -- 3.7x high (the slope at eps against the
whole step). Where it is least faithful is the shy distance's minimum: on 20269 the 821 buildings
whose zone shares a triangle with another's (a gap both shut, opened only by clearing both, whose
first-order gain the tie splits between them) rank at +0.83 and read 4.1x high, the 224 alone at
+0.97 and 2.5x, which makes the first-order searches' early choices noisier on the complex
[inferred], a property of the metric rather than an error in the model. The raster's searches end
below another clearing on their own metric too (its cheap preset behind the complex's clearing on
18895 and 5810).

**Timings** (gap_calib timing; summary_runtime_timing.csv; one core of ai01, each twice, the mean):
a random clearing at D 0.05, its exact score (the scorer's: the lenses' two solves), a SIMP
iteration (value_sgrad in the eps world at 1e-4, q 3) and a tension, on a small block (19570), a
large one (5810), the two oversized (32841, 7851), and 5810@major's baseline. AFTER is the runtime;
BEFORE the mesh as first run (SuperLU's COLAMD at every factorization, each solve refined toward a
residual of 1e-12, eight steps where the floor lies above it).

    block       unknowns: open / all    exact score  factor  solves  assembly  reach  terms
    19570       75k / 0.33M                    0.27    0.16    0.03      0.03   0.02   0.01
    5810        2.85M / 6.77M                  21.7    13.8     4.0       1.5    1.0    0.5
    32841       2.02M / 4.28M                  13.1     7.7     2.7       1.0    0.7    0.3
    7851        1.39M / 2.60M                   9.1     5.6     1.8       0.6    0.4    0.2
    5810@major  6.06M / 15.4M                  32.9    19.9     6.2       2.5    1.7    1.1

    block         SIMP iteration  factor  solves  gradient  assembly  reach
    19570                   1.52    1.22    0.07      0.07      0.08   0.05
    5810                    68.0    55.4     4.8       2.2       2.6    1.7
    32841                   31.2    23.3     3.0       1.4       1.5    1.0
    7851                    19.4    15.2     1.6       0.7       0.9    0.6

(seconds; 5810@major's is its baseline's score; a tension costs what a SIMP iteration does). The
factorization is 57 -- 63% of an exact score and 75 -- 81% of a SIMP iteration. BEFORE took
1.6 -- 1.8x as long for an exact score and 1.4 -- 1.7x for a SIMP iteration (5810: 35.0 and 105 s),
most of it the refinement (5810's solves 13.0 s against 4.0) and the rest the order (SuperLU's
COLAMD 18.2 s against the dissection's 13.8). A mesh builds in 18 s (19570), 236 -- 344 s (the
oversized), 496 s (5810) and 1,133 s at 22 GB (5810@major), loads in 0.1 -- 4.4 s; the model's
baseline (the order and four solves) 3 -- 78 s. A SIMP iteration factors every cell, the interiors'
and strips' too (they conduct eps): 1.9 -- 4.4x the open ground's unknowns.

**CPU first, and the GPU (the owner's question).** On the host the order is computed once and every
factorization reuses it; SuperLU in scipy keeps no symbolic factorization to reuse, and CHOLMOD
(scikit-sparse, symbolic reuse and rank updates) needs the system's cholmod.h, so its pilot was
skipped. Since the factorization dominates, the GPU pilot ran (gap_cudss.py: nvmath-python 1.0 and
cuDSS from their wheels, in an ephemeral environment, `uv run --with`, nothing added to the
project; one H100 PCIe, 0.21 GPU-h in all) on 5810's systems as the runtime builds them, a random
clearing's (x: 3.03M unknowns, 11.2M non-zeros) and the eps world's (e: 6.77M, 27.4M), each three
times, against SuperLU in the runtime's order on that node's host:

    system   cuDSS: plan  factorize  solve  refactorize  solve  SuperLU: factorize  solve
    x               15.7      0.026  0.011        0.025  0.011                6.05   0.30
    e               39.5       0.08   0.03         0.08  0.025                23.6   0.77

(seconds; residuals alike, 3 -- 9e-9 of the right-hand side before refinement). The plan
(reordering and the symbolic factorization, on the host, single-threaded without a threading
library) is paid once per sparsity pattern; a refactorization with new values on the same pattern
then takes 0.025 -- 0.08 s, 240 -- 290x SuperLU's. In the eps world every cell is an unknown at
every state, so one plan serves a whole SIMP run or every tension; an exact score's unknowns change
with the clearing, and keeping one pattern there means assembling every cell, the walled-in ones as
decoupled identity rows and a closed passage as an explicit zero, which changes no score [inferred:
not built]. With the factorization at 0.1 s a clearing's time is the host's: on 5810 everything but
the factorization is 7.9 of an exact score's 21.7 s and 12.6 of a SIMP iteration's 68 s, so about
3x and 5x as the host work stands [inferred], more once that work is cut too (on a fixed pattern
assembly is one product with a fixed incidence). So yes: a GPU direct solver could speed the method
greatly, SIMP and the tension most, at the price of a new dependency (nvmath-python, cuDSS's
wheels) and a GPU; the CPU path stays as it is, one core and scipy. Offering both (the
factorization as a Strategy chosen where the run is configured) is the owner's decision.

**Phase 2 needs**: the remaining ports (road re-scoring, which needs roads carved into the fixed
mesh; pweight; the atlas and corridor pictures; clearings_check; the exports; the translucent
search, which is the raster's alone: the complex has no search conductance), the dominance
measurement against the raster and the lifted metric and the retirement (each retired variant to
the wiki's dead-end record with its numbers), the region test on the complex (5810@major: a
19-minute build, 33 s a score, 22 GB), the consolidated pass, and the speed before them: the
owner's decision on a GPU path; the polish's moves scored exactly rather than in the eps world
(more than half of SIMP's time on 5810; an exact score factors 2.3x fewer unknowns there [inferred:
not built]); the eps world's interiors and strips (58% of a SIMP iteration's unknowns on 5810,
conducting eps: a coarser lattice inside the footprints would cut them [inferred]); the host's
assembly, reach and gradient on a fixed pattern. Compute: on the CPU nodes the acceptance 14.7
CPU-h (the 26 runs' own time 12.7, the rest a pilot and failed starts), the timings 1.7 and the
raster's clearings re-scored on a core 16.4; on the GPUs the raster's presets 1.4 GPU-h and the
pilot 0.21; here (single-threaded, at most 8 processes) the baseline and clearings' calibration 2.6
CPU-h and the guards and diagnoses about 4 [inferred], besides the development.

### The complex as the runtime: its speed and a GPU path (owner 2026-10-14: "Yes, add a GPU path")

BACKLOG 4, decision (3): the runtime's phase 2a (the brief:
.superpowers/sdd/2026-10-14-complex-speed): the complex's factorization a Strategy with a GPU path,
assembly on one fixed pattern, the host's own work cut, two result-changing options measured, and a
run identity that carries the code's constants. The metric is now
`cx+shy0.5+s0.125r1b1+R0.5g4x2q4+v34d40e8a` (phase 1's, its rows migrated). Code: gap_factor.py
(the Strategy), gap_clear.py (the fixed pattern, the identity, free first's cached complex),
search.py (the polish's scoring), gap_clear_check.py (the pattern, constants and device guards),
gap_calib.py (presets per device, speed, rescore;
gap_rows/summary_runtime_{presets,speed,rescore}.csv, committed; the rows are not), cluster.py
(`--complex`). The device moves scores at the floating-point level only, so it is not in the
metric's name; every run's rows record it.

**The factorization, a Strategy** (gap_factor.Factorization). What the model needs from a
factorization is three calls: a plan per sparsity pattern (`plan(Pattern)`), a refactorization with
new values on it (`Planned.factor(data, unknown)`) and solves (`Factor.solve`, each refined to the
backward-error floor by the one `Refined`, moved here from gap_flow with the dissection). SuperLU
on a core: the plan is phase 1's geometric nested-dissection order of the whole pattern and the
pattern permuted into it; a factorization restricts the values to the state's unknowns, which keeps
the order restricted to them, drops the explicit zeros (`_restrict`) and factors that, pivoting on
the diagonal, as phase 1 did. cuDSS on a GPU (nvmath-python's DirectSolver, SPD, the operands on
the host): the plan is cuDSS's reordering and symbolic factorization of the whole pattern, made at
the first factorization; every later state resets the operands and refactors on it, every cell a
row. A plan holds one factorization at a time, and a factor solved after a later one on its plan
raises. The device is chosen once where a run is configured (metrics.metric_of from the scripts'
`cpu|gpu` argument: gap_factor.factorization_of) and handed down in the metric; relax.one's,
polish_greedy's and search's rows have a `device` column (seam.Metric.device).

**The dependency.** `nvmath-python[cu12]==1.0.0` and `nvidia-cudss-cu12==0.8.0.10` in the `gpu`
group, pinned as cupy is (linux only); uv.lock adds with them cuda-bindings 12.9.9, cuda-core
1.2.1, cutensor-cu12 2.8.1 and nvidia-cuda-nvcc-cu12, -cufile-cu12 and -nvfatbin-cu12, and moves
nothing else; cupy stays (the raster's GPU path). It loads in the project's own environment with no
workaround: cuda.pathfinder finds libcudss in the venv's site-packages, here and on the cluster
after `cluster.py setup` (on ai06's H100 and ai04's Quadro RTX 8000). The pilot's preloading of
every CUDA wheel and its ephemeral `uv run --with` environment answered the overlay's problems, not
the libraries'; gap_cudss.py is deleted. nvmath ships no types: a mypy override, as numba's.
cluster.py's `blocks` kind takes `--complex` (no raster grid: the card sized by `--gb` alone).

**The fixed pattern** (gap_clear.Graph, System). Each cell structure of the fixed mesh has one
pattern: every pair of cells a static passage joins (the pair's conductance `W` the sum of its
passages') and the diagonal. A state's reach is a search from the street through the pairs that
conduct (`_bfs`; in the eps world, where every pair and street face conducts, the street's
component of the whole pattern, found once); its values are one pass over the pattern (`_values`):
an unknown row its conductances, a closed passage an explicit zero, a cell off the unknowns (walled
in, or ground) an identity row. Right-hand sides, potentials and the adjoint are per cell (0 off
the unknowns), the demand's products two fixed sparse matrices (cells x homes and its transpose).
One plan then serves a block's every state: exact scores, tensions, SIMP's iterations and the
polish. Not one sparse product as the brief suggested but one pass of a numba kernel over the fixed
CSR: the masking by the unknowns fuses into it, where a product would want two gathers of the
pattern's size after it. Scores equal phase 1's: on 20269 phase 1's code and this one at the same
states (the baseline's lenses and escape times, a random clearing's lenses and exact J_1 and J_2,
SIMP's value and gradient at x = D and at a random x for J_1 and J_2, the tension adding and
putting back, the catchment's gram, the polish's eps-world value) agree within 6.4e-11 of the
largest magnitude (P0 1.7e-12, the gradients 4.5e-12 -- 1.5e-11, the tensions 3.4 -- 6.2e-11);
every stored run re-scores within 7.1e-12 (the migration, below); and the `pattern` guard.

**The host's work.** Besides the pattern: the conductances and their pairs' sums in one pass
(`_conduct`), each passage's harmonic mean and the adjoint's back-propagation through the passages
in numba (`_harmonic`, `_grad_tri`), the refinement's residual and backward error in one pass over
the matrix (gap_factor._residual, where two products and |A| were), and free first's baseline
complex cached with the fixed mesh (Fixed.cx0; its triangles' geometries made again at load; the
cache's format in its key, a storage detail, not in the name): one build of the grown complex less
a run, the same held set (20269 23 -> 12 s, 23597 29 -> 16 s, 33717 42 -> 5 s, here). The breakdown
(gap_calib speed; summary_runtime_speed.csv; each twice, the mean; phase 1's timing's states: a
random clearing at D 0.05, its exact score, a SIMP iteration in the eps world at 1e-4 and q 3, the
tension), phase 1 on one core of ai01, this phase on one core of ai05 (quadro2, beside the
acceptance's 32 tasks) and on one GPU, seconds:

    block       state            phase 1 core      core now         GPU now
                                 all  fac  rest    all  fac  rest   all   fac  solves  rest
    19570       exact score      0.27 0.16 0.11    0.27 0.20 0.07   0.04  0.01  0.01   0.02  (H100)
    19570       SIMP iteration   1.52 1.22 0.30    1.86 1.65 0.21   0.20  0.01  0.01   0.18
    5810        exact score      21.7 13.8  7.9    18.7 14.5  4.2   1.86  0.10  0.84   0.93  (H100)
    5810        SIMP iteration   68.0 55.4 12.6    65.0 59.1  5.9   1.61  0.10  0.48   1.04
    5810        tension          67.0 55.2 11.8    65.1 59.2  6.0   1.53  0.10  0.38   1.05
    32841       exact score      13.1  7.7  5.4    10.9  8.1  2.7   0.85  0.05  0.30   0.50  (H100)
    32841       SIMP iteration   31.2 23.3  7.9    27.8 24.5  3.3   0.74  0.05  0.15   0.54
    7851        exact score       9.1  5.6  3.5     7.3  5.9  1.4   0.71  0.09  0.22   0.39  (Quadro)
    7851        SIMP iteration   19.4 15.2  4.2    17.7 15.9  1.9   0.81  0.09  0.12   0.61
    5810@major  baseline score   32.9 19.9 13.0    40.8 32.5  8.4   3.68  0.24  1.80   1.64  (H100)

(On a core `rest` is everything but the factorization, the solves among it; on a GPU the solves are
apart: the reach, the assembly, the factors and the gradient are `rest`.) On a core the host's own
work is cut 1.5 -- 2.5x (5810: 7.9 -> 4.2 s of an exact score, 12.6 -> 5.9 s of a SIMP iteration;
here, on a faster core, the work beside SuperLU's own 3.6 -> 1.3 and 6.1 -> 1.9 s), and the
factorization is what it was: a core's time falls 4 -- 20% on the blocks (the region's rose: ai05's
factorizations ran 4 -- 63% slower than ai01's, beside 32 other tasks). On a GPU the factorization
is 0.05 -- 0.24 s and the host's work is now the time: a SIMP iteration on 5810 takes 1.6 s against
68 s (42x), an exact score 1.9 against 21.7 s (12x), the region's baseline score 3.7 against 33 s
(9x). The cuDSS plan (the reordering on the host) is part of the model's setup: 55 s on 5810 with
its four baseline solves, 121 s on the region. What is left on a GPU, in order: the solves (each
refined on the host: the residual's pass and a copy of the right-hand side and solution each way,
0.3 -- 0.8 s on 5810), the gradient (0.4 s), the assembly (0.3 s), the reach (0.15 s in the eps
world, 0.3 s at a 0/1 state), the factors (0.15 s).

**The polish scored exactly (`E`).** A polish's scoring is a Strategy (search.PolishScoring) named
by its spec's suffix: `InEps` (no suffix: today's, its candidates scored in the eps world it ranks
them in) or `Exactly` (`E`: scored exactly, ranked alike), for relax's plans (`...p256w8x2E`),
polish_greedy (`S0.01cat.P64w8E`) and the swings. Measured on the 13 (both presets at D 0.05, free
first, P, area population, on a core and on a GPU): every clearing the same as today's (SIMP's and
the cheap preset's, on all 13 on either device), so the same scores; on a core SIMP's time a median
0.81 of today's (0.66 -- 0.91) and the cheap preset's 0.56 (0.43 -- 0.73) (an exact score factors
the open ground's unknowns, the eps world every cell); on a GPU 1.05 (1.02 -- 1.23, the 9 blocks
whose two runs had the same card) and 1.04 (0.94 -- 1.29, all 13 on one card each) of today's (on
one pattern every state factors every cell, and a 0/1 state's reach needs the search the eps
world's skips). Each is best on one device, so both stay: on a core the presets' `E` forms, on a
GPU today's.

**The eps world's interiors lumped (`+relaxlumped`): measured, dominated, deleted.** The brief's
second option, fewer unknowns in the eps world's interiors and zone strips (58% of 5810's cells:
interiors 2.30M, strips 1.62M, open ground 2.86M of 6.77M). A coarser lattice inside the footprints
could take at most the interiors' vertices off the outlines, 0.48M of their 1.61M on 5810 (the rest
the outlines' densification at s): at most 14% of the cells, and a new metric [inferred: not
built]. Built instead as one cell per building's interior in the eps world only (SIMP's, the
tension's and the eps-world polish's; the exact scores on the meshed interiors, so the same
metric), the metric's name `+relaxlumped`: 34% fewer cells in the eps world. On the 13 against its
counterpart without it (`L` against today's polish, `EL` against `E`; on a core SIMP's on 11, its
5810 and 30848 runs stopped once the acceptance's own had ended, 5810's after 2.4 h): Lens A lower
by a mean of 0.023 and 0.017 (SIMP, on a core) and 0.024 and 0.019 (on a GPU), worse on every block
measured, and 0.011 and 0.010 for the cheap preset, worse on 11 of 13 (its two gains: 22422 +0.042,
33717 +0.006); slower on a core (a median 1.84 and 1.76x for SIMP, 1.64 and 1.07x for the cheap
preset; 5810's SIMP iteration 110 s against 65: a building's interior is one node joined to every
cell on its outline, which fills the factor [inferred]) and on a GPU (1.21 and 1.41x, 1.11 and
1.17x, though an iteration on 5810 takes 1.36 s against 1.62), its host memory no less (a median
3.3 GB against 3.3 on a core) and a second cell structure and plan held beside the exact one. A
cleared building lumped to one node is a shortcut across its footprint, so the relaxation prices
interiors wrongly. It is beaten on every axis and deleted (its rows kept; these are the numbers for
the wiki's dead-end record). Its two gains are on 22422 (the gated block) and 33717 (where the
complex's own SIMP and cheap preset end behind the raster's clearing: the search's weakness): data
for that lead, not a variant.

**The run's identity** (gap_clear.CODE, code_constants). Every public module-level constant (an
UPPER_CASE name) of gap_clear, gap_complex, gap_flow and gap_factor, and clear.EPS (the tension's
eps world), is digested at import (sha256 of their reprs, 8 hex digits), and the digest ends every
complex metric's name. The constants are found by walking the modules, not listed, so a new one is
digested without anyone adding it; a constant of a kind the walk does not know raises at import,
and those that cannot change a result are exempt with their reasons (_EXEMPT: the cache's and this
file's paths, conformity's cap, which raises rather than change a mesh, the ring query's chunk, the
cuDSS logger's level, typing's flag). 28 are digested: MESH_REV, the 0.1 mm snap and the band's
tolerance, the unflat passes and P_{1/2}'s q (gap_clear); TOL, free first's step, the dust,
Delaunay, grow, snap and merge tolerances (gap_complex); the merge, T_MIN and flat tolerances, the
Steiner clearance and insertion passes, sigma, the lattice's offset, the tie tolerance, the empty
fill and the ramp's Gauss rule (gap_flow); the refinement's three and the dissection's leaf
(gap_factor); clear.EPS. CROSS_MIN did not survive phase 1 (conformity replaced its clip).
complex_of refuses a name whose digest is not this code's, so a row of other constants cannot be
read as this code's, and the mesh cache's key carries the digest. The migration: relax_rows/ and
polish_rows/<phase 1's name> renamed to the new name and each SIMP row's `along` rewritten; every
stored run of the complex's presets on the 13 (all 26, not a sample) re-scored under the new name
reproduces its Lens A within 7.1e-12 (gap_calib rescore; summary_runtime_rescore.csv). No code
reads the old name.

**The guards** (each passes on the final code and fails under its fault, both ways after the last
change).
- `device` (on a GPU node): cuDSS against SuperLU on one fixed mesh at the same states (the
  baseline's lenses and escape times, a random clearing's lenses and exact J_2, SIMP's value and
  gradient at x = D and at a random x for J_1 and J_2, the tension adding and putting back, the
  polish's eps-world value), within 1e-8 of each one's largest magnitude: 19570 within 5.0e-11,
  5810 7.9e-11 (the large), 7851 3.8e-11 (the oversized), all on ai06's H100. Fault `perturb` (the
  random clearing's refactorization handed its values with the diagonal raised by 1e-6, a leak to
  ground at every cell): that state's lenses 0.86 and its escape times 0.68 off, every other state
  within 5.2e-11.
- `pattern`: the operator on the fixed pattern against the operator assembled afresh over the
  unknowns from the passages (phase 1's assembly, kept in the guard as its oracle) at the baseline
  (253,031 of 19570's 327,921 cells and 932,252 of 20269's 1,423,143 off the unknowns: interiors,
  strips, walled-in ground), a random 0/1 clearing, a fractional state, the eps world (27 and 6
  off) and free first's state: equal on the unknowns to 1.2e-16 of the largest entry, every row off
  them an identity row, the structure the pattern's, P from the model's refined solve within
  1.5e-11 of a direct solve of the fresh operator. Fault `drop-identity` (one row off the unknowns
  given a diagonal 0): its identity rows fail at all five states (on a core the scores would not
  see it, SuperLU restricting the system to the unknowns; on a GPU it is a singular row: the guard
  reads the matrix).
- `constants`: each of the 28 constants changed alone changes the digest, and the metric's name
  carries the digest. Fault `omit` (the digest computed without MESH_REV): MESH_REV changed and the
  digest did not.
- Phase 1's (gap_clear_check baseline, weights, reach, gradient, free, identity, convex; gap_check
  conform, delaunay, apertures, filtration, free; gap_flow_check angle, shift, channel, field, law,
  closed): each passes and fails under each of its faults, with phase 1's numbers (baseline: 19570
  -5.4e-5, +1.3e-5, -7.4e-5; 20269 -1.6e-4, -2.1e-4, -3.5e-4; convex 6.7 -- 7.8e-16; the faults'
  failures as before).

**The acceptance run** (gap_calib preset; summary_runtime_presets.csv). The cheap preset
(S0.01cat.P64w8) and SIMP's default at D 0.05, free first, optimizing P, area population, on the
13, on a GPU (ai06's H100 and ai04's Quadro RTX 8000, routed by Slurm) and on one core of ai05
(quadro2), today's variants and `E`:
- Scores: every run of today's two presets and of their `E` forms, on either device, ends at phase
  1's clearing (13 of 13 each) with Lens A within 7.9e-12 of phase 1's row; no run leaves a home
  walled in.
- Time, the optimizer's own (`t`, the model built), a median over the 13: SIMP 58 s on the GPU (27
  -- 330 s) against phase 1's 1,841 s on a core (29x, 17 -- 49x) and the raster's 173 s on a GPU
  (0.38 of it, 0.13 -- 1.9x: faster on 10 of 13, slower on 20269, 30848 and 5810); the cheap preset
  39 s (12 -- 321 s) against 699 s (9.7x, 5 -- 46x) and the raster's 79 s (0.93 of it, 0.27 --
  5.1x, faster on 7 of 13: its greedy's rounds are few, and on the large blocks free first's region
  graph and the catchment's sweep, on the host, are most of it: 5810 295 s against the raster's
  79). On a core SIMP takes 0.97 (0.92 -- 1.07) of phase 1's time and the cheap preset 0.92 (0.80
  -- 1.19), their `E` forms 0.81 and 0.56 of that. Wall time, the model's setup with it (the mesh
  loaded or built on the node, the cuDSS plan, the baseline): a median 239 s for SIMP and 111 s for
  the cheap preset on the GPU where the mesh was built in the task, 83 s and 67 s where the node
  held it.
- Memory: on a core 1.9 -- 10.7 GB for either preset (phase 1's SIMP 2.2 -- 12.5); on a GPU node
  the host's 1.9 -- 11 GB for a task's first preset (up to 21 GB where the task built its mesh too,
  or still held an earlier preset's model), the device's 0.8 -- 4.6 GB (the cheap preset's, sampled
  over the run: the plan and factors of the whole pattern; 30848's 4.6 GB, 5810's 4.0).

**The region** (5810@major, its baseline on a GPU; gap_calib speed): the fixed mesh built in 1,034
s at 22.3 GB on ai06's host (phase 1 1,133 s on ai01), the model (the cuDSS plan and four solves)
in 121 s, a baseline score in 3.7 s (the factorization 0.2 s, the solves 1.8) against phase 1's 33
s on a core and 40.8 s on ai05's now, on 6.5 GB of the card: 15.4M cells, 6.06M unknowns at
baseline.

**Compute.** GPUs 6.1 GPU-h (the acceptance 4.8: SIMP's 1.2 and the other items' 3.7, every `L`
item among them; the speed runs and the region 0.6; the guards and the pilots 0.7), at most 4 cards
at once, one process each; CPUs on quadro2 54.9 CPU-h (the acceptance's 104 tasks 52.0, the
cancelled ones' partial runs among them; the speed runs 1.2; the meshes warmed on the node 1.8);
here, single-threaded, at most 8 processes: the guards twice, the calibration of the pattern
against phase 1's code, the rescore, besides the development.

**Phase 2b needs**: the remaining ports (road re-scoring, which needs roads carved into the fixed
mesh; pweight; the atlas and corridor pictures; clearings_check; the exports; the translucent
search if it is on the frontier) and the cross-checks against the raster, now cheaper than the
raster on a GPU (its presets at 0.38 -- 0.93 of the raster's time, median); on a GPU the presets'
`InEps` polish, on a core their `E` forms (the same clearings). On a GPU the host's work is the
time: the refined solves' transfers and residuals (moving the refinement onto the device, cupy
holding the pattern), the gradient, the assembly and the reach are each 0.15 -- 0.8 s on 5810, and
for the cheap preset free first's region graph and passes (a build of the grown complex a pass
where homes are walled in) and the catchment's sweep (numba, one core: GpuSweep exists for the
raster) dominate the large blocks. A mesh is built on each node's disk the first time a node meets
a block (8 min on 5810, 17 -- 29 min on the region: a CPU node could build them for a GPU run if
the cache could be shared without the home's quota). The search's weakness on the complex stays a
lead (22422's cheap preset under the lumped tension, +0.042, is a datum for it).

### The complex as the runtime: the ports and the cross-checks (owner 2026-10-14: "we can keep the raster for the cross-checks and then retire it")

BACKLOG 4, decision (3): the runtime's phase 2b (the brief:
.superpowers/sdd/2026-10-14-complex-ports): everything the raster still did alone ported to the
complex, and the cross-checks against the raster that phase 3, the retirement, records. The metric
is unchanged, `cx+shy0.5+s0.125r1b1+R0.5g4x2q4+v34d40e8a` (metrics.RUNTIME, its name now written
once): no constant moved, and the plain fixed mesh's scores are bit for bit phase 2a's (19570's
baseline, a random clearing's lenses and exact J_2, SIMP's value and gradient at q 3 for J_1 and
J_2, both tensions; a mesh built afresh equals its cache array for array). Code: gap_clear.py
(Carving and Prefix: roads cut into the fixed mesh; ComplexMetric.search_eps: the translucent
search; lattice_box), road_rescore.py (`cx`, `cxreport`), gap_maps.py (the pictures, new),
pweight.py and pweight_check.py, clearings_export.py, clearings_check.py and clearings.py,
free_roads.py and transparency_check.py's free-stage oracle, atlas/block5810/ (export, prior,
buildings, style, style_rel), gap_calib.py (the translucent search's preset items, `roads`),
crosscheck.py (the raster's side of the cross-checks, new: phase 3 deletes it), gap_clear_check.py
(roads, flux, device-roads, the identity's ported scripts), gap_complex.Geometry (local, utm),
relax.SIMP_DEFAULT and polish_greedy.CHEAP (the presets, each defined once).

**The inventory** (brief item 1). Every tracked module under research/roadless/ that, at the
phase's start (1edb6619), imported lifted, width or the raster's clear or relax, or used common's
raster names (Scorer, Carve, Obliterate, placement, carved, reached, mesh_of, GoalMesh): an AST
scan of the imports, those inside functions included, 70 modules (and gap_complex, which names them
in text only). Each is (a) needed by the project, ported here or on the complex already and
reaching the raster only through a mixed module's shared part, (b) the raster's own, retired in
phase 3, or (c) a finished study's record, kept or retired in phase 3 with its numbers.

(a), 21. Ported in this phase: road_rescore (`cx`, `cxreport`; its raster `score` and the rest b),
pweight and pweight_check, clearings_check (with clearings and clearings_export, which the scan
does not list: the store keyed by the metric's name), atlas/block5810/export and prior (with
buildings, style and style_rel), free_roads (with transparency_check's free-stage oracle and
transparency_run's FREE), gap_calib (the translucent search's items and `roads`; its `rasterscore`
b). On the complex already, reaching the raster through a shared part: gap_clear (clear.EPS, which
the code's digest reads, and CpuSweep; relax.project and dproject; check_disjoint's parse of the
raster's names, lifted.conductance_of and along_of), gap_compare (ComplexModel, the guards'
rebuild; its raster comparisons, RasterIso, `iso`, `lanes`, `gates`, are c), gap_study (spec_of,
which reads lifted.RING_M and BAND_M; the rest c), gap_check (spec_of, the same), gap_flow_check
(the walled channel's geometry from width: L_C, WALL_M, ANGLES), seam (clear.Sweep's type), metrics
(the RasterMetric branch), relax (simp, Plan, plan_of, one, frank_wolfe, knapsack, project,
rows_of: the optimizers), search, polish_greedy (a lifted.MeshSpec the complex ignores), cluster
(common.mesh_of for `--mesh`'s sizing), clear (EPS, RTOL_SCORE, Catchment, CpuSweep, Sweep,
GramSource, rows_dir, cleared_through, sweep_of) and common (the blocks, LINEUP, arms, the
populations, Demand, PopulationDemand's and PriorityDemand's `injection`, displacement,
canonical_order, prefix_to). New: gap_maps (a), crosscheck (b).

(b), 29, each with what it measured and where the numbers are. lifted (the lifted operator: the
heading lift and its turning cost, Uniform, Sightline, SoftSightline, TurnBoosted; the cells'
conductances OpenFraction, B_r and D_s; the face aperture Fws; Placement; the uniform, adaptive,
goal and composite grids; the AMG solvers on both devices: NOTES from "The model" to "The grid
fix"); width and width_check (B_r and D_s and their guards: no new corridors at r up to 1.5 m, U's
order kept, tau 1.00; width_rows/summary_*.csv; "Uniform conductance with an effective width");
aperture_check (the face aperture's guards: the levels' lattice drift 0.23 -> 0.027, the angle
error at 1 m 1.66 -> 1.21; aperture_rows/summary_*.csv; "The grid fix"); checks, angle_h, angle_q
(the lift's channel: P against angle, h and ell; "Results so far"); sightline_angle,
sightline_checks, line_scale, transparency (the sightline conductance: its angle error, 5.32 at 1 m
against U's 1.66, its width response, its length scale r0, and the translucent search's depth under
it; "Road-favouring conductances", "Length response and its scale", "Translucent search", "The
sightline metric's angle error"); conductance_map, current_map, opened_map, render_clear,
render_alongs, render_simp and atlas/block5810/export_ss (the raster's pictures, the PNGs not
committed: "Road-favouring conductances", "GPU-resident rounds", "Visualization: block 5810";
gap_maps replaces them); composite_proto (the composite mesh's prototype: "Local coarsening: the
prototype"); offset_check, resolution_check, rescore_grid (the lattice's offsets, its resolution
and the coarse grids: res_rows/; "The metric's resolution floor", "Where resolution matters", "What
a coarser grid does to Lens A", "Speed"); gramprobe, memprobe_greedy, profile_round, releaseprobe,
solver_bench, timing, conv (the raster's GPU, AMG and convergence probes: conv/ and conv2/;
"Results so far", "Greedy speed", "Translucent greedy on 30796"); and the mixed modules' raster
parts: clear's Clearing, Tension, free_first, RegionGraph, greedy_block, run, loo, exact and
GpuSweep; relax's Relaxation, RasterMetric, _clearing and the coarse plans `.c` and `.C`; common's
Scorer, Obliterate, Carve's raster rule, placement, carved, reached, GoalMesh, goal_indicator,
mesh_of and the Demands' `baseline`; road_rescore's `score`, Corridors, GpuWatch, `report`, `sizes`
and `project` (road_rows/summary_h0.5_*.csv the record); metrics' RasterMetric branch; cluster's
`--mesh`, `--scans`, GB_PER_MILLION, HELD_PER_MILLION, RASTER_BYTES and `_gb`; gap_calib's
`rasterscore`; clearings_check.metric_of_run's raster branch.

(c), 20. study (the 220's lineup under the lifted metric, rows_h0.5_ell3_K8*/: "Study: the lineup
on the 220"), tail and tails (the escape-time tails: tails_M4_area_p*.parquet), rescore (the
greedy's discretization: rescore.parquet), compare_clear, area_match, along_lineup (the first
clearings against roads: "Road-free clearing", "Road-favouring conductances"), picker_compare and
simp_compare (paired comparisons of stored rows: "Scaling", "Lens A at D 0.05"), mono_cross
(monotonicity over nested road sets), transparency_run and transparency_check (BACKLOG 5's
experiment and its guards: transparency_rows/summary_*.csv, "Transparency for the road methods";
their free stage now the complex's), gap_simp (the complex's study: gap_rows/summary_simp*.csv),
fft_homog's extract, extract2, extract_gated, lifted_ref, region_extract and region_score_gpu (the
FFT homogenization study's raster fabric and fine reference: fft_homog/*.csv, report.md; "FFT
homogenization", "... on 5810@major"), atlas/region5810/export (the region page on the composite
raster: re-exported on the complex with the region test).

**Roads on the fixed mesh** (gap_clear.Carving, Prefix; brief item 2). A road prefix frees exactly
its corridor (common.Carve's rule: every road buffered by half its width,
reblock.budget.road_corridor), so the footprints it touches stand less the corridor, and under the
shy distance the zero zone is the w / 2 offset of what stands. A network's scored prefixes are its
stops (the area prefixes at D 0.02 -- 0.20 and the whole network: road_rescore's prefix rows),
their corridors nested. Every footprint the last corridor touches is cut into atoms, the piece
within each stop's corridor less the earlier one's (carved from that stop on) and the piece outside
every corridor (never carved), and the fixed mesh is built on the atoms in place of the buildings
(full_mesh unchanged: the atoms' outlines and their w / 2 offsets noded together). A state at stop
k stands every building's atoms less those carved by k: the footprint less C_k, which is what the
rebuild stands (gap_complex.obstacles), and since a zone is the union of its standing atoms' zones
(dilation distributes over union) the shy distance's factor is exact on the atoms. The homes stay
the buildings (the demand, the rings, the population, B (i) and free first's complex are built on
them), a clearing's k is per building and an atom takes its building's k or 1 where the prefix
carves it (Carving.atom_k; the gradient sums the atoms to their buildings, the carved held at 1).
With no network every building is one atom and the plain mesh is phase 2a's bit for bit (above).
Each corridor is unioned afresh, as the rebuild and the raster carve it, so two stops' corridors
part by slivers: their nesting is held to the linework's snap along the shorter one's outline
(18895's clearance_looped: 3.4e-6 m^2), and a carved mesh's Steiner lattice is laid over the
buildings' baseline open ground (gap_clear.lattice_box: the same ground; GEOS's union of the atoms,
pieces sharing their cuts, threw a TopologyException on 38988's cycle_native_betweenness_contrast
network). A carved mesh is cached under its corridors' digest beside the block's identity;
road_rescore builds each in a temporary directory, since it serves one network.

One mesh a network or one a block. Both are the same Carving (several networks' corridors cut
together, each atom carved per network): on 20269 one network's mesh builds in 57 s (2.04M
triangles) and all six networks' in 152 s (4.60M), every state then 1.3 -- 3.4 s on a core against
0.8 -- 1.7 s, and the lenses agree within 6.5e-5. The run used one mesh a network (the brief's
suggestion: a block's networks spread over the nodes' cores); both are one code path.

**Calibration by equality** (gap_clear_check roads; gap_calib roads, gap_rows/runtime/roads/).
Every prefix of two networks (resistance_lp, the lineup's leader, and cycle_native) on six of the
60 (6312, 17608, 18895, 18910, 18985, 20080: 68 prefixes, D 0.02 to the whole network), as the
60-block run scored it on its carved mesh, against the complex rebuilt with the prefix carved
(gap_compare.ComplexModel, the guards' rebuild: gap_flow's mesh with the corridor cut from the
footprints). The homes walled in are the rebuild's at every prefix. On four of the six L_P is
within 6.2e-4 (median 2.6e-4; L_{P_1/2} within 6.6e-4, L_J2 6.4e-4) and the state's P within 1.1e-3
of the rebuild's, the difference growing with the carving (18895's resistance_lp: P 2.5e-4 low at
38 roads, 1.1e-3 at 1,057; L_P 1.2e-4 high, then 6.2e-4), the carved mesh reading a little faster
out: the fixed mesh triangulates the same ground differently, its faces sharing the atoms' cuts, as
at phase 1's clearings, and far below the scheme's own error. The other two carry phase 1's necks.
On 6312 every network's baseline P0 reads 3.3e-3 below the rebuild's (the homes most apart 4% fast,
as on 30848), and once a prefix's roads open that ground (resistance_lp from 166 roads, D 0.05;
cycle_native from 35, D 0.10) the states agree within 1.1e-3, so the lens carries the baseline's
difference: L_P up to 2.3e-3 below the rebuild's, L_{P_1/2} 1.2e-2, L_J2 7.8e-3. On 18985 the
baseline reads 1.0 -- 1.7e-3 high (not traced) and L_P within 1.1e-3. At area D 0.05, the lineup's
budget, L_P is within 2.2e-4 on the four and 2.3e-3 on 6312. The `roads` guard holds a prefix to
phase 1's tolerance for a clearing (2e-3, every lens): resistance_lp at stops 1, 3 and 5 on 20269
(36, 201 and 751 roads: L_P +2.9e-5, +2.3e-4, +4.2e-4) and 18895 (38, 260, 1,057: +1.2e-4, +3.2e-4,
+5.8e-4), the homes walled in equal.

**The lineup on the complex** (road_rescore `cx` and `cxreport`;
road_rows/<metric>/summary_cpu_area_<budgets>_{lens_D0.05,lens_D0.1,price,clearings}.csv,
committed). The six networks of item 4's lineup on the 60 large blocks (the sentinel's four, 22422,
30848, 5810 and 9712, among them), every area prefix a state on the network's carved mesh, scored
under P, P_{1/2} and J_2 with B (i) as item 4's were (the same prefixes, the same networks, the
same order), on the cluster's CPU nodes and here: resistance_lp leads the medians at both budgets
under all three lenses (L_P 0.194 at D 0.05, 0.273 at D 0.10), then cycle_native (0.163, 0.250) and
cycle_native_betweenness_contrast (0.162, 0.246), clearance_looped (0.140, 0.212), euclidean_grid
(0.124, 0.216) and clearance (0.116, 0.196); in B's order it leads on 30 of the 60 at D 0.05 under
L_P (28 and 29 under L_{P_1/2} and L_J2) and on 27 at D 0.10, clearance_looped and cycle_native on
9 each. Item 4's raster read the same medians within 0.007 (resistance_lp 0.194, clearance 0.116;
clearance_looped 0.147, cycle_native 0.169), but led with resistance_lp on 46: the order's
cross-check, below, finds why. A network's mesh: a median 2.5M triangles (1.3 -- 9.1M), built in
207 s (84 -- 772 s), then its model in 14 s and each prefix's state in 5.4 s (at most 20), on one
core and at most 13.1 GB (5810's).

**The price of roads on the complex**: SIMP's default and the cheap preset under P at area D 0.05
(B's optimizer: free first, then the search; the polish scored exactly on a core), each against the
best road prefix in B's order at the same D: SIMP's default is ahead of the best road on all 60
under every lens. On the 29 where B (i) ties it leads by a median +0.123 of L_P (+0.047 -- +0.217;
L_{P_1/2} +0.124, L_J2 +0.123), the best road reaching a median 0.49 of SIMP's L_P (p10 0.42, p90
0.65); on the other 31 it leads on B (i): the complex walls homes in on 39 of the 60 at baseline
(234 homes, a median 0.1% of a block's population), free first frees every one on all 39, and each
network's D 0.05 prefix frees them all on 1 -- 4 of the 39. The cheap preset is ahead on all 60
too, by +0.109 where (i) ties. On the raster (item 4) SIMP's default led on all 60 by +0.152 where
(i) tied, on 55: the price of roads stands, a little lower in L_P on the complex, and the complex's
enclosure decides more blocks.

**pweight on the complex** (pweight.py, pweight_check.py; brief item 3). P_{1/2}'s weights were
already the complex's own (gap_clear.Model: common.PriorityDemand(0.5) on the baseline's escape
times under the population's demand, Model.inj_q); what the raster still did alone was the study's
machinery. Its SIMP runs and bounds now run on the runtime's metric (relax.one and
relax.frank_wolfe through the seam, the metric pweight.METRIC, rows pweight_rows/<metric>/), and
its scoring of every clearing under every objective is the complex's model (one state a clearing,
one solve a demand; the depth of the cleared buildings from the catchment's shares of the complex's
tension). The study's own rows (pweight_rows/<simp|fw|score>/ under uni, and
pweight_rows/uni+apwfs/) stay its record. The guards, ported (each passes and fails under its
fault, 19570 and 20269): `identity` (q 0 is P bit for bit: injection, P0, escape times, a random
clearing's P and J_2; fault `no-population`: P0 22,410 against 22,287); `total` (the injection's
total and the demand points' the population's at q 1/2, 1 and 2, within 2.2e-16; on the complex the
right-hand side drops the points escaping at once in the band, so the points are the field's
analogue; fault `brief`: 8 -- 94% off at q 1/2 and 2); `displacement` (cost, population and B (i)
independent of the demand, bit for bit; fault `cost-from-demand`); `gradient` (P_q's and P's
relaxation gradient against the closest of the central differences at steps 1e-3, 1e-4 and 1e-5:
within 2.7e-6 of the largest, against a tolerance of 1e-4, gap_clear_check's; the raster's guard
took one step, 1e-4, with solves to 1e-9, and on the complex P is a direct solve whose roundoff on
20269's P of 9e4 reached 1.2e-4 of the gradient at a step of 1e-5; fault `J-population`: 0.22 --
0.37 on P_q, P's own rows untouched); and the brief's new `weights`: every home's weight against a
direct computation from the baseline's escape times, solved afresh (the baseline's operator
assembled from the passages, phase 1's oracle, and a direct solve), within 1.2e-11 (305 and 1,045
homes); fault `swap-time` (the most populous home's weight from the least populous one's time):
0.41 and 0.65 off.

**The corridor pictures and the atlas** (gap_maps.py, atlas/block5810/; brief item 4). The flow
under U on the complex, drawn from the triangles' fluxes: each static passage carries F = T kp (u_a
- u_b), each street face G u, a ground cell drains its inflow, and each cell's mean current density
is its boundary fluxes' first moment over its area (the divergence theorem with the normal flux
constant along each passage, less the injections at their demand points); the triangles are painted
onto pixels as the mean of their sub-samples (4 x 4 a pixel), so a lane narrower than a pixel keeps
its share. Kirchhoff's law holds at every open cell (the `flux` guard: 2.3e-11 -- 5.0e-11 of the
largest current through a cell on 19570 and 20269 at the baseline and two random clearings, the
drainage equal to the injection within 1.8e-11; fault `flip-one`, the largest passage flux's sign
flipped: 1.0). gap_maps `map` draws, for each arm (a stored run, or a network's area prefix on its
carved mesh), the flow before and after on the atlas's range (the before map's median to its 99.5th
percentile, log), its change (the corridors opened, the flow drawn away) and the change in the
escape time from each place, so it answers current_map's, opened_map's and render_clear's questions
on the metric that scores; conductance_map's road-favouring fields (sightline, width) and the
sightline arm are the raster's. The block atlas's export, prior and buildings run on the complex
(the complex's own presets on 5810, their store runs; the no-buildings prior every footprint
cleared on the fixed mesh), the page's layers unchanged in kind (one pixel per 0.5 m, the crossing
current |j| h): on 5810, SIMP's default 292 buildings at L_P 0.256 and the cheap preset 238 at
0.238, 135 in both; the median open pixel's escape time 12% (SIMP's default) and 10% (the cheap
preset) lower. The layers were drawn and looked at (flow before, the change after SIMP's default:
the lanes toward the eastern street and the cleared buildings), not published: the site is not this
phase's.

**The stored clearings, keyed by their metric** (clearings_export.py, clearings_check.py,
clearings.py; brief item 5). The store holds a run under its rows' directory, which names its
metric: the export now reads the raster's conductances by name (uni+apwfs; a bare
along-conductance, the step rule's before 2026-10-13) and the complex's (gap_clear.complex_of:
refused unless this code's; the translucent search's `@e<eps>` its search part), a complex run's
grid none; rows a cluster run shipped under another code's complex name (phase 1's runs shipped to
the raster's re-scoring, rcx-score*, 46 rows) are counted and not read, the migration having
carried them; the study stores since 2026-10-13 (aperture_rows, width_rows, gap_rows) and
cross_rows are excluded with their reasons. The export added SIMP's default and the cheap preset
under P on the 13, under uni+apwfs and under the complex, and the complex's two presets as the road
lineup's run made them on a core, their polish scored exactly (relax_rows/<metric>/<SIMP's
plan>E/D0.05.p1 and polish_rows/<metric>/S0.01cat.P64w8E/D0.05.p1, 61 blocks: the 60 and 19570):
157 runs, 61,525 rows on 303 blocks. clearings_check `rescore` scores through the seam: a run under
its own metric (`own`, metrics.metric_of from its name: the complex's model for a complex run, the
raster's for a conductance, a bare along as its +apwfs form; its demand from the run's, so the
priority-weighted runs too), or every run under a named metric, keyed by that name (the complex's
reading of the raster's runs). The complex's SIMP default on 20269 re-scores within 2.2e-12; the
raster's J_2 default there reads 0.4152 on the complex's L_J2 against its own 0.3924.

**Scripts that write rows without naming their metric** (brief item 6; 966bb8b0's note). The ported
ones are keyed by it: road_rescore `cx` (road_rows/<metric>/<device>/...), pweight
(pweight_rows/<metric>/), the presets' rows (relax_rows/<metric>/, polish_rows/<metric>/), the
atlas's info.json (`metric`); the identity guard holds them to it. Nothing will run the rest again,
each a finished study's record (c): study.py (the 220's lineup under the lifted metric,
rows_h0.5_ell3_K8*/, which it resumes into: on the complex the lineup is road_rescore `cx`'s,
keyed), rescore.py (the greedy's discretization, rescore.parquet), tails.py (the escape-time
tails), the FFT exports (fft_homog/*.csv: FFT homogenization on the raster's fabric),
atlas/region5810's export (the region page on the composite raster: re-exported, on the complex,
with the region test). road_rescore's raster `score` keys its rows by its conductance (b).

**The translucent search** (brief item 7). Its standing: measured only under the sightline metric,
where the greedy's tension ranked under a more translucent conductance (ss100k0.5) beat the plain
greedy (+0.014 [+0.002, +0.022] at D 0.05 on the 45 held out, at the same cost) and stayed below
SIMP + incumbent (+0.005 to SIMP), with a gate tail (NOTES, "Translucent search", "Sightline on the
held-out blocks at D 0.05"); whether it carried to the uniform metric was untried ("uni has no
kappa to soften"), and nothing in BACKLOG settled it. So it was ported: the complex's analogue is
the tension in an eps world with a larger eps (gap_clear.ComplexMetric.search_eps, the metric's
name's `@e<eps>`, the raster's `@<search>` on the complex: the greedy's and the polish's ranking in
that world, every score under the metric; Model.tension_eps). A pilot on a core (20269, 18895,
23597, 22422; the cheap preset's greedy and polish, gap_calib's items `cheapE@e<eps>`): eps 0.05
ahead of the plain greedy (eps clear.EPS, 0.01) on all four, 0.2 and 0.5 behind on all four
(everything reads open). Then the 13 on a GPU at eps 0.02, 0.05 and 0.1 (gap_calib presets
`cheap@e<eps>`, run cx-tl; gap_rows/runtime/preset/gpu/; P, area D 0.05, free first), paired with
phase 2a's GPU rows of the plain cheap preset and SIMP's default, Bayesian bootstrap over the 13:

    eps  greedy vs plain greedy        polished vs cheap preset      polished vs SIMP's default
    0.02 +0.0028 [-0.0033, +0.0071]  9 +0.0030 [-0.0015, +0.0066] 11 -0.0128 [-0.0194, -0.0067]  1
    0.05 +0.0121 [+0.0045, +0.0260] 11 +0.0099 [+0.0046, +0.0192] 11 -0.0058 [-0.0089, -0.0018]  2
    0.1  +0.0030 [-0.0099, +0.0190]  7 +0.0061 [-0.0033, +0.0159]  9 -0.0097 [-0.0145, -0.0058]  0

(mean [95% interval], blocks ahead of 13). At eps 0.05 the translucent greedy's medians are +0.006
over the plain one, its worst -0.006 (46841) and its best +0.077 (22422: 0.187 against 0.109, the
gate the plain greedy misses and SIMP finds); polished it reaches SIMP's default on 30848 (0.610
against 0.598) and 22422 (0.188 against 0.187). Its time is the cheap preset's (a median 1.03x,
0.65 -- 1.85x on cards Slurm routed; the greedy's own 1.02x), SIMP's default 1.22x it on the GPU.
No run leaves a home walled in. **Verdict: on the frontier** at the cheap preset's cost, and the
plain greedy is not dominated by the rule's letter (it ties on time and wins 2 of 13), so both stay
selectable; making eps 0.05 the cheap preset's search is the owner's call. SIMP's default stays
ahead of it. eps 0.02 and 0.1 are within noise of the plain search and behind 0.05 (too little
translucency, then too much).

**The road methods' free-first stage on the complex** (free_roads.FreeEnclosed; BACKLOG 5's TODO
for the consolidated pass). The stage frees the homes enclosed at baseline with roads on the
method's own graph; its enclosure, its targets and its test were the raster's (lifted.Placement,
the cells near a home, the scorer's reach). Now: the homes the metric walls in (the complex grown
as the width form grows the footprints, read at its filtration width: free first's test), the
targets the ground within ring_m of the home's outline and the sealed pockets of the complex its
ring touches, the test the complex rebuilt with the held roads' corridor carved; `metric` (the
complex's name) in place of the raster's `mesh`, in its identity. transparency_check's free-stage
oracle is the flow model's reach on the fixed mesh with the roads carved
(gap_clear.Model.road_factors on a Carving of the held roads), independent of the stage's own test;
transparency_run's FREE is the complex's stage. The guard (`free-roads`, 20269 and 18895, one
walled-in home each on the complex): every plan's roads the same on the four method graphs, each
held road frees its home by the oracle, nothing left enclosed, the true displacement the plan's (D
0.0004 and 0.0017 by area); fault `one-short` (every held road one segment short): the stage
raises, its road not freeing its home.

**Choices the brief left open** (each taken as the raster tool behaves, so the cross-checks compare
like with like). The carving is common.Carve's (every road buffered by half its width, the corridor
unioned afresh at each prefix) and B (i) the enclosed population share, as item 4 scored them. The
free stage's targets are the ground within ring_m of a walled-in home's outline and the sealed
pockets its ring touches, the complex's analogue of Placement's ring of free cells within 1 m. The
atlas's no-buildings prior clears every footprint on the fixed mesh, as the raster's carved every
one. The pictures paint the complex's current density onto the raster's own cells (the mean over 4
x 4 sub-samples a cell, so a lane narrower than a cell keeps its share) and compare on them, the
raster's flow its cells' summed heading fluxes as current_map drew it. The road lineup ran on
cores, not GPUs: the carved meshes' builds are 83% of a task (GEOS and the triangulation, host work
a card cannot speed; a median 207 s, 772 s at most on 9.1M triangles), so on cards the 360 tasks
would have held about 30 GPU-h, past the phase's 15; the device changes a score at the
floating-point level only, checked on carved meshes by the `device-roads` guard (below). The
presets' polish on a core scores exactly (`E`), as phase 2a chose for that device (the same
clearings as the eps world's).

**The cross-checks against the raster** (crosscheck.py;
cross_rows/summary_{roads,clearings,weights,pictures}.csv, committed). The raster is uni+apwfs at h
0.5 (ell 3 m, K 8, the area population, B's placement), its metric as phase 1's acceptance scored
it, and item 4's uni (the step rule before the face aperture) beside it where its rows stand. On
the 60 it scored what the complex scored (runs cx-raster-p, cx-raster and cx-raster2 on the
cluster's cards): each network's area prefixes at D 0.05 and 0.10, the stored clearings and the
complex's presets, every home's P_{1/2} weight and escape time at baseline. On four blocks its flow
stood against the complex's. Each difference is read as the raster's error (its lattice, its lift,
its leaks: the gap complex's study measured each) or the metric's choice (the shy distance at w 0.5
m: a gap narrower than w closed, a zone of w / 2 along every wall that conducts nothing), with the
evidence for which.

- The road methods' order. On the 60 at D 0.05 the complex orders the six networks as the raster
  does at a Kendall tau median of +0.83 under L_P (quartiles +0.33, +1.00; +0.83 under L_{P_1/2},
  +0.87 under L_J2), the same leader on 38 (40, 39); against item 4's uni +0.73, the same leader on
  34. The lens alone orders them alike (+0.87 [+0.78, +1.00]); the order parts where B (i) does. On
  the 26 blocks where the methods' enclosed shares tie on both models the order is the raster's
  (tau +1.00, the same leader on 25); on the 34 where they do not it is +0.60 [+0.20, +0.70]. That
  is enclosure: the complex walls homes in on 39 of the 60 at baseline, the raster on 23 (item 4's
  uni on 12), and each of the 174 homes walled in on one model only (37 blocks) is walled in on the
  complex, behind a gap narrower than w that the raster's cells pass. So resistance_lp, which leads
  on 40 on the raster (46 under item 4's uni), leads on 30 on the complex: in B's order a network
  whose D 0.05 prefix frees more of those homes leads, whatever its lens. Verdict: the metric's
  choice (the shy distance at 0.5 m walls in homes behind gaps a body does not pass, and B (i)
  ranks freeing them first), not the raster's error; at D 0.10 the same (L_P +0.77, +1.00 where (i)
  ties, +0.47 where not).
- The stored clearings on both metrics. The stored clearings on the 60 (SIMP's default and the
  cheap preset as the raster's uni made them under J_2, on 59, and as uni+apwfs and the complex
  made them under P, on the 13) and the complex's presets as the lineup's run made them, each
  scored on both. Each metric ranks its own optimizer's clearings first, as phase 1 found on the
  13: on the raster the complex's SIMP is behind the raster's own (uni+apwfs) on all 13 (where (i)
  ties, a median -0.010 of L_P; -0.53 on 22422, the gate) and its cheap preset ahead on 2; on the
  complex the complex's SIMP leads the raster's on 9 of 13 (5 of them on B (i), the raster's
  clearing leaving homes walled in on the complex; +0.0004 where (i) ties) and its cheap preset on
  12. Against the raster's J_2 clearings (uni, 59 blocks; their objective J_2, the presets' P) the
  complex's presets lead on the complex on 58 under L_P and 43 -- 46 under L_J2, and on the raster
  on 31 -- 32 under L_P (a median -0.001 where (i) ties) and 12 -- 13 under L_J2. The clearings'
  order, metric against metric, is a Kendall tau median +0.38 on L_P (+0.08 on L_{P_1/2}, +0.33 on
  L_J2). Levels over the 60 agree within 0.015 at the median (the complex's SIMP 0.358 on the
  complex and 0.361 on the raster, the raster's J_2 SIMP 0.355 and 0.370); on the 13 the raster's P
  clearings read 0.10 lower on the complex (0.275 against 0.378), the gates they open through lanes
  under 0.5 m (phase 1's 22422 and 23597). The raster's clearings leave homes walled in on the
  complex on 25 -- 29 of 59 (uni) and 5 of 13 (uni+apwfs); the complex's leave one walled in on the
  raster on one block (24240, 0.015% of its population; not traced). Verdict: the levels agree;
  each metric's search fits the metric it scores, so the clearings' order parts by optimizer, and
  the enclosure gap and the gates are the metric's choice (w 0.5).
- P_{1/2}'s weights. Every home's P_{1/2} weight at baseline on the 60: Spearman per block a median
  +0.99 [+0.97, +1.00] (at least +0.945, 41515) over the homes with demand on both (a weight is the
  home's escape time to the q, so the escape times rank alike); the complex's weight over the
  raster's at the median home 1.007 (p05 -- p95 0.89 -- 1.06, medians over the blocks); the 174
  homes walled in on the complex only carry no weight there; three homes weigh 0 on one model only
  (their demand escaping at once in the street band). Verdict: the weights agree; where they part
  it is enclosure, the metric's choice.
- The corridor pictures. On 20269, 22422, 23597 (on a core) and 5810 (on a card), the raster's flow
  and the complex's on the raster's cells, at the baseline and after SIMP's default as each metric
  made it (summary_pictures.csv). Where the flow goes agrees: Spearman 0.90 -- 0.995 over the cells
  both hold open (5810 0.95, 20269 0.90 -- 0.94), and the raster's main lanes (its top tenth of the
  cells) lie on ground the complex keeps open but for 0.1 -- 1.0% of their cells; the two models'
  main lanes share 55 -- 81% of their cells (75 -- 91% of the complex's lie in the raster's), the
  top tenth's edge falling differently along wide lanes, where the raster's lattice loses 0.14 --
  0.29 m of a lane at each wall (phase 1's corridor). The raster's flow on cells the complex holds
  dry or nearly so (under 1% of its median) is 0.7 -- 2.6% of it, of four kinds: on cells whose
  centres lie inside a standing footprint, 0.2 -- 1.5% (the raster's mixed cells, a cell's open
  share conducting at its centre: the raster's lattice, not a difference in where people walk;
  inferred from the kind, the cells not traced one by one); in the shy distance's zone, within w /
  2 of a wall, 0.07 -- 0.8% (the metric's choice); in pockets behind a gap narrower than w, 0 --
  0.4%, their bottlenecks a median 0.30 -- 0.44 m (the metric's choice); and in the street band,
  where the complex's demand escapes at once, 0.14 -- 0.43% (the two models' exits). 22422 is the
  brief's case: at the baseline, and after the complex's SIMP, which leaves the gate shut, the
  raster's busiest cell anywhere lies on ground the complex holds dry behind a bottleneck of 0.2 --
  0.5 m (the pockets' there a median 0.38 m: phase 1's 14 m lane, 0.25 -- 0.4 m wide, through which
  three homes hold 63% of the raster's P), so the raster's P reads 2.6x the complex's (219k against
  86k); after the raster's SIMP opens the gate that lane carries 0.02% of the flow. On 23597 such
  pockets hold 0.08% of the raster's flow at the baseline (bottlenecks a median 0.30 m), on 20269
  0.04%, on 5810 0.02%. Verdict: the corridors agree; where the raster's flow runs where the
  complex's does not, it is the shy distance closing gaps under 0.5 m and the zone along each wall
  (the metric's choice, the owner's at w 0.5), or the raster's lattice at footprint edges (its
  error), and no lane of the raster's main flow is lost otherwise.

**The guards** (each passes on the code as it is and fails under its fault; run both ways at
a1a0d47b, after the last change to any code a guard reads, gap_calib's report and crosscheck's road
order changing after it: on a core here, the device guards on the cluster's cards). New in this
phase: gap_clear_check `roads` (above; fault `one-face-closed`, the largest atom each prefix carves
left standing: L_P 5.2 -- 6.3e-3 low at the first prefix), `flux` (the pictures' fluxes conserve
current: every open cell within 5.0e-11 of the largest current through a cell, the drainage the
injection within 1.8e-11; fault `flip-one`, the largest passage flux's sign flipped: 1.0),
`device-roads` (cuDSS against SuperLU on a carved mesh at the `roads` prefixes, 20269 and 18895:
within 8.9e-11; fault `perturb`, the first prefix's refactorization's diagonal raised by 1e-6: 0.43
-- 0.88), and `identity` extended (the ported scripts write under the runtime's name and never a
raster's: road_rescore `cx`, the presets' rows, pweight; a stored run under a raster name reads as
the raster's metric, under the complex's as the complex's; fault `collide`,
gap_clear.check_disjoint disabled: a complex metric named as a raster conductance is accepted and
road_rescore would write into the raster's directory); pweight_check's five on the complex (above,
`weights` new; fault `swap-time`); transparency_check's `free-roads` on the complex (above; fault
`one-short`). Phase 1's and 2a's, re-run: gap_clear_check baseline (faults interior, exits),
weights (no-conform), reach (threshold), gradient (tie), free (drop-crossing), convex (multiply),
pattern (drop-identity), constants (omit), device (perturb; on cards); gap_check conform
(unconstrained), delaunay (flip, geos), apertures (corners), filtration (narrowest), free
(no-rebuild); gap_flow_check angle, shift, channel, field (centroids each), law (no-grow), closed
(no-clearance). On the core all 28 pass and each fails under each of its 30 faults; on the cards
`device` passes on 19570 and 5810 within 8.0e-11 and fails under `perturb` (0.68 -- 0.99 off),
`device-roads` as above (a third block named for `device`, 7851, was misspelt in the submit and did
not run).

**Compute.** GPU: 5.9 GPU-h of the phase's 15: the translucent search 1.75, the raster's side of
the cross-checks 3.24 over the 60, the device guards 0.80, the pictures on 5810 0.11. The raster
run's cost was first projected from the pilot's two blocks at 44 -- 90 GPU-h (6310 in 138 s, 30848
in 2,090 s: a power law through what proved an outlier), so the 24 largest blocks were held back;
the other 34 took 25 -- 745 s each (0.87 GPU-h), a refit on them put the 24 at about 1 GPU-h (5 if
several ran like 30848), and they ran in 1.74. CPU: 64 CPU-h on quadro2 (the 360 road networks 30,
the presets on the 60 33, the pilot 1), 8 of them lost when the cluster home's shared quota filled
(04:08 -- 04:19 UTC on 2026-10-15, every running task killed; the failed runs re-made at new
commits, here and on quadro2), and about 20 here (the guards twice, the calibration, the pictures
on three blocks, 13 blocks' presets and 5810's networks while the cluster was held; inferred from
the timed runs).

**Phase 3's list** (from the inventory). First the moves, so that the deletions break nothing the
project keeps: (1) clear's shared part (EPS, which the code's digest reads; RTOL_SCORE; Catchment,
CpuSweep, Sweep and GramSource; rows_dir, cleared_through, sweep_of less GpuSweep) and relax's SIMP
(Plan, plan_of, simp, the steppers, frank_wolfe, knapsack, project and dproject, round_by_x,
cut_to_budget, one, rows_of, D_LENS) into modules of their own, or kept with the raster parts
deleted around them; (2) common's raster names deleted and the Demands' `baseline` methods with
them (the complex calls `injection` only); (3) gap_clear.check_disjoint's parse of the raster's
names (lifted.conductance_of and along_of) replaced by a fixed grammar of them
(clearings_export.ALONG's), since the raster's rows stay on disk and a complex name must never read
as one; (4) the complex's Spec read where it is named (gap_study.spec_of and gap_check read
lifted.RING_M and BAND_M; the runtime parses ring and band from the metric's name), the walled
channel's geometry for gap_flow_check (width.walled_channel, L_C, WALL_M, ANGLES) moved beside it,
polish_greedy's and road_rescore's `mesh` argument (a lifted.UniformMesh the complex ignores)
dropped from the complex's path, metrics' RasterMetric branch and seam's raster mentions, cluster's
raster sizing (`--mesh`, `--scans`, GB_PER_MILLION and its kin), clearings_check.metric_of_run's
raster branch (the raster's rows then stand as records, not rescorable). Then the deletions: the
(b) modules and the mixed modules' raster parts as listed above, with gap_calib's `rasterscore` and
crosscheck.py after their numbers are in NOTES. The (c) modules' fate is the owner's (kept as
records, or deleted with their numbers in NOTES and their summaries committed). For the wiki's
dead-end record, each retired variant with its numbers: the lifted metric (the heading lift and
turning length: its boundary layer loses 0.3 -- 1.1 m of a lane at each wall, the gap complex's
study), the raster's lattice (0.14 -- 0.29 m of a lane at each wall; its isotropic model),
Sightline, SoftSightline and TurnBoosted (angle error 5.32 at 1 m; parked 2026-10-13), B_r and D_s
(no new corridors at r up to 1.5 m, U's order kept, tau 1.00), A, Aw and Aws (deleted with Fws's
adoption), the composite and goal meshes (local coarsening, the o1 goal mesh), the raster's speed
(phase 2a: the complex's presets at 0.38 and 0.93 of its GPU time), the cross-checks above, and
+relaxlumped (phase 2a).

### The complex as the runtime: the raster retired (owner 2026-10-15: "Delete the finished studies' code"; "Make translucent the cheap preset and delete the plain version")

BACKLOG 4, decision (3): phase 3 (the brief: .superpowers/sdd/2026-10-15-retire-raster). The
raster, its lift and turning cost, and the finished studies' code are deleted; the complex is the
only runtime. The metric is unchanged, `cx+shy0.5+s0.125r1b1+R0.5g4x2q4+v34d40e8a`: no constant of
the complex's code moved (its digest is the same at every step), and every row keeps its name.
Python under research/roadless: 125 tracked modules and 34,899 lines before, 67 modules and 17,826
lines after (fft_homog's 3,236 of them; outside it 31,342 lines before, 14,590 after). Every row,
summary, log and picture stays on disk: no code reads the raster's rows, they are records.

**Moved** (step 1), so the deletions broke nothing kept: the runtime's Spec at a spacing, ring and
band read from RUNTIME's name (metrics.spec_of, metrics.py:33; gap_study.spec_of and gap_check's
copy read lifted.RING_M and BAND_M); a fixed, private grammar of the raster's names in
gap_clear.check_disjoint (gap_clear.py:1619; it agrees with lifted's parsers on all 25 names on
disk and in the store, and is no constant of the code's digest); the block-id parser common.ids_of
and LARGE (common.py:44 -- 69, from gap_study); gap_calib.size_of (gap_calib.py:103 -- 117; the
size classes' line, 49M unknowns on the raster at h 0.5, kept as data); the walled channel's L_C,
WALL_M and ANGLES (gap_flow_check.py:62 -- 64, from width); seqscreen's loaders, TUNING and
blocks_of (seqscreen.py:58 -- 112, from simp_compare, the raster greedy's own loader dropped);
pweight.boot_weights (pweight.py:68 -- 69, 274, from transparency_run); transparency_check's
`free-roads` guard as free_roads_check.py with FREE's configuration (below). Cut from the kept
modules' paths: the `mesh` argument the complex ignored (polish_greedy, road_rescore: rows no
longer carry `mesh`; the exporter drops the old rows' placeholder by name); metrics.metric_of's
raster branch (it takes a complex name and a device); clearings_check.metric_of_run's (a raster run
is a record, not rescorable under its own metric: it raises); cluster.py's raster sizing (`--mesh`,
`--scans`, `--complex`, GB_PER_MILLION, HELD_PER_MILLION, RASTER_BYTES, `_gb`: `--gb` now sizes
every card) and cluster_check's `gpu`, its guard (`cpu` now plans gap_calib's `preset`);
gap_calib's `rasterscore`, its raster SOURCES and the acceptance's comparison with the raster
(summary_runtime_acceptance.csv the record); road_rescore's cx_stored without the raster's runs.
The exporter (clearings_export) reads this code's metric only (`this_code`: a complex name
complex_of reads back): every other metric's rows (the raster's names, clear_rows_<picker>_h<h>...,
pweight_rows/simp, fw, score and uni+apwfs, phase 1's undigested complex name) are records it
counts and does not read (57 directories), the raster's collected parquets are records, and a row
must state `enclosed` (the preB epoch went with the raster's rows). Its `inventory` raised on phase
1's undigested directories before this phase; it runs again.

**Deleted** (step 2): 58 modules. The raster's own (b), 29: lifted, width, width_check,
aperture_check, checks, angle_h, angle_q, sightline_angle, sightline_checks, line_scale,
transparency, conductance_map, current_map, opened_map, render_clear, render_alongs, render_simp,
atlas/block5810/export_ss, composite_proto/proto, offset_check, resolution_check, rescore_grid,
gramprobe, memprobe_greedy, profile_round, releaseprobe, solver_bench, timing, conv; and crosscheck
(the cross-checks' raster side). The finished studies (c), 20: study, tail, tails, rescore,
compare_clear, area_match, along_lineup, picker_compare, simp_compare, mono_cross,
transparency_run, transparency_check, gap_simp, fft_homog's extract, extract2, extract_gated,
lifted_ref, region_extract and region_score_gpu, atlas/region5810/export. Eight more the inventory
did not list (none imported a module of the raster's own): gap_study (left empty once spec_of and
ids_of moved: the rest the gap complex's stage-1 study), gap_report (its report), epireplay (the
replay against seqscreen, whose arms are the raster's rows), transparency_sets (the transparency
experiment's sets), mesh_bias (resolution_check's rows), area_share (the raster greedy's rows),
atlas/block5810/vectors_ss (export_ss's arm) and atlas/region5810/compare (the raster region run's
clearings). The mixed modules' raster parts: clear (Clearing, its Tension, loo, exact, RegionGraph,
free_first, GpuSweep, the greedy's runs, sweep_of, cleared_through, RTOL_TENSION); relax
(Relaxation, RasterMetric, _clearing, the coarse plans `.c` and `.C` and the grids `.h`, `.a`,
`.g<d0>x<smax>...`); common (Scorer, Obliterate, Carve, placement, carved, reached, GoalMesh,
goal_indicator, mesh_of, the Demands' `baseline`); road_rescore (`score`, Corridors, GpuWatch,
`report`, `sizes`, `project` and their helpers); gap_compare (all but ComplexModel and form_of: the
stage-2 study's commands). clear.rows_dir, the greedy format's directory, is the complex's:
clear_rows_<spec>_p<power>_<metric>, no `_h`, so no complex run's directory reads as a raster
record's (none existed in the old form).

**Folded** (step 3): seam.py's Protocols each had one implementation once the raster's went, so the
callers use the concrete types (gap_clear.Model, Relaxation, Tension, ComplexMetric) and seam.py is
deleted; Flow moved to clear without its device fields (xp, to_host: the complex's flow is on the
host); clear's Sweep and CpuSweep and GramSource folded into Catchment (one sweep, CATCH_COLUMNS 16
a pass, the same chunks as CpuSweep's, so the same shares), search.greedy_round and plan_of no
longer take a sweep, ComplexMetric.sweep and .coarse and Model.release gone; the relaxation's
`rtol`, kept for the seam (the raster's iterative tolerance; every solve on the complex is direct),
gone from Model.relaxation, with polish_greedy.RTOL, pweight.FW_RTOL and clear.RTOL_SCORE. The
factorization Strategy stays (SuperLU on a core, cuDSS on a GPU: both live).

**The cheap preset is the translucent search** (step 4): polish_greedy.CHEAP is S0.01cat P64w8 with
`search_eps` 0.05 (Preset.on gives the metric its runs are made and named under, `<metric>@e0.05`),
phase 2b's measure of it standing (+0.0099 [+0.0046, +0.0192] over the plain form on the 13, ahead
on 11, 1.03x its time). No named spec of the plain form is left: gap_calib's items spell their
preset's search (`simp[E]`, `cheap[E]@e0.05`; a bare `cheap` and an `@e` on `simp` are refused, so
no new row lands under the plain form's names), road_rescore's `cx` runs the cheap preset under its
search (its rows under polish_rows/<metric>@e0.05/, its clearing `cheap_cx<E>@e0.05` in a block's
plain rows, cxreport reading either). SIMP's default and its polish keep the metric's own tension
(search eps clear.EPS); the search eps stays a required field of ComplexMetric, in its name. The
plain form's rows stay under their names (polish_rows/<metric>/S0.01cat.P64w8[E]/,
gap_rows/runtime/preset/<device>/cheap[E]/: records, read as phase 1's references by gap_calib's
report and as the 13's stored clearing by road_rescore's cx_stored).

**An unknown block id raises** (step 5): common.build_blocks checks every id against what builds it
(`_known`: the bank's ids, or the source's blocks once built, its building floor applied) and names
every id it lacks; write_bank goes through it, so a submit's plan refuses a mistyped id before any
task is staged; and the entry points that take ids parse them with common.ids_of, which builds them
(CHECK_BATCH at a time) at the boundary, before any work: relax, search, road_rescore's propose-all
and prefixes, pweight's score, gap_calib, clearings_check's rescore, and the guards
(gap_clear_check, gap_check, pweight_check, free_roads_check); the single-block ones build at once.
Phase 2b's ZAF.9.3.1_1_7851 (for ZAF.9.5.4_1_7851) is a block the source holds under its building
floor, so it yielded nothing and write_bank wrote a bank without it. The guard, cluster_check
`ids`: the mistyped id is refused by ids_of, write_bank, a `blocks` submit's plan and build_blocks
on a bank that lacks it, and the id spelt right parses and plans; fault `no-check` (`_known` a
no-op): ids_of and write_bank accept it, the bank written with ZAF.9.5.4_1_7851 alone, and the plan
stages a task for it, as in phase 2b (both ways here, 14 s each).

**The guards** (each passes on the final code and fails under its fault; on a core here, the device
guards on the cluster's cards): after step 1 (moves) and after step 2 (deletions) all 28 then on a
core passed (f3ac2de5, 0c2e588e: pass only, 0.9 h each); at the final code (c86f030a) every guard
both ways here, 7 single-threaded processes at once: 29 pass and each of their 31 faults is seen
(60 runs, 1.9 h of core time). gap_clear_check: baseline (20269 P0 -1.6e-4 against the rebuild,
singles within 3.9e-6; faults interior, its single 4.0e-3 off, and exits, P0 +68%), weights
(no-conform: the mesh refused, 728 passages), reach (threshold), gradient (within 4.8e-6 of the
largest; tie 7.3e-2), free (drop-crossing), identity (7 of 7 raster names refused, each raster run
a record; collide: 0 of 7), convex (multiply), pattern (drop-identity), constants (28 constants,
the digest 34d40e8a; omit), roads (18895's resistance_lp L_P +1.2e-4 at 38 roads, +3.2e-4 at 260;
one-face-closed -5.2e-3), flux (imbalance 2.7e-11 -- 5.0e-11; flip-one 1.0); gap_check conform
(unconstrained), delaunay (flip, geos), apertures (corners: 1.70 for a gap of 1 m), filtration
(narrowest), free (no-rebuild); gap_flow_check angle (1.0003 at 4 m), shift, channel, field
(centroids each), law (no-grow 0.82), closed (no-clearance); pweight_check identity
(no-population), total (brief 0.47), displacement (cost-from-demand), gradient (J-population),
weights (1.1e-11; swap-time 0.65); free_roads_check free-roads (one-short, above); cluster_check
ids (no-check, above). cluster_check cpu plans on both nodes. On the cluster's cards (run cx-p3-dev
at c86f030a, one submit, gap_clear_check's new `both-ways` running both guards and their faults in
one task, 20269 and 18895 with their road rows shipped): device passes within 1.5e-10 and 5.1e-11
and fails under perturb (0.97 and 0.86 off), device-roads within 8.7e-11 and 3.6e-11 and perturb
0.88 and 0.69. clearings_check rescore: the complex's SIMP default and cheap preset on 20269 under
their own metric MATCH (diff 0); a raster run under its own is refused. clearings_check blocks: all
303 committed blocks rebuild through the checked build_blocks as the blocks their clearings were
committed for, and the 61,525 rows of the 157 runs pass the row checks (70 s).

**The tests** (tests/test_clearings.py, 36 pass): the exporter's tests ported to this code's
metric; deleted with the code they tested: the mesh name of a plan (clearings_export.mesh_name) and
the raster greedy directories' parse into settings (GREEDY_DIR's old form); new: the complex's
greedy directories parse back, the raster's are records, only this code's metric is a source (the
raster's names and another digest's complex listed as records), and a row must state `enclosed`
(each failing under its fault: `_record` False, `this_code` True, the check removed). mypy --strict
passes on the modules that were clean (metrics, search, polish_greedy, clearings, clearings_export,
clearings_check, free_roads, gap_maps, gap_factor, cluster_unpack, seqscreen; seam deleted) and on
free_roads_check, clear, cluster and cluster_check (new or newly clean); `ruff check .` passes.

**Choices the brief left open** (each the option that deletes the least knowledge):
- Finished-study code the inventory did not list was deleted where it read only the raster's rows
  or lost its input or its last caller (the eight above), and kept where neither: fft_homog's
  solver and two-scale code (fftk, ts, region_* and the rest: the FFT screening layer is a tracked
  lead, BACKLOG 4(c)), perm_floor, atlas/nbhd, the region page's style and vectors. They read no
  raster code; the consolidated pass can weigh them.
- transparency_check's `free-roads` guard moved to free_roads_check.py: it guards free_roads, which
  stays (the consolidated pass's road free-first stage), while the transparency experiment's code
  went. Its plans are held on the three plan methods' own graphs (clearance_looped's tree,
  cycle_native, resistance_lp); the claims' routed graphs (TransparentSubstrate over the stored
  raster clearings) were the experiment's. FREE's configuration (budget 0.05, margin 0.05 m, the
  length tie) moved with it. Its fault `one-short`, ad hoc in phase 2b, is coded: every road the
  stage routes one point short; it bites on 18895 (its road of 2 nodes and 8.5 m held 1 node short:
  the stage raises, home 510 not freed by the metric's test); 20269's road is a single node at the
  street, nothing to shorten.
- road_rescore's `cx` scores no raster clearings now: the 60's plain rows hold them (the
  cross-checks' record); a block run anew gets the complex's.
- gap_compare keeps ComplexModel (the guards' and the calibration's rebuild) under its name.
- relax.Plan's `t<rtol>` stays in the plan's name: it moves nothing on the complex, but dropping it
  renames every stored SIMP run (relax_rows/<metric>/fw0.q3.i10.t0.001..., the store's run ids); a
  migration for the consolidated pass, not done here.
- clearings.py's RunInfo keeps `mesh` and `metric` (B/preB): the store's schema, which the raster's
  committed runs fill.

**Not deleted, and why**: the raster's rows, summaries, logs and pictures (records, as the brief
says); clearings.py's schema fields above; cupy and pyamg in the `gpu` group (cupy reads the card's
memory in gap_calib and serves fft_homog; pyamg is fft_homog's solver); free_roads.CODE hashes
common.py's source, so FreeEnclosed's identity changed with common (its derive-cache entries are
recomputed once); the docstrings' history (what a deleted module did, by name, where it explains a
rule the complex keeps: the raster's Placement, its free first, its ground).

**For the wiki's dead-end record** (the controller writes it; each retired variant with its numbers
and where NOTES has them):
- The lifted metric (the heading lift and its turning cost, ell 3 m, K 8): its boundary layer loses
  0.3 m of a lane at each wall in a 1 m lane, 0.5 in 2 m, 0.7 in 4 m, 0.8 -- 1.1 m in 8 -- 16 m
  (lanes conduct 0.33 -- 0.89 of their width's share): the old superlinear width law was the
  lift's; it orders the methods as the raster's isotropic model does (tau median 1.00 on P and J_2,
  0.93 on P_{1/2}) and reads levels 0.04 -- 0.06 higher ("The gap complex: passability, enclosure
  and flow without the raster").
- The raster's lattice: its isotropic model conducts a lane like one 0.14 -- 0.29 m narrower at
  each wall (the complex within 3e-4 at every width); 22422's 14 m lane, 0.25 -- 0.4 m wide, it
  resists 4x as much (the same section). Under Fws the angle error at 1 m is 1.21 (U before the fix
  1.66), the levels' drift 0.027 ("The grid fix").
- The sightline family (Sightline, SoftSightline, TurnBoosted): the angle error 5.32 at 1 m against
  U's 1.66 ("The sightline metric's angle error"); parked 2026-10-13 with decision (3); its
  translucent search's +0.014 [+0.002, +0.022] on the 45 held out ("Sightline on the held-out
  blocks at D 0.05") is what the complex's `@e` search ported.
- B_r and D_s (effective width): no new corridors at r up to 1.5 m, U's method order kept (tau
  1.00), angle errors 1.72 -- 1.85 and 2.39, SIMP under B_r no gain on its own lens ("Uniform
  conductance with an effective width").
- A, Aw and Aws (the face aperture's rivals): angle ratios 1.54 -- 1.83 (A) and 1.07 -- 1.22 (Aw,
  Aws) against Fws's 1.03 -- 1.21, 7 -- 33% of demand cells unreached; deleted with Fws's adoption
  ("The grid fix").
- The composite and goal meshes: a5x8 reads Lens A low by a mean 0.0026 (14401 0.0095); the goal
  mesh g2.5x8f1.4p2o1 at the same cells 0.0004 (4.9x closer, median) ("Local coarsening: built",
  "Goal-oriented refinement"); the complex needs neither (one fixed mesh a block).
- The raster's speed: on its GPU (H100 or the tesla node) the presets took 26 -- 337 s (SIMP) and 8
  -- 166 s (the cheap preset) on the 13, the complex on one core 3.7 -- 73x that (median 11x) and
  4.7 -- 40x (median 13x) (phase 1: "one fixed mesh"); on a GPU the complex's presets take a median
  0.38 (SIMP) and 0.93 (the cheap preset) of the raster's GPU time (phase 2a: "its speed and a GPU
  path").
- The cross-checks (phase 2b, "the ports and the cross-checks"): the road order the raster's at
  Kendall tau +0.83 (+1.00 where B (i) ties); P_{1/2}'s weights Spearman +0.99; the corridors agree
  (the raster's main lanes open on the complex but for 0.1 -- 1.0% of their cells); each metric
  ranks its own optimizer's clearings first; where they part it is the shy distance (the metric's
  choice) or the raster's lattice (its error).
- +relaxlumped (the eps world's interiors one cell a building): Lens A lower by 0.023 and 0.017
  (SIMP, core), 0.011 and 0.010 (the cheap preset), slower on both devices ("its speed and a GPU
  path").

**Compute**: GPU 0.31 GPU-h (the device guards: 10 and 8.5 min on tesla2). CPU here,
single-threaded, at most 8 processes: the guards three times (pass after steps 1 and 2, 0.9 h each;
both ways at the end, 1.9 h), the checks of the fold (19570's catchment and greedy, old against
new) and of the preset (20269's cheapE@e0.05), the id guard, clearings_check: about 5 CPU-h
[inferred from the runs' times].

### Pruning by epistemetry: a replay against seqscreen (owner 2026-10-14: "Yes"; BACKLOG, "Pruning by epistemetry's decision layer")

Does GeoffChurch/epistemetry's decision layer decide which variants to keep better than
seqscreen.py, on comparisons already run on every block? epireplay.py replays both through
epistemetry's own `replay` (an ephemeral `uv run --with` overlay: reblock does not depend on it),
every policy on the same 100 orders of each design (seqscreen's 20 first, then 80 more from its
generator), all judged against the same truth: the call on all of the design's blocks.

**The rule, fixed before any run** (12a46054, the brief): adopt epistemetry for the consolidated
pass's pruning only if (1) its median realized loss is below seqscreen's at every rate tested and
(2) it makes no more wrong calls in all; a many-arm screen counts against the pairwise screens it
replaces. **Amended** (the coordinator, 2026-10-15; committed in 674a599e before any
adopt/keep/ask run): the pin moves from fc921f1 to c1784af, whose `AdoptKeepAsk(low, high)` is
seqscreen's own call over the bracket (per draw adopt if g > r c at both ends, keep if below at
both, else ask; the decision each arm's likeliest call). It is the primary comparison, scored with
`AsksApart` (an ask is never wrong; a definite call is wrong where it differs from the whole
design's call, an ask included), run with the veto nowhere (seqscreen's patched off inside the
process, no file edited; the truth without it) and everywhere (seqscreen's own, a wrapper around
epistemetry's policy, the truth "keep" on a collapse). Asks go to the owner, so (3) epistemetry
must also make no more asks in all. The fixed-rate runs already made at fc921f1 are reported
under the original rule. The statistic: per rate, the sum over the designs of each one's median
over orders; wrong calls and asks summed over designs and orders. The realized loss at rate r: a
call resolved at r (an ask answered at r from the screen's own means, which is seqscreen's
question) that differs from all the blocks' call at r costs |mu - r x extra|, plus r / FUTURE x
the GPU-seconds spent.

**Designs**, rebuilt through seqscreen's reader (units blocks, an arm a variant's Lens A gain over
its base, a unit's cost its variants' GPU-seconds; one kind of card a design):
- the six held-out screens seqscreen was built on (cluster V100 / RTX 8000): `.p64w8`, `.p256w8x2`
  (x2), `.r2l2.p64w8` (both) and `.r2` against base; x2 and both against `.p64w8`;
- the other pairs the readers hold on one card, the 46 on the local RTX 6000 Ada:
  `.pP256w8.gS0.01cat` (pPg) and `.p256w8x2.GS0.01cat` (GS) against `.p256w8`; the cheap preset's
  pairs polish (S0.01cat.P256w8x2) and `.p256w8` against the cheap preset; S0.005cat and the cheap
  preset against the greedy S0.01cat;
- one collapse, for the veto: `.p64w8.C1.m4` against `.p64w8` on its 12 tuning blocks (30848
  -0.255; fewer blocks than the floor, so only the veto can save anything);
- many-arm: r2, p64, x2 and both against base; x2 and both against p64; pPg and GS against
  `.p256w8`; the pairs polish and `.p256w8` against the cheap preset; S0.005cat and the cheap
  preset against S0.01cat.

The default's and the cheap preset's rows of 10 held-out blocks are free-first runs under B (the
overwrite of 2026-10-11, "The stored clearings, committed"; their rows carry `enclosed`), so every
design with either runs on the other 36. That moves the x2 screens from "Stopping by the value of
more blocks": read whole today, x2 against p64 is -0.00052 at +8.0 s a block (seqscreen: keep);
on the 36 it is +0.00016 at +11.4 s, the old near-tie (an ask). seqscreen's own output for the
four screens without dropped blocks is NOTES' table of 2026-10-09 to the digit.

**Choices the brief left open**, each to keep the two sides on one footing:
- tau and the cost model's ratio re-estimated from the blocks seen, as seqscreen re-estimates
  them, so epistemetry's policy is rebuilt each step; its floor (15) held in reblock's wrapper too,
  since no target exists before a block is seen;
- epistemetry's targets have 0-1 risks (the probability of another call), so ValueOfInformation's
  price is seqscreen's (rate / FUTURE; for adopt/keep/ask the bracket's mean rate, the rate
  seqscreen prices its stop at) over the Lens A a wrong call costs at the means so far, the mean
  over arms of |mu - tau|; no Monte Carlo margin (seqscreen has none);
- every horizon to the last block offered (seqscreen's stop weighs every k); outer draws 4000
  (seqscreen's paths), inner 200;
- the veto wrapper runs the policy's batches a block at a time, so it sees every block as
  seqscreen's does; no veto in the many-arm screens (a collapse would drop one arm, which the layer
  cannot do), and none of them has a block below -0.1. With and without the veto every run is
  identical, both policies, on all 12 collapse-free designs (checked job by job).

**The primary comparison, adopt/keep/ask over the bracket** (100 orders a design; blocks run and
cost as a share of all the blocks' are medians; wrong calls and asks are of 100):

    design          all     seqscreen                    adopt/keep/ask
                    blocks  blocks  cost  wrong  asks    blocks (max)  cost  wrong  asks
    p64/base        adopt   15      0.33  0      0       15 (15)       0.33  0      0
    x2/base         adopt   15      0.41  0      0       15 (15)       0.41  0      0
    both/base       adopt   15      0.34  0      0       15 (46)       0.35  0      0
    both/p64        keep    15      0.34  0      0       15 (46)       0.34  0      5
    r2/base         ask     20      0.43  2      98      15 (46)       0.36  2      98
    x2/p64          ask     15      0.44  35     65      15 (36)       0.46  18     82
    pPg/p256        keep    15      0.34  0      30      15 (46)       0.35  0      31
    GS/p256         ask     15.5    0.36  28     72      21.5 (46)     0.46  16     84
    cheapx2/cheap   adopt   15      0.43  1      15      15 (36)       0.46  0      12
    p256/cheap      adopt   15      0.42  0      0       15 (15)       0.42  0      0
    g005/g01        adopt   15      0.34  0      0       15 (15)       0.34  0      0
    cheap/g01       adopt   15      0.43  0      0       15 (36)       0.53  0      0
    C1m4/p64        keep    12      1.00  0      0       12 (12)       1.00  0      0
      with the veto         5       0.48                 5             0.48

    the rule         realized loss (1e-6) at 2e-6, 6.2e-6, 1.9e-5   wrong  asks  cost
    no veto  seqscreen        57.3   178.2   538.9                   66     280   0.431
             adopt/keep/ask   66.9   193.0   554.3                   36     312   0.447
    veto     seqscreen        53.5   166.3   502.0                   66     280   0.391
             adopt/keep/ask   63.0   181.0   517.4                   36     312   0.407
    many-arm seqscreen x arms 132.8  174.3   521.3                   67     279   0.424
             adopt/keep/ask   116.6  359.5   1007.3                  26     309   0.821

**Decision: keep seqscreen.** In both veto settings adopt/keep/ask's median loss is above
seqscreen's at all three rates and it asks more (312 against 280), though it makes fewer wrong
calls (36 against 66). The many-arm screen makes far fewer wrong calls (26 against 67) but asks
more (309 against 279) and costs more at the two upper rates: it runs every arm on all 36 blocks
in most orders of 4/base, 2/p64 and 2/cheap (cost 1.00, median), against 0.42 -- 0.46 for the
pairwise screens it would replace; it wins only at the bracket's bottom (116.6 against 132.8).

Where they differ:
- **Wrong calls for asks.** All but one of seqscreen's 66 wrong calls are definite calls on
  designs whose whole call is an ask (x2/p64 35, GS/p256 28, r2/base 2; the other a keep on
  cheapx2/cheap): adopt/keep/ask asks there instead (82 against 65 on x2/p64, 84 against 72 on
  GS/p256), about one more ask for each wrong call it saves (32 for 30). Under the amended rule
  both count, and the trade is a wash.
- **Whole horizons.** KG* picks the horizon with the best value per cost, and the policy runs all
  of it before deciding again: value of information is not concave, so that is often the rest of
  the design. adopt/keep/ask runs to the last block in some orders of 8 of the 12 designs larger
  than the floor, where seqscreen's longest run is 40 of 46. Its loss is at or above seqscreen's
  on every design but r2/base, where it settles the ask at the floor (15 blocks against 20).
- **The call at a stop.** epistemetry decides by the likeliest call over the draws, seqscreen by
  the posterior mean; at all 6400 of epistemetry's stops here (arms x orders x settings, the
  many-arm screens and the diagnostic below included) the two agree. The differences are in the
  stopping, not in the call.

A diagnostic after the results, not a candidate (`Stepped`, df863119): the same policy asked again
after every block, its stop still weighing every horizon. It closes most of the gap: with the
veto its loss is 56.3, 171.0 and 495.2 against seqscreen's 53.5, 166.3 and 502.0, below at the
bracket's top only, with 37 wrong calls and 318 asks (66 and 280). What remains is GS/p256 (17
blocks against 15.5; 7.3 against 5.1 at the bottom rate), pPg/p256 at the bottom rate and the
asks.

**The fixed-rate comparison under the original rule** (fc921f1; seqscreen judged at each rate by
its call there, an ask answered from its means; epistemetry run at each rate with tau = rate x
the extra seconds: three-way at tau, the brief's, functional d / tau with margin 1 and the price
as above at that rate; and seqscreen's own linear loss written as an epistemetry DrawTarget,
`Regret`, priced at rate / FUTURE, reported beside it):

    pairwise                 loss (1e-6) at 2e-6, 6.2e-6, 1.9e-5   wrong (3 rates)  cost
    seqscreen                53.5    166.3   502.0                  179              0.391
    three-way + veto         65.9    227.5   504.5                  173              0.398
    linear + veto            58.2    165.1   488.4                  149              0.390
    many-arm
    seqscreen x arms         132.8   174.3   521.3                  177              0.424
    three-way                79.8    334.8   684.1                  60               0.519
    linear                   71.6    182.2   471.1                  133              0.420

Keep seqscreen here too: three-way loses at every rate (it runs to the last block in 10% of its
runs, seqscreen never), with 173 wrong calls against 179. The linear loss comes closest: below
seqscreen at the two upper rates with 30 fewer wrong calls, above it at the bottom (58.2 against
53.5), so it fails (1). Its target is BestArm against a constant arm at tau, in the layer's terms.
seqscreen's losses here equal its veto-setting ones above, to the digit: the same stops, judged
the same way.

**What would have to change for epistemetry to win** (the first measured, the second in part,
the others inferred):
1. Re-decide after every block (or after a card's batch) while weighing every horizon for the
   stop, instead of committing to the horizon KG* picks: the stepped diagnostic above, which still
   falls short at two rates and on asks.
2. A linear-loss target over the bracket (seqscreen's loss: the opportunity loss of adopt against
   keep, averaged over the rate), priced in the owner's units, instead of a 0-1 risk priced
   through a plug-in conversion; at fixed rates the linear arm is the closest result.
3. The asks: under AsksApart and the amended rule a policy that asks instead of calling a near-tie
   trades one count for the other; to win on both it must stop where the call is clear or spend
   enough to settle it, which is a question of the target's loss on an ask (an ask costs the
   owner's attention, which no target here prices).
4. Many-arm: one screen pays every arm on every block it runs, so it needs a per-arm drop (an arm
   whose call is settled, or that collapses, stops costing) to beat the pairwise screens on cost.

**Guards** (each with its fault; `guards none|floor|ignored|shuffled`):
- wrapper: seqscreen as a Policy reproduces seqscreen's own replay order by order (stop and call,
  all 100 orders of all 13 pairwise designs, in both comparisons) and seqscreen.main's printed
  summary on the 8 designs without dropped blocks: 2,608 of 2,608. Fault (the wrapper's floor
  14): 725 of 2,608 (all of C1m4's, which stops at its collapse first), FAIL.
- degenerate: on p64 against base with p64's gains replaced by the base's (p64's time kept),
  every epistemetry policy (adopt/keep/ask and the 12 fixed-rate ones, with and without the
  veto) stops at the floor with "keep" in all 100 orders. Fault (the floor ignored, 1): every run
  stops at 1 block, FAIL.
- same orders: every policy on an order was handed that order, in both runs and on both/p64's
  every policy. Fault (adopt/keep/ask without the veto handed a reshuffled order 3): that one run
  flagged, FAIL.

**Feedback for epistemetry** (relayed to its author):
- `Policy.next` gets the upcoming units' indices but not the seen ones', and `calls` neither, so
  a policy that reads per-unit side data (the base's time on the blocks seen, for tau) keeps the
  order itself and must guess that a `calls` before any order is the whole design in index order.
- `CostModel.predict(seen_costs, upcoming)` cannot express seqscreen's model (the base's time on a
  block x the ratio over the blocks seen) without the seen indices; the policy is rebuilt each
  step to close over them.
- KG* commits to the whole horizon it picks: offer every horizon to the stop but step a unit (or a
  card's batch) at a time, the batch size apart from the horizon menu.
- The targets' risks are 0-1, so ValueOfInformation's price has no meaning in the owner's units
  (Lens A per GPU-second over FUTURE); a linear-loss threshold target (min(E(tau - x)+,
  E(x - tau)+), BestArm against a constant arm) and its bracket form would take the rate as given.
- ThreeWay's margin is fixed and symmetric about 0, so a data-dependent, one-sided tau needs
  functionals scaled by 1 / tau (impossible at tau <= 0); AdoptKeepAsk's extra-cost functionals
  fix this.
- One price, not a bracket: seqscreen averages the value and the price over the owner's unknown
  rate; ValueOfInformation could take the rate as a distribution.
- No veto, and no per-arm drop in a many-arm screen (a unit's cost covers every arm), which is
  what makes one screen dearer than the pairwise ones.
- FinitePopulation's population is the design's units, exact once all are seen: right for the
  replay's truth, not for a call that governs FUTURE block runs.
- `replay` spawns its own stream per order and does not pass the order's number, so a policy with
  its own per-order seed needs a replay call per order; a shared truth across policies has to be
  asserted by the caller (each policy's whole-design calls must agree).

Compute: here, single-threaded, at most 8 processes: the fixed-rate run 1.4 h on 8 workers, the
bracket run 1.1 h on 7, the stepped diagnostic 3.0 h on 4, the four guard runs 1.75 -- 1.85 h
each. Rows (git-ignored): epireplay.parquet, epireplay_bracket.parquet and its jobs'
epireplay_bracket_jobs/; epireplay.py, deleted in phase 3, regenerates them from 0c2e588e^.

### Pruning by epistemetry, again: step-wise re-decision and per-arm screens (owner 2026-10-15; BACKLOG, "Pruning by epistemetry's decision layer")

The first replay (above) kept seqscreen and named what would have to change: re-deciding after
every block, and a per-arm drop in many-arm screens. The owner approved both (2026-10-15) and
GeoffChurch/epistemetry built them: bbe2163, ValueOfMoreUnits' required `step` (every horizon
valued, at most `step` units of the best one's observed before deciding again), and 64aae7f,
per-arm screens in its replay harness (each arm on its own prefix of the order, stopping on its
own value at its marginal cost; the veto a drop rule of the harness, applied to every policy and
to the truth alike). The replay ran again before the consolidated pass, under the same rule,
pinned at 64aae7f through an ephemeral overlay as before.

**The rule, as fixed** (the first replay's, amended): adopt epistemetry for the consolidated
pass's pruning only if (1) its median realized loss is below seqscreen's at every rate tested
(per rate, the sum over the designs of each one's median over orders), (2) it makes no more wrong
calls in all and (3) no more asks in all; pairwise in both veto settings, and a many-arm screen
counted against the pairwise screens it replaces. Realized loss, wrong calls (AsksApart) and asks
as before.

**The designs, frozen as data.** The first replay's 13 pairwise designs (the collapse C1m4/p64
among them) and 5 many-arm ones were extracted once by its own code (`git archive 0c2e588e^`, run
in a scratch directory with the git-ignored rows linked in; at 0c2e588e^ seqscreen's loaders had
already lost the raster greedy's, so it was handed simp_compare's, as when the first replay ran)
into epireplay_designs.json (committed, 263 kB): per design the names, the block ids, each arm's
Lens A gain over the base and its GPU-seconds, the base's GPU-seconds and the 100 orders. The new
epireplay.py reads only that file; nothing reads the raster's rows. Its `check` holds the file to
the first replay's stored runs (epireplay_bracket.parquet, equal to its 95 jobs' files): 9,500 of
9,500 runs were handed the design's order, 9,500 of 9,500 whole-design calls are the data's, and
3,800 of 3,800 stored seqscreen runs (blocks, call, GPU-seconds) replay exactly from the data.

**The test.** Every policy on the same 100 orders a design (one `replay` call an order, its
generator [ORDER_SEED, order]), judged against the same truth, in both veto settings: everywhere,
the harness's VetoBelow (-0.1 on the gain, KEEP); nowhere, NoVeto with seqscreen's own COLLAPSE
patched off in the process.
- seqscreen unchanged, as a Policy of one arm: stop_at on the arm's prefix, a block a step.
- ValueOfMoreUnits on FinitePopulation with AdoptKeepAsk over the bracket, matched as before:
  floor 15, every horizon, seqscreen's ratio cost model (now a CostModel holding the base's times,
  handed the seen blocks' indices), no Monte Carlo margin, and the price rate / FUTURE over the
  Lens A at stake at the means so far, rebuilt each decision (the mean over the live arms, which
  share one prefix, of |gain - rate x extra|). At `step` 1. A card's batch is one block here (the
  local card runs one GPU process; the cluster one card a block-task), and seqscreen, unchanged,
  decides after every block, so step 1 is the only batch that fits the first replay's choices; no
  other was run.
- many-arm: one ValueOfMoreUnits at step 1 over every variant of a design, each arm stopping on
  its own value (the per-arm screen), against seqscreen once per variant on the same orders,
  summed. No block of any many-arm design is below -0.1 (the worst -0.012), so the two veto
  settings are the same run there; run without, as before.

Per design, without the veto (C1m4/p64's veto line beside); blocks and cost are medians over the
100 orders, wrong calls and asks counts in them; many-arm in block-arm runs, its whole calls per
variant (A adopt, k keep, ? ask), seqscreen once per variant summed:

    design         all    seqscreen             step 1                   first adopt/keep/ask
                          blocks cost  wr  ask  blocks    cost  wr  ask  blocks     cost  wr  ask
    p64/base       adopt  15     0.33   0    0  15 (15)   0.33   0    0  15 (15)    0.33   0    0
    x2/base        adopt  15     0.41   0    0  15 (15)   0.41   0    0  15 (15)    0.41   0    0
    both/base      adopt  15     0.34   0    0  15 (24)   0.34   0    0  15 (46)    0.35   0    0
    both/p64       keep   15     0.34   0    0  15 (43)   0.34   0    7  15 (46)    0.34   0    5
    r2/base        ask    20     0.43   2   98  15 (30)   0.35   2   98  15 (46)    0.36   2   98
    x2/p64         ask    15     0.44  35   65  15 (34)   0.46  19   81  15 (36)    0.46  18   82
    pPg/p256       keep   15     0.34   0   30  15 (36)   0.35   0   33  15 (46)    0.35   0   31
    GS/p256        ask    15.5   0.36  28   72  17 (44)   0.38  16   84  21.5 (46)  0.46  16   84
    cheapx2/cheap  adopt  15     0.43   1   15  15 (32)   0.45   0   15  15 (36)    0.46   0   12
    p256/cheap     adopt  15     0.42   0    0  15 (15)   0.42   0    0  15 (15)    0.42   0    0
    g005/g01       adopt  15     0.34   0    0  15 (15)   0.34   0    0  15 (15)    0.34   0    0
    cheap/g01      adopt  15     0.43   0    0  15 (30)   0.47   0    0  15 (36)    0.53   0    0
    C1m4/p64       keep   12     1.00   0    0  12 (12)   1.00   0    0  12 (12)    1.00   0    0
      with the veto       5      0.48   0    0  5 (12)    0.48   0    0  5 (12)     0.48   0    0
    4/base         ?AAA   64     0.46   3   97  62 (74)   0.45   2   98  144 (144)  1.00   1   99
    2/p64          ?k     33     0.44  35   65  40.5 (69) 0.55  18   87  72 (72)    1.00  13   93
    2/p256         k?     31     0.35  28  102  36 (76)   0.40  16  116  47 (92)    0.54  12  115
    2/cheap        AA     30     0.42   1   15  31 (47)   0.43   0   13  72 (72)    1.00   0    2
    2/g01          AA     30     0.45   0    0  31 (44)   0.49   0    1  30 (68)    0.56   0    0

    the rule        realized loss (1e-6) at 2e-6, 6.2e-6, 1.9e-5   wrong  asks  cost
    no veto  seqscreen               57.3   178.2   538.9          66   280  0.431
             step 1                  60.2   183.0   532.1          37   318  0.433
             first: adopt/keep/ask   66.9   193.0   554.3          36   312  0.447
             first: stepped          60.2   183.0   532.1          37   318  0.433
    veto     seqscreen               53.5   166.3   502.0          66   280  0.391
             step 1                  56.3   171.0   495.2          37   318  0.393
             first: adopt/keep/ask   63.0   181.0   517.4          36   312  0.407
             first: stepped          56.3   171.0   495.2          37   318  0.393
    many-arm seqscreen x arms       132.8   174.3   521.3          67   279  0.424
             per-arm, step 1         66.9   204.5   526.1          36   315  0.464
             first: adopt/keep/ask  116.6   359.5  1007.3          26   309  0.821

**Step 1 is the first replay's stepped diagnostic, run for run.** On all 2,600 pairwise runs (13
designs, 100 orders, both settings) its blocks, calls and GPU-seconds equal those of `Stepped`
(df863119) exactly: min(step, the best horizon) at step 1 is what that wrapper did by hand. The
pairwise result is therefore the diagnostic's, now from the layer itself.

**Decision: keep seqscreen, pairwise and many-arm.**
- Pairwise, in both settings, step 1 fails (1) and (3): its loss is above seqscreen's at the
  bracket's bottom and middle rates (no veto 60.2 against 57.3 and 183.0 against 178.2; veto 56.3
  against 53.5 and 171.0 against 166.3) and below at its top (532.1 against 538.9; 495.2 against
  502.0), with 37 wrong calls against 66 and 318 asks against 280.
- Many-arm, the per-arm screen fails (1) and (3): below seqscreen once per variant at the bottom
  rate (66.9 against 132.8), above at the middle (204.5 against 174.3) and top (526.1 against
  521.3), with 36 wrong calls against 67 and 315 asks against 279. The per-arm drop does what it
  was built for: the screen spends 0.464 of all its arms' GPU-seconds where the joint screen spent
  0.821 (seqscreen 0.424), and its loss at the bracket's top halves (526.1 against 1007.3).

Where they differ:
- **Asks for wrong calls, as before.** Step 1 makes 29 fewer wrong calls and 38 more asks, all on
  the near-ties: x2/p64 19 wrong and 81 asks against 35 and 65, GS/p256 16 and 84 against 28 and
  72, both/p64 7 asks against 0, pPg/p256 33 against 30. Under the amended rule both count.
- **Longer tails, and three near-ties.** Its median blocks are seqscreen's on 11 of 13 designs (15
  against 20 on r2/base; 17 against 15.5 on GS/p256), but its longest runs are longer on seven
  (24 -- 44 blocks against 18 -- 36: GS/p256 44 against 22, both/p64 43 against 36, pPg/p256 36
  against 24, x2/p64 34 against 22). Its loss is above seqscreen's at the bottom and middle rates
  on GS/p256 (7.3 and 22.8 against 5.1 and 15.4), x2/p64 (6.0 and 18.6 against 5.2 and 16.1) and
  pPg/p256 (4.6 and 11.0 against 3.5 and 10.5); its win on r2/base (6.9 and 21.6 against 8.4 and
  28.0), settling the ask at the floor, does not make that up.
- **Many-arm: the pairwise screens summed, priced together.** Where a design's variants have
  pairwise designs on its very blocks (2/p256, 2/cheap), the per-arm screen's median loss is within
  1.1e-6 of the pairwise step-1 screens' summed per order, at every rate (2/p256: 15.8, 45.2, 86.9
  against 16.5, 45.2, 85.8). It differs from them only by its price, which the live arms share
  (the layer takes one stop rule): beside a dearer arm a near-tie runs on, x2 in 2/p64 observing
  more blocks than in x2/p64 in 48 of 100 orders (median 17 against 15), cheapx2 in 2/cheap in 32.
  Against seqscreen per variant it wins 4/base at every rate (24.5, 77.6, 230.9 against 25.2,
  80.5, 238.2) and 2/p256 at the bottom (15.8 against 86.0: seqscreen's definite calls on GS,
  wrong at the low rate), and loses 2/p64 (60.9 against 46.8 at the middle) and 2/p256 at the
  middle (45.2 against 27.2).
- **The call at a stop.** The likeliest call and the posterior mean's agree at every stop, as
  before.

**What would still have to change** (the first measured, the others inferred):
1. Re-deciding after every block is built and measured: it closes most of the gap (above), not
   all of it.
2. A linear-loss target over the bracket, priced in the owner's units (Lens A per GPU-second over
   FUTURE), instead of a 0-1 risk priced through the plug-in stake: the remaining loss is on three
   near-ties at the bracket's bottom and middle.
3. The asks: an ask costs the owner's attention, which no target prices; to win on both counts a
   policy must stop where the call is clear or spend enough to settle it.
4. Many-arm: each arm priced by its own stake (a price, or a stop rule, per arm). The many-arm
   screen would then be the pairwise step-1 screens summed (it is already, where its arms' shared
   price does not move them), so it would pass where they pass: it needs 2 and 3 too.

**Guards** (each with its fault; `guards none|step|floor|ignored|observing|shuffled`):
- migration: the one-arm designs at step 46 (at least every horizon) reproduce the first replay's
  adopt/keep/ask runs exactly, 2,600 of 2,600 (blocks, calls, GPU-seconds, both settings). Fault
  (step 1 given instead): 2,046 of 2,600, FAIL.
- wrapper: seqscreen as a Policy reproduces seqscreen's own replay order by order (stop_at on the
  data's orders, 13 designs, both settings) and seqscreen.main's printed summary on the 8 designs
  without dropped blocks (main's reader handed the data file's rows): 2,608 of 2,608. Fault (the
  wrapper's floor 14): 725 of 2,608, FAIL.
- degenerate: on p64 against base with p64's gains replaced by the base's, step 1 and step 46,
  both settings, stop at the floor with "keep" in all 100 orders. Fault (the floor ignored, 1):
  every run stops at 1 block, FAIL.
- per-arm drop: on C1m4 beside a copy without its collapse (12 tuning blocks, the veto on), the
  collapsing arm observed no block past its collapse and was charged for none while the other ran
  on, in 100 of 100 orders; arms dropped on value (left out of a step while another goes on) kept
  their prefix and stopped paying, in 5 of 4/base's first 10 orders (15 arms) and in all 355 of
  the run's 500 many-arm screens with such a drop (504 arms). Fault (the dropped arm kept
  observing: every live arm stepped while any goes on, and no veto): 0 of 100 collapse runs and 0
  of 5 runs with a drop on 4/base, FAIL.
- same orders: every policy on an order was handed that order, at every decision, in all 9,500
  runs. Fault (step 1 without the veto on both/p64 handed a reshuffled order 3): that run
  flagged, FAIL.

**Feedback for epistemetry** (for its author):
- One stop rule, so one price, for every arm: the price here is the Lens A at stake at the means
  so far, which per arm is that arm's own; the layer cannot take it (its stop rule is not told the
  arm), so the live arms share their mean, and a near-tie beside a dearer arm runs on (above).
- The price depends on what was seen, so the policy is rebuilt at each decision; before any block
  there is nothing to price, and `calls` needs a ValueOfMoreUnits (a stop rule) though it reads
  none (`Observed(())` stands in for both). A price as a function of Seen would do away with both.
- `calls`' rows: AdoptKeepAsk's decision on one arm's draws is a length-1 array, so
  ValueOfMoreUnits' calls are (arms, 1) where a policy deciding one call per arm returns (arms,);
  `replay` takes both (its overrule and scoring broadcast), so Run.calls' shape depends on the
  policy.
- ValueOfMoreUnits' twelve fields are required but positional: outer and inner, or step and floor,
  swap without an error (kw_only would catch it).
- `replay` still passes no order number, so a policy with a per-order generator (seqscreen's)
  needs a call per order, each recomputing the whole-design calls; the truth across policies is
  still the caller's to assert.
- No record of the decisions: the guards wrap every policy (`Witness`) to see the order shown,
  each arm's prefix and the step answered; Run could carry its steps.
- Carried over: a linear-loss target over the bracket, the rate as a distribution, and
  FinitePopulation's population the design's (right for a replay's truth, not for FUTURE runs).

Compute: here, single-threaded, at most 4 processes: the extraction and the check minutes each,
the run 5.3 h on 4 workers (95 jobs, the step-1 ones most of it), the six guard runs 0.9 -- 1.1 h
each, four at a time. Rows (git-ignored): epireplay2.parquet, epireplay2_screens.parquet and the
jobs' epireplay2_jobs/; epireplay.py regenerates them from epireplay_designs.json.

**A price per arm** (epistemetry 3a3b9a3, the feedback above built; owner 2026-10-16). 3a3b9a3
asks each live arm's stop rule of a `StopOf` at each decision past the floor; its calls are one
row per arm, its fields keyword-only, its orders carry their own streams, one truth is computed
once, and a run records its decisions. epireplay.py moved to it (pinned 3a3b9a3; `Witness` gone,
the guards read `Run.decisions`; seqscreen's per-order generator is its order's stream, copied at
each decision) and prices each arm by its own stake, rate / FUTURE over that arm's Lens A at stake
at its means so far, as seqscreen prices each variant on its own (`OwnStake`). The shared price
(`SharedStake`, the live arms' stakes averaged) stays beside it: not dominated (below). On one arm
the two are one: the 2,600 pairwise step-1 runs at 3a3b9a3 reproduce the second replay's exactly
(blocks, calls, GPU-seconds), so the pairwise table above stands, and the shared price reproduces
its 500 many-arm runs. Many-arm, step 1:

    many-arm               realized loss (1e-6) at 2e-6, 6.2e-6, 1.9e-5   wrong  asks  cost
    seqscreen x arms               132.8   174.3   521.3                    67   279  0.424
    per-arm, own price              66.6   203.3   522.4                    37   317  0.463
    per-arm, shared price           66.9   204.5   526.1                    36   315  0.464

**Decision: keep seqscreen, many-arm too.** The own price fails (1) at the middle and top rates
(203.3 against 174.3; 522.4 against 521.3) and (3), 317 asks against 279, with 37 wrong calls
against 67. It closes almost none of the gap: its loss is below the shared price's at every rate
(by 0.3, 1.2 and 3.7e-6) at a hair less cost, for one wrong call and two asks more. Each arm now
runs as its pairwise screen does: x2 in 2/p64 observes the blocks x2/p64 does in 86 of 100 orders
(50 at the shared price), cheapx2 in 2/cheap in 94 (64), the rest the Monte Carlo draws the arms
share, and 2/cheap's screen is its pairwise step-1 screens summed to the medians' digit. So the
many-arm gap is the pairwise one: 2/p64 (59.5 against 46.8 at the middle rate) and 2/p256 (45.8
against 27.2) lose where x2/p64, GS/p256 and pPg/p256 lose pairwise. A per-arm price was not the
gap; items 2 and 3 above (a linear-loss target over the bracket, a price on asks) remain what
would have to change. The shared price is better on wrong calls (36 against 37) and asks (315
against 317) and worse on every loss and on cost, so neither dominates and both stay selectable,
though the margins are a run or two in 500.

Guards, each failing under its fault (`guards none` runs all, `guards <fault>` the one it breaks):
- migration: the pairwise step-1 runs at the own price reproduce the second replay's, 2,600 of
  2,600; the shared price its many-arm runs, 500 of 500; and every price the own price gave a live
  arm, read from the runs' decisions, is that arm's own stake's: 4,198 of 4,198 where the arm's
  and the next arm's differ (158 of 158 in 2/p64's first 10 orders, run now). Fault (each arm
  priced by the next arm's stake): the pairwise runs still 2,600 (one arm), 0 of 167 prices on
  2/p64, FAIL.
- wrapper: 2,608 of 2,608. Fault (the wrapper's floor 14): 725 of 2,608, FAIL.
- degenerate: the candidate stops at 15 with "keep" in 100 of 100 orders, both settings. Fault
  (the floor 1): 0 of 100, FAIL.
- per-arm drop, from `Run.decisions`: the vetoed arm in 100 of 100 orders; an arm paid for fewer
  blocks than its screen ran in 2 of 4/base's first 10 orders and in 633 of the run's 1,000
  many-arm runs, each arm dropped on value (895) live at no later decision, its blocks those seen
  when dropped, charged so. Fault (`KeepAll`, no veto): 0 of 100 and 0 of 10, FAIL.
- same orders: all 7,400 runs. Fault (one run handed a reshuffled order): that run flagged, FAIL.

Feedback on 3a3b9a3:
- `Decision` records the units seen and the live arms but not each arm's prefix: the guards
  rebuild a dropped arm's from the decision that left it out. Each arm's units would make it
  direct.
- The rule each arm was judged by is not recorded: to check the price, the migration guard
  rebuilds each decision's `Seen` and asks the `StopOf` again. A decision could carry its rules'
  readings, the ledger form the stopping rules already have.
- A Run does not carry its order; runs are matched to the orders handed by position.
- seqscreen replays from the first block with a fixed generator, so it takes a copy of its order's
  stream at each decision; that works because the harness never draws from the stream itself,
  which the docs could promise.

Compute: here, single-threaded, at most 4 processes: the run 4.1 h on 4 workers (69 jobs, the
pairwise step-1 migration most of it), the shared price's 5 jobs 2.0 h, the guard runs 0.1 -- 0.8
h each. Rows (git-ignored): epireplay3.parquet, epireplay3_screens.parquet and epireplay3_jobs/.

### Pruning by epistemetry: the realized loss as the target (owner 2026-10-16; BACKLOG, "Pruning by epistemetry's decision layer")

The third replay (above, "A price per arm") kept seqscreen and named what would still have to
change: a linear-loss target over the rate bracket, priced in the owner's units, and a price on
asks. GeoffChurch/epistemetry built the first, 674dba4's `AdoptKeepAskByStake(rates, weights)`,
on 046675b's API (`Truth.of`, a policy's `Judgement`, `Decision`'s readings, `Run.order`), and
the owner decided the second (2026-10-16): asks are queued and cost 0; an ask records the
question, leaves the default as it is, and is answered later, at the owner's true rate, from the
screen's means. The replay ran again before the consolidated pass's pruning step, under the same
rule, pinned at 674dba4 through an ephemeral overlay as before.

**The rule, as fixed** (the first replay's, amended): adopt epistemetry for the consolidated
pass's pruning only if (1) its median realized loss is below seqscreen's at every rate tested
(per rate, the sum over the designs of each one's median over orders), (2) it makes no more wrong
calls in all and (3) no more asks in all; pairwise in both veto settings, and a many-arm screen
counted against the pairwise screens it replaces. Realized loss, wrong calls (AsksApart) and
asks as before; an ask is answered at the judged rate from the screen's own means, as the owner
decided. Asks costing 0 changes how the new target weighs asking, not (3).

**The target.** At rate r a wrong call on an arm loses |gain - r x extra|; at each rate point
the call is the one the posterior means favour, ASK where the points disagree, and the risk is
that call's expected loss, summed over arms and averaged over the points. Its value is in Lens
A, so the stop's price is sum_r w_r r / FUTURE per GPU-second, one price for every arm with no
stake (`OwnStake` and `SharedStake` stay for the 0-1 target): over seqscreen's five geometric
rates, equally weighted, PRICE_RATE / FUTURE = 1.01e-8, the price seqscreen's stop uses. In the
layer's terms it is seqscreen's own stop: the same loss over the same points at the same price.
What still differs is how the value is drawn: the finite-population bootstrap of the design's
blocks (seqscreen: a Polya urn's posterior mean after k more blocks), the extra GPU-seconds
uncertain jointly with the gain (seqscreen plugs in their mean), and the Monte Carlo (4,000
outer and 200 inner draws, against 4,000 paths).

**The migration.** epireplay.py moved to 674dba4 (502dec65) with no reader of 3a3b9a3's API. The
truth is one per design and veto setting, seqscreen's call over the bracket on all the blocks
(`Truth.of` the wrapped seqscreen), handed to every replay; each row's order is the one its run
carries (`Run.order`); each decision keeps each judged arm's rule and reading, which the guards
read instead of rebuilding a `Seen`. The target is a Strategy (`ZeroOne`, the third replay's
AdoptKeepAsk at the own or the shared stake; `ByStake`), its rates, weights and price in each
row's identity. At 674dba4 the third replay's runs reproduce exactly (blocks, calls,
GPU-seconds, and wrong calls, asks and truth): 2,600 of 2,600 pairwise step-1 runs and 500 of 500
many-arm runs at each price. Its tables stand; they are the 0-1 rows below.

**The test.** On the frozen designs and their 100 orders, every policy on each order, all judged
against the one truth: seqscreen unchanged, as a Policy; the 0-1 rows (b), reproduced; (c)
ValueOfMoreUnits with `AdoptKeepAskByStake` over BRACKET, equally weighted, matched to seqscreen
as in the third replay (FinitePopulation, floor 15, the ratio cost model, step 1, both veto
settings pairwise, none many-arm), priced at PRICE_RATE / FUTURE, pairwise and many-arm (one
screen over a design's variants, each arm stopping on its own value, against seqscreen once per
variant, summed); (d) the same target over nine rates evenly spaced in log rate across RATES,
equally weighted, priced at their mean over FUTURE (9.66e-9, 4.5% below), pairwise. Every
policy's and target's own call on the whole design is the truth (131 of 131 jobs): no target
changes the test.

Per design, without the veto (C1m4/p64's veto line beside; the veto changes nothing else); blocks
and cost are medians over the 100 orders (the longest run in parentheses), wrong calls and asks
counts in them; many-arm in block-arm runs, seqscreen once per variant summed, its whole calls
per variant (A adopt, k keep, ? ask):

    design         all    seqscreen             0-1, own price           realized loss
                          blocks cost  wr  ask  blocks    cost  wr  ask  blocks     cost  wr  ask
    p64/base       adopt  15     0.33   0    0  15 (15)   0.33   0    0  15 (15)    0.33   0    0
    x2/base        adopt  15     0.41   0    0  15 (15)   0.41   0    0  15 (15)    0.41   0    0
    both/base      adopt  15     0.34   0    0  15 (24)   0.34   0    0  15 (18)    0.34   0    0
    both/p64       keep   15     0.34   0    0  15 (43)   0.34   0    7  15 (35)    0.34   0    2
    r2/base        ask    20     0.43   2   98  15 (30)   0.35   2   98  17.5 (27)  0.40   2   98
    x2/p64         ask    15     0.44  35   65  15 (34)   0.46  19   81  15 (20)    0.43  36   64
    pPg/p256       keep   15     0.34   0   30  15 (36)   0.35   0   33  15 (22)    0.34   0   33
    GS/p256        ask    15.5   0.36  28   72  17 (44)   0.38  16   84  16 (26)    0.36  28   72
    cheapx2/cheap  adopt  15     0.43   1   15  15 (32)   0.45   0   15  15 (26)    0.43   1   15
    p256/cheap     adopt  15     0.42   0    0  15 (15)   0.42   0    0  15 (15)    0.42   0    0
    g005/g01       adopt  15     0.34   0    0  15 (15)   0.34   0    0  15 (15)    0.34   0    0
    cheap/g01      adopt  15     0.43   0    0  15 (30)   0.47   0    0  15 (24)    0.43   0    0
    C1m4/p64       keep   12     1.00   0    0  12 (12)   1.00   0    0  12 (12)    1.00   0    0
      with the veto       5      0.48   0    0  5 (12)    0.48   0    0  5 (12)     0.48   0    0
    4/base         ?AAA   64     0.46   3   97  61 (74)   0.44   2   98  62.5 (75)  0.44   3   97
    2/p64          ?k     33     0.44  35   65  40.5 (68) 0.55  19   86  31.5 (45)  0.43  37   65
    2/p256         k?     31     0.35  28  102  36 (76)   0.40  16  117  32 (41)    0.35  28  105
    2/cheap        AA     30     0.42   1   15  30 (47)   0.43   0   15  30 (41)    0.42   1   15
    2/g01          AA     30     0.45   0    0  30 (44)   0.49   0    1  30 (40)    0.45   0    0

    the rule        realized loss (1e-6) at 2e-6, 6.2e-6, 1.9e-5   wrong  asks  cost
    no veto  seqscreen                57.3   178.2   538.9          66   280  0.431
             0-1, own price           60.2   183.0   532.1          37   318  0.433
             realized loss            57.0   174.7   533.7          67   284  0.428
             realized loss, 9 rates   57.9   177.0   538.6          75   273  0.429
    veto     seqscreen                53.5   166.3   502.0          66   280  0.391
             0-1, own price           56.3   171.0   495.2          37   318  0.393
             realized loss            53.1   162.7   496.8          67   284  0.388
             realized loss, 9 rates   54.0   165.0   501.7          75   273  0.389
    many-arm seqscreen x arms        132.8   174.3   521.3          67   279  0.424
             0-1, own price           66.6   203.3   522.4          37   317  0.463
             0-1, shared price        66.9   204.5   526.1          36   315  0.464
             realized loss           132.4   170.1   504.0          69   282  0.419

**Decision: keep seqscreen, pairwise and many-arm.** The realized loss fails (2) and (3), not
(3) alone:
- Pairwise, in both settings: (1) holds, its loss below seqscreen's at every rate (no veto 57.0,
  174.7 and 533.7 against 57.3, 178.2 and 538.9; veto 53.1, 162.7 and 496.8 against 53.5, 166.3
  and 502.0); (2) fails, 67 wrong calls against 66; (3) fails, 284 asks against 280.
- Many-arm: (1) holds (132.4, 170.1 and 504.0 against 132.8, 174.3 and 521.3); (2) fails, 69
  wrong calls against 67; (3) fails, 282 asks against 279.
So the owner's decision that asks cost 0 does not decide it: with (3) set aside it would still
fail (2), by one wrong call in 2,600 pairwise runs and two in 500 many-arm runs.

Where they differ:
- **Seqscreen's stop, nearly run for run.** In each setting its pairwise runs stop where
  seqscreen's do in 1,152 of 1,300 and make its call in 1,286; it stops earlier in 113 (the
  near-ties and the clear keeps: r2/base 47, x2/p64 26, both/p64 17, pPg/p256 15) and later in 35
  (GS/p256 18). Many-arm, each arm observes the blocks its pairwise seqscreen screen does in 47 to
  100 of 100 orders (r2 in 4/base 47, x2 in 2/p64 72, the rest 74 or more) and makes its call in
  92 to 100. The 0-1 target's gap on the three near-ties at the bracket's bottom and middle is
  gone: GS/p256 5.4 and 15.5 against seqscreen's 5.1 and 15.4 (0-1: 7.3 and 22.8), x2/p64 5.1
  and 15.8 against 5.2 and 16.1, pPg/p256 3.5 and 10.4 against 3.5 and 10.5; r2/base, where the
  0-1 target settled the ask at the floor, sits between (7.8 and 24.7, seqscreen 8.4 and 28.0).
- **Its counts are seqscreen's within a few, on the near-ties.** Its 67 wrong calls are
  seqscreen's kind, definite calls where the whole design asks (x2/p64 36 against 35, GS/p256 28,
  r2/base 2, cheapx2/cheap 1); the 0-1 target's trade of wrong calls for asks (37 and 318) went
  with its 0-1 risk. Its four more asks are keeps it turned into asks where it stopped earlier on
  a clear keep (both/p64 2 against 0, pPg/p256 33 against 30), less one on x2/p64 (64 against
  65): an ask is free to the target, so on a call that turns on the bracket's bottom it stops and
  queues the question, as its builder said it would. Many-arm the same: 2/p64 37 wrong calls
  against 35, 2/p256 105 asks against 102, 4/base, 2/cheap and 2/g01 equal.
- **Cost.** 0.428 of the arms' GPU-seconds against 0.431 pairwise, 0.419 against 0.424 many-arm;
  the loss margins are 0.3, 3.5 and 5.2e-6 pairwise and 0.4, 4.2 and 17.3e-6 many-arm.
- **The call at a stop.** The posterior means' call at each point, from 4,000 draws, is the
  posterior mean's over the bracket at every stop of every policy.

**The sensitivity** (d). Nine rates instead of five, priced at their mean, move the calls in 15
runs a setting (x2/p64 7, pPg/p256 3, cheapx2/cheap 3, both/p64 1, GS/p256 1) and the blocks in
172 of 1,300 (r2/base 67, GS/p256 42, x2/p64 25, pPg/p256 15, cheapx2/cheap 12, both/p64 8). The
decision does not move, but which criteria fail does: its loss is above seqscreen's at the
bracket's bottom (57.9 against 57.3; veto 54.0 against 53.5) and below at the middle and top,
with 75 wrong calls (x2/p64 43) and 273 asks (x2/p64 57), so it fails (1) and (2) and holds (3).
The five-point counts are therefore not robust: moving the points within the same bracket trades
8 wrong calls for 11 asks on the near-ties, more than the margins the rule decides on.

**What remains** (inferred). The target now carries seqscreen's loss and price, and the two
screens agree but for a few near-tie runs; what separates them under the rule is counts of 1 to 4
in 2,600 pairwise runs (2 to 3 in 500 many-arm), smaller than the sensitivity's own movement.
Neither a target nor a price is left to build for this comparison: whether counts this close
should decide, now that asks are queued at no cost, is the owner's call on the rule. Until then
seqscreen stays, and adopting the layer would buy only the per-arm screen's bookkeeping, at the
price of a dependency.

Guards, each failing under its fault (`guards none` runs all, `guards <fault>` the one it
breaks, `guard <guard>` one without its fault):
- migration: the third replay's runs at 674dba4, 2,600 of 2,600 pairwise and 500 of 500
  many-arm at each price; x2/p64's first 20 orders and 2/p64's first 10 run now, 30 of 30; every
  own price in the record the arm's own, 4,198 of 4,198. Fault (the 0-1 target priced as the
  realized loss is, its stake left out): x2/p64 2 of 20 (3 calls changed), 2/p64 0 of 10 (3
  calls changed), 0 of 371 prices the arm's own, FAIL.
- target as used: at all 6,514 judged decisions of (c)'s runs (4,908 pairwise, 1,606 many-arm;
  7,271 arm judgements) the recorded price is PRICE_RATE / FUTURE with no margin, and from the
  posterior's draws the risk recomputed outside epistemetry (each point's posterior-mean call,
  E|gain - r x extra| where it is wrong, averaged over the five rates in Lens A per GPU-second,
  summed over arms) is the target's own within 3.7e-15 relative, its call too; and the same on 54
  decisions run now. Faults: the rates in Lens A per GPU-hour, 0 of 15 prices and 1 of 15 risks,
  FAIL; the weights not normalized, refused by epistemetry at construction, FAIL.
- wrapper: 2,608 of 2,608. Fault (the wrapper's floor 14): 725 of 2,608, FAIL.
- degenerate: every epistemetry policy (the 0-1 target, the realized loss over five and nine
  rates), both settings, stops at 15 with "keep" in 100 of 100 orders. Fault (the floor 1): 0 of
  100, every run at 1 block, FAIL.
- per-arm drop, from `Run.decisions`, both targets: the vetoed arm in 100 of 100 orders; an arm
  paid for fewer blocks than its screen ran in 278, 355 and 226 of the 500 many-arm runs at the
  own price, the shared price and the realized loss (391, 504 and 355 arms dropped on value), each
  live at no later decision, its prefix kept, charged so, and every judged arm's verdict,
  re-derived from its recorded rule and reading, its take. Fault (`KeepAll`, no veto): 0 of 100
  and 0 of 10, the verdicts not the takes, FAIL.
- same orders and one truth: all 13,100 runs on their design's orders (the order each run
  carries); every policy's and target's own truth the one truth, 131 of 131 jobs; every run scored
  against it. Fault (one run handed a reshuffled order): that run flagged, FAIL.

**Feedback for epistemetry** (for its author):
- `AdoptKeepAskByStake` does what it says, measured: at 6,514 decisions its risk is the realized
  loss recomputed outside the layer to rounding, and weights not summing to 1 are refused.
- The target does not carry its price: sum_r w_r r / FUTURE is the caller's to pair with it, so a
  unit slip moves the two together unseen (rates per GPU-hour: the calls change and the price is
  3,600 times; only an outside recomputation caught it). A price for a given horizon from the
  target itself (`price(future=...)`), or the stop rule built from it, would keep them one.
- Each point's call is the mean of the `draws` Monte Carlo draws, not the belief's exact mean
  (FinitePopulation's is the seen units' mean, its `moments`): on a near-tie it carries Monte
  Carlo noise. Here it never differed from the exact mean's call at a stop.
- `Truth` is not comparable (`eq=False`), so checking that every policy's and target's own
  whole-design calls are one truth compares `calls` and `drop` by hand; every replay with more
  than one target needs that check.
- `StopRule.record()` has no inverse: to re-derive a stored run's verdicts the guards rebuild a
  ValueOfInformation from its record. A `from_record` would close the ledger for runs kept on
  disk.
- `Run.order` and `Decision.arm_units`, rules and readings made the same-orders and per-arm
  guards direct: no wrapper, no rebuilt `Seen`; the third replay's four notes are resolved.

Compute: here, single-threaded, at most 4 processes: the jobs 52.2 worker-hours over 14.7 h (the
0-1 rows 23.5, the realized loss 15.1, the sensitivity 13.4, seqscreen 0.2), the guards minutes
each but for the migration fault (0.5 h) and the degenerate guard and its fault (1.5 and 1.9 h).
Rows (git-ignored): epireplay4.parquet, epireplay4_screens.parquet and epireplay4_jobs/ (with the
wrapper fault's floor-14 jobs); the runs' logs epireplay4_*.log.

### Exits as leaks: the region model on the complex (5810@major)

BACKLOG 4, decision (4) (the brief: .superpowers/sdd/2026-10-13-region-leaks; the owner
2026-10-12 -- 13: "we'd still only leak at the outer boundary, so all current must exit ... some of
the outer boundary might not leak at all and might have 0 conductance (like a railroad)"; "we can
adjust boundary leakage, uniform leakage, and also building-boundary leakage"). The region model's
exits compiled into conductances on the region's fixed mesh, and tested on 5810@major to give the
owner the evidence to choose them. Code: gap_exits.py (the three Strategies, their name and
digest), gap_clear.py (the leak on the fixed pattern: Graph's seeds and diagonal, Model.leak_site,
ComplexMetric.exits and the name's grammar), gap_maps.py (the leaks' drainage), regions.py (the
region's OSM features; each building's member kblock), digests.py (a code's digest: CODE's and the
exits'), common.member_blocks, region_leaks.py (the study), gap_clear_check.py (seven new guards),
cluster.py (`blocks` takes `--node` and `--items`). Rows: region_rows/ (the runs, baselines, images
and maps git-ignored; summary_*.csv committed).

**The model** (gap_exits). Three Strategies a metric holds (ComplexMetric.exits), each resolved
once from the metric's name (gap_clear.complex_of, metrics.metric_of) and asked nothing downstream.
- Leaks (`Leak`): a conductance to ground per cell, on the diagonal of every unknown row
  (gap_clear.Graph.values: the street's G plus the leak) and among the reach's seeds (a leaking
  cell is grounded: Graph.reach), on the cells that are unknowns at baseline (live and not ground
  with every footprint standing: Model.leak_site) and nowhere else, the same at every state. So the
  sparsity pattern, the plan and the factorization are the per-block metric's (the leak changes
  values on the diagonal only), P stays convex in the clearing, the reach is the street's (a leak
  lies only where the street already reaches), B's enclosed homes stay enclosed, and a cleared
  building's interior and zone conduct but do not leak. The gradient is the adjoint's unchanged (a
  leak does not move with the clearing; its solve is the state's own system).
- Uniform (`UniformLeak(lam)`, `U<lam>`): l_c = sigma |c| / lam^2, |c| the cell's area (m^2) and
  sigma the sheet conductance (pi / 2, the FlowSpec's: a passage conducts sigma |e| / d_e). Cell by
  cell this is the screened Laplacian -div(sigma grad u) + (sigma / lam^2) u = f, whose point
  source in open ground decays as Q K0(r / lam) / (2 pi sigma). For a walker (u the escape time,
  the leak a constant rate of arriving anywhere), a trip's straight-line length r has the density
  (r / lam^2) K0(r / lam): median 1.26 lam, mean (pi / 2) lam, root mean square 2 lam, 10th -- 90th
  percentile 0.34 -- 3.21 lam; lam 250, 500 and 1,000 m are median trips of 314 m, 629 m and 1.26
  km.
- Building-boundary (`HomeLeak(lam)`, `H<lam>`): destinations at buildings by area population, l_c
  = kappa x the cell's population injection at baseline (System.Pc @ pop: the demand points of
  gap_flow.demand_on, weighted as the injection is), kappa making the total the uniform family's at
  the same lam, sigma sum |c| / lam^2 over the cells live at baseline. A home's own cells are not
  excluded, short trips not special-cased (owner 2026-10-13).
- Barriers (`OsmBarriers(line_m, cross_m)`, `B<line>c<cross>`): the region's railways (railway
  rail, light_rail, narrow_gauge) and water (waterway river, canal, stream, drain, ditch, not in a
  tunnel or culvert, as strips `line_m` wide about their lines, flat-ended; natural=water, landuse
  reservoir or basin and waterway riverbank areas as they are) cut out of the region's polygon:
  obstacles never cleared, holding no demand, and the region's edge no exit where one lies on it
  (no face of the mesh lies on the street line within it). Open only at crossings: a way of the
  walking network (any highway) on a bridge or in a tunnel over a barrier, or through a railway
  crossing node (railway crossing or level_crossing: the way's piece within `line_m` of the node),
  each a gap `cross_m` wide about the way's line. The mesh is built on the cut polygon
  (ComplexMetric.fixed and .model cut the block), so the mesh's cache key (the block's boundary)
  and the run's name both carry the barriers.
- Street boost (`StreetBoost(factor)`, `S<factor>`), a check only: sigma x factor on the triangles
  whose centroid lies within 3 m of an OSM way (half a 6 m residential street), a passage by the
  harmonic mean of its two triangles' factors, a street face by its triangle's.
- The boundary family has no free parameter here: the region's ring is the street (Dirichlet:
  today's street faces) where it is a road, none where a barrier lies on it.

**The leak's scale, checked** (the `leak-scale` guard): open ground 800 m square, its edge the
street, one 2 m home at its centre, the uniform leak at lam 60 m, on the runtime's graded mesh
(cells of 2.5 -- 30 m^2 between 10 and 200 m out): the home's potential over K0(r / lam) is flat
within 2.6% over twelve distance bins and its level 0.2% from Q / (2 pi sigma), the formula's, the
source's own current Q. With the same total leak on every cell alike (the fault `per-cell`: a leak
per cell, not per area) the ratio drifts 4.3x over the bins and its level is 34% off.

**The barriers on 5810@major** (regions.osm_of: the Geofabrik extract's ways, nodes and areas
meeting the face, their tags read once where the region is built;
region_rows/summary_barriers.csv). The railway (two tracks, railway=rail, 3.7 -- 7.7 m apart, a
median 4.25: an 8 m strip each merges them into one corridor about 12 m wide) runs 5.84 km through
the face from its west edge to its east edge, south of the N1's vacant land and north of 5810; the
water is 18 ponds and reservoirs (natural=water, 0.20 km^2 in the face), 1.26 km of seasonal
streams (their 104 m of culverts open) and 0.38 km of ditches. Strips 0.088 km^2 and areas 0.201
km^2 in the face: 0.289 km^2 cut, the region 6.45 km^2 of 6.74, 5 footprints touching a barrier.
Without crossings the railway parts the face in two (3.71 and 2.74 km^2). The crossings: the
region's one crossing of the railway inside it, an unpaved path over both tracks (two
railway=crossing nodes, the path's 16 m about each) at 5810's north-east corner; and three bridges
at its ends, the boundary roads' own (the R304 and the M167) and a sidewalk on one, at the region's
edge: crossing there is reaching the street. No highway crosses a stream inside the face (they pass
in culverts). The crossing's width, 3 m: a footway's (the path is unpaved; a level crossing's road
would be 6 m), the gap 484 m^2 in all.

**The name.** The per-block metric (no barriers, no leak, no boost: the block's whole street ring
the exit) has no suffix, so its name and digest are as before,
`cx+shy0.5+s0.125r1b1+R0.5g4x2q4+v34d40e8a` with its 28 constants, and every stored per-block row
keeps its name; its scores are bit for bit the previous code's (19570's baseline lenses and escape
times, a random clearing's lenses and escape times, SIMP's value and gradient, the tension). A
configuration with leaks, barriers or a boost names its parts in a suffix before the search's
`@e<eps>`, `+x[B<line>c<cross>][U<lam>|H<lam>][S<factor>]v<exits code>`, the exits code (a03a092b)
the digest of gap_exits' own 8 constants (its tag sets, the crossing node's tolerance, the boost's
3 m), walked as gap_clear walks its own (digests.constants_of) and not in CODE's walk: editing the
leak code renames only the leaky metrics. complex_of parses a name back (gap_exits.exits_of) and
refuses another exits code's. The reference is `...+v34d40e8a+xB8c3va03a092b`, U500's
`...+xB8c3U500va03a092b`, the cheap preset's runs `...@e0.05`; the boundary-only configuration
without barriers is the per-block metric's own name, on the region's block.

**The runs.** Ten configurations: `edge` (the boundary only, no barriers: the raster run's setting,
and the per-block metric's own name on the region's block), `bar` (the boundary with the barriers:
the reference), `U250`, `U500`, `U1000` (bar with the uniform leak at lam 250, 500, 1,000 m),
`H250`, `H500`, `H1000` (bar with the building-boundary leak), `S3` and `S3U500` (the street boost
3x at bar and at U500, a check). Each run with the cheap preset (`@e0.05`) and SIMP's default at
area D 0.05, free first, optimizing P under the area population (region_leaks `run`, on the
cluster's V100s of tesla2, one card each, at most four at once: runs rl-pilot, rl-runs, rl-runs2).
The pilot (the cheap preset under `bar`) took 1.01 GPU-h, 25 minutes of it the node building the
barrier mesh (16.8M triangles; 16 minutes here); a run once the node holds the mesh takes
0.44 -- 0.71 GPU-h (SIMP 1,358 -- 2,106 s of optimizing, the cheap preset 1,559 -- 1,838 s, free
first about 12 minutes of host work in each). From that timing the 20 runs, with the no-buildings
priors the maps need (below) and the device guards, would pass the 12 GPU-h, so, as ruled, **the
street boost's two configurations were cut** (their four runs and two priors, about 2.1 GPU-h); the
250 m lambda of both families fit and ran. 16 runs, 8 priors and the device guards took 10.0 GPU-h:
the 1.96 left would hold one of the two boost configurations (about 1.05 GPU-h each), not both. In
every run free first frees the 36 homes walled in at baseline (the barriers wall none in) with the
same 29 buildings, D 0.0013, and no run leaves a home walled in. Host memory peaks at 22 -- 53 GB,
the card at 8.8 -- 9.6 GB of 32. The clearings were scored here on cores (`score`: P, P_{1/2} and
J_2, each lens against its configuration's own baseline, so P_{1/2}'s weights are each
configuration's own baseline escape times) and agree with the runs' own L_P (on the cards) within
4e-11. The no-buildings prior (every footprint cleared: every cell an unknown, 16M on the region)
does not factor on a core (SuperLU's 32-bit indices: SUPERLU_MALLOC fails), so it ran on a card too
(`run ... prior`, 4 minutes each, painted at 2 m).

**The baselines** (region_rows/summary_baseline.csv, summary_escape.csv; P_0 in the area
population's units; the leaks' share of the current drained at baseline, the rest leaving by the
boundary; the median escape time of all homes, and of the formal and deep homes of question 2):

    config   P_0        P_1/2,0    J_2,0      leaks drain   median escape time: all (formal, deep)
    edge     9.44e+06   1.71e+07   1.08e+10   0             808 (1,307, 366)
    bar      1.76e+07   3.53e+07   3.96e+10   0             1,557 (2,252, 842)
    U250     4.33e+06   7.77e+06   2.22e+09   0.63          394 (577, 227)
    U500     9.68e+06   1.88e+07   1.17e+10   0.37          869 (1,264, 483)
    U1000    1.45e+07   2.89e+07   2.69e+10   0.14          1,294 (1,873, 704)
    H250     8.15e+05   9.80e+05   6.01e+07   0.76          81 (84, 76)
    H500     2.60e+06   3.74e+06   6.90e+08   0.61          268 (300, 210)
    H1000    6.84e+06   1.17e+07   5.39e+09   0.40          667 (841, 432)
    S3       1.28e+07   2.40e+07   1.99e+10   0             1,185 (1,594, 674)
    S3U500   8.01e+06   1.49e+07   7.77e+09   0.32          746 (1,028, 428)

The barriers alone nearly double P_0 (+86%) and raise J_2 3.7x: the railway closes the north, the
vacant land to the N1 that 5810's west part drained into, but for the one path. The uniform leak at
1,000 m drains 14% of the current and P_0 is 17% below bar's; at 250 m 63% and P_0 is a quarter of
it. The building-boundary leak at the same total drains more (40 -- 76%), its destinations where
the homes are.

**Self-absorption** (region_rows/summary_absorb.csv; 300 homes with demand drawn at random, each
one's unit injection solved alone at baseline): `own building` the share of a home's current that
leaks into its own building's destination (the self-trips), `own cells` all the leak on the cells
its injection lands on, `anywhere` all it leaks rather than leaving by the boundary; the median,
the population-weighted mean, the 90th percentile and the largest over the 300:

    config   own building: median, pop-weighted, p90, max   own cells (median)   anywhere (median)
    H250     0.0087, 0.091, 0.029, 0.21                     0.0097               0.95
    H500     0.0026, 0.029, 0.0097, 0.067                   0.0029               0.78
    H1000    0.00073, 0.0081, 0.0027, 0.018                 0.00083              0.49
    U250     --                                             0.00046              0.73
    U500     --                                             0.00014              0.41
    U1000    --                                             0.00004              0.16

For the typical home it is small at every lam (0.9% at 250 m), but the population-weighted mean is
not at 250 m (9%): a building's own destination is in proportion to its area, so the largest
buildings absorb up to 21% of their own current. It falls about as 1 / lam^2 and is small at 1,000
m (0.8% population-weighted, 1.8% at most). The uniform leak's absorption on a home's own cells is
a few in 10,000 at every lam.

**The lenses** (region_rows/summary_lenses.csv; Lens A of each configuration's own model: L_P the
optimized one; t the optimizer's time from the model built, free first in it; wall the task's, the
first task on the node building its mesh too):

    config   preset  L_P     L_P1/2  L_J2    cleared   t (s)   wall (s)
    edge     simp    0.2825  0.3088  0.2973   720      1,474   3,380 (the mesh built)
    edge     cheap   0.2643  0.2829  0.2737   523      1,618   1,839
    bar      simp    0.3105  0.3177  0.3120   753      1,358   1,583
    bar      cheap   0.2891  0.2935  0.2887   537      1,648   3,641 (the pilot: the mesh built)
    U250     simp    0.1427  0.1416  0.1338   656      1,624   1,845
    U250     cheap   0.1352  0.1342  0.1272   588      1,689   1,912
    U500     simp    0.2207  0.2242  0.2177   730      1,463   1,691
    U500     cheap   0.2052  0.2041  0.1987   541      1,625   1,843
    U1000    simp    0.2787  0.2843  0.2782   739      1,364   1,586
    U1000    cheap   0.2590  0.2623  0.2573   537      1,838   2,064
    H250     simp    0.0441  0.0370  0.0333   748      1,379   1,597
    H250     cheap   0.0426  0.0363  0.0330   554      1,559   1,782
    H500     simp    0.0843  0.0741  0.0688   653      2,106   2,538
    H500     cheap   0.0804  0.0704  0.0657   530      1,625   1,845
    H1000    simp    0.1573  0.1489  0.1431   656      1,703   1,917
    H1000    cheap   0.1470  0.1393  0.1343   518      1,720   1,944

Lens A falls with the leak: the shorter the trips, the less a clearing changes them (the home leak
at 250 m leaves 4% to gain). Under `edge` the raster run's Lens A was 0.2714 (cheap) and 0.3014
(SIMP), J_2 on the composite raster; on the complex its clearings read L_J2 0.2702 and 0.2966 (the
2b cross-check's 0.270 and 0.297), against the complex's own presets' 0.2737 and 0.2973, which
optimize P (region_rows/summary_raster.csv): the complex's clearings lead on the raster's own
objective too, and the raster's leave 22 and 31 homes walled in on the complex (no free first on
the complex's geometry).

**Question 1, tendrils or interconnection** (region_leaks.loops; region_rows/summary_loops.csv; the
maps below). The index: the corridor network at a threshold T is the open pixels (1 m, gap_maps'
current painted from the triangles' fluxes) whose flow is at least T, the exits the pixels within 2
m of the street line; its loops are its cycle rank with the exits contracted to one node, b0 -
chi(Y / X) with chi the Euler characteristic of the pixels' 4-connectivity complex and chi(Y / X) =
chi(Y) - chi(X) + 1, holes under 25 m^2 filled first (a pixel below T inside a lane). It measures
the question: a tree of tendrils reaching in from the edge has none, each corridor that joins two
points of the edge or closes a ring inside adds one, whatever its length. Checked where the answer
is known (the `loop-index` guard, blocks whose open ground is only 6 m streets, the flow painted
and every open pixel in the network): a grid of 3 x 3 blocks inside an exit ring gives 8, a ladder
of 5 bars off a street on one side with a back lane 5, the same bars reaching the far wall (a comb
of dead ends) 0; not contracting the exits gives the grid 9 (the fault). T is a quantile of the
baseline's flow over the open pixels, the same for every state of a configuration; the induced
corridors are the pixels whose flow rose by at least T, so their loops are the interconnection the
clearing adds. Loops of the induced corridors in the region (per km^2 at 0.9), SIMP's and the cheap
preset's:

    config   q 0.8     q 0.9 (per km^2)       q 0.95
    edge     10, 13    4, 4 (0.6, 0.6)        4, 5
    bar      39, 23    10, 11 (1.6, 1.7)      2, 5
    U250     91, 90    29, 23 (4.5, 3.6)      9, 10
    U500     82, 66    18, 12 (2.8, 1.9)      4, 6
    U1000    56, 50    10, 14 (1.6, 2.2)      3, 6
    H250     53, 61    24, 34 (3.7, 5.3)      14, 11
    H500     69, 70    26, 22 (4.0, 3.4)      7, 10
    H1000    77, 77    16, 18 (2.5, 2.8)      4, 7

(the raster's picks under `edge`: 3 and 11 at 0.9). With the edge as the only exit (edge) the
corridors a clearing induces are nearly a tree: 4 loops in 6.7 km^2 at q 0.9, about 0.06 km^2 of
new main lanes reaching up from the south road into the formal grid, the volume-to-point answer.
The barriers alone take that to 10 -- 11 (the drainage into the north closed, the grid's lanes join
east -- west too [inferred from the maps]); internal leaks interconnect more at every threshold but
the highest, the more the shorter the trips: uniform 10 -- 29, building-boundary 16 -- 34 at q 0.9;
at q 0.8 edge's 10 -- 13, bar's 23 -- 39, the leaks' 50 -- 91. The whole network's loops (the
baseline's main lanes and the clearing's together) move less cleanly, a clearing also merging lanes
around the buildings it removes (summary_loops.csv: bar's falls from 801 to 678 under SIMP, U500's
rises from 770 to 900). The maps (region_rows/maps/<config>_<preset>.png, 16: the flow after
against the no-buildings prior as the atlas's `rel` layer, the change in flow from the baseline,
and that change on 5810) show it: under edge and bar the opened lanes are north -- south strands
through the formal grid's blocks to the south road, the flow drawn away from 5810's informal west
and, under edge, from the vacant land north of the railway that it drained into; under the home
leak the flow rises across the whole settlement, the informal west too, in a web rather than
strands.

**Question 2, formal homes** (region_rows/summary_escape.csv). "On a formal street": a footprint
within 10 m of the centre line of an OSM residential, living_street or unclassified way (half a 13
m reserve and a few metres' setback), in a kblock of the formal grid: the member kblocks other than
5810 whose outline lies within 10 m of such ways along at least half its length (56 kblocks, 5,475
homes; 5810's own ring is 44% such streets). 2,708 formal homes; deep informal homes: in 5810, at
least 30 m from every drivable way, 1,112. Beside them the per-block model: each member kblock's
own baseline on the complex (its street ring the exit; region_leaks `members`, 76 of the 77 here on
cores, in the region's population units; one, ZAF.9.1.4_1_7283, 22 homes, fails its solve: below).
Median escape times (90th percentile):

    model       formal homes    deep informal homes    formal / deep
    per-block   2.9 (12)        173 (381)              0.017
    edge        1,307 (1,647)   366 (849)              3.6
    bar         2,252 (3,162)   842 (2,159)            2.7
    U250        577 (695)       227 (485)              2.5
    U500        1,264 (1,680)   483 (1,169)            2.6
    U1000       1,873 (2,592)   704 (1,782)            2.7
    H250        84 (85)         76 (84)                1.1
    H500        300 (330)       210 (311)              1.4
    H1000       841 (1,056)     432 (863)              1.9
    S3          1,594 (2,156)   674 (1,622)            2.4
    S3U500      1,028 (1,331)   428 (994)              2.4

Per block a formal home on its street is at an exit (3 against 173). With the region's edge as the
exits it is not only as deep as an informal one, as the raster found, but deeper (2.5 -- 3.6x, with
or without barriers and uniform leaks): the formal homes lie further from the region's edge (a
median 314 m, 90th percentile 531 m, against the deep informal homes' 191 and 436 m: 5810's
informal west lies along the region's west and south roads) and in the denser fabric, the lanes
they share carrying more of the region's walk [inferred]. Only the building-boundary leak brings
them together (1.1x at 250 m), destinations being where people are; the street boost narrows it a
little (2.7 to 2.4x).

**Question 3, what gets cleared** (region_rows/summary_cleared.csv, summary_jaccard.csv,
summary_cross.csv). 5810 holds 33% of the region's footprint area, the formal grid 44%.

    config   5810's share of D   the formal grid's   Jaccard with bar's   with the raster's
             (SIMP, cheap)       (SIMP, cheap)       (SIMP, cheap)        (SIMP, cheap)
    edge     0.16, 0.25          0.84, 0.75          0.55, 0.42           0.61, 0.33
    bar      0.11, 0.16          0.89, 0.83          --                   0.44, 0.27
    U250     0.12, 0.16          0.88, 0.84          0.55, 0.45           0.47, 0.24
    U500     0.11, 0.20          0.89, 0.80          0.76, 0.63           0.48, 0.26
    U1000    0.11, 0.15          0.89, 0.84          0.91, 0.70           0.47, 0.26
    H250     0.36, 0.31          0.62, 0.60          0.22, 0.19           0.20, 0.10
    H500     0.27, 0.31          0.72, 0.67          0.38, 0.31           0.34, 0.16
    H1000    0.23, 0.27          0.77, 0.73          0.60, 0.47           0.46, 0.23

Where: under the boundary and the uniform leak 75 -- 89% of the cleared population is the formal
grid's and 11 -- 25% 5810's (the raster run's 17 -- 20% of the cleared area in 5810); the
building-boundary leak moves a quarter to a third of it into 5810. The uniform leak at 1,000 m and
500 m clears much what bar does (Jaccard 0.91 and 0.76 for SIMP), at 250 m and the home leak less
(0.19 -- 0.60). The presets' order: SIMP's default leads the cheap preset under every configuration
on all three lenses (L_P +0.015 to +0.021 under the boundary and the uniform leak at 500 -- 1,000
m, +0.008 at U250, +0.0015 to +0.010 under the home leak; P_{1/2} and J_2 alike), as on the blocks,
their clearings overlapping at Jaccard 0.37 -- 0.59. Against the raster run's picks: under edge
SIMP's clearing shares 0.61 with the raster's SIMP clearing and the cheap preset's 0.33 with its,
less under every other configuration. Each clearing scored under every model (L_P; the diagonal its
own):

    clearing of     under: edge    bar     U250    U500    U1000   H250    H500    H1000
    edge   SIMP            0.2825  0.2886  0.1320  0.2058  0.2594  0.0301  0.0708  0.1448
    bar    SIMP            0.2611  0.3105  0.1356  0.2188  0.2785  0.0290  0.0705  0.1512
    U250   SIMP            0.2627  0.2970  0.1427  0.2158  0.2684  0.0348  0.0772  0.1521
    U500   SIMP            0.2642  0.3090  0.1401  0.2207  0.2781  0.0307  0.0728  0.1527
    U1000  SIMP            0.2634  0.3103  0.1376  0.2199  0.2787  0.0298  0.0717  0.1523
    H250   SIMP            0.1940  0.2068  0.1069  0.1515  0.1866  0.0441  0.0759  0.1204
    H500   SIMP            0.2432  0.2681  0.1289  0.1937  0.2416  0.0415  0.0843  0.1493
    H1000  SIMP            0.2613  0.2990  0.1373  0.2138  0.2690  0.0363  0.0811  0.1573

(the cheap preset's alike: summary_cross.csv). Bar's and the uniform leaks' clearings are
near-interchangeable (each within 0.0002 -- 0.014 of the others' own on their models); edge's loses
0.022 under bar and bar's 0.021 under edge; the home leak's clearings lose 0.012 -- 0.10 of L_P
under bar (the more the shorter the trips), and bar's keeps 66 -- 96% of the home leak's own L_P
under it.

**Question 4, the street boost: not run** (cut for the GPU budget, above). What the cores could say
without its runs: the boost (3x on the triangles within 3 m of an OSM way) lowers bar's P_0 27% and
U500's 17%, serves the street homes more (the formal homes' median escape time 29% lower, the deep
informal ones' 20%: formal / deep 2.7 to 2.4 under bar), and leaves the leak's role as it was (U500
drains 37% of the current, S3U500 32%). Bar's clearings under S3 read L_P 0.2782 (SIMP) and 0.2599
(cheap) against 0.3105 and 0.2891 under bar, U500's under S3U500 0.2153 and 0.2019 against 0.2207
and 0.2052 (region_rows/summary_cross.csv): a clearing gains less where the streets already carry
the flow, less so with leaks. Whether the boost changes which buildings are cleared needs its runs
(about 1.05 GPU-h a configuration).

**What each configuration changes and what it costs** (decision support; not a choice):
- edge (the boundary only, no barriers; the per-block metric's name on the region): the raster
  run's setting reproduced on the complex (its picks' L_J2 within 0.001 of the 2b cross-check, our
  clearings ahead on its own objective); the corridors a near-tree reaching in from the south road;
  the railway walkable, so the settlement drains into the vacant land to the N1; a formal home 3.6x
  as deep as an informal one. Cost: none beyond the per-block metric.
- bar (the reference): the railway and the water closed but at the one path and the boundary
  bridges; P_0 +86%, J_2 3.7x, the drainage all south and west; Lens A 0.3105 / 0.2891; the
  clearing 89% / 83% in the formal grid; induced loops 10 -- 11. Cost: the barrier mesh (16.8M
  triangles against 16.1M; 16 minutes here, 25 on tesla2) and the OSM features read with the
  region.
- Uniform leak: changes the level (Lens A 0.14 -- 0.28 at lam 250 -- 1,000 m) more than the choice
  (Jaccard with bar 0.55 -- 0.91, its clearings within 0.014 of bar's own under each other's model)
  and interconnects the induced corridors 1 -- 3x bar's; formal homes still 2.5x as deep.
  Self-absorption negligible. Cost: none measurable (a diagonal term; the same runs' times).
- Building-boundary leak: changes the choice (Jaccard with bar 0.19 -- 0.60, a quarter to a third
  of the clearing into 5810), interconnects as the uniform leak's shortest trips do (16 -- 34 loops
  at q 0.9), and is the only family under which a formal home is about as far from its trips' ends
  as an informal one (1.1 -- 1.9x); Lens A small (0.04 -- 0.16), the clearing worth little when
  trips are short. Self-absorption: the large buildings' own destinations take up to 21% of their
  current at 250 m (9% population-weighted), under 1% from 1,000 m.
- Street boost: not run (above); its baselines say it narrows the formal homes' disadvantage only a
  little (2.7 to 2.4x) and lowers what a clearing gains.
- lam: the uniform family's 1,000 m is near bar (14% of the current leaks); 250 m leaks 63% and
  halves Lens A; the home family's 250 m makes the clearing nearly worthless (4%) and its
  self-trips not small.

**The guards** (gap_clear_check; each passes and fails under its fault, both ways at the final
code: all 77 runs on cores at ea65a237, the six that read region_leaks again at b34b0b8e, the
device guards on tesla2's cards at 872fdc65). New:
- `leak-assembly` (19570, 20269; U50 and H50): at the baseline, a random clearing, the most
  populous building cleared, a fractional state and the eps world, the operator on the fixed
  pattern against `_operator`'s assembly afresh plus a leak computed from its definition alone
  (each cell's triangles' area, the cells live at baseline by scipy's own components, the injection
  from the demand's points): equal to 1.2e-16, P against a direct solve within 1.0e-11, the
  structure the pattern's. Faults: `drop-one` (the largest leak dropped: P 2.6e-08 -- 7.2e-05 off),
  `not-live` (a leak on the largest cell not live at baseline: the cell grounded and singular in
  the oracle, P nan or 7.6e-07 -- 6.9e-05 off).
- `enclosure` (20269, 23597, the blocks with walled-in homes): under both leak families the homes
  walled in at baseline are those without leaks (1 and 6). Fault `every-cell`: every walled-in home
  freed by its own pocket's leak.
- `leak-gradient` (19570, 38136): the `gradient` guard's states and powers on the runtime's form
  under both families, within 3.0e-05 of the largest (tolerance 1e-4). Fault `adjoint-unleaked`
  (J_2's adjoint solved without the leaks): 0.15 -- 0.71 off at power 2 (power 1 is self-adjoint
  and unaffected).
- `leak-scale`: above (2.6% spread, +0.2% level; fault `per-cell` 4.3x, +34%).
- `barriers` (a synthetic block: a street on its south edge only, a railway across with a crossing
  node's path and a footbridge, a water channel parting the north): 624 open triangles inside the
  barrier, none outside the gaps; no home walled in; the path removed walls in exactly the
  north-west, the footbridge the north-east, both both. Faults: `not-inserted` (718 triangles
  outside the gaps, nothing walled in), `gap-closed` (the north walled in with both crossings).
- `exits-identity`: the per-block metric `cx+...+v34d40e8a`, 28 constants digesting to 34d40e8a; 26
  names (the ten configurations and three more under both presets' searches) parse back to their
  exits and search and are distinct. Faults: `in-code` (the exits module in CODE's walk: 39
  constants, digest 092d40a2), `suffix-dropped` (the leak's part left out: U250 and bar share a
  name, 6 names for 26).
- `loop-index`: above (fault `uncontracted`: the grid 9).
- `flux` now also a random clearing under the uniform leak, its drainage the street's and the
  leaks': imbalance 2.9e-11 (fault `flip-one` still 1.0).

The earlier guards, each both ways with its faults (gap_clear_check baseline, weights, reach,
gradient, free, identity, convex, pattern, constants, roads, flux; gap_check conform, delaunay,
apertures, filtration, free; gap_flow_check angle, shift, channel, field, law, closed;
pweight_check identity, total, displacement, gradient, weights; free_roads_check free-roads;
cluster_check ids), with their previous numbers (baseline 20269 P0 -1.6e-4; constants 28; pattern
1.2e-16); on the cards (run rl-dev) `device` within 1.5e-10 and 5.1e-11, `device-roads` within
8.6e-11 and 3.6e-11 on 20269 and 18895, `perturb` seen (0.86 -- 0.97 and 0.69 -- 0.88).
tests/test_clearings.py: 36 pass. mypy --strict clean on gap_exits, digests, region_leaks (new) and
the clean modules touched (gap_maps, metrics, cluster); `ruff check .` passes.

**Choices the brief left open, and two defects found:**
- The barrier tags and widths (above): railways rail, light_rail, narrow_gauge (a tram runs in the
  street; disused tracks are not fenced); every waterway class but in tunnels and culverts; water
  areas natural=water, landuse reservoir or basin, waterway riverbank (no wetlands). One width for
  every line, 8 m, the least that merges the railway's two tracks; 3 m a crossing. A crossing is
  any highway's bridge or tunnel over a barrier (the region's are the boundary roads' own, at its
  edge) or the way through a railway crossing node.
- The leak set: the cells that are unknowns at baseline (a ground cell's leak would do nothing).
  The leaks are among the reach's seeds (a cell joined to ground is live): the same reach, since
  they lie where the street reaches, and the `enclosure` guard's fault is then seen.
- Self-absorption read three ways (own building, own cells, anywhere) over 300 homes drawn
  uniformly, one solve each, on cores.
- The loop index's thresholds (the baseline's 0.8, 0.9, 0.95 quantiles) and its 25 m^2 holes;
  formal and deep homes as defined above.
- The prior on a card at 2 m (the maps' pixel), the flows of the other states at 1 m here,
  coarsened 2 x 2 for the maps.
- The member kblock ZAF.9.1.4_1_7283 (22 buildings, under the source's floor, so never run before):
  the per-block metric's refined solve raises "refinement stalled at a backward error of 1.0e+00"
  though its solution is a direct spsolve's (the componentwise backward error of rows whose terms
  all vanish beside the street); the same at HEAD; its homes are left out of the per-block column.
  Not traced further.
- A region's all-cell states (the prior; the eps world of SIMP and the tension) overflow SuperLU's
  32-bit indices on a core: the region's optimizers run on a GPU only (cuDSS).

**Compute.** GPU, tesla2's V100s, one card a task, at most four at once: 10.0 GPU-h of the 12 (the
pilot 1.02; rl-runs, 11 runs and 6 priors, 6.62; rl-runs2, the 250 m lambda's 4 runs and 2 priors,
2.13; the device guards 0.27). CPU here, single-threaded, at most 8 processes: the two region
meshes (16 minutes each), the ten baselines with the self-absorption (about 3 CPU-h, a first pass
of six failing at the prior), the 76 member kblocks (about 1), the clearings' scores and the 9 x 16
cross-scores (about 4), the guards (2.4 h both ways, then 0.5), the report, smoke tests and
diagnoses: about 13 CPU-h [inferred from the runs' times]. orion and quadro2 were not needed.

### The region's leaks: destinations at buildings and on open ground (5810@major)

BACKLOG 4, decision (4) (the brief: .superpowers/sdd/2026-10-15-region-mixed-leak; the owner
2026-10-15: the barriers adopted; internal trips count, "especially if you e.g. cleared an entire
island so all traffic was internal"; "we should probably have some small leakage in open space,
since you can in principle live off the land, but buildings should have greater leakage"; streets
"more conductive, but not make them leak more"; "What if we just use T=500m?"). The two leak
families of the region test folded into one with a building share s, its strength to be set by the
median length T of an internal trip rather than by a nominal lam, then run at T 500 m. In three
parts. **First (to "What the owner could choose instead"): no lam gives a median internal trip of
500 m on 5810@major with its edge open, at any s; the runs waited for the owner.** **Then (from
"The owner's decisions after the refusal"): two constants, a leak per m^2 of open ground and one
per m^2 of footprint, chosen once on 5810@major with its edge closed for a 500 m median trip, the
same for every block and region; five of the eight runs at the 6 GPU-h cap; the 13 tuning blocks
under one metric.** **Last (from "A 10x boost on the drivable streets"): the boost on drivable
streets at 10x, and who each cleared building helps against whom it displaces.** Code: gap_exits.py
(InternalLeak, the edge), gap_clear.py (Model.leaked_state: a leak's other strength on the same
pattern; the leak site with the edge open), gap_maps.py (leak_drain and cell_geometry, shared with
flow_of), region_leaks.py (the configurations, home_solves, the trips, `calibrate`, `blocks`),
gap_clear_check.py (`leak-fold`, `trip-calibration`, `closed-edge`, the leak guards at mixed
leaks).

**The fold** (gap_exits.InternalLeak(s, lam), its name `L<lam>s<s>`). One Strategy, two parameters:
the total sigma |A| / lam^2 over the cells live at baseline, as before; a share s of it at the
buildings, placed as HomeLeak placed it (by each cell's population injection at baseline), and 1 -
s on open ground, as UniformLeak placed it (sigma |c| / lam^2 on every live cell). Each cell's
conductance is (1 - s) times the open ground's term plus s times the buildings', so at s 0 and s 1
the other term is multiplied by 0 and added: the old classes' answer bit for bit. `leak-fold`
(19570, lam 50 m): the leak per cell against the two formulas copied verbatim from before the fold,
0 apart at s 0, s 1 and s 0.75; against the definition's independent oracle 0 (s 0), 3.5e-16 (s 1)
and 4.7e-16 (s 0.75); the baseline's operator against the one assembled afresh 5.8e-17. On the
region the migrated `L500s0` and `L500s1` (U500 and H500) re-score bit for bit: P_0, P_{1/2},0,
J_2,0, the leak's total and its cells, and every home's escape time equal to the stored rows' (a
script, not kept).

**The migration.** The region test's names: U<lam> is `L<lam>s0`, H<lam> `L<lam>s1`, S3U500
`L500s0S3` (labels; the run directories region_rows/runs/<label>/; the baseline, cross, score and
image files and the maps; the rows' label, source and metric columns; the committed summaries,
field by field as text, their numbers untouched); bar, edge and S3 keep theirs. A suffix reads
`+xB8c3L500s0va03a092b` where it read `+xB8c3U500va03a092b`: the exits code is unchanged (a03a092b:
no constant moved), and so is the per-block metric (`cx+shy0.5+s0.125r1b1+R0.5g4x2q4+v34d40e8a`, 28
constants). No reader of the old names is left: the grammar is
`[B<line>c<cross>][L<lam>s<s>][S<factor>]`, and gap_clear.complex_of refuses a `U500`. The run rows
gain `trip_m` (the median trip a configuration's lam was calibrated to; empty for the nominal lam).
`exits-identity` reads the 42 stored names (summary_lenses.csv, the runs' rows,
summary_baseline.csv) and every one parses back to its label's configuration.

**The trips** (region_leaks, THE TRIPS). At baseline under a configuration's leak, a home's trips
are its current leaking inside: its unit injection solved alone (home_solves, the solves
self-absorption reads too), the current to ground through each cell's leak (gap_maps.leak_drain) a
trip's end, the trip's length the straight-line distance from the centroid of the home's injection
points (weighted as it injects) to the cell's centroid, its weight that current. Current leaving by
the boundary (the street faces and the ground cells) is not a trip of the leak and is not counted.
Trips pool over TRIP_N 2,000 homes drawn by population with replacement (seeded; among the homes
with demand not walled in), each draw its home's trips per unit of its injection; the median is the
pooled distribution's (1 m bins, linear within a bin). `calibrate <s> <T> <boost> <procs>` finds
lam by bisection in log lam over whole metres after a bracketing search (secant steps in log-log),
the same draws at every step, to a median within TRIP_TOL (1%) of T, and bootstraps the draws
(2,000 resamples) for the median's uncertainty. The median rises with lam toward a limit as the
leak vanishes (the trips then end where the no-leak potential times the leak's shape puts them,
nearly all the current leaving by the boundary; read at lam 10^6 m): a T past it is refused, its
steps recorded (ROWS/calib/, summary_calib.csv).

**Checked where the answer is known** (`trip-calibration`). On the `leak-scale` block (open ground
800 m square, its edge the street, one 2 m home at its centre) at s 0, calibrating to T 63 m gives
lam 50.07 m, and 1.2572 lam (the screened point source's median: x K1(x) = 1/2) is T within 8.6e-4;
a target past the limit (1.5 times it) is refused. With a second home 50 m from the street edge
(the boundary taking 44% of its current), the pooled median at lam 60 m, 67.75 m, is the method of
images' 67.54 m (each home's point source less its first images in the four edges, integrated on a
polar grid) within 3.1e-3; per home 75.02 and 54.83 m against 75.03 and 54.96. Faults:
`boundary-trips` (the street faces' drainage counted as trips) 62.44 m, -7.5e-2; `wrong-home` (each
home's trips measured from the next home's injection) 346 m.

**No lam reaches T 500 m on 5810@major.** The median internal trip at baseline with the barriers
(1,000 draws, 857 homes, by a scratch script on region_leaks' functions; * 400 draws, 368 homes;
`inside` the pooled share of the current leaking inside; the last column the median with the
boundary's trips counted too, each ending at the cell it drains from):

    s      lam (m)    median internal trip (m)   inside   with the boundary's trips (m)
    0      250        174                        0.63     173
    0      400 *      230                        0.47     --
    0      500        250                        0.37     262
    0      1,000      296                        0.14     340
    0      -> inf     319                        0        393
    0.75   2,000 *    245                        0.16     --
    0.75   5,000 *    264                        0.03     --
    0.75   -> inf     267                        0        393
    0.9    400        105                        0.67     122
    0.9    450 *      115                        0.66     --
    0.9    1,000      192                        0.40     231
    0.9    2,000 *    240                        0.18     --
    0.9    5,000 *    262                        0.04     --
    0.9    12,000 *   266                        0.007    --
    0.9    -> inf     265                        0        393
    1      250        66                         0.77     78
    1      500        119                        0.62     141
    1      1,000      187                        0.41     226
    1      2,000 *    237                        0.19     --
    1      -> inf     264                        0        393

(-> inf: the no-leak potential times the leak's shape, every share from one set of solves; at s
0.25 and 0.5 the limit is 281 and 271 m; the bootstrap's sd 1.5 -- 2.1 m at 400 draws.) So the
longest median internal trip the region allows is 264 -- 319 m, approached only as the leak
vanishes, and even with the boundary's trips counted the median is at most 393 m. The reason is the
region's edge: its roads are exits, the deep informal homes a median 191 m from it and the formal
homes 314 m (the region test's Question 2), so a trip that would end 500 m away mostly leaves by
the boundary first, and the trips that end inside are the short ones. At the region test's nominal
lam the median internal trips were 250 m (U500, now L500s0) and 119 m (H500, L500s1); the leak's
median on open ground, 1.257 lam, is 629 m at lam 500 m. The shipped command refuses on its own
(`calibrate <s> 500 <boost>`, 2,000 draws, 1,570 homes; ROWS/calib/, summary_calib.csv): the limit
at s 0.9 is 263.5 m (the bootstrap's sd 0.89 m, 95% 261.9 -- 265.3), at s 0.75 265.3 m (sd 0.88),
at s 1 262.6 m (sd 0.90); with the street boost (3x on the OSM ways) at s 0.9 270.5 m (sd 0.86 m):
the streets' conductance lengthens the internal trips by 7 m, not toward 500.

What the owner could choose instead (decision support; the runs wait for it):
- a T within reach of internal trips, at s 0.9: 150 m puts lam near 700 m (about 55% of the current
  leaking inside), 200 m near 1.15 km (about 35%), 250 m near 3 km (about a tenth), close to the
  limit, where lam is ill-determined (the median nearly flat in it) [lam and the shares inferred
  from the table];
- the trips with the boundary's counted: their median is at most 393 m (no leak at all), still
  short of 500 m;
- the leak's strength set by the share of the current that ends inside rather than by a length (at
  s 0.9 half of it between lam 400 and 1,000 m, 0.67 and 0.40);
- a region whose edge lies far from most homes, where 500 m internal trips fit.

The draws' size: at 2,000 draws (TRIP_N) the bootstrap's sd is 0.34% of the median at the limit
(1.4% at 400 draws, at lam 450 m), so the 1% tolerance (TRIP_TOL) is about 3 sd there; at a shorter
T the relative sd is larger (the 400 draws' 1.4% at 115 m would be about 0.6% at 2,000), and the
tolerance would be 2% or the draws more. A step on a core is a factorization of the region (about a
minute) and 1.8 s a home on one process (1,570 homes: 48 minutes; 24 on two).

**Not run then**: the eight runs (four configurations, two presets), the priors and maps, the
baselines, scores and cross-scores, the loop index and the boost's verdict, all waiting for T. No
GPU time was used.

**The guards at the refusal** (gap_clear_check; each passes and fails under its fault, both ways at
the final code): `leak-fold` and `trip-calibration` (above); `leak-scale` at s 0 and 0.5 (at s 0.5
the open ground's share alone screens the far field, lam' = lam / sqrt(1 - s), and the home's own
building drains 98% of its current: the ratio's spread 2.2%, its level +8e-4 against Q' / (2 pi
sigma); s 0 as before, 2.6%, +2.1e-3; fault `per-cell` 4.3x and 1.2x off); `leak-assembly` (19570,
20269, at s 0, 1 and 0.5: the operator against the one assembled afresh plus the oracle within
1.2e-16, P within 1.0e-11; faults `drop-one`, `not-live`), `enclosure` (20269, 23597: 1 and 6 homes
walled in at every s, as without leaks; fault `every-cell`), `leak-gradient` (19570, 38136: within
3.3e-5 of the largest at every s; fault `adjoint-unleaked` 0.06 -- 0.71 off at power 2), `flux`
(the leaky clearing at s 0 within 2.7e-11, every state within 5.0e-11; fault `flip-one`);
`loop-index` and `barriers` (they read region_leaks) as before (8, 5, 0; 624 open triangles inside
the barriers, none outside the gaps; their faults seen); `exits-identity` (30 names of 30
configurations and presets distinct and parsing back, the 42 stored names their labels', the
per-block metric's 28 constants digesting to 34d40e8a; faults `in-code` 39 constants, digest
ef375dc2, `suffix-dropped` 6 names for 30). mypy --strict clean on gap_exits, gap_maps,
region_leaks (gap_clear and gap_clear_check as before: 32 and 10 errors); `ruff check .` passes;
tests/test_clearings.py: 36 pass. The earlier guards (gap_clear_check's baseline to device-roads,
gap_check's, gap_flow_check's, pweight_check's) were not rerun at this code: it moves two of
gap_maps.flow_of's computations into functions (bit for bit; `flux` reads it) and adds
Model.leaked_state, which none of them calls; they are for the end of the study, with its runs'
code.

**Compute at the refusal.** CPU here, single-threaded, at most 8 processes, then 4 (the
controller's ruling: a second agent shares the machine): the trips' pilot, scan and limits (about 9
CPU-h), the fold's two region baselines (0.2), the four shipped calibrations (about 4), the guards
(about 3): about 16 CPU-h [inferred from the steps' times]. GPU: none.

**The owner's decisions after the refusal** (2026-10-15; the addendum:
.superpowers/sdd/2026-10-15-region-mixed-leak/addendum.md): the edge's trips are real ("If most
homes are near the boundary it perhaps suggests they do typically venture to the great beyond");
the leak's hyperparameters are set once for every block and region ("rather than them meaning
different things per block"), chosen on 5810@major with its edge closed for a 500 m median trip and
90% of the leak at buildings, then frozen; one metric everywhere, the per-block runs taking the
same leaks with their street ring as the boundary, RUNTIME switching at the consolidated pass.

**Two constants** (gap_exits.InternalLeak(open_m, foot_m), named `L[o<open_m>][b<foot_m>]`). The
open ground leaks sigma / open_m^2 per m^2 of it (live at baseline), and each building's
destinations sigma / foot_m^2 per m^2 of its footprint, on the cells where its homes inject, as
they inject (LeakSite.foot: System.Pc @ each building's footprint area). A person is a fixed area
of footprint (area population), so the building constant is a leak per person whatever the block;
it is per m^2 of footprint, not per unit of AreaPopulation's weights, which each block normalises
to mean 1. Either length may be infinite (none of that leak). Neither is a total matched to a
region: (s, lam) made the building leak per person depend on the region's open area over its
population, which the owner ruled out. A number in the name is spelt `:g` or, where that would
round it, by its repr, so every name parses back to its float (gap_exits.num). The exits code is
unchanged (a03a092b); the per-block metric too (`cx+shy0.5+s0.125r1b1+R0.5g4x2q4+v34d40e8a`, 28
constants).

**The edge** (gap_exits.Edge, a fourth Strategy of the exits): `OpenEdge` (the street's faces
ground, as the per-block metric has them) or `ClosedEdge` (`E0`: every street face's conductance to
ground 0, so every trip ends at a leak). A closed edge's leaks lie where the open edge's street
reaches (Model.leak_site's live cells are the open edge's reach, its ground cells among them), so
the leak is the same constants on the same ground either way.

**The migration, again.** Each of the region test's (s, lam) is exactly one pair of constants on
5810@major with the barriers (region_leaks.region_test_leak): at s 0 open_m = lam (`L500s0` is
`Lo500`), at s 1 foot_m = lam sqrt(F / A), A = 5,944,597 m^2 the open ground live at baseline and F
= 437,058 m^2 the footprint injected there (REF_OPEN_M2, REF_FOOT_M2), so `L250s1`, `L500s1`,
`L1000s1` are `Lb67.78731902755004`, `Lb135.57463805510008`, `Lb271.14927611020016`, and `L500s0S3`
is `Lo500S3`. The runs' and baselines' metric columns and the summaries' metric fields are
rewritten (the labels kept: they name how the region test set them); re-scored on the region,
L500s0's baseline is bit for bit the stored one, L500s1's and L250s1's within 5.0e-12 (P_{1/2},0;
P_0 2.1e-13, J_2,0 4.2e-13, every home's escape time within 3.5e-11, the leak's total 4.4e-16): the
same leak to floating point. The open-edge calibration's refusals (the section's first half) are
records, set apart: region_rows/calib_open/ and summary_calib_open.csv, read by no code.

**The closed reference and the constants** (`calibrate <share> 500 <start> <procs>`;
ROWS/calib/T500b<percent>.parquet, summary_calib.csv). 5810@major with the barriers and its edge
closed: every trip ends at a leak (inside 1.000 at every step). The two lengths scale together, the
share at buildings fixed by them on the reference: foot_m^-2 F = share / (1 - share) open_m^-2 A,
with A 5,944,597 m^2 of open ground live at baseline and F 437,058 m^2 of footprint injected there
(the same as the open edge's: the closed edge's leaks lie where the open edge's street reaches), so
the scale is the total's lam (sigma A / scale^2) and open_m = scale / sqrt(1 - share), foot_m =
scale sqrt(F / (share A)), each rounded to whole metres (region_leaks.pair_of: the frozen constants
are the evaluated ones). The search: the limit first (scale 10^6 m: 853 m at 90%, so 500 m is
within reach), then secant steps in log-log and bisection in log scale to a median within TRIP_TOL
of 500 m, the same draws at every step. Well posed (closed_check, at each pair's constants): the
closed baseline's live cells form one component, with a leak (none without), the 6,733,820 cells
the open edge's reach holds, the same 36 homes walled in.

The sample's size, sized at this reference: a first pass at 2,000 draws (1,570 homes) found the
constants below with a median 499.3 m, but the bootstrap's sd there was 4.5 m (0.9%; 95% 490.6 --
508.4), larger than at the open edge's limit (0.34%): with every trip internal, the homes' own
medians spread more. At a 1% tolerance a sample that does not matter would need about 15,000 draws;
the tolerance is 2% (10 m) and the sample 4,000 draws (TRIP_N), the sd then about a third of the
tolerance.

The constants (2,000 resamples of the 4,000 draws, 2,774 homes; each step about 28 minutes on three
processes, 80 on one):

    pair     at buildings  open_m  foot_m  median (m)  sd (m), 95%          steps: scale, median
    central  0.8999        6,214   562     499.5       3.2, 493.2 -- 505.9  1965 499.5
    75%      0.7499        3,608   565     500.1       3.0, 494.2 -- 506.1  1965 527.3, 1863 510.1,
                                                                            1804 500.1
    100%     1             inf     552     492.2       3.3, 485.9 -- 498.7  1965 482.3, 2037 492.2

(each after the limit, scale 10^6 m: 853.2, 949.8 and 798.7 m; the central pair's first pass at
2,000 draws: 1,581/143 m at scale 500, 169.2 m; 4,671/422 m at 1,477, 416.2 m; 5,825/526 m at
1,842, 479.6 m; 6,214/562 m at 1,965, 499.3 m; the 4,000 draws, started there, kept it). The scale
for 500 m is 1.8 -- 2.0 km at every share; the buildings' length hardly moves (552 -- 565 m) and
what the share changes is the open ground's (3.6 km at 75%, 6.2 km at 90%, none at 100%): a
person's destinations leak as sigma / (562 m)^2 per m^2 of footprint, about 120 times the open
ground's per m^2 at 90% (the open ground being 13.6 times the footprints' area there, its 10% of
the leak).

Frozen in region_leaks.CALIBRATED (the configurations `T500b90`, `T500b75`, `T500b100`,
`T500b90S3`): the central `Lo6214b562`, the 75% `Lo3608b565`, the 100% `Lb552`. The 100% pair
stopped at 492.2 m, inside the 2% but 1.6% short, its bootstrap's interval not reaching 500; a
further step was not taken.

**One metric everywhere: the 13 tuning blocks** (region_leaks `blocks`, on cores, no
re-optimization; region_rows/summary_blocks.csv). Each block under the per-block metric and under
it with the central constants (`cx+shy0.5+s0.125r1b1+R0.5g4x2q4+v34d40e8a+xLo6214b562va03a092b`:
the block's street ring the edge, open; no barriers), the stored SIMP-default and cheap-preset
clearings (relax_rows; the cheap preset's, the translucent search's,
gap_rows/runtime/preset/gpu/cheap@e0.05) re-scored under each; the per-block metric's scores of
them are the stored ones within 7.1e-12. As the controller expected, on a block nearly all the
current still leaves by the street ring: the leaks drain 0.26 -- 2.8% of it at baseline (5810 2.8%,
20269 1.7%, 30848 1.2%, the others under 0.8%), a typical home's share leaking inside 0.2 -- 2.9%
(median over 300 homes), its own building's destination 0.00 -- 0.03% (the largest home 1.6%, on
30848). The baselines fall by P_0 0.4 -- 4.5%, P_{1/2},0 0.6 -- 6.4%, J_2,0 1.0 -- 9.3% (5810 and
30848, the deepest, the most). Lens A falls a little and evenly: SIMP's L_P by a median 0.0020
(0.0008 -- 0.0091), the cheap preset's 0.0020 (0.0008 -- 0.011), P_{1/2} and J_2 alike; the leaky
metric keeps 96.6 -- 99.4% of each clearing's L_P (median 99.2%). The presets' order is unchanged
on every block and lens (SIMP ahead on 11 of 13 on L_P, 9 on L_{P_1/2}, 10 on L_J2, under either
metric). So the switch moves the blocks' levels by a percent or two and decides nothing among the
presets [the methods' order beyond the presets inferred, not measured].

**The switch** (not made here; the consolidated pass's start, which re-scores every block): RUNTIME
becomes `cx+shy0.5+s0.125r1b1+R0.5g4x2q4+v34d40e8a+xLo6214b562va03a092b`, the per-block metric with
the central constants (an exits code of a03a092b; the code's digest 34d40e8a until then), and a
region's runs add their barriers (`+xB8c3Lo6214b562va03a092b` on 5810@major). The per-block
metric's name and digest stay as they are in this task.

**The runs** (5810@major, the barriers, the edge open; SIMP's default and the cheap preset at area
D 0.05, free first, optimizing P; tesla2's V100s, the mesh already on the node: runs lc-runs90,
lc-runs100, lc-runs75b). Of the eight, five ran: the central pair's both presets, the boost's both,
the 100% pair's SIMP. The rest were cut at the 6 GPU-h cap: the leak's runs took longer than the
region test's (SIMP's optimizer 1,543 s on the central pair, 3,114 s on the 100% pair, as the
region test's building-only H500 took 2,106 s against bar's 1,358; the host's work on tesla2,
shared with other jobs at 8 CPUs each, slower too [inferred]), so the 100% pair's cheap preset was
stopped at 1.76 GPU-h to let the 75% pair's SIMP run, which reached the cap at 0.72 GPU-h before
finishing (in SIMP's second continuation stage); the 75% pair's cheap preset and prior never ran.
The 75% pair is measured at baseline only (below). Every run frees the 36 walled-in homes first (D
0.0013) and leaves none.

    config      preset  L_P     L_P1/2  L_J2    cleared  t (s)   wall (s)
    T500b90     simp    0.2438  0.2448  0.2391  707      1,543   1,766
    T500b90     cheap   0.2291  0.2282  0.2234  524      1,948   2,179
    T500b100    simp    0.2426  0.2436  0.2379  711      3,114   3,861
    T500b90S3   simp    0.2421  0.2383  0.2331  682      1,524   1,809
    T500b90S3   cheap   0.2280  0.2233  0.2191  495      1,894   2,175
    (bar        simp    0.3105  0.3177  0.3120  753      1,358   1,583)
    (L500s0     simp    0.2207  0.2242  0.2177  730      1,463   1,691)
    (L500s1     simp    0.0843  0.0741  0.0688  653      2,106   2,538)

**The baselines** (region_rows/summary_baseline.csv, summary_absorb.csv; P_0 in the area
population's units; the leaks' share of the current at baseline; a typical home's share leaking
inside and at its own building, over 300 homes: the median, the population-weighted mean and the
largest):

    config      P_0        P_1/2,0    J_2,0      leaks   inside: median,  own building: median,
                                                     drain   pop.-weighted    pop.-weighted, max
    bar         1.76e+07   3.53e+07   3.96e+10   0       --               --
    T500b90     1.27e+07   2.43e+07   1.99e+10   0.176   0.20, 0.13       0.00018, 0.0020, 0.0045
    T500b75     1.26e+07   2.41e+07   1.97e+10   0.180   0.21, 0.13       0.00018, 0.0020, 0.0045
    T500b100    1.26e+07   2.41e+07   1.97e+10   0.178   0.20, 0.13       0.00019, 0.0020, 0.0047
    T500b90S3   1.00e+07   1.83e+07   1.21e+10   0.139   0.16, 0.10       0.00015, 0.0016, 0.0037
    L500s0      9.68e+06   1.88e+07   1.17e+10   0.371   0.41, 0.31       0
    L500s1      2.60e+06   3.74e+06   6.90e+08   0.608   0.78, 0.46       0.0026, 0.029, 0.067

At a 500 m median trip on the closed reference the leaks are weak with the edge open: they drain
18% of the current (a typical home's 20%), P_0 28% below bar's, where U500 drained 37% and H500
61%. The three shares are near-identical at baseline (P_0 within 0.6%, the leaks' share 17.6 --
18.0%): with the trip length fixed, where the destinations lie barely moves the region's flow.
Self-absorption is negligible: a typical home's own building takes 0.02% of its current, the
largest 0.45% (H500's 6.7%), the pop-weighted 0.2%.

**Formal against deep informal homes** (the region test's definitions; median escape time, 90th
percentile, formal / deep): T500b90 1,604 (2,179) against 662 (1,580), 2.4; T500b75 and T500b100
alike (2.4); T500b90S3 1,243 against 564, 2.2; bar 2.7, L500s0 2.6, L500s1 1.4, per block 0.017.
With the leaks this weak a formal home stays 2.4 times as deep as a deep informal one: the edge
decides.

**What gets cleared** (summary_cleared.csv; 5810's share of the cleared population, the formal
grid's; Jaccard with bar's): T500b90 SIMP 0.17, 0.83, 0.85; cheap 0.21, 0.78, 0.64; T500b100 SIMP
0.17, 0.83, 0.85; T500b90S3 SIMP 0.20, 0.80, 0.54; cheap 0.27, 0.72, 0.45 (bar's own 0.11, 0.89;
L500s0's 0.11, 0.89; L500s1's 0.27, 0.72). The central pair's clearing is mostly bar's (Jaccard
0.85 SIMP, 0.64 cheap; with L500s0's 0.78, 0.67, with L1000s1's 0.69, 0.58, with L500s1's 0.43,
0.36), a little more in 5810 (17% against 11%). **How much s matters**: the 90% and 100% pairs'
SIMP clearings share 0.98 (702 in common of 716), and each scores within 0.0001 of the other's own
under the other's model (T500b100's under T500b90 0.2439 against its own 0.2438; T500b90's under
T500b100 0.2426, the same as T500b100's own): at a fixed 500 m trip the share at buildings changes
nothing a clearing does, between 90 and 100%; 75% was not run, and its baseline is between theirs.

**Cross-scores** (summary_cross.csv; L_P, each clearing under each model, SIMP's):

    clearing of   under: bar     L500s0  L500s1  L1000s1  T500b90  T500b100  T500b90S3
    bar                  0.3105  0.2188  0.0705  0.1512   0.2451   0.2438    0.2333
    L500s0               0.3090  0.2207  0.0728  0.1527   0.2448   0.2435    0.2348
    L500s1               0.2681  0.1937  0.0843  0.1493   0.2188   0.2179    0.2297
    L1000s1              0.2990  0.2138  0.0811  0.1573   0.2408   0.2397    0.2393
    T500b90              0.3070  0.2175  0.0742  0.1533   0.2438   0.2426    0.2347
    T500b100             0.3071  0.2176  0.0741  0.1532   0.2439   --        0.2347
    T500b90S3            0.2932  0.2084  0.0767  0.1508   0.2344   0.2333    0.2421

(the cheap preset's alike.) bar's, U500's and the central pair's clearings are near-interchangeable
under one another's models (within 0.0035), as bar's and the uniform leak's were in the region
test; under the central pair's model bar's SIMP clearing scores 0.2451, a little above the central
pair's own SIMP (0.2438): SIMP's run-to-run spread is that large here. H500's loses 0.025 there.
SIMP leads the cheap preset under each of the new configurations (+0.015 and +0.014 L_P), their
clearings overlapping at Jaccard 0.40 and 0.42.

**The induced corridors' loop index** (summary_loops.csv; loops of the induced corridors at q 0.9,
SIMP's and the cheap preset's; at q 0.8): T500b90 9, 10 (54, 42); T500b100 10 (54); T500b90S3 12,
10 (56, 41); bar 10, 11 (39, 23); L500s0 18, 12 (82, 66); L500s1 26, 22 (69, 70). With a weak leak
the corridors a clearing induces are about bar's at q 0.9 and somewhat more interconnected at q
0.8.

**The boost's verdict** (StreetBoost 3x on the OSM ways, with the central constants): it changes
what gets cleared moderately and what a clearing is worth little. Jaccard with and without it 0.58
(SIMP) and 0.52 (cheap), about the two presets' own overlap (0.40, 0.42); each clearing loses
0.0074 -- 0.0094 of L_P under the other's model (T500b90S3's SIMP under T500b90 0.2344 against its
own model's 0.2421 and T500b90's own 0.2438; T500b90's under the boost 0.2347 against T500b90S3's
own 0.2421; the cheap preset's 0.0041 and 0.0069), 3 -- 4% of Lens A. With the boost the clearing
moves a little into 5810 (20% of it against 17%), the leaks drain less (14% against 18%: the
streets carry more to the edge) and a formal home's depth falls 22% (2.2x a deep home's against
2.4x). Lens A is nearly the same (0.2421 against 0.2438).

**Maps** (region_rows/maps/<config>_<preset>.png: the flow after against the no-buildings prior,
the change from the baseline, the change on 5810): T500b90_simp, T500b90_cheap, T500b100_simp,
T500b90S3_simp, T500b90S3_cheap. The 75% pair has none (no run, no prior).

**The guards, at the final code** (each passes and fails under its fault; on a core here): new
`closed-edge` (TRIP_BLOCK closed, the open ground's leak at 60 m: one live component, with a leak;
the cells live and the homes walled in the open edge's; the street's drainage 0 and the leaks' the
injection within 3.0e-13; the pooled median trip 69.26 m against the method of images with the
images added, 69.18 m, +1.1e-3, every trip inside; fault `edge-open`, ClosedEdge.mesh leaving the
edge open: the street drains 22% and the median -2.1e-2); `leak-fold` on the constants (19570 at
lam 50 m: the region test's s 0, 1 and 0.75 as constants on the block, bit for bit at s 0, 2.4e-16
and 3.1e-16 at s 1 and 0.75 against the two formulas, the oracle within 1.6e-16, the operator
5.8e-17; fault `lengths-swapped`, the open ground's length on the footprints and theirs on open
ground: 0.43 -- 3.8 off); `leak-scale` (the open ground's 60 m alone, and 84.9 m with the home's
own 4 m^2 footprint leaking at 2 m, which drains 38% of its current: spread 2.6% and 2.1%, level
+2.1e-3 and +7.5e-4; fault `per-cell`); `leak-assembly`, `enclosure`, `leak-gradient` and `flux` at
the three pairs of LEAK_PAIRS (open ground only, footprints only, both) as before (1.2e-16, P
1.0e-11; 1 and 6 walled in; 3.3e-5; 5.0e-11); `trip-calibration` (the scale's search, unchanged
numbers); `exits-identity` (46 names of 46 configurations and presets distinct and parsing back, 46
stored names their labels', the per-block metric's 28 constants digesting to 34d40e8a; faults
`in-code`, 39 constants, `suffix-dropped`, 8 names for 46). The earlier guards, every one both ways
at the final code (86 runs, 4.4 h of core time, one or two at once): gap_clear_check baseline,
weights, reach, gradient (19570, 38136: within 3.5e-5), free, identity, convex, pattern, constants,
roads, flux, barriers, loop-index; gap_check conform, delaunay, apertures, filtration, free;
gap_flow_check angle, shift, channel, field, law, closed; pweight_check identity, total,
displacement, gradient, weights; free_roads_check free-roads (20269, 18895). Two first runs on
blocks outside a guard's own (the gradient guard on 20269: up to 5.6e-4 against its 1e-4;
free-roads on 23597: its road for home 272 does not free it) fail identically at 72cabeac, before
this work: not regressions. No device guard was run: the factorization and the solves are unchanged
(gap_factor untouched), Model only computes its leak site and mesh differently. mypy --strict clean
on gap_exits, gap_maps, region_leaks; gap_clear and gap_clear_check as before (32 and 10 errors);
`ruff check .` passes; tests/test_clearings.py: 36 pass.

**Compute.** GPU: 6.00 GPU-h of the 6 (sacct): the mesh check on tesla2 0.004 (the barrier mesh was
there: 3 s to load); the central pair's and the boost's four runs and two priors 2.38; the 100%
pair's SIMP and prior 1.14, its cheap preset 1.76 (stopped); the 75% pair's SIMP 0.72 (stopped at
the cap). CPU here, single-threaded, at most 8 processes, then 4: the closed reference's
calibrations (the central pair at 2,000 draws 4.2 CPU-h, at 4,000 2.8; the 75% and 100% pairs 5.4
and 3.9), the per-block check 0.4, the region's baselines, scores and cross-scores about 3, the
migration's re-scores 0.3, the guards about 5.3: about 25 CPU-h [inferred from the steps' times],
beside the refusal's 16.

**A 10x boost on the drivable streets, and who each clearing helps** (addendum 2:
.superpowers/sdd/2026-10-15-region-mixed-leak/addendum2.md; the owner 2026-10-16, on the formal
grid taking most of the clearing: a strong street boost "rewards the infrastructure investment,
which was likely undertaken for various factors we don't have visibility into"; "I don't want a
100x boost for drivable streets, but a 10x boost is I think reasonable"; and "Clearing N people's
homes to provide access to n<N people's homes seems unfair (on average at least)", to be measured
before anything changes in the objective).

**The boost's classes** (gap_exits.StreetBoost(factor, classes), named `S<factor>h<classes>`, the
classes sorted and joined by `-`, a way's highway tag with `_link` folded). The 3x check boosted
every way with a highway tag, footways, paths and tracks among them; its runs keep their labels
(`S3`, `L500s0S3`, `T500b90S3`) and their names now spell its classes out, every tag on 5810@major
(`S3hfootway-motorway-path-residential-secondary-service-services-tertiary-track-unclassified`: the
same triangles, S3's baseline re-scored bit for bit). The new boost is 10x on the drivable streets
alone, region_leaks.DRIVE_CLASSES
(`S10hliving_street-motorway-primary-residential-secondary-service-tertiary-trunk-unclassified`),
with the central constants: `T500b90D10`. New guard `street-boost` (a synthetic block with a
residential way, a footway and a tertiary link, the boost on residential and tertiary at 10x: the
366 triangles on the residential way and the 49 on the link boosted, none of the 96 on the footway,
passages between boosted triangles x10 and the others x1; fault `classes-dropped`, the class filter
dropped: the footway's 96 boosted).

**The 10x runs** (tesla2, run lc-d10; both presets; the edge open, the barriers, the central
constants):

    config       preset  L_P     L_P1/2  L_J2    cleared  t (s)   wall (s)
    T500b90D10   simp    0.2610  0.2585  0.2525  638      2,124   2,819
    T500b90D10   cheap   0.2428  0.2404  0.2352  465      1,881   2,170
    T500b90S3    simp    0.2421  0.2383  0.2331  682      1,524   1,809
    T500b90      simp    0.2438  0.2448  0.2391  707      1,543   1,766

At baseline the 10x boost carries the walk to the edge on the streets: P_0 7.28e+06, 43% under the
unboosted central pair's (3x: 21%), the leaks draining 10% of the current (18% unboosted; a typical
home 12% inside, against 20%), a formal home 2.1x as deep as a deep informal one (886 against 414;
2.4x unboosted, 2.2x at 3x). **What gets cleared** moves toward 5810: its share of the cleared
population 0.21 (SIMP) and 0.29 (cheap) against 0.17 and 0.21 unboosted, the formal grid's 0.78 and
0.68 against 0.83 and 0.78 (3x: 0.20, 0.27; 0.80, 0.72). The clearing changes much more than at 3x:
Jaccard with the unboosted central pair's 0.35 (SIMP) and 0.31 (cheap), with bar's 0.33, 0.29, with
the 3x boost's 0.54, 0.41 (the presets' own overlap 0.44). **Each clearing under the other's
model** (L_P, SIMP's): the unboosted clearing under the 10x model 0.2273 against the 10x clearing's
0.2610 (-0.034, 13% of Lens A), the 10x clearing under the unboosted model 0.2050 against 0.2438
(-0.039, 16%); the cheap preset's -0.028 and -0.026. At 3x the losses were 3 -- 4%: the 10x boost
changes both which buildings are cleared and what a clearing is worth. The induced corridors' loops
at q 0.9: 6 (SIMP) and 15 (cheap), against 9 and 10 unboosted; at q 0.8 28 and 36 against 54 and 42
(the network's own loops fall too, 797 to 377 for SIMP: the streets carry what lanes carried).
Maps: region_rows/maps/T500b90D10_simp.png, T500b90D10_cheap.png.

**Who each clearing helps** (region_leaks.leave_one_out; summary_loo.csv; a measurement, no
objective changed). Each building of a final clearing restored alone (leave-one-out from it, the
clearing otherwise kept): `helped` the population of the other homes standing whose escape time
rises by at least 5% (and 1%), or that the restoring walls in (none did); `displaced` its own
population; `gain` the rise in P. On the 13 tuning blocks' stored SIMP-default and cheap-preset
clearings under today's per-block metric (cores), and on the region's central SIMP clearing
(T500b90) on a card, in a seeded random order within a 3,000 s budget: 142 of its 707 buildings, a
sample.

    clearing                 buildings  5%: helped < displaced   median    1%: same,    median
                                        (buildings, pop.)        ratio     buildings    ratio
    13 blocks, SIMP          1,541      0.22, 0.16               9.9       0.018        66
    13 blocks, cheap         1,368      0.20, 0.13               10.5      0.016        71
      9712 SIMP; cheap       92; 92     0.55, 0.40; 0.46, 0.32   0; 1.4    0; 0.01      29; 31
      9717                   68; 63     0.40, 0.30; 0.35, 0.30   1.6; 2.6  0.015; 0     29; 32
      5810 (the kblock)      292; 260   0.56, 0.36; 0.49, 0.31   0.6; 1.1  0.04; 0.03   25; 30
      22422                  99; 97     0.32, 0.20; 0.25, 0.11   3.1; 4.6  0; 0.01      45; 53
      the other 9                       <= 0.20, <= 0.10         11 -- 31  <= 0.07      66 -- 166
    5810@major T500b90 SIMP  142 of 707 0.69, 0.49               0         0.11         16

(the per-block rows: SIMP's, then the cheap preset's; the ratio helped / displaced; at 1% the
displaced population's share under is at most 0.02 on every block and 0.06 on the region.) On a
block, the typical cleared building serves about ten times its own population at the 5% threshold
and some seventy at 1%; a fifth of the buildings serve fewer than they displace at 5%, holding 13
-- 16% of the displaced, almost none at 1%. The exceptions are the deep blocks whose clearings are
long corridors (9712, 9717, 5810, 22422): a building in a corridor changes many homes' escape times
by a few percent each, so at 5% it helps no one and at 1% it helps twenty to fifty times its
population. On the region it is sharper: escape times there are long (a formal home's median 1,604
under the central constants), one building's restoring moves most homes' by under 5%, and 69% of
the sampled buildings (49% of their population) pass the 5% test for fewer than they displace, the
median helping no one; at 1% 11% do (6% of the population), the median serving 16 times its own. So
whether a clearing is unfair in the owner's sense depends on the threshold more than on the
clearing: a building's clearing serves many people a little rather than a few people a lot
[inferred from the two thresholds]. The gain (the rise in P when a building is restored) follows
the population helped (correlation 0.81 on the blocks' SIMP clearings at 5%).

**The guards** for this step (both ways at the final code): `street-boost` (above),
`exits-identity` (48 names of 48 configurations and presets, 58 stored names their labels', the
per-block metric's 28 constants digesting to 34d40e8a; faults `in-code`, `suffix-dropped`). mypy
--strict clean on gap_exits and region_leaks; `ruff check .` passes; tests/test_clearings.py: 36
pass. The leave-one-out has no guard of its own [its homes' rise is read from Model.home_u and
live_homes_of, which the `reach` and `baseline` guards cover; inferred].

**Compute.** GPU 2.39 GPU-h of the 3: the 10x runs 0.79 (SIMP) and 0.61 (cheap), their prior 0.09;
the region's leave-one-out 0.91 (142 buildings in its 3,000 s: about 21 s each on a V100 with the
host's work, so the 707 would have taken about 4 h). CPU here, at most 4 processes: the 13 blocks'
leave-one-out about 3.5 CPU-h, the 10x baseline, scores and cross-scores about 1, the report and
the guards about 0.5.

### Before the consolidated pass: three fixes

BACKLOG, the two "Before that pass" entries (the brief: .superpowers/sdd/2026-10-16-prepass-fixes).
A pass over every block must not stop on a false error or lean on a guard whose failure nobody
understands: the refined solve's false stall on ZAF.9.1.4_1_7283, the `gradient` guard on 20269 and
the `free-roads` guard on 23597. Two were real defects (the stopping test, the free-first stage's
targets), one the guard's premise; each now has a guard that fails under its fault.

**1. The refined solve's false stall: the stopping test** (gap_factor._residual; guard
gap_clear_check `underflow`). 7283 (under the source's building floor, so built through
common.member_blocks) is a sliver between two streets, 5 -- 16 m wide and 3.5 km long, with a
triangle at its east end that holds the 22 homes (121,190 m^2 in all, its whole boundary the street
ring). Along the sliver the potential falls by an e-fold every 4.6 m or so (2e-8 at 200 m from the
nearest home, 4e-27 at 400 m: a strip grounded on both sides), below the smallest normal number
(tiny, 2^-1022) from 2.43 km and to exactly 0 from 2.47 km: of the 270,614 unknowns 57,342 are 0
and 1,843 subnormal. The region test's diagnosis, corrected: not 0 / 0. The 57,328 rows whose
denominator (|A| |x| + |b|) is exactly 0 were already passed over (`d > 0`). The 728 rows that read
above 1e-8 all have a denominator under tiny, on the underflow front, and the worst read exactly 1:
a row whose only nonzero term is a neighbour's 2^-1074 (the smallest subnormal) has a residual, a
2^-1074 rounded, equal to its whole denominator, and no refinement can lower it (x is held to the
subnormal grid there). "Beside the street" is right: every such row is within 1.6 m of a street
cell. The solve was right all along (its first solve 1.7e-12 of the largest from spsolve's).
LAPACK's xGERFS, whose rule Refined follows, reads such rows as 1 too (scipy's dgesvx: BERR 1 for A
= I, b = 0, and for a tridiagonal system whose solution underflows), but stops there and reports
it; the raise at REFINE_ACCEPT made it an error here.

The fix (gap_factor.py:92, `_residual`, the floor at :117): each term of the denominator counted at
no less than the format resolves. Below tiny a float is held, and a product rounded, to the
subnormal spacing eps tiny, an absolute precision; so x_j counts at max(|x_j|, tiny) and each
|a_ij| max(|x_j|, tiny) at no less than tiny. A row of normal terms is measured as before, bit for
bit (a stored zero's tiny vanishes beside any denominator above tiny / eps); a row on an underflow
front is measured by what the format can hold there. No tolerance moved and no constant changed:
tiny is read from the format inside the kernel, and the code's digest stays 34d40e8a, so the metric
keeps its name. Both factorizations share the test (SuperLU's and cuDSS's factors are each a
`Refined`, as is the rebuild's, gap_flow.factor). 7283 now solves at a backward error of 7.0e-16,
under 4 eps at the first solve, 9.6e-13 of the largest from spsolve's.

**No score changes where the old test passed.** On the 13 tuning blocks (the 4-block sentinel
22422, 30848, 5810 and 9712 among them), each model built twice, under the old test (the guard's
fault, verbatim) and under the new: P0, P_{1/2},0 and J_2,0, the stored SIMP default clearing's P,
P_{1/2}, J_2, B (i) and every home's escape time, and every refined solve's backward-error history
(7 -- 14 evaluations a block, at most 1.5e-15) are bit-identical, and the baselines are road_rows'
stored ones bit for bit. The least nonzero potential on any of them is 1.2e-18 (23597), far above
tiny.

The guard, `underflow`: 7283 meshed afresh, its model builds (every baseline solve refined), the
direct solve has subnormal cells (1,844: the premise, that the block still has an underflow front),
and the refined solve is within UNDERFLOW_TOL (1e-9) of spsolve's: 9.6e-13. Fault `unfloored` (the
old _residual, verbatim): the baseline's solve raises "refinement stalled at a backward error of
1.0e+00". The solve path changed, so the device guards ran on the cards (run pp-dev at 2cbc39ed,
one submit, both ways): `device` within 1.4e-10 (20269) and 5.1e-11 (18895), `perturb` 0.97 and
0.86 off; `device-roads` within 8.9e-11 and 3.7e-11, `perturb` 0.88 and 0.69: rl-dev's before the
fix were 1.5e-10, 5.1e-11, 8.6e-11 and 3.6e-11.

**2. The `gradient` guard on 20269: the guard's premise, not the code** (gap_clear_check
`against_differences`, GRAD_STEPS). At HEAD (and at 72cabeac, the region test found) the guard
failed 7 of its 24 checks on 20269: shy0.5 at the random x (J_1 q 1, J_2 q 1 and q 3: 1.0e-4 --
1.2e-4), shy0.5 at SIMP's start (q 3: 4.1e-4 and 5.6e-4) and close0.125 at the start (q 3: 1.3e-4,
1.8e-4). The evidence: every check of the guard (3 forms, 2 states, 2 powers, 2 q, its 6 buildings)
at steps 1e-3, 3e-4, 1e-4, 3e-5, 1e-5 and 1e-6 on 19570, 38136 and 20269, with the forward and
backward differences beside the central ones and, per building, the distance in x to the nearest
kink of the factors' minimum (a crossing of its term with another's at some point):
- At the random x the disagreement is the solves' roundoff over the step. On 20269 it grows as 1 /
  h below 3e-4 (shy0.5 J_1 q 1: 3.2e-6 at 3e-4, 1.1e-5 at 1e-4, 1.1e-4 at 1e-5, 1.1e-3 at 1e-6): J
  is deterministic (the same value twice, bit for bit) but rounded at about 3e-12 of itself
  [inferred from the 1 / h slope], and on 20269 (J 3.2e4 over 1.42M cells, its picked buildings'
  gradients 23 -- 142) that is 1e-4 of the gradient at 1e-5. No kink is near: the nearest crossing
  lies 2.5e-3 -- 9.7e-2 away in x under shy0.5 and close0.125 (ramp0.5's continuous phi has
  sub-point crossings within 5e-6, each a sliver of one triangle's factor, too small to see). The
  forward and backward differences close on each other linearly in h and straddle the adjoint:
  curvature, not a kink.
- At SIMP's start (x = D everywhere, each picked building in two-way ties: a kink exactly at x0, by
  design, for the fault `tie`) the central difference reaches the tie's shared gradient, the mean
  of the one-sided derivatives, only at O(h), the two sides curving apart: 38136 shy0.5 J_2 q 3
  1.9e-3 at 1e-3, 2.0e-4 at 1e-4, 2.0e-5 at 1e-5. So the start needs the small steps 20269's
  roundoff forbids, and there no single central difference passes: shy0.5 J_1 q 3 is at least
  1.9e-4 off at every step (2.9e-3 at 1e-3, 2.9e-4 at 1e-4, 1.9e-4 at 3e-5, 4.1e-4 at 1e-5, 2.4e-3
  at 1e-6). Over the 72 checks the best single step per building still leaves 2.2e-4.
- At those ties the one-sided derivatives, each extrapolated to h -> 0 from 1e-3 and 3e-4, differ
  by 0.3 -- 72% of the gradient (a real kink), and their mean is the adjoint's within 2.6e-5 --
  5.4e-5 of the largest on every block and form, each alone at least 1.0e-4 off: the adjoint takes
  the subgradient it is meant to (gap_flow.Terms.vjp, a tie's share split equally). The code is
  right.

The fix is the guard's (gap_clear_check.py:497, `against_differences`; GRAD_STEPS at :243): per
building, the closest over GRAD_STEPS = (1e-3, 1e-4) of the central difference extrapolated from h
and h / 2, 2 D(h / 2) - D(h). Where J is smooth it is the derivative to O(h^2), as D(h) is; at a
tie it cancels D's O(h) term, so it is the mean of the one-sided derivatives to O(h^2); and its
steps lie above 20269's roundoff (the sweep's extrapolations from steps 1e-3 and 3e-4 and from 1e-4
and 3e-5, the closer taken: at most 3.6e-5 over the 72 checks). One function now serves `gradient`,
`leak-gradient` and pweight_check's `gradient` (three copies of the loop before; pweight_check had
moved to steps 1e-3 -- 1e-5 against the same roundoff on 20269's P). The guard's blocks: 19570,
38136 and 20269. Now within 1.8e-5 (19570), 3.8e-5 (38136, ramp0.5 at the start) and 2.4e-5 (20269,
24 of 24; 77 minutes on a core); fault `tie` 6.1e-3 -- 0.10 on 19570 and 38136 and 8.0e-3 -- 0.19
on 20269, every start check of each block failing.

**3. The `free-roads` guard on 23597: the stage, not the guard** (free_roads.FreeEnclosed.plan;
guard free_roads_check `free-roads`). The stage itself raised, before the guard's oracle was asked
("the road held for home 272 (12 nodes, 94.6 m) does not free it by the metric's test"; the same at
72cabeac). Its targets are the graph's nodes and edges within half the road's width (less
`target_margin_m`, 0.05 m, for the buffer's polygon) of the ground that frees the home (within
ring_m, 1 m, of its outline, or its sealed pocket). But the metric carves a road as the footprints
less its corridor, grown by the width form's growth (the shy distance's w / 2, 0.25 m): the
footprints the corridor cuts grow back into it by 0.25 m, and its open ground starts that far
inside its edge. The targets did not know it. Home 272: the corridor comes within 0.80 m of its
outline (0.15 m^2 of it inside the 1 m ring), but its open ground, the complex rebuilt with it
carved, within 1.05 m (0.80 + 0.25): outside the ring, the home's ring still in its sealed pocket.
A mismatch between the routing's reach and the complex's, so the stage. The fix
(free_roads.py:208): the targets' reach less `metric.form.grow` (0 under Closing). A point within
half the width less the growth and the margin of the road lies at least the growth beyond every
footprint left standing, so it is open, and the corridor's open core (6.5 m wide) joins it to the
street: the road frees the home by construction.

**What it changes** (the stage before and after on the 39 of the 60 large blocks with homes walled
in on the complex, 234 homes, on clearance_looped's graph: the guard finds the three graphs' plans
the same on its blocks): before, the stage raised on 7 of the 39 (6310, 22267, 23597, 30796, 30848,
39240, 41515), so the consolidated pass would have stopped on each; after, it raises on none and
frees all 234 homes with 173 roads, none unreachable or over the budget. Of the 32 plans it made
before, 26 are unchanged and 6 change: five by a target the old reach allowed where the corridor
happened to border open ground rather than a cut footprint (the road freed its home but is excluded
now: D area +0.0001 -- +0.0003, 1558, 20080, 38528, 43708, 8235) and one cheaper (17608, two roads
for three, -0.0003). Not pursued: a target test exact to the footprints the corridor cuts, or
routing with the old reach and re-routing when the held road fails its check, would recover those
five, at most 0.0003 of D area a block against the stage's budget of 0.05.

With the fix, 23597's six walled-in homes (45, 93, 272, 375, 765, 889) are freed by five roads (D
area 0.0027, count 0.0107), each by the guard's oracle (the flow model's reach on the fixed mesh
with the roads carved), nothing left walled in; under `one-short` the stage raises on 23597 (home
45's road) and 18895. 23597 joins the guard's blocks (20269, 18895, 23597).

**Stored road results affected: none.** road_rows (item 4's lineup on the 60) hold no held roads
(every arm holds nothing: reblock.free_first.HoldNothing), and the only stored networks with held
roads (transparency_rows/free and free_local, 444 files) were written on 2026-10-12 by the raster's
stage, before the complex. No stored result was made by the stage on the complex: the fix changes
what the consolidated pass will make (BACKLOG 5's TODO, free first the road methods' default), on
13 of the 39 blocks above.

**The guards** at the final code (0d01da48), each passing on it and failing under each of its
faults, on a core here unless named (90 runs, 5.1 h of core time, two to six at once, besides
20269's `gradient` pass above): gap_clear_check baseline, weights, reach, gradient (19570, 38136;
20269 above), free, identity, convex, pattern, constants (28 constants, the digest 34d40e8a),
roads, flux, leak-assembly, leak-fold, enclosure, leak-gradient (19570, 38136: within 1.1e-5
through the shared check; `adjoint-unleaked` 0.062 -- 0.97), leak-scale, barriers, exits-identity,
loop-index, trip-calibration, closed-edge, street-boost and the new underflow; gap_check conform,
delaunay, apertures, filtration, free; gap_flow_check angle, shift, channel, field, law, closed;
pweight_check identity, total, displacement, gradient (within 5.2e-6; `J-population` 0.22 -- 0.37),
weights; free_roads_check free-roads (20269, 18895, 23597); cluster_check ids and cpu. The device
guards on the cards, above. mypy --strict clean on gap_factor and free_roads; gap_clear_check and
pweight_check at their counts before (10 and 6 errors); `ruff check .` passes;
tests/test_clearings.py: 36 pass.

**Compute.** GPU: 0.23 GPU-h of the 1 (pp-dev: 6.6 and 6.9 minutes on an H100 and a V100). CPU
here, single-threaded, at most 8 processes: the 7283 diagnosis and the no-change check on the 13
(about 0.4 CPU-h), the gradient sweep (9 runs, about 3), the stage before and after on the 39
(about 3.5), the guards (6.4, 20269's `gradient` pass among them) and the first runs at HEAD (about
1): about 15 CPU-h [inferred from the runs' times].

### The consolidated pass: every block under the final metric

BACKLOG 4 and the owner's decisions of 2026-10-13 -- 16 (the brief:
.superpowers/sdd/2026-10-16-consolidated-pass). RUNTIME is now the one metric for every block and
region:

    cx+shy0.5+s0.125r1b1+R0.5g4x2q4+v34d40e8a+xB8c3Lo6214b562S10hliving_street-motorway-primary-residential-secondary-service-tertiary-trunk-unclassifiedva03a092b

the per-block metric (the shy distance at w 0.5 m on the gap complex, its 28 constants digesting to
34d40e8a, the name every per-block row keeps) with the region model's exits, each spelt in the name
and read back by gap_exits.exits_of: the railways and water as barriers 8 m wide, open at real
crossings 3 m wide (`B8c3`); the edge open, a block's street ring its exit; the leaks' central
constants (`Lo6214b562`); 10x on the drivable streets (`S10h<classes>`). It is the region test's
T500b90D10, so the region's two runs under it are RUNTIME's. Plan names lost their `.t<rtol>`:
SIMP's default is `fw0.q3.i10.e0.0001.k1s1e-06.p256w8x2`.

**Phase 1: what changes no score.**
- relax.Plan's `.t<rtol>` dropped (the raster's solve tolerance; every solve on the complex is
  direct). Every stored run on the complex migrated (a scratch script): the rows' directories
  (relax_rows/<cx metric>/<plan>, gap_rows/simp*), the `plan` column of every complex SIMP row (369
  files and directories), the complex runs' ids in road_rows, cross_rows and the committed store
  (clearings/: two runs' files and their index lines). The store reads back identically (all 157
  runs' tables equal, the two complex runs under their new ids; a re-export of the migrated rows
  into a copy of the store reproduces every run). The raster's runs keep their names, `.t` and all:
  records, where `t` was a real setting (relax_rows/uni/fw0.q3.i20.t0.001 and .t1e-05 differ only
  in it). plan_of refuses a `.t`.
- The optimizers' no-overwrite guard: clearings.write_once writes a run's rows whole and only once
  (a hard link into place: atomic, failing if a row stands), clearings.check_new refuses a finished
  identity before any work; a row's path is its identity (block, preset or method, the metric's
  full name, budget, the plan's seeds); `--replace` names a deliberate replacement. Every optimizer
  writer goes through it: relax.py (its silent skip of finished blocks gone), polish_greedy, search
  (its skip gone), gap_calib `preset`, region_leaks `run`, pweight `task`, road_rescore `cx`. Guard
  clearings_check `overwrite` (on 38136, SIMP and the greedy + polish each refused on a second run
  with the first row's bytes kept, replaced under `--replace`; fault `no-check`: both second runs
  write over the first) and a unit test (the same fault).
- Every writer keyed by its metric. Already: relax, polish_greedy, search, pweight, road_rescore
  `cx`. Fixed: gap_calib (gap_rows/runtime/<METRIC's name>/, its rows moved under the per-block
  metric's), region_leaks (its runs under ROWS/runs/<label>/<preset>/<metric>/, all 23 moved, each
  row's metric checked; `members`, `blocks` and the leave-one-out pinned to the per-block metric,
  region_leaks.per_block, which the switch would otherwise have changed under them). Found on the
  way: the region test's migrations had left relax.one's `along` at the old names (U1000, H500, the
  3x boost without its classes) on 7 SIMP rows (migrated to their metric), and the exporter raised
  on region_rows since the region test (its runs are now a family, clearings_export.RegionFamily;
  the 23 region runs committed, 248 KB).
- The finished studies' code that read no raster code deleted: fft_homog's solver and two-scale
  code and every driver (29 modules), perm_floor, atlas/nbhd, the region page's style and vectors.
  Nothing imported them; their numbers and committed summaries stay. Tracked Python under
  research/roadless: 71 modules and 22,192 lines to 39 and 18,774. pyamg is now read by no code
  (fft_homog's solver): its removal, a dependency change, is left for the owner.
- BACKLOG's raster-era entries closed, each with a pointer to the wiki's
  `pages/reblock/raster-escape-time-metric-retired.md`: the composite and goal meshes, local
  coarsening, sightline (it went with the raster: item 3's follow-ups and SIMP under it), the cheap
  preset on the composite, coarse-to-fine, the Galerkin coarse model, the greedy's sweep kernel and
  speed, SIMP's memory, the restore screen's iterative solve, the raster's GPU sizing. Kept,
  applying on the complex: the add/remove search and swings, `.b`, `.u`, `.w`, `.r`, the greedy's
  width and batch, the price per displaced person, adaptive block selection, the epistemetry lead,
  the rate bracket, the CPU runs.
- No score moved: on the 13 tuning blocks the per-block metric's baselines (P0, P_{1/2},0, J_2,0)
  and the stored SIMP default's lenses (L_P, L_{P_1/2}, L_J2, B (i)) equal road_rows' stored values
  bit for bit (every difference 0.0).

**Phase 2: the switch.**
- Every block carries its OSM features (Block.attrs["osm"]) where blocks are built:
  common.build_blocks (so every bank common.write_bank pickles carries them: a cluster node needs
  no extract) and common.member_blocks. One read of the country's extract serves them all
  (regions.Extract: the features whose envelope meets the kblock source's bounds grown by 10 km,
  read once and cached beside the extract, keyed by the file, the CRS, the box and EXTRACT_REV;
  each block's features a query of it, about 1 s). The per-face read (a 36 s scan a call) is gone;
  on 5810@major the cache gave its 551 ways, 66 nodes and 228 areas identical to it. Then the
  features no reader reads were dropped at the read (lines with no highway, railway or waterway
  tag, points with no railway tag, areas that are not water: the province's and the city's
  boundaries met every block and filled a 280-block bank to 420 MB): the barriers and the boosted
  triangles on 3934, 5618, 21016 and 46841 are as before.
- RUNTIME switched in metrics.py (above): PER_BLOCK, the per-block metric unchanged, and EXITS
  (BARRIERS, LEAK, BOOST, DRIVE_CLASSES), each spelt in the name, so no digest moved: the exits
  code stays a03a092b and every stored region row keeps its name; region_leaks takes them from
  metrics (one copy). gap_clear_check's guards of the complex's core check the per-block metric
  (their oracles, the rebuild among them, have no exits).
- Three defects the switch exposed, fixed: (i) a block no barrier meets was returned by a
  difference with nothing (its rings reordered: another mesh key, a mesh the leak-only metric's
  only to floating point), now as it is; (ii) a railway crossing node no way passes through
  (38366's two: the extract has the track alone) raised, and now opens a crossing straight across
  the track at the node, 8 m either side; (iii) a block's local frame is its polygon's lower
  bounds, which a cut at the block's edge moves (3934's railway along its south-west edge), but the
  cut mesh was built on atoms in the uncut frame, the footprints shifted (found by
  Model.free_first's check: the complex and the flow model walled in different homes);
  ComplexMetric.fixed now moves the atoms into the cut frame (Carving.shifted). No stored result
  moved: 5810@major's cut keeps its frame, and FreeEnclosed now routes on the cut block too.
- The guards of the switch (gap_clear_check; each passes and fails under each fault): new
  `no-features` (19570 with its own features less every drivable way and barrier feature, as no
  real block reads: every kblock has a drivable street on its ring): RUNTIME's baseline, every
  home's escape time and the walled-in homes equal the leak-only metric's bit for bit (0.0); fault
  `neighbour-features` (the touching kblock that shares the most drivable street: its features)
  moves P0 3.7%, a home's escape time 91%. `barriers` on real blocks (3934, a railway on its edge;
  5618, a pond on 21% of it): no triangle in the cut under RUNTIME (10,091 and 8,132 under the leak
  alone), the cache key not the uncut block's, the name carrying B8c3, the atoms on the footprints
  in the cut frame, the complex's walled-in homes the flow model's; faults `features-dropped`,
  `unshifted` (besides `not-inserted`, `gap-closed`); the synthetic block's crossing node without
  its path still opens the north-west. `street-boost` on a real block (21016: all 11,986 triangles
  near its drivable way boosted, none of the 6,189 near only a footway or path; passages 10x and
  1x; fault `classes-dropped`). `exits-identity`: RUNTIME is the per-block metric with
  metrics.EXITS, read back, and the region's T500b90D10.
- What the switch touches (consolidated `features` and `baselines`, the store's 302 blocks on the
  cluster's cores, about 42 core-h; consolidated_rows/summary_features.csv, summary_switch.csv).
  Every block has a drivable way on its ring (none lacks one), and inside it on 106 of the 220
  small blocks and all 82 large. A barrier meets 54 blocks (2 small, 30 large, 21 of the 22
  oversized, 5810@major), cutting a median 1.7% of a large block (at most 12.1%, 6316), 0.3% of an
  oversized one, 8.3% and 20.6% of 3934 and 5618. Each part's move of the baseline, relative to the
  part before it (median [min, max] over the 302):

      part           P0                          P_{1/2},0                   J_2,0
      the leak       -0.8% [-9.0, -0.07]         -1.1% [-17.7, -0.07]        -1.8% [-21.8, -0.14]
      + barriers     0 [0, +273%] (53 blocks)    0 [-6.8, +601%]             0 [0, +1,724%]
      + the boost    -12.9% [-60.1, 0]           -4.4% [-60.9, +33.6]        -17.8% [-82.9, 0]
      all            -11.2% [-60.5, +265%]       -4.5% [-64.0, +316%]        -18.1% [-83.3, +896%]

by size P0 -8.5% (small), -15.1% (large), -19.0% (oversized); on the 13 tuning blocks -19.9%
[-42.8, +77.6] (18895's water, 30848's railway and 5810's take it up: a railway or a stream on a
block's edge closes that edge as an exit). The barriers move P0 by more than 10% on 22 blocks and
wall in 189 more homes on 10 (30848: 23 to 117); RUNTIME walls in 792 homes on 121 blocks (the
per-block metric 603 on 118). The boost lowers P0 by over 10% on 164 of the 302: the 3 m strip
along a drivable ring street conducts 10x, and its triangles sit next to the street's faces.

**Phase 3: the lineup and its cost** (the workspace's lineup.md). Every result-changing item
BACKLOG deferred to the pass, on the complex: the three presets, free first, on every block (the
default, the hard-block preset, whose greedy track now frees first too, and the cheap preset), on
the 82 large blocks one card a block and on the 220 small ones a core a block (each device's
polish, phase 2a's: the same clearings); free first for the plain greedy (the hard-block track, and
the cheap preset's own greedy clearing); the oversized blocks with the rest (one fixed mesh a
block, 4.2 -- 8.6M triangles); the road methods with free first their default
(road_rescore.lineup_arms: FreeEnclosed under RUNTIME in place of HoldNothing, resistance_lp
included, the arterial's cap 300 roads in the lineup's config only), on the 82; the measurement
gaps (one-card timing with setup: consolidated `presets` records it; S0.005cat's real clearing with
the polish: `cheap005`; the sightline lineup: closed, sightline went with the raster); the small
blocks' rows (the presets on the 220). Every preset optimizes P; nothing in the lineup optimizes
J_2 (pweight's J2 arm is that study's comparison; seqscreen reads its rows at power 2, POWER, to be
read at power 1 for the pass's screens). Every clearing and network scored on P, P_{1/2} and J_2,
Lens A at D 0.05 and Lens B, on cores.

The estimate, from the last runs' timings on the complex (a five-block GPU pilot: 7851, 5810,
30848, 63612, 4287 at 0.18, 0.74, 0.89, 1.61 and about 1.1 GPU-h for the four presets with their
setup on one card; four small blocks' presets on a core here, 0.5 -- 0.64 s a building; the
survey's meshes; item 4's road timings on the 60; the road pilot, cycle_desire with free first on
9712 3,444 s): about 34 GPU-h (the presets on the 82, GPU-seconds 81 + 3.85e-4 x the mesh's
triangles, fitted on the pilot) and at most 2,700 core-h (central about 1,650: the arterial at 300
roads and cycle_desire on the oversized blocks bounded by their tasks' 23.5 h wall limit, a block
not done within it reported). Within the caps (40 GPU-h, 3,000 core-h): on to the runs. The lines
to cut first, were the owner to cut: the arterial on the 22 oversized blocks, cycle_desire on the
five over 20 km^2 (150 -- 520 GB each), `cheap005` on the oversized blocks (dropped from their
runs, about 10% of their GPU line).

**Phase 4: the runs.** Every preset on every block under RUNTIME (consolidated `presets`: the large
and oversized blocks on one card each, cp-pilot-gpu and cp-presets-large / -oversized, 20.5 GPU-h;
the 220 small ones on a core each, cp-presets-small-r1, 12.5 core-h), every clearing scored on a
core (consolidated `lenses`); the store gained the 8 runs (clearings/, beside the old ones). The
cards were what the queue gave, a block's presets on one: V100 (49 tasks of the pass's GPU runs),
RTX 8000 (48), H100 (15). Lens A at area D 0.05 (mean; median), and the presets paired:

    L_P          small (220)       large (60)        oversized (22)
    default      0.4253; 0.4156    0.3549; 0.3628    0.1663; 0.1446
    hard         0.4259; 0.4156    0.3551; 0.3628    0.1664; 0.1453
    cheap        0.4163; 0.4069    0.3452; 0.3562    0.1617; 0.1415
    cheap005     0.4170; 0.4079    0.3470; 0.3571    (3 blocks)

    a - b, the 302     L_P: mean, median (ahead / behind)  L_{P_1/2}           L_J2
    hard - default     +0.0005, 0 (15 / 0)                  +0.0007 (12 / 3)    +0.0007 (12 / 3)
    cheap - default    -0.0088, -0.0064 (51 / 245)          -0.0078 (66 / 230)  -0.0079 (54 / 242)
    cheap005 - cheap   the 60 large +0.0018, +0.0006 (38 / 22); the 220 small +0.0007, 0 (41 / 38)

The hard-block preset is the default but where its greedy track wins (15 blocks, 13 of them small,
never behind on L_P; behind on P_{1/2} and J_2 on 3, by what P bought). The cheap preset trails the
default by 0.0088 L_P (0.0097 on the large blocks, ahead of it on 1 of 60; 0.0046 on the oversized,
on none), and is ahead on 50 of the 220 small blocks. In B's order (the least walled in first, then
the lens; ties count for each) the hard-block preset is first on 257 blocks, the default on 245,
the cheap preset on 52 (L_P). Free first leaves homes walled in on 6 blocks (4 large, 2 oversized:
beyond the free stage's reach within the budget), the same for every preset. Time per block
(median; the four presets one after another in one process with their setup, the first preset
carrying the block's mesh): large on a card 239 s for the default (setup in), 107 s the hard-block
preset, 69 s cheap, 73 s cheap005, 488 s the four; oversized 863, 280, 157, 226, 1,287 s; small on
a core 55, 60, 14, 17, 146 s.

The measurement gaps. One-card timing with setup: above
(consolidated_rows/summary_lineup_onecard.csv). S0.005cat's real clearing with the polish
(`cheap005`): on the 60 large blocks +0.0018 L_P over the cheap preset (median +0.0006, ahead on
38, behind on 22), +0.0030 L_{P_1/2}, +0.0030 L_J2, at 73 s against 69 s; on the 220 small blocks
+0.0007 (41 / 38) at 17 s against 14 s. The record's interpolated +0.0046 (44 blocks, J_2, the
raster) is +0.0018 as a real clearing polished under P: both are on the frontier (the finer batch
better, slower), and whether the cheap preset takes it is the owner's call. The sightline lineup:
closed (sightline went with the raster, phase 3).

**The road methods under free first.** road_rescore `lineup-one` on the cluster's cores, one task a
block and method: the network proposed by road_rescore.lineup_arms (FreeEnclosed under RUNTIME the
stage of clearance, clearance_looped, cycle_native, cycle_desire and resistance_lp; the grid holds
none), its canonical prefixes, its area prefixes scored on the complex under RUNTIME on a core;
rows under road_rows/lineup-caf14fa2 (lineup-f8fa4e53 since Task 7, the last section), the digest
of the stage's identity. The runs submitted before 2d5df972 wrote lineup-e3b882f9 and were moved
file for file (five networks of 30796 and 23597 proposed again at HEAD were identical WKB for WKB);
30 of their rows, which the cancelled reused run finished on the cluster after the same identities
were scored here, differed only in timings, host and peak memory and were dropped for the standing
row. The 21 large blocks where RUNTIME walls no home in keep item 4's networks (HoldNothing),
scored under RUNTIME. cycle_desire's cluster requests were sized from its measured peaks scaled by
area (8.8 GB/km^2 through the origin over the 16 tasks above 1 km^2, 4.5 -- 10.9 per task; 1.8 --
3.5 h/km^2 on the oversized), and 7499's ran on a core here (14.9 h, 83 GB at 8.4 km^2). Scored:
482 of the 492 (block, method) of the 82 blocks and six methods (clearance, clearance_looped,
cycle_native and the grid on all 82, resistance_lp on 81, cycle_desire on 73). Not scored: the two
carved meshes refused (below), and cycle_desire on the eight largest oversized blocks still on
orion (62441, 62803, 63612: job 79684; 4287: 79686; 1182, 472, 63818, 64070: 79687), which a
follow-up adds (the workspace's followup.md).
- Free first holds roads on 42 of the 82 blocks, for every stage method (cycle_desire on 41 of its
  networks so far); elsewhere the networks are HoldNothing's.
- At area D 0.05 (medians over the blocks each reaches it on; L_P, L_{P_1/2}, L_J2): resistance_lp
  0.127, 0.139, 0.135 (81 blocks); cycle_desire 0.126, 0.143, 0.131 (67); cycle_native 0.117,
  0.137, 0.130 (72); clearance 0.111, 0.130, 0.109 (65); clearance_looped 0.099, 0.112, 0.101 (82);
  the grid 0.086, 0.100, 0.095 (82). clearance reaches D 0.05 on 9 of the 22 oversized blocks,
  cycle_native on 13, cycle_desire on 9 of its 14; the rest on all.
- Each against SIMP's default, paired on the blocks the method reaches D 0.05 on (median, L_P):
  resistance_lp -0.136, clearance_looped -0.179, the grid -0.189, cycle_native -0.195, cycle_desire
  -0.196, clearance -0.225. No road method is ahead of SIMP's default on any block under any lens.
- The price of roads in B's order: SIMP's default is ahead of the best road network at D 0.05 on
  all 82 blocks under every lens. B (i) ties on 79; on the other 3 (30796, 30848, 45876) the best
  road leaves more of the population walled in (by 0.00005 -- 0.00046 of it). Where (i) ties SIMP's
  default leads by a median +0.161 L_P on the 57 large blocks (p10 +0.081, p90 +0.242, at least
  +0.049) and +0.077 on the 22 oversized (p10 +0.043, p90 +0.119, at least +0.034); the best road
  gets a median 50% of SIMP's L_P on both (p10 40%, p90 64%). The best road under L_P:
  resistance_lp on 50 blocks, cycle_native 13, clearance_looped 9, cycle_desire 7, the grid 2,
  clearance 1 (under L_{P_1/2} resistance_lp 43, cycle_desire 12).
- The oversized blocks: every lens drops with size (the roads' L_P medians 0.04 -- 0.07 on the
  oversized against 0.11 -- 0.17 on the large), and SIMP's lead with it, still on every block;
  resistance_lp comes closest there (-0.078 L_P against -0.187 on the large).
- The switch's effect on the order (Kendall tau per block between the road methods' order in B's
  order at D 0.05 under RUNTIME and item 4's under the per-block metric, on the 60 large blocks,
  the methods both have): median +0.60 under L_P (IQR +0.33 -- +0.87), +0.60 under L_{P_1/2} (+0.20
  -- +0.87), +0.47 under L_J2 (+0.20 -- +0.87), at least -0.73. The barriers, the boost and free
  first reorder the methods on a block; resistance_lp leads before and after.
- Cost (each network's proposal, median hours this pass): the grid 0.0, clearance 0.03,
  clearance_looped 0.04, resistance_lp 0.05, cycle_native 0.08, cycle_desire 0.46 on the large
  blocks and 10.9 on the oversized (17.9 h at most).
  consolidated_rows/summary_lineup_{roads,price,held,tau}.csv.
- **The arterial: superseded** (owner 2026-10-19). The lineup's old arterial (the access arterial
  capped at 300 roads, no displacement stop) gives way to research/arterial-speed's (an incremental
  exact gain, a stop at the ramped destroyed-area D 0.20, the lineup's scoring under that measure),
  whose Task 7 re-proposes it on every lineup block and re-scores every road method's prefixes
  under the new D. Its rows here are a record, read by no comparison (since Task 7,
  road_rows/superseded/lineup-caf14fa2): 70 of the 82 blocks scored (66 reaching D 0.05), a median
  8.2 h a proposal and 34.9 h at most; 17 of its tasks met the cluster's 23.5 h wall (8 oversized,
  9 large) and were being re-run on cores here when the ruling came.

(Roads: two networks could not be scored: the carved mesh is refused as not Delaunay, 39240's
cycle_desire (2 passages) and 64070's resistance_lp (1). On 39240 three vertices of the corridors'
cut lie within 0.3 mm (62.2045, 67.711; 62.2046, 67.711; 62.2049, 67.7111 in the block's frame),
and a sliver of 9e-8 m^2 with edges of 0.1 and 12 mm, inside one face, is above the flat test's
1.5e-8 at that edge (FLAT_REL L^2), so its neighbour's centres come out in the wrong order;
conformity splits only face boundaries. A defect of the carving, for after the pass: snap the cut's
vertices within a tolerance, or conform a constrained edge inside a face, then score the two.)

**Phase 5: pruning.** The speed-quality review of 2026-10-12 (a read of the record for the owner's
question on SIMP, the greedies and the cheap preset; its report in the session, not in NOTES)
listed 17 variants beaten on quality and time under J_2 on the raster: `.r2`, `.r2l2` alone,
`.p64w8`, `.pP`, `.p256w16x2`, seeding `.g`, the ungated two-track, `.c1`, `.k2`, MMA, damped OC,
`.m2`, `.C`, `.p256w8.c0.75`, floating search, grow-then-prune and swings from the greedy. On the
complex: `.c1`, `.c0.75` and `.C` went with the raster; `.r2`, `.p64w8`, `.p256w16x2` and `.k2` are
values of kept parameters (the roundings in every stage, which `.r2l2` restricts to the last two;
the polish's cap and width; the incumbent's stride), no code of their own and no preset selects
them; floating search, grow-then-prune and swings from the greedy are configurations of search.py
(floating search's niche, the whole curve in one run, is not what a one-budget screen measures;
grow-then-prune is its faster, weaker sibling on that curve; a swing from the greedy is the
finisher's code from another start); the ungated two-track keeps the better of SIMP's polished
answer and the greedy's polished track, so the gated one (the hard-block preset) is never ahead of
it on P by construction. The six with code of their own were screened under P, each in the
default's company, against SIMP's default re-run beside it on one card (consolidated `screen`; rows
screen_rows/, in the store): on seqscreen's held-out order, 15 blocks, then as its stop said
(cp-screen-w1, -w2a, -w2b: 6.3 GPU-h). Paired against the default (variant - default; Lens A at
area D 0.05 under RUNTIME; the time per block on one card, the ratio's geometric mean with its 95%
Bayesian-bootstrap interval; ahead / behind by more than 1e-9):

    variant                blocks  stop     L_P                L_{P_1/2}          L_J2
    .r2l2 (late roundings)     20  ask      +0.00013 (2 / 0)   -0.00031 (1 / 1)   -0.00005 (1 / 1)
    .pP (two-phase polish)     15  keep     +0.00001 (1 / 0)   -0.00005 (0 / 1)   +0.00004 (1 / 0)
    .gS0.01cat (the seed)      15  keep      0 (0 / 0)          0 (0 / 0)          0 (0 / 0)
    .m2 (a second start)       15* keep      0 (0 / 0)          0 (0 / 0)          0 (0 / 0)
    .umma0.2 (MMA)             15  keep     -0.00017 (6 / 8)   -0.00085 (5 / 9)   -0.00050 (7 / 7)
    .uoc0.3m0.2 (damped OC)    15  keep     -0.00068 (0 / 15)  -0.00066 (7 / 8)   -0.00063 (6 / 9)

    time on one card (variant / default)   .r2l2 1.24 [1.19, 1.30]   .pP 1.12 [1.09, 1.15]
                                           seed 1.25 [1.21, 1.29]    .m2 1.64 [1.59, 1.69]
                                           MMA 1.04 [0.99, 1.08]     damped 1.01 [0.94, 1.09]

(Every stop settled but `.m2`'s, which ran out: `.m2` is not built where free first holds
buildings, relax.built_for_holds, so its screen ran on the 15 of the 46 held-out blocks where it
holds nothing, all of them.) The rule (a variant goes only if another beats it on every axis a user
could care about and ties on none; absent numbers it stays) deletes none: `.r2l2`, `.pP` and MMA
are ahead of the default on some block (`.r2l2` by +0.0025 on 44602, MMA by up to +0.001 on 6 of
15); the seed and `.m2` give the default's clearing on every block (a tie on every lens, at 1.25x
and 1.64x the time); damped OC trails on every lens's mean (L_P on all 15: -0.00068 [-0.00098,
-0.00044]) at the default's time (a tie: 0.94 -- 1.09). The GPU's memory is not an axis here: the
rows' gpu_gb is the device's memory in use, which counts the pool the runs before it in the process
left (the default always ran first). Three are beaten on everything the rule reads but a tie: the
seed and `.m2` (quality) and damped OC (time); deleting them is the owner's call on a weaker
reading of the rule. `.r2l2`'s screen settled at 20 blocks on "ask": is +0.00013 Lens A of P (0.04%
of the default's mean) worth 1.19x its GPU time (17 s a block)? A yes puts RATES' top at its
break-even, 7.9e-6 Lens A per GPU-second, a no its bottom.

**The guards on the final code.** Every guard both ways here on cores (96 runs, 10.6 process-h at
68057a29; gap_clear_check's `identity` and `roads` again at 892f4ecd, road_rescore's last change;
all 96 again at 170a2525, the pass's last code, 11.9 process-h, every one as expected):
gap_clear_check's 24 guards on cores (55 runs with their faults), gap_check 5 (11), gap_flow_check
6 (12), pweight_check 5 (10), free_roads_check `free-roads` (with `one-short` and
`through-barriers`), clearings_check `overwrite` (`no-check`), cluster_check `ids` (`no-check`) and
`cpu`: each passes, and fails under each of its faults. The device guards (`device`,
`device-roads`) both ways on a card on 20269 and 18895 (cp-dev2 at 68057a29): cuDSS against SuperLU
within 1e-10 of the largest everywhere, the fault `perturb` caught (0.6 -- 0.97). The sweep found
one guard failing without its fault: pweight_check `weights`, its oracle with no leak while
pweight's metric (RUNTIME) now leaks (up to 9.9e-3 off); the oracle now adds the runtime's internal
leak from its definition (gap_clear_check._leak_oracle): within 2.2e-11, the fault `swap-time`
still caught. And the free-roads guard's `one-short` fault never ended under the retrying stage
(the shortened road's end was not its target, so the same road came back): the retries are keyed by
the target and counted, no network changed (2d5df972; five networks of 30796 and 23597 proposed
again at HEAD, identical WKB for WKB to the runs'). pytest tests/test_clearings.py: 37 pass. mypy
--strict: clean on every clean module the pass touched (atlas, clearings, clearings_check,
clearings_export, consolidated, free_roads, free_roads_check, gap_exits, metrics, polish_greedy,
region_leaks, search, tests/test_clearings.py); the others hold their errors at the pass's start
(common 12, gap_calib 8, gap_clear 32, gap_clear_check 10, pweight 4, pweight_check 6, regions 5,
relax 11, road_rescore 18) but gap_complex (25 against 24: origin_of's untyped block, left so, its
source in the free stage's identity). ruff check: clean.

**Compute.** GPU: 27.1 GPU-h of the 40 (the presets 20.5 with the pilot, the screens 6.3, the
device guards 0.26). The cluster's cores (orion and quadro2; sacct, each task's wall, one core a
task): 1,582 core-h by 2026-10-19 09:30 UTC, the road lineup 1,550 of it (the old arterial's runs
1,081: cp-roads2-arterial-large's 726 and the 22 oversized arterials' 355 of cp-roads2-oversized's
401), the presets on the small blocks 12.5, the survey 17.6, the lenses 1.7; cycle_desire's eight
still on orion add at most about 790 (their walls) [inferred]. Here, single-threaded: the old
arterial's re-runs 267 process-h (four finished in 14 -- 35 h each, eight killed at the owner's
ruling after 3 -- 38 h), 7499's cycle_desire 14.9, the guards 10.6 and 11.9, the reused blocks' and
the last lenses' scoring before them. Within the caps (40 GPU-h, 3,000 core-h).

### The arterial: an incremental gain, a cap that holds, a displacement budget (2026-10-18)

The plan (docs/superpowers/plans/2026-10-18-arterial-speed-and-budget.md; branch
research/arterial-speed, Tasks 1 -- 6) took the lineup's arterial
(greedy_arterial_access_displacement) from a road-count budget that never bound for the method (55
of 62 lineup networks at 300 commits ended with every parcel at depth 1, a median 54 road km/km2)
to a displacement budget, and made a step cheaper first.

**Before** (arterial_rows/before/: the brief's microbench on 20269): one step grew from 10 s with
nothing committed to 51 s at 607 segments; the exact evaluation of the 512 shortlisted candidates
was 8.8 of the 10.4 s at S 0, the access gain alone (a full re-peel per candidate) 9 -- 51 ms of
the 17 -- 54 ms each.

**The incremental gain** (Task 1, objectives._AccessStep): a snapped road's gain lowered from the
step's own peel (a breadth-first relaxation from the parcels it fronts, exact integer burdens) with
liveness one dwithin test against the step's seed (streets and committed roads, every committed
road live, asserted). Exact: every realized candidate of three greedy steps on 1808, DJI.1_2_602
and five fixture blocks equals the re-peel (==), and the guard arterial_check.py gain on 20269 and
38616 at 0/50/150/300 stored segments found 0 mismatches in 4,096 shortlisted candidates; the
faults always-live, no-min and streets-only each fail it. The gain is 1.0 -- 2.0 ms per candidate
at S 0 and 0.25 -- 1.1 ms after, against 7.7 -- 40 ms for the re-peel. The same 40 commits as
before on 20269, 6310 and 39240 (120 of 120, WKB for WKB), the steps' total time 455 -> 182 s, 608
-> 224 s, 491 -> 187 s (2.5 -- 2.9x: snap, ~3 ms, and the displacement cost, ~4.6 ms, now dominate
a candidate).

**The sampler** (Task 2, anchors.py): AnchorSampler as a Strategy, Uncapped (the old max_anchors:
0), PerLineFallback (the old capped sampler, moved unchanged) and HardCap (at most max_anchors
samples at one global arc-length stride). On 20269's stored network at 0/150/300/450/607 segments
PerLineFallback(32, 128) gives 112, 229, 323, 409, 525 anchors (its cap restarts on every line),
HardCap(32, 128) 112, 128, 128, 128, 128 (guard arterial_check.py anchors, fault per-line).

**The stop** (Task 3, stop.py; superseded the same day by the ramped destroyed area, next
section): a required keyword-only max_displacement; the greedy stops before
the step that would start with the committed roads' destroyed footprint share at or past it (a
touched building counted whole, the owner's definition of 2026-10-18), the crossing road kept, so a
stopped run is a commit prefix of a longer one; every proposal records why it stopped
(max_displacement, no_gain or max_roads). Every arterial preset at 0.20; the access pair's
max_roads 1000 a safety cap; road_rescore's ARTERIAL_MAX_ROADS deleted and the lineup's rows root
hashes the arterial's whole identity. Under the destroyed-area share a budget of 1.0 binds once
every building is touched (1808's ten do), so 'no budget' is math.inf at the engine. The directness
presets take 0.20 too: on a block as small as 1808 it binds after one road.

**The study** (Tasks 4 -- 5, arterial_study.py; orion, cluster run arterial-study-b, 12 single-CPU
tasks, all passed, 19 min wall, 1.7 task-h with scoring; rows arterial_rows/study untracked,
summaries arterial_rows/summary_{runs,lenses,prefixes}.csv). cur02 = PerLineFallback(32, 128),
cap02 = HardCap(32, 128), both at 0.20 destroyed and 1000 roads:

    block      km2   stored h  cur02 s  cap02 s  commits  scored D_area at the stop
    6310       0.084   4.04      257      283     23 / 28  0.072 / 0.067
    20269      0.103   2.81      143      126     16 / 14  0.044 / 0.046
    40144      0.153   4.29      242      285     19 / 24  0.055 / 0.053
    39240      0.208   2.98       84      168      9 / 20  0.040 / 0.046
    8480       0.387   5.46      726      579     50 / 62  0.026 / 0.018
    38616      0.461  12.69      847      625     41 / 41  0.043 / 0.040

(Proposal walls; the tasks with prefixes and complex scoring took 3.6 -- 17.3 min. Task 5's commit
subject says 0.95 h and 0.83 h; the proposal walls sum to 0.64 h and 0.57 h, the tasks to 0.89 h
and 0.82 h, against the stored 32.3 h.) Every run stopped on max_displacement (no max_roads stop).
Median step 8.8 -- 18.6 s (cur02, anchors up to 304) against 8.4 -- 15.3 s (cap02, 128).

**Destroyed area runs far above the scored area D** (the plan's Ruling 1, measured): the stop at
0.20 destroyed sits at scored area D 0.026 -- 0.072 (cur02) and 0.018 -- 0.067 (cap02), 3 -- 11x
below; the scored budgets 0.10, 0.15 and 0.20 are unreached on every block, 0.05 on four of six
(20269, 39240, 8480, 38616), and 0.02 on 8480 under cap02. The stored whole networks show the same
gap (6310: 0.854 destroyed against 0.448 scored). Lens A at 0.05 and 0.10 cannot be read off the
arterial at this budget: the owner's call whether the scored D follows the destroyed-area
definition or the arterial's budget rises.

**Verdict** (the plan's Ruling 7): neither sampler dominates. cap02's wall is 0.74 -- 2.00x cur02's
(faster on 20269, 8480, 38616; slower on 6310, 40144 and 39240, where it needs more commits to the
same budget); at D 0.05 only 6310 and 40144 reach the budget in both arms, cap02 behind on 6310
(L_P -0.061) and ahead on 40144 (+0.024), median -0.019 L_P, -0.012 L_Pq, -0.015 L_J2; no block
reaches 0.10. Both stay: greedy_arterial_access_displacement keeps PerLineFallback (quality-first)
and greedy_arterial_access_displacement_fast takes HardCap (speed-first: a flat per-step cost,
ahead on the two largest blocks); the repulsion twin keeps the quality-first sampler.

**The baseline is not reproduced.** The fidelity checks fail: cur02's network lies 1.7 -- 8.7 km
outside the stored lineup network (lineup-caf14fa2), from the first commit, so the prefix movement
(summary_prefixes.csv: symmetric differences 0.5 -- 3.7 of the baseline prefix's length) and the
lens deltas measure the stored network against a fresh one, not the stop. The stored networks came
from run cp-roads2-arterial-large at 0f4be69e on ai05 (quadro2), whose code, lock and conf match
18044cbb; today this host, orion and a probe on ai05 itself (3 commits on 20269 and 6310) agree
commit for commit with each other and with the pre-change tree, and none gives the baseline's first
commit (on 20269 the step-0 leader leads by 9% and none of the top 12 candidates lies in the
baseline). Checked and not the cause: the run's own bank block (identical parcels, streets and
footprints), dependency versions, the derive path, the shortlist's ties at its cut, the checkout's
editable install. Open; Task 7 re-proposes the lineup's arterial in any case.

**The migration** (Task 7, owner-gated): re-proposing the arterial on the lineup's 62 blocks at the
default preset. Against the stored walls (612.6 h for 62 networks at 300 commits) the study's
proposal walls are 0.8 -- 3.7% (median 1.7%): 5 -- 23 task-h of proposal (central ~10), plus the
scoring the stored rows took (8.1 h of complex and prefixes, its largest task 2,609 s at up to 64
GB): ~13 -- 31 task-h, central ~18. The fit in area alone (0.01 + 0.59 h per km2) would give 128
task-h, driven by the oversized blocks (up to 51.7 km2) that the six blocks (0.08 -- 0.46 km2) do
not cover: those are the uncertain part.

### The research lineup's displacement: the ramped destroyed area (2026-10-18)

The owner's change (2026-10-18, "now, before the full rerun"), after the arterial study above
showed the destroyed-area stop landing at scored D 0.018 -- 0.072: one rule for every building,
the research lineup's displacement from here on, in place of both the scored partial overlap and
the arterial's touched-building stop. The destroyed portion of building i under a corridor C is
X_i = 1 - area(largest piece of O_i - C) / area(O_i), so a corner clip destroys the clipped share
and a road through a building destroys the corridor plus every piece but the largest; building i
counts c_i = min(1, X_i / tau) destroyed, tau 0.10 (0 -> 10% destroyed ramps 0 -> 1); D = sum
c_i w_i / sum w_i under both populations (count, area) as before. A building with no area is
destroyed iff the corridor covers its point. It lives in one module, src/reblock/destroyed.py
(RampedDestroyed over a whole corridor; IncrementalDestroyed one piece at a time, each building
keeping its covered region and only the buildings a new road touches recomputed), used by
common.displacement and the prefix cuts (common.MEASURE = RampedDestroyed(tau=0.10), a required
keyword-only `measure` down through prefix_to, road_rescore.prefix_len and FreeEnclosed) and by
the arterial's stop (stop.Destroyed(share, tau), the access trio at Destroyed(0.20, 0.10);
directness at NoBudget; ScoredArea and the touched-only DestroyedArea deleted). The lineup's rows
root hashes the measure (road_rescore.lineup_rows; FreeEnclosed's identity too), so no
partial-overlap row reads as a ramped one. The public package (reblock.buildings, reblock.budget)
keeps the partial overlap until the site revamp (BACKLOG).

Roads and clearings are now scored alike: the clearing presets already count whole buildings
(gap_clear.Model's cost is each cleared building's full population share), and a road's D now
counts a building whole once a tenth of it is destroyed.

**Cost** (the coordinator's benchmark, scratchpad/arterial/destroyed_bench.py, on 20269 and 26061):
the largest-piece step costs about the overlap intersection (0.02 -- 0.37 s per whole-network
evaluation), so about 2x the partial D; clipping the corridor to each building's envelope first
does not help; bisected buildings are rare (1 -- 107 per network). The guards: tests/
test_destroyed.py (a corner clip, a bisection at 0.40 where the overlap says 0.10, zero-area
buildings, incremental == whole; faults: the whole remainder for its largest piece, a forgetful
meter), tests/methods/test_arterial_stop.py (the stop's meter equals the scoring function; a deaf
meter breaks the stop) and arterial_check.py meter (on the stored 20269 and 38616 networks at
1/25/100/300 segments the meter and common.displacement agree to 1.3e-11; the faults deaf and
forgetful each fail it). In the study the meter's D at the stop and the same network scored
afresh after planarizing differ by at most 1.1e-5 (planarization's noding).

**The gap per block** (area weights; partial overlap / ramped / touched, a touched building
whole). The stored lineup networks (lineup-caf14fa2, 300 commits):

    block   partial  ramped  touched   ramped/partial
    6310     0.448    0.798   0.854        1.8
    20269    0.343    0.832   0.904        2.4
    40144    0.456    0.845   0.896        1.9
    39240    0.338    0.611   0.673        1.8
    8480     0.119    0.484   0.595        4.1
    38616    0.249    0.541   0.599        2.2

and cur02's networks at the ramped 0.20 stop: 6310 0.081 / 0.202 / 0.223, 20269 0.056 / 0.202 /
0.253, 40144 0.067 / 0.206 / 0.258, 39240 0.058 / 0.201 / 0.261, 8480 0.039 / 0.204 / 0.297, 38616
0.055 / 0.202 / 0.248 (ramped/partial 2.5 -- 5.3: the gap is widest early, where a road's every
touch is a fresh building, and closes as the corridor covers whole buildings).

**The arterial study again, under the ramped stop and scoring** (arterial-study-d: orion, 12
single-CPU tasks, all passed, ~25 min wall, ~2.3 task-h with scoring; arterial_rows/summary_*.csv).
cur02 = PerLineFallback(32, 128), cap02 = HardCap(32, 128), both at Destroyed(0.20, 0.10) and 1000
roads:

    block      km2   stored h  cur02 s  cap02 s  commits  D at the stop  median step s
    6310       0.084   4.04      289      340     25 / 33  0.202 / 0.204  11.5 /  9.9
    20269      0.103   2.81      195      163     21 / 18  0.202 / 0.200   9.2 /  9.0
    40144      0.153   4.29      318      356     24 / 29  0.206 / 0.204  13.3 / 12.2
    39240      0.208   2.98      141      209     15 / 24  0.201 / 0.200   9.5 /  8.8
    8480       0.387   5.46     1254      831     69 / 87  0.204 / 0.202  15.7 /  9.6
    38616      0.461  12.69     1268      944     52 / 61  0.202 / 0.202  22.8 / 15.5

Every run stopped on its budget (no max_roads, no no_gain stop; 8480, which saturated at partial
0.12, now reaches ramped 0.20). Proposal walls sum to 0.96 h (cur02) and 0.79 h (cap02) against
the stored 32.3 h; scoring adds ~0.25 h an arm (complex 100 -- 227 s a network, peak 1.9 -- 3.8
GB). Every block now reaches area D 0.05 and 0.10 in both arms (and 0.15, 0.20).

**Verdict** (the plan's Ruling 7, read symmetrically): neither dominates. cap02's wall is 0.66 --
1.48x cur02's (faster on 8480 0.66, 38616 0.74, 20269 0.84; slower on 40144 1.12, 6310 1.18,
39240 1.48, where it needs more commits to the budget). On quality cur02 leads all six medians
(cap02 - cur02: L_P -0.0071, L_Pq -0.0045, L_J2 -0.0057 at D 0.05, ahead on 20269 and 40144; L_P
-0.0045, L_Pq -0.0093, L_J2 -0.0065 at 0.10, ahead on 2 -- 3), but it is not faster on every block,
so it does not dominate. Both presets stay: greedy_arterial_access_displacement (PerLineFallback,
quality-first) and greedy_arterial_access_displacement_fast (HardCap, speed-first: 0.66 -- 0.74x
the wall on the two largest blocks, its step flat at 128 anchors where cur02's grows to 358).

**At region scale** (arterial-region-5810b: ZAF.9.3.1_1_5810, the single-block Cape Town region
of the multiblock_depth_density example, 6,619 buildings, 0.58 km2; arterial_study.py propose on
orion, one core, unscored): greedy_arterial_access_displacement (the example's preset) stopped on
its budget at D 0.201 after 102 commits (378 segments) in 9,234 s (2.6 h; median step 73.5 s, max
178 s, up to 416 anchors, peak 0.46 GB); the speed-first twin at D 0.201 after 140 commits (727
segments) in 4,554 s (1.3 h; median step 32.4 s at 128 anchors). The preset used to stop at 60
roads in ~777 s. For regen-examples: nothing breaks (the arterial's derive entries miss on the new
identity, no other method's do), but the access arterial becomes the example bake's long pole at
~2.6 h single-core on the Cape Town region (Nairobi's multi-block region untimed, of the same
order), against ~7 min for the other methods together. Either budget the bake for it (the ranking's
threads: 8 locally, 1 here, take some of it back) or run the speed-first preset in the examples
(half the wall, behind on quality); the owner's call, and the published arterial frontier moves
either way.

**Task 7, re-estimated** (owner-gated): it now re-cuts and re-scores EVERY road method's prefixes
and lens rows under the ramped D (the networks are kept, except the arterial's, which is
re-proposed at its new stop). From the stored rows of lineup-caf14fa2 (550 networks): the prefix
stage took 5.6 h (4.7 h of it the bisecting cuts, ~2x under the ramped D: ~10 h), the complex
scoring 72.8 h (59.4 h of it meshing; the largest task 3,358 s, peak 79 GB); ~83 task-h of
re-scoring. The arterial's re-proposal on its 62 lineup networks: the study's proposal walls are
1.3 -- 6.4% of the stored (median 2.1%) against 612.6 h stored, so 8 -- 39 task-h, central ~13.
Total ~96 task-h central (~91 -- 122), the oversized blocks (to 51.7 km2) the uncertain part as
before.

### Mesh speedups before the lineup's rerun (2026-10-19)

Task 7 re-meshes every road network of the lineup (552 (block, network) tasks; in the stored rows
of lineup-caf14fa2, 59.9 of 71.2 task-h are `t_mesh`). A profile of the road meshes (scratchpad
study, 6310 and 20269) found two costs that change no score, both now gone (4014bc7f):

- **The baseline complex on every carving.** Fixed.of built free first's baseline complex (cx0,
  gc.build on the buildings) with every mesh, 15-16% of a road mesh's time, though only
  Model.free_first reads it and road scoring never frees first. It is now a Strategy settled
  where the mesh is made (gap_clear.Baseline): the plain mesh's StoredBaseline, built with it and
  kept in its file as before (the cache format and every plain cache file unchanged), and a
  carved mesh's LazyBaseline, built from the buildings on first use and never stored.
- **A scan per face in _faces.** Each face's cover and zone were read by a scan of every query
  pair (quadratic in the faces); now one sort groups them (gap_clear._by_face). On 6310's lineup
  networks carved into one mesh at once (14,465 atoms, 238,125 faces) _faces takes 13.0 s
  against 145.6 s, identical; on one network's carving (7k -- 26k faces) it was 1 -- 2 s.

Both are bit-identical. On 6310's seven lineup networks (proof.py, the code before and after,
one process each on rtx6000): every lens at all 40 stops, every array of every fixed mesh (by
digest), P0, P_{1/2},0 and J_2,0, the baseline complex (carved, plain, plain read back) and free
first's held set on a carved and on the plain mesh are identical. Guards (gap_clear_check.py,
both ways on 6310): `faces` (the grouped lists against the scan's, verbatim, on the block plain
and carved by a cross of two 6 m roads; fault `unsorted` fails it, 222 -- 2,317 lists apart) and
`lazy-baseline` (a carved mesh builds no complex and stores none, and its baseline built on use
equals the plain mesh's array for array; faults `atoms`, the baseline built on the carving's
atoms, 17 arrays apart, and `eager`, built with the mesh, fail it).

**Time, 6310 end to end** (carve + mesh + model + states, one network a process, rtx6000):

    network                               triangles  mesh before/after s  total before/after s
    resistance_lp                         1,540,260      56.0 / 51.0          69.1 / 64.1
    clearance                             1,281,361      49.6 / 39.4          60.8 / 50.0
    clearance_looped                      1,465,350      53.8 / 45.0          66.8 / 58.0
    cycle_native                          1,413,181      51.4 / 41.7          61.6 / 51.8
    cycle_native_betweenness_contrast     1,344,831      50.8 / 42.3          61.0 / 52.3
    euclidean_grid                        1,290,275      50.3 / 39.9          61.8 / 50.8
    greedy_arterial_access_displacement   1,543,234      59.7 / 46.8          73.7 / 60.1

The meshes 9 -- 22% faster (17% over the seven), the networks 14.9% (454.7 -> 387.1 s); model and
states unchanged (4 -- 7 s each). A carved model that does free first now pays the build there
(11 -> 19 s on resistance_lp), which no road scoring does.

**The nodes** (the same task, 6310 resistance_lp through `cx`, new code, one task alone on the
node; mesh-node-orion and mesh-node-quadro2 at 85083830; both nodes' lenses equal rtx6000's
exactly):

    node             host     us / triangle  mesh s  model + states s  total s  peak GB
    rtx6000 (local)  rtx6000      33.1         51.0      12.0             64.1     2.5
    quadro2          ai05         52.2         80.4      23.8            106.1     2.5
    orion            ai01         55.7         85.8      23.0            110.6     2.5

Alone, quadro2 and orion are within 7% of each other and 1.6 -- 1.7x rtx6000. Packed, the stored
lineup ran slower per triangle (median, small blocks: ai05 80, ai01 95, rtx6000 40; the large
ones ai05 89, ai01 107, rtx6000 43): memory bandwidth shared by 30-odd tasks, worse on orion.
Memory now (2026-10-19 15:30Z): quadro2 idle (64 CPUs, 385 GB, nothing allocated); orion 62 of 72
CPUs idle but 1,204 of 1,545 GB allocated, to this project's own region jobs (79684_0..2 at 210
GB, 79687_0 at 540 GB, 1-1.5 days into 2.5- and 6-day limits), so ~340 GB free. The heavy tasks:
the six of 4287 at ~65 GB, 7499's at 79 GB, five more at 41 -- 49 GB, 95% of tasks under 40 GB
(median 4.5 GB).

**Recommendation for Task 7's re-scoring:** pin it to quadro2 (the faster CPU node, all its
memory free: 4287's six tasks fit five at a time beside the small ones), with `--mem` per block
from the stored peaks; orion only for a second run of other blocks once the region jobs release
their memory (with ~340 GB free, 4287 alone would take all of it). rtx6000 is 1.6x faster again
but holds no free slot.

**Task 7, re-estimated:** the complex scoring, the stored 71.2 task-h at their hosts' packed
rates, is ~70 task-h at quadro2's (ai01's 35.2 mesh-h at 1.2x its rate, rtx6000's 4.9 at 0.5x);
less 16% of the meshing (the baseline complex; _faces' saving matters only on the largest
carvings and is not counted) ~61 task-h. With the prefix stage (~10 task-h) and the arterial's
re-proposal (8 -- 39, central ~13): ~84 task-h central (~79 -- 110), against ~96 (91 -- 122)
before; about 3 -- 4 h of wall on quadro2's 32 cores, 4287's tasks (~40 min each) the long pole.

### The lineup under the ramped destroyed area, with the new arterial (Task 7, 2026-10-19 -- 20)

The arterial plan's Task 7 (owner 2026-10-19: the budgets keep their numbers, 0.02 -- 0.20, D_LENS
0.05 and FREE_BUDGET 0.05, now read as the ramped destroyed-area D of common.MEASURE, tau 0.10;
the re-scoring on quadro2; the quality-first arterial; every road method's prefixes and lenses
re-cut and re-scored, only the arterial proposed again). research/arterial-speed merged into this
branch (a1e89acd); consolidated `lineup-report` reads the arterial again, and road_rescore
`lineup-score` cuts each network under the lineup's measure before it scores it. The rows moved
to lineup_rows() = road_rows/lineup-f8fa4e53: the six methods' networks and proposal rows kept,
their prefix rows, lens rows and canonical orders (cut under partial D) and every row of the old
arterial moved to road_rows/superseded/lineup-caf14fa2, a record nothing reads.

**The runs** (quadro2, one core a task, at a1e89acd; memory by each block's stored scoring peak:
12 GB for the 59 blocks under 8 GB, 24 for the 15 under 16, 56 for 7, 88 for 4287): the six
methods' networks shipped per block and re-cut and scored, 482 of 483 tasks in 70.3 task-h
(longest 0.84 h, 4287); the arterial proposed, cut and scored on the 59 smaller blocks in 43.6
task-h (longest 5.6 h); on the 23 larger ones proposed alone at 8 GB and then cut and scored by
memory class (20 so far, 8.2 task-h for the cuts; 4287's 3.0 h), since a task that proposes and
scores holds its scoring memory through hours of proposing at under 1 GB (the first such runs,
cancelled within minutes, filled quadro2's 381 GB with four tasks). cycle_desire's g2a networks
(63612, 62441, 62803; job 79684, finished on orion 2026-10-20) were copied in and cut and scored
(1.0 task-h). Scored: 565 of the 574 (block, method). Not scored: 64070's resistance_lp (the carved
mesh refused as not Delaunay, as before; 39240's cycle_desire, refused before, is scored now: its
prefixes moved with the measure), cycle_desire on g3's five blocks still on orion (4287, 1182, 472,
63818, 64070), and the arterial on 63612, 19593 and 38142, where its proposals were cancelled after
14.0 -- 16.1 h of a 23.5 h wall (owner 2026-10-20; the old arterial's walls there were 115 -- 138
ks, unfinished) and the speed-first preset ran instead (below).

**What the measure moved.** Ramped D runs about twice partial D on the median network
(whole-network medians: clearance 0.29 against 0.14, resistance_lp 0.46 against 0.21, the grid 0.62
against 0.33), so a budget buys fewer roads: every method now reaches D 0.05 on all its blocks
(clearance on 82 against 65, cycle_native 82 against 72; cycle_desire on 75 of its 77), and every
method's L_P median at D 0.05 falls to 0.38 -- 0.54 of what it was.
- At area D 0.05 (medians; L_P, L_{P_1/2}, L_J2): resistance_lp 0.054, 0.066, 0.065 (81 blocks);
  cycle_desire 0.048, 0.064, 0.049 (75); cycle_native 0.048, 0.056, 0.052 (82); clearance_looped
  0.047, 0.059, 0.053 (82); the grid 0.046, 0.062, 0.058 (82); clearance 0.042, 0.051, 0.048 (82);
  the arterial 0.030, 0.031, 0.030 (79).
- Each against SIMP's default, paired (median, L_P): resistance_lp -0.225, cycle_native -0.233,
  the grid -0.240, clearance_looped -0.244, clearance -0.256, cycle_desire -0.265, the arterial
  -0.270. No road method is ahead of SIMP's default on any block under any lens.
- The price of roads in B's order: SIMP's default is ahead of the best road at D 0.05 on all 82
  blocks under every lens. B (i) ties on 78; on the other 4 (30796, 30848, 41673, 45876) the best
  road walls in more of the population (by 0.00006 -- 0.0015 of it). Where (i) ties, SIMP's
  default leads by a median +0.254 L_P on the 56 large blocks (p10 +0.110, p90 +0.361, at least
  +0.065) and +0.118 on the 22 oversized (p10 +0.058, p90 +0.175, at least +0.052); the best road
  gets a median 27% of SIMP's L_P on the large (p10 17%, p90 41%) and 19% on the oversized (15%,
  42%). The best road under L_P: resistance_lp on 40 blocks, cycle_native 16, clearance_looped
  10, cycle_desire 9, the arterial 3, clearance 3, the grid 1.
- The order against item 4's (Kendall tau per block, the 60 large blocks, the methods both hold):
  median +0.40 under L_P (IQR +0.07 -- +0.60), +0.37 under L_{P_1/2} and L_J2, at least -0.87.
  resistance_lp still leads.
- Free first holds roads on the same 42 blocks: the six methods' networks are the pass's, whose
  stage FreeEnclosed budgeted at partial D 0.05. Under the ramped measure the stage would hold
  fewer roads on those blocks (its budget is measured by common.displacement, now ramped), so the
  networks are not what lineup_arms() proposes at HEAD; kept by the owner's ruling (only the
  arterial re-proposed).

**The arterial.** Every network stops at its budget (whole-network D median 0.203); a median 0.46 h
a proposal (5.5 h at most of the 79; the three cancelled after 14 -- 16 h) against the old
arterial's 8.2 h at its 300-road cap. At D 0.05 it is the weakest road of the seven: last by L_P on
42 of its 79 blocks, first on 6 (1182, 472, 4287, 30848, 41515, 6310); paired, behind clearance by
a median 0.009 L_P (ahead on 26), the grid 0.012 (17), cycle_desire 0.013 (18 of 72), cycle_native
0.018 (16), resistance_lp 0.022 (14). In B's order it is the best road on three oversized blocks
(1182, 472, 4287), where every road is weak (L_P 0.003 -- 0.062). Its whole network at D 0.20
reaches a median L_P 0.079, where the others' whole networks reach 0.14 -- 0.35 at D 0.29 -- 0.63.
consolidated_rows/summary_lineup_{roads,price,held,tau}.csv.

**Why the quality-first arterial is slow on the large blocks** (the arterial study's step logs;
BACKLOG, "The arterial's step toward linear"). Each step scores every pair of anchors: the
candidates are 0.52 -- 0.55 x anchors^2 in every log. PerLineFallback adds 1.3 -- 4.5 anchors a
commit (5810: 129 -> 416 anchors over 102 commits, the candidates 9,288 -> 89,648). A step costs a
fixed part plus a per-candidate part, both growing with the block (5810: 29 s + 1.63 ms a
candidate, R^2 0.99; the blocks of ~1,000 -- 1,350 buildings: 6 -- 11 s + 0.4 ms). So a step goes
as commits^2 and a proposal as commits^3, times the block's size, and the commits to D 0.20 grow
with the block: 63612, 19593 and 38142 ran 14 -- 16 h without stopping. HardCap holds the
candidates at 9,152 every step (5810's steps flat at 32 -- 36 s: 1.3 h against 2.6 h) but needs
more commits (140 against 102), so it is not always faster on the small blocks. The way toward
linear (re-rank only the candidates near or crossing the last commit, update the step's access
state locally, commit several non-interacting roads a step) is a BACKLOG decision item, gated on
the arterial staying on the frontier.

**The speed-first preset on the three** (owner 2026-10-20: reported beside the lineup, never as its
arterial; run t7-fast, job 81290, quadro2 at a1e89acd, arterial_study.py `run ... cap02`:
HardCap(32, 128), Destroyed(0.20, 0.10), max_roads 1000, the `_fast` preset; rows in
research/roadless/arterial_rows/t7-fast, network arterial-cap02-22bc8343). Each stopped on its
budget (whole-network D 0.200 -- 0.204): 63612 after 211 commits in 1.8 h, 19593 after 564 in 3.8
h, 38142 after 370 in 2.9 h (peak 0.4 GB; 9.5 task-h with the scoring), where the quality-first
proposal had not stopped after 14 -- 16 h. At area D 0.05 (L_P; the lineup's six beside it): on
63612 0.081, fourth of seven (resistance_lp 0.152, clearance_looped 0.109, the grid 0.088 ahead;
clearance 0.077, cycle_desire 0.073, cycle_native 0.027 behind), SIMP's default 0.352; on 19593
0.012, fifth (resistance_lp 0.026 first, the grid 0.005 last), SIMP's default 0.085; on 38142
0.009, fifth (resistance_lp 0.017 first, the grid 0.008 last), SIMP's default 0.075. On 63612 it
walls in 0.0040 of the population at D 0.05 (B (i)), the grid 0.0036; the other methods none. Its
whole network at D 0.20 reaches L_P 0.158, 0.027 and 0.023.

**Compute** (sacct walls, one core a task, quadro2): 229 task-h in all, against the ~84 estimated:
the re-cuts 70.3 (482 of 483 tasks), the arterial on the 59 smaller blocks 43.6, its proposals on
the 23 larger 96.5 (52.4 for the 20 that stopped, 44.1 for the three cancelled), their cuts 8.2,
g2a's cycle_desire 1.0, the speed-first three 9.5, the first arterial runs (cancelled) 0.2. The
arterial is the excess: its proposals ran a median 0.46 h but 14 -- 16 h unstopped on three blocks,
where the estimate took the study's 1.3 -- 6.4% of the stored walls. Here: the guards after the
merge (98 runs both ways, 9.9 process-h; the arterial and mesh guards and pytest about 0.5 more).

### The street boost on paved ways only, and on proposed roads (owner 2026-10-20)

**Why.** Block 5810's flow pictures (the pipeline page, 2026-10-20) showed long corridors along
its lanes that looked found by the flow; they were given to it. The street boost took a way by its
highway class alone, and OpenStreetMap classes 5810's sand lanes as residential and service:
inside the block 7,259 m of the 7,735 m boosted is tagged unpaved, 234 m paved and 242 m untagged
(Overpass, ways of 3 m or more inside the block), while 651 m of asphalt footway and 399 m of
paving-stone pedestrian way took no boost. With the boost off the lanes carry about their share of
the flow: 15.5% of the open ground, 17.8% of the power (1.1x; the drivable lanes 1.2x, the
footways 0.9x), 26.7% of the hottest 1% of pixels (70.9% with the boost); over the no-buildings
prior the strongest pixels lie in the narrow gaps between buildings, 3.3% of the top 1% on a lane
(6.9% boosted). P0 is 3.30e6 unboosted against 1.85e6. The owner: "Footpaths shouldn't get the 10x
boost until they are picked for paving"; "Let's only boost if it's explicitly tagged paved".

**The survey** (the store's 302 blocks and 5810@major; per block the drivable length inside it, off
its ring's edge strip as consolidated `features` measures it, from the Geofabrik extract): of
473 km, 37% tagged paved, 23% unpaved, 40% untagged. By size (paved / unpaved / untagged): small
29 / 28 / 42%, large 47 / 26 / 27%, oversized 32 / 21 / 46%, 5810@major 43 / 29 / 29%. The rule
below leaves the inside unboosted (paved inside 1 m or less) on 71 of the 105 small blocks with a
drivable way inside, 18 of the 60 large and 1 of the 22 oversized. The 158,872 drivable ways of
the extract's box are tagged asphalt 81,191, untagged 53,686, paved 13,160, unpaved 4,902,
paving_stones 3,123, concrete 860, gravel 797, the rest under 400 each.

**The rule** (gap_exits): `StreetBoost(factor, surfaces, classes)`, a way boosted when its highway
class (`_link` folded) is one of `classes` and its surface tag one `surfaces` takes: `SurfaceIn`,
an explicit tag among its values (`PAVED`: the OSM wiki's paved group of the surface key, asphalt,
paved, concrete, paving_stones, sett, cobblestone, bricks and the rest), so an untagged way is not
boosted; `AnySurface`, any (the region test's 3x on every highway, S3). RUNTIME's boost is
`StreetBoost(10, PAVED, DRIVE_CLASSES)`. The extract now reads the ways' surface tag
(regions.EXTRACT_REV 3; the guards' synthetic ways carry the column too).

**Proposed roads** (owner 2026-10-20: "proposed roads should get the same 10x boost as existing
drivable streets"): a road prefix's state conducts on its own mesh (gap_clear.Model.mesh_of, the
state's `fm`), every triangle whose centroid lies in the prefix's corridor (the roads' paved
ground, budget.road_corridor: each road buffered by half its width; Carving.corridors) at the
exits' road factor (`Boost.road`: 10 under RUNTIME, 1 with no boost), a triangle on a paved way too
at the larger; a clearing's state is on the exits' mesh as before, and cleared ground is not
boosted. The per-block metric has no boost, so its road scores and its name stand.

**The name.** RUNTIME is now

    cx+shy0.5+s0.125r1b1+R0.5g4x2q4+v34d40e8a+xB8c3Lo6214b562S10spavedhliving_street-motorway-primary-residential-secondary-service-tertiary-trunk-unclassifiedv65531d75

(`s<label>` after the boost's factor, none for any surface; EXITS_CODE a03a092b -> 65531d75 with
PAVED among gap_exits' constants, so every name with exits moved). The region study's stored names
moved with it (.superpowers/sdd/2026-10-20-paved-boost/migrate_paved.py): every configuration but
T500b90D10 scores as before (no surface-filtered boost, no road scored), so its names were renamed
in place; T500b90D10's rows (the old RUNTIME's boost) went to region_rows/superseded/ and its lines
left the region summaries. Every row under the old RUNTIME (the pass's presets and baselines, the
lineup's lens rows, the store's runs.csv lines naming it) is the old metric's.

**Guards** (both ways): `street-boost` gained an unpaved residential way and an untagged tertiary
on its synthetic block (0 of 77 and 0 of 74 triangles near them boosted, the asphalt residential
way 366 of 366, the asphalt footway 0 of 96) and runs on 4241 (paved drivable 6,431 of 6,431
boosted, 0 of the 6,304 near only a drivable way not tagged paved, 0 of the 10,318 near only a
footway or path); new fault `surfaces-dropped` (the unpaved and untagged ways boosted, and all
6,304 on 4241). New `road-boost`: on the synthetic block under its boost and under none, and on
4241 under RUNTIME, each carved by the faces guard's two nested 6 m roads, every passage between
two triangles in a prefix's corridor conducts the road factor times the unboosted mesh's (x10 on
558 and 2,219 synthetic passages, 19,828 and 34,243 on 4241; x1 with no boost), the exits' mesh
outside, and each state reaches on its own mesh; faults `unboosted-road` (a prefix on the exits'
mesh: x1) and `ways-dropped` (the corridor boosted on the unboosted mesh: outside apart). Every
guard both ways at 7e3af8dc (the pass's sweep with street-boost on 4241 and road-boost added: 100
runs, 10.2 process-h): 97 as expected; `no-features` under `neighbour-features` passed, its
neighbour chosen by the drivable way it shares, none of which was paved: the neighbour is now the
one sharing the most of the boost's ways (bd9256ef; 19570 then fails under it, homes 0.88 apart);
`roads` both ways in the main checkout, whose road_rows its prefixes are in (18895's
resistance_lp, as expected), and `exits-identity` both ways there over the 56 migrated stored
names (as expected).

### A uniform rule for walled-in homes: destinations in their pockets, no free first (owner 2026-10-21)

**The question.** The metric has internal destinations (the leaks), so can free first go? The
owner: "I'd like to measure that trade-off. A uniform rule would be nicer if it works". Today (T)
the leaks sit on the cells live at baseline, a walled-in home has no demand point (Model.nodem: out
of P, P_{1/2} and J_2), relax.py frees first (the cheapest crossings, least D per person freed,
held) and Lens B counts the people left walled in first. The uniform rule (U): destinations on all
open ground, sealed pockets included; every home's demand on its own open ground, a walled-in
home's in its pocket, so its trips end at the pocket's leak and its escape time is finite but
large; no free first, SIMP's default optimizing P over every home.

**The rule as built** (three Strategies, each in the run's identity; brief
.superpowers/sdd/2026-10-21-uniform-leak, its rulings in progress.md there):
- Where the leak sits: gap_exits.InternalLeak's `sites`, LiveSites (T) or OpenSites (U: every cell
  open at baseline, a triangle's factor positive with every footprint standing, not ground). That
  is T's set without the street-reach test: the shy zones and the interiors stay leak-free, as
  cleared ground is in T. Spelt `Lp` in the exits' name; no module constant, so EXITS_CODE
  (65531d75) and the stored names stand.
- Where the demand sits: gap_clear.Placement, LiveDemand (T) or PocketDemand (U, `+pockets` in the
  metric's name and in Fixed.key: every cached mesh re-keys, _CACHE_FORMAT 3). PocketDemand keeps
  T's placement for every home T places and places each home T leaves without demand by
  Placement's own rule on the open ground no street reaches at baseline (gap_flow.in_pockets: its
  ring's points there, else the shadow rule within ring_m); a connected home whose ring also
  touches a pocket keeps its T demand (else the pockets would move connected homes). It refuses
  a leak not on OpenSites (a pocket's home would escape at once).
- No free first: relax's plan option `.f0` (relax.FreeStage: FreeFirst, unnamed; NoFree).
- B (i) reads gap_clear.Reach.street, the reach from the street faces, the ground cells and the
  leaks on cells live at baseline (T and the closed edge's calibration unchanged): a pocket's leak
  drains its homes and frees none, so B (i) does not read 0 because every home now leaks. Each
  clearing's walled-in count is the same under both scorers (uniform.py refuses a block where
  they part; none did).

    metrics.UNIFORM = cx+shy0.5+s0.125r1b1+R0.5g4x2q4+pockets+v34d40e8a+xB8c3Lpo6214b562S10spavedh<classes>v65531d75

**Guards** (gap_clear_check; each both ways). `uniform-control` (41940, 41779, none walled in): U's
demand T's element for element, its leak T's but on the pockets (349 cells and 8.5 m^2 on 41940,
477 and 13.2 m^2 on 41779: slivers sealed between footprints' shy zones, 1.6e-5 and 2.2e-5 of the
leak, no demand there), P0 and every home's escape time T's to 3e-14 -- 7e-14 (tolerance 1e-12).
SIMP's value and gradient are not T's to floating point: the demand-free pockets' leak moves the
gradient in the eps world by 1.4e-5 -- 3.6e-5 of its largest at SIMP's start (7e-8 -- 8e-7 at a
random x), which is the uniform rule itself, so the guard holds it to GRAD_TOL (1e-4, the adjoint's
own tolerance against differences). Faults `every-cell` (the leak on interiors and zones: gradient
9.5e-5 -- 4.4e-4) and `placed-everywhere` (every home on its pocket ground too) fail.
`pocket-escape` (20269, 23597): T leaves the walled-in homes out (nan), U walls in the same ones
and each with open ground escapes from its own pocket (u0 57x the block median on 20269, 506 --
508x on 23597; each pocket's leak drains its homes' injection to 2.2e-9 and 5.8e-9, tolerance
1e-7); U's homes without demand are exactly those with no open ground within ring_m (23597's home
93, its ring all footprint and shy zone). Faults `pocket-unleaked`, `placement-dropped`.
`pocket-gradient` (19570, 23597): U's gradient against differences on the walled-in homes and
their neighbours, J_1 9.6e-7 and 7.7e-6, J_2 3.0e-7 and 1.1e-5 (on 20269 the differences are
roundoff-limited, T's as U's: the gradient guard's known case); faults `adjoint-unleaked`,
`pocket-unloaded` (J_2's adjoint without the pocket homes: 0.11, 0.66). `pocket-count` (20269,
23597): the walled-in homes left read the same under both scorers at the baseline, three single
clearings, two random clearings and T's free first; fault `leak-seeded` (U then reads 0 and 1 of
1 and 6). `enclosure` and `leak-assembly` now run each leak on both sites; enclosure's fault
`every-cell` can no longer fail (the street reach ignores a leak off the live cells by
construction) and is `leak-seeded`. `exits-identity` (56 names of 56 configurations), `identity`
(UNIFORM spells back), `constants` (28, 34d40e8a), baseline, free, reach, pattern, leak-fold,
leak-gradient, leak-scale, closed-edge, trip-calibration, barriers, lazy-baseline, no-features
(with bd9256ef), street-boost, road-boost, gradient (19570, 38136 and 20269: passes), weights,
flux, convex, loop-index, underflow, faces: as expected both ways (`roads` needs the main
checkout's road_rows).

**The measurement** (uniform.py; rows uniform_rows/, parquet ignored; summaries
summary_{blocks,homes,sizes}.csv). The 121 blocks with walled-in homes under RUNTIME (792 homes:
77 small with 327, 39 large with 416, 5 oversized with 49) and five controls with none (small
20954, 41940, 41779; large 8152, 20023: a seeded draw). Two arms at area D 0.05 optimizing P with
SIMP's default: T, RUNTIME with free first; U, UNIFORM with `.f0`. Every clearing scored under both
metrics: Lens A of P, P_{1/2} and J_2 (T's over the homes T scores, U's over every home with
demand) and B (i) by street reach; paired differences U - T with 95% Bayesian-bootstrap intervals.

**People left walled in** (B (i), homes). T leaves 28 on 6 blocks (at most 0.10% of a block's
population), U 100 on 33 blocks (a median 0.12% where it leaves any, 2.4% on 63718); U leaves 72
that T frees (on 30 blocks), T none that U frees. They split in two:
- 708 walled-in homes have open ground in a pocket. T frees all; U frees 687 and leaves 21 on 12
  blocks (7 small, 5 large; none oversized). On 5 of the 12 U's own metric prefers T's clearing
  (SIMP's local optimum, the gate free first was built for: 3786 by 0.44 L_P); on the other 7 it
  prefers spending the budget elsewhere.
- 84 have no open ground within ring_m at all (their ring all footprint and shy zone: 3.4% of the
  small blocks' walled-in homes, 14% of the large's, 27% of the oversized's). U gives them no
  demand, so its objective cannot see them, and leaves 79; T's free first frees 56 and finds no
  crossing for 28 (the 28 T leaves).
And of those freed, how well (U's metric, escape time after the clearing against the block's
median at baseline): of the 708 pocket homes T frees, 221 (31%) stay over 10x the median, freed
through a cleared neighbour's ground within 1 m while their pocket stays sealed (median 3.6x);
of U's 687, 70 (10%; median 1.38x). B (i)'s "freed" is generous: T never scores a freed home's
access (it has no demand), U scores only its pocket's (its demand is fixed at baseline, so a
neighbour's clearing does nothing for it: 40185's home 21, 756x before and after T's clearing,
0.21x after U's).

**Lens A**, U's clearing less T's, mean [95% Bayesian-bootstrap interval] over the blocks:

                     L_P                       L_{P_1/2}                 L_J2
    T's metric
      small (77)     -0.0146 [-0.024, -0.006]  -0.0161 [-0.028, -0.007]  -0.0159 [-0.028, -0.007]
      large (39)     -0.0066 [-0.011, -0.003]  -0.0037 [-0.007, -0.001]  -0.0061 [-0.011, -0.003]
      oversized (5)  -0.0047 [-0.016, +0.001]  -0.0032 [-0.013, +0.002]  -0.0052 [-0.018, +0.001]
      all (121)      -0.0116 [-0.018, -0.006]  -0.0116 [-0.019, -0.005]  -0.0123 [-0.020, -0.006]
    U's metric
      small (77)     +0.094 [+0.058, +0.134]   +0.175 [+0.108, +0.248]   +0.186 [+0.118, +0.259]
      large (39)     +0.039 [+0.020, +0.067]   +0.143 [+0.077, +0.217]   +0.160 [+0.085, +0.235]
      oversized (5)  +0.0044 [+0.001, +0.010]  +0.038 [+0.004, +0.105]   +0.048 [+0.005, +0.134]
      all (121)      +0.073 [+0.049, +0.101]   +0.159 [+0.112, +0.213]   +0.172 [+0.124, +0.224]

Under T's metric the median L_P difference is -0.004 (small -0.008, large -0.002, oversized 0) and
U is ahead on 29 blocks, behind on 88; under U's, +0.020, ahead on 85, behind on 32.

T's arm under T's metric: L_P 0.4778 small, 0.4109 large, 0.2497 oversized. Under U's metric U's
own optimizer ends below T's clearing on 32 of 121 blocks (22 small, 10 large; median -0.005, at
worst -0.44 on 3786). The clearings' Jaccard: median 0.61 small, 0.80 large, 0.88 oversized
(0.12 at least); identical on 4 walled-in blocks and every control. The arms spend D 0.0497 (T)
and 0.0495 (U) on average.

**How strongly P sees them** (U at baseline). A walled-in home's escape time is 208x its block's
median home (p10 79x, p90 478x, from 15.5x to 2,184x): its pocket's leak is small. The walled-in
homes hold a median 44% of U's P0 (at most 93%) and 98% of J_2,0 (at most 99.9%), so U's Lens A
is mostly whether the pockets open. The homes T scores keep T's escape times exactly (P over them
within 3e-12 on 118 of 121 blocks; on 63612, 24240 and 30796 four homes T leaves without demand
though B (i) calls them connected get pocket demand under U, +18%, +5.1% and +0.35% of P0).

**Free first's fairness** (T; the owner's "Clearing N people's homes to provide access to n<N
people's homes seems unfair"): the free stage spends a mean D 0.0048 (10% of the budget) and, per
unit of the population it frees, a median 0.69 (pooled 0.49, p90 1.6), over 1 on 27 of the 119
blocks where it frees anyone and over 2 on 8: 21.5 on 42404 (D 0.0139 for a home of 0.00065),
10.1 on 38616, 7.1 on 20080.

**Against the uniform-walls study** (2026-10-11, "Uniform walls: one number against two"). Its A
(every footprint conducting eps 1e-4) rescued the enclosed homes but moved 3.9% of the open homes'
escape times by more than 5%; U moves none at baseline (a pocket's leak touches no connected home),
and in the eps world only by the micro-pockets' 1e-5 of the gradient. But U weighs the walled-in
homes far more: A's had a median r of 12 and a median 41% of J_2 at 1e-4, U's a median 208x the
block median and 98% of J_2,0, an exchange rate set by each pocket's own leak (its area and its
homes' footprints) where A's was set by eps and the walls. As there, a finite exchange rate makes
the walled-in homes a toll the lens is dominated by; U puts it on P rather than ordering by B (i)
first.

**What it says.** U does not free the walled-in people by itself: it leaves 100 homes on 33 blocks
against T's 28 on 6, 79 of them homes with no open ground that its demand cannot reach and 21
pocket homes (12 blocks) that SIMP's optimum, or the budget's better use under U's lens, leaves
sealed. It costs the others' Lens A under T's metric a mean -0.012 (L_P, P_{1/2}, J_2 alike; the
interval excludes 0; small blocks -0.015, large -0.006). It is not dominated: under its own metric
it is ahead (+0.073 L_P, +0.16 P_{1/2}, +0.17 J_2), it serves the homes it frees better (10% left
over 10x the median against T's 31%: T's "freed" is often a neighbour cleared within 1 m, the
home's own ground still sealed), it spends no D on free first's unfair trades (ratio over 1 on 27
blocks), and it is one rule. The owner's call (BACKLOG, "A uniform rule for walled-in homes"),
with two variants that would answer U's two failures: demand for the no-ground homes (on their
own footprint, leaking as a pocket), and free first's crossings offered to U's incumbent (the
gate) without holding them.

**Compute.** GPU (cluster, at most 4 cards, b395b3d4): ul-oversized (5 blocks, 4.86 task-h, about
1.5 h) and ul-large (41 blocks, 7.02 task-h, 1.9 h), 11.9 task-h, a task 2.2x the consolidated
pass's default wall on one card (two arms, two meshes, two models for the lenses), at most 1.47 h
(63612; its process's peak RSS 55.8 GB of the task's 64, GPU 6.9 GB). The U arm took 0.93x T's
wall (0.84x SIMP's own time). Here, single-threaded, at most 2 processes: the 80 small blocks 2.1
process-h (0.96x the default's wall), the guards and diagnostics about 8 process-h.

### Pruning by epistemetry: the move (owner 2026-10-17)

**Why.** The owner's decision of 2026-10-17: the consolidated pass prunes with seqscreen (its
screens are done, phase 5 above), then reblock's screens decide through GeoffChurch/epistemetry and
seqscreen is deleted, so that epistemetry is dogfooded by several projects rather than shadowed by
a second decision layer inside reblock. seqscreen goes by that decision, not by dominance: the
fourth replay ("Pruning by epistemetry: the realized loss as the target", above) tied it within
noise, the realized loss below seqscreen's at every rate with 67 wrong calls and 284 asks against
its 66 and 280 in each setting's 1,300 pairwise runs, counts that moved more under the nine-rate
sensitivity than they differ by. The plan: docs/superpowers/plans/2026-10-17-epistemetry-move.md.

**What a screen is now.** epistemetry 165f98b's `Screen` (a policy, a drop rule and a seed:
`design.screen`), carried on by its live loop `advance` as each block's rows arrive. The policy is
`ValueOfMoreUnits`: past a 15-block floor, at step 1, it values the next blocks at every horizon (1
to 46, capped at the blocks left) on the finite-population Bayesian bootstrap of the variant's own
blocks, the extra GPU-seconds uncertain jointly with the gain and the next blocks' cost forecast in
proportion to the default's GPU-seconds on them (`ProportionalTo`, the cost model's covariate), and
goes on while some horizon is worth its GPU time; the collapse veto (`VetoBelow`, a block more than
0.1 Lens A below the default) stops it at once, keeping the default. What it values, prices and
calls each arm by is the rule, a Strategy in screening.py (`Rule`), and the two presets are built
once at consolidated's edge (`screen_presets()`, every number that can change a call a constant
there) and injected into `screen-report <preset> <wave>`, which never asks which rule it holds:
- `realized-loss`, the default: `AdoptKeepAskByStake` over the bracket's five geometric rates
  (`RATES_PER_GPU_SECOND`, 2e-6 -- 1.9e-5 Lens A per GPU-second), equally weighted, over FUTURE
  (820) uses, a wrong call losing |gain - rate x extra GPU-seconds|; the lowest realized loss at
  every rate in the fourth replay. It gives up wrong calls: 67 against 37 in the fourth replay's
  1,400 distinct pairwise runs (each design without the veto, C1m4/p64 with it too), for fewer asks
  (284 against 318).
- `fewest-wrong`: `AdoptKeepAsk` over the bracket's ends, each arm priced by its own stake
  (`screening.OwnStake`, the third replay's 0-1 target). It gives up asks (318 against 284) and
  realized loss at the bracket's bottom and middle.

The nine-point sensitivity is no preset (asks are free, owner 2026-10-16, so no operating point
wants fewest asks), and `SharedStake` was not ported (on one arm it is the own price, and reblock
screens pairwise). `Screen.record()` is a screen's identity (the rule's target and stop, the
horizons, the floor, the step, the cost model, the Monte Carlo sizes, the veto and the seed), and
it revives as the screen itself. Each wave continues from the screens' stored ledger
(consolidated_rows/screen_decisions.json: the metric, the Screen's record, and each screen's
Running or Stopped state with the blocks it was decided on), so a wave never re-values the
decisions that chose the earlier ones; a ledger of another metric or preset, on blocks the rows no
longer start with, or one epistemetry cannot read back is refused before anything is decided, and
removing the file starts afresh. The screens' metric is the campaign's
(`consolidated.SCREENS_METRIC`): phase 5's ran under the previous boost (every drivable way,
va03a092b), which this code no longer builds, so `screen` and `screen-lenses` refuse to extend that
campaign, and a campaign under RUNTIME moves the constant and starts afresh. Continuing equals
deriving afresh only while the cost covariates stay as they were: in a campaign run in real waves a
block's covariate is the pass's default time until the screen runs it and the screen's own re-run
of the default after (on the 30 held-out blocks with both they differ by up to 137 s, 65% of the
pass's), so r2l2 decided as a 17-block wave (the covariates of the seven blocks it runs next moving
by up to 10.8% once run) and continued onto its 24 rows reaches the same stop (20) and call (ask)
with readings about 1% off and a ledger unequal to the one derived afresh (BACKLOG, "A screen
continued over real waves", the owner's call).

The pin lives in pyproject.toml, as cluster_submit's does: epistemetry in the `research` dependency
group, `[tool.uv.sources]` at 165f98b's full SHA (cluster_submit mirrors uv.lock's git sources to
the cluster, so the nodes need no GitHub access). It moves so (screening.py, "THE PIN"): edit the
rev, `uv lock --upgrade-package epistemetry`, run `screening_check.py both-ways
target,records,adapter,why,acceptance,screens`; if `acceptance` fails only because the random
numbers changed, regenerate the runs (`runs`) and pass `equivalent` against the committed
screen_acceptance.csv before committing the new one; if `screens` fails the same way, re-derive the
screens afresh (remove screen_decisions.json, `consolidated.py screen-report <preset> 0`) and
report the moved stops in the pin move's commit. epistemetry requires Python >=3.12
(GeoffChurch/epistemetry #9 asks why), so the project's requires-python rose to >=3.12 (the venv is
3.12.13; main is untouched).

**The comparison** (screening_check.py `runs` and `equivalent`; screen_acceptance.csv). At 165f98b
every draw is keyed by a seed and its place (`design.screen.stream`), so no replay reproduces the
fourth replay's runs bit for bit: the move was checked statistically, run against stored run on the
same order. Both presets were replayed at the pin with epistemetry's `replay` on the fourth
replay's frozen designs and orders (screen_designs.json: every pairwise design's 100 orders without
the veto, C1m4/p64's with it too; order o's draws keyed by (0, o); 1,400 runs a preset, 2,800 in
all) and paired with the fourth replay's runs of the same policies. Exact where nothing is left to
draw (the whole design observed, or a veto): 14 of 14 truths a preset (each the call on the whole
design) and 200 of 200 C1m4/p64 runs a preset (its 12 blocks no more than the floor, so a run sees
them all or is vetoed; both settings), equal in blocks, call, wrong call, asks, truth and cost. The
test, registered 2026-10-18 before the run: four quantities a run (blocks; cost, the share of the
design's GPU-seconds; a wrong call; an ask), d = new - old on each order; equivalence if the 95%
bootstrap interval of mean(d) (10,000 resamples) lies inside +-delta, delta half the two presets'
gap on the old runs, so no shift that passes can reorder them (0.609 blocks, 0.0150, 0.0107,
0.0121); and no shift found by a sign-flip test (100,000 flips; exact under the null, since the two
runs on one order are exchangeable), on the 8 (preset, quantity) and 26 (preset, design) blocks
tests, Holm at 0.05, McNemar's exact p beside the 0-1 quantities agreeing:

    preset          quantity  old      new      d [95% interval]              p
    realized-loss   blocks    15.0336  15.0243  -0.0093 [-0.0529, +0.0379]    0.72
                    cost      0.4430   0.4426   -0.0004 [-0.0013, +0.0006]    0.44
                    wrong     0.0479   0.0486   +0.0007 [+0.0000, +0.0021]    1
                    asks      0.2029   0.2007   -0.0021 [-0.0050, +0.0000]    0.25
    fewest-wrong    blocks    16.2514  16.2586  +0.0071 [-0.0493, +0.0714]    0.84
                    cost      0.4731   0.4730   -0.0001 [-0.0016, +0.0015]    0.91
                    wrong     0.0264   0.0264    0      [-0.0021, +0.0021]    1
                    asks      0.2271   0.2257   -0.0014 [-0.0043, +0.0014]    0.63

Every interval inside its delta; per design the blocks move by at most 0.38 (smallest raw p 0.20);
Holm rejects none of the 34: pass. In counts at the pin: realized-loss 68 wrong calls and 281 asks
in its 1,400 runs (67 and 284 in the fourth replay), fewest-wrong 37 and 316 (37 and 318). The
test's own check (the migration's `self-check`, deleted with it): the old runs against themselves
pass; one block more on every r2/base run fails on Holm only (p 1e-05); realized-loss's wrong call
flipped on 20 runs fails on its interval ([+0.0086, +0.0207] outside 0.0107); a draw-free run
costing 1e-9 more fails the exact check, and a comparison with no draw-free run fails. The margins
are now a rule, not numbers (`deltas_of`: half the presets' gap on the old runs it is given), since
a pin move compares against the committed CSV, whose blocks gap gives 0.617, not 0.609.

**GeoffChurch/epistemetry #10, measured.** A replay charges a step unit by unit (an arm pays for
the blocks it observed up to its stop); a live wave run in parallel pays the whole step. Beside
each run's cost the acceptance runs record `wave_cost` (each arm's steps paid whole), equal to cost
bit for bit wherever no veto cuts a step short. It is higher on 93 of the 1,400 runs a preset, all
among C1m4/p64's 100 runs under the veto, where the collapse falls inside the floor's first step
(its 12 blocks): 1,609.3 GPU-seconds charged (5.9 blocks, mean) against 3,067.0 a wave pays;
(wave_cost - cost) is a mean 0.475 of the design's cost over those 100 runs and 0.034 over all
1,400; the same for both presets. It moves neither the comparison (both sides are replays charged
unit by unit) nor the presets' ranking (both share the floor's step); it biases a live screen's
cost forecast where a veto falls inside a multi-block step, which at step 1 is only the floor's
first. Commented on #10 (the owner's yes, 2026-10-18):
https://github.com/GeoffChurch/epistemetry/issues/10#issuecomment-6338646207.

**Phase 5's verdicts**, re-derived through `advance` under `realized-loss`
(consolidated_rows/summary_screen.csv, its `screen` column the Screen's record;
screen_decisions.json, six Stopped states, 10 judged readings): every stop and call seqscreen's.
The lenses and times are phase 5's (above), bit for bit; the break-even rate (the mean gain over
the mean extra GPU-seconds on the blocks used, Lens A per GPU-second: the rate an ask turns on) is
new:

    variant      eligible  run  used  stop     call  break-even
    .r2l2              46   24    20  settled  ask   7.9e-6
    .pP                46   15    15  settled  keep  7.7e-7
    seed               46   15    15  settled  keep  0 (-2.6e-16)
    .m2                15   15    15  ran out  keep  0 (3.2e-15)
    MMA                46   15    15  settled  keep  -6.9e-5
    damped OC          46   15    15  settled  keep  -7.2e-4

**The guards** (screening_check.py; each passes on the code and fails under each of its faults, a
patch inside the guard's process; `both-ways` runs both, single-threaded, after the deletion: 51
and 43 minutes):
- `target`: each preset's target as its policy builds it, against the loss recomputed outside
  epistemetry: realized-loss's risk, call and price 37 of 37 decisions, fewest-wrong's target and
  own price 37 of 37. Faults `dropped-uses` (the target over 1 use, not FUTURE: 21 of 37 risks off,
  the other 16 losing exactly 0) and `unstaked` (the own price without its stake: 37 of 37 off):
  FAIL.
- `records`: each preset's `Screen.record()` round-trips through JSON and names its target. Fault
  `array-rates` (the rates the array np.geomspace returns): the record refuses it, FAIL.
- `adapter`: a live screen fed through `screening.arrived_of` is epistemetry's replay, decision for
  decision, in 18 cases (r2/base, x2/p64, pPg/p256, C1m4/p64 in both settings and a zero-gain
  design, orders 0-2); cut one block short of its stop it is Running in 15, waiting for the block
  the replay took next; the zero-gain design settles at the floor keeping the default, C1m4/p64 is
  vetoed at its first collapsing block ("collapse") or runs out at 12. Faults `misaligned` (the
  covariate in the fixture's block order, not the screen's) and `gross-extra` (the variant's gross
  GPU-seconds as its extra): 8 cases each, FAIL.
- `why`: a collapse on the screen's last block reads "collapse", not "ran out". Fault
  `ran-out-first`: FAIL.
- `screens`: the committed summary_screen.csv and screen_decisions.json are what the code derives,
  continued from the ledger and afresh; every state round-trips; 10 of 10 judged readings'
  verdicts, rules and prices re-derived from their records; r2l2's ledger cut to 4 of its 7
  decisions (Running on 17 blocks), as a wave would store it were the covariates unchanged,
  continues to the screens derived afresh; refused before any decision, each naming its cause: a
  held-out block's default time missing, another metric's or preset's ledger, a state or screen
  record epistemetry cannot read back, a variant's stored blocks off its order. Faults `shuffled`
  (common.HELD reshuffled), `altered-record` (one stored price times 1 + 1e-6), `wave-seeded` (each
  wave its own stream): FAIL.
- `acceptance`: realized-loss replays screen_acceptance.csv's orders 0-9 (140 runs) exactly, 140 of
  140. Fault `dropped-uses`: both/p64 differs on 4 of 10 runs, FAIL.

**What was deleted** (abe75d6d): seqscreen.py, with its loaders, `next`, `question`, ORDERS,
STOPPED and POWER (no caller outside it; the ask's number is summary_screen.csv's `break_even`),
epireplay.py, and screening_migration.py (the move's one reader of the fourth replay's rows); no
reader of any remains (`git show abe75d6d^:research/roadless/seqscreen.py` has the code). The block
sets are common.py's (TUNING, 13; HELD, 46, in the order phase 5 ran them, frozen as data, since
165f98b's `random_orders` no longer draws that permutation). The replays' data (Ruling 6, the
owner's "delete"), after the comparison passed: 36 git-ignored paths, 58 MB (ten
epireplay*.parquet, five *_jobs directories with screen_acceptance_jobs, 22 epireplay4 logs);
screen_acceptance.csv holds every acceptance run. The fixture stays: epireplay_designs.json is
screen_designs.json, reblock's real-data acceptance fixture (epistemetry's spec puts each project's
real-data acceptance test in that project, run through its pin), its five many-arm designs read by
nothing until a many-arm screen exists (BACKLOG). The four replays' sections above stay as they
were, records of code that is gone.

**Compute.** Here, single-threaded: the acceptance runs 13.0 worker-hours over 8.5 h (112 jobs, 63
on 1 worker, 49 on 4; the fourth replay's same work about 12.7); the guards after the deletion 1.6
process-hours (`acceptance` 43 minutes, the others 51).
