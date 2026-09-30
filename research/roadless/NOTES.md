# Roadless scoring: heading-aware conduction on the space a network frees (living note)

Branch `research/roadless`, code in `research/roadless/`. Started 2026-09-29.
pyamg lives OUTSIDE the project env: `~/.cache/reblock-research/pydeps` (pip --target), imported by
`lifted.py`. Run scripts from the repo root with `PYTHONPATH=.`.

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

**Screened greedy (M 4) vs the lineup** (compare_clear.py, 139 of 220 blocks at first read):
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
