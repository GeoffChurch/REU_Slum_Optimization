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

(pending: offset spread + h with sub-cell fractions, conv2/)
