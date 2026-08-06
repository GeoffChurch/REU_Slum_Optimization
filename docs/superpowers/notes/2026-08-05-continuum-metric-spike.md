# Continuum conduction: the right object, still not a metric (2026-08-05)

Spiked before committing to `specs/2026-08-05-road-geometry-in-conductance-design.md`, because that
spec's machinery — planarized road graph, projections, bounded Dijkstra, series resistance — is
entirely DISPOSABLE if the continuum works.

**Three rounds.** On raw `NN/2` disks it does not converge and the free-space topology fragments
without limit. Shrinking every disk by a minimum separation `eps ~ 0.5-1.0 m` fixes that — free
space collapses to one component. But swept across 6 blocks rather than 2, `eps` is a regularizer on
four of them and a KNOB on one, where permeability climbs 0.608 -> 0.696 with no plateau. Worst-case
`eps` + `h` sensitivity then approaches the whole ~0.118 cross-method signal.

**Current state (A5 gate, 2026-08-06): BLOCKED.** `eps` reorders methods at the
criterion's own sample -- 119 rank flips, min Kendall tau +0.600. Round 6's contrary result was an
n=10 x 5-method x 1-eps-pair artifact. Superseded text follows.

~~Round 6:~~ `eps` moves absolute
permeability but does NOT reorder methods -- 0 of 50 rank flips, Kendall tau +1.000 on every block
-- and permeability is used comparatively. `h` converges on the common population; the rare
under-resolved block is per-block detectable. Region-scale cost
and memory are both cleared (60.4 s and 6.35 GB at region size); displacement coupling is decided
(couple them). See [Round 3](#round-3-breadth-the-rescue-holds-on-most-blocks-but-not-all).

## What was tested

Permeability as a continuous conduction problem on the free space rather than a graph on parcel
centroids:

    -div(sigma grad u) = f,   u = 0 on the street,   P = f^T u,   permeability = 1 - P(roads)/P(0)

Buildings are HOLES (excluded), `sigma` is `g_walk` in free space and `g_road_per_m` inside a road
corridor, discretized on a road-independent grid of spacing `h`.
`scratchpad/continuum/spike.py`.

Why this was attractive: it subsumes D1 and D2 (travel distance is intrinsic to the domain, so a
zigzag road is a zigzag high-sigma region and costs more with nobody measuring a route), the
road-coverage boolean, the node-placement question entirely (demand is a density), the
`(d - r_i - r_j)/d` clearance factor and its fair-normalization (a channel of width `w` and length
`L` conducts `sigma*w/L` automatically in 2D), and the `distance <= STREET_TOL` grounding test
(a Dirichlet condition instead). Monotonicity would stop being a proof and become a triviality:
`sigma' >= sigma` pointwise implies `P' <= P` by the Dirichlet principle — no nested-edge-set
argument, no lemma about planarization, no worry about entry points moving. And
`clearance.py`'s cost field and the metric's conductivity would become the SAME object, closing the
proxy gap that forced spec (A)'s scope cut.

One modelling point that did work and is worth keeping: **demand is injected over each building's
perimeter, not at a point.** A 2D point source has log-divergent self-energy, so `P` would grow
without bound as `h -> 0`. Perimeter injection is also what a person does — you leave through a wall.

## The decisive question: does it converge as h -> 0?

    ZAF.9.3.1_1_19362  (50 parcels)
       h      cells    components   stranded       P0      permeability
     1.00     5,947         7         0.8%     2476.23       0.586903
     0.50    23,826         8         0.5%     2480.92       0.599653
     0.35    48,608        14         0.6%     2468.08       0.595766
     0.25    95,336        21         0.7%     2396.95       0.587679

    ZAF.9.3.1_1_5530  (150 parcels)
       h      cells    components   stranded       P0      permeability
     1.00     6,487         8         0.3%    16022.00       0.672762
     0.50    25,929        14         0.4%    14597.79       0.717094
     0.35    52,920        18         0.4%    14235.79       0.721081
     0.25   103,695        24         0.4%    13951.85       0.703898

**No.** Permeability oscillates in a +/- 0.01 to 0.02 band with no plateau and no monotone trend,
across a 4x refinement. For scale, the whole cross-method spread the metric exists to resolve is
~0.118 (11.8 points at D = 10%), so a resolution artifact of +/- 0.015 is **~13% of the signal**.
Grid spacing would become a reported parameter that moves rankings.

Worse, and more diagnostic: **the free-space topology never stabilizes.** Component counts climb
7 -> 8 -> 14 -> 21 and 8 -> 14 -> 18 -> 24, roughly linear in `1/h` with no sign of a limit. Refining
the grid keeps DISCOVERING new disconnections rather than resolving a fixed domain.

## Why

`building_radii` is half the nearest-neighbour distance, so a mutual-nearest-neighbour pair has
`r_i + r_j = d` EXACTLY. Measured over 5,267 adjacent pairs
(`scratchpad/spectral/clearance_floor.py`):

    true gap |p_i - p_j| - r_i - r_j :  median 2.77 m
    gap <= 0    (disks touch/overlap):  10.0% of pairs
    gap <= 0.5 m                     :  18.0% of pairs

Free space is genuinely pinched shut across 10% of adjacencies. The current metric never notices,
for two reasons that are both accidents: it measures clearance CENTROID-to-centroid (only **0.9%**
of mesh edges hit the `FOOTPATH_EPS` floor, versus 10% of the true point-to-point gaps), and it
floors whatever is left. A continuum takes the geometry literally, so the fakeness of the geometry
becomes the dominant term.

**The disks are not footprints.** They are `NN/2` — a fair non-overlapping upper bound, not a
measurement. The continuum is the right OBJECT; the input is not good enough to support it.

Stranded demand stays low throughout (0.3-0.8%), so the proliferating components are mostly empty
slivers rather than stranded buildings. That is why the current graph metric survives the same
geometry: it never asks whether free space is connected.

## Cost at region scale — MEASURED, and not the blocker

Calibrated from real geometry (median block area 80.2 m^2 per parcel), the 11,006-parcel depth
region implies **3.53M cells at h = 0.5, 7.21M at h = 0.35, 14.12M at h = 0.25**.

No `pyamg` / `sksparse` / `petsc4py` is installed, so the field is scipy's `spsolve` (SuperLU) and
its iterative solvers. Benchmarked on representative domains — 5-point Laplacian, 10% holes, pruned
to the grounded component (`scratchpad/continuum/solver_scaling.py`):

    N cells      spsolve s   CG+Jacobi s   CG iters
        56,042      0.17          3.28        1,583
       144,247      0.58         16.56        2,518
       380,572      2.06         63.48        4,015
       728,995      5.31         83.29        5,568

    spsolve  scales as N^1.33      CG iters scales as N^0.49  (= sqrt(N), as 2D Poisson theory says)

**The DIRECT solver wins outright** and needs no new dependency. Extrapolated:

    region h=0.50 (3.53M):   ~42 s/solve    ~28 min per 40-solve lens curve
    region h=0.35 (7.21M):  ~108 s/solve    ~72 min
    region h=0.25 (14.1M):  ~265 s/solve   ~176 min

Today's region solve is 3-13 s (`specs/2026-07-30-road-first-mesh-design.md`), so `h = 0.5` is
roughly **3-10x current cost** — expensive for a batch regeneration, not disqualifying. And `h = 0.5`
appears sufficient once `eps >= 0.5 m`: the h = 0.5 -> 0.25 movement there was only 0.0003-0.002.

CG+Jacobi is far worse (855 s extrapolated at h = 0.5) and its iteration count grows as sqrt(N), so
"add an iterative solver" is the wrong instinct here; AMG would help but is not required.

Memory was the residual worry and is now **measured, not extrapolated** — see the round-3 list:
6.35 GB at 3.42M cells. Time is the cost; memory is not a constraint.

The `h = 0.5 -> 0.25` figures quoted just above ("only 0.0003-0.002") come from the two-block round-2
sample and do NOT generalize; round 3 measures `h` sensitivity up to 0.043 across six blocks.

## The rescue: a minimum separation

Round 1 concluded the failure was structural in the input and not fixable by tuning. **That was
wrong.** Shrinking every disk to `max(NN/2 - eps, 0.25)` — asserting that buildings are separated by
at least `2*eps` — fixes it outright:

    eps = 0.50 m                          eps = 1.00 m
      h     comps   permeability            h     comps   permeability
    1.00      1       0.571942            1.00      1       0.576801
    0.50      1       0.580982            0.50      1       0.576742     ZAF...19362
    0.35      1       0.581516            0.35      1       0.575071
    0.25      1       0.581281            0.25      1       0.576804

    1.00      1       0.762154            1.00      2       0.761145
    0.50      2       0.763359            0.50      2       0.765037     ZAF...5530
    0.35      2       0.765923            0.35      1       0.765738
    0.25      1       0.765328            0.25      1       0.766271

Against the raw-disk run's 7 -> 21 and 8 -> 24 components and +/- 0.013 to 0.024 oscillation:

- **The topology stops fragmenting.** Free space collapses to one component and stays there under 4x
  refinement, instead of climbing without limit.
- **Permeability converges to +/- 0.001-0.003** — an order of magnitude better, and now well under
  the ~0.118 cross-method spread the metric has to resolve.
- **`eps` does not dominate.** Doubling it 0.5 -> 1.0 m moves permeability by 0.0045 (19362) and
  0.0010 (5530), the same size as the residual resolution noise. It behaves as a regularizer, not
  as a knob that swings the answer.

Feasibility: median building NN distance is 5.60 m, so `eps = 1.0 m` takes a typical radius 2.80 ->
1.80 m and drives only **0.5%** of radii to the floor (`eps = 0.5 m`: 0.0%).

It is also a better parameter than the one it replaces. `FOOTPATH_EPS = 0.02` is an arbitrary
DIMENSIONLESS floor on a conductance ratio; "buildings are separated by at least 1 m" is a physical
length that can be defended, measured, or falsified against real footprints.

## ROUND 3 (breadth): the rescue holds on most blocks, but not all

Round 2 rested on two blocks and two `eps` values. Swept properly — 6 blocks, `eps` in
{0.25, 0.5, 0.75, 1.0, 1.5}, `h` in {1.0, 0.5} (`scratchpad/continuum/eps_sweep.py`):

    h-CONVERGENCE  |perm(h=1.0) - perm(h=0.5)|        max free-space components
      eps=0.25   median 0.0127   max 0.1106                    10
      eps=0.50   median 0.0080   max 0.0680                     2
      eps=0.75   median 0.0054   max 0.0271                     3
      eps=1.00   median 0.0068   max 0.0430                     3
      eps=1.50   median 0.0037   max 0.0147                     2

    eps-SENSITIVITY at h = 0.5, permeability per block
      eps:            0.25     0.50     0.75     1.00     1.50
      ...19362     0.57964  0.58098  0.57487  0.57674  0.56870
      ...19366     0.65867  0.65893  0.65931  0.65778  0.66128
      ...20053     0.57722  0.57191  0.57212  0.57045  0.56186
      ...39257     0.61897  0.72857  0.74170  0.74199  0.74851
      ...41782     0.73238  0.76823  0.76837  0.76968  0.76949
      ...41829     0.60771  0.62050  0.64567  0.68627  0.69575   <- NO PLATEAU

      per-block range over eps 0.25-1.5 : median 0.02632, max 0.12954
      restricted to eps 0.5-1.0         : median 0.00389, max 0.06577

**Round 2's headline was an n=2 artifact and is corrected.** "Doubling `eps` barely moves the answer"
holds for four of six blocks and fails badly on one: `41829` climbs monotonically 0.608 -> 0.696
across the whole `eps` range with no sign of settling, so there `eps` is a knob rather than a
regularizer. In the sane 0.5-1.0 band the MEDIAN sensitivity is a fine 0.0039, but the MAX is 0.066 —
over half the ~0.118 cross-method spread — and h-convergence still tops out at 0.043.

Stacked, worst-case uncertainty on the tail blocks approaches the entire signal the metric exists to
measure. **That is not shippable, and it is the current blocker.** What is special about `41829` is
not yet understood and is the first thing to investigate if this is picked up again.

## ROUND 4 (diagnosis): the tail is real but rare, and its cause is NOT identified

Round 3 left one question: what distinguishes a block where `eps` never plateaus? Two hypotheses
tested, both falsified, and the exercise reframed the size of the problem.

**Pinched fraction explains nothing.** The share of adjacent pairs with `gap <= 0` — the mechanism
behind round 1's failure — sits at **0.091-0.101 on all six blocks** and correlates `r = +0.104`
with `eps` sensitivity. It does not vary, so it cannot discriminate.

**Street frontage looked like the answer at n=6 and is not.** The three least-sensitive blocks all
had frontage >= 0.341 and the three most-sensitive <= 0.305, `r = -0.640`, with a coherent mechanism
(`eps` sets interior corridor width, so it should matter most where demand cannot escape directly
onto a street) and an alarming implication (frontage is inverse depth, so the metric would be least
trustworthy on exactly the deep blocks the project targets). **Tested on 20 blocks: Spearman
`+0.027`, p = 0.91; low-vs-high frontage Mann-Whitney p = 0.60.** No relationship. Neither does
block size (`+0.039`, p = 0.87).

**But the 20-block sample resizes the problem, in the good direction:**

    |perm(eps=1.0) - perm(eps=0.5)| at h = 0.5, over 20 blocks
      median 0.0057     max 0.0285
      (41782 reads 0.00144 here against 0.00145 in the 6-block run -- reproducible)

Median sensitivity is **~5% of the 0.118 cross-method signal**. Round 3's "max 0.066" came from one
block this sample did not draw. Over ~26 block-measurements the honest picture is: median ~0.006,
nearly all <= 0.029, **one outlier at 0.066**. A tail of roughly 1 in 25 — not the norm round 3
implied, but real, and **unexplained after two falsified hypotheses**.

That is the state to resume from. The typical case is fine; an uncharacterized 1-in-25 tail is still
disqualifying for a metric that grades published comparisons, because the risk cannot be bounded
without knowing what drives it. Next attempts should sample enough blocks to collect SEVERAL tail
cases and compare them against matched controls, rather than reasoning from one.

## ROUND 5 (mechanism): the tail is UNDER-RESOLUTION of narrow corridors

First, a correction to round 3's risk list. It named "4-neighbour grid anisotropy" as a candidate
cause. **That is the wrong worry for a conduction problem** — the 5-point stencil is a consistent
ISOTROPIC discretization of the Laplacian; anisotropy bites shortest-PATH problems, not conduction.
The real geometric error is the **staircase representation of curved disk boundaries**, which is
worst in narrow corridors, which is exactly where `eps` operates.

That reframing predicts the two knobs are two faces of one error: on a tail block, permeability
should fail to converge in `h` at fixed `eps` as well. Tested on the known outlier against a
well-behaved control (`scratchpad/continuum/tail_mechanism.py`):

    TAIL ...41829 (105 parcels)              CONTROL ...19366 (88 parcels)
      eps  h=1.0   0.5    0.35   0.25          eps  h=1.0   0.5    0.35   0.25
      0.5  0.604  0.621  0.626  0.670          0.5  0.658  0.659  0.661  0.660
           finest two differ by 0.04474             finest two differ by 0.00067
      1.0  0.643  0.686  0.690  0.696          1.0  0.658  0.658  0.658  0.659
           finest two differ by 0.00666             finest two differ by 0.00050

**Confirmed.** The tail block does not converge in `h` either — still moving 0.045 between the two
finest grids at `eps = 0.5`. Raising `eps` to 1.0 improves its `h`-convergence **6.7x**
(0.0447 -> 0.0067). The control is flat throughout, at both `eps`.

So `eps` sensitivity and `h` sensitivity are **the same phenomenon**: corridors too narrow for the
grid to resolve. Widening them with `eps` makes them resolvable, and then `h` converges. The tail is
not a defect of the continuum FORMULATION — it is under-resolution of the DISCRETIZATION.

### What that changes

The requirement stops being "choose a good `eps`" and becomes the standard meshing criterion: **cell
size must be small relative to the narrowest corridor carrying significant flow.** That is a
resolution rule, not a free parameter, and it can be checked per block rather than assumed.

It also names the fix. A **body-fitted triangulation** resolves curved disk boundaries directly
instead of by staircase, and can grade element size to local corridor width — ordinary FEM practice.
That would demote `eps` from a structural knob to, at most, a degeneracy guard, and would remove the
staircase error that is generating both sensitivities.

Not yet established: this is ONE tail block against ONE control. The mechanism is coherent and the
prediction held, but whether every tail block is under-resolution — and whether a triangulation
actually fixes it — are both untested.

## ROUND 6 (the gate): round 5's mechanism does NOT generalize — and the right question was different

Round 5 predicted, from one tail block, that `eps` and `h` sensitivity are one error. Tested as a
population claim on 40 blocks, taking the 4 most and 4 least `eps`-sensitive
(`scratchpad/continuum/mechanism_gate.py`):

    control  median eps_sens 0.00084   median h_sens 0.00060
    tail     median eps_sens 0.01634   median h_sens 0.00056     <- SAME h_sens as controls

    Spearman(eps_sens, h_sens) = +0.214, p = 0.61
    Mann-Whitney h_sens (tail > control): p = 0.44    ratio of medians 0.9x

**FALSIFIED.** The two do not covary. Every block in this sample — tail or control — converges
cleanly in `h` (all `h_sens <= 0.0018`, progressions flat). The sample's worst `eps_sens` was 0.0238
against 41829's 0.066, so it contained no true outlier. Two distinct phenomena, which round 5
conflated:

- **common**: `h` converges; `eps` still moves permeability ~0.007 median / 0.024 max. That is `eps`
  doing genuine MODELLING work, not compensating for under-resolution.
- **rare (41829)**: both knobs move it. Genuinely under-resolved, and rare enough that 40 blocks did
  not contain one.

So a body-fitted triangulation would fix the rare case, **not** the common one. Round 5's "the
triangulation is the answer" is retracted: `eps` is a modelling parameter and no meshing removes it.

### The question that should have been asked first

Every `eps` measurement to this point was ABSOLUTE sensitivity on ONE method. Permeability is used
COMPARATIVELY — both lenses compare methods on the same block. So what matters is whether `eps`
reorders methods, not whether it moves the level. Measured over 10 blocks x 5 methods
(`scratchpad/continuum/eps_ranking.py`):

    within-block spread of the eps shift ACROSS methods: median 0.0156, max 0.0309
      (NOT a uniform level shift -- clearance falls while flow_paths rises on the same block)

    method RANK changes:             0 of 50 cells (0.0%)
    blocks where the WINNER changes: 0 of 10
    per-block Kendall tau:           median +1.000, MIN +1.000

**`eps` moves levels differentially and never reorders a single pair.** It is a NUISANCE parameter
for comparisons, provided it is held fixed across a comparison — trivially satisfiable.

Scope: 10 blocks, 5 methods, `eps` in {0.5, 1.0}, blocks <= 150 parcels, no known outlier included.
A wider `eps` range and a tail block would strengthen it. But `tau = +1.000` with `min = +1.000` is
as clean as this measurement can come out.

### Status after the gate

Healthier than rounds 3-5 implied. `eps` affects the absolute value but not the ordering; `h`
converges on the common population; the rare under-resolved block is detectable per-block (its `h`
sweep does not converge) and fixable with finer `h` there. What remains is method breadth beyond
these five, a wider `eps` range, and behaviour on a known outlier.

## A5 GATE 2026-08-06: FAILED. `eps` DOES reorder methods.

Round 6 concluded `eps` is a nuisance parameter from 10 blocks x 5 methods at ONE eps pair
({0.5, 1.0}): 0 of 50 rank flips, Kendall tau median and min +1.000. The whole continuum spec was
built on that. Widened to the acceptance criterion's sample -- **21 blocks x 6 methods x 4 eps
values, all 21 blocks complete** -- it collapses:

    eps pair      flips  winners   min tau   blocks with any flip
    0.25 vs 0.5      19       1     +0.600         9 / 21
    0.25 vs 1.0      27       1     +0.600        11 / 21
    0.25 vs 1.5      29       1     +0.600        12 / 21
    0.5  vs 1.0      17       0     +0.733         8 / 21     <- the "safe" band
    0.5  vs 1.5      19       0     +0.600         7 / 21
    1.0  vs 1.5       8       0     +0.733         3 / 21

    TOTAL: 119 rank flips, 3 winner changes, min Kendall tau +0.600

**Every pair reorders**, including 0.5 vs 1.0 where round 6 measured none. Example
(`ZAF.9.3.1_1_38525`): `flow_paths` 0.5513 -> 0.6040 while `clearance` 0.5958 -> 0.5919, moving
`flow_paths` from last place to third. A genuine reorder, not a tie-break.

### Why round 6 missed it

Three ways at once, each of which shrank the chance of seeing a flip: 10 blocks not 21; 5 methods
not 6 (`clearance_looped`, which round 6 omitted, is involved in flips); and ONE eps pair not six.
The reported "0 of 50 cells" was 10 blocks x 5 methods x 1 pair = 50 comparisons against the 756 the
criterion actually asks for.

### What it means

`eps` is **not** a nuisance parameter. It is a free modelling constant that changes which method
wins, and it cannot be measured from point + `NN/2`-disk geometry — that is exactly what it exists
to paper over. So the continuum metric as specified cannot be shipped: two people choosing
defensible values of `eps` would publish different method rankings.

`specs/2026-08-06-continuum-permeability-design.md` is BLOCKED. Its A5 was written to be the
criterion that could kill it, placed first in the plan for that reason, and it did its job at the
cost of one task instead of ten.

**Real building footprints are now a PREREQUISITE, not an improvement.** With measured footprints
there is no `eps` to choose: free space is what the polygons leave. `data/provision.py:57` records
that Open Buildings polygons exist and were declined on size (*"points; the polygon variants are
14.09 GB"*).

## Displacement coupling: decoupling IS a confound, measured

If circulation shrinks disks by `eps` but `displacement` does not, the two axes disagree about the
same geometry. Measured over 10 blocks x 6 methods
(`scratchpad/continuum/displacement_coupling.py`), the disagreement is **not** a uniform level shift:

    method                    disp@eps=0   ratio@0.5   ratio@1.0
    flow_paths_noreinforce        0.2070      0.918       0.823   <- shrinks most
    topology                      0.4184      0.920       0.826
    clearance                     0.2446      0.947       0.881
    clearance_grid                0.2351      0.947       0.886
    euclidean_grid                0.4445      0.955       0.907
    clearance_looped              0.6419      0.964       0.922   <- shrinks least

    eps=1.0: ratio spans 0.823-0.922 (spread 11.3% of mean); 10% of per-block method ranks flip

The direction matches the predicted mechanism: `topology` and `flow_paths` thread tight gaps, so
their displacement is most sensitive to assumed building size. **So couple them** — one radius for
both axes. The cost is that published displacement moves 8-18%, which is arguably toward reality
since `NN/2` is a packing UPPER BOUND rather than an estimate.

## What still has to clear before building

Cost is settled and displacement coupling is decided. What remains:

- **THE BLOCKER: `eps` and `h` sensitivity on the tail blocks** (round 3). Median behaviour is fine;
  the worst block moves 0.066 over `eps` 0.5-1.0 and 0.043 over `h`, against a 0.118 signal.
  Diagnosing `41829` is the entry point — what distinguishes a block where `eps` never plateaus from
  the four where it does?
- **Region-scale MEMORY — CLEARED.** Measured directly rather than extrapolated
  (`scratchpad/continuum/memory_test.py`): at 3.42M cells, matching the region at `h = 0.5`,
  `spsolve` takes **60.4 s and peaks at 6.35 GB** against 227 GB available. Scaling of peak RSS is
  1.65 / 3.60 / 6.35 GB at 1M / 2M / 3.4M, so even `h = 0.25` (14.1M) projects to ~30 GB. Time, not
  memory, is the cost. The earlier 42 s/solve extrapolation was ~30% optimistic; ~40 min per
  40-solve lens curve is the honest figure.
- **Method breadth.** All spike work so far uses `clearance` only. Nothing yet shows the metric
  ranks methods sanely.
- ~~**Grid anisotropy.**~~ Retracted in round 5: the 5-point stencil is isotropic for conduction.
  The real geometric error is staircase boundary representation, now identified as the tail's cause.

## Real footprints remain the deeper fix

`data/provision.py:57` records that Open Buildings polygons exist and were declined on size —
*"points; the polygon variants are 14.09 GB"*. `eps` is a regularization standing in for a
measurement. Real footprints would also, independently:

- make `displacement` real — already a known gap, cut from Phase 0 because building points
  under-count ("a road clipping a parcel corner displaces 0 while destroying the home");
- retire `building_radii`'s `NN/2` fiction, `radius_frac`, `FOOTPATH_EPS`, and `parcel_radii`'s
  containment join, rather than requiring better values for any of them;
- allow parcels to become the generalized Voronoi diagram of polygons instead of points.

That is the change with the serendipitous knock-ons, and it would let `eps` be measured rather than
chosen. But it is no longer a PREREQUISITE: the minimum-separation regularization makes the
continuum converge on the geometry already on disk.

## Scope

Two Cape Town blocks, one method (`clearance`), 4-neighbour grid, uniform `sigma` per cell, four
grid spacings. Small. The round-1 failure mode was predicted in advance by the 10%-zero-gap
measurement and appeared in both blocks; the round-2 rescue is likewise consistent across both, at
two values of `eps`. Enough to keep going and to justify a wider spike; nowhere near enough to
commit to a metric rewrite.

## Correction recorded

Round 1 of this note asserted the failure was "structural in the input, not a tuning problem, so no
resolution choice or floor rescues it without reintroducing exactly the `eps` fudge the design was
meant to remove." That conclusion was too strong and is retracted. A minimum separation does rescue
it, does not reintroduce a dimensionless fudge (it is a length), and is measurably not the dominant
term. The general lesson is narrower than the one first drawn: raw `NN/2` disks are unusable as a
continuum domain, which is not the same as the input being unusable.
