# Roadless clearing: experiment backlog

A living list: add ideas, strike them when tried (the result goes to NOTES.md, with numbers),
delete them when shown dominated. Status tags: **prioritized** (the owner wants it done),
**queued**, **running**, **idea**, **owner's call**, **blocked**.

Baselines (NOTES.md, "Large blocks" and "Lens A at D 0.05"): greedy `S0.01cat` /
`S0.005cat` and SIMP + eps-scored incumbent `fw0.q3.i10.t0.001.e0.0001.k1s1e-06`, both J_2,
uni conductance, Lens A at D 0.05 (`D_LENS`; owner 2026-10-03, 0.10 saturates under sightline),
and Lens B from nested runs.

## Status of the leads (owner 2026-10-06: "leave no stone unturned. Every lead should be followed")

Presets now (NOTES, 2026-10-05 -- 06; D 0.05, J_2, uni):
- **Default: `.p256w8x2`** -- SIMP base + the undecided-set polish with pair moves. The 13 +0.0097
  at 1.26x, the 46 held out +0.004 vs base (ahead 89%, never behind).
- **Hard blocks: `.p256w8x2.GS0.01cate1`** -- plus the greedy as a second polished track, polished
  only when its raw clearing is within 1% of SIMP's polished one. The 13 +0.0119 at 1.21x the
  default (the greedy unscored); identical to the default on all 46 held out.
- **Cheap: greedy + polish** (polish_greedy.py, P64w8) -- the 13 -0.004 at 0.81x, the 46 -0.015.

Closed (measured, NOTES): multi-rounding `.r<n>` (held out +0.001 at 2x; late `.r2l2` the same at
1.58x); coarse-to-fine `.c0.75` (keeps gates at 0.85x, but with the polish dominated); coarse
multi-start `.C<h>.m<k>` and fine `.m<k>` (neither faster nor better); the two-phase polish `.pP`
(pairs reach the same blocks); seeding `.g` (loses where a better start polishes worse: `.G`
instead); greedy screening width `S0.01catw4` (+0.0006, marginal); pairs after the greedy (+0.0005,
marginal); resolution (res-check: h 0.75 within 0.01 on 35 of 59, h 1.0 understates gated blocks by
up to 0.31); sequential stopping (seqscreen.py: by the value of more blocks, 15 -- 24 of 46).

Also closed: the oversized blocks (the 17 over 1.4 km^2 and 53556), base and default at the
metric's resolution through local coarsening (`.a5x8`; 4287 `.a5x32`, 63612 `.a2.5x8`; NOTES, "The
oversized blocks at the metric's resolution"); 19593 and 6498 on the H100; 14401, 7851 and 32841's
default at a5x8. Every one of the 82 large blocks has the default at h 0.5 near its buildings.

Also closed: wider polish rounds (w16 = w8 at 1.28x) and late sampling on top of the hard-block
preset (+0.0001 at 1.33x); the greedy's unused scoring skipped (the preset 1.30x -> 1.21x).

Prioritized (owner 2026-10-08: "I want to do all of the following"), each entry below:
- **Goal-oriented refinement** (Coverage and model): done, the o1 goal mesh, on the 59 and the
  oversized blocks.
- **Sightline and the greedy on the composite** (Coverage and model): done, on the 59.
- **The cheap preset on the composite** (Coverage and model): done, on the oversized blocks.
- **The add/remove search as one Strategy** (New methods): built, with floating search.
- **An outcome model for kernelcore** (Experiment design): dropped on review (owner
  2026-10-09); the stop is built in seqscreen.py instead, by the value of more blocks (NOTES,
  "Stopping by the value of more blocks").

Open:
- **The rate bracket** (seqscreen.RATES, 2e-6 -- 1.9e-5 Lens A per GPU-second). A screen whose
  call turns on the rate prints an "ask" (this much Lens A for this much more time?); the
  owner's answer moves an end of the bracket to that screen's break-even rate. None is live.

Not pursued, with what would change that:
- *A Galerkin coarse model* (the solver's AMG level as the design problem's model, which keeps a
  sub-cell gate's conductance where re-gridding loses it): coarse runs were not cheap enough to pay
  even where they kept gates (`.C0.75.m2` 1.36x, dominated) -- worth it only if a coarse model were
  several times cheaper per update than h 0.75 and kept every gate.
- *The greedy's sweep kernel* (results identical, ~20 -- 30% of the greedy): the greedy is now a
  0.3 -- 0.4x add-on of the hard-block preset; worth it if the greedy becomes the main method.

## How we run experiments

- **Tune on the 13, report on the 44.** The 13 tuning blocks (6 gated collapses, 1558, 6
  controls incl. 5810) were used to choose the SIMP plan; the other 44 of the 56 large blocks
  that fit are untouched. Choose settings on the 13; the claim is the 44.
- **Screen before the full run.** Run a variant first on a small subset (the 13, or a 4-block
  sentinel: 22422 gated, 30848 gated and large, 5810 biggest, 9712 ordinary) and stop it if
  any block collapses (> 0.1 below its baseline) or its median time exceeds 3x the baseline's.
  Only survivors get the 44.
- **One GPU process at a time** (large blocks), queued in a detached script.
- **Sequential stopping** (done 2026-10-09). seqscreen.py: the Bayesian bootstrap's posterior
  on a variant's mean paired difference, a stop when no number of further blocks is worth its
  GPU time (the exact value of more blocks through the posterior mean's Polya urn), a 15-block
  floor and the collapse veto. kernelcore's reweighting collapses on these screens (NOTES).

## Greedy

- **Exchange refinement**: now a special case of the add/remove search below.
- **Batch size** (done: S0.005cat on the 44, +0.003 median, ahead 82%, 1.6x time; NOTES). Sentinel screen, Lens A at D 0.10, S0.005 /
  S0.01 / S0.02: 30848 0.776 / 0.770 / 0.750, 5810 0.329 / 0.319 / 0.306, 9712 0.181 / 0.176 /
  0.170, 22422 0.815 / 0.814 / -; monotone, smaller is better at 1.3 -- 3x the time; 0.02
  dropped from the full run.
- **Screening width** (idea). Exact-score more top candidates per round (as `M4` does).
- **Speed** (measured 2026-10-04, gramprobe.py and a timing probe; NOTES "Greedy speed"). A step
  on 30848 is 8 -- 12 s: the ranking (two solves at rtol 1e-3) 3 -- 4.5 s, the gram 1 -- 7 s,
  scoring 2 -- 6 s.
  - Scoring's rtol 1e-5 -> 1e-3 moves perm by at most 3.5e-5 and cannot change a pick (the
    picker ignores the scoring J). It is 5 -- 10% of a step: take it with the next change.
  - The gram scales with candidates x unknowns, about 0.03 s per candidate on 12 -- 16M
    unknowns. The waves do not matter (Kahn's pass is 0.1 -- 0.25 s for 3 -- 7K waves), so
    batching wave launches is dead.
  - The levers left: fewer candidates (the picker's `reach`, which changes picks), or a sweep
    kernel that reads each node's neighbour indices once rather than once per candidate column
    (results identical; GPU kernel work).
  - Altogether perhaps 20 -- 30% off the greedy, not 2x.

## SIMP

- **Cheaper incumbent** (done: `.k1s1e-06` on the 44 equals `.k1`, 15% faster; NOTES). Candidates scored in the eps world
  at eps 1e-6 (the relaxation's cached structure), the winner exactly. Sentinel: the same
  scores (0.8144 / 0.7822 / 0.3476 / 0.1871 vs 0.814 / 0.782 / 0.348 / 0.187), 10 -- 25%
  faster. Every 2nd iterate (`.k2`) is dead: it collapses on 22422, whose gate a single odd
  iterate holds (NOTES, "SIMP: every 2nd candidate, coarse-to-fine, damped OC, MMA").
- **Warm nested path** (measured, `.w3`: 2.6 -- 2.9x faster, Lens A 0.002 -- 0.022 below the
  cold path on 3 blocks; a cheaper preset; NOTES).
- **Memory** (partly done, 63fa484; 2026-10-04): structures are cached only for a pattern seen
  twice, and the greedy releases its tension before the exact scoring (22422 fits again); 1558,
  20023, 30796 fit under `.k1s1e-06`. Fixed 2026-10-04: the K-cycle's closures formed a reference
  cycle that kept every system's AMG hierarchy (2.6 GB on 30796) until the cycle collector ran,
  about 11 GB of dead hierarchies at the peak (NOTES, "Translucent greedy on 30796"). With a
  release of cupy's cached blocks before each new system, 14401, 7851 and 32841 now fit 48 GB for
  the greedy and SIMP (62 of 82 large blocks). Done: 19593 and 6498 on the H100; 53556, 45267 and
  the other 16 through local coarsening (Status, above).
- **Damped OC or MMA** (done 2026-10-04, null; NOTES). Damped OC ties or trails slightly on the
  13 (median -0.0005 / -0.0001), and MMA collapses on 22422. Both stay selectable (`.u`).
- **Sturdier gate capture** (from that null). On a gated block SIMP's answer is one lucky
  iterate's rounding: `.k2` and MMA each lost 22422 by perturbing the iterate sequence.
  - Several roundings per iterate (done, `.r<n>`, NOTES "SIMP multi-rounding"): `.r2` on the 13
    is ahead on 5 (+0.008 to +0.038) but on the 46 held out only +0.001 mean (none above 0.01),
    never behind, 2x the time: selectable for hard blocks, not a default. It partly rescues
    `.k2`'s and MMA's lost gate (0.55, not 0.78).
  - Cheaper `.r` (done): the winners come from the last two stages' random roundings, so
    `.r2l2` samples only there (1.58x, same gains as `.r2`).
  - Undecided-set polish (done, `.p<tries>w<width>`; NOTES "Late sampling and the polish"), and
    what followed it, all done: the larger cap `.p256w8`, the pair moves `x2` (together the
    default `.p256w8x2`), and the same polish on the greedy's clearing (the hard-block preset's
    second track `.GS0.01cate1`, and polish_greedy.py's cheap preset).
- **Multiple starts** (closed, NOTES "Multi-start on coarse grids"). A Frank-Wolfe start (fw5)
  lands where the uniform start does; a second, random start (`.p64w8.m2`) never beat the uniform
  one (identical on 12 of 13, 2.12x); coarse multi-start `.C<h>.m<k>` neither faster nor better.
- **Projection** (measured, kept selectable: `.b8-32` wins some gated blocks, loses some
  controls; not needed against collapses once the incumbent is on).
- **SIMP + incumbent under the sightline metric** (done: the 13 at D 0.10 and D 0.05, and the 46
  held-out blocks at D 0.05, NOTES "Sightline on the held-out blocks at D 0.05". SIMP leads the
  plain greedy by a median of +0.021 [+0.015, +0.026], ahead on 96% of blocks, and the translucent
  greedy by +0.005 [+0.002, +0.012] (all 46, with 30796 after the memory fix), at 3.1x their
  time. The translucent greedy keeps its gate tail, 18910 -0.165.)

## New methods

- **Add/remove subset search** (built 2026-10-09; NOTES "The add/remove search as one
  Strategy"). search.py: a Round (rank, refine, build, evaluate, accept) under a schedule; the
  greedy's pickers, exchange (the polish), grow-then-prune and floating search are its
  configurations. Floating search (FL3xS0.01catr0.005m8c100) is never below grow-then-prune or
  the greedy on the five blocks and repairs the prune's collapse (22422 at D 0.05: 0.782 against
  0.011), at 1.1 -- 2.2x its time. At one budget (d_max 0.05, the first live screen: NOTES,
  "Floating search at the cheap preset's budget") it is dominated by SIMP's default (-0.011 on
  15 held-out blocks at about 10x the time) and by swings (below); its niche is the whole curve
  in one run. Next rows, two new parts each (spec, "What the two new rows need"): a
  substitute-aware restore (a restore ranking that keeps the Tension, a restore-direction
  diverse builder) and screened adds (an add-direction refiner, a builder that reads the
  refined order). Wiki `pages/methods/plus-l-take-away-r.md`.
- **Swing schedules** (owner 2026-10-09; tested 2026-10-10, NOTES "Swing schedules at the
  cheap preset's budget" and "Swings as SIMP's finisher"). Several grow-then-prune swings past
  the budget, each smaller, then the polish (search.py
  `SW<grow>-<grow>-...x<picker>r<share>m<k>.P<t>w<w>`; from SIMP's answer `SIMP<plan>+SW...`).
  From the greedy they are dominated by SIMP's default (-0.005 at 3x its time; a 3x first swing
  ties 1.5x at 3x the time). As SIMP's finisher (one 1.2x swing and the polish) they are on the
  frontier: +0.0010 mean over SIMP's default on the 46 held out (half of it 30796, +0.024),
  never behind, at SIMP's time again; on the 13 tuning blocks +0.0001. Not a preset: the gain
  is one block's (30796), and no gate on SIMP's answer finds it (NOTES). The typical gain is a
  re-pick of a few buildings worth 0.8% more budget. What would change that: blocks like 30796
  common among the large ones (it is the only held-out block over 3000 buildings). The adaptive
  form waits on a schedule worth adapting. Neighbours: oscillating search, VNS, adaptive LNS
  (wiki page above, "Swing schedules").
- **The restore screen's solve on gated blocks** (2026-10-10, NOTES "Swings as SIMP's
  finisher"). Its eps world (EPS_SCREEN 1e-6) did not converge on 30848 (AMG-CG 2000 iterations,
  1.9e-5 -- 5.0e-5 against 1e-5); every restore round can meet it. Fixes to weigh: warm starts
  from the current state's solution, a looser rtol for a screen that only ranks, a larger
  screen eps.

## Experiment design

- **Adaptive block selection** (first look done, NOTES "Which blocks tell methods apart": collapses dominate; screen = gated sentinel + random sample; picking 'discriminating' blocks beyond the sentinel misleads. Remaining value: sequential sample size; wiki `pages/methods/adaptive-benchmark-items.md`).
  Fit a factor model to the method x block matrix of paired differences (axes like "opens
  gated pockets", "spreads in dense bands"); for a new variant run first the blocks with the
  most expected information per GPU-second about the decision, stop when it is made. The
  headline number still comes from the untouched set; keep some random blocks in every
  screen. Cheap first step: an SVD of the matrix we already have (greedy, SIMP variants,
  translucent search; 220 + 59 blocks) to see which blocks load on which axis.
- *Generalize kernelcore's outcome model* (dropped 2026-10-09). The screens' stop is built in
  seqscreen.py; reweighting draws collapses past about 4 blocks and a prior grid decides the stop
  (NOTES, "Stopping by the value of more blocks"). Would come back with a consumer that must
  choose among blocks that are not exchangeable (the adaptive selection above), designed against
  it.

## Coverage and model

- **Blocks over ~1.4 km^2** (done 2026-10-08): local coarsening (`lifted.CompositeGrid`, plan
  suffix `.a<d0>x<smax>`) keeps h 0.5 within d0 of every building, street and the block edge and
  doubles the cell with each doubling of distance; within 0.004 of uniform h 0.5 on the 59 at a5
  (order kept but for ties), though 14401 reads 0.009 low at a5 (0.0014 at a20: the order still
  kept), and 2.4 -- 5.1M cells on the oversized blocks. Also an idea on top of it: a separate d0
  for streets and the block edge.
- **Goal-oriented refinement** (**prioritized**). Refine where the dual-weighted residual (the
  adjoint, which the tension already solves for, weighting each cell's residual) says Lens A is
  sensitive, instead of by distance: 14401 reads 0.009 low at a5 where the 59 read at most 0.004,
  so distance alone misses some of what the score depends on. Measure: Lens A error against
  uniform h 0.5 per cell spent, on the 59 and 14401, against a5 and a10. Built (common.GoalMesh);
  with nothing cleared in the indicator's world it lost to distance on a third of the 59 (NOTES);
  with the buildings open (o1) it is ahead of distance on 49 of 53 blocks at f1.2 (2.8x less
  error at equal cells, median; 0.11 of the uniform cells fewer at equal error). Its five losers
  are two sign crossings of the distance meshes' bias (20952, 44602: a5 near zero by luck), a tie
  and two mild losses (NOTES). On the 19 oversized blocks at a5's cells it leaves a sixth of a5's
  error (median 4.9x less, ahead on 18, the 19th a tie), against the finest goal mesh that fits.
- **Sightline and the greedy on the composite** (**prioritized**). Sightline and SoftSightline
  scan raster lines (`needs_raster`), as the metric or as the greedy's translucent search; both
  raise on a composite. The greedy under uni has no raster check but has never run on one: first
  establish that it does (against uniform h 0.5 on the 59), then give the sightline scans a
  composite form. Both done (NOTES, "Sightline on the composite"): the scans run on the fine
  raster, painted and averaged per cell; under the sightline metric on the 59 every mesh reads as
  under uni (a20 p95 |error| 0.0006), the goal mesh ahead of distance on 38 of 40 but by less
  (1.9x, 0.05 of the cells; 8152 0.18x, not looked at). Next: the sightline presets on the
  oversized blocks.
- **The cheap preset on the composite** (**prioritized**). polish_greedy.py builds a uniform h 0.5
  grid; it takes a mesh, and greedy + polish runs on the oversized blocks. Needs the greedy on the
  composite (above).
- **Coarse-to-fine** (`.c1`, tried 2026-10-04; NOTES). 0.75x the time, but it collapses on 30848
  (-0.255). Kept selectable. Worth retrying only with a coarse h that keeps 30848's gate (0.75?)
  or with only the first stage coarse.

## Infrastructure

- **Faster submits** (done 2026-10-08; NOTES "Faster submits"): a staged `dwithin` near test, a
  count without the cells' attributes, 8 blocks counted at a time; sizing 16 -- 18x faster, the
  meshes identical. A goal-oriented mesh, whose cells come from a solve, will need its own
  `cells` (its cell budget, say) rather than a build at submit.
- **Cluster launcher: bookgen and mycooc on GeoffChurch/cluster_submit** (done): mycooc deleted
  its old launcher (3ad93ac5), bookgen deleted ltcluster with no compatibility path (b901184b).
