# Roadless clearing: experiment backlog

A living list: add ideas, strike them when tried (the result goes to NOTES.md, with numbers),
delete them when shown dominated. Status tags: **queued**, **running**, **idea**, **owner's
call**, **blocked**.

Baselines (NOTES.md, "Large blocks" and "Lens A at D 0.05"): greedy `S0.01cat` /
`S0.005cat` and SIMP + eps-scored incumbent `fw0.q3.i10.t0.001.e0.0001.k1s1e-06`, both J_2,
uni conductance, Lens A at D 0.05 (`D_LENS`; owner 2026-10-03, 0.10 saturates under sightline),
and Lens B from nested runs.

## How we run experiments

- **Tune on the 13, report on the 44.** The 13 tuning blocks (6 gated collapses, 1558, 6
  controls incl. 5810) were used to choose the SIMP plan; the other 44 of the 56 large blocks
  that fit are untouched. Choose settings on the 13; the claim is the 44.
- **Screen before the full run.** Run a variant first on a small subset (the 13, or a 4-block
  sentinel: 22422 gated, 30848 gated and large, 5810 biggest, 9712 ordinary) and stop it if
  any block collapses (> 0.1 below its baseline) or its median time exceeds 3x the baseline's.
  Only survivors get the 44.
- **One GPU process at a time** (large blocks), queued in a detached script.
- **Sequential stopping (idea).** Replace the fixed screen with a posterior on each variant's
  mean paired difference (Student-t or contaminated-normal likelihood for the -0.8 collapses),
  futility stop when P(mean < 0) or P(collapse) is high, and stop sampling when the value of
  the next blocks per GPU-second falls below cost. Ideas from GeoffChurch/kernelcore
  (`selection.value_of_calls`: EVSI by reweighting posterior draws over simulated outcomes,
  greedy picks valued on fresh patterns) and GeoffChurch/bookgen
  (`arena/arena/eval/stopping.py`, `ValueOfInformation`). Their code is for binary pairwise
  verdicts: reuse the idea, write ~100 lines of numpy here.

## Greedy

- **Exchange refinement**: now a special case of the add/remove search below.
- **Batch size** (done: S0.005cat on the 44, +0.003 median, ahead 82%, 1.6x time; NOTES). Sentinel screen, Lens A at D 0.10, S0.005 /
  S0.01 / S0.02: 30848 0.776 / 0.770 / 0.750, 5810 0.329 / 0.319 / 0.306, 9712 0.181 / 0.176 /
  0.170, 22422 0.815 / 0.814 / -; monotone, smaller is better at 1.3 -- 3x the time; 0.02
  dropped from the full run.
- **Screening width** (idea). Exact-score more top candidates per round (as `M4` does).
- **Speed** (idea). Remeasure: the structure cache and single-precision K-cycle apply to its
  tension solves (its 25 s median is stale). Its per-round exact rescoring has a new pattern
  each round: measure the tolerance it needs.

## SIMP

- **Cheaper incumbent** (done: `.k1s1e-06` on the 44 equals `.k1`, 15% faster; NOTES). Candidates scored in the eps world
  at eps 1e-6 (the relaxation's cached structure), the winner exactly. Sentinel: the same
  scores (0.8144 / 0.7822 / 0.3476 / 0.1871 vs 0.814 / 0.782 / 0.348 / 0.187), 10 -- 25%
  faster. Not yet tried: every 2nd update (`.k2`).
- **Warm nested path** (measured, `.w3`: 2.6 -- 2.9x faster, Lens A 0.002 -- 0.022 below the
  cold path on 3 blocks; a cheaper preset; NOTES).
- **Memory** (partly done, 63fa484; 2026-10-04): structures are cached only for a pattern seen
  twice, and the greedy releases its tension before the exact scoring (22422 fits again); 1558,
  20023, 30796 fit under `.k1s1e-06`. Fixed 2026-10-04: the K-cycle's closures formed a reference
  cycle that kept every system's AMG hierarchy (2.6 GB on 30796) until the cycle collector ran,
  about 11 GB of dead hierarchies at the peak (NOTES, "Translucent greedy on 30796"). **Running:**
  the greedy and SIMP on the three left-out blocks within reach of 48 GB (14401, 7851, 32841).
- **Damped OC or MMA** (idea). The gate and its substitutes flip x 0.3 <-> 0.5 every update
  under OC; a damped update or MMA should converge cleanly and propose better roundings.
- **Multiple starts** (idea). Uniform and Frank-Wolfe starts, best by the incumbent (~2x).
- **Projection** (measured, kept selectable: `.b8-32` wins some gated blocks, loses some
  controls; not needed against collapses once the incumbent is on).
- **SIMP + incumbent under the sightline metric** (done: the 13 at D 0.10 and D 0.05, and the 46
  held-out blocks at D 0.05, NOTES "Sightline on the held-out blocks at D 0.05". SIMP leads the
  plain greedy by a median of +0.021 [+0.015, +0.026], ahead on 96% of blocks, and the translucent
  greedy by +0.005 [+0.002, +0.012] (all 46, with 30796 after the memory fix), at 3.1x their
  time. The translucent greedy keeps its gate tail, 18910 -0.165.)

## New methods

- **Add/remove subset search** (grow-then-prune measured: dominated by the SIMP path for now,
  collapses on gated blocks through substitutes and is slow; NOTES "Warm SIMP path and
  grow-then-prune". Next schedule to try: floating (re-add after a restore that hurts), and a
  substitute-aware restore score; wiki `pages/methods/plus-l-take-away-r.md`).
  One family, plus-l take-away-r / floating search: a schedule of (add a, remove r) moves with
  an add scorer (tension, Spread's overlap discount), a remove scorer (the adjoint's closing
  loss on cleared buildings, the same discount for removal batches) and an acceptance rule
  (exact J). Special cases: the greedy (add batches, remove none), exchange refinement (at D,
  add 1 remove 1, accept if better), grow then prune (add to ~3x D, then remove batches back
  to D: the mycooc vocabulary recipe, `pages/mycooc/experiments/lattice_pool_em_audit-results.md`,
  overcomplete pool 3x the target, 20% pruned per round with a refit, beat top-K by 5-10 F1),
  floating search (add 1, remove while it helps). Removal judges a gate in its companions'
  company (the greedy's blind spot); its risk is substitutes, each removable alone, dropped
  together in one batch: small batches, a re-solve per round, the overlap discount. Build it
  as one Strategy (schedule x scorers x acceptance), the greedy becoming its first preset.

## Experiment design

- **Adaptive block selection** (first look done, NOTES "Which blocks tell methods apart": collapses dominate; screen = gated sentinel + random sample; picking 'discriminating' blocks beyond the sentinel misleads. Remaining value: sequential sample size; wiki `pages/methods/adaptive-benchmark-items.md`).
  Fit a factor model to the method x block matrix of paired differences (axes like "opens
  gated pockets", "spreads in dense bands"); for a new variant run first the blocks with the
  most expected information per GPU-second about the decision, stop when it is made. The
  headline number still comes from the untouched set; keep some random blocks in every
  screen. Cheap first step: an SVD of the matrix we already have (greedy, SIMP variants,
  translucent search; 220 + 59 blocks) to see which blocks load on which axis.
- **Generalize kernelcore's outcome model** (idea, other repo). Its EVSI reweighting is
  general; its likelihood is binary verdicts. A continuous heavy-tailed outcome model would
  serve this backlog's screens and adaptive selection.

## Coverage and model

- **Blocks over ~1.4 km^2** (owner's call). 21 -- 23 of the 82 large blocks did not fit 48 GB at
  h 0.5: a coarser grid, or the grid cropped to the built-up area. Both change the model. The
  2026-10-04 memory fix brings only three within reach of 48 GB (Memory, above). A few at 100 --
  125M unknowns might fit 80 GB, and the rest (130M -- 1.66 billion unknowns) still need one of
  the two.
- **Coarse-to-fine** (idea). Early SIMP stages at h 1.0 (4x fewer unknowns), the last at 0.5.

## Infrastructure

- **Cluster launcher: moving bookgen and mycooc onto GeoffChurch/cluster_submit** (reblock moved
  2026-10-03). The package and its migration guides (`docs/migrating/{bookgen,mycooc}.md` there)
  hold what each still needs; their old tools print a deprecation notice.
