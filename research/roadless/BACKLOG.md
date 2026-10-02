# Roadless clearing: experiment backlog

A living list: add ideas, strike them when tried (the result goes to NOTES.md, with numbers),
delete them when shown dominated. Status tags: **queued**, **running**, **idea**, **owner's
call**, **blocked**.

Baselines (NOTES.md, "Large blocks"): greedy `S0.01cat` and SIMP + incumbent
`fw0.q3.i10.t0.001.e0.0001.k1`, both J_2, uni conductance, Lens A at D 0.10 (and Lens B from
nested runs).

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

- **Exchange refinement at the budget** (idea, top pick). After the greedy reaches D, swap one
  cleared for one uncleared building while exact J improves; add candidates by tension, drop
  candidates by the adjoint's closing loss on cleared buildings. Targets the greedy's myopia
  (where SIMP wins). Also applies to SIMP's incumbent.
- **Batch size** (queued). S0.005cat and S0.02cat vs S0.01cat: chosen on the 220 small
  blocks; large blocks put many more buildings in a round.
- **Screening width** (idea). Exact-score more top candidates per round (as `M4` does).
- **Speed** (idea). Remeasure: the structure cache and single-precision K-cycle apply to its
  tension solves (its 25 s median is stale). Its per-round exact rescoring has a new pattern
  each round: measure the tolerance it needs.

## SIMP

- **Cheaper incumbent** (queued). Score candidates at rtol 1e-6, rescore the winner at 1e-9;
  or every 2nd update (`.k2`). The exact solves are about half of `.k1`'s time.
- **Warm nested path** (idea). With the incumbent keeping good clearings, warm-starting each
  budget from the last may no longer trap (it did without: -0.076 Lens A); would cut the
  path's 15x.
- **Memory** (idea). The exact scoring adds a second structure to the caches: 1558, 20023,
  30796 now run out of memory. Smaller LRU, or free the eps-world structures around scoring.
- **Damped OC or MMA** (idea). The gate and its substitutes flip x 0.3 <-> 0.5 every update
  under OC; a damped update or MMA should converge cleanly and propose better roundings.
- **Multiple starts** (idea). Uniform and Frank-Wolfe starts, best by the incumbent (~2x).
- **Projection** (measured, kept selectable: `.b8-32` wins some gated blocks, loses some
  controls; not needed against collapses once the incumbent is on).
- **SIMP + incumbent under the sightline metric**, and with the translucent-search gradient
  (idea; translucent search beat SIMP under ss100k2n2r30 before the incumbent existed).

## New methods

- **Grow then prune** (idea; from the mycooc vocabulary work, wiki
  `pages/mycooc/experiments/lattice_pool_em_audit-results.md`: an overcomplete seed pool 3x
  the target, pruned 20% per round by expected count under a refitted model, beat top-K by
  5-10 F1 points). Here: clear ~3x the budget (greedy or SIMP at D 0.3, or every building with
  positive tension), then reopen the least valuable batch per round, re-solving between
  rounds, down to D. A gate and its companions are judged in each other's company, the
  greedy's myopia from the other side. Risk: substitutes (three alternative gates each look
  removable when all are open) pruned in the same batch; small batches, re-solve per round,
  and the catchment overlap discount guard against it.

## Coverage and model

- **Blocks over ~1.4 km^2** (owner's call). 21 -- 23 of the 82 large blocks do not fit 48 GB at
  h 0.5: a coarser grid, or the grid cropped to the built-up area. Both change the model.
- **Coarse-to-fine** (idea). Early SIMP stages at h 1.0 (4x fewer unknowns), the last at 0.5.
