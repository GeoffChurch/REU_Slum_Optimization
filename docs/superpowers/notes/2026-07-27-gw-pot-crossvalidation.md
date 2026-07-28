# Cross-validating the hand-rolled GW core against POT

**Date:** 2026-07-27
**Status:** done. Found and fixed a real defect; the pair-matrix conclusion survives and
strengthens. The committed benchmark (`data/benchmarks/gw_pair_matrix.parquet`) was regenerated.

## Why this ran before anything else

`scratchpad/ot/ot_gw.py` is 108 hand-rolled lines — entropic Gromov-Wasserstein with an unbalanced
log-domain Sinkhorn inner solve — written that way because POT was not installed. It had only ever
been validated *self-consistently*: the 2026-07-23 self-transplant check shows the pipeline is
coherent, not that the solver is correct. Everything downstream rests on it. `real_gw_dist` is the
independent variable in the pair-matrix's within-recipient slope, and every number in the
2026-07-23 barycenter-consensus study came out of the same 108 lines.

So: cheap check, large blast radius, run it first.

## Verdict

The core is **correct**, with one genuine defect found and fixed:

| layer | what it checks | result |
|---|---|---|
| L1 | inner unbalanced Sinkhorn | **correct** — lowest primal objective in all 9 configurations |
| L2 | `gw_cost` objective | **exact** — machine precision vs. brute-force `einsum` |
| L3 | outer projected-gradient loop | **defect found**: missing factor of 2 in the gradient |
| L4 | does β survive an independent solver? | **yes**, and strengthens |
| L5 | per-iteration `cost.min()` shift | benign — measured, not assumed |

Reproduce with `pixi run python scratchpad/ot/pot_crossval.py` (L1–L3, L5; seconds) and
`pixi run python scratchpad/ot/pot_crossval_pairs.py` + `l4_analyze.py` (L4; ~9 min).

## L3: the defect

Peyré–Cuturi–Solomon Eq. 6 defines the tensor `L ⊗ T`; Prop. 2 gives the **gradient**, which is
what an outer loop actually needs, and it carries a factor of 2 the paper's own statement omits.
POT ships that correction explicitly — `gwggrad = 2 * tensor_product`, with the source comment
*"[12] Prop. 2 misses a 2 factor"*. `ot_gw._gw_gradient_cost` returned the undoubled tensor and fed
it straight to Sinkhorn as the cost.

Sinkhorn at `(cost, ε, τ)` and `(2·cost, 2ε, 2τ)` have the same argmin, so the effect was silent:
the solver **ran at twice the requested entropic regularization and twice the requested τ**. Nominal
ε=0.01 bought an effective ε=0.02 — outside the "ε ≤ 0.01" the brief called non-negotiable.

Nothing looked wrong, because nothing *was* wrong except the calibration. The test that caught it
discriminates in both directions, which is the only reason it is trustworthy:

| | vs. POT at ε | vs. POT at 2ε |
|---|---|---|
| before fix | 0.34 – 0.41 | **6e-8 – 2e-7** |
| after fix | **7e-8 – 2e-7** | 0.34 – 0.41 |

Exactly one of the two is at machine level in each row, and the fix flips which. Our `gw_cost` on
POT's own coupling also equals POT's `gwloss` on it to all printed digits, so the objective
functions agree too — the disagreement was purely the gradient's scale.

## L1: the inner solver is right, and POT's is not usable here

Nine configurations — `(n,m)` ∈ {12×12, 30×45, 60×20} × τ ∈ {1, 10, 10⁴}, ε=0.01. Couplings from
three solvers, each scored on the primal UOT functional
`⟨C,π⟩ + ε·Σπ(logπ−1) + τ·KL(π1|p) + τ·KL(πᵀ1|q)`. An agreement test can only say two solvers
differ; scoring the objective says which is **wrong**.

Ours attains the lowest objective in **all nine**. POT's `sinkhorn_stabilized` diverges at ε=0.01
(objective up to 2.3e20, coupling mass 2.8e17, and it warns as much); POT's `lbfgsb_unbalanced` — an
independent optimizer, not another Sinkhorn — lands 0.0002–0.0004 above ours everywhere, i.e. agrees
to its own convergence tolerance without ever beating us. At the operating point (τ=1) two of the
three shapes match POT to 5e-11.

This also settles the migration question: POT has **no unbalanced entropic GW at all**, and the
hand-rolled solver is ~5× faster (82 s vs. 429 s over the same 100 pairs). Keeping `ot_gw.py` is a
live engineering choice, not inherited history.

## L5: the min-shift is benign

`entropic_gw_unbalanced` subtracts `cost.min()` each outer iteration. That is exactly invariant
under balanced Sinkhorn (the potentials absorb a constant) but **not** under UOT, where a constant
cost offset trades directly against created mass — and τ=1.0 is the operating point, so the concern
was not hypothetical. Measured over four cloud shapes: the objective moves by 1e-5 to 4e-5 on
objectives of order 0.03–0.05 (~0.05% relative), and the sign is mixed — two shapes favour the
shift, two favour its removal. No measurable harm; it stays.

Recorded so nobody re-litigates it. This was the second defect hypothesis of the same species as
L3's, and it was tested rather than assumed.

## L4: does the conclusion survive?

Two independent substitutions, so a surviving slope is not the same code agreeing with itself.

**Substituting the solver.** Recomputing `real_gw_dist` for all 100 committed pairs with POT's
balanced GW (at matched calibration): Pearson **0.9970**, Spearman **0.9956**, median |relative
difference| 2.4%. Re-running the regression on POT's distances gives β = **−9.40** (p=0.018) against
the hand-rolled −9.23 (p=0.026) — the effect is not an artifact of the solver.

Rebuilding the parcel clouds reproduced every stored `real_gw_dist` to **0.000e+00**, which also
confirms the pipeline is deterministic.

**Substituting the estimator.** `statsmodels` OLS with recipient dummies reproduces
`pair_matrix.within_recipient_regression` **exactly** — β, SE, t, dof and p identical to four
decimals. The hand-rolled fixed-effects estimator is correct. Cluster-robust SEs (the more
defensible choice for 20 clusters) give SE 4.45 → p=0.038, versus the naive 0.026; still
significant, and worth reporting as the honest number.

## Regenerating the benchmark

The fix changes the coupling, so it changes the transplant — not just `real_gw_dist`. The matrix was
regenerated end-to-end at the corrected (ε=0.01, τ=1.0).

**The corrected solver produces materially better transplants.** On the 95 rows shared between the
old and new matrices, `perm_gap` improves in **63 of 95** (paired t p=0.040, Wilcoxon p=0.005), mean
−0.153 → −0.121. It is not free: the transplant also lays ~20% more road (251 m → 303 m mean) and
displaces more (0.097 → 0.120). `perm_gap` is the length-matched comparison, so it remains the right
summary, and it moved the right way.

Headline, corrected matrix:

```
within-recipient beta (perm_gap ~ real_gw_dist) = -9.5817  SE=3.8151  t=-2.5115  dof=79
  p (t-distribution)      = 0.0141
  p (cluster permutation) = 0.0110
jackknife beta range (leave-one-recipient-out) = [-13.0196, -7.7295]
```

versus β=−9.2286, p=0.0261 / 0.0356 before. **The effect strengthened.**

Five of the 100 rows differ in which donor was selected, all backfill around `empty_interior` skips
and three of them on one recipient. The original run hit live Overpass with retries; this one ran
entirely from the 362-response disk cache and recorded no `fetch_failed` at all, so the new row set
is the cleaner of the two.

### The ε ladder

Recomputing distances across a four-step ε ladder while holding `perm_gap` at its committed value
isolates the regularization's effect on the independent variable alone (pre-fix convention, so
"POT ε" is the doubled column):

| nominal ε | POT-equivalent ε | β | p | permutation p |
|---|---|---|---|---|
| 0.02 | 0.04 | −4.88 | 0.178 | 0.213 |
| 0.01 | 0.02 | −9.23 | 0.026 | 0.037 |
| 0.005 | 0.01 | −8.45 | 0.040 | 0.046 |
| 0.0025 | 0.005 | −8.45 | 0.041 | 0.046 |

The estimate **converges** by POT-ε=0.01 (the bottom two rungs agree to three significant figures)
and only dies at POT-ε=0.04, four times more smoothing than the brief allowed. So the "ε ≤ 0.01
mandatory" constraint was well chosen — that is precisely where the estimate stops moving — and the
pre-fix matrix sat one rung inside the window where the effect was still visible but inflated.

## What this changes going forward

1. `ot_gw.py`'s ε and τ now mean what POT and the literature mean. Any ε from before 2026-07-27 in
   the OT notes is **half** what it appears to be.
2. The 2026-07-23 barycenter-consensus study ran on the undoubled gradient, so its ε=0.01 was really
   0.02. Its conclusions were qualitative (single-donor dead, barycenter consensus strong) and the
   ladder shows the direction is stable across the window, so they stand — but any *number* from it
   should be re-measured before being built on.
3. The retrieval index work can treat `real_gw_dist` as trustworthy ground truth.
