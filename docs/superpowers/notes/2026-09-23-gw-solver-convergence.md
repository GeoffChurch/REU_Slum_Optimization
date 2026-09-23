# The GW inner solve is not converged, and converging it does not help (2026-09-23)

Measured while speeding up `reblock.transplant.gw` for the consensus re-measurement, on 40 real
donor/recipient pairs drawn from the committed `data/benchmarks/gw_pair_matrix.parquet` (clouds of
50-450 parcels), at the operating point eps = 0.01, tau = 1.0, 30 outer x 100 inner iterations.

## The inner Sinkhorn stops far from its fixed point

The unbalanced update is the balanced one damped by tau / (tau + eps) = 0.990, so it contracts by at
most that per iteration. Each outer step also restarts the potentials from zero. The potential
change per iteration, in units of eps (median over pairs):

    iteration   10      25      50      100
    change      7.9e-2  3.8e-2  2.0e-2  7.0e-3

So each outer step keeps a TRUNCATED inner solution. The outer loop still settles (a max coupling
change of 6e-9 by step 29), but on a fixed point of "100 cold-start Sinkhorn iterations", not on a
stationary point of the entropic UGW objective.

## Converging it is 7x the work and not better

A reference that warm-starts each inner solve and runs it to a 1e-11 potential tolerance, with the
outer loop to a 1e-10 coupling tolerance, needs a median 21,150 inner iterations per fit (against
3,000). Against today's schedule:

- GW distance: median relative difference 0.4%, max 9.5%.
- Anchor positions: median shift 5 cm, 90th percentile 35 cm, and one outlier pair at 47 m, on
  recipients spanning a median 140 m.
- **The regularized UGW objective** (exact GW term + eps KL(pi | p q^T) + tau KL on both
  marginals) is LOWER for today's truncated solution in **39 of 40 pairs**, by 0.3-1%. The one
  exception is the 47 m outlier, where the converged solve found a better optimum.

GW is non-convex, and the truncated, cold-started inner solves behave like damped steps that settle
in slightly better local optima. Converging buys nothing measured here and costs 7x.

## What is worth trying, if the solver ever needs to be both faster and converged

**Untested:** translation-invariant Sinkhorn (Sejourne, Vialard & Peyre 2022) removes exactly the
tau/(tau+eps) slow mode, so it should converge in far fewer iterations. But this note shows
convergence is not the goal. Any change to the schedule changes the answers, so it would have to be
judged by the downstream consensus results, not by solver residuals.

## What did ship

The same iterates in the scaling domain, log-stabilized: 6.8x faster, and equal to the log-domain
solver to ~1e-13 (`sinkhorn_unbalanced`, commit cb1b180). Pin BLAS to one thread for it: on a 48-core
machine the default threading is ~20% slower for these matrix sizes, even in a single process.
