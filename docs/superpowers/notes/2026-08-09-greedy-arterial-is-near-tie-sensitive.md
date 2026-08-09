# `greedy_arterial` is near-tie sensitive: 1e-10 moves a headline number by 13 points (2026-08-09)

Found by accident, chasing a performance win. An incremental reformulation of the displacement cost
agreed with the original to **1.14e-10** and produced a different network on **29% of runs**, moving
burden reduction by up to 11 points. That is not a bug in either formulation — it is a property of
the method, and it had never been looked at.

## The measurement

Perturb each candidate's gain by `(1 ± 1e-10)`, the size of that discrepancy, under six seeds.
Nothing else changes: same candidates, same objective, same code path (`scratchpad/perf/tie_sensitivity.py`,
monkeypatching the shipped `eval_candidate`). Any spread is attributable to near-ties in the argmax.

    metric        spread across seeds        relative
    burden_red    median 0.0000  max 0.1356   median 0.0%   max 15.5%
    permeability  median 0.0000  max 0.1297   median 0.0%   max 15.0%
    road_m        median 15.82   max 58.73    median 11.2%  max 23.7%

    blocks where every perturbation gave the same burden reduction: 5 of 8
    worst block, ZAF.9.3.1_1_19362: burden reduction 0.7627 .. 0.8983

**Most blocks are perfectly stable; a minority are wildly unstable.** And road length is unstable on
the MAJORITY — a median 11.2% spread — even where the objectives are not. So a method can land on
very different networks that happen to score alike.

## A first probe measured the wrong thing, and its null is still worth keeping

The obvious probe is to shuffle the ORDER candidates are scored in, since `max` returns the first
maximal element. Result: **zero spread, 8/8 blocks, 6 orderings** — verified live rather than
assumed (the shuffle reorders in place; `_candidate_chords` was intercepted once per greedy step).

That null is real but answers a different question. Shuffling decides only EXACT ties, and there are
none — **`greedy_arterial` is order-invariant**, which is worth knowing on its own. Near-ties are
resolved by value, deterministically, regardless of order; they flip only when the values move.
Two different sensitivities, and the first probe measured the one that is not the phenomenon.

## What this invalidates

**A single block's `greedy_arterial` number is not reproducible.** Not across shapely builds, GEOS
versions, platforms, or any refactor that touches the cost arithmetic in the last bits. The current
published values are one arbitrary draw from a distribution nobody had sampled.

**`displacement_fast`'s apparent 3-for-3 win is inside this noise.** It differed from the exact cost
on 3 runs and won all 3 (burden_red +0.011/+0.109/+0.016). Those magnitudes are precisely the
perturbation spread measured here, so the comparison carries no information about which formulation
is better. Establishing that would need enough blocks for the medians to separate — and since the
per-block noise reaches 15%, that is a lot of blocks.

**Medians over blocks are the only safe unit.** C19, C20 and the two lens tables already aggregate
that way, so those results stand. What does NOT stand is any per-block or per-region figure for an
arterial variant — which includes the `method_comparison` example, a single pinned block.

## What to do about it

Options, none taken yet:

* **Report it.** An arterial row in a single-block comparison carries a ±15% error bar that no other
  method has. Saying so is cheap and honest.
* **Average it away.** Run k perturbed seeds and report the median. Exact, trivially parallel, k×
  the cost — and it turns an arbitrary draw into an estimate.
* **Remove the near-ties.** They presumably come from candidates that are geometrically near-identical
  (chords between adjacent anchors). Deduplicating candidates within a tolerance before scoring would
  collapse the near-ties rather than resolving them arbitrarily. Speculative; unmeasured.

## Caveats

* 8 blocks, ≤110 parcels, `objective=access`, `cost=displacement`, `max_roads=8`. Whether other
  objectives (directness, efficiency) share the sensitivity is untested.
* `EPS = 1e-10` was chosen to match the observed discrepancy. The sensitivity presumably grows with
  eps; the shape of that curve is unmeasured.
