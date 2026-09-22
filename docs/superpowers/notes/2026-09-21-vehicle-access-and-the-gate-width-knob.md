# Vehicle access: saturation does not reward restraint, and line-sampled gates are a knob

2026-09-21. Prompted by asking why `clearance_looped` (Looped Tree) looks like the best plan and
scores worst, while `resistance_lp` scores best and is a mesh of 6.2 m stubs. Four things were
measured; one is usable, three are negative, and the negatives are the reason this note exists.

## The question

Permeability is a smooth flow over a mesh whose footpath conductance is non-zero everywhere, so it
never saturates: more road always buys more. Displacement charges homes, not metres, so a method
that threads gaps gets more road for the same price. Hence the hypothesis: make the benefit
**binary and saturating** — a building is served once a corridor of vehicle width reaches it, and
further road buys nothing — and restraint should start to pay.

## Result 1 (POSITIVE): the raster direct-access measure works

Free space rasterized at 0.5 m, Euclidean distance transform, a cell admits a `W`-wide corridor iff
its clearance is `>= W/2`, components touching the block edge are grounded (a kblock face is
street-bounded, so its boundary IS the street). Same primitive `gap_widths.py` used.

Non-degenerate and width-sensitive. On `ZAF.9.3.1_1_40972` (263 buildings) with no intervention:

    W = 3.0 m   0.806 served
    W = 3.5 m   0.741
    W = 4.0 m   0.513

It discriminates hardest right at the ambulance width, which is the property `eps` lacked in the
continuum spike: `W` is a policy input you report a sweep over, not a constant you tune.

## Result 2 (NEGATIVE): saturation does not flip the ranking

Scored at Lens A (matched displacement) on the same block, `W = 3.5`:

    method                road_m   served
    clearance_looped         110    0.814
    osm_footpaths            215    0.829
    euclidean_grid           218    0.875
    cycle_native             243    0.848
    greedy_arterial          339    0.913
    (no roads)                 0    0.741

Vehicle access tracks road length near-monotonically and **Looped Tree stays last**. Saturation is
never reached: the baseline is 0.741 and the ceiling 0.913, so more road keeps buying access the
whole way.

On reflection that is correct rather than a defect of the measure. At matched displacement
`greedy_arterial` gets 339 m for the same 10% that buys Looped Tree 110 m, and if displacement is
the whole cost then three times the road for the same price genuinely is better. **Saturation
cannot make restraint pay under a cost model that does not charge for what restraint saves.** The
lever is a cost on road length, which is where this started.

## Result 3 (NEGATIVE): line-sampled gate widths are a knob, not a detail

"Gate depth" is a clean generalisation of the existing BFS peel: between adjacent parcels the gate
is the channel width, an edge costs 0 if the gate admits `W` and 1 otherwise, and depth is a 0-1
BFS from street-fronting parcels. `W -> 0` gives depth 0 everywhere; `W -> inf` recovers the
current ring count exactly. `sum_i depth_i` is `access_burden`.

The idea is sound. **Sampling the gate along the centre-to-centre segment is not**, and the choice
of reduction reorders everything:

    variant                median gate   do-nothing served   verdict
    min along segment          1.00 m          0.190         Looped Tree wins, +61.0
    max (ridge) along segment  6.32 m          1.000         do nothing wins, all methods negative

Calibration settles which is wrong, and the answer is BOTH. `gap_widths.py` measured a 7.65 m
median channel for this block's density band and 2.5-3.0 m at 8k+ bldg/km². `min` is pinned at
`2 x RES`, the resolution floor. `max` reads **8.00 m on the spine block** — 3x too wide on the
dense fabric where the question actually lives, because the segment between two adjacent parcels'
building centres can cross open ground and `max` finds that instead of the passage.

This is the continuum spike's failure mode again (`eps` reordering methods, 119 rank flips): a
sampling choice that looks like an implementation detail turning out to be the thing that decides
the answer. The pairwise `dist_ij - r_i - r_j` fails the same way for the same reason — it is the
clearance between two discs, not the width of the passage, and a third disc pinching the channel
never appears in it.

**What it needs is a genuine bottleneck (widest-path) computation on the distance transform**: the
gate between two parcels is the maximin clearance over paths between them, not a statistic of a
line. That is real work, not a parameter.

## Result 4 (NEGATIVE, and a retraction): demolition/displacement divergence does not generalise

At equal 10% displacement on the 263-parcel block, `euclidean_grid` demolishes 17.5% of buildings
against `clearance_looped`'s 10.6% -- 65% more homes destroyed for the same score, because
fractional disc overlap charges grazing many buildings the same as flattening fewer. That looked
like a strong critique of the displacement metric.

**It does not hold on the spine block.** There every method lands at ~10% demolished, and
`greedy_arterial` demolishes the FEWEST (6.7%) while laying the MOST road (9,713 m). Claimed from
one block and withdrawn on the second.

## What to take from this

* The raster measure is usable and calibrated; reach for it rather than rebuilding one.
* Do not re-propose saturation as the fix for the `resistance_lp` pathology. It was measured and
  it does not flip the ranking, for a reason that survives the measurement: access genuinely
  increases with road, so only a cost on road can reward restraint.
* Do not sample a channel width along a line. Both reductions are wrong and they disagree about
  the answer.
* `W` is the parameter this family gets right -- physical, sweepable, and reportable -- where
  `g_walk` is not. See `2026-08-08-the-road-walk-ratio-has-no-physical-support.md`.
