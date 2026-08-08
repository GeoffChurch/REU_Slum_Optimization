# C5: the access objective was never wired up, and it wins (2026-08-08)

Two corrections to C4, one of them to my own plotting.

## 1. The non-monotonicity was an AGGREGATION artifact, not a metric one

The owner asked why the curves were non-monotone when the metric cannot be. Checked directly on
C4's data: burden increases on **0 of 163** (block, method) series, and so does `k0`. The prefixes
are cumulative and the metric behaves exactly as claimed.

The jaggedness came from my plot. Binning by displacement and taking a median across whatever blocks
landed in each bin means consecutive bins are medians over *different subsets* — and because curves
END at different displacements, the late bins were medians over a shrinking, biased subset.

Fixed by interpolate-then-aggregate: put every block's curve on one displacement grid first (holding
the final value beyond a method's own maximum, since a method that has built everything cannot
improve further), then take the median. Every aggregated series is now monotone, verified.

**This also answers the softer-metric question in the owner's favour.** The instability that
motivated asking for a soft depth was mostly my aggregation. No softening is needed to get monotone,
readable curves — and softening would have cost something real: `Σ(escape time)²` is free from
`egress_power`'s per-parcel `tau` but reintroduces `sigma` and its calibration problem.

## 2. `GreedyArterialReblocker(objective="access")` existed, was never configured, and is the champion

`_score`'s access branch maximizes `1 - access_burden(depths)/base_burden` — greedy optimization of
sum-of-squared-depths, the exact statistic this line converged on. **All four configured arterial
variants use `objective=directness`.** The access-native optimizer had never been run, the same
class of omission as `peel` not being in `all_methods`.

Two were added, mirroring the best directness configs. Median burden on the corrected grid:

    roads at 2 m                          2%     5%    10%    15%    30%   wins
    greedy_arterial_ACCESS_repulsion    0.05   0.01   0.00   0.00   0.00    90%
    greedy_arterial_repulsion           0.41   0.05   0.00   0.00   0.00     0%
    greedy_arterial_buildable           0.55   0.21   0.02   0.02   0.02     0%
    greedy_arterial_ACCESS_length       0.59   0.21   0.06   0.06   0.06     0%
    topology                            0.90   0.39   0.00   0.00   0.00     0%
    resistance_lp                       0.49   0.21   0.21   0.21   0.21     3%
    peel                                1.39   1.23   0.97   0.76   0.40     0%

    roads at 7 m                          2%     5%    10%    15%    30%   wins
    resistance_lp                       1.18   0.94   0.40   0.22   0.21    65%
    greedy_arterial_ACCESS_repulsion    1.31   1.04   0.58   0.35   0.08    29%
    greedy_arterial_repulsion           1.20   0.93   0.59   0.45   0.15     3%
    euclidean_grid                      1.50   1.46   1.36   1.18   0.54     0%
    greedy_arterial_displacement        1.37   1.31   1.31   1.31   1.31     0%

At 2 m the access objective is at burden **0.05 at 2% displacement**, where the next best is 0.41 —
an eight-fold gap — and it is at effectively universal access by 5–10%. It wins 90% of the grid.

At 7 m `resistance_lp` still leads early (65% of the grid) and the access variant overtakes it late
(0.08 vs 0.21 at 30%).

**Third time an omission from the roster changed the answer.** C2 and C3 were wrong because `peel`
and 11 others were missing; C4 was incomplete because the access objective was never configured.

## 3. Do the series stay in their lanes?

Pair-crossings along the corrected grid, 153 pairs:

    roads at 7 m    151 crossings    roads at 2 m     86 crossings

But they are not spread evenly. The extremes are stable and the middle churns:

    7 m   resistance_lp 7, euclidean_grid 2, peel 4, greedy_arterial_aspirational 5
          topology 35, flow_paths 31, demand_greedy 27, flow_paths_noreinforce 24

So the champion, the floor, and the clear failures hold their lanes; the reordering is concentrated
among methods whose curves genuinely overlap. That is the honest result rather than a defect — those
methods really are within noise of each other, and a softer metric would impose an order on them
that the data does not support. 2 m is markedly more stable than 7 m throughout.

## Other findings that survive

* `peel` is built for full access and is near the BOTTOM of the curve at every budget in this range.
  It reaches `k0 = 0` only around 90% displacement, off the right of this grid. Built for the
  objective is not efficient at it.
* `greedy_arterial_displacement` is flat at 1.31 everywhere — it builds almost nothing (0.008
  displacement) and buys nothing.
* `euclidean_grid` remains near-last on access while being mid-pack on permeability.

## Caveats

* 10 blocks, Cape Town, single blocks only — `load_pools` uses `IdentityRegionBuilder` at singleton
  granularity, so no multiblock regions anywhere in C1–C5.
* The screen is `density_compactness` (n/P² at the calibrated absolute floor). `conf/metric/` also
  offers `depth` and `depth_density`, and since this whole line is about access depth, screening on
  those is an obvious untested variant — the current pool is selected for COMPACTNESS, which is what
  made C1's coverage saturate.
* Parcels are Voronoi cells of building points, not a cadastre, and not footprint-seeded.
* `osm_footpaths` n = 5, `demand_greedy` n = 8 (HTTPError); both excluded from the curve tables above
  where fewer than 8 blocks contributed.
* This now measures the metric that `objective="access"` optimizes directly, so that method has a
  structural advantage. It is not circular — the comparison is at matched displacement and every
  method is free to target whatever it likes — but it should be stated when published.
