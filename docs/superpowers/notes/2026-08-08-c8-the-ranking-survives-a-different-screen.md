# C8: the access ranking survives a different screen — but not the test I meant to run (2026-08-08)

Every result in C1–C7 ran on `density_compactness`. Since that screen selects for COMPACTNESS, and
compactness is what made C1's coverage saturate, the obvious worry was that the pool is the wrong
population for a metric about access DEPTH. So the comparison was re-run on `depth_density`.

## The ranking holds

    === 7 m, AUC 0-30% ===         depth_density   (density_compactness)
    ACCESS_disp                           0.063             0.115    1st in both
    resistance_lp                         0.071             0.137    2nd in both
    ACCESS_repulsion                      0.085             0.156
    greedy_arterial_repulsion             0.110             0.159
    topology                              0.122             0.227

    === 2 m ===
    ACCESS_repulsion                      0.012             0.013    1st in both
    greedy_arterial_repulsion             0.016             0.021    2nd in both
    ACCESS_disp                           0.032             0.033
    resistance_lp                         0.045             0.080

Both the per-width champion AND the width-dependent flip reproduce on a different population. That
is the robustness a shipped metric needs, and it is the first cross-population check this line has
had.

## But this was NOT a deep-block test — correcting my own framing

I described this as testing "deeper blocks". **`depth_density` selected SHALLOWER ones.** Starting
complexity on this pool is `{1:1, 2:5, 3:4}` against `{3:3, 4:6, 5:1}` on the compactness pool.

The reason is in `conf/metric/depth_density.yaml`'s own comment: its gate is `percentile 100`, an
explicit pass-through, so the real selector is the shared top-`proxy_keep_pct`% proxy pre-filter.
The pool is 1,915 candidates against `density_compactness`'s 581 — **larger and more permissive,
not depth-selective.** `conf/metric/depth.yaml` has the same property: an absolute gate at 2.0
rings, which nearly every block in this corpus clears.

So neither configured "depth" metric actually selects for depth. The comment in `depth_density.yaml`
anticipates this — "a calibrated absolute depth*density floor is legitimate future work" — and this
is that work coming due. It is the same defect `DENSITY_COMPACTNESS_FLOOR` was calibrated to fix for
the compactness metric, still outstanding for the depth ones.

**As a robustness check across populations this stands. As evidence about deep blocks it is worse
than useless, since the pool is shallower than the one it was meant to improve on.**

C9 runs the real version: measure `parcel_access_layers` directly and filter to k0 >= 4, rather than
trusting a screen to do it.

## Caveats

* 10 blocks per pool, Cape Town, single blocks, Voronoi cells.
* AUC is over the median-of-blocks curve.
* The two pools overlap — `depth_density` is a superset-ish population, not a disjoint sample — so
  this is a weaker independence claim than two disjoint corpora would give.
