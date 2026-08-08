# C9: the access ranking holds on genuinely deep blocks (2026-08-08)

C8 was meant to test whether the access result survives a population selected for DEPTH rather than
compactness, and accidentally tested a shallower pool instead (neither `depth` nor `depth_density`
is depth-selective as configured — both gate permissively and let the proxy pre-filter do the
cutting). This is the honest version: measure `parcel_access_layers` directly and take blocks with
`k0 >= 4`.

40 such blocks exist in the pool; 10 sampled, **every one at k0 = 4**.

## Same champion, same runner-up, same width flip — on all three populations

    === 7 m, AUC 0-30% ===        deep(k0=4)   depth_density   density_compactness
    ACCESS_disp                        0.137           0.063                 0.115   1st ×3
    resistance_lp                      0.157           0.071                 0.137   2nd ×3
    ACCESS_repulsion                   0.207           0.085                 0.156
    greedy_arterial_repulsion          0.225           0.110                 0.159
    cycle_native                       0.225           0.113                  —
    clearance                          0.292           0.164                  —
    topology                           0.339           0.122                 0.227
    flow_paths                         0.345           0.155                  —

    === 2 m ===
    ACCESS_repulsion                   0.023           0.012                 0.013   1st ×3
    greedy_arterial_repulsion          0.028           0.016                 0.021   2nd ×3
    ACCESS_disp                        0.057           0.032                 0.033
    topology                           0.073           0.021                 0.050
    resistance_lp                      0.085           0.045                 0.080

Across three populations spanning k0 = 1–5, **the winner at each width and its runner-up are
identical**, and the width-dependent flip (displacement-cost at street width, repulsion-cost at lane
width) reproduces every time. That is as strong a robustness result as this corpus can give.

## Deep blocks discriminate better

Absolute burdens are higher — the problem is genuinely harder — and the field SPREADS. At 7 m the
range is 0.137–0.345 on deep blocks against 0.063–0.155 on `depth_density`. `topology` in particular
falls from mid-field to 7th of 8, and `clearance` and `flow_paths` sit at the bottom.

So deep blocks are not merely a robustness check; they are the more informative population, which is
what the block-complexity framing predicts. If a single pool has to be chosen for publication, this
is the one, and it argues for the calibrated absolute depth floor that `conf/metric/depth_density.yaml`
already flags as outstanding work.

## Caveats

* 10 blocks per pool, Cape Town, single blocks, Voronoi cells — unchanged throughout.
* The three pools overlap; they are not disjoint samples, so this is a weaker independence claim
  than three separate corpora would give.
* `ACCESS_disp` optimizes benefit-per-displacement and displacement is the x-axis it is scored on;
  `ACCESS_repulsion` and both ACCESS variants optimize the reported metric itself. Both facts are
  legitimate and both must be stated when published.
