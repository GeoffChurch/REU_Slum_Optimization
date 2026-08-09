# C15: the access ranking holds under the new default screen (2026-08-08)

Every access result C5–C9 was measured on `density_compactness`-selected blocks, which C13 later
showed are only 24.5% really informal. The default is now `depth_density_proxy` at its calibrated
floor, selecting a different and measurably better population — so the whole line needed
re-validating on it. Same construction as C9: build the pool with the new screen, measure
`parcel_access_layers` directly, keep `k0 >= 4`.

    pool: 2,977 blocks materialized, 720 usable recipients (against 578 under the old screen)
    40 blocks at k0 >= 4; 10 sampled; starting k0 {4: 9, 5: 1}

    === 7 m, AUC 0-30% ===                    === 2 m ===
    ACCESS_disp                 0.165         ACCESS_repulsion            0.031
    resistance_lp               0.179         greedy_arterial_repulsion   0.051
    cycle_native                0.253         ACCESS_disp                 0.073
    ACCESS_repulsion            0.253         topology                    0.079
    clearance                   0.299         resistance_lp               0.111
    greedy_arterial_repulsion   0.311         cycle_native                0.177
    topology                    0.355         clearance                   0.181
    flow_paths                  0.360         flow_paths                  0.240

**Identical champion and runner-up at both widths, for the fourth population in a row:**

    population                                    7 m (1st / 2nd)        2 m (1st / 2nd)
    density_compactness           (C5/C6)   ACCESS_disp / res_lp   ACCESS_rep / gar_rep
    depth_density                 (C8)      ACCESS_disp / res_lp   ACCESS_rep / gar_rep
    deep k0>=4, old screen        (C9)      ACCESS_disp / res_lp   ACCESS_rep / gar_rep
    deep k0>=4, NEW screen        (C15)     ACCESS_disp / res_lp   ACCESS_rep / gar_rep

The width-dependent flip — displacement-cost at street width, repulsion-cost at lane width —
reproduces every time as well.

Absolute burdens are higher than C9's on the same nominal criterion (0.165 vs 0.137 at 7 m), which
is the expected direction: the new screen selects denser, more genuinely informal fabric, so the
access problem on those blocks is harder.

## What this closes

The access metric and the two methods wired up in `c2e8cac` were validated on a population we now
know was 75% non-informal. They survive the correction. Nothing in C2–C9 needs retracting on
population grounds.

## Caveats

* 10 blocks per population, Cape Town, single blocks — unchanged throughout.
* The four populations overlap; they are not disjoint samples, so this is weaker than four
  independent corpora would be.
* AUC is over the median-of-blocks curve.
