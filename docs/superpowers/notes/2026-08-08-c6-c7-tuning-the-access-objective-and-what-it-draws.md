# C6/C7: the access objective beats `resistance_lp` at 7 m, and draws a plan rather than an accretion (2026-08-08)

## C6 — tuning

C5 left `resistance_lp` leading the 7 m early regime. Four hypotheses, one variant each.

    === 7 m ===                        AUC 0-30   AUC 0-10   beats resistance_lp
    ACCESS_disp    (cost=displacement)    0.115     0.0784    YES, both windows
    resistance_lp                         0.137     0.0885
    ACCESS_rep_a64 (n_anchors=64)         0.138     0.0979
    ACCESS_rep_r30 (max_roads=30)         0.153     0.0988
    ACCESS_repulsion (baseline)           0.156     0.1017
    ACCESS_asp     (mode=aspirational)    0.207     0.1080

    === 2 m ===
    ACCESS_rep_a64                        0.012     0.0121    YES
    ACCESS_rep_r30                        0.012     0.0123    YES
    ACCESS_repulsion (baseline)           0.013     0.0132    YES
    ACCESS_disp                           0.033     0.0160    YES
    resistance_lp                         0.080     0.0375

**`cost="displacement"` is a 26% AUC improvement at 7 m and takes the lead from `resistance_lp`**,
including over 0-10% where Lens A operates.

The mechanism is direct: the reported curve's x-axis IS displacement, so benefit-per-displacement
optimizes the plotted trade-off itself. **This must be disclosed when published.** It is legitimate
-- displacement is the real cost and every method is free to target it -- but it is optimizing the
exact axis being reported. Note the project's earlier finding that `cost=displacement` degenerates
on sparse fabric (inf-gain gap roads) was measured against the DIRECTNESS objective and does not
transfer: an access gain requires actually fronting parcels, so a gap-hugging road earns nothing.

**The best cost flips with width.** `ACCESS_disp` is ~3x worse at 2 m (0.033 vs 0.012), where
repulsion wins. The tuned recipe is width-dependent: displacement-cost at street width,
repulsion-cost at lane width.

Two arms were duds. `max_roads=30` and `n_anchors=64` are marginal -- the objective converges to
14-15 roads regardless, so the default 15 was never binding. `mode=aspirational` is worst at both
widths, consistent with buildable winning for directness.

## C7 — what it draws

At matched 10% displacement (2 m pricing), two blocks:

    method                     burden        disp        roads
    ACCESS_repulsion         0.00/0.02   6.6%/1.6%       15/18
    greedy_arterial_repulsion 0.00/0.00  9.8%/7.3%       36/61
    resistance_lp            0.34/0.08   8.8%/7.1%     114/139
    topology                 0.24/0.00   9.0%/9.2%       33/83

**It builds 15-18 long chords spanning the block where `resistance_lp` builds 114-139 short stubs
for a worse access result**, and on the second block reaches burden 0.02 at 1.6% displacement -- a
fifth of everyone else's budget.

The mechanism is visible in the render: long straight roads thread BETWEEN buildings (the repulsion
cost steers them into the gaps), giving frontage to many parcels at once, where stubs each pay
displacement for a handful. Parcels go almost uniformly to depth 0; `resistance_lp` keeps green
interior patches exactly where its stub network does not reach.

It looks like a **plan** rather than an accretion -- a coarse subdivision at roughly parcel scale,
which is what real reblocking schemes look like. Checked for gaming and found none: connectivity is
guaranteed by `street_first_ordered` and the chords terminate on the boundary street.

## Caveats

* 10 blocks, Cape Town, single blocks, Voronoi cells, `density_compactness` screen -- unchanged.
* AUC is over the median-of-blocks curve, not a median of per-block AUCs; the two can differ.
* The winning method optimizes the metric being reported, and now also the cost axis it is plotted
  against. Both are legitimate and both need stating.
