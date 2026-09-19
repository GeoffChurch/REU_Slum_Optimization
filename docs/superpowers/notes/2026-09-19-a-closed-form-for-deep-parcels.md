# A closed form for "how many dwellings are more than k houses from a street"

2026-09-19. Asked whether the peel-based vulnerability gate could be written algebraically in
`n`, `A`, `P`, the way `depth_proxy` is geometrically justified rather than fitted.

## The derivation

Parcels tessellate the block, so mean parcel width is `s = sqrt(A/n)`. The inradius of a square of
side `L` is `L/2 = 2A/P`, so the number of peel rings that fit is `(2A/P)/s = 2*sqrt(nA)/P`:

    max peel depth  ~=  2 * depth_proxy        (NOT depth_proxy -- it is half the ring count)

`k` rings consume a fraction `k/(2*depth_proxy)` of each linear dimension, and the surviving
interior shrinks in BOTH dimensions, so the deep-parcel count is

    D(depth > k)  ~=  n * (1 - k*P / (2*sqrt(nA)))^2

    k=2:          D  ~=  n * (1 - P/sqrt(nA))^2

It is zero exactly when `sqrt(nA)/P <= 1` -- the block is too thin or too sparse to have an
interior -- which falls out rather than being imposed. Sanity check on a 10x10 grid of parcels:
`depth_proxy = 2.5`, so `100*(1-0.4)^2 = 36`, and a two-ring peel of a 10x10 grid leaves exactly
36.

## Measured against the true peel

18,309 Cape Town blocks, 328,238 parcels at true peel depth > 2:

    candidate                              spearman   R2 log1p      total   ratio
    n*(1 - 1/dp)^2        [derived, k=2]      0.782      0.485    424,960    1.29
    n*(1 - 1/dp)          [linear]            0.775     -0.560    785,132    2.39
    depth_proxy  sqrt(nA)/P                   0.760      0.007     30,197    0.09
    n  (all parcels)                          0.725     -3.055  1,717,103    5.23

Best rank correlation of anything tried, untuned. The 1.29x overestimate is shape: real blocks are
less compact than squares, so more parcels sit on the boundary. Replacing the 1 with 1.2 calibrates
the total to 0.92x at a negligible rank cost.

**A FIRST DERIVATION WAS WRONG AND THE TEST CAUGHT IT.** `D ~= n - 2P*sqrt(n/A)` -- subtract two
boundary rings of `P/s` parcels each -- scored spearman 0.610, WORSE than plain `n`, and returned
zero for most blocks that do have deep parcels. Two errors: `depth_proxy` is half the ring count,
not the ring count; and the interior shrinks quadratically, not linearly, so dropping the corner
term over-subtracts (it gives 20 for the 10x10 grid where the truth is 36).

## Why it matters: the gate stops needing a peel

The physically-stated gate -- "at least 50 dwellings more than two houses' walk from a street, in
fabric above 100 buildings/ha" -- required tessellating and peeling every block. With the closed
form it does not:

    n*(1 - P/sqrt(nA))^2 >= 50  AND  n/A >= 100/ha
        333 blocks against the true gate's 310
        precision 90.7%, recall 97.4%
        captures 101% of the deep dwellings the true (peel-based) gate finds

    for reference, the shipped floor dd_proxy >= 0.0128
        3,169 blocks -- 10x as many -- for the same coverage

So the criterion is stated in units anyone can argue with, DERIVED rather than calibrated, and
computable from the free `n`, `A`, `P` columns. That last property is what makes it usable at
global scale, where no tessellation exists.

*Caveats.* Cape Town only. The thresholds 2 rings and 100/ha are choices, physically motivated but
not measured optima. The estimator is an estimator: spearman 0.782 means blocks of unusual shape
are misranked, and the 1.0-vs-1.2 constant trades recall (97.4%) against precision (98.9%) --
neither is "right" and the choice belongs to whoever is paying for the false positives.

*Falsification:* peel a second city and refit nothing. If the same constant reproduces the true
gate at comparable precision/recall on Nairobi, the derivation is transferable; if the constant has
to move, it is a Cape Town shape statistic wearing a geometric argument.
