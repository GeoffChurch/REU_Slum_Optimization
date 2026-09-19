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

## Falsification: run, and PASSED for the formula, FAILED for the thresholds

Nairobi, refitting nothing -- same formula, same constants:

                  spearman   est/true    gate precision / recall
    Cape Town        0.782      1.29x        90.7%  /  97.4%
    Nairobi          0.897      1.08x        75.0%  / 100.0%   (3 true blocks -- underpowered)

The estimator transfers and is BETTER in the second city. The GATE does not: only 3 Nairobi blocks
reach ">= 50 deep dwellings and >= 100/ha" despite the city holding 137,570 deep parcels, because
its dense blocks are smaller. Physical units bought transferability for the estimator and not for
the cut-points, which remain Cape Town numbers wearing physical clothing.

## Why the constant is 2, and when it is not

Everything rests on `max depth ~= 2 * depth_proxy`. Measured against true peel depth:

    capetown   27,821 blocks   median ratio 1.56   IQR 1.36-1.82
    nairobi     6,634 blocks   median ratio 1.98   IQR 1.74-2.27

`2A/P` is the inradius only for a SQUARE. For an elongated `w x L` block, `2A/P ~= w` while the
inradius is `w/2`, so the constant falls toward 1. Nairobi's blocks are square-ish (1.98); Cape
Town's are elongated (1.56), which is the same shape effect that makes its calibration 1.29x
against Nairobi's 1.08x. Rank correlation survives because it is invariant to a constant factor;
absolute counts do not, and a shape-aware constant would be a real refinement rather than a fit.

## At depth EXACTLY k, and weighted sums

The difference `D(>k-1) - D(>k)` is LINEAR in k -- simpler than the quadratic it came from:

    N(k)  =  P*sqrt(n/A)  -  (2k-1) * P^2/(4A)

An arithmetic progression: first ring `P*sqrt(n/A) - P^2/(4A)`, each subsequent ring losing
`P^2/(2A)`. The corner term checks out -- for a square `P^2/4A = 4`, exactly the four corners the
first ring does not get.

Because `N` is linear, every polynomial weight collapses. With `d = depth_proxy`:

    sum_k  k^m * N(k)   =   n * 2^(m+1) * d^m / ((m+1)(m+2))

    m=0:  n                 all parcels
    m=1:  (2/3) * n * d     TOTAL ACCESS BURDEN (sum of every parcel's depth)
    m=2:  (2/3) * n * d^2
    m=3:  (4/5) * n * d^3

so any polynomial weight gives `n` times a power of `depth_proxy`, and the m=1 case has a
punchline. Divide the total burden by area:

    (2/3) * n * d / A  =  (2/3) * n^1.5 / (P * sqrt(A))  =  (2/3) * dd_proxy

**The shipped `dd_proxy` IS the total access burden per unit area**, up to the constant 2/3. It
was derived as depth x density and turns out to be burden DENSITY -- which is exactly why it is
right for ranking blocks and wrong for counting people. A count wants the unnormalised burden
`n * depth_proxy`, i.e. `dd_proxy * A`.

Corollaries, all consistent with the linear ring decay: `mean depth ~= (2/3) * depth_proxy`, and
`mean = max/3`.
