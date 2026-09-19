# The best screen depends on what data you have, and footprint size is free

2026-09-19. Prompted by asking whether the backlogged "Open Buildings footprint size" feature still
wins now that the pool, the count source and the labels have all changed.

## The frontier, indexed by input

A screen is not one question. It is one question per INFORMATION SET, and the right way to hold the
answer is a tier list: the best metric given what you know about each block.

Download size and storage are the least interesting reading of that. The tiers are of independent
interest as a statement about **where the discriminative information lives** -- block geometry
alone, versus geometry plus a count, versus counts plus the size distribution of the structures,
versus their full shapes. Each row is a claim about what can be known from a signal, and it stays
true if the data becomes free. The megabytes are noted only because they happen to decide what is
already on disk.

| tier | inputs | extra download | best measured | AUC | prec@1% | prec@5% |
|---|---|---|---|---|---|---|
| T0 | kblock columns: Ecopia `n`, `A`, `P` | none | `n^1.5/(P*sqrt(A))` | 0.921 | 0.812* | 0.416* |
| T1 | + Open Buildings POINTS (`n` recounted) | 49 MB, already paid | same formula | 0.919 | 0.645 | 0.403 |
| T2 | + `area_in_meters` (same file) | **none** | `n^1.5/(P*sqrt(A)) / p90` | **0.941** | **0.656** | **0.427** |
| T3 | + polygon geometry | 174 MB | nothing beats T2 | 0.941 | 0.656 | 0.427 |

\* **T0's precision columns are not comparable with the rest and must not be read as the maximum.**
They are scored over a different population, and putting them in the same column invites exactly
that misreading -- it did. Within the comparable rows, prec@1% is maximised by `dd_proxy/sqrt(p90)`
at 0.672, then `dd_proxy/p90` at 0.656, then the shipped `dd_proxy` at 0.645.

T0 and T1 are NOT comparable and the table should not be read as a regression:
they are scored over different populations. `MIN_COUNT = 30` is a threshold on the count itself, so
Ecopia admits 16,451 Cape Town blocks and Open Buildings 18,309, and the 1,858 added are mostly
blocks Ecopia could not see. Rows T1-T3 share a population and ARE comparable.

## T2 is free, which is the finding

The backlog carried this as "a single Google Open Buildings feature, 90th-percentile footprint
area, scores AUC 0.943 and beats every metric here -- at the cost of a polygon download this screen
does not need."

Both halves are wrong now.

**The download is not needed.** `area_in_meters` is a column on `buildings_*_full.parquet`, the
49 MB POINTS file the pipeline already reads to tessellate parcels; the 174 MB polygon file is a
separate artifact. Computing p90 from the column and from `polygon.geometry.area` over all 18,309
blocks agrees to a median relative difference of **0.013%** with a rank correlation of **1.00000**,
and both give AUC 0.941 / prec@1% 0.656 to three decimals. The expensive download buys nothing.

**"Beats every metric here" is an AUC statement, and a screen is not an AUC.** Measured on the
current pool, footprint size ALONE:

    feature                            AUC   prec@1%
    OB p90 footprint alone (negated)  0.937     0.169
    depth_density proxy (shipped)     0.919     0.645

Still the best AUC of any single feature, and it is a **bad screen** -- 16.9% precision in the top
1% against the shipped screen's 64.5%. This is the same trap the bake-off page already documents
for `density` vs `depth_density`, which tie on AUC while one is far better at the head. Ranking a
corpus and ranking its head are different questions, and only the second is what a screen does.

The mechanism is that footprint size carries **no scale term**:

    top 1% by...            median n   median ha   n per ha   med p90 m2   survey-informal
    SMALLEST OB p90              101        0.91        138         37.3            16.9%
    depth_density proxy          322        2.18        151         50.2            64.5%
    (whole pool)                  56        1.64         36        162.7             4.2%

Comparable density, a third the buildings, half the area. It finds small blocks of small
structures -- outbuildings, garages, tight formal rows -- which is exactly `density_compactness`'s
failure, and `(1/p90) * (A/P)` without a density term scores 0.044 at the top, confirming which
term does the work.

As a DIVISOR it is a different proposition, dominating the shipped screen on every axis measured:

    depth_density proxy (shipped)     0.919   0.645   0.403
    dd_proxy / p90                    0.941   0.656   0.427
    dd_proxy / sqrt(p90)              0.935   0.672   0.423

## Three objectives, three different winners

AUC and precision are not the only things a metric here is asked to do, and the third objective has
a different answer from both. `proxy_keep_n` pre-filters by a CHEAP proxy before the expensive
Voronoi peel, so its only requirement is RETENTION: no block the fine metric would rank in its
top-k may fall outside the kept prefix. Scored as the worst proxy rank among the fine top-15,
lower being better:

    proxy                          ca:depth  ca:depth_density  na:depth  na:depth_density
    depth_proxy  sqrt(nA)/P              75               338       106               310
    dd_proxy  n^1.5/(P sqrt A)       18,003                57     6,178                19
    dd_proxy / sqrt(p90)             17,279                67     6,047                19
    dd_proxy / p90                   16,708                75     5,955                36
    A/P (hydraulic radius)            4,040            18,165       569             2,509
    density  n/A                     26,173               997     6,893             1,090
    dens_compact  n/P^2              27,369            23,019     6,911             5,330

**Each fine metric is best retained by its own MATCHED proxy**, and by a wide margin: `depth` wants
`sqrt(nA)/P` (75) where `dd_proxy` needs 18,003, and `depth_density` wants `dd_proxy` (57) where
`depth_proxy` needs 338. `metric.proxy()` already returns the matched one, so the shipped pipeline
is correct -- but nothing had measured WHY, and collapsing the two onto one shared proxy, which
looks like a tidy simplification, would silently cost ~250x in retention.

Note also that the footprint term HURTS here while helping above: 57 -> 75 on `ca:depth_density`.
More information improved the head of the ranking and degraded the pre-filter's recall, because
the two are scored on different things.

    objective     what it asks                    best
    AUC           rank the whole corpus           dd_proxy / p90        (0.941)
    prec@1%       rank the head                   dd_proxy / sqrt(p90)  (0.672)
    retention     do not lose the fine top-15     the matched proxy, per metric

*Caveat on the absolute ranks:* these seven were computed by hand from `block_area_m2` and
`geometry.length`, where `metric.proxy()` goes through `_cols()`, which may reproject -- a
metric-native run gave 92 for `ca:depth` against 75 here. The comparison ACROSS proxies is
unaffected (all computed identically), but the calibrated figure behind `proxy_keep_n: 1000` is
the metric-native 106, not 75.

## What would have to be true to ship it

Not yet, on three grounds, and the third is cheap to fix:

1. **The margin is small and the labels are wrong in a known direction.** 0.645 -> 0.656 at
   k=164 is about two blocks, scored against a February 2018 survey with 9 confirmed misses in 41
   hand adjudications. An improvement of that size against a label set biased against whichever
   screen finds NEW settlements is not a result, it is a hint.
2. **One city.** Nairobi has no ground truth, so this is Cape Town or nothing.
3. **The hand labels do not separate the two yet.** Shipped is 11/11 informal on the adjudicated
   part of its top-15; `dd_proxy/p90` is 8/8. Both perfect, neither distinguishing, because the
   blocks where they DISAGREE are unadjudicated: `_30677 _30665 _5833 _23714 _41782 _30700
   _21934`. Seven judgements settle it, and `_30677` is already in `control_sample.csv` with
   imagery rendered.

*Untested:* T3. Polygon geometry gives shape, orientation and inter-building adjacency, none of
which reduce to area, and none of which have been tried. The claim here is only that polygons buy
nothing **for this feature** -- the area column reproduces them exactly.

*Falsification:* adjudicate the seven. If `dd_proxy/p90`'s top-15 comes back materially worse than
the shipped screen's 11/11 on hand labels, the AUC gain is a survey artifact and T2 collapses back
into T1.
