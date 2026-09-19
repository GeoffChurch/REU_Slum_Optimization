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

**Corrected 2026-09-19.** The first version of these tables computed proxies by hand from
`block_area_m2` (metres^2) and `blocks.geometry.length` -- but the blocks parquet is in a
GEOGRAPHIC CRS, so that length is in DEGREES. Perimeters came out ~1e-5 of the truth and the
`A/P` "hydraulic radius" read in the millions of metres. `metric._cols()` reprojects to UTM
first, which is why a metric-native run gave 92 for `ca:depth` where the hand version said 75;
that unexplained gap WAS the bug. `scripts/retention_curve.py` now calls `_cols` itself, so the
measurement cannot drift from what `metric.proxy()` scores in production.

Every number in this note is the corrected one. The conclusions were unaffected -- the shift is
10-20% in places and zero in others (`na:depth` 106 and `na:depth_density` 19 are unchanged) --
because a near-uniform scale error on P barely reorders blocks within one city. It would NOT have
been harmless across cities at different latitudes, and it was pure luck that it was not load-
bearing here.

### The worst relative overpayment

`argmax_i [log f(i) - log i]` -- equivalently `argmax f(i)/i` -- is the scale-free version of
"how steep", and a better single-number summary of a proxy than `f(15)`:

    series                        argmax i     f(i)   f(i)/i
    ca:depth         depth_proxy       443   27,759       63     <- matched
    ca:depth         dd_proxy            4   17,892    4,473
    ca:depth_density dd_proxy           26      262       10     <- matched
    ca:depth_density depth_proxy        26    6,071      234
    na:depth         depth_proxy         1       66       66     <- matched
    na:depth_density dd_proxy           17      194       11     <- matched
    na:depth_density depth_proxy        56    1,738       31

Matched proxies cost 10-66x at their worst; mismatched ones 234-4,473x. That ratio separates them
an order of magnitude more cleanly than `f(15)` does.

**Two proxies sharing an argmax is not a coincidence -- it means they fail on the same block for
the same reason.** `ca:depth_density` has BOTH `dd_proxy` and `depth_proxy` peaking at i=26, and
they are `density^1.5 * (A/P)` and `density^0.5 * (A/P)`: they share the `A/P` factor.
`ZAF.9.3.1_1_43705` is 484 buildings on 3.21 ha -- 151/ha against a corpus median of 23 -- but its
A/P is 16.7 m against a median of 35.1, an abnormally thin block. The shared factor buries it
(262 and 6,071) while `density`, which has no `A/P`, ranks it 63.

`na:depth_density` is the mirror image at i=17: `KEN.30.6_1_80` is large (114.79 ha) and only
median-dense (38/ha), so the DENSITY factor is what fails -- `depth_proxy` (density^0.5) places it
66th and `density` alone 2,950th.

## Both shipped proxies are one family, and both sit at its optimum

`depth_proxy` and `dd_proxy` look like two separately-derived formulas. They are the same
expression at two exponents:

    f_alpha = density^alpha * (A/P) = n^alpha * A^(1-alpha) / P

    alpha 0.0   A/P             pure scale, no density
    alpha 0.5   sqrt(nA)/P      depth_proxy    <- shipped for `depth`
    alpha 1.0   n/P             frontage crowding
    alpha 1.5   n^1.5/(P sqrt A) dd_proxy      <- shipped for `depth_density`

Sweeping alpha against retention (worst proxy rank of the fine top-15):

    alpha             0.0    0.25     0.5    0.75     1.0    1.25     1.5    1.75     2.0
    ca:depth        4,486     748      92     635   5,424  13,381  17,892  20,444  21,940
    na:depth          569     230     106     311   2,435   5,271   6,177   6,478   6,629
    ca:depth_density 17,975  3,513     320     128      70      66      67      79      79
    na:depth_density  2,503  1,150     306      79      30      18      19      19      21

* **The shipped exponents are optimal, and this is the first check of that.** 0.5 is exactly the
  minimum for `depth` in both cities; 1.5 is within noise of the 1.25-1.75 minimum for
  `depth_density` (67 against 66). Two independently derived formulas land on the minima of one
  continuous family.
* **The matched-proxy result is a curve, not a coincidence.** The two fine metrics minimise at
  genuinely different alpha, so "use the matched proxy" is one family with two operating points.
* **The wells have very different widths.** `depth` falls from 4,486 to 92 and back to 5,424
  across +/-0.5 in alpha; `depth_density` runs 70 / 66 / 67 / 79 over the same span. `depth`'s
  proxy is delicately tuned and `depth_density`'s is robust -- which is exactly why mismatching
  costs `depth` 17,892 and `depth_density` only 320.

*Hypothesis that failed:* alpha=1 is `n/P`, buildings per metre of block perimeter -- frontage
crowding, a real urban quantity sitting exactly between the two shipped proxies, and untried. It
is not special: 5,424 for `depth` (wrong side of the well) and 70 for `depth_density` against 66
at 1.25. The gap in the family was real; the guess about what filled it was wrong.

**On pruning the dominated series:** do not. This sweep consists ENTIRELY of points that are never
on the frontier, and its shape is the finding -- a sharp well against a flat one, centred in two
different places. Keeping only frontier members would leave two dots and no way to see that one of
them is precarious. The same argument covers `density` (the naive choice a reader proposes) and
`density_compactness` (the previous default): their distance from the frontier is the argument for
what replaced them.

## What would have to be true to ship it

Not yet, on three grounds, and the third is cheap to fix:

1. **The margin is small and the labels are wrong in a known direction.** 0.645 -> 0.656 at
   k=164 is about two blocks, scored against a February 2018 survey with 9 confirmed misses in 41
   hand adjudications. An improvement of that size against a label set biased against whichever
   screen finds NEW settlements is not a result, it is a hint.
2. **One city.** Nairobi has no ground truth, so this is Cape Town or nothing.
3. **SETTLED 2026-09-19: the hand labels do not separate them, and that is the answer.** All
   seven outstanding blocks were adjudicated (`data/adjudication/adhoc_verdicts.csv`); every one
   is `all-dense-informal`. Both top-15s are now fully labelled:

       screen              hand labels    survey
       shipped dd_proxy      15/15          8/15
       dd_proxy / p90        15/15          4/15

   A tie at ceiling. The AUC and prec@1% advantage does not produce a better top-15 on truth, so
   there is nothing to buy: `dd_proxy/p90` costs an extra column read and a p90 per block, it is
   WORSE on retention (f(15) 67 -> 88), and it is not better where a screen is used. **Not
   shipped, and the reason is measured rather than cautious.**

   The survey's apparent preference for the shipped screen -- 8 against 4 -- is backwards. It
   measures which screen re-finds settlements the 2018 file already knows about, not which finds
   real ones. On truth the two tie, and the whole four-block gap is survey recall failure.

*Untested:* T3. Polygon geometry gives shape, orientation and inter-building adjacency, none of
which reduce to area, and none of which have been tried. The claim here is only that polygons buy
nothing **for this feature** -- the area column reproduces them exactly.

*Falsification protocol, and its result:* the test was to adjudicate the seven and see whether
`dd_proxy/p90`'s top-15 came back worse than the shipped screen's on hand labels. It came back
EQUAL (15/15 against 15/15), which is the third possible outcome and the informative one: the AUC
gain is real against the survey and worth nothing against truth. T2 does not collapse into T1 --
footprint size genuinely carries information, and it is free -- but at the operating point this
project uses, that information is redundant with what `n`, `A` and `P` already say.

**The other number this produced belongs in the bake-off, not here.** With both top-15s fully
hand-labelled, the February 2018 survey is wrong about **7 of the shipped screen's top 15** and 11
of `dd_proxy/p90`'s. Earlier adjudication found 9 misses across 35 top-k blocks; this says the
miss rate at the very HEAD of the ranking is around half. Every precision figure the bake-off
publishes is a lower bound, and the bound is looser than "9 misses" suggested.
