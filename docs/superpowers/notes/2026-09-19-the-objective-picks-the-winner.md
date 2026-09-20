# There is no best screen. There are five objectives and five winners.

2026-09-19. Accumulated while answering "which screen is best?" enough times to notice the
question is underspecified.

Every ranking below is over the same Cape Town corpus (18,309 Open Buildings-eligible blocks) with
the same candidate metrics. Only the QUESTION changes, and the winner changes with it.

| objective | what it asks | winner |
|---|---|---|
| AUC | separate informal from formal over the whole corpus | `density n/A` (0.921), `dd/p90` (0.941 with footprints) |
| prec@1% | be right about the head of the ranking | `dd_proxy / sqrt(p90)` (0.672) |
| retention | never lose the fine metric's top-k to the pre-filter | the metric's OWN matched proxy |
| dwellings captured | concentrate PEOPLE into a small flagged set | `depth proxy sqrt(nA)/P` (39.3% at top 1%) |
| deep + informal parcels | reach the most access-deprived | `depth_density proxy` (53.2% at top 1%) |

`density` has the best AUC and is close to the worst on every other axis. That is not a paradox:
AUC is BLOCK-COUNT weighted, so it treats a 3,000-shack block and a 30-shack block as equal
evidence. Any question about people is weighted by dwellings, and the two weightings disagree
violently.

## Capturing people

Share of the 117,336 surveyed informal DWELLINGS falling inside the flagged set:

    metric                              top 0.5%  top 1%   top 2%   top 5%  top 10%
    depth proxy  sqrt(nA)/P               27.9%    39.3%    51.6%    65.2%   75.2%
    count  n                              27.3%    35.3%    47.2%    64.4%   74.2%
    depth_density proxy (shipped)         24.2%    34.2%    46.5%    63.0%   73.9%
    density  n/A                           3.7%    10.7%    20.1%    38.3%   53.8%
    density_compactness  n/P^2             5.2%     9.4%    14.7%    24.2%   36.9%

This is CAPTURE, not estimation. `sqrt(nA)/P` at top 1% misses 60.7% of dwellings. To ESTIMATE a
population, use the ranking to STRATIFY and weight by inclusion probability -- the design already
in `scripts/gen_control_sample.py` -- rather than treating the flagged set as the answer.

## Are the flagged blocks actually deep?

True Voronoi peel depth (not the proxy) for each metric's top 1%:

    metric (top 1%)                  median TRUE depth   p90   dwellings
    depth proxy  sqrt(nA)/P                        7.0  11.0      39.3%
    count  n                                       7.0  11.0      35.3%
    depth_density proxy (shipped)                  6.0  10.0      34.2%
    density  n/A                                   4.0   8.0      10.7%
    (corpus median)                                3.0   4.0

Yes -- more than double the corpus median. But the depth targeting does NOT come from the depth
construction: raw `count n` selects blocks of identical depth (7.0 / 11.0). Since
`sqrt(nA)/P ~ sqrt(n) * shape`, deep-block selection is a byproduct of preferring large populous
blocks. The `sqrt(nA)/P` form buys ~4 points of dwelling capture over `n` and zero median depth.

`Depth.fine` is the block's MAXIMUM parcel depth, so 7.0 means "the deepest dwelling in a typical
flagged block is 7 rings from a street", not that the typical dwelling is.

## The most vulnerable: parcels at depth > 2

    ALL blocks (328,238 such parcels)      top 0.5%   top 1%   top 2%   top 5%
    count  n                                  25.5%    33.1%    42.6%    57.4%
    depth proxy  sqrt(nA)/P                   22.5%    30.6%    40.0%    55.1%
    depth_density proxy                       11.4%    17.8%    23.9%    33.2%
    density  n/A                               1.9%     4.7%     8.4%    17.5%

    survey-INFORMAL only (62,477 parcels)
    depth_density proxy (shipped)             39.6%    53.2%    68.4%    85.7%
    depth proxy  sqrt(nA)/P                   38.0%    52.6%    69.7%    84.1%
    count  n                                  32.1%    43.3%    58.1%    78.9%
    density  n/A                               5.7%    16.5%    30.0%    49.8%

**The two conditionings disagree, and neither is clean.** Unconditionally `count n` wins because
it also collects large FORMAL blocks with deep interiors -- 19% of all parcels sit at depth > 2,
and plenty are in formal fabric. Conditioned on survey-informal the shipped proxy wins, but that
conditioning EXCLUDES settlements the survey missed -- including the 55-block, 14,108-building
Racing Park cluster, which is survey-formal throughout -- so it penalises `count n` exactly for
finding what the ground truth does not know about. The truth is bracketed, not resolved. 48 hand
labels cannot break the tie.

## CORRECTION 2026-09-19: the capture tables above are block-matched, and that is unfair

Every capture number above selects the **top 1% of BLOCKS**. The families select blocks of wildly
different sizes -- the unnormalised members prefer 400-hectare blocks -- so a fixed block count
hands them far more ground and therefore more of everything. The like-for-like comparison is a
matched PARCEL budget, the same "at equal pool size" discipline the bake-off already applies to
the floors.

Redone at 100,000 parcels:

    metric                     blocks   parcels    dwell    deep  deep+inf
    unnorm m=1  n*d                48   100,353    23.1%   19.6%     26.3%
    unnorm m=2  n*d^2              56   100,486    22.5%   20.4%     27.0%
    NORM  m=1  dd_proxy           212   100,151    36.0%   18.6%     55.4%
    NORM  m=2  (n/P)^2            150   100,150    33.4%   20.0%     52.8%
    NORM  m=0  density            642   100,026    29.3%   12.2%     40.1%

**The ordering reverses.** Normalised beats unnormalised on dwellings 36.0% against 23.1%, and on
deep-and-informal 55.4% against 26.3%. Only the label-free deep-parcel column is a tie. The
unnormalised family's big blocks are largely RURAL -- lots of area, few slum dwellings per parcel
-- which the block-matched view concealed.

Two claims made here and elsewhere are withdrawn:

* "for counting people at global scale use `n`, or `sqrt(nA)/P` where blocks are reliable" -- at a
  matched parcel budget, which is what a survey budget actually looks like, `dd_proxy` captures
  half again as many dwellings. Use the shipped metric.
* "`(n/P)^2` beats `dd_proxy` on deep-and-informal, 57.3% to 53.2%" -- block-matched. At matched
  parcels it is 52.8% against 55.4%, the other way round.

The shipped `depth_density` proxy is the best of all six on both people-weighted measures once the
comparison is like-for-like.

## For global scale, without footprints

*Recommendation, with its caveats attached rather than filed elsewhere.*

**Superseded by the correction above -- read it first.** The recommendations below were derived
block-matched and the ordering does not survive a matched parcel budget. Kept because the
DATA-AVAILABILITY argument (what needs a perimeter, what needs a street network) is unaffected.

* **`count n` if you cannot build reliable blocks.** It needs no perimeter, therefore no
  street-bounded faces, therefore no dependence on OSM road coverage -- the weakest link outside
  well-mapped cities. It works on any tessellation or a raster grid, and costs ~4 points of
  dwelling capture against the best metric.
* **`sqrt(nA)/P` or the shipped `n^1.5/(P sqrt A)` where the street network is trustworthy.** The
  first is best for dwellings, the second for deep-and-informal parcels; they are within a point
  of each other on both and the choice hardly matters.
* **Never `density n/A`** for anything population-weighted, despite its winning AUC.
* **A RANK-BASED gate, per city or region -- never the absolute floor.** Measured, the floor
  selects 2.0x a different share of Nairobi than of Cape Town, two cities on one continent.
  `PercentileGate` exists for this and this is the case that justifies it.
* **For population, sum a gridded product (WorldPop, GHS-POP) inside the flagged blocks** rather
  than multiplying `n` by a persons-per-structure constant, which varies by country far more than
  the count does.

*Falsification:* every number here is ONE city, scored against a survey now shown to be wrong
about 7 of the shipped screen's top 15 blocks and blind to at least one 14,000-building
settlement. The ORDERINGS are more trustworthy than the levels, and the ordering that would most
change the advice is `count n` against the density-aware proxies on deep-and-informal parcels --
currently decided by a label set that is biased against `count n`.
