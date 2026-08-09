# C13: against real ground truth, the shipped screen is 24.5% precise — and `density` beats `density_compactness` (2026-08-08)

C11/C12 used hand-drawn discs and said so. This replaces them with the City of Cape Town's own
informal-settlement structure survey.

## The data

**117,336 informal dwelling polygons**, digitised from February 2018 City of Cape Town aerial
photography at 1:200, published via University of Edinburgh DataShare
([doi:10.7488/ds/2758](https://doi.org/10.7488/ds/2758)). Median structure area **29.5 m²** —
genuinely shack-scale. Cached at `~/.cache/reblock/coct_is/`.

Provenance note: the openAFRICA mirror is behind a Cloudflare challenge and returns HTML, and the
City's own ArcGIS ODP has no informal-settlement boundary layer (searching `settlement`, `dwelling`,
`housing`, `human settlements`, `backyard` returns only Informal *Trading* and Housing Offices).
DataShare is the working source.

The file has **no settlement-name field** — `FID_1` puts 115,327 of 117,336 rows in one group, so it
is not a grouping. Settlement extents are therefore derived: DBSCAN over structure centroids at 30 m
/ min 10, then a 20 m buffered union per cluster, keeping clusters of >= 20 structures.

    117,336 structures -> 244 clusters (1.5% noise)
    189 settlements, 15.4 km², 114,909 structures (97.9% of all)
    683 informal blocks of 16,451 (4.15%), at cover >= 30%

**Sanity check that lands well:** informal blocks have median density **9,229/km²** against formal
**3,031/km²**. C1–C7's sample — selected by an entirely different route — had median 9,539/km². The
blocks this project has been scoring really are informal-settlement fabric.

## Result

    metric                             AUC   prec@1%   prec@5%
    density n/A                      0.922     62.8%     41.7%
    depth_density proxy              0.921     81.7%     41.7%
    density_compactness n/P^2        0.841     56.1%     32.2%
    TRUE PEEL DEPTH                  0.725                        (n=484)
    depth proxy sqrt(nA)/P           0.708     46.3%     24.2%
    building_count                   0.662     24.4%     17.5%
    compactness A/P^2                0.536     10.4%      7.5%

    median depth: informal 4.0 vs formal 3.0

* **`density` and `depth_density` are tied on AUC (0.922 / 0.921)** — but `depth_density` has far
  better precision at the top: **81.7% vs 62.8% at the top 1%**. For a screen that takes a top
  slice, that is the number that matters, and `depth_density` wins it decisively.
* **`density_compactness` is third on both** (0.841, 56.1%).
* **`compactness` alone is near chance (0.536)**, confirming C12 on real labels. Multiplying density
  by it costs 0.922 -> 0.841.
* True depth is a genuine signal (0.725, median 4 vs 3) — much better than C12's disc-based Cape Town
  figure of 0.352, which is a measure of how bad those discs were.

**Stable across the labelling threshold.** Sweeping cover from 10% to 90%, `density` wins at every
threshold and the full ordering is unchanged; AUCs move monotonically (density 0.899 -> 0.954) as the
label tightens, which is the expected direction.

## The number that matters for this project

    DENSITY_COMPACTNESS_FLOOR selects 1,644 Cape Town blocks,
    of which 24.5% are really informal settlement.

Three quarters of what the shipped screen selects is not informal settlement. By comparison, the
top 1% by `depth_density` proxy (164 blocks) is 81.7% informal.

This does not invalidate C1–C9 — those blocks were additionally filtered by building count and
parcel count, and their measured density of 9,539/km² matches the informal median almost exactly, so
that particular sample was fine. But as a *screen*, `density_compactness` is selecting three
non-settlement blocks for every settlement one, and a better instrument is available at no extra
cost.

## Recommendation

Replace `density_compactness` with `depth_density` as the default screen metric, and calibrate its
absolute floor against this ground truth rather than against a percentile. C10 could not find a
usable floor because it was optimising for *depth*; the right target is *informality*, and on that
target `depth_density` is both the most precise metric available and already configured.

## Caveats

* Cape Town only. Nairobi has no equivalent published survey found so far.
* 2018 structures against kblock blocks built from later OSM/Open Buildings — some temporal drift.
* DBSCAN parameters (30 m, 20 m buffer, >= 20 structures) and the 30% cover threshold are choices.
  The cover threshold was swept and does not change the ranking; the DBSCAN parameters were not.
* `prec@1%` is over 164 blocks; `prec@5%` over 822.
