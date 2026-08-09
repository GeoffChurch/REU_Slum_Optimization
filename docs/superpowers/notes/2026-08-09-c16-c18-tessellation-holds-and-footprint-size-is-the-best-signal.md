# C16/C18: the Voronoi caveat largely closes, and footprint SIZE is the best informality signal (2026-08-08)

## C16 — does the tessellation matter?

The last open caveat on the access line: `block.parcels` are Voronoi cells of building POINTS, so
`k` — a quantity about crossing other people's PLOTS — has been computed over a modelled tenure
pattern rather than a measured one.

The experiment isolates the tessellation and nothing else. Both are seeded by the SAME buildings:
point-Voronoi from `block.building_points` (today's parcels) against footprint cells from the Open
Buildings POLYGONS for those same buildings, computed on a raster via
`distance_transform_edt(return_indices=True)` — propagating footprint labels outward gives exactly
the polygon analogue of Voronoi. Road prefixes are NOT re-derived; the same prefixes are re-scored
under both, so the methods are fixed and only the instrument varies.

    8 blocks, 6 methods, 659 curve points
    starting k0        voronoi median 2.0    footprint median 2.0
    burden correlation (Spearman)            +0.950

    budget   ranking under each tessellation                       tau
       5%    identical top three                                +1.000
      10%    identical top three                                +0.867
      20%    top two swap (topology <-> access)                 +0.690

**Same starting complexity, burdens correlating +0.950, and an unchanged champion at 5% and 10% —
the regime Lens A operates in.** Agreement loosens at 20%, where the top two swap.

So the access results hold under a measured tessellation. The caveat is not fully gone — it
degrades as budgets grow, and 8 blocks is a small sample — but it no longer threatens the
conclusions at the budgets actually compared.

## C18 — is Google Open Buildings viable as the missing Nairobi ground truth?

Answer: **as a signal yes, as an arbiter no**, and the second half is the trap worth recording.

### It predicts, better than anything else measured

On Cape Town against C13's real ground truth (683 informal blocks of 16,451):

    feature                        AUC     informal vs formal median
    p90 footprint area           0.943      63.7 vs 178.7 m²   (inverted)
    density n/A                  0.922
    depth_density proxy          0.921      <- the shipped screen
    median footprint area        0.919      28.9 vs  61.1 m²   (inverted)
    share of footprints <= 40 m² 0.911      0.69 vs 0.34
    built area fraction          0.748
    OB footprint count           0.703
    density_compactness          0.841

A single Open Buildings feature beats every block-geometry metric available. And the informal median
of **28.9 m² independently reproduces the City survey's 29.5 m²** — two unrelated datasets, agreeing
to within 2%, which is strong evidence both measure the real thing.

### But it is not independent of what it would be judging

    feature                     vs density    vs ddp   vs n/P^2
    p90 footprint area              -0.774    -0.743    -0.640
    built area fraction             +0.757    +0.703    +0.735
    median footprint area           -0.613    -0.614    -0.505
    share <= 40 m²                  +0.529    +0.539    +0.438
    OB footprint count              +0.011    +0.148    -0.204

The strong predictors correlate |rho| 0.5–0.77 with the metrics they would evaluate. The one
genuinely independent feature — count, rho +0.011 — is weak at 0.703. **Using OB morphology as
ground truth to adjudicate between density-based screens would largely be testing density against a
proxy for density.**

So the Nairobi gap is NOT solved by reaching for Open Buildings, which was the obvious move and is
the reason this is written down.

### What it IS good for

* a sanity check in cities without ground truth — the 29 vs 61 m² split is real and interpretable;
* a **feature** in a better screen. Backlogged: a single OB feature already beats the shipped metric,
  at the cost of a polygon download the current screen does not need (0.32 GB for Cape Town's tile;
  14.09 GB only for all 20 ZAF+KEN tiles at once). Combining morphology with block geometry is
  untested and should beat either alone.

## Caveats

* C16 is 8 blocks; C18 is Cape Town only.
* C16's footprint cells come from Open Buildings polygons, not the City structure survey, deliberately
  — the survey is a different building set, and the point of the experiment was to hold buildings
  fixed and vary only the tessellation.
* C18's join covers 16,431 of 16,451 blocks (20 dropped for fewer than 5 OB footprints).
