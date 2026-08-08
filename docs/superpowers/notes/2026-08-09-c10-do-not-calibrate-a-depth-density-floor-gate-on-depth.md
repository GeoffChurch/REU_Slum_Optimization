# C10: don't calibrate a `depth_density` floor — gate on depth itself (2026-08-08)

C9 established that deep blocks (k0 >= 4) are the more informative population. `conf/metric/depth_density.yaml`
says "a calibrated absolute depth*density floor is legitimate future work", so this is that work.
**The answer is not to do it.**

## The sweep: no usable floor exists

`DenseCompactScreen.selection_scores` returns the FINE score per block, which for `Product(Depth,
Density)` is `depth * (count/area)`. Density is free from the columns, so true depth is recoverable
as `fine / density` with no re-peel — C8 had already populated the memoized cache.

4,936 scored blocks (the top-30% proxy pre-filter over Cape Town). True k0 distribution:

    k0     -1     0      1      2     3    4    5   6
    n      95   836  2,560  1,215   105   56   51   5

Only **112 of 4,936 (2.3%)** are at k0 >= 4, and the median is k0 = 1.1. Sweeping an absolute floor:

       floor     pool  % scored  median k0  share k0>=4
           0    4,936    100.0%        1.1         1.4%
      0.0171    1,481     30.0%        1.8         4.5%
       0.025      494     10.0%        2.4        11.9%
      0.0332      247      5.0%        3.1        21.1%
      0.0457       99      2.0%        3.8        42.4%
      0.0577       50      1.0%        4.5        68.0%

**No floor produces a majority-deep pool at a usable size.** The best trade is the 2% cut: 99 blocks,
42% deep. The 1% cut reaches 68% deep with 50 blocks. Compare `DENSITY_COMPACTNESS_FLOOR`, which
metric.py records as buying median depth 4 while keeping 823-1,646 Cape Town blocks.

## My diagnosis was wrong, and the truth is more useful

I predicted the product would be swamped by whichever term has the wider dynamic range. **It is not.**
On the scored pool the p99/p1 ratios are comparable — depth 4.0x, density 4.5x, compactness 6.4x.

The actual correlations are the surprise:

    depth_density       vs true depth   +0.538
    depth_density       vs density      +0.712
    density_compactness vs true depth   -0.040
    compactness alone   vs true depth   +0.058
    density alone       vs true depth   -0.072

So `depth_density` **is** depth-selective (+0.538, as it must be — depth is a factor), just weakly,
which is why its floor buys depth slowly. And `density_compactness` has essentially **zero**
correlation with true depth on this pool.

That last number sits awkwardly against C1–C7, whose density_compactness-selected sample ran k0 3–5.
The likely resolution is that the depth in that sample came from `load_pools`' other filters
(`building_count` 60–300, `>= 50` parcels, `<= 150` parcels) rather than from the metric — but that
is a hypothesis, not a measurement, and the two populations here are different subsets so the
comparison is not clean. **Flagged rather than resolved.**

## The recommendation

**Gate on depth directly, and give `Depth` a calibrated floor.** The screen's whole rationale is
cheap proxy selection over free columns — but `Depth.needs_peel` is already `True`, so
`depth_density` is *paying for the peel anyway*. Once the peel is paid, gating on a product that
dilutes depth is strictly worse than gating on depth itself.

`conf/metric/depth.yaml` already exists with an absolute gate — at 2.0 rings, which nearly every
block clears and which does no work. From the distribution above, a floor at depth >= 4 (k0 >= 3)
keeps 217 of these 4,936, and depth >= 5 (k0 >= 4) keeps 112; over the full corpus rather than the
top-30% pre-filter both would be larger. That is the number to calibrate, and it is a one-line
change to an existing config rather than new machinery.

## A geopandas trap worth recording

`gdf.area` is geopandas' geometry-area **property** and silently shadows a column named `area`. A
first version of the correlation analysis used attribute access, got degree-squared areas, and
returned `n=0` after filtering. The C10 sweep itself used bracket notation throughout and is
unaffected. Use `gdf["area"]`, or do not name a column `area`.

## Caveats

* Cape Town only; the pool is the top-30% proxy pre-filter, not the full corpus, so absolute counts
  are lower bounds.
* 95 blocks recovered k0 = -1, i.e. depth 0 — `block_depths` returns 0.0 for blocks it cannot peel
  and `Depth.fine` passes that through. They are excluded from the correlations.
