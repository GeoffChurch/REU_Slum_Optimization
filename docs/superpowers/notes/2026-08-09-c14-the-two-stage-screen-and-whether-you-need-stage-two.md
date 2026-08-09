# C14: the two-stage screen — and whether stage 2 is worth running at all (2026-08-08)

The owner raised two points C13 did not address.

**First, a correction of my framing.** n/P^2's advantage of "never needing Voronoi" is real but is not
a differentiator: **every metric compared in C13 is a free-column metric.** `dd_proxy` there is
`Depth.proxy x Density.proxy = sqrt(nA)/P x n/A`, computed from `building_count`, area and perimeter
alone — it is `Product(Depth, Density).proxy`, the pre-filter form, not the peeled `fine` form. So
`density` (AUC 0.922) and the `depth_density` proxy (0.921) beat n/P^2 (0.841) with no expensive
stage anywhere.

**Second, the real design question.** The screen is a pipeline, and the two stages want different
things from stage 1:

* with **no stage 2**, stage 1 must be PRECISE at the retention you ship;
* with **a stage 2**, stage 1 only needs RECALL — it must not discard informal blocks before the
  fine metric can rank them, and precision becomes stage 2's job.

## Stage 1 alone

    metric                        1%              5%             10%             30%
                             prec   rec      prec   rec      prec   rec      prec   rec
    depth_density proxy     81.7% 19.6%    41.7% 50.2%    27.7% 66.6%    13.0% 93.7%
    density n/A             62.8% 15.1%    41.7% 50.2%    28.1% 67.6%    13.1% 94.6%
    density_compactness     56.1% 13.5%    32.2% 38.8%    24.4% 58.9%    11.1% 80.2%
    depth proxy             46.3% 11.1%    24.2% 29.1%    16.6% 40.0%     8.3% 60.3%
    building_count          24.4%  5.9%    17.5% 21.1%    13.3% 31.9%     7.6% 55.1%

**n/P^2 is dominated on BOTH axes at every retention.** At 30% it keeps only 80.2% of informal
blocks where `density` keeps 94.6% — so as a stage-1 filter it throws away one informal block in five
before the expensive stage ever sees it.

## Stage 1 -> stage 2 (peel the survivors, rank, take the final top-164 ≈ 1%)

    stage-1 metric        fine = true depth              fine = depth x density (true)
                          1%     5%    10%    30%   |    1%     5%    10%    30%
    depth_density proxy 81.7% 72.0% 66.5% 59.1%     | 81.7% 84.1% 84.1% 84.1%
    density n/A         62.8% 83.5% 75.6% 70.1%     | 62.8% 84.1% 84.1% 84.1%
    density_compactness    -- 75.0% 68.9% 69.5%     |    -- 79.3% 83.5% 83.5%
    depth proxy            -- 50.6% 47.6% 45.7%     |    -- 81.7% 85.4% 85.4%
    building_count         -- 45.7% 48.2% 45.1%     |    -- 71.3% 79.3% 84.8%

Three things fall out.

**1. Ranking on depth ALONE is the wrong fine metric.** 59–83% against 84–85% for depth x density.
Density carries most of the signal even after you have paid for the peel — consistent with C13,
where true depth scored AUC 0.725 and density 0.922.

**2. With a good fine metric, stage 1 barely matters once retention is >= 10%.** Everything converges
to 83–85%, and even `building_count` reaches 84.8% at 30%. This is the recall story made concrete:
stage 1's only job is not to lose things.

**3. And the punchline — stage 2 may not be worth running.** Compare the cheap screen alone at
top-164 against the full pipeline:

    stage 1 alone (no peel)      dd_proxy 81.7%   density 62.8%   n/P^2 56.1%
    full two-stage pipeline      dd_proxy 84.1%   density 84.1%   n/P^2 83.5%

**Screening with `dd_proxy` and stopping gets 81.7%; adding the entire expensive stage buys 2.4
points.** If stage 1 is `density` the peel is worth +21.3 points, and if it is n/P^2, +27.4 — but
those are arguments for changing stage 1, not for paying for stage 2.

## Answers

* **No expensive stage:** `depth_density` proxy at 1%, 81.7% precise. Beats `density` (62.8%) and
  n/P^2 (56.1%) outright.
* **With an expensive stage:** stage 1 should be `density` or `dd_proxy` (recall 94.6% / 93.7% at
  30%, against n/P^2's 80.2%), and the fine metric should be depth x density, not depth.
* **n/P^2 is not best in either regime, at any retention.** The intuition that it uniquely picks out
  informal fabric cheaply does not survive contact with real ground truth — `depth_density`'s proxy
  does the same job better, equally cheaply.
* The shipped absolute floor selects **1,644 blocks at 24.5% precision and 58.9% recall**.

## Caveats

* Cape Town only; ground truth is C13's derived settlement extents.
* 9,825 blocks needed peeling for this; 9,608 succeeded (217 unbuildable).
* `FINAL_N = 164` is ~1% of the corpus, chosen to match the shipped selection's order of magnitude.
  Different final sizes would shift the absolute numbers, though the ordering was stable at every
  retention tested.
* Precision here is "block overlaps a settlement extent by >= 30%", which C13 showed is insensitive
  to the 30% threshold.
