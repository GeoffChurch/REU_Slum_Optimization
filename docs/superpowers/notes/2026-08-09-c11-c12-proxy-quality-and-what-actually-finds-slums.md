# C11/C12: the depth proxy is decent, and `density` — not `density_compactness` — finds slums (2026-08-08)

Two questions from the owner. Both measured.

## C11 — how well does `Depth.proxy = sqrt(nA)/P` track true Voronoi depth?

This matters because the proxy runs as a top-`proxy_keep_pct`% PRE-FILTER before any gate sees a
block. If it were weak, C10's "no `depth_density` floor produces a deep pool" would be a symptom of
the pre-filter rather than the gate. Measured on a stratified sample, 60 blocks per proxy decile,
peeled for real (n = 568) — C10 could not answer this because it only had depths for top-30%
survivors, which is range-restricted by construction.

    Spearman rho vs true depth, full proxy range
      Depth.proxy = sqrt(nA)/P      +0.590
      building_count                +0.501
      area                          +0.170
      compactness A/P^2             +0.142
      density_compactness n/P^2     +0.128
      density n/A                   +0.093

    share of blocks at k0 >= 3, by proxy decile
      0     1     2     3     4     5     6     7      8      9
    3.6%  1.9%  5.9%  1.8%  5.3%  1.7%  3.4% 13.3%  39.7%  90.0%

**+0.590 overall, +0.655 within the top 30%.** The proxy is sharply NON-LINEAR — flat and
uninformative through deciles 0–6, then steep. And **87% of genuinely deep blocks land in the
top-30% band**, so the pre-filter discards only 13%.

So the pre-filter is not the problem, and C10's conclusion stands: gate on depth directly. Note also
that plain `building_count` gets +0.501 — most of the proxy's power is "more buildings, deeper block".

## C12 — the reference block is the whole argument, and it confirms the owner's intuition

The owner proposed that `density_compactness` beats depth at finding real slums, because rural blocks
can be deep while their buildings are spread out. Their own reference block settles the second half:

    ZAF.9.3.1_1_5810          value    percentile
    building_count              714         p99.4
    density               1,238/km²         p13.5
    compactness              0.0184         p11.9
    density_compactness    2.27e-05         p11.8
    depth proxy                3.62         p99.3
    area                    0.58 km²        p97.2
    TRUE PEEL DEPTH              24

**Depth 24 at 1,238 buildings/km²** — against this project's own informal floor of ~4,500–5,700/km²
and C1–C7's sample median of 9,539. It is deep because it is *large*, not because it is dense. Depth
picks it at p99.3; `density_compactness` rejects it at p11.8. Exactly the failure mode described.

**This also invalidated my first ground-truth run**, which pooled a 2 km disc around this block with
Khayelitsha and called both "settlement". Re-run per site.

## C12 — AUC against named settlements

AUC = P(a random settlement block outscores a random other block); 0.5 is chance.

    metric                     khayelitsha   kibera   mathare
    density n/A                      0.797    0.677     0.600
    depth_density proxy              0.747    0.690     0.609
    density_compactness n/P^2        0.678    0.571     0.541
    building_count                   0.523    0.663     0.570
    depth proxy sqrt(nA)/P           0.410    0.619     0.566
    compactness A/P^2                0.370    0.410     0.502
    TRUE PEEL DEPTH                  0.352    0.681     0.582

Consistent across all three sites:

* **`density` alone is top-2 everywhere** and beats `density_compactness` at every site.
* **`compactness` is at or below chance everywhere** (0.370 / 0.410 / 0.502). Multiplying density by
  it makes things worse, not better: 0.797 -> 0.678 at Khayelitsha.
* **`depth_density` is the other consistent performer** and wins both Nairobi sites.
* **True depth is city-dependent** — good in Nairobi (0.681 / 0.582, median depth 4 vs 3) and BELOW
  CHANCE in Cape Town (0.352, median 2 vs 3): Khayelitsha's blocks are *shallower* than Cape Town's
  average.

**So the owner's diagnosis is right and the proposed cure is not.** Depth alone really is fooled by
sprawling low-density blocks — their own reference block proves it. But compactness is not what
fixes it; density is, and `density_compactness` is consistently worse than its density half alone.

## Caveats, and they are load-bearing here

* **Hand-drawn discs, not settlement boundaries.** There is no official informal-settlement layer in
  this data. A 3 km disc around Khayelitsha's centre sweeps in a great deal of formal township — and
  Khayelitsha is substantially site-and-service rather than unplanned, which likely explains its
  median depth of 2. That single fact should temper every Cape Town row above.
* The Nairobi discs are tighter and better targeted, but their median density is only 4,440/km²
  (Kibera) and 4,069 (Mathare) against a corpus median of 2,854 — a 1.5x separation, so those discs
  are diluted too.
* Precision@1% is 0–20% throughout, against base rates of 1.8–4.3%. Real lift, low absolute numbers.
* Getting this right needs actual informal-settlement polygons (Cape Town's municipality publishes
  one; Nairobi has several sources). That is the measurement that would make this decisive.
