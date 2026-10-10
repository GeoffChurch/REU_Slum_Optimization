# FFT homogenization on 5810@major: the deciding experiment

2026-10-10, GeoffChurch/reblock, branch research/roadless.

The run added only new `region_*` files in `research/roadless/fft_homog/` and changed no other file. Every process ran with one thread, with at most 12 worker processes and at most one GPU process at a time. No `__pycache__` or numba cache was written into the repo.

## Summary

**Verdict.** Criterion 1 passes. Criterion 2 is not settled by the bound you asked for, but passes at k = 5 by a stronger bound I added. With the 5810 study's own screening map (the first-order map at L 50) the rule fails.

- **Criterion 1, tile Spearman ≥ 0.9: passes** for:
  - the two-scale re-solve, at L 50 and at L 100;
  - the first-order map at L 100.

  Their ρ is 0.975–0.989, raw and per displaced m², over 160 stratified tiles per rule; every 95% interval is at or above 0.958.

  It **fails** for the 5810 study's map, the first-order map at L 50: per m² on STRIP, ρ = 0.882 [0.823, 0.936].
- **Criterion 2, restricted run ≥ 90%: the requested bound does not clear it.** Truncating each stored clearing to the top tiles keeps at most 88.8% of its Lens A (two-scale L 50 ranking, k = 5). With the specified first-order L 50 ranking it keeps 50–85%.
- **The stronger bound (my addition) clears it at k = 5, not below.** It is a feasible clearing at the full budget built without an optimizer: the truncation, topped up within the same tiles by the map's own sensitivity.
  - At k = 5 it keeps 90.5–94.6% of each optimizer's Lens A with the first-order L 100 ranking, and 91.3–94.9% with the two-scale L 50 ranking.
  - At k ≤ 3 it keeps 66–88.5%.
- **What k = 5 means.** It is 112–114 of the 2,518 tiles. They hold 5 × the budget's footprint, which is 25% of all footprint in the region.
- **Recommendation.**
  - Keep two-scale as a screening layer, ranked by the first-order map at L 100, with k = 5.
  - Drop the L 50 first-order map.
  - A real restricted optimizer run is needed only to learn whether k = 2–3 also keeps 90%, or to confirm k = 5 with the optimizers themselves.
  - For the cheap preset that run needs no change to the search, only a driver flag; SIMP needs about 50 lines (section 5).
- **Why the L 50 first-order map fails.** Large buildings nearly block some 50 m windows: 78 of the 68,267 L 50 windows have σ < 0.01, while no L 100 window falls below 0.04.
  - On those tiles the linear prediction overshoots the fine gain 51–205×. One R20 tile is predicted at 21% of J₂; its fine gain is 0.18%.
  - Summed over all R20 tiles, the L 50 map predicts 153% of J₂. The L 50 re-solve predicts 63% and the L 100 map 60%.
- **J₂ (item 1).**
  - L 100 windows: +2.2% (inside-avg), +2.8% (extrapolated), −2.3% (open-outside). Per-home Spearman is 0.9986.
  - L 200 inside-avg: −1.8%. L 50: +9.6% to +12.1%.
  - At L 100 inside-avg, homes deeper than 100 m (97.8% of J₂) are within a median +0.7% to +1.9%. Homes within 10 m of the exit read +56%, but they carry about 0% of J₂.
- **Stored clearings (item 2).** Two-scale keeps the order cheap < SIMP in all six variants. The SIMP − cheap gap is:
  - 0.030 under lifted;
  - 0.018 under the fine scalar model;
  - 0.009–0.010 under two-scale at L 50;
  - 0.014–0.016 under two-scale at L 100.
- **Costs.**
  - Total: 14.1 CPU-hours logged plus about 0.4 CPU-hours of checks, 21 GPU-minutes, and about 2 h 10 min of wall time out of the 6 h budget.
  - Peaks: 33.7 GB for the largest process tree; about 60 GB machine-wide (estimated); 34.8 GiB on the GPU.
  - Fine scalar baseline: 252 s and 14.4 GB.
  - Tensor fields: 84–265 s of wall time on 11 workers.

## Setup, and what differs from the 5810 study

- **Region.**
  - Area 6.740 km² with 12,663 buildings (18 stranded), 18.8 buildings/ha.
  - The outer ring is the only exit (69,373 ground cells).
  - Lattice: h 0.5 at `lifted.OFFSET`, 5,998 × 11,354 px, 26.97M of them inside.
  - Fine scalar model: 25,361,310 unknowns, J₂ = 1.95920e10, P = 1.28588e7.
  - Home depth: median 269 m, p90 518 m, maximum 607 m.
- **Index check (passed).**
  - The stored ids, priced by `b.buildings.outlines` areas, give exactly the stored D (0.049957 and 0.049989) with n = 12,663.
  - Rescoring on the local GPU reproduces both stored values to 6 digits: lifted uni on h0.5a5x8 (5,524,273 cells, `relax.Relaxation.exact`) gives 0.271361 and 0.301406.
- **Two-scale model.** The 5810 study's model is unchanged:
  - `ts.tensor_field` and `ts.fill`;
  - the Q1 macro problem at H 2 m (1,681,493 unknowns; it fits);
  - readout U, as in the 5810 tables.

  I re-implemented the macro problem in `region_macro.py` with ts's formulas. Its readout is asserted equal to `ts.evaluate` + `ts.home_u` to 1e-9, and it adds an exact adjoint and a local re-solve.
- **Changes from the 5810 tile study (`tiles_clear.py`).**
  - Items 3–4 use the inside-avg near-exit treatment; the 5810 tile study used open-outside.
  - STRIP's direction comes from the L 50 inside-avg macro gradient.
  - Window updates use `ts._window`'s σ and σ_in without the source corrector. They reproduce the stored field to 8.7e-11.
- **Fine re-solves.**
  - Each re-solve warm-starts from the baseline. It reuses the baseline's SA hierarchy inside a symmetric multiplicative Schwarz preconditioner: an exact solve within 20 m of the change, the baseline's V-cycle, then the exact solve again.
  - It stops at 1e-8 of the initial residual: a median of 13 CG iterations and 59 s, against 293–319 s for a fresh solve.
  - Against fresh solves it agrees to 2.4e-7 and 1e-8 (relative) on 5810, and to 1.0e-6 and 5.5e-5 on the region. The 5.5e-5 is at the fresh solve's own precision.
  - An additive form of the same preconditioner needed 94–158 iterations.
  - The macro re-solves use the same local Schwarz and agree with fresh macro solves to 1e-7–1e-10.
- **Sample.** For each rule, 16 tile-rules per decile of the first-order L 50 prediction: 160 + 160 tile-rules, so 320 fine re-solves.
  - Of the 2,518 fully-inside 50 m tiles, 646 have R20 candidates and 563 have buildings on the STRIP.
  - Intervals come from a 2,000-fold bootstrap within the deciles.

## 1. Baseline J₂: fine scalar vs two-scale (macro H 2 m, readout U)

Error = two-scale minus fine. The band columns are the median relative error of the homes in that band (distance to the exit, m).

| L (stride), m | near-exit | J₂ err | P err | Spearman | median abs | 0–10 | 10–30 | 30–100 | 100–200 | 200–300 | 300–450 | ≥450 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 (25) | open-outside | −2.3% | −1.5% | 0.9983 | 1.6% | +25% | +2.6% | −7.0% | −2.4% | −0.8% | −0.6% | −0.5% |
| 100 (25) | extrapolated | +2.8% | +1.6% | 0.9985 | 2.3% | +56% | +33% | +4.3% | +2.7% | +2.4% | +1.6% | +1.0% |
| 100 (25) | inside-avg | +2.2% | +1.2% | 0.9986 | 1.9% | +56% | +26% | +2.6% | +1.9% | +1.8% | +1.1% | +0.7% |
| 200 (50) | open-outside | −12.1% | −6.5% | 0.9949 | 5.7% | +23% | +2.1% | −14% | −10% | −5.4% | −4.3% | −2.7% |
| 200 (50) | extrapolated | −3.7% | −1.4% | 0.9970 | 2.1% | +56% | +31% | +1.5% | −1.2% | −0.3% | −0.7% | −0.2% |
| 200 (50) | inside-avg | −1.8% | −0.3% | 0.9975 | 2.0% | +57% | +32% | +3.3% | +0.3% | +1.0% | +0.1% | +0.3% |
| 50 (10) | open-outside | +9.6% | +4.1% | 0.9984 | 2.7% | +22% | +2.9% | −0.4% | +1.8% | +2.5% | +2.5% | +2.6% |
| 50 (10) | extrapolated | +12.1% | +5.5% | 0.9983 | 3.9% | +54% | +22% | +4.2% | +4.0% | +3.8% | +3.4% | +3.2% |
| 50 (10) | inside-avg | +10.8% | +4.8% | 0.9984 | 3.3% | +46% | +15% | +2.7% | +3.1% | +3.2% | +3.0% | +2.9% |

What each band holds, and the median two-scale minus fine u:

| band (m) | 0–10 | 10–30 | 30–100 | 100–200 | 200–300 | 300–450 | ≥450 |
|---|---|---|---|---|---|---|---|
| homes / share of J₂ | 36 / 0.00% | 445 / 0.04% | 1,785 / 2.2% | 2,405 / 11.7% | 2,427 / 20.5% | 2,949 / 37.1% | 2,598 / 28.5% |
| fine median u | 14.5 | 60 | 247 | 834 | 1,317 | 1,576 | 1,445 |
| L 100 inside-avg | +7.7 | +12.5 | +4.0 | +9.3 | +16.9 | +14.2 | +9.1 |
| L 100 extrapolated | +8.2 | +16.8 | +7.5 | +13.2 | +24.5 | +20.2 | +12.9 |
| L 200 inside-avg | +8.5 | +16.3 | +5.4 | +1.4 | +9.0 | +1.1 | +2.8 |
| L 200 open-outside | +3.2 | +0.9 | −28.6 | −72.6 | −67.7 | −63.0 | −36.0 |
| L 50 inside-avg | +5.5 | +6.7 | +4.4 | +15.6 | +41.3 | +44.3 | +37.7 |

- **The 5810 picture holds at region scale.**
  - At L 100–200 the error is a roughly constant offset reached near the exit, so the interior has converged.
  - At L 50 the error grows with depth: the interior is biased high, as on 5810.
  - Counting outside cells as open (open-outside) inflates σ near the exit, which drives the large negative errors at L 200.
- **Macro resolution.** L 100 at H 5 m (269k unknowns, 4 s) gives J₂ errors of −1.7%, +3.7% and +2.9% (open-outside, extrapolated, inside-avg), against −2.3%, +2.8% and +2.2% at H 2 m.
- **Correctors.** Adding χ·∇U + w moves J₂ by at most 0.6 points. It cuts the 0–10 m band's median error, for example 56% → 33% at L 100 inside-avg.

## 2. The stored clearings (D 0.05)

Window counts recomputed per clearing:
- L 50: 4,733–5,051 of 68,267.
- L 100: 1,144–1,215 of 11,126.

| model | cheap preset | SIMP default | SIMP − cheap |
|---|---|---|---|
| lifted uni, a5x8 (stored, reproduced) | 0.2714 | 0.3014 | 0.0300 |
| fine scalar, h 0.5 | 0.2122 | 0.2302 | 0.0180 |
| two-scale L 50, open-outside | 0.2095 | 0.2184 | 0.0089 |
| two-scale L 50, extrapolated | 0.2138 | 0.2238 | 0.0100 |
| two-scale L 50, inside-avg | 0.2105 | 0.2203 | 0.0098 |
| two-scale L 100, open-outside | 0.1910 | 0.2046 | 0.0136 |
| two-scale L 100, extrapolated | 0.1969 | 0.2126 | 0.0156 |
| two-scale L 100, inside-avg | 0.1978 | 0.2140 | 0.0161 |

- **Order:** kept in all six variants.
- **Gap:**
  - L 50 keeps 49–56% of the fine scalar gap and about a third of the lifted one.
  - L 100 keeps 76–89% of the fine scalar gap and about half of the lifted one.
- **Levels:**
  - L 50 is within −0.012 to +0.002 of the fine scalar value.
  - L 100 reads 0.015–0.026 low; larger windows dilute a localized opening.

## 3. Tile ranking

320 tile-rules, 160 per rule, stratified as above. ρ is the Spearman correlation with the fine ΔJ₂, with 95% intervals. The top-decile column counts how many of the fine model's top 16 the predictor also puts in its top 16.

| predictor | ρ R20 | ρ/m² R20 | top 10% R20 (raw, /m²) | ρ STRIP | ρ/m² STRIP | top 10% STRIP (raw, /m²) |
|---|---|---|---|---|---|---|
| two-scale re-solve L 50 | 0.987 [0.980, 0.992] | 0.975 [0.959, 0.986] | 12/16, 13/16 | 0.988 [0.983, 0.992] | 0.987 [0.980, 0.991] | 13/16, 13/16 |
| first-order map L 50 | 0.982 [0.973, 0.988] | 0.935 [0.891, 0.968] | 10/16, 11/16 | 0.948 [0.918, 0.973] | **0.882** [0.823, 0.936] | 9/16, 10/16 |
| two-scale re-solve L 100 | 0.986 [0.979, 0.991] | 0.977 [0.961, 0.987] | 14/16, 14/16 | 0.989 [0.984, 0.993] | 0.989 [0.983, 0.992] | 13/16, 14/16 |
| first-order map L 100 | 0.986 [0.979, 0.992] | 0.976 [0.960, 0.987] | 14/16, 14/16 | 0.989 [0.983, 0.992] | 0.987 [0.979, 0.991] | 13/16, 14/16 |
| naive proxy | 0.795 | 0.709 | 1/16, 0/16 | 0.734 | 0.663 | 2/16, 1/16 |

- **Calibration** (prediction / fine, median with p10–p90 in parentheses):
  - R20: L 50 re-solve 1.22 (0.99–2.19), L 50 map 1.38 (1.06–2.84), L 100 re-solve 1.11 (0.83–1.61), L 100 map 1.14 (0.86–1.66).
  - STRIP: L 50 re-solve 1.14 (0.95–2.09), L 50 map 1.31 (1.04–4.12), L 100 re-solve 1.04 (0.83–1.63), L 100 map 1.10 (0.88–1.80).
- **Fine gains are small and skewed.**
  - R20: median 0.015% of J₂, maximum 3.8%; median 4 buildings and 186 m² per tile.
  - STRIP: median 0.038% of J₂, maximum 0.98%; median 6 buildings and 267 m² per tile.
  - One tile per rule raises J₂ slightly (by 0.0004%).
- **Top half by fine gain** (80 tile-rules per rule), ρ for R20 / STRIP:
  - L 50 re-solve 0.945 / 0.956;
  - L 50 map 0.916 / 0.814;
  - L 100 re-solve 0.954 / 0.957;
  - L 100 map 0.957 / 0.952.
- **Against 5810** (Spearman 0.97–0.99, top-decile overlap 88–94%): the region's top-decile overlap is lower, 56–88%.

## 4. Restricting to the map's top areas

- **Budget footprint:** 0.05 × 437,884 m² = 21,894 m².
- **Tiles:** ranked by predicted gain per displaced m² (R20 or STRIP). The top tiles are taken until their footprint reaches k × the budget's. A building belongs to the tile holding its polygon centroid.
- **Rankings:**
  - the first-order L 50 map (the specified one, the 5810 study's map);
  - the first-order L 100 map and the L 50 two-scale re-solve, both added after the L 50 failure.

  Those two were computed for every non-empty tile-rule: 1,209 in all, 889 new plus the 320 sampled ones.
- **Scoring:** lifted uni on a5x8, on the GPU, against the unrestricted 0.2714 (cheap) and 0.3014 (SIMP).
- **Top-up (my addition, not asked for):** start from the truncation, then add the same tiles' other buildings in order of (S_xx + S_yy) under their footprint, from the L 100 macro adjoint, while they fit D = 0.05.

Cells are cheap / SIMP; the truncated share is the requested bound.

| ranking | rule | k | tiles | cleared footprint in tiles | truncated D | truncated Lens A | **truncated share** | top-up Lens A (D 0.05) | top-up share |
|---|---|---|---|---|---|---|---|---|---|
| first-order L50 | R20 | 2 | 36 | 39% / 41% | 0.019 / 0.020 | 0.1596 / 0.1757 | **58.8% / 58.3%** | 0.2178 / 0.2174 | 80.2% / 72.1% |
| first-order L50 | R20 | 3 | 57 | 50% / 54% | 0.025 / 0.027 | 0.1854 / 0.2052 | **68.3% / 68.1%** | 0.2324 / 0.2383 | 85.6% / 79.1% |
| first-order L50 | R20 | 5 | 97 | 70% / 72% | 0.035 / 0.036 | 0.2213 / 0.2450 | **81.6% / 81.3%** | 0.2466 / 0.2634 | 90.9% / 87.4% |
| first-order L50 | STRIP | 2 | 38 | 36% / 35% | 0.018 / 0.017 | 0.1406 / 0.1520 | **51.8% / 50.4%** | 0.2043 / 0.2002 | 75.3% / 66.4% |
| first-order L50 | STRIP | 3 | 59 | 53% / 52% | 0.027 / 0.026 | 0.1891 / 0.2017 | **69.7% / 66.9%** | 0.2300 / 0.2322 | 84.7% / 77.0% |
| first-order L50 | STRIP | 5 | 102 | 73% / 74% | 0.036 / 0.037 | 0.2299 / 0.2520 | **84.7% / 83.6%** | 0.2533 / 0.2690 | 93.4% / 89.2% |
| first-order L100 | R20 | 2 | 43 | 42% / 49% | 0.021 / 0.025 | 0.1726 / 0.1966 | **63.6% / 65.2%** | 0.2268 / 0.2340 | 83.6% / 77.6% |
| first-order L100 | R20 | 3 | 65 | 56% / 61% | 0.028 / 0.030 | 0.2004 / 0.2228 | **73.8% / 73.9%** | 0.2384 / 0.2487 | 87.9% / 82.5% |
| first-order L100 | R20 | 5 | 114 | 75% / 79% | 0.038 / 0.039 | 0.2324 / 0.2600 | **85.6% / 86.3%** | 0.2525 / 0.2727 | 93.1% / 90.5% |
| first-order L100 | STRIP | 2 | 43 | 46% / 52% | 0.023 / 0.026 | 0.1825 / 0.2049 | **67.3% / 68.0%** | 0.2316 / 0.2387 | 85.4% / 79.2% |
| first-order L100 | STRIP | 3 | 64 | 60% / 63% | 0.030 / 0.031 | 0.2073 / 0.2284 | **76.4% / 75.8%** | 0.2401 / 0.2512 | 88.5% / 83.4% |
| first-order L100 | STRIP | 5 | 112 | 79% / 79% | 0.040 / 0.039 | 0.2395 / 0.2636 | **88.3% / 87.5%** | 0.2567 / 0.2764 | 94.6% / 91.7% |
| two-scale L50 | R20 | 2 | 42 | 42% / 46% | 0.021 / 0.023 | 0.1602 / 0.1820 | **59.0% / 60.4%** | 0.2177 / 0.2219 | 80.2% / 73.6% |
| two-scale L50 | R20 | 3 | 64 | 59% / 64% | 0.029 / 0.032 | 0.2044 / 0.2291 | **75.3% / 76.0%** | 0.2402 / 0.2531 | 88.5% / 84.0% |
| two-scale L50 | R20 | 5 | 113 | 79% / 81% | 0.039 / 0.040 | 0.2384 / 0.2652 | **87.9% / 88.0%** | 0.2548 / 0.2752 | 93.9% / 91.3% |
| two-scale L50 | STRIP | 2 | 41 | 45% / 47% | 0.023 / 0.023 | 0.1671 / 0.1857 | **61.6% / 61.6%** | 0.2177 / 0.2215 | 80.2% / 73.5% |
| two-scale L50 | STRIP | 3 | 62 | 59% / 61% | 0.029 / 0.031 | 0.1923 / 0.2193 | **70.9% / 72.7%** | 0.2301 / 0.2468 | 84.8% / 81.9% |
| two-scale L50 | STRIP | 5 | 114 | 80% / 80% | 0.040 / 0.040 | 0.2408 / 0.2657 | **88.8% / 88.2%** | 0.2575 / 0.2780 | 94.9% / 92.2% |

- **Ceiling of any set of fully-inside tiles.** Truncating to all 2,518 tiles keeps 95.8% / 93.6% of Lens A (96% / 91% of the cleared footprint, D 0.048 / 0.045). The other 4–9% of the optimizers' cleared footprint lies in boundary tiles, which are never in the tile set.
- **Two reasons the truncations fall short:**
  - They underspend the budget: D is 0.017–0.040 instead of 0.05.
  - Under the L 50 map, the near-blocked big-building tiles crowd the top of the ranking. At k = 2 it shares only 26–28 tiles with the L 50 re-solve ranking's 41–42.

## 5. Decision, and whether a real restricted run is needed

- **Truncation bound.** It does not reach 90% at any k or with any ranking: the best is 88.8%, and 85% with the specified ranking.
- **Full-budget bound.** It clears 90% at k = 5 for both clearings, with the L 100 first-order map or the L 50 re-solve. That clearing is feasible for the restricted problem, so an optimizer restricted to those tiles has it in its search space.
- **So a real run is needed:**
  - to test k = 2–3, where both bounds stay at 66–88.5%;
  - to confirm k = 5 with the optimizers themselves.
- **Code changes (estimated, not made):**
  - **Cheap preset (greedy S0.01cat + polish P64w8): no change to the search.** `search.py`'s pickers add only buildings in `SearchState.movable` (`_addable`), and the polish's `Swaps` draws only from `movable`. `polish_greedy.main` builds two SearchStates; a restricted run would set `movable` from a mask on both, plus a mask argument and a row-directory suffix. About 20 lines in `polish_greedy.py`, or a new driver.
  - **SIMP default (`fw0.q3.i10.t0.001.e0.0001.k1s1e-06.p256w8x2.a5x8`):** about 40–60 lines in `relax.py`, about half a day including a check that unmasked plans stay bit-identical:
    - a per-building upper bound through `simp()` and `OC.trial` / `_MMAStep.trial`;
    - `start_x` spread over the mask;
    - `round_by_x` / `round_sampled` skipping masked buildings (they sit at XMIN, above the 1e-9 cutoff);
    - `Incumbent.polish`'s movable set intersected with the mask;
    - a plan suffix.
- **Run cost (estimated).**
  - The stored runs took 170 s and 386 s on the H100.
  - Here each exact score took 8–11 s and 34.8 GiB on the local 48 GB card, so the runs should fit it.
  - 3 values of k × 2 methods ≈ 1 GPU-hour.
  - A restricted run still solves the whole region at every score, so it is not much faster than an unrestricted one. Screening pays for choosing where a pooled budget goes across faces, not for speeding up one region.

## 6. Costs (measured)

**Concurrency.** The GPU work was one process at a time. The machine was shared: another job used about 14 cores. All runs together stayed far under 120 GB.

| piece | wall | CPU | peak memory |
|---|---|---|---|
| Region Block (OSM face) / rasterize at h 0.5, demand, labels | 23 s / 37 s | 36 s / 37 s | 0.6 / 20.9 GB |
| Fine baseline: matrix 6 s, SA setup 75 s, CG 45 iterations 170 s | 252 s | 255 s | 14.4 GB |
| Fine fresh solve of a stored clearing | 293–319 s | 295–309 s | 20.1 GB |
| Fine local re-solve per tile-rule (11 running concurrently) | median 59 s (p10–p90 53–69) | ≈ wall | about 2 GB per worker over the shared base |
| Tensor field L 100, stride 25 (11,126 windows, 97 ms each) | 99 s on 11 workers | 1,079 s | 4.6 GB |
| Tensor field L 50, stride 10 (68,267 windows, 13 ms each) | 84 s on 11 workers | 918 s | 4.0 GB |
| Tensor field L 200, stride 50 (2,863 windows, 1.0 s each) | 265 s on 11 workers | 2,867 s | 4.9 GB |
| Macro, H 2 m (assemble, SA, CG to 1e-11); with adjoint | 13–27 s; 27–48 s | ≈ wall | 5–6 GB |
| Stored clearing, two-scale update: L 50 / L 100 windows | 103–113 s / 134–146 s on 2 workers | 209–228 s / 270–297 s | 6–6.6 GB |
| First-order L 50, every tile × rule (5,036; 1,209 non-empty) | 283 s on 9 workers | 2,573 s (median 2.1 s per non-empty) | 11 GB |
| L 50 re-solve + L 100 map, 889 tile-rules | 1,098 s on 9 workers ∥ 3,175 s on 1 | 13,269 s (14.9 s each) | 12.4 GB |
| Sampled re-solves: fine + L 50 and L 100, 320 tile-rules | 2,495 s (8 workers) ∥ 2,106 s (3 workers) | 26,769 s (median 80 s each) | 33.7 / 19.2 GB |
| Per sampled tile: L 50 windows + macro / L 100 windows + macro | 1.5 + 9.3 s / 3.1 + 7.0 s | | |
| GPU setup (Block, a5x8 grid, baseline) / one exact score | 102–139 s / 8–11 s | 119–146 s | 13 GB host / 34.8 GiB device |

**Extrapolation to Cape Town's major-road faces.**
- **Measured.** `region_capetown_faces.py` polygonizes the major network over the kblock source's extent.
  - Built-up faces (≥ 5 buildings/ha): 689 faces, 718 km², 1.33M buildings at 18.5/ha. Faces: median 0.60 km², p90 2.1 km², maximum 26 km²; 9 are at least as large as 5810@major.
  - All faces with ≥ 10 buildings: 4,779 km². Of that, 3,957 km² is 50 rural faces of 10 km² or more, the largest 715 km².
- **Estimated.** The figures below scale the measured rates by area (×106.5) at a similar density.
  - Fine scalar baselines: 7.5 CPU-h. The 26 km² face would take about 16 min and 56 GB.
  - Tensor fields: L 100 32 CPU-h, L 50 27 CPU-h, L 200 85 CPU-h.
  - First-order map over every tile: L 50 76 CPU-h; L 100 about 130 CPU-h, from its window time.
  - Two-scale re-solve of every tile: ≤ 530 CPU-h.
  - Fine re-solve of every tile: ≤ 2,100 CPU-h. Per-tile cost grows with face size, so linear scaling overstates both.
  - Lifted uni on the GPU: about 4 GPU-h for one score per face. A 26 km² face would need about 135 GiB of device memory at a5x8, which does not fit a 48 or 80 GB card.
  - For one region, scoring every tile with the real metric on this card would take about 3.4 GPU-h, against 0.7 CPU-h for the first-order map.

## Files

**In `research/roadless/fft_homog/` (lint clean).**
- **Scripts:**
  - `region_common.py`: paths, thread check, cost monitor, fabric helpers.
  - `region_extract.py`, `region_fine.py`, `region_fields.py`, `region_macro.py`.
  - `region_twoscale.py`: items 1–2.
  - `region_tiles.py`: modes `firstorder`, `sample`, `resolve`, `alltiles`, `restrict`.
  - `region_topup.py`.
  - `region_score_gpu.py`: needs `CUDA_PATH=/usr`.
  - `region_analyze.py`: modes `tiles`, `homes`, `restrict`, `report`.
  - `region_capetown_faces.py`.
- **Tables:**
  - `region_twoscale.csv`, `region_twoscale_bands.csv`, `region_decomp_bands.csv`;
  - `region_clearings.csv`, `region_clearings_fine.csv`;
  - `region_tiles_firstorder.csv`, `region_tiles_sample.csv`, `region_tiles_resolve.csv`, `region_tiles_resolve_summary.csv`, `region_tiles_alltiles.csv`;
  - `region_restrict.csv`, `region_restrict_summary.csv`, `region_gpu_scores.csv`;
  - `region_costs.csv`, `region_capetown_faces.csv`.

**Run order** (from the repo root, with `PYTHONPATH=.`, the thread variables at 1 and `PYTHONDONTWRITEBYTECODE=1`):
1. `region_extract`
2. `region_fine`
3. `region_fields 11 100:25 50:10 200:50`
4. `region_twoscale base`, then `region_twoscale clearings 2`
5. `region_tiles firstorder 9`, then `sample 160`, `resolve <w> <tag>`, `alltiles <w> <tag>`, `restrict lin50,lin100,ts50`
6. `region_topup lin50_lin100_ts50`
7. `region_score_gpu <restrict json> <topup json> <out.csv>`
8. `region_analyze`

**In the scratch directory** `/tmp/claude-1641171234/-home-gchurchill-src-reblock/fe8870c0-fbd9-4712-ac98-aebcb951b199/scratchpad/fft_homog_region/`. This is tmpfs, so it is lost on reboot; the CSVs above are copies, and steps 1–3 regenerate the large files in about 15 minutes.
- Fabric and fields: `fabric_region.npz` (11 MB), `fine_u0.npy` (203 MB), `fine_base.npz`, `fine_uh_*.npy`, `fields_L{50,100,200}_s{10,25,50}.npz` (0.82 GB each), `homes_L*_*.npz`.
- Clearing lists: `restrict_clearings_*.json`, `topup_clearings_*.json`, `restrict_tiles_*.json`, `item4_all.json`.
- Logs and `costs.csv`.
- Checks (committed as `region_checks/check_delta_5810.py`, `check_tiles.py` and `check_delta_region.py`; they read the files above).
- Superseded: `gpu_restrict_lin50_firstpass.*.old`, the first item 4 pass, rerun after the ranking's tie-break was made deterministic.
