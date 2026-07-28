# GW pair-matrix benchmark: 100 real (recipient, donor) pairs, Cape Town, osm_footpaths donors

**Date:** 2026-07-27
**Status:** committed artifact (`data/benchmarks/gw_pair_matrix.parquet`, 100 rows) + the script that
produced it (`scripts/pair_matrix.py`). This is Task 9 of the OT-retrieval-substrate Phase 1 plan —
the last unit, and the one that turns the 2026-07-23 scratchpad spike
(`docs/superpowers/notes/2026-07-23-ot-road-transplant.md`) into a reusable, re-scoreable benchmark
instead of a one-off finding.

**CORRECTION (same day, fix round 1):** the first version of this note's §1 reported a bare pooled
Pearson correlation (`corr(real_gw_dist, perm_gap)` ≈ 0.006) as "no measurable relationship" and
claimed this "closes" / "contradicts" the prior 2026-07-23 study. **That conclusion was wrong.** The
100 rows are 20 recipients × ~5 donors each — a clustered sample, not 100 independent draws — and
pooling them lets a positive *between-recipient* trend cancel a negative *within-recipient* one
(a textbook Simpson's-paradox mask). A clustering-aware re-analysis (§1 below, code in
`scripts/pair_matrix.py`'s `analyze_fidelity_vs_distance`) finds the opposite of what was first
published: **within a given recipient, a closer donor produces a measurably better transplant**
(fixed-effects slope β=-9.23 on `perm_gap` per unit `real_gw_dist`, p≈0.026 parametric / 0.036
cluster-permutation, robust to jackknife and to dropping the zero-length rows). The effect is
modest, not dramatic, and this dataset is *consistent with*, not contradictory to, the prior
study's direction. The pooled correlation is still reported below — but labeled as the artifact it
is, not the headline. No parquet rows changed; only the analysis of them did.

**CORRECTION 2 (same day, after the POT cross-validation):** `ot_gw._gw_gradient_cost` was missing
the factor of 2 that Peyré–Cuturi–Solomon Prop. 2 carries (and that the paper's own statement
omits — POT ships the same correction with a comment saying so). Because Sinkhorn at `(cost, ε, τ)`
and `(2·cost, 2ε, 2τ)` share an argmin, the solver silently ran at **twice** the requested entropic
regularization: nominal ε=0.01 bought an effective 0.02, outside the ε ≤ 0.01 this note calls
non-negotiable. The parquet has been **regenerated end-to-end** at the corrected calibration and
every number below is from that run. The correction *strengthened* the result — β=-9.58 at p=0.014
(cluster-permutation 0.011), against β=-9.23 at p=0.026 before — and improved the transplants
themselves, with `perm_gap` better on 63 of the 95 rows common to both matrices (Wilcoxon p=0.005).
Full write-up, including the ε ladder showing the estimate converges by the corrected ε=0.01 and
only vanishes at 4× that: `docs/superpowers/notes/2026-07-27-gw-pot-crossvalidation.md`.

## What this is

For 100 (recipient, donor) pairs of real Cape Town blocks: fit a real entropic Gromov-Wasserstein
correspondence (ε=0.01, τ=1.0 in the corrected, POT-standard convention — see CORRECTION 2; the
2026-07-23 note's ablation found ε=0.05 already collapses the transported network to ~3% of its
proper length, so ε≤0.01 was non-negotiable), transplant the
donor's real interior OSM footpaths through it, gap-snap the result onto the recipient's own
`ChordSubstrate`, and score it against a length-matched direct `ClearanceReblocker` solve using the
repo's real `permeability` and `displacement` metrics. `real_gw_dist` (the exact, non-linearized GW
objective of the fitted coupling) is recorded alongside a cheap shape-signature `feature_dist` proxy,
so any future featurization can be checked against real GW distance without re-solving anything.

## RESCOPE: what `load_pools()` actually reads

The plan assumed a census → shortlist → provisioned-building-points chain (Tasks 5+7). That chain
does not exist: Task 5's census needs a 417 MB Geofabrik PBF not on this machine, and Task 7's
provisioning was implemented but never run. `load_pools()` in `scripts/pair_matrix.py` instead reads
`~/.cache/reblock/{blocks,buildings}_capetown_full.parquet` directly:

- Qualified pool: `building_count` in [60, 300] AND `k_complexity` >= 4 → **1,136** Cape Town blocks
  (verified against the parquet on this machine, matching the RESCOPE note exactly).
- Constructed as real `Block`s (`KblockSource`, real building-point join → Voronoi parcels) → **1,109**
  survive (1 dropped: a MultiPolygon dissolve; ~26 dropped by `min_buildings`/Voronoi <4-site
  failures).
- Further restricted to **1,109 blocks with ≥ 50 real parcels** (`select_donor.signature`'s fixed
  subsample size N_SUB=50 — a block below that can't be signed at all, ValueError). This drops none
  beyond the 1,109 already computed above in this pool (the qualified building-count floor of 60
  already implies ≥50 parcels in every case observed).
- Donor material: each donor block's real interior OSM footpaths (`interior_desire_lines`, mirroring
  `OsmFootpathsReblocker`), fetched via `OSMDesireLines` — the on-disk cache first
  (`~/.cache/reblock/osm/`, 181 pre-existing responses from the coverage spike), a live Overpass call
  otherwise, wrapped in retry-with-exponential-backoff (2s/4s/8s, 4 tries) because Overpass is
  genuinely flaky right now (see below).
- Donor eligibility: `reblock.data.settlements.exclusion_holdout` at **radius_m=2000** (the default
  the brief suggested). Measured: at this radius nearly the *entire* pool is eligible for any given
  recipient (1,099–1,107 of 1,109, sampled at three recipients) — Cape Town's qualified blocks are
  spread across the metro, so the exclusion radius barely constrains anything at 2 km. This means
  donor **selection** (not eligibility) is what actually shapes the matrix's distance spread.
- A cheap, GW-consistent shape signature (`select_donor.signature` — bootstrap-averaged sorted
  eigenvalues of each block's normalized parcel-centroid pairwise-distance matrix) is precomputed
  once for the whole 1,109-block pool (1.7s) and used **only** to stratify candidate donors by
  proxy-distance to each recipient (near/mid/far), so the expensive real GW fit lands on a spread of
  similarity instead of an arbitrary sample. It is never written to the parquet in place of
  `real_gw_dist`.

Recipients (20 of them) were chosen spanning the pool's parcel-count range (min→max by rank, not
randomly) — deliberately including the pool's smallest and largest blocks. Donors (5 per recipient)
were chosen by evenly-spaced rank in proxy-signature distance among eligible candidates, backfilling
around OSM skips from a candidate list 3x the target size.

This deviates from the brief's literal `load_pools() -> (recipients, donors, donor_lines,
blocks_gdf)` 4-tuple, in which `donors`/`donor_lines` are indexed statically alongside `blocks_gdf`
— that shape implicitly requires prefetching OSM material for the *entire* pool eligible as donors
(up to 1,109 blocks) before scoring a single pair, which is neither affordable (Overpass fair-use)
nor safe (the flakiness hazard below) to do eagerly. `load_pools()` instead returns
`(blocks, blocks_gdf, signatures)` and donor material is fetched lazily, memoized per donor
`block_id`, only for candidates actually selected.

## THE TIMING PILOT (the gate) — per-stage breakdown and per-pair wall clock

`pixi run python -u -m scripts.pair_matrix --pairs 20 --timing-only`, run to completion in the
foreground (see "Two operational surprises" below for why foreground, not background):

```
20 pairs in 357s -- 17.9s/pair
  osm_fetch        310.1s  (87%)
  gw                40.4s  (11%)
  transplant         1.4s  (0%)
  clearance          2.2s  (1%)
  permeability       2.8s  (1%)
skips: {'empty_interior': 14, 'fetch_failed': 1}
```

**Headline: 17.9 s/pair, comfortably under the 30 s/pair "run the full ~100" threshold.** The
dominant cost is not the GW/transplant/clearance/permeability pipeline itself (which totals under
1.5 s/pair combined) — it's Overpass retry/backoff overhead from a genuinely flaky endpoint (504
Gateway Timeout and 429 Too Many Requests, both observed live during the pilot). The real GW fit
(entropic, ε=0.01, τ=1.0, 30 outer × 100 inner Sinkhorn iterations) is fast even at this pool's
upper end: an isolated 995×995-parcel synthetic pair (the single largest block in the qualified
pool) fit in 23.6s stand-alone; the pilot's 40.4s GW total across 20 real pairs (mostly much
smaller) confirms this is not the bottleneck.

Per the task's sizing rule (under ~30s/pair → run the full ~100 pairs), **the full matrix was run.**

## Two operational surprises (worth recording so the next run doesn't repeat them)

1. **Module invocation, not file invocation.** `python scripts/pair_matrix.py` fails at import
   (`ModuleNotFoundError: No module named 'scripts'`) because `reblock.data.provision` imports
   `from scripts.fetch_kblock_fixtures import ...`, and running a file directly puts only
   `scripts/` (not the repo root) on `sys.path`. The existing repo convention
   (`scripts/fetch_desire_lines_snapshot.py`'s docstring) is module form:
   `pixi run python -m scripts.pair_matrix ...`. The script's docstring and every invocation in
   this note use that form.
2. **`run_in_background` did not survive in this environment.** A first attempt to run the 20-pair
   pilot via a backgrounded Bash call was silently killed with no output beyond the two
   module-load lines and no completion notification — confirmed dead (no process, no log growth,
   no output file) after the fact. The fix was operational, not code: run everything in the
   **foreground**, unbuffered (`python -u`), under the tool's own timeout. Because the full
   ~100-pair run exceeds any single foreground call's time budget, `scripts/pair_matrix.py` gained
   an incremental-checkpoint / resume capability as a direct consequence: every successfully scored
   row is written to `--out` immediately (not just at the end), and re-running the same command
   against an existing `--out` skips `(recipient, donor)` pairs already on disk and continues. The
   full matrix was produced as five sequential foreground invocations of the same command, each
   safely interruptible.

## The full ~100-pair matrix

```
pixi run python -u -m scripts.pair_matrix --pairs 100 --out data/benchmarks/gw_pair_matrix.parquet
```
run 5 times in sequence (each resuming from the last), producing 100 rows total in a combined
**2,395s of wall clock (five ~9-minute foreground chunks) → 23.95 s/pair overall**, close to and
consistent with the pilot's 17.9 s/pair (the small increase reflects Overpass rate-limiting
building up over a longer session, plus a couple of the pool's largest blocks landing late in the
run). No timing threshold was crossed; nothing was cut short by expense — all 100 requested pairs
were produced.

**Donor skips (un-fetchable/unusable OSM):** across the four chunks whose logs were captured to a
file (74 of the 100 rows' worth of donor search — the very first chunk's terminal output was lost to
a pipe-buffering artifact when it was killed by the `timeout` wrapper, though its 26 rows are intact
and correct on disk): **188 `empty_interior`** skips (fetched fine, but zero interior footpath
material once perimeter-retracing streets are subtracted — this is the dominant skip reason by far,
consistent with the 2026-07-23 note's observation that many Cape Town candidate blocks have thin or
no OSM interior coverage) + **9 `fetch_failed`** skips (exhausted 4 retries against repeated
504/429s) + **0 scoring errors**. None of these are in the parquet — every row is a pair that was
actually scored end-to-end; skips are search overhead, not degraded rows. (The parquet contains 4
rows with `road_len_m == 0.0` — real, computed rows where the donor's interior OSM material, once
warped and gap-snapped, collapsed to nothing usable; these are legitimate data points, not skips —
see "Concerns" below.)

## The three measurements

### 1. Fidelity vs. GW distance

**The 100 rows are clustered (20 recipients × ~5 donors each), not 100 independent draws, and the
naive pooled correlation is unsafe to read at face value.** Recipients differ systematically in
both their achievable GW-distance range and their baseline `perm_gap` level; pooling conflates
"does a closer donor transplant better for a given recipient" with "do recipients whose donors
happen to sit at larger GW distances also happen to have larger `perm_gap` for other reasons." This
is a textbook Simpson's-paradox setup, and it is exactly what happened here. All of the numbers
below are reproducible via `pixi run python -m scripts.pair_matrix --analyze --out
data/benchmarks/gw_pair_matrix.parquet` (`analyze_fidelity_vs_distance` in `scripts/pair_matrix.py`)
— nothing here required a new GW fit or touched the parquet's rows.

**Step 1 — how clustered is it?** `ICC(1)` of `perm_gap` grouped by recipient (Fisher's
unbalanced-design formula) = **0.3288**; the raw ANOVA R² / η² (SSB/SST, uncorrected for
within-group noise — a related but distinct "how much variance is between-group" statistic, easy
to conflate with ICC(1) under the same informal label) = **0.4503**. Either way: a substantial
fraction of `perm_gap`'s variance sits between recipients rather than being donor-driven noise —
enough to make pooling across recipients unsafe.

**Step 2 — the pooled (wrong) headline.** `corr(real_gw_dist, perm_gap)` over all 100 rows =
**-0.0496** — indistinguishable from zero. *This is the number the first version of this note
reported as "no measurable relationship." It is an artifact of the clustering above, not evidence
of no effect.*

**Step 3 — the within-recipient (correct) estimate.** Demean both `real_gw_dist` and `perm_gap` by
recipient (the fixed-effects / "correct for cluster" transform) and regress:

```
within-recipient beta (perm_gap ~ real_gw_dist) = -9.5817
SE = 3.8151, t = -2.5115, dof = 79 (= 100 rows - 20 recipients - 1)
p (t-distribution)      = 0.0141
p (cluster permutation, 5000 within-recipient shuffles) = 0.0110
```

The estimator itself was later cross-checked against `statsmodels` OLS with recipient dummies,
which reproduces β, SE, t, dof and p exactly. Cluster-robust standard errors — the more defensible
choice for 20 clusters — widen the SE somewhat without changing the verdict.

**β is negative and significant: within a given recipient, a smaller `real_gw_dist` (closer donor)
is associated with a larger `perm_gap` (better transplant fidelity relative to direct clearance).**
The permutation test (shuffle `real_gw_dist` *within* each recipient's own rows only, preserving
every recipient's own value set and every recipient's own `perm_gap` values, 5000 draws) confirms
this isn't a t-distribution artifact of a small cluster count: p=0.011, consistent with the
parametric p=0.014.

**Step 4 — the cancelling counterpart.** The recipient-level aggregate correlation (one point per
recipient — mean `real_gw_dist` vs. mean `perm_gap`, n=20) = **+0.1900**. This is the *positive*
between-recipient trend that, pooled together with the negative within-recipient rows, washes the
pooled correlation down to ~0 (Step 2). Both trends are real; they answer different questions
("does a recipient with, on average, farther-away sampled donors also tend to have higher
`perm_gap` for other reasons" vs. "for one recipient, does its closer donor beat its farther
donor") and pooling them is the mistake, not either number individually.

**Robustness.** Excluding the zero-road-length rows: β=-9.5933 (p=0.0147). Leave-one-recipient-out
jackknife: β stays in **[-13.02, -7.73]** across all 20 holdouts — always negative, never close to
flipping sign.

**Range restriction (why the effect looks modest, not why it might not exist).**
`real_gw_dist` spans only **0.0064 to 0.0443** (mean 0.0184, sd 0.0067) — a 6.96x max/min ratio.
The `feature_dist` proxy used to *stratify* donor selection spans a much wider 28.50x ratio
(0.182 to 5.186) on the identical 100 pairs, and the two are only moderately correlated
(r=0.6301). This means stratifying evenly across `feature_dist` rank did **not** transfer
proportionally into a correspondingly wide `real_gw_dist` spread — the achieved range likely
undersamples the true achievable range in the qualified pool. Range restriction of this kind
attenuates a detectable correlation/slope's *magnitude* (a wider range would very plausibly
sharpen this signal further), but it is not a reason to doubt that the signal *exists* — the
within-recipient effect is already detectable, at conventional significance, inside this
restricted range.

**Bottom line:** the corrected finding is the *opposite* of what the first version of this note
claimed. Within a recipient, a closer (lower real-GW-distance) donor does transplant measurably
better — modestly, not dramatically, and consistent with (not contradicting) the 2026-07-23 prior
study's direction. This does **not** "close" the open question from that study in the sense of
settling it decisively (n=20 recipients is a real but not enormous cluster count, and the achieved
`real_gw_dist` range is restricted, per above) — but it is the first evidence, at n=100 pairs
across 20 recipients rather than the prior study's handful of manual comparisons, that donor
similarity is a genuine (if modest) predictor of transplant fidelity, not merely noise.

Overall `perm_gap` is still negative on average (mean -0.157, median -0.124; only 32/100 pairs have
`perm_gap > 0`) — direct clearance usually wins outright, consistent with the 2026-07-23 note's
single-donor conclusion. `perm_direct` averages 0.527 vs. `perm_proposal`'s 0.369; displacement is
roughly a wash (0.085 direct vs. 0.093 transplant). The within-recipient finding is about the
*slope* of fidelity vs. distance, not about transplant beating clearance outright — those are
different questions, and this dataset speaks only to the first.

### 2. Pool-size → rank-1-distance exponent

Measured (not assumed from a theoretical N^(-1/d)) by resampling the whole pool's cheap signature
vectors: for pool sizes N ∈ {10, 30, 100, 300, 1000}, 200 trials each of (random held-out recipient,
random N-block donor pool), recording the nearest neighbour's proxy-signature distance:

| pool size N | median rank-1 distance |
|---|---|
| 10 | 1.075 |
| 30 | 0.683 |
| 100 | 0.490 |
| 300 | 0.385 |
| 1000 | 0.286 |

Fitted exponent (slope of `log(median rank1_dist) ~ log(N)`): **-0.28**. A naive isometric
low-dimensional embedding guess (`N^(-1/d)` for, say, d=2 → exponent -0.5) is noticeably steeper than
what's actually observed — the measured -0.28 implies an effective dimensionality around 1/0.28 ≈
3.6, i.e., **growing the donor pool gives diminishing returns on donor similarity more slowly than a
low-dimensional shape-space picture would predict**, but still clearly diminishing (not flat) as N
grows across three orders of magnitude. This measurement uses only the cheap signature proxy (never
requires a real GW fit), reproducible via
`pixi run python -m scripts.pair_matrix --rank1-scaling "10,30,100,300,1000"` — a genuinely free
byproduct of `load_pools()`'s existing pool-wide signature precomputation.

### 3. Per-pair wall clock

Covered above (headline: 17.9s/pair pilot, 23.95s/pair over the full 100-pair run). Restated once
more for this section's sake: **OSM fetch/retry overhead (Overpass flakiness), not the
GW+transplant+clearance+permeability pipeline, is the actual cost driver** — per-pair scoring cost
alone (`wall_clock_s` column, i.e., everything *except* donor-material fetch) averages 1.04s
(min 0.23s, max 6.83s, the max belonging to the pool's largest recipient block).

## Full-suite result

`pixi run pytest -m "not network"` → **479 passed** (matches the stated baseline exactly; this task
added no `src/`/`tests/` changes, only `scripts/pair_matrix.py` and this note + the parquet).

## Files

- `scripts/pair_matrix.py` (new) — the benchmark driver; module-invoked
  (`pixi run python -m scripts.pair_matrix ...`), fails clearly naming this note's §"Two operational
  surprises" / `docs/superpowers/notes/2026-07-23-ot-road-transplant.md` §1 if `scratchpad/ot/` is
  missing. Also holds the clustering-aware analysis (`--analyze`) and the pool-size scaling
  measurement (`--rank1-scaling`) used to produce every number in this note, and (going forward)
  checkpoints skip counts to a `<out-stem>.skips.json` sidecar the same way rows are checkpointed.
- `data/benchmarks/gw_pair_matrix.parquet` (new, committed) — 100 rows, columns `recipient, donor,
  donor_type, real_gw_dist, feature_dist, perm_gap, perm_proposal, perm_direct,
  displacement_proposal, displacement_direct, road_len_m, wall_clock_s` (the last three are extras
  beyond the brief's listed columns — informative, not a schema break).
- This note.

## Concerns / self-review

- **The 4 zero-length rows** (`road_len_m == 0.0`): the donor's interior OSM material, once IDW-
  warped through the GW coupling and gap-snapped onto the recipient's substrate, collapsed to fewer
  than 2 distinct points and was dropped by `gap_snap`. These are real, computed outcomes (a
  legitimate way for a transplant to fail outright) and are kept in the parquet rather than silently
  filtered — but a consumer computing summary statistics should be aware `perm_gap` for these rows is
  mechanically ~0 (both proposal and length-matched-direct collapse to the same empty-roads
  baseline), which is a genuine data point (the transplant produced *nothing*), not a NaN/error to be
  imputed.
- **Chunk 1's skip counts are not recoverable.** Its 26 rows are correct and complete (verified: no
  `NaN`s, no duplicate `(recipient, donor)` pairs anywhere in the final 100-row file), but the
  donor-skip tally for that specific chunk was lost to a terminal-output artifact when its
  `timeout`-imposed kill interacted with a `| tail` pipe. The reported skip counts (188/9/0) cover
  74 of the 100 rows' search and are presented as such, not inflated to imply full coverage.
- **`feature_dist` and `real_gw_dist` are only moderately correlated (r=0.596).** This means donor
  *selection* (stratified by the cheap proxy) does not perfectly reproduce a stratification by real
  GW distance — the achieved `real_gw_dist` spread (0.011–0.051) is real but narrower/differently
  shaped than a hypothetical real-GW-distance-native stratification would give. This is exactly why
  the matrix records both columns rather than only the proxy.
- **`donor_cache` is per-process, not cross-chunk.** Because the full run was split into five
  separate process invocations (the operational fix above), a donor found `empty_interior` in one
  chunk is not remembered as such in the next chunk if a different recipient's candidate list picks
  it again — it gets re-fetched (cache-hit, since the on-disk OSM cache persists) and re-skipped.
  This does not affect correctness (each skip is a genuine, freshly-evaluated candidate-slot
  decision for that recipient) but means the raw skip counts are not deduplicated by donor identity.
- **No conclusion is asserted beyond what n=100 (20 clusters) supports.** The within-recipient
  effect (§1) is real and significant by two independent tests, but it is still one pool (Cape
  Town), one donor-material type (`osm_footpaths`), one exclusion radius (2000m), one metric
  pairing (`permeability` + `displacement`), and a restricted `real_gw_dist` range (6.96x max/min).
  `donor_type` is a column specifically so Phase 3 can extend this same matrix with other donor
  material rather than re-deriving these caveats from scratch.
- **Two statistics both get called "ICC" and disagree (0.3288 vs 0.4503) — this is a labeling
  ambiguity, not a bug.** `icc_one_way` computes the standard unbalanced-design-corrected Fisher
  ICC(1) (0.3288); `variance_explained_by_recipient` computes the raw one-way-ANOVA R²/η² (SSB/SST,
  0.4503) that some sources call "ICC" more loosely. The corrected estimator is smaller because it
  accounts for the fact that even pure within-group noise produces *some* apparent between-group
  dispersion in a finite sample of small (mostly n=5) groups; the raw ratio does not. Both point the
  same direction (substantial clustering) and both are reported (`scripts/pair_matrix.py`'s
  `_print_analysis`) so a future reader isn't stuck reconciling two numbers with no explanation.
- **Skip counts now persist across resumed invocations.** The original run lost chunk 1's skip
  tally because `skip_counts` was process-local and only ever printed, never checkpointed --
  exactly the failure mode the row-level checkpoint/resume was built to avoid, just applied to the
  wrong variable. `scripts/pair_matrix.py` now mirrors the parquet's checkpoint-every-write pattern
  for skips too: a `<out-stem>.skips.json` sidecar is loaded at start (if present) and rewritten
  after every single skip, so a kill mid-run loses at most the skip in flight, and a later resumed
  invocation's tally is added to the prior total rather than starting over. Verified end-to-end
  against a scratch output path (not the committed parquet): a first invocation recorded 3 skips
  into a fresh sidecar; a second, resumed invocation against the same `--out` re-attempted (and,
  since `donor_cache` is still per-process, re-skipped) the same 3 donors, and the sidecar
  correctly accumulated to 6 rather than resetting to 3.
