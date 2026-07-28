# GW pair-matrix benchmark: 100 real (recipient, donor) pairs, Cape Town, osm_footpaths donors

**Date:** 2026-07-27
**Status:** committed artifact (`data/benchmarks/gw_pair_matrix.parquet`, 100 rows) + the script that
produced it (`scripts/pair_matrix.py`). This is Task 9 of the OT-retrieval-substrate Phase 1 plan —
the last unit, and the one that turns the 2026-07-23 scratchpad spike
(`docs/superpowers/notes/2026-07-23-ot-road-transplant.md`) into a reusable, re-scoreable benchmark
instead of a one-off finding.

## What this is

For 100 (recipient, donor) pairs of real Cape Town blocks: fit a real entropic Gromov-Wasserstein
correspondence (ε=0.01, τ=1.0 — the note's own ablation found ε=0.05 already collapses the
transported network to ~3% of its proper length, so ε≤0.01 was non-negotiable), transplant the
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

`corr(real_gw_dist, perm_gap)` = **0.006** (Spearman 0.022) across the 100 pairs — indistinguishable
from zero. Binning by `real_gw_dist` quartile:

| GW-distance quartile | n | mean perm_gap | median perm_gap |
|---|---|---|---|
| Q1 (closest) | 25 | -0.189 | -0.164 |
| Q2 | 25 | -0.147 | -0.106 |
| Q3 | 25 | -0.140 | 0.000 |
| Q4 (farthest) | 25 | -0.154 | -0.130 |

**No monotone trend, and no clean zero-crossing as a function of real GW distance** — `perm_gap` is
negative (transplant underperforms length-matched direct clearance) across every quartile, hovering
in a narrow band regardless of how similar the donor is. This directly answers the open question the
prior 2026-07-23 study left hanging (transplant quality on a pool of only 111 blocks, sampled at 3
points): with 100 real pairs spanning the pool's actual GW-distance range, **transplant quality does
not measurably improve as the donor gets more similar.** The weak (r=0.11, still not significant at
n=100) positive trend against the cheap `feature_dist` proxy instead of real GW distance — closer-in-
feature-space donors trending *slightly worse*, not better — is the opposite sign from the naive
expectation and is best read as noise, not a real inverted effect; it does at least confirm the two
distances aren't interchangeable (`corr(real_gw_dist, feature_dist)` = 0.596 — correlated but far from
identical, which is exactly why this matrix records both instead of only the cheap proxy).

Overall `perm_gap` is negative on average (mean -0.157, median -0.124; only 32/100 pairs have
`perm_gap > 0`), consistent with the 2026-07-23 note's single-donor conclusion: direct clearance
usually wins. `perm_direct` averages 0.527 vs. `perm_proposal`'s 0.369; displacement is roughly a
wash (0.085 direct vs. 0.093 transplant). **n=100 is a real sample, not a toy one, and it says
plainly: real-GW-distance-based donor similarity is not, on this evidence, a usable predictor of
transplant fidelity.** This closes the "does it improve with more similar donors" question the prior
study left open — the answer is no, at least for this donor material (real OSM footpaths) and this
recipient/donor pool.

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
  missing.
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
- **No conclusion is asserted beyond what n=100 supports.** The near-zero fidelity/GW-distance
  correlation is a real, moderately-powered result (n=100, not n=3), but it is still one pool
  (Cape Town), one donor-material type (`osm_footpaths`), one exclusion radius (2000m), and one
  metric pairing (`permeability` + `displacement`). `donor_type` is a column specifically so Phase 3
  can extend this same matrix with other donor material rather than re-deriving these caveats from
  scratch.
