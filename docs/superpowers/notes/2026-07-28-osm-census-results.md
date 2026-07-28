# Country-wide OSM footpath census: ZAF + KEN

**Date:** 2026-07-28
**Status:** run and complete. `~/.cache/reblock/osm_coverage_{ZAF,KEN}.parquet`, 238,484 rows,
zero `census_failed`.

Reproduce: `pixi run python -m scripts.osm_census --iso ZAF` (then `--iso KEN`). Needs the
Geofabrik extracts in `~/.cache/reblock/osm_pbf/` (417 MB + 349 MB).

## Headline

| | blocks |
|---|---|
| corpus (ZAF + KEN) | 1,813,575 |
| censused after prefilter (k≥3, ≥40 buildings) | 238,484 |
| **qualified** (60–300 buildings, k≥4) | **65,364** |
| qualified **and covered** (≥1 interior footpath at 0.5 m) | **16,497 (25.2%)** |
| …after a density floor of 1,000 buildings/km² | **3,441** |
| …of those, with ≥100 m of interior footpath | **2,408** |

The qualified count landed on **65,364 — the spec's prediction, exactly**, which is a clean
independent check that the prefilter and the band agree with what was computed off the parquet
columns months earlier.

## Coverage is 25.2%, not the 65.5% the spike suggested

The 400-block spike measured 65.5% coverage and said in its own terms that Cape Town was an upper
bound for ZAF+KEN. It was right to, and the gap is large: **25.2% nationally** (ZAF 28.2%,
KEN 22.4%). Cape Town is a well-mapped metro; the corpus is mostly not.

The consequence for the programme's sizing: the earlier "~43,000 validatable recipients" figure was
an extrapolation from the spike's rate, and the measured number is **16,497** before any geometry
guard and **~2,400–3,400** after one. That is still 22–31× the 111-block pool the 2026-07-23 study
was stuck with, and thousands of times its n=1 OSM ground truth, so **the reopening premise holds** —
but the prediction branch has thousands of validation targets, not tens of thousands.

## The building-count band does not bound area, and outside a metro that dominates

This was flagged by the spike as a minor blemish — 5 of 251 covered blocks carrying >5 km of
"interior" footpath. Nationally it is not minor. Among covered qualified blocks:

- median area **1.12 km²**, p90 **98 km²**, max **2,000 km²**
- interior footpath length p90 **9,174 m**, p99 **47 km**, max **427 km**

A 2,000 km² polygon holding 60–300 buildings is not a block in any sense the reblocker cares about;
it is a rural district whose "interior footpaths" are a whole region's paths caught by the clip.
Half the qualified pool sits below 143 buildings/km².

**The guard should be a density floor, not an area cap.** Density is scale-free, so it does not
penalize a genuinely large dense settlement, and it turns out to subsume the length pathology
entirely: `density ≥ 1000` keeps 2,378 ZAF blocks and `density ≥ 1000 AND length/building ≤ 20 m`
keeps 2,375. The absurd-length blocks *are* the low-density blocks — not a separate phenomenon
needing its own threshold.

| floor (buildings/km²) | covered qualified | median area | median interior | p99 interior |
|---|---|---|---|---|
| none | 16,497 | 0.702 km² | 553 m | 37,325 m |
| ≥ 500 | 5,155 | 0.090 km² | 208 m | 2,240 m |
| **≥ 1000** | **3,441** | **0.055 km²** | **181 m** | **1,621 m** |
| ≥ 2000 | 2,052 | 0.033 km² | 185 m | 1,324 m |
| ≥ 3000 | 1,329 | 0.026 km² | 213 m | 1,290 m |

≥1000 is where the p99 stops collapsing (1,621 → 1,324 → 1,290 buys little for a third of the
pool). It is a **pre-screen**, not the screen: the repo's real gate is BFS-peel depth, which needs
building points and therefore runs on the shortlist, not here.

## This shrinks the Open Buildings download 3.1×

Recipients need building points but not OSM, so the provisioning target is qualified ∩ density
floor, regardless of coverage: **20,910 blocks** at ≥1000/km² (ZAF 15,504, KEN 5,406) against the
spec's 65,364. The spec sized 1b's tile download against a pool that is mostly rural polygons.

## Operational findings

**One invalid polygon killed a 40-minute run.** Kenya died after 70,951 of 355,830 blocks on
`TopologyException: side location conflict` inside the clip. `KblockSource._blocks_from` runs
`make_valid` before building a `Block`; the census reaches the same geometry through a different
door and did not. Fixed, plus a per-block `GEOSException` handler — a country run must not be one
polygon away from nothing. A failed block is now recorded as data (`census_failed=True`), because a
zero row is otherwise indistinguishable from a block that genuinely has no footpaths.

**The fix's own first attempt failed the same way.** `make_valid` on a *degenerate* polygon returns
a non-areal geometry, and an empty geometry's `.boundary` is `None` — an `AttributeError`, which
walked straight past the `except GEOSException`. Widening the catch would have been the wrong
instinct; the shape is decidable up front, so `_areal()` now reduces to the polygonal content
before anything touches it. Final run: **zero failures across 238,484 blocks.**

**The atomic checkpoint paid for itself immediately.** Kenya's crash left a valid 70,951-row
parquet, and the resume picked up from it. Under the old non-atomic write, a kill during the write
would have corrupted the file and blocked every subsequent resume.

**Timing.** ZAF ~31 min end to end, dominated by GDAL's PBF ingest (~20 min of it), exactly as
budgeted; per-block work ran at ~1,100 blocks/s. The 5–10 hour estimate was real, and the prefilter
plus single tolerance is what removed it.

## What to do next

1. Apply the density floor as the shortlist definition, and provision Open Buildings for those
   ~21k blocks rather than 65k.
2. Compute `settlement_labels` on that shortlist (decided 2026-07-27: not emitted by the census —
   chaining connected components over a prefiltered two-country corpus yields metro-scale blobs).
3. Run the BFS-peel depth screen on the shortlist; `k_complexity ≥ 4` is kblock's proxy, not the
   repo's gate, and the ~25% coverage rate means the real recipient pool is smaller again.
4. The 0.5/2/5 tolerance sweep and the near-miss tag set were run at one tolerance corpus-wide.
   Both remain worth a sample-sized diagnostic, now that there is a real shortlist to sample from.
