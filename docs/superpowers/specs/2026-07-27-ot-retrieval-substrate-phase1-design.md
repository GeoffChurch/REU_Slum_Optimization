# OT retrieval substrate — Phase 1 (design)

**Status: approved in brainstorming 2026-07-27.** Reopens the arc closed in
`notes/2026-07-23-ot-road-transplant.md`, on the grounds that its two decisive limits — a
111-block donor pool and an n=1 OSM ground-truth set — were *pool-size* limits, not mechanism
limits. This spec covers **Phase 1 only**: the shared substrate both downstream branches need.
Phases 2–3 are sketched for context and are **not** specced here.

## Goal

Make the OT/transplant direction measurable at scale, and de-risk the index build before paying
for it. Three independent units:

- **1a** — bulk OSM footpath extraction → a per-block coverage table over ZAF + KEN.
- **1b** — a cheap geometric agreement metric, for the prediction branch.
- **1c** — an experiment measuring how transplant fidelity scales with retrieval distance and
  pool size, plus two near-free re-tests of underpowered prior results.

Permeability and displacement remain the primary scorers throughout. Agreement is secondary.

## Why reopen

The 2026-07-23 note closed OT transplant as a reblocker. Its findings stand. What changed is the
observation that the pool it ran on was ~0.1% of the data already on disk:

| corpus | blocks |
|---|---|
| pool actually used (v3/v4) | 111 |
| Cape Town, 60–300 buildings (the v3 band) | 6,629 |
| Cape Town, all | 83,192 |
| Nairobi, all | 16,200 |
| ZAF country-wide (`ZAF_geodata.parquet`) | 1,457,745 |
| KEN country-wide (`KEN_geodata.parquet`) | 355,830 |

Applying v3's own 37% depth-screen pass rate to the Cape Town band gives ~2,450 qualified blocks —
22× the pool used, same city, zero new data. Two prior results are directly gated by this:

- **v3's rank-1 regime.** Gap correlates −0.90 with feature distance; rank-1 won in 2 of 3
  featurizations; rank-2 collapsed. The note's own verdict is "conditional go only for a single
  nearest-neighbor regime… too narrow to productionize." A larger pool *is* the mechanism that
  widens it, by pulling rank-1 distance down.
- **v5's n=1.** The strongest result in the study (93.7% of a block's own OSM) rests on one
  recipient, solely because donor discovery ran through per-block Overpass and only 40972 had
  committed ground truth.

Neither is a claim that the note was wrong. Both are claims that it was underpowered.

## Program shape

```
Phase 1 (this spec)         Phase 2                  Phase 3
────────────────────────    ────────────────────     ─────────────────────────────────
1a bulk OSM coverage ─────► masked-NCC FFT index ─┬─► 3a prediction (OSM consensus, n≫1)
                                                  │      ▲
1b agreement metric ──────────────────────────────┼──────┘
                                                  │
1c distance/pool measurement ─────────────────────┴─► 3b reblocker (patch quilting,
                                                          policy transplant)
```

Dependency structure: 1a gates everything downstream; 1b and 1c are independent of each other and
of 1a's completion; Phase 2 needs 1a; 3a needs 2 + 1b; 3b needs 2, and 1c informs its shape.

Phase 1 is roughly a week and can redirect Phase 2 before it is paid for.

## 1a — bulk OSM footpath coverage

**Module:** `src/reblock/data/osm_extract.py`, following the `data/provision.py`
`ensure_city_data` cached-download pattern.

**Placement decision.** The reader lands as `PbfDesireLines`, a **second implementation of the
existing `DesireLineSource` Protocol** (`methods/desire_lines.py:22`) — the same pluggable seam
that was always intended to host an imagery detector. `OSMDesireLines` keeps the live-Overpass
path for arbitrary bboxes outside ZAF/KEN. This is a Strategy, not a compatibility shim: two
sources with different operating ranges behind one interface, neither wrapping nor deprecating the
other.

**Pipeline.**

1. Fetch Geofabrik `south-africa-latest.osm.pbf` and `kenya-latest.osm.pbf` into
   `~/.cache/reblock/osm_pbf/`. Never committed, consistent with the existing city-data cache.
2. Read via `pyogrio.read_dataframe(path, layer="lines")` — the GDAL `OSM` driver is already
   available in the pixi env, so **no new dependency**. Filter on `highway` ∈
   `desire_lines._DEFAULT_TAGS` = `("path", "footway", "track", "steps", "pedestrian",
   "living_street")`, **imported, not re-declared**, so the bulk extract and the shipped
   `osm_footpaths` method can never diverge on what counts as a footpath.
3. Spatial-join the linework against the blocks parquet with a tiled STRtree pass.
4. Compute interiority by **calling `osm_footpaths._interior_desire_lines` directly** rather than
   reimplementing it — clip to block, subtract the `STREET_TOL` buffer of `block.streets`, keep
   LineStrings longer than `STREET_TOL`. Same reasoning as the tag tuple: one definition, one
   call site.

**Outputs.**

- `~/.cache/reblock/osm_coverage_{city}.parquet` — `block_id, n_interior_segments,
  interior_length_m, boundary_length_m, n_near_miss_segments, near_miss_length_m`.
- Per-block cached interior linework, so donor use never re-fetches.

**The near-miss columns are deliberate.** Informal-settlement paths are sometimes mapped
`highway=service`, `residential`, or `unclassified`. The near-miss columns count interior segments
carrying exactly those three tags — computed by the same interiority call, over a second tag set,
and never mixed into the primary counts. Reusing `_DEFAULT_TAGS` keeps 1a consistent with the
shipped method, but recording what the filter drops lets us see its cost before deciding whether
to widen it — rather than discovering the ceiling later and having to re-extract.

**Contract:** `osm_coverage(city) -> DataFrame` and `PbfDesireLines.desire_lines(bbox, crs)`.

## 1b — geometric agreement metric

**Module:** `src/reblock/eval/agreement.py`, an `Eval` beside `eval/kcomplexity.py` and
`eval/structure.py`.

Scoped to the **geometric half only**. A functional half (per-parcel egress-cost agreement,
served-set Jaccard) was considered and cut: permeability already measures function, and the
programme's primary scorers are permeability and displacement.

- `buffered_iou(r)` — IoU of `proposal.buffer(r)` and `reference.buffer(r)`, with
  `r = corridor_m = 3.0` to match the permeability corridor.
- Symmetric Chamfer over ~2 m densified samples, reported **directionally**:
  - proposal→reference = precision (paths drawn that aren't there),
  - reference→proposal = recall (real paths missed).

Reported as named fields, never averaged into one number. A single blended score would hide which
way a prediction fails, which is the only thing this metric is for.

**Held-out harness.** Leave-one-block-out over blocks with real interior footpaths (from 1a);
the recipient's own OSM is excluded from its donor pool; results **stratified by
distance-to-nearest-donor**, so the output is a curve over operating regimes rather than one
pooled number.

**Known limitation, accepted.** With the functional half cut and permeability as the headline,
the prediction branch reports a *usefulness* claim ("this network works about as well as the real
one"), not an *accuracy* claim ("this is the real network"). Geometric agreement is the only
check on the latter, and it is secondary by choice.

## 1c — distance/pool-size measurement

Scratchpad experiment. No `src/` changes. Deliverable is a note.

**Step zero — salvage.** The prior OT code survives in another session's scratchpad
(`ot_gw.py`, `transplant.py`, `select_donor.py`, `amortization_test.py`,
`barycenter_amortization.py`, `rsc_*`, plus pickled pools). It is one `/tmp` reclaim from gone.
Copy it into the working scratchpad before anything else. This is what makes 1c a day rather than
a rebuild.

**Primary measurement.** On a pool regrown from the 6,629-block Cape Town band: for each of ~20
recipients spanning the parcel-count range, against ~50 donors spanning the GW-distance range
(≈1,000 pairs — enough that the fit is not the limiting uncertainty, and affordable given v4 ran
6 donors × 3 recipients), compute real (non-linearized) GW distance and the length-matched
permeability gap (transplant − direct clearance). Plot gap vs distance; locate where the curve
crosses zero. Widen if the fit is still ambiguous at that size.

**Pool-size scaling, measured not assumed.** Subsample the pool at sizes 10/30/100/300/1000,
measure rank-1 GW distance at each, and fit the exponent empirically. An N^(−1/d) assumption with
a guessed effective dimension is not good enough to spend Phase 2 on.

**No pre-committed decision rule.** Report the zero-crossing distance and the pool size that
reaches it; decide in discussion. Note that 1c gates only the *reblocker* branch's shape — Phase
2's index is justified for the prediction branch regardless, since bulk retrieval over OSM-covered
blocks is the only way to run v5's method at n≫1.

**Two secondary readouts, near-free on the same pair matrix:**

- **Featurization bake-off, properly powered.** The raw-signature (−0.90) vs heat-trace (+0.60)
  comparison was **n=5 per space**. A rank correlation on five points is uninformative, and
  heat-trace's p=0.285 is explicitly noise — so "heat-trace is worse" is one draw, not a finding.
  The mechanistic argument for the raw signature (it shares the pairwise-distance structure the GW
  cost optimizes) is sound and remains the prior; this just settles it with real power.
- **Pairwise-OT trim re-test.** Falsified, but for a mechanism that is explicitly pool-dependent:
  the trim dropped the rank-1 and rank-2 neighbors *by distance to the recipient*, because they
  were the two most different *from each other*. That decoupling requires a k=6 spanning a wide
  distance range — what a 111-block pool forces. In a dense pool, "far from peers" is more likely
  to mean genuinely anomalous.

## Testing

- **1a** — unit tests for the join and interiority logic against the existing Cape Town sample
  fixtures with synthetic linework; the Geofabrik download is an integration test behind a network
  marker.
- **1b** — synthetic pairs with known answers: identical networks → IoU 1 / Chamfer 0; disjoint →
  IoU 0; and the discriminating case, a reference network **translated 20 m sideways**, which must
  score poorly. That test documents the design intent — this metric is about *location*, and the
  functional reading that would forgive the offset is deliberately not in scope.
- **1c** — no tests. It is an experiment; the note is the artifact.

## Risks

- **Tag coverage.** Mitigated by the excluded-tag counts (above), not by widening the filter
  speculatively.
- **Join scale.** 1.5M ZAF blocks against country-wide linework needs the tiled STRtree pass, not
  a naive join.
- **Env wrinkle.** Invoking `.pixi/envs/default/bin/python` directly raises a PROJ database error;
  go through `pixi run`.
- **Scratchpad loss.** Addressed by making salvage step zero of 1c.

## Noted, not scoped

**Supervised desire-line detection from building geometry.** `notes/2026-07-15-desire-line-detection.md`
closed this at a ~0.5 recall / 0.35 precision ceiling — but all eight approaches were
*unsupervised, hand-tuned* detectors, and the note's own stated cause for the precision ceiling is
"people don't walk every geometric gap," which is exactly what a model learns from labels and a
heuristic cannot. That work had one labeled block; 1a produces thousands. The note's closing line
names the revisit condition itself ("if a trained segmentation model … becomes available"). The
physical resolution-floor argument against **imagery** stands and is untouched by this — the
geometry route only. **Decide after Phase 1**, when the corpus size and label quality are known.

**Phase 2 (sketch).** Building-point density raster (`np.histogram2d` — parcels are Voronoi of
points, so no rasterizer is needed) → tiled **masked** normalized cross-correlation via
`scipy.fft`, sweeping ~24–36 discrete rotations at fixed scale → GW re-rank on the top-k. Masking
is load-bearing and is the one correction to the original "remove roads before building the FFT"
idea: roads cannot be removed from a building-density field by deleting road geometry, because a
road *is* a linear void in that field. Padfield-style masked NCC excludes pixels near known roads
from the correlation and is the raster analogue of the validated GEP. No new dependencies.

**Phase 2 also carries a fix for §7's GEP.** That result was obtained with a whitener estimated
from **45 regions over a 60-dimensional spectrum** — n < d, so the covariance was singular and the
whitening necessarily degenerate. The GEP's +11–55% neighbor street-length gain was achieved
*despite* this. A pool in the thousands fixes it outright.

**Phase 3b candidates (sketch).** Patch quilting — tile the recipient with overlapping windows,
give each its own best-matching donor patch, feather the demand fields, extract once with the
existing `demand_greedy_reblock` on the recipient's own gap graph. This attacks the note's
repeatedly-diagnosed root cause (a transplant reproduces the *donor's* coverage pattern) by
denying any single donor global influence — the local analogue of the barycenter consensus that
was the note's breakthrough. And policy transplant — fit the edge cost function under which a
donor's real footpaths are near-optimal, transplant the *parameters*, solve on the recipient;
ten numbers cannot carry a coverage gap.

## Explicitly not revisited

Shelved for reasons pool size does not touch:

- **§7 street-form donors** — streets are formal boundaries, footpaths are access-optimized.
  Categorical, not a distance problem.
- **Consensus-as-seed for `LoopClosureRefiner`** — the consensus was strong (0.722); *refining*
  it made it worse (0.706), below even an information-free cold seed (0.751). The refiner is the
  defect.
- **Imagery detection** — the resolution floor is physical (a 2 m path ≈ 8 px, a roof seam ≈ 3–4
  px, segmentation error ±2 px).
- **Resistance-as-builder, Voronoi-adjacency partition, distributional radius displacement** — all
  shelved on mechanism or metrics, unrelated to pool size.
