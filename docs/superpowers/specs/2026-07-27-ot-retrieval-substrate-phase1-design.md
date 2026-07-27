# OT retrieval substrate — Phase 1 (design)

**Status: approved in brainstorming 2026-07-27; revised the same day after subagent review.**
Reopens the arc closed in `notes/2026-07-23-ot-road-transplant.md`, on the grounds that its two
decisive limits — a 111-block donor pool and an n=1 OSM ground-truth set — were *corpus* limits,
not mechanism limits. Covers **Phase 1 only**; Phases 2–3 are sketched for context and are not
specced here.

The first draft of this spec was materially wrong about *which* corpus limit binds, and the
revision is structural rather than cosmetic. See §"What the review changed".

## Goal

Make the OT/transplant direction measurable at scale, and establish which corpus tier gates
which downstream question. Four units:

- **1a** — country-wide OSM footpath **census** over all 1.8M ZAF+KEN blocks. Free of building
  data; sizes the prediction branch.
- **1b** — **targeted** Open Buildings provisioning for the blocks the census says matter.
- **1c** — geometric agreement primitives plus a leakage-resistant holdout protocol.
- **1d** — a small GW pair-matrix pilot, with a timing pilot first, committed as a reusable
  artifact.

Permeability and displacement remain the primary scorers. Agreement is secondary.

## The corpus is three tiers, not one

This is the central correction. Different jobs need different data, and conflating them is what
produced both the first draft's over-claim and the review's under-claim.

| tier | needs | usable size | status |
|---|---|---|---|
| **T1 · OSM census** | blocks parquet + OSM linework | **all 1,813,575** | free of building data, behind one refactor |
| **T2 · Retrieval / index** | building **points**; Voronoi/peel computed in memory for the screen, never stored | ≤65k qualified (pre-screen) | after a targeted tile download |
| **T3 · Scoring / solving** | points + persisted parcels + substrate | grows with T2 | the genuinely expensive tier |

**T2's "no parcels" is a claim about storage, not computation.** The BFS-peel depth screen is the
real qualifying gate, and `screen/dense_compact.py:_chunk_depths` builds full `Block`s through
`KblockSource.region()` — Voronoi included — before calling `access_before`. Two of the note's
morphological features (compactness, access depth) need the same pass. Measured at 135 ms/block,
that is ~2.45 single-core hours for 65,364 blocks: cheap, but real, and it means **65,364 is a
pre-screen upper bound**, not the qualified pool.

**T1 is free because kblock `streets` *is* the block boundary.** `_interior_desire_lines`
currently takes a `Block` (`osm_footpaths.py:23`), and `Block.__post_init__` raises
`ValueError("Block.parcels must be non-empty")` (`contracts.py:54-56`) — parcels being Voronoi of
building points, which exist only for the Cape Town and Nairobi bboxes. That is a **code-structure
limit masquerading as a data limit**: refactored to `(lines, boundary, streets, crs)`, the census
runs country-wide with zero building downloads.

**T2 does not need parcels *stored*.** Parcels are `Voronoi(building_points)` clipped to the block
— a deterministic function of the points plus the block polygon — so the points carry the
information and a descriptor can be built from them directly, avoiding a boundary-clipping
artifact.

Measured on a random draw of 12 blocks from the **actual qualified pool** (Cape Town,
`building_count` 60–300, `k_complexity ≥ 4`, n=1,136), parcel centroid → nearest building point:

| statistic | value |
|---|---|
| median-of-medians | 2.13 m |
| **median / block diagonal** | **0.31 – 1.26%** |
| per-block p90 | 2.19 – 14.19 m |

An independent draw by a reviewer gave a median-of-medians of 3.21 m over the same relative range
(0.42–1.18%), with per-block p90 reaching 402.8 m on a 5.1 km-diagonal block. **The absolute
displacement is draw-dependent because it tracks block size; the ratio is stable.** The claim this
tier rests on is therefore the *relative* one — the substitution perturbs a normalized
pairwise-distance descriptor by ~1% of block scale — and not any comparison against the note's
6.24 m GW recovery error, which is an absolute figure on one small block and does not transfer.

An earlier version of this table was measured on `blocks_capetown_sample.parquet`, a fixture built
with `MAX_AREA_KM2 = 0.5` and a density floor — small compact blocks by construction, not the
target population. The numbers above replace it.

**Cardinality caveat.** Clipped Voronoi explodes multi-lobe cells, so parcels and points do not
agree on n (observed 205/191 and 92/87 in the draw above). Any descriptor must be defined so this
does not matter — a normalized distance spectrum over whichever set is used, not a per-element
correspondence.

**One honest limit on T1.** A block with OSM linework but no building points can be *counted*, but
cannot be GW-coupled to transport its roads — coupling needs both point clouds. T1 sizes the
prediction branch; it does not feed it.

## Why reopen, restated correctly

Qualified-pool size, measured from columns already on disk (no download):

| corpus | blocks | band 60–300 | k≥4 | k≥5 |
|---|---|---|---|---|
| ZAF (`ZAF_geodata.parquet`) | 1,457,745 | 104,533 | 32,320 | 8,896 |
| KEN (`KEN_geodata.parquet`) | 355,830 | 54,968 | 33,044 | 11,940 |
| **total** | **1,813,575** | **159,501** | **65,364** | **20,836** |
| *pool actually used (v3/v4)* | | | *111* | |

**~65,000 qualified blocks against 111 used — ~590×.** Gated on an Open Buildings tile download
that is already 90% implemented (public S2 level-4 tiles with a `tiles.geojson` index,
`fetch_kblock_fixtures.py:62-67`; the current code takes only the tile containing the bbox
centroid, `:216-223`). Measured: **20 tiles cover ZAF+KEN, 3.78 GB gzipped** (see 1b).

**`k_complexity` is a proxy, not the screen.** It is kblock's own metric, and the backlog records
peel-k ≈ √(building count) on Voronoi — a count proxy, not morphology. The repo's BFS-peel depth
screen is the real gate and must be run on the shortlist. Treat 65,364 as an upper bound on the
qualified pool, not a measurement of it. For reference, in Cape Town the band is 6,629 with 1,136
(17.1%) at k≥4; the first draft's "37% depth-screen pass rate" came from v3's anchor-conditional
300-block sample and was an upper bound.

Two prior results are gated by pool size specifically:

- **v3's rank-1 regime.** Gap correlates −0.90 with feature distance *in the spectral space*
  (morphological and pairwise-histogram were −0.70); rank-1 won in 2 of 3 featurizations; rank-2
  collapsed. A larger pool is the mechanism that widens that regime.
- **v5's n=1.** The strongest result in the study (93.7% of a block's own OSM) rests on one
  recipient, because donor discovery ran through per-block Overpass.

**What a larger pool does *not* fix.** The note's mechanistic diagnosis — a transplant reproduces
the donor's *coverage pattern*, not the recipient's needs — is about OSM **mapping** artifacts
(v2's donor had an unmapped flank), which are uncorrelated with block shape. Shrinking rank-1
shape distance does not repair an unmapped flank. A larger pool helps that diagnosis only
indirectly, by making it affordable to *filter to well-mapped donors*. The direct fix is averaging
over donors, which the note already found (barycenter consensus) and which Phase 3b's patch
quilting re-derives locally. **The reopening is strong for the prediction branch and weak for the
reblocker branch**, and Phase 1 is budgeted accordingly.

## Measured: coverage is not the constraint (spike, 2026-07-27)

Question 1 below was the one that could kill the programme, so it was measured before planning.
A 400-block random sample of the qualified Cape Town pool (`building_count` 60–300,
`k_complexity ≥ 4`), interiority computed straight from the country blocks parquet — **no
`KblockSource`, no Voronoi, no building points**, which is the T1 path executed rather than argued:

| interiority tolerance | ≥1 interior segment | ≥100 m | ≥300 m | median length (covered) |
|---|---|---|---|---|
| 0.5 m | **65.5%** | 55.4% | 37.1% | 356 m |
| 2 m | 65.3% | 53.8% | 36.3% | 341 m |
| 5 m | 62.9% | 46.7% | 26.9% | 226 m |

n=383 measured; 6 tiles / 17 blocks lost to Overpass 504s after 5 retries each, so rates are over
measured blocks.

**Implication: the prediction branch has tens of thousands of validatable recipients, not one.**
Against ~65k pre-screen qualified blocks, 65.5% implies ≈43,000 with some interior coverage and
≈36,000 with ≥100 m. That is the n≫1 Phase 3a needs.

**This is better than the note's 8-of-15-had-zero (≈47%)**, and the discrepancy is explicable:
those 15 were *similarity-ranked neighbours of one block*, a narrow non-random selection. A random
draw from the qualified pool does materially better.

**The tolerance sweep overturned its own hypothesis.** `STREET_TOL = 0.5 m` was expected to
inflate the coverage gate. It does not: the count gate moves only 2.6 points across 0.5→5 m, while
total interior *length* drops 17.8% (207.6 → 170.7 km). This generalizes the single-block
observation in 1a — **the tolerance choice matters for donor-quality ranking, not for the census
gate.** Keep the sweep, but coverage counts can be read at any tolerance in this range.

**New finding — the qualified filter needs an area guard.** 5 of 251 covered blocks carry >5 km of
"interior" footpath on 90–293 buildings (max 26.5 km on 258 buildings ≈ 100 m of path per
building), against a median of 356 m. These are geometrically huge block polygons where the clip
captures a whole neighbourhood's network rather than one settlement's. A `building_count` band
does not bound block *area*; 1a must add an area or density guard, and should report what it
excludes.

**Caveat:** Cape Town only. OSM footpath coverage outside major metros will be lower, so 65.5% is
an upper bound for ZAF+KEN as a whole. The full census measures this properly.

## Success criteria

Phase 1 answers three questions and produces four artifacts. It is *not* trying to beat anything.

1. How many blocks have genuine interior OSM footpaths, and how does that number move with the
   interiority tolerance? → sizes the prediction branch, or kills it.
2. Is the transplant-fidelity-vs-GW-distance relationship strong enough to be worth a bigger
   index? → informs, does not gate, Phase 2.
3. What does a pair cost, in wall-clock? → makes any later scale claim budgetable.

The Phase 3a deliverable this serves is a **mapping/data product** — predicted footpaths for
blocks OSM has not mapped — not a reblocker. The ceiling argument: `osm_footpaths`' own *real*
network reaches comparable permeability to a clearance solve at materially lower displacement
(v5: 0.7702 @ 28.4% vs 0.7830 @ 43.9%; v2: 0.8017 @ 28.4% vs 0.9094 @ 69.3%). On this repo's
two-axis basis the real network is Pareto-non-dominated in v5 and not dominated on both axes in
v2 — so a predicted network inherits a ceiling that is *attractive on cost and unremarkable on
benefit*, which supports the mapping-product framing without claiming the real network loses.

## 1a — country-wide OSM footpath census

**Refactor first.** `_interior_desire_lines(lines, block)` → `_interior_desire_lines(lines,
boundary, streets, crs)`, with `osm_footpaths` passing block fields at its one call site. One
definition, one implementation, and the census no longer needs a constructible `Block`. This is
the change that unlocks T1.

**Module:** `src/reblock/data/osm_extract.py`, following `data/provision.py`'s cached-download
pattern.

**`PbfDesireLines`** lands as a second implementation of the existing `DesireLineSource` Protocol
(`desire_lines.py:22`) — the seam whose docstring already anticipates multiple sources. At 1.8M
blocks a bulk extract genuinely earns itself; a per-block or tiled-bbox Overpass sweep does not
scale here. Required specifics the first draft omitted:

- **`tags` is a field defaulting to `_DEFAULT_TAGS`**, mirroring `OSMDesireLines`. Note that
  `conf/desire_source/osm.yaml` **re-declares** the tag list, so the shipped method's effective
  tags come from Hydra, not the Python default. Both configs must interpolate one shared list, or
  the "cannot diverge" property is fiction.
- **`identity`** is `(pbf_sha256, tuple(tags))` — stable, unlike `OSMDesireLines`' `None` when
  live. This flips `osm_footpaths` from uncacheable to cacheable and therefore changes
  `proposal_id`s. The six committed `examples/*/desire_lines_*.geojson` snapshots and their
  regenerated outputs must be checked before this lands.
- **Default source for ZAF/KEN after Phase 1 is `PbfDesireLines`**; `OSMDesireLines` remains for
  bboxes outside the extracts. The two *will* disagree on the same bbox (Geofabrik extract
  timestamp vs live Overpass; GDAL `lines` layer vs Overpass `out geom`), so a pinned-bbox test
  asserting agreement within tolerance is part of this unit. Without that, two sources is
  accommodation rather than a Strategy.
- **Read with OGR-side filtering** — `pyogrio.read_dataframe(path, layer="lines", where="highway
  IN (...)", use_arrow=True)`, verified working at 0.1 s on a real `.osm.pbf`. This reduces what
  materializes in Python; it does **not** avoid the GDAL `OSM` driver's multi-GB temp SQLite build,
  which happens regardless. Budget disk for both. Geofabrik inputs are 417 MB (ZAF) + 349 MB (KEN).
- **Load once per batch, not once per block.** `DesireLineSource.desire_lines(bbox, crs)` is a
  per-bbox API; the census must not call it 1.8M times. Read and index the country layer once per
  UTM batch and query the STRtree per block.
- **Stream the blocks parquet.** `ZAF_geodata.parquet` (833 MB) and `KEN_geodata.parquet` (386 MB)
  are single-row-group, so `gpd.read_parquet` will not stream a column — use `iter_batches`.

**Interiority is reported as a tolerance sweep, not a number.** `_interior_desire_lines` subtracts
`streets.buffer(STREET_TOL)` with `STREET_TOL = 0.5 m`, and for kblock `streets` is the block
outline. OSM ways are digitized against different imagery than the Open Buildings / kblock
outlines, so a genuinely boundary-running path more than 0.5 m off the outline is recorded as
*interior* — inflating the exact column that gates "which blocks have real coverage." Emit
interiority at **0.5 / 2 / 5 m** and decide after looking. Same instinct as the near-miss tags
below, applied to the thing that actually matters.

**Report segment count and length at every tolerance, not just length.** On block 40972 the sweep
gives 639.3 / 626.5 / 600.7 m while the segment count stays at **13 throughout** — the tolerance
trims the boundary-adjacent ends of paths without eliminating any. Since coverage is gated on
*count* (does this block have any interior footpath?) and donor quality on *length*, a
length-only sweep would misrepresent how much the tolerance choice actually moves the gate.

**Outputs** (`~/.cache/reblock/osm_coverage_{iso}.parquet`):
`block_id, settlement_id, n_interior_segments_{0.5,2,5}, interior_length_m_{0.5,2,5},
boundary_length_m, n_near_miss_segments, near_miss_length_m`.

Near-miss columns count interior segments tagged `service`/`residential`/`unclassified` — the
tags informal paths are sometimes mapped under — computed by the same interiority call over a
second tag set, never mixed into the primary counts. Recording what the filter drops lets us price
widening it before re-extracting.

`settlement_id` comes from a spatial clustering of qualified blocks (connected components under a
distance threshold). It exists for 1c's holdout and must be produced here, where the geometry is
already in hand.

## 1b — targeted Open Buildings provisioning

Extend `provision.py` from single-tile to multi-tile: enumerate `tiles.geojson` features
intersecting the **shortlist** — blocks passing the `k_complexity` cut *and* carrying real interior
coverage from 1a — download each, filter to `OB_MIN_CONFIDENCE = 0.7`, and **spatially join
against the shortlist's block polygons**.

The polygon join is the load-bearing part. `download_capetown_buildings` currently filters with a
lon/lat rectangle (`between()`), and a *rectangle* around a ZAF+KEN shortlist is both countries —
which would retain essentially every Open Buildings row and defeat the point of being
query-driven. It is the filter that must change, not the tile loop.

**Measured, not estimated:** `tiles.geojson` has 333 features, of which **20 cover ZAF+KEN and 19
cover the k≥4 band**. HEAD requests on the `points_s2_level_4_gzip` URLs total **3.78 GB**
gzipped (the polygon variants total 14.09 GB). Budget hours of streaming CSV parse, not minutes.

Query-driven, not country-wide: the census tells us which blocks matter before we pay for any of
them. This is the only genuinely new download in Phase 1.

## 1c — agreement primitives and holdout protocol

**Plain functions in `src/reblock/eval/agreement.py` — not an `Eval`.** `Eval.score(block,
proposal)` (`contracts.py:124`) has no slot for a reference network, and agreement is
proposal-vs-reference. Forcing it into the Protocol would mean smuggling the reference in through
construction and lying about the signature.

Geometric only. The functional half (per-parcel egress agreement) is cut: permeability already
measures function, and it is a primary scorer.

- `buffered_iou(proposal, reference, r)` — IoU of the two buffers.
- `chamfer(proposal, reference, step)` — symmetric, over densified samples, reported
  **directionally**: proposal→reference is precision (paths drawn that aren't there),
  reference→proposal is recall (real paths missed). Never averaged; a blend hides which way a
  prediction fails, which is the only thing this measures.

**Holdout protocol — a hard metric exclusion radius is primary.** Leave-one-*block*-out is not
safe here: donors are immediate neighbours, frequently the same continuous OSM way clipped at a
block edge, often one mapper in one session. That is a live explanation for v5's 94% requiring no
generalization at all, and it threatens the existing headline retroactively.

The obvious fix — leave-one-*settlement*-out — turns out not to be well-defined. Connected
components over Cape Town's 1,136 qualified blocks: at 0 m, 596 components with 424 singletons; at
100 m, 417 components, 256 singletons (23%), and a largest component of **150 blocks (13% of the
corpus) spanning 5.7 km**; at 200 m the largest is 207 blocks spanning 8.6 km. One threshold
simultaneously chains distinct Cape Flats settlements into a metro-scale blob and strands a
quarter of blocks alone — labelling by known centroids at 100 m puts Gugulethu inside the blob
while Nyanga, Langa, Delft, Mitchells Plain and Kraaifontein are each size 1. Transitive chaining
has no natural stopping point, and there is no free label to fall back on: `gadm_code` is just the
`block_id` prefix (one value for all Cape Town qualified blocks) and `urban_id` is metro-scale
(121 for all of ZAF).

So:

- **primary** — a hard metric exclusion radius per recipient. Monotone in leakage, no chaining,
  one interpretable number, sweepable.
- **secondary, labelled an optimistic bound** — block-adjacent donors permitted.
- `settlement_id` is retained as a **stratification/reporting label only**, with its threshold
  stated wherever it appears. It is not a fold definition.
- stratified by donor distance in **metres**, not only feature distance.

**1c-i (primitives) is independent of 1a; 1c-ii (harness) is fully gated on it** — it needs both
the coverage table and `settlement_id`. The first draft claimed the whole unit was independent;
only the primitives are.

## 1d — GW pair-matrix pilot

Scratchpad, but with a committed artifact.

**Step zero — salvage.** The prior OT code survives in another session's scratchpad (`ot_gw.py`,
`transplant.py`, `select_donor.py`, `barycenter_amortization.py`, `rsc_*`, pickled pools) and is
one `/tmp` reclaim from gone. Copy it into the working scratchpad first.

**Timing pilot before the matrix.** 20 pairs, measured end to end. Each pair needs a real entropic
GW fit at ε ≤ 0.01 (the note's ablation makes that mandatory; log-domain Sinkhorn converges slowly
there), plus transplant, snap, a length-matched clearance solve, and permeability on both. v4's
per-recipient cost also included real-GW fits against a 20-candidate shortlist. Nothing downstream
should be sized before this number exists.

**Then ~100 pairs**, not the 1,000 the first draft proposed. The reblocker branch is the weak one
(see "What a larger pool does not fix"), so its measurement gets a pilot, not a sweep. Recipients
span the parcel-count range; donors span the GW-distance range.

**The artifact is a committed parquet**, not a note: `(recipient, donor, donor_type, features,
real_gw_dist, perm_gap, displacement, wall_clock)`. This is the most reusable thing Phase 1
produces — a retrieval benchmark any future featurization or donor material can be scored against
without re-solving anything. The first draft named scratchpad loss as a risk and then guaranteed
it by making the deliverable a note.

**`donor_type` is load-bearing, not bookkeeping.** Retrieval matches on *fabric* (building
points) and is completely agnostic to what linework a donor carries — same index, same GW
coupling, different material transported. Baking the matrix to one material would throw away most
of its reuse value and make the donor-material comparison unrepeatable. The axis has four values:

| `donor_type` | material | needs OSM? | unit |
|---|---|---|---|
| `osm_footpaths` | interior footpaths (the §5 winner) | yes | block |
| `osm_full` | streets **and** footpaths together — untested anywhere | yes | block |
| `street_form` | geometry-derived inter-block streets (the §7 loser) | no | region |
| `clearance` | a solved `ClearanceReblocker` network (the §4/v4 material) | no | block |

Phase 1 populates `osm_footpaths` only; the column exists so Phase 3 extends the same matrix
rather than starting a new one.

`osm_full` is worth naming because nobody has tried it. `_interior_desire_lines` subtracts the
street corridor by construction, so §5 transported interior footpaths *only* — never a donor's
full mapped circulation, which is arguably the truer picture of how a settlement moves and is free
relative to what 1a already produces.

**Report, do not pre-commit a rule.** Report the fidelity-vs-distance relationship, the
pool-size→rank-1-distance exponent (measured by subsampling at 10/30/100/300/1000, not assumed
from N^(−1/d)), and the per-pair cost. Decide in discussion.

**No secondary readouts.** All three candidates are cut.

- *Raw-vs-heat-trace bake-off* — the n=5 critique cuts both ways, and the note's case against
  heat-trace rested on effect sizes (median gap −0.334 vs −0.130; 0/5 vs 1/5 avoiding collapse)
  plus a mechanistic argument this spec endorses. Re-running it is misallocated power.
- *Pairwise-OT trim* — in a denser pool the k neighbours are closer to each other, so the trim
  converges toward a no-op. It predicts its own null, and the best case is a tie with a component
  the note recommends deleting.
- *Fixed-n resampling* — proposed in an earlier draft of this spec and withdrawn on a closer read
  of note §4. The signature is *already* "eigenvalues of the max-normalized pairwise-distance
  matrix of a **fixed-size (30-point) random subsample**", so "fixed-n resampled vs the 30-point
  subsample" has no defined contrast — it compares the thing to itself. The note's actual
  complaint is about the sampled *fraction* (46% of a 65-parcel block vs 11% of a 280-parcel one),
  and its second recommendation, re-ranking by real GW distance, is recorded in the same paragraph
  as *already adopted* by the v4/v5 barycenter spikes. A fixed-*fraction* variant would be a real
  contrast, but the downstream real-GW re-rank absorbs shortlist noise anyway, so it does not earn
  a slot here.

## Testing

- **1a** — unit tests for the refactored `_interior_desire_lines` (a pure-geometry signature is
  far more testable than the `Block` version) and for the join, against existing sample fixtures
  with synthetic linework. Pinned-bbox agreement test between `PbfDesireLines` and
  `OSMDesireLines`. Geofabrik download behind a network marker.
- **1b** — tile-enumeration unit test against a recorded `tiles.geojson`; download behind a
  network marker.
- **1c** — identical networks → IoU 1, Chamfer 0. A **graded** translation series (0/1/3/6/12 m)
  asserting monotone decay with a pinned value near `r`; the first draft's single 20 m case is
  vacuous, since 3 m buffers stop overlapping past 6 m and score 0 for any metric. Plus the test
  that was actually missing: a proposal that is a strict **subset** of the reference, and its
  transpose — the only asymmetry directional Chamfer exists to expose. Plus a densification-step
  sensitivity check (2 m densification puts a ~1 m quantization floor on Chamfer).
- **1d** — no tests; the parquet and the note are the artifacts.

## Risks

- **Census-to-usable attrition.** The note found roughly half of similarity-ranked neighbours had
  zero interior coverage. If that rate holds at scale the prediction branch's n is much smaller
  than 65k, and 1a is precisely the measurement that reveals it — before 1b spends on downloads.
- **`k_complexity` is not the depth screen.** Run the real BFS-peel screen on the shortlist before
  treating 65,364 as the pool.
- **Census wall clock — "free" refers to the data, not the compute.** Measured at 3.31 ms/block
  (clip + corridor difference + filter, preloaded STRtree): **1.67 single-core hours per tolerance
  over 1.813M blocks**, so ~5 h for the 0.5/2/5 m sweep and ~10 h once the near-miss tag set is
  included. Roughly 1–2 h on a fork pool. Plus the peel screen's ~2.45 core-hours on the
  shortlist. Budget it explicitly rather than discovering it.
- **PBF disk cost.** The GDAL OSM driver builds a multi-GB temp SQLite DB per country regardless
  of OGR-side filtering. Needs headroom.
- **Committed-example churn.** A stable `PbfDesireLines.identity` changes `proposal_id`s for the
  six committed examples. Check before landing.
- **Scratchpad loss.** Salvage is step zero of 1d; the pair matrix is committed rather than left
  in scratch.
- **Multi-UTM extents fail *silently*, which is the actual hazard.** `estimate_utm_crs()` returns
  EPSG:32735 for the ZAF bbox and EPSG:32637 for KEN with no error, and 32735 for the combined
  extent — nothing crashes. What you get is a scale bias: measured projected/true length ratio
  under a single country-wide UTM is **+0.72% at Cape Town, +1.23% at lon 16.5, +3.46% at lon
  41.9**. That biases `interior_length_m` by 1–3%; it is negligible against the 0.5 m `STREET_TOL`
  itself. Mitigation: derive the zone per block from its centroid and `groupby` it, **plus an
  assertion that |centroid lon − central meridian| ≤ 3.5°**, so a forgotten batch is loud instead
  of a quiet 3% drift. (The backlog's "continental scale" entry covers the general form.)

## Noted, not scoped

**Supervised desire-line detection from building geometry.** `notes/2026-07-15-desire-line-detection.md`
closed this at a ~0.5 recall / 0.35 precision ceiling — but all eight approaches were
*unsupervised, hand-tuned* detectors, and the note's own stated cause for the precision ceiling
("people don't walk every geometric gap") is exactly what labels teach and a heuristic cannot.
That work had one labeled block; 1a produces the label set. The note's closing line names the
revisit condition itself. The physical resolution-floor argument against **imagery** stands and is
untouched — this is the geometry route only. Decide after Phase 1.

**Phase 2 (sketch), with the first draft's framing corrected.** Building-point density or KDE
field (`np.histogram2d` — no rasterizer, and per T2 above, no parcels) → rotation sweep at fixed
scale → GW re-rank on the top-k.

Three corrections to how the first draft described this:

1. **Masked NCC is not "the raster analogue of the GEP."** The GEP *projects out* the
   road-provision-correlated direction so a road-poor recipient surfaces road-**richer** donors
   (note §7: +11–55% neighbour street-length). Masking road pixels makes the score road-**blind**
   — invariant, with no preference for richer donors. Different operations, different effects.
2. **It is a scan, not an index.** A masked score depends on the *intersection* of both masks at
   each shift, so it cannot be embedded in a vector space. Phase 2 would be an FFT-accelerated
   O(N) scan with per-block precomputation. Any framing of Phase 2 as "building an index" is
   wrong.
3. **Cheaper options first, in order.** (a) Do nothing raster-side: the note already validated
   cheap-spectral-shortlist → real-GW re-rank, and brute-force feature retrieval stays trivial
   well past 65k. (b) If a raster is wanted, **low-pass it**: a road corridor is ~3–6 m against ~5
   m NN spacing and ~200 m block diagonals, so a Gaussian at σ ≫ corridor width washes out the
   linear void while preserving fabric structure — zero new machinery. (c) Don't mask at all and
   *measure* whether road voids hurt, using 1d's matrix as ground truth.

The premise behind all of this is still correct and worth keeping: in a `histogram2d` of building
points, roads are never drawn, so deleting road geometry is a no-op — a road is a linear void in
the field, and its road-provision correlation survives deletion.

**Dependency reality check.** `skimage` is **not installed** and is not in `pyproject.toml`'s
pixi deps; `skimage.registration.phase_cross_correlation` is the off-the-shelf Padfield
implementation. Hand-rolling masked NCC on `scipy.fft` means ~6 forward FFTs per pair plus
shift-dependent overlap normalization and a minimum-overlap threshold — a numerically fiddly
multi-day subproject, not a footnote. The first draft's "no new dependencies" claim was
technically true and practically misleading.

**Phase 3b candidates (sketch).** Patch quilting — tile the recipient with overlapping windows,
give each its own best-matching donor patch, feather the demand fields, extract once with the
existing `demand_greedy_reblock` on the recipient's own gap graph — attacks the coverage-pattern
diagnosis by denying any single donor global influence, the local analogue of the barycenter
consensus that was the note's breakthrough. And policy transplant — fit the edge cost function
under which a donor's real footpaths are near-optimal, transplant the *parameters*, solve on the
recipient; ten numbers cannot carry a coverage gap. Both need a reason to exist that 1d has not
yet supplied.

**Retrieval unit.** Region accretion currently uses `DenseClusterRegionBuilder` — deepest-first
greedy on `√(n·A)/P`, which is exactly `√(n × compactness)` (`metric.Compactness` = `A/P²`), scored
on the *candidate block alone* and never on the shape of the union. §7's donor pool therefore had
uncontrolled outlines (150–900 parcels, tendril-shaped), a confound the note attributes entirely to
donor material. The originally-specified "accrete into compact/isoperimetric regions" builder was
never implemented, and its record was overwritten by the commit reporting the substitute
(`3df147a`). If a retrieval unit is needed in Phase 2, three candidates: fixed squares (FFT-native),
disks (rotation-symmetric outline, so all orientation signal comes from fabric), or a
shape-standardizing accretion builder — which respects real block boundaries so roads aren't
sliced, while still standardizing outline. **Retrieval unit need not equal intervention unit**;
extraction happens on the recipient's own substrate regardless.

A **shape-standardizing** builder is **no longer optional if the Phase 3 donor-material test
runs**: street-form donors force accretion (a single block has no internal streets), so a clean
material comparison needs outline control, which is precisely what §7 lacked. It is a prerequisite
of that test, not a Phase 2 nicety. Whatever the objective, it must score the *union's* shape as it
grows — distinct from today's `√(n × compactness)`, which scores each candidate block in isolation
and never looks at the shape being assembled.

**The objective is open, and compactness is only the obvious first guess.** The requirement is that
outline variance across candidates be small enough not to dominate GW distance — *not* that regions
be maximally circular. Squareness or rectangularity may well win: squares tile, they are FFT-native
if Phase 2 goes that way, and settlement fabric often carries a dominant orientation a circle
discards. A regular-n-gon target, or a data-driven objective fitted to the corpus, are equally
admissible. Choose it **empirically against a measurable criterion** — the outline's share of
inter-region GW distance variance, which 1d's matrix can measure directly — rather than assuming
isoperimetric quotient is the right target because it is the familiar one.

## Phase 3 donor-material test (promoted from "not revisited")

An earlier draft shelved §7's street-form donors as categorically inferior. That overstated the
evidence. The note compares §5 (OSM footpaths, ~92% of direct clearance) against §7 (street-form,
~67%) and attributes the whole 25-point gap to donor material — but the two experiments differ in
**three** ways at once:

1. **material** — interior footpaths vs inter-block streets;
2. **unit** — a single block vs an accreted region;
3. **outline control** — a real block boundary vs a deepest-first tendril of 150–900 parcels whose
   shape is a growth-algorithm artifact (§7 used `DenseClusterRegionBuilder`, *not* the
   compactness-preserving accretion that was specified for it and never built — see "Retrieval
   unit" below).

The mechanism argument for (1) is sound and remains the prior: streets are formal boundaries,
footpaths are access-optimized. But (2) and (3) also moved, so the gap cannot be assigned to (1)
alone, and "categorically inferior" is not what was demonstrated.

**Test both, on the same recipients and the same retrieval, with unit and outline held fixed.**
This is cheap on the material axis — `donor_type` in 1d's matrix is the whole mechanism — but
street-form is *not* equal cost overall, for a structural reason:

**Street donors force accretion.** A kblock block is a street-bounded face (`KblockSource` sets
`streets = poly.boundary`), so a single block has **no internal streets to copy**; only a
multi-block region has an internal grid. This is recorded in the 2026-07-10 feasibility doc and it
means street-form cannot be run block-level at all. Testing it cleanly therefore requires a
shape-standardizing region builder as a prerequisite, which footpath donors do not need.

**Expected value:** footpaths stay well ahead on the prior. What makes the test worth running
anyway is that street-form needs no OSM — and the coverage spike measured that **34.5% of
qualified blocks have no interior footpaths at all**, which is exactly the population a free donor
would have to serve.

## Explicitly not revisited

Shelved for reasons corpus size does not touch: **consensus-as-seed for
`LoopClosureRefiner`** (consensus was strong at 0.722; *refining* it produced 0.706, below even an
information-free cold seed at 0.751 — the refiner is the defect, and this is a mechanism finding
pool size cannot touch); **imagery detection** (physical resolution floor); **resistance-as-builder,
Voronoi-adjacency partition, distributional radius displacement** (all shelved on mechanism).

Also dropped from the first draft: the **§7 GEP paragraph** — its claim that the whitener was
"rank-deficient by construction" was wrong (PCA-whitening truncates to the ≤44 non-zero
components rather than inverting a singular covariance, and the note says whitening is what *made*
the GEP work), and its n=45 counted accreted *regions*, so a larger *block* pool does not supply
proportionally more of them. The honest residual — re-estimate the whitener on whatever pool
Phase 2 uses — happens anyway.

## What the review changed

A subagent review of the first draft found, and this revision accepts: `_interior_desire_lines`
could not run as specified; agreement cannot be an `Eval`; the tag-coupling guarantee was already
false via `conf/desire_source/osm.yaml`; the 20 m translation test was vacuous; the "37%" pass
rate was anchor-conditional; masked NCC is neither a GEP analogue nor an index; `skimage` is
absent; the PROJ risk does not reproduce; the §7 GEP paragraph was wrong; and the two best catches
— **spatial leakage** in the holdout and **`STREET_TOL` inflation** of the coverage column — were
absent entirely.

It also concluded the usable pool was ~1,650 blocks and that the country rows were unusable. That
holds for "what runs today with zero new work" and is the wrong denominator for the strategic
question: Open Buildings is public, tile-indexed, and already 90% wired, so the ~65k figure is
scope rather than a ceiling — and per T1/T2 above, the census needs no buildings at all. That is
what this revision is built on.

**Second pass — verification review of the revision.** The load-bearing T1 claim was demonstrated,
not merely argued: the census path run on a raw `ZAF_geodata.parquet` row with no `Block`, no
parcels and no building points returned **13 segments / 639.3 m**, byte-identical to the `Block`
path on the same block. Both country parquets are 100% Polygon with zero invalid geometries. Every
corpus-size figure reproduced exactly, and both contested overrides went to this spec — the
Overpass-vs-PBF one by an order of magnitude (one 0.25° Cape Flats tile is 29.2 MB / 40,640 ways /
7.1 s; ZAF+KEN is ~4,598 such tiles against Overpass's ~1 GB/day fair-use policy, versus 766 MB of
PBF once).

What did not hold up, and is fixed above: the parcel-centroid evidence came from a curated test
fixture rather than the qualified pool and does not survive a random redraw in *absolute* terms
(only the ratio does); the T2 tier row contradicted this spec's own insistence that the BFS-peel
screen is the real gate; 1b's "shortlist's extent" would have downloaded both countries in full;
the tile count was 20, not 10–15, at 3.78 GB; `settlement_id` was named but never defined, and
measurement showed it cannot be; the census wall clock and the silent multi-UTM scale bias were
unbudgeted; the reblocking-ceiling argument dropped displacement, the co-primary metric; and the
1d secondary readout rested on a misreading of note §4 and specified no contrast, so it is cut.
