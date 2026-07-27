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
| **T1 · OSM census** | blocks parquet + OSM linework | **all 1,813,575** | free today, behind one refactor |
| **T2 · Retrieval / index** | building **points** only — no Voronoi, no `Block` | ~65k qualified | after a targeted tile download |
| **T3 · Scoring / solving** | points + Voronoi parcels + substrate | grows with T2 | the genuinely expensive tier |

**T1 is free because kblock `streets` *is* the block boundary.** `_interior_desire_lines`
currently takes a `Block` (`osm_footpaths.py:23`), and `Block.__post_init__` raises
`ValueError("Block.parcels must be non-empty")` (`contracts.py:54-56`) — parcels being Voronoi of
building points, which exist only for the Cape Town and Nairobi bboxes. That is a **code-structure
limit masquerading as a data limit**: refactored to `(lines, boundary, streets, crs)`, the census
runs country-wide with zero building downloads.

**T2 never needs parcels.** Parcels are `Voronoi(building_points)` clipped — a deterministic
function, so the points carry strictly more information and routing a descriptor through Voronoi
is a lossy detour that also introduces a boundary-clipping artifact. Measured on six real Cape
Town blocks, parcel centroid → nearest building point:

| block | n | diagonal | median | p90 | median / diagonal |
|---|---|---|---|---|---|
| …1_3934 | 211 | 279.5 m | 1.08 m | 3.15 m | 0.39% |
| …1_3968 | 894 | 965.7 m | 0.90 m | 2.41 m | 0.09% |
| …1_2698 | 67 | 133.1 m | 1.93 m | 5.20 m | 1.45% |
| …1_2701 | 82 | 185.1 m | 1.21 m | 2.34 m | 0.65% |
| …1_4310 | 147 | 176.2 m | 0.87 m | 1.92 m | 0.49% |
| …1_4314 | 86 | 158.4 m | 0.98 m | 2.95 m | 0.62% |

For scale, the OT note treats a GW barycentric recovery error of **6.24 m (3.2% of diagonal)** as
a *successful* mechanism check. The substitution sits several times below the transport
machinery's own validated noise floor.

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
centroid, `:216-223`). ZAF+KEN is on the order of 10–15 tiles.

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

## Success criteria

Phase 1 answers three questions and produces four artifacts. It is *not* trying to beat anything.

1. How many blocks have genuine interior OSM footpaths, and how does that number move with the
   interiority tolerance? → sizes the prediction branch, or kills it.
2. Is the transplant-fidelity-vs-GW-distance relationship strong enough to be worth a bigger
   index? → informs, does not gate, Phase 2.
3. What does a pair cost, in wall-clock? → makes any later scale claim budgetable.

The Phase 3a deliverable this serves is a **mapping/data product** — predicted footpaths for
blocks OSM has not mapped — not a reblocker. The note is explicit that `osm_footpaths`' own *real*
network is at best comparable to a clearance solve on permeability (0.77 vs 0.78 in v5; 0.80 vs
0.91 in v2), so a *predicted* network is capped by a ceiling that is already not a reblocking win.

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
  IN (...)", use_arrow=True)` — so a country PBF does not materialize every South African
  `highway=track`. The GDAL `OSM` driver is present in the env (no new dependency) but builds a
  multi-GB temp SQLite DB; budget disk for it.

**Interiority is reported as a tolerance sweep, not a number.** `_interior_desire_lines` subtracts
`streets.buffer(STREET_TOL)` with `STREET_TOL = 0.5 m`, and for kblock `streets` is the block
outline. OSM ways are digitized against different imagery than the Open Buildings / kblock
outlines, so a genuinely boundary-running path more than 0.5 m off the outline is recorded as
*interior* — inflating the exact column that gates "which blocks have real coverage." Emit
interiority at **0.5 / 2 / 5 m** and decide after looking. Same instinct as the near-miss tags
below, applied to the thing that actually matters.

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
coverage from 1a — download each tile, filter to `OB_MIN_CONFIDENCE = 0.7` and the shortlist's
extent, write per-ISO GeoParquet.

Query-driven, not country-wide: the census tells us which blocks matter before we pay for any of
them. This is what makes T2 and T3 real, and it is the only genuinely new download in Phase 1.

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

**Holdout protocol — leave-one-settlement-out is primary.** Leave-one-*block*-out is not safe
here: donors are immediate neighbours, frequently the same continuous OSM way clipped at a block
edge, often one mapper in one session. That is a live explanation for v5's 94% requiring no
generalization at all, and it threatens the existing headline retroactively. Report:

- **primary** — leave-one-settlement-out (or a hard metric exclusion radius),
- **secondary, labelled an optimistic bound** — block-adjacent donors permitted,
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

**The artifact is a committed parquet**, not a note: `(recipient, donor, features, real_gw_dist,
perm_gap, displacement, wall_clock)`. This is the most reusable thing Phase 1 produces — a
retrieval benchmark any future featurization can be scored against without re-solving anything.
The first draft named scratchpad loss as a risk and then guaranteed it by making the deliverable a
note.

**Report, do not pre-commit a rule.** Report the fidelity-vs-distance relationship, the
pool-size→rank-1-distance exponent (measured by subsampling at 10/30/100/300/1000, not assumed
from N^(−1/d)), and the per-pair cost. Decide in discussion.

**One secondary readout, replacing two.** Score a **fixed-n-resampled** raw signature against the
current 30-point subsample — the note's own open recommendation ("resample to a fixed n, or
re-rank a shortlist by actual GW distance"), which the first draft omitted in favour of two
re-tests that don't earn their place. The raw-vs-heat-trace bake-off is dropped: the n=5 critique
cuts both ways, the note's case against heat-trace rested on effect sizes (median gap −0.334 vs
−0.130; 0/5 vs 1/5 avoiding collapse) plus a mechanistic argument this spec endorses, so re-running
it is misallocated power. The pairwise-OT trim re-test is dropped: in a denser pool the k
neighbours are closer to each other, so the trim converges toward a no-op — it predicts its own
null, and the best case is a tie with a component the note recommends deleting.

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
- **PBF disk cost.** The GDAL OSM driver builds a multi-GB temp SQLite DB per country. Mitigated
  by OGR-side `where` filtering; still needs headroom.
- **Committed-example churn.** A stable `PbfDesireLines.identity` changes `proposal_id`s for the
  six committed examples. Check before landing.
- **Scratchpad loss.** Salvage is step zero of 1d; the pair matrix is committed rather than left
  in scratch.
- **Multi-UTM extents.** `Block.crs` must be projected and `STREET_TOL` is metres, but ZAF+KEN
  spans several UTM zones. The census must batch by zone. (The backlog's "continental scale" entry
  covers the general form of this.)

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
compactness-preserving accretion builder — which respects real block boundaries so roads aren't
sliced, while still standardizing outline. **Retrieval unit need not equal intervention unit**;
extraction happens on the recipient's own substrate regardless.

## Explicitly not revisited

Shelved for reasons corpus size does not touch: **§7 street-form donors** (streets are formal
boundaries, footpaths are access-optimized — categorical, not distance); **consensus-as-seed for
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
scope rather than a ceiling — and per T1/T2 above, the census needs no buildings at all and
retrieval needs no parcels. That is what this revision is built on.
