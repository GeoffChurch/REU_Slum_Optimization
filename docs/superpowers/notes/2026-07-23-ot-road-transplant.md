# OT road transplant: thoroughly explored, closed as a reblocker — keepers are footpath-prediction (OSM-consensus) + a GEP retrieval primitive

**Date:** 2026-07-23
**Status:** scratchpad-only feasibility exploration. Nothing here touched `src/` — no repo source
or config was modified, no method was shipped. All code (`ot_gw.py`, `transplant.py`,
`gap_snap_fix.py`, `select_donor.py`, `amortization_test.py`, `barycenter_amortization.py`,
`osm_barycenter.py`, and drivers) lives in ephemeral scratchpad directories and is gitignored; this
note is the sole committed record of the work. This is the concrete follow-through on the OT/GW
road-transplant idea flagged as backlog in `MEMORY.md` and scoped without an implementation in
`transfer-idea-feasibility.md` (2026-07-10).

**Outcome:** the literal geometric transplant this idea started as — copy one donor block's roads
onto a recipient via optimal transport — is a dead end: it reproduces the donor's coverage
pattern, not the recipient's needs, and loses to direct `clearance` and to the recipient's own OSM
at matched road length, on every variant tried (single donor, snap-fix, k-NN amortization). The
one genuinely promising mutation is **barycenter consensus**: transport several good, close
neighbors' networks and extract the demand they agree on. With real OSM-footpath donors this
reaches 93.7% of a block's own real network's permeability, and 92.2% of direct clearance's, at
*lower* displacement than clearance — the strongest result in the whole study, though so far on
a single recipient.

## 1. Goal and mechanism

**Goal:** predict/transplant a road network onto a recipient block by finding a structurally
similar donor block and transporting its roads across via optimal transport, instead of solving
the recipient from scratch. Mechanism: hand-rolled entropic Gromov-Wasserstein (GW) + unbalanced
OT (POT not installed, so the GW outer loop and log-domain Sinkhorn inner solve were implemented
directly, ~90 lines) to find a soft parcel↔parcel correspondence between donor and recipient, a
barycentric projection to map donor road vertices into recipient coordinates, then a gap-snap step
to pull the transported polylines onto the recipient's own buildable substrate (`ChordSubstrate`
boundary/diagonal graph) so the output never cuts through a building.

**Mechanism sanity check — coherent, not a blurred mush.** Self/perturbed transplant: 40972's own
13 interior OSM footpath segments (639 m) were transported through the full pipeline onto a
**rotated (53°) + jittered (0.84 m) + randomly reshuffled** copy of 40972's own 263 parcels.
Reshuffling erases index correspondence entirely, so GW has to recover the match from intrinsic
pairwise-distance geometry alone. Result: mean barycentric recovery error 6.24 m (median 5.51 m)
against the known ground-truth rigid transform — **3.2% of the block's 197 m diagonal**. The
discrete coupling is even sharper (argmax-coupling position error: mean 2.98 m, median 1.12 m,
close to the 0.84 m jitter floor). The transported network preserves branching topology rather
than collapsing to the centroid.

**Entropy regularization is load-bearing, not a safety margin.** An ε ablation on the same pair
confirms a sharp blur cliff:

| ε | transported length (pre-snap) | snapped segments |
|---|---|---|
| 0.01 | 955.8 m | 33 |
| 0.05 | 32.9 m | 12 |
| 0.20 | 14.8 m | 0 (fully degenerate) |
| 0.50 | 13.3 m | 0 (fully degenerate) |

ε ≤ 0.01 is required: ε = 0.05 already collapses the network to ~3% of its proper length. All
subsequent spikes fix ε = 0.01, τ = 1.0.

**Verdict on mechanism: sound and correctly implemented.** This was never the failure mode of any
later variant — every collapse below is a *coverage/selection* problem, not a transport-math
problem.

## 2. At a glance — variants and scores

| Variant | Donor material | Recipients | Best result | vs. baseline | Verdict |
|---|---|---|---|---|---|
| Mechanism check | recipient's own network, rotated/jittered/reshuffled | 1 (40972) | 6.24 m mean recovery error (3.2% of diagonal) | ground-truth rigid transform | Sound |
| v2 — single OSM donor | 1 OSM donor (`46279`) | 1 (40972) | perm 0.68 @ 47% disp, 1045 m | own OSM 0.80@28% (639 m); clearance 0.91@69% (1050 m) | Dead end |
| v2 + recipient-aware gap-snap fix | same | 1 (40972) | perm 0.666 @ 47% disp, 1057 m | no better than the naive straight-line snap | No improvement |
| v3 — single-donor k=5 amortization | 1 clearance donor → 5 NN × 3 featurizations | 14 unique neighbors | median perm gap −0.14 (13/15 collapse); rank-1 sometimes wins (+0.098, +0.029) | direct clearance, length-matched | Mostly dead; narrow rank-1-only regime |
| v3 — heat-trace featurization | same donor | 5 | median perm gap −0.334, 0/5 avoid collapse, correlation flips to +0.60 (n.s.) | raw-signature's −0.130, ρ=−0.90 | Worse than raw signature |
| v4 — barycenter consensus, clearance donors | k=6 clearance-solved neighbors, real-GW-cost selected | 3 (25th/50th/75th %ile) | 43% / 91% / 73% of direct clearance; 2–10× best single donor | direct clearance, length-matched | Promising, not yet a replacement |
| v5 — barycenter consensus, real OSM donors | k=7 real-OSM-footpath neighbors | 1 (40972) | perm 0.722 = 93.7% of own OSM, 92.2% of direct clearance, at *lower* displacement (40.1% vs 43.9%) | own OSM 0.77@28%; direct clearance 0.78@44% | Strongest result; n=1; a *predictor*, not a reblocker |
| v6 — barycenter consensus, region street-form (§7) | k=6 inter-block-street donors (free, no OSM) | 1 (40972+40976) | perm 0.233 = 67% of direct clearance; beats own streets 2–6× | direct clearance 0.348; own streets 0.039 | Not competitive (weak donor); GEP decorrelation validated |
| Consensus-as-seed + refine (§8.4) | OSM consensus as `LoopClosureRefiner` prior | 1 (40972) | perm 0.706 — below clearance 0.783 *and* below an info-free cold-refine 0.751 | direct clearance; cold-refine | No-go (seed adds *negative* value) |

## 3. Single-donor transplant is a dead end

### v2 — real cross-block transplant

Recipient `ZAF.9.3.1_1_40972` (263 parcels, 13 interior OSM segments, 639 m — the
method-comparison flagship with real ground truth). Donor selection matched a bootstrap-averaged
sorted-eigenvalue signature (GW-consistent shape descriptor) against 2,788 candidate blocks within
6 km of 4 known Cape Town informal-settlement anchors. **The top 6 signature-ranked candidates had
zero or near-zero real OSM coverage** (0, 2, 1, 7, 0, 2 raw interior segments) — a real-world OSM
sparsity problem, not a search failure. The donor actually used, `ZAF.9.3.1_1_46279` (rank 7 of
2,788, signature distance 0.522, 294 parcels), was chosen as the best-covered candidate near the
top of the ranking: 35 interior OSM segments, 1,187 m, richer than the recipient's own network.

Scored with the current metrics (`reblock.permeability.permeability` + `reblock.budget.displacement`,
`g_walk=1.0, g_road=20.0, g_street=20.0, corridor_m=3.0`), against direct `clearance`
(`depth_target=1`, truncated to a length-matched prefix) and the recipient's own OSM ground truth:

| network | road length (m) | permeability | displacement |
|---|---|---|---|
| transported (GW+UOT) | 1044.9 | 0.6819 | 46.85% |
| own OSM (ground truth) | 639.3 | 0.8017 | 28.40% |
| clearance (length-matched) | 1050.5 | 0.9094 | 69.31% |

The transplant **underperforms the recipient's own OSM despite using 63% more road** (0.68 vs.
0.80 permeability) — it is less permeability-efficient per metre than even the unoptimized real
network it is trying to approximate. Against length-matched direct clearance the gap is larger
still (0.68 vs. 0.91), though clearance buys that at much higher displacement (69% vs. 47%): the
transplant isn't dominated on both axes simultaneously, but it doesn't win on either against the
stronger baseline.

The reason is visible in the geometry: the donor's own OSM footpaths cover only ~60% of the donor
block's footprint (one flank is essentially unmapped). The transplant **faithfully reproduces that
same coverage gap, warped into recipient geometry** — one side of the recipient goes unserved —
rather than adapting to the recipient's actual interior structure. This is exactly the risk
`transfer-idea-feasibility.md` (2026-07-10) flagged without an implementation to test it against:
*"the naive 'warp the exemplar's road geometry onto the slum via OT' lands roads in arbitrary
positions."* This spike gives that concern direct empirical evidence.

### Recipient-aware gap-snap fix — did not help

Follow-up: does fixing the final snap step alone close the gap? The original snap nudges each
transported vertex onto the nearest `ChordSubstrate` node, then reconnects consecutive snapped
vertices with a straight line, which can cut across whatever sits between two arbitrary boundary
nodes. The fix keeps the same node-snapping but reconnects each pair via **Dijkstra shortest path
along the boundary graph's own edges**, so every hop is a real parcel-boundary segment.

**Result: no improvement, marginally worse.**

| network | road length (m) | permeability | displacement |
|---|---|---|---|
| transplant (old basic-snap) | 1044.9 | 0.6819 | 46.85% |
| transplant (new routed gap-snap) | 1057.2 | 0.6661 | 46.86% |
| own OSM (ground truth) | 639.3 | 0.8017 | 28.40% |
| clearance (length-matched to new) | 1063.4 | 0.9110 | 69.49% |

Displacement is flat (rounding-noise difference) and permeability is lower at a slightly longer
length — the new snap is strictly dominated by the old one. A repulsion-parameter sweep from pure
shortest-path to the same max-clearance-ridge cost `ClearanceReblocker` itself uses (s ∈
{−6…6}) left displacement flat at ~47–48% across the *entire* range, and made permeability
*worse* as repulsion increased (paths got longer, from 1057 m to 1173 m, without touching fewer
buildings).

**Root cause:** displacement is a function of road length and *which parcels get targeted*
(network coverage/placement), not small-scale path choice between already-fixed waypoints — both
snap variants only ever touch `ChordSubstrate` nodes, so neither ever routes through a building.
This confirms the coverage-pattern problem is structural to the transplant, not a fixable snap
detail.

## 4. Amortizing a single donor across neighbors (v3) mostly collapses

Question: solve one block, transplant its roads to several feature-space neighbors — does a single
solve amortize across a neighborhood, or does the collapse from section 3 generalize?

**Setup:** pool of 300 Cape Town blocks (building_count 60–300, depth ≥ 4 via the codebase's own
BFS-peel screen) → 111 qualified candidates. Donor `ZAF.9.3.1_1_37664` (139 parcels, pool-median
size), solved with default `ClearanceReblocker()`: 10 segments, 286.8 m. Transplanted (GW+UOT,
IDW vertex warp, recipient-aware routed gap-snap) onto the donor's k=5 nearest neighbors in three
featurizations — spectral (bootstrap sorted-eigenvalue signature), morphological (parcel count,
density, compactness, NN spacing, access depth), and pairwise-distance histogram — compared against
direct `clearance`, length-matched per neighbor.

| featurization | n | median perm. gap (transplant − direct) | Spearman(feature-dist, perm. gap) |
|---|---|---|---|
| spectral | 5 | **−0.130** | **−0.900** |
| morphological | 5 | −0.158 | −0.700 |
| pairwise-distance histogram | 5 | −0.142 | −0.700 |
| all 15 pooled | 15 | −0.142 | −0.677 (p=0.006) |

**Mostly collapses:** 13 of 15 selections land the transplant below direct clearance — a typical
transplant reaches roughly half the direct-clearance permeability at matched length. One selection
(pairwise-histogram) nearly fully degenerated to a single 10 m stub.

**But a tighter match transplants more faithfully — the real finding here.** Within every
featurization, permeability gap correlates strongly and negatively with feature distance, and in
all three spaces **the single closest-matched neighbor (rank 1) had the best (least-negative) gap
of its five candidates**, with two of three actually *beating* direct clearance outright:

- spectral rank-1 (`ZAF.9.3.1_1_42278`, dist 2.747): 0.476 vs. 0.379 direct → **+0.098**
- morphological rank-1 (`ZAF.9.3.1_1_37715`, dist 0.901): 0.239 vs. 0.210 direct → **+0.029**
- pairwise-histogram rank-1 (`ZAF.9.3.1_1_44529`, dist 8.118): 0.079 vs. 0.208 direct → −0.129
  (best of its own space, still not a win)

Rank 2 already falls back into the −0.13 to −0.17 collapse band shared by ranks 3–5: the gradient
is steep near the front and then flat. **Spectral is the best featurization on every axis**
(smallest median gap, strongest correlation, largest single win) — mechanistically sensible, since
it is built from the same normalized pairwise-distance structure the GW transport cost itself
optimizes over, so a small spectral distance is close to "GW will find an easy, low-distortion
coupling."

**Go/no-go for v3: no-go for k=5 blanket amortization**, conditional go only for a single
nearest-neighbor regime — and even that is just 2–3 winning data points, not validated at scale.

### Featurization lesson: raw eigenvalue signature vs. heat-trace

The spectral signature above is a **raw sorted-eigenvalue signature** — eigenvalues of the
max-normalized pairwise-distance matrix of a fixed-size (30-point) random subsample of parcel
centroids. It is not properly n-invariant even though its output vector length is fixed: a
30-point subsample of a 65-parcel block samples ~46% of the field; the same subsample of a
280-parcel block samples ~11%. A follow-on spike tested whether a **properly n-invariant**
descriptor — the heat-trace of the normalized graph Laplacian of a Gaussian-affinity graph over
*all* of a block's centroids (an intensive quantity intrinsically bounded to [0,2] regardless of
n, no subsampling needed) — would tighten the fidelity correlation and widen the non-collapse
regime past rank-1.

**Result: heat-trace is worse, not better — the correlation flips sign and the win regime
vanishes.**

| signature | median perm. gap | Spearman(dist, gap) | p | rank-1 gap | # avoiding collapse (of 5) |
|---|---|---|---|---|---|
| raw sorted-eigenvalue (old) | **−0.130** | **−0.900** | 0.037 | **+0.098** (win) | **1/5** |
| heat-trace (new, n-invariant) | −0.334 | +0.600 | 0.285 | −0.388 (worst of its own 5) | **0/5** |

Heat-trace's own rank ordering isn't even internally consistent with "closer match ⇒ better
transplant" (its rank-1 has a worse gap than its rank-4). Two n-invariance sanity checks confirmed
heat-trace's invariance property is real and correctly implemented (a clean 10×10-vs-20×20 lattice
re-tile lands the two shapes 20× closer under heat-trace than under the raw signature) — so this
is not a bug. The explanation is that **n-invariance and GW-transplant-relevance are different
properties**: heat-trace discards the global pairwise-distance structure the GW transport cost is
literally built from, in favor of a local Gaussian-affinity-graph spectrum (connectivity/clustering
at various scales, not point-to-point distances). Trading GW-cost alignment for provable
n-invariance is a bad trade for *this* downstream task — GW-alignment (global pairwise-distance
structure) matters more than n-invariance for predicting transplant fidelity. **If this direction
is revisited, fix the n-length issue by resampling to a fixed n or re-ranking a shortlist by
actual GW distance — not by swapping in a "cleaner" local-graph signature.** (This is exactly the
fix the barycenter spikes below adopted: a cheap feature-space shortlist, then real GW-cost
re-ranking for the final selection.)

## 5. Barycenter consensus is the breakthrough

Single-donor transplant copies one donor's idiosyncratic geometry, which rarely fits a different
block's fabric. The question: does a **weighted consensus (barycenter)** of several good, close
neighbors' transported networks — "the structure the good, close neighbors agree on" — do better
than betting on any one donor, close enough to direct clearance to be worth building?

**Method (both variants below):** two-stage candidate selection — a cheap feature-space shortlist
(spectral signature) narrows the pool, then the *real* (non-linearized) GW cost is fit and scored
against each shortlisted candidate, and the k closest **by real GW cost** (not feature distance)
are kept. Each neighbor is weighted by `quality_i × closeness_i`, where `quality_i = own_permeability_i
× (1 − own_displacement_i)` (scored on the neighbor's own block) and `closeness_i = exp(−gw_dist_i
/ τ)`. Each neighbor's own road network is transported onto the recipient via its own fitted
coupling, buffered into a demand field, and the field is combined as a weighted sum. Extraction is
gap-aware by construction: a new `demand_greedy_reblock` function (adapted from
`reblock.methods.clearance._greedy_reblock`'s deepest-parcel-first Dijkstra drainage-tree
construction) builds the actual road network directly on the recipient's own `ChordSubstrate`
graph, with edge cost `length / (demand + 0.05)` — so every edge is, by construction, a real
parcel-boundary/diagonal gap.

### v4 — clearance donors (n=3 recipients)

Recipients at the 25th/50th/75th percentile of parcel count from the qualified pool (101, 139, 195
parcels) — not one hand-picked block. k=6 neighbors selected by real GW cost from a 20-candidate
feature shortlist, each solved with default `ClearanceReblocker()`. Matched length = median of the
k neighbors' own transplant lengths (a "typical single-donor budget").

| recipient | method | road (m) | permeability | displacement |
|---|---|---|---|---|
| `_42839` (101 parcels) | direct clearance | 164.3 | **0.387** | 2.2% |
| | best-donor (rank-1) | 83.2 | 0.016 | 1.1% |
| | worst-donor (rank-6, floor) | 140.1 | 0.023 | 3.1% |
| | **barycenter consensus** | 217.0 | **0.166** | 3.4% |
| `_37664` (139 parcels) | direct clearance | 161.5 | **0.436** | 11.9% |
| | best-donor (rank-1) | 150.3 | 0.105 | 10.2% |
| | worst-donor (rank-6, floor) | 48.4 | 0.009 | 2.5% |
| | **barycenter consensus** | 157.3 | **0.396** | 15.9% |
| `_44529` (195 parcels) | direct clearance | 154.0 | **0.514** | 21.9% |
| | best-donor (rank-1) | 108.4 | 0.167 | 16.8% |
| | worst-donor (rank-6, floor) | 8.7 | 0.013 | 3.2% |
| | **barycenter consensus** | 157.7 | **0.375** | 20.0% |

**Beats single-donor decisively on all 3 recipients:** barycenter vs. best-donor (rank-1)
permeability is **10.4×, 3.8×, 2.2×**. Pooling several good, close neighbors' networks beats even
the single BEST donor by real GW cost, not just an average one.

**Does not yet match direct clearance:** barycenter reaches **43%, 91%, and 73%** of direct
clearance's permeability at matched length — the gap has closed dramatically from single-donor's
single-digit-percent collapse, but recipient `_42839` and `_44529` still show a real deficit
(`_37664`, 0.396 vs. 0.436, is a near-match).

**Robust median/trim did not help.** A pairwise-OT trim (drop the 2 of 6 neighbors with the
highest average pairwise distance to the rest of the group, before reweighting) either tied the
plain barycenter (recipient 1, identical network), was marginally worse (recipient 2), or was
clearly worse (recipient 3, 0.318 vs. 0.375). The mechanism is diagnostic: on recipient 3 the trim
dropped the group's rank-1 and rank-2 neighbors **by real distance to the recipient**, because they
were the two most different from *each other* — being close to the recipient and being similar to
your peers are different properties, and trimming on the latter threw away the ingredients that
mattered most for the former. **Recommendation if revisited: drop the pairwise-OT trim, keep the
plain weighted barycenter over all k.**

### v5 — real OSM-footpath donors (n=1 recipient, strongest result)

Same question with real OSM footpaths as donor material instead of synthesized clearance networks
— "predict an un-mapped block's footpaths by consensus of its similar mapped neighbors' real
footpaths" — on `ZAF.9.3.1_1_40972`, the only block in the pool with a committed ground-truth OSM
snapshot to check against.

**Donor pool:** the top 15 candidates by shape-signature similarity to 40972 were each queried live
via `OSMDesireLines`/Overpass. **8 of 15 had zero interior OSM footpaths** (only boundary streets
mapped — informal-settlement interiors are unevenly mapped in OSM); **7 of 15 had real interior
coverage** and formed the donor pool (interior segments ranging 1–35, lengths 52.6–1187.1 m; two
donors, `39233` and `39496`, contribute only ~53–55 m each — thin but real material).

**Matched length:** the recipient's own OSM footpath length (639.3 m) — its actual ground-truth
budget, since a real network is available here to target directly.

| network | road (m) | permeability | displacement |
|---|---|---|---|
| own OSM (ground truth) | 639.3 | **0.7702** | 28.4% |
| direct clearance | 642.4 | **0.7830** | 43.9% |
| best single-OSM-donor (rank-1 by GW dist, `43137`) | 80.5 | 0.1927 | 5.4% |
| **OSM-barycenter consensus** | 649.1 | **0.7216** | 40.1% |

An addendum scored all 7 donors individually (not just rank-1-by-GW-distance, which turned out to
be a weak footpath donor despite being the closest shape match): the true oracle-best single donor
is `46295` at 0.6055 permeability (605.1 m, 32.6% displacement). The consensus beats even this
oracle pick by ~19% relative (0.722 vs. 0.606).

**This is the strongest result in the whole study.** The consensus reaches:
- **93.7%** of the recipient's own ground-truth OSM permeability at matched length (0.722 vs.
  0.770) — far closer than the clearance-donor consensus got to its own target (43–91% across the
  3 recipients above).
- **92.2%** of direct clearance's permeability (0.722 vs. 0.783), while displacing **fewer**
  buildings than clearance does to get there (40.1% vs. 43.9%) — both still above own-OSM's 28.4%
  floor, but a materially better permeability/displacement tradeoff than clearance's own solution.

The map overlay shows the consensus network tracking the real OSM network's central and
lower-right circulation spine closely, diverging mainly where the ground truth has long individual
spurs that one particular real donor happened to have and the group consensus didn't reconstruct.
This is genuine evidence that a block's real desire-line network carries predictable structure from
its similar neighbors' real networks — not just that *a* network can be built that scores well, but
that the *shape* of real informal circulation generalizes across similar blocks well enough for a
weighted consensus to approximate it without ever seeing the recipient's own mapped footpaths.

## 6. Data notes

- **Overpass fetch works headless**, live, from this environment (`OSMDesireLines`'s default
  User-Agent/406 handling already covers it; routine 429/504 rate-limit retries were needed at
  donor-search scale). Results are cached to disk (`~/.cache/reblock/osm/`) so re-runs are
  idempotent.
- **~Half of similarity-ranked neighbors have zero interior OSM coverage** — a real
  data-availability ceiling, not a fetch failure (top-6 candidates in v2's search: 0/15 in v5's
  narrower search had zero interior segments; only mapped neighbors can donate real footpaths).
  This is the structural reason real-OSM barycenter consensus is currently n=1: only one recipient
  in the working pool has a committed ground-truth OSM snapshot to validate against at all.
- **Clearance donors have no such coverage gap** (`ClearanceReblocker` is solvable on any block
  regardless of OSM mapping density) but are structurally weaker donor material: real OSM networks
  start from an intrinsically lower-displacement shape (desire lines thread through existing gaps
  rather than bulldozing), and the OSM-donor consensus preserved much of that advantage, while the
  clearance-donor consensus converged back toward clearance's higher-displacement solution.

## 7. Variant-1 first cut — region street-form consensus (tested 2026-07-23)

Built the full region-level chain the "next steps" below proposed (accretion → geometry-derived
inter-block street network → rotation/scale-invariant distance-spectrum features → GEP-decorrelated
retrieval → GW+UOT weighted-barycenter consensus → scoring), on one recipient region
(`ZAF.9.3.1_1_40972+40976`, 716 parcels, own internal streets 87 m = 16th percentile of a 45-region
donor pool). **The chain holds end-to-end, and two pieces are validated — but the donor type fails.**

- **GEP decorrelation works** (the crux). The feature direction most correlated with road provision
  (+0.73) was projected out, after which the recipient's k-NN mean internal-street-length rose +11–55%
  at 4 of 5 tested `k` (k=8: 361→561 m). So a road-poor recipient *does* surface road-richer,
  shape-similar donors instead of near-duplicates of its own under-service. **Caveat:** it required
  PCA-whitening first — naively z-scoring the 60-dim eigenvalue spectrum made raw and decorrelated
  retrieval byte-identical, because ~58 noise dimensions swamped the ~2 real signal ones. Lesson:
  whiten, or use far fewer, more-informative spectral features.
- **Consensus beats the recipient's own sparse streets 2–6×** and beats the best single donor.
- **But the consensus loses to direct clearance** — 0.233 vs 0.348 permeability at the donor budget
  (349 m); 0.081 vs 0.212 at the thin own-network budget (87 m). The map shows a dense but *fragmented*
  network, not clearance's coherent interior-reaching tree. The **free inter-block-street donor is a
  materially weaker ingredient than the OSM footpaths of §5**: block-level OSM consensus reached ~92%
  of clearance, this region street-form only ~67%. Streets are formal boundaries; footpaths are
  access-optimized, and it shows. (Also n=1, plus a thin-budget truncation artifact where consensus
  dipped below the best single donor.)

**Verdict on street-form donors: not a green light** — the chain and the GEP are proven, but this donor
type isn't competitive with a from-scratch clearance solve; the free-data advantage costs too much
quality. The GEP is now a reusable retrieval primitive; the remaining leverage is donor quality.

## 8. Verdict and next steps

**Single-donor transplant: shelved.** Literal geometric OT transplant — from a single donor,
however well-matched — is dominated by direct `clearance` and by the recipient's own OSM on every
variant tried (single-pair, recipient-aware snap-fix, k=5 blanket amortization, heat-trace
refeaturization). The failure is structural, not fixable by better snapping or a "cleaner"
featurization: a transplant reproduces the *donor's* coverage pattern, not the *recipient's* needs.
Only a single-nearest-neighbor-by-real-GW-cost regime shows occasional wins, and it's too narrow
(1–2 of 5 candidates per pool) to productionize on its own.

**Barycenter consensus: the strongest result — but a *predictor*, not a reblocker.** Pooling several good, close neighbors' networks
into a demand-weighted consensus, extracted along the recipient's own gap graph, beats any single
donor by 2–10× and gets meaningfully close to direct clearance (43–91% with clearance donors, n=3)
— and with real OSM donors, reaches 93.7% of a block's own real network and 92.2% of direct
clearance at *lower* displacement (n=1). This validates the core hypothesis: a block's real
footpath network carries predictable structure from a consensus of its similar, *mapped* neighbors.

Next steps, in rough priority order:

1. **Validate consensus on more recipients (n>1).** The clearance-donor result covers 3 recipients;
   the (stronger) real-OSM-donor result covers exactly 1, because 40972 is the only block in the
   working pool with committed ground truth. Fetching/committing a second real-OSM-mapped block —
   or relaxing the donor-pool's anchor/size window to grow the coverage-filtered pool — is needed
   before treating the 93.7%/92.2% result as a generalizable rate rather than one strong data
   point.
2. **Region-level street-form transplant — TESTED (§7), mixed.** The full chain holds, but the free
   inter-block-street donor loses to direct clearance (~67% of it), so street-form is not a competitive
   donor on its own. Region-level accretion is a viable substrate; the *donor material* is the problem.
3. **GEP road-provision decorrelation — VALIDATED (§7).** Projecting out the road-provision-correlated
   feature direction does pull road-richer, shape-similar donors for a road-poor recipient (+11–55%
   neighbor street-length). It is now a reusable retrieval primitive for whichever donor type is used —
   but whiten the spectrum first (naive z-scoring is swamped by noise dimensions).
4. **Consensus-as-seed + light greedy refine — TESTED, NO-GO (2026-07-23).** Seeding the codebase's
   `LoopClosureRefiner` (the `Method.propose(block, prior)` seam) with the strong OSM consensus on
   40972: consensus+refine reaches only **0.706** permeability — below direct clearance (0.783), and
   *below the unrefined consensus itself* (0.722), because the refiner cannot dig out of the
   consensus's fragmented structure. Worse, the **same refiner started from a cheap, information-free
   clearance-partial seed scored 0.751 — higher than consensus+refine on every axis** (more
   permeability, less displacement, less road). So the GW/UOT/barycenter machinery adds *negative*
   value as a prior: you are better off seeding the refiner with nothing.

**Final verdict (2026-07-23): the arc is closed as a reblocker.** With consensus-as-seed ruled out, no
OT-transplant variant beats a cold `clearance` solve — the transplant reproduces borrowed coverage that
a from-scratch solver (or even a naive refiner seed) matches or beats. Two things survive as reusable:
(1) **OSM-footpath consensus as a *prediction* tool** — it reconstructs a block's *own* real footpaths
to ~94% from its mapped neighbors, a genuine "fill in footpaths where OSM coverage is missing"
complement to the `osm_footpaths` method (not a competitive reblocker); (2) the **GEP road-provision
decorrelation**, a validated, reusable retrieval primitive. Of the next steps above, only #1 remains
open — and only for that OSM-*prediction* use case (validating it on n>1 recipients), not for beating
a reblocker.
