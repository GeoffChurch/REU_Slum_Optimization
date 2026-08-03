# reblock — backlog

Deferred ideas and threads, captured so they aren't lost. Not committed work; groom before pulling into a slice.

## Cross-block reblocking (Phase 0 speced+planned; Phase 1 + red-team-cut metrics here)

The non-myopic direction: streets that span multiple blocks, stay smooth, and *cross* other streets
rather than dead-ending at them. **Phase 0** (merge_cluster + orthogonal metric basis + a falsifiable
probe) is speced/planned (`specs/2026-07-07-cross-block-phase0-design.md`,
`plans/2026-07-08-cross-block-phase0.md`) and **gates the rest**: build Phase 1 only if the probe
shows real cross-block headroom over boundary-reconciled block-local reblocking.

- **Phase 1 — cross-block placement methods.** Needs a `RegionMethod.propose(region)` path (declared,
  unused, `contracts.py:94`) + evaluation on the merged super-block. Five candidate approaches explored
  (distinct schools): planner arterials→feeders hierarchy; tensor-field streamlines; free-space medial
  axis; network-design ILP (explicit crossing/turn/displacement objective); continuous/variational
  elastica with a differentiable crossing bonus. Brainstorm → **red-team when concrete**, using the
  Phase-0 probe's headroom as ground truth. Cross-cutting risks (from the approach red-team): the
  dominant-orientation/grid assumption fails on organic settlements (needs a bearing-entropy gate);
  points-not-footprints ceilings displacement realism.
- **Arc roads + arc-aware smoothness metric.** Relax "straight" → "constant curvature" (arcs):
  penalize curvature *variation* (∫(κ′)²), which admits lines AND arcs equally. The naive
  `curvature_variation` was **cut from Phase 0** after the red-team — it reads 0 for 2-point-`LineString`
  methods, is sampling-density × micro-noise dominated (a clean arc swings 0.008→36 across sampling
  densities), and is scale-confounded. Do it right (arc-length resampling / a dimensionless bearing-TV),
  build it *with* the first arc-emitting method so it calibrates on real output, and add a junction
  continuation-pairing rule to extract polylines from branching networks. Arc rep = biarcs/arc-splines;
  arc noding is closed-form but shapely wants sampled polylines.
- **Through-going crossing refinement.** Phase 0 counts bare degree-≥4; the real "crossing" is a
  collinear pass-through — define via orientation **mod 180°** (directed bearings make a clean `+`
  score zero), bearing measured over the first L metres of each incident edge. Refine in Phase 1 where
  real crossings validate the threshold.
- **`dwellings_displaced` with real footprints.** Cut from Phase 0: building *points* under-count (a
  road clipping a parcel corner displaces 0 while destroying the home) and bias toward gap-threading
  methods; displacing a building also changes the Voronoi tessellation the eval scores (circular
  dependency). Needs real footprints, not points.
- **Spine-merge optimizer.** Phase 0 ships only a heuristic spine-merge *reference*; the real optimizer
  (merge redundant boundary-parallel spines into shared through-trunks) is a Phase-1 method.

**Phase-0 probe findings (from the built probe — these shape the go/no-go and Phase-1 eval):**
- **The metric basis is ~2D on real data, not 8D.** The correlation-matrix orthogonality check did its
  job: on the 29-cluster sample the structural tier collapses (`throughput ~ added_road_length ~
  dead_end_fraction`, all |r|>0.9) to ~2 independent directions (a tree-depth/cost axis + a weak
  cross-block axis). Prune the basis accordingly; don't treat all 8 axes as independent evidence.
- **The welfare/directness tier is DEGENERATE for peel-like methods.** `geometric_access_max_m`,
  `_p95_m`, and `circuity` read 0 / 1.0 because peel reblocking reaches every parcel (residual
  access = 0; no off-road parcels for circuity). These are vacuous, NOT evidence of optimality. A
  meaningful directness metric must measure **travel along the road network** (shortest path on the
  noded graph vs straight-line), not residual parcel access — a Phase-1-scale metric redesign.
- **`Region.blocks` one-shot-generator footgun.** `KblockSource.region()` returns a lazy generator
  (intentional for the fast single-block CLI path), so iterating `region.blocks` twice silently yields
  empty the second time (this bit the probe). Consider hardening the `Region.blocks` contract to a
  re-iterable `Sequence` — weigh against the lazy-build perf; touches `contracts.py`.
- **Robust frontage detection.** `_interior_boundaries` uses exact `.intersection().length>0` (could
  under-match sub-tol-gapped real geometry); and `_side` now length-invariantly gates on-boundary
  points — but a buffered/segment-aware frontage-and-side treatment would be sturdier for messy data.

## Visualizations (brainstorm needed — own its own brainstorm)

- **Draw the guessed building circle-footprints, and the graph the metric actually uses.** Every
  render currently shows building POINTS, but the metric and `displacement` both reason about
  *disks*: `building_radii` gives each building a radius of half its nearest-neighbour distance, and
  displacement is `clip(1 - d/r)` against that disk, while the footpath conductance is a function of
  the gap between two disks. None of that is visible, so the quantity being optimised is invisible
  too — a reader sees points and roads and has to take the corridor arithmetic on faith. Two pieces:
  (a) an overlay drawing each building's disk at its true radius, which makes displacement legible
  and would have made the r0-vs-per-pair-gap question obvious much earlier; (b) an overlay of the
  METRIC's graph itself — parcel-centroid nodes, adjacency edges, edge width or colour by
  conductance, ground edges marked. That second one is the honest picture of what permeability
  scores, and it makes visible that edges run straight through building interiors (the road-first
  mesh spec's motivating defect).

- **Region choropleth** — color every block in a region by a metric (peel-k, geometric access
  distance, population, building density, road-length-to-reblock). The first "zoom out from one
  block to the whole city" view. Needs a region-level render (many small polygons + a colorbar);
  decide static (matplotlib/GeoParquet → PNG) vs interactive (deck.gl/folium webpage artifact).
- **Distribution views of the layer sequences** — ridgeline / small-multiples of the topological
  layer sequence and the geometric distance histogram per block; a way to *see* morphology, not
  just the scalar k. Could overlay before/after (reblocking flattens the deep tail).
- **Block-similarity map** — cluster blocks by 1D-Wasserstein between their layer sequences, color
  the region by cluster. Directly visualizes the OT-similarity idea.
- **Before/after polish** — the existing `render_after` + the head-to-head webpage; a proper
  emitter with `format=webpage` and side-by-side layout (see flow-refactor spec §2).

## Retrieval may be the wrong problem: donor QUALITY is predictable, MATCHING is not (2026-08-02)

See [the note](notes/2026-08-02-donor-quality-is-predictable-matching-is-not.md). On the 500 pairs
already in `data/benchmarks/gw_pair_matrix.parquet`, with no new GW fits: cheap invariant
descriptors of the DONOR ALONE predict transplant fidelity at within-recipient rho +0.22 on held-out
recipients, against GW distance's +0.04. **But against six TRIVIAL donor numbers (+0.190) the margin
is +0.028, inside this design's noise -- the learned descriptor is not established as necessary, and
"good donor" is mostly "small and compact" (n_parcels alone reaches |rho| 0.229). Donor permeability
is among the WEAKEST predictors, -0.134.** Adding the recipient or the pairwise difference adds
nothing, 85% of donors appear exactly once so it is not memorisation, and the real score beats
30/30 permutation draws.

**This removes the motivation for the Phase 2 retrieval index rather than deferring it.** An index
answers "which donor is nearest this recipient"; the measurement says that question has little to do
with transplant quality, while "is this a good donor at all" is answerable from a per-block score
computed once, with no pairwise anything.

Small (20 recipients, Cape Town, one fidelity measure) and the permutation margin is modest, so
next steps are re-run at scale on the 65,364 provisioned blocks, then interpret the score before
building on it.

## Shape-standardizing RegionBuilder -- BUILT 2026-08-02, objective chosen empirically

`ShapeStandardizingRegionBuilder` (`src/reblock/region.py`, `conf/region_builder/
shape_standardizing.yaml`). Accretes by adjacency like `dense_cluster`, but ranks each frontier
block by `objective.score(union u candidate)` -- the shape of what is being ASSEMBLED, not of the
candidate in isolation. That one line is the whole point: dense_cluster's `sqrt(n*A)/P` is computed
per block and never looks at the union, which is why its regions came out as 150-900 parcel tendrils
whose outline is a growth artifact, the confound the Phase 3 donor-material test cannot tolerate.

**The objective was chosen against evidence, not assumed.** The spec warned that isoperimetric
compactness is "only the obvious first guess" and should not be adopted because it is familiar. It
turns out to be disqualified on a NECESSARY condition, before the GW-variance criterion is even
reached: polyomino perimeters tie constantly (a 1x3 strip and an L-tromino both have area 3 and
perimeter 8, so identical quotient), so on grid-like fabric the greedy cannot discriminate, falls
back to its `block_id` tie-break, and lands where the compact option is unreachable. Growing 4
blocks from the centre of a uniform 5x5 grid:

    isoperimetric   -> staircase,  quotient 0.503   (misses 0.785 on its OWN metric)
    rectangularity  -> 1x4 strip,  quotient 0.503   (blind to elongation by construction)
    squareness      -> the 2x2,    quotient 0.785

An objective that ties everywhere standardizes nothing. `Squareness` (rectangularity x the rotated
bounding box's aspect ratio) is the default on that basis; it also keeps the fabric's own
orientation, since the rectangle is rotated rather than axis-aligned.

**Still open, and it is the spec's actual criterion:** whether squareness beats the alternatives on
REAL fabric, measured as the outline's share of inter-region GW distance variance using the pair
matrix. The finding above only rules out the familiar quotient on a synthetic grid. `objective` is a
live Strategy precisely so that comparison can be run without touching the builder.

**Not yet done:** nothing calls this builder in a shipped path -- the Phase 3 donor-material test it
gates has not been written.

## Method lineup

- **Drop `greedy_arterial_repulsion` from the examples if it does not earn its place** in whatever
  final eval we settle on. It costs **37.6 s/block against 0.01-0.47 s for everything else** (~80x),
  which at region scale likely dominates the whole examples regeneration, and it currently trails on
  both reported lenses (permA 0.6650 / dispB 0.1262, last and second-to-last of five). Decide from
  the final eval, not from this one bakeoff.
- **`cycle_native` is NOT in the examples and should stay out for now.** It beats `clearance_looped`
  on Lens B (12/12, -0.0326) and ties it on Lens A, but loses to `resistance_lp` on both (Lens A
  1/14, -0.0696). Its distinguishing property -- near-zero bridge fraction, the only method that
  builds real circulation -- is invisible in a report that shows permeability and displacement only,
  so it would read purely as a dominated sixth method. Revisit if circulation ever becomes a
  reported axis (see the orientation-penalty section below, which says it cannot come free).

## Metric / research

- **Validate the geometric access metric** against something real (reachability, hand-labels, or
  correlation with kblock's k *on a fixed building source*). peel-k is confirmed ≈ √(building count)
  on Voronoi (a count proxy, not morphology); the geometric shortest-path-distance metric is the
  intended fix — but it too should be validated, not assumed.
- **Nav-mesh geometric access** — route the shortest path *around building footprints* (a real
  walkable path), not just through the parcel-adjacency graph. Fuller "real access"; needs a
  navigation mesh. The weighted-graph Dijkstra is the cheap first cut.
- **OT-inspired methods** — block similarity via Wasserstein over layer sequences; transport-based
  reblocking; Wasserstein barycenters of block morphologies; retrieval of similar blocks. The layer
  sequence + raw distances (emitted by the eval) are the substrate.
- **Voronoi vs real parcels** — the √count artifact is a Voronoi-regularization effect; real
  cadastral/footprint parcels retain morphology kblock's k captures. Revisit whether a
  morphology-preserving parcelization is worth it, if the geometric metric doesn't suffice.

## Lens prefix selection — the cheapest connected subnetwork (2026-07-29)

Both lenses truncate a method's roads to a PREFIX of one canonical order (`budget.street_first_ordered`)
and score that. The question the lens is really asking is **"what is the cheapest buildable
subnetwork of this proposal that reaches the target?"** — an optimization problem. A fixed ordering is
only a heuristic for it, and the heuristic's choice measurably changes conclusions, so it deserves to
be attacked properly.

**Why it matters:** on the depth region a wrong ordering moved `resistance_lp` from 1,951 m to
23,490 m to reach P\* = 0.60 (12×) and flipped `resistance_lp` vs `clearance_looped` from domination
on both axes to a Pareto trade. Nothing about the roads changed.

### Settled so far (don't re-derive)

- **The old order was broken, not merely suboptimal.** Plain drainage-descending IS a valid
  topological order for a drainage *tree*, so it held while every method was one. It fails on loops
  and where a later path branches off the middle of an earlier one (which clearance does). Measured
  fraction of lens-B prefix length reaching the street under it: `greedy_arterial_repulsion` 0.782,
  `resistance_greedy` 0.900, `clearance_looped` 0.831 (region scale). **Permeability credits
  disconnected fragments** — an isolated corridor still upgrades local adjacency conductance — so the
  lens was scoring road sets nobody could build.
- **Two replacement orderings were tried and are WRONG**, both erring the same way (making a method
  look costlier than a set it demonstrably achieves):
  - *distance from the street, ascending* — guarantees connectivity but is breadth-first: completes a
    ring before going deeper, penalizing granularity rather than geometry. Moved `resistance_lp` on
    one region from 2,236 m / 0.0403 to 5,104 m / 0.0630.
  - *highest drainage that merely TOUCHES the built set* — free at block scale, 12× too loose at
    region scale (the 23,490 m figure above).
- **What shipped:** drainage order with each road's **connectors bought on demand**. Reduces exactly
  to the old order whenever the old order was already buildable, so only broken prefixes changed.
  Every method now measures 1.000 prefix connectivity on 12/12 blocks. The method that had been
  cheating pays an honest premium (`clearance_looped` +7% metres / +8% homes on depth); methods
  already valid are bit-identical.
- **It is deliberately not a Hydra Strategy.** Two of three candidates are wrong rather than
  different, so a plug-in point would ship one implementation and a menu nobody selects (the
  no-dead-options rule). A real optimizer WOULD earn one — see below.
- **`road_drainage` semantics were fixed alongside** (it counted segment traversals, not parcels, so
  vertex-dense roads got inflated drainage — a subdivided road scored 4 against an identical plain
  road's 1). Any future ordering work inherits the corrected key.

### Headroom measured 2026-07-30: real on the curve, mixed at a threshold

`scratchpad/ot/prefix_headroom.py` compares today's drainage chain against a greedy chain on
marginal permeability per marginal displacement -- same road set, same connector-on-demand rule, so
it is budget-matched by construction. 6 blocks, 24 method-rows:

| method | disp to P* today | greedy | change | curve-area change |
|---|---|---|---|---|
| greedy_arterial_repulsion | 0.1100 | 0.0395 | **+55.9%** | +9.8% |
| resistance_lp | 0.0376 | 0.0339 | +16.4% | +16.6% |
| clearance | 0.1049 | 0.1139 | **-7.7%** | +2.8% |
| clearance_looped | 0.1108 | 0.1215 | **-8.0%** | +4.5% |

**Whole-curve area improves on 23/24 rows** (median +7.3%), which is the measure the frontier plot
and the GIFs actually display. **The P* crossing is mixed** -- big gains for arterial and the LP,
~8% WORSE for the two clearance variants. That split is structural, not noise: the greedy optimizes
value-per-cost at every step, so it shapes the whole curve, but nothing makes it optimal at one
chosen threshold. A chain taking cheap high-value roads early can plateau just under P* and then
need more displacement to cross.

**So "better order" is under-specified until we say better AT WHAT.** Optimizing the curve and
optimizing a threshold crossing are different objectives with different winners, and lens B reports
the threshold while the plots show the curve.

### Bake-off 2026-07-30: a ONE-SOLVE static order captures most of the gain

`scratchpad/ot/order_bakeoff.py`, five chains on the same road sets, 6 blocks. Gain vs today's
drainage order, with the solve count each needs:

| order | arterial | clearance | looped | lp | solves |
|---|---|---|---|---|---|
| greedy (re-solve every commit) | **+7.4%** | **+2.7%** | **+4.5%** | +18.3% | 19-103 |
| **static** (one solve on the full net, sort once) | +5.6% | +1.3% | +2.4% | **+18.1%** | **1** |
| neartie (batch near-ties in the upper bound) | +4.3% | +1.4% | +2.3% | +15.5% | 8-11 |
| runs (continue to a fork/terminus) | +6.7% | +1.3% | +2.7% | +14.9% | 15-97 |

(AUC = area under the permeability-vs-displacement curve, the thing the plots and GIFs show.)

**`static` is the value pick: 50-99% of the full greedy's gain for ONE solve.** On `resistance_lp` --
4,451 roads at region scale, where solve count is the whole problem -- it captures 99% of the benefit
(18.09 vs 18.27) at 1 solve against 103. Its weakness is the threshold measure, where it can regress
badly (clearance -23% on displacement-to-P*), while the full greedy was positive on every method
there (+0.8% to +22.7%).

Two negatives worth keeping:

- **Run-following does not work here.** It cut solves barely (31 vs 32.7 for arterial, 97 vs 103 for
  the LP) and scored slightly worse. The fork/terminus condition fires almost immediately on real
  road sets, so runs are short. It is not the granularity equalizer it looked like.
- **`neartie` is dominated by `static`** on both quality and cost, despite being the more principled
  construction (the linearized gain is a certified upper bound and submodularity keeps it valid, so
  a near-tie batch needs no re-solve). Principled did not beat cheap.

**The cost denominator must be DISPLACEMENT, not length.** A first run of this bake-off used road
length -- chasing CELF-safety, per `budget.repulsion`'s docstring -- and EVERY ordering then lost to
plain drainage (full greedy -3.4% AUC, having been +7.3% with displacement cost). Displacement is
the curve's x-axis, so per-metre optimizes a ratio unrelated to what is plotted. The CELF motivation
was void regardless: CELF removes per-candidate evaluations, which the linearization already makes
free, while the bottleneck is the number of ROUNDS.

### CLOSED 2026-07-30: do not roll out a new order. Four hypotheses, none explains the failure

Widened to 20 blocks and the full example method set (n=303 rows), the n=6 conclusion reversed in
BOTH directions -- the `euclidean_grid` regression vanished (-1.6% -> +1.3% AUC) and a new one
appeared on the FLAGSHIP:

| method | static AUC | greedy AUC | static lens-B | greedy lens-B |
|---|---|---|---|---|
| resistance_lp | +14.9% | +17.9% | +21.3% | +27.8% |
| greedy_arterial_repulsion | +4.4% | +6.9% | +28.0% | +28.5% |
| euclidean_grid | +1.3% | +1.3% | -2.4% | -0.2% |
| osm_footpaths | 0.0% | 0.0% | -- | -- |
| clearance | **-1.4%** | +2.7% | **-13.8%** | -3.3% |
| clearance_looped | **-1.9%** | +1.4% | **-8.6%** | -1.7% |

`static` regresses the flagship on both measures, so it must not ship. `greedy` is non-negative on
AUC everywhere but costs 18-112 solves per curve to give the flagship +1.4%. **The gains are
concentrated in `arterial` and `resistance_lp`** -- the two methods whose drainage order is worst
matched to their structure -- and `osm_footpaths` is exactly 0.0% under every ordering. That is a
poor trade for a change that touches the evaluation core and rewrites every published number.

**Four hypotheses for the flagship regression, all refuted by measurement:**

1. *Wrong operating point* -- gains scored at the full network while the chain starts empty. Refuted:
   the key correlates with true single-road gain at rho 0.42-0.68 either way, and `static_empty` /
   `static_avg` measured no better than `static_full`.
2. *Front-loading* -- a gain key picks a deep leaf and drags its whole trunk chain in at once.
   Refuted: static's first commit displaces LESS (0.015 vs 0.035) with SHORTER chains (0.44 vs 1.10).
3. *Drainage captures enabling value* the first-order gain cannot see. Refuted: drainage correlates
   with essentially nothing -- rho -0.08..0.18 against solo gain, -0.62..0.15 against the gain of the
   road plus its whole connector chain.
4. *Infinite-key ties* -- roads with zero marginal displacement tie at `inf` and argsort breaks the
   tie by index, i.e. arbitrary emission order. A real defect, but it hits 19.3% of `resistance_lp`'s
   roads (which wins anyway) and only 1.1% of `clearance_looped`'s (which regresses), so it does not
   explain this.

**The meta-finding, which is the useful part.** Hypothesis 3 exposes an assumption every ordering
idea here rested on: that a per-road key correlating with per-road value yields a better CHAIN. The
data denies it -- drainage correlates with nothing yet beats a key correlating 0.39-0.74 with solo
value. Chain quality is a property of the whole sequence, and per-road correlation does not predict
it. **Any future attempt should first characterize empirically what distinguishes a good chain from a
bad one** -- take a block where two orders diverge sharply and examine what actually differs -- before
proposing another key. Four keys were proposed and tested here without that step.

Also note the n=6 -> n=20 reversal in both directions: several probes in this thread ran at n~6-12,
which is not enough for a method-level ordering claim.

### Exactness is the wrong target

Two independent reasons, both worth recording so it is not attempted:

- "Cheapest connected subnetwork reaching P*" contains Steiner tree, so it is NP-hard even with the
  road set fixed.
- More decisively, **the exact answer is not nested.** The optimal subset at budget b1 need not be a
  subset of the optimal at b2, so the true Pareto frontier over subsets is not a CHAIN -- and the
  curves and GIFs require a chain (roads appear and never disappear). The right target is therefore
  the best chain, and the frontier-minus-chain gap is a price deliberately paid for interpretability,
  not a defect to remove.

### Cost, which decides feasibility

The greedy chain is O(R) `egress_power` solves (one per round, all candidates then scored O(1) by
the first-order sensitivity, as `resistance_greedy` does). Fine at block scale -- R is 27-105 here --
but `resistance_lp` emits 4,451 roads on a region, so O(R) solves is out of reach. A chunked variant
(commit the top-k per round, re-linearize) brings it to O(R/k) and is the only version that can ship.
Today's order costs one sort plus an O(log R) binary search, so this is a real cost increase that has
to be justified by the curve gain.

### The actual open problem

Minimize cost (displaced homes, or metres, or both) over **connected** subnetworks of a proposal
subject to permeability ≥ P\*. Notes toward it:

- This is a Steiner-tree-flavoured problem — connectivity constraint plus a submodular-ish benefit —
  so exact solution is out, but the LP-over-paths formulation in
  `notes/2026-07-29-lp-route-a.md` is directly reusable: decide over paths, not edges, and
  connectivity becomes structural (`x_p <= z_s`) rather than a repair.
- **Only then does a Strategy seam make sense**, with genuinely live alternatives: the shipped greedy
  as the cheap default, an LP/beam-search for accuracy, selected per run.
- Beware granularity: methods emitting thousands of short segments and methods emitting a few long
  paths must be treated even-handedly. An earlier equalize-to-3m control showed granularity alone is
  worth a third of one measured advantage.
- Sanity oracle worth reusing: `clearance` builds a drainage tree, so any road-set property a tree
  satisfies by construction must return the trivial answer for it. That oracle caught a broken
  connectivity instrument in this thread after it had produced three wrong conclusions.

### Also still open, from the same thread

- **Lens A (matched displacement) has no road-length budget.** It caps homes and says nothing about
  metres, and the two are not proportional, so an optimizer buys permeability with metres: 42,937 m
  against `clearance_looped`'s 9,878 m at the same displacement, for a capillary web nobody would
  build. Mitigated for now by demoting it to *secondary* in the generated READMEs with the caveat
  stated in its own table copy; matched permeability is primary because it prices both costs in their
  own units. A composite `homes + lambda*metres` budget was deliberately NOT built — lambda is a
  values question, not a measurement one. An in-objective length *price* on the method side was built
  and deleted: too weak where it was needed (still 33,623 m at price 16) and a pure loss at block
  scale.
- **The committed examples are stale** and need regenerating — `clearance_looped`, `arterial` and
  `resistance_greedy` prefixes all shift under the corrected order and drainage key.

## Permeability vs access depth on disconnected roads -- and the all-pairs alternative (2026-07-30)

### What ground actually is (settle this before reasoning about the rest)

`egress_power` grounds parcels within `STREET_TOL` of **`block.streets` -- the PRE-EXISTING street
network, never the method's own added roads.** Added roads only upgrade edge conductance
(footpath -> road) along their corridor; they never create a new exit.

For a single block `streets` is essentially the boundary, so "ground = the boundary" holds. **For a
multiblock region it does not**: `region_block` unions every member block's streets ("perimeter +
inter-block = the full existing road network"). Measured on `multiblock_density_compactness`
(18 blocks / 4,615 parcels): 1,059 grounded parcels, of which **only 389 sit on the outer boundary
and 670 are grounded by interior inter-block street**. Most ground in a region is interior.

### The inconsistency, and why the obvious fix is backwards

- `derive.access.street_connectivity` seeds only road components that touch a street -- "floating
  interior roads grant no access" -- so `parcel_access_layers` leaves a parcel beside a dead
  fragment deep.
- `permeability` has no such rule: an isolated corridor still upgrades local adjacency conductance
  and so lowers dissipated power.

Found via the examples: `osm_footpaths`'s lens-B prefix on `density_compactness` has **14 road
components, only 70.8% of its length street-connected**, and **all 136** of its
adjacent-but-still-deep parcels touch nothing but floating fragments. Its "reaches P* = 0.60"
therefore rests partly on road that grants no access. The after-image is honest; the confusion is
that a reader assumes any drawn road grants access.

**An earlier revision of this entry proposed "make permeability match access". That is backwards.**
A footpath linking three interior parcels genuinely does help you move -- you can use it to reach a
parcel that IS near a street, and you still pay the footpath resistance for the last leg.
Permeability's continuous-resistance model captures that faithfully; access depth's binary
touches-a-street-or-not is the cruder abstraction. The disagreement is real, but permeability is the
better model of the two, so the fix is not to make it stricter.

Worth doing regardless of anything below, both cheap and independent:

- **Report connected fraction alongside each method's lens rows**, so a number resting on floating
  road is visible rather than hidden.
- **Render floating segments differently** (dashed or paler) in the after-images, so the picture
  explains itself. This is what would have prevented the confusion that surfaced all of it.

### The real alternative: all-pairs (Kirchhoff index)

Today's metric is all-to-ground: `b = 1`, every parcel injects one unit of escape current, all of it
flowing to the street. The alternative is **total effective resistance**, where every parcel wants to
reach every OTHER parcel:

    R_tot = sum_{i<j} R_ij = n * trace(L^+)

on the UNGROUNDED Laplacian's pseudoinverse.

| | current (`egress_power`) | all-pairs (Kirchhoff) |
|---|---|---|
| model | all current -> street | every pair exchanges current |
| needs a ground | yes | **no** |
| rewards | getting OUT | getting AROUND |
| floating road linking A-B-C | helps a little (better routing toward ground) | helps properly: R_AB, R_BC, R_AC drop, and correctly NO help reaching the street |
| convex in edge conductances | yes | yes -- Ghosh-Boyd-Saberi state it for exactly this |
| monotone under an added road | yes (Rayleigh) | yes (Rayleigh) |

Three things recommend it beyond the question that prompted it:

1. **It is exactly what Ghosh-Boyd-Saberi optimize**, so the convexity result the route-(A) LP leans
   on applies natively rather than by analogy.
2. **It needs no ground**, which dissolves the entire "what counts as street" question above --
   including the region-vs-block asymmetry.
3. **It would capture internal circulation**, which no shipped metric sees -- the "Bermuda triangle"
   livability concern already in this backlog.

**Costs and open questions, not yet resolved:**

- `trace(L^+)` needs all eigenvalues or n solves against today's single sparse solve. Tractable
  exactly at n ~ 4,600; at n ~ 11,000 it wants the standard Spielman-Srivastava / Hutchinson
  estimator, which introduces sampling error into a reported metric.
- **Pure all-pairs is probably wrong on its own**: a settlement where everyone reaches everyone but
  nobody reaches the arterial scores perfectly. The honest version is likely a COMBINATION of egress
  and circulation, which reintroduces a weighting question of the same kind deliberately avoided for
  `homes + lambda*metres`.
- It would change every published permeability number, so it needs the brainstorm-then-spec
  treatment, not a patch. Start from `specs/2026-07-22-permeability-metric-design.md`.

**PROBE RUN 2026-07-30 -- closed as a replacement.** At matched displacement over 12 blocks,
all-pairs ranks methods essentially as permeability does: Kendall tau median +0.800, same winning
method 10/12, `resistance_lp` first on both. Changing every published number to reproduce nearly the
same ranking is not worth it.

Two findings kept: roads buy far more egress than circulation (permeability improves 0.65-0.85 at
D = 10% while all-pairs resistance falls only 0.35-0.64), which is the "Bermuda triangle" intuition
as a number and makes all-pairs worth a cheap DIAGNOSTIC column even though it is redundant as a
ranker; and the LP leads on both metrics, so it is not exploiting egress-specific structure. Full
reasoning and the probe's own methodological trap: `notes/2026-07-30-egress-vs-circulation.md`.

## Road-first mesh -- SPEC'D 2026-07-30, not built

`specs/2026-07-30-road-first-mesh-design.md`. Came out of the one-way thread but does NOT depend on
it. The motivating finding: **`permeability` does not model roads, it models which parcel pairs are
near one.** Nodes are parcel centroids, edges are parcel adjacencies, edge length is the crow-flies
centroid distance, and a road enters through a single boolean (does its corridor intersect that
centroid segment).

Four defects follow; the first three have nothing to do with one-way and are the reason to build it:

1. Road length and shape never enter the metric -- two roads covering the same parcel pairs score
   identically, so only `displacement` distinguishes a long winding road from a short straight one.
2. Travel distance is crow-flies parcel-to-parcel, not distance along the road.
3. Road-network connectivity is invisible, which is why a floating fragment still raises
   permeability (the `osm_footpaths` inconsistency: 14 components, 70.8% street-connected, all 136
   adjacent-but-deep parcels touching only floating road).
4. No capacity or hierarchy -- every covered edge gets the same `g_road / d`. This is the one-way
   prerequisite, and it is deliberately OUT OF SCOPE in the spec.

Acceptance criteria are written before implementation, because the all-pairs probe closed a metric
change for reproducing the same ranking at real cost: the ranking must actually move (Kendall tau vs
today over >= 20 blocks; tau near +1.0 is a stop signal), the conductance parameters must be
RECALIBRATED for road-edge length scales rather than inherited, and the resilience/one-way ideas stay
out.

### N-1 resilience: tried as the cheap alternative, INCONCLUSIVE

Tested whether `min over roads r of permeability(S - r)` could reward redundancy inside today's
metric, avoiding the mesh entirely. Un-equalized it looked strongly discriminating (`keep` 0.245 to
0.944) but that was pure GRANULARITY -- at D = 10% `resistance_lp` emits 111 roads and
`euclidean_grid` 2.5, so one removal is a scratch for one and an amputation for the other -- and it
failed its own decisive check (`clearance_looped` no better than its own tree base, 2/8 blocks,
median +0.0000). Re-run with all roads subdivided to 5 m the spread collapsed to 0.86-0.98, but the
run was killed after 3 blocks and 2 of those had the two methods produce coincident networks, so
there was effectively one informative block. **Not refuted, not supported -- inconclusive.** Redo at
n >= 20 with a budget where the loop refiner's loops actually exist, or with a synthetic same-length
tree-vs-loop pair, if the redundancy question is worth reopening.

## Width/direction solver layer -- PROMISING, and its premise is refuted (2026-07-31)

Owner's proposal: methods emit undirected topology only; a solver layer between method and eval
assigns per-road width and direction; the eval scores the solved network. Methods stay simple and
the solver figures out where thin one-way road pays.

**The decision variable is a 4-element lattice** -- the powerset of {forward, backward} ordered by
inclusion, i.e. don't-build at the bottom, the two one-way directions incomparable in the middle,
two-way on top. That is the right algebra and not merely a convenient picture, because BOTH benefit
and cost are monotone up it, and the middle->top step is pure addition: forward capacity is
IDENTICAL at one-way and two-way (20.0 either way, verified), so climbing adds the reverse direction
and 3 m of width without trading anything away. That falls out of `one_way_width` being defined as
the equal-per-direction-capacity width.

Consequences:

* **Truncation and flipping are the SAME decision** -- width 0 is "don't build" -- so this subsumes
  the open "cheapest connected subnetwork subject to P* >= target" problem below, retiring the
  arbitrary prefix ordering at the same time.
* **There is NO Robbins feasibility constraint.** An earlier note here said orientability is
  endogenous and needs a fixed-point iteration; that is WRONG. The reverse direction of a one-way
  road falls back to footpath conductance, never zero, so nothing disconnects and the metric prices
  a bad orientation by itself.
* **Connectivity is upward-closed** (climbing only adds roads), so the feasible set is a filter.
* The general form is N^2 -- (forward lanes, backward lanes) componentwise, width =
  margin + (f+b)*LANE -- and the 4-element lattice is {0,1}^2. Note this does NOT reopen the
  quantization question: continuous width is right for SCORING a road someone hands you (extra width
  buys robustness to a blocking vehicle), integer lanes are right for the solver's CHOICE SET
  (you build whole lanes). Different questions.
* Lives per SEGMENT, not per road; `orient.strong_orientation` already splits at those boundaries.
* NOT on the lattice: couplets (a parallel one-way pair replacing a two-way road) -- new geometry,
  so a method move rather than a solver move.

**BUILT AND MEASURED 2026-08-02: it does NOT pay. Do not build it again without a new mechanism.**

Block-scale go/no-go looked good (`scratchpad/width/flip_vs_truncate.py`, 24 blocks x 3 methods):
FLIP beats TRUNCATE at equal displacement on **54/72, median +0.0193, p=0.0020**. A real solver was
then written (four-element lattice per road, `linearized_gain` ranking, batched commits) and run
against the lens's own prefix truncation on the 4,615-parcel density_compactness region:

    promote from `don't build`   solver better on  6/12   median -0.0134
    demote  from two-way         solver better on  3/12   median -0.0266

**Cost is fine** -- 1.6 s median, 30 s worst, at 4,615 parcels; the one-solve-per-batch trick works.
This is a value failure, not a feasibility one.

**Why, and it is the same reason both ways.** The lattice's MIDDLE elements are dominated in both
search directions. Promoting, one-way offers 0.5x the gain for 4/7 = 0.57x the cost, so two-way
always wins gain-per-metre. Demoting, narrowing forfeits half the gain to save a little width while
DROPPING forfeits all of it to save everything, and dropping wins the ratio. One-way was chosen 0-17
times out of 8-3,522 pieces. With the middle unused the solver degenerates into road SELECTION --
and as a selector it loses to `street_first_ordered` at every budget.

**So the block-scale result does not translate, and it is worth understanding why.** That probe
FORCED the topology to stay and only allowed narrowing. Given the freedom to drop a road instead,
the optimizer drops. The honest reading of +0.0193 is therefore "IF you must keep every road,
narrow rather than truncate" -- a constraint someone might impose for access reasons, not an
optimization this metric rewards.

**What would have to change** for this to be worth revisiting: a benefit term that values keeping a
road at all (coverage, worst-case parcel access, N-1 style), because permeability alone is happy to
delete a road and spend the savings elsewhere. Absent that, the middle of the lattice has no reason
to be selected and the layer has nothing to add.

**Code status: `src/reblock/width_solver.py` is IN THE TREE AND UNUSED** (owner's call 2026-08-02),
kept so a variant can be tried without rewriting ~200 lines. Nothing imports it outside
`tests/test_width_solver.py`. It is a wart under the repo's own no-legacy rule for as long as that
stays true -- **delete it if the variant above is not attempted.**

**But the premise is refuted.** The expectation was that loopy networks benefit most. They do not:

    clearance_looped           +0.0269   24/24   p=1.2e-07     (0.37 orientable)
    greedy_arterial_repulsion  +0.0236   21/24   p=2.5e-05     (0.79 orientable)
    cycle_native               -0.0208    9/24   p=0.065       (0.50 orientable)

`cycle_native`, the only genuinely circulating method, is the one place flipping LOSES. Likely
because the benefit comes from SLACK, not loops: `clearance_looped` carries 487 m of road where ~72 m
reaches the lens target, so there is surplus to flip cheaply, while `cycle_native`'s 230 m is all
load-bearing and halving any road's reverse capacity bites. (At n=8 `cycle_native` read 5/8 POSITIVE
-- the small-n instability this thread keeps producing. Use n >= 24.)

So the layer is worth speccing as a general improvement, but NOT on the grounds that it rewards
circulation -- it does the opposite. Feasibility looks fine: `resistance_greedy.linearized_gain`
already gives one solve per round plus O(1) per candidate, which is what a solver over hundreds of
roads needs; the probe's ~8 exact solves per step would be hopeless at region scale.

**Open modelling wrinkle:** the footpath fallback means "go the wrong way at walking pace" -- right
for pedestrian egress, wrong for vehicle access, since you cannot drive the wrong way up a one-way
street. That is why there is no hard constraint, and it means one-way is probably UNDER-penalised
for vehicle-dependent trips.

## Can permeability price CIRCULATION on its own? NO -- tested 2026-07-31

The appealing idea: a network you can force one-way and still use is a circulating one, so the
ORIENTATION PENALTY `permeability(two-way) - permeability(oriented)` might price circulation from
inside permeability, keeping the reported basis at permeability + displacement with no third axis.

Tested on 12 blocks x 5 methods with width pinned at the two-way floor so only DIRECTION varies
(`scratchpad/width/orientation_prices_circulation.py`). It fails three ways:

1. **The shipped orientation rule provably cannot do it.** `orient.strong_orientation` leaves bridges
   two-way (Robbins), so a pure tree has nothing to orient and pays EXACTLY ZERO -- `euclidean_grid`
   and `resistance_lp` both 0.0000 -- and the correlation with bridge fraction comes out NEGATIVE
   (rho = -0.36). Keeping one-way as it was built would have bought nothing.
2. **Mandatory orientation gets the sign right and the ordering wrong.** rho = +0.49, but
   `clearance_looped` at 58% bridges pays the LEAST (0.0027) while `cycle_native` at 0% bridges pays
   9x more (0.0242). The penalty charges DETOUR, and detour scales with loop size, so it punishes
   big loops -- precisely the circulation it is meant to reward. Same mechanism the
   [one-way note](notes/2026-07-30-oneway-half-width.md) flagged as making the width discount
   exploitable, reappearing with the sign flipped.
3. **It is not a metric.** The scorer must pick an orientation, and the choice moves the score: DFS
   from the street vs BFS from the highest-degree node swings `cycle_native` by 0.1461, **6x its own
   penalty**, and exceeds 25% of the penalty on 29/60 rows. Trees are orientation-invariant (every
   edge is a cut edge, so egress uses it one way and ingress the other and the average is fixed --
   `resistance_lp` swings 0.0000); looped networks are not. So the measure is deterministic exactly
   where it is uninformative and non-deterministic exactly where it would be informative.

**Watch out:** an earlier version of this probe reported swing 0.0000 everywhere and looked clean.
The "alternative" orientation was the global REVERSE, which (egress + ingress)/2 is invariant under
by construction, so it could only ever report zero. A determinism check has to compare two genuinely
different orientations.

**What survives.** `orient.bridge_fraction` is deterministic, cheap, and directly measures whether a
network can circulate at all (`cycle_native` 0.000 / `clearance_looped` 0.577 / `resistance_lp` 1.000).
If circulation is ever worth a third reported axis, that is the candidate -- but it must first face
the attack that retired cycle density: is it gameable? It is length-weighted, so a token loop should
not move it much, but that is an argument, not a measurement.

## Deferred design (from the flow-refactor + peel-reblocker threads)

- **Structured configs (dataclasses + ConfigStore)** — deferred out of the flow refactor; do once
  the config surface (source + emitter + sweep shapes) stops moving. Note the sharp edges
  (`CRS|int|None` → `Any`, interpolation → `str`, polymorphic `data` node vs struct-mode).
- **joblib / disk caching** — only if sweeps ever get big enough to need cross-process/resumable
  runs. If so: **content-addressed keys** (hash of block geometry, not a positional `block_id`) and
  an **automatic source-hash version** (not a human-bumped tag). Otherwise in-process memoization.
  - **Voronoi tessellation is safe to cache** (verified pure/deterministic — what makes the
    pinned-value test stable) — but the measurement showed Voronoi is only ~6% of a block-run, so
    caching must also cover the method-independent before-derivations + method proposals (the real
    cost; topology's `propose` is minutes). **Now folded into the flow-refactor spec §6 (in-scope) as
    the L2 per-block persistent cache.** Two keying options weighed there: (a) reprojected-geometry-WKB
    hash → PROJ auto-folds, per-block-granular invalidation, costlier lookups; (b) `block_id` + raw
    source-file hash (owner's lean, adopted) → simpler, but coarse whole-source invalidation and
    **both** GEOS *and* PROJ versions must be explicit in the key. See spec §6 for the adopted design.
- **Peel-reblocker budget sweep + trunk-merging** — budget = a swept `PeelReblocker` param (no
  internal DSL); trunk-merging to make the peel spine length-competitive with topology. Retains the
  downward-closed / monotonic-prefix truncation correctness the peel Slice-2 red-team flagged.
- **Slum-detection component** — a filter/screen over a region (density + k + population thresholds)
  that selects informal-settlement blocks; the general-purpose version of the fixture density
  criterion. Handles the rural-block problem at region scale.
- **Continental scale** — multi-UTM-zone or equal-area CRS handling (fail-loud on a multi-zone
  extent); `git-lfs` (or a downloader instead of committed fixtures) once datasets multiply.

## Licensing / data hygiene (see kblock-source spec)

- Add a top-level `LICENSE` (Apache-2.0 recommended, pending owner confirmation vs GPLv3).
- Resolve the kblock Dataverse CC0-vs-ODbL contradiction with the depositor; treat ODbL as binding
  until then. `NOTICE` + `PROVENANCE.md` for Open Buildings (CC-BY) and kblock data.
