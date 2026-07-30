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

## Permeability credits disconnected roads; access depth does not (2026-07-30)

The two reported quantities disagree about what a road that never reaches a street is worth:

- `derive.access.street_connectivity` seeds ONLY road components that touch a street -- "floating
  interior roads grant no access" -- so `parcel_access_layers` correctly leaves a parcel beside a
  dead fragment deep.
- `permeability` has no such rule. An isolated corridor still upgrades the local parcel-adjacency
  conductance from footpath to road, so it RAISES the score while granting no access at all.

**Found via the examples.** On `multiblock_density_compactness`, `osm_footpaths`'s lens-B prefix has
**14 road components, only 70.8% of its length street-connected**, and **all 136** of its
adjacent-but-still-deep parcels touch nothing but floating fragments. So its "reaches P* = 0.60"
rests partly on road that provides zero access. The after-image is honest -- the depth colouring is
right and the confusion is that a reader assumes any drawn road grants access.

Chiefly hits `osm_footpaths`, a fixed real-world input (OSM footpath coverage is patchy, and
clipping to the block interior severs connections at boundaries). Synthetic methods mostly build
connected networks, so it flatters the real-world baseline rather than any of our methods -- but it
is the same leniency that made disconnected lens prefixes score well before the prefix-order fix.

### Options, none taken yet

1. **Make permeability match access**: eliminate road conductance on components that do not reach
   ground. Principled -- the metric is `b^T L^-1 b` on a grounded Laplacian, so an ungrounded
   component is arguably already meaningless -- but it changes every published permeability number
   and needs its own measurement pass.
2. **Report connected fraction alongside** each method's lens rows, so a number resting on floating
   road is visible rather than hidden. Cheap, honest, no metric change.
3. **Render floating segments differently** (dashed or paler) in the after-images, so the picture
   explains itself. Cheapest, purely presentational, fixes the confusion that surfaced this.

(2) and (3) are independent of (1) and worth doing regardless.

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
