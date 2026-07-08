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
