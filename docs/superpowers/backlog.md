# reblock — backlog

Deferred ideas and threads, captured so they aren't lost. Not committed work; groom before pulling into a slice.

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
