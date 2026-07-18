# True Access-Depth Everywhere (proxy confined to the cheap gate) — Design

**Status:** design approved in principle (2026-07-17), pending spec review
**Author:** owner + Claude
**Supersedes:** the squared depth-proxy screen coloring added in PR #9 (`emit._screen_proxy`).

## 1. Goal

Confine the cheap depth proxy `√(n·A)/P` to the **one** place it is load-bearing — the screen's
cheap pre-filter over all 83k metro blocks — and use the **true** BFS peel access-depth
(`access_before(block).max()`) everywhere else it currently leaks: the region builder's growth
metric, and both map colorings (`screen.png`, `region.png`). The screen map is recolored by true
access-depth on the absolute 0–max ring scale, with screen-deselected blocks blanked.

## 2. Why

The proxy is a **cheap estimate for un-peeled blocks**. Its only justification is avoiding the
expensive Voronoi+peel over all 83,192 blocks. Once a block has been peeled (every screen survivor,
and any block the region builder examines), its true access-depth is available and memoized, so
showing or ranking by the estimate instead is both less accurate and, for the map, misleading:

- The proxy over-colors large, low-density, compact blocks (empirically: 1,069 large sparse flagged
  blocks sit at the 74th proxy percentile though their true peel depth is only ~4). True depth on the
  absolute 0–max scale renders them correctly pale, so only the genuinely deep fabric (15–24 rings —
  the real reblock targets) pops.
- The region builder grows by the proxy even though it could rank candidates by true depth.

**The cheap gate keeps the proxy** — peeling all 83k blocks is exactly the cost the proxy avoids
(a ~24 s screen would become hours). That gate is the sole sanctioned proxy use.

## 3. Design

### 3.1 One BATCHED true-depth accessor

`block_depths(source, block_ids) -> dict[str, float]` — build the given blocks in **one**
`KblockSource(block_ids=block_ids).region()` call (from `source`'s `blocks_path`/`buildings_path`) and
return `{block_id: access_before(block).max()}`. **Batching is load-bearing:** `KblockSource.region()`
reads and spatial-joins the *entire* buildings parquet (~49 MB) on every call regardless of
`block_ids` (kblock.py:109,119), so a per-block accessor pays that ~2.7 s read *per block* — profiled:
one block = 2.7 s, three blocks in one call = 2.7 s *total*. The screen's fine pass is fast precisely
because it peels ~900 blocks per read; this mirrors that. Both the Voronoi build (`_voronoi_impl`) and
`access_before` are `derive()`-memoized, so already-peeled (survivor) blocks are cache hits.
Blocks that can't be built/peeled (below `min_buildings`, bad geometry) are simply absent from the
returned dict — callers default a missing id to `0.0` (never wins a "deepest" argmax). Returns `{}`
for a non-peel-capable source (no `blocks_path`, i.e. not a `KblockSource`).

### 3.2 Region builder → true depth (peel non-flagged candidates on demand)

`DenseClusterRegionBuilder.build` currently ranks adjacent candidate blocks by `_depth_proxy(count,
area, perim)`. Replace the growth metric with true max access-depth via an injected
`depth_fn: Callable[[str], float]` that is a **pure lookup into a pre-computed depth map** (no peeling
inside the growth loop — that would re-read the buildings per candidate):

- `RegionBuilder.build(block_geoms, groups, depth_fn=None)` — `depth_fn(bid)` returns the block's true
  max access-depth (a dict lookup). The builder keeps using `block_geoms` for **adjacency** and the
  **building-count budget**, but ranks candidates by `depth_fn(bid)` (ties by `building_count`, then
  `block_id` — unchanged). `convex_hull` / `identity` builders accept and ignore `depth_fn`.
- **The pipeline precomputes the depth map ONCE** (`build_regions` → `_region_depth_map`): the
  survivors' depths come free from the screen (`selection_depths`, a dict lookup for the ~14k flagged
  blocks); the reachable **non-survivors** are peeled together in a SINGLE `block_depths` call (one
  buildings read). The reachable set is a BFS from the seed groups over block adjacency, accumulating
  `building_count` to a generous multiple of the growth budget (≈ 3× `max_buildings`) so it locally
  covers everything the greedy growth can reach. `depth_fn = lambda bid: depth_map.get(bid, 0.0)`.
- **Growth is NOT restricted to flagged neighbors** (owner directive): non-survivors within the BFS
  bound get their true peeled depth; a non-survivor beyond the bound defaults to `0.0` (shallow) —
  harmless, because non-survivors failed the depth gate and are essentially never the deepest pick,
  and survivors (the real growth targets) are always covered by `selection_depths`.
- For a source that can't be peeled (geometry-only shapefile), the pipeline passes `depth_fn=None`
  and the builder falls back to the existing `_depth_proxy`. The proxy code in `region.py` stays only
  as this fallback. Delete nothing from the proxy path in `region.py`.

### 3.3 Screen exposes the depths it already computes (protocol unchanged)

The fine pass computes `(block_id, max_depth, mean_depth)` per survivor and discards the depths,
returning only ranked ids. Stop discarding — **without touching the `Screen` protocol** (so
`IdentityScreen`, which has no depths, is unaffected):

- `screen_selection` / `_compute_selection` / `_screen_selection_impl` return `list[tuple[str,
  float]]` — `(block_id, max_depth)`, deepest-first (the existing rank order). Cache-invalidating
  (recomputes once). This is an internal derivation type, not the `Screen` protocol.
- `Screen.select(source) -> list[str] | None` — **unchanged**. `DenseCompactScreen.select` returns
  `[bid for bid, _ in screen_selection(inp)]` (projects ids), so every existing caller
  (`_seed_groups`, `build_regions`, `run`) is unaffected.
- Add `DenseCompactScreen.selection_depths(source) -> dict[str, float]` = `dict(screen_selection(inp))`
  — a memoized L2 lookup (no recompute), returning the block_id → true max-depth map for the ~14k
  flagged blocks the `screen.png` coloring needs.
- `run.py` obtains the map by duck-typing the screen (mirrors how it duck-types `source.blocks_path`):
  `depths = sd(spec.source) if (sd := getattr(spec.screen, "selection_depths", None)) else None`, and
  passes it to `region_map`. No `RunOutput` change, no protocol change.

### 3.4 Coloring → true depth, deselected blank (`emit.region_map`)

- **Signature:** `region_map(source, regions, seed_groups, out_dir, *, selection, depths)` where
  `selection: list[str] | None` is the flagged set and `depths: dict[str, float] | None` maps
  block_id → true max access-depth. `run.py` passes `output.selection` and the duck-typed
  `selection_depths` map from §3.3.
- **`screen.png`:** fill each **flagged** block by `depths[block_id]` on a **continuous** `YlOrRd`
  ramp, `vmin=0`, `vmax = max(depths.values())` — the absolute ring scale, **no `scheme=`/binning**.
  **Deselected blocks are blanked** (white fill) with a faint outline; flagged blocks also get a
  faint per-block outline. Add a colorbar labelled "access depth (parcels from a street)". The whole
  expanded region is still located (dark member outline + locator box).
- **`region.png`:** fill the region **members** by their true depth (same ramp/scale). Member depths
  come from `depths` where present (flagged members); any builder-added non-flagged members are peeled
  together in ONE `block_depths(source, missing_member_ids)` call (batched — never per-member).
  Seed outline + building points unchanged.
- When `depths` is `None`/empty (a screen that selects all, or a non-kblock source), fall back to a
  single flat fill (no proxy coloring) — the squared/`√` proxy coloring is **removed**, not retained.
- **Delete `emit._screen_proxy`** and its test (superseded; migrate-not-accommodate).

### 3.5 Regenerate the multiblock example

Regenerate `screen.jpg` (true-depth colored, deselected blank, faint outlines, colorbar) and
`region.jpg` (members by true depth). Update the §1 prose: the screen **ranks/gates** on the proxy
(cheap) but the **map shows true access-depth**; note deselected blocks are blank. All other example
numbers/figures are unaffected.

## 4. Components & interfaces

- `reblock.region` — `block_depths(source, block_ids) -> dict[str, float]` (BATCHED, one buildings
  read); `DenseClusterRegionBuilder.build` and the `RegionBuilder` protocol + `Identity`/`ConvexHull`
  builders gain a `depth_fn` param (a pure map lookup); `_depth_proxy` retained as the no-peel fallback.
- `reblock.contracts` — **unchanged** (`Screen.select` stays `list[str] | None`).
- `reblock.derivations` — `screen_selection` / `_screen_selection_impl` return `list[tuple[str, float]]`.
- `reblock.screen.dense_compact` — `_compute_selection` returns pairs; `select` projects ids; add
  `selection_depths(source) -> dict[str, float]`.
- `reblock.pipeline` — `build_regions` precomputes the depth map via `_region_depth_map` (screen
  depths + one batched `block_depths` peel of the reachable non-survivors) and passes
  `depth_fn = depth_map.get` (0.0 default) to `region_builder.build`.
- `reblock.run` — duck-type the screen's `selection_depths`; pass `selection` + `depths` to `region_map`.
- `reblock.emit` — `region_map` colors by true depth (batched `block_depths` for any un-mapped
  members) + blanks deselected; `_screen_proxy` deleted.

## 5. Scope boundaries (YAGNI)

- **Cheap gate keeps the proxy** — not switched to true depth (intractable over 83k).
- **No change to the screen's selection outcome** — same cheap gate, same fine-pass mean/max gates,
  same ranking. Only what the proxy is *reused* for downstream changes.
- **Region builder growth may shift slightly** (true depth vs proxy ranking of candidates) — that is
  the intended behavior change, not a regression.
- No new metric, method, or config knob.

## 6. Testing

- `block_depths`: on fixture ids, returns `{id: access_before(block).max()}` for each buildable block;
  `{}` for a non-peel-capable source; peels a MULTI-id batch in one call (missing/unbuildable ids
  simply absent).
- `_region_depth_map`: prefers screen `selection_depths` for survivors (no peel) and batches the
  non-survivor peel; a block outside the BFS bound is absent (caller defaults 0.0).
- Region builder: on a small fixture with an injected `depth_fn` (a map lookup), growth adds the
  **highest-true-depth** adjacent block (a fixture where proxy-order and depth-order disagree, so the
  test discriminates); with `depth_fn=None`, behavior is byte-identical to today (proxy fallback).
- Screen: `screen_selection` returns `(id, depth)` pairs, deepest-first; `DenseCompactScreen.select`
  still returns plain ids (same order/values as before); `selection_depths` returns the id→depth map;
  `IdentityScreen.select` is untouched (still ids/`None`, no `selection_depths`).
- `region_map`: with a `depths` map, `screen.png`'s colored column equals true depth on `[0, max]`
  with no classification scheme; deselected blocks are blanked (not filled); `_screen_proxy` no longer
  importable. Existing `region_map` file-write tests still pass.
- `pixi run check` (ruff + mypy --strict + pytest) green.

## 7. Global constraints

- Migrate-not-accommodate: delete `_screen_proxy` and the squared-coloring path; the proxy survives
  only as the cheap gate and the region-builder no-peel fallback. No dual coloring path.
- Continuous colormap only (linear `Normalize`, no `scheme=`) — access-depth is integer-valued, so it
  renders as one exact shade per real depth; no lossy bucketing.
- `pixi run check` green; ruff (E702/E501/B905); mypy --strict. Commit trailers + PR footer per repo.
- Reproducible-by-CLI (Cape Town `capetown_full` + committed OSM snapshot).

## 8. Sequencing

This modifies `emit.region_map`, which the still-open **PR #9** also modified (squared proxy coloring)
and which this design supersedes. Land this as a follow-up branch off **PR #9's head** (or off `main`
after PR #9 merges) — do not reopen PR #9. If PR #9 has not merged at execution time, branch from its
head so the `_screen_proxy` deletion applies cleanly.

## 9. Implementation phasing

1. `block_depths` batched accessor + test.
2. Screen exposes depths (`screen_selection` returns pairs; `select` projects ids;
   `DenseCompactScreen.selection_depths`) + tests; `run.py` duck-types the map. Protocol unchanged.
3. Region builder `depth_fn` (+ pipeline injection) + tests; proxy fallback preserved.
4. `region_map` true-depth coloring + blank-deselected + `_screen_proxy` deletion + tests.
5. Regenerate the multiblock example (`screen.jpg`/`region.jpg` + §1 prose). Compute-heavy — last.
