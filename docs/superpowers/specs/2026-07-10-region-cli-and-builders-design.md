# Region CLI + pluggable RegionBuilders + region-map viz — Design

**Status:** draft (owner-approved design; details flagged) · **Date:** 2026-07-10

Make multi-block reblocking usable from the CLI and the README, on the right abstraction: a
**list-of-lists** block spec + a **pluggable `RegionBuilder`** that expands each group into the
region to reblock, plus a **region-map visualization** of what the builder produced. Builds on
`region.py` (`region_block`/`region_reblock`, already on this branch).

## 1. `block_ids` becomes a list of lists (migrate; drop the `region` flag)

`block_ids: list[list[str]] | null`. Each inner list is a **region seed group**; singletons are
single-block reblocking. `null` = every block a singleton (today's default). This unifies
single- and multi-block (single-block is the all-singletons case) and needs no mode flag — the
grouping *is* the spec. Migrate all configs/recipes (`block_ids=[[X]]` for the old single-block
form); no flat-list back-compat (owner's no-legacy rule).

## 2. Pluggable `RegionBuilder` (new pipeline stage)

A stage between selection and reblocking, sibling to Screen/Method/Eval:

```python
class RegionBuilder(Protocol):
    def build(self, block_geoms: GeoDataFrame, groups: list[list[str]]) -> list[list[str]]: ...
```
Input: cheap block **geometries** (block_id + polygon, no Voronoi) + the seed groups. Output:
the **expanded** groups (block_ids per region). Ships with two builders:

- **`identity`** (default) — returns `groups` unchanged; a singleton stays one block → reduces to
  today's per-block behavior exactly.
- **`convex_hull`** (`ConvexHullRegionBuilder`) — each group → every candidate block whose
  geometry **intersects the convex hull** of the group's block polygons. A singleton's hull is
  its own shape, so it reduces to identity; a disjoint group's hull fills in the blocks between
  them. Overlap between different groups' hulls is **allowed** (owner's call — regions are
  independent; no partition/merge). `k_layers` is a future drop-in on the same interface.

The builder works on cheap geometries so it can pick members *before* the expensive full-Block
(Voronoi) build — the `Source` exposes them (KblockSource: read `block_id`,`geometry` from the
blocks parquet). The pipeline then builds full `Block`s only for the union of region members and
`region_reblock`s each group. Regions may overlap and are reblocked/compared independently.

## 3. Pipeline placement

`Source → (Screen or explicit block_ids → seed groups) → RegionBuilder → per-region
region_reblock / region-compare → emit`. A Screen's flat output wraps as singleton seed groups,
so `screen=dense_compact region_builder=convex_hull` reads as "flag dense blocks, reblock each
with its hull-filled neighborhood." One method per `run`; the method-list per `compare`;
**per-group methods are not supported** — compose multiple invocations (a rare, shell-composable
need; keeps the spec doing one job).

## 4. `region_map` visualization (the builder-layer output)

An emitter that draws, over the candidate blocks, **each region's blocks colored by region
assignment**, with the **seed blocks outlined/highlighted** — so you can see what the builder
pulled in (essential for `convex_hull`, which expands). Gated `region_map.enabled` (like
`flagged_map`); writes `region_map.png` to the run dir. This is "output by the region-builder
layer" — it visualizes the builder's `build()` result.

## 5. `run` / `compare` region path

- **`reblock.run`** — for each expanded region: `region_reblock(member_blocks, method, evals)` →
  one `Result`; render the region's before/after (the region-Block is an ordinary `Block`, so the
  existing render works). `region_map` if enabled.
- **`reblock.compare`** — for each (region, method): `region_reblock` → the seed+added proposal →
  the existing 3-lens `cost_benefit_curve`/AUC on the region's `eval_block`. Per-metric AUC
  tables + curves as today, keyed by region.

## 6. README recipes (the deliverable)

```bash
# Multi-block arterial reblock (two adjacent blocks reblocked jointly, roads span the old line)
pixi run python -m reblock.run data=dji method=greedy_arterial \
  "block_ids=[[DJI.3_1_1808,DJI.3_1_1809]]" eval=kcomplexity render.enabled=true region_map.enabled=true

# Multi-block cost-benefit curve (methods graded on the region)
pixi run python -m reblock.compare data=dji \
  methods=[dijkstra,mesh,greedy_arterial_buildable] \
  "block_ids=[[DJI.3_1_1808,DJI.3_1_1809]]" eval=kcomplexity
```
(`region_builder=convex_hull` swaps in the hull expander.)

## 7. Testing

- `IdentityRegionBuilder`: groups pass through; singletons unchanged.
- `ConvexHullRegionBuilder`: on a synthetic block grid, a 2-disjoint-block group pulls in the
  block between them; a singleton returns itself; overlapping groups both keep their members.
- `run` region path: an adjacent pair → one region Result whose proposal spans both blocks;
  `region_map.png` written when enabled.
- `compare` region path: adjacent pair → per-metric AUC tables/curves; arterial leads directness.
- `block_ids` list-of-lists parsing + migration (old single-block recipes → `[[X]]`).

## 8. Out of scope (follow-ons)

- `k_layers` RegionBuilder (drop-in on the same interface).
- Auto-grouping a Screen's flat output into adjacent clusters (a builder variant).
- Efficient candidate loading for the full metro (v1 reads all block geometries — cheap for the
  DJI sample; a bbox-bounded candidate query is the scale follow-on).
- Merging overlapping regions for `run` (owner chose independent/overlap-allowed).
