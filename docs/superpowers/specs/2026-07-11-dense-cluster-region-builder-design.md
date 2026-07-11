# Dense-Cluster Region Builder — Design

**Status:** draft for review · **Date:** 2026-07-11

**Goal:** A `dense_cluster` `RegionBuilder` that grows each seed group into one **contiguous** region by block adjacency, up to a **buildings budget** (a parcel proxy). This makes "plump a single block (or a screen's flagged block) into a right-sized region" a one-knob operation — no hand-listing neighbors — and keeps the region within arterial's tractable range.

**Architecture:** Add `DenseClusterRegionBuilder` to `src/reblock/region.py` (mirrors `ConvexHullRegionBuilder`), a `conf/region_builder/dense_cluster.yaml`, and a small additive extension to `KblockSource.block_geometries` exposing `building_count`. The builder is auto-wired through the existing `region_builder` Hydra group + `instantiate`.

**Tech Stack:** Python, geopandas, shapely (`STRtree`), networkx (adjacency), Hydra, pixi, pytest, mypy --strict.

## Global Constraints

- **Additive / no behavior change to existing builders.** `identity` + `convex_hull` untouched; the `block_geometries` change only ADDS a `building_count` column (existing consumers select `["block_id","geometry"]` and are unaffected).
- **Deterministic:** growth order and output are byte-stable (sorted tie-breaks); returned member lists sorted, group order preserved (the `RegionBuilder` contract).
- **Contiguous:** every returned region is touch-adjacent (one connected component) — `dense_cluster` never emits a disjoint region.
- **Graceful without building counts:** if `block_geoms` lacks `building_count` (a non-kblock source), fall back to a block-count budget so the builder still works.
- `mypy --strict`, ruff clean, `pixi run check` green.

## Design

### 1. Expose `building_count` in `block_geometries` (additive)
`KblockSource.block_geometries` currently reads `["block_id","geometry"]`. Extend it to also read `building_count` from the blocks parquet and return `["block_id","building_count","geometry"]` (building_count as int/float). The `Source` protocol's `block_geometries` docstring notes `building_count` is present when the source has it (optional column). No existing caller breaks (they index by name).

### 2. `DenseClusterRegionBuilder` (in `region.py`)
```python
@dataclass
class DenseClusterRegionBuilder:
    max_buildings: int = 150        # budget; ~ parcels; keep arterial tractable
    def build(self, block_geoms, groups) -> list[list[str]]: ...
```
Per seed group:
- Build block adjacency over ALL of `block_geoms` (STRtree `dwithin` within `STREET_TOL`, as `_touch_adjacent` does) → `adj[i] = {neighbor indices}`.
- `cluster` = the seed group's block indices (already-listed seeds are always included, even if that alone exceeds the budget). `size` = Σ `building_count` over the cluster (or cluster block-count if no `building_count` column).
- **Grow (greedy, contiguous, need-toward-dense):** while `size < max_buildings` and the frontier is non-empty: `frontier` = blocks adjacent to the cluster but not in it; pick the frontier block with the **highest building density** (`building_count / area`), tie-broken by higher `building_count`, then by `block_id` (determinism); add it; update `size`. Stop when `size >= max_buildings` (the last block may push the total slightly over) or no adjacent block remains.
- Return `sorted(cluster block_ids)`.

Rationale for "densest neighbor first": density is the closest geometry-available proxy for reblocking need (dense blocks are where buried parcels concentrate), so the region grows toward the neediest surrounding fabric rather than sprawling into sparse edges. (True need = access depth isn't available at block-geometry level; the *seed* carries the need — via `block_ids` or the screen's worst-first ranking — and growth stays local + dense.)

### 3. Config `conf/region_builder/dense_cluster.yaml`
```yaml
_target_: reblock.region.DenseClusterRegionBuilder
max_buildings: 150
```

### 4. Composition (no new wiring)
- **Explicit seed:** `region_builder=dense_cluster "block_ids=[[DJI.3_1_3238]]"` → grows that block's contiguous cluster to ~150 buildings.
- **From the screen:** `screen=dense_compact region_builder=dense_cluster` → the screen flags need-ranked singletons; the builder grows each. (Each flagged block becomes its own grown region; `max_blocks` still caps how many regions.)

## Correctness gates (tests, `tests/test_region.py`)

1. **Grows to budget:** seed one small block; with `max_buildings` large enough, the returned region has >1 block and Σ building_count is within `[max_buildings, max_buildings + one-block]` (or all reachable blocks if the component is smaller than the budget).
2. **Budget respected / small budget:** `max_buildings` below the seed's own count → returns just the seed (seeds always included; no growth).
3. **Contiguous:** the returned region is `_touch_adjacent` (one component) on real DJI blocks.
4. **Determinism:** same inputs → identical sorted output across runs.
5. **Densest-first order:** on a hand-built fixture (seed + two neighbors of different density), the higher-density neighbor is added first when only one fits the budget.
6. **Fallback:** `block_geoms` without a `building_count` column → budget falls back to block-count; still grows contiguously and returns a valid region.
7. **Additive:** `identity`/`convex_hull` tests + the full suite unchanged; `block_geometries` now returns `building_count` (a new column) without breaking `build_regions`.

## Task decomposition

1. **Expose `building_count` in `block_geometries`** + a test that the column is present and correct on the DJI sample. (Small, isolated.)
2. **`DenseClusterRegionBuilder` + config** + tests (grow-to-budget, small-budget, contiguous, determinism, densest-first, block-count fallback).
3. **Docs + smoke:** README "Multi-block" section — document `region_builder=dense_cluster` with the one-block-to-region recipe; a smoke run confirming `region_builder=dense_cluster "block_ids=[[DJI.3_1_3238]]"` grows a region and renders.

## Out of scope (follow-ups)
- The Cape Town flagship example (screen → dense_cluster → compare) — a separate deliverable once the builder proves out.
- Need-weighted growth using true access depth (would require deriving depth at block-geometry level or threading the screen's per-block scores into the builder).
- Auto-picking the seed (top-N worst clusters) beyond what the screen already provides.
