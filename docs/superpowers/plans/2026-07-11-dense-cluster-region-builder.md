# Dense-Cluster Region Builder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `dense_cluster` `RegionBuilder` that grows each seed group into one contiguous region by block adjacency, up to a buildings budget (a parcel proxy). Purely additive.

**Architecture:** `DenseClusterRegionBuilder` in `src/reblock/region.py` (mirrors `ConvexHullRegionBuilder`) + `conf/region_builder/dense_cluster.yaml` + a small additive extension to `KblockSource.block_geometries` exposing `building_count`.

**Spec:** `docs/superpowers/specs/2026-07-11-dense-cluster-region-builder-design.md` — read it.

## Global Constraints

- **Additive:** `identity`/`convex_hull` untouched; the `block_geometries` change only ADDS a `building_count` column; existing consumers unaffected.
- **Deterministic, contiguous:** byte-stable sorted output; every returned region is one touch-adjacent component.
- **Graceful fallback:** no `building_count` column → budget on block-count.
- Verify each task with `pixi run check` (ruff + mypy --strict + pytest), run to completion.

---

### Task 1: Expose `building_count` in `block_geometries`

**Files:** Modify `src/reblock/data/kblock.py` (`block_geometries`, ~line 72-84); Test: `tests/` (add to an existing kblock/source test or a new `tests/test_block_geometries.py`).

- [ ] **Step 1: Write the test.**
```python
def test_block_geometries_includes_building_count():
    from reblock.data.kblock import KblockSource
    src = KblockSource("tests/data/kblock/blocks_dji_sample.parquet",
                       "tests/data/kblock/buildings_dji_sample.parquet", "dji")
    bg = src.block_geometries()
    assert "building_count" in bg.columns
    assert "block_id" in bg.columns and "geometry" in bg.columns
    row = bg[bg.block_id == "DJI.3_1_3238"].iloc[0]
    assert int(row.building_count) == 53           # matches the parquet
```
- [ ] **Step 2: Run — FAIL** (`building_count` absent). `pixi run pytest tests/test_block_geometries.py -v`
- [ ] **Step 3: Implement.** In `block_geometries`, read `building_count` too: `gpd.read_parquet(self.blocks_path, columns=["block_id", "building_count", "geometry"])`, and return `[["block_id", "building_count", "geometry"]]`. Keep the `block_ids` flat filter + `to_crs` + `_window` behavior unchanged. Update the docstring to note `building_count` is included.
- [ ] **Step 4: Run — PASS.** Also `pixi run pytest tests/test_region.py -q` (build_regions still consumes block_geoms fine).
- [ ] **Step 5: `pixi run check` green; commit** `feat: expose building_count in block_geometries`.

---

### Task 2: `DenseClusterRegionBuilder` + config

**Files:** Modify `src/reblock/region.py` (add `DenseClusterRegionBuilder`); Create `conf/region_builder/dense_cluster.yaml`; Test: `tests/test_region.py`.

**Interfaces — Produces:** `DenseClusterRegionBuilder(max_buildings: int = 150).build(block_geoms, groups) -> list[list[str]]`.

- [ ] **Step 1: Write tests** (in `tests/test_region.py`). Use real DJI blocks (DJI.3_1_3238 touches 3243 & 3240) + a hand-built fixture for the densest-first ordering:
```python
def test_dense_cluster_grows_seed_to_buildings_budget():
    from reblock.region import DenseClusterRegionBuilder
    from reblock.data.kblock import KblockSource
    bg = KblockSource("tests/data/kblock/blocks_dji_sample.parquet",
                      "tests/data/kblock/buildings_dji_sample.parquet", "dji").block_geometries()
    out = DenseClusterRegionBuilder(max_buildings=150).build(bg, [["DJI.3_1_3238"]])
    assert len(out) == 1
    region = out[0]
    assert "DJI.3_1_3238" in region and len(region) > 1        # grew past the seed
    total = int(bg[bg.block_id.isin(region)].building_count.sum())
    assert total >= 150 or _all_reachable(bg, region)          # hit budget (or exhausted component)

def test_dense_cluster_small_budget_returns_seed_only():
    ...  # max_buildings below the seed's own count (53) -> just [DJI.3_1_3238]

def test_dense_cluster_region_is_contiguous():
    ...  # _touch_adjacent(returned region geoms) is True

def test_dense_cluster_deterministic():
    ...  # two build() calls -> identical output

def test_dense_cluster_densest_neighbor_first():
    ...  # hand-built: seed + two neighbors (dense vs sparse); budget fits one -> the dense one is chosen

def test_dense_cluster_falls_back_to_block_count_without_building_count():
    ...  # drop building_count column -> budget on block count; still grows contiguously
```
Provide a `_touch_adjacent` / `_all_reachable` helper via the existing `region._touch_adjacent`.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement `DenseClusterRegionBuilder`** per spec §Design.2: STRtree `dwithin`(STREET_TOL) adjacency over all `block_geoms`; per group, `cluster` = seed indices; grow greedily by highest `building_count/area` frontier block (tie: higher building_count, then block_id) until `Σ building_count >= max_buildings` or frontier empty; fall back to block-count budget if no `building_count` column; `_validate_group_ids` first; return `sorted(cluster ids)`. Create `conf/region_builder/dense_cluster.yaml` (`_target_: reblock.region.DenseClusterRegionBuilder`, `max_buildings: 150`).
- [ ] **Step 4: Run — PASS** + `pixi run pytest tests/test_region.py -q`.
- [ ] **Step 5: `pixi run check` green; commit** `feat: DenseClusterRegionBuilder (grow contiguous region to a buildings budget)`.

---

### Task 3: Docs + smoke

**Files:** Modify `README.md` (Multi-block section).

- [ ] **Step 1: README** — under "Multi-block (region) reblocking", document `region_builder=dense_cluster`: grows a single seed block (or the screen's flagged blocks) into a contiguous region up to `max_buildings`, e.g.
  `pixi run python -m reblock.run data=dji method=greedy_arterial region_builder=dense_cluster region_builder.max_buildings=150 "block_ids=[[DJI.3_1_3238]]" render.enabled=true region_map.enabled=true`.
  Note it composes with `screen=dense_compact` (grows each flagged block). Position it alongside `identity`/`convex_hull`.
- [ ] **Step 2: Smoke run** (record the outcome): the recipe above completes, the region grew past the seed (region_map shows multiple member blocks), and before/after render + region_map emit.
- [ ] **Step 3: `pixi run check` green; commit** `docs: dense_cluster region builder recipe`.

## Self-Review

- **Spec coverage:** Task 1 = the block_geometries extension; Task 2 = the builder + config + all 6 correctness gates (grow-to-budget, small-budget, contiguous, deterministic, densest-first, fallback); Task 3 = docs + smoke + additive-invariant.
- **Placeholders:** the `53` building_count and `150` budget are concrete; the `_all_reachable`/hand-built-fixture helpers are named for the implementer to fill with real coords.
- **Type consistency:** `DenseClusterRegionBuilder.build(block_geoms, groups) -> list[list[str]]` matches the `RegionBuilder` Protocol; `max_buildings: int` is the sole field.
