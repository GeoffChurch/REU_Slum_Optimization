# Render Context Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overlay building points on the selected block/region and draw dimmed surrounding blocks (outlines + points) as context in the `region_map` and before/after heatmaps.

**Architecture:** Promote two cheap-geometry accessors (`block_geometries`, `building_points`) into the `Source` protocol (killing `hasattr`/`type: ignore`), each taking an optional `bbox` window; emitters take the typed `Source` and query windowed context per render frame; `render`/`region_map` draw the dimmed context + points.

**Tech Stack:** geopandas 1.1.4 (`.cx` spatial indexer; `read_parquet` bbox-pushdown is NOT available on the fixtures — see design §2), shapely, matplotlib, pytest.

**Design:** `docs/superpowers/specs/2026-07-10-render-context-overlay-design.md` (authoritative).

## Global Constraints

- No `hasattr` / `# type: ignore[attr-defined]` for source capabilities — the `Source` protocol declares them; both accessors are **total** (every source implements honestly).
- No back-compat / dual paths (owner's no-legacy rule): change call sites, delete the old duck-typing.
- `building_points` / `block_geometries` reproject to the region's UTM (`blocks.estimate_utm_crs()`) so overlays align exactly with the rendered parcels; the buildings/blocks parquets are WGS84.
- `bbox` is `(minx, miny, maxx, maxy)` in the source UTM; `bbox=None` → everything. Windowing is `.cx[minx:maxx, miny:maxy]` after reproject (single path).
- `ShapefileSource.building_points()` returns an **empty** GeoDataFrame (honest: no point cloud). Emitters guard `if not points.empty`.
- Determinism; matplotlib `Agg` (already set in render.py). Styling constants at module top.
- `pixi run check` (ruff + mypy --strict + pytest) green per task.

---

### Task 1: `Source` protocol + windowed accessors

**Files:**
- Modify: `src/reblock/contracts.py` (Source protocol + `BBox` alias)
- Modify: `src/reblock/data/kblock.py` (`block_geometries` gains `bbox`; add `building_points`)
- Modify: `src/reblock/data/shapefile.py` (add `block_geometries`, `building_points`)
- Modify: `src/reblock/pipeline.py` (drop `# type: ignore[attr-defined]` on line 94)
- Test: `tests/test_sources.py` (new)

**Interfaces:**
- Produces: `Source.block_geometries(bbox: BBox | None = None) -> GeoDataFrame` (cols `block_id`, `geometry`), `Source.building_points(bbox: BBox | None = None) -> GeoDataFrame` (col `geometry`, may be empty). `BBox = tuple[float, float, float, float]`.

- [ ] **Step 1 — `contracts.py`:** add `BBox = tuple[float, float, float, float]` (module level) and extend the `Source` Protocol:
```python
class Source(Protocol):
    def region(self) -> Region: ...
    def block_geometries(self, bbox: BBox | None = None) -> GeoDataFrame: ...
    def building_points(self, bbox: BBox | None = None) -> GeoDataFrame: ...
```
(Import `GeoDataFrame` under `TYPE_CHECKING` if not already; `BBox` is a plain alias.)

- [ ] **Step 2 — `kblock.py`:** add a cached target-UTM so both accessors and `region()` agree, and window. Replace the current `block_geometries` and add `building_points`:
```python
def _target_utm(self) -> CRS:
    if self._utm is None:
        self._utm = gpd.read_parquet(self.blocks_path, columns=["geometry"]).estimate_utm_crs()
    return self._utm

def block_geometries(self, bbox: BBox | None = None) -> gpd.GeoDataFrame:
    blocks = gpd.read_parquet(self.blocks_path, columns=["block_id", "geometry"])
    blocks["block_id"] = blocks["block_id"].astype(str)
    if self.block_ids is not None:
        blocks = cast(gpd.GeoDataFrame, blocks[blocks["block_id"].isin({str(b) for b in self.block_ids})])
    out = cast(gpd.GeoDataFrame, blocks.to_crs(self._target_utm())[["block_id", "geometry"]])
    return _window(out, bbox)

def building_points(self, bbox: BBox | None = None) -> gpd.GeoDataFrame:
    pts = gpd.read_parquet(self.buildings_path, columns=["geometry"]).to_crs(self._target_utm())
    return _window(cast(gpd.GeoDataFrame, pts), bbox)
```
Add `self._utm: CRS | None = None` in `__init__`. Add a module `_window(gdf, bbox)` helper: `gdf if bbox is None else cast(gpd.GeoDataFrame, gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]])`. (Keep `region()`'s existing UTM logic; do NOT reroute it — out of scope. `_target_utm` reads only the geometry column; on colocated data it equals the estimate `region()` uses.)

- [ ] **Step 3 — `shapefile.py`:** add both accessors. `building_points` is empty; `block_geometries` dissolves components (reuse `_components`). Factor the read+reproject+explode out of `region()` into a `_prepared() -> tuple[GeoDataFrame, CRS]` helper and call it from both `region()` and `block_geometries()` (DRY the shared prep):
```python
def block_geometries(self, bbox: BBox | None = None) -> gpd.GeoDataFrame:
    raw, utm = self._prepared()
    rows = []
    for k, idx in enumerate(_components(raw)):
        poly = gpd.GeoSeries(list(raw.iloc[idx].geometry), crs=utm).union_all()
        if isinstance(poly, Polygon):
            rows.append((f"{self.region_id}_{k}", poly))
    out = gpd.GeoDataFrame({"block_id": [r[0] for r in rows]},
                           geometry=[r[1] for r in rows], crs=utm)
    return _window(out, bbox)

def building_points(self, bbox: BBox | None = None) -> gpd.GeoDataFrame:
    _, utm = self._prepared()
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=utm)   # no point cloud
```
(`_window` importable from a shared spot or duplicated tiny helper — put `_window` in `reblock.data` `__init__` or a small `reblock/data/_util.py` and import in both sources, to avoid two copies.)

- [ ] **Step 4 — `pipeline.py`:** delete the `# type: ignore[attr-defined]` on the `source.block_geometries()` call (line ~94); it now type-checks against the protocol. Update the nearby comment that says sources "without a `block_geometries()` accessor" (all sources have it now; the classic path is chosen by *no groups*, not by capability).

- [ ] **Step 5 — Tests (`tests/test_sources.py`):**
```python
def test_kblock_building_points_are_points_in_region_utm():
    src = KblockSource(DJI_BLOCKS, DJI_BLD, region_id="dji")
    pts = src.building_points()
    assert not pts.empty and (pts.geometry.geom_type == "Point").all()
    assert pts.crs == src.block_geometries().crs                 # same UTM -> overlays align

def test_kblock_building_points_bbox_windows():
    src = KblockSource(DJI_BLOCKS, DJI_BLD, region_id="dji")
    allpts = src.building_points()
    minx, miny, maxx, maxy = allpts.total_bounds
    w, h = maxx - minx, maxy - miny
    sub = (minx + 0.4 * w, miny + 0.4 * h, minx + 0.6 * w, miny + 0.6 * h)
    assert 0 < len(src.building_points(sub)) < len(allpts)

def test_kblock_block_geometries_bbox_windows():
    src = KblockSource(DJI_BLOCKS, DJI_BLD, region_id="dji")
    allg = src.block_geometries()
    minx, miny, maxx, maxy = allg.total_bounds
    sub = (minx, miny, (minx + maxx) / 2, (miny + maxy) / 2)
    assert len(src.block_geometries(sub)) < len(allg)

def test_shapefile_building_points_empty_and_block_geometries_present():
    src = ShapefileSource(PHULE_SHP, region_id="phule")   # existing topology fixture
    assert src.building_points().empty
    bg = src.block_geometries()
    assert not bg.empty and set(bg.columns) >= {"block_id", "geometry"}
    assert (bg.geometry.geom_type == "Polygon").all()
```
(Find the Phule shapefile path from existing shapefile tests; if none is committed, cover `ShapefileSource` with a tiny synthetic 2-parcel shapefile written to a tmp_path, or a `gpd`-built GeoDataFrame if the source grows a from-frame constructor — else assert `building_points().empty` on a minimal instance and skip the block_geometries shapefile read.)

- [ ] **Step 6:** `pixi run check` green. Commit: `feat: Source.block_geometries/building_points (windowed) on the protocol; drop hasattr type:ignore`.

---

### Task 2: `render.py` — framing + context + points

**Files:**
- Modify: `src/reblock/render.py`
- Test: `tests/test_render.py` (extend; or `tests/test_emit.py` if that's where render smoke lives)

**Interfaces:**
- Consumes: `BBox` (contracts).
- Produces: `_frame_bbox(geoms, pad_frac=0.6) -> BBox`; `render_before(block, layers, *, vmax, context_outlines=None, context_points=None, own_points=None)`; `render_after(block, proposal, layers, *, vmax, metrics=None, context_outlines=None, context_points=None, own_points=None)`.

- [ ] **Step 1 — framing helper + draw order.** Add module constants and `_frame_bbox`:
```python
_CONTEXT_OUTLINE = "#dddddd"; _CONTEXT_PT = "#c9c9c9"; _OWN_PT = "#333333"

def _frame_bbox(geoms, pad_frac: float = 0.6) -> BBox:
    minx, miny, maxx, maxy = geoms.total_bounds
    half = max(maxx - minx, maxy - miny) / 2 + 1.0
    half += half * pad_frac
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    return (cx - half, cy - half, cx + half, cy + half)
```
Extend `_draw_heatmap` to accept `*, context_outlines=None, context_points=None, own_points=None`, and draw in this order (parcels heatmap first, as today):
1. parcels heatmap (unchanged)
2. `frame = _frame_bbox(parcels)`; after drawing, `ax.set_xlim(frame[0], frame[2]); ax.set_ylim(frame[1], frame[3])`
3. context (guard non-empty): `context_outlines.plot(ax=ax, facecolor="none", edgecolor=_CONTEXT_OUTLINE, linewidth=0.3)`; `context_points.plot(ax=ax, color=_CONTEXT_PT, markersize=2, alpha=0.6)`
4. boundary + `block.streets` (unchanged — the region inner-streets fix stays)
5. `own_points.plot(ax=ax, color=_OWN_PT, markersize=5)` (guard non-empty)

Draw context BEFORE the boundary/heatmap emphasis so the selection sits on top; set xlim/ylim so context outside the frame is clipped.

- [ ] **Step 2 — thread kwargs** through `render_before`/`render_after` to `_draw_heatmap`. No-arg calls (kwargs default `None`) render exactly as before (no context, but NOW framed — verify the single-block before/after still looks right with the added framing; framing a lone block to its own padded bounds is a no-op-ish tightening, acceptable).

- [ ] **Step 3 — Tests:** extend the render smoke test to pass `context_outlines`/`context_points`/`own_points` (small synthetic GeoDataFrames) and assert a file is written and no exception; keep the existing no-context test (proves the default path unchanged). Add a `_frame_bbox` unit test: square (equal width/height), centred on input bounds, min dimension respected.

- [ ] **Step 4:** `pixi run check` green. Commit: `feat: render draws dimmed context outlines/points + selection building points, framed to selection`.

---

### Task 3: `emit.region_map` points + `run.py` wiring

**Files:**
- Modify: `src/reblock/emit.py` (`region_map` signature → `Source`; add points; `render_results` → take `Source`, query windowed context per block)
- Modify: `src/reblock/run.py` (pass `spec.source`; drop `hasattr`)
- Test: `tests/test_emit.py`, `tests/test_run.py`

**Interfaces:**
- Consumes: `Source` (contracts), `render_before/after` kwargs (Task 2), `_frame_bbox` (Task 2).
- Produces: `region_map(source, regions, seed_groups, out_dir)`; `render_results(results, out_dir, cfg, source)`.

- [ ] **Step 1 — `render_results(results, out_dir, cfg, source)`.** For each block group, compute `frame = _frame_bbox(block.parcels)`, then `outlines = source.block_geometries(frame)`, `pts = source.building_points(frame)`. Split points by containment in `block.boundary`: `own = pts[pts.within(block.boundary)]`, `context_pts = pts[~pts.within(block.boundary)]`. Drop the selection's own blocks from `outlines` (by `block_id` membership, or by `intersects(block.boundary)` — outlines that are the selection itself shouldn't be dimmed). Pass `context_outlines`, `context_points=context_pts`, `own_points=own` to `render_before`/`render_after`.

- [ ] **Step 2 — `region_map(source, regions, seed_groups, out_dir)`.** Read all outlines `geoms = source.block_geometries()` (whole-area map, as today). Compute member geoms (filter by region member ids) → `frame = _frame_bbox(members)`; `pts = source.building_points(frame)`. Draw as today (context outlines, member fills, seed outlines, region framing) PLUS: member points (in members' boundary) normal (`_OWN_PT`, markersize 4), other points dimmed (`_CONTEXT_PT`, markersize 2, alpha 0.6). Guard empty points.

- [ ] **Step 3 — `run.py`.** Change the two emitter calls: `render_results(output.results, out_dir, cfg.render, spec.source)` and, in the `region_map.enabled` branch, `region_map(spec.source, output.regions, output.seed_groups, out_dir)`. **Delete** the `if not hasattr(spec.source, "block_geometries")` guard and its warning/`spec.source.block_ids = None` line — instead set `spec.source.block_ids = None` unconditionally before the call is no longer needed? (It was there to show ALL candidates as context; keep that intent: pass a source view over all candidates. If `block_ids` filtering on the source would hide context, set it to `None` for the map — but do so cleanly, e.g. read all via `block_geometries()` which already ignores `block_ids`? NO: `block_geometries` applies `block_ids`. Decision: `region_map` context should show ALL blocks, so temporarily clear or bypass `block_ids`. Simplest clean approach: `region_map` reads all candidates regardless of the run's `block_ids` filter — give `KblockSource.block_geometries` an `all_blocks: bool`? Avoid scope creep: keep the existing behavior — run.py sets `spec.source.block_ids = None` before calling region_map so the map shows the whole metro context — but now typed, no hasattr.)

- [ ] **Step 4 — Tests.** `test_emit.py`: `region_map` with a fake `Source` (returns small `block_geometries` + `building_points`) writes `region_map.png` with points; `render_results` with a fake source writes before/after. `test_run.py`: the region run path still produces before/after + `region_map.png` (extend the existing test); assert the run no longer references `hasattr` (the emitters call the protocol methods directly). Keep asserting the existing outputs.

- [ ] **Step 5:** `pixi run check` green. Commit: `feat: region_map + before/after draw building points and dimmed surroundings (typed Source, no hasattr)`.

---

## Notes for the executor
- Task 3 Step 3 has a real decision (how `region_map` shows all-candidate context under a `block_ids`-filtered run). Resolve by reading all candidates for the map (clear `block_ids` for that query) — keep it typed, no `hasattr`. Flag if a cleaner cut appears.
- Verify the single-block before/after still looks right once framing is added (Task 2 Step 2) — framing a lone block is a mild tightening, should be fine.
- After all tasks: whole-branch review, then finish-branch → merge to main + push (owner-gated).
