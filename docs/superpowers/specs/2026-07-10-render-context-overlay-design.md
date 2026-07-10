# Render context: building-point overlay + dimmed surroundings — Design

**Status:** owner-approved (2026-07-10) · **Date:** 2026-07-10

Situate a reblock render in its surroundings: overlay the building points on the selected
block/region, and draw the neighbouring blocks (dimmed outlines + dimmed points) as context, in
both the `region_map` and the before/after heatmaps — so the actual selection reads unambiguously
against what's around it.

## 1. The `Source` contract gains cheap geometry accessors (kills the `hasattr`)

Today the `Source` protocol declares only `region()`, so the region-map path guards with
`hasattr(spec.source, "block_geometries")` (run.py) and `# type: ignore[attr-defined]`
(pipeline.py). Promote the cheap-geometry capability into the contract — both accessors **total**,
so every source implements them honestly and the duck-typing warts are deleted:

```python
BBox = tuple[float, float, float, float]   # (minx, miny, maxx, maxy), source CRS-agnostic input

class Source(Protocol):
    def region(self) -> Region: ...
    def block_geometries(self, bbox: BBox | None = None) -> GeoDataFrame: ...   # block_id + geometry
    def building_points(self, bbox: BBox | None = None) -> GeoDataFrame: ...     # points; may be empty
```

- **`KblockSource`** — `block_geometries` reads the blocks parquet (as today), `building_points`
  reads the buildings parquet's points. Both reproject to the same UTM `region()` uses.
- **`ShapefileSource`** — `block_geometries` returns its dissolved connected-components (it
  genuinely has block polygons); `building_points` returns an **empty** GeoDataFrame. This is a
  correct, total implementation, not a throwing stub: a parcel shapefile has no building-point
  cloud, so its context renders as outlines only.

Removes the `hasattr` (run.py) and the `# type: ignore` (pipeline.py `build_regions`).

## 2. bbox windowing, up front

Both accessors take an optional `bbox` (the render frame, in the source's UTM) and return only
geometry intersecting it; `bbox=None` returns everything (the RegionBuilder / whole-metro use). The
accessor reads the parquet, reprojects to UTM (the buildings/blocks parquets are WGS84 — same
`estimate_utm_crs()` the region uses, so overlays align with the rendered parcels), then windows
with the `.cx[minx:maxx, miny:maxy]` spatial indexer. So a windowed query returns only in-frame
geometry — the emitter never draws the whole metro.

Note: `gpd.read_parquet(path, bbox=...)` disk-pushdown is **not** available on the current fixtures
(they lack the GeoParquet bbox-covering column — verified: it raises), so the window filters the
in-memory frame rather than skipping row-groups on read. A true read-pushdown is a provision-time
follow-on (write the parquets with `write_covering_bbox=True`); the accessor API is unchanged when
it lands. Single code path either way — `.cx` after reproject.

## 3. Emitters take the typed `Source` and query windowed context per frame

No source-type introspection anywhere. `render_results` and `region_map` receive the `Source`
(typed by the protocol) and call `source.block_geometries(frame)` / `source.building_points(frame)`
for each render's frame. A fake `Source` with the two methods makes these trivially testable.

## 4. Framing helper (shared)

```python
def _frame_bbox(geoms: GeoSeries | GeoDataFrame, pad_frac: float = 0.6) -> BBox:
    """A padded square bbox centred on `geoms`' total_bounds -- the render view, and the bbox the
    context query is windowed to. Square + padded so the selection dominates with a context margin."""
```

Used by both `render_results` (to window the context query) and the `region_map` framing already
added on `main`, and applied to the before/after axes (`set_xlim`/`set_ylim`). This makes the
before/after zoom to the selection; without it, drawing context would auto-expand the extent and
shrink the selection to a speck (the bug `region_map` already fixed).

## 5. `render.py` — before/after

`_draw_heatmap(block, layers, vmax, *, context_outlines=None, context_points=None,
own_points=None, frame=None)` draw order:
1. Selected block/region parcels → the access heatmap (as today).
2. Frame the axes to `frame` (the `_frame_bbox` of the block's parcels).
3. Dimmed context: `context_outlines` (thin light-grey edges, no fill) + `context_points` (small
   light-grey dots) — both already windowed to `frame`.
4. `block.boundary` + `block.streets` (as today — the region inner-streets fix stays).
5. `own_points` — the selection's own building points — on top, emphasised (small dark dots).

`render_before` / `render_after` gain and forward these keyword args. Single-block and region
renders both get context (a single block sees its neighbours dimmed; a region sees blocks around
the union). `own_points` are the points within `block.boundary`; context are those outside it —
`render_results` splits them by containment when it queries (points inside boundary = own).

## 6. `emit.py` — `region_map`

Add a points overlay: member-block points normal, context (non-member) points dimmed — matching
the existing member-fill / context-outline styling. `region_map` is one whole-area map, so it keeps
reading **all** candidate outlines (`block_geometries()`, as today) and frames to the region; it
computes that frame from the member geometries and queries `source.building_points(frame)` so only
the **points** layer is windowed (points are the expensive layer; outlines are cheap and already
read). It takes the typed `Source` in place of the pre-read `block_geoms` GeoDataFrame.

## 7. `run.py` — wiring

Pass `spec.source` to `render_results` and `region_map`; drop the `hasattr` guard. Both emitters
own their windowed fetch. `render_results` iterates results, computes each frame, queries context,
splits own/context points, and renders.

## 8. Styling

Heatmap colours dominate; overlays sit on top. Context outlines `#dddddd` / linewidth ~0.3, no
fill; context points `#c9c9c9` / size ~2 / alpha ~0.6; own points `#333333` / size ~5. Dimming
makes the selection unambiguous. (Colours are constants at the top of `render.py`/`emit.py`.)

## 9. Testing

- `KblockSource.building_points()` — non-empty, all `Point`, reprojected to the region UTM; with a
  `bbox`, returns only points inside it (a small bbox yields fewer than the full read).
- `KblockSource.block_geometries(bbox)` — bbox windows the result (fewer blocks than `bbox=None`).
- `ShapefileSource.building_points()` — returns an **empty** GeoDataFrame; `block_geometries()`
  returns its component blocks (block_id + Polygon geometry).
- `_frame_bbox` — square, centred, padded (min dimension respected).
- `_draw_heatmap` / `render_before` / `render_after` — run with context + points kwargs and write a
  file (matplotlib smoke, matching existing render tests); no-context path unchanged.
- `region_map` — runs with a fake `Source` supplying points; writes `region_map.png`.
- `run` region path — produces before/after + `region_map.png` with the overlays; asserts no
  `hasattr` remains (grep-style: the Source protocol methods are called directly).

## 10. Out of scope (follow-ons)

- Voronoi **cells** for context blocks (owner chose outlines + points only; cells stay
  selection-only).
- Building **footprints** (the data is points; footprints need a different source).
- A `MapContext` value object bundling outlines+points into one return — two methods suffice, and
  `building_points` is needed alone by nothing yet but reads cleaner than a bundled tuple.
