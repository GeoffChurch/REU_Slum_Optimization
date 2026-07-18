# True Access-Depth Everywhere Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Confine the depth proxy `√(n·A)/P` to the screen's cheap gate, and drive the region builder's growth and both map colorings by true BFS peel access-depth instead.

**Architecture:** One memoized accessor `block_max_depth(source, block_id)` (`access_before(block).max()`) is the single source of true depth. The region builder ranks candidates by an injected `depth_fn` (defaulting to the proxy for un-peelable sources). The screen exposes the depths it already computes (via the memoized `screen_selection` derivation + a `DenseCompactScreen.selection_depths` method — the `Screen` protocol is untouched). `region_map` colors flagged blocks by true depth on the absolute 0–max scale and blanks screen-deselected blocks.

**Tech Stack:** Python 3, Hydra, geopandas/shapely, scipy/networkx, matplotlib, pytest, ruff, mypy --strict, pixi.

## Global Constraints

- **Proxy confined to the cheap gate.** The screen's cheap pre-filter (`dense_compact._depth_proxy` over all 83k blocks) keeps the proxy — the sole sanctioned use. `region.py`'s `_depth_proxy` survives **only** as the region-builder fallback for un-peelable sources. No other proxy use.
- **Migrate, never accommodate.** Delete `emit._screen_proxy` and the squared-proxy coloring path; no dual coloring path, no legacy fallback branch.
- **Continuous colormap only** — linear `Normalize`, **no `scheme=`/mapclassify binning**. Access-depth is integer-valued, so it renders as one exact shade per real depth; no lossy bucketing.
- **`Screen` protocol is unchanged** (`select(source) -> list[str] | None`). `IdentityScreen` is untouched.
- **Region builder growth is NOT restricted to flagged neighbors** — non-survivor candidates are peeled on demand (memoized).
- `pixi run check` (ruff lint + mypy --strict + pytest) stays green at the end of every task. ruff forbids semicolons (E702), lines >100 chars (E501), `zip()` without `strict=` (B905). Use `cast(...)` for narrowing (codebase convention).
- Commit trailers on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
  ```

**Spec:** `docs/superpowers/specs/2026-07-17-true-depth-everywhere-design.md`

---

## File Structure

- `src/reblock/region.py` — add `block_max_depth(source, block_id)`; `RegionBuilder` protocol +
  `Identity`/`ConvexHull`/`DenseCluster` builders gain a `depth_fn` param; `_depth_proxy` kept as
  fallback. (Tasks 1, 3)
- `src/reblock/derivations.py` — `screen_selection`/`_screen_selection_impl` return `list[tuple[str,
  float]]`. (Task 2)
- `src/reblock/screen/dense_compact.py` — `_compute_selection` returns pairs; `select` projects ids;
  add `selection_depths`. (Task 2)
- `src/reblock/pipeline.py` — `build_regions` injects `depth_fn`. (Task 3)
- `src/reblock/run.py` — duck-type `selection_depths`; pass `selection` + `depths` to `region_map`. (Task 4)
- `src/reblock/emit.py` — `region_map` true-depth coloring + blank deselected; delete `_screen_proxy`. (Task 4)
- `tests/` — `test_region.py`, `test_derivations.py` or `tests/screen/`, `test_pipeline.py`,
  `test_emit.py`. (Tasks 1–4)
- `examples/multiblock/` — regenerate `screen.jpg`/`region.jpg` + §1 prose. (Task 5)

---

## Task 1: `block_max_depth` accessor

**Files:**
- Modify: `src/reblock/region.py` (add `block_max_depth` near `_depth_proxy`, ~line 233)
- Test: `tests/test_region.py`

**Interfaces:**
- Consumes: `reblock.derivations.access_before(block) -> pd.Series` (memoized); `reblock.data.kblock.KblockSource(blocks_path, buildings_path, *, region_id, min_buildings, block_ids)`.
- Produces: `block_max_depth(source: Source, block_id: str) -> float` — the block's true max BFS access-depth (`access_before(block).max()`), built via a `KblockSource` windowed to `block_id`. `0.0` for a non-peel-capable source or a block that can't be built/peeled.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_region.py` (it already imports from `reblock.region`; add these imports at the top if absent: `from pathlib import Path`):

```python
def test_block_max_depth_matches_access_before_peel() -> None:
    # On the committed DJI sample, block_max_depth(source, id) equals access_before(block).max()
    # -- the true BFS peel depth -- and a second call returns the same value (memoized).
    from reblock.data.kblock import KblockSource
    from reblock.derivations import access_before
    from reblock.region import block_max_depth
    root = Path(__file__).resolve().parent
    src = KblockSource(root / "data/kblock/blocks_dji_sample.parquet",
                       root / "data/kblock/buildings_dji_sample.parquet", "dji",
                       block_ids=["DJI.3_1_1808"])
    block = next(iter(src.region().blocks))
    expected = float(access_before(block).max())
    assert block_max_depth(src, "DJI.3_1_1808") == expected
    assert block_max_depth(src, "DJI.3_1_1808") == expected   # cache hit, same value


def test_block_max_depth_zero_for_non_peelable_source() -> None:
    # A source with no blocks_path (not a KblockSource) can't be peeled -> 0.0, so it never wins a
    # "deepest" argmax in the region builder.
    from reblock.region import block_max_depth

    class _Bare:
        def region(self): raise NotImplementedError
        def block_geometries(self, bbox=None): raise NotImplementedError
        def building_points(self, bbox=None): raise NotImplementedError

    assert block_max_depth(_Bare(), "anything") == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_region.py -k block_max_depth -v`
Expected: FAIL with `ImportError: cannot import name 'block_max_depth'`.

- [ ] **Step 3: Implement `block_max_depth`**

Add to `src/reblock/region.py` immediately after `_depth_proxy` (the local imports avoid an
import cycle, matching `derivations._screen_selection_impl`'s pattern):

```python
def block_max_depth(source: Source, block_id: str) -> float:
    """True max BFS access-depth (parcel rings from a street) of one block, built via a
    `KblockSource` windowed to `block_id` (from `source`'s parquet paths) and peeled with the
    memoized `access_before` -- so a block the screen already peeled is an end-to-end cache hit and
    a never-seen block is peeled once, then cached. Returns `0.0` for a non-peel-capable source (no
    `blocks_path`) or a block that can't be built/peeled, so it never wins a `deepest` argmax. This
    is the single source of true depth for the region builder's growth and the `region_map`
    colorings."""
    from reblock.data.kblock import KblockSource
    from reblock.derivations import access_before
    if not isinstance(source, KblockSource):
        return 0.0
    sub = KblockSource(source.blocks_path, source.buildings_path, "depth",
                       min_buildings=getattr(source, "min_buildings", 10), block_ids=[block_id])
    blocks = list(sub.region().blocks)
    if not blocks:
        return 0.0
    return float(access_before(blocks[0]).max())
```

`Source` is already imported in `region.py` (used by other signatures); confirm the import line
`from reblock.contracts import ... Source ...` includes it, and add `Source` to that import if not.

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run pytest tests/test_region.py -k block_max_depth -v`
Expected: PASS (both tests).

- [ ] **Step 5: Lint + type-check**

Run: `pixi run ruff check src/reblock/region.py tests/test_region.py && pixi run mypy --strict src/reblock/region.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/region.py tests/test_region.py
git commit -m "feat: block_max_depth accessor (true peel depth for one block, memoized)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

## Task 2: Screen exposes the depths it already computes (protocol unchanged)

**Files:**
- Modify: `src/reblock/derivations.py` (`_screen_selection_impl`/`screen_selection` return type, ~line 103–109)
- Modify: `src/reblock/screen/dense_compact.py` (`_compute_selection` return, ~line 138; `select`, ~line 151; add `selection_depths`)
- Test: `tests/screen/test_dense_compact.py` (create if absent) or `tests/test_derivations.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `screen_selection(inp) -> list[tuple[str, float]]` — `(block_id, max_depth)` pairs, deepest-first.
  - `DenseCompactScreen.select(source) -> list[str] | None` — **unchanged return type** (ids, same order/values as before).
  - `DenseCompactScreen.selection_depths(source) -> dict[str, float]` — block_id → true max-depth map.

- [ ] **Step 1: Write the failing test**

Create `tests/screen/test_dense_compact_depths.py` (a lightweight test over the committed DJI sample;
mirror the fixture path style of `tests/scoring_fixtures.py`):

```python
from pathlib import Path

from reblock.data.kblock import KblockSource
from reblock.screen.dense_compact import DenseCompactScreen

_ROOT = Path(__file__).resolve().parent.parent


def _src() -> KblockSource:
    return KblockSource(_ROOT / "data/kblock/blocks_dji_sample.parquet",
                        _ROOT / "data/kblock/buildings_dji_sample.parquet", "dji")


def test_select_returns_ids_and_selection_depths_maps_them() -> None:
    # select() still returns plain block_ids (protocol unchanged); selection_depths returns the same
    # ids mapped to their true max access-depth, and the two agree on membership.
    screen = DenseCompactScreen(min_buildings=1)
    ids = screen.select(_src())
    depths = screen.selection_depths(_src())
    assert ids is not None
    assert all(isinstance(b, str) for b in ids)
    assert set(depths) == set(ids)                       # same blocks
    assert all(d >= 1.0 for d in depths.values())        # every flagged block is >= 1 ring deep


def test_screen_selection_returns_pairs_deepest_first() -> None:
    from reblock.derivations import ScreenSelectionInput, screen_selection
    from reblock.derive_graph import source_hash
    src = _src()
    inp = ScreenSelectionInput(
        source_hash=source_hash(src.blocks_path, src.buildings_path),
        blocks_path=str(src.blocks_path), buildings_path=str(src.buildings_path),
        depth_proxy_min=1.5, mean_depth_min=1.3, max_depth_min=None, k_min=None, min_buildings=1)
    pairs = screen_selection(inp)
    assert pairs and all(isinstance(b, str) and isinstance(d, float) for b, d in pairs)
    depths = [d for _, d in pairs]
    assert depths == sorted(depths, reverse=True)        # deepest-first (the ranking order)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/screen/test_dense_compact_depths.py -v`
Expected: FAIL — `selection_depths` doesn't exist / `screen_selection` returns `list[str]` (the pair-unpack `for b, d in pairs` raises).

- [ ] **Step 3: `_compute_selection` returns pairs**

In `src/reblock/screen/dense_compact.py`, change the final return of `_compute_selection` (currently
`return [bid for _, bid in ranked]`) to return `(block_id, max_depth)` pairs — `ranked` is
`list[tuple[float, str]]` = `(max_d, bid)`:

```python
    return [(bid, max_d) for max_d, bid in ranked]
```

Update `_compute_selection`'s return annotation to `list[tuple[str, float]]`.

- [ ] **Step 4: `screen_selection` derivation returns pairs**

In `src/reblock/derivations.py`, change the two annotations:

```python
def _screen_selection_impl(inp: ScreenSelectionInput) -> list[tuple[str, float]]:
    from reblock.screen.dense_compact import _compute_selection  # local import avoids a cycle
    return _compute_selection(inp)


def screen_selection(inp: ScreenSelectionInput) -> list[tuple[str, float]]:
    return derive(_screen_selection_impl, inp)
```

- [ ] **Step 5: `DenseCompactScreen.select` projects ids; add `selection_depths`**

In `src/reblock/screen/dense_compact.py`, `select` currently ends `return screen_selection(inp)`.
Refactor to build the input once and expose both projections:

```python
    def _selection_input(self, source: Source) -> ScreenSelectionInput:
        if not isinstance(source, KblockSource):
            raise TypeError(
                f"DenseCompactScreen needs a KblockSource (kblock columns); "
                f"got {type(source).__name__}")
        return ScreenSelectionInput(
            source_hash=source_hash(source.blocks_path, source.buildings_path),
            blocks_path=str(source.blocks_path), buildings_path=str(source.buildings_path),
            depth_proxy_min=self.depth_proxy_min, mean_depth_min=self.mean_depth_min,
            max_depth_min=self.max_depth_min, k_min=self.k_min,
            min_buildings=self.min_buildings)

    def select(self, source: Source) -> list[str]:
        return [bid for bid, _ in screen_selection(self._selection_input(source))]

    def selection_depths(self, source: Source) -> dict[str, float]:
        """block_id -> true max access-depth for the flagged blocks (a memoized `screen_selection`
        L2 lookup, no recompute) -- what `region_map`'s screen.png coloring keys on."""
        return dict(screen_selection(self._selection_input(source)))
```

(The `ScreenSelectionInput` construction moves verbatim from the old `select` into `_selection_input`;
`select`'s existing `TypeError` guard moves there too.)

- [ ] **Step 6: Run the new tests + the screen/pipeline suites**

Run: `pixi run pytest tests/screen/ tests/test_pipeline.py tests/test_run.py -v`
Expected: PASS — the new depth tests, and every existing screen/pipeline/run test (they consume
`select()`'s ids, whose values/order are unchanged).

- [ ] **Step 7: Lint + type-check**

Run: `pixi run ruff check src/reblock/derivations.py src/reblock/screen/dense_compact.py tests/screen/test_dense_compact_depths.py && pixi run mypy --strict src/reblock/derivations.py src/reblock/screen/dense_compact.py`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add src/reblock/derivations.py src/reblock/screen/dense_compact.py tests/screen/test_dense_compact_depths.py
git commit -m "feat: expose screen fine-pass depths (screen_selection pairs + selection_depths)

Screen protocol unchanged: select() still returns ids; the memoized derivation now
carries (id, depth) pairs and DenseCompactScreen.selection_depths maps them.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

## Task 3: Region builder grows by true depth (`depth_fn`) + pipeline injection

**Files:**
- Modify: `src/reblock/region.py` (`RegionBuilder` protocol L128; `IdentityRegionBuilder.build` L168; `ConvexHullRegionBuilder.build` L196; `DenseClusterRegionBuilder.build` L273)
- Modify: `src/reblock/pipeline.py` (`build_regions`, L96)
- Test: `tests/test_region.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `block_max_depth(source, block_id) -> float` (Task 1).
- Produces: `RegionBuilder.build(block_geoms, groups, depth_fn=None)` — `depth_fn: Callable[[str], float] | None` maps a block_id to its true max access-depth; `None` = use the proxy fallback.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_region.py`. This fixture makes proxy-order and true-depth-order **disagree** so the
test discriminates: give `depth_fn` a hand map that ranks a low-proxy neighbor as the deepest, and
assert growth picks it. (`_grid_block`-style fixtures already exist in this file; build a minimal
3-block adjacency GeoDataFrame inline.)

```python
def _fork_gdf():
    # Seed "s" (centre) adjacent to BOTH "a" (east) and "b" (west); a and b are NOT adjacent to each
    # other (s separates them). All three identical unit squares -> identical proxy score. A budget
    # of seed + exactly one more forces a CHOICE between a and b: proxy ties and breaks to "a" (id
    # ascending); a depth_fn ranking b highest picks "b" instead. So the region MEMBERSHIP differs.
    import geopandas as gpd
    from pyproj import CRS
    from shapely.geometry import Polygon
    utm = CRS.from_epsg(32643)
    polys = {"s": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
             "a": Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),      # east, touches s at x=1
             "b": Polygon([(-1, 0), (0, 0), (0, 1), (-1, 1)])}    # west, touches s at x=0
    return gpd.GeoDataFrame({"block_id": list(polys), "building_count": [10.0, 10.0, 10.0]},
                            geometry=list(polys.values()), crs=utm)


def test_dense_cluster_grows_by_depth_fn_not_proxy() -> None:
    from reblock.region import DenseClusterRegionBuilder
    gdf = _fork_gdf()
    builder = DenseClusterRegionBuilder(max_buildings=15)        # seed(10) + exactly one more
    depth = {"s": 5.0, "a": 1.0, "b": 9.0}
    # depth-growth picks the deeper neighbor b; proxy-growth (equal proxy) ties to a by id.
    assert builder.build(gdf, [["s"]], depth_fn=lambda bid: depth[bid]) == [["b", "s"]]
    assert builder.build(gdf, [["s"]]) == [["a", "s"]]           # proxy tie -> "a"


def test_dense_cluster_depth_fn_none_is_proxy_behaviour() -> None:
    # depth_fn=None must be byte-identical to omitting it (both the proxy path).
    from reblock.region import DenseClusterRegionBuilder
    gdf = _fork_gdf()
    builder = DenseClusterRegionBuilder(max_buildings=15)
    assert builder.build(gdf, [["s"]], depth_fn=None) == builder.build(gdf, [["s"]])
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_region.py -k "depth_fn" -v`
Expected: FAIL with `TypeError: build() got an unexpected keyword argument 'depth_fn'`.

- [ ] **Step 3: Add `depth_fn` to the protocol + all three builders**

In `src/reblock/region.py`:

Protocol (L128):
```python
    def build(self, block_geoms: gpd.GeoDataFrame, groups: list[list[str]],
              depth_fn: Callable[[str], float] | None = None) -> list[list[str]]: ...
```

Add `from collections.abc import Callable` to the imports if absent.

`IdentityRegionBuilder.build` (L168) and `ConvexHullRegionBuilder.build` (L196) — add the parameter
and ignore it (they don't grow by depth):
```python
    def build(self, block_geoms: gpd.GeoDataFrame, groups: list[list[str]],
              depth_fn: Callable[[str], float] | None = None) -> list[list[str]]:
        del depth_fn   # these builders don't rank by depth
        # ... existing body unchanged ...
```

`DenseClusterRegionBuilder.build` (L273) — add the parameter and rank the frontier by `depth_fn`
when provided (local cache avoids re-evaluating a candidate across growth iterations), else the proxy:
```python
    def build(self, block_geoms: gpd.GeoDataFrame, groups: list[list[str]],
              depth_fn: Callable[[str], float] | None = None) -> list[list[str]]:
```
Then, just before the `for group in groups:` loop, add the cache + scorer:
```python
        depth_cache: dict[str, float] = {}

        def _score(j: int) -> float:
            if depth_fn is None:
                return _depth_proxy(counts[j], areas[j], perims[j])
            bid = ids[j]
            if bid not in depth_cache:
                depth_cache[bid] = depth_fn(bid)
            return depth_cache[bid]
```
And change the frontier pick from `-_depth_proxy(counts[j], areas[j], perims[j])` to `-_score(j)`:
```python
                best = min(
                    frontier,
                    key=lambda j: (-_score(j), -counts[j], ids[j]),
                )
```

- [ ] **Step 4: Inject `depth_fn` in `build_regions`**

In `src/reblock/pipeline.py`, `build_regions`, replace line 96
(`regions = region_builder.build(block_geoms, groups)[:max_blocks]`) with:

```python
    depth_fn = ((lambda bid: block_max_depth(source, bid))
                if getattr(source, "blocks_path", None) is not None else None)
    regions = region_builder.build(block_geoms, groups, depth_fn)[:max_blocks]
```

Add the import at the top of `pipeline.py`: `from reblock.region import ... block_max_depth ...`
(the file already imports `IdentityRegionBuilder, RegionBuilder, region_reblock` from `reblock.region`
— add `block_max_depth` to that line).

- [ ] **Step 5: Run the region + pipeline suites**

Run: `pixi run pytest tests/test_region.py tests/test_pipeline.py -v`
Expected: PASS — the two new tests + every existing region/pipeline test (the `depth_fn` default is
`None`, and `build_regions` now passes a real `depth_fn`, but existing pipeline tests over a
non-kblock or kblock source still resolve — verify none regress).

- [ ] **Step 6: Lint + type-check**

Run: `pixi run ruff check src/reblock/region.py src/reblock/pipeline.py tests/test_region.py && pixi run mypy --strict src/reblock/region.py src/reblock/pipeline.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/reblock/region.py src/reblock/pipeline.py tests/test_region.py
git commit -m "feat: region builder grows by true depth via injected depth_fn (proxy = fallback)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

## Task 4: `region_map` true-depth coloring + blank deselected; delete `_screen_proxy`

**Files:**
- Modify: `src/reblock/emit.py` (delete `_screen_proxy` L149–158; rewrite `region_map` coloring L161+)
- Modify: `src/reblock/run.py` (`region_map` call, L76)
- Test: `tests/test_emit.py`

**Interfaces:**
- Consumes: `block_max_depth(source, block_id)` (Task 1); `DenseCompactScreen.selection_depths` (Task 2, duck-typed).
- Produces: `region_map(source, regions, seed_groups, out_dir, *, selection: list[str] | None = None, depths: dict[str, float] | None = None) -> Path | None`.

- [ ] **Step 1: Write the failing test**

Replace `tests/test_emit.py::test_screen_proxy_is_squared_depth_and_rank_preserving` (delete it — the
helper is gone) and add coloring tests. Extend `_FakeSource` / `_source_with_neighbour_and_points`
already in the file (block "g" + "neighbour"):

```python
def test_screen_proxy_helper_is_gone() -> None:
    # migrate-not-accommodate: the squared-proxy coloring helper is deleted, not retained.
    import reblock.emit as emit
    assert not hasattr(emit, "_screen_proxy")


def test_region_map_colors_by_depth_and_blanks_deselected(tmp_path: Path) -> None:
    # With a selection + depths map, screen.png colors only the selected block ("g") by its true
    # depth; the deselected block ("neighbour") is blanked. Written without error.
    src = _source_with_neighbour_and_points()
    out = region_map(src, [["g"]], [["g"]], tmp_path,
                     selection=["g"], depths={"g": 7.0})
    assert out is not None and out.exists()
    assert (tmp_path / "screen.png").stat().st_size > 0


def test_region_map_without_depths_still_writes(tmp_path: Path) -> None:
    # depths=None (a non-DenseCompact screen) falls back to a flat located map -- no proxy coloring.
    out = region_map(_source_with_neighbour_and_points(), [["g"]], [["g"]], tmp_path)
    assert out is not None and out.exists()
```

The existing `test_region_map_draws_member_and_context_points` and
`test_region_map_guards_empty_building_points` call `region_map(src, [["g"]], [["g"]], tmp_path)`
with no `selection`/`depths` — the new keyword-only params default to `None`, so they still pass
(they exercise the flat fallback). Leave them as is.

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_emit.py -k "region_map or screen_proxy" -v`
Expected: FAIL — `region_map` has no `selection`/`depths` kwargs; `_screen_proxy` still present.

- [ ] **Step 3: Delete `_screen_proxy`**

Remove the whole `_screen_proxy` function (`src/reblock/emit.py` L149–158). Remove `NDArray` from the
`emit.py` imports **only if** nothing else in the file uses it (grep: `pct_displaced` uses
`NDArray[np.float64]` — so keep the `NDArray` import).

- [ ] **Step 4: Rewrite `region_map` to color by true depth + blank deselected**

Replace `region_map`'s signature and body from the docstring through the two `save_render` calls.
New signature + the screen.png/region.png rendering (keep `frame_bbox`, `_point_disks`,
`_CONTEXT_PT`, `_OWN_PT`, `_POINT_RADIUS_M`, `Rectangle`, `save_render` usage as today):

```python
def region_map(source: Source, regions: list[list[str]],
               seed_groups: list[list[str]], out_dir: Path, *,
               selection: list[str] | None = None,
               depths: dict[str, float] | None = None) -> Path | None:
    """Two maps for a region build. `screen.png`: the metro coloured by TRUE peel max access-depth
    (`depths`, from the screen's fine pass) on the absolute 0..max ring scale -- a continuous ramp,
    no bucketing -- with screen-DESELECTED blocks blanked, and the whole expanded region located
    (dark member outline + a locator box), clipped to the bulk block extent. `region.png`: the
    region's member blocks coloured by that same true depth against dimmed context, the pre-expansion
    seed outlined heavily, plus building points. When `depths` is None/empty (no depth-capable
    screen), both maps fall back to a flat located fill (NO proxy colouring). `selection` is the
    screen's flagged block_ids; `depths` maps block_id -> true max access-depth. Writes both; returns
    the `region.png` path, or None if there are no regions."""
    from matplotlib.patches import Rectangle

    from reblock.region import block_max_depth
    if not regions:
        return None
    geoms = source.block_geometries()
    geoms["block_id"] = geoms["block_id"].astype(str)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_seed_ids = {b for seeds in seed_groups for b in seeds}
    all_member_ids = {b for region in regions for b in region}

    sel = set(selection) if selection else set()
    dmap: dict[str, float] = dict(depths) if depths else {}
    vmax = float(max(dmap.values())) if dmap else 1.0
    geoms["depth"] = geoms["block_id"].map(dmap)          # NaN where deselected / unknown
    flagged = geoms[geoms["block_id"].isin(sel)] if sel else geoms.iloc[:0]
    blanked = geoms[~geoms["block_id"].isin(sel)] if sel else geoms
    members = geoms[geoms["block_id"].isin(all_member_ids)]
    seeds = geoms[geoms["block_id"].isin(all_seed_ids)]
    frame = frame_bbox(members.geometry) if not members.empty else None

    # --- screen.png: flagged blocks by true depth (0..max, continuous), deselected blanked ---
    fig_s, ax_s = plt.subplots(figsize=(10, 10))
    if not blanked.empty:
        blanked.plot(ax=ax_s, color="white", edgecolor="#dcdcdc", linewidth=0.12)
    if not flagged.empty and dmap:
        flagged.plot(ax=ax_s, column="depth", cmap="YlOrRd", vmin=0, vmax=vmax,
                     edgecolor="#33333330", linewidth=0.12)
        sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, vmax))
        sm.set_array([])
        fig_s.colorbar(sm, ax=ax_s, fraction=0.03, pad=0.01,
                       label="access depth (parcels from a street)")
    if not members.empty:
        members.plot(ax=ax_s, facecolor="none", edgecolor="#111111", linewidth=0.5)
    if frame is not None:
        ax_s.add_patch(Rectangle((frame[0], frame[1]), frame[2] - frame[0], frame[3] - frame[1],
                                 linewidth=1.6, edgecolor="#111111", facecolor="none", zorder=10))
    bnd = geoms.geometry.bounds
    ax_s.set_xlim(float(bnd["minx"].quantile(0.01)), float(bnd["maxx"].quantile(0.99)))
    ax_s.set_ylim(float(bnd["miny"].quantile(0.01)), float(bnd["maxy"].quantile(0.99)))
    ax_s.set_aspect("equal")
    ax_s.set_axis_off()
    ax_s.set_title(f"true access-depth (0..{vmax:.0f} rings); {len(all_member_ids)} blocks reblocked")
    save_render(fig_s, out_dir / "screen.png")
    plt.close(fig_s)

    # --- region.png: members by true depth against dimmed context + seed outline + points ---
    # member depths: from `depths` where present, else peel on demand (memoized; few members).
    member_depth = {b: dmap.get(b) if b in dmap else block_max_depth(source, b)
                    for b in all_member_ids}
    m_vmax = float(max([v for v in member_depth.values() if v] or [1.0]))
    members = members.copy()
    members["depth"] = members["block_id"].map(member_depth)
    fig_r, ax_r = plt.subplots(figsize=(10, 10))
    geoms.plot(ax=ax_r, color="#eeeeee", edgecolor="#cccccc", linewidth=0.3)
    if not members.empty and member_depth:
        members.plot(ax=ax_r, column="depth", cmap="YlOrRd", vmin=0, vmax=m_vmax,
                     edgecolor="#8a8a8a", linewidth=0.4)
    elif not members.empty:
        members.plot(ax=ax_r, color="#c0392b", edgecolor="#8a8a8a", linewidth=0.4)
    if not seeds.empty:
        seeds.plot(ax=ax_r, facecolor="none", edgecolor="black", linewidth=2.2)
    if frame is not None:
        ax_r.set_xlim(frame[0], frame[2])
        ax_r.set_ylim(frame[1], frame[3])
        pts = source.building_points(frame)
        if not pts.empty:
            members_union = members.geometry.union_all()
            own_pts = cast(gpd.GeoDataFrame, pts[pts.within(members_union)])
            context_pts = cast(gpd.GeoDataFrame, pts[~pts.within(members_union)])
            if not context_pts.empty:
                _point_disks(context_pts, _POINT_RADIUS_M).plot(
                    ax=ax_r, color=_CONTEXT_PT, alpha=0.6, linewidth=0)
            if not own_pts.empty:
                _point_disks(own_pts, _POINT_RADIUS_M).plot(ax=ax_r, color=_OWN_PT, linewidth=0)
    ax_r.set_aspect("equal")
    ax_r.set_axis_off()
    ax_r.set_title(f"{len(all_member_ids)} member block(s); {len(seeds)} seed(s) outlined")
    out_path = out_dir / "region.png"
    save_render(fig_r, out_path)
    plt.close(fig_r)
    return out_path
```

Update `region_map`'s module-level references: the two old proxy comments and the `has_proxy`/`proxy`
column logic are gone (replaced above). Confirm `np` is still used elsewhere in `emit.py` (it is —
`_displaced_points`); leave its import.

- [ ] **Step 5: Wire `run.py` to pass `selection` + `depths`**

In `src/reblock/run.py`, replace the `region_map` call (L76):

```python
    if cfg.region_map.enabled:
        sd = getattr(spec.screen, "selection_depths", None)
        depths = sd(spec.source) if sd is not None else None
        region_map(spec.source, output.regions, output.seed_groups, out_dir,
                   selection=output.selection, depths=depths)
```

(`spec.source.block_ids` was already set to `None` at line 56; `selection_depths` keys on the source
paths, not `block_ids`, so it's a cache hit regardless.)

- [ ] **Step 6: Run the emit + run suites**

Run: `pixi run pytest tests/test_emit.py tests/test_run.py -v`
Expected: PASS — new coloring/blank tests, `_screen_proxy` gone, existing `region_map` file-write
tests still green via the flat fallback.

- [ ] **Step 7: Lint + type-check**

Run: `pixi run ruff check src/reblock/emit.py src/reblock/run.py tests/test_emit.py && pixi run mypy --strict src/reblock/emit.py src/reblock/run.py`
Expected: no errors.

- [ ] **Step 8: Full check**

Run: `pixi run check`
Expected: ruff + mypy --strict + pytest all green.

- [ ] **Step 9: Commit**

```bash
git add src/reblock/emit.py src/reblock/run.py tests/test_emit.py
git commit -m "feat: region_map colors by true access-depth (0..max), blanks deselected; drop _screen_proxy

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

## Task 5: Regenerate the multiblock example

**Files:**
- Modify: `examples/multiblock/screen.jpg`, `examples/multiblock/region.jpg`, `examples/multiblock/README.md` (§1 prose), `examples/multiblock/run.log`

**Interfaces:**
- Consumes: Tasks 1–4 (the true-depth coloring flows through `reblock.run … region_map.enabled=true`).

- [ ] **Step 1: Regenerate the maps**

```bash
pixi run python -m reblock.run \
  data=capetown_full screen=dense_compact max_blocks=1 \
  region_builder=dense_cluster region_builder.max_buildings=3000 \
  method=clearance method.depth_target=3 method.max_roads=2000 \
  eval=kcomplexity render.enabled=true region_map.enabled=true \
  hydra.run.dir=/tmp/tde_region_run 2>&1 | tee examples/multiblock/run.log
```

Convert `/tmp/tde_region_run/screen.png` and `region.png` to the gallery JPEGs (same downsize the
current figures use):

```bash
pixi run python -c "from PIL import Image; \
  Image.open('/tmp/tde_region_run/screen.png').convert('RGB').save('examples/multiblock/screen.jpg', quality=85); \
  Image.open('/tmp/tde_region_run/region.png').convert('RGB').save('examples/multiblock/region.jpg', quality=85)"
```

Verify `screen.jpg` is titled `true access-depth (0..N rings); …`, colors only the flagged blocks by
depth, blanks the rest, and carries a colorbar.

- [ ] **Step 2: Update §1 prose in `examples/multiblock/README.md`**

Rewrite §1's coloring sentence(s): the screen still **ranks/gates** on the cheap proxy `√(n·A)/P`
(unchanged), but the map is now colored by the screen's **true peel access-depth** on the absolute
0–max ring scale, with screen-deselected blocks **blanked** (only the flagged deep fabric carries
color; sparse low-density blocks render pale because they genuinely are shallow). Keep the
`depth² = n × compactness` formula paragraph describing the *gate*; retitle the figure caption to
match. Do not touch §2–§4.

- [ ] **Step 3: Verify the check + consistency**

Run: `pixi run check`
Expected: green.

Re-read §1: the figure `screen.jpg` exists, the prose matches (screen gates on proxy, colors by true
depth, deselected blank), no stale "squared proxy coloring" claim remains.

- [ ] **Step 4: Commit**

```bash
git add examples/multiblock/screen.jpg examples/multiblock/region.jpg examples/multiblock/README.md examples/multiblock/run.log
git commit -m "docs: recolor multiblock screen map by true access-depth, blank deselected

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

## Self-Review

**1. Spec coverage:**
- §3.1 `block_max_depth` → Task 1. ✓
- §3.2 region builder `depth_fn` + peel non-flagged on demand + proxy fallback → Task 3. ✓
- §3.3 screen exposes depths (screen_selection pairs, protocol unchanged, `selection_depths`,
  run.py duck-type) → Task 2 (+ run.py wiring in Task 4). ✓
- §3.4 region_map true-depth coloring + blank deselected + delete `_screen_proxy` → Task 4. ✓
- §3.5 regenerate example → Task 5. ✓
- §7 continuous colormap (no scheme), migrate-not-accommodate (delete `_screen_proxy`), proxy only in
  cheap gate + region-builder fallback → Tasks 3, 4 + Global Constraints. ✓
- §5 cheap gate keeps proxy (no task touches `dense_compact._cheap_survivors`/`_depth_proxy` gate) ✓;
  no change to selection outcome (Task 2 preserves `select()` ids/order) ✓.

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code; the one conditional
(`depth_fn` None vs provided) has both arms specified.

**3. Type consistency:** `block_max_depth(source, block_id) -> float` identical in Tasks 1/3/4.
`RegionBuilder.build(..., depth_fn: Callable[[str], float] | None = None)` identical across the
protocol + all three builders + the `build_regions` call. `screen_selection -> list[tuple[str,
float]]` and `select -> list[str]` / `selection_depths -> dict[str, float]` consistent in Tasks 2/4.
`region_map(..., *, selection, depths)` identical in Task 4 code, tests, and the run.py call.

## Execution Handoff

Execution is **subagent-driven-development** (owner's standing preference — not asked). Fresh
implementer per task + task review (spec + quality) between tasks + a final whole-branch review.
Task 5 is compute-heavy (region-scale) and doc-heavy — run it last, after Tasks 1–4 are green.
