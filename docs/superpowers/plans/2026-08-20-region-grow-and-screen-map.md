# RegionGrow and ScreenMap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the last two widgets of site piece D — `RegionGrow` (the greedy region builder, run live in the browser) and `ScreenMap` (16,451 Cape Town + 3,500 Nairobi blocks, the screening floor as a live prefix) — both on the Screening page.

**Architecture:** Both bundles ship raw quantities (`building_count`, area, perimeter, geometry) and compute their model client-side, the pattern D2 established for displacement. `RegionGrow` runs the production greedy over a precomputed adjacency list, so the budget slider is live and any block is a seed. `ScreenMap` sorts blocks by the chosen metric once, making the floor a prefix length — selection, precision/recall and redraw all become O(prefix). One `RegionBuilder` contract change (build order, plus an adjacency-CRS correctness fix) supplies the fixtures that pin the TypeScript to production.

**Tech Stack:** Python 3 / geopandas / shapely / matplotlib (bakes); TypeScript / esbuild / `node:test` (widgets); mkdocs-material (site).

**Spec:** `docs/superpowers/specs/2026-08-20-region-grow-and-screen-map-design.md`

## Global Constraints

Copied verbatim from the spec and from standing project rules. **Every task's requirements implicitly include this section.**

* **`scripts/gen_site_pages.py` must stay stdlib-only and must NEVER import `reblock`.** CI runs it with only mkdocs-material installed.
* **`docs/js/` and `docs/assets/` are gitignored.** Never stage them. `web/scripts/test.sh` rebuilds the bundle as its first line with `|| exit 1`.
* **Bundles and their `.d.ts` are generated and committed, never hand-edited.** A `.d.ts` is copied into `web/src/` by its generator.
* **No `# type: ignore` and no mypy excludes as fixes.** No unreachable guards.
* **Never reach into a closed, known-at-authoring-time set with a runtime string, position or count.** Dynamic access must have no default, so an unknown name raises.
* **mypy dual-list hazard:** `pyproject.toml`'s `typecheck-py` passes **explicit file arguments that override `[tool.mypy] files` entirely**. Both lists must be updated for every new script module. `tests/test_typecheck_config.py` pins them together.
* **The simplification tolerance is 5 m for the city tier and 1 m for the region neighbourhood** (spec §1.1, §1.2).
* **Both new bundles carry ring LISTS, not single rings**, and must not use `_bundle_io.polygon_ring`, which raises on holes. 6,990 Cape Town blocks, 1,139 Nairobi blocks, and **7 of `RegionGrow`'s 213 blocks** have interior rings (spec §3.3).
* **Any code path calling `region.build()` or `_block_adjacency` must supply projected geometry.** `STREET_TOL = 0.5` is interpreted in the frame's own units; in lon/lat that is ~55 km (spec §1.5).
* **Test acceptance is fault injection: break the thing the test guards, observe red, restore.** An injection that will not redden is REPORTED, not tuned until it passes. See `~/wiki/pages/methodology/tests-that-cannot-fail.md` — D2 shipped nine tests that passed while guarding nothing.
* **Tests that load city data are ONE `@pytest.mark.slow` test** carrying every assertion needing the load. `pytest-xdist` scopes `scope="module"` fixtures per worker, not per session; D2 lost 18 minutes to this. Pattern: `tests/test_frontier_bundle.py`.
* **No invented numbers.** Every figure in a docstring, caption or comment is measured or it does not appear.
* Run scripts as `python -m scripts.<name>` — `pythonpath` is configured for pytest only.

---

## File Structure

**Python — modified**

| file | responsibility after this plan |
|---|---|
| `src/reblock/region.py` | `_projected()` helper; four builders use projected geometry; growing builders return accretion order; `_shared_parts` sorts its members |
| `scripts/_bundle_io.py` | gains `polygon_rings()` beside `polygon_ring()` — multi-ring encoder for holed polygons |
| `pyproject.toml` | two new script modules in **both** mypy lists |
| `scripts/gen_site_pages.py` | `REGIONGROW` + `SCREENMAP` markers and producers |
| `docs/_partials/screening.md` | two mount markers |

**Python — created**

| file | responsibility |
|---|---|
| `scripts/gen_region_grow.py` | bakes `examples/region-grow/` — hood bundle, fixtures, fallback PNG, README, `.d.ts` |
| `scripts/gen_screen_map.py` | bakes `examples/screen-map/` — two city bundles, fallback PNG, README, `.d.ts` |
| `tests/test_region_build_order.py` | accretion order, CRS correctness, `_shared_parts` order-independence |
| `tests/test_region_grow_bundle.py` | hood bundle schema, `.d.ts` parity, fixture/production identity |
| `tests/test_screen_map_bundle.py` | city bundle schema, `.d.ts` parity, precision/recall against the committed CSV |

**TypeScript — created**

| file | responsibility |
|---|---|
| `web/test/harness.ts` | the shared fake DOM + recording canvas, extracted from two duplicated copies |
| `web/src/hood.d.ts` | generated by `gen_region_grow.py` |
| `web/src/screen_map.d.ts` | generated by `gen_screen_map.py` |
| `web/src/model/accretion.ts` | the greedy, mirroring `DenseClusterRegionBuilder` |
| `web/src/model/screen.ts` | four metrics, metric-sorted order, prefix precision/recall |
| `web/src/render/region.ts` | neighbourhood + grown-region canvas draw |
| `web/src/render/city.ts` | base layer + selected prefix, `Path2D` cache |
| `web/src/widgets/region-grow.ts` | budget slider, click-to-reseed, readout |
| `web/src/widgets/screen-map.ts` | metric select, floor slider, city toggle, precision/recall readout |
| `web/test/accretion.test.ts`, `web/test/screen-model.test.ts` | model unit tests |
| `web/test/region-grow-boot.test.ts`, `web/test/screen-map-boot.test.ts` | widget boot tests |

**TypeScript — modified:** `web/src/mount.ts` (two registrations, after `REGISTRY` exists), `web/test/field-boot.test.ts` + `web/test/perm-graph-boot.test.ts` (consume the extracted harness).

---

## Task 1: `RegionBuilder` build order and adjacency CRS

**Files:**
- Modify: `src/reblock/region.py` — `_shared_parts` (53-72), `IdentityRegionBuilder.build` (171-186), `ConvexHullRegionBuilder.build` (~200), `DenseClusterRegionBuilder.build` (299-354), `ShapeStandardizingRegionBuilder.build` (475-518), `RegionBuilder` Protocol docstring (123-131)
- Test: `tests/test_region_build_order.py` (create), `tests/test_region.py`, `tests/test_shape_standardizing_region.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `RegionBuilder.build(block_geoms, groups, depth_fn=None) -> list[list[str]]` where each inner list is in **build order** — accretion order for `DenseClusterRegionBuilder` / `ShapeStandardizingRegionBuilder`, `sorted()` for `IdentityRegionBuilder` / `ConvexHullRegionBuilder`. Task 4 depends on this order being production's own.
- Produces: `region._projected(block_geoms: gpd.GeoDataFrame) -> gpd.GeoDataFrame`.

### Why this task exists

Two independent defects, fixed together because they are three lines apart:

1. **Accretion order is discarded.** Both growing builders end `result.append(sorted(ids[i] for i in cluster))`, throwing away the order they just computed — which is `RegionGrow`'s entire teaching point. Without it, Task 4's fixtures could only pin set membership, and Task 5's order test would have to compare against a hand-rolled copy of the greedy. That was D2's defect #3.

2. **Adjacency runs in the caller's CRS.** `_block_adjacency` does `STRtree.query(..., predicate="dwithin", distance=STREET_TOL)` with `STREET_TOL = 0.5`. Both growing builders reproject **for the metric only**, with a comment stating that the original-CRS geometries still drive adjacency. On a lon/lat frame that makes 0.5 mean 0.5 **degrees** — about 55 km — so every block in a metro becomes adjacent to every other and growth assembles a region from blocks kilometres apart, silently. `scripts/pair_matrix.py:304` already feeds `build()` an unprojected frame; it is harmless there only because its default builder is `IdentityRegionBuilder` over singleton groups, which never reaches the adjacency call.

Production output does not change: `KblockSource.block_geometries()` reprojects to UTM before any builder sees the frame, so the reprojection below is a no-op on every shipped path.

### Order-safety, and why `_shared_parts` changes

Member order is **not** inert downstream:

```python
parcels = pd.concat([b.parcels for b in blocks], ignore_index=True)
parcels["parcel_id"] = range(len(parcels))
```

Member order renumbers every parcel — while `block_id` (`"region:" + "+".join(sorted(...))`) and `source_content_hash` (`sorted(...)` of member hashes) both stay the same. A cached derivation keyed on that unchanged hash would be reused against differently numbered parcels: silent corruption. So `_shared_parts` sorts its own members, where the sensitivity lives. `pipeline.build_regions` already passes sorted members, so this is a no-op today and protects every future caller.

- [ ] **Step 1: Write the failing CRS test**

Create `tests/test_region_build_order.py`:

```python
"""Build order, and adjacency measured in the frame's own units.

`_block_adjacency` runs `dwithin(STREET_TOL)` with `STREET_TOL = 0.5`. That is 0.5 metres in a
projected frame and ~55 km in lon/lat, so a builder handed a geographic frame used to treat every
block in a metro as adjacent to every other -- and returned a plausible-looking region assembled
from blocks kilometres apart, with nothing raised. See the design's §1.5.
"""
from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from reblock.region import (
    ConvexHullRegionBuilder,
    DenseClusterRegionBuilder,
    IdentityRegionBuilder,
    ShapeStandardizingRegionBuilder,
)

# A 4x4 grid of 100 m blocks with a 2 m street gap, placed in UTM 33S near Cape Town. 100 m is
# small enough that 0.5 DEGREES (~55 km) swallows the whole grid, which is what makes the
# geographic-frame bug observable at all.
CELL, GAP, ORIGIN_X, ORIGIN_Y = 100.0, 2.0, 260000.0, 6240000.0


def _grid(counts: dict[tuple[int, int], float] | None = None) -> gpd.GeoDataFrame:
    ids, polys, ns = [], [], []
    for r in range(4):
        for c in range(4):
            x0 = ORIGIN_X + c * (CELL + GAP)
            y0 = ORIGIN_Y + r * (CELL + GAP)
            ids.append(f"{r}_{c}")
            polys.append(Polygon([(x0, y0), (x0 + CELL, y0), (x0 + CELL, y0 + CELL),
                                  (x0, y0 + CELL)]))
            ns.append((counts or {}).get((r, c), 10.0))
    return gpd.GeoDataFrame({"block_id": ids, "building_count": ns},
                            geometry=polys, crs="EPSG:32734")


@pytest.mark.parametrize("builder", [
    DenseClusterRegionBuilder(max_buildings=40),
    ShapeStandardizingRegionBuilder(max_buildings=40),
    IdentityRegionBuilder(),
    ConvexHullRegionBuilder(),
], ids=["dense_cluster", "shape_standardizing", "identity", "convex_hull"])
def test_geographic_frame_grows_the_same_region_as_its_projected_twin(builder: object) -> None:
    """A frame's CRS must not change which blocks a builder picks.

    This is the whole of the §1.5 bug: `dwithin(0.5)` in lon/lat is ~55 km, so before the fix
    `dense_cluster` on the geographic twin pulled in blocks by score alone, ignoring adjacency.
    """
    utm = _grid()
    geo = utm.to_crs("EPSG:4326")
    assert builder.build(utm, [["1_1"]]) == builder.build(geo, [["1_1"]])   # type: ignore[attr-defined]
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pixi run pytest tests/test_region_build_order.py -v`
Expected: FAIL for `dense_cluster` and `shape_standardizing` — the geographic frame grows a different member set. `identity` and `convex_hull` may already pass; that is fine, they are there so the fix is applied uniformly and stays applied.

**If a parametrisation passes before the fix, say so in the report.** A test that was already green is not evidence for this change.

- [ ] **Step 3: Add `_projected` and use it in all four builders**

In `src/reblock/region.py`, after `_check` (around line 51):

```python
def _projected(block_geoms: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """`block_geoms` in a METRIC frame, reprojected from a geographic one if needed.

    Every builder needs this for two separate reasons that used to be handled separately and
    inconsistently. The growing builders reprojected for their METRIC (`sqrt(n*A)/P` and the shape
    objectives are meaningless where area and length are anisotropic) but deliberately left
    ADJACENCY on the caller's frame -- and adjacency is `_block_adjacency`'s
    `dwithin(STREET_TOL)`, with `STREET_TOL = 0.5`. That is 0.5 metres in UTM and about 55 km in
    lon/lat, so a geographic frame made every block in a metro adjacent to every other and growth
    silently assembled regions from blocks kilometres apart.

    A no-op on every shipped path -- `KblockSource.block_geometries()` already returns UTM -- so
    this changes no production output. It exists because `scripts/pair_matrix.py:304` reads its
    frame straight out of the parquet (lon/lat) and reaches `build()` through a
    `cast(gpd.GeoDataFrame, ...)`, which is a type-checker assertion and not a runtime conversion.
    """
    if block_geoms.crs is not None and block_geoms.crs.is_geographic:
        return cast(gpd.GeoDataFrame, block_geoms.to_crs(block_geoms.estimate_utm_crs()))
    return block_geoms
```

In `DenseClusterRegionBuilder.build`, replace the geoms/metric preamble:

```python
        _validate_group_ids(block_geoms, groups)
        metric = _projected(block_geoms)
        ids = cast(list[str], list(block_geoms["block_id"]))
        # ONE frame drives everything: adjacency, the touch-adjacency warning AND the metric. The
        # previous split -- metric reprojected, adjacency on the caller's frame -- is the §1.5 bug.
        geoms = list(metric.geometry)
        areas = [float(g.area) for g in metric.geometry]
        perims = [float(g.length) for g in metric.geometry]
```

and delete the old four-line `metric = block_geoms; if ... to_crs(...)` block. Make the identical
substitution in `ShapeStandardizingRegionBuilder.build` (where `shape_geoms` and `geoms` both
become `list(metric.geometry)` — one list, used for both).

In `IdentityRegionBuilder.build` and `ConvexHullRegionBuilder.build`, insert
`block_geoms = _projected(block_geoms)` immediately after `_validate_group_ids(...)`, so their
`_touch_adjacent` warning (and the convex hull, which is not CRS-invariant) are computed in metres
too.

- [ ] **Step 4: Run the CRS test and watch it pass**

Run: `pixi run pytest tests/test_region_build_order.py -v`
Expected: PASS, all four parametrisations.

- [ ] **Step 5: Write the failing build-order tests**

Append to `tests/test_region_build_order.py`:

```python
def test_dense_cluster_returns_accretion_order_not_sorted_order() -> None:
    """The order blocks were ADDED in, which is what RegionGrow teaches and what pins its
    TypeScript to production. A sorted result throws it away.

    The grid is rigged so accretion order and sorted order disagree: the seed is `1_1`, and its
    neighbours are weighted so the greedy walks DOWN-then-LEFT while `sorted()` would put `0_1`
    before `1_0`. If this test ever passes against a `sorted()` implementation, the rigging has
    stopped working and the test guards nothing -- check by reverting the builder.
    """
    # depth proxy is sqrt(n*A)/P; A and P are equal for every cell, so a higher count wins.
    grid = _grid({(1, 0): 30.0, (0, 1): 20.0, (2, 1): 15.0})
    got = DenseClusterRegionBuilder(max_buildings=70).build(grid, [["1_1"]])[0]

    assert got[0] == "1_1", "the seed comes first"
    assert got == ["1_1", "1_0", "0_1", "2_1"], got
    assert got != sorted(got), "accretion order must not coincide with sorted order here"


def test_shape_standardizing_returns_accretion_order() -> None:
    """Same contract, the other growing builder."""
    grid = _grid()
    got = ShapeStandardizingRegionBuilder(max_buildings=40).build(grid, [["1_1"]])[0]
    assert got[0] == "1_1", "the seed comes first"
    assert len(got) == len(set(got)), "no block appears twice"


def test_non_growing_builders_return_sorted_order() -> None:
    """`identity` and `convex_hull` have no accretion to report, so sorted IS their build order --
    stated as a test so the contract is one sentence for all four builders."""
    grid = _grid()
    assert IdentityRegionBuilder().build(grid, [["1_1", "0_0"]]) == [["0_0", "1_1"]]
    hull = ConvexHullRegionBuilder().build(grid, [["0_0", "1_1"]])[0]
    assert hull == sorted(hull)
```

- [ ] **Step 6: Run them and watch them fail**

Run: `pixi run pytest tests/test_region_build_order.py -v`
Expected: FAIL on `test_dense_cluster_returns_accretion_order_not_sorted_order` — the builder still returns sorted order, so `got != sorted(got)` fails.

- [ ] **Step 7: Return accretion order**

In `DenseClusterRegionBuilder.build`, replace the cluster loop's bookkeeping:

```python
            cluster = {idx_by_id[b] for b in group}
            order = sorted(cluster, key=lambda i: ids[i])   # the seed group, deterministically
            size = sum(counts[i] for i in cluster)
            while size < self.max_buildings:
                frontier = {j for i in cluster for j in adj[i]} - cluster
                if not frontier:
                    break
                best = min(
                    frontier,
                    key=lambda j: (-_score(j), -counts[j], ids[j]),
                )
                cluster.add(best)
                order.append(best)
                size += counts[best]
            result.append([ids[i] for i in order])
```

Make the matching change in `ShapeStandardizingRegionBuilder.build` (same `order` list, appended at
the same point `cluster.add(best)` happens).

Rewrite the `RegionBuilder` Protocol docstring's last clause:

```python
class RegionBuilder(Protocol):
    """Maps user seed groups to expanded region member groups, on cheap block GEOMETRIES (no
    Voronoi) -- so members are chosen before the expensive full-Block build. `groups` is a list
    of seed groups (block_ids); returns the expanded groups (block_ids), each in BUILD ORDER,
    group order preserved.

    Build order means accretion order where there is one: `DenseClusterRegionBuilder` and
    `ShapeStandardizingRegionBuilder` return the seed group (sorted) followed by each block in the
    order it was added, which is what the site's RegionGrow widget replays and what pins its
    browser-side greedy to this code. `IdentityRegionBuilder` and `ConvexHullRegionBuilder` have no
    accretion, so sorted IS their build order. Determinism is unchanged either way -- accretion
    order is fixed by the tie-break rule.

    Consumers that need a SET must sort: `region._shared_parts` does, because it numbers parcels by
    member order and a renumbering under an unchanged `source_content_hash` would corrupt cached
    derivations.
    """
```

- [ ] **Step 8: Run and watch them pass**

Run: `pixi run pytest tests/test_region_build_order.py -v`
Expected: PASS.

- [ ] **Step 9: Write the failing order-independence test for `_shared_parts`**

Append to `tests/test_region_build_order.py`:

```python
def test_region_block_is_independent_of_member_order() -> None:
    """`region_block` must give the same parcels whatever order its members arrive in.

    `_shared_parts` does `pd.concat([b.parcels for b in blocks])` then
    `parcels["parcel_id"] = range(len(parcels))`, so member order RENUMBERS every parcel -- while
    `block_id` and `source_content_hash` are both built from `sorted(...)` and do NOT change. A
    cached derivation keyed on that unchanged hash would then be reused against differently
    numbered parcels. Task 1 makes builders return accretion order, which is exactly when unsorted
    members become reachable, so this is the guard that makes that change safe.
    """
    from tests.scoring_fixtures import two_adjacent_blocks   # see the module's own docstring
    from reblock.region import region_block

    a, b = two_adjacent_blocks()
    forward = region_block([a, b])
    reverse = region_block([b, a])

    assert forward.block_id == reverse.block_id
    assert forward.source_content_hash == reverse.source_content_hash
    assert list(forward.parcels["parcel_id"]) == list(reverse.parcels["parcel_id"])
    assert forward.parcels.geometry.equals(reverse.parcels.geometry), (
        "parcel geometry must be identical, not merely equivalent as a set")
```

**Implementer note:** `tests/scoring_fixtures.py` is the project's existing shared-fixture module.
If it has no `two_adjacent_blocks()` helper, build the two blocks inline in this test from the same
grid `_grid()` uses via whatever `Block` constructor the neighbouring region tests already use
(`tests/test_region.py` builds Blocks for exactly this purpose — copy its construction, not its
assertions). Report which route you took.

- [ ] **Step 10: Run it, then make it pass**

Run: `pixi run pytest tests/test_region_build_order.py::test_region_block_is_independent_of_member_order -v`
Expected: FAIL on the `parcel_id`/geometry comparison.

Then in `_shared_parts`, sort at the top:

```python
def _shared_parts(blocks: list[Block]) -> tuple[gpd.GeoDataFrame, Polygon | MultiPolygon, CRS, str]:
    """... (existing docstring) ...

    MEMBERS ARE SORTED BY block_id FIRST, and that is load-bearing rather than tidiness: the
    `parcel_id` assignment below numbers parcels by member order, while `block_id` and
    `source_content_hash` are both built from `sorted(...)` and would NOT change. Handed members in
    a different order, this would renumber every parcel under an identity the derivation cache
    treats as unchanged. A no-op when it landed -- `pipeline.build_regions` already passed sorted
    members -- and the reason builders may now return accretion order safely.
    """
    crs = _check(blocks, "region_block")
    blocks = sorted(blocks, key=lambda b: b.block_id)
```

Run again. Expected: PASS.

- [ ] **Step 11: Fix the existing region tests**

Run: `pixi run pytest tests/test_region.py tests/test_shape_standardizing_region.py -v`

Several assertions compare against sorted results. For each failure, decide which of two things it
was testing and rewrite accordingly:

* asserting **membership** (`set(got) == {...}`) — wrap in `sorted(...)` on both sides, and add a
  one-line comment saying membership is what is meant.
* asserting **sortedness itself** — that assertion is now wrong for the growing builders. Replace it
  with the build-order assertion the test actually wants.

`tests/test_shape_standardizing_region.py:127` (`runs = [... for _ in range(N)]`, a determinism
check) must keep comparing full lists — accretion order is deterministic, so it still holds and now
guards strictly more.

- [ ] **Step 12: Fault injection**

Prove each new guard can fail. For each, break it, run, confirm RED, restore:

| guard | injection |
|---|---|
| CRS equality | delete the `_projected(...)` call in `DenseClusterRegionBuilder.build` |
| accretion order | restore `result.append(sorted(ids[i] for i in cluster))` |
| `_shared_parts` order | delete the `blocks = sorted(...)` line |

**Report any injection that does not redden.** Do not adjust the test to make it redden.

- [ ] **Step 13: Full gate and commit**

Run: `pixi run lint && pixi run typecheck-py && pixi run pytest tests/test_region.py tests/test_shape_standardizing_region.py tests/test_region_build_order.py tests/test_pipeline.py -v`

```bash
git add src/reblock/region.py tests/test_region_build_order.py tests/test_region.py tests/test_shape_standardizing_region.py
git commit -m "fix: region builders measure adjacency in metres, and keep accretion order"
```

---

## Task 2: `polygon_rings` — the multi-ring encoder

**Files:**
- Modify: `scripts/_bundle_io.py`
- Test: `tests/test_bundle_io.py` (create if absent; otherwise extend)

**Interfaces:**
- Produces: `polygon_rings(geom: BaseGeometry, ox: float, oy: float, *, what: str) -> list[list[list[float]]]` — exterior ring first, then each interior ring, all origin-relative at `cm` precision. Raises on a non-Polygon. Consumed by Tasks 4 and 7.

### Why a second function rather than a flag on the first

`polygon_ring` raises on interior rings deliberately: the three bundles that use it give a polygon
exactly one ring, so a hole would have to lose geometry to fit, and geometry that silently vanishes
from a committed artifact is a wrong picture nobody is looking for. That contract is still right for
those three. The two new bundles have a different contract — they carry holes — so they get a
different function, not a parameter that lets a caller of the strict one opt out of its own guard.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bundle_io.py`:

```python
"""`_bundle_io`'s encoders. The multi-ring one exists because 6,990 Cape Town blocks, 1,139
Nairobi blocks and 7 of RegionGrow's 213 neighbourhood blocks have interior rings -- and
`polygon_ring`, which the three older bundles use, raises on every one of them."""
from __future__ import annotations

import pytest
from shapely.geometry import LineString, Polygon

from scripts._bundle_io import polygon_ring, polygon_rings


def _donut() -> Polygon:
    return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)],
                   [[(3, 3), (3, 6), (6, 6), (6, 3)]])


def test_polygon_rings_keeps_the_hole() -> None:
    rings = polygon_rings(_donut(), 0.0, 0.0, what="test block")
    assert len(rings) == 2, "exterior plus one interior"
    assert rings[0][0] == [0.0, 0.0], "exterior comes first"
    assert [3.0, 3.0] in rings[1], "the interior ring's coordinates survive"


def test_polygon_rings_is_origin_relative_at_cm_precision() -> None:
    rings = polygon_rings(_donut(), 1.0, 2.0, what="test block")
    assert rings[0][0] == [-1.0, -2.0]
    assert all(round(v, 2) == v for ring in rings for pt in ring for v in pt)


def test_polygon_rings_rejects_a_multipolygon() -> None:
    """Neither city has one, and the format gives a block one polygon -- so this raises rather
    than dropping a part. `what` names the offender, since a bundle has many blocks."""
    from shapely.geometry import MultiPolygon
    mp = MultiPolygon([_donut(), Polygon([(20, 20), (21, 20), (21, 21)])])
    with pytest.raises(ValueError, match="block 7"):
        polygon_rings(mp, 0.0, 0.0, what="block 7")


def test_polygon_ring_still_rejects_holes() -> None:
    """The strict encoder keeps its guard. The new function is a second contract, not an escape
    hatch from this one -- three shipped bundles depend on it raising."""
    with pytest.raises(ValueError, match="interior rings"):
        polygon_ring(_donut(), 0.0, 0.0, what="parcel 3")
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pixi run pytest tests/test_bundle_io.py -v`
Expected: FAIL with `ImportError: cannot import name 'polygon_rings'`.

- [ ] **Step 3: Implement it**

Append to `scripts/_bundle_io.py`:

```python
def polygon_rings(geom: BaseGeometry, ox: float, oy: float, *,
                  what: str) -> list[list[list[float]]]:
    """Every ring of a simple Polygon -- exterior first, then interiors -- origin-relative at `cm`.

    The multi-ring counterpart to `polygon_ring`, and a SEPARATE function rather than a flag on it.
    `polygon_ring` raises on interior rings because the three bundles that use it give a polygon
    exactly one ring, so a hole there would have to lose geometry to fit; that guard is still right
    and a parameter letting a caller switch it off would be a way to lose geometry quietly. The
    city tier and the region neighbourhood have a different contract: they CARRY holes, and their
    consumers fill with the even-odd rule.

    Measured on the data these bundles are baked from: 6,990 of 16,451 Cape Town blocks and 1,139
    of 3,500 Nairobi blocks have interior rings, as do 7 of the 213 blocks in RegionGrow's
    neighbourhood. Neither city has a single MultiPolygon block, which is why that case raises
    rather than being flattened.
    """
    if not isinstance(geom, Polygon):
        raise ValueError(
            f"{what} is a {geom.geom_type}, not a Polygon -- the bundle format gives a block one "
            f"polygon (measured: no MultiPolygon blocks in either city); report this instead of "
            f"silently dropping geometry")
    return [[[cm(x - ox), cm(y - oy)] for x, y in ring.coords]
            for ring in [geom.exterior, *geom.interiors]]
```

- [ ] **Step 4: Run and watch it pass**

Run: `pixi run pytest tests/test_bundle_io.py -v`
Expected: PASS.

- [ ] **Step 5: Fault injection**

Change `[geom.exterior, *geom.interiors]` to `[geom.exterior]`. Expected:
`test_polygon_rings_keeps_the_hole` RED. Restore.

- [ ] **Step 6: Commit**

```bash
git add scripts/_bundle_io.py tests/test_bundle_io.py
git commit -m "feat: polygon_rings, the multi-ring encoder for bundles that carry holes"
```

---

## Task 3: Extract the shared browser-test harness

**Files:**
- Create: `web/test/harness.ts`
- Modify: `web/test/field-boot.test.ts`, `web/test/perm-graph-boot.test.ts`, `web/test/frontier-boot.test.ts`

**Interfaces:**
- Produces: from `web/test/harness.ts` —
  `interface PathOp { op: "moveTo" | "lineTo" | "arc" | "closePath"; args: number[] }`,
  `interface Call { op: "clearRect" | "stroke" | "fill"; strokeStyle: string; fillStyle: string; lineWidth: number; lineCap: string; globalAlpha: number; path: PathOp[] }`,
  `class RecordingContext`, `class FakeElement`, `class FakeResizeObserver`,
  `function installStubs(): void`, `function fireResize(width: number, height: number): void`,
  `function armDrawFailure(message: string | null): void`,
  `function mountPoint(): FakeElement`, `function canvasOf(host: FakeElement): FakeElement`,
  `function lastFrame(cv: FakeElement): Call[]`.
  Consumed by Tasks 6 and 9.

### Scope, and what stays put

`FakeElement` is currently defined **five** times under `web/test/`. Only two of those copies are
genuine duplicates: `field-boot.test.ts` and `perm-graph-boot.test.ts` share the whole canvas stack
(`clock`, `NEXT_DRAW_FAILURE`, `RecordingContext`, `FakeElement`, `FakeResizeObserver`,
`mountPoint`, `canvasOf`). `frontier-boot.test.ts` shares the DOM half but has no
`RecordingContext` — Frontier renders SVG. `svg.test.ts` and `fallback.test.ts` have their own
narrower fakes for different jobs.

**Extract what the two canvas tests genuinely share. Migrate `frontier-boot.test.ts` only if it
needs no behavioural change; if it does, leave it and say so in the report.** Do not touch
`svg.test.ts` or `fallback.test.ts` — forcing one `FakeElement` to serve SVG attribute inspection,
canvas contexts and fallback child manipulation would be a worse abstraction than the duplication.

This task exists now because D3 adds two more widget boot tests. Without it there would be seven
copies. `scripts/_bundle_io.py`'s own docstring records what happened last time this was left:
"by the time the third arrived there were already two live copies of `sigfig` differing only in
name — one of them carrying a docstring that pointed at a function in the other file."

### This is a refactor, so the acceptance criterion is different

**No assertion may change.** The diff moves definitions and adds imports; it does not touch a single
`assert`. Two things prove that:

1. **The test names and counts are identical before and after.** Capture them first.
2. **Fault injection still reddens.** A harness refactor is exactly how a guard dies silently — a
   fake that stops recording `globalAlpha`, or a `fireResize` that fires synchronously when the real
   one is async, disables assertions without failing anything.

- [ ] **Step 1: Record the baseline**

```bash
cd web && npm ci && npm test 2>&1 | tee /tmp/harness-before.txt | tail -20
grep -c "^ok\|^not ok" /tmp/harness-before.txt
```

Record the pass count and the full list of test names. The report must quote both.

**The runner, verified.** `web/package.json`'s `test` script is `bash scripts/test.sh`: it
esbuild-builds `../docs/js/widgets.js`, compiles the suite with `tsc -p tsconfig.test.json`, then
hands `node --test` an explicit file list. There is no `tsx` in this project. The whole suite is
`npm test` (or `pixi run web-test`); one file is:

```bash
cd web && OUT=$(mktemp -d) && ./node_modules/.bin/tsc -p tsconfig.test.json --outDir "$OUT" \
  --noEmit false; node --test "$OUT/test/<name>.test.js"; rm -rf "$OUT"
```

Capture the baseline with the FULL suite: `test.sh` runs every file, and the before/after test-name
diff is what proves no assertion moved.

- [ ] **Step 2: Create `web/test/harness.ts`**

Move — do not copy — the shared definitions out of `field-boot.test.ts`, which has the most
developed versions (its `Call` records `lineCap` and `globalAlpha`; `perm-graph-boot.test.ts`'s does
not). Keep every comment: they explain why each field is recorded, and those explanations are the
only record of the bugs they were added for.

```ts
/** The shared fake DOM and recording 2D context every canvas widget test mounts against.
 *
 * Extracted from `field-boot.test.ts` and `perm-graph-boot.test.ts`, which had grown two copies of
 * the whole stack. The versions kept here are field-boot's: its `Call` records `lineCap` and
 * `globalAlpha`, which perm-graph's did not, and both are load-bearing (see `Call`).
 *
 * No jsdom on purpose -- one fake element class and one recording context, in the minimal-stub
 * spirit the three boot tests already shared. Neither the widgets' module bodies nor anything they
 * import touches `document`, `window` or `ResizeObserver` at evaluation time, only their function
 * bodies do, so a test file's static imports and the stubs `installStubs()` puts in place are in
 * the right order either way.
 */
```

Then the moved bodies, with these three deliberate changes:

* `let clock = 0` and `let NEXT_DRAW_FAILURE` become module state here, reached through
  `armDrawFailure(message)` rather than by assigning an exported `let` — an exported mutable binding
  is writable from any importer and reads as ordinary state rather than as a deliberate arming step.
* `installStubs()` wraps the `globalThis.ResizeObserver = FakeResizeObserver` assignments the two
  files each did at top level, so a consumer opts in with one call.
* `FakeResizeObserver` **must keep firing asynchronously.** D2's defect #5 was a fake that fired
  synchronously, which made the widget's `runOrReport` error path unreachable in every test that
  used it. Keep the `queueMicrotask` and keep the comment saying why.

- [ ] **Step 3: Point the two canvas tests at it**

In `field-boot.test.ts` and `perm-graph-boot.test.ts`, delete the moved definitions and add:

```ts
import {
  armDrawFailure, canvasOf, Call, fireResize, installStubs, lastFrame, mountPoint,
} from "./harness.js";

installStubs();
```

Keep every widget-specific helper where it is — `handleAt`, `layers`, `cost`,
`assertPictureMatchesRoads`, `drag`, `reference`, `unionBbox` are about `DisplacementField`, not
about faking a DOM, and moving them would make the harness a dumping ground.

- [ ] **Step 4: Verify the baseline is unchanged**

```bash
cd web && npm test 2>&1 | tee /tmp/harness-after.txt | tail -20
diff <(grep "^ok\|^not ok" /tmp/harness-before.txt | sed 's/[0-9]\+//') \
     <(grep "^ok\|^not ok" /tmp/harness-after.txt | sed 's/[0-9]\+//')
```

Expected: no differences. **A changed test name or count means an assertion moved; find it.**

- [ ] **Step 5: Fault injection on the harness itself**

Prove the moved fake still supports the guards that depend on its details. Break, run, confirm RED,
restore:

| harness detail | injection | must redden |
|---|---|---|
| `globalAlpha` recorded per call | record a constant `1` instead | a `field-boot` disk-shading assertion |
| `lineCap` recorded per call | drop the field, default `"butt"` | the corridor's round-cap assertion |
| async `FakeResizeObserver` | replace `queueMicrotask(...)` with a direct call | a `runOrReport` / draw-failure test |

**Report any injection that does not redden — it means that guard was already inert before this
task, which is a finding about the existing tests, not about the harness.**

- [ ] **Step 6: Commit**

```bash
cd .. && git add web/test/harness.ts web/test/field-boot.test.ts web/test/perm-graph-boot.test.ts web/test/frontier-boot.test.ts
git commit -m "test: extract the shared fake-DOM and recording-canvas harness"
```

---

## Task 4: Bake `examples/region-grow/`

**Files:**
- Create: `scripts/gen_region_grow.py`, `tests/test_region_grow_bundle.py`
- Generated: `examples/region-grow/{hood.json,hood.png,README.md}`, `web/src/hood.d.ts`
- Modify: `pyproject.toml` (both mypy lists)

**Interfaces:**
- Consumes: `region._projected`, build-order contract (Task 1); `_bundle_io.polygon_rings`, `cm`, `sigfig` (Task 2).
- Produces: `examples/region-grow/hood.json` matching `web/src/hood.d.ts` below. Consumed by Tasks 5 and 6.

### The bundle

`web/src/hood.d.ts`, generated:

```ts
// GENERATED by scripts/gen_region_grow.py -- do not edit.
// Regenerate: pixi run python -m scripts.gen_region_grow
// This file is what makes a renamed Python field a TypeScript error instead of a blank panel.
export interface HoodBlock {
  block_id: string;
  /** building_count. The growth budget is measured in these. */
  n: number;
  area_m2: number;
  perimeter_m: number;
  /** Exterior ring first, then interiors. 7 of the 213 blocks have one. Fill even-odd. */
  rings: [number, number][][];
  /** Indices into `blocks`, not block_ids -- the greedy runs over these directly. */
  adj: number[];
}
/** Production's own accretion, for one seed at one budget -- the parity fixtures. */
export interface GrowthCase {
  seed: string;
  max_buildings: number;
  /** block_ids in ACCRETION order, straight out of DenseClusterRegionBuilder. */
  order: string[];
  buildings: number;
}
export interface Budget { min: number; max: number; step: number; default: number }
export interface HoodEncoding {
  hood_color: string;
  hood_lw: number;
  region_color: string;
  region_lw: number;
  seed_color: string;
  frontier_color: string;
  pad: number;
}
export interface HoodBundle {
  city: string;
  /** The pinned seed -- the same block PermGraph, Frontier and DisplacementField use. */
  seed: string;
  /** UTM easting/northing subtracted from every coordinate; all geometry is local metres. */
  origin: [number, number];
  crs_epsg: number;
  blocks: HoodBlock[];
  budget: Budget;
  encoding: HoodEncoding;
  reference: GrowthCase[];
}
```

`budget` is `{min: 150, max: 10000, step: 50, default: 3000}`. **150 is deliberate**: at that value
the region is the seed alone, which is the design's §1.3 finding shown rather than hidden. 3,000 is
what every `conf/example/*.yaml` actually sets.

### Two assertions the bake makes, and does not assume

1. **The frame is projected before `build()` is called.** One line; without it a lon/lat frame makes
   `dwithin(0.5)` mean 55 km (spec §1.5). Task 1 makes the builder itself safe, so this is
   belt-and-braces at the boundary — keep it: it documents the requirement at the call site where a
   future reader will look.
2. **The accretion at `budget.max` is contained in the shipped neighbourhood.** Containment does
   **not** follow from the counts: a 54-block accretion could in principle reach 53 hops.

   **It did not hold at the value this plan first specified.** Measured before Task 4 was
   dispatched — the budget-10,000 accretion from the pinned seed is 54 blocks reaching a maximum
   **7 hops**, so the hood must be 7-hop, not 5-hop:

   | hops | blocks in hood | accretion blocks outside it |
   |---|---|---|
   | 4 | 90 | 10 |
   | 5 | 129 | **2** |
   | 6 | 163 | 1 |
   | **7** | **213** | **0** |

   `HOPS = 7` is therefore the shipped value, and it is exactly what containment requires — not a
   round number with margin added, which would be an invented figure of precisely the kind this
   project keeps catching. Keep the assertion anyway: it is what caught this.

- [ ] **Step 1: Write the failing bundle test**

Create `tests/test_region_grow_bundle.py`:

```python
"""The committed RegionGrow bundle: schema, .d.ts parity, and identity against production.

One @pytest.mark.slow test carries every assertion that needs the city parquet. pytest-xdist scopes
`scope="module"` fixtures PER WORKER, not per session, so a module-scoped city load runs once per
worker and D2 lost 18 minutes to exactly that. Same shape as tests/test_frontier_bundle.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.dts_keys import json_keys, ts_field_names

BUNDLE = Path("examples/region-grow/hood.json")
DTS = Path("web/src/hood.d.ts")

pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="bundle not baked")


@pytest.fixture(scope="session")
def bundle() -> dict:
    return json.loads(BUNDLE.read_text(encoding="utf-8"))


def test_dts_declares_exactly_the_keys_the_bundle_carries(bundle: dict) -> None:
    """Bidirectional: a field renamed in Python becomes a TypeScript error, and a field declared
    but never emitted is caught too."""
    declared = ts_field_names(DTS.read_text(encoding="utf-8"))
    carried = json_keys(bundle)
    assert carried - declared == set(), "carried but not declared"
    assert declared - carried == set(), "declared but not carried"


def test_adjacency_is_symmetric_and_excludes_self(bundle: dict) -> None:
    adj = {i: set(b["adj"]) for i, b in enumerate(bundle["blocks"])}
    for i, neighbours in adj.items():
        assert i not in neighbours, f"block {i} is adjacent to itself"
        for j in neighbours:
            assert i in adj[j], f"{i}->{j} is not mirrored"


def test_every_coordinate_is_at_centimetre_precision(bundle: dict) -> None:
    """`cm` rounds to 2 dp. A coordinate carrying more is one that bypassed the quantiser."""
    for b in bundle["blocks"]:
        for ring in b["rings"]:
            for x, y in ring:
                assert round(x, 2) == x and round(y, 2) == y, (b["block_id"], x, y)


def test_the_neighbourhood_carries_its_holed_blocks(bundle: dict) -> None:
    """7 of the 213 blocks have an interior ring, measured. If this drops to 0 the bundle went
    through `polygon_ring` (which would have raised) or a ring list got flattened -- neither of
    which changes any count the other tests check."""
    holed = [b["block_id"] for b in bundle["blocks"] if len(b["rings"]) > 1]
    assert sorted(holed) == [
        "ZAF.9.3.1_1_38616", "ZAF.9.3.1_1_38935", "ZAF.9.3.1_1_40664", "ZAF.9.3.1_1_40963",
        "ZAF.9.3.1_1_41055", "ZAF.9.3.1_1_41838", "ZAF.9.3.1_1_41976",
    ]


def test_the_shipped_budget_floor_is_a_no_op_on_the_seed(bundle: dict) -> None:
    """At `budget.min` the region is the seed ALONE -- the design's §1.3 finding, which the
    widget's caption states. If this ever changes, that caption is wrong."""
    floor = [c for c in bundle["reference"] if c["max_buildings"] == bundle["budget"]["min"]
             and c["seed"] == bundle["seed"]]
    assert len(floor) == 1, "the floor budget must be among the reference cases"
    assert floor[0]["order"] == [bundle["seed"]]


def test_reference_cases_are_prefixes_of_one_another(bundle: dict) -> None:
    """Growth is nested (design §1.4), so for one seed a bigger budget's order must EXTEND a
    smaller one's, not merely contain it. A set-containment assertion would pass against a
    reordering, and order is the whole teaching point."""
    for seed in {c["seed"] for c in bundle["reference"]}:
        cases = sorted((c for c in bundle["reference"] if c["seed"] == seed),
                       key=lambda c: c["max_buildings"])
        for small, big in zip(cases, cases[1:]):
            assert big["order"][:len(small["order"])] == small["order"], (
                seed, small["max_buildings"], big["max_buildings"])


@pytest.mark.slow
def test_bundle_is_what_production_builds_today() -> None:
    """The identity test, and the reason Task 1 exists: every reference case is recomputed by
    calling `DenseClusterRegionBuilder` ITSELF, not a copy of its rule. Also re-derives the
    neighbourhood, so a stale bundle fails here rather than shipping.
    """
    import geopandas as gpd
    from scripts.gen_region_grow import MIN_COUNT, load_blocks, neighbourhood
    from reblock.region import DenseClusterRegionBuilder

    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    blocks = load_blocks(bundle["city"])
    assert blocks.crs is not None and not blocks.crs.is_geographic, (
        "the bake must project before it grows -- dwithin(0.5) is 55 km in lon/lat")

    ids = [b["block_id"] for b in bundle["blocks"]]
    assert neighbourhood(blocks, bundle["seed"], hops=5) == ids, "stale neighbourhood"

    for case in bundle["reference"]:
        got = DenseClusterRegionBuilder(max_buildings=case["max_buildings"]).build(
            blocks, [[case["seed"]]])[0]
        assert got == case["order"], (case["seed"], case["max_buildings"])
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pixi run pytest tests/test_region_grow_bundle.py -v`
Expected: every test SKIPPED (`bundle not baked`). That is the correct starting state — the skip
guard is the dir-reader contract every other bundle test in this repo uses.

- [ ] **Step 3: Write the generator**

Create `scripts/gen_region_grow.py`. Structure, mirroring `scripts/gen_displacement_field.py`:

```python
"""Bake examples/region-grow/ -- the RegionGrow widget's neighbourhood bundle.

The widget runs the PRODUCTION greedy in the browser (web/src/model/accretion.ts), so this bundle
ships the raw quantities that greedy needs -- building_count, area, perimeter and a precomputed
adjacency list -- rather than a baked animation. What it also ships is `reference`: the accretion
`DenseClusterRegionBuilder` itself produces for several seeds and budgets, which is what pins the
TypeScript to this code instead of to a re-implementation of it.

The seed is ZAF.9.3.1_1_40972, the same block PermGraph, Frontier and DisplacementField pin.

Outputs, into examples/region-grow/:

    hood.json    the bundle (schema: web/src/hood.d.ts, generated here)
    hood.png     the fallback figure, drawn from the same encoding the bundle carries
    README.md    generated

Reproduce with `pixi run python -m scripts.gen_region_grow`.
"""
```

Module constants (no magic numbers at use sites):

```python
OUT = Path("examples/region-grow")
CITY = "capetown"
SEED = "ZAF.9.3.1_1_40972"
MIN_COUNT = 30                 # the same filter gen_screen_bakeoff.py applies
HOPS = 7                 # MEASURED: the budget-10,000 accretion reaches 7 hops; 5 leaves 2 blocks out
SIMPLIFY_M = 1.0               # region scale: 5 m would be visible here (design §1.2)
BUDGET = Budget(min=150, max=10_000, step=50, default=3000)
REFERENCE_BUDGETS = (150, 600, 3000, 10_000)
REFERENCE_SEEDS = (SEED, ...)  # SEED plus two neighbours, chosen in Step 4
```

The encoding is a frozen dataclass whose values are used **both** to emit `encoding` into the bundle
**and** as the matplotlib arguments that draw `hood.png`. That is not tidiness: D2 shipped
`street_lw: 1.0` in a bundle while the PNG drew 1.3, a live JS-on/JS-off divergence that no test
caught. One source, two consumers, in one file.

Required functions:

```python
def load_blocks(city: str) -> gpd.GeoDataFrame:
    """Blocks above MIN_COUNT, PROJECTED to the city UTM.

    Projection is not cosmetic. `_block_adjacency` measures `dwithin(STREET_TOL)` with
    STREET_TOL = 0.5, which is 0.5 m here and about 55 km in the parquet's native lon/lat -- where
    every block in the metro reads as adjacent to every other. Task 1 made the builders project
    defensively; this projects at the boundary so the requirement is visible at the call site.
    """


def neighbourhood(blocks: gpd.GeoDataFrame, seed: str, *, hops: int) -> list[str]:
    """block_ids within `hops` block-adjacency steps of `seed`, sorted. Includes the seed."""


def growth(blocks: gpd.GeoDataFrame, seed: str, budget: int) -> GrowthCase:
    """One reference case, by calling DenseClusterRegionBuilder itself."""
```

`main()` then, in order:

1. `blocks = load_blocks(CITY)`; assert `blocks.crs is not None and not blocks.crs.is_geographic`.
2. `ids = neighbourhood(blocks, SEED, hops=HOPS)`.
3. **Containment assertion** — the one the spec insists is asserted, not reasoned about:

```python
    full = growth(blocks, SEED, BUDGET.max)
    missing = [b for b in full.order if b not in set(ids)]
    if missing:
        raise ValueError(
            f"growth at the slider's maximum budget ({BUDGET.max}) leaves the shipped "
            f"{HOPS}-hop neighbourhood: {len(missing)} of {len(full.order)} blocks are not in it "
            f"({missing[:5]}...). The widget would draw a region with holes in it. Raise HOPS "
            f"until this passes -- do NOT lower BUDGET.max, which would hide the widget's most "
            f"interesting range. Block counts do not settle this: a 54-block accretion can in "
            f"principle reach 53 hops, which is why this is asserted and not inferred.")
```

4. Build `blocks` entries with `polygon_rings(geom, ox, oy, what=block_id)` after
   `.simplify(SIMPLIFY_M)`, and `adj` as **indices into the emitted list** (remap from the global
   frame; a neighbour outside the hood is dropped, which is what makes edge growth stop).
5. `reference = [growth(blocks, s, b) for s in REFERENCE_SEEDS for b in REFERENCE_BUDGETS]`.
6. Write `hood.json`, `web/src/hood.d.ts`, `hood.png`, `README.md`.

- [ ] **Step 4: Choose the two extra reference seeds, by measuring**

`REFERENCE_SEEDS` must include seeds whose accretion **order** differs from sorted order — otherwise
Task 5's order test would pass against a `sorted()` implementation and guard nothing (this is
exactly D2's defect #7, a fixture satisfied by its own twin).

Run this and put the result in the constant, with a comment recording what it is for:

```bash
pixi run python -c "
from scripts.gen_region_grow import load_blocks, neighbourhood, growth, SEED, BUDGET
b = load_blocks('capetown')
for s in neighbourhood(b, SEED, hops=2)[:12]:
    g = growth(b, s, 3000)
    print(s, len(g.order), 'ORDER != SORTED' if g.order != sorted(g.order) else 'coincides')
"
```

Pick two that print `ORDER != SORTED` and whose orders are at least 4 blocks long. **If fewer than
two qualify, report it rather than picking coinciding ones.**

- [ ] **Step 5: Bake, then run the tests**

Run: `pixi run python -m scripts.gen_region_grow && pixi run pytest tests/test_region_grow_bundle.py -v`
Expected: PASS, nothing skipped.

- [ ] **Step 6: Add the module to BOTH mypy lists**

In `pyproject.toml`: append `scripts/gen_region_grow.py` to the `typecheck-py` command **and** to
`[tool.mypy] files`. The explicit file arguments override `files` entirely, so a module in only one
list is not checked by the gate.

Run: `pixi run typecheck-py && pixi run pytest tests/test_typecheck_config.py -v`

- [ ] **Step 7: Fault injection**

| guard | injection | must redden |
|---|---|---|
| holed blocks survive | swap `polygon_rings` for `polygon_ring` | the bake RAISES — confirm the message names the block |
| containment | set `HOPS = 2` | the bake raises the containment error |
| identity | change the tie-break in `growth` to `ids[j]` descending | `test_bundle_is_what_production_builds_today` |
| nested prefixes | reverse one reference case's `order` before writing | `test_reference_cases_are_prefixes_of_one_another` |
| the no-op floor | change `BUDGET.min` to 600 | `test_the_shipped_budget_floor_is_a_no_op_on_the_seed` |

- [ ] **Step 8: Commit**

```bash
git add scripts/gen_region_grow.py tests/test_region_grow_bundle.py examples/region-grow web/src/hood.d.ts pyproject.toml
git commit -m "feat: bake examples/region-grow -- the neighbourhood bundle and its parity fixtures"
```

---

## Task 5: `web/src/model/accretion.ts`

**Files:**
- Create: `web/src/model/accretion.ts`, `web/test/accretion.test.ts`

**Interfaces:**
- Consumes: `HoodBundle`, `HoodBlock`, `GrowthCase` from `web/src/hood.d.ts` (Task 4).
- Produces:
  ```ts
  export function depthProxy(n: number, areaM2: number, perimeterM: number): number;
  export function grow(blocks: HoodBlock[], seedIndex: number, maxBuildings: number): number[];
  export interface Growth { order: number[]; buildings: number; stoppedAtEdge: boolean }
  export function growth(blocks: HoodBlock[], seedIndex: number, maxBuildings: number): Growth;
  ```
  Consumed by Task 6.

### The rule, stated once

`DenseClusterRegionBuilder`'s loop, exactly: start from the seed; while the running
`building_count` is below the budget, take the frontier (blocks adjacent to the cluster and not in
it), pick the one maximising `√(nA)/P`, breaking ties by higher `building_count` and then by
`block_id` **ascending**; stop when the budget is reached or the frontier is empty.

`stoppedAtEdge` is `true` when the loop exited on an empty frontier below budget. That is the
production builder's own `if not frontier: break`, and the widget labels it rather than pretending
the region was complete.

- [ ] **Step 1: Write the failing test**

Create `web/test/accretion.test.ts`:

```ts
import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { HoodBundle } from "../src/hood.js";
import { depthProxy, grow, growth } from "../src/model/accretion.js";

const bundle = JSON.parse(
  readFileSync("../examples/region-grow/hood.json", "utf8")) as HoodBundle;
const indexOf = new Map(bundle.blocks.map((b, i) => [b.block_id, i]));

test("depthProxy is sqrt(n*A)/P", () => {
  assert.equal(depthProxy(100, 10000, 400), Math.sqrt(100 * 10000) / 400);
});

test("depthProxy is zero-safe on a degenerate perimeter", () => {
  // `_depth_proxy` in region.py is documented zero-safe; a NaN here would silently win every
  // argmax, since NaN comparisons are false and `min` would keep whatever it saw first.
  assert.equal(depthProxy(10, 0, 0), 0);
});

test("every reference case reproduces production's accretion ORDER", () => {
  // Order, not membership. A set comparison passes against any permutation, and the order IS what
  // the widget draws -- which is why Task 1 changed RegionBuilder to stop discarding it.
  for (const c of bundle.reference) {
    const seed = indexOf.get(c.seed);
    assert.notEqual(seed, undefined, `reference seed ${c.seed} is not in the bundle`);
    const got = grow(bundle.blocks, seed!, c.max_buildings)
      .map((i) => bundle.blocks[i]!.block_id);
    assert.deepEqual(got, c.order, `${c.seed} @ ${c.max_buildings}`);
  }
});

test("at least one reference order differs from its own sorted order", () => {
  // Without this the test above would pass against a `sorted()` implementation and guard nothing.
  // D2's defect #7 was exactly a fixture satisfied by its own twin.
  const informative = bundle.reference.filter(
    (c) => c.order.length > 3 && c.order.join() !== [...c.order].sort().join());
  assert.ok(informative.length >= 2,
    `only ${informative.length} reference cases have order != sorted; the fixture set cannot ` +
    `distinguish accretion order from sorted order`);
});

test("growth reports reaching the edge of the loaded neighbourhood", () => {
  // The production builder's own `if not frontier: break`. A budget far past what 213 blocks can
  // supply must stop and SAY so, not silently return a short region.
  const seed = indexOf.get(bundle.seed)!;
  const huge = growth(bundle.blocks, seed, 10 ** 9);
  assert.equal(huge.stoppedAtEdge, true);
  assert.equal(huge.order.length, bundle.blocks.length,
    "an unbounded budget consumes the seed's whole connected component");
});

test("growth does not report the edge when the budget bound it", () => {
  const seed = indexOf.get(bundle.seed)!;
  assert.equal(growth(bundle.blocks, seed, bundle.budget.default).stoppedAtEdge, false);
});

test("growth is nested in the budget", () => {
  const seed = indexOf.get(bundle.seed)!;
  const small = grow(bundle.blocks, seed, 600);
  const big = grow(bundle.blocks, seed, 3000);
  assert.deepEqual(big.slice(0, small.length), small);
});
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cd web && OUT=$(mktemp -d) && ./node_modules/.bin/tsc -p tsconfig.test.json --outDir "$OUT" --noEmit false; node --test "$OUT/test/accretion.test.js"; rm -rf "$OUT"`
Expected: FAIL — `Cannot find module '../src/model/accretion.js'`.

- [ ] **Step 3: Implement**

Create `web/src/model/accretion.ts`:

```ts
/** The region-growth greedy, mirroring `DenseClusterRegionBuilder` in src/reblock/region.py.
 *
 * Run in the browser rather than baked, so the budget slider is live and any block in the shipped
 * neighbourhood can be a seed. That is affordable because the rule needs no geometry: the depth
 * proxy is three multiplications on numbers the bundle carries, and adjacency is precomputed.
 *
 * `web/test/accretion.test.ts` pins every step of this against `hood.json`'s `reference` cases,
 * which are `DenseClusterRegionBuilder`'s OWN output -- not a re-derivation of its rule. If this
 * file and region.py ever disagree, that test is what says so.
 */
import type { HoodBlock } from "../hood.js";

/** `sqrt(n*A)/P` -- region.py's `_depth_proxy`, including its zero-safety.
 *
 * Zero-safe matters more here than it looks: a NaN would win every argmax silently, because every
 * comparison against NaN is false and the reduction would simply keep whatever it started with. */
export function depthProxy(n: number, areaM2: number, perimeterM: number): number {
  if (perimeterM <= 0) return 0;
  return Math.sqrt(Math.max(0, n) * Math.max(0, areaM2)) / perimeterM;
}

export interface Growth {
  /** Indices into `blocks`, seed first, then each block in the order it was added. */
  order: number[];
  buildings: number;
  /** True when growth ran out of frontier below budget -- region.py's `if not frontier: break`.
   * In the widget this is the edge of the LOADED neighbourhood, and it is labelled, not hidden. */
  stoppedAtEdge: boolean;
}

export function growth(blocks: HoodBlock[], seedIndex: number, maxBuildings: number): Growth {
  const cluster = new Set<number>([seedIndex]);
  const order = [seedIndex];
  let buildings = blocks[seedIndex]!.n;

  while (buildings < maxBuildings) {
    const frontier = new Set<number>();
    for (const i of cluster) for (const j of blocks[i]!.adj) if (!cluster.has(j)) frontier.add(j);
    if (frontier.size === 0) return { order, buildings, stoppedAtEdge: true };

    let best = -1;
    for (const j of frontier) if (best < 0 || beats(blocks, j, best)) best = j;
    cluster.add(best);
    order.push(best);
    buildings += blocks[best]!.n;
  }
  return { order, buildings, stoppedAtEdge: false };
}

/** region.py's `min(frontier, key=lambda j: (-score(j), -counts[j], ids[j]))`, as a comparison:
 * higher depth proxy wins, then higher building_count, then LOWER block_id. */
function beats(blocks: HoodBlock[], a: number, b: number): boolean {
  const x = blocks[a]!;
  const y = blocks[b]!;
  const sx = depthProxy(x.n, x.area_m2, x.perimeter_m);
  const sy = depthProxy(y.n, y.area_m2, y.perimeter_m);
  if (sx !== sy) return sx > sy;
  if (x.n !== y.n) return x.n > y.n;
  return x.block_id < y.block_id;
}

export function grow(blocks: HoodBlock[], seedIndex: number, maxBuildings: number): number[] {
  return growth(blocks, seedIndex, maxBuildings).order;
}
```

- [ ] **Step 4: Run and watch it pass**

Run: `cd web && OUT=$(mktemp -d) && ./node_modules/.bin/tsc -p tsconfig.test.json --outDir "$OUT" --noEmit false; node --test "$OUT/test/accretion.test.js"; rm -rf "$OUT"`
Expected: PASS.

- [ ] **Step 5: Fault injection**

| guard | injection | must redden |
|---|---|---|
| order fidelity | in `beats`, return `x.block_id > y.block_id` (tie-break reversed) | the reference-order test |
| the argmax | change `sx > sy` to `sx < sy` | the reference-order test |
| edge reporting | return `stoppedAtEdge: false` always | the edge test |
| the fixture set is informative | — | run the "order differs from sorted" test alone and confirm it fails if you delete the multi-seed reference cases |

- [ ] **Step 6: Commit**

```bash
cd .. && git add web/src/model/accretion.ts web/test/accretion.test.ts
git commit -m "feat: the region-growth greedy in TypeScript, pinned to production's accretion order"
```

---

## Task 6: The `RegionGrow` widget

**Files:**
- Create: `web/src/render/region.ts`, `web/src/widgets/region-grow.ts`, `web/test/region-grow-boot.test.ts`
- Modify: `web/src/mount.ts`

**Interfaces:**
- Consumes: `growth`/`grow` (Task 5), `HoodBundle` (Task 4), `harness.ts` (Task 3), and from existing modules: `observeSize` (`dom/resize.js`), `removeFallbackImage` (`dom/fallback.js`), `requireAttr` (`dom/attrs.js`), `showWidgetError` (`dom/error.js`), `sizeCanvas` (`render/canvas.js`), `fitBbox`/`toScreen`/`toWorld`/`type View`/`type Bbox` (`view/transform.js`), `localState`/`type StateFactory` (`state.js`).
- Produces: `export function regionGrow(host: HTMLElement, makeState: StateFactory): void`, registered in `mount.ts` as `"region-grow"`.

### Layers, in draw order

1. every neighbourhood block, stroked in `encoding.hood_color` at `hood_lw`;
2. the grown region's blocks, filled in `encoding.region_color`;
3. the current frontier — blocks adjacent to the region and not in it — stroked in
   `encoding.frontier_color`, which is what makes "greedy" visible rather than merely asserted;
4. the seed, stroked in `encoding.seed_color`.

### Controls

* `<input type="range">` over `budget.min…budget.max` step `budget.step`, starting at
  `budget.default`.
* Click (`pointerdown`) on the canvas reseeds to the block under the cursor, via `toWorld` plus a
  point-in-ring test. A click that lands in no block leaves the seed alone.
* An `aria-live="polite"` readout: blocks in the region, total buildings, and — when
  `stoppedAtEdge` — the sentence that growth reached the edge of the loaded neighbourhood.

### Two failure modes this piece has hit before, both closed by construction

* **Renders inside a `ResizeObserver` callback sit outside the `.catch(showWidgetError)` chain**
  (D2's R19). Wrap every draw in the same `runOrReport` helper `displacement-field.ts` uses.
* **The fallback `<img>` must go only after the first successful draw**, not when the canvas is
  inserted — that is what makes `observeSize` skipping a zero width safe.

- [ ] **Step 1: Write the failing boot test**

Create `web/test/region-grow-boot.test.ts`. It mounts the widget against the **committed**
`hood.json`, through the Task 3 harness:

```ts
import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { HoodBundle } from "../src/hood.js";
import { grow } from "../src/model/accretion.js";
import { localState } from "../src/state.js";
import { canvasOf, Call, fireResize, installStubs, lastFrame, mountPoint } from "./harness.js";
import { regionGrow } from "../src/widgets/region-grow.js";

installStubs();

const bundle = JSON.parse(
  readFileSync("../examples/region-grow/hood.json", "utf8")) as HoodBundle;
const E = bundle.encoding;

/** Blocks the picture currently shows as REGION -- identified by the bundle's own fill colour,
 * never by a path count. D2's defect #1 was a layer identified by count, in a figure where two
 * layers happened to have the same number of paths; and defect #2 matched a partial alpha that
 * belonged to a different layer entirely. Name the layer by its colour, then assert on it. */
function regionPaths(cv: unknown): Call[] {
  return lastFrame(cv as never).filter((c) => c.op === "fill" && c.fillStyle === E.region_color);
}

function mount(width = 700): { host: ReturnType<typeof mountPoint>; cv: unknown } {
  const host = mountPoint();
  host.dataset.bundle = "../examples/region-grow/hood.json";
  regionGrow(host as never, localState);
  fireResize(width, width);
  return { host, cv: canvasOf(host) };
}

test("the region drawn at the default budget is the one the model computes", async () => {
  const { cv } = mount();
  await new Promise((r) => setTimeout(r, 0));
  const expected = grow(bundle.blocks,
    bundle.blocks.findIndex((b) => b.block_id === bundle.seed), bundle.budget.default);
  assert.equal(regionPaths(cv).length, expected.length);
});

test("at the slider floor the region is the seed alone", async () => {
  // The design's §1.3 finding, published rather than hidden. If this stops holding, the widget's
  // caption is wrong.
  const { host, cv } = mount();
  await new Promise((r) => setTimeout(r, 0));
  setSlider(host, bundle.budget.min);
  assert.equal(regionPaths(cv).length, 1);
});

test("the fallback image survives until the first successful draw", async () => {
  // `observeSize` SKIPS a zero width, so a widget that removed the <img> on canvas insertion
  // would leave a blank figure in a collapsed container. D2 closed this; keep it closed.
  const { host } = mount(0);
  await new Promise((r) => setTimeout(r, 0));
  assert.ok(host.querySelector("img"), "zero width drew nothing, so the PNG must remain");
});
```

**Implementer note:** `setSlider(host, value)` is a helper you write in this file (find the
`<input type="range">`, set `.value`, dispatch `"input"`). `web/test/field-boot.test.ts` has the
equivalent for its width slider — copy its shape, not its selector.

Add, at the end of the file, the assertion D2 learned to add last:

```ts
test("the picture still matches the model after a reseed and a budget change", async () => {
  // D2's defect #6: drawing was pinned to the model on the BOOT frame only, so every later frame
  // was unguarded. Assert after each interaction, not just at mount.
  const { host, cv } = mount();
  await new Promise((r) => setTimeout(r, 0));
  clickBlock(cv, bundle.blocks[3]!);
  setSlider(host, 600);
  const seed = 3;
  assert.equal(regionPaths(cv).length, grow(bundle.blocks, seed, 600).length);
});
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cd web && OUT=$(mktemp -d) && ./node_modules/.bin/tsc -p tsconfig.test.json --outDir "$OUT" --noEmit false; node --test "$OUT/test/region-grow-boot.test.js"; rm -rf "$OUT"`
Expected: FAIL — `Cannot find module '../src/widgets/region-grow.js'`.

- [ ] **Step 3: Write `web/src/render/region.ts`**

```ts
/** The RegionGrow canvas: neighbourhood, region, frontier, seed -- in that order.
 *
 * Recolouring must not touch geometry, so each block's screen-space path is rebuilt only when the
 * VIEW changes, never when the budget or the seed does. */
import type { HoodBlock, HoodEncoding } from "../hood.js";
import { toScreen, type View } from "../view/transform.js";

export interface RegionFrame {
  view: View;
  region: number[];
  frontier: number[];
  seed: number;
}

export function draw(ctx: CanvasRenderingContext2D, blocks: HoodBlock[], e: HoodEncoding,
                     f: RegionFrame, size: { width: number; height: number }): void { /* ... */ }

/** Index of the block whose exterior ring contains `(wx, wy)`, or -1. Even-odd ray cast over the
 * exterior only: a point inside a HOLE belongs to whatever block fills that hole, and that block
 * is tested on its own ring. */
export function blockAt(blocks: HoodBlock[], wx: number, wy: number): number { /* ... */ }
```

Implement both fully — `draw` strokes/fills each layer in the order listed above, taking colours and
widths from `e` and never from a literal.

- [ ] **Step 4: Write `web/src/widgets/region-grow.ts`**

Follow `web/src/widgets/displacement-field.ts` closely: `requireAttr(host.dataset.bundle,
"data-bundle", "RegionGrow")`, `fetch`, build controls, `observeSize(host, (size) =>
runOrReport(() => render(size)))`, `removeFallbackImage(host)` after the first successful draw.

- [ ] **Step 5: Register it**

In `web/src/mount.ts`, after the existing three registrations and following their comment:

```ts
// Fourth widget, same shape, same reason -- registered HERE, after REGISTRY exists, never from
// inside the widget module (see the paragraph above).
import { regionGrow } from "./widgets/region-grow.js";
register("region-grow", regionGrow);
```

- [ ] **Step 6: Run and watch it pass**

Run: `cd web && npm run check` then `cd web && OUT=$(mktemp -d) && ./node_modules/.bin/tsc -p tsconfig.test.json --outDir "$OUT" --noEmit false; node --test "$OUT/test/region-grow-boot.test.js"; rm -rf "$OUT"`
Expected: PASS.

- [ ] **Step 7: Fault injection**

| guard | injection | must redden |
|---|---|---|
| picture matches model | draw `grow(..., budget + 500)` | the default-budget and post-interaction tests |
| the floor no-op | clamp the slider's min to 600 | the floor test |
| fallback timing | move `removeFallbackImage` to canvas insertion | the fallback test |
| reseed | make `blockAt` always return the pinned seed | the post-interaction test |

- [ ] **Step 8: Commit**

```bash
cd .. && git add web/src/render/region.ts web/src/widgets/region-grow.ts web/src/mount.ts web/test/region-grow-boot.test.ts
git commit -m "feat: the RegionGrow widget -- live greedy, click-to-reseed, budget slider"
```

---

## Task 7: Bake `examples/screen-map/`

**Files:**
- Create: `scripts/gen_screen_map.py`, `tests/test_screen_map_bundle.py`
- Generated: `examples/screen-map/{capetown.json,nairobi.json,screen_map.png,README.md}`, `web/src/screen_map.d.ts`
- Modify: `pyproject.toml` (both mypy lists)

**Interfaces:**
- Consumes: `_bundle_io.polygon_rings`, `cm`, `sigfig` (Task 2); `reblock.data.informal.label_blocks`; `reblock.data.provision.cached_kblock_source`.
- Produces: `examples/screen-map/{capetown,nairobi}.json` matching `web/src/screen_map.d.ts`. Consumed by Tasks 8 and 9.

### The bundle, and why it is columnar

```ts
// GENERATED by scripts/gen_screen_map.py -- do not edit.
// Regenerate: pixi run python -m scripts.gen_screen_map
export interface CityFloor {
  /** The metric this floor belongs to, as `reblock.metric` names it. */
  metric: string;
  value: number;
  n: number;
  /** Read from examples/screen-bakeoff/screen_comparison.csv, which computes them independently
   * of this bundle -- so the widget's own prefix arithmetic has something to be checked against. */
  precision: number | null;
  recall: number | null;
}
export interface CityEncoding {
  base_color: string;
  selected_color: string;
  informal_color: string;
  block_lw: number;
  pad: number;
}
export interface CityBundle {
  city: string;
  crs_epsg: number;
  origin: [number, number];
  n_blocks: number;
  /** Column arrays, not an array of objects: 16,451 blocks with repeated keys would add megabytes
   * of field names. Same shape as field.json's `buildings: {x, y, r}`. */
  block_id: string[];
  n: number[];
  area_m2: number[];
  perimeter_m: number[];
  /** Per block, exterior ring first then interiors. Fill even-odd. */
  rings: [number, number][][][];
  /** 0/1 ground truth. ABSENT for Nairobi -- see the README. Not a null column: a null column is
   * a field that looks answerable and is not. */
  informal?: number[];
  floors: CityFloor[];
  encoding: CityEncoding;
}
```

`n_blocks` is emitted so a truncated column array fails a check rather than silently shortening the
map; the bundle test asserts every column's length against it.

### Nairobi omits `informal`

Cape Town has the City's own informal-structure survey; Nairobi has no equivalent published layer,
searched and documented in `reblock.data.informal`. Nairobi's bundle therefore **omits the field**,
and `floors` carries `precision: null, recall: null` for it.

- [ ] **Step 1: Write the failing bundle test**

Create `tests/test_screen_map_bundle.py`:

```python
"""The committed city tier: schema, column alignment, and precision/recall against the bake-off CSV.

The heavy test is ONE @pytest.mark.slow (see tests/test_region_grow_bundle.py's docstring for why).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from tests.dts_keys import json_keys, ts_field_names

OUT = Path("examples/screen-map")
DTS = Path("web/src/screen_map.d.ts")
CSV_PATH = Path("examples/screen-bakeoff/screen_comparison.csv")

pytestmark = pytest.mark.skipif(not (OUT / "capetown.json").exists(), reason="tier not baked")


@pytest.fixture(scope="session")
def capetown() -> dict:
    return json.loads((OUT / "capetown.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def nairobi() -> dict:
    return json.loads((OUT / "nairobi.json").read_text(encoding="utf-8"))


def test_dts_declares_exactly_the_keys_both_bundles_carry(capetown: dict, nairobi: dict) -> None:
    declared = ts_field_names(DTS.read_text(encoding="utf-8"))
    carried = json_keys(capetown) | json_keys(nairobi)
    assert carried - declared == set(), "carried but not declared"
    # `informal` is declared optional and carried only by Cape Town, so it is in `carried`.
    assert declared - carried == set(), "declared but not carried"


@pytest.mark.parametrize("city", ["capetown", "nairobi"])
def test_every_column_has_n_blocks_entries(city: str, request: pytest.FixtureRequest) -> None:
    """A truncated column would shorten the map without changing its shape -- no error, no blank
    canvas, just fewer blocks than the city has."""
    b = request.getfixturevalue(city)
    for column in ("block_id", "n", "area_m2", "perimeter_m", "rings"):
        assert len(b[column]) == b["n_blocks"], (city, column)


def test_capetown_carries_ground_truth_and_nairobi_does_not(capetown: dict, nairobi: dict) -> None:
    """Nairobi has no published informal layer (reblock.data.informal records the search). The
    field is ABSENT, not null -- a null column is a field that looks answerable and is not."""
    assert len(capetown["informal"]) == capetown["n_blocks"]
    assert set(capetown["informal"]) <= {0, 1}
    assert "informal" not in nairobi


def test_the_interior_rings_survived(capetown: dict, nairobi: dict) -> None:
    """Measured: 6,990 Cape Town and 1,139 Nairobi blocks have a hole. Losing them changes no
    count any other test here checks."""
    assert sum(len(r) - 1 for r in capetown["rings"]) == 6990
    assert sum(len(r) - 1 for r in nairobi["rings"]) == 1139


def test_precision_and_recall_at_the_shipped_floor_match_the_bakeoff(capetown: dict) -> None:
    """Two independently computed paths agreeing. The CSV comes from gen_screen_bakeoff.py's own
    ranking; this recomputes from the bundle's raw n/A/P and ground-truth column. The numbers are
    READ from the CSV, never restated here -- a literal would make this a test of my typing.
    """
    rows = {r["metric"]: r for r in csv.DictReader(CSV_PATH.open(encoding="utf-8"))
            if r.get("floor")}
    assert rows, "the bake-off CSV must carry at least one shipped floor"

    n = capetown["n"]
    a = capetown["area_m2"]
    p = capetown["perimeter_m"]
    informal = capetown["informal"]
    total_informal = sum(informal)

    for floor in capetown["floors"]:
        if floor["precision"] is None:
            continue
        row = next(r for r in rows.values() if r["metric"].startswith(floor["metric"].split("_")[0]))
        scores = [_metric(floor["metric"], n[i], a[i], p[i]) for i in range(capetown["n_blocks"])]
        selected = [i for i, s in enumerate(scores) if s >= floor["value"]]
        hits = sum(informal[i] for i in selected)
        assert len(selected) == int(float(row["floor_n"])), floor["metric"]
        assert math.isclose(hits / len(selected), float(row["floor_prec"]), rel_tol=1e-6)
        assert math.isclose(hits / total_informal, float(row["floor_recall"]), rel_tol=1e-6)
```

**Implementer note:** `_metric(name, n, a, p)` is a helper in this test file that computes the four
formulas. It must be keyed by a **closed** mapping that raises on an unknown name — no `.get(...,
default)`. Write it as an explicit `if/elif` chain ending in `raise ValueError(name)`, or a dict
lookup with no default.

The `row` lookup by prefix above is fragile; **replace it with an explicit, hard-coded mapping from
the bundle's metric names to the CSV's metric strings** and raise on a name not in it. Report the
mapping you used.

- [ ] **Step 2: Run it and watch it skip**

Run: `pixi run pytest tests/test_screen_map_bundle.py -v`
Expected: all SKIPPED (`tier not baked`).

- [ ] **Step 3: Write the generator**

Create `scripts/gen_screen_map.py`, with these constants and no others inline:

```python
OUT = Path("examples/screen-map")
CITIES = {"capetown": 32734, "nairobi": 32737}
MIN_COUNT = 30
SIMPLIFY_M = 5.0     # design §1.1: 5.49 MB / 1.85 MB gz for Cape Town, sub-pixel at city zoom
BAKEOFF_CSV = Path("examples/screen-bakeoff/screen_comparison.csv")
```

Ground truth comes from `reblock.data.informal.label_blocks` — the same 30%-area-cover rule
`gen_screen_bakeoff.py` applies, called rather than reimplemented. Floors and their published
precision/recall are **read from `BAKEOFF_CSV`**, not typed.

- [ ] **Step 4: Bake and run**

Run: `pixi run python -m scripts.gen_screen_map && pixi run pytest tests/test_screen_map_bundle.py -v`
Expected: PASS.

**Record the actual byte sizes in the report** and check them against the spec's §1.1 table (5.49 MB
Cape Town, 1.04 MB Nairobi at 5 m). A material difference means the encoding differs from what was
measured — say so rather than updating the spec silently.

- [ ] **Step 5: Both mypy lists**

Append `scripts/gen_screen_map.py` to `typecheck-py` **and** `[tool.mypy] files`.
Run: `pixi run typecheck-py && pixi run pytest tests/test_typecheck_config.py -v`

- [ ] **Step 6: Fault injection**

| guard | injection | must redden |
|---|---|---|
| column alignment | drop the last entry of `n` before writing | `test_every_column_has_n_blocks_entries` |
| interior rings | use `polygon_ring` | the bake raises |
| ground truth | emit `informal` for Nairobi as a zero column | `test_capetown_carries_ground_truth_and_nairobi_does_not` |
| the floor arithmetic | shift one floor's `value` by 1% | the precision/recall test |

- [ ] **Step 7: Commit**

```bash
git add scripts/gen_screen_map.py tests/test_screen_map_bundle.py examples/screen-map web/src/screen_map.d.ts pyproject.toml
git commit -m "feat: bake examples/screen-map -- the Cape Town and Nairobi city tiers"
```

---

## Task 8: `web/src/model/screen.ts`

**Files:**
- Create: `web/src/model/screen.ts`, `web/test/screen-model.test.ts`

**Interfaces:**
- Consumes: `CityBundle` from `web/src/screen_map.d.ts` (Task 7).
- Produces:
  ```ts
  export type MetricName = "density" | "depth_density_proxy" | "density_compactness" | "depth_proxy";
  export const METRICS: Record<MetricName, (n: number, areaM2: number, perimeterM: number) => number>;
  export function scores(b: CityBundle, metric: MetricName): Float64Array;
  /** Block indices, best-scoring first. */
  export function ranking(b: CityBundle, metric: MetricName): Int32Array;
  export interface Selection { count: number; precision: number | null; recall: number | null }
  export function selectAt(b: CityBundle, order: Int32Array, s: Float64Array, floor: number): Selection;
  ```
  Consumed by Task 9.

`METRICS` is a `Record` keyed by a **union type**, so TypeScript checks exhaustively that all four
exist and rejects a fifth name at the call site. That is a closed set spelled so the checker audits
it — not a string-keyed registry.

- [ ] **Step 1: Write the failing test**

Create `web/test/screen-model.test.ts`:

```ts
import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { CityBundle } from "../src/screen_map.js";
import { METRICS, ranking, scores, selectAt, type MetricName } from "../src/model/screen.js";

const ct = JSON.parse(
  readFileSync("../examples/screen-map/capetown.json", "utf8")) as CityBundle;
const nb = JSON.parse(
  readFileSync("../examples/screen-map/nairobi.json", "utf8")) as CityBundle;

test("each metric is its published formula", () => {
  assert.equal(METRICS.density(100, 10000, 400), 100 / 10000);
  assert.equal(METRICS.depth_proxy(100, 10000, 400), Math.sqrt(100 * 10000) / 400);
  assert.equal(METRICS.density_compactness(100, 10000, 400), 100 / (400 * 400));
  assert.equal(METRICS.depth_density_proxy(100, 10000, 400),
    (Math.sqrt(100 * 10000) / 400) * (100 / 10000));
});

test("the ranking is sorted descending and is a permutation", () => {
  const s = scores(ct, "depth_density_proxy");
  const order = ranking(ct, "depth_density_proxy");
  assert.equal(order.length, ct.n_blocks);
  assert.equal(new Set(order).size, ct.n_blocks, "a permutation, not a resampling");
  for (let i = 1; i < order.length; i++) {
    assert.ok(s[order[i - 1]!]! >= s[order[i]!]!, `out of order at ${i}`);
  }
});

test("selection at each shipped floor reproduces the baked pool size and precision/recall", () => {
  // The bundle's `floors` were READ from the bake-off CSV, which computed them by a different
  // route entirely. Two independent paths agreeing is the strongest guard on this widget.
  for (const f of ct.floors) {
    const metric = f.metric as MetricName;
    const s = scores(ct, metric);
    const got = selectAt(ct, ranking(ct, metric), s, f.value);
    assert.equal(got.count, f.n, `${f.metric} pool size`);
    if (f.precision !== null) {
      assert.ok(Math.abs(got.precision! - f.precision) < 1e-9, `${f.metric} precision`);
      assert.ok(Math.abs(got.recall! - f.recall!) < 1e-9, `${f.metric} recall`);
    }
  }
});

test("a city with no ground truth reports no precision or recall", () => {
  const s = scores(nb, "depth_density_proxy");
  const got = selectAt(nb, ranking(nb, "depth_density_proxy"), s, 0.0128);
  assert.ok(got.count > 0, "the pool is still counted");
  assert.equal(got.precision, null);
  assert.equal(got.recall, null);
});

test("raising the floor never enlarges the selection", () => {
  // Monotonicity. It is the property the prefix representation depends on, and a sort comparator
  // with a sign error would break it while leaving every count plausible.
  const s = scores(ct, "density");
  const order = ranking(ct, "density");
  let prev = Infinity;
  for (const floor of [0, 1e-4, 1e-3, 1e-2, 1e-1]) {
    const n = selectAt(ct, order, s, floor).count;
    assert.ok(n <= prev, `floor ${floor} selected ${n} after ${prev}`);
    prev = n;
  }
});
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cd web && OUT=$(mktemp -d) && ./node_modules/.bin/tsc -p tsconfig.test.json --outDir "$OUT" --noEmit false; node --test "$OUT/test/screen-model.test.js"; rm -rf "$OUT"`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `web/src/model/screen.ts`**

Full implementation, with `selectAt` using binary search over the descending `ranking` to find the
prefix length and a prefix sum over ground truth for precision/recall. `precision`/`recall` return
`null` when `b.informal === undefined`.

- [ ] **Step 4: Run and watch it pass**

Run: `cd web && OUT=$(mktemp -d) && ./node_modules/.bin/tsc -p tsconfig.test.json --outDir "$OUT" --noEmit false; node --test "$OUT/test/screen-model.test.js"; rm -rf "$OUT"`

- [ ] **Step 5: Fault injection**

| guard | injection | must redden |
|---|---|---|
| the formulas | swap `density` and `depth_proxy` bodies | the formula test AND the floor test |
| the sort direction | sort ascending | the ranking and monotonicity tests |
| the prefix sum | count hits over the whole array rather than the prefix | the floor test |
| the null path | return `precision: 0` when ground truth is absent | the no-ground-truth test |

- [ ] **Step 6: Commit**

```bash
cd .. && git add web/src/model/screen.ts web/test/screen-model.test.ts
git commit -m "feat: the screening metrics, ranking and prefix precision/recall in TypeScript"
```

---

## Task 9: The `ScreenMap` widget

**Files:**
- Create: `web/src/render/city.ts`, `web/src/widgets/screen-map.ts`, `web/test/screen-map-boot.test.ts`
- Modify: `web/src/mount.ts`

**Interfaces:**
- Consumes: Task 8's model, Task 7's bundles, Task 3's harness, and the same `dom/` and `view/` modules Task 6 lists.
- Produces: `export function screenMap(host: HTMLElement, makeState: StateFactory): void`, registered as `"screen-map"`.

### Rendering

`web/src/render/city.ts` builds **one `Path2D` per block at load** and never rebuilds it. A frame is:

1. blit the pre-rendered base layer (every block filled `encoding.base_color`) from an offscreen
   canvas;
2. fill `paths[order[0..k]]` in `encoding.selected_color`.

That is what keeps a floor-slider drag cheap: at the shipped Cape Town floor the prefix is 1,655
blocks, not 16,451. Redraws are `requestAnimationFrame`-coalesced so a drag cannot queue frames
faster than they render.

### Controls

A metric `<select>` (four options), a floor `<input type="range">`, a city toggle, and an
`aria-live="polite"` readout carrying pool size and — for Cape Town — precision and recall. For
Nairobi the readout says instead that no ground-truth layer exists for the city, so precision and
recall cannot be shown.

- [ ] **Step 1: Write the failing boot test**

Create `web/test/screen-map-boot.test.ts`, mounting against the committed bundles through the Task 3
harness. It must include:

```ts
test("the number of selected blocks drawn equals what the model selects", async () => {
  // Identified by the bundle's `selected_color`, never by a path count -- see the comment in
  // region-grow-boot.test.ts.
});

test("moving the floor changes the drawn selection", async () => {
  // A widget that computed the selection but drew a constant would pass a count test at one
  // floor. Assert at two floors and require the counts to differ.
});

test("switching metric re-ranks rather than re-filtering the old ranking", async () => {
  // Pick a block that is above the floor under one metric and below it under another, from the
  // committed bundle, and assert its membership flips.
});

test("Nairobi shows a pool size and no precision or recall", async () => {});

test("the base layer is drawn once, not per frame", async () => {
  // The performance claim, made checkable: count base-colour fills across two frames after a
  // floor change and require it not to grow by 16,451.
});
```

Write each body out in full against the committed bundles.

- [ ] **Step 2: Run it and watch it fail**
- [ ] **Step 3: Implement `web/src/render/city.ts`**
- [ ] **Step 4: Implement `web/src/widgets/screen-map.ts`**
- [ ] **Step 5: Register in `mount.ts`**

```ts
// Fifth widget, same shape, same reason -- registered HERE, after REGISTRY exists, never from
// inside the widget module (see the paragraph above).
import { screenMap } from "./widgets/screen-map.js";
register("screen-map", screenMap);
```

- [ ] **Step 6: Run and watch it pass**

Run: `cd web && npm run check` then `cd web && OUT=$(mktemp -d) && ./node_modules/.bin/tsc -p tsconfig.test.json --outDir "$OUT" --noEmit false; node --test "$OUT/test/screen-map-boot.test.js"; rm -rf "$OUT"`

- [ ] **Step 7: Fault injection**

| guard | injection | must redden |
|---|---|---|
| selection drawn | draw `order[0..k+50]` | the selection-count test |
| floor is live | ignore the slider's value | the two-floor test |
| metric switch | keep the first ranking | the re-rank test |
| base layer cached | rebuild it every frame | the base-layer test |
| Nairobi readout | show `0.0%` instead of the absence | the Nairobi test |

- [ ] **Step 8: Commit**

```bash
cd .. && git add web/src/render/city.ts web/src/widgets/screen-map.ts web/src/mount.ts web/test/screen-map-boot.test.ts
git commit -m "feat: the ScreenMap widget -- metric choice, a live floor, precision and recall"
```

---

## Task 10: Site wiring

**Files:**
- Modify: `scripts/gen_site_pages.py`, `docs/_partials/screening.md`, `tests/test_gen_site_pages.py`, `docs/superpowers/backlog.md`, `docs/superpowers/specs/2026-08-13-site-redesign-design.md`

**Interfaces:**
- Consumes: `examples/region-grow/`, `examples/screen-map/` (Tasks 4, 7); the registered widget names `"region-grow"` and `"screen-map"` (Tasks 6, 9).

- [ ] **Step 1: Write the failing marker test**

In `tests/test_gen_site_pages.py`, the existing test asserting `MARKERS` and the markers used in
`docs/_partials/` are the same set **in both directions** will fail once the partial gains markers
with no producer. Add first:

```python
def test_screening_page_mounts_both_widgets() -> None:
    """Each mount point carries data-widget and data-bundle, and the bundle path is relative to
    the GENERATED page's directory (docs/methodology/), not to docs/. D2 shipped `../assets/`
    where `../../assets/` was needed and the widget 404'd behind an intact-looking PNG."""
    page = render_page("screening")
    assert 'data-widget="screen-map"' in page
    assert 'data-widget="region-grow"' in page
    for url in re.findall(r'data-bundle="([^"]+)"', page):
        assert url.startswith("../../assets/"), url
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pixi run pytest tests/test_gen_site_pages.py -v`

- [ ] **Step 3: Add the producers**

In `scripts/gen_site_pages.py`, two producers following `_displacement_field_figure`'s shape exactly
— copy the asset, emit nothing when it is absent, let `_copy_asset` returning `None` **be** the
existence test (no `path.exists()` pre-check, which is what made `_frontier_figure`'s own
`bundle_url is None` branch unreachable).

Register them:

```python
    "SCREENMAP": _screen_map_figure,
    "REGIONGROW": _region_grow_figure,
```

**The caption is load-bearing.** `RegionGrow`'s must state the §1.3 finding — that at the shipped
`max_buildings: 150` the region is the seed alone on this block, because the constant is a *block*
budget under the default data source and a *building* budget here — and every number in it must be
read from `hood.json`, never typed. `ScreenMap`'s quotes pool size, precision and recall from
`capetown.json`'s `floors`, never from prose.

- [ ] **Step 4: Add the markers to the partial**

In `docs/_partials/screening.md`: `<!-- SCREENMAP -->` after *The shipped screen*'s table, and
`<!-- REGIONGROW -->` inside *From block to region*. Extend the partial's leading maintainer comment
to name both new markers, as the file's own comment convention requires.

- [ ] **Step 5: Run the site build**

Run: `pixi run python -m scripts.gen_site_pages && pixi run mkdocs build --strict && pixi run pytest tests/test_gen_site_pages.py -v`

- [ ] **Step 6: Update the record**

* `docs/superpowers/backlog.md` — mark **D3 SHIPPED** with its spec path; strike the four D3 handoff
  bullets, which are now history; note that piece D is complete and E is next.
* `docs/superpowers/specs/2026-08-13-site-redesign-design.md` — strike the **Nairobi's screening
  tier** open question, recording that D3 resolved it: both cities ship, Nairobi without the
  precision/recall readout.
* Add a `## §1.3` pointer in the backlog to the `max_buildings` two-regimes finding, so it is not
  "fixed" later by someone who measures only the kblock regime.

- [ ] **Step 7: Full gate**

Run: `pixi run check`
Expected: lint, typecheck (Python + web), pytest and the web tests all green.

- [ ] **Step 8: Commit**

```bash
git add scripts/gen_site_pages.py docs/_partials/screening.md tests/test_gen_site_pages.py docs/superpowers/backlog.md docs/superpowers/specs/2026-08-13-site-redesign-design.md
git commit -m "feat: mount RegionGrow and ScreenMap on the Screening page"
```

---

## Self-review

**Spec coverage.** §1.1 tolerance → Task 7 `SIMPLIFY_M`. §1.2 neighbourhood → Task 4. §1.3 the
no-op → Task 4 `BUDGET.min`, Task 6's floor test, Task 10's caption. §1.4 nestedness → Task 4's
prefix test, Task 5's nesting test. §1.5 CRS → Task 1, Task 4's `load_blocks`. §2.1–2.2 → Tasks 5,
6. §2.3 out → not implemented, by design. §3.1 metrics client-side → Task 8. §3.2 prefix → Tasks 8,
9. §3.3 interior rings → Task 2, and both bundle tests. §3.4 Nairobi → Task 7's omitted field, Task
8's null path, Task 9's readout. §4 contract → Task 1. §5 bakes → Tasks 4, 7. §6 widgets → Tasks 6,
9, plus Task 3 which the spec does not name (see below). §7 tests → every task's fault-injection
step. §8 corrections → Task 10 Step 6.

**One addition beyond the spec:** Task 3, the test-harness extraction. The spec does not call for
it; it is justified by D3 adding the sixth and seventh copies of a fake that is already duplicated,
and it is deliberately bounded to the two genuine duplicates. **A reviewer may reasonably reject it
as scope; if so, Tasks 6 and 9 each grow their own copy and the finding goes to the backlog.**

**Placeholder scan.** Tasks 6 Step 3, 8 Step 3 and 9 Steps 3–4 describe implementations by interface
and behaviour rather than showing every line. That is deliberate for the two canvas renderers and
the widget shells, which are long and follow a shipped template (`displacement-field.ts`) the
implementer is pointed at — but it is the weakest part of this plan, and an implementer who finds
the guidance insufficient should say so rather than guess.

**Type consistency.** `HoodBlock.n` (not `building_count`) is used in Tasks 4, 5, 6.
`CityBundle.n_blocks` gates every column in Tasks 7, 8, 9. `growth()` returns `Growth` in Task 5 and
is consumed as `Growth` in Task 6. `MetricName` is the union in Task 8 and the cast target in Task
9. `installStubs`/`fireResize`/`mountPoint`/`canvasOf`/`lastFrame`/`Call` are produced by Task 3 and
consumed by Tasks 6 and 9 under those exact names.
