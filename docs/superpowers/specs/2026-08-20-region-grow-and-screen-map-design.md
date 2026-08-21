# RegionGrow and ScreenMap — site piece D3

**Status:** approved 2026-08-20. Supersedes the piece-column of D1's §1 table (see §8).

**Parent design:** `2026-08-13-site-redesign-design.md`
**Prior pieces:** D1 `2026-08-16-frontier-widget-and-substrate-hardening-design.md`,
D2 `2026-08-19-displacement-field-widget-design.md`

## §0 What ships

The last two widgets of piece D, both mounted on **`docs/_partials/screening.md`** — which is why
one spec covers both. The page already argues both points in prose; the widgets make the prose
checkable.

| widget | page section | Explore stage |
|---|---|---|
| `ScreenMap` | *The shipped screen* | ① ② |
| `RegionGrow` | *From block to region* | ③ |

After this piece every widget in the parent design's §5 table exists. What remains is piece E (the
Explore chain) and piece F (draw-your-own-road).

## §1 Measurements

Everything below was measured while scoping, on the cached kblock parquets
(`~/.cache/reblock/blocks_{capetown,nairobi}_full.parquet`), reprojected to UTM (EPSG:32734 /
EPSG:32737) and filtered to `building_count >= 30` — the same `MIN_COUNT` `gen_screen_bakeoff.py`
uses. Reproduce with the probes recorded in §9.

### 1.1 The city tier

| city | raw blocks | after `MIN_COUNT = 30` | vertices (all rings) | interior rings |
|---|---|---|---|---|
| Cape Town | 83,192 | **16,451** | **843,838** (mean 51.3) | **6,990** |
| Nairobi | 16,200 | **3,500** | 155,484 + interiors | **1,139** |

**Neither city has a single MultiPolygon block** (verified for both), so one polygon per block is a
safe format assumption — but **6,990 Cape Town and 1,139 Nairobi blocks have holes**, and that is
not (see §3.3).

Payload against simplification tolerance, **including interior rings**, encoded through
`_bundle_io.cm` (centimetres, origin-relative) and gzipped at level 9:

| city | tolerance | vertices | JSON | gzipped |
|---|---|---|---|---|
| Cape Town | 3 m | 321,893 | 6.43 MB | 2.17 MB |
| Cape Town | **5 m** | **274,084** | **5.49 MB** | **1.85 MB** |
| Nairobi | 3 m | 64,243 | 1.25 MB | 0.42 MB |
| Nairobi | **5 m** | **53,315** | **1.04 MB** | **0.35 MB** |

**5 m is the shipped choice: 6.53 MB in-repo, 2.20 MB on the wire for both cities.**

> **An earlier draft of this spec said 4.35 MB / 1.45 MB for Cape Town at 5 m. That figure was
> measured with exteriors only** — the shape `_bundle_io.polygon_ring` produces — and was 21% low.
> It is recorded here because it is exactly the class of number this project keeps catching: one
> that was measured, and measured on the wrong thing.

Two alternatives were measured and rejected, kept per the frontier rule:

* **Delta-encoded integer decimetres.** Cape Town exteriors at 1 m: 3.20 MB / 1.32 MB gz — better
  fidelity for fewer bytes than 3 m plain on both axes, so genuinely Pareto-optimal on
  fidelity-per-byte. Rejected for *this* widget only: the extra fidelity buys nothing at the zooms
  ScreenMap serves, and it would put a second coordinate convention next to the `cm` floats that
  all three shipped bundles use. Revisit if a widget ever needs deep single-block zoom on the tier.
* **Centroids instead of polygons.** 0.33 MB (Cape Town) / 0.07 MB (Nairobi) — 16× smaller, and at
  city zoom a block is a few pixels anyway. Rejected because the settlement *texture* is what makes
  `city_map.png` legible, and the widget's zoom goes to settlement scale where dots read as noise.

**Geometry is ~95% of the payload.** The metric values are 0.18 MB (Cape Town) and 0.04 MB
(Nairobi), which is what makes §3.1's client-side computation free.

### 1.2 The region neighbourhood

Seed **`ZAF.9.3.1_1_40972`** — the block `PermGraph`, `Frontier` and `DisplacementField` already
pin. `building_count = 165`, adjacency degree 5.

`DenseClusterRegionBuilder` growth, replayed under the production rule:

| `max_buildings` | blocks | buildings |
|---|---|---|
| 150 | **1** | 165 |
| 600 | 3 | 721 |
| 3,000 | **11** | 3,072 |
| 10,000 | 54 | 10,011 |

Its k-hop neighbourhood, geometry at 1 m through `cm`:

| hops | blocks | vertices | geometry | adjacency entries |
|---|---|---|---|---|
| 2 | 23 | 637 | 14.2 KB | 106 |
| 3 | 54 | 1,133 | 25.3 KB | 262 |
| 4 | 90 | 1,509 | 33.8 KB | 434 |
| **5** | **129** | **1,899** | **42.5 KB** | **624** |

The whole bundle lands well under 100 KB. Geometry ships at **1 m**, not the tier's 5 m: this widget
is viewed at region scale where 5 m would be visible, and the difference is a few KB.

### 1.3 `max_buildings: 150` is two regimes, not a miscalibration

Measured across the **1,655** Cape Town blocks above the shipped floor
(`depth_density_proxy >= 0.0128`):

* **365 of them (22.1%) already carry >= 150 buildings**, so `dense_cluster` at its shipped default
  returns the seed unchanged — a literal no-op.
* Median seed `building_count` above the floor is **88**.
* On the **top 30 blocks by `depth_density_proxy`, budget 150 grows to exactly 1 block — all
  thirty.** At 3,000 (what every `conf/example/*.yaml` actually sets) the same seeds grow to 2–15.

This looks like a bad default and is not one. `ShapefileSource.block_geometries()` emits no
`building_count` column, so under the **default** `phule` data source `has_count` is False, every
block counts as 1, and `max_buildings: 150` means *up to 150 blocks* — entirely sensible. The
constant degenerates only on kblock sources, where counts are real and run to the hundreds.

**Nothing to change. The widget publishes the behaviour rather than hiding it** (§2.2), and this
section is the record so it is not "fixed" later by someone who measures only the kblock regime.

### 1.4 Growth is nested

Verified at budgets 150 / 300 / 600 / 1,200 / 2,400 / 4,800 / 9,600 on `ZAF.9.3.1_1_38528`: each
budget's member set contains the previous one, with no exceptions. Greedy accretion picks by a
budget-independent rule, so the budget slider is exactly a **prefix** of one sequence — the same
property `Frontier` relies on. This is what makes §2.1's live recomputation and a baked replay
equivalent in everything except reseeding.

### 1.5 A trap, recorded because it cost an hour

`region._block_adjacency` runs `STRtree.query(..., predicate="dwithin", distance=STREET_TOL)` with
**`STREET_TOL = 0.5` in whatever CRS it is handed**. Production is safe: `KblockSource.
block_geometries()` reprojects to UTM before any builder sees the frame, and
`DenseClusterRegionBuilder` reprojects *only* for the metric, documenting that `geoms` in the
original CRS drive adjacency.

The first scoping probe passed lon/lat. `dwithin(0.5°)` is ~55 km, every Cape Town block became
adjacent to every other, and it hung. **The bake calls `build()` directly and can make the same
mistake.**

There is already a second instance in the tree. `scripts/pair_matrix.py:304` reads its block frame
straight out of the parquet with `pd.read_parquet` — **lon/lat, and a bare `pd.DataFrame` that
reaches `build()` through a `cast(gpd.GeoDataFrame, ...)`, which is a type-checker assertion and not
a runtime conversion.** It is harmless today only because that script's default `region_builder` is
`IdentityRegionBuilder` and every group it passes is a singleton, so neither `_touch_adjacent` nor
`_block_adjacency` is ever reached. Pass it a `DenseClusterRegionBuilder` and it silently grows
regions from blocks 55 km apart, scored on areas measured in square degrees.

**So the guard belongs in `_block_adjacency`, not in the bake:** raise when handed a geographic CRS.
One line, in code this piece is already changing, and it converts a silent wrong answer into a loud
failure at every present and future call site. This is opportunistic hardening of code being touched
for other reasons, which the project's standing directives endorse; it is not licence to widen the
piece further.

## §2 `RegionGrow`

### 2.1 The interaction

The bundle ships, per block: `block_id`, `building_count`, area, perimeter, geometry, and an
adjacency list. **The browser runs the production greedy** — argmax of the depth proxy
`√(nA)/P`, ties broken by higher `building_count`, then by `block_id` ascending — so:

* a **budget slider** (`max_buildings`) redraws the region live;
* **any shipped block is clickable as a new seed**;
* the region outline, the accretion order and the running building count all follow.

This is D2's pattern: ship the raw quantities, compute the model client-side, pin the result to
Python with fixtures. It is available here because the depth proxy is three multiplications and
adjacency is precomputed — no geometry predicates in the browser.

**Growth that reaches the edge of the loaded neighbourhood stops and says so.** That is the
production builder's own `if not frontier: break` branch, not a widget limitation, and labelling it
as such is more honest than silently truncating.

### 2.2 What it teaches, including the no-op

The slider spans **150 to 10,000**, so the reader sees §1.3 directly: at 150 the region is the seed
alone, at 3,000 it is 11 blocks. The caption states why, in the terms §1.3 establishes — that the
constant is a *block* budget under the default data source and a *building* budget here.

The page's existing *From block to region* prose already says the architectural reason (a road
proposed for one block stops at that block's boundary). The widget makes the hinge visible.

### 2.3 What stays out

`convex_hull` and `shape_standardizing` are **not** in the widget. `shape_standardizing` scores the
union's outline at every step, which needs polygon unions in the browser; `convex_hull` is a set
operation with no accretion order to show. Both remain described in the page prose, which is where
a reader learns they are selectable.

> Recorded because it is seductive: `shape_standardizing` *could* run client-side if the bundle
> shipped each adjacent pair's shared boundary length — union perimeter is then exactly
> `Σ perimeters − 2 × Σ shared`, and union area is `Σ areas` since blocks do not overlap.
> `Rectangularity` would additionally need a convex hull in JS. Not worth it for a builder the
> default configuration does not select; the note exists so the idea is not re-derived from scratch.

## §3 `ScreenMap`

### 3.1 Metrics are computed, not shipped

The bundle ships `building_count`, area and perimeter per block. All four candidate screens are
arithmetic on those three:

| screen | formula |
|---|---|
| `density` | `n/A` |
| `depth_density_proxy` (shipped default) | `√(nA)/P · n/A` |
| `density_compactness` (retired) | `n/P²` |
| `depth_proxy` | `√(nA)/P` |

So the metric selector costs nothing, and the values the widget shows are computed the same way
`reblock.metric` computes them rather than copied from a table.

### 3.2 The floor is a prefix

**Sort blocks by the chosen metric once; the floor is then a prefix length `k`.** Three consequences,
and they are the whole performance design:

* **Selection** is `paths[0..k]` — no per-block predicate per frame.
* **Precision and recall** come off a prefix sum over the sorted ground-truth labels, in O(1):
  `precision = informal[0..k] / k`, `recall = informal[0..k] / total_informal`.
* **Drawing** paints a neutral base layer once into an offscreen canvas, then re-fills only the
  selected prefix — **1,655 blocks at the shipped floor, not 16,451**. One `Path2D` per block is
  built at load and never rebuilt, which is the parent design's "recolouring must not touch
  geometry" made concrete.

Sorting is per metric and cached, so switching metric costs one sort of 16,451 elements.

### 3.3 Interior rings

`_bundle_io.polygon_ring` **raises** on a polygon with holes, deliberately: geometry that silently
vanishes from a committed artifact is a wrong picture nobody is looking for.

**Neither bundle routes through that helper.** Both emit a list of rings per block — exterior first,
interiors after — and fill with the even-odd rule. The city tier has 6,990 Cape Town holes; and
**`RegionGrow`'s 129-block neighbourhood contains 3 holed blocks** — `ZAF.9.3.1_1_40664`,
`ZAF.9.3.1_1_40963`, `ZAF.9.3.1_1_41838` — which an earlier draft of this spec missed by writing
"ring" in §5.1's schema. Had it shipped, the bake would have raised on the first of them.

Dropping them would not be invisible. A doughnut block filled solid paints over whatever sits in its
hole, and if the enclosed block fell below `MIN_COUNT` it is not redrawn on top.

### 3.4 Nairobi ships without the readout

Both cities ship, selected by a city toggle. **Nairobi shows the map, the floor and the pool size,
and states on the widget that it has no precision or recall** — because the City of Cape Town's
informal-structure survey has no Kenyan equivalent, which `examples/screen-bakeoff/README.md`
already records as a searched-and-documented absence.

This resolves the parent design's Open Question ("whether the city tier ships for Nairobi at all —
and if so, without the precision/recall readout — is a piece-D decision"). It is a teaching point,
not a caveat: it makes concrete the page's existing argument that an **absolute** floor transfers
across corpora where a percentile does not.

## §4 The `RegionBuilder` contract change

### 4.1 What changes

`RegionBuilder.build` returns each group's members **in build order** — accretion order where one
exists, sorted where it does not (`IdentityRegionBuilder`, `ConvexHullRegionBuilder`: sorted *is*
their build order). Today `DenseClusterRegionBuilder` and `ShapeStandardizingRegionBuilder` both end
with `result.append(sorted(...))`, discarding the order they just computed.

This is what lets the bake pin the browser greedy against **production's own output** rather than
against a re-implementation. D2's defect #3 was a test that pinned an identity against a hand-rolled
copy of the function it was supposed to be testing; without this change, `RegionGrow`'s order test
would be exactly that.

### 4.2 The hazard, and where the fix goes

Member order is **not** inert downstream. In `region._shared_parts`:

```python
parcels = pd.concat([b.parcels for b in blocks], ignore_index=True)
parcels["parcel_id"] = range(len(parcels))
```

Member order renumbers every parcel in the region — while `block_id` (`"region:" + "+".join(sorted(
...))`) and `source_content_hash` (`sorted(...)` of member hashes) both **stay the same**. A cached
derivation keyed on that unchanged hash would be reused against differently numbered parcels. That
is silent corruption, not churn.

**The fix goes where the sensitivity lives: `_shared_parts` sorts its own members by `block_id`.**
Not at the one call site that happens to be safe today —
`pipeline.build_regions` already passes sorted members, so sorting inside `_shared_parts` is
**provably a no-op right now** and protects every future caller, including the bake.

Acceptance: a regenerated multiblock example must come back **byte-identical**. If it does not, the
premise of this section is wrong and the task stops.

### 4.3 Blast radius

* `src/reblock/pipeline.py:172` — `regions = region_builder.build(...)`, then
  `members = sorted({...})`. Unaffected.
* `scripts/pair_matrix.py:317` — flattens straight into `sorted({... for group in groups ...})`.
  **Order-independent; unaffected.**
* `tests/test_region.py`, `tests/test_shape_standardizing_region.py` — assert sorted results in
  several places; these become order assertions, and at least one must pin accretion order
  positively rather than merely accepting it.
* The `RegionBuilder` Protocol docstring says "each sorted for determinism" — rewritten. Determinism
  is preserved: accretion order is deterministic by the tie-break rule.

## §5 The bake

Two new generators, both stdlib-plus-`reblock` (unlike `gen_site_pages.py`, which must stay
stdlib-only and must never import `reblock`).

### 5.1 `scripts/gen_region_grow.py` → `examples/region-grow/`

| artifact | content |
|---|---|
| `hood.json` | 129 blocks: `block_id`, `building_count`, `area_m2`, `perimeter_m`, **rings**, adjacency |
| `hood.d.ts` | generated type, copied to `web/src/hood.d.ts` |
| `hood.png` | fallback figure — the region at the caption's budget |
| `README.md` | generated, per the existing `readme_markdown()` pattern |

Plus **reference fixtures**: the production `DenseClusterRegionBuilder` output, in accretion order,
for several seeds and budgets, which is what `web/test/` drives the TypeScript against.

**Two assertions the bake must make, not assume:**

1. Its block frame is **projected** before `build()` is called (§1.5).
2. The accretion at the slider's maximum budget is **contained in the shipped neighbourhood**. The
   5-hop hood is 129 blocks and growth at 10,000 is 54, but containment does not follow from those
   counts — a 54-block accretion could in principle reach 53 hops. Assert it; do not reason about it.

### 5.2 `scripts/gen_screen_map.py` → `examples/screen-map/`

| artifact | content |
|---|---|
| `capetown.json` | 16,451 blocks: `block_id`, `n`, `area_m2`, `perimeter_m`, **rings**, `informal` |
| `nairobi.json` | 3,500 blocks: same, **no `informal` field** |
| `screen_map.d.ts` | generated, copied to `web/src/` |
| `screen_map.png` | fallback figure — Cape Town at the shipped floor |
| `README.md` | generated |

Ground-truth labels reuse `reblock.data.informal.label_blocks`, which `gen_screen_bakeoff.py`
already calls — the same 30%-area-cover rule, not a second implementation.

Nairobi's bundle **omits** the `informal` field rather than carrying nulls. A missing field is a
type the widget must handle; a null column is a field that looks answerable and is not.

## §6 The widgets

New TypeScript, following the D1/D2 layout:

```
web/src/model/accretion.ts     the greedy, mirroring DenseClusterRegionBuilder
web/src/model/screen.ts        the four metrics, the sorted prefix, precision/recall
web/src/render/region.ts       the neighbourhood + region outline
web/src/render/city.ts         base layer + selected prefix, Path2D cache
web/src/widgets/region-grow.ts
web/src/widgets/screen-map.ts
```

Reused unchanged from D1/D2: `dom/resize.ts` (`observeSize`), `dom/fallback.ts`
(`removeFallbackImage`), `dom/attrs.ts` (`requireAttr`), `dom/error.ts` (`showWidgetError`),
`render/canvas.ts` (`sizeCanvas`), `view/transform.ts`.

Registration goes in `mount.ts` **after** `REGISTRY` exists, never from the widget module — see the
comment there for the module-cycle failure that shape caused.

**No tier-loader abstraction.** The parent design speculates about "a tiered data loader"; D1 and D2
shipped without one and this piece needs two `fetch` calls. YAGNI.

Both widgets get the keyboard- and screen-reader-operable route D1 established: `<input
type="range">` for continuous controls, a real `<select>`/buttons for discrete ones, and an
`aria-live="polite"` readout.

## §7 Tests

### 7.1 The discipline

D2 shipped nine tests that passed while guarding nothing, every one found by fault injection. The
catalogue is `~/wiki/pages/methodology/tests-that-cannot-fail.md`, and **every reviewer on this
piece is briefed with it** and told to assume one more exists.

**The acceptance criterion for a guard is: break the thing it guards, observe red, restore.** An
injection that will not redden gets reported, not tuned until it passes.

### 7.2 The guards that carry weight here

* **Order, not membership.** The browser greedy must reproduce Python's accretion **order** on
  several seeds and budgets. A set-equality assertion would pass against any permutation, and order
  is the entire teaching point.
* **The published floor.** Precision and recall at `depth_density_proxy >= 0.0128` must match
  `examples/screen-bakeoff/screen_comparison.csv`: `floor_n = 1655`, `floor_prec = 0.27492447…`,
  `floor_recall = 0.66715542…`. Two independently computed paths agreeing is the strongest guard on
  this branch — and it must read the CSV, not restate the numbers.
* **Containment.** §5.1's second bake assertion, with a fault-injection test that shrinks the hood
  and confirms the bake raises.
* **The no-op is real.** A test pinning that budget 150 on the pinned seed yields exactly 1 block —
  because if that ever changes, §2.2's caption and §1.3's argument are both wrong.
* **Interiors survive.** A test that a known holed block still has its interior ring in the bundle.
  6,990 rings vanishing would change no schema and no count the other tests check.
* **Byte-identical regeneration** after §4.2, as that section's acceptance.

### 7.3 The slow-test rule

D2 lost 18 minutes to a module-scoped fixture loading a block under `pytest-xdist`, which scopes
`scope="module"` fixtures **per worker**, not per session. Tests that load city data are one
`@pytest.mark.slow` test carrying every assertion that needs the load — the pattern
`tests/test_frontier_bundle.py` already uses.

## §8 Corrections to earlier specs

* **D1's §1 table piece column is wrong and is superseded here.** It assigns `ScreenMap`→D2,
  `DisplacementField`→D3, `RegionGrow`→D4, and says "D2 owns" the simplification tolerance. That was
  a *dependency* table written before sequencing was settled; D2 ruled against it and shipped
  `DisplacementField`. **The table's measurements are good; its piece labels are not.** The
  tolerance is owned here, and §1.1 settles it at 5 m.
* **D1's 843,838 vertices is confirmed** — including 6,990 interior rings, which is why an
  exteriors-only recount gives 697,031.
* **"~2× over budget" was an uncompressed-bytes framing.** The parent design's ~3 MB budget is met
  on the wire at 5 m for both cities combined (2.20 MB gz).
* **The parent design's Nairobi Open Question is resolved** by §3.4 and should be struck there.

## §9 Reproducing the measurements

The scoping probes are not committed (they are throwaway), but each §1 figure is reproducible from
the cached parquets with the recipe stated in its subsection: filter `building_count >= 30`,
reproject to the city UTM, and encode through `scripts/_bundle_io.cm`. The two that need care:

* **Simplification figures must include interior rings** — see the §1.1 warning.
* **Any probe calling `region.build()` or `_block_adjacency` must reproject first** — see §1.5.

## §10 Out of scope

* No change to `max_buildings` (§1.3).
* No `shape_standardizing` in the browser (§2.3).
* No tier-loader abstraction (§6).
* No Explore-chain wiring, URL sync or stage rail — that is piece E.
* No delta-encoded coordinate format (§1.1).
