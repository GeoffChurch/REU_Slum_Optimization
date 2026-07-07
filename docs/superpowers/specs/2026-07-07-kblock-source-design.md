# reblock — kblock source: real street-bounded blocks + Voronoi parcels

**Status:** draft for review · **Date:** 2026-07-07 · **Branch:** `kblock-source` (to be cut)

## Why this exists

Every block the pipeline has seen so far scores **k = 1** under the honest peel: topology's
CapeTown, Phule Nagar, and Epworth are all *dissolved parcel unions* whose boundary (the "streets"
proxy) hugs every parcel, so nothing is landlocked — no real access-depth to reblock. The fix is
real data: **kblock** (Mansueto Institute, Million Neighborhoods Africa) delineates genuine
**street-bounded blocks** — faces of the OSM street network — each with an interior of buildings
truly deep from the surrounding streets.

`KblockSource` ingests kblock blocks + a building layer into our `Region`/`Block` contracts, giving
the peel (and topology, and the peel-reblocker) real blocks with real depth. Two datasets:
**Djibouti** (OSM buildings — a quick real second city) and **Cape Town** (Google Open Buildings —
the rich informal-settlement demo, and the continental-goal building source). A spike confirmed the
transform: Djibouti block `DJI.1_2_602` → **peel-k = 4** — real signal, at last.

## Data + the pluggable building source (grounded — schemas verified on real files)

The source consumes two GeoParquet inputs per dataset — **blocks** and **building points**:

- **Blocks** — Harvard Dataverse `DVN/DQY54U`, per-country `{ISO3}_geodata.parquet` (real
  street-bounded `Polygon`s, EPSG:4326; columns used: `block_id`, `geometry`, `k_complexity`,
  `building_count`). Verified: `DJI` = 9 MB / 11,484 blocks (k up to 16); `ZAF` = 873 MB (all South
  Africa) → **filtered to Cape Town** (bbox / City-of-Cape-Town GADM) for the fixture.

- **Building points — source-agnostic.** The source only needs a points layer (`geometry = Point`,
  optional `building_area`); *where the points come from is a fixture-prep concern, not source
  logic*. This is the key design point: adding a new building source later is a new prep script +
  config, never a source change. Two sources, one per dataset:
  - **Djibouti → OSM** (reprex `sample-data.zip` → `buildings_points_DJI.parquet`, 145,251 points;
    already extracted). Simple, consistent, but OSM under-covers informal settlements.
  - **Cape Town → Google Open Buildings V3** (public GCS `points_s2_level_4_gzip`, HTTPS, CC BY-4.0;
    columns `latitude, longitude, area_in_meters, confidence`). ML footprints that comprehensively
    cover African informal settlements (the reblocking-relevant areas) and all of Africa — the free
    analog to kblock's own commercial Ecopia source. Prep filters to the Cape Town bbox and a
    `confidence ≥ 0.7` threshold, then emits centroid points in the same normalized schema as OSM.

**A kblock block *is* a street-bounded face**, so its **boundary genuinely is its streets** — no
gap-ring problem (unlike CapeTown's dissolved union). `Block.streets = boundary` is correct here,
and the peel measures honest depth from the bounding streets inward.

**What the metric actually is (quantified — do not skip):** peel-k measures access depth *on the
building-Voronoi tessellation*, and a spike proved it is **strongly sensitive to building
completeness** — halving a block's buildings drops peel-k by 1–3 (e.g. `DJI.2_1_102`: 340 pts → k=7,
170 pts → k=4-5). So a block's k is **not intrinsic**; it is a function of the building layer.
Consequences baked into this design:
- **Comparisons require a fixed building source.** peel-k from OSM and peel-k from Open Buildings are
  not comparable; only compare blocks/methods *within* one dataset. This is why denser Open Buildings
  is the right Cape Town source — OSM systematically *under*-counts depth (our OSM peel-k=7 vs
  kblock's Ecopia-based k=11 on the same block).
- **kblock's `k_complexity` is NOT a validation target.** It uses different buildings (Ecopia) and a
  different definition (weak-dual), and the spike showed the two metrics genuinely diverge in both
  directions — including a block where `kblock_k = 1` but our peel-k = 3. We carry `kblock_k` into
  `attrs` for *context only*; no test asserts any relationship between it and peel-k (see Validation).

## Architecture — `reblock.data.kblock.KblockSource`

A new `Source` (peer of `ShapefileSource`), yielding a lazy `Region` of `Block`s, **agnostic to the
building source** (it reads whatever points GeoParquet the fixture-prep produced):

```python
class KblockSource:
    def __init__(self, blocks_path: str | Path, buildings_path: str | Path,
                 region_id: str = "kblock", *, min_buildings: int = 10) -> None: ...
    def region(self) -> Region: ...
```

**`region()`**:
1. Read `blocks_path` (keep `block_id`, `k_complexity`, `geometry`) and `buildings_path` (keep
   `geometry`) as GeoDataFrames.
2. `utm = blocks.estimate_utm_crs()`; reproject both to `utm` (metric CRS for Voronoi + peel).
3. Spatial-join building points **into** blocks (`sjoin(..., predicate="within")`) to stamp each
   building with its containing `block_id` — spatial, so it works for any building source.
4. Return `Region(region_id, crs=utm, blocks=self._iter_blocks(...))` — a genuine generator.

**`_iter_blocks`** (per block, deterministic order by `block_id`):
- Skip blocks with fewer than `min_buildings` joined points (Voronoi + a meaningful peel need
  enough sites; ~5,000 of 11,484 DJI blocks receive no buildings at all).
- `poly = make_valid(block.geometry)`; if not a single `Polygon`, **skip with a warning** (the same
  non-fatal backstop `ShapefileSource` uses — one bad record can't crash the drain).
- Build parcels by **Voronoi tessellation** (§ below) → `GeoDataFrame(parcel_id, geometry)`.
- `streets = GeoDataFrame(geometry=[LineString(poly.exterior.coords)])` — the outer street frontage.
- `yield Block(block_id, crs=utm, boundary=poly, parcels=parcels, streets=streets,
  attrs={"kblock_k": <k_complexity>})`.

## Voronoi parcelization (the one new operation)

Buildings don't tile a block, but the peel's adjacency needs a tiling — so parcels are the
**Voronoi cells of the building points, clipped to the block**. This is the standard textbook
Voronoi-clip (a `voronoi_polygons` → `intersection(block)`); it is **implemented independently**, not
ported from kblock (which is GPLv3) — see Licensing. (kblock uses the same standard technique for its
own k-complexity; we do not copy or cite its source as authority.)

- Dedupe coincident points (round to mm) — `voronoi_polygons` requires distinct sites.
- `cells = shapely.voronoi_polygons(MultiPoint(points), extend_to=poly.envelope)`.
- Each parcel = `make_valid(cell).intersection(poly)`; drop empty / zero-area results.
- **`.explode()` MultiPolygon cells** into one `parcel_id` per lobe. A cell clipped against a concave
  block can split into disjoint lobes; left as one `MultiPolygon` parcel it becomes a **graph
  "wormhole"** — one node, one BFS depth, fusing lobes tens of km apart (verified on real data:
  a 5-lobe parcel spanning 43 km), so whichever lobe touches a street stamps the whole parcel
  layer-1. Exploding removes it (uniqueness is all `parcel_id` needs).
- `parcel_id` = sequential over the exploded cells (unique; no longer equals point count — fine).

## Metrics — peel-k (kept, honestly labeled) + geometric access + layer sequences

A red-team verified (independently reproduced) that **peel-k on a Voronoi tessellation ≈
√(building count)** — `corr(peel-k, √count)=0.88`, `corr(peel-k, density)≈0.05`. Because a
space-filling tessellation has **no landlocked parcel**, the peel just counts concentric rings ≈ √n;
it is a size/count proxy, *not* a morphology/access measure (a real access pathology is
indistinguishable from a uniformly dense block). peel-k *is* the same concept as kblock's
k-complexity (topological longest-shortest-path-to-street) — Voronoi's regularization is what washes
out the morphology real cadastral parcels retain. So:

- **Keep peel-k**, relabeled as **topological ring-depth** (not "access depth"). Reblocking it to
  k=1 ("every plot within one ring of a road") is still a coherent objective, and method comparison
  on road-length-to-reach-k1 is still valid — but do not claim it measures access morphology.
- **Add a geometric access metric.** For each building, the shortest-path **distance in metres** to
  the nearest street, via Dijkstra on a proximity graph of building points (Delaunay / the Voronoi
  adjacency) weighted by centroid-to-centroid distance. Report the **max** (worst-served building)
  and emit the **raw per-building distances** (one float per building — same cardinality as the peel
  layers already in `Metrics.fields`, so cheap) for downstream binning. This is morphology-sensitive
  where topological hops on Voronoi are not (a panhandle scores long metres with few rings). A fuller
  "route around footprints" version is backlog; weighted-graph Dijkstra is the first cut.
- **Emit the layer sequence(s)** — `[n_1, …, n_k]`, buildings per access layer, a free byproduct of
  the peel (`layers.value_counts().sort_index()`); and the analogous distance histogram for the
  geometric metric. These are richer block fingerprints than the scalar, and the substrate for
  **1D-Wasserstein block similarity** (OT-inspired retrieval/clustering — backlog). Normalize to
  fractions-per-layer for shape-only similarity; keep raw counts for size+shape.

## Robustness

- **CRS:** inputs are EPSG:4326; reproject to `estimate_utm_crs()` so `STREET_TOL = 0.5 m` and
  Voronoi behave. Fail loud if a layer has no CRS. (`estimate_utm_crs` single-zones its input — fine
  for one metro; add a bbox-width guard before any continental use.)
- **Interior rings — decide, don't inherit.** `poly.exterior` drops interior rings; 187 dense DJI
  blocks have them (courtyards / frame blocks), and seeding streets from the exterior only treats a
  courtyard as solid ground → over-counts depth. **Decision:** if a hole isn't itself another block
  it is open space and *should* seed the peel — use `poly.boundary` (all rings) for `streets`, not
  just `exterior`. (Differs from `ShapefileSource`, whose exterior-only rule was about sliver gap-
  rings in a dissolved union — a different situation.)
- **Boundary-as-streets mislabels coastline / GADM edges** on peripheral blocks (sea/border read as
  free frontage → under-counts). Keep the pinned validation block **interior + fully street-bounded**.
- **Degenerate blocks** (`< min_buildings`, non-`Polygon` dissolve, all-coincident points) are
  skipped (with a warning for the non-`Polygon` case), never crash the drain.
- **Determinism:** blocks in sorted `block_id` order; `parcel_id` sequential over exploded cells.

## Validation

Validate on **our own terms** — not against kblock's k (the spike showed genuine divergence both
directions; a `kblock_k=1` block scored peel-k=3):

- A test asserts the source yields well-formed `Block`s from a fixture (each: `Polygon` boundary,
  non-empty exploded `parcels` with unique `parcel_id`, `streets` present, `attrs["kblock_k"]` for
  context).
- **Pin exact expected values on a specific committed fixture block, not a near-free inequality.**
  `peel-k ≥ 2` is nearly vacuous (√10/2 ≈ 1.6, so any ~6-building block clears it); instead assert
  the *exact* `peel-k`, the *exact* geometric max-distance (metres), and the layer-sequence prefix on
  the pinned block — keyed to the committed fixture's fixed building set, so they're stable and a
  regression actually fails. This tests "the pipeline produced the morphology we saw," not "Voronoi
  made ≥ 2 rings."
- **No assertion relates peel-k to `attrs["kblock_k"]`** (different metrics/building sets); `kblock_k`
  is reported, not asserted against.
- **Confidence-threshold sanity (Open Buildings):** ML footprints can double-detect in dense informal
  imagery, inflating the Voronoi site count. Compute the pinned block's peel-k / geometric metric at
  `confidence ∈ {0.6, 0.7, 0.8}` once and record that they're stable (bound the detector-noise
  confound), since the metric is building-count-sensitive.

## Fixtures, wiring, and the payoff

- **New** `src/reblock/data/kblock.py` — `KblockSource`.
- **Two committed fixtures** (small subsets, a few MB each; committed so tests don't need the big
  downloads), under `tests/data/kblock/`:
  - `blocks_dji_sample.parquet` + `buildings_dji_sample.parquet` — dense DJI blocks + their OSM points.
  - `blocks_capetown_sample.parquet` + `buildings_capetown_sample.parquet` — Cape Town blocks + their
    Open Buildings points.
- **Select by density, not raw count** (red-team F1): `min_buildings` cannot screen a 2,694 km²
  desert face with 340 scattered points — the earlier "impressive" DJI flagship (peel-k=7) *was* that
  rural monster. The fixture predicate is a **building-density + block-area** threshold (e.g. ≥ ~10
  buildings/ha and block-area ≤ a cap), an explicit *number* for each dataset, so two prep runs pick
  the same blocks. The `KblockSource.min_buildings` stays a simple floor; the density screen is a
  fixture-prep / future slum-detection concern. **Force-include the pinned validation block** in the
  fixture independent of the cutoff, so it can't silently drop on re-prep.
- **Fixture prep (Task 1 of the plan).** Proven vs. to-de-risk, stated honestly:
  - **DJI — proven** (scripts run this session): blocks via Dataverse file download; buildings via
    HTTP-range extraction from `sample-data.zip`.
  - **Cape Town blocks — straightforward**: Dataverse `ZAF_geodata.parquet` (873 MB) filtered to the
    metro (bbox / GADM), subset to a few MB.
  - **Cape Town buildings — proven** (full path spiked this session): tile index at
    `https://openbuildings-public-dot-gweb-research.uw.r.appspot.com/public/tiles.geojson`; Cape Town
    is S2 tile **`1dd`**; download its points CSV
    (`.../v3/points_s2_level_4_gzip/1dd_buildings.csv.gz`, 82 MB), filter to the metro bbox
    (lon 18.3–19.0, lat −34.4 to −33.5) + `confidence ≥ 0.7` → **1.96M** centroid points.
  - **End-to-end verified:** real ZAF block `ZAF.9.3.1_1_44882` (ZAF file id `10801347`; 1.46M blocks
    → 83k in the Cape Town bbox; UTM 32734) + 713 Open Buildings points → Voronoi → **peel-k = 7**;
    both methods reblock it to k=1 (peel 7.1 km / topology 2.8 km). Open Buildings found *more*
    buildings than kblock's Ecopia (713 vs 342), and `kblock_k = 30` vs our peel-k = 7 — a 4× gap
    that re-confirms the two metrics must not be cross-asserted (see Validation).
- **Reproducible prep, not session one-offs** (red-team): commit
  `scripts/fetch_kblock_fixtures.py` (parameterized by dataset version + file id) as a slice
  deliverable, even though CI never runs it, plus a `tests/data/kblock/PROVENANCE.md` recording
  source URLs, retrieval date, checksums, and the exact selection predicate. Use **version-qualified**
  Dataverse access (`?persistentId=…&version=2.0`) not bare numeric ids (ids change across dataset
  versions). **Mirror** the DJI buildings (the only source with no DOI — a bare CloudFront URL) into a
  durable location (a GitHub release asset) with a SHA256, so the fixture is regenerable when that
  link rots.
- **New configs** `conf/data/dji.yaml` and `conf/data/capetown.yaml` — `_target_:
  reblock.data.kblock.KblockSource` + the two fixture paths + `region_id`. Selectable as `data=dji`
  / `data=capetown` (mirrors `conf/data/phule.yaml`). `run()` already instantiates `cfg.data` via
  `_target_`; the flat `RunConfig(shapefile=...)` sugar is `ShapefileSource`-specific and untouched.
- **Payoff:** kblock `Block`s flow through the *unchanged* eval/method/render — `peel-reblocker`
  and `topology` can both reblock a real block, scored by the connectivity-aware `KComplexityEval`
  and drawn by `render`. First real-signal end-to-end runs, on two cities and two building sources.

## Testing highlights

- Voronoi parcelization on a hand-built block (a square with a 3×3 grid of building points) →
  9 parcels tiling the block, centre parcel at peel-depth 2.
- Coincident-point dedupe (two buildings at one location → one site, no Voronoi crash).
- `min_buildings` filter (a sparse block skipped; a dense one yielded).
- Non-`Polygon` block dissolve → skipped with a warning (`pytest.warns`).
- Real-fixture integration, **both datasets**: load each committed sample, assert ≥ N blocks, and a
  pinned deep block gives peel-k ≥ 2 with `kblock_k` in `attrs`.
- Pipeline integration: `run()` with `data=capetown` + `method=peel` (or topology) produces a
  `Result` with `delta_k > 0` on a deep block — the first non-trivial real reblocking.

## Scope — first slice

**In:** `KblockSource` (Dataverse blocks + a points building layer → Voronoi parcels +
boundary-streets), the **two committed fixtures** (DJI/OSM and Cape Town/Open Buildings),
`conf/data/{dji,capetown}.yaml`, validation + integration tests. The two datasets exercise the
building-source pluggability (OSM and Open Buildings) with one unchanged source.
**Out (later):** OSM block-carving (we use ready Dataverse blocks); a Geofabrik/OSM building prep
for Cape Town (trivial pluggable addition if wanted); Ecopia; continental scale; using the OSM
street layer to label exact street boundary segments; carrying `building_id`/population onto parcels.

## Licensing & attribution (this slice's obligations)

- **Voronoi-clip is implemented independently** as the standard technique — **not** ported from or
  cited to GPLv3 kblock. No kblock source is copied; no comment/commit says "ported from kblock".
- **Committed data fixtures carry their own licenses** (orthogonal to reblock's code license): kblock
  Dataverse data has a **CC0-field-vs-ODbL-note conflict** (treat ODbL as binding — attribution +
  share-alike — until resolved with the depositor); Open Buildings V3 is **CC-BY-4.0** (we pick CC-BY
  over its ODbL alternative), requiring attribution + a note of modifications (bbox filter,
  `confidence≥0.7`, centroid). Add a `NOTICE` / `THIRD_PARTY_NOTICES.md`.
- **Project-level (tracked in the backlog, not blocking this slice):** `reblock` needs a top-level
  `LICENSE`; the `topology` dependency is unlicensed upstream (`brelsford/topology`) and is a
  candidate to **decouple into an optional extra** so the peel+kblock core can be permissively
  licensed. This slice's core (`KblockSource`, the peel, the new metrics) is **topology-free** and
  stays clean regardless of that decision.

## Migration note (owner's "migrate, don't accommodate")

Purely additive — a new `Source` alongside `ShapefileSource`, two new configs, two committed
fixtures, and new eval metrics (geometric access + layer sequences). No existing contract, source,
method, or config changes; `KComplexityEval` gains fields (additive). `ShapefileSource` and its
Phule/Epworth path stay exactly as they are.
