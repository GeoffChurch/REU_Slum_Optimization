# tests/data/kblock — provenance

Fixtures produced by `scripts/fetch_kblock_fixtures.py` (re-runnable end-to-end; see that script
for the full fetch/selection logic — download endpoints, HTTP-range zip extraction, Open Buildings
tile lookup, and the density-selection predicate are all implemented there, not just described
here). Retrieval / build date: **2026-07-07**.

## Raw sources

### 1. Blocks — Harvard Dataverse `DVN/DQY54U` ("Million Neighborhoods Africa Database")

- Dataset: <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DQY54U>
- Accessed **version-qualified**: `?persistentId=doi:10.7910/DVN/DQY54U&version=2.0` (bare numeric
  Dataverse file ids drift across dataset versions/reuploads; the script always resolves the file
  id for `{ISO3}_geodata.parquet` from this version-qualified metadata query — it never hardcodes
  a file id).
- `DJI_geodata.parquet` — Dataverse file id `10801275`, 8,995,350 bytes,
  sha256 `01ee937845d8b8a765413178fb217975fe0df44bc074a9e135f5cf10badf20d3`
  (Dataverse-reported md5 `ecccfd7e84c1c26cd4417d2bca8b705a` — verified byte-identical against the
  scratchpad copy downloaded earlier this session). 11,484 DJI blocks (EPSG:4326).
- `ZAF_geodata.parquet` — Dataverse file id `10801347`, 872,968,919 bytes,
  sha256 `ff2995f6ab08dd112034b4dfcf4dc7b82b0c961648133d1999eab00a7fb8aaea`
  (Dataverse-reported md5 `c24f7abe09f88099c31884d067bc1c94` — verified byte-identical).
  1,457,745 South Africa blocks (EPSG:4326); **filtered to the Cape Town bbox** (below) before
  fixture selection -> 83,192 blocks.
- **License conflict — flagged, not silently resolved:** the dataset's Dataverse `license` field
  reports **CC0 1.0**, but the dataset's own citation `notesText` (fetched live from the Dataverse
  API this session) states verbatim: *"Million Neighborhoods Africa Database (2023) is made
  available under the Open Database License: <http://opendatacommons.org/licenses/odbl/1.0/>."* We
  treat **ODbL as binding** (attribution + share-alike) pending resolution with the depositor —
  see `NOTICE`.

### 2. DJI building points — kblock reprex `sample-data.zip` (OSM)

- No DOI — a bare CloudFront/S3 URL, from kblock's own README.md "Minimal reproducible example":
  `https://dsbprylw7ncuq.cloudfront.net/_sampledata/sample-data.zip` (2,610,645,183 bytes / 2.6 GB).
  **This script + this file are the durable, regenerable record** of that URL, since the archive
  carries no DOI and could move or be replaced.
- Extracted the single entry
  `sample-data/_minreprex/buildings/osm/points/buildings_points_DJI.parquet` via **HTTP-range
  reads** (`zipfile.ZipFile` over a custom range-request file-like object) — verified this session:
  **6 HTTP requests, ~5.8 MB fetched total**, never the whole 2.6 GB archive; extracted bytes are
  byte-for-byte identical (md5-verified) to the scratchpad copy obtained earlier this session.
- 145,251 OSM building points (EPSG:4326), sha256
  `6b905d127018c6e0443f0697dfa6a94e5812c2dba06198d4d22da02fc66b13e1`.
- License: © OpenStreetMap contributors, **ODbL 1.0**
  (<https://www.openstreetmap.org/copyright>).

### 3. Cape Town building points — Google Open Buildings V3

- Tile index (fetched live): `https://openbuildings-public-dot-gweb-research.uw.r.appspot.com/public/tiles.geojson`
  → tile **`1dd`** is the tile whose polygon contains the Cape Town bbox centroid (looked up
  geometrically, not hardcoded).
- Points CSV: `https://storage.googleapis.com/open-buildings-data/v3/points_s2_level_4_gzip/1dd_buildings.csv.gz`
  (82,614,289 bytes gzip). The tile index only lists each tile's *polygon* CSV URL; the *points*
  (centroid) variant is the same tile id under a parallel `points_s2_level_4_gzip` prefix, which
  the script derives by substitution and which was verified reachable (HTTP 200, matching
  `Content-Length`) this session.
- Filtered to `lon ∈ [18.3, 19.0]`, `lat ∈ [-34.4, -33.5]`, `confidence >= 0.7`, emitted as
  `Point` geometries (columns `geometry`, `area_in_meters`, `confidence`): **1,959,563** points,
  sha256 `14c5c82dc227f83a9106459cb904479c0096649e6288f0473f1c044766a2d708`.
- License: Google Open Buildings V3, **CC-BY-4.0** — see `NOTICE` for the required attribution +
  modifications note.

## Fixture selection predicate (explicit, deterministic — `select_dense_blocks` in the script)

For each city, independently:

1. Spatial-join building points into blocks: `gpd.sjoin(buildings, blocks, predicate="within")` in
   the blocks' `estimate_utm_crs()` (DJI: EPSG:32638; Cape Town: EPSG:32734).
2. `density = joined_point_count / block_area_hectares`, with `block_area` computed from the
   **projected UTM geometry** (not a kblock-provided area column — the predicate is a pure function
   of block geometry + building points, independent of kblock's own bookkeeping).
3. Keep blocks with `density >= 10 buildings/ha` **and** `block_area <= 0.5 km²`.
4. Cap to the **densest 300** of those, by `density` descending.
5. **Force-include** the pinned validation block regardless of the cutoff. Verified this session
   that this is load-bearing, not a no-op: both pinned blocks clear the eligibility bar but miss
   the natural top-300 cutoff —

   | city | pinned block density (bld/ha) | natural top-300 cutoff (bld/ha) |
   |---|---|---|
   | DJI | 37.57 | 149.25 |
   | Cape Town | 107.82 | 151.29 |

6. Cape Town additionally: `ZAF_geodata` is bbox-filtered (`.cx[]`, bbox-intersects) to
   `lon ∈ [18.3, 19.0], lat ∈ [-34.4, -33.5]` **before** the density predicate is applied (per the
   brief) — 1,457,745 → 83,192 blocks.

Only the buildings that spatially joined into a **kept** block are written to the buildings
fixture — not all buildings in the raw city dataset.

This predicate is a pure function of the raw inputs (no RNG) — re-running the script against the
same raw files reproduces the same 301-row block selection and building points for both cities
every time (verified this session: two separate runs against the cached raw files produced
identical SHA256s for all four outputs, prior to the `blocks_*` fixtures below picking up their two
density columns). The `buildings_*` fixtures are still byte-identical on rerun. The `blocks_*`
fixtures now also carry `building_count`/`block_area_m2`, joined in by `scripts/augment_fixtures.py`
(see the fixtures table below): that join appends the two columns after `geometry`, whereas a
from-scratch `fetch_kblock_fixtures.py` run — whose `load_blocks` now reads all five `blocks_*`
columns directly — emits them before `geometry`. The two paths select identical rows/values but are
not byte-identical to each other; confirmed this session for Cape Town (fresh full-script rerun:
same 301 rows, sha256 differs from the committed fixture purely by column order).

## Pinned validation blocks

(Task 4 of the kblock-source plan will pin exact peel-k / geometric-access-metres / layer-sequence
values on these, computed from `KblockSource`, which doesn't exist yet — Task 2.)

- **Cape Town: `ZAF.9.3.1_1_44882`** — block area 96,001.96 m² (9.60 ha). Our own Open Buildings
  join finds **713** points inside it (kblock's own Ecopia-based `building_count` = 342 —
  expected divergence, different building source; see the kblock-source design doc's own spike,
  which found the same 713-vs-342 gap end-to-end). kblock's own `k_complexity` (weak-dual metric,
  different building set + different definition, carried in `attrs` for context only) = 30.
- **DJI: `DJI.1_2_602`** — block area 27,072.99 m² (2.71 ha). Our own OSM join finds **98** points
  inside it (kblock's own `building_count` = 132). kblock's own `k_complexity` = 7.
  **Coastal/GADM-edge check performed** (per the brief's ambiguity-resolution instruction): this
  block's boundary is **~3.7 km (0.033°) from the nearest point on Djibouti's national boundary**
  (the union of all six `gadm_DJI.parquet` admin polygons — whose boundary is the union of
  coastline *and* land borders, so this is a lower bound on distance-to-coast specifically) —
  i.e. it does not hug the coastline on any side. It is classified `area_type = "Non-urban"`,
  `urban_center_name = "Rest of Djibouti"` in the kblock geodata (read: a dense inland village, not
  a coastal edge block). Verdict: **kept as pinned** — no substitution needed.

## Fixtures produced (`tests/data/kblock/`)

| file | rows | bytes | sha256 |
|---|---|---|---|
| `blocks_dji_sample.parquet` | 301 | 48,198 | `02d1cd365afe0f3bd5b32dd1f1a509c4d7bab7f5cd39b9ff2ce04c8d352b35e4` |
| `buildings_dji_sample.parquet` | 13,002 | 235,238 | `d71ec7f66230f4145d0e47dd542ca5254baa46bc99dae5f895da62a7d17f5893` |
| `blocks_capetown_sample.parquet` | 301 | 113,080 | `cdcada5eee99f59e36c2b34e88f4b41f995749ab8626ef266fd910f93a33cd4d` |
| `buildings_capetown_sample.parquet` | 38,930 | 731,131 | `dbc436428d9d29909a14dfbc09287df5b3fa18a8b026a7d13293cb3abb934262` |

Total 1.1 MB. `blocks_*` columns: `block_id`, `k_complexity` (kblock's own weak-dual metric,
carried through for context only — **never** asserted against our own peel-k, per the
kblock-source design doc), `geometry` (`Polygon`, EPSG:4326), `building_count` (kblock's own
Ecopia-based building count per block) and `block_area_m2` (block area in m²) — these last two are
the free per-block density signals consumed by the Screen layer
(`density = building_count / (block_area_m2 / 1e4)`). `buildings_*` columns: `geometry`
(`Point`, EPSG:4326) only, per the brief ("building fixtures need only a geometry column").

301 rows per city = 300 densest blocks + 1 force-included pinned block (each pinned block, per the
table above, misses the natural top-300 cutoff, so the union is 301, not 300).

## Reproducing

```bash
pixi run python scripts/fetch_kblock_fixtures.py --out tests/data/kblock \
    --raw-dir /path/to/raw   # optional: point at already-downloaded raw files to skip the network
```

Without `--raw-dir` (or pointing it at an empty/incomplete directory), the script performs the full
network fetch described above: Dataverse metadata query + file download (blocks), HTTP-range zip
extraction (DJI buildings), and Open Buildings tile index + CSV download + filter (Cape Town
buildings) — each into `--raw-dir` (default: a temp cache dir), then re-applies the same selection
predicate.
