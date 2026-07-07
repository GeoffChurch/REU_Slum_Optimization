# kblock source + real-access metrics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ingest real kblock street-bounded blocks + a building-point layer into `Region`/`Block` (Voronoi parcels), and add a morphology-sensitive geometric access metric, so the pipeline runs on real data with an honest metric.

**Architecture:** New `KblockSource` (peer of `ShapefileSource`) reads a blocks GeoParquet + a points GeoParquet, spatial-joins points into blocks, and builds each `Block`'s parcels as the Voronoi cells of its points clipped to the block (exploded to single polygons). A new `derive/geometric_access.py` computes shortest-path distance (metres) from each parcel to the nearest street on the parcel-adjacency graph; `KComplexityEval` emits it alongside the (relabeled) topological peel-k and the per-parcel layers.

**Tech Stack:** Python 3.12, geopandas/shapely 2.x, networkx, pandas, Hydra, pytest, mypy --strict, ruff, pixi.

**Reference:** `docs/superpowers/specs/2026-07-07-kblock-source-design.md`.

## Global Constraints

- `pixi run check` (ruff + `mypy --strict src tests` + pytest) green at the end of every task.
- **peel-k is topological ring-depth (≈ √building-count on Voronoi), NOT access depth** — label it so; never cross-assert it against `attrs["kblock_k"]`.
- **Voronoi-clip implemented independently** (standard technique); do NOT copy or cite GPLv3 kblock source.
- **Explode** clipped Voronoi cells to single `Polygon`s (one `parcel_id` per lobe) — no MultiPolygon "wormhole" parcels.
- **`streets` = `poly.boundary`** (all rings, incl. interior — courtyards seed the peel), not `exterior` only.
- **Determinism:** blocks yielded in sorted `block_id`; `parcel_id` sequential over exploded cells; no RNG.
- Additive only: no changes to `contracts.py`, `run.py`, `ShapefileSource`, or existing methods.

---

### Task 1: Committed DJI + Cape Town fixtures + reproducible fetch script

**Files:**
- Create: `scripts/fetch_kblock_fixtures.py`
- Create: `tests/data/kblock/blocks_dji_sample.parquet`, `buildings_dji_sample.parquet`, `blocks_capetown_sample.parquet`, `buildings_capetown_sample.parquet`
- Create: `tests/data/kblock/PROVENANCE.md`, `NOTICE`

**Interfaces:** Produces the committed fixtures the later tasks test against, and the exact pinned validation block ids.

- [ ] **Step 1: Write `scripts/fetch_kblock_fixtures.py`**

A parameterized, re-runnable prep script (CI never runs it). It must: download the Dataverse
per-country geodata **version-qualified** (`?persistentId=doi:10.7910/DVN/DQY54U&version=2.0`) for
`DJI`/`ZAF`; get DJI buildings via HTTP-range extraction of `buildings_points_DJI.parquet` from the
`sample-data.zip` (the range-extraction logic already prototyped this session — reuse it); get Cape
Town buildings from Open Buildings V3 (tile `1dd` points CSV → filter `lon∈[18.3,19.0], lat∈[-34.4,-33.5]`,
`confidence≥0.7`, centroids); then **select the fixture subset by density** and write the four
parquets. Selection predicate (explicit, deterministic): keep blocks with **≥ 10 buildings/ha and
block-area ≤ 0.5 km²**, cap to the densest ~300 blocks per city, and **force-include** the pinned
validation blocks (`DJI` — pick one interior street-bounded block with peel-k ≥ 3 and record its id;
`ZAF.9.3.1_1_44882` for Cape Town). Compute + print SHA256 of each written parquet.

(The raw source data from this session is under the session scratchpad `kblock_dji/`; the script may
read those if present to avoid re-downloading, but must contain the full fetch for reproducibility.)

- [ ] **Step 2: Run it to produce the committed fixtures**

Run: `pixi run python scripts/fetch_kblock_fixtures.py --out tests/data/kblock`
Expected: four parquet files (a few MB total) written; SHA256s printed. Verify each loads:
`pixi run python -c "import geopandas; [print(p, len(geopandas.read_parquet(p))) for p in __import__('glob').glob('tests/data/kblock/*.parquet')]"`

- [ ] **Step 3: Write `PROVENANCE.md` + `NOTICE`**

`PROVENANCE.md`: for each fixture — source (Dataverse DOI + version / Open Buildings tile / sample-data),
retrieval date, the SHA256s, the exact selection predicate, and the pinned validation block ids.
`NOTICE`: attribution for Google Open Buildings V3 (CC-BY-4.0, link, modifications = bbox filter +
`confidence≥0.7` + centroid extraction) and the kblock Dataverse data (note the CC0-vs-ODbL conflict;
attribute + treat as ODbL pending resolution).

- [ ] **Step 4: Commit**

```bash
git add scripts/fetch_kblock_fixtures.py tests/data/kblock/
git commit -m "chore: committed DJI + Cape Town kblock fixtures + fetch script + provenance"
```

---

### Task 2: `KblockSource` — real blocks → Voronoi parcels

**Files:**
- Create: `src/reblock/data/kblock.py`
- Create: `tests/data/test_kblock_source.py`

**Interfaces:**
- Produces: `KblockSource(blocks_path, buildings_path, region_id="kblock", *, min_buildings=10)` with `.region() -> Region`. Yields `Block`s with exploded-Voronoi `parcels`, `streets = boundary`, `attrs={"kblock_k": float}`.
- Consumes: `parcel_adjacency`/`parcel_access_layers` unchanged (for the tests).

- [ ] **Step 1: Write the failing tests** in `tests/data/test_kblock_source.py`

```python
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box, Point, Polygon
from pyproj import CRS
from reblock.contracts import Block
from reblock.data.kblock import KblockSource
from reblock.derive.access import parcel_access_layers

ROOT = Path(__file__).resolve().parents[1]
DJI_BLOCKS = str(ROOT / "data" / "kblock" / "blocks_dji_sample.parquet")
DJI_BLD = str(ROOT / "data" / "kblock" / "buildings_dji_sample.parquet")


def test_yields_wellformed_blocks_from_fixture() -> None:
    blocks = list(KblockSource(DJI_BLOCKS, DJI_BLD, region_id="dji").region().blocks)
    assert len(blocks) >= 5
    b = blocks[0]
    assert isinstance(b, Block) and isinstance(b.boundary, Polygon) and b.crs.is_projected
    assert not b.parcels.empty and b.parcels["parcel_id"].is_unique
    assert all(g.geom_type == "Polygon" for g in b.parcels.geometry)     # exploded, no MultiPolygon
    assert "kblock_k" in b.attrs


def test_voronoi_parcels_tile_a_synthetic_block() -> None:
    # 3x3 grid of building points in a unit-ish block -> 9 tiling parcels, centre at peel-depth 2.
    utm = CRS.from_epsg(32638)
    poly = box(0, 0, 30, 30)
    pts = [Point(5 + 10 * i, 5 + 10 * j) for i in range(3) for j in range(3)]
    blocks = gpd.GeoDataFrame({"block_id": ["b"], "k_complexity": [0.0]}, geometry=[poly], crs=utm)
    bld = gpd.GeoDataFrame(geometry=pts, crs=utm)
    src = KblockSource("unused", "unused", region_id="t", min_buildings=4)
    block = next(src._blocks_from(blocks, bld))          # test helper: (blocks_gdf, bld_gdf) -> Iterator[Block]
    assert len(block.parcels) == 9
    assert parcel_access_layers(block, None).max() == 2


def test_all_parcels_single_polygon_on_concave_block() -> None:
    # Explode invariant: on a concave ("plus"-shaped) block, every yielded parcel is a single
    # Polygon even when a clipped Voronoi cell splits into disjoint lobes across the concavity.
    # (Real MultiPolygon splitting is exercised on real data by test_yields_wellformed_blocks_from_fixture,
    # which asserts all-Polygon on the dense informal fixture blocks — concave, and verified this
    # session to produce MultiPolygon cells pre-explode.)
    from shapely.ops import unary_union
    from shapely.geometry import Point
    utm = CRS.from_epsg(32638)
    poly = cast(Polygon, unary_union([box(10, 0, 20, 30), box(0, 10, 30, 20)]))  # a "+" (single Polygon)
    pts = [Point(15, 5), Point(15, 15), Point(15, 25), Point(5, 15), Point(25, 15)]  # all inside the "+"
    block = next(KblockSource("u", "u", region_id="t", min_buildings=4)._blocks_from(
        gpd.GeoDataFrame({"block_id": ["b"], "k_complexity": [0.0]}, geometry=[poly], crs=utm),
        gpd.GeoDataFrame(geometry=pts, crs=utm)))
    assert all(g.geom_type == "Polygon" for g in block.parcels.geometry)
    assert block.parcels["parcel_id"].is_unique
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/data/test_kblock_source.py -v` → `ModuleNotFoundError: reblock.data.kblock`.

- [ ] **Step 3: Implement `KblockSource`**

```python
"""KblockSource: real kblock street-bounded blocks + a building-point layer -> Blocks.

Parcels are the Voronoi cells of a block's building points, clipped to the block and
exploded to single polygons (standard Voronoi-clip, implemented independently).
streets = the block boundary (a kblock block is a street-bounded face). Agnostic to
the building source (reads whatever points GeoParquet fixture-prep produced).
"""
from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path

import geopandas as gpd
from shapely import make_valid, union_all, voronoi_polygons
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry

from reblock.contracts import Block, Region


def _voronoi_parcels(poly: Polygon, points: list[BaseGeometry],
                     crs: object) -> gpd.GeoDataFrame | None:
    seen: set[tuple[float, float]] = set()
    sites: list[BaseGeometry] = []
    for p in points:
        key = (round(p.x, 3), round(p.y, 3))          # mm dedupe: voronoi needs distinct sites
        if key not in seen:
            seen.add(key); sites.append(p)
    if len(sites) < 4:
        return None
    geoms: list[Polygon] = []
    for cell in voronoi_polygons(MultiPoint(sites), extend_to=poly.envelope).geoms:
        clipped = make_valid(cell).intersection(poly)
        parts = (clipped.geoms if clipped.geom_type in ("MultiPolygon", "GeometryCollection")
                 else [clipped])
        for part in parts:                             # explode lobes -> one parcel_id per polygon
            if part.geom_type == "Polygon" and not part.is_empty and part.area > 0:
                geoms.append(part)
    if not geoms:
        return None
    return gpd.GeoDataFrame({"parcel_id": list(range(len(geoms)))}, geometry=geoms, crs=crs)


class KblockSource:
    def __init__(self, blocks_path: str | Path, buildings_path: str | Path,
                 region_id: str = "kblock", *, min_buildings: int = 10) -> None:
        self.blocks_path = Path(blocks_path)
        self.buildings_path = Path(buildings_path)
        self.region_id = region_id
        self.min_buildings = min_buildings

    def region(self) -> Region:
        blocks = gpd.read_parquet(self.blocks_path, columns=["block_id", "k_complexity", "geometry"])
        bld = gpd.read_parquet(self.buildings_path, columns=["geometry"])
        utm = blocks.estimate_utm_crs()
        return Region(region_id=self.region_id, crs=utm,
                      blocks=self._blocks_from(blocks.to_crs(utm), bld.to_crs(utm)))

    def _blocks_from(self, blocks: gpd.GeoDataFrame, bld: gpd.GeoDataFrame) -> Iterator[Block]:
        utm = blocks.crs
        joined = gpd.sjoin(bld, blocks[["block_id", "geometry"]], predicate="within", how="inner")
        pts_by_block: dict[object, list[BaseGeometry]] = {
            bid: list(g.geometry) for bid, g in joined.groupby("block_id")}
        for _, row in blocks.sort_values("block_id").iterrows():
            pts = pts_by_block.get(row["block_id"], [])
            if len(pts) < self.min_buildings:
                continue
            poly = make_valid(row["geometry"])
            if not isinstance(poly, Polygon):
                warnings.warn(f"{self.region_id}:{row['block_id']}: dissolve is "
                              f"{poly.geom_type}, not Polygon; skipping", stacklevel=2)
                continue
            parcels = _voronoi_parcels(poly, pts, utm)
            if parcels is None:
                continue
            streets = gpd.GeoDataFrame(geometry=[poly.boundary], crs=utm)   # all rings (incl. holes)
            yield Block(block_id=str(row["block_id"]), crs=utm, boundary=poly,
                        parcels=parcels, streets=streets,
                        attrs={"kblock_k": float(row["k_complexity"])})
```

(The `test_..._synthetic_block` test calls `src._blocks_from(blocks_gdf, bld_gdf)` directly to avoid file I/O; note `crs` is `blocks.crs`.)

- [ ] **Step 4: Run tests**

Run: `pixi run check` — all pass; ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/reblock/data/kblock.py tests/data/test_kblock_source.py
git commit -m "feat: KblockSource — Voronoi parcels from real blocks + building points"
```

---

### Task 3: Geometric access metric + emit it (+ relabel peel-k) in `KComplexityEval`

**Files:**
- Create: `src/reblock/derive/geometric_access.py`
- Modify: `src/reblock/eval/kcomplexity.py`
- Create: `tests/derive/test_geometric_access.py`
- Modify: `tests/eval/test_kcomplexity.py`

**Interfaces:**
- Produces: `geometric_access_distances(block, roads=None, *, tol=STREET_TOL) -> pd.Series` (metres to nearest street per parcel, indexed by `parcel_id`). `KComplexityEval` emits `values["geometric_access_max_m"]` and `fields["geometric_access_m"]`.
- Consumes: `parcel_adjacency` (Task-1 refactor from the prior slice), `STREET_TOL`.

- [ ] **Step 1: Write failing tests** in `tests/derive/test_geometric_access.py`

```python
from typing import cast
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Polygon, LineString
from reblock.contracts import Block
from reblock.derive.geometric_access import geometric_access_distances

UTM = CRS.from_epsg(32643)


def _strip(n: int) -> Block:
    # 1xN strip, street on the left edge; parcel centroids at x=0.5,1.5,... so the k-th parcel
    # is ~k metres from the street through the adjacency chain.
    polys = [Polygon([(i, 0), (i+1, 0), (i+1, 1), (i, 1)]) for i in range(n)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(n))}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])], crs=UTM)
    return Block(block_id="s", crs=UTM, boundary=cast(Polygon, parcels.geometry.union_all()),
                 parcels=parcels, streets=streets)


def test_distance_grows_down_the_strip() -> None:
    d = geometric_access_distances(_strip(5), None)
    assert d.loc[0] == 0.0                       # touches the street
    assert d.loc[4] > d.loc[2] > d.loc[0]        # monotone in metres, not just hops
    assert abs(d.loc[4] - 4.0) < 1e-6            # 4 centroid-hops of 1 m each


def test_roads_add_street_sources() -> None:
    block = _strip(5)
    roads = gpd.GeoDataFrame(geometry=[LineString([(4.5, 0), (4.5, 1)])], crs=UTM)  # near parcel 4
    assert geometric_access_distances(block, roads).loc[4] == 0.0
```

- [ ] **Step 2: Run to verify fail** → `ModuleNotFoundError`.

- [ ] **Step 3: Implement `geometric_access_distances`**

```python
"""Geometric access: shortest-path distance (metres) from each parcel to the nearest
street, on the parcel-adjacency graph weighted by centroid distance. Morphology-
sensitive where the topological peel (hops on a Voronoi tiling) is not.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd
from geopandas import GeoDataFrame
from shapely import union_all

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL
from reblock.derive.adjacency import parcel_adjacency


def geometric_access_distances(
    block: Block, roads: GeoDataFrame | None = None, *, tol: float = STREET_TOL
) -> pd.Series:
    ids = list(block.parcels["parcel_id"])
    geoms = list(block.parcels.geometry)
    cents = [g.representative_point() for g in geoms]
    adj = parcel_adjacency(geoms, tol)

    seed_geoms = list(block.streets.geometry)
    if roads is not None and not roads.empty:
        seed_geoms += list(roads.geometry)
    street = union_all(seed_geoms) if seed_geoms else None

    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(range(len(geoms)))
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            if i < j:
                graph.add_edge(i, j, weight=cents[i].distance(cents[j]))
    SRC = -1
    if street is not None:
        for i, g in enumerate(geoms):
            if g.distance(street) <= tol:
                graph.add_edge(SRC, i, weight=0.0)
    lengths = nx.single_source_dijkstra_path_length(graph, SRC) if SRC in graph else {}
    d = [float(lengths.get(i, float("inf"))) for i in range(len(geoms))]
    return pd.Series(d, index=pd.Index(ids, name="parcel_id"), dtype="float64")
```

- [ ] **Step 4: Emit it + relabel peel-k in `KComplexityEval.score`**

In `src/reblock/eval/kcomplexity.py`: import `from reblock.derive.geometric_access import
geometric_access_distances`; in `KComplexityEval.score` compute `geo = geometric_access_distances(block, proposal.roads)`,
add `values["geometric_access_max_m"] = float(geo.max()) if len(geo) else 0.0`, and
`fields["geometric_access_m"] = geo`. Update the class/module docstring: **peel-k is topological
ring-depth (≈ √building-count on a Voronoi tiling), not access depth; `geometric_access_max_m` is the
morphology-sensitive metres-to-street measure.** (The per-parcel layer sequence is
`fields["access_before"].value_counts().sort_index()`, computed downstream — note it in the docstring.)

Add to `tests/eval/test_kcomplexity.py`: on the `_grid5()` fixture, assert `score(...).values` now
contains `geometric_access_max_m` (≥ 0) and `.fields` contains `geometric_access_m` (len == parcels).

- [ ] **Step 5: Run tests**

Run: `pixi run check` — green.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/derive/geometric_access.py src/reblock/eval/kcomplexity.py \
        tests/derive/test_geometric_access.py tests/eval/test_kcomplexity.py
git commit -m "feat: geometric access metric (Dijkstra metres); emit it + relabel peel-k"
```

---

### Task 4: Hydra wiring + real-fixture integration + pinned-value validation

**Files:**
- Create: `conf/data/dji.yaml`, `conf/data/capetown.yaml`
- Modify: `tests/test_run.py`; Modify: `tests/data/test_kblock_source.py`

**Interfaces:** Consumes `KblockSource` (Task 2), the new metrics (Task 3), the fixtures (Task 1), the existing `run()`.

- [ ] **Step 1: Create the config groups**

`conf/data/dji.yaml`:
```yaml
_target_: reblock.data.kblock.KblockSource
blocks_path: tests/data/kblock/blocks_dji_sample.parquet
buildings_path: tests/data/kblock/buildings_dji_sample.parquet
region_id: dji
```
`conf/data/capetown.yaml`: the same with the `_capetown_` fixture paths + `region_id: capetown`.

- [ ] **Step 2: Pinned-value validation test** in `tests/data/test_kblock_source.py`

Load the fixture, find the pinned validation block by id, and assert the **exact** `peel-k`,
`geometric_access_max_m` (within a metre), and the first few entries of the layer sequence
(`parcel_access_layers(block, None).value_counts().sort_index()`) — values the implementer reads off
the committed fixture and pins. (This replaces the vacuous `peel-k ≥ 2`.) Add a confidence-sanity note
only if feasible from the committed data.

```python
def test_pinned_capetown_block_morphology() -> None:
    src = KblockSource(CT_BLOCKS, CT_BLD, region_id="capetown", min_buildings=10)
    block = next(b for b in src.region().blocks if b.block_id == "ZAF.9.3.1_1_44882")
    layers = parcel_access_layers(block, None)
    geo = geometric_access_distances(block, None)
    assert int(layers.max()) == <PINNED_PEEL_K>            # exact, read from the fixture
    assert abs(float(geo.max()) - <PINNED_MAX_METRES>) < 1.0
    assert list(layers.value_counts().sort_index().values[:3]) == <PINNED_SEQ_PREFIX>
```

- [ ] **Step 3: Hydra-wiring + pipeline integration test** in `tests/test_run.py`

Mirror `test_hydra_compose_wires_config_groups` with `data=dji method=peel eval=kcomplexity`; assert
`run(cfg)` yields a `Result` whose kcomplexity metrics include `geometric_access_max_m`, and (on a
deep block) `delta_k > 0` — the first non-trivial real reblocking through the whole pipeline.

- [ ] **Step 4: Run the full suite**

Run: `pixi run check` — all green.

- [ ] **Step 5: Commit**

```bash
git add conf/data/dji.yaml conf/data/capetown.yaml tests/test_run.py tests/data/test_kblock_source.py
git commit -m "feat: wire dji/capetown data groups; pinned-value + pipeline integration tests"
```

---

## Notes for the executor

- **peel-k labeling is a hard constraint**, not cosmetic: the spec's whole point is that it's a
  count proxy, so docstrings/metric names must not call it "access depth."
- **Task 1 needs network + big downloads** (873 MB ZAF, 82 MB OB, 2.6 GB zip range-reads). The raw
  data is in the session scratchpad `kblock_dji/`; reuse it to build the subset if present, but the
  script must contain the full fetch. If the executor lacks network, escalate — the fixtures are the
  prerequisite for Tasks 2–4.
- **Pinned values (Task 4)** are read off the committed fixture by the implementer — they are stable
  because the fixture's building set is fixed. Do not invent them; compute and pin.
- Out of scope (backlog): the geometric "route around footprints" nav-mesh; 1D-Wasserstein block
  similarity; slum-detection region filter; the flow refactor.
