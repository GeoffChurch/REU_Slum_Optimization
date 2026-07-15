# dream_come_true (Phase 1: OSM desire-line baseline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reblocker method, `dream_come_true`, whose proposed roads are the real informal
footpaths people already walk — pulled from OpenStreetMap for the region — so they can be graded by
the existing compare/eval suite against the synthetic methods.

**Architecture:** A pluggable `DesireLineSource` (mirroring clearance's pluggable `Substrate`)
owns "get desire-line geometries for this bbox in this CRS"; `OSMDesireLines` is the Phase-1 source
(Overpass fetch → cache/snapshot → parse → reproject). `DreamComeTrueReblocker` is the source-agnostic
Method: region bbox → source → clip to boundary → dedupe against existing streets → `Proposal(roads)`.

**Tech Stack:** Python, geopandas/shapely, Hydra config, stdlib `urllib`+`json` (no new dependency),
pixi (`pixi run pytest`, `pixi run python -m reblock.compare`), Overpass API.

## Global Constraints

- **No new dependency.** Overpass reached with stdlib `urllib.request`; JSON parsed with stdlib
  `json`; geometry via shapely (already present). Do NOT add `osmnx`, `requests`, `overpy`, etc.
- **No network in CI.** Every test is fixture/stub-based; the live fetch is never exercised by pytest.
- **Overpass needs a real `User-Agent`** — the default urllib/curl UA gets HTTP 406. Constant
  `_USER_AGENT = "reblock-dream-come-true/0.1 (informal-settlement research)"`.
- **Default tags** (owner-approved): `[path, footway, track, steps, pedestrian, living_street]`.
- **Follow existing patterns:** methods are `@dataclass` with an `identity` property + `propose(self,
  block, prior=None) -> Proposal`; pluggable sources use a `Protocol` + a `conf/<group>/*.yaml` config
  group + `${group}` interpolation (like `substrate`). Roads need no `drain` column (peel omits it).
- **Reproducibility:** committed example loads a committed GeoJSON `snapshot` (no network, byte-stable);
  arbitrary regions fetch+cache to `~/.cache/reblock/osm` and the README dates the fetch.
- `STREET_TOL = 0.5` (from `reblock.derive.access`) is the street-dedup / min-segment tolerance.
- Lint/type gates: `pixi run ruff check <files>` and `pixi run mypy --strict <files>` must pass.

---

### Task 1: `DesireLineSource` protocol + Overpass query/parse (pure, network-free core)

**Files:**
- Create: `src/reblock/methods/desire_lines.py`
- Test: `tests/methods/test_desire_lines.py`

**Interfaces:**
- Produces:
  - `class DesireLineSource(Protocol)`: `desire_lines(self, bbox_wgs84: tuple[float, float, float,
    float], crs: CRS) -> gpd.GeoDataFrame` and a `identity` property (`Hashable | None`).
  - `_overpass_query(bbox_wgs84: tuple[float, float, float, float], tags: Sequence[str]) -> str`
    — `bbox_wgs84` is `(min_lon, min_lat, max_lon, max_lat)` (geopandas `total_bounds` order).
  - `_parse_overpass_geom(payload: dict, target_crs: CRS) -> gpd.GeoDataFrame` — GeoDataFrame of
    `LineString` in `target_crs`; ways with < 2 nodes dropped.

- [ ] **Step 1: Write the failing test**

```python
# tests/methods/test_desire_lines.py
from pyproj import CRS

from reblock.methods.desire_lines import _overpass_query, _parse_overpass_geom

UTM = CRS.from_epsg(32734)   # a South-Africa UTM zone (projected), for reprojection assertions


def test_overpass_query_uses_south_west_north_east_and_anchored_tags() -> None:
    # bbox is (min_lon, min_lat, max_lon, max_lat); Overpass wants (south,west,north,east) =
    # (min_lat, min_lon, max_lat, max_lon). Tags are anchored so "path" != "pathway".
    q = _overpass_query((18.735, -33.849, 18.755, -33.834), ["path", "footway"])
    assert "(-33.849,18.735,-33.834,18.755)" in q
    assert '["highway"~"^(path|footway)$"]' in q
    assert "out geom;" in q


def test_parse_overpass_geom_builds_linestrings_drops_short_and_reprojects() -> None:
    payload = {
        "elements": [
            {"type": "way", "id": 1, "geometry": [
                {"lat": -33.84, "lon": 18.74}, {"lat": -33.841, "lon": 18.741}]},
            {"type": "way", "id": 2, "geometry": [{"lat": -33.84, "lon": 18.74}]},  # 1 node: drop
            {"type": "node", "id": 3, "lat": -33.84, "lon": 18.74},                 # not a way: skip
        ]
    }
    gdf = _parse_overpass_geom(payload, UTM)
    assert len(gdf) == 1                                   # the single 2-node way
    assert gdf.crs == UTM                                  # reprojected off 4326
    assert gdf.geometry.iloc[0].geom_type == "LineString"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/methods/test_desire_lines.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reblock.methods.desire_lines'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/reblock/methods/desire_lines.py
"""Desire-line sources for the dream_come_true reblocker: pull the real informal circulation
network (worn footpaths) for a region instead of synthesizing one. `DesireLineSource` is the
pluggable seam (like a routing Substrate); `OSMDesireLines` (Phase 1) reads OpenStreetMap via
Overpass. A later imagery detector becomes another DesireLineSource behind the same interface.
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Protocol

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString


class DesireLineSource(Protocol):
    def desire_lines(
        self, bbox_wgs84: tuple[float, float, float, float], crs: CRS
    ) -> gpd.GeoDataFrame: ...
    @property
    def identity(self) -> Hashable: ...


def _overpass_query(bbox_wgs84: tuple[float, float, float, float], tags: Sequence[str]) -> str:
    """Overpass QL for every `highway` way of the given tag classes in the bbox. `bbox_wgs84` is
    (min_lon, min_lat, max_lon, max_lat) (geopandas total_bounds order); Overpass wants
    (south,west,north,east). Tags are `^(...)$`-anchored so `path` doesn't match `pathway`."""
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    tag_re = "|".join(tags)
    return (
        "[out:json][timeout:60];"
        f'way["highway"~"^({tag_re})$"]({min_lat},{min_lon},{max_lat},{max_lon});'
        "out geom;"
    )


def _parse_overpass_geom(payload: dict, target_crs: CRS) -> gpd.GeoDataFrame:
    """Overpass `out geom` JSON -> a GeoDataFrame of LineStrings in `target_crs`. Each `way` carries
    `geometry: [{lat, lon}, ...]`; ways with < 2 nodes are dropped. Coordinates are (lon, lat) =
    (x, y) in EPSG:4326, then reprojected to `target_crs`."""
    lines: list[LineString] = []
    for el in payload.get("elements", []):
        if el.get("type") != "way":
            continue
        coords = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if len(coords) < 2:
            continue
        lines.append(LineString(coords))
    gdf = gpd.GeoDataFrame(geometry=lines, crs=CRS.from_epsg(4326))
    return gdf.to_crs(target_crs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/methods/test_desire_lines.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Lint + type-check**

Run: `pixi run ruff check src/reblock/methods/desire_lines.py tests/methods/test_desire_lines.py && pixi run mypy --strict src/reblock/methods/desire_lines.py`
Expected: `All checks passed!` and `Success: no issues found`

- [ ] **Step 6: Commit**

```bash
git add src/reblock/methods/desire_lines.py tests/methods/test_desire_lines.py
git commit -m "feat: DesireLineSource protocol + Overpass query/parse for dream_come_true"
```

---

### Task 2: `OSMDesireLines` — fetch + cache + snapshot + identity

**Files:**
- Modify: `src/reblock/methods/desire_lines.py`
- Test: `tests/methods/test_desire_lines.py`

**Interfaces:**
- Consumes: `_overpass_query`, `_parse_overpass_geom`, `DesireLineSource` (Task 1).
- Produces: `@dataclass class OSMDesireLines` with fields `tags: Sequence[str]` (default the six
  approved), `endpoint: str`, `cache_dir: str | None`, `snapshot: str | None`. Methods:
  `desire_lines(bbox_wgs84, crs) -> GeoDataFrame`, `identity` property (`None` when live/uncacheable,
  a stable tuple when a snapshot is set). Fetch precedence: **snapshot → disk cache → Overpass**.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/methods/test_desire_lines.py
import json
from pathlib import Path

import pytest
from shapely.geometry import LineString

from reblock.methods.desire_lines import OSMDesireLines

_BBOX = (18.735, -33.849, 18.755, -33.834)


def _write_geojson(path: Path, lines_lonlat: list[list[tuple[float, float]]]) -> None:
    gdf = gpd.GeoDataFrame(geometry=[LineString(c) for c in lines_lonlat],
                           crs=CRS.from_epsg(4326))
    gdf.to_file(path, driver="GeoJSON")


def test_osm_snapshot_is_loaded_without_fetching(tmp_path: Path) -> None:
    snap = tmp_path / "snap.geojson"
    _write_geojson(snap, [[(18.74, -33.84), (18.741, -33.841)]])
    src = OSMDesireLines(snapshot=str(snap))
    src._fetch = lambda query: pytest.fail("must not fetch when a snapshot is present")  # type: ignore[method-assign]
    gdf = src.desire_lines(_BBOX, UTM)
    assert len(gdf) == 1 and gdf.crs == UTM


def test_osm_cache_hit_is_loaded_without_fetching(tmp_path: Path) -> None:
    src = OSMDesireLines(cache_dir=str(tmp_path))
    # Pre-seed the cache at the exact key path the source will look for.
    cache_path = src._cache_path(_BBOX)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _write_geojson(cache_path, [[(18.74, -33.84), (18.741, -33.841)]])
    src._fetch = lambda query: pytest.fail("must not fetch on a cache hit")  # type: ignore[method-assign]
    gdf = src.desire_lines(_BBOX, UTM)
    assert len(gdf) == 1


def test_osm_fetch_writes_cache_then_reuses_it(tmp_path: Path) -> None:
    calls = {"n": 0}
    payload = {"elements": [{"type": "way", "id": 1, "geometry": [
        {"lat": -33.84, "lon": 18.74}, {"lat": -33.841, "lon": 18.741}]}]}
    src = OSMDesireLines(cache_dir=str(tmp_path))
    src._fetch = lambda query: (calls.__setitem__("n", calls["n"] + 1), payload)[1]  # type: ignore[method-assign]
    a = src.desire_lines(_BBOX, UTM)
    b = src.desire_lines(_BBOX, UTM)          # second call: cache hit, no second fetch
    assert len(a) == 1 and len(b) == 1 and calls["n"] == 1


def test_osm_identity_none_when_live_stable_with_snapshot(tmp_path: Path) -> None:
    assert OSMDesireLines().identity is None                       # live -> uncacheable
    snap = tmp_path / "snap.geojson"
    _write_geojson(snap, [[(18.74, -33.84), (18.741, -33.841)]])
    ident = OSMDesireLines(snapshot=str(snap)).identity
    assert ident is not None and ident[0] == "osm"                 # snapshot -> stable identity
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/methods/test_desire_lines.py -v`
Expected: FAIL — `ImportError: cannot import name 'OSMDesireLines'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/reblock/methods/desire_lines.py` (and add imports at top):

```python
# add to the top-of-file imports:
import hashlib
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# module-level constant:
_USER_AGENT = "reblock-dream-come-true/0.1 (informal-settlement research)"
_DEFAULT_TAGS = ("path", "footway", "track", "steps", "pedestrian", "living_street")


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "reblock" / "osm"


@dataclass
class OSMDesireLines:
    """A DesireLineSource backed by OpenStreetMap. Fetch precedence: a committed `snapshot`
    GeoJSON (byte-stable, no network) -> a disk cache under `cache_dir` (default
    ~/.cache/reblock/osm; offline after first fetch) -> a live Overpass query. `identity` is None
    when live (uncacheable, so the derivation cache bypasses and never serves stale OSM), and a
    stable tuple keyed on the snapshot's content hash when a snapshot is pinned."""

    tags: Sequence[str] = _DEFAULT_TAGS
    endpoint: str = "https://overpass-api.de/api/interpreter"
    cache_dir: str | None = None
    snapshot: str | None = None

    @property
    def identity(self) -> Hashable:
        if self.snapshot is None:
            return None                                   # live: uncacheable (data can drift)
        digest = hashlib.sha256(Path(self.snapshot).read_bytes()).hexdigest()[:16]
        return ("osm", tuple(self.tags), digest)

    def _cache_path(self, bbox_wgs84: tuple[float, float, float, float]) -> Path:
        root = Path(self.cache_dir) if self.cache_dir else _default_cache_dir()
        key = f"{'|'.join(self.tags)}@{','.join(f'{c:.5f}' for c in bbox_wgs84)}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return root / f"{digest}.geojson"

    def _fetch(self, query: str) -> dict:
        """POST the Overpass query and return the parsed JSON. A real User-Agent is required
        (default UA -> HTTP 406)."""
        data = urllib.parse.urlencode({"data": query}).encode()
        req = urllib.request.Request(
            self.endpoint, data=data, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:   # noqa: S310 (trusted endpoint)
            return json.loads(resp.read().decode())

    def desire_lines(
        self, bbox_wgs84: tuple[float, float, float, float], crs: CRS
    ) -> gpd.GeoDataFrame:
        if self.snapshot is not None:
            return gpd.read_file(self.snapshot).to_crs(crs)
        cache_path = self._cache_path(bbox_wgs84)
        if cache_path.exists():
            return gpd.read_file(cache_path).to_crs(crs)
        payload = self._fetch(_overpass_query(bbox_wgs84, self.tags))
        gdf_4326 = _parse_overpass_geom(payload, CRS.from_epsg(4326))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        gdf_4326.to_file(cache_path, driver="GeoJSON")
        return gdf_4326.to_crs(crs)
```

Note: `_parse_overpass_geom(payload, CRS.from_epsg(4326))` returns 4326 for caching; the final
`.to_crs(crs)` reprojects for the caller. An empty result writes an empty GeoJSON and returns empty.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/methods/test_desire_lines.py -v`
Expected: PASS (6 tests total)

- [ ] **Step 5: Lint + type-check**

Run: `pixi run ruff check src/reblock/methods/desire_lines.py tests/methods/test_desire_lines.py && pixi run mypy --strict src/reblock/methods/desire_lines.py`
Expected: pass. (If ruff flags `S310`/urllib, the inline `# noqa: S310` covers the trusted-endpoint
POST; if mypy flags the `field` import as unused, drop it.)

- [ ] **Step 6: Commit**

```bash
git add src/reblock/methods/desire_lines.py tests/methods/test_desire_lines.py
git commit -m "feat: OSMDesireLines fetch/cache/snapshot source for dream_come_true"
```

---

### Task 3: `DreamComeTrueReblocker` — the Method (bbox → clip → dedupe → Proposal)

**Files:**
- Create: `src/reblock/methods/dream_come_true.py`
- Test: `tests/methods/test_dream_come_true.py`

**Interfaces:**
- Consumes: `DesireLineSource` (Task 1); `Block`, `Proposal` (`reblock.contracts`); `STREET_TOL`
  (`reblock.derive.access`).
- Produces: `@dataclass class DreamComeTrueReblocker` with fields `source: DesireLineSource`,
  `corridor_m: float = 3.0`. `propose(self, block, prior=None) -> Proposal`; `identity` property
  (propagates `None` when `source.identity is None`, like clearance+substrate).
  `_interior_desire_lines(lines: GeoDataFrame, block: Block) -> GeoDataFrame`.

- [ ] **Step 1: Write the failing test**

```python
# tests/methods/test_dream_come_true.py
from collections.abc import Hashable

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.contracts import Block, Proposal
from reblock.methods.dream_come_true import DreamComeTrueReblocker

UTM = CRS.from_epsg(32734)
Bbox = tuple[float, float, float, float]


def _block() -> Block:
    # A 100 m square block; its street is the south edge (y=0).
    boundary = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    parcels = gpd.GeoDataFrame({"parcel_id": [0]}, geometry=[boundary], crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (100, 0)])], crs=UTM)
    return Block(block_id="b", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


class _StubSource:
    """Structurally a DesireLineSource: returns fixed lines in the block CRS, ignoring the bbox --
    exercises the method's clip/dedupe without any network."""

    def __init__(self, lines: list[LineString], ident: Hashable = ("stub",)) -> None:
        self._lines = lines
        self._ident = ident

    @property
    def identity(self) -> Hashable:
        return self._ident

    def desire_lines(self, bbox_wgs84: Bbox, crs: CRS) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(geometry=self._lines, crs=UTM)


def test_propose_keeps_interior_paths_and_drops_those_on_the_street() -> None:
    interior = LineString([(50, 20), (50, 80)])          # a vertical interior path
    on_street = LineString([(10, 0), (90, 0)])           # runs along the south-edge street
    outside = LineString([(150, 150), (160, 160)])       # outside the boundary
    method = DreamComeTrueReblocker(source=_StubSource([interior, on_street, outside]))
    prop = method.propose(_block())
    assert isinstance(prop, Proposal) and prop.roads is not None
    lengths = sorted(round(g.length) for g in prop.roads.geometry)
    assert lengths == [60]                               # only the interior path survives


def test_propose_empty_coverage_returns_empty_roads_without_crashing() -> None:
    method = DreamComeTrueReblocker(source=_StubSource([]))
    prop = method.propose(_block())
    assert prop.roads is not None and prop.roads.empty


def test_identity_propagates_none_from_uncacheable_source() -> None:
    # A live (snapshot-less) source reports identity None; the method must propagate it.
    method = DreamComeTrueReblocker(source=_StubSource([], ident=None))
    assert method.identity is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/methods/test_dream_come_true.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reblock.methods.dream_come_true'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/reblock/methods/dream_come_true.py
"""DreamComeTrueReblocker: the reblocker whose 'proposed roads' are the REAL informal circulation
network for the region -- the worn footpaths people already walk -- rather than a synthesized one.
The desire-lines come from a pluggable DesireLineSource (OSM in Phase 1; a satellite-imagery
detector later). The method is source-agnostic: fetch desire-lines for the region bbox, clip them
to the block, drop the parts that merely retrace existing streets, and return the interior remainder
as the intervention. See docs/superpowers/specs/2026-07-15-dream-come-true-design.md."""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import geopandas as gpd
from shapely.ops import unary_union

from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL
from reblock.methods.desire_lines import DesireLineSource


def _interior_desire_lines(lines: gpd.GeoDataFrame, block: Block) -> gpd.GeoDataFrame:
    """Clip `lines` to the block, subtract the existing-street corridor (STREET_TOL buffer), and
    keep the interior LineString remainder above the tolerance length -- the added intervention,
    excluding the perimeter/inter-block streets that are already egress."""
    empty = gpd.GeoDataFrame(geometry=[], crs=block.crs)
    if lines.empty:
        return empty
    clipped = lines.clip(block.boundary)
    if clipped.empty:
        return empty
    street_corridor = unary_union(list(block.streets.geometry)).buffer(STREET_TOL)
    remainder = clipped.geometry.difference(street_corridor).explode(index_parts=False)
    kept = remainder[(~remainder.is_empty)
                     & (remainder.geom_type == "LineString")
                     & (remainder.length > STREET_TOL)]
    return gpd.GeoDataFrame(geometry=list(kept), crs=block.crs)


@dataclass
class DreamComeTrueReblocker:
    source: DesireLineSource
    corridor_m: float = 3.0

    @property
    def identity(self) -> Hashable:
        # Propagate an uncacheable (live) source up so the derivation cache bypasses -- else two
        # different live OSM pulls would key-collide (mirrors clearance + PrebuiltSubstrate).
        if self.source.identity is None:
            return None
        return ("dream_come_true", self.source.identity, float(self.corridor_m))

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior  # accepted for Method conformance; routing is block-only
        bbox = gpd.GeoSeries([block.boundary], crs=block.crs).to_crs(4326).total_bounds
        lines = self.source.desire_lines(
            (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])), block.crs)
        roads = _interior_desire_lines(lines, block)
        return Proposal(
            block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
            proposal_id="dream_come_true", method="dream_come_true",
            params={"segments": len(roads), "corridor_m": self.corridor_m},
            block_identity=block.identity)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/methods/test_dream_come_true.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Lint + type-check**

Run: `pixi run ruff check src/reblock/methods/dream_come_true.py tests/methods/test_dream_come_true.py && pixi run mypy --strict src/reblock/methods/dream_come_true.py`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/methods/dream_come_true.py tests/methods/test_dream_come_true.py
git commit -m "feat: DreamComeTrueReblocker method (clip + dedupe desire-lines into roads)"
```

---

### Task 4: Config wiring + Method-protocol conformance

**Files:**
- Create: `conf/desire_source/osm.yaml`
- Create: `conf/method/dream_come_true.yaml`
- Modify: `conf/config.yaml` (defaults list: add `- desire_source: osm`)
- Modify: `conf/compare_config.yaml` (defaults list: add `- desire_source: osm`; `all_methods`:
  add `dream_come_true` entry)
- Test: `tests/methods/test_dream_come_true.py`

**Interfaces:**
- Consumes: `DreamComeTrueReblocker`, `OSMDesireLines` (Tasks 2-3).
- Produces: a `desire_source` config group + a `dream_come_true` method instantiable from both
  `config.yaml` (`method=dream_come_true`) and `compare_config.yaml` (`all_methods.dream_come_true`).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/methods/test_dream_come_true.py
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def test_dream_come_true_instantiates_from_compare_config() -> None:
    conf_dir = str(Path("conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="compare_config",
                      overrides=["shapefile=x", "methods=[dream_come_true]"])
    method = instantiate(cfg.all_methods["dream_come_true"])
    assert type(method).__name__ == "DreamComeTrueReblocker"
    assert type(method.source).__name__ == "OSMDesireLines"
    assert list(method.source.tags) == ["path", "footway", "track", "steps",
                                         "pedestrian", "living_street"]
    assert method.identity is None                        # live source (no snapshot) -> uncacheable


def test_dream_come_true_instantiates_from_method_group() -> None:
    conf_dir = str(Path("conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config",
                      overrides=["shapefile=x", "method=dream_come_true"])
    assert type(instantiate(cfg.method)).__name__ == "DreamComeTrueReblocker"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/methods/test_dream_come_true.py -k instantiates -v`
Expected: FAIL — Hydra `ConfigCompositionException` / missing `dream_come_true` in `all_methods`.

- [ ] **Step 3: Create the config files and wire the defaults**

`conf/desire_source/osm.yaml`:
```yaml
# OpenStreetMap desire-line source for dream_come_true: the informal footpaths (worn desire-paths)
# for a region, via Overpass. See reblock.methods.desire_lines.OSMDesireLines.
_target_: reblock.methods.desire_lines.OSMDesireLines
tags: [path, footway, track, steps, pedestrian, living_street]
endpoint: https://overpass-api.de/api/interpreter
cache_dir: null       # null -> ~/.cache/reblock/osm
snapshot: null        # a committed GeoJSON path -> load it directly, skip Overpass
```

`conf/method/dream_come_true.yaml`:
```yaml
# Reblocker whose roads are the REAL informal footpaths for the region (from OSM), not synthesized.
# See reblock.methods.dream_come_true.DreamComeTrueReblocker.
_target_: reblock.methods.dream_come_true.DreamComeTrueReblocker
source: ${desire_source}
corridor_m: 3.0
```

In `conf/config.yaml`, add to the `defaults:` list (after `- substrate: chord_diag`):
```yaml
  - desire_source: osm
```

In `conf/compare_config.yaml`, add to the `defaults:` list (after `- substrate: chord_diag`):
```yaml
  - desire_source: osm
```
and add to `all_methods` (after the `clearance_grid` entry):
```yaml
  dream_come_true: {_target_: reblock.methods.dream_come_true.DreamComeTrueReblocker, source: "${desire_source}", corridor_m: 3.0}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/methods/test_dream_come_true.py -k instantiates -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full method test module + confirm the palette picks up the new method**

Run: `pixi run pytest tests/methods/test_dream_come_true.py tests/test_emit.py tests/test_compare.py -q`
Expected: PASS. (`dream_come_true` now appears in `list(cfg.all_methods)`, so `_method_colors` gives
it a registry hue automatically — no emit change needed.)

- [ ] **Step 6: Commit**

```bash
git add conf/desire_source/osm.yaml conf/method/dream_come_true.yaml conf/config.yaml conf/compare_config.yaml tests/methods/test_dream_come_true.py
git commit -m "feat: wire dream_come_true into method + compare config groups"
```

---

### Task 5: Flagship desire-line snapshot + coverage verification

**Files:**
- Create: `examples/multiblock/desire_lines_5810.geojson` (committed OSM snapshot for the region)
- Create: `scripts/fetch_desire_lines_snapshot.py` (one-off fetcher, committed for repeatability)

**Interfaces:**
- Consumes: `OSMDesireLines` (Task 2), the region geometry from `capetown_full`.

> **Executor note:** this task needs **network** (one live Overpass fetch) and the `capetown_full`
> dataset. If subagents lack network, the controller runs it. It is not TDD — its deliverable is a
> committed data file + a verified non-empty coverage count.

- [ ] **Step 1: Write the fetch script**

```python
# scripts/fetch_desire_lines_snapshot.py
"""One-off: fetch the OSM desire-lines for the multiblock flagship region (seed ZAF.9.3.1_1_5810)
and write a committed GeoJSON snapshot so examples/multiblock reproduces offline + byte-stable.
Run: pixi run python scripts/fetch_desire_lines_snapshot.py
"""
from pathlib import Path

import geopandas as gpd
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from reblock.methods.desire_lines import OSMDesireLines
from reblock.pipeline import build_regions

OUT = Path("examples/multiblock/desire_lines_5810.geojson")


def main() -> None:
    conf = str(Path("conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=conf):
        cfg = compose(config_name="compare_config", overrides=[
            "data=capetown_full", "region_builder=dense_cluster",
            "region_builder.max_buildings=3000", "block_ids=[[ZAF.9.3.1_1_5810]]", "max_blocks=1"])
    source = instantiate(cfg.data)
    screen = instantiate(cfg.screen)
    region_builder = instantiate(cfg.region_builder)
    regions = build_regions(source, screen, region_builder, [["ZAF.9.3.1_1_5810"]], 1)
    region = regions[0]
    boundary = gpd.GeoSeries([b.boundary for b in region], crs=region[0].crs).union_all()
    bbox = gpd.GeoSeries([boundary], crs=region[0].crs).to_crs(4326).total_bounds
    osm = OSMDesireLines()   # default tags, live fetch
    lines = osm.desire_lines(
        (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])), region[0].crs)
    print(f"fetched {len(lines)} desire-line ways for the 5810 region")
    assert len(lines) > 20, "sparse OSM coverage -- investigate before committing the snapshot"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines.to_crs(4326).to_file(OUT, driver="GeoJSON")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (network) and verify coverage**

Run: `pixi run python scripts/fetch_desire_lines_snapshot.py`
Expected: prints `fetched <N> desire-line ways` with N in the low hundreds (the earlier probe found
185 path/footway/track/steps ways), then `wrote examples/multiblock/desire_lines_5810.geojson`. If
the assert trips (N ≤ 20), STOP and report — coverage is inadequate and the plan's example-integration
assumptions need revisiting.

- [ ] **Step 3: Commit the snapshot + script**

```bash
git add scripts/fetch_desire_lines_snapshot.py examples/multiblock/desire_lines_5810.geojson
git commit -m "feat: commit OSM desire-line snapshot for the multiblock flagship region"
```

---

### Task 6: Integrate `dream_come_true` into the example comparison + regenerate

**Files:**
- Modify: `examples/multiblock/README.md`
- (Conditionally) Modify: `examples/method-comparison/README.md`
- Modify: `examples/multiblock/*.png` (regenerated compare curves)

**Interfaces:**
- Consumes: the method + config (Tasks 3-4) and the committed snapshot (Task 5).

> **Executor note:** long compute (region compare). Controller may run the regeneration. Deliverable:
> `dream_come_true` appears in the multiblock comparison with a real curve, READMEs match artifacts.

- [ ] **Step 1: Verify single-block (method-comparison) coverage to decide inclusion there**

Run:
```bash
pixi run python -c "
import geopandas as gpd
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from reblock.methods.desire_lines import OSMDesireLines
from reblock.pipeline import build_regions
with initialize_config_dir(version_base=None, config_dir='conf'):
    cfg = compose(config_name='compare_config', overrides=['data=capetown_full','block_ids=[[ZAF.9.3.1_1_40972]]','max_blocks=1'])
src, scr, rb = instantiate(cfg.data), instantiate(cfg.screen), instantiate(cfg.region_builder)
region = build_regions(src, scr, rb, [['ZAF.9.3.1_1_40972']], 1)[0]
b = region[0]
bbox = gpd.GeoSeries([b.boundary], crs=b.crs).to_crs(4326).total_bounds
lines = OSMDesireLines().desire_lines(tuple(float(x) for x in bbox), b.crs)
print('block 40972 desire-line ways:', len(lines))
"
```
Expected: prints a count. **Decision rule:** if ≥ 5 interior ways, include `dream_come_true` in the
method-comparison compare too; if fewer, feature it in multiblock only and add one sentence to
`examples/method-comparison/README.md` saying dream_come_true is a region-scale method shown in the
multiblock flagship (no silent empty curve).

- [ ] **Step 2: Regenerate the multiblock comparison with dream_come_true (using the snapshot)**

Run (length + displacement; the snapshot keeps it offline + reproducible):
```bash
SNAP=+all_methods.dream_come_true.source.snapshot=examples/multiblock/desire_lines_5810.geojson
pixi run python -m reblock.compare \
  data=capetown_full region_builder=dense_cluster region_builder.max_buildings=3000 \
  "block_ids=[[ZAF.9.3.1_1_5810]]" \
  methods=[dijkstra,mesh,clearance,greedy_arterial_buildable,dream_come_true] max_blocks=1 \
  all_methods.clearance.max_roads=3000 \
  all_methods.greedy_arterial_buildable.candidate_policy=fixed \
  +all_methods.greedy_arterial_buildable.max_anchors=64 $SNAP \
  hydra.run.dir=/tmp/dct_len
# repeat with cost=displacement -> hydra.run.dir=/tmp/dct_disp
```
Expected: writes `curve_{metric}_<label>.png` + `auc_table_*.csv` including a `dream_come_true` row.
Copy the four `curve_*` into `examples/multiblock/compare_{access,resistance,directness,efficiency}.png`
and the displacement `curve_directness_*` into `compare_directness_displacement.png` (matching the
existing curated names).

- [ ] **Step 3: Update the multiblock README**

Add `dream_come_true` to the §4 command's `methods=[...]` and the `+…snapshot=` override; add its
row/column to the AUC table and the displacement table using the values printed in the run log; add
one prose sentence interpreting where the *real* paths land (e.g. high directness at low paving, or
whatever the numbers show — read them from the run, do not invent). Update the run.log pointer.

- [ ] **Step 4: Run the full suite + confirm READMEs match artifacts**

Run: `pixi run pytest -q`
Expected: all pass. Manually confirm every number in the edited README tables equals the regenerated
CSV/run-log values (the "no stale numbers" rule).

- [ ] **Step 5: Commit**

```bash
git add examples/multiblock/ examples/method-comparison/README.md
git commit -m "docs: add dream_come_true (real OSM desire-lines) to the multiblock comparison"
```

---

## Notes for the executor

- **Out of scope (Phase 2):** the satellite-imagery `ImageryDesireLines` source. Do not build it here;
  the `DesireLineSource` seam is what it will plug into later.
- Tasks 1-4 are pure TDD (no network, fast) and ideal for fresh subagents. Tasks 5-6 need network +
  the `capetown_full` dataset + longer compute; run them where those are available (likely the
  controller) — their steps are exact regardless of who executes.
