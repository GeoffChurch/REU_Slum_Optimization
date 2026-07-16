# dream_come_true_cv (Phase 2: imagery wide-corridor desire-lines) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline, per the owner's choice for this feature). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ImageryDesireLines` — a `DesireLineSource` that detects the wide bare-earth corridors directly from satellite imagery — landing an imagery-derived reblocker variant `dream_come_true_cv` beside the OSM one.

**Architecture:** A new `src/reblock/methods/imagery.py` fetches an Esri World Imagery tile mosaic for a bbox and runs a classical detector (bare-earth likelihood → wide-disk opening → skeletonize → vectorize) → LineStrings. `ImageryDesireLines` mirrors `OSMDesireLines` (snapshot → cache → live). `DreamComeTrueReblocker` (Phase 1) is reused unchanged. The `dream_come_true` config key is renamed to `dream_come_true_osm` and `dream_come_true_cv` is added.

**Tech Stack:** Python, scikit-image (NEW), PIL, numpy, scipy.ndimage, networkx, shapely, geopandas, Hydra, pixi.

## Global Constraints

- **One new dependency only: `scikit-image`.** No `torch`, no `cv2`. Everything else is present.
- **No network / no live imagery in tests** — the mosaic fetch takes an injectable `tile_getter`; the detector is tested on a committed fixture image; the source is tested via snapshot/stub.
- **Reproducibility = committed detected-lines GeoJSON snapshot** (parallel to OSM). Examples load snapshots; detection runs once in the fetch script.
- **Migrate, never accommodate** — rename `dream_come_true` → `dream_come_true_osm` cleanly (no compat alias); inline each variant's source dict in `all_methods` (retires the `${desire_source}` interpolation quirk).
- **Reuse `DreamComeTrueReblocker` verbatim** — it is source-agnostic; do not modify it.
- Imagery: **Esri World Imagery** ArcGIS REST tiles, `https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}`, default **zoom 19** (0.25 m/px), no key; `User-Agent` header set.
- Every changed `src`/`test` file passes `pixi run ruff check` and `pixi run mypy --strict`. Full suite green (`pixi run pytest -q`) before each commit.

---

### Task 1: Add the scikit-image dependency

**Files:** Modify `pyproject.toml` (`[tool.pixi.dependencies]`).

**Interfaces:**
- Produces: `skimage.morphology.{binary_opening, skeletonize, disk, remove_small_objects}` importable.

- [ ] **Step 1: Add the dependency**

Run: `pixi add scikit-image`
Expected: pixi resolves + installs; `pyproject.toml` gains `scikit-image = "*"` under `[tool.pixi.dependencies]`.

- [ ] **Step 2: Confirm the imports needed downstream**

Run:
```bash
pixi run python -c "from skimage.morphology import binary_opening, skeletonize, disk, remove_small_objects; import skimage; print('skimage', skimage.__version__)"
```
Expected: prints a version, no ImportError.

- [ ] **Step 3: Confirm nothing broke**

Run: `pixi run pytest -q 2>&1 | tail -2`
Expected: `349 passed` (unchanged — the dep add is inert until used).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml pixi.lock
git commit -m "build: add scikit-image (for the dream_come_true_cv imagery detector)"
```

---

### Task 2: Imagery mosaic fetch + tile math

**Files:**
- Create: `src/reblock/methods/imagery.py`
- Test: `tests/methods/test_imagery.py`

**Interfaces:**
- Produces:
  - `_lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]`
  - `_mosaic_extent_3857(x0: int, y0: int, x1: int, y1: int, z: int) -> tuple[float, float, float, float]` — returns `(xmin, ymin, xmax, ymax)` in EPSG:3857 for the tile block `x0..x1`, `y0..y1`.
  - `fetch_mosaic(bbox_wgs84, zoom, endpoint, tile_getter=None) -> tuple[np.ndarray, tuple[float,float,float,float]]` — `bbox_wgs84 = (min_lon,min_lat,max_lon,max_lat)`; returns `(rgb HxWx3 uint8, extent_3857)`. `tile_getter(z,x,y)->PIL.Image` is injectable (default fetches Esri); the fetched tile block is padded by 1 tile each side.

- [ ] **Step 1: Write the failing tests**

```python
# tests/methods/test_imagery.py
import numpy as np
from PIL import Image

from reblock.methods.imagery import _lonlat_to_tile, _mosaic_extent_3857, fetch_mosaic


def test_lonlat_to_tile_matches_web_mercator() -> None:
    # z19 tile for the block-40972 centre (spike-verified values).
    assert _lonlat_to_tile(18.58064, -33.97795, 19) == (289204, 314812)


def test_mosaic_extent_3857_is_tile_aligned_and_ordered() -> None:
    x0, y0, x1, y1, z = 289203, 314811, 289205, 314813, 19
    xmin, ymin, xmax, ymax = _mosaic_extent_3857(x0, y0, x1, y1, z)
    assert xmin < xmax and ymin < ymax
    ts = 2 * np.pi * 6378137.0 / (2 ** z)                 # one tile in 3857 metres
    assert np.isclose(xmax - xmin, (x1 - x0 + 1) * ts)    # width spans the tile columns
    assert np.isclose(ymax - ymin, (y1 - y0 + 1) * ts)


def test_fetch_mosaic_stitches_with_injected_tiles() -> None:
    # A stub tile_getter returns a solid tile whose colour encodes (x,y) -- no network.
    def stub(z: int, x: int, y: int) -> Image.Image:
        return Image.new("RGB", (256, 256), (x % 256, y % 256, z))
    # a tiny bbox spanning ~1 tile; padded to 3x3
    rgb, ext = fetch_mosaic((18.5806, -33.9780, 18.5807, -33.9779), 19, "http://x", tile_getter=stub)
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    assert rgb.shape[0] % 256 == 0 and rgb.shape[1] % 256 == 0
    assert len(ext) == 4 and ext[0] < ext[2] and ext[1] < ext[3]
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/methods/test_imagery.py -v`
Expected: FAIL — `ModuleNotFoundError: reblock.methods.imagery`.

- [ ] **Step 3: Implement the mosaic math + fetch**

```python
# src/reblock/methods/imagery.py
"""Satellite-imagery desire-line source for dream_come_true_cv: fetch an Esri World Imagery tile
mosaic for a region bbox and detect the wide bare-earth corridors (classical CV -- no trained
model). See docs/superpowers/specs/2026-07-15-dream-come-true-cv-design.md."""
from __future__ import annotations

import io
import math
import urllib.request
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from PIL import Image

_R = 6378137.0
_ESRI = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile"
_UA = "reblock-dream-come-true-cv/0.1 (informal-settlement research)"
TileGetter = Callable[[int, int, int], Image.Image]


def _lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def _mosaic_extent_3857(x0: int, y0: int, x1: int, y1: int, z: int) -> tuple[float, float, float, float]:
    ts = 2 * math.pi * _R / (2 ** z)          # tile size in 3857 metres
    xmin = -math.pi * _R + x0 * ts
    xmax = -math.pi * _R + (x1 + 1) * ts
    ymax = math.pi * _R - y0 * ts
    ymin = math.pi * _R - (y1 + 1) * ts
    return xmin, ymin, xmax, ymax


def _esri_tile(z: int, x: int, y: int, endpoint: str = _ESRI) -> Image.Image:
    req = urllib.request.Request(f"{endpoint}/{z}/{y}/{x}", headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as r:      # noqa: S310 (trusted endpoint)
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def fetch_mosaic(
    bbox_wgs84: tuple[float, float, float, float], zoom: int, endpoint: str,
    tile_getter: TileGetter | None = None,
) -> tuple[NDArray[np.uint8], tuple[float, float, float, float]]:
    """Fetch + stitch the Esri tiles covering `bbox_wgs84` (padded 1 tile each side) at `zoom`.
    Returns (rgb HxWx3 uint8, EPSG:3857 extent (xmin,ymin,xmax,ymax)). `tile_getter(z,x,y)->Image`
    is injectable for tests; default fetches Esri from `endpoint`."""
    get = tile_getter or (lambda z, x, y: _esri_tile(z, x, y, endpoint))
    x0, y1 = _lonlat_to_tile(bbox_wgs84[0], bbox_wgs84[1], zoom)   # min lon / min lat
    x1, y0 = _lonlat_to_tile(bbox_wgs84[2], bbox_wgs84[3], zoom)   # max lon / max lat
    x0 -= 1; y0 -= 1; x1 += 1; y1 += 1                             # pad
    cols, rows = x1 - x0 + 1, y1 - y0 + 1
    mosaic = Image.new("RGB", (cols * 256, rows * 256))
    for j, ty in enumerate(range(y0, y1 + 1)):
        for i, tx in enumerate(range(x0, x1 + 1)):
            mosaic.paste(get(zoom, tx, ty), (i * 256, j * 256))
    return np.asarray(mosaic, dtype=np.uint8), _mosaic_extent_3857(x0, y0, x1, y1, zoom)
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run pytest tests/methods/test_imagery.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint + type-check**

Run: `pixi run ruff check src/reblock/methods/imagery.py tests/methods/test_imagery.py && pixi run mypy --strict src/reblock/methods/imagery.py`
Expected: pass. (If ruff flags `S310`, the inline noqa covers it; the tile-getter default lambda may need `# noqa` if flagged.)

- [ ] **Step 6: Commit**

```bash
git add src/reblock/methods/imagery.py tests/methods/test_imagery.py
git commit -m "feat: Esri imagery mosaic fetch + tile/3857 math for dream_come_true_cv"
```

---

### Task 3: The wide-corridor detector

**Files:**
- Modify: `src/reblock/methods/imagery.py`
- Test: `tests/methods/test_imagery.py`

**Interfaces:**
- Consumes: `fetch_mosaic` output (rgb, extent) from Task 2.
- Produces:
  - `_ground_mpp(extent_3857, width_px) -> float` — ground metres per pixel (3857 stretched by latitude).
  - `_skeleton_to_lines(skel: NDArray[np.bool_]) -> list[list[tuple[int, int]]]` — skeleton bool array → polylines as `(row, col)` pixel chains.
  - `detect_corridors(rgb, extent_3857, crs, *, min_corridor_m=3.0, min_len_m=8.0, smooth_sigma=0.10, shadow_v=0.28, lik_thr=0.35) -> gpd.GeoDataFrame` — the detected wide-corridor centrelines as LineStrings in `crs`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/methods/test_imagery.py
import geopandas as gpd
from pyproj import CRS
from reblock.methods.imagery import detect_corridors

UTM = CRS.from_epsg(32734)
EXT = (2068156.7, -4026014.7, 2068615.3, -4025632.5)   # spike mosaic 3857 extent (any plausible box)


def _synthetic_scene(h=384, w=384) -> np.ndarray:
    # textured grey "roofs" everywhere + a bright SMOOTH tan horizontal corridor band (rows 180-205).
    rng = np.random.default_rng(0)
    img = (rng.integers(70, 130, (h, w, 3))).astype(np.uint8)          # noisy grey roofs
    img[180:205, :, :] = np.array([200, 185, 150], dtype=np.uint8)     # smooth bright tan corridor
    return img


def test_detect_finds_the_smooth_corridor_band() -> None:
    lines = detect_corridors(_synthetic_scene(), EXT, UTM, min_corridor_m=1.0, min_len_m=2.0)
    assert isinstance(lines, gpd.GeoDataFrame) and lines.crs == UTM
    assert len(lines) >= 1
    # the band is horizontal, so the detected centreline spans most of the width
    widest = max(lines.geometry, key=lambda g: g.bounds[2] - g.bounds[0])
    assert (widest.bounds[2] - widest.bounds[0]) > 0.4 * (EXT[2] - EXT[0])


def test_detect_returns_empty_on_pure_texture() -> None:
    rng = np.random.default_rng(1)
    noise = (rng.integers(70, 130, (384, 384, 3))).astype(np.uint8)
    lines = detect_corridors(noise, EXT, UTM, min_corridor_m=1.0, min_len_m=2.0)
    assert len(lines) == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/methods/test_imagery.py -k detect -v`
Expected: FAIL — `ImportError: cannot import name 'detect_corridors'`.

- [ ] **Step 3: Implement the detector**

Append to `src/reblock/methods/imagery.py` (add imports at top: `import networkx as nx`, `import geopandas as gpd`, `from pyproj import CRS`, `from scipy import ndimage`, `from shapely.geometry import LineString`, `from matplotlib.colors import rgb_to_hsv`, `from skimage.morphology import binary_opening, remove_small_objects, skeletonize, disk`):

```python
def _ground_mpp(extent_3857: tuple[float, float, float, float], width_px: int) -> float:
    """Ground metres per pixel. 3857 is Web-Mercator-stretched, so correct by latitude at the
    mosaic centre: ground = (3857 width / px) * cos(lat)."""
    xmin, ymin, xmax, ymax = extent_3857
    lat = 2 * math.atan(math.exp((ymin + ymax) / 2 / _R)) - math.pi / 2
    return (xmax - xmin) / width_px * math.cos(lat)


def _skeleton_to_lines(skel: NDArray[np.bool_]) -> list[list[tuple[int, int]]]:
    """1-px skeleton -> polylines. Build an 8-neighbour pixel graph; each polyline is a chain of
    degree-2 pixels between two non-degree-2 nodes (junctions/endpoints), plus any pure loops."""
    ys, xs = np.nonzero(skel)
    pix = set(zip((int(v) for v in ys), (int(v) for v in xs)))
    g: nx.Graph = nx.Graph()
    g.add_nodes_from(pix)
    for (y, x) in pix:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy or dx) and (y + dy, x + dx) in pix:
                    g.add_edge((y, x), (y + dy, x + dx))

    def walk(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
        chain, prev, cur = [a, b], a, b
        while g.degree(cur) == 2:
            nxts = [z for z in g.neighbors(cur) if z != prev]
            if not nxts or nxts[0] == a:
                break
            prev, cur = cur, nxts[0]
            chain.append(cur)
        return chain

    lines: list[list[tuple[int, int]]] = []
    seen: set[frozenset[tuple[int, int]]] = set()
    for node in [n for n in g.nodes if g.degree(n) != 2]:
        for nb in g.neighbors(node):
            if frozenset((node, nb)) in seen:
                continue
            chain = walk(node, nb)
            seen.update(frozenset(e) for e in zip(chain, chain[1:]))
            lines.append(chain)
    for a, b in g.edges:                    # leftover pure loops
        if frozenset((a, b)) not in seen:
            chain = walk(a, b)
            seen.update(frozenset(e) for e in zip(chain, chain[1:]))
            lines.append(chain)
    return lines


def detect_corridors(
    rgb: NDArray[np.uint8], extent_3857: tuple[float, float, float, float], crs: CRS, *,
    min_corridor_m: float = 3.0, min_len_m: float = 8.0, smooth_sigma: float = 0.10,
    shadow_v: float = 0.28, lik_thr: float = 0.35,
) -> gpd.GeoDataFrame:
    """Detect wide bare-earth corridors: likelihood (bright*smooth*not-green*not-shadow) -> threshold
    -> wide-disk opening (keeps only WIDE bare earth) -> skeletonize -> vectorize -> LineStrings in
    `crs`. Only the main corridors survive; the fine interior network is out of scope by design."""
    h, w = rgb.shape[:2]
    f = rgb.astype(np.float64) / 255.0
    gray = f.mean(2)
    mean = ndimage.uniform_filter(gray, 7)
    var = ndimage.uniform_filter(gray * gray, 7) - mean * mean
    smooth = np.clip(1.0 - np.sqrt(np.clip(var, 0, None)) / smooth_sigma, 0.0, 1.0)
    hsv = rgb_to_hsv(f)
    green = (hsv[..., 0] > 0.18) & (hsv[..., 0] < 0.45) & (hsv[..., 1] > 0.22)
    lik = hsv[..., 2] * smooth * (~green)
    lik[hsv[..., 2] < shadow_v] = 0.0

    mpp = _ground_mpp(extent_3857, w)
    r = max(1, int(round((min_corridor_m / 2) / mpp)))
    mask = lik > lik_thr
    mask = binary_opening(mask, disk(r))
    mask = remove_small_objects(mask, min_size=int((min_corridor_m / mpp) ** 2))
    skel = skeletonize(mask)

    xmin, ymin, xmax, ymax = extent_3857
    geoms = []
    for chain in _skeleton_to_lines(np.asarray(skel, dtype=bool)):
        if len(chain) < 2:
            continue
        pts = [(xmin + (c + 0.5) / w * (xmax - xmin), ymax - (rr + 0.5) / h * (ymax - ymin))
               for (rr, c) in chain]
        geoms.append(LineString(pts))
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=CRS.from_epsg(3857)).to_crs(crs)
    gdf["geometry"] = gdf.geometry.simplify(mpp)                 # drop pixel jitter (~1 px)
    gdf = gdf[gdf.geometry.length >= min_len_m].reset_index(drop=True)
    return gpd.GeoDataFrame(geometry=list(gdf.geometry), crs=crs)
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run pytest tests/methods/test_imagery.py -k detect -v`
Expected: PASS (2 tests). If the synthetic corridor isn't found, loosen `lik_thr`/`smooth_sigma` in the *test call* only — the defaults target real imagery.

- [ ] **Step 5: Lint + type-check**

Run: `pixi run ruff check src/reblock/methods/imagery.py tests/methods/test_imagery.py && pixi run mypy --strict src/reblock/methods/imagery.py`
Expected: pass. (`nx.Graph` may need the annotation shown; `remove_small_objects` returns an array — fine.)

- [ ] **Step 6: Commit**

```bash
git add src/reblock/methods/imagery.py tests/methods/test_imagery.py
git commit -m "feat: classical wide-corridor detector (likelihood/opening/skeleton/vectorize)"
```

---

### Task 4: `ImageryDesireLines` source

**Files:**
- Modify: `src/reblock/methods/imagery.py`
- Test: `tests/methods/test_imagery.py`

**Interfaces:**
- Consumes: `fetch_mosaic`, `detect_corridors` (Tasks 2-3); the `DesireLineSource` protocol shape from `desire_lines.py` (`desire_lines(bbox_wgs84, crs) -> GeoDataFrame`, `identity`).
- Produces: `@dataclass class ImageryDesireLines` with fields `zoom:int=19`, `endpoint:str=_ESRI`, `cache_dir:str|None=None`, `snapshot:str|None=None`, `min_corridor_m:float=3.0`, `min_len_m:float=8.0`. Methods `desire_lines`, `identity`. Fetch precedence: snapshot → cache → live (fetch+detect). Mirrors `OSMDesireLines`.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/methods/test_imagery.py
from pathlib import Path
import pytest
from shapely.geometry import LineString
from reblock.methods.imagery import ImageryDesireLines


def _write_geojson(p: Path, lines) -> None:
    gpd.GeoDataFrame(geometry=[LineString(c) for c in lines], crs=CRS.from_epsg(4326)).to_file(p, driver="GeoJSON")


def test_snapshot_loaded_without_fetch(tmp_path: Path) -> None:
    # The snapshot branch returns before fetch_mosaic is ever reached; passing an unreachable
    # endpoint proves no network is touched (would raise if it fell through to a live fetch).
    snap = tmp_path / "cv.geojson"
    _write_geojson(snap, [[(18.74, -33.84), (18.741, -33.841)]])
    src = ImageryDesireLines(snapshot=str(snap), endpoint="http://127.0.0.1:0/unreachable")
    gdf = src.desire_lines((18.5, -34.0, 18.6, -33.9), UTM)
    assert len(gdf) == 1 and gdf.crs == UTM


def test_identity_none_when_live_stable_with_snapshot(tmp_path: Path) -> None:
    assert ImageryDesireLines().identity is None
    snap = tmp_path / "cv.geojson"
    _write_geojson(snap, [[(18.74, -33.84), (18.741, -33.841)]])
    ident = ImageryDesireLines(snapshot=str(snap)).identity
    assert ident is not None and ident[0] == "imagery"


def test_live_fetches_then_detects(tmp_path: Path) -> None:
    def stub(z, x, y):
        im = Image.new("RGB", (256, 256), (95, 95, 95))       # grey roof
        arr = np.asarray(im).copy(); arr[120:140, :, :] = (200, 185, 150)  # tan band
        return Image.fromarray(arr)
    src = ImageryDesireLines(cache_dir=str(tmp_path), min_corridor_m=1.0, min_len_m=2.0)
    gdf = src.desire_lines((18.5806, -33.9780, 18.5807, -33.9779), UTM, _tile_getter=stub)
    assert gdf.crs == UTM and len(gdf) >= 1
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/methods/test_imagery.py -k "snapshot or identity_none or live_fetches" -v`
Expected: FAIL — `ImportError: cannot import name 'ImageryDesireLines'`.

- [ ] **Step 3: Implement the source**

Append to `src/reblock/methods/imagery.py` (add imports: `import hashlib`, `from collections.abc import Hashable`, `from dataclasses import dataclass`, `from pathlib import Path`):

```python
def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "reblock" / "imagery"


@dataclass
class ImageryDesireLines:
    """A DesireLineSource that detects wide bare-earth corridors from Esri World Imagery. Fetch
    precedence: a committed `snapshot` GeoJSON of already-detected lines (byte-stable, no network) ->
    a disk cache -> a live mosaic fetch + detect. `identity` is None when live (uncacheable), and a
    stable tuple keyed on the snapshot's content hash when a snapshot is pinned (mirrors OSM)."""

    zoom: int = 19
    endpoint: str = _ESRI
    cache_dir: str | None = None
    snapshot: str | None = None
    min_corridor_m: float = 3.0
    min_len_m: float = 8.0

    @property
    def identity(self) -> Hashable:
        if self.snapshot is None:
            return None
        digest = hashlib.sha256(Path(self.snapshot).read_bytes()).hexdigest()[:16]
        return ("imagery", self.zoom, self.min_corridor_m, self.min_len_m, digest)

    def _cache_path(self, bbox_wgs84: tuple[float, float, float, float]) -> Path:
        root = Path(self.cache_dir) if self.cache_dir else _default_cache_dir()
        key = f"z{self.zoom}c{self.min_corridor_m}l{self.min_len_m}@{','.join(f'{c:.5f}' for c in bbox_wgs84)}"
        return root / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.geojson"

    def desire_lines(
        self, bbox_wgs84: tuple[float, float, float, float], crs: CRS,
        _tile_getter: TileGetter | None = None,
    ) -> gpd.GeoDataFrame:
        if self.snapshot is not None:
            return gpd.read_file(self.snapshot).to_crs(crs)
        cache_path = self._cache_path(bbox_wgs84)
        if cache_path.exists():
            return gpd.read_file(cache_path).to_crs(crs)
        rgb, ext = fetch_mosaic(bbox_wgs84, self.zoom, self.endpoint, tile_getter=_tile_getter)
        lines = detect_corridors(rgb, ext, crs, min_corridor_m=self.min_corridor_m,
                                 min_len_m=self.min_len_m)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lines.to_crs(4326).to_file(cache_path, driver="GeoJSON")
        return lines
```

Note: the `_tile_getter` keyword is accepted for tests only; the real `desire_lines(bbox, crs)` protocol call omits it (default None → live Esri). Drop the stub-only `desire_lines_via_fetch` reference in the first test — instead assert no fetch by giving the snapshot test no network reachability (it returns before `fetch_mosaic`). Simplify that test to just: snapshot set → returns the snapshot lines (the `pytest.fail` monkeypatch is unnecessary since the snapshot branch returns first).

- [ ] **Step 4: Run to verify pass**

Run: `pixi run pytest tests/methods/test_imagery.py -v`
Expected: PASS (all imagery tests). Adjust the snapshot test per the note above if needed.

- [ ] **Step 5: Lint + type-check + full module**

Run: `pixi run ruff check src/reblock/methods/imagery.py tests/methods/test_imagery.py && pixi run mypy --strict src/reblock/methods/imagery.py`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/methods/imagery.py tests/methods/test_imagery.py
git commit -m "feat: ImageryDesireLines source (snapshot/cache/live detect) for dream_come_true_cv"
```

---

### Task 5: Config wiring + the osm/cv rename

**Files:**
- Create: `conf/desire_source/imagery.yaml`
- Modify: `conf/compare_config.yaml` (rename `dream_come_true` → `dream_come_true_osm`, add `dream_come_true_cv`; inline both sources)
- Modify: `tests/methods/test_dream_come_true.py` (the conformance test that reads `all_methods["dream_come_true"]`)
- Test: `tests/methods/test_imagery.py`

**Interfaces:**
- Consumes: `ImageryDesireLines` (Task 4); `OSMDesireLines`, `DreamComeTrueReblocker` (Phase 1).
- Produces: `all_methods.dream_come_true_osm` + `all_methods.dream_come_true_cv`, both instantiable.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/methods/test_imagery.py
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def test_both_variants_instantiate_from_compare_config() -> None:
    conf = str(Path("conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=conf):
        cfg = compose(config_name="compare_config",
                      overrides=["shapefile=x", "methods=[dream_come_true_osm,dream_come_true_cv]"])
    osm = instantiate(cfg.all_methods["dream_come_true_osm"])
    cv = instantiate(cfg.all_methods["dream_come_true_cv"])
    assert type(osm).__name__ == "DreamComeTrueReblocker" and type(osm.source).__name__ == "OSMDesireLines"
    assert type(cv).__name__ == "DreamComeTrueReblocker" and type(cv.source).__name__ == "ImageryDesireLines"
    assert "dream_come_true" not in cfg.all_methods       # bare key is gone (renamed)
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/methods/test_imagery.py -k both_variants -v`
Expected: FAIL — `KeyError: dream_come_true_osm` (not yet in all_methods).

- [ ] **Step 3: Create the imagery source config + rewrite the all_methods entries**

`conf/desire_source/imagery.yaml`:
```yaml
# Satellite-imagery desire-line source for dream_come_true_cv: detect the wide bare-earth corridors
# from Esri World Imagery. See reblock.methods.imagery.ImageryDesireLines.
_target_: reblock.methods.imagery.ImageryDesireLines
zoom: 19
min_corridor_m: 3.0
min_len_m: 8.0
snapshot: null
```

In `conf/compare_config.yaml`, replace the single `dream_come_true: {...}` line with two entries that
**inline** their sources (so `all_methods.dream_come_true_*.source.snapshot=<path>` overrides work):
```yaml
  dream_come_true_osm: {_target_: reblock.methods.dream_come_true.DreamComeTrueReblocker, source: {_target_: reblock.methods.desire_lines.OSMDesireLines}, corridor_m: 3.0}
  dream_come_true_cv: {_target_: reblock.methods.dream_come_true.DreamComeTrueReblocker, source: {_target_: reblock.methods.imagery.ImageryDesireLines}, corridor_m: 3.0}
```

- [ ] **Step 4: Migrate the Phase-1 conformance test**

In `tests/methods/test_dream_come_true.py`, the test `test_dream_come_true_instantiates_from_compare_config` reads `cfg.all_methods["dream_come_true"]`. Change that key to `"dream_come_true_osm"` (its assertions about `OSMDesireLines` + default tags + `identity is None` still hold — the inline source has no snapshot, so it's live).

- [ ] **Step 5: Run to verify pass + no regressions in the touched suites**

Run: `pixi run pytest tests/methods/test_imagery.py tests/methods/test_dream_come_true.py tests/test_compare.py -q`
Expected: PASS. (`dream_come_true_osm`/`_cv` now appear in `list(cfg.all_methods)`, so each gets a registry hue automatically.)

- [ ] **Step 6: Commit**

```bash
git add conf/desire_source/imagery.yaml conf/compare_config.yaml tests/methods/test_imagery.py tests/methods/test_dream_come_true.py
git commit -m "feat: wire dream_come_true_cv + rename dream_come_true -> _osm (inline sources)"
```

---

### Task 6: Fetch + commit the CV detected-line snapshots

**Files:**
- Modify: `scripts/fetch_desire_lines_snapshot.py` (add an imagery mode)
- Create: `examples/method-comparison/desire_lines_cv_40972.geojson`, `examples/multiblock/desire_lines_cv_5810.geojson`

> **Executor note:** needs network (live Esri fetch) + `capetown_full`. Not TDD — deliverable is two committed GeoJSONs + a visual sanity check that each has real corridor lines.

- [ ] **Step 1: Add an imagery mode to the fetch script**

Add a `--source imagery` branch (or a second function) that, for the given region seed/overrides,
computes the region bbox and calls `ImageryDesireLines(zoom=19).desire_lines(bbox, region_crs)`
(live), then writes the detected LineStrings to the out path. Reuse the existing region-building code;
swap `OSMDesireLines()` for `ImageryDesireLines()`. Guard: `assert len(lines) >= 3, "too few
corridors detected — investigate"`.

- [ ] **Step 2: Fetch both snapshots**

Run (module form):
```bash
pixi run python -m scripts.fetch_desire_lines_snapshot examples/multiblock/desire_lines_cv_5810.geojson \
  block_ids=[[ZAF.9.3.1_1_5810]] region_builder=dense_cluster region_builder.max_buildings=3000 --source imagery
pixi run python -m scripts.fetch_desire_lines_snapshot examples/method-comparison/desire_lines_cv_40972.geojson \
  block_ids=[[ZAF.9.3.1_1_40972]] --source imagery
```
Expected: each prints a corridor count ≥ 3 and writes the GeoJSON. **Eyeball** each (plot the lines over the region) before committing — if a region's detection is empty/garbage, STOP and report (like OSM's coverage gate).

- [ ] **Step 3: Commit**

```bash
git add scripts/fetch_desire_lines_snapshot.py examples/multiblock/desire_lines_cv_5810.geojson examples/method-comparison/desire_lines_cv_40972.geojson
git commit -m "feat: commit CV wide-corridor snapshots for both flagship regions"
```

---

### Task 7: Example integration (both variants, one regeneration)

**Files:** Modify `examples/method-comparison/README.md`, `examples/multiblock/README.md`, and the regenerated `examples/**` plots/CSVs/renders.

> **Executor note:** long compute. Deliverable: both examples show `dream_come_true_osm` + `dream_come_true_cv`; READMEs match; numbers verbatim from the run logs.

- [ ] **Step 1: Regenerate both examples with both variants**

Run the compares with `methods=[...,dream_come_true_osm,dream_come_true_cv]` and per-variant snapshot
overrides `+all_methods.dream_come_true_osm.source.snapshot=<osm.geojson>` +
`+all_methods.dream_come_true_cv.source.snapshot=<cv.geojson>` (inline sources → these override
directly). method-comparison adds both to `[topology,clearance,greedy_arterial_buildable,...]`;
multiblock to `[clearance,greedy_arterial_buildable,...]`. Length + displacement passes. Copy the
regenerated `curve_*/compare_*/frontier_*/tradeoff_*` in.

- [ ] **Step 2: Render `after_dream_come_true_cv.jpg` for the method-comparison gallery**

Run `reblock.run method=dream_come_true desire_source=imagery +desire_source.snapshot=<cv_40972.geojson>
block_ids=[[ZAF.9.3.1_1_40972]] render.enabled=true`, resize the after-render to 1200×1277 → `after_dream_come_true_cv.jpg`.

- [ ] **Step 3: Rewrite both READMEs**

Rename all `dream_come_true` → `dream_come_true_osm`; add the `_cv` row/column/gallery-cell using the
regenerated run-log numbers verbatim. Frame honestly: OSM = fuller mapped network; CV = the main
bare-earth corridors detectable from orbit (a subset). Update the reproduce commands (the two snapshot
overrides). Confirm every number matches the CSVs/logs.

- [ ] **Step 4: Full suite + consistency**

Run: `pixi run pytest -q` (expect all pass) and grep the READMEs for any stale bare `dream_come_true`
(without `_osm`/`_cv`) or `auc`/`AUC` values.

- [ ] **Step 5: Commit**

```bash
git add examples/
git commit -m "docs: add dream_come_true_cv (imagery corridors) to both flagship examples"
```

---

## Notes for the executor (inline)

- Tasks 1-5 are TDD and fast (no network). Tasks 6-7 need network + `capetown_full` + compute.
- The spike scripts (`scratchpad/spike_mosaic.py`, `spike_detect3.py`) validated the mosaic math + the likelihood signal; the plan's detector uses **opening** (wide corridors) rather than the spike's Frangi (thin lines).
- Out of scope (do NOT build): the fine interior footpath network; any trained/deep model.
