# Screen layer (slum detection) S1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A region-scale `Screen` layer that flags dense/compact informal blocks — a `DenseCompactScreen` (cheap column gates → build survivors → mean-depth gate), on-demand cached full-city data, and a `reblock.screen` detect entrypoint.

**Architecture:** A new `Screen` protocol (selectable like `Method`/`Eval`) does `raw data → block_ids`, upstream of the block-build. `DenseCompactScreen` runs a vectorized cheap pass over free kblock columns (`density`, `k_complexity`), then reuses `KblockSource` to build only the survivors and applies a `mean_depth` gate. `ensure_city_data` downloads+caches full-city parquets under `~/.cache/reblock`. A thin `reblock.screen` Hydra app runs the selected Screen and prints the flagged `block_ids`.

**Tech Stack:** Python 3.12, geopandas/pandas, Hydra (`_target_`+instantiate, config groups), pixi (`pixi run check` = ruff + `mypy --strict src tests` + pytest), pyarrow.

## Global Constraints

- `pixi run check` green before every commit.
- **Reuse `KblockSource`** for the fine (build) pass — no duplicate Voronoi/peel.
- **Deterministic:** sorted `block_id` output, no RNG.
- **Downloads → `~/.cache/reblock/`** (never the repo); check-if-exists-else-download (a plain file cache, NOT joblib). The 900 MB real data is never committed; only the ~1.1 MB sample fixture stays committed.
- **Free density signals:** `density = building_count / (block_area_m2 / 1e4)` and `k_complexity` are columns in the kblock geodata — the cheap pass is pure column math (no build, no buildings sjoin).
- **Additive**, except: a new `Screen` protocol in `contracts.py`; the committed `tests/data/kblock/blocks_capetown_sample.parquet` is regenerated (same 301 blocks, two columns added); `fetch_kblock_fixtures.py`'s block-column list is extended.
- **Types:** fully annotated for `mypy --strict`; `X | None`, `list[...]`.
- Interfaces (unchanged, consumed): `KblockSource(blocks_path, buildings_path, region_id="kblock", *, min_buildings=10, block_ids=None)`; `parcel_access_layers(block, roads=None, *, tol=STREET_TOL) -> pd.Series`; the fetch script's `download_dataverse_blocks`, `load_blocks`, `download_capetown_buildings`, `CT_BBOX`.

---

### Task 1: Augment the committed Cape Town fixture with `building_count` + `block_area_m2`

**Files:**
- Modify: `scripts/fetch_kblock_fixtures.py` (retain the two columns for future full regens)
- Create: `scripts/augment_ct_fixture.py` (one-time targeted regen from cached raw)
- Modify (regenerated): `tests/data/kblock/blocks_capetown_sample.parquet`
- Test: `tests/data/test_kblock_source.py` (add a column-presence assertion)

**Interfaces:** Produces a Cape Town sample fixture carrying `building_count` + `block_area_m2` columns (same 301 blocks, same geometry), which Task 2's cheap pass reads.

- [ ] **Step 1: Extend the fetch script's block-column list** so future full regens retain the columns. In `scripts/fetch_kblock_fixtures.py`, change the two `load_blocks`/`_dataverse` read column lists that currently read `["block_id", "k_complexity", "geometry"]`:

In `load_blocks` (line ~283):
```python
    return gpd.read_parquet(
        path, columns=["block_id", "k_complexity", "building_count", "block_area_m2", "geometry"])
```
`select_dense_blocks` already returns `blocks[blocks["block_id"].isin(kept_ids)]`, so the extra columns flow through unchanged.

- [ ] **Step 2: Write the one-time targeted regen** `scripts/augment_ct_fixture.py`:

```python
#!/usr/bin/env python
"""One-time: add building_count + block_area_m2 to the committed Cape Town sample fixture
by joining them from the raw ZAF geodata (cached under outputs/kblock_raw, or downloaded)
onto the SAME 301 committed block_ids. Geometry + block set unchanged; two columns added.
"""
from __future__ import annotations
from pathlib import Path
import geopandas as gpd
from scripts.fetch_kblock_fixtures import download_dataverse_blocks

FIXTURE = Path("tests/data/kblock/blocks_capetown_sample.parquet")
RAW = Path("outputs/kblock_raw/ZAF_geodata.parquet")


def main() -> None:
    if not RAW.exists():
        download_dataverse_blocks("ZAF", RAW)
    fx = gpd.read_parquet(FIXTURE)
    fx["block_id"] = fx["block_id"].astype(str)
    if {"building_count", "block_area_m2"} <= set(fx.columns):
        print("already augmented"); return
    raw = gpd.read_parquet(RAW, columns=["block_id", "building_count", "block_area_m2"])
    raw["block_id"] = raw["block_id"].astype(str)
    merged = fx.merge(raw, on="block_id", how="left", validate="one_to_one")
    assert merged["building_count"].notna().all(), "some fixture blocks missing from raw ZAF"
    assert len(merged) == len(fx)
    gpd.GeoDataFrame(merged, geometry="geometry", crs=fx.crs).to_parquet(FIXTURE)
    print(f"augmented {FIXTURE}: +building_count +block_area_m2 ({len(merged)} blocks)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it** → `PYTHONPATH=. pixi run python scripts/augment_ct_fixture.py`. Expected: `augmented … (301 blocks)`.

- [ ] **Step 4: Add a column-presence test** in `tests/data/test_kblock_source.py`:

```python
def test_capetown_fixture_has_density_columns() -> None:
    import geopandas as gpd
    cols = set(gpd.read_parquet(CT_BLOCKS).columns)
    assert {"building_count", "block_area_m2"} <= cols   # the Screen's cheap-pass signals
```

- [ ] **Step 5: `pixi run check`** — green (the added columns are additive; `KblockSource` reads only `block_id`/`k_complexity`/`geometry`, so the pinned tests are unchanged). **Commit:**

```bash
git add scripts/fetch_kblock_fixtures.py scripts/augment_ct_fixture.py \
        tests/data/kblock/blocks_capetown_sample.parquet tests/data/test_kblock_source.py
git commit -m "chore: augment Cape Town fixture with building_count + block_area_m2 (Screen signals)"
```

---

### Task 2: `Screen` protocol + `DenseCompactScreen`

**Files:**
- Modify: `src/reblock/contracts.py` (add `Screen` protocol)
- Create: `src/reblock/screen/__init__.py`, `src/reblock/screen/dense_compact.py`
- Create: `conf/screen/dense_compact.yaml`
- Test: `tests/screen/test_dense_compact.py`

**Interfaces:**
- Produces: `Screen` protocol (`select(self) -> list[str]`); `DenseCompactScreen(blocks_path, buildings_path, *, density_min=30.0, mean_depth_min=1.3, k_min=None, min_buildings=10)` with `select() -> list[str]` and a pure helper `_cheap_survivors(blocks: GeoDataFrame) -> list[str]`.
- Consumes: `KblockSource`, `parcel_access_layers`.

- [ ] **Step 1: Add the `Screen` protocol** to `src/reblock/contracts.py`, after `Eval`:

```python
class Screen(Protocol):
    def select(self) -> list[str]: ...   # informal block_ids, sorted
```

- [ ] **Step 2: Write failing tests** in `tests/screen/test_dense_compact.py`:

```python
from pathlib import Path

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Point, box

from reblock.screen.dense_compact import DenseCompactScreen

ROOT = Path(__file__).resolve().parents[1]
CT_BLOCKS = str(ROOT / "data" / "kblock" / "blocks_capetown_sample.parquet")
CT_BLD = str(ROOT / "data" / "kblock" / "buildings_capetown_sample.parquet")
UTM = CRS.from_epsg(32734)     # Cape Town UTM: valid metric coords so KblockSource reprojects cleanly
EX, NY = 3.0e5, 6.25e6          # a realistic easting/northing base


def _write_synth(tmp: Path) -> tuple[str, str]:
    # A: dense + DEEP (5x5 grid in a 50x50 m block -> ring depths 1/2/3, mean depth 1.4);
    # B: dense but SHALLOW (30 buildings in two rows of a 30x2 m block -> all front a street, mean 1.0);
    # C: SPARSE (density 22/ha -> fails the cheap gate outright).
    a = box(EX, NY, EX + 50, NY + 50)
    b = box(EX + 70, NY, EX + 100, NY + 2)
    c = box(EX + 120, NY, EX + 150, NY + 30)
    blocks = gpd.GeoDataFrame({
        "block_id": ["A", "B", "C"], "k_complexity": [3.0, 2.0, 1.0],
        "building_count": [25, 30, 2], "block_area_m2": [2500.0, 60.0, 900.0],
    }, geometry=[a, b, c], crs=UTM)
    pts = [Point(EX + 5 + 10 * i, NY + 5 + 10 * j) for i in range(5) for j in range(5)]  # A: 5x5
    pts += [Point(EX + 71 + 2 * i, NY + row) for i in range(15) for row in (0.5, 1.5)]   # B: 2 rows
    pts += [Point(EX + 125, NY + 5), Point(EX + 140, NY + 20)]                            # C: 2
    bld = gpd.GeoDataFrame(geometry=pts, crs=UTM)
    bp, dp = tmp / "b.parquet", tmp / "d.parquet"
    blocks.to_parquet(bp); bld.to_parquet(dp)
    return str(bp), str(dp)


def test_cheap_survivors_gate(tmp_path: Path) -> None:
    bp, dp = _write_synth(tmp_path)
    s = DenseCompactScreen(bp, dp, density_min=50.0, min_buildings=10)
    # density/ha: A=25/(2500/1e4)=100, B=30/(60/1e4)=5000, C=2/(900/1e4)=22
    assert s._cheap_survivors(gpd.read_parquet(bp)) == ["A", "B"]   # C (22) fails; sorted


def test_select_two_tier_drops_shallow(tmp_path: Path) -> None:
    bp, dp = _write_synth(tmp_path)
    # cheap keeps A,B; fine gate mean_depth_min=1.2 keeps A (deep, ~1.4), drops B (strip, ~1.0)
    s = DenseCompactScreen(bp, dp, density_min=50.0, mean_depth_min=1.2, min_buildings=10)
    assert s.select() == ["A"]


def test_select_flags_flagship_on_real_fixture() -> None:
    # density_min=80 keeps the test fast: only the very densest blocks survive the cheap gate
    # and get built (the flagship's density is 108/ha, so it survives).
    s = DenseCompactScreen(CT_BLOCKS, CT_BLD, density_min=80.0, mean_depth_min=1.3)
    ids = s.select()
    assert "ZAF.9.3.1_1_44882" in ids and ids == sorted(ids)   # the deep flagship survives
```

- [ ] **Step 3: Run to verify fail** → `ModuleNotFoundError`.

- [ ] **Step 4: Implement** `src/reblock/screen/dense_compact.py` (+ an empty `src/reblock/screen/__init__.py`):

```python
"""DenseCompactScreen: flag dense/compact informal blocks. Cheap pass = vectorized
density (+ optional k) gate over free kblock columns; fine pass = build only survivors
(reusing KblockSource) and keep those whose mean parcel access-depth clears mean_depth_min.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from reblock.data.kblock import KblockSource
from reblock.derive.access import parcel_access_layers


class DenseCompactScreen:
    def __init__(self, blocks_path: str | Path, buildings_path: str | Path, *,
                 density_min: float = 30.0, mean_depth_min: float = 1.3,
                 k_min: float | None = None, min_buildings: int = 10) -> None:
        self.blocks_path = Path(blocks_path)
        self.buildings_path = Path(buildings_path)
        self.density_min = density_min
        self.mean_depth_min = mean_depth_min
        self.k_min = k_min
        self.min_buildings = min_buildings

    def _cheap_survivors(self, blocks: gpd.GeoDataFrame) -> list[str]:
        bid = blocks["block_id"].astype(str)
        density = blocks["building_count"] / (blocks["block_area_m2"] / 1e4)
        mask = density >= self.density_min
        if self.k_min is not None:
            mask = mask & (blocks["k_complexity"] >= self.k_min)
        return sorted(bid[mask.to_numpy()])

    def select(self) -> list[str]:
        blocks = gpd.read_parquet(
            self.blocks_path,
            columns=["block_id", "k_complexity", "building_count", "block_area_m2", "geometry"])
        survivors = self._cheap_survivors(blocks)
        if not survivors:
            return []
        src = KblockSource(self.blocks_path, self.buildings_path, region_id="screen",
                           min_buildings=self.min_buildings, block_ids=survivors)
        kept = [blk.block_id for blk in src.region().blocks
                if float(parcel_access_layers(blk, None).mean()) >= self.mean_depth_min]
        return sorted(kept)
```

- [ ] **Step 5: Create** `conf/screen/dense_compact.yaml` (thresholds only — the entrypoint injects paths, Task 4):

```yaml
# A Screen (selectable like conf/method, conf/eval). Paths are injected by the reblock.screen app.
_target_: reblock.screen.dense_compact.DenseCompactScreen
density_min: 30.0
mean_depth_min: 1.3
# k_min: null   # optional extra cheap gate; off by default (density + mean_depth do the work)
```

- [ ] **Step 6: Run tests** → PASS. **Step 7: `pixi run check`** → green. **Commit:**

```bash
git add src/reblock/contracts.py src/reblock/screen/ conf/screen/dense_compact.yaml \
        tests/screen/test_dense_compact.py
git commit -m "feat: Screen protocol + DenseCompactScreen (cheap density gate -> mean-depth fine gate)"
```

---

### Task 3: On-demand cached data provisioning

**Files:**
- Create: `src/reblock/data/provision.py`
- Create: `conf/data/capetown_full.yaml`
- Test: `tests/data/test_provision.py`

**Interfaces:**
- Produces: `ensure_city_data(city: str, *, cache_dir: Path = ~/.cache/reblock) -> tuple[Path, Path]`; `cached_kblock_source(city: str, *, block_ids=None, min_buildings=10, cache_dir=...) -> KblockSource`.
- Consumes: the fetch script's `download_dataverse_blocks`, `download_capetown_buildings`, `CT_BBOX`; `KblockSource`.

- [ ] **Step 1: Write failing tests** in `tests/data/test_provision.py`:

```python
import shutil
from pathlib import Path

from reblock.data.provision import cached_kblock_source, ensure_city_data

ROOT = Path(__file__).resolve().parents[1]
CT_BLOCKS = ROOT / "data" / "kblock" / "blocks_capetown_sample.parquet"
CT_BLD = ROOT / "data" / "kblock" / "buildings_capetown_sample.parquet"


def _seed(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy(CT_BLOCKS, cache / "blocks_capetown_full.parquet")
    shutil.copy(CT_BLD, cache / "buildings_capetown_full.parquet")


def test_ensure_city_data_uses_cache_no_download(tmp_path: Path) -> None:
    _seed(tmp_path)
    bp, dp = ensure_city_data("capetown", cache_dir=tmp_path)   # must NOT hit the network
    assert bp.exists() and dp.exists()
    assert bp == tmp_path / "blocks_capetown_full.parquet"


def test_cached_kblock_source_builds_from_cache(tmp_path: Path) -> None:
    _seed(tmp_path)
    src = cached_kblock_source("capetown", block_ids=["ZAF.9.3.1_1_44882"], cache_dir=tmp_path)
    blocks = list(src.region().blocks)
    assert [b.block_id for b in blocks] == ["ZAF.9.3.1_1_44882"]
```

- [ ] **Step 2: Run to verify fail** → `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `src/reblock/data/provision.py`:

```python
"""On-demand cached full-city data: download the kblock blocks + Open Buildings for a
city (retaining building_count/block_area_m2), cache under ~/.cache/reblock, return paths.
Plain file cache (check-if-exists), not joblib. The large data is never committed.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from reblock.data.kblock import KblockSource
from scripts.fetch_kblock_fixtures import (
    CT_BBOX, download_capetown_buildings, download_dataverse_blocks)

DEFAULT_CACHE = Path.home() / ".cache" / "reblock"
_ISO3 = {"capetown": "ZAF"}
_BBOX = {"capetown": CT_BBOX}
_BLOCK_COLS = ["block_id", "k_complexity", "building_count", "block_area_m2", "geometry"]


def ensure_city_data(city: str, *, cache_dir: Path = DEFAULT_CACHE) -> tuple[Path, Path]:
    if city not in _ISO3:
        raise ValueError(f"unknown city {city!r}; known: {sorted(_ISO3)}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = cache_dir / f"blocks_{city}_full.parquet"
    buildings_path = cache_dir / f"buildings_{city}_full.parquet"
    if not blocks_path.exists():
        raw = cache_dir / f"{_ISO3[city]}_geodata.parquet"
        if not raw.exists():
            download_dataverse_blocks(_ISO3[city], raw)
        bbox = _BBOX[city]
        blocks = gpd.read_parquet(raw, columns=_BLOCK_COLS)
        blocks.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].reset_index(drop=True).to_parquet(blocks_path)
    if not buildings_path.exists():
        download_capetown_buildings(_BBOX[city], buildings_path)
    return blocks_path, buildings_path


def cached_kblock_source(city: str, *, block_ids: list[str] | None = None,
                         min_buildings: int = 10, cache_dir: Path = DEFAULT_CACHE) -> KblockSource:
    blocks_path, buildings_path = ensure_city_data(city, cache_dir=cache_dir)
    return KblockSource(blocks_path, buildings_path, region_id=city,
                        min_buildings=min_buildings, block_ids=block_ids)
```

- [ ] **Step 4: Create** `conf/data/capetown_full.yaml`:

```yaml
# Real full Cape Town data: downloaded on demand, cached under ~/.cache/reblock (never committed).
_target_: reblock.data.provision.cached_kblock_source
city: capetown
block_ids: ${block_ids}
```

- [ ] **Step 5: Run tests** → PASS. **Step 6: `pixi run check`** → green. **Commit:**

```bash
git add src/reblock/data/provision.py conf/data/capetown_full.yaml tests/data/test_provision.py
git commit -m "feat: on-demand cached full-city data provisioning (~/.cache/reblock) + capetown_full"
```

---

### Task 4: The `reblock.screen` detect entrypoint + recipe

**Files:**
- Create: `src/reblock/screen/__main__.py`
- Create: `conf/screen_config.yaml`
- Modify: `README.md`
- Test: `tests/screen/test_screen_app.py`

**Interfaces:** Consumes `ensure_city_data` (Task 3), the `Screen` config group (Task 2). `python -m reblock.screen` runs `src/reblock/screen/__main__.py`.

- [ ] **Step 1: Write a failing test** in `tests/screen/test_screen_app.py` (drive the app's core through a pre-seeded cache, no network):

```python
import shutil
from pathlib import Path

from hydra import compose, initialize

from reblock.screen.__main__ import detect

ROOT = Path(__file__).resolve().parents[1]


def _seed(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for name in ("blocks_capetown", "buildings_capetown"):
        shutil.copy(ROOT / "data" / "kblock" / f"{name}_sample.parquet",
                    cache / f"{name}_full.parquet")


def test_detect_flags_flagship(tmp_path: Path) -> None:
    _seed(tmp_path)
    with initialize(version_base=None, config_path="../../conf"):
        cfg = compose(config_name="screen_config",
                      overrides=["screen=dense_compact", "screen.density_min=80"])  # fast: densest only
    ids = detect(cfg, cache_dir=tmp_path)
    assert "ZAF.9.3.1_1_44882" in ids and ids == sorted(ids)
```

- [ ] **Step 2: Run to verify fail** → import error.

- [ ] **Step 3: Implement** `src/reblock/screen/__main__.py` (note `config_path="../../../conf"` — this file is one level deeper than `run.py`):

```python
"""reblock.screen: run the selected Screen on a city's data and print the flagged block_ids.
Interim standalone app (the flow-refactor folds Screen into run() as a stage; see that spec).
"""
from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from reblock.data.provision import DEFAULT_CACHE, ensure_city_data


def detect(cfg: DictConfig, *, cache_dir: Path = DEFAULT_CACHE) -> list[str]:
    blocks_path, buildings_path = ensure_city_data(cfg.city, cache_dir=cache_dir)
    screen = instantiate(cfg.screen, blocks_path=str(blocks_path),
                         buildings_path=str(buildings_path))
    ids: list[str] = screen.select()
    return ids


@hydra.main(version_base=None, config_path="../../../conf", config_name="screen_config")
def main(cfg: DictConfig) -> None:
    ids = detect(cfg)
    print(f"{len(ids)} informal blocks flagged")
    for bid in ids:
        print(bid)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create** `conf/screen_config.yaml`:

```yaml
defaults:
  - screen: dense_compact
  - _self_

# Which city's data to screen (downloaded on demand + cached; see reblock.data.provision).
city: capetown
```

- [ ] **Step 5: Add the README recipe.** In `README.md`, after the single-block visuals section:

````markdown
## Detect informal settlements (Screen)

Flag the dense/compact informal blocks in a city — the settlement blocks worth reblocking:

```bash
pixi run python -m reblock.screen screen=dense_compact city=capetown
```

First run downloads + caches the full Cape Town data under `~/.cache/reblock` (nothing is
committed); later runs are instant. Prints the flagged `block_ids`. Tune the thresholds, e.g.
`screen.density_min=50 screen.mean_depth_min=1.5`.
````

- [ ] **Step 6: Run tests** → `pixi run pytest tests/screen/test_screen_app.py -v` → PASS.

- [ ] **Step 7: `pixi run check`** → green. **Commit:**

```bash
git add src/reblock/screen/__main__.py conf/screen_config.yaml README.md tests/screen/test_screen_app.py
git commit -m "feat: reblock.screen detect entrypoint + recipe (interim, superseded by flow-refactor)"
```

---

## Notes for the executor

- **`src/reblock/screen/` is a package** (`__init__.py`, `dense_compact.py`, `__main__.py`) — NOT a `screen.py` module, so `python -m reblock.screen` runs `__main__.py` and `DenseCompactScreen` lives at `reblock.screen.dense_compact` with no module/package collision.
- **`mypy --strict`** on `instantiate(...)` returns `Any`; annotate the local (`ids: list[str] = screen.select()`) and add a minimal `cast` only where a stub forces it — document each adjustment.
- **`from scripts.fetch_kblock_fixtures import ...`** (Task 3) resolves because `pyproject.toml` already sets `pythonpath=["."]` for pytest; `provision.py` importing it at module load is fine under `mypy --strict` (the script is typed). If mypy can't resolve `scripts`, add it to the mypy `files`/`mypy_path` the same way the cross-block probe did — note it.
- **Task 1's regen reads `outputs/kblock_raw/ZAF_geodata.parquet`** (cached this session, git-ignored); if absent it re-downloads (873 MB). It is present in this workspace.
- **Determinism:** no RNG anywhere; all Screen output sorted.
- Out of scope (do NOT add): the relaxed `merge_cluster`, reblock-a-settlement recipe, per-block scores, `IdentityScreen`/combinator Screens, region rendering, external validation, the run()-stage unification (all S2 / flow-refactor).
