# reblock — Screen layer (slum detection), Slice S1

**Status:** draft for review · **Date:** 2026-07-08 · **Branch:** `screen-slum-detection` (to be cut)

## Why this exists

Everything so far reblocks blocks you hand-pick. To point the pipeline at a city and have it
*find* the informal settlements worth reblocking, it needs a **Screen**: a region-scale selector that
classifies dense/compact informal blocks. This is the general-purpose form of the fetch script's
one-off density criterion, and it's the automatic version of the `block_ids` cluster-selector.

This session prototyped and validated the pieces on the full Cape Town data (83,192 blocks, 2M
buildings): density-gated growth from the flagship yields a coherent 252-block settlement, and a
per-block `mean_depth` cleanly separates a settlement from a thin sliver-chain (settlement median 1.57
vs sliver 1.00; deep-parcel mass 8,267 vs 233). S1 turns those into a first-class layer.

**Slice split (S1 here; S2 next):** S1 = the Screen layer + on-demand cached data provisioning + a
detect recipe. S2 (deferred) = the relaxed `merge_cluster` + a reblock-a-detected-settlement recipe.

## Scope

**In (S1):** a `Screen` protocol + config group; the v1 `DenseCompactScreen` (cheap column gates →
build survivors → `mean_depth` gate); on-demand cached data provisioning (`ensure_city_data` +
`~/.cache/reblock` + a `capetown_full` source config); a thin `reblock.screen` detect entrypoint + its
README recipe; re-generating the committed Cape Town sample fixture to retain `building_count` /
`block_area_m2`.

**Out (deferred):** the relaxed `merge_cluster` + reblock-a-settlement recipe (**S2**); per-block
*scores* on the Screen output (YAGNI until region-building needs them); combinator Screens
(`MinScreen`/`ProductScreen` — future composite-pattern implementations); external validation against
the City's informal-settlement layer; region-building (seed + neighbors); cross-block placement
methods (cross-block Phase 1); rendering the detected settlements.

## Global constraints

- `pixi run check` (ruff + `mypy --strict src tests` + pytest) green.
- **Reuse `KblockSource`** for the fine (build) pass — no duplicate Voronoi/peel.
- **Deterministic:** sorted `block_id` output, no RNG.
- **Downloads → `~/.cache/reblock/`** (XDG user cache, never the repo); check-if-exists-else-download
  (a plain file cache — **not** joblib, which is for derivation memoization). The large real data is
  never committed; only the ~1.1 MB test fixture stays committed.
- **Free density signals:** `density = building_count / (block_area_m2 / 1e4)` and `k_complexity` are
  columns in the kblock geodata — the cheap pass is pure column math (no build, no buildings sjoin).
- Additive: new files + a new `Screen` protocol in `contracts.py` + new config groups; the committed
  Cape Town sample fixture is regenerated (same blocks, two columns added).

---

## 1. The `Screen` protocol + placement

New protocol in `src/reblock/contracts.py`, alongside `Source`/`Method`/`Eval`:

```python
class Screen(Protocol):
    def select(self) -> list[str]: ...   # informal block_ids, sorted
```

- Selectable via a **`conf/screen/`** config group + `_target_`, exactly like methods/evals — pick
  one; combinators are future `Screen` implementations (composite pattern), not a built-in algebra.
- It sits **upstream of the block-build**: its `block_ids` feed `KblockSource(block_ids=…)` (the
  existing knob), so only flagged blocks are ever built/reblocked. The Screen never sees a built
  `Region` — that is what keeps it region-scale. Output is `block_ids` only (loose coupling;
  region-building and reblocking are separate downstream consumers).

## 2. `DenseCompactScreen` (v1) — `src/reblock/screen/dense_compact.py`

```python
class DenseCompactScreen:
    def __init__(self, blocks_path, buildings_path, *,
                 density_min=30.0, mean_depth_min=1.3, k_min=None, min_buildings=10): ...
    def select(self) -> list[str]: ...
```

`select()` runs two internal stages (the two-tier is entirely internal — the interface is just
`raw data → block_ids`):

1. **Cheap pass** (vectorized, no build, no sjoin): read `[block_id, k_complexity, building_count,
   block_area_m2, geometry]` from `blocks_path`; compute `density = building_count / (block_area_m2 /
   1e4)`; keep rows where `density >= density_min` (and, **if `k_min` is set**, `k_complexity >=
   k_min`) → **survivors** (hundreds, out of tens of thousands).
2. **Fine pass** (build only survivors, **reusing `KblockSource`**): `KblockSource(blocks_path,
   buildings_path, block_ids=survivors, min_buildings=min_buildings).region()`; for each built block,
   `mean_depth = parcel_access_layers(block, None).mean()`; keep blocks where `mean_depth >=
   mean_depth_min`.
3. Return the surviving `block_ids`, **sorted**.

Threshold defaults from the prototype (tunable params): `density_min = 30`/ha (flagship = 108) and
`mean_depth_min = 1.3` (settlement median 1.57 vs thread 1.00) — the two signals we validated.
`k_min` is an **optional** extra cheap gate, **off by default** (`None`); density alone prunes the
83k → survivors, and `mean_depth` does the discrimination, so `k_complexity` is an available knob to
calibrate later rather than a required threshold we'd be inventing now. The params are exposed in
`conf/screen/dense_compact.yaml` (`_target_` + thresholds, **without paths** — the entrypoint injects
those, §4).

## 3. On-demand cached data provisioning — `src/reblock/data/provision.py`

```python
def ensure_city_data(city: str, *, cache_dir: Path = Path.home()/".cache"/"reblock",
                     ) -> tuple[Path, Path]:   # (blocks_path, buildings_path)
```

- If the city's `blocks_*.parquet` / `buildings_*.parquet` aren't in `cache_dir`, download them by
  reusing the fetch script's logic (Dataverse ZAF geodata filtered to the metro bbox — **retaining
  `building_count`/`block_area_m2`** — + the Open Buildings tile), cache under `~/.cache/reblock/`,
  and return the paths. First call downloads; every later call hits the cache transparently.
- `cached_kblock_source(city, *, block_ids=None, min_buildings=10) -> KblockSource`: a factory that
  calls `ensure_city_data(city)` then returns `KblockSource(blocks_path, buildings_path,
  region_id=city, block_ids=block_ids, min_buildings=min_buildings)`.
- **`conf/data/capetown_full.yaml`**: `_target_: reblock.data.provision.cached_kblock_source, city:
  capetown` — the recipe/Screen run on the real full data with zero manual download, nothing
  committed. `conf/data/capetown.yaml` stays pointing at the committed sample fixture (the instant,
  offline, out-of-the-box demo + test fixture).

**Fixture regeneration (one-time, no re-download):** the committed
`tests/data/kblock/blocks_capetown_sample.parquet` is regenerated to carry `building_count` /
`block_area_m2` by joining those columns from the already-cached raw ZAF geodata onto the *same*
committed block set (join on `block_id`). No re-selection, no re-download; geometry and pinned
KblockSource values are unchanged (the two columns are additive and unread by `KblockSource`).

## 4. The detect entrypoint + recipe — `src/reblock/screen.py`

A thin `@hydra.main` app mirroring `reblock.run`, driven by **`conf/screen_config.yaml`** (defaults:
`screen: dense_compact`, `city: capetown`) composing the **`conf/screen/dense_compact.yaml`** group.
It resolves the city's data, injects the paths into the configured Screen, runs `select()`, and prints
the flagged `block_ids` + a count.

```python
@hydra.main(version_base=None, config_path="../../conf", config_name="screen_config")
def main(cfg):
    blocks_path, buildings_path = ensure_city_data(cfg.city)
    screen = instantiate(cfg.screen, blocks_path=blocks_path, buildings_path=buildings_path)
    ids = screen.select()
    print(f"{len(ids)} informal blocks flagged"); print("\n".join(ids))
```

README recipe (with `pixi run`, matching the existing convention):

```bash
pixi run python -m reblock.screen screen=dense_compact city=capetown
```

First run downloads + caches the full Cape Town data; subsequent runs are instant. (Rendering the
flagged blocks is out of scope for S1.)

## 5. Testing

- **Cheap gates** (`test_dense_compact.py`): a synthetic blocks `GeoDataFrame` with `building_count` /
  `block_area_m2` / `k_complexity` columns — a mix of dense-and-deep, dense-but-shallow (fine pass
  drops it), and sparse rows → assert the survivor set is exactly the dense rows after the cheap gate,
  and the final set after the fine gate. (Uses a stub/synthetic path or a small on-disk parquet the
  test writes.)
- **Fine gate**: `mean_depth` computed on a synthetic deep block (a grid, mean > threshold) vs a
  shallow strip (mean ≈ 1) → threshold keeps the deep one.
- **Integration** (`test_dense_compact.py`): the regenerated Cape Town sample fixture → the Screen's
  `select()` includes the flagship `ZAF.9.3.1_1_44882` and reproduces the prototype's `mean_depth`
  ordering. (The sample is all-dense, so this checks "flags the settlement core," not "rejects
  sparse" — the synthetic test covers rejection.)
- **Provisioning** (`test_provision.py`): `ensure_city_data` returns the cached paths without
  downloading when the cache is populated (point `cache_dir` at a temp dir pre-seeded with the sample
  fixtures); the download branch is not exercised in CI (network).
- `cached_kblock_source` composes: instantiating `conf/data/capetown_full.yaml` against a pre-seeded
  cache yields a working `KblockSource`.

## 6. Deferred (to S2 / later)

Relaxed `merge_cluster` (tolerate the largest contiguous component / a `MultiPolygon` boundary — the
36/288-blocks-don't-build finding) + the reblock-a-detected-settlement recipe → **S2**. Per-block
scores; combinator Screens; external validation against the City of Cape Town informal-settlement
layer; region-building (seed + adjacency window); rendering detected settlements; cross-block
placement methods.
