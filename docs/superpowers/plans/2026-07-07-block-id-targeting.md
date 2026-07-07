# Block-id targeting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a run target specific blocks by id (`block_ids=[...]`) so one block can be built/rendered without processing the whole region.

**Architecture:** `block_ids` is a top-level interpolated Hydra scalar (like `${shapefile}`) that flows into `KblockSource`; `region()` filters the blocks frame **before** the sjoin + Voronoi loop, so a targeted run does O(k) work. UTM is estimated from the *full* frame first, so filtering can't shift the CRS. Independent of the flow-refactor — works with the current `run()` render path.

**Tech Stack:** geopandas, Hydra 1.3.2 (`hydra.utils.instantiate`, compose), pytest, mypy --strict.

## Global Constraints

- `pixi run check` (ruff + `mypy --strict src tests` + pytest) green.
- **Additive / surgical.** Touch only: `src/reblock/data/kblock.py`, `conf/config.yaml`, `conf/data/capetown.yaml`, `conf/data/dji.yaml`, `tests/data/test_kblock_source.py`, `tests/test_run.py` (Task 1); `README.md`, `tests/test_run.py` (Task 2). Do NOT change `contracts.py`, `run.py`, `ShapefileSource`, `_voronoi_parcels`, `_blocks_from`, or other config groups.
- **`block_ids=None` (default) is exactly today's behaviour** — all blocks. No existing test may change.
- **Estimate UTM from the full blocks frame, before filtering** — a filtered block must produce the identical geometry/metrics as in a full-region run (the pinned flagship stays peel-k=7).
- **Fail loud** on a requested `block_id` absent from the source (typo guard) — `ValueError` naming the missing id(s). (A matched-but-sparse block below `min_buildings` is still silently skipped, unchanged.)
- Scope `block_ids` to `KblockSource` only; `ShapefileSource`/`phule.yaml` are untouched.

---

### Task 1: `block_ids` on `KblockSource` + config wiring

**Files:**
- Modify: `src/reblock/data/kblock.py` (constructor + `region()`)
- Modify: `conf/config.yaml`, `conf/data/capetown.yaml`, `conf/data/dji.yaml`
- Test: `tests/data/test_kblock_source.py`, `tests/test_run.py`

**Interfaces:**
- Consumes: existing `KblockSource._blocks_from`, `Region`, `parcel_access_layers`, `run()`, `Result.metric`.
- Produces: `KblockSource(..., *, min_buildings=10, block_ids: list[str] | None = None)`; `region()` filters by `block_ids` before build. Config scalar `block_ids` (null default) interpolated into kblock data groups.

- [ ] **Step 1: Write failing unit tests** in `tests/data/test_kblock_source.py`

Add `import pytest` (top of file, with the other imports). `CT_BLOCKS`/`CT_BLD` constants already exist (added in the kblock slice). Append:

```python
def test_block_ids_filters_to_requested_block() -> None:
    src = KblockSource(CT_BLOCKS, CT_BLD, region_id="capetown",
                       block_ids=["ZAF.9.3.1_1_44882"])
    blocks = list(src.region().blocks)
    assert [b.block_id for b in blocks] == ["ZAF.9.3.1_1_44882"]
    # UTM is estimated from the full frame, so filtering to one block can't shift the
    # CRS: the block reproduces its full-region morphology (pinned peel-k == 7).
    assert int(parcel_access_layers(blocks[0], None).max()) == 7


def test_block_ids_selects_exactly_the_listed_blocks() -> None:
    ids = ["ZAF.9.3.1_1_44882", "ZAF.9.3.1_1_44571"]
    src = KblockSource(CT_BLOCKS, CT_BLD, region_id="capetown", block_ids=ids)
    got = [b.block_id for b in src.region().blocks]
    assert got == sorted(ids)   # _blocks_from yields in sorted block_id order


def test_block_ids_unknown_raises() -> None:
    src = KblockSource(CT_BLOCKS, CT_BLD, region_id="capetown", block_ids=["NOPE"])
    with pytest.raises(ValueError, match="NOPE"):
        src.region()
```

- [ ] **Step 2: Run to verify they fail** → `pytest tests/data/test_kblock_source.py -k block_ids -v` → `TypeError` (unexpected `block_ids` kwarg).

- [ ] **Step 3: Implement `block_ids`** in `src/reblock/data/kblock.py`

Constructor — add the keyword-only param and store it:

```python
    def __init__(self, blocks_path: str | Path, buildings_path: str | Path,
                 region_id: str = "kblock", *, min_buildings: int = 10,
                 block_ids: list[str] | None = None) -> None:
        self.blocks_path = Path(blocks_path)
        self.buildings_path = Path(buildings_path)
        self.region_id = region_id
        self.min_buildings = min_buildings
        self.block_ids = list(block_ids) if block_ids is not None else None
```

`region()` — normalize the id column to str, estimate UTM from the **full** frame, then filter:

```python
    def region(self) -> Region:
        blocks = gpd.read_parquet(
            self.blocks_path, columns=["block_id", "k_complexity", "geometry"])
        blocks["block_id"] = blocks["block_id"].astype(str)
        utm = blocks.estimate_utm_crs()   # full frame => CRS is stable under block_ids filtering
        if self.block_ids is not None:
            wanted = {str(b) for b in self.block_ids}
            missing = wanted - set(blocks["block_id"])
            if missing:
                raise ValueError(
                    f"{self.region_id}: block_ids not found in source: {sorted(missing)}")
            blocks = blocks[blocks["block_id"].isin(wanted)]
        bld = gpd.read_parquet(self.buildings_path, columns=["geometry"])
        return Region(region_id=self.region_id, crs=utm,
                      blocks=self._blocks_from(blocks.to_crs(utm), bld.to_crs(utm)))
```

- [ ] **Step 4: Run unit tests** → `pytest tests/data/test_kblock_source.py -k block_ids -v` → PASS. Also run the existing `test_pinned_capetown_block_morphology` to confirm no regression.

- [ ] **Step 5: Wire the config**

`conf/config.yaml` — add the scalar (below `render_dir: null`) and the explicit chdir setting:

```yaml
# Optional list of block ids to build (kblock sources only); null => all blocks.
# Filters at the source before the Voronoi build, so targeting one block is O(1).
block_ids: null

hydra:
  job:
    chdir: false   # keep CWD at the invocation dir so a source's relative fixture paths resolve
```

`conf/data/capetown.yaml` and `conf/data/dji.yaml` — add one line each:

```yaml
block_ids: ${block_ids}
```

- [ ] **Step 6: Write the pipeline integration test** in `tests/test_run.py`

```python
def test_block_ids_targets_one_capetown_block_through_the_pipeline() -> None:
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=[
            "data=capetown", "method=peel", "eval=kcomplexity",
            "block_ids=[ZAF.9.3.1_1_44882]", "max_blocks=10",
        ])
        results = run(cfg)
    # block_ids overrides the coarse max_blocks front-selection: exactly the one block.
    assert [r.block.block_id for r in results] == ["ZAF.9.3.1_1_44882"]
    r = results[0]
    assert r.metric("kcomplexity", "geometric_access_max_m") >= 0.0
    assert r.metric("kcomplexity", "delta_k") > 0   # peel flattens this deep block
```

- [ ] **Step 7: Full check** → `pixi run check` — green. (Confirms the new `hydra:` node + `block_ids` scalar don't break existing `data=phule`/`data=dji` compose/CLI tests.)

- [ ] **Step 8: Commit**

```bash
git add src/reblock/data/kblock.py conf/config.yaml conf/data/capetown.yaml \
        conf/data/dji.yaml tests/data/test_kblock_source.py tests/test_run.py
git commit -m "feat: block_ids targeting on KblockSource (early filter) + Hydra wiring"
```

---

### Task 2: README recipe + CLI end-to-end validation

**Files:**
- Modify: `README.md`
- Test: `tests/test_run.py`

**Interfaces:** Consumes the `block_ids` wiring from Task 1 and the existing `@hydra.main` entrypoint (`reblock.run`) render path (`render_dir` → PNGs under the Hydra output dir).

- [ ] **Step 1: Write the failing CLI validation test** in `tests/test_run.py` (mirrors `test_cli_entrypoint_smoke`, but targets one Cape Town block via `block_ids`):

```python
def test_cli_block_ids_renders_single_capetown_block(tmp_path: Path) -> None:
    # Validates the README recipe end-to-end through the real @hydra.main entrypoint:
    # block_ids builds ONLY the flagship, and render_dir writes its before/after PNGs.
    result = subprocess.run(
        [sys.executable, "-m", "reblock.run",
         "data=capetown", "method=peel", "eval=kcomplexity",
         "block_ids=[ZAF.9.3.1_1_44882]", "render_dir=renders",
         f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ZAF.9.3.1_1_44882" in result.stdout

    befores = list(tmp_path.glob("renders/ZAF.9.3.1_1_44882_before.png"))
    afters = list(tmp_path.glob("renders/ZAF.9.3.1_1_44882_*_after.png"))
    assert len(befores) == 1 and befores[0].stat().st_size > 0
    assert len(afters) >= 1 and afters[0].stat().st_size > 0
```

- [ ] **Step 2: Run it to verify it passes** → `pytest tests/test_run.py -k block_ids_renders -v`. (No new src code needed — Task 1's wiring makes this pass. If it fails on a relative-path/CWD error, confirm `hydra.job.chdir: false` is present in `conf/config.yaml` from Task 1.)

- [ ] **Step 3: Add the README recipe.** Append a section to `README.md` after "Common tasks":

````markdown
## Generate before/after visuals for one block

Render a block's access-depth heatmaps (before, and after a road-building method)
by targeting it with `block_ids` — no need to process the whole region:

```bash
python -m reblock.run data=capetown method=peel eval=kcomplexity \
  "block_ids=[ZAF.9.3.1_1_44882]" render_dir=renders hydra.run.dir=outputs/ct-flagship
```

Writes `outputs/ct-flagship/renders/ZAF.9.3.1_1_44882_before.png` and one
`_<proposal>_after.png`. Swap `data=capetown` → `data=dji`, or `method=peel` →
`method=topology`. Omit `block_ids` to process the first `max_blocks` blocks instead.
````

(Quote `"block_ids=[...]"` so the shell doesn't glob the brackets.)

- [ ] **Step 4: Full check** → `pixi run check` — green.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_run.py
git commit -m "docs: README recipe for single-block visuals + CLI validation test"
```

---

## Notes for the executor

- **`block_ids` list parsing:** in a compose `overrides` list or a subprocess arg list, `"block_ids=[ZAF.9.3.1_1_44882]"` is passed literally (no shell) and Hydra parses it as a 1-element string list — the id's dots/underscores need no quoting *inside* the grammar. In an interactive shell, quote the whole token so bash doesn't glob `[...]`.
- **Why UTM from the full frame (Task 1 Step 3):** `estimate_utm_crs()` picks the zone from the frame's extent; estimating *after* filtering to one block could in principle pick a different zone for a multi-zone region and shift every coordinate. Estimating first guarantees `block_ids=[X]` reproduces X's full-region metrics (the peel-k==7 assertion guards this).
- **Out of scope (flow-refactor slice):** the L2 per-block persistent cache; `block_ids` on `ShapefileSource`; the pure-`run()`/emitter refactor.
