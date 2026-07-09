# Flow-refactor F3 — Screen stage + city flagged-map + L1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fold the Screen into `reblock.run` as a stage (default `IdentityScreen`), emit a city flagged-map, add a lightweight in-process L1 cache, and delete the standalone `reblock.screen` app — so one command does detect → reblock → render + city map.

**Architecture:** `Screen.select(source) -> list[str] | None` (None = all). `run.main` orchestrates the impure edges — instantiate the Source, `screen.select(source)`, set `cfg.block_ids` to the selection, call pure `run(cfg)`, then run the enabled emitters (`render` + the new `flagged_map` + a `flagged_blocks.txt` write). `run()` itself stays the pure reblock core (unchanged). An in-process L1 dict above the F2 joblib L2 makes the screen→reblock double-build and the flagged-map read free memory hits.

**Tech Stack:** Python 3.12, geopandas/shapely 2.1, joblib, matplotlib, Hydra (`_target_` + `instantiate` + interpolation), pixi, pytest, `mypy --strict`, ruff.

## Global Constraints

- `pixi run check` stays green — `ruff check` + `ruff format --check` + `mypy --strict src tests scripts/crossblock_probe.py` + `pytest`. Suite is currently 119 tests.
- **No dual path / no compat shim** (owner directive): the standalone `reblock.screen` app is **deleted**, not deprecated; its `flagged_blocks.txt`/detect logic migrates into `run.main`. The `Screen.select()` signature is changed outright (no overload).
- **`run(cfg) -> list[Result]` is UNCHANGED** — pure, no screen inside it, no return-type change. The screen stage + emitters live in `run.main` (the Hydra entrypoint).
- **`Screen.select(source: Source) -> list[str] | None`** — `None` means "all blocks" (feeds `block_ids=None`); a list means exactly those.
- **Flagged-map uses the screen's FULL selection** (not the `max_blocks`-limited results) and re-reads the blocks parquet geometry; kept out of the Screen so the Screen stays a pure `Source → block_ids` selector.
- **L1 keys on the SAME content-address** the F2 L2 wrappers compute; empty `source_content_hash` bypasses both L1 and L2. L1 is cleared per-test (conftest) so the suite stays hermetic.
- **Both visualizations opt-in** via their config `enabled` flags; both write into the Hydra run dir.
- Commit trailers on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
  ```

---

### Task 1: `Screen.select(source)` protocol + `IdentityScreen`

**Files:**
- Modify: `src/reblock/contracts.py` (the `Screen` protocol)
- Create: `src/reblock/screen/identity.py`
- Create: `conf/screen/identity.yaml`
- Test: `tests/screen/test_identity.py`

**Interfaces:**
- Consumes: `reblock.contracts.Source`.
- Produces:
  - `Screen.select(self, source: Source) -> list[str] | None`.
  - `IdentityScreen(block_ids: list[str] | None = None)`; `select(source)` returns `self.block_ids` (ignores `source`).

- [ ] **Step 1: Write the failing test**

Create `tests/screen/test_identity.py`:

```python
from reblock.screen.identity import IdentityScreen


class _StubSource:
    def region(self):  # satisfies Source structurally; unused by IdentityScreen
        raise NotImplementedError


def test_identity_passthrough_returns_configured_block_ids() -> None:
    assert IdentityScreen(["a", "b"]).select(_StubSource()) == ["a", "b"]


def test_identity_default_is_none_meaning_all() -> None:
    assert IdentityScreen().select(_StubSource()) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/screen/test_identity.py -v`
Expected: FAIL — `No module named 'reblock.screen.identity'`.

- [ ] **Step 3: Change the `Screen` protocol**

In `src/reblock/contracts.py`, change the `Screen` protocol (currently `def select(self) -> list[str]: ...`) to:

```python
class Screen(Protocol):
    def select(self, source: Source) -> list[str] | None: ...   # block_ids, or None => all
```

(`Source` is already defined in this module.)

- [ ] **Step 4: Implement `IdentityScreen`**

Create `src/reblock/screen/identity.py`:

```python
"""IdentityScreen: the passthrough Screen (run()'s default). Selects nothing of
its own -- returns the configured block_ids (or None => all blocks), so a run
with no real screen behaves exactly as a plain reblock."""
from __future__ import annotations

from reblock.contracts import Source


class IdentityScreen:
    def __init__(self, block_ids: list[str] | None = None) -> None:
        self.block_ids = list(block_ids) if block_ids is not None else None

    def select(self, source: Source) -> list[str] | None:
        del source   # a passthrough needs no data
        return list(self.block_ids) if self.block_ids is not None else None
```

Create `conf/screen/identity.yaml`:

```yaml
# The passthrough Screen (run()'s default): selects the configured block_ids
# (or all blocks). block_ids is injected from the top-level ${block_ids}.
_target_: reblock.screen.identity.IdentityScreen
block_ids: ${block_ids}
```

- [ ] **Step 5: Run to verify pass**

Run: `pixi run pytest tests/screen/test_identity.py -v`
Expected: PASS (2 tests). Then `pixi run check` — note `DenseCompactScreen.select()` still has the OLD signature (no `source` param) and now violates the `Screen` protocol; if mypy flags it, that's expected and fixed in Task 2 (the two tasks land together in the branch; if you need this task green in isolation, Task 2 immediately follows). If `pixi run check` fails ONLY on `DenseCompactScreen`/`test_dense_compact` signature mismatch, proceed to Task 2 before running the full suite; commit this task's files now.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/contracts.py src/reblock/screen/identity.py conf/screen/identity.yaml tests/screen/test_identity.py
git commit -m "$(cat <<'EOF'
feat: Screen.select(source) protocol + IdentityScreen passthrough (F3)

Screen.select now takes the Source and returns list[str] | None (None => all).
IdentityScreen is the default passthrough returning its configured block_ids.
DenseCompactScreen is migrated to the new signature in the next task.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

### Task 2: `DenseCompactScreen.select(source)` migration

**Files:**
- Modify: `src/reblock/screen/dense_compact.py`
- Modify: `conf/screen/dense_compact.yaml` (drop the path note)
- Test: `tests/screen/test_dense_compact.py`

**Interfaces:**
- Consumes: `reblock.contracts.Source`; `KblockSource` exposes `.blocks_path`/`.buildings_path`.
- Produces: `DenseCompactScreen(*, density_min=30.0, mean_depth_min=1.3, k_min=None, min_buildings=10)` (NO path args); `select(source)` reads `source.blocks_path`/`source.buildings_path`.

- [ ] **Step 1: Update the tests to the new signature**

In `tests/screen/test_dense_compact.py`, the tests currently construct `DenseCompactScreen(blocks_path, buildings_path, ...)` and call `.select()`. Migrate them to construct `DenseCompactScreen(density_min=..., mean_depth_min=...)` (thresholds only) and call `.select(source)` where `source = KblockSource(blocks_path, buildings_path, region_id="test")`. Example shape for the real-fixture test:

```python
def test_select_flags_flagship_on_capetown_fixture() -> None:
    from reblock.data.kblock import KblockSource
    src = KblockSource(CT_BLOCKS, CT_BUILDINGS, region_id="capetown")   # existing fixture consts
    screen = DenseCompactScreen(density_min=35.0, mean_depth_min=1.3)
    ids = screen.select(src)
    assert ids is not None and "ZAF.9.3.1_1_44882" in ids and ids == sorted(ids)
```

Apply the same `select(source)` change to the synthetic-geometry tests in that file (they construct a small on-disk parquet + a `KblockSource` over it, then `screen.select(src)`). Keep every existing assertion (survivor set, mean-depth gate) — only the construction/call shape changes.

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/screen/test_dense_compact.py -v`
Expected: FAIL — `DenseCompactScreen.__init__` still requires positional `blocks_path` (or `select()` takes no `source`).

- [ ] **Step 3: Migrate `DenseCompactScreen`**

Rewrite `src/reblock/screen/dense_compact.py` so `__init__` takes thresholds only and `select(source)` reads the paths off the source:

```python
"""DenseCompactScreen: flag dense/compact informal blocks. Cheap pass = vectorized
density (+ optional k) gate over free kblock columns; fine pass = build only survivors
(reusing the source's KblockSource paths) and keep those whose mean parcel
access-depth clears mean_depth_min.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd

from reblock.contracts import Source
from reblock.data.kblock import KblockSource
from reblock.derive.access import parcel_access_layers

log = logging.getLogger(__name__)


class DenseCompactScreen:
    def __init__(self, *, density_min: float = 30.0, mean_depth_min: float = 1.3,
                 k_min: float | None = None, min_buildings: int = 10) -> None:
        self.density_min = density_min
        self.mean_depth_min = mean_depth_min
        self.k_min = k_min
        self.min_buildings = min_buildings

    def _cheap_survivors(self, blocks: gpd.GeoDataFrame) -> list[str]:
        bid = blocks["block_id"].astype(str)
        density: pd.Series = blocks["building_count"] / (blocks["block_area_m2"] / 1e4)
        mask: pd.Series = density >= self.density_min
        if self.k_min is not None:
            mask = mask & (blocks["k_complexity"] >= self.k_min)
        return sorted(bid[mask.to_numpy()])

    def select(self, source: Source) -> list[str]:
        if not isinstance(source, KblockSource):
            raise TypeError(
                f"DenseCompactScreen needs a KblockSource (kblock columns); got {type(source).__name__}")
        blocks = gpd.read_parquet(
            source.blocks_path,
            columns=["block_id", "k_complexity", "building_count", "block_area_m2", "geometry"])
        survivors = self._cheap_survivors(blocks)
        log.info("cheap pass: %d/%d blocks pass density_min=%.1f%s",
                 len(survivors), len(blocks), self.density_min,
                 f", k_min={self.k_min}" if self.k_min is not None else "")
        if not survivors:
            return []
        log.info("fine pass: building %d survivor blocks (Voronoi + peel) -- the slow step",
                 len(survivors))
        src = KblockSource(source.blocks_path, source.buildings_path, region_id="screen",
                            min_buildings=self.min_buildings, block_ids=survivors)
        kept = [blk.block_id for blk in src.region().blocks
                if float(parcel_access_layers(blk, None).mean()) >= self.mean_depth_min]
        log.info("fine pass: kept %d blocks with mean access-depth >= %.2f",
                 len(kept), self.mean_depth_min)
        return sorted(kept)
```

`select` returns `list[str]` (a subtype of the protocol's `list[str] | None`). Update `conf/screen/dense_compact.yaml`'s comment (drop "Paths are injected by the reblock.screen app" — paths now come from the source at `select()` time); the `_target_` + thresholds stay.

- [ ] **Step 4: Run tests + full check**

Run: `pixi run pytest tests/screen -v` then `pixi run check`
Expected: PASS — `DenseCompactScreen` now satisfies the `Screen` protocol (`select(source)`), the migrated fixture/synthetic tests pass, mypy `--strict` is clean.

- [ ] **Step 5: Commit**

```bash
git add src/reblock/screen/dense_compact.py conf/screen/dense_compact.yaml tests/screen/test_dense_compact.py
git commit -m "$(cat <<'EOF'
refactor: DenseCompactScreen.select(source) reads paths from the source (F3)

Drop the blocks_path/buildings_path constructor args (was injected by the
standalone app); select(source) reads source.blocks_path/buildings_path off a
KblockSource (fail loud otherwise). Thresholds-only constructor.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

### Task 3: Lightweight L1 in-process cache

**Files:**
- Modify: `src/reblock/cache.py` (L1 dict + `clear_l1`; the four wrappers check L1 before L2)
- Modify: `tests/conftest.py` (clear L1 per test)
- Test: `tests/test_cache.py`

**Interfaces:**
- Produces: `clear_l1() -> None`; an internal `_L1: dict[tuple[str, ...], Any]`. Each `cached_*` wrapper checks L1 (keyed on the same content-address tuple it already builds for L2), returns the memory hit if present, else computes via L2 and stores in L1.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cache.py` (an L1 hit is proven by clearing the L2 `memory` store and showing the wrapper still returns without recompute):

```python
def test_l1_serves_after_l2_store_cleared(tmp_path, monkeypatch) -> None:
    import joblib
    monkeypatch.setattr(cache, "memory", joblib.Memory(location=str(tmp_path), verbose=0))
    monkeypatch.setattr(cache, "_access_impl_cached",
                        cache.cached(cache._access_impl, ignore=["block", "roads"]))
    cache.clear_l1()
    calls = {"n": 0}
    real = cache.parcel_access_layers

    def spy(block, roads=None, **kw):
        calls["n"] += 1
        return real(block, roads)
    monkeypatch.setattr(cache, "parcel_access_layers", spy)
    monkeypatch.setattr(cache, "_access_impl_cached",
                        cache.cached(cache._access_impl, ignore=["block", "roads"]))

    block = _grid_block("deadbeef")
    cache.cached_access_layers(block, None, "__before__")   # miss -> compute (n=1), stores L1+L2
    cache.memory.clear(warn=False)                          # wipe L2 disk store
    cache.cached_access_layers(block, None, "__before__")   # L1 hit -> NO recompute
    assert calls["n"] == 1


def test_clear_l1_forces_recompute(tmp_path, monkeypatch) -> None:
    import joblib
    monkeypatch.setattr(cache, "memory", joblib.Memory(location=str(tmp_path), verbose=0))
    monkeypatch.setattr(cache, "_access_impl_cached",
                        cache.cached(cache._access_impl, ignore=["block", "roads"]))
    cache.clear_l1()
    block = _grid_block(cache.SOURCE_HASH_UNSET)   # bypass -> never stored in L1
    cache.cached_access_layers(block, None, "__before__")
    assert not cache._L1   # empty-hash bypass populates neither L1 nor L2
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_cache.py -k "l1 or clear_l1" -v`
Expected: FAIL — `cache.clear_l1` / `cache._L1` not defined.

- [ ] **Step 3: Implement L1 in `cache.py`**

Add a module-level L1 and route each wrapper through it. Add near the top (after `memory`):

```python
_L1: dict[tuple[str, ...], Any] = {}


def clear_l1() -> None:
    """Drop the in-process L1 cache (call between independent runs/tests)."""
    _L1.clear()
```

Refactor each of the four wrappers to check L1 first. The pattern (shown for `cached_access_layers`; apply the same to `cached_geometric`, `cached_voronoi_parcels`, `cached_propose`, using each one's existing key components):

```python
def cached_access_layers(block: Block, roads: GeoDataFrame | None, roads_key: str) -> pd.Series:
    if block.source_content_hash == SOURCE_HASH_UNSET:
        return parcel_access_layers(block, roads)
    geos, proj, code = key_parts()
    k = ("access", block.block_id, block.source_content_hash, geos, proj, code, roads_key)
    if k in _L1:
        return cast("pd.Series", _L1[k])
    out = _access_impl_cached(block, roads, block_id=block.block_id,
                              src_hash=block.source_content_hash, geos=geos, proj=proj,
                              code=code, roads_key=roads_key)
    _L1[k] = out
    return out
```

The L1 key MUST be distinct per wrapper (prefix `"access"`/`"geometric"`/`"voronoi"`/`"propose"`) and include each wrapper's full content-address (e.g. `cached_propose`'s key includes `repr(method)`; `cached_voronoi_parcels`'s includes `block_id` + `source_content_hash` only, matching its L2 key). Keep the empty-hash bypass BEFORE the L1 check so uncacheable blocks touch neither layer.

- [ ] **Step 4: Clear L1 per test in conftest**

In `tests/conftest.py`, add a function-scoped autouse fixture so L1 never leaks across tests (the existing session fixture isolates L2; L1 needs per-test reset because it's keyed on content-address, not on the store location):

```python
@pytest.fixture(autouse=True)
def _clear_l1() -> Iterator[None]:
    cache.clear_l1()
    yield
    cache.clear_l1()
```

- [ ] **Step 5: Run tests + full check**

Run: `pixi run pytest tests/test_cache.py -v` then `pixi run check`
Expected: PASS. The existing L2 spy tests still pass — they assert "underlying computed once", which an L1 hit also satisfies; the per-test `clear_l1` keeps them independent. 119+ tests green.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/cache.py tests/conftest.py tests/test_cache.py
git commit -m "$(cat <<'EOF'
feat: L1 in-process cache above the joblib L2 (F3)

Each cached wrapper checks an in-process dict (keyed on the same content
address as L2) before the joblib disk cache -> the screen->reblock double-build
and the flagged-map read become memory hits (no pickle round-trip). Empty-hash
bypass skips both layers; conftest clears L1 per test for hermeticity.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

### Task 4: Screen stage + `flagged_map` emitter in `run.main`; delete the standalone app

**Files:**
- Modify: `src/reblock/emit.py` (add `FlaggedMapConfig` + `flagged_map`)
- Modify: `src/reblock/run.py` (`main` orchestrates screen → run → emitters + `flagged_blocks.txt`)
- Modify: `conf/config.yaml` (add `screen: identity` default + `flagged_map` block)
- Delete: `src/reblock/screen/__main__.py`, `conf/screen_config.yaml`, `tests/screen/test_screen_app.py`
- Modify: `README.md` (one-command recipe)
- Test: `tests/test_run.py` (screen-stage end-to-end + flagged_map)

**Interfaces:**
- Consumes: `Screen`, `Source`, `render_results`, `RenderConfig`, `IdentityScreen`/`DenseCompactScreen` via `cfg.screen`.
- Produces: `FlaggedMapConfig(enabled=False)`; `flagged_map(blocks_path, flagged_ids, out_dir, cfg) -> Path | None`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_run.py`, add an end-to-end screen-stage test (uses the committed Cape Town fixture; `screen=dense_compact` at a fixture-appropriate `density_min`, `flagged_map.enabled=true`, `render.enabled=true`, `max_blocks=1`) driving the real `@hydra.main` via subprocess, asserting: exit 0; `flagged_map.png` written to the run dir; a `flagged_blocks.txt` written with the flagship id; a reblock `*_before.png`/`*_after.png` for the one reblocked block:

```python
def test_cli_screen_stage_end_to_end(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "reblock.run",
         "data=capetown", "screen=dense_compact", "screen.density_min=35",
         "method=peel", "eval=kcomplexity", "max_blocks=1",
         "render.enabled=true", "flagged_map.enabled=true",
         f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "flagged_map.png").stat().st_size > 0
    flagged = (tmp_path / "flagged_blocks.txt").read_text()
    assert "ZAF.9.3.1_1_44882" in flagged
    assert list(tmp_path.glob("*_before.png")) and list(tmp_path.glob("*_after.png"))


def test_cli_identity_screen_default_passthrough(tmp_path: Path) -> None:
    # Default screen=identity: block_ids drives selection exactly as before.
    result = subprocess.run(
        [sys.executable, "-m", "reblock.run",
         "data=capetown", "method=peel", "eval=kcomplexity",
         "block_ids=[ZAF.9.3.1_1_44882]", "max_blocks=10",
         f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ZAF.9.3.1_1_44882" in result.stdout
```

Also add a focused `flagged_map` emitter unit test:

```python
def test_flagged_map_writes_png(tmp_path: Path) -> None:
    from reblock.emit import FlaggedMapConfig, flagged_map
    out = flagged_map("tests/data/kblock/blocks_capetown_sample.parquet",
                      ["ZAF.9.3.1_1_44882"], tmp_path, FlaggedMapConfig(enabled=True))
    assert out is not None and out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_run.py -k "screen_stage or flagged_map or identity_screen" -v`
Expected: FAIL — `cfg.screen`/`cfg.flagged_map` not in config; `flagged_map` not defined.

- [ ] **Step 3: Add the `flagged_map` emitter to `emit.py`**

```python
@dataclass
class FlaggedMapConfig:
    enabled: bool = False


def flagged_map(blocks_path: str, flagged_ids: list[str], out_dir: Path,
                cfg: FlaggedMapConfig) -> Path | None:
    """Binary city choropleth: all metro blocks drawn light, the flagged ones
    highlighted. Re-reads the blocks parquet geometry (kept out of the Screen so
    it stays a pure selector). Returns the written path, or None if no ids."""
    import geopandas as gpd
    if not flagged_ids:
        return None
    blocks = gpd.read_parquet(blocks_path, columns=["block_id", "geometry"])
    blocks["block_id"] = blocks["block_id"].astype(str)
    blocks["flagged"] = blocks["block_id"].isin(set(flagged_ids))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    blocks[~blocks["flagged"]].plot(ax=ax, color="#e8e8e8", edgecolor="none")
    blocks[blocks["flagged"]].plot(ax=ax, color="#c0392b", edgecolor="none")
    ax.set_title(f"{len(flagged_ids)} flagged blocks")
    ax.set_axis_off()
    out_path = out_dir / "flagged_map.png"
    save_render(fig, out_path)   # reuse the shared savefig helper
    plt.close(fig)
    return out_path
```

(Add `from reblock.render import save_render` — already imported for the render emitter.)

- [ ] **Step 4: Wire the screen stage + emitters into `run.main`**

Rewrite `main` in `src/reblock/run.py` (keep `run()` itself unchanged):

```python
from reblock.contracts import Screen  # add
from reblock.emit import FlaggedMapConfig, RenderConfig, flagged_map, render_results  # widen import


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    source = instantiate(cfg.data)
    screen = cast(Screen, instantiate(cfg.screen))
    selected = screen.select(source)
    if selected is not None:
        cfg.block_ids = list(selected)   # run()'s source picks this up via ${block_ids}
    results = run(cfg)
    for r in results:
        log.info("%s %s", r.block.block_id, {m.eval: dict(m.values) for m in r.metrics})

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    if selected is not None:
        flagged_path = out_dir / "flagged_blocks.txt"
        flagged_path.write_text("".join(f"{b}\n" for b in selected))
        log.info("%d blocks flagged -> %s", len(selected), flagged_path)
    if cfg.render.enabled:
        render_results(results, out_dir, cfg.render)
    if cfg.flagged_map.enabled:
        blocks_path = getattr(source, "blocks_path", None)
        if blocks_path is None:
            log.warning("flagged_map: source %s has no blocks_path; skipping", type(source).__name__)
        else:
            flagged_map(str(blocks_path), selected or [], out_dir, cfg.flagged_map)
```

(Setting `cfg.block_ids` requires the key to exist and be mutable — it does, see Step 5. `selected or []` passes the full flagged set to the map; for IdentityScreen-`None`, the map is skipped since there's no explicit flagged set.)

- [ ] **Step 5: Update `conf/config.yaml`**

Add `screen: identity` to the `defaults` list and a `flagged_map` block. The defaults become:

```yaml
defaults:
  - data: phule
  - screen: identity
  - method: topology
  - eval: kcomplexity
  - _self_
```

And add after the `render:` block:

```yaml
# City flagged-map emitter (reblock.emit.flagged_map): opt-in; draws all metro
# blocks light with the screen's flagged blocks highlighted, to the run dir.
flagged_map:
  enabled: false
```

(`block_ids: null` already exists at top level and is what `main` reassigns.)

- [ ] **Step 6: Delete the standalone app + migrate the README**

```bash
git rm src/reblock/screen/__main__.py conf/screen_config.yaml tests/screen/test_screen_app.py
```

In `README.md`: replace the two-step "Detect informal settlements (Screen)" recipe (which ran `python -m reblock.screen`) with the single end-to-end command, and note it emits both visuals:

```bash
pixi run python -m reblock.run data=capetown_full screen=dense_compact \
  method=peel eval=kcomplexity render.enabled=true flagged_map.enabled=true max_blocks=5
```

Note in the README that this writes `flagged_map.png` (city map), `flagged_blocks.txt`, and per-block `*_before.png`/`*_after.png` into the Hydra run dir; the default `screen=identity` leaves behaviour unchanged for a plain reblock.

- [ ] **Step 7: Run tests + full check**

Run: `pixi run check`
Expected: PASS — the screen-stage end-to-end test writes `flagged_map.png` + `flagged_blocks.txt` + reblock PNGs; the identity-default test reblocks the flagship; the deleted app leaves no dangling import (grep `reblock.screen.__main__` / `screen_config` → none). Note: `python -m reblock.screen` no longer exists (deleted); confirm nothing references it.

- [ ] **Step 8: Commit**

```bash
git add src/reblock/emit.py src/reblock/run.py conf/config.yaml README.md tests/test_run.py
git commit -m "$(cat <<'EOF'
feat: screen stage + city flagged-map in reblock.run; delete standalone app (F3)

run.main orchestrates: instantiate source -> screen.select(source) -> set
cfg.block_ids -> run() -> emit render + flagged_map + flagged_blocks.txt.
Default screen=identity is a passthrough. flagged_map emitter draws the city
choropleth of flagged blocks. Deletes the standalone reblock.screen app and its
screen_config; README collapses to one detect->reblock->render+map command.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

## Self-Review

**Spec coverage (F3 slice, per the updated §"Slicing & sequencing"):**
- Screen protocol `-> list[str] | None` → Task 1. ✓
- `IdentityScreen` default passthrough → Task 1 (+ config default in Task 4). ✓
- Injection (Source → Screen) → refined to `select(source)` (Task 1 protocol, Task 2 DenseCompact reads `source.blocks_path`). ✓
- Screen stage in the run entrypoint (`run.main`, keeping `run()` pure) → Task 4. ✓
- `flagged_map` emitter re-reads the parquet → Task 4. ✓
- Lightweight L1 above L2 → Task 3. ✓
- Delete the standalone app; migrate `flagged_blocks.txt` → Task 4. ✓
- One-command end-to-end + README → Task 4. ✓
- Out of F3 (F4): the emitter registry + `reblock.compare` + scorecard.

**Placeholder scan:** every code step is complete; the Task-1 note about `pixi run check` possibly failing on `DenseCompactScreen` until Task 2 is a real cross-task ordering fact (the two land together on the branch), not a placeholder — Task 1 commits its own files and Task 2 immediately restores protocol conformance.

**Type consistency:** `Screen.select(source) -> list[str] | None` is used identically in `IdentityScreen` (Task 1), `DenseCompactScreen` (Task 2, returns `list[str]`), and `main` (Task 4, `selected: list[str] | None`). `FlaggedMapConfig`/`flagged_map(blocks_path, flagged_ids, out_dir, cfg)` match between Task 4's definition and call site. `clear_l1`/`_L1` (Task 3) are used consistently in the wrappers and conftest. `cfg.block_ids` reassignment (Task 4) targets the existing top-level key from `conf/config.yaml`.
