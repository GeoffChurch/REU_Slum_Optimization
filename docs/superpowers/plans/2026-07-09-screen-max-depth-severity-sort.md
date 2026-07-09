# Screen — min max-depth gate + severity sort — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `DenseCompactScreen` (1) a `max_depth_min` gate — keep only blocks with at least one parcel at access-depth ≥ N — and (2) a **severity sort**: return survivors ranked by **max access-depth, descending**, so a downstream `max_blocks=N` selects the *worst-access* N blocks instead of the first N alphabetically.

**Architecture:** The fine pass already builds each cheap-survivor block and computes its per-parcel access-depth via `reblock.derivations.access_before(blk)`. Today it uses only `.mean()` for a single gate and returns `sorted(kept)` (alphabetical). Change it to compute each survivor's depth series once, apply the existing mean gate **and** the new optional max gate, retain `(max_depth, block_id)` per survivor, and return block_ids sorted by max_depth descending (ties broken by block_id). This slots into the redesign cleanly: `block_ids` is already a priority-ordered selection, so `sample()` takes the top-N by severity for free.

**Tech Stack:** Python 3.12, geopandas/shapely, Hydra, pixi, pytest, `mypy --strict`, ruff.

## Global Constraints

- `pixi run check` stays green — `ruff check` + `mypy --strict src tests scripts/crossblock_probe.py` + pytest. Suite is currently **139 tests**.
- **Both gates apply when set:** a survivor must clear `mean_depth_min` (existing) AND, if `max_depth_min is not None`, `access_before(blk).max() >= max_depth_min`. `max_depth_min=None` (default) = max gate off, behaviour identical to today except for the sort order.
- **Return order changes** from alphabetical (`sorted(kept)`) to **max access-depth descending, ties by block_id ascending** — deterministic. This changes which blocks a `max_blocks`-limited run reblocks (now the deepest), not the flagged *set*.
- **No new derivation / no behaviour change to the depth computation** — reuse `access_before(blk)` (already cached through `derive()`); `.max()` and `.mean()` are read from the same Series (one derivation call per block).
- Commit trailers on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
  ```

---

### Task 1: `max_depth_min` gate + max-depth-descending sort in `DenseCompactScreen`

**Files:**
- Modify: `src/reblock/screen/dense_compact.py`
- Modify: `conf/screen/dense_compact.yaml`
- Test: `tests/screen/test_dense_compact.py`

**Interfaces:**
- Consumes: `reblock.derivations.access_before(block) -> pd.Series` (per-parcel access-depth), `KblockSource`.
- Produces: `DenseCompactScreen(*, density_min=30.0, mean_depth_min=1.3, max_depth_min: float | None = None, k_min=None, min_buildings=10)`; `select(source) -> list[str]` ordered by max access-depth descending.

- [ ] **Step 1: Write / update the tests**

In `tests/screen/test_dense_compact.py`:

(a) The real-fixture test currently asserts alphabetical order (`ids == sorted(ids)`), which is no longer true — the screen now returns severity-ordered ids. Change it to assert membership only:

```python
def test_select_flags_flagship_on_real_fixture() -> None:
    # density_min=35.0 clears the flagship's real column-based density (~35.6/ha over
    # the free building_count/block_area_m2 columns). Returned order is now max-access-
    # depth descending (not alphabetical), so assert membership, not sort.
    s = DenseCompactScreen(density_min=35.0, mean_depth_min=1.3)
    src = KblockSource(CT_BLOCKS, CT_BLD, region_id="capetown")
    ids = s.select(src)
    assert ids is not None and "ZAF.9.3.1_1_44882" in ids
```

(b) Add a `max_depth_min`-gate test using the existing `_write_synth` fixture (block A = 5×5 grid, ring depths 1/2/3 → max-depth 3, mean ~1.4; block B = strip, all street-fronting → max-depth 1, mean 1.0):

```python
def test_max_depth_min_gate_drops_blocks_without_a_deep_parcel(tmp_path: Path) -> None:
    bp, dp = _write_synth(tmp_path)
    # A: max-depth 3; B: max-depth 1. mean_depth_min=1.0 passes BOTH on the mean gate,
    # so max_depth_min=3 is the deciding gate -> only A (has a parcel at depth 3) survives.
    s = DenseCompactScreen(density_min=50.0, mean_depth_min=1.0, max_depth_min=3.0, min_buildings=10)
    src = KblockSource(bp, dp, region_id="test", min_buildings=10)
    assert s.select(src) == ["A"]
```

(c) Add a severity-sort test with a dedicated fixture where depth order ≠ alphabetical order — a **shallow** block with an early id (`"aaa"`) and a **deep** block with a late id (`"zzz"`), so a correct max-depth-descending sort returns `["zzz", "aaa"]` (the reverse of alphabetical, proving it is not `sorted()`):

```python
def _write_sort_fixture(tmp: Path) -> tuple[str, str]:
    # "aaa": shallow — one row of buildings all fronting the block edge (max-depth 1).
    # "zzz": deep — a 5x5 grid in a compact block (ring depths 1/2/3 -> max-depth 3).
    shallow = box(EX, NY, EX + 30, NY + 2)
    deep = box(EX + 60, NY, EX + 110, NY + 50)
    blocks = gpd.GeoDataFrame({
        "block_id": ["aaa", "zzz"], "k_complexity": [2.0, 3.0],
        "building_count": [15, 25], "block_area_m2": [60.0, 2500.0],
    }, geometry=[shallow, deep], crs=UTM)
    pts = [Point(EX + 1 + 2 * i, NY + row) for i in range(15) for row in (0.5, 1.5)]      # aaa: 2 rows
    pts += [Point(EX + 65 + 10 * i, NY + 5 + 10 * j) for i in range(5) for j in range(5)]  # zzz: 5x5
    bld = gpd.GeoDataFrame(geometry=pts, crs=UTM)
    bp, dp = tmp / "b.parquet", tmp / "d.parquet"
    blocks.to_parquet(bp)
    bld.to_parquet(dp)
    return str(bp), str(dp)


def test_select_ranks_by_max_depth_descending(tmp_path: Path) -> None:
    bp, dp = _write_sort_fixture(tmp_path)
    s = DenseCompactScreen(density_min=50.0, mean_depth_min=1.0, min_buildings=10)
    src = KblockSource(bp, dp, region_id="test", min_buildings=10)
    # deep "zzz" (max-depth 3) outranks shallow "aaa" (max-depth 1) -> reverse-alphabetical,
    # which alphabetical sorted() could never produce -> proves the severity sort.
    assert s.select(src) == ["zzz", "aaa"]
```

Keep the existing `test_cheap_survivors_gate` and `test_select_two_tier_drops_shallow` as-is (the two-tier test keeps a single survivor `["A"]`, unaffected by the sort).

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/screen/test_dense_compact.py -v`
Expected: FAIL — `DenseCompactScreen.__init__` has no `max_depth_min`; the sort/gate tests fail (still alphabetical, no max gate).

- [ ] **Step 3: Implement the gate + sort**

Rewrite `DenseCompactScreen` in `src/reblock/screen/dense_compact.py` — add the `max_depth_min` param and replace the fine-pass tail:

```python
class DenseCompactScreen:
    def __init__(self, *, density_min: float = 30.0, mean_depth_min: float = 1.3,
                 max_depth_min: float | None = None, k_min: float | None = None,
                 min_buildings: int = 10) -> None:
        self.density_min = density_min
        self.mean_depth_min = mean_depth_min
        self.max_depth_min = max_depth_min
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
                f"DenseCompactScreen needs a KblockSource (kblock columns); "
                f"got {type(source).__name__}")
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
        # One access-depth series per block; keep those clearing the mean gate (and the
        # optional max gate), ranked deepest-parcel-first so a downstream max_blocks picks
        # the worst-access blocks rather than an alphabetical slice.
        ranked: list[tuple[float, str]] = []
        for blk in src.region().blocks:
            depths = access_before(blk)
            mean_d, max_d = float(depths.mean()), float(depths.max())
            if mean_d < self.mean_depth_min:
                continue
            if self.max_depth_min is not None and max_d < self.max_depth_min:
                continue
            ranked.append((max_d, blk.block_id))
        ranked.sort(key=lambda r: (-r[0], r[1]))   # max-depth desc; ties by block_id asc
        log.info("fine pass: kept %d blocks (mean-depth >= %.2f%s), ranked by max access-depth",
                 len(ranked), self.mean_depth_min,
                 "" if self.max_depth_min is None else f", max-depth >= {self.max_depth_min:.1f}")
        return [bid for _, bid in ranked]
```

- [ ] **Step 4: Add `max_depth_min` to the screen config**

In `conf/screen/dense_compact.yaml`, add the tunable (keep the rest unchanged):

```yaml
# A Screen (selectable like conf/method, conf/eval). Reads paths from the run's
# Source (a KblockSource) at select() time; the thresholds below are the gates.
_target_: reblock.screen.dense_compact.DenseCompactScreen
density_min: 30.0
mean_depth_min: 1.3
# max_depth_min: keep only blocks with at least one parcel this deep (null = off).
# Survivors are always returned ranked by max access-depth, descending.
max_depth_min: null
# k_min: null   # optional extra cheap gate; off by default (density + mean_depth do the work)
```

- [ ] **Step 5: Run tests + full check**

Run: `pixi run pytest tests/screen/test_dense_compact.py -v` then `pixi run check`
Expected: PASS. The gate test returns `["A"]`, the sort test returns `["zzz", "aaa"]` (reverse-alphabetical → sort proven), the real-fixture test still flags the flagship. `mypy --strict` clean (`max_depth_min: float | None`; `ranked: list[tuple[float, str]]`). 139 + 2 new tests.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/screen/dense_compact.py conf/screen/dense_compact.yaml tests/screen/test_dense_compact.py
git commit -m "$(cat <<'EOF'
feat: DenseCompactScreen max_depth_min gate + max-depth-descending severity sort

The screen now keeps only blocks with at least one parcel at access-depth >=
max_depth_min (optional gate, complements mean_depth_min), and returns survivors
ranked by max access-depth descending instead of alphabetically -- so a downstream
max_blocks selects the worst-access blocks, not an arbitrary alphabetical slice.
One access_before() series per block feeds both gates and the rank key.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

## Self-Review

**Spec coverage:** `max_depth_min` gate (Step 3 `__init__` + the `max_d < self.max_depth_min` guard) ✓; severity sort by max access-depth descending (Step 3 `ranked.sort(key=lambda r: (-r[0], r[1]))`) ✓; config knob (Step 4) ✓; tests for the gate, the sort (order ≠ alphabetical), and preserved flagship membership (Step 1) ✓.

**Placeholder scan:** complete code in every step; concrete synthetic fixtures with known depths (`_write_synth` A/B, `_write_sort_fixture` aaa/zzz). No TBD.

**Type consistency:** `max_depth_min: float | None`; `access_before(blk) -> pd.Series` → `float(depths.mean())`/`float(depths.max())`; `ranked: list[tuple[float, str]]` → `select` returns `list[str]` (matches the `Screen.select -> list[str] | None` protocol). `_cheap_survivors` unchanged.

**Behaviour note:** with `max_depth_min=None` the gate is off and the *set* of survivors is identical to before; only the return order changes (severity, not alphabetical) — which is the intended point (`max_blocks` picks the worst). The real-fixture test asserts membership (order-independent); the sort test pins the new ordering on a fixture where it is distinguishable from `sorted()`.
