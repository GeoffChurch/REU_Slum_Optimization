# Pluggable Block Metric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the per-block scoring metric a Hydra-swappable composable algebra (`depth`, `depth_density`, `density_compactness`), used by the screen, region growth, and map coloring.

**Architecture:** A `BlockMetric` algebra in a new `src/reblock/metric.py` — primitives (`Depth`/`Density`/`Compactness`) and combinators (`Power`/`Product`), each a node exposing `proxy(blocks)->Series` (fast, columns), `fine(depth,count,area,perim)->float` (true, uses peel), `needs_peel`, and `identity` (cache key). A per-metric `Gate` selects. The screen holds the metric; region growth and coloring duck-type it off the screen. This generalizes the already-landed hardcoded-`depth` work (branch `true-depth-everywhere`) into the `depth` preset.

**Tech Stack:** Python 3, Hydra, geopandas/shapely, scipy/networkx, matplotlib, pytest, ruff, mypy --strict, pixi.

## Global Constraints

- **Migrate, never accommodate:** the hardcoded depth path BECOMES the `depth` preset — no dual metric path. The screen's old scalar gate params (`depth_proxy_min`/`mean_depth_min`/`max_depth_min`/`k_min`) are replaced by the metric's `Gate` + a pre-filter knob; delete them, migrate configs/tests. (`_screen_proxy` already deleted.)
- **Continuous colormap only** — no `scheme=`/mapclassify binning.
- **`needs_peel=False` metrics (`density_compactness`) must skip the Voronoi+peel entirely** — assert via a spy/cache in tests.
- **Default `metric=depth`** — preserves the shipped example target. The other two are opt-in via `metric=…`.
- **Mean-vs-max caveat (accepted, spec §5):** today's fine pass drops on *mean* depth and ranks on *max*; a single fine scalar can't carry both. The `depth` preset's `fine` = max peel depth (rank+color), and its `Gate` is `absolute` on that score, calibrated so the flagged count stays near **13,906 of 83,192**. Do NOT byte-reproduce the mean gate; a small shift is deliberate.
- **`ScreenSelectionInput` is the memoized `derive()` key** (source hash + gate params). The metric's `identity` MUST be folded into `.identity` so a metric change busts the cache.
- **Threading:** the metric is instantiated at the config edge and HELD BY the screen (`DenseCompactScreen(metric=…)`, wired via `metric: ${metric}` in `conf/screen/dense_compact.yaml`). Region growth (`build_regions`) and coloring (`run.py`→`region_map`) obtain it by `getattr(screen, "metric", None)`, mirroring the existing `getattr(screen, "selection_depths")` pattern. A screen without a metric (`IdentityScreen`) → region growth uses the existing `_depth_proxy` fallback.
- `pixi run check` (ruff lint + mypy --strict + pytest) green every task; ruff forbids E702 semicolons, E501 >100-char lines, B905 bare `zip`. Use `cast(...)` for narrowing.
- Commit trailers on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
  ```

**Spec:** `docs/superpowers/specs/2026-07-18-pluggable-block-metric-design.md`
**SDD base (pre-Task-1):** `dd9ef74` (branch `true-depth-everywhere`).

---

## File Structure

- `src/reblock/metric.py` — NEW: the `BlockMetric` algebra (`Depth`/`Density`/`Compactness`/`Power`/`Product`) + `Gate`. (Task 1)
- `src/reblock/derivations.py` — `ScreenSelectionInput` carries the metric + folds its identity. (Task 2)
- `src/reblock/screen/dense_compact.py` — `_compute_selection` uses `metric.proxy`/`fine`/`gate`; `DenseCompactScreen(metric=…, proxy_keep_pct=…)`; `selection_scores`. (Task 2)
- `conf/metric/{depth,depth_density,density_compactness}.yaml`, `conf/config.yaml`, `conf/compare_config.yaml`, `conf/screen/dense_compact.yaml`, `src/reblock/run.py`, `src/reblock/compare.py` — config wiring + metric default. (Task 3)
- `src/reblock/pipeline.py` (`_region_score_map`), `src/reblock/emit.py` (`region_map` label), `src/reblock/run.py` (metric label) — region growth + coloring use the metric. (Task 4)
- `examples/multiblock/` — regenerate on `metric=depth`. (Task 5)

---

## Task 1: The `BlockMetric` algebra + `Gate`

**Files:**
- Create: `src/reblock/metric.py`
- Test: `tests/test_metric.py`

**Interfaces:**
- Produces:
  - `class Depth`, `class Density`, `class Compactness` — leaves.
  - `class Power(base: BlockMetric, exp: float)`, `class Product(terms: list[BlockMetric])` — combinators.
  - Every node: `proxy(blocks: GeoDataFrame) -> pd.Series`, `fine(depth, count, area, perim) -> float`, `needs_peel: bool`, `identity` (a hashable tuple), `name: str`.
  - `class Gate(kind: Literal["absolute","percentile"], value: float)` with `keep(scores: Mapping[str,float]) -> set[str]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metric.py`:

```python
import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.metric import Compactness, Density, Depth, Gate, Power, Product

_UTM = CRS.from_epsg(32643)


def _blocks() -> gpd.GeoDataFrame:
    # two unit-ish squares with known n, A, P
    a = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])       # A=4, P=8
    b = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])       # A=1, P=4
    return gpd.GeoDataFrame({"block_id": ["a", "b"], "building_count": [16.0, 4.0],
                             "block_area_m2": [4.0, 1.0]}, geometry=[a, b], crs=_UTM)


def test_primitive_proxy_and_fine_closed_forms() -> None:
    b = _blocks()
    # Depth proxy = sqrt(n*A)/P ; fine = the passed peel depth (columns ignored for the depth factor)
    assert np.allclose(Depth().proxy(b).to_numpy(), [np.sqrt(16 * 4) / 8, np.sqrt(4 * 1) / 4])
    assert Depth().fine(7.0, 16.0, 4.0, 8.0) == 7.0
    assert Depth().needs_peel is True
    # Density = n/A (proxy == fine, closed form, no peel)
    assert np.allclose(Density().proxy(b).to_numpy(), [16 / 4, 4 / 1])
    assert Density().fine(0.0, 16.0, 4.0, 8.0) == 16 / 4
    assert Density().needs_peel is False
    # Compactness = A/P^2
    assert np.allclose(Compactness().proxy(b).to_numpy(), [4 / 64, 1 / 16])
    assert Compactness().fine(0.0, 16.0, 4.0, 8.0) == 4 / 64
    assert Compactness().needs_peel is False


def test_combinators_fold_proxy_fine_and_needs_peel() -> None:
    b = _blocks()
    dd = Product([Depth(), Density()])
    assert np.allclose(dd.proxy(b).to_numpy(),
                       Depth().proxy(b).to_numpy() * Density().proxy(b).to_numpy())
    assert dd.fine(7.0, 16.0, 4.0, 8.0) == 7.0 * (16 / 4)
    assert dd.needs_peel is True                        # OR over children
    dc = Product([Density(), Compactness()])
    assert dc.needs_peel is False                       # no Depth in the tree
    assert dc.fine(0.0, 16.0, 4.0, 8.0) == (16 / 4) * (4 / 64)
    # Power over a SUB-EXPRESSION: sqrt(depth*density)
    root = Power(Product([Depth(), Density()]), 0.5)
    assert root.fine(9.0, 16.0, 4.0, 8.0) == (9.0 * (16 / 4)) ** 0.5
    assert np.allclose(root.proxy(b).to_numpy(), dd.proxy(b).to_numpy() ** 0.5)


def test_identity_distinguishes_expressions() -> None:
    assert Product([Depth(), Density()]).identity != Product([Density(), Compactness()]).identity
    assert Depth().identity == Depth().identity
    assert Power(Depth(), 2.0).identity != Power(Depth(), 3.0).identity


def test_gate_absolute_and_percentile() -> None:
    scores = {"a": 10.0, "b": 5.0, "c": 1.0, "d": 0.5}
    assert Gate("absolute", 5.0).keep(scores) == {"a", "b"}         # >= 5
    assert Gate("percentile", 50.0).keep(scores) == {"a", "b"}      # top 50%
    assert Gate("percentile", 25.0).keep(scores) == {"a"}           # top 25%
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_metric.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reblock.metric'`.

- [ ] **Step 3: Implement `src/reblock/metric.py`**

```python
"""A composable BlockMetric algebra: primitives (Depth/Density/Compactness) and combinators
(Power/Product), each a node exposing `proxy` (fast, from cheap columns), `fine` (true, uses the
peel depth), `needs_peel`, and `identity` (a hashable cache key). A per-metric `Gate` selects.
Only `Depth`'s proxy and fine differ (proxy estimates depth as sqrt(nA)/P; fine uses the real peel);
the geometry primitives are closed forms identical in both. `needs_peel` is an OR over the tree, so
the screen peels iff the expression contains a Depth. See the design spec."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

_Identity = tuple[object, ...]


def _cols(blocks: GeoDataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(count, area, perim) Series from the free kblock columns -- perimeter in the UTM CRS so it's
    metric, area from `block_area_m2` when present else the reprojected geometry area."""
    utm = blocks.to_crs(blocks.estimate_utm_crs())
    count = blocks["building_count"].astype(float)
    area = (blocks["block_area_m2"].astype(float) if "block_area_m2" in blocks.columns
            else utm.geometry.area)
    perim = utm.geometry.length
    return count.reset_index(drop=True), area.reset_index(drop=True), perim.reset_index(drop=True)


@runtime_checkable
class BlockMetric(Protocol):
    name: str
    needs_peel: bool
    def proxy(self, blocks: GeoDataFrame) -> pd.Series: ...
    def fine(self, depth: float, count: float, area: float, perim: float) -> float: ...
    @property
    def identity(self) -> _Identity: ...


@dataclass(frozen=True)
class Depth:
    name: str = "depth"
    needs_peel: bool = True
    def proxy(self, blocks: GeoDataFrame) -> pd.Series:
        count, area, perim = _cols(blocks)
        return np.sqrt(count * area) / perim.where(perim > 0)
    def fine(self, depth: float, count: float, area: float, perim: float) -> float:
        return depth
    @property
    def identity(self) -> _Identity:
        return ("depth",)


@dataclass(frozen=True)
class Density:
    name: str = "density"
    needs_peel: bool = False
    def proxy(self, blocks: GeoDataFrame) -> pd.Series:
        count, area, _ = _cols(blocks)
        return count / area.where(area > 0)
    def fine(self, depth: float, count: float, area: float, perim: float) -> float:
        return count / area if area > 0 else 0.0
    @property
    def identity(self) -> _Identity:
        return ("density",)


@dataclass(frozen=True)
class Compactness:
    name: str = "compactness"
    needs_peel: bool = False
    def proxy(self, blocks: GeoDataFrame) -> pd.Series:
        _, area, perim = _cols(blocks)
        return area / (perim.where(perim > 0) ** 2)
    def fine(self, depth: float, count: float, area: float, perim: float) -> float:
        return area / perim ** 2 if perim > 0 else 0.0
    @property
    def identity(self) -> _Identity:
        return ("compactness",)


@dataclass(frozen=True)
class Power:
    base: BlockMetric
    exp: float
    name: str = "power"
    def proxy(self, blocks: GeoDataFrame) -> pd.Series:
        return self.base.proxy(blocks) ** self.exp
    def fine(self, depth: float, count: float, area: float, perim: float) -> float:
        return self.base.fine(depth, count, area, perim) ** self.exp
    @property
    def needs_peel(self) -> bool:
        return self.base.needs_peel
    @property
    def identity(self) -> _Identity:
        return ("power", self.exp, self.base.identity)


@dataclass(frozen=True)
class Product:
    terms: Sequence[BlockMetric]
    name: str = "product"
    def proxy(self, blocks: GeoDataFrame) -> pd.Series:
        out = self.terms[0].proxy(blocks)
        for t in self.terms[1:]:
            out = out * t.proxy(blocks)
        return out
    def fine(self, depth: float, count: float, area: float, perim: float) -> float:
        out = 1.0
        for t in self.terms:
            out *= t.fine(depth, count, area, perim)
        return out
    @property
    def needs_peel(self) -> bool:
        return any(t.needs_peel for t in self.terms)
    @property
    def identity(self) -> _Identity:
        return ("product", tuple(t.identity for t in self.terms))


@dataclass(frozen=True)
class Gate:
    kind: Literal["absolute", "percentile"]
    value: float
    def keep(self, scores: Mapping[str, float]) -> set[str]:
        """The selected block_ids. `absolute` keeps score >= value; `percentile` keeps the top
        `value`% by score (ties included at the cutoff score)."""
        if not scores:
            return set()
        if self.kind == "absolute":
            return {b for b, s in scores.items() if s >= self.value}
        k = max(1, math.ceil(len(scores) * self.value / 100.0))
        cutoff = sorted(scores.values(), reverse=True)[k - 1]
        return {b for b, s in scores.items() if s >= cutoff}

    @property
    def identity(self) -> _Identity:
        return ("gate", self.kind, self.value)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run pytest tests/test_metric.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Lint + type-check**

Run: `pixi run ruff check src/reblock/metric.py tests/test_metric.py && pixi run mypy --strict src/reblock/metric.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/metric.py tests/test_metric.py
git commit -m "feat: composable BlockMetric algebra (Depth/Density/Compactness + Power/Product) + Gate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

## Task 2: The screen selects by the metric

**Files:**
- Modify: `src/reblock/derivations.py` (`ScreenSelectionInput` carries the metric + identity)
- Modify: `src/reblock/screen/dense_compact.py` (`_compute_selection`, `DenseCompactScreen`, `selection_scores`)
- Test: `tests/screen/test_dense_compact_depths.py` (rename concepts) + `tests/screen/` construction sites

**Interfaces:**
- Consumes: `reblock.metric.BlockMetric`, `Gate` (Task 1); existing `_cheap_survivors`, `_survivor_depths`.
- Produces:
  - `ScreenSelectionInput(... , metric: BlockMetric, proxy_keep_pct: float)` — the memoized derive input; `.identity` folds `metric.identity`.
  - `DenseCompactScreen(metric: BlockMetric, *, proxy_keep_pct: float = 30.0, min_buildings: int = 10)` — `.metric` attribute, `select() -> list[str]`, `selection_scores(source) -> dict[str, float]`.
  - `_compute_selection(inp) -> list[tuple[str, float]]` — `(block_id, fine_score)` deepest/highest-first.

- [ ] **Step 1: Write the failing test**

Rewrite `tests/screen/test_dense_compact_depths.py` (its old name referenced "depths"; keep the file, update the body):

**Gate location — decided:** Task 1's metric nodes are pure scorers (no gate field). The gate is an
explicit `DenseCompactScreen` argument: `DenseCompactScreen(metric, gate, *, proxy_keep_pct,
min_buildings)`. Every construction in Tasks 2–4 + configs uses that signature.

```python
from pathlib import Path

from reblock.data.kblock import KblockSource
from reblock.metric import Compactness, Density, Depth, Gate, Product
from reblock.screen.dense_compact import DenseCompactScreen

_ROOT = Path(__file__).resolve().parent.parent


def _src() -> KblockSource:
    return KblockSource(_ROOT / "data/kblock/blocks_dji_sample.parquet",
                        _ROOT / "data/kblock/buildings_dji_sample.parquet", "dji")


def test_depth_metric_selects_and_scores_by_true_depth() -> None:
    # metric=Depth with a permissive absolute gate: select() returns ids, selection_scores maps them
    # to the fine score (true max peel depth for Depth), and they agree on membership.
    screen = DenseCompactScreen(Depth(), Gate("absolute", 1.0), proxy_keep_pct=100.0,
                                min_buildings=1)
    ids = screen.select(_src())
    scores = screen.selection_scores(_src())
    assert ids and set(scores) == set(ids)
    assert all(s >= 1.0 for s in scores.values())      # gate floor


def test_density_compactness_metric_skips_the_peel() -> None:
    # needs_peel=False -> the fine pass must NOT call _survivor_depths (no Voronoi/peel).
    import reblock.screen.dense_compact as dc
    calls = {"n": 0}
    real = dc._survivor_depths

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    dc._survivor_depths = spy      # type: ignore[assignment]
    try:
        screen = DenseCompactScreen(Product([Density(), Compactness()]),
                                    Gate("percentile", 20.0), min_buildings=1)
        ids = screen.select(_src())
    finally:
        dc._survivor_depths = real   # type: ignore[assignment]
    assert ids                       # still selects
    assert calls["n"] == 0           # peel skipped entirely
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/screen/test_dense_compact_depths.py -v`
Expected: FAIL (DenseCompactScreen signature mismatch / no `selection_scores`).

- [ ] **Step 3: Carry the metric + gate through `ScreenSelectionInput`**

In `src/reblock/derivations.py`, extend `ScreenSelectionInput` (keep it a frozen dataclass): replace
the four scalar gate fields (`depth_proxy_min`, `mean_depth_min`, `max_depth_min`, `k_min`) with the
metric + gate + pre-filter, and fold their identities into `.identity`:

```python
@dataclass(frozen=True)
class ScreenSelectionInput:
    source_hash: str
    blocks_path: str
    buildings_path: str
    metric: "BlockMetric"          # the scorer (carried for _compute_selection; identity below)
    gate: "Gate"                   # the selection gate
    proxy_keep_pct: float          # cheap recall pre-filter: keep top-% by proxy (peel metrics)
    min_buildings: int

    @property
    def identity(self) -> tuple[object, ...] | None:
        return (("dense-compact-screen", self.source_hash, self.metric.identity,
                 self.gate.identity, self.proxy_keep_pct, self.min_buildings)
                if self.source_hash else None)
```

Add `from reblock.metric import BlockMetric, Gate` under `TYPE_CHECKING` (string annotations) or
directly. `screen_selection`/`_screen_selection_impl` return types are unchanged (`list[tuple[str,
float]]`).

- [ ] **Step 4: Rewrite `_compute_selection` to use the metric**

In `src/reblock/screen/dense_compact.py`, replace `_compute_selection` (L97–138) so it: computes
`metric.proxy`, applies the recall pre-filter (peel metrics only), peels only if `needs_peel`, applies
`metric.fine`, then the `gate`, ranked by score. `_cheap_survivors` / `_survivor_depths` stay.

```python
def _compute_selection(inp: ScreenSelectionInput) -> list[tuple[str, float]]:
    """Full screen under the configured metric: proxy over all blocks -> recall pre-filter (peel
    metrics only) -> (peel survivors iff metric.needs_peel) -> metric.fine -> gate -> ranked
    (block_id, fine_score) highest-first. Memoized via screen_selection.identity (metric + gate)."""
    metric, gate = inp.metric, inp.gate
    blocks = gpd.read_parquet(
        inp.blocks_path,
        columns=["block_id", "building_count", "block_area_m2", "geometry"])
    bid = blocks["block_id"].astype(str).to_numpy()
    count = blocks["building_count"].to_numpy(dtype=float)
    utm = blocks.to_crs(blocks.estimate_utm_crs())
    area = (blocks["block_area_m2"].to_numpy(dtype=float) if "block_area_m2" in blocks.columns
            else utm.geometry.area.to_numpy())
    perim = utm.geometry.length.to_numpy()
    eligible = count >= inp.min_buildings
    proxy = metric.proxy(blocks).to_numpy()

    if not metric.needs_peel:
        scores = {bid[i]: metric.fine(0.0, count[i], area[i], perim[i])
                  for i in range(len(bid)) if eligible[i] and np.isfinite(proxy[i])}
    else:
        # recall pre-filter: keep the top proxy_keep_pct% by proxy among eligible blocks, then peel.
        order = [i for i in np.argsort(proxy)[::-1] if eligible[i] and np.isfinite(proxy[i])]
        k = max(1, math.ceil(len(order) * inp.proxy_keep_pct / 100.0))
        survivors = [str(bid[i]) for i in order[:k]]
        idx = {b: i for i, b in enumerate(bid)}
        depth_by = {b: mx for b, mx, _ in                                   # {bid: max_depth}
                    _survivor_depths(survivors, inp.blocks_path, inp.buildings_path,
                                     inp.min_buildings)}
        scores = {b: metric.fine(depth_by.get(b, 0.0), count[idx[b]], area[idx[b]], perim[idx[b]])
                  for b in survivors}

    kept = gate.keep(scores)
    ranked = sorted(((scores[b], b) for b in kept), key=lambda r: (-r[0], r[1]))
    log.info("screen: %d/%d blocks selected by metric=%s (needs_peel=%s)",
             len(ranked), len(bid), metric.name, metric.needs_peel)
    return [(b, s) for s, b in ranked]
```

`_survivor_depths` returns `list[tuple[str, float, float]]` = `(bid, max_d, mean_d)`; take the
`max_d` (`dict((b, mx) for b, mx, _ in ...)`). Add `import math`, keep `numpy as np` (add if absent).
Delete the now-unused `mean_depth_min`/`max_depth_min` drop logging.

- [ ] **Step 5: Update `DenseCompactScreen` (metric + gate + pre-filter; `selection_scores`)**

```python
class DenseCompactScreen:
    def __init__(self, metric: BlockMetric, gate: Gate, *, proxy_keep_pct: float = 30.0,
                 min_buildings: int = 10) -> None:
        self.metric = metric
        self.gate = gate
        self.proxy_keep_pct = proxy_keep_pct
        self.min_buildings = min_buildings

    def _selection_input(self, source: Source) -> ScreenSelectionInput:
        if not isinstance(source, KblockSource):
            raise TypeError(
                f"DenseCompactScreen needs a KblockSource (kblock columns); "
                f"got {type(source).__name__}")
        return ScreenSelectionInput(
            source_hash=source_hash(source.blocks_path, source.buildings_path),
            blocks_path=str(source.blocks_path), buildings_path=str(source.buildings_path),
            metric=self.metric, gate=self.gate, proxy_keep_pct=self.proxy_keep_pct,
            min_buildings=self.min_buildings)

    def select(self, source: Source) -> list[str]:
        return [bid for bid, _ in screen_selection(self._selection_input(source))]

    def selection_scores(self, source: Source) -> dict[str, float]:
        """block_id -> the metric's fine score for the flagged blocks (memoized screen_selection
        lookup) -- what region_map's coloring keys on."""
        return dict(screen_selection(self._selection_input(source)))
```

Add `from reblock.metric import BlockMetric, Gate` to the imports.

- [ ] **Step 6: Migrate existing screen construction sites**

Grep every `DenseCompactScreen(` and `selection_depths` in `tests/` and `src/`:
`grep -rn "DenseCompactScreen(\|selection_depths\|depth_proxy_min\|mean_depth_min" src tests`.
Update each construction to `DenseCompactScreen(Depth(), Gate("absolute", <v>), proxy_keep_pct=<p>,
min_buildings=<m>)` and each `selection_depths` reference to `selection_scores`. (Task 4 handles
`run.py`/`pipeline.py`; this step covers tests + any other src caller.)

- [ ] **Step 7: Run the screen + pipeline + run suites**

Run: `pixi run pytest tests/screen/ tests/test_pipeline.py tests/test_run.py -v`
Expected: PASS (new metric tests + migrated existing tests).

- [ ] **Step 8: Lint + type-check**

Run: `pixi run ruff check src/reblock/derivations.py src/reblock/screen/dense_compact.py tests/screen/ && pixi run mypy --strict src/reblock/derivations.py src/reblock/screen/dense_compact.py`
Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add src/reblock/derivations.py src/reblock/screen/dense_compact.py tests/screen/
git commit -m "feat: screen selects by the configured BlockMetric (proxy pre-filter + fine + gate)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

## Task 3: Config group + `metric=depth` default

**Files:**
- Create: `conf/metric/depth.yaml`, `conf/metric/depth_density.yaml`, `conf/metric/density_compactness.yaml`
- Modify: `conf/config.yaml`, `conf/compare_config.yaml`, `conf/screen/dense_compact.yaml`, `src/reblock/run.py` (`spec_from_cfg`), `src/reblock/compare.py`
- Test: `tests/test_run.py` (a Hydra-compose smoke test)

**Interfaces:**
- Consumes: Task 1 metric classes, Task 2 `DenseCompactScreen(metric, gate, …)`.
- Produces: `metric=depth|depth_density|density_compactness` selectable at the CLI; default `depth`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_run.py`:

```python
def test_metric_config_group_instantiates_each_preset() -> None:
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from reblock.metric import BlockMetric
    from pathlib import Path
    cfgdir = str(Path(__file__).resolve().parent.parent / "conf")
    for name, needs_peel in [("depth", True), ("depth_density", True),
                             ("density_compactness", False)]:
        with initialize_config_dir(version_base=None, config_dir=cfgdir):
            cfg = compose(config_name="config", overrides=[f"metric={name}"])
        metric = instantiate(cfg.metric)
        assert isinstance(metric, BlockMetric) and metric.needs_peel is needs_peel
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_run.py::test_metric_config_group_instantiates_each_preset -v`
Expected: FAIL (no `metric` config group / `cfg.metric`).

- [ ] **Step 3: Create the metric configs**

`conf/metric/depth.yaml`:
```yaml
# The `depth` metric: score = true max access-depth (rings). Proxy = sqrt(nA)/P.
_target_: reblock.metric.Depth
```
`conf/metric/depth_density.yaml`:
```yaml
# Deep AND crowded: depth * density.
_target_: reblock.metric.Product
terms:
  - {_target_: reblock.metric.Depth}
  - {_target_: reblock.metric.Density}
```
`conf/metric/density_compactness.yaml`:
```yaml
# Dense, compact fabric from geometry alone (no peel): density * compactness.
_target_: reblock.metric.Product
terms:
  - {_target_: reblock.metric.Density}
  - {_target_: reblock.metric.Compactness}
```

- [ ] **Step 4: Add the `metric` default + thread the gate/pre-filter**

In `conf/config.yaml` and `conf/compare_config.yaml`, add to `defaults:` (before `_self_`):
```yaml
  - metric: depth
```
Add a top-level gate + pre-filter block to both (the gate lives at the run level so it ports across
metrics; calibrated for `depth` in Task 5):
```yaml
# Screen selection gate (applied to the metric's fine score) + the cheap recall pre-filter
# (keep top-% by proxy before the peel). Calibrated for the default `depth` metric.
metric_gate: {_target_: reblock.metric.Gate, kind: absolute, value: 2.0}
proxy_keep_pct: 30.0
```
In `conf/screen/dense_compact.yaml`, replace the scalar gate fields with metric wiring:
```yaml
_target_: reblock.screen.dense_compact.DenseCompactScreen
metric: ${metric}
gate: ${metric_gate}
proxy_keep_pct: ${proxy_keep_pct}
min_buildings: 10
```

- [ ] **Step 5: Instantiate in the config edges**

`src/reblock/run.py` `spec_from_cfg` and `src/reblock/compare.py` already do
`instantiate(cfg.screen)` — because `conf/screen/dense_compact.yaml` now interpolates `${metric}` /
`${metric_gate}`, the screen is built with them automatically; no code change is needed there IF the
top-level `metric`/`metric_gate` keys exist. Verify: `instantiate(cfg.screen)` returns a
`DenseCompactScreen` with `.metric` set. If Hydra can't resolve the interpolation into the nested
`_target_`, instantiate the metric explicitly in `spec_from_cfg`/`compare` and set it on the screen:
`screen = instantiate(cfg.screen); # metric already wired via ${metric}`. No new code unless the
smoke test in Step 6 fails.

- [ ] **Step 6: Run the config smoke test + full screen/run suites**

Run: `pixi run pytest tests/test_run.py tests/test_pipeline.py tests/screen/ -v`
Expected: PASS. Manually verify one compose:
`pixi run python -c "from hydra import compose, initialize_config_dir; from hydra.utils import instantiate; from pathlib import Path;\nwith initialize_config_dir(version_base=None, config_dir=str(Path('conf').resolve())):\n cfg=compose(config_name='config', overrides=['data=capetown_full','screen=dense_compact','metric=density_compactness']);\n s=instantiate(cfg.screen); print(type(s).__name__, s.metric.name, s.metric.needs_peel)"`
Expected: `DenseCompactScreen product False`.

- [ ] **Step 7: Lint + type-check + commit**

Run: `pixi run ruff check src/reblock/run.py src/reblock/compare.py && pixi run mypy --strict src/reblock/run.py src/reblock/compare.py`

```bash
git add conf/metric conf/config.yaml conf/compare_config.yaml conf/screen/dense_compact.yaml src/reblock/run.py src/reblock/compare.py tests/test_run.py
git commit -m "feat: conf/metric group (depth/depth_density/density_compactness), default metric=depth

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

## Task 4: Region growth + coloring use the metric

**Files:**
- Modify: `src/reblock/pipeline.py` (`_region_depth_map` → `_region_score_map`; `build_regions`)
- Modify: `src/reblock/emit.py` (`region_map` colorbar/title label)
- Modify: `src/reblock/run.py` (`region_map` call: pass `selection_scores` + metric label)
- Test: `tests/test_pipeline.py`, `tests/test_emit.py`

**Interfaces:**
- Consumes: `getattr(screen, "metric", None)` (Task 2); `block_depths` (existing); `metric.fine`/`needs_peel`/`name`.
- Produces: `_region_score_map(source, screen, block_geoms, groups, bound_buildings) -> dict[str, float]`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline.py`:

```python
def test_region_score_map_uses_metric_fine_and_skips_peel_when_geometry_only() -> None:
    # A density_compactness metric (needs_peel=False) -> _region_score_map must NOT call block_depths;
    # scores come from columns. A depth metric (needs_peel=True) -> block_depths supplies the depth.
    import reblock.pipeline as pl
    from reblock.metric import Compactness, Density, Depth, Product

    class _Screen:            # carries the metric, mirrors DenseCompactScreen
        def __init__(self, m): self.metric = m
        def select(self, s): return []

    class _Src:
        blocks_path = "x"

    calls = {"n": 0}
    real = pl.block_depths
    pl.block_depths = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1) or {})  # type: ignore
    try:
        gdf = _chain_gdf()
        pl._region_score_map(cast(Source, _Src()), _Screen(Product([Density(), Compactness()])),
                             gdf, [["s"]], 100.0)
        assert calls["n"] == 0        # geometry-only: no peel
        pl._region_score_map(cast(Source, _Src()), _Screen(Depth()), gdf, [["s"]], 100.0)
        assert calls["n"] == 1        # depth: one batched block_depths call
    finally:
        pl.block_depths = real        # type: ignore
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_pipeline.py -k region_score_map -v`
Expected: FAIL (`_region_score_map` doesn't exist).

- [ ] **Step 3: `_region_score_map` (metric-aware)**

Replace `_region_depth_map` in `src/reblock/pipeline.py`:

```python
def _region_score_map(source: Source, screen: Screen, block_geoms: pd.DataFrame,
                      groups: list[list[str]], bound_buildings: float) -> dict[str, float]:
    """metric.fine for every block the DenseCluster growth could reach from `groups`. Peels the
    reachable neighbourhood in ONE `block_depths` call ONLY when the metric needs depth; otherwise
    scores from columns alone (no peel). `{}` for a non-peel-capable source with a depth metric."""
    metric = getattr(screen, "metric", None)
    if metric is None or getattr(source, "blocks_path", None) is None:
        return {}
    reach = _reachable_blocks(block_geoms, groups, bound_buildings)
    cols = {str(b): (c, a, p) for b, c, a, p in _reach_cols(block_geoms, reach)}
    depths = block_depths(source, reach) if metric.needs_peel else {}
    return {b: metric.fine(depths.get(b, 0.0), *cols[b]) for b in reach if b in cols}


def _reach_cols(block_geoms: pd.DataFrame, ids: list[str]
                ) -> list[tuple[str, float, float, float]]:
    """(block_id, count, area_m2, perim_m) for `ids`, from the cheap columns (perimeter in UTM)."""
    want = set(ids)
    sub = block_geoms[block_geoms["block_id"].astype(str).isin(want)]
    utm = sub.to_crs(sub.estimate_utm_crs())
    area = (sub["block_area_m2"].astype(float) if "block_area_m2" in sub.columns
            else utm.geometry.area)
    return [(str(b), float(c), float(ar), float(pe)) for b, c, ar, pe in
            zip(sub["block_id"], sub["building_count"], area, utm.geometry.length, strict=True)]
```

In `build_regions`, replace the `_region_depth_map(...)` call with `_region_score_map(source, screen,
block_geoms, groups, 3.0 * mb)` (the `mb = getattr(region_builder, "max_buildings", None)` guard is
unchanged; pass `screen` — `build_regions` already has it).

- [ ] **Step 4: `region_map` label + `run.py` metric threading**

`emit.region_map` already colors flagged blocks by the passed `depths` map (now the metric's fine
scores) — rename the local for clarity and set the colorbar label + title from a `metric_name`
parameter: add `metric_name: str = "score"` to `region_map`'s keyword-only params, and use it in the
colorbar `label=` and the `screen.png` title (replace "true access-depth (0..N rings)" with
`f"{metric_name} (0..{vmax:.0f})"`). In `src/reblock/run.py`, update the `region_map` call:

```python
    if cfg.region_map.enabled:
        sc = getattr(spec.screen, "selection_scores", None)
        scores = sc(spec.source) if sc is not None else None
        m = getattr(spec.screen, "metric", None)
        region_map(spec.source, output.regions, output.seed_groups, out_dir,
                   selection=output.selection, depths=scores,
                   metric_name=m.name if m is not None else "score")
```

(`selection_depths` → `selection_scores`; `depths=` keeps its name in `region_map` — it's the score
map.)

- [ ] **Step 5: Run pipeline + emit + run suites**

Run: `pixi run pytest tests/test_pipeline.py tests/test_emit.py tests/test_run.py -v`
Expected: PASS.

- [ ] **Step 6: Full check**

Run: `pixi run check`
Expected: ruff + mypy --strict + pytest green.

- [ ] **Step 7: Commit**

```bash
git add src/reblock/pipeline.py src/reblock/emit.py src/reblock/run.py tests/test_pipeline.py tests/test_emit.py
git commit -m "feat: region growth + region_map color by the configured metric (fine score, no peel if geometry-only)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

> **NOTE (calibration done):** the `depth` gate calibration is already committed (`a2277e3`):
> `proxy_keep_pct=50` (≈ old `proxy≥1.5`; metro eligible-proxy median ~1.53) + `metric_gate` absolute
> `max_depth≥2` → ~13,800 flagged. Tasks 5–6 below build the example generator + emit the two variants.

## Task 5: The dir-reader README generator

**Files:**
- Create: `scripts/gen_example_readme.py`
- Test: `tests/test_gen_example_readme.py` (+ a fixture dir under `tests/data/example_fixture/`)

**Interfaces:**
- Produces: `gen_example_readme(run_dir: Path, *, metric_name: str, formula: str, blurb: str) -> str`
  — reads the artifacts on disk and returns the README markdown; **each section is emitted iff its
  artifacts are present** (data-gated). Also a `write_readme(run_dir, out_dir, *, metric_name, formula,
  blurb)` that writes `out_dir/README.md`.

- [ ] **Step 1: Write the failing test (fixture directory, no compute)**

Create `tests/data/example_fixture/` with: `meta.json` =
`{"metric": "depth", "flagged": 13800, "total_blocks": 83192, "deepest_block": "ZAF.9.3.1_1_5810", "deepest_depth": 24, "region_members": 12, "region_parcels": 11006, "region_mean_depth": 8.7, "region_mean_density_per_ha": 62.0}`;
`lens_a_depth.csv` + `lens_b_matched.csv` (two rows each, the columns the two-lens driver emits); a
zero-byte `screen.jpg`, `region.jpg`. Then `tests/test_gen_example_readme.py`:

```python
import json
from pathlib import Path

from scripts.gen_example_readme import gen_example_readme

_FIX = Path(__file__).resolve().parent / "data/example_fixture"


def test_generated_readme_reflects_meta_and_lens_csvs() -> None:
    md = gen_example_readme(_FIX, metric_name="depth", formula="depth = √(nA)/P",
                            blurb="Deepest street-access fabric.")
    assert "depth = √(nA)/P" in md                     # formula line
    assert "13,800" in md and "83,192" in md           # screen stat from meta.json (thousands-sep)
    assert "12" in md and "11,006" in md               # region stats
    assert "clearance" in md                            # a lens-CSV row rendered
    assert "![screen](screen.jpg)" in md               # figure embed (present file)


def test_sections_are_data_gated(tmp_path) -> None:
    # a dir with meta.json but NO lens CSVs omits the two-lens section, without erroring.
    (tmp_path / "meta.json").write_text(json.dumps(
        {"metric": "depth", "flagged": 5, "total_blocks": 10, "deepest_block": "b",
         "deepest_depth": 3, "region_members": 1, "region_parcels": 2,
         "region_mean_depth": 2.0, "region_mean_density_per_ha": 9.0}))
    md = gen_example_readme(tmp_path, metric_name="depth", formula="f", blurb="b")
    assert "two-lens" not in md.lower() and "Lens A" not in md   # no lens CSVs -> no section
    assert "flagged" in md.lower()                                # screen section still present
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_gen_example_readme.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.gen_example_readme'`.

- [ ] **Step 3: Implement `scripts/gen_example_readme.py`**

A pure dir-reader. Structure (fill the templating; keep it a real report — headers, captioned
tables, thousands-separated numbers):

```python
"""Machine-generated README for a metric example variant. PURE dir-reader: reads the run outputs
already on disk (meta.json of structured stats, the two-lens lens_*.csv, frontier CSVs, figure
files) and returns the markdown. Each section is emitted only if its artifacts are present, so the
numbers can never drift from the data and a partial run yields a partial (never-erroring) README."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def _n(x: float) -> str:
    return f"{x:,.0f}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def gen_example_readme(run_dir: Path, *, metric_name: str, formula: str, blurb: str) -> str:
    parts: list[str] = []
    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
    parts.append(f"# Multiblock, screened by `{metric_name}`\n")
    parts.append(f"*{blurb}*\n")
    parts.append(f"**Metric:** `{formula}` — one metric drives the screen, region growth, and "
                 f"colouring end to end.\n")
    if meta:
        parts.append("## 1. Screen the metro\n")
        parts.append(f"`{metric_name}` flags **{_n(meta['flagged'])} of {_n(meta['total_blocks'])}** "
                     f"blocks. Deepest: `{meta['deepest_block']}` at {meta['deepest_depth']:.0f} rings.\n")
        if (run_dir / "screen.jpg").exists():
            parts.append("![screen](screen.jpg)\n")
        parts.append("## 2. Grow the region\n")
        parts.append(f"The metric grows a **{meta['region_members']}-block** region "
                     f"(**{_n(meta['region_parcels'])} parcels**), mean depth "
                     f"{meta['region_mean_depth']:.1f} rings, mean density "
                     f"{meta['region_mean_density_per_ha']:.0f} bldg/ha.\n")
        if (run_dir / "region.jpg").exists():
            parts.append("![region](region.jpg)\n")
    lens_a, lens_b = run_dir / "lens_a_depth.csv", run_dir / "lens_b_matched.csv"
    if lens_a.exists() and lens_b.exists():
        parts.append("## 3. Compare the methods (two lenses)\n")
        parts.append("**Lens A — every parcel to the depth target:**\n")
        parts.append(_md_table(_read_csv(lens_a)))
        parts.append("\n**Lens B — matched road budget:**\n")
        parts.append(_md_table(_read_csv(lens_b)))
    return "\n".join(parts) + "\n"


def _md_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    cols = list(rows[0])
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = "\n".join("| " + " | ".join(r[c] for c in cols) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}\n"


def write_readme(run_dir: Path, out_dir: Path, *, metric_name: str, formula: str, blurb: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = gen_example_readme(run_dir, metric_name=metric_name, formula=formula, blurb=blurb)
    path = out_dir / "README.md"
    path.write_text(md)
    return path
```

- [ ] **Step 4: Run to verify it passes + lint/type**

Run: `pixi run pytest tests/test_gen_example_readme.py -v && pixi run ruff check scripts/gen_example_readme.py tests/test_gen_example_readme.py && pixi run mypy --strict scripts/gen_example_readme.py`
Expected: PASS + no errors.

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_example_readme.py tests/test_gen_example_readme.py tests/data/example_fixture
git commit -m "feat: dir-reader README generator for metric example variants (data-gated sections)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

## Task 6: Orchestrate + emit the two example variants

**Files:**
- Create: `scripts/gen_multiblock_example.py`
- Create: `examples/multiblock_depth/`, `examples/multiblock_depthdensity/`
- Delete: `examples/multiblock/`
- Modify: `examples/README.md`

**Interfaces:**
- Consumes: Task 5 `write_readme`; the existing example commands.

- [ ] **Step 1: Write the orchestrator**

`scripts/gen_multiblock_example.py` — `main()` takes a metric name (`depth`|`depth_density`), a
formula, and a blurb (a small authored map in the script for the two shipped variants), runs the
example's commands with `metric=<name>` into `examples/multiblock_<name>/`, captures the structured
stats into `meta.json`, converts the run's `screen.png`/`region.png` → `.jpg`, then calls
`write_readme`. Reuse the existing example command lines (the `reblock.run` screen/region/map command
with `metric=<name>` added; `scripts.compare_budgets` for the two lenses; `reblock.compare` for the
frontier), all pointed at the variant dir. The two blurbs (authored):
- `depth`: "The deepest street-access fabric — how many parcels a home sits from a street, regardless of crowding."
- `depth_density`: "Deep *and* crowded — the metric that isolates the genuine informal settlements, fading the deep-but-sparse blocks."

- [ ] **Step 2: Emit both variants**

```bash
pixi run python -m scripts.gen_multiblock_example depth
pixi run python -m scripts.gen_multiblock_example depth_density
```

Each writes `examples/multiblock_<name>/` with `screen.jpg`, `region.jpg`, the lens/frontier CSVs +
figures, `meta.json`, `run.log`, and the generated `README.md`. Verify each README's numbers match its
CSVs and figures resolve.

- [ ] **Step 3: Delete the old example + update the index**

```bash
git rm -r examples/multiblock
```
Update `examples/README.md`: replace the single `multiblock` row with the two variant rows, noting
they demonstrate the swappable block metric (same pipeline, `metric=depth` vs `metric=depth_density`).

- [ ] **Step 4: Verify + commit**

Run: `pixi run check`
Expected: green. Confirm both variant READMEs render, figures resolve, `examples/multiblock/` is gone.

```bash
git add examples scripts/gen_multiblock_example.py
git rm -r examples/multiblock
git commit -m "docs: emit multiblock_depth + multiblock_depthdensity example variants (generated READMEs)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

## Self-Review

**1. Spec coverage:**
- §3.1 algebra (Depth/Density/Compactness + Power/Product, proxy/fine/needs_peel/identity) → Task 1. ✓
- §3.2 gate per-metric (absolute|percentile) + shared recall pre-filter → Task 1 (`Gate`) + Task 2 (`proxy_keep_pct`). ✓
- §3.3 wiring: screen (proxy gate + needs_peel-skip fine + selection_scores) → Task 2; region growth + coloring → Task 4. ✓
- §3.4 needs_peel=False skips the peel → Task 2 (`test_density_compactness_metric_skips_the_peel`) + Task 4 (`test_region_score_map...skips_peel`). ✓
- §4 config group + default depth → Task 3. ✓
- §5 mean-vs-max caveat + calibration → Task 5. ✓
- §6 scope (minimal algebra) → Task 1 (only Power/Product). ✓

**2. Placeholder scan:** No TBD/TODO. Task 2 Step 1 flags a real decision (gate as a screen arg vs on the metric) and RESOLVES it (gate as an explicit `DenseCompactScreen(metric, gate, …)` arg) — all later tasks use that signature consistently.

**3. Type consistency:** `DenseCompactScreen(metric: BlockMetric, gate: Gate, *, proxy_keep_pct, min_buildings)` identical in Tasks 2–4 + configs. `ScreenSelectionInput(metric, gate, proxy_keep_pct, min_buildings)` consistent. `selection_scores(source) -> dict[str,float]` in Tasks 2/4. `_region_score_map(source, screen, block_geoms, groups, bound)` in Tasks 3-code/4. `metric.proxy/fine/needs_peel/identity/name` and `Gate.keep/identity` match Task 1's definitions. `region_map(..., metric_name=…)` in Task 4 code + run.py call.

## Execution Handoff

Execution is **subagent-driven-development** (owner's standing preference — not asked). Fresh implementer per task + task review (spec + quality) + a final whole-branch review. Task 2 is the largest (screen-selection rewrite + construction-site migration); Task 5 is compute-heavy (calibration + metro regeneration) — run it last.
