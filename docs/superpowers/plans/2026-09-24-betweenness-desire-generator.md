# Betweenness Desire Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the curvature-aware repelled-betweenness desire field as a `DesireLineSource`, with four `method=` presets (tree/looped x raw/contrast), reproducing the research code's fields and proposals exactly.

**Architecture:** A new package `reblock.methods.betweenness` computes, per block, route counts over a
(cell, direction-bucket) state graph with numba (egress to the street, and all pairs of homes), plus
the same counts on the block with no buildings (the prior). A `FieldContrast` Strategy turns counts
into a field (`RawShare`, `PriorDeviance`); nested top-quantile skeletons of the field become a
`DesireField`, which the existing `DemandGreedyReblocker` routes toward. `LoopClosureRefiner`
(unchanged) wraps it for the looped presets. Counts are memoized through `derive_graph.derive`.

**Tech Stack:** numpy, scipy (cKDTree), shapely 2 (STRtree), numba (new), Hydra `_target_` presets,
pytest, mypy --strict, ruff.

**Spec:** this plan's Background section. The research it productionizes lives in the session
scratchpad `gaps/` (`curvbetween.py`, `llr.py`, `gen_study.py`, `flagship.py`, `harness.py`) and is
written up by Task 9 in `docs/superpowers/notes/2026-09-24-roadless-fields-and-the-betweenness-generator.md`.

## Background (the spec)

Measured on real footprints (all numbers paired medians vs clearance, 95% bootstrap):
- 220 small Cape Town recipients: routing `demand_greedy` toward the raw field gives Lens A
  permeability +0.022 [+.014,+.031], Lens B displacement -0.006; toward the contrast field +0.032,
  -0.007. `demand_greedy` with `NoDesire` ties clearance exactly, so the gain is the field's.
- Held out (sparser Cape Town, Nairobi): Lens B replicates in both (-0.005..-0.006).
- 36 large blocks (>= 1,000 buildings, top of the depth_density screen, CT 24 + NBO 12): the looped
  generators beat clearance_looped on Lens A by +0.043..+0.048 (94-97% of blocks) and Lens B by
  -0.004..-0.006; cycle_native still leads Lens A by 0.015-0.022 and greedy_arterial leads Lens B
  (better in 83-94% of blocks); a generator arm is on the (Lens A, Lens B) Pareto frontier in 17 of
  36 blocks, as the middle point between them, while clearance and clearance_looped are on it in
  none. So it is a frontier method, shipped as four presets, one per measured operating point.

The exact algorithm (every constant below was the measured configuration; do not "improve" any):
- Raster at `res_m`: cell centres `x0 + j*res`, `y0 + i*res` from the boundary's bounds; `inside` by
  `contains_xy`; `clearance` = exact distance to the nearest building OUTLINE (0 inside one; NaN
  outside the block); `edge` = distance to the block boundary. `free = inside & clearance > 0`.
- Homes: one per building, the free cell nearest its anchor (`block.buildings.xy`), by cKDTree.
- Cost density `c = 1 + (r0 / max(clearance, res/2))^2` on free cells, 50 in buildings. 16 lattice
  directions (king + knight); state = (cell, bucket between adjacent directions); inside a bucket the
  path alternates freely between its two bounding steps; changing to an adjacent bucket costs
  `lam * dmid^2 / 2 m`. Step cost = step length x mean density over the cells it touches.
- O1 egress: roots = inside cells with `edge <= 1.5*res`, in every bucket; per cell, homes whose best
  route to the street passes through it. O2 all pairs: one Dijkstra per source home; per cell,
  (source, target) pairs whose best route passes through it; sources are all homes, or
  `max_sources` sampled with `np.random.default_rng(seed).choice(n, max_sources, replace=False)`,
  scaled by `n_homes / max_sources`.
- Prior E1, E2: the same, with `clearance = 1e12` on every inside cell and `r0 = 0` (density exactly 1).
- Counts are stored as float32 (NaN outside) -- the measured pipeline read them from a float32 cache.
- RawShare: `O1/nanmax(O1) + O2/nanmax(O2)` (floor 1e-12 on each max).
- PriorDeviance (floor 1e-9): `o = O1+O2`, `e = max(E1+E2, floor)`,
  `dev = 2*(where(o>0, o*log(o/e), 0) - (o-e))`, `z = sign(o-e)*sqrt(max(dev, 0))`.
- Desire: for q in (0.80, 0.85, 0.90, 0.95, 0.98): threshold at `np.quantile(field[free & finite], q)`,
  skeletonize (scikit-image's Zhang-Suen), one group of unit segments between 8-adjacent skeleton
  pixels, weight 1/5.
- Presets: tree = `DemandGreedyReblocker` (buffer_m 3.0, eps 0.1, gamma 1.0, depth_target 2,
  max_roads 400, road_width_m 7.0, substrate `${substrate}`); looped = `LoopClosureRefiner` with the
  region-scale loop settings the frontier study measured (budget_frac 0.30, min_bridges_per_m 0.01,
  max_loops 400, min_loop_len_m 40.0, search_radius_m 60.0, snap_lam 2.0, max_candidates 1500,
  road_width_m 7.0). Field: res_m 1.0, r0_m 2.0, bend_lambda 50.0, max_sources 400, seed 0.

## Global Constraints

- Python >=3.11,<3.13; `pixi run mypy` (strict, no excludes) and `pixi run ruff check .` clean; line length 100.
- No defaults on any dataclass field of a data carrier or a configured strategy/method: presets spell every field.
- No string-selected modes: the contrast is a Strategy object (`FieldContrast`), injected by config.
- No silent fallbacks: a condition the code cannot honour raises with an actionable message.
- `src/reblock` may import numba (new runtime dependency) but NOT scikit-image (research-only feature).
- Every numba kernel goes through the typed `reblock._jit.njit` wrapper; `numba.*` gets a mypy override shim like joblib's.
- Commits end with the session attribution lines; work on branch `betweenness-desire`; merge to main at the end.
- Do not touch `conf/example/*` or the site (not in scope).

## Review Focus

1. A degenerate block -- no free cells, no homes, or no street-band cell -- expected: `desire_field` returns `DesireField(groups=())` (demand_greedy then behaves as NoDesire), never an IndexError from `np.quantile` on an empty array (the smoke-run crash), and never the whole free area as one "ridge" from thresholding an all-zero field. Test in Task 6.
2. `workers > 1` inside a daemonic process (a study's `multiprocessing.Pool` worker) -- expected: `RuntimeError` naming the fix ("set workers=1"), not multiprocessing's bare `AssertionError: daemonic processes are not allowed to have children`, and not a silent serial fallback. Test in Task 3.
3. NaN outside the block -- expected: normalisation uses `nanmax`; `.max()` would make every cell NaN (the smoke-run bug). Test in Task 4.
4. Several homes nearest the same grid cell -- expected: counted with multiplicity, not deduplicated away. Test in Task 3.
5. A disc-tier block (SpacingDiscs) -- expected: works (every tier has `outlines`), and the counts cache key differs by tier (Block.identity carries the tier). Test in Task 4.

---

### Task 1: numba dependency and the typed `njit` wrapper

**Files:**
- Modify: `pyproject.toml` (`[tool.pixi.dependencies]`; a `[[tool.mypy.overrides]]` shim)
- Modify: `pixi.lock` (regenerated by `pixi install`)
- Create: `src/reblock/_jit.py`
- Test: `tests/test_jit.py`

**Interfaces:**
- Produces: `reblock._jit.njit(fn: F) -> F` (F bound to `Callable[..., Any]`): numba `njit(cache=True)` with the callable's type preserved for mypy.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_jit.py
"""The typed njit wrapper compiles and keeps the function's signature for mypy."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reblock._jit import njit


@njit
def _total(a: NDArray[np.float64]) -> float:
    s = 0.0
    for i in range(a.shape[0]):
        s += a[i]
    return s


def test_a_jitted_function_runs_compiled() -> None:
    assert _total(np.arange(10.0)) == 45.0
    assert hasattr(_total, "signatures") and _total.signatures   # numba dispatcher, compiled
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run pytest tests/test_jit.py -q -p no:cacheprovider --no-cov`
Expected: FAIL, `ModuleNotFoundError: No module named 'reblock._jit'` (and numba missing).

- [ ] **Step 3: Add the dependency and the shim**

In `pyproject.toml`, under `[tool.pixi.dependencies]` after `segno = "*"`:

```toml
# Compiles reblock.methods.betweenness's state-graph Dijkstra and thinning kernels. Measured: the
# pure-numpy/scipy alternative needs an explicit 16-state-per-cell sparse graph (9M states on a
# 6,619-building block) and was rejected on memory and time.
numba = "*"
```

After the `threadpoolctl` override block add:

```toml
[[tool.mypy.overrides]]
# numba ships no py.typed marker and no stub package. Every kernel goes through the typed
# `reblock._jit.njit` wrapper, so untyped numba never reaches a signature; same shim as joblib.
module = ["numba", "numba.*"]
ignore_missing_imports = true
```

Run: `pixi install` (updates `pixi.lock`).

- [ ] **Step 4: Write the wrapper**

```python
# src/reblock/_jit.py
"""numba's `njit`, typed: the decorated function keeps its own signature for mypy.

numba has no stubs, so `numba.njit` is `Any` and would erase every kernel's types under
--strict (`disallow_untyped_decorators`). This is the one place that cast happens."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

import numba

F = TypeVar("F", bound=Callable[..., Any])


def njit(fn: F) -> F:
    """`numba.njit(cache=True)`: compiled on first call, the machine code cached on disk."""
    return cast(F, numba.njit(cache=True)(fn))
```

- [ ] **Step 5: Run the test, mypy and ruff**

Run: `pixi run pytest tests/test_jit.py -q -p no:cacheprovider --no-cov && pixi run mypy && pixi run ruff check .`
Expected: 1 passed; mypy "Success"; ruff "All checks passed!"

- [ ] **Step 6: Commit**

```bash
git checkout -b betweenness-desire
git add pyproject.toml pixi.lock src/reblock/_jit.py tests/test_jit.py
git commit -m "numba joins the runtime dependencies, through one typed njit wrapper"
```

---

### Task 2: The block raster and home cells

**Files:**
- Create: `src/reblock/methods/betweenness/__init__.py` (empty for now: `"""Curvature-aware repelled betweenness as a desire field."""`)
- Create: `src/reblock/methods/betweenness/raster.py`
- Test: `tests/methods/betweenness/test_raster.py` (+ `tests/methods/betweenness/__init__.py` only if the sibling test dirs have one -- check `ls tests/methods/`)

**Interfaces:**
- Produces: `BlockRaster` (frozen, eq=False): `x0: float, y0: float, res: float, inside: NDArray[np.bool_], clearance: NDArray[np.float64], edge: NDArray[np.float64]`; property `free -> NDArray[np.bool_]`; property `shape -> tuple[int, int]`; classmethod `BlockRaster.of(block: Block, res: float) -> BlockRaster`.
- Produces: `home_cells(raster: BlockRaster, block: Block) -> NDArray[np.int64]` shape (n_buildings, 2) of (row, col); empty (0, 2) when the block has no buildings or no free cell.

- [ ] **Step 1: Write the failing tests**

```python
# tests/methods/betweenness/test_raster.py
from __future__ import annotations

import numpy as np
import shapely

from reblock.methods.betweenness.raster import BlockRaster, home_cells
from tests.scoring_fixtures import _block_1808


def test_clearance_is_the_exact_distance_to_the_nearest_outline() -> None:
    block = _block_1808()
    r = BlockRaster.of(block, 1.0)
    ii, jj = np.nonzero(r.inside)
    k = np.linspace(0, len(ii) - 1, 50).astype(int)
    pts = shapely.points(r.x0 + r.res * jj[k], r.y0 + r.res * ii[k])
    want = np.array([block.buildings.outlines.distance(p).min() for p in pts])
    assert np.allclose(r.clearance[ii[k], jj[k]], want)
    assert np.isnan(r.clearance[~r.inside]).all() and np.isnan(r.edge[~r.inside]).all()
    assert (r.clearance[r.inside] >= 0).all()


def test_every_home_is_a_free_cell_nearest_its_building() -> None:
    block = _block_1808()
    r = BlockRaster.of(block, 1.0)
    homes = home_cells(r, block)
    assert homes.shape == (len(block.buildings), 2)
    assert r.free[homes[:, 0], homes[:, 1]].all()
    fr = np.argwhere(r.free)
    for (i, j), (x, y) in zip(homes[:5], block.buildings.xy[:5], strict=True):
        d = np.hypot(r.x0 + r.res * fr[:, 1] - x, r.y0 + r.res * fr[:, 0] - y)
        assert np.isclose(np.hypot(r.x0 + r.res * j - x, r.y0 + r.res * i - y), d.min())
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/methods/betweenness/test_raster.py -q -p no:cacheprovider --no-cov`
Expected: FAIL, `ModuleNotFoundError: reblock.methods.betweenness.raster`.

- [ ] **Step 3: Implement** (ported verbatim from the research harness; the arithmetic order matters for bit-identity)

```python
# src/reblock/methods/betweenness/raster.py
"""The block as a raster: which cells are inside, how far each is from the nearest building, and
how far from the street. Every betweenness count is computed on this grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from shapely import STRtree

from reblock.contracts import Block


@dataclass(frozen=True, eq=False)
class BlockRaster:
    x0: float                                 # centre of cell (0, 0)
    y0: float
    res: float
    inside: NDArray[np.bool_]                 # (ny, nx): cell centre inside the block
    clearance: NDArray[np.float64]            # metres to the nearest building outline; 0 inside
    edge: NDArray[np.float64]                 # metres to the block boundary (the street)

    @property
    def shape(self) -> tuple[int, int]:
        ny, nx = self.inside.shape
        return int(ny), int(nx)

    @property
    def free(self) -> NDArray[np.bool_]:
        return self.inside & (np.nan_to_num(self.clearance) > 0)

    @classmethod
    def of(cls, block: Block, res: float) -> BlockRaster:
        minx, miny, maxx, maxy = block.boundary.bounds
        xs = np.arange(minx, maxx + res, res)
        ys = np.arange(miny, maxy + res, res)
        X, Y = np.meshgrid(xs, ys)
        inside = shapely.contains_xy(block.boundary, X.ravel(), Y.ravel()).reshape(X.shape)
        pts = shapely.points(X.ravel()[inside.ravel()], Y.ravel()[inside.ravel()])
        clearance = np.full(X.shape, np.nan)
        outlines = np.asarray(block.buildings.outlines.geometry)
        if len(outlines) and len(pts):
            # query_nearest returns (INPUT index, TREE index) pairs; only the distance is kept,
            # scattered by INPUT index (reading them the other way round wrote cell ids into
            # polygon slots once, and every width read 0.00).
            idx, dist = STRtree(outlines).query_nearest(pts, return_distance=True,
                                                         all_matches=False)
            d = np.full(len(pts), np.nan)
            d[idx[0]] = dist
            clearance[inside] = d
        elif len(pts):
            clearance[inside] = np.inf            # no buildings: every inside cell is free
        edge = np.full(X.shape, np.nan)
        edge[inside] = shapely.distance(pts, block.boundary.boundary)
        return cls(x0=float(xs[0]), y0=float(ys[0]), res=res, inside=inside,
                   clearance=clearance, edge=edge)


def home_cells(raster: BlockRaster, block: Block) -> NDArray[np.int64]:
    """(row, col) of the free cell nearest each building's anchor, one row per building."""
    free_rc = np.argwhere(raster.free)
    if len(block.buildings) == 0 or len(free_rc) == 0:
        return np.empty((0, 2), dtype=np.int64)
    tree = cKDTree(np.c_[raster.x0 + raster.res * free_rc[:, 1],
                         raster.y0 + raster.res * free_rc[:, 0]])
    _d, k = tree.query(block.buildings.xy)
    return free_rc[np.asarray(k, dtype=np.int64)].astype(np.int64)
```

Note on bit-identity: the research harness defined `free = inside & (clearance > 0)` with NaN
outside (NaN > 0 is False); `np.nan_to_num(clearance) > 0` is identical on inside cells and
False outside. Keep it.

- [ ] **Step 4: Run tests, mypy, ruff** -- expected all pass/clean.
- [ ] **Step 5: Commit** -- `git commit -m "betweenness: the block raster and each building's home cell"`

---

### Task 3: The state-graph router and the two count passes

**Files:**
- Create: `src/reblock/methods/betweenness/routing.py`
- Test: `tests/methods/betweenness/test_routing.py`

**Interfaces:**
- Consumes: `reblock._jit.njit`.
- Produces:
  - `NB = 16`, `BUILDING_COST = 50.0`, `L0_M = 2.0`, `DIRS` (16,2) int64 (dx=col, dy=row), `bend_table(lam: float) -> NDArray[np.float64]` (16, 3).
  - `StateGraph` (frozen, eq=False): `node: NDArray[np.int64]` (ny,nx; -1 outside), `nbr: NDArray[np.int32]` (N,16), `cost: NDArray[np.float64]` (N,16), `rr: NDArray[np.int64]`, `cc: NDArray[np.int64]`.
  - `build_graph(inside, clearance, res: float, r0: float) -> StateGraph`.
  - `egress_counts(g: StateGraph, bend, home_nodes: NDArray[np.int64], band_nodes: NDArray[np.int64]) -> NDArray[np.float64]` (N,).
  - `pair_counts(g: StateGraph, bend, home_nodes, source_homes: NDArray[np.int64], workers: int) -> NDArray[np.float64]` (N,); raises `RuntimeError` if `workers > 1` in a daemonic process.
  - `to_grid(values: NDArray[np.float64], g: StateGraph, inside) -> NDArray[np.float64]` (ny,nx; NaN outside).

- [ ] **Step 1: Write the failing tests** (the research checks, as assertions)

```python
# tests/methods/betweenness/test_routing.py
"""The router's four behavioural checks, each measured on the research code first:
straight roads cost their length at every angle; one gap carries every crossing pair; the bend
penalty moves traffic from a shorter zigzag to a straight route; repulsion widens routes."""
from __future__ import annotations

import math
import multiprocessing as mp

import numpy as np
import pytest
from scipy import ndimage

from reblock.methods.betweenness.routing import (
    bend_table, build_graph, egress_counts, pair_counts, route_cost, to_grid)


def test_a_straight_route_costs_its_length_at_every_angle() -> None:
    n, c0, R = 241, (120, 120), 100.0
    inside = np.ones((n, n), bool)
    cl = np.full((n, n), 1e6)
    for lam in (5.0, 20.0, 50.0):
        ratios = []
        for a in np.linspace(0, 90, 29):
            r = int(round(c0[0] + R * math.sin(math.radians(a))))
            c = int(round(c0[1] + R * math.cos(math.radians(a))))
            cost, _cells = route_cost(inside, cl, 1.0, 0.0, lam, c0, (r, c))
            ratios.append(cost / math.hypot(r - c0[0], c - c0[1]))
        ratios_a = np.array(ratios)
        assert np.max(np.abs(ratios_a / ratios_a[0] - 1)) <= 0.03, (lam, ratios_a)


def _two_rooms() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = 21, 41
    inside = np.ones((ny, nx), bool)
    wall = np.zeros((ny, nx), bool)
    wall[:, 20] = True
    wall[9:12, 20] = False
    cl = ndimage.distance_transform_edt(~wall) * 1.0
    cl[wall] = 0.0
    return inside, cl, wall


def test_one_gap_carries_every_crossing_pair_and_the_wall_none() -> None:
    inside, cl, wall = _two_rooms()
    g = build_graph(inside, cl, 1.0, 1.0)
    rng = np.random.default_rng(1)
    left = [(int(r), int(c)) for r, c in zip(rng.integers(1, 20, 6), rng.integers(1, 18, 6))]
    right = [(int(r), int(c)) for r, c in zip(rng.integers(1, 20, 5), rng.integers(23, 40, 5))]
    homes = np.array([g.node[p] for p in left + right], dtype=np.int64)
    cross = 2 * len(left) * len(right)
    for lam in (5.0, 50.0):
        f = to_grid(pair_counts(g, bend_table(lam), homes, homes, 1), g, inside)
        gap = float(np.nansum(f[9:12, 20]))
        assert cross <= gap <= 2 * cross      # every crossing route touches 1-2 gap cells
        assert float(np.nansum(f[wall])) == 0.0


def test_homes_sharing_a_cell_count_with_multiplicity() -> None:
    inside, cl, _wall = _two_rooms()
    g = build_graph(inside, cl, 1.0, 1.0)
    a, b = g.node[5, 5], g.node[5, 35]
    one = pair_counts(g, bend_table(5.0), np.array([a, b]), np.array([a, b]), 1)
    two = pair_counts(g, bend_table(5.0), np.array([a, a, b]), np.array([a, a, b]), 1)
    assert np.isclose(two.max(), 2 * one.max())   # 2 homes at `a` -> twice the a<->b traffic


def test_the_bend_penalty_moves_traffic_from_a_shorter_zigzag_to_a_straight_route() -> None:
    ny, nx = 90, 130
    inside = np.ones((ny, nx), bool)
    free = np.zeros((ny, nx), bool)
    P, Q = (45, 10), (45, 120)
    rr, cc = np.mgrid[0:ny, 0:nx]

    def carve(pts: list[tuple[int, int]], w: float) -> None:
        for (r1, c1), (r2, c2) in zip(pts[:-1], pts[1:]):
            for t in np.linspace(0, 1, max(1, int(math.hypot(r2 - r1, c2 - c1) * 4))):
                r, c = r1 + t * (r2 - r1), c1 + t * (c2 - c1)
                free[(rr - r) ** 2 + (cc - c) ** 2 <= (w / 2) ** 2] = True

    carve([P, (5, 65), Q], 3.0)                                      # straight-legged, 136 m
    carve([P, (51, 20)] + [(51 + (6 if i % 2 else 0), 20 + 10 * i) for i in range(1, 10)] + [Q],
          3.0)                                                        # zigzag, 132 m
    cl = ndimage.distance_transform_edt(free) * 1.0
    cl[~free] = 0.0
    g = build_graph(inside, cl, 1.0, 0.5)
    homes = np.array([g.node[P[0] + d, P[1]] for d in (-1, 0, 1)]
                     + [g.node[Q[0] + d, Q[1]] for d in (-1, 0, 1)], dtype=np.int64)

    def through(lam: float) -> tuple[float, float]:
        f = to_grid(pair_counts(g, bend_table(lam), homes, homes, 1), g, inside)
        return float(np.nanmax(f[3:20, 58:72])), float(np.nanmax(f[48:62, 58:72]))

    s_lo, w_lo = through(0.01)
    s_hi, w_hi = through(50.0)
    assert w_lo > s_lo and s_hi > w_hi


def test_repulsion_pushes_routes_into_wider_space() -> None:
    inside = np.ones((41, 41), bool)
    obs = np.zeros((41, 41), bool)
    obs[16:25, 16:25] = True
    cl = ndimage.distance_transform_edt(~obs) * 1.0
    cl[obs] = 0.0
    means = []
    for r0 in (0.25, 1.0, 4.0):
        _c, cells = route_cost(inside, cl, 1.0, r0, 50.0, (20, 2), (20, 38))
        means.append(float(cl[cells[:, 0], cells[:, 1]].mean()))
    assert means[0] < means[1] < means[2]


def test_egress_counts_each_home_once_on_its_way_to_the_street() -> None:
    inside = np.zeros((30, 30), bool)
    inside[1:-1, 1:-1] = True
    cl = np.where(inside, 1e6, np.nan)
    g = build_graph(inside, cl, 1.0, 0.0)
    band = g.node[inside & ((np.arange(30)[:, None] == 1) | (np.arange(30)[None, :] == 1))]
    home = np.array([g.node[20, 20]], dtype=np.int64)
    f = egress_counts(g, bend_table(50.0), home, band)
    assert f.max() == 1.0 and f[g.node[20, 20]] == 0.0      # endpoints are never credited


def _pairs_in_daemon(_: int) -> None:
    inside, cl, _wall = _two_rooms()
    g = build_graph(inside, cl, 1.0, 1.0)
    homes = np.array([g.node[5, 5], g.node[5, 35]], dtype=np.int64)
    pair_counts(g, bend_table(5.0), homes, homes, 2)


def test_parallel_pairs_inside_a_daemonic_worker_fail_by_name() -> None:
    with mp.get_context("fork").Pool(1) as pool, pytest.raises(RuntimeError, match="workers=1"):
        pool.map(_pairs_in_daemon, [0])
```

- [ ] **Step 2: Run to verify failure** -- `ModuleNotFoundError` for `routing`.

- [ ] **Step 3: Implement.** Port `/tmp/claude-1641171234/-home-gchurchill-src-reblock/ab8d38f1-74d4-4fea-83ad-93f582da4739/scratchpad/gaps/curvbetween.py` (the research code; read it in full first) lines 41-315 (constants, `bucket_mid`, `bend_table`, `density`, `build`, `_push`, `_sift_up`, `_pop`, `_dijkstra`, `_credit`, `_alloc`, `_pairs_chunk`, `all_pairs`, `egress`, `to_field`) and `_route_cost` (lines 364-378) with ONLY these changes -- every arithmetic expression stays byte-for-byte (bit-identity is checked in Task 8):
  - `@njit(cache=False)` -> `@njit` from `reblock._jit` (cache=True). Add numpy type annotations to every kernel parameter (`NDArray[np.int64]` etc.); numba ignores them, mypy reads them.
  - `build` returns `StateGraph(node, nbr, cost, rr, cc)` (a frozen dataclass) instead of a tuple; rename to `build_graph`. `egress` -> `egress_counts(g, bend, home_nodes, band_nodes)`, `all_pairs` -> `pair_counts(g, bend, home_nodes, source_homes, workers)`, `to_field` -> `to_grid(values, g, inside)`, `_route_cost` -> public `route_cost(inside, clearance, res, r0, lam, a, b) -> tuple[float, NDArray[np.int64]]` (used by tests and by the Task 8 check).
  - The module global `_S` dict becomes a frozen dataclass `_PairWork(nbr, cost, bend, tnodes, tw, is_target, rr, cc, grid)` held in a module global `_WORK: _PairWork | None`, set immediately before forking and read by `_pairs_chunk` (the arterial engine's pattern: forked children inherit it; never pass the arrays through pickling).
  - The fork pool is `concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork"))`, results summed in submission order with `np.sum(list(ex.map(_pairs_chunk, parts)), axis=0)` -- the research code used `Pool.map`; both return results in input order, so the float sum is identical.
  - Before forking, in `pair_counts`:

```python
    if workers > 1 and multiprocessing.current_process().daemon:
        raise RuntimeError(
            f"pair_counts(workers={workers}) cannot fork: this process is a daemonic pool worker, "
            f"and daemonic processes may not have children. Parallelize across blocks OR across "
            f"sources, not both -- set workers=1 on the betweenness desire source for runs that "
            f"already fork per block.")
```

  - Keep the serial path for `workers <= 1 or len(jobs) < 16` exactly as the research code.

- [ ] **Step 4: Run the tests** -- all 7 pass; mypy, ruff clean.
- [ ] **Step 5: Commit** -- `git commit -m "betweenness: the (cell, direction) router and its egress and all-pairs counts"`

---

### Task 4: Counts, the no-buildings prior, and the contrast strategies

**Files:**
- Create: `src/reblock/methods/betweenness/counts.py`
- Create: `src/reblock/methods/betweenness/contrast.py`
- Test: `tests/methods/betweenness/test_counts.py`, `tests/methods/betweenness/test_contrast.py`

**Interfaces:**
- Consumes: `BlockRaster`, `home_cells` (Task 2); `build_graph`, `bend_table`, `egress_counts`, `pair_counts`, `to_grid` (Task 3); `reblock.derive_graph.derive`, `config_identity`.
- Produces:
  - `CountParams` (frozen): `res_m: float, r0_m: float, bend_lambda: float, max_sources: int, seed: int`; property `identity -> Hashable` = `config_identity(self)`.
  - `Counts` (frozen, eq=False): `raster: BlockRaster, egress: NDArray[np.float32], pairs: NDArray[np.float32]` (ny,nx, NaN outside; `pairs` already scaled).
  - `observed(block: Block, params: CountParams, workers: int) -> Counts` and `prior(block: Block, params: CountParams, workers: int) -> Counts`, each memoized through `derive` (keyed on block identity + params; `workers` excluded).
  - `FieldContrast` Protocol (runtime_checkable): `identity -> Hashable`, `field(block: Block, params: CountParams, workers: int) -> tuple[BlockRaster, NDArray[np.float64]]`.
  - `RawShare()` and `PriorDeviance(floor: float)` implementing it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/methods/betweenness/test_counts.py
from __future__ import annotations

import dataclasses

import numpy as np

from reblock.buildings import AreaDiscs
from reblock.contracts import Block
from reblock.methods.betweenness.counts import (
    NO_BUILDINGS, CountParams, CountsInput, observed, prior)
from reblock.methods.betweenness.routing import build_graph
from tests.scoring_fixtures import _block_1808

P = CountParams(res_m=1.0, r0_m=2.0, bend_lambda=50.0, max_sources=400, seed=0)


def _keyed() -> Block:
    # A real source hash, so Block.identity (and every key built on it) is not None.
    return dataclasses.replace(_block_1808(), source_content_hash="fixture")


def test_with_no_buildings_the_observed_graph_is_the_prior_graph() -> None:
    # r0 = 2 over clearance 1e12 is density 1 + (2/1e12)^2 == 1.0 exactly: the prior's own graph,
    # so on an empty block observed and prior must agree bit for bit.
    inside = np.zeros((40, 60), bool)
    inside[1:-1, 1:-1] = True
    clear = np.where(inside, NO_BUILDINGS, np.nan)
    a, b = build_graph(inside, clear, 0.5, 2.0), build_graph(inside, clear, 0.5, 0.0)
    assert np.array_equal(a.cost, b.cost) and np.array_equal(a.nbr, b.nbr)


def test_counts_share_the_raster_and_are_nan_outside() -> None:
    block = _block_1808()
    o, e = observed(block, P, 1), prior(block, P, 1)
    assert o.egress.shape == e.pairs.shape == o.raster.shape
    assert np.isnan(o.pairs[~o.raster.inside]).all()
    assert np.nanmax(o.pairs) > 0 and np.nanmax(e.pairs) > 0


def test_counts_are_keyed_on_the_building_tier() -> None:
    spacing = _keyed()
    area = dataclasses.replace(spacing, building_tier=AreaDiscs)
    assert CountsInput(spacing, P, 1).identity != CountsInput(area, P, 1).identity


def test_workers_do_not_enter_the_cache_key() -> None:
    b = _keyed()
    assert CountsInput(b, P, 1).identity == CountsInput(b, P, 8).identity
```

```python
# tests/methods/betweenness/test_contrast.py
from __future__ import annotations

import numpy as np

from reblock.methods.betweenness.contrast import deviance_field, raw_share_field


def test_raw_share_ignores_nan_outside_the_block() -> None:
    o1 = np.array([[np.nan, 1.0, 2.0]], np.float32)
    o2 = np.array([[np.nan, 10.0, 5.0]], np.float32)
    f = raw_share_field(o1, o2)
    assert np.isnan(f[0, 0]) and np.allclose(f[0, 1:], [0.5 + 1.0, 1.0 + 0.5])


def test_deviance_is_zero_where_observation_equals_prior_and_positive_where_it_exceeds() -> None:
    o = np.array([[0.0, 5.0, 50.0, 1.0]], np.float32)
    e = np.array([[0.0, 5.0, 2.0, 10.0]], np.float32)
    z = deviance_field(o, np.zeros_like(o), e, np.zeros_like(e), floor=1e-9)
    assert abs(z[0, 0]) < 1e-4 and z[0, 1] == 0.0 and z[0, 2] > 5 and z[0, 3] < -1
```

- [ ] **Step 2: Run to verify failure** -- ModuleNotFoundError.

- [ ] **Step 3: Implement** `counts.py` (a port of `gaps/llr.py::oe_arrays` split into its two halves):

```python
# src/reblock/methods/betweenness/counts.py
"""Route counts per cell: observed on the block, and the prior -- the same computation with no
buildings at all. Both memoized through `derive`, keyed on the block (tier included) and the
counting parameters; `workers` changes nothing and is not in the key."""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reblock.contracts import Block
from reblock.derive_graph import config_identity, derive
from reblock.methods.betweenness.raster import BlockRaster, home_cells
from reblock.methods.betweenness.routing import (
    bend_table, build_graph, egress_counts, pair_counts, to_grid)

NO_BUILDINGS = 1e12      # clearance with no buildings: 1 + (0/1e12)^2 == 1.0 exactly


@dataclass(frozen=True)
class CountParams:
    res_m: float
    r0_m: float
    bend_lambda: float
    max_sources: int
    seed: int

    @property
    def identity(self) -> Hashable:
        return config_identity(self)


@dataclass(frozen=True, eq=False)
class Counts:
    raster: BlockRaster
    egress: NDArray[np.float32]      # homes per cell on their best route to the street
    pairs: NDArray[np.float32]       # (source, target) pairs per cell, scaled to all homes


@dataclass(frozen=True, eq=False)
class CountsInput:
    block: Block
    params: CountParams
    workers: int                     # not in `identity`: it changes nothing

    @property
    def identity(self) -> Hashable | None:
        b = self.block.identity
        return None if b is None else (b, self.params.identity)


def _count(inp: CountsInput, *, with_buildings: bool) -> Counts:
    p = inp.params
    raster = BlockRaster.of(inp.block, p.res_m)
    node = -np.ones(raster.shape, dtype=np.int64)
    node[raster.inside] = np.arange(int(raster.inside.sum()))
    rc = home_cells(raster, inp.block)
    empty = np.full(raster.shape, np.nan, dtype=np.float32)
    empty[raster.inside] = 0.0
    band_mask = raster.inside & (np.nan_to_num(raster.edge, nan=np.inf) <= 1.5 * p.res_m)
    if len(rc) == 0 or not band_mask.any():
        return Counts(raster=raster, egress=empty, pairs=empty.copy())
    homes = node[rc[:, 0], rc[:, 1]]
    band = node[band_mask]
    src, scale = homes, 1.0
    if p.max_sources < len(homes):
        src = homes[np.random.default_rng(p.seed).choice(len(homes), p.max_sources,
                                                          replace=False)]
        scale = len(homes) / p.max_sources
    if with_buildings:
        g = build_graph(raster.inside, raster.clearance, p.res_m, p.r0_m)
    else:
        g = build_graph(raster.inside, np.where(raster.inside, NO_BUILDINGS, np.nan),
                        p.res_m, 0.0)
    bend = bend_table(p.bend_lambda)
    c1 = egress_counts(g, bend, homes, band)
    c2 = pair_counts(g, bend, homes, src, inp.workers)
    return Counts(raster=raster,
                  egress=to_grid(c1, g, raster.inside).astype(np.float32),
                  pairs=to_grid(c2 * scale, g, raster.inside).astype(np.float32))


def _observed_impl(inp: CountsInput) -> Counts:
    return _count(inp, with_buildings=True)


def _prior_impl(inp: CountsInput) -> Counts:
    return _count(inp, with_buildings=False)


def observed(block: Block, params: CountParams, workers: int) -> Counts:
    return derive(_observed_impl, CountsInput(block, params, workers))


def prior(block: Block, params: CountParams, workers: int) -> Counts:
    return derive(_prior_impl, CountsInput(block, params, workers))
```

Bit-identity notes for the implementer: the research code scaled `o2 * scale` AFTER `to_field` on
the float64 counts and then cast to float32 -- `to_grid(c2 * scale, ...)` multiplies the (N,)
vector first; the per-element product is the same float64 value either way, so the cast result is
identical. The prior's graph uses `r0 = 0.0` exactly as `llr.oe_arrays`. If `to_grid` returns
float64, `.astype(np.float32)` reproduces the cache's rounding.

`contrast.py`:

```python
# src/reblock/methods/betweenness/contrast.py
"""How route counts become a desire field: a Strategy, chosen in config.

RawShare  -- egress and all-pairs counts, each as a share of its own maximum, summed.
PriorDeviance -- the signed root Poisson deviance of the observed total against the no-buildings
                 prior: what the buildings CHANNEL, with centrality and street approaches divided
                 out. Ranks worse as a picture, generates better (it spreads the network to the
                 detours the buildings force)."""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from reblock.contracts import Block
from reblock.derive_graph import config_identity
from reblock.methods.betweenness.counts import CountParams, observed, prior
from reblock.methods.betweenness.raster import BlockRaster


def raw_share_field(o1: NDArray[np.float32], o2: NDArray[np.float32]) -> NDArray[np.float64]:
    a, b = o1.astype(np.float64), o2.astype(np.float64)
    # nanmax: the grids are NaN outside the block, and .max() would make every cell NaN.
    return a / max(float(np.nanmax(a)), 1e-12) + b / max(float(np.nanmax(b)), 1e-12)


def deviance_field(o1: NDArray[np.float32], o2: NDArray[np.float32], e1: NDArray[np.float32],
                   e2: NDArray[np.float32], floor: float) -> NDArray[np.float64]:
    o = o1.astype(np.float64) + o2.astype(np.float64)
    e = np.maximum(e1.astype(np.float64) + e2.astype(np.float64), floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2.0 * (np.where(o > 0, o * np.log(o / e), 0.0) - (o - e))
    out: NDArray[np.float64] = np.sign(o - e) * np.sqrt(np.maximum(dev, 0.0))
    return out


@runtime_checkable
class FieldContrast(Protocol):
    @property
    def identity(self) -> Hashable: ...

    def field(self, block: Block, params: CountParams,
              workers: int) -> tuple[BlockRaster, NDArray[np.float64]]: ...


@dataclass(frozen=True)
class RawShare:
    @property
    def identity(self) -> Hashable:
        return config_identity(self)

    def field(self, block: Block, params: CountParams,
              workers: int) -> tuple[BlockRaster, NDArray[np.float64]]:
        o = observed(block, params, workers)
        return o.raster, raw_share_field(o.egress, o.pairs)


@dataclass(frozen=True)
class PriorDeviance:
    floor: float                     # the measured value is 1e-9

    @property
    def identity(self) -> Hashable:
        return config_identity(self)

    def field(self, block: Block, params: CountParams,
              workers: int) -> tuple[BlockRaster, NDArray[np.float64]]:
        o, e = observed(block, params, workers), prior(block, params, workers)
        return o.raster, deviance_field(o.egress, o.pairs, e.egress, e.pairs, self.floor)
```

Note: `deviance_field`'s `sign(o - e)` uses the FLOORED `e`, exactly as the measured
`gen_study.llr_field`; a cell with O = E = 0 therefore reads -4.5e-5, not 0 (the test allows 1e-4).
`config_identity` on a field-less dataclass must return a class-specific key -- verify with
`assert RawShare().identity != PriorDeviance(1e-9).identity`; if `config_identity` returns `()` for
no fields, return `(type(self).__name__,)` explicitly in `RawShare.identity` instead.

- [ ] **Step 4: Run tests, mypy, ruff.**
- [ ] **Step 5: Commit** -- `git commit -m "betweenness: observed and prior route counts, derive-cached; raw and deviance contrasts"`

---

### Task 5: Zhang-Suen thinning, equal to scikit-image's

**Files:**
- Create: `src/reblock/methods/betweenness/thinning.py`
- Test: `tests/methods/betweenness/test_thinning.py`

**Interfaces:**
- Produces: `skeletonize(mask: NDArray[np.bool_]) -> NDArray[np.bool_]`, equal to `skimage.morphology.skeletonize(mask)` (scikit-image 0.26, 2-D, default method).

- [ ] **Step 1: Write the failing test** (scikit-image is available to tests through the research feature; `src` must not import it)

```python
# tests/methods/betweenness/test_thinning.py
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize as skimage_skeletonize

from reblock.methods.betweenness.thinning import skeletonize


def test_equals_scikit_image_on_random_blobs_and_lines() -> None:
    rng = np.random.default_rng(0)
    for trial in range(60):
        shape = (int(rng.integers(5, 120)), int(rng.integers(5, 120)))
        noise = rng.random(shape)
        mask = ndimage.gaussian_filter(noise, sigma=float(rng.uniform(0.5, 4))) > 0.5
        if trial % 3 == 0:
            mask |= rng.random(shape) > 0.97          # isolated pixels and specks
        assert np.array_equal(skeletonize(mask), skimage_skeletonize(mask)), trial


def test_edges_empty_and_full() -> None:
    for mask in (np.zeros((7, 9), bool), np.ones((7, 9), bool), np.ones((1, 5), bool),
                 np.eye(12, dtype=bool)):
        assert np.array_equal(skeletonize(mask), skimage_skeletonize(mask))
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** -- a port of scikit-image v0.26.0
`src/skimage/morphology/_skeletonize_various_cy.pyx::_fast_skeletonize` (BSD-3-Clause; keep the notice):

```python
# src/reblock/methods/betweenness/thinning.py
"""Zhang-Suen thinning, ported from scikit-image 0.26.0's `_fast_skeletonize`
(src/skimage/morphology/_skeletonize_various_cy.pyx), so `src` needs no scikit-image at runtime.
Pinned EQUAL to `skimage.morphology.skeletonize` by tests/methods/betweenness/test_thinning.py.

Copyright (C) 2019, the scikit-image team. BSD-3-Clause; see
https://github.com/scikit-image/scikit-image/blob/v0.26.0/LICENSE.txt"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reblock._jit import njit

# One entry per 8-neighbourhood; 1 and 3 are removed in the first pass, 2 and 3 in the second.
_LUT = np.array([0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0,
                 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0,
                 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 2, 0, 0, 0, 3, 0, 2, 2, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
                 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0,
                 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0,
                 2, 0, 0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3,
                 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0,
                 2, 0, 0, 0], dtype=np.uint8)


@njit
def _thin(skeleton: NDArray[np.uint8], lut: NDArray[np.uint8]) -> NDArray[np.uint8]:
    nrows, ncols = skeleton.shape
    cleaned = skeleton.copy()
    removed = True
    while removed:
        removed = False
        for pass_num in range(2):
            first = pass_num == 0
            for row in range(1, nrows - 1):
                for col in range(1, ncols - 1):
                    if skeleton[row, col]:
                        nb = lut[skeleton[row - 1, col - 1] + 2 * skeleton[row - 1, col]
                                 + 4 * skeleton[row - 1, col + 1] + 8 * skeleton[row, col + 1]
                                 + 16 * skeleton[row + 1, col + 1] + 32 * skeleton[row + 1, col]
                                 + 64 * skeleton[row + 1, col - 1] + 128 * skeleton[row, col - 1]]
                        if nb == 0:
                            continue
                        if nb == 3 or (nb == 1 and first) or (nb == 2 and not first):
                            cleaned[row, col] = 0
                            removed = True
            skeleton[:, :] = cleaned[:, :]
    return skeleton


def skeletonize(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask
    out: NDArray[np.bool_] = _thin(padded, _LUT)[1:-1, 1:-1].astype(bool)
    return out
```

Numba integer note: `skeleton[...]` is uint8; `2 * skeleton[...]` may wrap in uint8 arithmetic under
numba. If the equality test fails, cast each term: `np.int64(skeleton[r, c])`. The test decides.

- [ ] **Step 4: Run tests (both must be exact), mypy, ruff.**
- [ ] **Step 5: Commit** -- `git commit -m "betweenness: Zhang-Suen thinning, ported from scikit-image and pinned equal to it"`

---

### Task 6: Ridges and the `BetweennessDesire` source

**Files:**
- Create: `src/reblock/methods/betweenness/ridges.py`
- Create: `src/reblock/methods/betweenness/source.py`
- Modify: `src/reblock/methods/betweenness/__init__.py` (exports)
- Test: `tests/methods/betweenness/test_source.py`

**Interfaces:**
- Consumes: Tasks 2-5; `reblock.methods.desire_lines.{DesireField, WeightedLines, DesireLineSource}`.
- Produces:
  - `ridge_desire(raster: BlockRaster, field: NDArray[np.float64], quantiles: tuple[float, ...], crs: CRS) -> DesireField`.
  - `BetweennessDesire` (frozen): `res_m: float, r0_m: float, bend_lambda: float, max_sources: int, seed: int, quantiles: tuple[float, ...], contrast: FieldContrast, workers: int`; `identity = config_identity(self, exempt=frozenset({"workers"}))`; `desire_field(block) -> DesireField`; `__post_init__` validates (res_m > 0, r0_m >= 0, bend_lambda >= 0, max_sources >= 1, workers >= 1, quantiles a non-empty strictly increasing tuple in (0, 1)).
  - `reblock.methods.betweenness` exports `BetweennessDesire, RawShare, PriorDeviance`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/methods/betweenness/test_source.py
from __future__ import annotations

import dataclasses

import pytest

from reblock.methods.betweenness import BetweennessDesire, PriorDeviance, RawShare
from reblock.methods.desire_lines import DesireLineSource
from tests.block_fixtures import no_buildings
from tests.scoring_fixtures import _block_1808

Q = (0.80, 0.85, 0.90, 0.95, 0.98)


def _src(**kw: object) -> BetweennessDesire:
    base = dict(res_m=1.0, r0_m=2.0, bend_lambda=50.0, max_sources=400, seed=0, quantiles=Q,
                contrast=RawShare(), workers=1)
    base.update(kw)
    return BetweennessDesire(**base)  # type: ignore[arg-type]


def test_it_is_a_desire_line_source_with_one_group_per_quantile() -> None:
    src = _src()
    assert isinstance(src, DesireLineSource)
    field = src.desire_field(_block_1808())
    assert len(field.groups) == len(Q)
    assert all(abs(g.weight - 1 / len(Q)) < 1e-12 for g in field.groups)
    assert field.n_lines > 0
    # nested levels: the top level's ridge is shorter than the bottom level's
    lens = [float(g.lines.length.sum()) for g in field.groups]
    assert lens[0] > lens[-1] > 0


def test_a_block_with_no_buildings_has_no_desire_instead_of_crashing() -> None:
    b = _block_1808()
    empty = dataclasses.replace(b, building_geometries=no_buildings(b.crs))
    assert _src().desire_field(empty).groups == ()


def test_the_contrast_strategy_runs_too() -> None:
    assert len(_src(contrast=PriorDeviance(floor=1e-9)).desire_field(_block_1808()).groups) == 5


def test_identity_covers_every_setting_but_workers() -> None:
    a = _src()
    assert a.identity == _src(workers=8).identity
    for change in (dict(res_m=0.5), dict(r0_m=1.0), dict(bend_lambda=5.0), dict(max_sources=100),
                   dict(seed=1), dict(quantiles=(0.8, 0.9)), dict(contrast=PriorDeviance(1e-9))):
        assert _src(**change).identity != a.identity, change


@pytest.mark.parametrize("bad", [dict(quantiles=()), dict(quantiles=(0.9, 0.8)),
                                 dict(quantiles=(0.0, 0.5)), dict(res_m=0.0),
                                 dict(max_sources=0), dict(workers=0)])
def test_invalid_settings_raise_at_construction(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _src(**bad)
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement.**

```python
# src/reblock/methods/betweenness/ridges.py
"""A field's ridges as desire lines: the skeleton of each top-quantile level set, one group per
level, weights summing to 1 -- so a point's demand is the share of levels whose ridge runs within
the reblocker's corridor of it."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
from pyproj import CRS
from shapely.geometry import LineString

from reblock.methods.betweenness.raster import BlockRaster
from reblock.methods.betweenness.thinning import skeletonize
from reblock.methods.desire_lines import DesireField, WeightedLines


def _segments(skel: NDArray[np.bool_], raster: BlockRaster) -> list[LineString]:
    r, c = np.nonzero(skel)
    on = set(zip(r.tolist(), c.tolist(), strict=True))
    x0, y0, res = raster.x0, raster.y0, raster.res
    out = []
    for i, j in on:                               # set iteration: the measured order, kept
        for di, dj in ((0, 1), (1, -1), (1, 0), (1, 1)):
            if (i + di, j + dj) in on:
                out.append(LineString([(x0 + res * j, y0 + res * i),
                                       (x0 + res * (j + dj), y0 + res * (i + di))]))
    return out


def ridge_desire(raster: BlockRaster, field: NDArray[np.float64],
                 quantiles: tuple[float, ...], crs: CRS) -> DesireField:
    free = raster.free & np.isfinite(field)
    vals = field[free]
    # No free cell, or a field with no contrast at all (no homes, or no street to reach: every
    # count is 0): there is no ridge to follow, so there is no desire. Thresholding a constant
    # field would instead return the whole free area as one "ridge".
    if vals.size == 0 or float(vals.max()) == float(vals.min()):
        return DesireField(groups=())
    groups = []
    for q in quantiles:
        skel = skeletonize(free & (field >= np.quantile(vals, q)))
        groups.append(WeightedLines(lines=gpd.GeoDataFrame(geometry=_segments(skel, raster),
                                                           crs=crs),
                                    weight=1.0 / len(quantiles)))
    return DesireField(groups=tuple(groups))
```

```python
# src/reblock/methods/betweenness/source.py
"""`BetweennessDesire`: a `DesireLineSource` from curvature-aware repelled betweenness.

Where many homes' best routes -- to the street, and to each other -- run, with routes that pay for
turning and are repelled by buildings. Measured as a road GENERATOR (demand_greedy toward its
ridges; docs/superpowers/notes/2026-09-24-roadless-fields-and-the-betweenness-generator.md): it
beats clearance on both lenses, and looped it beats clearance_looped, on 220 small and 36 large
blocks; cycle_native still leads Lens A on large blocks."""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

from reblock.contracts import Block
from reblock.derive_graph import config_identity
from reblock.methods.betweenness.contrast import FieldContrast
from reblock.methods.betweenness.counts import CountParams
from reblock.methods.betweenness.ridges import ridge_desire
from reblock.methods.desire_lines import DesireField


@dataclass(frozen=True)
class BetweennessDesire:
    res_m: float                     # raster resolution; 1.0 is the large-block operating point
    r0_m: float                      # repulsion length: cost density 1 + (r0 / clearance)^2
    bend_lambda: float               # turning cost lam * dtheta^2 / 2 m
    max_sources: int                 # all-pairs sources sampled when homes exceed this
    seed: int                        # the sample's seed
    quantiles: tuple[float, ...]     # the nested ridge levels
    contrast: FieldContrast          # counts -> field: RawShare or PriorDeviance
    workers: int                     # fork pool for the all-pairs pass; changes nothing

    def __post_init__(self) -> None:
        q = self.quantiles
        if not (isinstance(q, tuple) and q and all(0.0 < a < 1.0 for a in q)
                and all(a < b for a, b in zip(q, q[1:]))):
            raise ValueError(f"quantiles must be a non-empty increasing tuple in (0, 1), got {q!r}")
        if self.res_m <= 0 or self.r0_m < 0 or self.bend_lambda < 0:
            raise ValueError(f"res_m > 0, r0_m >= 0, bend_lambda >= 0 required: {self!r}")
        if self.max_sources < 1 or self.workers < 1:
            raise ValueError(f"max_sources and workers must be >= 1: {self!r}")

    @property
    def identity(self) -> Hashable | None:
        return config_identity(self, exempt=frozenset({"workers"}))

    def desire_field(self, block: Block) -> DesireField:
        params = CountParams(res_m=self.res_m, r0_m=self.r0_m, bend_lambda=self.bend_lambda,
                             max_sources=self.max_sources, seed=self.seed)
        raster, field = self.contrast.field(block, params, self.workers)
        return ridge_desire(raster, field, self.quantiles, block.crs)
```

`__init__.py`:

```python
"""Curvature-aware repelled betweenness as a desire field (see `source.py`)."""
from reblock.methods.betweenness.contrast import FieldContrast, PriorDeviance, RawShare
from reblock.methods.betweenness.source import BetweennessDesire

__all__ = ["BetweennessDesire", "FieldContrast", "PriorDeviance", "RawShare"]
```

If the "no buildings" test shows `observed` failing earlier than `ridge_desire` (e.g. an empty
STRtree), the fix belongs in `BlockRaster.of` / `_count` (already guarded) -- never a try/except.

- [ ] **Step 4: Run tests, mypy, ruff.**
- [ ] **Step 5: Commit** -- `git commit -m "betweenness: ridge desire lines and the BetweennessDesire source"`

---

### Task 7: Presets and wiring

**Files:**
- Create: `conf/betweenness.yaml`
- Modify: `conf/config.yaml`, `conf/compare_config.yaml` (defaults `- betweenness`; four `all_methods` entries)
- Create: `conf/method/betweenness_tree.yaml`, `betweenness_tree_contrast.yaml`, `betweenness_looped.yaml`, `betweenness_looped_contrast.yaml`
- Modify: `tests/test_config_identity.py` (`CANNOT_CHANGE_OUTPUT` += `"BetweennessDesire.workers"`)
- Modify: `src/reblock/presets.py` (register the `cpu_count` resolver), `scripts/consensus_matrix.py` (drop its registration)
- Test: `tests/methods/betweenness/test_presets.py`

**Interfaces:**
- Consumes: `reblock.methods.betweenness.{BetweennessDesire, RawShare, PriorDeviance}`, `DemandGreedyReblocker`, `LoopClosureRefiner`.
- Produces: `${betweenness.raw}` and `${betweenness.contrast}` desire-source nodes; `method=betweenness_tree|betweenness_tree_contrast|betweenness_looped|betweenness_looped_contrast`; the same four names in `compare_config.all_methods`.

- [ ] **Step 1: Write the failing test**

```python
# tests/methods/betweenness/test_presets.py
"""Each preset builds the method it names, with the betweenness source it names -- never the
config-wide `desire_source` default (osm), which would build a different method silently."""
from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from reblock.methods.betweenness import BetweennessDesire, PriorDeviance, RawShare
from reblock.methods.demand_greedy import DemandGreedyReblocker
from reblock.methods.loop_closure import LoopClosureRefiner
from reblock.presets import load_method, load_methods

CASES = [("betweenness_tree", False, RawShare), ("betweenness_tree_contrast", False, PriorDeviance),
         ("betweenness_looped", True, RawShare), ("betweenness_looped_contrast", True, PriorDeviance)]


def _dg(m: object, looped: bool) -> DemandGreedyReblocker:
    if looped:
        assert isinstance(m, LoopClosureRefiner)
        assert (m.budget_frac, m.search_radius_m) == (0.30, 60.0)   # the measured loop settings
        m = m.base
    assert isinstance(m, DemandGreedyReblocker)
    return m


@pytest.mark.parametrize("name,looped,contrast", CASES)
def test_method_preset_builds_what_it_names(name: str, looped: bool, contrast: type) -> None:
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="config", overrides=[f"method={name}", "shapefile=x"])
    src = _dg(load_method(cfg.method), looped).desire_source
    assert isinstance(src, BetweennessDesire) and isinstance(src.contrast, contrast)
    assert (src.res_m, src.r0_m, src.bend_lambda, src.max_sources, src.seed) == (1.0, 2.0, 50.0, 400, 0)
    assert src.quantiles == (0.80, 0.85, 0.90, 0.95, 0.98)


def test_compare_config_lists_all_four() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config", overrides=["shapefile=x"])
    registry = load_methods(cfg.all_methods)
    for name, looped, contrast in CASES:
        src = _dg(registry[name], looped).desire_source
        assert isinstance(src, BetweennessDesire) and isinstance(src.contrast, contrast)
```

- [ ] **Step 2: Run to verify failure** (`Could not find 'method/betweenness_tree'`).

- [ ] **Step 3: Write the configs.**

`conf/betweenness.yaml`:

```yaml
# @package betweenness
# The betweenness desire sources every betweenness_* method preset routes toward: curvature-aware
# repelled betweenness (reblock.methods.betweenness). Composed into both root configs like
# `permeability`, so a preset interpolates ${betweenness.raw} / ${betweenness.contrast} instead of
# the config-wide `desire_source` default (osm), which would build a different method silently.
# Every value is the measured large-block operating point (docs/superpowers/notes/
# 2026-09-24-roadless-fields-and-the-betweenness-generator.md).
raw:
  _target_: reblock.methods.betweenness.BetweennessDesire
  res_m: 1.0
  r0_m: 2.0
  bend_lambda: 50.0
  max_sources: 400
  seed: 0
  quantiles: {_target_: builtins.tuple, _args_: [[0.80, 0.85, 0.90, 0.95, 0.98]]}
  contrast: {_target_: reblock.methods.betweenness.RawShare}
  # The all-pairs pass forks one worker per core. Set 1 for runs that already fork per block: a
  # daemonic pool worker may not fork again, and the source raises rather than silently serialize.
  workers: ${cpu_count:}
contrast:
  _target_: reblock.methods.betweenness.BetweennessDesire
  res_m: 1.0
  r0_m: 2.0
  bend_lambda: 50.0
  max_sources: 400
  seed: 0
  quantiles: {_target_: builtins.tuple, _args_: [[0.80, 0.85, 0.90, 0.95, 0.98]]}
  contrast: {_target_: reblock.methods.betweenness.PriorDeviance, floor: 1.0e-9}
  workers: ${cpu_count:}
```

`${cpu_count:}` is today registered only in `scripts/consensus_matrix.py:94`
(`OmegaConf.register_new_resolver("cpu_count", os.cpu_count, replace=True)`), so a preset using it
would fail to resolve anywhere else. MOVE that line (with its `import os` / `OmegaConf` imports as
needed) to module level in `src/reblock/presets.py` -- the one instantiate site, imported by the
script too -- and delete it from the script. One registration, where every config is built.

In `conf/config.yaml` and `conf/compare_config.yaml` defaults, after `- permeability`: `- betweenness`.

`conf/method/betweenness_tree.yaml`:

```yaml
# demand_greedy routed toward curvature-aware betweenness ridges (raw counts). Frontier point:
# on the (Lens A, Lens B) frontier in 6 of 36 large blocks; beats clearance on both lenses on 220
# small blocks and on the large ones. Same builder settings as conf/method/demand_greedy.yaml.
_target_: reblock.methods.demand_greedy.DemandGreedyReblocker
desire_source: ${betweenness.raw}
substrate: ${substrate}
buffer_m: 3.0
eps: 0.1
gamma: 1.0
depth_target: 2
max_roads: 400
road_width_m: 7.0
```

`conf/method/betweenness_tree_contrast.yaml`: identical except the comment (frontier 1/36 large; best
Lens A of all the generators on the 220 small blocks, +0.032 over clearance) and `desire_source: ${betweenness.contrast}`.

`conf/method/betweenness_looped.yaml`:

```yaml
# The loop-closing refiner around betweenness_tree. The loop settings are the region-scale values
# the frontier study measured (conf/example/explore.yaml's clearance_looped tuning), not
# conf/method/loop_closure.yaml's block-scale 0.12 / 45. On the frontier in 11 of 36 large blocks;
# beats clearance_looped on both lenses in 94% / 78% of them; cycle_native leads Lens A by 0.015,
# greedy_arterial leads Lens B by 0.008.
_target_: reblock.methods.loop_closure.LoopClosureRefiner
base:
  _target_: reblock.methods.demand_greedy.DemandGreedyReblocker
  desire_source: ${betweenness.raw}
  substrate: ${substrate}
  buffer_m: 3.0
  eps: 0.1
  gamma: 1.0
  depth_target: 2
  max_roads: 400
  road_width_m: 7.0
budget_frac: 0.30
min_bridges_per_m: 0.01
max_loops: 400
min_loop_len_m: 40.0
search_radius_m: 60.0
snap_lam: 2.0
max_candidates: 1500
road_width_m: 7.0
```

`conf/method/betweenness_looped_contrast.yaml`: identical except comment (frontier 8/36) and
`desire_source: ${betweenness.contrast}`.

In `conf/compare_config.yaml` `all_methods`, after `demand_greedy`, add four entries
`betweenness_tree`, `betweenness_tree_contrast`, `betweenness_looped`,
`betweenness_looped_contrast` with exactly the same content as the four method files (flow-style or
block-style, matching the neighbours).

In `tests/test_config_identity.py`: `CANNOT_CHANGE_OUTPUT = {"GreedyArterialReblocker.workers",
"ShortlistEngine.threads", "BetweennessDesire.workers"}`.

- [ ] **Step 4: Run** `pixi run pytest tests/methods/betweenness tests/test_presets.py tests/test_config_identity.py -q -p no:cacheprovider --no-cov`, then mypy and ruff. The identity test perturbs every numeric field of the four new all_methods entries (quantiles is a tuple, not perturbed -- the Task 6 identity test covers it).
- [ ] **Step 5: Fault-inject** the preset test: temporarily change `betweenness_tree.yaml`'s `desire_source` to `${desire_source}` and confirm `test_method_preset_builds_what_it_names[betweenness_tree...]` fails (it would build `OSMDesireLines`); revert.
- [ ] **Step 6: Commit** -- `git commit -m "betweenness: four method presets, one per measured operating point"`

---

### Task 8: Bit-identity against the research code (verification, not a repo test)

**Files:** none in the repo. A scratch script in the session scratchpad.

- [ ] **Step 1:** For three of the 220 recipients (`ZAF.9.1.2_1_3771`, `ZAF.9.3.1_1_40972`, and the
largest by building count in `data/benchmarks/consensus_matrix.parquet`), build the block at the
footprint tier exactly as `gaps/study220.py::build_blocks` does, and assert that
`observed(block, CountParams(0.5, 2.0, 50.0, 10_000, 0), 8)` and `prior(...)` equal the scratch
cache `gaps/fields220/<id>.npz` arrays `O1, O2, E1, E2` with `np.array_equal(..., equal_nan=True)`.
- [ ] **Step 2:** For Bloekombos (`ZAF.9.3.1_1_5810`, `+example=explore`, `buildings=footprints`),
assert the same against `gaps/fields220/ZAF.9.3.1_1_5810.npz` with `CountParams(1.0, 2.0, 50.0, 400, 0)`.
- [ ] **Step 3:** On the three recipients, assert the production `betweenness_tree(_contrast)`
roads equal the scratch `dg_c3` / `dg_llr` roads (rerun `gaps/gen_study.py`'s arms in-process;
compare with `GeoSeries.geom_equals_exact(tolerance=0)` row by row after sorting by WKT). Note
the source's `res_m` must be overridden to 0.5 and `max_sources` to 10_000 for this comparison.
- [ ] **Step 4:** Record timings (fields on Bloekombos at `workers=40`) and the result in the Task 9 note.
Any mismatch is a stop: find the divergent expression and fix the PORT, never the check.

---

### Task 9: The findings note, full verification, merge

**Files:**
- Create: `docs/superpowers/notes/2026-09-24-roadless-fields-and-the-betweenness-generator.md`

- [ ] **Step 1: Write the note** (the research trail, with numbers, from this session's results):
  the owner's path field (best ranking on small blocks, saturates on large ones); repelled
  betweenness (best line placement; July's pinch-point failure gone with repulsion); the soft
  path field (negative: extensive path entropy; X/Y count checks fail; ties the hard field at best);
  curvature-aware betweenness (best overall as a picture); the contrast vs the no-buildings prior
  (sharper picture, worse ranking, better generator); ground truths (OSM edge-biased on large
  blocks; Microsoft RoadDetections, ODbL, covers cores but is mostly perimeter streets on small
  blocks; Google Maps must not be extracted); the generator on 220 small blocks, held-out
  replication, and the 36-block large frontier (with greedy_arterial's final numbers); why
  `flow_paths` (2026-07-29, refuted as a mimicry model) is a different thing (it emits flow edges
  as roads; this uses the field as a prior for a coverage-guaranteeing builder); what shipped
  (four presets) and what did not (the path field, the soft field: scratch only, recorded here);
  Task 8's bit-identity result and timings.
- [ ] **Step 2:** Run the full suite `pixi run pytest -q -p no:cacheprovider`, `pixi run mypy`,
`pixi run ruff check .` -- all green.
- [ ] **Step 3:** Commit the note; merge `betweenness-desire` into main with `--no-ff`; push.
