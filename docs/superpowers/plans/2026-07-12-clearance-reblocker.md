# Clearance Reblocker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `ClearanceReblocker` Method — a greedy least-cost-path reblocker whose single physical knob (`repulsion`) spans aspirational straight roads → buildable Voronoi-following roads — wire it into the compare/run pipeline, and ship a true-end-to-end 5-repulsion example gallery on an auto-detected deep Cape Town region.

**Architecture:** Each road is a least-cost path from the current deepest parcel to the road+street network, on an 8-connected grid whose edge weights come from a cost field that repels from building points (`edge_weight = length·[(1−t) + t/clearance]`, `t = sigmoid(repulsion)`). Access depth is maintained incrementally (a road only lowers depth) so large regions stay fast. The Method follows the existing `Method` protocol (`identity` + `propose`) and slots into `compare`/`run` alongside dijkstra/arterial. Spec: `docs/superpowers/specs/2026-07-12-clearance-reblocker-design.md`.

**Tech Stack:** Python, numpy, scipy (`sparse.csr_matrix`, `csgraph.dijkstra`, `spatial.cKDTree`), shapely 2.x (`STRtree`, `contains_xy`, `points`, `dwithin`, `nearest_points`), geopandas, Hydra config, pytest, pixi (`pixi run check` = ruff + mypy --strict + pytest).

## Global Constraints

- **Method protocol:** `ClearanceReblocker` is a `@dataclass` exposing `identity` (a hashable tuple) and `propose(self, block: Block, prior: Proposal | None = None) -> Proposal`. `propose` is side-effect-free — **no global RNG use** (mirror `test_dijkstra.py::test_dijkstra_propose_is_deterministic_and_leaves_rng_untouched`).
- **Defaults (verbatim from spec):** `repulsion = 0.0`, `depth_target = 2`, `res = 1.5`, `max_roads = 400`.
- **Knob:** user-facing `repulsion` is the logit `s`; internal blend `t = sigmoid(s) ∈ (0,1)`. `s → −∞` ⇒ straight (aspirational); `s = 0` ⇒ balanced; `s → +∞` ⇒ Voronoi-following (buildable).
- **Cost field:** `edge_weight = edist · 0.5 · (node_cost[u] + node_cost[v])`, `node_cost = (1−t) + t/clearance`, `clearance = max(dist_to_nearest_building − radius, 0) + ε` with `ε = 0.3`. Radii default to 0 (plain clearance == exact Euclidean/Voronoi distance field); nonzero radii (weighted footprints) are opt-in and exercised only by a unit test, **not** wired into `propose` (out of scope per spec).
- **Determinism:** two `propose` calls on the same block return **WKT-identical** roads. Argmax-depth ties broken by ascending `parcel_id`. No `Date.now`/RNG.
- **Cache correctness:** `proposal_id` must encode the params (`repulsion`, `depth_target`, `res`) so distinct configs get distinct `Proposal.identity` — otherwise `access_after`/`geometric_after` (keyed on the proposal) collide across repulsion values. Add `src/reblock/methods/clearance.py` to `reblock.derive_graph._DERIVATION_MODULES` so algorithm changes bust the memoized `propose` cache (matches dijkstra/mesh/arterial).
- **No legacy/dual-path code** (owner directive): one algorithm, one code path. No back-compat shims.
- **Quality gate:** `pixi run check` (ruff + mypy --strict + pytest) green before every commit.
- **Commit trailers (verbatim, every commit):**
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
  ```

## File Structure

- **Create** `src/reblock/methods/clearance.py` — the Method + algorithm. One file, focused: grid/cost primitives, the incremental-depth relax, the greedy loop, the `ClearanceReblocker` dataclass.
- **Create** `tests/methods/test_clearance.py` — unit + integration tests.
- **Modify** `src/reblock/derive_graph.py` — add `methods/clearance.py` to `_DERIVATION_MODULES`.
- **Create** `conf/method/clearance.yaml` — Hydra config for `method=clearance`.
- **Modify** `conf/compare_config.yaml` — register `clearance` in `all_methods`.
- **Create** `examples/clearance-repulsion/generate.py` — reproducible gallery generator (auto-detect region → 5-repulsion sweep → renders).
- **Create** `examples/clearance-repulsion/README.md` + committed PNGs (`before.png`, `region_map.png`, `after_s{-6,-2,0,+2,+6}.png`).

---

### Task 1: Grid, sigmoid, and cost-field primitives

**Files:**
- Create: `src/reblock/methods/clearance.py`
- Test: `tests/methods/test_clearance.py`

**Interfaces:**
- Consumes: `shapely.contains_xy`, `scipy.spatial.cKDTree`.
- Produces (used by Tasks 2–3):
  - `_sigmoid(s: float) -> float`
  - `_build_grid(boundary: Polygon, res: float) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]` returning `(pts, rows, cols, edist)` — an 8-connected grid; `rows`/`cols`/`edist` are symmetric COO edges (both directions) with Euclidean lengths.
  - `_node_clearance(pts: NDArray[np.float64], building_pts: NDArray[np.float64], radii: NDArray[np.float64]) -> NDArray[np.float64]`
  - `_edge_weights(clear: NDArray[np.float64], t: float, rows: NDArray[np.int64], cols: NDArray[np.int64], edist: NDArray[np.float64]) -> NDArray[np.float64]`
  - `_CLEARANCE_EPS: float = 0.3`

- [ ] **Step 1: Write the failing tests**

Create `tests/methods/test_clearance.py`:

```python
import math
from typing import cast

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon

from reblock.methods.clearance import (
    _build_grid,
    _edge_weights,
    _node_clearance,
    _sigmoid,
)


def test_sigmoid_is_bounded_and_symmetric() -> None:
    assert _sigmoid(0.0) == pytest.approx(0.5)
    assert _sigmoid(6.0) == pytest.approx(1.0, abs=1e-2)
    assert _sigmoid(-6.0) == pytest.approx(0.0, abs=1e-2)
    assert _sigmoid(3.0) + _sigmoid(-3.0) == pytest.approx(1.0)
    # never saturates to exactly 0/1 (weights stay finite), and no overflow at extreme s
    assert 0.0 < _sigmoid(-800.0) < _sigmoid(800.0) < 1.0


def test_build_grid_is_8_connected_and_inside_boundary() -> None:
    # contains_xy is strict (excludes the boundary), so a 4x4 box at res=1 gives interior nodes
    # {1,2,3}x{1,2,3}; the center (2,2) is a true interior node with 8 neighbours.
    boundary = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    pts, rows, cols, edist = _build_grid(boundary, 1.0)
    assert len(pts) > 0
    assert all(boundary.contains(Point(p)) for p in pts)   # strict, matches contains_xy
    # edges are symmetric (both directions present) and lengths are 1 or sqrt(2)
    assert len(rows) == len(cols) == len(edist)
    assert set(np.round(np.unique(edist), 6)) <= {1.0, round(math.sqrt(2.0), 6)}
    undirected = {frozenset((int(a), int(b))) for a, b in zip(rows, cols)}
    assert len(undirected) * 2 == len(rows)  # every undirected edge stored both ways
    # an interior node has 8 neighbours
    tree = cKDTree(pts)
    center = int(tree.query([2.0, 2.0])[1])
    assert int((rows == center).sum()) == 8


def test_node_clearance_is_euclidean_when_unweighted() -> None:
    pts = np.array([[0.0, 0.0], [5.0, 0.0]])
    buildings = np.array([[0.0, 0.0]])
    radii = np.zeros(1)
    clear = _node_clearance(pts, buildings, radii)
    # node ON the building -> eps; node 5 away -> 5 + eps
    assert clear[0] == pytest.approx(0.3)
    assert clear[1] == pytest.approx(5.3)


def test_node_clearance_weighted_radius_shrinks_clearance() -> None:
    pts = np.array([[5.0, 0.0]])
    buildings = np.array([[0.0, 0.0]])
    plain = _node_clearance(pts, buildings, np.zeros(1))
    weighted = _node_clearance(pts, buildings, np.array([3.0]))  # radius-3 footprint
    assert weighted[0] < plain[0]
    assert weighted[0] == pytest.approx(5.0 - 3.0 + 0.3)


def test_node_clearance_no_buildings_is_uniform() -> None:
    pts = np.array([[0.0, 0.0], [10.0, 10.0]])
    clear = _node_clearance(pts, np.empty((0, 2)), np.zeros(0))
    assert clear[0] == clear[1]  # uniform -> straight regardless of t


def test_repulsion_bends_the_path_around_buildings() -> None:
    # A straight route (t≈0) crosses a vertical wall of buildings; a repelled route (t≈1)
    # bows away and stays farther from them, at >= the straight length.
    boundary = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    buildings = np.array([[5.0, 3.0], [5.0, 5.0], [5.0, 7.0]])
    pts, rows, cols, edist = _build_grid(boundary, 0.5)
    clear = _node_clearance(pts, buildings, np.zeros(len(buildings)))
    tree = cKDTree(pts)
    src = int(tree.query([5.0, 9.0])[1])
    dst = int(tree.query([5.0, 1.0])[1])

    def route(t: float) -> tuple[float, float]:
        w = _edge_weights(clear, t, rows, cols, edist)
        csr = csr_matrix((w, (rows, cols)), shape=(len(pts), len(pts)))
        _d, pred, _s = dijkstra(csr, indices=[src], return_predecessors=True, min_only=True)
        node, path = dst, [dst]
        while pred[node] >= 0:
            node = int(pred[node])
            path.append(node)
        line = LineString([tuple(pts[k]) for k in path])
        min_clear = min(Point(cast(tuple[float, float], tuple(b))).distance(line) for b in buildings)
        return float(line.length), float(min_clear)

    len_straight, clear_straight = route(_sigmoid(-6.0))
    len_repelled, clear_repelled = route(_sigmoid(6.0))
    assert clear_straight < clear_repelled           # repelled path keeps farther from buildings
    assert len_repelled >= len_straight - 1e-9        # ...at no less than the straight length
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/methods/test_clearance.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'reblock.methods.clearance'`.

- [ ] **Step 3: Write the primitives**

Create `src/reblock/methods/clearance.py`:

```python
"""ClearanceReblocker: greedy least-cost-path reblocker with one physical knob (repulsion)
spanning aspirational straight roads -> buildable Voronoi-following roads.

Each road is a least-cost path from the current deepest parcel to the road+street network, on
an 8-connected grid whose edge weights come from a cost field that repels from building points:
    edge_weight = length * [(1 - t) + t / clearance],   clearance = dist(nearest building) (+ eps)
The user-facing knob is the logit s (`repulsion`); t = sigmoid(s) in (0, 1). s -> -inf: uniform
cost -> the straight line (aspirational, best directness). s -> +inf: hug the max-clearance ridges
= the Voronoi edges (equidistant from the two nearest buildings) = the buildable gaps. Access
depth is maintained incrementally (a road only lowers depth) so large regions stay fast.
See docs/superpowers/specs/2026-07-12-clearance-reblocker-design.md.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from shapely import contains_xy
from shapely.geometry import Polygon
from scipy.spatial import cKDTree

_CLEARANCE_EPS = 0.3       # keeps node cost finite on a grid node sitting on a building point
_NET_TOL_FACTOR = 1.5      # a grid node within res * this of the street seeds the network


def _sigmoid(s: float) -> float:
    """t = sigmoid(s) in (0, 1) -- the logit knob's internal blend weight. Overflow-safe for
    extreme |s| (never returns exactly 0.0 or 1.0, so the cost field stays finite)."""
    if s >= 0.0:
        return 1.0 / (1.0 + math.exp(-s))
    e = math.exp(s)
    return e / (1.0 + e)


def _build_grid(
    boundary: Polygon, res: float
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """8-connected grid of points inside `boundary` at spacing `res`. Returns
    (pts (M,2), rows, cols, edist): `rows`/`cols` are symmetric COO edge endpoints (each
    undirected edge stored both ways) and `edist` their Euclidean lengths (res or res*sqrt2)."""
    minx, miny, maxx, maxy = boundary.bounds
    xs = np.arange(minx, maxx + res, res)
    ys = np.arange(miny, maxy + res, res)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.c_[gx.ravel(), gy.ravel()]
    pts = pts[contains_xy(boundary, pts[:, 0], pts[:, 1])]
    idx = {(round(float(x), 3), round(float(y), 3)): k for k, (x, y) in enumerate(pts)}
    rows: list[int] = []
    cols: list[int] = []
    dist: list[float] = []
    for k, (x, y) in enumerate(pts):
        for dx, dy in ((res, 0.0), (0.0, res), (res, res), (res, -res)):
            nb = idx.get((round(float(x + dx), 3), round(float(y + dy), 3)))
            if nb is not None:
                d = float(np.hypot(dx, dy))
                rows += [k, nb]
                cols += [nb, k]
                dist += [d, d]
    return (
        pts.astype(np.float64),
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(dist, dtype=np.float64),
    )


def _node_clearance(
    pts: NDArray[np.float64], building_pts: NDArray[np.float64], radii: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-node clearance = distance to the nearest building minus that building's exclusion
    radius (0 in the plain case), floored at 0, plus `_CLEARANCE_EPS`. With uniform/zero radii
    this is the exact Euclidean clearance (== the Voronoi distance field); nonzero radii use the
    nearest-building approximation of the additively-weighted (Apollonius) clearance. No
    buildings -> uniform clearance, so the path is straight regardless of t."""
    if len(building_pts) == 0:
        return np.ones(len(pts), dtype=np.float64)
    dist, nearest = cKDTree(building_pts).query(pts)
    clear = np.maximum(dist - radii[nearest], 0.0) + _CLEARANCE_EPS
    return clear.astype(np.float64)


def _edge_weights(
    clear: NDArray[np.float64], t: float,
    rows: NDArray[np.int64], cols: NDArray[np.int64], edist: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Edge weight = length * average node cost, node cost = (1 - t) + t / clearance. t=0 ->
    uniform (straight); t->1 -> cost dominated by 1/clearance (hug the high-clearance gaps)."""
    node_cost = (1.0 - t) + t / clear
    return edist * 0.5 * (node_cost[rows] + node_cost[cols])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/methods/test_clearance.py -x -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Lint + type-check**

Run: `pixi run check`
Expected: ruff + mypy --strict + pytest all green.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/methods/clearance.py tests/methods/test_clearance.py
git commit -m "feat: clearance reblocker grid + cost-field primitives

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

### Task 2: Incremental access-depth relax

**Files:**
- Modify: `src/reblock/methods/clearance.py`
- Test: `tests/methods/test_clearance.py`

**Interfaces:**
- Consumes: `reblock.derive.access.parcel_access_layers`, `reblock.derive.adjacency.parcel_adjacency`, `reblock.derive.access.STREET_TOL`.
- Produces (used by Task 3): `_relax_depth(depth: NDArray[np.float64], adj: list[set[int]], served: Iterable[int]) -> None` — in place; parcels in `served` become depth 1 and the lowering propagates outward along `adj` (BFS); never raises a depth.

- [ ] **Step 1: Write the failing test**

Append to `tests/methods/test_clearance.py`:

```python
import geopandas as gpd
from pyproj import CRS
from shapely import STRtree
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.clearance import _relax_depth

UTM = CRS.from_epsg(32643)


def _column_block(h: int) -> Block:
    """A 1-wide, h-tall column of unit parcels with street frontage only on the bottom edge ->
    access depth 1..h from the street upward. parcel_id == row index (bottom = 0)."""
    polys = [Polygon([(0, j), (1, j), (1, j + 1), (0, j + 1)]) for j in range(h)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(h))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, unary_union(polys))
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs=UTM)
    return Block(block_id="col", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_relax_depth_matches_full_recompute() -> None:
    # A single street-connected road up the column should, incrementally, reproduce exactly what
    # parcel_access_layers computes from scratch for that road.
    block = _column_block(6)
    geoms = list(block.parcels.geometry)
    adj = parcel_adjacency(geoms, STREET_TOL)
    depth = parcel_access_layers(block, None, adj=adj).to_numpy().astype(float)
    assert list(depth) == [1, 2, 3, 4, 5, 6]  # sanity: deep column

    road = gpd.GeoDataFrame(geometry=[LineString([(0.5, 0.0), (0.5, 6.0)])], crs=UTM)
    served = [int(p) for p in STRtree(geoms).query(
        road.geometry.iloc[0], predicate="dwithin", distance=STREET_TOL)]
    _relax_depth(depth, adj, served)

    naive = parcel_access_layers(block, road, adj=adj).to_numpy().astype(float)
    assert list(depth) == list(naive)
    assert max(depth) == 1.0  # every parcel now fronts the street-connected road
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/methods/test_clearance.py::test_relax_depth_matches_full_recompute -x -q`
Expected: FAIL with `ImportError: cannot import name '_relax_depth'`.

- [ ] **Step 3: Add `_relax_depth`**

Add to the imports at the top of `src/reblock/methods/clearance.py`:

```python
from collections import deque
from collections.abc import Iterable
```

Add the function (place it after `_edge_weights`):

```python
def _relax_depth(depth: NDArray[np.float64], adj: list[set[int]], served: Iterable[int]) -> None:
    """In place: given parcels `served` now front a street-connected road (depth 1), lower
    `depth` and propagate depth[j] = depth[i] + 1 outward along parcel adjacency `adj` (BFS),
    never raising a value. Equals a full `parcel_access_layers` recompute because a road only
    adds street frontage (parcel adjacency is unchanged), so the post-road depth is a BFS from
    (original street seeds) union (newly served parcels)."""
    q: deque[int] = deque()
    for p in served:
        if depth[p] > 1.0:
            depth[p] = 1.0
            q.append(int(p))
    while q:
        i = q.popleft()
        di = depth[i]
        for j in adj[i]:
            if depth[j] > di + 1.0:
                depth[j] = di + 1.0
                q.append(j)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/methods/test_clearance.py::test_relax_depth_matches_full_recompute -x -q`
Expected: PASS.

- [ ] **Step 5: Lint + type-check**

Run: `pixi run check`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/methods/clearance.py tests/methods/test_clearance.py
git commit -m "feat: clearance incremental access-depth relax (== full recompute)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

### Task 3: Greedy loop + `ClearanceReblocker` Method

**Files:**
- Modify: `src/reblock/methods/clearance.py`
- Test: `tests/methods/test_clearance.py`

**Interfaces:**
- Consumes (Tasks 1–2): `_sigmoid`, `_build_grid`, `_node_clearance`, `_edge_weights`, `_relax_depth`, `_NET_TOL_FACTOR`.
- Consumes: `reblock.contracts.{Block, Proposal}`, `reblock.derive.access.{STREET_TOL, parcel_access_layers}`, `reblock.derive.adjacency.parcel_adjacency`.
- Produces (used by Tasks 4–5):
  - `_greedy_reblock(block: Block, *, t: float, res: float, depth_target: int, max_roads: int, radii: NDArray[np.float64]) -> tuple[gpd.GeoDataFrame, dict[str, object]]`
  - `ClearanceReblocker` dataclass: `repulsion: float = 0.0`, `depth_target: int = 2`, `res: float = 1.5`, `max_roads: int = 400`; `identity` property → `("clearance", repulsion, depth_target, res, max_roads)`; `propose(block, prior=None) -> Proposal`.

Design notes the implementer must honour:
- **Early exit:** compute adjacency + depth first; if `depth.max() <= depth_target`, return an empty roads GeoDataFrame (no grid needed).
- **Deepest parcel, deterministic:** `worst = the parcel among argmax(depth) with the smallest parcel_id`.
- **Road geometry:** `parcel representative point → grid least-cost path nodes → (conditionally) nearest actual street point`. Append the street point **only when the path terminates within `res*_NET_TOL_FACTOR` of the street** (bridging the sub-`res` grid→street gap). When the path instead terminates on a prior road's grid node (already street-connected via `net`), do **not** append a street point — that would draw a spurious segment to a far street. Drop consecutive-duplicate coords; skip any road with `< 2` distinct coords.
- **Grid-unroutable parcel — a discretization artifact, NOT parcel-graph reachability:** the greedy's Dijkstra runs on the *grid* graph (the cost-field lattice), a third graph distinct from both the parcel/Voronoi-adjacency graph (always connected → `parcel_access_layers` gives every parcel a finite depth) and the road network. A grid node is unroutable only when a sub-`res` pinch in a concave boundary severs its lattice component from every street seed in `net` (a convex-hull region can't pinch, so this never fires there; lower `res` dissolves it otherwise). If the deepest parcel's nearest grid node is grid-unroutable (non-finite Dijkstra distance) or the road degenerates to `< 2` coords, set its depth to `-inf` so it is excluded from further selection, increment `n_grid_unreachable`, and continue — do **not** break the whole loop (other deep parcels are still routable). Name the param `grid_unreachable` (NOT `unreachable` — that is `DijkstraReblocker`'s distinct parcel-graph notion), and compute `max_depth_after` from one honest final `parcel_access_layers` recompute so a stranded parcel is surfaced, never masked.
- **After adding a road:** `served = parcels within STREET_TOL of the road (STRtree dwithin)`; `_relax_depth`; extend `net` with the road's grid path nodes so later roads can join it.
- **`proposal_id`** encodes params: `f"clearance:r{repulsion:g}:d{depth_target}:res{res:g}"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/methods/test_clearance.py`:

```python
import numpy as np  # (already imported at top; harmless if deduped)

from reblock.contracts import Proposal
from reblock.methods.clearance import ClearanceReblocker, _greedy_reblock


def _column_block_with_buildings(h: int) -> Block:
    block = _column_block(h)
    pts = [g.representative_point() for g in block.parcels.geometry]
    block_bp = gpd.GeoDataFrame(geometry=pts, crs=UTM)
    return Block(block_id="colb", crs=UTM, boundary=block.boundary, parcels=block.parcels,
                 streets=block.streets, building_points=block_bp)


def test_greedy_reblock_achieves_depth_target() -> None:
    block = _column_block_with_buildings(8)  # depth 1..8
    roads, params = _greedy_reblock(block, t=0.5, res=0.5, depth_target=2, max_roads=400,
                                    radii=np.zeros(len(block.building_points)))
    assert len(roads) > 0
    after = parcel_access_layers(block, roads).to_numpy()
    assert int(after.max()) <= 2
    assert params["max_roads_hit"] is False


def test_greedy_reblock_returns_empty_when_already_shallow() -> None:
    block = _column_block_with_buildings(2)  # depth 1..2, target 2 -> nothing to do
    roads, params = _greedy_reblock(block, t=0.5, res=0.5, depth_target=2, max_roads=400,
                                    radii=np.zeros(len(block.building_points)))
    assert len(roads) == 0
    assert params["roads"] == 0


def test_propose_is_deterministic_and_leaves_rng_untouched() -> None:
    block = _column_block_with_buildings(8)
    np.random.seed(123)
    state = np.random.get_state()[1].tolist()
    p1 = ClearanceReblocker(depth_target=2, res=0.5).propose(block)
    p2 = ClearanceReblocker(depth_target=2, res=0.5).propose(block)
    assert np.random.get_state()[1].tolist() == state
    assert p1.roads is not None and p2.roads is not None and len(p1.roads) > 0
    assert [g.wkt for g in p1.roads.geometry] == [g.wkt for g in p2.roads.geometry]


def test_propose_metadata_and_identity() -> None:
    m = ClearanceReblocker(repulsion=2.0, depth_target=3, res=0.75, max_roads=50)
    assert m.identity == ("clearance", 2.0, 3, 0.75, 50)
    p = m.propose(_column_block_with_buildings(4))
    assert p.method == "clearance"
    assert p.proposal_id == "clearance:r2:d3:res0.75"
    assert p.block_identity == _column_block_with_buildings(4).identity
    assert p.params["repulsion"] == 2.0 and p.params["depth_target"] == 3


def test_distinct_repulsions_get_distinct_proposal_identity() -> None:
    # so access_after / geometric_after (keyed on the proposal) never collide across the knob
    block = _column_block_with_buildings(6)
    a = ClearanceReblocker(repulsion=-6.0).propose(block)
    b = ClearanceReblocker(repulsion=6.0).propose(block)
    assert a.proposal_id != b.proposal_id
    assert a.identity != b.identity


def test_propose_achieves_target_on_real_block() -> None:
    from tests.scoring_fixtures import _block_1808
    block = _block_1808()
    m = ClearanceReblocker(depth_target=2, res=0.75)
    roads = m.propose(block).roads
    assert roads is not None
    after = parcel_access_layers(block, roads).to_numpy()
    assert int(after.max()) <= 2  # invariant holds whether or not roads were needed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/methods/test_clearance.py -x -q -k "greedy_reblock or propose or repulsions_get"`
Expected: FAIL with `ImportError: cannot import name 'ClearanceReblocker'`.

- [ ] **Step 3: Add the greedy loop + Method**

Add these imports to the top of `src/reblock/methods/clearance.py`:

```python
from dataclasses import dataclass

import geopandas as gpd
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points, unary_union

from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
```

Add the loop and the class (at the end of the file):

```python
def _greedy_reblock(
    block: Block, *, t: float, res: float, depth_target: int, max_roads: int,
    radii: NDArray[np.float64],
) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    """Greedy least-cost-path reblock: repeatedly connect the deepest parcel to the growing
    road+street network by a Dijkstra path on the repulsion cost field, maintaining access
    depth incrementally, until every parcel is within `depth_target` (or `max_roads` is hit)."""
    parcels = block.parcels
    geoms = list(parcels.geometry)
    parcel_ids = np.asarray(parcels["parcel_id"])
    adj = parcel_adjacency(geoms, STREET_TOL)
    depth = parcel_access_layers(block, None, adj=adj).to_numpy().astype(np.float64)

    empty = gpd.GeoDataFrame(geometry=[], crs=block.crs)
    if depth.size == 0 or float(depth.max()) <= depth_target:
        return empty, {"roads": 0, "max_depth_after": int(depth.max()) if depth.size else 0,
                       "grid_unreachable": 0, "max_roads_hit": False}

    pts, rows, cols, edist = _build_grid(block.boundary, res)
    if len(pts) == 0:
        raise ValueError("Block.boundary yields no grid nodes at this resolution")
    building_pts = (
        np.array([[p.x, p.y] for p in block.building_points.geometry], dtype=np.float64)
        if not block.building_points.empty else np.empty((0, 2), dtype=np.float64)
    )
    clear = _node_clearance(pts, building_pts, radii)
    w = _edge_weights(clear, t, rows, cols, edist)
    csr = csr_matrix((w, (rows, cols)), shape=(len(pts), len(pts)))

    pt_tree = cKDTree(pts)
    reps = np.array([[g.representative_point().x, g.representative_point().y] for g in geoms])
    street = unary_union(list(block.streets.geometry))
    parcel_tree = STRtree(geoms)
    net = np.flatnonzero(
        shapely.dwithin(shapely.points(pts), street, res * _NET_TOL_FACTOR)).tolist()
    if not net:
        raise ValueError(
            "Block.streets yields no grid seed nodes: with no street frontage the least-cost "
            "forest has no root")

    roads: list[LineString] = []
    n_grid_unreachable = 0
    while len(roads) < max_roads:
        maxd = float(depth.max())
        if maxd <= depth_target:
            break
        cands = np.flatnonzero(depth == maxd)
        worst = int(cands[np.argmin(parcel_ids[cands])])          # deepest, ties by parcel_id
        start = int(pt_tree.query(reps[worst])[1])
        d, pred, _src = dijkstra(csr, indices=net, return_predecessors=True, min_only=True)
        if not np.isfinite(d[start]):
            depth[worst] = -np.inf                                # grid-unroutable: drop from selection
            n_grid_unreachable += 1
            continue
        pathn = [start]
        while pred[pathn[-1]] >= 0:
            pathn.append(int(pred[pathn[-1]]))
        coords: list[tuple[float, float]] = [(float(reps[worst][0]), float(reps[worst][1]))]
        coords += [(float(pts[k][0]), float(pts[k][1])) for k in pathn]
        term = Point(pts[pathn[-1]])
        if street.distance(term) <= res * _NET_TOL_FACTOR:        # bridge grid->street gap only
            sp = nearest_points(term, street)[1]                  # when we actually reached street
            coords.append((sp.x, sp.y))
        coords = [c for i, c in enumerate(coords) if i == 0 or c != coords[i - 1]]
        if len(coords) < 2:
            depth[worst] = -np.inf
            n_grid_unreachable += 1
            continue
        road = LineString(coords)
        roads.append(road)
        served = [int(p) for p in
                  parcel_tree.query(road, predicate="dwithin", distance=STREET_TOL)]
        _relax_depth(depth, adj, served)
        net.extend(pathn)

    gdf = gpd.GeoDataFrame(geometry=roads, crs=block.crs)
    final = parcel_access_layers(block, gdf, adj=adj)             # honest max over the ACTUAL network
    max_depth_after = int(final.max())                            # surfaces any grid-stranded parcel
    max_roads_hit = len(roads) >= max_roads and max_depth_after > depth_target
    params: dict[str, object] = {
        "roads": len(roads), "max_depth_after": max_depth_after,
        "grid_unreachable": n_grid_unreachable, "max_roads_hit": bool(max_roads_hit)}
    return gdf, params


@dataclass
class ClearanceReblocker:
    """Greedy least-cost-path reblocker. `repulsion` is the logit knob (s): s -> -inf straight
    (aspirational), 0 balanced, s -> +inf Voronoi-following (buildable). See module docstring."""

    repulsion: float = 0.0
    depth_target: int = 2
    res: float = 1.5
    max_roads: int = 400

    @property
    def identity(self) -> tuple[str, float, int, float, int]:
        return ("clearance", float(self.repulsion), int(self.depth_target),
                float(self.res), int(self.max_roads))

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior  # accepted for Method conformance; the routing is block-only
        t = _sigmoid(self.repulsion)
        n_b = 0 if block.building_points.empty else len(block.building_points)
        radii = np.zeros(n_b, dtype=np.float64)   # plain clearance; weighted footprints are future
        roads, params = _greedy_reblock(
            block, t=t, res=self.res, depth_target=self.depth_target,
            max_roads=self.max_roads, radii=radii)
        pid = f"clearance:r{self.repulsion:g}:d{self.depth_target}:res{self.res:g}"
        return Proposal(
            block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
            proposal_id=pid, method="clearance",
            params={**params, "repulsion": self.repulsion,
                    "depth_target": self.depth_target, "res": self.res},
            block_identity=block.identity)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/methods/test_clearance.py -x -q`
Expected: PASS (all tests, ~13).

- [ ] **Step 5: Lint + type-check**

Run: `pixi run check`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/methods/clearance.py tests/methods/test_clearance.py
git commit -m "feat: ClearanceReblocker greedy least-cost-path Method

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

### Task 4: Pipeline + cache wiring

**Files:**
- Modify: `src/reblock/derive_graph.py` (add to `_DERIVATION_MODULES`, line ~44)
- Create: `conf/method/clearance.yaml`
- Modify: `conf/compare_config.yaml` (`all_methods`, line ~15)
- Test: `tests/methods/test_clearance.py`

**Interfaces:**
- Consumes: `hydra.utils.instantiate`, `reblock.derive_graph._DERIVATION_MODULES`, `reblock.derivations.propose`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/methods/test_clearance.py`:

```python
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def test_clearance_module_is_in_derivation_modules() -> None:
    # so a change to the algorithm busts the memoized propose() cache (like dijkstra/mesh/arterial)
    from reblock.derive_graph import _DERIVATION_MODULES
    assert any(p.name == "clearance.py" and p.parent.name == "methods"
               for p in _DERIVATION_MODULES)


def test_clearance_method_yaml_instantiates_with_defaults() -> None:
    conf_dir = str(Path(__file__).resolve().parents[2] / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config", overrides=["method=clearance"])
    method = instantiate(cfg.method)
    assert isinstance(method, ClearanceReblocker)
    assert method.identity == ("clearance", 0.0, 2, 1.5, 400)


def test_clearance_registered_in_compare_all_methods() -> None:
    conf_dir = str(Path(__file__).resolve().parents[2] / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="compare_config")
    method = instantiate(cfg.all_methods["clearance"])
    assert isinstance(method, ClearanceReblocker)


def test_propose_routes_through_memoized_derivation() -> None:
    # region_reblock / compare call derivations.propose; a cacheable block returns identical roads
    from reblock.derivations import propose
    from tests.scoring_fixtures import _block_1808
    block = _block_1808()
    m = ClearanceReblocker(depth_target=2, res=0.75)
    r1 = propose(m, block).roads
    r2 = propose(m, block).roads
    assert r1 is not None and r2 is not None
    assert [g.wkt for g in r1.geometry] == [g.wkt for g in r2.geometry]
```

> Implementer note: confirm the exact override key names by checking an existing method config test if `method=clearance` compose fails (the repo's default `config.yaml` group is `method`; `compare_config.yaml` uses inline `all_methods`). Do not change the config schema — only add the clearance entries.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/methods/test_clearance.py -x -q -k "derivation_modules or yaml_instantiates or all_methods or memoized"`
Expected: FAIL — `clearance.py` not in `_DERIVATION_MODULES`; `conf/method/clearance.yaml` missing; `all_methods["clearance"]` KeyError.

- [ ] **Step 3: Add clearance to `_DERIVATION_MODULES`**

In `src/reblock/derive_graph.py`, add one line inside the `_DERIVATION_MODULES` tuple, after the `arterial.py` entry:

```python
    Path(__file__).parent / "methods" / "arterial.py",
    Path(__file__).parent / "methods" / "clearance.py",
```

- [ ] **Step 4: Create `conf/method/clearance.yaml`**

```yaml
# Greedy least-cost-path reblocker with the logit repulsion knob -- aspirational straight roads
# (repulsion -> -inf) .. buildable Voronoi-following roads (repulsion -> +inf). See
# reblock.methods.clearance.ClearanceReblocker.
_target_: reblock.methods.clearance.ClearanceReblocker
repulsion: 0.0
depth_target: 2
res: 1.5
max_roads: 400
```

- [ ] **Step 5: Register in `conf/compare_config.yaml`**

In `all_methods`, add one line (after the `greedy_arterial_displacement` entry):

```yaml
  clearance: {_target_: reblock.methods.clearance.ClearanceReblocker, repulsion: 0.0, depth_target: 2, res: 1.5, max_roads: 400}
```

(Leave the `methods:` run-list unchanged — clearance is opt-in via `methods=[clearance]`.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `pixi run pytest tests/methods/test_clearance.py -x -q`
Expected: PASS (all).

- [ ] **Step 7: Lint + type-check + full suite (cache-hash change touches memoization)**

Run: `pixi run check`
Expected: green. (Adding to `_DERIVATION_MODULES` changes `_CODE_HASH`; confirm no cache-pinned test asserts a stale hash.)

- [ ] **Step 8: Commit**

```bash
git add src/reblock/derive_graph.py conf/method/clearance.yaml conf/compare_config.yaml tests/methods/test_clearance.py
git commit -m "feat: wire clearance into compare/run + derivation cache

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

### Task 5: Repulsion-sweep example gallery (true end-to-end)

**Files:**
- Create: `examples/clearance-repulsion/generate.py`
- Create: `examples/clearance-repulsion/README.md`
- Create (committed outputs): `examples/clearance-repulsion/*.png`

**Interfaces:**
- Consumes: `reblock.data.provision.cached_kblock_source`, `reblock.data.kblock.KblockSource`, `reblock.screen.dense_compact.DenseCompactScreen`, `reblock.region.{DenseClusterRegionBuilder, region_block}`, `reblock.methods.clearance.ClearanceReblocker`, `reblock.derivations.access_before`, `reblock.derive.access.parcel_access_layers`, `reblock.render.{render_before, render_after, save_render, frame_bbox}`, `reblock.budget.displacement_count`.

This task produces artifacts from `capetown_full` (auto-downloaded to `~/.cache/reblock`, never committed) — mirroring `examples/capetown-flagship/`. It is not gated by `pixi run check` (it needs the full metro, not the test sample); validate it by running the generator and eyeballing the PNGs + printed table.

- [ ] **Step 1: Write the generator**

Create `examples/clearance-repulsion/generate.py`:

```python
"""Clearance reblocker — the repulsion knob on ONE auto-detected deep Cape Town region.

True end-to-end: screen the full metro (DenseCompactScreen, memoized), auto-pick the deepest
seed whose own building_count is in a tractable window, grow it into a small multi-block
neighborhood (DenseClusterRegionBuilder), build the region, then reblock at five repulsions
(the two extremes, two moderates, and the balanced default) and render each. Reproduces the
committed PNGs. Run: pixi run python examples/clearance-repulsion/generate.py
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np

from reblock.budget import displacement_count
from reblock.data.kblock import KblockSource
from reblock.data.provision import cached_kblock_source
from reblock.derivations import access_before
from reblock.derive.access import parcel_access_layers
from reblock.methods.clearance import ClearanceReblocker
from reblock.region import DenseClusterRegionBuilder, region_block
from reblock.render import frame_bbox, render_after, render_before, save_render
from reblock.screen.dense_compact import DenseCompactScreen

OUT = Path(__file__).parent
REPULSIONS = [-6.0, -2.0, 0.0, 2.0, 6.0]     # extremes, moderates, balanced default
SEED_MIN, SEED_MAX = 120, 350                # tractable deep seed (giants alone are 1000-3000)
REGION_MAX = 300                             # region buildings budget -> ~a dozen roads/panel


def main() -> None:
    source = cached_kblock_source(city="capetown")
    screen = DenseCompactScreen(max_depth_min=6.0)         # deep informal fabric, deepest-first
    ranked = screen.select(source)                          # memoized -> instant on rerun
    geoms = source.block_geometries()                      # block_id + geometry + building_count
    count = dict(zip(geoms["block_id"].astype(str), geoms["building_count"]))

    seed = next(b for b in ranked if SEED_MIN <= count.get(b, 0) <= SEED_MAX)
    members = DenseClusterRegionBuilder(max_buildings=REGION_MAX).build(geoms, [[seed]])[0]
    print(f"auto-detected region: seed={seed}  members={members}")

    blocks = list(KblockSource(source.blocks_path, source.buildings_path, region_id="clearance",
                               block_ids=members).region().blocks)
    region = region_block(blocks)
    print(f"region: {len(region.parcels)} parcels, {len(region.building_points)} buildings")

    before = access_before(region)
    vmax = int(before.max())
    frame = frame_bbox(region.parcels)
    own_pts = region.building_points
    save_render(render_before(region, before, vmax=vmax, own_points=own_pts, frame=frame),
                OUT / "before.png")

    print(f"\n{'repulsion':>9} {'roads':>5} {'length_m':>9} {'displaced':>9} {'max_depth':>9}")
    for s in REPULSIONS:
        proposal = ClearanceReblocker(repulsion=s).propose(region)
        roads = proposal.roads
        after = parcel_access_layers(region, roads)
        length = float(sum(g.length for g in roads.geometry)) if len(roads) else 0.0
        displaced = displacement_count(region.building_points, roads, 3.0)
        fig = render_after(region, proposal, after, vmax=vmax, own_points=own_pts, frame=frame)
        tag = f"{s:+.0f}".replace("+0", "0")
        save_render(fig, OUT / f"after_s{tag}.png")
        print(f"{s:>9.0f} {len(roads):>5d} {length:>9.0f} {displaced:>9d} {int(after.max()):>9d}")


if __name__ == "__main__":
    main()
```

> Implementer notes:
> - Verify `cached_kblock_source` and `KblockSource` expose `.blocks_path` / `.buildings_path`; `scoring_fixtures.py` constructs `KblockSource(blocks_path, buildings_path, region_id, block_ids=...)` and `dense_compact.py` reads `source.blocks_path`/`source.buildings_path`, so the attributes exist. If `cached_kblock_source`'s signature differs, match `conf/data/capetown_full.yaml` (`city: capetown`).
> - If `next(...)` finds no seed in `[SEED_MIN, SEED_MAX]`, widen the window and re-run; **do not** hand-pick a block_id — the point is auto-detection. Print the chosen seed + members so the README can name them.
> - If a panel's road count is large (region deeper/denser than expected), lower `REGION_MAX` until each panel is legible (~a dozen roads); record the final value.

- [ ] **Step 2: Run the generator**

Run: `pixi run python examples/clearance-repulsion/generate.py`
Expected: prints the auto-detected seed + members, the region size, and a 5-row table (repulsion → roads, length, displaced, max_depth); writes `before.png` + five `after_s*.png`. Sanity: `max_depth ≤ 2` for every row; `displaced` trends **down** and `length` trends **up** as repulsion goes `−6 → +6` (straight crosses more buildings; Voronoi-following weaves the gaps).

- [ ] **Step 3: Optional region-context map**

If a region-context map improves the README (as in flagship's `region_map.png`), add a small block to `generate.py` that saves `render_before` with the neighboring-block context outlines (`source.block_geometries` windowed to `frame`, passed as `context_outlines`) to `region_map.png`. Skip if `before.png` already reads clearly.

- [ ] **Step 4: Write the README**

Create `examples/clearance-repulsion/README.md` documenting: (1) what the knob is (logit `repulsion`, `t = sigmoid(s)`, straight ↔ Voronoi); (2) the auto-detection (screen → tractable deep seed → dense_cluster grow → `region_block`), naming the actual seed + members the generator printed; (3) the reproduce command (`pixi run python examples/clearance-repulsion/generate.py`); (4) the five panels side-by-side in a markdown table (`before` + `after_s-6 … after_s+6`); (5) the metrics table with the real printed numbers; (6) the reading — coverage (max depth) held constant across panels while the knob trades displacement for directness/straightness. Follow `examples/capetown-flagship/README.md`'s tone and structure. Use only numbers the generator actually printed — no placeholders.

- [ ] **Step 5: Commit**

```bash
git add examples/clearance-repulsion/
git commit -m "docs: clearance repulsion-sweep gallery (auto-detected deep CT region, 5 knobs)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

## Self-Review

**Spec coverage:**
- Least-cost path + logit knob (`edge_weight`, `t=sigmoid(s)`) → Task 1 primitives + Task 3 loop. ✓
- Greedy + incremental depth (deepest parcel, relax, network growth) → Tasks 2–3. ✓
- Weighted-points-ready (`clearance = dist − radius`, `r_i=0` default) → Task 1 `_node_clearance` + weighted unit test; not wired into `propose` (spec defers footprints). ✓
- Interface (`repulsion/depth_target/res/max_roads`, `identity`, `propose`) → Task 3. ✓
- Config + `compare_config` `all_methods` + `_DERIVATION_MODULES` → Task 4. ✓
- Correctness strategy: deterministic (WKT-identical, parcel_id tie-break) ✓; terminates + achieves target ✓; incremental == recompute ✓; knob bends path ✓; weighted-ready ✓; `pixi run check` green ✓.
- Examples gallery (5 repulsions, auto-detected deep region, true e2e, not too many roads) → Task 5. ✓
- Out-of-scope (sequencer, sparsified global, weighted footprints e2e) → correctly omitted.

**Placeholder scan:** No TBD/TODO. Every code step shows complete code. Task 5's README uses generator-printed numbers (the one runtime-dependent artifact) — flagged explicitly as "no placeholders," with a fallback (widen window / lower `REGION_MAX`) rather than a guess.

**Type consistency:** `_greedy_reblock` / `_relax_depth` / `_node_clearance` / `_edge_weights` / `_build_grid` / `_sigmoid` signatures are identical between their Interfaces blocks, their definitions, and their call sites. `ClearanceReblocker.identity` shape `("clearance", float, int, float, int)` matches the `test_propose_metadata_and_identity` assertion. `proposal_id` format `clearance:r{g}:d{}:res{g}` matches the test's `"clearance:r2:d3:res0.75"`.
