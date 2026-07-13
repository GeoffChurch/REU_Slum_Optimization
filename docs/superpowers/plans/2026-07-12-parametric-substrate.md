# Parametric Routing Substrate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the clearance reblocker's routing substrate a pluggable Strategy (grid | chord_diag default | theta_spanner | cdt_gap | prebuilt), reusing the existing cost field + greedy, with 3-point edge sampling everywhere.

**Architecture:** A new `reblock/methods/substrates.py` holds a `Substrate` protocol returning a `RoutingGraph` (node coords + symmetric COO edges + a network-seed tolerance). `reblock/methods/clearance.py` keeps the cost field, incremental relax, and greedy loop — refactored to take a `RoutingGraph` instead of building a grid — and `ClearanceReblocker` gains a `substrate` field defaulting to `ChordSubstrate`. The three tessellation substrates (chord_diag, theta_spanner, cdt_gap) share one boundary-vertex node set and differ only in edge selection.

**Tech Stack:** Python, numpy, scipy (`sparse.csr_matrix`, `csgraph.dijkstra`, `spatial.{cKDTree, Delaunay}`), shapely 2.x, geopandas, networkx (via `dijkstra._boundary_graph`), Hydra, pytest, pixi (`pixi run check` = ruff + mypy --strict + pytest).

## Global Constraints

- **`chord_diag` is the default substrate.** `ClearanceReblocker.substrate` defaults to `ChordSubstrate()`; grid is opt-in.
- **3-point edge sampling everywhere** (endpoints + midpoint), one rule. The old endpoint-only `_edge_weights` is deleted, not kept. This changes the grid at high repulsion (~5.4% at `s=+6`); the new output is the golden reference.
- **No top-level `res`** on `ClearanceReblocker` — it lives on `GridSubstrate(res=1.5)`.
- **`proposal_id = f"clearance:{tag}:r{repulsion:g}:d{depth_target}:mr{max_roads}"`** where `tag` is the substrate short name; **`identity = ("clearance", substrate.identity, repulsion, depth_target, max_roads)`**. So distinct substrates/configs never collide in the `access_after`/`geometric_after` caches.
- **Substrate short names / identities:** `GridSubstrate.identity = ("grid", res)` tag `grid`; `ChordSubstrate.identity = ("chord_diag",)` tag `chord_diag`; `SpannerSubstrate.identity = ("theta_spanner", cones)` tag `theta_spanner`; `CdtSubstrate.identity = ("cdt_gap",)` tag `cdt_gap`.
- **Determinism:** two `propose` calls → WKT-identical roads; no global RNG. Substrate builders sort nodes/edges for determinism.
- **Depth seeding precondition preserved:** the greedy seeds `parcel_access_layers(block, None, adj=adj, unreached_depth=len(geoms)+1)` (the `_relax_depth` precondition).
- **No legacy / dual-path code** (owner directive): migrate, delete the old path.
- `reblock/methods/substrates.py` must be added to `reblock.derive_graph._DERIVATION_MODULES`.
- `pixi run check` green before every commit.
- **Commit trailers (verbatim, every commit):**
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
  ```

## File Structure

- **Create** `src/reblock/methods/substrates.py` — `RoutingGraph`, `Substrate` protocol, `_pack_edges`, `_boundary_vertices`, `_build_grid` (moved from clearance.py), `GridSubstrate`, `ChordSubstrate`, `SpannerSubstrate`, `CdtSubstrate`, `PrebuiltSubstrate`.
- **Modify** `src/reblock/methods/clearance.py` — 3-point `_edge_weights`; `_greedy_reblock` takes a `RoutingGraph`; `ClearanceReblocker` takes `substrate`. Remove `_build_grid` (moved), `res`, `_NET_TOL_FACTOR` usage.
- **Modify** `src/reblock/derive_graph.py` — register `methods/substrates.py`.
- **Create** `conf/substrate/{grid,chord_diag,theta_spanner,cdt_gap}.yaml`; **Modify** `conf/config.yaml`, `conf/compare_config.yaml`, `conf/method/clearance.yaml`.
- **Create** `tests/methods/test_substrates.py`; **Modify** `tests/methods/test_clearance.py`.
- **Modify** `examples/clearance-repulsion/` (regenerated outputs + README).

---

### Task 1: `substrates.py` foundation — RoutingGraph, protocol, GridSubstrate, PrebuiltSubstrate

**Files:**
- Create: `src/reblock/methods/substrates.py`
- Modify: `src/reblock/methods/clearance.py` (remove `_build_grid`; import it from substrates)
- Modify: `src/reblock/derive_graph.py` (register substrates.py)
- Test: `tests/methods/test_substrates.py`

**Interfaces:**
- Produces (used by later tasks):
  - `@dataclass(frozen=True) RoutingGraph` with `pts: NDArray[float64]`, `rows: NDArray[int64]`, `cols: NDArray[int64]`, `edist: NDArray[float64]`, `net_tol: float`.
  - `class Substrate(Protocol)`: `build(self, block: Block) -> RoutingGraph`; `identity` property (Hashable); `tag` property (str).
  - `_pack_edges(pts, edges: set[frozenset[int]]) -> tuple[pts, rows, cols, edist]`
  - `_build_grid(boundary: Polygon, res: float) -> tuple[pts, rows, cols, edist]` (moved verbatim from clearance.py)
  - `GridSubstrate(res: float = 1.5)`, `PrebuiltSubstrate(graph: RoutingGraph)`

- [ ] **Step 1: Write the failing tests**

Create `tests/methods/test_substrates.py`:

```python
from typing import cast

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import Polygon
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.methods.substrates import (
    GridSubstrate,
    PrebuiltSubstrate,
    RoutingGraph,
    Substrate,
    _pack_edges,
)

UTM = CRS.from_epsg(32643)


def _grid_block(n: int) -> Block:
    polys, ids = [], []
    for i in range(n):
        for j in range(n):
            polys.append(Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]))
            ids.append(i * n + j)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="grid", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_pack_edges_is_symmetric_and_sorted() -> None:
    pts = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
    pts2, rows, cols, edist = _pack_edges(pts, {frozenset((0, 1)), frozenset((0, 2))})
    assert len(rows) == len(cols) == len(edist) == 4          # 2 undirected -> 4 directed
    undirected = {frozenset((int(a), int(b))) for a, b in zip(rows, cols)}
    assert undirected == {frozenset((0, 1)), frozenset((0, 2))}
    # lengths correct (3-4-5 triangle and the unit edge)
    assert set(np.round(edist, 6)) == {5.0, 1.0}


def test_grid_substrate_builds_valid_routing_graph() -> None:
    graph = GridSubstrate(res=1.0).build(_grid_block(4))
    assert isinstance(graph, RoutingGraph)
    assert graph.pts.shape[1] == 2 and len(graph.pts) > 0
    assert len(graph.rows) == len(graph.cols) == len(graph.edist)
    assert graph.net_tol == pytest.approx(1.5)                 # res * 1.5
    assert GridSubstrate(res=1.0).identity == ("grid", 1.0)
    assert GridSubstrate(res=1.0).tag == "grid"


def test_prebuilt_substrate_round_trips() -> None:
    g = RoutingGraph(pts=np.array([[0.0, 0.0], [1.0, 0.0]]),
                     rows=np.array([0, 1]), cols=np.array([1, 0]),
                     edist=np.array([1.0, 1.0]), net_tol=0.5)
    sub: Substrate = PrebuiltSubstrate(g)
    assert sub.build(_grid_block(2)) is g
    assert sub.tag == "prebuilt"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/methods/test_substrates.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'reblock.methods.substrates'`.

- [ ] **Step 3: Create `substrates.py`**

Create `src/reblock/methods/substrates.py`:

```python
"""Routing substrates for the clearance reblocker: a pluggable graph the least-cost-path greedy
routes on. Each Substrate.build(block) returns a RoutingGraph (node coords + symmetric COO edges
+ a network-seed tolerance). The cost field + greedy (reblock.methods.clearance) are substrate-
agnostic; substrates differ only in node set + edge selection. See
docs/superpowers/specs/2026-07-12-parametric-substrate-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Protocol

import numpy as np
from numpy.typing import NDArray
from shapely import contains_xy
from shapely.geometry import Polygon

from reblock.contracts import Block


@dataclass(frozen=True)
class RoutingGraph:
    """A built substrate: node coords `pts` (M,2), symmetric COO edges `rows`/`cols` (each
    undirected edge stored both ways) with lengths `edist`, and `net_tol` (a node within this of
    a street both seeds the network and gates the final street-snap)."""

    pts: NDArray[np.float64]
    rows: NDArray[np.int64]
    cols: NDArray[np.int64]
    edist: NDArray[np.float64]
    net_tol: float


class Substrate(Protocol):
    def build(self, block: Block) -> RoutingGraph: ...
    @property
    def identity(self) -> Hashable: ...
    @property
    def tag(self) -> str: ...


def _pack_edges(
    pts: NDArray[np.float64], edges: set[frozenset[int]]
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Symmetric COO from a set of undirected {i, j} edges, sorted for determinism."""
    ordered = sorted(tuple(sorted(e)) for e in edges)
    rows: list[int] = []
    cols: list[int] = []
    dist: list[float] = []
    for i, j in ordered:
        d = float(np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1]))
        rows += [i, j]
        cols += [j, i]
        dist += [d, d]
    return (pts.astype(np.float64), np.asarray(rows, dtype=np.int64),
            np.asarray(cols, dtype=np.int64), np.asarray(dist, dtype=np.float64))


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


_GRID_NET_TOL_FACTOR = 1.5   # a grid node within res * this of the street seeds the network


@dataclass(frozen=True)
class GridSubstrate:
    """8-connected regular grid at resolution `res` (m). Faithful cost-field sampler, but node
    count scales with block AREA (∝ area/res²) and paths staircase at 45°."""

    res: float = 1.5

    @property
    def identity(self) -> Hashable:
        return ("grid", float(self.res))

    @property
    def tag(self) -> str:
        return "grid"

    def build(self, block: Block) -> RoutingGraph:
        pts, rows, cols, edist = _build_grid(block.boundary, self.res)
        return RoutingGraph(pts, rows, cols, edist, net_tol=self.res * _GRID_NET_TOL_FACTOR)


@dataclass(frozen=True)
class PrebuiltSubstrate:
    """The 'provided graph' escape hatch: `build` returns the given RoutingGraph verbatim.
    identity is None (uncacheable) — an ad-hoc graph must not key-collide with a named substrate."""

    graph: RoutingGraph

    @property
    def identity(self) -> Hashable:
        return None

    @property
    def tag(self) -> str:
        return "prebuilt"

    def build(self, block: Block) -> RoutingGraph:
        del block
        return self.graph
```

- [ ] **Step 4: Move `_build_grid` out of clearance.py**

In `src/reblock/methods/clearance.py`: delete the `_build_grid` function (lines ~53–82) and the `_NET_TOL_FACTOR` constant's grid-only use stays for now (the greedy still references it until Task 3). Add an import so the greedy keeps working:

```python
from reblock.methods.substrates import _build_grid
```

Remove the now-unused imports if any (`contains_xy` moves to substrates; if clearance.py no longer uses it, drop it — ruff will flag). Run `pixi run pytest tests/methods/test_clearance.py -q` to confirm clearance still works with the imported `_build_grid`.

- [ ] **Step 5: Register substrates.py in `_DERIVATION_MODULES`**

In `src/reblock/derive_graph.py`, add after the `clearance.py` line:

```python
    Path(__file__).parent / "methods" / "clearance.py",
    Path(__file__).parent / "methods" / "substrates.py",
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pixi run pytest tests/methods/test_substrates.py tests/methods/test_clearance.py -q`
Expected: PASS (new substrate tests + unchanged clearance tests).

- [ ] **Step 7: Full gate + commit**

Run: `pixi run check` → green.
```bash
git add src/reblock/methods/substrates.py src/reblock/methods/clearance.py src/reblock/derive_graph.py tests/methods/test_substrates.py
git commit -m "feat: substrates.py — RoutingGraph, Substrate protocol, GridSubstrate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

### Task 2: 3-point edge sampling

**Files:**
- Modify: `src/reblock/methods/clearance.py` (`_edge_weights` + its greedy call site)
- Test: `tests/methods/test_clearance.py`

**Interfaces:**
- Produces: `_edge_weights(pts, rows, cols, edist, building_pts, radii, t) -> NDArray[float64]` — 3-point (endpoints + midpoint) sampled edge weights. Signature CHANGES from the old `(clear, t, rows, cols, edist)`.
- Consumes: `_node_clearance` (unchanged).

- [ ] **Step 1: Write the failing test**

Append to `tests/methods/test_clearance.py`:

```python
from reblock.methods.clearance import _edge_weights, _node_clearance, _sigmoid


def test_edge_weights_3point_sees_a_midspan_building() -> None:
    # A long edge whose two endpoints sit in the clear but whose MIDPOINT skims a building must
    # read as more expensive than a plain endpoint-only average would make it. Endpoints at
    # (0,0) and (10,0) are far from the single building at (5,3); the midpoint (5,0) is close.
    pts = np.array([[0.0, 0.0], [10.0, 0.0]])
    rows = np.array([0, 1])
    cols = np.array([1, 0])
    edist = np.array([10.0, 10.0])
    buildings = np.array([[5.0, 3.0]])          # nearest to the midpoint, far from endpoints
    radii = np.zeros(1)
    t = _sigmoid(6.0)                            # high repulsion -> clearance dominates cost
    w = _edge_weights(pts, rows, cols, edist, buildings, radii, t)
    # endpoint-only weight for comparison: mean of the two endpoint node costs * length
    clear_ends = _node_clearance(pts, buildings, radii)
    node_cost_ends = (1.0 - t) + t / clear_ends
    endpoint_only = 10.0 * 0.5 * (node_cost_ends[0] + node_cost_ends[1])
    assert w[0] == pytest.approx(w[1])          # symmetric COO
    assert w[0] > endpoint_only                 # the midpoint building raised the cost
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/methods/test_clearance.py::test_edge_weights_3point_sees_a_midspan_building -x -q`
Expected: FAIL — the old `_edge_weights(clear, t, rows, cols, edist)` signature rejects these args (TypeError).

- [ ] **Step 3: Replace `_edge_weights` with the 3-point version**

In `src/reblock/methods/clearance.py`, replace the whole `_edge_weights` function with:

```python
def _edge_weights(
    pts: NDArray[np.float64], rows: NDArray[np.int64], cols: NDArray[np.int64],
    edist: NDArray[np.float64], building_pts: NDArray[np.float64],
    radii: NDArray[np.float64], t: float,
) -> NDArray[np.float64]:
    """Edge weight = length * mean(node cost) sampled at BOTH endpoints AND the midpoint, node
    cost = (1 - t) + t / clearance. 3-point (not endpoint-only) so a long edge whose midpoint
    skims a building — but whose endpoints sit in the open — still reads as expensive. Returns
    weights aligned to the symmetric COO `rows`/`cols` order."""
    n = len(pts)
    mask = rows < cols                                   # one direction per undirected edge
    ui, uj, ulen = rows[mask], cols[mask], edist[mask]
    e = len(ui)
    if e == 0:
        return np.zeros(0, dtype=np.float64)
    mid = (pts[ui] + pts[uj]) / 2.0
    sample_pts = np.vstack([pts[ui], pts[uj], mid])
    clear = _node_clearance(sample_pts, building_pts, radii)
    ci, cj, cm = clear[:e], clear[e:2 * e], clear[2 * e:]
    mean_cost = ((1.0 - t) + t / ci) + ((1.0 - t) + t / cj) + ((1.0 - t) + t / cm)
    uw = ulen * (mean_cost / 3.0)
    # scatter the per-undirected-edge weight back onto BOTH directed COO entries
    key = np.minimum(rows, cols).astype(np.int64) * n + np.maximum(rows, cols).astype(np.int64)
    ukey = ui.astype(np.int64) * n + uj.astype(np.int64)
    order = np.argsort(ukey)
    pos = np.searchsorted(ukey[order], key)
    return uw[order][pos]
```

- [ ] **Step 4: Update the greedy call site**

In `_greedy_reblock`, replace the two lines that compute `clear` + `w`:

```python
    clear = _node_clearance(pts, building_pts, radii)
    w = _edge_weights(clear, t, rows, cols, edist)
```

with (drop the standalone `clear`; 3-point samples internally):

```python
    w = _edge_weights(pts, rows, cols, edist, building_pts, radii, t)
```

- [ ] **Step 5: Run tests**

Run: `pixi run pytest tests/methods/test_clearance.py -q`
Expected: PASS. The property tests (determinism, achieves-target) still hold — they assert behavior, not specific road WKT. (If any test pins a specific road geometry at high repulsion, re-pin it to the new 3-point output; none is expected.)

- [ ] **Step 6: Full gate + commit**

Run: `pixi run check` → green.
```bash
git add src/reblock/methods/clearance.py tests/methods/test_clearance.py
git commit -m "feat: 3-point edge-cost sampling (endpoints + midpoint) for clearance

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

### Task 3: Substrate-agnostic greedy + `ClearanceReblocker(substrate=…)` (grid default)

**Files:**
- Modify: `src/reblock/methods/clearance.py` (`_greedy_reblock`, `ClearanceReblocker`)
- Modify: `conf/method/clearance.yaml`
- Test: `tests/methods/test_clearance.py`

**Interfaces:**
- Produces: `_greedy_reblock(block, graph: RoutingGraph, *, t, depth_target, max_roads, radii) -> (GeoDataFrame, dict)`; `ClearanceReblocker(substrate: Substrate = GridSubstrate(), repulsion=0.0, depth_target=2, max_roads=400)` — **no `res`**.
- Consumes: `RoutingGraph`, `GridSubstrate` (Task 1); `_edge_weights` (Task 2).

- [ ] **Step 1: Update the tests to the new interface**

In `tests/methods/test_clearance.py`, the metadata test currently constructs `ClearanceReblocker(..., res=0.75, ...)` and pins the old `proposal_id`/`identity`. Replace `test_propose_metadata_and_identity` with:

```python
from reblock.methods.substrates import GridSubstrate


def test_propose_metadata_and_identity() -> None:
    m = ClearanceReblocker(substrate=GridSubstrate(res=0.75), repulsion=2.0,
                           depth_target=3, max_roads=50)
    assert m.identity == ("clearance", ("grid", 0.75), 2.0, 3, 50)
    p = m.propose(_column_block_with_buildings(4))
    assert p.method == "clearance"
    assert p.proposal_id == "clearance:grid:r2:d3:mr50"
    assert p.block_identity == _column_block_with_buildings(4).identity
    assert p.params["repulsion"] == 2.0 and p.params["depth_target"] == 3
```

Any other test that passes `res=` to `ClearanceReblocker` or `_greedy_reblock` must switch to a substrate. Update the greedy-level tests (`test_greedy_reblock_achieves_depth_target`, `test_greedy_reblock_returns_empty_when_already_shallow`) to build a graph and pass it:

```python
from reblock.methods.substrates import GridSubstrate

def test_greedy_reblock_achieves_depth_target() -> None:
    block = _column_block_with_buildings(8)
    graph = GridSubstrate(res=0.5).build(block)
    roads, params = _greedy_reblock(block, graph, t=0.5, depth_target=2, max_roads=400,
                                    radii=np.zeros(len(block.building_points)))
    assert len(roads) > 0
    after = parcel_access_layers(block, roads).to_numpy()
    assert int(after.max()) <= 2
    assert params["max_roads_hit"] is False


def test_greedy_reblock_returns_empty_when_already_shallow() -> None:
    block = _column_block_with_buildings(2)
    graph = GridSubstrate(res=0.5).build(block)
    roads, params = _greedy_reblock(block, graph, t=0.5, depth_target=2, max_roads=400,
                                    radii=np.zeros(len(block.building_points)))
    assert len(roads) == 0
    assert params["roads"] == 0
```

Update the propose-based tests that used `res=0.5`/`res=0.75` to pass `substrate=GridSubstrate(res=…)` instead (e.g. `test_propose_is_deterministic_and_leaves_rng_untouched`, `test_propose_achieves_target_on_real_block`, `test_distinct_repulsions_get_distinct_proposal_identity`). Keep every other assertion the same.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/methods/test_clearance.py -x -q`
Expected: FAIL — `ClearanceReblocker` still has `res` / `_greedy_reblock` still builds a grid (TypeError on the new `graph` positional / missing `substrate`).

- [ ] **Step 3: Refactor `_greedy_reblock` to take a RoutingGraph**

In `src/reblock/methods/clearance.py`, change the signature and body. Replace the grid-build + net lines. New signature and the changed region:

```python
def _greedy_reblock(
    block: Block, graph: RoutingGraph, *, t: float, depth_target: int, max_roads: int,
    radii: NDArray[np.float64],
) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    """Greedy least-cost-path reblock on a routing substrate `graph`: repeatedly connect the
    deepest parcel to the growing road+street network by a Dijkstra path on the repulsion cost
    field, maintaining access depth incrementally, until every parcel is within `depth_target`
    (or `max_roads` is hit)."""
    parcels = block.parcels
    geoms = list(parcels.geometry)
    parcel_ids = np.asarray(parcels["parcel_id"])
    adj = parcel_adjacency(geoms, STREET_TOL)
    depth = parcel_access_layers(
        block, None, adj=adj, unreached_depth=len(geoms) + 1).to_numpy().astype(np.float64)

    empty = gpd.GeoDataFrame(geometry=[], crs=block.crs)
    if depth.size == 0 or float(depth.max()) <= depth_target:
        return empty, {"roads": 0, "max_depth_after": int(depth.max()) if depth.size else 0,
                       "grid_unreachable": 0, "max_roads_hit": False}

    pts, rows, cols, edist, net_tol = (
        graph.pts, graph.rows, graph.cols, graph.edist, graph.net_tol)
    if len(pts) == 0:
        raise ValueError("substrate yields no nodes for this block")
    building_pts = (
        shapely.get_coordinates(block.building_points.geometry.to_numpy())
        if not block.building_points.empty else np.empty((0, 2), dtype=np.float64)
    )
    w = _edge_weights(pts, rows, cols, edist, building_pts, radii, t)
    csr = csr_matrix((w, (rows, cols)), shape=(len(pts), len(pts)))

    pt_tree = cKDTree(pts)
    reps = np.array([[g.representative_point().x, g.representative_point().y] for g in geoms])
    street = unary_union(list(block.streets.geometry))
    parcel_tree = STRtree(geoms)
    net = np.flatnonzero(shapely.dwithin(shapely.points(pts), street, net_tol)).tolist()
    if not net:
        raise ValueError(
            "substrate net seed empty: no node within net_tol of the street -- with no street "
            "frontage the least-cost forest has no root")

    roads: list[LineString] = []
    n_grid_unreachable = 0
    while len(roads) < max_roads:
        maxd = float(depth.max())
        if maxd <= depth_target:
            break
        cands = np.flatnonzero(depth == maxd)
        worst = int(cands[np.argmin(parcel_ids[cands])])
        start = int(pt_tree.query(reps[worst])[1])
        d, pred, _src = dijkstra(csr, indices=net, return_predecessors=True, min_only=True)
        if not np.isfinite(d[start]):
            depth[worst] = -np.inf
            n_grid_unreachable += 1
            continue
        pathn = [start]
        while pred[pathn[-1]] >= 0:
            pathn.append(int(pred[pathn[-1]]))
        coords: list[tuple[float, float]] = [(float(reps[worst][0]), float(reps[worst][1]))]
        coords += [(float(pts[k][0]), float(pts[k][1])) for k in pathn]
        term = Point(pts[pathn[-1]])
        if street.distance(term) <= net_tol:
            sp = nearest_points(term, street)[1]
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
    final = parcel_access_layers(block, gdf, adj=adj)
    max_depth_after = int(final.max())
    max_roads_hit = len(roads) >= max_roads and max_depth_after > depth_target
    params: dict[str, object] = {
        "roads": len(roads), "max_depth_after": max_depth_after,
        "grid_unreachable": n_grid_unreachable, "max_roads_hit": bool(max_roads_hit)}
    return gdf, params
```

Then remove the now-dead `from reblock.methods.substrates import _build_grid` import and the `_NET_TOL_FACTOR` constant (no longer used in clearance.py — `net_tol` comes from the graph). Add the needed imports at the top of clearance.py:

```python
from reblock.methods.substrates import GridSubstrate, RoutingGraph, Substrate
```

- [ ] **Step 4: Refactor `ClearanceReblocker`**

Replace the `ClearanceReblocker` dataclass with:

```python
@dataclass
class ClearanceReblocker:
    """Greedy least-cost-path reblocker on a pluggable routing substrate (default chord_diag,
    set in a later task; here grid). `repulsion` is the logit knob (s): s -> -inf straight
    (aspirational), 0 balanced, s -> +inf Voronoi-following (buildable)."""

    substrate: Substrate = field(default_factory=GridSubstrate)
    repulsion: float = 0.0
    depth_target: int = 2
    max_roads: int = 400

    @property
    def identity(self) -> tuple[object, ...]:
        return ("clearance", self.substrate.identity, float(self.repulsion),
                int(self.depth_target), int(self.max_roads))

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior  # accepted for Method conformance; the routing is block-only
        t = _sigmoid(self.repulsion)
        n_b = 0 if block.building_points.empty else len(block.building_points)
        radii = np.zeros(n_b, dtype=np.float64)   # plain clearance; weighted footprints are future
        graph = self.substrate.build(block)
        roads, params = _greedy_reblock(
            block, graph, t=t, depth_target=self.depth_target,
            max_roads=self.max_roads, radii=radii)
        pid = (f"clearance:{self.substrate.tag}:r{self.repulsion:g}"
               f":d{self.depth_target}:mr{self.max_roads}")
        return Proposal(
            block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
            proposal_id=pid, method="clearance",
            params={**params, "substrate": self.substrate.tag, "repulsion": self.repulsion,
                    "depth_target": self.depth_target},
            block_identity=block.identity)
```

Add `from dataclasses import dataclass, field` (add `field`) to the imports.

- [ ] **Step 5: Update `conf/method/clearance.yaml`**

Replace its body (remove `res`; add an inline grid substrate — Task 6 migrates this to the config group):

```yaml
_target_: reblock.methods.clearance.ClearanceReblocker
substrate:
  _target_: reblock.methods.substrates.GridSubstrate
  res: 1.5
repulsion: 0.0
depth_target: 2
max_roads: 400
```

(The existing config test in `test_clearance.py` asserts `method.identity == ("clearance", 0.0, 2, 1.5, 400)`; update it to `("clearance", ("grid", 1.5), 0.0, 2, 400)` and the target type to a `ClearanceReblocker` whose substrate is a `GridSubstrate`. Task 4 flips this to chord_diag.)

- [ ] **Step 6: Run tests + full gate + commit**

Run: `pixi run pytest tests/methods/test_clearance.py -q` → PASS. Then `pixi run check` → green.
```bash
git add src/reblock/methods/clearance.py conf/method/clearance.yaml tests/methods/test_clearance.py
git commit -m "refactor: clearance greedy takes a RoutingGraph; ClearanceReblocker(substrate=…), drop res

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

### Task 4: `ChordSubstrate` + make it the default

**Files:**
- Modify: `src/reblock/methods/substrates.py` (`_boundary_vertices`, `ChordSubstrate`)
- Modify: `src/reblock/methods/clearance.py` (default substrate → `ChordSubstrate`)
- Modify: `conf/method/clearance.yaml` (substrate → chord_diag)
- Test: `tests/methods/test_substrates.py`, `tests/methods/test_clearance.py`

**Interfaces:**
- Produces: `_boundary_vertices(parcels) -> tuple[NDArray[float64], dict[tuple[float,float], int]]` (node coords + coord→index map); `ChordSubstrate()` (`identity = ("chord_diag",)`, `tag = "chord_diag"`).
- Consumes: `reblock.methods.dijkstra._boundary_graph`, `dijkstra._rnd`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/methods/test_substrates.py` a fixture with building points + a chord test:

```python
from reblock.derive.access import parcel_access_layers
from reblock.methods.clearance import ClearanceReblocker, _greedy_reblock
from reblock.methods.substrates import ChordSubstrate


def _column_block_with_buildings(h: int) -> Block:
    polys = [Polygon([(0, j), (1, j), (1, j + 1), (0, j + 1)]) for j in range(h)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(h))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, unary_union(polys))
    from shapely.geometry import LineString
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs=UTM)
    pts = [g.representative_point() for g in parcels.geometry]
    bp = gpd.GeoDataFrame(geometry=pts, crs=UTM)
    return Block(block_id="colb", crs=UTM, boundary=boundary, parcels=parcels,
                 streets=streets, building_points=bp)


def test_chord_substrate_builds_connected_graph_and_hits_target() -> None:
    block = _column_block_with_buildings(8)
    graph = ChordSubstrate().build(block)
    assert len(graph.pts) > 0 and len(graph.rows) == len(graph.cols) == len(graph.edist)
    undirected = {frozenset((int(a), int(b))) for a, b in zip(graph.rows, graph.cols)}
    assert len(undirected) * 2 == len(graph.rows)          # symmetric
    roads, params = _greedy_reblock(block, graph, t=0.5, depth_target=2, max_roads=400,
                                    radii=np.zeros(len(block.building_points)))
    after = parcel_access_layers(block, roads).to_numpy()
    assert int(after.max()) <= 2 and params["grid_unreachable"] == 0
    assert ChordSubstrate().identity == ("chord_diag",) and ChordSubstrate().tag == "chord_diag"
```

And update `tests/methods/test_clearance.py` so the DEFAULT is chord_diag — add:

```python
def test_default_substrate_is_chord_diag() -> None:
    m = ClearanceReblocker(repulsion=0.0)
    assert m.substrate.tag == "chord_diag"
    p = m.propose(_column_block_with_buildings(6))
    assert p.proposal_id == "clearance:chord_diag:r0:d2:mr400"
    assert p.params["substrate"] == "chord_diag"
    after = parcel_access_layers(_column_block_with_buildings(6), p.roads).to_numpy()
    assert int(after.max()) <= 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/methods/test_substrates.py::test_chord_substrate_builds_connected_graph_and_hits_target tests/methods/test_clearance.py::test_default_substrate_is_chord_diag -x -q`
Expected: FAIL — `ChordSubstrate` undefined; default substrate is still grid.

- [ ] **Step 3: Add `_boundary_vertices` + `ChordSubstrate`**

In `src/reblock/methods/substrates.py`, add near the top:

```python
import geopandas as gpd

from reblock.derive.access import STREET_TOL
```

and the shared helper + chord substrate:

```python
def _boundary_vertices(
    parcels: gpd.GeoDataFrame,
) -> tuple[NDArray[np.float64], dict[tuple[float, float], int], set[frozenset[int]]]:
    """The parcel-tessellation boundary graph as (node coords, coord->index map, boundary-edge
    set). Nodes are the boundary vertices (snapped to cm by dijkstra._rnd), edges the party-wall
    segments. Shared node set for the chord / spanner / cdt substrates — nodes sit in the gaps
    between buildings, never on them."""
    from reblock.methods import dijkstra as dijkstra_mod
    g = dijkstra_mod._boundary_graph(parcels)
    nodes_sorted = sorted(g.nodes())
    node_idx = {n: i for i, n in enumerate(nodes_sorted)}
    pts = np.asarray(nodes_sorted, dtype=np.float64)
    edges = {frozenset((node_idx[a], node_idx[b])) for a, b in g.edges()}
    return pts, node_idx, edges


@dataclass(frozen=True)
class ChordSubstrate:
    """Boundary-vertex graph + ALL within-cell diagonals (every non-adjacent pair in each
    parcel's exterior ring; parcels are ~convex so every diagonal is interior/valid). Node count
    ∝ parcels (not area); the winner of the substrate head-to-head. `net_tol = STREET_TOL`."""

    @property
    def identity(self) -> Hashable:
        return ("chord_diag",)

    @property
    def tag(self) -> str:
        return "chord_diag"

    def build(self, block: Block) -> RoutingGraph:
        from reblock.methods import dijkstra as dijkstra_mod
        pts, node_idx, edges = _boundary_vertices(block.parcels)
        for geom in block.parcels.geometry:
            coords = list(geom.exterior.coords)[:-1]        # drop closing duplicate
            ring = [node_idx[ni] for c in coords if (ni := dijkstra_mod._rnd(c)) in node_idx]
            m = len(ring)
            if m < 3:
                continue
            for a in range(m):
                for b in range(a + 2, m):
                    if a == 0 and b == m - 1:
                        continue                            # wraparound-adjacent (a boundary edge)
                    if ring[a] != ring[b]:
                        edges.add(frozenset((ring[a], ring[b])))
        r, ro, co, di = _pack_edges(pts, edges)
        return RoutingGraph(r, ro, co, di, net_tol=STREET_TOL)
```

- [ ] **Step 4: Make chord_diag the default**

In `src/reblock/methods/clearance.py`, change the import and the default:

```python
from reblock.methods.substrates import ChordSubstrate, GridSubstrate, RoutingGraph, Substrate
```

```python
    substrate: Substrate = field(default_factory=ChordSubstrate)   # chord_diag — the head-to-head winner
```

In `conf/method/clearance.yaml`, swap the substrate to chord_diag:

```yaml
_target_: reblock.methods.clearance.ClearanceReblocker
substrate:
  _target_: reblock.methods.substrates.ChordSubstrate
repulsion: 0.0
depth_target: 2
max_roads: 400
```

Update the config-instantiation test in `test_clearance.py` to expect `("clearance", ("chord_diag",), 0.0, 2, 400)`.

- [ ] **Step 5: Run tests + full gate + commit**

Run: `pixi run pytest tests/methods/test_substrates.py tests/methods/test_clearance.py -q` → PASS. Then `pixi run check` → green.
```bash
git add src/reblock/methods/substrates.py src/reblock/methods/clearance.py conf/method/clearance.yaml tests/methods/test_substrates.py tests/methods/test_clearance.py
git commit -m "feat: ChordSubstrate (all within-cell diagonals) as the default clearance substrate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

### Task 5: `SpannerSubstrate` + `CdtSubstrate`

**Files:**
- Modify: `src/reblock/methods/substrates.py`
- Test: `tests/methods/test_substrates.py`

**Interfaces:**
- Produces: `SpannerSubstrate(cones: int = 6)` (`identity = ("theta_spanner", cones)`, tag `theta_spanner`); `CdtSubstrate()` (`identity = ("cdt_gap",)`, tag `cdt_gap`).
- Consumes: `_boundary_vertices`, `_pack_edges` (Task 4/1); `scipy.spatial.Delaunay`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/methods/test_substrates.py`:

```python
import pytest as _pytest  # (pytest already imported; harmless)
from reblock.methods.substrates import CdtSubstrate, SpannerSubstrate


@_pytest.mark.parametrize("sub,tag,ident", [
    (SpannerSubstrate(), "theta_spanner", ("theta_spanner", 6)),
    (CdtSubstrate(), "cdt_gap", ("cdt_gap",)),
])
def test_extra_substrates_build_and_hit_target(sub, tag, ident) -> None:
    block = _column_block_with_buildings(8)
    graph = sub.build(block)
    assert len(graph.pts) > 0 and len(graph.rows) == len(graph.cols) == len(graph.edist)
    undirected = {frozenset((int(a), int(b))) for a, b in zip(graph.rows, graph.cols)}
    assert len(undirected) * 2 == len(graph.rows)
    roads, params = _greedy_reblock(block, graph, t=0.5, depth_target=2, max_roads=400,
                                    radii=np.zeros(len(block.building_points)))
    after = parcel_access_layers(block, roads).to_numpy()
    assert int(after.max()) <= 2 and params["grid_unreachable"] == 0
    assert sub.tag == tag and sub.identity == ident
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/methods/test_substrates.py -x -q -k extra_substrates`
Expected: FAIL — `SpannerSubstrate`/`CdtSubstrate` undefined.

- [ ] **Step 3: Add the two substrates**

In `src/reblock/methods/substrates.py`, add `from scipy.spatial import Delaunay` and `import shapely` at the top, then:

```python
@dataclass(frozen=True)
class SpannerSubstrate:
    """Theta/Yao geometric spanner on the boundary vertices: per node, partition directions into
    `cones` angular cones and connect to the nearest node in each non-empty cone — an O(n·cones)-
    edge spanner with bounded stretch. Sparsest of the tessellation substrates. `net_tol =
    STREET_TOL`."""

    cones: int = 6

    @property
    def identity(self) -> Hashable:
        return ("theta_spanner", int(self.cones))

    @property
    def tag(self) -> str:
        return "theta_spanner"

    def build(self, block: Block) -> RoutingGraph:
        pts, _node_idx, edges = _boundary_vertices(block.parcels)
        n = len(pts)
        two_pi = 2.0 * np.pi
        cone_width = two_pi / self.cones
        for i in range(n):
            dx = pts[:, 0] - pts[i, 0]
            dy = pts[:, 1] - pts[i, 1]
            dist = np.hypot(dx, dy)
            cone = np.floor(np.mod(np.arctan2(dy, dx), two_pi) / cone_width).astype(np.int64)
            for c in range(self.cones):
                mask = (cone == c) & (dist > 0.0)
                if not np.any(mask):
                    continue
                idxs = np.flatnonzero(mask)
                j = int(idxs[np.argmin(dist[idxs])])
                edges.add(frozenset((i, j)))
        r, ro, co, di = _pack_edges(pts, edges)
        return RoutingGraph(r, ro, co, di, net_tol=STREET_TOL)


@dataclass(frozen=True)
class CdtSubstrate:
    """Delaunay triangulation of the boundary vertices, edges clipped to the block (an edge is
    kept only if its whole segment stays within the boundary, so it can't cut across a concave
    notch). Delaunay-SELECTED diagonals — sparser edges than chord_diag. `net_tol = STREET_TOL`."""

    @property
    def identity(self) -> Hashable:
        return ("cdt_gap",)

    @property
    def tag(self) -> str:
        return "cdt_gap"

    def build(self, block: Block) -> RoutingGraph:
        pts, _node_idx, edges = _boundary_vertices(block.parcels)
        pts_u = np.unique(pts, axis=0)
        if len(pts_u) >= 4:                                 # Delaunay needs >=3 non-collinear pts
            tri = Delaunay(pts_u)
            tri_edges: set[frozenset[int]] = set()
            for s in tri.simplices:
                i0, i1, i2 = (int(x) for x in s)
                tri_edges |= {frozenset((i0, i1)), frozenset((i1, i2)), frozenset((i0, i2))}
            ordered = sorted(tuple(sorted(e)) for e in tri_edges)
            from shapely.geometry import LineString
            boundary_buf = block.boundary.buffer(1e-6)      # tolerate exact frontage-segment runs
            segs = np.array([LineString([pts_u[i], pts_u[j]]) for i, j in ordered], dtype=object)
            keep = shapely.covers(boundary_buf, segs)
            # remap unique-point indices back to the boundary-vertex indices via coordinate match
            uidx = {(round(float(x), 3), round(float(y), 3)): k for k, (x, y) in enumerate(pts)}
            for k, ok in enumerate(keep):
                if ok:
                    i, j = ordered[k]
                    a = uidx[(round(float(pts_u[i, 0]), 3), round(float(pts_u[i, 1]), 3))]
                    b = uidx[(round(float(pts_u[j, 0]), 3), round(float(pts_u[j, 1]), 3))]
                    edges.add(frozenset((a, b)))
        r, ro, co, di = _pack_edges(pts, edges)
        return RoutingGraph(r, ro, co, di, net_tol=STREET_TOL)
```

> Note: the CDT builder triangulates `pts_u` (deduplicated) but the RoutingGraph's node array is the full boundary-vertex `pts` (so node indices stay consistent with the shared `edges`); the coordinate-keyed `uidx` remap bridges the two. Keeping the boundary edges (`edges` seed) guarantees connectivity even where the clip drops Delaunay edges.

- [ ] **Step 4: Run tests + full gate + commit**

Run: `pixi run pytest tests/methods/test_substrates.py -q` → PASS. `pixi run check` → green.
```bash
git add src/reblock/methods/substrates.py tests/methods/test_substrates.py
git commit -m "feat: SpannerSubstrate (theta) + CdtSubstrate (clipped Delaunay) routing substrates

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

### Task 6: Config group + compare registration

**Files:**
- Create: `conf/substrate/{grid,chord_diag,theta_spanner,cdt_gap}.yaml`
- Modify: `conf/config.yaml`, `conf/compare_config.yaml`, `conf/method/clearance.yaml`
- Test: `tests/methods/test_clearance.py`

**Interfaces:**
- Consumes: `hydra.utils.instantiate`, the substrate classes (Tasks 1/4/5).

- [ ] **Step 1: Write the failing tests**

Append to `tests/methods/test_clearance.py`:

```python
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def test_substrate_config_group_instantiates_each() -> None:
    conf_dir = str(Path(__file__).resolve().parents[2] / "conf")
    expected = {"grid": ("grid", 1.5), "chord_diag": ("chord_diag",),
                "theta_spanner": ("theta_spanner", 6), "cdt_gap": ("cdt_gap",)}
    for name, ident in expected.items():
        with initialize_config_dir(version_base=None, config_dir=conf_dir):
            cfg = compose(config_name="config", overrides=[f"substrate={name}"])
        sub = instantiate(cfg.substrate)
        assert sub.identity == ident


def test_clearance_method_defaults_to_chord_diag_substrate() -> None:
    conf_dir = str(Path(__file__).resolve().parents[2] / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config", overrides=["method=clearance"])
    method = instantiate(cfg.method)
    assert isinstance(method, ClearanceReblocker)
    assert method.substrate.tag == "chord_diag"


def test_compare_registers_clearance_and_grid_variant() -> None:
    conf_dir = str(Path(__file__).resolve().parents[2] / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="compare_config")
    assert instantiate(cfg.all_methods["clearance"]).substrate.tag == "chord_diag"
    assert instantiate(cfg.all_methods["clearance_grid"]).substrate.tag == "grid"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/methods/test_clearance.py -x -q -k "substrate_config or defaults_to_chord or registers_clearance"`
Expected: FAIL — no `conf/substrate/` group; no `clearance_grid` entry.

- [ ] **Step 3: Create the substrate config group**

`conf/substrate/grid.yaml`:
```yaml
_target_: reblock.methods.substrates.GridSubstrate
res: 1.5
```
`conf/substrate/chord_diag.yaml`:
```yaml
_target_: reblock.methods.substrates.ChordSubstrate
```
`conf/substrate/theta_spanner.yaml`:
```yaml
_target_: reblock.methods.substrates.SpannerSubstrate
cones: 6
```
`conf/substrate/cdt_gap.yaml`:
```yaml
_target_: reblock.methods.substrates.CdtSubstrate
```

- [ ] **Step 4: Wire the group into the top configs**

Add to `conf/config.yaml`'s `defaults` list a `- substrate: chord_diag` entry (mirroring how `region_builder`/`method` groups are listed there — check the existing file and match its style). Do the same in `conf/compare_config.yaml`'s `defaults`.

Change `conf/method/clearance.yaml` to consume the composed group by interpolation (so `substrate=grid` at the CLI swaps it):

```yaml
_target_: reblock.methods.clearance.ClearanceReblocker
substrate: ${substrate}
repulsion: 0.0
depth_target: 2
max_roads: 400
```

In `conf/compare_config.yaml` `all_methods`, make the `clearance` entry reference the group and add a grid reference entry:

```yaml
  clearance: {_target_: reblock.methods.clearance.ClearanceReblocker, substrate: ${substrate}}
  clearance_grid: {_target_: reblock.methods.clearance.ClearanceReblocker, substrate: {_target_: reblock.methods.substrates.GridSubstrate, res: 1.5}}
```

> Implementer note: verify against the repo's existing config-group usage (`conf/region_builder/` is referenced from `compare_config.yaml`'s `defaults` as `- region_builder: identity`). If `${substrate}` interpolation into `all_methods` doesn't resolve under `compose`, fall back to an inline nested substrate for the `clearance` entry (`substrate: {_target_: ...ChordSubstrate}`) and keep the group for `method=clearance` runs — the three tests above are the acceptance gate; make them green.

- [ ] **Step 5: Run tests + full gate + commit**

Run: `pixi run pytest tests/methods/test_clearance.py -q` → PASS. `pixi run check` → green.
```bash
git add conf/substrate conf/config.yaml conf/compare_config.yaml conf/method/clearance.yaml tests/methods/test_clearance.py
git commit -m "feat: conf/substrate config group + clearance/clearance_grid compare entries

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

### Task 7: Regenerate the `clearance-repulsion` gallery

**Files:**
- Modify: `examples/clearance-repulsion/generate.py` (only if the `ClearanceReblocker(...)` call needs updating), `README.md`, and the six `*.png`.

This task is not pytest-gated (it runs on `capetown_full`, cached at `~/.cache/reblock`). Validate by running.

- [ ] **Step 1: Check the generator still calls the new interface**

Read `examples/clearance-repulsion/generate.py`. It calls `ClearanceReblocker(repulsion=s).propose(region)` — which now defaults to chord_diag with no `res`, so it needs **no change**. (If it passed `res=`, remove it.) Confirm by reading; do not edit unless it references `res`.

- [ ] **Step 2: Regenerate**

Run: `PYTHONPATH=. PYTHONUNBUFFERED=1 pixi run python examples/clearance-repulsion/generate.py`
Expected: prints the auto-detected region + a 5-row table (repulsion → roads, length_m, displaced, max_depth); rewrites `before.png` + five `after_s*.png`. Every row's `max_depth ≤ 2`. Roads are now the chord_diag (straighter, sparser) network — the counts/lengths/displacements will differ from the old grid gallery.

- [ ] **Step 3: Update the README numbers**

In `examples/clearance-repulsion/README.md`, replace the metrics table + any inline numbers with the freshly printed values (roads/length/displaced/max_depth per repulsion, and the flagged `X of Y` count). Note the substrate is now chord_diag. No placeholders — every number from the run. The `s=-6`/`s=-2` identical-panel caption may no longer hold under chord_diag; keep it only if the two panels are still byte-identical (check the printed table), else drop it.

- [ ] **Step 4: Commit**

```bash
git add examples/clearance-repulsion/
git commit -m "docs: regenerate clearance-repulsion gallery under chord_diag + 3-point default

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x"
```

---

## Self-Review

**Spec coverage:**
- Substrate protocol + RoutingGraph → Task 1. ✓
- GridSubstrate/ChordSubstrate/SpannerSubstrate/CdtSubstrate/PrebuiltSubstrate → Tasks 1/4/5. ✓ (cdt_bldg excluded ✓)
- 3-point sampling everywhere (one rule) → Task 2. ✓
- Substrate-agnostic greedy + `ClearanceReblocker(substrate=…)`, remove res → Task 3. ✓
- chord_diag default → Task 4. ✓
- substrate-tagged proposal_id + identity → Tasks 3/4. ✓
- conf/substrate group + compare clearance/clearance_grid → Task 6. ✓
- `_DERIVATION_MODULES` registration → Task 1. ✓
- Regenerate clearance-repulsion (migration) → Task 7. ✓
- Depth-seeding precondition preserved → Task 3 (kept `unreached_depth=len+1`). ✓
- Determinism (sorted nodes/edges) → `_pack_edges`/`_boundary_vertices` sort. ✓
- Flagship + README rework EXCLUDED (deferred) → not in plan. ✓

**Placeholder scan:** No TBD/TODO. Every code step has complete code. Task 6 flags a Hydra-interpolation risk with a concrete fallback + the acceptance test as the gate (not a placeholder — a named contingency).

**Type consistency:** `_greedy_reblock(block, graph, *, t, depth_target, max_roads, radii)` signature identical across Tasks 3/4/5 tests and its definition. `_edge_weights(pts, rows, cols, edist, building_pts, radii, t)` consistent between Task 2 def and the Task 3 call site. Substrate `identity`/`tag` shapes match between each class and the tests that assert them (`("grid", res)`, `("chord_diag",)`, `("theta_spanner", 6)`, `("cdt_gap",)`). `proposal_id` format `clearance:{tag}:r{}:d{}:mr{}` consistent (Task 3 grid → `clearance:grid:…`, Task 4 default → `clearance:chord_diag:…`). `RoutingGraph` field names (pts/rows/cols/edist/net_tol) consistent throughout.
