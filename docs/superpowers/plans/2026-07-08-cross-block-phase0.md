# Cross-block reblocking Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the cross-block foundation — a super-block merge, an orthogonal network-quality metric basis, and a falsifiable probe — that measures whether cross-block reblocking has real headroom over block-local reblocking, before any Phase-1 placement method is built.

**Architecture:** `merge_cluster` folds a set of adjacent blocks (selected by `block_ids`) into one synthetic `Block`, so every existing derivation runs on it unchanged. A correctly-noded planar graph of (roads ∪ streets) feeds a metric basis (one representative per axis: reachability, equity, directness, throughput, redundancy, permeability, cost, cross-block). A probe script runs the basis over a stratified sample of real Cape Town clusters on the boundary-reconciled block-local peel baseline (and a heuristic cross-block reference), validates the basis's orthogonality with a correlation matrix, and reports the headroom for a documented go/no-go.

**Tech Stack:** Python 3.12, geopandas + shapely 2.1 (`set_precision`, `union_all`, `make_valid`, `buffer`), networkx (`maximum_flow_value`, graph metrics), pandas/numpy, pixi (`pixi run check` = ruff + `mypy --strict src tests` + pytest), Hydra (`_target_` config groups).

## Global Constraints

- `pixi run check` green (ruff + `mypy --strict src tests` + pytest) before every commit.
- **Additive only:** create new files under `src/reblock/derive/`, `src/reblock/eval/`, `conf/eval/`, `scripts/`, `tests/`. Do NOT edit `contracts.py`, `run.py`, `ShapefileSource`, or the existing methods/evals/derivations.
- **Reuse existing derivations unchanged** on the merged block: `parcel_access_layers`, `street_connectivity`, `geometric_access_distances`, `parcel_adjacency`, `KComplexityEval` (all key on `parcel_id`/geometry, not `block_id`).
- **Deterministic:** sorted `block_id` order, sequential `parcel_id`, no RNG anywhere. Baseline method = `PeelReblocker` (deterministic); do NOT use `TopologyMethod` (seeds an RNG).
- **`STREET_TOL = 0.5`** (`from reblock.derive.access import STREET_TOL`) is the snap/precision grid everywhere.
- **Correct noding:** `shapely.set_precision(geom, STREET_TOL)` on every line **before** `union_all` — never round after.
- **`n_cross_block_streets`** = a road with vertices strictly on **both sides** of an interior boundary line (not `.crosses()`).
- **Decision A:** the merged block's `streets` keeps interior former-boundaries (they are real streets).
- **Types:** every function fully annotated for `mypy --strict`; new-style unions (`X | None`), `list[...]`.

---

### Task 1: `merge_cluster` — fold adjacent blocks into a super-block

**Files:**
- Create: `src/reblock/derive/cluster.py`
- Test: `tests/derive/test_cluster.py`

**Interfaces:**
- Consumes: `reblock.contracts.{Block, Region}`; `reblock.derive.access.STREET_TOL`.
- Produces: `merge_cluster(region: Region) -> Block` — the merged block; `block.attrs["block_ids"]: list[str]`, `block.attrs["interior_boundaries"]: MultiLineString` (the shared inter-block frontage lines). Raises `ValueError` if the blocks are not contiguous.

- [ ] **Step 1: Write failing tests** in `tests/derive/test_cluster.py`

```python
from typing import cast

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import MultiLineString, Polygon, box

from reblock.contracts import Block, Region
from reblock.derive.cluster import merge_cluster

UTM = CRS.from_epsg(32734)


def _block(bid: str, poly: Polygon, n: int) -> Block:
    # n unit-ish parcels tiling the block, ids 0..n-1; streets = the block boundary.
    minx, miny, maxx, maxy = poly.bounds
    w = (maxx - minx) / n
    polys = [box(minx + i * w, miny, minx + (i + 1) * w, maxy) for i in range(n)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(n))}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[poly.boundary], crs=UTM)
    return Block(block_id=bid, crs=UTM, boundary=poly, parcels=parcels, streets=streets,
                 attrs={"kblock_k": 3.0})


def _region(*blocks: Block) -> Region:
    return Region(region_id="t", crs=UTM, blocks=list(blocks))


def test_merge_two_adjacent_blocks() -> None:
    a = _block("a", box(0, 0, 10, 10), 2)
    b = _block("b", box(10, 0, 20, 10), 3)          # shares the x=10 edge with a
    m = merge_cluster(_region(a, b))
    assert isinstance(m.boundary, Polygon)           # contiguous -> single polygon
    assert len(m.parcels) == 5                        # 2 + 3
    assert list(m.parcels["parcel_id"]) == [0, 1, 2, 3, 4]   # re-indexed, unique
    assert m.attrs["block_ids"] == ["a", "b"]
    assert isinstance(m.attrs["interior_boundaries"], MultiLineString)
    assert cast(MultiLineString, m.attrs["interior_boundaries"]).length > 0   # the shared x=10 edge


def test_merge_non_adjacent_raises() -> None:
    a = _block("a", box(0, 0, 10, 10), 2)
    c = _block("c", box(50, 50, 60, 60), 2)          # disjoint from a
    with pytest.raises(ValueError, match="not contiguous"):
        merge_cluster(_region(a, c))
```

- [ ] **Step 2: Run to verify fail** → `pixi run pytest tests/derive/test_cluster.py -x` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `src/reblock/derive/cluster.py`

```python
"""merge_cluster: fold a cluster of adjacent blocks into one synthetic super-block so
every per-block derivation (parcel_access_layers, geometric_access_distances,
KComplexityEval, ...) runs on it unchanged. Interior former-boundaries are kept as
real streets (Decision A); the shared frontage lines are exposed via
attrs["interior_boundaries"] for the cross-block metrics.
"""
from __future__ import annotations

import pandas as pd
import geopandas as gpd
from shapely import make_valid, union_all
from shapely.geometry import MultiLineString, Polygon
from shapely.geometry.base import BaseGeometry

from reblock.contracts import Block, Region
from reblock.derive.access import STREET_TOL


def _blocks_sorted(region: Region) -> list[Block]:
    return sorted(region.blocks, key=lambda b: b.block_id)


def _interior_boundaries(blocks: list[Block], tol: float) -> MultiLineString:
    """The shared frontage lines between adjacent blocks (each pair's boundary
    intersection, kept only where it is a positive-length line)."""
    lines: list[BaseGeometry] = []
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            shared = make_valid(blocks[i].boundary).intersection(make_valid(blocks[j].boundary))
            if shared.length > 0:
                lines.append(shared)
    merged = union_all(lines) if lines else MultiLineString([])
    return merged if isinstance(merged, MultiLineString) else MultiLineString([merged])


def merge_cluster(region: Region) -> Block:
    blocks = _blocks_sorted(region)
    if not blocks:
        raise ValueError(f"{region.region_id}: cluster has no blocks")
    crs = blocks[0].crs

    boundary = make_valid(union_all([b.boundary for b in blocks]))
    if not isinstance(boundary, Polygon):
        raise ValueError(
            f"{region.region_id}: blocks are not contiguous (union is "
            f"{boundary.geom_type}, not a single Polygon): {[b.block_id for b in blocks]}")

    parcels = pd.concat([b.parcels[["geometry"]] for b in blocks], ignore_index=True)
    parcels = gpd.GeoDataFrame(
        {"parcel_id": list(range(len(parcels)))}, geometry=parcels.geometry.to_numpy(), crs=crs)

    streets = gpd.GeoDataFrame(
        geometry=[union_all([g for b in blocks for g in b.streets.geometry])], crs=crs)

    interior = _interior_boundaries(blocks, STREET_TOL)
    return Block(
        block_id="+".join(b.block_id for b in blocks), crs=crs, boundary=boundary,
        parcels=parcels, streets=streets,
        attrs={"block_ids": [b.block_id for b in blocks], "interior_boundaries": interior})
```

- [ ] **Step 4: Run tests** → `pixi run pytest tests/derive/test_cluster.py -v` → PASS.

- [ ] **Step 5: `pixi run check`** → green. **Commit:**

```bash
git add src/reblock/derive/cluster.py tests/derive/test_cluster.py
git commit -m "feat: merge_cluster — fold adjacent blocks into a super-block"
```

---

### Task 2: Noded planar graph + structural metrics

**Files:**
- Create: `src/reblock/derive/network_metrics.py`
- Test: `tests/derive/test_network_metrics.py`

**Interfaces:**
- Consumes: `reblock.derive.access.STREET_TOL`.
- Produces:
  - `node_network(roads: gpd.GeoDataFrame | None, streets: gpd.GeoDataFrame, tol: float = STREET_TOL) -> nx.Graph` — nodes are `(x, y)` float tuples (snapped to `tol`), edges carry `length` and `is_road: bool`.
  - `meshedness(graph: nx.Graph) -> float`; `degree_fractions(graph: nx.Graph) -> dict[str, float]` returning `{"four_way_fraction", "dead_end_fraction", "t_fraction"}`; `crossing_counts(graph: nx.Graph) -> dict[str, int]` returning `{"n_crossings", "n_t_junctions", "n_dead_ends"}`.

- [ ] **Step 1: Write failing tests** in `tests/derive/test_network_metrics.py`

```python
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString

from reblock.derive.network_metrics import (
    crossing_counts, degree_fractions, meshedness, node_network)

UTM = CRS.from_epsg(32734)


def _lines(*coords: list[tuple[float, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[LineString(c) for c in coords], crs=UTM)


def test_plus_is_one_crossing() -> None:
    # a "+" of two lines crossing at the origin -> one degree-4 node, 4 dead-end tips
    roads = _lines([(-1, 0), (1, 0)], [(0, -1), (0, 1)])
    empty = gpd.GeoDataFrame(geometry=[], crs=UTM)
    g = node_network(roads, empty)
    cc = crossing_counts(g)
    assert cc["n_crossings"] == 1
    assert cc["n_dead_ends"] == 4
    assert degree_fractions(g)["four_way_fraction"] > 0


def test_near_miss_nodes_via_set_precision() -> None:
    # two stubs 0.4 m apart (< STREET_TOL=0.5) crossed by a through-line: set_precision
    # snaps them so the crossing is a single degree-4 node, not a missed near-miss.
    roads = _lines([(-1, 0), (-0.2, 0)], [(0.2, 0), (1, 0)], [(0, -1), (0, 1)])
    empty = gpd.GeoDataFrame(geometry=[], crs=UTM)
    g = node_network(roads, empty)
    assert crossing_counts(g)["n_crossings"] == 1


def test_tree_has_zero_meshedness_grid_positive() -> None:
    # a path (tree) has no cycles; a 2x2 grid of squares has cycles
    tree = _lines([(0, 0), (1, 0)], [(1, 0), (2, 0)], [(1, 0), (1, 1)])
    empty = gpd.GeoDataFrame(geometry=[], crs=UTM)
    assert meshedness(node_network(tree, empty)) == 0.0
    grid = _lines([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])   # one closed loop
    assert meshedness(node_network(grid, empty)) > 0.0
```

- [ ] **Step 2: Run to verify fail** → `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `src/reblock/derive/network_metrics.py`

```python
"""Network-quality metrics on a noded planar graph of (proposed roads ∪ existing
streets). Correct noding is load-bearing: set_precision(grid≈STREET_TOL) on every line
BEFORE union_all, so real ~0.5 m cadastral drift nodes at true intersections instead of
being missed (a post-hoc round only relabels, it cannot split an edge).
"""
from __future__ import annotations

import geopandas as gpd
import networkx as nx
from shapely import line_merge, set_precision, union_all
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry

from reblock.derive.access import STREET_TOL

_Node = tuple[float, float]


def _snap(x: float, y: float, tol: float) -> _Node:
    return (round(x / tol) * tol, round(y / tol) * tol)


def _iter_lines(geom: BaseGeometry) -> list[LineString]:
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    return []


def node_network(
    roads: gpd.GeoDataFrame | None, streets: gpd.GeoDataFrame, tol: float = STREET_TOL,
) -> nx.Graph:
    road_lines = list(roads.geometry) if roads is not None and not roads.empty else []
    street_lines = list(streets.geometry) if not streets.empty else []
    tagged: list[tuple[LineString, bool]] = []
    for is_road, src in ((True, road_lines), (False, street_lines)):
        snapped = union_all([set_precision(g, tol) for g in src]) if src else None
        if snapped is None or snapped.is_empty:
            continue
        for ls in _iter_lines(line_merge(snapped)):
            tagged.append((ls, is_road))

    graph: nx.Graph = nx.Graph()
    for ls, is_road in tagged:
        coords = list(ls.coords)
        for (x0, y0), (x1, y1) in zip(coords, coords[1:], strict=False):
            u, v = _snap(x0, y0, tol), _snap(x1, y1, tol)
            if u == v:
                continue
            seg = LineString([u, v])
            if graph.has_edge(u, v):
                graph[u][v]["is_road"] = graph[u][v]["is_road"] or is_road
            else:
                graph.add_edge(u, v, length=seg.length, is_road=is_road)
    return graph


def meshedness(graph: nx.Graph) -> float:
    """(E - N + C) / (2N - 5): 0 for a tree/forest, up to 1 for a maximal planar mesh."""
    n, e = graph.number_of_nodes(), graph.number_of_edges()
    if n < 3:
        return 0.0
    c = nx.number_connected_components(graph)
    denom = 2 * n - 5
    return max(0.0, (e - n + c) / denom) if denom > 0 else 0.0


def degree_fractions(graph: nx.Graph) -> dict[str, float]:
    n = graph.number_of_nodes()
    if n == 0:
        return {"four_way_fraction": 0.0, "dead_end_fraction": 0.0, "t_fraction": 0.0}
    degs = [d for _, d in graph.degree()]
    return {
        "four_way_fraction": sum(d >= 4 for d in degs) / n,
        "dead_end_fraction": sum(d == 1 for d in degs) / n,
        "t_fraction": sum(d == 3 for d in degs) / n,
    }


def crossing_counts(graph: nx.Graph) -> dict[str, int]:
    """Bare-degree node taxonomy (the through-going/collinear refinement is deferred to
    Phase 1): degree>=4 -> crossing, degree 3 -> T, degree-1 road tip -> dead-end."""
    n_cross = n_t = n_dead = 0
    for node, deg in graph.degree():
        if deg >= 4:
            n_cross += 1
        elif deg == 3:
            n_t += 1
        elif deg == 1 and any(graph[node][nb]["is_road"] for nb in graph[node]):
            n_dead += 1
    return {"n_crossings": n_cross, "n_t_junctions": n_t, "n_dead_ends": n_dead}
```

- [ ] **Step 4: Run tests** → PASS. **Step 5: `pixi run check`** → green. **Commit:**

```bash
git add src/reblock/derive/network_metrics.py tests/derive/test_network_metrics.py
git commit -m "feat: noded planar graph + structural network metrics"
```

---

### Task 3: Cross-block + directness metrics

**Files:**
- Modify: `src/reblock/derive/network_metrics.py`
- Modify: `tests/derive/test_network_metrics.py`

**Interfaces:**
- Consumes: Task-1 `merge_cluster` attrs (`interior_boundaries`); `reblock.derive.geometric_access.geometric_access_distances`.
- Produces (add to `network_metrics.py`):
  - `n_cross_block_streets(roads: gpd.GeoDataFrame | None, interior: MultiLineString, tol: float = STREET_TOL) -> int`
  - `cross_block_trunk_length_m(roads, interior, tol=STREET_TOL) -> float` (total length of road features that cross a boundary)
  - `boundary_redundant_road_fraction(roads, interior, tol=STREET_TOL, band: float = 20.0) -> float`
  - `circuity(block: Block, roads: gpd.GeoDataFrame | None, tol: float = STREET_TOL) -> float`

- [ ] **Step 1: Add failing tests** to `tests/derive/test_network_metrics.py`

```python
from shapely.geometry import MultiLineString
from reblock.derive.network_metrics import (
    boundary_redundant_road_fraction, circuity, cross_block_trunk_length_m,
    n_cross_block_streets)

VBOUND = MultiLineString([[(0, -5), (0, 5)]])   # the interior boundary line x=0


def test_cross_block_counts_both_sides_only() -> None:
    crossing = _lines([(-2, 0), (2, 0)])          # vertices strictly on both sides of x=0
    along = _lines([(0, -3), (0, 3)])             # runs ALONG the boundary -> not a crossing
    kiss = _lines([(-2, 1), (0, 0), (-2, -1)])    # touches x=0 but stays on the left -> not
    assert n_cross_block_streets(crossing, VBOUND) == 1
    assert n_cross_block_streets(along, VBOUND) == 0
    assert n_cross_block_streets(kiss, VBOUND) == 0
    assert cross_block_trunk_length_m(crossing, VBOUND) == 4.0


def test_circuity_straight_is_one_detour_is_more() -> None:
    from reblock.contracts import Block
    import geopandas as gpd
    from shapely.geometry import Polygon, box
    # a 1x5 strip; street on the left edge. Parcel k sits ~k m along the adjacency chain
    # but only ~k m straight-line -> circuity ~1 for a strip (adjacency ~ euclidean here).
    polys = [box(i, 0, i + 1, 1) for i in range(5)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(5))}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])], crs=UTM)
    block = Block(block_id="s", crs=UTM, boundary=Polygon([(0, 0), (5, 0), (5, 1), (0, 1)]),
                  parcels=parcels, streets=streets)
    c = circuity(block, None)
    assert c >= 1.0 and c < 1.5
```

- [ ] **Step 2: Run to verify fail** → `ImportError`.

- [ ] **Step 3: Append implementations** to `src/reblock/derive/network_metrics.py`

```python
import pandas as pd
from shapely import union_all
from reblock.contracts import Block
from reblock.derive.geometric_access import geometric_access_distances


def _road_lines(roads: gpd.GeoDataFrame | None) -> list[LineString]:
    if roads is None or roads.empty:
        return []
    out: list[LineString] = []
    for g in roads.geometry:
        out.extend(_iter_lines(g))
    return out


def _side(b: LineString, x: float, y: float) -> int:
    """Which side of the chord through b's endpoints the point (x, y) lies on
    (+1 / -1), or 0 if on it. Endpoint-chord is exact for straight shared frontages
    and an accepted approximation for gently-curved ones (a genuine crossing still
    has vertices clearly on both sides)."""
    (x0, y0), (x1, y1) = b.coords[0], b.coords[-1]
    cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
    return 1 if cross > 1e-9 else (-1 if cross < -1e-9 else 0)


def _crosses_boundary(line: LineString, interior: MultiLineString, tol: float) -> bool:
    """True iff `line` has vertices strictly on both sides of some interior boundary
    segment AND runs within `tol` of it — robust where shapely's `.crosses` is not
    (runs-along -> side 0 only; kiss-and-bounce -> one side only)."""
    for b in interior.geoms:
        sides = {_side(b, x, y) for x, y in line.coords}
        if {-1, 1} <= sides and line.distance(b) <= tol:
            return True
    return False


def n_cross_block_streets(
    roads: gpd.GeoDataFrame | None, interior: MultiLineString, tol: float = STREET_TOL) -> int:
    if interior.is_empty:
        return 0
    return sum(_crosses_boundary(ls, interior, tol) for ls in _road_lines(roads))


def cross_block_trunk_length_m(
    roads: gpd.GeoDataFrame | None, interior: MultiLineString, tol: float = STREET_TOL) -> float:
    if interior.is_empty:
        return 0.0
    return float(sum(ls.length for ls in _road_lines(roads)
                     if _crosses_boundary(ls, interior, tol)))


def boundary_redundant_road_fraction(
    roads: gpd.GeoDataFrame | None, interior: MultiLineString, tol: float = STREET_TOL,
    band: float = 20.0) -> float:
    """Fraction of road length running within `band` of an interior boundary WITHOUT
    crossing it — the boundary-parallel spine road a shared through-trunk would merge."""
    lines = _road_lines(roads)
    total = sum(ls.length for ls in lines)
    if total == 0 or interior.is_empty:
        return 0.0
    corridor = interior.buffer(band)
    redundant = sum(ls.intersection(corridor).length for ls in lines
                    if not _crosses_boundary(ls, interior, tol))
    return float(redundant / total)


def circuity(block: Block, roads: gpd.GeoDataFrame | None, tol: float = STREET_TOL) -> float:
    """mean(network distance / straight-line distance) from each parcel to the nearest
    street, over parcels genuinely off-street. Floor 1.0 (direct); higher = detours."""
    net = geometric_access_distances(block, roads, tol=tol)
    street = union_all(list(block.streets.geometry)
                       + _road_lines(roads))     # type: ignore[arg-type]
    ratios: list[float] = []
    for pid, geom in zip(block.parcels["parcel_id"], block.parcels.geometry, strict=True):
        euc = geom.representative_point().distance(street)
        nd = float(net.loc[pid])
        if euc > tol and nd < float("inf"):
            ratios.append(nd / euc)
    return float(pd.Series(ratios).mean()) if ratios else 1.0
```

- [ ] **Step 4: Run tests** → PASS. **Step 5: `pixi run check`** → green. **Commit:**

```bash
git add src/reblock/derive/network_metrics.py tests/derive/test_network_metrics.py
git commit -m "feat: cross-block + directness (circuity) network metrics"
```

---

### Task 4: Throughput (max-flow interior → perimeter)

**Files:**
- Modify: `src/reblock/derive/network_metrics.py`
- Modify: `tests/derive/test_network_metrics.py`

**Interfaces:**
- Consumes: Task-2 `node_network` graph; the merged `Block`.
- Produces: `throughput_ratio(graph: nx.Graph, block: Block, tol: float = STREET_TOL) -> float` — max-flow from a unit-demand super-source over parcel access-nodes to a super-sink at the perimeter, normalized by the parcel count. `1.0` = no bottleneck; lower = the network chokes.

- [ ] **Step 1: Add a failing test** to `tests/derive/test_network_metrics.py`

```python
def test_throughput_tree_bottlenecks_grid_does_not() -> None:
    from reblock.contracts import Block
    import geopandas as gpd
    from shapely.geometry import Polygon, box
    from reblock.derive.network_metrics import node_network, throughput_ratio
    # 3 interior parcels behind a single-file corridor (tree) -> min-cut 1 -> ratio ~1/3;
    # the same parcels with a second parallel corridor -> higher throughput.
    parcels = gpd.GeoDataFrame({"parcel_id": [0, 1, 2]},
                               geometry=[box(1, 0, 2, 1), box(2, 0, 3, 1), box(3, 0, 4, 1)], crs=UTM)
    boundary = Polygon([(0, 0), (4, 0), (4, 1), (0, 1)])
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])], crs=UTM)   # left perimeter
    block = Block(block_id="s", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)
    single = _lines([(0.5, 0.5), (3.5, 0.5)])                       # one spine
    g1 = node_network(single, streets)
    both = _lines([(0.5, 0.2), (3.5, 0.2)], [(0.5, 0.8), (3.5, 0.8)])  # two parallel spines
    g2 = node_network(both, streets)
    assert throughput_ratio(g2, block) >= throughput_ratio(g1, block)
```

- [ ] **Step 2: Run to verify fail** → `ImportError`.

- [ ] **Step 3: Append implementation** to `src/reblock/derive/network_metrics.py`

```python
from shapely import STRtree
from shapely.geometry import Point


def throughput_ratio(graph: nx.Graph, block: Block, tol: float = STREET_TOL) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    nodes = list(graph.nodes)
    node_pts = [Point(n) for n in nodes]
    tree = STRtree(node_pts)

    flow: nx.DiGraph = nx.DiGraph()
    for u, v in graph.edges:
        flow.add_edge(u, v, capacity=1.0)
        flow.add_edge(v, u, capacity=1.0)

    perim = block.boundary.exterior
    sink = "__SINK__"
    for n, pt in zip(nodes, node_pts, strict=True):
        if pt.distance(perim) <= tol:
            flow.add_edge(n, sink, capacity=float("inf"))

    src = "__SRC__"
    demand = 0
    for geom in block.parcels.geometry:
        idx = int(tree.nearest(geom.representative_point()))
        node = nodes[idx]
        if flow.has_edge(src, node):
            flow[src][node]["capacity"] += 1.0
        else:
            flow.add_edge(src, node, capacity=1.0)
        demand += 1

    if demand == 0 or src not in flow or sink not in flow:
        return 0.0
    return float(nx.maximum_flow_value(flow, src, sink)) / demand
```

- [ ] **Step 4: Run tests** → PASS. **Step 5: `pixi run check`** → green. **Commit:**

```bash
git add src/reblock/derive/network_metrics.py tests/derive/test_network_metrics.py
git commit -m "feat: throughput (max-flow interior->perimeter) metric"
```

---

### Task 5: `StructureEval` — assemble the basis into a `Metrics`

**Files:**
- Create: `src/reblock/eval/structure.py`
- Create: `conf/eval/structure.yaml`
- Test: `tests/eval/test_structure.py`

**Interfaces:**
- Consumes: all of `network_metrics`; `reblock.contracts.{Block, Proposal, Metrics}`; `reblock.derive.geometric_access.geometric_access_distances`.
- Produces: `StructureEval` conforming to `Eval` — `score(block, proposal) -> Metrics(eval="structure", values={...})` emitting the full basis (minus the deferred smoothness axis).

- [ ] **Step 1: Write failing test** in `tests/eval/test_structure.py`

```python
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon, box

from reblock.contracts import Block, Proposal
from reblock.eval.structure import StructureEval

UTM = CRS.from_epsg(32734)


def _grid_block(n: int) -> Block:
    polys, ids = [], []
    for i in range(n):
        for j in range(n):
            polys.append(box(i, j, i + 1, j + 1))
            ids.append(i * n + j)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    boundary = Polygon([(0, 0), (n, 0), (n, n), (0, n)])
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_structure_eval_emits_the_basis() -> None:
    block = _grid_block(3)
    roads = gpd.GeoDataFrame(geometry=[LineString([(1.5, 0), (1.5, 3)])], crs=UTM)
    m = StructureEval().score(block, Proposal(block_id="g", crs=UTM, roads=roads, method="x"))
    for key in ("meshedness", "four_way_fraction", "dead_end_fraction", "n_crossings",
                "n_dead_ends", "circuity", "throughput_ratio", "geometric_access_p95_m",
                "added_road_length_per_parcel", "n_cross_block_streets"):
        assert key in m.values
    assert m.eval == "structure"
    assert m.values["circuity"] >= 1.0
    assert m.values["n_cross_block_streets"] == 0.0   # no interior_boundaries on a lone block
```

- [ ] **Step 2: Run to verify fail** → `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `src/reblock/eval/structure.py`

```python
"""StructureEval: the orthogonal network-quality metric basis for a (block, proposal),
emitted as a Metrics. Reachability/equity via geometric access, directness via circuity,
throughput via max-flow, redundancy/permeability/crossings via the noded graph,
cross-block continuity via the interior boundaries. Smoothness (Axis I) is deferred to
the arc-emitting Phase-1 slice.
"""
from __future__ import annotations

from shapely.geometry import MultiLineString

from reblock.contracts import Block, Metrics, Proposal
from reblock.derive.geometric_access import geometric_access_distances
from reblock.derive.network_metrics import (
    boundary_redundant_road_fraction, circuity, cross_block_trunk_length_m, crossing_counts,
    degree_fractions, meshedness, n_cross_block_streets, node_network, throughput_ratio)


class StructureEval:
    def score(self, block: Block, proposal: Proposal) -> Metrics:
        roads = proposal.roads
        interior = block.attrs.get("interior_boundaries")
        if not isinstance(interior, MultiLineString):
            interior = MultiLineString([])

        graph = node_network(roads, block.streets)
        geo = geometric_access_distances(block, roads)
        n_parcels = max(len(block.parcels), 1)
        road_len = float(roads.geometry.length.sum()) if roads is not None and not roads.empty else 0.0

        values = {
            # A reachability
            "geometric_access_max_m": float(geo.max()) if len(geo) else 0.0,
            # B equity
            "geometric_access_p95_m": float(geo.quantile(0.95)) if len(geo) else 0.0,
            # C directness
            "circuity": circuity(block, roads),
            # D throughput
            "throughput_ratio": throughput_ratio(graph, block),
            # E redundancy
            "meshedness": meshedness(graph),
            # G cost
            "added_road_length_per_parcel": road_len / n_parcels,
            # H cross-block
            "n_cross_block_streets": float(n_cross_block_streets(roads, interior)),
            "cross_block_trunk_length_m": cross_block_trunk_length_m(roads, interior),
            "boundary_redundant_road_fraction": boundary_redundant_road_fraction(roads, interior),
        }
        values.update({k: float(v) for k, v in degree_fractions(graph).items()})  # F permeability
        values.update({k: float(v) for k, v in crossing_counts(graph).items()})   # F crossings/T/dead
        return Metrics(block_id=block.block_id, method=proposal.method, eval="structure", values=values)
```

- [ ] **Step 4: Create** `conf/eval/structure.yaml`

```yaml
# A list (see conf/eval/kcomplexity.yaml): run() instantiates cfg.eval as a list of Evals.
- _target_: reblock.eval.structure.StructureEval
```

- [ ] **Step 5: Run tests** → PASS. **Step 6: `pixi run check`** → green. **Commit:**

```bash
git add src/reblock/eval/structure.py conf/eval/structure.yaml tests/eval/test_structure.py
git commit -m "feat: StructureEval — orthogonal network-quality metric basis"
```

---

### Task 6: Boundary-reconciled baseline + heuristic spine-merge reference

**Files:**
- Create: `src/reblock/derive/crossblock.py`
- Test: `tests/derive/test_crossblock.py`

**Interfaces:**
- Consumes: `reblock.contracts.{Block, Region, Proposal}`; `reblock.methods.peel.PeelReblocker`; `reblock.derive.access.STREET_TOL`; Task-1 block attrs.
- Produces:
  - `reconciled_baseline(region: Region, merged: Block, tol: float = STREET_TOL) -> Proposal` — run `PeelReblocker` per constituent block, union the roads onto the super-block, and snap co-located stub endpoints across interior boundaries (removes the naive-union strawman).
  - `spine_merge_reference(merged: Block, baseline: Proposal, tol: float = STREET_TOL, band: float = 20.0) -> Proposal` — a heuristic cross-block reference: replace pairs of near-parallel boundary-flanking road segments with one through-trunk crossing the boundary.

- [ ] **Step 1: Write failing tests** in `tests/derive/test_crossblock.py`

```python
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString, Polygon, box

from reblock.contracts import Block, Proposal, Region
from reblock.derive.crossblock import reconciled_baseline, spine_merge_reference

UTM = CRS.from_epsg(32734)


def _block(bid: str, poly: Polygon, xs: list[float]) -> Block:
    polys = [box(x, poly.bounds[1], x + 1, poly.bounds[3]) for x in xs]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[poly.boundary], crs=UTM)
    return Block(block_id=bid, crs=UTM, boundary=poly, parcels=parcels, streets=streets)


def test_reconciled_baseline_unions_per_block_roads() -> None:
    from reblock.derive.cluster import merge_cluster
    a = _block("a", box(0, 0, 4, 3), [0, 1, 2, 3])
    b = _block("b", box(4, 0, 8, 3), [4, 5, 6, 7])
    region = Region(region_id="t", crs=UTM, blocks=[a, b])
    merged = merge_cluster(region)
    prop = reconciled_baseline(region, merged)
    assert prop.roads is not None and not prop.roads.empty       # peel produced roads for both blocks
    assert prop.block_id == merged.block_id


def test_spine_merge_adds_a_crossing_trunk() -> None:
    from reblock.derive.network_metrics import n_cross_block_streets
    interior = MultiLineString([[(4, 0), (4, 3)]])
    merged = Block(block_id="a+b", crs=UTM, boundary=box(0, 0, 8, 3),
                   parcels=gpd.GeoDataFrame({"parcel_id": [0]}, geometry=[box(0, 0, 8, 3)], crs=UTM),
                   streets=gpd.GeoDataFrame(geometry=[box(0, 0, 8, 3).boundary], crs=UTM),
                   attrs={"interior_boundaries": interior})
    # two boundary-parallel spines flanking x=4, no crossing yet
    base = Proposal(block_id="a+b", crs=UTM, method="peel", roads=gpd.GeoDataFrame(
        geometry=[LineString([(3.0, 0.5), (3.0, 2.5)]), LineString([(5.0, 0.5), (5.0, 2.5)])], crs=UTM))
    assert n_cross_block_streets(base.roads, interior) == 0
    ref = spine_merge_reference(merged, base)
    assert n_cross_block_streets(ref.roads, interior) >= 1        # a through-trunk now crosses x=4
```

- [ ] **Step 2: Run to verify fail** → `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `src/reblock/derive/crossblock.py`

```python
"""Probe baselines: the boundary-reconciled block-local peel union (the fair myopia
baseline), and a heuristic automatic spine-merge cross-block reference (replace
boundary-flanking parallel spines with a single through-trunk) — no optimizer, no hand
drawing; just enough to isolate the cross-block-specific gain.
"""
from __future__ import annotations

import geopandas as gpd
from shapely import snap, union_all
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry

from reblock.contracts import Block, Proposal, Region
from reblock.derive.access import STREET_TOL
from reblock.derive.network_metrics import _crosses_boundary, _road_lines
from reblock.methods.peel import PeelReblocker


def reconciled_baseline(region: Region, merged: Block, tol: float = STREET_TOL) -> Proposal:
    blocks = sorted(region.blocks, key=lambda b: b.block_id)
    segments: list[BaseGeometry] = []
    for b in blocks:
        prop = PeelReblocker(tol=tol).propose(b)
        if prop.roads is not None and not prop.roads.empty:
            segments.extend(prop.roads.geometry)
    if not segments:
        return Proposal(block_id=merged.block_id, crs=merged.crs, method="peel_reconciled",
                        roads=gpd.GeoDataFrame(geometry=[], crs=merged.crs))
    # snap co-located endpoints together (reconcile stubs meeting across a boundary)
    reference = union_all(segments)
    reconciled = [snap(g, reference, tol) for g in segments]
    roads = gpd.GeoDataFrame(geometry=reconciled, crs=merged.crs)
    return Proposal(block_id=merged.block_id, crs=merged.crs, method="peel_reconciled",
                    proposal_id="peel_reconciled", roads=roads)


def _midline(a: LineString, b: LineString) -> LineString:
    """A trunk from a's start-ish to b's end-ish — a crude through-trunk replacing two
    boundary-parallel spines."""
    pa, pb = a.interpolate(0.5, normalized=True), b.interpolate(0.5, normalized=True)
    return LineString([Point(a.coords[0]), pa, pb, Point(b.coords[-1])])


def spine_merge_reference(
    merged: Block, baseline: Proposal, tol: float = STREET_TOL, band: float = 20.0) -> Proposal:
    interior = merged.attrs.get("interior_boundaries")
    if not isinstance(interior, MultiLineString) or interior.is_empty:
        return baseline
    lines = _road_lines(baseline.roads)
    corridor = interior.buffer(band)
    flanking = [ls for ls in lines
                if not _crosses_boundary(ls, interior, tol) and ls.intersects(corridor)]
    others = [ls for ls in lines if ls not in flanking]
    trunks: list[BaseGeometry] = list(others)
    used = [False] * len(flanking)
    for i in range(len(flanking)):
        if used[i]:
            continue
        for j in range(i + 1, len(flanking)):
            if used[j]:
                continue
            trunk = _midline(flanking[i], flanking[j])
            if _crosses_boundary(trunk, interior, tol):
                trunks.append(trunk)
                used[i] = used[j] = True
                break
        else:
            trunks.append(flanking[i])
            used[i] = True
    roads = gpd.GeoDataFrame(geometry=trunks, crs=merged.crs)
    return Proposal(block_id=merged.block_id, crs=merged.crs, method="spine_merge_ref",
                    proposal_id="spine_merge_ref", roads=roads)
```

- [ ] **Step 4: Run tests** → PASS. **Step 5: `pixi run check`** → green. **Commit:**

```bash
git add src/reblock/derive/crossblock.py tests/derive/test_crossblock.py
git commit -m "feat: reconciled block-local baseline + heuristic spine-merge reference"
```

---

### Task 7: The probe script — sample, score, correlation matrix, headroom report

**Files:**
- Create: `scripts/crossblock_probe.py`
- Test: `tests/test_crossblock_probe.py`

**Interfaces:**
- Consumes: `KblockSource`, `merge_cluster`, `reconciled_baseline`, `spine_merge_reference`, `StructureEval`, `KComplexityEval`.
- Produces: `enumerate_adjacent_pairs(blocks_gdf) -> list[tuple[str, str]]`; `probe_cluster(block_ids, blocks_path, buildings_path) -> dict[str, float]`; a `main()` that samples clusters, tabulates baseline vs reference metrics, prints the correlation matrix + headroom summary.

- [ ] **Step 1: Write a failing smoke test** in `tests/test_crossblock_probe.py`

```python
from pathlib import Path

import geopandas as gpd

from scripts.crossblock_probe import enumerate_adjacent_pairs, probe_cluster

ROOT = Path(__file__).resolve().parents[1]
CT_BLOCKS = str(ROOT / "tests" / "data" / "kblock" / "blocks_capetown_sample.parquet")
CT_BLD = str(ROOT / "tests" / "data" / "kblock" / "buildings_capetown_sample.parquet")


def test_enumerate_adjacent_pairs_finds_neighbours() -> None:
    blocks = gpd.read_parquet(CT_BLOCKS, columns=["block_id", "geometry"])
    pairs = enumerate_adjacent_pairs(blocks.to_crs(blocks.estimate_utm_crs()))
    assert len(pairs) > 0
    assert all(a < b for a, b in pairs)              # canonical ordering, no dup/self pairs


def test_probe_cluster_returns_baseline_and_reference_metrics() -> None:
    # a real adjacent Cape Town pair including the flagship's neighbour
    row = probe_cluster(["ZAF.9.3.1_1_44882", "ZAF.9.3.1_1_44673"], CT_BLOCKS, CT_BLD)
    assert row["base_n_cross_block_streets"] == 0.0    # block-local baseline never crosses
    assert row["base_circuity"] >= 1.0
    assert "ref_circuity" in row                        # the spine-merge reference was scored too
```

- [ ] **Step 2: Run to verify fail** → `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `scripts/crossblock_probe.py`

```python
"""Cross-block Phase-0 probe: over a stratified sample of adjacent Cape Town clusters,
score the boundary-reconciled block-local peel baseline and the heuristic spine-merge
reference on the metric basis, validate the basis's orthogonality with a correlation
matrix, and report the cross-block headroom (metric distributions vs their floors +
baseline->reference improvement) for a documented go/no-go. No pre-registered bar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import STRtree, make_valid

from reblock.contracts import Region
from reblock.data.kblock import KblockSource
from reblock.derive.cluster import merge_cluster
from reblock.derive.crossblock import reconciled_baseline, spine_merge_reference
from reblock.eval.kcomplexity import KComplexityEval
from reblock.eval.structure import StructureEval

ROOT = Path(__file__).resolve().parents[1]
CT_BLOCKS = str(ROOT / "tests" / "data" / "kblock" / "blocks_capetown_sample.parquet")
CT_BLD = str(ROOT / "tests" / "data" / "kblock" / "buildings_capetown_sample.parquet")
_BASIS = ["geometric_access_max_m", "geometric_access_p95_m", "circuity", "throughput_ratio",
          "meshedness", "four_way_fraction", "dead_end_fraction",
          "added_road_length_per_parcel", "n_cross_block_streets",
          "boundary_redundant_road_fraction"]


def enumerate_adjacent_pairs(blocks: gpd.GeoDataFrame) -> list[tuple[str, str]]:
    geoms = [make_valid(g) for g in blocks.geometry]
    ids = [str(b) for b in blocks["block_id"]]
    tree = STRtree(geoms)
    pairs: set[tuple[str, str]] = set()
    left, right = tree.query(geoms, predicate="intersects")
    for i, j in zip(left.tolist(), right.tolist(), strict=True):
        if i < j and geoms[i].intersection(geoms[j]).length > 0:
            pairs.add((ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i]))
    return sorted(pairs)


def _score(merged, proposal, prefix: str) -> dict[str, float]:
    sv = StructureEval().score(merged, proposal).values
    kv = KComplexityEval().score(merged, proposal).values
    row = {f"{prefix}_{k}": float(sv[k]) for k in _BASIS if k in sv}
    row[f"{prefix}_k_after"] = float(kv["k_after"])
    return row


def probe_cluster(block_ids: list[str], blocks_path: str, buildings_path: str) -> dict[str, float]:
    src = KblockSource(blocks_path, buildings_path, region_id="capetown", block_ids=block_ids)
    region: Region = src.region()
    merged = merge_cluster(region)
    base = reconciled_baseline(region, merged)
    ref = spine_merge_reference(merged, base)
    row = {"block_ids": "+".join(block_ids), "n_parcels": float(len(merged.parcels))}
    row.update(_score(merged, base, "base"))
    row.update(_score(merged, ref, "ref"))
    return row


def main(n_sample: int = 30) -> None:
    blocks = gpd.read_parquet(CT_BLOCKS, columns=["block_id", "k_complexity", "geometry"])
    blocks = blocks.to_crs(blocks.estimate_utm_crs())
    kmap = {str(b): float(k) for b, k in zip(blocks["block_id"], blocks["k_complexity"], strict=True)}
    pairs = enumerate_adjacent_pairs(blocks)
    # stratify by min(kblock_k) across the pair, sampled deterministically (sorted, strided)
    pairs.sort(key=lambda p: (min(kmap[p[0]], kmap[p[1]]), p))
    step = max(1, len(pairs) // n_sample)
    sample = pairs[::step][:n_sample]

    rows: list[dict[str, float]] = []
    for a, b in sample:
        try:
            rows.append(probe_cluster([a, b], CT_BLOCKS, CT_BLD))
        except (ValueError, KeyError) as exc:      # non-contiguous / sparse cluster -> skip, log
            print(f"skip {a}+{b}: {exc}", file=sys.stderr)
    df = pd.DataFrame(rows)

    print("\n=== orthogonality: correlation matrix (base_* basis) ===")
    base_cols = [f"base_{k}" for k in _BASIS if f"base_{k}" in df]
    print(df[base_cols].corr().round(2).to_string())

    print("\n=== headroom: baseline metric distributions (vs floors) ===")
    print(df[base_cols].describe(percentiles=[0.5]).round(3).to_string())

    print("\n=== baseline -> spine-merge reference improvement (median) ===")
    for k in ("circuity", "meshedness", "n_cross_block_streets", "added_road_length_per_parcel"):
        if f"base_{k}" in df and f"ref_{k}" in df:
            print(f"  {k}: {df[f'base_{k}'].median():.3f} -> {df[f'ref_{k}'].median():.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests** → `pixi run pytest tests/test_crossblock_probe.py -v` → PASS.

- [ ] **Step 5: Run the probe end-to-end** → `pixi run python scripts/crossblock_probe.py` → prints the correlation matrix, baseline distributions, and baseline→reference improvement without error. (This is the deliverable; eyeball that circuity/meshedness/n_cross_block_streets look sane.)

- [ ] **Step 6: `pixi run check`** → green. **Commit:**

```bash
git add scripts/crossblock_probe.py tests/test_crossblock_probe.py
git commit -m "feat: cross-block Phase-0 probe (sample, score, correlation, headroom)"
```

---

## Notes for the executor

- **`mypy --strict` on shapely/geopandas/networkx stubs** repeatedly needs small type-level adjustments (`isinstance` narrowing, `cast`, annotating a `nx.Graph`/`nx.DiGraph` local). Make the minimal adjustment that preserves the algorithm; document each in the task report.
- **`_crosses_boundary`/`_road_lines`/`_side` are shared** by `network_metrics.py` and `crossblock.py` — import them from `network_metrics` (Task 6 depends on Task 3). They start with `_` but are deliberately module-internal-shared; a reviewer may flag the underscore-import — it is intentional, note it.
- **Determinism:** no `Date.now`/RNG; the sample is a sorted-and-strided slice, so the probe output is reproducible.
- **The probe can return evidence against Phase 1** (baseline circuity near 1.0, negligible baseline→reference improvement) — that is a valid, expected possible outcome, not a failure.
- Out of scope (do not add): `curvature_variation`/smoothness, `dwellings_displaced`, the collinear through-going crossing refinement, automatic cluster selection, any Phase-1 placement method or `RegionMethod` path.
