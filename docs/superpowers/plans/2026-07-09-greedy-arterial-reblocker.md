# GreedyArterialReblocker Implementation Plan

> **RETIRED / SUPERSEDED (2026-07-11) — do NOT execute.** This plan is COMPLETE (the method
> shipped), and its "price of buildability / aspirational is the directness *ceiling*" framing is
> **retired**, not live guidance. In particular the Task 4 test
> `test_aspirational_ceiling_dominates_buildable_on_directness` and its "ceiling ≥ buildable / do not
> weaken the assertion" instruction assert a claim that is **false** under the shipped door-to-door
> metric: directness measures internal circulation, and a frontage-hugging buildable road matches or
> beats the ideal chord on a compact block. See the correction notes in
> `docs/superpowers/specs/2026-07-09-greedy-arterial-reblocker-design.md` and
> `docs/metrics-north-star.md`. Kept only as a historical record.

**Goal:** A `Method` that greedily inserts the single best straight arterial (highest objective gain per meter) one at a time until a road budget runs out, in a buildable mode (snapped to parcel frontages) and an aspirational mode (ideal chords), so the compare reports the "price of buildability."

**Architecture:** One new module `src/reblock/methods/arterial.py` holding pure geometry helpers, a greedy engine, and the `GreedyArterialReblocker` class. It reuses the boundary graph (`methods/dijkstra._boundary_graph`), access/connectivity (`derive/access`), and the benefit metrics (`budget`) — no new scoring code. Candidates are through-roads (network↔network) + spurs (network→deep pocket); continuations are through-roads from committed-segment endpoints (which are always anchors), so a spur completes into a through-road automatically, and crossings planarize into true intersections. Two modes share one code path via a `mode` parameter.

**Tech Stack:** Python, geopandas, shapely (`unary_union`, `STRtree`, `LineString`, `Point`), networkx, Hydra config, pixi, pytest, mypy --strict.

## Global Constraints

- Roads follow parcel frontages in **buildable** mode (snapped to the boundary graph); **aspirational** mode emits ideal straight chords (demolition implied). No back-compat shims — reuse existing functions, don't fork them.
- Deterministic: no RNG (fixed-spacing sampling, sorted candidates, argmax with a geometry tiebreak).
- `identity = ("greedy_arterial", mode, objective)`; `proposal_id = f"greedy_arterial_{mode}_{objective}"`; `method = "greedy_arterial"`.
- Default objective = `directness`. Objectives are `access` / `efficiency` / `directness`, mirroring the metrics in `reblock.budget`.
- Connectivity: every committed segment attaches to the current network (streets ∪ committed segments) at ≥1 point — no floating roads. Buildable roads must be street-connected (`street_connectivity(...).connected_frac == 1.0`).
- The emitted road set is **noded** (planarized) at crossings so an intersection is a shared graph node.
- Budget-sliceable by the *unchanged* `cost_benefit_curve`: emitted roads carry `drain` = descending greedy rank.
- Every command runs under pixi: `pixi run pytest ...`, `pixi run check` (ruff + mypy --strict + pytest).
- `STREET_TOL = 0.5` (import from `reblock.derive.access`).

---

## File Structure

- **Create `src/reblock/methods/arterial.py`** — geometry helpers (`_anchor_points`, `_deep_targets`, `_candidate_chords`, `_snap`, `_planarize`), the greedy engine (`_score`, `_greedy_arterials`), and `GreedyArterialReblocker`.
- **Create `conf/method/greedy_arterial.yaml`** — Hydra target + `mode`/`objective` defaults.
- **Modify `src/reblock/derive_graph.py`** — add `methods/arterial.py` to `_DERIVATION_MODULES`.
- **Modify `conf/compare_config.yaml`** — add two `all_methods` entries (buildable + aspirational) and append them to the `methods` run-list.
- **Create `tests/methods/test_arterial.py`** — unit + integration tests.

Reused (do not modify): `methods/dijkstra._boundary_graph`, `_rnd`; `derive/access.STREET_TOL`, `parcel_access_layers`, `street_connectivity`; `budget.access_burden`, `network_efficiency`, `access_benefit`, `efficiency_benefit`, `directness_benefit`, `cost_benefit_curve`, `efficiency_directness_curves`; `derive/adjacency.parcel_adjacency`.

---

## Task 1: Geometry helpers (anchors, targets, candidates, snap, planarize)

**Files:**
- Create: `src/reblock/methods/arterial.py`
- Test: `tests/methods/test_arterial.py`

**Interfaces:**
- Produces:
  - `_anchor_points(network: list[LineString], n: int) -> list[tuple[float, float]]` — `n` points sampled evenly by arc-length along the merged network, plus every network vertex, `_rnd`-snapped, de-duplicated, sorted.
  - `_deep_targets(block: Block, roads: GeoDataFrame | None, k: int, adj: list[set[int]]) -> list[tuple[float, float]]` — representative points of the `k` deepest-access parcels, `_rnd`-snapped.
  - `_candidate_chords(anchors, targets) -> list[LineString]` — through-roads (anchor pairs) + spurs (anchor→target), de-duplicated, sorted.
  - `_snap(chord: LineString, g: nx.Graph, node_tree: STRtree, nodes: list[tuple[float,float]], lam: float) -> LineString | None` — buildable realization: boundary-graph shortest path between the chord endpoints' nearest nodes, edge cost `length + lam·dist(edge_midpoint, chord_line)`; `None` if endpoints coincide or no path.
  - `_planarize(lines: list[LineString], crs) -> GeoDataFrame` — `unary_union` the lines (nodes crossings), explode to 2-point-or-more `LineString`s, one row each, `geometry` column, given CRS.

- [ ] **Step 1: Write failing tests** (`tests/methods/test_arterial.py`)

```python
from typing import cast

import geopandas as gpd
import networkx as nx
from pyproj import CRS
from shapely import STRtree
from shapely.geometry import LineString, Point, Polygon

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.arterial import (
    _anchor_points,
    _candidate_chords,
    _deep_targets,
    _planarize,
    _snap,
)
from reblock.methods.dijkstra import _boundary_graph

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


def test_anchor_points_sample_the_network_and_include_vertices() -> None:
    net = [LineString([(0.0, 0.0), (4.0, 0.0)])]
    pts = _anchor_points(net, n=4)
    assert (0.0, 0.0) in pts and (4.0, 0.0) in pts        # endpoints/vertices kept
    assert len(pts) >= 4 and pts == sorted(pts)           # sampled + deterministic order


def test_deep_targets_are_the_deepest_parcels() -> None:
    block = _grid_block(5)                                 # center parcel is deepest with full-boundary streets
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    targets = _deep_targets(block, None, k=1, adj=adj)
    assert len(targets) == 1
    assert Point(targets[0]).distance(Point(2.5, 2.5)) < 1.0   # near the 5x5 center


def test_candidate_chords_include_through_roads_and_spurs() -> None:
    anchors = [(0.0, 0.0), (4.0, 0.0)]
    targets = [(2.0, 2.0)]
    chords = _candidate_chords(anchors, targets)
    assert LineString([(0.0, 0.0), (4.0, 0.0)]) in chords            # through-road
    assert any(t == (2.0, 2.0) for c in chords for t in c.coords)    # a spur reaches the target
    assert chords == sorted(chords, key=lambda ls: ls.wkt)           # deterministic order


def test_snap_returns_a_boundary_following_street_connected_path() -> None:
    block = _grid_block(5)
    g = _boundary_graph(block.parcels)
    nodes = list(g.nodes)
    tree = STRtree([Point(nd) for nd in nodes])
    chord = LineString([(0.0, 2.5), (5.0, 2.5)])          # a horizontal cut across the grid
    path = _snap(chord, g, tree, nodes, lam=2.0)
    assert path is not None
    # every vertex of the snapped path is a boundary-graph node (buildable)
    assert all(_round(c) in set(g.nodes) for c in path.coords)


def _round(c: tuple[float, ...]) -> tuple[float, float]:
    return (round(c[0], 2), round(c[1], 2))


def test_planarize_nodes_two_crossing_chords() -> None:
    a = LineString([(0.0, 1.0), (2.0, 1.0)])
    b = LineString([(1.0, 0.0), (1.0, 2.0)])             # crosses a at (1, 1)
    gdf = _planarize([a, b], UTM)
    coords = {c for geom in gdf.geometry for c in geom.coords}
    assert (1.0, 1.0) in coords                           # crossing became a shared vertex
    assert len(gdf) == 4                                  # each chord split into two at the crossing
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run pytest tests/methods/test_arterial.py -x`
Expected: FAIL — `ModuleNotFoundError: No module named 'reblock.methods.arterial'`.

- [ ] **Step 3: Implement `src/reblock/methods/arterial.py` (helpers only)**

```python
"""GreedyArterialReblocker: greedily insert the single straight arterial with the best
objective gain per meter, one at a time, until a road budget runs out. Two modes -- buildable
(snapped to the parcel-boundary graph) and aspirational (ideal chords) -- so the compare reports
the price of buildability. Candidates are through-roads (network<->network) + spurs
(network->deep pocket); continuations are through-roads from committed-segment endpoints (always
anchors), so a spur completes into a through-road for free and crossings planarize into true
intersections. See docs/superpowers/specs/2026-07-09-greedy-arterial-reblocker-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import geopandas as gpd
import networkx as nx
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.dijkstra import _boundary_graph, _rnd


def _anchor_points(network: list[LineString], n: int) -> list[tuple[float, float]]:
    """`n` points sampled evenly by arc-length along the merged network, plus every network
    vertex (so committed-segment endpoints are always anchors -> continuations come for free).
    _rnd-snapped, de-duplicated, sorted for determinism."""
    merged = unary_union(network)
    lines = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    pts: set[tuple[float, float]] = set()
    for ln in lines:
        pts.update(_rnd(c) for c in ln.coords)                       # vertices
    total = sum(ln.length for ln in lines)
    if total > 0 and n > 0:
        step = total / n
        for ln in lines:
            d = 0.0
            while d <= ln.length:
                pts.add(_rnd(ln.interpolate(d).coords[0]))
                d += step
    return sorted(pts)


def _deep_targets(block: Block, roads: GeoDataFrame | None, k: int,
                  adj: list[set[int]]) -> list[tuple[float, float]]:
    """Representative points of the k deepest-access parcels (spur targets), _rnd-snapped."""
    depths = parcel_access_layers(block, roads, tol=STREET_TOL, adj=adj)
    order = depths.sort_values(ascending=False, kind="stable")
    id_to_pos = {pid: i for i, pid in enumerate(block.parcels["parcel_id"])}
    geoms = list(block.parcels.geometry)
    out: list[tuple[float, float]] = []
    for pid in list(order.index)[:k]:
        rep = geoms[id_to_pos[pid]].representative_point()
        out.append(_rnd(rep.coords[0]))
    return out


def _candidate_chords(anchors: list[tuple[float, float]],
                      targets: list[tuple[float, float]]) -> list[LineString]:
    """Through-roads (anchor pairs) + spurs (anchor -> deep target). De-duplicated, sorted."""
    seen: set[frozenset[tuple[float, float]]] = set()
    chords: list[LineString] = []
    for i, a in enumerate(anchors):
        for b in anchors[i + 1:]:
            key = frozenset((a, b))
            if a != b and key not in seen:
                seen.add(key)
                chords.append(LineString(sorted((a, b))))
        for t in targets:
            key = frozenset((a, t))
            if a != t and key not in seen:
                seen.add(key)
                chords.append(LineString(sorted((a, t))))
    return sorted(chords, key=lambda ls: ls.wkt)


def _snap(chord: LineString, g: nx.Graph, node_tree: STRtree,
          nodes: list[tuple[float, float]], lam: float) -> LineString | None:
    """Buildable realization: the boundary-graph path between the chord endpoints' nearest
    nodes that hugs the ideal line (edge cost = length + lam * dist(edge midpoint, chord)).
    None if the endpoints snap to the same node or no path exists."""
    p, q = _rnd(chord.coords[0]), _rnd(chord.coords[-1])
    np_ = nodes[int(node_tree.nearest(Point(p)))]
    nq_ = nodes[int(node_tree.nearest(Point(q)))]
    if np_ == nq_:
        return None

    def w(u: tuple[float, float], v: tuple[float, float], d: dict[str, float]) -> float:
        mid = Point((u[0] + v[0]) / 2, (u[1] + v[1]) / 2)
        return float(d["weight"]) + lam * mid.distance(chord)

    try:
        path = nx.shortest_path(g, np_, nq_, weight=w)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return LineString([tuple(node) for node in path])


def _planarize(lines: list[LineString], crs: CRS) -> GeoDataFrame:
    """unary_union the lines (nodes crossings), explode to LineStrings, one row each."""
    if not lines:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
    merged = unary_union(lines)
    parts: list[BaseGeometry] = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    rows = [ln for ln in parts if "LineString" in ln.geom_type and ln.length > 0]
    return gpd.GeoDataFrame({"geometry": rows}, geometry="geometry", crs=crs)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run pytest tests/methods/test_arterial.py -x`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/reblock/methods/arterial.py tests/methods/test_arterial.py
git commit -m "feat: arterial geometry helpers (anchors, targets, candidates, snap, planarize)"
```

---

## Task 2: Greedy engine (`_score`, `_greedy_arterials`)

**Files:**
- Modify: `src/reblock/methods/arterial.py`
- Test: `tests/methods/test_arterial.py`

**Interfaces:**
- Consumes: all Task 1 helpers; `_boundary_graph`, `_rnd` (dijkstra); `parcel_adjacency`; `parcel_access_layers`, `access_burden`, `network_efficiency` (see below).
- Produces:
  - `_score(objective: str, block: Block, roads: GeoDataFrame, adj: list[set[int]], base_burden: float) -> float` — higher = better: access → `1 - access_burden(...)/base_burden`; efficiency/directness → `network_efficiency(block, roads)[0 or 1]`.
  - `_greedy_arterials(block: Block, *, mode: str, objective: str, n_anchors: int = 32, top_k: int = 8, lam: float = 2.0, max_roads: int = 15) -> GeoDataFrame` — the greedy loop; returns roads with `geometry` + `drain` (descending greedy rank), planarized.

- [ ] **Step 1: Write failing tests** (append to `tests/methods/test_arterial.py`)

```python
from reblock.derive.access import street_connectivity
from reblock.methods.arterial import _greedy_arterials


def test_greedy_first_arterial_cuts_the_deep_block() -> None:
    # A long 3x9 block, full-boundary streets: the deepest parcels run down the spine, so the
    # best first directness arterial is a lengthwise cut. Assert one gets built and it is long.
    import geopandas as gpd
    from shapely.geometry import Polygon
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(3) for j in range(9)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    block = Block(block_id="long", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)
    roads = _greedy_arterials(block, mode="buildable", objective="directness", max_roads=3)
    assert len(roads) >= 1
    assert roads.geometry.length.max() >= 6.0                        # a real lengthwise arterial
    conn = street_connectivity(block.streets, roads, STREET_TOL)
    assert conn.connected_frac == 1.0                                # buildable + street-connected


def test_greedy_is_deterministic() -> None:
    block = _grid_block(5)
    r1 = _greedy_arterials(block, mode="buildable", objective="directness", max_roads=4)
    r2 = _greedy_arterials(block, mode="buildable", objective="directness", max_roads=4)
    assert [g.wkt for g in r1.geometry] == [g.wkt for g in r2.geometry]


def test_greedy_drain_is_descending_rank_and_slices_monotone() -> None:
    from reblock.budget import cost_benefit_curve
    block = _grid_block(6)
    roads = _greedy_arterials(block, mode="buildable", objective="directness", max_roads=5)
    assert list(roads["drain"]) == sorted(roads["drain"], reverse=True)   # rank descending
    curve = cost_benefit_curve(block, roads)                              # slices in greedy order
    assert curve.benefit == sorted(curve.benefit)                        # monotone non-decreasing


def test_aspirational_planarizes_crossings_into_true_intersections() -> None:
    block = _grid_block(6)
    roads = _greedy_arterials(block, mode="aspirational", objective="directness", max_roads=6)
    from reblock.budget import _road_street_graph
    g = _road_street_graph(block, roads, STREET_TOL)
    # at least one interior node has degree >= 4 -> a real crossroads, not overlapping lines
    interior = [nd for nd in g.nodes if Point(nd).distance(block.boundary.boundary) > STREET_TOL]
    assert any(g.degree[nd] >= 4 for nd in interior)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run pytest tests/methods/test_arterial.py -x -k greedy or aspirational`
Expected: FAIL — `ImportError: cannot import name '_greedy_arterials'`.

- [ ] **Step 3: Implement the engine (append to `src/reblock/methods/arterial.py`)**

```python
from reblock.budget import access_burden, network_efficiency


def _score(objective: str, block: Block, roads: GeoDataFrame, adj: list[set[int]],
           base_burden: float) -> float:
    """Objective value of the road set (higher = better). Mirrors the budget metrics with a
    cached parcel-adjacency for the greedy's inner loop."""
    if objective == "access":
        if base_burden == 0.0:
            return 0.0
        depths = parcel_access_layers(block, roads, tol=STREET_TOL, adj=adj,
                                      unreached_depth=len(block.parcels) + 1)
        return 1.0 - access_burden(depths) / base_burden
    e, direct = network_efficiency(block, roads)
    return e if objective == "efficiency" else direct


def _greedy_arterials(block: Block, *, mode: str, objective: str, n_anchors: int = 32,
                      top_k: int = 8, lam: float = 2.0, max_roads: int = 15) -> GeoDataFrame:
    """Greedily commit the straight arterial with the best objective gain per meter until
    `max_roads` are placed or no candidate improves. `mode` in {"buildable", "aspirational"}."""
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    base_burden = access_burden(parcel_access_layers(
        block, None, tol=STREET_TOL, adj=adj, unreached_depth=len(block.parcels) + 1))
    g = _boundary_graph(block.parcels)
    nodes = list(g.nodes)
    node_tree = STRtree([Point(nd) for nd in nodes])

    committed: list[LineString] = []                        # realized geometry, in commit order
    while len(committed) < max_roads:
        network = list(block.streets.geometry) + committed
        anchors = _anchor_points([ln for ln in network if isinstance(ln, LineString)]
                                 or list(block.streets.geometry), n_anchors)
        base = _planarize(committed, block.crs)
        base_val = _score(objective, block, base, adj, base_burden)
        curr_roads = base if len(committed) else None
        targets = _deep_targets(block, curr_roads, top_k, adj)

        best_gain, best_real = 0.0, None
        for chord in _candidate_chords(anchors, targets):
            real = chord if mode == "aspirational" else _snap(chord, g, node_tree, nodes, lam)
            if real is None or real.length == 0:
                continue
            trial = _planarize(committed + [real], block.crs)
            gain = (_score(objective, block, trial, adj, base_burden) - base_val) / real.length
            if gain > best_gain or (best_real is not None and gain == best_gain
                                    and real.wkt < best_real.wkt):
                best_gain, best_real = gain, real
        if best_real is None:                               # no candidate improves -> stop
            break
        committed.append(best_real)

    return _planarize_ranked(committed, block.crs)


def _planarize_ranked(committed: list[LineString], crs: CRS) -> GeoDataFrame:
    """Planarize the committed segments and tag each noded piece with `drain` = descending
    greedy rank of the source segment (first-committed = highest), so cost_benefit_curve slices
    in greedy order. A piece belongs to the earliest-committed segment that covers it."""
    gdf = _planarize(committed, crs)
    if gdf.empty:
        gdf["drain"] = []
        return gdf
    n = len(committed)
    drains: list[int] = []
    for geom in gdf.geometry:
        mid = geom.interpolate(0.5, normalized=True)
        src = next((i for i, seg in enumerate(committed)
                    if mid.distance(seg) <= STREET_TOL), n - 1)
        drains.append(n - src)                              # earliest source -> highest drain
    gdf["drain"] = drains
    return gdf.sort_values("drain", ascending=False, kind="stable").reset_index(drop=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run pytest tests/methods/test_arterial.py -x`
Expected: PASS (all Task 1 + Task 2 tests). If `test_greedy_first_arterial_cuts_the_deep_block` is flaky on the length threshold, inspect `roads.geometry.length` — the arterial should span the block; do not weaken the assertion without understanding why (a too-short "arterial" means the snap or candidate set is wrong).

- [ ] **Step 5: Commit**

```bash
git add src/reblock/methods/arterial.py tests/methods/test_arterial.py
git commit -m "feat: greedy arterial engine (score + loop + ranked planarized emit)"
```

---

## Task 3: `GreedyArterialReblocker` + config + derivation + compare wiring

**Files:**
- Modify: `src/reblock/methods/arterial.py`
- Create: `conf/method/greedy_arterial.yaml`
- Modify: `src/reblock/derive_graph.py:34-45` (the `_DERIVATION_MODULES` tuple)
- Modify: `conf/compare_config.yaml`
- Test: `tests/methods/test_arterial.py`

**Interfaces:**
- Consumes: `_greedy_arterials`.
- Produces: `GreedyArterialReblocker(mode="buildable", objective="directness", n_anchors=32, top_k=8, lam=2.0, max_roads=15)` with `identity`, `propose(block, prior=None) -> Proposal`.

- [ ] **Step 1: Write failing tests** (append to `tests/methods/test_arterial.py`)

```python
from reblock.methods.arterial import GreedyArterialReblocker


def test_identity_and_proposal_metadata() -> None:
    m = GreedyArterialReblocker(mode="buildable", objective="directness")
    assert m.identity == ("greedy_arterial", "buildable", "directness")
    proposal = m.propose(_grid_block(5))
    assert proposal.method == "greedy_arterial"
    assert proposal.proposal_id == "greedy_arterial_buildable_directness"
    assert proposal.roads is not None and len(proposal.roads) > 0
    assert proposal.block_identity == _grid_block(5).identity


def test_both_modes_produce_roads() -> None:
    block = _grid_block(6)
    for mode in ("buildable", "aspirational"):
        p = GreedyArterialReblocker(mode=mode, objective="directness", max_roads=4).propose(block)
        assert p.roads is not None and len(p.roads) > 0


def test_config_and_derivation_wiring() -> None:
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from reblock.derive_graph import _DERIVATION_MODULES
    assert any(p.name == "arterial.py" for p in _DERIVATION_MODULES)
    conf_dir = str(Path("conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="compare_config",
                      overrides=["shapefile=x", "methods=[greedy_arterial_buildable]"])
    m = instantiate(cfg.all_methods["greedy_arterial_buildable"])
    assert m.identity == ("greedy_arterial", "buildable", "directness")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run pytest tests/methods/test_arterial.py -x -k identity or modes or wiring`
Expected: FAIL — `ImportError: cannot import name 'GreedyArterialReblocker'`.

- [ ] **Step 3a: Implement the class (append to `src/reblock/methods/arterial.py`)**

```python
@dataclass
class GreedyArterialReblocker:
    mode: str = "buildable"          # "buildable" | "aspirational"
    objective: str = "directness"    # "access" | "efficiency" | "directness"
    n_anchors: int = 32
    top_k: int = 8
    lam: float = 2.0
    max_roads: int = 15

    @property
    def identity(self) -> tuple[str, str, str]:
        return ("greedy_arterial", self.mode, self.objective)

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior
        roads = _greedy_arterials(block, mode=self.mode, objective=self.objective,
                                  n_anchors=self.n_anchors, top_k=self.top_k, lam=self.lam,
                                  max_roads=self.max_roads)
        return Proposal(
            block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
            proposal_id=f"greedy_arterial_{self.mode}_{self.objective}", method="greedy_arterial",
            params={"segments": len(roads), "mode": self.mode, "objective": self.objective},
            block_identity=block.identity)
```

Remove the now-unused `field` import from the top of the file if ruff flags it.

- [ ] **Step 3b: Create `conf/method/greedy_arterial.yaml`**

```yaml
_target_: reblock.methods.arterial.GreedyArterialReblocker
mode: buildable
objective: directness
```

- [ ] **Step 3c: Add the derivation-module entry** — in `src/reblock/derive_graph.py`, inside `_DERIVATION_MODULES`, immediately after the `.../ "methods" / "mesh.py"` line, add:

```python
    Path(__file__).parent / "methods" / "arterial.py",
```

- [ ] **Step 3d: Wire the compare** — in `conf/compare_config.yaml`, add to the `all_methods:` mapping (after the `mesh:` entry):

```yaml
  greedy_arterial_buildable: {_target_: reblock.methods.arterial.GreedyArterialReblocker, mode: buildable, objective: directness}
  greedy_arterial_aspirational: {_target_: reblock.methods.arterial.GreedyArterialReblocker, mode: aspirational, objective: directness}
```

and append both to the `methods:` run-list so it reads:

```yaml
methods: [dijkstra, peel, topology, mesh, greedy_arterial_buildable, greedy_arterial_aspirational]
```

- [ ] **Step 4: Run the tests + full check**

Run: `pixi run pytest tests/methods/test_arterial.py -x` then `pixi run check`
Expected: all arterial tests PASS; `pixi run check` green (ruff + mypy --strict + full pytest).

- [ ] **Step 5: Commit**

```bash
git add src/reblock/methods/arterial.py conf/method/greedy_arterial.yaml src/reblock/derive_graph.py conf/compare_config.yaml tests/methods/test_arterial.py
git commit -m "feat: GreedyArterialReblocker class + config + derivation + compare wiring"
```

---

## Task 4: Real-block efficacy + price-of-buildability

**Files:**
- Test: `tests/methods/test_arterial.py`

**Interfaces:**
- Consumes: `GreedyArterialReblocker`; `reblock.data.kblock.KblockSource`; `reblock.budget.efficiency_directness_curves`, `auc`.

- [ ] **Step 1: Write the failing test** (append to `tests/methods/test_arterial.py`)

```python
def _dji_block(block_id: str) -> Block:
    from reblock.data.kblock import KblockSource
    src = KblockSource(blocks_path="tests/data/kblock/blocks_dji_sample.parquet",
                       buildings_path="tests/data/kblock/buildings_dji_sample.parquet",
                       region_id="dji", block_ids=[block_id])
    return next(iter(src.region().blocks))


def test_aspirational_ceiling_dominates_buildable_on_directness() -> None:
    # Price of buildability >= 0: the ideal-chord greedy must reach at least as high a directness
    # AUC as the snapped one on a real block (the ceiling can't be below the buildable floor).
    from reblock.budget import auc, efficiency_directness_curves
    block = _dji_block("DJI.3_1_2781")
    build = GreedyArterialReblocker(mode="buildable", objective="directness", max_roads=6)
    asp = GreedyArterialReblocker(mode="aspirational", objective="directness", max_roads=6)
    _, dc_build = efficiency_directness_curves(block, build.propose(block).roads)
    _, dc_asp = efficiency_directness_curves(block, asp.propose(block).roads)
    cap = max(dc_build.cost[-1], dc_asp.cost[-1])
    assert auc(dc_asp, cap) >= auc(dc_build, cap) - 1e-9
```

- [ ] **Step 2: Run the test to verify it fails, then passes**

Run: `pixi run pytest tests/methods/test_arterial.py::test_aspirational_ceiling_dominates_buildable_on_directness -x`
Expected: PASS once the method works end-to-end on a real block. If it fails, the ceiling is below the floor — investigate the aspirational scoring (the ideal chords must be planarized before scoring, or a buildable road is scoring higher than its own ideal chord, which signals a snap/scoring inconsistency). Do not weaken the assertion.

- [ ] **Step 3: Manual efficacy read (no code — record numbers in the task report)**

Run:
```bash
pixi run python -m reblock.compare data=dji eval=kcomplexity max_blocks=3 \
  methods=[dijkstra,mesh,greedy_arterial_buildable,greedy_arterial_aspirational] \
  hydra.run.dir=/tmp/arterial_probe
```
Read `/tmp/arterial_probe/auc_table_directness.csv`, `auc_table_efficiency.csv`, `auc_table_access.csv`. Record in the report: greedy_arterial (buildable & aspirational) vs dijkstra/mesh mean AUC on all three lenses, and the buildable-vs-aspirational gap (the price of buildability). Expected: the arterial method leads on directness/efficiency; the aspirational ceiling sits at or above the buildable curve.

- [ ] **Step 4: Run the full check**

Run: `pixi run check`
Expected: green (ruff + mypy --strict + full pytest).

- [ ] **Step 5: Commit**

```bash
git add tests/methods/test_arterial.py
git commit -m "test: arterial real-block efficacy + price-of-buildability"
```

---

## Self-Review

**1. Spec coverage:**
- Greedy loop, best-per-meter, budget stop → Task 2 (`_greedy_arterials`). ✓
- Two modes (buildable snap + aspirational chords), price of buildability → Task 2 (mode param) + Task 4 (ceiling ≥ floor). ✓
- Candidates = through-roads + spurs; continuations via committed-endpoint anchors → Task 1 (`_anchor_points` keeps vertices) + `_candidate_chords`. ✓
- True intersections via planarization → Task 1 `_planarize` + Task 2 aspirational-crossing test. ✓
- Transitive-network anchoring, no floating roads, street-connected → Task 2 (`network` includes committed) + connectivity assertion. ✓
- Pluggable objective, default directness → Task 2 `_score` + Task 3 defaults. ✓
- `drain` = greedy rank; unchanged `cost_benefit_curve` → Task 2 `_planarize_ranked` + monotone-slice test. ✓
- identity/proposal_id/config/derivation/compare wiring → Task 3. ✓
- Deterministic (no RNG) → Task 2 determinism test. ✓
- Straight-first (curvature deferred), single-block v1 → whole plan (no arcs, `Block` contract). ✓

**2. Placeholder scan:** every code step carries complete code; commands have expected output; no TBD/TODO. ✓

**3. Type consistency:** `_greedy_arterials(...) -> GeoDataFrame(geometry, drain)` consumed by `propose`; `_score(objective, block, roads, adj, base_burden) -> float` used in the loop; `_snap(...) -> LineString | None` handled with a `None` guard; `identity` is `tuple[str,str,str]`; `mode`/`objective` strings consistent across class, config, and compare entries. ✓

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-09-greedy-arterial-reblocker.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
