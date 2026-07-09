# MeshReblocker + multi-metric grading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `MeshReblocker` that adds crossing roads (loops) on top of the dijkstra forest, graded by a cost-benefit framework generalized to **pluggable benefit metrics** — access (existing), E (network efficiency), directness (1/circuity).

**Architecture:** `reblock.budget` gains a `benefit_fn` param on `cost_benefit_curve` + three benefit factories (`access_benefit`, `efficiency_benefit`, `directness_benefit`, the last two from one sampled shortest-path pass); `reblock.methods.mesh.MeshReblocker` builds the dijkstra forest + greedy shortcut-ratio loops; `reblock.compare` grades every method on all three metrics. Spec: `docs/superpowers/specs/2026-07-09-mesh-reblocker-design.md`.

**Tech Stack:** Python 3.12, networkx, shapely/geopandas, Hydra, pixi, pytest, `mypy --strict`, ruff.

## Global Constraints

- `pixi run check` stays green — currently **160 tests**.
- **`cost_benefit_curve` change is backward-compatible** — `benefit_fn` defaults to `access_benefit`, so existing callers/tests (which pass no `benefit_fn`) behave exactly as today (access Σdepth² curve).
- **E/directness are sampled** (K≈40 seeded source parcels) — O(K·N), deterministic per block (fixed sampling, no RNG). The curve stays affordable at 20 budget points.
- **MeshReblocker is deterministic**, builds ON `_reblock_dijkstra` (does not reimplement the forest), and `methods/mesh.py` goes in `derive_graph._DERIVATION_MODULES`.
- Commit trailers on every commit (Co-Authored-By / Claude-Session as in prior commits).

---

### Task 1: pluggable `benefit_fn` + E/directness metrics (`reblock.budget`)

**Files:** Modify `src/reblock/budget.py`; Test `tests/test_budget.py`.

**Interfaces:**
- `access_benefit(block, *, tol=STREET_TOL) -> Callable[[GeoDataFrame|None], float]` — prepared closure; `f(roads)` = `1 - Σdepth²(roads)/Σdepth²(∅)` (the current metric, refactored).
- `network_efficiency(block, roads, *, k=40, tol=STREET_TOL) -> tuple[float, float]` — sampled `(E, directness)`.
- `efficiency_benefit` / `directness_benefit` — benefit factories returning `f(roads)` = the E / directness scalar.
- `cost_benefit_curve(block, roads, *, benefit_fn=access_benefit, n_points=20, tol=STREET_TOL)`.

- [ ] **Step 1: Write failing tests** — add to `tests/test_budget.py`:

```python
def test_efficiency_and_directness_rise_with_roads() -> None:
    from reblock.budget import network_efficiency
    block = _grid_block(5)
    roads = DijkstraReblocker().propose(block).roads
    e_none, d_none = network_efficiency(block, roads.iloc[:0])   # no roads
    e_full, d_full = network_efficiency(block, roads)
    assert e_full > e_none and d_full > d_none


def test_cost_benefit_curve_accepts_a_benefit_fn() -> None:
    from reblock.budget import cost_benefit_curve, efficiency_benefit
    block = _grid_block(5)
    roads = DijkstraReblocker().propose(block).roads
    curve = cost_benefit_curve(block, roads, benefit_fn=efficiency_benefit, n_points=8)
    assert curve.benefit[-1] >= curve.benefit[0]      # efficiency non-decreasing with roads
    assert len(curve.cost) == len(curve.benefit)
```

- [ ] **Step 2: Verify failure** — `pixi run pytest tests/test_budget.py -k "efficiency or benefit_fn" -v` → FAIL (`network_efficiency`/`efficiency_benefit` undefined; `cost_benefit_curve` has no `benefit_fn`).

- [ ] **Step 3: Implement.** Add a shared road-graph builder (extract from `road_drainage`), the sampled metric, the factories, and the `benefit_fn` refactor:

```python
from collections.abc import Callable

BenefitFactory = Callable[..., Callable[["GeoDataFrame | None"], float]]


def _road_street_graph(block: Block, roads: GeoDataFrame | None,
                       tol: float) -> tuple[nx.Graph, dict[frozenset, int]]:
    """Graph over the road segments PLUS block.streets (so inter-parcel trips can use the
    street), nodes = snapped endpoints. (Shared with road_drainage's graph build.)"""
    g: nx.Graph = nx.Graph()
    lines = [] if roads is None else list(roads.geometry)
    lines += list(block.streets.geometry)
    for geom in lines:
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for part in parts:
            cs = list(part.coords)
            for a, b in zip(cs, cs[1:], strict=False):
                na, nb = _rnd(a), _rnd(b)
                if na != nb:
                    g.add_edge(na, nb, weight=Point(na).distance(Point(nb)))
    return g, {}


def network_efficiency(block: Block, roads: GeoDataFrame | None, *, k: int = 40,
                       tol: float = STREET_TOL) -> tuple[float, float]:
    """Sampled (E, directness): from K seeded source parcels to all parcels, over the
    road+street graph. E = mean(1/d), directness = mean(euclid/d) over reachable pairs;
    (0,0) if the graph is empty. Deterministic: sources are evenly spaced by sorted id."""
    g, _ = _road_street_graph(block, roads, tol)
    if g.number_of_nodes() == 0:
        return 0.0, 0.0
    nodes = list(g.nodes)
    tree = STRtree([Point(n) for n in nodes])
    reps = [gm.representative_point() for gm in block.parcels.geometry]
    # each parcel -> nearest graph node (its access point); parcels with none are unreachable
    entry: list[tuple[float, float] | None] = []
    for p in reps:
        near = tree.query(p, predicate="dwithin", distance=tol * 4)
        entry.append(min((nodes[j] for j in near), key=lambda n: p.distance(Point(n)),
                         default=None))
    idx = [i for i, e in enumerate(entry) if e is not None]
    if len(idx) < 2:
        return 0.0, 0.0
    step = max(1, len(idx) // k)
    sources = idx[::step][:k]
    inv_sum = dir_sum = pairs = 0.0
    for si in sources:
        dist = nx.single_source_dijkstra_path_length(g, entry[si])
        for j in idx:
            if j == si:
                continue
            d = dist.get(entry[j])
            pairs += 1
            if d and d > 0:
                inv_sum += 1.0 / d
                dir_sum += reps[si].distance(reps[j]) / d
    if pairs == 0:
        return 0.0, 0.0
    return inv_sum / pairs, dir_sum / pairs


def access_benefit(block: Block, *, tol: float = STREET_TOL) -> Callable[..., float]:
    adj = parcel_adjacency(list(block.parcels.geometry), tol)
    cap = len(block.parcels) + 1
    base = access_burden(parcel_access_layers(block, None, tol=tol, adj=adj, unreached_depth=cap))

    def f(roads: GeoDataFrame | None) -> float:
        if base == 0.0:
            return 0.0
        return 1.0 - access_burden(
            parcel_access_layers(block, roads, tol=tol, adj=adj, unreached_depth=cap)) / base
    return f


def efficiency_benefit(block: Block, *, tol: float = STREET_TOL) -> Callable[..., float]:
    return lambda roads: network_efficiency(block, roads, tol=tol)[0]


def directness_benefit(block: Block, *, tol: float = STREET_TOL) -> Callable[..., float]:
    return lambda roads: network_efficiency(block, roads, tol=tol)[1]
```

Refactor `cost_benefit_curve` to use `benefit_fn` (replaces the inline `_burden`/`base`):

```python
def cost_benefit_curve(block: Block, roads: GeoDataFrame, *, benefit_fn: BenefitFactory = access_benefit,
                       n_points: int = 20, tol: float = STREET_TOL) -> Curve:
    value = benefit_fn(block, tol=tol)
    cost, benefit = [0.0], [value(roads.iloc[:0])]
    if len(roads) == 0 or block.boundary.area == 0.0:
        return Curve(cost, benefit)
    drain = road_drainage(block, roads, tol=tol)
    order = sorted(range(len(roads)), key=lambda i: (-drain[i], i))
    ordered = roads.iloc[order].reset_index(drop=True)
    cum = ordered.geometry.length.to_numpy().cumsum()
    total, area_ha = float(cum[-1]), block.boundary.area / 1e4
    seen = 0
    for kk in range(1, n_points + 1):
        m = int((cum <= (kk / n_points) * total + 1e-9).sum())
        if m <= seen:
            continue
        seen = m
        cost.append(float(cum[m - 1]) / area_ha)
        benefit.append(value(ordered.iloc[:m]))
    return Curve(cost, benefit)
```

(`value(roads.iloc[:0])` is the no-roads baseline — an empty roads frame, not `None`, so `access_benefit` returns 0.0 and `efficiency_benefit` returns E-with-street-only.)

- [ ] **Step 4: Run tests + full check** — `pixi run check`. The existing access tests still pass (default `benefit_fn=access_benefit`); the two new tests pass; E/directness rise with roads. ~162 tests.

- [ ] **Step 5: Commit** (`feat: reblock.budget -- pluggable benefit_fn + E/directness metrics`).

---

### Task 2: `MeshReblocker` (dijkstra forest + shortcut-ratio loops)

**Files:** Create `src/reblock/methods/mesh.py`; Modify `src/reblock/derive_graph.py`; Create `conf/method/mesh.yaml`; Test `tests/methods/test_mesh.py`.

**Interfaces:** `MeshReblocker()`, `identity=("mesh",)`, `propose(block, prior=None) -> Proposal` with `roads` (forest + loops, `drain` column), `proposal_id/method="mesh"`.

- [ ] **Step 1: Write failing tests** (`tests/methods/test_mesh.py`) — mirror `test_dijkstra.py`'s `_grid_block`; assert: mesh roads ⊇ dijkstra roads and has MORE (≥1 loop); deterministic (WKT-equal); every road street-connected (`street_connectivity(...).connected_frac == 1.0`); a real efficacy check — mesh directness ≥ dijkstra directness on the grid:

```python
def test_mesh_adds_loops_over_the_dijkstra_forest() -> None:
    block = _grid_block(5)
    tree = DijkstraReblocker().propose(block).roads
    mesh = MeshReblocker().propose(block).roads
    assert len(mesh) > len(tree)                       # closed >=1 loop
    conn = street_connectivity(block.streets, mesh, STREET_TOL)
    assert conn.connected_frac == 1.0


def test_mesh_more_direct_than_the_tree() -> None:
    from reblock.budget import network_efficiency
    block = _grid_block(5)
    _, d_tree = network_efficiency(block, DijkstraReblocker().propose(block).roads)
    _, d_mesh = network_efficiency(block, MeshReblocker().propose(block).roads)
    assert d_mesh >= d_tree                            # crossings straighten inter-parcel travel
```

- [ ] **Step 2: Verify failure** — `No module named 'reblock.methods.mesh'`.

- [ ] **Step 3: Implement `src/reblock/methods/mesh.py`.** Reuse the dijkstra forest, then close loops:

```python
"""MeshReblocker: the dijkstra forest plus crossing roads. Closes boundary-graph loops in
descending shortcut-ratio order (forest-path-distance / edge-length -- a one-BFS proxy for
circuity reduction), so the network gains through-roads and redundancy the tree lacks."""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString

from reblock.contracts import Block, Proposal
from reblock.methods.dijkstra import _boundary_graph, _reblock_dijkstra, _rnd


def _mesh_roads(block: Block) -> gpd.GeoDataFrame:
    forest = _reblock_dijkstra(block)
    g = _boundary_graph(block.parcels)
    forest_edges = {frozenset((_rnd(a), _rnd(b)))
                    for line in forest.geometry for a, b in zip(list(line.coords),
                                                                list(line.coords)[1:])}
    fg = nx.Graph()
    for e in forest_edges:
        u, v = tuple(e)
        fg.add_edge(u, v, weight=Point_dist(u, v))
    # candidate loops: graph edges NOT in the forest with both endpoints already on the forest
    cands = []
    for u, v, w in g.edges(data="weight"):
        if frozenset((u, v)) in forest_edges or u not in fg or v not in fg:
            continue
        try:
            fp = nx.shortest_path_length(fg, u, v, weight="weight")
        except nx.NetworkXNoPath:
            continue
        cands.append((fp / w if w else 0.0, w, frozenset((u, v))))   # (shortcut ratio, len, edge)
    cands.sort(key=lambda c: (-c[0], sorted(c[2])))
    loops = [LineString(sorted(e)) for ratio, _w, e in cands if ratio > 1.0]   # a real detour

    rows = [{"geometry": geom, "drain": int(d)} for geom, d in
            zip(forest.geometry, forest["drain"])]
    rows += [{"geometry": geom, "drain": 0} for geom in loops]   # loops: drain 0 (not tree carriers)
    return gpd.GeoDataFrame(rows, columns=["geometry", "drain"], geometry="geometry", crs=block.crs)


@dataclass
class MeshReblocker:
    @property
    def identity(self) -> tuple[str]:
        return ("mesh",)

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior
        roads = _mesh_roads(block)
        return Proposal(block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
                        proposal_id="mesh", method="mesh",
                        params={"segments": len(roads)}, block_identity=block.identity)
```

(Replace `Point_dist(u, v)` with `shapely.geometry.Point(u).distance(Point(v))` — import `Point`. The forest already guarantees street-connectivity; loops attach to two forest nodes so they never float.)

- [ ] **Step 4: Config + derivation-module hash** — `conf/method/mesh.yaml` (`_target_: reblock.methods.mesh.MeshReblocker`); add `methods/mesh.py` to `derive_graph._DERIVATION_MODULES` next to `dijkstra.py`; add `mesh` to `conf/compare_config.yaml`'s `all_methods`.

- [ ] **Step 5: Run tests + full check** — `pixi run check`. Mesh has more roads than the tree, all street-connected, deterministic; directness ≥ tree. ~166 tests.

- [ ] **Step 6: Commit** (`feat: MeshReblocker -- dijkstra forest + shortcut-ratio crossing roads`).

---

### Task 3: multi-metric grading in `reblock.compare`

**Files:** Modify `src/reblock/compare.py`, `src/reblock/emit.py`; `conf/compare_config.yaml`; Test `tests/test_compare.py`.

**Interfaces:** `MethodCurve` gains a `metric: str` field; `compare()` sweeps `(block × method × metric)`; `compare_report` writes a table + curves **per metric**.

- [ ] **Step 1: Write failing test** — extend the compose e2e to assert an `auc_table` and curves exist per metric (`access`, `efficiency`, `directness`), e.g. `auc_table_access.csv` + `curve_access_<block>.png`:

```python
def test_compare_emits_per_metric_tables(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "reblock.compare", "data=dji", "eval=kcomplexity",
         "max_blocks=1", "methods=[dijkstra,mesh]", f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    for metric in ("access", "efficiency", "directness"):
        assert (tmp_path / f"auc_table_{metric}.csv").exists()
        assert list(tmp_path.glob(f"curve_{metric}_*.png"))
```

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement.** In `compare.py`: a `METRICS` dict `{"access": access_benefit, "efficiency": efficiency_benefit, "directness": directness_benefit}`; loop `(block, method, metric)`, computing `cost_benefit_curve(block, roads, benefit_fn=fn)`; `MethodCurve` gains `metric`. In `emit.py`: `compare_report` groups by metric, writing `auc_table_{metric}.csv` + `curve_{metric}_{block}.png` (each metric's per-block overlay). Keep the AUC per-block common-cap logic per metric.

- [ ] **Step 4: Run tests + full check** — `pixi run check`; the e2e writes 3 metrics' tables + curves. ~167 tests.

- [ ] **Step 5: Commit** (`feat: compare grades every method on access + efficiency + directness`).

---

## Self-Review

**Spec coverage:** pluggable `benefit_fn` + `access_benefit`/`efficiency_benefit`/`directness_benefit` with sampled E/directness (Task 1); `MeshReblocker` = forest + shortcut-ratio loops, deterministic, street-connected, config + derive-module (Task 2); 3-lens compare + per-metric emitters (Task 3). Mesh ≈ dijkstra on access, higher on directness/E — guarded by tests.

**Placeholder scan:** complete code except the two noted substitutions (`Point(u).distance(Point(v))`, imports) — flagged inline, not TBD.

**Type consistency:** `benefit_fn: BenefitFactory` (block → `roads -> float`) defaults to `access_benefit`, used by `cost_benefit_curve`; `network_efficiency -> (E, directness)` feeds both factories; `MethodCurve.metric` added and read by `compare_report`; `MeshReblocker.propose -> Proposal` matches the `Method` protocol; `roads` carries `drain` (forest) + 0 (loops) so `road_drainage`/curve ordering still work.

**Backward-compat:** `cost_benefit_curve`'s `benefit_fn` defaults to access, so every existing caller/test is unaffected.
