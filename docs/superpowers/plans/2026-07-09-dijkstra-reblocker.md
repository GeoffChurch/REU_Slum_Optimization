# DijkstraReblocker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new `Method`, `DijkstraReblocker`, that reblocks a block by routing roads **along parcel boundaries** as a shortest-path forest rooted at the street (drainage-weighted, coverage-complete) — buildable, minimal, hierarchical, at peel's ~1 s cost.

**Architecture:** A pure `_reblock_dijkstra(block) -> GeoDataFrame` builds the parcel-boundary graph (fresh from shapely), runs one multi-source Dijkstra from the street, unions each interior parcel's shortest route (accumulating per-edge drainage), attaches a frontage spur for parcels the forest reaches only at a vertex, and returns road segments with a `drain` column ordered arterials-first. `DijkstraReblocker` is a thin, deterministic `Method` wrapper. Spec: `docs/superpowers/specs/2026-07-09-dijkstra-reblocker-design.md`.

**Tech Stack:** Python 3.12, networkx, shapely/geopandas, Hydra, pixi, pytest, `mypy --strict`, ruff.

## Global Constraints

- `pixi run check` stays green — `ruff check` + `mypy --strict src tests scripts/crossblock_probe.py` + pytest. Suite is currently **141 tests**.
- **Deterministic, no RNG** — same block → byte-identical roads; `run()`'s purity contract holds (no global `np.random`/`random` mutation). Dijkstra/`min`/sort tie-breaks are on node/edge tuples; the boundary graph adds edges in sorted order.
- **Roads must be street-connected** — the forest is rooted at the street; spurs attach to it (incident to the parcel's routing node), never floating. `street_connectivity(...).connected_frac == 1.0` for every non-trivial block (floating roads grant no access, so this is also the efficacy guarantee).
- **New method, coexists** with `PeelReblocker`/`TopologyMethod` — does not touch them.
- **`methods/dijkstra.py` is a derivation module** — add it to `reblock.derive_graph._DERIVATION_MODULES` so editing it invalidates the `derive()` cache (like `methods/peel.py`/`topology.py`).
- Commit trailers on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
  ```

---

### Task 1: `_reblock_dijkstra` — the boundary-routed forest algorithm

**Files:**
- Create: `src/reblock/methods/dijkstra.py`
- Test: `tests/methods/test_dijkstra.py`

**Interfaces:**
- Consumes: `contracts.Block`; `derive.access.STREET_TOL` + `street_connectivity`; `networkx`, `shapely`.
- Produces: `_reblock_dijkstra(block: Block) -> gpd.GeoDataFrame` — columns `geometry` (LineString) + `drain` (int), rows ordered by `drain` descending; every segment street-connected.

- [ ] **Step 1: Write the failing test**

Create `tests/methods/test_dijkstra.py`. Use a synthetic 5×5 grid whose only street is the outer boundary (interior parcels are genuinely deep):

```python
from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, street_connectivity
from reblock.methods.dijkstra import _reblock_dijkstra

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


def test_reblock_dijkstra_is_deterministic() -> None:
    block = _grid_block(5)
    r1, r2 = _reblock_dijkstra(block), _reblock_dijkstra(block)
    assert [g.wkt for g in r1.geometry] == [g.wkt for g in r2.geometry]


def test_reblock_dijkstra_roads_all_reach_the_street() -> None:
    # forest rooted at street + attached spurs -> every segment street-connected
    block = _grid_block(5)
    roads = _reblock_dijkstra(block)
    assert len(roads) > 0
    conn = street_connectivity(block.streets, roads, STREET_TOL)
    assert conn.connected_frac == 1.0


def test_reblock_dijkstra_has_ordered_drainage() -> None:
    roads = _reblock_dijkstra(_grid_block(5))
    drains = list(roads["drain"])
    assert all(d >= 1 for d in drains)               # every road serves >=1 parcel
    assert drains == sorted(drains, reverse=True)     # arterials first
    assert max(drains) > 1                             # a real arterial exists (shared prefix)
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/methods/test_dijkstra.py -v`
Expected: FAIL — `No module named 'reblock.methods.dijkstra'`.

- [ ] **Step 3: Implement `src/reblock/methods/dijkstra.py`**

```python
"""DijkstraReblocker: route roads along the parcel-boundary graph as a shortest-path
forest rooted at the street (drainage-weighted, coverage-complete).

Deterministic, network-forming alternative to PeelReblocker's center-to-center descent:
roads follow parcel frontages (buildable) instead of cutting through parcels, shared
route-prefixes coalesce into arterials, and every segment reaches the street (so the
k-metric's street_connectivity grants each fronted parcel depth-1 access).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL

INF = float("inf")


def _rnd(c: tuple[float, float]) -> tuple[float, float]:
    return (round(c[0], 2), round(c[1], 2))   # snap to cm so shared vertices coincide


def _boundary_graph(parcels: gpd.GeoDataFrame) -> nx.Graph:
    """Planar graph of the tessellation: nodes = boundary vertices, edges = parcel
    boundary segments (shared party-walls dedup via unary_union), weight = length.
    Edges are added in sorted order for determinism."""
    noded = unary_union([g.boundary for g in parcels.geometry])
    lines = list(noded.geoms) if hasattr(noded, "geoms") else [noded]
    edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for ln in lines:
        cs = list(ln.coords)
        for a, b in zip(cs, cs[1:]):
            na, nb = _rnd(a), _rnd(b)
            if na != nb:
                edges.add((min(na, nb), max(na, nb)))
    g = nx.Graph()
    for na, nb in sorted(edges):
        g.add_edge(na, nb, weight=Point(na).distance(Point(nb)))
    return g


def _reblock_dijkstra(block: Block) -> gpd.GeoDataFrame:
    parcels = block.parcels
    g = _boundary_graph(parcels)
    street = unary_union(list(block.streets.geometry))
    corridor = street.buffer(STREET_TOL)
    snodes = {n for n in g.nodes if Point(n).distance(street) <= STREET_TOL}
    dist, paths = (nx.multi_source_dijkstra(g, snodes) if snodes else ({}, {}))

    drain: dict[frozenset[tuple[float, float]], int] = defaultdict(int)
    info: list[tuple[list[tuple[tuple[float, float], tuple[float, float]]],
                     tuple[float, float] | None]] = []
    # 1. shortest-path forest: route each non-street parcel's nearest node to the street.
    for geom in parcels.geometry:
        coords = [_rnd(c) for c in geom.exterior.coords]
        pes = [(a, b) for a, b in zip(coords, coords[1:]) if g.has_edge(a, b)]
        if not pes or any(LineString([a, b]).within(corridor) for a, b in pes):
            info.append((pes, None))                        # street-fronting -> served
            continue
        pn = [n for e in pes for n in e if n in dist]
        if not pn:
            info.append((pes, None))
            continue
        entry = min(pn, key=lambda n: (dist[n], n))          # deterministic
        for a, b in zip(paths[entry], paths[entry][1:]):
            drain[frozenset((a, b))] += 1
        info.append((pes, entry))

    forest = set(drain)
    # 2. coverage spurs: a parcel served only at a vertex gets its boundary edge incident
    #    to its routing node (so the spur attaches to the forest -- never floating).
    for pes, entry in info:
        if entry is None or any(frozenset(e) in forest for e in pes):
            continue
        incident = [e for e in pes if entry in e]
        if not incident:
            continue
        spur = min(incident, key=lambda e: (dist.get(e[0] if e[1] == entry else e[1], INF), e))
        drain[frozenset(spur)] += 1

    items = sorted(drain.items(), key=lambda kv: (-kv[1], sorted(kv[0])))
    rows = [{"geometry": LineString(sorted(e)), "drain": d} for e, d in items]
    return gpd.GeoDataFrame(rows, columns=["geometry", "drain"], geometry="geometry",
                            crs=block.crs)
```

(The `DijkstraReblocker` class is added in Task 2; leaving the `dataclass`/`Proposal` imports in place is fine — Task 2 uses them. If ruff flags them unused between tasks, add the class in Task 2 immediately after, or remove-then-readd; the two tasks land together on the branch.)

- [ ] **Step 4: Run to verify pass**

Run: `pixi run pytest tests/methods/test_dijkstra.py -v`
Expected: PASS (3 tests). If `ruff` flags the not-yet-used `dataclass`/`Proposal` imports, proceed to Task 2 (which uses them) before the full `pixi run check`; commit this task's files now.

- [ ] **Step 5: Commit**

```bash
git add src/reblock/methods/dijkstra.py tests/methods/test_dijkstra.py
git commit -m "$(cat <<'EOF'
feat: _reblock_dijkstra -- boundary shortest-path forest with drainage (DijkstraReblocker)

Builds the parcel-boundary graph, one multi-source Dijkstra from the street, unions
each interior parcel's route (accumulating per-edge drainage), attaches frontage spurs
for vertex-only-served parcels. Deterministic; every segment street-connected. The
DijkstraReblocker Method wraps it next.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

### Task 2: `DijkstraReblocker` Method + config + efficacy

**Files:**
- Modify: `src/reblock/methods/dijkstra.py` (add the `DijkstraReblocker` class)
- Modify: `src/reblock/derive_graph.py` (add `methods/dijkstra.py` to `_DERIVATION_MODULES`)
- Create: `conf/method/dijkstra.yaml`
- Test: `tests/methods/test_dijkstra.py` (efficacy + determinism-of-propose), `tests/test_run.py` (hydra-compose)

**Interfaces:**
- Consumes: `_reblock_dijkstra`; `contracts.{Block,Proposal,Method}`; `eval.kcomplexity.KComplexityEval`.
- Produces: `DijkstraReblocker()`; `identity = ("dijkstra",)`; `propose(block, prior=None) -> Proposal` with `roads` (drain-columned), `proposal_id="dijkstra"`, `method="dijkstra"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/methods/test_dijkstra.py`:

```python
import numpy as np

from reblock.eval.kcomplexity import KComplexityEval
from reblock.methods.dijkstra import DijkstraReblocker


def test_dijkstra_reblocks_a_synthetic_nested_block() -> None:
    # 3x3 grid: the centre parcel is landlocked at peel-depth 2; the boundary-routed
    # network reaches it, so k_after collapses to 1 (matches the peel/topology capstone).
    block = _grid_block(3)
    proposal = DijkstraReblocker().propose(block)
    m = KComplexityEval().score(block, proposal).values
    assert m["k_before"] == 2.0
    assert m["k_after"] == 1.0 and m["delta_k"] > 0
    assert m["connected_road_frac"] == 1.0
    assert proposal.roads is not None and len(proposal.roads) > 0
    assert proposal.proposal_id == "dijkstra" and proposal.method == "dijkstra"


def test_dijkstra_propose_is_deterministic_and_leaves_rng_untouched() -> None:
    block = _grid_block(5)
    np.random.seed(123)
    state = np.random.get_state()[1].tolist()
    p1 = DijkstraReblocker().propose(block)
    p2 = DijkstraReblocker().propose(block)
    assert np.random.get_state()[1].tolist() == state          # no global RNG side-effect
    assert [g.wkt for g in p1.roads.geometry] == [g.wkt for g in p2.roads.geometry]
```

Add a hydra-compose wiring test to `tests/test_run.py` (mirrors `test_hydra_compose_wires_peel_method`), proving `method=dijkstra` composes + reblocks a real block with real access improvement:

```python
def test_hydra_compose_wires_dijkstra_method() -> None:
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=[
            "data=dji", "method=dijkstra", "eval=kcomplexity", "max_blocks=1",
        ])
        results = run(spec_from_cfg(cfg)).results
    assert len(results) >= 1
    r = results[0]
    assert r.proposal.method == "dijkstra"
    assert r.metric("kcomplexity", "connected_road_frac") == 1.0
    assert r.metric("kcomplexity", "delta_k") > 0     # boundary network flattens a real block
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/methods/test_dijkstra.py tests/test_run.py -k "dijkstra" -v`
Expected: FAIL — `DijkstraReblocker` not defined; `method=dijkstra` config group missing.

- [ ] **Step 3: Add the `DijkstraReblocker` class**

Append to `src/reblock/methods/dijkstra.py`:

```python
@dataclass
class DijkstraReblocker:
    @property
    def identity(self) -> tuple[str]:
        return ("dijkstra",)

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior  # accepted for Method conformance; the routing is block-only
        roads = _reblock_dijkstra(block)
        spurs = int((roads["drain"] == 1).sum()) if len(roads) else 0
        return Proposal(block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
                        proposal_id="dijkstra", method="dijkstra",
                        params={"segments": len(roads), "leaf_roads": spurs},
                        block_identity=block.identity)
```

- [ ] **Step 4: Add the config group + the derivation-module hash entry**

Create `conf/method/dijkstra.yaml`:

```yaml
# Boundary-routed street network (shortest-path forest rooted at the street).
# Deterministic, no params -- see reblock.methods.dijkstra.DijkstraReblocker.
_target_: reblock.methods.dijkstra.DijkstraReblocker
```

In `src/reblock/derive_graph.py`, add `methods/dijkstra.py` to `_DERIVATION_MODULES` (next to `methods/peel.py`), so editing the method invalidates the `derive()` cache:

```python
    Path(__file__).parent / "methods" / "peel.py",
    Path(__file__).parent / "methods" / "dijkstra.py",
```

- [ ] **Step 5: Run tests + full check**

Run: `pixi run check`
Expected: PASS. The 3×3 grid collapses to `k_after == 1` (Δk > 0); `method=dijkstra` composes and reblocks DJI with `connected_road_frac == 1.0` + `delta_k > 0`; propose is deterministic and RNG-clean; `mypy --strict` clean. ~147 tests.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/methods/dijkstra.py src/reblock/derive_graph.py conf/method/dijkstra.yaml tests/methods/test_dijkstra.py tests/test_run.py
git commit -m "$(cat <<'EOF'
feat: DijkstraReblocker Method -- boundary-routed street network (method=dijkstra)

Thin deterministic Method over _reblock_dijkstra: propose() emits a drain-columned,
street-connected road network scored by the existing KComplexityEval. conf/method/
dijkstra.yaml + added to derive_graph code-version modules. 3x3 grid collapses to
k=1; method=dijkstra flattens a real DJI block (connected_road_frac 1.0, delta_k>0).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

## Self-Review

**Spec coverage:** the boundary graph + multi-source-Dijkstra forest + drainage + attached coverage spurs (Task 1); the `DijkstraReblocker` Method, `identity`, `Proposal` with `drain`, `conf/method/dijkstra.yaml`, and the `_DERIVATION_MODULES` entry (Task 2); efficacy verified on the synthetic 3×3 (Δk, k_after==1) and a real DJI block via the pipeline; determinism + RNG-cleanliness + street-connectivity tested. Drainage-weighted *rendering* and the incremental budget-curve framework are explicitly out of scope (sub-project 2).

**Placeholder scan:** complete code in every step. The Task-1 note about possibly-unused `dataclass`/`Proposal` imports until Task 2 is a real cross-task ordering fact (the two land together), not a placeholder.

**Type consistency:** `_reblock_dijkstra(block) -> gpd.GeoDataFrame` (columns `geometry`,`drain`) is consumed by `DijkstraReblocker.propose`; `identity -> tuple[str]` and `propose(block, prior=None) -> Proposal` match the `Method` protocol; `proposal_id`/`method` == `"dijkstra"` are asserted identically in the tests and set in `propose`. `street_connectivity(...).connected_frac` (float) is what both Task-1 and the eval assert `== 1.0`.

**Efficacy is guarded** by the synthetic 3×3 (`k_after == 1`) and the real-block compose test (`delta_k > 0`, `connected_road_frac == 1.0`), which together prove the boundary network actually reduces access-depth, not just that it draws roads.
