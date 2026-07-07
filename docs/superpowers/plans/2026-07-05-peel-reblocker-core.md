# Peel-reblocker (core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the peel-reblocker's connected descent spine (Slice 1: `full`-only, single `Proposal`) plus a connectivity-aware peel metric, so a second, deterministic, graph-free road-builder can be scored honestly against `topology`.

**Architecture:** New `methods/peel.py` routes roads by steepest descent on the parcel-adjacency graph, materialized as connected centerline corridors reaching the street. The existing peel metric (`derive/access.py`) is made connectivity-aware so a road only grants access if it connects to the street; `eval/kcomplexity.py` emits connectivity diagnostics. A shared `derive/adjacency.py` is factored out of `access.py`.

**Tech Stack:** Python 3.12, geopandas/shapely 2.x, networkx, pandas, Hydra, pytest, mypy --strict, ruff, pixi.

**Reference:** `docs/superpowers/specs/2026-07-05-peel-reblocker-design.md`.

## Global Constraints

- `pixi run check` (ruff + `mypy --strict src tests` + pytest) must be green at the end of every task.
- **No RNG anywhere in peel** — determinism is a hard requirement; all tie-breaks are explicit `min`/`sorted` by `parcel_id`.
- **One `tol`**, threaded: the reblocker passes its `self.tol` to both `parcel_access_layers` and `parcel_adjacency`.
- **`parcel_id`-space, never row position**, for layer lookups and tie-breaks (row order must not affect output).
- **No contract change this slice**: `Method.propose(block, prior=None) -> Proposal` returns a single `Proposal`; `run.py`, `contracts.py`, `topology.py` are untouched.
- **Connectivity-aware seeding = variant (a)** (confirmed by spike): keep road-segment touch-components that reach `block.streets`; discard floating ones.
- **Migrate, don't accommodate** (owner directive): move helpers and delete the old paths; no dual-path/back-compat.
- Peel roads are clean 2-point `LineString`s (centroid corridors).
- `STREET_TOL = 0.5` stays the single source of the seam tolerance (in `derive.access`).

---

### Task 1: Factor out `parcel_adjacency` into `derive/adjacency.py`

**Files:**
- Create: `src/reblock/derive/adjacency.py`
- Modify: `src/reblock/derive/access.py` (delete `_adjacency`/`_shared_len`, import from new module)
- Create: `tests/derive/test_adjacency.py`
- Modify: `tests/derive/test_access.py` (remove the moved snap-crash test + its fixture/imports)

**Interfaces:**
- Produces: `parcel_adjacency(geoms: list[BaseGeometry], tol: float) -> list[set[int]]` — positional neighbour sets; adjacent iff a positive-length shared boundary within `tol`. Also `_shared_len(gi, gj, tol) -> float` (module-private, GEOS-robust).
- Consumes: nothing new.

- [ ] **Step 1: Create `src/reblock/derive/adjacency.py`**

```python
"""Parcel adjacency: shared-boundary neighbour sets, robust to invalid geometry.

Two parcels are adjacent iff their boundaries share a positive-length run within
`tol`. Snapping bridges sub-`tol` digitization gaps; on real cadastral data
`snap()` can raise a GEOS side-location conflict for a messy touching pair, so
fall back to a direct intersection on validated operands. Positional index in,
neighbour sets out.
"""
from __future__ import annotations

from shapely import STRtree, make_valid, snap
from shapely.errors import GEOSException
from shapely.geometry.base import BaseGeometry


def _shared_len(gi: BaseGeometry, gj: BaseGeometry, tol: float) -> float:
    """Length of the boundary run parcels `gi`, `gj` share (0 if only a point)."""
    try:
        return float(snap(gi, gj, tol).intersection(gj).length)
    except GEOSException:
        return float(make_valid(gi).intersection(make_valid(gj)).length)


def parcel_adjacency(geoms: list[BaseGeometry], tol: float) -> list[set[int]]:
    """Positional neighbour sets: adjacent iff a positive-length shared boundary
    within `tol`. Candidate pairs come from an STRtree `dwithin` query; a snap
    bridges sub-`tol` gaps, and the shared run must be a line (not just a point),
    which excludes diagonally-touching parcels."""
    tree = STRtree(geoms)
    left, right = tree.query(geoms, predicate="dwithin", distance=tol)
    adj: list[set[int]] = [set() for _ in geoms]
    for i, j in zip(left.tolist(), right.tolist(), strict=True):
        if i >= j:
            continue  # each unordered pair once; also drops self-pairs (i == j)
        if _shared_len(geoms[i], geoms[j], tol) > 0:
            adj[i].add(j)
            adj[j].add(i)
    return adj
```

- [ ] **Step 2: Point `access.py` at the new module**

In `src/reblock/derive/access.py`: delete the `_shared_len` and `_adjacency` function definitions. Add `from reblock.derive.adjacency import parcel_adjacency`. In `parcel_access_layers`, replace `adj = _adjacency(geoms, tol)` with `adj = parcel_adjacency(geoms, tol)`. Then remove every import ruff now flags as unused — after this move that is `STRtree`, `make_valid`, `snap`, `GEOSException`, and `BaseGeometry` (all only used by the moved helpers); **keep** `union_all`, `deque`, `pd`, `GeoDataFrame`, `Block`, and the `STREET_TOL` definition. Run `ruff check src/reblock/derive/access.py` to confirm no unused-import warnings remain.

- [ ] **Step 3: Move the snap-crash regression test** — create `tests/derive/test_adjacency.py`

Move the `_SNAP_CRASH_A`/`_SNAP_CRASH_C` WKT fixtures and `test_snap_geosexception_falls_back_to_direct_intersection` out of `tests/derive/test_access.py` into the new file, importing `parcel_adjacency` instead of `_adjacency`. Add a grid adjacency test.

```python
from shapely import from_wkt, snap
from shapely.errors import GEOSException
from shapely.geometry import box

from reblock.derive.adjacency import parcel_adjacency

# <paste the exact _SNAP_CRASH_A and _SNAP_CRASH_C WKT constants from the current
#  tests/derive/test_access.py, verbatim>


def test_grid_adjacency_edge_only() -> None:
    # 2x2 unit grid: each cell has exactly 2 edge-neighbours (diagonal excluded).
    cells = [box(i, j, i + 1, j + 1) for i in range(2) for j in range(2)]
    adj = parcel_adjacency(cells, 0.5)
    assert all(len(n) == 2 for n in adj)          # no diagonal (corner-only) links
    assert adj[0] == {1, 2}                        # (0,0) touches (0,1) and (1,0)


def test_snap_geosexception_falls_back_to_direct_intersection() -> None:
    a, c = from_wkt(_SNAP_CRASH_A), from_wkt(_SNAP_CRASH_C)
    try:
        snap(a, c, 0.5).intersection(c)
    except GEOSException:
        pass
    else:  # pragma: no cover - fixture-integrity guard
        raise AssertionError("fixture no longer triggers the snap GEOSException")
    adj = parcel_adjacency([a, c], 0.5)
    assert adj[0] == {1}
    assert adj[1] == {0}
```

Then delete those moved symbols from `tests/derive/test_access.py` (the two WKT constants, the test, and the now-unused `from_wkt`/`snap`/`GEOSException`/`_adjacency` imports).

- [ ] **Step 4: Run the suite**

Run: `pixi run check`
Expected: ruff + mypy clean; all tests pass (adjacency behaviour is unchanged, just relocated).

- [ ] **Step 5: Commit**

```bash
git add src/reblock/derive/adjacency.py src/reblock/derive/access.py tests/derive/test_adjacency.py tests/derive/test_access.py
git commit -m "refactor: extract parcel_adjacency into derive/adjacency.py"
```

---

### Task 2: Connectivity-aware peel seeding (`derive/access.py`)

**Files:**
- Modify: `src/reblock/derive/access.py`
- Modify: `tests/derive/test_access.py`

**Interfaces:**
- Produces: `street_connectivity(streets: GeoDataFrame, roads: GeoDataFrame | None, tol: float) -> StreetConnectivity` where `StreetConnectivity = NamedTuple(seed_geom: BaseGeometry | None, n_components: int, connected_frac: float)`. `parcel_access_layers` seeds from `.seed_geom`.
- Consumes: `parcel_adjacency` (Task 1).

- [ ] **Step 1: Write failing tests** in `tests/derive/test_access.py`

```python
def test_access_before_unchanged_by_connectivity_change() -> None:
    # roads=None: seed_geom is exactly the streets, so before-layers are identical
    # to the pre-change behaviour (2x2 -> all layer 1, 3x3 -> centre layer 2).
    assert parcel_access_layers(_grid_block(2), None).max() == 1
    assert parcel_access_layers(_grid_block(3), None).max() == 2


def test_disconnected_road_gives_no_credit() -> None:
    # 1x5 strip, street on the left edge only -> depths 1..5. A road segment
    # floating in the interior (NOT touching the street) must confer NO access:
    # k stays 5, exactly as with no roads.
    polys = [Polygon([(i, 0), (i+1, 0), (i+1, 1), (i, 1)]) for i in range(5)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(5))}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])], crs=UTM)
    block = Block(block_id="s", crs=UTM, boundary=cast(Polygon, parcels.geometry.union_all()),
                  parcels=parcels, streets=streets)
    floating = gpd.GeoDataFrame(geometry=[LineString([(3, 0.5), (4, 0.5)])], crs=UTM)  # deep interior
    assert parcel_access_layers(block, floating).max() == 5          # no unearned credit
    assert parcel_access_layers(block, None).max() == 5


def test_connected_road_reduces_depth() -> None:
    # Same strip, but a road running from the street (x=0) inward DOES connect,
    # so it grants access and reduces depth.
    polys = [Polygon([(i, 0), (i+1, 0), (i+1, 1), (i, 1)]) for i in range(5)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(5))}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])], crs=UTM)
    block = Block(block_id="s", crs=UTM, boundary=cast(Polygon, parcels.geometry.union_all()),
                  parcels=parcels, streets=streets)
    connected = gpd.GeoDataFrame(geometry=[LineString([(0, 0.5), (4, 0.5)])], crs=UTM)  # touches street
    assert parcel_access_layers(block, connected).max() == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/derive/test_access.py -k "connectivity or disconnected or connected_road" -v`
Expected: `test_disconnected_road_gives_no_credit` FAILS (current blind seeding wrongly credits the floating segment, giving k < 5).

- [ ] **Step 3: Implement `street_connectivity` and use it for seeding**

In `src/reblock/derive/access.py` add `import networkx as nx` and `from typing import NamedTuple`, `from shapely import STRtree` (re-add), and:

```python
class StreetConnectivity(NamedTuple):
    seed_geom: BaseGeometry | None      # streets + road segments that reach a street
    n_components: int                   # touch-components among the road segments
    connected_frac: float               # road length in street-connected components / total


def _line_parts(roads: GeoDataFrame | None) -> list[BaseGeometry]:
    """Positive-length line parts of `roads` (explode Multi*; drop non-lines)."""
    if roads is None or roads.empty:
        return []
    parts: list[BaseGeometry] = []
    for g in roads.geometry:
        if g is None or g.is_empty:
            continue
        geoms = list(g.geoms) if g.geom_type.startswith("Multi") else [g]
        parts.extend(p for p in geoms if "LineString" in p.geom_type and p.length > 0)
    return parts


def street_connectivity(
    streets: GeoDataFrame, roads: GeoDataFrame | None, tol: float
) -> StreetConnectivity:
    """Seed geometry = streets plus only those road segments whose touch-component
    reaches a street (variant a). Floating interior roads grant no access."""
    street_geom = union_all(list(streets.geometry)) if len(streets) else None
    segs = _line_parts(roads)
    if not segs:
        return StreetConnectivity(street_geom, 0, 0.0)
    tree = STRtree(segs)
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(range(len(segs)))
    for i, g in enumerate(segs):
        for j in tree.query(g, predicate="dwithin", distance=tol):
            jj = int(j)
            if i < jj:
                graph.add_edge(i, jj)
    comps = list(nx.connected_components(graph))
    live: list[int] = []
    for comp in comps:
        if street_geom is not None and any(segs[i].distance(street_geom) <= tol for i in comp):
            live.extend(comp)
    total = sum(segs[i].length for i in range(len(segs)))
    live_len = sum(segs[i].length for i in live)
    frac = live_len / total if total > 0 else 0.0
    parts = ([street_geom] if street_geom is not None else []) + [segs[i] for i in live]
    seed = union_all(parts) if parts else None
    return StreetConnectivity(seed, len(comps), frac)
```

Then in `parcel_access_layers`, replace the current seed construction:

```python
    # was: seed_geoms = list(block.streets.geometry); if roads...; street = union_all(...)
    street = street_connectivity(block.streets, roads, tol).seed_geom
```

(Leave the rest of the BFS unchanged; `roads=None` yields `seed == streets`, so `access_before` is identical.)

- [ ] **Step 4: Run tests**

Run: `pixi run check`
Expected: the three new tests pass; the existing `test_added_road_reduces_depth` (connector touches the street) still passes; ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/reblock/derive/access.py tests/derive/test_access.py
git commit -m "feat: connectivity-aware peel seeding (roads must reach the street)"
```

---

### Task 3: Connectivity diagnostics + topology-unchanged gate (`eval/kcomplexity.py`)

**Files:**
- Modify: `src/reblock/eval/kcomplexity.py`
- Modify: `tests/eval/test_kcomplexity.py`

**Interfaces:**
- Consumes: `street_connectivity` (Task 2).
- Produces: `KComplexityEval.score` adds `values["n_road_components"]` and `values["connected_road_frac"]`.

- [ ] **Step 1: Write failing tests** in `tests/eval/test_kcomplexity.py`

```python
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import box
from reblock.contracts import Block, Proposal
from reblock.eval.kcomplexity import KComplexityEval
from reblock.methods.topology import TopologyMethod

UTM = CRS.from_epsg(32643)


def _grid5() -> Block:
    polys = [box(i, j, i + 1, j + 1) for i in range(5) for j in range(5)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(25))}, geometry=polys, crs=UTM)
    b = parcels.geometry.union_all()
    return Block(block_id="g5", crs=UTM, boundary=b, parcels=parcels,
                 streets=gpd.GeoDataFrame(geometry=[b.exterior], crs=UTM))


def test_topology_roads_are_street_connected_and_unchanged() -> None:
    # Gate: topology's interior roads reach block.streets, so the connectivity-
    # aware metric neither drops its access (k stays 1) nor flags disconnection.
    block = _grid5()
    proposal = TopologyMethod(alpha=2.0, seed=0).propose(block)
    m = KComplexityEval().score(block, proposal)
    assert m.values["k_after"] == 1.0
    assert m.values["connected_road_frac"] == 1.0
    assert m.values["n_road_components"] >= 1


def test_diagnostics_present_and_zero_for_no_roads() -> None:
    block = _grid5()
    empty = Proposal(block_id="g5", crs=UTM, roads=gpd.GeoDataFrame(geometry=[], crs=UTM),
                     method="none")
    m = KComplexityEval().score(block, empty)
    assert m.values["n_road_components"] == 0.0
    assert m.values["connected_road_frac"] == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/eval/test_kcomplexity.py -k "connected or diagnostics" -v`
Expected: FAIL with `KeyError: 'connected_road_frac'`.

- [ ] **Step 3: Emit diagnostics in `KComplexityEval.score`**

In `src/reblock/eval/kcomplexity.py`, import `from reblock.derive.access import STREET_TOL, parcel_access_layers, street_connectivity`, and extend the `values` dict:

```python
        sc = street_connectivity(block.streets, proposal.roads, STREET_TOL)
        return Metrics(block_id=block.block_id, method=proposal.method, eval="kcomplexity",
                       values={"k_before": float(kb), "k_after": float(ka),
                               "delta_k": float(kb - ka), "added_road_length_m": added,
                               "n_road_components": float(sc.n_components),
                               "connected_road_frac": sc.connected_frac},
                       fields={"access_before": pre, "access_after": post})
```

- [ ] **Step 4: Run tests**

Run: `pixi run check`
Expected: new tests pass; existing kcomplexity tests pass; ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/reblock/eval/kcomplexity.py tests/eval/test_kcomplexity.py
git commit -m "feat: emit road-connectivity diagnostics; gate topology unchanged"
```

---

### Task 4: `PeelReblocker` — connected descent spine (`methods/peel.py`)

**Files:**
- Create: `src/reblock/methods/peel.py`
- Create: `tests/methods/test_peel.py`

**Interfaces:**
- Consumes: `parcel_access_layers` (Task 2), `parcel_adjacency` (Task 1), `STREET_TOL`.
- Produces: `PeelReblocker(tol=STREET_TOL).propose(block, prior=None) -> Proposal`, `proposal_id="peel"`, `method="peel"`, `roads` = spine LineStrings, `edges=None`, `params={"unreachable": int}`.

- [ ] **Step 1: Write failing tests** in `tests/methods/test_peel.py`

```python
from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Polygon, box

from reblock.contracts import Block
from reblock.derive.access import parcel_access_layers, street_connectivity, STREET_TOL
from reblock.methods.peel import PeelReblocker

UTM = CRS.from_epsg(32643)


def _grid5() -> Block:
    polys = [box(i, j, i + 1, j + 1) for i in range(5) for j in range(5)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(25))}, geometry=polys, crs=UTM)
    b = parcels.geometry.union_all()
    return Block(block_id="g5", crs=UTM, boundary=b, parcels=parcels,
                 streets=gpd.GeoDataFrame(geometry=[b.exterior], crs=UTM))


def test_spine_reaches_k1_and_is_street_connected() -> None:
    block = _grid5()
    proposal = PeelReblocker().propose(block)
    assert parcel_access_layers(block, proposal.roads).max() == 1          # full access
    sc = street_connectivity(block.streets, proposal.roads, STREET_TOL)
    assert sc.connected_frac == 1.0                                        # every corridor reaches street
    assert proposal.proposal_id == "peel" and proposal.method == "peel"
    assert proposal.edges is None


def test_deterministic_under_row_shuffle() -> None:
    # parcel_id != row position, rows shuffled: identical roads (min-id tie-break,
    # not row order). Compare sorted WKT of the produced segments.
    block = _grid5()
    shuffled = block.parcels.sample(frac=1, random_state=3).reset_index(drop=True)
    block2 = Block(block_id="g5", crs=block.crs, boundary=block.boundary,
                   parcels=shuffled, streets=block.streets)
    r1 = sorted(g.wkt for g in PeelReblocker().propose(block).roads.geometry)
    r2 = sorted(g.wkt for g in PeelReblocker().propose(block2).roads.geometry)
    assert r1 == r2


def test_unreachable_island_is_skipped_and_counted() -> None:
    # A parcel disconnected from everything (no adjacency, no street) has no
    # descent parent -> skipped, counted, and left deep in k_after.
    near = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])          # touches left-edge street
    mid = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])           # layer 2 via `near`
    island = Polygon([(50, 50), (51, 50), (51, 51), (50, 51)])  # disconnected
    parcels = gpd.GeoDataFrame({"parcel_id": [0, 1, 2]}, geometry=[near, mid, island], crs=UTM)
    from shapely.geometry import LineString
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])], crs=UTM)
    hull = cast(Polygon, parcels.geometry.union_all().convex_hull)
    block = Block(block_id="d", crs=UTM, boundary=hull, parcels=parcels, streets=streets)
    proposal = PeelReblocker().propose(block)
    assert proposal.params["unreachable"] == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/methods/test_peel.py -v`
Expected: FAIL with `ModuleNotFoundError: reblock.methods.peel`.

- [ ] **Step 3: Implement `PeelReblocker`**

```python
"""PeelReblocker: route roads by steepest descent down the access-depth peel.

Deterministic, graph-free second method. Each interior parcel links (via its
centroid) to a descent parent one layer shallower; every root (a street-adjacent
parcel that is some parcel's parent) is tied to the nearest street point. The
union is a connected centerline network reaching the street (full access).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
from shapely import union_all
from shapely.geometry import LineString
from shapely.ops import nearest_points

from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency


@dataclass
class PeelReblocker:
    tol: float = STREET_TOL

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior  # accepted for Method conformance; steepest-descent is block-only
        ids = list(block.parcels["parcel_id"])
        geoms = list(block.parcels.geometry)
        pos = {pid: i for i, pid in enumerate(ids)}
        layer = parcel_access_layers(block, None, tol=self.tol)
        depth = {pid: int(layer.loc[pid]) for pid in ids}
        adj = parcel_adjacency(geoms, self.tol)
        cent = [g.centroid for g in geoms]
        street = union_all(list(block.streets.geometry))

        segments: list[LineString] = []
        roots: set[int] = set()
        unreachable = 0
        for i, pid in enumerate(ids):
            if depth[pid] < 2:
                continue
            # descent parent: adjacent parcel one layer shallower, min parcel_id.
            # Keyed by parcel_id (not row position) so row order can't change it.
            cands = [ids[q] for q in adj[i] if depth[ids[q]] == depth[pid] - 1]
            if not cands:
                unreachable += 1          # disconnected island: no gradient to a street
                continue
            parent = min(cands)
            segments.append(LineString([cent[i], cent[pos[parent]]]))
            if depth[parent] == 1:
                roots.add(parent)
        for pid in sorted(roots):         # sorted -> deterministic street stubs
            on_street = nearest_points(cent[pos[pid]], street)[1]
            segments.append(LineString([cent[pos[pid]], on_street]))

        roads = gpd.GeoDataFrame(geometry=segments, crs=block.crs)
        return Proposal(block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
                        proposal_id="peel", method="peel",
                        params={"unreachable": unreachable})
```

- [ ] **Step 4: Run tests**

Run: `pixi run check`
Expected: all `test_peel.py` pass; ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/reblock/methods/peel.py tests/methods/test_peel.py
git commit -m "feat: PeelReblocker connected descent spine (full-only)"
```

---

### Task 5: Wire the method into Hydra + head-to-head efficacy

**Files:**
- Create: `conf/method/peel.yaml`
- Modify: `tests/test_run.py` (Hydra-wiring test)
- Modify: `tests/methods/test_peel.py` (direct head-to-head)

**Interfaces:**
- Consumes: `PeelReblocker` (Task 4), the existing `run()` (unchanged — single `Proposal` per method).

Two separate concerns, because `run()` can only source blocks via a `_target_`-instantiable
`ShapefileSource` (no synthetic `Block`), and every real fixture (Phule, Epworth_demo) scores
`k_before == 1` with no interior — so peel proposes empty roads through `run()`. So: the `run()`-
level test verifies **Hydra wiring** (peel composes + runs clean, producing a `peel` `Result`); the
**efficacy** head-to-head (peel reaches k=1 with a connected network) runs on a direct synthetic
`Block`, where topology already demonstrably reblocks (`tests/test_run.py::test_topology_reblocks_a_synthetic_nested_block`).

- [ ] **Step 1: Create `conf/method/peel.yaml`**

`conf/method/topology.yaml` is a **list** (not a dict, no `# @package` line) so a block can be
scored under several methods. Mirror that exactly; `PeelReblocker` takes only `tol`, which defaults
to `STREET_TOL`, so no fields are needed:

```yaml
# A list, not a dict (see conf/method/topology.yaml): run() instantiates
# cfg.method as a list of Methods so a block can be scored under several.
- _target_: reblock.methods.peel.PeelReblocker
```

- [ ] **Step 2: Write the failing Hydra-wiring test** in `tests/test_run.py`

Mirrors `test_hydra_compose_wires_config_groups` (same file) but selects `method=peel`. On Phule the
peel proposes empty roads (k stays 1), so this asserts the config group instantiates and `run()`
completes with a well-formed `peel` `Result` — not efficacy.

```python
def test_hydra_compose_wires_peel_method() -> None:
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=[
            "data=phule", "method=peel", "eval=kcomplexity",
            f"shapefile={PHULE}", "assumed_crs=3857", "max_blocks=1",
        ])
        results = run(cfg)
    assert len(results) == 1
    r = results[0]
    assert r.proposal.method == "peel" and r.proposal.proposal_id == "peel"
    assert r.metric("kcomplexity", "k_after") <= r.metric("kcomplexity", "k_before")
```

- [ ] **Step 3: Run to verify it fails**

Run: `pytest tests/test_run.py -k wires_peel -v`
Expected: FAIL — `conf/method/peel.yaml` missing → Hydra `MissingConfigException`.

- [ ] **Step 4: Add the head-to-head efficacy test** in `tests/methods/test_peel.py`

Direct synthetic `Block` (reuse the `_grid5()` helper from Task 4), both methods, connectivity-aware
eval — the "compared side by side" deliverable without `run()`'s shapefile constraint.

```python
def test_head_to_head_both_reach_k1_peel_connected() -> None:
    from reblock.methods.topology import TopologyMethod
    from reblock.eval.kcomplexity import KComplexityEval
    block = _grid5()
    topo = KComplexityEval().score(block, TopologyMethod(alpha=2.0, seed=0).propose(block)).values
    peel = KComplexityEval().score(block, PeelReblocker().propose(block)).values
    assert topo["k_after"] == 1.0 and peel["k_after"] == 1.0        # both fully reblock
    assert peel["connected_road_frac"] == 1.0                        # peel network reaches the street
    assert peel["added_road_length_m"] > 0                           # it actually laid roads
```

- [ ] **Step 5: Run the full suite**

Run: `pixi run check`
Expected: both new tests pass; all green; ruff + mypy clean.

- [ ] **Step 6: Commit**

```bash
git add conf/method/peel.yaml tests/test_run.py tests/methods/test_peel.py
git commit -m "feat: wire PeelReblocker into Hydra; head-to-head efficacy test"
```

---

## Notes for the executor

- **Risk already retired:** the connectivity-aware metric was spike-verified safe for topology (`connected_road_frac == 1.0` on a 5×5 grid) before this plan — Task 3's gate re-confirms it in-suite. If that gate ever fails on a real block, stop and escalate (do not silently weaken the metric).
- **Determinism is load-bearing:** Task 4's shuffle test is the guard. If it fails, the bug is a row-position leaking into a tie-break — fix the id-keying, don't relax the test.
- **Out of scope (Slice 2):** the budget sweep (`length`/`depth`/`any_of`), the `Method.propose -> Iterable[Proposal]` migration, `Metrics.proposal_id`, trunk-merging. Do not build them here.
