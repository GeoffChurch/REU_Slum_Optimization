# Slice 1 Walking Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove reblock's data→methods→eval interfaces end-to-end on in-repo data: topology's Phule Nagar parcels → a `Block` → topology's greedy road-builder → a k-complexity Δ eval, wired under Hydra — with topology itself made a properly-typed, co-developed dependency.

**Architecture:** A canonical `Block` waist (typed contracts) with thin adapters. topology is restructured into a `topology.*` package with `py.typed`, its public surface typed, and its native k-complexity exposed. reblock's `ShapefileSource` reads Phule Nagar into `Block`s; `to_parcel_graph` derives topology's `MyGraph`; `TopologyMethod` wraps the road-builder into a `Proposal`; `KComplexityEval` calls topology's exposed `k_complexity` before/after. `reblock.run` composes them via Hydra.

**Tech Stack:** Python ≥3.11, geopandas/shapely 2.0, networkx, scipy, matplotlib, pyshp, Hydra, pixi/conda-forge, pytest, mypy --strict, ruff.

## Global Constraints

- Python `>=3.11,<3.13`; conda deps via pixi/conda-forge (`platforms = ["linux-64"]`).
- `mypy --strict` clean on reblock `src`/`tests`; topology ships `py.typed` with its public surface typed (no `ignore_missing_imports` for `topology`).
- ruff select `["E","F","I","UP","B"]`, line-length 100, `ext/` excluded from reblock's ruff.
- CRS: in-memory `Block`s are projected local UTM (metres); never a geographic CRS.
- **Two-repo workflow:** Task 1 edits live in the `ext/topology` submodule (own git repo, branch `master`) — commit there, then repin reblock (`git add ext/topology`). All later tasks are reblock-only.
- **topology's standalone shapefile/GeoJSON I/O stays** — it's a legitimate option, not a shim; the shapefile loader powers the Task 4 oracle.
- TDD: failing-test → verify-fail → implement → verify-pass → commit. `pixi run check` green at each task end.

---

## File structure

```
ext/topology/                         # submodule (owned)
  topology/__init__.py                # typed facade: re-exports MyGraph, build_all_roads, k_complexity, ...
  topology/py.typed
  topology/graph/{__init__,my_graph,my_graph_helpers}.py   # moved from ext/topology/graph/
  topology/utils/{__init__,lazy_property}.py               # moved from ext/topology/utils/
  setup.py                            # packages=find_packages(include=["topology","topology.*"])

src/reblock/
  contracts.py            # Region, Block, Proposal, Metrics, Protocols + validation
  data/shapefile.py       # ShapefileSource: shapefile -> Region[Block]
  derive/parcel_graph.py  # PlanarParcelGraph + to_parcel_graph(Block)
  methods/topology.py     # TopologyMethod.propose(Block) -> Proposal
  eval/kcomplexity.py     # KComplexityEval.score -> Metrics (wraps topology.k_complexity)
  run.py                  # Hydra entrypoint
tests/ ...                # mirrors src
```

---

### Task 1: topology facade — `topology.*` namespace, typing, exposed `k_complexity`

**Files (all under `ext/topology`):**
- Create: `topology/__init__.py`, `topology/py.typed`
- Move: `graph/` → `topology/graph/`, `utils/` → `topology/utils/`
- Modify: `topology/graph/my_graph.py` (one import line), `setup.py`, `tests/*.py`, `examples/*.py`
- Create: `topology/graph/complexity.py` (the exposed `k_complexity`)
- Test: `ext/topology/tests/test_k_complexity.py`
- Modify (reblock): `pyproject.toml` (deps + editable install)

**Interfaces:**
- Produces (importable from reblock): `from topology import MyGraph, MyNode, MyEdge, MyFace, graphFromMyFaces, build_all_roads, import_and_setup, define_roads_on, k_complexity`.
- `k_complexity(graph: MyGraph) -> int` — Brelsford weak-dual nesting depth; road-relative (the dual excludes road edges). Calibrated so a street-on-one-end strip of N parcels yields k = N.

- [ ] **Step 1: Move the packages under a `topology/` namespace.**

```bash
cd ext/topology
mkdir -p topology
git mv graph topology/graph
git mv utils topology/utils
touch topology/__init__.py topology/py.typed
```

- [ ] **Step 2: Fix the one absolute intra-package import.** In `topology/graph/my_graph.py`, change the `utils` import (line ~10):

```python
# from utils.lazy_property import lazy_property
from topology.utils.lazy_property import lazy_property
```

(The `from . import my_graph_helpers as mgh` in `my_graph.py` and `from . import my_graph as mg` in `my_graph_helpers.py` are relative and need no change.)

- [ ] **Step 3: Update topology's own tests and examples.** In `tests/perf_tests.py`, `tests/unit_tests.py`, `tests/legacy_util_tests.py`, `examples/Example.py`, `examples/Barriers_example.py`, rewrite:

```python
# from graph import my_graph as mg
# from graph import my_graph_helpers as mgh
from topology.graph import my_graph as mg
from topology.graph import my_graph_helpers as mgh
```

- [ ] **Step 4: Restrict the installed package in `setup.py`.**

```python
from setuptools import setup, find_packages

setup(name='topology', version='2.0',
      packages=find_packages(include=['topology', 'topology.*']),
      package_data={'topology': ['py.typed']},
      python_requires='>=3.7')
```

- [ ] **Step 5: Write the exposed `k_complexity`** in `topology/graph/complexity.py`. topology's `stacked_duals`/`form_equivalence_classes` already compute the nesting; this wraps them behind a typed, side-effect-free function.

```python
"""k-complexity: the Brelsford weak-dual nesting depth of a parcel graph.

The weak dual excludes road edges, so k is relative to the current road set and
drops as roads are added. Exposed as the canonical metric reblock's eval imports.
"""
from __future__ import annotations

from .my_graph import MyGraph
from .my_graph_helpers import form_equivalence_classes


def k_complexity(graph: MyGraph) -> int:
    if graph.G.number_of_nodes() < 2:
        return 0
    graph.inner_facelist  # ensure faces are traced
    result, _ = form_equivalence_classes(graph)
    layers = [depth for depth, faces in result.items() if faces]
    return (max(layers) + 1) // 2 if layers else 0
```

- [ ] **Step 6: Add a typed road-marking helper + facade re-exports.** Append to `topology/graph/complexity.py`:

```python
from .my_graph import MyEdge


def define_roads_on(graph: MyGraph, is_road: "Callable[[MyEdge], bool]") -> None:
    """Mark exactly the edges satisfying `is_road` as roads (idempotent reset)."""
    for edge in graph.myedges():
        edge.road = bool(is_road(edge))
```

Add the import at the top of the file: `from typing import Callable`. Then write `topology/__init__.py`:

```python
"""topology: parcel-graph reblocking + the Brelsford k-complexity metric."""
from topology.graph.complexity import define_roads_on, k_complexity
from topology.graph.my_graph import MyEdge, MyFace, MyGraph, MyNode
from topology.graph.my_graph_helpers import (
    build_all_roads,
    graphFromMyFaces,
    import_and_setup,
)

__all__ = [
    "MyGraph", "MyNode", "MyEdge", "MyFace",
    "graphFromMyFaces", "build_all_roads", "import_and_setup",
    "k_complexity", "define_roads_on",
]
```

- [ ] **Step 7: Write the failing calibration test** `ext/topology/tests/test_k_complexity.py`. A 1×N grid of parcels with the two long sides forming the block; mark the far-left edge as the only road → the strip is N layers deep.

```python
from topology import MyEdge, MyFace, MyNode, define_roads_on, graphFromMyFaces, k_complexity


def _strip(n: int):
    # n unit squares in a row; build faces from explicit edges
    faces = []
    for i in range(n):
        c = [MyNode((i, 0)), MyNode((i + 1, 0)), MyNode((i + 1, 1)), MyNode((i, 1))]
        edges = [MyEdge((c[j], c[(j + 1) % 4])) for j in range(4)]
        faces.append(MyFace(edges))
    return graphFromMyFaces(faces)


def test_k_of_strip_with_far_left_street() -> None:
    for n in (1, 3, 5):
        g = _strip(n)
        define_roads_on(g, lambda e: e.nodes[0].x == 0 and e.nodes[1].x == 0)
        assert k_complexity(g) == n
```

- [ ] **Step 8: Run the calibration test.**

Run: `cd ext/topology && python -m pytest tests/test_k_complexity.py -v` (topology must be importable — install first, Step 10, or `pip install -e .` in a scratch env)
Expected: PASS. **If the returned integers are off by a constant** (e.g. all +1, or the `(max+1)//2` convention is wrong), adjust the final expression in `k_complexity` until the strip yields exactly `n`, then re-run. The strip values are the definition; pin the formula to them.

- [ ] **Step 9: Run topology's existing suite to confirm the move didn't break it.**

Run: `cd ext/topology && python -m pytest tests/unit_tests.py -v`
Expected: PASS (same as before the move).

- [ ] **Step 10: Add deps + editable install to reblock `pyproject.toml`.** Under `[tool.pixi.dependencies]` add topology's runtime stack:

```toml
networkx = "*"
scipy = "*"
matplotlib-base = "*"
pyshp = "*"
hydra-core = "*"
```

Under `[tool.pixi.pypi-dependencies]` add:

```toml
topology = { path = "ext/topology", editable = true }
```

- [ ] **Step 11: Install and verify the typed import from reblock.** Create `tests/methods/test_topology_import.py`:

```python
def test_topology_public_api() -> None:
    import topology
    for name in ("MyGraph", "graphFromMyFaces", "build_all_roads", "k_complexity"):
        assert hasattr(topology, name)
```

Run: `pixi install && pixi run pytest tests/methods/test_topology_import.py -v && pixi run typecheck`
Expected: PASS; mypy `--strict` clean (topology's `py.typed` + typed facade means no ignore needed).

- [ ] **Step 12: Commit — in the submodule, then repin reblock.**

```bash
git -C ext/topology add -A
git -C ext/topology commit -m "Restructure into topology.* package; add py.typed + k_complexity facade"
git add ext/topology pyproject.toml tests/methods/test_topology_import.py
git commit -m "feat: vendor topology as typed editable dep; expose k_complexity"
```

---

### Task 2: Core contracts

**Files:**
- Create: `src/reblock/contracts.py`
- Test: `tests/test_contracts.py`
- Create (empty): `src/reblock/data/__init__.py`, `src/reblock/derive/__init__.py`, `src/reblock/methods/__init__.py`, `src/reblock/eval/__init__.py`, `tests/__init__.py`, `tests/data/__init__.py`, `tests/derive/__init__.py`, `tests/methods/__init__.py`

**Interfaces:**
- Produces: dataclasses `Region`, `Block`, `Proposal`, `Metrics` and Protocols `Source`, `Screen`, `Method`, `Eval`. `Block.__post_init__` raises `ValueError` on geographic CRS, missing `parcel_id`/`geometry` column, or empty parcels.

- [ ] **Step 1: Create the empty package dirs.**

```bash
mkdir -p src/reblock/data src/reblock/derive src/reblock/methods src/reblock/eval tests/data tests/derive tests/methods
touch src/reblock/data/__init__.py src/reblock/derive/__init__.py src/reblock/methods/__init__.py src/reblock/eval/__init__.py
touch tests/__init__.py tests/data/__init__.py tests/derive/__init__.py tests/methods/__init__.py
```

- [ ] **Step 2: Write the failing tests** `tests/test_contracts.py`:

```python
import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.contracts import Block, Metrics, Proposal

UTM = CRS.from_epsg(32643)  # WGS84 / UTM 43N (metres)


def _parcels() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"parcel_id": [0]},
                            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs=UTM)


def _streets() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs=UTM)


def test_block_constructs() -> None:
    b = Block(block_id="phule_0", crs=UTM, boundary=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
              parcels=_parcels(), streets=_streets())
    assert b.block_id == "phule_0" and b.crs.is_projected


def test_block_rejects_geographic_crs() -> None:
    with pytest.raises(ValueError, match="projected"):
        Block(block_id="x", crs=CRS.from_epsg(4326),
              boundary=Polygon([(0, 0), (1, 0), (1, 1)]),
              parcels=_parcels().to_crs(4326), streets=_streets().to_crs(4326))


def test_block_rejects_missing_parcel_id() -> None:
    with pytest.raises(ValueError, match="parcel_id"):
        Block(block_id="x", crs=UTM, boundary=Polygon([(0, 0), (1, 0), (1, 1)]),
              parcels=_parcels().drop(columns=["parcel_id"]), streets=_streets())


def test_metrics_and_proposal_records() -> None:
    m = Metrics(block_id="x", method="topology", eval="kcomplexity",
                values={"k_before": 3.0, "k_after": 1.0})
    assert m.values["k_before"] == 3.0
    assert Proposal(block_id="x", crs=UTM, method="topology").roads is None
```

- [ ] **Step 3: Run tests to verify they fail.**

Run: `pixi run pytest tests/test_contracts.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'reblock.contracts'`

- [ ] **Step 4: Write `src/reblock/contracts.py`.**

```python
"""Canonical typed contracts — the waist every layer adapts to."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Protocol

from geopandas import GeoDataFrame
from pyproj import CRS
from shapely.geometry import Polygon


def _require_columns(gdf: GeoDataFrame, cols: set[str], name: str) -> None:
    missing = cols - set(gdf.columns)
    if missing:
        raise ValueError(f"{name} is missing required column(s): {sorted(missing)}")


def _require_projected(crs: CRS, name: str) -> None:
    if crs is None or not CRS.from_user_input(crs).is_projected:
        raise ValueError(f"{name} must have a projected (metric) CRS, got: {crs}")


@dataclass(frozen=True)
class Region:
    region_id: str
    crs: CRS
    blocks: Iterable["Block"]
    roads: GeoDataFrame | None = None
    water: GeoDataFrame | None = None
    food: GeoDataFrame | None = None
    healthcare: GeoDataFrame | None = None
    attrs: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Block:
    block_id: str
    crs: CRS
    boundary: Polygon
    parcels: GeoDataFrame
    streets: GeoDataFrame
    buildings: GeoDataFrame | None = None
    water: GeoDataFrame | None = None
    barriers: GeoDataFrame | None = None
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_projected(self.crs, "Block.crs")
        _require_columns(self.parcels, {"parcel_id", "geometry"}, "Block.parcels")
        if self.parcels.empty:
            raise ValueError("Block.parcels must be non-empty")
        _require_columns(self.streets, {"geometry"}, "Block.streets")


@dataclass(frozen=True)
class Proposal:
    block_id: str
    crs: CRS
    roads: GeoDataFrame | None = None
    water_points: GeoDataFrame | None = None
    water_mains: GeoDataFrame | None = None
    edges: GeoDataFrame | None = None
    method: str = ""
    params: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Metrics:
    block_id: str
    method: str
    eval: str
    values: Mapping[str, float]


class Source(Protocol):
    def region(self) -> Region: ...


class Screen(Protocol):
    def rank(self, region: Region) -> Mapping[str, float]: ...


class Method(Protocol):
    def propose(self, block: Block) -> Proposal: ...


class Eval(Protocol):
    def score(self, block: Block, proposal: Proposal) -> Metrics: ...
```

- [ ] **Step 5: Run tests to verify they pass.**

Run: `pixi run pytest tests/test_contracts.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Verify + commit.**

```bash
pixi run check
git add src/reblock tests/test_contracts.py tests/__init__.py tests/data tests/derive tests/methods
git commit -m "feat: add reblock core contracts"
```

---

### Task 3: ShapefileSource

**Files:**
- Create: `src/reblock/data/shapefile.py`
- Test: `tests/data/test_shapefile_source.py`

**Interfaces:**
- Produces: `ShapefileSource(path, region_id="region")` implementing `Source`; `.region() -> Region` whose `blocks` is one `Block` per connected component of edge-adjacent parcels, reprojected to `estimate_utm_crs()`; `Block.streets` = component boundary lines; `block_id = f"{region_id}_{k}"`.

- [ ] **Step 1: Write the failing test** `tests/data/test_shapefile_source.py`:

```python
from pathlib import Path

from reblock.contracts import Block
from reblock.data.shapefile import ShapefileSource

PHULE = Path(__file__).resolve().parents[2] / "ext" / "topology" / "examples" / "data" / "phule_nagar_v6.shp"


def test_source_yields_metric_blocks() -> None:
    region = ShapefileSource(PHULE, region_id="phule").region()
    blocks = list(region.blocks)
    assert len(blocks) >= 1
    b = blocks[0]
    assert isinstance(b, Block) and b.crs.is_projected
    assert not b.parcels.empty and "parcel_id" in b.parcels.columns
    assert b.boundary.area > 0 and b.block_id.startswith("phule_")
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `pixi run pytest tests/data/test_shapefile_source.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'reblock.data.shapefile'`

- [ ] **Step 3: Write `src/reblock/data/shapefile.py`.**

```python
"""ShapefileSource: read a parcel shapefile into a Region of Blocks (geopandas)."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString

from reblock.contracts import Block, Region


def _components(gdf: gpd.GeoDataFrame) -> list[list[int]]:
    geoms = list(gdf.geometry)
    tree = STRtree(geoms)
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(range(len(geoms)))
    for i, g in enumerate(geoms):
        for j in tree.query(g):
            if i < int(j) and g.touches(geoms[int(j)]):
                graph.add_edge(i, int(j))
    return [sorted(c) for c in nx.connected_components(graph)]


def _boundary_lines(boundary: object) -> list[LineString]:
    if isinstance(boundary, MultiLineString):
        return list(boundary.geoms)
    if isinstance(boundary, LineString):
        return [boundary]
    return []


class ShapefileSource:
    def __init__(self, path: str | Path, region_id: str = "region") -> None:
        self.path = Path(path)
        self.region_id = region_id

    def region(self) -> Region:
        raw = gpd.read_file(self.path)
        raw = raw[raw.geometry.notna() & ~raw.geometry.is_empty]
        utm = raw.estimate_utm_crs()
        raw = raw.to_crs(utm).reset_index(drop=True)

        blocks: list[Block] = []
        for k, idx in enumerate(_components(raw)):
            geoms = list(raw.iloc[idx].geometry)
            parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(geoms)))},
                                       geometry=geoms, crs=utm)
            boundary_poly = parcels.geometry.union_all()
            streets = gpd.GeoDataFrame(geometry=_boundary_lines(boundary_poly.boundary), crs=utm)
            blocks.append(Block(block_id=f"{self.region_id}_{k}", crs=utm,
                                boundary=boundary_poly, parcels=parcels, streets=streets))
        return Region(region_id=self.region_id, crs=utm, blocks=tuple(blocks))
```

- [ ] **Step 4: Run test to verify it passes.**

Run: `pixi run pytest tests/data/test_shapefile_source.py -v`
Expected: PASS

- [ ] **Step 5: Verify + commit.**

```bash
pixi run check
git add src/reblock/data/shapefile.py tests/data/test_shapefile_source.py
git commit -m "feat: add ShapefileSource (parcels -> Region of metric Blocks)"
```

---

### Task 4: `to_parcel_graph` derivation + port oracle

**Files:**
- Create: `src/reblock/derive/parcel_graph.py`
- Test: `tests/derive/test_parcel_graph.py`

**Interfaces:**
- Consumes: `Block`; `topology` (Task 1).
- Produces: `@dataclass PlanarParcelGraph{ graph: MyGraph, origin: tuple[float,float], crs: CRS }` and `to_parcel_graph(block) -> PlanarParcelGraph` (builds a `MyGraph` from `block.parcels`, re-zeroed to `(minx, miny)`).

- [ ] **Step 1: Write the failing tests** `tests/derive/test_parcel_graph.py`:

```python
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.contracts import Block
from reblock.derive.parcel_graph import to_parcel_graph

UTM = CRS.from_epsg(32643)


def _two_parcels() -> Block:
    polys = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
             Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])]
    parcels = gpd.GeoDataFrame({"parcel_id": [0, 1]}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])], crs=UTM)
    return Block(block_id="t", crs=UTM, boundary=parcels.geometry.union_all(),
                 parcels=parcels, streets=streets)


def test_derivation_builds_planar_graph() -> None:
    ppg = to_parcel_graph(_two_parcels())
    assert len(ppg.graph.inner_facelist) == 2
    assert ppg.graph.G.number_of_nodes() == 6   # two squares sharing the x=1 edge
    assert ppg.crs == UTM
```

- [ ] **Step 2: Run tests to verify they fail.**

Run: `pixi run pytest tests/derive/test_parcel_graph.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'reblock.derive.parcel_graph'`

- [ ] **Step 3: Write `src/reblock/derive/parcel_graph.py`.**

```python
"""Derivation: Block -> topology's planar parcel graph (a MyGraph view)."""
from __future__ import annotations

from dataclasses import dataclass

from pyproj import CRS
from topology import MyEdge, MyFace, MyGraph, MyNode, graphFromMyFaces

from reblock.contracts import Block


@dataclass(frozen=True)
class PlanarParcelGraph:
    graph: MyGraph
    origin: tuple[float, float]
    crs: CRS


def to_parcel_graph(block: Block) -> PlanarParcelGraph:
    minx, miny, _, _ = block.parcels.total_bounds
    origin = (float(minx), float(miny))

    faces: list[MyFace] = []
    for geom in block.parcels.geometry:
        ring = list(geom.exterior.coords)[:-1]
        nodes = [MyNode((x - origin[0], y - origin[1])) for x, y in ring]
        edges = [MyEdge((nodes[i], nodes[(i + 1) % len(nodes)])) for i in range(len(nodes))]
        faces.append(MyFace(edges))

    return PlanarParcelGraph(graph=graphFromMyFaces(faces), origin=origin, crs=block.crs)
```

- [ ] **Step 4: Run tests to verify they pass.**

Run: `pixi run pytest tests/derive/test_parcel_graph.py -v`
Expected: PASS. If node/face counts differ, inspect `graphFromMyFaces` at `ext/topology/topology/graph/my_graph_helpers.py` and set the expected numbers to the geometry — do not bend the derivation to hit a number.

- [ ] **Step 5: Add the port-fidelity oracle test** (append). Our Block→graph must match topology's native shapefile loader on Phule Nagar by inner-face count — deterministic, independent of the stochastic road-builder.

```python
from pathlib import Path

PHULE = Path(__file__).resolve().parents[2] / "ext" / "topology" / "examples" / "data" / "phule_nagar_v6"


def test_matches_topology_native_facecount() -> None:
    from topology import import_and_setup

    from reblock.data.shapefile import ShapefileSource

    native = import_and_setup(str(PHULE), threshold=0.5, byblock=False, name="phule")
    native_faces = len(native.inner_facelist)

    region = ShapefileSource(PHULE.with_suffix(".shp"), region_id="phule").region()
    ours = sum(len(to_parcel_graph(b).graph.inner_facelist) for b in region.blocks)
    assert ours == native_faces
```

- [ ] **Step 6: Run the oracle.**

Run: `pixi run pytest tests/derive/test_parcel_graph.py -v`
Expected: PASS. If counts differ, topology's `import_and_setup` runs `clean_up_geometry(threshold, byblock)` (near-coincident node merge); add `block.graph.clean_up_geometry(0.5, byblock=False)` inside `to_parcel_graph` before returning, and document why in a comment.

- [ ] **Step 7: Verify + commit.**

```bash
pixi run check
git add src/reblock/derive/parcel_graph.py tests/derive/test_parcel_graph.py
git commit -m "feat: add Block->PlanarParcelGraph derivation + port oracle"
```

---

### Task 5: TopologyMethod

**Files:**
- Create: `src/reblock/methods/topology.py`
- Test: `tests/methods/test_topology_method.py`

**Interfaces:**
- Produces: `TopologyMethod(alpha=2.0, seed=0)` implementing `Method`; `.propose(block) -> Proposal` with `roads` (new road LineStrings, Block CRS) and `edges` (all edges tagged `road`/`interior`/`barrier`).

- [ ] **Step 1: Write the failing test** `tests/methods/test_topology_method.py`. A 3×3 parcel grid with only the outer boundary as street has an interior centre parcel that topology must connect.

```python
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.contracts import Block
from reblock.methods.topology import TopologyMethod

UTM = CRS.from_epsg(32643)


def _grid(n: int) -> Block:
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(n) for j in range(n)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = parcels.geometry.union_all()
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_proposes_roads_for_interior_parcel() -> None:
    proposal = TopologyMethod().propose(_grid(3))
    assert proposal.method == "topology" and proposal.crs == UTM
    assert proposal.roads is not None and len(proposal.roads) >= 1
    assert proposal.roads.geometry.length.sum() > 0
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `pixi run pytest tests/methods/test_topology_method.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'reblock.methods.topology'`

- [ ] **Step 3: Write `src/reblock/methods/topology.py`.**

```python
"""TopologyMethod: wrap topology's greedy road-builder into a Proposal."""
from __future__ import annotations

import random
from dataclasses import dataclass

import geopandas as gpd
from shapely.geometry import LineString
from topology import MyEdge, build_all_roads

from reblock.contracts import Block, Proposal
from reblock.derive.parcel_graph import to_parcel_graph


def _edge_line(edge: MyEdge, origin: tuple[float, float]) -> LineString:
    a, b = edge.nodes
    return LineString([(a.x + origin[0], a.y + origin[1]),
                       (b.x + origin[0], b.y + origin[1])])


@dataclass
class TopologyMethod:
    alpha: float = 2.0
    seed: int = 0

    def propose(self, block: Block) -> Proposal:
        ppg = to_parcel_graph(block)
        graph = ppg.graph
        graph.define_roads()                 # boundary edges = initial streets
        graph.define_interior_parcels()
        initial = {e for e in graph.myedges() if e.road}

        random.seed(self.seed)               # build_all_roads is probabilistic
        build_all_roads(graph, alpha=self.alpha, vquiet=True)

        new_edges = [e for e in graph.myedges() if e.road and e not in initial]
        roads = gpd.GeoDataFrame(geometry=[_edge_line(e, ppg.origin) for e in new_edges],
                                 crs=block.crs)
        all_edges = list(graph.myedges())
        edges = gpd.GeoDataFrame(
            {"road": [e.road for e in all_edges],
             "interior": [e.interior for e in all_edges],
             "barrier": [e.barrier for e in all_edges]},
            geometry=[_edge_line(e, ppg.origin) for e in all_edges], crs=block.crs)
        return Proposal(block_id=block.block_id, crs=block.crs, roads=roads, edges=edges,
                        method="topology", params={"alpha": self.alpha, "seed": self.seed})
```

- [ ] **Step 4: Run test to verify it passes.**

Run: `pixi run pytest tests/methods/test_topology_method.py -v`
Expected: PASS

- [ ] **Step 5: Add the "interior parcels resolved" invariant** (append):

```python
def test_all_interior_parcels_connected() -> None:
    import random

    from topology import build_all_roads

    from reblock.derive.parcel_graph import to_parcel_graph
    ppg = to_parcel_graph(_grid(3))
    ppg.graph.define_roads()
    ppg.graph.define_interior_parcels()
    random.seed(0)
    build_all_roads(ppg.graph, alpha=2.0, vquiet=True)
    ppg.graph.define_interior_parcels()
    assert len(ppg.graph.interior_parcels) == 0
```

- [ ] **Step 6: Run + verify + commit.**

```bash
pixi run pytest tests/methods/test_topology_method.py -v && pixi run check
git add src/reblock/methods/topology.py tests/methods/test_topology_method.py
git commit -m "feat: add TopologyMethod (parcels -> Proposal of new roads)"
```

---

### Task 6: KComplexityEval (wraps topology's `k_complexity`)

**Files:**
- Create: `src/reblock/eval/kcomplexity.py`
- Test: `tests/eval/test_kcomplexity.py` (+ `tests/eval/__init__.py`)

**Interfaces:**
- Consumes: `Block`, `Proposal`, `Metrics`; `to_parcel_graph`; `topology.k_complexity`.
- Produces: `KComplexityEval` implementing `Eval`; `.score(block, proposal) -> Metrics` with `values` keys `k_before`, `k_after`, `delta_k`, `added_road_length_m`. k is computed on the parcel graph: `graph.define_roads()` marks the boundary (Slice 1: `Block.streets` == boundary) → `k_before`; additionally marking the `Proposal.roads` edges (exact endpoint match) → `k_after`.

- [ ] **Step 1: Create `tests/eval/__init__.py` and write the failing test** `tests/eval/test_kcomplexity.py`. A 3×3 grid's centre parcel is enclosed → `k_before = 2`; an interior road reaching the centre drops it to `k_after = 1`. (Geometry and values verified against topology's weak-dual `k_complexity` in Task 1 — a 1-wide strip is degenerate for this metric and must NOT be used.)

```python
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.contracts import Block, Proposal
from reblock.eval.kcomplexity import KComplexityEval

UTM = CRS.from_epsg(32643)


def _grid_block(n: int) -> Block:
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(n) for j in range(n)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = parcels.geometry.union_all()
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_delta_k_from_interior_connector() -> None:
    # 3x3 grid: centre parcel enclosed -> k_before = 2. An interior road from
    # boundary node (1,0) up to the centre's corner (1,1) reaches the centre
    # -> k_after = 1. Values verified in Task 1's k_complexity road-sensitivity.
    block = _grid_block(3)
    connector = gpd.GeoDataFrame(geometry=[LineString([(1, 0), (1, 1)])], crs=UTM)
    proposal = Proposal(block_id="g", crs=UTM, roads=connector, method="topology")

    v = KComplexityEval().score(block, proposal).values
    assert v["k_before"] == 2
    assert v["k_after"] == 1
    assert v["delta_k"] == 1
    assert v["added_road_length_m"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `pixi run pytest tests/eval/test_kcomplexity.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'reblock.eval.kcomplexity'`

- [ ] **Step 3: Write `src/reblock/eval/kcomplexity.py`.** Roads are matched to graph edges by exact (origin-shifted, 2-dp-rounded) endpoint pairs — proposal roads *are* graph edges, so matching is exact.

```python
"""KComplexityEval: Δ of topology's k-complexity from inserting proposed roads."""
from __future__ import annotations

from geopandas import GeoDataFrame
from shapely.geometry import LineString
from topology import k_complexity

from reblock.contracts import Block, Metrics, Proposal
from reblock.derive.parcel_graph import to_parcel_graph


def _endpoint_keys(lines: GeoDataFrame, origin: tuple[float, float]) -> set[frozenset[tuple[float, float]]]:
    keys: set[frozenset[tuple[float, float]]] = set()
    for geom in lines.geometry:
        if isinstance(geom, LineString):
            pts = [(round(x - origin[0], 2), round(y - origin[1], 2)) for x, y in geom.coords]
            for a, b in zip(pts, pts[1:]):
                keys.add(frozenset((a, b)))
    return keys


def _k(block: Block, extra_roads: GeoDataFrame | None) -> int:
    # Slice 1: Block.streets == the block boundary, so topology's native
    # define_roads() (outer-face detection) marks the initial streets robustly.
    # Proposed interior roads are 2-point method edges matched by exact endpoints.
    ppg = to_parcel_graph(block)
    ppg.graph.define_roads()
    if extra_roads is not None and not extra_roads.empty:
        keys = _endpoint_keys(extra_roads, ppg.origin)
        for edge in ppg.graph.myedges():
            a, b = edge.nodes
            if frozenset(((a.x, a.y), (b.x, b.y))) in keys:
                edge.road = True
    return k_complexity(ppg.graph)


class KComplexityEval:
    def score(self, block: Block, proposal: Proposal) -> Metrics:
        k_before = _k(block, None)
        k_after = _k(block, proposal.roads)
        added = (float(proposal.roads.geometry.length.sum())
                 if proposal.roads is not None and not proposal.roads.empty else 0.0)
        return Metrics(block_id=block.block_id, method=proposal.method, eval="kcomplexity",
                       values={"k_before": float(k_before), "k_after": float(k_after),
                               "delta_k": float(k_before - k_after),
                               "added_road_length_m": added})
```

- [ ] **Step 4: Run test to verify it passes.**

Run: `pixi run pytest tests/eval/test_kcomplexity.py -v`
Expected: PASS (`k_before`=2, `k_after`=1). If `k_before` isn't 2, `define_roads()` isn't marking the derived graph's boundary — confirm `to_parcel_graph` yields a traceable planar graph (Task 4), and if needed fall back to marking boundary edges whose midpoint lies on `block.streets`. If `k_after` isn't 1, confirm the connector endpoints `(1,0)-(1,1)` coincide with graph nodes (they do for unit parcels) and that Task 1's `k_complexity` road-sensitivity holds.

- [ ] **Step 5: Verify + commit.**

```bash
pixi run check
git add src/reblock/eval tests/eval
git commit -m "feat: add KComplexityEval wrapping topology's k_complexity"
```

---

### Task 7: Hydra run + end-to-end

**Files:**
- Create: `src/reblock/run.py`
- Test: `tests/test_run.py`
- Modify: `pyproject.toml` (`run` pixi task)

**Interfaces:**
- Produces: `RunConfig`, `run(cfg) -> list[Metrics]`, and a Hydra `main()` printing per-block metrics.

- [ ] **Step 1: Write the failing test** `tests/test_run.py`:

```python
from pathlib import Path

from reblock.run import RunConfig, run

PHULE = str(Path(__file__).resolve().parents[1] / "ext" / "topology" / "examples" / "data" / "phule_nagar_v6.shp")


def test_end_to_end_phule() -> None:
    results = run(RunConfig(shapefile=PHULE, region_id="phule", alpha=2.0, seed=0, max_blocks=1))
    assert len(results) == 1
    v = results[0].values
    assert v["k_after"] <= v["k_before"] and v["delta_k"] >= 0
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `pixi run pytest tests/test_run.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'reblock.run'`

- [ ] **Step 3: Write `src/reblock/run.py`.**

```python
"""Hydra entrypoint: data=phule method=topology eval=kcomplexity."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

import hydra
from hydra.core.config_store import ConfigStore

from reblock.contracts import Metrics
from reblock.data.shapefile import ShapefileSource
from reblock.eval.kcomplexity import KComplexityEval
from reblock.methods.topology import TopologyMethod


@dataclass
class RunConfig:
    shapefile: str = "???"
    region_id: str = "phule"
    alpha: float = 2.0
    seed: int = 0
    max_blocks: int = 1


ConfigStore.instance().store(name="run", node=RunConfig)


def run(cfg: RunConfig) -> list[Metrics]:
    source = ShapefileSource(cfg.shapefile, region_id=cfg.region_id)
    method = TopologyMethod(alpha=cfg.alpha, seed=cfg.seed)
    evaluator = KComplexityEval()
    region = source.region()
    return [evaluator.score(b, method.propose(b))
            for b in islice(region.blocks, cfg.max_blocks)]


@hydra.main(version_base=None, config_name="run")
def main(cfg: RunConfig) -> None:
    for m in run(cfg):
        print(m.block_id, dict(m.values))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes.**

Run: `pixi run pytest tests/test_run.py -v`
Expected: PASS

- [ ] **Step 5: Add the `run` pixi task** to `pyproject.toml` `[tool.pixi.tasks]`:

```toml
run = "python -m reblock.run"
```

- [ ] **Step 6: Smoke-run the real entrypoint.**

Run: `pixi run run shapefile=ext/topology/examples/data/phule_nagar_v6.shp max_blocks=1`
Expected: one line like `phule_0 {'k_before': ..., 'k_after': ..., 'delta_k': ..., 'added_road_length_m': ...}`, `k_after <= k_before`.

- [ ] **Step 7: Full check + commit.**

```bash
pixi run check
git add src/reblock/run.py tests/test_run.py pyproject.toml
git commit -m "feat: add Hydra run entrypoint + end-to-end Phule Nagar pipeline"
```

---

## Self-review notes

**Spec coverage.** Task 1 ↔ decision 10 (owned/typed topology, `k_complexity` exposed via `stacked_duals`) + decision 11; Task 2 ↔ contracts; Task 3 ↔ Slice-1 data component + decision 9 (local UTM); Task 4 ↔ derivation (decision 6) + the graph-construction port oracle; Task 5 ↔ method (`Block.streets`→initial roads, decision 10; standalone I/O untouched); Task 6 ↔ eval definition (topology-native k, road-relative, Block CRS); Task 7 ↔ `reblock.run`.

**Deferred (spec "out of scope"):** kblock source, population/displacement, water, screening, tessellation, `pandera`, batch/scale, result persistence. `Region` amenity layers, `Screen`, and `Proposal` water layers exist in contracts but are unused in Slice 1.

**Type consistency:** `Block(block_id, crs, boundary, parcels, streets, …)`, `Proposal(block_id, crs, roads, edges, method, params)`, `Metrics(block_id, method, eval, values)`, `PlanarParcelGraph(graph, origin, crs)`, `to_parcel_graph`, `TopologyMethod(alpha, seed).propose`, `KComplexityEval().score`, `ShapefileSource(path, region_id).region()` — consistent across tasks. topology public names (`MyGraph`, `MyNode`, `MyEdge`, `MyFace`, `graphFromMyFaces`, `build_all_roads`, `import_and_setup`, `k_complexity`, `define_roads_on`) all come from Task 1's `topology/__init__.py`.

**Calibration risks to watch (both have explicit fallback steps):** (1) Task 1 Step 8 — the exact integer convention of `k_complexity` is pinned by the strip→k=N test, not assumed. (2) Task 4 Step 6 — node de-dup / `clean_up_geometry` may be needed for the oracle face-count to match.
```