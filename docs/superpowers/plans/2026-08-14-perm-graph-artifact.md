# Permeability Graph Artifact Type Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive the permeability graph's drawable form once in Python and draw it to PNG, so the Permeability methodology page gets its first figure and piece C's widget inherits a static fallback that already exists.

**Architecture:** `permeability.py` exposes its Laplacian assembly once as `solve_egress`; a new matplotlib-free `perm_graph.py` turns that solution into a flat, serialisable `GraphFigure` (adding per-edge current and an `upgraded` mask); `render.py` draws a `GraphFigure` with edge width encoding either conductance or current; a new `scripts/gen_perm_graph.py` writes four PNGs plus a JSON of their numbers into `examples/perm-graph/`; `gen_site_pages.py` renders them into the page through a marker so no number is typed into prose.

**Tech Stack:** Python 3.11, numpy, scipy.sparse, geopandas/shapely, matplotlib (Agg), pytest, mypy --strict, ruff, Hydra/OmegaConf, pixi.

**Spec:** `docs/superpowers/specs/2026-08-14-perm-graph-artifact-design.md`

## Global Constraints

- **Run everything through pixi.** `pixi run test`, `pixi run typecheck`, `pixi run lint`. A bare `pytest` may work; `python -m scripts.<name>` outside pixi will not (pythonpath is configured for pytest only — use `pixi run python -m scripts.<name>`).
- **`src/` is `mypy --strict`.** Every new function in `src/reblock/` needs full annotations. `scripts/gen_perm_graph.py` is NOT in `[tool.mypy] files`, so do not add it there and do not expect strict coverage of it.
- **`perm_graph.py` must not import matplotlib**, directly or transitively. Piece C serialises it from `gen_site_pages.py` and piece F guards the Pyodide import closure. Allowed: numpy, scipy, geopandas, shapely, and `reblock.{contracts,mesh,permeability}`.
- **`solve_egress` is a pure extraction.** Every existing permeability number must stay **bit-identical**. If any test in `tests/test_permeability.py` or `tests/test_permeability_width.py` moves a digit, the extraction is wrong — do not re-baseline.
- **Palette constants come from `render.py`** (`_PERM_CMAP`, `_ROAD_COLOR`, `_BOUNDARY_COLOR`, `_CONTEXT_OUTLINE`). Introduce no new colours.
- **No unreachable guards.** `Block.__post_init__` already rejects empty parcels, so no `n == 0` branch. A guard whose branch cannot be reached is a silencer, not a defence.
- **No `getattr` with a name you knew when you wrote the line.** The layer set (`"conductance"`, `"current"`) is closed at authoring time, so it is spelled once as `GRAPH_LAYERS` (Task 2) — a table of accessors that name the fields — and every consumer indexes that. `getattr(figure, layer)` would make a renamed field a silent blank instead of a type error.
- **`scripts/gen_site_pages.py` is stdlib-only and MUST NOT import `reblock`.** CI builds the site with only `mkdocs-material` installed (see `_load_friendly_method_name`'s docstring, line 48). Task 6 reads `perm_graph.json`; it never imports `perm_graph`.
- **Out of scope, do not "helpfully" wire up:** `scripts/compare_budgets.py` (the per-method render grid stays as it is), `src/reblock/animate.py` (no graph GIFs), and `permeability.parcel_potentials` (a duplicate solve to remove later, not now).
- **Every guard test must be shown to fail.** For each fault-injection row in Task 3, break the implementation, run the test, paste the failure into the task report, then restore. A test never observed failing does not count as a guard.
- **Pinned block:** `ZAF.9.3.1_1_40972` (from `conf/example/method_comparison.yaml`). **Method:** `clearance`. **Prefix rule:** Lens B, `prefix_to_permeability` at `matched_permeability` from `conf/permeability.yaml`.

---

### Task 1: Expose the egress assembly once (`solve_egress`)

The figure needs per-edge final conductances; `egress_power` returns only `(P, v)`. Extract the one assembly rather than re-deriving a second Laplacian (which `permeability.py`'s module docstring forbids) or building the mesh twice to read back what the first pass computed.

**Files:**
- Modify: `src/reblock/permeability.py:275-329` (`egress_power`)
- Test: `tests/test_permeability.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `EgressSolution(p: float, potential: NDArray[np.float64], mesh: Mesh, conductance: NDArray[np.float64])` and `solve_egress(block, roads, params=PermeabilityParams(), *, adj=None, radii=None) -> EgressSolution`, both public in `reblock.permeability`. `egress_power` keeps its exact present signature and return type `tuple[float, NDArray[np.float64]]`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_permeability.py` (the file's `_grid_block` / `_roads` helpers already exist at the top; reuse them):

```python
def test_solve_egress_agrees_with_egress_power():
    """solve_egress is the one assembly; egress_power is its wrapper, so they cannot disagree."""
    b = _grid_block()
    r = _roads([LineString([(2, 0), (2, 4)])])
    sol = solve_egress(b, r)
    p, v = egress_power(b, r)
    assert sol.p == p
    assert np.array_equal(sol.potential, v)


def test_solve_egress_conductance_covers_the_whole_mesh():
    """The returned conductance is one value per mesh edge, at or above the footpath term
    everywhere -- roads enter only through a max(), so no edge can come back lower."""
    b = _grid_block(15, cell=10.0)
    r = _roads([LineString([(15, 0), (15, 135)])])
    sol = solve_egress(b, r)
    assert len(sol.conductance) == len(sol.mesh.rows) == len(sol.mesh.footpath_g)
    assert np.all(sol.conductance >= sol.mesh.footpath_g)
    assert np.any(sol.conductance > sol.mesh.footpath_g)   # that road really does upgrade edges
```

Add `solve_egress` to the existing `from reblock.permeability import (...)` block in that file.

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_permeability.py -k solve_egress -v`
Expected: FAIL — `ImportError: cannot import name 'solve_egress' from 'reblock.permeability'`

- [ ] **Step 3: Write minimal implementation**

In `src/reblock/permeability.py`, add the dataclass above `egress_power` and rewrite `egress_power` as a wrapper. Import `Mesh` from `reblock.mesh` (the module already imports `footpath_mesh`, `_footpath_conductance` and `parcel_radii` from there).

```python
@dataclass(frozen=True)
class EgressSolution:
    """One grounded egress solve, with the assembly it was built from.

    `egress_power` returns only (P, v) because that is all the metric needs. The graph FIGURE
    (`reblock.perm_graph`) also needs the per-edge conductances the Laplacian was assembled from --
    so the assembly is exposed here once, rather than re-derived (which this module's docstring
    forbids) or recomputed alongside a second mesh build that would be free to disagree with the
    first.

    `conductance` is one value per `mesh` edge, after road upgrades: `edge_conductances`' output.
    """
    p: float
    potential: NDArray[np.float64]
    mesh: Mesh
    conductance: NDArray[np.float64]


def solve_egress(
    block: Block,
    roads: GeoDataFrame | None,
    params: PermeabilityParams = PermeabilityParams(),  # noqa: B008 (frozen, immutable)
    *,
    adj: list[set[int]] | None = None,
    radii: NDArray[np.float64] | None = None,
) -> EgressSolution:
    """The one grounded-Laplacian assembly and sparse solve. See `egress_power` for the model, and
    `EgressSolution` for why the intermediate quantities are returned rather than discarded.

    Degenerate cases keep `egress_power`'s contract exactly: an ungrounded block (no parcel within
    STREET_TOL of a street) or a non-finite solve yields `p = inf` and zero potentials, still paired
    with the mesh and conductances that were built -- a caller that wants to REFUSE those (the figure
    generator does) checks `p` rather than being handed a silently-zero field."""
    parcels = block.parcels
    n = len(parcels)

    mesh = footpath_mesh(block, params, adj=adj, radii=radii)
    rows_arr, cols_arr, dist_arr = mesh.rows, mesh.cols, mesh.dist
    conds_arr = edge_conductances(mesh.segments, dist_arr, mesh.footpath_g, roads, params)
    zeros = np.zeros(n, dtype=np.float64)

    if not mesh.ground.any():
        return EgressSolution(float("inf"), zeros, mesh, conds_arr)

    diag = np.zeros(n, dtype=np.float64)
    np.add.at(diag, rows_arr, conds_arr)
    np.add.at(diag, cols_arr, conds_arr)
    diag[mesh.ground] += params.g_street

    # off-diagonal entries: -g_ij at (i,j) and (j,i)
    off_rows = np.concatenate([rows_arr, cols_arr])
    off_cols = np.concatenate([cols_arr, rows_arr])
    off_data = np.concatenate([-conds_arr, -conds_arr])

    all_rows = np.concatenate([off_rows, np.arange(n, dtype=np.int64)])
    all_cols = np.concatenate([off_cols, np.arange(n, dtype=np.int64)])
    all_data = np.concatenate([off_data, diag])

    lap = coo_matrix((all_data, (all_rows, all_cols)), shape=(n, n)).tocsr()
    b = np.ones(n, dtype=np.float64)
    v = spsolve(cast(csr_matrix, lap), b)
    if not np.all(np.isfinite(v)):
        return EgressSolution(float("inf"), zeros, mesh, conds_arr)
    return EgressSolution(float(b @ v), cast(NDArray[np.float64], v), mesh, conds_arr)


def egress_power(
    block: Block,
    roads: GeoDataFrame | None,
    params: PermeabilityParams = PermeabilityParams(),  # noqa: B008 (frozen, immutable)
    *,
    adj: list[set[int]] | None = None,
    radii: NDArray[np.float64] | None = None,
) -> tuple[float, NDArray[np.float64]]:
    """P = b^T L^-1 b ... (KEEP THIS DOCSTRING VERBATIM from the current implementation)"""
    sol = solve_egress(block, roads, params, adj=adj, radii=radii)
    return sol.p, sol.potential
```

Three details that matter, because the original body branched on `dist_arr.size`:

* the old `if dist_arr.size:` guards around `edge_conductances` and the off-diagonal build are gone. They were only avoiding numpy calls on empty arrays; `edge_conductances` already returns early on `segments.size == 0`, and `np.concatenate` / `np.add.at` are correct on empty inputs. Verify by running the whole permeability suite, not by reasoning.
* the old `if n == 0: return inf, zeros(0)` early return is dropped, not moved: `Block.__post_init__` rejects empty parcels, so it was dead. If keeping the diff minimal matters more to the reviewer than deleting dead code, leaving it in `solve_egress` is acceptable — but do not add a *new* one.
* the ungrounded check now happens after `edge_conductances` rather than before it. That costs one STRtree pass on a block that will be refused anyway, and buys a complete `EgressSolution` for every input. Note it in the task report.

- [ ] **Step 4: Run test to verify it passes, and that nothing moved**

Run: `pixi run pytest tests/test_permeability.py tests/test_permeability_width.py -v`
Expected: PASS, all of them — including `test_default_road_one_lane_conductance`'s pinned `20.0`. Any changed value means the extraction is wrong; do not adjust an expected number.

Run: `pixi run test`
Expected: PASS. The whole suite, because `egress_power` has callers across `budget.py`, `compare_budgets.py` and the eval modules.

Run: `pixi run typecheck && pixi run lint`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/reblock/permeability.py tests/test_permeability.py
git commit -m "refactor: expose the egress assembly once as solve_egress

The graph figure needs the per-edge conductances the Laplacian was
assembled from, and egress_power returns only (P, v). Rather than
re-derive an assembly this module's docstring forbids re-deriving, or
build the mesh twice so the second pass can read back what the first
computed, the assembly becomes solve_egress and egress_power becomes its
wrapper. One Laplacian, one spsolve, unchanged numbers."
```

---

### Task 2: `GraphFigure` and `permeability_graph`

**Files:**
- Create: `src/reblock/perm_graph.py`
- Test: `tests/test_perm_graph.py` (created here, extended in Task 3)

**Interfaces:**
- Consumes: `solve_egress`, `EgressSolution` from Task 1.
- Produces: `GraphFigure` (frozen dataclass, fields exactly as below); `permeability_graph(block, roads, params=PermeabilityParams(), *, adj=None, radii=None) -> GraphFigure`; the type alias `GraphLayer = Literal["conductance", "current"]`; and `GRAPH_LAYERS: dict[GraphLayer, Callable[[GraphFigure], NDArray[np.float64]]]`. Task 3 asserts on the arrays; Task 4's `render_graph` takes a `GraphLayer` and indexes `GRAPH_LAYERS`; Task 5's generator builds two figures and normalizes through the same table.

- [ ] **Step 1: Write the failing test**

Create `tests/test_perm_graph.py`:

```python
import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.contracts import Block
from reblock.perm_graph import GraphFigure, permeability_graph
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, with_width

UTM = CRS.from_epsg(32734)


def _grid_block(k: int = 6, cell: float = 10.0, street: bool = True) -> Block:
    """k x k `cell`-sized parcels; the south edge (y=0) is the street unless `street` is False,
    in which case the street sits far away and NO parcel is grounded."""
    polys, ids = [], []
    for r in range(k):
        for c in range(k):
            x0, x1, y0, y1 = c * cell, (c + 1) * cell, r * cell, (r + 1) * cell
            polys.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
            ids.append(r * k + c)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    line = (LineString([(0.0, 0.0), (k * cell, 0.0)]) if street
            else LineString([(0.0, -1e5), (k * cell, -1e5)]))
    streets = gpd.GeoDataFrame(geometry=[line], crs=UTM)
    boundary = Polygon([(0, 0), (k * cell, 0), (k * cell, k * cell), (0, k * cell)])
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _roads(lines: list[LineString]) -> gpd.GeoDataFrame:
    return with_width(gpd.GeoDataFrame(geometry=lines, crs=UTM), DEFAULT_ROAD_WIDTH_M)


SPINE = [LineString([(15.0, 0.0), (15.0, 55.0)])]


def test_shapes_are_consistent():
    fig = permeability_graph(_grid_block(), _roads(SPINE))
    assert isinstance(fig, GraphFigure)
    assert fig.n == 36
    for arr in (fig.cx, fig.cy, fig.potential, fig.ground_g):
        assert arr.shape == (36,)
    m = len(fig.rows)
    for arr in (fig.cols, fig.conductance, fig.footpath_g, fig.upgraded, fig.current):
        assert arr.shape == (m,)
    assert m > 0


def test_upgraded_is_empty_without_roads_and_nonempty_with_a_road():
    """`upgraded` means the road RAISED this edge -- so it is vacuous with no roads, and a spine
    road through the grid must raise at least one edge."""
    assert not permeability_graph(_grid_block(), None).upgraded.any()
    assert permeability_graph(_grid_block(), _roads(SPINE)).upgraded.any()


def test_ground_g_is_g_street_on_street_fronting_parcels_only():
    fig = permeability_graph(_grid_block(), None)
    grounded = fig.ground_g > 0.0
    assert grounded.sum() == 6                      # the south row of a 6x6 grid
    assert np.allclose(fig.ground_g[grounded], 20.0)   # PermeabilityParams.g_street


def test_ungrounded_block_raises():
    """A figure of an ungrounded block would be a picture of no flow anywhere -- absent is fine,
    silently zero is not."""
    with pytest.raises(ValueError, match="ungrounded"):
        permeability_graph(_grid_block(street=False), None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_perm_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reblock.perm_graph'`

- [ ] **Step 3: Write minimal implementation**

Create `src/reblock/perm_graph.py`:

```python
"""The permeability graph in DRAWABLE form: one derivation, two renderings.

`permeability_graph` turns a block plus a road set into a flat, serialisable description of the
egress graph -- node positions and potentials, edge endpoints, conductances, which edges a road
raised, and the current flowing along each. `reblock.render.render_graph` draws it to PNG; the site
generator serialises the same structure to JSON for the browser widget. One definition of what the
graph IS, so the picture and the widget cannot disagree about it.

DELIBERATELY FREE OF MATPLOTLIB. The JSON path must not drag a plotting stack behind it, and the
browser explorer boots this module's import closure under Pyodide -- which is why the drawing lives
in `render.py` and only the derivation lives here.

Nothing here re-derives the metric. `solve_egress` performs the one Laplacian assembly and solve;
this module adds only what a picture needs and the metric does not: the per-edge current, and the
mask of edges a road actually raised.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from geopandas import GeoDataFrame
from numpy.typing import NDArray

from reblock.contracts import Block
from reblock.permeability import PermeabilityParams, solve_egress


@dataclass(frozen=True)
class GraphFigure:
    """Everything needed to DRAW the egress graph, and nothing that will not serialise to JSON.

    Node arrays are in `block.parcels` order (length `n`); edge arrays are in `Mesh` order
    (length `m`, each undirected pair stored once with `rows[k] < cols[k]`).

    `ground_g` carries the per-node conductance to ground rather than a bool, for two reasons: the
    energy identity in `tests/test_perm_graph.py` needs the value, and a halo can be drawn weighted
    by the conductance actually present. The bool is `ground_g > 0`, so nothing is lost.

    `upgraded` is stored rather than left to each renderer to recompute from
    `conductance > footpath_g`: computing it once, here, is what stops the PNG and the browser
    widget forming two opinions about which edges the road raised. Note what it claims -- an edge a
    road COVERS whose road term comes in below the footpath keeps the footpath under `max()` and
    reads as not upgraded. That is the honest caption: the drawing shows the edges the road actually
    raised. Possible in principle; never observed in 19,023 mesh edges over 60 real blocks
    (notes/2026-07-31-width-is-per-road.md).

    `current` is signed `rows -> cols`: positive means flow from `rows[k]` toward `cols[k]`.
    """
    cx: NDArray[np.float64]
    cy: NDArray[np.float64]
    potential: NDArray[np.float64]
    ground_g: NDArray[np.float64]
    rows: NDArray[np.int64]
    cols: NDArray[np.int64]
    conductance: NDArray[np.float64]
    footpath_g: NDArray[np.float64]
    upgraded: NDArray[np.bool_]
    current: NDArray[np.float64]
    n: int
    p: float


def permeability_graph(
    block: Block,
    roads: GeoDataFrame | None,
    params: PermeabilityParams = PermeabilityParams(),  # noqa: B008 (frozen, immutable)
    *,
    adj: list[set[int]] | None = None,
    radii: NDArray[np.float64] | None = None,
) -> GraphFigure:
    """The drawable form of `block`'s egress graph under `roads`.

    One solve (`solve_egress`), then two derived quantities: `upgraded` and `current`. `adj` and
    `radii` are threaded exactly as `egress_power`/`permeability` accept them, so a region-scale
    caller does not rebuild `parcel_adjacency` per figure.

    Raises `ValueError` for an ungrounded block. `solve_egress` reports those as `p = inf` with zero
    potentials, because a network with no path to ground has no well-defined dissipated power; a
    figure built from those zeros would show no flow anywhere, which is wrong in a way a reader
    cannot see. This is a figure generator, not a batch metric -- there is no aggregate for it to
    keep marching through.
    """
    sol = solve_egress(block, roads, params, adj=adj, radii=radii)
    if not np.isfinite(sol.p):
        raise ValueError(
            f"block {block.block_id!r} is ungrounded (no parcel within STREET_TOL of a street), so "
            f"its egress power is infinite and every potential is zero -- there is no flow to draw")
    mesh = sol.mesh
    ground_g = np.where(mesh.ground, params.g_street, 0.0)
    current = sol.conductance * (sol.potential[mesh.rows] - sol.potential[mesh.cols])
    return GraphFigure(
        cx=mesh.cx, cy=mesh.cy, potential=sol.potential, ground_g=ground_g,
        rows=mesh.rows, cols=mesh.cols,
        conductance=sol.conductance, footpath_g=mesh.footpath_g,
        upgraded=sol.conductance > mesh.footpath_g,
        current=current, n=mesh.n, p=sol.p)


GraphLayer = Literal["conductance", "current"]

GRAPH_LAYERS: dict[GraphLayer, Callable[[GraphFigure], NDArray[np.float64]]] = {
    "conductance": lambda f: f.conductance,
    "current": lambda f: f.current,
}
"""The two quantities an edge's WIDTH can encode, each spelled as a field access.

A table of accessors rather than `getattr(figure, layer)`: the set is closed while this line is being
written, so a renamed `GraphFigure` field must be a type error at one site, not a silently blank
drawing. Lives here, beside the dataclass whose fields it names, so the PNG renderer, the figure
generator and (in piece C) the JSON emitter all read one definition of what a layer means.
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_perm_graph.py -v`
Expected: PASS, all five.

If `test_ground_g_is_g_street_on_street_fronting_parcels_only` fails on the count, print `grounded.sum()` and check the fixture — `STREET_TOL` decides membership and the south row's parcels touch `y=0` exactly. Fix the fixture or the expected count to match the geometry; do not loosen the assertion to `> 0`.

Run: `pixi run typecheck && pixi run lint`
Expected: clean. `np.where(mesh.ground, params.g_street, 0.0)` returns `NDArray[np.float64]`; if mypy disagrees, add an explicit `dtype=np.float64` cast rather than silencing it.

- [ ] **Step 5: Commit**

```bash
git add src/reblock/perm_graph.py tests/test_perm_graph.py
git commit -m "feat: derive the permeability graph's drawable form

GraphFigure is the flat, JSON-serialisable description of the egress
graph -- nodes, potentials, edges, conductances, the mask of edges a
road raised, and per-edge current. One derivation, so the PNG and (in
piece C) the browser widget cannot disagree about what the graph is.
Deliberately matplotlib-free: the JSON path must not drag a plotting
stack behind it, and Pyodide boots this closure."
```

---

### Task 3: The physics guards, with fault injection

The structure is checkable against the solver it came from. This is what makes a second renderer safe to add later.

**Files:**
- Modify: `tests/test_perm_graph.py`

**Interfaces:**
- Consumes: `GraphFigure`, `permeability_graph` from Task 2.
- Produces: nothing consumed downstream.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_perm_graph.py`:

```python
def _node_net_current(fig: GraphFigure) -> np.ndarray:
    """Signed current leaving each node, plus what it sheds to ground. Equals (L v)_i, which is
    b_i = 1 for every node: one unit of escape current injected per parcel."""
    out = np.zeros(fig.n, dtype=np.float64)
    np.add.at(out, fig.rows, fig.current)     # leaves rows[k]
    np.add.at(out, fig.cols, -fig.current)    # arrives at cols[k]
    return out + fig.ground_g * fig.potential


@pytest.mark.parametrize("roads", [None, _roads(SPINE)], ids=["no_roads", "spine"])
def test_energy_identity(roads):
    """Dissipated power recomputed from the DRAWN quantities equals the solver's own P, which for
    b = ones also equals the sum of potentials:

        sum_edges g (dphi)^2 + sum_nodes ground_g phi^2  ==  p  ==  sum(phi)

    because v = L^-1 b makes v^T L v = v^T b. Exact up to solver residual.
    """
    fig = permeability_graph(_grid_block(), roads)
    dphi = fig.potential[fig.rows] - fig.potential[fig.cols]
    drawn = float((fig.conductance * dphi**2).sum() + (fig.ground_g * fig.potential**2).sum())
    assert drawn == pytest.approx(fig.p, rel=1e-9)
    assert float(fig.potential.sum()) == pytest.approx(fig.p, rel=1e-9)


@pytest.mark.parametrize("roads", [None, _roads(SPINE)], ids=["no_roads", "spine"])
def test_per_node_kirchhoff(roads):
    """Every node injects exactly one unit. Catches indexing and sign errors the aggregate energy
    identity can absorb -- a globally-flipped current still squares to the same power."""
    fig = permeability_graph(_grid_block(), roads)
    assert np.allclose(_node_net_current(fig), 1.0, rtol=1e-9, atol=1e-9)


def test_current_is_zero_when_every_parcel_fronts_the_street():
    """A sanity anchor for the sign convention: with every parcel grounded and the fabric
    symmetric, no unit has any reason to cross the mesh -- each leaves through its own ground edge,
    so the potentials are equal and every dphi is 0.

    Built with the full boundary ring as street, not the shared south-edge fixture, and asserted to
    have edges: on a 1x1 block `current` is empty and `allclose` would pass vacuously.
    """
    base = _grid_block(2, 10.0)
    ring = gpd.GeoDataFrame(geometry=[base.boundary.boundary], crs=UTM)
    block = Block(block_id="ring", crs=UTM, boundary=base.boundary,
                  parcels=base.parcels, streets=ring)

    fig = permeability_graph(block, None)
    assert len(fig.rows) > 0                       # not vacuous
    assert np.all(fig.ground_g > 0.0)              # every parcel fronts the ring
    assert np.allclose(fig.current, 0.0)
```

- [ ] **Step 2: Run to verify they fail for the right reason, then pass**

Run: `pixi run pytest tests/test_perm_graph.py -v`
Expected: PASS immediately — Task 2's implementation already satisfies these. That is exactly why Step 3 exists: a test that has never been observed failing is not yet a guard.

- [ ] **Step 3: Fault-inject each guard and record the failure**

For each row, make the edit in `src/reblock/perm_graph.py`, run the command, **paste the failing output into the task report**, then `git checkout src/reblock/perm_graph.py` to restore.

| # | break | edit | must fail |
|---|---|---|---|
| 1 | flipped current sign | `current = sol.conductance * (sol.potential[mesh.cols] - sol.potential[mesh.rows])` | `test_per_node_kirchhoff` (net reads −1) |
| 2 | current from the footpath term | `current = mesh.footpath_g * (sol.potential[mesh.rows] - sol.potential[mesh.cols])` | `test_per_node_kirchhoff[spine]`, `test_energy_identity` unaffected — note in the report which of the two catches it |
| 3 | dropped ground terms | `ground_g = np.zeros(mesh.n, dtype=np.float64)` | `test_energy_identity` and `test_per_node_kirchhoff` (grounded nodes) |
| 4 | transposed endpoints | swap `rows=mesh.cols, cols=mesh.rows` in the `GraphFigure(...)` call | `test_per_node_kirchhoff` |

Run for each: `pixi run pytest tests/test_perm_graph.py -v`

If a row does **not** fail, the guard is insufficient — say so in the report and strengthen the test rather than deleting the row. Row 2 is the one most likely to slip past the energy identity (which recomputes power from `conductance`, not from `current`); if Kirchhoff also misses it on the `spine` fixture, pick a road set with more upgraded edges.

- [ ] **Step 4: Confirm restored and green**

Run: `git diff --stat src/reblock/perm_graph.py`
Expected: empty — every injection reverted.

Run: `pixi run pytest tests/test_perm_graph.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_perm_graph.py
git commit -m "test: guard the drawn graph with the solver's own physics

The energy identity (power recomputed from the drawn conductances and
potentials equals the solver's P, and equals the sum of potentials) plus
per-node Kirchhoff (signed incident current plus the ground term is
exactly the one unit injected). Both were fault-injected first -- a
flipped sign, current taken from the footpath term, dropped ground
terms, transposed endpoints -- and each break observed failing before
the tests were kept."
```

---

### Task 4: `render_graph`

**Files:**
- Modify: `src/reblock/render.py` (extract a boundary/streets helper from `_draw_heatmap:150-158`; add `render_graph` after `render_after`)
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `GraphFigure` from Task 2.
- Produces: `render_graph(figure, block, *, layer, vmax, width_norm, frame=None, roads=None) -> Figure` where `layer: Literal["conductance", "current"]`. Task 5 calls it four times.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_render.py`:

```python
def test_render_graph_returns_figure_with_axes() -> None:
    from reblock.perm_graph import permeability_graph
    from reblock.render import render_graph

    block = _grid_block(6)
    fig_data = permeability_graph(block, None)
    fig = render_graph(fig_data, block, layer="conductance",
                       vmax=float(fig_data.potential.max()),
                       width_norm=float(fig_data.conductance.max()))

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_render_graph_draws_upgraded_edges_in_the_road_colour() -> None:
    """Road-raised edges are drawn in the road blue, and only when a road actually raised one.

    Asserted on COLOUR, not on a count of collections: on this unit-cell fixture a 7 m road
    blankets the whole mesh, so the grey collection can legitimately be absent and a count would
    read the wrong way round.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba

    from reblock.perm_graph import permeability_graph
    from reblock.render import _ROAD_COLOR, render_graph

    block = _grid_block(6)
    roads = with_width(
        gpd.GeoDataFrame(geometry=[LineString([(3.0, 0.0), (3.0, 5.0)])], crs=UTM),
        DEFAULT_ROAD_WIDTH_M)

    plain = permeability_graph(block, None)
    roaded = permeability_graph(block, roads)
    assert roaded.upgraded.any(), "fixture road must upgrade an edge or the test is vacuous"

    def _edge_colours(f: Figure) -> set[tuple[float, ...]]:
        return {tuple(round(float(v), 4) for v in colour)
                for coll in f.axes[0].collections if isinstance(coll, LineCollection)
                for colour in coll.get_colors()}

    blue = tuple(round(float(v), 4) for v in to_rgba(_ROAD_COLOR))
    a = render_graph(plain, block, layer="current", vmax=1.0, width_norm=1.0)
    b = render_graph(roaded, block, layer="current", vmax=1.0, width_norm=1.0, roads=roads)
    assert blue not in _edge_colours(a)
    assert blue in _edge_colours(b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_render.py -k render_graph -v`
Expected: FAIL — `ImportError: cannot import name 'render_graph' from 'reblock.render'`

- [ ] **Step 3: Write minimal implementation**

First extract the shared drawing. In `src/reblock/render.py`, replace the boundary/streets block inside `_draw_heatmap` (currently lines 150-158, comments included) with a call to a new module-level helper, moving those comments verbatim into it:

```python
def _draw_boundary_and_streets(ax: Axes, block: Block) -> None:
    """The block outline and the EXISTING street network, in `_BOUNDARY_COLOR`.

    Shared by the heatmap and the graph renderers: both need it, and the MultiPolygon reasoning
    below is not worth stating twice.

    A single block (or a gap-free region) is a Polygon -- draw its ring. A gappy multi-block
    region's boundary is a MultiPolygon whose `.boundary` is one ring PER member: redundant with
    the inter-block streets drawn below, so skip it there (drawing it added a misleading
    convex-hull-like outline across the empty gaps between members).

    The street network is never skipped. For a single block it is the outer ring; for a region it
    also carries the inter-block streets between members -- existing egress the 'before' access
    depth is measured against, so they must be visible (a parcel next to one is shallow, not deep),
    and for a gappy region this IS the region outline.
    """
    if isinstance(block.boundary, Polygon):
        gpd.GeoSeries([block.boundary], crs=block.crs).boundary.plot(
            ax=ax, color=_BOUNDARY_COLOR, linewidth=1.3)
    if block.streets is not None and not block.streets.empty:
        block.streets.plot(ax=ax, color=_BOUNDARY_COLOR, linewidth=1.3)
```

Add `from matplotlib.axes import Axes` and `from matplotlib.collections import LineCollection` to the imports, plus `import numpy as np` and `from reblock.perm_graph import GRAPH_LAYERS, GraphFigure, GraphLayer`. `Literal` is already imported.

Then append the renderer:

```python
_GRAPH_PARCEL_EDGE = "#cccccc"     # the wireframe parcels recede to; graph is the subject
_EDGE_GREY = "#8c8c8c"
_EDGE_LW_MIN = 0.15                # a hairline, so the mesh stays present rather than vanishing
_EDGE_LW_MAX = 6.0
_GROUND_HALO = "#1a1a1a"
_NODE_RADIUS_FRAC = 0.18           # of the median edge length, so this holds at region scale


def render_graph(
    figure: GraphFigure,
    block: Block,
    *,
    layer: GraphLayer,
    vmax: float,
    width_norm: float,
    frame: BBox | None = None,
    roads: gpd.GeoDataFrame | None = None,
) -> Figure:
    """The egress graph itself: nodes coloured by potential, edges widthed by `layer`.

    `layer` picks what edge width encodes -- `"conductance"` shows the clearance-fraction mesh (and,
    with roads, which edges they raised); `"current"` shows the drainage, `i = g(phi_i - phi_j)`,
    where the tree appears and a road visibly concentrates flow into itself. Both quantities live on
    the same `GraphFigure`; the choice is the caller's, made once where a figure SET is defined
    rather than re-derived per draw.

    `vmax` (node potential) and `width_norm` (the edge quantity) are explicit for the same reason
    `render_before`/`render_after` take an explicit `vmax`: a before/after pair has to share its
    scales or the comparison means nothing.

    Parcels are drawn as a pale wireframe, NOT filled by potential. Filling them would state the
    same quantity twice in two shapes and drown the graph -- this is not the `perm` choropleth with
    dots on top. Node colour uses `_PERM_CMAP` with `vmin=0`, the same scale that choropleth uses,
    so colour means the same thing on both images a reader meets on one page.
    """
    fig, ax = plt.subplots(figsize=(16, 16))

    block.parcels.plot(ax=ax, facecolor="none", edgecolor=_GRAPH_PARCEL_EDGE, linewidth=0.4)

    view = frame if frame is not None else frame_bbox(block.parcels)
    ax.set_xlim(view[0], view[2])
    ax.set_ylim(view[1], view[3])

    # The corridor under the graph, so corridor and upgraded edges read as one fact.
    if roads is not None and not roads.empty:
        gpd.GeoDataFrame(
            geometry=roads.geometry.buffer(roads["width_m"].to_numpy(dtype=float) / 2.0),
            crs=block.crs).plot(ax=ax, color=_ROAD_COLOR, alpha=0.25, zorder=2, linewidth=0)

    _draw_boundary_and_streets(ax, block)

    # Edges: ONE LineCollection per colour, never a per-row plot -- a region's ~60k edges are
    # cheap as two collections and hopeless as 60k artists.
    quantity = GRAPH_LAYERS[layer](figure)
    segs = np.stack([
        np.column_stack([figure.cx[figure.rows], figure.cy[figure.rows]]),
        np.column_stack([figure.cx[figure.cols], figure.cy[figure.cols]]),
    ], axis=1)
    frac = np.clip(np.abs(quantity) / width_norm, 0.0, 1.0) if width_norm > 0 else np.zeros(
        len(quantity))
    widths = _EDGE_LW_MIN + frac * (_EDGE_LW_MAX - _EDGE_LW_MIN)

    plain = ~figure.upgraded
    if plain.any():
        ax.add_collection(LineCollection(
            segs[plain], linewidths=widths[plain], colors=_EDGE_GREY, zorder=3))
    if figure.upgraded.any():
        ax.add_collection(LineCollection(
            segs[figure.upgraded], linewidths=widths[figure.upgraded], colors=_ROAD_COLOR,
            zorder=4))

    # Nodes as geographic-radius disks (the `_point_disks` treatment), so this does not collapse
    # into a screen-size thicket at region scale.
    node_r = _NODE_RADIUS_FRAC * (float(np.median(np.hypot(
        figure.cx[figure.rows] - figure.cx[figure.cols],
        figure.cy[figure.rows] - figure.cy[figure.cols]))) if len(figure.rows) else 1.0)
    nodes = gpd.GeoDataFrame(
        {"phi": figure.potential},
        geometry=gpd.points_from_xy(figure.cx, figure.cy).buffer(node_r), crs=block.crs)

    grounded = figure.ground_g > 0.0
    if grounded.any():
        nodes[grounded].geometry.buffer(node_r * 0.6).plot(
            ax=ax, facecolor="none", edgecolor=_GROUND_HALO, linewidth=1.6, zorder=5)
    nodes.plot(ax=ax, column="phi", cmap=_PERM_CMAP, vmin=0.0, vmax=vmax, linewidth=0, zorder=6)

    ax.set_aspect("equal")
    ax.axis("off")
    return fig
```

`_GRAPH_LAYER_FIELD` maps the literal to a field name once, so a renamed `GraphFigure` field is a single edit and an unknown `layer` raises `KeyError` rather than silently drawing nothing.

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_render.py -v`
Expected: PASS — the new pair plus every pre-existing render test (the `_draw_boundary_and_streets` extraction must not change the heatmap).

Run: `pixi run typecheck && pixi run lint`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/reblock/render.py tests/test_render.py
git commit -m "feat: render the egress graph to PNG

render_graph draws a GraphFigure -- pale parcel wireframe, one
LineCollection per edge colour with width from conductance or current,
road-upgraded edges in the road blue, ground haloed, nodes coloured by
potential on the same scale the perm choropleth uses. Not a third
field= value: those keys are choropleth colourings fed to
parcels.plot(column=...), and a graph is a different drawing. The
boundary/streets block is now shared with the heatmap."
```

---

### Task 5: The generator and its four artifacts

**Files:**
- Create: `scripts/gen_perm_graph.py`
- Create: `examples/perm-graph/README.md` (written by hand; the PNGs and JSON are generated)
- Modify: `scripts/regenerate_examples.sh:72-76` (add an entry beside `gen_screen_bakeoff`)
- Modify: `examples/README.md` (one line of prose, not the flagship table)

**Interfaces:**
- Consumes: `permeability_graph` (Task 2), `render_graph` (Task 4).
- Produces: `examples/perm-graph/{graph_conductance_before,graph_current_before,graph_conductance_after,graph_current_after}.png` and `examples/perm-graph/perm_graph.json` with keys `block_id, method, p_star, permeability_before, permeability_after, road_m, n_parcels, n_edges, n_upgraded`. Task 6 reads that JSON.

- [ ] **Step 1: Write the generator**

Create `scripts/gen_perm_graph.py`:

```python
"""The Permeability page's figure set: the egress graph, drawn four ways on one block.

Two layers (edge width from conductance, then from current) x two states (no roads, then a real
method's roads). The conductance pair teaches the clearance-fraction mesh and what a road does to
it; the current pair teaches drainage, and is the image where adding a road visibly concentrates
flow into the new corridor.

WHY THIS IS NOT PART OF gen_example. It shares that pipeline's block and roads exactly -- same
pinned block, same config, same content-addressed derivation cache -- but iterating on a FIGURE's
design must not cost a ten-method comparison run. This loads one block and one method and takes
seconds.

Run:  pixi run python -m scripts.gen_perm_graph
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from reblock.budget import prefix_to_permeability
from reblock.compare import load_permeability_config
from reblock.contracts import Method, Source
from reblock.derivations import propose
from reblock.perm_graph import GRAPH_LAYERS, permeability_graph
from reblock.permeability import permeability
from reblock.pipeline import build_regions
from reblock.render import frame_bbox, render_graph, save_render

log = logging.getLogger(__name__)

VARIANT = "method_comparison"      # pins ZAF.9.3.1_1_40972; see conf/example/method_comparison.yaml
METHOD = "clearance"
OUT = Path("examples/perm-graph")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    OUT.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config",
                      overrides=[f"+example={VARIANT}", "data=capetown_full"])
    pcfg = load_permeability_config()
    params = pcfg.params

    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    region_builder = cast(RegionBuilder, instantiate(cfg.region_builder))
    groups = [list(g) for g in cfg.block_ids]
    region = build_regions(source, screen, region_builder, groups, int(cfg.max_blocks))[0]
    assert len(region) == 1, "this figure set is single-block by design"
    block = region[0]

    method = cast(Method, instantiate(cfg.all_methods[METHOD]))
    roads = cast(GeoDataFrame, propose(method, block).roads)
    prefix, reached = prefix_to_permeability(block, roads, pcfg.matched_permeability, params)
    if not reached:
        raise SystemExit(
            f"{METHOD} never reached P*={pcfg.matched_permeability} on {block.block_id}; the "
            f"'after' figure would not be the Lens-B prefix the site publishes")
    log.info("block %s: %d parcels, %s prefix %.0f m", block.block_id, len(block.parcels),
             METHOD, float(prefix.geometry.length.sum()))

    before = permeability_graph(block, None, params)
    after = permeability_graph(block, prefix, params)

    # Shared scales. Both figures are derived FIRST so every image can be put on one scale per
    # quantity -- the same discipline compare_budgets applies to vmax and frame. Without it the
    # before/after pair is two pictures at two zoom levels of ink, and teaches nothing.
    frame = frame_bbox(block.parcels)
    vmax = max(float(before.potential.max()), float(after.potential.max()))
    norms = {layer: max(_p99(read(before)), _p99(read(after)))
             for layer, read in GRAPH_LAYERS.items()}

    for state, fig_data, prefix_roads in (("before", before, None), ("after", after, prefix)):
        for layer in GRAPH_LAYERS:
            fig = render_graph(fig_data, block, layer=layer, vmax=vmax,
                               width_norm=norms[layer], frame=frame, roads=prefix_roads)
            path = OUT / f"graph_{layer}_{state}.png"
            save_render(fig, path)
            plt.close(fig)
            log.info("wrote %s", path)

    meta = {
        "block_id": block.block_id,
        "method": METHOD,
        "p_star": pcfg.matched_permeability,
        "permeability_before": 0.0,      # by definition: 1 - P(no roads)/P(no roads)
        "permeability_after": permeability(block, prefix, params),
        "road_m": float(prefix.geometry.length.sum()),
        "n_parcels": int(before.n),
        "n_edges": int(len(before.rows)),
        "n_upgraded": int(after.upgraded.sum()),
    }
    (OUT / "perm_graph.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s", OUT / "perm_graph.json")


def _p99(x: np.ndarray) -> float:
    """A robust maximum for edge-width normalization: one trunk edge orders of magnitude above the
    rest would otherwise flatten the whole mesh to the hairline floor."""
    return float(np.percentile(np.abs(x), 99)) if len(x) else 0.0


if __name__ == "__main__":
    main()
```

Add the imports the body needs that are not listed above (`from geopandas import GeoDataFrame`, `from reblock.contracts import Screen`, `from reblock.region import RegionBuilder`) — match `scripts/gen_example.py:44-51`, which imports the same set for the same calls.

- [ ] **Step 2: Run it**

Run: `pixi run python -m scripts.gen_perm_graph`
Expected: four PNGs and `perm_graph.json` in `examples/perm-graph/`. First run downloads/reads `capetown_full` from `~/.cache/reblock` and may take minutes; a warm derivation cache makes it seconds.

Run: `ls -la examples/perm-graph/ && cat examples/perm-graph/perm_graph.json`
Expected: four files ≈0.5–1.5 MB each, and JSON whose `permeability_after` is at or just above `p_star`, `n_upgraded > 0`.

- [ ] **Step 3: Look at the images before going further**

Open all four. This is the deliverable — a passing script that produces an illegible figure has failed. Check:
- the drainage tree is visible in `graph_current_before.png` (flow crowding toward the street edge);
- `graph_current_after.png` shows current concentrated into the corridor, visibly different from before;
- upgraded edges are blue and sit on the road corridor in `graph_conductance_after.png`;
- the parcel wireframe recedes; nodes and edges are the subject.

If edges are uniformly hairline, `_p99` is being dominated — report the actual quantile spread rather than silently switching to a max. If nodes overlap into a blob, lower `_NODE_RADIUS_FRAC`. Record any constant you changed and why in the task report.

- [ ] **Step 4: Write the README and wire the entry point**

Create `examples/perm-graph/README.md`:

```markdown
# The egress graph

The figure set for the site's [Permeability](../../docs/_partials/permeability.md) section: the graph
the metric is actually computed on, drawn four ways.

|  | no roads | with roads |
|---|---|---|
| width ∝ conductance | ![](graph_conductance_before.png) | ![](graph_conductance_after.png) |
| width ∝ current | ![](graph_current_before.png) | ![](graph_current_after.png) |

Nodes are parcel centroids, coloured by egress potential φ on the same `YlOrRd` scale the `_perm`
heatmaps use — dark means a harder escape. Edge width encodes either the mesh conductance (the
clearance fraction between two footprints) or the current `i = g(φᵢ − φⱼ)` flowing along that edge.
Blue edges are the ones a road raised; haloed nodes front the existing street and drain straight to
ground.

**Provenance.** Block `ZAF.9.3.1_1_40972` — the block `conf/example/method_comparison.yaml` pins,
so this is the same block every method page's before/after uses. The roads are `clearance` at its
Lens-B prefix: the minimal drainage-ordered prefix reaching the matched-permeability standard `P*`
from `conf/permeability.yaml`, which is the same road set `examples/method-comparison/` publishes for
that method. Every number quoted on the site page comes from `perm_graph.json`, written by the
generator.

Not one of the flagships in [`../README.md`](../README.md): those are walkthroughs that reproduce a
result from the CLI, and this is a figure set for one page.

Regenerate: `pixi run python -m scripts.gen_perm_graph`
```

In `scripts/regenerate_examples.sh`, after the `gen_screen_bakeoff` line in the no-args branch:

```bash
  # The graph figure set is likewise its own entry point, and for the same kind of reason: it is a
  # figure set for one site page rather than a graded example, and it deliberately does NOT re-run
  # the ten-method comparison whose block and roads it borrows.
  run pixi run python -m scripts.gen_perm_graph
```

In `examples/README.md`, after the paragraph that ends "...overlaid on every heatmap automatically.", add:

```markdown
Alongside the flagships, [`perm-graph/`](perm-graph/) holds the egress graph itself drawn four ways
— conductance and current, before and after roads — on the same pinned block `method-comparison`
grades. It is a figure set for the site's Permeability section rather than a graded example, which
is why it is not in the table above.
```

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_perm_graph.py scripts/regenerate_examples.sh examples/README.md examples/perm-graph/
git commit -m "feat: generate the egress graph figure set

Four images on the pinned deep block -- conductance and current, before
and after clearance's Lens-B prefix -- plus the JSON of their numbers so
the site page can quote them without anyone typing a figure into prose.
Its own entry point rather than part of gen_example: it borrows that
pipeline's block, config and derivation cache exactly, but iterating on
a figure's design must not cost a ten-method comparison run."
```

---

### Task 6: Put the figures on the page

**Files:**
- Modify: `scripts/gen_site_pages.py` (path constant near line 38; producer near `_bakeoff_figures:228`; `MARKERS` dict near line 940)
- Modify: `docs/_partials/permeability.md` (the placeholder comment at the end)
- Test: `tests/test_gen_site_pages.py` (existing marker tests cover this; add one for the numbers)

**Interfaces:**
- Consumes: `examples/perm-graph/*.png` and `perm_graph.json` from Task 5.
- Produces: the `PERMGRAPHFIGS` marker; nothing downstream consumes it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gen_site_pages.py`:

```python
def test_perm_graph_figures_quote_the_artifact_not_a_typed_number() -> None:
    """The captions' numbers must come from perm_graph.json. A hand-typed figure in the partial is
    exactly the drift the site's truth pass closed -- and 'seven methods' drifted because a count
    did not look like a metric."""
    import json

    from scripts.gen_site_pages import PERMGRAPH, _perm_graph_figures

    meta = json.loads((PERMGRAPH / "perm_graph.json").read_text(encoding="utf-8"))
    html = _perm_graph_figures()

    assert "graph_current_after.png" in html
    assert f"{meta['permeability_after'] * 100:.1f}" in html
    assert meta["block_id"] in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_gen_site_pages.py -k perm_graph -v`
Expected: FAIL — `ImportError: cannot import name 'PERMGRAPH' from 'scripts.gen_site_pages'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/gen_site_pages.py`, beside the other path constants (near line 38):

```python
PERMGRAPH = ROOT / "examples" / "perm-graph"
```

Beside `_bakeoff_figures` (near line 228), the producer:

```python
def _perm_graph_figures() -> str:
    """The Permeability section's 2x2: the egress graph, conductance and current, before and after.

    Every number in these captions is read from `examples/perm-graph/perm_graph.json`, written by
    `scripts/gen_perm_graph.py` beside the images. Typing one here would reintroduce the drift class
    the truth pass closed -- and `_intro.md`'s "seven methods" is the proof that a figure does not
    have to look like a metric to rot.
    """
    meta = json.loads((PERMGRAPH / "perm_graph.json").read_text(encoding="utf-8"))
    block, method = meta["block_id"], friendly_method_name(meta["method"])
    p_after, road_m = meta["permeability_after"] * 100.0, meta["road_m"]
    n_parcels, n_edges = meta["n_parcels"], meta["n_edges"]
    out: list[str] = []
    for layer, what in (("conductance", "conductance"), ("current", "current")):
        for state, roads_phrase in (
            ("before", "with no new roads"),
            ("after", f"under {method}'s roads at the matched-permeability standard"),
        ):
            url = _copy_asset(PERMGRAPH / f"graph_{layer}_{state}.png", "perm-graph")
            if url is None:
                continue
            caption = (
                f"Block {block}: {n_parcels} parcels, {n_edges} footpath-mesh edges, {roads_phrase}."
                f" Edge width is {what}; node colour is egress potential φ on the same scale as the "
                f"heatmaps above, dark meaning a harder escape."
            )
            if state == "after":
                caption += (f" That prefix is {road_m:,.0f} m of road and reaches "
                            f"{p_after:.1f}% permeability.")
            out.append(_figure(url, f"egress graph, {what}, {state} roads", caption))
    return "\n".join(out)
```

`friendly_method_name` is already a module-level name in that file (bound at line 63 from `_load_friendly_method_name()`), and is how every other producer turns `clearance` into its display name — call it directly, add nothing. Note *why* it is loaded that way: this script must stay stdlib-only, because CI builds the site with `reblock` not importable. `json` is stdlib, so the producer is fine; importing `reblock.perm_graph` here would break the site build while every test still passed.

In the `MARKERS` dict (near line 940), add:

```python
    "PERMGRAPHFIGS": _perm_graph_figures,
```

In `docs/_partials/permeability.md`, replace the trailing placeholder comment (the four-line
`<!-- Figure: the parcel graph on a real block ... static image. -->`) with:

```markdown
## The graph, drawn

<!-- PERMGRAPHFIGS -->
```

- [ ] **Step 4: Run the tests and build the page**

Run: `pixi run pytest tests/test_gen_site_pages.py -v`
Expected: PASS — including `test_every_marker_in_a_partial_has_a_producer` and `test_every_producer_is_used_by_a_partial`, both of which now cover the new marker in each direction.

Run: `pixi run python -m scripts.gen_site_pages && sed -n '45,80p' docs/methodology/permeability.md`
Expected: four `<figure>` blocks with `src="../../assets/perm-graph/..."`, and no literal `<!-- PERMGRAPHFIGS -->` left in the output.

Run: `ls docs/assets/perm-graph/`
Expected: the four PNGs (this directory is gitignored — generated, not committed).

Confirm the URL depth is right: `permeability.md` is written with `depth=1, url_depth=2`, so raw-HTML `src` must carry `../../`. A wrong depth 404s every figure silently.

Run: `pixi run test && pixi run typecheck && pixi run lint`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_site_pages.py docs/_partials/permeability.md tests/test_gen_site_pages.py
git commit -m "feat: the Permeability page gets its figure

A PERMGRAPHFIGS marker and producer replace the placeholder comment
piece A left behind, rendering the four graph images as a 2x2 with
captions that read their numbers from perm_graph.json. The existing
marker tests guard both directions, so a half-wired figure block cannot
ship, and no figure is typed into the prose."
```

---

### Task 7: Review the page as a reader

Not a code task. The deliverable is a page, and every previous task's tests can pass while the page reads badly.

**Files:** none necessarily; possibly `docs/_partials/permeability.md` prose, possibly a constant in `render.py`.

- [ ] **Step 1: Build and serve the site**

Run: `pixi run python -m scripts.gen_site_pages && pixi run mkdocs serve`
Open the Permeability page.

- [ ] **Step 2: Check the four claims the page now makes**

- The prose describes the *clearance fraction* and the figure shows it: packed fabric thin, gaps thick.
- The prose says roads enter only through a `max()` and can never lower an edge; the conductance pair shows edges getting thicker, never thinner.
- The prose says `permeability = 1 − P(roads)/P(no roads)`; the caption's `permeability_after` is a number a reader can tie to the frontier page.
- Ground is visible as haloed nodes on the street edge, matching "ground is eliminated, never a graph node you could route through."

If the prose now says something the figure contradicts, fix the prose — the figure is generated from the metric and is the more reliable witness.

- [ ] **Step 3: Decide the conductance-after image's fate**

The spec's open item: `graph_conductance_after.png` differs from `before` only by the blue overlay, since the mesh is road-independent. If the 2×2 reads as three images and a near-duplicate, drop it from `_perm_graph_figures`' loop and leave it as an artifact — it costs nothing to keep on disk and piece C may want it. Record the decision either way.

- [ ] **Step 4: Verify the whole suite and the site build**

Run: `pixi run check`
Expected: lint, typecheck and tests all clean.

Run: `pixi run mkdocs build --strict`
Expected: no warnings. `--strict` catches a broken image path, which is the failure mode a wrong `url_depth` produces.

- [ ] **Step 5: Commit any adjustments and open the PR**

```bash
git add -A
git commit -m "docs: reconcile the permeability prose with its new figures"
git push -u origin perm-graph-artifact
gh pr create --title "Site piece B: the permeability graph as an artifact type" --body "..."
```

Write the PR body from the spec's Why and the §6 exclusions, and state what the reader gets: the Permeability page's first figure, and piece C's static fallback existing before the widget does. Note `gh pr edit --body` silently does nothing on this repo — get the body right in `gh pr create`.

---

## Task summary

| task | deliverable | reviewable on its own |
|---|---|---|
| 1 | `solve_egress` extraction, numbers bit-identical | yes — pure refactor |
| 2 | `GraphFigure` + `permeability_graph` | yes — structure, shapes, refusals |
| 3 | physics guards, each fault-injected | yes — the report is the evidence |
| 4 | `render_graph` | yes — a figure exists |
| 5 | four PNGs + JSON + entry point | yes — the images are the deliverable |
| 6 | the page renders them | yes — the site builds |
| 7 | the page reads correctly | yes — judgement, not tests |

Tasks 1–4 need no data download. Task 5 onward needs `capetown_full` in `~/.cache/reblock`.
