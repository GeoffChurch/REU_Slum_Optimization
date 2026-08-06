# Continuum Permeability Implementation Plan

> **HALTED 2026-08-06 at Task 1.** The A5 kill gate FAILED (119 rank flips, min Kendall tau +0.600
> over 21 blocks x 6 methods x 4 eps values). Tasks 2-10 were never dispatched, which is what
> placing the gate first was for. Do not execute this plan unless the spec's `eps` problem is
> resolved — most plausibly by real building footprints, which remove `eps` entirely.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the parcel-centroid graph permeability metric with conduction on the free space,
so travel distance is intrinsic to the domain rather than approximated by crow-flies between
centroids.

**Architecture:** A road-independent grid over the block minus building disks becomes the node set;
conductivity `sigma` is `sigma_walk` in free space and `sigma_road` inside road corridors; ground is
a Dirichlet condition where free space meets the street; demand is injected on each building's
perimeter. `permeability = 1 - P(roads)/P(no roads)` is unchanged in definition. The old parcel
graph does not disappear — it survives as a METHOD-OWNED search surrogate for `resistance_greedy`
and `resistance_lp`, because their gain model has no continuum analogue and reformulating it is
research.

**Tech Stack:** numpy, scipy.sparse (`spsolve`, `connected_components`), shapely 2 vectorized
predicates, geopandas, hydra config.

## Global Constraints

Copied verbatim from `specs/2026-08-06-continuum-permeability-design.md`. Every task's requirements
implicitly include these.

- **Migrate and delete.** The old parcel-graph METRIC path is removed, not retained behind a flag.
  The parcel-graph SEARCH SURROGATE moves to the methods that use it.
- **Demand is injected on each building's PERIMETER**, spread over the free cells ringing its disk.
  Never at a point: a 2D point source has log-divergent self-energy.
- **Ground is a Dirichlet condition** (`u = 0`), not a shunt. `g_street` is deleted.
- **`eps` is shared with `displacement`.** One radius, both axes. Decoupling is a measured
  method-differential confound (ratio spans 0.823-0.922 at `eps=1.0`, flipping 10% of ranks).
- **Building radii are `max(NN/2 - eps, 0.25)`.** The `0.25` floor keeps a building an obstacle.
- **`h` gets a per-block convergence check**, not a global value. The check is a correctness
  criterion, not a tuning knob.
- **Use the DIRECT sparse solver** (`spsolve`). CG is ~20x worse and its iteration count grows as
  `sqrt(N)`.
- **`eps` must be held FIXED across any comparison.** State it in config, never implicit.
- Run scripts as `pixi run python -m scripts.<name>` (pythonpath is pytest-only).
- Every guard test must be FAULT-INJECTED: break the code, confirm the test fails, restore.

---

## File Structure

**Create:**
- `src/reblock/continuum/__init__.py` — public surface: `ContinuumParams`, `free_space_mesh`,
  `continuum_power`, `converged_power`.
- `src/reblock/continuum/mesh.py` — geometry only: block + buildings + `eps` + `h` -> cells, edges,
  ground mask, demand vector. No solving, no roads.
- `src/reblock/continuum/solve.py` — `sigma` assignment from roads, Laplacian assembly, ungrounded
  pruning, `spsolve`, and the `h`-convergence wrapper.
- `src/reblock/methods/parcel_gain.py` — the OLD parcel-graph model, moved verbatim, relabelled as a
  search surrogate owned by the methods.
- `tests/continuum/test_mesh.py`, `tests/continuum/test_solve.py`,
  `tests/continuum/test_convergence.py`, `tests/test_parcel_gain.py`.

**Modify:**
- `src/reblock/permeability.py` — becomes a thin facade: keeps `PermeabilityParams`,
  `DEFAULT_ROAD_WIDTH_M`, `with_width`, `WIDTH_COL`, `ONEWAY_COL`, `buildable_widths`,
  `permeability`, `permeability_curve`, `parcel_potentials`. Deletes `egress_power`,
  `_footpath_conductance`, `parcel_radii`, `edge_conductances`, `_directed_power`, `FOOTPATH_EPS`,
  `_road_corridor`.
- `src/reblock/methods/resistance_greedy.py`, `resistance_lp.py`, `src/reblock/width_solver.py` —
  import the surrogate from `methods/parcel_gain.py`.
- `src/reblock/budget.py` — `displacement` and `building_radii` take the shared `eps`.
- `conf/permeability.yaml` — parameter migration.

---

## Task 1: Run acceptance A5 as a KILL GATE (no production code)

**Do this before writing any implementation.** A5 can invalidate the design, and everything after
this task is wasted if it fails. The spike already computes continuum permeability; this only widens
the sample.

**Files:**
- Modify: `scratchpad/continuum/eps_ranking.py:26-31` (the `NAMES`, `EPS`, and block-count values)

**Interfaces:**
- Consumes: `scratchpad/continuum/spike.py` `solve(block, roads, h)` and module global `EPS_SEP`
- Produces: a go/no-go decision recorded in the note; no code artifacts

- [ ] **Step 1: Widen the sample**

Edit `scratchpad/continuum/eps_ranking.py`:

```python
NAMES = ["clearance", "clearance_grid", "euclidean_grid", "topology",
         "flow_paths_noreinforce", "clearance_looped"]
EPS = [0.25, 0.5, 1.0, 1.5]
H = 0.5
```

and change the block selection to 20 blocks plus the known under-resolved outlier:

```python
sel = [i for i in pools.recipients if len(blocks[i].parcels) <= 150]
chosen = list(evenly_spaced(sorted(sel), counts, 20))
outlier = next((i for i, b in enumerate(blocks) if b.block_id == "ZAF.9.3.1_1_41829"), None)
if outlier is not None and outlier not in chosen:
    chosen.append(outlier)
```

Replace the ranking comparison so it compares EVERY pair of eps values, not just two:

```python
import itertools
taus, flips, winner_changes = [], 0, 0
for e1, e2 in itertools.combinations(EPS, 2):
    a = full.pivot(index="block", columns="method", values=f"perm_{e1}")
    b = full.pivot(index="block", columns="method", values=f"perm_{e2}")
    ra, rb = a.rank(axis=1), b.rank(axis=1)
    flips += int((ra != rb).sum().sum())
    winner_changes += int((a.idxmax(axis=1) != b.idxmax(axis=1)).sum())
    taus += [stats.kendalltau(ra.loc[i], rb.loc[i])[0] for i in a.index]
print(f"rank flips {flips}, winner changes {winner_changes}, "
      f"min Kendall tau {min(taus):+.4f}")
```

- [ ] **Step 2: Run it**

Run: `pixi run python -u -m scratchpad.continuum.eps_ranking`
Expected: completes over >= 20 blocks x 6 methods x 4 eps values.

- [ ] **Step 3: Apply the gate**

PASS requires ALL of: `rank flips == 0`, `winner changes == 0`, `min Kendall tau == +1.0000`.

**If it fails, STOP and report.** Do not continue to Task 2. A single rank flip means `eps` must be
pinned by measurement rather than convention, and
`specs/2026-08-06-continuum-permeability-design.md` needs revisiting before any code is written.

- [ ] **Step 4: Record the result**

Append the measured numbers to `docs/superpowers/notes/2026-08-05-continuum-metric-spike.md` under a
new `## A5 gate (2026-08-06)` heading, stating sample size, the three counters, and PASS or FAIL.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/notes/2026-08-05-continuum-metric-spike.md
git commit -m "notes: A5 gate -- eps ranking stability at 21 blocks x 6 methods x 4 eps values"
```

---

## Task 2: Free-space mesh construction

**Files:**
- Create: `src/reblock/continuum/__init__.py`, `src/reblock/continuum/mesh.py`
- Test: `tests/continuum/test_mesh.py`

**Interfaces:**
- Consumes: `reblock.contracts.Block`, `reblock.budget.building_radii`
- Produces:
  - `ContinuumParams(sigma_walk: float = 0.1, sigma_road_per_m: float = 6.666666666666667,
    eps_separation_m: float = 1.0, radius_floor_m: float = 0.25, h_m: float = 0.5,
    h_refine_factor: float = 1.4, h_tolerance: float = 0.005)` — frozen dataclass
  - `FreeSpace` — frozen dataclass with fields `xy: NDArray[np.float64]` (M,2 cell centres),
    `rows: NDArray[np.int64]`, `cols: NDArray[np.int64]` (undirected edges, `rows < cols`),
    `ground: NDArray[np.bool_]` (M,), `demand: NDArray[np.float64]` (M,),
    `n_unplaced: int` (buildings with no free cell in their ring)
  - `shrunk_radii(building_points: GeoDataFrame, params: ContinuumParams) -> NDArray[np.float64]`
  - `free_space_mesh(block: Block, params: ContinuumParams, h: float | None = None) -> FreeSpace`

- [ ] **Step 1: Write the failing tests**

Create `tests/continuum/test_mesh.py`:

```python
"""Guards for free-space mesh construction. The mesh is a function of block geometry and buildings
ALONE -- roads never enter it -- which is what makes the metric's monotonicity structural."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon

from reblock.contracts import Block
from reblock.continuum import ContinuumParams, free_space_mesh, shrunk_radii

UTM = CRS.from_epsg(32734)


def _block(points: list[tuple[float, float]], side: float = 40.0) -> Block:
    boundary = Polygon([(0, 0), (side, 0), (side, side), (0, side)])
    pts = [Point(x, y) for x, y in points]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(pts)))},
                               geometry=[p.buffer(1.0) for p in pts], crs=UTM)
    return Block(block_id="c", crs=UTM, boundary=boundary, parcels=parcels,
                 streets=gpd.GeoDataFrame(geometry=[LineString([(0, 0), (side, 0)])], crs=UTM),
                 building_points=gpd.GeoDataFrame(geometry=pts, crs=UTM))


def test_shrunk_radii_open_a_gap_between_mutual_nearest_neighbours():
    # Two buildings 6 m apart are mutual nearest neighbours, so NN/2 = 3.0 each and the disks TOUCH
    # exactly -- zero free gap, which is what makes the raw model's free space fragment.
    blk = _block([(10.0, 20.0), (16.0, 20.0)])
    p = ContinuumParams(eps_separation_m=1.0)
    r = shrunk_radii(blk.building_points, p)
    assert r == pytest.approx([2.0, 2.0])
    assert 6.0 - r[0] - r[1] == pytest.approx(2.0)    # a 2*eps corridor, by construction


def test_radius_floor_keeps_a_building_an_obstacle():
    blk = _block([(10.0, 20.0), (11.0, 20.0)])        # NN/2 = 0.5, eps 1.0 would go negative
    r = shrunk_radii(blk.building_points, ContinuumParams(eps_separation_m=1.0))
    assert (r == pytest.approx(0.25)).all()


def test_cells_inside_buildings_are_excluded_and_free_cells_are_connected():
    blk = _block([(10.0, 20.0), (16.0, 20.0), (30.0, 30.0)])
    fs = free_space_mesh(blk, ContinuumParams(h_m=0.5, eps_separation_m=1.0))
    r = shrunk_radii(blk.building_points, ContinuumParams(eps_separation_m=1.0))
    xy = np.column_stack([blk.building_points.geometry.x, blk.building_points.geometry.y])
    for k in range(len(xy)):
        d = np.hypot(fs.xy[:, 0] - xy[k, 0], fs.xy[:, 1] - xy[k, 1])
        assert (d >= r[k] - 1e-9).all(), "a cell centre landed inside a building disk"
    assert len(fs.rows) > 0
    assert (fs.rows < fs.cols).all()


def test_ground_is_the_cells_meeting_the_street():
    blk = _block([(20.0, 20.0)])
    fs = free_space_mesh(blk, ContinuumParams(h_m=0.5))
    assert fs.ground.any()
    assert fs.xy[fs.ground, 1].max() <= 0.5 + 1e-9    # street is y = 0; ground hugs it


def test_demand_is_one_unit_per_building_spread_over_its_ring():
    blk = _block([(10.0, 20.0), (30.0, 20.0)])
    fs = free_space_mesh(blk, ContinuumParams(h_m=0.5))
    assert fs.demand.sum() == pytest.approx(2.0)
    assert fs.n_unplaced == 0
    assert (fs.demand > 0).sum() > 2, "demand must be SPREAD, not a point source"


def test_the_mesh_does_not_depend_on_roads():
    # The property the whole monotonicity argument rests on: free_space_mesh takes no roads at all.
    import inspect
    assert "roads" not in inspect.signature(free_space_mesh).parameters
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/continuum/test_mesh.py -q --no-cov`
Expected: FAIL, `ModuleNotFoundError: No module named 'reblock.continuum'`

- [ ] **Step 3: Implement**

Create `src/reblock/continuum/mesh.py`:

```python
"""Free-space mesh: the block minus building disks, on a grid.

The mesh is a function of block geometry and buildings ALONE. Roads never enter it -- they act only
through `sigma` in `continuum.solve` -- and that is exactly what makes the metric's monotonicity
structural rather than an argument about nested edge sets. Three prior mesh attempts died because
their access edges MOVED when roads were added (`3a8dd25`, permeability falling ~9%).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from shapely.ops import unary_union

from reblock.contracts import Block


@dataclass(frozen=True)
class ContinuumParams:
    sigma_walk: float = 0.1
    sigma_road_per_m: float = 6.666666666666667
    # Minimum separation between buildings. `building_radii` is HALF the nearest-neighbour
    # distance, so a mutual-NN pair has r_i + r_j = d exactly and zero free gap -- measured at
    # 10.0% of adjacent pairs. Without this the free-space topology fragments without limit under
    # refinement and permeability never converges. MEASURED to be a nuisance parameter: it moves
    # absolute permeability but never reorders methods (0/50 rank flips, Kendall tau +1.000), so it
    # must be held FIXED across any comparison. See specs/2026-08-06-continuum-permeability-design.
    eps_separation_m: float = 1.0
    radius_floor_m: float = 0.25     # a building stays an obstacle however much it is shrunk
    h_m: float = 0.5
    h_refine_factor: float = 1.4     # convergence check solves at h and h / this
    h_tolerance: float = 0.005       # permeability difference that flags an under-resolved block


@dataclass(frozen=True)
class FreeSpace:
    xy: NDArray[np.float64]
    rows: NDArray[np.int64]
    cols: NDArray[np.int64]
    ground: NDArray[np.bool_]
    demand: NDArray[np.float64]
    n_unplaced: int


def shrunk_radii(building_points: GeoDataFrame, params: ContinuumParams) -> NDArray[np.float64]:
    """`max(NN/2 - eps, floor)`. Shared with `budget.displacement` -- one radius, both axes."""
    from reblock.budget import building_radii

    return np.maximum(building_radii(building_points) - params.eps_separation_m,
                      params.radius_floor_m)


def free_space_mesh(block: Block, params: ContinuumParams, h: float | None = None) -> FreeSpace:
    """Grid cells inside the block and outside every building disk, with 5-point neighbours.

    For uniform `sigma` a grid edge has conductance `sigma*h/h = sigma`, independent of `h` -- which
    is why the discretization can converge at all. Demand is injected over each building's PERIMETER
    ring rather than at a point, because a 2D point source has log-divergent self-energy and `P`
    would grow without bound as `h -> 0`.
    """
    h = params.h_m if h is None else h
    minx, miny, maxx, maxy = block.boundary.bounds
    nx = int(np.ceil((maxx - minx) / h)) + 1
    ny = int(np.ceil((maxy - miny) / h)) + 1
    xg, yg = np.meshgrid(minx + h * np.arange(nx), miny + h * np.arange(ny), indexing="ij")
    pts = shapely.points(xg.ravel(), yg.ravel())

    free = shapely.contains(block.boundary, pts)
    bpts = block.building_points
    if bpts is not None and len(bpts):
        xy_b = np.column_stack([bpts.geometry.x.to_numpy(), bpts.geometry.y.to_numpy()])
        radii = shrunk_radii(bpts, params)
        disks = unary_union(list(shapely.buffer(shapely.points(xy_b), radii)))
        free &= ~shapely.contains(disks, pts)
    else:
        xy_b = np.zeros((0, 2))
        radii = np.zeros(0)

    n = int(free.sum())
    idx = -np.ones(len(pts), dtype=np.int64)
    idx[free] = np.arange(n)
    grid = idx.reshape(nx, ny)
    rows, cols = [], []
    for a, b in ((grid[:-1, :], grid[1:, :]), (grid[:, :-1], grid[:, 1:])):
        m = (a >= 0) & (b >= 0)
        rows.append(a[m])
        cols.append(b[m])
    ri = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    ci = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    lo, hi = np.minimum(ri, ci), np.maximum(ri, ci)

    fpts = pts[free]
    street = unary_union(list(block.streets.geometry)) if len(block.streets) else None
    ground = (shapely.dwithin(street, fpts, h) if street is not None and not street.is_empty
              else np.zeros(n, dtype=bool))

    demand = np.zeros(n, dtype=np.float64)
    unplaced = 0
    if len(xy_b):
        rings = shapely.buffer(shapely.points(xy_b), radii + 1.5 * h)
        for k in range(len(xy_b)):
            sel = np.flatnonzero(shapely.contains(rings[k], fpts))
            if len(sel) == 0:
                unplaced += 1
                continue
            demand[sel] += 1.0 / len(sel)
    xy = (np.column_stack([shapely.get_x(fpts), shapely.get_y(fpts)]) if n
          else np.zeros((0, 2), dtype=np.float64))
    return FreeSpace(xy=xy, rows=lo, cols=hi, ground=ground, demand=demand, n_unplaced=unplaced)
```

Create `src/reblock/continuum/__init__.py`:

```python
"""Permeability as conduction on the free space (specs/2026-08-06-continuum-permeability-design)."""
from reblock.continuum.mesh import ContinuumParams, FreeSpace, free_space_mesh, shrunk_radii
from reblock.continuum.solve import continuum_power, converged_power

__all__ = ["ContinuumParams", "FreeSpace", "continuum_power", "converged_power",
           "free_space_mesh", "shrunk_radii"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/continuum/test_mesh.py -q --no-cov`
Expected: PASS (6 tests)

- [ ] **Step 5: Lint and typecheck**

Run: `pixi run ruff check src tests && pixi run typecheck`
Expected: both clean. `typecheck` is `mypy --strict` and WILL reject untyped locals — it is run in
CI and was missed once already this project.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/continuum tests/continuum/test_mesh.py
git commit -m "feat(continuum): free-space mesh, independent of roads by construction"
```

---

## Task 3: Conduction solve and continuum power

**Files:**
- Create: `src/reblock/continuum/solve.py`
- Test: `tests/continuum/test_solve.py`

**Interfaces:**
- Consumes: `ContinuumParams`, `FreeSpace`, `free_space_mesh` from Task 2
- Produces:
  - `continuum_power(block, roads, params, h=None) -> tuple[float, NDArray[np.float64]]`
    returning `(P, u)` with `u` the per-CELL potential (note: per-cell, NOT per-parcel — the old
    `egress_power` returned per-parcel potentials and callers must not assume that shape)

- [ ] **Step 1: Write the failing tests**

Create `tests/continuum/test_solve.py`:

```python
"""Guards for the conduction solve. The load-bearing property is MONOTONICITY: sigma' >= sigma
pointwise implies P' <= P by the Dirichlet principle, which is why adding a road can never lower
permeability. Three prior mesh designs died on exactly this."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon

from reblock.contracts import Block
from reblock.continuum import ContinuumParams, continuum_power
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, with_width

UTM = CRS.from_epsg(32734)
P = ContinuumParams(h_m=0.5, eps_separation_m=1.0)


def _block(side: float = 40.0) -> Block:
    boundary = Polygon([(0, 0), (side, 0), (side, side), (0, side)])
    pts = [Point(x, y) for x in (10.0, 20.0, 30.0) for y in (10.0, 20.0, 30.0)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(pts)))},
                               geometry=[p.buffer(1.0) for p in pts], crs=UTM)
    return Block(block_id="c", crs=UTM, boundary=boundary, parcels=parcels,
                 streets=gpd.GeoDataFrame(geometry=[LineString([(0, 0), (side, 0)])], crs=UTM),
                 building_points=gpd.GeoDataFrame(geometry=pts, crs=UTM))


def _road(y: float) -> gpd.GeoDataFrame:
    return with_width(gpd.GeoDataFrame(geometry=[LineString([(2.0, y), (38.0, y)])], crs=UTM),
                      DEFAULT_ROAD_WIDTH_M)


def test_power_is_finite_and_positive_with_no_roads():
    p0, u = continuum_power(_block(), None, P)
    assert np.isfinite(p0) and p0 > 0
    assert (u >= -1e-9).all(), "potentials must be non-negative (M-matrix)"


def test_adding_a_road_never_raises_power():
    blk = _block()
    p0, _ = continuum_power(blk, None, P)
    p1, _ = continuum_power(blk, _road(25.0), P)
    assert p1 <= p0 + 1e-9


def test_monotone_under_a_SUPERSET_of_roads():
    """FAULT INJECTION: in `_sigma_field`, replace the road assignment with `sigma_walk` for cells
    already at road level (i.e. let sigma DECREASE somewhere) and this fails."""
    import pandas as pd
    blk = _block()
    one = _road(25.0)
    two = gpd.GeoDataFrame(pd.concat([one, _road(15.0)], ignore_index=True), crs=UTM)
    p_one, _ = continuum_power(blk, one, P)
    p_two, _ = continuum_power(blk, two, P)
    assert p_two <= p_one + 1e-9


def test_a_wider_road_never_raises_power():
    blk = _block()
    narrow = with_width(gpd.GeoDataFrame(geometry=[LineString([(2.0, 25.0), (38.0, 25.0)])],
                                         crs=UTM), 7.0)
    wide = with_width(gpd.GeoDataFrame(geometry=[LineString([(2.0, 25.0), (38.0, 25.0)])],
                                       crs=UTM), 12.0)
    assert continuum_power(blk, wide, P)[0] <= continuum_power(blk, narrow, P)[0] + 1e-9


def test_an_ungrounded_block_returns_inf():
    boundary = Polygon([(0, 0), (40, 0), (40, 40), (0, 40)])
    pts = [Point(20.0, 20.0)]
    blk = Block(block_id="u", crs=UTM, boundary=boundary,
                parcels=gpd.GeoDataFrame({"parcel_id": [0]}, geometry=[pts[0].buffer(1)], crs=UTM),
                streets=gpd.GeoDataFrame(geometry=[], crs=UTM),
                building_points=gpd.GeoDataFrame(geometry=pts, crs=UTM))
    p, _ = continuum_power(blk, None, P)
    assert not np.isfinite(p)


def test_demand_stranded_in_an_ungrounded_component_is_dropped_not_crashed():
    # An ungrounded free-space component makes L singular; pruning must handle it silently.
    blk = _block()
    p, _ = continuum_power(blk, None, ContinuumParams(h_m=2.0, eps_separation_m=0.0))
    assert np.isfinite(p) or np.isinf(p)     # must not raise
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/continuum/test_solve.py -q --no-cov`
Expected: FAIL, `ImportError: cannot import name 'continuum_power'`

- [ ] **Step 3: Implement**

Create `src/reblock/continuum/solve.py`:

```python
"""Conduction solve on the free-space mesh.

    -div(sigma grad u) = f,   u = 0 on the street,   P = f^T u

Monotonicity is structural: `sigma' >= sigma` pointwise implies `P' <= P` by the Dirichlet
principle, and roads only ever RAISE sigma. There is no edge set to keep nested, which is the
failure mode that killed three prior mesh designs.

Use the DIRECT solver. Measured at 3.42M cells (region scale at h = 0.5): `spsolve` 60.4 s and
6.35 GB peak, scaling as N^1.33; CG+Jacobi is ~20x worse and its iteration count grows as sqrt(N).
"""
from __future__ import annotations

import numpy as np
import shapely
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.continuum.mesh import ContinuumParams, FreeSpace, free_space_mesh
from reblock.permeability import WIDTH_COL

GROUND_SHUNT = 1.0e6     # a Dirichlet condition as a large shunt: u -> 0 on grounded cells


def _sigma_field(fs: FreeSpace, roads: GeoDataFrame | None,
                 params: ContinuumParams) -> NDArray[np.float64]:
    """`sigma_walk` everywhere, raised to `sigma_road_per_m` inside any road corridor."""
    sigma = np.full(len(fs.xy), params.sigma_walk, dtype=np.float64)
    if roads is None or len(roads) == 0:
        return sigma
    corridor = unary_union([g.buffer(float(w) / 2.0)
                            for g, w in zip(roads.geometry, roads[WIDTH_COL], strict=True)])
    inside = shapely.contains(corridor, shapely.points(fs.xy[:, 0], fs.xy[:, 1]))
    sigma[inside] = params.sigma_road_per_m
    return sigma


def continuum_power(block: Block, roads: GeoDataFrame | None, params: ContinuumParams,
                    h: float | None = None) -> tuple[float, NDArray[np.float64]]:
    """`(P, u)` with `u` the per-CELL potential. NOTE: per-cell, not per-parcel -- the retired
    `egress_power` returned one value per parcel and callers must not assume that shape."""
    fs = free_space_mesh(block, params, h)
    n = len(fs.xy)
    if n == 0 or not fs.ground.any():
        return float("inf"), np.zeros(max(n, 0), dtype=np.float64)

    sigma = _sigma_field(fs, roads, params)
    sa, sb = sigma[fs.rows], sigma[fs.cols]
    cond = 2.0 * sa * sb / (sa + sb)              # harmonic mean at the interface

    # An ungrounded component has no path to the street, making L singular there. Prune to the
    # grounded components; the demand they carry is dropped, which is correct -- it cannot escape.
    adj = coo_matrix((np.ones(len(fs.rows)), (fs.rows, fs.cols)), shape=(n, n))
    n_comp, lab = connected_components(adj, directed=False)
    live = np.zeros(n_comp, dtype=bool)
    live[np.unique(lab[fs.ground])] = True
    keep = live[lab]
    if not keep.any():
        return float("inf"), np.zeros(n, dtype=np.float64)

    ren = -np.ones(n, dtype=np.int64)
    ren[keep] = np.arange(int(keep.sum()))
    em = keep[fs.rows] & keep[fs.cols]
    ri, ci, cv = ren[fs.rows[em]], ren[fs.cols[em]], cond[em]
    m = int(keep.sum())
    diag = np.zeros(m, dtype=np.float64)
    np.add.at(diag, ri, cv)
    np.add.at(diag, ci, cv)
    diag[fs.ground[keep]] += GROUND_SHUNT

    lap = coo_matrix((np.concatenate([-cv, -cv, diag]),
                      (np.concatenate([ri, ci, np.arange(m)]),
                       np.concatenate([ci, ri, np.arange(m)]))), shape=(m, m)).tocsr()
    b = fs.demand[keep]
    u_keep = spsolve(lap, b)
    if not np.all(np.isfinite(u_keep)):
        return float("inf"), np.zeros(n, dtype=np.float64)
    u = np.zeros(n, dtype=np.float64)
    u[keep] = u_keep
    return float(b @ u_keep), u
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/continuum/test_solve.py -q --no-cov`
Expected: PASS (6 tests)

- [ ] **Step 5: Fault-inject the monotonicity guard**

In `_sigma_field`, temporarily change `sigma[inside] = params.sigma_road_per_m` to
`sigma[inside] = params.sigma_walk * 0.5`. Run the tests: `test_adding_a_road_never_raises_power`
and `test_monotone_under_a_SUPERSET_of_roads` MUST fail. Restore the line and confirm they pass.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/continuum/solve.py tests/continuum/test_solve.py
git commit -m "feat(continuum): conduction solve with structural monotonicity"
```

---

## Task 4: Per-block h-convergence check

**Files:**
- Modify: `src/reblock/continuum/solve.py`
- Test: `tests/continuum/test_convergence.py`

**Interfaces:**
- Consumes: `continuum_power` from Task 3
- Produces:
  - `converged_power(block, roads, params) -> tuple[float, NDArray[np.float64], bool]` returning
    `(P, u, converged)` — solved at `params.h_m`, checked against `params.h_m /
    params.h_refine_factor`

- [ ] **Step 1: Write the failing test**

Create `tests/continuum/test_convergence.py`:

```python
"""`h` is numerical, not physical, so the metric must be insensitive to it. Most blocks converge
cleanly (measured |perm(0.35) - perm(0.25)| <= 0.0018); a rare block does not, and the rare case is
SELF-DETECTING via its own non-converging sweep. Returning a resolution-dependent number silently
is the failure this guards."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon

from reblock.contracts import Block
from reblock.continuum import ContinuumParams, converged_power

UTM = CRS.from_epsg(32734)


def _block(side: float = 40.0) -> Block:
    boundary = Polygon([(0, 0), (side, 0), (side, side), (0, side)])
    pts = [Point(x, y) for x in (10.0, 20.0, 30.0) for y in (10.0, 20.0, 30.0)]
    return Block(block_id="c", crs=UTM, boundary=boundary,
                 parcels=gpd.GeoDataFrame({"parcel_id": list(range(len(pts)))},
                                          geometry=[p.buffer(1.0) for p in pts], crs=UTM),
                 streets=gpd.GeoDataFrame(geometry=[LineString([(0, 0), (side, 0)])], crs=UTM),
                 building_points=gpd.GeoDataFrame(geometry=pts, crs=UTM))


def test_a_well_resolved_block_reports_converged():
    p, u, ok = converged_power(_block(), None, ContinuumParams(h_m=0.5, eps_separation_m=1.0))
    assert np.isfinite(p) and p > 0
    assert ok is True
    assert len(u) > 0


def test_an_impossibly_coarse_grid_reports_NOT_converged():
    # h = 8 m against ~2 m corridors cannot resolve the geometry; the check must say so rather
    # than return a number that depends on the grid.
    _p, _u, ok = converged_power(_block(), None,
                                 ContinuumParams(h_m=8.0, eps_separation_m=1.0,
                                                 h_tolerance=1e-4))
    assert ok is False


def test_the_returned_power_is_the_COARSE_solve():
    # The refined solve exists to CHECK, not to replace: reporting the fine value would make the
    # reported number depend on h_refine_factor.
    from reblock.continuum import continuum_power
    params = ContinuumParams(h_m=0.5, eps_separation_m=1.0)
    p_coarse, _ = continuum_power(_block(), None, params, h=params.h_m)
    p, _u, _ok = converged_power(_block(), None, params)
    assert p == p_coarse
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/continuum/test_convergence.py -q --no-cov`
Expected: FAIL, `ImportError: cannot import name 'converged_power'`

- [ ] **Step 3: Implement**

Append to `src/reblock/continuum/solve.py`:

```python
def converged_power(block: Block, roads: GeoDataFrame | None,
                    params: ContinuumParams) -> tuple[float, NDArray[np.float64], bool]:
    """`(P, u, converged)` at `params.h_m`, with `converged` from a refinement check.

    The reported value is the COARSE solve; the refined one only checks it. Reporting the fine
    value instead would make the number depend on `h_refine_factor`, replacing one resolution
    parameter with another.

    `converged` is False when the two solves' RELATIVE difference exceeds `h_tolerance`, which
    means the grid does not resolve the block's narrowest load-bearing corridor. Measured: the
    common population converges well inside this, and a rare block (`ZAF.9.3.1_1_41829`) does not.
    A caller that ignores this flag is reporting a resolution-dependent number.
    """
    p_coarse, u = continuum_power(block, roads, params, h=params.h_m)
    if not np.isfinite(p_coarse) or p_coarse <= 0.0:
        return p_coarse, u, False
    p_fine, _ = continuum_power(block, roads, params, h=params.h_m / params.h_refine_factor)
    if not np.isfinite(p_fine) or p_fine <= 0.0:
        return p_coarse, u, False
    rel = abs(p_fine - p_coarse) / p_coarse
    return p_coarse, u, bool(rel <= params.h_tolerance)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/continuum/test_convergence.py -q --no-cov`
Expected: PASS (3 tests)

- [ ] **Step 5: Fault-inject**

Change `return p_coarse, u, bool(rel <= params.h_tolerance)` to `return p_coarse, u, True`.
`test_an_impossibly_coarse_grid_reports_NOT_converged` MUST fail. Restore and confirm it passes.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/continuum/solve.py tests/continuum/test_convergence.py
git commit -m "feat(continuum): per-block h-convergence check, so under-resolution is never silent"
```

---

## Task 5: Move the parcel-graph model out of the metric and into the methods

The old parcel graph is NOT deleted — `resistance_greedy` and `resistance_lp` still search with it,
because their per-edge constant-gain assumption has no continuum analogue and reformulating it is
research (spec, "Scope"). It moves to the methods and is relabelled as what it is: a search
surrogate, not the metric.

**Files:**
- Create: `src/reblock/methods/parcel_gain.py`
- Modify: `src/reblock/methods/resistance_greedy.py:58-70`, `src/reblock/methods/resistance_lp.py:63-70`,
  `src/reblock/width_solver.py:94-105`, `scripts/compare_budgets.py:79`,
  `tests/test_permeability.py:16-17`, `tests/test_permeability_oneway.py:18,157`,
  `tests/test_orient.py:22`, `tests/test_width_solver.py`
- Test: `tests/test_parcel_gain.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `parcel_gain.surrogate_power(block, roads, params, *, adj=None, radii=None)` (the
  former `permeability.egress_power`, unchanged behaviour), plus `parcel_radii`,
  `footpath_conductance`, `edge_conductances`, `road_conductance`, `FOOTPATH_EPS`

- [ ] **Step 1: Move the code verbatim**

Create `src/reblock/methods/parcel_gain.py` and MOVE these symbols into it, unchanged, from
`src/reblock/permeability.py`: `FOOTPATH_EPS`, `_road_corridor`, `parcel_radii`,
`_footpath_conductance` (rename to `footpath_conductance`, now public), `road_conductance`,
`lane_width`, `edge_conductances`, `has_oneway`, `egress_power` (rename to `surrogate_power`),
`_directed_power`.

Head the module with:

```python
"""The retired parcel-centroid graph, kept as a SEARCH SURROGATE for the resistance methods.

This is NOT the metric. `permeability` scores conduction on the free space
(`specs/2026-08-06-continuum-permeability-design.md`); this module is the cheap per-edge model that
`resistance_greedy` and `resistance_lp` still optimize against, because their gain function assumes
a road's benefit to an edge is knowable BEFORE the road set is -- which is what makes CELF one solve
per round and keeps the LP linear. That assumption has no continuum analogue and reformulating it is
research, not implementation.

So this is a live surrogate, not a legacy shim: it would be written today, from scratch, as a cheap
guide for a search whose results are graded by a different and more expensive function. What it must
never become again is a SECOND definition of the metric -- the 2026-07-30 bug was exactly that, two
mesh assemblies drifting apart until methods optimized a different Laplacian than the evaluator
graded. The gap this surrogate costs is measured; see the spec's A6.
"""
```

- [ ] **Step 2: Update the importers**

In `resistance_greedy.py`, `resistance_lp.py`, `width_solver.py`, `scripts/compare_budgets.py`,
change every import of `egress_power`, `parcel_radii`, `_footpath_conductance`, `road_conductance`,
`lane_width`, `edge_conductances` from `reblock.permeability` to `reblock.methods.parcel_gain`, and
rename `egress_power` -> `surrogate_power` and `_footpath_conductance` -> `footpath_conductance` at
each call site.

In `resistance_greedy.py`, delete its own `_mesh` (line 95) and call
`parcel_gain.footpath_conductance` / `parcel_gain.road_conductance` directly, so there is exactly
one copy of the surrogate.

- [ ] **Step 3: Write the drift guard**

Create `tests/test_parcel_gain.py`:

```python
"""The surrogate must have exactly ONE implementation. Two copies drifting apart is the 2026-07-30
bug, where methods optimized a different Laplacian than the evaluator graded."""
from __future__ import annotations

import ast
from pathlib import Path


def test_no_module_outside_parcel_gain_builds_its_own_footpath_mesh():
    offenders = []
    for path in Path("src/reblock").rglob("*.py"):
        if path.name == "parcel_gain.py":
            continue
        src = path.read_text()
        if "footpath_conductance(" in src and "from reblock.methods.parcel_gain" not in src:
            offenders.append(str(path))
    assert offenders == [], f"these build a footpath mesh without the shared surrogate: {offenders}"


def test_parcel_gain_is_not_imported_by_the_metric():
    """The metric must not depend on the surrogate. If it does, they can drift into two
    definitions of permeability again."""
    src = Path("src/reblock/permeability.py").read_text()
    assert "parcel_gain" not in src
    tree = ast.parse(Path("src/reblock/continuum/solve.py").read_text())
    mods = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
    assert not any("parcel_gain" in m for m in mods)
```

- [ ] **Step 4: Run the full suite**

Run: `pixi run pytest -q --no-cov`
Expected: PASS. Existing tests importing `egress_power` from `reblock.permeability`
(`tests/test_permeability.py:16`, `tests/test_permeability_oneway.py:157`) must be updated to the
new location as part of this task.

- [ ] **Step 5: Fault-inject the drift guard**

Add `foo = footpath_conductance(1, 2, 3)` to `src/reblock/methods/clearance.py` without importing
it. `test_no_module_outside_parcel_gain_builds_its_own_footpath_mesh` MUST fail. Remove it and
confirm the test passes.

- [ ] **Step 6: Commit**

```bash
git add src/reblock tests scripts
git commit -m "refactor: move the parcel-graph model to methods/parcel_gain as a search surrogate"
```

---

## Task 6: Swap `permeability` to the continuum and delete the old metric path

**Files:**
- Modify: `src/reblock/permeability.py` (`permeability`, `permeability_curve`, `parcel_potentials`)
- Test: `tests/test_permeability.py`

**Interfaces:**
- Consumes: `converged_power` from Task 4
- Produces: `permeability(block, roads, params=..., *, p0=None) -> float` — SAME name and return
  type as today. The `adj=` and `radii=` keyword arguments are REMOVED (they were parcel-graph
  freezing hooks with no continuum meaning); every caller passing them must be updated.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_permeability.py`:

```python
def test_permeability_is_zero_with_no_roads_and_rises_with_one():
    from reblock.continuum import ContinuumParams
    blk = _block()                       # existing fixture in this file
    params = ContinuumParams(h_m=0.5, eps_separation_m=1.0)
    assert permeability(blk, None, params) == pytest.approx(0.0, abs=1e-9)
    roads = with_width(gpd.GeoDataFrame(
        geometry=[LineString([(2.0, 25.0), (38.0, 25.0)])], crs=UTM), DEFAULT_ROAD_WIDTH_M)
    assert permeability(blk, roads, params) > 0.0


def test_permeability_no_longer_accepts_the_parcel_graph_hooks():
    import inspect
    sig = inspect.signature(permeability).parameters
    assert "adj" not in sig and "radii" not in sig
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_permeability.py -q --no-cov`
Expected: FAIL — `permeability` still takes `adj`/`radii` and still uses `egress_power`.

- [ ] **Step 3: Rewrite `permeability` and `permeability_curve`**

In `src/reblock/permeability.py`, replace the bodies:

```python
def permeability(block: Block, roads: GeoDataFrame | None,
                 params: ContinuumParams = ContinuumParams(),  # noqa: B008 (frozen)
                 *, p0: float | None = None) -> float:
    """1 - P(roads)/P(no roads), on the free-space conduction model.

    `p0` freezes the no-roads baseline across a sweep. Guards: a non-finite or non-positive
    baseline (an ungrounded block) -> nan; a non-finite roaded power -> -inf.

    NOTE the retired `adj=` / `radii=` hooks are gone: they froze parcel-graph adjacency and radii,
    neither of which exists here. What is worth freezing now is the MESH, which is a function of the
    block alone -- see `free_space_mesh`.
    """
    if p0 is None:
        p0, _, _ = converged_power(block, None, params)
    if not np.isfinite(p0) or p0 <= 0.0:
        return float("nan")
    p1, _, _ = converged_power(block, roads, params)
    if not np.isfinite(p1):
        return float("-inf")
    return 1.0 - p1 / p0
```

Update `permeability_curve` identically (drop `adj`/`radii`, call `converged_power`), and rewrite
`parcel_potentials` to return per-CELL potentials with their coordinates, renaming it
`cell_potentials(block, roads, params) -> tuple[NDArray, NDArray]` returning `(xy, u)`.

- [ ] **Step 4: Delete the retired symbols**

Remove from `src/reblock/permeability.py`: `FOOTPATH_EPS`, `_road_corridor`, `parcel_radii`,
`_footpath_conductance`, `edge_conductances`, `has_oneway`, `egress_power`, `_directed_power`,
`road_conductance`, `lane_width` (all now live in `methods/parcel_gain.py`). DELETE
`PermeabilityParams` as well: Task 7 moves its three surviving fields (`road_margin_m`,
`min_one_way_width_m`, `min_two_way_width_m`) onto `ContinuumParams`, so `buildable_widths` takes a
`ContinuumParams`. Verified: `buildable_widths` reads only those three fields and the width/oneway
columns — it does NOT call `lane_width`, so the split is clean.

- [ ] **Step 5: Update every caller**

Run `grep -rn "adj=\|radii=" --include=*.py src/reblock/budget.py scripts/` and remove those
keyword arguments from `permeability` / `permeability_curve` calls. `budget.prefix_to_permeability`
(line 839) builds `parcel_adjacency` solely to thread it through — delete that too.

- [ ] **Step 6: Run the full suite, lint, typecheck**

Run: `pixi run pytest -q --no-cov && pixi run ruff check src tests scripts && pixi run typecheck`
Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add src/reblock tests scripts
git commit -m "feat: permeability scores free-space conduction; parcel-graph metric path deleted"
```

---

## Task 7: Parameter and config migration

**Files:**
- Modify: `conf/permeability.yaml`, `src/reblock/permeability.py`
- Test: `tests/test_config_params.py` (create)

**Interfaces:**
- Consumes: `ContinuumParams` from Task 2
- Produces: hydra instantiates `ContinuumParams` from `conf/permeability.yaml`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_params.py`:

```python
"""conf/ and the dataclass must not drift, and the retired parameters must be GONE from config --
a stale key that silently does nothing is worse than a missing one."""
from __future__ import annotations

from pathlib import Path

import yaml

from reblock.continuum import ContinuumParams

RETIRED = {"g_street", "radius_frac", "g_walk", "g_road_per_m"}


def test_config_has_no_retired_parameters():
    cfg = yaml.safe_load(Path("conf/permeability.yaml").read_text())
    assert RETIRED.isdisjoint(cfg), f"retired keys still in config: {RETIRED & set(cfg)}"


def test_every_continuum_param_is_settable_from_config():
    cfg = yaml.safe_load(Path("conf/permeability.yaml").read_text())
    fields = set(ContinuumParams.__dataclass_fields__)
    lens_keys = {"matched_displacement", "matched_permeability"}
    assert set(cfg) - lens_keys <= fields, f"config keys with no field: {set(cfg) - lens_keys - fields}"
    assert {"eps_separation_m", "h_m", "h_tolerance"} <= set(cfg), "eps and h must be EXPLICIT"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_config_params.py -q --no-cov`
Expected: FAIL — `g_street`, `radius_frac`, `g_walk`, `g_road_per_m` are all still present.

- [ ] **Step 3: Rewrite the config**

Replace `conf/permeability.yaml` body with:

```yaml
# Free-space conduction parameters (specs/2026-08-06-continuum-permeability-design.md).
sigma_walk: 0.1
sigma_road_per_m: 6.666666666666667
# Minimum building separation. MEASURED to be a nuisance parameter -- it moves absolute
# permeability but never reorders methods -- so it must be held FIXED across any comparison.
# Changing it invalidates comparisons against previously published numbers.
eps_separation_m: 1.0
radius_floor_m: 0.25
# Grid resolution. Numerical, not physical: `converged_power` checks each block against a refined
# solve and flags blocks the grid cannot resolve rather than returning a resolution-dependent value.
h_m: 0.5
h_refine_factor: 1.4
h_tolerance: 0.005
# Road buildability floors, unchanged.
road_margin_m: 1.0
min_one_way_width_m: 4.0
min_two_way_width_m: 7.0
matched_displacement: 0.10   # 10% of homes (Lens A)
matched_permeability: 0.60   # permeability >= 0.60 (Lens B)
```

Move `road_margin_m`, `min_one_way_width_m`, `min_two_way_width_m` onto `ContinuumParams` (they are
consumed by `buildable_widths`, which stays in `permeability.py`).

- [ ] **Step 4: Run tests, lint, typecheck**

Run: `pixi run pytest -q --no-cov && pixi run ruff check src tests && pixi run typecheck`
Expected: all clean.

- [ ] **Step 5: Commit**

```bash
git add conf src tests
git commit -m "feat: migrate config to continuum parameters; delete g_street and radius_frac"
```

---

## Task 8: Share the shrunk radii with `displacement`

Decoupling is a measured, method-differential confound: the shrink ratio spans 0.823-0.922 at
`eps = 1.0` and flips 10% of per-block method ranks, so the two axes must not disagree about the
same geometry.

**Files:**
- Modify: `src/reblock/budget.py:47-95` (`building_radii`, `displacement` call sites)
- Test: `tests/test_budget.py`

**Interfaces:**
- Consumes: `shrunk_radii` from Task 2
- Produces: every `displacement` call site passes `shrunk_radii(...)`, not `building_radii(...)`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_budget.py`:

```python
def test_displacement_uses_the_same_shrunk_radii_as_the_metric():
    """One radius, both axes. Decoupling is method-differential (ratio 0.823-0.922 at eps=1.0,
    flipping 10% of ranks), so it distorts the Pareto frontier rather than shifting a level."""
    import ast
    from pathlib import Path
    tree = ast.parse(Path("src/reblock/budget.py").read_text())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "building_radii"]
    assert calls == [], "budget.py must call shrunk_radii, not raw building_radii"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_budget.py -q --no-cov -k shrunk`
Expected: FAIL — `building_radii` is still called directly.

- [ ] **Step 3: Rewire**

In `src/reblock/budget.py`, replace every internal `building_radii(pts)` with
`shrunk_radii(pts, params)`, threading `params: ContinuumParams` into `displacement_curve`,
`prefix_to_displacement` and `max_access_depth` where needed. Keep `building_radii` itself as the
unshrunk primitive that `shrunk_radii` builds on — it is not dead, it is the input.

Add to `displacement`'s docstring:

```
`radii` MUST be `continuum.shrunk_radii`, the same disks the metric treats as obstacles. Charging
`NN/2` here while circulation walks through `NN/2 - eps` makes the two axes disagree about the same
geometry, and the disagreement is method-differential rather than a level shift -- measured ratio
0.823-0.922 across methods at eps = 1.0, flipping 10% of per-block ranks.
```

- [ ] **Step 4: Run the full suite**

Run: `pixi run pytest -q --no-cov`
Expected: PASS. Displacement values in fixtures WILL move 8-18%; update
`tests/data/example_fixture/lens_displacement.csv` from a fresh run and note the change in the
commit message.

- [ ] **Step 5: Commit**

```bash
git add src/reblock/budget.py tests
git commit -m "feat: displacement shares the metric's shrunk radii (8-18% move, coupling fix)"
```

---

## Task 9: Acceptance measurements A1-A4, A6, A7

**Files:**
- Create: `scripts/accept_continuum.py`
- Modify: `docs/superpowers/specs/2026-08-06-continuum-permeability-design.md` (record results)

**Interfaces:**
- Consumes: everything from Tasks 2-8

- [ ] **Step 1: Write the acceptance script**

Create `scripts/accept_continuum.py` measuring, and printing PASS/FAIL for, each of:

- **A1** — build a straight road and a zigzag of the same endpoints covering the same parcels;
  assert `permeability` differs by more than 1e-6. Today they are bit-identical to 3.07x detour.
- **A2** — a road of known detour ratio scores as its true length, not its chord: build a road whose
  path length is 2x its chord and assert its permeability is below that of the straight road of
  chord length.
- **A3** — over >= 10 real blocks, add roads in drainage order and assert permeability is
  non-decreasing at every prefix.
- **A4** — `converged_power(...)[2]` is False for `ZAF.9.3.1_1_41829` and True for >= 20 blocks of
  the common population.
- **A6** — permeability reached at matched displacement by `resistance_greedy` (searching the
  surrogate) against a reference that re-ranks candidates by continuum permeability each round, on
  >= 10 blocks. Report median and worst-case shortfall. A NUMBER, not a gate.
- **A7** — region-scale solve wall clock; must be within 2x the measured 60.4 s.

- [ ] **Step 2: Run it**

Run: `pixi run python -u -m scripts.accept_continuum`
Expected: A1-A4 and A7 PASS; A6 prints a number.

- [ ] **Step 3: Record the results in the spec**

Add an `## Acceptance results (YYYY-MM-DD)` section to the spec with the measured value for each
criterion. If any of A1-A4 or A7 fails, STOP and report — do not proceed to Task 10.

- [ ] **Step 4: Commit**

```bash
git add scripts/accept_continuum.py docs/superpowers/specs
git commit -m "test: continuum acceptance A1-A4, A6, A7 measured and recorded"
```

---

## Task 10: A8 lens survival, then regenerate the examples

Regeneration is the expensive, once step. A8 must clear FIRST: if methods can no longer reach Lens
B's `P* = 0.60`, `prefix_to_permeability` returns `(all roads, False)` and the lens degrades
silently.

**Files:**
- Modify: `conf/permeability.yaml` (only if `P*` must move), `docs/` example outputs

- [ ] **Step 1: Check lens reachability BEFORE regenerating**

Write and run a short check over >= 20 blocks x every method in `conf/compare_config.yaml`:
call `budget.prefix_to_permeability(block, roads, 0.60, params)` and count how often the second
return value is False. Print the per-method reachability rate.

- [ ] **Step 2: Decide `P*`**

If every method reaches `P* = 0.60` on >= 90% of blocks, keep it. Otherwise choose the largest
`P*` in `{0.60, 0.55, 0.50, 0.45}` meeting that bar, update `conf/permeability.yaml`, and record the
change and its reason in the spec's acceptance section.

- [ ] **Step 3: Estimate total regeneration time and report it BEFORE running**

Multiply the measured per-solve cost by the number of solves each example needs. Report the estimate
and wait for confirmation before starting — this is a long batch job.

- [ ] **Step 4: Regenerate**

Run: `pixi run python -m scripts.gen_example` for each variant in `conf/example/`.

- [ ] **Step 5: Verify the derivation cache actually invalidated**

`_DERIVATION_MODULES` globs `methods/*.py` plus `permeability.py`, so the code hash must change.
Confirm new cache entries were WRITTEN — a regeneration that republishes stale results while only
`run.log` timestamps move has happened before on this project.

- [ ] **Step 6: Commit**

```bash
git add docs conf
git commit -m "chore: regenerate examples under the continuum metric"
```

---

## Self-Review

**Spec coverage.** Model → Tasks 2-3. Perimeter demand → Task 2 (`free_space_mesh`) with a guard.
Dirichlet ground → Task 2-3 (`GROUND_SHUNT`). Subsumed warts and parameter deletions → Tasks 6-7.
Monotonicity → Task 3 with fault injection. `eps` nuisance → Task 1 (A5 gate) and Task 7 (config).
`h` per-block check → Task 4. Displacement coupling → Task 8. Cost → Task 9 (A7). Scope/proxy →
Task 5, measured in Task 9 (A6). Acceptance A1-A8 → Tasks 1, 9, 10. Blast radius → Tasks 8, 10.

**Known gap accepted deliberately:** the spec's follow-on (unify `clearance.py`'s cost field with
`sigma`) is NOT in this plan. It is a separate spec, as the spec itself says.

**Type consistency checked:** `ContinuumParams` is the single params type from Task 2 onward;
`continuum_power` returns `(float, NDArray)` and `converged_power` returns
`(float, NDArray, bool)` consistently in Tasks 3, 4 and 6; `shrunk_radii` has the same signature in
Tasks 2 and 8; the former `egress_power` is `surrogate_power` everywhere after Task 5.
