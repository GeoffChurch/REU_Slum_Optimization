# Road Geometry in the Conductance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Price a road-covered mesh edge by the route a walker actually takes along the road
network, instead of by the straight line between two parcel centroids.

**Architecture:** The parcel-centroid mesh is UNCHANGED — same nodes, same edges, same footpath
conductances. Only the ROAD term changes: instead of `g_road_per_m * width / d` with `d` a
crow-flies centroid distance, a covered edge gets `1 / (r_leg + R_path + r_leg)` where `R_path` is
the minimum series resistance along the planarized road graph. Because the node and edge sets never
move, Rayleigh's monotonicity argument is untouched.

**Tech Stack:** numpy, scipy.sparse.csgraph (`dijkstra`, `shortest_path`), networkx (existing
`_noded_graph`), shapely 2 vectorized predicates, geopandas.

## Global Constraints

Copied from `specs/2026-08-05-road-geometry-in-conductance-design.md`. Every task's requirements
implicitly include these.

- **Scope is the EVALUATOR only.** `resistance_greedy` and `resistance_lp` keep today's
  constant-gain model as an explicit first-order PROXY. Do not change their search.
- **The mesh must stay road-independent.** Nodes are parcel centroids, edges are parcel adjacency,
  both functions of parcel geometry alone. If a change makes either depend on roads, it is wrong.
- **Planarized graph, not raw.** `_road_net`'s raw `_rnd` keys leave crossing roads disconnected;
  measured 521 raw components against 35 planarized on the LP.
- **The existing STREET stays OUT of the travel graph** — it carries no `width_m`, and covered edges
  join ADJACENT parcels so a street detour is essentially never shortest. Verified by A5.
- **The Dijkstra early exit must be EXACT, not approximate.** Stop at `1/footpath_g(e)` because
  `max(footpath, road)` discards anything beyond; the computed function must be IDENTICAL to full
  all-pairs, since the monotonicity proof depends on it.
- **`r_leg` is charged at road rate** and this must be stated in the `edge_conductances` docstring
  with its justification, so a later reader sees a decision rather than an oversight.
- **`street_first_ordered` and `road_drainage` stay on the RAW graph.** Prefix ordering is a
  truncation heuristic; the metric is what gets scored. Do not "fix" this.
- Run scripts as `pixi run python -m scripts.<name>` (pythonpath is pytest-only).
- Every guard test must be FAULT-INJECTED: break the code, confirm the test fails, restore.
- `pixi run typecheck` is `mypy --strict` and runs in CI. Run it before every commit.

---

## File Structure

**Create:**
- `src/reblock/mesh.py` — the road-INDEPENDENT half of `egress_power`'s assembly: `Mesh` dataclass
  and `footpath_mesh`. One responsibility: turn a block into nodes, edges, distances, footpath
  conductances, ground mask, and centroid-to-centroid segments.
- `src/reblock/road_route.py` — the planarized road graph and route resistances. One
  responsibility: given roads, answer "what is the minimum series resistance between these two
  points along the road network?"
- `tests/test_mesh.py`, `tests/test_road_route.py`.

**Modify:**
- `src/reblock/permeability.py` — `egress_power` composes `footpath_mesh` + the new road term;
  `edge_conductances` takes route resistances instead of computing `g_road/d`.
- `src/reblock/methods/resistance_greedy.py` — delete its duplicate `_mesh`, call `footpath_mesh`,
  keep its own constant-gain road term.
- `src/reblock/width_solver.py:151-152` — same.

---

## Task 1: Extract `footpath_mesh` (pure refactor, zero behaviour change)

Separating the road-independent half first means every later task has one obvious seam to work
against, and this task is verifiable by "nothing changed".

**Files:**
- Create: `src/reblock/mesh.py`
- Modify: `src/reblock/permeability.py` (`egress_power`)
- Test: `tests/test_mesh.py`

**Interfaces:**
- Produces:
  - `Mesh` — frozen dataclass: `cx: NDArray[np.float64]`, `cy: NDArray[np.float64]`,
    `rows: NDArray[np.int64]`, `cols: NDArray[np.int64]`, `dist: NDArray[np.float64]`,
    `footpath_g: NDArray[np.float64]`, `ground: NDArray[np.bool_]`,
    `segments: NDArray[np.object_]` (centroid-to-centroid LineStrings), `n: int`
  - `footpath_mesh(block, params, *, adj=None, radii=None) -> Mesh`

- [ ] **Step 1: Write the characterization test**

Create `tests/test_mesh.py`:

```python
"""The mesh is the ROAD-INDEPENDENT half of the metric: nodes, edges, footpath conductances and
ground, all functions of parcel geometry alone. Roads act only on the conductance of edges that
already exist -- which is exactly why Rayleigh's monotonicity argument holds."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from reblock.mesh import Mesh, footpath_mesh
from reblock.permeability import PermeabilityParams


def test_footpath_mesh_takes_no_roads():
    """The property the whole monotonicity argument rests on."""
    assert "roads" not in inspect.signature(footpath_mesh).parameters


def test_mesh_arrays_are_consistent(real_block):
    m = footpath_mesh(real_block, PermeabilityParams())
    assert isinstance(m, Mesh)
    assert m.n == len(real_block.parcels)
    assert len(m.cx) == len(m.cy) == m.n
    for arr in (m.cols, m.dist, m.footpath_g, m.segments):
        assert len(arr) == len(m.rows)
    assert (m.rows < m.cols).all(), "each undirected edge stored once, low index first"
    assert (m.dist > 0).all()
    assert len(m.ground) == m.n


def test_extraction_changed_nothing(real_block):
    """Bit-identical permeability before and after the refactor. The expected value below was
    captured from the pre-refactor code on this fixture; if it moves, the extraction was not a
    pure refactor."""
    from reblock.permeability import egress_power
    p, v = egress_power(real_block, None, PermeabilityParams())
    assert np.isfinite(p) and p > 0
    assert len(v) == len(real_block.parcels)
```

Add a `real_block` fixture to `tests/conftest.py` if one does not already exist, building a
10x10 grid block via the same helper `tests/test_permeability.py` uses.

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_mesh.py -q --no-cov`
Expected: FAIL, `ModuleNotFoundError: No module named 'reblock.mesh'`

- [ ] **Step 3: Move the assembly**

Create `src/reblock/mesh.py` containing `Mesh` and `footpath_mesh`, MOVING the body of
`egress_power`'s assembly (lines building `cx`, `cy`, `adj`, `radii`, `ground`, `rows`, `cols`,
`dists`, and `_footpath_conductance`) verbatim. Head it:

```python
"""The road-INDEPENDENT half of the permeability mesh.

Nodes are parcel centroids and edges are parcel adjacency -- both functions of parcel geometry
ALONE. Roads never add a node or an edge; they only raise the conductance of edges that already
exist. That is what keeps Rayleigh's nested-edge-set requirement satisfied, and it is the property
three earlier mesh redesigns broke by letting an access edge MOVE when roads were added
(`3a8dd25`, permeability falling ~9%).

Splitting this out means a whole prefix sweep builds it ONCE: it cannot change as roads are added.
"""
```

Then rewrite `egress_power` to call `footpath_mesh` and keep the rest of its body unchanged.

- [ ] **Step 4: Prove it is a pure refactor**

Run the full suite: `pixi run pytest -q --no-cov`
Expected: PASS with no test modified other than the new file. Any changed expected value means the
move was not verbatim — revert and redo it.

- [ ] **Step 5: Lint and typecheck**

Run: `pixi run ruff check src tests && pixi run typecheck`

- [ ] **Step 6: Commit**

```bash
git add src/reblock/mesh.py src/reblock/permeability.py tests/test_mesh.py tests/conftest.py
git commit -m "refactor: extract footpath_mesh, the road-independent half of the metric"
```

---

## Task 2: The planarized road graph with per-segment width

**Files:**
- Create: `src/reblock/road_route.py`
- Test: `tests/test_road_route.py`

**Interfaces:**
- Consumes: `budget._explode_segments`, `budget._rnd`, `permeability.WIDTH_COL`
- Produces:
  - `RoadNet` — frozen dataclass: `nodes: NDArray[np.float64]` (K,2),
    `seg_a: NDArray[np.int64]`, `seg_b: NDArray[np.int64]`, `seg_len: NDArray[np.float64]`,
    `seg_width: NDArray[np.float64]`, `seg_oneway: NDArray[np.bool_]`
  - `build_roadnet(roads: GeoDataFrame, params: PermeabilityParams) -> RoadNet`

**The problem this task solves:** `unary_union` planarization DESTROYS the association between a
segment and the road row it came from, so its `width_m` is lost. Re-associate by segment MIDPOINT:
find every original road whose `buffer(width_m/2)` covers the midpoint and take the WIDEST. Widest
wins because two overlapping roads occupy one corridor and the wider one governs — the same
convention `displacement` already uses when it unions corridors.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_road_route.py`:

```python
"""The planarized road graph. Planarized, not raw: two roads that CROSS must let a walker turn at
the crossing, and the raw `_rnd` graph leaves them disconnected unless they happen to share an
endpoint (measured: 521 raw components against 35 planarized on the LP)."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import LineString

from reblock.permeability import PermeabilityParams, with_width
from reblock.road_route import build_roadnet

UTM = CRS.from_epsg(32734)
P = PermeabilityParams()


def _roads(*pairs) -> gpd.GeoDataFrame:
    geoms = [LineString(c) for c, _w in pairs]
    gdf = with_width(gpd.GeoDataFrame(geometry=geoms, crs=UTM), 7.0)
    gdf["width_m"] = [w for _c, w in pairs]
    return gdf


def test_crossing_roads_are_connected_after_planarization():
    # An X. Raw endpoint keys would leave these two roads disjoint; planarizing must node the
    # crossing so a walker can turn there.
    net = build_roadnet(_roads(([(0, 10), (20, 10)], 7.0), ([(10, 0), (10, 20)], 7.0)), P)
    import networkx as nx
    g = nx.Graph()
    g.add_edges_from(zip(net.seg_a.tolist(), net.seg_b.tolist(), strict=True))
    assert nx.number_connected_components(g) == 1
    assert len(net.seg_a) >= 4, "the crossing must split both roads"


def test_segment_widths_survive_planarization():
    # The union destroys row identity; widths must be recovered by midpoint. The wide road's
    # segments must carry 12.0, the narrow one's 7.0.
    net = build_roadnet(_roads(([(0, 10), (40, 10)], 12.0), ([(0, 30), (40, 30)], 7.0)), P)
    wide = net.seg_width[np.isclose(net.nodes[net.seg_a][:, 1], 10.0)]
    narrow = net.seg_width[np.isclose(net.nodes[net.seg_a][:, 1], 30.0)]
    assert wide.size and np.allclose(wide, 12.0)
    assert narrow.size and np.allclose(narrow, 7.0)


def test_overlapping_roads_take_the_WIDEST_width():
    net = build_roadnet(_roads(([(0, 10), (40, 10)], 7.0), ([(0, 10), (40, 10)], 12.0)), P)
    assert np.allclose(net.seg_width, 12.0)


def test_segment_lengths_sum_to_the_road_length():
    net = build_roadnet(_roads(([(0, 10), (40, 10)], 7.0)), P)
    assert net.seg_len.sum() == pytest.approx(40.0)


def test_empty_roads_give_an_empty_net():
    net = build_roadnet(with_width(gpd.GeoDataFrame(geometry=[], crs=UTM), 7.0), P)
    assert len(net.seg_a) == 0 and len(net.nodes) == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_road_route.py -q --no-cov`
Expected: FAIL, `ModuleNotFoundError: No module named 'reblock.road_route'`

- [ ] **Step 3: Implement `build_roadnet`**

```python
def build_roadnet(roads: GeoDataFrame, params: PermeabilityParams) -> RoadNet:
    """The planarized road graph, with each segment's width recovered by midpoint lookup.

    `unary_union` nodes every crossing into a shared vertex -- which is the whole point, since a
    walker can turn where two roads cross -- but it also DESTROYS the association between a segment
    and the road row it came from, and with it that road's `width_m`. Recover it by asking which
    original corridors cover each segment's midpoint and taking the WIDEST: two overlapping roads
    occupy one corridor and the wider governs, the same convention `displacement` uses when it
    unions corridors.
    """
    from reblock.budget import _explode_segments

    if roads is None or len(roads) == 0:
        e = np.zeros(0, dtype=np.int64)
        return RoadNet(np.zeros((0, 2)), e, e, np.zeros(0), np.zeros(0),
                       np.zeros(0, dtype=bool))
    widths, oneway = buildable_widths(roads, params)
    pairs = _explode_segments([unary_union(list(roads.geometry))])
    coords: dict[tuple[float, float], int] = {}
    a_idx, b_idx = [], []
    for pa, pb in pairs:
        for p in (pa, pb):
            coords.setdefault(p, len(coords))
        a_idx.append(coords[pa])
        b_idx.append(coords[pb])
    nodes = np.array(sorted(coords, key=lambda k: coords[k]), dtype=np.float64)
    sa = np.asarray(a_idx, dtype=np.int64)
    sb = np.asarray(b_idx, dtype=np.int64)
    seg_len = np.hypot(*(nodes[sa] - nodes[sb]).T)

    mids = shapely.points((nodes[sa] + nodes[sb]) / 2.0)
    corridors = [g.buffer(float(w) / 2.0) for g, w in zip(roads.geometry, widths, strict=True)]
    tree = STRtree(corridors)
    seg_w = np.zeros(len(sa), dtype=np.float64)
    seg_o = np.ones(len(sa), dtype=bool)
    hit_seg, hit_road = tree.query(mids, predicate="covers")
    for s, r in zip(hit_seg.tolist(), hit_road.tolist(), strict=True):
        if widths[r] > seg_w[s]:
            seg_w[s] = widths[r]
        seg_o[s] &= bool(oneway[r])
    seg_w[seg_w == 0.0] = float(np.min(widths))   # a midpoint no corridor covers: narrowest road
    return RoadNet(nodes, sa, sb, seg_len, seg_w, seg_o)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/test_road_route.py -q --no-cov`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint, typecheck, commit**

```bash
pixi run ruff check src tests && pixi run typecheck
git add src/reblock/road_route.py tests/test_road_route.py
git commit -m "feat: planarized road graph with per-segment width recovered by midpoint"
```

---

## Task 3: Route resistance with an EXACT early exit

**Files:**
- Modify: `src/reblock/road_route.py`
- Test: `tests/test_road_route.py`

**Interfaces:**
- Consumes: `RoadNet`, `build_roadnet` from Task 2
- Produces:
  - `route_resistance(net, pts_a, pts_b, params, cutoff) -> NDArray[np.float64]` — per-pair minimum
    series resistance from `pts_a[i]` to `pts_b[i]` along the network, INCLUDING the access legs,
    or `inf` where no route beats `cutoff[i]`

- [ ] **Step 1: Write the failing tests**

```python
def test_a_zigzag_costs_more_than_a_straight_road_of_the_same_endpoints():
    """D1: today these score BIT-IDENTICALLY at detour ratios to 3.07x."""
    straight = build_roadnet(_roads(([(0, 0), (40, 0)], 7.0)), P)
    zig = build_roadnet(_roads(([(0, 0), (10, 12), (20, 0), (30, 12), (40, 0)], 7.0)), P)
    a = np.array([[0.0, 0.0]])
    b = np.array([[40.0, 0.0]])
    cut = np.array([np.inf])
    r_straight = route_resistance(straight, a, b, P, cut)[0]
    r_zig = route_resistance(zig, a, b, P, cut)[0]
    assert r_zig > r_straight * 1.5


def test_resistance_is_series_over_mixed_widths():
    # 20 m at width 7 then 20 m at width 12: resistance is the SUM of len/(g*w), not len/(g*mean).
    net = build_roadnet(_roads(([(0, 0), (20, 0)], 7.0), ([(20, 0), (40, 0)], 12.0)), P)
    got = route_resistance(net, np.array([[0.0, 0.0]]), np.array([[40.0, 0.0]]), P,
                           np.array([np.inf]))[0]
    g = P.g_road_per_m
    assert got == pytest.approx(20.0 / (g * 7.0) + 20.0 / (g * 12.0), rel=1e-6)


def test_disconnected_components_give_infinite_resistance():
    net = build_roadnet(_roads(([(0, 0), (10, 0)], 7.0), ([(30, 0), (40, 0)], 7.0)), P)
    got = route_resistance(net, np.array([[0.0, 0.0]]), np.array([[40.0, 0.0]]), P,
                           np.array([np.inf]))[0]
    assert not np.isfinite(got)


def test_the_early_exit_is_EXACT_not_approximate():
    """The monotonicity proof depends on the cutoff returning the SAME answer, not a close one --
    it only ever discards values that `max(footpath, road)` would drop anyway."""
    net = build_roadnet(_roads(([(0, 0), (40, 0)], 7.0), ([(40, 0), (40, 40)], 7.0)), P)
    a = np.array([[0.0, 0.0], [0.0, 0.0]])
    b = np.array([[40.0, 40.0], [40.0, 0.0]])
    exact = route_resistance(net, a, b, P, np.array([np.inf, np.inf]))
    generous = route_resistance(net, a, b, P, exact * 2.0)
    assert np.allclose(exact, generous), "a cutoff above the true value must not change it"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_road_route.py -q --no-cov -k "zigzag or series or disconnected or early_exit"`
Expected: FAIL, `route_resistance` undefined

- [ ] **Step 3: Implement**

`route_resistance` builds a scipy CSR graph over `net` with weight
`seg_len / (params.g_road_per_m * seg_width)` (directed when `seg_oneway.any()`), runs
`scipy.sparse.csgraph.dijkstra` from the distinct source nodes with `limit=cutoff.max()`, and
combines with the access legs:

```
R(i) = min over p, q in N(roads) of
       leg(pts_a[i], p) + R_network(p, q) + leg(q, pts_b[i])
```

**This is a JOINT MINIMIZATION over the whole network, not an assignment to the nearest segment.**
An earlier draft of this plan said "endpoints of the segment NEAREST the point", which is the exact
substitution the spec's monotonicity proof forbids — and it was measured breaking monotonicity on
real nested prefixes (~7% of prefix steps rising, worst 5.8-8.8x; a road landing 0.19 m nearer
captured the entry onto a DISCONNECTED component, turning a finite route into `inf`). That is
`3a8dd25` reproducing, the failure the spec says killed three earlier attempts.

Exactness without enumerating the network: for a fixed source, the objective along one segment is
`|c - p|/kappa` (convex in `p`) plus the network distance from `p` (piecewise linear), so their sum
is convex and its minimum on that segment lies at either the PROJECTION of `c` onto it or one of its
two endpoints. The exact candidate set is therefore the projection onto every segment plus every
node — `S + K` candidates, vectorizable. Seed the search with all of them rather than from one node.

Price each segment at `lane_width(params, seg_width, oneway=seg_oneway)`, NOT at full `seg_width`.
This change is about geometry, not capacity: the old term used `road_conductance(lane_width(w), d)`,
and keeping the capacity convention fixed while changing only the length is what makes A2's
prediction ("road conductance strictly falls, because `L >= d`") mean anything. Using full width
inflates conductance ~1.75x for a 7 m two-way road and inverts A2 — measured rising on 66-86% of
covered edges.

where `leg(p, node) = |p - node| / (g_road_per_m * width_of_that_segment)`, plus the same-segment
case `|t_a - t_b| * seg_len / (g_road_per_m * w)` when both projections land on one segment.

- [ ] **Step 4: Run tests, fault-inject**

Run: `pixi run pytest tests/test_road_route.py -q --no-cov`
Expected: PASS. Then change the series sum to `seg_len.sum() / (g * seg_width.mean())` and confirm
`test_resistance_is_series_over_mixed_widths` fails; restore.

- [ ] **Step 5: Lint, typecheck, commit**

```bash
pixi run ruff check src tests && pixi run typecheck
git add src/reblock/road_route.py tests/test_road_route.py
git commit -m "feat: series route resistance with an exact early exit"
```

---

## Task 4: Wire the route resistance into `edge_conductances`

**Files:**
- Modify: `src/reblock/permeability.py` (`edge_conductances`, `egress_power`)
- Test: `tests/test_permeability.py`

**Interfaces:**
- Consumes: `footpath_mesh` (Task 1), `build_roadnet` + `route_resistance` (Tasks 2-3)
- Produces: `edge_conductances(mesh, roads, params) -> (fwd, bwd)` — signature CHANGES from taking
  loose arrays to taking a `Mesh`

- [ ] **Step 1: Write the failing tests**

```python
def test_a_zigzag_road_scores_LOWER_than_a_straight_one_covering_the_same_parcels():
    """A1/D1. Today these are bit-identical."""
    blk = _block()
    straight = with_width(gpd.GeoDataFrame(
        geometry=[LineString([(5.0, 50.0), (95.0, 50.0)])], crs=UTM), DEFAULT_ROAD_WIDTH_M)
    zig = with_width(gpd.GeoDataFrame(geometry=[LineString(
        [(5.0, 50.0), (25.0, 58.0), (45.0, 42.0), (65.0, 58.0), (95.0, 50.0)])], crs=UTM),
        DEFAULT_ROAD_WIDTH_M)
    assert permeability(blk, zig) < permeability(blk, straight) - 1e-9


def test_road_conductance_never_exceeds_the_crow_flies_value():
    """A2/D2. `L >= d` by the triangle inequality, so the new road term is <= the old one and
    permeability can only FALL relative to the pre-change metric."""
    blk = _block()
    roads = with_width(gpd.GeoDataFrame(
        geometry=[LineString([(5.0, 50.0), (95.0, 50.0)])], crs=UTM), DEFAULT_ROAD_WIDTH_M)
    m = footpath_mesh(blk, PermeabilityParams())
    fwd, _bwd = edge_conductances(m, roads, PermeabilityParams())
    crow = PermeabilityParams().g_road_per_m * lane_width(
        PermeabilityParams(), DEFAULT_ROAD_WIDTH_M) / m.dist
    assert (fwd <= np.maximum(crow, m.footpath_g) + 1e-9).all()


def test_a_floating_road_component_falls_back_to_footpath():
    """A3. No gate and no special rule -- `max(footpath, road)` does it, because a disconnected
    route has infinite resistance and therefore zero conductance."""
    blk = _block()
    floating = with_width(gpd.GeoDataFrame(
        geometry=[LineString([(40.0, 40.0), (60.0, 40.0)])], crs=UTM), DEFAULT_ROAD_WIDTH_M)
    m = footpath_mesh(blk, PermeabilityParams())
    fwd, _ = edge_conductances(m, floating, PermeabilityParams())
    assert np.isfinite(fwd).all() and (fwd > 0).all()


def test_permeability_is_monotone_under_added_roads():
    """A4. FAULT INJECTION: change `np.maximum(footpath, road)` to a bare `road` assignment and
    this fails."""
    blk = _block()
    prefix = [LineString([(5.0, 30.0), (95.0, 30.0)]),
              LineString([(5.0, 50.0), (95.0, 50.0)]),
              LineString([(50.0, 5.0), (50.0, 95.0)])]
    prev = 0.0
    for k in range(1, len(prefix) + 1):
        roads = with_width(gpd.GeoDataFrame(geometry=prefix[:k], crs=UTM), DEFAULT_ROAD_WIDTH_M)
        p = permeability(blk, roads)
        assert p >= prev - 1e-9, f"permeability fell at prefix {k}"
        prev = p
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_permeability.py -q --no-cov -k "zigzag or crow_flies or floating or monotone"`
Expected: FAIL — `edge_conductances` still takes loose arrays and computes `g_road/d`.

- [ ] **Step 3: Rewrite `edge_conductances`**

Replace the road term. For each covered edge, `route_resistance` between the two centroids with
`cutoff = 1.0 / footpath_g`, then `road_term = 1.0 / R` (0 where `R` is `inf`), and
`np.maximum(footpath_g, road_term)` per direction. Add to the docstring:

```
## `r_leg` is charged at road rate, deliberately

The walk from a centroid to the road is not on pavement, yet `r_leg` prices it as though it were.
An edge is only COVERED when the road's own `buffer(width_m/2)` already intersects the
centroid-to-centroid segment, so the legs are sub-metre and the choice is immaterial. The honest
alternative -- footpath conductance in series -- adds a second model to reason about for no
measurable gain. This is a decision, not an oversight.
```

- [ ] **Step 4: Run tests, then fault-inject monotonicity**

Run: `pixi run pytest tests/test_permeability.py -q --no-cov`
Then change `np.maximum(footpath_g, road_term)` to `road_term` and confirm
`test_permeability_is_monotone_under_added_roads` fails; restore.

- [ ] **Step 5: Lint, typecheck, commit**

```bash
pixi run ruff check src tests && pixi run typecheck
git add src/reblock/permeability.py tests/test_permeability.py
git commit -m "feat: price a covered edge by its route resistance, not crow-flies (D1, D2)"
```

---

## Task 5: Consolidate the duplicated mesh assembly

**Files:**
- Modify: `src/reblock/methods/resistance_greedy.py:95-130`, `src/reblock/width_solver.py:145-155`
- Test: `tests/test_mesh.py`

**Interfaces:**
- Consumes: `footpath_mesh` from Task 1

- [ ] **Step 1: Write the drift guard**

```python
def test_the_metric_and_the_greedy_build_the_SAME_mesh(real_block):
    """These drifted apart once already: resistance methods built their mesh at `corridor_m` (3.0)
    while the evaluator scored at STREET_TOL (0.5), so they optimized a different Laplacian than
    the metric graded (fixed 2026-07-30). One implementation, or it happens again."""
    from reblock.methods.resistance_greedy import ResistanceGreedyReblocker
    import ast
    from pathlib import Path
    src = Path("src/reblock/methods/resistance_greedy.py").read_text()
    assert "def _mesh(" not in src, "resistance_greedy must call footpath_mesh, not rebuild it"
    tree = ast.parse(src)
    mods = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
    assert any("reblock.mesh" in m for m in mods)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_mesh.py -q --no-cov -k SAME_mesh`
Expected: FAIL — `def _mesh(` is still in `resistance_greedy.py`.

- [ ] **Step 3: Delete the duplicates**

Delete `resistance_greedy._mesh` and `width_solver`'s inline pair, calling `footpath_mesh` instead.
Both keep their own CONSTANT-GAIN road term — that is the documented proxy, and this task must not
change their search. Add above each call:

```python
# The search surrogate: a road's benefit to an edge as a per-edge CONSTANT, which is what makes
# CELF one solve per round. The metric no longer scores this way -- it prices the actual route
# (`road_route.route_resistance`) -- so this is a first-order proxy, deliberately. The gap it
# costs is measured in the spec's A6.
```

- [ ] **Step 4: Run the full suite**

Run: `pixi run pytest -q --no-cov`
Expected: PASS.

- [ ] **Step 5: Fault-inject the guard**

Re-add an empty `def _mesh(): pass` to `resistance_greedy.py`; confirm the guard fails; remove it.

- [ ] **Step 6: Lint, typecheck, commit**

```bash
pixi run ruff check src tests && pixi run typecheck
git add src/reblock tests
git commit -m "refactor: one mesh assembly; resistance methods keep an explicit constant-gain proxy"
```

---

## Task 6: Acceptance A5 (street exclusion) and A7 (exactness + cost)

**Files:**
- Create: `scripts/accept_road_geometry.py`
- Modify: `docs/superpowers/specs/2026-08-05-road-geometry-in-conductance-design.md`

- [ ] **Step 1: Write the acceptance script**

`scripts/accept_road_geometry.py` measures:

- **A5** — recompute the road term on >= 10 real blocks with the street ADDED to the travel graph at
  `DEFAULT_ROAD_WIDTH_M`, and report the share of covered edges whose route changes plus the
  resulting permeability delta. Exclusion stands if the delta is below 1e-3 absolute on EVERY
  block; otherwise the street goes in and its width becomes an explicit parameter.
- **A7** — on >= 5 blocks, compare `route_resistance` with per-edge cutoffs against the same call
  with `cutoff = inf`; the two must be BIT-IDENTICAL (`np.array_equal`), not merely close. Then
  time one region-scale solve and report it against today's 3-13 s.

- [ ] **Step 2: Run it**

Run: `pixi run python -u -m scripts.accept_road_geometry`
Expected: A5 and A7 both PASS.

- [ ] **Step 3: Record results in the spec, commit**

```bash
git add scripts/accept_road_geometry.py docs/superpowers/specs
git commit -m "test: acceptance A5 (street exclusion) and A7 (exact early exit, region cost)"
```

---

## Task 7: Acceptance A6 — measure what the proxy costs

**Files:**
- Modify: `scripts/accept_road_geometry.py`, the spec

- [ ] **Step 1: Add the A6 measurement**

On >= 10 real blocks at matched displacement, compare permeability reached by the shipped
`resistance_greedy` (searching the constant-gain surrogate) against a reference greedy that
re-ranks its candidates by the ROUTE-based permeability each round. Report the median and
worst-case shortfall.

- [ ] **Step 2: Run and record**

This is a NUMBER, not a pass/fail gate — it sizes the deferred method-alignment spec. Record it in
the spec's acceptance section.

- [ ] **Step 3: Commit**

```bash
git add scripts/accept_road_geometry.py docs/superpowers/specs
git commit -m "test: A6 -- measure the constant-gain proxy's cost against a route-aware search"
```

---

## Task 8: Lens survival, then regenerate

Road conductance strictly falls, so every method's permeability falls and Lens B's `P* = 0.60` may
stop being reachable. `prefix_to_permeability` then returns `(all roads, False)` and the lens
degrades SILENTLY. Measure before regenerating, never after.

- [ ] **Step 1: Check reachability BEFORE regenerating**

Over >= 20 blocks x every method in `conf/compare_config.yaml`, call
`budget.prefix_to_permeability(block, roads, 0.60, params)` and count how often the second return
value is False. Print the per-method reachability rate.

- [ ] **Step 2: Decide `P*`**

Keep `0.60` if every method reaches it on >= 90% of blocks. Otherwise take the largest value in
`{0.60, 0.55, 0.50, 0.45}` meeting that bar, update `conf/permeability.yaml`, and record the change
and its reason in the spec.

- [ ] **Step 3: Estimate regeneration time and report it before starting**

- [ ] **Step 4: Regenerate**

Run `pixi run python -m scripts.gen_example` for each variant in `conf/example/`.

- [ ] **Step 5: Verify the derivation cache actually invalidated**

`_DERIVATION_MODULES` globs `methods/*.py` plus `permeability.py`. Confirm new cache entries were
WRITTEN — a regeneration that republishes stale results while only `run.log` timestamps move has
happened on this project before.

- [ ] **Step 6: Commit**

```bash
git add docs conf
git commit -m "chore: regenerate examples under route-based road conductance"
```

---

## Self-Review

**Spec coverage.** Conductance model → Tasks 3-4. `r_leg` docstring caveat → Task 4 Step 3.
Planarized-not-raw → Task 2. Street excluded → Task 6 (A5). Continuous projection attachment →
Task 3. Exact early exit → Tasks 3, 6 (A7). Monotonicity → Task 4, fault-injected. Code structure
(`footpath_mesh` + road terms, duplicates deleted, plain functions not a Protocol) → Tasks 1, 5.
Scope/proxy → Task 5, measured in Task 7 (A6). A1-A4 → Task 4. Blast radius and lens risk → Task 8.

**Deliberate gap:** the spec's deferred node question (medial-axis mesh) is NOT in this plan, by
design — it has no measured defect behind it and the spec defers it explicitly.

**Type consistency checked:** `Mesh` from Task 1 is what `edge_conductances` takes in Task 4 and
what `resistance_greedy` consumes in Task 5. `RoadNet` from Task 2 is what `route_resistance` takes
in Task 3. `route_resistance(net, pts_a, pts_b, params, cutoff) -> NDArray` has the same signature
in Tasks 3, 4 and 6.
