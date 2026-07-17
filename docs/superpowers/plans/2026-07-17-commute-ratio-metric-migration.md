# Commute-Ratio Metric Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the perverse `cycle_density` internal-connectivity metric with the commute-ratio ρ (mean over reachable dwellings of `1 − R(dwelling→street)/R_geodesic`), computed with a plain dense grounded-Laplacian solve, and re-wire the frontier reporting to it.

**Architecture:** ρ is a `budget.py` function on the noded road∪street graph: a geometric street ground set, a component-wise **dense** grounded-Laplacian solve for `R(v)`, a multi-source Dijkstra for `R_geo(v)`, line-proximity parcel→entry mapping, reachable-conditioned mean, clipped to `[0,1)`. It plugs into the existing `cost_benefit_curve`/`_sweep` frontier via a `BenefitFactory` under the unchanged metric key `"internal_connectivity"`. The old `cycle_density`/`cycle_benefit` path is deleted (migrate-not-accommodate); its `_noded_graph` tests are re-homed first.

**Tech Stack:** Python, numpy (`np.linalg.inv`), networkx (`_noded_graph`, `multi_source_dijkstra_path_length`, `connected_components`), shapely (`STRtree`, `Point`, `unary_union`), geopandas, pixi, pytest, ruff, mypy --strict.

**Spec:** `docs/superpowers/specs/2026-07-17-redundancy-metric-and-refiner-design.md` (committed `ce3e6df`). This plan is **Plan 1 of 2**; the loop-closure refiner (spec §4) is a separate later plan — NOT in scope here.

## Global Constraints

- **Scalability is first-class:** ρ uses a component-wise **dense** grounded solve (the de-risk spike showed dense `np.linalg.inv` beats sparse per-entry solve for our entry diagonal; ~0.5 s at m≈3700 nodes). Task 0 opens with a region-scale benchmark gate that **blocks the rest of the plan** if ρ regresses beyond a small factor of `cycle_density`'s reporting cost.
- **Migrate, don't accommodate:** delete `cycle_density`, `cycle_benefit`, `tests/test_cycle_density.py`; no dual path, no deprecated alias.
- **`pixi run check`** (ruff lint + mypy --strict + pytest) must be green at every commit.
- **ruff:** no semicolons (E702), lines ≤ 100 chars (E501), `zip(..., strict=…)` (B905).
- **ρ ∈ [0, 1)** — clip `min(max(ρ, 0.0), 1 − 1e-12)`.
- **Ground set is geometric:** street nodes = graph nodes with `Point(node).distance(street_geom) <= STREET_TOL`. NOT "nodes in raw `block.streets`".
- **Parcel entries via line-proximity** (nearest point on graph EDGES), NOT centroid→nearest-vertex.
- **ρ is NON-MONOTONE** in road length — never assert monotonicity; rank by terminal value, compare at matched budget.
- **Semantics:** a **single-egress** tree route → ρ = 0; a **multi-egress** route (two diverging paths to the street) → ρ > 0 **by design** (do not assert 0 there).
- **Guards:** no roads / no parcels / no interior nodes / empty reachable set / empty graph → `0.0`.
- **Metric key stays `"internal_connectivity"`.**
- **Interfaces:** `BenefitFactory = Callable[..., Callable[[GeoDataFrame | None], float]]`; `commute_ratio(block: Block, roads: GeoDataFrame | None) -> float`; `commute_ratio_benefit(block: Block, roads_full: GeoDataFrame | None, *, tol: float = STREET_TOL) -> Callable[[GeoDataFrame | None], float]`.
- Reuse `_noded_graph`, `_explode_segments` (kept). The deleted engine's geometric-ground + line-entry *conventions* are in commit `fe4180c` for reference (conventions only — NOT its sparse solver).

**PLAN-VS-SPEC DEVIATION (flag to human before Task 0):** Spec §3.3.1 says freeze the parcel→entry map in `commute_ratio_benefit` "to cut remap churn." This plan implements `commute_ratio_benefit` as a bare per-prefix call (no freeze), because the de-risk spike proved (a) freezing does not make the curve monotone and (b) reporting ranks by terminal value (freeze-invariant) and compares at matched budget (one point), so the intricate frozen-entry machinery is YAGNI. If the human wants churn-reduction anyway, add it as a follow-up task.

---

### Task 0: `commute_ratio` metric + scalability gate

**Files:**
- Modify: `src/reblock/budget.py` (add `commute_ratio` near `cycle_density`; add imports)
- Create: `tests/test_commute_ratio.py`
- Create (scratch, not committed): `/tmp/bench_commute_ratio.py`

**Interfaces:**
- Consumes: `_noded_graph`, `_explode_segments`, `STREET_TOL`, `Block` (all in `budget.py`).
- Produces: `commute_ratio(block: Block, roads: GeoDataFrame | None) -> float` ∈ `[0, 1)`.

- [ ] **Step 1: Write the failing semantics tests** (`tests/test_commute_ratio.py`)

```python
# tests/test_commute_ratio.py
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.budget import _noded_graph, commute_ratio
from reblock.contracts import Block

UTM = CRS.from_epsg(32734)


def _block(n_parcels: int, parcel_geoms=None) -> Block:
    boundary = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    geoms = parcel_geoms or [boundary] * n_parcels
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(n_parcels))}, geometry=geoms, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (100, 0)])], crs=UTM)
    return Block(block_id="b", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _roads(lines):
    return gpd.GeoDataFrame(geometry=lines, crs=UTM)


def _parcels_at(pts):
    # small distinct parcels so each maps to a nearby entry node
    return [Polygon([(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1)]) for x, y in pts]


def test_single_egress_tree_is_zero() -> None:
    # one path from an interior point down to the single street: no parallel route -> rho = 0
    block = _block(3, _parcels_at([(50, 40), (50, 30), (50, 20)]))
    roads = _roads([LineString([(50, 0), (50, 50)])])
    assert commute_ratio(block, roads) == 0.0


def test_loop_gives_positive_rho() -> None:
    block = _block(3, _parcels_at([(40, 40), (50, 40), (60, 40)]))
    loop = _roads([LineString([(30, 0), (30, 50), (70, 50), (70, 0)])])  # two arms to the street
    assert commute_ratio(block, loop) > 0.0


def test_big_loop_beats_tiny_loop() -> None:
    parcels = _parcels_at([(20, 40), (40, 40), (60, 40), (80, 40)])
    block = _block(4, parcels)
    big = _roads([LineString([(10, 0), (10, 60), (90, 60), (90, 0)])])       # spans all 4 parcels
    tiny = _roads([LineString([(18, 0), (18, 30), (22, 30), (22, 0)])])      # spans ~1 parcel
    assert commute_ratio(block, big) > commute_ratio(block, tiny)


def test_range_and_empty_guards() -> None:
    block = _block(4, _parcels_at([(40, 40), (50, 40), (60, 40), (30, 40)]))
    loop = _roads([LineString([(30, 0), (30, 50), (70, 50), (70, 0)])])
    r = commute_ratio(block, loop)
    assert 0.0 <= r < 1.0
    assert commute_ratio(block, _roads([])) == 0.0
    assert commute_ratio(block, None) == 0.0
    assert commute_ratio(_block(0, []), loop) == 0.0


def test_stranded_spur_excluded_no_blowup() -> None:
    # a road that never reaches the street: its parcels are excluded (reachable-conditioned), no crash
    block = _block(2, _parcels_at([(50, 80), (50, 70)]))
    spur = _roads([LineString([(50, 60), (50, 90)])])  # detached from the south street
    assert commute_ratio(block, spur) == 0.0


def test_subdivision_invariance() -> None:
    # effective resistance is subdivision-invariant -> rho unchanged by an added mid-vertex
    block = _block(3, _parcels_at([(40, 40), (50, 40), (60, 40)]))
    loop = _roads([LineString([(30, 0), (30, 50), (70, 50), (70, 0)])])
    sub = _roads([LineString([(30, 0), (30, 25), (30, 50), (50, 50), (70, 50), (70, 0)])])
    assert abs(commute_ratio(block, loop) - commute_ratio(block, sub)) < 1e-9


def test_crossing_is_noded_into_a_shared_vertex() -> None:  # RE-HOMED from test_cycle_density.py
    block = _block(4)
    roads = _roads([LineString([(20, 20), (80, 80)]), LineString([(20, 80), (80, 20)])])
    g = _noded_graph(roads, block.streets)
    assert (50.0, 50.0) in g.nodes
    assert g.degree[(50.0, 50.0)] == 4
```

- [ ] **Step 2: Run to verify they fail**

Run: `pixi run pytest tests/test_commute_ratio.py -q`
Expected: FAIL — `ImportError: cannot import name 'commute_ratio' from 'reblock.budget'`.

- [ ] **Step 3: Add imports to `budget.py`**

At the top of `src/reblock/budget.py`, ensure these are imported (add any missing; `np`, `nx`, `unary_union` already present — verify `Point`, `LineString`, `STRtree`):

```python
import math
from shapely import STRtree
from shapely.geometry import Point, LineString
```

- [ ] **Step 4: Implement `commute_ratio`** (add directly after `_noded_graph` in `src/reblock/budget.py`)

```python
def commute_ratio(block: Block, roads: GeoDataFrame | None) -> float:
    """Internal connectivity: mean over reachable parcels of 1 - R(dwelling->street)/R_geodesic on
    the noded road-union-street graph. R = grounded effective resistance to the whole street (a
    component-wise DENSE solve); R_geo = single-best-route (shortest-path) resistance. A single-egress
    tree route -> 0; ->1 as parallel backup routes thicken. Clipped to [0, 1). NON-MONOTONE in road
    length (ratio of co-decreasing R/R_geo) -- reporting ranks by terminal value, never assumes rise.
    Resists loop-COUNT gaming (big loops beat many tiny ones) but not corridor-duplication (the suite's
    cost axes penalize that). 0.0 with no roads / no parcels / no interior nodes / empty reachable set."""
    if roads is None or len(roads) == 0 or len(block.parcels) < 1:
        return 0.0
    g = _noded_graph(roads, block.streets)
    if g.number_of_nodes() == 0:
        return 0.0
    street_geom = unary_union(list(block.streets.geometry))
    snodes = {n for n in g.nodes if Point(n).distance(street_geom) <= STREET_TOL}  # GEOMETRIC ground
    interior = [n for n in g.nodes if n not in snodes]
    if not snodes or not interior:
        return 0.0
    for u, v in g.edges():
        g[u][v]["len"] = max(math.hypot(u[0] - v[0], u[1] - v[1]), 1e-6)
    geo = nx.multi_source_dijkstra_path_length(g, snodes, weight="len")           # R_geo per node
    rg: dict[tuple[float, float], float] = {}                                     # R(v) per interior node
    for comp in nx.connected_components(g):
        comp_streets = comp & snodes
        comp_int = [n for n in comp if n not in snodes]
        if not comp_streets or not comp_int:                                     # stranded -> excluded
            continue
        idx = {n: i for i, n in enumerate(comp_int)}
        m = len(comp_int)
        lg = np.zeros((m, m))
        for u, v in g.subgraph(comp).edges():
            c = 1.0 / g[u][v]["len"]
            ui, vi = idx.get(u), idx.get(v)
            if ui is not None and vi is not None:
                lg[ui, ui] += c
                lg[vi, vi] += c
                lg[ui, vi] -= c
                lg[vi, ui] -= c
            elif ui is not None:
                lg[ui, ui] += c
            elif vi is not None:
                lg[vi, vi] += c
        diag = np.diag(np.linalg.inv(lg))                                        # DENSE grounded solve
        for n, i in idx.items():
            rg[n] = float(diag[i])
    edges = list(g.edges())
    tree = STRtree([LineString([u, v]) for u, v in edges])
    ratios: list[float] = []
    for geom in block.parcels.geometry:
        pt = geom.centroid
        u, v = edges[int(tree.nearest(pt))]                                      # line-proximity entry
        cand = [n for n in (u, v) if n in rg]
        if not cand:
            continue
        node = min(cand, key=lambda n: pt.distance(Point(n)))
        r_eff, r_geo = rg[node], geo.get(node, math.inf)
        if math.isfinite(r_geo) and r_geo > 1e-9:
            ratios.append(min(max(1.0 - r_eff / r_geo, 0.0), 1.0 - 1e-12))       # clip [0,1)
    return float(np.mean(ratios)) if ratios else 0.0
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pixi run pytest tests/test_commute_ratio.py -q`
Expected: PASS (8 passed).

- [ ] **Step 6: Scalability gate (BLOCKING) — benchmark ρ vs cycle_density on a region block**

Create `/tmp/bench_commute_ratio.py`:

```python
import sys, time
sys.path.insert(0, "/home/gchurchill/src/reblock")
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from reblock.pipeline import build_regions
from reblock.budget import commute_ratio, cycle_density, truncate_to_length

with initialize_config_dir(version_base=None, config_dir="/home/gchurchill/src/reblock/conf"):
    cfg = compose(config_name="compare_config", overrides=[
        "data=capetown_full", "max_blocks=1", "block_ids=[[ZAF.9.3.1_1_38528]]"])  # ~2017 parcels
B = build_regions(instantiate(cfg.data), instantiate(cfg.screen),
                  instantiate(cfg.region_builder), [["ZAF.9.3.1_1_38528"]], 1)[0][0]
roads = instantiate(cfg.all_methods["clearance"]).propose(B).roads
L = float(roads.geometry.length.sum())
prefixes = [truncate_to_length(B, roads, (k / 20) * L) for k in range(1, 21)]  # ~20-prefix sweep

def sweep(fn):
    t = time.perf_counter()
    for p in prefixes:
        fn(B, p)
    return time.perf_counter() - t

print(f"parcels={len(B.parcels)}  cycle_density sweep={sweep(cycle_density):.2f}s  "
      f"commute_ratio sweep={sweep(commute_ratio):.2f}s")
```

Run: `pixi run python /tmp/bench_commute_ratio.py`
Expected: `commute_ratio` sweep within ~3× of `cycle_density` sweep and under ~15 s absolute (the de-risk spike predicts ~10 s at m≈3700). **If it regresses beyond that, STOP and report BLOCKED** — revisit the solve before proceeding.

- [ ] **Step 7: Commit**

```bash
git add src/reblock/budget.py tests/test_commute_ratio.py
git commit -m "feat(budget): commute_ratio internal-connectivity metric (dense grounded solve)"
```

---

### Task 1: Corpus validation gate (aggregation, orthogonality, gaming) — BLOCKING

**Files:**
- Create (scratch, not committed): `/tmp/gate_commute_ratio.py`
- Modify: `src/reblock/budget.py` (record the measured numbers in the `commute_ratio` docstring)

**Interfaces:**
- Consumes: `commute_ratio`, `access_benefit` (external), `displacement` (`budget.py`).
- Produces: a go/no-go decision. GO → proceed to Task 2. NO-GO → escalate BLOCKED (spec §2.1 fallback to 2ec is a *different* plan; do not silently switch here).

- [ ] **Step 1: Write the gate script** (`/tmp/gate_commute_ratio.py`) — build the two-block corpus (blocks `ZAF.9.3.1_1_40972`, `ZAF.9.3.1_1_39229`; per block: clearance repulsion sweep `[-4,-2,0,2,4]` + `greedy_arterial_buildable` + `greedy_arterial_aspirational` + 5 random road-subsets, each truncated to the matched budget `L`). For every network record `access_benefit(...)(...)`, `commute_ratio`, and the terminal `displacement`. Then compute and print:
  - `corr(commute_ratio, access)` over the pooled corpus (the orthogonality number).
  - BIG-vs-TINY: on block 40972, clearance + 3 big vs + 3 tiny gap-snapped loops → `commute_ratio` for each (BIG must exceed TINY).
  - Corridor-duplication suite check: clearance + a k-parallel-stub bundle vs clearance + a genuine loop at **matched added length** → print `(commute_ratio, displacement, added_length)` for each; the bundle must be Pareto-dominated (≥ displacement AND ≤ internal-per-metre).
  - Curve behaviour: a drainage-ordered `cost_benefit_curve(B, roads, benefit_fn=commute_ratio_benefit)` terminal value equals `commute_ratio(B, roads_full)` (sanity; do NOT assert monotone).

  (Reuse the corpus-building pattern from `scratchpad/spike_gate4.py`; this is analysis code, not committed.)

- [ ] **Step 2: Run the gate**

Run: `pixi run python /tmp/gate_commute_ratio.py`
Expected output includes the five measurements above.

- [ ] **Step 3: Apply the committed exit rule**

- **GO** iff BIG > TINY **and** `corr(commute_ratio, access) ≤ 0.49`. (`≤ 0.25` is the aspiration; reachable-conditioning — already in `commute_ratio` — is the aggregation.) If the duplication bundle is NOT Pareto-dominated, note it for the refiner plan (near-parallel-collapse fallback) but it does not block the metric.
- **NO-GO** (BIG ≤ TINY, or `corr > 0.49`): STOP, report **BLOCKED** with the numbers — the spec's committed fallback is 2ec, which is a separate plan; do not improvise.

- [ ] **Step 4: Record the numbers in the `commute_ratio` docstring** — append one line to the docstring stating the measured `corr(internal, access)`, BIG/TINY values, and duplication-suite outcome (so the decision is auditable in code).

- [ ] **Step 5: Commit**

```bash
git add src/reblock/budget.py
git commit -m "docs(budget): record commute_ratio corpus-gate numbers (loading/orthogonality/gaming)"
```

---

### Task 2: `commute_ratio_benefit` + wire the frontier reporting

**Files:**
- Modify: `src/reblock/budget.py` (add `commute_ratio_benefit`)
- Modify: `src/reblock/compare.py` (import line ~24, comment line ~38, benefit_fn line ~127)
- Modify: `src/reblock/emit.py` (ylabel line ~244, benefit CSV precision line ~309)
- Modify: `tests/test_commute_ratio.py` (add the benefit-factory test)

**Interfaces:**
- Consumes: `commute_ratio`, `BenefitFactory`, `STREET_TOL`, `cost_benefit_curve` (`budget.py`).
- Produces: `commute_ratio_benefit(block, roads_full, *, tol=STREET_TOL) -> Callable[[GeoDataFrame | None], float]`, consumed by `compare.py` under metric key `"internal_connectivity"`.

- [ ] **Step 1: Write the failing benefit-factory test** (append to `tests/test_commute_ratio.py`)

```python
from reblock.budget import commute_ratio_benefit, cost_benefit_curve


def test_benefit_factory_terminal_matches_metric() -> None:
    block = _block(4, _parcels_at([(20, 40), (40, 40), (60, 40), (80, 40)]))
    roads = _roads([LineString([(10, 0), (10, 60), (90, 60), (90, 0)])])
    f = commute_ratio_benefit(block, roads)
    assert f(roads) == commute_ratio(block, roads)          # factory delegates to the metric
    curve = cost_benefit_curve(block, roads, benefit_fn=commute_ratio_benefit)
    assert curve.benefit[-1] == commute_ratio(block, roads)  # terminal == full-roads metric
    assert all(0.0 <= b < 1.0 for b in curve.benefit)        # do NOT assert monotone (rho isn't)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_commute_ratio.py::test_benefit_factory_terminal_matches_metric -q`
Expected: FAIL — `cannot import name 'commute_ratio_benefit'`.

- [ ] **Step 3: Implement `commute_ratio_benefit`** (add after `commute_ratio` in `src/reblock/budget.py`)

```python
def commute_ratio_benefit(block: Block, roads_full: GeoDataFrame | None, *,
                          tol: float = STREET_TOL) -> Callable[[GeoDataFrame | None], float]:
    """Internal-connectivity benefit factory (shares the access_benefit signature so it plugs into
    cost_benefit_curve(..., benefit_fn=commute_ratio_benefit) and the _sweep frontier). commute_ratio
    is self-contained per prefix; the resulting curve is NON-MONOTONE (see commute_ratio) -- reporting
    compares at matched budget and ranks by terminal value. roads_full/tol are unused, kept for the
    shared BenefitFactory signature."""
    del roads_full, tol

    def f(roads: GeoDataFrame | None) -> float:
        return commute_ratio(block, roads)
    return f
```

- [ ] **Step 4: Run the benefit-factory test to verify it passes**

Run: `pixi run pytest tests/test_commute_ratio.py::test_benefit_factory_terminal_matches_metric -q`
Expected: PASS.

- [ ] **Step 5: Wire `compare.py`** — three edits:
  - Line ~24 import: change `cycle_benefit,` → `commute_ratio_benefit,`.
  - Line ~38 comment: change `internal connectivity (independent cycles per parcel, cycle_benefit)` → `internal connectivity (backup-route redundancy, commute_ratio_benefit)`.
  - Line ~127: change `internal = cost_benefit_curve(block, roads, benefit_fn=cycle_benefit)` → `internal = cost_benefit_curve(block, roads, benefit_fn=commute_ratio_benefit)`.

- [ ] **Step 6: Wire `emit.py`** — two edits:
  - Line ~244 ylabel: change `"internal connectivity (independent cycles per parcel)"` → `"internal connectivity (backup-route redundancy, mean 1 − R/R_geo)"`.
  - Line ~309 benefit CSV precision: in the `frontier_{metric}.csv` writer row, change the benefit format `f"{b:.4f}"` → `f"{b:.6g}"` (so small ρ values don't round to `0.0000`). Leave the road-length `f"{c:.4f}"` and the displacement-CSV writer unchanged.

- [ ] **Step 7: Run the full check**

Run: `pixi run check`
Expected: green (ruff + mypy + pytest all pass). `test_compare.py` still passes because the metric key and filenames are unchanged.

- [ ] **Step 8: Commit**

```bash
git add src/reblock/budget.py src/reblock/compare.py src/reblock/emit.py tests/test_commute_ratio.py
git commit -m "feat: wire internal_connectivity frontier to commute_ratio_benefit"
```

---

### Task 3: Delete the `cycle_density` path + re-author `test_compare` + reconcile docs

**Files:**
- Modify: `src/reblock/budget.py` (delete `cycle_density`, `cycle_benefit`)
- Delete: `tests/test_cycle_density.py`
- Modify: `tests/test_compare.py` (re-author the arterial-vs-clearance internal test)
- Modify: `examples/method-comparison/README.md`, `examples/multiblock/README.md`
- Modify: `/home/gchurchill/.claude/projects/-home-gchurchill-src-reblock/memory/road-structure-metric-basis.md`

**Interfaces:**
- Consumes: `commute_ratio` (for the re-authored test's ground truth).
- Produces: no new symbols; removes `cycle_density`/`cycle_benefit`.

- [ ] **Step 1: Confirm the noding tests are already re-homed** — `tests/test_commute_ratio.py` contains `test_crossing_is_noded_into_a_shared_vertex` (from Task 0) and `test_subdivision_invariance` (a ρ version, from Task 0). These cover `_noded_graph`, so deleting `test_cycle_density.py` loses no `_noded_graph` coverage.

Run: `grep -rn "_noded_graph" src tests`
Expected: `_noded_graph` is called by `commute_ratio` (and its def) in `src/`, and tested in `tests/test_commute_ratio.py` — never caller-less.

- [ ] **Step 2: Delete `cycle_density` and `cycle_benefit`** from `src/reblock/budget.py` (the two functions shown in the spec §3.5 / current lines ~719–746). Delete the file `tests/test_cycle_density.py`.

```bash
git rm tests/test_cycle_density.py
```

- [ ] **Step 3: Verify nothing else imports them**

Run: `grep -rn "cycle_density\|cycle_benefit" src tests`
Expected: **no matches** (arterial's objective is directness/efficiency, not cycles; only compare/emit referenced them and were re-wired in Task 2).

- [ ] **Step 4: Measure the true ρ ordering for the `test_compare` fixture** — from the metric function, NOT the rounded CSV:

Create `/tmp/measure_ordering.py`:

```python
import sys
sys.path.insert(0, "/home/gchurchill/src/reblock")
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from reblock.pipeline import build_regions
from reblock.budget import commute_ratio

BIDS = ["ZAF.9.3.1_1_40972", "ZAF.9.3.1_1_39229"]  # the two adjacent blocks the test region uses
with initialize_config_dir(version_base=None, config_dir="/home/gchurchill/src/reblock/conf"):
    cfg = compose(config_name="compare_config", overrides=[
        "data=capetown_full", "max_blocks=2", f"block_ids=[{BIDS}]"])
region = build_regions(instantiate(cfg.data), instantiate(cfg.screen),
                       instantiate(cfg.region_builder), [BIDS], 2)[0]
for name in ["clearance", "greedy_arterial_buildable"]:
    m = instantiate(cfg.all_methods[name])
    vals = [commute_ratio(b, m.propose(b).roads) for b in region]
    print(name, "per-block rho terminal:", [round(v, 5) for v in vals], "mean:", round(sum(vals) / len(vals), 5))
```

Run: `pixi run python /tmp/measure_ordering.py`
Expected: prints each method's terminal ρ. **Record which method wins.**

- [ ] **Step 5: Re-author `tests/test_compare.py`** — the test `test_compare_two_adjacent_block_region_arterial_beats_clearance_internal_connectivity` (reads `frontier_internal_connectivity.csv` via `_terminal_benefit_by_method`, lines ~82–102). Change the assertion to match the **measured** Step-4 ordering, reading terminal benefit from the un-rounded metric where possible. If clearance wins under ρ (plausible — arterial's loops are few at region scale), **rename** the test to `..._clearance_beats_arterial_internal_connectivity` and flip the assertion. Do NOT tune ρ to preserve the old direction.

- [ ] **Step 6: Run the check**

Run: `pixi run check`
Expected: green.

- [ ] **Step 7: Reconcile the example READMEs + memory** — replace the stale internal-connectivity text (the metric swap regenerates the PNGs under a new scale; this task only fixes the prose/tables so nothing ships self-contradictory):
  - `examples/method-comparison/README.md`: the `internal connectivity — independent cycles/parcel` table row and the "greedy_arterial owns internal connectivity (0.015…)" prose → relabel to "internal connectivity (backup-route redundancy, mean 1 − R/R_geo)" and mark the numbers "regenerate on next `compare` run" (or regenerate if cheap).
  - `examples/multiblock/README.md`: same for its internal-connectivity row + "wins internal connectivity (0.0072…)" prose.
  - Memory note `road-structure-metric-basis.md`: change "the internal representative = cycle density" wording to note it is now `commute_ratio` (ρ) per the shipped migration.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: delete cycle_density path; re-author test_compare + reconcile docs for rho"
```

---

## Self-Review

**1. Spec coverage:** §3.1 metric + semantics → Task 0. §3.2 gate + committed exit branch + gaming vectors → Task 1. §3.3/§3.3.1 dense solve + guards + clip + line-proximity entries → Task 0; benefit factory → Task 2. §3.4 wiring (compare/emit/comment/CSV) → Task 2. §3.5 migration + re-home noding tests + test_compare + README/memory reconciliation → Tasks 0 (re-home) + 3. §5 scalability gate → Task 0 Step 6. §6 phasing (Task-0/Task-1 gates first) → honored. Deferred to Plan 2: the refiner (§4). Deferred/flagged: frozen entries (see the DEVIATION note — YAGNI per de-risk).

**2. Placeholder scan:** no TBD/TODO; all code blocks concrete; the gate script (Task 1 Step 1) references the committed `spike_gate4.py` pattern rather than re-listing it, which is acceptable for a scratch analysis step, but the five measurements it must print are enumerated exactly.

**3. Type consistency:** `commute_ratio(block, roads) -> float`, `commute_ratio_benefit(block, roads_full, *, tol) -> Callable[[GeoDataFrame|None], float]` are consistent across Tasks 0/1/2/3 and match the spec/`BenefitFactory` seam. Metric key `"internal_connectivity"` unchanged throughout.
