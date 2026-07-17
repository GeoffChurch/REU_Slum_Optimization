# Metric-basis Reporting Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the five entangled benefit lenses (access, efficiency, directness, resistance, displacement) with the validated orthogonal basis — External connectivity (`access`) + Internal connectivity (`cycle_density`) as the two benefit curves, plus Displacement as the cost — and add a 2D connectivity-plane summary figure.

**Architecture:** `cycle_density` is a new topological metric (circuit rank per parcel over a planarized road∪street graph) added to `budget.py` beside the existing curve machinery; `compare.py` emits three `MethodCurve`s per (region, method) instead of five; `emit.py` renders the three per-metric artifacts (unchanged, name-generic plumbing) plus a new connectivity-plane figure; the deleted efficiency/directness/resistance *reporting* engine is removed wholesale. The arterial method's `objective=directness/efficiency` continues to use `network_efficiency`/`_BlockScoringContext.score`, which are NOT deleted.

**Tech Stack:** Python 3.11, geopandas/shapely, networkx, numpy/scipy, matplotlib, Hydra, pixi.

## Global Constraints

- **Migrate, not accommodate:** DELETE `resistance_benefit`, `_resistance_core`, `_BlockScoringContext.resistance_frozen`, `_BlockScoringContext._ground_indices`, `efficiency_directness_curves`, `efficiency_benefit`, `directness_benefit`, and `_efficiency_factory` (iff unused after), the whole of `tests/test_resistance.py`, and the old `compare_efficiency*/compare_directness*/compare_resistance*` (+ `frontier_efficiency/directness/resistance.csv`) example artifacts. No back-compat shims.
- **KEEP** `network_efficiency`, `_BlockScoringContext.score`/`.score_frozen`, `_sampled_efficiency_core`, `_build_csr`, `_line_entries` — arterial's `objective=directness|efficiency` depends on them.
- `pixi run check` (lint + typecheck + test) must stay green after every task.
- Ruff bans: semicolons (E702), >100-char lines (E501), `zip()` without `strict=` (B905).
- Internal metric normalization is **per parcel** `(E − N + C) / P`, P = `len(block.parcels)` (topological-invariance rationale in the spec — do not use per-graph-node `/N`).
- Metric names in `MethodCurve.metric`: exactly `"external_connectivity"`, `"internal_connectivity"`, `"displacement"`.

## File Structure

- `src/reblock/budget.py` — ADD `_noded_graph`, `cycle_density`, `cycle_benefit`; DELETE the resistance + efficiency/directness reporting functions.
- `src/reblock/compare.py` — emit three metrics; swap imports.
- `src/reblock/emit.py` — `_METRIC_YLABELS` for the new names; ADD `_connectivity_plane`; call it from `compare_report`.
- `tests/test_cycle_density.py` — NEW, unit tests for the metric + noded builder.
- `tests/test_resistance.py` — DELETE.
- `tests/test_compare.py` / `tests/test_emit.py` — update for the 3-metric set + the plane (grep for the existing tests; adjust the expected metric names/artifacts).
- `examples/method-comparison/`, `examples/multiblock/` — regenerate (final task).

---

### Task 1: Cycle-density metric + noded-graph builder (`budget.py`)

**Files:**
- Modify: `src/reblock/budget.py` (add near `cost_benefit_curve`, ~line 860)
- Test: `tests/test_cycle_density.py` (create)

**Interfaces:**
- Produces: `cycle_density(block: Block, roads: GeoDataFrame | None) -> float`; `cycle_benefit(block: Block, roads_full: GeoDataFrame | None, *, tol: float = STREET_TOL) -> Callable[[GeoDataFrame | None], float]` (matches the `access_benefit` factory signature so it plugs into `cost_benefit_curve(..., benefit_fn=cycle_benefit)`); `_noded_graph(roads: GeoDataFrame, streets: GeoDataFrame) -> nx.Graph`.
- Consumes: `_rnd` (budget.py:36), `nx` (already imported), `unary_union` (already imported).

- [ ] **Step 0: Re-confirm per-parcel loading (scratch, not committed).** Write a throwaway script that, on block `ZAF.9.3.1_1_40972`, computes both `(E−N+C)/P` and `(E−N+C)/N_graph` on the noded graph for the six methods `clearance` at repulsion {−3,0,3}, `greedy_arterial_buildable`, `greedy_arterial_aspirational` (skip topology for speed), and asserts `scipy.stats.spearmanr(perP, perN).statistic > 0.9`. Expected: ρ ≈ 1 (per-parcel `∝` circuit rank, highly correlated with per-node). If ρ ≤ 0.9, STOP and escalate — the per-parcel choice would not track the validated axis. (Use the block-build pattern from `scratchpad/spike_metric_basis.py`.) Delete the scratch script after.

- [ ] **Step 1: Write the failing tests** — `tests/test_cycle_density.py`:

```python
# tests/test_cycle_density.py
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.budget import _noded_graph, cycle_density
from reblock.contracts import Block

UTM = CRS.from_epsg(32734)


def _block(n_parcels: int) -> Block:
    # A square block whose `streets` is the south edge; parcel count controls the /P denominator.
    boundary = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(n_parcels))},
                               geometry=[boundary] * n_parcels, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (100, 0)])], crs=UTM)
    return Block(block_id="b", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _roads(lines: list[LineString]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=lines, crs=UTM)


def test_tree_has_zero_cycles() -> None:
    # A path touching the street: no loop -> circuit rank 0.
    block = _block(4)
    roads = _roads([LineString([(50, 0), (50, 40), (70, 40)])])
    assert cycle_density(block, roads) == 0.0


def test_single_loop_is_one_over_parcels() -> None:
    # A closed square interior road = one independent cycle; /P with P=4 -> 0.25.
    block = _block(4)
    loop = LineString([(20, 20), (60, 20), (60, 60), (20, 60), (20, 20)])
    assert cycle_density(block, _roads([loop])) == 0.25


def test_two_disjoint_loops_have_circuit_rank_two() -> None:
    block = _block(8)   # P=8 -> 2/8 = 0.25
    a = LineString([(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)])
    b = LineString([(60, 60), (80, 60), (80, 80), (60, 80), (60, 60)])
    assert cycle_density(block, _roads([a, b])) == 0.25


def test_crossing_is_noded_into_a_shared_vertex() -> None:
    # Two crossing roads that individually have no loop: planarizing nodes the X into 4 edges sharing
    # the centre. With the street edge they still form no cycle here, but the crossing MUST create a
    # degree-4 centre node (5 nodes, 4 edges, 1 component -> circuit rank 0), proving noding happened.
    block = _block(4)
    roads = _roads([LineString([(20, 20), (80, 80)]), LineString([(20, 80), (80, 20)])])
    g = _noded_graph(roads, block.streets)
    assert (50.0, 50.0) in g.nodes            # the crossing became a shared vertex
    assert g.degree[(50.0, 50.0)] == 4


def test_subdivision_invariance() -> None:
    # Circuit rank is a topological invariant: adding a mid-vertex to a loop edge must not change it.
    block = _block(4)
    loop = LineString([(20, 20), (60, 20), (60, 60), (20, 60), (20, 20)])
    subdivided = LineString([(20, 20), (40, 20), (60, 20), (60, 60), (20, 60), (20, 20)])
    assert cycle_density(block, _roads([loop])) == cycle_density(block, _roads([subdivided]))


def test_empty_roads_zero() -> None:
    assert cycle_density(_block(4), _roads([])) == 0.0
    assert cycle_density(_block(4), None) == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_cycle_density.py -q`
Expected: FAIL (`ImportError: cannot import name '_noded_graph'`).

- [ ] **Step 3: Implement** — add to `src/reblock/budget.py` immediately after `cost_benefit_curve` (~line 868). `GeoDataFrame | None`, `nx`, `unary_union`, `_rnd`, `Block`, `STREET_TOL`, `Callable` are all already imported.

```python
def _noded_graph(roads: GeoDataFrame, streets: GeoDataFrame) -> nx.Graph:
    """The PLANARIZED road∪street graph: unary_union nodes every crossing/touch into shared
    vertices, then each _rnd-snapped (2-dp) segment becomes one undirected edge (deduped). Non-
    LineString union fragments (stray points) are skipped. Empty input -> empty graph."""
    geoms = list(roads.geometry) + list(streets.geometry)
    g: nx.Graph = nx.Graph()
    if not geoms:
        return g
    merged = unary_union(geoms)
    parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    for part in parts:
        if not hasattr(part, "coords"):
            continue
        cs = [_rnd(c) for c in part.coords]
        for a, b in zip(cs, cs[1:], strict=False):
            if a != b:
                g.add_edge(a, b)
    return g


def cycle_density(block: Block, roads: GeoDataFrame | None) -> float:
    """Internal connectivity: circuit rank per parcel, (E - N + C) / P, over the noded road∪street
    graph (E/N/C = edge/node/component counts, P = parcel count). The number of independent cycles
    (redundant internal routes) per dwelling; a tree -> 0. Circuit rank is a topological invariant
    (subdivision-insensitive), so /P (fixed, exogenous) keeps the whole metric discretization-
    invariant. 0.0 with no roads / no parcels / an empty graph."""
    p = len(block.parcels)
    if roads is None or len(roads) == 0 or p < 1:
        return 0.0
    g = _noded_graph(roads, block.streets)
    n = g.number_of_nodes()
    if n == 0:
        return 0.0
    circuit_rank = g.number_of_edges() - n + nx.number_connected_components(g)
    return circuit_rank / p


def cycle_benefit(block: Block, roads_full: GeoDataFrame | None, *,
                  tol: float = STREET_TOL) -> Callable[[GeoDataFrame | None], float]:
    """Internal-connectivity benefit factory (shares the `access_benefit` signature so it plugs into
    `cost_benefit_curve(..., benefit_fn=cycle_benefit)` and the `_sweep` frontier). `roads_full`/`tol`
    are unused -- cycle_density is self-contained and needs no frozen entries -- but kept for the
    shared BenefitFactory signature."""
    del roads_full, tol

    def f(roads: GeoDataFrame | None) -> float:
        return cycle_density(block, roads)
    return f
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run pytest tests/test_cycle_density.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Lint + typecheck the new code**

Run: `pixi run lint && pixi run typecheck`
Expected: clean / Success.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/budget.py tests/test_cycle_density.py
git commit -m "feat(budget): cycle_density internal-connectivity metric + noded-graph builder"
```

---

### Task 2: Emit the three-metric set (`compare.py` + `emit.py`)

**Files:**
- Modify: `src/reblock/compare.py` (imports ~20-27; the curve block 127-135)
- Modify: `src/reblock/emit.py` (`_METRIC_YLABELS` 242-248; `compare_report` docstring 270-285)
- Test: `tests/test_compare.py` (grep for the existing compare/emit test; adjust)

**Interfaces:**
- Consumes: `cycle_benefit`, `access_benefit`, `cost_benefit_curve`, `displacement_curve` from `budget`.
- Produces: `MethodCurve.metric` ∈ {`"external_connectivity"`, `"internal_connectivity"`, `"displacement"`}.

- [ ] **Step 1: Update `compare.py` imports.** Replace the `from reblock.budget import (...)` block (lines ~20-27) so it imports `cycle_benefit` and drops `efficiency_directness_curves` + `resistance_benefit`:

```python
from reblock.budget import (
    Curve,
    access_benefit,
    building_radii,
    cost_benefit_curve,
    cycle_benefit,
    displacement_curve,
)
```

- [ ] **Step 2: Replace the five-curve block** in `compare.py` (lines 127-135) with the three-metric block:

```python
            external = cost_benefit_curve(block, roads, benefit_fn=access_benefit)
            internal = cost_benefit_curve(block, roads, benefit_fn=cycle_benefit)
            disp = displacement_curve(block, roads, radii, corridor_m=corridor_m)
            raw.append((name, label, "external_connectivity", external, pp, pd_))
            raw.append((name, label, "internal_connectivity", internal, pp, pd_))
            raw.append((name, label, "displacement", disp, pp, pd_))
```

- [ ] **Step 3: Update `emit.py` `_METRIC_YLABELS`** (lines 242-248) to the new metric names:

```python
_METRIC_YLABELS = {
    "external_connectivity": "external connectivity (fraction of access-burden removed)",
    "internal_connectivity": "internal connectivity (independent cycles per parcel)",
    "displacement": "buildings displaced (Σ disk-graze probability)",
}
```

Also update the `compare_report` docstring (line ~272) list "access, efficiency, directness, resistance, displacement" -> "external_connectivity, internal_connectivity, displacement", and the phrase "the four benefit metrics" -> "the two benefit metrics". (The frontier-CSV / `curve_{metric}_{block}.png` naming is metric-name-generic, so no other emit changes are needed.)

- [ ] **Step 4: Update the compare/emit test.** Grep `grep -rln "efficiency\|directness\|resistance\|_METRIC_YLABELS\|external_connectivity\|MethodCurve" tests/` to find the compare/emit test(s). In them, replace any assertion over the old five metric names with the three new names (`external_connectivity`, `internal_connectivity`, `displacement`), and any expected artifact filename `curve_efficiency*/directness*/resistance*` / `frontier_efficiency*` with `curve_external_connectivity*` / `curve_internal_connectivity*` / `frontier_external_connectivity.csv` / `frontier_internal_connectivity.csv`.

- [ ] **Step 5: Run the compare/emit tests + a smoke compare**

Run: `pixi run pytest tests/test_compare.py -q` (or the file(s) found in Step 4)
Expected: PASS.
Run: `pixi run compare data=kblock shapefile=x methods=[clearance] max_blocks=1 render.enabled=false 2>&1 | tail -5` (use the existing sample fixture; confirm no crash and that `frontier_external_connectivity.csv` / `frontier_internal_connectivity.csv` appear in the run dir).

- [ ] **Step 6: Commit**

```bash
git add src/reblock/compare.py src/reblock/emit.py tests/
git commit -m "feat(compare): emit External/Internal connectivity + displacement (3 metrics, was 5)"
```

---

### Task 3: 2D connectivity-plane summary figure (`emit.py`)

**Files:**
- Modify: `src/reblock/emit.py` (add `_connectivity_plane`; call it from `compare_report`)
- Test: `tests/test_emit.py` (or the emit test found in Task 2)

**Interfaces:**
- Consumes: the `MethodCurve` list passed to `compare_report` (each has `.method`, `.block_id`, `.metric`, `.curve` where `curve.benefit` is the per-prefix value list and `curve.cost` the per-prefix road length).
- Produces: one `connectivity_plane_{block_id}.png` per block/region.

- [ ] **Step 1: Write the failing test** — in the emit test file:

```python
def test_connectivity_plane_written(tmp_path):
    from reblock.budget import Curve
    from reblock.compare import MethodCurve
    from reblock.emit import _connectivity_plane
    ext = Curve([0.0, 100.0], [0.0, 0.6])
    inn = Curve([0.0, 100.0], [0.0, 0.3])
    rows = [MethodCurve("clearance", "blk", "external_connectivity", ext),
            MethodCurve("clearance", "blk", "internal_connectivity", inn)]
    _connectivity_plane(rows, tmp_path)
    assert (tmp_path / "connectivity_plane_blk.png").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_emit.py::test_connectivity_plane_written -q`
Expected: FAIL (`cannot import name '_connectivity_plane'`).

- [ ] **Step 3: Implement `_connectivity_plane`** in `emit.py` (place near `compare_report`; reuse the existing `save_render` helper and the method-colour helper the module already uses for curves — grep `method_color`/colour keying in emit.py and reuse it; if none, fall back to `plt.cm.tab10`):

```python
def _connectivity_plane(results: list["MethodCurve"], out_dir: Path) -> None:
    """Per block/region: each method's trajectory through (external, internal) connectivity space as
    road grows (marker size grows with cumulative road length). A communication figure, not a metric
    -- no scalar summary. Pairs the `external_connectivity` and `internal_connectivity` curves per
    (block, method); methods missing either are skipped."""
    paired: dict[str, dict[str, dict[str, "MethodCurve"]]] = {}
    for r in results:
        if r.metric in ("external_connectivity", "internal_connectivity"):
            paired.setdefault(r.block_id, {}).setdefault(r.method, {})[r.metric] = r
    for block_id, by_method in paired.items():
        fig, ax = plt.subplots(figsize=(8, 6.5))
        for i, (method, mc) in enumerate(sorted(by_method.items())):
            ext, inn = mc.get("external_connectivity"), mc.get("internal_connectivity")
            if ext is None or inn is None:
                continue
            x, y, cost = ext.curve.benefit, inn.curve.benefit, ext.curve.cost
            colour = plt.cm.tab10(i % 10)
            ax.plot(x, y, "-", color=colour, lw=1.2, alpha=0.6)
            cmax = max(cost) or 1.0
            sizes = [20 + 120 * (c / cmax) for c in cost]
            ax.scatter(x, y, s=sizes, color=colour, label=method, zorder=3,
                       edgecolors="white", linewidths=0.4)
        ax.set_xlabel("External connectivity (reach to the street)")
        ax.set_ylabel("Internal connectivity (cycles per parcel)")
        ax.set_title(f"connectivity plane: {block_id}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        save_render(fig, out_dir / f"connectivity_plane_{block_id}.png")
```

(Note: `mc` values are `MethodCurve`s; `.curve.benefit`/`.curve.cost` are the lists. Adjust the attribute access if `MethodCurve` stores the `Curve` under a different name — it's `.curve` per `compare.py:44-48`.)

- [ ] **Step 4: Call it from `compare_report`.** At the end of `compare_report` (after the per-metric loop that ends ~line 327), add `_connectivity_plane(results, out_dir)`.

- [ ] **Step 5: Run to verify pass + no regression**

Run: `pixi run pytest tests/test_emit.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/emit.py tests/
git commit -m "feat(emit): 2D connectivity-plane summary figure per block"
```

---

### Task 4: Delete the efficiency/directness/resistance reporting engine

**Files:**
- Modify: `src/reblock/budget.py` (delete the listed functions + now-unused `_BlockScoringContext` resistance state + now-unused imports)
- Delete: `tests/test_resistance.py`

**Interfaces:** none produced; this only removes code. Precondition: Tasks 2-3 already removed every *reporting* reference to these symbols.

- [ ] **Step 1: Delete `tests/test_resistance.py`**

```bash
git rm tests/test_resistance.py
```

- [ ] **Step 2: Delete the functions in `budget.py`** (each in full, with its docstring):
`_resistance_core` (~206-308), `resistance_frozen` (method, ~551-570), `_ground_indices` (method, ~538-549), `_efficiency_factory` (~705-727), `efficiency_benefit` (~748-751), `directness_benefit` (~754-757), `resistance_benefit` (~760-785), `efficiency_directness_curves` (~871-877). Also delete the `_BlockScoringContext.__init__` state used ONLY by resistance: the `self.cap` and `self.streets_geom` assignments and their computing lines (the `bounds`/`streets_geom` block ~476-478) — after confirming (grep) they are referenced nowhere else.

- [ ] **Step 3: Fix now-unused imports.** After deletion, `factorized`, `diags`, `connected_components`, `dijkstra` (from scipy), and any others may be unused in `budget.py`. Run `pixi run lint` and remove exactly what ruff (F401) reports — do not remove imports still used by the KEPT scoring core (`_sampled_efficiency_core` uses `dijkstra`; verify per ruff, don't guess).

- [ ] **Step 4: Grep-confirm no dangling references**

Run: `grep -rn "resistance_benefit\|_resistance_core\|resistance_frozen\|_ground_indices\|efficiency_directness_curves\|efficiency_benefit\|directness_benefit\|_efficiency_factory" src/ tests/`
Expected: no matches (empty).

- [ ] **Step 5: Full check**

Run: `pixi run check`
Expected: lint clean, `mypy Success`, all tests pass (357 minus the deleted resistance tests, plus the new cycle-density/plane tests).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(budget): delete efficiency/directness/resistance reporting engine (migrated to metric basis)"
```

---

### Task 5: Regenerate both flagship examples (FINAL — compute-heavy)

**Files:**
- Modify: `examples/method-comparison/` and `examples/multiblock/` (regenerate artifacts + READMEs)

**Interfaces:** none; produces the shipped example artifacts.

- [ ] **Step 1: Delete stale metric artifacts**

```bash
git rm examples/method-comparison/compare_efficiency.png examples/method-comparison/compare_directness.png examples/method-comparison/compare_resistance.png examples/method-comparison/frontier_efficiency.csv examples/method-comparison/frontier_directness.csv examples/method-comparison/frontier_resistance.csv 2>/dev/null || true
git rm examples/multiblock/compare_efficiency.png examples/multiblock/compare_directness.png examples/multiblock/compare_resistance.png examples/multiblock/frontier_efficiency.csv examples/multiblock/frontier_directness.csv examples/multiblock/frontier_resistance.csv 2>/dev/null || true
```
(Use `ls examples/*/` first to confirm the exact stale filenames; the curve plots may be `curve_{metric}_{block}.png` rather than `compare_{metric}.png` — delete whichever the deleted metrics produced.)

- [ ] **Step 2: Regenerate `examples/method-comparison`.** Read its `README.md` for the exact reproduce command; run it (it will now emit `frontier_external_connectivity.csv`, `frontier_internal_connectivity.csv`, `frontier_displacement`/`displacement_vs_length.csv`, the `curve_external_connectivity_*`/`curve_internal_connectivity_*`/`displacement_*` plots, `connectivity_plane_*`, and the matched-budget renders). Copy the produced artifacts into the example dir.

- [ ] **Step 3: Regenerate `examples/multiblock`** the same way from its README reproduce command (region-scale; the slow one — topology on the deep block + region clearance/arterial; expect minutes). Run in the background via `pixi run compare ... > /tmp/mb.log 2>&1 &` and poll.

- [ ] **Step 4: Update both READMEs.** Replace the five-lens description with the three-metric basis (External connectivity = access; Internal connectivity = independent cycles per parcel; Displacement = cost), reference the new `connectivity_plane_*.png`, drop the deleted-metric figures/tables, and update the reproduce commands (they need no metric flags — the compare emits the fixed three).

- [ ] **Step 5: Verify the suite is still green**

Run: `pixi run check`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "docs(examples): regenerate for the metric basis (External/Internal connectivity + displacement + connectivity plane)"
```

---

## Notes for the executor
- **Branch:** `metric-basis-refactor` (already created, off `fix-mypy-typecheck`). Rebase onto `main` after PR #3 (mypy fix) merges — do not merge this before that.
- Between Tasks 1 and 2 the code is green but the metric is unused; that's fine (Task 1 is independently testable).
- The deletion (Task 4) MUST follow Tasks 2-3 so nothing imports the removed symbols mid-flight.
