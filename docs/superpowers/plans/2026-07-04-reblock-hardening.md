# reblock Architecture Hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Fold the post-red-team hardening into reblock: per-parcel `Metrics.fields` + `Result`, a robust BFS-peel access metric (primary) with weak-dual retained (optional), trimmed contracts, fail-loud data loading, topology `trace_faces`/`maxdepth` fixes, real Hydra pluggability, and the `render` component.

**Architecture:** The canonical `Block` waist stays. The access metric moves off topology's fragile weak-dual onto a parcel-native BFS peel (`reblock.derive.access`), so eval + render are topology-independent and per-parcel by `parcel_id`. `Metrics` gains a per-parcel `fields` channel; a `Result` bundles block+proposal+metrics. Runner becomes Hydra-config-group pluggable over method/eval lists.

**Tech Stack:** Python ≥3.11, geopandas/shapely 2.x, networkx, pandas, Hydra, pixi/conda-forge, pytest, mypy --strict, ruff. Branch: `hardening`.

## Global Constraints

- Python `>=3.11,<3.13`; conda deps via pixi/conda-forge (linux-64); `mypy --strict` clean on `src`/`tests`; ruff `["E","F","I","UP","B"]`, line-length 100.
- **Migrate, don't accommodate** (owner rule): replace old paths, delete removed fields/logic outright, migrate existing tests. No back-compat/dual-path branches.
- Blocks are projected local-UTM; per-parcel data keyed by `parcel_id` (never row position).
- Two-repo: Task 6 edits the `ext/topology` submodule (branch `hardening-topology` off `master`), then reblock repins.
- TDD; `pixi run check` (ruff + mypy --strict + pytest) green at each task end.
- Spec: `docs/superpowers/specs/2026-07-04-reblock-hardening-design.md`.

---

## File structure

```
src/reblock/
  contracts.py            # REWORK: Metrics.fields, Result, Method(prior)+RegionMethod, trim fields, proposal_id
  derive/access.py        # NEW: parcel_access_layers(block, roads) -> pd.Series[int]  (BFS peel)
  eval/kcomplexity.py     # REWORK: peel-based KComplexityEval (emits fields); + WeakDualKEval (optional)
  data/shapefile.py       # REWORK: explicit assumed_crs, explode multi-part records, lazy blocks
  methods/topology.py     # REWORK: propose(block, prior=None); consume Block.streets; params->proposal_id
  render.py               # NEW: render_before / render_after / save_render
  run.py                  # REWORK: Result output, method/eval lists, Hydra config groups, render_dir
conf/                     # NEW: data/*.yaml, method/*.yaml, eval/*.yaml, config.yaml
ext/topology/topology/graph/my_graph.py   # FIX trace_faces (outer face by signed area) + maxdepth loud
```

Task order is dependency-driven: **1 contracts → 2 peel → 3 eval → 4 loading → 5 topology-fix → 6 method → 7 render → 8 run/hydra.**

---

### Task 1: Contracts v2

**Files:** Modify `src/reblock/contracts.py`; Modify `tests/test_contracts.py`.

**Interfaces — Produces:**
- `Metrics(block_id, method, eval, values: Mapping[str,float], fields: Mapping[str, "pd.Series"] = {})`.
- `Result(block: Block, proposal: Proposal, metrics: tuple[Metrics, ...])` with `metric(eval, key) -> float`.
- `Proposal(..., proposal_id: str = "", method: str = "", params=...)` (roads/edges kept; `water_points`/`water_mains` removed).
- `Method.propose(block, prior: Proposal | None = None) -> Proposal`; new `RegionMethod.propose(region) -> Iterable[Proposal]`.
- `Region` loses `water`/`food`/`healthcare`; `Block` loses `buildings`/`water`/`barriers`; `Screen` removed.

- [ ] **Step 1: Migrate the tests first** (`tests/test_contracts.py`). Keep the existing validation tests (projected-CRS, missing `parcel_id`, empty parcels, missing geometry). Add:

```python
import pandas as pd
from reblock.contracts import Metrics, Proposal, Result

def test_metrics_carries_per_parcel_fields() -> None:
    s = pd.Series([1, 2], index=pd.Index([0, 1], name="parcel_id"))
    m = Metrics(block_id="b", method="topology", eval="kcomplexity",
                values={"k_before": 2.0}, fields={"access_after": s})
    assert m.fields["access_after"].loc[1] == 2

def test_result_metric_lookup(_block, _proposal) -> None:   # reuse block/proposal fixtures
    m = Metrics(block_id="b", method="topology", eval="kcomplexity",
                values={"delta_k": 3.0}, fields={})
    r = Result(block=_block, proposal=_proposal, metrics=(m,))
    assert r.metric("kcomplexity", "delta_k") == 3.0
```

Also delete any test references to removed fields (`Region.water`, `Block.buildings`, `Proposal.water_points`, `Screen`).

- [ ] **Step 2: Run to verify fail.** `pixi run pytest tests/test_contracts.py -v` → FAIL (`Metrics`/`Result` shape).

- [ ] **Step 3: Rework `contracts.py`.** Keep `_require_columns`/`_require_projected`/`Block.__post_init__`/`Source`/`Eval` as-is. Apply: `Region` drops `water/food/healthcare`; `Block` drops `buildings/water/barriers`; `Proposal` drops `water_points/water_mains`, gains `proposal_id: str = ""`; delete `Screen`; add:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd

@dataclass(frozen=True)
class Metrics:
    block_id: str
    method: str
    eval: str
    values: Mapping[str, float]
    fields: Mapping[str, "pd.Series"] = field(default_factory=dict)

@dataclass(frozen=True)
class Result:
    block: "Block"
    proposal: "Proposal"
    metrics: tuple[Metrics, ...]
    def metric(self, eval: str, key: str) -> float:
        for m in self.metrics:
            if m.eval == eval:
                return m.values[key]
        raise KeyError(f"no metric {key!r} for eval {eval!r}")

class Method(Protocol):
    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal: ...

class RegionMethod(Protocol):
    def propose(self, region: Region) -> Iterable[Proposal]: ...
```

Add `pandas` to `[tool.pixi.dependencies]` if not resolvable, and `types-pandas`/`pandas-stubs` to dev deps if mypy needs it (mirror the geopandas-stubs precedent).

- [ ] **Step 4: Run to verify pass.** `pixi run pytest tests/test_contracts.py -v` → PASS.
- [ ] **Step 5: Verify no other module references removed fields.** `pixi run check` → will surface breakages in eval/method/run (fixed in their tasks) ONLY if they reference removed names; the removed fields (`water_points` etc.) aren't referenced by current code (verified), so check should pass. If it flags a removed-field use, note it for that task.
- [ ] **Step 6: Commit.** `git add -A && git commit -m "feat: contracts v2 — Metrics.fields, Result, Method(prior)+RegionMethod, trim speculative fields"`

---

### Task 2: BFS-peel access derivation

**Files:** Create `src/reblock/derive/access.py`; Create `tests/derive/test_access.py`.

**Interfaces — Produces:** `parcel_access_layers(block: Block, roads: GeoDataFrame | None, *, tol: float = 0.5) -> "pd.Series[int]"` — access layer (1 = street-adjacent, …) per parcel, **indexed by `block.parcels["parcel_id"]`**. `k = int(series.max())`. Consumes: `Block` (Task 1).

- [ ] **Step 1: Write failing tests** `tests/derive/test_access.py`:

```python
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon
from reblock.contracts import Block
from reblock.derive.access import parcel_access_layers

UTM = CRS.from_epsg(32643)

def _grid_block(n: int, x0: float = 0.0) -> Block:
    polys, ids = [], []
    for i in range(n):
        for j in range(n):
            polys.append(Polygon([(x0+i, j), (x0+i+1, j), (x0+i+1, j+1), (x0+i, j+1)]))
            ids.append(i * n + j)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    boundary = parcels.geometry.union_all()
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)

def test_2x2_all_on_street() -> None:
    assert parcel_access_layers(_grid_block(2), None).max() == 1

def test_3x3_centre_is_layer_2() -> None:
    layers = parcel_access_layers(_grid_block(3), None)
    assert layers.max() == 2
    assert (layers == 2).sum() == 1        # exactly the centre parcel (id 4)
    assert layers.loc[4] == 2

def test_strip_is_honest_not_degenerate() -> None:
    # 1xN strip, only the far-left parcel touches the (left-edge) street -> depth N
    polys = [Polygon([(i, 0), (i+1, 0), (i+1, 1), (i, 1)]) for i in range(5)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(5))}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])], crs=UTM)
    block = Block(block_id="s", crs=UTM, boundary=parcels.geometry.union_all(),
                  parcels=parcels, streets=streets)
    assert parcel_access_layers(block, None).max() == 5     # weak-dual wrongly gives 1

def test_indexed_by_parcel_id_survives_reorder() -> None:
    layers = parcel_access_layers(_grid_block(3), None)
    assert layers.index.name == "parcel_id"
    assert layers.loc[4] == 2                                # by id, not position

def test_nonzero_origin() -> None:
    assert parcel_access_layers(_grid_block(3, x0=1000.0), None).max() == 2

def test_added_road_reduces_depth() -> None:
    block = _grid_block(3)
    connector = gpd.GeoDataFrame(geometry=[LineString([(1, 0), (1, 1)])], crs=UTM)
    assert parcel_access_layers(block, connector).max() == 1   # centre now reached
```

- [ ] **Step 2: Run to verify fail.** `pixi run pytest tests/derive/test_access.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `src/reblock/derive/access.py`.**

```python
"""Access-depth via a BFS parcel peel (kblock's k-complexity definition).

Layer 1 = touches a street; layer L = L-1 parcels from the nearest street.
Runs on parcels directly (STRtree adjacency + BFS) — no topology graph, so it
is robust to the weak-dual degeneracies and native per-parcel by parcel_id.
"""
from __future__ import annotations

import pandas as pd
from geopandas import GeoDataFrame
from shapely import STRtree

from reblock.contracts import Block


def _adjacency(geoms: list, tol: float) -> list[set[int]]:
    tree = STRtree(geoms)
    adj: list[set[int]] = [set() for _ in geoms]
    for i, g in enumerate(geoms):
        gi = g.buffer(tol) if tol else g
        for j in tree.query(gi):
            j = int(j)
            if i < j and gi.intersection(geoms[j].buffer(tol) if tol else geoms[j]).area >= 0 \
               and g.buffer(tol).intersects(geoms[j]) and i != j:
                # share a boundary within tolerance (covers pinch-point gaps)
                if g.buffer(tol).intersection(geoms[j]).length > 0 or g.intersects(geoms[j]):
                    adj[i].add(j)
                    adj[j].add(i)
    return adj


def parcel_access_layers(block: Block, roads: GeoDataFrame | None, *, tol: float = 0.5) -> "pd.Series":
    parcels = block.parcels.reset_index(drop=True)
    geoms = list(parcels.geometry)
    ids = list(parcels["parcel_id"])
    adj = _adjacency(geoms, tol)

    seed_lines = list(block.streets.geometry)
    if roads is not None and not roads.empty:
        seed_lines += list(roads.geometry)
    from shapely import union_all
    street = union_all(seed_lines) if seed_lines else None

    layer = [0] * len(geoms)
    frontier = {i for i, g in enumerate(geoms)
                if street is not None and g.buffer(tol).intersects(street)}
    d = 1
    seen: set[int] = set()
    while frontier:
        for i in frontier:
            layer[i] = d
        seen |= frontier
        frontier = {j for i in frontier for j in adj[i] if j not in seen}
        d += 1
    # any parcel unreached (disconnected from a street) gets the deepest layer + 1
    if 0 in layer:
        far = d
        layer = [far if v == 0 else v for v in layer]
    return pd.Series(layer, index=pd.Index(ids, name="parcel_id"), dtype="int64")
```

Note to implementer: the `_adjacency` tolerance test above is deliberately belt-and-suspenders; simplify to the single robust predicate that passes the tests (share positive-length boundary within `tol`), and add a mypy override for this module if `shapely`/`STRtree` typing needs it (mirror `reblock.derive.parcel_graph`).

- [ ] **Step 4: Run to verify pass.** `pixi run pytest tests/derive/test_access.py -v` → PASS (6 tests). If `test_strip_is_honest_not_degenerate` fails, the seed/BFS is wrong — a left-edge street must seed only parcel 0, giving depths 1..5.
- [ ] **Step 5: `pixi run check`** → green.
- [ ] **Step 6: Commit.** `git commit -am "feat: add BFS-peel parcel_access_layers derivation (robust, per-parcel by id)"`

---

### Task 3: Peel-based eval (+ optional weak-dual eval)

**Files:** Modify `src/reblock/eval/kcomplexity.py`; Modify `tests/eval/test_kcomplexity.py`.

**Interfaces — Produces:**
- `KComplexityEval.score(block, proposal) -> Metrics` — **peel-based**: `k_before = max(parcel_access_layers(block, None))`, `k_after = max(parcel_access_layers(block, proposal.roads))`; `values` = `{k_before, k_after, delta_k, added_road_length_m}`; `fields` = `{"access_before": layers_pre, "access_after": layers_post}` (both `pd.Series` by parcel_id).
- `WeakDualKEval.score(...) -> Metrics` — the *current* topology-weak-dual logic, moved verbatim, `eval="weakdual_k"`, no `fields`.

- [ ] **Step 1: Migrate tests.** Rewrite `tests/eval/test_kcomplexity.py` so `KComplexityEval` uses the peel: the existing 3×3 case (`k_before=2`, `k_after=1`, `delta_k=1`, `added_road_length_m=1.0`) still holds (peel == weak-dual on the grid); ADD assertions that `metrics.fields["access_before"]` / `["access_after"]` are per-parcel `pd.Series` indexed by `parcel_id` with `access_after.max() == 1`. Keep the nonzero-origin and no-roads tests. Add a `WeakDualKEval` test pinning the old behavior on the same 3×3 (`k_before=2`).

- [ ] **Step 2: Run to verify fail.** `pixi run pytest tests/eval/test_kcomplexity.py -v` → FAIL.

- [ ] **Step 3: Rework `eval/kcomplexity.py`.** `KComplexityEval` no longer imports topology or `to_parcel_graph`; it calls `parcel_access_layers`:

```python
from reblock.derive.access import parcel_access_layers
class KComplexityEval:
    def score(self, block, proposal):
        pre = parcel_access_layers(block, None)
        post = parcel_access_layers(block, proposal.roads)
        added = float(proposal.roads.geometry.length.sum()) if proposal.roads is not None and not proposal.roads.empty else 0.0
        kb, ka = int(pre.max()), int(post.max())
        return Metrics(block_id=block.block_id, method=proposal.method, eval="kcomplexity",
                       values={"k_before": float(kb), "k_after": float(ka),
                               "delta_k": float(kb - ka), "added_road_length_m": added},
                       fields={"access_before": pre, "access_after": post})
```

Move the **current** `_k`/`_endpoint_keys`/topology-weak-dual logic verbatim into a new `WeakDualKEval` class in the same file (`eval="weakdual_k"`), so the Brelsford metric is retained as an optional eval.

- [ ] **Step 4: Run to verify pass.** `pixi run pytest tests/eval/test_kcomplexity.py -v` → PASS.
- [ ] **Step 5: `pixi run check`** → green (note `test_run.py` may break until Task 8; if so, mark and proceed — its runner still constructs the old `run()`).
- [ ] **Step 6: Commit.** `git commit -am "feat: peel-based KComplexityEval emitting per-parcel fields; retain WeakDualKEval"`

---

### Task 4: Fail-loud, multi-part-safe loading

**Files:** Modify `src/reblock/data/shapefile.py`; Modify `tests/data/test_shapefile_source.py`.

**Interfaces — Produces:** `ShapefileSource(path, region_id="region", *, assumed_crs: CRS | int | None = None)`. `.region()` — raises `ValueError` if the file has no CRS and no `assumed_crs`; **explodes multi-part geometries** to single Polygons at the row level before component grouping; `Region.blocks` is a lazy generator.

- [ ] **Step 1: Migrate/add tests.** Keep the Phule "yields metric blocks" test but pass `assumed_crs=3857` (Phule has no `.prj`). Add: (a) a `.prj`-less shapefile with NO `assumed_crs` raises `ValueError(match="CRS")`; (b) loading `ext/topology/Data/Epworth_Before.shp` (with `assumed_crs=...`) no longer raises and yields ≥1 valid `Block` (the native MultiPolygon record is exploded, not fatal). (Use `assumed_crs=3857` for Epworth too; assert blocks are projected + valid.)

- [ ] **Step 2: Run to verify fail.** `pixi run pytest tests/data/test_shapefile_source.py -v` → FAIL.

- [ ] **Step 3: Rework `data/shapefile.py`.** (a) Constructor takes `assumed_crs`. (b) In `region()`: if `raw.crs is None`: `if assumed_crs is None: raise ValueError("shapefile has no CRS; pass assumed_crs=...")` else `raw = raw.set_crs(assumed_crs)`. (c) After reprojecting to UTM, **explode**: `raw = raw.explode(index_parts=False, ignore_index=True)` then drop non-Polygon rows; this turns each MultiPolygon record into its constituent Polygons so no component dissolves to a MultiPolygon from a *native* multi-part record. (d) Make `region()` return `Region(..., blocks=self._iter_blocks(raw, utm))` where `_iter_blocks` is a generator `yield`ing one `Block` per component (keep the existing per-component logic; the `not isinstance(..., Polygon)` guard stays as a real-defect backstop, now rarely hit).

- [ ] **Step 4: Run to verify pass.** `pixi run pytest tests/data/test_shapefile_source.py -v` → PASS. Note: `Region.blocks` is now a one-shot generator; if any test iterates twice, wrap in `list()`.
- [ ] **Step 5: `pixi run check`** → green.
- [ ] **Step 6: Commit.** `git commit -am "feat: ShapefileSource explicit assumed_crs + multi-part explode + lazy blocks"`

---

### Task 5: topology `trace_faces` + `maxdepth` fixes (submodule)

**Files:** Modify `ext/topology/topology/graph/my_graph.py` (`trace_faces`, `stacked_duals`); Create `ext/topology/tests/test_trace_faces.py`.

**Interfaces — Produces:** `trace_faces` selects the outer face by **signed shoelace area** (the unbounded face winds opposite), not edge count; `stacked_duals` raises when it hits `maxdepth` instead of silently truncating.

- [ ] **Step 1: Create branch** `git -C ext/topology switch -c hardening-topology`.
- [ ] **Step 2: Write failing test** `ext/topology/tests/test_trace_faces.py`: build a 3×3-block-sized case with a **U-shaped frontage parcel (8 edges) wrapping a small rear parcel (4 edges)** whose union tiles the block; assert both parcels survive in `inner_facelist` (2 inner faces) and the outer face is the block boundary (largest |area|) — NOT the U-parcel. (Use `MyNode`/`MyEdge`/`MyFace`/`graphFromMyFaces`.)
- [ ] **Step 3: Run to verify fail** (`python -m pytest ext/topology/tests/test_trace_faces.py` via `pixi run`) → the U-parcel is currently dropped (outer chosen by `len`).
- [ ] **Step 4: Fix `trace_faces`.** Replace the `sorted(faces, key=len)` outer-face pick (lines ~508-511) with: build a `MyFace` per traced path, compute each path's **signed** shoelace area, and select as `outerface` the face whose signed-area sign is the minority (in a planar embedding exactly one face — the unbounded one — winds opposite; tie-break by max `abs(area)`). Everything else (inner_facelist construction, `iface.edges`, `down1_node`) unchanged.
- [ ] **Step 5: Fix `stacked_duals`** (`maxdepth=15`): raise the default to e.g. `50`, and after the loop, if the last appended level was non-empty (cap actually reached), `raise RuntimeError(f"stacked_duals hit maxdepth={maxdepth}; block nesting deeper than supported")` — loud, not silent. (Keep `k_complexity` returning the count.)
- [ ] **Step 6: Run** the new test + `ext/topology/tests/unit_tests.py` + `test_k_complexity.py` → all PASS (fixes must not regress the metric on well-behaved blocks).
- [ ] **Step 7: Commit in submodule + repin.** `git -C ext/topology commit -am "Fix trace_faces outer-face selection (signed area); make stacked_duals maxdepth loud"`; then in reblock `git add ext/topology && git commit -m "chore: repin topology (trace_faces + maxdepth fixes)"`.

---

### Task 6: TopologyMethod — `prior`, consume `Block.streets`, memoize trace

**Files:** Modify `src/reblock/methods/topology.py`; Modify `src/reblock/derive/parcel_graph.py` (memoize); Modify `tests/methods/test_topology_method.py`.

**Interfaces — Produces:** `TopologyMethod.propose(block, prior: Proposal | None = None) -> Proposal` (prior currently unused — topology is independent — but accepted); marks initial roads from **`Block.streets`** (edges coincident with street lines) rather than `define_roads()`; sets `proposal_id` (e.g. `f"topology_a{alpha}_s{seed}"`). `to_parcel_graph` memoized per block.

- [ ] **Step 1: Migrate/add tests.** Keep the 3×3 "proposes ≥1 road / interior resolved / determinism / edges columns" tests (add `prior=None` still works). ADD a `streets ⊊ boundary` test: a block whose `Block.streets` is only ONE side of the boundary yields a different (larger) interior set than full-boundary streets → the proposal differs. ADD: `proposal.proposal_id == "topology_a2.0_s0"`.

- [ ] **Step 2: Run to verify fail** → FAIL (signature/`proposal_id`/streets).

- [ ] **Step 3: Rework `methods/topology.py`.** `propose(self, block, prior=None)`. Replace `graph.define_roads()` with marking edges whose geometry lies on `Block.streets`: reuse the endpoint-matching helper (factor the `_endpoint_keys`-style match from the old eval into a small shared helper, or inline) to set `edge.road = True` for graph edges coincident with `block.streets` (origin-shifted, 2-dp). Then `define_interior_parcels()` and proceed as today. Set `proposal_id=f"topology_a{self.alpha}_s{self.seed}"` on the returned `Proposal`.

- [ ] **Step 4: Memoize the trace.** In `derive/parcel_graph.py`, add a module-level cache keyed by `id(block)` (or a `(block_id, len(parcels))` tuple) returning the built `PlanarParcelGraph`, so repeated `to_parcel_graph(block)` in one run reuses the trace. Since `Block` is frozen and holds unhashable GeoDataFrames, key on `block.block_id` + parcel count; document the cache is per-process and small. Provide `clear_cache()` for tests.

- [ ] **Step 5: Run to verify pass** → PASS. `pixi run check` green.
- [ ] **Step 6: Commit.** `git commit -am "feat: TopologyMethod consumes Block.streets, accepts prior, sets proposal_id; memoize graph trace"`

---

### Task 7: `render` component

**Files:** Create `src/reblock/render.py`; Create `tests/test_render.py`.

**Interfaces — Produces:** `render_before(block, layers: "pd.Series", *, vmax: int) -> Figure`; `render_after(block, proposal, layers: "pd.Series", *, vmax: int, metrics: Metrics | None = None) -> Figure`; `save_render(fig, path) -> None`. Single-panel `YlOrRd` heatmap (parcels joined to `layers` on `parcel_id`), `vmin=1`/`vmax`, block-boundary outline, colourbar; after overlays `proposal.roads`.

- [ ] **Step 1: Write failing tests** `tests/test_render.py`: build the 3×3 block + `parcel_access_layers`; assert `render_before(...)` returns a `matplotlib.figure.Figure` with ≥1 `Axes`; `render_after(block, proposal, layers_post, vmax=2)` returns a Figure and (roads present) has >1 collection/line artist; `save_render(fig, tmp_path/"b.png")` writes a non-empty file. Use `matplotlib.use("Agg")`.

- [ ] **Step 2: Run to verify fail** → FAIL (module missing).

- [ ] **Step 3: Implement `src/reblock/render.py`** (geopandas `.plot(column=...)`): join `block.parcels` to `layers` on `parcel_id` into a `layer` column, `parcels.plot(column="layer", cmap="YlOrRd", vmin=1, vmax=vmax)`, plot `block.boundary.boundary` as an outline, add a colourbar; `render_after` additionally plots `proposal.roads`. `Agg` backend; `save_render` = `fig.savefig(path, dpi=140, bbox_inches="tight")`. Add a mypy override for the module if matplotlib typing needs it.

- [ ] **Step 4: Run to verify pass** → PASS. `pixi run check` green.
- [ ] **Step 5: Commit.** `git commit -am "feat: add render component (before/after access-depth heatmaps)"`

---

### Task 8: Hydra pluggability + Result + render wiring

**Files:** Modify `src/reblock/run.py`; Create `conf/config.yaml`, `conf/data/phule.yaml`, `conf/method/topology.yaml`, `conf/eval/{kcomplexity,weakdual_k}.yaml`; Modify `tests/test_run.py`.

**Interfaces — Produces:** `run(cfg, *, render_base: Path | None = None) -> list[Result]`; a Hydra `main()` that instantiates `data`/`method(s)`/`eval(s)` from config groups, computes layers once, scores all evals, renders before + per-method after under `HydraConfig.get().runtime.output_dir / render_dir`.

- [ ] **Step 1: Migrate/add tests.** Rewrite `tests/test_run.py`: `run(RunConfig(...))` returns `list[Result]`; each `Result` has `.block`, `.proposal`, `.metrics`; the end-to-end phule test asserts `r.metric("kcomplexity","k_after") <= r.metric("kcomplexity","k_before")`. Keep the efficacy scan (a real block with `delta_k>0`). Keep the CLI subprocess smoke test, adding `render_dir=renders` and asserting `{block_id}_before.png` + an `_after.png` appear under the run's output dir. Add a Hydra `initialize`/`compose` test that composes `method=topology eval=kcomplexity` from the config groups and runs `run`.

- [ ] **Step 2: Run to verify fail** → FAIL.

- [ ] **Step 3: Rework `run.py` + add `conf/`.** `RunConfig` gains `render_dir: str | None = None` and Hydra `defaults`/`_target_` groups: `conf/method/topology.yaml` → `_target_: reblock.methods.topology.TopologyMethod`, etc. `run()` uses `hydra.utils.instantiate` for source/methods/evals (lists), computes `layers_before` once per block, loops methods → `proposal` → `layers_after` → evals → `Result`, and (if `render_base`/`render_dir`) writes `render_before` once + `render_after` per proposal. `main()` passes `render_base=Path(HydraConfig.get().runtime.output_dir)`.

- [ ] **Step 4: Run to verify pass** → PASS.
- [ ] **Step 5: Smoke-run.** `pixi run run data=phule method=topology eval=kcomplexity shapefile=ext/topology/examples/data/phule_nagar_v6.shp render_dir=renders max_blocks=1` → prints a `Result` line and writes `<run>/renders/phule_0_before.png` + `phule_0_topology_a2.0_s0_after.png`.
- [ ] **Step 6: Full `pixi run check`** → green. Commit. `git commit -am "feat: Hydra config-group pluggability, Result output, render wiring under run dir"`

---

## Self-review notes

**Spec coverage:** contracts v2 (T1), peel metric primary + weak-dual optional (T2/T3), render before/after separate files from fields (T7/T8), robustness assumed_crs+multipart+lazy (T4), topology trace_faces+maxdepth (T5), Block.streets consumed + method prior + memoize (T6), Hydra pluggability + Result (T8). Peel-reblocker is the *next* slice (own plan), enabled by T2's derivation.

**Type consistency:** `Metrics(values, fields)`, `Result.metric(eval,key)`, `parcel_access_layers(block, roads) -> pd.Series[parcel_id]`, `Method.propose(block, prior=None)`, `render_{before,after}(..., vmax=)` — consistent across tasks.

**Risks to watch:** (T2) the peel adjacency tolerance for pinch-point gaps — calibrate against the real Phule blocks (`_components` already proves 0/370 issues at tol≈0 via shared-length; the peel can reuse that predicate). (T5) the signed-area outer-face pick must not regress the metric — run topology's own suite. (T8) Hydra `instantiate` of methods/evals as lists needs `defaults` list syntax; the compose test is the guard.
