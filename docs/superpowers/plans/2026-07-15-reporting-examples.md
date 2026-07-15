# Reporting & Examples Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add area-normalized (`pct_paved` / `pct_displaced`) reporting to the compare tables, surface the efficiency curve + a hard-displacement view in both flagship examples, and bring CELF/lazy arterial into both — arterial rejoining the multiblock region flagship.

**Architecture:** Two small pure geometry helpers in `emit.py` compute the percentages; `compare()` computes them per (method, block) where it already holds `roads`+`block` and threads them onto `MethodCurve`; `compare_report` writes them as new columns. Config switches arterial to lazy; the two example galleries are regenerated and their READMEs updated to match.

**Tech Stack:** Python, geopandas/shapely, pytest (`pixi run pytest`), Hydra compare (`pixi run python -m reblock.compare`).

## Global Constraints

- No-legacy: extend the existing emit tables (append columns); existing `auc_table_*` / `tradeoff_table_*` CSVs and curve PNGs keep working unchanged.
- The four lenses, two cost axes, and cost-benefit math are unchanged in behavior.
- READMEs must match the regenerated artifacts — no stale numbers.
- `corridor_m` for `pct_paved` = the compare's own `corridor_m` (compare.py:94, default 3.0), the SAME footprint the `cost=displacement` axis uses — so paved-area and displacement stay consistent.
- Run tests with `pixi run pytest`; run compares with `pixi run python -m reblock.compare`. Branch: `reporting-examples`.

---

### Task 1: `pct_paved` / `pct_displaced` geometry helpers

**Files:**
- Modify: `src/reblock/emit.py` (add two helpers near `_displaced_points` ~49)
- Test: `tests/test_emit_pct.py` (new)

**Interfaces:**
- Produces: `pct_paved(roads: GeoDataFrame | None, corridor_m: float, block_area: float) -> float` and `pct_displaced(roads: GeoDataFrame | None, corridor_m: float, building_points: GeoDataFrame) -> float`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_emit_pct.py
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon
from reblock.emit import pct_paved, pct_displaced


def _roads(*lines):
    return gpd.GeoDataFrame(geometry=[LineString(l) for l in lines], crs="EPSG:32734")


def test_pct_paved_is_buffer_area_over_block_area():
    roads = _roads([(0, 0), (100, 0)])
    block_area = 10_000.0
    expected = roads.geometry.buffer(3.0).union_all().area / block_area
    assert abs(pct_paved(roads, 3.0, block_area) - expected) < 1e-9
    assert 0.0 < pct_paved(roads, 3.0, block_area) < 1.0


def test_pct_paved_empty_or_zero_area_is_zero():
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:32734")
    assert pct_paved(empty, 3.0, 10_000.0) == 0.0
    assert pct_paved(None, 3.0, 10_000.0) == 0.0
    assert pct_paved(_roads([(0, 0), (100, 0)]), 3.0, 0.0) == 0.0


def test_pct_displaced_is_fraction_of_points_in_corridor():
    roads = _roads([(0, 0), (100, 0)])          # corridor is |y| <= 3 along the x-axis
    pts = gpd.GeoDataFrame(geometry=[Point(50, 0), Point(50, 1), Point(50, 50), Point(50, 80)],
                           crs="EPSG:32734")     # first two inside, last two outside
    assert abs(pct_displaced(roads, 3.0, pts) - 0.5) < 1e-9


def test_pct_displaced_empty_roads_or_no_points_is_zero():
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:32734")
    assert pct_displaced(gpd.GeoDataFrame(geometry=[], crs="EPSG:32734"), 3.0, pts) == 0.0
    assert pct_displaced(None, 3.0, pts) == 0.0
    assert pct_displaced(_roads([(0, 0), (100, 0)]), 3.0,
                         gpd.GeoDataFrame(geometry=[], crs="EPSG:32734")) == 0.0
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run pytest tests/test_emit_pct.py -q`
Expected: FAIL (`cannot import name 'pct_paved'`).

- [ ] **Step 3: Implement the helpers in `emit.py`** (place right after `_displaced_points`)

```python
def pct_paved(roads: gpd.GeoDataFrame | None, corridor_m: float, block_area: float) -> float:
    """Fraction of the block's area under the roads' corridor footprint
    (union(roads).buffer(corridor_m)) -- the same buffer the displacement metric uses. 0 for an
    empty road set or a non-positive block area."""
    if roads is None or len(roads) == 0 or block_area <= 0:
        return 0.0
    return float(roads.geometry.buffer(corridor_m).union_all().area / block_area)


def pct_displaced(roads: gpd.GeoDataFrame | None, corridor_m: float,
                  building_points: gpd.GeoDataFrame) -> float:
    """Fraction of building points inside the roads' corridor (union(roads).buffer(corridor_m)).
    0 for an empty road set or no buildings."""
    n = len(building_points)
    if roads is None or len(roads) == 0 or n == 0:
        return 0.0
    corridor = roads.geometry.buffer(corridor_m).union_all()
    return float(int(building_points.geometry.within(corridor).sum()) / n)
```

- [ ] **Step 4: Run to confirm pass**

Run: `pixi run pytest tests/test_emit_pct.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/reblock/emit.py tests/test_emit_pct.py
git commit -m "feat(emit): pct_paved / pct_displaced geometry helpers"
```

---

### Task 2: Thread the percentages onto `MethodCurve` + into the tables

**Files:**
- Modify: `src/reblock/compare.py` (`MethodCurve` ~41; `compare()` loop ~100-134)
- Modify: `src/reblock/emit.py` (`compare_report` ~236-262)
- Test: `tests/test_compare.py` (add a column-presence assertion) or `tests/test_emit_pct.py`

**Interfaces:**
- Consumes: `pct_paved`, `pct_displaced` (Task 1).
- Produces: `MethodCurve` gains `pct_paved: float` and `pct_displaced: float` fields; `auc_table_{metric}.csv` gains a `mean_pct_paved` column; `tradeoff_table_{metric}.csv` gains a `mean_pct_displaced` column.

- [ ] **Step 1: Write a failing test** (the length table gains `mean_pct_paved`)

```python
# tests/test_compare.py — add
def test_auc_table_has_mean_pct_paved_column(tmp_path):
    import csv
    from reblock.compare import MethodCurve
    from reblock.contracts import Curve
    from reblock.emit import compare_report
    c = Curve(cost=[0.0, 100.0], benefit=[0.0, 0.8])
    mc = MethodCurve("dijkstra", "B1", "access", c, 0.5, pct_paved=0.041, pct_displaced=0.0)
    compare_report([mc], tmp_path, cost="length")
    with (tmp_path / "auc_table_access.csv").open() as f:
        header = next(csv.reader(f))
    assert "mean_pct_paved" in header
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run pytest tests/test_compare.py::test_auc_table_has_mean_pct_paved_column -q`
Expected: FAIL (`MethodCurve.__init__() got an unexpected keyword argument 'pct_paved'`).

- [ ] **Step 3: Add fields to `MethodCurve`** (compare.py ~41)

```python
@dataclass
class MethodCurve:
    method: str
    block_id: str
    metric: str
    curve: Curve
    auc: float
    pct_paved: float = 0.0
    pct_displaced: float = 0.0
```

- [ ] **Step 4: Compute + thread the percentages in `compare()`**

Add the import at the top of compare.py: `from reblock.emit import compare_report as compare_report, pct_paved, pct_displaced`.

Change the `raw` type and the loop body (compare.py:100-127). Compute the two percentages ONCE per (method, block) right after `roads`/`block` are set, and carry them in every raw tuple:

```python
    raw: list[tuple[str, str, str, Curve, float, float]] = []
    for region in regions:
        if not region:
            continue
        label = _region_label(region)
        for name, method in zip(names, methods, strict=True):
            if len(region) == 1:
                block = region[0]
                roads = cast(GeoDataFrame, propose(method, block).roads)
            else:
                result = region_reblock(region, method, [])
                block = result.block
                roads = cast(GeoDataFrame, result.proposal.roads)
            block_area = float(block.parcels.geometry.union_all().area)
            pp = pct_paved(roads, corridor_m, block_area)
            pd_ = pct_displaced(roads, corridor_m, block.building_points)
            access = cost_benefit_curve(block, roads, benefit_fn=access_benefit,
                                        cost=cost, corridor_m=corridor_m)
            eff, direct = efficiency_directness_curves(block, roads, cost=cost, corridor_m=corridor_m)
            resistance = cost_benefit_curve(block, roads, benefit_fn=resistance_benefit,
                                            cost=cost, corridor_m=corridor_m)
            raw.append((name, label, "access", access, pp, pd_))
            raw.append((name, label, "efficiency", eff, pp, pd_))
            raw.append((name, label, "directness", direct, pp, pd_))
            raw.append((name, label, "resistance", resistance, pp, pd_))
    results: list[MethodCurve] = []
    groups = {(label, metric) for _, label, metric, _, _, _ in raw}
    for label, metric in groups:
        group = [(m, c, pp, pd_) for m, lbl, met, c, pp, pd_ in raw if lbl == label and met == metric]
        cap = max((c.cost[-1] for _, c, _, _ in group if c.cost), default=0.0)
        for m, c, pp, pd_ in group:
            results.append(MethodCurve(m, label, metric, c, auc(c, cap), pct_paved=pp, pct_displaced=pd_))
    return results
```

- [ ] **Step 5: Add the columns in `compare_report`** (emit.py)

In the `displacement` branch (emit.py ~244-253), collect `pct_displaced` per method and add the column:

```python
            by_bd: dict[str, list[tuple[float, float]]] = {}
            by_pd: dict[str, list[float]] = {}
            for r in metric_results:
                by_bd.setdefault(r.method, []).append((r.curve.benefit[-1], r.curve.cost[-1]))
                by_pd.setdefault(r.method, []).append(r.pct_displaced)
            with (out_dir / f"tradeoff_table_{metric}.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["method", "mean_terminal_benefit", "mean_buildings_displaced",
                            "mean_pct_displaced", "n_blocks"])
                for m, bd in sorted(by_bd.items(), key=lambda kv: -mean(b for b, _ in kv[1])):
                    w.writerow([m, f"{mean(b for b, _ in bd):.4f}", f"{mean(d for _, d in bd):.1f}",
                                f"{mean(by_pd[m]):.4f}", len(bd)])
```

In the `length` branch (emit.py ~255-262), collect `pct_paved` per method and add the column:

```python
            by_method: dict[str, list[float]] = {}
            by_pp: dict[str, list[float]] = {}
            for r in metric_results:
                by_method.setdefault(r.method, []).append(r.auc)
                by_pp.setdefault(r.method, []).append(r.pct_paved)
            with (out_dir / f"auc_table_{metric}.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["method", "mean_auc", "mean_pct_paved", "n_blocks"])
                for m, aucs in sorted(by_method.items(), key=lambda kv: -mean(kv[1])):
                    w.writerow([m, f"{mean(aucs):.4f}", f"{mean(by_pp[m]):.4f}", len(aucs)])
```

- [ ] **Step 6: Run the test + the existing compare/emit suites**

Run: `pixi run pytest tests/test_compare.py tests/test_emit_pct.py -q`
Expected: PASS (new column present; existing compare tests still green — columns are additive).

- [ ] **Step 7: Commit**

```bash
git add src/reblock/compare.py src/reblock/emit.py tests/test_compare.py
git commit -m "feat(compare): thread pct_paved/pct_displaced into MethodCurve + tables"
```

---

### Task 3: Switch arterial to lazy in the compare config

**Files:**
- Modify: `conf/compare_config.yaml` (`all_methods.greedy_arterial_buildable`)
- Test: none (config); verified by the regeneration tasks.

**Interfaces:**
- Produces: `greedy_arterial_buildable` runs the CELF lazy engine.

- [ ] **Step 1: Set lazy on the arterial entry**

In `conf/compare_config.yaml`, add to the `all_methods.greedy_arterial_buildable` mapping:
```yaml
    lazy: true
    candidate_policy: grow
```
(Keep its existing keys — mode/objective/max_roads/etc. — unchanged.)

- [ ] **Step 2: Sanity-check it instantiates**

Run: `pixi run python -c "from hydra import compose, initialize; from hydra.utils import instantiate; initialize(config_path='conf', version_base=None); cfg=compose('compare_config'); m=instantiate(cfg.all_methods.greedy_arterial_buildable); print(m.lazy, m.candidate_policy)"`
Expected: prints `True grow`.

- [ ] **Step 3: Commit**

```bash
git add conf/compare_config.yaml
git commit -m "feat(compare): greedy_arterial_buildable uses CELF lazy (grow)"
```

---

### Task 4: Regenerate method-comparison + update its README

**Files:**
- Modify: `examples/method-comparison/README.md`; regenerate its images/CSVs into `examples/method-comparison/`.

- [ ] **Step 1: Regenerate the length-cost comparison (now with lazy arterial)**

Run (from repo root; writes to a hydra run dir it prints):
```bash
pixi run python -m reblock.compare data=capetown_full \
  "block_ids=[[ZAF.9.3.1_1_40972]]" \
  methods=[dijkstra,peel,topology,mesh,greedy_arterial_buildable,clearance] max_blocks=1 \
  all_methods.greedy_arterial_buildable.max_roads=8
```
Copy the regenerated `curve_*.png` (incl. `curve_efficiency_*`), `auc_table_*.csv` into `examples/method-comparison/`. Confirm `auc_table_*.csv` now carries `mean_pct_paved`.

- [ ] **Step 2: Run the displacement-cost pass**

Run the SAME command with `cost=displacement` appended. Copy the resulting displacement-axis `curve_*.png` (rename to disambiguate, e.g. `curve_directness_displacement.png`) + `tradeoff_table_directness.csv` into the example dir. Confirm `tradeoff_table_*.csv` carries `mean_pct_displaced`.

- [ ] **Step 3: Update the README**

- Replace the arterial description to note it runs via CELF/lazy (`greedy_arterial_buildable` = lazy grow).
- In the AUC table, add the paved figure alongside road cost: report **raw AND %** (e.g. a `% paved` column or inline "260 m/ha (4.1% paved)"), reading the values from `auc_table_*.csv`'s `mean_pct_paved`.
- Add a short **displacement** subsection: the displacement-axis curve + the tradeoff table (terminal benefit, buildings displaced, `% displaced`), with the honest framing that only arterial/clearance displace and frontage methods sit at ~0.
- Every number must come from the regenerated CSVs — no stale values.

- [ ] **Step 4: Commit**

```bash
git add examples/method-comparison/
git commit -m "docs(method-comparison): lazy arterial, efficiency + displacement + %-paved/displaced"
```

---

### Task 5: Regenerate multiblock (arterial rejoins) + tractability check + README

**Files:**
- Modify: `examples/multiblock/README.md`; regenerate images/CSVs into `examples/multiblock/`.

- [ ] **Step 1: Tractability check — time lazy arterial on block 5810**

Run and capture wall-time:
```bash
time pixi run python -c "
from reblock.data.provision import ensure_city_data
from reblock.data.kblock import KblockSource
from reblock.methods.arterial import GreedyArterialReblocker
bp, bld = ensure_city_data('capetown')
blk = next(iter(KblockSource(bp, bld, region_id='p', min_buildings=10, block_ids=['ZAF.9.3.1_1_5810']).region().blocks))
r = GreedyArterialReblocker(mode='buildable', lazy=True, candidate_policy='grow', max_roads=15, workers=16).propose(blk).roads
print('roads', len(r))
"
```
If wall-time is impractical (> ~10 min), re-run with a reduced `n_anchors` (e.g. add `n_anchors=16` to the constructor) until it is; record the chosen `n_anchors` + measured time.

- [ ] **Step 2: Regenerate the multiblock method comparison WITH arterial (length cost)**

Run (add `greedy_arterial_buildable` to the methods list; append the chosen `all_methods.greedy_arterial_buildable.n_anchors=<N>` override from Step 1 if reduced):
```bash
pixi run python -m reblock.compare data=capetown_full \
  region_builder=dense_cluster region_builder.max_buildings=3000 \
  "block_ids=[[ZAF.9.3.1_1_5810]]" \
  methods=[dijkstra,peel,mesh,clearance,greedy_arterial_buildable] max_blocks=1
```
Copy the regenerated `curve_*.png` (INCLUDING `curve_efficiency_*` — previously omitted) + `auc_table_*.csv` into `examples/multiblock/`.

- [ ] **Step 3: Run the displacement-cost pass** (same command + `cost=displacement`); copy the displacement curves + `tradeoff_table_*.csv`.

- [ ] **Step 4: Update the README**

- §4 method comparison: add **arterial** back as a scalable method (the CELF payoff — call it out), and add the **efficiency** curve (with the "near-inert at scale" caveat).
- Add the paved % alongside road density (raw + %) from `mean_pct_paved`; add a **displacement** subsection (curve + tradeoff table incl. `% displaced`).
- Update the reproduce block to the exact command used, including any reduced `n_anchors` + the measured lazy-arterial runtime from Step 1.
- All numbers from the regenerated CSVs.

- [ ] **Step 5: Commit**

```bash
git add examples/multiblock/
git commit -m "docs(multiblock): arterial rejoins via CELF; efficiency + displacement + %-paved/displaced"
```

---

## Self-Review

**Spec coverage:** pct_paved/pct_displaced code (Tasks 1-2); efficiency curve in multiblock (Task 5 Step 2/4); displacement view both (Task 4 Step 2, Task 5 Step 3); raw+% READMEs (Tasks 4-5); lazy arterial both (Task 3 config + Task 4/5 regen); multiblock arterial rejoin + tractability (Task 5 Step 1-2). Covered.

**Placeholder scan:** No TBD/TODO. Task 5 Step 1's "> ~10 min" is a decision threshold with a concrete fallback (reduce n_anchors + record), not a placeholder.

**Type consistency:** `MethodCurve(..., pct_paved=..., pct_displaced=...)` matches the dataclass fields; `raw` 6-tuple `(name, label, metric, curve, pp, pd_)` is consistent between append sites and the regroup unpacking; `pct_paved(roads, corridor_m, block_area)` / `pct_displaced(roads, corridor_m, building_points)` signatures match between Task 1 definition and Task 2 call sites; table columns `mean_pct_paved` / `mean_pct_displaced` consistent between emit writes and the Task 2 test.
