# Displacement-Road Reblocker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Lay straight roads and let the greedy trade navigability benefit against buildings displaced (site in the road corridor); benefit is measured exactly as today on the full settlement (nothing removed), displacement is a pure cost axis.

**Architecture:** `Block` gains the real building points; a `displacement_count` primitive; a `cost="displacement"` axis on `GreedyArterialReblocker` (Δbenefit per building displaced) and on `cost_benefit_curve` (x = cumulative buildings displaced). No change to the access/directness/efficiency computation.

**Tech Stack:** shapely (`buffer`, `.within`/STRtree), geopandas, existing budget/arterial machinery, pytest.

**Design:** `docs/superpowers/specs/2026-07-10-displacement-road-reblocker-design.md` (authoritative).

## Global Constraints

- Benefit is computed EXACTLY as today (road = 1-D line, full parcel set retained, all points in place, occlusion intact). Displacement never touches the access computation — it is only a cost count.
- Displacement = building **sites** (real points) whose location lies in `road.buffer(corridor_m)`. Marginal displacement de-dups overlapping corridors (a site hit by two roads counts once).
- No back-compat / dual paths (no-legacy). New fields/params are genuine additions with sensible defaults, not shims.
- `corridor_m` default `3.0` (road half-width + setback), a parameter.
- Determinism; mypy --strict + ruff clean; `pixi run check` green per task.

---

### Task 1: `Block.building_points`

**Files:**
- Modify: `src/reblock/contracts.py` (`Block` gains `building_points`)
- Modify: `src/reblock/data/kblock.py` (populate it)
- Modify: `src/reblock/region.py` (`region_block` unions members)
- Test: `tests/test_contracts.py` (or wherever Block tests live) + `tests/test_region.py`

**Interfaces:**
- Produces: `Block.building_points: GeoDataFrame` (geometry = Points, region UTM; may be empty).

- [ ] **Step 1 — `contracts.py`:** add a module-level empty-points factory and the field (after `attrs`, so it has a default):
```python
def _empty_points() -> GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry")

@dataclass(frozen=True)
class Block:
    ...
    attrs: Mapping[str, object] = field(default_factory=dict)
    building_points: GeoDataFrame = field(default_factory=_empty_points)
```
Do NOT add a `__post_init__` column requirement for it (it may be empty). It is NOT part of `Block.identity`. (Import `gpd`/`GeoDataFrame` as the module already does for other fields — match existing style.)

- [ ] **Step 2 — `kblock.py`:** in `_blocks_from`, the per-block points are already in hand (`pts = pts_by_block.get(...)`). Attach them:
```python
yield Block(block_id=..., crs=utm, boundary=poly, parcels=parcels, streets=streets,
            source_content_hash=source_content_hash, attrs={...},
            building_points=gpd.GeoDataFrame(geometry=list(pts), crs=utm))
```

- [ ] **Step 3 — `region.py`:** `region_block` unions member building_points. In `_shared_parts` (or `region_block`) build:
```python
member_pts = [b.building_points for b in blocks if not b.building_points.empty]
building_points = (gpd.GeoDataFrame(pd.concat(member_pts, ignore_index=True), crs=crs)
                   if member_pts else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs))
```
and pass `building_points=building_points` to the region `Block(...)`.

- [ ] **Step 4 — `ShapefileSource`:** no change needed — `_iter_blocks` omits the field → default empty (honest, a parcel shapefile has no point cloud). Add a test asserting it's empty.

- [ ] **Step 5 — Tests:**
```python
def test_kblock_block_carries_building_points():
    src = KblockSource(DJI_BLOCKS, DJI_BLD, region_id="dji", block_ids=["<a real block id>"])
    block = next(src.region().blocks)
    assert not block.building_points.empty
    assert (block.building_points.geometry.geom_type == "Point").all()
    assert block.building_points.crs == block.crs
    # one site per building; >= parcel count is not guaranteed (dedupe/Voronoi), but > 0
    assert len(block.building_points) >= len(block.parcels) // 2

def test_region_block_unions_member_building_points():   # in test_region.py, synthetic
    # two _grid_blocks given explicit building_points -> region_block.building_points == their union
    ...

def test_shapefile_block_has_empty_building_points():
    block = next(ShapefileSource(PHULE, region_id="phule", assumed_crs=3857).region().blocks)
    assert block.building_points.empty
```
(For the synthetic region test, extend `_grid_block` to accept optional points, or construct Blocks with `building_points=` directly.)

- [ ] **Step 6:** `pixi run check` green. Commit: `feat: Block.building_points -- carry the real building sites (kblock populates, region unions, shapefile empty)`.

---

### Task 2: `displacement_count` + `cost="displacement"` on the arterial greedy

**Files:**
- Modify: `src/reblock/budget.py` (`displacement_count`)
- Modify: `src/reblock/methods/arterial.py` (`cost`/`corridor_m` params, greedy denominator)
- Test: `tests/test_budget.py`, `tests/methods/test_arterial.py`

**Interfaces:**
- Consumes: `Block.building_points` (Task 1).
- Produces: `budget.displacement_count(building_points, roads, corridor_m) -> int`; `GreedyArterialReblocker(cost="length"|"displacement", corridor_m=3.0)`.

- [ ] **Step 1 — `budget.displacement_count`:**
```python
def displacement_count(building_points: GeoDataFrame, roads: GeoDataFrame, corridor_m: float) -> int:
    """Buildings whose site lies in the road corridor = union of roads.buffer(corridor_m).
    0 when there are no points or no roads."""
    if building_points is None or building_points.empty or roads is None or len(roads) == 0:
        return 0
    corridor = roads.geometry.buffer(corridor_m).union_all()
    return int(building_points.geometry.within(corridor).sum())
```

- [ ] **Step 2 — Test `displacement_count`:** a straight road through a known point grid counts exactly the sites within `corridor_m`; points beyond the corridor are not counted; two overlapping roads' union counts a shared site once. (Synthetic points + a LineString.)

- [ ] **Step 3 — `arterial.py` params:** add `cost: str = "length"` and `corridor_m: float = 3.0` to `GreedyArterialReblocker`; `identity` → `("greedy_arterial", self.mode, self.objective, self.cost)`; thread `cost`/`corridor_m` into `_greedy_arterials`.

- [ ] **Step 4 — greedy denominator (`_greedy_arterials`):** when `cost == "displacement"`, the per-candidate gain denominator is the road's **marginal displacement** instead of `real.length`:
```python
committed_disp = displacement_count(block.building_points, _planarize(committed, block.crs), corridor_m)
# inside the candidate loop, for realized road `real`:
if cost == "displacement":
    marg = displacement_count(block.building_points, _planarize(committed + [real], block.crs),
                              corridor_m) - committed_disp
    denom = marg
else:
    denom = real.length
# zero-displacement (or zero-length) beneficial road: rank above any positive-denom road
raw = _score(objective, block, trial, adj, base_burden) - base_val
gain = float("inf") if (denom <= 0 and raw > 0) else (raw / denom if denom > 0 else 0.0)
```
Keep the deterministic tie-break (`real.wkt`). `float("inf")` candidates are taken first; guard the `best_gain` comparison so `inf` works. Recompute `committed_disp` after each commit.

- [ ] **Step 5 — Tests (`tests/methods/test_arterial.py`):**
  - `identity` includes `cost` (`("greedy_arterial","aspirational","directness","displacement")`).
  - Determinism: two runs identical.
  - Preference: on a fixture with two equal-benefit straight candidates where one's corridor hits fewer building points, `cost="displacement"` commits the lower-displacement one (construct so `cost="length"` would differ, proving the denominator switched).
  - A zero-displacement beneficial straight road is committed (not skipped by div-by-zero).

- [ ] **Step 6:** `pixi run check` green. Commit: `feat: displacement_count + greedy_arterial cost=displacement (Δbenefit per building displaced)`.

---

### Task 3: displacement cost-benefit curve + config + compare + render

**Files:**
- Modify: `src/reblock/budget.py` (`_sweep` cost_fn, `cost_benefit_curve`/`efficiency_directness_curves` `cost=` arg)
- Create: `conf/method/greedy_arterial_displacement.yaml`
- Modify: `conf/compare_config.yaml` (add the method) — or the compare methods list
- Modify: `src/reblock/render.py` (mark displaced points in the after)
- Modify: `src/reblock/emit.py` (pass displaced points / corridor to render)
- Test: `tests/test_budget.py`, `tests/test_render.py`, a config-instantiation test

**Interfaces:**
- Consumes: `displacement_count` (Task 2), `Block.building_points` (Task 1).
- Produces: `cost_benefit_curve(block, roads, *, benefit_fn, cost="length"|"displacement", corridor_m=3.0)`; `conf/method/greedy_arterial_displacement.yaml`.

- [ ] **Step 1 — `_sweep` cost_fn (`budget.py`):** parameterize the reported cost. Add `cost_fn: Callable[[GeoDataFrame], float]` (default the current density). Replace the two cost lines:
```python
# baseline
costs: list[float] = [cost_fn(cast(GeoDataFrame, roads.iloc[:0]))]
...
costs.append(cost_fn(ordered.iloc[:m]))    # was float(cum[m-1]) / area_ha
```
The road ORDER + length-budget SAMPLING stay (drainage order, `cum` budgets) — only the reported cost changes. Default `cost_fn = lambda prefix: prefix.geometry.length.sum() / area_ha` (identical to today).

- [ ] **Step 2 — `cost_benefit_curve`/`efficiency_directness_curves` `cost=` arg:** add `cost: str = "length", corridor_m: float = 3.0`. For `cost="displacement"`, pass `cost_fn = lambda prefix: float(displacement_count(block.building_points, prefix, corridor_m))`. Update `Curve.cost`'s comment (it's now "m/ha OR buildings displaced").

- [ ] **Step 3 — Test the displacement curve:** `cost_benefit_curve(block, roads, cost="displacement", corridor_m=...)` returns a `Curve` whose `cost` is monotone non-decreasing and equals cumulative displaced at each prefix; with empty `building_points` the cost is all-zeros (degenerate, no crash); the `benefit` is identical to the `cost="length"` curve's benefit (same prefixes, only the x-axis changed).

- [ ] **Step 4 — config:** `conf/method/greedy_arterial_displacement.yaml`:
```yaml
_target_: reblock.methods.arterial.GreedyArterialReblocker
mode: aspirational
objective: directness
cost: displacement
corridor_m: 3.0
```
Add a test that `instantiate` on it yields the right params + identity. Add it to the compare config's method list (mirror how `greedy_arterial_buildable` is wired).

- [ ] **Step 5 — render highlight:** in `render_after` (render.py), mark the displaced building points distinctly (e.g. `edgecolor="#c0392b"`, hollow ring) on top of the `own_points`. The displaced set = `displacement`-hit points; pass them in from `emit` (emit computes `block.building_points` within the proposal's corridor, using `corridor_m` from `proposal.params` if present else the default). Guard empty. Keep it a small, additive overlay.

- [ ] **Step 6 — Tests:** render smoke with a displaced-points arg writes a file; config instantiation test; (optional) a compare smoke on a tiny fixture producing a displacement curve.

- [ ] **Step 7:** `pixi run check` green. Commit: `feat: displacement cost-benefit curve (x = buildings displaced) + greedy_arterial_displacement config + render mark`.

---

## Notes for the executor
- Task 2 Step 4 is the subtle one: the `inf`-gain path must integrate with the existing `best_gain`/tie-break loop so zero-displacement beneficial roads are taken deterministically. Verify a fixture actually exercises it.
- Displacement-cost curves are meaningful for straight (aspirational) roads; a buildable/frontage method lands near x≈0 (informative contrast, not a bug).
- After all tasks: whole-branch review, then finish-branch → merge to main + push (owner-gated). Then a real-data demo: the directness-vs-buildings-displaced curve + a marked render.
