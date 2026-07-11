# Arterial Frozen-Context Performance Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make greedy-arterial scoring — and the shared `network_efficiency` metric and compare cost-benefit curves that share its core — 10–50× faster by freezing per-block constants and migrating the hot distance path to `scipy.sparse.csgraph`, with every metric value preserved to 1e-9.

**Architecture:** A single `_BlockScoringContext` in `budget.py`, built once per block, freezes the base street graph (scipy CSR + node→index map), the representative points, the K sampled sources, the K×N euclidean matrix, and each parcel's street-edge projection. `network_efficiency`, `_efficiency_factory`, and `GreedyArterialReblocker` all score through it; the greedy builds it once per block and evaluates candidates incrementally (per-step `StepContext` over streets ∪ committed, per-candidate trial-road delta). Shortest paths move to `csgraph.dijkstra`; networkx stays only in `road_drainage` and `_snap`.

**Tech Stack:** Python, numpy, scipy (`sparse`, `csgraph`), shapely (2.x ufuncs), geopandas, networkx (residual), pixi, pytest, mypy --strict, ruff.

**Spec:** `docs/superpowers/specs/2026-07-11-arterial-frozen-context-perf-design.md` (finalized, twice adversarially reviewed). Read it — this plan implements its 6 staged tasks.

## Global Constraints

- **Metric values preserved to 1e-9 (relative).** `network_efficiency`, both directness/efficiency curves, and both AUCs must equal the pre-refactor values on the equivalence fixtures after every task. Pure perf refactor.
- **Monotonicity preserved.** `_efficiency_factory`'s frozen-entry sweep stays non-decreasing across cost-benefit prefixes.
- **Recorded numbers stand.** `tests/test_budget.py` + `tests/test_region.py` recorded values (compact-block AUC ordering, `directness ∈ [0,1]`, deep-region ~5.5× ratio, assertion `>1.2×`) stay green.
- **No dual code paths in metric scoring.** networkx graph-build + K single-source dijkstra are *replaced* by CSR + `csgraph.dijkstra`. networkx remains ONLY in `road_drainage` (cold) and `_snap` (identical tie-break).
- **Determinism.** Sorted sources, stable node indexing, deterministic entry tie-breaks; byte-stable output.
- **Verify with:** `pixi run check` = ruff + mypy --strict + pytest. Per-task gate also runs the equivalence harness + a re-timing.

## Reference values (equivalence ground truth — `scratchpad/ref_values_1808.json`)

Block `DJI.3_1_1808` (10 parcels), current code:
- no roads: `E=0.026106`, `directness=0.326370`
- dijkstra: `E=0.023619`, `directness=0.239429`, `E_auc=0.015844`, `dir_auc=0.172189`
- arterial-buildable: `E=0.052905`, `directness=0.643840`, `E_auc=0.024198`, `dir_auc=0.273571`

The JSON also stores each road set's WKT so tests reload exact road GeoDataFrames without re-running `propose` (arterial propose is ~19.5s). Baseline arterial-buildable wall time on 1808: **~19.5s** (un-profiled). Task-6 target block: the ~80-parcel cluster `DJI.3_1_2914,DJI.3_1_2923,DJI.3_1_2925,DJI.3_1_2930` (currently >15-min timeout).

---

### Task 1: Equivalence harness + numpy-vectorized `_sampled_efficiency`

**Files:**
- Create: `tests/test_scoring_equivalence.py`
- Create: `tests/scoring_fixtures.py` (fixture builders shared by the harness)
- Modify: `src/reblock/budget.py` — `_sampled_efficiency` (lines ~166-198)

**Interfaces:**
- Consumes: `network_efficiency(block, roads) -> (E, directness)`, `efficiency_directness_curves(block, roads) -> (Curve, Curve)`, `auc(curve, cap) -> float` (unchanged signatures).
- Produces: a `sampled_fixtures()` helper returning `[(name, block, roads_gdf_or_None, expected_dict)]`, reused by tasks 2-6.

- [ ] **Step 1: Write the equivalence harness (pins CURRENT values).**

`tests/scoring_fixtures.py` builds the fixtures. Reload the 1808 road sets from WKT (no propose):

```python
import json
from pathlib import Path
import geopandas as gpd
from shapely import wkt
from reblock.data.kblock import KblockSource

_REF = json.loads(Path("scratchpad/ref_values_1808.json").read_text())  # committed copy: see Step 3

def _block_1808():
    src = KblockSource("tests/data/kblock/blocks_dji_sample.parquet",
                       "tests/data/kblock/buildings_dji_sample.parquet", "dji",
                       block_ids=["DJI.3_1_1808"])
    return next(iter(src.region().blocks))

def _roads(block, key):
    r = _REF[key]
    if "wkt" not in r:
        return None
    return gpd.GeoDataFrame(geometry=[wkt.loads(w) for w in r["wkt"]], crs=block.parcels.crs)

def sampled_fixtures():
    b = _block_1808()
    return [(k, b, _roads(b, k), _REF[k]) for k in ("no_roads", "dijkstra", "arterial_buildable")]
```

`tests/test_scoring_equivalence.py`:

```python
import math
from reblock.budget import network_efficiency, efficiency_directness_curves, auc
from tests.scoring_fixtures import sampled_fixtures

def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))

def test_network_efficiency_matches_reference():
    for name, block, roads, exp in sampled_fixtures():
        e, d = network_efficiency(block, roads)
        assert _close(e, exp["E"]), (name, "E", e, exp["E"])
        assert _close(d, exp["directness"]), (name, "directness", d, exp["directness"])

def test_curves_and_auc_match_reference():
    for name, block, roads, exp in sampled_fixtures():
        if roads is None or "E_auc" not in exp:
            continue
        ec, dc = efficiency_directness_curves(block, roads)
        for got, want in zip(ec.benefit, exp["E_curve_benefit"]):
            assert _close(got, want), (name, "E_curve", got, want)
        for got, want in zip(dc.benefit, exp["dir_curve_benefit"]):
            assert _close(got, want), (name, "dir_curve", got, want)
        cap = min(ec.cost[-1], dc.cost[-1])
        assert _close(auc(ec, cap), exp["E_auc"]), (name, "E_auc")
        assert _close(auc(dc, cap), exp["dir_auc"]), (name, "dir_auc")
```

- [ ] **Step 2: Commit the reference JSON into the repo** so the harness is hermetic (don't depend on scratchpad). Copy `scratchpad/ref_values_1808.json` → `tests/data/scoring/ref_values_1808.json`; update `_REF` path in `scoring_fixtures.py` to that committed location.

- [ ] **Step 3: Run the harness against CURRENT code — expect PASS.**
Run: `pixi run pytest tests/test_scoring_equivalence.py -v`
Expected: PASS (it pins the current implementation's own output).

- [ ] **Step 4: Vectorize `_sampled_efficiency` with numpy (keep the networkx graph input).**
Replace the shapely per-pair inner loop. Precompute once per call: `rep_xy = np.array([[p.x, p.y] for p in reps])`; `src_euclid[i,j] = hypot(rep_xy[si]-rep_xy[j])` for `si` in `sources` (shape `K×N`); `entry_xy` (N,2) with NaN rows for `None` entries; `legs = np.hypot(rep_xy[:,0]-entry_xy[:,0], rep_xy[:,1]-entry_xy[:,1])` (NaN where entry is None). Per source `si`, build `nd = np.array([dist.get(entry[j], inf) if entry[j] is not None else inf for j in range(N)])` from the existing `nx.single_source_dijkstra_path_length` dict; `d = legs[si] + nd + legs`; valid mask = `(entry[si] is not None) & isfinite(nd) & isfinite(legs) & (d>0)`; accumulate `inv_sum += (1/d)[mask].sum()`, `dir_sum += (src_euclid[i]/d)[mask].sum()`; `pairs` counts all `j != si` (unchanged). Preserve exactly: coincident-entry `nd=0` scored (not dropped), `(0,0)` when no pairs, same source list. Keep the same function signature.

- [ ] **Step 5: Run harness + invariants + full budget tests.**
Run: `pixi run pytest tests/test_scoring_equivalence.py tests/test_budget.py -v`
Expected: PASS (values identical to 1e-9; `directness ∈ [0,1]`; monotonicity tests green).

- [ ] **Step 6: Re-time (record the number in the commit body).**
Run: `pixi run python -c "import time; from tests.scoring_fixtures import _block_1808; from reblock.methods.arterial import GreedyArterialReblocker; b=_block_1808(); t=time.time(); GreedyArterialReblocker(mode='buildable',objective='directness').propose(b); print('arterial 1808:', round(time.time()-t,1),'s')"`
Expected: modestly below the 19.5s baseline (this task removes the ~10% shapely-in-`_sampled_efficiency` cost).

- [ ] **Step 7: Commit.**
```bash
git add tests/test_scoring_equivalence.py tests/scoring_fixtures.py tests/data/scoring/ src/reblock/budget.py
git commit -m "perf: numpy-vectorize _sampled_efficiency + equivalence harness"
```

---

### Task 2: csgraph batched dijkstra in `_sampled_efficiency`

**Files:**
- Modify: `src/reblock/budget.py` — `_sampled_efficiency` (dijkstra), and a new private `_graph_to_csr(g) -> (csr, node_index)` helper.

**Interfaces:**
- Produces: `_graph_to_csr(g: nx.Graph) -> tuple[csr_matrix, dict[node, int]]` — symmetric CSR from an nx graph, deduping undirected pairs (see Global Constraint I3). Reused conceptually by task 3's context (which builds the CSR without nx).

- [ ] **Step 1: Write a CSR-parity unit test.**
```python
# in tests/test_scoring_equivalence.py
def test_csgraph_matches_networkx_distances():
    import networkx as nx, numpy as np
    from scipy.sparse.csgraph import dijkstra
    from reblock.budget import _graph_to_csr
    g = nx.Graph()
    for a, b, w in [((0,0),(1,0),1.0), ((1,0),(2,0),1.0), ((0,0),(2,0),3.0), ((1,0),(1,0),0.0)]:
        if a != b: g.add_edge(a, b, weight=w)
    csr, idx = _graph_to_csr(g)
    src = idx[(0,0)]
    d = dijkstra(csr, directed=False, indices=src)
    ref = nx.single_source_dijkstra_path_length(g, (0,0))
    for node, i in idx.items():
        rv = ref.get(node, float("inf"))
        assert abs(d[i] - rv) <= 1e-12, (node, d[i], rv)
```
- [ ] **Step 2: Run — FAIL (`_graph_to_csr` undefined).** `pixi run pytest tests/test_scoring_equivalence.py::test_csgraph_matches_networkx_distances -v`
- [ ] **Step 3: Implement `_graph_to_csr` + swap the dijkstra.**
`_graph_to_csr`: assign each node an int index (sorted for determinism); build COO arrays from `g.edges(data="weight")`; make symmetric (both `(i,j)` and `(j,i)`); `csr = scipy.sparse.csr_matrix((w, (rows, cols)), shape=(n,n))`. Because nx already deduped, no duplicate pairs here (task 3 handles dedup for the non-nx build). In `_sampled_efficiency`, replace the K `nx.single_source_dijkstra_path_length` calls with one `dist_mat = dijkstra(csr, directed=False, indices=[node_index[entry[si]] for si in sources])` (rows aligned to `sources`); look up `nd` from `dist_mat[row, node_index[entry[j]]]` (inf if entry `None`/absent). `_sampled_efficiency` now takes the CSR + node_index instead of the nx graph, OR builds them internally from the passed graph — keep `network_efficiency`/`_efficiency_factory` callers working (build CSR inside for now).
- [ ] **Step 4: Run — PASS.** `pixi run pytest tests/test_scoring_equivalence.py tests/test_budget.py -v`
- [ ] **Step 5: Re-time (Step 6 of Task 1's command). Expect the ~11% dijkstra cost gone.**
- [ ] **Step 6: Commit.** `git commit -am "perf: batched csgraph dijkstra in _sampled_efficiency"`

---

### Task 3: `_BlockScoringContext` + migrate `network_efficiency` / `_efficiency_factory`

**Files:**
- Modify: `src/reblock/budget.py` — add `class _BlockScoringContext`; rewrite `network_efficiency` and `_efficiency_factory` to use it.
- Modify: `src/reblock/methods/arterial.py` — `_greedy_arterials` builds ONE context per block; `_score` uses it (full re-derivation per candidate for now — not yet incremental).

**Interfaces:**
- Produces:
  - `_BlockScoringContext(block, *, k=40, tol=STREET_TOL)`
  - `.score(roads: GeoDataFrame | None) -> tuple[float, float]` — re-derives entries against streets+roads (matches `network_efficiency`).
  - `.score_frozen(roads_prefix, *, entry, splits) -> tuple[float, float]` — uses the passed frozen entries/splits, builds its own per-prefix CSR (matches `_efficiency_factory`; isolated frozen source → contributes 0).

- [ ] **Step 1: Add a context-parity test** (context `.score` == `network_efficiency`):
```python
def test_context_score_matches_network_efficiency():
    from reblock.budget import _BlockScoringContext, network_efficiency
    for name, block, roads, exp in sampled_fixtures():
        ctx = _BlockScoringContext(block)
        assert _close(ctx.score(roads)[1], network_efficiency(block, roads)[1])
        assert _close(ctx.score(roads)[0], network_efficiency(block, roads)[0])
```
- [ ] **Step 2: Run — FAIL.** 
- [ ] **Step 3: Implement `_BlockScoringContext`.**
Freeze in `__init__`: `reps`, `rep_xy`, `sources`, `src_euclid`; the STREET-only graph as CSR with every parcel's street projection injected as a colinear split node (reuse `_edge_lines`/`_line_entries`/`_split_graph` logic against `block.streets` only), plus `street_edge_lines`, `street_tree`, and per-parcel `(nearest_street_edge_distance, street_entry_node or None)`. **CSR build rule (I3):** assemble undirected edges as a dict keyed by the unordered index pair, last-write-wins; on split injection delete the parent pair first; then one symmetric `csr_matrix`.
`.score(roads)`: derive entries over streets+roads exactly as `_line_entries` does today (full re-derivation), build the per-call CSR (street CSR + road edges + splits, deduped), call the numpy+csgraph `_sampled_efficiency` core.
`.score_frozen(prefix, entry, splits)`: build the per-prefix CSR from the passed frozen entries/splits (an isolated frozen source whose entry node is absent in the prefix contributes 0), same core.
Rewrite `network_efficiency` → `_BlockScoringContext(block).score(roads)`. Rewrite `_efficiency_factory` → build context once, freeze `entry, splits` against the full graph exactly as today, return `lambda prefix: ctx.score_frozen(prefix, entry=entry, splits=splits)`.
In `arterial._greedy_arterials`: build `ctx = _BlockScoringContext(block)` once before the loop; `_score` calls `ctx.score(_planarize(committed+[real]))` for directness/efficiency objectives (still full re-derivation).
- [ ] **Step 4: Run harness + full suite.** `pixi run pytest tests/test_scoring_equivalence.py tests/test_budget.py tests/test_region.py tests/methods/test_arterial.py -v` — all PASS.
- [ ] **Step 5: Re-time. Expect the 51% graph-construction cost to collapse (context built once, not 7220×).**
- [ ] **Step 6: Commit.** `git commit -am "perf: _BlockScoringContext; migrate network_efficiency + factory"`

---

### Task 4: Two-level incremental scorer (highest risk — C1/I2/R1)

**Files:**
- Modify: `src/reblock/budget.py` — add `_StepContext` (or a `ctx.step(committed)` builder) and `ctx.score_candidate(step, real)`.
- Modify: `src/reblock/methods/arterial.py` — `_greedy_arterials` builds a `StepContext` on each commit; scores candidates via `score_candidate`.

**Interfaces:**
- Produces: `ctx.step(committed: GeoDataFrame | None) -> StepContext` (streets ∪ committed: CSR + per-parcel nearest-edge base, rebuilt per commit); `step.score_candidate(real: LineString) -> tuple[float, float]`.

- [ ] **Step 1: Write the incremental-scorer parity test (the C1/I2 gate) — MUST include an aspirational mid-span-crossing trial (R1).**
```python
def test_incremental_scorer_matches_full_rederivation():
    from reblock.budget import _BlockScoringContext, network_efficiency
    from reblock.methods.arterial import _planarize
    from shapely.geometry import LineString
    from tests.scoring_fixtures import _block_1808
    block = _block_1808()
    committed = _planarize([...one committed road as a LineString...], block.crs)  # ≥1 committed
    ctx = _BlockScoringContext(block); step = ctx.step(committed)
    trials = [ LineString([...]),            # buildable-style, meets at a vertex
               LineString([...diagonal crossing a committed road MID-SPAN...]) ]  # R1 case
    for real in trials:
        got = step.score_candidate(real)
        want = network_efficiency(block, _planarize(list(committed.geometry) + [real], block.crs))
        assert _close(got[0], want[0]) and _close(got[1], want[1]), (real.wkt, got, want)
```
(Choose a committed road + a genuinely mid-span-crossing diagonal from block 1808's coordinates; assert byte-equal to 1e-9.)
- [ ] **Step 2: Run — FAIL (`ctx.step`/`score_candidate` undefined).**
- [ ] **Step 3: Implement the two-level incrementalism.**
`ctx.step(committed)`: extend the frozen street context with committed roads — planarize committed, node into the street CSR, inject committed entry splits, and compute each parcel's `(nearest_edge_distance, entry_node)` over streets ∪ committed. `step.score_candidate(real)`: entries = for each parcel `min(step nearest-edge distance, distance to real's edges)` → candidate entry node; **graph = frozen street CSR + a road CSR built from `unary_union([base_merged, real]).explode()`** (base_merged = the step's planarized committed union; this re-nodes committed×trial crossings — R1) + entry splits; call the core. In `_greedy_arterials`, build `step = ctx.step(_planarize(committed))` once per commit (after appending to `committed`), and score each candidate via `step.score_candidate(real)` instead of `ctx.score(_planarize(committed+[real]))`.
- [ ] **Step 4: Run the parity gate + full suite.** `pixi run pytest tests/test_scoring_equivalence.py tests/test_budget.py tests/test_region.py tests/methods/test_arterial.py -v` — all PASS, especially the incremental-scorer parity (byte-match incl. the crossing trial).
- [ ] **Step 5: Re-time. Expect the big drop (`_line_entries` no longer O(N×all-edges) per candidate).**
- [ ] **Step 6: Commit.** `git commit -am "perf: two-level incremental candidate scorer (StepContext + trial delta)"`

---

### Task 5: `_snap` shapely-ufunc weights + incremental `_planarize`

**Files:**
- Modify: `src/reblock/methods/arterial.py` — `_snap` (lines ~93-113), `_planarize` usage in `_greedy_arterials`.

- [ ] **Step 1: Write a `_snap`-parity + proposal-identity test.**
```python
def test_arterial_proposal_wkt_unchanged():
    # arterial buildable roads on 1808 must match the reference WKT exactly
    import json; from pathlib import Path
    from reblock.methods.arterial import GreedyArterialReblocker
    from tests.scoring_fixtures import _block_1808
    ref = json.loads(Path("tests/data/scoring/ref_values_1808.json").read_text())["arterial_buildable"]["wkt"]
    roads = GreedyArterialReblocker(mode="buildable", objective="directness").propose(_block_1808()).roads
    assert sorted(g.wkt for g in roads.geometry) == sorted(ref)
```
- [ ] **Step 2: Run — PASS on current code** (guards against geometry drift; keep it green through this task).
- [ ] **Step 3: Speed `_snap` without changing geometry.** Precompute `edge_midpoints` once per block as a shapely `Point` array. Per chord: `weights = base_length + lam * shapely.distance(edge_midpoints, chord)` (shapely 2.x ufunc — **bit-identical** to `Point.distance`; do NOT reimplement in numpy). Build an `{(u,v): weight}` dict; pass `nx.shortest_path` a weight callback that looks it up. Incrementalize `_planarize`: compute `base_merged = unary_union(committed)` once per step; per candidate `unary_union([base_merged, real])` (this is also the road graph source for Task 4's `score_candidate`).
- [ ] **Step 4: Run proposal-identity + full suite.** `pixi run pytest tests/test_scoring_equivalence.py tests/test_region.py tests/methods/test_arterial.py -v` — proposals WKT-identical.
- [ ] **Step 5: Re-time. Expect the ~9% `_snap` shapely + ~16% planarize costs reduced.**
- [ ] **Step 6: Commit.** `git commit -am "perf: _snap shapely-ufunc weights + incremental planarize"`

---

### Task 6: Remove networkx from the metric path + final measurement

**Files:**
- Modify: `src/reblock/budget.py` — delete dead nx graph builders from the scoring path (`_road_street_graph`/`_split_graph`/`_edge_lines` if no longer used by scoring; keep any still needed by `road_drainage`).
- Modify: `tests/methods/test_arterial.py` — migrate the direct `_road_street_graph` import in `test_aspirational_planarizes_crossings_into_true_intersections`.

- [ ] **Step 1: Grep for residual nx in the scoring path.** `pixi run python -c "import ast,sys; ..."` or `grep -n "networkx\|nx\." src/reblock/budget.py` — confirm the only remaining nx use is `road_drainage`; `_sampled_efficiency`/context are nx-free.
- [ ] **Step 2: Migrate the test import.** Update `test_aspirational_planarizes_crossings_into_true_intersections` to assert the degree-≥4 crossroads via the context's CSR (node degree = CSR row nnz) or the planarized geometry, not `_road_street_graph`. Keep the assertion's meaning.
- [ ] **Step 3: Delete now-dead scoring nx builders** (only if truly unused — check `road_drainage`). Run `pixi run check`.
- [ ] **Step 4: Final measurement — the real proof.**
Run the ~80-parcel cluster with a 15-min timeout:
`timeout 900 pixi run python -m reblock.compare data=dji eval=kcomplexity "block_ids=[[DJI.3_1_2914,DJI.3_1_2923,DJI.3_1_2925,DJI.3_1_2930]]" methods=[dijkstra,greedy_arterial_buildable] hydra.run.dir=/tmp/tierb_final` — must COMPLETE (was a >15-min timeout). Record wall time. Also re-time arterial 1808 (target ~1–4s vs 19.5s).
- [ ] **Step 5: Commit + update the compare-timing note in README if warranted.** `git commit -am "perf: remove networkx from metric scoring path; final timings"`

---

## Self-Review

**Spec coverage:** all 6 spec tasks map 1:1 to Tasks 1-6. The three correctness gates (equivalence harness, incremental-scorer parity with the R1 crossing trial, proposal-WKT-identity) are present. The CSR dedup (I3), committed roads (C1), crossing re-noding (R1), and `_snap` shapely ufunc (R2) each have an explicit implementation step + a guarding test. networkx exemptions (`road_drainage`, `_snap`) are respected.

**Placeholder scan:** the `LineString([...])` coordinates in Task 4 Step 1 and the committed road are deliberately left for the implementer to pick from block 1808's geometry (the exact coordinates are data-dependent) — the implementer MUST choose a genuinely mid-span-crossing diagonal and assert byte-equality; this is called out, not a hidden TODO. All other steps are concrete.

**Type consistency:** `_BlockScoringContext.score` / `.score_frozen` / `.step` / `StepContext.score_candidate` signatures are consistent across Tasks 3-4; `_graph_to_csr` returns `(csr, node_index)` used in Tasks 2-3.
