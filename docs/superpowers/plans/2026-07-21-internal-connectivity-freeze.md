# Internal-Connectivity Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the displayed internal-connectivity (`commute_ratio`) curves read monotone by freezing the averaged parcel set to each sweep's terminal road network, without changing any terminal value or metric semantics — then ship it across all examples.

**Architecture:** `commute_ratio`'s per-prefix non-monotonicity was traced (2026-07-21 investigation) entirely to *composition churn* — parcels flickering in and out of the averaged set as their single-nearest topology edge flips between a road and a bare street segment while roads are added. The fix: determine the averaged membership set `S` ONCE from the terminal (superset) network, and have every prefix in a sweep average over that fixed `S` (a parcel not yet connected contributes `0.0`, rather than being dropped). Frozen-to-self (S from the same roads) is byte-identical to today's dynamic metric, so terminal values and every existing test are unchanged; only intermediate prefixes move, and the composition-churn dips vanish. The freeze lives inside `budget.py` and is wired in at the two reporting call sites (curve factory + matched-budget scalar).

**Tech Stack:** Python, numpy, networkx, shapely (STRtree), geopandas; pytest; pixi run.

## Global Constraints

- **No back-compat shims / no dual code paths (owner directive).** `commute_ratio` gets ONE new keyword `membership`; the default (`None`) path must be *byte-identical* to the current implementation (verified: freeze-to-self ≡ dynamic). Do not keep a separate "old" function or a legacy flag. Migrate the two call sites; delete nothing that is still the single source of truth.
- **Terminal invariance is load-bearing and asserted.** `tests/test_commute_ratio.py::test_benefit_factory_terminal_matches_metric` asserts `f(roads) == commute_ratio(block, roads)` and `curve.benefit[-1] == commute_ratio(block, roads)` with **exact `==`**. The frozen path MUST iterate members in **ascending parcel-index order** (`sorted(membership)`) so its `np.mean` summation order matches the dynamic path's filtered-ascending order exactly — otherwise a last-ULP float difference breaks `==`.
- The raw metric `commute_ratio(block, roads)` (default) remains legitimately non-monotone across *different* road sets; only the *frozen sweep* is monotone-reading. Docstrings must say this precisely — do not claim the metric itself became monotone.
- All existing tests stay green with no value edits (they are range-based or terminal-invariant). Do not weaken or delete an assertion to make it pass; if one genuinely fails, that is a real regression to fix in code.
- `Block`, `Curve`, `GeoDataFrame`, `_Node`, `STREET_TOL`, `_noded_graph`, `_entry_resistance`, `_entry_resistance_ground` already exist in `budget.py` — reuse them; do not reimplement.

---

## File Structure

- `src/reblock/budget.py` — extract the graph-build + per-parcel-ratio internals of `commute_ratio` into two reused helpers (`_commute_setup`, `_nearest_edge_ratio`), add `_commute_membership`, add the `membership` keyword to `commute_ratio`, and freeze `commute_ratio_benefit`. (Task 1 + the factory half of Task 2.)
- `scripts/compare_budgets.py` — freeze the matched-budget lens-B scalar. (Task 2.)
- `tests/test_commute_ratio.py` — add frozen-behavior unit tests; existing tests unchanged. (Task 1.)

Already-in-tree display changes that ride along on the branch but are NOT tasks here (done + user-approved, covered by the final whole-branch review): `src/reblock/emit.py` (`compare_report` benefit-vs-displacement x-axis) and `scripts/gen_multiblock_example.py` (drop clearance, euclidean spacing 250, self-clean stale artifacts).

---

### Task 1: Frozen-membership `commute_ratio` in `budget.py`

**Files:**
- Modify: `src/reblock/budget.py` (refactor `commute_ratio` at lines ~866-957; add three module-level defs above it)
- Test: `tests/test_commute_ratio.py`

**Interfaces:**
- Consumes: existing `_noded_graph`, `_entry_resistance`, `_entry_resistance_ground`, `STREET_TOL`, `_Node`, `Block`.
- Produces (used by Task 2):
  - `_commute_setup(roads: GeoDataFrame | None, streets: GeoDataFrame) -> _CommuteSetup | None`
  - `_nearest_edge_ratio(setup: _CommuteSetup, pt: Point) -> tuple[bool, float]`
  - `_commute_membership(block: Block, roads: GeoDataFrame | None) -> frozenset[int]`
  - `commute_ratio(block: Block, roads: GeoDataFrame | None, *, membership: frozenset[int] | None = None) -> float`

- [ ] **Step 1: Write failing tests for the frozen behavior**

Add to `tests/test_commute_ratio.py` (import updated to include the new names):

```python
from shapely.geometry import Polygon  # already imported; keep
from reblock.budget import (
    _noded_graph, commute_ratio, commute_ratio_benefit, cost_benefit_curve,
    _commute_membership,
)


def test_frozen_membership_matches_dynamic_inclusion() -> None:
    # freeze-to-self is a no-op: S from `roads`, evaluated at `roads`, == the dynamic metric.
    block = _block(3, _parcels_at([(40, 40), (50, 40), (60, 40)]))
    loop = _roads([LineString([(30, 0), (30, 50), (70, 50), (70, 0)])])
    S = _commute_membership(block, loop)
    assert len(S) == 3                                         # all three served parcels are members
    assert commute_ratio(block, loop, membership=S) == commute_ratio(block, loop)


def test_frozen_membership_includes_zeros_for_unconnected() -> None:
    # A frozen member with no interior entry under `roads` contributes 0.0 (not skipped), so a
    # denominator that includes it drags the mean DOWN vs the dynamic (skip-it) metric.
    served = _parcels_at([(40, 40), (50, 40), (60, 40)])
    on_street = Polygon([(9, -1), (11, -1), (11, 1), (9, 1)])  # centroid ~(10, 0): nearest edge is the street
    block = _block(4, served + [on_street])
    loop = _roads([LineString([(30, 0), (30, 50), (70, 50), (70, 0)])])
    dyn = commute_ratio(block, loop)                          # averages the 3 served only
    S = _commute_membership(block, loop)
    assert 3 not in S and len(S) == 3                         # the on-street parcel is not a member
    assert commute_ratio(block, loop, membership=S) == dyn    # freeze-to-self identity
    frozen_all = commute_ratio(block, loop, membership=frozenset(range(4)))
    assert frozen_all < dyn                                   # forcing the 0.0 in lowers the mean


def test_frozen_empty_and_missing_guards() -> None:
    block = _block(3, _parcels_at([(40, 40), (50, 40), (60, 40)]))
    loop = _roads([LineString([(30, 0), (30, 50), (70, 50), (70, 0)])])
    assert _commute_membership(block, _roads([])) == frozenset()
    assert _commute_membership(block, None) == frozenset()
    assert commute_ratio(block, loop, membership=frozenset()) == 0.0      # empty S -> 0.0
    assert commute_ratio(block, None, membership=_commute_membership(block, loop)) == 0.0  # no graph -> 0.0
```

- [ ] **Step 2: Run the new tests, verify they fail**

Run: `pixi run pytest tests/test_commute_ratio.py::test_frozen_membership_matches_dynamic_inclusion tests/test_commute_ratio.py::test_frozen_membership_includes_zeros_for_unconnected tests/test_commute_ratio.py::test_frozen_empty_and_missing_guards -v`
Expected: FAIL with `ImportError: cannot import name '_commute_membership'`.

- [ ] **Step 3: Refactor `commute_ratio` internals into reused helpers + add the frozen path**

In `src/reblock/budget.py`, replace the single `commute_ratio` function (currently lines ~866-957) with the following four definitions. The extracted bodies are the SAME code currently inside `commute_ratio` — do not alter the numerics, only relocate them, so the default path stays byte-identical.

```python
_CommuteSetup = tuple[
    dict[_Node, tuple[dict[_Node, int], NDArray[np.float64]]],  # interior node -> (idx map, its component's grounded G)
    dict[_Node, float],                                          # R_geo per node (multi-source dijkstra)
    list[tuple[_Node, _Node]],                                   # edges
    list[LineString],                                            # edge_lines (index-aligned to edges)
    STRtree,                                                     # STRtree over edge_lines
]


def _commute_setup(roads: GeoDataFrame | None, streets: GeoDataFrame) -> _CommuteSetup | None:
    """Build the planarized road-union-street graph and, per connected component, its grounded
    Green's function (dense inverse of the interior-node Laplacian) + R_geo (multi-source dijkstra
    from the street nodes), plus an STRtree of edges for line-proximity parcel entry. Returns None
    when there is no usable graph: no/empty roads, no graph nodes, or no street node / no interior
    node. Street nodes are those within STREET_TOL of the street geometry (GEOMETRIC test)."""
    if roads is None or len(roads) == 0:
        return None
    g = _noded_graph(roads, streets)
    if g.number_of_nodes() == 0:
        return None
    street_geom = unary_union(list(streets.geometry))
    snodes = {n for n in g.nodes if Point(n).distance(street_geom) <= STREET_TOL}
    interior = [n for n in g.nodes if n not in snodes]
    if not snodes or not interior:
        return None
    for u, v in g.edges():
        g[u][v]["len"] = max(math.hypot(u[0] - v[0], u[1] - v[1]), 1e-6)
    geo = nx.multi_source_dijkstra_path_length(g, snodes, weight="len")
    green: dict[_Node, tuple[dict[_Node, int], NDArray[np.float64]]] = {}
    for comp in nx.connected_components(g):
        comp_streets = comp & snodes
        comp_int = [n for n in comp if n not in snodes]
        if not comp_streets or not comp_int:                            # stranded -> excluded
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
        ginv = np.linalg.inv(lg)                                        # DENSE grounded solve
        for n in comp_int:
            green[n] = (idx, ginv)
    edges = list(g.edges())
    edge_lines = [LineString([u, v]) for u, v in edges]
    return green, geo, edges, edge_lines, STRtree(edge_lines)


def _nearest_edge_ratio(setup: _CommuteSetup, pt: Point) -> tuple[bool, float]:
    """(included, ratio) for parcel point `pt` entering via its single geometrically-nearest
    topology edge. included=False (ratio 0.0) when that edge has NO interior endpoint (the parcel's
    closest frontage is the bare street) or R_geo is non-finite/zero. The caller decides what
    False means: 'skip' (dynamic membership) or 'contribute 0.0' (frozen membership)."""
    green, geo, edges, edge_lines, tree = setup
    j = int(tree.nearest(pt))                                           # line-proximity entry
    ls = edge_lines[j]
    u, v = edges[j]
    proj = ls.project(pt)
    r = max(ls.length, 1e-6)
    a, b = proj, r - proj
    u_int, v_int = u in green, v in green
    if u_int and v_int:
        idx, ginv = green[u]
        guu, gvv, guv = ginv[idx[u], idx[u]], ginv[idx[v], idx[v]], ginv[idx[u], idx[v]]
        r_eff = _entry_resistance(guu, gvv, guv, a, b, r)
    elif u_int:                                                         # v is a street node
        idx, ginv = green[u]                                           # ground dist=b, interior=a
        r_eff = _entry_resistance_ground(ginv[idx[u], idx[u]], b, a, r)
    elif v_int:                                                         # u is a street node
        idx, ginv = green[v]                                           # ground dist=a, interior=b
        r_eff = _entry_resistance_ground(ginv[idx[v], idx[v]], a, b, r)
    else:
        return False, 0.0                                              # both ends on the street
    r_geo = min(geo.get(u, math.inf) + a, geo.get(v, math.inf) + b)
    if not (math.isfinite(r_geo) and r_geo > 1e-9):
        return False, 0.0
    return True, min(max(1.0 - r_eff / r_geo, 0.0), 1.0 - 1e-12)       # clip [0, 1)


def _commute_membership(block: Block, roads: GeoDataFrame | None) -> frozenset[int]:
    """The frozen averaged-parcel set S: indices of parcels with a valid interior entry under
    `roads` (the sweep's terminal/superset network), computed ONCE. A prefix sweep that averages
    over this fixed S has a fixed denominator, which removes the composition churn (parcels
    flickering in/out of the mean) that makes the per-prefix curve non-monotone. Empty if `roads`
    yields no usable graph."""
    setup = _commute_setup(roads, block.streets)
    if setup is None:
        return frozenset()
    return frozenset(i for i, geom in enumerate(block.parcels.geometry)
                     if _nearest_edge_ratio(setup, geom.centroid)[0])


def commute_ratio(block: Block, roads: GeoDataFrame | None, *,
                  membership: frozenset[int] | None = None) -> float:
    """Internal connectivity: mean over parcels of 1 - R(dwelling->street)/R_geodesic on the noded
    road-union-street graph. R = grounded effective resistance to the whole street (a component-wise
    DENSE solve); R_geo = single-best-route (shortest-path) resistance. A single-egress tree route
    -> 0; ->1 as parallel backup routes thicken. Clipped to [0, 1). Rewards added redundancy via
    Rayleigh monotonicity (adding a redundant connector to an existing loop can only help). A small
    tight loop can legitimately outscore a large loose one -- coverage-insensitive by design (see
    access_benefit for coverage). Task-1 corpus gate (2026-07-17): corr(rho, access)=+0.294;
    anti-gaming holds on realistic networks -- loops ADDED to clearance give rho 0.000->TINY
    0.060->BIG 0.278 (BIG >> TINY); a matched-length parallel bundle scores 0.00145/m vs a genuine
    loop's 0.00234/m and costs displacement, so corridor duplication is Pareto-dominated on the
    {external, internal, displacement} suite.

    Each parcel enters by TRUE line-proximity -- the nearest POINT on its nearest topology edge
    (via _entry_resistance/_entry_resistance_ground, computed analytically from the edge's two
    endpoints, so the dense per-component solve never grows with parcel count).

    `membership` selects the averaged set:
    - None (default): DYNAMIC -- exactly the parcels with a valid interior entry under THIS `roads`,
      each contributing its real ratio; a parcel whose nearest edge is a bare street segment, or
      that is unreachable, is skipped. Self-contained per road set. The raw metric is NON-MONOTONE
      across different road sets (a ratio of co-decreasing R/R_geo AND a changing averaged set) --
      standalone reporting ranks by terminal value, never assumes rise.
    - a frozen index set S (from `_commute_membership(block, terminal_roads)`): FROZEN -- every
      parcel in S contributes on EVERY call, its real ratio if it has an interior entry under
      `roads`, else 0.0 ('not yet connected', NOT skipped). Members are averaged in ascending index
      order so a frozen-to-self call (S from the same `roads`) is byte-identical to the dynamic
      default; freezing changes ONLY intermediate prefixes of a sweep, never the terminal value,
      and removes the composition churn so the swept curve reads monotone.

    0.0 with no roads / no parcels / no interior nodes / no usable graph / empty membership."""
    if roads is None or len(roads) == 0 or len(block.parcels) < 1:
        return 0.0
    setup = _commute_setup(roads, block.streets)
    if setup is None:
        return 0.0
    if membership is None:
        ratios = [ratio for geom in block.parcels.geometry
                  for included, ratio in [_nearest_edge_ratio(setup, geom.centroid)] if included]
        return float(np.mean(ratios)) if ratios else 0.0
    if not membership:
        return 0.0
    ratios = [_nearest_edge_ratio(setup, block.parcels.geometry.iloc[i].centroid)[1]
              for i in sorted(membership)]
    return float(np.mean(ratios)) if ratios else 0.0
```

Notes for the implementer:
- Verify `Point`, `LineString`, `STRtree`, `unary_union`, `np`, `nx`, `math`, `NDArray`, `_Node` are already imported at module top (they are used by the current `commute_ratio`); no new imports needed.
- The dynamic-branch list comprehension must preserve **parcel-iteration (ascending) order** and skip `included=False`, exactly as the old `for ... : if ...: continue` loop did — so its `np.mean` equals the old value bit-for-bit.

- [ ] **Step 4: Run the full `commute_ratio` test file**

Run: `pixi run pytest tests/test_commute_ratio.py -v`
Expected: PASS — the three new tests plus all pre-existing tests (including `test_benefit_factory_terminal_matches_metric`, which now exercises the frozen path via `commute_ratio_benefit` — see Task 2; if Task 2 is not yet applied, that test still passes because the default path is unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/reblock/budget.py tests/test_commute_ratio.py
git commit -m "Refactor commute_ratio into reused helpers + add frozen-membership path"
```

---

### Task 2: Freeze the curve factory + matched-budget scalar

**Files:**
- Modify: `src/reblock/budget.py` (`commute_ratio_benefit`, ~lines 960-971)
- Modify: `scripts/compare_budgets.py` (import; `two_lens_rows`, ~lines 108-122)
- Test: `tests/test_commute_ratio.py` (existing `test_benefit_factory_terminal_matches_metric` now covers the frozen factory — no new test needed), `tests/test_compare_budgets.py` (existing, must stay green)

**Interfaces:**
- Consumes: `_commute_membership`, `commute_ratio` (with `membership`) from Task 1.
- Produces: frozen internal-connectivity curves (both `reblock.compare` and `scripts.compare_budgets` paths, which both call `cost_benefit_curve(..., benefit_fn=commute_ratio_benefit)`) and a frozen matched-budget lens-B scalar.

- [ ] **Step 1: Freeze the benefit factory in `budget.py`**

Replace `commute_ratio_benefit` (~lines 960-971) with:

```python
def commute_ratio_benefit(block: Block, roads_full: GeoDataFrame | None, *,
                          tol: float = STREET_TOL) -> Callable[[GeoDataFrame | None], float]:
    """Internal-connectivity benefit factory (shares the access_benefit signature so it plugs into
    cost_benefit_curve(..., benefit_fn=commute_ratio_benefit) and the _sweep frontier). Freezes the
    averaged parcel set S to `roads_full` -- the terminal network of the sweep -- via
    _commute_membership, so every prefix scores commute_ratio over the SAME denominator. This
    removes the composition churn that made the per-prefix curve non-monotone; the terminal value
    is unchanged (frozen-to-self == the dynamic metric). `tol` is unused, kept for the shared
    BenefitFactory signature."""
    del tol
    membership = _commute_membership(block, roads_full)

    def f(roads: GeoDataFrame | None) -> float:
        return commute_ratio(block, roads, membership=membership)
    return f
```

- [ ] **Step 2: Freeze the matched-budget lens-B scalar in `compare_budgets.py`**

In `scripts/compare_budgets.py`, add `_commute_membership` to the `from reblock.budget import (...)` block (near line 50, alongside `commute_ratio`, `commute_ratio_benefit`, `cost_benefit_curve`).

Then in `two_lens_rows` (the loop at ~lines 108-122), freeze the scalar to the method's FULL road set so the lens-B point is consistent with the (now frozen) internal curve. Change:

```python
        prefix_b = truncate_to_length(block, roads, budget_m)
        lens_b.append(LensBRow(
            method=name, budget_m=budget_m,
            external_connectivity=ext_factory(prefix_b),
            internal_connectivity=commute_ratio(block, prefix_b),
            displacement=displacement(block.building_points, radii, prefix_b, corridor_m),
            pct_displaced=pct_displaced(prefix_b, corridor_m, block.building_points, radii)))
```

to:

```python
        prefix_b = truncate_to_length(block, roads, budget_m)
        internal_membership = _commute_membership(block, roads)   # freeze to the method's FULL network
        lens_b.append(LensBRow(
            method=name, budget_m=budget_m,
            external_connectivity=ext_factory(prefix_b),
            internal_connectivity=commute_ratio(block, prefix_b, membership=internal_membership),
            displacement=displacement(block.building_points, radii, prefix_b, corridor_m),
            pct_displaced=pct_displaced(prefix_b, corridor_m, block.building_points, radii)))
```

Also update the `two_lens_rows` docstring clause "scores external (`access_benefit`) + internal (`commute_ratio`) connectivity" to note the internal scalar is frozen to the method's full network (membership from `roads`), matching the internal curve.

- [ ] **Step 3: Run the affected test files**

Run: `pixi run pytest tests/test_commute_ratio.py tests/test_compare_budgets.py -v`
Expected: PASS. `test_benefit_factory_terminal_matches_metric` passes because frozen-to-self at the terminal equals the dynamic metric (exact `==`, guaranteed by ascending-order averaging). `test_compare_budgets.py` lens-B assertions are range-based (`internal_connectivity >= 0.0`) so remain green.

- [ ] **Step 4: Run the full suite**

Run: `pixi run pytest -q`
Expected: PASS with no value edits to any test. If any test fails, diagnose: a genuine regression is a code bug to fix (do not edit the assertion). The static fixture `tests/data/example_fixture/lens_b_matched.csv` is input-only (no test golden-compares its internal values) and needs no regeneration.

- [ ] **Step 5: Commit**

```bash
git add src/reblock/budget.py scripts/compare_budgets.py tests/test_commute_ratio.py
git commit -m "Freeze internal-connectivity curve + matched-budget scalar to terminal membership"
```

---

## Post-Implementation (controller-run, after both tasks pass review — NOT SDD tasks)

These are mechanical/orchestration steps the controller performs after the whole-branch review, before the finishing-a-development-branch flow:

1. The already-in-tree display changes (`emit.py` `compare_report` x-axis, `gen_multiblock_example.py` drop-clearance/euclidean-250/self-clean) are committed as part of this branch — the final whole-branch review covers them.
2. Regenerate ALL examples with the frozen + display-updated pipeline:
   `pixi run bash scripts/regenerate_examples.sh` (regenerates every multiblock metric×city + method-comparison).
3. Commit the regenerated `examples/**` artifacts (curves now read monotone; terminal values and method-comparison headline numbers are unchanged because the freeze is a no-op at the terminal).
4. Open the PR.

## Self-Review

- **Spec coverage:** freeze curve (Task 2 factory) ✓; freeze scalar (Task 2) ✓; default byte-identical (Task 1) ✓; monotone-reading curves (frozen sweep) ✓; ship to all examples (post-impl) ✓.
- **Placeholder scan:** none — all code is concrete.
- **Type consistency:** `membership: frozenset[int] | None`; `_commute_membership -> frozenset[int]`; `_CommuteSetup` alias reused by `_commute_setup`/`_nearest_edge_ratio`. `commute_ratio_benefit` returns `Callable[[GeoDataFrame | None], float]` (unchanged signature). Names match across tasks.
