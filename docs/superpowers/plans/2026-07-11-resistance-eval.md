# Grounded Resistance Eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `resistance` cost-benefit lens (grounded effective resistance to egress) as a 4th compare metric, reusing the `_BlockScoringContext` machinery. Purely additive.

**Architecture:** `_resistance_core` (grounded Laplacian solve) → `_BlockScoringContext.resistance_frozen` → `resistance_benefit` factory (mirrors `_efficiency_factory`) → one line in `compare.py`.

**Tech Stack:** numpy, scipy.sparse (`linalg.factorized`, `csgraph.connected_components`), the post-Tier-B budget machinery.

**Spec:** `docs/superpowers/specs/2026-07-11-resistance-eval-design.md` — read it. **Reference solve:** `scratchpad/spectral_proto.py` (`_laplacian`, `_reachable_from_ground`, `grounded_resistances`) — proven math, but adapt from nx to the current CSR machinery (`_derive_entries`/`_build_csr`/`rep_xy`); the old `_road_street_graph`/`_split_graph` it imports were deleted by Tier B.

## Global Constraints

- **Purely additive:** no existing metric value changes; `access`/`efficiency`/`directness` + all existing tests stay identical.
- **The metric is the INTENSIVE per-parcel `mean(R_i)`** — NOT the extensive `wᵀL_G⁻¹w` or raw Kirchhoff index (they rank road sets wrongly; investigation §3).
- **Monotone:** entries frozen against the full road set; `benefit` non-decreasing across prefixes; `benefit(∅)==0`.
- Verify each task with `pixi run check` (ruff + mypy --strict + pytest), run to completion.

---

### Task 1: `_resistance_core` — grounded Laplacian solve

**Files:** Modify `src/reblock/budget.py` (add `_resistance_core`); Create `tests/test_resistance.py`.

**Interfaces — Produces:**
`_resistance_core(csr: csr_matrix, node_index: dict[_Node, int], entry: list[_Node | None], rep_xy: np.ndarray, ground_idx: np.ndarray, cap: float) -> float` — mean per-parcel grounded resistance-to-egress.

- [ ] **Step 1: Write unit tests** (`tests/test_resistance.py`). Use BARE imports if any test helpers, but this task's tests build tiny CSRs directly.
  - `test_two_parallel_wires`: ground node g, free node a, two unit-length edges g–a (as one deduped edge of length 1 → conductance 1; or two nodes to test parallel). Simplest analytic: a single free node `a` at distance 1 from ground via one edge → `(L_G⁻¹)_{aa} = 1.0`; with an entry at `a` and `leg=0`, `R = 1.0`. Then a second parallel path halves it to 0.5. Assert `_resistance_core` returns these (`cap` large, one parcel whose entry is `a`, `rep_xy` s.t. `leg=0`).
  - `test_unreached_is_cap`: a parcel whose `entry` is `None` → `R_i == cap`; a parcel whose entry node is in a component with NO ground → `cap`.
  - `test_entry_on_ground_is_leg_only`: a parcel whose entry node is a ground node → `R_i == leg_i` (drive term 0).
  - `test_intensive_mean`: build a graph; compute mean R over 2 parcels. Add an isolated extra (ungrounded, no-parcel) node to the CSR → the mean over the SAME 2 parcels is unchanged (guards the intensive-mean requirement).
- [ ] **Step 2: Run — FAIL** (`_resistance_core` undefined). `pixi run pytest tests/test_resistance.py -v`
- [ ] **Step 3: Implement `_resistance_core`** per spec §Design.1: conductance `1/csr.data`; `L = diags(deg) − C`; `connected_components` → grounded components → `reach`; `free = ~ground & reach`; `LG = L[free][:,free]`; `factorized(LG)`; per-distinct-entry-node diagonal solve; `R_i = leg + diag` (or `leg` if entry on ground, `cap` if unreached); `float(mean(R_i))`. Handle `free` empty (all `cap`), no parcels, `n<1`.
- [ ] **Step 4: Run — PASS.** `pixi run pytest tests/test_resistance.py -v`
- [ ] **Step 5: `pixi run check` green; commit** `perf: _resistance_core grounded Laplacian solve`.

---

### Task 2: `resistance_frozen` + `resistance_benefit` factory

**Files:** Modify `src/reblock/budget.py` (`_BlockScoringContext.__init__`, `_ground_indices`, `resistance_frozen`; `resistance_benefit`); add tests to `tests/test_resistance.py`.

**Interfaces:**
- Consumes: `_resistance_core`, `_BlockScoringContext._derive_entries`, `_build_csr`, `_explode_segments`, `ctx.rep_xy`/`street_segs`/`n`.
- Produces: `resistance_benefit(block, roads_full, *, tol=STREET_TOL, k=40) -> Callable[[GeoDataFrame|None], float]` (matches `BenefitFactory`).

- [ ] **Step 1: Write tests.**
  - `test_tree_equals_shortest_path`: build a block + a dijkstra (tree) road set; for each parcel, its `R_i` (via a direct `resistance_frozen` on the full roads) equals its door-to-door shortest-path distance to the nearest street (reuse `network_efficiency`'s entry/leg + a Dijkstra to ground) to ~1e-6. (A hand-built deep grid block is cleanest.)
  - `test_benefit_monotone_and_zero_at_empty`: `f = resistance_benefit(block, roads); ` over `cost_benefit_curve`-style growing prefixes of a real dijkstra/arterial road set, `f(prefix)` is non-decreasing and `f(None)==0`, `f` values finite.
  - `test_benefit_in_sane_range`: `0 <= f(full) <= 1` (R only drops as roads add, so benefit ∈ [0,1]).
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** per spec §Design.2–3: freeze `self.cap` (bbox diagonal) + `self.streets_geom` in `__init__`; `_ground_indices(node_index)` (vectorized `shapely.distance(points, streets_geom) <= tol`); `resistance_frozen(prefix, *, entry, splits)` mirroring `score_frozen`'s CSR build then `_resistance_core`; `resistance_benefit` mirroring `_efficiency_factory` (freeze entries, `R0 = resistance_frozen(None,...)`, `f(prefix) = (R0 - resistance_frozen(prefix,...))/R0`).
- [ ] **Step 4: Run — PASS** + `pixi run pytest tests/test_budget.py tests/test_scoring_equivalence.py -q` (confirm existing metrics untouched).
- [ ] **Step 5: `pixi run check` green; commit** `feat: resistance_benefit factory (grounded egress-resistance lens)`.

---

### Task 3: Wire into `compare.py` + AUC sanity + docs

**Files:** Modify `src/reblock/compare.py`; Modify `README.md` (compare section) + `docs/metrics-north-star.md` (mark Piece 2 adopted). Add an AUC-sanity test to `tests/test_resistance.py`.

- [ ] **Step 1: Write the AUC-sanity test** `test_resistance_auc_ranks_sensibly`: on ≥3 DJI blocks (incl. a deep one), compute `cost_benefit_curve(block, roads, benefit_fn=resistance_benefit)` for a dijkstra and an arterial road set; assert AUCs finite, both > the empty baseline's, and on the deep block the through-road (arterial) resistance AUC ≥ dijkstra's (grounded R credits the egress shortcut). Keep block/road construction cheap (dijkstra is fast; use a small hand-built deep region if arterial is slow).
- [ ] **Step 2: Run — FAIL** (or reveal ranking).
- [ ] **Step 3: Wire `compare.py`:** import `resistance_benefit`; in the per-(region,method) loop add `resistance = cost_benefit_curve(block, roads, benefit_fn=resistance_benefit, cost=cost, corridor_m=cfg.get("corridor_m",3.0), tol=STREET_TOL)` and `raw.append((name, label, "resistance", resistance))`; update the "three lenses" comment to four.
- [ ] **Step 4: Run** the sanity test + a smoke compare: `pixi run python -m reblock.compare data=dji eval=kcomplexity "block_ids=[[DJI.3_1_1808]]" methods=[dijkstra,mesh,greedy_arterial_buildable]` — confirm `auc_table_resistance.csv` + `curve_resistance_*.png` emit.
- [ ] **Step 5: Docs:** README compare section — add `resistance` to the graded lenses with a one-line description (egress resistance, redundancy-aware, lower-is-better→benefit). `docs/metrics-north-star.md` — change Piece 2 from "prototype" to "adopted 2026-07-11 (compare lens)".
- [ ] **Step 6: `pixi run check` green; commit** `feat: wire resistance lens into compare + docs`.

## Self-Review

- **Spec coverage:** the 3 tasks cover the core (`_resistance_core`), the factory+context (`resistance_benefit`/`resistance_frozen`), and the wiring+docs+AUC sanity. All 7 spec correctness gates map to a test (1→t2 tree, 2→t1 known-network, 3→t2 monotone, 4→t1 intensive-mean, 5→t1 cap, 6→t3 AUC sanity, 7→t2/t3 additive-invariant).
- **Placeholder scan:** the analytic resistor value in Task 1 Step 1 (`1.0` single edge, `0.5` two parallel) and the tree-distance in Task 2 are concrete; the implementer picks exact node coords.
- **Type consistency:** `_resistance_core` signature is fixed in Task 1 and consumed unchanged in Task 2; `resistance_benefit` matches the `BenefitFactory` signature used by `cost_benefit_curve`.
