# Remove the Dijkstra and Mesh baseline reblockers — Implementation Plan

> **For agentic workers:** execute via superpowers:subagent-driven-development (fresh implementer per
> task + task review). Steps use checkbox syntax.

**Goal:** Remove the `DijkstraReblocker` and `MeshReblocker` baseline methods entirely, relocating the
shared graph helpers they house, and retire the now-dead `auc`/`efficiency_directness_curves`/
`_efficiency_factory` machinery (whose only remaining users were the arterial-vs-dijkstra directness
tests).

**Owner decisions (2026-07-22, via clarifying Qs):**
- **Scope = "cull mesh":** remove BOTH `DijkstraReblocker` and `MeshReblocker`; delete
  `_reblock_dijkstra` (mesh's forest routine); RELOCATE `_boundary_graph`/`_rnd` (needed by
  arterial/arterial_lazy/loop_closure/substrates). `peel` is NOT in scope.
- **Arterial-directness validation = "delete + retire auc":** delete the 4 arterial-vs-dijkstra tests
  and retire `auc`/`efficiency_directness_curves`/`_efficiency_factory` (now truly dead). Arterial's
  directness SCORING stays tested via `network_efficiency` (3 kept `test_budget.py` tests).
- **Timing:** in THIS branch (`dual-target-connectivity`), BEFORE Task 8 regeneration.
- **Replacements:** `DijkstraReblocker`-as-road-producer utility usages + the `conf/config.yaml`
  default → `ClearanceReblocker` (`reblock.methods.clearance.ClearanceReblocker`).

**Migrate, never accommodate (owner standing directive):** no back-compat shims, no dead code left.

## Global Constraints
- Run everything via pixi: `pixi run pytest -q`, `pixi run ruff check`, `pixi run mypy src scripts`.
- The full suite must stay green after each task. Pre-existing baseline lint/type debt: ~10 ruff
  E501s + 6 mypy errors in `scripts/fetch_kblock_fixtures.py` — introduce none beyond these.
- `nx.multi_source_dijkstra` / `scipy.sparse.csgraph.dijkstra` are the shortest-path ALGORITHM (used
  in `budget.py`, `clearance.py`, `euclidean_grid.py`) — NOT the method. Never touch these.
- Commit trailers on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01MLHAJnMJzWeR7xN725dFkg
  ```
- Do NOT push; stay on `dual-target-connectivity`.

---

## Task D1: Relocate the shared boundary-graph helpers (pure refactor, no behavior change)

**Files:** Create `src/reblock/methods/boundary_graph.py`; Modify `src/reblock/methods/dijkstra.py`,
`src/reblock/methods/arterial.py`, `src/reblock/methods/arterial_lazy.py`,
`src/reblock/methods/loop_closure.py`, `src/reblock/methods/mesh.py`,
`src/reblock/methods/substrates.py`, `src/reblock/derive_graph.py`.

Move `_boundary_graph` and `_rnd` VERBATIM out of `dijkstra.py` into a new
`src/reblock/methods/boundary_graph.py` (keep their exact signatures, bodies, and any module-level
constants/imports they depend on — e.g. `_rnd`'s rounding precision). Leave `_reblock_dijkstra` and
`DijkstraReblocker` in `dijkstra.py` for now (D2 deletes the file).

Re-point every importer to the new home (grep `from reblock.methods.dijkstra import` and
`from reblock.methods import dijkstra` to be exhaustive):
- `arterial.py:41`, `arterial_lazy.py:39`: `from reblock.methods.boundary_graph import _boundary_graph, _rnd`
  (arterial_lazy imports both; arterial imports both).
- `loop_closure.py:27`: `from reblock.methods.boundary_graph import _boundary_graph`.
- `mesh.py:21`: import `_boundary_graph, _rnd` from `boundary_graph`; keep `_reblock_dijkstra` imported
  from `dijkstra` (mesh + that import are deleted in D2).
- `substrates.py:125,149`: replace `from reblock.methods import dijkstra as dijkstra_mod` +
  `dijkstra_mod._boundary_graph`/`dijkstra_mod._rnd` with `from reblock.methods import boundary_graph`
  and `boundary_graph._boundary_graph`/`boundary_graph._rnd`.
- `dijkstra.py` itself: `_reblock_dijkstra` uses `_boundary_graph`/`_rnd` — import them back from
  `boundary_graph` so the (still-present-until-D2) DijkstraReblocker keeps working.
- `derive_graph.py`: the hashed-source list (~line 44) lists `methods/dijkstra.py` + `methods/mesh.py`.
  Leave those two entries for D1 (files still exist); ADD `methods/boundary_graph.py` to the list so the
  new source is provenance-hashed like its siblings.

**Verify:** `pixi run pytest -q` fully green (pure move — zero behavior change; all method tests,
including `test_dijkstra.py`/`test_mesh.py`, still pass). ruff + mypy clean on touched files.

**Commit:** `Relocate _boundary_graph/_rnd out of dijkstra into methods/boundary_graph`.

---

## Task D2: Delete dijkstra + mesh, retire the dead auc cluster, migrate tests/configs/fixtures

**Files:** Delete `src/reblock/methods/dijkstra.py`, `src/reblock/methods/mesh.py`,
`tests/methods/test_dijkstra.py`, `tests/methods/test_mesh.py`, `conf/method/dijkstra.yaml`,
`conf/method/mesh.yaml`; Modify `src/reblock/budget.py`, `src/reblock/derive_graph.py`,
`conf/config.yaml`, `conf/compare_config.yaml`, `tests/test_scoring_equivalence.py`,
`tests/scoring_fixtures.py`, `tests/data/scoring/ref_values_1808.json`, `tests/test_region.py`,
`tests/methods/test_arterial.py`, `tests/test_budget.py`, `tests/test_compare_budgets.py`,
`tests/test_run.py`.

### D2.1 — delete the methods + their direct tests + configs
- Delete `src/reblock/methods/dijkstra.py` (DijkstraReblocker + `_reblock_dijkstra`; the relocated
  helpers already live in `boundary_graph.py`) and `src/reblock/methods/mesh.py` (MeshReblocker).
- Delete `tests/methods/test_dijkstra.py`, `tests/methods/test_mesh.py`.
- Delete `conf/method/dijkstra.yaml`, `conf/method/mesh.yaml`.
- `conf/config.yaml`: change the default `- method: dijkstra` → `- method: clearance`.
- `conf/compare_config.yaml:13`: the comment lists the deliberately-absent baselines `dijkstra`,
  `mesh`, `peel` — reword to drop the two removed names (leave `peel`).
- `derive_graph.py` hashed-source list: remove the `methods/dijkstra.py` and `methods/mesh.py` entries
  (keep `boundary_graph.py` added in D1).

### D2.2 — retire the now-dead auc cluster from budget.py
Once the 4 tests below are gone, `auc`, `efficiency_directness_curves`, `_efficiency_factory` have zero
callers (grep-confirm across `src/`/`scripts/`/`tests/` before deleting). Delete all three. Update
`network_efficiency`'s docstring to drop the `efficiency_directness_curves` mention. Remove any import
left unused by the deletion (e.g. if `auc`/`_efficiency_factory` were the sole users of some import).

### D2.3 — delete the 4 arterial-vs-dijkstra tests (they used the retired auc + dijkstra baseline)
- `tests/test_scoring_equivalence.py`: delete `test_curves_and_auc_match_reference` (uses `auc` +
  `efficiency_directness_curves`) and drop the now-unused `auc`/`efficiency_directness_curves` imports.
  KEEP the network_efficiency tests (`test_network_efficiency_matches_reference`,
  `test_context_score_matches_network_efficiency`, `test_one_context_scores_many_road_sets`,
  `test_incremental_scorer_matches_full_rederivation`, `test_greedy_routes_aspirational_to_full_rederivation`).
- `tests/test_region.py`: delete `test_region_reblock_arterial_beats_dijkstra_with_a_margin_on_a_wide_region`
  and `test_greedy_arterial_beats_dijkstra_directness_auc_on_a_deep_region` (both import + use
  `auc`/`efficiency_directness_curves`/`DijkstraReblocker`). Drop the `auc`/`efficiency_directness_curves`
  imports if now unused.
- `tests/methods/test_arterial.py`: delete `test_buildable_arterial_more_direct_than_dijkstra` (~396,
  uses `auc`/`efficiency_directness_curves`/`DijkstraReblocker`). Drop the now-unused imports.

### D2.4 — clean the scoring reference fixture (no golden-value regeneration needed)
`scoring_fixtures.py`/`ref_values_1808.json` STORE road geometry keyed by label; nothing calls
`DijkstraReblocker` at runtime. After D2.3, `sampled_fixtures()` (which fed only the deleted
`test_curves_and_auc_match_reference`) and the `_REF_EXTRA`/`deep_region_*` fixtures + `_region_deep()`
are unused — grep-confirm, then delete them from `scoring_fixtures.py`. In `ref_values_1808.json`,
remove the auc/curve fields (`E_curve_benefit`, `dir_curve_benefit`, `E_auc`, `dir_auc`) that only the
deleted test read. The surviving `test_one_context_scores_many_road_sets` replays the `no_roads`/
`dijkstra`/`arterial_buildable` stored road sets through `network_efficiency`: RENAME the `"dijkstra"`
key → `"least_cost"` in BOTH `ref_values_1808.json` and the test's key tuple (the stored geometry and
its `E`/`directness` golden values are unchanged — it is a fixed road set whose provenance was
dijkstra; only the vestigial label changes). Delete the separate deep-region reference file if one
exists solely for the deleted AUC tests (grep for its path in `scoring_fixtures.py`).

### D2.5 — migrate the DijkstraReblocker road-producer utility usages → ClearanceReblocker
These tests only need *a reblocker that produces some roads*; swap `DijkstraReblocker()` →
`ClearanceReblocker()` (`from reblock.methods.clearance import ClearanceReblocker`), and update any
`{"dijkstra": ...}` method-dict label → `"clearance"`:
- `tests/test_budget.py:44,75,111` (roads for the network_efficiency / displacement tests).
- `tests/test_compare_budgets.py:94,158,180` (the `run_permeability_lenses` smoke tests).
- `tests/test_run.py:279,299` (reblock_block smoke).
- `tests/test_region.py:197,205` (region_reblock smoke — the NON-AUC survivors).
- If ClearanceReblocker needs params to run on these tiny grid/fixture blocks (it is depth-targeted),
  pass the minimal sensible params (e.g. a small `depth_target`/`max_roads`) so it produces a
  non-empty road set; adjust any assertion that pinned a dijkstra-specific road count to be
  method-agnostic (assert non-empty / monotone, not an exact count). Prefer the smallest change that
  keeps each test meaningful.

**Verify:** `pixi run pytest -q` fully green; grep-confirm ZERO remaining references to
`DijkstraReblocker`, `MeshReblocker`, `_reblock_dijkstra`, `auc`, `efficiency_directness_curves`,
`_efficiency_factory` in `src/`/`scripts/`/`tests/` (docs/ historical design specs may retain prose
mentions — leave those). ruff + mypy: no new errors vs baseline.

**Commit:** `Remove dijkstra + mesh reblockers and the dead auc/directness-curve machinery`.

---

## Self-Review
- **Scope coverage:** relocate helpers (D1) ✓; delete both methods + configs (D2.1) ✓; retire auc
  cluster (D2.2) ✓; delete 4 arterial-vs-dijkstra tests (D2.3) ✓; clean scoring fixture (D2.4) ✓;
  migrate utility usages + default (D2.5) ✓.
- **KEEP (do not delete):** `_boundary_graph`/`_rnd` (relocated), `network_efficiency` +
  `_BlockScoringContext`/`_StepContext` (live arterial scoring), `access_burden`, `clearance`, `peel`,
  the shortest-path algorithm calls, the network_efficiency tests.
- **No dead code:** D2.2/D2.4 grep-confirm zero callers before deleting; the fixture's auc fields +
  unused generators go with the tests that read them.
