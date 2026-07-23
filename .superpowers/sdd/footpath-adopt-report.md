# Adopt r0-corridor footpath conductance (g_walk=0.1, adaptive r0) — implementation report

Worktree: `/home/gchurchill/src/reblock/.claude/worktrees/agent-a5d41842df0c57203` (isolated;
not pushed). Note: the task asked for this report at
`/home/gchurchill/src/reblock/.superpowers/sdd/footpath-adopt-report.md` (the outer checkout);
this agent is sandboxed to the worktree, so it is written at the worktree-local mirror of that
path instead — same relative path, `.claude/worktrees/agent-a5d41842df0c57203` prefixed.

## The new conductance

`src/reblock/permeability.py`:

- **Footpath edge (i, j)**: `g_walk * scale(r0, block) * max(eps, 1 - 2*r0/dist(i,j))`, where
  `scale(r0, block) = median(1/dist over the WHOLE adjacency mesh) / median(max(eps, 1-2*r0/dist)
  over the whole mesh)` — i.e. the raw open-corridor-fraction shape `max(eps, 1-2*r0/dist)` is
  rescaled per block so its **median** over the mesh equals what the plain `g_walk/dist` model's
  median would be at the same `g_walk`. `eps = FOOTPATH_EPS = 0.02` (module constant, matches the
  scratchpad).
  - This fair-normalization (`_footpath_conductance`) is **not** optional cosmetic scaling: the
    scratchpad's winning result (`combination_experiment.py`/`r0_sweep.py`) was itself produced
    under exactly this median-match recipe (`k = median(uniform g_walk/d) / median(shape)`), which
    keeps `g_walk` playing its *original* role as the footpath/road **balance** knob (permeability
    is invariant to scaling every conductance together, but roads are pinned at a fixed
    `g_road/dist`, so only the footpath level relative to that fixed road level moves the metric).
    A literal unscaled `g_walk * max(eps, 1-2r0/dist)` does **not** reproduce the tested balance
    (its own median differs from `g_walk/dist`'s by a large, r0-and-region-dependent factor) — I
    implemented the calibrated (median-matched) form, not the literal one, because only the
    calibrated form is what was actually validated in the scratchpad experiments.
  - Additionally **capped at `g_road/dist`** (`np.minimum` in `egress_power`) so a footpath edge
    can never out-conduct the road it would upgrade to, **for every edge, unconditionally** (see
    "Monotonicity" below) — this cap is a no-op on the calibration region (all mesh edges there
    are well under the crossover distance) but is required for correctness on other regions/params
    where the raw corridor shape's constant asymptote (`~scale`) can exceed `g_road/dist` for long
    edges.
- **Road-covered edge**: unchanged, `g_road / dist`.
- **Ground (`g_street`) edges**: unchanged.
- `r0 = r0_frac * median(NN)` — `NN` = nearest-neighbour distance among `block.building_points`,
  computed via `reblock.budget.building_radii(block.building_points, corridor_m)` (`NN = 2 *
  building_radii`, since `building_radii` already returns `NN/2`). Fewer than 2 building points →
  `r0 = 0.0`, which degenerates the shape to a constant 1 everywhere → `_footpath_conductance`
  collapses exactly to the pre-change `g_walk/dist` baseline (safe fallback for blocks without a
  building-point cloud, e.g. every synthetic test fixture in this repo).
- `r0` and `p0` (no-roads baseline) and `adj` (adjacency) are all block-invariant (independent of
  road prefix), so `permeability_curve` computes each ONCE and threads them through every
  `egress_power`/`permeability` call in its sweep, mirroring the existing `adj`/`p0` freeze
  pattern.

## FRAC (r0_frac) chosen: **0.55**

`r0 = 0.55 * median(NN)`. On `multiblock_density_compactness` (capetown, seed
`ZAF.9.3.1_1_44531`), median building NN ≈ 5.198 m → `r0 ≈ 2.86 m`, inside the scratchpad's
measured flat plateau (r0 in [1.5, 3.5] m all give ~11–11.8 pts D=10% spread; `r0_sweep.log`).
This is the FRAC value suggested directly in the task brief ("FRAC ≈ 0.55" to land r0≈3m on this
region) — I did not need to deviate from it; the production run (below) confirms it lands in
the target band.

## g_walk chosen: **0.1** (PermeabilityParams default, `conf/permeability.yaml`)

Per the task's explicit ask, `g_walk` is the calibrated multiplier in the fair-normalized formula
above (not a raw, unscaled multiplier on the corridor shape) — `_footpath_conductance`'s
`target_median = g_walk * median(1/dist)` is exactly the median footpath conductance a plain
`g_walk/dist` model would give at this same `g_walk`, so lowering `g_walk` 1.0 → 0.1 (roads fixed
at `g_road=20`) reproduces the same footpath/road **balance** shift the scratchpad's
`combination_experiment.py` swept and found best at `g_walk=0.1` (uniform-model spread 8.27pts →
r0-corridor spread 11.31pts at r0=2.08m; r0_sweep.py then found r0=3.0m does slightly better,
11.82pts, still on the plateau).

## Reproduction validation (production code, not the scratchpad replica)

Ran `reblock.budget.displacement_curve` + `reblock.permeability.permeability_curve` (the actual
shipped functions, imported from this worktree's `src/reblock/permeability.py`) on the SAME
cached region + per-method reblock (`clearance_looped`, `euclidean_grid`, `osm_footpaths`,
`greedy_arterial_repulsion`) the scratchpad experiments used, with `PermeabilityParams` loaded
from this worktree's `conf/permeability.yaml` via `scripts.compare_budgets.load_permeability_config`
(script: `/tmp/.../scratchpad/validate_production_r0.py`):

```
params: g_walk=0.1 g_road=20.0 g_street=20.0 corridor_m=3.0 r0_frac=0.55
region seed=ZAF.9.3.1_1_44531, block parcels=4677, buildings=4675

  greedy_arterial_repulsion: terminal permeability=27.7%  terminal displacement=4.0%  perm@D=10%=NA
  clearance_looped: terminal permeability=84.5%  terminal displacement=18.0%  perm@D=10%=73.6%
  euclidean_grid: terminal permeability=63.9%  terminal displacement=10.9%  perm@D=10%=61.9%
  osm_footpaths: terminal permeability=91.5%  terminal displacement=32.2%  perm@D=10%=66.5%

D=10% method-spread across (clearance_looped, euclidean_grid, osm_footpaths): 11.72 pts
(leader=clearance_looped, trailing=euclidean_grid)
```

**11.72 pts** — squarely in the target ~10–12 pt range (old baseline was ~0.9 pts; a ~13x
improvement), and consistent with `r0_sweep.log`'s r0=3.0m point (11.82pts) — r0=2.86m sits
between the r0=2.08m (11.31pts) and r0=3.0m (11.82pts) reference points, interpolating to ~11.7,
which is exactly what was measured. **Terminal spread** (91.5-63.9=27.6 pts) also matches
`r0_sweep.log`'s r0=3.0 reference (27.65 pts) closely. Status: **reproduced, not blocked.**

## Property preservation

- **Monotone** (roads only add conductance): guaranteed **unconditionally** via
  `footpath_g = np.minimum(_footpath_conductance(...), g_road/dist)` in `egress_power` — every
  edge's footpath conductance is capped at what its road upgrade would give, for every edge,
  regardless of region/params (not just verified once on the calibration region). Regression
  tests: `test_footpath_conductance_can_exceed_road_conductance_without_a_clamp` (hand-verified,
  proves the raw uncapped formula CAN violate this — a real failure mode, not hypothetical — and
  that the cap fixes it); existing `test_monotone_under_added_roads` and
  `test_permeability_curve_is_monotone_non_decreasing` still pass unmodified.
- **Bounded ∈ [0, 1)**: unchanged code path (`1 - P1/P0`); `test_permeability_curve_starts_at_zero_
  and_is_bounded` still passes.
- **All-parcels / permeability(no roads) = 0**: unchanged; `test_no_roads_permeability_is_zero`
  still passes.
- **r0-corridor form** (focused test): `test_footpath_conductance_cramped_edge_lower_than_open_edge`
  — a cramped edge (dist=4m, inside 2·r0=6m, floors at eps) gets strictly lower conductance than
  an open edge (dist=40m, well beyond 2·r0); hand-computed exact values, `pytest.approx`-checked.
- **Adaptive r0** (focused test): `test_adaptive_r0_scales_with_median_nearest_neighbor_distance`
  — r0 scales linearly with median building NN spacing (dense vs sparse point fields, `r0_frac`
  factored out exactly); `test_adaptive_r0_falls_back_to_zero_without_enough_building_points` for
  the degenerate-block fallback.

One pre-existing test needed **recalibration, not the new formula**:
`test_loop_beats_spur_at_equal_length` demonstrates a topological property (loop redundancy beats
single-arm reach at equal road length) that empirically flips once `g_walk` drops from 1.0 to 0.1
at fixed `g_road=20` (a 200:1 ratio makes REACH dominate REDUNDANCY on that single-arm grid) — this
fixture has no `building_points` (r0=0 fallback → formula reduces exactly to the pre-change
`g_walk/dist` baseline), so it is **not** a symptom of the r0-corridor change, purely the g_walk
level. Fixed by pinning that one test to an explicit `PermeabilityParams(g_walk=1.0)` (its original
calibration level) with an explanatory comment, rather than changing the geometry — verified by
direct sweep (`g_walk=1.0`→loop wins by 0.017, `g_walk∈{0.5,0.2,0.1,0.05}`→spur wins) that this is
a real, expected regime shift, not a bug.

## Verification commands (all run inside the worktree)

- **`pixi run pytest -q`**: `446 passed, 4704 warnings in 77.00s` (0 failures, whole suite).
- **`pixi run ruff check .`**: `Found 12 errors` — all in files this task did NOT touch
  (`scripts/gen_example_readme.py`, `scripts/gen_multiblock_example.py`,
  `scripts/gen_site_pages.py`, `src/reblock/budget.py` line 79, `src/reblock/methods/
  arterial_lazy.py`, `tests/methods/test_arterial.py`) — matches the stated ~12 pre-existing
  baseline exactly; **0 new**.
- **`pixi run mypy src scripts`**: `Found 7 errors in 2 files` — all in
  `scripts/fetch_kblock_fixtures.py` and `scripts/gen_site_pages.py`, neither touched by this task
  — matches the stated ~7 pre-existing baseline exactly; **0 new**. (Extra check, not required by
  the task but run for confidence: `pixi run mypy --strict src tests scripts/crossblock_probe.py
  scripts/calibrate_permeability.py` → `Success: no issues found in 96 source files`, i.e. the new
  test file is also strict-clean.)

## Files changed

- `src/reblock/permeability.py` — the metric change (module docstring, `PermeabilityParams`
  (`g_walk: 0.1`, new `r0_frac: 0.55`), `_adaptive_r0`, `_footpath_conductance`, `egress_power`'s
  edge assembly + monotonicity clamp, `r0` threading through `permeability`/`permeability_curve`).
- `conf/permeability.yaml` — `g_walk: 0.1`, `r0_frac: 0.55` (thresholds `matched_displacement`/
  `matched_permeability` left untouched, as instructed).
- `src/reblock/compare.py`, `scripts/compare_budgets.py`, `scripts/calibrate_permeability.py` —
  their `PermeabilityParams(...)` construction from `conf/permeability.yaml` now also reads
  `r0_frac` (previously would have silently used the dataclass default regardless of yaml).
- `tests/test_permeability.py` — new imports; recalibrated `test_loop_beats_spur_at_equal_length`
  (explicit `g_walk=1.0`, documented why); 6 new tests (r0-corridor form, adaptive r0 x2, clamp
  necessity/effectiveness).
- `tests/test_compare_budgets.py` — `test_load_permeability_config_reads_the_committed_yaml`
  updated for the new `g_walk`/`r0_frac` values.

## Not done (out of scope per the task)

- `matched_displacement`/`matched_permeability` (P*) recalibration — untouched, per instructions
  (separate follow-up step).
- `examples/**` regeneration — untouched.
- Not pushed.
