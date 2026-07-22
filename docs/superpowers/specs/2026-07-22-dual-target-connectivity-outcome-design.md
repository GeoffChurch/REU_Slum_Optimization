# Dual-Target Connectivity Outcome — Design

**Status:** design approved (brainstorming 2026-07-22); pending spec review → implementation plan.

## Goal

Replace the current external-only example calibration (Lens A: "roads until external connectivity
≥ 0.70"; Lens B: matched road budget) with a single **joint-target outcome**: run each synthetic
(non-osm) method until it *simultaneously* attains a minimum internal **and** external connectivity,
subject to a **displacement ceiling** that kills methods which cannot reach the joint target within a
home-displacement budget. One universal, empirically-calibrated `(I_min, E_min, D_max)` governs every
example (both the multiblock flagships and the single-block method-comparison), so outcomes are
comparable across examples.

## Motivation

The examples currently target external connectivity only; internal connectivity (`commute_ratio`) is
measured and plotted but never a stopping criterion. The metric basis treats external + internal as
two orthogonal axes plus displacement as the cost axis — so the fair "outcome" for a method is the
road set at which it first delivers *both* connectivity floors without displacing more homes than a
livability budget allows. This also gives a principled, self-selecting way to distinguish methods
that build genuine redundancy from those that cannot (drainage trees) or that only reach connectivity
by bulldozing.

## Displacement as a fraction of homes (global)

`displacement()` is intrinsically an expected-homes-displaced *count* (`Σ cᵢ`, cᵢ = graze probability
of building i). Everywhere it is **reported, plotted, or thresholded**, it becomes the **fraction of
homes** `Σcᵢ / n_buildings` (n_buildings = `len(block.building_points)` — buildings, not parcels; the
existing `pct_displaced` already computes exactly this). This makes displacement region-scale-invariant
and lets a single universal `D_max` fraction apply to every region without rescaling.

- The raw `displacement()` stays as the `Σcᵢ` primitive (needed to form the fraction; the absolute
  "≈75 homes" remains usable in prose).
- `displacement_curve` yields the **fraction** per prefix; the connectivity curves' displacement x-axis,
  `displacement_table.csv` / `displacement_vs_length.csv`, all axis labels ("fraction of homes
  displaced"), and `D_max` are the fraction. This is a **global** change — the existing examples'
  displacement axes/tables switch to the fraction too (folded into the regeneration).

## The stopping rule

Per synthetic method, on a region (or the method-comparison block):

1. Reblock to a road set long enough to either reach the joint target or exhaust `D_max` — i.e. run
   **past natural convergence** (over-provisioning, below).
2. Order roads drainage-descending and build the three index-aligned curves over the shared
   `_sweep` sampling (they already share sample points — the invariant `test_curves_share_cost_samples`
   locks this):
   - **external** `e[i]` = `access_benefit` (monotone non-decreasing)
   - **internal** `int[i]` = frozen `commute_ratio` (membership frozen to the method's terminal
     over-provisioned roads — the near-monotone curve already shipped)
   - **displacement** `d[i]` = fraction of homes displaced, `Σcᵢ / n_buildings` (monotone non-decreasing)
3. **Outcome** = the first sample index `i*` with `e[i*] ≥ E_min` **and** `int[i*] ≥ I_min` **and**
   `d[i*] ≤ D_max`. Because the three curves are index-aligned, this is a cheap post-hoc scan of
   curves already computed for the report — no extra reblock or resistance solve. Touch-and-go is
   accepted: stop at the first qualifying index even if internal later dips (owner decision).
4. **Killed** if no index qualifies within `d[i] ≤ D_max`. Two disjoint causes, both reported as
   "failed the joint target within the home budget":
   - *never reaches `I_min`* — a drainage tree (`clearance`, `topology`) has no loops by construction,
     so internal ≈ 0 at every prefix regardless of paving; killed by the connectivity requirement.
   - *reaches the floors only past `D_max`* — a bulldozer (potentially `clearance_looped`, which on the
     method-comparison block hit internal 0.316 only at 77% homes displaced); killed by the
     displacement ceiling. **Accepted: `clearance_looped` may be killed on some blocks** — an honest
     "over-builds at this scale" result; "every internal-capable method survives" is NOT a requirement.

The outcome index `i*` lies on the displayed curves, so the report marks it directly.

## Over-provisioning

The report curves currently stop at each method's natural convergence; the scan needs them extended
to ~`D_max` displacement so a late-blooming method gets its shot. Generalize the existing `extend`
mechanism in `run_two_lens` (over-provisioned re-runs, truncated in drainage order): each synthetic
method is run to a road set whose terminal displacement ≥ `D_max` (or its structural maximum). The
over-provisioning knob is method-specific (arterial `max_roads`, clearance-family `depth_target`/
`max_roads`, euclidean finer `spacing`) and lives in a per-method config block reused by both the
calibration probe and the example generators. A method that structurally cannot pave to `D_max`
(e.g. euclidean at its finest sensible spacing) is scanned over whatever it produces; if that never
reaches the floors, it is killed.

## Calibration (universal, empirically probed)

A one-time probe derives `(I_min, E_min, D_max)` as universal constants (baked into config), reviewed
before adoption:

- **`E_min` = 0.70** — retained. Already every method clears it; kept for continuity and comparability
  with prior examples. (The probe reports each method's external headroom; revisit only if the data
  argues for it.)
- **`D_max` = a home-FRACTION** (e.g. 0.45 of buildings). Displacement is a fraction everywhere (see
  "Displacement as a fraction of homes"), so `D_max` compares directly against the fractional
  displacement curve — no per-region rescaling. Chosen as a livability cap so the efficient methods
  survive with margin; the probe reports each method's displacement-at-outcome so the fraction can be
  set to sit comfortably above the reference method (arterial) and below the bulldozers.
- **`I_min`** — derived by the probe: the largest internal floor that the reference internal-capable
  method (`greedy_arterial_repulsion`) clears on **every** example region within `(E_min, D_max)`, with
  a small safety margin so it is reliably clearable. The probe reports, at this `I_min`, which methods
  survive on which regions.

Probe = `scripts/calibrate_joint_target.py`: over-provision each internal-capable method on all example
regions (6 multiblock + the method-comparison block), build the three aligned curves, and emit a table
of per-(method, region) achievable joint frontiers + the derived `(I_min, E_min, D_max)`. Its output is
reviewed; the chosen constants land in `conf/` (e.g. `conf/joint_target.yaml`).

## Output (replace both lenses)

Per method, a single **outcome view** replaces the two lens truncations:

- **After-image** at the outcome prefix (`truncate_to_length` to `external.cost[i*]`). Killed methods
  render at the `D_max`-capped prefix, titled with the failure reason (`internal X < I_min` or
  `displaced Y% > budget`).
- **Curves** keep the full over-provisioned frontier, with the outcome index marked (a star) on each
  of the three curves; killed methods show the frontier with no outcome marker (or a marker at the
  `D_max` cap).
- **GIF** unchanged (full over-provisioned drainage sweep).
- **Outcome table** replaces `lens_a_external.csv` + `lens_b_matched.csv`: one row per method —
  `method, reached (bool), reason, road_m, external, internal, displacement` (displacement = home
  fraction, so the old separate `pct_displaced` column is subsumed).

**`osm_footpaths`** is the fixed as-built network, not "run to" anything: shown at its full network with
its `(external, internal, displacement)` and a flag for whether it happens to clear the floors — a
real-world reference point, never truncated to a target.

## Affected components / interfaces

- `src/reblock/budget.py` — new `prefix_to_joint_target(external: Curve, internal: Curve,
  displacement: Curve, roads: GeoDataFrame, block: Block, *, i_min, e_min, d_max) ->
  JointTargetOutcome` where `JointTargetOutcome = (prefix: GeoDataFrame, reached: bool, reason: str,
  external: float, internal: float, displacement: float, sample_index: int)` (`d_max`, `displacement`
  are the home fraction). Pure post-hoc over the three aligned curves; retire
  `prefix_to_external_connectivity` (migrate its one call site — no back-compat shim, per owner
  directive). `displacement_curve` yields the **fraction** (`Σcᵢ / n_buildings`) instead of raw `Σcᵢ`;
  migrate its call sites (`compare.py`, `compare_budgets.py`, `gen_method_comparison.py`).
- `scripts/compare_budgets.py` — `run_two_lens` → `run_dual_target`: over-provision each synthetic
  method, build the three curves once, compute the outcome, render the single after-image (+ killed
  handling), write the outcome table, mark the outcome on the curves. `two_lens_rows`/`LensARow`/
  `LensBRow` retired.
- `src/reblock/emit.py` — `compare_report` gains the outcome-marker overlay and switches all
  displacement axis labels/values to the home fraction ("fraction of homes displaced"); the
  depth-vs-road report is unaffected.
- `scripts/gen_multiblock_example.py`, `scripts/gen_method_comparison.py` — call the dual-target path;
  pass the universal `(I_min, E_min, D_max_frac)` from config.
- `scripts/gen_example_readme.py` + the method-comparison hand-README — rewrite the two-lens sections
  to the joint-target outcome story (surviving methods at their outcome; killed methods + reasons).
- `conf/joint_target.yaml` — the universal constants + per-method over-provisioning knobs.
- `scripts/calibrate_joint_target.py` — the calibration probe (run first; its output sets the config).

## Testing

- `prefix_to_joint_target`: constructed aligned curves — first-qualifying-index selection; touch-and-go
  (internal crosses then dips → still stops at first cross); killed-by-`I_min` (internal never clears);
  killed-by-`D_max` (floors met only past the displacement cap); empty/degenerate guards.
- The index-alignment invariant test already exists (`test_curves_share_cost_samples`) — the outcome
  scan depends on it; reference it.
- `run_dual_target` smoke test (like the existing `test_run_two_lens_writes_tables_and_renders`):
  produces the outcome table + one after-image per method + curves, on a small fixture.
- Calibration probe is a script, not unit-tested for values (its output is reviewed), but its
  frontier-extraction helper (max internal within `(E_min, D_max)` per method) gets a unit test.

## Sequencing (for the plan)

1. `prefix_to_joint_target` + tests (pure, no rendering).
2. Over-provisioning config + the calibration probe; **run it, review `(I_min, E_min, D_max)`**, bake
   constants into `conf/joint_target.yaml`.
3. `run_dual_target` refactor (retire the two-lens path) + outcome table + curve outcome marker + tests.
4. README generators (multiblock auto + method-comparison hand) → joint-target story.
5. Regenerate all examples; verify the surviving/killed pattern matches the probe.

## Open items / risks

- **Outcome resolution**: using the `_sweep` sample grid (20 points) makes the outcome coincide with a
  plotted curve point (clean), but coarse. If a finer outcome is wanted, refine near `i*` with a local
  bisection on road length — deferred unless the coarse outcome reads poorly.
- **Killed-method after-image**: showing a killed method at the `D_max` cap is informative but the image
  no longer represents a "chosen" reblock; the title must make the failure explicit so it is not
  mistaken for a recommended outcome.
- **`D_max` as a fraction** assumes homes-displaced scales with region size; validated by the probe
  across the 6 regions of differing size.
- **method-comparison** shifts from a "full frontier + terminal points" story to a "who reaches a
  livable joint outcome" story; its curves still carry the frontier, so the trees-at-zero-internal
  contrast is preserved.
