# Permeability — a single flow-based reblocking metric (design)

**Status: approved in brainstorming 2026-07-22.** SUPERSEDES `2026-07-22-dual-target-connectivity-outcome-design.md` and the internal/external two-axis basis. Retires `commute_ratio` (internal connectivity), the freeze, and `access_benefit` as a *reported* metric.

## Goal

Replace external + internal connectivity with ONE monotone, contention-aware, tractable metric — **permeability** — computed from a collective-egress electrical flow. Report only **permeability** (benefit) and **displacement fraction** (cost). This dissolves the freeze, the on/off membership, the over-provisioning, and the non-monotonicity, all at once, and collapses the two-axis story to a single frontier.

## The metric: permeability

**Physical model.** Model collective egress as an electrical flow. Every parcel injects 1 unit of "escape current"; the existing street is ground (potential 0). Roads are low-resistance shortcuts through an always-present high-resistance walkable mesh. The **total dissipated power** of that flow measures how hard it is for everyone to get out — accounting for distance, contention (many parcels sharing a road), and redundancy (loops spread the current), in one scalar.

**Graph.** Nodes = parcels (one per parcel, at its polygon centroid). Ground = the street (eliminated node at potential 0). Edges (conductances):
- **Footpath mesh (always present):** for each Voronoi-adjacent parcel pair (i,j) — adjacency from `reblock.derive.access.parcel_adjacency` — conductance `g_walk / dist(centroid_i, centroid_j)`.
- **Road upgrade:** an adjacency edge (i,j) whose centroid segment is within `corridor_m` of a road (`roads.buffer(corridor_m).intersects(LineString([cᵢ,cⱼ]))`) gets conductance `g_road / dist` instead. Roads make that traversal fast.
- **Ground edges:** a parcel within `STREET_TOL` of the street gets a ground edge of conductance `g_street` (added to its diagonal; ground is eliminated).

**Formulation.** Assemble the sparse grounded Laplacian `L` (n×n). Inject `b = 1ₙ`. Solve `L v = b` (sparse, `scipy.sparse.linalg`). Raw egress power `P = bᵀv = Σᵢ vᵢ`. The reported metric normalizes against the no-roads (footpath-only) baseline:

> **permeability = 1 − P(roads) / P(no_roads)**

**Properties (validated on a block + the 11k region, 2026-07-22 prototype):**
- **Monotone** — roads only add conductance, so P only falls (Rayleigh); permeability rises with 0 exceptions. No freeze, no non-monotonicity.
- **Bounded percentage** — `∈ [0, 1)`: 0 = walkable baseline, → 1 only in the infinite-pavement limit; real full networks land ~0.72–0.83. Same character as the retired `access_benefit` (also caps below 1). Pairs naturally with displacement-fraction on the same `[0,1)` axis type.
- **All-parcels, smooth** — b is all-ones; no membership, no on/off, no skipped parcels.
- **Contention + redundancy + distance in one number** — power is `Σ Iₑ²Rₑ`: a road funnelling many parcels' current is penalised quadratically; a loop spreads the current and lowers P; resistance ∝ length. A drainage tree funnels all egress through its root → high P → low permeability; a mesh/loop network spreads it → low P → high permeability. Prototype: arterial (loops) 0.825 vs clearance (tree) 0.723 at full networks, arterial leading across ~90% of the shared-budget range.
- **Tractable** — one sparse solve ≈ **4 s** on 11k parcels (adjacency ~0.7 s); a 20-point sweep ≈ 80 s. Region-scalable.

**Parameters.** `g_walk`, `g_road`, `g_street`, `corridor_m`. Tuning (validated): **`g_road/g_walk ≈ 20`** is the sweet spot — strong overall effect (~82% power reduction at full networks) without one or two trunk roads saturating the score (low ratios leave footpaths too competitive; high ratios let a single road claim most of the benefit). `g_street` on the order of `g_road`. These live in a `conf/permeability.yaml`.

**Baseline framing.** As defined, permeability is *gain over the walkable baseline* (0 = footpaths only). This is the right "improvement over status quo" reading for comparing reblockers. (If an absolute network-permeability is later wanted, renormalize; not needed now.)

## Cost: displacement fraction

Unchanged from Task 1 (already shipped on this branch): `displacement_curve` yields `Σcᵢ / n_buildings` per prefix, displayed as a percentage. This is the single cost axis.

## Reporting / output

Only permeability and displacement are shown. Per example:

- **One frontier curve** — permeability (y) vs displacement (x), one line per method, overlaid. **No title**; x-axis label `displacement`, y-axis label `permeability`. This replaces the old three curves. Pareto-dominance is read straight off it.
- **Two lenses**, each a set of after-images + a table (both thresholds calibrated — see below):
  - **Lens A — matched displacement** (universal displacement % `D`): truncate each method to the first prefix with displacement ≥ `D`; table + after-image compare **permeability** at equal home-cost. Every method reaches it (displacement is monotone).
  - **Lens B — matched permeability** (universal permeability `P*`): truncate each method to the first prefix with permeability ≥ `P*`; table + after-image compare **displacement** to reach the standard. A method whose full network never reaches `P*` reads as **unreached** (informative — e.g. a tree that can't hit the bar).
  With permeability monotone + always-defined, there is **no "kill" machinery** — a weak method simply reads low on the curve / unreached on Lens B.
- **Before-image (once per example)** — the status-quo heatmap with no roads. **After-image (per method per lens).** **Generate BOTH heatmap colorings** for every before/after render (both fields are computed anyway): (1) **access-depth** (parcels-from-a-street, the intuitive "deep interior becomes reachable" story) and (2) **per-parcel egress potential `vᵢ`** from the permeability flow (dark = hard escape, light = easy — the metric-aligned view). Suffix the files to distinguish them (e.g. `before_depth.jpg`/`before_perm.jpg`, `after_<method>_<lens>_depth.jpg`/`..._perm.jpg`). Roads overlaid in blue; displaced homes as red disks with opacity = graze fraction (shipped style). The README shows both (or is trivially switchable).
- **Per-method reblock GIF** — unchanged (current low-res, current dimensions).
- **Screen map** — drop the heatmap **colorbar and title**, and remove the **region boundary-following outline** (the per-member black outline that occludes the metric colors); **keep only the thick black bounding box**. Selection logic (the `depth` proxy `√(nA)/P`) is unchanged.
- **No `block_id`/parcel name on any plot** (it's in the README).
- **Poster-grade static images** — every static PNG/JPG at **300 dpi** and large figure dimensions (target ≳ 3000 px on the long edge, crisp on a 3′×4′ poster). GIFs excepted (stay low-res / current dims).

## Calibration (clean probe + checkpoint)

The two thresholds — `D` (Lens A) and `P*` (Lens B) — are **not yet calibrated** (30% was a placeholder; the old `D_max=0.45` was for the invalidated internal-connectivity probe). Build a **clean** permeability calibration:
- Per method (arterial_repulsion, clearance_looped, euclidean_grid; osm_footpaths as reference), on each example region (6 multiblock + the method-comparison block), build the **permeability-vs-displacement** frontier — **no over-provisioning** (natural convergence), **per-region isolation** (fresh subprocess per region, to avoid the in-memory cross-region bleed that corrupted the earlier probe).
- From the frontiers, propose: a **displacement %** `D` in the dynamic-range sweet spot and a *humane* home budget (likely < 30%); a **permeability level** `P*` the serious methods clear with separation.
- **CHECKPOINT:** present the frontiers + proposed `(D, P*)` for human sign-off before baking into `conf/permeability.yaml`.

## Affected components / interfaces

**New:**
- `src/reblock/permeability.py` (or in `budget.py`): `egress_power(block, roads, *, params) -> (P, v)` (sparse build + solve; returns raw power + per-parcel potentials for the heatmap); `permeability(block, roads) -> float` (`1 − P/P₀`); `permeability_curve(block, roads) -> Curve` (per-prefix, monotone; x = road length like the others, re-plotted vs displacement).
- Lens truncations in `budget.py`: `prefix_to_displacement(block, roads, d)` (Lens A), `prefix_to_permeability(block, roads, p_star)` (Lens B) — both scan the shared `_sweep` grid; the latter is a clean monotone first-crossing.
- `conf/permeability.yaml` (metric params + `g_road/g_walk≈20`), `conf/reporting` thresholds `(D, P*)`.
- `scripts/calibrate_permeability.py` (the isolated-per-region probe).
- The reporting driver: `run_two_lens` → a permeability two-lens driver (frontier curve + two lens image/table sets + before + per-method GIF).

**Retired (migrate, no back-compat):** `commute_ratio` + `commute_ratio_benefit` + `_commute_membership` + the freeze; `access_benefit`/`access_burden`/`parcel_access_layers` **as a reported metric and curve** (KEEP `parcel_access_layers`/depth for the *screen* selection; the after/before heatmap moves to the permeability potential); `prefix_to_external_connectivity`; the dual-target `prefix_to_joint_target`/`JointTargetOutcome` and its probe (superseded by the permeability probe + the two lenses); the external/internal curves + their frontier CSVs.

**Kept:** displacement (fraction) + `displacement_curve`; the GIF pipeline; the screen/region maps (minus colorbar/title); `render_before`/`render_after` (re-pointed to the permeability heatmap + poster sizing); the example-generation orchestration (`gen_multiblock_example`, `gen_method_comparison`, `gen_example_readme`, `regenerate_examples.sh`).

## Testing

- `egress_power`/`permeability`: hand-computable small circuits (a single loop vs a spur → loop lower P; a known 2-node ladder); monotonicity on a constructed prefix sweep (0 decreasing steps); baseline `permeability(no roads)=0`; bounded `∈ [0,1)`.
- `prefix_to_displacement` / `prefix_to_permeability`: first-crossing on constructed curves; unreached case for `P*`.
- Two-lens driver smoke test (tables + before + per-method after per lens + one frontier curve).
- Calibration probe: unit-test its frontier-extraction helper; numeric output is human-reviewed.

## Sequencing (for the plan)

1. `permeability` metric (sparse egress flow + normalization) + tests. Port the validated prototype from scratchpad; make it region-scalable (sparse).
2. `permeability_curve` + the two lens truncations + tests.
3. Calibration probe (isolated per region, no over-provisioning); **run it, review `(D, P*)`, bake into config** (checkpoint).
4. Reporting driver: frontier curve (permeability vs displacement, cleaned), two lens image/table sets, before-image, permeability-potential heatmap, poster sizing, screen-map/plot cleanups. Retire the external/internal/commute_ratio/freeze surface.
5. README generators → permeability story (frontier + two lenses + before/after).
6. Regenerate all examples; sanity-check the survivor/permeability pattern vs the probe.

## Open items / risks

- **Sparse solver at scale:** one `spsolve` ≈ 4 s at 11k parcels is fine for a 20-point sweep; if the full example set's total time is heavy, cache the adjacency per region and reuse the symbolic factorization across prefixes.
- **Heatmap coloring:** RESOLVED — generate BOTH access-depth and permeability-potential colorings per render (both fields already computed).
- **`g_road/g_walk` and `g_street`** are the only real knobs; 20 validated, but re-confirm on the calibration regions.
- **Poster sizing** may need per-figure tuning (fonts/线 widths/marker sizes scale with figsize); verify a sample renders crisp before the full regeneration.
- Retiring `access_benefit` as a metric while keeping `parcel_access_layers` for the *screen* must not break the screen — the screen uses the `depth` proxy, not `access_benefit`, so this is clean, but confirm no reporting code still imports the retired benefit factories.
