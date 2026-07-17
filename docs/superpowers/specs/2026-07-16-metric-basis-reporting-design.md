# Metric-basis reporting refactor — design

**Goal:** Replace the five entangled benefit lenses (access, efficiency, directness, resistance,
displacement) with the validated **orthogonal metric basis**: two benefit axes — *External
connectivity* and *Internal connectivity* — plus *Displacement* as the cost.

**Status:** design approved 2026-07-16; execute via subagent-driven-development.

## Background — why this basis

A spectral investigation this session (PCA / eigendecomposition of the metric correlation matrix over
a diverse corpus of road networks) established that road-structure quality is **2-dimensional** once
road *quantity* is controlled (the quantity axis is the displacement/cost dimension). The two quality
axes are **stable across two independent blocks** (40972 @ 263 parcels, 39229 @ 365 parcels; each gives
2 eigenvalues > 1, 90–97% variance, same varimax-rotated loadings). See memory
`road-structure-metric-basis`. This refactor consolidates that finding into the reporting.

The five lenses were entangled: efficiency ↔ directness were ρ=0.99 duplicates; efficiency straddled
both axes; resistance rode with access on the external axis. Collapsing to two orthogonal axes + the
cost is both more elegant and more legible.

## The metric set

Road-structure **quality = two orthogonal benefit axes**; **displacement = the separate cost**.

### External connectivity — representative metric: `access`
Reach/drainage of parcels to the *external* street network. Loads 0.98 on the external axis. This is
the existing `access_benefit` factory, **unchanged** — only relabeled "External connectivity" in
reporting. (The existing egress-`resistance` metric loaded on this same axis and is now subsumed by
access; it is deleted — see Migration.)

### Internal connectivity — representative metric: **cycle density** (NEW)
Richness/redundancy of the *internal* network — the resilient-egress value (alternative routes). Loads
0.96. Defined as the **circuit rank per parcel**:

```
cycle_density(roads) = (E − N + C) / P
```

where `E`, `N`, `C` are the edge, node, and connected-component counts of the **noded** road∪street
planar graph, and `P = len(block.parcels)` is the parcel (dwelling) count. `E − N + C` is the number of
independent cycles (circuit rank); a tree → 0, each extra loop → +1. Empty roads / `P < 1` → 0.0.

**Why per-parcel, not per-graph-node** (the spectral study used per-graph-node `/N`; per-parcel is
theoretically cleaner and is the chosen form): circuit rank is a **topological invariant** — subdividing
an edge does `E→E+1, N→N+1, C` unchanged, so the loop count is insensitive to how finely roads are
discretized. Dividing by the fixed, exogenous parcel count `P` preserves that invariance and yields an
interpretable "independent loops per dwelling"; dividing by the graph node count `N` (endogenous to the
road network, and sensitive to vertex/crossing density) spuriously varies with discretization. Because
`P` is constant per block, per-parcel `∝ circuit rank`, highly correlated with the validated `/N` form —
so it will land on the same internal axis. **Task 1 re-confirms** this (recompute the internal-axis
loading with per-parcel cycle density; expect ρ≈unchanged) before the reporting is wired to it.

**Noded graph builder (NEW, required):** the raw road+street edge set is disconnected at the vertex
level (roads reach streets geometrically but do not share exact graph nodes — a 2D-plane diagnostic
confirmed λ₂=0 / many components on the un-noded graph). So cycle density must be computed on a
*planarized* graph:
1. `merged = unary_union(list(roads.geometry) + list(block.streets.geometry))` — nodes every
   crossing/touch into shared vertices.
2. Explode `merged` to segments; `_rnd`-snap endpoints (2-dp, matching `budget._rnd`); build an
   undirected graph (dedupe repeated edges).
3. `E` = edge count, `N` = node count, `C` = `connected_components`; return `(E − N + C) / n_parcels`.

Empty roads → 0.0. This builder lives in `budget.py` next to the other graph helpers.

**Why not Fiedler (algebraic connectivity):** it loaded 0.97 in the raw PCA but is a **dud** — reblocking
road networks are near-trees, so λ₂ ≈ 0 (measured 0.0007–0.014, no discrimination), and the un-noded
graph is disconnected (λ₂=0). Its "0.97" was near-zero noise. Cycle density is the robust internal
representative (measured 0.006 → 0.192, a 30× tree→mesh spread). **Do not use Fiedler.**

### Displacement — the cost
`displacement_curve` / disk-displacement, **unchanged** (r=NN/2, Σ homes grazed). It is the quantity/
cost axis and stays reported separately, exactly as today.

## Frontier reporting

**x-axis stays cumulative added road length (metres)** — the intervention knob — reusing `_sweep`,
`auc`, `truncate_to_length`, `matched_budget`, and the matched-budget renders unchanged.

Per (region, method) the compare sweep emits **three curves** (down from five):
- `external_connectivity` — `cost_benefit_curve(block, roads, benefit_fn=access_benefit)`
- `internal_connectivity` — `cost_benefit_curve(block, roads, benefit_fn=cycle_benefit)` where
  `cycle_benefit` is a new factory returning `f(roads) = cycle_density(roads)` (plugs into `_sweep`
  like the other factories; NOT normalized to [0,1] — AUC is the mean cycle density over the budget,
  a valid scalar).
- `displacement` — `displacement_curve(...)`, unchanged.

`emit.py`'s `compare_report` emits the corresponding per-metric plots + frontier CSVs for these three
metric names (the plumbing is metric-name-generic; only the metric set shrinks).

### 2D connectivity-plane summary figure (NEW)
Per block/region, a single figure plotting each method's **trajectory through (External connectivity,
Internal connectivity) space** as road grows (drainage-ordered prefixes; marker size ∝ cumulative road
length; one colour per method; endpoint labelled). This is a communication figure, not a metric — it
has no scalar AUC. It is emitted alongside the per-metric curves for the flagship examples.

## Migration (delete, do not accommodate)

**Delete:**
- `budget.py`: `resistance_benefit`, `_resistance_core`, `resistance_frozen`,
  `_BlockScoringContext._ground_indices` (the whole egress-resistance engine — reporting-only, now
  subsumed by access); `efficiency_directness_curves`, `efficiency_benefit`, `directness_benefit`, and
  `_efficiency_factory` **iff** it falls unused after those go (verify: arterial uses
  `_BlockScoringContext.score`, not `_efficiency_factory`).
- `tests/test_resistance.py` — entire file (tests deleted machinery).
- `compare.py`: the lines emitting the `efficiency`/`directness`/`resistance` curves; the raw-tuple
  metric rows for them; imports of the deleted symbols.
- Old example artifacts for the deleted metrics (`compare_efficiency.png`, `compare_directness.png`,
  `compare_resistance.png`, their frontier CSVs) in both flagship examples.

**Keep (load-bearing for methods, NOT reporting):**
- `network_efficiency`, `_BlockScoringContext.score`/`.score_frozen`, `_sampled_efficiency_core`,
  `_build_csr`, `_line_entries`, the whole entry-projection scoring core — **arterial's
  `objective=directness|efficiency` still depends on them.** Directness survives as an arterial
  *objective*, just not as a reporting *curve*.
- `access_benefit`, `cost_benefit_curve`, `_sweep`, `auc`, `truncate_to_length`, `matched_budget`,
  `road_drainage`, `building_radii`, `displacement`, `displacement_curve`, `access_burden`.

**Add:**
- `cycle_density(roads)` + the noded-graph builder + `cycle_benefit` factory in `budget.py`.
- The 2D connectivity-plane figure in the render/emit path.

## Testing
- **Task 1 first:** re-confirm per-parcel cycle density loads on the internal axis (recompute the
  spectral-study loading with `/P` instead of `/N`; expect ≈unchanged) before wiring the reporting.
- `cycle_density` unit tests on hand-built geometries: a single tree → 0; one closed loop → circuit
  rank 1 → `1/P` (parcel count); two disjoint loops → circuit rank 2; verify the noded builder nodes an
  X-crossing into 4 edges sharing the centre vertex (so a road crossing a street counts its cycle);
  subdividing a road's edge leaves cycle_density unchanged (topological-invariance regression test).
- Noded-graph builder: empty roads → empty graph → 0.0; a road touching a street mid-segment shares a
  node (component count correct).
- `cycle_benefit` monotone-enough across a drainage-ordered prefix sweep (values defined at every
  prefix; no crash on empty prefix).
- `compare.py`/`emit.py`: the sweep emits exactly the three metric names; no reference to the deleted
  metrics remains (grep-clean).
- Regenerate **both** flagship examples once, at the end (compute-heavy): the READMEs, the three
  per-metric curves, the matched-budget renders, and the new 2D plane. Reproduce commands drop the
  deleted metrics.
- `pixi run check` (lint + typecheck + test) stays green.

## Non-goals / caveats
- **Not** re-deriving or re-validating the basis at larger scale — the 2-block validation is accepted
  (a broader pass with topology + more blocks was offered and deferred). If a future validation shifts
  the axes, that is a separate change.
- **Not** combining the two axes into a single score — the whole point is that they are orthogonal;
  they are reported as two separate benefit curves.
- **Not** changing any reblocker method or the displacement metric itself.

## Execution
Subagent-driven-development on branch `metric-basis-refactor` (off `fix-mypy-typecheck` for a working
typecheck; rebased onto `main` after PR #3 lands). Example regeneration is the final task.
