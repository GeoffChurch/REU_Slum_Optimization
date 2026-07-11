# Grounded Resistance Eval — Design

**Status:** draft for review · **Date:** 2026-07-11

**Goal:** Add a `resistance` cost-benefit lens — **grounded effective resistance to egress** — as a 4th metric in the compare, alongside `access` / `efficiency` / `directness`. This is the north-star "Piece 2" (the egress axis the shortest-path lenses structurally cannot express). Purely **additive**: no existing metric or value changes.

**Architecture:** A `resistance_benefit` factory in `budget.py` mirroring `_efficiency_factory` (freeze entries against the full road set for monotonicity), a `_BlockScoringContext.resistance_frozen` method that reuses the existing per-prefix CSR build but solves a grounded Laplacian instead of running Dijkstra, a `_resistance_core` numeric function, and one line in `compare.py`. Reuses `_derive_entries` / `_build_csr` / `rep_xy`; adds a sparse Laplacian solve (`scipy.sparse.linalg`, `scipy.sparse.csgraph`).

**Tech Stack:** Python, numpy, scipy.sparse (`linalg.splu`/`factorized`, `csgraph.connected_components`), the post-Tier-B `_BlockScoringContext` machinery, pixi, pytest, mypy --strict.

**Design source:** validated in `docs/superpowers/notes/2026-07-11-spectral-metric-investigation.md`; working solve in `scratchpad/spectral_proto.py` (`_laplacian`, `_reachable_from_ground`, `grounded_resistances`) — but that prototype used the now-DELETED `_road_street_graph`/`_split_graph`; the implementation must build on the current CSR machinery.

## Global Constraints

- **Purely additive.** No change to any existing metric value: `access`/`efficiency`/`directness` curves + AUCs and every existing test stay identical. `network_efficiency`, `_efficiency_factory`, the arterial greedy — untouched.
- **The metric is the INTENSIVE per-parcel mean `R_i`.** NOT the extensive single-solve `wᵀL_G⁻¹w` and NOT the raw Kirchhoff index — the investigation (§3 caveat) proved those are extensive (grow with node/parcel count) and rank road sets WRONGLY (e.g. call arterial worse purely because it adds nodes). Use `mean(R_i)` (or normalized `Kf/binom(N,2)` if a global number is ever needed — not here).
- **Monotone.** Entries frozen against the full road set (like `_efficiency_factory`); adding road edges only lowers resistances (Rayleigh monotonicity), so `benefit` is non-decreasing across cost-benefit prefixes.
- **Determinism**, `mypy --strict`, ruff clean, `pixi run check` green.

## The metric (validated)

For a road set (streets + roads), on the same graph + line-proximity parcel entries the efficiency metric uses:

- **Edge conductance** `c_e = 1/length_e` (so on a single wire, resistance distance = length in metres; resistance = shortest-path on a tree, strictly lower with loops). Weighted **Laplacian** `L = D − C`, `C` the conductance adjacency, `D = diag(row sums)`.
- **Ground set** `S` = graph nodes within `tol` of `block.streets` (the existing egress network, potential 0). **Grounded reduced Laplacian** `L_G = L` with the `S` rows/cols removed (SPD on each component that reaches ground).
- **Per-parcel egress resistance**
  ```
  R_i = (L_G⁻¹)_{entry_i, entry_i} + leg_i          # leg_i = euclid(rep_i, entry_i), the walk to the road
  ```
  - entry ON a ground node → drive term 0 → `R_i = leg_i`.
  - no entry within `tol`, or entry not in a component that reaches ground → `R_i = cap` = the block bbox diagonal (`hypot(bounds[2:]-bounds[:2])`), analogous to `access_burden`'s unreached cap.
- **Block score** = `mean_i R_i` (lower = better). **Benefit** = `(R(∅) − R(prefix)) / R(∅)`, where `R(∅)` is the no-roads (streets-only) score under the frozen entries — 0 at the empty prefix, rising as roads are added.
- **Solve:** one SuperLU factorization of `L_G` per prefix (`scipy.sparse.linalg.factorized`/`splu`), then a back-substitution per DISTINCT entry node for its diagonal `(L_G⁻¹)_{kk}`. At our sizes (≤~1400 nodes) this is ≤~30 ms — 3–45× cheaper than the sampled Dijkstra metric.

## Design

### 1. `_resistance_core(csr, node_index, entry, rep_xy, ground_idx, cap) -> float`
Pure numeric core (adapt `spectral_proto.grounded_resistances`, nx→CSR):
- Conductance from the CSR: `C` = same sparsity as `csr` with `data → 1/data` (csr holds edge lengths; the CSR is already symmetric + deduped). `deg = C.sum(axis=1)`; `L = diags(deg) − C` (csc).
- **Reachable-from-ground:** `scipy.sparse.csgraph.connected_components(csr)`; a component is grounded iff it contains a `ground_idx` node; `reach` = union of grounded components.
- `free` = node indices NOT in `ground_idx` AND in `reach`; `LG = L[free][:,free].tocsc()`; `solve = scipy.sparse.linalg.factorized(LG)` (skip if `free` empty).
- Per parcel: entry node index `gi` (via `node_index[entry_i]`, or None); `leg_i = hypot(rep_xy[i] − entry_xy_i)`. If `gi` is a ground node → `R_i = leg_i`; elif `gi` in `free` (reachable) → solve `LG x = e_k` once per distinct `k`, `R_i = x[k] + leg_i`; else (`gi` None / unreachable / `leg` None) → `R_i = cap`.
- Return `float(mean(R_i))`.

### 2. `_BlockScoringContext` additions
- Freeze in `__init__`: `self.cap` = block bbox diagonal; `self.streets_geom` = `unary_union(block.streets.geometry)` (for the ground test) — cheap, once per block.
- `_ground_indices(self, node_index) -> np.ndarray`: node indices whose `(x,y)` is within `tol` of `self.streets_geom` (vectorized `shapely.distance(points, streets_geom) <= tol`).
- `resistance_frozen(self, roads_prefix, *, entry, splits) -> float`: mirror `score_frozen`'s CSR build (`base_pairs = [*prefix_segs, *street_segs]`, `_build_csr(base_pairs, splits)`), then `_resistance_core(csr, node_index, entry, self.rep_xy, self._ground_indices(node_index), self.cap)`. Empty graph → return `self.cap` (all parcels unreached).

### 3. `resistance_benefit(block, roads_full, *, tol=STREET_TOL, k=40)` factory
Mirror `_efficiency_factory` exactly:
```
ctx = _BlockScoringContext(block, k=k, tol=tol)
entry, splits, edge_pairs = ctx._derive_entries(roads_full)
if ctx.n < 2 or not edge_pairs: return lambda _roads: 0.0
R0 = ctx.resistance_frozen(None, entry=entry, splits=splits)          # no-roads baseline
def f(prefix): return 0.0 if R0 <= 0 else (R0 - ctx.resistance_frozen(prefix, entry=entry, splits=splits)) / R0
return f
```
Signature matches `BenefitFactory` (`(block, roads_full, *, tol) -> f(prefix) -> float`).

### 4. Wire into `compare.py`
In the per-(region, method) loop, alongside the existing three:
```
resistance = cost_benefit_curve(block, roads, benefit_fn=resistance_benefit, cost=cost, corridor_m=..., tol=tol)
raw.append((name, label, "resistance", resistance))
```
`groups`/AUC-table/curve emission are metric-agnostic (built dynamically from `raw`), so the resistance AUC table + curve plot flow automatically. Update the module comment ("The three lenses…" → four).

## Correctness gates (tests)

1. **Tree ⇒ shortest-path (investigation §2a).** On a TREE road set (a dijkstra output), each parcel's `R_i` equals its door-to-door shortest-path distance to the nearest street to ~1e-6 (resistance distance = shortest-path on a tree). A small hand-built tree with known distances is the cleanest unit case.
2. **Known resistor network.** A tiny graph (e.g. two parallel unit wires ground→node) with an analytic effective resistance (0.5) — `_resistance_core` returns it.
3. **Monotonicity (Rayleigh).** `benefit` is non-decreasing across `cost_benefit_curve` prefixes on a real block (frozen entries; edges only added). `benefit(∅) == 0`.
4. **Intensive, not extensive (the caveat — MUST test).** Duplicating/adding disconnected extra nodes to a graph does NOT change `mean(R_i)` for the original parcels (guards against anyone swapping in `wᵀL_G⁻¹w`/raw Kf). Assert the per-parcel mean is used.
5. **Unreached cap.** A block/road set where a parcel has no entry or no path to ground → that `R_i == cap`; `benefit` finite and in a sane range.
6. **AUC sanity on real data.** On ≥3 DJI blocks, `resistance` AUC is finite, arterial/dijkstra rank above the empty baseline, and on a DEEP block resistance credits the through-road (agrees with the investigation's §2c/§2d: grounded R prefers the connecting road where directness may not).
7. **Additive invariant.** `access`/`efficiency`/`directness` curves + AUCs unchanged; the full existing suite stays green.

## Task decomposition

1. **`_resistance_core` + Laplacian solve** (+ unit tests: known resistor network, tree⇒shortest-path, unreached cap, intensive-mean). Pure function; no context wiring yet.
2. **`_BlockScoringContext` additions (`cap`, `streets_geom`, `_ground_indices`, `resistance_frozen`) + `resistance_benefit` factory** (+ monotonicity + benefit-∈-sane-range tests).
3. **Wire into `compare.py` + AUC sanity-check on DJI blocks + docs** (README compare section mentions the resistance lens; `metrics-north-star.md` marks Piece 2 adopted). Confirm the additive invariant (existing lenses unchanged).

## Out of scope (follow-ups)

- Resistance-based greedy arterial with rank-1 (Sherman-Morrison) marginals (the investigation's §4 — the big perf lever for a resistance-objective method).
- The internal-circulation resistance analog `(E_R, directness_R)`.
- GPU batching.
- Making `resistance` a run-time `Eval` (kcomplexity-style single-proposal score) — this spec scopes it as a compare LENS only.
