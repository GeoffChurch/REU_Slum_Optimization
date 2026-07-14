> **STATUS (2026-07-14): ADOPTED — both pieces are on `main`.** Piece 1 (line-proximity
> entries) = commit `1769ee2` (fully replaced vertex-entry; see `_line_entries` in budget.py
> + `test_line_proximity_scores_a_sparse_straight_chord`). Piece 2 (grounded resistance) =
> `resistance_benefit` in budget.py. Preserved from the retired `north-star-metric` branch
> for the record; the branch's dual-path prototype (`entry=` option) is obsolete.

# North-star metric — implementation design (BRANCH PROTOTYPE, for review; NOT for merge)

Turns `docs/metrics-north-star.md` from vision into a prototype on branch `north-star-metric`.
It touches the **shared** `budget.py` metric, so it is built + measured on a branch and left for
the owner to review before anything merges (it changes every method's numbers).

Two pieces, independent, both prototyped:

## Piece 1 — Line-proximity entries (fixes the vertex artifact)

**Problem (established while building arterial):** `_entry_nodes` (budget.py) maps each parcel
to the nearest graph *vertex* within `tol`. Dense-vertex roads (boundary paths) are fine; sparse
2-point chords (aspirational arterial) are undercounted, inverting the price-of-buildability.

**Fix:** a parcel's entry is the nearest *point on a road edge* within `tol`, injected as a graph
node by splitting that edge at the projection.
- Build the road+street graph as today (`_road_street_graph`).
- STRtree over the *edges* (LineStrings). For each parcel rep point: nearest edge within `tol`;
  project the rep point onto it (`edge.interpolate(edge.project(pt))` → P); split the edge at P
  (remove (u,v), add (u,P) and (P,v) with length weights); P is the parcel's entry node.
- Do all edge-splits first (collect per edge, sort split points along the edge, rebuild), then
  run the existing sampled-shortest-path pass. Deterministic (sorted).

**Guardrail — monotonicity.** The vertex-entry rule was a *deliberate* fix for E/directness
*falling* as roads were added (see `_entry_nodes` docstring). Line-proximity must not regress it.
The existing monotonicity (freeze entries against the FULL road set, vary only edges) still
applies: compute entry projections against the full-road graph once, then evaluate prefixes with
those fixed entries. Re-run `tests/test_budget.py::...monotone...` — it MUST stay green.

**Measurement (the point of the branch):** an offline script comparing OLD vs NEW entry on real
DJI blocks:
- dijkstra / mesh / greedy_arterial (buildable) AUC on access/E/directness — expect ~unchanged
  (dense-vertex roads: nearest-vertex ≈ nearest-point).
- greedy_arterial **aspirational** — expect the inversion FIXED (aspirational ≥ buildable), i.e.
  the price-of-buildability becomes measurable, *without* the densification hack.
- Report a before/after table. This is what tells the owner whether to adopt.

## Piece 2 — Grounded effective resistance (cheap, redundancy-aware, monotone)

A new `benefit_fn` factory `resistance_benefit(block, roads_full, *, tol)` in `budget.py`
(alongside `access_benefit`/`efficiency_benefit`/`directness_benefit`), returning `f(roads_prefix)
-> float`:
- Nodes/edges = `_road_street_graph` (roads + streets); edge conductance `c = 1/length`.
- Ground the egress nodes = street nodes (nodes within `tol` of `block.streets`).
- Effective resistance of each parcel's entry node to ground = `(L_G^{-1})_{ii}`, where `L_G` is
  the weighted graph Laplacian with grounded rows/cols removed (invertible). Solve the sparse
  system (`scipy.sparse.linalg.spsolve` / a Cholesky) for the diagonal — or `numpy` on the dense
  reduced Laplacian for prototype-scale blocks.
- Benefit = `1 - Σ_i R_i(roads) / Σ_i R_i(∅)` (normalized resistance-to-egress reduction; higher
  = better). Demand weights uniform for v1.
- **Monotone** under edge addition (Rayleigh) — verify with a test (resistance only falls as roads
  are added). **Redundancy-aware** — verify a mesh (loops) beats a tree of equal length on this
  metric where directness/E are near-equal (the whole reason to prefer resistance).
- Note the cheap-marginal (Sherman-Morrison) opportunity in a comment; do NOT implement the greedy
  integration in v1 — just the metric + the two property tests.

**scipy dependency:** if scipy isn't already a dep, use dense `numpy.linalg` on the reduced
Laplacian for prototype-scale blocks (fine up to a few thousand nodes) rather than adding a dep.

## Deliverables (branch `north-star-metric`, off `main`)
1. Line-proximity entry (Piece 1) as an *option* (a flag / separate function — do NOT delete the
   vertex path yet) + the OLD-vs-NEW measurement script + report table.
2. `resistance_benefit` (Piece 2) + monotonicity test + redundancy-aware test.
3. A short report: does line-proximity fix aspirational without regressing dijkstra/mesh? does
   resistance behave (monotone, redundancy-aware)? Recommendation on adoption.
Left UNMERGED for owner review (changes the shared metric).
