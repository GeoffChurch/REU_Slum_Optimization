# DijkstraReblocker — boundary-routed street network — Design

**Status:** draft for review · **Date:** 2026-07-09

A new `Method` that reblocks a block by routing roads **along parcel boundaries** as a
shortest-path forest rooted at the street, instead of `PeelReblocker`'s straight
center-to-center descent. Validated by a spike (`scratchpad/peel_v2c_spike.py`) on the
Cape Town flagship: **98% frontage coverage vs peel's 23%, 25% less total road, a clear
arterial→lane hierarchy, at peel's ~1 s cost** (one multi-source Dijkstra).

This is **sub-project 1** of the peel-upgrade. The incremental budget-curve comparison
framework (methods emit *ordered* roads; a `Σ_buildings depth²` eval; a cost–benefit-curve
comparator) is **sub-project 2**, specced separately. This method is designed to feed that
framework (it carries per-segment drainage → a natural build order) but stands alone: it is
a drop-in `Method`, scored by the existing `KComplexityEval` today.

## Why

`PeelReblocker` links each interior parcel to its steepest-descent parent with a straight
line between representative points. Those lines **cut diagonally through parcels** (unbuildable
— you'd demolish structures), emit one segment per parcel (a flat thicket, no hierarchy), and
don't consolidate. It's a connectivity sketch, not a street plan. Topology produces a good
network but is O(parcels²) with an exponential `all_simple_paths` inner call — 40+ min on a
2,000-parcel block. The opportunity: peel's *cost* is fine (O(parcels)); only its *output* is
crude, and that's cheap to fix.

## Algorithm

Given a `Block` (parcels = Voronoi cells, `streets`):

1. **Boundary graph** (deterministic, fresh from shapely — no `ext/topology` coupling):
   `unary_union` of all parcel boundaries → a noded planar edge set (shared party-walls
   dedup automatically); nodes = boundary vertices (snapped to cm so shared vertices match),
   edges = boundary segments weighted by length.
2. **Street sources**: graph nodes within `STREET_TOL` (0.5 m) of `block.streets`.
3. **Shortest-path forest**: one **multi-source Dijkstra** from all street nodes → every node's
   shortest route to the street. For each non-street-fronting parcel, route its nearest boundary
   node to the street; the union of these routes is the road network. Shared prefixes coalesce
   into arterials near the street → natural consolidation and hierarchy. **Drainage** = count of
   parcels whose route uses each edge.
4. **Coverage spurs**: a parcel served only at a *vertex* (its route leaves via a neighbor's
   edge) has access but no buildable *frontage edge*. For each such parcel, add its cheapest
   boundary edge **incident to its network-touching node** (so the spur attaches to the
   network — never a floating road) with drainage = its through-count.
5. **Emit** `Proposal.roads`: a `GeoDataFrame` of the road segments with a `drain` column
   (int), rows ordered by `drain` descending (arterials first) — the priority order
   sub-project 2 slices for budgets, and the width channel the renderer uses for hierarchy.

**Determinism:** no RNG. Dijkstra ties and `min(...)` selections break on node/edge tuples,
so the same block yields byte-identical roads. (Mirrors `PeelReblocker`'s determinism; the
`run()` purity contract holds — no global RNG touched.)

## Efficacy (why Δk drops)

`KComplexityEval` → `parcel_access_layers` → `street_connectivity` seeds BFS depth-1 for
parcels within 0.5 m of the **street-connected** road network (floating roads grant nothing).
Because the forest is rooted at the street, **every segment is street-connected**, so every
routed parcel reaches depth 1 → `k_after ≈ 1`, `connected_road_frac = 1.0`. The forest alone
drives Δk; spurs add frontage realism without changing access. Expect Δk **≥** peel's on the
same block (peel's through-parcel roads also confer access, but the network fronts more parcels
cleanly).

## Decisions (my calls — flag any in review)

- **Name / placement:** `DijkstraReblocker` in `src/reblock/methods/dijkstra.py`;
  `conf/method/dijkstra.yaml`; `proposal_id = "dijkstra"`; `identity = ("dijkstra",)` (deterministic,
  no params). Named for its algorithm, like its siblings (`PeelReblocker` = steepest-descent,
  `TopologyMethod` = topology graph).
- **Graph built fresh from shapely, method-internal.** Self-contained (spike-proven), avoids
  `ext/topology`. `propose` is cached via `derive`, so no separate graph derivation for v1.
- **`Proposal.roads` gains a `drain` int column** (the `Proposal` contract is unchanged — `roads`
  is a `GeoDataFrame`; columns are free). Ordered drainage-descending.
- **Spurs attach to the network** (incident to the touching node) — never floating.
- **`networkx` dependency:** already vendored via `ext/topology` and used in `derive/access.py`;
  no new dep.

## Preserves / integration

- A drop-in `Method` (`propose(block, prior=None) -> Proposal`); selectable via `method=dijkstra`;
  scored by existing evals; renders via the existing emitter (drainage-weighted width is a small
  render enhancement — a task below).
- Does **not** touch `PeelReblocker`/`TopologyMethod` (new method, no dual path — they coexist as
  distinct methods, which is correct; "no warts" binds duplicated *machinery*, not alternative
  algorithms the user explicitly wants side-by-side).

## Testing

- **Efficacy (real):** on the committed flagship `ZAF.9.3.1_1_44882`, `delta_k > 0`,
  `k_after` small, `connected_road_frac == 1.0`, frontage coverage high (≥90%).
- **Efficacy (synthetic):** the 3×3 grid block (center parcel depth 2, as used by the peel/topology
  tests) → `k_after == 1`, `delta_k > 0`, roads non-empty and street-connected.
- **Determinism:** same block twice → WKT-equal roads; no global `np.random`/`random` mutation.
- **Coverage/hierarchy:** `drain` column present, positive, ordered descending; every road segment
  street-connected (no floating spur).

## Out of scope (sub-project 2)

The `Method` interface exposing an *ordered* road sequence as a first-class contract, the
`Σ_buildings depth²` (q=2) eval, the coarse-sampled cost–benefit curve + area-under-curve
comparator, and cross-method budget comparison. This method carries `drain` so it plugs in later,
but none of that framework is built here.
