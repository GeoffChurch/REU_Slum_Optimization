# MeshReblocker + multi-metric grading — Design

**Status:** draft for review · **Date:** 2026-07-09

A new `Method` that produces **crossing roads** (loops, through-roads) instead of the
myopic tree the dijkstra/peel methods grow — plus the grading it needs: the cost-benefit
framework gains **pluggable benefit metrics** so every method is graded on three lenses —
access (Σdepth², existing), **directness/circuity**, and **E (network efficiency)**.

First of the "new methods" set (mesh → grid → reassess → multi-block). Depends on
DijkstraReblocker (the mesh builds on its forest) and the cost-benefit framework (shipped).

## Why a mesh, and how to grade it

Dijkstra/peel are **trees** (a shortest-path forest is acyclic), so they can't have
crossings — every parcel gets one myopic path to the street. They already max *access*
(every parcel depth 1), so on the access curve a crossing road is pure cost, zero benefit.
The mesh's value is **navigability** — direct inter-parcel movement, redundancy — which
needs a network-quality lens the access curve doesn't provide.

## The metrics (three lenses)

All computed at the ~20 budget prefixes, benefit rising with roads, per the existing curve
machinery. The framework's `cost_benefit_curve` is generalized to take a `benefit_fn`.

- **access** — `1 - Σdepth²(prefix)/Σdepth²(∅)` (existing; refactored behind `benefit_fn`).
- **E (network efficiency)** — global efficiency over the road network: `E = mean(1/d_ij)`
  over parcel pairs, `d_ij` = shortest road-network distance (∞ for unreached → 0). Higher =
  better; both access (connect → finite d) and circuity (loops → shorter d) raise it. Curve
  = raw `E(prefix)` vs road density; AUC = area (higher = more navigability per meter).
- **directness / circuity** — `mean(euclid_ij / d_ij)` over the same reachable pairs
  (∈ (0,1], higher = straighter). This is `1/circuity`. Loops raise it; a tree is low.

**Cost**: E/directness are all-pairs, O(N²). Compute on a **fixed random sample of K source
parcels** (K≈40, seeded per block for determinism) to a landmark destination set (the
street + the same K) — turns O(N²) into O(K·N), affordable at 20 budget points. E and
directness come from the *same* sampled shortest-path pass. (Access stays the cheap
single-BFS-to-street.)

## MeshReblocker

Given a `Block`:
1. **Forest** — the dijkstra forest (reuse `_reblock_dijkstra`): roads + drainage, the base.
2. **Candidate loops** — boundary-graph edges NOT in the forest whose *both* endpoints are
   already forest nodes (adding one closes a loop = a through-road/shortcut).
3. **Value proxy (cheap, avoids the greedy-circuity blowup)** — rank candidates by the
   **shortcut ratio** `forest_path_distance(u,v) / edge_length`: a loop joining two nodes
   far apart *in the tree* but close in space eliminates a big detour per meter. One
   Dijkstra on the forest gives all the path distances; per-candidate value is a lookup.
4. **Emit** — forest roads (drainage-ordered) followed by loop roads (shortcut-ratio
   ordered), one incremental sequence with a `drain` column carried through — budget-
   sliceable exactly like the tree methods, and on the E/directness curves the loop segments
   are where it pulls ahead of plain dijkstra.

`MeshReblocker` is deterministic (no RNG); `identity = ("mesh",)`; `conf/method/mesh.yaml`;
`proposal_id/method = "mesh"`.

## Decisions (my calls — flag any in review)

- **Circuity-per-meter objective via the shortcut-ratio proxy** — true per-candidate circuity
  is O(N²) each (the greedy trap we dodged with drainage); the forest-path/length ratio is a
  one-BFS proxy for the same thing.
- **Three grading lenses, `benefit_fn`-pluggable** (keep access + circuity, add E) — the
  compare tool emits a curve + AUC per (method, metric).
- **Sampled E/directness** (K≈40 seeded sources) — O(N²) → O(K·N); one pass yields both.
- **Mesh builds ON the dijkstra forest**, doesn't reimplement it.

## Testing

- **Framework**: `cost_benefit_curve(benefit_fn=...)` — access unchanged (existing tests
  green); an `efficiency_benefit`/`directness_benefit` on a synthetic block rises with roads;
  a loop (mesh) scores higher E/directness than the same block's tree (dijkstra) at equal or
  near-equal cost.
- **MeshReblocker**: on the 5×5 grid, produces the dijkstra forest PLUS ≥1 loop (a candidate
  with shortcut ratio > 1 exists); deterministic (WKT-equal); every road street-connected;
  `identity`/`proposal_id`/config wiring; efficacy — mesh's directness/E > dijkstra's.
- **Real block**: `method=mesh` composes + reblocks a DJI block; the 3-lens compare shows mesh
  ≈ dijkstra on access but higher on E/directness (crossings help navigability, not depth).

## Out of scope

The grid reblocker (next sub-project) and multi-block joint reblocking (after reassessment).
Trip-distribution weighting of E (egress vs circulation) — uniform sample for v1.
