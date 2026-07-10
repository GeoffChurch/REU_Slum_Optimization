# GreedyArterialReblocker + price-of-buildability — Design

**Status:** draft for review · **Date:** 2026-07-09

A new `Method` that grows a road network by **greedily inserting the single best straight
arterial** — the through-road (or interior spur) with the highest objective gain per meter —
one at a time until a road budget runs out. Unlike the tree methods (dijkstra/peel, one myopic
path per parcel) and the mesh (forest + local loops), this puts *long, high-value roads where
they help most*, which should make it a competitive baseline — especially for multi-block
regions later, where region-spanning arterials are the whole point.

It ships in **two modes** — a *buildable* one (arterials snapped to parcel frontages, zero
demolition, like every other method) and an *aspirational* one (true straight chords through
the settlement, an upper bound assuming you could rebuild ideally). Grading both on the same
cost-benefit curves yields a quantity nobody has measured: **the price of buildability** — how
much connectivity you forfeit by refusing to demolish.

Third of the "new methods" set (mesh → grid/arterial → reassess → multi-block). Depends on the
boundary-graph routing (dijkstra internals) and the cost-benefit / multi-metric compare
framework (both shipped). The fixed regular grid was considered and **dropped** — the greedy
arterial subsumes its value (it produces grid-like structure wherever a grid actually helps)
without imposing regularity where it doesn't.

## The greedy loop

Given a `Block` (parcels + `streets` = existing egress) and a `benefit_fn`:

```
network  = streets                      # the connected road network; transitively reaches egress
roads    = []                           # committed arterials, in greedy (value-descending) order
while cumulative_road_density(roads) < density_cap:
    cands = generate_candidates(network, parcels)          # straight chords (below)
    scored = [(delta_benefit(commit c) / length(c), c) for c in cands]
    best_gain, best = max(scored)
    if best_gain < gain_floor:                             # diminishing returns -> stop early
        break
    roads.append(best)
    network = network ∪ realize(best)                      # grow the network for the next step
return roads
```

Roads emerge in value order, so the emitted sequence is **natively budget-sliceable** by
`cost_benefit_curve` — no drainage/shortcut-ratio proxy needed; the greedy order *is* the
budget order.

`realize(c)`, `length(c)`, and the scored geometry are all **mode-dependent** (below): in
buildable mode `realize` is the snapped frontage path and `length` is that path's length (what
actually gets built); in aspirational mode `realize`/`length` are the ideal chord itself. The
gain-per-meter denominator is always the *realized* road's length in the active mode.

## Candidates + guaranteed connectivity (transitive-network anchoring)

The connectivity requirement is **not** "endpoints on the outer boundary" — it is *"attaches
to the current connected network, which is transitively connected to egress."* Since `streets`
is egress and every committed arterial attaches to the network, any new arterial that touches
the network at ≥1 point is transitively connected — **no floating roads**, by construction.

Each step, from the *current* network (streets ∪ committed arterials):
- **Anchor points** `A` = points sampled at fixed spacing along the current network geometry.
- **Interior targets** `T` = the centroids (snapped to the nearest boundary node) of the
  parcels with the deepest current access (top-k by access depth) — the places most starved
  of a road.
- **Candidate chords** = `{ A_i–A_j }` (through-roads: both ends on the network → loops /
  shortcuts) `∪ { A_i–T_k }` (spurs: one end anchored, the other reaching a deep pocket).

The candidate set *grows with the network* (new arterials add anchor points), so later
arterials can **branch off earlier ones** and drive spurs deep into the interior. Counts are
bounded — O(B²) through-roads + O(B·k) spurs per step, and steps are few (arterials, not
per-parcel), which is exactly why honest per-candidate marginal scoring is tractable here where
it was not at the per-parcel scale.

## Regime C: two modes, one code path

A single algorithm parameterized by `mode`:

- **buildable** — a candidate chord `p–q` is *realized* on the parcel-boundary graph: snap
  `p`,`q` to their nearest boundary nodes, then take the boundary path between them that hugs
  the ideal line — a shortest path with edge weight `length(e) + λ·mean_dist(e, line_pq)`. The
  emitted road is that frontage-following path; it attaches to the network at the anchored end,
  so `street_connectivity` holds. Scoring uses the snapped path.
- **aspirational** — the candidate *is* the true straight chord (no snap). It is scored by
  treating the chord as a new linear street: parcels within `tol` of the line (including ones
  it passes through — demolition implied) are served. This is an idealization / ceiling.

Both modes reuse the *same* proximity-based access/E/directness scoring — an ideal chord is
just a `LineString` that happens to cut across parcels rather than follow their frontages;
`parcel_access_layers` already serves parcels by road proximity, so no new scoring code is
needed. Each mode runs its own greedy (each optimizes the roads *it* can build), so the
aspirational curve is the genuine best-ideal ceiling, not merely the unsnapped shadow of the
buildable choices.

**Price of buildability** falls out for free: register both modes as two entries in
`compare_config`; the existing multi-metric compare emits both curves per lens, and the gap
between the aspirational and buildable curves *is* the price of buildability. No new machinery.

## Objective (pluggable, default access-burden)

`benefit_fn` is one of the existing factories (`access_benefit` / `efficiency_benefit` /
`directness_benefit`) — the greedy maximizes marginal `benefit_fn` per meter. Default is
**access-burden reduction**: in a deep informal block a new through-street's most direct payoff
is dropping many parcels to depth-1, it is the primary reblocking goal, and it is the cheap
single-BFS score (E/directness cost K× more per candidate). E/directness come along for the
ride and are reported by the compare. The objective is part of the method identity, so
`greedy_arterial` optimizing access vs directness are distinct, separately-graded methods.

## Budget / emit / eval integration

The greedy emits arterials in value order. To slice with the *unchanged* `cost_benefit_curve`
(which orders by the `drain` column descending), the emitted roads carry `drain` = descending
greedy rank (first-committed arterial = highest), so `drain`-descending order reproduces the
greedy commit order. (`drain` here is the generic "budget priority" slot the curve machinery
already keys on, not literal drainage.) Graded on all three lenses by the existing compare.

`GreedyArterialReblocker` is deterministic (sampling at fixed spacing, sorted candidates,
argmax with a geometry tiebreak; no RNG). `identity = ("greedy_arterial", mode, objective)`;
`conf/method/greedy_arterial.yaml` (mode + objective params); `proposal_id/method` encode
mode + objective; two `compare_config` entries (buildable + aspirational).

## Scope

**Single-block v1**, same `Block` contract as every other method. It extends to region scale
for free once the multi-block substrate lands — the network anchors simply include inter-block
streets, and region-spanning arterials become candidates with no algorithm change.

## Testing

- **Obvious-arterial fixture** — a long block with a deep interior pocket reachable by one
  through-road; the greedy's *first* committed arterial is that road (highest gain/meter).
- **buildable mode** — every road ⊆ the boundary graph and street-connected
  (`street_connectivity(...).connected_frac == 1.0`); deterministic (WKT-equal across runs).
- **transitive anchoring** — a fixture where the 2nd arterial's best anchor is on the *1st
  arterial*, not on `streets`; assert it commits there (proves network-growth anchoring, not
  frontage-only).
- **aspirational mode** — ideal chords; deterministic; at equal budget aspirational benefit
  ≥ buildable benefit (price-of-buildability ≥ 0 — the ceiling dominates).
- **budget slicing** — `cost_benefit_curve` on the emitted roads reproduces greedy order
  (monotone benefit as budget grows).
- **real DJI block** — `method=greedy_arterial` reblocks; the 3-lens compare vs dijkstra/mesh
  reports where it wins (expected: strong on directness/E; competitive on access via new
  streets), plus the aspirational-vs-buildable gap.

## Decisions (my calls — flag any in review)

- **Straight-first; constant-curvature arcs deferred.** Arcs add exactly one search dimension
  (~10× more candidates/step) with no closed-form optimum, and the boundary snap absorbs most
  of what curvature buys in the buildable output — so arcs mainly help the aspirational ceiling
  and are a clean later parameter extension (the road primitive is a curvature-parameterized
  polyline; straight = curvature 0).
- **Two modes as two method instances** (buildable + aspirational) → price-of-buildability via
  the existing compare, no new machinery.
- **Pluggable objective, default access-burden.**
- **Transitive-network anchoring** (attach to streets ∪ arterials at ≥1 point) rather than
  direct-frontage endpoints.
- **`drain` = greedy rank** so the unchanged curve machinery slices in greedy order.
- **Honest full marginal re-scoring** each step (tractable because arterials are few); lazy-
  greedy (CELF, exploiting access submodularity) is the escape hatch if it drags, deferred.

## Out of scope / backlog

- **Constant-curvature arcs** — the curvature-parameterized primitive is in place; the arc
  search + aspirational-ceiling evaluation is the follow-up.
- **Region-level / multi-block** — this method extends there for free once the multi-block
  substrate (one boundary graph over a region) exists; that substrate is a separate item.
- **Segment-group egress model** — a cross-cutting enhancement to the egress/`streets`
  definition: egress = a set of boundary-segment *groups*, the network must reach ≥1 *live*
  segment per required group, with some segments **inert** (beach, mountain). At region scale
  "connect to the outer boundary" is wrong (an island's perimeter is mostly inert); this
  generalizes it. Applies to *every* method (dijkstra, mesh, arterial), so it belongs to the
  egress model, not here.
- **Lazy-greedy / CELF acceleration** and cheap incremental marginal-gain proxies — only if
  naive full re-scoring is too slow on large blocks.
- **Fixed regular grid baseline** — dropped; the greedy arterial subsumes it.
