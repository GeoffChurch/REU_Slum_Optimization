# GreedyArterialReblocker + price-of-buildability — Design

**Status:** draft for review · **Date:** 2026-07-09

A new `Method` that grows a road network by **greedily inserting the single best straight
segment** — the through-road, spur, or continuation with the highest objective gain per meter —
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
roads    = []                           # committed segments, in greedy (value-descending) order
while cumulative_road_density(roads) < density_cap:
    cands = generate_candidates(network, parcels)          # through-roads + spurs + continuations
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

The connectivity requirement is *"attaches to the current connected network, which is
transitively connected to egress."* Since `streets` is egress and every committed segment
attaches to the network, any new segment that touches the network at ≥1 point is transitively
connected — **no floating roads**, by construction.

Each step, from the *current* network (streets ∪ committed segments), with anchor points `A`
sampled at fixed spacing along the network geometry, the candidate families are:

- **Through-roads** `{ A_i–A_j }` — chords between two anchors (cross-cuts / loops / shortcuts).
- **Spurs** `{ A_i–T_k }` — one anchor → a deep-access pocket `T_k` (the top-k deepest parcels),
  terminal for now. Cheap immediate access; whether it wins is up to the objective.
- **Continuations** — extend an existing committed segment from *either* endpoint outward to the
  network. This lets a spur committed cheaply now be *completed into a through-road later*, once
  the objective rewards the through-route. Continuations are largely automatic already (a
  committed segment's endpoints are network anchors, so through-roads from them exist), but are
  made first-class so every dead-end is always a continuation candidate.

**Either-direction continuations make true intersections.** Because both endpoints of every
committed segment are anchors, and a continuation can extend a segment's *line across its
anchor* to the far side, a spur that branched off a street can later be continued in the
*opposite* direction — skewering the street it branched from into a real **4-way intersection**
(a crossroads), not just a T. This is not special-cased: a crossing shortens *more* inter-parcel
routes than a stub, so the default directness objective actively wants it and builds it when it
pays. The one requirement it imposes: the emitted network must be **noded** at crossings —
segments split at the shared intersection point so it is a true graph node, not two overlapping
lines. In buildable mode crossings fall on boundary-graph nodes and node automatically; in
aspirational mode the ideal chords are planarized at their intersections.

Which family wins is decided by the **objective**, not the candidate set (see below): under the
default directness objective, through-roads and completing continuations dominate and spurs
survive only where they genuinely help; under an access objective, cheap spurs win and often
stay terminal. The candidate set *grows with the network*, so later segments branch off — and
cross — earlier ones. Counts stay bounded — O(B²) through-roads + O(B·k) spurs/continuations per
step, and steps are few (arterials, not per-parcel), which is why honest per-candidate marginal
scoring is tractable here where it was not at the per-parcel scale.

## Regime C: two modes, one code path

A single algorithm parameterized by `mode`:

- **buildable** — a candidate chord `p–q` is *realized* on the parcel-boundary graph: snap
  `p`,`q` to their nearest boundary nodes, then take the boundary path between them that hugs
  the ideal line — a shortest path with edge weight `length(e) + λ·mean_dist(e, line_pq)`. The
  emitted road is that frontage-following path; both endpoints anchor on the network, so
  `street_connectivity` holds. Scoring uses the snapped path.
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

## Objective (pluggable, default directness)

`benefit_fn` is one of the existing factories (`access_benefit` / `efficiency_benefit` /
`directness_benefit`) — the greedy maximizes marginal `benefit_fn` per meter. Default is
**directness** (navigability): this is the *arterial* method, and directness is what makes it
build actual arterials. Under it, through-roads and completing continuations dominate, spurs
survive only where they help, and a bare `method=greedy_arterial` run yields a through-road
network rather than a myopic spur pile — the objective, not a candidate restriction, is the
lever that decides the network's character. **Access-burden** is available as a non-default and
is a *revealing* contrast: it demonstrates that access-only optimization produces cheap terminal
spurs with poor navigability (a real result the compare surfaces). Note directness/E score K×
more per candidate than access's single BFS (see cost, below). The objective is part of the
method identity, so `greedy_arterial` under directness vs access are distinct, separately-graded
methods.

## Budget / emit / eval integration

The greedy emits segments in value order. To slice with the *unchanged* `cost_benefit_curve`
(which orders by the `drain` column descending), the emitted roads carry `drain` = descending
greedy rank (first-committed segment = highest), so `drain`-descending order reproduces the
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
  through-road; the greedy's *first* committed segment is that road (highest gain/meter).
- **buildable mode** — every road ⊆ the boundary graph and street-connected
  (`street_connectivity(...).connected_frac == 1.0`); deterministic (WKT-equal across runs).
- **transitive anchoring** — a fixture where the 2nd segment's best anchor is on the *1st
  segment*, not on `streets`; assert it commits there (proves network-growth anchoring, not
  frontage-only).
- **spur → continuation → true intersection** — under an access objective a spur is committed to
  a deep pocket; a continuation then extends it *across its street anchor* to the far side;
  assert the result is noded — the crossing is a shared graph node (degree ≥ 3 at the
  intersection), not two overlapping lines.
- **aspirational mode** — ideal chords; deterministic; the ideal chords are planarized at
  crossings; at equal budget aspirational benefit ≥ buildable benefit (price-of-buildability ≥ 0
  — the ceiling dominates).
- **budget slicing** — `cost_benefit_curve` on the emitted roads reproduces greedy order
  (monotone benefit as budget grows).
- **real DJI block** — `method=greedy_arterial` (default directness) reblocks; the 3-lens compare
  vs dijkstra/mesh reports where it wins (expected: strong on directness/E), plus the
  aspirational-vs-buildable gap.

## Decisions (my calls — flag any in review)

- **Straight-first; constant-curvature arcs deferred.** Arcs add exactly one search dimension
  (~10× more candidates/step) with no closed-form optimum, and the boundary snap absorbs most
  of what curvature buys in the buildable output — so arcs mainly help the aspirational ceiling
  and are a clean later parameter extension (the road primitive is a curvature-parameterized
  polyline; straight = curvature 0).
- **Two modes as two method instances** (buildable + aspirational) → price-of-buildability via
  the existing compare, no new machinery.
- **Pluggable objective, default directness** — this is the arterial method; directness makes it
  build through-roads (access would win cheap spurs). Access available as a revealing contrast.
- **Transitive-network anchoring** (attach to streets ∪ committed segments) rather than
  direct-frontage endpoints — the network is transitively connected to egress, so anchored
  segments are too.
- **Unrestricted candidates — through-roads + spurs + either-direction continuations.** The
  *objective* decides character, not a candidate ban: a spur can be bought cheaply and completed
  into a through-road later. A continuation extending a spur *across its anchor* forms a true
  crossroads, which directness naturally rewards. The emitted network is **noded** at crossings.
- **`drain` = greedy rank** so the unchanged curve machinery slices in greedy order.
- **Honest full marginal re-scoring** each step (tractable because arterials are few); lazy-
  greedy (CELF, exploiting submodularity) is the escape hatch if it drags, deferred.

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
- **Peninsula pockets under directness** — a spur into a true dead-end pocket adds little
  directness, so the default objective won't build it and such pockets stay underserved; serving
  them is an access concern, and *which* egress is live is the segment-group egress model's job
  (a "dead-end" is really "no live egress on the far side").
- **Lazy-greedy / CELF acceleration** and cheap incremental marginal-gain proxies — only if
  naive full re-scoring is too slow on large blocks.
- **Fixed regular grid baseline** — dropped; the greedy arterial subsumes it.
