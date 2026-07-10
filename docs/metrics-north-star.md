# Reblocking metrics — north star

**Status:** guiding vision · **Date:** 2026-07-09 · **Piece 1 (line-proximity entries) ADOPTED
2026-07-10** — `network_efficiency` now maps a parcel to the nearest point on a road edge (not the
nearest graph vertex), fixing the sparse-chord undercount that inverted the price-of-buildability;
this unblocked aspirational arterial as a valid directness ceiling. Piece 2 (grounded effective
resistance) remains a prototype on branch `north-star-metric`.

This is the metric we are *aiming* at. Today's three lenses — access (Σ depth²),
network-efficiency E (mean 1/d), and directness (mean euclid/d) — are useful proxies, but
they disagree with each other because none of them *is* the quantity of interest; they are
three shadows of it. This doc names the real target and the cheap proxies worth exploring on
the way there.

## What a road network is actually for

Let people **get places cheaply** — out to jobs / transit / services (egress) and around
internally (neighbours, shared facilities) — subject to **everyone being reachable at all**.
Everything else (directness, redundancy, efficiency) is a means to that end.

## The north-star metric

**Demand-weighted mean travel cost — equivalently, accessibility — over real network distance,
with universal reachability as a hard constraint.**

1. **Reachability is a constraint, not a score.** Every parcel within `tol` of a road that is
   connected to a *live* egress — 100%, or the design is disqualified. You do not "trade off"
   leaving homes stranded. (Access-depth gestures at this; the right form is binary, and *which*
   egress counts is the [segment-group egress model](superpowers/specs/2026-07-09-greedy-arterial-reblocker-design.md)'s job.)

2. **Objective:** minimize
   ```
   C = Σ_ij  w_ij · d_ij  /  Σ_ij  w_ij
   ```
   - `d_ij` — **real network distance in metres** (not adjacency hops, not a unitless ratio).
     Each parcel's entry is the nearest **point on the noded road line** (line-proximity, graph
     split at the projection), *not* the nearest graph vertex. This one change is the root fix
     for the vertex-vs-line artifact (see below): a straight chord then serves every parcel it
     passes, and it is exactly what `access` scoring already does correctly.
   - `w_ij` — **trip demand**. For informal settlements the dominant trip is egress, so weight
     parcel→egress heavily, plus a gravity term for internal trips. Uniform all-pairs (today's
     E) is the demand-blind special case.
   - **Benefit** = `C(∅) − C(roads)`; the cost-benefit curve becomes travel-cost-removed per
     road-metre — same shape as today, but grounded in metres and real behaviour.

3. **Cost:** road density (m/ha) — unchanged; already right.

### Why this unifies the three lenses

- **access** falls out — an unreachable parcel is ∞/huge trip cost, strongly penalized.
- **directness / circuity** falls out — detours are longer `d_ij`, higher cost.
- **E** *is* this metric with `f(d)=1/d` and uniform demand — a special case.

It is the "one desideratum to rule them all" we kept circling, done properly: right units,
demand-aware, with the parcel→graph entry problem fixed at the root.

### The artifact this explains

We hit a concrete symptom while building the arterial method: `network_efficiency` scores a
parcel as "served" only if a graph **vertex** sits within `tol` of it. Buildable roads (boundary
paths) are vertex-dense so this ≈ line-proximity; ideal straight chords (2 vertices) are not, so
they get massively undercounted, inverting the price-of-buildability. Densifying the chords
was tried and **rejected** — it bloats the graph and makes scoring too slow — so the aspirational
ceiling is currently *deferred*, not patched (the artifact stands). **The north-star's
line-proximity noded entry would dissolve it** — every method, including the aspirational
ceiling, "just works" — which is the real reason to move to it.

## Cheap proxies (and yes — spectral ones are the interesting part)

The exact objective needs all-pairs (or sampled) shortest paths — O(N·(E + N log N)). Two
families of cheaper surrogate:

### Spectral — model movement as electrical flow

Treat the network as a resistor grid, each edge a conductance ∝ 1/length (or ∝ width). The
**effective resistance** `R_ij` between two nodes is a distance that accounts for *all* paths,
not just the shortest — so it **rewards redundancy** (mesh loops lower resistance), which
shortest-path E does not. Movement ≈ current flow; "easy to get around" ≈ low resistance.

- **Grounded resistance-to-egress — the best match for the north star.** Ground the egress /
  street nodes; each parcel's effective resistance *to ground* is how hard it is to get out.
  `R_i = (L_G^{-1})_{ii}` where `L_G` is the weighted graph Laplacian with the grounded rows/cols
  removed (invertible); `Σ_i w_i R_i` is a demand-weightable, egress-focused accessibility — a
  spectral/linear-algebra form of the north-star's egress-weighted travel cost. Redundancy-aware
  and cheaper than all-pairs shortest paths (one sparse linear solve).
- **Kirchhoff index `Kf = Σ_{i<j} R_ij = N · Σ_{k≥2} 1/λ_k`** (λ_k = nonzero Laplacian
  eigenvalues) — total all-pairs resistance in one spectral number; a global navigability score
  that credits mesh redundancy. Estimable in near-linear time (Hutchinson trace of `L^+`, or a
  few Lanczos eigenpairs).
- **Algebraic connectivity `λ₂`** (Fiedler value of the weighted Laplacian) — a single number
  for global connectivity / bottleneck-resistance / mixing speed. Crude (topology-ish, not a
  demand-weighted cost) but the cheapest spectral pulse-check.
- **Heat-kernel / random-walk** cousins — communicability `Σ e^{-tλ_k}`, mean first-passage
  time, current-flow (random-walk) betweenness for *which* segment matters. All spectral, all
  redundancy-aware.

**Two properties make grounded resistance especially attractive for *us*:**
1. **Monotone under edge addition** (Rayleigh monotonicity) — resistances only fall as roads are
   added, matching our cost-benefit monotonicity requirement for free.
2. **Cheap marginals** — adding an edge is a rank-1 Laplacian update, so Sherman-Morrison gives
   the new resistances (and the marginal gain of a candidate arterial) *without* re-solving from
   scratch. That directly attacks the greedy arterial method's dominant cost (today it re-scores
   every candidate with a full sampled all-pairs pass). A resistance-based greedy could score
   candidates by a near-`O(1)` resistance-drop lookup.

### Non-spectral — cheap and blunt

- **Landmark / sampled shortest paths** (what we do now: K seeded sources) — approximate C
  directly; simplest, tunable accuracy.
- **Sampled detour ratio** — Monte-Carlo `euclid/d` over random OD pairs; a cheap directness read.
- **Road density + reachable fraction** — near-free coarse screen (are all parcels reached, at
  what pavement cost) before any expensive metric.

## Caveats

- **Resistance distance ≠ shortest-path distance.** It is an all-paths average, so it *rewards
  redundancy* — a feature for navigability/robustness, but if you specifically care about the
  single best evacuation route, shortest-path is the truer model. The two answer different
  questions; likely we want resistance for "ease of circulation" and shortest-path/`d_ij` for
  "egress time."
- **Demand is modelled, not measured.** We have no real origin-destination flows; `w_ij` would
  start egress-weighted + gravity and needs validating against something (movement traces,
  surveys, or at least plausibility).
- **Entry-noding cost.** Line-proximity entries mean splitting the graph at each parcel's
  projection — more setup than today's vertex-touch, though the resistance solve amortizes it.

## Near-term vs long-term

- **Near-term:** keep the three lenses; the vertex artifact stays *unpatched* (densification
  was evaluated and rejected as too slow), so the aspirational ceiling is deferred.
- **Long-term:** replace them with demand-weighted travel cost, entries line-noded — and
  evaluate **grounded effective resistance** as the cheap, redundancy-aware, monotone,
  cheap-marginal spectral surrogate (which may also let the greedy methods score candidates far
  faster than the current full re-scoring).
