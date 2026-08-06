# Road geometry in the conductance: replacing crow-flies with a route (2026-08-05)

**Status: SPEC'D, not built.** Scope is the EVALUATOR only; optimizing methods keep today's
constant-gain model as a documented first-order proxy (see [Scope](#scope-what-this-does-not-touch)).

**Goal in one line:** an edge covered by a road should be priced by the route a walker actually
takes, not by the straight line between two parcel centroids.

## Why this, and why not the mesh

This spec is the survivor of a larger idea. `specs/2026-07-30-road-first-mesh-design.md` listed four
defects; D3 was falsified 2026-08-04 (that spec's own "D3 was our own artifact" section), D4 closed
with per-road `width_m`, and D1/D2 remained:

- **D1** — a straight road and a zigzag covering an identical covered-edge set score
  **bit-identically**, at detour ratios up to **3.07x**. A method may draw a long winding road and
  be graded as though it drew the short straight one.
- **D2** — travel/crow-flies on covered edges has **median 1.395**, per-edge max 2.63. Conductance
  is overstated ~40%.

The plan going in was a medial-axis mesh: move nodes off parcel centroids onto the Voronoi
skeleton. **Measurement killed that framing.** Over 4,358 adjacent parcel pairs
(`scratchpad/spectral/corner_vs_gate.py`):

    gate / crow      median 1.000   p90 1.025   p99 1.351   mean 1.017
    corner / crow    median 1.094   p90 1.636   p99 2.649   mean 1.234

The route through the gate between two buildings is *exactly* the straight line — geometrically
forced, since the Voronoi edge is the perpendicular bisector of the two generators, so the
generator-to-generator segment crosses it at the midpoint. **Today's crow-flies length is already
correct for footpath edges.** D1 and D2 both say "on covered edges": they are about how ROADS enter
the conductance, not about where nodes sit. Node relocation is therefore a separate concern with no
measured defect attached, and is deferred (see [Deferred](#deferred-the-node-question)).

## Scope: what this does NOT touch

Every optimizing consumer assumes the road upgrade is a **per-edge constant knowable before the road
set is known** — `road_conductance(params, lane_width(width), dist)` depends only on that edge's own
`dist`. That assumption is load-bearing twice: it gets `linearized_gain`/CELF down to one solve per
round, and it keeps `resistance_lp`'s objective LINEAR in its decision variables.

A route-dependent road term breaks both. Reformulating the LP is a research problem, not an
implementation task. So:

- **In scope:** `permeability.egress_power` and the conductance model it solves.
- **Out of scope:** `resistance_greedy` and `resistance_lp` keep optimizing the constant-gain model
  as an explicit first-order PROXY. This is a deliberate, documented approximation, not drift.
- **Required:** measure what the proxy costs (see A6). A silent proxy is the 2026-07-30 bug again;
  a measured one is a design decision.

Every method already optimizes a proxy (`clearance` optimizes depth, `euclidean_grid` optimizes
nothing). This makes that true of two more, in a smaller way, on purpose.

## The conductance model

Today a covered edge's road term is `g_road_per_m * width / d`, `d` the crow-flies centroid
distance. It becomes a **series resistance**, because a route crossing several road segments of
different widths has no single `width`:

    road_term(i,j) = 1 / ( r_leg(i) + R_path(p*, q*) + r_leg(j) )

    R_path(p,q) = min over routes of  sum_s  len_s / (g_road_per_m * w_s)
    r_leg(i)    = |c_i - p| / (g_road_per_m * w_at_p)

    (p*, q*) = argmin over p, q in N(R) of the whole bracket   -- a JOINT minimization

    edge conductance = max( footpath, road_term )              -- unchanged switch

Minimizing RESISTANCE rather than length is what makes a mixed-width route well defined: a short
narrow alley and a long wide street trade off correctly, and no arbitrary "which width" rule is
needed.

Three properties fall out rather than being imposed:

- **`L >= d` automatically.** The legs plus the path form a route from `c_i` to `c_j`, so the
  triangle inequality bounds it below by the straight line. No artificial floor. Road conductance
  therefore strictly falls relative to today, which is exactly what a 1.395 detour ratio predicts.
- **Disconnection needs no rule.** If `p` and `q` lie on road components that do not connect,
  `R_path = inf`, the road term is 0, and `max(footpath, road)` leaves the edge at footpath. The
  same conclusion the D3 investigation reached from the other direction — no gate, no strict rule.
- **The entry point is a minimization, not an assignment.** This is the distinction that makes the
  design safe; see [Monotonicity](#monotonicity-as-a-proof).

### The one arbitrary choice: `r_leg`'s rate

The walk from a centroid to the road is not on pavement, yet `r_leg` charges it at road rate. This
is deliberate: an edge is only COVERED when the road's own `buffer(width_m/2)` already intersects
the centroid-to-centroid segment, so the legs are sub-metre and the choice is immaterial. The honest
alternative — footpath conductance in series — adds a second model to reason about for no measurable
gain. **This must be stated in the `edge_conductances` docstring**, so a later reader sees a decision
rather than an oversight.

## The road graph

**Planarized, not raw.** The two conventions already coexist and disagree badly: `_road_net` (raw
`_rnd` endpoint keys) vs `_noded_graph` (`unary_union`), measured at **521 vs 35 components** on the
LP. Only planarized is defensible for a travel distance — two roads that cross must let a walker
turn at the crossing, and the raw graph leaves crossing roads disconnected unless they happen to
share an endpoint.

`street_first_ordered` and `road_drainage` stay on the raw graph. That is correct and must be
recorded so nobody later "fixes" it: prefix ordering is a truncation heuristic, the metric is what
gets scored, and they are allowed to differ.

**The documented sliver trap does not reach us.** The old spec warns that planarizing yields 0.0100 m
minimum segments, giving `g_road/L = 2000` against a median 3.2 — six orders of conditioning. That
trap is about putting road segments INTO the Laplacian as mesh edges. Here the graph is only used to
measure routes: a 1 cm sliver contributes 1 cm to a path and nothing to conditioning.

**The existing street stays OUT of the travel graph.** Including it would require inventing a
`width_m` the street does not carry, and covered edges join ADJACENT parcels, so a detour out to the
street and back is essentially never the shortest route. Cheap to verify (A5), cheaper than a new
parameter.

**Projections attach continuously.** A projection lands at fraction `t` along segment `(u,v)`.
Snapping to the nearest node would cost up to half a segment — `topology`'s median segment is 4.83 m,
against legs that are themselves sub-metre. With node-to-node resistances `D` precomputed:

    R(p,q) = min over a in {u,v}, b in {u',v'} of   off(p,a) + D[a,b] + off(q,b)
    and, when p and q share a segment,              |t_p - t_q| * len / (g_road_per_m * w)

Exact, and one node-level solve serves every projection pair.

### Cost, and an early exit that is exact

Per block this is a few hundred road nodes; full all-pairs is trivial. At region scale (11,006
parcels) node counts reach thousands and it recurs per prefix (~20 per curve), so all-pairs is too
slow.

The way out is exact, not approximate. Because the edge takes `max(footpath, road)`, the road term
only matters while it BEATS the footpath, so Dijkstra from each projection may stop the moment
accumulated resistance reaches `1/footpath_g(e)`. Beyond that the `max` discards the value whatever
it is. **The cutoff is the point where the answer stops affecting the result, not a tuned radius** —
so the computed function is identical to the exact one, which matters because an approximation here
would void the monotonicity proof.

Region-scale cost must still be MEASURED before committing (A7), with full all-pairs as the
small-graph fallback.

## Monotonicity, as a proof

Permeability is `1 - P(R)/P0` with `P0` fixed, so the claim reduces to `P(R') <= P(R)` for
`R subset R'`. Since `L(R') - L(R) = sum_e (c'_e - c_e)(e_i - e_j)(e_i - e_j)^T` and each outer
product is PSD, `L(R') >= L(R)` in the Loewner order, hence `L(R')^-1 <= L(R)^-1` and
`b^T L(R')^-1 b <= b^T L(R)^-1 b`. **The entire burden is that no edge conductance ever falls.**

1. **Nodes and edges are fixed.** Nodes are parcel centroids, edges are parcel adjacency — functions
   of parcel geometry alone, untouched by roads. This is why (A) is safe where the road-first mesh
   was not: that design ADDED road nodes and MOVED access edges, so the Laplacians being compared
   were not even the same size.
2. **Coverage and routing only improve.** Coverage is monotone trivially. Routing needs one
   non-obvious step: `G(R)` is not literally a subgraph of `G(R')`, because a new road can SPLIT an
   existing segment at a new crossing. But splitting a resistor `rho` into `rho_1 + rho_2 = rho`
   preserves every path's resistance, so `G(R)` embeds isometrically into `G(R')` and the minimum
   can only fall. **Planarization refines; it does not reroute.**
3. **The entry point is a minimization over a growing set.** As `R` grows, `N(R)` grows AND every
   `R_R(p,q)` falls, so the joint minimum is non-increasing and `road_term` non-decreasing.

Then `max(footpath, road)` is a max of a constant and a non-decreasing function, and the ground
shunts depend only on street geometry. `P(R') <= P(R)`. QED

### Why this is not the failure that killed three attempts

`3a8dd25 fix: network_efficiency monotone via fixed entry mapping` records values FALLING (~9%
drops) because a nearest-road access edge MOVED when roads were added, breaking Rayleigh's
nested-edge-set requirement. Freezing entries fixed the efficiency form but not the resistance form
(memory `commute-ratio-monotonicity-fundamental`: every fully monotone variant became gameable,
multi-frontage worst).

The mechanism sounds identical and is categorically different. There, the moving entry changed the
EDGE SET. Here the edge set is parcel adjacency and is fixed; the entry point is an internal variable
of a scalar formula, and it moves in the safe direction by construction.

### The proof is exact; the implementation will not be

`unary_union` and `set_precision` move vertices on a ~1 cm grid, so step 2's isometric embedding is
approximate in practice, and floating-point Dijkstra can make a strictly nested pair come out
infinitesimally wrong. Monotonicity therefore gets a TEST on real blocks (A4), not just this
argument.

## Code structure

Verified duplication of the conductance model, all of it the drift that produced the 2026-07-30 bug:

- `methods/resistance_greedy.py:95` `_mesh` — rebuilds adjacency, dists, `_footpath_conductance`,
  `road_conductance`, `max`. Its own docstring says it mirrors `egress_power` "because the scorer
  below is only valid if it differentiates the SAME Laplacian the metric solves".
- `width_solver.py:151-152` — the same `_footpath_conductance` + `road_conductance` pair inline.

Split the road-independent part from the road-dependent part:

    footpath_mesh(block, params, adj=None, radii=None) -> Mesh
        nodes, edges, dists, footpath conductances, ground mask, segments.
        A function of parcel geometry ALONE -- freezable across a whole prefix sweep.

    road_terms(mesh, roads, params) -> (fwd, bwd)
        route-based road conductances, per this spec.

`egress_power` composes them. `resistance_greedy._mesh` and `width_solver`'s inline copy are DELETED
and call `footpath_mesh` (they keep their own constant-gain road term, per Scope).

**Plain functions, not a Protocol.** The CLAUDE.md preference for Protocol + injection applies to a
live choice; there is one mesh model here, and injecting a Strategy nobody selects is the
speculative reading of that rule rather than the intended one. Likewise no old/new toggle: migrate
and delete, comparisons live in `scratchpad/`.

## Acceptance

Mechanical gate to build. External validation against as-built OSM footpaths is a SEPARATE spec and
is deliberately not a gate here. Method ranking/discrimination is explicitly NOT a criterion — the
old spec's S1 was circular for exactly that reason.

- **A1 (D1)** A zigzag and a straight road covering an identical edge set must no longer score
  identically. Today: bit-identical up to 3.07x detour.
- **A2 (D2)** Road conductance falls by what the measured detour distribution predicts:
  median 1.395 -> factor 0.717.
- **A3** A disconnected road component falls back to footpath, via `max` and with no special rule.
- **A4 (M)** Monotonicity holds on real blocks under incremental road addition. FAULT-INJECTED:
  revert one `max` to a bare assignment and confirm the test fails.
- **A5** The street's exclusion from the travel graph is justified by measurement, not assertion:
  recompute `road_term` on >= 10 real blocks with the street added to the travel graph at
  `DEFAULT_ROAD_WIDTH_M` and report the share of covered edges whose route changes at all, plus the
  resulting permeability delta. Exclusion stands if the permeability delta is below 1e-3 absolute on
  every block; otherwise the street goes in and its width becomes an explicit parameter.
- **A6** The `resistance_greedy` proxy gap is quantified: on >= 10 real blocks, compare permeability
  reached at matched displacement by the shipped constant-gain greedy against a reference greedy
  whose per-round gains are recomputed route-aware. Report the median and worst-case shortfall in
  permeability. This is a recorded number, not a pass/fail gate — it sizes the deferred
  method-alignment spec.
- **A7** The bounded Dijkstra is **bit-identical** to full all-pairs on every block tested (required
  — the "Cost" section makes the monotonicity proof depend on the early exit being exact, so this is
  pass/fail, not a tolerance). Region-scale cost is measured on the 11,006-parcel `multiblock_depth`
  region and must not exceed 2x today's per-solve wall clock; if it does, the fallback is to cache
  node-to-node resistances across the prefixes of one curve, which are nested by construction.

## Blast radius

Every published permeability number FALLS, because road conductance strictly falls. `P0` has no
roads and is **unchanged**, so `permeability = 1 - P(R)/P0` moves only through the numerator — the
old spec's C3 objection ("P0 is highly sensitive to the parameters section 4 mandates re-deriving")
does not apply.

**The first measurement after the core lands, gating everything downstream:** Lens B truncates at
`P* = 0.60`, and methods that currently reach it may stop. `prefix_to_permeability` then returns
`(all roads, False)` and the lens quietly degrades. If that happens broadly, `P*` must be re-chosen
and the published comparison changes shape. Cheap to measure now, expensive to discover after
regenerating every example.

What `g_walk` should be is a question for data, not assumption: it encodes the footpath/road
BALANCE, and that balance shifts when road conductance drops ~28%. Re-derive only if measurement
says so.

## Testing

- Unit, on hand-built geometry where the answer is arithmetic: straight vs zigzag of known length;
  a narrow-then-wide route giving the correct SERIES resistance; disconnected components.
- Property: monotonicity on real blocks (A4), fault-injected.
- Equivalence: bounded Dijkstra vs full all-pairs, bit-identical (A7).
- Invariant: `permeability` and `resistance_greedy` build the same mesh. Framed as a LIVE invariant,
  not an old-vs-new comparison, because that drift has already bitten once and the test should
  outlive this migration.

## Deferred: the node question

Moving nodes onto building points, with Voronoi corners and gates as skeleton nodes, remains
attractive on first principles — a Voronoi cell's centroid is not even its own generator, so today's
node is not where the building is. But the gate measurement shows crow-flies between generators is
already right for footpath edges, so there is **no measured defect** driving it, and it would force
re-deriving `_footpath_conductance`'s fair-normalization.

Recorded so the reasoning is not redone:

- **Corners alone are not the continuum.** Cell corners are clearance MAXIMA; the osculation point on
  a shared wall is the clearance MINIMUM. Connecting only to corners forces neighbour trips out to a
  chamber and back: median 1.069, **p90 1.618, p99 2.619** against the gate route. That trades D2's
  40% understatement for a comparable overstatement. Any such design needs corners AND gates.
- **The lobe problem is a phantom.** Measured over 20 blocks / 2,618 parcels
  (`scratchpad/spectral/lobe_census.py`): parcels are exactly 1:1 with building points, ZERO contain
  no point, zero contain more than one, zero are MultiPolygons. `parcel_radii`'s radius-0 branch is
  defensive, not live. Concavity is negligible too: 0.4% of cells exceed 5% concave, none exceed 20%,
  worst area/hull = 0.809.
- **Per-cell convex hulls would break the tessellation** — hulls of adjacent clipped cells overlap
  and spill outside the block, and parcels are a PARTITION that `parcel_adjacency` and `displacement`
  both depend on.
- **Boundary pseudo-sites want MIRRORING, not corners.** Reflecting each building across each
  boundary edge makes the bisector coincide with the boundary, so cells terminate with no clipping.
  Corner sites alone do not achieve this. Given the census, it solves a problem we do not have.
- **Segments as tessellation sites is real** — the generalized Voronoi diagram of points and
  segments, where point-vs-segment bisectors are parabolic arcs (CGAL `Segment_Delaunay_graph_2`,
  Held's VRONI). Neither shapely nor scipy does it; dense point sampling along a segment is the
  standard approximation.
- **But PROPOSED roads must never participate in the tessellation.** The mesh would then change with
  the road set, `P(R)` and `P(0)` would be computed on different graphs, and Rayleigh would give
  nothing — the moving-edge-set failure arriving by a new door. The EXISTING street could
  participate (it is fixed), which would make cells terminate naturally at the street and grounding
  geometric rather than a `distance <= STREET_TOL` test; the cost is a sampling-density parameter
  and that parcel areas move, so `displacement` moves with them.
