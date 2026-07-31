# One-way streets at half-width corridors: does it reward loops?

Owner's idea: make streets one-way with half-width corridors, so a method that builds loops is
rewarded -- one-way alone gets you in but not out, so round trips need cycles. Optionally make ALL
streets one-way with some separation rule to stop two opposite one-ways being stacked on one
alignment.

## The crux: one-way alone does NOT reward loops

`permeability` is `b = ones(n)` -- all current flowing OUTWARD to the street. Orient every road
outward and you get a perfect out-tree: every parcel exits, no loops, top score. The loop reward
never appears.

Loops only become necessary when demand is **bidirectional**: score egress (every parcel -> street)
AND ingress (street -> every parcel) on the same directed network and sum them. On an undirected
network these are identical by reciprocity, which is why one has always sufficed. On a directed one
they diverge, and **an out-tree fails ingress completely.** That is the ambulance test.

So the design is *directed arcs + ingress-and-egress demand*. Either half alone is inert.

## The formalism survives

Effective resistance does not generalize to directed edges, but the underlying object does:
permeability is already a min-energy flow, and directionality is a sign constraint.

    min  sum_a  f_a^2 / g_a     s.t.  flow conservation,  f_a >= 0

Undirected edge = both arcs; one-way = one arc. Still a convex QP, so still optimizable, and the
Ghosh-Boyd-Saberi convexity the route-(A) LP leans on survives. `resistance_lp` extends most
naturally of anything shipped, since it already decides over PATHS and paths are directed.

## The stacking cheat is real, and worse than expected

`budget.displacement` measures against the UNION corridor. Two EXACTLY COINCIDENT half-width
one-ways union to a HALF-width corridor -- so a stacked pair buys full two-way function at half the
displacement of a real two-way street, while genuinely separated one-ways each pay their own
corridor. The incentive points backwards: coinciding is rewarded, separating punished.

The fix is not a penalty term but honest physics: **width determines directional capacity.** A 1.5 m
strip carries one direction; two directions need 3 m. A coincident pair either stays 1.5 m -- one
lane, not bidirectional -- or widens to 3 m and pays a two-way street's displacement.

## Robbins' theorem gives the width rule for free

A connected graph has a strongly-connected orientation **iff it is bridgeless**. So:

    bridge edge                     must stay two-way  -> FULL corridor (3.0 m)
    edge in a 2-edge-connected comp can be one-way     -> HALF corridor (1.5 m)

A tree is all bridges, hence doubled everywhere -- exactly the owner's TSP 2-approximation framing
(walk each edge in both directions). A cycle-rich network pays half width but needs more length to
close its cycles. **That also means the cost side needs no directed solver**: the whole economic
effect lands in displacement accounting.

## Probe result: the rule rewards loops substantially and re-ranks methods

`scratchpad/ot/oneway_probe.py`, 12 blocks, full road sets:

| method | cycle-edge fraction | displacement saving | rank today -> one-way |
|---|---|---|---|
| **greedy_arterial_repulsion** | **0.958** | **57.6%** | 2.0 -> 2.0 |
| clearance_looped | 0.423 | 9.5% | 4.0 -> **3.0** |
| clearance (tree) | 0.000 | 0% | 3.0 -> **4.0** |
| resistance_greedy (tree) | 0.000 | 0% | 5.0 -> 5.0 |
| **resistance_lp** | **0.000** | **0%** | 1.0 -> 1.0 |

Three things follow:

1. **The effect is large.** A fully cyclic network halves its displacement. This is not a rounding
   effect that a directed model would have to rescue.
2. **It re-ranks.** `clearance` and `clearance_looped` swap. Loop closure stops being marginal.
3. **Arterial becomes the structural favourite** -- obvious in hindsight, since it builds
   street-to-street through-routes, which are bridgeless by construction. A method that currently
   looks mediocre on lens B would look best.
4. **The LP builds a pure tree** (`loop_frac` 0.000), independently confirming
   `notes/2026-07-29-resistance-greedy.md`: loop candidates are generated in quantity and never
   selected, because access outbids redundancy per metre at every step. **Under a one-way regime the
   LP forfeits its cost-side advantage entirely.** It holds rank 1 here only because it is so
   road-frugal, and that column is not budget-matched.

### Probe caveats, stated because they bound the claim

- Scored on FULL road sets, not budget-matched, so the rank columns partly reflect road quantity.
  The budget-independent column is `saving` (within-method ratio), and that is what the argument
  rests on. Scoring a lens prefix instead would have been worse: `LoopClosureRefiner` appends
  connectors last, so the P* = 0.60 prefix contains 2 of 19 roads and no loops at all.
- A first version built the bridge graph from raw coordinates and found ZERO cycles everywhere.
  Roads meet streets mid-segment, so they share no vertex: measured 9 road nodes, 8 street nodes,
  0 shared, 3 components -- every road edge trivially a bridge. The graph must be planarized
  (`unary_union`) first. Same class of error as the `_noded_graph` connectivity bug.

## What the width rule CANNOT capture, and why the directed model is still needed

The width rule is a pure **cost** model: it is monotone in loop fraction, so more cycles is always
cheaper. It cannot produce an interior optimum.

The owner's argument for a non-trivial optimum is a **benefit-side** effect and needs the directed
flow model: in a one-way system you cannot reverse, so reaching a parcel may mean going all the way
around. A large loop serves more parcels per metre but imposes longer forced detours, raising
ingress+egress resistance; a small loop means short detours but many connector metres. That optimum
lives in the QP, not in the width accounting.

| | captures | interior optimum? |
|---|---|---|
| half-width on cycle edges | loops are cheaper to build | no -- monotone |
| directed ingress + egress flow | loops impose forced detours | **yes** |

So the full idea needs both halves. The probe validates the cost half decisively.

## Directed probe: the interior optimum did NOT appear, but the geometry preference REVERSES

`scratchpad/ot/directed_probe.py`. 200 parcels uniform on a disc, a one-way ring road at radius `r`,
a spoke from the ring to ground on the boundary; parcels walk radially to the ring. Both sides of
the predicted trade are present -- a small ring is cheap to ride but far from most parcels; a large
ring passes close to the many outer parcels but is long to go round when you cannot reverse.

| r | egress | ingress | **total (one-way)** | undirected |
|---|---|---|---|---|
| 0.15 | 41198 | 42068 | **83266 (best)** | 75782 |
| 0.35 | 41011 | 43040 | 84050 | 66586 |
| 0.55 | **40939 (best)** | 44127 | 85066 | 57623 |
| 0.75 | 41016 | 45364 | 86379 | 48957 |
| 0.95 | 41344 | 46851 | 88195 | **40793 (best)** |

**The interior optimum did not materialize in the scored total.** It is monotone: the smallest ring
wins. Egress ALONE does show a genuine interior minimum at r = 0.55, rising on both sides -- so the
mechanism the owner described is real and measurable -- but ingress is monotone increasing and
dominates the sum.

**The larger result is a REVERSAL of geometric preference.** Undirected prefers the LARGEST ring
(75782 -> 40793 as r grows); one-way prefers the SMALLEST (83266 -> 88195). Opposite directions.
Undirected rewards a big ring for passing close to the many outer parcels; one-way penalizes it
because a rider may traverse the whole circumference without reversing, and that cost outgrows the
proximity benefit. **So the directed model does not merely re-weight the existing metric -- it wants
a different shape of network.** That is a stronger argument for building it than the interior
optimum would have been.

### Caveats that bound this

- **One idealized fixture**: a single ring with one spoke. Real proposals have several loops and
  many connections to the street, where the forced-detour cost is shared out differently. The
  reversal should be re-tested on real road sets before it is relied on.
- Parcels attach only at their nearest bearing node, and the radial walk conductance `g_walk / gap`
  blows up when a parcel sits almost exactly on the ring. That flatters mid-radius rings and is
  probably why egress alone peaks near the middle.
- **Three solver attempts, two of which failed in ways that looked like findings**, recorded so they
  are not repeated: penalising only the conservation residual minimises constraint violation rather
  than energy; a stacked-identity penalty least-squares was ill-conditioned enough to return `inf`
  at scattered radii, which read as infeasible geometry rather than non-convergence -- and its few
  converged points suggested an "interior optimum at 0.35" that was pure solver noise. What works is
  an IRLS Laplacian solve: the per-edge cost is a convex asymmetric quadratic in the NET flow, so
  the problem is an ordinary Laplacian solve whose conductance depends on the sign of its own
  solution; iterate with damping to the fixed point.

## Attempt on real road sets: the shipped mesh CANNOT express one-way streets

Re-testing the reversal on real proposals failed, and the failure is the most useful result in this
note.

**The permeability mesh has no road network in it.** `egress_power` builds edges between adjacent
PARCEL CENTROIDS; a road is only a per-edge conductance modifier -- an adjacency edge whose
centroid-to-centroid segment intersects `roads.buffer(corridor_m)` gets `g_road` instead of
`g_walk`. There are no road nodes and no road edges. So:

- A road corridor covers many adjacency edges, including ones running PERPENDICULAR to the road
  (parcels facing each other across it). Orienting those has nothing to do with orienting traffic.
- There is no object in the mesh corresponding to "this street, in this direction".

`scratchpad/ot/directed_real.py` tried anyway -- dropping the reverse conductance of every
road-covered adjacency edge -- and produced a result that is diagnostic precisely because it is
backwards: `greedy_arterial_repulsion`, the most loop-rich method at 96% cycle edges, showed the
WORST one-way penalty (5.65x), while a pure tree showed the BEST (3.49x). Loops are supposed to be
what survives one-way. The numbers are not reported as findings.

**Consequence for the design: one-way cannot be layered onto the current metric.** It needs a mesh
in which road segments are first-class edges and parcels attach to them -- closer to a conventional
transport network model than to today's parcel-adjacency conductance field. That is a bigger change
than the width rule (which needs no solver at all) and should be scoped as such.

### Also: the budget trap, for the third time

The probe compared FULL road sets, so `egress_power` correctly reported the LP as worse (112 vs
clearance's 45 on one block) purely because the LP stops at its 10% displacement cap and builds less
road. The LP wins at MATCHED budget. This is the same error that invalidated the first Kirchhoff
probe and the first one-way width probe. **Any probe comparing methods must match a budget or report
only within-method ratios** -- there is no third option, and the failure mode is silent.

The solver itself was validated against the shipped metric before any of this was read: run
undirected it reproduces `egress_power` to a ratio of 1.0000 on every case tested. That is the same
known-answer-oracle discipline that caught the broken connectivity instrument, and it is what
localized the fault to the framing rather than the arithmetic.

## The width rule ALONE IS GAMEABLE -- retracting "the cheap half is validated"

Owner's objection, and it is correct: the half-width discount is granted because Robbins' theorem
says a bridgeless network COULD be oriented one-way -- but **the scoring never orients anything.**
Permeability stays undirected, so every discounted edge is still traversed both ways at full road
conductance. You pay a one-way price for two-way function.

And the exploit scales with loop size, which is the part that makes it fatal rather than merely
generous: a huge loop that would be tedious to traverse in one direction costs nothing in an
undirected score, because the forced detour -- the entire reason one-way is a real constraint -- is
never charged. Bigger loops mean more edges on a cycle, hence more discount, with no offsetting term
anywhere in the metric.

**This is already visible in this note's own probe.** `greedy_arterial_repulsion` scoring 96% cycle
edges and a 57.6% displacement saving is not evidence that it is one-way-functional; it is the
largest UNEARNED discount in the set. It is bridgeless because it builds street-to-street
through-routes, so nearly every edge qualifies for half width -- while the metric continues to let
all of them carry traffic in both directions. The earlier framing in this note, "arterial becomes the
structural favourite", is withdrawn: the correct reading is "arterial harvests the biggest free
lunch".

A loop-size sweep (`scratchpad/ot/width_gaming.py`, `LoopClosureRefiner.search_radius_m` from 30 m
to 480 m) was inconclusive rather than confirmatory: `loop_frac` stayed flat at ~0.49 and
permeability-per-displacement moved -1%, because the refiner saturates on `budget_frac` and
`min_bridges_per_m` long before the search radius binds. The dial does not move, so the sweep tests
nothing. The argument stands on its own construction regardless.

### What this means for the design

**The two halves are not separable.** The earlier claim in this note -- that the cost half is
"validated decisively" and ready to spec while only the benefit half needs the new mesh -- was
wrong. The width discount is a claim that the road can function one-way, and that claim is only
cashable if the scoring enforces one-wayness. Without directed scoring it is free money, and the
cheapest way to collect it is to build one enormous loop.

So the honest position on the whole idea:

| half | needs | status |
|---|---|---|
| half-width on cycle edges | displacement accounting only | **unsound alone** -- gameable by loop size |
| directed ingress + egress | a mesh with explicit road edges | not expressible on the current metric |

The cheap half is only meaningful as part of the expensive half. There is no incremental version.

## Before building

The circulation history applies here as it did to the all-pairs probe
(`notes/2026-07-30-egress-vs-circulation.md`): cycle density was already tried and RETIRED as
perverse and gameable, which is what led to `commute_ratio` and then permeability. An ingress+egress
directed metric is a circulation measure and inherits that history. Any spec must say what stops it
degenerating into "cycle count", and the honest test is whether it prefers a SMALL number of
well-placed loops over many tiny ones -- exactly the interior optimum above.

Cost note: the directed QP is per-block and convex, but it is a QP per evaluation rather than one
sparse solve, so the region-scale cost needs checking before this can sit inside a lens.


## REOPENED 2026-07-31: directed scoring works, and every method is at the tree end

The gating question -- does directed scoring separate our methods? -- was tested on the ROAD GRAPH
ONLY (`scratchpad/ot/directed_roadgraph.py`), so `permeability` is untouched, no monotonicity risk,
no recalibration, and the rejected road-first mesh is not needed. Budget-matched at D = 10%,
planarized, orientation chosen by the solver rather than by hand.

    penalty = (egress + ingress) / (2 * undirected)     1.0 = one-way is free (richly looped)

**Raw result looked null**: every method 5.4439-5.4846, a 0.7% spread, and `clearance_looped` beat
its own tree base on 12/20 blocks with median -0.0011. But `bridge_frac` varied 0.15-0.37, so the
topologies clearly differ while the penalty did not move -- suspicious enough to validate the
instrument before believing it.

**Known-answer test** (same node count, same edge lengths, ground at one end):

| topology | penalty |
|---|---|
| pure PATH -- every edge a bridge | **5.46** |
| pure CYCLE -- no bridges | **2.27** |

The measure separates them by 2.4x, so it sees topology fine. **Every method sits at the PATH value.**

### What that means

The flat result is not "directed scoring cannot discriminate". It is **every method we have builds a
network that behaves as a TREE under directed flow** -- `clearance_looped` and
`greedy_arterial_repulsion` included, despite bridge fractions of only 0.15-0.37. Their cycles are
decorative for circulation: local loops that do not provide an alternate route where flow needs one.

So there is a **2.4x headroom that nothing in the method set captures**, and one-way is not moot --
it is a target none of our methods hit. That is a stronger argument for the direction than the
original loop-reward framing, and it is the first evidence in this thread that came from a validated
instrument rather than an idealized fixture.

The model's known weakness cuts the safe way: each solve picks its own orientation independently, so
it is OPTIMISTIC about directionality. A single shared orientation -- the real physics -- would
penalize these networks more, not less.

### What this does and does not license

- It does NOT revive the road-first mesh. That was rejected on monotonicity, and this result needs
  none of it: the measure lives on the road graph, beside permeability rather than inside it.
- It does NOT revive the half-width discount alone, which is still gameable (an undirected score pays
  a one-way price for two-way function).
- It DOES make `penalty` a candidate DIAGNOSTIC column -- cheap, validated, and it separates
  something no shipped number does.
- The obvious next question is whether any method can be made to score near the cycle end at
  comparable displacement, or whether 5.47 is a property of dense informal fabric rather than of our
  methods. **Tested -- see below.**

## Is the headroom reachable? Partly, and it costs access (2026-07-31)

`scratchpad/ot/cycle_reachable.py`. Rather than ask a method for circulation, build the best
circulating network the block admits: seed with half the budget on the tree, then spend every
remaining metre ONLY on connectors that remove bridge length, best bridges-removed-per-displacement
first. 10 blocks, D = 10%.

| | method | forced circulation |
|---|---|---|
| directed penalty | 5.4754 | **4.8177** |
| bridge fraction | 0.1723 | **0.0189** |
| permeability | 0.6939 | **0.5806** |
| displacement used | 0.100 | 0.084 |

Lower penalty on 9/10 blocks, median -0.6631.

**Bridgelessness is achievable; the cycle end is not.** Bridge length drops from 17% to 1.9% -- the
network becomes essentially bridgeless, so Robbins-orientability is reachable in this fabric. But the
penalty moves only 5.48 -> 4.82, closing about **21% of the gap** to a pure cycle's 2.27.

So **bridgelessness is necessary but nowhere near sufficient**: circulation being POSSIBLE is not
circulation being CHEAP. A near-bridgeless real network still scores 4.82 because the way around is
long. The pure cycle is an idealization a block cannot reach while also serving interior parcels,
which requires branches -- and branches are what the penalty punishes.

Two details as important as the headline:

- **It costs 16% of permeability** (0.694 -> 0.581). Forcing circulation trades away access. One-way
  is therefore a **Pareto axis, not a free win**, and adopting it means deciding that circulation is
  worth access -- a values question, not a measurement one.
- **It stopped at 0.084 displacement without spending the 0.10 budget**, having run out of
  bridge-removing candidates worth taking. The ceiling is structural, not budgetary: more money does
  not buy more circulation here.

### Verdict

The 5.47 figure is **mostly the fabric, partly the methods**. There is real but modest headroom
(~21%), it is bought with access, and the idealized cycle end is unreachable in dense informal
fabric. That is enough to stop treating one-way as an unexplored opportunity: it is explored, its
size is measured, and its price is known.

What remains defensible: `penalty` as a cheap DIAGNOSTIC column (validated, and it separates
something no shipped number does). What is not supported: rebuilding the metric, the road-first mesh,
or the half-width discount, for a 21% move on an axis that costs 16% of the primary one.
