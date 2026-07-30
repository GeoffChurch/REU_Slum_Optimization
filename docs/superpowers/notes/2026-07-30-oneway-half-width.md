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

## Before building

The circulation history applies here as it did to the all-pairs probe
(`notes/2026-07-30-egress-vs-circulation.md`): cycle density was already tried and RETIRED as
perverse and gameable, which is what led to `commute_ratio` and then permeability. An ingress+egress
directed metric is a circulation measure and inherits that history. Any spec must say what stops it
degenerating into "cycle count", and the honest test is whether it prefers a SMALL number of
well-placed loops over many tiny ones -- exactly the interior optimum above.

Cost note: the directed QP is per-block and convex, but it is a QP per evaluation rather than one
sparse solve, so the region-scale cost needs checking before this can sit inside a lens.
