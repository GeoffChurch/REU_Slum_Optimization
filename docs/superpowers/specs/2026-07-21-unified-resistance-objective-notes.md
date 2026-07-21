# Design note: a unified grounded-effective-resistance objective for reblocking

**Status: exploratory research direction — NOT approved for implementation.** Captured 2026-07-21
from a session exploring how to fuse clearance (external connectivity / access) and arterial
(internal connectivity / circulation) into one method. The empirical work behind it lives in the
session's scratchpad (`compare_fusion*.py`, `fusion_frontier.png`) and in the memories
[[loop-candidate-generation-study]] and [[clearance-loops-tournament]].

## Motivation

Today we have three roughly-frontier-equivalent ways to get both external and internal connectivity:
- **clearance + LoopClosureRefiner** (shipped): clearance drains external, loops add redundancy. Fast
  (~25 s on an 11k-parcel region), simple, region-scalable. ρ ceiling ~0.47 (local loops).
- **arterial with `directness` objective**: reaches a higher ρ ceiling (~0.6) because through-corridors
  are global shortcuts that also close big loops. Heavier.
- **fused objective** (access + directness in one arterial greedy): self-schedules access→circulation;
  on a matched-length frontier it ~ties arterial-directness, marginally more road-efficient.

Frontier finding (152-parcel block, aspirational, length-budgeted): all three land on ~one
(external, ρ) frontier; external pins at the block max (~0.77) and ρ climbs with road. arterial/fused
get a modestly higher ρ-per-road and ceiling; clearance+loops is far cheaper and simpler. **No fusion
Pareto-dominates clearance+loops enough to justify a rewrite** on the metric alone.

## Scalability correction (important)

An earlier claim that arterial is O(n²) / region-intractable was WRONG. Directness/efficiency is
**K-sampled** (`network_efficiency(k=40)` → `_sampled_efficiency_core`; `src_euclid` is (K,N)): a score
is O(K·(E log V)), **linear in the graph** with K=40 constant. `buildable` mode has an incremental
scorer (`_StepContext.score_candidate`) that updates only touched entries, and a **lazy/CELF path
exists** (`GreedyArterialReblocker(lazy=True)` → `arterial_lazy`, ~80× at regional budgets). The slow
250 s/block probe was the worst config (aspirational + exact non-lazy). So arterial/fused are heavier
than clearance+loops but NOT intractable.

## The unifying idea: grounded effective resistance

Ground the road+street network at the existing street and measure each parcel's **effective
resistance to ground**. This single quantity captures BOTH goals:
- Poor access ⇒ electrically far from the street ⇒ high resistance (a smooth generalization of access
  depth = EXTERNAL connectivity).
- A parcel on a spur has higher resistance than one on a loop ⇒ adding redundancy lowers resistance
  (this IS `commute_ratio`'s basis = INTERNAL connectivity).

So `minimize Σ_parcels grounded_resistance` promotes access AND redundancy in ONE objective — external
and internal stop being two objectives joined by an exchange-rate weight. Displacement stays a separate
COST (budget constraint or penalty), not part of connectivity.

## Two ways to optimize it

**(A) Convex program + rounding (elegant, global).** Total effective resistance is CONVEX in edge
conductances (Ghosh–Boyd–Saberi 2008, "Minimizing effective resistance of a graph"; R_tot = n·tr(L†)).
Over the FINITE candidate-connector set (the `query_pairs` connectors + arterial chords), pick
fractional conductances to minimize grounded resistance s.t. Σlength ≤ budget — a convex/SDP solve —
then **round** to a discrete buildable road set. The convex part is polynomial; the ROUNDING is the
hard part, because a "road" is a combinatorial frontage PATH, not a free edge. This is also the
survivable-network-design (2-edge-connected spanning subgraph) view: external = 1-edge-connected to the
street, internal = 2-edge-connected; Jain's iterative LP rounding gives a 2-approximation for
r(u,v)∈{1,2}.

**(B) Stochastic greedy (keeps the incremental architecture).** Resistance-reduction is submodular
(well-established), as are coverage/access and bridges-removed. So "Lazier than Lazy Greedy"
(Mirzasoleiman et al., arXiv:1409.7938) applies: sample ~ (N/k)·log(1/ε) candidates per step, giving
**(1−1/e−ε) at O(N log 1/ε) evaluations** instead of O(Nk) — the win grows with budget k, i.e. exactly
at regional road budgets. Composes with the existing CELF/lazy machinery. CAVEAT: raw `directness`
(trip efficiency) is NOT obviously submodular (shortcuts are complementary), so use RESISTANCE-REDUCTION
as the internal term to keep the guarantee.

## Verdict / if we build it

The most intellectually coherent redesign: **grounded effective resistance as the single objective**,
optimized by either (A) convex-relax + round or (B) stochastic greedy over the candidate-connector set.
It subsumes clearance, loops, and arterial-directness into one principled, region-scalable method whose
external/internal fall out of one convex quantity, with approximation guarantees.

BUT: it's a real research build (a grounded-resistance scorer; the convex solve + a buildable-path
rounding heuristic OR the stochastic-greedy loop), and the frontier data says the practical payoff over
shipped clearance+loops is MODEST. So this is the right long-term architecture, not a near-term win.
Recommended only if internal-connectivity quality becomes a priority worth a rewrite; otherwise
clearance+loops stands.
