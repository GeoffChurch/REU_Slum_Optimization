# Design note: a unified grounded-effective-resistance objective for reblocking

**Status: exploratory research direction — NOT approved for implementation.** Captured 2026-07-21
from a session exploring how to fuse clearance (external connectivity / access) and arterial
(internal connectivity / circulation) into one method. The empirical probes were in a session
scratchpad (`compare_fusion*.py`, `fusion_frontier.png`) that is **ephemeral and gone** — reproduce
via the "Orientation for a fresh session" section below. Related memories: the auto-memory files
`loop-candidate-generation-study`, `clearance-loops-tournament`, `clearance-arterial-fusion`.

## Motivation

Today we have three roughly-frontier-equivalent ways to get both external and internal connectivity:
- **clearance + LoopClosureRefiner** (shipped): clearance drains external, loops add redundancy. Fast
  (~25 s on an 11k-parcel region), simple, region-scalable. ρ ceiling ~0.47 (local loops).
- **arterial with `directness` objective**: reaches a higher ρ ceiling (~0.6) because through-corridors
  are global shortcuts that also close big loops. Heavier.
- **fused objective** (access + directness in one arterial greedy): self-schedules access→circulation;
  on a matched-length frontier it ~ties arterial-directness, marginally more road-efficient.

Frontier finding (block `ZAF.9.3.1_1_5544`, n=152, aspirational, length-budgeted) — the concrete
(external, ρ) targets a new method must beat:
- **clearance+loops**: (0.770, 0.259) @683 m → (0.770, 0.466) @982 m; ρ plateaus ~0.47–0.50.
- **arterial-directness**: (0.770, 0.528) @1215 m → (0.770, 0.606) @1862 m.
- **fused (w_int=2)**: (0.770, 0.539) @1173 m → (0.770, 0.625) @1857 m.

All three land on ~one (external, ρ) frontier; external pins at the block max (~0.77) and ρ climbs
with road. arterial/fused get a modestly higher ρ-per-road and a higher ρ ceiling (through-corridors
are global shortcuts that also close big loops); clearance+loops is far cheaper and simpler. On the
11k-parcel depth region (dt=3 base, sr=60, budget_frac 0.30) the shipped clearance+loops reaches ρ
0.503 / ext 0.954. **No fusion Pareto-dominates clearance+loops enough to justify a rewrite** on the
metric alone. A new objective must dominate these frontiers, not just match them.

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

## Orientation for a fresh session (code hooks, harness, first experiment)

**The scorer already exists.** `commute_ratio(block, roads)` in `src/reblock/budget.py` (~L844) IS a
grounded-effective-resistance scorer: it planarizes road∪street via `_noded_graph`, then does a
per-connected-component **dense grounded solve** and returns mean over parcels of `1 − R/R_geo`
(grounded resistance vs geodesic; a spanning tree → 0). The `access_benefit(block, None)` factory
(~L589) returns the external-connectivity function `f(roads)`. So you do NOT need to write a resistance
solver — generalize `commute_ratio`'s grounded solve into (a) a marginal-gain scorer for the greedy, or
(b) the objective/gradient for the convex solve. (`commute_ratio` is O(component³) dense — fine per
candidate on a block; for regions use its per-component structure or a sparse/iterative solve.)

**The candidate-connector set already exists.** `loop_candidates(base_roads, block, *,
search_radius_m, min_loop_len_m, snap_lam, max_candidates)` in
`src/reblock/methods/loop_closure.py` returns `list[(connector LineString, u, v)]` — the finite
buildable-connector set to optimize over. Arterial's `_candidate_chords` (through-roads + deep-parcel
spurs) in `src/reblock/methods/arterial.py` is the richer alternative. `greedy_close_loops` and the
bridge-tree engine (same file) are the reference greedy to swap the objective into.

**Rebuild the ground-truth harness** (~73 s; reads the full Cape Town buildings parquet). From repo
root, `PYTHONPATH=$(pwd) pixi run python`:
```python
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from reblock.pipeline import build_regions
from reblock.region import region_block
with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
    cfg = compose(config_name="compare_config", overrides=[
        "metric=depth", "data=capetown_full", "screen=dense_compact",
        "region_builder=dense_cluster", "region_builder.max_buildings=3000", "max_blocks=1"])
source, screen = instantiate(cfg.data), instantiate(cfg.screen)
rbuilder = instantiate(cfg.region_builder)
source.block_ids = None
region = build_regions(source, screen, rbuilder, None, 1)[0]   # 12 blocks / 11006 parcels
rb = region_block(region)                                       # the merged region Block
# pickle rb + region once, then iterate cheaply.
```
Clearance base: `propose(ClearanceReblocker(depth_target=3, max_roads=3000), rb).roads` → 310 roads,
13903 m, ext 0.951. Small member blocks for fast probes: `sorted(region, key=lambda b: len(b.parcels))`.

**First experiment (the cheap, guaranteed route (B)):** prototype the stochastic-greedy version —
greedily add the candidate connector with the best **resistance-reduction per metre** (score via a
marginal `commute_ratio` gain, or a direct grounded-resistance drop), sampling ~(N/k)·log(1/ε)
candidates per step. Trace its (external, ρ) vs road frontier on block `ZAF.9.3.1_1_5544` and the depth
region, and check whether it **dominates** the clearance+loops numbers above. If it does not dominate,
stop — clearance+loops stands. Only if it does, consider route (A)'s convex solve + buildable rounding.

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
