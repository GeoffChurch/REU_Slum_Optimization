# Cost-benefit framework — comparing reblockers at a budget — Design

**Status:** draft for review · **Date:** 2026-07-09

Sub-project 2 of the peel upgrade. Compares reblocking methods (`dijkstra`, `peel`,
`topology`) not just at their terminal full-build state — where they converge — but across
the **whole budget range**, where they differ. Each method's roads are added incrementally
(cheapest-value first) and access is scored at each budget, tracing a **cost-benefit curve**;
the area under it is a single efficiency score. Answers "which method delivers the most
access per meter of road" with a curve and a number, not a hand-wave.

Depends on sub-project 1 (`DijkstraReblocker`, shipped) — the methods it compares.

## The measurements (settled in brainstorming)

- **Benefit** = reduction in **Σ_buildings depth²** (q=2, severity-weighted). kblock parcels are
  one Voronoi cell per building, so this is `(depths² ).sum()` over the per-parcel access-depth
  series (`reblock.derive.access.parcel_access_layers`) — building-weighted for free. Reported as
  the **fraction removed** vs the no-roads baseline: `1 - burden(prefix)/burden(∅)` ∈ [0, 1], so
  it is unit-free and comparable/aggregatable across blocks of any depth.
- **Cost** = cumulative added road length / block area → **road density (m/ha)**
  (`roads.length.cumsum() / (block.boundary.area / 1e4)`). Unit-free across blocks.
- **Incremental order = drainage** (confirmed vs greedy: drainage is O(B·N) linear vs greedy's
  ~O(N²), for a near-identical frontier — see the design log). A budget prefix must stay
  **street-connected** (floating roads grant no access), and high-drainage roads are the trunks,
  so drainage-descending order is both connectivity-valid and near-optimal.
- **Curve** = benefit(fraction) vs cost(m/ha), sampled at **~20 budget points** (evenly spaced by
  cumulative road length). **AUC**, normalized to a common cost cap across the compared methods
  (each curve held at its terminal benefit beyond its own full-build), gives a **0–1 efficiency
  score** per method per block — higher = more access per meter.

## Architecture

Three layers, matching the redesign's "sweep + emit are outer combinators":

**1 — `reblock.budget` (pure curve machinery):**
```python
def access_burden(depths: pd.Series) -> float:        # (depths ** 2).sum()
def road_drainage(block, roads) -> list[int]           # per-segment parcel counts (uniform)
def cost_benefit_curve(block, roads, n_points=20) -> Curve   # drainage-order -> prefixes -> (cost, benefit)
def auc(curve, cost_cap) -> float                      # normalized 0-1 efficiency
```
- **`road_drainage` is uniform across methods** — it builds a graph from the method's road
  *segments* (whatever their geometry: dijkstra's boundary edges, peel's center-to-center links,
  topology's parcel edges), finds street-adjacent road-nodes, routes each parcel to the street
  through that graph, and counts parcels per segment. One function, applied identically to all
  three, so the comparison is apples-to-apples. (Dijkstra's native `drain` is the special case
  where the road graph is the boundary graph; the framework recomputes for consistency.)
- **`cost_benefit_curve`** orders roads by drainage descending, then at each of `n_points` budget
  levels takes the road prefix and scores `benefit = 1 - access_burden(parcel_access_layers(block,
  prefix)) / access_burden(parcel_access_layers(block, None))`. **Parcel adjacency is computed once
  per block and reused across all prefixes** (the one real optimization — else it's 20× redundant
  STRtree work); this needs `parcel_access_layers` to accept an optional precomputed adjacency (a
  small, backward-compatible signature addition).

**2 — `reblock.compare` (the Hydra edge / sweep):** a `@hydra.main` entrypoint that instantiates a
Source + Screen (to pick the block set) + a **list of methods**, sweeps `cost_benefit_curve` over
(block × method), and hands the results to the emitters. `max_blocks` bounds the sweep; **methods**
is a config list (default `[dijkstra, peel, topology]`). Reuses `derive`-cached `propose`, so each
method's per-block reblock is computed once then free.

**3 — Emitters:** `compare_report(results, out_dir)` writes (a) an **AUC table** — mean efficiency
per method across the swept blocks, with spread, and the headline winner; and (b) **overlaid
cost-benefit curve plots** for a handful of representative (deepest) blocks. Both into the run dir.

## Decisions (my calls — flag any in review)

- **Drainage order, ~20 sampled budget points** — confirmed; linear cost, near-optimal frontier.
- **Benefit = fraction of Σ_buildings depth² removed** (q=2); **cost = road density m/ha**; **AUC
  normalized to a common cost cap → 0–1 efficiency**.
- **Uniform `road_drainage`** recomputed per method (not reusing dijkstra's native `drain`), so all
  methods are ordered by the identical rule.
- **Topology is in the default compare list but the sweep is tens of blocks**, not hundreds —
  topology is minutes–hours/block (paid once via `derive` cache). For broad dijkstra-vs-peel
  comparison, drop topology from the list.
- **`parcel_access_layers` gains an optional `adj` param** (precomputed adjacency) — additive, no
  behavior change, so the curve's 20 evals share one adjacency build.

## Testing

- **`access_burden`**: `(depths²).sum()` on a known series.
- **`road_drainage`**: on a synthetic block, trunk segments (near street) carry more than leaves;
  a floating segment carries 0.
- **`cost_benefit_curve`**: monotonic non-decreasing benefit; benefit 0 at cost 0, ~1 at full build
  for a method that flattens the block; drainage-order beats a shuffled order (steeper early curve).
- **`auc`**: a dominating curve scores higher; identical curves score equal; a method reaching full
  benefit at lower cost beats one reaching it later.
- **Efficacy sanity**: on a real deep block, `dijkstra` and `topology` reach high benefit at lower
  road density than `peel` (peel's through-parcel roads are long) — a concrete cross-method result.
- **`compare` entrypoint**: composes methods + screen, writes the AUC table + curve PNGs; small
  block set, no network.

## Out of scope

Method internals (unchanged). A live web dashboard. Per-building (vs per-parcel) weighting for
non-Voronoi sources (parcels are one-per-building here). Optimizing topology itself (separate).
