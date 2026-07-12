# Clearance Reblocker — Design

**Status:** draft for review · **Date:** 2026-07-12 · **Working name:** `clearance` (confirm below)

**Goal:** A fast, sparse reblocker that adds a few long roads and exposes ONE physical knob spanning
**aspirational straight roads** (best directness, crosses parcels) → **buildable Voronoi-following
roads** (low displacement, follows the gaps). It generalizes the validated straight-line reblocker
(= the knob's straight limit) and gives the same directness/displacement frontier topology walks,
but in seconds instead of minutes. Basis: the spike findings in
`docs/superpowers/notes/2026-07-12-straight-line-reblocker-findings.md` (all limits + tradeoffs
validated on real Cape Town blocks).

## The idea (validated)
Each road is a **least-cost path** from the deepest parcel to the current road+street network, on a
grid cost field that **repels from building points**:
```
edge_weight = length · [ (1−t) + t / clearance ]      clearance(x) = dist(x, nearest building) (+ε)
```
- `t = 0` → uniform cost → the **straight line** (aspirational; best directness).
- `t → 1` → path hugs max-clearance ridges = the **Voronoi edges** (equidistant from the two nearest
  building points) = the buildable gaps (topology/dijkstra's frontage-following).

**User-facing knob is the logit** `s = log(t/(1−t)) ∈ (−∞, ∞)`, internally `t = sigmoid(s) ∈ (0,1)`:
symmetric, linear in the log of the repulsion/rigidity *ratio*, `s=0` = balanced, **unbounded input
but numerically bounded** (t never hits 0 or 1, weight always finite — no instability, no ∞ in the
math). Default `s = 0`.

## Algorithm (greedy + incremental depth)
1. Build a grid over the block boundary (resolution `res`); 8-connected; edge weights from the cost
   field above. Clearance = KDTree distance from each grid node to the nearest building point.
2. Network nodes = grid nodes within `~res` of the street; maintain incremental **access-depth**
   (`parcel_access_layers` once, then relax a multi-source BFS from each new road's touched parcels
   — a road only lowers depth). Grid `res·1.5` street seeding.
3. Per step: pick the **deepest** parcel (argmax depth, ties by `parcel_id` — deterministic);
   multi-source Dijkstra from the network nodes; trace the least-cost path to the deepest parcel's
   nearest grid node; the road = `parcel rep → grid path → nearest actual street point` (the last
   two connect it to the parcel and to the street so it registers). Add it; relax depth; the road's
   grid nodes join the network. Repeat until `max depth ≤ depth_target` (or a road cap).

**Validated (real blocks):** 103-parcel: 7–9 roads, 0.1s, wins directness (0.056 vs dijkstra 0.009),
knob trades displacement (28→21) for directness. 2017-parcel: 162 roads, **6.3s** (vs topology not
finishing 347 in 10 min), sparse (162 long paths vs dijkstra's 3109 segs), wins directness (0.037 vs
0.004). Tradeoffs, inherent + expected: loses **resistance** (tree, no loops) and **displacement**
(crosses parcels; the knob and finer `res` reduce it).

## Weighted points (built, opt-in)
`clearance(x) = min_i ( dist(x, bᵢ) − rᵢ )`, `rᵢ = √(areaᵢ/π)` — treat each footprint as a disk, so
bigger buildings expand their exclusion (an additively-weighted / Apollonius clearance, the footprint
generalization of the Voronoi diagram). One KDTree query minus a radius; no efficiency hit. Needs
per-building areas (points today → `rᵢ=0`, the plain case); wire the column through when footprints
land.

## Interface
```python
@dataclass
class ClearanceReblocker:
    repulsion: float = 0.0        # s = logit(t); t = sigmoid(s). -inf straight .. +inf Voronoi
    depth_target: int = 2         # stop when every parcel is within this access-depth
    res: float = 1.5              # grid resolution (m); finer = truer straight/gap limits, slower
    max_roads: int = 400          # safety cap
    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal: ...
    @property
    def identity(self) -> object: ...   # (class, repulsion, depth_target, res, max_roads)
```
- `src/reblock/methods/clearance.py`; `conf/method/clearance.yaml`; add to `compare_config.yaml`
  `all_methods` so it grades alongside dijkstra/arterial/topology on all four lenses + renders.
- Follows the `Method` protocol; `propose` is side-effect-free (no global RNG use).

## Correctness strategy
- **Deterministic:** argmax-depth ties broken by `parcel_id`; the grid + KDTree + csgraph Dijkstra
  are deterministic; the road geometry is a pure function of the inputs. (The spike's ~6% big-block
  tie-divergence came from undeterministic nearest-ties — pin them.) Test: two `propose` calls →
  WKT-identical roads.
- **Terminates + achieves target:** test that `parcel_access_layers(block, roads).max() ≤
  depth_target` on real blocks; and that `max_roads` is a real backstop, not normally hit.
- **Incremental depth == full recompute:** the relax must match `parcel_access_layers` from scratch
  (the road only lowers depth). Test on a small block: byte-identical roads to a naive-recompute
  reference; assert the full recompute on the output hits the target.
- **Knob limits:** `s → −∞` roads are ~straight (low circuity vs the field); `s → +∞` roads track the
  gaps (lower displacement). Test the monotone displacement↓ / directness↓ trend across `s`.
- **Weighted-ready:** `rᵢ=0` path is the plain clearance; a test with a large synthetic weight pushes
  a path farther from that building.
- `mypy --strict`, ruff, `pixi run check` green.

## Out of scope (follow-ups, agreed)
- **Sequencer method** (`SequenceReblocker`): a higher-order method taking a list of `(method, budget)`
  configs so any method can follow any other — the home for a **resistance-marginal loop-closer**
  (add the k cycle-closing edges with the largest effective-resistance drop, from the earlier
  resistance rank-1 marginal) to give the tree redundancy. Separate spec.
- **Sparsified global forest** (one Dijkstra, prune to trunk edges) — a faster non-myopic mode; not now.
- **Weighted footprints** end-to-end (needs the footprint-area column in the source).
- Grid alternatives (parcel-adjacency substrate) if `res`-vs-speed on very large regions needs it.

## Decision to confirm
- **Name.** `clearance` (mechanism: clearance-from-buildings cost field). Alternatives: `flow`,
  `repel`, `weave`, `leastcost`. Pick one.
- **Default `res`** (1.5 m) and **`depth_target`** (2) — reasonable from the spikes; adjustable.
