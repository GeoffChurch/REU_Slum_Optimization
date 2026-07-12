# Clearance Reblocker — Design

**Status:** approved (name + defaults confirmed 2026-07-12) · **Date:** 2026-07-12 · **Name:** `clearance`

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

## Examples gallery — the repulsion knob on a real region (true end-to-end)
A committed gallery `examples/clearance-repulsion/` that shows ONE auto-detected deep Cape Town
region reblocked at **five repulsion values** — the two extremes, two moderates, and the balanced
default — so the knob's effect (straight aspirational chords → gap-weaving Voronoi-following roads)
is visible on real fabric. Mirrors `examples/capetown-flagship/` (reproduces from `capetown_full`,
commits the PNGs + a README; the underlying data is downloaded on demand, never committed).

- **Repulsion sweep:** `repulsion ∈ {−6, −2, 0, +2, +6}` (logit `s`; `t = sigmoid(s)` = 0.002, 0.12,
  0.5, 0.88, 0.998). `−6` ≈ straight (aspirational), `0` = balanced default, `+6` ≈ Voronoi
  (buildable). Depth target fixed at the default `2`, so coverage is held constant across panels —
  what changes panel-to-panel is road **geometry** and **displacement**, not road count. That IS the
  story: same reachability, different directness↔displacement point on the knob.
- **Auto-detected region (no hand-picked block_id):** run `DenseCompactScreen` on `capetown_full`
  (deepest-first ranked, memoized → instant on rerun); pick the **deepest seed whose own
  `building_count` is in a tractable window** (so growth genuinely pulls in neighbors and the region
  stays legible — the deepest blocks alone are 1000–3000-building giants); grow it with
  `DenseClusterRegionBuilder(max_buildings=…)` into a contiguous multi-block neighborhood sized so
  the road count per panel stays modest (a bounded few-hundred-parcel region → ~a dozen roads, not
  the 162 a 2017-parcel block needs). Build the region via `region.region_block`. The seed window
  and region budget are constants in the generator, chosen (and printed) so the gallery is legible.
- **Rendering:** reuse `render.render_before` / `render_after` (access-depth heatmap + roads) for a
  `before` panel and one `after` panel per repulsion; lay the five side-by-side in the README with a
  metrics table (repulsion → roads, length m, buildings displaced, directness AUC, max depth after).
  No `render.py` changes — the existing helpers suffice.
- **Reproducible generator:** `examples/clearance-repulsion/generate.py`, runnable with
  `pixi run python examples/clearance-repulsion/generate.py`; the README documents the command and
  the auto-detected region it lands on. This is a genuine end-to-end (real screen on the full metro
  → real region growth → the new Method → real renders), just orchestrated in a script because a
  five-value knob sweep with auto-seed-selection is not a single `compare` invocation.

## Decisions (confirmed)
- **Name:** `clearance` (mechanism: the clearance-from-buildings cost field).
- **Defaults:** `res = 1.5 m`, `depth_target = 2`, `repulsion = 0.0` (balanced), `max_roads = 400`.
