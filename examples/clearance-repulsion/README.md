# Clearance reblocker: one knob, one region, five repulsions

`ClearanceReblocker` grows each new road as a least-cost path from the deepest remaining parcel
to the existing street network on a pluggable routing substrate (`chord_diag` by default — the
parcel-boundary graph plus all within-cell diagonals). The cost field blends a uniform term with
a repulsion-from-buildings term:

```
edge_weight = length * [(1 - t) + t / clearance],   clearance = distance to nearest building
```

The user-facing knob is a logit `repulsion` (`s`); internally `t = sigmoid(s) ∈ (0, 1)`. As
`s → -∞`, `t → 0` and the cost field is uniform, so the least-cost path is the **straight line**
— aspirational, maximally direct, and indifferent to what it crosses. As `s → +∞`, `t → 1` and
the path is pulled onto the **max-clearance ridges** — the Voronoi edges equidistant from the two
nearest buildings — i.e. it follows the gaps that are actually there to build in. `s = 0` is the
balanced default. This example reblocks the same region at five points on that spectrum
(`s = -6, -2, 0, +2, +6`) so the trade is visible directly in the road geometry.

Like [`capetown-flagship`](../capetown-flagship/), this reproduces from **`capetown_full`** (the
full metro, auto-downloaded to `~/.cache/reblock` on first use), not the committed 301-block
sample — but everything here runs on one small, auto-detected region instead of a full
screen-and-grow deep-dive.

## Auto-detecting the region

The generator never hand-picks a `block_id`. It:

1. **Screens the metro.** `DenseCompactScreen(max_depth_min=6.0)` flags blocks that are dense
   (≥ 30 buildings/ha), have mean access-depth ≥ 1.3, *and* have at least one parcel ≥ 6 parcels
   deep — deep informal fabric, not just any dense block. That's **191 of 83192** blocks in the
   metro, ranked deepest-first. `screen.select` is memoized (a `derive()` keyed on the source
   content hash + gate params), so this is instant on rerun.
2. **Picks a tractable seed.** Cape Town's deepest blocks are single 1000–3000-building informal
   giants — too large to reblock as a five-panel gallery entry. So the generator walks the ranked
   list deepest-first and takes the first block whose own `building_count` (kblock metadata) falls
   in a small window (`SEED_MIN=40, SEED_MAX=90`) — deep, but small enough to grow into a legible
   region.
3. **Grows a small multi-block neighborhood.** `DenseClusterRegionBuilder(max_buildings=100)`
   adds the seed's densest touch-adjacent neighbor(s) until the cluster's building-count budget is
   reached.
4. **Builds the region.** The member blocks are re-read through `KblockSource(...).region()` and
   merged with `region_block` into a single `Block` (parcels + buildings + Voronoi geometry).

On the run that produced the committed PNGs, this auto-detection landed on:

```
auto-detected region: seed=ZAF.9.3.1_1_23732  members=['ZAF.9.3.1_1_23732', 'ZAF.9.3.1_1_23733']
region: 250 parcels, 250 buildings
```

a two-block neighborhood — the dark cluster near the middle of `before.png` is where access depth
reaches 6 parcels from any street.

## Reproduce

```bash
PYTHONPATH=. pixi run python examples/clearance-repulsion/generate.py
```

(`PYTHONPATH=.` is needed because `reblock.data.provision` imports the repo-root `scripts/`
package, which is outside the package's own `sys.path` entry when the script is run directly —
the same reason `scripts/bench_cache.py` documents the same prefix.) First run screens the whole
metro and builds the region's Voronoi tessellation from scratch; both are memoized, so a warm
rerun (as here, caches already built by earlier development in this repo) completes in well under
a minute end-to-end for one small region.

## The five panels

| before | s = -6 (straight) | s = -2 | s = 0 (default) | s = +2 | s = +6 (Voronoi) |
|---|---|---|---|---|---|
| ![before](before.png) | ![s=-6](after_s-6.png) | ![s=-2](after_s-2.png) | ![s=0](after_s0.png) | ![s=+2](after_s+2.png) | ![s=+6](after_s+6.png) |

## Metrics (actual run output)

The routing substrate is `chord_diag` (the default winner from the substrate sweep) with 3-point
edge-cost sampling (both endpoints plus the midpoint), not the older `res=1.5` search grid.

| repulsion | roads | length_m | displaced | max_depth |
|---:|---:|---:|---:|---:|
| -6 | 19 | 348 | 67 | 2 |
| -2 | 18 | 337 | 64 | 2 |
| 0 | 18 | 329 | 62 | 2 |
| +2 | 17 | 344 | 58 | 2 |
| +6 | 18 | 381 | 64 | 2 |

(`roads`: road segments added. `length_m`: total road length. `displaced`: buildings within 3 m of
a new road, from `displacement_count`. `max_depth`: worst remaining access depth after reblocking,
from `parcel_access_layers`.)

## Reading the sweep

**Coverage is held constant.** Every panel reaches the same `depth_target=2` — the method keeps
adding least-cost roads from the deepest remaining parcel until no parcel is more than 2 parcels
from a street, regardless of `repulsion`. So the panels aren't a coverage comparison; they isolate
what the knob actually changes: *how* each road gets there.

**Displacement falls as repulsion rises, up to a point.** From `s=-6` to `s=+2`, `displaced` falls
monotonically (67 -> 64 -> 62 -> 58) as roads are pulled toward the max-clearance Voronoi ridges and
cross fewer building footprints. But at the most repulsive setting, `s=+6`, `displaced` jumps back
up to 64: hugging the gap network on `chord_diag`'s sparser, straighter road candidates can route a
path past one cluster of buildings only by threading closer to another, so the trend is not
monotonic across the full range — the extremes aren't guaranteed to be the best or worst case on
every substrate.

**Length and road count track the same non-monotonic shape.** `length_m` falls from `s=-6` to
`s=0` (348 -> 337 -> 329) then rises through `s=+2` and `s=+6` (344 -> 381); `roads` falls from 19
at `s=-6` to 17 at `s=+2`, then ticks back up to 18 at `s=+6`. The turning points don't perfectly
coincide -- `length_m` bottoms out at `s=0`, while `roads` and `displaced` bottom one step later at
`s=+2` -- but all three share the same reversal: past a point, following the clearance ridges more
aggressively costs extra route length and an extra road segment without buying further displacement
reduction on this region's `chord_diag` graph.

In short: `repulsion` doesn't change *whether* the block gets reblocked to depth 2, it changes *how
much building you cut through to get there* versus *how much extra road you lay to avoid it* — and
on the sparser `chord_diag` substrate that trade isn't perfectly monotone at the repulsion-seeking
extreme, unlike the old dense-grid substrate where it was.
