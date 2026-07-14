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

1. **Screens the metro.** `DenseCompactScreen(max_depth_min=6.0)` flags blocks that clear the depth
   proxy `√(n·A)/P` (≥ 1.5), have mean access-depth ≥ 1.3, *and* have at least one parcel ≥ 6 parcels
   deep — deep informal fabric, not just any dense block. That's **395 of 83192** blocks in the
   metro, ranked deepest-first. `screen.select` is memoized (a `derive()` keyed on the source
   content hash + gate params), so this is instant on rerun.
2. **Picks a tractable seed.** Cape Town's deepest blocks are single 1000–3000-building informal
   giants — too large to reblock as a five-panel gallery entry. So the generator walks the ranked
   list deepest-first and takes the first block whose own `building_count` (kblock metadata) falls
   in a small window (`SEED_MIN=40, SEED_MAX=90`) — deep, but small enough to grow into a legible
   region.
3. **Grows a small multi-block neighborhood.** `DenseClusterRegionBuilder(max_buildings=100)`
   adds the seed's deepest touch-adjacent neighbor(s) — by that same depth proxy — until the
   cluster's building-count budget is reached.
4. **Builds the region.** The member blocks are re-read through `KblockSource(...).region()` and
   merged with `region_block` into a single `Block` (parcels + buildings + Voronoi geometry).

On the run that produced the committed PNGs, this auto-detection landed on:

```
auto-detected region: seed=ZAF.9.3.1_1_20205  members=['ZAF.9.3.1_1_20205', 'ZAF.9.3.1_1_20206']
region: 806 parcels, 806 buildings
```

a two-block neighborhood — the dark cluster near the middle of `before.jpg` is where access depth
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
| ![before](before.jpg) | ![s=-6](after_s-6.jpg) | ![s=-2](after_s-2.jpg) | ![s=0](after_s0.jpg) | ![s=+2](after_s+2.jpg) | ![s=+6](after_s+6.jpg) |

## Metrics (actual run output)

The routing substrate is `chord_diag` (the default winner from the substrate sweep) with 3-point
edge-cost sampling (both endpoints plus the midpoint), not the older `res=1.5` search grid.

| repulsion | roads | length_m | displaced | max_depth |
|---:|---:|---:|---:|---:|
| -6 | 69 | 1932 | 192 | 2 |
| -2 | 67 | 1905 | 192 | 2 |
| 0 | 69 | 1894 | 182 | 2 |
| +2 | 66 | 1915 | 155 | 2 |
| +6 | 66 | 2266 | 127 | 2 |

(`roads`: road segments added. `length_m`: total road length. `displaced`: buildings within 3 m of
a new road, from `displacement_count`. `max_depth`: worst remaining access depth after reblocking,
from `parcel_access_layers`.)

## Reading the sweep

**Coverage is held constant.** Every panel reaches the same `depth_target=2` — the method keeps
adding least-cost roads from the deepest remaining parcel until no parcel is more than 2 parcels
from a street, regardless of `repulsion`. So the panels aren't a coverage comparison; they isolate
what the knob actually changes: *how* each road gets there.

**Displacement falls monotonically as repulsion rises.** `displaced` drops steadily from **192** at
`s=-6` to **127** at `s=+6` (192 → 192 → 182 → 155 → 127) — a **34% reduction** in homes cut through.
As `t → 1` the cost field pulls each road onto the max-clearance Voronoi ridges (the gaps between
buildings), so it crosses fewer footprints. This is the knob's whole point: dial repulsion up to
spare homes.

**The cost is extra road at the high end.** `length_m` is U-shaped — it bottoms at the balanced
`s=0` (1894 m) and climbs to **2266 m** at `s=+6` (+20%), because hugging the clearance ridges
means longer detours around building clusters. `roads` stays roughly flat (66–69). So repulsion
buys lower displacement with more road: at `s=+6` you cut through 34% fewer homes but lay ~20% more
road than the balanced default.

In short: `repulsion` doesn't change *whether* the block reaches depth 2, it changes *how much
building you cut through* versus *how much extra road you lay to avoid it* — here a clean monotonic
trade, displacement down 192 → 127 for road up 1894 → 2266 m.
