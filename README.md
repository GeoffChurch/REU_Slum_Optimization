# reblock

## Setup

```bash
git clone --recurse-submodules <repo-url>
# (or, if already cloned: git submodule update --init --recursive)

# Install pixi: https://pixi.sh/latest/#installation
pixi install
```

## Common tasks

```bash
pixi run test        # pytest + coverage
pixi run typecheck   # mypy --strict
pixi run lint        # ruff check
pixi run fmt         # ruff format
pixi run check       # lint + typecheck + test
```

## Generate before/after visuals for one block

Render a block's access-depth heatmaps (before, and after a road-building method)
by targeting it with `block_ids` — no need to process the whole region:

```bash
pixi run python -m reblock.run data=capetown method=dijkstra eval=kcomplexity "block_ids=[[ZAF.9.3.1_1_44882]]" render.enabled=true
```

Writes `ZAF.9.3.1_1_44882_before.png` and one `_<proposal>_after.png` into the Hydra
run dir (`outputs/<date>/<time>/`). `dijkstra` is the default method — a buildable
frontage-routed street network (~1 s/block); swap `method=peel` (fast through-parcel
sketch), `method=topology` (slow greedy optimizer), or `method=greedy_arterial` (the
navigability flagship — straight through-roads, best directness, minutes/block). Swap
`data=capetown` → `data=dji`,
or omit `block_ids` to process the first `max_blocks` blocks instead.

(Quote `"block_ids=[...]"` so the shell doesn't glob the brackets.)

## Detect → reblock → visualize (one command)

Screen a city for its dense/compact informal blocks, reblock the worst survivors, and
emit both the city flagged-map and per-block before/after heatmaps:

```bash
pixi run python -m reblock.run data=capetown_full screen=dense_compact screen.density_min=35 method=dijkstra eval=kcomplexity render.enabled=true flagged_map.enabled=true max_blocks=5
```

The first run downloads + caches the full Cape Town metro under `~/.cache/reblock`
(nothing committed); later runs are instant. `method=dijkstra` (the default) routes each
block's buildable street network in ~1 s, so the screen pass dominates the runtime and
`max_blocks=5` adds only seconds — swap `method=topology` for a slower, higher-quality
greedy optimizer (minutes per block) or `method=peel` for a fast through-parcel sketch.
Outputs land in the Hydra run dir (`outputs/<date>/<time>/`):
`flagged_map.png` (whole metro, flagged blocks in red over grey context),
`flagged_blocks.txt` (every flagged id, worst-access first), and `*_before.png` /
`*_<proposal>_after.png` for each reblocked block.

Tune the gates: `screen.density_min=50 screen.mean_depth_min=1.5 screen.max_depth_min=4`
(keep only blocks with a parcel at least that deep). Survivors are ranked by max
access-depth, so `max_blocks` takes the deepest/worst blocks.

For a quick, no-download try, swap `data=capetown_full` → `data=capetown` (the committed
301-block sample; its map is geographically sparse — the full metro fills in). The
default `screen=identity` is a passthrough — a plain reblock with no screening.

## Compare methods at a budget (cost-benefit curves)

Rank the reblockers by *efficiency* — how much benefit each buys per meter of road,
across the whole budget range (not just at full build, where they converge):

```bash
pixi run python -m reblock.compare data=dji eval=kcomplexity methods=[dijkstra,mesh,greedy_arterial_buildable] max_blocks=2
```

Grades every method on three lenses — `access` (fraction of access-burden removed),
`efficiency` (network efficiency E, mean 1/distance), and `directness` (1/circuity) —
and writes, **per metric**, `auc_table_{metric}.csv` (mean AUC per method, higher = more
benefit per meter of road) and `curve_{metric}_{block}.png` (overlaid cost-benefit curves:
that metric vs road density, m/ha), for `metric ∈ {access, efficiency, directness}`. Add
`topology` and `peel` to `methods=[...]` for the full field — topology is minutes/block, so
keep the block count small (results are cached after the first run). On real data, dijkstra
tracks topology closely on access at a fraction of the compute, while peel needs ~3× the road
for the same access; `mesh` adds the dijkstra forest's cross-tree through-roads for extra
directness.

**`greedy_arterial_buildable` is the navigability flagship.** It greedily inserts the straight
through-road with the best directness gain per meter, one at a time, and leads the field on
`directness` and `efficiency` (≈10× dijkstra/mesh on real blocks) — the method to reach for when
circulation matters. The tradeoff: it is slow (minutes/block — it scores every candidate road
honestly), and it trades a little `access` AUC for that navigability, so **dijkstra remains the
fast default** for access-first reblocking. The compare is exactly how you see this Pareto
tradeoff: arterial wins directness/efficiency, dijkstra wins access-per-second.

## Multi-block (region) reblocking

`block_ids` is a **list of lists**: each inner list is a *region* reblocked jointly, so roads can
span the old block boundaries. Singletons are ordinary single-block reblocking (`[[X]]`). Existing
inter-block roads are kept as a pre-added seed the method *extends*; egress is the region's outer
perimeter (see `docs/superpowers/specs/2026-07-10-multi-block-reblocking-design.md`).

```bash
# Reblock two adjacent blocks jointly with the arterial method (roads span the old block line)
pixi run python -m reblock.run data=dji method=greedy_arterial \
  "block_ids=[[DJI.3_1_1808,DJI.3_1_1809]]" eval=kcomplexity render.enabled=true region_map.enabled=true

# Multi-block cost-benefit: grade the methods on the region
pixi run python -m reblock.compare data=dji methods=[dijkstra,mesh,greedy_arterial_buildable] \
  "block_ids=[[DJI.3_1_1808,DJI.3_1_1809]]" eval=kcomplexity
```

The reblock writes `region:..._before.png` / `_after.png` plus `region_map.png` (the region-builder's
block-membership map); the compare writes the same per-metric `auc_table_{metric}.csv` /
`curve_{metric}_{region}.png` as the single-block case, keyed by region. **This is where arterial
pulls furthest ahead** — on a region there is room for long cross-block through-roads, so its
directness AUC leads dijkstra/mesh by a wide margin (the seed's existing roads already give decent
access, so the tree methods only add local spurs; arterial adds the region-spanning routes).

A pluggable **`region_builder`** expands each seed group: `identity` (default; reblock exactly the
listed blocks) or `convex_hull` (`region_builder=convex_hull`) which fills the blocks inside a
disjoint group's convex hull into one contiguous region. `identity` warns if a group's blocks are
not adjacent (a disjoint group reblocks independently, not jointly — use `convex_hull` to fill it).
