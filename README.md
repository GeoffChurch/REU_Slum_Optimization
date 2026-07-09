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
pixi run python -m reblock.run data=capetown method=dijkstra eval=kcomplexity "block_ids=[ZAF.9.3.1_1_44882]" render.enabled=true
```

Writes `ZAF.9.3.1_1_44882_before.png` and one `_<proposal>_after.png` into the Hydra
run dir (`outputs/<date>/<time>/`). `dijkstra` is the default method — a buildable
frontage-routed street network (~1 s/block); swap `method=peel` (fast through-parcel
sketch) or `method=topology` (slow greedy optimizer). Swap `data=capetown` → `data=dji`,
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
