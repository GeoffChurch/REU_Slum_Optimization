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
pixi run python -m reblock.run data=capetown method=peel eval=kcomplexity \
  "block_ids=[ZAF.9.3.1_1_44882]" render.enabled=true hydra.run.dir=outputs/ct-flagship
```

Writes `outputs/ct-flagship/ZAF.9.3.1_1_44882_before.png` and one
`_<proposal>_after.png`. Swap `data=capetown` → `data=dji`, or `method=peel` →
`method=topology`. Omit `block_ids` to process the first `max_blocks` blocks instead.

(Quote `"block_ids=[...]"` so the shell doesn't glob the brackets.)

## Detect → reblock → visualize (one command)

Screen a city for its dense/compact informal blocks, reblock the top survivors, and
emit both the city flagged-map and per-block before/after heatmaps. This runs on the
committed 301-block Cape Town sample — no download, ~5 s:

```bash
pixi run python -m reblock.run data=capetown screen=dense_compact screen.density_min=35 \
  method=peel eval=kcomplexity render.enabled=true flagged_map.enabled=true \
  max_blocks=5 hydra.run.dir=outputs/ct-screen
```

Writes into `outputs/ct-screen/`: `flagged_map.png` (the sample, flagged blocks in red
over grey context), `flagged_blocks.txt` (every flagged id), and `*_before.png` /
`*_<proposal>_after.png` for each of the `max_blocks` reblocked survivors. Tune the
gates with `screen.density_min=50 screen.mean_depth_min=1.5`.

For the **full Cape Town metro** — thousands of contiguous blocks, a dense city map —
swap `data=capetown` → `data=capetown_full`; the first run downloads + caches the real
data under `~/.cache/reblock` (nothing committed), later runs are instant. (The sample
above is geographically sparse, so its map reads as scattered blocks; the full metro
fills in.)

The default `screen=identity` is a passthrough — a plain reblock with no screening.
