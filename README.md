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
emit both the city flagged-map and per-block before/after heatmaps:

```bash
pixi run python -m reblock.run data=capetown_full screen=dense_compact \
  method=peel eval=kcomplexity render.enabled=true flagged_map.enabled=true max_blocks=5
```

First run downloads + caches the full Cape Town data under `~/.cache/reblock` (nothing
is committed); later runs are instant. Writes `flagged_map.png` (whole city, flagged
blocks highlighted), `flagged_blocks.txt` (every flagged id), and `*_before.png` /
`*_<proposal>_after.png` for each of the `max_blocks` reblocked blocks into the Hydra
run dir. Tune the gates with `screen.density_min=50 screen.mean_depth_min=1.5`. The
default `screen=identity` is a passthrough — a plain reblock with no screening.
