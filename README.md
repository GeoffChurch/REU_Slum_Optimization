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
python -m reblock.run data=capetown method=peel eval=kcomplexity \
  "block_ids=[ZAF.9.3.1_1_44882]" render_dir=renders hydra.run.dir=outputs/ct-flagship
```

Writes `outputs/ct-flagship/renders/ZAF.9.3.1_1_44882_before.png` and one
`_<proposal>_after.png`. Swap `data=capetown` → `data=dji`, or `method=peel` →
`method=topology`. Omit `block_ids` to process the first `max_blocks` blocks instead.

(Quote `"block_ids=[...]"` so the shell doesn't glob the brackets.)
