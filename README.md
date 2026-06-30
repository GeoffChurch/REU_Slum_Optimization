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
