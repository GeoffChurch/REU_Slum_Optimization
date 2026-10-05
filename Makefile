# reblock's tasks. Every command runs in the project's uv environment: `uv run` syncs it to uv.lock
# first, so a fresh clone needs only `uv sync` (or nothing) before `make check`.
#
# Running the pipeline is not a target: `uv run python -m reblock.run data=... method=...` (and
# `reblock.compare`) passes Hydra's `key=value` overrides through untouched, which make would not.
#
# `fmt` (`ruff format .`) is deliberately absent. It reformatted 200 of 215 files, left 12 lines
# over the `E501` limit it cannot split, and moved a `# type: ignore` off the line it suppresses --
# all measured 2026-09-20, all argued at `.githooks/pre-commit`. A task nobody can safely run is
# worse than no task: it was fired by accident once in this repo's history and cost a scoped revert
# to unpick.

.PHONY: wheel test-py web-test test typecheck-py web web-check typecheck lint hooks check \
	regen-examples

# The reblock wheel `micropip` installs into DrawRoad's Pyodide runtime (web/src/widgets/
# draw-road.ts's `data-wheel`), and the same wheel web/test/pyodide-parity.test.ts loads from
# `dist/` under Node. `dist/` is gitignored, so a fresh checkout has none, and gen_site_pages.py's
# DRAWROAD producer raises rather than emit a page pointing at a wheel that was never built --
# tests/test_gen_site_pages.py renders every partial through every producer, so `test-py` would
# fail on any checkout that had not built the wheel without this prerequisite. web/scripts/test.sh
# builds the SAME wheel again for its own leaf (`web-test`): the two give no ordering between them,
# so a bare `npm test` from web/ has to stay self-sufficient. `uv build` installs only the build
# backend (hatchling, pinned in pyproject.toml's [tool.uv]), never the environment.
wheel:
	uv build --wheel --out-dir dist

test-py: wheel
	uv run pytest

# web/'s transform.ts unit tests and the rest of the node suite. Folded into `test`, which CI runs,
# so they are on the gate with no CI step of their own for a future workflow edit to forget.
web-test:
	cd web && uv run npm ci && uv run npm test

test: test-py web-test

# No file arguments: they would override `[tool.mypy] files` in pyproject.toml, the one list of
# what this checks.
typecheck-py:
	uv run mypy --strict

web:
	cd web && uv run npm ci && uv run npm run build

web-check:
	cd web && uv run npm ci && uv run npm run check

typecheck: typecheck-py web-check

lint:
	uv run ruff check .

hooks:
	git config core.hooksPath .githooks

check: lint typecheck test

regen-examples:
	bash scripts/regenerate_examples.sh
