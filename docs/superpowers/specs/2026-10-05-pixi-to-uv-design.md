# Moving reblock from pixi to uv (2026-10-05)

The owner decided on 2026-10-04/05 that every repo converges on uv.
That was decision 4 of the GeoffChurch/bookgen migration; GeoffChurch/cluster_submit's `docs/superpowers/specs/2026-10-05-uv-env-and-git-mirrors.md` records it.
reblock is the only pixi repo, and it moves now.
On 2026-10-05 the owner chose a Makefile to replace pixi's tasks and accepted every ruling below; they are binding.

The move turns the derive cache cold, because its keys carry the native libraries' versions.
So the examples are regenerated as part of the move, once, under uv, and not before it.

## Where things stand

- **pixi 0.71.3, configured in `pyproject.toml`.**
  - Channel conda-forge; platforms linux-64 and osx-arm64.
  - Runtime packages: Python `>=3.11,<3.13` (it resolves 3.12.13), geopandas, pandas, pyarrow, networkx, scipy, matplotlib-base, pyshp, hydra-core, joblib, segno and numba.
  - From PyPI: `reblock` and `topology` (from `ext/topology`), both editable.
  - A `dev` feature: hatchling, pip, mypy, pytest, pytest-cov, pytest-xdist, ruff, types-geopandas, types-shapely, types-pyyaml, and nodejs `22.*`.
  - A `research` feature: pot, statsmodels, scikit-image, threadpoolctl.
  - One environment, `default` (dev plus research), and 14 tasks: wheel, test-py, web-test, test, typecheck-py, web, web-check, typecheck, lint, hooks, check, run, compare and regen-examples.
- **`research/roadless` is 112 commits ahead of `main`, and `main` is 0 ahead.**
  - The branch adds a `launch` feature and environment holding GeoffChurch/cluster_submit (a git dependency pinned to a commit), and its cluster jobs run with `cs.Pixi()`.
  - Its GPU dependencies are outside any lock: `cupy-cuda12x[ctk]==14.2.0` and `pyamg==5.3.0`.
    `pip install --target` puts them in `~/.cache/reblock-research/pydeps`: setup commands do this on the cluster, and it was done by hand here.
    Code reaches them through `PYTHONPATH` (`research/roadless/cluster_env.sh`) and a `sys.path.insert` (`research/roadless/lifted.py`).
    The setup commands delete numpy from that directory so it cannot shadow the env's own.
- **What PyPI provides, checked 2026-10-05 with uv 0.11.28 and Python 3.12:**
  - pyogrio 0.13.0 bundles GDAL 3.12.4, which has the OSM driver; it is read-only, which is all `read_pbf_lines` needs.
    The pixi env has GDAL 3.13.1.
  - shapely 2.1.2 bundles GEOS 3.13.1; the pixi env has 3.14.1.
  - pyproj 3.8.0 bundles PROJ 9.8.1, the same as the pixi env.
  - `nodejs-wheel~=22.0` puts node v22.20.0, npm and npx in the venv's `bin/`.
  - `types-pyogrio` exists (0.13.0.20260807).
  - `uv build --wheel` with `[tool.uv] build-constraint-dependencies = ["hatchling==1.32.0"]` builds with hatchling 1.32.0, and a project whose packages are all in dependency groups gets a wheel with no `Requires-Dist`.
- **reblock is public.**
  CI (`.github/workflows/ci.yml`: lint, typecheck, test) runs on pull requests and on pushes to `main`.
  `.github/workflows/deploy-site.yml` publishes the site from `main`.
- **The Pyodide widget installs the reblock wheel with `micropip.install(wheelUrl)`** (`web/src/py/runtime.ts`).
  That call resolves the wheel's `Requires-Dist`, and PyPI has no wasm wheels for the science stack, so the wheel must keep declaring no dependencies.

## The design

### Dependencies (`main`)

- **`[project] dependencies` stays empty**, because the wheel goes to Pyodide (see above).
  The packages pixi listed go into dependency groups, which uv installs but never writes into the wheel:
  - `runtime`: today's runtime packages plus `topology`.
  - `dev`: mypy, pytest, pytest-cov, pytest-xdist, ruff, types-geopandas, types-shapely, types-pyyaml, `types-pyogrio`, and `nodejs-wheel~=22.0`.
  - `research`: pot, statsmodels, scikit-image, threadpoolctl.

  `[tool.uv] default-groups = ["runtime", "dev", "research"]`, because pixi's one environment had all three.
- **PyPI names replace the conda names:** `matplotlib-base` becomes `matplotlib`, and `pot` becomes `POT`.
  Requirements stay unpinned, as they are now; `uv.lock` pins them.
- **`topology`** comes from `[tool.uv.sources] topology = { path = "ext/topology", editable = true }`.
  uv installs reblock itself editable, as pixi did.
- **`types-pyogrio` replaces the `pyogrio.*` `ignore_missing_imports` override.**
  The override existed because conda-forge had no stub package.
- **hatchling and pip leave the dev group:** the wheel is built by `uv build` (see "The wheel" below).
- **Python:** `.python-version` holds 3.12.
  `requires-python` stays `>=3.11`, since mypy and ruff target 3.11.
- **Platforms:** `[tool.uv] environments` limits the lock to Linux and macOS on arm64, as pixi's list did.
- **Files:** `uv.lock` is committed.
  `pixi.lock` and every `[tool.pixi.*]` table are deleted.
  `.gitignore` drops `.pixi/` and adds `.venv/`.

### Tasks: a Makefile (`main`)

- **The Makefile keeps pixi's task names and dependency graph:**
  - `wheel`, `test-py`, `web-test`, `test`, `typecheck-py`, `web`, `web-check`, `typecheck`, `lint`, `hooks`, `check` and `regen-examples`;
  - `test` = `test-py` + `web-test`; `typecheck` = `typecheck-py` + `web-check`; `check` = `lint` + `typecheck` + `test`;
  - `test-py` depends on `wheel`.

  Commands run through `uv run`.
  The explanations in today's `[tool.pixi.tasks]` comments move with their tasks, including why there is no `fmt`.
- **`run` and `compare` do not become targets.**
  The README gives `uv run python -m reblock.run …` and `uv run python -m reblock.compare …`, since make would mangle Hydra's `key=value` overrides.
- **`typecheck-py` runs `mypy --strict` with no file arguments**, so `[tool.mypy] files` is the only list.
  The part of `tests/test_typecheck_config.py` that keeps the two lists in step goes, as does the pyproject comment saying the `files` list is inert on its own.
  Whatever else that test pins (no `exclude`) stays.

### The wheel: one build (`main`)

- **Every wheel is built by `uv build --wheel --out-dir dist`:** the `wheel` target, `web/scripts/test.sh`, and `deploy-site.yml`.
  hatchling is pinned once, by `[tool.uv] build-constraint-dependencies = ["hatchling==1.32.0"]`, the version the deploy job pins now.
- **The deploy job installs uv (`astral-sh/setup-uv`) for this step.**
  It still installs no science stack, because `uv build` installs only the build backend.
  Its `pip install hatchling`, its `pip wheel`, and the comment arguing that the two builds are byte-identical all go.
- **A test pins that the built wheel's `METADATA` has no `Requires-Dist`,** if no test does already.

### The cache key (`main`)

- **`derive_graph.env_version` gains GDAL's version** (`pyogrio.__gdal_version_string__`), next to GEOS and PROJ.
  Blocks (`data/shapefile.py`), footprints, informal structures, desire-line snapshots and OSM footpaths are all read through GDAL, and this move is the first time GDAL's version changes under the cache.
  Every key misses after the move anyway, so adding it now costs nothing, and the next GDAL change will not go unnoticed.

### CI, the hook, the docs (`main`)

- **`ci.yml`** uses `astral-sh/setup-uv` and runs `uv sync --locked`, which fails on a stale `uv.lock` as pixi's `locked: true` did.
  Then it runs `make lint`, `make typecheck` and `make test`.
  The submodule checkout stays.
- **`deploy-site.yml`** changes as above.
  `actions/setup-node` stays at 22, and its comment now points at the dev group's `nodejs-wheel`.
- **`.githooks/pre-commit`** runs `make lint`, and skips with a message when uv is not on `PATH`.
  `make hooks` installs it.
- **Every mention of pixi in a file still in use moves:**
  - README; `docs/_partials/*` (`reproduce.md`'s install steps, and its site build, which becomes `uvx` with the same pinned mkdocs packages); `mkdocs.yml`;
  - `.gitignore`'s comments; `data/adjudication/` docs; `tests/data/kblock/PROVENANCE.md`; `docs/superpowers/backlog.md`;
  - docstrings and messages in `scripts/` and `tests/` (`src/` has none); `web/scripts/test.sh`;
  - each generator's `Regenerate:` line, and the copies of it in committed artifacts (bundles, `.d.ts` files, `meta.json`, example READMEs).

  A commit that changes a generator's line changes the committed copies in the same commit, so the suite stays green before the regen.
- **Dated records keep their commands as history.**
  These are the specs, plans and notes under `docs/superpowers/`: 882 mentions in 93 files.
  At the end, `git grep -i pixi` on `main` finds only those records and this spec.

### The regen (`main`)

- **After `make check` passes, `make regen-examples` runs once, in the background, under the uv env.**
  It takes hours, because the cache is cold.
  At most one GPU process runs at a time, and pool workers run single-threaded.
- **Its output is committed as it lands:** examples, bundles and meta files.
  Figures may move a little, since GEOS goes from 3.14 to 3.13 and GDAL from 3.13 to 3.12; they are allowed to change with no before-and-after accounting.
- **Then `make check` runs again,** this time with the slow tests, which need a warm cache.

### `research/roadless`

- **The branch merges `main`.**
- **Groups:**
  - `gpu` holds `cupy-cuda12x[ctk]==14.2.0` and `pyamg==5.3.0`, with Linux markers.
  - `submit` holds cluster_submit, its git source in `[tool.uv.sources]`, pinned to the commit the branch pins now.

  Both are default groups on this branch, so a plain `uv run` or `uv sync` never removes the launcher.
  bookgen's final review found that failure, where a sync without its group uninstalled cluster_submit.
  The `launch` environment goes.
- **`research/roadless/cluster.py` uses `cs.Uv()`.**
  The default groups include `gpu` and `submit`, and cluster_submit reaches the cluster through the package's git mirrors.
  `DEPS` and the `pip --target` setup commands go.
- **`cluster_env.sh`** keeps the thread variables and `PYTHONPATH=.`; the `pydeps` entries go.
  `CUDA_PATH` and `LD_LIBRARY_PATH` stay only if cupy in the venv still needs them, pointed into the venv; this is checked on a node.
- **Other files:**
  - `lifted.py`'s `sys.path.insert` goes, and `NOTES.md`'s `pydeps` line changes.
  - Usage lines become `uv run python research/roadless/cluster.py …` and `PYTHONPATH=. uv run python research/roadless/<script> …`.
  - The local `CUDA_PATH=/usr` stays only if it is still needed.
- **Verification:**
  - On the cluster, `setup` builds the checkout's `.venv` through the mirrors, cluster_submit included.
  - A short `blocks` run on one small block with the GPU solver goes end to end, through `wait` and the pull, with cupy and pyamg imported from the venv.
  - Here, one short GPU run of `relax.py` on a small block.

### GeoffChurch/cluster_submit

- **Once `research/roadless` runs on `Uv`, the `Pixi` env, its tests and its docs go** (README, `docs/migrating/`).
  `Env` stays a Protocol, with `Uv` its only implementation, so the injection point remains.
  `research/roadless` then pins the new commit.

### Clean-up, each step only when the owner says

- **After the regen:**
  - reblock's `.pixi/` (2.0 GB);
  - the derive-cache entries no run has used since the move, identified by last access, in `~/.cache/reblock` (20 GB with the downloaded data);
  - the `pydeps` directories, here and on the cluster.
- **Uninstalling pixi from this machine** is also the owner's call; no other repo uses it.

## Order

1. **Branch `uv` from `main`:** dependencies, Makefile, wheel, cache key, CI, hook and docs.
   `make check` passes.
2. **The regen,** committed on `uv`; `make check` passes, slow tests included.
3. **Merge `uv` into `main` and push**, which runs CI and deploys the site.
   This step is the owner's call.
4. **`research/roadless` merges `main` and converts;** live verification.
5. **cluster_submit deletes `Pixi`;** `research/roadless` pins it.
6. **The clean-up the owner approves.**

## Verification

- **A fresh clone,** with submodules, builds its env with `uv sync --locked`, and `make check` passes: ruff; mypy strict and `tsc --noEmit`; pytest and the web suite.
- **The wheel from `uv build` has no `Requires-Dist`.**
  The site builds locally per `reproduce.md`, and the Pyodide parity test passes on that wheel.
- **`read_pbf_lines` reads a real PBF under the wheel's GDAL.**
- **The regen completes and the slow tests pass.**
- **`research/roadless` passes its live checks** (above).

## Not in scope

- SIMP, or any other research change.
- Merging `research/roadless` into `main`.
- Deleting anything the owner has not approved.
