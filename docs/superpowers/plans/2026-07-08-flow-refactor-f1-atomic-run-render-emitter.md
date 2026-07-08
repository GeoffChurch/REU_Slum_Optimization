# Flow-refactor F1 — Atomic pure `run()` + render emitter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `run()` a pure function (single method, no rendering, no global RNG side-effect) and move rendering into a standalone, opt-in render emitter.

**Architecture:** `run(cfg) -> list[Result]` loads one Source, runs one Method per block, scores with each Eval, returns Results — writing nothing. `reblock.run:main` (the only Hydra app) logs a per-Result summary and, when `render.enabled`, calls `render_results(results, out_dir, cfg.render)` from the new `reblock.emit` module. `TopologyMethod` isolates its RNG seeding so a run leaves the caller's global numpy/stdlib RNG untouched.

**Tech Stack:** Python 3.12, geopandas/shapely 2.1, numpy, Hydra (`_target_` + `instantiate`), matplotlib, pixi, pytest, `mypy --strict`, ruff.

## Global Constraints

- `pixi run check` must stay green — it runs `ruff check` + `ruff format --check` + `mypy --strict src tests scripts/crossblock_probe.py` + `pytest`. The suite is currently 107 tests.
- **No dual path / no compat shim** (owner directive): the `method` list is dropped outright to a single method; rendering is removed from `run()`, not left behind a flag. No fallback branches.
- **`run()` is pure:** returns `list[Result]`, takes no output-dir argument, writes no files, has no global side-effect (does not perturb the process-global numpy/stdlib RNG), and is bit-identical on repeat for a given `cfg`.
- **`Result.metrics` stays a tuple** and the **`eval` list is retained** — the render path needs the `kcomplexity` eval's `fields` (`access_before`/`access_after`) regardless of which eval a caller cares about.
- **No emitter registry in F1.** `main` calls the single render emitter directly; the `enabled_emitters(cfg)` fan-out is deferred to F4 (arrives with the scorecard emitter).
- **Render scope = `png` / `separate` only.** `RenderConfig` carries `enabled`/`format`/`layout`; any other `format`/`layout` value fails loud (`NotImplementedError`) — `webpage`/`side_by_side` are deferred.
- **`block_ids` early-filter is already shipped** (kblock `Source`) and is unchanged by this slice.
- Commit trailers on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
  ```

---

### Task 1: `TopologyMethod` — isolate RNG seeding (no global side-effect)

**Files:**
- Modify: `src/reblock/methods/topology.py:82-84`
- Test: `tests/methods/test_topology_method.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `TopologyMethod.propose` unchanged signature/output for a given `seed`; new guarantee that it leaves the process-global numpy **and** stdlib RNG state untouched.

**Why save/restore, not a threaded `Generator`:** the spec's §1 wording ("local `default_rng` passed into the builder") is not achievable without editing vendored code — `topology.build_all_roads` → `choose_path` → `WeightedPick` (`ext/topology/topology/graph/my_graph_helpers.py:174`) draws from the **global** `np.random.choice` and accepts no rng. Save/restore delivers the same observable contract (deterministic for a given seed **and** no net global side-effect) without touching `ext/topology`.

- [ ] **Step 1: Write the failing test**

Add to `tests/methods/test_topology_method.py` (the `_grid` helper already exists there):

```python
def test_propose_does_not_perturb_global_rng() -> None:
    # run() purity depends on propose() not mutating the caller's global RNG.
    # propose() seeds np.random/random internally (build_all_roads draws from
    # the global np.random.choice), so it must save+restore that global state.
    import numpy as np
    block = _grid(3)
    np.random.seed(12345)
    np_state_before = np.random.get_state()[1].tolist()
    py_state_before = random.getstate()
    TopologyMethod(alpha=2.0, seed=0).propose(block)
    assert np.random.get_state()[1].tolist() == np_state_before
    assert random.getstate() == py_state_before
```

Add `import random` at the top of the test file if not already present (it is imported locally inside `test_all_interior_parcels_connected`; add a module-level `import random` and drop the local one, or keep both — a module-level import is cleanest).

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/methods/test_topology_method.py::test_propose_does_not_perturb_global_rng -v`
Expected: FAIL — `propose` currently calls `np.random.seed(self.seed)` / `random.seed(self.seed)`, leaving global state changed, so the `assert` fails.

- [ ] **Step 3: Implement the save/restore isolation**

In `src/reblock/methods/topology.py`, replace the seeding block (currently lines 79-84):

```python
        # build_all_roads is probabilistic: choose_path -> WeightedPick draws
        # from numpy's global RNG (np.random.choice), so np.random.seed is what
        # actually pins the road layout; seed random too for any stdlib draws.
        random.seed(self.seed)
        np.random.seed(self.seed)
        build_all_roads(graph, alpha=self.alpha, vquiet=True)
```

with:

```python
        # build_all_roads is probabilistic: choose_path -> WeightedPick draws
        # from numpy's GLOBAL RNG (np.random.choice; ext/topology, not threadable
        # without editing vendored code), so we seed the global RNGs to pin the
        # layout but SAVE+RESTORE them so propose() leaves the caller's RNG
        # untouched -- run() must be side-effect free (spec §1).
        np_state = np.random.get_state()
        py_state = random.getstate()
        try:
            random.seed(self.seed)
            np.random.seed(self.seed)
            build_all_roads(graph, alpha=self.alpha, vquiet=True)
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)
```

- [ ] **Step 4: Run the topology tests**

Run: `pixi run pytest tests/methods/test_topology_method.py -v`
Expected: PASS — the new test passes, and the existing `test_propose_is_deterministic_across_runs` / efficacy tests still pass (the draws are identical; only the surrounding state is now restored).

- [ ] **Step 5: Commit**

```bash
git add src/reblock/methods/topology.py tests/methods/test_topology_method.py
git commit -m "$(cat <<'EOF'
refactor: TopologyMethod isolates global RNG seeding (save/restore)

propose() seeds np.random/random for build_all_roads (which draws from the
global RNG via vendored WeightedPick), then restores prior state so it has no
global side-effect -- a prerequisite for run() purity. Deterministic output
for a given seed is unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

### Task 2: `reblock.emit` — the render emitter (new module)

**Files:**
- Create: `src/reblock/emit.py`
- Test: `tests/test_emit.py`

**Interfaces:**
- Consumes: `reblock.contracts.Result` / `Metrics`; `reblock.render.render_before(block, layers, *, vmax)`, `render_after(block, proposal, layers, *, vmax, metrics=None)`, `save_render(fig, path)`.
- Produces:
  - `RenderConfig` dataclass: `enabled: bool = False`, `format: str = "png"`, `layout: str = "separate"`.
  - `render_results(results: list[Result], out_dir: Path, cfg: RenderConfig) -> None` — per block, a shared-`vmax` `{block_id}_before.png` + one `{block_id}_{proposal}_after.png` per Result (proposal-id fallback `proposal{i}`). Reads kcomplexity `fields`; a block with no kcomplexity metric is skipped. Writes into `out_dir` directly (no `renders/` subdir).

This is purely additive — it does not touch `run.py`. `run.py`'s existing `_render_block` keeps working until Task 4 removes it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_emit.py`:

```python
from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
import pytest
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.contracts import Block, Metrics, Proposal, Result
from reblock.emit import RenderConfig, render_results

UTM = CRS.from_epsg(32643)


def _grid_block(n: int) -> Block:
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(n) for j in range(n)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _kc(block: Block) -> Metrics:
    layers = pd.Series([1] * len(block.parcels),
                       index=pd.Index(block.parcels["parcel_id"], name="parcel_id"))
    return Metrics(block_id=block.block_id, method="x", eval="kcomplexity",
                   values={"delta_k": 0.0},
                   fields={"access_before": layers, "access_after": layers})


def test_render_results_after_filenames_unique_for_empty_proposal_ids(tmp_path: Path) -> None:
    # Two proposals for one block that both leave proposal_id="" must not collide
    # onto one filename -- the emitter falls back to a per-proposal index.
    block = _grid_block(3)
    results = [
        Result(block=block, proposal=Proposal(block_id="g", crs=UTM, proposal_id=""),
               metrics=(_kc(block),)),
        Result(block=block, proposal=Proposal(block_id="g", crs=UTM, proposal_id=""),
               metrics=(_kc(block),)),
    ]
    render_results(results, tmp_path, RenderConfig(enabled=True))
    afters = sorted(p.name for p in tmp_path.glob("*_after.png"))
    assert afters == ["g_proposal0_after.png", "g_proposal1_after.png"]
    assert (tmp_path / "g_before.png").exists()


def test_render_results_skips_block_without_kcomplexity(tmp_path: Path) -> None:
    block = _grid_block(3)
    other = Metrics(block_id="g", method="x", eval="weakdual_k", values={"k": 1.0})
    render_results([Result(block=block, proposal=Proposal(block_id="g", crs=UTM), metrics=(other,))],
                   tmp_path, RenderConfig(enabled=True))
    assert list(tmp_path.glob("*.png")) == []


def test_render_results_rejects_unsupported_format(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        render_results([], tmp_path, RenderConfig(enabled=True, format="webpage"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_emit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reblock.emit'`.

- [ ] **Step 3: Implement `src/reblock/emit.py`**

```python
"""Output emitters: consumers of run()'s Result list. F1 ships the render
emitter (per block: a shared-vmax before + one after per proposal). The
enabled-emitter registry/fan-out arrives in F4 with the scorecard emitter.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from reblock.contracts import Metrics, Result
from reblock.render import render_after, render_before, save_render

_KCOMPLEXITY = "kcomplexity"


@dataclass
class RenderConfig:
    enabled: bool = False
    format: str = "png"       # F1: "png" only
    layout: str = "separate"  # F1: "separate" only


def _kcomplexity_metrics(metrics: tuple[Metrics, ...]) -> Metrics | None:
    """The kcomplexity `Metrics` in a Result's metrics, if scored -- the eval
    that emits the per-parcel access-depth arrays render consumes
    (`fields["access_before"]` / `fields["access_after"]`)."""
    return next((m for m in metrics if m.eval == _KCOMPLEXITY), None)


def render_results(results: list[Result], out_dir: Path, cfg: RenderConfig) -> None:
    """Per block: a shared-`vmax` `{block_id}_before.png` + one
    `{block_id}_{proposal}_after.png` per Result. Reads the kcomplexity
    access-depth arrays from `Result.metrics` (render never recomputes the
    peel), so a block scored without kcomplexity is skipped."""
    if cfg.format != "png" or cfg.layout != "separate":
        raise NotImplementedError(
            f"render F1 supports format=png/layout=separate only; "
            f"got format={cfg.format!r} layout={cfg.layout!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    by_block: dict[str, list[Result]] = {}
    for r in results:
        by_block.setdefault(r.block.block_id, []).append(r)
    for group in by_block.values():
        _render_block_group(group, out_dir)


def _render_block_group(group: list[Result], out_dir: Path) -> None:
    block = group[0].block
    # access_before is method-independent: take it from the first Result that
    # carries kcomplexity metrics; a block scored without kcomplexity has no
    # peel layers to draw and is skipped.
    kc_first = next(
        (kc for r in group if (kc := _kcomplexity_metrics(r.metrics)) is not None), None)
    if kc_first is None:
        return
    access_before = kc_first.fields["access_before"]
    # access_after can only shrink depth, so access_before.max() bounds the
    # shared color scale across the before and every after.
    vmax = int(access_before.max())

    fig_before = render_before(block, access_before, vmax=vmax)
    save_render(fig_before, out_dir / f"{block.block_id}_before.png")
    plt.close(fig_before)

    for i, r in enumerate(group):
        kc = _kcomplexity_metrics(r.metrics)
        if kc is None:
            continue
        # proposal_id defaults to "" (a method may leave it unset); fall back to
        # a per-proposal index so multiple afters never collide/overwrite.
        name = r.proposal.proposal_id or f"proposal{i}"
        fig_after = render_after(block, r.proposal, kc.fields["access_after"],
                                 vmax=vmax, metrics=kc)
        save_render(fig_after, out_dir / f"{block.block_id}_{name}_after.png")
        plt.close(fig_after)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/test_emit.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/reblock/emit.py tests/test_emit.py
git commit -m "$(cat <<'EOF'
feat: reblock.emit render emitter (lifted from run._render_block)

Additive: RenderConfig + render_results(results, out_dir, cfg) group Results
by block and draw a shared-vmax before + one after per proposal, reading the
kcomplexity access-depth fields. run.py still renders via _render_block until
F1 Task 4 wires this emitter in and removes it.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

### Task 3: Drop the `method` list → single method

**Files:**
- Modify: `src/reblock/run.py:75-105` (the `run()` loop), `src/reblock/run.py:66-70` (`RunConfig.__post_init__`)
- Modify: `conf/method/topology.yaml`, `conf/method/peel.yaml`
- Test: `tests/test_run.py` (migrate `test_runconfig_accepts_explicit_data_method_eval_overrides`; delete `test_run_scores_multiple_methods_in_one_call`)

**Interfaces:**
- Consumes: `cfg.method` is now a **single** `_target_` dict (not a list); `cfg.eval` stays a **list**.
- Produces: `run()` yields one `Result` per block (one method), `Result.metrics` still a tuple over the eval list. Rendering inside `run()` is untouched in this task (removed in Task 4).

- [ ] **Step 1: Update the config groups to single-method dicts**

`conf/method/topology.yaml` — replace entire contents with:

```yaml
# Single method per run (F1 atomic run(): cfg.method is ONE _target_ dict).
# Multi-method comparison lives in reblock.compare (F4), not here.
_target_: reblock.methods.topology.TopologyMethod
alpha: ${alpha}
seed: ${seed}
```

`conf/method/peel.yaml` — replace entire contents with:

```yaml
# Single method per run (see conf/method/topology.yaml).
_target_: reblock.methods.peel.PeelReblocker
```

- [ ] **Step 2: Migrate the explicit-overrides test to a single method dict**

In `tests/test_run.py`, change `test_runconfig_accepts_explicit_data_method_eval_overrides` so `method` is a single dict (was a one-element list):

```python
    cfg = RunConfig(
        max_blocks=1,
        data={"_target_": "reblock.data.shapefile.ShapefileSource",
              "path": PHULE, "region_id": "phule", "assumed_crs": 3857},
        method={"_target_": "reblock.methods.topology.TopologyMethod", "alpha": 2.0, "seed": 0},
        eval=[{"_target_": "reblock.eval.kcomplexity.KComplexityEval"},
              {"_target_": "reblock.eval.kcomplexity.WeakDualKEval"}],
    )
```

Delete the whole `test_run_scores_multiple_methods_in_one_call` test (lines ~161-203) — its "one before, N afters" render coverage now lives in `tests/test_emit.py::test_render_results_after_filenames_unique_for_empty_proposal_ids`, and multi-method comparison moves to `reblock.compare` (F4).

- [ ] **Step 3: Run to verify the failure**

Run: `pixi run pytest tests/test_run.py -v`
Expected: FAIL — `run()` still does `[instantiate(m) for m in cfg.method]`, which now iterates the single method dict's **keys** (`_target_`, `alpha`, `seed`) and blows up. This confirms the loop must change.

- [ ] **Step 4: Update `run()` and `RunConfig` to single method**

In `src/reblock/run.py`, in `RunConfig.__post_init__` change the method default (currently lines 66-70) from a list to a single dict:

```python
        if self.method is None:
            self.method = {
                "_target_": "reblock.methods.topology.TopologyMethod",
                "alpha": self.alpha, "seed": self.seed,
            }
```

In `run()`, replace the method-list instantiate + inner method loop. The body (currently lines 83-105) becomes:

```python
    # Per-element instantiate for the eval LIST (not instantiate(cfg.eval) as a
    # whole): instantiating a ListConfig of @dataclass _target_s short-circuits
    # to schema-validated DictConfig nodes instead of calling the constructor.
    # cfg.method is a single _target_ dict, so instantiate(cfg.method) calls the
    # constructor directly and is safe.
    source = cast(Source, instantiate(cfg.data))
    method = cast(Method, instantiate(cfg.method))
    evals = cast("list[Eval]", [instantiate(e) for e in cfg.eval])

    render_dir = (render_base / cfg.render_dir
                  if render_base is not None and cfg.render_dir else None)
    if render_dir is not None:
        render_dir.mkdir(parents=True, exist_ok=True)

    region = source.region()
    results: list[Result] = []
    for block in islice(region.blocks, cfg.max_blocks):
        proposal = method.propose(block)
        metrics = tuple(ev.score(block, proposal) for ev in evals)
        results.append(Result(block=block, proposal=proposal, metrics=metrics))

        if render_dir is not None:
            _render_block(block, [(proposal, metrics)], render_dir)

    return results
```

(`_render_block`, `_kcomplexity_metrics`, the render imports, and `render_base` stay for now — Task 4 removes them. `Proposal` is still imported and used in `_render_block`'s type hint.)

- [ ] **Step 5: Run the full suite**

Run: `pixi run pytest -q`
Expected: PASS — the migrated override test passes, the deleted multi-method test is gone, the compose/CLI tests still render one before + one after per block. Then `pixi run check` for lint+types.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/run.py conf/method/topology.yaml conf/method/peel.yaml tests/test_run.py
git commit -m "$(cat <<'EOF'
refactor: run() takes a single method, not a method list (F1)

Drop the method LIST -> single method: cfg.method is one _target_ dict,
run() produces one Result per block. eval stays a list (Result.metrics is a
tuple). Multi-method "one before, N afters" comparison moves to reblock.compare
(F4); its render coverage is retained by tests/test_emit.py. Rendering inside
run() is untouched here (removed in Task 4).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

### Task 4: Purify `run()` (remove render coupling) + wire the emitter into `main`

**Files:**
- Modify: `src/reblock/run.py` (remove `render_base`, `render_dir`, `_render_block`, `_kcomplexity_metrics`, `_KCOMPLEXITY`, render/matplotlib imports; wire `render_results` into `main`)
- Modify: `conf/config.yaml` (replace `render_dir: null` with a `render:` block)
- Modify: `README.md` (recipe: `render_dir=renders` → `render.enabled=true`)
- Test: `tests/test_run.py` (migrate CLI + phule-wiring tests; delete the `_render_block` filename test; add a purity test)

**Interfaces:**
- Consumes: `reblock.emit.render_results`, `reblock.emit.RenderConfig`.
- Produces: `run(cfg: RunConfig | DictConfig) -> list[Result]` — no `render_base` parameter, writes nothing. `main` renders only when `cfg.render.enabled`, into the Hydra run dir.

- [ ] **Step 1: Write the failing purity test + migrate the render tests**

In `tests/test_run.py`:

Update the import line (drop `_render_block`):

```python
from reblock.run import RunConfig, run
```

Delete `test_render_after_filenames_stay_unique_when_proposal_id_is_empty` entirely (it calls `_render_block`, which this task removes; the behavior is covered by `tests/test_emit.py`). The `_grid_block` helper it used can stay if other tests use it, otherwise delete it too.

Migrate `test_end_to_end_phule_wiring` to a pure-`run()` test (no render args, no PNG assertions):

```python
def test_end_to_end_phule_wiring() -> None:
    # Wiring proof on real data: run() returns well-formed Results and writes
    # nothing (rendering is an emitter now, exercised by the CLI test below).
    results = run(RunConfig(shapefile=PHULE, region_id="phule", alpha=2.0, seed=0,
                            max_blocks=1, assumed_crs=3857))
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, Result)
    assert r.block.block_id == "phule_0"
    assert r.metric("kcomplexity", "k_after") <= r.metric("kcomplexity", "k_before")
    assert r.metric("kcomplexity", "delta_k") >= 0
```

Add a purity test:

```python
def test_run_is_pure_deterministic_and_leaves_global_rng_untouched() -> None:
    import numpy as np
    cfg = RunConfig(shapefile=PHULE, region_id="phule", alpha=2.0, seed=0,
                    max_blocks=1, assumed_crs=3857)
    np.random.seed(777)
    state_before = np.random.get_state()[1].tolist()
    r1 = run(cfg)
    r2 = run(cfg)
    # no global RNG side-effect
    assert np.random.get_state()[1].tolist() == state_before
    # bit-identical repeats
    assert [x.proposal.proposal_id for x in r1] == [x.proposal.proposal_id for x in r2]
    assert (r1[0].metric("kcomplexity", "delta_k")
            == r2[0].metric("kcomplexity", "delta_k"))
```

Migrate the two CLI subprocess tests: change `"render_dir=renders"` → `"render.enabled=true"`, and the PNG globs from `tmp_path/renders/...` to `tmp_path/...` (the emitter writes into the run dir directly). For `test_cli_entrypoint_smoke`:

```python
    result = subprocess.run(
        [sys.executable, "-m", "reblock.run",
         f"shapefile={PHULE}", "max_blocks=1", "assumed_crs=3857",
         "render.enabled=true", f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "phule_0" in result.stdout
    assert "k_before" in result.stdout

    befores = list(tmp_path.glob("phule_0_before.png"))
    afters = list(tmp_path.glob("phule_0_*_after.png"))
    assert len(befores) == 1 and befores[0].stat().st_size > 0
    assert len(afters) >= 1 and afters[0].stat().st_size > 0
```

For `test_cli_block_ids_renders_single_capetown_block`: same two edits — `"render.enabled=true"` and glob `tmp_path.glob("ZAF.9.3.1_1_44882_before.png")` / `tmp_path.glob("ZAF.9.3.1_1_44882_*_after.png")`.

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_run.py -v`
Expected: FAIL — `render.enabled` isn't a config key yet (Hydra override error) and `run()` still has render coupling; the purity test also can't pass while `main`/config are unmigrated.

- [ ] **Step 3: Add the `render` config block**

In `conf/config.yaml`, replace the line `render_dir: null` with:

```yaml
# Render emitter (reblock.emit): opt-in, writes {block}_before.png +
# {block}_{proposal}_after.png into the Hydra run dir. F1 supports png/separate.
render:
  enabled: false
  format: png
  layout: separate
```

- [ ] **Step 4: Purify `run.py` and wire the emitter into `main`**

Rewrite `src/reblock/run.py` so `run()` has no render coupling and `main` calls the emitter. The new file:

```python
"""Hydra entrypoint: composes conf/{data,method,eval} config groups into a
pluggable Source -> Method -> [Eval] pipeline. `run()` is a pure function
(returns Results, writes nothing); rendering is an opt-in emitter called by
`main` (see reblock.emit).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, cast

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from reblock.contracts import Eval, Method, Result, Source
from reblock.emit import render_results

log = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Flat, ergonomic constructor for direct/programmatic use (tests, small
    scripts). `data`/`method`/`eval` are the same `_target_`-bearing shapes
    Hydra's config-group composition produces (see conf/config.yaml); when left
    unset, __post_init__ derives them from the flat fields below, so `run()`
    has exactly one code path (`hydra.utils.instantiate`).
    """
    shapefile: str = "???"
    region_id: str = "phule"
    alpha: float = 2.0
    seed: int = 0
    max_blocks: int = 1
    # ShapefileSource fails loud instead of guessing a CRS when a shapefile has
    # no .prj (e.g. Phule Nagar); None preserves "fail loud" as the CLI default.
    assumed_crs: int | None = None
    data: Any = None
    method: Any = None
    eval: Any = None

    def __post_init__(self) -> None:
        if self.data is None:
            self.data = {
                "_target_": "reblock.data.shapefile.ShapefileSource",
                "path": self.shapefile, "region_id": self.region_id,
                "assumed_crs": self.assumed_crs,
            }
        if self.method is None:
            self.method = {
                "_target_": "reblock.methods.topology.TopologyMethod",
                "alpha": self.alpha, "seed": self.seed,
            }
        if self.eval is None:
            self.eval = [{"_target_": "reblock.eval.kcomplexity.KComplexityEval"}]


def run(cfg: RunConfig | DictConfig) -> list[Result]:
    """Pure: one Source, one Method per block, scored by each Eval -> one Result
    per block (Result.metrics is a tuple over the eval list). Writes nothing and
    has no global side-effect."""
    # Per-element instantiate for the eval LIST (not instantiate(cfg.eval) whole):
    # instantiating a ListConfig of @dataclass _target_s short-circuits to
    # schema-validated DictConfig nodes instead of calling the constructor.
    # cfg.method is a single _target_ dict, so instantiate(cfg.method) is safe.
    source = cast(Source, instantiate(cfg.data))
    method = cast(Method, instantiate(cfg.method))
    evals = cast("list[Eval]", [instantiate(e) for e in cfg.eval])

    region = source.region()
    results: list[Result] = []
    for block in islice(region.blocks, cfg.max_blocks):
        proposal = method.propose(block)
        metrics = tuple(ev.score(block, proposal) for ev in evals)
        results.append(Result(block=block, proposal=proposal, metrics=metrics))
    return results


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    results = run(cfg)
    for r in results:
        log.info("%s %s", r.block.block_id, {m.eval: dict(m.values) for m in r.metrics})
    if cfg.render.enabled:
        render_results(results, Path(HydraConfig.get().runtime.output_dir), cfg.render)


if __name__ == "__main__":
    main()
```

Note what's gone: the `render_base` param, `render_dir`, `_render_block`, `_kcomplexity_metrics`, `_KCOMPLEXITY`, and the `matplotlib`/`render`/`Block`/`Metrics`/`Proposal` imports that only the render path used. `render_results` accepts `cfg.render` (a Hydra `DictConfig` node here) — it reads `.format`/`.layout`/nothing-else, so the `DictConfig` duck-types as a `RenderConfig`.

- [ ] **Step 5: Update the README recipes**

In `README.md`, the "Generate before/after visuals for one block" recipe: change `render_dir=renders` to `render.enabled=true`, and update the output-path sentence to say the PNGs land in the run dir (`outputs/ct-flagship/ZAF.9.3.1_1_44882_before.png`, no `renders/` subdir). Apply the same `render_dir=renders` → `render.enabled=true` change to any other `reblock.run` recipe in the README.

- [ ] **Step 6: Run the full check**

Run: `pixi run check`
Expected: PASS — ruff clean, `mypy --strict` clean (the render-only imports are gone from `run.py`), pytest green (the purity test passes; CLI tests render into the run dir; `test_hydra_compose_*` still wire and run).

- [ ] **Step 7: Commit**

```bash
git add src/reblock/run.py conf/config.yaml README.md tests/test_run.py
git commit -m "$(cat <<'EOF'
refactor: run() is pure; rendering moves to the render emitter (F1)

Remove render coupling from run() (render_base/render_dir/_render_block); run()
now returns Results and writes nothing. main logs a per-Result summary and, when
render.enabled, calls reblock.emit.render_results into the Hydra run dir. Adds
the conf render block, drops RunConfig.render_dir, migrates the CLI/wiring tests
to render.enabled, and adds a purity test (deterministic + no global RNG
side-effect). README recipes updated.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

## Self-Review

**Spec coverage (F1 slice of the atomic-flow spec):**
- §1 "drop method list → single" → Task 3. ✓
- §1 "remove render/output-dir coupling from run()" → Task 4. ✓
- §1 "TopologyMethod local RNG / no global side-effect" → Task 1 (save/restore, deviation from literal "threaded Generator" documented — vendored `WeightedPick` uses global `np.random`). ✓
- §1 "run() deterministic on repeat" → Task 4 purity test. ✓
- §2 "render emitter lifted from `_render_block`, `RenderConfig`" → Task 2 (F1: `png`/`separate`; registry deferred to F4). ✓
- §3 "Hydra only at `main`; run() pure function" → Task 4. ✓
- §8 testing (`run()` pure/deterministic/writes-no-files; render emitter filenames; disabled emitter) → Tasks 2+4. ✓
- §9 migration (`Result.metrics` stays a tuple; multi-method tests migrate; no dual path) → Tasks 3+4. ✓
- Out of F1 (deferred): L2 cache (F2), screen stage + flagged-map (F3), compare/scorecard/sweep + emitter registry (F4). ✓

**Placeholder scan:** none — every code step shows full code; no TBD/TODO.

**Type consistency:** `RenderConfig`/`render_results` signatures match between Task 2 (definition) and Task 4 (call site: `render_results(results, Path(...), cfg.render)`). `run(cfg) -> list[Result]` (no `render_base`) consistent across Tasks 3→4. `cfg.method` is a single dict in `conf/method/*.yaml`, `RunConfig.__post_init__`, the override test, and `instantiate(cfg.method)` — all aligned. `cfg.eval` stays a list everywhere.
