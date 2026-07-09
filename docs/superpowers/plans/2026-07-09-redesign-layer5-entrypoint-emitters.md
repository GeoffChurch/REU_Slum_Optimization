# Redesign Layer 5 — typed entrypoint + screen stage + emitters — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Realize **D3** of the content-addressed dataflow redesign: the Screen becomes a pipeline stage (`selection = screen.select(source)`), `run()` becomes pure typed composition over a `PipelineSpec` (Hydra only at the edge, no `cfg` mutation), and the standalone `reblock.screen` app is replaced by one command that does detect → reblock → render + city flagged-map.

**Architecture:** The pure core moves to `reblock.pipeline`: `PipelineSpec(source, screen, method, evals, max_blocks)` → `run(spec) -> RunOutput`. `run` screens the source for the retained `selection`, samples it, builds only the sample, reblocks each. `reblock.run` becomes the thin Hydra **edge**: `spec_from_cfg(cfg) -> PipelineSpec` adapts the composed config into typed stages, and `main` fires the opt-in emitters (`render` + `flagged_map` + `flagged_blocks.txt`) into the run dir. The core never sees a `DictConfig`.

**Tech Stack:** Python 3.12, Hydra (`_target_` + `instantiate` + `${...}` interpolation), geopandas/shapely 2.1, matplotlib, joblib, pixi, pytest, `mypy --strict`, ruff.

## Global Constraints

- `pixi run check` stays green — `ruff check` + `ruff format --check` + `mypy --strict src tests scripts/crossblock_probe.py` + `pytest`. Suite is currently **139 tests**.
- **No dual path / no compat shim** (owner directive): `RunConfig` is **deleted** and every caller migrated to `PipelineSpec`; the standalone `reblock.screen` app (`__main__.py` + `conf/screen_config.yaml` + `tests/screen/test_screen_app.py`) is **deleted**, its `flagged_blocks.txt` logic migrated into `main`. `Screen.select()` changes signature outright (no overload).
- **Screen is a pipeline stage inside `run()`** — `selection = spec.screen.select(spec.source)` (spec §4). The full selection is retained in `RunOutput.selection` (spec §5) so the flagged-map gets all of it, not just the sampled results.
- **`Selection` is `list[str] | None`** (`None` = ALL), **not** the spec's literal `frozenset[str] | ALL`. This is a deliberate, *forced* departure: L4's `sample(selection, n) = selection[:n]` and its reviewed priority-ordered / no-backfill semantics require an **ordered sequence**; a frozenset has no order. `None` is the ALL sentinel. This keeps one selection representation across `RunConfig`→`PipelineSpec`, `sample`, and `RunOutput` (the value L4 already shipped). Do not introduce a second selection type.
- **Pure core, config at the edge** (spec §6, D3): `reblock.pipeline` imports **no** `hydra`/`omegaconf`; all `instantiate`/`DictConfig` handling lives in `reblock.run.spec_from_cfg`. No `cfg` mutation as a message bus.
- **Behavior is unchanged** for existing runs: `screen=identity` (the new default) is a passthrough returning the configured `${block_ids}` (or `None`), so a plain reblock and the `block_ids=[...]` recipe behave exactly as today. The pinned kblock/Phule values are unchanged.
- **The screen's fine pass shares the derivation cache**: `DenseCompactScreen`'s mean-depth gate routes through `reblock.derivations.access_before` (a `derive()` call) rather than calling `parcel_access_layers` directly, so the survivor blocks it builds are L1 hits when `run()` later scores them (spec §3 "double-build becomes free").
- Commit trailers on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
  ```

---

### Task 1: `Screen.select(source)` stage — protocol + `IdentityScreen` + `DenseCompactScreen`; delete the standalone app

**Files:**
- Modify: `src/reblock/contracts.py` (the `Screen` protocol)
- Create: `src/reblock/screen/identity.py`
- Create: `conf/screen/identity.yaml`
- Modify: `src/reblock/screen/dense_compact.py`
- Modify: `conf/screen/dense_compact.yaml` (comment only)
- Delete: `src/reblock/screen/__main__.py`, `conf/screen_config.yaml`, `tests/screen/test_screen_app.py`
- Test: `tests/screen/test_identity.py` (new), `tests/screen/test_dense_compact.py` (migrate)

**Interfaces:**
- Consumes: `reblock.contracts.Source`; `KblockSource` (exposes `.blocks_path` / `.buildings_path`); `reblock.derivations.access_before`.
- Produces:
  - `Screen.select(self, source: Source) -> list[str] | None` (protocol).
  - `IdentityScreen(block_ids: list[str] | None = None)`; `select(source)` returns the configured `block_ids` (or `None` = ALL), ignoring `source`.
  - `DenseCompactScreen(*, density_min=30.0, mean_depth_min=1.3, k_min=None, min_buildings=10)` (thresholds only, **no path args**); `select(source)` reads `source.blocks_path` / `source.buildings_path`.

**Why the app is deleted here:** the standalone app's only job was to inject paths and call `screen.select()`; changing `select` to take the `Source` breaks its `screen.select()` call (and `test_screen_app.py`). The app is obsolete once the Screen is a pipeline stage, so it is deleted in the same change that breaks it. Its `flagged_blocks.txt` output is re-created in `main` in Task 3.

- [ ] **Step 1: Write the failing test for `IdentityScreen`**

Create `tests/screen/test_identity.py`:

```python
from reblock.screen.identity import IdentityScreen


class _StubSource:
    def region(self):  # satisfies Source structurally; unused by IdentityScreen
        raise NotImplementedError


def test_identity_passthrough_returns_configured_block_ids() -> None:
    assert IdentityScreen(["a", "b"]).select(_StubSource()) == ["a", "b"]


def test_identity_default_is_none_meaning_all() -> None:
    assert IdentityScreen().select(_StubSource()) is None


def test_identity_copies_the_list_defensively() -> None:
    src = ["a", "b"]
    out = IdentityScreen(src).select(_StubSource())
    assert out == src and out is not src
```

- [ ] **Step 2: Migrate the `DenseCompactScreen` tests to `select(source)`**

Rewrite `tests/screen/test_dense_compact.py` so the screen is constructed with thresholds only and `select(source)` is called with a `KblockSource`. `_cheap_survivors` is still a direct method (unit-tested as-is). Full new file:

```python
from pathlib import Path

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Point, box

from reblock.data.kblock import KblockSource
from reblock.screen.dense_compact import DenseCompactScreen

ROOT = Path(__file__).resolve().parents[1]
CT_BLOCKS = str(ROOT / "data" / "kblock" / "blocks_capetown_sample.parquet")
CT_BLD = str(ROOT / "data" / "kblock" / "buildings_capetown_sample.parquet")
UTM = CRS.from_epsg(32734)     # Cape Town UTM: valid metric coords, KblockSource reprojects cleanly
EX, NY = 3.0e5, 6.25e6          # a realistic easting/northing base


def _write_synth(tmp: Path) -> tuple[str, str]:
    # A: dense + DEEP (5x5 grid in a 50x50 m block -> ring depths 1/2/3, mean depth 1.4);
    # B: dense but SHALLOW (30 buildings in two rows of a 30x2 m block -> all front a
    #    street, mean 1.0);
    # C: SPARSE (density 22/ha -> fails the cheap gate outright).
    a = box(EX, NY, EX + 50, NY + 50)
    b = box(EX + 70, NY, EX + 100, NY + 2)
    c = box(EX + 120, NY, EX + 150, NY + 30)
    blocks = gpd.GeoDataFrame({
        "block_id": ["A", "B", "C"], "k_complexity": [3.0, 2.0, 1.0],
        "building_count": [25, 30, 2], "block_area_m2": [2500.0, 60.0, 900.0],
    }, geometry=[a, b, c], crs=UTM)
    pts = [Point(EX + 5 + 10 * i, NY + 5 + 10 * j) for i in range(5) for j in range(5)]  # A: 5x5
    pts += [Point(EX + 71 + 2 * i, NY + row) for i in range(15) for row in (0.5, 1.5)]   # B: 2 rows
    pts += [Point(EX + 125, NY + 5), Point(EX + 140, NY + 20)]                            # C: 2
    bld = gpd.GeoDataFrame(geometry=pts, crs=UTM)
    bp, dp = tmp / "b.parquet", tmp / "d.parquet"
    blocks.to_parquet(bp)
    bld.to_parquet(dp)
    return str(bp), str(dp)


def test_cheap_survivors_gate(tmp_path: Path) -> None:
    bp, _ = _write_synth(tmp_path)
    s = DenseCompactScreen(density_min=50.0, min_buildings=10)
    # density/ha: A=25/(2500/1e4)=100, B=30/(60/1e4)=5000, C=2/(900/1e4)=22
    assert s._cheap_survivors(gpd.read_parquet(bp)) == ["A", "B"]   # C (22) fails; sorted


def test_select_two_tier_drops_shallow(tmp_path: Path) -> None:
    bp, dp = _write_synth(tmp_path)
    # cheap keeps A,B; fine gate mean_depth_min=1.2 keeps A (deep, ~1.4), drops B (strip, ~1.0)
    s = DenseCompactScreen(density_min=50.0, mean_depth_min=1.2, min_buildings=10)
    src = KblockSource(bp, dp, region_id="test", min_buildings=10)
    assert s.select(src) == ["A"]


def test_select_flags_flagship_on_real_fixture() -> None:
    # density_min=35.0 is the smallest round threshold that clears the flagship's real
    # column-based density (~35.6/ha over the free building_count/block_area_m2 columns;
    # see git history / PROVENANCE for why it is not the ~108/ha spatial-join figure).
    s = DenseCompactScreen(density_min=35.0, mean_depth_min=1.3)
    src = KblockSource(CT_BLOCKS, CT_BLD, region_id="capetown")
    ids = s.select(src)
    assert ids is not None and "ZAF.9.3.1_1_44882" in ids and ids == sorted(ids)
```

- [ ] **Step 3: Run to verify failure**

Run: `pixi run pytest tests/screen/test_identity.py tests/screen/test_dense_compact.py -v`
Expected: FAIL — `No module named 'reblock.screen.identity'`; `DenseCompactScreen.__init__` still requires positional `blocks_path`.

- [ ] **Step 4: Change the `Screen` protocol**

In `src/reblock/contracts.py`, change the `Screen` protocol (currently `def select(self) -> list[str]: ...`) to:

```python
class Screen(Protocol):
    def select(self, source: Source) -> list[str] | None: ...   # block_ids (sorted), or None => all
```

(`Source` is already defined above it in this module.)

- [ ] **Step 5: Create `IdentityScreen` + its config**

Create `src/reblock/screen/identity.py`:

```python
"""IdentityScreen: the passthrough Screen (run()'s default). Selects nothing of its
own -- returns the configured block_ids (or None => all blocks), so a run with no
real screen behaves exactly as a plain reblock."""
from __future__ import annotations

from reblock.contracts import Source


class IdentityScreen:
    def __init__(self, block_ids: list[str] | None = None) -> None:
        self.block_ids = list(block_ids) if block_ids is not None else None

    def select(self, source: Source) -> list[str] | None:
        del source   # a passthrough needs no data
        return list(self.block_ids) if self.block_ids is not None else None
```

Create `conf/screen/identity.yaml`:

```yaml
# The passthrough Screen (run()'s default): selects the configured block_ids (or
# all blocks). block_ids is interpolated from the top-level ${block_ids}, so the
# `block_ids=[...]` CLI override flows through the identity screen unchanged.
_target_: reblock.screen.identity.IdentityScreen
block_ids: ${block_ids}
```

- [ ] **Step 6: Migrate `DenseCompactScreen` to `select(source)`**

Rewrite `src/reblock/screen/dense_compact.py`:

```python
"""DenseCompactScreen: flag dense/compact informal blocks. Cheap pass = vectorized
density (+ optional k) gate over free kblock columns; fine pass = build only survivors
(reusing the source's KblockSource paths) and keep those whose mean parcel access-depth
clears mean_depth_min. The fine-pass depth goes through reblock.derivations.access_before
(a derive() call), so building a survivor here is an L1 hit when run() later scores it.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd

from reblock.contracts import Source
from reblock.data.kblock import KblockSource
from reblock.derivations import access_before

log = logging.getLogger(__name__)


class DenseCompactScreen:
    def __init__(self, *, density_min: float = 30.0, mean_depth_min: float = 1.3,
                 k_min: float | None = None, min_buildings: int = 10) -> None:
        self.density_min = density_min
        self.mean_depth_min = mean_depth_min
        self.k_min = k_min
        self.min_buildings = min_buildings

    def _cheap_survivors(self, blocks: gpd.GeoDataFrame) -> list[str]:
        bid = blocks["block_id"].astype(str)
        density: pd.Series = blocks["building_count"] / (blocks["block_area_m2"] / 1e4)
        mask: pd.Series = density >= self.density_min
        if self.k_min is not None:
            mask = mask & (blocks["k_complexity"] >= self.k_min)
        return sorted(bid[mask.to_numpy()])

    def select(self, source: Source) -> list[str]:
        if not isinstance(source, KblockSource):
            raise TypeError(
                f"DenseCompactScreen needs a KblockSource (kblock columns); "
                f"got {type(source).__name__}")
        blocks = gpd.read_parquet(
            source.blocks_path,
            columns=["block_id", "k_complexity", "building_count", "block_area_m2", "geometry"])
        survivors = self._cheap_survivors(blocks)
        log.info("cheap pass: %d/%d blocks pass density_min=%.1f%s",
                 len(survivors), len(blocks), self.density_min,
                 f", k_min={self.k_min}" if self.k_min is not None else "")
        if not survivors:
            return []
        log.info("fine pass: building %d survivor blocks (Voronoi + peel) -- the slow step",
                 len(survivors))
        src = KblockSource(source.blocks_path, source.buildings_path, region_id="screen",
                           min_buildings=self.min_buildings, block_ids=survivors)
        kept = [blk.block_id for blk in src.region().blocks
                if float(access_before(blk).mean()) >= self.mean_depth_min]
        log.info("fine pass: kept %d blocks with mean access-depth >= %.2f",
                 len(kept), self.mean_depth_min)
        return sorted(kept)
```

(`select` returns `list[str]`, a subtype of the protocol's `list[str] | None`.) Update the comment on line 1 of `conf/screen/dense_compact.yaml` — drop "Paths are injected by the reblock.screen app" (paths now come from the source at `select()` time); keep the `_target_` + thresholds unchanged:

```yaml
# A Screen (selectable like conf/method, conf/eval). Reads paths from the run's
# Source (a KblockSource) at select() time; the thresholds below are the gates.
_target_: reblock.screen.dense_compact.DenseCompactScreen
density_min: 30.0
mean_depth_min: 1.3
# k_min: null   # optional extra cheap gate; off by default (density + mean_depth do the work)
```

- [ ] **Step 7: Delete the standalone app**

```bash
git rm src/reblock/screen/__main__.py conf/screen_config.yaml tests/screen/test_screen_app.py
```

- [ ] **Step 8: Run tests + full check**

Run: `pixi run pytest tests/screen -v` then `pixi run check`
Expected: PASS. `IdentityScreen` + migrated `DenseCompactScreen` satisfy `Screen.select(source)`; the flagship survives at `density_min=35`; the deleted app leaves no dangling import (`grep -rn "reblock.screen.__main__\|screen_config" src tests conf` → none). `run.py`/`main` are untouched (still on the old `run(cfg)` path — rewired in Task 2), so the CLI subprocess tests still pass. ~138 tests (−3 app tests, +3 identity tests, dense_compact count unchanged).

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat: Screen.select(source) stage + IdentityScreen; delete standalone app (redesign L5)

Screen.select now takes the Source and returns list[str] | None (None => all), so
the Screen is a Source->selection pipeline stage. IdentityScreen is the passthrough
default (returns configured ${block_ids}); DenseCompactScreen reads paths off the
source and routes its fine-pass depth through derivations.access_before so survivor
builds are L1 hits at scoring time. Deletes the now-obsolete standalone reblock.screen
app + screen_config (its flagged_blocks.txt output returns in run.main next).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

### Task 2: `PipelineSpec` + pure `run(spec)` in `reblock.pipeline`; `spec_from_cfg` edge; delete `RunConfig`

**Files:**
- Modify: `src/reblock/pipeline.py` (add `PipelineSpec`; move `run()` here as pure typed composition)
- Modify: `src/reblock/run.py` (becomes the edge: `spec_from_cfg` + `main`; delete `RunConfig`; `run()` no longer lives here)
- Modify: `conf/config.yaml` (add `screen: identity` default)
- Migrate: `tests/test_run.py` (all `RunConfig`/`run(cfg)` callers), `scripts/bench_cache.py`

**Interfaces:**
- Consumes: `reblock.contracts.{Source,Screen,Method,Eval}`; `reblock.derivations.propose`.
- Produces:
  - `PipelineSpec(source: Source, screen: Screen, method: Method, evals: list[Eval], max_blocks: int = 1)` (frozen).
  - `run(spec: PipelineSpec) -> RunOutput` (in `reblock.pipeline`; pure, no Hydra).
  - `spec_from_cfg(cfg: DictConfig) -> PipelineSpec` (in `reblock.run`; the only `instantiate`/`DictConfig` adapter).

**This is one atomic change:** `run()`'s signature changes and every caller must move at once (no dual path). The screen stage enters `run()` here too.

- [ ] **Step 1: Migrate `tests/test_run.py` to `PipelineSpec` / `spec_from_cfg`**

Replace the whole imports + programmatic-caller region. New imports at the top of `tests/test_run.py`:

```python
from reblock.contracts import Block, Eval, Result
from reblock.data.shapefile import ShapefileSource
from reblock.eval.kcomplexity import KComplexityEval, WeakDualKEval
from reblock.methods.topology import TopologyMethod
from reblock.pipeline import PipelineSpec, run
from reblock.run import spec_from_cfg
from reblock.screen.identity import IdentityScreen
```

Add a small helper (below the `_grid_block` helper) that builds the common Phule spec.
The param is typed `list[Eval]` (NOT `list[object]`: `mypy --strict` rejects passing a
`list[object]` into `PipelineSpec.evals: list[Eval]` — `list` is invariant; the call-site
list literals infer as `list[Eval]` against this param and pass cleanly):

```python
def _phule_spec(evals: list[Eval], max_blocks: int = 1) -> PipelineSpec:
    return PipelineSpec(
        source=ShapefileSource(PHULE, region_id="phule", assumed_crs=3857),
        screen=IdentityScreen(),
        method=TopologyMethod(alpha=2.0, seed=0),
        evals=evals,
        max_blocks=max_blocks,
    )
```

Migrate the four in-process tests:

`test_end_to_end_phule_wiring`:
```python
def test_end_to_end_phule_wiring() -> None:
    # Wiring proof on real data: run() returns well-formed Results and writes
    # nothing (rendering is an emitter now, exercised by the CLI test below).
    results = run(_phule_spec([KComplexityEval()])).results
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, Result)
    assert r.block.block_id == "phule_0"
    assert r.metric("kcomplexity", "k_after") <= r.metric("kcomplexity", "k_before")
    assert r.metric("kcomplexity", "delta_k") >= 0
```

`test_run_is_pure_deterministic_and_leaves_global_rng_untouched`:
```python
def test_run_is_pure_deterministic_and_leaves_global_rng_untouched() -> None:
    import numpy as np
    spec = _phule_spec([KComplexityEval()])
    np.random.seed(777)
    state_before = np.random.get_state()[1].tolist()
    r1 = run(spec).results
    r2 = run(spec).results
    assert np.random.get_state()[1].tolist() == state_before   # no global RNG side-effect
    assert [x.proposal.proposal_id for x in r1] == [x.proposal.proposal_id for x in r2]
    assert (r1[0].metric("kcomplexity", "delta_k")
            == r2[0].metric("kcomplexity", "delta_k"))
```

Replace `test_runconfig_accepts_explicit_data_method_eval_overrides` (its subject, `RunConfig`'s flat-vs-explicit dual path, is gone) with a multi-eval test through the typed spec — preserving the multi-eval coverage:

```python
def test_multiple_evals_through_the_pipeline_spec() -> None:
    # PipelineSpec.evals is a plain list of typed Evals: run() scores each block
    # with every eval, so a block's metrics carry one Metrics per eval.
    results = run(_phule_spec([KComplexityEval(), WeakDualKEval()])).results
    assert len(results) == 1
    r = results[0]
    assert {m.eval for m in r.metrics} == {"kcomplexity", "weakdual_k"}
    assert r.metric("kcomplexity", "k_after") <= r.metric("kcomplexity", "k_before")
```

Migrate the four `compose(...)`-based tests: each currently does `results = run(cfg).results`; change to `results = run(spec_from_cfg(cfg)).results`. The four are `test_hydra_compose_wires_config_groups`, `test_hydra_compose_wires_peel_method`, `test_hydra_compose_wires_kblock_source_and_peel_pipeline`, and `test_block_ids_targets_one_capetown_block_through_the_pipeline`. Keep every assertion; only the `run(cfg)` → `run(spec_from_cfg(cfg))` line changes. (The two `subprocess`-based CLI tests and `test_topology_reblocks_a_synthetic_nested_block` are unchanged.)

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_run.py -v`
Expected: FAIL — `cannot import name 'PipelineSpec' from 'reblock.pipeline'` / `cannot import name 'spec_from_cfg' from 'reblock.run'`.

- [ ] **Step 3: Add `PipelineSpec` + move `run()` into `reblock.pipeline`**

Rewrite `src/reblock/pipeline.py` in full (adds `PipelineSpec`, moves `run` here, keeps `RunOutput`/`reblock_block`/`sample`; **no** hydra/omegaconf import):

```python
"""The dataflow pipeline core (pure typed composition -- no Hydra, no DictConfig;
config lives at the edge in reblock.run). PipelineSpec bundles the typed stages;
run() threads them: screen.select(source) yields the retained selection, sample
splits "how many" from "which", each picked block goes through reblock_block
(propose + score). See docs/superpowers/specs/2026-07-08-content-addressed-dataflow-redesign.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

from reblock.contracts import Block, Eval, Method, Result, Screen, Source
from reblock.derivations import propose


@dataclass(frozen=True)
class PipelineSpec:
    """The typed stages of one run, composed at the edge (reblock.run.spec_from_cfg)
    from Hydra config, or directly in Python. The core pipeline (run) is exactly a
    function of this value -- it never sees a DictConfig."""
    source: Source
    screen: Screen
    method: Method
    evals: list[Eval]
    max_blocks: int = 1


@dataclass(frozen=True)
class RunOutput:
    selection: list[str] | None   # the full block selection (None = all blocks)
    results: list[Result]         # one per reblocked (sampled) block


def reblock_block(block: Block, method: Method, evals: list[Eval]) -> Result:
    """One block through method + evals -> a Result (metrics tuple over the evals)."""
    proposal = propose(method, block)
    metrics = tuple(ev.score(block, proposal) for ev in evals)
    return Result(block=block, proposal=proposal, metrics=metrics)


def sample(selection: list[str] | None, n: int) -> list[str] | None:
    """The first `n` block_ids to actually build/reblock. `None` (ALL) passes
    through -- the caller then islices the built region to `n`.

    `block_ids` is treated as a **priority-ordered** selection: `sample` takes
    the first `n` in order (the screen returns sorted ids; a caller's explicit
    list is its own priority). If a sampled block fails to build (e.g. too few
    building points for a valid Voronoi cell), it is skipped and the run yields
    **fewer than `n`** results -- there is no silent backfill from later in the
    selection. This is intentional: it builds only what it reblocks (the redesign
    keeps selection and sampling separate), and the screen feeds pre-verified
    survivors, so the shortfall case does not arise there."""
    return selection[:n] if selection is not None else None


def run(spec: PipelineSpec) -> RunOutput:
    """The dataflow pipeline: screen the source for the selection, sample it, build
    only the sample, reblock each -> RunOutput(selection, results). The full
    selection is retained (results cover only the sampled max_blocks). Pure: reads
    its inputs and returns a value; writes nothing (emitters, at the edge, write)."""
    selection = spec.screen.select(spec.source)
    picked = sample(selection, spec.max_blocks)
    if picked is not None:
        # block_ids is a kblock-specific mutable filter, not part of the Source
        # Protocol (ShapefileSource has none); the assignment is guarded by
        # `picked is not None`, so it is only reached for kblock-backed runs.
        spec.source.block_ids = picked  # type: ignore[attr-defined]
        blocks = spec.source.region().blocks
    else:
        blocks = islice(spec.source.region().blocks, spec.max_blocks)   # ALL -> islice
    results = [reblock_block(block, spec.method, spec.evals) for block in blocks]
    return RunOutput(selection=selection, results=results)
```

- [ ] **Step 4: Rewrite `reblock.run` as the Hydra edge (delete `RunConfig`)**

Replace `src/reblock/run.py` in full — `spec_from_cfg` + `main`; `RunConfig` and the old `run()` are gone (this task's `main` does not yet fire the flagged-map; Task 3 widens it):

```python
"""Hydra entrypoint (the config edge): parse the conf/ config groups into a typed
PipelineSpec, run the pure pipeline (reblock.pipeline.run), then fire the opt-in
emitters into the Hydra run dir. The core pipeline never sees this DictConfig --
spec_from_cfg is the only adapter.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from reblock.contracts import Eval, Method, Screen, Source
from reblock.emit import render_results
from reblock.pipeline import PipelineSpec, run

log = logging.getLogger(__name__)


def spec_from_cfg(cfg: DictConfig) -> PipelineSpec:
    """Adapt a composed Hydra config into a typed PipelineSpec (the config edge).
    Per-element instantiate for the eval LIST: instantiate(cfg.eval) whole would
    short-circuit a ListConfig of @dataclass _target_s to schema-validated
    DictConfig nodes instead of constructing them; cfg.data/screen/method are
    single _target_ dicts, so instantiate(...) on each is safe."""
    return PipelineSpec(
        source=cast(Source, instantiate(cfg.data)),
        screen=cast(Screen, instantiate(cfg.screen)),
        method=cast(Method, instantiate(cfg.method)),
        evals=cast("list[Eval]", [instantiate(e) for e in cfg.eval]),
        max_blocks=cfg.max_blocks,
    )


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    spec = spec_from_cfg(cfg)
    output = run(spec)
    for r in output.results:
        log.info("%s %s", r.block.block_id, {m.eval: dict(m.values) for m in r.metrics})
    if cfg.render.enabled:
        render_results(output.results, Path(HydraConfig.get().runtime.output_dir), cfg.render)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Add `screen: identity` to the config defaults**

In `conf/config.yaml`, add `screen: identity` to the `defaults` list (after `data`):

```yaml
defaults:
  - data: phule
  - screen: identity
  - method: topology
  - eval: kcomplexity
  - _self_
```

(The existing top-level `block_ids: null` is what `conf/screen/identity.yaml` interpolates via `${block_ids}`.)

- [ ] **Step 6: Migrate `scripts/bench_cache.py`**

`bench_cache.py` imports the deleted `RunConfig`; migrate it to `PipelineSpec` and point its cache-clear/size at `reblock.derive_graph` (the live cache; `reblock.cache` is orphaned after L3 and deleted in L6). Full new file:

```python
"""Benchmark the derive() L2 cache: cold (cleared) vs warm wall-time for a real
Cape Town multi-block reblock, plus the derivation cache's disk footprint.
Usage: PYTHONPATH=. pixi run python scripts/bench_cache.py
"""
from __future__ import annotations

import time
from pathlib import Path

from reblock import derive_graph
from reblock.data.kblock import KblockSource
from reblock.data.provision import ensure_city_data
from reblock.eval.kcomplexity import KComplexityEval
from reblock.methods.peel import PeelReblocker
from reblock.pipeline import PipelineSpec, run
from reblock.screen.identity import IdentityScreen

BLOCK_IDS = ["ZAF.9.3.1_1_44882", "ZAF.9.3.1_1_42413", "ZAF.9.3.1_1_21255"]


def _dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _timed_run(blocks_path: Path, buildings_path: Path) -> float:
    spec = PipelineSpec(
        source=KblockSource(str(blocks_path), str(buildings_path),
                            region_id="capetown", block_ids=BLOCK_IDS),
        screen=IdentityScreen(BLOCK_IDS),
        method=PeelReblocker(),
        evals=[KComplexityEval()],
        max_blocks=len(BLOCK_IDS),
    )
    t0 = time.perf_counter()
    run(spec)
    return time.perf_counter() - t0


def main() -> None:
    blocks_path, buildings_path = ensure_city_data("capetown")
    cache_dir = Path(derive_graph._CACHE_DIR)

    derive_graph.memory.clear(warn=False)
    derive_graph.clear_l1()
    cold = _timed_run(blocks_path, buildings_path)
    cold_disk = _dir_bytes(cache_dir)

    warm = _timed_run(blocks_path, buildings_path)
    warm_disk = _dir_bytes(cache_dir)

    print(f"blocks: {len(BLOCK_IDS)}  method=peel")
    print(f"COLD (cache cleared): {cold:6.2f}s")
    print(f"WARM (cache hit):     {warm:6.2f}s   speedup {cold / warm:4.1f}x")
    print(f"cache disk: {cold_disk/1e6:6.2f} MB after cold, {warm_disk/1e6:6.2f} MB after warm")


if __name__ == "__main__":
    main()
```

Verify (no network / not in `pixi run check`): `pixi run ruff check scripts/bench_cache.py` — clean (catches undefined names / unused imports).

- [ ] **Step 7: Run the full suite**

Run: `pixi run check`
Expected: PASS. `mypy --strict` clean (`PipelineSpec` typed; the `spec.source.block_ids` `type: ignore[attr-defined]` is the only one). The in-process tests build `PipelineSpec` directly; the compose tests go through `spec_from_cfg`; the CLI subprocess tests drive the real `main`. Pinned kblock/Phule values unchanged. `grep -rn "RunConfig" src tests scripts` → none.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: PipelineSpec + pure run(spec) in reblock.pipeline; Hydra only at the edge (redesign L5)

run() moves to reblock.pipeline as pure typed composition over a PipelineSpec
(source, screen, method, evals, max_blocks) and gains the screen stage
(selection = screen.select(source)); the pipeline module imports no Hydra.
reblock.run is now the config edge: spec_from_cfg adapts a DictConfig into a
PipelineSpec, main runs it + renders. RunConfig is deleted; test_run and
bench_cache build PipelineSpecs (or spec_from_cfg(compose(...))) directly.
No cfg mutation, no dual path. Pinned values unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

### Task 3: `flagged_map` + `flagged_blocks.txt` emitters; one-command end-to-end + README

**Files:**
- Modify: `src/reblock/emit.py` (add `FlaggedMapConfig` + `flagged_map`)
- Modify: `src/reblock/run.py` (`main` fires `flagged_blocks.txt` + `flagged_map`)
- Modify: `conf/config.yaml` (add `flagged_map` block)
- Modify: `README.md` (replace the two-step detect recipe with the one command)
- Test: `tests/test_run.py` (screen-stage end-to-end + `flagged_map` unit test)

**Interfaces:**
- Consumes: `reblock.render.save_render`; `output.selection` (from `RunOutput`); `spec.source.blocks_path`.
- Produces: `flagged_map(blocks_path: str, flagged_ids: list[str], out_dir: Path) -> Path | None`. (No `FlaggedMapConfig`: unlike `RenderConfig`, the map has no draw params — its only knob is `enabled`, a caller-side gate read as `cfg.flagged_map.enabled`. Adding a one-field dataclass the function then ignores would be a wart.)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_run.py` a focused `flagged_map` unit test and a screen-stage end-to-end CLI test:

```python
def test_flagged_map_writes_png(tmp_path: Path) -> None:
    from reblock.emit import flagged_map
    blocks = str(Path(__file__).resolve().parents[0] / "data" / "kblock"
                 / "blocks_capetown_sample.parquet")
    out = flagged_map(blocks, ["ZAF.9.3.1_1_44882"], tmp_path)
    assert out is not None and out.exists() and out.stat().st_size > 0


def test_flagged_map_none_when_no_ids(tmp_path: Path) -> None:
    from reblock.emit import flagged_map
    blocks = str(Path(__file__).resolve().parents[0] / "data" / "kblock"
                 / "blocks_capetown_sample.parquet")
    assert flagged_map(blocks, [], tmp_path) is None


def test_cli_screen_stage_end_to_end(tmp_path: Path) -> None:
    # One command: screen (dense_compact) -> reblock (peel) -> render + city map.
    result = subprocess.run(
        [sys.executable, "-m", "reblock.run",
         "data=capetown", "screen=dense_compact", "screen.density_min=35",
         "method=peel", "eval=kcomplexity", "max_blocks=1",
         "render.enabled=true", "flagged_map.enabled=true",
         f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "flagged_map.png").stat().st_size > 0
    flagged = (tmp_path / "flagged_blocks.txt").read_text()
    assert "ZAF.9.3.1_1_44882" in flagged
    assert list(tmp_path.glob("*_before.png")) and list(tmp_path.glob("*_after.png"))
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_run.py -k "flagged_map or screen_stage" -v`
Expected: FAIL — `cannot import name 'flagged_map' from 'reblock.emit'`; `cfg.flagged_map` missing in config.

- [ ] **Step 3: Add the `flagged_map` emitter to `emit.py`**

Add to `src/reblock/emit.py` (module already imports `save_render` from `reblock.render` and `plt`):

```python
def flagged_map(blocks_path: str, flagged_ids: list[str], out_dir: Path) -> Path | None:
    """Binary city choropleth: every metro block drawn light, the flagged ones
    highlighted. Re-reads the blocks parquet geometry (kept out of the Screen so it
    stays a pure selector). Returns the written path, or None if there are no ids.
    Gating is the caller's (cfg.flagged_map.enabled)."""
    import geopandas as gpd
    if not flagged_ids:
        return None
    blocks = gpd.read_parquet(blocks_path, columns=["block_id", "geometry"])
    blocks["block_id"] = blocks["block_id"].astype(str)
    blocks["flagged"] = blocks["block_id"].isin(set(flagged_ids))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    blocks[~blocks["flagged"]].plot(ax=ax, color="#e8e8e8", edgecolor="none")
    blocks[blocks["flagged"]].plot(ax=ax, color="#c0392b", edgecolor="none")
    ax.set_title(f"{len(flagged_ids)} flagged blocks")
    ax.set_axis_off()
    out_path = out_dir / "flagged_map.png"
    save_render(fig, out_path)
    plt.close(fig)
    return out_path
```

- [ ] **Step 4: Fire the emitters in `main`**

In `src/reblock/run.py`, widen the emit import and add the `flagged_blocks.txt` + `flagged_map` emitters to `main`:

```python
from reblock.emit import flagged_map, render_results  # widen
```

`main` becomes:

```python
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    spec = spec_from_cfg(cfg)
    output = run(spec)
    for r in output.results:
        log.info("%s %s", r.block.block_id, {m.eval: dict(m.values) for m in r.metrics})

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    if output.selection is not None:
        flagged_path = out_dir / "flagged_blocks.txt"
        flagged_path.write_text("".join(f"{b}\n" for b in output.selection))
        log.info("%d blocks flagged -> %s", len(output.selection), flagged_path)
    if cfg.render.enabled:
        render_results(output.results, out_dir, cfg.render)
    if cfg.flagged_map.enabled:
        blocks_path = getattr(spec.source, "blocks_path", None)
        if blocks_path is None:
            log.warning("flagged_map: source %s has no blocks_path; skipping",
                        type(spec.source).__name__)
        else:
            flagged_map(str(blocks_path), output.selection or [], out_dir)
```

(`output.selection or []`: an IdentityScreen `None` selection passes `[]`, so `flagged_map` returns `None` and no map is drawn — a plain reblock has no city map.)

- [ ] **Step 5: Add the `flagged_map` block to `conf/config.yaml`**

After the `render:` block:

```yaml
# City flagged-map emitter (reblock.emit.flagged_map): opt-in; draws all metro
# blocks light with the screen's flagged blocks highlighted, into the run dir.
flagged_map:
  enabled: false
```

- [ ] **Step 6: Replace the README detect recipe**

In `README.md`, replace the "## Detect informal settlements (Screen)" section (lines ~39-49, the `python -m reblock.screen ...` recipe) with the one-command end-to-end:

```markdown
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
```

- [ ] **Step 7: Run tests + full check**

Run: `pixi run check`
Expected: PASS. The end-to-end test writes `flagged_map.png` + `flagged_blocks.txt` + reblock PNGs; the `flagged_map` unit tests pass; no reference to `reblock.screen` (app) or `screen_config` remains (`grep -rn "reblock.screen \|screen_config\|python -m reblock.screen" README.md src tests conf` → none). ~141 tests.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat: flagged_map + flagged_blocks.txt emitters; one-command end-to-end (redesign L5)

emit.flagged_map draws the city choropleth (all blocks light, flagged highlighted)
from the retained RunOutput.selection; main writes flagged_blocks.txt and the map
into the run dir. README collapses to one detect->reblock->render+map command;
the standalone screen app is fully replaced.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

## Self-Review

**Spec coverage (L5 = migration step 5, "Entrypoint + emitters"):**
- Screen as a pipeline stage `selection = screen.select(source)` (spec §4) → Task 1 (protocol/screens) + Task 2 (stage inside `run`). ✓
- `IdentityScreen` default passthrough → Task 1 (class) + Task 2 (config default). ✓
- Typed `PipelineSpec`; Hydra only at the edge; no `cfg` mutation (spec §6, D3) → Task 2 (`PipelineSpec`, pure `run` in `reblock.pipeline`, `spec_from_cfg`). ✓
- Sweep/emit as outer combinators consuming typed `RunOutput`; flagged-map gets the **full** selection (spec §5) → Task 3 (`flagged_map(selection)` from `output.selection`). ✓
- `render` + `flagged_map` + `flagged_blocks.txt`; one-command end-to-end (spec migration step 5) → Task 3. ✓
- Delete the old entrypoint shim (`RunConfig`) + standalone screen app → Task 2 / Task 1. ✓
- Out of L5 (L6): delete `reblock.cache` + old remnants + migrate remaining tests + merge. Out of L5 (F4): `scorecard`/`compare` + sweep.

**Deliberate departures from the spec's literal text (documented in Global Constraints):**
- `Selection` is `list[str] | None`, not `frozenset[str] | ALL` — forced by L4's ordered, priority-based `sample`. One selection representation, no second type.
- Screen selects over the **Source** (`select(source)`), not a materialized `Iterable[RawBlock]` — consequence of the L2 Option-Y decision (no `RawBlock` type; the Source still builds `Block`s; the cheap pass reads the parquet columns off `source.blocks_path`).

**Placeholder scan:** every code step is complete. The Task-1 note that the app is deleted in the same task as the breaking protocol change is a real cohesion decision (the app exists only to call `screen.select()`), not a placeholder; `flagged_blocks.txt` is re-created in Task 3's `main`.

**Type consistency:** `Screen.select(source) -> list[str] | None` is used identically in `IdentityScreen` (Task 1), `DenseCompactScreen` (Task 1, returns `list[str]`), and `run` (Task 2, `selection: list[str] | None`). `PipelineSpec(source, screen, method, evals, max_blocks)` matches `spec_from_cfg`'s constructor call and the direct `PipelineSpec(...)` in tests/bench. `run(spec) -> RunOutput` (Task 2) is what `main`, `spec_from_cfg` callers, and the migrated tests consume via `.results` / `.selection`. `flagged_map(blocks_path, flagged_ids, out_dir) -> Path | None` (Task 3) matches its `main` call site and the unit tests. `${block_ids}` interpolation (`conf/screen/identity.yaml`, Task 1) targets the existing top-level `block_ids: null` key.

**Behavior-unchanged** is guarded by: the pinned Phule/kblock assertions in the migrated `test_run.py` (Task 2), the identity-passthrough default keeping `block_ids=[...]` runs identical, and the `access_before` fine-pass swap being the same computation as `parcel_access_layers(blk, None)` (`access_before` = `derive(_access_before_impl, block)`, `_access_before_impl(block) = parcel_access_layers(block, None)`), so the flagship still clears `mean_depth_min=1.3`.
