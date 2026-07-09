# Redesign Layer 4 — the dataflow pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `run()` a dataflow pipeline that returns `RunOutput(selection, results)` — the block selection retained as a value — with **sampling split from selection** (build only the sampled blocks), so the selection is never lost to `max_blocks` truncation.

**Architecture:** A new `reblock.pipeline` module: `RunOutput` (the pipeline's typed output), `reblock_block(block, method, evals) -> Result` (the per-block stage), and `sample(selection, n)` (which block_ids to actually build). `run()` becomes: resolve source/method/evals → `sample` the selection → build only the sample → `reblock_block` each → `RunOutput(selection, results)`. The full selection (from `cfg.block_ids`, or `None` = ALL) is returned so downstream emitters (the L5 flagged-map) get all of it.

**Tech Stack:** Python 3.12, Hydra, pixi, pytest, `mypy --strict`, ruff.

## Global Constraints

- `pixi run check` stays green — `ruff check` + `ruff format --check` + `mypy --strict src tests scripts/crossblock_probe.py` + pytest. Suite is currently 135 tests.
- **`run()` return type changes** `list[Result]` → `RunOutput`. This is a redesign (no compat shim): migrate every caller (`main`, all of `tests/test_run.py`) to `.results`/`.selection`. No dual return.
- **Selection retained, sampling split** — `RunOutput.selection` is the full `cfg.block_ids` (or `None` = ALL); `results` cover only the sampled `max_blocks`. The source builds only the sampled blocks (efficiency + the clean split), not all of the selection.
- **Results are IDENTICAL** — the per-block reblock (`reblock_block`) is the current loop body extracted verbatim; pinned kblock/run values unchanged.
- `reblock_block` is the reusable per-block stage (F4's sweep will map it over blocks).
- Commit trailers on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
  ```

---

### Task 1: `reblock.pipeline` — `RunOutput`, `reblock_block`, `sample`

**Files:**
- Create: `src/reblock/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `contracts.Block`/`Method`/`Eval`/`Result`; `derivations.propose`.
- Produces:
  - `RunOutput` (frozen dataclass): `selection: list[str] | None`, `results: list[Result]`.
  - `reblock_block(block: Block, method: Method, evals: list[Eval]) -> Result` — propose (via `derivations.propose`) + score with each eval → one `Result`.
  - `sample(selection: list[str] | None, n: int) -> list[str] | None` — `selection[:n]` for a list, `None` (ALL) passes through.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline.py`:

```python
from reblock.pipeline import RunOutput, sample


def test_sample_takes_first_n_of_a_list() -> None:
    assert sample(["a", "b", "c", "d"], 2) == ["a", "b"]


def test_sample_passes_through_all() -> None:
    assert sample(None, 5) is None            # None = ALL


def test_sample_n_larger_than_selection() -> None:
    assert sample(["a", "b"], 10) == ["a", "b"]


def test_runoutput_holds_selection_and_results() -> None:
    out = RunOutput(selection=["a", "b", "c"], results=[])
    assert out.selection == ["a", "b", "c"] and out.results == []
```

(`reblock_block` is covered end-to-end by `test_run.py` in Task 2 — it needs a real source-built Block, which the `run()` path provides; a unit test here would duplicate the run() wiring.)

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_pipeline.py -v`
Expected: FAIL — `No module named 'reblock.pipeline'`.

- [ ] **Step 3: Implement `src/reblock/pipeline.py`**

```python
"""The dataflow pipeline: run() composes these stages. reblock_block is the
per-block stage (F4's sweep maps it over blocks); sample splits "how many to
reblock" from "which blocks are selected"; RunOutput carries the full selection
out so downstream emitters (the city flagged-map) get all of it, not just the
sampled results.
"""
from __future__ import annotations

from dataclasses import dataclass

from reblock.contracts import Block, Eval, Method, Result
from reblock.derivations import propose


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
    through -- the caller then islices the built region to `n`."""
    return selection[:n] if selection is not None else None
```

- [ ] **Step 4: Run to verify pass + full check**

Run: `pixi run pytest tests/test_pipeline.py -v` then `pixi run check`
Expected: PASS (4 tests). `pixi run check` green — additive module. 139 tests.

- [ ] **Step 5: Commit**

```bash
git add src/reblock/pipeline.py tests/test_pipeline.py
git commit -m "$(cat <<'EOF'
feat: reblock.pipeline -- RunOutput + reblock_block + sample (redesign L4)

The dataflow pipeline's stages: reblock_block (per-block, reusable by the F4
sweep), sample (split "how many" from "which"), and RunOutput (carries the full
selection out so emitters get all of it, not just the sampled results). Additive
-- run() is rewired to use them next.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

### Task 2: rewire `run()` to the pipeline (returns `RunOutput`)

**Files:**
- Modify: `src/reblock/run.py`
- Test: `tests/test_run.py` (migrate every `run(cfg)` caller to `.results`)

**Interfaces:**
- Consumes: `reblock.pipeline.RunOutput`/`reblock_block`/`sample`.
- Produces: `run(cfg: RunConfig | DictConfig) -> RunOutput`.

- [ ] **Step 1: Migrate the tests to `.results`**

In `tests/test_run.py`, every test currently does `results = run(cfg)` then indexes/asserts on `results`. Change each to `results = run(cfg).results` (the pipeline now returns `RunOutput`). Grep the file for `run(` calls and update each. Do NOT change the assertions on the results themselves — only unwrap `.results`. (There are ~8 such tests: the phule-wiring, override, hydra-compose ×4, block_ids, and purity tests.) The CLI subprocess tests call `python -m reblock.run` and assert on stdout/PNGs — those are unaffected (they don't touch the return value).

The purity test (`test_run_is_pure_deterministic_and_leaves_global_rng_untouched`) compares `[x.proposal.proposal_id for x in r1]` — change `r1 = run(cfg)` → `r1 = run(cfg).results` (and `r2` likewise).

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_run.py -v`
Expected: FAIL — `run()` still returns `list[Result]`; `.results` doesn't exist on a list.

- [ ] **Step 3: Rewire `run()`**

Replace `run()` in `src/reblock/run.py` (keep `RunConfig`/`main` structure; `main` is updated in Step 4):

```python
def run(cfg: RunConfig | DictConfig) -> RunOutput:
    """The dataflow pipeline: sample the selection, build only the sample, reblock
    each -> RunOutput(selection, results). The full selection is retained (results
    cover only the sampled max_blocks)."""
    source = cast(Source, instantiate(cfg.data))
    method = cast(Method, instantiate(cfg.method))
    evals = cast("list[Eval]", [instantiate(e) for e in cfg.eval])

    selection = list(cfg.block_ids) if cfg.block_ids is not None else None
    picked = sample(selection, cfg.max_blocks)
    if picked is not None:
        source.block_ids = picked          # build only the sampled blocks
        blocks = source.region().blocks
    else:
        blocks = islice(source.region().blocks, cfg.max_blocks)   # ALL -> islice

    results = [reblock_block(block, method, evals) for block in blocks]
    return RunOutput(selection=selection, results=results)
```

Add `from reblock.pipeline import RunOutput, reblock_block, sample` at the top; drop the now-unused `propose`/`Result` imports if `run.py` no longer references them directly (it doesn't — `reblock_block` owns them). Keep `islice`, `Source`, `Method`, `Eval`, `cast`.

Note: `source.block_ids = picked` requires the source to expose a mutable `block_ids` (KblockSource does). For a source without it (e.g. ShapefileSource, which ignores block_ids), `picked` is `None` unless `cfg.block_ids` was set — and you would not set `block_ids` on a shapefile run — so the `picked is not None` branch is only taken for kblock sources. If a shapefile run ever sets `block_ids`, `setattr` is harmless (ShapefileSource just won't filter). Keep it simple: the assignment is guarded by `picked is not None`.

- [ ] **Step 4: Update `main` to use `RunOutput`**

In `main`, `results = run(cfg)` becomes `output = run(cfg)`; iterate `output.results` for the log; pass `output.results` to `render_results`:

```python
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    output = run(cfg)
    for r in output.results:
        log.info("%s %s", r.block.block_id, {m.eval: dict(m.values) for m in r.metrics})
    if cfg.render.enabled:
        render_results(output.results, Path(HydraConfig.get().runtime.output_dir), cfg.render)
```

(`output.selection` is consumed by the L5 flagged-map emitter — nothing uses it yet in L4.)

- [ ] **Step 5: Run the full suite — results UNCHANGED**

Run: `pixi run check`
Expected: PASS, 139 tests. The pinned kblock/run values are unchanged (`reblock_block` is the extracted loop body); the block_ids test still targets exactly its block; the CLI tests still render. Confirm `run()`'s new return type is used consistently (`grep -n "run(cfg)" tests/test_run.py` → all followed by `.results`).

- [ ] **Step 6: Commit**

```bash
git add src/reblock/run.py tests/test_run.py
git commit -m "$(cat <<'EOF'
refactor: run() is a dataflow pipeline returning RunOutput (redesign L4)

run() now samples the selection, builds only the sampled blocks, reblocks each
via pipeline.reblock_block, and returns RunOutput(selection, results) -- the full
selection retained as a value (not lost to max_blocks). main + all test_run
callers migrated to .results. Results are byte-identical.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

## Self-Review

**Spec coverage (L4):** the dataflow pipeline — `RunOutput(selection, results)` with the selection retained (Task 1/2), sampling split from selection (`sample` + build-only-the-sample), `reblock_block` as the reusable per-block stage. ✓ The screen stage that *produces* the selection, and the emitters that consume it, are L5.

**Placeholder scan:** complete code in every step; the Task-2 test migration directs the implementer to unwrap `.results` at each `run(cfg)` site (necessary — those are the callers being migrated) with the exact change shown. No TBD.

**Type consistency:** `run() -> RunOutput`; `RunOutput.results` / `.selection` are what `main` (Task 2) and the migrated tests read; `reblock_block(block, method, evals) -> Result` and `sample(selection, n) -> list[str] | None` match their call sites in `run()`. `selection` is `list[str] | None` throughout (matching `cfg.block_ids`).

**Results-unchanged** is guarded by the pinned-value tests passing after the `reblock_block` extraction + the sample-before-build change (which only reduces how many blocks are built, never which results are produced).
