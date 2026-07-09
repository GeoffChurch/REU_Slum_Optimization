"""The dataflow pipeline core (pure typed composition -- no Hydra, no DictConfig;
config lives at the edge in reblock.run). PipelineSpec bundles the typed stages;
run() threads them: screen.select(source) yields the retained selection, sample
splits "how many" from "which", each picked block goes through reblock_block
(propose + score). See docs/superpowers/specs/2026-07-08-content-addressed-dataflow-redesign.md.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from itertools import islice

from reblock.contracts import Block, Eval, Method, Result, Screen, Source
from reblock.derivations import propose

log = logging.getLogger(__name__)


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
    the first `n` in order (a screen may rank its ids, e.g. DenseCompactScreen
    returns them worst-access first; a caller's explicit list is its own
    priority). If a sampled block fails to build (e.g. too few
    building points for a valid Voronoi cell), it is skipped and the run yields
    **fewer than `n`** results -- there is no silent backfill from later in the
    selection. This is intentional: it builds only what it reblocks (the redesign
    keeps selection and sampling separate), and the screen feeds pre-verified
    survivors, so the shortfall case does not arise there."""
    return selection[:n] if selection is not None else None


def select_blocks(source: Source, screen: Screen,
                  max_blocks: int) -> tuple[list[str] | None, list[Block]]:
    """Screen -> selection -> sample -> build. Returns (full selection retained for
    emitters, built blocks in selection/priority order). Shared by run() and
    reblock.compare so the selection semantics live in one place. `source.block_ids` is a
    kblock-specific mutable filter (not the Source Protocol); the assignment is guarded by
    `picked is not None`, so it is only reached for kblock-backed runs."""
    selection = screen.select(source)
    picked = sample(selection, max_blocks)
    if picked is None:
        return selection, list(islice(source.region().blocks, max_blocks))
    source.block_ids = picked  # type: ignore[attr-defined]
    built = {b.block_id: b for b in source.region().blocks}
    # Yield in `picked` (screen priority / severity) order, not the parquet order region()
    # happens to build in, so a max_blocks-limited run reblocks/reports worst-first.
    return selection, [built[bid] for bid in picked if bid in built]


def run(spec: PipelineSpec) -> RunOutput:
    """The dataflow pipeline: screen the source for the selection, sample it, build only
    the sample (in priority order), reblock each -> RunOutput(selection, results). The
    full selection is retained (results cover only the sampled max_blocks). Writes no
    files (emitters, at the edge, do the writing) and touches no config or global state."""
    selection, blocks = select_blocks(spec.source, spec.screen, spec.max_blocks)
    results = _reblock_all(blocks, spec.method, spec.evals)
    return RunOutput(selection=selection, results=results)


def _estimate_seconds(done: list[tuple[int, float]], n_parcels: int) -> float | None:
    """Rough per-parcel time estimate from the blocks finished so far (None until the
    first completes). Self-calibrating and method-agnostic; assumes reblock time scales
    ~linearly with parcel count -- a first approximation (topology may be super-linear),
    so it is a guide, not a guarantee."""
    total_parcels = sum(p for p, _ in done)
    if total_parcels == 0:
        return None
    rate = sum(t for _, t in done) / total_parcels
    return rate * n_parcels


def _reblock_all(blocks: list[Block], method: Method, evals: list[Eval]) -> list[Result]:
    """Reblock each block, logging live per-block progress -- id, parcel count, a running
    time estimate, and elapsed. The reblock phase is the slow step for method=topology, so
    this turns a silent wait into visible progress (one pair of lines per reblocked block;
    `blocks` is only the sampled max_blocks, so the volume stays bounded)."""
    n = len(blocks)
    results: list[Result] = []
    done: list[tuple[int, float]] = []   # (n_parcels, seconds) per finished block
    for i, block in enumerate(blocks, 1):
        n_parcels = len(block.parcels)
        est = _estimate_seconds(done, n_parcels)
        log.info("reblocking (%d/%d) %s: %d parcels%s", i, n, block.block_id, n_parcels,
                 "" if est is None else f" (~{est:.0f}s est)")
        t0 = time.perf_counter()
        result = reblock_block(block, method, evals)
        dt = time.perf_counter() - t0
        done.append((n_parcels, dt))
        results.append(result)
        log.info("  reblocked %s in %.1fs", block.block_id, dt)
    return results
