"""The dataflow pipeline core (pure typed composition -- no Hydra, no DictConfig;
config lives at the edge in reblock.run). PipelineSpec bundles the typed stages;
run() threads them: screen.select(source) yields the retained selection, sample
splits "how many" from "which", each picked block goes through reblock_block
(propose + score). See docs/superpowers/specs/2026-07-08-content-addressed-dataflow-redesign.md.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from itertools import islice

from reblock.contracts import Block, Eval, Method, Result, Screen, Source
from reblock.derivations import propose
from reblock.region import IdentityRegionBuilder, RegionBuilder, region_reblock

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineSpec:
    """The typed stages of one run, composed at the edge (reblock.run.spec_from_cfg)
    from Hydra config, or directly in Python. The core pipeline (run) is exactly a
    function of this value -- it never sees a DictConfig.

    `block_groups` is the region grouping (a list of seed groups, block_ids per group);
    None means no explicit grouping (the screen decides). `region_builder` expands each
    seed group into the region actually reblocked (identity = passthrough)."""
    source: Source
    screen: Screen
    method: Method
    evals: list[Eval]
    max_blocks: int = 1
    region_builder: RegionBuilder = field(default_factory=IdentityRegionBuilder)
    block_groups: list[list[str]] | None = None


@dataclass(frozen=True)
class RunOutput:
    selection: list[str] | None       # the full block selection (None = all blocks)
    results: list[Result]             # one per reblocked region (or sampled block)
    regions: list[list[str]] = field(default_factory=list)  # the builder's expanded groups


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


def _seed_groups(spec: PipelineSpec) -> tuple[list[list[str]] | None, list[str] | None]:
    """The seed region groups + the retained selection. Explicit `block_groups` win (each
    inner list is one region's seed group). Otherwise the screen selects: a real selection
    wraps as singleton seed groups; None (all blocks) returns (None, None) -- the caller's
    signal to take the classic build-limited all-blocks path (a region builder has nothing
    to expand over an unenumerated full metro, and that path also serves sources without a
    block_geometries() accessor, e.g. ShapefileSource)."""
    if spec.block_groups is not None:
        selection = sorted({b for group in spec.block_groups for b in group})
        return spec.block_groups, selection
    sel = spec.screen.select(spec.source)
    if sel is None:
        return None, None
    return [[b] for b in sel], sel


def run(spec: PipelineSpec) -> RunOutput:
    """The region-aware dataflow pipeline: resolve seed groups (explicit block_groups, or the
    screen's selection wrapped as singletons), expand them with the region_builder over cheap
    block geometries, build full Blocks only for the union of members, then reblock each region
    -- a singleton region routes through the single-block `reblock_block` (behaviour identical
    to a plain per-block reblock), a genuine multi-block region through `region_reblock`.
    Writes no files (emitters, at the edge, do that) and touches no config or global state."""
    groups, selection = _seed_groups(spec)
    if groups is None:
        # Screen passed everything through (no explicit groups): the classic all-blocks path.
        # region() already filters to buildable blocks and sorts, so islice takes the first
        # max_blocks buildable ones -- exactly the pre-region behaviour, and no block_geometries
        # accessor is required (so ShapefileSource works). Each built block is its own singleton
        # region.
        blocks = list(islice(spec.source.region().blocks, spec.max_blocks))
        results = _reblock_all(blocks, spec.method, spec.evals)
        return RunOutput(selection=selection, results=results,
                         regions=[[b.block_id] for b in blocks])

    spec.source.block_ids = None                     # type: ignore[attr-defined]  # ALL candidates
    block_geoms = spec.source.block_geometries()     # type: ignore[attr-defined]
    regions = spec.region_builder.build(block_geoms, groups)[:spec.max_blocks]
    members = sorted({b for region in regions for b in region})
    spec.source.block_ids = members                  # type: ignore[attr-defined]  # members only
    built = {b.block_id: b for b in spec.source.region().blocks}
    results = []
    for region in regions:
        rblocks = [built[b] for b in region if b in built]
        if len(rblocks) == 1:
            results.append(reblock_block(rblocks[0], spec.method, spec.evals))
        elif len(rblocks) > 1:
            results.append(region_reblock(rblocks, spec.method, spec.evals))
    return RunOutput(selection=selection, results=results, regions=regions)


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
