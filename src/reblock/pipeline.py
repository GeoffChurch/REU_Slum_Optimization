"""The dataflow pipeline core (pure typed composition -- no Hydra, no DictConfig;
config lives at the edge in reblock.run). PipelineSpec bundles the typed stages;
run() resolves the seed groups (explicit block_groups, or the screen's selection wrapped as
singletons), build_regions expands + builds each region's member Blocks, and each region goes
through reblock_block (a single member) or region_reblock (multiple members). See
docs/superpowers/specs/2026-07-10-region-cli-and-builders-design.md.
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
    seed_groups: list[list[str]] = field(default_factory=list)  # pre-expansion seed groups


def reblock_block(block: Block, method: Method, evals: list[Eval]) -> Result:
    """One block through method + evals -> a Result (metrics tuple over the evals)."""
    proposal = propose(method, block)
    metrics = tuple(ev.score(block, proposal) for ev in evals)
    return Result(block=block, proposal=proposal, metrics=metrics)


def _seed_groups(
    source: Source, screen: Screen, block_groups: list[list[str]] | None,
) -> tuple[list[list[str]] | None, list[str] | None]:
    """The seed region groups + the retained selection. Explicit `block_groups` win (each
    inner list is one region's seed group) -- and short-circuit without touching `screen` at
    all, so a caller that already resolved groups (e.g. `run`, which needs the selection
    itself) can hand them straight back in without re-invoking a possibly-expensive screen.
    Otherwise the screen selects: a real selection wraps as singleton seed groups; None (all
    blocks) returns (None, None) -- the caller's signal to take the classic build-limited
    all-blocks path (a region builder has nothing to expand over an unenumerated full metro;
    every Source has block_geometries(), so this path is chosen by *no groups*, not by
    capability)."""
    if block_groups is not None:
        selection = sorted({b for group in block_groups for b in group})
        return block_groups, selection
    sel = screen.select(source)
    if sel is None:
        return None, None
    return [[b] for b in sel], sel


def build_regions(source: Source, screen: Screen, region_builder: RegionBuilder,
                  block_groups: list[list[str]] | None, max_blocks: int) -> list[list[Block]]:
    """Resolve seed groups (explicit `block_groups`, or the screen's selection wrapped as
    singletons) into the region member Blocks to actually reblock/compare: `region_builder`
    expands each seed group over cheap `block_geometries()`, full Blocks are then built only
    for the union of every region's members, and each expanded group's Blocks come back as one
    inner list (its region; a singleton list for a single-block region). A screen that passes
    everything through (None -- no explicit groups either) takes the classic build-limited
    all-blocks path instead: `source.region()` already filters to buildable blocks and sorts,
    so `islice` takes the first `max_blocks` buildable ones as singleton "regions" -- chosen
    by the absence of groups, not by source capability (every Source has block_geometries()).
    Shared by `pipeline.run` and `reblock.compare` so the region-resolution semantics live in
    one place."""
    groups, _selection = _seed_groups(source, screen, block_groups)
    if groups is None:
        blocks = list(islice(source.region().blocks, max_blocks))
        return [[b] for b in blocks]

    source.block_ids = None                     # type: ignore[attr-defined]  # ALL candidates
    block_geoms = source.block_geometries()
    regions = region_builder.build(block_geoms, groups)[:max_blocks]
    members = sorted({b for region in regions for b in region})
    source.block_ids = members                  # type: ignore[attr-defined]  # members only
    built = {b.block_id: b for b in source.region().blocks}
    result: list[list[Block]] = []
    for region in regions:
        region_blocks = [built[b] for b in region if b in built]
        dropped = sorted(b for b in region if b not in built)
        if dropped:
            log.warning(
                "region %s lost member(s) %s to a build failure -- reblocking only %s",
                "+".join(sorted(region)), dropped, [b.block_id for b in region_blocks],
            )
        result.append(region_blocks)
    return result


def run(spec: PipelineSpec) -> RunOutput:
    """The region-aware dataflow pipeline: resolve the seed groups once (for the retained
    `selection` and, unexpanded, for `RunOutput.seed_groups` -- the pre-expansion groups
    `emit.region_map` outlines, e.g. against a `convex_hull` builder's fill-in), build each
    region's member Blocks via `build_regions` (handing the already-resolved groups back in as
    its `block_groups`, so the screen -- possibly expensive, e.g. DenseCompactScreen's fine pass
    -- runs at most once), then reblock each region -- a singleton region routes through the
    single-block `reblock_block` (behaviour identical to a plain per-block reblock), a genuine
    multi-block region through `region_reblock`. Logs each region's label, parcel count, and
    wall-clock time as it finishes (the reblock loop is the slow step for topology/arterial/
    multi-block, so this is the run's only progress signal). Writes no files (emitters, at the
    edge, do that) and touches no config or global state."""
    groups, selection = _seed_groups(spec.source, spec.screen, spec.block_groups)
    region_blocks = build_regions(spec.source, spec.screen, spec.region_builder,
                                  groups, spec.max_blocks)
    results: list[Result] = []
    regions: list[list[str]] = []
    for rblocks in region_blocks:
        regions.append([b.block_id for b in rblocks])
        if not rblocks:
            continue
        label = "+".join(b.block_id for b in rblocks)
        n_parcels = sum(len(b.parcels) for b in rblocks)
        start = time.monotonic()
        if len(rblocks) == 1:
            results.append(reblock_block(rblocks[0], spec.method, spec.evals))
        else:
            results.append(region_reblock(rblocks, spec.method, spec.evals))
        log.info("reblocked %s (%d parcels) in %.1fs", label, n_parcels, time.monotonic() - start)
    return RunOutput(selection=selection, results=results, regions=regions,
                     seed_groups=groups if groups is not None else list(regions))
