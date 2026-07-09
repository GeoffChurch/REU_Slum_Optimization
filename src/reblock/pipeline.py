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
