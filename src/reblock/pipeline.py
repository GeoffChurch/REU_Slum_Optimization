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
