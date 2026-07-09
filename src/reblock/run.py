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

from reblock.contracts import Eval, Method, Source
from reblock.emit import render_results
from reblock.pipeline import RunOutput, reblock_block, sample

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
    # Optional list of block ids to build (kblock sources only); None => all
    # blocks. Mirrors conf/config.yaml's top-level `block_ids: null`.
    block_ids: list[str] | None = None
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


def run(cfg: RunConfig | DictConfig) -> RunOutput:
    """The dataflow pipeline: sample the selection, build only the sample, reblock
    each -> RunOutput(selection, results). The full selection is retained (results
    cover only the sampled max_blocks)."""
    # Per-element instantiate for the eval LIST (not instantiate(cfg.eval) whole):
    # instantiating a ListConfig of @dataclass _target_s short-circuits to
    # schema-validated DictConfig nodes instead of calling the constructor.
    # cfg.method is a single _target_ dict, so instantiate(cfg.method) is safe.
    source = cast(Source, instantiate(cfg.data))
    method = cast(Method, instantiate(cfg.method))
    evals = cast("list[Eval]", [instantiate(e) for e in cfg.eval])

    selection = list(cfg.block_ids) if cfg.block_ids is not None else None
    picked = sample(selection, cfg.max_blocks)
    if picked is not None:
        # Source is a structural Protocol (region() only); block_ids is a
        # kblock-specific mutable attr, not part of the Source contract
        # (ShapefileSource has none -- see RunConfig.block_ids docstring).
        source.block_ids = picked  # type: ignore[attr-defined]
        blocks = source.region().blocks
    else:
        blocks = islice(source.region().blocks, cfg.max_blocks)   # ALL -> islice

    results = [reblock_block(block, method, evals) for block in blocks]
    return RunOutput(selection=selection, results=results)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    output = run(cfg)
    for r in output.results:
        log.info("%s %s", r.block.block_id, {m.eval: dict(m.values) for m in r.metrics})
    if cfg.render.enabled:
        render_results(output.results, Path(HydraConfig.get().runtime.output_dir), cfg.render)


if __name__ == "__main__":
    main()
