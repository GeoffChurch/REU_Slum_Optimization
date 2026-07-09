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
from reblock.derivations import propose
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
        proposal = propose(method, block)
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
