"""Hydra entrypoint: data=phule method=topology eval=kcomplexity."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

import hydra
from hydra.core.config_store import ConfigStore

from reblock.contracts import Metrics
from reblock.data.shapefile import ShapefileSource
from reblock.eval.kcomplexity import KComplexityEval
from reblock.methods.topology import TopologyMethod


@dataclass
class RunConfig:
    shapefile: str = "???"
    region_id: str = "phule"
    alpha: float = 2.0
    seed: int = 0
    max_blocks: int = 1
    # ShapefileSource now fails loud instead of guessing a CRS when a
    # shapefile has no .prj (e.g. Phule Nagar); the CLI needs a way to state
    # that assumption. None preserves "fail loud" as the CLI default too.
    assumed_crs: int | None = None


ConfigStore.instance().store(name="run", node=RunConfig)


def run(cfg: RunConfig) -> list[Metrics]:
    source = ShapefileSource(cfg.shapefile, region_id=cfg.region_id, assumed_crs=cfg.assumed_crs)
    method = TopologyMethod(alpha=cfg.alpha, seed=cfg.seed)
    evaluator = KComplexityEval()
    region = source.region()
    return [evaluator.score(b, method.propose(b))
            for b in islice(region.blocks, cfg.max_blocks)]


@hydra.main(version_base=None, config_name="run")
def main(cfg: RunConfig) -> None:
    for m in run(cfg):
        print(m.block_id, dict(m.values))


if __name__ == "__main__":
    main()
