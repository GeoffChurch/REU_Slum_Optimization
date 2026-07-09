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
