"""Hydra entrypoint (the config edge): parse the conf/ config groups into a typed
PipelineSpec, run the pure pipeline (reblock.pipeline.run), then fire the opt-in
emitters into the Hydra run dir. The core pipeline never sees this DictConfig --
spec_from_cfg is the only adapter.
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from reblock.contracts import CountingScreen, ScoringScreen
from reblock.data.kblock import KblockSource
from reblock.emit import flagged_map, region_map, render_results
from reblock.pipeline import PipelineSpec, run
from reblock.presets import load_evals, load_method, load_stages
from reblock.render import google_maps_url

log = logging.getLogger(__name__)


def spec_from_cfg(cfg: DictConfig) -> PipelineSpec:
    """Adapt a composed Hydra config into a typed PipelineSpec (the config edge). Every stage is
    built through `reblock.presets`, the one typed loading boundary. `block_ids` is the region
    grouping (a list of seed groups); it threads through as block_groups."""
    block_groups = (
        [[str(b) for b in group] for group in cfg.block_ids]
        if cfg.block_ids is not None else None
    )
    stages = load_stages(cfg)
    return PipelineSpec(
        source=stages.source,
        screen=stages.screen,
        method=load_method(cfg.method),
        evals=load_evals(cfg.eval),
        max_blocks=cfg.max_blocks,
        region_builder=stages.region_builder,
        block_groups=block_groups,
    )


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    spec = spec_from_cfg(cfg)
    output = run(spec)
    for r in output.results:
        log.info("%s %s", r.block.block_id, {m.eval: dict(m.values) for m in r.metrics})
        log.info("  map: %s", google_maps_url(r.block.boundary, r.block.crs))

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    if output.selection is not None:
        flagged_path = out_dir / "flagged_blocks.txt"
        flagged_path.write_text("".join(f"{b}\n" for b in output.selection))
        log.info("%d blocks flagged -> %s", len(output.selection), flagged_path)
    if cfg.render.enabled:
        render_results(output.results, out_dir, spec.source)
    if cfg.flagged_map.enabled:
        if isinstance(spec.source, KblockSource):
            flagged_map(str(spec.source.blocks_path), output.selection or [], out_dir)
        else:
            log.warning("flagged_map: source %s has no blocks_path; skipping",
                        type(spec.source).__name__)
    if cfg.region_map.enabled:
        scoring = spec.screen if isinstance(spec.screen, ScoringScreen) else None
        m = scoring.metric if scoring is not None else None
        region_map(spec.source, output.regions, output.seed_groups, out_dir,
                   selection=output.selection,
                   depths=scoring.selection_scores(spec.source) if scoring is not None else None,
                   metric_name=m.name if m is not None else "score", metric=m,
                   counts=spec.screen.counts if isinstance(spec.screen, CountingScreen) else None)


if __name__ == "__main__":
    main()
