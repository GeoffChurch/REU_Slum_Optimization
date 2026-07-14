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
from reblock.emit import flagged_map, region_map, render_results
from reblock.pipeline import PipelineSpec, run
from reblock.region import RegionBuilder
from reblock.render import google_maps_url

log = logging.getLogger(__name__)


def spec_from_cfg(cfg: DictConfig) -> PipelineSpec:
    """Adapt a composed Hydra config into a typed PipelineSpec (the config edge).
    Per-element instantiate for the eval LIST: instantiate(cfg.eval) whole would
    short-circuit a ListConfig of @dataclass _target_s to schema-validated
    DictConfig nodes instead of constructing them; cfg.data/screen/method are
    single _target_ dicts, so instantiate(...) on each is safe. `block_ids` is the
    region grouping (a list of seed groups); it threads through as block_groups."""
    block_groups = (
        [[str(b) for b in group] for group in cfg.block_ids]
        if cfg.block_ids is not None else None
    )
    return PipelineSpec(
        source=cast(Source, instantiate(cfg.data)),
        screen=cast(Screen, instantiate(cfg.screen)),
        method=cast(Method, instantiate(cfg.method)),
        evals=cast("list[Eval]", [instantiate(e) for e in cfg.eval]),
        max_blocks=cfg.max_blocks,
        region_builder=cast(RegionBuilder, instantiate(cfg.region_builder)),
        block_groups=block_groups,
    )


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    spec = spec_from_cfg(cfg)
    output = run(spec)
    # build_regions narrows source.block_ids to the selected members; clear it once here, before
    # ANY emitter, so the context layers (render_results' surrounding outlines + region_map's
    # whole-metro outlines) query ALL candidate blocks, not just the selection. Order-independent:
    # nothing below reblocks, and building_points ignores block_ids anyway.
    spec.source.block_ids = None   # type: ignore[attr-defined]
    for r in output.results:
        log.info("%s %s", r.block.block_id, {m.eval: dict(m.values) for m in r.metrics})
        log.info("  map: %s", google_maps_url(r.block.boundary, r.block.crs))

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    if output.selection is not None:
        flagged_path = out_dir / "flagged_blocks.txt"
        flagged_path.write_text("".join(f"{b}\n" for b in output.selection))
        log.info("%d blocks flagged -> %s", len(output.selection), flagged_path)
    if cfg.render.enabled:
        render_results(output.results, out_dir, cfg.render, spec.source)
    if cfg.flagged_map.enabled:
        blocks_path = getattr(spec.source, "blocks_path", None)
        if blocks_path is None:
            log.warning("flagged_map: source %s has no blocks_path; skipping",
                        type(spec.source).__name__)
        else:
            flagged_map(str(blocks_path), output.selection or [], out_dir)
    if cfg.region_map.enabled:
        region_map(spec.source, output.regions, output.seed_groups, out_dir)


if __name__ == "__main__":
    main()
