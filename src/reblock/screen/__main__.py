"""reblock.screen: run the selected Screen on a city's data and persist the flagged block_ids.
Interim standalone app (the flow-refactor folds Screen into run() as a stage; see that spec).
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from reblock.data.provision import DEFAULT_CACHE, ensure_city_data

log = logging.getLogger(__name__)


def detect(cfg: DictConfig, *, cache_dir: Path = DEFAULT_CACHE) -> list[str]:
    blocks_path, buildings_path = ensure_city_data(cfg.city, cache_dir=cache_dir)
    screen = instantiate(cfg.screen, blocks_path=str(blocks_path),
                         buildings_path=str(buildings_path))
    ids: list[str] = screen.select()
    return ids


def emit(ids: list[str], out_dir: Path) -> Path:
    """Write the flagged block_ids (one per line) to out_dir/flagged_blocks.txt and log the
    count and path. Returns the written path."""
    out_path = out_dir / "flagged_blocks.txt"
    out_path.write_text("".join(f"{bid}\n" for bid in ids))
    log.info("%d informal blocks flagged -> %s", len(ids), out_path)
    return out_path


@hydra.main(version_base=None, config_path="../../../conf", config_name="screen_config")
def main(cfg: DictConfig) -> None:
    ids = detect(cfg)
    emit(ids, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
