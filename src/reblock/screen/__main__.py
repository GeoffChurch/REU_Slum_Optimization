"""reblock.screen: run the selected Screen on a city's data and print the flagged block_ids.
Interim standalone app (the flow-refactor folds Screen into run() as a stage; see that spec).
"""
from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from reblock.data.provision import DEFAULT_CACHE, ensure_city_data


def detect(cfg: DictConfig, *, cache_dir: Path = DEFAULT_CACHE) -> list[str]:
    blocks_path, buildings_path = ensure_city_data(cfg.city, cache_dir=cache_dir)
    screen = instantiate(cfg.screen, blocks_path=str(blocks_path),
                         buildings_path=str(buildings_path))
    ids: list[str] = screen.select()
    return ids


@hydra.main(version_base=None, config_path="../../../conf", config_name="screen_config")
def main(cfg: DictConfig) -> None:
    ids = detect(cfg)
    print(f"{len(ids)} informal blocks flagged")
    for bid in ids:
        print(bid)


if __name__ == "__main__":
    main()
