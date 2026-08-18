"""The pinned example block, and every method's roads on it -- declared ONCE.

Both bakers and the frontier baker load the same block. When each declared its own VARIANT/METHOD,
changing the pin in one left the other describing a different block while every test still passed
(piece C's final review, finding I7). One module, one pin.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

from geopandas import GeoDataFrame
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from reblock.contracts import Block, Method, Screen, Source
from reblock.derivations import propose
from reblock.pipeline import build_regions
from reblock.region import RegionBuilder

PINNED_VARIANT = "method_comparison"
PINNED_METHOD = "clearance"


def load_example_block(method: str | None = None) -> tuple[Block, dict[str, GeoDataFrame]]:
    """The pinned block plus roads per method name. `method=None` runs all 8 the variant declares.

    `topology` is single-block-only, which is why this variant pins one block rather than growing a
    region -- see conf/example/method_comparison.yaml.
    """
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config",
                      overrides=[f"+example={PINNED_VARIANT}", "data=capetown_full"])
    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    region_builder = cast(RegionBuilder, instantiate(cfg.region_builder))
    groups = [list(g) for g in cfg.block_ids]
    region = build_regions(source, screen, region_builder, groups, int(cfg.max_blocks))[0]
    assert len(region) == 1, "method_comparison pins a single block by design"
    block = region[0]

    names = [method] if method is not None else list(cfg.methods)
    roads: dict[str, GeoDataFrame] = {}
    for name in names:
        m = cast(Method, instantiate(cfg.all_methods[name]))
        roads[name] = cast(GeoDataFrame, propose(m, block).roads)
    return block, roads
