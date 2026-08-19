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
from omegaconf import DictConfig, open_dict

from reblock.contracts import Block, Method, Screen, Source
from reblock.derivations import propose
from reblock.pipeline import build_regions
from reblock.region import RegionBuilder

PINNED_VARIANT = "method_comparison"
PINNED_METHOD = "clearance"


def _compose_pinned_config() -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        return compose(config_name="compare_config",
                       overrides=[f"+example={PINNED_VARIANT}", "data=capetown_full"])


def _snapshot_path(cfg: DictConfig, block_id: str) -> Path:
    """Same derivation gen_example.py:178 uses: examples/<slug>/desire_lines_<block_id>.geojson,
    since this loader always composes with data=capetown_full (city == "capetown" there, which is
    the flat, no-city-nesting branch of gen_example.py's own `out` computation)."""
    return Path(f"examples/{cfg.example.slug}") / f"desire_lines_{block_id}.geojson"


def example_method_names() -> list[str]:
    """The methods the pinned example runs, resolved WITHOUT proposing any of them.

    Separated from `load_example_block` because selection is a config list plus a
    snapshot-existence check, while loading is minutes of solving -- so the selection logic can be
    tested for the cost of reading a yaml.
    """
    cfg = _compose_pinned_config()
    names = list(cfg.methods)
    block_id = str(cfg.block_ids[0][0])   # method_comparison pins a single block by design
    if _snapshot_path(cfg, block_id).exists():
        names.append("osm_footpaths")
    return names


def load_example_block(method: str | None = None) -> tuple[Block, dict[str, GeoDataFrame]]:
    """The pinned block plus roads per method name. `method=None` runs all eight:
    `example_method_names()` -- the seven conf/example/method_comparison.yaml declares, plus
    `osm_footpaths` -- the real as-built informal network, the reference the whole comparison is
    measured against, not a competitor.

    `osm_footpaths` is injected exactly as scripts/gen_example.py:175-182 injects it: only when a
    committed OSM snapshot sits beside the example (fetched once by
    scripts.fetch_desire_lines_snapshot), and omitted -- not raised -- when it is absent, which is
    why that example reproduces offline.

    `topology` is single-block-only, which is why this variant pins one block rather than growing a
    region -- see conf/example/method_comparison.yaml.
    """
    cfg = _compose_pinned_config()
    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    region_builder = cast(RegionBuilder, instantiate(cfg.region_builder))
    groups = [list(g) for g in cfg.block_ids]
    region = build_regions(source, screen, region_builder, groups, int(cfg.max_blocks))[0]
    assert len(region) == 1, "method_comparison pins a single block by design"
    block = region[0]

    names = [method] if method is not None else example_method_names()

    snapshot = _snapshot_path(cfg, block.block_id)
    if "osm_footpaths" in names and snapshot.exists():
        with open_dict(cfg):
            cfg.desire_source.snapshot = str(snapshot)

    roads: dict[str, GeoDataFrame] = {}
    for name in names:
        m = cast(Method, instantiate(cfg.all_methods[name]))
        roads[name] = cast(GeoDataFrame, propose(m, block).roads)
    return block, roads
