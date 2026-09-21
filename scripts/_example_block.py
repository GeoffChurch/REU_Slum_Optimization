"""The pinned example block, and every method's roads on it -- declared ONCE.

Every baker behind the Explore page loads the same block: PermGraph, WebBundle, Frontier,
DisplacementField and AuthoringBlock. When each declared its own VARIANT/METHOD, changing the pin
in one left the other describing a different block while every test still passed (piece C's final
review, finding I7). One module, one pin.

The pin moved from `method_comparison` (`ZAF.9.3.1_1_40972`, 263 parcels) to `explore`
(`ZAF.9.3.1_1_5810`, 6,619 parcels) on 2026-09-20, so the page's five live stages follow the
block the screen actually ranks first instead of a small one chosen for a different reason.

It is a SEPARATE variant rather than a repinned `method_comparison` because that variant exists to
host `topology`, the single-block-only prior art, and topology cannot run on this block -- MEASURED,
`NodeNotFound: Source (305620.00,8022470.00) is not in G` after 454 s. Being one block is necessary
and not sufficient for it: at 6,619 parcels this block's street seam disconnects the source the same
way a multi-block region does. Repinning `method_comparison` would have deleted the one method it
exists for. `conf/example/explore.yaml` carries the rest of that reasoning.

Nothing here serves the method-comparison PAGE: that is `gen_example method_comparison`, reading
its own config. So this module still declares exactly one pin, and topology keeps its home.
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

PINNED_VARIANT = "explore"
# One of the five `explore.yaml` runs, so the roads the web bundle draws are a method
# the same page's frontier also scores. `clearance` would work here (2.8 s on this
# block) but is not in this variant's list -- it is shown on the flagship page
# against the loop-closing refinement that supersedes it.
PINNED_METHOD = "clearance_looped"


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
    block_id = str(cfg.block_ids[0][0])   # the pinned variant pins ONE block by design
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

    The variant pins one block rather than growing a region so that all five Explore bundles
    describe the same thing -- see conf/example/explore.yaml.
    """
    cfg = _compose_pinned_config()
    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    region_builder = cast(RegionBuilder, instantiate(cfg.region_builder))
    groups = [list(g) for g in cfg.block_ids]
    region = build_regions(source, screen, region_builder, groups, int(cfg.max_blocks))[0]
    assert len(region) == 1, f"{PINNED_VARIANT} pins a single block by design"
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
