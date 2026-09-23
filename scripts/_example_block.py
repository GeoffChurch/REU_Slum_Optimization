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
from omegaconf import DictConfig, open_dict

from reblock.contracts import Block
from reblock.derivations import propose
from reblock.pipeline import build_regions
from reblock.presets import load_methods, load_stages

PINNED_VARIANT = "explore"
# One of the five `explore.yaml` runs, so the roads the web bundle draws are a method
# the same page's frontier also scores. `clearance` would work here (2.8 s on this
# block) but is not in this variant's list -- it is shown on the flagship page
# against the loop-closing refinement that supersedes it.
PINNED_METHOD = "clearance_looped"

# The TEST fixture: the same pipeline on a 263-parcel block. Parity tests pass this rather than
# re-deriving the shipped 6,619-parcel artifact, which MEASURED at 2,882 s of a 2,895 s test --
# `tests/conftest.py` gives every session a cold `REBLOCK_CACHE_DIR` by design, so nothing is
# cached across runs and every method is re-proposed. Same five methods, same order, ~105 s.
# See `conf/example/explore_small.yaml` for the measurements and for why it is not
# `method_comparison` (topology, 377 s on this block, which `explore` does not run).
TEST_VARIANT = "explore_small"


def _compose_pinned_config(variant: str = PINNED_VARIANT) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        return compose(config_name="compare_config",
                       overrides=[f"+example={variant}", "data=capetown_full"])


def _snapshot_path(cfg: DictConfig, block_id: str) -> Path:
    """Same derivation gen_example.py:178 uses: examples/<slug>/desire_lines_<block_id>.geojson,
    since this loader always composes with data=capetown_full (city == "capetown" there, which is
    the flat, no-city-nesting branch of gen_example.py's own `out` computation)."""
    return Path(f"examples/{cfg.example.slug}") / f"desire_lines_{block_id}.geojson"


def example_method_names(variant: str = PINNED_VARIANT) -> list[str]:
    """The methods the pinned example runs, resolved WITHOUT proposing any of them.

    Separated from `load_example_block` because selection is a config list plus a
    snapshot-existence check, while loading is minutes of solving -- so the selection logic can be
    tested for the cost of reading a yaml.
    """
    cfg = _compose_pinned_config(variant)
    names = list(cfg.methods)
    block_id = str(cfg.block_ids[0][0])   # the pinned variant pins ONE block by design
    if _snapshot_path(cfg, block_id).exists():
        names.append("osm_footpaths")
    return names


def load_example_region(variant: str = PINNED_VARIANT) -> Block:
    """The pinned block ALONE -- built, never reblocked.

    `load_example_block` proposes, and proposing is the entire cost: on the spine block
    `greedy_arterial` is 2,011 s and `cycle_native` 849 s, and `tests/conftest.py` gives every
    test session a cold `REBLOCK_CACHE_DIR` so none of it is ever cached between runs. A caller
    that only needs geometry -- the displacement bundle's parity check compares parcels, boundary
    and streets against the live block and never looks at a road -- was paying all of it to read
    `block.parcels`. Region build alone is 18.6 s.
    """
    cfg = _compose_pinned_config(variant)
    stages = load_stages(cfg)
    groups = [list(g) for g in cfg.block_ids]
    region = build_regions(stages.source, stages.screen, stages.region_builder, groups,
                           int(cfg.max_blocks))[0]
    assert len(region) == 1, f"{variant} pins a single block by design"
    return region[0]


def load_example_block(method: str | None = None,
                       variant: str = PINNED_VARIANT) -> tuple[Block, dict[str, GeoDataFrame]]:
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
    cfg = _compose_pinned_config(variant)
    names = [method] if method is not None else example_method_names(variant)
    # The pinned block's id is known before it is built, so the snapshot -- and with it every
    # method -- is configured up front, and a broken preset fails before the block build.
    snapshot = _snapshot_path(cfg, str(cfg.block_ids[0][0]))
    if "osm_footpaths" in names and snapshot.exists():
        with open_dict(cfg):
            cfg.desire_source.snapshot = str(snapshot)
    stages = load_stages(cfg)
    registry = load_methods(cfg.all_methods)

    groups = [list(g) for g in cfg.block_ids]
    region = build_regions(stages.source, stages.screen, stages.region_builder, groups,
                           int(cfg.max_blocks))[0]
    assert len(region) == 1, f"{PINNED_VARIANT} pins a single block by design"
    block = region[0]

    roads: dict[str, GeoDataFrame] = {}
    for name in names:
        roads[name] = cast(GeoDataFrame, propose(registry[name], block).roads)
    return block, roads
