"""A POOL of region blocks, cached -- so region-scale claims can have an n greater than one.

`snap_vs_peel.region_block_cached` builds exactly one region (`max_blocks=1`, `regions[0]`), which
is what every region-scale result so far rests on. `max_anchors` winning by +0.088 permeability is
currently a single data point; replication needs independent regions, and each one costs minutes to
grow, so they are cached the same way the first one is.

Regions come back from `build_regions` sorted, so index 0 here IS the block every earlier region
result used -- `region_block_cached()`'s pickle and `blocks(1)[0]` are the same object. That makes
the existing 15-road measurement directly comparable to anything run here rather than a separate
baseline.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import cast

from reblock.contracts import Block, Screen, Source

CACHE_DIR = Path("scratchpad/perf")
OVERRIDES = ["metric=depth", "data=capetown_full", "screen=dense_compact",
             "region_builder=dense_cluster", "region_builder.max_buildings=3000"]


def blocks(n: int) -> list[Block]:
    """The first `n` regions, each collapsed to one Block. Cached per index."""
    paths = [CACHE_DIR / f"region_block_{i}.pkl" for i in range(n)]
    if all(p.exists() for p in paths):
        out = []
        for p in paths:
            with p.open("rb") as fh:
                out.append(cast(Block, pickle.load(fh)))
        return out

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from reblock.pipeline import build_regions
    from reblock.region import RegionBuilder, region_block

    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config", overrides=[*OVERRIDES, f"max_blocks={n}"])
    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    rb = cast(RegionBuilder, instantiate(cfg.region_builder))
    t0 = time.perf_counter()
    regions = build_regions(source, screen, rb, None, n)
    print(f"  built {len(regions)} regions in {time.perf_counter() - t0:.0f} s", flush=True)

    out = []
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for i, region in enumerate(regions):
        blk = region_block(region) if len(region) > 1 else region[0]
        with paths[i].open("wb") as fh:
            pickle.dump(blk, fh)
        out.append(blk)
    return out


def main() -> None:
    pool = blocks(6)
    print(f"\n  {'idx':>4}{'blocks':>9}{'parcels':>10}{'buildings':>12}{'street_rows':>13}")
    for i, b in enumerate(pool):
        print(f"  {i:>4}{'-':>9}{len(b.parcels):>10,}{len(b.building_points):>12,}"
              f"{len(b.streets):>13,}")
    print("\n  index 0 is the block every earlier region-scale result used.")


if __name__ == "__main__":
    main()
