"""Does `max_anchors` make the access arterial affordable at REGION scale?

Measured: the `depth` variant takes 1,115 s without `greedy_arterial_access_displacement` and had
not finished after 41,700 s with it -- a >=37x penalty on an 11,006-parcel region.

The suspected cause is candidate count. `_anchor_points` includes EVERY network vertex by default,
so the anchor set grows as roads commit and candidates grow ~C(anchors, 2). `max_anchors > 0` caps
anchors at that many arc-length samples and drops the per-vertex ones, bounding candidates to
~C(max_anchors, 2) regardless of how complex the network gets.

Times one propose on the real grown region at several caps. Unbounded is NOT re-run -- it is known
to exceed 11 h.
"""
from __future__ import annotations

import time
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from reblock.methods.arterial import GreedyArterialReblocker, SnapToBoundary
from reblock.pipeline import build_regions
from reblock.region import region_block


def main() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config", overrides=[
            "data=capetown_full", "screen=dense_compact", "metric=depth",
            "region_builder=dense_cluster", "region_builder.max_buildings=3000"])
        src = instantiate(cfg.data)
        screen = instantiate(cfg.screen)
        rb = instantiate(cfg.region_builder)
    regions = build_regions(src, screen, rb, None, 1)
    region = regions[0]
    blk = region_block(region) if len(region) > 1 else region[0]
    print(f"region: {len(region)} blocks, {len(blk.parcels):,} parcels\n", flush=True)
    print(f"  {'max_anchors':>12}{'seconds':>10}{'roads':>7}{'total_m':>10}")
    for ma in (24, 48, 96):
        m = GreedyArterialReblocker(realizer=SnapToBoundary(), objective="access",
                                    cost="displacement", workers=16, max_anchors=ma)
        t = time.monotonic()
        r = m.propose(blk).roads
        el = time.monotonic() - t
        print(f"  {ma:>12}{el:>10.1f}{len(r):>7}{r.geometry.length.sum():>10.0f}", flush=True)

if __name__ == "__main__":
    main()
