"""How large must `proxy_keep_n` be so the cheap pre-filter never loses a true top-k block?

`proxy_keep_n` is the ONLY screen knob that saves time. `DenseCompactScreen` runs
proxy -> keep the top `proxy_keep_n` by proxy -> Voronoi + BFS peel over those survivors ->
metric.fine -> gate. The peel is the expensive stage and the gate runs AFTER it, so the gate's
width (the absolute floors in `reblock.metric`) costs no compute at all -- widen or narrow it and
the same blocks were peeled either way. Cutting `proxy_keep_n` is what removes work.

The knob only exists for `needs_peel=True` metrics. `depth_density_proxy` (the shipped default)
and `density_compactness` score straight from the free kblock columns and never reach this branch,
so for them the setting is inert.

## What this measures, and why not a sweep

A tighter pre-filter can only LOSE blocks: one that is not peeled cannot be scored, and a block
with a modest proxy but a large true peel depth is exactly what a tight filter drops. Sweeping
retentions and watching for the top-k to change finds the cliff only to within the sweep's own
grid, and only for the cities swept.

This asks the question directly instead. Run the fine pass over the WHOLE eligible corpus
(`proxy_keep_n` effectively infinite), take the final ranked top-k, and report where those blocks
sat in the PROXY ranking. That worst rank IS the requirement: any `proxy_keep_n` above it cannot
lose a top-k block on this corpus, and any value below it demonstrably does.

Measured 2026-09-18, and it is the justification for the shipped `proxy_keep_n: 1000`:

    city      metric          eligible   top1   top5   top15
    capetown  depth             28,722      2     58      92
    capetown  depth_density     28,722      2     10      67
    nairobi   depth              6,965     66     66     106
    nairobi   depth_density      6,965      1      6      19

Nairobi's true #1 block by fine depth sits at proxy rank 66 -- one rank inside what a 1% cut (70
blocks) would have kept. That is why the knob is a COUNT and not a percentage: the requirement is
a rank, and 2% is 575 blocks in Cape Town but 140 in Nairobi, shrinking the margin precisely where
this measurement says the risk is highest.

    pixi run python -m scripts.proxy_keep_retention
    pixi run python -m scripts.proxy_keep_retention --cities capetown

EXPENSIVE: a full-corpus fine pass per city x metric. Warm, every peel is an L2 hit and this is
minutes; cold it is the fine pass over every eligible block in the city.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
from hydra import compose, initialize_config_dir

from reblock.data.counts import resolved
from reblock.data.kblock import KblockSource
from reblock.presets import load_stages
from reblock.screen.dense_compact import DenseCompactScreen

KS = (1, 5, 15)
VARIANTS = ("depth", "depth_density")     # the only two needs_peel metrics with example configs
UNBOUNDED = 10**9                          # "no pre-filter" -- every eligible block is peeled


def required_n(variant: str, city: str) -> tuple[int, dict[int, int]]:
    """`(eligible, {k: worst proxy rank among the final top-k})` under the SHIPPED screen."""
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config",
                      overrides=[f"+example={variant}", f"data={city}_full",
                                 f"proxy_keep_n={UNBOUNDED}"])
    stages = load_stages(cfg)
    source, screen = stages.source, stages.screen
    assert isinstance(source, KblockSource) and isinstance(screen, DenseCompactScreen)
    ranked = screen.select(source)

    blocks = gpd.read_parquet(
        source.blocks_path, columns=["block_id", "building_count", "block_area_m2", "geometry"])
    blocks = resolved(blocks, source.buildings_path, screen.counts)
    pr = screen.metric.proxy(blocks).to_numpy()
    bid = blocks["block_id"].astype(str).to_numpy()
    cnt = blocks["building_count"].to_numpy(dtype=float)
    ok = (cnt >= screen.min_buildings) & np.isfinite(pr)
    order = [i for i in np.argsort(pr)[::-1] if ok[i]]
    rank = {bid[i]: r + 1 for r, i in enumerate(order)}
    # A block the screen returned must have been scored, hence eligible, hence ranked -- so a
    # missing id would be a contradiction, not a default to paper over.
    worst = {k: max(rank[b] for b in ranked[:k]) for k in KS if len(ranked) >= k}
    return len(order), worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", default="capetown,nairobi")
    args = ap.parse_args()

    print(f"{'city':<10}{'metric':<15}{'eligible':>9}" + "".join(f"{f'top{k}':>8}" for k in KS))
    overall = 0
    for city in args.cities.split(","):
        for variant in VARIANTS:
            n, worst = required_n(variant, city)
            overall = max(overall, *worst.values())
            print(f"{city:<10}{variant:<15}{n:>9}"
                  + "".join(f"{worst.get(k, 0):>8}" for k in KS))
    print(f"\nworst proxy rank of any true top-{max(KS)} block: {overall}")
    print("`proxy_keep_n` must exceed it; conf/config.yaml ships 1000.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
