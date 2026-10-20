"""Anchor and candidate counts (no scoring) at committed networks of S segments: the first S roads
of each block's stored arterial network, street-first order."""
import sys
sys.path.insert(0, "research/roadless")
import geopandas as gpd
import common
from reblock.methods.arterial.primitives import _anchor_points
for b in common.build_blocks(sys.argv[1].split(",")):
    o = gpd.read_parquet(f"research/roadless/road_rows/lineup-caf14fa2/scratch/networks/{b.block_id}/greedy_arterial_access_displacement.ordered.parquet")
    st = list(b.streets.geometry)
    for S in sorted({0, 50, 100, 200, 400, len(o) // 2, len(o)}):
        if S > len(o):
            continue
        a = len(_anchor_points([*st, *o.geometry.iloc[:S]], 32, 128))
        print(b.block_id, S, a, a * (a - 1) // 2 + 8 * a, flush=True)
