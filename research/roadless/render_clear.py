"""Greedy's cleared buildings at Lens A next to a road method's Lens A corridor.

    PYTHONPATH=. pixi run python research/roadless/render_clear.py <block idxs> [arm]
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402


def main(idx: list[int], arm: str, pop: str = "count", power: float = 1.0) -> None:
    from clear import rows_dir as clear_dir
    from study import rows_dir
    from reblock.budget import prefix_to_displacement, road_corridor
    from reblock.derivations import propose
    blocks = common.build_blocks(common.recipients())
    arms = common.arms()
    for i in idx:
        b = blocks[i]
        g = pd.read_parquet(clear_dir(4, 0.5, pop, power) / f"{b.block_id}.parquet")
        g = g.sort_values("step")
        stop = g[g.D >= 0.10 - 1e-9].iloc[0]
        order = g[(g.step >= 1) & (g.step <= stop.step)].cleared.to_numpy()
        polys = gpd.GeoSeries(list(b.buildings.outlines), crs=None)
        w = common.POPULATIONS[pop].weights(np.asarray(b.buildings.outlines))
        pre = common.prefix_to(b, propose(arms[arm], b).roads, 0.10, w)
        corr = gpd.GeoSeries([road_corridor(pre)])
        lin = pd.read_parquet(rows_dir(0.5, 3.0, 8, pop) / f"{b.block_id}.parquet")
        lp = lin[(lin.arm == arm) & (lin.budget == 0.10)].P_carve.iloc[0]
        fig, axes = plt.subplots(1, 2, figsize=(22, 11))
        for ax in axes:
            gpd.GeoSeries([b.boundary]).boundary.plot(ax=ax, color="k", lw=1)
            gpd.GeoSeries(list(b.streets.geometry)).plot(ax=ax, color="k", lw=2.5)
            ax.set_axis_off()
            ax.set_aspect("equal")
        polys.plot(ax=axes[0], color="0.75", ec="0.4", lw=0.3)
        cl = polys.iloc[order]
        cl.plot(ax=axes[0], color="#d62728", ec="k", lw=0.5)
        for rank, (j, geom) in enumerate(zip(order, cl, strict=True)):
            c = geom.representative_point()
            axes[0].annotate(str(rank + 1), (c.x, c.y), fontsize=5, ha="center", va="center",
                             color="w", weight="bold")
        axes[0].set_title(f"greedy clearing (population {pop}, p {power:g}): {len(order)} of "
                          f"{len(polys)} buildings, D {stop.D:.3f}, roadless perm (p 1) "
                          f"{stop.get('perm1', stop.perm):.3f}\n(numbers = order)")
        polys.plot(ax=axes[1], color="0.75", ec="0.4", lw=0.3)
        corr.plot(ax=axes[1], color="#1f77b4", alpha=0.45)
        pre.plot(ax=axes[1], color="#1f77b4", lw=1)
        axes[1].set_title(f"{arm} at 10% displaced: roadless perm {lp:.3f}")
        fig.suptitle(b.block_id)
        fig.tight_layout()
        out = HERE / f"clear_vs_{arm}_{b.block_id}_{pop}_p{power:g}.png"
        fig.savefig(out, dpi=110)
        print(out, flush=True)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1].split(",")],
         sys.argv[2] if len(sys.argv) > 2 else "cycle_native",
         sys.argv[3] if len(sys.argv) > 3 else "count",
         float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
