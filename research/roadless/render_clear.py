"""Greedy's cleared buildings at Lens A next to a road method's Lens A corridor.

    PYTHONPATH=. uv run python research/roadless/render_clear.py <block ids,> [arm] [pop] [power] [picker]
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
import lifted  # noqa: E402


def main(ids: list[str], arm: str, pop: str = "count", power: float = 1.0,
         picker: str = "M4") -> None:
    from clear import cleared_through
    from clear import rows_dir as clear_dir
    from reblock.budget import road_corridor
    from reblock.derivations import propose
    by_id = {b.block_id: b for b in common.build_blocks(ids)}
    arms = common.arms()
    for i in ids:
        b = by_id[i]
        g = pd.read_parquet(clear_dir(picker, 0.5, pop, power) / f"{b.block_id}.parquet")
        g = g.sort_values("step")
        stop = g[g.D >= 0.10 - 1e-9].iloc[0]
        order = cleared_through(g, int(stop.step))
        polys = gpd.GeoSeries(list(b.buildings.outlines), crs=None)
        w = common.POPULATIONS[pop].weights(np.asarray(b.buildings.outlines))
        pre = common.prefix_to(b, propose(arms[arm], b).roads, 0.10, w)
        corr = gpd.GeoSeries([road_corridor(pre)])
        mesh = lifted.UniformMesh(0.5, offset=lifted.OFFSET)
        sc = common.Scorer(b, mesh, lifted.Params(ell_m=3.0, K=8),
                           population=common.POPULATIONS[pop])
        lp = sc.perm(pre)
        # the round each cleared building went in (colours it when there are too many to number)
        rnd = np.concatenate([np.full(len(js), st) for st, js in
                              zip(g.step, g.cleared, strict=True) if 1 <= st <= stop.step])
        zoom = len(polys) > 1500
        fig, axes = plt.subplots(2 if zoom else 1, 2, figsize=(22, 22 if zoom else 11),
                                 squeeze=False)
        cl = polys.iloc[order]
        cx = np.array([p.centroid.x for p in cl])
        cy = np.array([p.centroid.y for p in cl])
        for row in range(axes.shape[0]):
            for ax in axes[row]:
                gpd.GeoSeries([b.boundary]).boundary.plot(ax=ax, color="k", lw=1)
                gpd.GeoSeries(list(b.streets.geometry)).plot(ax=ax, color="k", lw=2.5)
                ax.set_axis_off()
                ax.set_aspect("equal")
                polys.plot(ax=ax, color="0.8", ec="0.5", lw=0.2)
            if len(order) <= 150:
                cl.plot(ax=axes[row, 0], color="#d62728", ec="k", lw=0.5)
                for rank, geom in enumerate(cl):
                    c = geom.representative_point()
                    axes[row, 0].annotate(str(rank + 1), (c.x, c.y), fontsize=5, ha="center",
                                          va="center", color="w", weight="bold")
            else:
                gpd.GeoDataFrame(dict(r=rnd), geometry=list(cl)).plot(
                    ax=axes[row, 0], column="r", cmap="plasma", ec="k", lw=0.3,
                    legend=row == 0, legend_kwds=dict(label="round", shrink=0.5))
            corr.plot(ax=axes[row, 1], color="#1f77b4", alpha=0.45)
            pre.plot(ax=axes[row, 1], color="#1f77b4", lw=1)
        if zoom:
            # a window around the median cleared building, a quarter of the block's width
            minx, miny, maxx, maxy = b.boundary.bounds
            half = 0.125 * max(maxx - minx, maxy - miny)
            mx, my = float(np.median(cx)), float(np.median(cy))
            for ax in axes[1]:
                ax.set_xlim(mx - half, mx + half)
                ax.set_ylim(my - half, my + half)
            for ax in axes[0]:
                ax.add_patch(plt.Rectangle((mx - half, my - half), 2 * half, 2 * half,
                                           fill=False, ec="#2ca02c", lw=2))
        axes[0, 0].set_title(f"greedy {picker} (population {pop}, p {power:g}): {len(order)} of "
                             f"{len(polys)} buildings, D {stop.D:.3f}, roadless perm (p 1) "
                             f"{stop.get('perm1', stop.perm):.3f}\n"
                             + ("(numbers = order)" if len(order) <= 150 else "(colour = round)"))
        axes[0, 1].set_title(f"{arm} at 10% displaced: roadless perm {lp:.3f}")
        fig.suptitle(b.block_id)
        fig.tight_layout()
        out = HERE / f"clear_vs_{arm}_{b.block_id}_{picker}_{pop}_p{power:g}.png"
        fig.savefig(out, dpi=110)
        print(out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1].split(","),
         sys.argv[2] if len(sys.argv) > 2 else "cycle_native",
         sys.argv[3] if len(sys.argv) > 3 else "count",
         float(sys.argv[4]) if len(sys.argv) > 4 else 1.0,
         sys.argv[5] if len(sys.argv) > 5 else "M4")
