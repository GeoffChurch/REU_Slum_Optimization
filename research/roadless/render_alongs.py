"""The same greedy under different along-conductances, side by side at 10% displaced: does a
road-favouring conductance turn nibbling into lanes? Cleared buildings coloured by round.

    PYTHONPATH=. uv run python research/roadless/render_alongs.py <ids,> [picker] [alongs,] [h of the run]
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
from clear import cleared_through, rows_dir  # noqa: E402

D_LENS = 0.10


def main(ids: list[str], picker: str, alongs: list[str], h: float = 0.5) -> None:
    for b in common.build_blocks(ids):
        polys = gpd.GeoSeries(list(b.buildings.outlines))
        big = len(polys) > 1500          # add a zoomed row
        rows_n = 2 if big else 1
        fig, axes = plt.subplots(rows_n, len(alongs), figsize=(11 * len(alongs), 11 * rows_n),
                                 squeeze=False)
        window = None
        for col, along in enumerate(alongs):
            g = pd.read_parquet(rows_dir(picker, h, "area", 2.0, along)
                                / f"{b.block_id}.parquet").sort_values("step")
            stop = g[g.D >= D_LENS - 1e-9].iloc[0]
            order = cleared_through(g, int(stop.step))
            rnd = np.concatenate([np.full(len(js), st) for st, js in
                                  zip(g.step, g.cleared, strict=True) if 1 <= st <= stop.step])
            if window is None:
                cl = polys.iloc[order]
                cx = float(np.median([p.centroid.x for p in cl]))
                cy = float(np.median([p.centroid.y for p in cl]))
                minx, miny, maxx, maxy = b.boundary.bounds
                half = 0.125 * max(maxx - minx, maxy - miny)
                window = (cx - half, cx + half, cy - half, cy + half)
            for row in range(rows_n):
                ax = axes[row, col]
                gpd.GeoSeries([b.boundary]).boundary.plot(ax=ax, color="k", lw=1)
                gpd.GeoSeries(list(b.streets.geometry)).plot(ax=ax, color="r", lw=2.5)
                polys.plot(ax=ax, color="0.8", ec="0.5", lw=0.2)
                gpd.GeoDataFrame(dict(r=rnd), geometry=list(polys.iloc[order])).plot(
                    ax=ax, column="r", cmap="plasma", ec="k", lw=0.3, vmin=1, vmax=stop.step)
                ax.set_axis_off()
                ax.set_aspect("equal")
                if row == 1:
                    ax.set_xlim(window[0], window[1])
                    ax.set_ylim(window[2], window[3])
                elif big:
                    ax.add_patch(plt.Rectangle((window[0], window[2]), window[1] - window[0],
                                               window[3] - window[2], fill=False, ec="#2ca02c",
                                               lw=2))
            axes[0, col].set_title(f"{along}: {len(order)} of {len(polys)} cleared, D "
                                   f"{stop.D:.3f}, perm' {stop.perm:.3f} (own metric, h {h:g})",
                                   fontsize=14)
        fig.suptitle(f"{b.block_id}, greedy {picker}, colour = round", fontsize=16)
        fig.tight_layout()
        out = HERE / f"alongs_{b.block_id}_{picker}_h{h:g}.png"
        fig.savefig(out, dpi=90)
        print(out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1].split(","), sys.argv[2] if len(sys.argv) > 2 else "S0.01cat",
         sys.argv[3].split(",") if len(sys.argv) > 3 else ["uni", "sl3", "sl10"],
         float(sys.argv[4]) if len(sys.argv) > 4 else 0.5)
