"""The same greedy under different along-conductances, side by side at 10% displaced: does a
road-favouring conductance turn nibbling into lanes? Cleared buildings coloured by round.

    PYTHONPATH=. pixi run python research/roadless/render_alongs.py <ids,> [picker] [alongs,]
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


def main(ids: list[str], picker: str, alongs: list[str]) -> None:
    for b in common.build_blocks(ids):
        polys = gpd.GeoSeries(list(b.buildings.outlines))
        fig, axes = plt.subplots(1, len(alongs), figsize=(11 * len(alongs), 11), squeeze=False)
        for ax, along in zip(axes[0], alongs, strict=True):
            g = pd.read_parquet(rows_dir(picker, 0.5, "area", 2.0, along)
                                / f"{b.block_id}.parquet").sort_values("step")
            stop = g[g.D >= D_LENS - 1e-9].iloc[0]
            order = cleared_through(g, int(stop.step))
            rnd = np.concatenate([np.full(len(js), st) for st, js in
                                  zip(g.step, g.cleared, strict=True) if 1 <= st <= stop.step])
            gpd.GeoSeries([b.boundary]).boundary.plot(ax=ax, color="k", lw=1)
            gpd.GeoSeries(list(b.streets.geometry)).plot(ax=ax, color="r", lw=2.5)
            polys.plot(ax=ax, color="0.8", ec="0.5", lw=0.2)
            gpd.GeoDataFrame(dict(r=rnd), geometry=list(polys.iloc[order])).plot(
                ax=ax, column="r", cmap="plasma", ec="k", lw=0.3, vmin=1, vmax=stop.step)
            ax.set_axis_off()
            ax.set_aspect("equal")
            ax.set_title(f"{along}: {len(order)} of {len(polys)} cleared, D {stop.D:.3f}, perm' "
                         f"{stop.perm:.3f} (own metric)", fontsize=14)
        fig.suptitle(f"{b.block_id}, greedy {picker}, colour = round", fontsize=16)
        fig.tight_layout()
        out = HERE / f"alongs_{b.block_id}_{picker}.png"
        fig.savefig(out, dpi=90)
        print(out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1].split(","), sys.argv[2] if len(sys.argv) > 2 else "S0.01cat",
         sys.argv[3].split(",") if len(sys.argv) > 3 else ["uni", "sl3", "sl10"])
