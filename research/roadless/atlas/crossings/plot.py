"""Plots of a run's crossings (analyze.py's ends.parquet and pairs.parquet): the block's
footprints, the built roads as 7 m corridors (darker the earlier), crossings (opposite-side
pairs within analyze.TOL_M) ringed in orange, jogged pairs (up to JOG_M) in blue.

    P=research/roadless/atlas/crossings/plot.py       # from the repo root, PYTHONPATH=.
    uv run python $P <run dir> <out.png> [<title>]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import LineString  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as an  # noqa: E402

JOG_M = 20.0
CROSS = "#eb6834"
JOG = "#2a78d6"
BLD = "#d6d5d0"
ROAD = "#52514e"


def plot(d: Path, out: Path, title: str) -> None:
    run = an.load(d, budget=None)
    ra = run.ra
    ends = pd.read_parquet(d / "ends.parquet")
    pairs = pd.read_parquet(d / "pairs.parquet")
    nx, ny = ra.shape
    img = np.ones((ny, nx, 3))
    grid_in = ra.inside.reshape(nx, ny).T
    grid_b = (ra.bld > 0).reshape(nx, ny).T
    img[~grid_in] = (0.97, 0.97, 0.96)
    img[grid_in & grid_b] = matplotlib.colors.to_rgb(BLD)
    ext = (ra.lo[0], ra.lo[0] + nx * an.PIX, ra.lo[1], ra.lo[1] + ny * an.PIX)
    w = ext[1] - ext[0]
    h = ext[3] - ext[2]
    fig, ax = plt.subplots(figsize=(10, 10 * h / w + 0.6), dpi=150)
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    n = len(run.paths)
    for i, p in enumerate(run.paths):
        poly = LineString(p).buffer(an.ROAD_W / 2)
        for g in shapely.get_parts(poly):
            x, y = g.exterior.xy
            ax.fill(x, y, color=ROAD, alpha=0.95 - 0.5 * i / max(n - 1, 1), lw=0)
    for r in pairs.itertuples():
        if r.ds > JOG_M:
            continue
        a = ends[(ends.ear == r.a_ear) & (ends.road == r.road)]
        b = ends[(ends.ear == r.b_ear) & (ends.road == r.road)]
        cx = (a.x.mean() + b.x.mean()) / 2
        cy = (a.y.mean() + b.y.mean()) / 2
        col = CROSS if r.ds <= an.TOL_M else JOG
        ax.scatter([cx], [cy], s=260 if r.ds <= an.TOL_M else 160, facecolors="none",
                   edgecolors=col, linewidths=2.2, zorder=5)
    ax.scatter([], [], s=120, facecolors="none", edgecolors=CROSS, linewidths=2,
               label=f"crossing (arms within {an.TOL_M:g} m along the road)")
    ax.scatter([], [], s=80, facecolors="none", edgecolors=JOG, linewidths=2,
               label=f"jogged pair ({an.TOL_M:g}-{JOG_M:g} m)")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    plot(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) == 4 else "")
