"""What three road-favouring conductances would see on a block, from geometry alone (no solves).

  width    local clear width, 2 x the distance to the nearest building or the block edge: the
           multiplier under which a channel conducts like width^2.
  sight    the longest straight run through a cell over the K headings along which the clear
           width stays >= W_WALK: the multiplier that favours long straight corridors.
  vehicle  where clear width >= W (4.5 m), split by whether that space touches a street: the
           fast layer a vehicle mode would add.

    PYTHONPATH=. pixi run python research/roadless/conductance_map.py <block id> [picker]
    PYTHONPATH=. pixi run python research/roadless/conductance_map.py sightline <block id> [picker]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from numba import njit  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402

W_VEHICLE = 4.5
W_WALK = 1.5          # a sightline has to be walkable: width at least this all along it
STREET_M = 10.0       # outside the block within this of a street centreline counts as open


@njit(cache=True)
def _runs(clear, dx, dy):
    """Per cell: the number of clear cells in the straight lattice run through it along (dx, dy)
    (every cell a step crosses must be clear; checked at the step's endpoints and midpoint)."""
    ny, nx = clear.shape
    fwd = np.zeros((ny, nx), dtype=np.int64)
    bwd = np.zeros((ny, nx), dtype=np.int64)
    # forward: successor is (r + dy, c + dx); dy >= 0, and dx > 0 when dy == 0, so visiting rows
    # in decreasing order (columns in decreasing order within a row) settles successors first
    for r in range(ny - 1, -1, -1):
        for c in range(nx - 1, -1, -1):
            if not clear[r, c]:
                continue
            fwd[r, c] = 1
            r2, c2 = r + dy, c + dx
            rm, cm = r + dy // 2, c + dx // 2
            if 0 <= r2 < ny and 0 <= c2 < nx and clear[r2, c2] and clear[rm, cm] and \
                    clear[r + (dy + 1) // 2, c + (dx + 1) // 2]:
                fwd[r, c] += fwd[r2, c2]
    for r in range(ny):
        for c in range(nx):
            if not clear[r, c]:
                continue
            bwd[r, c] = 1
            r2, c2 = r - dy, c - dx
            rm, cm = r - dy // 2, c - dx // 2
            if 0 <= r2 < ny and 0 <= c2 < nx and clear[r2, c2] and clear[rm, cm] and \
                    clear[r - (dy + 1) // 2, c - (dx + 1) // 2]:
                bwd[r, c] += bwd[r2, c2]
    return fwd + bwd - 1


def fields(g: lifted.Grid, streets, K: int = 8) -> dict[str, np.ndarray]:
    """Width is measured to buildings and to the block edge, except where the edge is a street
    (the street is open: a lane widening onto it is not narrowed by it)."""
    clear = g.ff0 >= 0.5
    st = shapely.union_all(np.asarray(streets)).buffer(STREET_M)
    shapely.prepare(st)
    out = ~g.inside
    street_side = np.zeros_like(out)
    street_side[out] = shapely.contains_xy(st, g.xy[..., 0][out], g.xy[..., 1][out])
    width = 2.0 * ndimage.distance_transform_edt(clear | street_side) * g.h
    walk = clear & (width >= W_WALK)
    v, _th, _m, _gap = lifted.axes(K)
    sight = np.zeros(clear.shape)
    for dx, dy in v:
        n = _runs(walk, int(dx), int(dy))
        sight = np.maximum(sight, n * np.hypot(dx, dy) * g.h)
    wide = width >= W_VEHICLE
    lab, _ = ndimage.label(wide)
    on_street = np.unique(lab[wide & g.ground])
    vehicle = np.where(wide, np.where(np.isin(lab, on_street[on_street > 0]), 2.0, 1.0), 0.0)
    nan = ~clear
    return {k: np.where(nan, np.nan, f) for k, f in
            dict(width=width, sight=sight, vehicle=vehicle).items()}


def main(bid: str, picker: str) -> None:
    from clear import cleared_through, rows_dir
    [b] = common.build_blocks([bid])
    polys = np.asarray(b.buildings.outlines)
    g = lifted.Grid.of(b.boundary, polys, list(b.streets.geometry), 0.5)
    F = fields(g, list(b.streets.geometry))
    ext = (g.x0 - g.h / 2, g.x0 + (g.inside.shape[1] - 0.5) * g.h,
           g.y0 - g.h / 2, g.y0 + (g.inside.shape[0] - 0.5) * g.h)
    # zoom: the window render_clear uses (around the median building the greedy cleared by 10%)
    rows = pd.read_parquet(rows_dir(picker, 0.5, "area", 2.0) / f"{bid}.parquet")
    stop = rows[rows.D >= 0.10 - 1e-9].step.min()
    cl = polys[cleared_through(rows, int(stop))]
    mx = float(np.median([p.centroid.x for p in cl]))
    my = float(np.median([p.centroid.y for p in cl]))
    minx, miny, maxx, maxy = b.boundary.bounds
    half = 0.125 * max(maxx - minx, maxy - miny)
    spec = dict(
        width=dict(cmap="viridis", norm=mcolors.LogNorm(0.5, 30),
                   title="clear width (m): width^2 conduction"),
        sight=dict(cmap="magma", norm=mcolors.LogNorm(2, 300),
                   title=f"longest straight run >= {W_WALK:g} m wide (m): sightline conduction"),
        vehicle=dict(cmap=mcolors.ListedColormap(["#eeeeee", "#9ecae1", "#08519c"]),
                     norm=mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3),
                     title=f"width >= {W_VEHICLE:g} m: light = isolated, dark = reaches a street"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(33, 22))
    for col, (name, s) in enumerate(spec.items()):
        for row in range(2):
            ax = axes[row, col]
            im = ax.imshow(F[name], origin="lower", extent=ext, cmap=s["cmap"], norm=s["norm"],
                           interpolation="nearest")
            gpd.GeoSeries(list(polys)).plot(ax=ax, color="0.35", lw=0)
            gpd.GeoSeries([b.boundary]).boundary.plot(ax=ax, color="k", lw=1)
            gpd.GeoSeries(list(b.streets.geometry)).plot(ax=ax, color="r", lw=2)
            ax.set_axis_off()
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(s["title"], fontsize=16)
                ax.add_patch(plt.Rectangle((mx - half, my - half), 2 * half, 2 * half,
                                           fill=False, ec="#2ca02c", lw=2))
                if name != "vehicle":
                    fig.colorbar(im, ax=ax, shrink=0.5)
            else:
                ax.set_xlim(mx - half, mx + half)
                ax.set_ylim(my - half, my + half)
    fig.suptitle(f"{bid}: before any clearing (buildings dark grey, streets red)", fontsize=18)
    fig.tight_layout()
    out = HERE / f"conductance_{bid}.png"
    fig.savefig(out, dpi=90)
    print(out, flush=True)


def sightline(bid: str, picker: str) -> None:
    """The runs the Sightline factor is built from: their mean over all headings (the segment
    measure) and their dominant orientation (the doubled-angle mean), full block and zoomed."""
    from clear import cleared_through, rows_dir
    [b] = common.build_blocks([bid])
    polys = np.asarray(b.buildings.outlines)
    g = lifted.Grid.of(b.boundary, polys, list(b.streets.geometry), 0.5)
    sl = lifted.Sightline(1.0)
    ang, R = sl.runs(g.ff0, g.h)
    strength = R.mean(axis=0)                     # the segment measure: all headings alike
    c2 = np.tensordot(np.cos(2 * ang), R, axes=1) / len(ang)
    s2 = np.tensordot(np.sin(2 * ang), R, axes=1) / len(ang)
    heading = (np.arctan2(s2, c2) / 2) % np.pi / np.pi
    aniso = np.hypot(c2, s2)                      # how much of it is along one orientation
    clear = g.ff0 >= sl.clear_frac
    rgb = mcolors.hsv_to_rgb(np.stack([heading, np.full_like(heading, 0.9),
                                       np.clip(aniso / 0.25, 0, 1)], axis=-1))
    rgb[~clear] = 1.0
    ext = (g.x0 - g.h / 2, g.x0 + (g.inside.shape[1] - 0.5) * g.h,
           g.y0 - g.h / 2, g.y0 + (g.inside.shape[0] - 0.5) * g.h)
    rows = pd.read_parquet(rows_dir(picker, 0.5, "area", 2.0) / f"{bid}.parquet")
    stop = rows[rows.D >= 0.10 - 1e-9].step.min()
    cl = polys[cleared_through(rows, int(stop))]
    mx = float(np.median([p.centroid.x for p in cl]))
    my = float(np.median([p.centroid.y for p in cl]))
    minx, miny, maxx, maxy = b.boundary.bounds
    half = 0.125 * max(maxx - minx, maxy - miny)
    fig, axes = plt.subplots(2, 2, figsize=(22, 22))
    for row in range(2):
        for col in range(2):
            ax = axes[row, col]
            if col == 0:
                im = ax.imshow(np.where(clear, strength, np.nan), origin="lower", extent=ext,
                               cmap="magma", norm=mcolors.LogNorm(0.02, 1),
                               interpolation="nearest")
                if row == 0:
                    fig.colorbar(im, ax=ax, shrink=0.5, label="mean capped run fraction")
                    ax.set_title(f"segment measure: clear run through the cell averaged over "
                                 f"{len(ang)} headings (cap {sl.r_max_m:g} m)", fontsize=16)
            else:
                ax.imshow(rgb, origin="lower", extent=ext, interpolation="nearest")
                if row == 0:
                    ax.set_title("dominant orientation (hue) and how one-directional the "
                                 "runs are (brightness)", fontsize=16)
            gpd.GeoSeries(list(polys)).plot(ax=ax, color="0.35", lw=0)
            gpd.GeoSeries([b.boundary]).boundary.plot(ax=ax, color="k", lw=1)
            gpd.GeoSeries(list(b.streets.geometry)).plot(ax=ax, color="r", lw=2)
            ax.set_axis_off()
            ax.set_aspect("equal")
            if row == 1:
                ax.set_xlim(mx - half, mx + half)
                ax.set_ylim(my - half, my + half)
            else:
                ax.add_patch(plt.Rectangle((mx - half, my - half), 2 * half, 2 * half,
                                           fill=False, ec="#2ca02c", lw=2))
    fig.suptitle(f"{bid}: straight clear runs, before any clearing", fontsize=18)
    fig.tight_layout()
    out = HERE / f"sightline_{bid}.png"
    fig.savefig(out, dpi=90)
    print(out, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "sightline":
        sightline(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "B0.01g3")
    else:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "B0.01g3")
