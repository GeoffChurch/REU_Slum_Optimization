"""Which corridors did a clearing open? The straight-run field after the greedy's 10% clearing
minus before it: per cell, the gain in the mean over 48 headings of R / (R + r0) (R = soft
expected free path, kappa 2, r0 20 m -- the same field for every run, whatever conductance it
was optimized under), and the heading that gained most, shown only on space that was already
open (the cleared footprints, green, gain trivially). A new corridor shows as a coherent streak
running through the cleared buildings into the open space beyond.

    PYTHONPATH=. pixi run python research/roadless/opened_map.py <block id> [picker] [alongs,]
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402
from clear import cleared_through, rows_dir  # noqa: E402

D_LENS = 0.10
FIELD = lifted.SoftSightline(beta=1.0, kappa=2.0, r0_m=20.0)


def gained(g: lifted.Grid, before: np.ndarray, after: np.ndarray, K: int = 8):
    """Per cell, the gain in g(R) = R / (R + r0) after minus before, three ways: the MAX over the
    48 fine headings (a whole unblocked line lights up along its length) with its heading in
    [0, 1); the max over the K layers of the layer factor's gain (what the metric feels: each
    layer averages ~6 nearby headings); and the MEAN over headings (the segment measure; falls
    off like 1/distance from a cleared building, since only the headings through it gain)."""
    _dirs, ang = FIELD._dirs()
    W = lifted._angle_weights(K, ang)
    best = np.zeros(before.shape)
    arg = np.zeros(before.shape)
    layer = np.zeros((K, *before.shape))
    mean = np.zeros(before.shape)
    for (i, R0), (_i, R1) in zip(FIELD.runs(before, g.h), FIELD.runs(after, g.h), strict=True):
        d = R1 / (R1 + FIELD.r0_m) - R0 / (R0 + FIELD.r0_m)
        up = d > best
        best = np.where(up, d, best)
        arg = np.where(up, ang[i] / np.pi, arg)
        mean += d
        for k in np.flatnonzero(W[:, i]):
            layer[k] += W[k, i] * d
    return best, arg, layer.max(axis=0), mean / len(ang)


def main(bid: str, picker: str, alongs: list[str]) -> None:
    [b] = common.build_blocks([bid])
    polys = np.asarray(b.buildings.outlines)
    g = lifted.Grid.of(b.boundary, polys, list(b.streets.geometry), 0.5)
    lab = g.label_sub(polys)
    ext = (g.x0 - g.h / 2, g.x0 + (g.inside.shape[1] - 0.5) * g.h,
           g.y0 - g.h / 2, g.y0 + (g.inside.shape[0] - 0.5) * g.h)
    big = len(polys) > 1500
    ncol = 4
    fig, axes = plt.subplots(len(alongs), ncol, figsize=(11 * ncol, 11 * len(alongs)),
                             squeeze=False)
    window = None
    for row, along in enumerate(alongs):
        rows = pd.read_parquet(rows_dir(picker, 0.5, "area", 2.0, along)
                               / f"{bid}.parquet").sort_values("step")
        stop = rows[rows.D >= D_LENS - 1e-9].iloc[0]
        cl = cleared_through(rows, int(stop.step))
        after = (g.isub & (~g.bsub | np.isin(lab, cl))).mean(axis=-1)
        best, hue, layer, mean = gained(g, g.ff0, after)
        if window is None:
            cx = np.median([p.centroid.x for p in polys[cl]])
            cy = np.median([p.centroid.y for p in polys[cl]])
            minx, miny, maxx, maxy = b.boundary.bounds
            half = 0.125 * max(maxx - minx, maxy - miny)
            window = (cx - half, cx + half, cy - half, cy + half)
        # only space that was ALREADY open: the cleared footprints themselves trivially gain
        was_open = g.ff0 >= 0.5
        fields = dict(line=(best, "max over 48 headings (lines)"),
                      layer=(layer, "max over the 8 layers (what the metric feels)"),
                      mean=(mean, "mean over 48 headings (segment measure)"))
        tops = {k: np.quantile(v[was_open & (v > 0)], 0.99) for k, (v, _t) in fields.items()}
        rgb = mcolors.hsv_to_rgb(np.stack([hue, np.full_like(hue, 0.9),
                                           np.clip(best / tops["line"], 0, 1)], axis=-1))
        rgb[~was_open] = 1.0
        panels = [("zoom", "line"), ("zoom", "heading"), ("zoom", "layer"), ("zoom", "mean")] \
            if big else [("full", "line"), ("full", "heading"), ("full", "layer"),
                         ("full", "mean")]
        for col, (where, what) in enumerate(panels):
            ax = axes[row, col]
            if what in fields:
                v, label = fields[what]
                im = ax.imshow(np.where(was_open, v, np.nan), origin="lower", extent=ext,
                               cmap="magma", vmin=0, vmax=tops[what], interpolation="nearest")
                fig.colorbar(im, ax=ax, shrink=0.5, label="gain in R/(R+r0)")
            else:
                label = "heading of the max (hue), its size (brightness)"
                ax.imshow(rgb, origin="lower", extent=ext, interpolation="nearest")
            gpd.GeoSeries(list(np.delete(polys, cl))).plot(ax=ax, color="0.55", lw=0)
            gpd.GeoSeries(list(polys[cl])).plot(ax=ax, color="#2ca02c", lw=0)
            gpd.GeoSeries([b.boundary]).boundary.plot(ax=ax, color="k", lw=1)
            gpd.GeoSeries(list(b.streets.geometry)).plot(ax=ax, color="r", lw=2)
            ax.set_axis_off()
            ax.set_aspect("equal")
            if where == "zoom":
                ax.set_xlim(window[0], window[1])
                ax.set_ylim(window[2], window[3])
            ax.set_title(f"{along}: {len(cl)} cleared (green), D {stop.D:.3f}\n{label}",
                         fontsize=13)
    fig.suptitle(f"{bid}, greedy {picker}: what the 10% clearing opened", fontsize=16)
    fig.tight_layout()
    out = HERE / f"opened_{bid}_{picker}.png"
    fig.savefig(out, dpi=80)
    print(out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "S0.01cat",
         sys.argv[3].split(",") if len(sys.argv) > 3 else ["uni", "ss10k2", "ss100k2"])
