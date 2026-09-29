"""Render the solver's networks on the largest blocks: buildings grey, street black, new roads red at
their real 7 m width; the Lens A (10% displacement) prefix darker than the rest of the tree."""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import large_study as LS  # noqa: E402

from reblock.compare import lens_prefixes  # noqa: E402
from reblock.emit import pct_displaced  # noqa: E402
from reblock.permeability import EgressContext, permeability  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "fast_k0_lam100"
LS.build()
big = LS.by_size()[:4]


def draw(ax, block, roads, lens, title, window=None):
    polys = gpd.GeoSeries(block.buildings.outlines, crs=block.crs)
    polys.plot(ax=ax, color="#b8b8b8", linewidth=0)
    block.streets.plot(ax=ax, color="black", linewidth=0.8)
    if len(roads):
        gpd.GeoSeries(roads.geometry.buffer(3.5), crs=block.crs).plot(
            ax=ax, color="#f4a3a3", linewidth=0, alpha=0.9)
    if lens is not None and len(lens):
        gpd.GeoSeries(lens.geometry.buffer(3.5), crs=block.crs).plot(
            ax=ax, color="#c0142b", linewidth=0, alpha=0.95)
    if window is not None:
        ax.set_xlim(window[0], window[2])
        ax.set_ylim(window[1], window[3])
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    ax.set_aspect("equal")


fig, axes = plt.subplots(2, 2, figsize=(16, 16))
stats = []
for ax, bid in zip(axes.flat, big, strict=True):
    city, block = LS._BLOCKS[bid]
    roads = gpd.read_parquet(HERE / f"trees_{ARM}" / f"{bid}.parquet")
    ctx = EgressContext.of(block, LS.PCFG.params)
    lp = lens_prefixes(ctx, roads, LS.PCFG)
    km = roads.geometry.length.sum() / 1000
    d_full = pct_displaced(roads, block.buildings)
    k = int(ARM.split("_k")[1][0])
    reach = ("every parcel fronts a road" if k == 0
             else f"every parcel within {k} parcel{'s' if k > 1 else ''} of one that does")
    if ARM.startswith("2r"):
        reach += ", with two disjoint exits to different street sectors"
    title = (f"{bid} ({city}) — {len(block.buildings):,} buildings — k = {k}\n"
             f"tree {km:.1f} km, displaces {d_full:.1%} of homes; {reach}\n"
             f"dark red = Lens A prefix: {lp.displacement.geometry.length.sum() / 1000:.1f} km, "
             f"permeability {permeability(ctx, lp.displacement):.3f} at 10% displaced")
    draw(ax, block, roads, lp.displacement, title)
    stats.append((bid, block, roads, lp.displacement))
fig.tight_layout()
out = HERE / f"large_{ARM}.png"
fig.savefig(out, dpi=110)
print(out)

# close-up: a 300 m window in the densest block, around the tree's centroid
bid, block, roads, lens = stats[0]
# the SAME window for every k, so the close-ups compare: the k=0 tree's centroid
CENTRE = gpd.read_parquet(HERE / "trees_fast_k0_lam100" / f"{bid}.parquet").geometry.union_all().centroid.coords[0]
w = 150
fig, ax = plt.subplots(figsize=(11, 11))
cx, cy = CENTRE
draw(ax, block, roads, lens, f"{bid}: 300 m close-up (grey = buildings, red = new 7 m roads)",
     window=(cx - w, cy - w, cx + w, cy + w))
fig.tight_layout()
out2 = HERE / f"large_{ARM}_zoom.png"
fig.savefig(out2, dpi=120)
print(out2)
