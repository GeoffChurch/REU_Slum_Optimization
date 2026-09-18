"""Two independent renderings of each worksheet block, for adjudication by eye or by agent.

`satellite` -- Esri World Imagery with the block outlined. Carries information no screen has:
roof material, construction uniformity, vehicles, road surfacing, vegetation. This is the LESS
circular input, and the one to prefer when the two disagree.

Note the two degrade differently with block size. The schematic is VECTOR, so it stays sharp
for a 31 km2 block as readily as for a 0.3 ha one; the satellite tile is raster and capped by
the export endpoint, so past roughly 150 ha a shack is smaller than a pixel. The image says
which regime it is in, and the honest verdict on a coarse one is "unclear".

`schematic` -- building footprints, the official street network, and empty space. Deliberately
the CIRCULAR input: it shows a judge essentially what the screens themselves compute (footprint
size, density, block geometry), so verdicts from it should correlate with the metrics being
graded whether or not they are correct. Rendered so that circularity can be MEASURED -- if the
two modes agree, it did not bite; if they diverge, the satellite reading is the evidence and
the schematic reading is the artifact. Never the other way round.

A kblock block IS a street-bounded face, so every block boundary in view is an official street.
That is what the schematic draws as roads, and it is also why interior linear features in the
satellite view are footpaths rather than streets -- the distinction a judge most easily gets
wrong.

The maps links in the worksheet open at a fixed zoom, which is wrong at both ends: a 0.3 ha
block fills a pixel and a 58 ha block overflows the viewport. These images are framed to the
block's own extent, and draw its boundary, so "is this block informal" is answerable without
guessing which part of the picture is the block.

Imagery: Esri World Imagery (`services.arcgisonline.com`), the public ArcGIS REST export
endpoint. Attribution is stamped into each image. Resolution varies by area and date; where it
is too coarse to separate shacks from small formal houses, that is visible in the picture and
the adjudicator should say so rather than guess.

    pixi run python -m scripts.fetch_block_imagery              # every un-adjudicated block
    pixi run python -m scripts.fetch_block_imagery 12           # the first 12 of them
    pixi run python -m scripts.fetch_block_imagery --all        # including adjudicated ones
    pixi run python -m scripts.fetch_block_imagery --control     # the random control sample
"""
from __future__ import annotations

import csv
import math
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.image import imread  # noqa: E402
from shapely import STRtree  # noqa: E402
from shapely.geometry import box as shapely_box  # noqa: E402

from reblock.data.counts import COUNTERS  # noqa: E402
from scripts.gen_screen_bakeoff import load  # noqa: E402

WORKSHEET = Path("data/adjudication/screen_top15_worksheet.csv")
CONTROL = Path("data/adjudication/control_sample.csv")
# Which count source defines the POOL each sheet's blocks were drawn from. They differ, and a
# mismatch is not a slow path but a KeyError: the worksheet was built on the Ecopia-eligible pool
# (16,451 blocks) and the control sample on the Open Buildings one (18,309), and 1,858 blocks are
# in the second and not the first.
POOL_OF = {WORKSHEET: "kblock", CONTROL: "open_buildings"}
EXPORT = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
UA = "reblock-research/0.1 (https://github.com/GeoffChurch/REU_Slum_Optimization)"
MARGIN = 0.25          # of the block's own extent, so the surrounding fabric is visible too


TARGET_MPP = 0.35      # a shack is ~4 m across; at 0.35 m/px that is ~11 px, judgeable
MAX_PX = 2048          # 4096 returns HTTP 504 from this endpoint; 2048 is served reliably


def tile_px(span_m: float) -> int:
    """Pixels to request so the tile lands near `TARGET_MPP`, capped at the endpoint's limit.

    A fixed size was the bug this replaces: at 900 px a 0.3 ha block rendered at 0.10 m/px and a
    31 km2 one at 11.1 m/px, where a shack is a fifth of a pixel. Sixteen of sixty-eight blocks
    were too coarse to adjudicate and nothing on the image said so.
    """
    return max(600, min(MAX_PX, int(span_m / TARGET_MPP)))


def fetch(bbox: tuple[float, float, float, float], px: int, block_id: str) -> Path:
    """One imagery tile, cached on disk.

    The cache key carries the BBOX, not just the block id: bbox is what determines the content,
    and a block-keyed cache would hand back a stale tile the moment geometry, margin or pixel
    size changed. The block id is in the name too, purely so a human can tell what a cache file
    is -- which is how the resolution bug above got noticed.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_".join(f"{v:.5f}" for v in bbox)
    path = OUT / f".tile_{block_id}_{tag}_{px}.png"
    if path.exists():
        return path
    q = urllib.parse.urlencode({"bbox": ",".join(f"{v:.6f}" for v in bbox), "bboxSR": "4326",
                                "size": f"{px},{px}", "format": "png", "f": "image"})
    # A 4096 px export is tens of megabytes and routinely exceeds a 60 s socket timeout, which
    # is how the first full run died partway. Retry rather than abandon a run that has already
    # paid for most of its tiles.
    size = px
    for attempt in range(4):
        try:
            q2 = q.replace(f"size={px}%2C{px}", f"size={size}%2C{size}")
            req2 = urllib.request.Request(f"{EXPORT}?{q2}", headers={"User-Agent": UA})
            with urllib.request.urlopen(req2, timeout=300) as r:  # noqa: S310 (fixed https host)
                path.write_bytes(r.read())
            return path
        except (TimeoutError, OSError) as exc:
            if attempt == 3:
                raise
            size = max(600, size // 2)      # step down: a coarse tile beats no tile
            print(f"    {type(exc).__name__}; retrying at {size}px", flush=True)
    return path


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    every = "--all" in sys.argv
    modes = ([m for m in ("satellite", "schematic", "masked") if f"--{m}" in sys.argv]
             or ["satellite", "schematic"])
    limit = int(args[0]) if args else None
    sheet = CONTROL if "--control" in sys.argv else WORKSHEET
    # Per-sheet output root, so the top-k set and the random control set never mix on disk. They
    # answer different questions and one is the other's negatives.
    global OUT
    OUT = Path(f"data/adjudication/{'control_' if sheet is CONTROL else ''}imagery")

    rows = list(csv.DictReader(sheet.open()))
    todo = [r for r in rows if every or not r["verdict"].strip()][: limit or None]
    print(f"{len(todo)} block(s) to render")

    # The counter is chosen to reproduce the POOL the sheet was drawn from, not because this
    # script reads a count -- it only needs geometry. Get it wrong and the block_id lookup below
    # raises KeyError on whichever blocks the other pool does not contain.
    b, _ = load(COUNTERS[POOL_OF[sheet]])
    wgs = b.to_crs("EPSG:4326")
    by_id = {str(v): i for i, v in enumerate(b["block_id"])}
    OUT.mkdir(parents=True, exist_ok=True)

    foot = block_tree = None
    if "schematic" in modes:
        foot = gpd.read_parquet(
            Path.home() / ".cache/reblock/buildings_capetown_polygons.parquet").to_crs("EPSG:4326")
        foot_tree = STRtree(list(foot.geometry))
        block_tree = STRtree(list(wgs.geometry))

    for n, r in enumerate(todo, 1):
        geom = wgs.geometry.iloc[by_id[r["block_id"]]]
        x0, y0, x1, y1 = geom.bounds
        mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
        side = max((x1 - x0) + 2 * mx, (y1 - y0) + 2 * my)   # square: the export is square
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        bbox = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        box = shapely_box(*bbox)
        span_m = side * 111_320 * math.cos(math.radians(cy))
        mpp = span_m / tile_px(span_m)
        for mode in modes:
            d = OUT / mode
            d.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
            if mode in ("satellite", "masked"):
                ax.imshow(imread(fetch(bbox, tile_px(span_m), r["block_id"])),
                          extent=(bbox[0], bbox[2], bbox[1], bbox[3]))
                edge, credit, cc = "#ff2d55", "Imagery: Esri World Imagery", "white"
                if mode == "masked":
                    # Everything outside the block is painted out. This exists to TEST a specific
                    # confound: a judge shown a block inside a settlement sees informal fabric
                    # filling the frame and may label the neighbourhood rather than the block.
                    # Whether that happens is measurable -- re-judge the same blocks with the
                    # context removed and see whether verdicts move. The unmasked view stays the
                    # default, because context is genuinely informative (it is how a judge tells
                    # an upgraded row from a formal one); this is the control, not a replacement.
                    # The hole is cut in SHAPELY, not by a matplotlib compound path. The first
                    # attempt built one path from the bbox ring plus a reversed block ring and
                    # relied on the fill rule to leave a hole; it painted over the block as well,
                    # so every masked image was a blank rectangle. A geometric difference cannot
                    # get the winding wrong.
                    gpd.GeoSeries([box.difference(geom)], crs="EPSG:4326").plot(
                        ax=ax, color="#efeae1", linewidth=0, zorder=5)
                    credit = "context masked - judge ONLY the exposed fabric"
            else:
                ax.set_facecolor("#f2f0eb")                       # empty space
                assert foot is not None and block_tree is not None
                near = foot.geometry.iloc[foot_tree.query(box, predicate="intersects")]
                gpd.GeoSeries(near, crs="EPSG:4326").plot(ax=ax, color="#2b2b2b", linewidth=0)
                streets = wgs.geometry.iloc[block_tree.query(box, predicate="intersects")]
                gpd.GeoSeries(streets, crs="EPSG:4326").boundary.plot(
                    ax=ax, color="#4a90d9", linewidth=1.4)
                credit = "dark = buildings · blue = official streets · pale = open ground"
                edge, cc = "#ff2d55", "#444"
            gpd.GeoSeries([geom], crs="EPSG:4326").boundary.plot(
                ax=ax, color=edge, linewidth=2.4, zorder=6)
            ax.set_xlim(bbox[0], bbox[2])
            ax.set_ylim(bbox[1], bbox[3])
            ax.set_xticks([])
            ax.set_yticks([])
            # The resolution is stated because it decides whether the picture can answer the
            # question at all: a shack is ~4 m, so past ~1 m/px it is a smudge and the honest
            # verdict is "unclear", not a guess.
            raster = mode in ("satellite", "masked")
            warn = "   ⚠ TOO COARSE FOR SHACKS" if raster and mpp > 1.0 else ""
            res = f"   {mpp:.2f} m/px{warn}" if raster else ""
            ax.set_title(f"{r['block_id']}  ·  {r['place']}  ·  {r['area_ha']} ha{res}",
                         fontsize=11)
            # Below the axes, not inside them: in the schematic the bottom-left corner is data.
            ax.set_xlabel(credit, fontsize=6, color=cc, loc="left")
            fig.savefig(d / f"{r['block_id']}.png", bbox_inches="tight")
            plt.close(fig)
        print(f"  [{n}/{len(todo)}] {r['block_id']}  ({', '.join(modes)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
