"""Aerial imagery per worksheet block, with the block outlined, for adjudication by eye.

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
"""
from __future__ import annotations

import csv
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.image import imread  # noqa: E402

from scripts.gen_screen_bakeoff import load  # noqa: E402

WORKSHEET = Path("data/adjudication/screen_top15_worksheet.csv")
OUT = Path("data/adjudication/imagery")
EXPORT = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
UA = "reblock-research/0.1 (https://github.com/GeoffChurch/REU_Slum_Optimization)"
MARGIN = 0.25          # of the block's own extent, so the surrounding fabric is visible too


def fetch(bbox: tuple[float, float, float, float], px: int = 900) -> Path:
    """One imagery tile for a WGS84 bbox, cached on disk by bbox so a re-run is free."""
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_".join(f"{v:.5f}" for v in bbox)
    path = OUT / f".tile_{tag}.png"
    if path.exists():
        return path
    q = urllib.parse.urlencode({"bbox": ",".join(f"{v:.6f}" for v in bbox), "bboxSR": "4326",
                                "size": f"{px},{px}", "format": "png", "f": "image"})
    req = urllib.request.Request(f"{EXPORT}?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:      # noqa: S310 (fixed https host)
        path.write_bytes(r.read())
    return path


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    every = "--all" in sys.argv
    limit = int(args[0]) if args else None

    rows = list(csv.DictReader(WORKSHEET.open()))
    todo = [r for r in rows if every or not r["verdict"].strip()][: limit or None]
    print(f"{len(todo)} block(s) to render")

    b, _ = load()
    wgs = b.to_crs("EPSG:4326")
    by_id = {str(v): i for i, v in enumerate(b["block_id"])}
    OUT.mkdir(parents=True, exist_ok=True)

    for n, r in enumerate(todo, 1):
        geom = wgs.geometry.iloc[by_id[r["block_id"]]]
        x0, y0, x1, y1 = geom.bounds
        mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
        side = max((x1 - x0) + 2 * mx, (y1 - y0) + 2 * my)   # square: the export is square
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        bbox = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        tile = fetch(bbox)

        fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
        ax.imshow(imread(tile), extent=(bbox[0], bbox[2], bbox[1], bbox[3]))
        gpd.GeoSeries([geom], crs="EPSG:4326").boundary.plot(
            ax=ax, color="#ff2d55", linewidth=2.2)
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{r['block_id']}  ·  {r['place']}\n"
                     f"survey={r['survey_label']}  cover={r['survey_cover']}  "
                     f"OB={r['ob_count']} bldgs  median {r['ob_median_m2']} m²  "
                     f"fine depth={r['fine_depth']}", fontsize=10)
        ax.text(0.005, 0.005, "Imagery: Esri World Imagery", transform=ax.transAxes,
                fontsize=6, color="white", ha="left", va="bottom")
        out = OUT / f"{r['block_id']}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{n}/{len(todo)}] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
