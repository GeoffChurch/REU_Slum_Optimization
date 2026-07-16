"""Satellite-imagery desire-line source for dream_come_true_cv: fetch an Esri World Imagery tile
mosaic for a region bbox and detect the wide bare-earth corridors (classical CV -- no trained
model). See docs/superpowers/specs/2026-07-15-dream-come-true-cv-design.md."""
from __future__ import annotations

import io
import math
import urllib.request
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from PIL import Image

_R = 6378137.0
_ESRI = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile"
_UA = "reblock-dream-come-true-cv/0.1 (informal-settlement research)"
TileGetter = Callable[[int, int, int], Image.Image]


def _lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def _mosaic_extent_3857(
    x0: int, y0: int, x1: int, y1: int, z: int
) -> tuple[float, float, float, float]:
    ts = 2 * math.pi * _R / (2 ** z)          # tile size in 3857 metres
    xmin = -math.pi * _R + x0 * ts
    xmax = -math.pi * _R + (x1 + 1) * ts
    ymax = math.pi * _R - y0 * ts
    ymin = math.pi * _R - (y1 + 1) * ts
    return xmin, ymin, xmax, ymax


def _esri_tile(z: int, x: int, y: int, endpoint: str = _ESRI) -> Image.Image:
    req = urllib.request.Request(f"{endpoint}/{z}/{y}/{x}", headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as r:      # noqa: S310 (trusted endpoint)
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def fetch_mosaic(
    bbox_wgs84: tuple[float, float, float, float], zoom: int, endpoint: str,
    tile_getter: TileGetter | None = None,
) -> tuple[NDArray[np.uint8], tuple[float, float, float, float]]:
    """Fetch + stitch the Esri tiles covering `bbox_wgs84` (padded 1 tile each side) at `zoom`.
    Returns (rgb HxWx3 uint8, EPSG:3857 extent (xmin,ymin,xmax,ymax)). `tile_getter(z,x,y)->Image`
    is injectable for tests; default fetches Esri from `endpoint`."""
    def _live(z: int, x: int, y: int) -> Image.Image:
        return _esri_tile(z, x, y, endpoint)
    get = tile_getter or _live
    x0, y1 = _lonlat_to_tile(bbox_wgs84[0], bbox_wgs84[1], zoom)   # min lon / min lat
    x1, y0 = _lonlat_to_tile(bbox_wgs84[2], bbox_wgs84[3], zoom)   # max lon / max lat
    x0, y0, x1, y1 = x0 - 1, y0 - 1, x1 + 1, y1 + 1               # pad 1 tile each side
    cols, rows = x1 - x0 + 1, y1 - y0 + 1
    mosaic = Image.new("RGB", (cols * 256, rows * 256))
    for j, ty in enumerate(range(y0, y1 + 1)):
        for i, tx in enumerate(range(x0, x1 + 1)):
            mosaic.paste(get(zoom, tx, ty), (i * 256, j * 256))
    return np.asarray(mosaic, dtype=np.uint8), _mosaic_extent_3857(x0, y0, x1, y1, zoom)
