"""Satellite-imagery desire-line source for dream_come_true_cv: fetch an Esri World Imagery tile
mosaic for a region bbox and detect the wide bare-earth corridors (classical CV -- no trained
model). See docs/superpowers/specs/2026-07-15-dream-come-true-cv-design.md."""
from __future__ import annotations

import io
import math
import urllib.request
from collections.abc import Callable

import geopandas as gpd
import networkx as nx
import numpy as np
from matplotlib.colors import rgb_to_hsv
from numpy.typing import NDArray
from PIL import Image
from pyproj import CRS
from scipy import ndimage
from shapely.geometry import LineString
from skimage.morphology import binary_opening, disk, remove_small_objects, skeletonize

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


def _ground_mpp(extent_3857: tuple[float, float, float, float], width_px: int) -> float:
    """Ground metres per pixel. 3857 is Web-Mercator-stretched, so correct by latitude at the
    mosaic centre: ground = (3857 width / px) * cos(lat)."""
    xmin, ymin, xmax, ymax = extent_3857
    lat = 2 * math.atan(math.exp((ymin + ymax) / 2 / _R)) - math.pi / 2
    return (xmax - xmin) / width_px * math.cos(lat)


def _skeleton_to_lines(skel: NDArray[np.bool_]) -> list[list[tuple[int, int]]]:
    """1-px skeleton -> polylines. Build an 8-neighbour pixel graph; each polyline is a chain of
    degree-2 pixels between two non-degree-2 nodes (junctions/endpoints), plus any pure loops."""
    ys, xs = np.nonzero(skel)
    pix = set(zip((int(v) for v in ys), (int(v) for v in xs), strict=True))
    g: nx.Graph = nx.Graph()
    g.add_nodes_from(pix)
    for (y, x) in pix:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy or dx) and (y + dy, x + dx) in pix:
                    g.add_edge((y, x), (y + dy, x + dx))

    def walk(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
        chain, prev, cur = [a, b], a, b
        while g.degree(cur) == 2:
            nxts = [z for z in g.neighbors(cur) if z != prev]
            if not nxts or nxts[0] == a:
                break
            prev, cur = cur, nxts[0]
            chain.append(cur)
        return chain

    lines: list[list[tuple[int, int]]] = []
    seen: set[frozenset[tuple[int, int]]] = set()
    for node in [n for n in g.nodes if g.degree(n) != 2]:
        for nb in g.neighbors(node):
            if frozenset((node, nb)) in seen:
                continue
            chain = walk(node, nb)
            seen.update(frozenset(e) for e in zip(chain, chain[1:], strict=False))
            lines.append(chain)
    for a, b in g.edges:                    # leftover pure loops
        if frozenset((a, b)) not in seen:
            chain = walk(a, b)
            seen.update(frozenset(e) for e in zip(chain, chain[1:], strict=False))
            lines.append(chain)
    return lines


def detect_corridors(
    rgb: NDArray[np.uint8], extent_3857: tuple[float, float, float, float], crs: CRS, *,
    min_corridor_m: float = 3.0, min_len_m: float = 8.0, smooth_sigma: float = 0.10,
    shadow_v: float = 0.28, lik_thr: float = 0.35,
) -> gpd.GeoDataFrame:
    """Detect wide bare-earth corridors: likelihood (bright*smooth*not-green*not-shadow) ->
    threshold
    -> wide-disk opening (keeps only WIDE bare earth) -> skeletonize -> vectorize -> LineStrings in
    `crs`. Only the main corridors survive; the fine interior network is out of scope by design."""
    h, w = rgb.shape[:2]
    f = rgb.astype(np.float64) / 255.0
    gray = f.mean(2)
    mean = ndimage.uniform_filter(gray, 7)
    var = ndimage.uniform_filter(gray * gray, 7) - mean * mean
    smooth = np.clip(1.0 - np.sqrt(np.clip(var, 0, None)) / smooth_sigma, 0.0, 1.0)
    hsv = rgb_to_hsv(f)
    green = (hsv[..., 0] > 0.18) & (hsv[..., 0] < 0.45) & (hsv[..., 1] > 0.22)
    lik = hsv[..., 2] * smooth * (~green)
    lik[hsv[..., 2] < shadow_v] = 0.0

    mpp = _ground_mpp(extent_3857, w)
    r = max(1, int(round((min_corridor_m / 2) / mpp)))
    mask = lik > lik_thr
    mask = binary_opening(mask, disk(r))
    mask = remove_small_objects(mask, min_size=int((min_corridor_m / mpp) ** 2))
    skel = skeletonize(mask)

    xmin, ymin, xmax, ymax = extent_3857
    geoms = []
    for chain in _skeleton_to_lines(np.asarray(skel, dtype=bool)):
        if len(chain) < 2:
            continue
        pts = [(xmin + (c + 0.5) / w * (xmax - xmin), ymax - (rr + 0.5) / h * (ymax - ymin))
               for (rr, c) in chain]
        geoms.append(LineString(pts))
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=CRS.from_epsg(3857)).to_crs(crs)
    simplified = [g.simplify(mpp) for g in gdf.geometry]        # drop pixel jitter (~1 px)
    kept = [g for g in simplified if g.length >= min_len_m]
    return gpd.GeoDataFrame(geometry=kept, crs=crs)
