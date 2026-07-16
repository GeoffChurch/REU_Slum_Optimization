"""Satellite-imagery desire-line source for dream_come_true_cv: fetch an Esri World Imagery tile
mosaic for a region bbox and detect the settlement's MAJOR bare-earth corridors -- its main
thoroughfares -- with classical CV (no trained model). This deliberately does NOT recover the fine
interior alley network: at ~0.25 m/px the narrow alleys are ~2-4 px, shadowed, and tonally identical
to the packed metal roofs, putting them below the classical signal floor (six detection approaches
confirmed this -- see the design doc's "Detection limits" section). The wide corridors it does find
are the settlement's primary desire lines, which is an honest, useful reblocking input.
See docs/superpowers/specs/2026-07-15-dream-come-true-cv-design.md."""
from __future__ import annotations

import hashlib
import io
import math
import urllib.request
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
from matplotlib.colors import rgb_to_hsv
from numpy.typing import NDArray
from PIL import Image
from pyproj import CRS
from scipy import ndimage
from shapely.geometry import LineString
from skimage.morphology import closing, disk, skeletonize

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


def _keep_large_components(mask: NDArray[np.bool_], min_area: int) -> NDArray[np.bool_]:
    """Keep only connected components whose pixel area is >= `min_area` (the "wide corridor" filter:
    a wide corridor is a large blob; scattered between-shack earth patches are small)."""
    lbl, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lbl.ravel())
    keep = sizes >= min_area
    keep[0] = False                                                 # label 0 is background
    return keep[lbl]


def detect_corridors(
    rgb: NDArray[np.uint8], extent_3857: tuple[float, float, float, float], crs: CRS, *,
    min_corridor_m: float = 3.0, min_len_m: float = 8.0, warm_thr: float = 0.05,
    std_thr: float = 0.085,
) -> gpd.GeoDataFrame:
    """Detect the settlement's MAJOR bare-earth corridors (see the module docstring for why the fine
    interior network is out of scope). Bare earth reads WARM (red channel notably above blue) and
    SMOOTH; a genuine corridor is a LARGE connected warm-earth component, whereas scattered
    between-shack earth patches are small and get area-filtered out. Pipeline: warm & smooth & not-
    green & bright -> closing (bridge ruts) -> area filter (keep corridors of at least `min_len_m` x
    `min_corridor_m`) -> skeletonize -> vectorize -> LineStrings in `crs`."""
    h, w = rgb.shape[:2]
    f = rgb.astype(np.float64) / 255.0
    r_ch, b_ch = f[..., 0], f[..., 2]
    gray = f.mean(2)
    mean = ndimage.uniform_filter(gray, 9)
    std = np.sqrt(np.clip(ndimage.uniform_filter(gray * gray, 9) - mean * mean, 0.0, None))
    hsv = rgb_to_hsv(f)
    green = (hsv[..., 0] > 0.18) & (hsv[..., 0] < 0.45) & (hsv[..., 1] > 0.20)
    mask = ((r_ch - b_ch) > warm_thr) & (std < std_thr) & (gray > 0.28) & (gray < 0.82) & (~green)

    mpp = _ground_mpp(extent_3857, w)
    mask = closing(mask, disk(max(1, int(round(0.5 / mpp)))))       # bridge ruts (~0.5 m)
    min_area = int((min_len_m / mpp) * (min_corridor_m / mpp))      # a min_len x min_corridor blob
    mask = _keep_large_components(np.asarray(mask, dtype=bool), min_area)
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


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "reblock" / "imagery"


@dataclass
class ImageryDesireLines:
    """A DesireLineSource that detects wide bare-earth corridors from Esri World Imagery. Fetch
    precedence: a committed `snapshot` GeoJSON of already-detected lines (byte-stable, offline) -> a
    disk cache -> a live mosaic fetch + detect. `identity` is None when live (uncacheable), and a
    stable tuple keyed on the snapshot's content hash when a snapshot is pinned (mirrors OSM)."""

    zoom: int = 19
    endpoint: str = _ESRI
    cache_dir: str | None = None
    snapshot: str | None = None
    min_corridor_m: float = 3.0
    min_len_m: float = 8.0

    @property
    def identity(self) -> Hashable:
        if self.snapshot is None:
            return None
        digest = hashlib.sha256(Path(self.snapshot).read_bytes()).hexdigest()[:16]
        return ("imagery", self.zoom, self.min_corridor_m, self.min_len_m, digest)

    def _cache_path(self, bbox_wgs84: tuple[float, float, float, float]) -> Path:
        root = Path(self.cache_dir) if self.cache_dir else _default_cache_dir()
        bbox = ",".join(f"{c:.5f}" for c in bbox_wgs84)
        key = f"z{self.zoom}c{self.min_corridor_m}l{self.min_len_m}@{bbox}"
        return root / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.geojson"

    def desire_lines(
        self, bbox_wgs84: tuple[float, float, float, float], crs: CRS,
        _tile_getter: TileGetter | None = None,
    ) -> gpd.GeoDataFrame:
        if self.snapshot is not None:
            return gpd.read_file(self.snapshot).to_crs(crs)
        cache_path = self._cache_path(bbox_wgs84)
        if cache_path.exists():
            return gpd.read_file(cache_path).to_crs(crs)
        rgb, ext = fetch_mosaic(bbox_wgs84, self.zoom, self.endpoint, tile_getter=_tile_getter)
        lines = detect_corridors(rgb, ext, crs, min_corridor_m=self.min_corridor_m,
                                 min_len_m=self.min_len_m)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lines.to_crs(4326).to_file(cache_path, driver="GeoJSON")
        return lines
