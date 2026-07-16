from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image
from pyproj import CRS
from shapely.geometry import LineString

from reblock.methods.imagery import (
    ImageryDesireLines,
    _lonlat_to_tile,
    _mosaic_extent_3857,
    detect_corridors,
    fetch_mosaic,
)

UTM = CRS.from_epsg(32734)
EXT = (2068156.7, -4026014.7, 2068615.3, -4025632.5)   # a plausible mosaic 3857 extent (spike box)


def test_lonlat_to_tile_matches_web_mercator() -> None:
    # z19 tile for the block-40972 centre (spike-verified values).
    assert _lonlat_to_tile(18.58064, -33.97795, 19) == (289204, 314812)


def test_mosaic_extent_3857_is_tile_aligned_and_ordered() -> None:
    x0, y0, x1, y1, z = 289203, 314811, 289205, 314813, 19
    xmin, ymin, xmax, ymax = _mosaic_extent_3857(x0, y0, x1, y1, z)
    assert xmin < xmax and ymin < ymax
    ts = 2 * np.pi * 6378137.0 / (2 ** z)                 # one tile in 3857 metres
    assert np.isclose(xmax - xmin, (x1 - x0 + 1) * ts)    # width spans the tile columns
    assert np.isclose(ymax - ymin, (y1 - y0 + 1) * ts)


def test_fetch_mosaic_stitches_with_injected_tiles() -> None:
    # A stub tile_getter returns a solid tile whose colour encodes (x,y) -- no network.
    def stub(z: int, x: int, y: int) -> Image.Image:
        return Image.new("RGB", (256, 256), (x % 256, y % 256, z))
    rgb, ext = fetch_mosaic(
        (18.5806, -33.9780, 18.5807, -33.9779), 19, "http://x", tile_getter=stub)
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    assert rgb.shape[0] % 256 == 0 and rgb.shape[1] % 256 == 0
    assert len(ext) == 4 and ext[0] < ext[2] and ext[1] < ext[3]


def _synthetic_scene(h: int = 384, w: int = 384) -> np.ndarray:
    # textured grey "roofs" + a bright SMOOTH tan horizontal corridor band (rows 180-205).
    rng = np.random.default_rng(0)
    img = rng.integers(70, 130, (h, w, 3)).astype(np.uint8)          # noisy grey roofs
    img[180:205, :, :] = np.array([200, 185, 150], dtype=np.uint8)   # smooth bright tan corridor
    return img


def test_detect_finds_the_smooth_corridor_band() -> None:
    lines = detect_corridors(_synthetic_scene(), EXT, UTM, min_corridor_m=1.0, min_len_m=2.0)
    assert isinstance(lines, gpd.GeoDataFrame) and lines.crs == UTM
    assert len(lines) >= 1
    # the band is horizontal, so the detected centreline spans most of the width
    widest = max(lines.geometry, key=lambda g: g.bounds[2] - g.bounds[0])
    assert (widest.bounds[2] - widest.bounds[0]) > 0.4 * (EXT[2] - EXT[0])


def test_detect_returns_empty_on_pure_texture() -> None:
    rng = np.random.default_rng(1)
    noise = rng.integers(70, 130, (384, 384, 3)).astype(np.uint8)
    lines = detect_corridors(noise, EXT, UTM, min_corridor_m=1.0, min_len_m=2.0)
    assert len(lines) == 0


def _write_geojson(p: Path, lines: list[list[tuple[float, float]]]) -> None:
    gpd.GeoDataFrame(geometry=[LineString(c) for c in lines],
                     crs=CRS.from_epsg(4326)).to_file(p, driver="GeoJSON")


def test_snapshot_loaded_without_fetch(tmp_path: Path) -> None:
    # The snapshot branch returns before fetch_mosaic is reached; an unreachable endpoint proves no
    # network is touched (it would raise if it fell through to a live fetch).
    snap = tmp_path / "cv.geojson"
    _write_geojson(snap, [[(18.74, -33.84), (18.741, -33.841)]])
    src = ImageryDesireLines(snapshot=str(snap), endpoint="http://127.0.0.1:0/unreachable")
    gdf = src.desire_lines((18.5, -34.0, 18.6, -33.9), UTM)
    assert len(gdf) == 1 and gdf.crs == UTM


def test_identity_none_when_live_stable_with_snapshot(tmp_path: Path) -> None:
    assert ImageryDesireLines().identity is None
    snap = tmp_path / "cv.geojson"
    _write_geojson(snap, [[(18.74, -33.84), (18.741, -33.841)]])
    ident = ImageryDesireLines(snapshot=str(snap)).identity
    assert ident is not None and ident[0] == "imagery"


def test_live_fetches_then_detects(tmp_path: Path) -> None:
    def stub(z: int, x: int, y: int) -> Image.Image:
        arr = np.full((256, 256, 3), 95, dtype=np.uint8)          # grey roof
        arr[120:140, :, :] = (200, 185, 150)                      # tan corridor band
        return Image.fromarray(arr)
    src = ImageryDesireLines(cache_dir=str(tmp_path), min_corridor_m=1.0, min_len_m=2.0)
    gdf = src.desire_lines((18.5806, -33.9780, 18.5807, -33.9779), UTM, _tile_getter=stub)
    assert gdf.crs == UTM and len(gdf) >= 1


def test_both_variants_instantiate_from_compare_config() -> None:
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    conf = str(Path("conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=conf):
        cfg = compose(config_name="compare_config",
                      overrides=["shapefile=x", "methods=[dream_come_true_osm,dream_come_true_cv]"])
    osm = instantiate(cfg.all_methods["dream_come_true_osm"])
    cv = instantiate(cfg.all_methods["dream_come_true_cv"])
    assert type(osm).__name__ == "DreamComeTrueReblocker"
    assert type(osm.source).__name__ == "OSMDesireLines"
    assert type(cv).__name__ == "DreamComeTrueReblocker"
    assert type(cv.source).__name__ == "ImageryDesireLines"
    assert "dream_come_true" not in cfg.all_methods       # bare key is gone (renamed)
