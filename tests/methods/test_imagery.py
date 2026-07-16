import numpy as np
from PIL import Image

from reblock.methods.imagery import _lonlat_to_tile, _mosaic_extent_3857, fetch_mosaic


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
