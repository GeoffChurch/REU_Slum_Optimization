"""The Demo's escape-time plate draws damage as a fade (owner 2026-10-29): a destroyed building's
pixels are the destroyed colour, and a half-damaged one's sit halfway, in OKLab, between its
escape-time colour and it. Renders a real plate (block 41785's baseline, every building intact but
two) and reads the two buildings' pixels back.

Machine-local, like everything under explore_proto: it needs the block's baseline in the scratch
build-out directory and the screen's cached source. From the repo root:

    PYTHONPATH=. uv run python -m pytest \\
        research/roadless/atlas/explore_proto/test_damage_plate.py -p no:cacheprovider --no-cov -n0
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build as B  # noqa: E402

PIXEL_SLACK = 2 / 255        # lossy WebP's rounding, at a building's interior point


class Plate(NamedTuple):
    world: B.World
    rgb: np.ndarray          # the rendered plate, 0-1
    destroyed: int           # the building drawn wholly destroyed
    half: int                # the building drawn half damaged


@pytest.fixture(scope="module")
def plate(tmp_path_factory: pytest.TempPathFactory) -> Plate:
    import matplotlib
    from PIL import Image

    matplotlib.use("Agg")
    w = B.world(B.Site.of(B.COMPARED.bid), B.context_source())
    # Large homes, so each has interior pixels clear of its outline: of the larger half, one at the
    # top of the escape scale half damaged -- the full red, where a fade in sRGB and one in OKLab
    # part furthest (6 levels, against 2-4 elsewhere on the ramp) -- and the largest other
    # destroyed.
    area = np.array([p.area for p in w.polys])
    area[~np.isfinite(w.u_plain)] = 0
    big = np.argsort(area)[-len(area) // 2:]
    half = int(big[np.argmax(np.minimum(w.u_plain[big], w.scale) + 1e-9 * area[big])])
    destroyed = int(next(i for i in np.argsort(area)[::-1] if i != half))
    c = np.zeros(len(w.polys))
    c[destroyed], c[half] = 1.0, 0.5
    out = tmp_path_factory.mktemp("plate") / "plate.webp"
    B.homes_plate(w, w.u_plain, c=c, corridor=None, out=out)
    rgb = np.asarray(Image.open(out).convert("RGB"), dtype=float) / 255
    return Plate(world=w, rgb=rgb, destroyed=destroyed, half=half)


def _pixel(p: Plate, i: int) -> np.ndarray:
    """The plate's colour at building `i`'s interior point."""
    pt = p.world.polys[i].representative_point()
    x0, x1, y0, y1 = p.world.lim
    h, wd = p.rgb.shape[:2]
    return p.rgb[int((y1 - pt.y) / (y1 - y0) * h), int((pt.x - x0) / (x1 - x0) * wd)]


def test_a_destroyed_building_is_the_destroyed_colour(plate: Plate) -> None:
    got = _pixel(plate, plate.destroyed)
    assert np.abs(got - np.asarray(B.DESTROYED)).max() <= PIXEL_SLACK, got


def test_a_half_damaged_building_sits_halfway_to_destroyed(plate: Plate) -> None:
    i = plate.half
    norm = B.mcolors.Normalize(0.0, plate.world.scale, clip=True)
    own = np.asarray(B.SEQ(norm(plate.world.u_plain[i]))[:3])
    got = _pixel(plate, i)
    lab = B.to_oklab(got[None, :])[0]
    a, z = B.to_oklab(own[None, :])[0], B.to_oklab(np.asarray([B.DESTROYED]))[0]
    # On the OKLab segment from its own colour to destroyed, near its middle...
    t = float(np.dot(lab - a, z - a) / np.dot(z - a, z - a))
    off = float(np.linalg.norm(lab - (a + t * (z - a))))
    assert 0.4 < t < 0.6 and off < 0.02, (t, off)
    # ... and at the OKLab midpoint itself, which a blend in sRGB misses (by 6 levels here).
    want = B.from_oklab(((a + z) / 2)[None, :])[0]
    assert np.abs(got - want).max() <= PIXEL_SLACK, (got, want)
