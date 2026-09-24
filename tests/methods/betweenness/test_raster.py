from __future__ import annotations

import numpy as np
import shapely

from reblock.methods.betweenness.raster import BlockRaster, home_cells
from tests.scoring_fixtures import _block_1808


def test_clearance_is_the_exact_distance_to_the_nearest_outline() -> None:
    block = _block_1808()
    r = BlockRaster.of(block, 1.0)
    ii, jj = np.nonzero(r.inside)
    k = np.linspace(0, len(ii) - 1, 50).astype(int)
    pts = shapely.points(r.x0 + r.res * jj[k], r.y0 + r.res * ii[k])
    want = np.array([block.buildings.outlines.distance(p).min() for p in pts])
    assert np.allclose(r.clearance[ii[k], jj[k]], want)
    assert np.isnan(r.clearance[~r.inside]).all() and np.isnan(r.edge[~r.inside]).all()
    assert (r.clearance[r.inside] >= 0).all()


def test_every_home_is_a_free_cell_nearest_its_building() -> None:
    block = _block_1808()
    r = BlockRaster.of(block, 1.0)
    homes = home_cells(r, block)
    assert homes.shape == (len(block.buildings), 2)
    assert r.free[homes[:, 0], homes[:, 1]].all()
    fr = np.argwhere(r.free)
    for (i, j), (x, y) in zip(homes[:5], block.buildings.xy[:5], strict=True):
        d = np.hypot(r.x0 + r.res * fr[:, 1] - x, r.y0 + r.res * fr[:, 0] - y)
        assert np.isclose(np.hypot(r.x0 + r.res * j - x, r.y0 + r.res * i - y), d.min())
