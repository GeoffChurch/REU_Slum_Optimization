from __future__ import annotations

import numpy as np
import shapely

from reblock.buildings import SpacingDiscs
from reblock.contracts import Block
from reblock.data.kblock import KblockSource
from reblock.methods.betweenness.raster import BlockRaster, home_cells
from reblock.region import region_block
from tests.scoring_fixtures import _REPO_ROOT, _block_1808


def _region_1808_1809() -> Block:
    """The adjacent DJI pair as one region block: its `streets` hold the street between them."""
    src = KblockSource(_REPO_ROOT / "tests/data/kblock/blocks_dji_sample.parquet",
                       _REPO_ROOT / "tests/data/kblock/buildings_dji_sample.parquet", "dji",
                       block_ids=["DJI.3_1_1808", "DJI.3_1_1809"], min_buildings=10,
                       building_tier=SpacingDiscs, member_buildings=None)
    return region_block(list(src.region().blocks))


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


def test_a_single_faces_street_distance_is_its_outlines_exactly() -> None:
    """A kblock face's `streets` is `[poly.boundary]`, so reading the street from `streets`
    changes nothing on one: bit for bit the distance to the outline."""
    block = _block_1808()
    r = BlockRaster.of(block, 1.0)
    ii, jj = np.nonzero(r.inside)
    pts = shapely.points(r.x0 + r.res * jj, r.y0 + r.res * ii)
    assert np.array_equal(r.edge[ii, jj], shapely.distance(pts, block.boundary.boundary))


def test_the_street_band_follows_a_regions_inter_block_streets() -> None:
    """A region block's inter-block streets run through its interior: in `streets`, not on the
    outline. Egress routes end in the band within 1.5 cells of a street, so the band must lie
    along those streets too.

    FAULT INJECTION: measuring `edge` to `block.boundary.boundary` (the outline) fails this: no
    band cell is then more than 1.5 cells from the outline.
    """
    rb = _region_1808_1809()
    streets = shapely.union_all(np.asarray(rb.streets.geometry))
    inner = streets.difference(rb.boundary.boundary.buffer(1e-6))
    assert inner.length > 20.0          # the street between the two faces, ~30 m of it
    r = BlockRaster.of(rb, 1.0)
    ii, jj = np.nonzero(r.inside)
    pts = shapely.points(r.x0 + r.res * jj, r.y0 + r.res * ii)
    assert np.allclose(r.edge[ii, jj], shapely.distance(pts, streets))
    band = r.edge[ii, jj] <= 1.5 * r.res
    along_inner = shapely.distance(pts, inner) <= 1.5 * r.res
    off_outline = shapely.distance(pts, rb.boundary.boundary) > 1.5 * r.res
    assert int((band & along_inner & off_outline).sum()) > 0
