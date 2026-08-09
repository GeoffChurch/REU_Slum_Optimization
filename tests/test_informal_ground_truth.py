"""Guards for the informal-settlement ground truth. Fault-injected: each named fault was applied,
the test confirmed to fail, and the code restored.

These do NOT download. `settlement_extents` needs an 18 MB shapefile from Edinburgh DataShare, so
the network-touching path is exercised by the example generator, not the suite; what is guarded here
is the clustering and labelling logic, on synthetic geometry.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import Polygon, box

from reblock.data.informal import EPS_M, MIN_SAMPLES, _dbscan, label_blocks

UTM = CRS.from_epsg(32734)


def _blob(cx: float, cy: float, n: int, spread: float = 8.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(cx, spread, n), rng.normal(cy, spread, n)])


def test_two_separated_blobs_are_two_clusters() -> None:
    """The basic contract. The blobs are 500 m apart, far beyond EPS_M.

    FAULT INJECTION: dropping the core-to-core edge filter merges them via any stray between.
    """
    xy = np.vstack([_blob(0, 0, 60, seed=1), _blob(500, 0, 60, seed=2)])
    lab = _dbscan(xy)
    assert lab.max() == 1, f"expected 2 clusters, got {lab.max() + 1}"
    assert set(np.unique(lab[:60])) == {0} or set(np.unique(lab[:60])) == {1}
    assert len(set(np.unique(lab[:60])) & set(np.unique(lab[60:]))) == 0


def test_an_isolated_point_is_noise_not_its_own_settlement() -> None:
    """A lone structure far from anything must be -1, not a one-member cluster.

    FAULT INJECTION: `deg >= 1` in place of `deg >= MIN_SAMPLES` makes the stray a cluster.
    """
    xy = np.vstack([_blob(0, 0, 60, seed=3), np.array([[9_000.0, 9_000.0]])])
    lab = _dbscan(xy)
    assert lab[-1] == -1
    assert lab.max() == 0


def test_border_points_are_assigned_not_discarded() -> None:
    """Border assignment is load-bearing: it pushed 21 of 189 real settlements over MIN_STRUCTURES.
    A point just inside EPS_M of a dense core is a border point -- non-core on its own, but part of
    the settlement.

    FAULT INJECTION: removing the border-assignment block leaves this point at -1.
    """
    core = _blob(0.0, 0.0, 60, spread=3.0, seed=4)
    border = np.array([[float(np.max(core[:, 0])) + EPS_M * 0.9, 0.0]])
    lab = _dbscan(np.vstack([core, border]))
    assert lab[-1] == lab[0] >= 0, "a point within EPS_M of a core must join its cluster"


def test_core_needs_min_samples_counting_itself() -> None:
    """scikit-learn counts the point itself toward min_samples; a ring of exactly MIN_SAMPLES-1
    neighbours is therefore core, and one fewer is not.

    FAULT INJECTION: `deg = np.zeros(n)` (not counting self) makes the first assertion fail.
    """
    tight = np.column_stack([np.arange(MIN_SAMPLES) * 1.0, np.zeros(MIN_SAMPLES)])
    assert (_dbscan(tight) >= 0).all()
    sparse = np.column_stack([np.arange(MIN_SAMPLES - 1) * 1.0, np.zeros(MIN_SAMPLES - 1)])
    assert (_dbscan(sparse) == -1).all()


def test_label_blocks_measures_area_share_not_mere_intersection() -> None:
    """A block that merely clips a settlement corner must NOT be labelled informal -- the whole
    point of a cover fraction over a boolean `intersects`.

    FAULT INJECTION: returning `cover = 1.0` on any hit labels the corner-clipping block too.
    """
    extents = gpd.GeoDataFrame({"n_structures": [100]}, geometry=[box(0, 0, 100, 100)], crs=UTM)
    blocks = gpd.GeoDataFrame(
        {"block_id": ["mostly_in", "corner"]},
        geometry=[box(10, 10, 90, 90), Polygon([(95, 95), (150, 95), (150, 150), (95, 150)])],
        crs=UTM)
    cover, label = label_blocks(blocks, extents, cover_frac=0.30)
    assert cover[0] == pytest.approx(1.0)
    assert cover[1] < 0.02
    assert label.tolist() == [True, False]
