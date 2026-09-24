from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize as skimage_skeletonize

from reblock.methods.betweenness.thinning import skeletonize


def test_equals_scikit_image_on_random_blobs_and_lines() -> None:
    rng = np.random.default_rng(0)
    for trial in range(60):
        shape = (int(rng.integers(5, 120)), int(rng.integers(5, 120)))
        noise = rng.random(shape)
        mask = ndimage.gaussian_filter(noise, sigma=float(rng.uniform(0.5, 4))) > 0.5
        if trial % 3 == 0:
            mask |= rng.random(shape) > 0.97          # isolated pixels and specks
        assert np.array_equal(skeletonize(mask), skimage_skeletonize(mask)), trial


def test_edges_empty_and_full() -> None:
    for mask in (np.zeros((7, 9), bool), np.ones((7, 9), bool), np.ones((1, 5), bool),
                 np.eye(12, dtype=bool)):
        assert np.array_equal(skeletonize(mask), skimage_skeletonize(mask))
