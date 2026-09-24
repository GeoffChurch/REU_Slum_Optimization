"""Zhang-Suen thinning, ported from scikit-image 0.26.0's `_fast_skeletonize`
(src/skimage/morphology/_skeletonize_various_cy.pyx), so `src` needs no scikit-image at runtime.
Pinned EQUAL to `skimage.morphology.skeletonize` by tests/methods/betweenness/test_thinning.py.

Copyright (C) 2019, the scikit-image team. BSD-3-Clause; see
https://github.com/scikit-image/scikit-image/blob/v0.26.0/LICENSE.txt"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reblock._jit import njit

# One entry per 8-neighbourhood; 1 and 3 are removed in the first pass, 2 and 3 in the second.
_LUT = np.array([0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0,
                 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0,
                 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 2, 0, 0, 0, 3, 0, 2, 2, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
                 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0,
                 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0,
                 2, 0, 0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3,
                 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0,
                 2, 0, 0, 0], dtype=np.uint8)


@njit
def _thin(skeleton: NDArray[np.uint8], lut: NDArray[np.uint8]) -> NDArray[np.uint8]:
    nrows, ncols = skeleton.shape
    cleaned = skeleton.copy()
    removed = True
    while removed:
        removed = False
        for pass_num in range(2):
            first = pass_num == 0
            for row in range(1, nrows - 1):
                for col in range(1, ncols - 1):
                    if skeleton[row, col]:
                        nb = lut[skeleton[row - 1, col - 1] + 2 * skeleton[row - 1, col]
                                 + 4 * skeleton[row - 1, col + 1] + 8 * skeleton[row, col + 1]
                                 + 16 * skeleton[row + 1, col + 1] + 32 * skeleton[row + 1, col]
                                 + 64 * skeleton[row + 1, col - 1] + 128 * skeleton[row, col - 1]]
                        if nb == 0:
                            continue
                        if nb == 3 or (nb == 1 and first) or (nb == 2 and not first):
                            cleaned[row, col] = 0
                            removed = True
            skeleton[:, :] = cleaned[:, :]
    return skeleton


def skeletonize(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask
    out: NDArray[np.bool_] = _thin(padded, _LUT)[1:-1, 1:-1].astype(bool)
    return out
