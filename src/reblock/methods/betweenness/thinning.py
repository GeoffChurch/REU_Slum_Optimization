"""Zhang-Suen thinning, ported from scikit-image 0.26.0's `_fast_skeletonize`
(src/skimage/morphology/_skeletonize_various_cy.pyx), so `src` needs no scikit-image at runtime.
Pinned EQUAL to `skimage.morphology.skeletonize` by tests/methods/betweenness/test_thinning.py.
The port is under scikit-image's license, reproduced in full below."""
# From https://github.com/scikit-image/scikit-image/blob/v0.26.0/LICENSE.txt, whose "Files: *"
# entry covers `_fast_skeletonize` (the file's one separate entry is `_skeletonize_loop`, not
# ported here):
#
# Copyright: 2009-2022 the scikit-image team
# License: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the University nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE HOLDERS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
