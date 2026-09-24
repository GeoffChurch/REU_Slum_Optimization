"""A field's ridges as desire lines: the skeleton of each top-quantile level set, one group per
level, weights summing to 1 -- so a point's demand is the share of levels whose ridge runs within
the reblocker's corridor of it."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
from pyproj import CRS
from shapely.geometry import LineString

from reblock.methods.betweenness.raster import BlockRaster
from reblock.methods.betweenness.thinning import skeletonize
from reblock.methods.desire_lines import DesireField, WeightedLines


def _segments(skel: NDArray[np.bool_], raster: BlockRaster) -> list[LineString]:
    r, c = np.nonzero(skel)
    on = set(zip(r.tolist(), c.tolist(), strict=True))
    x0, y0, res = raster.x0, raster.y0, raster.res
    out = []
    for i, j in on:                               # set iteration: the measured order, kept
        for di, dj in ((0, 1), (1, -1), (1, 0), (1, 1)):
            if (i + di, j + dj) in on:
                out.append(LineString([(x0 + res * j, y0 + res * i),
                                       (x0 + res * (j + dj), y0 + res * (i + di))]))
    return out


def ridge_desire(raster: BlockRaster, field: NDArray[np.float64],
                 quantiles: tuple[float, ...], crs: CRS) -> DesireField:
    free = raster.free & np.isfinite(field)
    vals = field[free]
    # No free cell, or a field with no contrast at all (no homes, or no street to reach: every
    # count is 0): there is no ridge to follow, so there is no desire. Thresholding a constant
    # field would instead return the whole free area as one "ridge".
    if vals.size == 0 or float(vals.max()) == float(vals.min()):
        return DesireField(groups=())
    groups = []
    for q in quantiles:
        skel = skeletonize(free & (field >= np.quantile(vals, q)))
        groups.append(WeightedLines(lines=gpd.GeoDataFrame(geometry=_segments(skel, raster),
                                                           crs=crs),
                                    weight=1.0 / len(quantiles)))
    return DesireField(groups=tuple(groups))
