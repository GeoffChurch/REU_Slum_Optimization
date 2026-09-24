"""The block as a raster: which cells are inside, how far each is from the nearest building, and
how far from the street. Every betweenness count is computed on this grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from shapely import STRtree

from reblock.contracts import Block


@dataclass(frozen=True, eq=False)
class BlockRaster:
    x0: float                                 # centre of cell (0, 0)
    y0: float
    res: float
    inside: NDArray[np.bool_]                 # (ny, nx): cell centre inside the block
    clearance: NDArray[np.float64]            # metres to the nearest building outline; 0 inside
    edge: NDArray[np.float64]                 # metres to the block boundary (the street)

    @property
    def shape(self) -> tuple[int, int]:
        ny, nx = self.inside.shape
        return int(ny), int(nx)

    @property
    def free(self) -> NDArray[np.bool_]:
        return self.inside & (np.nan_to_num(self.clearance) > 0)

    @classmethod
    def of(cls, block: Block, res: float) -> BlockRaster:
        minx, miny, maxx, maxy = block.boundary.bounds
        xs = np.arange(minx, maxx + res, res)
        ys = np.arange(miny, maxy + res, res)
        X, Y = np.meshgrid(xs, ys)
        inside = shapely.contains_xy(block.boundary, X.ravel(), Y.ravel()).reshape(X.shape)
        pts = shapely.points(X.ravel()[inside.ravel()], Y.ravel()[inside.ravel()])
        clearance = np.full(X.shape, np.nan)
        outlines = np.asarray(block.buildings.outlines.geometry)
        if len(outlines) and len(pts):
            # query_nearest returns (INPUT index, TREE index) pairs; only the distance is kept,
            # scattered by INPUT index (reading them the other way round wrote cell ids into
            # polygon slots once, and every width read 0.00).
            idx, dist = STRtree(outlines).query_nearest(pts, return_distance=True,
                                                         all_matches=False)
            d = np.full(len(pts), np.nan)
            d[idx[0]] = dist
            clearance[inside] = d
        elif len(pts):
            clearance[inside] = np.inf            # no buildings: every inside cell is free
        edge = np.full(X.shape, np.nan)
        edge[inside] = shapely.distance(pts, block.boundary.boundary)
        return cls(x0=float(xs[0]), y0=float(ys[0]), res=res, inside=inside,
                   clearance=clearance, edge=edge)


def home_cells(raster: BlockRaster, block: Block) -> NDArray[np.int64]:
    """(row, col) of the free cell nearest each building's anchor, one row per building."""
    free_rc = np.argwhere(raster.free)
    if len(block.buildings) == 0 or len(free_rc) == 0:
        return np.empty((0, 2), dtype=np.int64)
    tree = cKDTree(np.c_[raster.x0 + raster.res * free_rc[:, 1],
                         raster.y0 + raster.res * free_rc[:, 0]])
    _d, k = tree.query(block.buildings.xy)
    return free_rc[np.asarray(k, dtype=np.int64)].astype(np.int64)
