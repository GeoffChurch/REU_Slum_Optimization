"""Synthetic checks of lifted.py.

    pixi run python research/roadless/checks.py angle      # open channel: P vs angle
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lifted  # noqa: E402


def channel(alpha_deg: float, W: float, Lc: float, h: float, p: lifted.Params) -> float:
    a = np.radians(alpha_deg)
    e = np.array([np.cos(a), np.sin(a)])
    n = np.array([-e[1], e[0]])
    o = np.array([0.0, 0.0])
    corners = [o - n * W / 2, o + e * Lc - n * W / 2, o + e * Lc + n * W / 2, o + n * W / 2]
    poly = Polygon(corners)
    street = LineString([corners[1], corners[2]])          # the far end is the street
    grid = lifted.Grid.of(poly, [], [street], h, band_m=1.0)
    src = LineString([corners[0], corners[3]]).buffer(1.0)
    f = (grid.mask_of(src) & grid.inside & ~grid.ground).astype(float)
    f /= f.sum()
    free = grid.ff0
    return lifted.solve(grid, free, f, p).P


def angle() -> None:
    for K in (8, 16):
        for ell in (1.0, 3.0, 10.0):
            for W in (2.0, 4.0):
                ps = [channel(a, W, 40.0, 0.5, lifted.Params(ell_m=ell, K=K))
                      for a in range(0, 91, 5)]
                ps = np.array(ps)
                print(f"K={K:2d} ell={ell:4.1f} W={W:3.1f}  P min {ps.min():8.3f} max "
                      f"{ps.max():8.3f}  max/min {ps.max() / ps.min():6.3f}  "
                      f"by angle {np.round(ps / ps[0], 2).tolist()}", flush=True)


if __name__ == "__main__":
    {"angle": angle}[sys.argv[1]]()
