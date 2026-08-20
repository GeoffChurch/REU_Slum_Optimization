"""The two default roads, and the closed-form corridor distance the widget implements.

Declared once because three callers need it: the bake (scripts/gen_displacement_field.py), the
identity test (tests/test_displacement_closed_form.py), and the fixture generator inside the bake.
`scripts/_example_block.py` set this precedent -- when each caller declared its own copy, changing
one left the others describing something else while every test still passed.
"""
from __future__ import annotations

import numpy as np
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

from reblock.contracts import Block


def default_roads(block: Block, width_m: float) -> GeoDataFrame:
    """Two straight roads, derived by rule so the PNG, the bundle and the caption agree.

    Road 1 runs along the building field's PRINCIPAL AXIS through its centroid, clipped to the
    block. Road 2 is road 1 shifted perpendicular by `3 * width_m` -- far enough that the two
    corridors start disjoint, so merging them is something the reader DOES rather than something
    they arrive to find already done.

    This is a REFERENCE LINE, not a discovered structural axis, and the docstring must not imply
    otherwise: measured on the pinned block the field is nearly isotropic (singular values 567.4 and
    523.0, anisotropy 1.085), so there is no meaningful "long axis" of this settlement to follow.

    Do not "improve" this to the convex-hull diameter or the longest interior chord. Both were
    measured and both are far WORSE conditioned: the hull diameter beats its runner-up pair by 0.07%
    (161.19 m against 161.07 m) and swings 3.28 degrees under 10 cm of coordinate jitter, where the
    principal axis swings 0.23 degrees -- because it averages 263 points while a diameter is decided
    by exactly two extreme vertices. The two alternatives also agree with each other to 0.0 degrees
    here, so they are one idea, not two.

    A rule rather than a hand-placed line: the widget's boot state and the committed PNG have to be
    the same road for fallback parity to mean anything, and the caption's numbers have to be
    measurements of it.
    """
    pts = block.building_points
    xy = np.column_stack([pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()])
    centre = xy.mean(axis=0)
    # First principal component. `np.linalg.svd` on the centred cloud; the sign of a singular
    # vector is arbitrary, so normalise it -- otherwise the "deterministic" road flips between runs
    # of the same code on the same data.
    _, _, vt = np.linalg.svd(xy - centre, full_matrices=False)
    axis = vt[0]
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis
    normal = np.array([-axis[1], axis[0]])

    hull = block.parcels.union_all()
    return GeoDataFrame(
        {"width_m": [float(width_m), float(width_m)]},
        geometry=[_chord(hull, centre, axis),
                  _chord(hull, centre + normal * (3.0 * float(width_m)), axis)],
        crs=block.crs)


def _chord(hull: BaseGeometry, through: NDArray[np.float64],
           direction: NDArray[np.float64]) -> LineString:
    """The longest piece of the infinite line `through + t*direction` that lies inside `hull`.

    Longest, not first: a concave block cuts the line into several pieces and only the longest is
    the road a reader would recognise as crossing the settlement.
    """
    span = float(np.hypot(*(np.asarray(hull.bounds[2:]) - np.asarray(hull.bounds[:2])))) * 2.0
    line = LineString([through - direction * span, through + direction * span])
    inside = line.intersection(hull)
    parts = list(inside.geoms) if isinstance(inside, BaseMultipartGeometry) else [inside]
    longest = max(parts, key=lambda g: g.length)
    return LineString([longest.coords[0], longest.coords[-1]])


def segments(roads: GeoDataFrame) -> NDArray[np.float64]:
    """Every road flattened to `(x0, y0, x1, y1, half_width)` -- exactly what the widget receives.

    Flattening here rather than in the bake means the identity test and the widget consume the same
    shape, so a parity failure is a failure of the FORMULA and never of two different flattenings.
    """
    out: list[tuple[float, float, float, float, float]] = []
    for geom, w in zip(roads.geometry, roads["width_m"].to_numpy(dtype=float), strict=True):
        parts = list(geom.geoms) if isinstance(geom, BaseMultipartGeometry) else [geom]
        for part in parts:
            coords = np.asarray(part.coords, dtype=np.float64)
            for a, b in zip(coords[:-1], coords[1:], strict=True):
                out.append((float(a[0]), float(a[1]), float(b[0]), float(b[1]), float(w) / 2.0))
    return np.asarray(out, dtype=np.float64).reshape(-1, 5)


def closed_form_distance(px: NDArray[np.float64], py: NDArray[np.float64],
                         segs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-point distance to the corridor, without ever building the corridor.

        dist(p, U_i buffer(L_i, w_i/2)) == min_i max(0, dist(p, L_i) - w_i/2)

    This is the reference `web/src/model/displacement.ts` mirrors line for line. Kept in numpy here
    and per-point there; same arithmetic.
    """
    if len(segs) == 0:
        return np.full(len(px), np.inf)
    x0, y0, x1, y1, hw = segs.T
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    t = ((px[:, None] - x0) * dx + (py[:, None] - y0) * dy) / np.where(L2 > 0, L2, 1.0)
    t = np.clip(np.where(L2 > 0, t, 0.0), 0.0, 1.0)      # a zero-length road is its own endpoint
    d = np.hypot(px[:, None] - (x0 + t * dx), py[:, None] - (y0 + t * dy)) - hw
    return np.maximum(0.0, d).min(axis=1)
