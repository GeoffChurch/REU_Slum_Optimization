"""The two default roads the displacement widget boots with, and the chord rule behind them.

Declared once because more than one caller needs them: the bake (`scripts/gen_displacement_field`)
and the bundle test that re-derives what it baked. `scripts/_example_block.py` set this precedent
-- when each caller declared its own copy, changing one left the others describing something else
while every test still passed.
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
    xy = block.buildings.xy
    centre = xy.mean(axis=0)
    # First principal component. `np.linalg.svd` on the centred cloud; the SIGN of a singular
    # vector is arbitrary in the linear algebra, not in this code's execution -- SVD is
    # deterministic for one fixed input on one machine, but the sign can still differ across LAPACK
    # builds, platforms, or a reordering of the input points, so it is normalised for stability
    # across those, not against nondeterminism that doesn't exist here.
    #
    # If the normalisation were ever dropped, only ROAD 2 would move: road 1's chord is the same
    # line either way (`chord`'s +-direction extension is direction-sign-symmetric), but `normal`
    # flips, putting road 2's offset on the other side of `centre`. Nothing in THIS module is meant
    # to catch that flip: a reviewer confirmed containment and disjointness survive it even on an
    # asymmetric block (only a pinned coordinate would notice), and on the symmetric synthetic
    # fixture the tests here use, the principal axis is genuinely degenerate (equal singular
    # values) -- a pinned coordinate there would pin arbitrary LAPACK output, not a real invariant.
    # Task 3's committed artifact plus its staleness test is the intended guard against a flip on
    # the real, non-degenerate block.
    _, _, vt = np.linalg.svd(xy - centre, full_matrices=False)
    axis = vt[0]
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis
    normal = np.array([-axis[1], axis[0]])

    hull = block.parcels.union_all()
    return GeoDataFrame(
        {"width_m": [float(width_m), float(width_m)]},
        geometry=[chord(hull, centre, axis),
                  chord(hull, centre + normal * (3.0 * float(width_m)), axis)],
        crs=block.crs)


def chord(hull: BaseGeometry, through: NDArray[np.float64],
          direction: NDArray[np.float64]) -> LineString:
    """The longest piece of the infinite line `through + t*direction` that lies inside `hull`.

    Public, not private: `scripts/gen_displacement_field.py`'s `in_a_gap` fixture is a chord too,
    and the rule below -- longest piece, raise if the line misses the interior -- is exactly the
    thing a second copy would get subtly wrong.

    Longest, not first: a concave block cuts the line into several pieces and only the longest is
    the road a reader would recognise as crossing the settlement.

    Raises `ValueError` if no piece has positive length -- either the line misses the hull's
    interior entirely, or it only grazes the boundary tangentially. It is a real input check and
    not a defensive one, because `through` is a COMPUTED point that both callers derive rather than
    choose, and neither derivation is constrained to land inside the hull:

    * `default_roads` offsets road 2 by `3 * width_m` perpendicular to the centroid, so a block
      whose short dimension is under three road widths puts it clear of the parcels entirely;
    * `gen_displacement_field`'s `in_a_gap` fixture passes the MIDPOINT of the widest
      nearest-neighbour pair, which on a concave block can sit in a notch outside the parcel union
      even though both of its endpoints are inside.

    Neither fires on the pinned block at the width the bake runs today, and the alternative to
    raising is not "no branch" but returning whatever `max(..., default=None)` found -- an empty
    geometry that would travel silently into `road_specs` and bake a road with no coordinates.

    NOT reachable through the widget's 20 m width slider, which an earlier draft of this note
    claimed: the slider is a browser control that re-widths roads already baked into `field.json`,
    and nothing in the browser ever calls this. Only the bake does.
    """
    span = float(np.hypot(*(np.asarray(hull.bounds[2:]) - np.asarray(hull.bounds[:2])))) * 2.0
    line = LineString([through - direction * span, through + direction * span])
    inside = line.intersection(hull)
    parts = list(inside.geoms) if isinstance(inside, BaseMultipartGeometry) else [inside]
    longest = max(parts, key=lambda g: g.length, default=None)
    if longest is None or longest.length <= 0.0:
        minx, miny, maxx, maxy = hull.bounds
        raise ValueError(
            f"no chord through {tuple(through)} heading {tuple(direction)} crosses the block's "
            f"interior (block extent x=[{minx:.1f}, {maxx:.1f}], y=[{miny:.1f}, {maxy:.1f}]); the "
            "offset pushed the line clear of the hull, or it only grazed the hull's boundary")
    return LineString([longest.coords[0], longest.coords[-1]])
