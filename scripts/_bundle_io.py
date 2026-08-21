"""The quantisers and geometry exploders every committed browser bundle is written through.

Declared once because all three bakers need them: `scripts/gen_web_bundle.py` (PermGraph),
`scripts/gen_frontier_bundle.py` (Frontier) and `scripts/gen_displacement_field.py`
(DisplacementField). They started private to the first, and by the time the third arrived there were
already two live copies of `sigfig` differing only in name -- one of them carrying a docstring that
pointed at a function in the other file. Copying is how the coordinate-precision trap in `cm` gets
reintroduced in a file where nobody is looking for it, and how `polygon_ring`'s two raises quietly
become one.
"""
from __future__ import annotations

from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.geometry.base import BaseGeometry

SIGFIGS = 6


def sigfig(x: float) -> float:
    """FIELD VALUES at 6 significant digits -- far beyond what a canvas shows or the readout
    quotes, and it keeps the payload small. Both bundles' parity assertions are stated at this
    precision (tests/test_web_bundle.py compares at it; tests/test_displacement_field_bundle.py
    recomputes THROUGH this function so it needs no tolerance at all), so changing it means
    revisiting both.

    NOT for coordinates -- see `cm`."""
    return float(f"%.{SIGFIGS}g" % x)


def cm(x: float) -> float:
    """COORDINATES, as centimetres of absolute precision.

    Significant digits are the wrong tool here and dangerously so: a Cape Town UTM northing is
    ~6,240,000, so `%.6g` would round it to the nearest 10 METRES and dissolve the parcel geometry.
    Coordinates are emitted relative to each bundle's `origin`, which both fixes the precision
    problem and shrinks the payload, since local metres are 3-4 digits instead of 7."""
    return round(x, 2)


def line_coords(geom: BaseGeometry, ox: float, oy: float) -> list[list[list[float]]]:
    """Explode a LineString/MultiLineString into one coordinate list per component, at the same
    centimetre precision `cm` gives every other coordinate in a bundle (see its docstring): a
    street's northing is exactly as far from the origin as a parcel's, and significant-digit
    rounding would dissolve it the same way. `Block.streets` is documented as line geometry
    (`_draw_boundary_and_streets` in render.py draws it with no other case); anything else is a
    contract violation worth raising on, not silently dropping."""
    if isinstance(geom, LineString):
        lines: list[LineString] = [geom]
    elif isinstance(geom, MultiLineString):
        lines = list(geom.geoms)
    else:
        raise ValueError(
            f"unexpected street geometry type {geom.geom_type!r} -- report this instead of "
            f"silently dropping it")
    return [[[cm(x - ox), cm(y - oy)] for x, y in line.coords] for line in lines]


def polygon_ring(geom: BaseGeometry, ox: float, oy: float, *, what: str) -> list[list[float]]:
    """A simple Polygon's exterior ring, origin-relative at `cm` precision.

    isinstance, not geom_type, so this line IS the runtime guard mypy can also verify: it narrows
    `geom` to Polygon, which is what makes `.interiors`/`.exterior` below type-check instead of
    resolving through BaseGeometry, the union GeoSeries iteration yields.

    Raises rather than dropping: every bundle format here gives a polygon ONE ring, so a
    MultiPolygon or a polygon with holes would have to lose geometry to fit -- and geometry that
    silently vanishes from a committed artifact is a wrong picture nobody is looking for.
    `what` names the offender, since a bundle has many polygons and only one of them is wrong.

    DELIBERATE ADDITION when this moved out of `gen_web_bundle.py`: that file checked `interiors`
    on parcels but not on the BOUNDARY, whose inline guard only rejected a MultiPolygon. A
    boundary with a hole would have had its hole silently dropped. Nothing observable changed --
    the pinned block's boundary has no interior rings, and `examples/perm-graph/bundle.json`
    regenerates byte-identical -- so this widens an error path only.
    """
    if not isinstance(geom, Polygon):
        raise ValueError(
            f"{what} is a {geom.geom_type}, not a Polygon -- the bundle format assumes a simple "
            f"Polygon with an exterior ring and no holes; report this instead of silently "
            f"dropping geometry")
    if len(geom.interiors) != 0:
        raise ValueError(
            f"{what} is a non-simple Polygon ({len(geom.interiors)} interior rings) -- the bundle "
            f"format assumes an exterior ring and no holes; report this instead of silently "
            f"dropping geometry")
    return [[cm(x - ox), cm(y - oy)] for x, y in geom.exterior.coords]


def polygon_rings(geom: BaseGeometry, ox: float, oy: float, *,
                  what: str) -> list[list[list[float]]]:
    """Every ring of a simple Polygon -- exterior first, then interiors -- origin-relative at `cm`.

    The multi-ring counterpart to `polygon_ring`, and a SEPARATE function rather than a flag on it.
    `polygon_ring` raises on interior rings because the three bundles that use it give a polygon
    exactly one ring, so a hole there would have to lose geometry to fit; that guard is still right
    and a parameter letting a caller switch it off would be a way to lose geometry quietly. The
    city tier and the region neighbourhood have a different contract: they CARRY holes, and their
    consumers fill with the even-odd rule.

    Measured on the data these bundles are baked from: 6,990 of 16,451 Cape Town blocks and 1,139
    of 3,500 Nairobi blocks have interior rings, as do 7 of the 213 blocks in RegionGrow's
    neighbourhood. Neither city has a single MultiPolygon block, which is why that case raises
    rather than being flattened.
    """
    if not isinstance(geom, Polygon):
        raise ValueError(
            f"{what} is a {geom.geom_type}, not a Polygon -- the bundle format gives a block one "
            f"polygon (measured: no MultiPolygon blocks in either city); report this instead of "
            f"silently dropping geometry")
    return [[[cm(x - ox), cm(y - oy)] for x, y in ring.coords]
            for ring in [geom.exterior, *geom.interiors]]
