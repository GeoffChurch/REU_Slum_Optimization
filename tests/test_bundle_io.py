"""`_bundle_io`'s encoders. The multi-ring one exists because 6,990 Cape Town blocks, 1,139
Nairobi blocks and 3 of RegionGrow's 129 neighbourhood blocks have interior rings -- and
`polygon_ring`, which the three older bundles use, raises on every one of them."""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from scripts._bundle_io import polygon_ring, polygon_rings


def _donut() -> Polygon:
    return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)],
                   [[(3, 3), (3, 6), (6, 6), (6, 3)]])


def test_polygon_rings_keeps_the_hole() -> None:
    rings = polygon_rings(_donut(), 0.0, 0.0, what="test block")
    assert len(rings) == 2, "exterior plus one interior"
    assert rings[0][0] == [0.0, 0.0], "exterior comes first"
    assert [3.0, 3.0] in rings[1], "the interior ring's coordinates survive"


def test_polygon_rings_is_origin_relative_at_cm_precision() -> None:
    rings = polygon_rings(_donut(), 1.0, 2.0, what="test block")
    assert rings[0][0] == [-1.0, -2.0]
    assert all(round(v, 2) == v for ring in rings for pt in ring for v in pt)


def test_polygon_rings_rejects_a_multipolygon() -> None:
    """Neither city has one, and the format gives a block one polygon -- so this raises rather
    than dropping a part. `what` names the offender, since a bundle has many blocks."""
    from shapely.geometry import MultiPolygon
    mp = MultiPolygon([_donut(), Polygon([(20, 20), (21, 20), (21, 21)])])
    with pytest.raises(ValueError, match="block 7"):
        polygon_rings(mp, 0.0, 0.0, what="block 7")


def test_polygon_ring_still_rejects_holes() -> None:
    """The strict encoder keeps its guard. The new function is a second contract, not an escape
    hatch from this one -- three shipped bundles depend on it raising."""
    with pytest.raises(ValueError, match="interior rings"):
        polygon_ring(_donut(), 0.0, 0.0, what="parcel 3")
