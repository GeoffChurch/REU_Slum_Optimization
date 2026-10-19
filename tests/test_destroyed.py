"""The ramped destroyed-area measure (reblock.destroyed): a corner clip destroys the clipped share,
a road through a building destroys the corridor plus every piece but the largest, the ramp
min(1, X / tau), the weighted D, and the incremental evaluation equal to the whole-corridor one.
The faults (the largest piece read as the whole remainder; a meter that forgets earlier pieces)
are shown to break the checks."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import shapely
from pyproj import CRS
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

import reblock.destroyed as destroyed_mod
from reblock.buildings import ANCHOR_COL, Footprints
from reblock.destroyed import IncrementalDestroyed, RampedDestroyed

UTM = CRS.from_epsg(32643)
TAU = RampedDestroyed(tau=0.10)


def _footprints(polys: list[shapely.Polygon]) -> Footprints:
    return Footprints(gpd.GeoDataFrame({ANCHOR_COL: [p.centroid for p in polys]},
                                       geometry=polys, crs=UTM))


def test_a_corner_clip_destroys_the_clipped_share() -> None:
    b = _footprints([box(0, 0, 10, 10)])
    corridor = box(9, -1, 11, 11)                 # the east metre: 10% of the building
    assert TAU.destroyed(b, corridor)[0] == pytest.approx(0.10, abs=1e-12)
    assert RampedDestroyed(tau=0.20).c(b, corridor)[0] == pytest.approx(0.5, abs=1e-12)
    assert TAU.c(b, box(9.5, -1, 11, 11))[0] == pytest.approx(0.5, abs=1e-12)


def test_a_road_through_destroys_every_piece_but_the_largest() -> None:
    # A 1 m corridor at x 3..4 cuts a 10 x 10 building into 3 m and 6 m pieces: the 6 m piece
    # stands, so 4 of 10 is destroyed (the overlap fraction would say 1 of 10).
    b = _footprints([box(0, 0, 10, 10)])
    corridor = box(3, -1, 4, 11)
    assert TAU.destroyed(b, corridor)[0] == pytest.approx(0.40, abs=1e-12)
    assert b.displacement(corridor)[0] == pytest.approx(0.10, abs=1e-12)


def test_the_bisection_guard_sees_the_whole_remainder(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fault: the remainder's total area in place of its largest piece -- the overlap fraction.
    monkeypatch.setattr(destroyed_mod, "_largest_piece_area", lambda rest: shapely.area(rest))
    b = _footprints([box(0, 0, 10, 10)])
    assert TAU.destroyed(b, box(3, -1, 4, 11))[0] != pytest.approx(0.40, abs=1e-12)


def test_untouched_and_whole_buildings() -> None:
    b = _footprints([box(0, 0, 10, 10), box(20, 0, 22, 2), box(40, 0, 50, 10)])
    corridor = box(19, -1, 23, 3).union(box(10, 0, 12, 10))   # the second whole; the first edge
    x = TAU.destroyed(b, corridor)
    assert x[0] == pytest.approx(0.0, abs=1e-12) and x[1] == 1.0 and x[2] == 0.0


def test_share_weights_by_population() -> None:
    b = _footprints([box(0, 0, 10, 10), box(20, 0, 30, 10)])
    corridor = box(9.5, -1, 11, 11)               # 5% of the first: c = 0.5
    assert TAU.share(b, corridor, np.array([1.0, 1.0])) == pytest.approx(0.25, abs=1e-12)
    assert TAU.share(b, corridor, np.array([3.0, 1.0])) == pytest.approx(0.375, abs=1e-12)
    assert TAU.share(b, None, np.array([1.0, 1.0])) == 0.0


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, float("nan"), float("inf")])
def test_tau_out_of_range_is_refused(bad: float) -> None:
    with pytest.raises(ValueError):
        RampedDestroyed(tau=bad)


def test_tau_is_keyword_only_and_in_the_identity() -> None:
    with pytest.raises(TypeError):
        RampedDestroyed(0.1)  # type: ignore[call-arg]
    assert RampedDestroyed(tau=0.1).identity != RampedDestroyed(tau=0.2).identity


def _grid() -> Footprints:
    return _footprints([box(i * 4.0, j * 4.0, i * 4.0 + 3.0, j * 4.0 + 3.0)
                        for i in range(6) for j in range(6)])


ROADS = [LineString([(1.5, -2), (1.5, 26)]), LineString([(-2, 9.2), (26, 9.2)]),
         LineString([(-2, -2), (26, 26)]), LineString([(1.0, -2), (1.0, 26)])]


def _incremental(b: Footprints, roads: list[LineString], hw: float) -> IncrementalDestroyed:
    inc = IncrementalDestroyed(TAU, b)
    for r in roads:
        inc.add(r.buffer(hw))
    return inc


def test_incremental_equals_the_whole_corridor() -> None:
    b = _grid()
    for k in range(1, len(ROADS) + 1):
        whole = unary_union([r.buffer(0.6) for r in ROADS[:k]])
        np.testing.assert_allclose(_incremental(b, ROADS[:k], 0.6).destroyed(),
                                   TAU.destroyed(b, whole), atol=1e-9)
    assert (TAU.destroyed(b, whole) > 0).sum() > 10


def test_incremental_guard_sees_a_forgetful_meter(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fault: each piece replaces a building's covered region instead of joining it.
    monkeypatch.setattr(shapely, "union", lambda a, b: b)
    b = _grid()
    whole = unary_union([r.buffer(0.6) for r in ROADS])
    assert not np.allclose(_incremental(b, ROADS, 0.6).destroyed(), TAU.destroyed(b, whole),
                           atol=1e-9)


def test_a_building_with_no_area_is_destroyed_when_covered() -> None:
    from reblock.buildings import Discs
    pts = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(10, 0)], crs=UTM)
    b = Discs(pts, np.array([0.0, 0.0]))
    corridor = Point(0, 0).buffer(1.0)
    np.testing.assert_array_equal(TAU.destroyed(b, corridor), [1.0, 0.0])
    inc = IncrementalDestroyed(TAU, b)
    inc.add(corridor)
    np.testing.assert_array_equal(inc.destroyed(), [1.0, 0.0])
