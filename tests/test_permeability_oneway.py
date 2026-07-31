"""Guards for the two invariants the one-way extension must satisfy.

Both fault-injected: the named fault was applied, the test confirmed to fail, and the code restored.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon

from reblock.contracts import Block
from reblock.permeability import has_oneway, permeability

UTM = CRS.from_epsg(32734)


def _block(n: int = 6, step: float = 10.0) -> Block:
    polys, pts = [], []
    for i in range(n):
        for j in range(n):
            x0, y0 = i * step, j * step
            polys.append(Polygon([(x0, y0), (x0 + step, y0), (x0 + step, y0 + step),
                                  (x0, y0 + step)]))
            pts.append(Point(x0 + step / 2, y0 + step / 2))
    return Block(
        block_id="ow", crs=UTM,
        boundary=Polygon([(0, 0), (n * step, 0), (n * step, n * step), (0, n * step)]),
        parcels=gpd.GeoDataFrame({"parcel_id": [str(k) for k in range(len(polys))]},
                                 geometry=polys, crs=UTM),
        streets=gpd.GeoDataFrame(
            geometry=[LineString([(-step, 0.0), (n * step + step, 0.0)])], crs=UTM),
        building_points=gpd.GeoDataFrame(geometry=pts, crs=UTM))


def _roads() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[
        LineString([(20.0, 0.0), (20.0, 50.0)]),
        LineString([(20.0, 30.0), (50.0, 30.0)]),
    ], crs=UTM)


def test_two_way_reduces_exactly_to_todays_metric() -> None:
    """An all-two-way road set must score BIT-IDENTICALLY to the metric without direction at all.

    This is what makes the extension safe to ship: every existing number is preserved unless a
    method opts in. "Close" is not good enough -- the published comparison rests on these values.

    NOT fault-injectable by the boost formula: a two-way road takes the `dvec is None` branch and
    never evaluates it. This guards the ship-safety property (existing numbers are preserved unless
    a method opts in), and `test_crossing_edge_keeps_full_road_conductance` guards the formula.
    """
    block, roads = _block(), _roads()
    plain = float(permeability(block, roads))
    two_way = roads.copy()
    two_way["oneway"] = False
    assert not has_oneway(two_way)
    assert float(permeability(block, two_way)) == plain


def test_one_way_never_scores_better_than_two_way() -> None:
    """Restricting a road to one direction cannot IMPROVE flow.

    A physical impossibility, so it is the sharpest available check on the directed solve.

    Weak on a small synthetic fixture -- the observed ground-shunt leak (0.9964 vs 0.9894 on a real
    block) did NOT flip this inequality here. `test_directed_solver_matches_a_hand_computed_case`
    is the sharp guard on the shunt; this one is the cheap sanity bound.
    """
    block, roads = _block(), _roads()
    two_way = roads.copy()
    two_way["oneway"] = False
    one_way = roads.copy()
    one_way["oneway"] = True
    assert has_oneway(one_way)
    p_two = float(permeability(block, two_way))
    p_one = float(permeability(block, one_way))
    assert np.isfinite(p_one)
    assert p_one <= p_two + 1e-12, f"one-way {p_one:.8f} beat two-way {p_two:.8f}"


def test_crossing_edge_keeps_full_road_conductance() -> None:
    """A parcel pair facing each other ACROSS a one-way road must score as if it were two-way.

    This is the property that distinguishes gating (`min(1, 1 + cos t)`) from scaling
    (`(1 + cos t)/2`): crossing a one-way street does not care which way its traffic runs. The
    fixture is two parcels either side of a road, so every covered edge is perpendicular to it and
    directionality must be invisible.

    FAULT INJECTION: `(1 + cos t)/2` halves the crossing edge's conductance, so the one-way score
    drops below the two-way score and this fails.
    """
    # Two constraints the fixture must satisfy or the assertion is vacuous:
    #   * the parcels must TOUCH -- `parcel_adjacency` uses STREET_TOL (0.5 m), so a gap leaves no
    #     mesh edge between them at all;
    #   * only ONE may reach the street -- if both are grounded each drains directly and the edge
    #     between them carries almost no current, so halving its conductance changes nothing.
    # Both were violated by earlier versions, and each made this test pass under the injected fault.
    road = LineString([(-5.0, 20.0), (25.0, 20.0)])
    parcels = gpd.GeoDataFrame(
        {"parcel_id": ["near", "far"]},
        geometry=[Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]),
                  Polygon([(0, 20), (20, 20), (20, 40), (0, 40)])], crs=UTM)
    block = Block(
        block_id="cross", crs=UTM,
        boundary=Polygon([(0, 0), (20, 0), (20, 40), (0, 40)]), parcels=parcels,
        streets=gpd.GeoDataFrame(geometry=[LineString([(0.0, 0.0), (20.0, 0.0)])], crs=UTM),
        building_points=gpd.GeoDataFrame(geometry=[Point(10, 10), Point(10, 30)], crs=UTM))
    two = gpd.GeoDataFrame(geometry=[road], crs=UTM)
    one = gpd.GeoDataFrame(geometry=[road], crs=UTM)
    two["oneway"], one["oneway"] = False, True
    p_two, p_one = float(permeability(block, two)), float(permeability(block, one))
    assert p_one == p_two, f"crossing a one-way road cost something: {p_one} vs {p_two}"


def test_directed_solver_matches_a_hand_computed_case() -> None:
    """`_directed_power` on a two-node, one-edge network, against the closed-form answer.

    Both nodes inject 1 unit; node 0 is grounded with shunt `s`. All 2 units leave through node 0,
    so the edge carries exactly 1 unit from node 1 to node 0 and the ground carries 2. With the edge
    oriented 0->1, that traversal is BACKWARD and pays `gb`:

        P = 1^2/gb + s*v0^2,   v0 = 2/s   =>   P = 1/gb + 4/s

    FAULT INJECTION: deriving the ground shunt by subtracting edge degrees from a diagonal built
    with max(gf, gb) -- while the iteration re-derives them from the current g = sqrt(gf*gb) --
    leaks (max - sqrt) into the shunt and breaks this equality badly.
    """
    from reblock.permeability import _directed_power

    gf, gb, s = np.array([1.0]), np.array([0.1]), 20.0
    rows, cols = np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)
    ground = np.array([s, 0.0])
    p, v = _directed_power(2, rows, cols, gf, gb, ground, np.ones(2))
    expected = 1.0 / gb[0] + 4.0 / s
    assert np.isfinite(p)
    assert abs(p - expected) < 1e-9 * expected, f"got {p:.9f}, closed form {expected:.9f}"
