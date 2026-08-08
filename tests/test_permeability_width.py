"""Guards on the affine WIDTH model: what a road's width buys, and what is too narrow to build.

Every one of these is fault-injected: the named fault was applied, the test confirmed to fail, and
the code restored.

These outlived the one-way extension they were written alongside. Direction was deleted on
2026-08-07 (nothing selected `width_solver`, its only producer, and one-way was measured dominated
at p = 5e-07), but the width law it shared is load-bearing and stayed.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon

from reblock.contracts import Block
from reblock.permeability import (
    DEFAULT_ROAD_WIDTH_M,
    PermeabilityParams,
    permeability,
    road_conductance,
    with_width,
)

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
        block_id="w", crs=UTM,
        boundary=Polygon([(0, 0), (n * step, 0), (n * step, n * step), (0, n * step)]),
        parcels=gpd.GeoDataFrame({"parcel_id": [str(k) for k in range(len(polys))]},
                                 geometry=polys, crs=UTM),
        streets=gpd.GeoDataFrame(
            geometry=[LineString([(-step, 0.0), (n * step + step, 0.0)])], crs=UTM),
        building_points=gpd.GeoDataFrame(geometry=pts, crs=UTM))


def _roads() -> gpd.GeoDataFrame:
    return with_width(gpd.GeoDataFrame(geometry=[
        LineString([(20.0, 0.0), (20.0, 50.0)]),
        LineString([(20.0, 30.0), (50.0, 30.0)]),
    ], crs=UTM), DEFAULT_ROAD_WIDTH_M)


def test_a_default_road_gives_one_direction_exactly_the_calibrated_lane_conductance() -> None:
    """The affine width law must give a default road's each direction the CALIBRATED 20.0.

    20.0 is the calibrated quantity -- one lane, tuned against `g_walk` for method discrimination --
    and it must survive a re-base of the lane width. When the floor moved from 6.0 to 7.0,
    `g_road_per_m` fell from 8.0 to 20/3 precisely so this assertion still holds: believing a lane
    takes more SPACE is not a claim that it carries more traffic. Asserting the calibrated value
    rather than `g_road_per_m * <lane>` is what makes this test survive the next re-base too.

    FAULT INJECTION: dropping the margin from `road_conductance` (pure `k*W/2d`) gives 23.33.
    """
    pr = PermeabilityParams()
    assert road_conductance(pr, DEFAULT_ROAD_WIDTH_M, 1.0) == pytest.approx(20.0)
    # ...and the default road IS the floor, so the cheapest legal road is exactly one lane each way
    assert DEFAULT_ROAD_WIDTH_M == pytest.approx(pr.min_road_width_m)


def test_widening_is_superlinear_because_the_margin_is_paid_once() -> None:
    """Doubling a corridor's width must MORE than double its conductance.

    A behavioural consequence, not just realism: it rewards fewer wide roads over many narrow ones,
    which is what real street hierarchies look like. Worth pinning so the incentive cannot be
    removed by accident.

    FAULT INJECTION: dropping the margin makes this exactly 2.0 and the strict inequality fails.
    """
    pr = PermeabilityParams()
    w = DEFAULT_ROAD_WIDTH_M
    ratio = road_conductance(pr, 2 * w, 1.0) / road_conductance(pr, w, 1.0)
    assert ratio > 2.0, f"widening should be superlinear, got {ratio:.4f}"


def test_a_road_below_the_floor_is_refused() -> None:
    """The guard that was missing: a 4 m road has one lane of usable width, and the affine model
    would otherwise credit it as two 1.5 m lanes running side by side. An unbuildable road of
    exactly this kind decided a comparison it had no business deciding.

    FAULT INJECTION: removing the `bad.any()` raise scores the road instead of refusing it.
    """
    pr = PermeabilityParams()
    block, roads = _block(), _roads()
    narrow = with_width(roads, 4.0)
    with pytest.raises(ValueError, match="below the 7 m floor"):
        permeability(block, narrow, pr)


def test_above_the_floor_width_still_buys_capacity_continuously() -> None:
    """This pins the DESIGN DECISION: a floor, not a quantization. Extra width above the floor is
    real -- in a dense settlement one parked vehicle otherwise blocks the way outright -- so
    conductance must keep rising between whole-lane multiples, not sit flat until the next one.

    FAULT INJECTION: quantizing `road_conductance`'s usable width to whole lanes flattens the
    diff to zero between 7.0 and 7.4, and between 8.2 and 9.5.
    """
    pr = PermeabilityParams()
    widths = np.array([7.0, 7.4, 8.2, 9.5])
    g = road_conductance(pr, widths, np.ones(len(widths)))
    assert (np.diff(g) > 0).all(), f"width must buy capacity continuously above the floor, got {g}"


def test_a_wider_road_scores_at_least_as_well_end_to_end() -> None:
    """Monotonicity in width, through the whole metric rather than just `road_conductance`.

    `edge_conductances` takes `max(footpath, road)` and a wider road both raises that road term AND
    can only ever cover MORE edges (its buffer grows), so permeability cannot fall. This is the
    property the module docstring calls load-bearing.

    FAULT INJECTION: replacing the `max(g[hit], ...)` with a plain assignment makes a wide road
    OVERWRITE a footpath edge that was already better, and this fails.
    """
    pr = PermeabilityParams()
    block, roads = _block(), _roads()
    scores = [float(permeability(block, with_width(roads, w), pr)) for w in (7.0, 9.0, 14.0)]
    assert all(np.isfinite(scores))
    assert scores[0] <= scores[1] + 1e-12 <= scores[2] + 1e-12, scores
