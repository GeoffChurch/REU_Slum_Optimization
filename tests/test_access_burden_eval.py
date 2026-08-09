"""Guards for AccessBurdenEval. Every one is fault-injected: the named fault was applied, the test
confirmed to fail, and the code restored."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon

from reblock.contracts import Block, Proposal
from reblock.eval.access_burden import AccessBurdenEval, burden
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, with_width

UTM = CRS.from_epsg(32734)


def _strip(n: int = 6, step: float = 10.0) -> Block:
    """A 1 x n strip of parcels with the street at one end -- depths run 1, 2, ..., n."""
    polys = [Polygon([(0, j * step), (step, j * step), (step, (j + 1) * step), (0, (j + 1) * step)])
             for j in range(n)]
    pts = [Point(step / 2, j * step + step / 2) for j in range(n)]
    return Block(
        block_id="strip", crs=UTM,
        boundary=Polygon([(0, 0), (step, 0), (step, n * step), (0, n * step)]),
        parcels=gpd.GeoDataFrame({"parcel_id": [str(k) for k in range(n)]},
                                 geometry=polys, crs=UTM),
        streets=gpd.GeoDataFrame(geometry=[LineString([(-step, 0.0), (step * 2, 0.0)])], crs=UTM),
        building_points=gpd.GeoDataFrame(geometry=pts, crs=UTM))


def test_burden_is_zero_exactly_at_universal_street_access() -> None:
    """THE defining property, and the reason for the zero-indexing. `parcel_access_layers` returns 1
    for a street-fronting parcel, so the un-shifted sum of squares scores a PERFECT block at n --
    which is why `budget.access_burden`'s form could never be read as a deficit.

    FAULT INJECTION: dropping the `- 1.0` makes an all-fronting block score 1.0 instead of 0.0.
    """
    assert burden(pd.Series([1, 1, 1, 1])) == 0.0
    assert burden(pd.Series([1, 2])) == pytest.approx(0.5)      # (0^2 + 1^2)/2
    assert burden(pd.Series([1, 2, 3])) == pytest.approx(5 / 3)  # (0 + 1 + 4)/3
    assert burden(pd.Series([], dtype=float)) == 0.0


def test_a_road_reaching_the_deep_end_reduces_the_burden() -> None:
    """End to end through the Eval, on a strip whose depths are 1..6 with no roads.

    FAULT INJECTION: scoring `access_before` in place of `access_after` makes the reduction 0.0.
    """
    block = _strip()
    ev = AccessBurdenEval()
    empty = Proposal(block_id=block.block_id, crs=UTM, method="none",
                     roads=with_width(gpd.GeoDataFrame(geometry=[], crs=UTM),
                                      DEFAULT_ROAD_WIDTH_M))
    spine_road = gpd.GeoDataFrame(geometry=[LineString([(5.0, 0.0), (5.0, 60.0)])], crs=UTM)
    spine = Proposal(block_id=block.block_id, crs=UTM, method="spine",
                     roads=with_width(spine_road, DEFAULT_ROAD_WIDTH_M))

    m0 = ev.score(block, empty)
    m1 = ev.score(block, spine)
    assert m0.values["burden_before"] == m0.values["burden_after"] > 0.0
    assert m0.values["burden_reduction"] == 0.0
    # a spine down the middle gives every parcel frontage -> universal access
    assert m1.values["burden_after"] == 0.0
    assert m1.values["burden_reduction"] == 1.0
    assert m1.values["k0_after"] == 0.0
    assert m1.values["share_deficient_after"] == 0.0
    assert m1.values["burden_before"] == m0.values["burden_before"]   # baseline is road-independent


def test_reduction_is_zero_not_nan_when_the_block_already_has_universal_access() -> None:
    """A block with nothing to fix must not divide by zero.

    FAULT INJECTION: `1.0 - b_post / b_pre` without the guard raises ZeroDivisionError here.
    """
    block = _strip(n=1)
    ev = AccessBurdenEval()
    m = ev.score(block, Proposal(block_id=block.block_id, crs=UTM, method="none", roads=None))
    assert m.values["burden_before"] == 0.0
    assert m.values["burden_reduction"] == 0.0


def test_it_reports_under_its_own_eval_name() -> None:
    """`Result.metric(eval, key)` looks metrics up BY EVAL NAME, so a collision with kcomplexity
    would silently shadow one of them -- and conf/eval/access_burden.yaml runs both together.

    FAULT INJECTION: renaming the eval to "kcomplexity" fails this.
    """
    block = _strip()
    m = AccessBurdenEval().score(
        block, Proposal(block_id=block.block_id, crs=UTM, method="none", roads=None))
    assert m.eval == "access_burden"
    assert set(m.fields) == {"access_before", "access_after"}
