from typing import cast

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.budget import Curve, access_burden, auc, cost_benefit_curve, road_drainage
from reblock.contracts import Block
from reblock.methods.dijkstra import DijkstraReblocker

UTM = CRS.from_epsg(32643)


def _grid_block(n: int) -> Block:
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(n) for j in range(n)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(n * n))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_access_burden_is_sum_of_squared_depths() -> None:
    assert access_burden(pd.Series([1, 2, 3])) == 1 + 4 + 9


def test_road_drainage_trunks_exceed_leaves() -> None:
    # dijkstra's roads on a 5x5 grid: a segment near the street carries more parcels than a leaf.
    block = _grid_block(5)
    roads = DijkstraReblocker().propose(block).roads
    assert roads is not None
    drain = road_drainage(block, roads)
    assert len(drain) == len(roads) and max(drain) > min(drain) and max(drain) >= 2


def test_cost_benefit_curve_is_monotonic_and_reaches_full() -> None:
    block = _grid_block(5)
    roads = DijkstraReblocker().propose(block).roads
    assert roads is not None
    curve = cost_benefit_curve(block, roads, n_points=10)
    assert curve.cost[0] == 0.0 and curve.benefit[0] == 0.0
    assert curve.benefit == sorted(curve.benefit)          # monotonic non-decreasing
    assert curve.benefit[-1] > 0.5                         # flattens the block substantially


def test_auc_rewards_reaching_benefit_at_lower_cost() -> None:
    cheap = Curve(cost=[0.0, 1.0], benefit=[0.0, 1.0])     # full benefit by cost 1
    dear = Curve(cost=[0.0, 4.0], benefit=[0.0, 1.0])      # full benefit only by cost 4
    assert auc(cheap, cost_cap=4.0) > auc(dear, cost_cap=4.0)
    assert 0.0 <= auc(dear, cost_cap=4.0) <= 1.0


def test_auc_interpolates_a_cap_straddling_segment() -> None:
    # A curve whose data crosses the cap BETWEEN points must interpolate the partial area,
    # not drop the whole segment (regression: dropped it -> 0.30 instead of 0.5125).
    c = Curve(cost=[0.0, 3.0, 5.0], benefit=[0.0, 0.8, 1.0])
    assert abs(auc(c, cost_cap=4.0) - 0.5125) < 1e-6
