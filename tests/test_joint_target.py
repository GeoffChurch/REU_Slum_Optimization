import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.budget import Curve, prefix_to_joint_target
from reblock.contracts import Block

UTM = CRS.from_epsg(32734)


def _block() -> Block:
    boundary = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    parcels = gpd.GeoDataFrame({"parcel_id": [0]}, geometry=[boundary], crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (100, 0)])], crs=UTM)
    return Block(block_id="b", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _roads(n: int) -> gpd.GeoDataFrame:  # n unit segments so truncate_to_length has lengths to cut
    return gpd.GeoDataFrame(geometry=[LineString([(i, 0), (i, 1)]) for i in range(n)], crs=UTM)


def _curves(ext, inte, disp):
    cost = [float(i) for i in range(len(ext))]
    return Curve(cost, ext), Curve(cost, inte), Curve(cost, disp)


def test_first_qualifying_index() -> None:
    ext, inte, disp = _curves([0.4, 0.6, 0.75, 0.8], [0.0, 0.1, 0.30, 0.35],
                              [0.05, 0.10, 0.20, 0.30])
    o = prefix_to_joint_target(_block(), _roads(4), ext, inte, disp,
                               i_min=0.25, e_min=0.70, d_max=0.45)
    # first index with ext>=.70 AND int>=.25
    assert o.reached and o.sample_index == 2


def test_touch_and_go_stops_at_first_cross() -> None:
    # internal crosses at i=1 then dips below; still stop at i=1
    ext, inte, disp = _curves([0.7, 0.72, 0.74], [0.30, 0.20, 0.31], [0.1, 0.2, 0.3])
    o = prefix_to_joint_target(_block(), _roads(3), ext, inte, disp,
                               i_min=0.25, e_min=0.70, d_max=0.45)
    assert o.reached and o.sample_index == 0


def test_killed_internal_below() -> None:
    ext, inte, disp = _curves([0.8, 0.9, 0.95], [0.0, 0.02, 0.03], [0.1, 0.2, 0.3])
    o = prefix_to_joint_target(_block(), _roads(3), ext, inte, disp,
                               i_min=0.25, e_min=0.70, d_max=0.45)
    assert not o.reached and o.reason == "internal_below"


def test_killed_over_budget() -> None:
    # floors met only at i=2 where displacement 0.5 > d_max 0.45
    ext, inte, disp = _curves([0.6, 0.68, 0.75], [0.1, 0.2, 0.30], [0.2, 0.4, 0.5])
    o = prefix_to_joint_target(_block(), _roads(3), ext, inte, disp,
                               i_min=0.25, e_min=0.70, d_max=0.45)
    assert not o.reached and o.reason == "over_budget"


def test_max_internal_within() -> None:
    from reblock.budget import Curve
    from scripts.calibrate_joint_target import max_internal_within
    ext = Curve([0, 1, 2, 3], [0.5, 0.72, 0.8, 0.9])
    inte = Curve([0, 1, 2, 3], [0.1, 0.30, 0.45, 0.50])
    disp = Curve([0, 1, 2, 3], [0.1, 0.2, 0.40, 0.60])
    # samples with ext>=.70 and disp<=.45: i=1 (int .30), i=2 (int .45); i=3 disp .60 excluded
    assert max_internal_within(ext, inte, disp, e_min=0.70, d_max=0.45) == 0.45
    assert max_internal_within(ext, inte, disp, e_min=0.99, d_max=0.45) == float("-inf")
