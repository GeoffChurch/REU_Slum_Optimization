from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.contracts import Block
from reblock.derive.geometric_access import geometric_access_distances

UTM = CRS.from_epsg(32643)


def _strip(n: int) -> Block:
    # 1xN strip, street on the left edge; parcel centroids at x=0.5,1.5,... so the k-th parcel
    # is ~k metres from the street through the adjacency chain.
    polys = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i in range(n)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(n))}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])], crs=UTM)
    return Block(block_id="s", crs=UTM, boundary=cast(Polygon, parcels.geometry.union_all()),
                 parcels=parcels, streets=streets)


def test_distance_grows_down_the_strip() -> None:
    d = geometric_access_distances(_strip(5), None)
    assert d.loc[0] == 0.0                       # touches the street
    assert d.loc[4] > d.loc[2] > d.loc[0]        # monotone in metres, not just hops
    assert abs(d.loc[4] - 4.0) < 1e-6            # 4 centroid-hops of 1 m each


def test_roads_add_street_sources() -> None:
    block = _strip(5)
    # A spine that touches the street at x=0 and reaches into parcel 4 -- street-connected,
    # so it should add a source there.
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0.5), (4.5, 0.5)])], crs=UTM)
    assert geometric_access_distances(block, roads).loc[4] == 0.0


def test_floating_road_grants_no_access() -> None:
    block = _strip(5)
    # Same road as before the fix, but now it does NOT touch the street at x=0 -- floating,
    # so it must be ignored, same as the topological peel (`street_connectivity`).
    roads = gpd.GeoDataFrame(geometry=[LineString([(4.5, 0), (4.5, 1)])], crs=UTM)  # near parcel 4
    with_roads = geometric_access_distances(block, roads).loc[4]
    without_roads = geometric_access_distances(block, None).loc[4]
    assert with_roads == without_roads
    assert abs(without_roads - 4.0) < 1e-6
