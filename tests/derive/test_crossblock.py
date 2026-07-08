import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString, Polygon, box

from reblock.contracts import Block, Proposal, Region
from reblock.derive.crossblock import reconciled_baseline, spine_merge_reference

UTM = CRS.from_epsg(32734)


def _block(bid: str, poly: Polygon, xs: list[float]) -> Block:
    # A genuine 2D grid (xs columns x rows tiling poly's height in 1-unit steps),
    # not a single row of full-height columns: a full-height column touches both
    # the top and bottom block edges, so with streets = the whole block boundary
    # every parcel would be depth-1 and PeelReblocker would emit zero segments.
    # The grid's border ring is depth-1 and its interior cells are depth >= 2, so
    # peel actually has something to route.
    miny, maxy = poly.bounds[1], poly.bounds[3]
    ys = [miny + i for i in range(int(maxy - miny))]
    polys = [box(x, y, x + 1, y + 1) for x in xs for y in ys]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[poly.boundary], crs=UTM)
    return Block(block_id=bid, crs=UTM, boundary=poly, parcels=parcels, streets=streets)


def test_reconciled_baseline_unions_per_block_roads() -> None:
    from reblock.derive.cluster import merge_cluster
    a = _block("a", box(0, 0, 4, 3), [0, 1, 2, 3])
    b = _block("b", box(4, 0, 8, 3), [4, 5, 6, 7])
    region = Region(region_id="t", crs=UTM, blocks=[a, b])
    merged = merge_cluster(region)
    prop = reconciled_baseline(region, merged)
    # peel produced roads for both blocks
    assert prop.roads is not None and not prop.roads.empty
    assert prop.block_id == merged.block_id


def test_spine_merge_adds_a_crossing_trunk() -> None:
    from reblock.derive.network_metrics import n_cross_block_streets
    interior = MultiLineString([[(4, 0), (4, 3)]])
    merged = Block(
        block_id="a+b", crs=UTM, boundary=box(0, 0, 8, 3),
        parcels=gpd.GeoDataFrame({"parcel_id": [0]}, geometry=[box(0, 0, 8, 3)], crs=UTM),
        streets=gpd.GeoDataFrame(geometry=[box(0, 0, 8, 3).boundary], crs=UTM),
        attrs={"interior_boundaries": interior})
    # two boundary-parallel spines flanking x=4, no crossing yet
    base = Proposal(block_id="a+b", crs=UTM, method="peel", roads=gpd.GeoDataFrame(
        geometry=[LineString([(3.0, 0.5), (3.0, 2.5)]), LineString([(5.0, 0.5), (5.0, 2.5)])],
        crs=UTM))
    assert n_cross_block_streets(base.roads, interior) == 0
    ref = spine_merge_reference(merged, base)
    # a through-trunk now crosses x=4
    assert n_cross_block_streets(ref.roads, interior) >= 1
