import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import MultiLineString, Polygon, box

from reblock.contracts import Block, Region
from reblock.derive.cluster import merge_cluster

UTM = CRS.from_epsg(32734)


def _block(bid: str, poly: Polygon, n: int) -> Block:
    # n unit-ish parcels tiling the block, ids 0..n-1; streets = the block boundary.
    minx, miny, maxx, maxy = poly.bounds
    w = (maxx - minx) / n
    polys = [box(minx + i * w, miny, minx + (i + 1) * w, maxy) for i in range(n)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(n))}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[poly.boundary], crs=UTM)
    return Block(block_id=bid, crs=UTM, boundary=poly, parcels=parcels, streets=streets,
                 attrs={"kblock_k": 3.0})


def _region(*blocks: Block) -> Region:
    return Region(region_id="t", crs=UTM, blocks=list(blocks))


def test_merge_two_adjacent_blocks() -> None:
    a = _block("a", box(0, 0, 10, 10), 2)
    b = _block("b", box(10, 0, 20, 10), 3)          # shares the x=10 edge with a
    m = merge_cluster(_region(a, b))
    assert isinstance(m.boundary, Polygon)           # contiguous -> single polygon
    assert len(m.parcels) == 5                        # 2 + 3
    assert list(m.parcels["parcel_id"]) == [0, 1, 2, 3, 4]   # re-indexed, unique
    assert m.attrs["block_ids"] == ["a", "b"]
    assert isinstance(m.attrs["interior_boundaries"], MultiLineString)
    assert m.attrs["interior_boundaries"].length > 0   # the shared x=10 edge


def test_merge_non_adjacent_raises() -> None:
    a = _block("a", box(0, 0, 10, 10), 2)
    c = _block("c", box(50, 50, 60, 60), 2)          # disjoint from a
    with pytest.raises(ValueError, match="not contiguous"):
        merge_cluster(_region(a, c))
