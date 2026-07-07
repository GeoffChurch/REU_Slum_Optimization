from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Polygon, box

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, parcel_access_layers, street_connectivity
from reblock.methods.peel import PeelReblocker

UTM = CRS.from_epsg(32643)


def _grid5() -> Block:
    polys = [box(i, j, i + 1, j + 1) for i in range(5) for j in range(5)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(25))}, geometry=polys, crs=UTM)
    b = cast(Polygon, parcels.geometry.union_all())
    return Block(block_id="g5", crs=UTM, boundary=b, parcels=parcels,
                 streets=gpd.GeoDataFrame(geometry=[b.exterior], crs=UTM))


def test_spine_reaches_k1_and_is_street_connected() -> None:
    block = _grid5()
    proposal = PeelReblocker().propose(block)
    assert parcel_access_layers(block, proposal.roads).max() == 1          # full access
    sc = street_connectivity(block.streets, proposal.roads, STREET_TOL)
    assert sc.connected_frac == 1.0                            # every corridor reaches street
    assert proposal.proposal_id == "peel" and proposal.method == "peel"
    assert proposal.edges is None


def test_deterministic_under_row_shuffle() -> None:
    # parcel_id != row position, rows shuffled: identical roads (min-id tie-break,
    # not row order). Compare sorted WKT of the produced segments.
    block = _grid5()
    shuffled = block.parcels.sample(frac=1, random_state=3).reset_index(drop=True)
    block2 = Block(block_id="g5", crs=block.crs, boundary=block.boundary,
                   parcels=shuffled, streets=block.streets)
    roads1 = PeelReblocker().propose(block).roads
    roads2 = PeelReblocker().propose(block2).roads
    assert roads1 is not None and roads2 is not None
    r1 = sorted(g.wkt for g in roads1.geometry)
    r2 = sorted(g.wkt for g in roads2.geometry)
    assert r1 == r2


def test_head_to_head_both_reach_k1_peel_connected() -> None:
    from reblock.eval.kcomplexity import KComplexityEval
    from reblock.methods.topology import TopologyMethod
    block = _grid5()
    topo = KComplexityEval().score(block, TopologyMethod(alpha=2.0, seed=0).propose(block)).values
    peel = KComplexityEval().score(block, PeelReblocker().propose(block)).values
    assert topo["k_after"] == 1.0 and peel["k_after"] == 1.0  # both fully reblock
    assert peel["connected_road_frac"] == 1.0                 # peel network reaches the street
    assert peel["added_road_length_m"] > 0                    # it actually laid roads


def test_unreachable_island_is_skipped_and_counted() -> None:
    # A parcel disconnected from everything (no adjacency, no street) has no
    # descent parent -> skipped, counted, and left deep in k_after.
    near = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])          # touches left-edge street
    mid = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])           # layer 2 via `near`
    island = Polygon([(50, 50), (51, 50), (51, 51), (50, 51)])  # disconnected
    parcels = gpd.GeoDataFrame({"parcel_id": [0, 1, 2]}, geometry=[near, mid, island], crs=UTM)
    from shapely.geometry import LineString
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])], crs=UTM)
    hull = cast(Polygon, parcels.geometry.union_all().convex_hull)
    block = Block(block_id="d", crs=UTM, boundary=hull, parcels=parcels, streets=streets)
    proposal = PeelReblocker().propose(block)
    assert proposal.params["unreachable"] == 1
