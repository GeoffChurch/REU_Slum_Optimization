from typing import cast

import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry import Polygon
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, street_connectivity
from reblock.eval.kcomplexity import KComplexityEval
from reblock.methods.dijkstra import DijkstraReblocker, _reblock_dijkstra

UTM = CRS.from_epsg(32643)


def _grid_block(n: int) -> Block:
    polys, ids = [], []
    for i in range(n):
        for j in range(n):
            polys.append(Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]))
            ids.append(i * n + j)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="grid", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_reblock_dijkstra_is_deterministic() -> None:
    block = _grid_block(5)
    r1, r2 = _reblock_dijkstra(block), _reblock_dijkstra(block)
    assert [g.wkt for g in r1.geometry] == [g.wkt for g in r2.geometry]


def test_reblock_dijkstra_roads_all_reach_the_street() -> None:
    # forest rooted at street + attached spurs -> every segment street-connected
    block = _grid_block(5)
    roads = _reblock_dijkstra(block)
    assert len(roads) > 0
    conn = street_connectivity(block.streets, roads, STREET_TOL)
    assert conn.connected_frac == 1.0


def test_reblock_dijkstra_has_ordered_drainage() -> None:
    roads = _reblock_dijkstra(_grid_block(5))
    drains = list(roads["drain"])
    assert all(d >= 1 for d in drains)               # every road serves >=1 parcel
    assert drains == sorted(drains, reverse=True)     # arterials first
    assert max(drains) > 1                             # a real arterial exists (shared prefix)


def _t_junction_block() -> Block:
    # 3x3 area (0,0)-(3,3): centre unit square (1,1)-(2,2); each of the 4 edge cells is
    # SPLIT into two (creating a T-junction at each of the centre's edge-midpoints); 4 corners.
    polys = [
        Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),                 # 0: centre (interior)
        Polygon([(1, 0), (1.5, 0), (1.5, 1), (1, 1)]),             # bottom split L
        Polygon([(1.5, 0), (2, 0), (2, 1), (1.5, 1)]),             # bottom split R
        Polygon([(1, 2), (1.5, 2), (1.5, 3), (1, 3)]),             # top split L
        Polygon([(1.5, 2), (2, 2), (2, 3), (1.5, 3)]),             # top split R
        Polygon([(0, 1), (1, 1), (1, 1.5), (0, 1.5)]),             # left split B
        Polygon([(0, 1.5), (1, 1.5), (1, 2), (0, 2)]),             # left split T
        Polygon([(2, 1), (3, 1), (3, 1.5), (2, 1.5)]),             # right split B
        Polygon([(2, 1.5), (3, 1.5), (3, 2), (2, 2)]),             # right split T
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),                 # corners
        Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        Polygon([(0, 2), (1, 2), (1, 3), (0, 3)]),
        Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
    ]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, unary_union(polys))
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="tjunc", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_reblock_dijkstra_covers_t_junction_parcels() -> None:
    block = _t_junction_block()
    roads = _reblock_dijkstra(block)
    assert len(roads) > 0                                  # centre not stranded (pre-fix: empty)
    conn = street_connectivity(block.streets, roads, STREET_TOL)
    assert conn.connected_frac == 1.0
    centre = cast(Polygon, block.parcels.geometry.iloc[0])  # the interior centre square
    rnet = unary_union(list(roads.geometry))
    assert centre.exterior.distance(rnet) <= STREET_TOL    # a road reaches its frontage


def test_dijkstra_reblocks_a_synthetic_nested_block() -> None:
    # 3x3 grid: the centre parcel is landlocked at peel-depth 2; the boundary-routed
    # network reaches it, so k_after collapses to 1 (matches the peel/topology capstone).
    block = _grid_block(3)
    proposal = DijkstraReblocker().propose(block)
    m = KComplexityEval().score(block, proposal).values
    assert m["k_before"] == 2.0
    assert m["k_after"] == 1.0 and m["delta_k"] > 0
    assert m["connected_road_frac"] == 1.0
    assert proposal.roads is not None and len(proposal.roads) > 0
    assert proposal.proposal_id == "dijkstra" and proposal.method == "dijkstra"


def test_dijkstra_propose_is_deterministic_and_leaves_rng_untouched() -> None:
    block = _grid_block(5)
    np.random.seed(123)
    state = np.random.get_state()[1].tolist()
    p1 = DijkstraReblocker().propose(block)
    p2 = DijkstraReblocker().propose(block)
    assert np.random.get_state()[1].tolist() == state          # no global RNG side-effect
    assert p1.roads is not None and p2.roads is not None
    assert [g.wkt for g in p1.roads.geometry] == [g.wkt for g in p2.roads.geometry]
