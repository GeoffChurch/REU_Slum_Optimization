import math
from typing import cast

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.clearance import (
    _build_grid,
    _edge_weights,
    _node_clearance,
    _relax_depth,
    _sigmoid,
)


def test_sigmoid_is_bounded_and_symmetric() -> None:
    assert _sigmoid(0.0) == pytest.approx(0.5)
    assert _sigmoid(6.0) == pytest.approx(1.0, abs=1e-2)
    assert _sigmoid(-6.0) == pytest.approx(0.0, abs=1e-2)
    assert _sigmoid(3.0) + _sigmoid(-3.0) == pytest.approx(1.0)
    # never saturates to exactly 0/1 (weights stay finite), and no overflow at extreme s
    assert 0.0 < _sigmoid(-800.0) < _sigmoid(800.0) < 1.0


def test_build_grid_is_8_connected_and_inside_boundary() -> None:
    # contains_xy is strict (excludes the boundary), so a 4x4 box at res=1 gives interior nodes
    # {1,2,3}x{1,2,3}; the center (2,2) is a true interior node with 8 neighbours.
    boundary = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    pts, rows, cols, edist = _build_grid(boundary, 1.0)
    assert len(pts) > 0
    assert all(boundary.contains(Point(p)) for p in pts)   # strict, matches contains_xy
    # edges are symmetric (both directions present) and lengths are 1 or sqrt(2)
    assert len(rows) == len(cols) == len(edist)
    assert set(np.round(np.unique(edist), 6)) <= {1.0, round(math.sqrt(2.0), 6)}
    undirected = {frozenset((int(a), int(b))) for a, b in zip(rows, cols, strict=True)}
    assert len(undirected) * 2 == len(rows)  # every undirected edge stored both ways
    # an interior node has 8 neighbours
    tree = cKDTree(pts)
    center = int(tree.query([2.0, 2.0])[1])
    assert int((rows == center).sum()) == 8


def test_node_clearance_is_euclidean_when_unweighted() -> None:
    pts = np.array([[0.0, 0.0], [5.0, 0.0]])
    buildings = np.array([[0.0, 0.0]])
    radii = np.zeros(1)
    clear = _node_clearance(pts, buildings, radii)
    # node ON the building -> eps; node 5 away -> 5 + eps
    assert clear[0] == pytest.approx(0.3)
    assert clear[1] == pytest.approx(5.3)


def test_node_clearance_weighted_radius_shrinks_clearance() -> None:
    pts = np.array([[5.0, 0.0]])
    buildings = np.array([[0.0, 0.0]])
    plain = _node_clearance(pts, buildings, np.zeros(1))
    weighted = _node_clearance(pts, buildings, np.array([3.0]))  # radius-3 footprint
    assert weighted[0] < plain[0]
    assert weighted[0] == pytest.approx(5.0 - 3.0 + 0.3)


def test_node_clearance_no_buildings_is_uniform() -> None:
    pts = np.array([[0.0, 0.0], [10.0, 10.0]])
    clear = _node_clearance(pts, np.empty((0, 2)), np.zeros(0))
    assert clear[0] == clear[1]  # uniform -> straight regardless of t


def test_repulsion_bends_the_path_around_buildings() -> None:
    # A straight route (t≈0) crosses a vertical wall of buildings; a repelled route (t≈1)
    # bows away and stays farther from them, at >= the straight length.
    boundary = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    buildings = np.array([[5.0, 3.0], [5.0, 5.0], [5.0, 7.0]])
    pts, rows, cols, edist = _build_grid(boundary, 0.5)
    clear = _node_clearance(pts, buildings, np.zeros(len(buildings)))
    tree = cKDTree(pts)
    src = int(tree.query([5.0, 9.0])[1])
    dst = int(tree.query([5.0, 1.0])[1])

    def route(t: float) -> tuple[float, float]:
        w = _edge_weights(clear, t, rows, cols, edist)
        csr = csr_matrix((w, (rows, cols)), shape=(len(pts), len(pts)))
        _d, pred, _s = dijkstra(csr, indices=[src], return_predecessors=True, min_only=True)
        node, path = dst, [dst]
        while pred[node] >= 0:
            node = int(pred[node])
            path.append(node)
        line = LineString([tuple(pts[k]) for k in path])
        min_clear = min(
            Point(cast(tuple[float, float], tuple(b))).distance(line) for b in buildings
        )
        return float(line.length), float(min_clear)

    len_straight, clear_straight = route(_sigmoid(-6.0))
    len_repelled, clear_repelled = route(_sigmoid(6.0))
    assert clear_straight < clear_repelled           # repelled path keeps farther from buildings
    assert len_repelled >= len_straight - 1e-9        # ...at no less than the straight length


UTM = CRS.from_epsg(32643)


def _column_block(h: int) -> Block:
    """A 1-wide, h-tall column of unit parcels with street frontage only on the bottom edge ->
    access depth 1..h from the street upward. parcel_id == row index (bottom = 0)."""
    polys = [Polygon([(0, j), (1, j), (1, j + 1), (0, j + 1)]) for j in range(h)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(h))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, unary_union(polys))
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs=UTM)
    return Block(block_id="col", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_relax_depth_matches_full_recompute() -> None:
    # A single street-connected road up the column should, incrementally, reproduce exactly what
    # parcel_access_layers computes from scratch for that road.
    block = _column_block(6)
    geoms = list(block.parcels.geometry)
    adj = parcel_adjacency(geoms, STREET_TOL)
    depth = parcel_access_layers(block, None, adj=adj).to_numpy().astype(float)
    assert list(depth) == [1, 2, 3, 4, 5, 6]  # sanity: deep column

    road = gpd.GeoDataFrame(geometry=[LineString([(0.5, 0.0), (0.5, 6.0)])], crs=UTM)
    served = [int(p) for p in STRtree(geoms).query(
        road.geometry.iloc[0], predicate="dwithin", distance=STREET_TOL)]
    _relax_depth(depth, adj, served)

    naive = parcel_access_layers(block, road, adj=adj).to_numpy().astype(float)
    assert list(depth) == list(naive)
    assert max(depth) == 1.0  # every parcel now fronts the street-connected road


def test_relax_depth_matches_recompute_on_disconnected_component() -> None:
    # The relax equals a full recompute ONLY when the base array pins unreached parcels to a
    # high sentinel (len+1). This locks in that precondition and shows the default-seeded base
    # (unreached = max(reached)+1) diverges -- which is exactly why Task 3's greedy seeds with
    # unreached_depth=len+1. Row A (3 parcels) fronts the street; column B (5 parcels, disjoint
    # from A) is unreached until a road connects its near end.
    a = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i in range(3)]
    b = [Polygon([(0, y), (1, y), (1, y + 1), (0, y + 1)]) for y in range(5, 10)]  # gap at y=1..5
    polys = a + b
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, unary_union(polys))
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (3, 0)])], crs=UTM)
    block = Block(block_id="disc", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)
    adj = parcel_adjacency(cast(list[BaseGeometry], polys), STREET_TOL)
    n = len(polys)

    # street-connected road reaching ONLY B's near end (top at y=5.4 -> >0.5 from B[1] at y=6)
    road = gpd.GeoDataFrame(geometry=[LineString([(0.5, 0.0), (0.5, 5.4)])], crs=UTM)
    served = [int(p) for p in STRtree(polys).query(
        road.geometry.iloc[0], predicate="dwithin", distance=STREET_TOL)]
    naive = parcel_access_layers(block, road, adj=adj).to_numpy().astype(float)

    seeded = parcel_access_layers(
        block, None, adj=adj, unreached_depth=n + 1).to_numpy().astype(float)
    _relax_depth(seeded, adj, served)
    assert list(seeded) == list(naive)                 # correct precondition -> exact

    default_base = parcel_access_layers(block, None, adj=adj).to_numpy().astype(float)
    _relax_depth(default_base, adj, served)
    assert list(default_base) != list(naive)           # default seeding -> falsely shallow
