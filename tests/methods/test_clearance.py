import math
from typing import cast

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon

from reblock.methods.clearance import (
    _build_grid,
    _edge_weights,
    _node_clearance,
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
