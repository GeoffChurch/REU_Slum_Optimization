"""The fitted map and the line warp: anchors carry their targets exactly, polylines keep their
vertex structure, and a rigid copy of a block is transplanted onto itself."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString

from reblock.transplant.gw import GWParams, gw_cost
from reblock.transplant.transport import (
    Transport,
    TransportParams,
    fit_transport,
    normalized_dist_matrix,
    transport_lines,
)

UTM = CRS.from_epsg(32734)
PARAMS = TransportParams(gw=GWParams(eps=0.01, tau=1.0, outer_iters=30, inner_iters=100),
                         idw_k=8)


def test_the_distance_matrix_is_scale_free_and_refuses_a_point() -> None:
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    d = normalized_dist_matrix(xy)
    assert d.max() == 1.0
    np.testing.assert_array_equal(d, normalized_dist_matrix(xy * 37.0 + 5.0))
    with pytest.raises(ValueError, match="degenerate"):
        normalized_dist_matrix(np.ones((4, 2)))


def test_the_fit_reports_the_exact_objective_of_its_own_coupling() -> None:
    rng = np.random.default_rng(11)
    x, y = rng.uniform(0, 50, size=(20, 2)), rng.uniform(0, 80, size=(30, 2))
    fit = fit_transport(x, y, PARAMS)
    assert fit.pi.shape == (20, 30)
    assert fit.gw_dist == gw_cost(fit.pi, normalized_dist_matrix(x), normalized_dist_matrix(y))
    assert fit.idw_k == PARAMS.idw_k


def _identity_map(anchors: np.ndarray, targets: np.ndarray, k: int) -> Transport:
    n = len(anchors)
    return Transport(pi=np.eye(n) / n, bary=targets, donor_xy=anchors, recipient_xy=targets,
                     gw_dist=0.0, idw_k=k)


def test_a_vertex_on_an_anchor_takes_that_anchors_target() -> None:
    """The IDW floor is tiny, so an anchor dominates its own neighbourhood: a vertex exactly on it
    lands on its target, and every vertex keeps its place in its line."""
    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [5.0, 5.0]])
    targets = anchors * 2.0 + [100.0, 100.0]
    lines = gpd.GeoDataFrame(geometry=[LineString(anchors[:3])], crs=UTM)
    out = transport_lines(lines, _identity_map(anchors, targets, 3), crs=UTM)
    np.testing.assert_allclose(np.asarray(out.geometry.iloc[0].coords), targets[:3], atol=1e-6)
    assert out.crs == UTM


def test_parts_are_split_and_degenerate_parts_dropped() -> None:
    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    lines = gpd.GeoDataFrame(geometry=[
        MultiLineString([[(0, 0), (1, 1), (2, 1)], [(5, 5), (6, 6)]]),
        LineString([(3, 3), (4, 3)]),
    ], crs=UTM)
    out = transport_lines(lines, _identity_map(anchors, anchors, 2), crs=UTM)
    assert [len(g.coords) for g in out.geometry] == [3, 2, 2]


def test_a_rotated_reshuffled_copy_receives_the_lines_where_they_were() -> None:
    """The mechanism check the 2026-07-23 study ran on the flagship block, in miniature: warp a
    network onto a rotated, translated and reshuffled copy of its own anchors, and it must land
    near where the rigid motion puts it, at close to its own length.

    Entropic blur pulls barycentric targets toward the centroid, which is what both bounds watch:
    here eps=0.01 keeps 80% of the length with the worst vertex 15 m off, where eps=0.05 keeps 53%
    at 32 m and eps=0.2 collapses the line to 5% of itself.

    FAULT INJECTION: fitting at eps=0.05 fails both assertions.
    """
    rng = np.random.default_rng(3)
    donor = rng.uniform(0, 100, size=(40, 2))
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    def move(p: np.ndarray) -> np.ndarray:
        return np.asarray(p @ rot.T + [1000.0, 500.0])

    recipient = move(donor)[rng.permutation(40)]
    path = np.array([[10.0, 10.0], [50.0, 40.0], [90.0, 80.0]])
    lines = gpd.GeoDataFrame(geometry=[LineString(path)], crs=UTM)
    out = transport_lines(lines, fit_transport(donor, recipient, PARAMS), crs=UTM)
    err = np.linalg.norm(np.asarray(out.geometry.iloc[0].coords) - move(path), axis=1)
    assert err.max() < 20.0, err                     # the block's diagonal is ~141 m
    assert out.geometry.iloc[0].length > 0.7 * LineString(path).length
