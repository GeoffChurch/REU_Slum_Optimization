"""The router's four behavioural checks, each measured on the research code first:
straight roads cost their length at every angle; one gap carries every crossing pair; the bend
penalty moves traffic from a shorter zigzag to a straight route; repulsion widens routes."""
from __future__ import annotations

import math
import multiprocessing as mp

import numpy as np
import pytest
from scipy import ndimage

from reblock.methods.betweenness.routing import (
    bend_table,
    build_graph,
    egress_counts,
    pair_counts,
    route_cost,
    to_grid,
)


def test_a_straight_route_costs_its_length_at_every_angle() -> None:
    n, c0, R = 241, (120, 120), 100.0
    inside = np.ones((n, n), bool)
    cl = np.full((n, n), 1e6)
    for lam in (5.0, 20.0, 50.0):
        ratios = []
        for a in np.linspace(0, 90, 29):
            r = int(round(c0[0] + R * math.sin(math.radians(a))))
            c = int(round(c0[1] + R * math.cos(math.radians(a))))
            cost, _cells = route_cost(inside, cl, 1.0, 0.0, lam, c0, (r, c))
            ratios.append(cost / math.hypot(r - c0[0], c - c0[1]))
        ratios_a = np.array(ratios)
        assert np.max(np.abs(ratios_a / ratios_a[0] - 1)) <= 0.03, (lam, ratios_a)


def _two_rooms() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = 21, 41
    inside = np.ones((ny, nx), bool)
    wall = np.zeros((ny, nx), bool)
    wall[:, 20] = True
    wall[9:12, 20] = False
    cl = ndimage.distance_transform_edt(~wall) * 1.0
    cl[wall] = 0.0
    return inside, cl, wall


def test_one_gap_carries_every_crossing_pair_and_the_wall_none() -> None:
    inside, cl, wall = _two_rooms()
    g = build_graph(inside, cl, 1.0, 1.0)
    rng = np.random.default_rng(1)
    left = [(int(r), int(c))
            for r, c in zip(rng.integers(1, 20, 6), rng.integers(1, 18, 6), strict=True)]
    right = [(int(r), int(c))
             for r, c in zip(rng.integers(1, 20, 5), rng.integers(23, 40, 5), strict=True)]
    homes = np.array([g.node[p] for p in left + right], dtype=np.int64)
    cross = 2 * len(left) * len(right)
    for lam in (5.0, 50.0):
        f = to_grid(pair_counts(g, bend_table(lam), homes, homes, 1), g, inside)
        gap = float(np.nansum(f[9:12, 20]))
        assert cross <= gap <= 2 * cross      # every crossing route touches 1-2 gap cells
        assert float(np.nansum(f[wall])) == 0.0


def test_homes_sharing_a_cell_count_with_multiplicity() -> None:
    inside, cl, _wall = _two_rooms()
    g = build_graph(inside, cl, 1.0, 1.0)
    a, b = g.node[5, 5], g.node[5, 35]
    one = pair_counts(g, bend_table(5.0), np.array([a, b]), np.array([a, b]), 1)
    two = pair_counts(g, bend_table(5.0), np.array([a, a, b]), np.array([a, a, b]), 1)
    assert np.isclose(two.max(), 2 * one.max())   # 2 homes at `a` -> twice the a<->b traffic


def test_the_bend_penalty_moves_traffic_from_a_shorter_zigzag_to_a_straight_route() -> None:
    ny, nx = 90, 130
    inside = np.ones((ny, nx), bool)
    free = np.zeros((ny, nx), bool)
    P, Q = (45, 10), (45, 120)
    rr, cc = np.mgrid[0:ny, 0:nx]

    def carve(pts: list[tuple[int, int]], w: float) -> None:
        for (r1, c1), (r2, c2) in zip(pts[:-1], pts[1:], strict=True):
            for t in np.linspace(0, 1, max(1, int(math.hypot(r2 - r1, c2 - c1) * 4))):
                r, c = r1 + t * (r2 - r1), c1 + t * (c2 - c1)
                free[(rr - r) ** 2 + (cc - c) ** 2 <= (w / 2) ** 2] = True

    carve([P, (5, 65), Q], 3.0)                                      # straight-legged, 136 m
    carve([P, (51, 20)] + [(51 + (6 if i % 2 else 0), 20 + 10 * i) for i in range(1, 10)] + [Q],
          3.0)                                                        # zigzag, 132 m
    cl = ndimage.distance_transform_edt(free) * 1.0
    cl[~free] = 0.0
    g = build_graph(inside, cl, 1.0, 0.5)
    homes = np.array([g.node[P[0] + d, P[1]] for d in (-1, 0, 1)]
                     + [g.node[Q[0] + d, Q[1]] for d in (-1, 0, 1)], dtype=np.int64)

    def through(lam: float) -> tuple[float, float]:
        f = to_grid(pair_counts(g, bend_table(lam), homes, homes, 1), g, inside)
        return float(np.nanmax(f[3:20, 58:72])), float(np.nanmax(f[48:62, 58:72]))

    s_lo, w_lo = through(0.01)
    s_hi, w_hi = through(50.0)
    assert w_lo > s_lo and s_hi > w_hi


def test_repulsion_pushes_routes_into_wider_space() -> None:
    inside = np.ones((41, 41), bool)
    obs = np.zeros((41, 41), bool)
    obs[16:25, 16:25] = True
    cl = ndimage.distance_transform_edt(~obs) * 1.0
    cl[obs] = 0.0
    means = []
    for r0 in (0.25, 1.0, 4.0):
        _c, cells = route_cost(inside, cl, 1.0, r0, 50.0, (20, 2), (20, 38))
        means.append(float(cl[cells[:, 0], cells[:, 1]].mean()))
    assert means[0] < means[1] < means[2]


def test_egress_counts_each_home_once_on_its_way_to_the_street() -> None:
    inside = np.zeros((30, 30), bool)
    inside[1:-1, 1:-1] = True
    cl = np.where(inside, 1e6, np.nan)
    g = build_graph(inside, cl, 1.0, 0.0)
    band = g.node[inside & ((np.arange(30)[:, None] == 1) | (np.arange(30)[None, :] == 1))]
    home = np.array([g.node[20, 20]], dtype=np.int64)
    f = egress_counts(g, bend_table(50.0), home, band)
    assert f.max() == 1.0 and f[g.node[20, 20]] == 0.0      # endpoints are never credited


def _pairs_in_daemon(_: int) -> None:
    inside, cl, _wall = _two_rooms()
    g = build_graph(inside, cl, 1.0, 1.0)
    homes = np.array([g.node[5, 5], g.node[5, 35]], dtype=np.int64)
    pair_counts(g, bend_table(5.0), homes, homes, 2)


def test_parallel_pairs_inside_a_daemonic_worker_fail_by_name() -> None:
    with mp.get_context("fork").Pool(1) as pool, pytest.raises(RuntimeError, match="workers=1"):
        pool.map(_pairs_in_daemon, [0])
