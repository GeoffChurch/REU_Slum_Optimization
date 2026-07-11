"""Tests for the resistance metric lens (see
docs/superpowers/specs/2026-07-11-resistance-eval-design.md): `_resistance_core`, the pure
grounded-Laplacian numeric core (tiny hand-built CSRs with known analytic effective resistances,
task 1's scope), plus `_BlockScoringContext.resistance_frozen`/`resistance_benefit`, the
block-scoring wiring (task 2's scope, mirroring `_efficiency_factory`/`score_frozen`)."""
from __future__ import annotations

from typing import cast

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from shapely.geometry import LineString, Polygon

from reblock.budget import (
    _BlockScoringContext,
    _build_csr,
    _explode_segments,
    _Node,
    _resistance_core,
    cost_benefit_curve,
    resistance_benefit,
)
from reblock.contracts import Block
from reblock.methods.dijkstra import DijkstraReblocker
from reblock.methods.mesh import MeshReblocker
from reblock.methods.peel import PeelReblocker

CAP = 1000.0
UTM = CRS.from_epsg(32643)


def _grid_block(n: int) -> Block:
    """n x n grid of unit parcels, street frontage on the WHOLE boundary (all sides) -- the same
    fixture `test_budget.py` uses for its cost-benefit/method-comparison tests."""
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(n) for j in range(n)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(n * n))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _deep_column_block(depth: int) -> Block:
    """A single 10m-wide column of `depth` stacked 10x10 parcels, street frontage on the bottom
    edge only -- deep, so only the bottom parcel fronts the street directly. Paired with a single
    straight spur road up the column's centerline (see `test_tree_equals_shortest_path`), this
    keeps each entry's own graph component an unbranched CHAIN reaching exactly one ground node,
    so grounded resistance is provably identical to shortest-path distance there (no parallel-path
    shortcut a branching/coalescing road tree could introduce -- see that test's docstring)."""
    polys = [Polygon([(0.0, 10.0 * j), (10.0, 10.0 * j),
                      (10.0, 10.0 * (j + 1)), (0.0, 10.0 * (j + 1))])
             for j in range(depth)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(depth))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[LineString([(0.0, 0.0), (10.0, 0.0)])], crs=UTM)
    return Block(block_id="col", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _csr(nodes: list[_Node], edges: list[tuple[int, int, float]]
        ) -> tuple[csr_matrix, dict[_Node, int]]:
    """A symmetric CSR over `nodes` (index = position in the list) with each `(u, v, length)` in
    `edges` written into both (u, v) and (v, u)."""
    node_index = {n: i for i, n in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for u, v, length in edges:
        rows += [u, v]
        cols += [v, u]
        data += [length, length]
    csr = csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
    return csr, node_index


def test_single_edge_resistance() -> None:
    # g (ground) --1.0-- a (free); one parcel entering at a, rep_xy == a so leg == 0.
    # (L_G^-1)_aa == 1/conductance == length == 1.0 -> R == 1.0.
    g, a = (0.0, 0.0), (1.0, 0.0)
    csr, node_index = _csr([g, a], [(0, 1, 1.0)])
    entry: list[_Node | None] = [a]
    rep_xy = np.array([[1.0, 0.0]])
    ground_idx = np.array([0])
    r = _resistance_core(csr, node_index, entry, rep_xy, ground_idx, CAP)
    assert r == pytest.approx(1.0, abs=1e-9)


def test_parallel_paths_halve() -> None:
    # g (ground) reaches a (free) via two independent length-0.5+0.5 series paths through b1/b2.
    # Each path has series resistance 1.0; two such paths in parallel -> effective R == 0.5.
    g, b1, b2, a = (0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (1.0, 0.0)
    csr, node_index = _csr(
        [g, b1, b2, a],
        [(0, 1, 0.5), (1, 3, 0.5), (0, 2, 0.5), (2, 3, 0.5)],
    )
    entry: list[_Node | None] = [a]
    rep_xy = np.array([[1.0, 0.0]])
    ground_idx = np.array([0])
    r = _resistance_core(csr, node_index, entry, rep_xy, ground_idx, CAP)
    assert r == pytest.approx(0.5, abs=1e-9)


def test_unreached_is_cap() -> None:
    # Parcel 0 has no entry at all; parcel 1's entry sits in a component with NO ground node.
    g, a, c, d = (0.0, 0.0), (1.0, 0.0), (10.0, 0.0), (11.0, 0.0)
    csr, node_index = _csr([g, a, c, d], [(0, 1, 1.0), (2, 3, 1.0)])
    entry: list[_Node | None] = [None, d]
    rep_xy = np.array([[0.0, 0.0], [11.0, 0.0]])
    ground_idx = np.array([0])
    r = _resistance_core(csr, node_index, entry, rep_xy, ground_idx, 500.0)
    assert r == pytest.approx(500.0, abs=1e-9)


def test_entry_on_ground_is_leg_only() -> None:
    # Entry node IS the ground node g; drive term is 0, so R == leg exactly (leg != 0 here).
    g, a = (0.0, 0.0), (1.0, 0.0)
    csr, node_index = _csr([g, a], [(0, 1, 1.0)])
    entry: list[_Node | None] = [g]
    rep_xy = np.array([[3.0, 4.0]])
    leg = float(np.hypot(3.0, 4.0))
    ground_idx = np.array([0])
    r = _resistance_core(csr, node_index, entry, rep_xy, ground_idx, CAP)
    assert leg != 0.0
    assert r == pytest.approx(leg, abs=1e-9)


def test_intensive_mean() -> None:
    # Two parcels: entry a at distance 1 from ground (R_a == 1.0), entry b at distance 2
    # (R_b == 2.0) -- disjoint direct spokes off ground, so mean == 1.5.
    g, a, b = (0.0, 0.0), (1.0, 0.0), (2.0, 0.0)
    csr, node_index = _csr([g, a, b], [(0, 1, 1.0), (0, 2, 2.0)])
    entry: list[_Node | None] = [a, b]
    rep_xy = np.array([[1.0, 0.0], [2.0, 0.0]])
    ground_idx = np.array([0])
    baseline = _resistance_core(csr, node_index, entry, rep_xy, ground_idx, CAP)
    assert baseline == pytest.approx(1.5, abs=1e-9)

    # Add an isolated, ungrounded, parcel-unreferenced node z to the graph. It forms its own
    # trivial (unreached) component and no parcel's entry ever names it -- the INTENSIVE mean
    # over the SAME 2 parcels must be exactly unchanged.
    z = (99.0, 99.0)
    csr2, node_index2 = _csr([g, a, b, z], [(0, 1, 1.0), (0, 2, 2.0)])
    augmented = _resistance_core(csr2, node_index2, entry, rep_xy, ground_idx, CAP)
    assert augmented == pytest.approx(baseline, abs=1e-9)


def test_tree_chain_resistance_equals_shortest_path() -> None:
    # A loop-free multi-hop chain ground(g)--a--b--c, unit edges: resistance distance on a TREE
    # equals shortest-path length, so R_c == 1 + 1 + 1 == 3.0 (not < 3.0, as a loop would give).
    g, a, b, c = (0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)
    csr, node_index = _csr([g, a, b, c], [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
    entry: list[_Node | None] = [c]
    rep_xy = np.array([[3.0, 0.0]])
    ground_idx = np.array([0])
    r = _resistance_core(csr, node_index, entry, rep_xy, ground_idx, CAP)
    assert r == pytest.approx(3.0, abs=1e-9)


def test_zero_conductance_edge_cannot_ground_a_component() -> None:
    # g (ground) --0.0-- mid --1.0-- a (free); the ONLY path from a to ground is a ZERO-length
    # edge (zero conductance). Structural (raw-CSR) connectivity would say {mid, a} IS grounded,
    # handing `factorized` a singular reduced Laplacian (no positive-conductance path to ground)
    # -- reach must instead come from the POSITIVE-conductance graph, so {mid, a} is correctly
    # UNREACHABLE and every parcel there degrades gracefully to `cap`, no crash.
    g, mid, a = (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)
    csr, node_index = _csr([g, mid, a], [(0, 1, 0.0), (1, 2, 1.0)])
    entry: list[_Node | None] = [a]
    rep_xy = np.array([[1.0, 1.0]])
    ground_idx = np.array([0])
    r = _resistance_core(csr, node_index, entry, rep_xy, ground_idx, CAP)
    assert r == pytest.approx(CAP, abs=1e-9)


def test_tree_equals_shortest_path() -> None:
    # `_deep_column_block` + a single straight spur up its centerline is an unbranched CHAIN with
    # exactly ONE ground node -- unlike a coalescing dijkstra tree (verified separately: a real
    # DijkstraReblocker output IS a structural tree, but grounding MULTIPLE street-adjacent nodes
    # of that tree ties separate branches to a shared potential, which is a genuine PARALLEL path
    # for any node with two branches reaching different ground points, so R_i < shortest-path
    # there -- the metric working as designed, not a bug). Here the spur's own base and the
    # street's two corners are all "ground" by `_ground_indices`'s tol-proximity rule, but the
    # spur chain is graph-DISCONNECTED from the street edge (nothing splits it there) -- so each
    # entry's own connected component has exactly ONE reachable ground node (checked below), no
    # branching, and grounded resistance must collapse exactly to shortest-path distance --
    # exercised through the REAL `_BlockScoringContext` machinery (`_derive_entries`,
    # `_ground_indices`, `_build_csr` with colinear splits), not a hand-built CSR (that's task 1's
    # `test_tree_chain_resistance_...`).
    block = _deep_column_block(5)
    roads = gpd.GeoDataFrame(geometry=[LineString([(5.0, 0.0), (5.0, 50.0)])], crs=UTM)

    ctx = _BlockScoringContext(block)
    entry, splits, edge_pairs = ctx._derive_entries(roads)
    assert edge_pairs

    prefix_segs = _explode_segments(roads.geometry)
    base_pairs = [*prefix_segs, *ctx.street_segs]
    csr, node_index = _build_csr(base_pairs, splits)
    ground_idx = ctx._ground_indices(node_index)

    _n_comp, labels = connected_components(csr, directed=False)
    ground_labels = labels[ground_idx]

    dist_to_ground = dijkstra(csr, directed=False, indices=ground_idx, min_only=True)
    assert ctx.n == 5
    for i in range(ctx.n):
        e = entry[i]
        assert e is not None                      # every parcel reaches the graph on this block
        gi = node_index[e]
        # the premise: this entry's own component reaches exactly ONE ground node -- no
        # parallel-path shortcut is possible.
        assert int(np.sum(ground_labels == labels[gi])) == 1
        leg = float(np.hypot(ctx.rep_xy[i, 0] - e[0], ctx.rep_xy[i, 1] - e[1]))
        expected = float(dist_to_ground[gi]) + leg
        assert expected == pytest.approx(5.0 + 10.0 * i, abs=1e-9)   # sanity: matches hand math
        r_i = _resistance_core(csr, node_index, [e], ctx.rep_xy[i:i + 1], ground_idx, ctx.cap)
        assert r_i == pytest.approx(expected, abs=1e-6)


def test_benefit_monotone_and_zero_at_empty() -> None:
    # Mirrors test_budget.py::test_efficiency_and_directness_are_monotone_across_the_full_curve:
    # entries frozen against the FULL road set (Rayleigh monotonicity -- adding edges only lowers
    # resistances), so resistance_benefit must be non-decreasing across cost_benefit_curve's
    # drainage-ordered prefixes, for every method's road layout, starting at 0 for the empty prefix.
    block = _grid_block(5)
    for method in (DijkstraReblocker(), PeelReblocker(), MeshReblocker()):
        roads = method.propose(block).roads
        assert roads is not None
        curve = cost_benefit_curve(block, roads, benefit_fn=resistance_benefit, n_points=20)
        assert curve.benefit[0] == 0.0
        assert all(np.isfinite(b) for b in curve.benefit)
        assert curve.benefit == sorted(curve.benefit), (
            f"{type(method).__name__}: resistance benefit not monotone: {curve.benefit}"
        )


def test_benefit_in_unit_range() -> None:
    # R only drops (or holds) as roads are added -- never rises above the no-roads baseline -- so
    # benefit = (R0 - R(prefix)) / R0 is bounded in [0, 1].
    block = _grid_block(5)
    roads = DijkstraReblocker().propose(block).roads
    assert roads is not None
    f = resistance_benefit(block, roads)
    assert f(None) == pytest.approx(0.0, abs=1e-12)
    val = f(roads)
    assert 0.0 <= val <= 1.0


def test_resistance_benefit_degenerates_with_fewer_than_two_parcels() -> None:
    # Matches _efficiency_factory's degenerate case: < 2 parcels (or no edges at all) -> a
    # constant-0.0 function, no crash.
    block = _grid_block(1)
    f = resistance_benefit(block, None)
    assert f(None) == 0.0
