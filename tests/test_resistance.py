"""Unit tests for `_resistance_core`, the grounded-Laplacian numeric core of the resistance
metric lens (see docs/superpowers/specs/2026-07-11-resistance-eval-design.md, "The metric" +
Design.1). Tiny hand-built CSRs with known analytic effective resistances -- no
`_BlockScoringContext` wiring, no shapely/geopandas, per task 1's scope."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from reblock.budget import _Node, _resistance_core

CAP = 1000.0


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
