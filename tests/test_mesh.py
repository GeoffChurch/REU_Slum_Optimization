"""The mesh is the ROAD-INDEPENDENT half of the metric: nodes, edges, footpath conductances and
ground, all functions of parcel geometry alone. Roads act only on the conductance of edges that
already exist -- which is exactly why Rayleigh's monotonicity argument holds."""
from __future__ import annotations

import inspect

import pytest

from reblock.mesh import Mesh, footpath_mesh
from reblock.permeability import PermeabilityParams


def test_footpath_mesh_takes_no_roads():
    """The property the whole monotonicity argument rests on."""
    assert "roads" not in inspect.signature(footpath_mesh).parameters


def test_mesh_arrays_are_consistent(real_block):
    m = footpath_mesh(real_block, PermeabilityParams())
    assert isinstance(m, Mesh)
    assert m.n == len(real_block.parcels)
    assert len(m.cx) == len(m.cy) == m.n
    for arr in (m.cols, m.dist, m.footpath_g, m.segments):
        assert len(arr) == len(m.rows)
    assert (m.rows < m.cols).all(), "each undirected edge stored once, low index first"
    assert (m.dist > 0).all()
    assert len(m.ground) == m.n


def test_extraction_changed_nothing(real_block):
    """Pins `egress_power`'s no-roads value on `real_block` to the number the CURRENT
    (post-extraction) implementation produces, at rel=1e-12 -- tight enough that this must stay
    exact, not merely close. This is a regression guard GOING FORWARD: any later change to
    `footpath_mesh` (or what `egress_power` does with it) that silently moves this number now has
    to touch this assertion, which forces it to be a deliberate, reviewed change rather than a
    silent drift.

    It does NOT retroactively prove the mesh extraction itself was pure -- the value was captured
    from the already-refactored code, not a pre-refactor snapshot, so this cannot by itself
    distinguish "extraction preserved the computation" from "extraction changed it and this pins
    the new number." That property -- the actual purity claim for this task -- was established
    separately, by running the full pre-existing test suite (592 tests, including several
    hand-computed values in tests/test_permeability.py and tests/test_permeability_width.py)
    against the extracted code and confirming every one passed with zero expectations edited (see
    task-1-report.md).
    """
    from reblock.permeability import egress_power
    p, v = egress_power(real_block, None, PermeabilityParams())
    assert p == pytest.approx(28549.99999999984, rel=1e-12)
    assert len(v) == len(real_block.parcels)
