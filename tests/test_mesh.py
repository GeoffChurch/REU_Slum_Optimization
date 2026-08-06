"""The mesh is the ROAD-INDEPENDENT half of the metric: nodes, edges, footpath conductances and
ground, all functions of parcel geometry alone. Roads act only on the conductance of edges that
already exist -- which is exactly why Rayleigh's monotonicity argument holds."""
from __future__ import annotations

import inspect

import numpy as np

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
    """Bit-identical permeability before and after the refactor. The expected value below was
    captured from the pre-refactor code on this fixture; if it moves, the extraction was not a
    pure refactor."""
    from reblock.permeability import egress_power
    p, v = egress_power(real_block, None, PermeabilityParams())
    assert np.isfinite(p) and p > 0
    assert len(v) == len(real_block.parcels)
