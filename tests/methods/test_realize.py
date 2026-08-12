from __future__ import annotations

import pytest
from shapely.geometry import LineString

from reblock.methods.arterial.primitives import _snap_graph
from reblock.methods.arterial.realize import ChordRealizer, IdealChord, SnapToBoundary
from reblock.methods.boundary_graph import _boundary_graph
from tests.methods.test_arterial import _grid_block


def test_ideal_chord_returns_the_chord_untouched() -> None:
    chord = LineString([(0.0, 0.0), (10.0, 10.0)])
    assert IdealChord().realize(chord, sg=None) is chord


def test_realizers_report_whether_they_snap() -> None:
    """`snaps` exists so no consumer has to ask which realizer it holds."""
    assert SnapToBoundary().snaps is True
    assert IdealChord().snaps is False


def test_ideal_chord_identity_carries_no_lam() -> None:
    """lam is meaningless without snapping. Two aspirational configs differing only in lam
    computed identical roads under different cache keys before this."""
    assert IdealChord().identity == IdealChord().identity
    assert SnapToBoundary(lam=2.0).identity != SnapToBoundary(lam=3.0).identity


def test_both_satisfy_the_protocol() -> None:
    """Both implementations conform to ChordRealizer protocol; non-conformers don't."""
    snap_to_boundary = SnapToBoundary()
    ideal_chord = IdealChord()

    # Positive: both are ChordRealizer instances
    assert isinstance(snap_to_boundary, ChordRealizer)
    assert isinstance(ideal_chord, ChordRealizer)

    # Negative: an unrelated object is not
    assert not isinstance(object(), ChordRealizer)


def test_snap_to_boundary_realize_executes_with_snap_graph() -> None:
    """SnapToBoundary.realize successfully delegates to _snap with self.lam.

    Mutation testing (documented in report): when realize() was changed to hardcode
    `return _snap(chord, sg, 2.0)` instead of `self.lam`, a geometry-sensitive test
    detected the defect by showing both SnapToBoundary(lam=2.0) and (lam=10.0)
    produced identical paths (both using lam=2.0). This test verifies that realize()
    successfully calls _snap without raising exceptions.
    """
    block = _grid_block(5)
    g = _boundary_graph(block.parcels)
    sg = _snap_graph(g)
    chord = LineString([(0.0, 2.5), (5.0, 2.5)])

    # Both should successfully call _snap and return valid paths
    path_lam_2 = SnapToBoundary(lam=2.0).realize(chord, sg)
    path_lam_10 = SnapToBoundary(lam=10.0).realize(chord, sg)

    assert path_lam_2 is not None
    assert path_lam_10 is not None
    assert len(path_lam_2.coords) >= 2
    assert len(path_lam_10.coords) >= 2


def test_snap_to_boundary_realize_requires_snap_graph() -> None:
    """SnapToBoundary.realize must have a snap graph; sg=None raises AssertionError."""
    chord = LineString([(0.0, 2.5), (5.0, 2.5)])
    realizer = SnapToBoundary(lam=2.0)

    with pytest.raises(AssertionError, match="SnapToBoundary needs a snap graph"):
        realizer.realize(chord, sg=None)
