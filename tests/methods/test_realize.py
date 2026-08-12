from __future__ import annotations

import pytest
from shapely.geometry import LineString

from reblock.methods.arterial.primitives import _snap_graph, _SnapGraph
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


def test_snap_to_boundary_realize_forwards_self_lam(monkeypatch: pytest.MonkeyPatch) -> None:
    """SnapToBoundary.realize forwards self.lam to _snap, not a hardcoded value.

    Uses a spy on _snap to verify the exact lam argument received. This test FAILS
    if realize() hardcodes lam instead of using self.lam, catching the mutation immediately.
    """
    from reblock.methods.arterial import realize as realize_module

    block = _grid_block(5)
    g = _boundary_graph(block.parcels)
    sg = _snap_graph(g)
    chord = LineString([(0.0, 2.5), (5.0, 2.5)])

    # Spy on _snap to capture the lam argument it receives
    received_lams: list[float] = []

    original_snap = realize_module._snap

    def spy_snap(chord_arg: LineString, sg_arg: _SnapGraph, lam_arg: float) -> LineString | None:
        """Spy that records lam before delegating to original _snap."""
        received_lams.append(lam_arg)
        return original_snap(chord_arg, sg_arg, lam_arg)

    monkeypatch.setattr(realize_module, "_snap", spy_snap)

    # Call realize with a specific lam value
    target_lam = 7.5
    realizer = SnapToBoundary(lam=target_lam)
    result = realizer.realize(chord, sg)

    # Verify _snap was called
    assert result is not None
    assert len(received_lams) == 1, f"Expected 1 call to _snap, got {len(received_lams)}"

    # This assertion FAILS if realize hardcodes lam instead of using self.lam
    assert received_lams[0] == target_lam, (
        f"realize() passed lam={received_lams[0]} to _snap, "
        f"but self.lam was {target_lam}. "
        f"If this fails, realize() may be hardcoding lam."
    )


def test_snap_to_boundary_realize_requires_snap_graph() -> None:
    """SnapToBoundary.realize must have a snap graph; sg=None raises AssertionError."""
    chord = LineString([(0.0, 2.5), (5.0, 2.5)])
    realizer = SnapToBoundary(lam=2.0)

    with pytest.raises(AssertionError, match="SnapToBoundary needs a snap graph"):
        realizer.realize(chord, sg=None)
