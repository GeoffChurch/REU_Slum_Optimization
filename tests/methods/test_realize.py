from __future__ import annotations

from shapely.geometry import LineString

from reblock.methods.arterial.realize import ChordRealizer, IdealChord, SnapToBoundary


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
    realizers: list[ChordRealizer] = [SnapToBoundary(), IdealChord()]
    assert len(realizers) == 2
