"""The arterial's anchor samplers (`reblock.methods.arterial.anchors`): `HardCap` holds its cap
where `PerLineFallback`'s does not, all three agree when the uncapped family fits, and each is its
own identity. The fault (`anchors._global_samples` replaced by the per-line samples) is shown to
break the cap."""
from __future__ import annotations

from dataclasses import replace

import pytest
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from reblock.methods.arterial import (
    AnchorSampler,
    HardCap,
    PerLineFallback,
    Uncapped,
    anchors,
)
from tests.methods.test_arterial import ARTERIAL, _grid_block


def _cap_violations(sampler: AnchorSampler, networks: list[list[BaseGeometry]]) -> list[str]:
    """Every network on which the sampler returns more than its cap, or points that are unsorted
    or repeated."""
    assert isinstance(sampler, HardCap | PerLineFallback)
    bad = []
    for k, net in enumerate(networks):
        pts = sampler.points(net)
        if len(pts) > sampler.max_anchors:
            bad.append(f"network {k}: {len(pts)} anchors over a cap of {sampler.max_anchors}")
        if pts != sorted(set(pts)):
            bad.append(f"network {k}: points unsorted or repeated")
    return bad


def _many_lines() -> list[BaseGeometry]:
    """60 disjoint 3 m segments plus one 200 m line: the network of many short lines on which a
    per-line restart exceeds its cap."""
    short: list[BaseGeometry] = [LineString([(10.0 * i, 50.0), (10.0 * i + 3.0, 50.0)])
                                 for i in range(60)]
    return [*short, LineString([(0.0, 0.0), (200.0, 0.0)])]


def _zigzag() -> list[BaseGeometry]:
    return [LineString([(float(i), 0.0 if i % 2 == 0 else 1.0) for i in range(40)])]


def test_hard_cap_never_exceeds_max_anchors() -> None:
    networks = [_many_lines(), _zigzag()]
    assert _cap_violations(HardCap(n=32, max_anchors=16), networks) == []
    # The measured bug, pinned as a fact of the kept sampler: one extra sample per line.
    assert len(PerLineFallback(n=32, max_anchors=16).points(_many_lines())) > 16


def test_hard_cap_guard_sees_per_line_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(anchors, "_global_samples", anchors._arclength_samples)
    assert _cap_violations(HardCap(n=32, max_anchors=16), [_many_lines(), _zigzag()])


def test_hard_cap_is_the_uncapped_family_when_it_fits() -> None:
    net: list[BaseGeometry] = [LineString([(0.0, 0.0), (4.0, 0.0), (4.0, 3.0)])]
    full = Uncapped(8).points(net)
    assert len(full) < 64
    assert HardCap(8, 64).points(net) == full
    assert PerLineFallback(8, 64).points(net) == full


def test_hard_cap_spreads_by_arc_length() -> None:
    net: list[BaseGeometry] = [LineString([(0.0, 0.0), (100.0, 0.0)])]
    assert len(Uncapped(32).points(net)) > 10
    assert HardCap(n=32, max_anchors=10).points(net) == [(5.0 + 10.0 * j, 0.0) for j in range(10)]


def test_capped_samplers_refuse_a_zero_cap() -> None:
    with pytest.raises(ValueError):
        HardCap(32, 0)
    with pytest.raises(ValueError):
        PerLineFallback(32, 0)
    with pytest.raises(ValueError):
        Uncapped(-1)


def test_samplers_split_the_identity() -> None:
    samplers: list[AnchorSampler] = [Uncapped(32), PerLineFallback(32, 128), HardCap(32, 128)]
    assert len({s.identity for s in samplers}) == 3
    block = _grid_block(3)
    methods = [replace(ARTERIAL, anchors=s, max_roads=1, workers=1) for s in samplers]
    assert len({m.identity for m in methods}) == 3
    assert len({m.propose(block).proposal_id for m in methods}) == 3


# --- moved from tests/methods/test_arterial.py, as the samplers they now are ---

def test_anchor_points_sample_the_network_and_include_vertices() -> None:
    net: list[BaseGeometry] = [LineString([(0.0, 0.0), (4.0, 0.0)])]
    pts = Uncapped(4).points(net)
    assert (0.0, 0.0) in pts and (4.0, 0.0) in pts        # endpoints/vertices kept
    assert len(pts) >= 4 and pts == sorted(pts)           # sampled + deterministic order


def test_per_line_fallback_caps_a_many_vertex_network() -> None:
    # A network with many vertices (mimics a large block's boundary graph, e.g. block 5810's ~229
    # vertices) so the cap actually bounds the count instead of coincidentally landing on it.
    net = _zigzag()
    uncapped = Uncapped(8).points(net)
    capped = PerLineFallback(8, 8).points(net)
    assert len(capped) <= 15
    assert len(capped) < len(uncapped)


def test_per_line_fallback_never_pessimises() -> None:
    """A cap must never produce MORE anchors than uncapped. An earlier `max_anchors > 0` REPLACED
    the per-vertex family with arc-length samples, so a cap above the vertex count inflated the set
    -- measured 1.69x wall clock at block scale. The name promises a maximum; this makes it one."""
    net = _zigzag()
    uncapped = Uncapped(8).points(net)
    for cap in (4, 8, 16, 32, 64, 128, 256):
        got = PerLineFallback(8, cap).points(net)
        assert len(got) <= len(uncapped), (
            f"max_anchors={cap} produced {len(got)} anchors, more than uncapped's {len(uncapped)}")


def test_a_cap_above_the_anchor_count_is_a_no_op() -> None:
    """A cap the network never reaches must be exactly uncapped -- not a different anchor family."""
    net = _zigzag()
    uncapped = Uncapped(8).points(net)
    assert PerLineFallback(8, 10_000).points(net) == uncapped
    assert HardCap(8, 10_000).points(net) == uncapped
