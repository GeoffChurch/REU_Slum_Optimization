"""Where the arterial greedy's candidate chords start and end: an injected `AnchorSampler` --
`Uncapped`, `PerLineFallback` or `HardCap` -- in place of an `n_anchors`/`max_anchors` pair whose
`max_anchors: 0` meant "uncapped".

All three start from the same uncapped family: every network vertex (so committed-road endpoints
are anchors and continuations come for free) plus `n` arc-length samples. They differ only when
that family is larger than a cap:

- `Uncapped(n)` never caps (the directness presets).
- `PerLineFallback(n, max_anchors)` falls back to arc-length samples restarted on every line, which
  yields ~`max_anchors` plus one per line: on a network of hundreds of short lines the "cap" does
  not hold (measured on ZAF.9.3.1_1_20269 at a cap of 128: 112, 229, 323, 409 and 525 anchors at
  0, 150, 300, 450 and 607 committed segments).
- `HardCap(n, max_anchors)` falls back to at most `max_anchors` samples at one arc-length stride
  over the whole merged network, so the cap holds however many lines the network has.

Each sampler is a frozen dataclass and its own cache identity.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import unary_union

from reblock.methods.arterial.primitives import _xy
from reblock.methods.boundary_graph import _rnd


def _merged_lines(network: Sequence[BaseGeometry]) -> tuple[list[BaseGeometry], float]:
    """The network as a flat line list plus its total length. `unary_union` explodes any Multi*
    input, so streets given as a MultiLineString (a block with a hole/courtyard) are handled."""
    merged = unary_union(network)
    lines = list(merged.geoms) if isinstance(merged, BaseMultipartGeometry) else [merged]
    return lines, sum(ln.length for ln in lines)


def _arclength_samples(lines: Sequence[BaseGeometry], total: float,
                       count: int) -> list[tuple[float, float]]:
    """`count` points spaced evenly by arc-length along the network, the walk restarted at 0 on
    every line. `_rnd`-snapped, de-duplicated, sorted for determinism. Yields ~`count` + one per
    line, not exactly `count`."""
    pts: set[tuple[float, float]] = set()
    if total > 0 and count > 0:
        step = total / count
        for ln in lines:
            d = 0.0
            while d <= ln.length:
                pts.add(_rnd(_xy(ln.interpolate(d).coords[0])))
                d += step
    return sorted(pts)


def _global_samples(lines: Sequence[BaseGeometry], total: float,
                    count: int) -> list[tuple[float, float]]:
    """At most `count` points at one arc-length stride over the whole network: positions
    `(j + 0.5) * total / count`, walked along `lines` in order, `_rnd`-snapped, de-duplicated and
    sorted. Never more than `count`, however many lines there are."""
    pts: set[tuple[float, float]] = set()
    if total > 0 and count > 0:
        stride = total / count
        j, start = 0, 0.0
        for ln in lines:
            end = start + ln.length
            while j < count and (j + 0.5) * stride <= end:
                pts.add(_rnd(_xy(ln.interpolate((j + 0.5) * stride - start).coords[0])))
                j += 1
            start = end
        # The last line ends at `total` exactly (the same sum, in the same order), half a stride
        # past the last position, so every position was placed.
        assert j == count, (j, count, start, total)
    return sorted(pts)


def _vertices_and_samples(lines: Sequence[BaseGeometry], total: float,
                          n: int) -> list[tuple[float, float]]:
    """Every network vertex plus `n` arc-length samples -- the uncapped anchor family. Vertices are
    what make committed-segment endpoints anchors, so continuations come for free."""
    pts: set[tuple[float, float]] = {_rnd(_xy(c)) for ln in lines for c in ln.coords}
    pts.update(_arclength_samples(lines, total, n))
    return sorted(pts)


@runtime_checkable
class AnchorSampler(Protocol):
    """The anchors a greedy step's candidate chords join: sorted, unique, deterministic."""

    @property
    def identity(self) -> AnchorIdentity: ...

    def points(self, network: Sequence[BaseGeometry]) -> list[tuple[float, float]]: ...


def _check_n(n: int) -> None:
    if n < 0:
        raise ValueError(f"n is a count of arc-length samples, got {n}")


def _check_cap(max_anchors: int) -> None:
    if max_anchors < 1:
        raise ValueError(f"max_anchors is a cap of at least 1 (Uncapped never caps), got "
                         f"{max_anchors}")


@dataclass(frozen=True)
class Uncapped:
    """Every network vertex plus `n` arc-length samples, however many that is."""
    n: int

    def __post_init__(self) -> None:
        _check_n(self.n)

    @property
    def identity(self) -> AnchorIdentity:
        return self

    def points(self, network: Sequence[BaseGeometry]) -> list[tuple[float, float]]:
        lines, total = _merged_lines(network)
        return _vertices_and_samples(lines, total, self.n)


@dataclass(frozen=True)
class PerLineFallback:
    """The uncapped family if it fits under `max_anchors`; otherwise ~`max_anchors` arc-length
    samples restarted on every line, whichever set is smaller.

    The cap can never inflate the anchor count -- which an earlier implementation did, replacing
    the vertex family outright and yielding 129 anchors where uncapped gave 39 (measured 1.69x wall
    clock at block scale). When it binds the result is the sampled family, exactly as measured in
    notes/2026-08-11-max-anchors-is-a-region-scale-win.md. It does NOT hold as a cap on a network
    of many lines (one extra sample per line: 525 anchors at a cap of 128 on ZAF.9.3.1_1_20269);
    `HardCap` is the sampler whose cap holds."""
    n: int
    max_anchors: int

    def __post_init__(self) -> None:
        _check_n(self.n)
        _check_cap(self.max_anchors)

    @property
    def identity(self) -> AnchorIdentity:
        return self

    def points(self, network: Sequence[BaseGeometry]) -> list[tuple[float, float]]:
        lines, total = _merged_lines(network)
        full = _vertices_and_samples(lines, total, self.n)
        if len(full) <= self.max_anchors:
            return full
        sampled = _arclength_samples(lines, total, self.max_anchors)
        return sampled if len(sampled) < len(full) else full


@dataclass(frozen=True)
class HardCap:
    """The uncapped family if it fits under `max_anchors`; otherwise at most `max_anchors` points
    at one arc-length stride over the merged network (`_global_samples`). The cap holds on any
    network; what it gives up when it binds is the vertex family (committed-road junctions are
    anchors only where a sample lands on them)."""
    n: int
    max_anchors: int

    def __post_init__(self) -> None:
        _check_n(self.n)
        _check_cap(self.max_anchors)

    @property
    def identity(self) -> AnchorIdentity:
        return self

    def points(self, network: Sequence[BaseGeometry]) -> list[tuple[float, float]]:
        lines, total = _merged_lines(network)
        full = _vertices_and_samples(lines, total, self.n)
        if len(full) <= self.max_anchors:
            return full
        return _global_samples(lines, total, self.max_anchors)


AnchorIdentity: TypeAlias = Uncapped | PerLineFallback | HardCap
