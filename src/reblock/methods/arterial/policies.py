"""Candidate policies for the lazy (CELF) engine: which chords are live at each step.
`_FixedPolicy` never adds candidates after the initial set, `_GrowPolicy` only adds (monotonic,
matches CELF's lazy re-scoring), and `_FaithfulPolicy` regenerates arterial's own candidate set
exactly (add/remove) -- byte-identical to the exact greedy when paired with `rescore_every=1`.

`Fixed`/`Grow`/`Faithful` are the CONFIGURABLE specs -- injected rather than selected by a
`candidate_policy` string, so nothing downstream asks which policy it has. Each closes over
nothing (frozen, no fields); the block-specific state (block, adjacency, seed candidates) is
per-proposal and lives on the `CandidatePolicy` instance `build()` returns, not on the spec.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from geopandas import GeoDataFrame
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from reblock.contracts import Block
from reblock.methods.arterial.primitives import (
    _anchor_points,
    _candidate_chords,
    _deep_targets,
    _xy,
)
from reblock.methods.boundary_graph import _rnd


def _road_vertices(road: LineString) -> list[tuple[float, float]]:
    return [_rnd(_xy(c)) for c in road.coords]


def _committed_gdf(committed: list[LineString], block: Block) -> GeoDataFrame | None:
    import geopandas as gpd
    return gpd.GeoDataFrame(geometry=list(committed), crs=block.crs) if committed else None


@dataclass
class _FixedPolicy:
    _initial: list[LineString]

    def initial(self) -> list[LineString]:
        return self._initial

    def after_commit(self, committed: list[LineString], step: int
                     ) -> tuple[list[LineString], list[str]]:
        return [], []


@dataclass
class _GrowPolicy:
    block: Block
    adj: list[set[int]]
    anchors: list[tuple[float, float]]         # accumulates committed-road vertices
    top_k: int
    seen: set[str]                              # wkt of every candidate ever emitted
    _initial: list[LineString]

    def initial(self) -> list[LineString]:
        return self._initial

    def after_commit(self, committed: list[LineString], step: int
                     ) -> tuple[list[LineString], list[str]]:
        for v in _road_vertices(committed[-1]):
            if v not in self.anchors:
                self.anchors.append(v)
        self.anchors.sort()
        targets = _deep_targets(self.block, _committed_gdf(committed, self.block),
                                self.top_k, self.adj)
        cands = _candidate_chords(self.anchors, targets)
        added = [ls for ls in cands if ls.wkt not in self.seen]
        for ls in added:
            self.seen.add(ls.wkt)
        return added, []


@dataclass
class _FaithfulPolicy:
    block: Block
    streets: list[BaseGeometry]
    n_anchors: int
    adj: list[set[int]]
    top_k: int
    live: set[str]
    _initial: list[LineString]
    max_anchors: int = 0

    def initial(self) -> list[LineString]:
        return self._initial

    def after_commit(self, committed: list[LineString], step: int
                     ) -> tuple[list[LineString], list[str]]:
        network = [*self.streets, *committed]
        cands = _candidate_chords(
            _anchor_points(network, self.n_anchors, self.max_anchors),
            _deep_targets(self.block, _committed_gdf(committed, self.block), self.top_k, self.adj))
        now = {ls.wkt: ls for ls in cands}
        added = [ls for k, ls in now.items() if k not in self.live]
        removed = [k for k in self.live if k not in now]
        self.live = set(now.keys())
        return added, removed


@runtime_checkable
class CandidatePolicy(Protocol):
    """Which candidates the lazy engine keeps alive as roads commit. Stateful, per block."""

    def initial(self) -> list[LineString]: ...
    def after_commit(self, committed: list[LineString],
                     step: int) -> tuple[list[LineString], list[str]]: ...


@runtime_checkable
class CandidatePolicySpec(Protocol):
    """The CONFIGURABLE half of a policy. The policies themselves close over block state (block,
    adjacency, seed candidates), which is per-proposal and cannot be built where config is read --
    so config injects a spec and the engine calls `build` once per block."""

    def build(self, block: Block, streets: list[BaseGeometry], n_anchors: int, top_k: int,
              adj: list[set[int]], max_anchors: int) -> CandidatePolicy: ...


def _seed(block: Block, streets: list[BaseGeometry], n_anchors: int, top_k: int,
          adj: list[set[int]], max_anchors: int
          ) -> tuple[list[tuple[float, float]], list[LineString]]:
    anchors0 = _anchor_points(streets, n_anchors, max_anchors)
    targets0 = _deep_targets(block, None, top_k, adj)
    return anchors0, _candidate_chords(anchors0, targets0)


@dataclass(frozen=True)
class Fixed:
    """Score only the step-0 candidate set forever. Cheapest, and blind to continuations."""

    def build(self, block: Block, streets: list[BaseGeometry], n_anchors: int, top_k: int,
              adj: list[set[int]], max_anchors: int) -> CandidatePolicy:
        _, initial = _seed(block, streets, n_anchors, top_k, adj, max_anchors)
        return _FixedPolicy(initial)


@dataclass(frozen=True)
class Grow:
    """Add continuations from each committed road's vertices as they appear. The shipped default."""

    def build(self, block: Block, streets: list[BaseGeometry], n_anchors: int, top_k: int,
              adj: list[set[int]], max_anchors: int) -> CandidatePolicy:
        anchors0, initial = _seed(block, streets, n_anchors, top_k, adj, max_anchors)
        return _GrowPolicy(block, adj, list(anchors0), top_k,
                           {ls.wkt for ls in initial}, initial)


@dataclass(frozen=True)
class Faithful:
    """Regenerate the exact greedy's candidate set every step. With rescore_every=1 this makes the
    lazy engine byte-identical to the exact one -- the oracle the lazy path is checked against."""

    def build(self, block: Block, streets: list[BaseGeometry], n_anchors: int, top_k: int,
              adj: list[set[int]], max_anchors: int) -> CandidatePolicy:
        _, initial = _seed(block, streets, n_anchors, top_k, adj, max_anchors)
        return _FaithfulPolicy(block, list(streets), n_anchors, adj, top_k,
                               {ls.wkt for ls in initial}, initial, max_anchors)
