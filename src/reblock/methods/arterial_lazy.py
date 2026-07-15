"""CELF / lazy-greedy engine + candidate policies for GreedyArterialReblocker. Reuses arterial's
exact scoring machinery unchanged; only changes which candidates get scored each step."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from reblock.contracts import Block
from reblock.derive.adjacency import parcel_adjacency  # noqa: F401 (kept for callers/tests)
from reblock.methods.arterial import _anchor_points, _candidate_chords, _deep_targets, _xy
from reblock.methods.dijkstra import _rnd


def _road_vertices(road: LineString) -> list[tuple[float, float]]:
    return [_rnd(_xy(c)) for c in road.coords]


def _committed_gdf(committed: list[LineString], block: Block):
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

    def initial(self) -> list[LineString]:
        return self._initial

    def after_commit(self, committed: list[LineString], step: int
                     ) -> tuple[list[LineString], list[str]]:
        network = [*self.streets, *committed]
        cands = _candidate_chords(
            _anchor_points(network, self.n_anchors),
            _deep_targets(self.block, _committed_gdf(committed, self.block), self.top_k, self.adj))
        now = {ls.wkt: ls for ls in cands}
        added = [ls for k, ls in now.items() if k not in self.live]
        removed = [k for k in self.live if k not in now]
        self.live = set(now.keys())
        return added, removed


def _make_policy(name: str, block: Block, streets: Sequence[BaseGeometry],
                 n_anchors: int, top_k: int, adj: list[set[int]]):
    anchors0 = _anchor_points(list(streets), n_anchors)
    targets0 = _deep_targets(block, None, top_k, adj)
    initial = _candidate_chords(anchors0, targets0)
    if name == "fixed":
        return _FixedPolicy(initial)
    if name == "grow":
        return _GrowPolicy(block, adj, list(anchors0), top_k, {ls.wkt for ls in initial}, initial)
    if name == "faithful":
        return _FaithfulPolicy(block, list(streets), n_anchors, adj, top_k,
                               {ls.wkt for ls in initial}, initial)
    raise ValueError(f"unknown candidate_policy {name!r}")
