"""The arterial's access gain scored incrementally (`objectives._AccessStep`) against the full
re-peel it replaces: exact on every candidate of real and fixture blocks, with the three faults
(`always-live`, `no-min`, `streets-only`) each shown to break the equivalence.

The pinned proposals (tests/data/scoring/arterial_access_ref.json) were captured before the
incremental gain existed: the 1808 block (one road) as the plan names it, and DJI.1_2_602 (15
commits, 33 segments), whose depth-4 parcels make the pin bind on more than one step.
"""
from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import pytest
import shapely
from numpy.typing import NDArray
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from reblock.buildings import SpacingDiscs
from reblock.contracts import Block
from reblock.data.kblock import KblockSource
from reblock.derive.access import (
    STREET_TOL,
    ParcelAdjacency,
    parcel_access_layers,
    past_every_parcel,
)
from reblock.methods.arterial import (
    Access,
    Displacement,
    GreedyArterialReblocker,
    PerLineFallback,
    ShortlistEngine,
    SnapToBoundary,
    Uncapped,
    engines,
)
from reblock.methods.arterial.objectives import _AccessStep
from reblock.methods.arterial.primitives import _explode, _snap_graph
from reblock.methods.boundary_graph import _boundary_graph
from reblock.permeability import with_width
from tests.block_fixtures import no_buildings
from tests.methods.test_arterial import (
    UTM,
    _deep_block,
    _holed_block,
    _two_arm_block,
)
from tests.scoring_fixtures import _block_1808

_REPO = Path(__file__).resolve().parents[2]
_REF = json.loads((_REPO / "tests/data/scoring/arterial_access_ref.json").read_text())
HW = 3.5                     # the shipped road's half-width (7 m)
REALIZER = SnapToBoundary(lam=2.0)


def _block_602() -> Block:
    src = KblockSource(_REPO / "tests/data/kblock/blocks_dji_sample.parquet",
                       _REPO / "tests/data/kblock/buildings_dji_sample.parquet", "dji",
                       block_ids=["DJI.1_2_602"], min_buildings=10, building_tier=SpacingDiscs,
                       member_buildings=None)
    return next(iter(src.region().blocks))


def _access_snap() -> GreedyArterialReblocker:
    return GreedyArterialReblocker(
        realizer=REALIZER, objective=Access(), cost=Displacement(),
        anchors=PerLineFallback(32, 128), top_k=8, max_roads=15, road_width_m=7.0, workers=1,
        engine=ShortlistEngine(k=512, threads=1))


def _sorted_wkt(roads: gpd.GeoDataFrame | None) -> list[str]:
    assert roads is not None
    return sorted(g.wkt for g in roads.geometry)


def test_access_snap_proposal_wkt_unchanged() -> None:
    m = _access_snap()
    assert _sorted_wkt(m.propose(_block_1808()).roads) == _REF["access_displacement_snap"]["wkt"]
    assert (_sorted_wkt(m.propose(_block_602()).roads)
            == _REF["access_displacement_snap_602"]["wkt"])


@contextmanager
def _recording() -> Iterator[tuple[list[list[LineString]], list[LineString]]]:
    """Each step's candidate chords (`engines._candidate_chords`) and each commit
    (`engines._best_candidate`'s winner), through the engine module's instrument seams."""
    chords, commits = [], []
    real_chords, real_best = engines._candidate_chords, engines._best_candidate

    def chords_rec(anchors: list[tuple[float, float]],
                   targets: list[tuple[float, float]]) -> list[LineString]:
        out = real_chords(anchors, targets)
        chords.append(out)
        return out

    def best_rec(results: Iterable[tuple[float, BaseGeometry | None]]
                 ) -> tuple[float, BaseGeometry | None]:
        gain, real = real_best(results)
        if real is not None:
            assert isinstance(real, LineString)
            commits.append(real)
        return gain, real

    engines._candidate_chords, engines._best_candidate = chords_rec, best_rec
    try:
        yield chords, commits
    finally:
        engines._candidate_chords, engines._best_candidate = real_chords, real_best


def _mismatches(block: Block, steps: int) -> tuple[list[str], int, int]:
    """The Access + Displacement + SnapToBoundary(2.0) greedy for `steps` commits; at every step,
    every realized candidate scored by the step's `_AccessStep` and by the full re-peel of the
    committed union plus the candidate. Returns the mismatches (naming block, step and chord),
    the count of non-live candidates and the count with gain > 0."""
    with _recording() as (chords, commits):
        engines._greedy_arterials(block, realizer=REALIZER, objective=Access(),
                                  cost=Displacement(), anchors=Uncapped(8), top_k=4,
                                  max_roads=steps,
                                  half_width_m=HW, workers=1)
    scorer = Access().for_block(ParcelAdjacency.of(block, STREET_TOL))
    sg = _snap_graph(_boundary_graph(block.parcels))
    bad: list[str] = []
    dead = gaining = 0
    for step, cands in enumerate(chords):
        net = engines._step_network(commits[:step], block, HW)
        gain = scorer.at_step(net, REALIZER)
        assert isinstance(gain, _AccessStep)
        base = scorer.value(net.roads)
        merged: list[BaseGeometry] = [net.merged] if net.merged is not None else []
        for chord in cands:
            real = REALIZER.realize(chord, sg)
            if real is None or real.length == 0:
                continue
            want = scorer.value(_explode(unary_union([*merged, real]), block.crs, 2 * HW)) - base
            got = gain.of(real)
            dead += not gain.live(real)
            gaining += want > 0
            if got != want:
                bad.append(f"{block.block_id} step {step} {chord.wkt}: {got!r} != {want!r}")
    return bad, dead, gaining


def _verge_block() -> Block:
    """A 3x6 grid fronted on its west edge, with a second street 0.8 m off its east edge (a verge
    wider than STREET_TOL): the east street seeds no parcel, and an anchor on it snaps to an
    east-edge boundary node that no street reaches. A snapped road along the east edge then fronts
    the deepest parcels and is not live -- the fixture none of the others gives."""
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(3) for j in range(6)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0.0, 0.0), (0.0, 6.0)]),
                                         LineString([(3.8, 0.0), (3.8, 6.0)])], crs=UTM)
    boundary = cast(Polygon | MultiPolygon, parcels.geometry.union_all())
    return Block(block_id="verge", crs=UTM, boundary=boundary, parcels=parcels, streets=streets,
                 source_content_hash=None, building_geometries=no_buildings(UTM),
                 building_tier=SpacingDiscs)


def _blocks() -> list[Block]:
    pts = gpd.GeoDataFrame(geometry=[Point(0.5, y) for y in range(1, 8)] + [Point(10.5, 4)],
                           crs=UTM)
    return [_block_1808(), _block_602(), _deep_block(), _holed_block(), _two_arm_block(pts),
            _verge_block()]


def test_access_step_matches_the_full_repeel() -> None:
    bad, gaining = [], 0
    for block in _blocks():
        b, _, g = _mismatches(block, 3)
        bad += b
        gaining += g
    assert not bad, bad[:5]
    assert gaining > 0
    # The non-live branch is exercised: on the verge block (see there) and nowhere else.
    assert _mismatches(_verge_block(), 3)[1] > 0


def _grid_with_stranded_parcel() -> tuple[Block, int]:
    """A 3x3 grid fronted all round, plus one parcel 2 m east of it that touches neither the grid
    nor the street (stranded at n_parcels + 1). Returns the block and that parcel's position."""
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(3) for j in range(3)]
    grid = cast(Polygon, unary_union(polys))
    polys.append(Polygon([(5, 1), (6, 1), (6, 2), (5, 2)]))
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[grid.boundary], crs=UTM)
    boundary = cast(Polygon | MultiPolygon, parcels.geometry.union_all())
    block = Block(block_id="stranded", crs=UTM, boundary=boundary, parcels=parcels,
                  streets=streets, source_content_hash=None,
                  building_geometries=no_buildings(UTM), building_tier=SpacingDiscs)
    return block, len(polys) - 1


def test_access_step_lowers_a_stranded_parcel() -> None:
    block, iso = _grid_with_stranded_parcel()
    adjacency = ParcelAdjacency.of(block, STREET_TOL)
    scorer = Access().for_block(adjacency)
    net = engines._step_network([], block, HW)
    gain = scorer.at_step(net, REALIZER)
    assert isinstance(gain, _AccessStep)
    assert gain.depths[iso] == len(block.parcels) + 1
    road = LineString([(3.0, 1.5), (5.0, 1.5)])         # from the street to the stranded parcel
    roads = with_width(gpd.GeoDataFrame(geometry=[road], crs=UTM), 2 * HW)
    after = parcel_access_layers(adjacency, roads, unreached=past_every_parcel).to_numpy()
    assert after[iso] == 1
    want = scorer.value(roads) - scorer.value(net.roads)
    assert want > 0 and gain.of(road) == want


def _no_min(self: _AccessStep, seeds: NDArray[np.intp]) -> int:
    """Fault `no-min`: the relaxation's BFS assigns `layer + 1` to every unvisited neighbour,
    never comparing to the base depth."""
    depths = self.depths.copy()
    seen = {int(s) for s in seeds}
    frontier = deque(sorted(seen))
    for s in seen:
        depths[s] = 1
    while frontier:
        i = frontier.popleft()
        for j in self.neighbours[i]:
            if j not in seen:
                seen.add(j)
                depths[j] = depths[i] + 1
                frontier.append(j)
    return int((depths ** 2).sum())


def _streets_only(block: Block) -> Callable[[_AccessStep, LineString], bool]:
    """Fault `streets-only`: liveness against the streets' union only, committed roads ignored."""
    streets = shapely.union_all(list(block.streets.geometry))
    return lambda self, real: bool(shapely.dwithin(real, streets, self.tol))


Patch = Callable[[pytest.MonkeyPatch, Block], None]


def _faulted_mismatches(monkeypatch: pytest.MonkeyPatch, patch: Patch) -> list[str]:
    bad: list[str] = []
    for block in _blocks():
        with monkeypatch.context() as m:
            patch(m, block)
            bad += _mismatches(block, 3)[0]
    return bad


def test_access_step_guard_sees_always_live(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _faulted_mismatches(
        monkeypatch, lambda m, block: m.setattr(_AccessStep, "live", lambda self, real: True))


def test_access_step_guard_sees_no_min(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _faulted_mismatches(
        monkeypatch, lambda m, block: m.setattr(_AccessStep, "relaxed_burden", _no_min))


def test_access_step_guard_sees_streets_only(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _faulted_mismatches(
        monkeypatch, lambda m, block: m.setattr(_AccessStep, "live", _streets_only(block)))
