"""The arterial's displacement budget (`reblock.methods.arterial.stop`): the meter's two shares
(the lineup's scored area D, and the footprint area destroyed with a touched building counted
whole), the budget Strategies that read them (NoBudget, ScoredArea, DestroyedArea), the stop
before the step that would start past the budget (so a stopped run is a commit prefix of a longer
one), and every run's stop reason and shares. The faults (a deaf meter; a ScoredArea that reads
the destroyed share) are shown to break the stop."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import replace

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from reblock.budget import road_corridor
from reblock.buildings import Discs
from reblock.contracts import Block
from reblock.methods.arterial import (
    Access,
    DestroyedArea,
    Displacement,
    DisplacementBudget,
    NoBudget,
    ScoredArea,
    Shares,
    SnapToBoundary,
    Uncapped,
    engines,
)
from reblock.methods.arterial.primitives import _planarize
from reblock.methods.arterial.stop import FootprintMeter, GreedyRun
from tests.methods.test_arterial import ARTERIAL, UTM, _grid_block, _grid_block_with_points

HW = 0.25                    # a narrow corridor, so each road touches only its own neighbours


def _dense_block() -> Block:
    """An 8 x 6 unit grid fronted on its bottom edge, a building at every parcel's centre: deep
    enough that the access greedy commits several roads, each touching buildings no earlier one
    did."""
    pts = gpd.GeoDataFrame(geometry=[Point(i + 0.5, j + 0.5) for i in range(8) for j in range(6)],
                           crs=UTM)
    return _grid_block_with_points(pts, w=8, h=6)


@contextmanager
def _commits() -> Iterator[list[LineString]]:
    """Each committed road, through the engine's `_best_candidate` seam."""
    out: list[LineString] = []
    real_best = engines._best_candidate

    def best(results: Iterable[tuple[float, BaseGeometry | None]]
             ) -> tuple[float, BaseGeometry | None]:
        gain, real = real_best(results)
        if real is not None:
            assert isinstance(real, LineString)
            out.append(real)
        return gain, real

    engines._best_candidate = best
    try:
        yield out
    finally:
        engines._best_candidate = real_best


def _run(block: Block, *, budget: DisplacementBudget, max_roads: int) -> GreedyRun:
    return engines._greedy_arterials(
        block, realizer=SnapToBoundary(lam=2.0), objective=Access(), cost=Displacement(),
        anchors=Uncapped(8), top_k=4, max_roads=max_roads, half_width_m=HW, workers=1,
        budget=budget)


def _reference(block: Block, roads: list[LineString]) -> Shares:
    """Both shares from scratch: c from the block's own `displacement` over the whole corridor;
    scored = sum c w / sum w with research/roadless common.py's AreaPopulation weights (footprint
    area over its mean); destroyed = the area of the buildings with c > 0 over the total."""
    c = block.buildings.displacement(road_corridor(_planarize(roads, block.crs, 2 * HW)))
    area = np.asarray(shapely.area(block.buildings.outlines.to_numpy()), dtype=np.float64)
    w = area / area.mean()
    return Shares(scored_area=float((c * w).sum() / w.sum()),
                  destroyed_area=float((area * (c > 0)).sum() / area.sum()))


def _meter_after(block: Block, roads: list[LineString]) -> Shares:
    meter = FootprintMeter.of(block, HW)
    for r in roads:
        meter.add(r)
    return meter.shares()


def test_meter_shares_match_the_whole_corridor() -> None:
    block = _dense_block()
    with _commits() as commits:
        _run(block, budget=NoBudget(), max_roads=4)
    assert len(commits) >= 2
    meter = FootprintMeter.of(block, HW)
    assert meter.shares() == Shares(0.0, 0.0)
    for k, road in enumerate(commits, start=1):
        meter.add(road)
        got, want = meter.shares(), _reference(block, commits[:k])
        assert got.scored_area == pytest.approx(want.scored_area, abs=1e-9)
        assert got.destroyed_area == pytest.approx(want.destroyed_area, abs=1e-9)
        assert 0.0 < got.scored_area < got.destroyed_area


def test_meter_counts_a_clipped_building_whole_only_when_destroyed() -> None:
    # Two equal discs (radius NN / 2 = 10), a road clipping the first one's east edge: the
    # destroyed share is that building's whole area, half the total; the scored share is the
    # clipped sliver's, half of c_0.
    pts = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5), Point(20.5, 0.5)], crs=UTM)
    block = _grid_block_with_points(pts)
    road = LineString([(10.0, -5.0), (10.0, 5.0)])
    meter = FootprintMeter.of(block, HW)
    meter.add(road)
    c = block.buildings.displacement(road.buffer(HW))
    assert 0.0 < c[0] < 0.1 and c[1] == 0.0
    assert meter.shares().destroyed_area == pytest.approx(0.5, abs=1e-12)
    assert meter.shares().scored_area == pytest.approx(c[0] / 2, abs=1e-12)


def test_scored_share_weights_by_footprint_area() -> None:
    # A small disc (r 1) wholly covered and a large one (r 3) untouched: by area the scored share
    # is 1 / (1 + 9) = 0.1; by count it would be 0.5.
    pts = gpd.GeoDataFrame(geometry=[Point(0.0, 0.0), Point(20.0, 0.0)], crs=UTM)
    buildings = Discs(pts, np.array([1.0, 3.0]))
    meter = FootprintMeter(buildings, 2.0)
    meter.add(LineString([(-3.0, 0.0), (3.0, 0.0)]))
    assert meter.shares().scored_area == pytest.approx(0.1, abs=1e-9)
    assert meter.shares().destroyed_area == pytest.approx(0.1, abs=1e-9)


def _scored(s: Shares) -> float:
    return s.scored_area


def _destroyed(s: Shares) -> float:
    return s.destroyed_area


@pytest.mark.parametrize(("make", "read"), [(ScoredArea, _scored), (DestroyedArea, _destroyed)])
def test_budget_stop_is_a_commit_prefix(make: type[ScoredArea] | type[DestroyedArea],
                                        read: Callable[[Shares], float]) -> None:
    block = _dense_block()
    with _commits() as commits:
        run = _run(block, budget=NoBudget(), max_roads=6)
    assert run.commits == len(commits) >= 4 and run.stop == "max_roads"
    d = read(_meter_after(block, commits[:3]))
    assert read(_meter_after(block, commits[:2])) < d
    stopped = _run(block, budget=make(share=d), max_roads=6)
    assert stopped.commits == 3 and stopped.stop == "budget"
    want = _planarize(commits[:3], block.crs, 2 * HW)
    assert [g.wkt for g in stopped.roads.geometry] == [g.wkt for g in want.geometry]
    assert stopped.shares == _meter_after(block, commits[:3])


def test_budget_stop_guard_sees_a_deaf_meter(monkeypatch: pytest.MonkeyPatch) -> None:
    block = _dense_block()
    with _commits() as commits:
        _run(block, budget=NoBudget(), max_roads=6)
    d = _meter_after(block, commits[:3]).scored_area
    monkeypatch.setattr(FootprintMeter, "add", lambda self, real: None)
    assert _run(block, budget=ScoredArea(share=d), max_roads=6).commits != 3


def test_scored_stop_guard_sees_the_destroyed_share(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fault a mix-up of the two definitions would be: ScoredArea reading the destroyed share,
    # which is larger, so the run stops early.
    block = _dense_block()
    with _commits() as commits:
        _run(block, budget=NoBudget(), max_roads=6)
    d = _meter_after(block, commits[:3]).scored_area
    monkeypatch.setattr(ScoredArea, "reached",
                        lambda self, shares: shares.destroyed_area >= self.share)
    assert _run(block, budget=ScoredArea(share=d), max_roads=6).commits < 3


def test_stop_reasons() -> None:
    block = _dense_block()
    capped = _run(block, budget=NoBudget(), max_roads=2)
    assert capped.stop == "max_roads" and capped.commits == 2
    small = _run(_grid_block(4), budget=NoBudget(), max_roads=50)
    assert small.stop == "no_gain" and 0 < small.commits < 50


def test_no_budget_is_never_reached() -> None:
    assert not NoBudget().reached(Shares(scored_area=1.0, destroyed_area=1.0))
    # Every building touched, and the run goes on to its road cap.
    run = _run(_dense_block(), budget=NoBudget(), max_roads=6)
    assert run.stop == "max_roads" and run.commits == 6


def test_a_run_with_no_gain_commits_nothing() -> None:
    # Every parcel of a 2 x 2 grid fronts a street.
    run = _run(_grid_block(2), budget=NoBudget(), max_roads=5)
    assert run.stop == "no_gain" and run.commits == 0 and run.shares == Shares(0.0, 0.0)
    assert len(run.roads) == 0 and "drain" in run.roads.columns


def test_no_buildings_never_stops_on_displacement() -> None:
    block = _grid_block(5)                         # tests.block_fixtures.no_buildings
    assert len(block.buildings) == 0
    assert FootprintMeter.of(block, HW).shares() == Shares(0.0, 0.0)
    for budget in (ScoredArea(share=0.01), DestroyedArea(share=0.01)):
        run = _run(block, budget=budget, max_roads=3)
        assert run.stop in ("no_gain", "max_roads")


@pytest.mark.parametrize("make", [ScoredArea, DestroyedArea])
def test_a_budget_share_out_of_range_is_refused(
        make: type[ScoredArea] | type[DestroyedArea]) -> None:
    for bad in (0.0, 1.5, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            make(share=bad)
    assert make(share=1.0).share == 1.0


def test_a_budget_share_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        ScoredArea(0.2)  # type: ignore[call-arg]


def test_the_budget_splits_the_identity() -> None:
    budgets: list[DisplacementBudget] = [NoBudget(), ScoredArea(share=0.1), ScoredArea(share=0.2),
                                         DestroyedArea(share=0.2)]
    methods = [replace(ARTERIAL, budget=b, workers=1) for b in budgets]
    assert len({m.identity for m in methods}) == len(budgets)
    block = _grid_block(2)
    assert len({m.propose(block).proposal_id for m in methods}) == len(budgets)


def test_proposal_params_record_the_stop_and_both_shares() -> None:
    block = _dense_block()
    m = replace(ARTERIAL, objective=Access(), cost=Displacement(), road_width_m=2 * HW,
                max_roads=3, workers=1)
    with _commits() as commits:
        p = m.propose(block)
    assert p.params["stop"] in ("max_roads", "no_gain", "budget")
    assert p.params["commits"] == len(commits) <= 3
    assert p.params["budget"] == repr(NoBudget())
    want = _reference(block, commits)
    assert p.params["scored_area"] == pytest.approx(want.scored_area, abs=1e-9)
    assert p.params["destroyed_area"] == pytest.approx(want.destroyed_area, abs=1e-9)
