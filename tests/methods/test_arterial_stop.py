"""The arterial's displacement budget (`reblock.methods.arterial.stop`): `Destroyed(share, tau)`'s
meter equals the research lineup's ramped destroyed-area D (reblock.destroyed over the whole
corridor, area weights) on the same roads, the greedy stops before the step that would start past
the budget (so a stopped run is a commit prefix of a longer one), `NoBudget` is never reached, and
every run reports why it stopped and its D at the stop. The fault (a deaf meter) is shown to break
the stop and the meter's check."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import replace

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from reblock.budget import road_corridor
from reblock.contracts import Block
from reblock.destroyed import IncrementalDestroyed, RampedDestroyed
from reblock.methods.arterial import (
    Access,
    Destroyed,
    Displacement,
    DisplacementBudget,
    NoBudget,
    SnapToBoundary,
    Uncapped,
    engines,
)
from reblock.methods.arterial.primitives import _planarize
from reblock.methods.arterial.stop import GreedyRun
from tests.methods.test_arterial import ARTERIAL, UTM, _grid_block, _grid_block_with_points

HW = 0.25                    # a narrow corridor, so each road touches only its own neighbours
TAU = 0.10


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


def _scored(block: Block, roads: list[LineString]) -> float:
    """The scoring function on `roads`: ramped destroyed-area D over their whole corridor, with
    research/roadless common.py's AreaPopulation weights (footprint area over its mean)."""
    if not roads:
        return 0.0
    area = np.asarray(shapely.area(block.buildings.outlines.to_numpy()), dtype=np.float64)
    corridor = road_corridor(_planarize(roads, block.crs, 2 * HW))
    return RampedDestroyed(tau=TAU).share(block.buildings, corridor, area / area.mean())


def _meter_after(block: Block, roads: list[LineString]) -> float:
    meter = Destroyed(share=1.0, tau=TAU).start(block, HW)
    for r in roads:
        meter.add(r)
    return meter.share()


def _meter_mismatches(block: Block, commits: list[LineString]) -> list[int]:
    """The commit counts k at which the meter's D differs from the scoring function's."""
    meter = Destroyed(share=1.0, tau=TAU).start(block, HW)
    out = []
    for k, road in enumerate(commits, start=1):
        meter.add(road)
        if abs(meter.share() - _scored(block, commits[:k])) > 1e-9:
            out.append(k)
    return out


def test_meter_equals_the_scoring_function() -> None:
    block = _dense_block()
    with _commits() as commits:
        _run(block, budget=NoBudget(), max_roads=4)
    assert len(commits) >= 2
    assert _meter_after(block, []) == 0.0
    assert _meter_mismatches(block, commits) == []
    assert 0.0 < _meter_after(block, commits) < 1.0


def test_meter_guard_sees_a_deaf_meter(monkeypatch: pytest.MonkeyPatch) -> None:
    block = _dense_block()
    with _commits() as commits:
        _run(block, budget=NoBudget(), max_roads=4)
    monkeypatch.setattr(IncrementalDestroyed, "add", lambda self, piece: None)
    assert _meter_mismatches(block, commits) != []


def test_meter_ramps_a_clipped_building() -> None:
    # Two equal discs (radius NN / 2 = 10, touching at x 10.5), a road whose corridor covers the
    # first one's WEST edge (x -9.65 -- -9.15 against an edge at -9.5: a corner clip, no piece cut
    # off; overlap c_0 < 0.1): ramped, that building counts c_0 / tau; D is half that.
    pts = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5), Point(20.5, 0.5)], crs=UTM)
    block = _grid_block_with_points(pts)
    road = LineString([(-9.4, -5.0), (-9.4, 5.0)])
    c = block.buildings.displacement(road.buffer(HW))
    assert 0.0 < c[0] < TAU and c[1] == 0.0
    assert _meter_after(block, [road]) == pytest.approx(c[0] / TAU / 2, abs=1e-9)


def test_budget_stop_is_a_commit_prefix() -> None:
    block = _dense_block()
    with _commits() as commits:
        run = _run(block, budget=NoBudget(), max_roads=6)
    assert run.commits == len(commits) >= 4 and run.stop == "max_roads"
    assert run.displaced is None
    d = _meter_after(block, commits[:3])
    assert _meter_after(block, commits[:2]) < d
    stopped = _run(block, budget=Destroyed(share=d, tau=TAU), max_roads=6)
    assert stopped.commits == 3 and stopped.stop == "budget"
    want = _planarize(commits[:3], block.crs, 2 * HW)
    assert [g.wkt for g in stopped.roads.geometry] == [g.wkt for g in want.geometry]
    assert stopped.displaced == d


def test_budget_stop_guard_sees_a_deaf_meter(monkeypatch: pytest.MonkeyPatch) -> None:
    block = _dense_block()
    with _commits() as commits:
        _run(block, budget=NoBudget(), max_roads=6)
    d = _meter_after(block, commits[:3])
    monkeypatch.setattr(IncrementalDestroyed, "add", lambda self, piece: None)
    assert _run(block, budget=Destroyed(share=d, tau=TAU), max_roads=6).commits != 3


def test_stop_reasons() -> None:
    block = _dense_block()
    capped = _run(block, budget=NoBudget(), max_roads=2)
    assert capped.stop == "max_roads" and capped.commits == 2
    small = _run(_grid_block(4), budget=NoBudget(), max_roads=50)
    assert small.stop == "no_gain" and 0 < small.commits < 50


def test_no_budget_is_never_reached() -> None:
    meter = NoBudget().start(_dense_block(), HW)
    meter.add(LineString([(0, 0), (8, 6)]))
    assert not meter.reached() and meter.share() is None
    run = _run(_dense_block(), budget=NoBudget(), max_roads=6)
    assert run.stop == "max_roads" and run.commits == 6


def test_a_run_with_no_gain_commits_nothing() -> None:
    # Every parcel of a 2 x 2 grid fronts a street.
    run = _run(_grid_block(2), budget=Destroyed(share=0.2, tau=TAU), max_roads=5)
    assert run.stop == "no_gain" and run.commits == 0 and run.displaced == 0.0
    assert len(run.roads) == 0 and "drain" in run.roads.columns


def test_no_buildings_never_stops_on_displacement() -> None:
    block = _grid_block(5)                         # tests.block_fixtures.no_buildings
    assert len(block.buildings) == 0
    run = _run(block, budget=Destroyed(share=0.01, tau=TAU), max_roads=3)
    assert run.stop in ("no_gain", "max_roads") and run.displaced == 0.0


@pytest.mark.parametrize("bad", [0.0, 1.5, -0.1, float("nan"), float("inf")])
def test_a_share_or_tau_out_of_range_is_refused(bad: float) -> None:
    with pytest.raises(ValueError):
        Destroyed(share=bad, tau=TAU)
    with pytest.raises(ValueError):
        Destroyed(share=0.2, tau=bad)


def test_share_and_tau_are_required_and_keyword_only() -> None:
    with pytest.raises(TypeError):
        Destroyed(0.2, 0.1)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Destroyed(share=0.2)  # type: ignore[call-arg]


def test_the_budget_splits_the_identity() -> None:
    budgets: list[DisplacementBudget] = [NoBudget(), Destroyed(share=0.1, tau=0.1),
                                         Destroyed(share=0.2, tau=0.1),
                                         Destroyed(share=0.2, tau=0.2)]
    methods = [replace(ARTERIAL, budget=b, workers=1) for b in budgets]
    assert len({m.identity for m in methods}) == len(budgets)
    block = _grid_block(2)
    assert len({m.propose(block).proposal_id for m in methods}) == len(budgets)


def test_proposal_params_record_the_stop_and_d() -> None:
    block = _dense_block()
    budget = Destroyed(share=1.0, tau=TAU)
    m = replace(ARTERIAL, objective=Access(), cost=Displacement(), road_width_m=2 * HW,
                max_roads=3, workers=1, budget=budget)
    with _commits() as commits:
        p = m.propose(block)
    assert p.params["stop"] in ("max_roads", "no_gain", "budget")
    assert p.params["commits"] == len(commits) <= 3
    assert p.params["budget"] == repr(budget)
    assert p.params["displaced"] == pytest.approx(_scored(block, commits), abs=1e-9)
